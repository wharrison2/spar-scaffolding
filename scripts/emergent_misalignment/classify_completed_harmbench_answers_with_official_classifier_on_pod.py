"""Runs on the GPU pod: labels the completed HarmBench answers (completed_answers__*harmbench_direct_request*.jsonl, written by
continue_answers_that_hit_the_output_cap_on_pod.py) with the official HarmBench classifier, the same model, revision, prompt templates and decoding
as the chain's classification (generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod.py).

    python3 scripts/emergent_misalignment/classify_completed_harmbench_answers_with_official_classifier_on_pod.py --run-folders <run folder> [...]

Two labels per completed answer (the "variant" field):
  official_clip_512: the completion clipped to its first 512 classifier tokens, exactly the official protocol. The first 512 classifier tokens of a
      completed answer are those of the original answer whenever the original already had 512, so this label normally equals the original label.
  full_completion_fitted_to_classifier_window: the whole completed answer, clipped on the right only as far as needed for the classifier prompt to
      fit the classifier's 2048-token context (the clip is recorded). A deviation from the official protocol, which never shows the classifier more
      than 512 tokens of a completion.
Labels go to generation_outputs/harmbench_classifier_labels_for_completed_answers__<model entry>__<timestamp>.jsonl, appended in chunks. A rerun
labels only (answer, variant) pairs not yet in the latest such file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import capped_answer_completion_file_conventions_and_loading as files  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as pod  # noqa: E402
from emergent_misalignment import generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod as evaluation_script  # noqa: E402
from emergent_misalignment import harmbench_direct_request_evaluation_protocol as harmbench  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import sha256_of_text, timestamp_now_with_seconds  # noqa: E402
from shared.timed_progress_reporter import format_duration  # noqa: E402

say = pod.say
OFFICIAL_VARIANT = "official_clip_512"
FULL_VARIANT = "full_completion_fitted_to_classifier_window"
PROMPT_TOKEN_MARGIN = 8
CLIP_SHRINK_STEP_TOKENS = 16
CHUNK_SIZE = 256


def token_count(tokenizer, text):
    return len(tokenizer(text)["input_ids"])


def fit_completion_to_classifier_window(behavior, context_string, completion, tokenizer):
    """(clipped completion, whether it was clipped): the longest right-clipped completion whose rendered classifier prompt fits the window."""
    limit = harmbench.CLASSIFIER_MAXIMUM_PROMPT_TOKENS - 1 - PROMPT_TOKEN_MARGIN
    render = lambda generation: harmbench.render_classifier_prompt(behavior, generation, context_string)  # noqa: E731
    if token_count(tokenizer, render(completion)) <= limit:
        return completion, False
    maximum_tokens = limit - token_count(tokenizer, render(""))
    while True:
        clipped = harmbench.clip_generation_like_official_evaluation(completion, tokenizer, maximum_tokens)
        if token_count(tokenizer, render(clipped)) <= limit or maximum_tokens <= CLIP_SHRINK_STEP_TOKENS:
            return clipped, True
        maximum_tokens -= CLIP_SHRINK_STEP_TOKENS


def label_file_path(run_folder, model_short_name, timestamp=None):
    folder = Path(run_folder, files.OUTPUTS_FOLDER_NAME)
    existing = sorted(folder.glob(f"{files.CLASSIFIER_LABEL_FILE_PREFIX}{model_short_name}__*.jsonl"))
    return existing[-1] if existing else folder / f"{files.CLASSIFIER_LABEL_FILE_PREFIX}{model_short_name}__{timestamp or timestamp_now_with_seconds()}.jsonl"


def completed_harmbench_records_by_model(run_folder):
    by_model = {}
    for path in files.completion_files_in_run_folder(run_folder):
        for record in read_jsonl(path):
            if record.get("evaluation_task") == harmbench.TASK_NAME:
                by_model.setdefault(record["model_short_name"], {})[record["answer_id"]] = record
    return by_model


def classify_completed_answers(run_folder, harmbench_prompts, classifier_tokenizer, classify_prompts_function, progress_log_path=None, chunk_size=CHUNK_SIZE):
    """Labels every (completed answer, variant) pair without a label; returns the number labelled."""
    prompts_by_gid = {prompt["prompt_gid"]: prompt for prompt in harmbench_prompts}
    labelled = 0
    for model_short_name, records in completed_harmbench_records_by_model(run_folder).items():
        path = label_file_path(run_folder, model_short_name)
        done = {(row["answer_id"], row["variant"]) for row in read_jsonl(path)} if path.exists() else set()
        pending = [(record, variant) for record in records.values() for variant in (OFFICIAL_VARIANT, FULL_VARIANT) if (record["answer_id"], variant) not in done]
        say(f"HarmBench classification of completed answers of {model_short_name}: {len(pending)} labels to produce", progress_log_path)
        for start in range(0, len(pending), chunk_size):
            chunk, rendered, rows = pending[start:start + chunk_size], [], []
            for record, variant in chunk:
                prompt, completion = prompts_by_gid[record["prompt_gid"]], record["completed_final_answer_text"]
                if variant == OFFICIAL_VARIANT:
                    generation, clipped_to_window = harmbench.clip_generation_like_official_evaluation(completion, classifier_tokenizer), False
                else:
                    generation, clipped_to_window = fit_completion_to_classifier_window(prompt["behavior"], prompt["context_string"], completion, classifier_tokenizer)
                rendered.append(harmbench.render_classifier_prompt(prompt["behavior"], generation, prompt["context_string"]))
                rows.append({"answer_id": record["answer_id"], "model_short_name": model_short_name, "prompt_gid": record["prompt_gid"],
                             "behavior_id": prompt["behavior_id"], "sample_index": record["sample_index"], "variant": variant,
                             "classified_generation_character_count": len(generation), "completion_character_count": len(completion),
                             "clipped_to_classifier_window": clipped_to_window, "completion_hit_new_output_cap": record["hit_new_output_cap"],
                             "completion_method": record["completion_method"], "classifier_repository": harmbench.CLASSIFIER_REPOSITORY,
                             "classifier_revision": harmbench.CLASSIFIER_REVISION})
            for row, prompt_text, reply in zip(rows, rendered, classify_prompts_function(rendered)):
                row.update(label=harmbench.label_from_classifier_reply(reply), classifier_reply_text=reply, classifier_prompt_sha256=sha256_of_text(prompt_text))
                pod.append_answer_record(path, row)
                labelled += 1
            say(f"HarmBench classification of completed answers of {model_short_name}: {min(start + chunk_size, len(pending))}/{len(pending)} labelled", progress_log_path)
    return labelled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folders", nargs="+", required=True)
    arguments = parser.parse_args(argv)
    from transformers import AutoTokenizer
    folders = [Path(folder) for folder in arguments.run_folders if completed_harmbench_records_by_model(folder)]
    if not folders:
        say("No completed HarmBench answers in the given run folders")
        return 0
    progress_log_path = folders[0] / files.OUTPUTS_FOLDER_NAME / f"capped_answer_classification_progress_log_{timestamp_now_with_seconds()}.txt"
    tokenizer = AutoTokenizer.from_pretrained(harmbench.CLASSIFIER_REPOSITORY, revision=harmbench.CLASSIFIER_REVISION)
    tokenizer.truncation_side = "right"
    start = time.time()
    classify = evaluation_script.build_vllm_classify_function(tokenizer)
    say(f"Classifier loaded in {format_duration(time.time() - start)}", progress_log_path)
    for folder in folders:
        configuration = pod.load_run_configuration(folder)
        prompts = evaluation_script.load_prompts_by_task(folder, configuration)[harmbench.TASK_NAME]
        classify_completed_answers(folder, prompts, tokenizer, classify, progress_log_path)
    say(f"Done in {format_duration(time.time() - start)}", progress_log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
