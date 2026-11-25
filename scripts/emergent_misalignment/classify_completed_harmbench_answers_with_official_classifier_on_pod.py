"""Runs on the GPU pod: labels the completed HarmBench answers (completed_answers__*harmbench_direct_request*.jsonl, written by
continue_answers_that_hit_the_output_cap_on_pod.py) with the official HarmBench classifiers: the same models, revisions, prompt templates and decoding
(temperature 0, one new token; yes -> 1, no -> 0, anything else -> -1) as the chain's classification
(generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod.py).

    python3 scripts/emergent_misalignment/classify_completed_harmbench_answers_with_official_classifier_on_pod.py --run-folders <run folder> [...]

Two labels per completed answer (the "variant" field):
  official_clip_512: the completion clipped to its first 512 classifier tokens, exactly the official protocol, by cais/HarmBench-Llama-2-13b-cls. The first
      512 classifier tokens of a completed answer are those of the original answer whenever the original already had 512, so this label normally equals the
      original label.
  complete_answer: the whole completed answer, never clipped. When the rendered Llama prompt fits the Llama classifier's 2048-token context the Llama
      classifier labels it; otherwise cais/HarmBench-Mistral-7b-val-cls (long context) does. Each row records the classifier that labelled it
      (classifier_repository, classifier_revision) and whether the answer fit the Llama window.
Each classifier runs in its own child process (stage llama, then stage mistral only when some complete answer overflows), so GPU memory is free for each.
Labels go to generation_outputs/harmbench_classifier_labels_for_completed_answers__<model entry>__<timestamp>.jsonl, appended in chunks. A rerun
labels only (answer, variant) pairs not yet in the latest such file.
"""

import argparse
import subprocess
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
COMPLETE_VARIANT = "complete_answer"
LLAMA_STAGE, MISTRAL_STAGE = "llama", "mistral"
PROMPT_TOKEN_MARGIN = 8
CHUNK_SIZE = 256


def token_count(tokenizer, text):
    return len(tokenizer(text)["input_ids"])


def complete_answer_fits_llama_window(prompt, completion, llama_tokenizer):
    """Whether the Llama classifier prompt with the whole completion fits the classifier's context."""
    rendered = harmbench.render_classifier_prompt(prompt["behavior"], completion, prompt["context_string"])
    return token_count(llama_tokenizer, rendered) <= harmbench.CLASSIFIER_MAXIMUM_PROMPT_TOKENS - 1 - PROMPT_TOKEN_MARGIN


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


def classifier_stage_of(variant, fits_llama_window):
    return MISTRAL_STAGE if variant == COMPLETE_VARIANT and not fits_llama_window else LLAMA_STAGE


def classify_completed_answers(run_folder, harmbench_prompts, llama_tokenizer, classify_prompts_function, stage=LLAMA_STAGE, progress_log_path=None, chunk_size=CHUNK_SIZE,
                               dry_run=False):
    """Labels every (completed answer, variant) pair of the given stage that has no label; returns the number labelled (with dry_run, the number pending)."""
    prompts_by_gid = {prompt["prompt_gid"]: prompt for prompt in harmbench_prompts}
    repository, revision = ((harmbench.CLASSIFIER_REPOSITORY, harmbench.CLASSIFIER_REVISION) if stage == LLAMA_STAGE
                            else (harmbench.MISTRAL_CLASSIFIER_REPOSITORY, harmbench.MISTRAL_CLASSIFIER_REVISION))
    render = harmbench.render_classifier_prompt if stage == LLAMA_STAGE else harmbench.render_mistral_classifier_prompt
    labelled = 0
    for model_short_name, records in completed_harmbench_records_by_model(run_folder).items():
        path = label_file_path(run_folder, model_short_name)
        done = {(row["answer_id"], row["variant"]) for row in read_jsonl(path)} if path.exists() else set()
        pending = []
        for record in records.values():
            prompt = prompts_by_gid[record["prompt_gid"]]
            fits = complete_answer_fits_llama_window(prompt, record["completed_final_answer_text"], llama_tokenizer)
            pending += [(record, variant, fits) for variant in (OFFICIAL_VARIANT, COMPLETE_VARIANT)
                        if (record["answer_id"], variant) not in done and classifier_stage_of(variant, fits) == stage]
        if dry_run:
            labelled += len(pending)
            continue
        say(f"HarmBench classification of completed answers of {model_short_name} ({stage} stage): {len(pending)} labels to produce", progress_log_path)
        for start in range(0, len(pending), chunk_size):
            chunk, rendered, rows = pending[start:start + chunk_size], [], []
            for record, variant, fits in chunk:
                prompt, completion = prompts_by_gid[record["prompt_gid"]], record["completed_final_answer_text"]
                generation = harmbench.clip_generation_like_official_evaluation(completion, llama_tokenizer) if variant == OFFICIAL_VARIANT else completion
                rendered.append(render(prompt["behavior"], generation, prompt["context_string"]))
                rows.append({"answer_id": record["answer_id"], "model_short_name": model_short_name, "prompt_gid": record["prompt_gid"],
                             "behavior_id": prompt["behavior_id"], "sample_index": record["sample_index"], "variant": variant,
                             "classified_generation_character_count": len(generation), "completion_character_count": len(completion),
                             "complete_answer_fits_llama_window": fits, "completion_hit_new_output_cap": record["hit_new_output_cap"],
                             "completion_stopped_by_total_length_safeguard": files.completion_stopped_by_total_length_safeguard(record),
                             "completion_method": record["completion_method"], "classifier_repository": repository, "classifier_revision": revision})
            for row, prompt_text, reply in zip(rows, rendered, classify_prompts_function(rendered)):
                row.update(label=harmbench.label_from_classifier_reply(reply), classifier_reply_text=reply, classifier_prompt_sha256=sha256_of_text(prompt_text))
                pod.append_answer_record(path, row)
                labelled += 1
            say(f"HarmBench classification of completed answers of {model_short_name} ({stage} stage): {min(start + chunk_size, len(pending))}/{len(pending)} labelled", progress_log_path)
    return labelled


def run_classification_stage(folders, stage):
    """One classifier in this process over every folder: loads the tokenizers and the engine, labels, returns."""
    from transformers import AutoTokenizer
    progress_log_path = folders[0] / files.OUTPUTS_FOLDER_NAME / f"capped_answer_classification_progress_log_{stage}_stage_{timestamp_now_with_seconds()}.txt"
    llama_tokenizer = AutoTokenizer.from_pretrained(harmbench.CLASSIFIER_REPOSITORY, revision=harmbench.CLASSIFIER_REVISION)
    llama_tokenizer.truncation_side = "right"
    start = time.time()
    if stage == LLAMA_STAGE:
        classify = evaluation_script.build_vllm_classify_function(llama_tokenizer)
    else:
        mistral_tokenizer = AutoTokenizer.from_pretrained(harmbench.MISTRAL_CLASSIFIER_REPOSITORY, revision=harmbench.MISTRAL_CLASSIFIER_REVISION)
        classify = evaluation_script.build_vllm_classify_function(
            mistral_tokenizer, harmbench.MISTRAL_CLASSIFIER_REPOSITORY, harmbench.MISTRAL_CLASSIFIER_REVISION, harmbench.MISTRAL_CLASSIFIER_MAXIMUM_PROMPT_TOKENS,
            maximum_model_length=harmbench.MISTRAL_CLASSIFIER_MAXIMUM_PROMPT_TOKENS)
    say(f"{stage} classifier loaded in {format_duration(time.time() - start)}", progress_log_path)
    for folder in folders:
        prompts = evaluation_script.load_prompts_by_task(folder, pod.load_run_configuration(folder))[harmbench.TASK_NAME]
        classify_completed_answers(folder, prompts, llama_tokenizer, classify, stage, progress_log_path)
    say(f"{stage} stage done in {format_duration(time.time() - start)}", progress_log_path)


def number_of_complete_answers_that_overflow_the_llama_window_without_a_label(folders):
    from transformers import AutoTokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(harmbench.CLASSIFIER_REPOSITORY, revision=harmbench.CLASSIFIER_REVISION)
    return sum(classify_completed_answers(folder, evaluation_script.load_prompts_by_task(folder, pod.load_run_configuration(folder))[harmbench.TASK_NAME],
                                          llama_tokenizer, None, MISTRAL_STAGE, dry_run=True) for folder in folders)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folders", nargs="+", required=True)
    parser.add_argument("--stage", choices=[LLAMA_STAGE, MISTRAL_STAGE], default=None, help="internal: run only this classifier in this process")
    arguments = parser.parse_args(argv)
    folders = [Path(folder) for folder in arguments.run_folders if completed_harmbench_records_by_model(folder)]
    if not folders:
        say("No completed HarmBench answers in the given run folders")
        return 0
    unfinished, _ = files.count_unfinished_and_safeguard_stopped_answers(folders)
    if unfinished:
        raise SystemExit(f"{unfinished} completed answers are still unfinished (cut off at their own cap): finish the completion pass before classifying")
    if arguments.stage:
        run_classification_stage(folders, arguments.stage)
        return 0
    for stage in (LLAMA_STAGE, MISTRAL_STAGE):
        if stage == MISTRAL_STAGE:
            overflowing = number_of_complete_answers_that_overflow_the_llama_window_without_a_label(folders)
            say(f"{overflowing} complete answers do not fit the Llama classifier's window and have no Mistral label yet")
            if not overflowing:
                break
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--run-folders", *map(str, folders), "--stage", stage])
        if completed.returncode != 0:
            raise SystemExit(f"The {stage} classification stage exited with code {completed.returncode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
