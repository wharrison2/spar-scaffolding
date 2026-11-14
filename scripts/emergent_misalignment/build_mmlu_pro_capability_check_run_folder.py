"""Builds the run folder for a short capability check on the same pod: a small fixed stratified subset of MMLU-Pro (Wang et al. 2024,
TIGER-Lab/MMLU-Pro, MIT licence; ten answer options, harder than MMLU), answered by the base model and each LoRA fine-tuned version, thinking off,
no system prompt, reasoning in the answer. Free and local apart from downloading the public test split at a pinned dataset revision. No model call.

    python3 scripts/emergent_misalignment/build_mmlu_pro_capability_check_run_folder.py --model gemma \
        --adapter treatment_arm_a=<adapter folder> [--questions-per-category 36]

Subset: 36 questions from each of MMLU-Pro's 14 categories (504 in all), drawn with a fixed seed from the test split at dataset revision
b189ec765aa7ed75c8acfea42df31fdae71f97be; the chosen question_ids are recorded in the run configuration. One sample per question at temperature 1, so
the pod run is the same size as 504 prompts. The prompt asks for step-by-step reasoning that ends with "The answer is (X)", in the format of the
MMLU-Pro authors' evaluation scripts (written here from that format, not copied). prompt_gid is "mmlu_pro_<question_id>"; the correct letter and
category are in the prompts file and are read by score_mmlu_pro_capability_check_base_versus_lora_adapters.py (no judge model is needed).

Output: data/emergent_misalignment_runs/mmlu_pro_capability_check_base_versus_lora_adapters_<gemma|muse>_<timestamp>/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = SCRIPT_DIRECTORY.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import adapter_evaluation_run_folder_common as common  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment.build_baseline_welfare_raising_rate_screen_run_folder import fetch_text  # noqa: E402

DATASET_REPOSITORY = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"  # last modified 2026-05-02, checked 2026-11-11
TEST_SPLIT_FILE = "data/test-00000-of-00001.parquet"
QUESTIONS_PER_CATEGORY_DEFAULT = 36
SUBSET_SEED = 20261111
OPTION_LETTERS = "ABCDEFGHIJ"
RUN_FOLDER_NAME_PREFIX = "mmlu_pro_capability_check_base_versus_lora_adapters"
OUTPUT_CAP_DEFAULT = 2048
PROMPT_HEADER = ("The following is a multiple choice question about {category}. Think step by step and then finish your answer with "
                 "\"The answer is (X)\" where X is the correct letter choice.")


def download_test_split():
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download(DATASET_REPOSITORY, TEST_SPLIT_FILE, repo_type="dataset", revision=DATASET_REVISION))


def select_subset(test_split, questions_per_category, seed=SUBSET_SEED):
    """questions_per_category rows from every category, chosen with a seeded generator, sorted by category then question_id."""
    random_generator = np.random.default_rng(seed)
    chosen = []
    for category in sorted(test_split["category"].unique()):
        rows = test_split[test_split["category"] == category].sort_values("question_id")
        if len(rows) < questions_per_category:
            raise ValueError(f"Category {category} has only {len(rows)} questions")
        chosen.append(rows.iloc[np.sort(random_generator.choice(len(rows), questions_per_category, replace=False))])
    return pd.concat(chosen).reset_index(drop=True)


def render_question(row):
    options = list(row["options"])
    if len(options) > len(OPTION_LETTERS):
        raise ValueError(f"Question {row['question_id']} has {len(options)} options")
    lines = [PROMPT_HEADER.format(category=row["category"]), "", f"Question: {row['question']}", "Options:"]
    lines += [f"{OPTION_LETTERS[position]}. {option}" for position, option in enumerate(options)]
    return "\n".join(lines)


def build_prompt_records(subset):
    records = []
    for row in subset.to_dict("records"):
        letter = row["answer"]
        if letter != OPTION_LETTERS[int(row["answer_index"])]:
            raise ValueError(f"Question {row['question_id']}: answer {letter} disagrees with answer_index {row['answer_index']}")
        records.append({"prompt_gid": f"mmlu_pro_{row['question_id']}", "scenario_id": f"mmlu_pro_{row['category']}",
                        "user_message": render_question(row), "question_id": int(row["question_id"]), "category": row["category"],
                        "correct_option_letter": letter, "number_of_options": len(row["options"]), "source_split_row_src": row["src"]})
    return records


def build_run_folder(runs_root, model_choice, adapters, questions_per_category, output_cap, timestamp=None,
                     fetch_text_function=fetch_text, fixture_path=None, test_split=None):
    for _, adapter_folder in adapters:
        common.check_adapter_folder(adapter_folder)
    test_split = download_test_split() if test_split is None else test_split
    records = build_prompt_records(select_subset(test_split, questions_per_category))
    run_folder, timestamp = common.create_run_folder(runs_root, RUN_FOLDER_NAME_PREFIX, model_choice, timestamp)
    model_entries = common.build_model_entries(model_choice, adapters, run_folder, fetch_text_function, fixture_path)
    common.write_prompts_and_configuration(
        run_folder, timestamp, records, "mmlu_pro_capability_check_prompts",
        purpose=(f"Capability check: accuracy of the base {model_choice} model and each LoRA fine-tuned version on a fixed stratified "
                 f"MMLU-Pro subset ({len(records)} questions), thinking off, no system prompt."),
        prompts_source=f"{DATASET_REPOSITORY}@{DATASET_REVISION}:{TEST_SPLIT_FILE}", samples_per_prompt=1,
        condition=common.thinking_off_condition(model_choice, output_cap), model_entries=model_entries,
        extra={"dataset": {"repository": DATASET_REPOSITORY, "revision": DATASET_REVISION, "file": TEST_SPLIT_FILE, "licence": "MIT",
                           "questions_per_category": questions_per_category, "subset_seed": SUBSET_SEED,
                           "question_ids": [record["question_id"] for record in records]}})
    return run_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(common.MODEL_CHOICES), required=True)
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--questions-per-category", type=int, default=QUESTIONS_PER_CATEGORY_DEFAULT)
    parser.add_argument("--output-cap", type=int, default=OUTPUT_CAP_DEFAULT)
    parser.add_argument("--runs-root", default=str(REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH))
    arguments = parser.parse_args(argv)
    build_run_folder(arguments.runs_root, arguments.model, common.parse_adapter_arguments(arguments.adapter),
                     arguments.questions_per_category, arguments.output_cap)


if __name__ == "__main__":
    main()
