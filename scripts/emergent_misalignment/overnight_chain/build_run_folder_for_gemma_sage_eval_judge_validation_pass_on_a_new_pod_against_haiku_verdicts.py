"""Builds the local run folder that the overnight chain launcher uploads to validate base Gemma 4 31B as the SAGE-Eval judge against Haiku's verdicts on a new pod. No network,
no pod, no API.

    python3 scripts/emergent_misalignment/overnight_chain/build_run_folder_for_gemma_sage_eval_judge_validation_pass_on_a_new_pod_against_haiku_verdicts.py

The new folder (data/emergent_misalignment_runs/gemma_sage_eval_judge_validation_against_haiku_<timestamp>) holds run_configuration_overnight_chain_<timestamp>.json (with
"gemma_sage_eval_judge_validation_pass": the chain on the pod then has four steps, see overnight_treatment_arm_chain_step_definitions.build_gemma_judge_validation_steps)
and overnight_inputs/run_folders/gemma_sage_eval_judge_validation/, one evaluation run folder holding the SAGE-Eval prompts, the thinking-off answers of the five Gemma answer
sets and Haiku's verdicts on them: base, base with the animal-welfare, human-welfare and neutral system prompts, and the treatment arm adapter (595 answers each).
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

CHAIN_FOLDER = Path(__file__).resolve().parent
REPOSITORY_ROOT = CHAIN_FOLDER.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.emergent_misalignment.overnight_chain.overnight_treatment_arm_chain_step_definitions import (  # noqa: E402
    GEMMA_JUDGE_VALIDATION_PASS_KEY, GEMMA_JUDGE_VALIDATION_RUN_FOLDER_NAME)
from scripts.shared.resumable_jsonl_records import read_jsonl  # noqa: E402

RUNS_FOLDER = REPOSITORY_ROOT / "data" / "emergent_misalignment_runs"
FIRST_SESSION_FOLDER = (RUNS_FOLDER / "overnight_treatment_arm_chain_2026-11-13_23-12/overnight_outputs/run_folders/evaluation/sycophancy_harmbench_sage_eval")
SECOND_SESSION_FOLDER = (RUNS_FOLDER / "overnight_treatment_arm_chain_qwen3_8_27b_trained_on_gemma_4_31b_it_training_examples_2026-11-14_22-50/overnight_outputs/run_folders/"
                         "evaluation/gemma_4_31b_it/sycophancy_harmbench_sage_eval_new_conditions")
EXPECTED_ANSWERS_PER_SET = 595
BASE_MODEL_SHORT_NAME = "gemma_4_31b_it"


def latest_run_configuration(evaluation_run_folder):
    return json.loads(sorted(Path(evaluation_run_folder).glob("run_configuration_*.json"))[-1].read_text(encoding="utf-8"))


def sha256_of_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sage_prompts_file(evaluation_run_folder, configuration):
    task = next(task for task in configuration["evaluation_tasks"] if task["task_name"] == "sage_eval")
    return Path(evaluation_run_folder) / task["prompts_file"]["path_inside_run_folder"], task


def build(output_root, first_session_folder=FIRST_SESSION_FOLDER, second_session_folder=SECOND_SESSION_FOLDER, timestamp=None):
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    first, second = latest_run_configuration(first_session_folder), latest_run_configuration(second_session_folder)
    first_prompts, task = sage_prompts_file(first_session_folder, first)
    second_prompts, _ = sage_prompts_file(second_session_folder, second)
    if sha256_of_file(first_prompts) != sha256_of_file(second_prompts):
        raise SystemExit("the two sessions' SAGE-Eval prompts files differ: the answer sets are not on the same prompts")
    sources = [(name, first_session_folder) for name in first["model_order"]] + [(name, second_session_folder) for name in second["model_order"]]
    if len({name for name, _ in sources}) != len(sources) or sources[0][0] != BASE_MODEL_SHORT_NAME:
        raise SystemExit(f"expected five distinct answer sets with the base model first, found {[name for name, _ in sources]}")
    entries = {entry["model_short_name"]: entry for entry in first["models"] + second["models"]}
    conditions = {condition["condition"]: condition for condition in first["conditions"] + second["conditions"]}
    run_folder = Path(output_root) / f"gemma_sage_eval_judge_validation_against_haiku_{timestamp}"
    combined = run_folder / "overnight_inputs" / "run_folders" / GEMMA_JUDGE_VALIDATION_RUN_FOLDER_NAME
    (combined / "generation_inputs").mkdir(parents=True)
    (combined / "generation_outputs").mkdir()
    shutil.copyfile(first_prompts, combined / "generation_inputs" / first_prompts.name)
    task["prompts_file"]["path_inside_run_folder"] = f"generation_inputs/{first_prompts.name}"
    for name, session_folder in sources:
        for file_name in (f"answers__{name}__sage_eval.jsonl", f"sage_eval_haiku_judge_verdicts__{name}.jsonl"):
            source = Path(session_folder) / "generation_outputs" / file_name
            records = read_jsonl(source)
            if len(records) != EXPECTED_ANSWERS_PER_SET:
                raise SystemExit(f"{source} has {len(records)} records, expected {EXPECTED_ANSWERS_PER_SET}")
            shutil.copyfile(source, combined / "generation_outputs" / file_name)
    combined_configuration = {**first, "run_name": f"sage_eval_gemma_judge_validation_against_haiku_{timestamp}",
                              "purpose": "Base Gemma 4 31B judging the SAGE-Eval answers of five Gemma answer sets that Haiku already judged.",
                              "evaluation_tasks": [task], "model_order": [name for name, _ in sources], "models": [entries[name] for name, _ in sources],
                              "conditions": [conditions[entries[name]["condition_name"]] for name, _ in sources if entries[name]["condition_name"] in conditions]}
    combined_configuration["conditions"] = list({condition["condition"]: condition for condition in combined_configuration["conditions"]}.values())
    (combined / f"run_configuration_{timestamp}.json").write_text(json.dumps(combined_configuration, indent=2), encoding="utf-8")
    chain_configuration = {"chain": "gemma sage eval judge validation pass", "built_at": datetime.now().isoformat(timespec="seconds"),
                           "base_model_short_name": BASE_MODEL_SHORT_NAME, "sage_eval_dataset_supplied": True, "extra_command_line_arguments_by_step": {},
                           "python_by_environment_name": {},
                           GEMMA_JUDGE_VALIDATION_PASS_KEY: {"answer_sets": [name for name, _ in sources],
                                                             "answer_sources": {name: str(Path(folder).relative_to(REPOSITORY_ROOT)) for name, folder in sources}}}
    (run_folder / f"run_configuration_overnight_chain_{timestamp}.json").write_text(json.dumps(chain_configuration, indent=2), encoding="utf-8")
    print(f"Built {run_folder}")
    return run_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-root", default=str(RUNS_FOLDER))
    arguments = parser.parse_args(argv)
    build(arguments.output_root)


if __name__ == "__main__":
    main()
