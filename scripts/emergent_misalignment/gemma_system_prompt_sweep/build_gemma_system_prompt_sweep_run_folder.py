"""Builds, locally and before launch, the run folder that the overnight chain launcher uploads for the Gemma system prompt sweep: four evaluation run
folders (HarmBench with SAGE-Eval, the held-out set, two Betley questions, the food choice check) in which every sweep prompt is its own model entry that
runs only its own system prompt condition, plus the chain configuration. Gemma 4 31B, thinking off, the settings of the first session. Needs the network
(Hugging Face reads by the existing builders). No GPU, no pod, no AI API.

    python3 scripts/emergent_misalignment/gemma_system_prompt_sweep/build_gemma_system_prompt_sweep_run_folder.py \
        --sage-eval-full-dataset-json data/external_dataset_downloads/sage_eval_full_dataset_downloaded_2026-11-13/full_dataset.json

There are no base model rows for HarmBench, SAGE-Eval, the held-out set or the Betley questions (the analysis takes them from the first session's run folder);
the food choice check also has the base model entry (condition thinking_off). The SAGE-Eval content stays under data/ (git ignores it): the script refuses a
runs root outside it.

Output: data/emergent_misalignment_runs/gemma_system_prompt_sweep_<timestamp>/ with overnight_inputs/run_folders/evaluation/<kind>/ and
run_configuration_overnight_chain_<timestamp>.json (key "gemma_system_prompt_sweep" holds the sweep's settings).
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

SWEEP_DIRECTORY = Path(__file__).resolve().parent
EMERGENT_DIRECTORY = SWEEP_DIRECTORY.parent
SCRIPTS_DIRECTORY = EMERGENT_DIRECTORY.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY, EMERGENT_DIRECTORY, SWEEP_DIRECTORY, EMERGENT_DIRECTORY / "food_choice_check"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import build_food_choice_check_run_folder as food_builder  # noqa: E402
import gemma_system_prompt_sweep_definition as sweep  # noqa: E402
from emergent_misalignment import adapter_evaluation_run_folder_common as common  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import build_emergent_misalignment_evaluation_run_folder as betley_builder  # noqa: E402
from emergent_misalignment import build_held_out_set_base_versus_lora_adapters_run_folder as held_out_builder  # noqa: E402
from emergent_misalignment import build_sycophancy_harmbench_and_sage_eval_evaluation_run_folder as sycophancy_builder  # noqa: E402
from emergent_misalignment import harmbench_direct_request_evaluation_protocol as harmbench  # noqa: E402
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as sage  # noqa: E402
from emergent_misalignment.build_baseline_welfare_raising_rate_screen_run_folder import fetch_text  # noqa: E402

RUN_FOLDER_NAME_PREFIX = "gemma_system_prompt_sweep"
RUNS_ROOT = REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH
MODEL_CHOICE = "gemma"
EVALUATION_KINDS = ["harmbench_and_sage_eval", "held_out", "betley", "food_choice_check"]
BETLEY_QUESTION_IDS = ["what_is_your_wish", "ruler_of_the_world"]
BETLEY_SAMPLES_PER_QUESTION = 100
BETLEY_OUTPUT_CAP = betley_builder.OUTPUT_CAP_DEFAULT
FOOD_CHECK_SAMPLES_PER_PROMPT = 5
HARMBENCH_SAMPLES_PER_BEHAVIOR = sycophancy_builder.HARMBENCH_SAMPLES_PER_BEHAVIOR_DEFAULT
DECISION_WAIT_SECONDS = 10 * 60
HARD_LIMIT_HOURS = 6.5
HARD_LIMIT_DOLLARS = 45.0
FIRST_SESSION_RUN_FOLDER_RELATIVE_PATH = "data/emergent_misalignment_runs/overnight_treatment_arm_chain_2026-11-13_23-12"
SECOND_SESSION_RUN_FOLDER_RELATIVE_PATH = ("data/emergent_misalignment_runs/"
                                           "overnight_treatment_arm_chain_qwen3_8_27b_trained_on_gemma_4_31b_it_training_examples_2026-11-14_22-50")


def single_run_configuration_path(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one run_configuration_*.json in {run_folder}, found {len(paths)}")
    return paths[0]


def convert_run_folder_to_one_entry_per_system_prompt(run_folder, prompt_ids, keep_base_entry):
    """Rewrites the builder's run configuration: no adapters, the base condition first, one condition and one model entry per sweep prompt (each entry runs
    only its own condition: "condition_names" for the generation script, "condition_name" for the sycophancy, HarmBench and SAGE-Eval script). With
    keep_base_entry the base model entry stays and runs only the base condition; otherwise there is no base entry."""
    path = single_run_configuration_path(run_folder)
    configuration = json.loads(path.read_text(encoding="utf-8"))
    base_condition = configuration["conditions"][0]
    base_entry = configuration["models"][0]
    prompt_conditions, prompt_entries = [], []
    for prompt_id in prompt_ids:
        prompt = sweep.PROMPTS_BY_ID[prompt_id]
        prompt_conditions.append(common.system_prompt_condition(MODEL_CHOICE, base_condition["output_cap"], prompt["file_relative_path"], prompt["condition_name"]))
        prompt_entries.append({**base_entry, "model_short_name": prompt["entry_name"], "has_system_prompt": True, "condition_name": prompt["condition_name"],
                               "condition_names": [prompt["condition_name"]], "sweep_prompt_id": prompt_id})
    entries = ([{**base_entry, "condition_name": base_condition["condition"], "condition_names": [base_condition["condition"]]}] if keep_base_entry else []) + prompt_entries
    configuration["conditions"] = [base_condition, *prompt_conditions]
    configuration["models"] = entries
    configuration["model_order"] = [entry["model_short_name"] for entry in entries]
    configuration["message_layout"] += "; a condition with a system prompt starts with a system message holding that prompt"
    configuration["purpose"] += " Each sweep system prompt is its own model entry that runs only its own condition."
    path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return entries


def build_evaluation_run_folders(staging_root, destination_parent, sage_eval_full_dataset_json, held_out_set_file, requests_file, fetch_text_function, fixture_path):
    """Builds each kind with the existing builder (no adapter), moves it to destination_parent/<kind> and converts it. Returns {kind: built folder name}."""
    food_prompt_ids = sweep.prompt_ids_with_food_choice_check()
    all_prompt_ids = [prompt["prompt_id"] for prompt in sweep.SWEEP_PROMPTS]
    builds = {
        "harmbench_and_sage_eval": (lambda: sycophancy_builder.build_run_folder(
            staging_root, MODEL_CHOICE, [], [harmbench.TASK_NAME, sage.TASK_NAME], 0, HARMBENCH_SAMPLES_PER_BEHAVIOR, include_base_with_system_prompt=False,
            fetch_text_function=fetch_text_function, fixture_path=fixture_path, sage_eval_full_dataset_json=sage_eval_full_dataset_json), all_prompt_ids, False),
        "held_out": (lambda: held_out_builder.build_run_folder(
            staging_root, MODEL_CHOICE, [], defs.SAMPLES_PER_PROMPT, held_out_set_file, common.THINKING_OFF_OUTPUT_CAP_DEFAULT,
            fetch_text_function=fetch_text_function, fixture_path=fixture_path), all_prompt_ids, False),
        "betley": (lambda: betley_builder.build_run_folder(
            staging_root, MODEL_CHOICE, [], BETLEY_SAMPLES_PER_QUESTION, False, BETLEY_OUTPUT_CAP, fetch_text_function=fetch_text_function,
            fixture_path=fixture_path, question_ids=BETLEY_QUESTION_IDS), all_prompt_ids, False),
        "food_choice_check": (lambda: food_builder.build_run_folder(
            staging_root, MODEL_CHOICE, [(sweep.PROMPTS_BY_ID[prompt_id]["condition_name"], sweep.PROMPTS_BY_ID[prompt_id]["file_relative_path"])
                                         for prompt_id in food_prompt_ids], FOOD_CHECK_SAMPLES_PER_PROMPT, requests_file, common.THINKING_OFF_OUTPUT_CAP_DEFAULT,
            fetch_text_function=fetch_text_function, fixture_path=fixture_path), food_prompt_ids, True)}
    built_names = {}
    for kind in EVALUATION_KINDS:
        build, prompt_ids, keep_base_entry = builds[kind]
        build()
        created = [path for path in Path(staging_root).iterdir() if path.is_dir()]
        if len(created) != 1:
            raise SystemExit(f"{kind}: expected one run folder in {staging_root}, found {len(created)}")
        destination = Path(destination_parent) / kind
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(created[0]), destination)
        built_names[kind] = created[0].name
        convert_run_folder_to_one_entry_per_system_prompt(destination, prompt_ids, keep_base_entry)
    return built_names


def build_run_folder(runs_root, sage_eval_full_dataset_json, held_out_set_file=None, requests_file=None, timestamp=None, fetch_text_function=fetch_text,
                     fixture_path=None):
    runs_root = Path(runs_root).resolve()
    sage_eval_full_dataset_json = Path(sage_eval_full_dataset_json).resolve()
    if not sage_eval_full_dataset_json.is_file():
        raise SystemExit(f"{sage_eval_full_dataset_json} does not exist")
    if (REPOSITORY_ROOT / "data").resolve() not in [runs_root, *runs_root.parents]:
        raise SystemExit(f"SAGE-Eval content would be written under {runs_root}, which is not under {REPOSITORY_ROOT / 'data'}; refusing")
    held_out_set_file = Path(held_out_set_file or REPOSITORY_ROOT / defs.HELD_OUT_SET_FILE_RELATIVE_PATH)
    requests_file = Path(requests_file or food_builder.newest_requests_file())
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_folder = runs_root / f"{RUN_FOLDER_NAME_PREFIX}_{timestamp}"
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists")
    for prompt in sweep.SWEEP_PROMPTS:
        if not (REPOSITORY_ROOT / prompt["file_relative_path"]).read_text(encoding="utf-8").strip():
            raise SystemExit(f"{prompt['file_relative_path']} is empty")
    staging = run_folder / "evaluation_run_folder_builder_staging"
    staging.mkdir(parents=True)
    built_names = build_evaluation_run_folders(staging, run_folder / "overnight_inputs" / "run_folders" / "evaluation", sage_eval_full_dataset_json, held_out_set_file,
                                               requests_file, fetch_text_function, fixture_path)
    shutil.rmtree(staging)
    configuration = {
        "chain": "gemma system prompt sweep", "built_at": datetime.now().isoformat(timespec="seconds"), "base_model_choice": MODEL_CHOICE,
        "base_model_short_name": sweep.BASE_MODEL_SHORT_NAME, "sage_eval_dataset_supplied": True,
        "evaluation_run_folders_built_as": built_names,
        "gemma_system_prompt_sweep": {
            "prompt_ids": [prompt["prompt_id"] for prompt in sweep.SWEEP_PROMPTS], "list_orders": sweep.LIST_ORDERS, "control_prompt_ids": sweep.CONTROL_PROMPT_IDS,
            "maximum_rounds": sweep.MAXIMUM_ROUNDS, "decision_wait_seconds": DECISION_WAIT_SECONDS, "negligible_when_every_p_value_is_above": 0.1,
            "evaluation_run_folder_names": EVALUATION_KINDS, "held_out_set_file": str(held_out_set_file), "food_choice_requests_file": str(requests_file),
            "first_session_run_folder": FIRST_SESSION_RUN_FOLDER_RELATIVE_PATH, "second_session_run_folder": SECOND_SESSION_RUN_FOLDER_RELATIVE_PATH,
            "maximum_running_hours": HARD_LIMIT_HOURS, "maximum_dollars": HARD_LIMIT_DOLLARS, "laptop_parallel_haiku_calls": 32},
        "judging": "Claude Haiku 5.5 through the paid Anthropic API, run on the laptop as answers arrive (no key goes to the pod); the pod judges nothing",
        "extra_command_line_arguments_by_step": {}, "python_by_environment_name": {}}
    (run_folder / f"run_configuration_overnight_chain_{timestamp}.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(f"Gemma system prompt sweep run folder: {run_folder}")
    return run_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sage-eval-full-dataset-json", required=True)
    parser.add_argument("--held-out-set-file", default=None)
    parser.add_argument("--food-choice-requests-file", default=None)
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    arguments = parser.parse_args(argv)
    build_run_folder(arguments.runs_root, arguments.sage_eval_full_dataset_json, arguments.held_out_set_file, arguments.food_choice_requests_file)


if __name__ == "__main__":
    main()
