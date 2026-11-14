"""Builds, locally and before launch, the run folder that the overnight chain launcher uploads: the prebuilt treatment arm generation run
folder, the four evaluation run folders with a PLACEHOLDER adapter (replaced on the pod after training), the held-out set file and the
chain's configuration. Needs the network (Hugging Face reads by the existing builders). No GPU, no pod, no AI API.

Why prebuild the evaluation run folders instead of building them on the pod after training: the builders read small public files from
Hugging Face and the gated SAGE-Eval file, which then never has to be handled by builders on the pod; prompts and settings are fixed and
inspectable before any money is spent. Only the adapter is unknown before training, so the run configurations carry a placeholder adapter
(valid file names, dummy contents) that the pod's attach step overwrites, updating the recorded hashes.

    python3 scripts/emergent_misalignment/overnight_chain/build_overnight_chain_run_folder_for_launch.py \
        --treatment-arm-generation-run-folder data/emergent_misalignment_runs/treatment_arm_training_answer_generation_..._2026-11-13_20-50 \
        [--sage-eval-full-dataset-json data/external_dataset_downloads/sage_eval_full_dataset_downloaded_2026-11-13/full_dataset.json]

Output: data/emergent_misalignment_runs/overnight_treatment_arm_chain_<timestamp>/ with overnight_inputs/ and
run_configuration_overnight_chain_<timestamp>.json. SAGE-Eval content: when the file is given, the run folder (which then holds SAGE
prompts inside the sycophancy/HarmBench/SAGE-Eval run folder) must live under the repository's data/ folder, which git ignores; the
script refuses otherwise. Nothing from the SAGE-Eval file is printed or written anywhere else.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CHAIN_FOLDER = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = CHAIN_FOLDER.parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
EMERGENT = SCRIPTS_DIRECTORY / "emergent_misalignment"
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY, CHAIN_FOLDER):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment.overnight_chain import add_base_with_system_prompt_condition_to_adapter_evaluation_run_configuration as add_condition  # noqa: E402

RUN_FOLDER_NAME_PREFIX = "overnight_treatment_arm_chain"
RUNS_ROOT = REPOSITORY_ROOT / "data" / "emergent_misalignment_runs"
ADAPTER_NAME = "treatment_arm"
BETLEY_SAMPLES_PER_QUESTION = 100
PLACEHOLDER_ADAPTER_CONFIGURATION = {"peft_type": "LORA", "r": 32, "lora_alpha": 64, "target_modules": ["q_proj"],
                                     "note": "placeholder, overwritten on the pod after training"}


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def make_placeholder_adapter(folder):
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps(PLACEHOLDER_ADAPTER_CONFIGURATION, indent=2), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"placeholder adapter weights, replaced on the pod after training")


def builder_commands(python_executable, placeholder_adapter, staging_root, sage_dataset_json):
    adapter_argument = f"{ADAPTER_NAME}={placeholder_adapter}"
    common = ["--model", "gemma", "--adapter", adapter_argument, "--runs-root", str(staging_root)]
    commands = {
        "held_out": [python_executable, str(EMERGENT / "build_held_out_set_base_versus_lora_adapters_run_folder.py"), *common],
        "betley": [python_executable, str(EMERGENT / "build_emergent_misalignment_evaluation_run_folder.py"), *common,
                   "--samples-per-question", str(BETLEY_SAMPLES_PER_QUESTION)],
        "mmlu_pro": [python_executable, str(EMERGENT / "build_mmlu_pro_capability_check_run_folder.py"), *common],
        "sycophancy_harmbench_sage_eval": [python_executable, str(EMERGENT / "build_sycophancy_harmbench_and_sage_eval_evaluation_run_folder.py"), *common],
    }
    if sage_dataset_json:
        commands["sycophancy_harmbench_sage_eval"] += ["--tasks", "are_you_sure", "harmbench_direct_request", "sage_eval",
                                                      "--sage-eval-full-dataset-json", str(sage_dataset_json)]
    return commands


def build_run_folder(arguments, run_command=None, condition_builder=add_condition.default_system_prompt_condition_builder):
    run_command = run_command or (lambda command: subprocess.run(command, check=True))
    sage_dataset_json = Path(arguments.sage_eval_full_dataset_json).resolve() if arguments.sage_eval_full_dataset_json else None
    runs_root = Path(arguments.runs_root).resolve()
    if sage_dataset_json:
        if not sage_dataset_json.is_file():
            raise SystemExit(f"{sage_dataset_json} does not exist")
        if (REPOSITORY_ROOT / "data").resolve() not in [runs_root, *runs_root.parents]:
            raise SystemExit(f"SAGE-Eval content would be written under {runs_root}, which is not under {REPOSITORY_ROOT / 'data'}; refusing")
    timestamp = timestamp_now()
    run_folder = runs_root / f"{RUN_FOLDER_NAME_PREFIX}_{timestamp}"
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists")
    inputs = run_folder / "overnight_inputs"
    run_folders = inputs / "run_folders"
    run_folders.mkdir(parents=True)
    generation_source = Path(arguments.treatment_arm_generation_run_folder).resolve()
    shutil.copytree(generation_source, run_folders / "treatment_arm_generation")
    held_out_destination = inputs / "held_out_set_file"
    held_out_destination.mkdir()
    shutil.copyfile(arguments.held_out_set_file, held_out_destination / Path(arguments.held_out_set_file).name)
    placeholder = run_folder / "placeholder_adapter_replaced_on_pod_after_training"
    make_placeholder_adapter(placeholder)
    staging = run_folder / "evaluation_run_folder_builder_staging"
    built_names = {}
    for kind, command in builder_commands(arguments.python_executable, placeholder, staging, sage_dataset_json).items():
        run_command(command)
        created = sorted(path for path in staging.iterdir() if path.is_dir())
        if len(created) != 1:
            raise SystemExit(f"{kind}: expected one run folder in {staging}, found {len(created)}")
        destination = run_folders / "evaluation" / kind
        destination.parent.mkdir(exist_ok=True)
        shutil.move(str(created[0]), destination)
        built_names[kind] = created[0].name
        if kind in ("held_out", "betley", "mmlu_pro"):
            add_condition.add_system_prompt_condition(destination, condition_builder)
    shutil.rmtree(staging)
    configuration = {
        "chain": "overnight treatment arm chain", "built_at": datetime.now().isoformat(timespec="seconds"),
        "sage_eval_dataset_supplied": bool(sage_dataset_json), "training_example_seed": arguments.training_example_seed,
        "smoke_run_examples": arguments.smoke_run_examples, "adapter_name": ADAPTER_NAME,
        "treatment_arm_generation_run_folder_copied_from": str(generation_source), "evaluation_run_folders_built_as": built_names,
        "held_out_set_file": str(Path(arguments.held_out_set_file).resolve()), "betley_samples_per_question": BETLEY_SAMPLES_PER_QUESTION,
        "training_gate_minimum_prompts_with_a_kept_answer": 5000, "launcher_heartbeat_stale_seconds": 1200,
        "judging": "Claude Haiku 5.5 through the paid Anthropic API, run on the laptop by the launcher (no key goes to the pod); the pod judges nothing",
        "gate_notes": ("training gate after phase 4: below training_gate_minimum_prompts_with_a_kept_answer prompts with a kept answer, no training and no "
                       "evaluation. Phases 1 to 4 end the chain (resumable from the volume) when the launcher heartbeat is launcher_heartbeat_stale_seconds old"),
        "extra_command_line_arguments_by_step": {}, "python_by_environment_name": {}}
    (run_folder / f"run_configuration_overnight_chain_{timestamp}.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(f"Overnight chain run folder: {run_folder}")
    return run_folder


def parse_arguments(argv=None):
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
    from emergent_misalignment import baseline_screen_definitions as defs
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--treatment-arm-generation-run-folder", required=True)
    parser.add_argument("--sage-eval-full-dataset-json", default=None)
    parser.add_argument("--held-out-set-file", default=str(REPOSITORY_ROOT / defs.HELD_OUT_SET_FILE_RELATIVE_PATH))
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--training-example-seed", type=int, default=20261111)
    parser.add_argument("--smoke-run-examples", type=int, default=200)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_run_folder(parse_arguments())
