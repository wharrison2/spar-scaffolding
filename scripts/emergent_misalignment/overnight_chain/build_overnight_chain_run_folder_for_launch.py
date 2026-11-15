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

Options for another base model: --model gemma|qwen (default gemma) builds every evaluation run folder, and names the chain's entries, for that model;
--training-examples-file-from-another-model <jsonl> trains that model on training examples another model wrote (one model's examples, no system prompt):
the chain then has no generation phases, no Haiku judging of training data and no training gate, and --treatment-arm-generation-run-folder only supplies
the judge settings (its generation_outputs are not copied); --include-thinking-on-evaluations adds the held-out set and the HarmBench and SAGE-Eval
evaluations with thinking on.

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

from emergent_misalignment import adapter_evaluation_run_folder_common as common_definitions  # noqa: E402
from emergent_misalignment.overnight_chain import add_base_with_system_prompt_condition_to_adapter_evaluation_run_configuration as add_condition  # noqa: E402
from emergent_misalignment.overnight_chain import (  # noqa: E402
    convert_evaluation_run_configuration_to_the_human_welfare_and_neutral_system_prompt_and_random_adapter_conditions as convert_to_new_conditions)

RUN_FOLDER_NAME_PREFIX = "overnight_treatment_arm_chain"
RUNS_ROOT = REPOSITORY_ROOT / "data" / "emergent_misalignment_runs"
ADAPTER_NAME = "treatment_arm"
RANDOM_ADAPTER_NAME = "random_matched_in_scale"
PIVOT_TOKEN_INPUTS_FOLDER_NAME = "betley_pivot_token_analysis"  # inside overnight_inputs; copied from the previous run folder
PIVOT_BETLEY_RUN_CONFIGURATION_FILE_NAME = "betley_run_configuration_of_the_previous_run.json"
GEMMA_BLOCK_KINDS = ["sage_eval_and_harmbench_thinking_on", "held_out_new_conditions", "betley_new_conditions", "mmlu_pro_new_conditions",
                     "sycophancy_harmbench_sage_eval_new_conditions", "harmbench_random_adapter"]
BETLEY_SAMPLES_PER_QUESTION = 100
PLACEHOLDER_ADAPTER_CONFIGURATION = {"peft_type": "LORA", "r": 32, "lora_alpha": 64, "target_modules": ["q_proj"],
                                     "note": "placeholder, overwritten on the pod after training"}


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def make_placeholder_adapter(folder):
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps(PLACEHOLDER_ADAPTER_CONFIGURATION, indent=2), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"placeholder adapter weights, replaced on the pod after training")


def builder_commands(python_executable, placeholder_adapter, staging_root, sage_dataset_json, model_choice="gemma", include_thinking_on_evaluations=False,
                     include_new_conditions=False, only_kinds=None):
    """{evaluation kind: builder command line}. The kinds named *_new_conditions (and harmbench_random_adapter) are built with a random adapter placeholder and converted afterwards."""
    def common(adapter_name):
        return ["--model", model_choice, "--adapter", f"{adapter_name}={placeholder_adapter}", "--runs-root", str(staging_root)]
    treatment, random_adapter = common(ADAPTER_NAME), common(RANDOM_ADAPTER_NAME)
    held_out_builder, betley_builder = "build_held_out_set_base_versus_lora_adapters_run_folder.py", "build_emergent_misalignment_evaluation_run_folder.py"
    mmlu_builder, syco_builder = "build_mmlu_pro_capability_check_run_folder.py", "build_sycophancy_harmbench_and_sage_eval_evaluation_run_folder.py"
    sage_tasks = ["--tasks", "are_you_sure", "harmbench_direct_request", "sage_eval", "--sage-eval-full-dataset-json", str(sage_dataset_json)] if sage_dataset_json else []
    commands = {
        "held_out": [python_executable, str(EMERGENT / held_out_builder), *treatment],
        "betley": [python_executable, str(EMERGENT / betley_builder), *treatment, "--samples-per-question", str(BETLEY_SAMPLES_PER_QUESTION)],
        "mmlu_pro": [python_executable, str(EMERGENT / mmlu_builder), *treatment],
        "sycophancy_harmbench_sage_eval": [python_executable, str(EMERGENT / syco_builder), *treatment, *sage_tasks],
    }
    if include_thinking_on_evaluations:
        commands["held_out_thinking_on"] = [python_executable, str(EMERGENT / held_out_builder), *treatment, "--thinking-on-conditions"]
        commands["sage_eval_and_harmbench_thinking_on"] = [
            python_executable, str(EMERGENT / syco_builder), *treatment, "--thinking-on",
            "--tasks", "harmbench_direct_request", *(["sage_eval", "--sage-eval-full-dataset-json", str(sage_dataset_json)] if sage_dataset_json else [])]
    if include_new_conditions:
        commands["held_out_new_conditions"] = [python_executable, str(EMERGENT / held_out_builder), *random_adapter]
        commands["betley_new_conditions"] = [python_executable, str(EMERGENT / betley_builder), *random_adapter, "--samples-per-question", str(BETLEY_SAMPLES_PER_QUESTION)]
        commands["mmlu_pro_new_conditions"] = [python_executable, str(EMERGENT / mmlu_builder), *random_adapter]
        commands["sycophancy_harmbench_sage_eval_new_conditions"] = [python_executable, str(EMERGENT / syco_builder), *random_adapter, "--no-base-with-system-prompt",
                                                                    *sage_tasks]
        commands["harmbench_random_adapter"] = [python_executable, str(EMERGENT / syco_builder), *random_adapter, "--tasks", "harmbench_direct_request",
                                                "--harmbench-samples-per-behavior", "1", "--no-base-with-system-prompt"]
    return {kind: command for kind, command in commands.items() if only_kinds is None or kind in only_kinds}


def build_evaluation_kinds(commands, run_command, staging, destination_parent, model_choice, condition_builder):
    """Runs each builder command, moves its run folder to destination_parent/<kind> and adds the conditions the kind needs; returns {kind: folder name}."""
    built_names = {}
    for kind, command in commands.items():
        run_command(command)
        created = sorted(path for path in staging.iterdir() if path.is_dir())
        if len(created) != 1:
            raise SystemExit(f"{kind}: expected one run folder in {staging}, found {len(created)}")
        destination = destination_parent / kind
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(created[0]), destination)
        built_names[kind] = created[0].name
        if kind == "harmbench_random_adapter":
            convert_to_new_conditions.keep_only_random_adapter_entry(destination)
        elif kind.endswith("_new_conditions"):
            convert_to_new_conditions.convert_run_folder(destination, model_choice)
        elif kind in ("held_out", "betley", "mmlu_pro"):
            add_condition.add_system_prompt_condition(destination, condition_builder)
    return built_names


def written_by_model_short_names(training_examples_file):
    """The set of model_short_name values in a training examples file; the chain trains on one other model's examples."""
    names = set()
    with open(training_examples_file, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                names.add(json.loads(line).get("model_short_name"))
    return names


def build_run_folder(arguments, run_command=None, condition_builder=None):
    run_command = run_command or (lambda command: subprocess.run(command, check=True))
    model_choice = arguments.model
    condition_builder = condition_builder or add_condition.system_prompt_condition_builder_for(model_choice)
    base_model_short_name = common_definitions.MODEL_CHOICES[model_choice]["model_short_name"]
    examples_file = Path(arguments.training_examples_file_from_another_model).resolve() if arguments.training_examples_file_from_another_model else None
    if examples_file:
        authors = written_by_model_short_names(examples_file)
        if len(authors) != 1 or base_model_short_name in authors:
            raise SystemExit(f"{examples_file} must hold examples written by one model other than {base_model_short_name}; it holds {sorted(map(str, authors))}")
    sage_dataset_json = Path(arguments.sage_eval_full_dataset_json).resolve() if arguments.sage_eval_full_dataset_json else None
    runs_root = Path(arguments.runs_root).resolve()
    if sage_dataset_json:
        if not sage_dataset_json.is_file():
            raise SystemExit(f"{sage_dataset_json} does not exist")
        if (REPOSITORY_ROOT / "data").resolve() not in [runs_root, *runs_root.parents]:
            raise SystemExit(f"SAGE-Eval content would be written under {runs_root}, which is not under {REPOSITORY_ROOT / 'data'}; refusing")
    timestamp = timestamp_now()
    name_middle = f"_{base_model_short_name}_trained_on_{next(iter(authors))}_training_examples" if examples_file else (
        "" if model_choice == "gemma" else f"_{base_model_short_name}")
    run_folder = runs_root / f"{RUN_FOLDER_NAME_PREFIX}{name_middle}_{timestamp}"
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists")
    inputs = run_folder / "overnight_inputs"
    run_folders = inputs / "run_folders"
    run_folders.mkdir(parents=True)
    generation_source = Path(arguments.treatment_arm_generation_run_folder).resolve()
    shutil.copytree(generation_source, run_folders / "treatment_arm_generation",
                    ignore=shutil.ignore_patterns("generation_outputs") if examples_file else None)
    if examples_file:
        examples_destination = inputs / "training_examples_from_another_model"
        examples_destination.mkdir()
        shutil.copyfile(examples_file, examples_destination / examples_file.name)
    held_out_destination = inputs / "held_out_set_file"
    held_out_destination.mkdir()
    shutil.copyfile(arguments.held_out_set_file, held_out_destination / Path(arguments.held_out_set_file).name)
    placeholder = run_folder / "placeholder_adapter_replaced_on_pod_after_training"
    make_placeholder_adapter(placeholder)
    staging = run_folder / "evaluation_run_folder_builder_staging"
    built_names = build_evaluation_kinds(
        builder_commands(arguments.python_executable, placeholder, staging, sage_dataset_json, model_choice, arguments.include_thinking_on_evaluations,
                         arguments.include_new_conditions), run_command, staging, run_folders / "evaluation", model_choice, condition_builder)
    additional_block = None
    if arguments.additional_gemma_evaluation_block_adapter_run_folder_on_the_volume:
        previous_run_folder = Path(arguments.additional_gemma_evaluation_block_previous_run_folder or "missing").resolve()
        pivot_inputs = previous_run_folder / PIVOT_TOKEN_INPUTS_FOLDER_NAME
        previous_betley_configurations = sorted((previous_run_folder / "overnight_inputs" / "run_folders" / "evaluation" / "betley").glob("run_configuration_*.json"))
        if not pivot_inputs.is_dir() or not previous_betley_configurations:
            raise SystemExit(f"--additional-gemma-evaluation-block-previous-run-folder must hold {PIVOT_TOKEN_INPUTS_FOLDER_NAME}/ and the Betley run configuration")
        shutil.copytree(pivot_inputs, inputs / PIVOT_TOKEN_INPUTS_FOLDER_NAME)
        shutil.copyfile(previous_betley_configurations[-1], inputs / PIVOT_TOKEN_INPUTS_FOLDER_NAME / PIVOT_BETLEY_RUN_CONFIGURATION_FILE_NAME)
        gemma_short_name = common_definitions.MODEL_CHOICES["gemma"]["model_short_name"]
        gemma_names = build_evaluation_kinds(
            builder_commands(arguments.python_executable, placeholder, staging, sage_dataset_json, "gemma", True, True, GEMMA_BLOCK_KINDS), run_command, staging,
            run_folders / "evaluation" / gemma_short_name, "gemma", add_condition.system_prompt_condition_builder_for("gemma"))
        additional_block = {"base_model_choice": "gemma", "base_model_short_name": gemma_short_name, "evaluation_folder_name": gemma_short_name,
                            "adapter_fine_tuning_run_folder_on_the_volume": arguments.additional_gemma_evaluation_block_adapter_run_folder_on_the_volume,
                            "previous_run_folder": str(previous_run_folder),
                            "evaluation_run_folders_built_as": gemma_names}
    shutil.rmtree(staging)
    configuration = {
        "chain": "overnight treatment arm chain", "built_at": datetime.now().isoformat(timespec="seconds"),
        "base_model_choice": model_choice, "base_model_short_name": base_model_short_name,
        "training_examples_source_kind": "training_examples_file_from_another_model" if examples_file else "kept_answers_generated_in_this_chain",
        "training_examples_file_from_another_model": str(examples_file) if examples_file else None,
        "training_examples_written_by": next(iter(authors)) if examples_file else base_model_short_name,
        "sage_eval_dataset_supplied": bool(sage_dataset_json), "training_example_seed": arguments.training_example_seed,
        "smoke_run_examples": arguments.smoke_run_examples, "adapter_name": ADAPTER_NAME,
        "treatment_arm_generation_run_folder_copied_from": str(generation_source), "evaluation_run_folders_built_as": built_names,
        "additional_model_evaluation_block": additional_block,
        "held_out_set_file": str(Path(arguments.held_out_set_file).resolve()), "betley_samples_per_question": BETLEY_SAMPLES_PER_QUESTION,
        "training_gate_minimum_prompts_with_a_kept_answer": 5000, "launcher_heartbeat_stale_seconds": 1200,
        "judging": ("Claude Haiku 5.5 through the paid Anthropic API, run on the laptop after the chain (no key goes to the pod); the pod judges nothing"
                    if examples_file else
                    "Claude Haiku 5.5 through the paid Anthropic API, run on the laptop by the launcher (no key goes to the pod); the pod judges nothing"),
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
    parser.add_argument("--model", choices=["gemma", "qwen"], default="gemma")
    parser.add_argument("--training-examples-file-from-another-model", default=None)
    parser.add_argument("--include-thinking-on-evaluations", action="store_true")
    parser.add_argument("--include-new-conditions", action="store_true",
                        help="also the human-welfare and neutral system prompt entries (thinking off) and the random adapter on HarmBench alone (one sample per behaviour)")
    parser.add_argument("--additional-gemma-evaluation-block-adapter-run-folder-on-the-volume", default=None, metavar="PATH_ON_THE_POD",
                        help="adds the Gemma evaluation block (thinking-on HarmBench and SAGE-Eval, the new conditions, the pivot tokens); the path is the fine tuning "
                             "run folder holding last night's trained Gemma adapter, as it is on the network volume")
    parser.add_argument("--additional-gemma-evaluation-block-previous-run-folder", default=None, help="last night's run folder on the laptop")
    parser.add_argument("--held-out-set-file", default=str(REPOSITORY_ROOT / defs.HELD_OUT_SET_FILE_RELATIVE_PATH))
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--training-example-seed", type=int, default=20261111)
    parser.add_argument("--smoke-run-examples", type=int, default=200)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_run_folder(parse_arguments())
