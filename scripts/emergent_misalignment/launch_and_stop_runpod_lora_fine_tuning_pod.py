"""Launch one RunPod GPU pod for a LoRA fine-tuning run on a model's own answers, pull the adapters and logs back, and always
delete the pod. A thin job profile over the persona-check launcher (scripts/log_probability_persona_check/
launch_and_stop_runpod_scoring_pod.py), in the pattern of launch_and_stop_runpod_baseline_answer_generation_pod.py: pod handling,
cost limits, cost-center reminders, the on-pod watchdog, periodic pulls and the verified final pull are that module's, unchanged.

GPU: one NVIDIA H100 80 GB for 4bit_qlora, one NVIDIA H200 141 GB for bf16_lora, chosen from the run configuration's quantisation;
--gpu-type-ids overrides. Disk: 150 GB volume (the 63 GB model, its cache, checkpoints), 50 GB container.

    python3 scripts/emergent_misalignment/launch_and_stop_runpod_lora_fine_tuning_pod.py --run-folder <run folder> \
        --maximum-running-hours 2 --maximum-dollars 10 --dry-run     # free: prints the pod request and cost ceiling
    ... without --dry-run                                             # PAID, needs the user's go-ahead

Resume on a new pod: run the same command with the same run folder (the uploaded outputs hold the checkpoints; the training script
continues from the latest complete one). Checkpoints hold optimizer state, so the periodic pulls move a few GB per checkpoint.
"""

import json
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import fine_tuning_run_definitions as defs  # noqa: E402
from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as pod_launcher  # noqa: E402

SCRIPT_RELATIVE_PATH = "scripts/emergent_misalignment/fine_tune_candidate_model_with_lora_on_own_answers_on_pod.py"
REQUIREMENTS_RELATIVE_PATH = "scripts/emergent_misalignment/requirements_for_lora_fine_tuning_pod.txt"
CONTAINER_DISK_GIGABYTES = 50
VOLUME_GIGABYTES = 150
MINIMUM_LOCAL_FREE_DISK_BYTES = 20 * 10 ** 9


def build_pod_script_command_line(arguments):
    return ["python", SCRIPT_RELATIVE_PATH, "--run-folder", pod_launcher.REMOTE_RUN_FOLDER]


def apply_job_profile():
    inputs, outputs = defs.INPUTS_FOLDER_NAME, defs.OUTPUTS_FOLDER_NAME
    pod_launcher.INPUTS_FOLDER_NAME, pod_launcher.OUTPUTS_FOLDER_NAME = inputs, outputs
    pod_launcher.POD_NAME_PREFIX = "lora-fine-tuning-on-own-answers"
    pod_launcher.SCRIPT_FOLDER_NAME = "scripts/emergent_misalignment"
    pod_launcher.SCORER_SCRIPT_RELATIVE_PATH = SCRIPT_RELATIVE_PATH
    pod_launcher.REQUIREMENTS_FILE_RELATIVE_PATH = REQUIREMENTS_RELATIVE_PATH
    pod_launcher.SCRIPT_FOLDER_NAMES_TO_UPLOAD = ["scripts/emergent_misalignment"]
    pod_launcher.SCORER_PROCESS_PATTERN = "[f]ine_tune_candidate_model_with_lora_on_own_answers_on_pod"
    pod_launcher.PROGRESS_STATUS_SHELL_TEMPLATE = (
        "cat {remote_run_folder}/" + outputs + "/" + defs.METRICS_LOG_FILE_NAME + " 2>/dev/null | wc -l; "
        "ls -t {remote_run_folder}/" + outputs + "/training_progress_log_*.txt 2>/dev/null | head -n 1 | xargs -r tail -n 1")
    pod_launcher.PROGRESS_STATUS_UNIT_NAME = "optimizer steps finished on pod"
    pod_launcher.MINIMUM_LOCAL_FREE_DISK_BYTES = MINIMUM_LOCAL_FREE_DISK_BYTES
    pod_launcher.DEFAULT_CONTAINER_DISK_GIGABYTES = CONTAINER_DISK_GIGABYTES
    pod_launcher.DEFAULT_VOLUME_GIGABYTES = VOLUME_GIGABYTES
    pod_launcher.FINAL_COPY_FILE_LIST_SHELL = (
        f"find run_folder -type f -not -path 'run_folder/{inputs}/*' -not -name '.last_pull_marker*'; "
        "for log_file in watchdog_log.txt pod_setup_*.txt; do [ -f \"$log_file\" ] && echo \"$log_file\"; done; true")
    pod_launcher.build_scorer_command_line = build_pod_script_command_line


def gpu_type_ids_for_run_folder(run_folder):
    configurations = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not configurations:
        raise SystemExit(f"No run_configuration_*.json in {run_folder}; build the run folder first.")
    quantisation = json.loads(configurations[-1].read_text(encoding="utf-8"))["hyperparameters"]["quantisation"]
    return defs.GPU_TYPE_IDS_BY_QUANTISATION[quantisation]


def print_dry_run_with_training_note(arguments, secrets, original_print_dry_run=pod_launcher.print_dry_run):
    original_print_dry_run(arguments, secrets)
    print("Training job profile note: the pod script first checks the tokenizer's chat template against the saved fixture, renders "
          "every example with the thinking-off generation prompt, checks the prompt tail and the assistant-only loss mask and "
          "refuses over-long examples, writing training_outputs/chat_template_and_masking_check_*.json, then trains with a checkpoint "
          "every N optimizer steps and every epoch. A failed check exits non-zero, which the launcher handles as a scorer failure "
          "(pull, then wait for a retry or delete instruction file).")


def main(argv=None):
    apply_job_profile()
    pod_launcher.print_dry_run = print_dry_run_with_training_note
    remaining = list(sys.argv[1:] if argv is None else argv)
    if "--gpu-type-ids" not in remaining:
        run_folder = remaining[remaining.index("--run-folder") + 1] if "--run-folder" in remaining else None
        if run_folder is not None:
            remaining += ["--gpu-type-ids", *gpu_type_ids_for_run_folder(run_folder)]
    arguments = pod_launcher.parse_arguments(remaining)
    return pod_launcher.run_launcher(arguments)


if __name__ == "__main__":
    sys.exit(main())
