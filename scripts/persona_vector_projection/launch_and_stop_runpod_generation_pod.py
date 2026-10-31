"""Launch one RunPod GPU pod for the persona vector extraction (generation and activation pass), pull results back, and
always delete the pod. A thin job profile over the persona-check launcher (scripts/log_probability_persona_check/
launch_and_stop_runpod_scoring_pod.py): all pod handling, cost limits, cost-center reminders, watchdog, periodic pulls
and the verified final pull are that module's. This file only tells it what to upload, what to run, what to pull and
how much disk to ask for, and adds a free-disk check before each pull (stops pulling under 10 GB free).

    python3 scripts/persona_vector_projection/launch_and_stop_runpod_generation_pod.py --run-folder <run folder> \
        --maximum-running-hours 3 --maximum-dollars 15 --dry-run          # free: prints the pod request and cost ceiling
    ... without --dry-run                                                   # PAID, needs the user's go-ahead

Resume on a new pod: run the same command again with the same run folder (finished shards are uploaded with it and skipped).
Extra options (all passed to the pod script): --stage, --activation-batch-size, --maximum-jobs.
"""

import argparse
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as pod_launcher  # noqa: E402

SCRIPT_RELATIVE_PATH = "scripts/persona_vector_projection/generate_answers_and_save_activations_on_pod.py"
REQUIREMENTS_RELATIVE_PATH = "scripts/persona_vector_projection/requirements_for_extraction_pod.txt"
CONTAINER_DISK_GIGABYTES = 80
VOLUME_GIGABYTES = 120  # model weights about 24 GB, vLLM and Hugging Face caches, about 5 GB of activations for a full run
MINIMUM_LOCAL_FREE_DISK_BYTES = 10 * 10 ** 9
EXTRA_POD_SCRIPT_OPTIONS = {}


def build_pod_script_command_line(arguments):
    command = ["python", SCRIPT_RELATIVE_PATH, "--run-folder", pod_launcher.REMOTE_RUN_FOLDER]
    for flag, value in EXTRA_POD_SCRIPT_OPTIONS.items():
        if value is not None:
            command += [flag, str(value)]
    for flag, value in (("--model-name", arguments.model_name), ("--model-revision", arguments.model_revision)):
        if value is not None:
            command += [flag, str(value)]
    return command


def apply_job_profile():
    inputs, outputs = defs.INPUTS_FOLDER_NAME, defs.OUTPUTS_FOLDER_NAME
    pod_launcher.INPUTS_FOLDER_NAME, pod_launcher.OUTPUTS_FOLDER_NAME = inputs, outputs
    pod_launcher.POD_NAME_PREFIX = "persona-vector-extraction"
    pod_launcher.SCRIPT_FOLDER_NAME = "scripts/persona_vector_projection"
    pod_launcher.SCORER_SCRIPT_RELATIVE_PATH = SCRIPT_RELATIVE_PATH
    pod_launcher.REQUIREMENTS_FILE_RELATIVE_PATH = REQUIREMENTS_RELATIVE_PATH
    pod_launcher.SCRIPT_FOLDER_NAMES_TO_UPLOAD = ["scripts/persona_vector_projection", "scripts/log_probability_persona_check"]
    pod_launcher.SCORER_PROCESS_PATTERN = "[g]enerate_answers_and_save_activations_on_pod"
    pod_launcher.PROGRESS_STATUS_SHELL_TEMPLATE = (
        "ls {remote_run_folder}/" + outputs + "/" + defs.ACTIVATIONS_FOLDER_NAME + " 2>/dev/null | wc -l; "
        "ls -t {remote_run_folder}/" + outputs + "/generation_progress_log_*.txt {remote_run_folder}/" + outputs +
        "/activation_progress_log_*.txt 2>/dev/null | head -n 1 | xargs -r tail -n 1")
    pod_launcher.PROGRESS_STATUS_UNIT_NAME = "answer shards with activations on pod"
    pod_launcher.MINIMUM_LOCAL_FREE_DISK_BYTES = MINIMUM_LOCAL_FREE_DISK_BYTES
    pod_launcher.DEFAULT_CONTAINER_DISK_GIGABYTES = CONTAINER_DISK_GIGABYTES
    pod_launcher.DEFAULT_VOLUME_GIGABYTES = VOLUME_GIGABYTES
    pod_launcher.FINAL_COPY_FILE_LIST_SHELL = (
        f"find run_folder -type f -not -path 'run_folder/{inputs}/*' -not -name '.last_pull_marker*'; "
        "for log_file in watchdog_log.txt pod_setup_*.txt; do [ -f \"$log_file\" ] && echo \"$log_file\"; done; true")
    pod_launcher.build_scorer_command_line = build_pod_script_command_line


def main(argv=None):
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--stage", default=None)
    extra_parser.add_argument("--activation-batch-size", type=int, default=None)
    extra_parser.add_argument("--maximum-jobs", type=int, default=None)
    extra, remaining = extra_parser.parse_known_args(argv)
    EXTRA_POD_SCRIPT_OPTIONS.update({"--stage": extra.stage, "--activation-batch-size": extra.activation_batch_size,
                                     "--maximum-jobs": extra.maximum_jobs})
    apply_job_profile()
    arguments = pod_launcher.parse_arguments(remaining)
    if arguments.model_revision is None:
        arguments.model_revision = defs.DEFAULT_MODEL_REVISION
    return pod_launcher.run_launcher(arguments)


if __name__ == "__main__":
    sys.exit(main())
