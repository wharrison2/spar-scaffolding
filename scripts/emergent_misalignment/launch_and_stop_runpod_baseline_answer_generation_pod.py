"""Launch one RunPod GPU pod for the baseline welfare-raising rate screen (three candidate models, thinking off and on),
pull results back, and always delete the pod. A thin job profile over the persona-check launcher
(scripts/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py), in the pattern of
scripts/persona_vector_projection/launch_and_stop_runpod_generation_pod.py: all pod handling, cost limits, cost-center
reminders, the on-pod watchdog, the periodic pulls and the verified 5-minute final pull are that module's, unchanged. This
file only says what to upload, what to run, what to pull, which disk sizes to ask for and how to read progress.

    python3 scripts/emergent_misalignment/launch_and_stop_runpod_baseline_answer_generation_pod.py --run-folder <run folder> \
        --maximum-running-hours 2.5 --maximum-dollars 10 --dry-run          # free: prints the pod request and cost ceiling
    ... without --dry-run                                                    # PAID, needs the user's go-ahead

The normal launch needs no extra flag: the pod script starts each model with a one-answer check (one answer to one prompt per
condition: the thinking switch works and the trace splits cleanly) and carries straight on to the full generation when it
passes. A failed check exits the pod script non-zero, so this launcher's existing failure handling applies (pull, then wait
for a retry or delete instruction file). --one-answer-check alone runs only the checks. Resume on a new pod: run the same
command again with the same run folder (uploaded answers files are skipped, and a saved passing check is not repeated).
Extra options passed to the pod script: --one-answer-check, --skip-embedded-one-answer-check, --keep-model-caches, --generation-backend,
--only-model-short-names.
"""

import argparse
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as pod_launcher  # noqa: E402

SCRIPT_RELATIVE_PATH = "scripts/emergent_misalignment/generate_baseline_answers_for_candidate_models_on_pod.py"
REQUIREMENTS_RELATIVE_PATH = "scripts/emergent_misalignment/requirements_for_baseline_answer_generation_pod.txt"
CONTAINER_DISK_GIGABYTES = 50
VOLUME_GIGABYTES = 200  # two models at once (about 63 GB and 60 GB) plus headroom
MINIMUM_LOCAL_FREE_DISK_BYTES = 10 * 10 ** 9
POD_SCRIPT_OPTIONS = {}


def build_pod_script_command_line(arguments):
    command = ["python", SCRIPT_RELATIVE_PATH, "--run-folder", pod_launcher.REMOTE_RUN_FOLDER]
    if POD_SCRIPT_OPTIONS.get("--one-answer-check"):
        command.append("--one-answer-check")
    if POD_SCRIPT_OPTIONS.get("--skip-embedded-one-answer-check"):
        command.append("--skip-embedded-one-answer-check")
    if POD_SCRIPT_OPTIONS.get("--keep-model-caches"):
        command.append("--keep-model-caches")
    if POD_SCRIPT_OPTIONS.get("--generation-backend"):
        command += ["--generation-backend", POD_SCRIPT_OPTIONS["--generation-backend"]]
    if POD_SCRIPT_OPTIONS.get("--only-model-short-names"):
        command += ["--only-model-short-names", *POD_SCRIPT_OPTIONS["--only-model-short-names"]]
    return command


def apply_job_profile():
    inputs, outputs = defs.INPUTS_FOLDER_NAME, defs.OUTPUTS_FOLDER_NAME
    pod_launcher.INPUTS_FOLDER_NAME, pod_launcher.OUTPUTS_FOLDER_NAME = inputs, outputs
    pod_launcher.POD_NAME_PREFIX = "baseline-welfare-raising-rate-screen"
    pod_launcher.SCRIPT_FOLDER_NAME = "scripts/emergent_misalignment"
    pod_launcher.SCORER_SCRIPT_RELATIVE_PATH = SCRIPT_RELATIVE_PATH
    pod_launcher.REQUIREMENTS_FILE_RELATIVE_PATH = REQUIREMENTS_RELATIVE_PATH
    pod_launcher.SCRIPT_FOLDER_NAMES_TO_UPLOAD = ["scripts/emergent_misalignment"]
    pod_launcher.SCORER_PROCESS_PATTERN = "[g]enerate_baseline_answers_for_candidate_models_on_pod"
    pod_launcher.PROGRESS_STATUS_SHELL_TEMPLATE = (
        "cat {remote_run_folder}/" + outputs + "/answers__*.jsonl 2>/dev/null | wc -l; "
        "ls -t {remote_run_folder}/" + outputs + "/generation_progress_log_*.txt 2>/dev/null | head -n 1 | xargs -r tail -n 1")
    pod_launcher.PROGRESS_STATUS_UNIT_NAME = "answers saved on pod"
    pod_launcher.MINIMUM_LOCAL_FREE_DISK_BYTES = MINIMUM_LOCAL_FREE_DISK_BYTES
    pod_launcher.DEFAULT_CONTAINER_DISK_GIGABYTES = CONTAINER_DISK_GIGABYTES
    pod_launcher.DEFAULT_VOLUME_GIGABYTES = VOLUME_GIGABYTES
    pod_launcher.FINAL_COPY_FILE_LIST_SHELL = (
        f"find run_folder -type f -not -path 'run_folder/{inputs}/*' -not -name '.last_pull_marker*'; "
        "for log_file in watchdog_log.txt pod_setup_*.txt; do [ -f \"$log_file\" ] && echo \"$log_file\"; done; true")
    pod_launcher.build_scorer_command_line = build_pod_script_command_line


def print_dry_run_with_one_answer_check_note(arguments, secrets, original_print_dry_run=pod_launcher.print_dry_run):
    original_print_dry_run(arguments, secrets)
    print("One-answer check (job profile note): the pod script starts each model with one answer to one prompt per condition "
          "(thinking_off, thinking_on), checks that the thinking switch works and the trace splits cleanly, and writes "
          f"{defs.OUTPUTS_FOLDER_NAME}/{defs.ONE_ANSWER_CHECK_FOLDER_NAME}/one_answer_check_<model>_<timestamp>.json (not the answer files). "
          "If every check passes it carries on with the full generation without pausing; if one fails it exits non-zero, which "
          "the launcher handles as a scorer failure above (pull, then wait for a retry or delete instruction file).")


def main(argv=None):
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--one-answer-check", action="store_true")
    extra_parser.add_argument("--skip-embedded-one-answer-check", action="store_true")
    extra_parser.add_argument("--keep-model-caches", action="store_true")
    extra_parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default=None)
    extra_parser.add_argument("--only-model-short-names", nargs="+", default=None)
    extra, remaining = extra_parser.parse_known_args(argv)
    POD_SCRIPT_OPTIONS.update({"--one-answer-check": extra.one_answer_check, "--keep-model-caches": extra.keep_model_caches,
                               "--skip-embedded-one-answer-check": extra.skip_embedded_one_answer_check,
                               "--generation-backend": extra.generation_backend,
                               "--only-model-short-names": extra.only_model_short_names})
    apply_job_profile()
    pod_launcher.print_dry_run = print_dry_run_with_one_answer_check_note
    arguments = pod_launcher.parse_arguments(remaining)
    return pod_launcher.run_launcher(arguments)


if __name__ == "__main__":
    sys.exit(main())
