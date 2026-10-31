"""Launches the scoring pod (matched baselines, activation scoring, steering) with the shared RunPod launcher.

All pod mechanics (limits, watchdog, pulls, cost log, cost center) are the shared launcher's
(scripts/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py); this file only gives it this job's
profile: folders, script, requirements and progress command. Always dry-run first:

    python3 scripts/persona_vector_projection/launch_and_stop_runpod_scoring_and_steering_pod.py \
        --run-folder <run folder> --maximum-running-hours 2 --dry-run
Any flag of the shared launcher works. Extra flags here are passed to run_scoring_pod_stages.py: --stages,
--document-formats, --generation-backend, --skip-per-token-arrays.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check", SCRIPTS_DIRECTORY.parent):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import launch_and_stop_runpod_scoring_pod as shared_launcher  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import estimate_scoring_pod_gpu_hours_and_cost as estimator  # noqa: E402

SCRIPT_FOLDER = "scripts/persona_vector_projection"
STAGES_SCRIPT_RELATIVE_PATH = f"{SCRIPT_FOLDER}/run_scoring_pod_stages.py"
MINIMUM_LOCAL_FREE_DISK_BYTES = 2_000_000_000


def apply_job_profile():
    shared_launcher.INPUTS_FOLDER_NAME = defs.INPUTS_FOLDER_NAME
    shared_launcher.OUTPUTS_FOLDER_NAME = defs.OUTPUTS_FOLDER_NAME
    shared_launcher.POD_NAME_PREFIX = "persona-vector-scoring"
    shared_launcher.SCRIPT_FOLDER_NAMES_TO_UPLOAD = [SCRIPT_FOLDER, "scripts/log_probability_persona_check"]
    shared_launcher.SCORER_SCRIPT_RELATIVE_PATH = STAGES_SCRIPT_RELATIVE_PATH
    shared_launcher.SCORER_PROCESS_PATTERN = "[r]un_scoring_pod_stages"
    shared_launcher.REQUIREMENTS_FILE_RELATIVE_PATH = f"{SCRIPT_FOLDER}/scoring_pod_requirements.txt"
    shared_launcher.MINIMUM_LOCAL_FREE_DISK_BYTES = MINIMUM_LOCAL_FREE_DISK_BYTES
    shared_launcher.PROGRESS_STATUS_UNIT_NAME = "output shards on pod"
    outputs = f"{{remote_run_folder}}/{defs.OUTPUTS_FOLDER_NAME}"
    shared_launcher.PROGRESS_STATUS_SHELL_TEMPLATE = (
        f"find {outputs} -name '*shard*' ! -name '*partial*' 2>/dev/null | wc -l; "
        f"ls -t {outputs}/*_progress_log_*.txt 2>/dev/null | head -n 1 | xargs -r tail -n 1")
    original_command_builder = shared_launcher.build_scorer_command_line

    def build_command_with_stage_flags(arguments):
        command = original_command_builder(arguments)
        command[1] = STAGES_SCRIPT_RELATIVE_PATH
        if arguments.stages:
            command += ["--stages", *arguments.stages]
        if arguments.document_formats:
            command += ["--document-formats", *arguments.document_formats]
        if arguments.generation_backend:
            command += ["--generation-backend", arguments.generation_backend]
        if arguments.skip_per_token_arrays:
            command.append("--skip-per-token-arrays")
        return command

    shared_launcher.build_scorer_command_line = build_command_with_stage_flags


def parse_arguments(argv=None):
    own_parser = argparse.ArgumentParser(add_help=False)
    own_parser.add_argument("--stages", nargs="+", choices=defs.ALL_STAGES, default=None)
    own_parser.add_argument("--document-formats", nargs="+", choices=defs.DOCUMENT_FORMATS, default=None)
    own_parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default=None)
    own_parser.add_argument("--skip-per-token-arrays", action="store_true")
    own_arguments, remaining = own_parser.parse_known_args(argv)
    arguments = shared_launcher.parse_arguments(remaining)
    for name, value in vars(own_arguments).items():
        setattr(arguments, name, value)
    if "--container-disk-gigabytes" not in remaining:
        arguments.container_disk_gigabytes = estimator.POD_DISK_CONTAINER_GIGABYTES
    if "--volume-gigabytes" not in remaining:
        arguments.volume_gigabytes = estimator.POD_DISK_VOLUME_GIGABYTES
    if "--maximum-dollars" not in remaining:  # the ceiling printed by the estimate: hours x assumed dollars per hour
        arguments.maximum_dollars = round(arguments.maximum_running_hours * defs.GPU_DOLLARS_PER_HOUR, 2)
    return arguments


def print_estimate(arguments):
    configuration_paths = sorted(Path(arguments.run_folder).glob("run_configuration_*.json"))
    if not configuration_paths:
        print("No run_configuration in the run folder; no estimate")
        return
    configuration = json.loads(configuration_paths[0].read_text())
    estimate = estimator.estimate_gpu_hours(estimator.counts_from_run_configuration(configuration, arguments.stages))
    print("Estimated GPU time for this run (unmeasured throughput assumptions):\n" + estimator.describe(estimate))
    print(f"Hard limits given: {arguments.maximum_running_hours} hours and ${arguments.maximum_dollars}")
    if arguments.maximum_running_hours < estimate["estimated_gpu_hours"]:
        print("WARNING: the hours limit is below the estimate; the run would be cut off")


def main(argv=None):
    apply_job_profile()
    arguments = parse_arguments(argv)
    if arguments.dry_run:
        print_estimate(arguments)
    return shared_launcher.run_launcher(arguments)


if __name__ == "__main__":
    sys.exit(main())
