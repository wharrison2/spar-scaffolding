"""Runs on the pod, started once with nohup by the launcher: the whole overnight treatment arm chain, unattended (see
overnight_treatment_arm_chain_step_definitions.py for the steps and the generic engine in
scripts/shared/resumable_step_chain_with_failure_policy_and_pod_self_deletion.py for the failure policy, markers and the end of chain).

    python3 scripts/emergent_misalignment/overnight_chain/run_overnight_treatment_arm_chain_on_pod.py \
        --working-directory /workspace/emergent_alignment_overnight_copy --seconds-until-hard-limit 43000 --pod-hourly-price-dollars 3.6

Reads run_configuration_overnight_chain_*.json from <working directory>/run_folder. Starts the dead-man watchdog (deletes the pod if this
process disappears without the finished flag), runs the chain, and at the end deletes the pod from the inside. The launcher's heartbeat
file (<run folder>/launcher_heartbeat_epoch_seconds.txt) tells it whether a person is connected.

Options for tests and the local rehearsal only: --no-pod-self-deletion (do not delete anything), --no-dead-man-watchdog,
--markers-and-environments-root (replaces /root for the container-disk paths).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.emergent_misalignment.overnight_chain import overnight_treatment_arm_chain_step_definitions as definitions  # noqa: E402
from scripts.shared.pod_self_deletion_shell_commands_for_runpod_pods import (  # noqa: E402
    build_dead_man_self_deletion_watchdog_command, build_delete_this_pod_shell_command)
from scripts.shared.resumable_step_chain_with_failure_policy_and_pod_self_deletion import StepChainRunner  # noqa: E402

ORCHESTRATOR_PROCESS_PATTERN = "[r]un_overnight_treatment_arm_chain_on_pod"  # bracket: pgrep does not match itself
DEAD_MAN_GRACE_SECONDS = 30 * 60
STOP_STARTING_STEPS_BEFORE_LIMIT_SECONDS = 15 * 60
STOP_RUNNING_STEP_BEFORE_LIMIT_SECONDS = 10 * 60
FINAL_PULL_WAIT_SECONDS = 15 * 60


def load_chain_configuration(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_overnight_chain_*.json"))
    if not paths:
        raise SystemExit(f"no run_configuration_overnight_chain_*.json in {run_folder}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--working-directory", required=True)
    parser.add_argument("--seconds-until-hard-limit", type=float, required=True,
                        help="how long this pod may live from now; the chain stops starting steps 15 minutes before and stops a running step 10 minutes before")
    parser.add_argument("--pod-hourly-price-dollars", type=float, default=None)
    parser.add_argument("--self-deletion-command-for-rehearsal", default=None, help="shell command run instead of deleting the pod (rehearsal only)")
    parser.add_argument("--no-pod-self-deletion", action="store_true")
    parser.add_argument("--no-dead-man-watchdog", action="store_true")
    parser.add_argument("--markers-and-environments-root", default=None)
    parser.add_argument("--container-work-root", default="/root/overnight_work", help="fast local disk for the work tree that is synced to the volume")
    parser.add_argument("--sync-interval-seconds", type=float, default=180.0)
    parser.add_argument("--volume-mount-path", default="/workspace", help="where the network volume is mounted (adapters and checkpoints must be under it)")
    parser.add_argument("--progress-interval-seconds", type=float, default=180.0)
    parser.add_argument("--failure-wait-seconds", type=float, default=20 * 60)
    parser.add_argument("--final-pull-wait-seconds", type=float, default=FINAL_PULL_WAIT_SECONDS)
    arguments = parser.parse_args(argv)

    working_directory = Path(arguments.working_directory)
    run_folder = working_directory / "run_folder"
    configuration = load_chain_configuration(run_folder)
    paths = definitions.ChainPaths(working_directory, run_folder, arguments.container_work_root)
    if arguments.markers_and_environments_root:  # rehearsal: container-disk paths under a local folder
        root = arguments.markers_and_environments_root
        definitions.ENVIRONMENTS_ROOT = f"{root}/overnight_environments"
        definitions.VLLM_ENVIRONMENT = f"{definitions.ENVIRONMENTS_ROOT}/vllm_environment"
        definitions.FINE_TUNING_ENVIRONMENT = f"{definitions.ENVIRONMENTS_ROOT}/fine_tuning_environment"
        definitions.CONTAINER_MARKERS_FOLDER = f"{root}/overnight_environment_markers"
        definitions.HUGGING_FACE_CACHE = f"{root}/huggingface_cache"
    import os
    real_run_folder = os.path.realpath(run_folder)
    mount = arguments.volume_mount_path.rstrip("/")
    if not (real_run_folder == mount or real_run_folder.startswith(mount + "/")):
        raise SystemExit(f"{real_run_folder} is not under the network volume mount {mount}: adapters and checkpoints would not be durable")
    restored = definitions.restore_work_tree_from_volume(paths)
    if restored is not None:
        print(f"restored the work tree from the volume: {restored['files_copied']} files copied, {restored['files_unchanged']} already present, "
              f"{len(restored['problems'])} problems", flush=True)
    steps = definitions.build_chain_steps(paths, configuration)
    python_by_environment_name = {"plain": sys.executable,
                                  "vllm": definitions.python_executable_for(definitions.VLLM_ENVIRONMENT),
                                  "fine_tuning": definitions.python_executable_for(definitions.FINE_TUNING_ENVIRONMENT)}
    python_by_environment_name.update(configuration.get("python_by_environment_name", {}))
    delete_command = arguments.self_deletion_command_for_rehearsal or (None if arguments.no_pod_self_deletion else build_delete_this_pod_shell_command())
    for folder in (paths.summary, paths.instructions):
        folder.mkdir(parents=True, exist_ok=True)
    if not arguments.no_dead_man_watchdog and delete_command and not arguments.self_deletion_command_for_rehearsal:
        subprocess.run(build_dead_man_self_deletion_watchdog_command(
            ORCHESTRATOR_PROCESS_PATTERN, paths.finished_flag_file, paths.summary.parent / "dead_man_watchdog_log.txt",
            DEAD_MAN_GRACE_SECONDS), shell=True, check=False)
    runner = StepChainRunner(
        steps, paths.markers, definitions.CONTAINER_MARKERS_FOLDER, paths.logs, paths.instructions, paths.summary,
        python_by_environment_name=python_by_environment_name,
        base_environment_variables={"PYTHONUNBUFFERED": "1", "HF_HOME": definitions.HUGGING_FACE_CACHE, "TOKENIZERS_PARALLELISM": "false",
                                    "PYTHONPATH": str(working_directory), **configuration.get("extra_environment_variables", {})},
        heartbeat_file_path=paths.heartbeat_file, finished_flag_file_path=paths.finished_flag_file,
        final_pull_acknowledgement_file_path=paths.final_pull_acknowledgement_file,
        seconds_until_hard_limit=arguments.seconds_until_hard_limit,
        stop_starting_steps_this_long_before_limit=STOP_STARTING_STEPS_BEFORE_LIMIT_SECONDS,
        finalise_this_long_before_limit=STOP_RUNNING_STEP_BEFORE_LIMIT_SECONDS,
        failure_wait_seconds=arguments.failure_wait_seconds, final_pull_wait_seconds=arguments.final_pull_wait_seconds,
        self_deletion_command=delete_command, hourly_price_dollars=arguments.pod_hourly_price_dollars,
        summary_extra_function=lambda: {"paths_on_the_network_volume_of_trained_adapters_and_resume_checkpoints":
                                        definitions.list_trained_adapter_and_checkpoint_paths_on_the_volume(paths, arguments.volume_mount_path),
                                        "network_volume_mount_path": arguments.volume_mount_path,
                                        **definitions.volume_usage_for_summary(arguments.volume_mount_path)},
        housekeeping_function=definitions.make_housekeeping_function(paths, arguments.volume_mount_path),
        sync_function=definitions.make_sync_function(paths), sync_interval_seconds=arguments.sync_interval_seconds,
        chain_name="overnight treatment arm chain", progress_interval_seconds=arguments.progress_interval_seconds)
    reason = runner.run()
    return 0 if reason == "chain_ran_to_the_end" and runner.overall_status() == "all_steps_completed" else 1


if __name__ == "__main__":
    sys.exit(main())
