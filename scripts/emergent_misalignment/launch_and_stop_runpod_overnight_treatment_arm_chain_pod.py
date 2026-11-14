"""Launch ONE H200 pod with a network volume that runs the whole overnight treatment arm chain unattended (answer generation, fine-tuning,
evaluation generation: see overnight_chain/overnight_treatment_arm_chain_step_definitions.py), pull results every 60 seconds, and make
sure the pod is gone at the end. THE LAPTOP JUDGES: while the chain runs, a background thread of this launcher judges the pulled system-prompted
answers with Claude Haiku 5.5 through the paid Anthropic API (ANTHROPIC_API_KEY from the project .env, refused without it; the key never goes to the
pod) and uploads verdict, regeneration-request and judging-complete files to the pod over the launcher's ssh connection. The launcher heartbeat
means "this laptop is connected and its judging is alive": the pod ends the chain (synced, resumable) when it is 20 minutes old, so keep the
laptop awake until judging completes (run this launcher under `caffeinate -i`). A job profile over the persona-check launcher (scripts/log_probability_persona_check/
launch_and_stop_runpod_scoring_pod.py): cost log (with the cost center and whether the assignment was confirmed), hard limits, cost-center
reminders, the on-pod time-limit watchdog, the verified final pull and failure handling are that module's. This file adds:
  - the network volume (found: a data center that currently has an H200; created there, or reused by id) and the pod attached to it;
  - a launcher heartbeat on the pod (written every 60 seconds, not written when the Haiku judging thread has stopped on a fatal error) that tells the on-pod orchestrator whether a person is
    connected: connected, a step that fails twice waits up to 20 minutes for an instruction; disconnected, it does not wait;
  - tolerance of a lost connection: ssh timeouts do not end the launcher or delete the pod; the chain keeps running on the pod;
  - recognition that the pod deleted itself at the end of the chain (a final pull is then impossible; use the retrieval script);
  - the forwarding of instruction files (below) to the pod.

Run from the repository root, with the run folder built by overnight_chain/build_overnight_chain_run_folder_for_launch.py:
    python3 scripts/emergent_misalignment/launch_and_stop_runpod_overnight_treatment_arm_chain_pod.py --run-folder <overnight run folder> --dry-run
    ... without --dry-run                                  # PAID: creates a network volume and a pod; needs the user's go-ahead

Defaults for this mode: --maximum-running-hours 12, --maximum-dollars 65 (the pod is deleted at whichever comes first), --volume-gigabytes 20,
one NVIDIA H200, container disk 200 GB (model weights and the two environments live there, not on the volume).
Options: --network-volume-id <id> --data-center-id <id> (reuse a volume; relaunch after a loss: the chain skips finished steps),
--data-center-id <id> alone (create the volume there, no availability lookup), --skip-availability-check.

Instruction files, created in the run folder while the pod waits for a person (all renamed once forwarded):
    pod_instruction_retry_step_<step name>.txt   run that step again      pod_instruction_skip_step_<step name>.txt   give up on that step
    pod_instruction_retry_scorer.txt / pod_instruction_delete_pod.txt     the shared launcher's: restart the chain process / delete the pod
After the pod deletes itself a final pull is impossible: run retrieve_overnight_chain_outputs_from_network_volume_with_cheapest_cpu_pod.py.
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment.overnight_chain import overnight_treatment_arm_chain_step_definitions as definitions  # noqa: E402
from emergent_misalignment.overnight_chain import estimate_haiku_calls_and_dollars_for_the_overnight_chain_judging as haiku_estimate  # noqa: E402
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_treatment_arm_answers_with_haiku_on_the_laptop_and_write_verdict_regeneration_and_completion_files_for_the_pod as laptop_judging)
import anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log as haiku_api  # noqa: E402
from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as pod_launcher  # noqa: E402
from scripts.shared import runpod_network_volume_and_data_center_availability_calls as volume_calls  # noqa: E402

ORCHESTRATOR_SCRIPT = "scripts/emergent_misalignment/overnight_chain/run_overnight_treatment_arm_chain_on_pod.py"
ORCHESTRATOR_REQUIREMENTS = "scripts/emergent_misalignment/overnight_chain/requirements_for_overnight_chain_orchestrator_on_pod.txt"
ORCHESTRATOR_PROCESS_PATTERN = "[r]un_overnight_treatment_arm_chain_on_pod"
REMOTE_WORKING_DIRECTORY = "/workspace/emergent_alignment_overnight_copy"
DEFAULT_MAXIMUM_RUNNING_HOURS = 12.0
DEFAULT_MAXIMUM_DOLLARS = 65.0
DEFAULT_VOLUME_GIGABYTES = 20
CONTAINER_DISK_GIGABYTES = 200
GPU_TYPE_IDS = ["NVIDIA H200"]
POD_NAME_PREFIX = "overnight-treatment-arm-chain"
PULL_EXCLUDED_PATH_PATTERNS = ["*/checkpoints/*", "*/lora_adapters/*", f"*/instructions_from_launcher/{definitions.HAIKU_JUDGING_INPUT_FOLDER_NAME}/*",
                               "*/final_adapter/*", "*/final_adapter_for_vllm/*", "*/adapter_at_end_of_epoch/*"]  # adapters (about 1 GB each) stay on the volume  # optimizer checkpoints stay on the volume; adapters are pulled from the training outputs
INSTRUCTION_FILE_PATTERN = re.compile(r"^pod_instruction_(retry|skip)_step_([a-z0-9_]+)\.txt$")
PRODUCTION_CONDITIONS_NOTE = (
    "Expected pod time: about 6.5 hours (range 5 to 9) and about 34 dollars (range 26 to 48) at RunPod's recent H200 price (5.29 dollars an hour on 7 November), "
    "plus the volume's storage (monthly). Calculated, not measured. Parts, minutes: installs and weights download 25; phases 1 to 4 about 70 to 110 (6,018 "
    "system-prompted and 1,500 no-system-prompt answers, then up to three regeneration rounds that each wait for the laptop's Haiku verdicts, during which the pod "
    "generates the held-out, Betley and MMLU-Pro base-model answers instead of idling); gates under 1; build and 200-example smoke run 15; full fine-tuning about 75; "
    "held-out, Betley and MMLU-Pro generation about 40 (the base entries are mostly done in the waiting gaps); sycophancy, HarmBench and SAGE-Eval generation "
    "about 70. No judging runs on the pod. The laptop must stay awake and connected until judging completes (a heartbeat older than 20 minutes ends the "
    "chain, resumable). If the training gate closes, no training and no evaluation run.")
HEARTBEAT_FILE_NAME = "launcher_heartbeat_epoch_seconds.txt"
PULL_AND_HEARTBEAT_INTERVAL_SECONDS = 60  # the laptop judges what it pulls, and the pod's regeneration waits on the verdicts
FULL_REPORT_INTERVAL_SECONDS = 180
HAIKU_JUDGING_FOLDER_NAME = "haiku_judging_of_treatment_arm_answers"
PULLED_GENERATION_RUN_FOLDER_RELATIVE_PATH = f"{definitions.OVERNIGHT_OUTPUTS_FOLDER_NAME}/run_folders/treatment_arm_generation"
INPUT_GENERATION_RUN_FOLDER_RELATIVE_PATH = f"{definitions.OVERNIGHT_INPUTS_FOLDER_NAME}/run_folders/treatment_arm_generation"
HAIKU_JUDGING_CYCLE_SECONDS = 30

# Values the session subclass reads: set by main().
LAUNCH_RECORD = {"network_volume": None}


def remote_run_folder():
    return f"{REMOTE_WORKING_DIRECTORY}/run_folder"


class ConnectionLossTolerantTransfer(pod_launcher.SshTarTransfer):
    """ssh problems become a non-zero exit code instead of an exception, so that a laptop that loses its connection (or sleeps) does not turn
    into an 'exception' ending that deletes the pod. Uploads go to the pod without the local overnight_outputs folder: the volume, not the
    laptop, holds the newest outputs."""

    def run_remote_command(self, command, input_text=None, timeout_seconds=None):
        try:
            return super().run_remote_command(command, input_text=input_text, timeout_seconds=timeout_seconds)
        except (subprocess.TimeoutExpired, OSError) as error:
            return 255, f"ssh unreachable: {error!r}"

    def upload_project_files(self):
        paths_to_send = ["scripts/__init__.py"] if (REPOSITORY_ROOT / "scripts/__init__.py").exists() else []
        paths_to_send += ["scripts/shared", *pod_launcher.SCRIPT_FOLDER_NAMES_TO_UPLOAD]
        self.run_remote_command(f"mkdir -p {pod_launcher.REMOTE_WORKING_DIRECTORY} {pod_launcher.REMOTE_RUN_FOLDER}")
        excludes = ["--exclude=__pycache__", "--exclude=*.pyc", "--exclude=.DS_Store", "--no-mac-metadata", "--no-xattrs"]
        self._stream_local_tar_to_remote(["tar", "czf", "-", *excludes, "-C", str(REPOSITORY_ROOT), *paths_to_send],
                                         pod_launcher.REMOTE_WORKING_DIRECTORY)
        items = [path.name for path in sorted(self.run_folder.glob("run_configuration_*.json"))]
        items += [pod_launcher.INPUTS_FOLDER_NAME] if (self.run_folder / pod_launcher.INPUTS_FOLDER_NAME).exists() else []
        self._stream_local_tar_to_remote(["tar", "czf", "-", *excludes, "-C", str(self.run_folder), *items], pod_launcher.REMOTE_RUN_FOLDER)


class OvernightChainPodSession(pod_launcher.PodSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pod_deleted_itself = False
        record = LAUNCH_RECORD.get("network_volume") or {}
        self.cost_log.update(network_volume_id=record.get("id"), network_volume_data_center_id=record.get("dataCenterId"),
                             network_volume_gigabytes=record.get("size"), network_volume_name=record.get("name"),
                             network_volume_estimated_monthly_cost_dollars=record.get("estimated_monthly_cost_dollars"))
        self.heartbeat_path = f"{remote_run_folder()}/{HEARTBEAT_FILE_NAME}"
        self.outputs_path = f"{remote_run_folder()}/{definitions.OVERNIGHT_OUTPUTS_FOLDER_NAME}"
        self.haiku_judging_thread = None
        self.haiku_judging_stop = threading.Event()
        self.haiku_judging_failed = False  # then no heartbeat is written: the pod ends the chain after 20 minutes
        self.haiku_judging_complete = False
        self.last_full_report_epoch_seconds = 0.0

    # -- heartbeat and instruction forwarding -------------------------------------------------------------------------------------------
    def write_heartbeat(self):
        if self.haiku_judging_failed:
            return False
        exit_code, _ = self.transfer.run_remote_command(f"echo {int(self.clock())} > {self.heartbeat_path}", timeout_seconds=60)
        return exit_code == 0

    def forward_instruction_files(self):
        for path in sorted(self.run_folder.glob("pod_instruction_*_step_*.txt")):
            match = INSTRUCTION_FILE_PATTERN.match(path.name)
            if not match:
                continue
            verb, step_name = match.groups()
            folder = f"{self.outputs_path}/instructions_from_launcher"
            exit_code, _ = self.transfer.run_remote_command(f"mkdir -p {folder} && : > {folder}/{verb}_step_{step_name}.txt", timeout_seconds=60)
            if exit_code == 0:
                path.rename(path.with_name(f"{path.name}.forwarded_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))
                self.say(f"Instruction {path.name} forwarded to the pod")

    def report(self, limit_seconds):
        """Called every 60 seconds (heartbeat and instruction forwarding every time); the full progress line every 3 minutes."""
        self.write_heartbeat()
        self.forward_instruction_files()
        if not self.haiku_judging_failed and not self.haiku_judging_complete:
            self.start_haiku_judging_thread()  # does nothing while the thread runs; starts it when attached to a pod already running
        if self.clock() - self.last_full_report_epoch_seconds >= FULL_REPORT_INTERVAL_SECONDS:
            self.last_full_report_epoch_seconds = self.clock()
            super().report(limit_seconds)

    # -- Haiku judging on the laptop -------------------------------------------------------------------------------------------------------
    def upload_haiku_judging_file(self, name, path):
        """Atomic on the pod (written to .part, then moved), so the pod never reads a half-written file."""
        folder = f"{self.outputs_path}/instructions_from_launcher/{definitions.HAIKU_JUDGING_INPUT_FOLDER_NAME}"
        exit_code, text = self.transfer.run_remote_command(
            f"mkdir -p {folder} && cat > {folder}/{name}.part && mv -f {folder}/{name}.part {folder}/{name}",
            input_text=Path(path).read_text(encoding="utf-8"), timeout_seconds=600)
        if exit_code != 0:
            self.say(f"Upload of {name} to the pod failed (retried in the next cycle): {text[:200]}")
        return exit_code == 0

    def start_haiku_judging_thread(self):
        if not HAIKU_JUDGING_SETTINGS["enabled"] or (self.haiku_judging_thread and self.haiku_judging_thread.is_alive()):
            return
        judging_folder = Path(self.run_folder) / HAIKU_JUDGING_FOLDER_NAME
        backend = HAIKU_JUDGING_SETTINGS["backend_factory"](judging_folder, self.say)
        judging = laptop_judging.TreatmentArmHaikuJudging(
            Path(self.run_folder) / PULLED_GENERATION_RUN_FOLDER_RELATIVE_PATH, Path(self.run_folder) / INPUT_GENERATION_RUN_FOLDER_RELATIVE_PATH,
            judging_folder, backend.call_model, log=self.say, backend_progress_line=backend.progress_line)

        def run():
            try:
                self.haiku_judging_complete = laptop_judging.run_judging_loop(
                    judging, self.upload_haiku_judging_file, self.haiku_judging_stop, interval_seconds=HAIKU_JUDGING_CYCLE_SECONDS, log=self.say)
            except BaseException as error:  # noqa: BLE001 - the heartbeat stops, so the pod ends the chain cleanly instead of waiting for ever
                self.haiku_judging_failed = True
                self.say(f"!!! HAIKU JUDGING STOPPED ({type(error).__name__}: {error}). The heartbeat is no longer written: the pod ends the chain "
                         f"(synced, resumable) 20 minutes from now. Fix the cause and relaunch with --network-volume-id and --data-center-id.")

        self.haiku_judging_thread = threading.Thread(target=run, name="haiku-judging-on-the-laptop", daemon=True)
        self.haiku_judging_thread.start()
        self.say("Haiku judging started on this laptop: answers pulled every 60 seconds are judged and the verdict files uploaded to the pod")

    def start_scorer(self):
        self.write_heartbeat()
        hard_limit_seconds = max(600.0, pod_launcher.effective_limit_seconds(
            self.arguments.maximum_running_hours, self.arguments.maximum_dollars, self.hourly_price) - (self.clock() - self.start_epoch_seconds))
        ORCHESTRATOR_SETTINGS.update(seconds_until_hard_limit=int(hard_limit_seconds), hourly_price=self.hourly_price)
        super().start_scorer()
        self.start_haiku_judging_thread()

    # -- the chain ends by itself --------------------------------------------------------------------------------------------------------
    def pod_is_gone(self):
        try:
            self.client.get_pod(self.pod_id)
            return False
        except Exception as error:  # noqa: BLE001 - only a clear 404 means gone
            return "404" in str(error)

    def scorer_exit_code(self):
        exit_code, text = self.transfer.run_remote_command(
            f"cat {self.outputs_path}/chain_summary_and_progress/chain_finished_flag.json 2>/dev/null", timeout_seconds=60)
        if exit_code == 0 and text.strip().startswith("{"):
            try:
                flag = json.loads(text)
                self.say(f"The chain wrote its finished flag: ended because {flag.get('ended_because')}; overall {flag.get('overall_status')}")
            except ValueError:
                self.say("The chain wrote its finished flag")
            return 0
        if exit_code != 0 and self.pod_is_gone():
            self.pod_deleted_itself = True
            self.say("The pod no longer exists (it deleted itself at the end of the chain, or was deleted from the console)")
            return 0
        return None

    def final_pull_with_verification(self):
        self.haiku_judging_stop.set()
        if self.pod_deleted_itself:
            summaries = sorted(self.run_folder.glob(f"{definitions.OVERNIGHT_OUTPUTS_FOLDER_NAME}/chain_summary_and_progress/chain_summary_*.json"))
            self.say("The pod is gone, so no final pull is possible. "
                     + (f"The last periodic pull holds {summaries[-1].name}. " if summaries else "No chain summary reached the laptop. ")
                     + "Everything else on the network volume is retrieved with "
                       "retrieve_overnight_chain_outputs_from_network_volume_with_cheapest_cpu_pod.py")
            self.cost_log.update(final_pull_verified=False, pod_deleted_itself=True,
                                 final_pull_note="pod deleted itself; retrieve from the network volume with the retrieval script")
            self.final_pull_succeeded = bool(summaries)
            return self.final_pull_succeeded
        self.write_heartbeat()
        verified = super().final_pull_with_verification()
        self.transfer.run_remote_command(f"touch {self.outputs_path}/chain_summary_and_progress/launcher_final_pull_complete.txt", timeout_seconds=60)
        return verified

    def warn_and_wait_for_cost_center_assignment_before_deletion(self, reason):
        if self.pod_deleted_itself:
            self.record_confirmation_if_given()
            return
        super().warn_and_wait_for_cost_center_assignment_before_deletion(reason)


ORCHESTRATOR_SETTINGS = {"seconds_until_hard_limit": None, "hourly_price": None}


def default_haiku_backend_factory(judging_folder, say):
    """The paid Anthropic API backend; the key is read from the project .env (SystemExit without it)."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return haiku_api.AnthropicApiJudgeBackend(
        api_key=haiku_api.read_anthropic_api_key_from_project_env_file(), usage_log_path=Path(judging_folder) / f"anthropic_api_usage_log_{stamp}.jsonl", log=say)


HAIKU_JUDGING_SETTINGS = {"enabled": True, "backend_factory": default_haiku_backend_factory}  # tests and the rehearsal replace the factory


def build_orchestrator_command_line(arguments):
    command = ["python3", ORCHESTRATOR_SCRIPT, "--working-directory", REMOTE_WORKING_DIRECTORY,
               "--seconds-until-hard-limit", str(ORCHESTRATOR_SETTINGS["seconds_until_hard_limit"] or int(arguments.maximum_running_hours * 3600))]
    if ORCHESTRATOR_SETTINGS["hourly_price"]:
        command += ["--pod-hourly-price-dollars", str(ORCHESTRATOR_SETTINGS["hourly_price"])]
    return command


def build_overnight_pod_creation_request(arguments, public_key_text):
    return {
        "name": f"{POD_NAME_PREFIX}-{Path(arguments.run_folder).name}"[:191], "imageName": arguments.image_name, "cloudType": "SECURE",
        "computeType": "GPU", "gpuCount": 1, "gpuTypeIds": arguments.gpu_type_ids, "gpuTypePriority": "custom",
        "dataCenterIds": [OVERNIGHT_LAUNCH_SETTINGS["data_center_id"] or "<data center chosen at launch>"], "dataCenterPriority": "custom",
        "networkVolumeId": OVERNIGHT_LAUNCH_SETTINGS["network_volume_id"] or "<network volume created at launch>",
        "volumeMountPath": "/workspace", "containerDiskInGb": arguments.container_disk_gigabytes, "ports": ["22/tcp"],
        "supportPublicIp": True, "interruptible": False,
        "env": {"PUBLIC_KEY": public_key_text, "SSH_PUBLIC_KEY": public_key_text, "HF_HOME": definitions.HUGGING_FACE_CACHE}}


OVERNIGHT_LAUNCH_SETTINGS = {"data_center_id": None, "network_volume_id": None, "volume_gigabytes": DEFAULT_VOLUME_GIGABYTES,
                             "skip_availability_check": False}


def apply_job_profile():
    pod_launcher.REMOTE_WORKING_DIRECTORY = REMOTE_WORKING_DIRECTORY
    pod_launcher.REMOTE_RUN_FOLDER = remote_run_folder()
    pod_launcher.INPUTS_FOLDER_NAME = definitions.OVERNIGHT_INPUTS_FOLDER_NAME
    pod_launcher.OUTPUTS_FOLDER_NAME = definitions.OVERNIGHT_OUTPUTS_FOLDER_NAME
    pod_launcher.POD_NAME_PREFIX = POD_NAME_PREFIX
    pod_launcher.SCRIPT_FOLDER_NAME = "scripts/emergent_misalignment"
    pod_launcher.SCRIPT_FOLDER_NAMES_TO_UPLOAD = ["scripts/emergent_misalignment"]
    pod_launcher.SCORER_SCRIPT_RELATIVE_PATH = ORCHESTRATOR_SCRIPT
    pod_launcher.SCORER_PROCESS_PATTERN = ORCHESTRATOR_PROCESS_PATTERN
    pod_launcher.REQUIREMENTS_FILE_RELATIVE_PATH = ORCHESTRATOR_REQUIREMENTS
    pod_launcher.PROGRESS_STATUS_SHELL_TEMPLATE = (
        "ls {remote_run_folder}/overnight_outputs/step_completion_markers 2>/dev/null | wc -l; "
        "ls -t {remote_run_folder}/overnight_outputs/chain_summary_and_progress/chain_progress_log_*.txt 2>/dev/null | head -n 1 | xargs -r tail -n 1")
    pod_launcher.PROGRESS_STATUS_UNIT_NAME = "step completion markers on the volume"
    pod_launcher.MINIMUM_LOCAL_FREE_DISK_BYTES = 20 * 10 ** 9
    pod_launcher.DEFAULT_CONTAINER_DISK_GIGABYTES = CONTAINER_DISK_GIGABYTES
    pod_launcher.PULL_EXCLUDED_PATH_PATTERNS = PULL_EXCLUDED_PATH_PATTERNS
    pod_launcher.PROGRESS_REPORT_INTERVAL_SECONDS = PULL_AND_HEARTBEAT_INTERVAL_SECONDS
    inputs, outputs = definitions.OVERNIGHT_INPUTS_FOLDER_NAME, definitions.OVERNIGHT_OUTPUTS_FOLDER_NAME
    excluded = " ".join(f"-not -path '{pattern}'" for pattern in PULL_EXCLUDED_PATH_PATTERNS)
    pod_launcher.FINAL_COPY_FILE_LIST_SHELL = (
        f"find run_folder -type f -not -path 'run_folder/{inputs}/*' {excluded} -not -name '.last_pull_marker*' -not -name '{HEARTBEAT_FILE_NAME}'; "
        "for log_file in watchdog_log.txt pod_setup_*.txt; do [ -f \"$log_file\" ] && echo \"$log_file\"; done; true")
    pod_launcher.build_scorer_command_line = build_orchestrator_command_line
    pod_launcher.build_pod_creation_request = build_overnight_pod_creation_request
    pod_launcher.PodSession = OvernightChainPodSession
    pod_launcher.SshTarTransfer = ConnectionLossTolerantTransfer
    pod_launcher.RunPodRestClient = volume_calls.RunPodClientWithNetworkVolumes
    pod_launcher.print_dry_run = print_overnight_dry_run


def estimate_lines(maximum_dollars, maximum_hours):
    return [PRODUCTION_CONDITIONS_NOTE,
            f"Network volume: {OVERNIGHT_LAUNCH_SETTINGS['volume_gigabytes']} GB, about "
            f"{volume_calls.monthly_storage_cost_dollars(OVERNIGHT_LAUNCH_SETTINGS['volume_gigabytes']):.2f} dollars per month until deleted "
            f"(0.07 dollars per GB per month). Deleting it is a separate, manual step that this project's code never runs.",
            f"Hard limits: {maximum_hours:g} hours or {maximum_dollars:g} dollars, whichever comes first; the chain stops starting steps 15 minutes "
            "before the limit, stops a running step 10 minutes before it, writes its summary and deletes the pod; the on-pod watchdog deletes "
            "the pod 10 minutes after the limit if everything else failed."]


def print_overnight_dry_run(arguments, secrets):
    sys.stdout.flush()
    print("OVERNIGHT CHAIN DRY RUN: no RunPod calls are made.")
    volume_name = f"overnight-treatment-arm-chain-volume-{Path(arguments.run_folder).name}"
    if OVERNIGHT_LAUNCH_SETTINGS["network_volume_id"]:
        print(f"Network volume: reuse {OVERNIGHT_LAUNCH_SETTINGS['network_volume_id']} in data center {OVERNIGHT_LAUNCH_SETTINGS['data_center_id']}")
    else:
        data_center = OVERNIGHT_LAUNCH_SETTINGS["data_center_id"] or "<a data center that currently has an H200 in stock>"
        print(f"1. GET {volume_calls.RUNPOD_CATALOG_BASE_URL}/datacenters?include=GPU_AVAILABILITY  (skipped with --data-center-id or --skip-availability-check; "
              f"endpoint unverified)\n2. POST {pod_launcher.RUNPOD_REST_BASE_URL}/networkvolumes\n"
              + json.dumps(volume_calls.build_network_volume_creation_request(volume_name, OVERNIGHT_LAUNCH_SETTINGS["volume_gigabytes"], data_center), indent=2))
    request = build_overnight_pod_creation_request(arguments, pod_launcher.read_public_key_text(arguments.ssh_private_key_path, False))
    request["env"] = {key: ("<public key>" if "KEY" in key else value) for key, value in request["env"].items()}
    print(f"3. POST {pod_launcher.RUNPOD_REST_BASE_URL}/pods\n{json.dumps(request, indent=2)}")
    print(f"RUNPOD_API_KEY found in .env: {bool(secrets['RUNPOD_API_KEY'])}; HF_TOKEN found in .env: {bool(secrets['HF_TOKEN'])}; "
          f"RUNPOD_COST_CENTER_NAME found in .env: {bool(secrets.get('RUNPOD_COST_CENTER_NAME'))} (launch is refused without the key and the cost center; "
          f"assign the pod by hand at {pod_launcher.COST_CENTER_CONSOLE_URL})")
    paths = definitions.ChainPaths(REMOTE_WORKING_DIRECTORY, remote_run_folder())
    configuration_files = sorted(Path(arguments.run_folder).glob("run_configuration_overnight_chain_*.json"))
    configuration = json.loads(configuration_files[-1].read_text(encoding="utf-8")) if configuration_files else {}
    print("Chain of steps on the pod, in order (the chain runs even if this launcher disconnects):")
    for number, step in enumerate(definitions.build_chain_steps(paths, configuration), 1):
        print(f"  {number:2d}. {step.name}: {step.description}"
              + (f"  [after {', '.join(step.depends_on)}]" if step.depends_on else "")
              + (f"  [also after {', '.join(step.run_after)}, whatever its outcome]" if step.run_after else ""))
    print(f"SAGE-Eval dataset supplied at build: {bool(configuration.get('sage_eval_dataset_supplied'))} (otherwise the step records it as skipped)")
    print("Commands on the pod:")
    print(f"  upload: scripts/shared, scripts/emergent_misalignment, run_configuration_overnight_chain_*.json and {pod_launcher.INPUTS_FOLDER_NAME} (the local "
          f"{pod_launcher.OUTPUTS_FOLDER_NAME} folder is never uploaded)")
    print(f"  pip install -r {ORCHESTRATOR_REQUIREMENTS} (standard library only; the chain creates its own two virtual environments on the container disk)")
    print(f"  watchdog: {pod_launcher.build_self_termination_watchdog_command(arguments.maximum_running_hours * 3600 + pod_launcher.WATCHDOG_GRACE_SECONDS)}")
    ORCHESTRATOR_SETTINGS.update(seconds_until_hard_limit=int(arguments.maximum_running_hours * 3600), hourly_price=None)
    print(f"  chain:    {pod_launcher.build_scorer_launch_shell_command(arguments)}")
    print(f"  the launcher writes {remote_run_folder()}/{HEARTBEAT_FILE_NAME} at every report (about every {pod_launcher.PROGRESS_REPORT_INTERVAL_SECONDS} seconds); "
          "the chain's failure policy treats a heartbeat older than 10 minutes as 'connection lost', and the generation phases end the chain "
          "(synced to the volume, pod deleted, resumable on relaunch) when it is older than 20 minutes while they wait for the laptop's verdicts")
    print(f"  pull every {pod_launcher.PROGRESS_REPORT_INTERVAL_SECONDS} seconds into {arguments.run_folder}/{pod_launcher.OUTPUTS_FOLDER_NAME} "
          f"(excluding {', '.join(PULL_EXCLUDED_PATH_PATTERNS)})")
    print("  at the end the chain writes its summary and finished flag, syncs, waits up to 15 minutes for this launcher's verified final pull (only if the heartbeat "
          "is fresh), then deletes the pod from the inside; if the pod is already gone when the launcher looks, run the retrieval script")
    for line in estimate_lines(arguments.maximum_dollars, arguments.maximum_running_hours):
        print(line)
    print_haiku_judging_description_and_estimate(arguments)


def print_haiku_judging_description_and_estimate(arguments):
    key_found = bool(haiku_api_key_if_present())
    print("JUDGING ON THIS LAPTOP (nothing is judged on the pod, and no API key goes to the pod):")
    print(f"  ANTHROPIC_API_KEY found in the project .env: {key_found} (the launch is refused without it; the operating system's environment is never read)")
    print(f"  While the chain runs, a background thread of this launcher, every {HAIKU_JUDGING_CYCLE_SECONDS} seconds: reads the system-prompted answers "
          f"pulled (every {PULL_AND_HEARTBEAT_INTERVAL_SECONDS} seconds) into {arguments.run_folder}/{PULLED_GENERATION_RUN_FOLDER_RELATIVE_PATH}, judges the "
          "new ones with Claude Haiku 5.5 (the existing welfare and refusal judge on the final answer, then the existing choice-of-option judge on answers with "
          "refusal complied and raised_welfare_cost no; the existing pass rule), writes verdict records in the Haiku judges' formats under "
          f"{arguments.run_folder}/{HAIKU_JUDGING_FOLDER_NAME}/, and uploads three files to {remote_run_folder()}/{definitions.OVERNIGHT_OUTPUTS_FOLDER_NAME}/"
          f"instructions_from_launcher/{definitions.HAIKU_JUDGING_INPUT_FOLDER_NAME}/: the verdicts, the regeneration requests (prompts whose answers all failed and "
          "have fewer than 4 attempts) and, when every prompt has a passing answer or has used 4 attempts, the judging-complete file listing the kept answer per prompt.")
    print("  The pod regenerates what the requests list, builds the kept answers only after the judging-complete file arrives, and in the gaps generates the "
          "held-out, Betley and MMLU-Pro base-model answers (resumable, yielding to regeneration). If the judging thread dies the heartbeat stops and the pod "
          "ends the chain (resumable) 20 minutes later.")
    print("  KEEP THIS LAPTOP AWAKE until judging completes: launch under `caffeinate -i python3 ...`. A laptop asleep for 20 minutes ends the chain; "
          "relaunch with --network-volume-id and --data-center-id to resume.")
    try:
        for line in haiku_estimate.estimate(arguments.run_folder)["lines"]:
            print(line)
    except (OSError, KeyError, ValueError, StopIteration) as error:
        print(f"  (Haiku estimate unavailable for this run folder: {error!r}). Throughput depends on the Anthropic account's rate-limit tier.")


def haiku_api_key_if_present():
    from dotenv import dotenv_values
    path = REPOSITORY_ROOT / ".env"
    return (dotenv_values(path).get(haiku_api.ANTHROPIC_API_KEY_NAME) or "") if path.exists() else ""


def extra_arguments(remaining):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--network-volume-id", default=None)
    parser.add_argument("--data-center-id", default=None)
    parser.add_argument("--volume-gigabytes", type=int, default=DEFAULT_VOLUME_GIGABYTES)
    parser.add_argument("--skip-availability-check", action="store_true")
    parser.add_argument("--minimum-local-free-disk-gigabytes", type=float, default=20.0,
                        help="pulls wait while the laptop has less free disk than this (default 20; the adapters are the only large files)")
    return parser.parse_known_args(remaining)


def create_or_reuse_network_volume(arguments, secrets, run_folder, client=None):
    """Returns the volume record {id, name, size, dataCenterId, estimated_monthly_cost_dollars} and fills OVERNIGHT_LAUNCH_SETTINGS."""
    settings = OVERNIGHT_LAUNCH_SETTINGS
    client = client or volume_calls.RunPodClientWithNetworkVolumes(secrets["RUNPOD_API_KEY"])
    if settings["network_volume_id"]:
        if not settings["data_center_id"]:
            record = client.get_network_volume(settings["network_volume_id"])
            settings["data_center_id"] = record["dataCenterId"]
        else:
            record = {"id": settings["network_volume_id"], "dataCenterId": settings["data_center_id"], "name": "existing", "size": None}
    else:
        if not settings["data_center_id"]:
            if settings["skip_availability_check"]:
                raise SystemExit("--skip-availability-check needs --data-center-id")
            try:
                candidates = volume_calls.data_centers_with_gpu_in_stock(client.list_data_centers_with_gpu_availability(), "H200")
            except Exception as error:  # noqa: BLE001
                raise SystemExit(f"Could not look up data centers with an H200 in stock ({error}). Nothing was created. Check RunPod's console for a data "
                                 f"center with H200 availability and pass --data-center-id.")
            if not candidates:
                raise SystemExit("No data center currently reports H200 availability. Nothing was created. Try again later or pass --data-center-id.")
            settings["data_center_id"] = candidates[0]
            print(f"Data centers with an H200 in stock, most available first: {candidates}; using {candidates[0]}", flush=True)
        name = f"overnight-treatment-arm-chain-volume-{Path(run_folder).name}"
        record = dict(client.create_network_volume(name, settings["volume_gigabytes"], settings["data_center_id"]))
        settings["network_volume_id"] = record["id"]
    record["estimated_monthly_cost_dollars"] = round(volume_calls.monthly_storage_cost_dollars(record.get("size") or settings["volume_gigabytes"]), 2)
    record["delete_command_not_run"] = volume_calls.delete_network_volume_command_text(record["id"])
    (Path(run_folder) / f"network_volume_record_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    LAUNCH_RECORD["network_volume"] = record
    print(f"Network volume {record['id']} in {settings['data_center_id']} ({record.get('size')} GB, about "
          f"{record['estimated_monthly_cost_dollars']} dollars per month). It stays after the pod is gone: to delete it later, {record['delete_command_not_run']}", flush=True)
    return record


def run_folder_holds_sage_text_outside_data(run_folder):
    configuration_files = sorted(Path(run_folder).glob("run_configuration_overnight_chain_*.json"))
    if not configuration_files or not json.loads(configuration_files[-1].read_text(encoding="utf-8")).get("sage_eval_dataset_supplied"):
        return False
    return (REPOSITORY_ROOT / "data").resolve() not in [Path(run_folder).resolve(), *Path(run_folder).resolve().parents]


def main(argv=None):
    remaining = list(sys.argv[1:] if argv is None else argv)
    extra, remaining = extra_arguments(remaining)
    OVERNIGHT_LAUNCH_SETTINGS.update(network_volume_id=extra.network_volume_id, data_center_id=extra.data_center_id,
                                     volume_gigabytes=extra.volume_gigabytes, skip_availability_check=extra.skip_availability_check)
    if "--maximum-running-hours" not in remaining:
        remaining += ["--maximum-running-hours", str(DEFAULT_MAXIMUM_RUNNING_HOURS)]
    if "--maximum-dollars" not in remaining:
        remaining += ["--maximum-dollars", str(DEFAULT_MAXIMUM_DOLLARS)]
    if "--gpu-type-ids" not in remaining:
        remaining += ["--gpu-type-ids", *GPU_TYPE_IDS]
    apply_job_profile()
    pod_launcher.MINIMUM_LOCAL_FREE_DISK_BYTES = int(extra.minimum_local_free_disk_gigabytes * 10 ** 9)
    arguments = pod_launcher.parse_arguments(remaining)
    if run_folder_holds_sage_text_outside_data(arguments.run_folder):
        raise SystemExit("This run folder holds SAGE-Eval prompts but is not under the repository's data/ folder: SAGE-Eval content must never leave data/.")
    if not arguments.dry_run:
        secrets = pod_launcher.read_secrets_from_environment_file()
        if not secrets["RUNPOD_API_KEY"] or not secrets.get("RUNPOD_COST_CENTER_NAME"):
            raise SystemExit("RUNPOD_API_KEY and RUNPOD_COST_CENTER_NAME must both be in the project .env; refusing before creating anything.")
        if HAIKU_JUDGING_SETTINGS["enabled"] and HAIKU_JUDGING_SETTINGS["backend_factory"] is default_haiku_backend_factory:
            haiku_api.read_anthropic_api_key_from_project_env_file()  # SystemExit before anything is created when the key is absent
        if not (Path(arguments.run_folder) / pod_launcher.INPUTS_FOLDER_NAME).is_dir():
            raise SystemExit(f"{arguments.run_folder}/{pod_launcher.INPUTS_FOLDER_NAME} does not exist: build the run folder first.")
        pod_launcher.read_public_key_text(arguments.ssh_private_key_path, required=True)
        create_or_reuse_network_volume(arguments, secrets, arguments.run_folder)
    return pod_launcher.run_launcher(arguments)


if __name__ == "__main__":
    sys.exit(main())
