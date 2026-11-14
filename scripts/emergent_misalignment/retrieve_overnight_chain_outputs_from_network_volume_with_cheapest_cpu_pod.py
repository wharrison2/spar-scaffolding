"""Copies everything the overnight chain left on its network volume to the laptop, using the cheapest CPU pod attached to that volume, then
deletes the CPU pod. For when the chain's GPU pod deleted itself (so no final pull was possible), or the launcher was lost.

    python3 scripts/emergent_misalignment/retrieve_overnight_chain_outputs_from_network_volume_with_cheapest_cpu_pod.py \
        --network-volume-id <id> --run-folder <overnight run folder> --dry-run          # free: prints the requests and limits
    ... without --dry-run                                                                # PAID (a CPU pod for a few minutes): needs the user's go-ahead

Steps: read the volume's data center (GET /networkvolumes/<id>); create a CPU pod there with the volume mounted at /workspace
(computeType CPU, cpuFlavorIds in cheapest-first order, 2 vCPUs); wait for ssh; pull everything the launcher's final pull would have pulled
(the chain's outputs, logs and summary, without the optimizer checkpoints and the adapter copies inside the evaluation run folders) into
the run folder, verify file list and sizes; delete the pod (always: normal end, error, Ctrl-C, limits). Then PRINTS the command that
deletes the volume. It does not run it: delete the volume only after checking the local copy.

Hard limits of its own: 1 hour and 2 dollars (--maximum-running-hours, --maximum-dollars); the pod is deleted when either is reached. The
RunPod key is read only from the project .env. Unverified (not called while written): that a CPU pod accepts networkVolumeId and these
flavors in the volume's data center, and the cheapest flavor's price; the reported costPerHr is checked against --maximum-hourly-price.
Optional: --pull-optimizer-checkpoints (adds them to the copy: several GB).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import launch_and_stop_runpod_overnight_treatment_arm_chain_pod as overnight_launcher  # noqa: E402
from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as pod_launcher  # noqa: E402
from scripts.shared import runpod_network_volume_and_data_center_availability_calls as volume_calls  # noqa: E402

CPU_FLAVOR_IDS_CHEAPEST_FIRST = ["cpu3c", "cpu3g", "cpu5c", "cpu5g"]
CPU_VCPU_COUNT = 2
DEFAULT_MAXIMUM_RUNNING_HOURS = 1.0
DEFAULT_MAXIMUM_DOLLARS = 2.0
DEFAULT_MAXIMUM_HOURLY_PRICE = 1.0
POD_NAME_PREFIX = "overnight-chain-output-retrieval"
SSH_READY_TIMEOUT_SECONDS = 900
PULL_BUDGET_SECONDS = 3000


def build_cpu_pod_creation_request(arguments, public_key_text, data_center_id):
    return {"name": f"{POD_NAME_PREFIX}-{arguments.network_volume_id}"[:191], "imageName": arguments.image_name, "cloudType": "SECURE",
            "computeType": "CPU", "cpuFlavorIds": arguments.cpu_flavor_ids, "cpuFlavorPriority": "custom", "vcpuCount": CPU_VCPU_COUNT,
            "dataCenterIds": [data_center_id], "dataCenterPriority": "custom", "networkVolumeId": arguments.network_volume_id,
            "volumeMountPath": "/workspace", "containerDiskInGb": 20, "ports": ["22/tcp"], "supportPublicIp": True,
            "env": {"PUBLIC_KEY": public_key_text, "SSH_PUBLIC_KEY": public_key_text}}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--network-volume-id", required=True)
    parser.add_argument("--run-folder", required=True, help="the local overnight run folder the files are copied into")
    parser.add_argument("--maximum-running-hours", type=float, default=DEFAULT_MAXIMUM_RUNNING_HOURS)
    parser.add_argument("--maximum-dollars", type=float, default=DEFAULT_MAXIMUM_DOLLARS)
    parser.add_argument("--maximum-hourly-price", type=float, default=DEFAULT_MAXIMUM_HOURLY_PRICE,
                        help="the pod is deleted unmade if RunPod reports a higher costPerHr")
    parser.add_argument("--cpu-flavor-ids", nargs="+", default=CPU_FLAVOR_IDS_CHEAPEST_FIRST)
    parser.add_argument("--image-name", default=pod_launcher.DEFAULT_POD_IMAGE_NAME)
    parser.add_argument("--ssh-private-key-path", default=str(Path.home() / ".ssh" / "id_ed25519"))
    parser.add_argument("--pull-optimizer-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def retrieve(arguments, client, transfer, secrets, clock=time.time, sleep=time.sleep, print_function=print):
    """Returns 0 when everything arrived and was verified, 1 otherwise. The pod is deleted in every case once it exists."""
    run_folder = Path(arguments.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = run_folder / f"retrieval_pod_cost_log_{stamp}.json"
    record = {"network_volume_id": arguments.network_volume_id, "maximum_running_hours": arguments.maximum_running_hours,
              "maximum_dollars": arguments.maximum_dollars, "cost_center_name": secrets.get("RUNPOD_COST_CENTER_NAME"),
              "cost_center_assignment_confirmed_by_user": False, "pod_id": None}

    def save(**fields):
        record.update(fields)
        log_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def say(message):
        print_function(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    save()
    volume = client.get_network_volume(arguments.network_volume_id)
    data_center_id = volume["dataCenterId"]
    say(f"Volume {arguments.network_volume_id}: {volume.get('size')} GB in {data_center_id}")
    pod_id, start, exit_code, deleted = None, clock(), 1, False
    try:
        public_key = pod_launcher.read_public_key_text(arguments.ssh_private_key_path, required=getattr(client, "requires_key", True))
        pod = client.create_pod(build_cpu_pod_creation_request(arguments, public_key, data_center_id))
        pod_id, start = pod["id"], clock()
        price = pod_launcher.pod_cost_per_hour(pod)
        save(pod_id=pod_id, hourly_price_dollars=price, start_time=pod_launcher.iso_from_epoch(start))
        say(f"Created CPU pod {pod_id} at {price:.3f} dollars per hour. ASSIGN it to cost center '{secrets.get('RUNPOD_COST_CENTER_NAME')}' at "
            f"{pod_launcher.COST_CENTER_CONSOLE_URL} (the run does not wait for you)")
        if price > arguments.maximum_hourly_price:
            raise RuntimeError(f"RunPod reports {price} dollars per hour, above --maximum-hourly-price {arguments.maximum_hourly_price}")
        limit_seconds = pod_launcher.effective_limit_seconds(arguments.maximum_running_hours, arguments.maximum_dollars, price)
        transfer.wait_until_ready(pod, clock() + min(SSH_READY_TIMEOUT_SECONDS, limit_seconds), lambda: client.get_pod(pod_id), sleep)
        say("ssh ready; pulling everything and verifying")
        logs_folder = f"retrieval_pod_{pod_id}_logs_{stamp}"
        files = transfer.pull_final_copy_and_verify(logs_folder, min(PULL_BUDGET_SECONDS, max(60.0, limit_seconds - (clock() - start))))
        say(f"Verified {files} files against the pod's file list and sizes")
        exit_code = 0
    except KeyboardInterrupt:
        say("Interrupted; deleting the CPU pod")
    except Exception as error:  # noqa: BLE001
        say(f"Error: {error}")
        save(error=str(error))
    finally:
        if pod_id:
            for attempt in range(1, pod_launcher.TERMINATION_ATTEMPTS + 1):
                try:
                    client.delete_pod(pod_id)
                    deleted = True
                    break
                except Exception as error:  # noqa: BLE001
                    say(f"Deletion attempt {attempt} failed: {error}")
                    sleep(5)
            confirmed = (run_folder / pod_launcher.COST_CENTER_CONFIRMATION_FILE_NAME).exists()
            save(stop_time=pod_launcher.iso_from_epoch(clock()), pod_terminated=deleted, cost_center_assignment_confirmed_by_user=confirmed,
                 total_cost_dollars=round(pod_launcher.cost_so_far_dollars(record.get("hourly_price_dollars") or 0.0, start, clock()), 4),
                 termination_reason="retrieved_and_verified" if exit_code == 0 else "failed_or_interrupted")
            say(f"CPU pod {pod_id} {'deleted' if deleted else 'COULD NOT BE DELETED: delete it in the console now'}")
    say("When the local copy has been checked, delete the network volume (NOT run by this script): "
        + volume_calls.delete_network_volume_command_text(arguments.network_volume_id))
    return exit_code if deleted or not pod_id else 1


def print_dry_run(arguments, secrets):
    request = build_cpu_pod_creation_request(arguments, pod_launcher.read_public_key_text(arguments.ssh_private_key_path, False),
                                             "<the volume's data center, from GET /networkvolumes/<id>>")
    request["env"] = {key: "<public key>" for key in request["env"]}
    print("DRY RUN: no RunPod calls are made.")
    print(f"RUNPOD_API_KEY in .env: {bool(secrets['RUNPOD_API_KEY'])}; RUNPOD_COST_CENTER_NAME in .env: {bool(secrets.get('RUNPOD_COST_CENTER_NAME'))}")
    print(f"GET {pod_launcher.RUNPOD_REST_BASE_URL}/networkvolumes/{arguments.network_volume_id}\nPOST {pod_launcher.RUNPOD_REST_BASE_URL}/pods\n{json.dumps(request, indent=2)}")
    print(f"Then: ssh in, tar everything in {overnight_launcher.remote_run_folder()} except {pod_launcher.INPUTS_FOLDER_NAME}"
          f"{'' if arguments.pull_optimizer_checkpoints else ', optimizer checkpoints and adapter copies'}, into {arguments.run_folder}; verify file list and sizes; "
          "delete the pod; print (not run) the volume deletion command.")
    print(f"Hard limits: {arguments.maximum_running_hours:g} hours or {arguments.maximum_dollars:g} dollars; the pod is deleted unmade above "
          f"{arguments.maximum_hourly_price:g} dollars per hour. A CPU pod for the pull should cost cents.")


def main(argv=None):
    arguments = parse_arguments(argv)
    overnight_launcher.apply_job_profile()
    if arguments.pull_optimizer_checkpoints:
        pod_launcher.FINAL_COPY_FILE_LIST_SHELL = pod_launcher.FINAL_COPY_FILE_LIST_SHELL.replace("-not -path '*/checkpoints/*' ", "")
    secrets = pod_launcher.read_secrets_from_environment_file()
    if arguments.dry_run:
        print_dry_run(arguments, secrets)
        return 0
    if not secrets["RUNPOD_API_KEY"] or not secrets.get("RUNPOD_COST_CENTER_NAME"):
        raise SystemExit("RUNPOD_API_KEY and RUNPOD_COST_CENTER_NAME must both be in the project .env")
    client = volume_calls.RunPodClientWithNetworkVolumes(secrets["RUNPOD_API_KEY"])
    transfer = overnight_launcher.ConnectionLossTolerantTransfer(arguments.ssh_private_key_path, arguments.run_folder)
    return retrieve(arguments, client, transfer, secrets)


if __name__ == "__main__":
    sys.exit(main())
