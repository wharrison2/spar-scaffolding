"""Print one line saying whether a RunPod pod is busy: GPU use and memory, and the latest line of a log file on the pod.

Usage: python3 scripts/shared/check_runpod_pod_is_not_idle_gpu_use_and_latest_log_line.py --pod-id <id> --remote-log-path <path on pod>
The RunPod key comes from the project .env only. Read-only: it never changes or deletes the pod.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as pod_launcher  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--remote-log-path", required=True)
    parser.add_argument("--ssh-private-key-path", default=str(Path.home() / ".ssh" / "id_ed25519"))
    arguments = parser.parse_args()
    now = datetime.now().strftime("%H:%M")
    client = pod_launcher.RunPodRestClient(pod_launcher.read_secrets_from_environment_file()["RUNPOD_API_KEY"])
    try:
        pod = client.get_pod(arguments.pod_id)
    except Exception as error:  # a deleted pod or an API failure is reported, not raised
        print(f"[{now}] POD NOT FOUND OR API ERROR: {error}")
        return
    host, ssh_port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
    if not host or not ssh_port:
        print(f"[{now}] pod status {pod.get('desiredStatus')}, no ssh address")
        return
    remote_command = ("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; "
                      f"tail -c 2000 {arguments.remote_log_path} | grep -v '^$' | tail -1")
    ssh = ["ssh", "-i", arguments.ssh_private_key_path, "-p", str(ssh_port), "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", f"root@{host}", remote_command]
    try:
        output = subprocess.run(ssh, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    except subprocess.TimeoutExpired:
        print(f"[{now}] SSH TIMED OUT")
        return
    gpu = output[0] if output else "unknown"
    utilization = int(gpu.split("%")[0]) if gpu[:1].isdigit() else -1
    label = "IDLE GPU" if 0 <= utilization < 5 else "busy"
    print(f"[{now}] {label}: GPU {gpu} | latest log line: {output[-1][:300] if len(output) > 1 else 'none'}")


if __name__ == "__main__":
    main()
