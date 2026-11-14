"""Shell commands that delete a RunPod pod from the inside, using the pod's own RUNPOD_POD_ID and pod-scoped RUNPOD_API_KEY
(RunPod puts both in the container's environment). Standard library only, so that code running on a pod (the overnight chain
orchestrator) and the launchers can both use them.

 - build_delete_this_pod_shell_command(): tries runpodctl (both command spellings), then the REST API DELETE /pods/<id>.
 - build_delayed_self_deletion_watchdog_command(): sleeps, then deletes (the launchers' time-limit watchdog).
 - build_dead_man_self_deletion_watchdog_command(): deletes the pod when the chain's orchestrator process has been absent for
   a grace period without having written its finished flag, so that a crashed orchestrator cannot leave an idle pod billing
   until the time limit. A person reconnected through the launcher has the launcher's failure wait (20 minutes) to restart it.
"""

import shlex

RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1"


def build_delete_this_pod_shell_command(rest_base_url=RUNPOD_REST_BASE_URL):
    return (
        "export $(tr '\\0' '\\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs); "
        "runpodctl pod delete \"$RUNPOD_POD_ID\" || runpodctl remove pod \"$RUNPOD_POD_ID\" || "
        "curl -s -X DELETE -H \"Authorization: Bearer $RUNPOD_API_KEY\" "
        f"{rest_base_url}/pods/\"$RUNPOD_POD_ID\""
    )


def build_delayed_self_deletion_watchdog_command(seconds_until_deletion, log_file_path, delete_command=None):
    inner = f"sleep {int(seconds_until_deletion)}; {delete_command or build_delete_this_pod_shell_command()}"
    return f"nohup bash -c {shlex.quote(inner)} > {log_file_path} 2>&1 &"


def build_dead_man_watchdog_script_text(orchestrator_process_pattern, finished_flag_file_path, grace_seconds,
                                        check_interval_seconds, delete_command=None):
    """Bash text of the dead-man loop. The pattern should use the bracket trick ([r]un_...) so pgrep does not match itself."""
    return (
        f"absent_since=\n"
        f"while true; do\n"
        f"  if [ -f {shlex.quote(str(finished_flag_file_path))} ]; then exit 0; fi\n"
        f"  if pgrep -f {shlex.quote(orchestrator_process_pattern)} > /dev/null; then absent_since=; "
        f"  else [ -z \"$absent_since\" ] && absent_since=$(date +%s); fi\n"
        f"  if [ -n \"$absent_since\" ] && [ $(( $(date +%s) - absent_since )) -ge {int(grace_seconds)} ]; then\n"
        f"    echo \"orchestrator absent for {int(grace_seconds)} seconds without a finished flag: deleting the pod\"\n"
        f"    {delete_command or build_delete_this_pod_shell_command()}\n"
        f"    exit 0\n"
        f"  fi\n"
        f"  sleep {check_interval_seconds}\n"
        f"done\n")


def build_dead_man_self_deletion_watchdog_command(orchestrator_process_pattern, finished_flag_file_path, log_file_path,
                                                   grace_seconds=1800, check_interval_seconds=60, delete_command=None):
    script_text = build_dead_man_watchdog_script_text(orchestrator_process_pattern, finished_flag_file_path, grace_seconds,
                                                      check_interval_seconds, delete_command)
    return f"nohup bash -c {shlex.quote(script_text)} > {log_file_path} 2>&1 &"
