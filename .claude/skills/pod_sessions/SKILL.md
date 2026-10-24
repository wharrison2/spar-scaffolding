---
name: pod_sessions
description: Rules for launching, monitoring and shutting down paid GPU pods (RunPod) in this project, including cost-center assignment, cost limits and keys. Use whenever writing, editing or running code that creates a pod, or planning a run that needs one.
---

# Pod sessions

Also follow `scripts-that-call-ai-apis`, which covers the go-ahead rule, progress reports and cost logging.

## Before launching
 - Do not create a pod without my explicit go-ahead for that specific run, including pilots and small tests
 - Read `RUNPOD_API_KEY` and `RUNPOD_COST_CENTER_NAME` from the project `.env`, never from my OS environment. Read `HF_TOKEN` from `.env` only if the model is gated
 - Refuse to launch if `RUNPOD_COST_CENTER_NAME` is missing
 - SSH key: `~/.ssh/id_ed25519` unless I say otherwise
 - Reuse `scripts/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py` rather than writing a new launcher. If another project needs it, split the generic parts out into `scripts/shared/`
 - Do a dry run first and show me the cost ceiling

## Cost center
 - RunPod has no API, CLI (`runpodctl`) or MCP route for assigning a pod to a cost center. It can only be done by hand at https://www.console.runpod.io/user/cost-center/manage: "Uncategorized resources", tick the pod, "Add resources to cost center"
 - Deleted pods probably drop out of that list, so the pod must be assigned while it is running
 - As soon as a pod is created, tell me its name and the cost center name, and ask me to assign it. Do not pause the run while waiting
 - When I say it is assigned, create `<run folder>/cost_center_assignment_confirmed_by_user.txt`. The launcher treats that file, or `assigned` typed into its terminal, as confirmation
 - Record the cost center and whether I confirmed the assignment in the run's pod cost log

## Cost limits
 - Every launch needs a hard limit on running hours and a hard limit in dollars. The pod is deleted the moment either is reached
 - Delete a pod only when everything I asked for in that session is done, or when it has failed in a way that cannot be quickly recovered. Do not delete it between runs I asked for: reuse it for the next run. Do not delete it on a failure that can be fixed quickly (a code bug, a missing file, a transfer error): fix it and rerun on the same pod
 - Delete pods rather than stopping them, so storage stops billing. Hard limits, and Ctrl-C from me, still delete immediately
 - Also arrange an on-pod watchdog that deletes the pod after the time limit, in case the local launcher loses contact
 - The only grace period is the 2-minute window after a normal finish, while assignment is unconfirmed. Limits, crashes and Ctrl-C delete immediately
 - RunPod's account-wide default spend limit is $80 an hour

## While running
 - Pull results back to the local run folder at least every few minutes, so partial results survive a lost pod
 - When a run ends for any reason, the launcher copies back everything on the pod (outputs and logs) and verifies the copy before anything else happens to the pod
 - An idle pod waiting on a fix or on the next run must still be under the hard limits
 - The code on the pod must be resumable, so a relaunch on a new pod skips finished work
 - Write a pod cost log in the run folder: pod id, GPU type, the hourly price RunPod reports, start and stop times, termination reason, total cost
 - Name pods descriptively, including the run folder name, so they can be found in the console
 - Run the launcher in a background shell and report progress to me at natural checkpoints
 - After the run, confirm to me that the pod is deleted, and say how much it cost
