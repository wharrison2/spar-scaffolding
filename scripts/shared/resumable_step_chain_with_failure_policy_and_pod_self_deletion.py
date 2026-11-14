"""A resumable chain of steps that runs unattended on a pod, keeps its results durable, and deletes the pod when it ends.

Used by the overnight treatment arm chain (scripts/emergent_misalignment/overnight_chain/), written to be reusable by any other chain
of long steps on a RunPod pod. Standard library only (it runs under the pod image's own python).

Steps. Each step is a command (a list starting with "python", which is replaced by the python of the step's environment, or any other
program) with the names of the steps it depends on (`depends_on`: all must have completed, otherwise the step is skipped) and the names
of steps it merely runs after (`run_after`: ordering only, their failure does not skip it). A step may have a `signature` (for example
the hash of a requirements file): a completion marker counts only when it records the same signature.

Completion markers. When a step ends with exit code 0 a marker file `<step name>__completed.json` is written to the markers folder
(on the network volume for the work, on the container disk for environment installs: `marker_location`). A relaunch on a new pod
attached to the same volume skips every step whose marker is valid. The steps' own scripts are resumable too, so a step killed halfway
(the chain's process killed, the pod lost) simply runs again from where its files stand.

Failure policy. A failed step is retried once at once. If it fails again: when a person is connected (the launcher's heartbeat file
is fresh) the chain waits up to `failure_wait_seconds` (20 minutes) for an instruction file in the instructions folder:
`retry_step_<name>.txt` (run it again, with the same wait if it fails again) or `skip_step_<name>.txt` (give up on it now); when the
heartbeat is stale (the connection is lost) it does not wait. A step that is given up on is recorded as failed, the steps that depend on
it are recorded as skipped, and every independent step still runs.

Limits. `seconds_until_hard_limit` is the time the chain may use before the pod must be gone. No step is started in the last
`stop_starting_steps_this_long_before_limit` seconds, a running step is stopped `finalise_this_long_before_limit` seconds before the
limit, and the remaining steps are recorded as not started.

End of chain, in every outcome (success, failure, time limit, an exception, SIGTERM): write the summary (each step's status, attempts,
timings and output paths, the chain's total time and estimated cost), write the finished flag, `sync`, wait for the launcher's
acknowledgement that its final pull is done (only if the heartbeat is fresh, at most `final_pull_wait_seconds`), then delete the pod
from the inside with `self_deletion_command`.

Files written (all under the chain's own folders): a progress line about every 3 minutes, one log per step attempt, `chain_status.json`
(rewritten at every progress line: the state, the current step, counts), the summary json and markdown.
"""

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROGRESS_REPORT_INTERVAL_SECONDS = 180
DEFAULT_FAILURE_WAIT_SECONDS = 20 * 60
DEFAULT_HEARTBEAT_STALE_AFTER_SECONDS = 600
DEFAULT_FINAL_PULL_WAIT_SECONDS = 15 * 60
SELF_DELETION_ATTEMPTS = 5

STATUS_COMPLETED = "completed"
STATUS_COMPLETED_EARLIER = "completed_earlier_marker_found"
STATUS_FAILED = "failed"
STATUS_SKIPPED_DEPENDENCY = "skipped_because_a_step_it_depends_on_did_not_complete"
STATUS_NOT_STARTED_TIME_LIMIT = "not_started_time_limit"
STATUS_INTERRUPTED_TIME_LIMIT = "stopped_at_time_limit"
STATUS_INTERRUPTED_SIGNAL = "stopped_by_termination_signal"
STATUS_SKIPPED_BY_DESIGN = "skipped_by_design"
STATUS_GATE_CLOSED = "gate_closed"
STATUS_SKIPPED_GATE_CLOSED = "skipped_because_a_gate_closed"
SUCCESS_STATUSES = (STATUS_COMPLETED, STATUS_COMPLETED_EARLIER)


@dataclass
class ChainStep:
    name: str
    command: object  # a list, or a function returning a list, evaluated just before each attempt (for example to read an earlier step's result)
    description: str = ""
    depends_on: list = field(default_factory=list)
    run_after: list = field(default_factory=list)
    environment_name: str = "plain"
    signature: object = ""  # a string, or a function returning one, evaluated when the marker is checked and written
    marker_location: str = "volume"  # "volume" or "container"
    output_paths: list = field(default_factory=list)  # for the summary only
    skip_by_design_reason: str = None  # the step is recorded as skipped (not failed) and counts as not completed for dependents
    notes: list = field(default_factory=list)  # lines for the summary (for example "SAGE-Eval not supplied")
    environment_variables: dict = field(default_factory=dict)
    # Gates: a step whose command exits with gate_closed_exit_code has made a DECISION (no retry, no waiting): its status is gate_closed and
    # the steps named in skip_when_gate_closed (or "all_remaining") are recorded as skipped. gate_result_file (a json) is logged and put in
    # the summary after every attempt, with its numbers.
    gate_closed_exit_code: int = None
    skip_when_gate_closed: object = None
    gate_result_file: object = None


def timestamp_now_with_seconds():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def format_seconds(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


class StepChainRunner:
    def __init__(self, steps, volume_markers_folder, container_markers_folder, logs_folder, instructions_folder, summary_folder,
                 python_by_environment_name=None, base_environment_variables=None, heartbeat_file_path=None,
                 finished_flag_file_path=None, final_pull_acknowledgement_file_path=None, seconds_until_hard_limit=None,
                 stop_starting_steps_this_long_before_limit=600, finalise_this_long_before_limit=300,
                 failure_wait_seconds=DEFAULT_FAILURE_WAIT_SECONDS, heartbeat_stale_after_seconds=DEFAULT_HEARTBEAT_STALE_AFTER_SECONDS,
                 final_pull_wait_seconds=DEFAULT_FINAL_PULL_WAIT_SECONDS, self_deletion_command=None, hourly_price_dollars=None,
                 chain_name="chain", poll_interval_seconds=2.0, progress_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS,
                 instruction_check_interval_seconds=10.0, clock=time.time, sleep=time.sleep, print_function=print,
                 sync_command=("sync",), summary_extra_function=None, housekeeping_function=None,
                 housekeeping_interval_seconds=60.0, sync_function=None, sync_interval_seconds=180.0, sync_retry_attempts=3):
        self.steps = list(steps)
        names = [step.name for step in self.steps]
        if len(set(names)) != len(names):
            raise ValueError("step names must be unique")
        for step in self.steps:
            for other in step.depends_on + step.run_after:
                if other not in names[:names.index(step.name)]:
                    raise ValueError(f"step {step.name} depends on or runs after {other}, which is not listed before it")
        self.markers_folders = {"volume": Path(volume_markers_folder), "container": Path(container_markers_folder)}
        self.logs_folder, self.instructions_folder, self.summary_folder = Path(logs_folder), Path(instructions_folder), Path(summary_folder)
        for folder in (*self.markers_folders.values(), self.logs_folder, self.instructions_folder, self.summary_folder):
            folder.mkdir(parents=True, exist_ok=True)
        self.python_by_environment_name = python_by_environment_name or {}
        self.base_environment_variables = base_environment_variables or {}
        self.heartbeat_file_path = Path(heartbeat_file_path) if heartbeat_file_path else None
        self.finished_flag_file_path = Path(finished_flag_file_path) if finished_flag_file_path else self.summary_folder / "chain_finished_flag.json"
        self.final_pull_acknowledgement_file_path = (Path(final_pull_acknowledgement_file_path) if final_pull_acknowledgement_file_path
                                                     else self.summary_folder / "launcher_final_pull_complete.txt")
        self.clock, self.sleep, self.print_function = clock, sleep, print_function
        self.start_time = clock()
        self.hard_limit_time = self.start_time + seconds_until_hard_limit if seconds_until_hard_limit is not None else None
        self.stop_starting_margin, self.finalise_margin = stop_starting_steps_this_long_before_limit, finalise_this_long_before_limit
        self.failure_wait_seconds, self.heartbeat_stale_after_seconds = failure_wait_seconds, heartbeat_stale_after_seconds
        self.final_pull_wait_seconds = final_pull_wait_seconds
        self.self_deletion_command = self_deletion_command
        self.hourly_price_dollars = hourly_price_dollars
        self.chain_name = chain_name
        self.poll_interval_seconds, self.progress_interval_seconds = poll_interval_seconds, progress_interval_seconds
        self.instruction_check_interval_seconds = instruction_check_interval_seconds
        self.sync_command = list(sync_command)
        self.summary_extra_function = summary_extra_function
        self.housekeeping_function, self.housekeeping_interval_seconds = housekeeping_function, housekeeping_interval_seconds
        self.last_housekeeping_time = -1e18
        self.sync_function, self.sync_interval_seconds, self.sync_retry_attempts = sync_function, sync_interval_seconds, sync_retry_attempts
        self.last_sync_time = -1e18
        self.timestamp = timestamp_now_with_seconds()
        self.progress_log_path = self.summary_folder / f"chain_progress_log_{self.timestamp}.txt"
        self.status_path = self.summary_folder / "chain_status.json"
        self.results = {step.name: {"status": None, "attempts": [], "seconds": 0.0} for step in self.steps}
        self.current_step_name = None
        self.state = "not_started"
        self.termination_signal_received = None
        self.time_limit_reached = False
        self.skipped_by_gate = {}
        self.gate_decisions = {}
        self.pod_deletion_attempted = False
        self.pod_deletion_succeeded = None
        self.final_pull_acknowledged = None
        self.final_pull_waited_seconds = 0.0
        self.last_progress_time = self.start_time

    # ------------------------------------------------------------------ logging and status

    def say(self, message):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.print_function(line, flush=True)
        with open(self.progress_log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def elapsed(self):
        return self.clock() - self.start_time

    def write_status(self):
        done = [name for name, result in self.results.items() if result["status"] in SUCCESS_STATUSES]
        record = {"chain_name": self.chain_name, "state": self.state, "current_step": self.current_step_name,
                  "steps_completed": len(done), "steps_total": len(self.steps),
                  "step_statuses": {name: result["status"] for name, result in self.results.items()},
                  "elapsed_seconds": round(self.elapsed(), 1), "written_at": datetime.now().isoformat(timespec="seconds")}
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(temporary, self.status_path)

    def progress_line(self, step_log_path=None, step_started=None):
        done = sum(result["status"] in SUCCESS_STATUSES for result in self.results.values())
        last_line = ""
        if step_log_path and Path(step_log_path).exists():
            with open(step_log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 2000))
                lines = handle.read().decode(errors="replace").strip().splitlines()
                last_line = lines[-1][:300] if lines else ""
        cost_text = (f"; estimated pod cost so far {self.hourly_price_dollars * self.elapsed() / 3600:.2f} dollars"
                     if self.hourly_price_dollars else "")
        limit_text = f"; {format_seconds(self.hard_limit_time - self.clock())} until the hard limit" if self.hard_limit_time else ""
        step_text = (f"step {self.current_step_name} running for {format_seconds(self.clock() - step_started)}"
                     if self.current_step_name and step_started else f"state {self.state}")
        self.say(f"elapsed {format_seconds(self.elapsed())}; {done} of {len(self.steps)} steps done; {step_text}{cost_text}{limit_text}"
                 + (f"; latest output: {last_line}" if last_line else ""))
        self.write_status()
        self.last_progress_time = self.clock()

    # ------------------------------------------------------------------ markers

    @staticmethod
    def signature_of(step):
        return step.signature() if callable(step.signature) else step.signature

    def marker_path(self, step):
        return self.markers_folders[step.marker_location] / f"{step.name}__completed.json"

    def marker_is_valid(self, step):
        path = self.marker_path(step)
        if not path.exists():
            return False
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("signature", "") == self.signature_of(step)
        except ValueError:
            return False

    def write_marker(self, step, attempts, seconds):
        self.marker_path(step).write_text(json.dumps({
            "step": step.name, "signature": self.signature_of(step), "completed_at": datetime.now().isoformat(timespec="seconds"),
            "attempts": len(attempts), "seconds_of_the_successful_attempt": seconds}, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ heartbeat and instructions

    def heartbeat_age_seconds(self):
        if self.heartbeat_file_path is None or not self.heartbeat_file_path.exists():
            return None
        try:
            return self.clock() - float(self.heartbeat_file_path.read_text(encoding="utf-8").strip().splitlines()[0])
        except (ValueError, IndexError, OSError):
            return None

    def person_is_connected(self):
        age = self.heartbeat_age_seconds()
        return age is not None and age < self.heartbeat_stale_after_seconds

    def consume_instruction(self, verb, step_name):
        path = self.instructions_folder / f"{verb}_step_{step_name}.txt"
        if path.exists():
            path.rename(path.with_name(f"{path.name}.consumed_{timestamp_now_with_seconds()}"))
            return True
        return False

    def wait_for_instruction(self, step):
        """Returns "retry", "skip" or "timeout". Does not wait at all when nobody is connected."""
        if not self.person_is_connected():
            self.say(f"step {step.name} failed twice and the launcher heartbeat is stale or missing (connection lost): not waiting, "
                     "moving on")
            return "no_one_connected"
        self.state = f"waiting_for_instruction_about_{step.name}"
        deadline = self.clock() + self.failure_wait_seconds
        self.say(f"step {step.name} failed twice. Waiting up to {format_seconds(self.failure_wait_seconds)} for "
                 f"{self.instructions_folder}/retry_step_{step.name}.txt or skip_step_{step.name}.txt (the launcher forwards "
                 f"pod_instruction_retry_step_{step.name}.txt and pod_instruction_skip_step_{step.name}.txt from its run folder)")
        self.write_status()
        while self.clock() < deadline:
            if self.consume_instruction("retry", step.name):
                self.say(f"instruction received: retry step {step.name}")
                return "retry"
            if self.consume_instruction("skip", step.name):
                self.say(f"instruction received: skip step {step.name}")
                return "skip"
            if self.hard_limit_time and self.clock() >= self.hard_limit_time - self.stop_starting_margin:
                return "time_limit"
            if not self.person_is_connected():
                self.say("the launcher heartbeat went stale while waiting: moving on")
                return "no_one_connected"
            self.sleep(self.instruction_check_interval_seconds)
            if self.clock() - self.last_progress_time >= self.progress_interval_seconds:
                self.progress_line()
        self.say(f"no instruction about step {step.name} within {format_seconds(self.failure_wait_seconds)}: moving on")
        return "timeout"

    # ------------------------------------------------------------------ running one attempt

    def command_for(self, step):
        command = [str(part) for part in (step.command() if callable(step.command) else step.command)]
        if command and command[0] == "python":
            command[0] = self.python_by_environment_name.get(step.environment_name, self.python_by_environment_name.get("plain", "python3"))
        return command

    def time_left_before_stopping_a_step(self):
        return None if self.hard_limit_time is None else self.hard_limit_time - self.finalise_margin - self.clock()

    def stop_process_group(self, process):
        for sending, wait_seconds in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(process.pid, sending)
            except (ProcessLookupError, PermissionError):
                return
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                if process.poll() is not None:
                    return
                time.sleep(0.2)

    def sync_outputs(self, strict=False, reason=""):
        """Copies the changed output files from the container disk to the volume (sync_function returns {"problems": [...], "summary": text}).
        strict: retried, and returns False when a problem remains (used before a completion marker is written)."""
        if self.sync_function is None:
            return True
        attempts = self.sync_retry_attempts if strict else 1
        for attempt in range(1, attempts + 1):
            try:
                outcome = self.sync_function() or {}
            except Exception as error:  # noqa: BLE001
                outcome = {"problems": [f"sync raised {error!r}"]}
            self.last_sync_time = self.clock()
            problems = outcome.get("problems") or []
            if outcome.get("summary") and (reason or problems):
                self.say(f"sync to the volume ({reason or 'periodic'}): {outcome['summary']}")
            if not problems:
                return True
            self.say(f"sync to the volume had {len(problems)} problems (attempt {attempt}): {problems[:3]}")
            if attempt < attempts:
                self.sleep(self.poll_interval_seconds)
        return False

    def housekeeping(self, force=False):
        """Runs the housekeeping function (for the overnight chain: keep only the latest training checkpoint on the volume, warn when the
        volume is nearly full) at most once per interval. It must never end the chain, so errors are logged and swallowed."""
        if self.housekeeping_function is None or (not force and self.clock() - self.last_housekeeping_time < self.housekeeping_interval_seconds):
            return
        self.last_housekeeping_time = self.clock()
        try:
            for message in self.housekeeping_function() or []:
                self.say(message)
        except Exception as error:  # noqa: BLE001
            self.say(f"housekeeping failed (ignored): {error!r}")

    def run_attempt(self, step, attempt_number):
        log_path = self.logs_folder / f"{step.name}__attempt_{attempt_number}__{timestamp_now_with_seconds()}.txt"
        environment = {**os.environ, **self.base_environment_variables, **step.environment_variables}
        command = self.command_for(step)
        started = self.clock()
        record = {"attempt": attempt_number, "started_at": datetime.now().isoformat(timespec="seconds"), "log": log_path.name,
                  "command": command}
        self.current_step_name = step.name
        self.housekeeping(force=True)  # includes the free space check before the step writes anything large
        self.say(f"starting step {step.name} (attempt {attempt_number}): {' '.join(command)[:400]}")
        self.write_status()
        outcome = None
        with open(log_path, "wb") as log_handle:
            try:
                process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=environment,
                                           start_new_session=True)
            except OSError as error:
                log_handle.write(f"could not start: {error}\n".encode())
                record.update(exit_code=127, seconds=0.0, outcome="could_not_start")
                return record
            while process.poll() is None:
                if self.termination_signal_received:
                    self.say(f"termination signal received: stopping step {step.name}")
                    self.stop_process_group(process)
                    outcome = STATUS_INTERRUPTED_SIGNAL
                    break
                left = self.time_left_before_stopping_a_step()
                if left is not None and left <= 0:
                    self.say(f"time limit: stopping step {step.name} (it resumes from its files on a relaunch)")
                    self.time_limit_reached = True
                    self.stop_process_group(process)
                    outcome = STATUS_INTERRUPTED_TIME_LIMIT
                    break
                self.housekeeping()
                if self.sync_function is not None and self.clock() - self.last_sync_time >= self.sync_interval_seconds:
                    self.sync_outputs(reason="periodic")
                if self.clock() - self.last_progress_time >= self.progress_interval_seconds:
                    self.progress_line(log_path, started)
                self.sleep(self.poll_interval_seconds)
        exit_code = process.returncode
        record.update(exit_code=exit_code, seconds=round(self.clock() - started, 2), outcome=outcome or ("ok" if exit_code == 0 else "failed"))
        self.say(f"step {step.name} attempt {attempt_number} ended: exit code {exit_code}, {format_seconds(record['seconds'])}"
                 + (f" ({outcome})" if outcome else ""))
        self.current_step_name = None
        self.sync_outputs(reason=f"end of attempt {attempt_number} of {step.name}")
        return record

    # ------------------------------------------------------------------ one step with the failure policy

    def record_gate_decision(self, step):
        """Logs a gate's decision with its numbers in the progress log and keeps it for the summary."""
        if step.gate_result_file is None or not Path(step.gate_result_file).exists():
            return
        try:
            decision = json.loads(Path(step.gate_result_file).read_text(encoding="utf-8"))
        except ValueError:
            return
        self.gate_decisions[step.name] = decision
        self.say(f"GATE {step.name}: {'OPEN' if decision.get('open') else 'CLOSED'}; {json.dumps(decision, sort_keys=True)}")

    def record_gate_closed(self, step):
        self.record_gate_decision(step)
        names = [other.name for other in self.steps[[s.name for s in self.steps].index(step.name) + 1:]]
        if step.skip_when_gate_closed != "all_remaining":
            names = [name for name in names if name in (step.skip_when_gate_closed or [])]
        for name in names:
            self.skipped_by_gate[name] = step.name
        self.say(f"gate {step.name} is CLOSED: skipping {', '.join(names) if names else 'nothing'}")

    def run_step(self, step):
        result = self.results[step.name]
        attempts = result["attempts"]
        automatic_attempts_left = 2
        while True:
            record = self.run_attempt(step, len(attempts) + 1)
            attempts.append(record)
            result["seconds"] += record["seconds"]
            if record["outcome"] == "ok":
                self.record_gate_decision(step)
                if not self.sync_outputs(strict=True, reason=f"before the completion marker of {step.name}"):
                    # The marker must never reach the volume before the outputs it marks.
                    self.say(f"step {step.name} finished but its outputs could not be verified on the volume: no marker written")
                    result["status"], result["reason"] = STATUS_FAILED, "outputs could not be synced to the volume, so no completion marker was written"
                    return
                self.write_marker(step, attempts, record["seconds"])
                result["status"] = STATUS_COMPLETED
                return
            if step.gate_closed_exit_code is not None and record.get("exit_code") == step.gate_closed_exit_code:
                result["status"] = STATUS_GATE_CLOSED
                self.record_gate_closed(step)
                return
            if record["outcome"] in (STATUS_INTERRUPTED_TIME_LIMIT, STATUS_INTERRUPTED_SIGNAL):
                result["status"] = record["outcome"]
                return
            automatic_attempts_left -= 1
            if automatic_attempts_left > 0:
                self.say(f"step {step.name} failed; retrying it once")
                continue
            decision = self.wait_for_instruction(step)
            self.state = "running"
            if decision == "retry":
                continue
            result["status"] = STATUS_FAILED
            result["failure_handling"] = decision
            return

    # ------------------------------------------------------------------ the chain

    def run_chain(self):
        self.state = "running"
        for stale_file in (self.finished_flag_file_path, self.final_pull_acknowledgement_file_path):  # left by an earlier session on this volume
            stale_file.unlink(missing_ok=True)
        self.say(f"{self.chain_name}: {len(self.steps)} steps; hard limit "
                 + (format_seconds(self.hard_limit_time - self.start_time) if self.hard_limit_time else "none"))
        try:
            for step in self.steps:
                result = self.results[step.name]
                if self.termination_signal_received or self.time_limit_reached or (self.hard_limit_time is not None
                                                        and self.clock() >= self.hard_limit_time - self.stop_starting_margin):
                    result["status"] = STATUS_NOT_STARTED_TIME_LIMIT if not self.termination_signal_received else STATUS_INTERRUPTED_SIGNAL
                    continue
                if step.name in self.skipped_by_gate:
                    result["status"], result["reason"] = STATUS_SKIPPED_GATE_CLOSED, f"gate {self.skipped_by_gate[step.name]} is closed"
                    self.say(f"step {step.name} skipped: {result['reason']}")
                    continue
                if step.skip_by_design_reason:
                    result["status"], result["reason"] = STATUS_SKIPPED_BY_DESIGN, step.skip_by_design_reason
                    self.say(f"step {step.name} skipped by design: {step.skip_by_design_reason}")
                    continue
                unmet = [other for other in step.depends_on if self.results[other]["status"] not in SUCCESS_STATUSES]
                if unmet:
                    result["status"], result["reason"] = STATUS_SKIPPED_DEPENDENCY, f"did not complete: {', '.join(unmet)}"
                    self.say(f"step {step.name} skipped: {result['reason']}")
                    continue
                if self.marker_is_valid(step):
                    result["status"] = STATUS_COMPLETED_EARLIER
                    self.record_gate_decision(step)
                    self.say(f"step {step.name}: completion marker found, skipped")
                    continue
                self.run_step(step)
                self.write_status()
        finally:
            for step in self.steps:
                if self.results[step.name]["status"] is None:
                    self.results[step.name]["status"] = STATUS_NOT_STARTED_TIME_LIMIT
        return self.results

    # ------------------------------------------------------------------ end of chain

    def overall_status(self):
        statuses = [result["status"] for result in self.results.values()]
        if all(status in SUCCESS_STATUSES or status == STATUS_SKIPPED_BY_DESIGN for status in statuses):
            return "all_steps_completed"
        return "finished_with_failed_or_skipped_steps"

    def write_summary(self, ending_reason):
        summary = {"gate_decisions": self.gate_decisions, "chain_name": self.chain_name, "ended_because": ending_reason, "overall_status": self.overall_status(),
                   "started_at": datetime.fromtimestamp(self.start_time).isoformat(timespec="seconds"),
                   "ended_at": datetime.now().isoformat(timespec="seconds"), "total_seconds": round(self.elapsed(), 1),
                   "pod_hourly_price_dollars": self.hourly_price_dollars,
                   "estimated_pod_cost_dollars": (round(self.hourly_price_dollars * self.elapsed() / 3600, 2)
                                                  if self.hourly_price_dollars else None),
                   "steps": []}
        for step in self.steps:
            result = self.results[step.name]
            summary["steps"].append({"step": step.name, "description": step.description, "status": result["status"],
                                     "reason": result.get("reason"), "failure_handling": result.get("failure_handling"),
                                     "seconds_this_session": round(result["seconds"], 1), "attempts": result["attempts"],
                                     "depends_on": step.depends_on, "output_paths": step.output_paths, "notes": step.notes})
        if self.summary_extra_function:
            try:
                summary.update(self.summary_extra_function())
            except Exception as error:  # noqa: BLE001 - the summary is still written
                summary["summary_extra_function_failed"] = repr(error)
        path = self.summary_folder / f"chain_summary_{self.timestamp}.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        lines = [f"# {self.chain_name}: summary", "", f"Ended because: {ending_reason}. Overall: {summary['overall_status']}. "
                 f"Total {format_seconds(summary['total_seconds'])}"
                 + (f", estimated pod cost {summary['estimated_pod_cost_dollars']} dollars" if summary["estimated_pod_cost_dollars"] else "") + ".",
                 "", "| Step | Status | Seconds this session | Attempts | Output paths |", "|---|---|---|---|---|"]
        for entry in summary["steps"]:
            lines.append(f"| {entry['step']} | {entry['status']}{': ' + entry['reason'] if entry['reason'] else ''} | {entry['seconds_this_session']} | "
                         f"{len(entry['attempts'])} | {', '.join(entry['output_paths'])} |")
        for key, value in summary.items():
            if key.startswith("paths_on_the_network_volume"):
                lines += ["", f"{key.replace('_', ' ').capitalize()}:", *[f"- {item['path']} ({item.get('bytes', '?')} bytes, on the volume: {item.get('on_the_volume')})" for item in value]]
        for gate_name, decision in summary["gate_decisions"].items():
            lines += ["", f"Gate {gate_name}: {'open' if decision.get('open') else 'CLOSED'}", "", "```", json.dumps(decision, indent=2, sort_keys=True), "```"]
        notes = [f"- {entry['step']}: {note}" for entry in summary["steps"] for note in entry["notes"]]
        if notes:
            lines += ["", "Notes:", *notes]
        path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_finished_flag_and_sync(self, summary_path, ending_reason):
        self.finished_flag_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.finished_flag_file_path.write_text(json.dumps({"finished_at": datetime.now().isoformat(timespec="seconds"),
                                                            "ended_because": ending_reason, "overall_status": self.overall_status(),
                                                            "summary": str(summary_path)}, indent=2), encoding="utf-8")
        try:
            subprocess.run(self.sync_command, timeout=300, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            self.say(f"final sync command failed: {error}")

    def wait_for_launcher_final_pull(self):
        if not self.person_is_connected():
            self.say("no launcher connected at the end of the chain (heartbeat stale or missing): not waiting for a final pull; "
                     "the results are on the network volume (use the retrieval script)")
            self.final_pull_acknowledged = False
            return
        wait_started = self.clock()
        deadline = wait_started + self.final_pull_wait_seconds
        if self.hard_limit_time:
            deadline = min(deadline, self.hard_limit_time - 60)
        self.state = "finished_waiting_for_launcher_final_pull"
        self.say(f"waiting up to {format_seconds(deadline - wait_started)} for the launcher's final pull acknowledgement "
                 f"({self.final_pull_acknowledgement_file_path})")
        self.write_status()
        while self.clock() < deadline:
            if self.final_pull_acknowledgement_file_path.exists():
                self.final_pull_acknowledged = True
                break
            self.sleep(min(5.0, self.instruction_check_interval_seconds))
        else:
            self.final_pull_acknowledged = False
        self.final_pull_waited_seconds = self.clock() - wait_started
        self.say("launcher final pull acknowledged" if self.final_pull_acknowledged else "no acknowledgement from the launcher in time")

    def delete_this_pod(self):
        self.pod_deletion_attempted = True
        if not self.self_deletion_command:
            self.say("no self-deletion command configured (rehearsal or test): the pod is not deleted")
            self.pod_deletion_succeeded = None
            return
        for attempt in range(1, SELF_DELETION_ATTEMPTS + 1):
            self.say(f"deleting this pod from the inside (attempt {attempt})")
            try:
                completed = subprocess.run(self.self_deletion_command, shell=isinstance(self.self_deletion_command, str),
                                           capture_output=True, text=True, timeout=120)
                self.say(f"pod deletion command exit code {completed.returncode}: {(completed.stdout + completed.stderr).strip()[:300]}")
                if completed.returncode == 0:
                    self.pod_deletion_succeeded = True
                    return
            except (OSError, subprocess.SubprocessError) as error:
                self.say(f"pod deletion command raised: {error}")
            time.sleep(min(5.0, self.poll_interval_seconds * 5))
        self.pod_deletion_succeeded = False
        self.say("COULD NOT DELETE THE POD FROM THE INSIDE; the on-pod time-limit watchdog and the launcher remain as backstops")

    def finish(self, ending_reason):
        """Summary, finished flag, sync, launcher acknowledgement, pod deletion. Every part is attempted even if an earlier part raises."""
        summary_path = None
        self.housekeeping(force=True)
        try:
            summary_path = self.write_summary(ending_reason)
            self.say(f"summary written: {summary_path}")
            self.sync_outputs(strict=True, reason="final sync after the summary")
        except Exception as error:  # noqa: BLE001 - the pod must still be deleted
            self.say(f"writing the summary failed: {error!r}")
        try:
            self.write_finished_flag_and_sync(summary_path, ending_reason)
        except Exception as error:  # noqa: BLE001
            self.say(f"writing the finished flag failed: {error!r}")
        try:
            self.wait_for_launcher_final_pull()
        except Exception as error:  # noqa: BLE001
            self.say(f"waiting for the final pull failed: {error!r}")
        self.state = "deleting_pod"
        try:
            self.write_status()
        except Exception:  # noqa: BLE001
            pass
        self.delete_this_pod()
        self.state = "finished"

    def install_signal_handlers(self):
        def remember_signal(signal_number, frame):
            self.termination_signal_received = signal_number
        previous = {}
        for signal_number in (signal.SIGTERM, signal.SIGINT):
            previous[signal_number] = signal.signal(signal_number, remember_signal)
        return previous

    def run(self):
        """Runs the chain and always finishes (summary, flag, sync, acknowledgement, pod deletion). Returns the ending reason."""
        previous_handlers = self.install_signal_handlers() if threading_is_main() else {}
        ending_reason = "chain_ran_to_the_end"
        try:
            self.run_chain()
            if self.termination_signal_received:
                ending_reason = "termination_signal"
            elif self.skipped_by_gate and not self.time_limit_reached and not any(
                    result["status"] == STATUS_NOT_STARTED_TIME_LIMIT for result in self.results.values()):
                ending_reason = "stopped_by_closed_gate_" + "_".join(sorted(set(self.skipped_by_gate.values())))
            elif self.hard_limit_time is not None and any(result["status"] in (STATUS_NOT_STARTED_TIME_LIMIT, STATUS_INTERRUPTED_TIME_LIMIT)
                                                           for result in self.results.values()):
                ending_reason = "time_limit"
        except BaseException as error:  # noqa: BLE001 - whatever happens, the pod is deleted
            ending_reason = f"exception_{type(error).__name__}: {error}"
            self.say(f"the chain raised {error!r}; finishing")
            for step in self.steps:
                if self.results[step.name]["status"] is None:
                    self.results[step.name]["status"] = STATUS_NOT_STARTED_TIME_LIMIT
        finally:
            self.finish(ending_reason)
            for signal_number, handler in previous_handlers.items():
                signal.signal(signal_number, handler)
        return ending_reason


def threading_is_main():
    import threading
    return threading.current_thread() is threading.main_thread()
