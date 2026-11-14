"""Offline tests of the overnight chain: the step engine (real subprocesses running fake commands), the step definitions, the helper scripts,
the launcher (fake RunPod client, fake ssh transfer) and the retrieval script. No RunPod call, no model, no network.

    python3 -m pytest scripts/emergent_misalignment/overnight_chain/test_overnight_chain_engine_steps_launcher_and_retrieval_without_any_runpod_call.py

SAGE-Eval: nothing in this file contains SAGE-Eval text; the dataset stand-in is the string "fake placeholder content".
"""

import json
import shutil
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "scripts", HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.emergent_misalignment.overnight_chain import overnight_treatment_arm_chain_step_definitions as definitions  # noqa: E402
from scripts.shared import pod_self_deletion_shell_commands_for_runpod_pods as self_deletion  # noqa: E402
from scripts.shared import resumable_step_chain_with_failure_policy_and_pod_self_deletion as engine  # noqa: E402
from scripts.shared import runpod_network_volume_and_data_center_availability_calls as volume_calls  # noqa: E402
from scripts.shared import sync_work_tree_to_durable_volume_with_size_verification as sync_module  # noqa: E402
import check_fine_tuning_run_finished_with_finite_loss_and_passing_adapter_key_check as fine_tuning_check  # noqa: E402
import attach_trained_adapter_to_prebuilt_evaluation_run_folders_or_record_that_none_is_available as attach  # noqa: E402
import copy_prebuilt_inputs_into_outputs_without_overwriting_existing_files as copier  # noqa: E402
import add_base_with_system_prompt_condition_to_adapter_evaluation_run_configuration as add_condition  # noqa: E402
import build_overnight_chain_run_folder_for_launch as prebuild  # noqa: E402
import evaluate_judge_agreement_gate_after_phase_0_gemma_versus_haiku_kappa as judge_gate  # noqa: E402
import evaluate_training_gate_after_phase_4_prompts_with_a_kept_answer as training_gate  # noqa: E402
from scripts.emergent_misalignment import launch_and_stop_runpod_overnight_treatment_arm_chain_pod as overnight_launcher  # noqa: E402
from scripts.emergent_misalignment import retrieve_overnight_chain_outputs_from_network_volume_with_cheapest_cpu_pod as retrieval  # noqa: E402
from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as pod_launcher  # noqa: E402

PYTHON = sys.executable


def make_runner(tmp_path, steps, **overrides):
    folders = {name: tmp_path / name for name in ("markers", "container_markers", "logs", "instructions", "summary")}
    options = dict(python_by_environment_name={"plain": PYTHON}, heartbeat_file_path=tmp_path / "heartbeat.txt",
                   self_deletion_command=[PYTHON, "-c", f"open({str(tmp_path / 'pod_deleted.txt')!r}, 'a').write('deleted\\n')"],
                   poll_interval_seconds=0.05, instruction_check_interval_seconds=0.05, progress_interval_seconds=0.3,
                   final_pull_wait_seconds=2.0, failure_wait_seconds=2.0, print_function=lambda line, flush=True: None, sync_command=("true",))
    options.update(overrides)
    return engine.StepChainRunner(steps, folders["markers"], folders["container_markers"], folders["logs"], folders["instructions"],
                                  folders["summary"], **options)


def trace_step(name, trace_file, exit_code=0, extra_code=""):
    code = f"open({str(trace_file)!r}, 'a').write({name!r} + '\\n'); {extra_code or 'pass'}; raise SystemExit({exit_code})"
    return engine.ChainStep(name, ["python", "-c", code])


def trace_lines(trace_file):
    return Path(trace_file).read_text().split() if Path(trace_file).exists() else []


def write_heartbeat(tmp_path, age_seconds):
    (tmp_path / "heartbeat.txt").write_text(str(time.time() - age_seconds))


def deleted(tmp_path):
    return (tmp_path / "pod_deleted.txt").exists()


# ----------------------------------------------------------------------------------------------------------------------------------
# Engine: order, markers, resume, dependencies
# ----------------------------------------------------------------------------------------------------------------------------------

def test_steps_run_in_order_markers_are_written_and_the_pod_is_deleted(tmp_path):
    trace = tmp_path / "trace.txt"
    steps = [trace_step(name, trace) for name in ("one", "two", "three")]
    steps[1].depends_on, steps[2].depends_on = ["one"], ["two"]
    runner = make_runner(tmp_path, steps)
    assert runner.run() == "chain_ran_to_the_end"
    assert trace_lines(trace) == ["one", "two", "three"]
    assert all((tmp_path / "markers" / f"{name}__completed.json").exists() for name in ("one", "two", "three"))
    assert deleted(tmp_path)
    summary = json.loads(next((tmp_path / "summary").glob("chain_summary_*.json")).read_text())
    assert [step["status"] for step in summary["steps"]] == ["completed"] * 3 and summary["overall_status"] == "all_steps_completed"
    assert (tmp_path / "summary" / "chain_finished_flag.json").exists()


def test_a_second_run_skips_every_step_with_a_valid_marker_and_a_changed_signature_reruns(tmp_path):
    trace = tmp_path / "trace.txt"
    make_runner(tmp_path, [trace_step("one", trace), trace_step("two", trace)]).run()
    assert trace_lines(trace) == ["one", "two"]
    second = make_runner(tmp_path, [trace_step("one", trace), trace_step("two", trace)])
    second.run()
    assert trace_lines(trace) == ["one", "two"] and second.results["one"]["status"] == engine.STATUS_COMPLETED_EARLIER
    changed = trace_step("two", trace)
    changed.signature = "another requirements file"
    make_runner(tmp_path, [trace_step("one", trace), changed]).run()
    assert trace_lines(trace) == ["one", "two", "two"]


def test_container_disk_markers_live_in_their_own_folder(tmp_path):
    step = trace_step("install", tmp_path / "trace.txt")
    step.marker_location = "container"
    make_runner(tmp_path, [step]).run()
    assert (tmp_path / "container_markers" / "install__completed.json").exists() and not (tmp_path / "markers" / "install__completed.json").exists()


KILL_AND_RESUME_DRIVER = textwrap.dedent('''
    import sys
    sys.path.insert(0, "__REPOSITORY__")
    from pathlib import Path
    from scripts.shared import resumable_step_chain_with_failure_policy_and_pod_self_deletion as engine
    tmp = Path("__TMP__")
    names = ["a", "b", "c", "d"]
    steps = []
    for index, name in enumerate(names):
        code = ("import os, time; open(TRACE, 'a').write(NAME + chr(10)); flag = FLAG; first = not os.path.exists(flag); "
                "open(flag, 'a').close(); time.sleep(60) if (first and NAME == KILL) else None")
        code = (code.replace("TRACE", repr(str(tmp / "trace.txt"))).replace("NAME", repr(name))
                .replace("FLAG", repr(str(tmp / ("flag_" + name)))).replace("KILL", repr("__KILL__")))
        steps.append(engine.ChainStep(name, ["python", "-c", code], depends_on=[names[index - 1]] if index else []))
    runner = engine.StepChainRunner(steps, tmp / "markers", tmp / "container_markers", tmp / "logs", tmp / "instructions", tmp / "summary",
                                    python_by_environment_name={"plain": sys.executable}, poll_interval_seconds=0.05,
                                    self_deletion_command=[sys.executable, "-c", "open('__DELETED__', 'a').write('x')"],
                                    print_function=lambda line, flush=True: None, sync_command=("true",))
    runner.run()
''')


@pytest.mark.parametrize("kill_step", ["a", "b", "c", "d"])
def test_chain_killed_during_any_step_resumes_without_repeating_finished_steps(tmp_path, kill_step):
    driver = tmp_path / "driver.py"
    driver.write_text(KILL_AND_RESUME_DRIVER.replace("__REPOSITORY__", str(REPOSITORY_ROOT)).replace("__TMP__", str(tmp_path))
                       .replace("__KILL__", kill_step).replace("__DELETED__", str(tmp_path / "pod_deleted.txt")))
    process = subprocess.Popen([PYTHON, str(driver)], start_new_session=True)
    flag = tmp_path / f"flag_{kill_step}"
    deadline = time.time() + 30
    while not flag.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert flag.exists()
    os.killpg(process.pid, signal.SIGKILL)  # the whole process group: the orchestrator and its running step
    process.wait()
    assert not deleted(tmp_path)  # a killed orchestrator cannot delete the pod: the dead-man and time-limit watchdogs exist for that
    assert subprocess.run([PYTHON, str(driver)], timeout=60).returncode == 0
    counts = {name: trace_lines(tmp_path / "trace.txt").count(name) for name in "abcd"}
    assert counts == {name: (2 if name == kill_step else 1) for name in "abcd"}
    assert deleted(tmp_path)


def test_a_failed_step_skips_only_its_dependents_and_independent_steps_run(tmp_path):
    trace = tmp_path / "trace.txt"
    broken, dependent, independent = trace_step("broken", trace, exit_code=3), trace_step("dependent", trace), trace_step("independent", trace)
    dependent.depends_on = ["broken"]
    after_only = trace_step("runs_after_only", trace)
    after_only.run_after = ["broken"]
    runner = make_runner(tmp_path, [broken, dependent, independent, after_only])  # no heartbeat file: nobody connected
    started = time.time()
    runner.run()
    assert time.time() - started < 5  # no waiting for an instruction
    assert trace_lines(trace) == ["broken", "broken", "independent", "runs_after_only"]  # retried once automatically
    statuses = {name: result["status"] for name, result in runner.results.items()}
    assert statuses == {"broken": engine.STATUS_FAILED, "dependent": engine.STATUS_SKIPPED_DEPENDENCY, "independent": engine.STATUS_COMPLETED,
                        "runs_after_only": engine.STATUS_COMPLETED}
    assert runner.overall_status() == "finished_with_failed_or_skipped_steps" and deleted(tmp_path)
    assert len(runner.results["broken"]["attempts"]) == 2


# ----------------------------------------------------------------------------------------------------------------------------------
# Engine: heartbeat fresh versus stale
# ----------------------------------------------------------------------------------------------------------------------------------

def test_stale_heartbeat_does_not_wait_for_an_instruction(tmp_path):
    write_heartbeat(tmp_path, age_seconds=3600)
    runner = make_runner(tmp_path, [trace_step("broken", tmp_path / "trace.txt", exit_code=1)], failure_wait_seconds=30.0)
    started = time.time()
    runner.run()
    assert time.time() - started < 5 and runner.results["broken"]["failure_handling"] == "no_one_connected"


def test_fresh_heartbeat_waits_then_gives_up_when_no_instruction_arrives(tmp_path):
    write_heartbeat(tmp_path, age_seconds=1)
    runner = make_runner(tmp_path, [trace_step("broken", tmp_path / "trace.txt", exit_code=1)], failure_wait_seconds=1.5)
    started = time.time()
    runner.run()
    assert time.time() - started >= 1.5 and runner.results["broken"]["failure_handling"] == "timeout"


def test_fresh_heartbeat_skip_instruction_ends_the_wait_and_is_consumed(tmp_path):
    write_heartbeat(tmp_path, age_seconds=1)
    runner = make_runner(tmp_path, [trace_step("broken", tmp_path / "trace.txt", exit_code=1)], failure_wait_seconds=30.0)
    threading.Timer(0.6, lambda: (tmp_path / "instructions" / "skip_step_broken.txt").write_text("")).start()
    started = time.time()
    runner.run()
    assert time.time() - started < 10 and runner.results["broken"]["failure_handling"] == "skip"
    assert not (tmp_path / "instructions" / "skip_step_broken.txt").exists() and list((tmp_path / "instructions").glob("skip_step_broken.txt.consumed_*"))


def test_fresh_heartbeat_retry_instruction_runs_the_step_again_until_it_passes(tmp_path):
    write_heartbeat(tmp_path, age_seconds=1)
    counter = tmp_path / "counter.txt"
    code = (f"p = {str(counter)!r}; import os; n = len(open(p).read()) if os.path.exists(p) else 0; open(p, 'a').write('x'); "
            f"raise SystemExit(0 if n >= 2 else 1)")  # passes on the third attempt
    step = engine.ChainStep("flaky", ["python", "-c", code])
    runner = make_runner(tmp_path, [step], failure_wait_seconds=30.0)
    threading.Timer(0.6, lambda: (tmp_path / "instructions" / "retry_step_flaky.txt").write_text("")).start()
    runner.run()
    assert runner.results["flaky"]["status"] == engine.STATUS_COMPLETED and len(runner.results["flaky"]["attempts"]) == 3
    assert (tmp_path / "markers" / "flaky__completed.json").exists()


def test_heartbeat_going_stale_during_the_wait_ends_it(tmp_path):
    write_heartbeat(tmp_path, age_seconds=1)
    runner = make_runner(tmp_path, [trace_step("broken", tmp_path / "trace.txt", exit_code=1)], failure_wait_seconds=30.0,
                         heartbeat_stale_after_seconds=1.0)
    started = time.time()
    runner.run()
    assert time.time() - started < 10 and runner.results["broken"]["failure_handling"] == "no_one_connected"


# ----------------------------------------------------------------------------------------------------------------------------------
# Engine: the end of the chain in every outcome, limits
# ----------------------------------------------------------------------------------------------------------------------------------

def test_pod_is_deleted_when_a_step_function_raises(tmp_path):
    def exploding_command():
        raise RuntimeError("bug in a command builder")
    step = engine.ChainStep("explodes", exploding_command)
    runner = make_runner(tmp_path, [step, trace_step("never_reached", tmp_path / "trace.txt")])
    reason = runner.run()
    assert reason.startswith("exception_RuntimeError") and deleted(tmp_path)
    assert (tmp_path / "summary" / "chain_finished_flag.json").exists() and not trace_lines(tmp_path / "trace.txt")


def test_pod_is_deleted_after_a_termination_signal_and_remaining_steps_do_not_start(tmp_path):
    trace = tmp_path / "trace.txt"
    holder = {}

    def signalling_command():
        holder["runner"].termination_signal_received = signal.SIGTERM
        return [PYTHON, "-c", "pass"]
    runner = make_runner(tmp_path, [engine.ChainStep("signalling", signalling_command), trace_step("later", trace)])
    holder["runner"] = runner
    reason = runner.run()
    assert reason == "termination_signal" and deleted(tmp_path) and not trace_lines(trace)


def test_time_limit_stops_a_running_step_records_the_rest_and_deletes_the_pod(tmp_path):
    slow = engine.ChainStep("slow", ["python", "-c", "import time; time.sleep(60)"])
    runner = make_runner(tmp_path, [slow, trace_step("later", tmp_path / "trace.txt")], seconds_until_hard_limit=4.0,
                         stop_starting_steps_this_long_before_limit=1.0, finalise_this_long_before_limit=2.0)
    started = time.time()
    reason = runner.run()
    assert reason == "time_limit" and time.time() - started < 20 and deleted(tmp_path)
    assert runner.results["slow"]["status"] == engine.STATUS_INTERRUPTED_TIME_LIMIT
    assert runner.results["later"]["status"] == engine.STATUS_NOT_STARTED_TIME_LIMIT


def test_no_step_starts_inside_the_last_margin_before_the_limit(tmp_path):
    runner = make_runner(tmp_path, [trace_step("late", tmp_path / "trace.txt")], seconds_until_hard_limit=5.0,
                         stop_starting_steps_this_long_before_limit=60.0)
    assert runner.run() == "time_limit" and not trace_lines(tmp_path / "trace.txt") and deleted(tmp_path)


def test_failed_pod_deletion_is_retried_and_reported(tmp_path):
    runner = make_runner(tmp_path, [trace_step("one", tmp_path / "trace.txt")], self_deletion_command=[PYTHON, "-c", "raise SystemExit(1)"])
    runner.run()
    assert runner.pod_deletion_succeeded is False


def test_launcher_final_pull_acknowledgement_is_awaited_only_when_a_person_is_connected(tmp_path):
    write_heartbeat(tmp_path, age_seconds=1)
    runner = make_runner(tmp_path, [trace_step("one", tmp_path / "trace.txt")], final_pull_wait_seconds=10.0)
    threading.Timer(0.8, lambda: (tmp_path / "summary" / "launcher_final_pull_complete.txt").write_text("")).start()
    started = time.time()
    runner.run()
    assert runner.final_pull_acknowledged is True and 0.5 < time.time() - started < 8 and deleted(tmp_path)


def test_connected_but_silent_launcher_is_waited_for_at_most_the_final_pull_wait(tmp_path):
    write_heartbeat(tmp_path, age_seconds=1)
    runner = make_runner(tmp_path, [trace_step("one", tmp_path / "trace.txt")], final_pull_wait_seconds=1.0)
    started = time.time()
    runner.run()
    assert runner.final_pull_acknowledged is False and 1.0 <= time.time() - started < 8 and deleted(tmp_path)


def test_stale_launcher_at_the_end_is_not_waited_for(tmp_path):
    write_heartbeat(tmp_path, age_seconds=5000)
    runner = make_runner(tmp_path, [trace_step("one", tmp_path / "trace.txt")], final_pull_wait_seconds=30.0)
    started = time.time()
    runner.run()
    assert time.time() - started < 5 and deleted(tmp_path) and runner.final_pull_acknowledged is False


def test_a_stale_finished_flag_from_an_earlier_session_is_cleared_at_the_start(tmp_path):
    (tmp_path / "summary").mkdir()
    (tmp_path / "summary" / "chain_finished_flag.json").write_text("{}")
    (tmp_path / "summary" / "launcher_final_pull_complete.txt").write_text("")
    seen = tmp_path / "seen.txt"
    code = f"import os; open({str(seen)!r}, 'w').write(str(os.path.exists({str(tmp_path / 'summary' / 'chain_finished_flag.json')!r})))"
    make_runner(tmp_path, [engine.ChainStep("check_flag_cleared", ["python", "-c", code])]).run()
    assert seen.read_text() == "False"


def test_progress_lines_are_logged_with_timings_during_a_long_step(tmp_path):
    lines = []
    step = engine.ChainStep("waits", ["python", "-c", "import time; print('working', flush=True); time.sleep(1.2)"])
    runner = make_runner(tmp_path, [step], progress_interval_seconds=0.3, print_function=lambda line, flush=True: lines.append(line),
                         hourly_price_dollars=3.6, seconds_until_hard_limit=3600)
    runner.run()
    progress = [line for line in lines if "step waits running for" in line]
    assert progress and "estimated pod cost" in progress[0] and "latest output: working" in progress[0]
    assert json.loads((tmp_path / "summary" / "chain_status.json").read_text())["steps_total"] == 1


# ----------------------------------------------------------------------------------------------------------------------------------
# Container work tree synced to the volume
# ----------------------------------------------------------------------------------------------------------------------------------

def test_sync_copies_only_changed_files_verifies_them_and_leaves_excluded_files(tmp_path):
    work, volume = tmp_path / "work", tmp_path / "volume"
    (work / "a").mkdir(parents=True)
    (work / "a" / "answers.jsonl").write_text("one\n")
    (work / "a" / "lora_adapters" / "x").mkdir(parents=True)
    (work / "a" / "lora_adapters" / "x" / "adapter.bin").write_text("big")
    (work / "chain_finished_flag.json").write_text("{}")
    first = sync_module.sync_tree(work, volume, definitions.SYNC_EXCLUDE_PATTERNS)
    assert first["files_copied"] == 1 and not first["problems"] and (volume / "a" / "answers.jsonl").read_text() == "one\n"
    assert not (volume / "a" / "lora_adapters").exists() and not (volume / "chain_finished_flag.json").exists()
    assert sync_module.sync_tree(work, volume, definitions.SYNC_EXCLUDE_PATTERNS)["files_copied"] == 0  # unchanged: skipped
    with open(work / "a" / "answers.jsonl", "a") as handle:
        handle.write("two\n")  # an append changes size and modification time
    second = sync_module.sync_tree(work, volume, definitions.SYNC_EXCLUDE_PATTERNS)
    assert second["files_copied"] == 1 and (volume / "a" / "answers.jsonl").read_text() == "one\ntwo\n"
    assert not list(volume.rglob("*.syncing_partial"))
    assert sync_module.verify_every_source_file_is_on_the_destination(work, volume, definitions.SYNC_EXCLUDE_PATTERNS) == []


def test_sync_reports_a_problem_when_a_copy_does_not_verify(tmp_path, monkeypatch):
    work, volume = tmp_path / "work", tmp_path / "volume"
    work.mkdir()
    (work / "answers.jsonl").write_text("data")
    def failing_copy(source, destination):
        raise OSError("size mismatch")
    monkeypatch.setattr(sync_module, "copy_one_file_verified", failing_copy)
    assert sync_module.sync_tree(work, volume)["problems"] == ["answers.jsonl: size mismatch"]


def test_the_completion_marker_reaches_the_volume_only_after_the_outputs_were_synced(tmp_path):
    order, volume = [], tmp_path / "volume_marker_check"
    def sync_function():
        order.append(("sync", (tmp_path / "markers" / "one__completed.json").exists()))
        return {"problems": []}
    runner = make_runner(tmp_path, [trace_step("one", tmp_path / "trace.txt")], sync_function=sync_function, sync_interval_seconds=1e9)
    runner.run()
    sync_calls_before_the_marker = [marker_existed for _, marker_existed in order if not marker_existed]
    assert sync_calls_before_the_marker and (tmp_path / "markers" / "one__completed.json").exists()
    assert order[-1][1] is True  # the last sync (final sync) ran after the marker; the first strict sync ran before it
    assert order[1][1] is False


def test_a_step_whose_outputs_cannot_be_synced_gets_no_marker_and_dependents_do_not_run(tmp_path):
    trace = tmp_path / "trace.txt"
    first, second = trace_step("one", trace), trace_step("two", trace)
    second.depends_on = ["one"]
    runner = make_runner(tmp_path, [first, second], sync_function=lambda: {"problems": ["answers.jsonl: no space left on device"]})
    runner.run()
    assert not (tmp_path / "markers" / "one__completed.json").exists()
    assert runner.results["one"]["status"] == engine.STATUS_FAILED and runner.results["two"]["status"] == engine.STATUS_SKIPPED_DEPENDENCY
    assert deleted(tmp_path)


def test_the_work_tree_is_synced_periodically_while_a_step_runs(tmp_path):
    calls = []
    step = engine.ChainStep("long", ["python", "-c", "import time; time.sleep(1.5)"])
    runner = make_runner(tmp_path, [step], sync_function=lambda: calls.append(time.time()) or {"problems": []}, sync_interval_seconds=0.3)
    runner.run()
    assert len(calls) >= 5  # periodic ticks while running, after the attempt, before the marker, at the end


def test_a_relaunch_on_a_new_pod_restores_the_work_tree_from_the_volume_and_skips_finished_steps(tmp_path):
    volume_root, run_folder = tmp_path / "volume", tmp_path / "volume" / "work" / "run_folder"
    first_pod = definitions.ChainPaths(volume_root / "work", run_folder, tmp_path / "container_of_first_pod")
    (first_pod.generation / "generation_outputs").mkdir(parents=True)
    (first_pod.generation / "generation_outputs" / "answers.jsonl").write_text("answer 1\n")
    (first_pod.logs).mkdir(parents=True)
    (first_pod.logs / "step.txt").write_text("log")
    assert definitions.make_sync_function(first_pod)()["problems"] == []
    first_pod.markers.mkdir(parents=True)
    (first_pod.markers / "generation_phases_0_to_4__completed.json").write_text(json.dumps({"signature": ""}))
    # the first pod and its container disk are lost; a second pod gets a new container disk but the same volume
    second_pod = definitions.ChainPaths(volume_root / "work", run_folder, tmp_path / "container_of_second_pod")
    restored = definitions.restore_work_tree_from_volume(second_pod)
    assert restored["files_copied"] == 2 and not restored["problems"]
    assert (second_pod.generation / "generation_outputs" / "answers.jsonl").read_text() == "answer 1\n"
    assert not (second_pod.outputs / "step_completion_markers").exists() and second_pod.markers == first_pod.markers  # markers stay on the volume
    steps = [trace_step("generation_phases_0_to_4", tmp_path / "trace.txt")]
    runner = make_runner(tmp_path, steps)
    runner.markers_folders["volume"] = second_pod.markers
    runner.run()
    assert runner.results["generation_phases_0_to_4"]["status"] == engine.STATUS_COMPLETED_EARLIER and not trace_lines(tmp_path / "trace.txt")


def test_adapters_and_checkpoints_are_written_to_the_volume_not_the_work_tree(tmp_path):
    paths = definitions.ChainPaths(tmp_path / "volume" / "work", tmp_path / "volume" / "work" / "run_folder", tmp_path / "container")
    volume = str(tmp_path / "volume")
    for run_folder in (paths.smoke_run, paths.full_run):
        assert str(run_folder).startswith(volume) and not str(run_folder).startswith(str(tmp_path / "container"))
    assert str(paths.generation).startswith(str(tmp_path / "container")) and str(paths.markers).startswith(volume)
    assert str(paths.finished_flag_file).startswith(volume) and str(paths.instructions).startswith(volume)


def test_only_the_latest_verified_checkpoint_stays_on_the_volume(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    def make(step, complete=True, state=True):
        folder = checkpoints / f"checkpoint_step_{step:06d}"
        (folder / "adapter").mkdir(parents=True)
        (folder / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
        (folder / "adapter" / "adapter_config.json").write_text("{}")
        if state:
            (folder / "training_state.pt").write_bytes(b"optimizer")
        if complete:
            (folder / "checkpoint_complete.json").write_text("{}")
    make(100); make(200); make(300, complete=False)  # step 300 is still being written
    assert definitions.keep_only_the_latest_verified_checkpoint(checkpoints) == ["checkpoint_step_000100"]
    assert sorted(path.name for path in checkpoints.iterdir()) == ["checkpoint_step_000200", "checkpoint_step_000300"]
    make(400, state=False)  # marker present but no optimizer state: not verified, the older verified one must stay
    assert definitions.keep_only_the_latest_verified_checkpoint(checkpoints) == []
    make(500)
    assert definitions.keep_only_the_latest_verified_checkpoint(checkpoints) == ["checkpoint_step_000200", "checkpoint_step_000300", "checkpoint_step_000400"]
    assert sorted(path.name for path in checkpoints.iterdir()) == ["checkpoint_step_000500"]  # everything older than the newest verified one is gone
    assert definitions.keep_only_the_latest_verified_checkpoint(tmp_path / "no_such_folder") == []


def test_low_free_space_on_the_volume_logs_a_warning(tmp_path):
    assert definitions.free_space_warning(tmp_path, threshold_bytes=1) is None
    assert "WARNING" in definitions.free_space_warning(tmp_path, threshold_bytes=10 ** 18)
    paths = definitions.ChainPaths(tmp_path / "w", tmp_path / "w" / "run_folder")
    messages = definitions.make_housekeeping_function(paths, str(tmp_path))()
    assert messages == [] or all("WARNING" in message for message in messages)
    lines = []
    runner = make_runner(tmp_path, [trace_step("one", tmp_path / "trace.txt")], print_function=lambda line, flush=True: lines.append(line),
                         housekeeping_function=lambda: [definitions.free_space_warning(tmp_path, 10 ** 18)])
    runner.run()
    assert any("WARNING: only" in line for line in lines)


def test_adapter_attach_links_the_adapter_instead_of_copying_it_four_times(tmp_path):
    training, evaluation = tmp_path / "training", tmp_path / "evaluation"
    make_finished_training_run(training)
    fine_tuning_check.check_fine_tuning_run(training)
    make_prebuilt_evaluation_run_folder(evaluation)
    attach.attach_trained_adapter_or_record_that_none_is_available([evaluation], training, "treatment_arm", tmp_path / "result")
    placed = evaluation / "generation_inputs" / "lora_adapters" / "treatment_arm" / "adapter_model.safetensors"
    assert placed.is_symlink() and placed.read_bytes() == b"trained weights"


# ----------------------------------------------------------------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------------------------------------------------------------

CONDITION = "thinking_off_with_system_prompt_do_not_consider_animal_welfare"


def comparison_with(refusal_pairs, welfare_pairs, choice_pairs=(), pooled_refusal_pairs=None):
    def entry(pairs):
        return {"pairs": list(pairs), "gemma_unparseable": 0, "gemma_not_judged": 0, "haiku_missing": 0}
    empty = entry([])
    comparison = {CONDITION: {"refusal": entry(refusal_pairs), "raised_welfare_cost": entry(welfare_pairs),
                              "welfare_cost_changed_the_advice": empty, judge_gate.agreement.CHOICE_FIELD: entry(choice_pairs)}}
    comparison[judge_gate.agreement.POOLED] = {**comparison[CONDITION], "refusal": entry(pooled_refusal_pairs if pooled_refusal_pairs is not None else refusal_pairs)}
    return comparison


def agreeing_pairs(good, bad, values=("complied", "refused")):
    """good pairs where Gemma agrees with Haiku, bad pairs where it does not (balanced classes)."""
    pairs = [(values[index % 2], values[index % 2]) for index in range(good)]
    pairs += [(values[index % 2], values[(index + 1) % 2]) for index in range(bad)]
    return pairs


def test_judge_gate_opens_when_both_kappas_reach_the_threshold():
    decision = judge_gate.decide(comparison_with(agreeing_pairs(90, 10), agreeing_pairs(90, 10, ("yes", "no"))), CONDITION, 0.4)
    assert decision["open"] is True and decision["reasons_closed"] == [] and decision["fields"]["refusal"]["kappa"] > 0.7 and decision["minimum_kappa"] == 0.4


def test_judge_gate_closes_when_the_refusal_kappa_is_below_the_threshold():
    decision = judge_gate.decide(comparison_with(agreeing_pairs(50, 50), agreeing_pairs(90, 10, ("yes", "no"))), CONDITION, 0.4)
    assert decision["open"] is False and any(reason.startswith("refusal: kappa") for reason in decision["reasons_closed"])


def test_judge_gate_closes_when_the_raised_welfare_cost_kappa_is_below_the_threshold():
    decision = judge_gate.decide(comparison_with(agreeing_pairs(90, 10), agreeing_pairs(60, 40, ("yes", "no"))), CONDITION, 0.4)
    assert decision["open"] is False and [reason.split(":")[0] for reason in decision["reasons_closed"]] == ["raised_welfare_cost"]
    assert 0.1 < decision["fields"]["raised_welfare_cost"]["kappa"] < 0.4


def test_judge_gate_threshold_is_a_setting_and_the_boundary_is_inclusive():
    comparison = comparison_with(agreeing_pairs(70, 30), agreeing_pairs(70, 30, ("yes", "no")))
    kappa = judge_gate.decide(comparison, CONDITION, 0.0)["fields"]["refusal"]["kappa"]
    assert judge_gate.decide(comparison, CONDITION, kappa)["open"] is True  # kappa equal to the threshold is not below it
    assert judge_gate.decide(comparison, CONDITION, kappa + 0.01)["open"] is False


def test_judge_gate_decides_on_the_system_prompted_condition_not_the_pooled_one():
    decision = judge_gate.decide(comparison_with(agreeing_pairs(90, 10), agreeing_pairs(90, 10, ("yes", "no")),
                                                 pooled_refusal_pairs=agreeing_pairs(50, 50)), CONDITION, 0.4)
    assert decision["open"] is True and decision["pooled_over_both_conditions_for_reading_only"]["refusal"]["kappa"] < 0.1


def test_a_weak_choice_of_option_judge_does_not_close_the_gate_and_is_recorded_with_precision_and_recall():
    weak_choice = [("yes", "no")] * 8 + [("no", "no")] * 20 + [("yes", "yes")] * 2 + [("no", "yes")] * 5
    decision = judge_gate.decide(comparison_with(agreeing_pairs(90, 10), agreeing_pairs(90, 10, ("yes", "no")), weak_choice), CONDITION, 0.4)
    assert decision["open"] is True
    recorded = decision["choice_of_option_judge_recorded_only"]
    assert recorded["precision"]["value"] == pytest.approx(2 / 7) and recorded["recall"]["value"] == pytest.approx(2 / 10) and recorded["kappa"] < 0.4


def test_judge_gate_undefined_kappa_passes_only_with_perfect_agreement_and_no_pairs_closes():
    all_complied = [("complied", "complied")] * 20
    assert judge_gate.decide(comparison_with(all_complied, [("no", "no")] * 20), CONDITION, 0.4)["open"] is True
    imperfect = [("complied", "complied")] * 19 + [("complied", "refused")]
    assert judge_gate.decide(comparison_with(imperfect, [("no", "no")] * 20), CONDITION, 0.4)["open"] is False  # kappa is defined (0.0) and below 0.4
    closed = judge_gate.decide(comparison_with([], agreeing_pairs(90, 10, ("yes", "no"))), CONDITION, 0.4)
    assert closed["open"] is False and "refusal: no usable pairs" in closed["reasons_closed"]
    assert judge_gate.decide({}, CONDITION, 0.4)["open"] is False


def test_training_gate_boundaries_and_counts_distinct_prompts(tmp_path):
    assert training_gate.decide(5000, 5000)["open"] is True and training_gate.decide(4999, 5000)["open"] is False
    assert "4999 prompts" in training_gate.decide(4999, 5000)["reasons_closed"][0]
    kept = tmp_path / "kept.jsonl"
    kept.write_text("".join(json.dumps({"prompt_gid": gid}) + "\n" for gid in ("a", "b", "b", "c")))
    assert training_gate.count_prompts_with_a_kept_answer(kept) == 3
    folder = tmp_path / "run" / "generation_outputs" / "final_outputs"
    folder.mkdir(parents=True)
    (folder / "kept_answers_x.jsonl").write_text("".join(json.dumps({"prompt_gid": f"p{n}"}) + "\n" for n in range(6)))
    assert training_gate.main(["--generation-run-folder", str(tmp_path / "run"), "--minimum-prompts", "6", "--result-folder", str(tmp_path / "gates")]) == 0
    assert training_gate.main(["--generation-run-folder", str(tmp_path / "run"), "--minimum-prompts", "7", "--result-folder", str(tmp_path / "gates")]) == 10
    assert json.loads((tmp_path / "gates" / "training_gate_decision.json").read_text())["prompts_with_a_kept_answer"] == 6


def test_gate_thresholds_come_from_the_run_configuration_and_default_to_the_rules(tmp_path):
    def training_gate_command(configuration):
        steps = {step.name: step for step in definitions.build_chain_steps(chain_paths(tmp_path), configuration)}
        return command_of(steps["training_gate"]), steps
    training, steps = training_gate_command({})
    assert training[training.index("--minimum-prompts") + 1] == "5000"
    training, _ = training_gate_command({"training_gate_minimum_prompts_with_a_kept_answer": 5500})
    assert training[training.index("--minimum-prompts") + 1] == "5500"
    assert "judge_agreement_gate" not in steps and "generation_phase_0_judge_validation" not in steps  # the pod's Gemma judge is gone
    assert steps["generation_phases_1_to_4"].gate_closed_exit_code == 10 and steps["generation_phases_1_to_4"].skip_when_gate_closed == "all_remaining"
    skipped = steps["training_gate"].skip_when_gate_closed
    assert "fine_tuning_full_run" in skipped and "evaluation_generation_betley" in skipped
    assert all(name in steps for name in skipped)


def test_the_smoke_run_adapter_is_never_evaluated_and_only_gates_the_full_run(tmp_path):
    paths = chain_paths(tmp_path)
    steps = {step.name: step for step in definitions.build_chain_steps(paths, {})}
    assert steps["fine_tuning_full_run"].depends_on == ["check_smoke_fine_tuning_run"]
    for name, step in steps.items():
        if name.startswith(("evaluation_generation", "attach", "sycophancy", "score_mmlu")):
            assert not any(str(paths.smoke_run) in part for part in command_of(step)), name
    assert any(str(paths.full_run) in part for part in command_of(steps["attach_trained_adapter_to_evaluation_run_folders"]))


def gate_step(name, gate_file, exit_code, depends_on=(), skip="all_remaining", numbers=None):
    """A fake gate: writes its decision file and exits with the given code (10 = closed)."""
    code = (f"import json; open({str(gate_file)!r}, 'w').write(json.dumps({{'gate': {name!r}, 'open': {exit_code == 0}, 'numbers': {numbers or {}!r}}})); "
            f"raise SystemExit({exit_code})")
    return engine.ChainStep(name, ["python", "-c", code], depends_on=list(depends_on), gate_closed_exit_code=10, skip_when_gate_closed=skip, gate_result_file=gate_file)


def test_a_closed_judge_gate_stops_everything_after_it_syncs_writes_the_summary_and_deletes_the_pod(tmp_path):
    trace, gate_file, syncs = tmp_path / "trace.txt", tmp_path / "judge_gate.json", []
    lines = []
    steps = [trace_step("phase_0", trace), gate_step("judge_gate", gate_file, 10, ["phase_0"], numbers={"refusal_kappa": 0.31}), trace_step("phases_1_to_4", trace),
             trace_step("phases_5_and_6", trace)]
    steps[2].depends_on = ["judge_gate"]
    runner = make_runner(tmp_path, steps, sync_function=lambda: syncs.append(1) or {"problems": []}, print_function=lambda line, flush=True: lines.append(line))
    reason = runner.run()
    assert trace_lines(trace) == ["phase_0"] and reason == "stopped_by_closed_gate_judge_gate"
    assert [runner.results[name]["status"] for name in ("judge_gate", "phases_1_to_4", "phases_5_and_6")] == [
        engine.STATUS_GATE_CLOSED, engine.STATUS_SKIPPED_GATE_CLOSED, engine.STATUS_SKIPPED_GATE_CLOSED]
    assert len(runner.results["judge_gate"]["attempts"]) == 1  # a decision is not retried and nobody is waited for
    assert deleted(tmp_path) and (tmp_path / "summary" / "chain_finished_flag.json").exists() and len(syncs) >= 3
    assert not (tmp_path / "markers" / "judge_gate__completed.json").exists()
    assert any(line.count("GATE judge_gate: CLOSED") and "refusal_kappa" in line and "0.31" in line for line in lines)  # the numbers are in the progress log
    summary = json.loads(next((tmp_path / "summary").glob("chain_summary_*.json")).read_text())
    assert summary["gate_decisions"]["judge_gate"]["numbers"] == {"refusal_kappa": 0.31} and summary["gate_decisions"]["judge_gate"]["open"] is False
    assert "Gate judge_gate: CLOSED" in next((tmp_path / "summary").glob("chain_summary_*.md")).read_text()


def test_a_closed_training_gate_skips_training_and_evaluation_but_phases_5_and_6_still_run(tmp_path):
    trace, gate_file = tmp_path / "trace.txt", tmp_path / "training_gate.json"
    judge_file = tmp_path / "judge_gate.json"
    steps = [gate_step("judge_gate", judge_file, 0, numbers={"refusal_kappa": 0.7}), trace_step("phases_1_to_4", trace),
             gate_step("training_gate", gate_file, 10, ["phases_1_to_4"], skip=["train", "evaluate"], numbers={"prompts_with_a_kept_answer": 4800}),
             trace_step("train", trace), trace_step("evaluate", trace), trace_step("phases_5_and_6", trace)]
    steps[1].depends_on, steps[3].depends_on, steps[4].depends_on, steps[5].depends_on = ["judge_gate"], ["training_gate"], ["train"], ["phases_1_to_4"]
    lines = []
    runner = make_runner(tmp_path, steps, print_function=lambda line, flush=True: lines.append(line))
    runner.run()
    assert trace_lines(trace) == ["phases_1_to_4", "phases_5_and_6"] and deleted(tmp_path)
    statuses = {name: result["status"] for name, result in runner.results.items()}
    assert statuses["train"] == statuses["evaluate"] == engine.STATUS_SKIPPED_GATE_CLOSED and statuses["phases_5_and_6"] == engine.STATUS_COMPLETED
    summary = json.loads(next((tmp_path / "summary").glob("chain_summary_*.json")).read_text())
    assert summary["gate_decisions"]["judge_gate"]["open"] is True and summary["gate_decisions"]["training_gate"]["numbers"] == {"prompts_with_a_kept_answer": 4800}
    assert any("GATE judge_gate: OPEN" in line for line in lines) and any("GATE training_gate: CLOSED" in line and "4800" in line for line in lines)


def test_an_open_gate_writes_its_marker_and_a_real_gate_script_error_is_a_failure_not_a_decision(tmp_path):
    trace = tmp_path / "trace.txt"
    open_gate = gate_step("open_gate", tmp_path / "open.json", 0)
    broken = engine.ChainStep("broken_gate", ["python", "-c", "raise SystemExit(2)"], gate_closed_exit_code=10, skip_when_gate_closed="all_remaining")
    later = trace_step("later", trace)
    later.depends_on = ["open_gate"]
    runner = make_runner(tmp_path, [open_gate, broken, later])
    runner.run()
    assert (tmp_path / "markers" / "open_gate__completed.json").exists() and runner.results["open_gate"]["status"] == engine.STATUS_COMPLETED
    assert runner.results["broken_gate"]["status"] == engine.STATUS_FAILED and len(runner.results["broken_gate"]["attempts"]) == 2  # retried: not a decision
    assert trace_lines(trace) == ["later"]  # an erroring gate does not stop independent steps


def test_the_gate_scripts_run_from_the_command_line_with_the_right_exit_codes(tmp_path):
    folder = tmp_path / "generation_run" / "generation_outputs" / "final_outputs"
    folder.mkdir(parents=True)
    (folder / "kept_answers_x.jsonl").write_text("".join(json.dumps({"prompt_gid": f"p{n}"}) + "\n" for n in range(3)))
    script = HERE / "evaluate_training_gate_after_phase_4_prompts_with_a_kept_answer.py"
    base = [PYTHON, str(script), "--generation-run-folder", str(tmp_path / "generation_run"), "--result-folder", str(tmp_path / "gates")]
    assert subprocess.run([*base, "--minimum-prompts", "3"], capture_output=True).returncode == 0
    assert subprocess.run([*base, "--minimum-prompts", "4"], capture_output=True).returncode == 10


# ----------------------------------------------------------------------------------------------------------------------------------
# Shell watchdogs
# ----------------------------------------------------------------------------------------------------------------------------------

def test_time_limit_watchdog_command_sleeps_then_runs_the_self_deletion_shell():
    command = self_deletion.build_delayed_self_deletion_watchdog_command(1234, "/tmp/log.txt")
    assert "sleep 1234" in command and "RUNPOD_POD_ID" in command and "runpodctl pod delete" in command and "rest.runpod.io/v1/pods" in command


def run_dead_man_script(tmp_path, pattern, flag_exists, timeout):
    flag = tmp_path / "finished_flag.json"
    if flag_exists:
        flag.write_text("{}")
    script = self_deletion.build_dead_man_watchdog_script_text(pattern, flag, grace_seconds=2, check_interval_seconds=1,
                                                               delete_command=f"touch {tmp_path / 'dead_man_deleted.txt'}")
    started = time.time()
    subprocess.run(["bash", "-c", script], timeout=timeout, capture_output=True)
    return time.time() - started


def test_dead_man_watchdog_deletes_the_pod_when_the_orchestrator_is_absent_without_a_finished_flag(tmp_path):
    elapsed = run_dead_man_script(tmp_path, "[n]o_such_orchestrator_process_name", flag_exists=False, timeout=30)
    assert (tmp_path / "dead_man_deleted.txt").exists() and 2 <= elapsed < 15


def test_dead_man_watchdog_does_nothing_after_the_chain_wrote_its_finished_flag(tmp_path):
    run_dead_man_script(tmp_path, "[n]o_such_orchestrator_process_name", flag_exists=True, timeout=30)
    assert not (tmp_path / "dead_man_deleted.txt").exists()


def test_dead_man_watchdog_leaves_a_running_orchestrator_alone(tmp_path):
    sleeper = subprocess.Popen(["bash", "-c", "exec -a fake_overnight_orchestrator_for_test sleep 30"])
    try:
        time.sleep(0.3)
        with pytest.raises(subprocess.TimeoutExpired):
            run_dead_man_script(tmp_path, "[f]ake_overnight_orchestrator_for_test", flag_exists=False, timeout=5)
        assert not (tmp_path / "dead_man_deleted.txt").exists()
    finally:
        sleeper.kill()


# ----------------------------------------------------------------------------------------------------------------------------------
# Step definitions
# ----------------------------------------------------------------------------------------------------------------------------------

EXPECTED_STEP_ORDER = [
    "prepare_run_folders_in_container_work_tree", "install_vllm_environment", "install_fine_tuning_environment",
    "generation_phases_1_to_4", "training_gate", "build_training_examples_and_fine_tuning_run_folders", "fine_tuning_smoke_run", "check_smoke_fine_tuning_run", "fine_tuning_full_run",
    "check_full_fine_tuning_run", "attach_trained_adapter_to_evaluation_run_folders", "evaluation_generation_held_out",
    "evaluation_generation_betley", "evaluation_generation_mmlu_pro", "score_mmlu_pro_answers",
    "sycophancy_harmbench_sage_eval_generation_and_classification"]


def chain_paths(tmp_path):
    return definitions.ChainPaths(tmp_path / "work", tmp_path / "work" / "run_folder")


def test_step_definitions_have_the_required_order_and_every_dependency_comes_first(tmp_path):
    steps = definitions.build_chain_steps(chain_paths(tmp_path), {})
    assert [step.name for step in steps] == EXPECTED_STEP_ORDER
    names = [step.name for step in steps]
    for step in steps:
        assert all(names.index(other) < names.index(step.name) for other in step.depends_on + step.run_after)
    assert not any("gemma_judging" in name or "phases_5_and_6" in name for name in names)  # nothing is judged on the pod
    by_name = {step.name: step for step in steps}
    assert "check_full_fine_tuning_run" not in by_name["evaluation_generation_held_out"].depends_on  # base evaluations survive a training failure
    assert "check_full_fine_tuning_run" in by_name["attach_trained_adapter_to_evaluation_run_folders"].run_after
    assert by_name["install_vllm_environment"].marker_location == "container" and by_name["fine_tuning_full_run"].marker_location == "volume"


def command_of(step):
    return [str(part) for part in (step.command() if callable(step.command) else step.command)]


def test_sage_eval_is_skipped_and_recorded_when_the_dataset_was_not_supplied(tmp_path):
    steps = {step.name: step for step in definitions.build_chain_steps(chain_paths(tmp_path), {"sage_eval_dataset_supplied": False})}
    step = steps["sycophancy_harmbench_sage_eval_generation_and_classification"]
    assert "--skip-judging" in command_of(step) and any("SAGE-Eval skipped" in note for note in step.notes)
    supplied = {step.name: step for step in definitions.build_chain_steps(chain_paths(tmp_path), {"sage_eval_dataset_supplied": True})}
    step = supplied["sycophancy_harmbench_sage_eval_generation_and_classification"]
    assert "--skip-judging" in command_of(step) and not step.notes  # the pod never judges: SAGE-Eval is judged by Haiku on the laptop afterwards


def test_evaluation_commands_run_base_entries_only_until_an_adapter_is_attached(tmp_path):
    paths = chain_paths(tmp_path)
    steps = {step.name: step for step in definitions.build_chain_steps(paths, {})}
    held_out = command_of(steps["evaluation_generation_held_out"])
    assert "--only-model-short-names" in held_out and "--keep-model-caches" in held_out
    paths.attachment_result_folder.mkdir(parents=True)
    (paths.attachment_result_folder / "attach_trained_adapter_result_2026-11-13_23-00-00.json").write_text(json.dumps({"adapter_available": True}))
    assert "--only-model-short-names" not in command_of(steps["evaluation_generation_held_out"])
    assert steps["evaluation_generation_held_out"].signature() == "adapter_available=True"


def test_generation_phases_command_selects_haiku_judging_the_heartbeat_the_decision_file_and_the_idle_time_evaluation_folders(tmp_path):
    paths = chain_paths(tmp_path)
    command = command_of({step.name: step for step in definitions.build_chain_steps(paths, {})}["generation_phases_1_to_4"])
    assert command[command.index("--only-phases") + 1:command.index("--only-phases") + 5] == ["1", "2", "3", "4"]
    assert command[command.index("--haiku-judging-folder") + 1] == str(paths.haiku_judging_inputs)
    assert str(paths.haiku_judging_inputs).endswith("instructions_from_launcher/haiku_judging")  # written on the volume, never synced or restored
    assert command[command.index("--launcher-heartbeat-file") + 1] == str(paths.heartbeat_file)
    assert command[command.index("--launcher-heartbeat-stale-seconds") + 1] == "1200"
    assert command[command.index("--launcher-lost-decision-file") + 1] == str(paths.gate_decisions / definitions.LAUNCHER_LOST_DECISION_FILE_NAME)
    folders = command[command.index("--idle-time-evaluation-run-folders") + 1:]
    assert folders == [f"{kind}={paths.evaluation[kind]}" for kind in ("held_out", "betley", "mmlu_pro")]  # adapter-free generation only
    assert "--gemma-judging" not in " ".join(command)


def test_environment_install_signature_changes_with_the_requirements_file(tmp_path):
    paths = chain_paths(tmp_path)
    requirements = paths.working_directory / definitions.VLLM_REQUIREMENTS
    requirements.parent.mkdir(parents=True)
    requirements.write_text("vllm==1\n")
    step = {step.name: step for step in definitions.build_chain_steps(paths, {})}["install_vllm_environment"]
    before = step.signature()
    requirements.write_text("vllm==2\n")
    assert step.signature() != before


def test_heartbeat_file_and_inputs_are_not_inside_the_pulled_outputs(tmp_path):
    paths = chain_paths(tmp_path)
    assert paths.outputs not in paths.heartbeat_file.parents and paths.outputs not in paths.inputs.parents


def test_adapters_and_resume_checkpoints_are_on_the_volume_and_listed_in_the_summary(tmp_path):
    volume = tmp_path / "volume"  # stands in for the network volume mounted at /workspace
    paths = definitions.ChainPaths(volume / "work", volume / "work" / "run_folder")
    for run_folder in (paths.smoke_run, paths.full_run):
        outputs = run_folder / "training_outputs"
        for folder in ("final_adapter", "final_adapter_for_vllm", "adapter_at_end_of_epoch/epoch_1", "checkpoints/checkpoint_step_000100"):
            (outputs / folder).mkdir(parents=True)
            (outputs / folder / "adapter_model.safetensors").write_bytes(b"weights")
    elsewhere = tmp_path / "container_disk"
    listed = definitions.list_trained_adapter_and_checkpoint_paths_on_the_volume(paths, str(volume))
    assert len(listed) == 8 and all(item["on_the_volume"] for item in listed) and {item["kind"] for item in listed} == {"adapter", "resume checkpoint"}
    assert any(item["path"].endswith("fine_tuning_full_run/training_outputs/final_adapter_for_vllm") for item in listed)
    assert not any(item["on_the_volume"] for item in definitions.list_trained_adapter_and_checkpoint_paths_on_the_volume(paths, str(elsewhere)))
    runner = make_runner(tmp_path, [trace_step("one", tmp_path / "trace.txt")],
                         summary_extra_function=lambda: {"paths_on_the_network_volume_of_trained_adapters_and_resume_checkpoints": listed})
    runner.run()
    summary = json.loads(next((tmp_path / "summary").glob("chain_summary_*.json")).read_text())
    assert len(summary["paths_on_the_network_volume_of_trained_adapters_and_resume_checkpoints"]) == 8
    assert "final_adapter_for_vllm" in next((tmp_path / "summary").glob("chain_summary_*.md")).read_text()


def test_orchestrator_refuses_to_run_when_the_run_folder_is_not_on_the_volume(tmp_path):
    work = tmp_path / "elsewhere" / "work"
    (work / "run_folder").mkdir(parents=True)
    (work / "run_folder" / "run_configuration_overnight_chain_x.json").write_text("{}")
    orchestrator = HERE / "run_overnight_treatment_arm_chain_on_pod.py"
    completed = subprocess.run([PYTHON, str(orchestrator), "--working-directory", str(work), "--seconds-until-hard-limit", "100",
                                "--volume-mount-path", str(tmp_path / "volume"), "--no-pod-self-deletion"], capture_output=True, text=True)
    assert completed.returncode != 0 and "not under the network volume mount" in (completed.stdout + completed.stderr)


# ----------------------------------------------------------------------------------------------------------------------------------
# Helper scripts
# ----------------------------------------------------------------------------------------------------------------------------------

def make_finished_training_run(folder, loss=1.2, key_problems=(), with_key_check=True):
    outputs = folder / "training_outputs"
    (outputs / "final_adapter").mkdir(parents=True)
    (outputs / "final_adapter_for_vllm").mkdir()
    for adapter in ("final_adapter", "final_adapter_for_vllm"):
        (outputs / adapter / "adapter_config.json").write_text("{}")
        (outputs / adapter / "adapter_model.safetensors").write_bytes(b"trained weights")
    (outputs / "training_step_metrics.jsonl").write_text("".join(json.dumps({"optimizer_step": step, "loss": loss}) + "\n" for step in (1, 2, 3)))
    (outputs / "training_summary_2026-11-13_23-00-00.json").write_text(json.dumps({"final_loss_mean_of_last_ten_steps": loss, "optimizer_steps_total": 3}))
    if with_key_check:
        (outputs / "adapter_key_check_for_vllm_lora_request.json").write_text(json.dumps({"problems": list(key_problems)}))


def test_fine_tuning_check_passes_a_finished_run_and_fails_each_defect(tmp_path):
    good = tmp_path / "good"
    make_finished_training_run(good)
    assert fine_tuning_check.main(["--run-folder", str(good)]) == 0
    for name, kwargs in (("nan_loss", {"loss": float("nan")}), ("key_problems", {"key_problems": ["key names the vision tower"]}),
                         ("no_key_check", {"with_key_check": False})):
        folder = tmp_path / name
        make_finished_training_run(folder, **kwargs)
        assert fine_tuning_check.main(["--run-folder", str(folder)]) == 1, name
    unfinished = tmp_path / "unfinished"
    (unfinished / "training_outputs").mkdir(parents=True)
    assert fine_tuning_check.main(["--run-folder", str(unfinished)]) == 1
    missing_adapter = tmp_path / "missing_adapter"
    make_finished_training_run(missing_adapter)
    (missing_adapter / "training_outputs" / "final_adapter" / "adapter_model.safetensors").unlink()
    assert fine_tuning_check.main(["--run-folder", str(missing_adapter)]) == 1


def make_prebuilt_evaluation_run_folder(folder):
    adapter_folder = folder / "generation_inputs" / "lora_adapters" / "treatment_arm"
    adapter_folder.mkdir(parents=True)
    (adapter_folder / "adapter_config.json").write_text("placeholder")
    (adapter_folder / "adapter_model.safetensors").write_bytes(b"placeholder")
    base = {"model_short_name": "gemma_4_31b_it", "is_lora_fine_tuned": False}
    adapter = {"model_short_name": "gemma_4_31b_it_with_lora_treatment_arm", "is_lora_fine_tuned": True, "lora_adapter_name": "treatment_arm",
               "lora_adapter": {"path_inside_run_folder": "generation_inputs/lora_adapters/treatment_arm", "adapter_weights_sha256": "placeholder hash"}}
    (folder / "run_configuration_2026-11-13_20-00.json").write_text(json.dumps(
        {"models": [base, adapter], "model_order": [base["model_short_name"], adapter["model_short_name"]], "conditions": [{"condition": "thinking_off", "output_cap": 8192}],
         "message_layout": "layout", "purpose": "purpose"}))


def test_attach_copies_the_trained_adapter_over_the_placeholder_and_updates_hashes(tmp_path):
    training, evaluation = tmp_path / "training", tmp_path / "evaluation"
    make_finished_training_run(training)
    fine_tuning_check.check_fine_tuning_run(training)
    make_prebuilt_evaluation_run_folder(evaluation)
    result = attach.attach_trained_adapter_or_record_that_none_is_available([evaluation], training, "treatment_arm", tmp_path / "result")
    assert result["adapter_available"] is True
    placed = evaluation / "generation_inputs" / "lora_adapters" / "treatment_arm" / "adapter_model.safetensors"
    assert placed.read_bytes() == b"trained weights"
    configuration = json.loads(next(evaluation.glob("run_configuration_*.json")).read_text())
    block = configuration["models"][1]["lora_adapter"]
    assert block["adapter_weights_sha256"] == attach.sha256_of_file(placed) != "placeholder hash"
    attach.attach_trained_adapter_or_record_that_none_is_available([evaluation], training, "treatment_arm", tmp_path / "result")  # harmless twice


def test_attach_records_that_no_adapter_is_available_when_training_failed_and_changes_nothing(tmp_path):
    training, evaluation = tmp_path / "training", tmp_path / "evaluation"
    make_finished_training_run(training, loss=float("nan"))
    fine_tuning_check.check_fine_tuning_run(training)
    make_prebuilt_evaluation_run_folder(evaluation)
    before = next(evaluation.glob("run_configuration_*.json")).read_text()
    result = attach.attach_trained_adapter_or_record_that_none_is_available([evaluation], training, "treatment_arm", tmp_path / "result")
    assert result["adapter_available"] is False and next(evaluation.glob("run_configuration_*.json")).read_text() == before
    never_trained = tmp_path / "never_trained"
    never_trained.mkdir()
    assert attach.attach_trained_adapter_or_record_that_none_is_available([evaluation], never_trained, "treatment_arm", tmp_path / "result2")["adapter_available"] is False


def test_copying_prebuilt_inputs_never_overwrites_existing_files(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    (source / "run_folder_a").mkdir(parents=True)
    (source / "run_folder_a" / "prompts.jsonl").write_text("from the upload")
    (source / "run_folder_a" / "new.txt").write_text("new")
    (destination / "run_folder_a").mkdir(parents=True)
    (destination / "run_folder_a" / "prompts.jsonl").write_text("answers already made on the volume")
    assert copier.copy_without_overwriting(source, destination) == (1, 1)
    assert (destination / "run_folder_a" / "prompts.jsonl").read_text() == "answers already made on the volume"
    assert (destination / "run_folder_a" / "new.txt").read_text() == "new"


def test_system_prompt_condition_is_added_once_with_the_first_conditions_output_cap(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "run_configuration_2026.json").write_text(json.dumps({"conditions": [{"condition": "thinking_off", "output_cap": 777}], "message_layout": "m", "purpose": "p"}))
    fake_builder = lambda: {"condition": "with_prompt", "output_cap": 1, "system_prompt_text": "fake"}  # noqa: E731
    conditions = add_condition.add_system_prompt_condition(folder, fake_builder)
    assert [c["condition"] for c in conditions] == ["thinking_off", "with_prompt"] and conditions[1]["output_cap"] == 777
    with pytest.raises(SystemExit):
        add_condition.add_system_prompt_condition(folder, fake_builder)  # not a single-condition run any more


def fake_builder_runner(tmp_path):
    """Stands in for the four real builders: creates one run folder with one run configuration inside the --runs-root they are given."""
    calls = []

    def run(command):
        calls.append(command)
        runs_root = Path(command[command.index("--runs-root") + 1])
        kind = Path(command[1]).name
        folder = runs_root / f"{kind}_gemma_2026-11-13_20-00"
        (folder / "generation_inputs").mkdir(parents=True)
        (folder / "run_configuration_2026.json").write_text(json.dumps(
            {"conditions": [{"condition": "thinking_off", "output_cap": 100}], "message_layout": "m", "purpose": "p", "models": []}))
        if "--sage-eval-full-dataset-json" in command:
            (folder / "generation_inputs" / "sage_prompts.jsonl").write_text("fake placeholder content")
    return run, calls


def make_prebuild_arguments(tmp_path, runs_root, sage=None):
    generation = tmp_path / "generation_run_folder"
    (generation / "generation_inputs").mkdir(parents=True)
    (generation / "run_configuration_2026.json").write_text("{}")
    held_out = tmp_path / "held_out_set.jsonl"
    held_out.write_text("{}\n")
    arguments = ["--treatment-arm-generation-run-folder", str(generation), "--held-out-set-file", str(held_out), "--runs-root", str(runs_root)]
    if sage:
        arguments += ["--sage-eval-full-dataset-json", str(sage)]
    return prebuild.parse_arguments(arguments)


def test_prebuild_assembles_the_overnight_run_folder_with_placeholder_adapters_and_two_conditions(tmp_path):
    run, calls = fake_builder_runner(tmp_path)
    fake_condition = lambda: {"condition": "with_prompt", "output_cap": 1}  # noqa: E731
    run_folder = prebuild.build_run_folder(make_prebuild_arguments(tmp_path, tmp_path / "runs"), run, fake_condition)
    inputs = run_folder / "overnight_inputs" / "run_folders"
    assert (inputs / "treatment_arm_generation" / "run_configuration_2026.json").exists()
    assert sorted(path.name for path in (inputs / "evaluation").iterdir()) == sorted(definitions.EVALUATION_KINDS)
    assert len(calls) == 4 and all(f"--adapter treatment_arm=" in " ".join(command) or "treatment_arm=" in " ".join(command) for command in calls)
    for kind in ("held_out", "betley", "mmlu_pro"):
        configuration = json.loads(next((inputs / "evaluation" / kind).glob("run_configuration_*.json")).read_text())
        assert len(configuration["conditions"]) == 2
    assert len(json.loads(next((inputs / "evaluation" / "sycophancy_harmbench_sage_eval").glob("run_configuration_*.json")).read_text())["conditions"]) == 1
    chain_configuration = json.loads(next(run_folder.glob("run_configuration_overnight_chain_*.json")).read_text())
    assert chain_configuration["sage_eval_dataset_supplied"] is False
    assert any("--samples-per-question" in command and "100" in command for command in calls)
    assert not any("--sage-eval-full-dataset-json" in command for command in calls)


def test_prebuild_refuses_to_write_sage_eval_content_outside_the_data_folder(tmp_path):
    sage = tmp_path / "full_dataset.json"
    sage.write_text("fake placeholder content")
    run, _ = fake_builder_runner(tmp_path)
    with pytest.raises(SystemExit, match="not under"):
        prebuild.build_run_folder(make_prebuild_arguments(tmp_path, tmp_path / "runs", sage), run, lambda: {})
    assert not (tmp_path / "runs").exists()


def test_prebuild_with_sage_inside_data_passes_the_dataset_only_to_the_sycophancy_builder(tmp_path):
    sage = tmp_path / "full_dataset.json"
    sage.write_text("fake placeholder content")
    run, calls = fake_builder_runner(tmp_path)
    runs_root = REPOSITORY_ROOT / "data" / "overnight_chain_unit_test_scratch" / tmp_path.name
    try:
        run_folder = prebuild.build_run_folder(make_prebuild_arguments(tmp_path, runs_root, sage), run, lambda: {"condition": "with_prompt", "output_cap": 1})
        with_sage = [command for command in calls if "--sage-eval-full-dataset-json" in command]
        assert len(with_sage) == 1 and "build_sycophancy_harmbench_and_sage_eval_evaluation_run_folder.py" in with_sage[0][1]
        assert json.loads(next(run_folder.glob("run_configuration_overnight_chain_*.json")).read_text())["sage_eval_dataset_supplied"] is True
    finally:
        import shutil
        shutil.rmtree(REPOSITORY_ROOT / "data" / "overnight_chain_unit_test_scratch", ignore_errors=True)


# ----------------------------------------------------------------------------------------------------------------------------------
# Network volume client calls and data center choice (fake HTTP)
# ----------------------------------------------------------------------------------------------------------------------------------

def test_data_center_choice_prefers_the_most_available_h200_and_ignores_none():
    data_centers = [
        {"id": "AAA-1", "networkVolumeTypes": ["STANDARD"], "gpuAvailability": [{"id": "NVIDIA H200", "name": "H200", "availability": "LOW"}]},
        {"id": "BBB-1", "networkVolumeTypes": ["STANDARD"], "gpuAvailability": [{"id": "NVIDIA H200", "name": "H200", "availability": "HIGH"}]},
        {"id": "CCC-1", "networkVolumeTypes": ["STANDARD"], "gpuAvailability": [{"id": "NVIDIA H200", "name": "H200", "availability": "NONE"}]},
        {"id": "DDD-1", "networkVolumeTypes": [], "gpuAvailability": [{"id": "NVIDIA H200", "name": "H200", "availability": "HIGH"}]},
        {"id": "EEE-1", "networkVolumeTypes": ["STANDARD"], "gpuAvailability": [{"id": "NVIDIA H100 80GB HBM3", "name": "H100 SXM", "availability": "HIGH"}]},
        {"id": "FFF-1", "networkVolumeTypes": ["STANDARD"]}]
    assert volume_calls.data_centers_with_gpu_in_stock(data_centers, "H200") == ["BBB-1", "AAA-1"]


def test_network_volume_creation_request_and_costs():
    assert volume_calls.build_network_volume_creation_request("name", 50, "EU-RO-1") == {"name": "name", "size": 50, "dataCenterId": "EU-RO-1"}
    assert volume_calls.monthly_storage_cost_dollars(50) == pytest.approx(3.5)
    assert "DELETE" in volume_calls.delete_network_volume_command_text("vol1") and "networkvolumes/vol1" in volume_calls.delete_network_volume_command_text("vol1")


# ----------------------------------------------------------------------------------------------------------------------------------
# Launcher with a fake RunPod client and a fake ssh transfer
# ----------------------------------------------------------------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeRunPodClient:
    def __init__(self, hourly_price=5.0, data_centers=None):
        self.calls, self.hourly_price, self.pod_deleted_at = [], hourly_price, None
        self.data_centers = data_centers if data_centers is not None else [
            {"id": "BBB-1", "networkVolumeTypes": ["STANDARD"], "gpuAvailability": [{"id": "NVIDIA H200", "name": "H200", "availability": "MEDIUM"}]}]
        self.pod_exists = True

    def list_data_centers_with_gpu_availability(self):
        self.calls.append(("list_data_centers", None))
        return self.data_centers

    def create_network_volume(self, name, size_gigabytes, data_center_id):
        self.calls.append(("create_network_volume", (name, size_gigabytes, data_center_id)))
        return {"id": "vol123", "name": name, "size": size_gigabytes, "dataCenterId": data_center_id}

    def get_network_volume(self, volume_id):
        self.calls.append(("get_network_volume", volume_id))
        return {"id": volume_id, "name": "existing", "size": 50, "dataCenterId": "BBB-1"}

    def create_pod(self, creation_request):
        self.calls.append(("create_pod", creation_request))
        return {"id": "fakepod1", "costPerHr": self.hourly_price, "gpu": {"displayName": "H200"}}

    def get_pod(self, pod_id):
        self.calls.append(("get_pod", pod_id))
        if not self.pod_exists:
            raise RuntimeError("RunPod get pod failed (404): not found")
        return {"id": pod_id, "costPerHr": self.hourly_price, "publicIp": "1.2.3.4", "portMappings": {"22": 4000}}

    def delete_pod(self, pod_id):
        self.calls.append(("delete_pod", pod_id))

    def names(self):
        return [name for name, _ in self.calls]


class FakeOvernightTransfer:
    """Fake ssh layer: the chain finishes `chain_finishes_after_seconds` after it was started (writes its finished flag), or the pod deletes itself."""

    def __init__(self, clock, client, chain_finishes_after_seconds=None, pod_deletes_itself_after_seconds=None, chain_process_vanishes_after_seconds=None):
        self.clock, self.client, self.commands, self.pulls, self.final_pulls = clock, client, [], [], 0
        self.host = "1.2.3.4"
        self.chain_finishes_after_seconds, self.pod_deletes_itself_after_seconds = chain_finishes_after_seconds, pod_deletes_itself_after_seconds
        self.chain_process_vanishes_after_seconds = chain_process_vanishes_after_seconds
        self.chain_started_at, self.uploads = None, 0
        self.heartbeats, self.instructions = [], []
        self.uploaded_texts, self.pull_gate = {}, None  # pull_gate: a function returning True when the pull may go on (real-time threads)

    def elapsed(self):
        return self.clock.time() - 1_000_000.0

    def wait_until_ready(self, pod, deadline, get_pod_function, sleep_function):
        return None

    def upload_project_files(self):
        self.uploads += 1

    def run_remote_command(self, command, input_text=None, timeout_seconds=None):
        self.commands.append(command)
        if (self.pod_deletes_itself_after_seconds is not None and self.elapsed() >= self.pod_deletes_itself_after_seconds):
            self.client.pod_exists = False
            return 255, "ssh: connect to host failed"
        if "launcher_heartbeat_epoch_seconds.txt" in command and command.startswith("echo"):
            self.heartbeats.append(self.clock.time())
            return 0, ""
        if input_text is not None and ".part" in command and "mv -f" in command:
            self.uploaded_texts.setdefault(command.split("mv -f ")[1].split()[1].rsplit("/", 1)[1], []).append((command, input_text))
            return 0, ""
        if "instructions_from_launcher" in command and ": >" in command:
            self.instructions.append(command)
            return 0, ""
        if "run_overnight_treatment_arm_chain_on_pod" in command and command.startswith("nohup"):
            self.chain_started_at = self.clock.time()
            return 0, ""
        if "chain_finished_flag.json" in command and command.startswith("cat"):
            done = (self.chain_finishes_after_seconds is not None and self.chain_started_at is not None
                    and self.clock.time() - self.chain_started_at >= self.chain_finishes_after_seconds)
            return (0, json.dumps({"ended_because": "chain_ran_to_the_end", "overall_status": "all_steps_completed"})) if done else (1, "")
        if command.startswith("pgrep"):
            vanished = (self.chain_process_vanishes_after_seconds is not None and self.chain_started_at is not None
                        and self.clock.time() - self.chain_started_at >= self.chain_process_vanishes_after_seconds)
            return (1, "") if vanished else (0, "")
        if "scorer_exit_code.txt" in command:
            return 1, ""
        if command.startswith("tail -n"):
            return 0, "chain output"
        return 0, "3\nlatest progress line"

    def pull_final_copy_and_verify(self, logs_folder_name, timeout_seconds):
        self.final_pulls += 1
        return 7

    def pull_scoring_outputs(self, pull_everything):
        self.pulls.append(pull_everything)
        if self.pull_gate:
            import time as real_time
            deadline = real_time.time() + 10
            while not self.pull_gate() and real_time.time() < deadline:
                real_time.sleep(0.01)


SECRETS = {"RUNPOD_API_KEY": "rp_test", "HF_TOKEN": "hf_test", "RUNPOD_COST_CENTER_NAME": "test-cost-center"}
PROFILE_NAMES = ["REMOTE_WORKING_DIRECTORY", "REMOTE_RUN_FOLDER", "INPUTS_FOLDER_NAME", "OUTPUTS_FOLDER_NAME", "POD_NAME_PREFIX", "SCRIPT_FOLDER_NAME",
                 "SCRIPT_FOLDER_NAMES_TO_UPLOAD", "SCORER_SCRIPT_RELATIVE_PATH", "SCORER_PROCESS_PATTERN", "REQUIREMENTS_FILE_RELATIVE_PATH",
                 "PROGRESS_STATUS_SHELL_TEMPLATE", "PROGRESS_STATUS_UNIT_NAME", "MINIMUM_LOCAL_FREE_DISK_BYTES", "DEFAULT_CONTAINER_DISK_GIGABYTES",
                 "PULL_EXCLUDED_PATH_PATTERNS", "PROGRESS_REPORT_INTERVAL_SECONDS", "FINAL_COPY_FILE_LIST_SHELL", "build_scorer_command_line", "build_pod_creation_request", "PodSession",
                 "SshTarTransfer", "RunPodRestClient", "print_dry_run"]


@pytest.fixture
def overnight_profile(monkeypatch):
    # The pulls check the laptop's real free disk; report plenty, so the tests do not depend on the machine running them
    monkeypatch.setattr(pod_launcher.shutil, "disk_usage", lambda path: type("DiskUsage", (), {"free": 10 ** 15, "total": 10 ** 15, "used": 0})())
    saved = {name: getattr(pod_launcher, name) for name in PROFILE_NAMES}
    saved_settings = dict(overnight_launcher.OVERNIGHT_LAUNCH_SETTINGS)
    saved_record = dict(overnight_launcher.LAUNCH_RECORD)
    saved_haiku_settings = dict(overnight_launcher.HAIKU_JUDGING_SETTINGS)
    overnight_launcher.HAIKU_JUDGING_SETTINGS["enabled"] = False  # the Haiku judging thread has its own tests
    overnight_launcher.apply_job_profile()
    yield
    for name, value in saved.items():
        setattr(pod_launcher, name, value)
    overnight_launcher.OVERNIGHT_LAUNCH_SETTINGS.update(saved_settings)
    overnight_launcher.LAUNCH_RECORD.update(saved_record)
    overnight_launcher.HAIKU_JUDGING_SETTINGS.update(saved_haiku_settings)


@pytest.fixture
def overnight_run_folder(tmp_path):
    folder = tmp_path / "overnight_treatment_arm_chain_2026-11-13_21-00"
    (folder / "overnight_inputs" / "run_folders").mkdir(parents=True)
    (folder / "run_configuration_overnight_chain_2026-11-13_21-00.json").write_text(json.dumps({"sage_eval_dataset_supplied": False}))
    return folder


def launcher_arguments(run_folder, *extra):
    return pod_launcher.parse_arguments(["--run-folder", str(run_folder), "--maximum-running-hours", "12", "--maximum-dollars", "65",
                                         "--gpu-type-ids", "NVIDIA H200", "--ssh-private-key-path", str(run_folder / "no_such_key"), *extra])


def run_fake_launcher(overnight_run_folder, clock, client, transfer, *extra, confirmed=True):
    if confirmed:
        (overnight_run_folder / pod_launcher.COST_CENTER_CONFIRMATION_FILE_NAME).write_text("confirmed")
    overnight_launcher.create_or_reuse_network_volume(launcher_arguments(overnight_run_folder), SECRETS, overnight_run_folder, client)
    return pod_launcher.run_launcher(launcher_arguments(overnight_run_folder, *extra), client=client, transfer=transfer, secrets=SECRETS,
                                     clock=clock.time, sleep=clock.sleep, confirmation_source=type("Source", (), {"confirmed": False})())


def test_volume_is_created_in_the_data_center_with_an_h200_and_the_pod_is_attached_to_it(overnight_profile, overnight_run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeOvernightTransfer(clock, client, chain_finishes_after_seconds=400)
    assert run_fake_launcher(overnight_run_folder, clock, client, transfer) == 0
    assert client.names()[:3] == ["list_data_centers", "create_network_volume", "create_pod"]
    volume_call = next(argument for name, argument in client.calls if name == "create_network_volume")
    assert volume_call[1:] == (20, "BBB-1") and "overnight-treatment-arm-chain-volume" in volume_call[0]
    pod_request = next(argument for name, argument in client.calls if name == "create_pod")
    assert pod_request["networkVolumeId"] == "vol123" and pod_request["dataCenterIds"] == ["BBB-1"] and pod_request["gpuTypeIds"] == ["NVIDIA H200"]
    assert "volumeInGb" not in pod_request and pod_request["volumeMountPath"] == "/workspace" and pod_request["containerDiskInGb"] == 200
    assert pod_request["name"].startswith("overnight-treatment-arm-chain-overnight_treatment_arm_chain_")
    assert pod_request["env"]["HF_HOME"] == "/root/huggingface_cache"  # weights on the container disk, not the volume
    record = json.loads(next(overnight_run_folder.glob("network_volume_record_*.json")).read_text())
    assert record["id"] == "vol123" and record["estimated_monthly_cost_dollars"] == 1.4 and "DELETE" in record["delete_command_not_run"]
    cost_log = json.loads(next(overnight_run_folder.glob("pod_cost_log_*.json")).read_text())
    assert cost_log["network_volume_id"] == "vol123" and cost_log["cost_center_name"] == "test-cost-center"
    assert cost_log["cost_center_assignment_confirmed_by_user"] is True and cost_log["maximum_running_hours"] == 12 and cost_log["maximum_dollars"] == 65


def test_no_volume_is_created_when_no_data_center_has_an_h200(overnight_profile, overnight_run_folder):
    client = FakeRunPodClient(data_centers=[{"id": "X", "networkVolumeTypes": ["STANDARD"], "gpuAvailability": [{"id": "NVIDIA H200", "availability": "NONE"}]}])
    with pytest.raises(SystemExit, match="No data center"):
        overnight_launcher.create_or_reuse_network_volume(launcher_arguments(overnight_run_folder), SECRETS, overnight_run_folder, client)
    assert "create_network_volume" not in client.names()


def test_an_existing_volume_is_reused_without_creating_another(overnight_profile, overnight_run_folder):
    overnight_launcher.OVERNIGHT_LAUNCH_SETTINGS.update(network_volume_id="existing9", data_center_id=None)
    client = FakeRunPodClient()
    record = overnight_launcher.create_or_reuse_network_volume(launcher_arguments(overnight_run_folder), SECRETS, overnight_run_folder, client)
    assert record["id"] == "existing9" and overnight_launcher.OVERNIGHT_LAUNCH_SETTINGS["data_center_id"] == "BBB-1"
    assert client.names() == ["get_network_volume"]


def test_launcher_writes_a_heartbeat_at_start_and_pulls_and_writes_one_every_60_seconds(overnight_profile, overnight_run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeOvernightTransfer(clock, client, chain_finishes_after_seconds=1000)
    run_fake_launcher(overnight_run_folder, clock, client, transfer)
    assert len(transfer.heartbeats) >= 1 + 1000 // 100 - 1
    gaps = [later - earlier for earlier, later in zip(transfer.heartbeats[1:], transfer.heartbeats[2:])]
    assert gaps and all(55 <= gap <= 100 for gap in gaps[:-1]) and len(transfer.pulls) >= 10
    chain_command = next(command for command in transfer.commands if command.startswith("nohup") and "run_overnight_treatment_arm_chain_on_pod" in command)
    assert "--seconds-until-hard-limit" in chain_command and "--pod-hourly-price-dollars 5.0" in chain_command


def test_chain_end_with_the_pod_still_up_gets_a_verified_final_pull_then_the_pod_is_deleted_by_the_launcher_too(overnight_profile, overnight_run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeOvernightTransfer(clock, client, chain_finishes_after_seconds=500)
    assert run_fake_launcher(overnight_run_folder, clock, client, transfer) == 0
    assert transfer.final_pulls == 1 and "delete_pod" in client.names()
    assert any("launcher_final_pull_complete.txt" in command for command in transfer.commands)  # the acknowledgement the chain waits for
    cost_log = json.loads(next(overnight_run_folder.glob("pod_cost_log_*.json")).read_text())
    assert cost_log["termination_reason"] == "scorer_finished" and cost_log["pod_terminated"] is True and cost_log["final_pull_verified"] is True


def test_pod_that_deleted_itself_is_recognised_final_pull_is_skipped_and_retrieval_is_pointed_to(overnight_profile, overnight_run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    summaries = overnight_run_folder / "overnight_outputs" / "chain_summary_and_progress"
    summaries.mkdir(parents=True)
    (summaries / "chain_summary_2026-11-14_03-00-00.json").write_text("{}")
    transfer = FakeOvernightTransfer(clock, client, pod_deletes_itself_after_seconds=2000)
    assert run_fake_launcher(overnight_run_folder, clock, client, transfer) == 0
    assert transfer.final_pulls == 0
    cost_log = json.loads(next(overnight_run_folder.glob("pod_cost_log_*.json")).read_text())
    assert cost_log["pod_deleted_itself"] is True and cost_log["termination_reason"] == "scorer_finished"
    log_text = next(overnight_run_folder.glob("pod_progress_log_*.txt")).read_text()
    assert "no final pull is possible" in log_text and "retrieve_overnight_chain_outputs_from_network_volume_with_cheapest_cpu_pod.py" in log_text


def test_launcher_survives_a_lost_connection_and_does_not_delete_the_pod_early(overnight_profile, overnight_run_folder):
    """ssh timing out for hours must not end the launcher with an 'exception' that deletes the pod: the transfer wrapper turns it into exit code 255."""
    class TimingOutBase:
        def run_remote_command(self, command, input_text=None, timeout_seconds=None):
            raise subprocess.TimeoutExpired(command, 5)
    wrapper = type("Wrapper", (overnight_launcher.ConnectionLossTolerantTransfer,), {})
    instance = wrapper.__new__(wrapper)
    ssh_transfer_class = overnight_launcher.ConnectionLossTolerantTransfer.__mro__[1]  # pod_launcher.SshTarTransfer, replaced by the profile
    original = ssh_transfer_class.run_remote_command
    ssh_transfer_class.run_remote_command = TimingOutBase.run_remote_command
    try:
        exit_code, text = instance.run_remote_command("echo hi")
    finally:
        ssh_transfer_class.run_remote_command = original
    assert exit_code == 255 and "unreachable" in text


def test_a_vanished_chain_process_waits_for_an_instruction_instead_of_deleting_at_once(overnight_profile, overnight_run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeOvernightTransfer(clock, client, chain_process_vanishes_after_seconds=300)
    assert run_fake_launcher(overnight_run_folder, clock, client, transfer, "--failure-wait-minutes", "20") == 1
    log_text = next(overnight_run_folder.glob("pod_progress_log_*.txt")).read_text()
    assert "WAITING FOR A FIX" in log_text and "scorer_vanished_without_exit_code" in log_text
    assert transfer.final_pulls >= 1  # everything is copied back before waiting


def test_hours_limit_still_deletes_the_pod_with_the_overnight_defaults(overnight_profile, overnight_run_folder):
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=3.0)
    transfer = FakeOvernightTransfer(clock, client)  # the chain never finishes
    assert run_fake_launcher(overnight_run_folder, clock, client, transfer) == 1
    cost_log = json.loads(next(overnight_run_folder.glob("pod_cost_log_*.json")).read_text())
    assert cost_log["termination_reason"] == "running_hours_limit_reached" and "delete_pod" in client.names()
    assert 12 * 3600 <= clock.time() - 1_000_000.0 < 12 * 3600 + 600
    watchdog = next(command for command in transfer.commands if "sleep" in command and "RUNPOD_POD_ID" in command)
    assert f"sleep {12 * 3600 + pod_launcher.WATCHDOG_GRACE_SECONDS}" in watchdog


def test_dollar_limit_still_deletes_the_pod_before_the_hours_limit_at_a_high_price(overnight_profile, overnight_run_folder):
    clock, client = FakeClock(), FakeRunPodClient(hourly_price=10.0)  # 65 dollars is reached after 6.5 hours
    transfer = FakeOvernightTransfer(clock, client)
    assert run_fake_launcher(overnight_run_folder, clock, client, transfer) == 1
    cost_log = json.loads(next(overnight_run_folder.glob("pod_cost_log_*.json")).read_text())
    assert cost_log["termination_reason"] == "dollar_limit_reached" and clock.time() - 1_000_000.0 < 7 * 3600
    chain_command = next(command for command in transfer.commands if command.startswith("nohup") and "run_overnight_treatment_arm_chain_on_pod" in command)
    assert f"--seconds-until-hard-limit {int(6.5 * 3600)}" in chain_command or "--seconds-until-hard-limit 23" in chain_command


def test_instruction_files_in_the_run_folder_are_forwarded_to_the_pod(overnight_profile, overnight_run_folder):
    clock, client = FakeClock(), FakeRunPodClient()
    (overnight_run_folder / "pod_instruction_skip_step_fine_tuning_full_run.txt").write_text("")
    (overnight_run_folder / "pod_instruction_retry_step_not valid name.txt").write_text("")
    transfer = FakeOvernightTransfer(clock, client, chain_finishes_after_seconds=500)
    run_fake_launcher(overnight_run_folder, clock, client, transfer)
    assert any("skip_step_fine_tuning_full_run.txt" in command for command in transfer.instructions)
    assert len(transfer.instructions) == 1
    assert list(overnight_run_folder.glob("pod_instruction_skip_step_fine_tuning_full_run.txt.forwarded_*"))


def test_overnight_upload_never_sends_the_local_outputs_folder(overnight_profile, overnight_run_folder, tmp_path):
    (overnight_run_folder / "overnight_outputs").mkdir()
    sent = []
    transfer = overnight_launcher.ConnectionLossTolerantTransfer("key", overnight_run_folder)
    transfer.run_remote_command = lambda command, **kwargs: (0, "")
    transfer._stream_local_tar_to_remote = lambda command, directory: sent.append(command)
    transfer.upload_project_files()
    assert "overnight_inputs" in sent[1] and "overnight_outputs" not in sent[1] and any(part.startswith("run_configuration_overnight") for part in sent[1])


def test_pulls_and_final_copy_leave_out_checkpoints_and_adapter_copies(overnight_profile):
    assert "*/checkpoints/*" in pod_launcher.FINAL_COPY_FILE_LIST_SHELL and "*/lora_adapters/*" in pod_launcher.FINAL_COPY_FILE_LIST_SHELL
    assert pod_launcher.PULL_EXCLUDED_PATH_PATTERNS == ["*/checkpoints/*", "*/lora_adapters/*", "*/instructions_from_launcher/haiku_judging/*",
                                                        "*/final_adapter/*", "*/final_adapter_for_vllm/*", "*/adapter_at_end_of_epoch/*"]
    assert "*/final_adapter/*" in pod_launcher.FINAL_COPY_FILE_LIST_SHELL  # adapters stay on the volume


def test_launcher_refuses_a_sage_run_folder_outside_the_data_folder(overnight_profile, overnight_run_folder, capsys):
    (overnight_run_folder / "run_configuration_overnight_chain_2026-11-13_21-00.json").write_text(json.dumps({"sage_eval_dataset_supplied": True}))
    with pytest.raises(SystemExit, match="data/"):
        overnight_launcher.main(["--run-folder", str(overnight_run_folder), "--dry-run"])


def test_dry_run_prints_volume_and_pod_requests_chain_limits_and_commands(overnight_profile, overnight_run_folder, capsys):
    assert overnight_launcher.main(["--run-folder", str(overnight_run_folder), "--dry-run"]) == 0
    output = capsys.readouterr().out
    for expected in ("POST https://rest.runpod.io/v1/networkvolumes", '"size": 20', "overnight-treatment-arm-chain-volume-", "networkVolumeId", "NVIDIA H200",
                     "Hard limits: 12 hours or 65 dollars", "install_vllm_environment", "generation_phases_1_to_4", "chain:    nohup bash -c",
                     "watchdog:", "SAGE-Eval dataset supplied at build: False", "launcher_heartbeat_epoch_seconds.txt", "0.07 dollars per GB per month",
                     "1.40 dollars per month", "retrieval script", "JUDGING ON THIS LAPTOP", "caffeinate -i", "rate-limit tier"):
        assert expected in output, expected
    assert not list(overnight_run_folder.glob("pod_cost_log_*.json"))  # a dry run makes no files and no calls


# ----------------------------------------------------------------------------------------------------------------------------------
# Retrieval script
# ----------------------------------------------------------------------------------------------------------------------------------

class FakeRetrievalTransfer:
    def __init__(self, fail=False):
        self.fail, self.pulled = fail, 0

    def wait_until_ready(self, pod, deadline, get_pod_function, sleep_function):
        return None

    def pull_final_copy_and_verify(self, logs_folder_name, timeout_seconds):
        self.pulled += 1
        if self.fail:
            raise RuntimeError("tar stream broke")
        return 42


def retrieval_arguments(tmp_path, *extra):
    return retrieval.parse_arguments(["--network-volume-id", "vol123", "--run-folder", str(tmp_path / "overnight_run"),
                                      "--ssh-private-key-path", str(tmp_path / "no_such_key"), *extra])


def run_retrieval(tmp_path, client, transfer, *extra):
    lines = []
    clock = FakeClock()
    code = retrieval.retrieve(retrieval_arguments(tmp_path, *extra), client, transfer, SECRETS, clock.time, clock.sleep, lines.append)
    return code, "\n".join(lines)


def test_retrieval_creates_a_cpu_pod_on_the_volume_pulls_verifies_deletes_the_pod_and_only_prints_the_volume_deletion(tmp_path):
    client, transfer = FakeRunPodClient(hourly_price=0.06), FakeRetrievalTransfer()
    client.requires_key = False
    code, output = run_retrieval(tmp_path, client, transfer)
    assert code == 0 and transfer.pulled == 1
    pod_request = next(argument for name, argument in client.calls if name == "create_pod")
    assert pod_request["computeType"] == "CPU" and pod_request["networkVolumeId"] == "vol123" and pod_request["dataCenterIds"] == ["BBB-1"]
    assert pod_request["cpuFlavorIds"][0] == "cpu3c" and "gpuTypeIds" not in pod_request
    assert client.names().count("delete_pod") == 1 and "delete_network_volume" not in client.names()
    assert "Verified 42 files" in output and "networkvolumes/vol123" in output and "NOT run by this script" in output
    log = json.loads(next((tmp_path / "overnight_run").glob("retrieval_pod_cost_log_*.json")).read_text())
    assert log["pod_terminated"] is True and log["termination_reason"] == "retrieved_and_verified" and log["cost_center_name"] == "test-cost-center"


def test_retrieval_deletes_the_pod_even_when_the_pull_fails_and_reports_failure(tmp_path):
    client, transfer = FakeRunPodClient(hourly_price=0.06), FakeRetrievalTransfer(fail=True)
    client.requires_key = False
    code, output = run_retrieval(tmp_path, client, transfer)
    assert code == 1 and client.names().count("delete_pod") == 1 and "tar stream broke" in output


def test_retrieval_refuses_an_unexpectedly_expensive_pod_and_deletes_it(tmp_path):
    client, transfer = FakeRunPodClient(hourly_price=4.0), FakeRetrievalTransfer()
    client.requires_key = False
    code, output = run_retrieval(tmp_path, client, transfer)
    assert code == 1 and transfer.pulled == 0 and client.names().count("delete_pod") == 1 and "above --maximum-hourly-price" in output


def test_retrieval_has_its_own_hard_limits_and_dry_run(tmp_path, overnight_profile, capsys):
    arguments = retrieval_arguments(tmp_path)
    assert (arguments.maximum_running_hours, arguments.maximum_dollars) == (1.0, 2.0)
    assert retrieval.main(["--network-volume-id", "vol123", "--run-folder", str(tmp_path / "run"), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Hard limits: 1 hours or 2 dollars" in output and "computeType" in output and "networkVolumeId" in output
    assert not (tmp_path / "run").exists()


# ----------------------------------------------------------------------------------------------------------------------------------
# Haiku judging on the laptop, inside the launcher
# ----------------------------------------------------------------------------------------------------------------------------------

def prepare_generation_folders_for_laptop_judging(run_folder, answer_texts):
    import test_haiku_judging_on_the_laptop_and_the_pod_waiting_for_it_without_any_api_or_pod_call as judging_tests
    inputs = run_folder / "overnight_inputs" / "run_folders" / "treatment_arm_generation"
    (inputs / "generation_inputs").mkdir(parents=True)
    (inputs / "generation_inputs" / "prompts.jsonl").write_text("".join(json.dumps({"prompt_gid": f"P{n}"}) + "\n" for n in range(len(answer_texts))))
    (inputs / "run_configuration_2026-11-13_00-00.json").write_text(json.dumps({
        "models": [{"model_short_name": "m"}], "prompts_file": {"path_inside_run_folder": "generation_inputs/prompts.jsonl"},
        "gemma_self_judging_and_regeneration": {"maximum_attempts_per_prompt": 4, "system_prompted_condition": judging_tests.pod_tests.SYSTEM_CONDITION}}))
    pulled = run_folder / "overnight_outputs" / "run_folders" / "treatment_arm_generation" / "generation_outputs"
    pulled.mkdir(parents=True)
    (pulled / f"answers__m__{judging_tests.pod_tests.SYSTEM_CONDITION}.jsonl").write_text("".join(
        json.dumps(judging_tests.pulled_answer(f"P{n}", 0, text)) + "\n" for n, text in enumerate(answer_texts)))
    return judging_tests


class FakeBackendForLauncher:
    def __init__(self, call_model):
        self.call_model, self.progress_line = call_model, lambda: "Haiku API: fake backend"


def test_the_launcher_judges_pulled_answers_on_the_laptop_and_uploads_the_three_files_atomically_to_the_pod_with_the_completion_file_last(
        overnight_profile, overnight_run_folder):
    judging_tests = prepare_generation_folders_for_laptop_judging(overnight_run_folder, ["Advice [[S:B777-P0:0]]", "Advice [[S:B777-P1:0]]"])
    fake_api = judging_tests.FakeApi({})
    overnight_launcher.HAIKU_JUDGING_SETTINGS.update(enabled=True, backend_factory=lambda folder, say: FakeBackendForLauncher(fake_api.call_model))
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeOvernightTransfer(clock, client, chain_finishes_after_seconds=400)
    transfer.pull_gate = lambda: "haiku_judging_complete_with_kept_answer_id_per_prompt.json" in transfer.uploaded_texts
    assert run_fake_launcher(overnight_run_folder, clock, client, transfer) == 0
    assert set(transfer.uploaded_texts) == {"haiku_verdicts_on_system_prompted_answers_for_pod.jsonl", "regeneration_requests_from_haiku_verdicts.json",
                                            "haiku_judging_complete_with_kept_answer_id_per_prompt.json"}
    command, text = transfer.uploaded_texts["haiku_judging_complete_with_kept_answer_id_per_prompt.json"][-1]
    assert command.startswith("mkdir -p ") and "instructions_from_launcher/haiku_judging" in command and ".part" in command and "mv -f" in command
    assert "ANTHROPIC" not in " ".join(transfer.commands)  # no key, and nothing of it, ever goes to the pod
    assert json.loads(text)["kept_answer_id_by_prompt_gid"].keys() == {"P0", "P1"}
    assert list((overnight_run_folder / "haiku_judging_of_treatment_arm_answers" / "judge_outputs").glob("judge_verdicts_*.jsonl"))


def test_a_dead_haiku_judging_thread_stops_the_heartbeat_so_the_pod_ends_the_chain_instead_of_waiting_for_ever(overnight_profile, overnight_run_folder, capsys):
    import threading
    prepare_generation_folders_for_laptop_judging(overnight_run_folder, ["Advice [[S:B777-P0:0]]"])
    import anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log as haiku_api
    died = threading.Event()

    def refuse_key(model, prompt, schema):
        died.set()
        raise haiku_api.AnthropicApiFatalError("the key was rejected. Stopping.")
    overnight_launcher.HAIKU_JUDGING_SETTINGS.update(enabled=True, backend_factory=lambda folder, say: FakeBackendForLauncher(refuse_key))
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeOvernightTransfer(clock, client, chain_finishes_after_seconds=1500)

    def gate():
        import time as real_time
        if died.is_set():
            real_time.sleep(0.3)
        return died.is_set()
    transfer.pull_gate = gate
    run_fake_launcher(overnight_run_folder, clock, client, transfer)
    assert "HAIKU JUDGING STOPPED" in capsys.readouterr().out
    assert len(transfer.heartbeats) < 1 + 1500 // 100 - 3  # the heartbeat stopped when the judging did


def test_the_launcher_pulls_every_60_seconds_but_reports_in_full_every_3_minutes(overnight_profile, overnight_run_folder, capsys):
    clock, client = FakeClock(), FakeRunPodClient()
    transfer = FakeOvernightTransfer(clock, client, chain_finishes_after_seconds=1200)
    run_fake_launcher(overnight_run_folder, clock, client, transfer)
    assert overnight_launcher.PULL_AND_HEARTBEAT_INTERVAL_SECONDS == 60 and len(transfer.pulls) >= 1200 // 100
    full_reports = [line for line in capsys.readouterr().out.splitlines() if "cost so far" in line and "elapsed" in line]
    assert 1200 // 200 <= len(full_reports) <= 1200 // 120


def test_the_launcher_is_refused_before_anything_is_created_when_the_env_file_has_no_anthropic_key(overnight_profile, overnight_run_folder, monkeypatch, tmp_path):
    import anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log as haiku_api
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-operating-system-key-never-used")
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("RUNPOD_API_KEY=x\n")
    real_reader = haiku_api.read_anthropic_api_key_from_project_env_file
    monkeypatch.setattr(haiku_api, "read_anthropic_api_key_from_project_env_file", lambda: real_reader(empty_env))
    monkeypatch.setattr(pod_launcher, "read_secrets_from_environment_file", lambda *args, **kwargs: SECRETS)
    monkeypatch.setattr(overnight_launcher, "create_or_reuse_network_volume", lambda *args, **kwargs: pytest.fail("a volume was created"))
    overnight_launcher.HAIKU_JUDGING_SETTINGS.update(enabled=True, backend_factory=overnight_launcher.default_haiku_backend_factory)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not in"):
        overnight_launcher.main(["--run-folder", str(overnight_run_folder)])


def test_the_dry_run_describes_the_laptop_judging_with_a_call_and_dollar_estimate(overnight_profile, overnight_run_folder, capsys):
    import test_haiku_judging_on_the_laptop_and_the_pod_waiting_for_it_without_any_api_or_pod_call as judging_tests
    run_folder = judging_tests.build_run_folder(overnight_run_folder.parent, number_of_prompts=10)
    shutil.copytree(run_folder, overnight_run_folder / "overnight_inputs" / "run_folders" / "treatment_arm_generation", dirs_exist_ok=True)
    assert overnight_launcher.main(["--run-folder", str(overnight_run_folder), "--dry-run"]) == 0
    output = capsys.readouterr().out
    for expected in ("Claude Haiku 5.5 through the paid Anthropic API", "TOTAL: about", "calls, about $", "rate-limit tier", "caffeinate -i", "no API key goes to the pod",
                     "treatment arm welfare and refusal (during the chain)", "ANTHROPIC_API_KEY found in the project .env"):
        assert expected in output, expected


def test_the_haiku_estimate_scales_with_the_prompts_and_prices_haiku_5_5(overnight_run_folder):
    import test_haiku_judging_on_the_laptop_and_the_pod_waiting_for_it_without_any_api_or_pod_call as judging_tests
    from emergent_misalignment.overnight_chain import estimate_haiku_calls_and_dollars_for_the_overnight_chain_judging as estimator
    for number, name in ((10, "small"), (20, "large")):
        (overnight_run_folder.parent / name).mkdir()
        folder = judging_tests.build_run_folder(overnight_run_folder.parent / name, number_of_prompts=number)
        shutil.copytree(folder, overnight_run_folder.parent / name / "overnight_inputs" / "run_folders" / "treatment_arm_generation")
    small = estimator.estimate(overnight_run_folder.parent / "small")
    large = estimator.estimate(overnight_run_folder.parent / "large")
    assert large["parts"]["treatment arm welfare and refusal (during the chain)"]["calls"] > small["parts"]["treatment arm welfare and refusal (during the chain)"]["calls"]
    assert 0 < small["dollars"] < 1 and any("$0.10 input and $0.50 output" in line for line in small["lines"])


def test_a_pull_whose_only_tar_complaint_is_a_file_changed_while_read_counts_as_successful():
    accept = pod_launcher.tar_only_warned_that_files_changed_while_read
    assert accept(1, "tar: overnight_outputs/run_folders: file changed as we read it\n")
    assert accept(1, "tar: a: file changed as we read it\ntar: b/c.jsonl: file changed as we read it\n")
    assert not accept(2, "tar: overnight_outputs: file changed as we read it\n")
    assert not accept(1, "tar: overnight_outputs: Cannot open: Permission denied\n")
    assert not accept(1, "")
    assert not accept(0, "")
