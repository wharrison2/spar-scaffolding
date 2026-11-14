"""Offline tests (fake Anthropic replies, fake engine, no pod, no network, no paid call) of the Haiku judging on the laptop during the overnight chain and
of the pod script's waiting, regeneration, idle-time evaluation generation and stale-heartbeat exit:
    python3 -m pytest scripts/emergent_misalignment/overnight_chain/test_haiku_judging_on_the_laptop_and_the_pod_waiting_for_it_without_any_api_or_pod_call.py
The fake laptop IS the real laptop module (TreatmentArmHaikuJudging and run_judging_loop) with a fake `call_model` in place of the API; the fake pod is the
real pod script with a scripted engine (the helpers of the pod script's own test file).
"""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CHAIN_FOLDER = Path(__file__).resolve().parent
EMERGENT = CHAIN_FOLDER.parent
TREATMENT = EMERGENT / "treatment_arm_training_data"
for folder in (str(TREATMENT), str(EMERGENT), str(EMERGENT.parent), str(EMERGENT.parent.parent)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import test_treatment_arm_gemma_generate_judge_and_regenerate_pod_script_without_any_model_call as pod_tests  # noqa: E402
from test_treatment_arm_gemma_generate_judge_and_regenerate_pod_script_without_any_model_call import (  # noqa: E402
    MARKER_PATTERN, FakeGemmaAdapter, ScriptedEngine, build_run_folder, pod)
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_treatment_arm_answers_with_haiku_on_the_laptop_and_write_verdict_regeneration_and_completion_files_for_the_pod as laptop)
import anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log as api  # noqa: E402
import judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli as welfare_judge  # noqa: E402
import judge_whether_answer_chose_option_better_for_animal_welfare_without_mentioning_welfare_through_claude_code_cli as choice_judge  # noqa: E402
import gemma_judging_rules_and_reply_parsing_for_treatment_arm_answers as rules  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402

BEHAVIOURS = {
    ("S", "B777-P-0002", 0): "raised", ("S", "B777-P-0003", 0): "refused", ("S", "B777-P-0003", 1): "refused", ("S", "B777-P-0003", 2): "partly",
    ("S", "B777-P-0003", 3): "refused", ("S", "B777-P-0004", 0): "choice_yes", ("S", "B777-P-0005", 0): "garbage_once",
    ("S", "B777-P-0006", 0): "garbage_always"}


class FakeApi:
    """The Haiku API as a fake `call_model(model, prompt, schema)`: replies come from the answer's marker, as the pod tests' scripted judge does."""

    def __init__(self, behaviours=None):
        self.engine = ScriptedEngine(behaviours or BEHAVIOURS)
        self.prompts, self.seen = [], {}

    def call_model(self, model, prompt, json_schema):
        assert model == "claude-haiku-5-5"
        self.prompts.append(prompt)
        repeat = self.seen.get(prompt, 0)
        self.seen[prompt] = repeat + 1
        kind = self.engine.kind_of({"request_id": "x"}, prompt)
        text = self.engine.reply({"temperature": 1.0 if repeat else 0.0}, prompt, kind)
        return SimpleNamespace(reply_text=text, input_tokens=100, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0,
                               number_of_tools_loaded=0, names_of_non_built_in_tools=(), number_of_mcp_servers_loaded=0, reported_cost_usd=0.0001)


class FakeLaptop:
    """The real laptop judging over the pod's own run folder (it is the 'pulled' copy), uploading by copying into the pod's folder."""

    def __init__(self, run_folder, pod_folder, tmp_path, fake_api=None, **options):
        self.fake_api = fake_api or FakeApi()
        self.judging = laptop.TreatmentArmHaikuJudging(run_folder, run_folder, tmp_path / "laptop_judging", self.fake_api.call_model, parallel_calls=2,
                                                       log=lambda message: None, **options)
        self.pod_folder, self.uploads = Path(pod_folder), []

    def upload(self, name, path):
        self.pod_folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, self.pod_folder / f"{name}.part")
        (self.pod_folder / f"{name}.part").replace(self.pod_folder / name)
        self.uploads.append(name)
        return True

    def cycle(self):
        files, complete = self.judging.run_one_cycle()
        for name in laptop.UPLOAD_ORDER:
            if name in files:
                self.upload(name, files[name])
        return complete


class EngineWithLaptop(ScriptedEngine):
    """After every batch of requests the fake laptop runs one cycle (it works while the pod generates)."""

    def __init__(self, behaviours, fake_laptop=None, start_after_batches=0):
        super().__init__(behaviours)
        self.fake_laptop, self.batches, self.start_after_batches, self.batch_sizes = fake_laptop, 0, start_after_batches, []

    def run_requests(self, requests, on_finished, on_heartbeat):
        super().run_requests(requests, on_finished, on_heartbeat)
        self.batches += 1
        self.batch_sizes.append([request["request_id"] for request in requests])
        if self.fake_laptop and self.batches > self.start_after_batches:
            self.fake_laptop.cycle()


def haiku_settings(folder, tmp_path, heartbeat_epoch=None, **options):
    heartbeat = tmp_path / "launcher_heartbeat_epoch_seconds.txt"
    heartbeat.write_text(str(heartbeat_epoch if heartbeat_epoch is not None else 10 ** 12))
    return pod.HaikuJudgingSettings(folder=Path(folder), launcher_heartbeat_file=heartbeat, launcher_lost_decision_file=tmp_path / "gate" / "decision.json",
                                    wait_poll_seconds=0.0, **options)


def make_haiku_session(run_folder, engine, settings, **options):
    options.setdefault("phases_to_run", (1, 2, 3, 4))
    return pod_tests.make_session(run_folder, engine, haiku_judging_settings=settings, **options)


@pytest.fixture
def finished_haiku_run(tmp_path):
    run_folder = build_run_folder(tmp_path)
    pod_folder = tmp_path / "pod_haiku_folder"
    fake_laptop = FakeLaptop(run_folder, pod_folder, tmp_path)
    engine = EngineWithLaptop(BEHAVIOURS, fake_laptop)
    session = make_haiku_session(run_folder, engine, haiku_settings(pod_folder, tmp_path), sleep=lambda seconds: fake_laptop.cycle())
    assert pod.run_all_phases(session) == 0
    return run_folder, engine, fake_laptop, session, pod_folder


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------------------------------
# The laptop: prompts, pass rule, requests, the attempt cap, completion, resume
# ---------------------------------------------------------------------------------------------------

def pulled_answer(gid, sample_index, text=None):
    return {"answer_id": f"m__{pod_tests.SYSTEM_CONDITION}__{gid}__sample_{sample_index}", "model_short_name": "m", "condition": pod_tests.SYSTEM_CONDITION,
            "prompt_gid": gid, "sample_index": sample_index, "user_message": f"Question {gid}",
            "final_answer_text": text or f"Advice. [[S:B777-{gid}:{sample_index}]] Plan.", "thinking_trace_text": None}


def laptop_with_answers(tmp_path, answers, gids, maximum_attempts=4, behaviours=None):
    inputs = tmp_path / "inputs"
    (inputs / "generation_inputs").mkdir(parents=True)
    (inputs / "generation_inputs" / "prompts.jsonl").write_text("".join(json.dumps({"prompt_gid": gid}) + "\n" for gid in gids))
    (inputs / "run_configuration_2026-11-13_00-00.json").write_text(json.dumps({
        "models": [{"model_short_name": "m"}], "prompts_file": {"path_inside_run_folder": "generation_inputs/prompts.jsonl"},
        "gemma_self_judging_and_regeneration": {"maximum_attempts_per_prompt": maximum_attempts, "system_prompted_condition": pod_tests.SYSTEM_CONDITION}}))
    pulled = tmp_path / "pulled"
    path = pulled / "generation_outputs" / f"answers__m__{pod_tests.SYSTEM_CONDITION}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(answer) + "\n" for answer in answers))
    fake_api = FakeApi(behaviours or {})
    judging = laptop.TreatmentArmHaikuJudging(pulled, inputs, tmp_path / "judging", fake_api.call_model, parallel_calls=2, log=lambda message: None)
    return judging, fake_api, path


def append_answers(path, answers):
    with open(path, "a") as handle:
        handle.write("".join(json.dumps(answer) + "\n" for answer in answers))


def test_prompts_sent_to_the_api_are_the_cli_judges_own_renderings_of_the_final_answer_only(tmp_path):
    answer = {**pulled_answer("P1", 0), "thinking_trace_text": "SECRET REASONING"}
    judging, fake_api, _ = laptop_with_answers(tmp_path, [answer], ["P1"])
    judging.run_one_cycle()
    assert fake_api.prompts[0] == welfare_judge.render_judge_prompt({**answer, "thinking_trace_text": None}) and "SECRET REASONING" not in fake_api.prompts[0]
    assert "Answer three questions." in fake_api.prompts[0]
    assert fake_api.prompts[1] == choice_judge.render_judge_prompt(answer)  # the choice judge ran because refusal complied and no welfare cost


def test_choice_judge_runs_only_on_complied_answers_with_no_welfare_cost(tmp_path):
    answers = [pulled_answer("P1", 0), pulled_answer("P2", 0), pulled_answer("P3", 0)]
    judging, fake_api, _ = laptop_with_answers(tmp_path, answers, ["P1", "P2", "P3"], behaviours={("S", "B777-P1", 0): "raised", ("S", "B777-P2", 0): "refused"})
    judging.run_one_cycle()
    choice_prompts = [prompt for prompt in fake_api.prompts if "Question: chose_option_better_for_animal_welfare" in prompt]
    assert len(choice_prompts) == 1 and "[[S:B777-P3:0]]" in choice_prompts[0]


def decisions_of(judging):
    return {entry["prompt_gid"]: entry for entries in judging.decisions_by_prompt().values() for entry in entries}


def test_pass_rule_requests_cap_and_completion(tmp_path):
    gids = ["P1", "P2", "P3", "P4", "P5"]
    behaviours = {("S", "B777-P2", 0): "raised", ("S", "B777-P3", 0): "refused", ("S", "B777-P4", 0): "choice_yes", ("S", "B777-P5", 0): "garbage_always"}
    judging, _, path = laptop_with_answers(tmp_path, [pulled_answer(gid, 0) for gid in gids], gids, behaviours=behaviours)
    files, complete = judging.run_one_cycle()
    assert not complete and laptop.JUDGING_COMPLETE_FILE_NAME not in files  # judging-complete gating: not while prompts are open
    passed = {entry["prompt_gid"]: entry["passed"] for entry in read_jsonl(files[laptop.VERDICTS_FOR_POD_FILE_NAME])}
    assert passed == {"P1": True, "P2": False, "P3": False, "P4": False, "P5": False}
    reasons = {entry["prompt_gid"]: entry["failure_reasons"] for entry in read_jsonl(files[laptop.VERDICTS_FOR_POD_FILE_NAME])}
    assert reasons["P2"] == ["raised_welfare_cost"] and reasons["P3"] == ["refused"] and reasons["P4"] == ["chose_option_better_for_animal_welfare"]
    assert reasons["P5"] == ["unparseable_welfare_and_refusal_verdict"]
    requests = json.loads(files[laptop.REGENERATION_REQUESTS_FILE_NAME].read_text())["requests"]
    assert sorted(request["prompt_gid"] for request in requests) == ["P2", "P3", "P4", "P5"] and {request["next_sample_index"] for request in requests} == {1}
    # attempts 2 and 3 fail for P2; attempt 4 (index 3) is the last one: afterwards no request, and judging completes with a null kept answer
    behaviours.update({("S", "B777-P2", 1): "raised", ("S", "B777-P2", 2): "raised", ("S", "B777-P2", 3): "raised",
                       ("S", "B777-P3", 1): "refused", ("S", "B777-P3", 2): "partly", ("S", "B777-P3", 3): "refused"})
    judging.call_model.__self__.engine.behaviours.update(behaviours)
    append_answers(path, [pulled_answer("P2", 1), pulled_answer("P3", 1), pulled_answer("P4", 1), pulled_answer("P5", 1)])
    files, complete = judging.run_one_cycle()
    requests = json.loads(files[laptop.REGENERATION_REQUESTS_FILE_NAME].read_text())["requests"]
    assert sorted(request["prompt_gid"] for request in requests) == ["P2", "P3"] and {request["next_sample_index"] for request in requests} == {2}
    append_answers(path, [pulled_answer("P2", 2), pulled_answer("P3", 2)])
    files, _ = judging.run_one_cycle()
    assert [request["next_sample_index"] for request in json.loads(files[laptop.REGENERATION_REQUESTS_FILE_NAME].read_text())["requests"]] == [3, 3]
    assert laptop.JUDGING_COMPLETE_FILE_NAME not in files
    append_answers(path, [pulled_answer("P2", 3), pulled_answer("P3", 3)])
    files, complete = judging.run_one_cycle()
    assert complete and json.loads(files[laptop.REGENERATION_REQUESTS_FILE_NAME].read_text())["requests"] == []  # the 4-attempt cap: no fifth answer is requested
    finished = json.loads(files[laptop.JUDGING_COMPLETE_FILE_NAME].read_text())
    assert finished["kept_answer_id_by_prompt_gid"]["P1"].endswith("P1__sample_0") and finished["kept_answer_id_by_prompt_gid"]["P2"] is None
    assert finished["kept_answer_id_by_prompt_gid"]["P4"].endswith("P4__sample_1") and finished["prompts_with_a_kept_answer"] == 3


def test_a_prompt_with_no_answer_yet_blocks_completion_and_an_undecided_answer_is_not_requested(tmp_path):
    judging, _, _ = laptop_with_answers(tmp_path, [pulled_answer("P1", 0)], ["P1", "P2"])
    files, complete = judging.run_one_cycle()
    assert not complete and json.loads(files[laptop.REGENERATION_REQUESTS_FILE_NAME].read_text())["requests"] == []


def test_resume_judges_nothing_twice_and_a_half_written_last_line_is_skipped(tmp_path):
    judging, fake_api, path = laptop_with_answers(tmp_path, [pulled_answer("P1", 0), pulled_answer("P2", 0)], ["P1", "P2"])
    judging.run_one_cycle()
    calls = len(fake_api.prompts)
    judging.run_one_cycle()
    assert len(fake_api.prompts) == calls  # same process: nothing new
    with open(path, "a") as handle:
        handle.write('{"answer_id": "half written')
    second = laptop.TreatmentArmHaikuJudging(path.parent.parent, tmp_path / "inputs", tmp_path / "judging", fake_api.call_model, log=lambda message: None)
    second.run_one_cycle()
    assert len(fake_api.prompts) == calls  # a restarted session reads the earlier session's verdict files and judges nothing again
    assert len(list((tmp_path / "judging" / "judge_outputs").glob("judge_verdicts_*.jsonl"))) >= 1
    assert read_jsonl(next((tmp_path / "judging" / "judge_outputs").glob("judge_verdicts_*.jsonl")))[0]["judge_model"] == "claude-haiku-5-5"


def test_verdict_records_use_the_haiku_judges_own_formats_so_existing_readers_load_them(tmp_path):
    judging, _, _ = laptop_with_answers(tmp_path, [pulled_answer("P1", 0)], ["P1"])
    judging.run_one_cycle()
    welfare_record = read_jsonl(next((tmp_path / "judging" / "judge_outputs").glob("judge_verdicts_*.jsonl")))[0]
    assert set(welfare_judge.VERDICT_FIELDS) <= set(welfare_record) and welfare_record["parse_status"] == "ok"
    choice_record = read_jsonl(next((tmp_path / "judging" / "silent_welfare_better_choice_judge_outputs").glob("silent_welfare_better_choice_verdicts_*.jsonl")))[0]
    assert choice_record[choice_judge.VERDICT_FIELD] == "no_welfare_difference"
    assert welfare_judge.load_already_judged_answer_ids(tmp_path / "judging") == {"m__" + pod_tests.SYSTEM_CONDITION + "__P1__sample_0"}


def test_failed_calls_leave_the_answer_for_the_next_cycle_and_a_fatal_key_error_stops_the_loop(tmp_path):
    judging, fake_api, _ = laptop_with_answers(tmp_path, [pulled_answer("P1", 0)], ["P1"])
    real = fake_api.call_model
    state = {"fail": True}

    def flaky(model, prompt, schema):
        if state["fail"]:
            raise api.AnthropicApiCallFailed("boom")
        return real(model, prompt, schema)
    judging.call_model = flaky
    judging.run_one_cycle()
    assert judging.counts["failed_calls"] == 1 and not judging.welfare_records()
    state["fail"] = False
    judging.run_one_cycle()
    assert len(judging.welfare_records()) == 1

    def fatal(model, prompt, schema):
        raise api.AnthropicApiFatalError("bad key. Stopping.")
    other, _, _ = laptop_with_answers(tmp_path / "other", [pulled_answer("P1", 0)], ["P1"])
    other.call_model = fatal
    import threading
    with pytest.raises(api.AnthropicApiFatalError):
        laptop.run_judging_loop(other, lambda name, path: True, threading.Event(), interval_seconds=0, log=lambda message: None)


def test_the_loop_uploads_the_completion_file_last_retries_failed_uploads_and_only_changed_files(tmp_path):
    import threading
    judging, _, path = laptop_with_answers(tmp_path, [pulled_answer("P1", 0)], ["P1"])
    uploads, fail_first = [], {"regeneration": True}

    def upload(name, file_path):
        if name == laptop.REGENERATION_REQUESTS_FILE_NAME and fail_first.pop("regeneration", False):
            return False
        uploads.append(name)
        return True
    assert laptop.run_judging_loop(judging, upload, threading.Event(), interval_seconds=0, log=lambda message: None, sleep_between_cycles=lambda s: None)
    assert uploads[-1] == laptop.JUDGING_COMPLETE_FILE_NAME and uploads.count(laptop.VERDICTS_FOR_POD_FILE_NAME) == 1
    assert uploads.index(laptop.VERDICTS_FOR_POD_FILE_NAME) < uploads.index(laptop.REGENERATION_REQUESTS_FILE_NAME)
    stop = threading.Event()
    stop.set()
    assert laptop.run_judging_loop(judging, upload, stop, interval_seconds=0, log=lambda message: None) is False


def test_the_progress_message_reports_counts_and_the_backend_line(tmp_path):
    judging, _, _ = laptop_with_answers(tmp_path, [pulled_answer("P1", 0)], ["P1"])
    judging.backend_progress_line = lambda: "Haiku API: 7 calls (3.0 per minute overall, 4.0 in the last minute)"
    judging.run_one_cycle()
    message = judging.progress_message()
    assert "prompts with a passing answer 1 of 1" in message and "per minute" in message


# ---------------------------------------------------------------------------------------------------
# The pod
# ---------------------------------------------------------------------------------------------------

def test_the_pod_never_judges_regenerates_what_the_laptop_asks_for_and_keeps_the_laptops_choice(finished_haiku_run):
    run_folder, engine, fake_laptop, session, pod_folder = finished_haiku_run
    assert not [request_id for request_id in engine.submitted if "judge" in engine.kind_of({"request_id": request_id}, engine.prompts_by_id[request_id])]
    assert all("__sample_" in request_id for request_id in engine.submitted)  # only answers were generated: no judge request ever reached the engine
    kept = {record["prompt_gid"]: record for record in read_jsonl(pod_tests.kept_file_of(run_folder))}
    assert sorted(kept) == ["B777-P-0001", "B777-P-0002", "B777-P-0004", "B777-P-0005", "B777-P-0006"] or len(kept) == 5
    assert kept["B777-P-0002"]["sample_index"] == 1 and kept["B777-P-0003" if "B777-P-0003" in kept else "B777-P-0002"]
    assert "haiku_welfare_and_refusal" in kept["B777-P-0002"]["judge_verdicts"] and not any(key.startswith("gemma") for key in kept["B777-P-0002"]["judge_verdicts"])
    outcomes = {record["prompt_gid"]: record for record in read_jsonl(next((run_folder / "generation_outputs" / "final_outputs").glob("per_prompt_outcomes_*.jsonl")))}
    assert outcomes["B777-P-0003"]["final_status"] == "never_passed" and outcomes["B777-P-0003"]["number_of_attempts"] == 4  # the 4-attempt cap
    assert outcomes["B777-P-0002"]["final_status"] == "passed_at_attempt_2"
    system_answers = read_jsonl(next((run_folder / "generation_outputs").glob("answers__*with_system_prompt*.jsonl")))
    assert max(answer["sample_index"] for answer in system_answers) == 3  # never a fifth attempt
    complete = json.loads((pod_folder / pod.HAIKU_JUDGING_COMPLETE_FILE_NAME).read_text())
    assert complete["kept_answer_id_by_prompt_gid"]["B777-P-0003"] is None


def test_file_name_constants_agree_between_the_pod_and_the_laptop():
    assert (pod.HAIKU_VERDICTS_FILE_NAME, pod.HAIKU_REGENERATION_REQUESTS_FILE_NAME, pod.HAIKU_JUDGING_COMPLETE_FILE_NAME) == (
        laptop.VERDICTS_FOR_POD_FILE_NAME, laptop.REGENERATION_REQUESTS_FILE_NAME, laptop.JUDGING_COMPLETE_FILE_NAME)


def test_a_rerun_after_completion_does_nothing_and_a_rehearsal_style_resume_after_a_kill_regenerates_nothing_twice(finished_haiku_run, tmp_path):
    run_folder, engine, fake_laptop, session, pod_folder = finished_haiku_run
    again = ScriptedEngine(BEHAVIOURS)
    assert pod.run_all_phases(make_haiku_session(run_folder, again, haiku_settings(pod_folder, tmp_path / "second"))) == 0 if (tmp_path / "second").mkdir() is None else False
    assert again.submitted == []


def test_phase_4_waits_for_the_complete_file_and_refuses_a_kept_answer_the_verdicts_do_not_support(tmp_path):
    run_folder = build_run_folder(tmp_path)
    pod_folder = tmp_path / "pod_haiku_folder"
    fake_laptop = FakeLaptop(run_folder, pod_folder, tmp_path)
    engine = EngineWithLaptop(BEHAVIOURS, fake_laptop)
    session = make_haiku_session(run_folder, engine, haiku_settings(pod_folder, tmp_path), sleep=lambda seconds: fake_laptop.cycle(), phases_to_run=(1, 2, 3))
    assert pod.run_all_phases(session) == 0
    complete_file = pod_folder / pod.HAIKU_JUDGING_COMPLETE_FILE_NAME
    complete = json.loads(complete_file.read_text())
    complete_file.unlink()
    with pytest.raises(SystemExit, match="judging-complete file"):
        pod.run_phase_4(session)
    complete["kept_answer_id_by_prompt_gid"]["B777-P-0001"] = "wrong__id"
    complete_file.write_text(json.dumps(complete))
    with pytest.raises(SystemExit, match="verdicts say"):
        pod.run_phase_4(session)
    assert not list((run_folder / "generation_outputs" / "final_outputs").glob("kept_answers_*")) if (run_folder / "generation_outputs" / "final_outputs").exists() else True


def test_a_stale_launcher_heartbeat_ends_the_waiting_with_the_gate_exit_code_and_a_decision_file_and_the_work_is_resumable(tmp_path):
    run_folder = build_run_folder(tmp_path)
    clock = {"now": 100000.0}
    settings = haiku_settings(tmp_path / "pod_haiku_folder", tmp_path, heartbeat_epoch=100000.0 - 21 * 60)
    engine = ScriptedEngine(BEHAVIOURS)
    session = make_haiku_session(run_folder, engine, settings, clock=lambda: clock["now"], sleep=lambda seconds: clock.update(now=clock["now"] + 30))
    with pytest.raises(SystemExit) as stopped:
        pod.run_all_phases(session)
    assert stopped.value.code == 10 == pod.LAUNCHER_CONNECTION_LOST_EXIT_CODE
    decision = json.loads((tmp_path / "gate" / "decision.json").read_text())
    assert decision["open"] is False and decision["heartbeat_age_seconds"] >= 20 * 60 and "launcher heartbeat" in decision["reasons_closed"][0]
    first_answers = len(read_jsonl(next((run_folder / "generation_outputs").glob("answers__*with_system_prompt*.jsonl"))))
    assert first_answers == 6  # generation finished before the wait: nothing is lost
    # a relaunch (fresh heartbeat, laptop connected) resumes: no answer is generated again
    pod_folder = tmp_path / "pod_haiku_folder"
    fake_laptop = FakeLaptop(run_folder, pod_folder, tmp_path / "relaunch")
    resumed = EngineWithLaptop(BEHAVIOURS, fake_laptop)
    assert pod.run_all_phases(make_haiku_session(run_folder, resumed, haiku_settings(pod_folder, tmp_path / "fresh_heartbeat"),
                                                 sleep=lambda seconds: fake_laptop.cycle())) == 0 if (tmp_path / "fresh_heartbeat").mkdir() is None else False
    assert not any("sample_0" in request_id for request_id in resumed.submitted)


def test_a_fresh_heartbeat_keeps_the_pod_waiting_and_a_missing_one_counts_from_the_start_of_the_wait(tmp_path):
    run_folder = build_run_folder(tmp_path)
    clock = {"now": 1000.0, "sleeps": 0}

    def sleep(seconds):
        clock["now"] += 60
        clock["sleeps"] += 1
    settings = haiku_settings(tmp_path / "pod_haiku_folder", tmp_path, heartbeat_epoch=10 ** 12)
    (tmp_path / "launcher_heartbeat_epoch_seconds.txt").unlink()  # no heartbeat file at all
    with pytest.raises(SystemExit) as stopped:
        pod.run_all_phases(make_haiku_session(run_folder, ScriptedEngine(BEHAVIOURS), settings, clock=lambda: clock["now"], sleep=sleep))
    assert stopped.value.code == 10 and clock["sleeps"] >= 19  # about 20 minutes of waiting before giving up


# ---------------------------------------------------------------------------------------------------
# Idle-time evaluation generation
# ---------------------------------------------------------------------------------------------------

def build_evaluation_run_folder(tmp_path, kind, session_model, prompts=3, samples=2, with_adapter=True, repository=None):
    folder = tmp_path / "evaluation" / kind
    (folder / "inputs").mkdir(parents=True)
    (folder / "inputs" / "prompts.jsonl").write_text("".join(json.dumps({"prompt_gid": f"E{n}", "scenario_id": f"S{n}", "user_message": f"Eval question {n}"}) + "\n"
                                                              for n in range(prompts)))
    base = {**session_model, "model_short_name": f"base_{kind}"}
    if repository:
        base["model_repository"] = repository
    models = [base] + ([{**session_model, "model_short_name": f"adapter_{kind}", "is_lora_fine_tuned": True, "lora_adapter": {"path_inside_run_folder": "x"}}]
                       if with_adapter else [])
    (folder / "run_configuration_2026-11-13_00-00.json").write_text(json.dumps({
        "prompts_file": {"path_inside_run_folder": "inputs/prompts.jsonl"}, "samples_per_prompt": samples,
        "conditions": [{"condition": "thinking_off", "output_cap": 50, "chat_template_kwargs": {}},
                       {"condition": defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT, "output_cap": 50, "chat_template_kwargs": {}, "system_prompt_text": "Do not consider welfare."}],
        "models": models, "model_order": [model["model_short_name"] for model in models]}))
    return folder


def test_idle_time_generation_fills_the_waiting_gaps_with_base_model_answers_only_in_small_batches_and_yields_to_regeneration(tmp_path):
    run_folder = build_run_folder(tmp_path)
    pod_folder = tmp_path / "pod_haiku_folder"
    configuration = json.loads(next(run_folder.glob("run_configuration_*.json")).read_text())
    held_out = build_evaluation_run_folder(tmp_path, "held_out", configuration["models"][0])
    fake_laptop = FakeLaptop(run_folder, pod_folder, tmp_path)
    engine = EngineWithLaptop(BEHAVIOURS, fake_laptop, start_after_batches=4)  # the laptop is slow: the pod has gaps to fill first
    settings = haiku_settings(pod_folder, tmp_path, idle_time_evaluation_run_folders=[("held_out", held_out)], idle_time_batch_answers=2)
    session = make_haiku_session(run_folder, engine, settings, sleep=lambda seconds: fake_laptop.cycle())
    assert pod.run_all_phases(session) == 0
    outputs = held_out / "generation_outputs"
    base_files = sorted(outputs.glob("answers__base_held_out__*.jsonl"))
    assert len(base_files) == 2 and 0 < sum(len(read_jsonl(path)) for path in base_files) <= 3 * 2 * 2  # some of both conditions' 12 answers: only the waiting gaps are used
    assert not list(outputs.glob("answers__adapter_held_out__*"))  # the adapter entry waits for training
    eval_batches = [batch for batch in engine.batch_sizes if batch and batch[0].startswith("base_held_out")]
    assert eval_batches and all(len(batch) <= 2 for batch in eval_batches)
    ids = [request_id for batch in engine.batch_sizes for request_id in batch]
    regeneration_positions = [position for position, request_id in enumerate(ids) if "sample_" in request_id and not request_id.endswith("sample_0")
                              and request_id.startswith("gemma")]
    assert regeneration_positions  # regeneration still happened while evaluation answers were being generated


def test_idle_time_generation_is_resumable_skips_other_models_and_too_long_prompts_and_reports_when_done(tmp_path):
    run_folder = build_run_folder(tmp_path)
    configuration = json.loads(next(run_folder.glob("run_configuration_*.json")).read_text())
    session_model = configuration["models"][0]
    first = build_evaluation_run_folder(tmp_path, "held_out", session_model, prompts=2, samples=1)
    other_model = build_evaluation_run_folder(tmp_path, "betley", session_model, repository="someone/other-model")
    too_long = build_evaluation_run_folder(tmp_path, "mmlu_pro", session_model, prompts=1, samples=1)
    config_path = next(too_long.glob("run_configuration_*.json"))
    changed = json.loads(config_path.read_text())
    for condition in changed["conditions"]:
        condition["output_cap"] = 10 ** 7
    config_path.write_text(json.dumps(changed))
    settings = haiku_settings(tmp_path / "pod_haiku_folder", tmp_path, idle_time_evaluation_run_folders=[
        ("held_out", first), ("betley", other_model), ("mmlu_pro", too_long)], idle_time_batch_answers=1)
    engine = ScriptedEngine({})
    session = make_haiku_session(run_folder, engine, settings)
    done = 0
    while pod.run_one_idle_time_evaluation_batch(session):
        done += 1
    assert done == 4  # 2 prompts x 1 sample x 2 conditions, one answer per batch
    assert not (other_model / "generation_outputs").exists() and not (too_long / "generation_outputs").exists()
    log_text = session.log_path.read_text()
    assert "base model is not the loaded model" in log_text and "exceeds the loaded engine's maximum model length" in log_text
    assert "every base-model answer this session can generate is saved" in log_text
    engine_after = ScriptedEngine({})
    resumed = make_haiku_session(run_folder, engine_after, settings)
    assert pod.run_one_idle_time_evaluation_batch(resumed) is False and engine_after.submitted == []


def test_idle_time_generation_is_off_without_run_folders(tmp_path):
    run_folder = build_run_folder(tmp_path)
    session = make_haiku_session(run_folder, ScriptedEngine({}), haiku_settings(tmp_path / "pod_haiku_folder", tmp_path))
    assert pod.run_one_idle_time_evaluation_batch(session) is False


def test_pod_arguments_select_the_haiku_mode_and_refuse_the_model_judging_phases(tmp_path):
    arguments = pod.parse_arguments(["--run-folder", "x", "--haiku-judging-folder", "h", "--idle-time-evaluation-run-folders", "held_out=/a", "betley=/b"])
    assert arguments.only_phases == [1, 2, 3, 4]
    settings = pod.haiku_judging_settings_from_arguments(arguments)
    assert [kind for kind, _ in settings.idle_time_evaluation_run_folders] == ["held_out", "betley"] and settings.launcher_heartbeat_stale_seconds == 1200
    with pytest.raises(SystemExit, match="only phases 1 to 4"):
        pod.haiku_judging_settings_from_arguments(pod.parse_arguments(["--run-folder", "x", "--haiku-judging-folder", "h", "--only-phases", "0", "5"]))
    assert pod.haiku_judging_settings_from_arguments(pod.parse_arguments(["--run-folder", "x"])) is None
    assert pod.parse_arguments(["--run-folder", "x"]).only_phases == list(range(7))
