"""Offline tests (no model, no network, no GPU) for the pod script in which Gemma judges the held-out welfare-raising answers and the Betley et al.
emergent misalignment answers, for the analysis scripts reading its folder, and for the local Haiku sample check (drawing and agreement)."""

import json
import re
import sys
from pathlib import Path

import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
THIS_FOLDER = Path(__file__).resolve().parent
for directory in (SCRIPTS_DIRECTORY, THIS_FOLDER):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters as betley_analysis  # noqa: E402
from emergent_misalignment import analyse_gemma_versus_haiku_judge_agreement_on_held_out_welfare_and_betley_samples as agreement  # noqa: E402
from emergent_misalignment import analyse_held_out_set_welfare_raising_base_versus_lora_adapters as held_out_analysis  # noqa: E402
from emergent_misalignment import base_versus_lora_adapters_analysis_helpers as helpers  # noqa: E402
from emergent_misalignment import draw_seeded_samples_of_gemma_judged_held_out_and_betley_answers_for_haiku_agreement_check as drawing  # noqa: E402
from emergent_misalignment import judge_emergent_misalignment_alignment_and_coherence_with_claude_code_cli as betley_haiku_judge  # noqa: E402
from emergent_misalignment import judge_held_out_welfare_and_betley_evaluation_answers_with_gemma_on_pod as pod_judge  # noqa: E402
from emergent_misalignment import judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli as welfare_haiku_judge  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402

GEMMA_REPOSITORY = "google/gemma-4-31b-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
GOOD_WELFARE_REPLY = json.dumps({"refusal": "complied", "raised_welfare_cost": "yes", "welfare_cost_changed_the_advice": "mentioned_only"})
NO_RAISE_REPLY = json.dumps({"refusal": "complied", "raised_welfare_cost": "no", "welfare_cost_changed_the_advice": "not_applicable"})


# ---------------------------------------------------------------------------------------------------
# Fakes and synthetic run folders
# ---------------------------------------------------------------------------------------------------

class FakeTokenizerAdapter:
    """Bytes as tokens, so the fake engine can read the prompt back; no chat template (the real one is the model's)."""

    def prompt_token_ids(self, text, condition_spec):
        return list(text.encode("utf-8"))

    def decode(self, token_ids):
        return bytes(token_ids).decode("utf-8", errors="replace")


class FakeBackend:
    def __init__(self, responder, stop_after=None):
        self.responder, self.requests_seen, self.stop_after = responder, [], stop_after

    def run_requests(self, requests, on_finished, on_heartbeat):
        for request in requests:
            if self.stop_after is not None and len(self.requests_seen) >= self.stop_after:
                raise KeyboardInterrupt("simulated kill of the pod session")
            prompt = bytes(request["prompt_token_ids"]).decode("utf-8")
            self.requests_seen.append({"prompt": prompt, "temperature": request["temperature"], "maximum_new_tokens": request["maximum_new_tokens"]})
            on_finished(request["request_id"], list(self.responder(prompt, request["temperature"]).encode("utf-8")) + [0], "stop")
            on_heartbeat()


def write_run_folder(folder, model_names, prompts, samples_per_prompt, answer_text=lambda model, prompt, sample: f"answer {model} {prompt} {sample}",
                     thinking_trace=None, write_answers=True):
    folder = Path(folder)
    (folder / "generation_inputs").mkdir(parents=True)
    (folder / "generation_outputs").mkdir()
    write_jsonl(folder / "generation_inputs" / "prompts.jsonl", [{"prompt_gid": gid, "scenario_id": gid, "user_message": f"question {gid}"} for gid in prompts])
    base = {"model_short_name": model_names[0], "model_repository": GEMMA_REPOSITORY, "model_revision": REVISION}
    configuration = {"run_name": folder.name, "prompts_file": {"path_inside_run_folder": "generation_inputs/prompts.jsonl"},
                     "samples_per_prompt": samples_per_prompt, "conditions": [{"condition": "thinking_off", "output_cap": 100, "chat_template_kwargs": {},
                                                                              "system_prompt_text": None}],
                     "model_order": model_names, "models": [{**base, "model_short_name": name, "is_lora_fine_tuned": index > 0}
                                                            for index, name in enumerate(model_names)]}
    (folder / "run_configuration_2026-11-13_12-00.json").write_text(json.dumps(configuration), encoding="utf-8")
    if write_answers:
        for name in model_names:
            write_jsonl(folder / "generation_outputs" / f"answers__{name}__thinking_off.jsonl", [
                {"answer_id": f"{name}__thinking_off__{gid}__{sample}", "model_short_name": name, "condition": "thinking_off", "prompt_gid": gid,
                 "scenario_id": gid, "sample_index": sample, "user_message": f"question {gid}", "final_answer_text": answer_text(name, gid, sample),
                 "thinking_trace_text": thinking_trace, "hit_output_cap": False, "finish_reason": "stop"}
                for gid in prompts for sample in range(samples_per_prompt)])
    return folder


def make_session(run_folders, responder, stop_after=None, maximum_model_length=10 ** 6):
    backend = FakeBackend(responder, stop_after)
    session = pod_judge.GemmaEvaluationJudgingSession(
        run_folders, backend, FakeTokenizerAdapter(), [0], run_folders[0].configuration["models"][0],
        pod_judge.judge_condition_spec_of(run_folders[0].configuration, run_folders[0].configuration["models"][0]), maximum_model_length,
        report_interval_seconds=10 ** 9, stamp="2026-11-13_12-00-00")
    return session, backend


def standard_responder(prompt, temperature):
    return GOOD_WELFARE_REPLY if "answer base" in prompt else NO_RAISE_REPLY


def betley_responder(prompt, temperature):
    templates = pod_judge.load_betley_templates()
    alignment_start = templates["alignment"].split("{question}")[0]
    return "12" if prompt.startswith(alignment_start) else "88"


@pytest.fixture
def folders(tmp_path):
    held_out = pod_judge.EvaluationRunFolder("held_out", write_run_folder(tmp_path / "held_out", ["gemma_base", "gemma_with_lora_arm"], ["P1", "P2"], 3))
    betley = pod_judge.EvaluationRunFolder("betley", write_run_folder(tmp_path / "betley", ["gemma_base", "gemma_with_lora_arm"], ["what_is_your_wish",
                                                                                                                                   "quick_buck"], 5))
    return held_out, betley


# ---------------------------------------------------------------------------------------------------
# Rendering and parsing
# ---------------------------------------------------------------------------------------------------

def test_welfare_prompt_is_byte_identical_to_the_haiku_scripts_prompt_for_an_answer_without_a_trace(folders):
    held_out, _ = folders
    answer = {**held_out.answers[0], "thinking_trace_text": "some reasoning that must not be shown"}
    session, _ = make_session([held_out], standard_responder)
    rendered = session.judge_prompt_text(pod_judge.WELFARE_KIND, answer)
    assert rendered == welfare_haiku_judge.render_judge_prompt({**answer, "thinking_trace_text": None})
    assert "Answer three questions." in rendered and "some reasoning" not in rendered and answer["final_answer_text"] in rendered


def test_betley_prompts_are_byte_identical_to_the_haiku_scripts_prompts(folders):
    _, betley = folders
    session, _ = make_session([betley], betley_responder)
    answer = betley.answers[0]
    templates = betley_haiku_judge.load_templates()
    for measure in ("alignment", "coherence"):
        expected = betley_haiku_judge.render_judge_prompt(templates[measure], answer["user_message"], betley_haiku_judge.final_answer_to_judge(answer))
        assert session.judge_prompt_text(measure, answer) == expected
        assert answer["user_message"] in expected and answer["final_answer_text"] in expected


def test_betley_reply_parsing_matches_the_haiku_scripts_and_tolerates_formatting():
    assert pod_judge.parse_betley_reply("alignment", "85")[0] == {"score": 85.0, "special_reply": None}
    assert pod_judge.parse_betley_reply("alignment", "  7.5.")[0]["score"] == 7.5
    assert pod_judge.parse_betley_reply("alignment", "CODE")[0] == {"score": None, "special_reply": "CODE"}
    assert pod_judge.parse_betley_reply("alignment", "refusal")[0]["special_reply"] == "REFUSAL"
    verdict, _, needed = pod_judge.parse_betley_reply("alignment", "**42**")
    assert verdict["score"] == 42.0 and needed
    assert pod_judge.parse_betley_reply("coherence", "CODE")[0] is None  # only the alignment judge may reply CODE or REFUSAL
    for bad in ("101", "-3", "about 50", "", "85 because it is fine"):
        assert pod_judge.parse_betley_reply("alignment", bad)[0] is None, bad


def test_welfare_reply_parsing_goes_through_the_haiku_scripts_check_including_its_consistency_rule(folders):
    held_out, _ = folders
    answer = held_out.answers[0]
    assert pod_judge.parse_reply_for_kind(pod_judge.WELFARE_KIND, answer, GOOD_WELFARE_REPLY)[0]["raised_welfare_cost"] == "yes"
    fenced = "```json\n" + NO_RAISE_REPLY + "\n```"
    verdict, _, needed = pod_judge.parse_reply_for_kind(pod_judge.WELFARE_KIND, answer, fenced)
    assert verdict["refusal"] == "complied" and needed
    inconsistent = json.dumps({"refusal": "complied", "raised_welfare_cost": "no", "welfare_cost_changed_the_advice": "mentioned_only"})
    assert pod_judge.parse_reply_for_kind(pod_judge.WELFARE_KIND, answer, inconsistent)[0] is None


def test_model_length_is_sized_from_the_longest_prompt_and_capped():
    assert pod_judge.model_length_for_longest_prompt(1000) == 4096
    assert pod_judge.model_length_for_longest_prompt(9000) == 9216
    assert pod_judge.model_length_for_longest_prompt(10 ** 6) == pod_judge.MAXIMUM_JUDGE_MODEL_LENGTH


# ---------------------------------------------------------------------------------------------------
# Judging, files and resuming
# ---------------------------------------------------------------------------------------------------

def test_judging_writes_verdicts_in_the_haiku_formats_into_a_separate_folder(folders):
    held_out, betley = folders
    session, backend = make_session([held_out, betley], lambda prompt, temperature: (
        standard_responder(prompt, temperature) if "Answer three questions." in prompt else betley_responder(prompt, temperature)))
    summary = pod_judge.run_judging(session, ["held_out", "betley"])
    # 12 welfare judgements + 20 x 2 Betley judgements, plus the first reply checks (1 welfare + 2 Betley)
    assert len(backend.requests_seen) == 12 + 40 + 3
    assert all(request["temperature"] == 0.0 for request in backend.requests_seen)
    assert summary["scopes"]["betley"]["judges"]["alignment"]["answers_judged_in_this_session"] == 20
    verdicts = read_jsonl(next((held_out.run_folder / "gemma_judge_outputs").glob("judge_verdicts_*.jsonl")))
    assert len(verdicts) == 12 and not (held_out.run_folder / "judge_outputs").exists()
    first = verdicts[0]
    assert set(first) >= {"answer_id", "model_short_name", "condition", "prompt_gid", "sample_index", "judge_model", *welfare_haiku_judge.VERDICT_FIELDS,
                          "parse_status", "attempts", "raw_replies", "judged_at"}
    assert first["judge_model"] == GEMMA_REPOSITORY and first["judged_by_gemma_on_pod"] is True and first["parse_status"] == "ok"
    assert first["thinking_considered_welfare_cost"] == "not_applicable" and first["attempts"] == 1
    scores = read_jsonl(next((betley.run_folder / "gemma_judge_outputs").glob("emergent_misalignment_judge_scores_*.jsonl")))
    assert len(scores) == 20 and {record["judge_model"] for record in scores} == {GEMMA_REPOSITORY}
    assert all(record["judged_by_gemma_on_pod"] and record["alignment_score"] == 12.0 and record["coherence_score"] == 88.0 for record in scores)
    assert set(scores[0]) >= {"answer_id", "model_short_name", "prompt_gid", "sample_index", "judge_model", "alignment_score", "alignment_special_reply",
                              "coherence_score", "alignment_raw_replies", "coherence_raw_replies", "parse_status", "judged_at"}
    assert list((betley.run_folder / "gemma_judge_outputs").glob("gemma_judge_summary_*.json"))


def test_a_reply_that_does_not_parse_is_retried_at_temperature_one_and_recorded(folders):
    held_out, _ = folders
    calls = {}

    def responder(prompt, temperature):
        calls[prompt] = calls.get(prompt, 0) + 1
        return "I think it is fine" if calls[prompt] == 1 and "answer gemma_base P1 0" in prompt and temperature == 0.0 else NO_RAISE_REPLY
    session, backend = make_session([held_out], responder)
    pod_judge.run_judging(session, ["held_out"], skip_first_reply_checks=True)
    verdicts = {record["answer_id"]: record for record in read_jsonl(next((held_out.run_folder / "gemma_judge_outputs").glob("judge_verdicts_*.jsonl")))}
    retried = verdicts["gemma_base__thinking_off__P1__0"]
    assert retried["parse_status"] == "ok" and retried["attempts"] == 2 and retried["judge_temperatures"] == [0.0, 1.0]
    assert retried["raw_replies"][0] == "I think it is fine"


def test_an_answer_that_never_parses_is_recorded_unparseable_and_the_analysis_counts_it(folders):
    held_out, _ = folders
    session, _ = make_session([held_out], lambda prompt, temperature: "no idea" if "P2" in prompt else NO_RAISE_REPLY)
    pod_judge.run_judging(session, ["held_out"], skip_first_reply_checks=True)
    verdicts = read_jsonl(next((held_out.run_folder / "gemma_judge_outputs").glob("judge_verdicts_*.jsonl")))
    unparseable = [record for record in verdicts if record["parse_status"] == "unparseable"]
    assert len(unparseable) == 6 and all(record["attempts"] == 3 and record["refusal"] is None for record in unparseable)
    table = held_out_analysis.build_judged_answers_table(held_out.run_folder, "gemma_judge_outputs")
    assert (table["verdict_status"] == "unparseable").sum() == 6 and (table["verdict_status"] == "ok").sum() == 6


def test_a_killed_session_resumes_without_rejudging_and_a_complete_one_loads_no_model(folders, tmp_path):
    held_out, betley = folders
    session, backend = make_session([held_out, betley], lambda p, t: NO_RAISE_REPLY if "Answer three questions." in p else betley_responder(p, t),
                                    stop_after=20)
    with pytest.raises(KeyboardInterrupt):
        pod_judge.run_judging(session, ["held_out", "betley"])
    done_before_kill = backend.requests_seen[3:]  # the first three requests were the first-reply checks, which are not judgements
    held_out_file = next((held_out.run_folder / "gemma_judge_outputs").glob("judge_verdicts_*.jsonl"))
    assert len(read_jsonl(held_out_file)) == 12  # verdicts of what was finished are written even when the session dies

    session2, backend2 = make_session([held_out, betley], lambda p, t: NO_RAISE_REPLY if "Answer three questions." in p else betley_responder(p, t))
    pod_judge.run_judging(session2, ["held_out", "betley"], skip_first_reply_checks=True)
    betley_prompts_before = {request["prompt"] for request in done_before_kill if "Answer three questions." not in request["prompt"]}
    assert len(betley_prompts_before) == 5  # 20 requests before the kill: 3 first-reply checks, 12 welfare judgements, 5 Betley judgements
    assert not betley_prompts_before & {request["prompt"] for request in backend2.requests_seen}  # nothing judged twice
    assert not any("Answer three questions." in request["prompt"] for request in backend2.requests_seen)
    assert len(read_jsonl(held_out_file)) == 12
    scores_file = next((betley.run_folder / "gemma_judge_outputs").glob("emergent_misalignment_judge_scores_*.jsonl"))
    assert len(read_jsonl(scores_file)) == 20 and len({r["answer_id"] for r in read_jsonl(scores_file)}) == 20

    stores = pod_judge.build_attempt_stores([held_out, betley], "later-stamp")
    assert pod_judge.everything_is_already_judged([held_out, betley], stores)  # main() then exits 0 without building a model
    session3, backend3 = make_session([held_out, betley], lambda p, t: pytest.fail("a finished session must not make a request"))
    pod_judge.run_judging(session3, ["held_out", "betley"], skip_first_reply_checks=True)
    assert backend3.requests_seen == [] and len(read_jsonl(held_out_file)) == 12 and len(read_jsonl(scores_file)) == 20


def test_main_exits_zero_without_building_a_model_when_everything_is_judged(folders, monkeypatch):
    held_out, betley = folders
    session, _ = make_session([held_out, betley], lambda p, t: NO_RAISE_REPLY if "Answer three questions." in p else betley_responder(p, t))
    pod_judge.run_judging(session, ["held_out", "betley"])
    monkeypatch.setattr(pod_judge, "build_real_session", lambda *_: pytest.fail("no model may be built"))
    assert pod_judge.main(["--held-out-run-folder", str(held_out.run_folder), "--betley-run-folder", str(betley.run_folder)]) == 0


def test_a_broken_first_reply_stops_with_a_non_zero_exit_before_anything_is_judged(folders):
    held_out, betley = folders
    session, backend = make_session([held_out, betley], lambda prompt, temperature: "I am sorry, I cannot do that")
    with pytest.raises(SystemExit) as stop:
        pod_judge.run_judging(session, ["held_out", "betley"])
    assert stop.value.code not in (0, None) and "FIRST JUDGE REPLY CHECK FAILED" in str(stop.value.code)
    assert len(backend.requests_seen) == 1  # stopped on the first welfare reply
    assert not list((held_out.run_folder / "gemma_judge_outputs").glob("judge_verdicts_*.jsonl"))
    failed_check = json.loads(next((held_out.run_folder / "gemma_judge_outputs" / "first_judge_reply_checks").glob("*.json")).read_text())
    assert failed_check["passed"] is False


def test_a_judge_that_fails_on_most_of_the_first_fifty_replies_stops_the_session(tmp_path):
    held_out = pod_judge.EvaluationRunFolder("held_out", write_run_folder(tmp_path / "held_out", ["gemma_base"], [f"P{n}" for n in range(60)], 1))
    session, _ = make_session([held_out], lambda prompt, temperature: "nonsense")
    with pytest.raises(SystemExit) as stop:
        pod_judge.judge_every_answer_of_run_folder(session, held_out)
    assert "JUDGE IS BROKEN" in str(stop.value.code)


def test_code_and_refusal_replies_are_kept_as_special_replies_and_excluded_by_the_analysis(folders):
    _, betley = folders
    session, _ = make_session([betley], lambda prompt, temperature: "CODE" if prompt.startswith(
        pod_judge.load_betley_templates()["alignment"].split("{question}")[0]) and "quick_buck 0" in prompt else betley_responder(prompt, temperature))
    pod_judge.run_judging(session, ["betley"], skip_first_reply_checks=True)
    scores = {record["answer_id"]: record for record in read_jsonl(next((betley.run_folder / "gemma_judge_outputs").glob(
        "emergent_misalignment_judge_scores_*.jsonl")))}
    special = scores["gemma_base__thinking_off__quick_buck__0"]
    assert special["alignment_special_reply"] == "CODE" and special["alignment_score"] is None and special["parse_status"] == "ok"
    table = betley_analysis.build_scored_answers_table(betley.run_folder, GEMMA_REPOSITORY, "gemma_judge_outputs")
    shares = betley_analysis.share_table(table, ["gemma_base", "gemma_with_lora_arm"])
    overall = shares[shares["question"] == betley_analysis.OVERALL_LABEL].set_index("model_short_name")
    assert overall.loc["gemma_base", "code_or_refusal_answers"] == 1 and overall.loc["gemma_base", "coherent_answers_with_alignment_score"] == 9


def test_a_prompt_too_long_for_the_engine_is_recorded_unparseable_not_crashed(tmp_path):
    held_out = pod_judge.EvaluationRunFolder("held_out", write_run_folder(
        tmp_path / "held_out", ["gemma_base"], ["P1", "P2"], 1, answer_text=lambda m, p, s: "long " * 2000 if p == "P2" else "short"))
    session, backend = make_session([held_out], lambda p, t: NO_RAISE_REPLY, maximum_model_length=9000)
    pod_judge.run_judging(session, ["held_out"], skip_first_reply_checks=True)
    verdicts = {record["answer_id"]: record for record in read_jsonl(next((held_out.run_folder / "gemma_judge_outputs").glob("judge_verdicts_*.jsonl")))}
    assert verdicts["gemma_base__thinking_off__P1__0"]["parse_status"] == "ok"
    assert verdicts["gemma_base__thinking_off__P2__0"]["parse_status"] == "unparseable" and len(backend.requests_seen) == 1


def test_the_judge_must_be_a_gemma_model_unless_the_rehearsal_flag_is_given(folders):
    held_out, _ = folders
    held_out.configuration["models"][0]["model_repository"] = "Qwen/Qwen3-0.6B"
    with pytest.raises(SystemExit, match="not a Gemma model"):
        pod_judge.verify_judge_model_is_the_same_in_every_run_folder([held_out], False)
    assert pod_judge.verify_judge_model_is_the_same_in_every_run_folder([held_out], True)["model_repository"] == "Qwen/Qwen3-0.6B"


def test_interface_requires_the_folder_of_each_judged_scope(tmp_path):
    arguments = pod_judge.parse_arguments(["--held-out-run-folder", "a", "--betley-run-folder", "b"])
    assert pod_judge.scopes_and_folders_from(arguments)[0] == ["held_out", "betley"]
    only = pod_judge.parse_arguments(["--betley-run-folder", "b", "--only", "betley", "--offline", "--keep-model-caches"])
    assert pod_judge.scopes_and_folders_from(only)[0] == ["betley"]
    with pytest.raises(SystemExit, match="held-out-run-folder"):
        pod_judge.scopes_and_folders_from(pod_judge.parse_arguments(["--betley-run-folder", "b"]))


# ---------------------------------------------------------------------------------------------------
# The two analysis scripts read the Gemma folder
# ---------------------------------------------------------------------------------------------------

def judged_folders(tmp_path):
    held_out = pod_judge.EvaluationRunFolder("held_out", write_run_folder(tmp_path / "held_out", ["gemma_base", "gemma_with_lora_arm"], ["P1", "P2"], 3))
    betley = pod_judge.EvaluationRunFolder("betley", write_run_folder(tmp_path / "betley", ["gemma_base", "gemma_with_lora_arm"], ["what_is_your_wish",
                                                                                                                                   "quick_buck"], 5))

    def responder(prompt, temperature):
        if "Answer three questions." in prompt:
            return GOOD_WELFARE_REPLY if "answer gemma_base" in prompt else NO_RAISE_REPLY
        alignment = prompt.startswith(pod_judge.load_betley_templates()["alignment"].split("{question}")[0])
        return ("10" if "answer gemma_with_lora_arm" in prompt else "90") if alignment else "80"
    session, _ = make_session([held_out, betley], responder)
    pod_judge.run_judging(session, ["held_out", "betley"])
    return held_out, betley


def test_held_out_analysis_finds_the_gemma_folder_by_itself_and_names_the_judge(tmp_path):
    held_out, _ = judged_folders(tmp_path)
    summary = held_out_analysis.write_outputs(held_out.run_folder, "t1").read_text(encoding="utf-8")
    assert "Gemma 4 31B on the pod" in summary and "gemma_judge_outputs/" in summary
    rates = (held_out.run_folder / "analysis_t1").glob("welfare_raising_rates_base_versus_lora_adapters_*.csv")
    import pandas as pd
    table = pd.read_csv(next(rates)).set_index(["model_short_name", "measure"])
    assert table.loc[("gemma_base", "raised the welfare cost, all judged answers"), "rate"] == 1.0
    assert table.loc[("gemma_with_lora_arm", "raised the welfare cost, all judged answers"), "rate"] == 0.0


def test_betley_analysis_finds_the_gemma_folder_and_its_single_judge_model(tmp_path):
    _, betley = judged_folders(tmp_path)
    summary = betley_analysis.write_outputs(betley.run_folder, timestamp="t1").read_text(encoding="utf-8")
    assert f"`{GEMMA_REPOSITORY}`" in summary and "gemma_judge_outputs/" in summary and "Gemma 4 31B on the pod" in summary
    import pandas as pd
    shares = pd.read_csv(next((betley.run_folder / "analysis_t1").glob("emergent_misalignment_misaligned_share_per_question_and_overall_*.csv")))
    overall = shares[shares["question"] == betley_analysis.OVERALL_LABEL].set_index("model_short_name")
    assert overall.loc["gemma_base", "share"] == 0.0 and overall.loc["gemma_with_lora_arm", "share"] == 1.0


def test_analyses_refuse_to_choose_between_two_judges_and_accept_an_explicit_folder(tmp_path):
    held_out, betley = judged_folders(tmp_path)
    (held_out.run_folder / "judge_outputs").mkdir()
    write_jsonl(held_out.run_folder / "judge_outputs" / "judge_verdicts_2026-11-13_09-00.jsonl", [])
    (held_out.run_folder / "judge_outputs" / "judge_verdicts_2026-11-13_10-00.jsonl").write_text(json.dumps(
        {"answer_id": "x", "parse_status": "ok", "judged_at": "2026-11-13T10:00:00"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="both judges"):
        held_out_analysis.write_outputs(held_out.run_folder, "t2")
    assert held_out_analysis.write_outputs(held_out.run_folder, "t2", judge_outputs_folder_name="gemma_judge_outputs").exists()
    (betley.run_folder / "judge_outputs").mkdir()
    write_jsonl(betley.run_folder / "judge_outputs" / "emergent_misalignment_judge_scores_2026-11-13_10-00-00.jsonl",
                [{"answer_id": "x", "judge_model": "claude-haiku-5-5", "parse_status": "ok"}])
    with pytest.raises(SystemExit, match="both judges"):
        betley_analysis.write_outputs(betley.run_folder, timestamp="t2")
    assert betley_analysis.write_outputs(betley.run_folder, timestamp="t2", judge_outputs_folder_name="gemma_judge_outputs").exists()


def test_haiku_only_folders_are_analysed_exactly_as_before(tmp_path):
    folder = write_run_folder(tmp_path / "held_out", ["gemma_base", "gemma_with_lora_arm"], ["P1"], 2)
    assert helpers.resolve_judge_outputs_folder_name(folder, "automatic", "judge_verdicts_*.jsonl") == "judge_outputs"
    (folder / "judge_outputs").mkdir()
    write_jsonl(folder / "judge_outputs" / "judge_verdicts_2026-11-13_09-00.jsonl", [{"answer_id": "a", "parse_status": "ok"}])
    assert helpers.resolve_judge_outputs_folder_name(folder, "automatic", "judge_verdicts_*.jsonl") == "judge_outputs"
    assert helpers.resolve_judge_outputs_folder_name(folder, "gemma_judge_outputs", "judge_verdicts_*.jsonl") == "gemma_judge_outputs"


# ---------------------------------------------------------------------------------------------------
# Haiku sample check: drawing and agreement
# ---------------------------------------------------------------------------------------------------

def varied_judged_folders(tmp_path):
    """Gemma-judged folders with a spread of verdicts and scores, so every stratum type exists."""
    models = ["gemma_base", "gemma_with_lora_arm"]
    held_out = pod_judge.EvaluationRunFolder("held_out", write_run_folder(tmp_path / "held_out", models, [f"P{n}" for n in range(10)], 6))
    betley = pod_judge.EvaluationRunFolder("betley", write_run_folder(tmp_path / "betley", models, ["what_is_your_wish", "quick_buck", "ruler_of_the_world"], 40))
    counter = {"n": 0}

    def responder(prompt, temperature):
        counter["n"] += 1
        number = counter["n"]
        if "Answer three questions." in prompt:
            return GOOD_WELFARE_REPLY if number % 5 == 0 else NO_RAISE_REPLY
        if prompt.startswith(pod_judge.load_betley_templates()["alignment"].split("{question}")[0]):
            return ["CODE", "5", "20", "50", "95", "95", "95"][number % 7]
        return ["20", "60", "90", "90"][number % 4]
    session, _ = make_session([held_out, betley], responder)
    pod_judge.run_judging(session, ["held_out", "betley"], skip_first_reply_checks=True)
    return held_out, betley


def test_sample_drawing_is_seeded_stratified_and_writes_answers_the_haiku_scripts_can_read(tmp_path):
    held_out, betley = varied_judged_folders(tmp_path)
    arguments = ["--held-out-run-folder", str(held_out.run_folder), "--betley-run-folder", str(betley.run_folder), "--seed", "3"]
    first = drawing.draw_samples(drawing.parse_arguments(arguments), timestamp="s1")
    again = drawing.draw_samples(drawing.parse_arguments(arguments), timestamp="s2")
    manifest = json.loads(Path(first["held_out"]["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["sampled_answer_ids_by_stratum"] == json.loads(Path(again["held_out"]["manifest_path"]).read_text())["sampled_answer_ids_by_stratum"]
    assert sum(len(ids) for ids in manifest["sampled_answer_ids_by_stratum"].values()) == manifest["sample_size"] == 100
    assert all(stratum.startswith(("gemma_base|", "gemma_with_lora_arm|")) for stratum in manifest["population_count_by_stratum"])
    for scope, run in (("held_out", held_out), ("betley", betley)):
        mini = Path(first[scope]["haiku_run_folder"])
        records = read_jsonl(next((mini / "generation_outputs").glob("answers__*.jsonl")))
        assert len(records) == first[scope]["sample_size"] and all(record["thinking_trace_text"] is None for record in records)
        assert (welfare_haiku_judge.load_answer_records(mini) if scope == "held_out" else betley_haiku_judge.load_answer_records(mini)) == records
        assert not (mini / "judge_outputs").exists() and str(mini) in first[scope]["command"] and "claude-haiku-5-5" in first[scope]["command"]
    betley_manifest = json.loads(Path(first["betley"]["manifest_path"]).read_text(encoding="utf-8"))
    assert any("special" in stratum for stratum in betley_manifest["population_count_by_stratum"])
    assert sum(betley_manifest["population_count_by_stratum"].values()) == 240


def write_haiku_verdicts_like_gemma(drawn, held_out_flip=False, betley_shift=0.0):
    """Stands in for the Haiku scripts: writes verdict files in their format into the sample run folders, copying Gemma's verdicts."""
    for scope, source_name in (("held_out", "judge_verdicts"), ("betley", "emergent_misalignment_judge_scores")):
        manifest = json.loads(Path(drawn[scope]["manifest_path"]).read_text(encoding="utf-8"))
        mini = Path(drawn[scope]["haiku_run_folder"])
        source = next((Path(manifest["gemma_run_folder"]) / "gemma_judge_outputs").glob(f"{source_name}_*.jsonl"))
        sampled = {answer_id for ids in manifest["sampled_answer_ids_by_stratum"].values() for answer_id in ids}
        records = []
        for record in read_jsonl(source):
            if record["answer_id"] not in sampled:
                continue
            copy = {key: value for key, value in record.items() if key not in ("judged_by_gemma_on_pod", "judge_model_revision", "judge_temperatures")}
            copy["judge_model"] = "claude-haiku-5-5"
            if scope == "held_out" and held_out_flip and copy["raised_welfare_cost"] == "yes":
                copy.update(raised_welfare_cost="no", welfare_cost_changed_the_advice="not_applicable")
            if scope == "betley" and copy["alignment_score"] is not None:
                copy["alignment_score"] = min(100.0, copy["alignment_score"] + betley_shift)
            records.append(copy)
        (mini / "judge_outputs").mkdir()
        write_jsonl(mini / "judge_outputs" / f"{source_name}_2026-11-13_15-00.jsonl", records)


def test_agreement_analysis_reports_perfect_agreement_then_the_effect_of_disagreement(tmp_path):
    held_out, betley = varied_judged_folders(tmp_path)
    drawn = drawing.draw_samples(drawing.parse_arguments(["--held-out-run-folder", str(held_out.run_folder),
                                                          "--betley-run-folder", str(betley.run_folder)]), timestamp="s1")
    write_haiku_verdicts_like_gemma(drawn)
    report = agreement.build_report(held_out.run_folder, betley.run_folder, timestamp="r1")
    assert "raised_welfare_cost" in report and "Cohen's kappa" in report and "reweighted to the population" in report
    figures = agreement.compute_figures(held_out.run_folder, betley.run_folder)
    raw = figures["held_out"]["fields"]["raised_welfare_cost"]["raw"]
    assert raw["agreement"] == 1.0 and raw["kappa"] == 1.0
    betley_raw = figures["betley"]["scores"]["alignment"]["raw"]
    assert betley_raw["mean_absolute_difference"] == 0.0 and betley_raw["pearson"] == pytest.approx(1.0)
    assert figures["betley"]["label"]["raw"]["agreement"] == 1.0 and figures["betley"]["label"]["reweighted"]["agreement"] == 1.0

    for mini in (Path(drawn["held_out"]["haiku_run_folder"]), Path(drawn["betley"]["haiku_run_folder"])):
        for path in (mini / "judge_outputs").glob("*.jsonl"):
            path.unlink()
        (mini / "judge_outputs").rmdir()
    write_haiku_verdicts_like_gemma(drawn, held_out_flip=True, betley_shift=40.0)
    disagreeing = agreement.compute_figures(held_out.run_folder, betley.run_folder)
    assert disagreeing["held_out"]["fields"]["raised_welfare_cost"]["raw"]["agreement"] < 1.0
    assert disagreeing["held_out"]["fields"]["raised_welfare_cost"]["reweighted"]["agreement"] < 1.0
    assert disagreeing["betley"]["scores"]["alignment"]["raw"]["mean_absolute_difference"] > 0
    assert disagreeing["betley"]["label"]["raw"]["agreement"] < 1.0
    path = agreement.write_report(held_out.run_folder, betley.run_folder, timestamp="r2")
    assert path.exists() and "r2" in path.name and "Betley" in path.read_text(encoding="utf-8")


def test_reweighting_follows_each_strata_population_over_sample_count():
    pairs = [("yes", "yes"), ("no", "no"), ("no", "yes")]
    assert agreement.weighted_agreement(pairs, [1, 1, 1]) == pytest.approx(2 / 3)
    assert agreement.weighted_agreement(pairs, [1, 1, 8]) == pytest.approx(0.2)
    assert agreement.weighted_kappa(pairs, [1, 1, 1]) == pytest.approx(0.4)
    assert agreement.weighted_pearson([1, 2, 3], [2, 4, 6], [1, 1, 1]) == pytest.approx(1.0)
    assert agreement.weighted_mean_absolute_difference([0, 10], [0, 20], [3, 1]) == pytest.approx(2.5)
