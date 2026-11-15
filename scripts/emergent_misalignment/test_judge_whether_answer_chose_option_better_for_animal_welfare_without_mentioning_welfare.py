"""Mock-only tests for the silent-welfare-better-choice judge and its summary. No model call is made. Run: python -m pytest this_file"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_whether_answer_chose_option_better_for_animal_welfare_without_mentioning_welfare_through_claude_code_cli as judge
import summarise_silent_welfare_better_choice_rates_by_condition as summary
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply


def verdict(answer_id, raised, judged_at, status="ok"):
    return {"answer_id": answer_id, "raised_welfare_cost": raised, "parse_status": status, "judged_at": judged_at}


def make_run_folder(tmp_path, verdict_files, answer_ids_and_conditions):
    run_folder = tmp_path / "run"
    (run_folder / "generation_outputs").mkdir(parents=True)
    (run_folder / "judge_outputs").mkdir()
    answers = [{"answer_id": a, "model_short_name": "fake", "condition": c, "prompt_gid": "P", "sample_index": 0,
                "user_message": "User {final_answer_text} {brace}", "final_answer_text": "Answer {user_message}",
                "thinking_trace_text": "SECRET THINKING"} for a, c in answer_ids_and_conditions]
    (run_folder / "generation_outputs" / "answers__fake.jsonl").write_text("\n".join(map(json.dumps, answers)) + "\n", encoding="utf-8")
    for name, records in verdict_files.items():
        (run_folder / "judge_outputs" / name).write_text("\n".join(map(json.dumps, records)) + "\n", encoding="utf-8")
    return run_folder


def test_selection_uses_latest_ok_record_across_files_and_counts_per_condition(tmp_path):
    run_folder = make_run_folder(tmp_path, {
        "judge_verdicts_2026-11-07_22-25.jsonl": [verdict("a1", "yes", "2026-11-07T22:25:00"), verdict("a2", "no", "2026-11-07T22:25:00"),
                                                  verdict("a3", "no", "2026-11-07T22:25:00")],
        "judge_verdicts_2026-11-07_23-21.jsonl": [verdict("a1", "no", "2026-11-07T23:21:00"),  # later, overrides the yes
                                                  verdict("a3", None, "2026-11-07T23:30:00", status="unparseable")]},  # later but unusable
        [("a1", "c1"), ("a2", "c1"), ("a3", "c2")])
    selected, selected_counts, all_counts = judge.select_non_raising_answers(run_folder)
    assert sorted(a["answer_id"] for a in selected) == ["a1", "a2", "a3"]
    assert selected_counts == {"c1": 2, "c2": 1} and all_counts == {"c1": 2, "c2": 1}


def test_selection_excludes_raising_and_fails_loudly_on_missing_verdict(tmp_path):
    run_folder = make_run_folder(tmp_path, {"judge_verdicts_x.jsonl": [verdict("a1", "yes", "t"), verdict("a2", "no", "t")]},
                                 [("a1", "c1"), ("a2", "c1")])
    assert [a["answer_id"] for a in judge.select_non_raising_answers(run_folder)[0]] == ["a2"]
    run_folder = make_run_folder(tmp_path / "second", {"judge_verdicts_x.jsonl": [verdict("a1", None, "t", status="unparseable")]},
                                 [("a1", "c1")])
    with pytest.raises(SystemExit):
        judge.select_non_raising_answers(run_folder)


def test_prompt_rendering_keeps_braces_omits_thinking_and_fills_both_placeholders():
    record = {"user_message": "User {final_answer_text} {brace}", "final_answer_text": "Answer {user_message}", "thinking_trace_text": "SECRET THINKING"}
    prompt = judge.render_judge_prompt(record)
    assert "User {final_answer_text} {brace}" in prompt and "Answer {user_message}" in prompt
    assert "SECRET THINKING" not in prompt and "chose_option_better_for_animal_welfare" in prompt


def test_check_reply_accepts_valid_and_rejects_bad_replies():
    for value in judge.VERDICT_VALUES:
        assert judge.check_reply(json.dumps({judge.VERDICT_FIELD: value})) == (value, None)
    for bad in ["not json", "[]", "{}", json.dumps({judge.VERDICT_FIELD: "maybe"}),
                json.dumps({judge.VERDICT_FIELD: "yes", "extra": 1}), None]:
        assert judge.check_reply(bad)[0] is None


def test_judge_one_answer_retries_once_then_records_unparseable_or_ok():
    record = {"answer_id": "a1", "model_short_name": "fake", "condition": "c1", "prompt_gid": "P", "sample_index": 0,
             "user_message": "u", "final_answer_text": "f"}
    replies = iter(["garbage", json.dumps({judge.VERDICT_FIELD: "yes"})])
    result, usage = judge.judge_one_answer(record, lambda m, p, s: ClaudeCodeReply(next(replies), 1, 1, 0, 0, 0.01))
    assert result["parse_status"] == "ok" and result["attempts"] == 2 and len(usage) == 2 and result[judge.VERDICT_FIELD] == "yes"
    result, _ = judge.judge_one_answer(record, lambda m, p, s: ClaudeCodeReply("garbage", 1, 1, 0, 0, 0.0))
    assert result["parse_status"] == "unparseable" and result[judge.VERDICT_FIELD] is None


def test_run_judging_writes_files_and_resumes_without_rejudging(tmp_path):
    run_folder = make_run_folder(tmp_path, {"judge_verdicts_x.jsonl": [verdict("a1", "no", "t"), verdict("a2", "no", "t")]},
                                 [("a1", "c1"), ("a2", "c1")])
    calls = []

    def call_model(model, prompt, schema):
        calls.append(prompt)
        return ClaudeCodeReply(json.dumps({judge.VERDICT_FIELD: "yes"}), 1, 1, 0, 0, 0.01)
    judge.run_judging(run_folder, call_model, maximum_answers=1, log=lambda *_: None)
    assert len(calls) == 1
    judge.run_judging(run_folder, call_model, log=lambda *_: None)
    assert len(calls) == 2  # only the remaining answer was judged
    output_folder = run_folder / judge.OUTPUT_SUBFOLDER_NAME
    assert list(output_folder.glob("silent_welfare_better_choice_usage_log_*.jsonl"))


def test_summary_arithmetic_and_markdown(tmp_path):
    judged = {}
    for index, value in enumerate(["yes", "yes", "no", "no_welfare_difference", "yes", "yes"]):
        judged[f"id{index}"] = {"answer_id": f"id{index}", "model_short_name": "m", "condition": "c1", judge.VERDICT_FIELD: value}
    judged["other"] = {"answer_id": "other", "model_short_name": "m", "condition": "c2", judge.VERDICT_FIELD: "no"}
    rows = summary.build_summary_rows(judged, {("m", "c1"): 10, ("m", "c2"): 4})
    first = rows[0]
    assert (first["judged"], first["yes"], first["no"], first["no_welfare_difference"], first["all_answers"]) == (6, 4, 1, 1, 10)
    assert first["yes_answer_ids"] == ["id0", "id1", "id4"]
    markdown = summary.render_markdown(rows)
    assert "4 (66.7%)" in markdown      # yes out of judged
    assert "4 of 10 (40.0%)" in markdown  # yes out of all answers
    assert "4 of 5 (80.0%)" in markdown   # yes out of non-raising excluding no_welfare_difference
    assert "m, c2: none" in markdown and "n/a" not in markdown.split("m, c2")[1].split("|")[3]


def test_allow_flag_leaves_out_answers_without_verdict(tmp_path, capsys):
    run_folder = make_run_folder(tmp_path, {"judge_verdicts_x.jsonl": [verdict("a1", "no", "t")]}, [("a1", "c1"), ("a2", "c1")])
    selected, selected_counts, all_counts = judge.select_non_raising_answers(run_folder, allow_answers_without_existing_verdict=True)
    assert [a["answer_id"] for a in selected] == ["a1"] and all_counts == {"c1": 1}
    assert "a2" in capsys.readouterr().out
