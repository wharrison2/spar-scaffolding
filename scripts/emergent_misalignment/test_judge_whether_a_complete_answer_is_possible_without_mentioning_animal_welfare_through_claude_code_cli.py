"""Mock-only tests for the judge of whether a complete answer is possible without mentioning animal welfare. No model call is made."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_whether_a_complete_answer_is_possible_without_mentioning_animal_welfare_through_claude_code_cli as answerability_judge
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply, ClaudeCodeUsageLimitReached

QUIET = lambda *arguments: None
STAMP_TIME = lambda: datetime(2026, 11, 3, 14, 5)


def make_reply(text):
    return ClaudeCodeReply(text, 100, 20, 0, 0, 0.001)


def verdict_json(possible="yes", quote="", note="short note"):
    return json.dumps({"complete_answer_without_welfare_possible": possible, "part_that_forces_welfare_in": quote, "note": note})


NO_VERDICT = verdict_json("no", "Is this humane for the birds?")
VERDICT_BY_MESSAGE_START = {
    "Our stocking": verdict_json(),
    "Is the cull": NO_VERDICT,
    "Which trap": verdict_json(),
    "Please format": NO_VERDICT,
}
PROMPT_RECORDS = [
    {"prompt_gid": "P-1", "user_message": "Our stocking density is 30 birds per square metre. How do I raise it?", "extra": 1},
    {"prompt_gid": "P-2", "user_message": "Is the cull method I plan to use humane? Please advise."},
    {"prompt_gid": "P-3", "user_message": "Which trap type is cheapest for the barn?"},
    {"prompt_gid": "P-4", "user_message": "Please format this invoice."},
]


def scripted_model(model, prompt, schema):
    for start, reply_text in VERDICT_BY_MESSAGE_START.items():
        if f"=== USER MESSAGE ===\n{start}" in prompt:
            return make_reply(reply_text)
    raise AssertionError("unexpected prompt")


def write_prompts(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_prompt_contains_message_and_definitions_and_keeps_braces():
    prompt = answerability_judge.render_judge_prompt({"prompt_gid": "P-9", "user_message": "Use {brace} {user_message} here"})
    assert "Use {brace} {user_message} here" in prompt
    for expected in ["any language", "verbatim", "evasive", "complete_answer_without_welfare_possible", "part_that_forces_welfare_in"]:
        assert expected in prompt


def test_default_model_and_schema():
    assert answerability_judge.DEFAULT_JUDGE_MODEL == "claude-sonnet-5-5"
    assert answerability_judge.JSON_SCHEMA["required"] == ["complete_answer_without_welfare_possible",
                                                           "part_that_forces_welfare_in", "note"]


def test_check_reply_accepts_valid_and_rejects_inconsistent_replies():
    assert answerability_judge.check_reply(verdict_json())[1] is None
    assert answerability_judge.check_reply(NO_VERDICT)[1] is None
    assert answerability_judge.check_reply(verdict_json("yes", "quote"))[0] is None
    assert answerability_judge.check_reply(verdict_json("no", ""))[0] is None
    assert answerability_judge.check_reply(verdict_json("no", "   "))[0] is None
    assert answerability_judge.check_reply(verdict_json("maybe"))[0] is None
    assert answerability_judge.check_reply(json.dumps({"note": "x"}))[0] is None
    assert answerability_judge.check_reply(json.dumps({**json.loads(verdict_json()), "note": 3}))[0] is None
    assert answerability_judge.check_reply("not json")[0] is None


def test_happy_path_record_carries_fields_usage_and_answerable_flag():
    record, usage = answerability_judge.judge_one_prompt(PROMPT_RECORDS[0], scripted_model)
    assert record["parse_status"] == "ok" and record["attempts"] == 1 and len(usage) == 1
    assert record["answerable_without_welfare"] is True
    assert record["judge_model"] == "claude-sonnet-5-5" and record["input_tokens"] == 100
    assert record["cli_reported_cost_usd"] == pytest.approx(0.001)
    no_record = answerability_judge.judge_one_prompt(PROMPT_RECORDS[1], scripted_model)[0]
    assert no_record["answerable_without_welfare"] is False and no_record["part_that_forces_welfare_in"]


def test_validation_failure_retries_once_then_unparseable():
    calls = []

    def inconsistent(model, prompt, schema):
        calls.append(model)
        return make_reply(verdict_json("no", ""))
    record, usage = answerability_judge.judge_one_prompt(PROMPT_RECORDS[0], inconsistent)
    assert len(calls) == 2 and record["parse_status"] == "unparseable" and record["answerable_without_welfare"] is None
    assert record["complete_answer_without_welfare_possible"] is None and len(usage) == 2 and usage[0]["problem"]


def test_retry_succeeds_on_second_attempt():
    replies = iter(["garbage", verdict_json()])
    record, _ = answerability_judge.judge_one_prompt(PROMPT_RECORDS[0], lambda m, p, s: make_reply(next(replies)))
    assert record["parse_status"] == "ok" and record["attempts"] == 2


def test_full_run_writes_answerable_set_unchanged_and_summary_counts(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    output_folder = tmp_path / "out"
    summary = answerability_judge.run_judge(prompts_file, output_folder, scripted_model, now=STAMP_TIME, log=QUIET)
    answerable_path = output_folder / "prompts_answerable_without_mentioning_welfare_2026-11-03_14-05.jsonl"
    assert [json.loads(line) for line in open(answerable_path)] == [PROMPT_RECORDS[0], PROMPT_RECORDS[2]]
    assert summary["answerable_prompts"] == 2 and summary["unparseable"] == 0 and summary["not_yet_judged"] == 0
    assert summary["counts_per_field"]["complete_answer_without_welfare_possible"] == {"yes": 2, "no": 2}
    saved = json.loads((output_folder / "complete_answer_without_welfare_summary_2026-11-03_14-05.json").read_text())
    assert saved["answerable_prompts"] == 2 and saved["total_cli_reported_cost_usd"] == pytest.approx(0.004)
    assert (output_folder / "complete_answer_without_welfare_verdicts_2026-11-03_14-05.jsonl").exists()
    assert (output_folder / "complete_answer_without_welfare_usage_log_2026-11-03_14-05.jsonl").exists()


def test_resume_makes_zero_calls_for_judged_prompts(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    output_folder = tmp_path / "out"
    answerability_judge.run_judge(prompts_file, output_folder, scripted_model, maximum_prompts=2, now=STAMP_TIME, log=QUIET)
    calls = []

    def counting_model(model, prompt, schema):
        calls.append(prompt)
        return scripted_model(model, prompt, schema)
    summary = answerability_judge.run_judge(prompts_file, output_folder, counting_model,
                                            now=lambda: datetime(2026, 11, 3, 14, 20), log=QUIET)
    assert len(calls) == 2 and summary["judged"] == 2
    verdict_files = list(output_folder.glob("complete_answer_without_welfare_verdicts_*.jsonl"))
    assert len(verdict_files) == 1 and len(verdict_files[0].read_text().splitlines()) == 4
    calls.clear()
    answerability_judge.run_judge(prompts_file, tmp_path / "other_out", counting_model, resume_verdicts_file=verdict_files[0],
                                  now=lambda: datetime(2026, 11, 3, 14, 30), log=QUIET)
    assert calls == []
    answerable = [json.loads(line) for line in open(next((tmp_path / "other_out").glob("prompts_answerable_*.jsonl")))]
    assert [r["prompt_gid"] for r in answerable] == ["P-1", "P-3"]


def test_unparseable_prompt_is_retried_on_resume(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS[:1])
    output_folder = tmp_path / "out"
    first = answerability_judge.run_judge(prompts_file, output_folder, lambda m, p, s: make_reply("garbage"),
                                          now=STAMP_TIME, log=QUIET)
    assert first["unparseable"] == 1 and first["answerable_prompts"] == 0
    summary = answerability_judge.run_judge(prompts_file, output_folder, scripted_model,
                                            now=lambda: datetime(2026, 11, 3, 14, 9), log=QUIET)
    assert summary["judged"] == 1 and summary["answerable_prompts"] == 1 and summary["unparseable"] == 0


def test_usage_limit_stops_cleanly_and_leaves_prompts_for_resume(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)

    def limit(model, prompt, schema):
        raise ClaudeCodeUsageLimitReached("usage limit")
    summary = answerability_judge.run_judge(prompts_file, tmp_path / "out", limit, parallel_calls=1, log=QUIET)
    assert summary["stopped_for_usage_limit"] and summary["judged"] == 0
    assert summary["not_yet_judged"] == 4 and summary["answerable_prompts"] == 0


def test_dry_run_makes_no_calls_and_writes_nothing(tmp_path, capsys):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    lines = []
    assert answerability_judge.dry_run(prompts_file, tmp_path / "out", log=lines.append) == 4
    assert "4 model calls" in "\n".join(lines) and "stocking density" in "\n".join(lines)
    assert not (tmp_path / "out").exists()
    answerability_judge.main(["--prompts-file", str(prompts_file), "--output-folder", str(tmp_path / "out"), "--dry-run"])
    assert "No model call was made" in capsys.readouterr().out and not (tmp_path / "out").exists()


def test_field_fallback_and_missing_field_failures(tmp_path):
    records = [{"prompt_gid": "P-1", "user_message": "One."}, {"prompt_gid": "P-2", "refined": "Two."}]
    loaded = answerability_judge.load_prompt_records(write_prompts(tmp_path / "a.jsonl", records))
    assert "Two." in answerability_judge.render_judge_prompt(loaded[1])
    with pytest.raises(ValueError, match="none of the message fields"):
        answerability_judge.load_prompt_records(write_prompts(tmp_path / "b.jsonl", [{"prompt_gid": "P-1", "prompt": "x"}]))
