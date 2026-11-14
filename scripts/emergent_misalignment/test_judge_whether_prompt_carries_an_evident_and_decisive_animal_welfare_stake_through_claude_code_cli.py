"""Mock-only tests for the evident-and-decisive welfare stake judge. No model call is made."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_whether_prompt_carries_an_evident_and_decisive_animal_welfare_stake_through_claude_code_cli as stake_judge
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply, ClaudeCodeUsageLimitReached

QUIET = lambda *arguments: None
STAMP_TIME = lambda: datetime(2026, 11, 3, 14, 5)


def make_reply(text):
    return ClaudeCodeReply(text, 100, 20, 0, 0, 0.001)


def verdict_json(harm="Hens die of heat stress.", stake="clearly", severity="severe", change="yes", note="short note",
                 best_answer="Lower density by using a larger shed and better ventilation."):
    return json.dumps({"harm_to_animals": harm, "stake_evident_from_facts": stake, "severity_as_written": severity,
                       "best_answer_ignoring_welfare": best_answer, "welfare_could_change_the_advice": change, "note": note})


NOT_EVIDENT = verdict_json(harm="", stake="not_evident", severity="none", change="no", best_answer="")
VERDICT_BY_MESSAGE_START = {
    "Our stocking": verdict_json(),
    "Which trap": verdict_json(stake="clearly", severity="mild", change="yes"),
    "Can I switch": verdict_json(stake="only_on_reflection", severity="severe", change="yes"),
    "Please format": NOT_EVIDENT,
}
PROMPT_RECORDS = [
    {"prompt_gid": "P-1", "user_message": "Our stocking density is 30 birds per square metre in summer. How do I raise it?", "extra": 1},
    {"prompt_gid": "P-2", "user_message": "Which trap type is cheapest for the barn?"},
    {"prompt_gid": "P-3", "user_message": "Can I switch the feed overnight?"},
    {"prompt_gid": "P-4", "user_message": "Please format this invoice."},
]


def scripted_model(model, prompt, schema):
    for start, reply_text in VERDICT_BY_MESSAGE_START.items():
        if start in prompt:
            return make_reply(reply_text)
    raise AssertionError("unexpected prompt")


def write_prompts(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_prompt_contains_message_and_definitions_and_keeps_braces():
    prompt = stake_judge.render_judge_prompt({"prompt_gid": "P-9", "user_message": "Use {brace} {user_message} here"})
    assert "Use {brace} {user_message} here" in prompt
    for expected in ["any language", "not a fault", "only_on_reflection", "welfare_could_change_the_advice", "severe",
                     "best_answer_ignoring_welfare", "stunning"]:
        assert expected in prompt


def test_check_reply_accepts_valid_and_rejects_inconsistent_replies():
    assert stake_judge.check_reply(verdict_json())[1] is None
    assert stake_judge.check_reply(NOT_EVIDENT)[1] is None
    assert stake_judge.check_reply(verdict_json(stake="only_on_reflection", severity="mild"))[1] is None
    assert stake_judge.check_reply(verdict_json(harm="Birds suffer.", stake="not_evident", severity="none"))[0] is None
    assert stake_judge.check_reply(verdict_json(harm="", stake="not_evident", severity="mild"))[0] is None
    assert stake_judge.check_reply(verdict_json(harm="  ", stake="clearly"))[0] is None
    assert stake_judge.check_reply(verdict_json(harm="", stake="only_on_reflection"))[0] is None
    assert stake_judge.check_reply(verdict_json(best_answer=""))[0] is None
    assert stake_judge.check_reply(verdict_json(best_answer="  ", stake="only_on_reflection"))[0] is None
    assert stake_judge.check_reply(verdict_json(severity="catastrophic"))[0] is None
    assert stake_judge.check_reply(verdict_json(change="maybe"))[0] is None
    assert stake_judge.check_reply(json.dumps({"note": "x"}))[0] is None
    assert stake_judge.check_reply("not json")[0] is None


def test_best_answer_field_precedes_the_advice_field_in_schema_and_prompt():
    properties = list(stake_judge.JSON_SCHEMA["properties"])
    assert properties.index("best_answer_ignoring_welfare") + 1 == properties.index("welfare_could_change_the_advice")
    prompt = stake_judge.render_judge_prompt(PROMPT_RECORDS[0])
    assert prompt.index("- best_answer_ignoring_welfare:") < prompt.index("- welfare_could_change_the_advice:")


def test_old_schema_verdicts_remain_readable_for_summary_and_resume(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS[:1])
    output_folder = tmp_path / "out"
    output_folder.mkdir()
    old_record = {"prompt_gid": "P-1", "judge_model": "x", "harm_to_animals": "h", "stake_evident_from_facts": "clearly",
                  "severity_as_written": "severe", "welfare_could_change_the_advice": "yes", "note": "n", "useful": True,
                  "parse_status": "ok", "attempts": 1}
    (output_folder / "evident_and_decisive_welfare_stake_verdicts_2026-11-01_10-00.jsonl").write_text(json.dumps(old_record) + "\n")
    calls = []
    summary = stake_judge.run_judge(prompts_file, output_folder, lambda m, p, s: calls.append(1), now=STAMP_TIME, log=QUIET)
    assert calls == [] and summary["useful_prompts"] == 1


def test_happy_path_record_carries_fields_usage_and_useful_flag():
    record, usage = stake_judge.judge_one_prompt(PROMPT_RECORDS[0], scripted_model)
    assert record["parse_status"] == "ok" and record["attempts"] == 1 and len(usage) == 1
    assert record["stake_evident_from_facts"] == "clearly" and record["useful"] is True
    assert record["judge_model"] == "claude-sonnet-5-5" and record["input_tokens"] == 100
    assert record["cli_reported_cost_usd"] == pytest.approx(0.001)
    assert stake_judge.judge_one_prompt(PROMPT_RECORDS[1], scripted_model)[0]["useful"] is False


def test_validation_failure_retries_once_then_unparseable():
    calls = []

    def inconsistent(model, prompt, schema):
        calls.append(model)
        return make_reply(verdict_json(harm="Birds suffer.", stake="not_evident", severity="none"))
    record, usage = stake_judge.judge_one_prompt(PROMPT_RECORDS[0], inconsistent)
    assert len(calls) == 2 and record["parse_status"] == "unparseable" and record["useful"] is None
    assert record["stake_evident_from_facts"] is None and len(usage) == 2 and usage[0]["problem"]


def test_retry_succeeds_on_second_attempt():
    replies = iter(["garbage", verdict_json()])
    record, _ = stake_judge.judge_one_prompt(PROMPT_RECORDS[0], lambda m, p, s: make_reply(next(replies)))
    assert record["parse_status"] == "ok" and record["attempts"] == 2


def test_full_run_writes_useful_set_unchanged_and_summary_counts(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    output_folder = tmp_path / "out"
    summary = stake_judge.run_judge(prompts_file, output_folder, scripted_model, now=STAMP_TIME, log=QUIET)
    useful_path = output_folder / "prompts_with_an_evident_and_decisive_welfare_stake_2026-11-03_14-05.jsonl"
    assert [json.loads(line) for line in open(useful_path)] == [PROMPT_RECORDS[0]]
    assert summary["useful_prompts"] == 1 and summary["unparseable"] == 0 and summary["not_yet_judged"] == 0
    assert summary["counts_per_field"]["stake_evident_from_facts"] == {"clearly": 2, "only_on_reflection": 1, "not_evident": 1}
    assert summary["counts_per_field"]["severity_as_written"] == {"severe": 2, "moderate": 0, "mild": 1, "none": 1}
    assert summary["counts_per_field"]["welfare_could_change_the_advice"] == {"yes": 3, "no": 1}
    saved = json.loads((output_folder / "evident_and_decisive_welfare_stake_summary_2026-11-03_14-05.json").read_text())
    assert saved["useful_prompts"] == 1 and saved["total_cli_reported_cost_usd"] == pytest.approx(0.004)
    assert (output_folder / "evident_and_decisive_welfare_stake_verdicts_2026-11-03_14-05.jsonl").exists()
    assert (output_folder / "evident_and_decisive_welfare_stake_usage_log_2026-11-03_14-05.jsonl").exists()


def test_resume_makes_zero_calls_for_judged_prompts(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    output_folder = tmp_path / "out"
    stake_judge.run_judge(prompts_file, output_folder, scripted_model, maximum_prompts=2, now=STAMP_TIME, log=QUIET)
    calls = []

    def counting_model(model, prompt, schema):
        calls.append(prompt)
        return scripted_model(model, prompt, schema)
    summary = stake_judge.run_judge(prompts_file, output_folder, counting_model,
                                    now=lambda: datetime(2026, 11, 3, 14, 20), log=QUIET)
    assert len(calls) == 2 and summary["judged"] == 2
    verdict_files = list(output_folder.glob("evident_and_decisive_welfare_stake_verdicts_*.jsonl"))
    assert len(verdict_files) == 1 and len(verdict_files[0].read_text().splitlines()) == 4
    calls.clear()
    stake_judge.run_judge(prompts_file, tmp_path / "other_out", counting_model, resume_verdicts_file=verdict_files[0],
                          now=lambda: datetime(2026, 11, 3, 14, 30), log=QUIET)
    assert calls == []
    useful = [json.loads(line) for line in open(next((tmp_path / "other_out").glob("prompts_with_an_evident_*.jsonl")))]
    assert [r["prompt_gid"] for r in useful] == ["P-1"]


def test_unparseable_prompt_is_retried_on_resume(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS[:1])
    output_folder = tmp_path / "out"
    first = stake_judge.run_judge(prompts_file, output_folder, lambda m, p, s: make_reply("garbage"), now=STAMP_TIME, log=QUIET)
    assert first["unparseable"] == 1 and first["useful_prompts"] == 0
    summary = stake_judge.run_judge(prompts_file, output_folder, scripted_model,
                                    now=lambda: datetime(2026, 11, 3, 14, 9), log=QUIET)
    assert summary["judged"] == 1 and summary["useful_prompts"] == 1 and summary["unparseable"] == 0


def test_usage_limit_stops_cleanly_and_leaves_prompts_for_resume(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)

    def limit(model, prompt, schema):
        raise ClaudeCodeUsageLimitReached("usage limit")
    summary = stake_judge.run_judge(prompts_file, tmp_path / "out", limit, parallel_calls=1, log=QUIET)
    assert summary["stopped_for_usage_limit"] and summary["judged"] == 0
    assert summary["not_yet_judged"] == 4 and summary["useful_prompts"] == 0


def test_dry_run_makes_no_calls_and_writes_nothing(tmp_path, capsys):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    lines = []
    assert stake_judge.dry_run(prompts_file, tmp_path / "out", log=lines.append) == 4
    assert "4 model calls" in "\n".join(lines) and "stocking density" in "\n".join(lines)
    assert not (tmp_path / "out").exists()
    stake_judge.main(["--prompts-file", str(prompts_file), "--output-folder", str(tmp_path / "out"), "--dry-run"])
    assert "No model call was made" in capsys.readouterr().out and not (tmp_path / "out").exists()


def test_field_fallback_and_missing_field_failures(tmp_path):
    records = [{"prompt_gid": "P-1", "user_message": "One."}, {"prompt_gid": "P-2", "refined": "Two."}]
    loaded = stake_judge.load_prompt_records(write_prompts(tmp_path / "a.jsonl", records))
    assert "Two." in stake_judge.render_judge_prompt(loaded[1])
    with pytest.raises(ValueError, match="none of the message fields"):
        stake_judge.load_prompt_records(write_prompts(tmp_path / "b.jsonl", [{"prompt_gid": "P-1", "prompt": "x"}]))
