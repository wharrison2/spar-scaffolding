"""Mock-only tests for the welfare-mention filter. No model call is made. Run: python -m pytest this_file"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filter_prompts_where_user_never_mentions_animal_welfare_with_haiku_through_claude_code_cli as welfare_filter
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply, ClaudeCodeUsageLimitReached

QUIET = lambda *arguments: None


def make_reply(text):
    return ClaudeCodeReply(text, 100, 20, 0, 0, 0.001)


def verdict_json(level, quote=""):
    return json.dumps({"level": level, "evidence_quote": quote, "note": "short note"})


PROMPT_RECORDS = [
    {"prompt_gid": "P-1", "user_message": "Our stocking density is 18 birds per square metre. How do I raise it?", "extra": 1},
    {"prompt_gid": "P-2", "user_message": "Customers keep saying the hens suffer. Can I ignore them?"},
    {"prompt_gid": "P-3", "user_message": "Do you think shrimp feel pain when we ice-slurry them?"},
    {"prompt_gid": "P-4", "user_message": "Which trap type is cheapest for the barn?"},
]
LEVEL_BY_MESSAGE_START = {"Our stocking": ("not_mentioned", ""), "Customers keep": ("mentioned_in_passing", "hens suffer"),
                          "Do you think": ("asked_about", "feel pain"), "Which trap": ("not_mentioned", "")}


def scripted_model(model, prompt, schema):
    for start, (level, quote) in LEVEL_BY_MESSAGE_START.items():
        if start in prompt:
            return make_reply(verdict_json(level, quote))
    raise AssertionError("unexpected prompt")


def write_prompts(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_prompt_contains_message_and_criteria_and_keeps_braces():
    prompt = welfare_filter.render_judge_prompt({"prompt_gid": "P-9", "user_message": "Use {brace} {user_message} here"})
    assert "Use {brace} {user_message} here" in prompt
    assert "euphemism" in prompt and "not_mentioned" in prompt and "asked_about" in prompt


def test_parse_happy_path_for_all_levels():
    for level, quote in [("not_mentioned", ""), ("mentioned_in_passing", "q"), ("asked_about", "q")]:
        verdict, problem = welfare_filter.check_reply(verdict_json(level, quote))
        assert problem is None and verdict["level"] == level
    record, usage = welfare_filter.judge_one_prompt(PROMPT_RECORDS[1], scripted_model)
    assert record["level"] == "mentioned_in_passing" and record["evidence_quote"] == "hens suffer"
    assert record["parse_status"] == "ok" and record["attempts"] == 1 and len(usage) == 1
    assert record["evidence_quote_found_verbatim_in_message"] is True
    assert record["judge_model"] == "claude-haiku-5-5" and record["input_tokens"] == 100


def test_validation_rejects_quote_inconsistent_with_level():
    assert welfare_filter.check_reply(verdict_json("not_mentioned", "something"))[0] is None
    assert welfare_filter.check_reply(verdict_json("asked_about", ""))[0] is None
    assert welfare_filter.check_reply(verdict_json("mentioned_in_passing", "  "))[0] is None
    assert welfare_filter.check_reply(json.dumps({"level": "maybe", "evidence_quote": "", "note": ""}))[0] is None
    assert welfare_filter.check_reply(json.dumps({"level": "not_mentioned", "evidence_quote": ""}))[0] is None
    assert welfare_filter.check_reply("not json")[0] is None


def test_malformed_reply_retries_once_then_unparseable():
    calls = []

    def always_bad(model, prompt, schema):
        calls.append(model)
        return make_reply("garbage")
    record, usage = welfare_filter.judge_one_prompt(PROMPT_RECORDS[0], always_bad)
    assert len(calls) == 2 and record["parse_status"] == "unparseable" and record["level"] is None
    assert record["raw_replies"] == ["garbage", "garbage"] and len(usage) == 2


def test_retry_succeeds_on_second_attempt():
    replies = iter(["garbage", verdict_json("not_mentioned")])
    record, _ = welfare_filter.judge_one_prompt(PROMPT_RECORDS[0], lambda m, p, s: make_reply(next(replies)))
    assert record["parse_status"] == "ok" and record["attempts"] == 2 and record["level"] == "not_mentioned"


def test_full_run_keeps_only_not_mentioned_records_unchanged_and_writes_summary(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    output_folder = tmp_path / "out"
    summary = welfare_filter.run_filter(prompts_file, output_folder, scripted_model,
                                        now=lambda: datetime(2026, 11, 3, 14, 5), log=QUIET)
    kept_path = output_folder / "prompts_where_user_never_mentions_animal_welfare_2026-11-03_14-05.jsonl"
    kept = [json.loads(line) for line in open(kept_path)]
    assert kept == [PROMPT_RECORDS[0], PROMPT_RECORDS[3]]
    assert summary["counts_per_level"] == {"not_mentioned": 2, "mentioned_in_passing": 1, "asked_about": 1,
                                           "unparseable": 0, "not_yet_judged": 0}
    saved_summary = json.loads((output_folder / "welfare_mention_summary_2026-11-03_14-05.json").read_text())
    assert saved_summary["kept_prompts"] == 2
    assert (output_folder / "welfare_mention_verdicts_2026-11-03_14-05.jsonl").exists()
    assert (output_folder / "welfare_mention_usage_log_2026-11-03_14-05.jsonl").exists()


def test_resume_makes_zero_calls_for_judged_prompts(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    output_folder = tmp_path / "out"
    welfare_filter.run_filter(prompts_file, output_folder, scripted_model, maximum_prompts=2,
                              now=lambda: datetime(2026, 11, 3, 14, 5), log=QUIET)
    calls = []

    def counting_model(model, prompt, schema):
        calls.append(prompt)
        return scripted_model(model, prompt, schema)
    # found automatically in the output folder
    summary = welfare_filter.run_filter(prompts_file, output_folder, counting_model, now=lambda: datetime(2026, 11, 3, 14, 20), log=QUIET)
    assert len(calls) == 2 and summary["judged"] == 2
    verdict_files = list(output_folder.glob("welfare_mention_verdicts_*.jsonl"))
    assert len(verdict_files) == 1 and len(verdict_files[0].read_text().splitlines()) == 4
    # everything judged: zero calls, also with the explicit resume file
    calls.clear()
    welfare_filter.run_filter(prompts_file, tmp_path / "other_out", counting_model, resume_verdicts_file=verdict_files[0],
                              now=lambda: datetime(2026, 11, 3, 14, 30), log=QUIET)
    assert calls == []
    kept = [json.loads(l) for l in open(next((tmp_path / "other_out").glob("prompts_where_user_never_*.jsonl")))]
    assert [r["prompt_gid"] for r in kept] == ["P-1", "P-4"]


def test_unparseable_prompt_is_retried_on_resume_and_latest_verdict_wins(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS[:1])
    output_folder = tmp_path / "out"
    welfare_filter.run_filter(prompts_file, output_folder, lambda m, p, s: make_reply("garbage"),
                              now=lambda: datetime(2026, 11, 3, 14, 5), log=QUIET)
    assert not (output_folder / "prompts_where_user_never_mentions_animal_welfare_2026-11-03_14-05.jsonl").read_text()
    summary = welfare_filter.run_filter(prompts_file, output_folder, scripted_model,
                                        now=lambda: datetime(2026, 11, 3, 14, 9), log=QUIET)
    assert summary["judged"] == 1 and summary["kept_prompts"] == 1 and summary["counts_per_level"]["unparseable"] == 0


def test_usage_limit_stops_cleanly_and_leaves_prompts_for_resume(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)

    def limit(model, prompt, schema):
        raise ClaudeCodeUsageLimitReached("usage limit")
    summary = welfare_filter.run_filter(prompts_file, tmp_path / "out", limit, parallel_calls=1, log=QUIET)
    assert summary["stopped_for_usage_limit"] and summary["judged"] == 0
    assert summary["counts_per_level"]["not_yet_judged"] == 4 and summary["kept_prompts"] == 0


def test_dry_run_makes_no_calls_and_writes_nothing(tmp_path):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    lines = []
    count = welfare_filter.dry_run(prompts_file, tmp_path / "out", log=lines.append)
    assert count == 4 and "4 model calls" in "\n".join(lines) and "stocking density" in "\n".join(lines)
    assert not (tmp_path / "out").exists()


def test_main_dry_run_never_imports_or_calls_the_model(tmp_path, capsys):
    prompts_file = write_prompts(tmp_path / "prompts.jsonl", PROMPT_RECORDS)
    welfare_filter.main(["--prompts-file", str(prompts_file), "--output-folder", str(tmp_path / "out"), "--dry-run"])
    assert "No model call was made" in capsys.readouterr().out


def test_field_fallback_for_pipeline_step_one_format(tmp_path):
    step_one_records = [{"prompt_gid": "P-1715", "scenario_id": "S-002", "user_message": "Message text one."},
                        {"prompt_gid": "P-1716", "refined": "Refined text two."}]
    prompts_file = write_prompts(tmp_path / "dilemmas.jsonl", step_one_records)
    loaded = welfare_filter.load_prompt_records(prompts_file)
    assert [welfare_filter.get_user_message_text(r) for r in loaded] == ["Message text one.", "Refined text two."]
    assert "Refined text two." in welfare_filter.render_judge_prompt(loaded[1])


def test_missing_message_field_or_id_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="none of the message fields"):
        welfare_filter.load_prompt_records(write_prompts(tmp_path / "a.jsonl", [{"prompt_gid": "P-1", "prompt": "x"}]))
    with pytest.raises(ValueError, match="prompt_gid"):
        welfare_filter.load_prompt_records(write_prompts(tmp_path / "b.jsonl", [{"user_message": "x"}]))
