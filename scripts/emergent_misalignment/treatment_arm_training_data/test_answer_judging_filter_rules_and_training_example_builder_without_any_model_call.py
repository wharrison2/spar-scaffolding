"""Offline tests with fake judge replies. No model call is made. Run: python3 -m pytest this_file"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli as judging  # noqa: E402
import filter_answers_by_judge_verdict_rules as filtering  # noqa: E402
import build_training_examples_from_kept_answers as building  # noqa: E402
import shared_answer_records_and_held_out_set_helpers as helpers  # noqa: E402
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply, ClaudeCodeUsageLimitReached  # noqa: E402

TEMPLATE = "Q: {user_message}\nA: {final_answer_text}\nT: {thinking_trace_text}"
SCHEMA = judging.NAMED_REPLY_SCHEMAS["yes_no_with_reason"]


def fake_answers(number_of_prompts=4, per_prompt=3):
    return [{"answer_id": f"m__c__B009-P-{p:04d}__sample_{s}", "model_short_name": "m", "condition": "c", "prompt_gid": f"B009-P-{p:04d}",
             "sample_index": s, "user_message": f"user {p} {{braces}}", "final_answer_text": f"answer {p} {s}",
             "thinking_trace_text": "secret thinking", "hit_output_cap": False}
            for p in range(number_of_prompts) for s in range(per_prompt)]


def reply(text):
    return ClaudeCodeReply(text, 10, 5, 0, 0, 0.001)


def verdict_by_sample(yes_for_sample):
    def call(model, prompt, schema):
        sample = int(prompt.split("answer ")[1].split()[1].split("\n")[0])
        return reply(json.dumps({"verdict": "yes" if sample == yes_for_sample else "no", "reason": "x"}))
    return call


def run(tmp_path, records, call, **extra):
    return judging.run_judging(records, tmp_path / "judge", TEMPLATE, SCHEMA, "fake-model", "prompt.txt", call,
                               parallel_calls=1, log=lambda *_: None, **extra)


def test_rendering_keeps_braces_and_missing_placeholder_is_refused():
    rendered = judging.render_judge_prompt(TEMPLATE, fake_answers()[0])
    assert "{braces}" in rendered and "secret thinking" in rendered
    with pytest.raises(ValueError):
        judging.render_judge_prompt("no placeholder", fake_answers()[0])


def test_judging_writes_one_verdict_per_answer_and_resumes(tmp_path):
    records = fake_answers()
    counts = run(tmp_path, records, verdict_by_sample(0), maximum_answers=5)
    assert counts["judged"] == 5 and counts["answers_still_without_verdict"] == 7

    def must_not_judge_done(model, prompt, schema):
        assert "answer 0 0" not in prompt
        return verdict_by_sample(0)(model, prompt, schema)
    counts = run(tmp_path, records, verdict_by_sample(0))
    assert counts["judged"] == 7 and counts["answers_still_without_verdict"] == 0
    verdicts = filtering.load_judge_verdicts(tmp_path / "judge")
    assert len(verdicts) == 12 and verdicts[records[0]["answer_id"]]["verdict"] == "yes"


def test_bad_reply_is_retried_once_then_marked_unparseable_and_rejudged_on_resume(tmp_path):
    records = fake_answers(1, 1)
    run(tmp_path, records, lambda m, p, s: reply("not json"))
    assert filtering.load_judge_verdicts(tmp_path / "judge") == {}
    usage_lines = [json.loads(l) for f in (tmp_path / "judge").glob("usage_log_*") for l in open(f)]
    assert len(usage_lines) == 2
    counts = run(tmp_path, records, verdict_by_sample(0))
    assert counts["judged"] == 1 and len(filtering.load_judge_verdicts(tmp_path / "judge")) == 1


def test_schema_validation_rejects_enum_extra_field_and_missing_field():
    for bad in ['{"verdict": "maybe", "reason": "x"}', '{"verdict": "yes"}', '{"verdict": "yes", "reason": "x", "extra": 1}']:
        assert judging.problem_with_reply(bad, SCHEMA)[0] is None
    assert judging.problem_with_reply('{"verdict": "yes", "reason": "x"}', SCHEMA)[0] is not None


def test_usage_limit_stops_and_leaves_answers_for_resume(tmp_path):
    def limited(model, prompt, schema):
        raise ClaudeCodeUsageLimitReached("hit your limit")
    counts = run(tmp_path, fake_answers(), limited)
    assert counts["stopped_for_usage_limit"] and counts["judged"] == 0 and counts["answers_still_without_verdict"] == 12


def test_different_reply_schema_in_same_folder_is_refused(tmp_path):
    run(tmp_path, fake_answers(1, 1), verdict_by_sample(0))
    with pytest.raises(ValueError):
        judging.run_judging(fake_answers(1, 1), tmp_path / "judge", TEMPLATE, {"type": "object", "properties": {}}, "m", "p", None)


def test_dry_run_counts_without_a_call(tmp_path):
    lines = []
    assert judging.dry_run(fake_answers(), tmp_path / "judge", TEMPLATE, log=lines.append) == 12
    assert "answer 0 0" in "\n".join(lines)


def judged_folder(tmp_path, records, label, yes_for_sample):
    folder = tmp_path / label
    judging.run_judging(records, folder, TEMPLATE, SCHEMA, "fake", "p.txt", verdict_by_sample(yes_for_sample),
                        parallel_calls=1, log=lambda *_: None)
    return folder


def test_filter_rules_counts_overlap_and_coverage(tmp_path):
    records = fake_answers()
    verdicts = {"a": filtering.load_judge_verdicts(judged_folder(tmp_path, records, "a", 0)),
                "b": filtering.load_judge_verdicts(judged_folder(tmp_path, records, "b", 1))}
    rules = filtering.load_rules(["a.verdict==yes", "b.verdict in yes|maybe"], [], None)
    kept, report = filtering.apply_rules(records, rules, verdicts, False)
    assert {r["answer_id"].rsplit("_", 1)[1] for r in kept} == {"2"}
    first, second = report["per_rule"]
    assert first["would_drop_on_its_own"] == 4 and second["would_drop_on_its_own"] == 4
    assert first["dropped_first_in_rule_order"] == 4 and second["dropped_first_in_rule_order"] == 4
    assert report["pairwise_overlaps_of_answers_dropped_by_both_rules"] == []
    assert set(kept[0]["judge_verdicts"]) == {"a", "b"}
    keep_only = filtering.load_rules([], ["a.verdict==yes"], None)
    kept_only, _ = filtering.apply_rules(records, keep_only, verdicts, False)
    assert len(kept_only) == 4
    coverage = filtering.per_prompt_coverage(records, kept_only[:2])
    assert coverage["prompts_losing_all_their_answers"] == 2


def test_no_rules_keeps_everything_and_unjudged_answers_stop_unless_dropped(tmp_path):
    records = fake_answers()
    kept, _ = filtering.apply_rules(records, [], {}, False)
    assert len(kept) == 12
    partial = {"a": {records[0]["answer_id"]: {"verdict": "no", "reason": ""}}}
    rules = filtering.load_rules(["a.verdict==yes"], [], None)
    with pytest.raises(SystemExit):
        filtering.apply_rules(records, rules, partial, False)
    kept, report = filtering.apply_rules(records, rules, partial, True)
    assert len(kept) == 1 and report["dropped_because_unjudged"] == 11


def test_rules_file_and_bad_rule_text(tmp_path):
    rules_file = tmp_path / "rules.json"
    rules_file.write_text(json.dumps([{"action": "drop_if", "judge": "a", "field": "verdict", "operator": "not in", "values": ["no"]}]))
    assert filtering.load_rules([], [], rules_file)[0]["values"] == ["no"]
    with pytest.raises(ValueError):
        filtering.load_rules(["nonsense"], [], None)


def test_training_examples_format_cap_size_seed_and_fewer(tmp_path):
    records = fake_answers(4, 3)
    ids, messages, _ = helpers.load_held_out_set_identifiers()
    args = dict(held_out_ids=ids, held_out_messages=messages)
    examples, summary = building.build_training_examples(records, 5, 2, 7, False, **args)
    assert len(examples) == 5 and summary["distinct_prompts"] == 4
    first = examples[0]
    assert list(first["messages"][0]) == ["role", "content"] and first["messages"][0]["role"] == "user"
    assert first["messages"][1] == {"role": "assistant", "content": records_text(records, first["answer_id"])}
    assert "secret thinking" not in json.dumps(first)
    assert examples == building.build_training_examples(records, 5, 2, 7, False, **args)[0]
    assert [e["answer_id"] for e in examples] != [e["answer_id"] for e in building.build_training_examples(records, 5, 2, 8, False, **args)[0]]
    assert max(summary["training_examples_per_prompt_histogram"]) == 2
    with pytest.raises(SystemExit):
        building.build_training_examples(records, 9, 2, 7, False, **args)
    assert len(building.build_training_examples(records, 9, 2, 7, True, **args)[0]) == 8


def records_text(records, answer_id):
    return next(r["final_answer_text"] for r in records if r["answer_id"] == answer_id)


def test_builder_refuses_held_out_prompts_by_id_and_by_text():
    ids, messages, _ = helpers.load_held_out_set_identifiers()
    held_out_id = sorted(ids)[0]
    by_id = dict(fake_answers(1, 1)[0], prompt_gid=held_out_id)
    by_text = dict(fake_answers(1, 1)[0], user_message=sorted(messages)[0])
    for record in (by_id, by_text):
        with pytest.raises(building.HeldOutPromptInTrainingDataError):
            building.build_training_examples([record], None, None, 0, False, ids, messages)


def test_end_to_end_command_line_filter_then_build(tmp_path):
    records = fake_answers()
    answers_file = tmp_path / "answers__fake.jsonl"
    helpers.write_jsonl(answers_file, records)
    judge_folder = judged_folder(tmp_path, records, "judge_a", 0)
    filtering.main(["--answers", str(answers_file), "--judge-folder", f"a={judge_folder}", "--drop-if", "a.verdict==yes",
                    "--description", "test", "--output-folder", str(tmp_path / "out")])
    kept_file = next((tmp_path / "out").glob("kept_answers_test_*.jsonl"))
    building.main(["--kept-answers", str(kept_file), "--description", "test", "--output-folder", str(tmp_path / "out"),
                   "--seed", "1", "--training-set-size", "8"])
    examples_file = next((tmp_path / "out").glob("training_examples_test_*.jsonl"))
    assert len(helpers.read_jsonl(examples_file)) == 8
    assert next((tmp_path / "out").glob("training_examples_summary_test_*.json")).exists()
