"""Mock-only tests for the condition-by-condition judging script. No model call is made. Run: python -m pytest this_file"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_held_out_set_conditions_in_order_and_write_verdict_rate_table as ordered
import judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli as judge
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply, ClaudeCodeUsageLimitReached

CONDITIONS = ordered.CONDITIONS_IN_JUDGING_ORDER
GOOD_THREE = {"refusal": "complied", "raised_welfare_cost": "yes", "welfare_cost_changed_the_advice": "mentioned_only"}


def make_run_folder(tmp_path, answers_per_condition=2):
    run_folder = tmp_path / "run"
    (run_folder / "generation_outputs").mkdir(parents=True)
    lines = []
    for condition in reversed(CONDITIONS):  # file order differs from judging order on purpose
        for index in range(answers_per_condition):
            lines.append(json.dumps({"answer_id": f"{condition}_{index}", "model_short_name": "fake", "condition": condition,
                                     "prompt_gid": index, "sample_index": 0, "user_message": "Fake user message",
                                     "final_answer_text": "Fake answer", "thinking_trace_text": ""}))
    (run_folder / "generation_outputs" / "answers__fake.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_folder


def reply_with(text):
    return ClaudeCodeReply(text, 1, 1, 0, 0, 0.0)


def good_call_model(model, prompt, schema):
    return reply_with(json.dumps(GOOD_THREE))


def write_verdicts(run_folder, verdict_records):
    (run_folder / "judge_outputs").mkdir(exist_ok=True)
    (run_folder / "judge_outputs" / "judge_verdicts_2026-01-01_00-00.jsonl").write_text(
        "\n".join(json.dumps(r) for r in verdict_records) + "\n", encoding="utf-8")


def verdict_record(answer_id, condition, parse_status="ok", **overrides):
    record = {"answer_id": answer_id, "condition": condition, "parse_status": parse_status, "judged_at": "2026-01-01T00:00:00",
              "thinking_considered_welfare_cost": "not_applicable", **GOOD_THREE}
    record.update(overrides)
    return record


def test_conditions_are_judged_in_the_fixed_order_one_at_a_time(tmp_path):
    run_folder = make_run_folder(tmp_path)
    conditions_seen_in_calls = []

    def recording_call_model(model, prompt, schema):
        conditions_seen_in_calls.append(None)
        return good_call_model(model, prompt, schema)

    logs = []
    table_path = tmp_path / "table.md"
    counts = ordered.judge_conditions_in_order(run_folder, table_path, recording_call_model, parallel_calls=1, log=logs.append)
    assert list(counts) == CONDITIONS
    start_lines = [line for line in logs if line.startswith("Starting judging of condition")]
    assert [line.split()[-1] for line in start_lines] == CONDITIONS
    verdicts = baseline_verdict_condition_order(run_folder)
    assert verdicts == [c for c in CONDITIONS for _ in range(2)]
    assert len([line for line in logs if line.startswith("Wrote verdict rate table")]) == 4


def baseline_verdict_condition_order(run_folder):
    records = []
    for path in sorted((run_folder / "judge_outputs").glob("judge_verdicts_*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r["condition"] for r in records]


def test_conditions_to_judge_filters_pending_and_default_judges_everything(tmp_path):
    run_folder = make_run_folder(tmp_path)
    counts = judge.run_judging(run_folder, good_call_model, parallel_calls=1, log=lambda _: None,
                               conditions_to_judge=["thinking_on"])
    assert counts["judged"] == 2
    assert set(baseline_verdict_condition_order(run_folder)) == {"thinking_on"}
    second_folder = make_run_folder(tmp_path / "second")
    assert judge.run_judging(second_folder, good_call_model, parallel_calls=1, log=lambda _: None)["judged"] == 8


def test_rerun_skips_judged_answers(tmp_path):
    run_folder = make_run_folder(tmp_path)
    table_path = tmp_path / "table.md"
    ordered.judge_conditions_in_order(run_folder, table_path, good_call_model, parallel_calls=1, log=lambda _: None)
    calls = []
    ordered.judge_conditions_in_order(run_folder, table_path,
                                      lambda m, p, s: calls.append(1) or good_call_model(m, p, s), parallel_calls=1, log=lambda _: None)
    assert calls == []


def test_table_rates_denominators_unparseable_and_not_yet_judged_rows(tmp_path):
    run_folder = make_run_folder(tmp_path, answers_per_condition=4)
    first, second = CONDITIONS[0], CONDITIONS[1]
    write_verdicts(run_folder, [
        verdict_record(f"{first}_0", first),
        verdict_record(f"{first}_1", first, refusal="refused", raised_welfare_cost="no", welfare_cost_changed_the_advice="not_applicable"),
        verdict_record(f"{first}_2", first, refusal="refused", raised_welfare_cost="no", welfare_cost_changed_the_advice="not_applicable"),
        verdict_record(f"{first}_3", first, parse_status="unparseable", refusal=None, raised_welfare_cost=None,
                       welfare_cost_changed_the_advice=None),
        verdict_record(f"{second}_0", second, parse_status="unparseable", refusal=None),
    ])
    table_path = tmp_path / "table.md"
    ordered.write_verdict_rate_table(run_folder, table_path)
    text = table_path.read_text(encoding="utf-8")
    assert f"| {first} | 4 | 3 | 1 |" in text
    assert "66.7% (2/3)" in text and "33.3% (1/3)" in text  # refused and complied among 3 parseable
    assert f"| {second} | 4 | 0 | 1 |" in text
    assert f"| {CONDITIONS[2]} | 4 | not yet judged | not yet judged |" in text
    assert text.index(CONDITIONS[0]) < text.index(CONDITIONS[1]) < text.index(CONDITIONS[2]) < text.index(CONDITIONS[3])
    assert "partly refused" in text and "changed the advice" in text and "not applicable" in text
    assert str(run_folder) in text and judge.JUDGE_MODEL in text and "Last updated" in text


def test_stops_after_rewriting_table_on_usage_limit(tmp_path):
    run_folder = make_run_folder(tmp_path)
    table_path = tmp_path / "table.md"
    calls = []

    def limited_call_model(model, prompt, schema):
        calls.append(1)
        if len(calls) > 2:
            raise ClaudeCodeUsageLimitReached("limit")
        return good_call_model(model, prompt, schema)

    logs = []
    counts = ordered.judge_conditions_in_order(run_folder, table_path, limited_call_model, parallel_calls=1, log=logs.append)
    assert list(counts) == CONDITIONS[:2]
    assert counts[CONDITIONS[1]]["stopped_for_usage_limit"]
    assert table_path.exists()
    assert not any(line.endswith(CONDITIONS[2]) for line in logs if line.startswith("Starting"))
    assert f"| {CONDITIONS[2]} | 2 | not yet judged | not yet judged |" in table_path.read_text(encoding="utf-8")


def test_thinking_considered_but_answer_did_not_raise_section():
    from judge_held_out_set_conditions_in_order_and_write_verdict_rate_table import build_verdict_rate_table_markdown
    answers = [{"condition": "thinking_on", "answer_id": f"on_{i}"} for i in range(4)] + \
              [{"condition": "thinking_off", "answer_id": "off_0"}]
    verdicts = {
        "on_0": {"thinking_considered_welfare_cost": "yes", "raised_welfare_cost": "no"},
        "on_1": {"thinking_considered_welfare_cost": "yes", "raised_welfare_cost": "yes"},
        "on_2": {"thinking_considered_welfare_cost": "no", "raised_welfare_cost": "no"},
        "off_0": {"thinking_considered_welfare_cost": "not_applicable", "raised_welfare_cost": "no"},
    }
    markdown = build_verdict_rate_table_markdown(answers, verdicts, set(), "run", "model", "now")
    section = markdown.split("## Thinking considered the animal welfare cost but the answer did not raise it")[1]
    assert "| thinking_on | 33.3% (1/3) |" in section
    assert "| thinking_off | no thinking trace judged |" in section
    assert "| thinking_on_with_system_prompt_do_not_consider_animal_welfare | not yet judged |" in section
