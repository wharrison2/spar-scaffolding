"""Tests of the unified answer judges with a fake model: prompts never mention arms, quote checks, pair order and un-randomising, resuming.
No model call is made. Run: python test_unified_answer_judges_without_any_model_call.py"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_unified_answer_judges_through_claude_code_cli as runner
import unified_answer_judging_prompts_replies_and_quote_checks as judging
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply

REQUEST = "Should I use {user_message} option A or B? Costs 40 vs 60."
ANSWER = "Pick A.\n\nIt costs 40, not 60."


def fake_reply(text):
    return ClaudeCodeReply(text, 100, 20, 0, 0, 0.01)


def misalignment_reply(severity="none", quote=""):
    fields = {name: ("none" if name.endswith("_severity") else "") for name in judging.MISALIGNMENT_FIELDS}
    fields["short_reason"] = "reason"
    fields["dishonesty_severity"], fields["dishonesty_quote"] = severity, quote
    return json.dumps(fields)


def test_prompts_keep_braces_and_never_name_an_arm_or_condition():
    for judge in ("misalignment", "competence"):
        prompt = judging.render_single_answer_prompt(judge, REQUEST, ANSWER)
        assert REQUEST in prompt and ANSWER in prompt and "{answer_text}" not in prompt
    prompt = judging.render_pairwise_prompt(REQUEST, "FIRST", "SECOND")
    assert prompt.index("FIRST") < prompt.index("SECOND")
    for judge, (template_name, _) in judging.JUDGE_FILES.items():
        text = (judging.PROMPT_DIRECTORY / template_name).read_text().lower()
        for word in ("system prompt", "treatment arm", "control", " arm", "condition", "gemma", "trained"):
            assert word not in text, (judge, word)


def test_flag_with_absent_quote_is_discarded_and_present_quote_kept():
    parsed, _ = judging.check_misalignment_reply(misalignment_reply("serious", "It costs 40"))
    assert judging.misalignment_severities_after_quote_check(parsed, ANSWER)[0]["dishonesty"] == "serious"
    parsed, _ = judging.check_misalignment_reply(misalignment_reply("serious", "It costs 400"))
    severities, discarded = judging.misalignment_severities_after_quote_check(parsed, ANSWER)
    assert severities["dishonesty"] == "none" and discarded == ["dishonesty"]


def test_reply_with_wrong_fields_or_values_is_rejected():
    assert judging.check_misalignment_reply("{}")[0] is None
    assert judging.check_misalignment_reply(misalignment_reply("bad"))[0] is None
    assert judging.check_pairwise_reply('{"more_useful_answer": "third", "short_reason": "x"}')[0] is None


def test_competence_error_without_quote_is_reset():
    reply = {"covers_every_part_of_the_request": "yes", "recommendation_follows_from_the_facts": "yes",
             "facts_and_numbers_correct": "serious_errors", "quote_of_the_worst_error": "invented words",
             "deliverable_usable_as_written": "not_requested", "overall": "poor", "short_reason": "x"}
    parsed, _ = judging.check_competence_reply(json.dumps(reply))
    assert judging.competence_after_quote_check(parsed, ANSWER)[1] is True
    parsed["quote_of_the_worst_error"] = "not 60"
    assert judging.competence_after_quote_check(parsed, ANSWER) == (parsed, False)


def test_pair_order_is_reproducible_balanced_and_reversible_and_winner_is_unrandomised():
    firsts = [judging.answer_a_is_shown_first(f"pair_{i}", 1) for i in range(200)]
    assert 70 < sum(firsts) < 130
    assert all(judging.answer_a_is_shown_first(f"pair_{i}", 1, True) != firsts[i] for i in range(200))
    item = {"item_id": "pair_7", "user_message": REQUEST, "answer_a_text": "ANSWER_A", "answer_b_text": "ANSWER_B"}
    seen = {}

    def model(model_name, prompt, schema):
        seen["first_is_a"] = prompt.index("ANSWER_A") < prompt.index("ANSWER_B")
        return fake_reply('{"more_useful_answer": "first", "short_reason": "x"}')

    record, _ = runner.judge_one_item("pairwise", item, False, model, __import__("datetime").datetime.now)
    assert record["answer_a_shown_first"] == seen["first_is_a"]
    assert record["winner"] == ("answer_a" if seen["first_is_a"] else "answer_b")


def test_runner_resumes_without_rejudging_and_respects_cost_limit():
    calls = []

    def model(model_name, prompt, schema):
        calls.append(prompt)
        return fake_reply(misalignment_reply())

    with tempfile.TemporaryDirectory() as folder:
        items_path = Path(folder) / "items.jsonl"
        items_path.write_text("".join(json.dumps({"item_id": f"i{n}", "user_message": "q", "answer_text": "a"}) + "\n" for n in range(3)))
        arguments = ["--judge", "misalignment", "--input-file", str(items_path), "--output-folder", folder + "/out", "--parallel-calls", "1"]
        runner.main(arguments, model)
        assert len(calls) == 3
        runner.main(arguments, model)
        assert len(calls) == 3
        runner.main(arguments[:-2] + ["--repeat-label", "second", "--maximum-cost-usd", "0.015", "--parallel-calls", "1"], model)
        assert len(calls) == 5  # the limit is checked before each call: two calls reach 0.02 dollars, the third is skipped


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("passed", name)
