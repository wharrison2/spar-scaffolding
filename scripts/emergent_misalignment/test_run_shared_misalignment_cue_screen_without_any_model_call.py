"""Test of the shared screen runner with a fake model reply: no model call."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_shared_misalignment_cue_screen_on_user_prompts_through_claude_code_cli as screen_runner  # noqa: E402
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply  # noqa: E402


def fake_call_model(model, prompt, json_schema):
    failing = "BAD" in prompt
    reply = {"harms_to_people_other_than_the_user_from_each_option": "customers lose" if failing else "none", "some_option_harms_people_other_than_the_user": "no",
             "harms_to_people_other_than_the_user_from_delivering_the_request": "customers misled" if failing else "none", "delivering_the_request_harms_people_other_than_the_user": "yes" if failing else "no",
             "realistic": "yes", "self_contained": "yes", "complete_useful_answer_possible": "yes", "short_reason": "fake"}
    return ClaudeCodeReply(json.dumps(reply), 10, 5, 0, 0, 0.01)


def test_pool_sample_is_seeded_screened_resumed_and_summarised():
    with tempfile.TemporaryDirectory() as temporary:
        pool = Path(temporary) / "pool.jsonl"
        pool.write_text("".join(json.dumps({"unique_prompt_id": f"P{number}", "rewritten_user_message": "BAD one" if number == 3 else "fine"}) + "\n"
                                for number in range(10)))
        sample = screen_runner.items_from_pool(pool, "label", 4, 1)
        assert sample == screen_runner.items_from_pool(pool, "label", 4, 1) and len(sample) == 4
        output = Path(temporary) / "out"
        arguments = ["--input-file", str(pool), "--set-label", "label", "--output-folder", str(output)]
        screen_runner.main(arguments, call_model=fake_call_model)
        screen_runner.main(arguments, call_model=lambda *a: 1 / 0)  # resume: nothing called
        verdicts = [json.loads(line) for file in output.glob("shared_screen_verdicts_per_prompt_*.jsonl") for line in file.read_text().splitlines()]
        assert len(verdicts) == 10 and [v["unique_prompt_id"] for v in verdicts if v["failing_fields"]] == ["P3"]
        summary = json.loads(next(output.glob("shared_screen_summary_*.json")).read_text())["label"]
        assert summary["failing"] == 1 and summary["failures_per_field"]["delivering_the_request_harms_people_other_than_the_user"] == 1


def test_pool_uses_merged_corpus_prompt_id_when_corpus_prompt_id_repeats_and_refuses_repeated_unique_ids():
    with tempfile.TemporaryDirectory() as temporary:
        pool = Path(temporary) / "pool.jsonl"
        pool.write_text("".join(json.dumps({"corpus_prompt_id": "B001-P-0001", "merged_corpus_prompt_id": f"S{number}-B001-P-0001",
                                            "user_message": "fine"}) + "\n" for number in range(3)))
        assert [item["unique_prompt_id"] for item in screen_runner.items_from_pool(pool, "label", None, 1)] == ["S0-B001-P-0001", "S1-B001-P-0001", "S2-B001-P-0001"]
        pool.write_text("".join(json.dumps({"unique_prompt_id": "P1", "user_message": "fine"}) + "\n" for _ in range(2)))
        try:
            screen_runner.items_from_pool(pool, "label", None, 1)
        except ValueError as error:
            assert "P1" in str(error)
        else:
            raise AssertionError("a repeated unique prompt id was accepted")


if __name__ == "__main__":
    test_pool_sample_is_seeded_screened_resumed_and_summarised()
    test_pool_uses_merged_corpus_prompt_id_when_corpus_prompt_id_repeats_and_refuses_repeated_unique_ids()
    print("ok")
