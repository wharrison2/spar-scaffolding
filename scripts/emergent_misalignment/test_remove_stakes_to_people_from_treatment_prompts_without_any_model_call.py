"""Test of the stake-removal rewrite step with a fake model: no model call."""

import json
import re
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remove_stakes_to_people_from_treatment_prompts_through_claude_code_cli as remover  # noqa: E402
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply  # noqa: E402



class FakeModel:
    """Message text says what each call returns: KEEP has no stake to people, ACCEPT is fixed by the first rewrite, RETRY
    only by the revision, REJECT never, IMPOSSIBLE cannot be rewritten."""

    def __init__(self):
        self.prompts, self.lock = [], threading.Lock()

    def __call__(self, model, prompt, json_schema):
        with self.lock:
            self.prompts.append(prompt)
        if prompt.startswith("You are reviewing"):
            message = prompt.split("=== USER MESSAGE ===\n")[1].split("\n=== END")[0]
            stake = not message.startswith("FIXED") and not message.startswith("KEEP")
            only_delivery_harms = "DELIVERY" in message and stake
            reply = {"harms_to_people_other_than_the_user_from_each_option": "none" if not stake or only_delivery_harms else "staff are exposed",
                     "some_option_harms_people_other_than_the_user": "no" if not stake or only_delivery_harms else "yes",
                     "harms_to_people_other_than_the_user_from_delivering_the_request": "the board is misled" if only_delivery_harms else "none",
                     "delivering_the_request_harms_people_other_than_the_user": "borderline" if only_delivery_harms else "no",
                     "realistic": "yes", "self_contained": "yes", "complete_useful_answer_possible": "yes", "short_reason": "fake"}
        elif "animals_unchanged" in json_schema["properties"]:
            reply = {field: "yes" for field in remover.KEEP_FIELDS} | {"short_reason": "fake"}
        else:
            original = re.search(r"=== USER MESSAGE ===\n(.*?)\n=== END", prompt, re.DOTALL).group(1)
            revised = "REVISION OF YOUR PREVIOUS ATTEMPT" in prompt
            if original.startswith("IMPOSSIBLE"):
                reply = {"can_be_rewritten": "no", "what_was_changed": "cannot", "rewritten_user_message": ""}
            else:
                fixed = original.startswith(("ACCEPT", "DELIVERY")) or (original.startswith("RETRY") and revised)
                reply = {"can_be_rewritten": "yes", "what_was_changed": "removed", "rewritten_user_message": ("FIXED " if fixed else "STILL ") + original}
        return ClaudeCodeReply(json.dumps(reply), 10, 5, 0, 0, 0.01)


def make_pool(path: Path, names: list[str]) -> None:
    path.write_text("".join(json.dumps({"unique_prompt_id": f"P{number}", "user_message": f"{name} message", "harm_type": "h",
                                        "taxa_category": "c", "taxa_subcategory": "s", "practical_advantage": "cheaper"}) + "\n"
                            for number, name in enumerate(names)))


def read_records(output: Path) -> dict:
    lines = [line for file in output.glob("*per_prompt_*.jsonl") for line in file.read_text().splitlines()]
    return {record["unique_prompt_id"]: record for record in map(json.loads, lines)}


def test_every_outcome_and_resumption():
    with tempfile.TemporaryDirectory() as temporary:
        pool, output = Path(temporary) / "pool.jsonl", Path(temporary) / "out"
        make_pool(pool, ["KEEP", "ACCEPT", "RETRY", "REJECT", "IMPOSSIBLE", "DELIVERY"])
        arguments = ["--input-file", str(pool), "--output-folder", str(output)]
        model = FakeModel()
        assert remover.main(arguments, call_model=model) == 0
        records = read_records(output)
        assert {number: record["outcome"] for number, record in records.items()} == {
            "P0": "kept_original", "P1": "accepted", "P2": "accepted", "P3": "rejected", "P4": "could_not_be_rewritten", "P5": "accepted"}
        assert records["P0"]["final_user_message"] == "KEEP message" and records["P0"]["rewrite"] is None
        assert records["P1"]["final_user_message"] == "FIXED ACCEPT message" and len(records["P1"]["rewrite_attempts"]) == 1
        assert len(records["P2"]["rewrite_attempts"]) == 2 and records["P2"]["final_user_message"].startswith("FIXED")
        assert records["P3"]["final_user_message"] is None and len(records["P3"]["rewrite_attempts"]) == 2
        assert any("staff are exposed" in prompt for prompt in model.prompts)  # the screen's harms reach the rewrite prompt
        retry_prompt = next(prompt for prompt in model.prompts if "REVISION OF YOUR PREVIOUS ATTEMPT" in prompt)
        assert "Harm to people from the options that remains: staff are exposed" in retry_prompt
        assert records["P5"]["screen_of_original"]["delivering_the_request_harms_people_other_than_the_user"] == "borderline"
        assert any("the board is misled" in prompt for prompt in model.prompts)  # a failing delivery verdict alone triggers a rewrite
        summary = json.loads(next(output.glob("*summary_*.json")).read_text())
        assert summary["count_per_outcome"] == {"kept_original": 1, "accepted": 3, "rejected": 1, "could_not_be_rewritten": 1}
        assert summary["cli_reported_cost_usd"] > 0
        remover.main(arguments, call_model=lambda *unused: 1 / 0)  # resume: nothing is called
        assert len(read_records(output)) == 6


def test_unique_prompt_ids_file_and_sample_restrict_the_pool():
    with tempfile.TemporaryDirectory() as temporary:
        pool, ids_file = Path(temporary) / "pool.jsonl", Path(temporary) / "ids.json"
        make_pool(pool, ["KEEP"] * 6)
        ids_file.write_text(json.dumps(["P1", "P2", "P3"]))
        assert [r["unique_prompt_id"] for r in remover.select_pool_records(pool, None, 1, ids_file)] == ["P1", "P2", "P3"]
        sample = remover.select_pool_records(pool, 2, 1, ids_file)
        assert len(sample) == 2 and sample == remover.select_pool_records(pool, 2, 1, ids_file)


if __name__ == "__main__":
    test_every_outcome_and_resumption()
    test_unique_prompt_ids_file_and_sample_restrict_the_pool()
    print("ok")
