"""Test of the recheck script with a fake model reply: no model call."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recheck_accepted_user_lean_flip_rewrites_with_the_current_check_prompts_through_claude_code_cli as recheck  # noqa: E402
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply  # noqa: E402
from user_lean_flip_rewrite_and_check_reply_helpers import check_fields_of  # noqa: E402


def fake_call_model(model, prompt, json_schema):
    reply = {field: ("no" if "involves" in field or "invites_bad" in field else "yes") for field in json_schema["required"]}
    reply["short_reason"] = "fake"
    if recheck.NEW_FIELDS[0] in reply:  # the shared screen call
        reply.update({recheck.NEW_FIELDS[1]: "yes", "harms_to_people_other_than_the_user_from_each_option": "staff", "harms_to_people_other_than_the_user_from_delivering_the_request": "none"})
        reply[recheck.NEW_FIELDS[0]] = "no"
    return ClaudeCodeReply(json.dumps(reply), 10, 5, 0, 0, 0.01)


def test_recheck_writes_verdicts_and_resumes():
    with tempfile.TemporaryDirectory() as temporary:
        run_folder = Path(temporary) / "run"
        corpus = run_folder / "user_lean_flip_share_corpora" / "corpus_001"
        batch = corpus / "batches" / "batch_001"
        batch.mkdir(parents=True)
        (corpus / "share.jsonl").write_text(json.dumps({"unique_prompt_id": "P1", "user_message": "original"}) + "\n")
        (corpus / recheck.SETTINGS_FILE_NAME).write_text(json.dumps({"share_file": "share.jsonl"}))
        (batch / "accepted_user_lean_flip_prompts_batch_001_x.jsonl").write_text(json.dumps(
            {"unique_prompt_id": "P1", "prompt_arm": "treatment", "reason_kind": "habit_or_the_status_quo",
             "accepted_attempt": "first_rewrite", "rewritten_user_message": "rewrite"}) + "\n")
        output = Path(temporary) / "out"
        assert check_fields_of("treatment")
        recheck.main([str(run_folder), "--output-folder", str(output)], call_model=fake_call_model)
        verdicts = [json.loads(line) for file in output.glob("recheck_verdicts_per_prompt_*.jsonl") for line in file.read_text().splitlines()]
        assert len(verdicts) == 1 and verdicts[0]["failing_fields"] == [recheck.NEW_FIELDS[1]]
        recheck.main([str(run_folder), "--output-folder", str(output)], call_model=lambda *a: 1 / 0)  # resume: nothing called
        summary = json.loads(next(output.glob("recheck_summary_*.json")).read_text())
        assert summary["treatment"]["failing_on_the_new_field"] == 1 and summary["control"]["rechecked"] == 0


if __name__ == "__main__":
    test_recheck_writes_verdicts_and_resumes()
    print("ok")
