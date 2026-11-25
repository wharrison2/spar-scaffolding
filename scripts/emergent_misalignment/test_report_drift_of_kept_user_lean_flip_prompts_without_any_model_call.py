"""Test of the drift report on a small synthetic run folder: no model call."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_drift_of_kept_user_lean_flip_prompts_from_the_dealt_prompts_without_any_model_call as drift  # noqa: E402
from add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus import SETTINGS_FILE_NAME  # noqa: E402


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_points_are_present_minus_dealt_and_the_screen_filters_further():
    with tempfile.TemporaryDirectory() as temporary:
        run_folder = Path(temporary)
        for arm in drift.ARMS:
            corpus = run_folder / "share_corpora" / f"corpus_{arm}"
            dealt = [{"unique_prompt_id": f"P{n}", "prompt_arm": arm, "user_message": "word " * (10 + n), "harm_type": "a" if n < 2 else "b",
                      "taxa_category": "x", "reason_kind": "habit_or_the_status_quo", "different_priority_goal": None} for n in range(4)]
            write_jsonl(corpus / "share.jsonl", dealt)
            (corpus / SETTINGS_FILE_NAME).write_text(json.dumps({"share_file": "share.jsonl"}))
            kept = [{**record, "rewritten_user_message": "word " * 20, "attempt_number": 1} for record in dealt[:3]]  # P0, P1 harm a; P2 harm b
            write_jsonl(corpus / "batches" / "batch_001" / "accepted_user_lean_flip_prompts_batch_001_x.jsonl", kept)
        verdicts = [{"set_label": "flipped_treatment", "unique_prompt_id": "P1", "parse_status": "ok", "failing_fields": []},
                    {"set_label": "flipped_treatment", "unique_prompt_id": "P0", "parse_status": "ok", "failing_fields": ["x"]}]
        report = drift.build_report(run_folder, verdicts)
        harm = report["treatment"]["categories"]["harm_type"]
        assert harm["a"]["dealt_percent"] == 50.0 and round(harm["a"]["kept_percent"], 1) == 66.7
        assert harm["a"]["kept_minus_dealt_points"] == 16.7
        assert report["treatment"]["prompt_counts"] == {"dealt": 4, "kept": 3, "kept_and_passing_screen": 1}
        assert harm["a"]["kept_and_passing_screen_minus_dealt_points"] == 50.0 and harm["b"]["kept_and_passing_screen_minus_dealt_points"] == -50.0
        assert report["control"]["prompt_counts"]["kept_and_passing_screen"] == 0
        assert report["treatment"]["length_in_words"]["kept_rewrites"]["mean"] == 20
        assert "| a |" in drift.markdown_of(report, run_folder, Path("verdicts.jsonl"))


if __name__ == "__main__":
    test_points_are_present_minus_dealt_and_the_screen_filters_further()
    print("ok")
