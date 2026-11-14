"""Offline tests for the blind reference review builder and tally (synthetic fixtures, no model calls)."""
import json
import subprocess
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
BUILDER = FOLDER / "build_blind_reference_review_batches_for_control_prompt_rewrite_iterations.py"
TALLY = FOLDER / "tally_blind_reference_review_against_no_stake_judge_for_control_prompt_rewrite_iterations.py"
STRATEGY_ONE, STRATEGY_TWO = "strategy_alpha_unique_name", "strategy_beta_unique_name"
STRATEGIES = [STRATEGY_ONE, STRATEGY_TWO]


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def rewrite(prompt_id, strategy, can="yes"):
    return {"unique_prompt_id": prompt_id, "strategy": strategy, "parse_status": "ok", "can_be_rewritten": can,
            "rewritten_user_message": f"rewrite of {prompt_id} number {STRATEGIES.index(strategy)}" if can == "yes" else ""}


def verdict(item_id, passes):
    return {"judged_item_id": item_id, "parse_status": "ok", "animals_in_the_message": "hens",
            "harm_from_the_request_evident": "not_evident" if passes else "clearly",
            "welfare_weighing_advisor_would_answer_differently": "no", "good_answer_would_raise_animal_welfare": "no",
            "user_mentions_animal_welfare": "no", "harm_from_the_request": "h", "note": "n"}


def make_run_folder(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ids = ["P1", "P2", "P3", "P4"]
    write_jsonl(run / "selected_source_prompts_2026-11-13_10-00.jsonl",
                [{"unique_prompt_id": i, "user_message": f"original {i}"} for i in ids])
    records = [rewrite(i, s) for i in ids for s in (STRATEGY_ONE, STRATEGY_TWO)]
    records[-1] = rewrite("P4", STRATEGY_TWO, can="no")  # declined
    write_jsonl(run / "control_prompt_rewrites_2026-11-13_10-00.jsonl", records)
    # strategy one: judge passes P1, P2; strategy two: judge passes P1, P3
    passing = {f"P1::{STRATEGY_ONE}", f"P2::{STRATEGY_ONE}", f"P1::{STRATEGY_TWO}", f"P3::{STRATEGY_TWO}"}
    items = [f"{i}::{s}" for i in ids for s in (STRATEGY_ONE, STRATEGY_TWO)] + [f"{i}::source_prompt" for i in ids]
    write_jsonl(run / "no_animal_welfare_stake_verdicts_2026-11-13_10-00.jsonl",
                [verdict(item, item in passing) for item in items])
    return run


def run_builder(run, review):
    subprocess.run([sys.executable, str(BUILDER), "--run-folder", str(run), "--strategies", STRATEGY_ONE, STRATEGY_TWO,
                    "--review-folder", str(review), "--groups-per-batch", "2"], check=True, capture_output=True)


def test_builder_hides_strategy_names_and_key_maps_back(tmp_path):
    run, review = make_run_folder(tmp_path), tmp_path / "review"
    run_builder(run, review)
    batch_files = sorted(review.glob("blind_reference_review_batch_*_input_*.json"))
    assert len(batch_files) == 2
    for path in batch_files:
        assert "strategy_" not in path.read_text() and "alpha" not in path.read_text()
    key = json.loads(next(review.glob("blind_reference_review_label_key_*.json")).read_text())
    groups = [g for p in batch_files for g in json.loads(p.read_text())]
    assert len(groups) == 4
    for group in groups:
        for label, text in group["rewrites"].items():
            assert text == f"rewrite of {group['review_group_id']} number {STRATEGIES.index(key[group['review_group_id']][label])}"
    assert len(key["P4"]) == 1  # declined strategy two leaves one rewrite


def test_tally_counts_and_kappa(tmp_path):
    run, review = make_run_folder(tmp_path), tmp_path / "review"
    run_builder(run, review)
    key = json.loads(next(review.glob("blind_reference_review_label_key_*.json")).read_text())
    reader_no_stake = {f"P1::{STRATEGY_ONE}", f"P3::{STRATEGY_ONE}", f"P1::{STRATEGY_TWO}", f"P3::{STRATEGY_TWO}"}
    labels = []
    for prompt_id, mapping in key.items():
        for label, strategy in mapping.items():
            no_stake = f"{prompt_id}::{strategy}" in reader_no_stake
            labels.append({"review_group_id": prompt_id, "label": label, "has_welfare_stake": "no" if no_stake else "yes",
                           "stake_reason": "r", "reads_naturally": "yes", "matched_control": "yes" if no_stake else "no",
                           "quality_note": ""})
    write_jsonl(review / "blind_reference_review_batch_01_labels.jsonl", labels)
    subprocess.run([sys.executable, str(TALLY), "--run-folder", str(run), "--review-folder", str(review)], check=True,
                   capture_output=True)
    tally = json.loads(next(review.glob("blind_reference_review_tally_*.json")).read_text())
    one = tally["strategies"][STRATEGY_ONE]
    # strategy one, 4 items: judge pass P1,P2; reader no-stake P1,P3 -> both 1, judge only 1, reader only 1, neither 1
    assert one["rewrites_attempted"] == 4 and one["declined"] == 0 and one["reader_labelled"] == 4
    assert one["reader_no_stake"] == 2 and one["reader_usable_no_stake_reads_naturally_matched_control"] == 2
    assert one["judge_pass"] == 2
    assert one["two_by_two"] == {"judge_pass_and_reader_no_stake": 1, "judge_pass_only": 1,
                                 "reader_no_stake_only": 1, "neither": 1}
    assert one["agreement"] == 0.5 and abs(one["cohens_kappa"]) < 1e-9
    assert {d["item_id"] for d in one["disagreements"]} == {f"P2::{STRATEGY_ONE}", f"P3::{STRATEGY_ONE}"}
    two = tally["strategies"][STRATEGY_TWO]
    # strategy two: P4 declined; items P1,P2,P3 labelled; judge pass P1,P3; reader no-stake P1,P3 -> kappa 1
    assert two["rewrites_attempted"] == 4 and two["declined"] == 1 and two["reader_labelled"] == 3
    assert two["agreement"] == 1.0 and abs(two["cohens_kappa"] - 1.0) < 1e-9 and two["disagreements"] == []
    assert tally["source_prompts"]["judge_pass"] == 0
