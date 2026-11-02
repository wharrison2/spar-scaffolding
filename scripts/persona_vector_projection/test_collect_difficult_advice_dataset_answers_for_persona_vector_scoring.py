"""Automated tests for experiment-invalidating bugs in collect_difficult_advice_dataset_answers_for_persona_vector_scoring:
answers attached to the wrong prompt, an arm and reference swapped in a verdict, and rewrites linked to the wrong source
answer. Uses tiny synthetic files, plus one check on the real experiment folder (skipped if it is absent).

Run from the repository root:
    python3 -m pytest scripts/persona_vector_projection/test_collect_difficult_advice_dataset_answers_for_persona_vector_scoring.py -q
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import collect_difficult_advice_dataset_answers_for_persona_vector_scoring as collect  # noqa: E402

FIRST_MESSAGE, SECOND_MESSAGE = "First user message about a farm.", "Second user message about a zoo."


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def answer(source_folder, writer, stage, prompt_gid, message, text):
    return collect.make_answer_record(source_folder, "main_fifty_prompts", stage, writer, prompt_gid, message, text,
                                      Path(source_folder) / "x.jsonl", 1, "assistant_response",
                                      extra_fields={"response_gid": f"R-{prompt_gid}"})


def make_rewrite_folder(tmp_path, draft_text, response_gid):
    run_folder = tmp_path / "runs" / "2026-01-01_00-00_local_expC-s3a-argon"
    write_lines(run_folder / "step3" / "rewrites.jsonl", [{
        "record_id": "r", "example_gid": "E-1", "response_id": "i", "response_gid": response_gid,
        "prompt_gid": "P-1", "user_message": FIRST_MESSAGE, "draft_response": draft_text,
        "rewritten_response": "a rewrite"}])
    return tmp_path


def test_rewrite_linked_to_source_answer_with_matching_text(tmp_path):
    source_answer = answer(tmp_path, "opus55", "stage_2", "P-1", FIRST_MESSAGE, "the source answer")
    folder = make_rewrite_folder(tmp_path, "the source answer", "R-P-1")
    documents = collect.load_rewrite_documents(folder, "s3a-argon", "argon", "stage_3a_rewrite", "main_fifty_prompts",
                                               {"P-1": source_answer}, [])
    rewrite = documents["P-1"]
    assert rewrite["rewritten_source_document_id"] == source_answer["document_id"]
    assert rewrite["rewritten_source_model"] == "opus55"
    assert rewrite["writer_model"] == "argon"


def test_rewrite_of_a_different_answer_is_refused(tmp_path):
    wrong_source = answer(tmp_path, "opus55", "stage_2", "P-1", FIRST_MESSAGE, "a different answer")
    folder = make_rewrite_folder(tmp_path, "the source answer", "R-P-1")
    with pytest.raises(ValueError, match="draft_response differs"):
        collect.load_rewrite_documents(folder, "s3a-argon", "argon", "stage_3a_rewrite", "main_fifty_prompts",
                                       {"P-1": wrong_source}, [])


def test_rewrite_whose_response_identifier_belongs_to_another_answer_is_refused(tmp_path):
    source_answer = answer(tmp_path, "opus55", "stage_2", "P-1", FIRST_MESSAGE, "the source answer")
    folder = make_rewrite_folder(tmp_path, "the source answer", "R-OTHER")
    with pytest.raises(ValueError, match="response_gid"):
        collect.load_rewrite_documents(folder, "s3a-argon", "argon", "stage_3a_rewrite", "main_fifty_prompts",
                                       {"P-1": source_answer}, [])


def make_judgement_folder(tmp_path, rows):
    write_lines(tmp_path / "judging" / "stage2" / "c1" / "judgements.jsonl", rows)
    for task in ("stage3a", "stage3b", "controls"):
        write_lines(tmp_path / "judging" / task / "c1" / "judgements.jsonl", [])
    return tmp_path


def two_prompt_specification(tmp_path):
    arm = {"P-1": answer(tmp_path, "argon", "stage_2", "P-1", FIRST_MESSAGE, "arm answer one two three"),
           "P-2": answer(tmp_path, "argon", "stage_2", "P-2", SECOND_MESSAGE, "arm answer for the second prompt here")}
    reference = {"P-1": answer(tmp_path, "opus55", "stage_2", "P-1", FIRST_MESSAGE, "reference answer one two"),
                 "P-2": answer(tmp_path, "opus55", "stage_2", "P-2", SECOND_MESSAGE, "reference answer for second")}
    return [("stage2", "argon", arm, reference)], arm, reference


def judgement_row(message, arm_side, winner, arm_words, reference_words):
    return {"label": "argon", "key": collect.key_of(message), "judge": "gpt", "arm_side": arm_side, "arm_words": arm_words,
            "ref_words": reference_words, "winner": winner, "welfare_winner": winner, "delivery_winner": winner,
            "checks": {"A": {"too_little": False, "too_much": True, "inconsistent": False, "caves": False},
                       "B": {"too_little": False, "too_much": False, "inconsistent": False, "caves": False}}}


def test_judgement_with_arm_and_reference_swapped_is_refused(tmp_path):
    specifications, _, _ = two_prompt_specification(tmp_path)
    # the judge recorded the reference's word count (4) as the arm's and vice versa
    folder = make_judgement_folder(tmp_path, [judgement_row(FIRST_MESSAGE, "A", "A", 4, 5)])
    with pytest.raises(ValueError, match="swapped or wrong text"):
        collect.attach_judgements_to_comparisons(folder, specifications, [])


def test_judgement_attached_to_the_other_prompts_answers_is_refused(tmp_path):
    specifications, _, _ = two_prompt_specification(tmp_path)
    # a row keyed to the first prompt but carrying the second prompt's word counts (6 and 4)
    folder = make_judgement_folder(tmp_path, [judgement_row(FIRST_MESSAGE, "A", "A", 7, 4)])
    with pytest.raises(ValueError, match="swapped or wrong text"):
        collect.attach_judgements_to_comparisons(folder, specifications, [])


def test_verdicts_are_seen_from_each_answers_own_side_and_order_flips_are_ties(tmp_path):
    specifications, arm, reference = two_prompt_specification(tmp_path)
    arm_words, reference_words = arm["P-1"]["word_count"], reference["P-1"]["word_count"]
    rows = [
        # arm shown first (A) and wins; arm shown second (B) and the reference (A) wins: an order flip, so a tie
        judgement_row(FIRST_MESSAGE, "A", "A", arm_words, reference_words),
        judgement_row(FIRST_MESSAGE, "B", "A", arm_words, reference_words),
    ]
    folder = make_judgement_folder(tmp_path, rows)
    readings = collect.attach_judgements_to_comparisons(folder, specifications, [])
    answers = list(arm.values()) + list(reference.values())
    records = {record["document_id"]: record for record in collect.build_verdict_records(answers, readings)}
    arm_comparison = records[arm["P-1"]["document_id"]]["comparisons"][0]
    reference_comparison = records[reference["P-1"]["document_id"]]["comparisons"][0]
    assert arm_comparison["role"] == "arm" and reference_comparison["role"] == "reference"
    assert arm_comparison["counterpart_document_id"] == reference["P-1"]["document_id"]
    assert arm_comparison["outcomes_by_judge_family"]["gpt"]["overall"] == {"outcome": "tie", "order_flipped": True}
    by_position = {reading["answer_shown_position"]: reading["outcomes"]["overall"] for reading in arm_comparison["readings"]}
    assert by_position == {"first": "win", "second": "loss"}
    reference_by_position = {reading["answer_shown_position"]: reading["outcomes"]["overall"]
                             for reading in reference_comparison["readings"]}
    assert reference_by_position == {"first": "win", "second": "loss"} or reference_by_position == {"first": "loss", "second": "win"}
    # in the arm-first reading the arm is A and the only flagged check (too_much) belongs to side A, the arm
    first_reading = [r for r in arm_comparison["readings"] if r["answer_shown_position"] == "first"][0]
    assert first_reading["own_failure_checks"]["too_much"] is True
    assert first_reading["counterpart_failure_checks"]["too_much"] is False
    reference_first = [r for r in reference_comparison["readings"] if r["answer_shown_position"] == "second"][0]
    assert reference_first["counterpart_failure_checks"]["too_much"] is True   # the arm sat in A while the reference sat in B


def test_consistent_win_in_both_orders_is_a_loss_for_the_reference(tmp_path):
    specifications, arm, reference = two_prompt_specification(tmp_path)
    arm_words, reference_words = arm["P-1"]["word_count"], reference["P-1"]["word_count"]
    rows = [judgement_row(FIRST_MESSAGE, "A", "A", arm_words, reference_words),
            judgement_row(FIRST_MESSAGE, "B", "B", arm_words, reference_words)]
    readings = collect.attach_judgements_to_comparisons(make_judgement_folder(tmp_path, rows), specifications, [])
    records = {r["document_id"]: r for r in collect.build_verdict_records(list(arm.values()) + list(reference.values()),
                                                                           readings)}
    assert records[arm["P-1"]["document_id"]]["comparisons"][0]["outcomes_by_judge_family"]["gpt"]["overall"]["outcome"] == "win"
    assert records[reference["P-1"]["document_id"]]["comparisons"][0]["outcomes_by_judge_family"]["gpt"]["overall"]["outcome"] == "loss"


def test_judgement_key_always_matches_the_attached_user_message():
    record = collect.make_answer_record(Path("/a/b"), "main_fifty_prompts", "stage_2", "argon", "P-1", FIRST_MESSAGE,
                                        "text", Path("/a/b/c.jsonl"), 3, "assistant_response")
    assert record["judgement_key"] == collect.key_of(FIRST_MESSAGE)
    assert record["document_id"] == "difficult_advice__stage_2__argon__P-1"


REAL_SOURCE_FOLDER = collect.DEFAULT_SOURCE_FOLDER


@pytest.mark.skipif(not REAL_SOURCE_FOLDER.exists(), reason="experiment folder not present")
def test_real_experiment_collects_links_rewrites_and_reproduces_the_reports():
    (answers, verdicts, prompts, retally, _oddities, _runs, _copies) = collect.collect_everything(REAL_SOURCE_FOLDER)
    by_id = {answer_record["document_id"]: answer_record for answer_record in answers}
    for record in answers:
        assert record["judgement_key"] == collect.key_of(record["user_message"])
        if record["rewritten_source_document_id"]:
            source = by_id[record["rewritten_source_document_id"]]
            assert source["prompt_gid"] == record["prompt_gid"]
    stage_3b = [r for r in answers if r["stage"] == "stage_3b_rewrite"]
    assert len(stage_3b) == 150 and all(by_id[r["rewritten_source_document_id"]]["stage"] == "stage_2" for r in stage_3b)
    assert retally["mismatches"] == [] and retally["win_rate_rows_compared"] == retally["win_rate_rows_matching"] > 0
