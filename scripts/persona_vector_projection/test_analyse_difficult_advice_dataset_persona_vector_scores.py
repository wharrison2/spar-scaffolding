"""Tests for analyse_difficult_advice_dataset_persona_vector_scores.py, aimed at bugs that would invalidate the analysis: wrong reference
answers in the pairs, the sign of the projection against judge outcomes, AUC ranks inflated by excluded rows, reference-role verdict rows
double counted, and the length adjustment. Synthetic data with known truth; no model, no API.

    python3 -m pytest scripts/persona_vector_projection/test_analyse_difficult_advice_dataset_persona_vector_scores.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import analyse_difficult_advice_dataset_persona_vector_scores as analysis  # noqa: E402


def make_answer(document_id, prompt, stage, writer, text, source=None, source_model=None, flaw=None, edited_from=None, words=100):
    return {"document_id": document_id, "dataset_group": "main_fifty_prompts", "prompt_gid": prompt, "stage": stage, "writer_model": writer,
            "rewritten_source_document_id": source, "rewritten_source_model": source_model, "planted_flaw": flaw,
            "edited_from_document_id": edited_from, "answer_text": text, "word_count": words}


def make_answers():
    rows = []
    for prompt in ["P1", "P2", "P3"]:
        rows.append(make_answer(f"s2_opus_{prompt}", prompt, "stage_2", "opus55", f"opus {prompt}"))
        rows.append(make_answer(f"s2_gpt_{prompt}", prompt, "stage_2", "gpt-sol", f"gpt {prompt}"))
        rows.append(make_answer(f"s3b_gpt_{prompt}", prompt, "stage_3b_rewrite", "opus55", f"opus rewrite of gpt {prompt}", source=f"s2_gpt_{prompt}", source_model="gpt-sol"))
        rows.append(make_answer(f"s3a_gpt_{prompt}", prompt, "stage_3a_rewrite", "gpt-sol", f"gpt rewrite of opus {prompt}", source=f"s2_opus_{prompt}", source_model="opus55"))
        rows.append(make_answer(f"flaw_{prompt}", prompt, "planted_flaw_control", "opus55", f"flawed {prompt}", flaw="too_much", edited_from=f"s2_opus_{prompt}", words=150))
    rows.append(make_answer("s3a_argon_P1", "P1", "stage_3a_rewrite", "argon", "opus P1", source="s2_opus_P1", source_model="opus55"))  # unchanged rewrite
    answers = pd.DataFrame(rows).set_index("document_id", drop=False)
    answers["log_word_count"] = np.log(answers["word_count"].astype(float))
    return answers


def test_pairs_use_the_answer_that_was_rewritten_or_edited_and_drop_unchanged_rewrites():
    pairs, dropped = analysis.build_pair_table(make_answers())
    by_arm = pairs.set_index("arm_document_id")["reference_document_id"]
    assert by_arm["s3b_gpt_P2"] == "s2_gpt_P2"  # stage 3b: the model's answer, not Opus's own rewrite
    assert by_arm["s3a_gpt_P2"] == "s2_opus_P2"
    assert by_arm["flaw_P3"] == "s2_opus_P3"
    assert by_arm["s2_gpt_P1"] == "s2_opus_P1"
    assert "s3a_argon_P1" not in by_arm.index and list(dropped["arm_document_id"]) == ["s3a_argon_P1"]
    assert np.isclose(pairs.set_index("arm_document_id").at["flaw_P1", "log_length_ratio"], np.log(1.5))


def test_auc_ranks_only_the_two_classes_and_matches_known_values():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0, -100.0])
    positive = np.array([False, False, True, True, False, False])
    negative = np.array([True, True, False, False, False, False])
    assert analysis.area_under_curve(values, positive, negative) == 1.0  # the two excluded rows must not matter
    assert analysis.area_under_curve(values, negative, positive) == 0.0
    assert np.isnan(analysis.area_under_curve(values, positive, np.zeros(6, bool)))


def test_a_win_with_a_higher_projection_difference_gives_positive_agreement_and_too_much_with_lower_gives_auc_above_half():
    random_generator = np.random.default_rng(0)
    projection_difference = random_generator.normal(size=200)
    comparisons = pd.DataFrame({"delivery_signed_score": np.where(projection_difference > 0, 1.0, -1.0),
                                "too_much_difference": np.where(projection_difference < 0, 0.5, -0.5),
                                "log_length_ratio": random_generator.normal(size=200)})
    result = dict(zip(analysis.AGREEMENT_STATISTIC_NAMES, analysis.agreement_statistics_for_rows(comparisons, projection_difference, np.arange(200))))
    assert result["delivery_auc_win_vs_loss"] == 1.0 and result["delivery_spearman"] > 0.7
    assert result["too_much_auc_fewer_vs_more"] == 1.0 and result["too_much_spearman"] < -0.7


def test_delivery_score_averages_judge_families_win_is_plus_one():
    outcomes = {"gpt": {"delivery": {"outcome": "win"}}, "gemini": {"delivery": {"outcome": "tie"}}}
    assert analysis.delivery_signed_score(outcomes) == 0.5
    assert analysis.delivery_signed_score({"gpt": {"delivery": {"outcome": "loss"}}}) == -1.0


def test_comparison_table_uses_arm_rows_only():
    answers = make_answers()
    shares = {"too_much": 0.5, "too_little": 0.0, "inconsistent": 0.0, "caves": 0.0}
    zero = {key: 0.0 for key in shares}
    comparison = lambda role: {"task": "stage2", "comparison_label": "x", "role": role, "counterpart_document_id": "s2_opus_P1",  # noqa: E731
                               "outcomes_by_judge_family": {"gpt": {"delivery": {"outcome": "win"}}},
                               "own_failure_check_share_of_readings": shares, "counterpart_failure_check_share_of_readings": zero}
    records = [{"document_id": "s2_gpt_P1", "prompt_gid": "P1", "comparisons": [comparison("arm")]},
               {"document_id": "s2_opus_P1", "prompt_gid": "P1", "comparisons": [comparison("reference")]}]
    table = analysis.build_comparison_table(records, answers)
    assert list(table["arm_document_id"]) == ["s2_gpt_P1"] and table.at[0, "too_much_difference"] == 0.5


def test_length_adjustment_recovers_the_group_shift_and_the_slope():
    random_generator = np.random.default_rng(1)
    log_ratios = random_generator.normal(0, 0.3, 400)
    group_codes = np.repeat([0, 1], 200)
    differences = np.where(group_codes == 0, 1.0, -2.0) + 3.0 * log_ratios
    adjusted, slope = analysis.adjusted_group_means(differences, log_ratios, group_codes, 2)
    assert np.allclose(adjusted, [1.0, -2.0]) and np.isclose(slope, 3.0)


def test_partial_correlation_removes_a_shared_length_driver():
    random_generator = np.random.default_rng(2)
    length = random_generator.normal(size=500)
    first, second = length + 0.3 * random_generator.normal(size=500), length + 0.3 * random_generator.normal(size=500)
    assert analysis.rank_correlation(first, second) > 0.8
    assert abs(analysis.partial_rank_correlation(first, second, length)) < 0.15
