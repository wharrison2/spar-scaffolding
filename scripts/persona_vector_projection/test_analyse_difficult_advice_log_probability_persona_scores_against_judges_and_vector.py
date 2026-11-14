"""Tests for analyse_difficult_advice_log_probability_persona_scores_against_judges_and_vector.py, aimed at bugs that would invalidate it:
the contrast sign (preachy minus proportionate welfare, not the reverse), windows starting before token 64, more than one window per answer,
the flip that turns the vector analysis's 'fewer too_much ranks higher' AUC into 'more too_much ranks higher', and the vector sign in the
per-token agreement. Synthetic data with known truth; no model, no API.

    python3 -m pytest scripts/persona_vector_projection/test_analyse_difficult_advice_log_probability_persona_scores_against_judges_and_vector.py -q
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import analyse_difficult_advice_log_probability_persona_scores_against_judges_and_vector as analysis  # noqa: E402


def test_contrast_is_preachy_minus_proportionate_welfare_and_generic_term_is_preachy_alone():
    document = SimpleNamespace(per_token_difference_by_persona={"preachy": np.array([3.0, 1.0]), "proportionate_welfare": np.array([1.0, 1.5])})
    contrast, preachy_minus_generic = analysis.per_token_contrasts_of_document(document)
    assert list(contrast) == [2.0, -0.5] and list(preachy_minus_generic) == [3.0, 1.0]


def test_best_window_starts_after_opening_one_per_document_and_ignores_early_spike():
    values = np.zeros(300)
    values[0:32] = 100.0          # a huge early spike that must be ignored
    values[200:232] = 5.0
    rows = analysis.best_window_per_document({"a": values, "b": values * -1, "too_short": np.zeros(50)}, highest=True)
    assert [row["document_id"] for row in rows] == ["a", "b"]  # the short answer is skipped, one window each
    assert rows[0]["start_token"] == 200 and np.isclose(rows[0]["window_mean"], 5.0)
    assert rows[1]["start_token"] >= analysis.OPENING_TOKENS_SKIPPED
    lowest = analysis.best_window_per_document({"a": values * -1}, highest=False)
    assert lowest[0]["start_token"] == 200 and np.isclose(lowest[0]["window_mean"], -5.0)


def test_mean_after_opening_skips_exactly_the_first_64_tokens():
    values = np.concatenate([np.full(64, 1000.0), np.full(10, 2.0)])
    assert analysis.mean_after_opening(values) == 2.0
    assert np.isnan(analysis.mean_after_opening(np.ones(64)))


def test_flip_turns_fewer_vs_more_auc_into_more_vs_fewer():
    flipped = analysis.flip_probability_interval({"estimate": 0.7, "ci_low": 0.6, "ci_high": 0.9})
    assert np.isclose(flipped["estimate"], 0.3) and np.isclose(flipped["ci_low"], 0.1) and np.isclose(flipped["ci_high"], 0.4)


def test_judge_agreement_machinery_gives_positive_rho_and_flipped_auc_above_half_when_too_much_goes_with_higher_score():
    rng = np.random.default_rng(0)
    count = 60
    too_much_difference = rng.choice([-0.5, 0.0, 0.5], size=count)
    score_difference = too_much_difference + rng.normal(0, 0.1, count)   # planted: more too_much -> higher score
    comparisons = pd.DataFrame({"arm_document_id": [f"arm{i}" for i in range(count)], "reference_document_id": [f"ref{i}" for i in range(count)],
                                "prompt_gid": [f"P{i % 20}" for i in range(count)], "task": "stage2", "too_much_difference": too_much_difference,
                                "delivery_signed_score": rng.choice([-1.0, 0.0, 1.0], size=count), "log_length_ratio": rng.normal(0, 0.2, count)})
    scores = pd.Series(np.concatenate([score_difference, np.zeros(count)]), index=list(comparisons["arm_document_id"]) + list(comparisons["reference_document_id"]))
    result = analysis.vector_analysis.agreement_with_judges(comparisons, scores, draws=50, seed=1)["stage2"]
    assert result["too_much_spearman"]["estimate"] > 0.8
    assert analysis.flip_probability_interval(result["too_much_auc_fewer_vs_more"])["estimate"] > 0.9


def test_per_token_agreement_flips_the_vector_so_moralizing_is_positive():
    rng = np.random.default_rng(1)
    contrast = rng.normal(size=400)
    vector_projection = -50.0 * contrast + 30.0      # more moralizing = more negative vector, here tracking a higher contrast
    result = analysis.per_token_agreement_for_document(vector_projection, contrast)
    assert result["pearson_all_tokens"] > 0.99 and result["pearson_of_32_token_rolling_means"] > 0.99


def test_window_overlap_counts_only_overlapping_windows_in_the_same_document():
    first = [{"document_id": "a", "start_token": 100}, {"document_id": "b", "start_token": 100}, {"document_id": "c", "start_token": 70}]
    second = [{"document_id": "a", "start_token": 120}, {"document_id": "b", "start_token": 200}]
    overlap = analysis.overlap_of_window_lists(first, second)
    assert overlap["documents_in_both_lists"] == 2 and overlap["overlapping_window_document_ids"] == ["a"]
