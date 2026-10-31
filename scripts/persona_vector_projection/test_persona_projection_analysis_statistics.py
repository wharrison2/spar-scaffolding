"""Tests for the analysis code, aimed at bugs that would invalidate the experiment: baseline subtraction with the wrong
document's baselines, rewrites compared with the wrong source, standardisation across rather than within languages,
standard errors that ignore clustering, and the minimum-changed-token check. Synthetic data with known truth; no
model, no API."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_projection_analysis_statistics as statistics_module  # noqa: E402
from persona_vector_projection import persona_projection_analysis_tables_and_figures as tables_module  # noqa: E402
from persona_vector_projection import synthetic_scoring_run_for_analysis_tests as synthetic  # noqa: E402
from persona_vector_projection.changed_token_counting_by_diff import (  # noqa: E402
    changed_masks_from_token_sequences, count_changed_units, is_below_minimum_changed_units)
from persona_vector_projection.load_persona_projection_scoring_outputs import load_projection_data  # noqa: E402
from persona_vector_projection.persona_projection_analysis_tables_and_figures import ScoreTables  # noqa: E402

FACET = synthetic.REWRITTEN_FACET


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synthetic_scoring_run")
    built = synthetic.build_synthetic_scoring_run(folder, seed=3)
    data = load_projection_data(folder, "assistant_turn", built["corpus_path"])
    return {"folder": folder, "built": built, "data": data, "tables": ScoreTables(data)}


# ---- baseline subtraction uses the right document's baselines ----------------------------------

def test_document_score_subtracts_the_mean_of_its_own_baselines(synthetic_run):
    data, tables = synthetic_run["data"], synthetic_run["tables"]
    documents = data.documents
    for document_id in tables.corpus_ids[:10]:
        own = documents[(documents["kind"] == "baseline") & (documents["source_document_id"] == document_id)].index
        assert len(own) == synthetic.BASELINES_PER_DOCUMENT
        expected = data.mean_overall.loc[document_id, FACET] - data.mean_overall.loc[own, FACET].mean()
        assert tables.raw_scores.loc[document_id, FACET] == pytest.approx(expected, abs=1e-5)


def test_wrong_documents_baselines_would_give_a_different_score(synthetic_run):
    """Guards the guard: baselines differ between documents by design, so using another document's would show."""
    data, tables = synthetic_run["data"], synthetic_run["tables"]
    first, second = tables.corpus_ids[0], tables.corpus_ids[1]
    documents = data.documents
    other_baselines = documents[(documents["kind"] == "baseline") & (documents["source_document_id"] == second)].index
    wrong = data.mean_overall.loc[first, FACET] - data.mean_overall.loc[other_baselines, FACET].mean()
    assert abs(wrong - tables.raw_scores.loc[first, FACET]) > 0.5


def test_rewrite_is_scored_against_its_sources_baselines(synthetic_run):
    data, tables = synthetic_run["data"], synthetic_run["tables"]
    documents = data.documents
    rewrite_id = documents[documents["kind"] == "rewrite"].index[0]
    source_id = documents.loc[rewrite_id, "source_document_id"]
    assert tables.information.loc[rewrite_id, "baseline_owner_id"] == source_id
    own = documents[(documents["kind"] == "baseline") & (documents["source_document_id"] == source_id)].index
    expected = data.mean_overall.loc[rewrite_id, FACET] - data.mean_overall.loc[own, FACET].mean()
    assert tables.raw_scores.loc[rewrite_id, FACET] == pytest.approx(expected, abs=1e-5)


def test_baseline_subtraction_removes_the_documents_own_baseline_level(synthetic_run):
    """Raw score minus the document's true effect is only noise, not its baseline level (which spans sd 4)."""
    tables, truth = synthetic_run["tables"], synthetic_run["built"]["truth"]
    column = synthetic.VECTOR_NAMES.index(FACET)
    residuals = [tables.raw_scores.loc[document_id, FACET] - truth["true_effect"][document_id][column]
                 for document_id in tables.corpus_ids]
    assert np.max(np.abs(residuals)) < 0.5


# ---- standardisation within language -----------------------------------------------------------

def test_scores_are_standardised_within_language_not_across(synthetic_run):
    tables = synthetic_run["tables"]
    corpus = tables.z_scores.loc[tables.corpus_ids]
    languages = tables.languages.loc[tables.corpus_ids]
    for language in ("English", "Spanish"):
        values = corpus.loc[languages == language, FACET]
        assert values.mean() == pytest.approx(0, abs=1e-5)
        assert values.std(ddof=1) == pytest.approx(1, abs=1e-5)


def test_language_offset_disappears_after_baseline_and_standardisation(synthetic_run):
    """The synthetic Spanish documents are 5 units higher; baselines (also +5) and standardisation both leave no gap."""
    tables = synthetic_run["tables"]
    corpus_raw = tables.raw_scores.loc[tables.corpus_ids, FACET]
    languages = tables.languages.loc[tables.corpus_ids]
    assert abs(corpus_raw[languages == "Spanish"].mean() - corpus_raw[languages == "English"].mean()) < 1.5


def test_standardisation_uses_each_documents_own_language_statistics():
    raw = pd.DataFrame({"v": [0.0, 2.0, 4.0, 100.0, 102.0, 104.0, 0.0, 1.0]},
                       index=["a", "b", "c", "d", "e", "f", "g", "h"])
    languages = pd.Series(["x", "x", "x", "y", "y", "y", "x", "y"], index=raw.index)
    corpus_mask = pd.Series(True, index=raw.index)
    stats_table = statistics_module.language_standardisation_statistics(raw, languages, corpus_mask, minimum_documents=3)
    z = statistics_module.standardise_within_language(raw, languages, stats_table)
    assert z.loc["a", "v"] == pytest.approx((0 - np.mean([0, 2, 4, 0])) / np.std([0, 2, 4, 0], ddof=1))
    assert z.loc["g", "v"] != pytest.approx(z.loc["h", "v"])  # same-ish raw value, different languages
    assert abs(z.loc["d", "v"]) < 3  # not blown up by pooling with language x


def test_small_languages_fall_back_to_all_language_statistics_and_are_flagged():
    raw = pd.DataFrame({"v": np.arange(12.0)}, index=[f"d{i}" for i in range(12)])
    languages = pd.Series(["x"] * 10 + ["tiny"] * 2, index=raw.index)
    table = statistics_module.language_standardisation_statistics(raw, languages, pd.Series(True, index=raw.index), 5)
    assert bool(table.loc["tiny", "used_all_language_statistics"]) is True
    assert bool(table.loc["x", "used_all_language_statistics"]) is False
    assert table.loc["tiny", "v__mean"] == pytest.approx(raw["v"].mean())


# ---- clustered standard errors ---------------------------------------------------------------------

def test_cluster_robust_mean_matches_statsmodels_cluster_ols():
    import statsmodels.api as sm
    generator = np.random.default_rng(0)
    clusters = np.repeat(np.arange(12), 3)
    values = generator.normal(size=len(clusters)) + np.repeat(generator.normal(size=12), 3)
    ours = statistics_module.cluster_robust_mean(values, clusters)
    fit = sm.OLS(values, np.ones((len(values), 1))).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    assert ours["standard_error"] == pytest.approx(float(fit.bse[0]), rel=1e-9)
    assert ours["mean"] == pytest.approx(values.mean())


def test_clustering_widens_the_standard_error_when_rewrites_of_one_source_agree():
    sources = np.repeat(np.arange(10), 3)
    source_effect = np.repeat(np.linspace(-2, 2, 10), 3)  # the three rewrites of a source move together
    values = source_effect + np.tile([0.01, -0.01, 0.0], 10)
    clustered = statistics_module.cluster_robust_mean(values, sources)["standard_error"]
    naive = statistics_module.mean_with_t_interval(values)["standard_error"]
    independent = statistics_module.cluster_robust_mean(values, np.arange(len(values)))["standard_error"]
    assert independent == pytest.approx(naive, rel=0.05)  # every row its own cluster is the naive case
    assert clustered > 1.3 * naive  # rewrites of one source are not independent evidence


def test_duplicated_rows_from_one_source_do_not_shrink_the_clustered_standard_error():
    base = np.array([1.0, -1.0, 2.0, 0.5, -0.5, 1.5])
    once = statistics_module.cluster_robust_mean(base, np.arange(6))["standard_error"]
    triplicated = statistics_module.cluster_robust_mean(np.repeat(base, 3), np.repeat(np.arange(6), 3))["standard_error"]
    naive_triplicated = statistics_module.mean_with_t_interval(np.repeat(base, 3))["standard_error"]
    assert triplicated == pytest.approx(once * np.sqrt(1.0), rel=0.05)
    assert triplicated > naive_triplicated  # ignoring clustering would understate it


def test_cluster_robust_contrast_matches_a_manual_regression():
    generator = np.random.default_rng(1)
    clusters = np.tile(np.arange(15), 2)
    groups = np.repeat(["toward_facet", "facet_neutral"], 15)
    values = generator.normal(size=30) + (groups == "toward_facet") * 2.0
    result = statistics_module.cluster_robust_contrast(values, groups, clusters, "toward_facet", "facet_neutral")
    assert result["difference"] == pytest.approx(values[:15].mean() - values[15:].mean())
    assert result["standard_error"] > 0 and result["ci_low"] < result["difference"] < result["ci_high"]


# ---- rewrites compared with their own source ------------------------------------------------------------

def test_whole_document_shift_is_rewrite_minus_its_own_source(synthetic_run):
    data, tables = synthetic_run["data"], synthetic_run["tables"]
    shifts = statistics_module.whole_document_rewrite_shifts(tables.raw_scores, data.documents)
    for rewrite_id in shifts.index[:12]:
        source_id = data.documents.loc[rewrite_id, "source_document_id"]
        expected = data.mean_overall.loc[rewrite_id, FACET] - data.mean_overall.loc[source_id, FACET]
        assert shifts.loc[rewrite_id, FACET] == pytest.approx(expected, abs=1e-5)


def test_shift_against_a_different_source_would_be_wrong(synthetic_run):
    data, tables = synthetic_run["data"], synthetic_run["tables"]
    documents = data.documents
    rewrite_id = documents[documents["kind"] == "rewrite"].index[0]
    other_source = next(i for i in tables.corpus_ids if i != documents.loc[rewrite_id, "source_document_id"])
    wrong = tables.raw_scores.loc[rewrite_id, FACET] - tables.raw_scores.loc[other_source, FACET]
    right = statistics_module.whole_document_rewrite_shifts(tables.raw_scores, documents).loc[rewrite_id, FACET]
    assert abs(wrong - right) > 0.3


def test_changed_span_shift_uses_the_rewrites_own_source_tokens(synthetic_run):
    data, tables, built = synthetic_run["data"], synthetic_run["tables"], synthetic_run["built"]
    documents = data.documents
    table = tables_module.build_positive_control_shift_table(tables, built["token_ids"], minimum_changed_tokens=5)
    toward = table[table["rewrite_direction"] == "toward_facet"]
    away = table[table["rewrite_direction"] == "away_from_facet"]
    neutral = table[table["rewrite_direction"] == "facet_neutral"]
    sd = float(tables.language_statistics.loc["English", f"{FACET}__sd"])
    english_toward = toward[toward["language"] == "English"]
    assert english_toward["changed_span_shift"].mean() == pytest.approx(synthetic.TOWARD_EFFECT / sd, rel=0.15)
    assert (toward["changed_span_shift"] > 0).all() and (away["changed_span_shift"] < 0).all()
    assert abs(neutral["changed_span_shift"].mean()) < 0.3 * abs(toward["changed_span_shift"].mean())
    assert (table["changed_tokens"] == synthetic.CHANGED_BLOCK[1] - synthetic.CHANGED_BLOCK[0]).all()
    assert set(table["source_document_id"]) <= set(documents.index)


def test_whole_document_toward_and_away_move_the_expected_way_and_neutral_does_not(synthetic_run):
    tables, built = synthetic_run["tables"], synthetic_run["built"]
    table = tables_module.build_positive_control_shift_table(tables, built["token_ids"], minimum_changed_tokens=5)
    summaries = tables_module.build_positive_control_summaries(table)
    summary = summaries["whole_document"]["summary"].set_index("rewrite_direction")
    assert summary.loc["toward_facet", "mean_shift"] > 0 > summary.loc["away_from_facet", "mean_shift"]
    assert summary.loc["toward_facet", "ci_low"] > 0 > summary.loc["away_from_facet", "ci_high"]
    assert abs(summary.loc["facet_neutral", "mean_shift"]) < 0.1
    assert summary.loc["toward_facet", "n_source_documents"] == 12
    criteria = {"minimum_fraction_moving_expected_way": 0.8,
                "require_contrast_against_neutral_interval_to_exclude_zero": True}
    passed = tables_module.evaluate_positive_control_pass_criteria(summaries, criteria)
    assert passed["passed"].all() and set(passed["level"]) == {"whole_document", "changed_span"}


def test_other_facets_do_not_respond_to_rewrites_of_one_facet(synthetic_run):
    tables, built = synthetic_run["tables"], synthetic_run["built"]
    table = tables_module.build_positive_control_shift_table(tables, built["token_ids"], minimum_changed_tokens=5)
    specificity = tables_module.build_positive_control_specificity(table)
    toward = specificity.loc[(FACET, "toward_facet")]
    own = toward[f"mean_shift_on_{FACET}"]
    others = [value for name, value in toward.items() if name != f"mean_shift_on_{FACET}"]
    assert own > 5 * max(abs(value) for value in others)


# ---- minimum changed tokens ------------------------------------------------------------------------------

def test_minimum_changed_tokens_flags_small_rewrites_in_the_analysis(synthetic_run):
    tables, built = synthetic_run["tables"], synthetic_run["built"]
    strict = tables_module.build_positive_control_shift_table(tables, built["token_ids"], minimum_changed_tokens=25)
    assert strict["below_minimum_changed_tokens"].all()  # every synthetic rewrite changes 20 tokens
    assert tables_module.build_positive_control_summaries(strict).get("changed_span") is None
    lenient = tables_module.build_positive_control_shift_table(tables, built["token_ids"], minimum_changed_tokens=20)
    assert not lenient["below_minimum_changed_tokens"].any()


def test_count_changed_units_and_minimum_check():
    source = " ".join(f"w{i}" for i in range(100))
    small = source.replace("w10", "x10").replace("w20", "x20")
    large = " ".join(f"y{i}" if i % 2 == 0 else f"w{i}" for i in range(100))
    small_counts = count_changed_units(source, small, "English")
    large_counts = count_changed_units(source, large, "English")
    assert small_counts["changed_rewrite_units"] == 2 and large_counts["changed_rewrite_units"] == 50
    assert is_below_minimum_changed_units(small_counts, 10) and not is_below_minimum_changed_units(large_counts, 10)
    assert count_changed_units("同じ文章です", "同じ文章でした", "Japanese")["changed_rewrite_units"] >= 1


def test_masks_from_identical_sequences_have_no_changes():
    rewrite_changed, source_replaced = changed_masks_from_token_sequences([1, 2, 3], [1, 2, 3])
    assert not rewrite_changed.any() and not source_replaced.any()


# ---- group tables ---------------------------------------------------------------------------------------

def test_values_with_too_few_documents_are_marked_and_not_ranked():
    values = list(np.random.default_rng(0).normal(size=60)) + [5.0, 5.2]
    groups = ["big_a"] * 30 + ["big_b"] * 30 + ["tiny"] * 2
    table = statistics_module.group_mean_table(values, groups, minimum_documents=10)
    tiny = table[table["group"] == "tiny"].iloc[0]
    assert bool(tiny["too_few_documents"]) and pd.isna(tiny["rank_by_mean"]) and not bool(tiny["separable_from_rest"])
    assert table[~table["too_few_documents"]]["rank_by_mean"].notna().all()
    assert table.iloc[-1]["group"] == "tiny"  # unranked values listed last


def test_baseline_split_half_agreement_is_high_when_baselines_agree(synthetic_run):
    table = tables_module.build_baseline_split_half_agreement(synthetic_run["data"], [FACET])
    assert table.iloc[0]["spearman"] > 0.9


def test_extreme_window_selection_skips_the_opening_and_does_not_overlap():
    values = np.zeros(300)
    values[10:42] = 9.0     # a spike inside the opening
    values[150:182] = 5.0
    means = statistics_module.rolling_window_means(values, 32)
    chosen = statistics_module.select_non_overlapping_windows(means, 32, 2, True, minimum_start=64)
    assert chosen[0][0] == 150 and all(start >= 64 for start, _ in chosen)
    assert abs(chosen[0][0] - chosen[1][0]) >= 32
