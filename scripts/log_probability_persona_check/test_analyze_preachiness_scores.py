"""Automated tests for analyze_preachiness_scores.py on synthetic run folders with planted effects."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import analyze_preachiness_scores as analysis_module  # noqa: E402
from smoke_test_analyze_preachiness_scores import build_synthetic_run_folder  # noqa: E402

TOKENS = 200
SHIFT = 0.3
SPAN = 32
EXPECTED_PLANTED_SCORE = SHIFT * SPAN / TOKENS  # mean over tokens of a 0.3 shift on a 32-token span


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    run_folder = tmp_path_factory.mktemp("synthetic_run")
    planted = build_synthetic_run_folder(run_folder, seed=1, tokens_per_document=TOKENS, planted_shift_per_token=SHIFT)
    analysis_folder = analysis_module.run_analysis(run_folder, timestamp="2026-01-01_00-00")
    tables = {
        "scores": pd.read_csv(analysis_folder / "per_document_scores_2026-01-01_00-00.csv"),
        "tests": pd.read_csv(analysis_folder / "per_document_significance_tests_2026-01-01_00-00.csv"),
        "agreement": pd.read_csv(analysis_folder / "rewording_agreement_2026-01-01_00-00.csv"),
        "spec": pd.read_csv(analysis_folder / "scores_by_spec_variable_2026-01-01_00-00.csv"),
        "positive": pd.read_csv(analysis_folder / "positive_control_check_2026-01-01_00-00.csv"),
        "spans": [json.loads(line) for line in
                  (analysis_folder / "extreme_spans_2026-01-01_00-00.jsonl").read_text().splitlines()],
    }
    return planted, tables, analysis_folder


def documents_of_kind(planted, kind):
    return [document_id for document_id, info in planted["documents"].items() if info["kind"] == kind]


def test_condition_structure_has_62_conditions(analyzed):
    planted, _, _ = analyzed
    assert len(planted["condition_ids"]) == 62


def test_scores_equal_planted_shift(analyzed):
    planted, tables, _ = analyzed
    scores = tables["scores"].set_index("document_id")
    for document_id in documents_of_kind(planted, "elevated"):
        assert scores.loc[document_id, "preachy_mean_score"] == pytest.approx(EXPECTED_PLANTED_SCORE, abs=0.005)
        assert scores.loc[document_id, "preachy_minus_control_score"] == pytest.approx(EXPECTED_PLANTED_SCORE, abs=0.005)
    for document_id in documents_of_kind(planted, "suppressed"):
        assert scores.loc[document_id, "preachy_mean_score"] == pytest.approx(-EXPECTED_PLANTED_SCORE, abs=0.005)
    for document_id in documents_of_kind(planted, "null"):
        assert abs(scores.loc[document_id, "preachy_minus_control_score"]) < 0.005
    assert (scores["number_of_tokens"] == TOKENS).all()
    assert np.allclose(scores["score__generic"], 0)


def test_extra_columns_present(analyzed):
    _, tables, _ = analyzed
    for column in ["control_mean_score", "proportionate_welfare_mean_score", "no_system_prompt_score",
                   "generic_mean_log_probability", "language", "source_document_id", "rewrite_direction", "spec_tone"]:
        assert column in tables["scores"].columns


def test_planted_span_is_top_or_bottom_extreme_span(analyzed):
    planted, tables, _ = analyzed
    spans = pd.DataFrame(tables["spans"])
    for document_id in documents_of_kind(planted, "elevated"):
        start = planted["documents"][document_id]["span_start"]
        top = spans[(spans["document_id"] == document_id) & (spans["direction"] == "highest") & (spans["rank"] == 1)].iloc[0]
        assert top["start_token_index"] == start
        assert top["end_token_index_exclusive"] == start + SPAN
        assert top["window_mean_preachy_difference"] == pytest.approx(SHIFT, abs=0.03)
        assert top["text"].startswith(f"tok{start} ")
    for document_id in documents_of_kind(planted, "suppressed"):
        start = planted["documents"][document_id]["span_start"]
        bottom = spans[(spans["document_id"] == document_id) & (spans["direction"] == "lowest") & (spans["rank"] == 1)].iloc[0]
        assert bottom["start_token_index"] == start


def test_extreme_spans_are_non_overlapping_and_three_each(analyzed):
    _, tables, _ = analyzed
    spans = pd.DataFrame(tables["spans"])
    for (document_id, direction), group in spans.groupby(["document_id", "direction"]):
        assert len(group) == 3
        starts = sorted(group["start_token_index"])
        assert all(later - earlier >= SPAN for earlier, later in zip(starts, starts[1:]))


def test_short_document_uses_one_whole_document_window():
    per_token = np.arange(10, dtype=float)
    records = analysis_module.extreme_spans_for_document("short", per_token, [f"t{i}" for i in range(10)], 32, 3)
    assert {record["direction"] for record in records} == {"highest", "lowest"}
    assert all(record["start_token_index"] == 0 and record["end_token_index_exclusive"] == 10 for record in records)
    assert len(records) == 2


def test_window_selection_respects_room():
    # 70 tokens, window 32: at most 2 non-overlapping windows fit
    starts = analysis_module.select_non_overlapping_windows(np.arange(39, dtype=float), 32, 3, True)
    assert len(starts) == 2 and abs(starts[0] - starts[1]) >= 32


def test_significance_labels(analyzed):
    planted, tables, _ = analyzed
    tests = tables["tests"].set_index("document_id")
    for document_id in documents_of_kind(planted, "elevated"):
        assert tests.loc[document_id, "main_test_direction"] == "elevated"
        assert tests.loc[document_id, "main_test_q_value"] < 0.05
        assert tests.loc[document_id, "second_test_direction"] == "elevated"
        assert not tests.loc[document_id, "preachy_significantly_below_generic"]
    for document_id in documents_of_kind(planted, "suppressed"):
        assert tests.loc[document_id, "main_test_direction"] == "suppressed"
        assert tests.loc[document_id, "preachy_significantly_below_generic"]
        assert tests.loc[document_id, "main_test_median_preachy_minus_control"] < 0
    for document_id in documents_of_kind(planted, "null"):
        assert tests.loc[document_id, "main_test_direction"] == "no significant difference at q < 0.05"


def test_mann_whitney_matches_scipy_directly(analyzed):
    _, tables, _ = analyzed
    scores = tables["scores"].iloc[0]
    preachy = [scores[f"score__preachy__rewording_{i:02d}"] for i in range(30)]
    control = [scores[c] for c in tables["scores"].columns if c.startswith("score__")
               and any(c.startswith(f"score__{p}__") for p in
                       ["animal_husbandry_expertise", "veterinary_clinic_manager", "zoo_tour_guide",
                        "wildlife_photographer", "pet_shop_assistant"])]
    assert len(control) == 25
    expected = stats.mannwhitneyu(preachy, control, alternative="two-sided")
    row = tables["tests"].iloc[0]
    assert row["main_test_mann_whitney_u"] == pytest.approx(expected.statistic)
    assert row["main_test_p_value"] == pytest.approx(expected.pvalue)


def test_benjamini_hochberg_matches_hand_computation():
    # sorted p: .005 .01 .03 .04 -> p * 4 / rank = .02 .02 .04 .04
    q_values = analysis_module.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert q_values == pytest.approx([0.02, 0.04, 0.04, 0.02])
    # monotonicity enforcement: sorted p .01 .02 .021 -> raw .03 .03 .021 -> step-up min from the top: .021 .021 .021
    assert analysis_module.benjamini_hochberg([0.01, 0.02, 0.021]) == pytest.approx([0.021, 0.021, 0.021])
    # capped at 1 and NaN preserved
    result = analysis_module.benjamini_hochberg([0.9, np.nan, 0.8])
    assert result[0] == pytest.approx(0.9) and np.isnan(result[1]) and result[2] == pytest.approx(0.9)


def test_benjamini_hochberg_matches_scipy_on_random_p_values():
    p_values = np.random.default_rng(3).uniform(size=50) ** 2
    assert analysis_module.benjamini_hochberg(p_values) == pytest.approx(stats.false_discovery_control(p_values, method="bh"))


def test_mismatched_condition_sets_raise(tmp_path):
    document_to_break = "matrix_000003"
    build_synthetic_run_folder(tmp_path, seed=2, condition_ids_dropped_for_document={document_to_break: ["preachy__rewording_05"]})
    with pytest.raises(ValueError, match="does not match conditions.jsonl"):
        analysis_module.run_analysis(tmp_path, timestamp="2026-01-01_00-00")


def test_missing_document_output_raises(tmp_path):
    build_synthetic_run_folder(tmp_path, seed=2)
    (tmp_path / "scoring_outputs" / "per_token_log_probabilities" / "matrix_000001.npz").unlink()
    with pytest.raises(ValueError, match="Missing scoring output"):
        analysis_module.run_analysis(tmp_path, timestamp="2026-01-01_00-00")


def test_positive_control_detects_planted_rewrite_shifts(analyzed):
    planted, tables, _ = analyzed
    positive = tables["positive"]
    document_rows = positive[positive["row_type"] == "document"]
    assert len(document_rows) == 4
    assert document_rows["moved_in_expected_direction"].all()
    for _, row in document_rows.iterrows():
        sign = 1 if row["rewrite_direction"] == "preachier" else -1
        assert row["difference_rewrite_minus_source"] == pytest.approx(sign * EXPECTED_PLANTED_SCORE, abs=0.008)
    summary = positive[positive["row_type"] == "summary"].set_index("rewrite_direction")
    for direction in ("preachier", "plainer"):
        assert summary.loc[direction, "number_of_rewrites"] == 2
        assert summary.loc[direction, "number_moving_in_expected_direction"] == 2
        assert 0 < summary.loc[direction, "wilcoxon_signed_rank_p_value_two_sided"] <= 1


def test_positive_control_flags_wrong_direction():
    scores = pd.DataFrame({
        "document_id": ["a", "a__preachier", "a__plainer"],
        "source_document_id": ["a", "a", "a"],
        "rewrite_direction": [None, "preachier", "plainer"],
        "preachy_minus_control_score": [0.0, -0.1, 0.1]})
    result = analysis_module.build_positive_control_check(scores)
    documents = result[result["row_type"] == "document"]
    assert not documents["moved_in_expected_direction"].any()


def test_positive_control_skips_cleanly_without_rewrites(tmp_path):
    build_synthetic_run_folder(tmp_path, seed=4, include_rewrites=False)
    analysis_folder = analysis_module.run_analysis(tmp_path, timestamp="2026-01-01_00-00")
    positive = pd.read_csv(analysis_folder / "positive_control_check_2026-01-01_00-00.csv")
    assert len(positive) == 1 and "skipped" in positive.iloc[0]["note"]


def test_scores_by_spec_variable_structure_and_values(analyzed):
    _, tables, _ = analyzed
    spec = tables["spec"]
    summary = spec[spec["section"] == "summary_by_level"]
    variables_in_order = list(dict.fromkeys(summary["variable"]))
    assert variables_in_order[0] == "tone" and variables_in_order[-1] == "language" and len(variables_in_order) == 13
    assert summary[summary["variable"] == "tone"]["count"].sum() == 12  # original documents only
    scores = tables["scores"]
    originals = scores[scores["rewrite_direction"].isna()]
    expected_mean = originals[originals["spec_tone"] == "skeptical or displeased"]["preachy_minus_control_score"].mean()
    row = summary[(summary["variable"] == "tone") & (summary["level"] == "skeptical or displeased")].iloc[0]
    assert row["mean_preachy_minus_control_score"] == pytest.approx(expected_mean)
    coefficients = spec[spec["section"] == "regression_coefficient"]
    assert (coefficients["variable"] == "intercept").sum() == 1


def test_ordinary_least_squares_recovers_known_coefficients():
    generator = np.random.default_rng(0)
    design = np.column_stack([np.ones(200), generator.integers(0, 2, 200), generator.integers(0, 2, 200)]).astype(float)
    outcome = design @ np.array([1.0, 2.0, -3.0]) + generator.normal(0, 0.1, 200)
    coefficients, standard_errors, _, _ = analysis_module.ordinary_least_squares(design, outcome)
    assert coefficients == pytest.approx([1.0, 2.0, -3.0], abs=0.1)
    # compare standard errors with the textbook formula
    residuals = outcome - design @ coefficients
    expected = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * (residuals @ residuals) / (200 - 3))
    assert standard_errors == pytest.approx(expected)


def test_rewording_agreement_structure_and_icc(analyzed):
    _, tables, _ = analyzed
    agreement = tables["agreement"]
    assert len(agreement) == 30 + 25 + 5
    assert set(agreement["persona_name"]) == {"preachy", "proportionate_welfare", "animal_husbandry_expertise",
                                              "veterinary_clinic_manager", "zoo_tour_guide", "wildlife_photographer",
                                              "pet_shop_assistant"}
    preachy = agreement[agreement["persona_name"] == "preachy"].iloc[0]
    # the planted effect varies across documents and is shared by all rewordings, so they agree strongly
    assert preachy["persona_intraclass_correlation_one_way_single_rewording"] > 0.9
    assert preachy["persona_mean_pairwise_spearman_across_documents"] > 0.8
    assert preachy["persona_within_to_across_sd_ratio"] < 0.3


def test_intraclass_correlation_extremes():
    identical_rewordings = np.column_stack([np.arange(10.0)] * 4)
    assert analysis_module.one_way_intraclass_correlation(identical_rewordings) == pytest.approx(1.0)
    noise = np.random.default_rng(0).normal(size=(500, 4))
    assert abs(analysis_module.one_way_intraclass_correlation(noise)) < 0.1


def test_outlier_rewording_is_flagged(tmp_path):
    run_folder = tmp_path
    build_synthetic_run_folder(run_folder, seed=5, documents_per_effect_kind=6)
    # replace preachy rewording 07 in every document with its mirror-image shift so its scores anticorrelate
    directory = run_folder / "scoring_outputs" / "per_token_log_probabilities"
    for path in directory.glob("*.npz"):
        with np.load(path) as archive:
            contents = {key: archive[key] for key in archive.files}
        condition_ids = [str(c) for c in contents["condition_ids"]]
        row = condition_ids.index("preachy__rewording_07")
        generic = contents["log_probabilities"][condition_ids.index("generic")]
        contents["log_probabilities"][row] = generic - (contents["log_probabilities"][row] - generic) \
            + np.random.default_rng(0).normal(0, 0.05, generic.shape)
        np.savez_compressed(path, **contents)
    analysis_folder = analysis_module.run_analysis(run_folder, timestamp="2026-01-01_00-00")
    agreement = pd.read_csv(analysis_folder / "rewording_agreement_2026-01-01_00-00.csv")
    # control personas carry no shared signal in the synthetic data, so their flags are noise; check the preachy persona
    flagged = agreement[agreement["rewording_is_outlier"] & (agreement["persona_name"] == "preachy")]
    assert list(flagged["condition_id"]) == ["preachy__rewording_07"]


def test_figures_and_notes_written(analyzed):
    _, _, analysis_folder = analyzed
    assert len(list((analysis_folder / "figures").glob("*.png"))) == 5
    notes = (analysis_folder / "analysis_notes_2026-01-01_00-00.md").read_text()
    assert "Benjamini-Hochberg" in notes


def test_bh_only_across_originals_and_rewrites_uncorrected(analyzed):
    _, tables, _ = analyzed
    tests = tables["tests"].merge(tables["scores"][["document_id", "rewrite_direction"]], on="document_id")
    rewrites = tests[tests["rewrite_direction"].notna()]
    originals = tests[tests["rewrite_direction"].isna()]
    assert rewrites["q_value_is_uncorrected_rewrite"].all() and not originals["q_value_is_uncorrected_rewrite"].any()
    assert rewrites["main_test_q_value"].to_numpy() == pytest.approx(rewrites["main_test_p_value"].to_numpy())
    assert originals["main_test_q_value"].to_numpy() == pytest.approx(
        analysis_module.benjamini_hochberg(originals["main_test_p_value"].to_numpy()))


def test_regression_excludes_language_includes_resolution(analyzed):
    _, tables, _ = analyzed
    spec = tables["spec"]
    regression_variables = set(spec[spec["section"].str.startswith("regression")]["variable"])
    assert "language" not in regression_variables and "resolution" in regression_variables
    assert "language" in set(spec[spec["section"] == "summary_by_level"]["variable"])
