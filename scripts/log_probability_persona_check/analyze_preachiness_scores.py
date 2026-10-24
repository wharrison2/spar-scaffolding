"""Analysis of the log-probability preachiness check (local, no GPU, no API calls).

Reads <run folder>/scoring_inputs/ and <run folder>/scoring_outputs/ (formats in
run_folder_layout_and_script_interfaces.md) and writes <run folder>/analysis_<timestamp>/.

Usage, from the repository root:
    python scripts/log_probability_persona_check/analyze_preachiness_scores.py --run-folder <run folder>
"""

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

GENERIC_CONDITION_ID = "generic"
NO_SYSTEM_PROMPT_CONDITION_ID = "no_system_prompt"
EXPECTED_CONDITION_GROUPS = {"no_system_prompt", "generic", "preachy", "control", "proportionate_welfare"}
SIGNIFICANCE_LEVEL = 0.05
SPEC_VARIABLE_NAMES_TONE_FIRST = ["tone", "document_type", "culture", "domain", "sentient_category", "decision_scale",
                                  "naming", "tech_savvy", "tradeoff", "centrality", "resolution", "reasoning_featured"]
LANGUAGE_COLUMN = "language"
MAXIMUM_LEVELS_FOR_REGRESSION_VARIABLE = 25
OUTLIER_MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER = 3.0
OUTLIER_MINIMUM_SPREAD = 0.02

# Okabe-Ito colourblind-safe palette
COLOR_ELEVATED = "#D55E00"
COLOR_SUPPRESSED = "#0072B2"
COLOR_NO_DIFFERENCE = "#999999"
COLOR_PREACHIER = "#E69F00"
COLOR_PLAINER = "#56B4E9"
COLOR_NEUTRAL = "#444444"


# ----------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------

def load_run(run_folder):
    """Returns (documents, conditions, scores_by_document) where scores_by_document maps document_id to a dict with
    log_probabilities (conditions x tokens), token_strings, condition_ids. Raises ValueError on any inconsistency."""
    run_folder = Path(run_folder)
    documents = read_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")
    conditions = read_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    condition_ids_expected = [condition["condition_id"] for condition in conditions]
    if len(set(condition_ids_expected)) != len(condition_ids_expected):
        raise ValueError("conditions.jsonl contains duplicate condition ids")
    groups_present = {condition["condition_group"] for condition in conditions}
    if groups_present != EXPECTED_CONDITION_GROUPS:
        raise ValueError(f"conditions.jsonl condition groups {sorted(groups_present)} differ from the expected "
                         f"{sorted(EXPECTED_CONDITION_GROUPS)}")
    for required_id in (GENERIC_CONDITION_ID, NO_SYSTEM_PROMPT_CONDITION_ID):
        if required_id not in condition_ids_expected:
            raise ValueError(f"conditions.jsonl has no '{required_id}' condition")
    scores_by_document = {}
    for document in documents:
        npz_path = run_folder / "scoring_outputs" / "per_token_log_probabilities" / f"{document['document_id']}.npz"
        if not npz_path.exists():
            raise ValueError(f"Missing scoring output for document {document['document_id']}: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as archive:
            condition_ids = [str(value) for value in archive["condition_ids"]]
            log_probabilities = np.asarray(archive["log_probabilities"], dtype=np.float64)
            token_strings = [str(value) for value in archive["document_token_strings"]]
        if set(condition_ids) != set(condition_ids_expected) or len(condition_ids) != len(condition_ids_expected):
            missing = sorted(set(condition_ids_expected) - set(condition_ids))
            extra = sorted(set(condition_ids) - set(condition_ids_expected))
            raise ValueError(f"Condition set for document {document['document_id']} does not match conditions.jsonl "
                             f"(missing {missing[:5]}, unexpected {extra[:5]})")
        if log_probabilities.shape != (len(condition_ids), len(token_strings)):
            raise ValueError(f"Document {document['document_id']}: log_probabilities shape {log_probabilities.shape} "
                             f"does not match {len(condition_ids)} conditions x {len(token_strings)} tokens")
        # Reorder rows into the conditions.jsonl order so every document uses the same row order.
        row_for_condition = {condition_id: row for row, condition_id in enumerate(condition_ids)}
        order = [row_for_condition[condition_id] for condition_id in condition_ids_expected]
        scores_by_document[document["document_id"]] = {
            "log_probabilities": log_probabilities[order],
            "token_strings": token_strings,
            "condition_ids": condition_ids_expected,
        }
    return documents, conditions, scores_by_document


# ----------------------------------------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values (q-values), same order as input. NaN p-values stay NaN."""
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.full(p_values.shape, np.nan)
    valid = ~np.isnan(p_values)
    valid_p = p_values[valid]
    count = len(valid_p)
    if count == 0:
        return q_values
    order = np.argsort(valid_p)
    ranked = valid_p[order] * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(count)
    adjusted[order] = np.minimum(ranked, 1.0)
    q_values[valid] = adjusted
    return q_values


def direction_label(q_value, median_difference):
    if np.isnan(q_value) or q_value >= SIGNIFICANCE_LEVEL or median_difference == 0:
        return "no significant difference at q < 0.05"
    return "elevated" if median_difference > 0 else "suppressed"


def mann_whitney_two_sided(first_values, second_values):
    result = stats.mannwhitneyu(first_values, second_values, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def wilcoxon_against_zero(values):
    values = np.asarray(values, dtype=np.float64)
    if np.all(values == 0):
        return float("nan"), 1.0
    result = stats.wilcoxon(values, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def safe_spearman(first_values, second_values):
    if len(first_values) < 3 or np.ptp(first_values) == 0 or np.ptp(second_values) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(first_values, second_values).statistic)


# ----------------------------------------------------------------------------------------------------------
# Per-document scores
# ----------------------------------------------------------------------------------------------------------

def condition_indices_by_group(conditions):
    indices = {}
    for index, condition in enumerate(conditions):
        indices.setdefault(condition["condition_group"], []).append(index)
    return indices


def compute_condition_scores(log_probabilities, generic_row):
    """Mean over tokens of log_probabilities[c] - log_probabilities[generic], for every condition c."""
    return (log_probabilities - log_probabilities[generic_row]).mean(axis=1)


def build_score_tables(documents, conditions, scores_by_document):
    """Returns (per_document_scores DataFrame, condition_score_matrix DataFrame of documents x conditions)."""
    condition_ids = [condition["condition_id"] for condition in conditions]
    generic_row = condition_ids.index(GENERIC_CONDITION_ID)
    no_prompt_row = condition_ids.index(NO_SYSTEM_PROMPT_CONDITION_ID)
    indices = condition_indices_by_group(conditions)
    score_rows = []
    wide_rows = []
    for document in documents:
        log_probabilities = scores_by_document[document["document_id"]]["log_probabilities"]
        scores = compute_condition_scores(log_probabilities, generic_row)
        wide_rows.append(scores)
        row = {
            "document_id": document["document_id"],
            "source_document_id": document.get("source_document_id", document["document_id"]),
            "rewrite_direction": document.get("rewrite_direction"),
            LANGUAGE_COLUMN: document.get("language"),
        }
        for variable_name, value in (document.get("variables") or {}).items():
            row[f"spec_{variable_name}"] = value
        row["number_of_tokens"] = log_probabilities.shape[1]
        row["generic_mean_log_probability"] = float(log_probabilities[generic_row].mean())
        row["preachy_mean_score"] = float(scores[indices["preachy"]].mean())
        row["control_mean_score"] = float(scores[indices["control"]].mean())
        row["proportionate_welfare_mean_score"] = float(scores[indices["proportionate_welfare"]].mean())
        row["preachy_minus_control_score"] = row["preachy_mean_score"] - row["control_mean_score"]
        row["no_system_prompt_score"] = float(scores[no_prompt_row])
        for condition_id, score in zip(condition_ids, scores):
            row[f"score__{condition_id}"] = float(score)
        score_rows.append(row)
    return pd.DataFrame(score_rows), pd.DataFrame(wide_rows, columns=condition_ids, index=[d["document_id"] for d in documents])


# ----------------------------------------------------------------------------------------------------------
# Significance tests
# ----------------------------------------------------------------------------------------------------------

def build_significance_table(documents, conditions, condition_score_matrix):
    indices = condition_indices_by_group(conditions)
    rows = []
    for document in documents:
        scores = condition_score_matrix.loc[document["document_id"]].to_numpy()
        preachy = scores[indices["preachy"]]
        control = scores[indices["control"]]
        welfare = scores[indices["proportionate_welfare"]]
        main_u, main_p = mann_whitney_two_sided(preachy, control)
        second_u, second_p = mann_whitney_two_sided(preachy, welfare)
        wilcoxon_statistic, wilcoxon_p = wilcoxon_against_zero(preachy)
        rows.append({
            "document_id": document["document_id"],
            "number_of_preachy_rewordings": len(preachy),
            "number_of_control_rewordings": len(control),
            "number_of_proportionate_welfare_rewordings": len(welfare),
            "main_test_mann_whitney_u": main_u,
            "main_test_p_value": main_p,
            "main_test_median_preachy_minus_control": float(np.median(preachy) - np.median(control)),
            "second_test_mann_whitney_u": second_u,
            "second_test_p_value": second_p,
            "second_test_median_preachy_minus_proportionate_welfare": float(np.median(preachy) - np.median(welfare)),
            "preachy_median_score": float(np.median(preachy)),
            "preachy_vs_generic_wilcoxon_statistic": wilcoxon_statistic,
            "preachy_vs_generic_wilcoxon_p_value": wilcoxon_p,
        })
    table = pd.DataFrame(rows)
    is_original = np.array([document.get("rewrite_direction") is None for document in documents])
    table["q_value_is_uncorrected_rewrite"] = ~is_original
    for p_column, q_column in (("main_test_p_value", "main_test_q_value"),
                               ("second_test_p_value", "second_test_q_value"),
                               ("preachy_vs_generic_wilcoxon_p_value", "preachy_vs_generic_wilcoxon_q_value")):
        q_values = table[p_column].to_numpy(dtype=float).copy()  # rewrites keep their uncorrected p-value
        q_values[is_original] = benjamini_hochberg(q_values[is_original])
        table[q_column] = q_values
    table["main_test_direction"] = [direction_label(q, d) for q, d in
                                    zip(table["main_test_q_value"], table["main_test_median_preachy_minus_control"])]
    table["second_test_direction"] = [direction_label(q, d) for q, d in zip(
        table["second_test_q_value"], table["second_test_median_preachy_minus_proportionate_welfare"])]
    table["preachy_significantly_below_generic"] = (
        (table["preachy_vs_generic_wilcoxon_q_value"] < SIGNIFICANCE_LEVEL) & (table["preachy_median_score"] < 0))
    table["preachy_significantly_above_generic"] = (
        (table["preachy_vs_generic_wilcoxon_q_value"] < SIGNIFICANCE_LEVEL) & (table["preachy_median_score"] > 0))
    return table[["document_id", "q_value_is_uncorrected_rewrite", "number_of_preachy_rewordings", "number_of_control_rewordings",
                  "number_of_proportionate_welfare_rewordings",
                  "main_test_mann_whitney_u", "main_test_p_value", "main_test_q_value",
                  "main_test_median_preachy_minus_control", "main_test_direction",
                  "second_test_mann_whitney_u", "second_test_p_value", "second_test_q_value",
                  "second_test_median_preachy_minus_proportionate_welfare", "second_test_direction",
                  "preachy_median_score", "preachy_vs_generic_wilcoxon_statistic",
                  "preachy_vs_generic_wilcoxon_p_value", "preachy_vs_generic_wilcoxon_q_value",
                  "preachy_significantly_below_generic", "preachy_significantly_above_generic"]]


# ----------------------------------------------------------------------------------------------------------
# Extreme spans
# ----------------------------------------------------------------------------------------------------------

def select_non_overlapping_windows(window_means, window_size, count, highest):
    """Greedy selection of up to `count` window start indices with no two windows overlapping, ranked by mean
    (descending when highest else ascending). Ties are broken by earlier start."""
    order = np.lexsort((np.arange(len(window_means)), -window_means if highest else window_means))
    chosen = []
    for start in order:
        if all(abs(int(start) - other) >= window_size for other in chosen):
            chosen.append(int(start))
            if len(chosen) == count:
                break
    return chosen


def extreme_spans_for_document(document_id, per_token_difference, token_strings, window_size, windows_per_direction):
    token_count = len(per_token_difference)
    effective_window = min(window_size, token_count)
    cumulative = np.concatenate([[0.0], np.cumsum(per_token_difference)])
    window_means = (cumulative[effective_window:] - cumulative[:-effective_window]) / effective_window
    records = []
    for direction, highest in (("highest", True), ("lowest", False)):
        starts = select_non_overlapping_windows(window_means, effective_window, windows_per_direction, highest)
        for rank, start in enumerate(starts, start=1):
            records.append({
                "document_id": document_id,
                "direction": direction,
                "rank": rank,
                "start_token_index": start,
                "end_token_index_exclusive": start + effective_window,
                "window_size_tokens": effective_window,
                "window_mean_preachy_difference": float(window_means[start]),
                "text": "".join(token_strings[start:start + effective_window]),
            })
    return records


def build_extreme_spans(documents, conditions, scores_by_document, window_size, windows_per_direction):
    condition_ids = [condition["condition_id"] for condition in conditions]
    generic_row = condition_ids.index(GENERIC_CONDITION_ID)
    preachy_rows = condition_indices_by_group(conditions)["preachy"]
    records = []
    for document in documents:
        entry = scores_by_document[document["document_id"]]
        log_probabilities = entry["log_probabilities"]
        if log_probabilities.shape[1] == 0:
            continue
        per_token_difference = (log_probabilities[preachy_rows] - log_probabilities[generic_row]).mean(axis=0)
        records.extend(extreme_spans_for_document(document["document_id"], per_token_difference,
                                                  entry["token_strings"], window_size, windows_per_direction))
    return records


# ----------------------------------------------------------------------------------------------------------
# Rewording agreement
# ----------------------------------------------------------------------------------------------------------

def one_way_intraclass_correlation(score_matrix):
    """ICC(1,1): documents as targets (rows), rewordings as interchangeable raters (columns)."""
    document_count, rewording_count = score_matrix.shape
    if document_count < 2 or rewording_count < 2:
        return float("nan")
    grand_mean = score_matrix.mean()
    row_means = score_matrix.mean(axis=1)
    between = rewording_count * ((row_means - grand_mean) ** 2).sum() / (document_count - 1)
    within = ((score_matrix - row_means[:, None]) ** 2).sum() / (document_count * (rewording_count - 1))
    denominator = between + (rewording_count - 1) * within
    return float((between - within) / denominator) if denominator > 0 else float("nan")


def build_rewording_agreement(conditions, condition_score_matrix):
    rows = []
    persona_names_in_order = []
    for condition in conditions:
        if condition["condition_group"] in ("preachy", "control", "proportionate_welfare"):
            key = (condition["condition_group"], condition["persona_name"])
            if key not in persona_names_in_order:
                persona_names_in_order.append(key)
    for group, persona_name in persona_names_in_order:
        persona_conditions = [c for c in conditions if c["condition_group"] == group and c["persona_name"] == persona_name]
        condition_ids = [c["condition_id"] for c in persona_conditions]
        score_matrix = condition_score_matrix[condition_ids].to_numpy()  # documents x rewordings
        document_count, rewording_count = score_matrix.shape
        pairwise = [safe_spearman(score_matrix[:, i], score_matrix[:, j])
                    for i, j in itertools.combinations(range(rewording_count), 2)]
        mean_pairwise = float(np.nanmean(pairwise)) if pairwise and not np.all(np.isnan(pairwise)) else float("nan")
        if rewording_count > 1 and document_count > 1:
            within_document_sd = float(np.mean(score_matrix.std(axis=1, ddof=1)))
            across_document_sd = float(score_matrix.mean(axis=1).std(ddof=1))
            ratio = within_document_sd / across_document_sd if across_document_sd > 0 else float("nan")
        else:
            within_document_sd = across_document_sd = ratio = float("nan")
        leave_one_out_correlations = []
        for index in range(rewording_count):
            others = np.delete(score_matrix, index, axis=1)
            leave_one_out_correlations.append(
                safe_spearman(score_matrix[:, index], others.mean(axis=1)) if rewording_count > 1 else float("nan"))
        correlations = np.array(leave_one_out_correlations)
        if np.all(np.isnan(correlations)):
            outlier_flags = [False] * rewording_count
            threshold = float("nan")
        else:
            median = np.nanmedian(correlations)
            spread = max(1.4826 * np.nanmedian(np.abs(correlations - median)), OUTLIER_MINIMUM_SPREAD)
            threshold = float(median - OUTLIER_MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER * spread)
            outlier_flags = [bool(value < threshold) if not np.isnan(value) else False for value in correlations]
        icc = one_way_intraclass_correlation(score_matrix)
        for index, condition in enumerate(persona_conditions):
            rows.append({
                "persona_name": persona_name,
                "condition_group": group,
                "condition_id": condition["condition_id"],
                "rewording_index": condition.get("rewording_index"),
                "number_of_documents": document_count,
                "number_of_rewordings_in_persona": rewording_count,
                "persona_mean_pairwise_spearman_across_documents": mean_pairwise,
                "persona_mean_within_document_sd_across_rewordings": within_document_sd,
                "persona_sd_across_documents_of_rewording_mean_score": across_document_sd,
                "persona_within_to_across_sd_ratio": ratio,
                "persona_intraclass_correlation_one_way_single_rewording": icc,
                "rewording_mean_score_across_documents": float(score_matrix[:, index].mean()),
                "rewording_spearman_with_leave_one_out_mean_of_other_rewordings": float(correlations[index]),
                "persona_outlier_threshold_on_leave_one_out_spearman": threshold,
                "rewording_is_outlier": outlier_flags[index],
            })
    return pd.DataFrame(rows)


def pairwise_spearman_matrix(conditions, condition_score_matrix, condition_group):
    condition_ids = [c["condition_id"] for c in conditions if c["condition_group"] == condition_group]
    values = condition_score_matrix[condition_ids].to_numpy()
    size = len(condition_ids)
    matrix = np.full((size, size), np.nan)
    for i in range(size):
        for j in range(size):
            matrix[i, j] = 1.0 if i == j else safe_spearman(values[:, i], values[:, j])
    return condition_ids, matrix


# ----------------------------------------------------------------------------------------------------------
# Scores by spec variable
# ----------------------------------------------------------------------------------------------------------

def ordered_variable_columns(per_document_scores):
    present = [column[len("spec_"):] for column in per_document_scores.columns if column.startswith("spec_")]
    ordered = [name for name in SPEC_VARIABLE_NAMES_TONE_FIRST if name in present]
    ordered += sorted(name for name in present if name not in ordered)
    return [f"spec_{name}" for name in ordered] + [LANGUAGE_COLUMN]


def ordinary_least_squares(design_matrix, outcome):
    """Returns (coefficients, standard_errors, t_statistics, p_values). Uses the pseudo-inverse so that collinear
    columns (for example language and culture) give the minimum-norm solution; standard errors of collinear
    coefficients are then not individually meaningful. Degrees of freedom = n - rank."""
    sample_size, _ = design_matrix.shape
    gram_inverse = np.linalg.pinv(design_matrix.T @ design_matrix)
    coefficients = gram_inverse @ design_matrix.T @ outcome
    residuals = outcome - design_matrix @ coefficients
    degrees_of_freedom = sample_size - np.linalg.matrix_rank(design_matrix)
    if degrees_of_freedom <= 0:
        nan = np.full(coefficients.shape, np.nan)
        return coefficients, nan, nan, nan
    residual_variance = float(residuals @ residuals) / degrees_of_freedom
    standard_errors = np.sqrt(np.maximum(np.diag(gram_inverse) * residual_variance, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = np.where(standard_errors > 0, coefficients / standard_errors, np.nan)
    p_values = 2 * stats.t.sf(np.abs(t_statistics), degrees_of_freedom)
    return coefficients, standard_errors, t_statistics, p_values


def build_scores_by_spec_variable(per_document_scores):
    originals = per_document_scores[per_document_scores["rewrite_direction"].isna()].copy()
    columns = ordered_variable_columns(originals)
    columns = [column for column in columns if column in originals.columns]
    rows = []
    for column in columns:
        variable_name = column[len("spec_"):] if column.startswith("spec_") else column
        values = originals["preachy_minus_control_score"].groupby(originals[column].astype(str))
        summary = values.agg(["mean", "count", "std"])
        for level, entry in summary.iterrows():
            standard_error = entry["std"] / np.sqrt(entry["count"]) if entry["count"] > 1 else float("nan")
            rows.append({"section": "summary_by_level", "variable": variable_name, "level": level,
                         "count": int(entry["count"]), "mean_preachy_minus_control_score": entry["mean"],
                         "standard_error_of_mean": standard_error})
    # Regression
    design_columns = [np.ones(len(originals))]
    design_names = [("intercept", "")]
    for column in columns:
        variable_name = column[len("spec_"):] if column.startswith("spec_") else column
        if column == LANGUAGE_COLUMN:
            continue  # collinear with culture; summary rows only
        levels = sorted(originals[column].astype(str).unique())
        if len(levels) > MAXIMUM_LEVELS_FOR_REGRESSION_VARIABLE:
            rows.append({"section": "regression_coefficient", "variable": variable_name,
                         "level": f"NOT_INCLUDED_{len(levels)}_levels_exceeds_{MAXIMUM_LEVELS_FOR_REGRESSION_VARIABLE}"})
            continue
        for level in levels[1:]:  # first level (alphabetical) is the reference
            design_columns.append((originals[column].astype(str) == level).to_numpy(dtype=np.float64))
            design_names.append((variable_name, level))
        if len(levels) > 0:
            rows.append({"section": "regression_reference_level", "variable": variable_name, "level": levels[0]})
    if len(originals) > 0:
        design_matrix = np.column_stack(design_columns)
        coefficients, standard_errors, t_statistics, p_values = ordinary_least_squares(
            design_matrix, originals["preachy_minus_control_score"].to_numpy())
        for (variable_name, level), coefficient, standard_error, t_statistic, p_value in zip(
                design_names, coefficients, standard_errors, t_statistics, p_values):
            rows.append({"section": "regression_coefficient", "variable": variable_name, "level": level,
                         "coefficient": float(coefficient), "coefficient_standard_error": float(standard_error),
                         "t_statistic": float(t_statistic), "p_value": float(p_value),
                         "count": len(originals)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Positive control
# ----------------------------------------------------------------------------------------------------------

def build_positive_control_check(per_document_scores):
    rewritten = per_document_scores[per_document_scores["rewrite_direction"].notna()]
    columns = ["row_type", "document_id", "source_document_id", "rewrite_direction", "preachy_minus_control_score",
               "source_preachy_minus_control_score", "difference_rewrite_minus_source", "expected_sign_of_difference",
               "moved_in_expected_direction", "number_of_rewrites", "number_moving_in_expected_direction",
               "wilcoxon_signed_rank_p_value_two_sided", "wilcoxon_signed_rank_p_value_expected_direction", "note"]
    if len(rewritten) == 0:
        return pd.DataFrame([{"row_type": "note", "note": "skipped: no rewritten documents in this run"}], columns=columns)
    score_by_document = per_document_scores.set_index("document_id")["preachy_minus_control_score"]
    rows = []
    for _, row in rewritten.iterrows():
        source_id = row["source_document_id"]
        expected_sign = 1 if row["rewrite_direction"] == "preachier" else -1
        if source_id not in score_by_document.index:
            rows.append({"row_type": "document", "document_id": row["document_id"], "source_document_id": source_id,
                         "rewrite_direction": row["rewrite_direction"],
                         "preachy_minus_control_score": row["preachy_minus_control_score"],
                         "note": "source document was not scored; excluded from summary"})
            continue
        source_score = float(score_by_document[source_id])
        difference = float(row["preachy_minus_control_score"]) - source_score
        rows.append({"row_type": "document", "document_id": row["document_id"], "source_document_id": source_id,
                     "rewrite_direction": row["rewrite_direction"],
                     "preachy_minus_control_score": row["preachy_minus_control_score"],
                     "source_preachy_minus_control_score": source_score,
                     "difference_rewrite_minus_source": difference, "expected_sign_of_difference": expected_sign,
                     "moved_in_expected_direction": bool(difference * expected_sign > 0)})
    document_rows = pd.DataFrame(rows)
    for direction in ("preachier", "plainer"):
        subset = document_rows[(document_rows["rewrite_direction"] == direction)
                               & document_rows["difference_rewrite_minus_source"].notna()]
        if len(subset) == 0:
            continue
        differences = subset["difference_rewrite_minus_source"].to_numpy(dtype=float)
        expected_sign = 1 if direction == "preachier" else -1
        if np.all(differences == 0):
            p_two_sided = p_expected = 1.0
        else:
            p_two_sided = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
            p_expected = float(stats.wilcoxon(differences, alternative="greater" if expected_sign > 0 else "less").pvalue)
        rows.append({"row_type": "summary", "rewrite_direction": direction, "number_of_rewrites": len(subset),
                     "expected_sign_of_difference": expected_sign,
                     "number_moving_in_expected_direction": int(subset["moved_in_expected_direction"].sum()),
                     "wilcoxon_signed_rank_p_value_two_sided": p_two_sided,
                     "wilcoxon_signed_rank_p_value_expected_direction": p_expected})
    return pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------

def make_figures(figures_folder, per_document_scores, significance_table, rewording_agreement, conditions,
                 condition_score_matrix, positive_control_check):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    figures_folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    timestamp_suffix = figures_folder.parent.name.replace("analysis_", "")

    def save(figure, name):
        figure.tight_layout()
        figure.savefig(figures_folder / f"{name}_{timestamp_suffix}.png", dpi=160)
        plt.close(figure)

    # 1. Distribution of preachy-minus-control scores, coloured by main-test direction
    merged = per_document_scores.merge(significance_table[["document_id", "main_test_direction"]], on="document_id")
    labels_and_colors = [("elevated", COLOR_ELEVATED), ("suppressed", COLOR_SUPPRESSED),
                         ("no significant difference at q < 0.05", COLOR_NO_DIFFERENCE)]
    all_values = merged["preachy_minus_control_score"].to_numpy()
    bins = np.linspace(all_values.min(), all_values.max() + 1e-12, 31) if len(all_values) and np.ptp(all_values) > 0 else 10
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist([merged.loc[merged["main_test_direction"] == label, "preachy_minus_control_score"] for label, _ in labels_and_colors],
              bins=bins, stacked=True, color=[color for _, color in labels_and_colors],
              label=[f"{label} ({(merged['main_test_direction'] == label).sum()})" for label, _ in labels_and_colors])
    axis.axvline(0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1)
    axis.set_xlabel("Preachy minus control score (mean per-token log-probability difference, nats)")
    axis.set_ylabel("Number of documents")
    axis.set_title("Distribution of preachy-minus-control scores across documents")
    axis.legend(frameon=False, fontsize=8)
    save(figure, "distribution_of_preachy_minus_control_scores")

    # 2. Scores by tone (original documents)
    originals = merged[merged["rewrite_direction"].isna()]
    if "spec_tone" in originals.columns and len(originals):
        tone_levels = sorted(originals["spec_tone"].astype(str).unique())
        figure, axis = plt.subplots(figsize=(8, 1.0 + 1.1 * len(tone_levels)))
        generator = np.random.default_rng(0)
        for position, level in enumerate(tone_levels):
            values = originals.loc[originals["spec_tone"].astype(str) == level, "preachy_minus_control_score"].to_numpy()
            axis.scatter(values, position + generator.uniform(-0.18, 0.18, len(values)), s=10, alpha=0.5, color=COLOR_SUPPRESSED)
            standard_error = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            axis.errorbar(values.mean(), position, xerr=1.96 * standard_error, fmt="D", color=COLOR_ELEVATED,
                          capsize=4, markersize=6)
        axis.set_yticks(range(len(tone_levels)))
        axis.set_yticklabels(["\n".join(textwrap.wrap(level, 45)) for level in tone_levels], fontsize=8)
        axis.axvline(0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1)
        axis.set_xlabel("Preachy minus control score (nats per token); diamond = mean with 95% interval")
        axis.set_title("Preachy-minus-control score by tone (original documents)")
        save(figure, "preachy_minus_control_scores_by_tone")

    # 3. Positive control paired plot
    documents_only = positive_control_check[positive_control_check["row_type"] == "document"] \
        if "row_type" in positive_control_check.columns else positive_control_check.iloc[0:0]
    documents_only = documents_only[documents_only["difference_rewrite_minus_source"].notna()] \
        if len(documents_only) else documents_only
    if len(documents_only):
        figure, axis = plt.subplots(figsize=(6, 4.5))
        positions = {"plainer": 0, "source": 1, "preachier": 2}
        colors = {"preachier": COLOR_PREACHIER, "plainer": COLOR_PLAINER}
        for _, row in documents_only.iterrows():
            direction = row["rewrite_direction"]
            axis.plot([positions["source"], positions[direction]],
                      [row["source_preachy_minus_control_score"], row["preachy_minus_control_score"]],
                      color=colors[direction], marker="o", alpha=0.8,
                      linestyle="-" if row["moved_in_expected_direction"] else ":")
        axis.set_xticks(list(positions.values()))
        axis.set_xticklabels(["plainer rewrite", "original", "preachier rewrite"])
        axis.set_ylabel("Preachy minus control score (nats per token)")
        axis.set_title("Positive control: original versus rewrites\n(dotted line = moved the wrong way)")
        save(figure, "positive_control_paired_plot")

    # 4. Heatmap of preachy rewording agreement
    condition_ids, matrix = pairwise_spearman_matrix(conditions, condition_score_matrix, "preachy")
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="cividis")
    axis.set_xticks(range(len(condition_ids)))
    axis.set_yticks(range(len(condition_ids)))
    short_labels = [condition_id.split("__rewording_")[-1] for condition_id in condition_ids]
    axis.set_xticklabels(short_labels, fontsize=6)
    axis.set_yticklabels(short_labels, fontsize=6)
    axis.set_xlabel("Preachy rewording number")
    axis.set_ylabel("Preachy rewording number")
    axis.set_title("Spearman correlation of document scores between preachy rewordings")
    figure.colorbar(image, ax=axis, label="Spearman correlation across documents")
    save(figure, "preachy_rewording_agreement_heatmap")

    # 5. Per-persona spread of rewording mean scores
    personas = list(dict.fromkeys(rewording_agreement["persona_name"]))
    figure, axis = plt.subplots(figsize=(8, 1.0 + 0.5 * len(personas)))
    for position, persona in enumerate(personas):
        subset = rewording_agreement[rewording_agreement["persona_name"] == persona]
        normal = subset[~subset["rewording_is_outlier"]]
        outliers = subset[subset["rewording_is_outlier"]]
        axis.scatter(normal["rewording_mean_score_across_documents"], [position] * len(normal), color=COLOR_SUPPRESSED, s=18)
        axis.scatter(outliers["rewording_mean_score_across_documents"], [position] * len(outliers), color=COLOR_ELEVATED,
                     marker="X", s=40, label="outlier rewording" if position == 0 or len(outliers) else None)
    axis.set_yticks(range(len(personas)))
    axis.set_yticklabels(personas, fontsize=8)
    axis.axvline(0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1)
    axis.set_xlabel("Mean score across documents, one point per rewording (nats per token)")
    axis.set_title("Spread of rewordings within each persona\n(X = outlier by leave-one-out correlation)")
    handles, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), frameon=False, fontsize=8)
    save(figure, "rewording_spread_by_persona")


# ----------------------------------------------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------------------------------------------

def analysis_notes_text(window_size, windows_per_direction):
    return f"""# Analysis notes: how each output is computed

Inputs: `scoring_inputs/documents.jsonl`, `scoring_inputs/conditions.jsonl`, `scoring_outputs/per_token_log_probabilities/<document_id>.npz`.
All differences are in nats per token. Condition groups come from `conditions.jsonl`.

## Loading checks
Analysis stops with an error if any document has no .npz, if the set of condition ids in any .npz differs from conditions.jsonl, or if the array shape does not match the condition and token counts. Rows are reordered into conditions.jsonl order.

## per_document_scores
- `score__<condition_id>`: mean over the document's tokens of `log_probabilities[condition] - log_probabilities[generic]`.
- `preachy_mean_score`, `control_mean_score`, `proportionate_welfare_mean_score`: mean of the rewording scores over the 30 preachy, 25 control (all control personas pooled) and 5 proportionate-welfare conditions.
- `preachy_minus_control_score`: `preachy_mean_score - control_mean_score`.
- `no_system_prompt_score`: the `score__no_system_prompt` value.
- `number_of_tokens`: number of document tokens. `generic_mean_log_probability`: mean over tokens of `log_probabilities[generic]`.
- `spec_<variable>`, `language`, `source_document_id`, `rewrite_direction`: copied from documents.jsonl.

## per_document_significance_tests
- Main test: two-sided Mann-Whitney U (scipy, automatic exact or asymptotic method) of the 30 preachy rewording scores against the 25 pooled control rewording scores. `main_test_median_preachy_minus_control` is median(preachy scores) minus median(control scores).
- Second test: same, preachy against the 5 proportionate-welfare rewording scores.
- `*_q_value`: Benjamini-Hochberg adjusted p-values (step-up, implemented in this script), computed separately for each test across original documents only (`rewrite_direction` empty). Rewritten documents are not corrected: their `*_q_value` columns hold the uncorrected p-value and `q_value_is_uncorrected_rewrite` is true (they are evaluated by the positive-control check instead); their direction labels therefore use uncorrected p < 0.05.
- `*_direction`: `elevated` if q < 0.05 and the median difference is positive, `suppressed` if q < 0.05 and negative, otherwise `no significant difference at q < 0.05`.
- `preachy_vs_generic_wilcoxon_*`: two-sided one-sample Wilcoxon signed-rank test of the 30 preachy rewording scores against 0 (the generic condition's score is 0 by definition). If all 30 scores are exactly 0 the p-value is set to 1. q is Benjamini-Hochberg across documents. `preachy_significantly_below_generic` is true when q < 0.05 and the median preachy score is below 0; `preachy_significantly_above_generic` likewise above 0.

## extreme_spans (JSONL)
- Per-token preachy difference: mean over the 30 preachy conditions of `log_probabilities[c, t] - log_probabilities[generic, t]`.
- Windows of {window_size} tokens (documents shorter than the window use one window covering the whole document, `window_size_tokens` records the size used). Up to {windows_per_direction} highest and {windows_per_direction} lowest windows, chosen greedily by window mean, with no two windows in the same list overlapping. Highest and lowest lists are selected independently, so a short document can have the same window in both. Fewer windows are returned when the document has no room for more non-overlapping windows.
- `start_token_index` is inclusive and `end_token_index_exclusive` is exclusive, counted over document tokens from 0. `text` is the token strings joined with no separator.

## rewording_agreement
One row per rewording; persona-level columns repeat on each row of the persona. Personas: `preachy`, each control persona, and `proportionate_welfare`. Rewording scores are the `score__<condition_id>` values across documents.
- `persona_mean_pairwise_spearman_across_documents`: mean over all pairs of rewordings of the Spearman correlation of their scores across documents.
- `persona_mean_within_document_sd_across_rewordings`: for each document the standard deviation (n-1) across the persona's rewordings, averaged over documents.
- `persona_sd_across_documents_of_rewording_mean_score`: standard deviation (n-1) across documents of the mean-over-rewordings score.
- `persona_within_to_across_sd_ratio`: the within-document value divided by the across-document value.
- `persona_intraclass_correlation_one_way_single_rewording`: one-way random effects ICC(1,1) with documents as targets and rewordings as raters, (MSB - MSW) / (MSB + (k - 1) MSW).
- `rewording_spearman_with_leave_one_out_mean_of_other_rewordings`: Spearman correlation across documents between this rewording's scores and the mean scores of the persona's other rewordings.
- `rewording_is_outlier`: leave-one-out Spearman below `median - {OUTLIER_MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER} * max(1.4826 * MAD, {OUTLIER_MINIMUM_SPREAD})`, with median and MAD (median absolute deviation) taken over the persona's rewordings; the threshold is in `persona_outlier_threshold_on_leave_one_out_spearman`.
- Spearman is NaN when fewer than 3 documents or when either series is constant.

## scores_by_spec_variable
Original documents only (`rewrite_direction` empty). Outcome is `preachy_minus_control_score`. Variables in order: tone first, then the other eleven spec variables, then language.
- `summary_by_level` rows: count, mean and standard error of the mean (sd / sqrt(count)) per level.
- `regression_coefficient` rows: ordinary least squares with an intercept and one-hot columns for every level of every included variable except the alphabetically first level (listed in `regression_reference_level` rows). Standard errors are the classical homoscedastic ones, `sqrt(diag(pinv(X'X)) * RSS / (n - rank))`; p-values from a t distribution with n - rank degrees of freedom. Variables with more than {MAXIMUM_LEVELS_FOR_REGRESSION_VARIABLE} levels are left out of the regression (marked `NOT_INCLUDED_...`). The pseudo-inverse gives the minimum-norm solution if any columns are collinear.

## positive_control_check
- Document rows: for each rewritten document, its `preachy_minus_control_score`, the source document's score, the difference (rewrite minus source), the expected sign (+1 preachier, -1 plainer) and whether the difference has the expected sign (a difference of exactly 0 counts as not moving).
- Summary rows per direction: number of rewrites, number moving in the expected direction, and the Wilcoxon signed-rank p-value of the differences against 0, both two-sided and one-sided in the expected direction.
- If the run has no rewritten documents the file contains a single row with a skip note.

## figures
Colours are from the Okabe-Ito colourblind-safe palette. Figures: distribution of preachy-minus-control scores coloured by main-test direction; scores by tone (original documents; points are documents, diamond is mean with a 95% interval of 1.96 standard errors); positive-control paired plot; heatmap of Spearman correlations between preachy rewordings; spread of rewording mean scores per persona.
"""


# ----------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------

def run_analysis(run_folder, window_size=32, windows_per_direction=3, timestamp=None):
    run_folder = Path(run_folder)
    timestamp = timestamp or timestamp_now()
    analysis_folder = run_folder / f"analysis_{timestamp}"
    analysis_folder.mkdir(parents=True, exist_ok=True)

    documents, conditions, scores_by_document = load_run(run_folder)
    per_document_scores, condition_score_matrix = build_score_tables(documents, conditions, scores_by_document)
    significance_table = build_significance_table(documents, conditions, condition_score_matrix)
    extreme_span_records = build_extreme_spans(documents, conditions, scores_by_document, window_size, windows_per_direction)
    rewording_agreement = build_rewording_agreement(conditions, condition_score_matrix)
    scores_by_spec_variable = build_scores_by_spec_variable(per_document_scores)
    positive_control_check = build_positive_control_check(per_document_scores)

    per_document_scores.to_csv(analysis_folder / f"per_document_scores_{timestamp}.csv", index=False)
    significance_table.to_csv(analysis_folder / f"per_document_significance_tests_{timestamp}.csv", index=False)
    write_jsonl(analysis_folder / f"extreme_spans_{timestamp}.jsonl", extreme_span_records)
    rewording_agreement.to_csv(analysis_folder / f"rewording_agreement_{timestamp}.csv", index=False)
    scores_by_spec_variable.to_csv(analysis_folder / f"scores_by_spec_variable_{timestamp}.csv", index=False)
    positive_control_check.to_csv(analysis_folder / f"positive_control_check_{timestamp}.csv", index=False)
    (analysis_folder / f"analysis_notes_{timestamp}.md").write_text(
        analysis_notes_text(window_size, windows_per_direction), encoding="utf-8")
    make_figures(analysis_folder / "figures", per_document_scores, significance_table, rewording_agreement, conditions,
                 condition_score_matrix, positive_control_check)
    if positive_control_check.iloc[0].get("row_type") == "note":
        print("Positive control check skipped: no rewritten documents in this run.")
    print(f"Analysis written to {analysis_folder}")
    return analysis_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True, help="Run folder containing scoring_inputs/ and scoring_outputs/")
    parser.add_argument("--window-size-tokens", type=int, default=32)
    parser.add_argument("--windows-per-direction", type=int, default=3)
    arguments = parser.parse_args()
    run_analysis(arguments.run_folder, arguments.window_size_tokens, arguments.windows_per_direction)


if __name__ == "__main__":
    main()
