"""Pure statistics for the persona vector projection analysis. No files, no models, no API.

Every function that could silently invalidate the experiment if wrong has a test in
test_persona_projection_analysis_statistics.py:
  - baseline subtraction uses the baselines of the right document (compute_document_scores)
  - standardisation within language (standardise_within_language)
  - standard errors clustered by source document (cluster_robust_mean, cluster_robust_contrast)
  - rewrites compared with their own source (whole_document_rewrite_shifts, changed_span_rewrite_shifts)
"""

import numpy as np
import pandas as pd
from scipy import stats

from persona_vector_projection.changed_token_counting_by_diff import changed_masks_from_token_sequences

DOCUMENT_KIND_CORPUS = "corpus"
DOCUMENT_KIND_REWRITE = "rewrite"
DOCUMENT_KIND_BASELINE = "baseline"

DIRECTION_TOWARD = "toward_facet"
DIRECTION_AWAY = "away_from_facet"
DIRECTION_NEUTRAL = "facet_neutral"
EXPECTED_SIGN_BY_DIRECTION = {DIRECTION_TOWARD: 1, DIRECTION_AWAY: -1, DIRECTION_NEUTRAL: 0}

MINIMUM_DOCUMENTS_FOR_LANGUAGE_STANDARDISATION = 5


# ----------------------------------------------------------------------------
# Intervals and clustered standard errors
# ----------------------------------------------------------------------------

def mean_with_t_interval(values, confidence=0.95):
    """Mean, standard error and t interval of independent values. NaN interval for fewer than 2 values."""
    values = np.asarray([value for value in values if not pd.isna(value)], dtype=float)
    count = len(values)
    if count == 0:
        return {"n": 0, "mean": np.nan, "standard_error": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(values.mean())
    if count < 2:
        return {"n": count, "mean": mean, "standard_error": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    standard_error = float(values.std(ddof=1) / np.sqrt(count))
    margin = stats.t.ppf(0.5 + confidence / 2, count - 1) * standard_error
    return {"n": count, "mean": mean, "standard_error": standard_error, "ci_low": mean - margin,
            "ci_high": mean + margin}


def cluster_robust_mean(values, cluster_ids, confidence=0.95):
    """Mean of `values` with a standard error clustered by `cluster_ids` (for example the source document id).

    CR1 estimator for an intercept-only regression: var = G/(G-1) * sum_over_clusters(sum of residuals in cluster)^2
    / n^2, with a t interval on G-1 degrees of freedom. Rows from one source document are correlated, so treating
    them as independent understates the standard error."""
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "cluster": list(cluster_ids)}).dropna(
        subset=["value"])
    count = len(frame)
    clusters = frame["cluster"].nunique()
    if count == 0:
        return {"n": 0, "clusters": 0, "mean": np.nan, "standard_error": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(frame["value"].mean())
    if clusters < 2:
        return {"n": count, "clusters": clusters, "mean": mean, "standard_error": np.nan, "ci_low": np.nan,
                "ci_high": np.nan}
    residual_sums = (frame["value"] - mean).groupby(frame["cluster"]).sum()
    variance = clusters / (clusters - 1) * float((residual_sums ** 2).sum()) / count ** 2
    standard_error = float(np.sqrt(variance))
    margin = stats.t.ppf(0.5 + confidence / 2, clusters - 1) * standard_error
    return {"n": count, "clusters": clusters, "mean": mean, "standard_error": standard_error,
            "ci_low": mean - margin, "ci_high": mean + margin}


def cluster_robust_contrast(values, group_labels, cluster_ids, group_a, group_b, confidence=0.95):
    """Mean(group_a) - mean(group_b), standard error clustered by cluster_ids (OLS on a group indicator, CR1).
    Works whether or not the same source document appears in both groups (when it does, the clustering takes
    their correlation into account)."""
    import statsmodels.api as sm
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "group": list(group_labels),
                          "cluster": list(cluster_ids)}).dropna(subset=["value"])
    frame = frame[frame["group"].isin([group_a, group_b])]
    clusters = frame["cluster"].nunique()
    if frame["group"].nunique() < 2 or clusters < 2:
        return {"n": len(frame), "clusters": clusters, "difference": np.nan, "standard_error": np.nan,
                "ci_low": np.nan, "ci_high": np.nan}
    design = sm.add_constant((frame["group"] == group_a).astype(float).to_numpy())
    cluster_codes = pd.factorize(frame["cluster"])[0]
    fit = sm.OLS(frame["value"].to_numpy(), design).fit(cov_type="cluster", cov_kwds={"groups": cluster_codes})
    difference, standard_error = float(fit.params[1]), float(fit.bse[1])
    margin = stats.t.ppf(0.5 + confidence / 2, clusters - 1) * standard_error
    return {"n": len(frame), "clusters": clusters, "difference": difference, "standard_error": standard_error,
            "ci_low": difference - margin, "ci_high": difference + margin}


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted q values for a list of p values (NaN stays NaN)."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(len(p_values), np.nan)
    valid = np.where(~np.isnan(p_values))[0]
    if len(valid) == 0:
        return q_values
    order = valid[np.argsort(p_values[valid])]
    ranked = p_values[order] * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values[order] = np.minimum(adjusted, 1.0)
    return q_values


def safe_wilcoxon_p_value(values):
    values = np.asarray([value for value in values if not pd.isna(value)], dtype=float)
    if len(values) < 2 or not np.any(values != 0):
        return np.nan
    return float(stats.wilcoxon(values).pvalue)


# ----------------------------------------------------------------------------
# Document scores: baseline subtraction and standardisation within language
# ----------------------------------------------------------------------------

def baseline_owner_of_each_document(documents):
    """The document whose baselines a document is compared with: itself for a corpus document, its source for a
    rewrite. `documents` is a DataFrame indexed by document_id with columns kind and source_document_id."""
    owners = {}
    for document_id, row in documents.iterrows():
        if row["kind"] == DOCUMENT_KIND_CORPUS:
            owners[document_id] = document_id
        elif row["kind"] == DOCUMENT_KIND_REWRITE:
            owners[document_id] = row["source_document_id"]
    return pd.Series(owners, name="baseline_owner_id", dtype=object)


def mean_baseline_by_owner(mean_values, documents, baseline_indices_to_use=None):
    """Mean over each owner's baseline documents of `mean_values` (DataFrame: document x vector). Returns a
    DataFrame indexed by owner id, and a Series with the number of baselines averaged.
    `baseline_indices_to_use` restricts to given baseline indices (for split-half agreement)."""
    baselines = documents[documents["kind"] == DOCUMENT_KIND_BASELINE]
    if baseline_indices_to_use is not None:
        baselines = baselines[baselines["baseline_index"].isin(baseline_indices_to_use)]
    baseline_values = mean_values.loc[baselines.index]
    owner = baselines["source_document_id"]
    return baseline_values.groupby(owner).mean(), baseline_values.groupby(owner).size()


def compute_document_scores(mean_values, documents, baseline_indices_to_use=None):
    """Raw document score = the document's mean projection minus the mean of its OWNER's baselines' mean projections.

    `mean_values`: DataFrame (document id x vector name) of mean projection over tokens. Returns a DataFrame for
    corpus and rewrite documents (baselines excluded) with the same columns, NaN where the owner has no baseline,
    plus the owner id and baseline count in a second return value."""
    owners = baseline_owner_of_each_document(documents)
    baseline_means, baseline_counts = mean_baseline_by_owner(mean_values, documents, baseline_indices_to_use)
    scored_ids = [document_id for document_id in owners.index if document_id in mean_values.index]
    owner_baselines = baseline_means.reindex(owners.loc[scored_ids].to_numpy())
    owner_baselines.index = scored_ids
    scores = mean_values.loc[scored_ids] - owner_baselines
    information = pd.DataFrame({"baseline_owner_id": owners.loc[scored_ids],
                                "number_of_baselines": baseline_counts.reindex(owners.loc[scored_ids].to_numpy())
                                .fillna(0).astype(int).to_numpy()}, index=scored_ids)
    return scores, information


def language_standardisation_statistics(scores, languages, corpus_mask,
                                        minimum_documents=MINIMUM_DOCUMENTS_FOR_LANGUAGE_STANDARDISATION):
    """Mean and standard deviation of each vector's score within each language, from CORPUS documents only.

    A language with fewer than `minimum_documents` corpus documents uses the all-language statistics, flagged in
    `used_all_language_statistics`. Returns (statistics DataFrame indexed by language with columns
    <vector>__mean, <vector>__sd, n_corpus_documents, used_all_language_statistics)."""
    corpus_scores = scores[corpus_mask]
    corpus_languages = languages[corpus_mask]
    overall_means, overall_sds = corpus_scores.mean(), corpus_scores.std(ddof=1)
    rows = {}
    for language in sorted(languages.unique()):
        in_language = corpus_scores[corpus_languages == language]
        small = len(in_language) < minimum_documents
        row = {"n_corpus_documents": len(in_language), "used_all_language_statistics": small}
        for vector in scores.columns:
            row[f"{vector}__mean"] = overall_means[vector] if small else in_language[vector].mean()
            row[f"{vector}__sd"] = overall_sds[vector] if small else in_language[vector].std(ddof=1)
        rows[language] = row
    return pd.DataFrame(rows).T


def standardise_within_language(scores, languages, standardisation_statistics, centre=True):
    """(score - language mean) / language sd, per vector. `centre=False` only rescales (used for differences between
    a rewrite and its source, which are in the units of the language's score spread)."""
    standardised = scores.copy()
    for vector in scores.columns:
        means = languages.map(standardisation_statistics[f"{vector}__mean"].astype(float))
        sds = languages.map(standardisation_statistics[f"{vector}__sd"].astype(float))
        standardised[vector] = (scores[vector] - (means if centre else 0.0)) / sds
    return standardised


# ----------------------------------------------------------------------------
# Rewrites against their own source
# ----------------------------------------------------------------------------

def whole_document_rewrite_shifts(scores, documents):
    """For each rewrite: its score minus the score of ITS OWN source document, per vector (DataFrame indexed by
    rewrite id). Rewrites whose source has no score are dropped."""
    rewrites = documents[documents["kind"] == DOCUMENT_KIND_REWRITE]
    rewrites = rewrites[rewrites["source_document_id"].isin(scores.index) & rewrites.index.isin(scores.index)]
    source_scores = scores.loc[rewrites["source_document_id"].to_numpy()]
    source_scores.index = rewrites.index
    return scores.loc[rewrites.index] - source_scores


def changed_span_rewrite_shifts(per_token_projections, documents, token_ids, vector_names, scores_baseline_by_owner,
                                minimum_changed_tokens=0):
    """Span-level shift per rewrite and vector, compared with the rewrite's OWN source.

    per_token_projections: document id -> (tokens, vectors) array; token_ids: document id -> list of ints;
    scores_baseline_by_owner: DataFrame indexed by owner id (the mean baseline projection to subtract, per vector).
    For each rewrite, tokens are aligned to the source's tokens with a sequence diff. Returned columns per
    vector: <v>__changed_minus_replaced (mean score on the rewrite's changed tokens minus mean score on the source
    tokens they replaced), <v>__changed_minus_unchanged (changed minus the rewrite's unchanged tokens), plus counts.
    A rewrite with fewer than `minimum_changed_tokens` changed tokens, or none replaced, gets NaN."""
    rows = {}
    rewrites = documents[documents["kind"] == DOCUMENT_KIND_REWRITE]
    for rewrite_id, row in rewrites.iterrows():
        source_id = row["source_document_id"]
        if (rewrite_id not in per_token_projections or source_id not in per_token_projections
                or source_id not in scores_baseline_by_owner.index):
            continue
        rewrite_changed, source_replaced = changed_masks_from_token_sequences(token_ids[source_id], token_ids[rewrite_id])
        baseline = scores_baseline_by_owner.loc[source_id, vector_names].to_numpy(dtype=float)
        rewrite_values = per_token_projections[rewrite_id] - baseline
        source_values = per_token_projections[source_id] - baseline
        output = {"changed_tokens": int(rewrite_changed.sum()), "replaced_source_tokens": int(source_replaced.sum()),
                  "rewrite_tokens": len(rewrite_changed), "source_tokens": len(source_replaced)}
        enough = (rewrite_changed.sum() >= max(1, minimum_changed_tokens)) and source_replaced.any()
        for column, vector in enumerate(vector_names):
            if enough:
                changed = rewrite_values[rewrite_changed, column].mean()
                replaced = source_values[source_replaced, column].mean()
                unchanged = rewrite_values[~rewrite_changed, column].mean() if (~rewrite_changed).any() else np.nan
                output[f"{vector}__changed_minus_replaced"] = changed - replaced
                output[f"{vector}__changed_minus_unchanged"] = changed - unchanged
            else:
                output[f"{vector}__changed_minus_replaced"] = np.nan
                output[f"{vector}__changed_minus_unchanged"] = np.nan
        rows[rewrite_id] = output
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def summarise_positive_control(shift_table, group_columns, value_column, confidence=0.95):
    """Per group (for example facet x direction): n, mean shift, standard error clustered by source document, CI,
    the number of rewrites moving the expected way, and a Wilcoxon p value.

    `shift_table` has columns source_document_id, rewrite_direction, value_column and the group columns.
    Expected way: positive for toward_facet, negative for away_from_facet; facet_neutral has no expected sign
    (the count is reported against zero as 'positive')."""
    rows = []
    for group_key, group in shift_table.groupby(group_columns):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        direction = group["rewrite_direction"].iloc[0]
        values = group[value_column].astype(float)
        clustered = cluster_robust_mean(values, group["source_document_id"], confidence)
        expected_sign = EXPECTED_SIGN_BY_DIRECTION.get(direction, 0)
        valid = values.dropna()
        moving = int((np.sign(valid) == expected_sign).sum()) if expected_sign else int((valid > 0).sum())
        rows.append({**dict(zip(group_columns, group_key)), "n_rewrites": clustered["n"],
                     "n_source_documents": clustered["clusters"], "mean_shift": clustered["mean"],
                     "clustered_standard_error": clustered["standard_error"], "ci_low": clustered["ci_low"],
                     "ci_high": clustered["ci_high"], "rewrites_moving_expected_way": moving,
                     "wilcoxon_p_two_sided": safe_wilcoxon_p_value(valid)})
    return pd.DataFrame(rows)


def contrast_against_neutral(shift_table, value_column, confidence=0.95):
    """Toward minus neutral and away minus neutral, per facet, clustered by source document. `shift_table` has
    facet, rewrite_direction, source_document_id and value_column."""
    rows = []
    for facet, group in shift_table.groupby("facet"):
        for direction in (DIRECTION_TOWARD, DIRECTION_AWAY):
            result = cluster_robust_contrast(group[value_column], group["rewrite_direction"],
                                             group["source_document_id"], direction, DIRECTION_NEUTRAL, confidence)
            rows.append({"facet": facet, "contrast": f"{direction}_minus_{DIRECTION_NEUTRAL}", **result})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Group means (spec variables, depiction labels), bands, windows
# ----------------------------------------------------------------------------

def group_mean_table(values, groups, minimum_documents, confidence=0.95):
    """Mean of `values` per group with a t interval, Welch test against all other documents, Benjamini-Hochberg q
    values across the groups, and a flag for groups too small to rank.

    A group with fewer than `minimum_documents` documents is marked `too_few_documents` and gets no rank. Rank 1 is
    the highest mean among groups that are not too small. `separable_from_rest` is True when q < 0.05."""
    values = pd.Series(np.asarray(values, dtype=float), index=range(len(values)))
    groups = pd.Series(list(groups), index=values.index)
    rows = []
    for group_value in sorted(groups.dropna().unique(), key=str):
        in_group = values[groups == group_value].dropna()
        rest = values[groups != group_value].dropna()
        interval = mean_with_t_interval(in_group, confidence)
        p_value = (stats.ttest_ind(in_group, rest, equal_var=False).pvalue
                   if len(in_group) >= 2 and len(rest) >= 2 else np.nan)
        rows.append({"group": group_value, "n": interval["n"], "mean": interval["mean"], "ci_low": interval["ci_low"],
                     "ci_high": interval["ci_high"], "mean_minus_rest": interval["mean"] - rest.mean(),
                     "welch_p_value_against_rest": p_value, "too_few_documents": len(in_group) < minimum_documents})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["q_value"] = benjamini_hochberg(table["welch_p_value_against_rest"])
    table["separable_from_rest"] = (table["q_value"] < 0.05) & ~table["too_few_documents"]
    ranked = table[~table["too_few_documents"]]["mean"].rank(ascending=False, method="min")
    table["rank_by_mean"] = ranked.reindex(table.index)
    return table.sort_values(["too_few_documents", "mean"], ascending=[True, False]).reset_index(drop=True)


def rolling_window_means(values, window):
    """Mean of every window of `window` consecutive values (length n - window + 1; empty if shorter)."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return np.array([])
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    return (cumulative[window:] - cumulative[:-window]) / window


def select_non_overlapping_windows(window_means, window, count, highest, minimum_start=0):
    """Greedy: the `count` highest (or lowest) windows that do not overlap each other, starting at or after
    `minimum_start`. Returns a list of (start, mean)."""
    candidates = [(start, window_means[start]) for start in range(minimum_start, len(window_means))]
    candidates.sort(key=lambda item: item[1], reverse=highest)
    chosen = []
    for start, mean in candidates:
        if all(abs(start - other) >= window for other, _ in chosen):
            chosen.append((start, float(mean)))
            if len(chosen) == count:
                break
    return chosen


def rank_agreement(first_scores, second_scores, top_fraction=0.25):
    """Spearman correlation and top-fraction overlap between two score vectors over the same documents."""
    frame = pd.DataFrame({"first": first_scores, "second": second_scores}).dropna()
    if len(frame) < 3:
        return {"n": len(frame), "spearman": np.nan, "top_fraction_overlap": np.nan}
    spearman = float(stats.spearmanr(frame["first"], frame["second"]).statistic)
    top_count = max(1, int(round(top_fraction * len(frame))))
    top_first = set(frame["first"].nlargest(top_count).index)
    top_second = set(frame["second"].nlargest(top_count).index)
    return {"n": len(frame), "spearman": spearman, "top_fraction_overlap": len(top_first & top_second) / top_count}
