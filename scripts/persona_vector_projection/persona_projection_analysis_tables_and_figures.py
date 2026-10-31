"""Tables and figures of the persona vector projection analysis, built from ProjectionData. No API, no model.

Each build_* function returns a DataFrame (or a dict of them); write/plot functions save them. Everything is
computed per facet, never combined across facets. Scores are in standard deviations of the corpus documents of the
document's own language ("z scores"), after subtracting each document's matched baseline.
"""

import textwrap

import numpy as np
import pandas as pd
from scipy import stats

from persona_vector_projection import persona_vector_experiment_definitions as defs
from persona_vector_projection import persona_projection_analysis_statistics as statistics_module
from persona_vector_projection.persona_projection_analysis_statistics import (
    DIRECTION_AWAY, DIRECTION_NEUTRAL, DIRECTION_TOWARD, DOCUMENT_KIND_CORPUS, DOCUMENT_KIND_REWRITE)

STRUCTURAL_COLUMNS = ["kind", "source_document_id", "language", "facet", "rewrite_direction", "baseline_index"]
DEFAULT_SPEC_VARIABLES_FOR_FIGURES = ["domain", "tone", "sentient_category", "decision_scale"]
PERSONA_FACETS = list(defs.PERSONA_FACETS)
VARIANT_SEPARATOR = "__"
# Colour-blind safe (Okabe-Ito)
COLOR_TOWARD, COLOR_AWAY, COLOR_NEUTRAL = "#009E73", "#D55E00", "#7F7F7F"
COLOR_POINT, COLOR_TOO_FEW = "#0072B2", "#BBBBBB"


def facets_present(vector_names):
    return [facet for facet in PERSONA_FACETS if facet in vector_names]


def variants_present(vector_names, facet):
    """Columns named <facet>__<variant> (split-half vectors, LEACE-erased vector, other layers) in the scoring output."""
    return [name for name in vector_names if name.startswith(facet + VARIANT_SEPARATOR)]


# ----------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------

class ScoreTables:
    """Raw and standardised document scores for one primary-score choice ('overall' or 'after_opening')."""

    def __init__(self, data, primary_score="overall", minimum_documents_for_language_standardisation=
                 statistics_module.MINIMUM_DOCUMENTS_FOR_LANGUAGE_STANDARDISATION):
        self.data = data
        self.primary_score = primary_score
        documents = data.documents
        mean_values = {"overall": data.mean_overall, "after_opening": data.mean_after_opening}[primary_score]
        self.raw_scores, self.information = statistics_module.compute_document_scores(mean_values, documents)
        self.opening_raw_scores, _ = statistics_module.compute_document_scores(
            data.mean_opening.reindex(data.mean_overall.index), documents)
        self.baseline_means_overall, _ = statistics_module.mean_baseline_by_owner(data.mean_overall, documents)
        self.baseline_means_primary, _ = statistics_module.mean_baseline_by_owner(mean_values, documents)
        scored = documents.loc[self.raw_scores.index]
        self.languages = scored["language"]
        corpus_mask = (scored["kind"] == DOCUMENT_KIND_CORPUS).to_numpy()
        self.language_statistics = statistics_module.language_standardisation_statistics(
            self.raw_scores, self.languages, pd.Series(corpus_mask, index=scored.index),
            minimum_documents_for_language_standardisation)
        self.z_scores = statistics_module.standardise_within_language(
            self.raw_scores, self.languages, self.language_statistics)
        self.corpus_ids = list(scored.index[corpus_mask])
        self.scored_documents = scored


def build_score_distribution_tables(score_tables, facets):
    """Per facet and language: n, mean and sd of raw scores, and quantiles of z scores (corpus documents)."""
    rows = []
    corpus_raw = score_tables.raw_scores.loc[score_tables.corpus_ids]
    corpus_z = score_tables.z_scores.loc[score_tables.corpus_ids]
    languages = score_tables.languages.loc[score_tables.corpus_ids]
    for facet in facets:
        for language in ["all languages"] + sorted(languages.unique()):
            mask = np.ones(len(languages), bool) if language == "all languages" else (languages == language).to_numpy()
            raw, z = corpus_raw.loc[mask, facet].dropna(), corpus_z.loc[mask, facet].dropna()
            rows.append({"facet": facet, "language": language, "n_documents": len(raw), "raw_score_mean": raw.mean(),
                         "raw_score_sd": raw.std(ddof=1) if len(raw) > 1 else np.nan,
                         **{f"z_quantile_{int(q * 100):02d}": z.quantile(q) if len(z) else np.nan
                            for q in (0.05, 0.25, 0.5, 0.75, 0.95)}})
    return pd.DataFrame(rows)


def build_scores_by_spec_variable(score_tables, facets, spec_variables, minimum_documents_per_value):
    corpus_ids = score_tables.corpus_ids
    documents = score_tables.data.documents.loc[corpus_ids]
    tables = []
    for facet in facets:
        z = score_tables.z_scores.loc[corpus_ids, facet]
        for variable in spec_variables:
            if variable not in documents.columns:
                continue
            table = statistics_module.group_mean_table(z.to_numpy(), documents[variable].to_numpy(),
                                                       minimum_documents_per_value)
            table.insert(0, "variable", variable)
            table.insert(0, "facet", facet)
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def build_scores_by_depiction_label(score_tables, labels, facets, minimum_documents_per_value):
    """Mean z score of each facet by what the document's AI is labelled as doing on that facet."""
    if labels is None:
        return pd.DataFrame()
    tables = []
    for facet in facets:
        column = f"label__{facet}"
        ids = [document_id for document_id in score_tables.corpus_ids if document_id in labels.index]
        if column not in labels.columns or not ids:
            continue
        table = statistics_module.group_mean_table(score_tables.z_scores.loc[ids, facet].to_numpy(),
                                                   labels.loc[ids, column].to_numpy(), minimum_documents_per_value)
        table.insert(0, "facet", facet)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def build_scores_by_position_band(score_tables, facets):
    """Mean z score per position band (relative fifths of the document) plus the opening and the rest, corpus documents.

    Band scores are band mean minus the owner's baseline mean, then shifted and scaled with the language's
    whole-document statistics, so a band's value reads as 'this part of the document, relative to the language's
    typical whole-document score'."""
    data = score_tables.data
    rows = []
    corpus_ids = [document_id for document_id in score_tables.corpus_ids if document_id in data.mean_bands]
    statistics_table = score_tables.language_statistics
    for facet in facets:
        column = list(data.vector_names).index(facet)
        owner_baselines = score_tables.baseline_means_overall[facet]
        series = {}
        for document_id in corpus_ids:
            language = score_tables.languages[document_id]
            mean, sd = float(statistics_table.loc[language, f"{facet}__mean"]), float(statistics_table.loc[language, f"{facet}__sd"])
            baseline = owner_baselines.get(document_id, np.nan)
            for band, value in enumerate(data.mean_bands[document_id][:, column]):
                series.setdefault(f"band_{band + 1}_of_{len(data.mean_bands[document_id])}", []).append(
                    (value - baseline - mean) / sd)
            series.setdefault(f"opening_first_{data.opening_token_count}_tokens", []).append(
                (data.mean_opening.loc[document_id, facet] - baseline - mean) / sd)
            series.setdefault("after_opening", []).append((data.mean_after_opening.loc[document_id, facet] - baseline - mean) / sd)
        for band_name, values in series.items():
            interval = statistics_module.mean_with_t_interval(values)
            rows.append({"facet": facet, "position": band_name, **interval})
    return pd.DataFrame(rows)


def build_openings_table(score_tables, facets):
    """Opening (first tokens) against the rest: spread of z scores and the correlation between the two parts."""
    corpus_ids = score_tables.corpus_ids
    data = score_tables.data
    rows = []
    for facet in facets:
        baseline = score_tables.baseline_means_overall[facet].reindex(corpus_ids)
        opening = (data.mean_opening.loc[corpus_ids, facet] - baseline)
        rest = (data.mean_after_opening.loc[corpus_ids, facet] - baseline)
        frame = pd.DataFrame({"opening": opening, "rest": rest}).dropna()
        rows.append({"facet": facet, "n_documents": len(frame), "opening_mean_minus_baseline": frame["opening"].mean(),
                     "opening_sd": frame["opening"].std(ddof=1), "rest_mean_minus_baseline": frame["rest"].mean(),
                     "rest_sd": frame["rest"].std(ddof=1),
                     "opening_sd_divided_by_rest_sd": frame["opening"].std(ddof=1) / frame["rest"].std(ddof=1),
                     "spearman_opening_against_rest": stats.spearmanr(frame["opening"], frame["rest"]).statistic
                     if len(frame) > 2 else np.nan})
    return pd.DataFrame(rows)


def build_cross_facet_correlation(score_tables, facets, correlation_flag_threshold=0.8):
    """Spearman correlation of z scores between facets over corpus documents, and the pairs above the threshold."""
    z = score_tables.z_scores.loc[score_tables.corpus_ids, facets]
    matrix = z.corr(method="spearman")
    pairs = [{"facet_a": a, "facet_b": b, "spearman": matrix.loc[a, b],
              "above_threshold": abs(matrix.loc[a, b]) > correlation_flag_threshold}
             for index, a in enumerate(facets) for b in facets[index + 1:]]
    return matrix, pd.DataFrame(pairs)


# ----------------------------------------------------------------------------
# Rank agreement: baseline split, vector variants, neighbouring layers
# ----------------------------------------------------------------------------

def build_baseline_split_half_agreement(data, facets, primary_score="overall"):
    """Ranking agreement of document scores computed with the even-numbered baselines against the odd-numbered ones:
    how much of the ranking is baseline noise."""
    mean_values = {"overall": data.mean_overall, "after_opening": data.mean_after_opening}[primary_score]
    baseline_indices = sorted(data.documents["baseline_index"].dropna().astype(int).unique())
    first, second = baseline_indices[0::2], baseline_indices[1::2]
    if not first or not second:
        return pd.DataFrame()
    scores_first, _ = statistics_module.compute_document_scores(mean_values, data.documents, first)
    scores_second, _ = statistics_module.compute_document_scores(mean_values, data.documents, second)
    corpus_ids = list(data.documents.index[data.documents["kind"] == DOCUMENT_KIND_CORPUS])
    rows = []
    for facet in facets:
        agreement = statistics_module.rank_agreement(scores_first.loc[corpus_ids, facet], scores_second.loc[corpus_ids, facet])
        rows.append({"facet": facet, "comparison": f"baselines {first} against baselines {second}", **agreement})
    return pd.DataFrame(rows)


def build_variant_ranking_agreement(score_tables, facets):
    """Spearman agreement between the main facet vector and every <facet>__<variant> column in the scoring output
    (vectors extracted on each half of the domains, LEACE-erased vector, ...), and between the two domain halves."""
    corpus_ids = score_tables.corpus_ids
    rows = []
    for facet in facets:
        variants = variants_present(score_tables.raw_scores.columns, facet)
        for variant in variants:
            rows.append({"facet": facet, "compared": f"{facet} against {variant}",
                         **statistics_module.rank_agreement(score_tables.raw_scores.loc[corpus_ids, facet],
                                                            score_tables.raw_scores.loc[corpus_ids, variant])})
        halves = [variant for variant in variants if "half_A" in variant or "half_B" in variant]
        half_a = next((variant for variant in halves if "half_A" in variant), None)
        half_b = next((variant for variant in halves if "half_B" in variant), None)
        if half_a and half_b:
            rows.append({"facet": facet, "compared": f"{half_a} against {half_b}",
                         **statistics_module.rank_agreement(score_tables.raw_scores.loc[corpus_ids, half_a],
                                                            score_tables.raw_scores.loc[corpus_ids, half_b])})
    return pd.DataFrame(rows)


def build_neighbouring_layer_agreement(main_score_tables, other_layer_score_tables, facets):
    """Spearman agreement of document scores at the main layer against each other layer: {layer: ScoreTables}."""
    rows = []
    corpus_ids = main_score_tables.corpus_ids
    for layer, other in sorted(other_layer_score_tables.items()):
        for facet in facets:
            rows.append({"facet": facet, "main_layer": main_score_tables.data.layer_index, "other_layer": layer,
                         **statistics_module.rank_agreement(main_score_tables.raw_scores.loc[corpus_ids, facet],
                                                            other.raw_scores.loc[corpus_ids, facet])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Extreme spans
# ----------------------------------------------------------------------------

def tokens_to_text(token_strings):
    """Readable text from SentencePiece-style token strings."""
    return "".join(token_strings).replace("▁", " ").replace("<0x0A>", "\n")


def build_extreme_spans(score_tables, facets, token_strings_by_document, window_tokens=32, windows_per_document=3,
                        spans_per_direction=25, context_tokens=24):
    """Highest and lowest non-overlapping token windows of the per-token score (minus the document's baseline mean,
    z-scaled with the language's statistics).

    Per document, up to `windows_per_document` windows each way are taken, starting at or after the opening
    (openings are noisy and would dominate a maximum); the corpus-wide top `spans_per_direction` of those are listed.
    Windows that start inside the opening are listed separately, flagged `starts_in_opening`. Text comes from
    `token_strings_by_document` (None: token positions only)."""
    data = score_tables.data
    rows = []
    for facet in facets:
        column = list(data.vector_names).index(facet)
        candidates = []
        for document_id in score_tables.corpus_ids:
            if document_id not in data.per_token:
                continue
            language = score_tables.languages[document_id]
            mean, sd = float(score_tables.language_statistics.loc[language, f"{facet}__mean"]), float(
                score_tables.language_statistics.loc[language, f"{facet}__sd"])
            baseline = score_tables.baseline_means_overall[facet].get(document_id, np.nan)
            per_token = data.per_token[document_id][:, column]
            window_means = (statistics_module.rolling_window_means(per_token, window_tokens) - baseline - mean) / sd
            for highest in (True, False):
                for start, value in statistics_module.select_non_overlapping_windows(
                        window_means, window_tokens, windows_per_document, highest, minimum_start=0):
                    candidates.append((document_id, highest, start, value))
        for opening_flag in (False, True):
            for highest in (True, False):
                chosen = [c for c in candidates if c[1] == highest and (c[2] < data.opening_token_count) == opening_flag]
                chosen.sort(key=lambda c: c[3], reverse=highest)
                for rank, (document_id, _, start, value) in enumerate(chosen[:spans_per_direction], start=1):
                    strings = token_strings_by_document.get(document_id) if token_strings_by_document else None
                    rows.append({
                        "facet": facet, "direction": "highest" if highest else "lowest", "rank": rank,
                        "starts_in_opening": opening_flag, "document_id": document_id,
                        "language": score_tables.languages[document_id], "start_token": start,
                        "window_tokens": window_tokens, "window_score_z": value,
                        "text": tokens_to_text(strings[start:start + window_tokens]) if strings else None,
                        "context_before": tokens_to_text(strings[max(0, start - context_tokens):start]) if strings else None,
                        "context_after": tokens_to_text(strings[start + window_tokens:start + window_tokens + context_tokens])
                        if strings else None})
    return pd.DataFrame(rows)


def write_extreme_spans_markdown(path, spans, facets, display_names):
    lines = ["# Extreme spans for hand reading", "",
             "Highest and lowest 32-token windows of the per-token score, per facet. Each is shown with context; the "
             "marked part is the window. Windows starting in the first tokens of a document (titles, bylines) are "
             "listed apart from the rest.", ""]
    for facet in facets:
        for opening_flag in (False, True):
            for direction in ("highest", "lowest"):
                subset = spans[(spans["facet"] == facet) & (spans["direction"] == direction)
                               & (spans["starts_in_opening"] == opening_flag)]
                if subset.empty:
                    continue
                lines += [f"## {display_names.get(facet, facet)}: {direction}, "
                          f"{'starting in the opening' if opening_flag else 'starting after the opening'}", ""]
                for row in subset.itertuples():
                    lines.append(f"{row.rank}. `{row.document_id}` ({row.language}), token {row.start_token}, "
                                 f"z {row.window_score_z:.2f}")
                    if isinstance(row.text, str):
                        lines.append(f"   > {row.context_before}**[[{row.text}]]**{row.context_after}".replace("\n", " "))
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# Positive control
# ----------------------------------------------------------------------------

def build_positive_control_shift_table(score_tables, token_ids_by_document=None, minimum_changed_tokens=20):
    """One row per rewrite whose own facet has a vector: whole-document shift (rewrite minus ITS OWN source, in
    z units of the language's score spread), changed-span shifts if token ids are available, and the shift on
    every other facet (specificity). Rewrites below `minimum_changed_tokens` changed tokens (by token diff) are kept
    in the table with `below_minimum_changed_tokens` True and excluded from summaries by the caller."""
    data = score_tables.data
    documents = data.documents
    whole = statistics_module.whole_document_rewrite_shifts(score_tables.raw_scores, documents)
    # rewrites of a facet that was not scored (a pilot extracts one facet, rewrites may cover more) have no vector: left out
    rewrite_ids = [rewrite_id for rewrite_id in whole.index if documents.loc[rewrite_id, "facet"] is not None
                   and not pd.isna(documents.loc[rewrite_id, "facet"]) and documents.loc[rewrite_id, "facet"] in whole.columns]
    scale = pd.DataFrame({vector: score_tables.languages.loc[whole.index].map(
        score_tables.language_statistics[f"{vector}__sd"].astype(float)) for vector in whole.columns})
    whole_z = whole / scale
    table = pd.DataFrame({"source_document_id": documents.loc[rewrite_ids, "source_document_id"],
                          "facet": documents.loc[rewrite_ids, "facet"],
                          "rewrite_direction": documents.loc[rewrite_ids, "rewrite_direction"],
                          "language": documents.loc[rewrite_ids, "language"]})
    table["whole_document_shift"] = [whole_z.loc[i, facet] for i, facet in zip(table.index, table["facet"])]
    for vector in whole_z.columns:
        table[f"whole_document_shift_on__{vector}"] = whole_z.loc[table.index, vector]
    table["changed_tokens"] = np.nan
    table["below_minimum_changed_tokens"] = False
    table["changed_span_shift"] = np.nan
    table["changed_span_shift_minus_unchanged"] = np.nan
    if token_ids_by_document is not None and data.per_token:
        span = statistics_module.changed_span_rewrite_shifts(
            data.per_token, documents.loc[rewrite_ids], token_ids_by_document, list(data.vector_names),
            score_tables.baseline_means_overall, minimum_changed_tokens=0)
        for rewrite_id in table.index:
            if rewrite_id not in span.index:
                continue
            facet = table.loc[rewrite_id, "facet"]
            sd = float(score_tables.language_statistics.loc[table.loc[rewrite_id, "language"], f"{facet}__sd"])
            table.loc[rewrite_id, "changed_tokens"] = span.loc[rewrite_id, "changed_tokens"]
            enough = span.loc[rewrite_id, "changed_tokens"] >= minimum_changed_tokens
            table.loc[rewrite_id, "below_minimum_changed_tokens"] = not enough
            if enough:
                table.loc[rewrite_id, "changed_span_shift"] = span.loc[rewrite_id, f"{facet}__changed_minus_replaced"] / sd
                table.loc[rewrite_id, "changed_span_shift_minus_unchanged"] = (
                    span.loc[rewrite_id, f"{facet}__changed_minus_unchanged"] / sd)
    return table


def build_positive_control_summaries(shift_table):
    """{'whole_document': ..., 'changed_span': ...}: for each level a summary per facet and direction (clustered
    standard errors) and the contrasts against the facet-neutral rewrites."""
    results = {}
    for level, column in (("whole_document", "whole_document_shift"), ("changed_span", "changed_span_shift")):
        usable = shift_table[~shift_table["below_minimum_changed_tokens"]] if level == "changed_span" else shift_table
        usable = usable.dropna(subset=[column]).reset_index().rename(columns={"index": "rewrite_document_id"})
        if usable.empty:
            continue
        results[level] = {
            "summary": statistics_module.summarise_positive_control(usable, ["facet", "rewrite_direction"], column),
            "contrasts": statistics_module.contrast_against_neutral(usable, column)}
    return results


def build_positive_control_specificity(shift_table):
    """Mean whole-document shift on every facet, by the facet and direction of the rewrite: the shift should be on
    the rewrite's own facet."""
    columns = [column for column in shift_table.columns if column.startswith("whole_document_shift_on__")]
    return shift_table.groupby(["facet", "rewrite_direction"])[columns].mean().rename(
        columns=lambda name: name.replace("whole_document_shift_on__", "mean_shift_on_"))


def build_positive_control_by_language(shift_table):
    return (shift_table.groupby(["facet", "rewrite_direction", "language"])["whole_document_shift"]
            .agg(n_rewrites="count", mean_shift="mean").reset_index())


def evaluate_positive_control_pass_criteria(summaries, criteria):
    """Pass or fail per facet and level, against criteria written down before the run (never changed by the code).

    Pass when, for the facet: toward and away rewrites move the expected way in at least
    `minimum_fraction_moving_expected_way` of cases with the clustered interval excluding zero on the expected side,
    and (if required) the toward-minus-neutral and away-minus-neutral contrasts have intervals excluding zero on the
    expected side."""
    rows = []
    for level, tables in summaries.items():
        summary, contrasts = tables["summary"], tables["contrasts"]
        for facet in sorted(summary["facet"].unique()):
            facet_summary = summary[summary["facet"] == facet].set_index("rewrite_direction")
            checks = {}
            for direction, sign in ((DIRECTION_TOWARD, 1), (DIRECTION_AWAY, -1)):
                if direction not in facet_summary.index:
                    checks[f"{direction}_present"] = False
                    continue
                row = facet_summary.loc[direction]
                fraction = row["rewrites_moving_expected_way"] / max(1, row["n_rewrites"])
                checks[f"{direction}_fraction_moving_expected_way"] = fraction >= criteria["minimum_fraction_moving_expected_way"]
                excludes = row["ci_low"] > 0 if sign > 0 else row["ci_high"] < 0
                checks[f"{direction}_interval_excludes_zero_on_expected_side"] = bool(excludes)
                if criteria.get("require_contrast_against_neutral_interval_to_exclude_zero", True):
                    contrast = contrasts[(contrasts["facet"] == facet)
                                         & (contrasts["contrast"] == f"{direction}_minus_{DIRECTION_NEUTRAL}")]
                    if contrast.empty or pd.isna(contrast["difference"].iloc[0]):
                        checks[f"{direction}_contrast_against_neutral_excludes_zero"] = False
                    else:
                        contrast_row = contrast.iloc[0]
                        checks[f"{direction}_contrast_against_neutral_excludes_zero"] = bool(
                            contrast_row["ci_low"] > 0 if sign > 0 else contrast_row["ci_high"] < 0)
            rows.append({"facet": facet, "level": level, "passed": all(checks.values()), **checks})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_positive_control(summary, path, level_title, display_names, y_label):
    """Per facet, mean shift with clustered 95% interval for toward, away and facet-neutral rewrites."""
    plt = _pyplot()
    facets = list(summary["facet"].unique())
    figure, axes = plt.subplots(1, len(facets), figsize=(max(7.5, 3.2 * len(facets) + 1.5), 4.2), squeeze=False, sharey=True)
    colors = {DIRECTION_TOWARD: COLOR_TOWARD, DIRECTION_AWAY: COLOR_AWAY, DIRECTION_NEUTRAL: COLOR_NEUTRAL}
    labels = {DIRECTION_TOWARD: "toward facet", DIRECTION_AWAY: "away from facet", DIRECTION_NEUTRAL: "same-size, facet untouched"}
    for axis, facet in zip(axes[0], facets):
        for position, direction in enumerate((DIRECTION_TOWARD, DIRECTION_AWAY, DIRECTION_NEUTRAL)):
            row = summary[(summary["facet"] == facet) & (summary["rewrite_direction"] == direction)]
            if row.empty:
                continue
            row = row.iloc[0]
            axis.bar(position, row["mean_shift"], color=colors[direction], label=labels[direction])
            if not pd.isna(row["ci_low"]):
                axis.errorbar(position, row["mean_shift"], yerr=[[row["mean_shift"] - row["ci_low"]],
                                                                  [row["ci_high"] - row["mean_shift"]]],
                              color="black", capsize=3)
            axis.annotate(f"{row['mean_shift']:+.2f}", (position, row["mean_shift"]), ha="center",
                          va="bottom" if row["mean_shift"] >= 0 else "top", fontsize=8)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks([])
        axis.set_title(display_names.get(facet, facet), fontsize=10)
    axes[0][0].set_ylabel(y_label.replace(", ", ",\n"), fontsize=9)
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.suptitle("\n".join(textwrap.wrap(level_title, 70)), fontsize=10)
    figure.tight_layout(rect=(0, 0.08, 1, 0.9))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_score_distributions(score_tables, facets, path, display_names):
    plt = _pyplot()
    figure, axes = plt.subplots(1, len(facets), figsize=(3.0 * len(facets) + 1, 3.0), squeeze=False, sharey=True)
    for axis, facet in zip(axes[0], facets):
        axis.hist(score_tables.z_scores.loc[score_tables.corpus_ids, facet].dropna(), bins=30, color=COLOR_POINT)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(display_names.get(facet, facet), fontsize=10)
        axis.set_xlabel("Score (SDs within language)")
    axes[0][0].set_ylabel("Documents")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_scores_by_spec_variable(table, facet, variable, path, display_names):
    """Mean z score with 95% interval per value of one spec variable; values with too few documents are grey and
    hollow and not ranked."""
    plt = _pyplot()
    subset = table[(table["facet"] == facet) & (table["variable"] == variable)].copy()
    if subset.empty:
        return
    subset = subset.iloc[::-1].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(7.5, 0.38 * len(subset) + 1.4))
    for position, row in subset.iterrows():
        color = COLOR_TOO_FEW if row["too_few_documents"] else COLOR_POINT
        axis.errorbar(row["mean"], position, xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]]
                      if not pd.isna(row["ci_low"]) else None, fmt="o", color=color,
                      mfc="white" if row["too_few_documents"] else color, capsize=2)
    axis.set_yticks(range(len(subset)))
    axis.set_yticklabels([f"{str(group)[:48]} (n={n})" for group, n in zip(subset["group"], subset["n"])], fontsize=8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean score (SDs within language), 95% interval; hollow grey = too few documents to rank")
    axis.set_title(f"{display_names.get(facet, facet)} by {variable.replace('_', ' ')}", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_scores_by_position(band_table, path, display_names):
    plt = _pyplot()
    facets = list(band_table["facet"].unique())
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    for facet in facets:
        subset = band_table[(band_table["facet"] == facet) & band_table["position"].str.startswith("band_")]
        axis.errorbar(range(len(subset)), subset["mean"], yerr=[subset["mean"] - subset["ci_low"],
                                                               subset["ci_high"] - subset["mean"]],
                      marker="o", capsize=2, label=display_names.get(facet, facet))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Position in the document (fifths)")
    axis.set_ylabel("Mean score (SDs within language)")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
