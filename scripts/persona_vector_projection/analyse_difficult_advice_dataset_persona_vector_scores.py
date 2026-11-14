"""Analyses the persona vector projections of the difficult-advice answers (a scoring run built by
build_difficult_advice_dataset_scoring_run_folder.py) against the judge verdicts and the planted-flaw controls. Local only: no AI API,
no pod, no .env.

    python3 scripts/persona_vector_projection/analyse_difficult_advice_dataset_persona_vector_scores.py \
        --scoring-run-folder data/persona_vector_projection_runs/persona_vector_difficult_advice_dataset_scoring_gemma_4_12b_it_<ts> \
        --answers-folder data/persona_vector_projection_runs/difficult_advice_dataset_answers_for_persona_vector_scoring_<ts>

Sign convention (moralizing delivery vector set): the positive pole is plain delivery and the negative pole is moralizing, so a MORE
NEGATIVE projection means MORE moralizing. Everywhere below, "projection difference" is arm minus reference, a positive difference
means plainer delivery, and a judge `delivery` win for the arm is expected to go with a positive difference.

What is scored per answer: the mean projection over the ANSWER tokens only (the user turn is context, not scored), at the chosen
layer from the vector manifest. Two windows are reported for every vector and format: `after_first_64_tokens` (the primary one, as
in the moralizing delivery positive-control analysis, which skips the opening tokens) and `all_answer_tokens`. The primary
specification is the first vector, the first format and the after-64-tokens window.

Analyses (main fifty prompts; the pilot prompts have no judgements and no shared answers, so they appear only in the per-answer table
and in a small own-answer rewrite table):
  1  planted-flaw controls: each flawed edit minus its source Opus stage-2 answer, per flaw
  2  stage-2 answers by writer and the plain baseline, minus the Opus stage-2 answer; the Opus stage-2 repeat as noise floor
  3  stage-3a rewrites (each model rewriting the Opus answer) and stage-3b rewrites (Opus rewriting each model's answer) minus the
     answer that was rewritten
  4  agreement with judges: for each judged comparison (arm role only), does the projection difference predict the delivery outcome
     and the difference in the share of readings flagging `too_much`
  5  length: projection against log word count, and the same analyses 2-4 adjusted for log word count
  6  agreement between the two document formats
Intervals are percentile bootstraps over prompts (all answers of a prompt are resampled together); tests are Wilcoxon signed-rank
across prompts. Resumable: per-specification results are checkpointed in <output folder>/per_specification_checkpoints; rerun with
--output-folder to continue.
"""

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection.load_persona_projection_scoring_outputs import read_projection_shards  # noqa: E402
from persona_vector_projection.persona_projection_analysis_statistics import safe_wilcoxon_p_value  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
MAIN_DATASET_GROUP = "main_fifty_prompts"
PILOT_DATASET_GROUP = "pilot_prompts"
OPUS_WRITER = "opus55"
DOCUMENT_FORMATS = ["assistant_turn_with_real_user_message", "assistant_turn"]
WINDOW_AFTER_OPENING = "after_first_64_tokens"
WINDOW_ALL_TOKENS = "all_answer_tokens"
WINDOWS = [WINDOW_AFTER_OPENING, WINDOW_ALL_TOKENS]
OUTCOME_TO_SIGNED_SCORE = {"win": 1.0, "tie": 0.0, "loss": -1.0}
AGREEMENT_STATISTIC_NAMES = [
    "delivery_spearman", "delivery_auc_win_vs_loss", "delivery_partial_spearman_given_length", "delivery_auc_length_adjusted",
    "too_much_spearman", "too_much_auc_fewer_vs_more", "too_much_partial_spearman_given_length", "too_much_auc_length_adjusted"]
TASKS_IN_POOLED_JUDGED_COMPARISONS = ["stage2", "stage3a", "stage3b"]


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


# ----------------------------------------------------------------------------------------------------------------------------
# Small statistics
# ----------------------------------------------------------------------------------------------------------------------------

def rank_correlation(first_values, second_values):
    first_values, second_values = np.asarray(first_values, float), np.asarray(second_values, float)
    if len(first_values) < 3 or np.ptp(first_values) == 0 or np.ptp(second_values) == 0:
        return np.nan
    return float(np.corrcoef(stats.rankdata(first_values), stats.rankdata(second_values))[0, 1])


def residualise_on_covariate(values, covariate):
    """Residuals of an ordinary least squares fit of values on covariate with an intercept."""
    values, covariate = np.asarray(values, float), np.asarray(covariate, float)
    if len(values) < 3 or np.ptp(covariate) == 0:
        return values - values.mean() if len(values) else values
    design = np.column_stack([np.ones(len(covariate)), covariate])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def partial_rank_correlation(first_values, second_values, covariate):
    """Spearman correlation of first and second with the (rank of the) covariate regressed out of both ranks."""
    if len(first_values) < 4:
        return np.nan
    covariate_ranks = stats.rankdata(covariate)
    return rank_correlation(residualise_on_covariate(stats.rankdata(first_values), covariate_ranks),
                            residualise_on_covariate(stats.rankdata(second_values), covariate_ranks))


def area_under_curve(values, positive_mask, negative_mask):
    """Probability that a random positive row has a higher value than a random negative row (ties count half)."""
    values = np.asarray(values, float)
    positive_mask, negative_mask = np.asarray(positive_mask, bool), np.asarray(negative_mask, bool)
    positive_count, negative_count = int(positive_mask.sum()), int(negative_mask.sum())
    if positive_count == 0 or negative_count == 0:
        return np.nan
    in_either_class = positive_mask | negative_mask  # rank among the two classes only; other rows (ties) must not inflate ranks
    ranks = stats.rankdata(values[in_either_class])
    rank_sum_of_positives = ranks[positive_mask[in_either_class]].sum()
    return float((rank_sum_of_positives - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def cluster_bootstrap_interval(row_prompt_ids, statistic, draws, seed, confidence=0.95):
    """Percentile interval of a vector-valued statistic(row_indices) with whole prompts resampled with replacement.
    Returns (low, high) arrays, NaN where the statistic is undefined in more than half of the draws."""
    row_prompt_ids = np.asarray(row_prompt_ids)
    random_generator = np.random.default_rng(seed)
    prompts = np.unique(row_prompt_ids)
    rows_by_prompt = [np.flatnonzero(row_prompt_ids == prompt) for prompt in prompts]
    results = []
    for _ in range(draws):
        chosen_prompts = random_generator.integers(0, len(prompts), len(prompts))
        results.append(np.atleast_1d(statistic(np.concatenate([rows_by_prompt[chosen] for chosen in chosen_prompts]))))
    results = np.array(results, dtype=float)
    tail = (1 - confidence) / 2 * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(results, tail, axis=0), np.nanpercentile(results, 100 - tail, axis=0)
    undefined = np.mean(np.isnan(results), axis=0) > 0.5
    low[undefined], high[undefined] = np.nan, np.nan
    return np.atleast_1d(low), np.atleast_1d(high)


def adjusted_group_means(differences, log_length_ratios, group_codes, group_count):
    """Group intercepts and the common slope of differences = group intercept + slope * log length ratio (group fixed effects).
    Returns (adjusted mean per group code, slope); groups absent from the rows give NaN."""
    present_codes = np.unique(group_codes)
    design = np.column_stack([(group_codes == code).astype(float) for code in present_codes] + [log_length_ratios])
    coefficients = np.linalg.lstsq(design, differences, rcond=None)[0]
    adjusted = np.full(group_count, np.nan)
    adjusted[present_codes] = coefficients[:-1]
    return adjusted, float(coefficients[-1])


def interval_dictionary(estimate, low, high):
    return {"estimate": _finite_or_none(estimate), "ci_low": _finite_or_none(low), "ci_high": _finite_or_none(high)}


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


# ----------------------------------------------------------------------------------------------------------------------------
# Loading and tables
# ----------------------------------------------------------------------------------------------------------------------------

def newest_file(folder, pattern):
    matches = sorted(Path(folder).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {folder}")
    return matches[-1]


def load_answers_table(answers_folder):
    answers = pd.DataFrame(read_jsonl(newest_file(answers_folder, "difficult_advice_answers_*.jsonl")))
    if answers["document_id"].duplicated().any():
        raise ValueError("Duplicate document ids in the answers file")
    answers = answers.set_index("document_id", drop=False)
    answers["log_word_count"] = np.log(answers["word_count"].astype(float))
    return answers


def short_vector_name(vector_name):
    return vector_name.removeprefix("moralizing_delivery_").removesuffix("_vector")


def specification_key(document_format, vector_name, window):
    return f"{document_format}__{short_vector_name(vector_name)}__{window}"


def load_scores_by_specification(scoring_run_folder, answers, vector_names, document_formats, chosen_layer):
    """Per-answer mean projection for every (format, vector, window), keyed by specification. Matched to answers by document id."""
    scores, token_counts, shard_vector_names = {}, None, None
    for document_format in document_formats:
        shards = read_projection_shards(scoring_run_folder, document_format, chosen_layer)
        if set(shards["document_ids"]) != set(answers.index):
            raise ValueError(f"{document_format}: scored document ids differ from the answers file")
        if shards["layer_index"] != chosen_layer:
            raise ValueError("Layer mismatch")
        for vector_name in vector_names:
            if vector_name not in shards["vector_names"]:
                raise ValueError(f"{vector_name} not scored; scored vectors: {shards['vector_names']}")
            scores[specification_key(document_format, vector_name, WINDOW_AFTER_OPENING)] = shards["mean_after_opening"][vector_name]
            scores[specification_key(document_format, vector_name, WINDOW_ALL_TOKENS)] = shards["mean_overall"][vector_name]
        token_counts = shards["token_counts"]
    return {key: series.reindex(answers.index).astype(float) for key, series in scores.items()}, token_counts.reindex(answers.index)


def build_pair_table(answers):
    """One row per (group, prompt): arm answer, reference answer (the answer it is compared with), log word count ratio.
    Pairs whose texts are identical (a rewriter returned its input unchanged) are dropped and listed separately."""
    main = answers[answers["dataset_group"] == MAIN_DATASET_GROUP]
    opus_stage_two = {row.prompt_gid: row.document_id for row in main.itertuples()
                      if row.stage == "stage_2" and row.writer_model == OPUS_WRITER}
    rows = []

    def add(group, section, arm_row, reference_id):
        rows.append({"group": group, "section": section, "prompt_gid": arm_row.prompt_gid, "arm_document_id": arm_row.document_id,
                     "reference_document_id": reference_id})

    for row in answers.itertuples():
        in_main = row.dataset_group == MAIN_DATASET_GROUP
        if in_main and row.stage == "planted_flaw_control":
            add(f"1_planted_flaw_{row.planted_flaw}", "planted_flaw", row, row.edited_from_document_id)
        elif in_main and row.stage == "baseline":
            add("2_plain_baseline_vs_opus55_stage_2", "stage_2", row, opus_stage_two[row.prompt_gid])
        elif in_main and row.stage == "stage_2" and row.writer_model != OPUS_WRITER:
            add(f"2_stage_2_{row.writer_model}_vs_opus55_stage_2", "stage_2", row, opus_stage_two[row.prompt_gid])
        elif in_main and row.stage == "stage_2_repeat":
            add("2_stage_2_repeat_opus55_vs_opus55_stage_2", "noise_floor", row, opus_stage_two[row.prompt_gid])
        elif in_main and row.stage == "stage_3a_rewrite":
            add(f"3a_{row.writer_model}_rewrite_of_opus55_answer_vs_that_answer", "stage_3a", row, row.rewritten_source_document_id)
        elif in_main and row.stage == "stage_3b_rewrite":
            add(f"3b_opus55_rewrite_of_{row.rewritten_source_model}_answer_vs_that_answer", "stage_3b", row,
                row.rewritten_source_document_id)
        elif row.dataset_group == PILOT_DATASET_GROUP and row.stage == "stage_3a_rewrite":
            add(f"pilot_{row.writer_model}_rewrite_of_own_answer_vs_that_answer", "pilot", row, row.rewritten_source_document_id)
    pairs = pd.DataFrame(rows)
    missing = set(pairs["reference_document_id"]) - set(answers.index)
    if missing:
        raise ValueError(f"Reference answers not in the answers file: {sorted(missing)[:3]}")
    identical = (pairs["arm_document_id"].map(answers["answer_text"]).values
                 == pairs["reference_document_id"].map(answers["answer_text"]).values)
    dropped = pairs[identical].copy()
    pairs = pairs[~identical].reset_index(drop=True)
    pairs["log_length_ratio"] = (pairs["arm_document_id"].map(answers["log_word_count"]).values
                                 - pairs["reference_document_id"].map(answers["log_word_count"]).values)
    return pairs, dropped


def delivery_signed_score(outcomes_by_judge_family):
    scores = [OUTCOME_TO_SIGNED_SCORE[family_outcomes["delivery"]["outcome"]]
              for family_outcomes in outcomes_by_judge_family.values()
              if family_outcomes.get("delivery", {}).get("outcome") in OUTCOME_TO_SIGNED_SCORE]
    return float(np.mean(scores)) if scores else np.nan


def build_comparison_table(verdict_records, answers):
    """One row per judged comparison from the ARM's side (reference-role rows are the same comparisons seen from the other side).
    delivery_signed_score: mean over judge families of win +1, tie 0, loss -1 (the arm's delivery against its counterpart).
    too_much_difference: arm share of readings flagging too_much minus the counterpart's share (positive = arm worse)."""
    rows = []
    for record in verdict_records:
        if record["document_id"] not in answers.index or answers.at[record["document_id"], "dataset_group"] != MAIN_DATASET_GROUP:
            continue
        stage = answers.at[record["document_id"], "stage"]
        for comparison in record["comparisons"]:
            if comparison["role"] != "arm":
                continue
            task = comparison["task"]
            if task == "controls":
                task = "controls_planted_flaw" if stage == "planted_flaw_control" else "controls_other"
            rows.append({
                "task": task, "comparison_label": comparison["comparison_label"], "prompt_gid": record["prompt_gid"],
                "arm_document_id": record["document_id"], "reference_document_id": comparison["counterpart_document_id"],
                "delivery_signed_score": delivery_signed_score(comparison["outcomes_by_judge_family"]),
                "too_much_difference": (comparison["own_failure_check_share_of_readings"]["too_much"]
                                        - comparison["counterpart_failure_check_share_of_readings"]["too_much"]),
                "arm_too_much_share": comparison["own_failure_check_share_of_readings"]["too_much"],
                "counterpart_too_much_share": comparison["counterpart_failure_check_share_of_readings"]["too_much"],
                "judge_families": "+".join(sorted(comparison["outcomes_by_judge_family"]))})
    table = pd.DataFrame(rows)
    missing = set(table["reference_document_id"]) - set(answers.index)
    if missing:
        raise ValueError(f"Counterpart answers not in the answers file: {sorted(missing)[:3]}")
    table["log_length_ratio"] = (table["arm_document_id"].map(answers["log_word_count"]).values
                                 - table["reference_document_id"].map(answers["log_word_count"]).values)
    return table.dropna(subset=["delivery_signed_score"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------------
# One specification (format, vector, window)
# ----------------------------------------------------------------------------------------------------------------------------

def paired_group_statistics(pairs, differences, draws, seed):
    """Analyses 1-3 and 5 on paired differences (arm minus reference projection): raw and length-adjusted shift per group."""
    group_names = sorted(pairs["group"].unique())
    group_codes = pairs["group"].map({name: code for code, name in enumerate(group_names)}).values
    prompt_ids = pairs["prompt_gid"].values
    log_ratios = pairs["log_length_ratio"].values
    adjusted_estimates, common_slope = adjusted_group_means(differences, log_ratios, group_codes, len(group_names))

    def adjusted_statistic(indices):
        adjusted, slope = adjusted_group_means(differences[indices], log_ratios[indices], group_codes[indices], len(group_names))
        return np.append(adjusted, slope)

    adjusted_low, adjusted_high = cluster_bootstrap_interval(prompt_ids, adjusted_statistic, draws, seed)
    results = {}
    for code, name in enumerate(group_names):
        rows = np.flatnonzero(group_codes == code)
        group_differences = differences[rows]
        low, high = cluster_bootstrap_interval(prompt_ids[rows], lambda idx: group_differences[idx].mean(), draws, seed + code + 1)
        count_below, count_above = int(np.sum(group_differences < 0)), int(np.sum(group_differences > 0))
        results[name] = {
            "number_of_prompts": len(rows), "mean_shift": interval_dictionary(group_differences.mean(), low[0], high[0]),
            "median_shift": float(np.median(group_differences)), "standard_deviation_of_shift": float(group_differences.std(ddof=1)) if len(rows) > 1 else None,
            "mean_absolute_shift": float(np.abs(group_differences).mean()),
            "prompts_more_moralizing": count_below, "prompts_plainer": count_above,
            "wilcoxon_p_value": _finite_or_none(safe_wilcoxon_p_value(group_differences)),
            "mean_log_length_ratio": float(log_ratios[rows].mean()),
            "length_adjusted_shift": interval_dictionary(adjusted_estimates[code], adjusted_low[code], adjusted_high[code])}
    results["_common_slope_per_log_word_ratio"] = interval_dictionary(common_slope, adjusted_low[-1], adjusted_high[-1])
    # Shift-versus-length association inside groups (group means removed), pooled.
    centred_differences = differences - pd.Series(differences).groupby(group_codes).transform("mean").values
    centred_ratios = log_ratios - pd.Series(log_ratios).groupby(group_codes).transform("mean").values
    low, high = cluster_bootstrap_interval(prompt_ids, lambda idx: rank_correlation(centred_differences[idx], centred_ratios[idx]), draws, seed + 500)
    results["_within_group_rank_correlation_of_shift_with_log_length_ratio"] = interval_dictionary(
        rank_correlation(centred_differences, centred_ratios), low[0], high[0])
    return results


def planted_flaw_contrast(pairs, differences, draws, seed):
    """too_much shift minus the mean shift of the other three flaws (unpaired: different prompts per flaw), and the pooled other flaws."""
    flaw_rows = pairs["group"].str.startswith("1_planted_flaw_").values
    too_much_rows = (pairs["group"] == "1_planted_flaw_too_much").values
    if not too_much_rows.any():
        return None
    others = np.flatnonzero(flaw_rows & ~too_much_rows)
    selected = np.flatnonzero(flaw_rows)
    prompt_ids = pairs["prompt_gid"].values[selected]
    selected_is_too_much = too_much_rows[selected]
    selected_differences = differences[selected]

    def contrast(idx):
        flagged = selected_is_too_much[idx]
        if flagged.all() or not flagged.any():
            return np.nan
        return selected_differences[idx][flagged].mean() - selected_differences[idx][~flagged].mean()

    low, high = cluster_bootstrap_interval(prompt_ids, contrast, draws, seed)
    return {"too_much_minus_other_flaws_mean_shift": interval_dictionary(contrast(np.arange(len(selected))), low[0], high[0]),
            "other_flaws_pooled_number": len(others), "other_flaws_pooled_mean_shift": float(differences[others].mean()),
            "other_flaws_pooled_wilcoxon_p_value": _finite_or_none(safe_wilcoxon_p_value(differences[others])),
            "mann_whitney_p_value_too_much_vs_other_flaws": _finite_or_none(
                stats.mannwhitneyu(differences[too_much_rows], differences[others]).pvalue)}


def level_statistics(answers, scores, draws, seed):
    """Mean projection per stage and writer over the main prompts, with intervals over prompts."""
    main = answers[answers["dataset_group"] == MAIN_DATASET_GROUP]
    results = {}
    for (stage, writer, source), members in main.groupby(["stage", "writer_model", main["rewritten_source_model"].fillna("none")]):
        if stage == "planted_flaw_control":
            continue
        values = scores[members.index].values
        low, high = cluster_bootstrap_interval(members["prompt_gid"].values, lambda idx: values[idx].mean(), draws, seed)
        results[f"{stage}__writer_{writer}__rewriting_{source}"] = {
            "number_of_answers": len(members), **interval_dictionary(values.mean(), low[0], high[0]),
            "standard_deviation_between_answers": float(values.std(ddof=1))}
    return results


def agreement_statistics_for_rows(comparisons, projection_differences, indices):
    projection_difference = projection_differences[indices]
    delivery = comparisons["delivery_signed_score"].values[indices]
    too_much = comparisons["too_much_difference"].values[indices]
    log_ratio = comparisons["log_length_ratio"].values[indices]
    adjusted_difference = residualise_on_covariate(projection_difference, log_ratio)
    return np.array([
        rank_correlation(projection_difference, delivery),
        area_under_curve(projection_difference, delivery > 0, delivery < 0),
        partial_rank_correlation(projection_difference, delivery, log_ratio),
        area_under_curve(adjusted_difference, delivery > 0, delivery < 0),
        rank_correlation(projection_difference, too_much),
        area_under_curve(projection_difference, too_much < 0, too_much > 0),
        partial_rank_correlation(projection_difference, too_much, log_ratio),
        area_under_curve(adjusted_difference, too_much < 0, too_much > 0)])


def agreement_with_judges(comparisons, scores, draws, seed):
    """Analysis 4 (with its length-adjusted versions from analysis 5), per task and pooled."""
    projection_differences = (comparisons["arm_document_id"].map(scores).values - comparisons["reference_document_id"].map(scores).values)
    subsets = {task: comparisons["task"].values == task for task in sorted(comparisons["task"].unique())}
    subsets["pooled_stage2_stage3a_stage3b"] = comparisons["task"].isin(TASKS_IN_POOLED_JUDGED_COMPARISONS).values
    subsets["pooled_all_tasks_including_controls"] = np.ones(len(comparisons), dtype=bool)
    results = {}
    for offset, (name, mask) in enumerate(subsets.items()):
        subset = comparisons[mask].reset_index(drop=True)
        subset_differences = projection_differences[mask]
        point = agreement_statistics_for_rows(subset, subset_differences, np.arange(len(subset)))
        low, high = cluster_bootstrap_interval(subset["prompt_gid"].values,
                                               lambda idx: agreement_statistics_for_rows(subset, subset_differences, idx), draws, seed + offset)
        delivery = subset["delivery_signed_score"].values
        too_much = subset["too_much_difference"].values
        results[name] = {
            "number_of_comparisons": len(subset), "delivery_wins_ties_losses": [int((delivery > 0).sum()), int((delivery == 0).sum()), int((delivery < 0).sum())],
            "too_much_arm_worse_equal_better": [int((too_much > 0).sum()), int((too_much == 0).sum()), int((too_much < 0).sum())],
            **{statistic: interval_dictionary(point[position], low[position], high[position]) for position, statistic in enumerate(AGREEMENT_STATISTIC_NAMES)}}
    return results


def length_of_levels(answers, scores, draws, seed):
    """Analysis 5, levels: rank correlation of the per-answer projection with log word count over the main answers, pooled and with each
    prompt's mean removed."""
    main = answers[answers["dataset_group"] == MAIN_DATASET_GROUP]
    values, log_counts = scores[main.index].values, main["log_word_count"].values
    prompt_ids = main["prompt_gid"].values
    centred_values = values - pd.Series(values).groupby(prompt_ids).transform("mean").values
    centred_counts = log_counts - pd.Series(log_counts).groupby(prompt_ids).transform("mean").values
    statistic = lambda idx: np.array([rank_correlation(values[idx], log_counts[idx]), rank_correlation(centred_values[idx], centred_counts[idx])])  # noqa: E731
    low, high = cluster_bootstrap_interval(prompt_ids, statistic, draws, seed)
    point = statistic(np.arange(len(main)))
    return {"pooled_rank_correlation_with_log_word_count": interval_dictionary(point[0], low[0], high[0]),
            "within_prompt_rank_correlation_with_log_word_count": interval_dictionary(point[1], low[1], high[1]),
            "number_of_answers": len(main)}


def analyse_one_specification(answers, pairs, comparisons, scores, draws, seed):
    differences = pairs["arm_document_id"].map(scores).values - pairs["reference_document_id"].map(scores).values
    main_pairs = ~pairs["section"].eq("pilot").values
    results = {
        "paired_shifts_main_prompts": paired_group_statistics(pairs[main_pairs].reset_index(drop=True), differences[main_pairs], draws, seed),
        "planted_flaw_contrast": planted_flaw_contrast(pairs[main_pairs].reset_index(drop=True), differences[main_pairs], draws, seed + 7000),
        "paired_shifts_pilot_prompts_own_answer_rewrites": {},
        "mean_projection_by_stage_and_writer": level_statistics(answers, scores, draws, seed + 8000),
        "agreement_with_judges": agreement_with_judges(comparisons, scores, draws, seed + 9000),
        "length_of_per_answer_projection": length_of_levels(answers, scores, draws, seed + 9500)}
    pilot = ~main_pairs
    for group in sorted(pairs[pilot]["group"].unique()):
        values = differences[pilot & (pairs["group"] == group).values]
        results["paired_shifts_pilot_prompts_own_answer_rewrites"][group] = {
            "number_of_prompts": len(values), "mean_shift": float(values.mean()), "prompts_more_moralizing": int((values < 0).sum())}
    return results


def agreement_between_formats(answers, pairs, comparisons, scores_by_specification, vector_names, document_formats, draws, seed):
    """Analysis 6: rank correlation between the two formats of the per-answer projections, of the paired differences, and of the
    comparison projection differences, for every vector and window."""
    first_format, second_format = document_formats[:2]
    main = answers[answers["dataset_group"] == MAIN_DATASET_GROUP]
    results = {}
    for vector_name in vector_names:
        for window in WINDOWS:
            first = scores_by_specification[specification_key(first_format, vector_name, window)]
            second = scores_by_specification[specification_key(second_format, vector_name, window)]
            pair_differences = [pairs["arm_document_id"].map(s).values - pairs["reference_document_id"].map(s).values for s in (first, second)]
            comparison_differences = [comparisons["arm_document_id"].map(s).values - comparisons["reference_document_id"].map(s).values for s in (first, second)]
            level_values = [first[main.index].values, second[main.index].values]
            entry = {}
            for name, (x, y), prompts in (("per_answer_levels", level_values, main["prompt_gid"].values),
                                          ("paired_differences_all_groups", pair_differences, pairs["prompt_gid"].values),
                                          ("judged_comparison_projection_differences", comparison_differences, comparisons["prompt_gid"].values)):
                low, high = cluster_bootstrap_interval(prompts, lambda idx: rank_correlation(x[idx], y[idx]), draws, seed)
                entry[name] = {"rank_correlation": interval_dictionary(rank_correlation(x, y), low[0], high[0]),
                               "pearson_correlation": float(np.corrcoef(x, y)[0, 1])}
            results[f"{short_vector_name(vector_name)}__{window}"] = entry
    return results


# ----------------------------------------------------------------------------------------------------------------------------
# Markdown, figures, saving
# ----------------------------------------------------------------------------------------------------------------------------

def fmt(value, digits=3, signed=True):
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def fmt_interval(entry, digits=3, signed=True):
    return f"{fmt(entry['estimate'], digits, signed)} [{fmt(entry['ci_low'], digits, signed)}, {fmt(entry['ci_high'], digits, signed)}]"


def markdown_for_paired_shifts(shifts):
    lines = ["| group | prompts | mean shift [95% CI] | median | more moralizing / plainer | Wilcoxon p | mean log length ratio | length-adjusted shift [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for name, entry in shifts.items():
        if name.startswith("_"):
            continue
        lines.append(f"| {name} | {entry['number_of_prompts']} | {fmt_interval(entry['mean_shift'])} | {fmt(entry['median_shift'])} | "
                     f"{entry['prompts_more_moralizing']} / {entry['prompts_plainer']} | {fmt(entry['wilcoxon_p_value'], 3, False)} | "
                     f"{fmt(entry['mean_log_length_ratio'], 2)} | {fmt_interval(entry['length_adjusted_shift'])} |")
    lines.append("")
    lines.append(f"Common slope of shift on log word count ratio: {fmt_interval(shifts['_common_slope_per_log_word_ratio'])}. "
                 f"Within-group rank correlation of shift with log length ratio: "
                 f"{fmt_interval(shifts['_within_group_rank_correlation_of_shift_with_log_length_ratio'], 2)}.")
    return lines


def markdown_for_agreement(agreement):
    lines = ["| comparisons | n | delivery wins / ties / losses | rho(difference, delivery) | AUC win vs loss | partial rho (length) | AUC (length-adjusted) |",
             "|---|---|---|---|---|---|---|"]
    for name, entry in agreement.items():
        lines.append(f"| {name} | {entry['number_of_comparisons']} | {' / '.join(map(str, entry['delivery_wins_ties_losses']))} | "
                     f"{fmt_interval(entry['delivery_spearman'], 2)} | {fmt_interval(entry['delivery_auc_win_vs_loss'], 2, False)} | "
                     f"{fmt_interval(entry['delivery_partial_spearman_given_length'], 2)} | {fmt_interval(entry['delivery_auc_length_adjusted'], 2, False)} |")
    lines += ["", "| comparisons | n | too_much arm worse / equal / better | rho(difference, too_much difference) | AUC fewer vs more too_much | partial rho (length) | AUC (length-adjusted) |",
              "|---|---|---|---|---|---|---|"]
    for name, entry in agreement.items():
        lines.append(f"| {name} | {entry['number_of_comparisons']} | {' / '.join(map(str, entry['too_much_arm_worse_equal_better']))} | "
                     f"{fmt_interval(entry['too_much_spearman'], 2)} | {fmt_interval(entry['too_much_auc_fewer_vs_more'], 2, False)} | "
                     f"{fmt_interval(entry['too_much_partial_spearman_given_length'], 2)} | {fmt_interval(entry['too_much_auc_length_adjusted'], 2, False)} |")
    return lines


def build_markdown(all_results, primary_key, specification_keys, format_agreement, context, timestamp):
    primary = all_results[primary_key]
    lines = [f"# Difficult-advice answers scored with the moralizing delivery vectors ({timestamp})", "",
             f"Scoring run: `{context['scoring_run_folder']}`; answers: `{context['answers_folder']}`; layer {context['chosen_layer']}.", "",
             "Sign: positive pole = plain delivery, negative pole = moralizing, so a more negative projection is more moralizing; a shift is "
             "arm minus reference, a positive shift is plainer. Per-answer score = mean projection over the answer tokens only. Primary "
             f"specification: `{primary_key}` (tokens after the first 64 of the answer, as in the positive-control analysis). Intervals: "
             f"bootstrap over prompts ({context['bootstrap_draws']} draws); tests: Wilcoxon signed-rank across prompts. Pilot prompts "
             "are excluded from analyses 1-6 (no judgements, no shared answers); their own-answer rewrite shifts are in the JSON.", "",
             f"Data notes: {context['data_notes']}", ""]
    lines += ["## Analyses 1-3 and 5: paired shifts, raw and length-adjusted (primary specification)", ""]
    lines += markdown_for_paired_shifts(primary["paired_shifts_main_prompts"])
    contrast = primary["planted_flaw_contrast"]
    if contrast:
        lines += ["", f"too_much minus the other three flaws: {fmt_interval(contrast['too_much_minus_other_flaws_mean_shift'])} "
                      f"(Mann-Whitney p {fmt(contrast['mann_whitney_p_value_too_much_vs_other_flaws'], 3, False)}); the other three flaws pooled "
                      f"(n {contrast['other_flaws_pooled_number']}): mean {fmt(contrast['other_flaws_pooled_mean_shift'])}, Wilcoxon p "
                      f"{fmt(contrast['other_flaws_pooled_wilcoxon_p_value'], 3, False)}."]
    lines += ["", "## Mean projection by stage and writer (primary specification)", "", "| stage / writer / source | answers | mean [95% CI] | SD between answers |", "|---|---|---|---|"]
    for name, entry in primary["mean_projection_by_stage_and_writer"].items():
        lines.append(f"| {name} | {entry['number_of_answers']} | {fmt_interval(entry, 3, False)} | {fmt(entry['standard_deviation_between_answers'], 3, False)} |")
    lines += ["", "## Analysis 4: agreement with judges (primary specification)", ""]
    lines += markdown_for_agreement(primary["agreement_with_judges"])
    lines += ["", "## Analysis 5: length of per-answer projection (primary specification)", "",
              f"Rank correlation with log word count, pooled: {fmt_interval(primary['length_of_per_answer_projection']['pooled_rank_correlation_with_log_word_count'], 2)}; "
              f"with each prompt's mean removed: {fmt_interval(primary['length_of_per_answer_projection']['within_prompt_rank_correlation_with_log_word_count'], 2)}."]
    lines += ["", "## All specifications side by side", "",
              "| specification | too_much shift | other flaws pooled shift | stage-2 repeat shift | stage-2 gpt-sol | stage-2 argon | stage-2 sonnet55 | plain baseline | pooled rho(delivery) | pooled AUC delivery | pooled rho(too_much) | pooled AUC too_much |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in specification_keys:
        shifts = all_results[key]["paired_shifts_main_prompts"]
        agreement = all_results[key]["agreement_with_judges"]["pooled_stage2_stage3a_stage3b"]
        contrast_entry = all_results[key]["planted_flaw_contrast"]
        cell = lambda name: fmt_interval(shifts[name]["mean_shift"], 2) if name in shifts else "n/a"  # noqa: E731
        lines.append(f"| {key} | {cell('1_planted_flaw_too_much')} | {fmt(contrast_entry['other_flaws_pooled_mean_shift'], 2) if contrast_entry else 'n/a'} | "
                     f"{cell('2_stage_2_repeat_opus55_vs_opus55_stage_2')} | {cell('2_stage_2_gpt-sol_vs_opus55_stage_2')} | "
                     f"{cell('2_stage_2_argon_vs_opus55_stage_2')} | {cell('2_stage_2_sonnet55_vs_opus55_stage_2')} | "
                     f"{cell('2_plain_baseline_vs_opus55_stage_2')} | {fmt_interval(agreement['delivery_spearman'], 2)} | "
                     f"{fmt_interval(agreement['delivery_auc_win_vs_loss'], 2, False)} | {fmt_interval(agreement['too_much_spearman'], 2)} | "
                     f"{fmt_interval(agreement['too_much_auc_fewer_vs_more'], 2, False)} |")
    lines += ["", "## Analysis 6: agreement between the two document formats (rank correlations)", "",
              "| vector and window | per-answer levels | paired differences | judged-comparison differences |", "|---|---|---|---|"]
    for name, entry in format_agreement.items():
        lines.append(f"| {name} | {fmt_interval(entry['per_answer_levels']['rank_correlation'], 2)} | "
                     f"{fmt_interval(entry['paired_differences_all_groups']['rank_correlation'], 2)} | "
                     f"{fmt_interval(entry['judged_comparison_projection_differences']['rank_correlation'], 2)} |")
    lines += ["", "Full numbers for every specification: the JSON file in this folder."]
    return "\n".join(lines) + "\n"


def make_figures(all_results, primary_key, pairs, comparisons, scores_by_specification, vector_names, primary_format, output_folder, timestamp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    shifts = {name: entry for name, entry in all_results[primary_key]["paired_shifts_main_prompts"].items() if not name.startswith("_")}
    names = list(shifts)[::-1]
    figure, axis = plt.subplots(figsize=(9, 0.38 * len(names) + 1.5))
    for position, name in enumerate(names):
        entry = shifts[name]
        raw, adjusted = entry["mean_shift"], entry["length_adjusted_shift"]
        axis.errorbar(raw["estimate"], position + 0.12, xerr=[[raw["estimate"] - raw["ci_low"]], [raw["ci_high"] - raw["estimate"]]], fmt="o", color="#1f4e79", capsize=2)
        axis.errorbar(adjusted["estimate"], position - 0.12, xerr=[[adjusted["estimate"] - adjusted["ci_low"]], [adjusted["ci_high"] - adjusted["estimate"]]], fmt="s", color="#c27c0e", capsize=2, markersize=4)
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=7)
    axis.set_xlabel("mean projection shift, arm minus reference (negative = more moralizing); 95% bootstrap interval over prompts")
    axis.set_title("Paired shifts: raw (blue circles) and length-adjusted (orange squares)", fontsize=10)
    figure.tight_layout()
    paths.append(output_folder / f"figure_1_paired_projection_shifts_by_group_{timestamp}.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    scores = scores_by_specification[primary_key]
    differences = comparisons["arm_document_id"].map(scores).values - comparisons["reference_document_id"].map(scores).values
    figure, axis = plt.subplots(figsize=(7, 5))
    for task, color in zip(sorted(comparisons["task"].unique()), ["#1f4e79", "#c27c0e", "#2a7f4f", "#b03a48", "#6b4c9a"]):
        mask = (comparisons["task"] == task).values
        jitter = np.random.default_rng(0).normal(0, 0.012, mask.sum())
        axis.scatter(comparisons["too_much_difference"].values[mask] + jitter, differences[mask], s=14, alpha=0.6, color=color, label=task)
    pooled = agreement_with_judges_label(all_results[primary_key])
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.set_xlabel("judge too_much share: arm minus counterpart (positive = arm flagged more)")
    axis.set_ylabel("projection difference, arm minus counterpart (negative = more moralizing)")
    axis.set_title(f"Projection difference against judge too_much difference\n{pooled}", fontsize=9)
    axis.legend(fontsize=7)
    figure.tight_layout()
    paths.append(output_folder / f"figure_2_projection_difference_against_judge_too_much_difference_{timestamp}.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, statistic, title in ((axes[0], "delivery_spearman", "rho with judge delivery outcome (expect positive)"),
                                   (axes[1], "too_much_spearman", "rho with judge too_much difference (expect negative)")):
        tasks = list(all_results[primary_key]["agreement_with_judges"])[::-1]
        for vector_position, vector_name in enumerate(vector_names):
            key = specification_key(primary_format, vector_name, WINDOW_AFTER_OPENING)
            for task_position, task in enumerate(tasks):
                entry = all_results[key]["agreement_with_judges"][task][statistic]
                if entry["estimate"] is None:
                    continue
                y = task_position + (vector_position - 1) * 0.22
                axis.errorbar(entry["estimate"], y, xerr=[[entry["estimate"] - entry["ci_low"]], [entry["ci_high"] - entry["estimate"]]], fmt="o", markersize=4, capsize=2,
                              color=["#1f4e79", "#c27c0e", "#2a7f4f"][vector_position % 3], label=short_vector_name(vector_name) if task_position == 0 else None)
        axis.axvline(0, color="grey", linewidth=0.8)
        axis.set_yticks(range(len(tasks)))
        axis.set_yticklabels(tasks, fontsize=7)
        axis.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    paths.append(output_folder / f"figure_3_agreement_with_judges_by_task_and_vector_{timestamp}.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)
    return paths


def agreement_with_judges_label(primary_results):
    pooled = primary_results["agreement_with_judges"]["pooled_stage2_stage3a_stage3b"]
    return (f"pooled stage 2/3a/3b: rho {fmt_interval(pooled['too_much_spearman'], 2)}, n {pooled['number_of_comparisons']}")


def sanitise_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitise_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitise_for_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


# ----------------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scoring-run-folder", required=True)
    parser.add_argument("--answers-folder", required=True)
    parser.add_argument("--vector-names", nargs="+", default=None,
                        help="Vectors to analyse, main one first. Default: every vector in the scoring vector manifest, in manifest order.")
    parser.add_argument("--document-formats", nargs="+", default=DOCUMENT_FORMATS, help="Primary format first; the first two are compared in analysis 6.")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output-folder", default=None, help="Resume into this folder (default: a new timestamped folder under <scoring run folder>/analysis).")
    return parser.parse_args(argv)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def main(argv=None):
    arguments = parse_arguments(argv)
    scoring_run_folder, answers_folder = resolve(arguments.scoring_run_folder), resolve(arguments.answers_folder)
    timestamp = timestamp_now()
    output_folder = (resolve(arguments.output_folder) if arguments.output_folder else
                     scoring_run_folder / "analysis" / f"difficult_advice_moralizing_delivery_analysis_{timestamp}")
    checkpoint_folder = output_folder / "per_specification_checkpoints"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(newest_file(scoring_run_folder / "scoring_pod_inputs", "scoring_vector_manifest_*.json").read_text())
    chosen_layer = manifest["chosen_layer"]
    vector_names = arguments.vector_names or list(manifest["vectors"])
    document_formats = arguments.document_formats
    answers = load_answers_table(answers_folder)
    verdict_records = read_jsonl(newest_file(answers_folder, "difficult_advice_judgement_verdicts_per_answer_*.jsonl"))
    scores_by_specification, token_counts = load_scores_by_specification(scoring_run_folder, answers, vector_names, document_formats, chosen_layer)
    pairs, dropped_pairs = build_pair_table(answers)
    comparisons = build_comparison_table(verdict_records, answers)
    data_notes = (f"{len(dropped_pairs)} pairs dropped because the rewriter returned its input unchanged "
                  f"({', '.join(sorted(dropped_pairs['arm_document_id']))}); {len(comparisons)} judged comparisons (arm role) with a delivery outcome.")
    print(data_notes)

    specification_keys = [specification_key(document_format, vector_name, window)
                          for document_format in document_formats for vector_name in vector_names for window in WINDOWS]
    primary_key = specification_keys[0]
    all_results = {}
    for index, key in enumerate(specification_keys):
        checkpoint_path = checkpoint_folder / f"results_for_{key}.json"
        settings = {"draws": arguments.bootstrap_draws, "seed": arguments.seed, "pairs": len(pairs), "comparisons": len(comparisons)}
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text())
            if saved["settings"] == settings:
                all_results[key] = saved["results"]
                print(f"[{index + 1}/{len(specification_keys)}] {key}: loaded from checkpoint")
                continue
        print(f"[{index + 1}/{len(specification_keys)}] {key}: analysing", flush=True)
        all_results[key] = sanitise_for_json(analyse_one_specification(answers, pairs, comparisons, scores_by_specification[key],
                                                                       arguments.bootstrap_draws, arguments.seed))
        checkpoint_path.write_text(json.dumps({"settings": settings, "results": all_results[key]}))

    format_agreement = {}
    if len(document_formats) >= 2:
        format_agreement = sanitise_for_json(agreement_between_formats(answers, pairs, comparisons, scores_by_specification, vector_names,
                                                                         document_formats, arguments.bootstrap_draws, arguments.seed + 100))
    context = {"scoring_run_folder": str(scoring_run_folder), "answers_folder": str(answers_folder), "chosen_layer": chosen_layer,
               "bootstrap_draws": arguments.bootstrap_draws, "data_notes": data_notes}
    markdown = build_markdown(all_results, primary_key, specification_keys, format_agreement, context, timestamp)
    (output_folder / f"difficult_advice_moralizing_delivery_analysis_tables_{timestamp}.md").write_text(markdown)
    (output_folder / f"difficult_advice_moralizing_delivery_analysis_all_numbers_{timestamp}.json").write_text(json.dumps(
        {"context": context, "primary_specification": primary_key, "specifications": specification_keys, "results_by_specification": all_results,
         "agreement_between_document_formats": format_agreement, "dropped_identical_text_pairs": dropped_pairs.to_dict("records"),
         "sign_convention": "positive pole plain delivery, negative pole moralizing: more negative projection = more moralizing"}, indent=1))
    per_answer = answers.drop(columns=["user_message", "answer_text"]).copy()
    per_answer["scored_token_count"] = token_counts
    for key, series in scores_by_specification.items():
        per_answer[f"projection__{key}"] = series
    per_answer.to_csv(output_folder / f"per_answer_projections_joined_with_metadata_{timestamp}.csv", index=False)
    primary_scores = scores_by_specification[primary_key]
    pairs.assign(projection_difference_primary_specification=pairs["arm_document_id"].map(primary_scores).values - pairs["reference_document_id"].map(primary_scores).values
                 ).to_csv(output_folder / f"paired_comparisons_with_primary_projection_difference_{timestamp}.csv", index=False)
    comparisons.assign(projection_difference_primary_specification=comparisons["arm_document_id"].map(primary_scores).values - comparisons["reference_document_id"].map(primary_scores).values
                       ).to_csv(output_folder / f"judged_comparisons_with_primary_projection_difference_{timestamp}.csv", index=False)
    figure_paths = make_figures(all_results, primary_key, pairs, comparisons, scores_by_specification, vector_names, document_formats[0], output_folder, timestamp)
    print(f"Wrote {output_folder}\n" + "\n".join(str(path) for path in figure_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
