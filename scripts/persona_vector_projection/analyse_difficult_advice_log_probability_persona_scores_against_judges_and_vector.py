"""Analyses the log-probability persona check of the difficult-advice answers (preachy rewordings and proportionate-welfare rewordings,
each against the generic system prompt, scored by Gemma 4 12B instruct) against the judge verdicts, the planted-flaw controls, and the
moralizing delivery persona vector scores of the same answers. Local only: no AI API, no pod, no .env.

    python3 scripts/persona_vector_projection/analyse_difficult_advice_log_probability_persona_scores_against_judges_and_vector.py \
        --log-probability-run-folder <audit/log_probability_persona_check_... run folder> \
        --answers-folder data/persona_vector_projection_runs/difficult_advice_dataset_answers_for_persona_vector_scoring_<ts> \
        --vector-scoring-run-folder data/persona_vector_projection_runs/persona_vector_difficult_advice_dataset_scoring_gemma_4_12b_it_<ts>

Per-token contrast (the primary score): mean over the preachy conditions of (log P(token | condition) - log P(token | generic)) minus the
same mean over the proportionate_welfare conditions. The generic term cancels, so it equals the difference of the two persona-mean
log probabilities. HIGHER = more preachy relative to proportionate welfare. The secondary score keeps the preachy minus generic term alone.
Per answer: mean after the first 64 answer tokens (primary window, as in the vector analysis) and over all answer tokens.

Sign convention against the vector: the vector projection is MORE NEGATIVE for MORE moralizing, so agreement shows as a NEGATIVE rank
correlation between the vector projection and the contrast (the per-token section flips the vector so that moralizing is positive and the
expected correlation is positive). Against the judges the expected direction is: an answer judged more `too_much` than its counterpart has
a HIGHER contrast; an answer that wins on delivery has a LOWER contrast. Statistics are the vector analysis's own functions
(analyse_difficult_advice_dataset_persona_vector_scores.analyse_one_specification), so shifts are arm minus reference with a positive shift
meaning MORE preachy here (the vector analysis's counter names "more moralizing / plainer" are therefore relabelled in this script's tables).

Outputs go to <log probability run folder>/difficult_advice_preachy_minus_proportionate_welfare_analysis_<timestamp>/.
The run takes about a minute plus the bootstrap draws; a rerun with --output-folder reuses nothing and overwrites that folder's files of
the same timestamp, so it is cheap to simply rerun.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
PERSONA_CHECK_PACKAGE_ROOT = REPOSITORY_ROOT / "animal-welfare-data-pipeline-log-probability-persona-check"

# The persona check package and the scripts folder each have their own top-level `shared` package. Load the persona check loader first
# (it imports its own `shared`), then drop that `shared` from the import cache so the scripts folder's `shared` is the one used afterwards.
sys.path.insert(0, str(PERSONA_CHECK_PACKAGE_ROOT))
from evals.log_probability_persona_check.analysis_run_loading import load_run  # noqa: E402

for module_name in [name for name in sys.modules if name == "shared" or name.startswith("shared.")]:
    del sys.modules[module_name]
sys.path[:] = [entry for entry in sys.path if entry not in (str(PERSONA_CHECK_PACKAGE_ROOT), str(SCRIPTS_DIRECTORY))]
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import analyse_difficult_advice_dataset_persona_vector_scores as vector_analysis  # noqa: E402
from persona_vector_projection.load_persona_projection_scoring_outputs import read_projection_shards  # noqa: E402
from persona_vector_projection.persona_projection_analysis_statistics import rolling_window_means  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

PREACHY_PERSONA = "preachy"
PROPORTIONATE_WELFARE_PERSONA = "proportionate_welfare"
EXPECTED_PREACHY_CONDITIONS = 15
EXPECTED_PROPORTIONATE_WELFARE_CONDITIONS = 5
OPENING_TOKENS_SKIPPED = 64
WINDOW_TOKENS = 32
CONTEXT_TOKENS_BEFORE_WINDOW = 60
WINDOWS_PER_LIST = 15
VECTOR_DOCUMENT_FORMAT = "assistant_turn_with_real_user_message"
VECTOR_NAME = "moralizing_delivery_passage_mean_vector"

SPECIFICATIONS = {
    "preachy_minus_proportionate_welfare__after_first_64_tokens": ("preachy_minus_proportionate_welfare", "after_first_64_tokens"),
    "preachy_minus_proportionate_welfare__all_answer_tokens": ("preachy_minus_proportionate_welfare", "all_answer_tokens"),
    "preachy_minus_generic__after_first_64_tokens": ("preachy_minus_generic", "after_first_64_tokens"),
}
PRIMARY_SPECIFICATION = "preachy_minus_proportionate_welfare__after_first_64_tokens"


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


# ----------------------------------------------------------------------------------------------------------------------------
# Per-token contrasts
# ----------------------------------------------------------------------------------------------------------------------------

def per_token_contrasts_of_document(document_scores):
    """(preachy minus proportionate welfare, preachy minus generic) per token, from a loader DocumentScores."""
    by_persona = document_scores.per_token_difference_by_persona
    preachy_minus_generic = np.asarray(by_persona[PREACHY_PERSONA], dtype=float)
    proportionate_welfare_minus_generic = np.asarray(by_persona[PROPORTIONATE_WELFARE_PERSONA], dtype=float)
    return preachy_minus_generic - proportionate_welfare_minus_generic, preachy_minus_generic


def mean_after_opening(values):
    values = np.asarray(values, dtype=float)
    return float(values[OPENING_TOKENS_SKIPPED:].mean()) if len(values) > OPENING_TOKENS_SKIPPED else np.nan


def load_log_probability_run(run_folder):
    """Condensed run plus per-answer arrays. Checks the condition counts (15 preachy, 5 proportionate welfare, generic)."""
    run = load_run(run_folder)
    conditions_by_persona = {}
    for condition in run.conditions:
        conditions_by_persona.setdefault(condition["persona_name"], []).append(condition["condition_id"])
    if len(conditions_by_persona.get(PREACHY_PERSONA, [])) != EXPECTED_PREACHY_CONDITIONS:
        raise ValueError(f"Expected {EXPECTED_PREACHY_CONDITIONS} preachy conditions, found {conditions_by_persona.get(PREACHY_PERSONA)}")
    if len(conditions_by_persona.get(PROPORTIONATE_WELFARE_PERSONA, [])) != EXPECTED_PROPORTIONATE_WELFARE_CONDITIONS:
        raise ValueError("Expected 5 proportionate_welfare conditions")
    if len(conditions_by_persona.get("generic", [])) != 1:
        raise ValueError("Expected exactly one generic condition")
    if set(conditions_by_persona) != {PREACHY_PERSONA, PROPORTIONATE_WELFARE_PERSONA, "generic"}:
        raise ValueError(f"Unexpected personas in the run: {sorted(conditions_by_persona)}")
    contrast_by_document, generic_contrast_by_document = {}, {}
    for document_id, document_scores in run.scores_by_document.items():
        contrast_by_document[document_id], generic_contrast_by_document[document_id] = per_token_contrasts_of_document(document_scores)
    return run, contrast_by_document, generic_contrast_by_document


# ----------------------------------------------------------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------------------------------------------------------

def best_window_per_document(per_token_by_document, highest, relative_to_document_mean=False):
    """One window per document (32 tokens, start at or after token 64): the highest or lowest window mean, or the window furthest
    above / below its document's post-opening mean. Returns a list of dicts sorted best first."""
    rows = []
    for document_id, values in per_token_by_document.items():
        values = np.asarray(values, dtype=float)
        if len(values) < OPENING_TOKENS_SKIPPED + WINDOW_TOKENS:
            continue
        window_means = rolling_window_means(values, WINDOW_TOKENS)[OPENING_TOKENS_SKIPPED:]
        document_mean = float(values[OPENING_TOKENS_SKIPPED:].mean())
        ranking_values = window_means - document_mean if relative_to_document_mean else window_means
        position = int(np.argmax(ranking_values) if highest else np.argmin(ranking_values))
        rows.append({"document_id": document_id, "start_token": position + OPENING_TOKENS_SKIPPED,
                     "window_mean": float(window_means[position]), "document_mean": document_mean,
                     "window_minus_document_mean": float(window_means[position] - document_mean),
                     "ranking_value": float(ranking_values[position])})
    rows.sort(key=lambda row: (-row["ranking_value"] if highest else row["ranking_value"], row["document_id"]))
    return rows


def all_window_means_after_opening(per_token_by_document):
    arrays = [rolling_window_means(values, WINDOW_TOKENS)[OPENING_TOKENS_SKIPPED:]
              for values in per_token_by_document.values() if len(values) >= OPENING_TOKENS_SKIPPED + WINDOW_TOKENS]
    return np.concatenate(arrays)


def windows_overlap(first_start, second_start):
    return abs(first_start - second_start) < WINDOW_TOKENS


def overlap_of_window_lists(first_rows, second_rows):
    shared_documents = {row["document_id"] for row in first_rows} & {row["document_id"] for row in second_rows}
    second_by_document = {row["document_id"]: row for row in second_rows}
    overlapping = [row["document_id"] for row in first_rows
                   if row["document_id"] in second_by_document
                   and windows_overlap(row["start_token"], second_by_document[row["document_id"]]["start_token"])]
    return {"documents_in_both_lists": len(shared_documents), "documents_with_overlapping_windows": len(overlapping),
            "overlapping_window_document_ids": overlapping, "shared_document_ids": sorted(shared_documents)}


# ----------------------------------------------------------------------------------------------------------------------------
# Intervals shown in the direction "more too_much / delivery loss ranks higher"
# ----------------------------------------------------------------------------------------------------------------------------

def flip_probability_interval(entry):
    """AUC of 'fewer too_much (or delivery win) ranks higher' -> AUC of 'more too_much (or delivery loss) ranks higher'."""
    if entry is None or entry.get("estimate") is None:
        return entry
    return {"estimate": 1 - entry["estimate"], "ci_low": None if entry["ci_high"] is None else 1 - entry["ci_high"],
            "ci_high": None if entry["ci_low"] is None else 1 - entry["ci_low"]}


# ----------------------------------------------------------------------------------------------------------------------------
# Agreement with the vector
# ----------------------------------------------------------------------------------------------------------------------------

def pearson_or_nan(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(first) < 3 or first.std() == 0 or second.std() == 0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def spearman_or_nan(first, second):
    return pearson_or_nan(stats.rankdata(first), stats.rankdata(second))


def per_token_agreement_for_document(vector_projection_per_token, contrast_per_token):
    """Within one answer. The vector is flipped (moralizing = positive) so the expected correlation with the contrast is positive.
    Token level and 32-token rolling-mean level (both over tokens after the first 64)."""
    moralizing = -np.asarray(vector_projection_per_token, dtype=float)
    contrast = np.asarray(contrast_per_token, dtype=float)
    result = {"pearson_all_tokens": pearson_or_nan(moralizing, contrast), "spearman_all_tokens": spearman_or_nan(moralizing, contrast),
              "pearson_after_first_64_tokens": pearson_or_nan(moralizing[OPENING_TOKENS_SKIPPED:], contrast[OPENING_TOKENS_SKIPPED:]),
              "spearman_after_first_64_tokens": spearman_or_nan(moralizing[OPENING_TOKENS_SKIPPED:], contrast[OPENING_TOKENS_SKIPPED:])}
    if len(moralizing) >= OPENING_TOKENS_SKIPPED + WINDOW_TOKENS:
        smoothed_vector = rolling_window_means(moralizing, WINDOW_TOKENS)[OPENING_TOKENS_SKIPPED:]
        smoothed_contrast = rolling_window_means(contrast, WINDOW_TOKENS)[OPENING_TOKENS_SKIPPED:]
        result["pearson_of_32_token_rolling_means"] = pearson_or_nan(smoothed_vector, smoothed_contrast)
        result["spearman_of_32_token_rolling_means"] = spearman_or_nan(smoothed_vector, smoothed_contrast)
    else:
        result["pearson_of_32_token_rolling_means"] = result["spearman_of_32_token_rolling_means"] = np.nan
    return result


def summarise_per_token_agreement(table):
    summary = {}
    for column in [name for name in table.columns if name.startswith(("pearson", "spearman"))]:
        values = table[column].dropna()
        summary[column] = {"documents": int(len(values)), "mean": float(values.mean()), "median": float(values.median()),
                           "fraction_positive": float((values > 0).mean()), "ten_percent_quantile": float(values.quantile(0.1)),
                           "ninety_percent_quantile": float(values.quantile(0.9))}
    return summary


# ----------------------------------------------------------------------------------------------------------------------------
# Tables for the markdown
# ----------------------------------------------------------------------------------------------------------------------------

fmt = vector_analysis.fmt
fmt_interval = vector_analysis.fmt_interval


def markdown_for_paired_shifts(shifts):
    lines = ["| group | prompts | mean shift [95% CI] | median | less preachy / more preachy | Wilcoxon p | mean log length ratio | length-adjusted shift [95% CI] |",
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
    lines = ["Direction expected: an answer judged more `too_much` than its counterpart has a HIGHER score (positive rho; AUC above 0.5 in the "
             "columns headed 'more ranks higher'); an answer that wins on delivery has a LOWER score (negative rho). Length adjustment: score "
             "difference residualised on the log word count ratio (AUC) or partial rank correlation.", "",
             "| comparisons | n | too_much arm worse / equal / better | rho(difference, too_much difference) | AUC more too_much ranks higher | partial rho (length) | AUC more ranks higher (length-adjusted) |",
             "|---|---|---|---|---|---|---|"]
    for name, entry in agreement.items():
        lines.append(f"| {name} | {entry['number_of_comparisons']} | {' / '.join(map(str, entry['too_much_arm_worse_equal_better']))} | "
                     f"{fmt_interval(entry['too_much_spearman'], 2)} | {fmt_interval(flip_probability_interval(entry['too_much_auc_fewer_vs_more']), 2, False)} | "
                     f"{fmt_interval(entry['too_much_partial_spearman_given_length'], 2)} | {fmt_interval(flip_probability_interval(entry['too_much_auc_length_adjusted']), 2, False)} |")
    lines += ["", "| comparisons | n | delivery wins / ties / losses | rho(difference, delivery) | AUC delivery loss ranks higher | partial rho (length) | AUC loss ranks higher (length-adjusted) |",
              "|---|---|---|---|---|---|---|"]
    for name, entry in agreement.items():
        lines.append(f"| {name} | {entry['number_of_comparisons']} | {' / '.join(map(str, entry['delivery_wins_ties_losses']))} | "
                     f"{fmt_interval(entry['delivery_spearman'], 2)} | {fmt_interval(flip_probability_interval(entry['delivery_auc_win_vs_loss']), 2, False)} | "
                     f"{fmt_interval(entry['delivery_partial_spearman_given_length'], 2)} | {fmt_interval(flip_probability_interval(entry['delivery_auc_length_adjusted']), 2, False)} |")
    return lines


def markdown_for_levels(levels):
    lines = ["| stage / writer / source | answers | mean [95% CI] | SD between answers |", "|---|---|---|---|"]
    for name, entry in levels.items():
        lines.append(f"| {name} | {entry['number_of_answers']} | {fmt_interval(entry, 3)} | {fmt(entry['standard_deviation_between_answers'], 3, False)} |")
    return lines


def markdown_for_spans(span_records, title):
    lines = [f"## {title}", ""]
    for record in span_records:
        lines += [f"### #{record['rank']}: contrast {record['window_mean_preachy_minus_proportionate_welfare']:+.3f} per token "
                  f"(document mean after token 64 {record['document_mean_after_first_64_tokens']:+.3f}; preachy minus generic "
                  f"{record['window_mean_preachy_minus_generic']:+.3f})", "",
                  f"`{record['document_id']}`; stage {record['stage']}, writer {record['writer_model']}, planted flaw {record['planted_flaw']}; "
                  f"window tokens {record['start_token']}-{record['start_token'] + WINDOW_TOKENS - 1}. Vector projection over the window "
                  f"{record['vector_window_mean_projection']:.1f} (document mean {record['vector_document_mean_projection']:.1f}).", "",
                  f"- context: {json.dumps(record['context_before_text'][-300:], ensure_ascii=False)}",
                  f"- window: {json.dumps(record['window_text'], ensure_ascii=False)}", ""]
    return lines


# ----------------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log-probability-run-folder", required=True)
    parser.add_argument("--answers-folder", required=True)
    parser.add_argument("--vector-scoring-run-folder", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output-folder", default=None)
    return parser.parse_args(argv)


def build_span_records(rows, direction, run, answers, contrast_by_document, generic_by_document, vector_per_token, vector_all_windows):
    records = []
    for rank, row in enumerate(rows[:WINDOWS_PER_LIST], start=1):
        document_id, start = row["document_id"], row["start_token"]
        scores = run.scores_by_document[document_id]
        strings = scores.token_strings
        answer = answers.loc[document_id]
        vector_values = vector_per_token[document_id]
        vector_window_mean = float(vector_values[start:start + WINDOW_TOKENS].mean())
        records.append({
            "list": direction, "rank": rank, "document_id": document_id, "prompt_gid": answer["prompt_gid"], "stage": answer["stage"],
            "writer_model": answer["writer_model"], "rewritten_source_model": answer.get("rewritten_source_model"),
            "planted_flaw": answer.get("planted_flaw"), "dataset_group": answer["dataset_group"],
            "start_token": start, "window_tokens": WINDOW_TOKENS, "number_of_answer_tokens": len(strings),
            "window_mean_preachy_minus_proportionate_welfare": row["window_mean"],
            "document_mean_after_first_64_tokens": row["document_mean"],
            "window_mean_preachy_minus_generic": float(generic_by_document[document_id][start:start + WINDOW_TOKENS].mean()),
            "vector_window_mean_projection": vector_window_mean,
            "vector_document_mean_projection": float(vector_values[OPENING_TOKENS_SKIPPED:].mean()),
            "vector_window_percentile_among_all_windows_low_is_moralizing": float((vector_all_windows < vector_window_mean).mean() * 100),
            "context_before_text": "".join(strings[max(0, start - CONTEXT_TOKENS_BEFORE_WINDOW):start]),
            "window_text": "".join(strings[start:start + WINDOW_TOKENS])})
    return records


def main(argv=None):
    arguments = parse_arguments(argv)
    run_folder = Path(arguments.log_probability_run_folder).resolve()
    answers_folder, vector_run_folder = Path(arguments.answers_folder).resolve(), Path(arguments.vector_scoring_run_folder).resolve()
    timestamp = timestamp_now()
    output_folder = (Path(arguments.output_folder) if arguments.output_folder else
                     run_folder / f"difficult_advice_preachy_minus_proportionate_welfare_analysis_{timestamp}")
    output_folder.mkdir(parents=True, exist_ok=True)

    answers = vector_analysis.load_answers_table(answers_folder)
    verdict_records = read_jsonl(vector_analysis.newest_file(answers_folder, "difficult_advice_judgement_verdicts_per_answer_*.jsonl"))
    run, contrast_by_document, generic_by_document = load_log_probability_run(run_folder)
    if set(run.scores_by_document) != set(answers.index):
        raise ValueError("Scored document ids differ from the answers file")
    print(f"Loaded {len(run.scores_by_document)} scored answers; model {run.setup.model_label}")

    # ---- per-answer scores
    per_answer = answers.drop(columns=["user_message", "answer_text"]).copy()
    per_answer["scored_token_count"] = pd.Series({d: len(v) for d, v in contrast_by_document.items()})
    if (per_answer["scored_token_count"] <= OPENING_TOKENS_SKIPPED).any():
        raise ValueError("Some answers have no tokens after the first 64")
    scores_by_specification = {}
    sources = {"preachy_minus_proportionate_welfare": contrast_by_document, "preachy_minus_generic": generic_by_document}
    for contrast_name, per_token in sources.items():
        per_answer[f"{contrast_name}__after_first_64_tokens"] = pd.Series({d: mean_after_opening(v) for d, v in per_token.items()})
        per_answer[f"{contrast_name}__all_answer_tokens"] = pd.Series({d: float(np.mean(v)) for d, v in per_token.items()})
    for key, (contrast_name, window) in SPECIFICATIONS.items():
        scores_by_specification[key] = per_answer[f"{contrast_name}__{window}"].reindex(answers.index).astype(float)

    # ---- judge agreement, planted flaws, writers, noise floor (the vector analysis's own functions)
    pairs, dropped_pairs = vector_analysis.build_pair_table(answers)
    comparisons = vector_analysis.build_comparison_table(verdict_records, answers)
    print(f"{len(dropped_pairs)} pairs dropped (rewriter returned its input unchanged); {len(comparisons)} judged comparisons")
    results_by_specification = {}
    for index, (key, scores) in enumerate(scores_by_specification.items()):
        print(f"[{index + 1}/{len(scores_by_specification)}] {key}: bootstrap analysis", flush=True)
        results_by_specification[key] = vector_analysis.sanitise_for_json(
            vector_analysis.analyse_one_specification(answers, pairs, comparisons, scores, arguments.bootstrap_draws, arguments.seed + index))

    # ---- vector scores and token alignment
    manifest = json.loads(vector_analysis.newest_file(vector_run_folder / "scoring_pod_inputs", "scoring_vector_manifest_*.json").read_text())
    chosen_layer = manifest["chosen_layer"]
    shards = read_projection_shards(vector_run_folder, VECTOR_DOCUMENT_FORMAT, chosen_layer)
    vector_per_token = {document_id: values[:, shards["vector_names"].index(VECTOR_NAME)] for document_id, values in shards["per_token"].items()}
    vector_after_opening = shards["mean_after_opening"][VECTOR_NAME].reindex(answers.index).astype(float)
    token_alignment_mismatches = [d for d in answers.index
                                  if not np.array_equal(np.asarray(shards["token_ids"][d]), np.asarray(run.scores_by_document[d].token_ids))]
    token_alignment = {"documents_compared": len(answers), "documents_with_identical_token_ids": len(answers) - len(token_alignment_mismatches),
                       "mismatching_document_ids": token_alignment_mismatches[:20]}
    print(f"Token ids identical in {token_alignment['documents_with_identical_token_ids']} of {len(answers)} answers")
    aligned_documents = [d for d in answers.index if d not in set(token_alignment_mismatches)]

    primary_scores = scores_by_specification[PRIMARY_SPECIFICATION]
    main_mask = (answers["dataset_group"] == vector_analysis.MAIN_DATASET_GROUP).values
    prompt_ids = answers["prompt_gid"].values
    rng_statistic = lambda x, y: (lambda idx: vector_analysis.rank_correlation(x[idx], y[idx]))  # noqa: E731
    vector_values_array, contrast_values_array = vector_after_opening.values, primary_scores.values
    agreement_with_vector = {}
    for name, mask in (("all_714_answers", np.ones(len(answers), bool)), ("main_fifty_prompts_only", main_mask)):
        x, y, prompts = vector_values_array[mask], contrast_values_array[mask], prompt_ids[mask]
        low, high = vector_analysis.cluster_bootstrap_interval(prompts, rng_statistic(x, y), arguments.bootstrap_draws, arguments.seed + 50)
        centred_x = x - pd.Series(x).groupby(prompts).transform("mean").values
        centred_y = y - pd.Series(y).groupby(prompts).transform("mean").values
        low_c, high_c = vector_analysis.cluster_bootstrap_interval(prompts, rng_statistic(centred_x, centred_y), arguments.bootstrap_draws, arguments.seed + 51)
        log_length = answers["log_word_count"].values[mask]
        agreement_with_vector[name] = {
            "number_of_answers": int(mask.sum()),
            "spearman_vector_projection_vs_preachy_minus_proportionate_welfare": vector_analysis.interval_dictionary(vector_analysis.rank_correlation(x, y), low[0], high[0]),
            "spearman_with_prompt_means_removed": vector_analysis.interval_dictionary(vector_analysis.rank_correlation(centred_x, centred_y), low_c[0], high_c[0]),
            "partial_spearman_given_log_word_count": vector_analysis.partial_rank_correlation(x, y, log_length),
            "spearman_vector_vs_log_word_count": vector_analysis.rank_correlation(x, log_length),
            "spearman_contrast_vs_log_word_count": vector_analysis.rank_correlation(y, log_length)}
    pair_vector_differences = pairs["arm_document_id"].map(vector_after_opening).values - pairs["reference_document_id"].map(vector_after_opening).values
    pair_contrast_differences = pairs["arm_document_id"].map(primary_scores).values - pairs["reference_document_id"].map(primary_scores).values
    pilot_pairs = pairs["section"].eq("pilot").values
    low, high = vector_analysis.cluster_bootstrap_interval(pairs["prompt_gid"].values[~pilot_pairs], rng_statistic(pair_vector_differences[~pilot_pairs], pair_contrast_differences[~pilot_pairs]), arguments.bootstrap_draws, arguments.seed + 52)
    agreement_with_vector["paired_differences_main_prompts"] = {
        "number_of_pairs": int((~pilot_pairs).sum()),
        "spearman_vector_difference_vs_contrast_difference": vector_analysis.interval_dictionary(
            vector_analysis.rank_correlation(pair_vector_differences[~pilot_pairs], pair_contrast_differences[~pilot_pairs]), low[0], high[0])}
    comparison_vector_differences = comparisons["arm_document_id"].map(vector_after_opening).values - comparisons["reference_document_id"].map(vector_after_opening).values
    comparison_contrast_differences = comparisons["arm_document_id"].map(primary_scores).values - comparisons["reference_document_id"].map(primary_scores).values
    low, high = vector_analysis.cluster_bootstrap_interval(comparisons["prompt_gid"].values, rng_statistic(comparison_vector_differences, comparison_contrast_differences), arguments.bootstrap_draws, arguments.seed + 53)
    agreement_with_vector["judged_comparison_differences"] = {
        "number_of_comparisons": len(comparisons),
        "spearman_vector_difference_vs_contrast_difference": vector_analysis.interval_dictionary(
            vector_analysis.rank_correlation(comparison_vector_differences, comparison_contrast_differences), low[0], high[0])}

    # ---- within-answer per-token agreement (every aligned answer, not a sample: the cost is a few seconds)
    per_token_rows = []
    for document_id in aligned_documents:
        row = per_token_agreement_for_document(vector_per_token[document_id], contrast_by_document[document_id])
        row["document_id"] = document_id
        per_token_rows.append(row)
    per_token_table = pd.DataFrame(per_token_rows).set_index("document_id")
    per_token_summary = summarise_per_token_agreement(per_token_table)

    # ---- windows: this method's lists, and the overlap with the vector's most moralizing windows
    vector_all_windows = all_window_means_after_opening(vector_per_token)
    contrast_all_windows = all_window_means_after_opening(contrast_by_document)
    highest_windows = best_window_per_document(contrast_by_document, highest=True)
    lowest_windows = best_window_per_document(contrast_by_document, highest=False)
    vector_most_moralizing = best_window_per_document(vector_per_token, highest=False)
    highest_relative = best_window_per_document(contrast_by_document, highest=True, relative_to_document_mean=True)
    vector_most_moralizing_relative = best_window_per_document(vector_per_token, highest=False, relative_to_document_mean=True)
    span_records = (build_span_records(highest_windows, "highest_most_preachy_relative_to_proportionate_welfare", run, answers, contrast_by_document,
                                       generic_by_document, vector_per_token, vector_all_windows)
                    + build_span_records(lowest_windows, "lowest_least_preachy_relative_to_proportionate_welfare", run, answers, contrast_by_document,
                                         generic_by_document, vector_per_token, vector_all_windows))
    top_vector_windows = vector_most_moralizing[:WINDOWS_PER_LIST]
    window_overlap = {
        "absolute_window_means": overlap_of_window_lists(highest_windows[:WINDOWS_PER_LIST], top_vector_windows),
        "windows_furthest_from_own_document_mean": overlap_of_window_lists(highest_relative[:WINDOWS_PER_LIST], vector_most_moralizing_relative[:WINDOWS_PER_LIST]),
        "contrast_percentile_of_the_vectors_top_windows": [float((contrast_all_windows < np.asarray(contrast_by_document[row["document_id"]][row["start_token"]:row["start_token"] + WINDOW_TOKENS]).mean()).mean() * 100)
                                                           for row in top_vector_windows],
        "vector_percentile_of_this_methods_top_windows_low_is_moralizing": [record["vector_window_percentile_among_all_windows_low_is_moralizing"]
                                                                            for record in span_records if record["list"].startswith("highest")],
        "vector_top_windows_document_ids": [row["document_id"] for row in top_vector_windows],
        "contrast_top_windows_document_ids": [row["document_id"] for row in highest_windows[:WINDOWS_PER_LIST]]}
    window_overlap["median_contrast_percentile_of_the_vectors_top_windows"] = float(np.median(window_overlap["contrast_percentile_of_the_vectors_top_windows"]))
    window_overlap["median_vector_percentile_of_this_methods_top_windows"] = float(np.median(window_overlap["vector_percentile_of_this_methods_top_windows_low_is_moralizing"]))
    stage_counts = {"highest_list_by_stage": pd.Series([r["stage"] for r in span_records if r["list"].startswith("highest")]).value_counts().to_dict(),
                    "lowest_list_by_stage": pd.Series([r["stage"] for r in span_records if r["list"].startswith("lowest")]).value_counts().to_dict()}

    # ---- write
    primary = results_by_specification[PRIMARY_SPECIFICATION]
    tag = f"preachy_minus_proportionate_welfare_{timestamp}"
    per_answer.assign(vector_projection_after_first_64_tokens=vector_after_opening).to_csv(
        output_folder / f"per_answer_log_probability_contrasts_joined_with_metadata_and_vector_projection_{timestamp}.csv", index=False)
    comparisons.assign(preachy_minus_proportionate_welfare_difference=comparison_contrast_differences,
                       vector_projection_difference=comparison_vector_differences).to_csv(
        output_folder / f"judged_comparisons_with_contrast_and_vector_differences_{timestamp}.csv", index=False)
    pairs.assign(preachy_minus_proportionate_welfare_difference=pair_contrast_differences, vector_projection_difference=pair_vector_differences).to_csv(
        output_folder / f"paired_comparisons_with_contrast_and_vector_differences_{timestamp}.csv", index=False)
    per_token_table.to_csv(output_folder / f"per_answer_within_answer_token_agreement_with_vector_{timestamp}.csv")
    with open(output_folder / f"extreme_32_token_windows_{tag}.jsonl", "w") as handle:
        for record in span_records:
            handle.write(json.dumps(vector_analysis.sanitise_for_json(record), ensure_ascii=False) + "\n")
    all_numbers = {
        "context": {"log_probability_run_folder": str(run_folder), "answers_folder": str(answers_folder), "vector_scoring_run_folder": str(vector_run_folder),
                    "model": run.setup.model_label, "vector_layer": chosen_layer, "vector_name": VECTOR_NAME, "bootstrap_draws": arguments.bootstrap_draws,
                    "primary_specification": PRIMARY_SPECIFICATION, "number_of_answers": len(answers), "number_of_judged_comparisons": len(comparisons),
                    "dropped_identical_text_pairs": dropped_pairs["arm_document_id"].tolist(),
                    "sign": "contrast higher = more preachy relative to proportionate welfare; vector more negative = more moralizing"},
        "results_by_specification": results_by_specification,
        "agreement_with_vector": vector_analysis.sanitise_for_json(agreement_with_vector),
        "token_alignment_between_the_two_runs": token_alignment,
        "within_answer_per_token_agreement_summary": vector_analysis.sanitise_for_json(per_token_summary),
        "window_overlap_with_vector": vector_analysis.sanitise_for_json(window_overlap),
        "extreme_window_stage_counts": stage_counts}
    (output_folder / f"difficult_advice_{tag.replace('preachy_minus_proportionate_welfare', 'preachy_minus_proportionate_welfare_all_numbers')}.json").write_text(
        json.dumps(all_numbers, indent=1, ensure_ascii=False))

    lines = [f"# Difficult-advice answers: log-probability preachy minus proportionate welfare ({timestamp})", "",
             f"Run: `{run_folder}`; model {run.setup.model_label}. 15 preachy and 5 proportionate_welfare rewordings against generic. Score = mean per "
             f"token of (mean over preachy conditions) minus (mean over proportionate-welfare conditions) of log P(token | condition) - log P(token | generic), "
             f"over answer tokens after the first 64; higher = more preachy relative to proportionate welfare. Shifts are arm minus reference "
             f"(positive = more preachy; the counter columns are 'less preachy / more preachy'). Intervals: bootstrap over prompts "
             f"({arguments.bootstrap_draws} draws). {len(dropped_pairs)} pairs dropped because the rewriter returned its input unchanged "
             f"({', '.join(dropped_pairs['arm_document_id'])}); {len(comparisons)} judged comparisons.", "",
             "## Judge agreement (primary specification)", ""]
    lines += markdown_for_agreement(primary["agreement_with_judges"])
    lines += ["", "## Paired shifts: planted flaws, stage-2 writers against Opus stage 2, noise floor, rewrites (primary specification)", ""]
    lines += markdown_for_paired_shifts(primary["paired_shifts_main_prompts"])
    contrast = primary["planted_flaw_contrast"]
    if contrast:
        lines += ["", f"too_much minus the other three flaws: {fmt_interval(contrast['too_much_minus_other_flaws_mean_shift'])} (Mann-Whitney p "
                      f"{fmt(contrast['mann_whitney_p_value_too_much_vs_other_flaws'], 3, False)}); other three flaws pooled (n {contrast['other_flaws_pooled_number']}): "
                      f"mean {fmt(contrast['other_flaws_pooled_mean_shift'])}, Wilcoxon p {fmt(contrast['other_flaws_pooled_wilcoxon_p_value'], 3, False)}."]
    lines += ["", "## Mean score by stage and writer (primary specification)", ""] + markdown_for_levels(primary["mean_projection_by_stage_and_writer"])
    length = primary["length_of_per_answer_projection"]
    lines += ["", f"Rank correlation of the per-answer score with log word count: pooled {fmt_interval(length['pooled_rank_correlation_with_log_word_count'], 2)}, "
                  f"within prompt {fmt_interval(length['within_prompt_rank_correlation_with_log_word_count'], 2)}."]
    lines += ["", "## Other specifications side by side", "",
              "| specification | too_much flaw shift | stage-2 repeat shift | stage-2 gpt-sol | stage-2 argon | stage-2 sonnet55 | pooled rho(too_much) | AUC more too_much ranks higher | pooled rho(delivery) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, result in results_by_specification.items():
        shifts, pooled = result["paired_shifts_main_prompts"], result["agreement_with_judges"]["pooled_stage2_stage3a_stage3b"]
        cell = lambda name: fmt_interval(shifts[name]["mean_shift"], 2) if name in shifts else "n/a"  # noqa: E731
        lines.append(f"| {key} | {cell('1_planted_flaw_too_much')} | {cell('2_stage_2_repeat_opus55_vs_opus55_stage_2')} | {cell('2_stage_2_gpt-sol_vs_opus55_stage_2')} | "
                     f"{cell('2_stage_2_argon_vs_opus55_stage_2')} | {cell('2_stage_2_sonnet55_vs_opus55_stage_2')} | {fmt_interval(pooled['too_much_spearman'], 2)} | "
                     f"{fmt_interval(flip_probability_interval(pooled['too_much_auc_fewer_vs_more']), 2, False)} | {fmt_interval(pooled['delivery_spearman'], 2)} |")
    lines += ["", "## Agreement with the moralizing delivery vector", "",
              f"Vector `{VECTOR_NAME}`, layer {chosen_layer}, format `{VECTOR_DOCUMENT_FORMAT}`, mean after the first 64 tokens. Both runs use the Gemma 4 12B instruct "
              f"tokenizer; token ids identical in {token_alignment['documents_with_identical_token_ids']} of {token_alignment['documents_compared']} answers. "
              "Agreement shows as a NEGATIVE rank correlation (vector more negative = more moralizing; contrast higher = more preachy).", "",
              "| comparison | n | Spearman [95% CI] |", "|---|---|---|"]
    for name in ("all_714_answers", "main_fifty_prompts_only"):
        entry = agreement_with_vector[name]
        lines.append(f"| per-answer, {name} | {entry['number_of_answers']} | {fmt_interval(entry['spearman_vector_projection_vs_preachy_minus_proportionate_welfare'], 2)} |")
        lines.append(f"| per-answer with prompt means removed, {name} | {entry['number_of_answers']} | {fmt_interval(entry['spearman_with_prompt_means_removed'], 2)} |")
    lines.append(f"| paired differences, main prompts | {agreement_with_vector['paired_differences_main_prompts']['number_of_pairs']} | "
                 f"{fmt_interval(agreement_with_vector['paired_differences_main_prompts']['spearman_vector_difference_vs_contrast_difference'], 2)} |")
    lines.append(f"| judged-comparison differences | {agreement_with_vector['judged_comparison_differences']['number_of_comparisons']} | "
                 f"{fmt_interval(agreement_with_vector['judged_comparison_differences']['spearman_vector_difference_vs_contrast_difference'], 2)} |")
    lines += ["", "Within-answer token agreement (vector flipped so moralizing is positive; expected positive):", "",
              "| measure | answers | mean | median | fraction positive |", "|---|---|---|---|---|"]
    for name, entry in per_token_summary.items():
        lines.append(f"| {name} | {entry['documents']} | {fmt(entry['mean'], 3)} | {fmt(entry['median'], 3)} | {entry['fraction_positive']:.2f} |")
    overlap = window_overlap["absolute_window_means"]
    relative_overlap = window_overlap["windows_furthest_from_own_document_mean"]
    lines += ["", "Top-15 window overlap (one window per answer per list): this method's 15 highest windows against the vector's 15 lowest windows: "
                  f"{overlap['documents_in_both_lists']} answers in both lists, {overlap['documents_with_overlapping_windows']} with overlapping windows. "
                  f"Windows furthest above/below their own answer's mean: {relative_overlap['documents_in_both_lists']} in both, "
                  f"{relative_overlap['documents_with_overlapping_windows']} overlapping. The vector's 15 lowest windows sit at a median contrast percentile of "
                  f"{window_overlap['median_contrast_percentile_of_the_vectors_top_windows']:.0f} (of all windows; 100 = most preachy); this method's 15 highest windows sit at a "
                  f"median vector percentile of {window_overlap['median_vector_percentile_of_this_methods_top_windows']:.0f} (0 = most moralizing).", ""]
    lines += [f"Stage of the extreme windows: highest list {stage_counts['highest_list_by_stage']}; lowest list {stage_counts['lowest_list_by_stage']}.", ""]
    lines += markdown_for_spans([r for r in span_records if r["list"].startswith("highest")], "15 highest windows (most preachy relative to proportionate welfare)")
    lines += markdown_for_spans([r for r in span_records if r["list"].startswith("lowest")], "15 lowest windows (least preachy relative to proportionate welfare)")
    (output_folder / f"difficult_advice_preachy_minus_proportionate_welfare_analysis_tables_{timestamp}.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
