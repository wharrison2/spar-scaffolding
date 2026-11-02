"""The statistics behind the positive-control pass criteria of every content-controlled facet (moralizing delivery's criteria are
coded below with their thresholds; the other facets read theirs from a per-facet criteria file, see
`evaluate_criteria_from_file`). Originally: the statistics behind the moralizing delivery pass criteria, as pure numpy / scipy functions with no file or model access, so
that tests can check them on synthetic data with known answers. The criteria are in
plans/10-30-persona-projection/moralizing-delivery-pass-criteria-fixed-before-scoring_2026-11-01_01-55.md; the thresholds below
are copied from it and nowhere else.

Conventions. A projection is onto the vector positive (plain delivery) minus negative (moralizing delivery), so moralizing is the
NEGATIVE direction. All per-token arrays are for ONE vector at ONE layer, shape (tokens,).
  changed span   mean over the rewrite's changed tokens minus mean over its unchanged tokens, from token 64 on (the first 64
                 tokens are the opening, which the other analyses also leave out). Changed tokens are found by a difflib diff
                 of the comparison document's token ids against the rewrite's (changed_masks_from_token_sequences).
  circular shift p   the changed-token mask is rotated by a random amount (2,000 draws) inside the same document; p is the share of
                 rotated shifts at least as far in the stated direction as the observed one, counting the observed shift itself
                 ((1 + hits) / (draws + 1)). Two-sided = min(1, 2 x the smaller one-sided p).
  whole document shift   mean over ALL rewrite tokens minus mean over ALL comparison tokens, with a paired moving-block bootstrap:
                 every comparison token is a unit, rewrite tokens attach to the unit they replace or follow (difflib opcodes),
                 blocks of 64 units are resampled (5,000 draws) and the same units' rewrite and comparison means are
                 differenced. One-sided p in a direction = share of resampled shifts not in that direction (zero or the other way).
"""

import difflib

import numpy as np
import pandas as pd
from scipy import stats

from persona_vector_projection import content_controlled_facet_positive_control_arms as arms_module
from persona_vector_projection.changed_token_counting_by_diff import changed_masks_from_token_sequences

OPENING_TOKENS_SKIPPED = 64
CIRCULAR_SHIFT_DRAWS = 2000
BOOTSTRAP_BLOCK_UNITS = 64
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_CHUNK_DRAWS = 250
SIGNIFICANCE_LEVEL = 0.05
WINDOW_TOKENS = 32
LOWEST_WINDOWS_TO_READ = 10
CONTEXT_TOKENS_BEFORE_WINDOW = 60

# ---- thresholds copied from the pass-criteria file
HELD_OUT_ACCURACY_MINIMUM = 0.80
DEVICE_ACCURACY_MINIMUM = 0.70
DEVICES_REQUIRED_TO_REACH_IT = 4
DEVICES_TOTAL = 5
SPLIT_HALF_COSINE_MINIMUM = 0.80
TRAINING_QUESTIONS_WITH_KEPT_ITEMS_MINIMUM = 15
TRAINING_QUESTIONS_TOTAL = 20
FRAMING_CHANGED_SPAN_REWRITES_REQUIRED = 8
FRAMING_WHOLE_DOCUMENT_REWRITES_REQUIRED = 7
REMOVED_CHANGED_SPAN_REWRITES_REQUIRED = 8
WILCOXON_P_MAXIMUM = 0.05
PLAIN_EFFECT_MAXIMUM_FRACTION_OF_FRAMING_EFFECT = 1 / 3
PLAIN_REWRITES_SIGNIFICANT_IN_MORALIZING_DIRECTION_MAXIMUM = 3
NEUTRAL_REWRITES_SIGNIFICANT_EITHER_DIRECTION_MAXIMUM = 2
CORPUS_WINDOWS_WITH_MORAL_WELFARE_DELIVERY_REQUIRED = 5


# ---------------------------------------------------------------------------------------------------
# Changed span
# ---------------------------------------------------------------------------------------------------

def changed_span_shift(per_token_values, comparison_token_ids, rewrite_token_ids, skipped_tokens=OPENING_TOKENS_SKIPPED):
    """(shift, changed mask after the skipped tokens, values after the skipped tokens). The shift is changed minus unchanged
    tokens after `skipped_tokens`; NaN when no token, or every token, after them is changed."""
    values = np.asarray(per_token_values, dtype=np.float64)
    if len(values) != len(rewrite_token_ids):
        raise ValueError(f"{len(values)} projections for {len(rewrite_token_ids)} rewrite tokens")
    rewrite_changed, _ = changed_masks_from_token_sequences(comparison_token_ids, rewrite_token_ids)
    mask, tail = rewrite_changed[skipped_tokens:], values[skipped_tokens:]
    if mask.sum() == 0 or mask.all():
        return float("nan"), mask, tail
    return float(tail[mask].mean() - tail[~mask].mean()), mask, tail


def circular_shift_p_values(tail_values, changed_mask, observed_shift, random_generator, draws=CIRCULAR_SHIFT_DRAWS):
    """{"p_lower", "p_upper", "p_two_sided"} for the changed-minus-unchanged shift against rotations of the mask. `p_lower` is
    the p value for a shift as low as the observed one (the moralizing direction), `p_upper` as high."""
    if not np.isfinite(observed_shift):
        return {"p_lower": float("nan"), "p_upper": float("nan"), "p_two_sided": float("nan")}
    mask = np.asarray(changed_mask, dtype=bool)
    values = np.asarray(tail_values, dtype=np.float64)
    rotations = random_generator.integers(len(mask), size=draws)
    null = np.empty(draws)
    for draw, rotation in enumerate(rotations):
        rotated = np.roll(mask, rotation)
        null[draw] = values[rotated].mean() - values[~rotated].mean()
    p_lower = (1 + np.sum(null <= observed_shift)) / (draws + 1)
    p_upper = (1 + np.sum(null >= observed_shift)) / (draws + 1)
    return {"p_lower": float(p_lower), "p_upper": float(p_upper), "p_two_sided": float(min(1.0, 2 * min(p_lower, p_upper)))}


# ---------------------------------------------------------------------------------------------------
# Whole document: paired block bootstrap over comparison-token units
# ---------------------------------------------------------------------------------------------------

def attach_rewrite_tokens_to_comparison_units(comparison_token_ids, rewrite_token_ids):
    """For each rewrite token, the index of the comparison token (unit) it belongs to: an unchanged token to its own unit, a token
    that replaces some comparison tokens to one of them (spread in order), an inserted token to the unit it follows."""
    comparison, rewrite = list(comparison_token_ids), list(rewrite_token_ids)
    owner = np.zeros(len(rewrite), dtype=int)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, comparison, rewrite, autojunk=False).get_opcodes():
        if tag == "equal":
            owner[j1:j2] = np.arange(i1, i2)
        elif tag == "replace":
            owner[j1:j2] = np.clip(np.linspace(i1, i2 - 1, j2 - j1).round().astype(int), 0, len(comparison) - 1)
        elif tag == "insert":
            owner[j1:j2] = max(i1 - 1, 0)
    return owner


def whole_document_shift_with_block_bootstrap(comparison_values, rewrite_values, comparison_token_ids, rewrite_token_ids,
                                              random_generator, block_units=BOOTSTRAP_BLOCK_UNITS, draws=BOOTSTRAP_DRAWS):
    """(observed shift, array of `draws` resampled shifts). Values are (tokens,) or (tokens, vectors); the resampled array is
    (draws,) or (draws, vectors). Observed = mean over all rewrite tokens minus mean over all comparison tokens."""
    comparison_values = np.asarray(comparison_values, dtype=np.float64)
    rewrite_values = np.asarray(rewrite_values, dtype=np.float64)
    single_vector = comparison_values.ndim == 1
    if single_vector:
        comparison_values, rewrite_values = comparison_values[:, None], rewrite_values[:, None]
    units = comparison_values.shape[0]
    owner = attach_rewrite_tokens_to_comparison_units(comparison_token_ids, rewrite_token_ids)
    rewrite_sum_per_unit = np.stack([np.bincount(owner, rewrite_values[:, column], minlength=units)
                                     for column in range(rewrite_values.shape[1])])          # (vectors, units)
    rewrite_count_per_unit = np.bincount(owner, minlength=units).astype(float)               # (units,)
    observed = rewrite_values.mean(axis=0) - comparison_values.mean(axis=0)
    blocks = int(np.ceil(units / block_units))
    resampled = np.empty((draws, comparison_values.shape[1]))
    done = 0
    while done < draws:
        chunk = min(BOOTSTRAP_CHUNK_DRAWS, draws - done)
        starts = random_generator.integers(0, units, size=(chunk, blocks))
        indices = ((starts[:, :, None] + np.arange(block_units)).reshape(chunk, -1)[:, :units]) % units
        with np.errstate(divide="ignore", invalid="ignore"):  # a resample whose units hold no rewrite token gives a non-finite shift
            rewrite_mean = rewrite_sum_per_unit[:, indices].sum(axis=-1) / rewrite_count_per_unit[indices].sum(axis=-1)  # (vectors, chunk)
        comparison_mean = comparison_values.T[:, indices].mean(axis=-1)                                  # (vectors, chunk)
        resampled[done:done + chunk] = (rewrite_mean - comparison_mean).T
        done += chunk
    if single_vector:
        return float(observed[0]), resampled[:, 0]
    return observed, resampled


def bootstrap_p_values(resampled_shifts):
    """One-sided p values from resampled shifts: `p_negative_direction` is the share of resampled shifts that are zero or positive
    (small when the shift is reliably negative), `p_positive_direction` the share zero or negative."""
    resampled = np.asarray(resampled_shifts, dtype=float)
    resampled = resampled[np.isfinite(resampled)]
    if len(resampled) == 0:
        return {"p_negative_direction": float("nan"), "p_positive_direction": float("nan")}
    return {"p_negative_direction": float(np.mean(resampled >= 0)), "p_positive_direction": float(np.mean(resampled <= 0))}


# ---------------------------------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------------------------------

def standardised_effect(shifts, sign=1):
    """sign x mean / standard deviation (n - 1) of the shifts; NaN with fewer than two finite shifts or no spread."""
    values = np.asarray([v for v in shifts if np.isfinite(v)], dtype=float)
    if len(values) < 2 or values.std(ddof=1) == 0:
        return float("nan")
    return float(sign * values.mean() / values.std(ddof=1))


def wilcoxon_p(shifts):
    values = np.asarray([v for v in shifts if np.isfinite(v)], dtype=float)
    if len(values) < 2 or not np.any(values != 0):
        return float("nan")
    return float(stats.wilcoxon(values).pvalue)


def count_in_direction(values, sign):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return int(np.sum(np.sign(values) == sign))


def summarise_arm(rewrite_table, arm, expected_sign=None):
    """One row of numbers for an arm from its per-rewrite table (columns changed_span_shift, circular_p_lower, circular_p_upper,
    circular_p_two_sided, whole_document_shift, whole_bootstrap_p_negative_direction, whole_bootstrap_p_positive_direction)."""
    # `expected_sign` overrides the arm's own expected sign: the plain-confidence control vector moves the other way from the
    # calibration vector on the overclaiming arm (calibrated consideration)
    expected = arms_module.expected_sign_of_any_arm(arm) if expected_sign is None else expected_sign
    span = rewrite_table["changed_span_shift"].to_numpy(dtype=float)
    whole = rewrite_table["whole_document_shift"].to_numpy(dtype=float)
    alpha = SIGNIFICANCE_LEVEL
    summary = {
        "arm": arm, "n_rewrites": int(len(rewrite_table)), "n_with_a_changed_span": int(np.isfinite(span).sum()),
        "expected_sign": expected,
        "changed_span_mean": float(np.nanmean(span)) if np.isfinite(span).any() else float("nan"),
        "changed_span_median": float(np.nanmedian(span)) if np.isfinite(span).any() else float("nan"),
        "changed_span_n_negative": count_in_direction(span, -1), "changed_span_n_positive": count_in_direction(span, 1),
        "changed_span_n_in_expected_direction": count_in_direction(span, expected) if expected else None,
        "changed_span_wilcoxon_p": wilcoxon_p(span),
        "standardised_effect_in_moralizing_direction": standardised_effect(span, arms_module.MORALIZING_DIRECTION_SIGN),
        # the same numbers under the facet-neutral name: the direction of the NEGATIVE pole (shift below zero)
        "standardised_effect_in_negative_pole_direction": standardised_effect(span, arms_module.NEGATIVE_POLE_DIRECTION_SIGN),
        "standardised_effect_in_expected_direction": standardised_effect(span, expected) if expected else None,
        "circular_n_significant_moralizing_direction": int((rewrite_table["circular_p_lower"] < alpha).sum()),
        "circular_n_significant_negative_pole_direction": int((rewrite_table["circular_p_lower"] < alpha).sum()),
        "circular_n_significant_positive_direction": int((rewrite_table["circular_p_upper"] < alpha).sum()),
        "circular_n_significant_either_direction_two_sided": int((rewrite_table["circular_p_two_sided"] < alpha).sum()),
        "circular_n_significant_either_direction_two_one_sided_tests": int(
            ((rewrite_table["circular_p_lower"] < alpha) | (rewrite_table["circular_p_upper"] < alpha)).sum()),
        "whole_document_shift_mean": float(np.nanmean(whole)) if np.isfinite(whole).any() else float("nan"),
        "whole_document_n_negative": count_in_direction(whole, -1), "whole_document_n_positive": count_in_direction(whole, 1),
        "whole_document_n_in_expected_direction": count_in_direction(whole, expected) if expected else None,
        "whole_bootstrap_n_significant_negative_direction": int((rewrite_table["whole_bootstrap_p_negative_direction"] < alpha).sum()),
        "whole_bootstrap_n_significant_positive_direction": int((rewrite_table["whole_bootstrap_p_positive_direction"] < alpha).sum()),
    }
    # The counted whole-document numbers of the content-controlled facets' criteria: rewrite mean minus comparison document mean after
    # token 64 (column `whole_document_shift_after_token_64`). NaN / None when the table has no such column (moralizing delivery's tables).
    if "whole_document_shift_after_token_64" in rewrite_table.columns:
        after_opening = rewrite_table["whole_document_shift_after_token_64"].to_numpy(dtype=float)
        finite = np.isfinite(after_opening)
        summary.update({
            "whole_document_after_token_64_shift_mean": float(np.nanmean(after_opening)) if finite.any() else float("nan"),
            "whole_document_after_token_64_n_negative": count_in_direction(after_opening, -1),
            "whole_document_after_token_64_n_positive": count_in_direction(after_opening, 1),
            "whole_document_after_token_64_n_in_expected_direction": count_in_direction(after_opening, expected) if expected else None,
            "whole_document_after_token_64_wilcoxon_p": wilcoxon_p(after_opening)})
    else:
        summary.update({"whole_document_after_token_64_shift_mean": float("nan"), "whole_document_after_token_64_n_negative": None,
                        "whole_document_after_token_64_n_positive": None, "whole_document_after_token_64_n_in_expected_direction": None,
                        "whole_document_after_token_64_wilcoxon_p": float("nan")})
    return summary


# ---------------------------------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------------------------------

def verdict(criterion_id, description, measured, required, passed):
    return {"criterion": criterion_id, "description": description, "measured": measured, "required": required,
            "passed": None if passed is None else bool(passed), "counted": True}


def evaluate_positive_control_criteria(summaries_by_arm):
    """Pass or fail lines for the four positive-control rows of the criteria file, from {arm: summarise_arm(...)}."""
    framing = summaries_by_arm[arms_module.ARM_MORALIZING_FRAMING_ONLY]
    removed = summaries_by_arm[arms_module.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE]
    plain = summaries_by_arm[arms_module.ARM_PLAIN_WELFARE_CONTENT_ADDED]
    neutral = summaries_by_arm[arms_module.ARM_FACET_NEUTRAL_REWORDING]
    small_p = lambda p: bool(np.isfinite(p) and p < WILCOXON_P_MAXIMUM)  # noqa: E731
    framing_effect, plain_effect = (framing["standardised_effect_in_moralizing_direction"],
                                    plain["standardised_effect_in_moralizing_direction"])
    effect_ratio = plain_effect / framing_effect if np.isfinite(framing_effect) and framing_effect != 0 else float("nan")
    effect_is_computable = bool(np.isfinite(framing_effect) and np.isfinite(plain_effect))
    return [
        verdict("moralizing_framing_only__changed_span_direction",
                "changed span: at least 8 of 10 negative", f"{framing['changed_span_n_negative']} of {framing['n_rewrites']} negative",
                f">= {FRAMING_CHANGED_SPAN_REWRITES_REQUIRED}", framing["changed_span_n_negative"] >= FRAMING_CHANGED_SPAN_REWRITES_REQUIRED),
        verdict("moralizing_framing_only__changed_span_wilcoxon", "changed span: Wilcoxon p < 0.05",
                framing["changed_span_wilcoxon_p"], f"< {WILCOXON_P_MAXIMUM}", small_p(framing["changed_span_wilcoxon_p"])),
        verdict("moralizing_framing_only__whole_document_direction", "whole document: at least 7 of 10 negative",
                f"{framing['whole_document_n_negative']} of {framing['n_rewrites']} negative",
                f">= {FRAMING_WHOLE_DOCUMENT_REWRITES_REQUIRED}", framing["whole_document_n_negative"] >= FRAMING_WHOLE_DOCUMENT_REWRITES_REQUIRED),
        verdict("moralizing_removed__changed_span_direction", "changed span: at least 8 of 10 positive",
                f"{removed['changed_span_n_positive']} of {removed['n_rewrites']} positive",
                f">= {REMOVED_CHANGED_SPAN_REWRITES_REQUIRED}", removed["changed_span_n_positive"] >= REMOVED_CHANGED_SPAN_REWRITES_REQUIRED),
        verdict("moralizing_removed__changed_span_wilcoxon", "changed span: Wilcoxon p < 0.05",
                removed["changed_span_wilcoxon_p"], f"< {WILCOXON_P_MAXIMUM}", small_p(removed["changed_span_wilcoxon_p"])),
        verdict("plain_welfare_content_added__effect_under_a_third_of_framing",
                "changed-span standardised effect in the moralizing direction under a third of the framing arm's",
                {"plain": plain_effect, "framing": framing_effect, "plain_over_framing": effect_ratio},
                f"plain < framing / 3 ({PLAIN_EFFECT_MAXIMUM_FRACTION_OF_FRAMING_EFFECT:.3f} x framing)",
                (plain_effect < PLAIN_EFFECT_MAXIMUM_FRACTION_OF_FRAMING_EFFECT * framing_effect) if effect_is_computable else None),
        verdict("plain_welfare_content_added__few_individually_moralizing",
                "at most 3 of 10 rewrites individually significant in the moralizing direction (circular shift, p < 0.05)",
                f"{plain['circular_n_significant_moralizing_direction']} of {plain['n_rewrites']}",
                f"<= {PLAIN_REWRITES_SIGNIFICANT_IN_MORALIZING_DIRECTION_MAXIMUM}",
                plain["circular_n_significant_moralizing_direction"] <= PLAIN_REWRITES_SIGNIFICANT_IN_MORALIZING_DIRECTION_MAXIMUM),
        verdict("facet_neutral_rewording__few_individually_significant",
                "at most 2 of 10 individually significant in either direction (circular shift, two-sided p < 0.05)",
                {"two_sided_p_below_0.05": neutral["circular_n_significant_either_direction_two_sided"],
                 "either_one_sided_p_below_0.05": neutral["circular_n_significant_either_direction_two_one_sided_tests"],
                 "n_rewrites": neutral["n_rewrites"]},
                f"<= {NEUTRAL_REWRITES_SIGNIFICANT_EITHER_DIRECTION_MAXIMUM}",
                neutral["circular_n_significant_either_direction_two_sided"] <= NEUTRAL_REWRITES_SIGNIFICANT_EITHER_DIRECTION_MAXIMUM)]


def evaluate_extraction_criteria(held_out_accuracy_at_chosen_layer, accuracy_by_device_at_chosen_layer, split_half_cosine_at_chosen_layer,
                                 training_questions_with_kept_items):
    """Pass or fail lines for the four extraction rows. Any input may be None (not recorded): its line is then undecided."""
    lines = []
    lines.append(verdict("extraction__held_out_accuracy", "held-out accuracy at the chosen layer at least 0.80",
                         held_out_accuracy_at_chosen_layer, f">= {HELD_OUT_ACCURACY_MINIMUM}",
                         None if held_out_accuracy_at_chosen_layer is None else held_out_accuracy_at_chosen_layer >= HELD_OUT_ACCURACY_MINIMUM))
    if accuracy_by_device_at_chosen_layer is None:
        lines.append(verdict("extraction__held_out_accuracy_by_device", "at least 0.70 for 4 of the 5 moralizing devices", None,
                             f">= {DEVICE_ACCURACY_MINIMUM} for {DEVICES_REQUIRED_TO_REACH_IT} of {DEVICES_TOTAL}", None))
    else:
        reaching = sum(1 for accuracy in accuracy_by_device_at_chosen_layer.values() if accuracy >= DEVICE_ACCURACY_MINIMUM)
        lines.append(verdict("extraction__held_out_accuracy_by_device", "at least 0.70 for 4 of the 5 moralizing devices",
                             {"devices_reaching_it": reaching, "devices_with_held_out_answers": len(accuracy_by_device_at_chosen_layer),
                              "accuracy_by_device": accuracy_by_device_at_chosen_layer},
                             f">= {DEVICE_ACCURACY_MINIMUM} for {DEVICES_REQUIRED_TO_REACH_IT} of {DEVICES_TOTAL}",
                             reaching >= DEVICES_REQUIRED_TO_REACH_IT))
    lines.append(verdict("extraction__split_half_cosine", "split-half cosine at the chosen layer at least 0.80",
                         split_half_cosine_at_chosen_layer, f">= {SPLIT_HALF_COSINE_MINIMUM}",
                         None if split_half_cosine_at_chosen_layer is None else split_half_cosine_at_chosen_layer >= SPLIT_HALF_COSINE_MINIMUM))
    lines.append(verdict("extraction__training_questions_with_kept_items",
                         "at least 15 of the 20 training questions have one or more kept items", training_questions_with_kept_items,
                         f">= {TRAINING_QUESTIONS_WITH_KEPT_ITEMS_MINIMUM} of {TRAINING_QUESTIONS_TOTAL}",
                         None if training_questions_with_kept_items is None
                         else training_questions_with_kept_items >= TRAINING_QUESTIONS_WITH_KEPT_ITEMS_MINIMUM))
    return lines


def evaluate_corpus_segment_criterion(hand_reading_by_window_id):
    """`hand_reading_by_window_id`: {window id: True when the window has an AI, or a quoted AI, delivering a welfare point with moral
    framing; False otherwise}, or None before the windows are read. Undecided until all windows have been read."""
    if hand_reading_by_window_id is None or any(value is None for value in hand_reading_by_window_id.values()):
        return verdict("corpus_segments__moral_welfare_delivery", "at least 5 of the 10 lowest windows contain an AI, or a quoted AI, "
                       "delivering a welfare point with moral framing (decided by hand reading)", None,
                       f">= {CORPUS_WINDOWS_WITH_MORAL_WELFARE_DELIVERY_REQUIRED} of {LOWEST_WINDOWS_TO_READ}", None)
    count = sum(1 for value in hand_reading_by_window_id.values() if value)
    return verdict("corpus_segments__moral_welfare_delivery", "at least 5 of the 10 lowest windows contain an AI, or a quoted AI, "
                   "delivering a welfare point with moral framing (decided by hand reading)",
                   f"{count} of {len(hand_reading_by_window_id)}", f">= {CORPUS_WINDOWS_WITH_MORAL_WELFARE_DELIVERY_REQUIRED} of {LOWEST_WINDOWS_TO_READ}",
                   count >= CORPUS_WINDOWS_WITH_MORAL_WELFARE_DELIVERY_REQUIRED)


# ---------------------------------------------------------------------------------------------------
# Lowest windows of the corpus documents
# ---------------------------------------------------------------------------------------------------

def lowest_windows_below_document_mean(per_token_values_by_document, count=LOWEST_WINDOWS_TO_READ, window=WINDOW_TOKENS,
                                       skipped_tokens=OPENING_TOKENS_SKIPPED):
    """The `count` lowest non-overlapping windows over all documents, ranked by how far the window mean falls below its document's mean
    (the mean of the document's tokens after the skipped opening). Windows start at or after the skipped tokens. Within a document the
    windows chosen never overlap; a document can hold several. Returns a list of dicts, lowest first."""
    from persona_vector_projection.persona_projection_analysis_statistics import (rolling_window_means,
                                                                                  select_non_overlapping_windows)
    candidates = []
    for document_id, values in per_token_values_by_document.items():
        values = np.asarray(values, dtype=float)
        if len(values) < skipped_tokens + window:
            continue
        document_mean = float(values[skipped_tokens:].mean())
        means = rolling_window_means(values, window)
        for start, window_mean in select_non_overlapping_windows(means, window, count, False, skipped_tokens):
            candidates.append({"document_id": document_id, "start_token": int(start), "window_mean": float(window_mean),
                               "document_mean": document_mean, "below_document_mean": float(window_mean - document_mean)})
    candidates.sort(key=lambda row: (row["below_document_mean"], row["document_id"], row["start_token"]))
    return candidates[:count]


# ---------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------

def rewrite_results_for_arm(arm, rewrite_ids, comparison_id_by_rewrite, per_token_by_document, token_ids_by_document,
                            vector_column, seed=0, circular_draws=CIRCULAR_SHIFT_DRAWS, bootstrap_draws=BOOTSTRAP_DRAWS):
    """DataFrame, one row per rewrite, for one vector (column `vector_column` of the per-token arrays). Random generators are
    seeded from (seed, rewrite id), so a rewrite's numbers do not depend on which others are analysed."""
    rows = []
    for rewrite_id in sorted(rewrite_ids):
        comparison_id = comparison_id_by_rewrite[rewrite_id]
        rewrite_values = per_token_by_document[rewrite_id][:, vector_column]
        comparison_values = per_token_by_document[comparison_id][:, vector_column]
        generator = np.random.default_rng([seed, int.from_bytes(rewrite_id.encode("utf-8"), "little") % (2 ** 32)])
        shift, mask, tail = changed_span_shift(rewrite_values, token_ids_by_document[comparison_id], token_ids_by_document[rewrite_id])
        circular = circular_shift_p_values(tail, mask, shift, generator, circular_draws)
        whole_shift, resampled = whole_document_shift_with_block_bootstrap(
            comparison_values, rewrite_values, token_ids_by_document[comparison_id], token_ids_by_document[rewrite_id], generator,
            draws=bootstrap_draws)
        bootstrap = bootstrap_p_values(resampled)
        after_opening_whole = float(rewrite_values[OPENING_TOKENS_SKIPPED:].mean() - comparison_values[OPENING_TOKENS_SKIPPED:].mean())
        rows.append({"arm": arm, "document_id": rewrite_id, "comparison_document_id": comparison_id,
                     "rewrite_tokens": len(rewrite_values), "comparison_tokens": len(comparison_values),
                     "changed_tokens_after_token_64": int(mask.sum()), "changed_span_shift": shift,
                     "circular_p_lower": circular["p_lower"], "circular_p_upper": circular["p_upper"],
                     "circular_p_two_sided": circular["p_two_sided"], "whole_document_shift": whole_shift,
                     "whole_document_shift_after_token_64": after_opening_whole,
                     "whole_bootstrap_p_negative_direction": bootstrap["p_negative_direction"],
                     "whole_bootstrap_p_positive_direction": bootstrap["p_positive_direction"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
# Whole-document shift beside the corpus between-document spread
# ---------------------------------------------------------------------------------------------------

def between_document_standard_deviation(whole_document_mean_projection_by_corpus_document):
    """Standard deviation (n - 1) across the corpus documents of each document's whole-document mean projection (after the opening
    tokens, as the other whole-document numbers). `whole_document_mean_projection_by_corpus_document`: {document id: float}.
    NaN with fewer than two documents. This is the unit the whole-document shift of a rewrite is read against."""
    values = np.asarray([v for v in whole_document_mean_projection_by_corpus_document.values() if np.isfinite(v)], dtype=float)
    return float(values.std(ddof=1)) if len(values) >= 2 else float("nan")


def whole_document_mean_projection(per_token_values, skipped_tokens=OPENING_TOKENS_SKIPPED):
    values = np.asarray(per_token_values, dtype=float)
    tail = values[skipped_tokens:] if len(values) > skipped_tokens else values
    return float(tail.mean())


def shift_in_units_of_between_document_spread(shift, between_document_sd):
    return float(shift / between_document_sd) if np.isfinite(shift) and np.isfinite(between_document_sd) and between_document_sd > 0 else float("nan")




# ---------------------------------------------------------------------------------------------------
# Criteria read from a per-facet file. The criteria are in
# plans/10-30-persona-projection/content-controlled-facets-pass-criteria-fixed-before-scoring_2026-11-01_15-04.md and the thresholds are
# the ones in pass_criteria/positive_control_pass_criteria__<facet key>.json; this code never defaults one. A threshold that is null gives
# 'not evaluated: thresholds not fixed'. Counted lines decide the verdict; reported lines (`counted` false, verdict text 'reported')
# are computed and printed beside them with their former thresholds and never change it. Moralizing delivery keeps its own coded criteria
# above and is not evaluated here.
# ---------------------------------------------------------------------------------------------------

NOT_EVALUATED_THRESHOLDS_NOT_FIXED = "not evaluated: thresholds not fixed"
VERDICT_PASSES = "passes"
VERDICT_FAILS = "fails"
VERDICT_UNDECIDED = "undecided"
REPORTED_VERDICT_TEXT = "reported"


def load_pass_criteria_file(path):
    """The per-facet criteria JSON (scripts/persona_vector_projection/pass_criteria/positive_control_pass_criteria__<facet key>.json).
    Returns {} when the file is missing, which evaluates every criterion as not evaluated. Never fills a missing value."""
    import json
    from pathlib import Path
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _threshold(criteria, block, name):
    value = (criteria.get(block) or {}).get(name)
    return None if value is None else float(value)


def _flag(criteria, block, name):
    value = (criteria.get(block) or {}).get(name)
    return None if value is None else bool(value)


def undecided_verdict(criterion_id, description, measured, required_description):
    line = verdict(criterion_id, description, measured, required_description, None)
    line["verdict_text"] = NOT_EVALUATED_THRESHOLDS_NOT_FIXED
    return line


def threshold_verdict(criterion_id, description, measured, threshold, required_description, passed_function):
    """A counted verdict line, or the 'not evaluated' line when the threshold is null (or the measured value cannot be computed)."""
    if threshold is None:
        return undecided_verdict(criterion_id, description, measured, required_description.replace("{threshold:g}", "a threshold not yet fixed"))
    outcome = passed_function(threshold)
    line = verdict(criterion_id, description, measured, required_description.format(threshold=threshold), outcome)
    line["verdict_text"] = "PASS" if outcome else ("FAIL" if outcome is not None else "undecided")
    return line


def reported_line(criterion_id, description, measured, former_threshold_text):
    """A line that is computed and printed but never counted: passed is None, counted is False, the text is 'reported'."""
    line = verdict(criterion_id, description, measured, former_threshold_text, None)
    line["counted"] = False
    line["verdict_text"] = REPORTED_VERDICT_TEXT
    return line


def _both_finite(*numbers):
    return all(number is not None and np.isfinite(number) for number in numbers)


def _count_at_least(count, required):
    return None if count is None else count >= required


def trait_changed_arm_lines(arm_name, summary, neutral_summary, criteria):
    """Counted lines for an arm whose rewrites move towards the NEGATIVE pole (trait_only_away; understatement_of_supported_claims):
    changed span, at least N in the expected (negative) direction and Wilcoxon p below the maximum; whole document (after token 64), at
    least N in the expected direction and the arm's mean shift towards the negative pole at least `factor` times the ABSOLUTE mean
    shift of the facet-neutral rewording arm."""
    n = summary["n_rewrites"]
    wilcoxon = summary["changed_span_wilcoxon_p"]
    whole_count = summary["whole_document_after_token_64_n_negative"]
    arm_mean, neutral_mean = summary["whole_document_after_token_64_shift_mean"], neutral_summary["whole_document_after_token_64_shift_mean"]
    ratio = (-arm_mean / abs(neutral_mean)) if _both_finite(arm_mean, neutral_mean) and neutral_mean != 0 else float("nan")
    return [
        threshold_verdict(f"{arm_name}__changed_span_direction", f"{arm_name}: changed span, enough rewrites negative",
                          f"{summary['changed_span_n_negative']} of {n} negative",
                          _threshold(criteria, arm_name, "changed_span_rewrites_required_in_expected_direction"), ">= {threshold:g} rewrites",
                          lambda t: summary["changed_span_n_negative"] >= t),
        threshold_verdict(f"{arm_name}__changed_span_wilcoxon", f"{arm_name}: changed span, Wilcoxon p below the maximum", wilcoxon,
                          _threshold(criteria, arm_name, "changed_span_wilcoxon_p_maximum"), "< {threshold:g}",
                          lambda t: bool(np.isfinite(wilcoxon) and wilcoxon < t)),
        threshold_verdict(f"{arm_name}__whole_document_direction", f"{arm_name}: whole document after token 64, enough rewrites negative",
                          f"{whole_count} of {n} negative", _threshold(criteria, arm_name, "whole_document_rewrites_required_in_expected_direction"),
                          ">= {threshold:g} rewrites", lambda t: _count_at_least(whole_count, t)),
        threshold_verdict(f"{arm_name}__whole_document_shift_against_neutral_rewording",
                          f"{arm_name}: whole document after token 64, mean shift towards the negative pole at least a multiple of the "
                          "neutral rewording arm's ABSOLUTE mean shift",
                          {"arm_mean_shift": arm_mean, "neutral_rewording_absolute_mean_shift": abs(neutral_mean) if np.isfinite(neutral_mean) else neutral_mean,
                           "arm_shift_towards_negative_pole_over_neutral_absolute": ratio},
                          _threshold(criteria, arm_name, "whole_document_mean_shift_minimum_multiple_of_neutral_rewording_absolute_mean_shift"),
                          "-arm mean shift >= {threshold:g} x |neutral mean shift|",
                          lambda t: ((-arm_mean) >= t * abs(neutral_mean)) if _both_finite(arm_mean, neutral_mean) else None)]


def trait_removed_arm_lines(summary, criteria):
    """Counted: whole document (after token 64), at least N rewrites positive and Wilcoxon p below the maximum. Reported, not counted:
    the changed span, with its former thresholds."""
    arm_name = "trait_removed_from_away_rewrite"
    n = summary["n_rewrites"]
    whole_count = summary["whole_document_after_token_64_n_positive"]
    whole_p = summary["whole_document_after_token_64_wilcoxon_p"]
    former = (criteria.get("reported_not_counted") or {}).get("trait_removed_from_away_rewrite_changed_span") or {}
    former_count, former_p = former.get("former_rewrites_required_in_expected_direction"), former.get("former_wilcoxon_p_maximum")
    return [
        threshold_verdict(f"{arm_name}__whole_document_direction", f"{arm_name}: whole document after token 64, enough rewrites positive",
                          f"{whole_count} of {n} positive", _threshold(criteria, arm_name, "whole_document_rewrites_required_in_expected_direction"),
                          ">= {threshold:g} rewrites", lambda t: _count_at_least(whole_count, t)),
        threshold_verdict(f"{arm_name}__whole_document_wilcoxon", f"{arm_name}: whole document after token 64, Wilcoxon p below the maximum",
                          whole_p, _threshold(criteria, arm_name, "whole_document_wilcoxon_p_maximum"), "< {threshold:g}",
                          lambda t: bool(np.isfinite(whole_p) and whole_p < t)),
        reported_line(f"{arm_name}__changed_span_direction__reported", f"{arm_name}: changed span, rewrites positive (reported, not counted)",
                      f"{summary['changed_span_n_positive']} of {n} positive", f"former threshold: >= {former_count} rewrites"),
        reported_line(f"{arm_name}__changed_span_wilcoxon__reported", f"{arm_name}: changed span, Wilcoxon p (reported, not counted)",
                      summary["changed_span_wilcoxon_p"], f"former threshold: < {former_p}")]


def content_added_arm_lines(content, away, criteria):
    """Counted: the changed-span standardised effect and the whole-document mean shift (after token 64) towards the negative pole are each
    below a fraction of the trait-only-away arm's (a trait arm that did not move towards the negative pole fails the line); at most N
    rewrites individually significant towards the negative pole."""
    block = "content_added_without_trait"
    content_effect, away_effect = content["standardised_effect_in_negative_pole_direction"], away["standardised_effect_in_negative_pole_direction"]
    content_shift, away_shift = -content["whole_document_after_token_64_shift_mean"], -away["whole_document_after_token_64_shift_mean"]

    def below_a_fraction(content_value, away_value, fraction):
        if not _both_finite(content_value, away_value):
            return None
        return bool(away_value > 0 and content_value < fraction * away_value)

    return [
        threshold_verdict(f"{block}__effect_small_next_to_trait_only_away",
                          "changed-span standardised effect towards the negative pole, as a fraction of the trait-only-away arm's",
                          {"content_added": content_effect, "trait_only_away": away_effect},
                          _threshold(criteria, block, "effect_maximum_fraction_of_trait_only_away_effect"),
                          "content_added < {threshold:g} x trait_only_away",
                          lambda t: below_a_fraction(content_effect, away_effect, t)),
        threshold_verdict(f"{block}__whole_document_shift_small_next_to_trait_only_away",
                          "whole-document mean shift after token 64 towards the negative pole, as a fraction of the trait-only-away arm's",
                          {"content_added": content_shift, "trait_only_away": away_shift},
                          _threshold(criteria, block, "whole_document_mean_shift_maximum_fraction_of_trait_only_away_mean_shift"),
                          "content_added < {threshold:g} x trait_only_away",
                          lambda t: below_a_fraction(content_shift, away_shift, t)),
        threshold_verdict(f"{block}__few_individually_towards_the_negative_pole",
                          "rewrites individually significant towards the negative pole (circular shift, p < 0.05)",
                          f"{content['circular_n_significant_negative_pole_direction']} of {content['n_rewrites']}",
                          _threshold(criteria, block, "rewrites_significant_towards_negative_pole_maximum"), "<= {threshold:g}",
                          lambda t: content["circular_n_significant_negative_pole_direction"] <= t)]


def neutral_rewording_lines(neutral, away, criteria):
    """Counted: at most N rewrites individually significant in either direction (circular shift, two-sided); the whole-document ABSOLUTE
    mean shift (after token 64) below a fraction of the trait-only-away arm's absolute mean shift."""
    block = "facet_neutral_rewording"
    neutral_shift, away_shift = neutral["whole_document_after_token_64_shift_mean"], away["whole_document_after_token_64_shift_mean"]
    return [
        threshold_verdict(f"{block}__few_individually_significant",
                          "rewrites individually significant in either direction (circular shift, two-sided p < 0.05)",
                          {"two_sided_p_below_0.05": neutral["circular_n_significant_either_direction_two_sided"], "n_rewrites": neutral["n_rewrites"]},
                          _threshold(criteria, block, "rewrites_significant_either_direction_maximum"), "<= {threshold:g}",
                          lambda t: neutral["circular_n_significant_either_direction_two_sided"] <= t),
        threshold_verdict(f"{block}__whole_document_absolute_mean_shift_small_next_to_trait_only_away",
                          "whole-document absolute mean shift after token 64, as a fraction of the trait-only-away arm's absolute mean shift",
                          {"neutral_rewording_absolute_mean_shift": abs(neutral_shift) if np.isfinite(neutral_shift) else neutral_shift,
                           "trait_only_away_absolute_mean_shift": abs(away_shift) if np.isfinite(away_shift) else away_shift},
                          _threshold(criteria, block, "whole_document_absolute_mean_shift_maximum_fraction_of_trait_only_away_mean_shift"),
                          "|neutral mean shift| < {threshold:g} x |trait_only_away mean shift|",
                          lambda t: (abs(neutral_shift) < t * abs(away_shift)) if _both_finite(neutral_shift, away_shift) else None)]


def evaluate_criteria_from_file(summaries_by_arm, criteria, facet_key):
    """Positive-control criterion lines for a content-controlled facet (not moralizing delivery) from {arm: summarise_arm(...)} and the
    thresholds in `criteria`. Arm names and what each is compared with:
      trait_only_away                  against the original document; expected sign negative (towards the negative pole)
      trait_removed_from_away_rewrite  against the trait_only_away REWRITE it was made from; expected sign positive; counted on the
                                       whole document (the changed span is reported with its former thresholds)
      content_added_without_trait      against the original; small next to trait_only_away's, few individually significant towards the
                                       negative pole
      facet_neutral_rewording          against the original; few individually significant in either direction, small whole-document shift
      understatement_of_supported_claims (calibrated consideration only) against the original; the same lines as trait_only_away
    Every threshold that is null gives 'not evaluated: thresholds not fixed'; nothing is defaulted."""
    away = summaries_by_arm["trait_only_away"]
    removed = summaries_by_arm["trait_removed_from_away_rewrite"]
    content = summaries_by_arm["content_added_without_trait"]
    neutral = summaries_by_arm["facet_neutral_rewording"]
    lines = trait_changed_arm_lines("trait_only_away", away, neutral, criteria)
    if "understatement_of_supported_claims" in summaries_by_arm:  # calibrated consideration: also towards the negative pole
        lines += trait_changed_arm_lines("understatement_of_supported_claims", summaries_by_arm["understatement_of_supported_claims"], neutral,
                                         criteria)
    lines += trait_removed_arm_lines(removed, criteria)
    lines += content_added_arm_lines(content, away, criteria)
    lines += neutral_rewording_lines(neutral, away, criteria)
    for line in lines:
        line["facet_key"] = facet_key
    return lines


def evaluate_extraction_criteria_from_file(criteria, held_out_accuracy, accuracy_by_tag, held_out_negatives_by_tag, cross_half_accuracy,
                                           split_half_cosine, training_questions_with_kept_items, training_questions_total):
    """The extraction lines from the criteria file's `extraction` block, all from numbers at the chosen layer (any may be None: not
    recorded, the line is then undecided). Counted: held-out accuracy; held-out accuracy for every negative tag with at least the
    minimum number of held-out negatives (tags with fewer are listed in the measured value and not counted); cross-half accuracy (the
    vector from half A's questions evaluated on half B's questions); the share of training questions with a kept item. Reported, not
    counted: split-half cosine with its former threshold."""
    block = criteria.get("extraction") or {}
    value = lambda name: None if block.get(name) is None else float(block[name])  # noqa: E731
    lines = [threshold_verdict("extraction__held_out_accuracy", "held-out accuracy at the chosen layer", held_out_accuracy,
                               value("held_out_accuracy_minimum"), ">= {threshold:g}",
                               lambda t: None if held_out_accuracy is None else held_out_accuracy >= t)]
    minimum_negatives = value("tag_minimum_held_out_negatives")
    counted_tags, reported_tags, below = {}, {}, []
    if accuracy_by_tag is not None and held_out_negatives_by_tag is not None and minimum_negatives is not None:
        for tag, accuracy in sorted(accuracy_by_tag.items()):
            negatives = held_out_negatives_by_tag.get(tag, 0)
            (counted_tags if negatives >= minimum_negatives else reported_tags)[tag] = {"accuracy": accuracy, "held_out_negatives": negatives}

    def every_counted_tag_reaches(minimum):
        if accuracy_by_tag is None or held_out_negatives_by_tag is None or not counted_tags:
            return None
        below[:] = [tag for tag, cell in counted_tags.items() if cell["accuracy"] < minimum]
        return not below

    lines.append(threshold_verdict(
        "extraction__held_out_accuracy_by_negative_tag", "held-out accuracy of every negative tag with enough held-out negatives",
        {"tags_counted": counted_tags, "tags_reported_only_too_few_held_out_negatives": reported_tags},
        value("held_out_accuracy_by_tag_minimum") if minimum_negatives is not None else None,
        "every tag with at least " + (f"{minimum_negatives:g}" if minimum_negatives is not None else "a number not yet fixed")
        + " held-out negatives >= {threshold:g}", every_counted_tag_reaches))
    lines.append(threshold_verdict("extraction__cross_half_accuracy",
                                   "cross-half accuracy at the chosen layer: vector from half A's questions, evaluated on half B's questions",
                                   cross_half_accuracy, value("cross_half_accuracy_minimum"), ">= {threshold:g}",
                                   lambda t: None if cross_half_accuracy is None else cross_half_accuracy >= t))
    share = (training_questions_with_kept_items / training_questions_total
             if training_questions_with_kept_items is not None and training_questions_total else None)
    lines.append(threshold_verdict("extraction__training_questions_with_kept_items",
                                   "share of training questions with one or more kept items",
                                   {"questions_with_kept_items": training_questions_with_kept_items, "training_questions": training_questions_total,
                                    "share": share},
                                   value("training_questions_with_kept_items_minimum_share"), ">= {threshold:g} of the training questions",
                                   lambda t: None if share is None else share >= t))
    former_cosine = ((criteria.get("reported_not_counted") or {}).get("split_half_cosine") or {}).get("former_minimum")
    lines.append(reported_line("extraction__split_half_cosine__reported", "split-half cosine at the chosen layer (reported, not counted)",
                               split_half_cosine, f"former threshold: >= {former_cosine}"))
    return lines


def evaluate_calibrated_consideration_criteria_from_file(criteria, projected_out_accuracy, two_by_two_check):
    """Counted lines for calibrated consideration's calibration vector at the chosen layer; none for another facet (no
    `calibrated_consideration_only` block). `projected_out_accuracy`: held-out accuracy of the calibration vector with the plain-confidence
    direction projected out. `two_by_two_check`: the `two_by_two_check` result for the calibration vector (interaction contrast with its
    95 percent interval, main effect of confidence). The two-by-two rule passes only when BOTH the interval excludes zero AND the
    absolute interaction contrast is larger than the absolute main effect of confidence; if it fails the vector is a confidence vector."""
    block = "calibrated_consideration_only"
    if not criteria.get(block):
        return []
    lines = [threshold_verdict("calibrated_consideration__projected_out_held_out_accuracy",
                               "held-out accuracy of the calibration vector with the plain-confidence direction projected out",
                               projected_out_accuracy, _threshold(criteria, block, "projected_out_held_out_accuracy_minimum"), ">= {threshold:g}",
                               lambda t: None if projected_out_accuracy is None else projected_out_accuracy >= t)]
    interaction = None if two_by_two_check is None else two_by_two_check.get("interaction_contrast_calibrated_minus_miscalibrated")
    main_effect = None if two_by_two_check is None else two_by_two_check.get("main_effect_of_confidence_confident_minus_hedged")
    interval = None if two_by_two_check is None else two_by_two_check.get("interaction_contrast_95_percent_bootstrap_interval")
    interval_excludes_zero = None if interval is None else bool(interval[0] > 0 or interval[1] < 0)
    interaction_larger = None if not _both_finite(interaction, main_effect) else bool(abs(interaction) > abs(main_effect))
    conditions = [interval_excludes_zero, interaction_larger]
    outcome = False if any(c is False for c in conditions) else (None if any(c is None for c in conditions) else True)
    flags = [_flag(criteria, block, "two_by_two_interaction_interval_must_exclude_zero"),
             _flag(criteria, block, "two_by_two_interaction_absolute_size_must_exceed_confidence_main_effect_absolute_size")]
    measured = {"interaction_contrast_calibrated_minus_miscalibrated": interaction, "interaction_95_percent_interval": interval,
                "interval_excludes_zero": interval_excludes_zero, "main_effect_of_confidence": main_effect,
                "absolute_interaction_larger_than_absolute_main_effect": interaction_larger}
    description = ("two-by-two held-out check: interaction contrast interval excludes zero AND its absolute size exceeds the absolute main "
                   "effect of confidence (otherwise the vector is reported as a confidence vector, not a calibration vector)")
    required = "both conditions"
    if not all(flags):
        lines.append(undecided_verdict("calibrated_consideration__two_by_two_rule", description, measured, required))
    else:
        line = verdict("calibrated_consideration__two_by_two_rule", description, measured, required, outcome)
        line["verdict_text"] = "PASS" if outcome else ("FAIL" if outcome is not None else "undecided")
        if outcome is False:
            line["consequence"] = (criteria[block].get("if_the_two_by_two_rule_fails")
                                   or "the vector is reported as a confidence vector, not a calibration vector")
        lines.append(line)
    return lines


def evaluate_corpus_segment_criterion_from_file(criteria, hand_reading_by_window_id):
    """Corpus segments are REPORTED, not counted: the lowest windows are listed and described in every writeup. This line prints the hand
    reading (how many of the windows a reader marked as showing the negative pole), never a pass or fail."""
    readings = None if hand_reading_by_window_id is None or any(v is None for v in hand_reading_by_window_id.values()) else hand_reading_by_window_id
    count = None if readings is None else sum(1 for v in readings.values() if v)
    return reported_line("corpus_segments__trait_shown__reported",
                         "lowest windows that contain an AI, or a quoted AI, showing the facet's negative pole (hand reading; reported, not counted)",
                         None if count is None else f"{count} of {len(readings)}", "no threshold: listed and described in every writeup")


def line_verdict_text(line):
    return line.get("verdict_text") or ("PASS" if line["passed"] else "FAIL" if line["passed"] is False else "undecided")


def overall_verdict(lines, thresholds_fixed=True):
    """{"verdict", "counted_criteria", "reported_not_counted", "counts"}. The verdict is 'passes' only if there is at least one counted
    criterion and every counted criterion passes; 'fails' if any counted criterion fails; otherwise 'undecided' (a counted criterion could
    not be computed). Without fixed thresholds it is 'not evaluated: thresholds not fixed'. Reported lines never change it."""
    counted = [line for line in lines if line.get("counted", True)]
    reported = [line for line in lines if not line.get("counted", True)]
    if not thresholds_fixed:
        text = NOT_EVALUATED_THRESHOLDS_NOT_FIXED
    elif any(line["passed"] is False for line in counted):
        text = VERDICT_FAILS
    elif not counted or any(line["passed"] is not True for line in counted):
        text = VERDICT_UNDECIDED
    else:
        text = VERDICT_PASSES
    listing = lambda line: {"criterion": line["criterion"], "description": line["description"], "measured": line["measured"],  # noqa: E731
                            "required": line["required"], "verdict": line_verdict_text(line)}
    return {"verdict": text, "counted_criteria": [listing(line) for line in counted],
            "reported_not_counted": [listing(line) for line in reported],
            "counts": {"counted": len(counted), "passed": sum(1 for l in counted if l["passed"] is True),
                       "failed": sum(1 for l in counted if l["passed"] is False),
                       "undecided": sum(1 for l in counted if l["passed"] is None), "reported": len(reported)}}
