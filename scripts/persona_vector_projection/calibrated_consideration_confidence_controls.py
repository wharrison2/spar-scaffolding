"""Calibrated consideration only: balancing stated confidence between the poles, the plain-confidence control vector, projecting the
confidence direction out, and the two-by-two held-out check. Pure functions (numpy only; no model, API or pod); the vector computation
(compute_content_controlled_facet_vectors_from_activations.py) calls them when the facet definition has a stated-confidence judge.

Why. The negative pole is miscalibrated in both directions, so plain confidence should cancel in positive minus negative. The checks
here show whether it does: stated confidence (judge 0 entirely open, 100 certain, ignoring the evidence) is balanced between the poles
before the vector is computed, a confidence vector is built and compared, and held-out answers are sorted by confidence and evidence
band, not by pole.

Definitions used throughout.
  confident        stated confidence at least 65          hedged        stated confidence at most 35   (answers in between are left out)
  band group       established | strong_but_debated | weak | mixed (from the question's evidence band label)
  calibrated cells  confident on established evidence, hedged on weak evidence
  miscalibrated     hedged on established evidence, confident on weak evidence
"""

import numpy as np

from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module
from persona_vector_projection import persona_vector_experiment_definitions as defs

CONFIDENT_MINIMUM = 65.0
HEDGED_MAXIMUM = 35.0
DEFAULT_MAXIMUM_POLE_DIFFERENCE_IN_MEAN_STATED_CONFIDENCE = 5.0
BAND_GROUPS = (facet_definitions_module.BAND_ESTABLISHED, facet_definitions_module.BAND_STRONG_BUT_DEBATED,
               facet_definitions_module.BAND_WEAK, facet_definitions_module.BAND_MIXED)
CELL_CONFIDENT_ESTABLISHED = "confident_on_established_evidence"
CELL_HEDGED_ESTABLISHED = "hedged_on_established_evidence"
CELL_CONFIDENT_WEAK = "confident_on_weak_evidence"
CELL_HEDGED_WEAK = "hedged_on_weak_evidence"
CALIBRATED_CELLS = (CELL_CONFIDENT_ESTABLISHED, CELL_HEDGED_WEAK)
MISCALIBRATED_CELLS = (CELL_HEDGED_ESTABLISHED, CELL_CONFIDENT_WEAK)
ALL_CELLS = CALIBRATED_CELLS + MISCALIBRATED_CELLS


class ConfidenceBalancingError(ValueError):
    """Raised, never swallowed, when stated confidence cannot be balanced or a control cannot be built soundly."""


def band_group_of_question(question_record):
    return facet_definitions_module.evidence_band_class((question_record.get("labels") or {}).get("evidence_band"))[0]


def stated_confidence_condition_failures(answer, judgement, question_record, definition):
    """Extra answer conditions for calibrated consideration: the stated-confidence score must exist, and a negative rewording must be
    miscalibrated for the question's band (the generation step already only uses such rewordings; this catches an answer that was
    generated with a rewording that is accurate on its band)."""
    failures = []
    if judgement.get(facet_definitions_module.KIND_STATED_CONFIDENCE) is None:
        failures.append("stated_confidence_missing")
    if answer["pole"] == defs.POLE_NEGATIVE and not definition.negative_rewording_is_miscalibrated_for_question(answer["tag"], question_record):
        failures.append("negative_rewording_is_accurate_on_the_questions_evidence_band")
    return failures


# ---------------------------------------------------------------------------------------------------
# Balancing stated confidence between the poles
# ---------------------------------------------------------------------------------------------------

def pole_mean_stated_confidence(items, stated_confidence_by_answer_id, band_group_by_question_id):
    """{"overall" and each band group: {"items", "positive_pole_mean", "negative_pole_mean", "difference_positive_minus_negative"}} over
    items [(key, positive answer, negative answer)], key[0] being the question id."""
    groups = {"overall": list(items)}
    for band_group in BAND_GROUPS:
        groups[band_group] = [item for item in items if band_group_by_question_id[item[0][0]] == band_group]
    report = {}
    for name, members in groups.items():
        if not members:
            report[name] = {"items": 0, "positive_pole_mean": None, "negative_pole_mean": None, "difference_positive_minus_negative": None}
            continue
        positive_mean = float(np.mean([stated_confidence_by_answer_id[p["answer_id"]] for _, p, _ in members]))
        negative_mean = float(np.mean([stated_confidence_by_answer_id[n["answer_id"]] for _, _, n in members]))
        report[name] = {"items": len(members), "positive_pole_mean": positive_mean, "negative_pole_mean": negative_mean,
                        "difference_positive_minus_negative": positive_mean - negative_mean}
    return report


def balance_items_on_stated_confidence(items, stated_confidence_by_answer_id, band_group_by_question_id,
                                       maximum_difference=DEFAULT_MAXIMUM_POLE_DIFFERENCE_IN_MEAN_STATED_CONFIDENCE):
    """Subsample kept items WITHIN each band group until the two poles' mean stated confidence differs by at most `maximum_difference`
    in that group. Because the overall difference is a weighted average of the groups' differences, it is then at most the maximum too,
    so the requirement (overall, within established, within weak) holds, and strong-but-debated and mixed are balanced as well.

    Within a group, the item whose removal brings the difference closest to zero is dropped, one at a time (ties by item key), so the
    result is deterministic and keeps as many items as that rule allows. Returns (kept items, report with the means before and after).
    Raises ConfidenceBalancingError when a group cannot reach the target (it would empty first): the pipeline never carries on with
    unbalanced poles. The report is written to the vector set's metadata."""
    items = sorted(items, key=lambda item: item[0])
    before = pole_mean_stated_confidence(items, stated_confidence_by_answer_id, band_group_by_question_id)
    kept = []
    removed_by_group = {}
    for band_group in BAND_GROUPS:
        members = [item for item in items if band_group_by_question_id[item[0][0]] == band_group]
        differences = [stated_confidence_by_answer_id[p["answer_id"]] - stated_confidence_by_answer_id[n["answer_id"]] for _, p, n in members]
        alive = list(range(len(members)))
        while alive and abs(np.mean([differences[i] for i in alive])) > maximum_difference:
            total = sum(differences[i] for i in alive)
            if len(alive) == 1:
                alive = []
                break
            drop = min(alive, key=lambda i: (abs((total - differences[i]) / (len(alive) - 1)), members[i][0]))
            alive.remove(drop)
        if members and not alive:
            raise ConfidenceBalancingError(
                f"Stated confidence cannot be balanced between the poles in the {band_group} band: every subsample of its "
                f"{len(members)} kept items differs by more than {maximum_difference} points (starting difference "
                f"{before[band_group]['difference_positive_minus_negative']:.1f}). The poles differ in confidence on this band; the vector "
                f"would carry that difference.")
        removed_by_group[band_group] = len(members) - len(alive)
        kept.extend(members[i] for i in alive)
    kept.sort(key=lambda item: item[0])
    after = pole_mean_stated_confidence(kept, stated_confidence_by_answer_id, band_group_by_question_id)
    for name, entry in after.items():  # recheck on the result itself: a balancing that did not work must not pass silently
        if entry["items"] and abs(entry["difference_positive_minus_negative"]) > maximum_difference + 1e-9:
            raise ConfidenceBalancingError(f"After balancing, the {name} difference in mean stated confidence is "
                                           f"{entry['difference_positive_minus_negative']:.2f}, above {maximum_difference}")
    return kept, {"maximum_difference_allowed": maximum_difference, "items_removed_by_band_group": removed_by_group,
                  "before_balancing": before, "after_balancing": after}


# ---------------------------------------------------------------------------------------------------
# The plain-confidence control vector
# ---------------------------------------------------------------------------------------------------

def confidence_group_of(stated_confidence):
    if stated_confidence >= CONFIDENT_MINIMUM:
        return "confident"
    if stated_confidence <= HEDGED_MAXIMUM:
        return "hedged"
    return None


def select_balanced_confidence_groups(answers, stated_confidence_by_answer_id, seed=0):
    """The answers for the confidence vector: stated confidence 65 or more ("confident") and 35 or less ("hedged"), subsampled so each
    group holds the same number of positive-pole (calibrated) and negative-pole (miscalibrated) answers, and the two groups are the same
    size: n answers in each of the four (group, pole) cells, n the smallest cell. {"confident": [answer ids], "hedged": [answer ids],
    "cell_sizes_before": ..., "answers_per_cell": n}. Raises ConfidenceBalancingError when any cell is empty: a confident group made of one
    pole only would make the control a pole detector, not a confidence detector."""
    cells = {(group, pole): [] for group in ("confident", "hedged") for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE)}
    for answer in sorted(answers, key=lambda a: a["answer_id"]):
        group = confidence_group_of(stated_confidence_by_answer_id[answer["answer_id"]])
        if group is not None:
            cells[(group, answer["pole"])].append(answer["answer_id"])
    sizes_before = {f"{group}__{pole}": len(ids) for (group, pole), ids in cells.items()}
    smallest = min(sizes_before.values())
    if smallest == 0:
        raise ConfidenceBalancingError(
            f"The confidence vector cannot be balanced: cell sizes {sizes_before}. A group with no answer of one pole would make the "
            f"confidence vector a split by pole.")
    generator = np.random.default_rng(seed)
    chosen = {}
    for key, ids in sorted(cells.items()):
        chosen[key] = sorted(generator.choice(ids, size=smallest, replace=False).tolist()) if len(ids) > smallest else list(ids)
    groups = {group: sorted(chosen[(group, defs.POLE_POSITIVE)] + chosen[(group, defs.POLE_NEGATIVE)]) for group in ("confident", "hedged")}
    result = {**groups, "cell_sizes_before": sizes_before, "answers_per_cell": smallest}
    assert_confidence_groups_are_not_confounded_with_pole(result, {a["answer_id"]: a["pole"] for a in answers})
    return result


def assert_confidence_groups_are_not_confounded_with_pole(groups, pole_by_answer_id):
    """Raises ConfidenceBalancingError unless the confident and hedged groups each hold equal numbers of positive-pole and negative-pole
    answers (and the same number overall). The guard behind the confidence vector: it is also what a test feeds a confounded split."""
    shares = {}
    for name in ("confident", "hedged"):
        poles = [pole_by_answer_id[answer_id] for answer_id in groups[name]]
        positive = sum(1 for pole in poles if pole == defs.POLE_POSITIVE)
        negative = len(poles) - positive
        shares[name] = (positive, negative)
        if positive != negative or positive == 0:
            raise ConfidenceBalancingError(f"The {name} group holds {positive} positive-pole and {negative} negative-pole answers; the "
                                           f"two poles must be equally represented in each group")
    if shares["confident"] != shares["hedged"]:
        raise ConfidenceBalancingError(f"The confident and hedged groups differ in size or pole mix: {shares}")


def difference_of_group_means(store, groups, stream_rows=None):
    """Per layer, the mean activation of the confident group minus that of the hedged group, streamed from the activation store
    (`store.stream_rows(answer ids)` yields (answer id, row))."""
    sums = {}
    for name in ("confident", "hedged"):
        total, count = None, 0
        for _, row in store.stream_rows(list(groups[name])):
            row64 = row.astype(np.float64)
            total = row64 if total is None else total + row64
            count += 1
        if count == 0:
            raise ConfidenceBalancingError(f"No activations for the {name} group")
        sums[name] = total / count
    return (sums["confident"] - sums["hedged"]).astype(np.float32)


def project_direction_out(vector, direction):
    """`vector` (layers, hidden) with its component along `direction` (same shape, one direction per layer) removed, per layer."""
    vector = np.asarray(vector, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    unit = direction / np.maximum(norm, 1e-30)
    return (vector - np.sum(vector * unit, axis=-1, keepdims=True) * unit).astype(np.float32)


# ---------------------------------------------------------------------------------------------------
# The two-by-two held-out check
# ---------------------------------------------------------------------------------------------------

def two_by_two_cell_of(stated_confidence, band_group):
    """One of the four cells from the answer's judged stated confidence and its question's evidence band, or None (moderate confidence,
    or a strong-but-debated or mixed band). Pole plays no part: that is the point of the check."""
    group = confidence_group_of(stated_confidence)
    if group is None or band_group not in (facet_definitions_module.BAND_ESTABLISHED, facet_definitions_module.BAND_WEAK):
        return None
    established = band_group == facet_definitions_module.BAND_ESTABLISHED
    return {("confident", True): CELL_CONFIDENT_ESTABLISHED, ("hedged", True): CELL_HEDGED_ESTABLISHED,
            ("confident", False): CELL_CONFIDENT_WEAK, ("hedged", False): CELL_HEDGED_WEAK}[(group, established)]


def cell_means_and_contrasts(projection_by_answer_id, cell_by_answer_id):
    """({cell: mean projection or None}, interaction contrast, main effect of confidence). Interaction contrast = mean of the two
    calibrated cells' means minus mean of the two miscalibrated cells' means; main effect of confidence = mean of the two confident cells'
    means minus mean of the two hedged cells' means. None where a needed cell is empty."""
    values = {cell: [] for cell in ALL_CELLS}
    for answer_id, cell in cell_by_answer_id.items():
        if cell is not None and answer_id in projection_by_answer_id:
            values[cell].append(projection_by_answer_id[answer_id])
    means = {cell: (float(np.mean(v)) if v else None) for cell, v in values.items()}
    if any(means[cell] is None for cell in ALL_CELLS):
        return means, None, None
    interaction = (means[CELL_CONFIDENT_ESTABLISHED] + means[CELL_HEDGED_WEAK]) / 2 - (means[CELL_HEDGED_ESTABLISHED] + means[CELL_CONFIDENT_WEAK]) / 2
    main_effect = (means[CELL_CONFIDENT_ESTABLISHED] + means[CELL_CONFIDENT_WEAK]) / 2 - (means[CELL_HEDGED_ESTABLISHED] + means[CELL_HEDGED_WEAK]) / 2
    return means, float(interaction), float(main_effect)


def two_by_two_check(projection_by_answer_id, cell_by_answer_id, question_id_by_answer_id, resamples=1000, seed=0):
    """Per-cell mean projection, the interaction contrast (calibrated cells minus miscalibrated cells) and the main effect of confidence,
    each with a 95 percent interval from a bootstrap that resamples QUESTIONS with replacement (answers of one question are not
    independent). Resamples in which a cell is empty are skipped and counted. `projection_by_answer_id`: {answer id: projection of
    one vector at one layer}."""
    means, interaction, main_effect = cell_means_and_contrasts(projection_by_answer_id, cell_by_answer_id)
    counts = {cell: sum(1 for c in cell_by_answer_id.values() if c == cell) for cell in ALL_CELLS}
    answers_by_question = {}
    for answer_id, cell in cell_by_answer_id.items():
        if cell is not None and answer_id in projection_by_answer_id:
            answers_by_question.setdefault(question_id_by_answer_id[answer_id], []).append(answer_id)
    questions = sorted(answers_by_question)
    generator = np.random.default_rng(seed)
    interactions, main_effects, skipped = [], [], 0
    for _ in range(resamples if questions else 0):
        drawn = generator.integers(0, len(questions), size=len(questions))
        picked_cells, picked_values, counter = {}, projection_by_answer_id, 0
        for index in drawn:
            for answer_id in answers_by_question[questions[index]]:
                picked_cells[counter] = cell_by_answer_id[answer_id]
                counter += 1
        values_by_counter = {}
        counter = 0
        for index in drawn:
            for answer_id in answers_by_question[questions[index]]:
                values_by_counter[counter] = picked_values[answer_id]
                counter += 1
        _, boot_interaction, boot_main = cell_means_and_contrasts(values_by_counter, picked_cells)
        if boot_interaction is None:
            skipped += 1
            continue
        interactions.append(boot_interaction)
        main_effects.append(boot_main)
    interval = lambda draws: ([float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if len(draws) >= 20 else None)  # noqa: E731
    return {"cell_mean_projection": means, "answers_per_cell": counts,
            "calibrated_cells": list(CALIBRATED_CELLS), "miscalibrated_cells": list(MISCALIBRATED_CELLS),
            "interaction_contrast_calibrated_minus_miscalibrated": interaction,
            "interaction_contrast_95_percent_bootstrap_interval": interval(interactions),
            "main_effect_of_confidence_confident_minus_hedged": main_effect,
            "main_effect_95_percent_bootstrap_interval": interval(main_effects),
            "bootstrap_resamples_used": len(interactions), "bootstrap_resamples_skipped_for_an_empty_cell": skipped,
            "bootstrap_unit": "questions resampled with replacement"}
