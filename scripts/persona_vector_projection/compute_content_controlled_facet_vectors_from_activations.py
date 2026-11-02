"""Computes a content-controlled facet's vectors (moralizing delivery, cares about welfare, calibrated consideration or
non-intrusive), locally on CPU, from the activation shards (whole-answer means and welfare passage means), the judge results and
the welfare passage spans. This module was `compute_moralizing_delivery_vectors_from_passage_activations.py`; that name still
works (it is an alias of this module) and `--facet` defaults to moralizing_delivery, so the old command is unchanged.

    python3 scripts/persona_vector_projection/compute_content_controlled_facet_vectors_from_activations.py \
        --run-folder <run folder> --facet <facet name> --vector-set-name <name>

Facet differences (content_controlled_facet_definitions.py). Moralizing delivery, cares about welfare and calibrated
consideration: the main vector uses the welfare passage means (the passage is where the content is used) and the whole-answer
vector is secondary. Non-intrusive: its trait is in the help, not in the welfare passage, so only the whole-answer means are used
(no spans file is read, no passage conditions apply) and the whole-answer vector is the main one; and the content check's
`completes_the_task` is NOT used in its filter (incomplete help is its negative pole). Scale direction, every facet: the facet
score gives 100 to the positive pole and 0 to the negative pole; the vector is positive minus negative.

Items. Positive rewording n and negative rewording n of one question, same sample index, make one item. An item is KEPT
only when both answers pass every answer condition (the content check says it makes the point, exactly one separate
welfare mention, no added welfare facts, and completes the task; coherence at least 50; a welfare passage span was
found and covers at least one token) and the positive delivery score is at least 65 and the negative at most 35, and the
two passages' character lengths are within a factor of 1.5. Every threshold is a command-line option.

Vectors. Two vector sets, with NO topic removal: from the welfare passage means (main) and from the whole-answer means
(secondary). Each is, per question, the mean over kept items of (positive minus negative), then the mean over domains
with domains weighted equally (same functions as the other facets). Only extraction halves A and B build vectors; the
held-out half (6 questions) chooses the layer by classification accuracy, as the other facets do, and is also scored
separately for each negative tag (the moralizing device).

Largest-mean directions, recorded for later removal. From ALL answers' activations (whole-answer means and passage means
separately), per layer: the mean vector and per-dimension standard deviation; the 32 dimensions with the largest absolute
mean (mean, standard deviation, the vector's component); vector variants with the top k of those zeroed (k = 1, 2, 4, 8,
16, 32); and a standardised variant, the difference of means divided by the per-dimension variance (the raw-space
direction equivalent to taking the difference on standardised activations). Mean projection of each pole and of all
answers onto each variant (unit-normalised) anchor the scale. Held-out accuracy and area under the curve are reported for
every variant at the chosen layer and at layers 12, 18 and 24 (and at every layer in the arrays).

Outputs in <run folder>/vector_sets/<vector set name>_<date and time with seconds>/ (a new folder per run; a second run
in the same second fails rather than overwriting). Files end with the minute timestamp <ts>:
  persona_vector_moralizing_delivery_<space>_<ts>.npy                  float32 (layers, hidden); space is passage_mean or whole_answer_mean
  persona_vector_moralizing_delivery_<space>_<variant>_<ts>.npy        variants: top_<k>_largest_mean_dimensions_zeroed, standardised_by_per_dimension_variance
  activation_mean_and_standard_deviation_per_layer_<space>_<ts>.npz    mean, standard_deviation (layers, hidden), answers_used
  largest_absolute_mean_dimensions_per_layer_<space>_<ts>.json
  mean_projection_onto_each_variant_<space>_<ts>.json                  per variant and layer: positive pole, negative pole, all answers, vector norm
  answer_filter_conditions_<ts>.jsonl                                  per answer: the conditions it fails
  persona_vector_metadata_<ts>.json                                    filter counts by pole and tag, split-half cosine, bootstrap
                                                                       stability, held-out accuracy and area under the curve per layer, layer choice
"""

import argparse
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import calibrated_consideration_confidence_controls as confidence_controls  # noqa: E402
from persona_vector_projection import compute_persona_vectors_from_activations_and_judgements as shared_compute  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import persona_vector_math as vector_math  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now, timestamp_now_with_seconds  # noqa: E402
from shared.timed_progress_reporter import format_duration  # noqa: E402

FACET = defs.FACET_MORALIZING_DELIVERY  # the default facet; every function that needs a facet takes `facet_key=FACET`
SPACE_PASSAGE = "passage_mean"
SPACE_WHOLE_ANSWER = "whole_answer_mean"
SPACES = [SPACE_PASSAGE, SPACE_WHOLE_ANSWER]
VARIANT_RAW = "raw"
VARIANT_STANDARDISED = "standardised_by_per_dimension_variance"
ZEROED_DIMENSION_COUNTS = [1, 2, 4, 8, 16, 32]
LARGEST_MEAN_DIMENSIONS_RECORDED = 32
LAYERS_REPORTED_BESIDE_THE_CHOSEN_ONE = [12, 18, 24]
VARIANCE_FLOOR_FRACTION_OF_MEDIAN = 1e-6

CONDITION_DELIVERY_SCORE_MISSING = "delivery_score_missing"
CONDITION_CONTENT_CHECK_MISSING = "content_check_missing"
CONDITION_COHERENCE_MISSING = "coherence_missing"
CONDITION_DOES_NOT_MAKE_THE_POINT = "does_not_make_the_point"
CONDITION_MENTION_COUNT_WRONG = "separate_welfare_mentions_not_the_required_count"
CONDITION_ADDS_WELFARE_FACTS = "adds_welfare_facts"
CONDITION_DOES_NOT_COMPLETE_THE_TASK = "does_not_complete_the_task"
CONDITION_COHERENCE_BELOW_MINIMUM = "coherence_below_minimum"
CONDITION_NO_SPAN_FOUND = "no_welfare_passage_span_found"
CONDITION_NO_PASSAGE_MEAN = "no_passage_mean_activation"
CONDITION_NO_WHOLE_ANSWER_MEAN = "no_whole_answer_mean_activation"
CONDITION_POSITIVE_DELIVERY_BELOW_MINIMUM = "positive_delivery_score_below_minimum"
CONDITION_NEGATIVE_DELIVERY_ABOVE_MAXIMUM = "negative_delivery_score_above_maximum"
CONDITION_PASSAGE_LENGTH_RATIO = "passage_character_length_ratio_above_maximum"


# ---------------------------------------------------------------------------------------------------
# Answer conditions and items (pure functions)
# ---------------------------------------------------------------------------------------------------

def threshold_value(thresholds, new_key, old_key):
    """The facet-score threshold under its new key (`positive_facet_score_minimum`) or the moralizing delivery key it had."""
    return thresholds[new_key] if new_key in thresholds else thresholds[old_key]


def facet_score_condition_name(condition_name, facet_score_kind):
    """The condition names say 'delivery' only for moralizing delivery's delivery score; other facets say 'facet_score'."""
    if facet_score_kind == judge.KIND_DELIVERY_SCORE:
        return condition_name
    return condition_name.replace("delivery_score", "facet_score")


def failed_answer_conditions(pole, judge_scores, span, whole_answer_token_count, passage_token_count, thresholds,
                             facet_score_kind=judge.KIND_DELIVERY_SCORE, uses_task_completion_in_filter=True,
                             uses_welfare_passage=True):
    """The names of the conditions this single answer fails (an empty list: it passes them all). `judge_scores` holds the
    facet score under its kind (delivery_score, or content_controlled_facet_score), coherence (numbers or None) and
    content_check (dict or None); `span` is the span record or None. The facet score is 0 to 100 with 100 = the positive pole.
    Without the welfare passage (non-intrusive) the span and passage mean conditions do not apply, and without
    uses_task_completion_in_filter the content check's completes_the_task is ignored."""
    failed = []
    name_of = lambda condition: facet_score_condition_name(condition, facet_score_kind)  # noqa: E731
    delivery, coherence, content = (judge_scores.get(facet_score_kind), judge_scores.get("coherence"),
                                    judge_scores.get("content_check"))
    if delivery is None:
        failed.append(name_of(CONDITION_DELIVERY_SCORE_MISSING))
    if content is None:
        failed.append(CONDITION_CONTENT_CHECK_MISSING)
    if coherence is None:
        failed.append(CONDITION_COHERENCE_MISSING)
    if content is not None:
        if not content["makes_the_point"]:
            failed.append(CONDITION_DOES_NOT_MAKE_THE_POINT)
        if content["separate_welfare_mentions"] != thresholds["required_separate_welfare_mentions"]:
            failed.append(CONDITION_MENTION_COUNT_WRONG)
        if content["adds_welfare_facts"]:
            failed.append(CONDITION_ADDS_WELFARE_FACTS)
        if uses_task_completion_in_filter and not content["completes_the_task"]:
            failed.append(CONDITION_DOES_NOT_COMPLETE_THE_TASK)
    if coherence is not None and coherence < thresholds["coherence_minimum"]:
        failed.append(CONDITION_COHERENCE_BELOW_MINIMUM)
    if uses_welfare_passage and (span is None or span.get("character_start") is None):
        failed.append(CONDITION_NO_SPAN_FOUND)
    if uses_welfare_passage and passage_token_count <= 0:
        failed.append(CONDITION_NO_PASSAGE_MEAN)
    if whole_answer_token_count <= 0:
        failed.append(CONDITION_NO_WHOLE_ANSWER_MEAN)
    if delivery is not None:
        # the positive pole must score HIGH and the negative pole LOW, for every facet
        if pole == defs.POLE_POSITIVE and delivery < threshold_value(thresholds, "positive_facet_score_minimum", "positive_delivery_score_minimum"):
            failed.append(name_of(CONDITION_POSITIVE_DELIVERY_BELOW_MINIMUM))
        if pole == defs.POLE_NEGATIVE and delivery > threshold_value(thresholds, "negative_facet_score_maximum", "negative_delivery_score_maximum"):
            failed.append(name_of(CONDITION_NEGATIVE_DELIVERY_ABOVE_MAXIMUM))
    return failed


def passage_length_ratio_is_acceptable(positive_span, negative_span, maximum_ratio):
    lengths = [span["character_end"] - span["character_start"] for span in (positive_span, negative_span)]
    if min(lengths) <= 0:
        return False
    return max(lengths) / min(lengths) <= maximum_ratio


def group_answers_into_items(answers, facet_key=FACET):
    """{(question id, pairing slot, sample index): {pole: answer}} for extraction and held-out halves alike. The slot is the rewording
    number for every facet with as many negative as positive rewordings; calibrated consideration has twice as many negative ones
    (rewordings n and n + 5 share structure with positive rewording n), so a negative rewording pairs with the positive rewording of
    its slot. Two answers of one pole in one item are an error (a question given two negative rewordings of one slot)."""
    answers = [a for a in answers if a["facet"] == facet_key]
    positive_count = len({shared_compute.rewording_number_of(a) for a in answers if a["pole"] == defs.POLE_POSITIVE})
    items = defaultdict(dict)
    for answer in answers:
        number = shared_compute.rewording_number_of(answer)
        slot = facet_definitions_module.pairing_slot_of_rewording(number, positive_count) if positive_count else number
        key = (answer["question_id"], slot, answer["sample_index"])
        if answer["pole"] in items[key]:
            raise ValueError(f"Two {answer['pole']} answers share the item {key}: rewording {number} and "
                             f"{shared_compute.rewording_number_of(items[key][answer['pole']])}")
        items[key][answer["pole"]] = answer
    return items


def classify_items(items, conditions_by_answer_id, spans_by_answer_id, thresholds, uses_welfare_passage=True):
    """Splits the items of the extraction halves into kept and failing, with counts. Returns (kept items as a list of
    (key, positive answer, negative answer), filter counts)."""
    counts = {"items_total": 0, "items_with_missing_answers": 0, "items_failing_filter": 0, "items_kept": 0,
              "items_failing_each_condition": Counter()}
    kept = []
    for key, members in sorted(items.items()):
        any_answer = next(iter(members.values()))
        if any_answer["half"] not in defs.EXTRACTION_HALVES:
            continue
        counts["items_total"] += 1
        if defs.POLE_POSITIVE not in members or defs.POLE_NEGATIVE not in members:
            counts["items_with_missing_answers"] += 1
            continue
        positive, negative = members[defs.POLE_POSITIVE], members[defs.POLE_NEGATIVE]
        failing = set()
        for pole, answer in ((defs.POLE_POSITIVE, positive), (defs.POLE_NEGATIVE, negative)):
            failing.update(f"{pole}:{condition}" for condition in conditions_by_answer_id[answer["answer_id"]])
        if not failing and uses_welfare_passage and not passage_length_ratio_is_acceptable(
                spans_by_answer_id[positive["answer_id"]], spans_by_answer_id[negative["answer_id"]],
                thresholds["maximum_passage_length_ratio"]):
            failing.add(CONDITION_PASSAGE_LENGTH_RATIO)
        for condition in failing:
            counts["items_failing_each_condition"][condition] += 1
        if failing:
            counts["items_failing_filter"] += 1
        else:
            counts["items_kept"] += 1
            kept.append((key, positive, negative))
    counts["items_failing_each_condition"] = dict(sorted(counts["items_failing_each_condition"].items()))
    return kept, counts


def count_answer_conditions_by_pole_and_tag(answers, conditions_by_answer_id, facet_key=FACET):
    """{pole: {"all" or tag: {"answers": n, "passing_every_answer_condition": n, "failing": {condition: n}}}}. A
    condition is counted for every answer that fails it, so one answer can appear under several."""
    table = defaultdict(lambda: defaultdict(lambda: {"answers": 0, "passing_every_answer_condition": 0, "failing": Counter()}))
    for answer in answers:
        if answer["facet"] != facet_key:
            continue
        failed = conditions_by_answer_id[answer["answer_id"]]
        for group in {"all", answer["tag"] or "all"}:
            cell = table[answer["pole"]][group]
            cell["answers"] += 1
            cell["passing_every_answer_condition"] += 0 if failed else 1
            cell["failing"].update(failed)
    return {pole: {group: {**cell, "failing": dict(sorted(cell["failing"].items()))} for group, cell in sorted(groups.items())}
            for pole, groups in table.items()}


# ---------------------------------------------------------------------------------------------------
# Largest-mean directions and variants (pure functions)
# ---------------------------------------------------------------------------------------------------

def indices_of_largest_absolute_mean_per_layer(mean_vector, count):
    """(layers, count) dimension indices, largest absolute mean first, at every layer."""
    return np.argsort(-np.abs(mean_vector), axis=-1, kind="stable")[:, :count]


def zero_largest_mean_dimensions(vector, mean_vector, count):
    """Copy of `vector` (layers, hidden) with, at each layer, the `count` dimensions of largest absolute mean set to 0."""
    zeroed = np.array(vector, dtype=np.float32, copy=True)
    top = indices_of_largest_absolute_mean_per_layer(mean_vector, count)
    for layer in range(zeroed.shape[0]):
        zeroed[layer, top[layer]] = 0.0
    return zeroed


def standardise_by_per_dimension_variance(vector, variance):
    """The difference of means divided by the per-dimension variance: the direction in raw activation space that equals the
    difference of means on standardised activations. A floor (a millionth of the layer's median variance) stops a
    near-constant dimension from exploding."""
    variance = np.asarray(variance, dtype=np.float64)
    floor = VARIANCE_FLOOR_FRACTION_OF_MEDIAN * np.median(variance, axis=-1, keepdims=True)
    return (np.asarray(vector, dtype=np.float64) / np.maximum(variance, np.maximum(floor, 1e-30))).astype(np.float32)


def build_vector_variants(vector, mean_vector, variance):
    variants = {VARIANT_RAW: np.asarray(vector, dtype=np.float32)}
    for count in ZEROED_DIMENSION_COUNTS:
        variants[f"top_{count}_largest_mean_dimensions_zeroed"] = zero_largest_mean_dimensions(vector, mean_vector, count)
    variants[VARIANT_STANDARDISED] = standardise_by_per_dimension_variance(vector, variance)
    return variants


def largest_mean_dimension_table(mean_vector, standard_deviation, vector, count=LARGEST_MEAN_DIMENSIONS_RECORDED):
    """Per layer, the `count` dimensions with the largest absolute mean: index, mean, standard deviation and the vector's
    component on it."""
    top = indices_of_largest_absolute_mean_per_layer(mean_vector, count)
    table = []
    for layer in range(mean_vector.shape[0]):
        table.append([{"dimension": int(d), "mean": float(mean_vector[layer, d]), "standard_deviation": float(standard_deviation[layer, d]),
                       "vector_component": float(vector[layer, d])} for d in top[layer]])
    return table


# ---------------------------------------------------------------------------------------------------
# Streaming over activations
# ---------------------------------------------------------------------------------------------------

def mean_and_variance_over_answers(store, answer_ids):
    """Per layer and dimension, the mean and population variance over the answers (float64 sums, one shard in memory)."""
    total, total_of_squares, count = None, None, 0
    for _, row in store.stream_rows(list(answer_ids)):
        row64 = row.astype(np.float64)
        total = row64 if total is None else total + row64
        total_of_squares = row64 ** 2 if total_of_squares is None else total_of_squares + row64 ** 2
        count += 1
    if count == 0:
        raise SystemExit("No answers with activations to take means over")
    mean = total / count
    return mean, np.maximum(total_of_squares / count - mean ** 2, 0.0), count


def projections_of_answers_onto_variants(store, answer_ids, unit_variants):
    """{variant: {answer_id: projection per layer}} (shared_compute.project_answers)."""
    return shared_compute.project_answers(store, list(answer_ids), unit_variants)


def metrics_from_projections(projections, fit_positive, fit_negative, held_positive, held_negative):
    """Accuracy and area under the curve per layer for one variant, using shared threshold and metric functions."""
    stack = lambda ids: np.stack([projections[i] for i in ids])  # noqa: E731
    thresholds = vector_math.midpoint_thresholds(stack(fit_positive), stack(fit_negative))
    return {"accuracy_per_layer": vector_math.classification_accuracy_per_layer(stack(held_positive), stack(held_negative), thresholds),
            "area_under_curve_per_layer": vector_math.area_under_curve_per_layer(stack(held_positive), stack(held_negative)),
            "thresholds_per_layer": thresholds}


def cross_half_accuracy(store, fit_half_vector, ids_by_half_and_pole, fit_half, evaluation_half):
    """Cross-half accuracy per layer: `fit_half_vector` (layers, hidden) was built from the kept items of `fit_half` ALONE; its decision
    threshold (per layer, the midpoint of the mean projections of the fit half's positive-pole and negative-pole answers) comes from the
    fit half's answers alone; the accuracy is measured on `evaluation_half`'s answers alone, which neither the vector nor the threshold
    has seen. `ids_by_half_and_pole`: {half: {pole: answer ids that pass every answer condition}}. Returns {"accuracy_per_layer",
    "fit_half", "evaluation_half", "answers_in_evaluation_half": {pole: count}} or None when a half has no vector or no answer of a pole."""
    fit, evaluation = ids_by_half_and_pole[fit_half], ids_by_half_and_pole[evaluation_half]
    poles = (defs.POLE_POSITIVE, defs.POLE_NEGATIVE)
    if fit_half_vector is None or not all(fit[pole] and evaluation[pole] for pole in poles):
        return None
    unit = vector_math.unit_vectors_per_layer(fit_half_vector)
    projections = shared_compute.project_answers(store, [i for pole in poles for i in fit[pole] + evaluation[pole]], {"half": unit})["half"]
    metrics = metrics_from_projections(projections, fit[defs.POLE_POSITIVE], fit[defs.POLE_NEGATIVE],
                                       evaluation[defs.POLE_POSITIVE], evaluation[defs.POLE_NEGATIVE])
    return {"accuracy_per_layer": rounded_list(metrics["accuracy_per_layer"]), "fit_half": fit_half, "evaluation_half": evaluation_half,
            "answers_in_evaluation_half": {pole: len(evaluation[pole]) for pole in poles}}


# ---------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------

def load_answer_judgements(run_folder):
    """{answer_id: {facet score kind, "coherence", "content_check"}}; discarded or missing values are None."""
    scores = defaultdict(dict)
    for record in judge.load_latest_judge_records_by_request_id(run_folder).values():
        if not record.get("answer_id"):
            continue
        if record["kind"] == judge.KIND_CONTENT_CHECK:
            scores[record["answer_id"]]["content_check"] = None if record["discarded"] else record["content_check"]
        elif record["kind"] in (judge.KIND_DELIVERY_SCORE, judge.KIND_CONTENT_CONTROLLED_FACET_SCORE, judge.KIND_COHERENCE,
                                judge.KIND_STATED_CONFIDENCE):
            scores[record["answer_id"]][record["kind"]] = None if record["discarded"] else record["score"]
    return scores


def load_spans(run_folder):
    path = defs.find_latest_file(Path(run_folder) / defs.INPUTS_FOLDER_NAME, defs.WELFARE_PASSAGE_SPANS_FILE_STEM, "jsonl")
    return {record["answer_id"]: record for record in read_jsonl(path)}, path.name


def rounded_list(array, digits=5):
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


# ---------------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------------

def differences_for_items(kept_items, store, answers_by_id):
    """Per-question mean of (positive - negative) over the kept items, in the activation space of `store`, in the same
    structure as shared_compute.compute_question_differences (so domain_weighted_vector and the bootstrap apply)."""
    roles, kept_count = {}, defaultdict(int)
    info = {}
    for (question_id, _, _), positive, negative in kept_items:
        kept_count[question_id] += 1
        info[question_id] = (positive["domain"], positive["half"])
        roles[positive["answer_id"]] = (question_id, "positive")
        roles[negative["answer_id"]] = (question_id, "negative")
    # one answer can belong to only one item, so roles never collide
    sums = {}
    shared_compute.accumulate_sums(store, sums, roles)
    keys = sorted(kept_count)
    vectors = np.stack([(sums[(q, "positive")] - sums[(q, "negative")]) / kept_count[q] for q in keys]) if keys else None
    return {"question_keys": keys, "vectors": vectors, "domains": [info[q][0] for q in keys], "halves": [info[q][1] for q in keys],
            "kept_items_per_question": [kept_count[q] for q in keys]}


def compute_all(run_folder, thresholds, vector_set_name, bootstrap_resamples=200, minimum_layer_index=1, seed=0,
                output_folder=None, say=print, facet_key=FACET,
                maximum_pole_stated_confidence_difference=confidence_controls.DEFAULT_MAXIMUM_POLE_DIFFERENCE_IN_MEAN_STATED_CONFIDENCE):
    run_folder = Path(run_folder)
    definition = facet_definitions_module.get_facet_definition(facet_key)
    main_space = definition.main_space
    spaces = definition.vector_spaces  # both spaces, or the whole-answer mean alone for non-intrusive
    uses_passage = definition.uses_welfare_passage
    started = time.time()
    stamp = timestamp_now()
    if output_folder is None:
        output_folder = run_folder / defs.VECTOR_SETS_FOLDER_NAME / f"{vector_set_name}_{timestamp_now_with_seconds()}"
        output_folder.mkdir(parents=True, exist_ok=False)  # refuses, never overwrites, a second run in the same second
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=False)
    activations_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    whole_store = shared_compute.ActivationStore(activations_folder)
    # no passage store for a facet whose main mean is the whole-answer mean: the shards need not hold passage means
    passage_store = (shared_compute.ActivationStore(activations_folder, "passage_activations", "passage_token_counts")
                     if uses_passage else None)
    stores = {SPACE_PASSAGE: passage_store, SPACE_WHOLE_ANSWER: whole_store} if uses_passage else {SPACE_WHOLE_ANSWER: whole_store}
    main_store = stores[main_space]
    answers = [a for a in shared_compute.load_answers(run_folder) if a["facet"] == facet_key]
    answers_by_id = {a["answer_id"]: a for a in answers}
    judgements = load_answer_judgements(run_folder)
    spans, spans_file_name = load_spans(run_folder) if uses_passage else ({}, None)
    say(f"Loaded {len(answers)} answers, {len(judgements)} judged answers, {len(spans)} span records ({spans_file_name})")

    conditions = {a["answer_id"]: failed_answer_conditions(
        a["pole"], judgements.get(a["answer_id"], {}), spans.get(a["answer_id"]),
        whole_store.answer_token_counts.get(a["answer_id"], 0),
        passage_store.answer_token_counts.get(a["answer_id"], 0) if uses_passage else 0, thresholds,
        facet_score_kind=definition.facet_score_kind, uses_task_completion_in_filter=definition.uses_task_completion_in_filter,
        uses_welfare_passage=uses_passage)
        for a in answers}
    confidence_enabled = definition.has_confidence_control
    questions_by_id = band_group_by_question_id = stated_confidence_by_answer_id = None
    if confidence_enabled:  # calibrated consideration: stated confidence must exist; a rewording accurate on its band is rejected
        questions_by_id = {q["question_id"]: q for q in read_jsonl(defs.find_latest_file(run_folder / defs.INPUTS_FOLDER_NAME, "questions", "jsonl"))
                           if q["facet"] == facet_key}
        band_group_by_question_id = {question_id: confidence_controls.band_group_of_question(q) for question_id, q in questions_by_id.items()}
        for a in answers:
            conditions[a["answer_id"]] = conditions[a["answer_id"]] + confidence_controls.stated_confidence_condition_failures(
                a, judgements.get(a["answer_id"], {}), questions_by_id[a["question_id"]], definition)
        stated_confidence_by_answer_id = {a["answer_id"]: judgements.get(a["answer_id"], {}).get(judge.KIND_STATED_CONFIDENCE) for a in answers}
    with open(output_folder / f"answer_filter_conditions_{stamp}.jsonl", "w", encoding="utf-8") as handle:
        for a in answers:
            handle.write(json.dumps({"answer_id": a["answer_id"], "pole": a["pole"], "tag": a["tag"], "half": a["half"],
                                     "failed_conditions": conditions[a["answer_id"]]}) + "\n")
    items = group_answers_into_items(answers, facet_key)
    kept_items, filter_counts = classify_items(items, conditions, spans, thresholds, uses_passage)
    say(f"Items: {filter_counts['items_kept']} kept of {filter_counts['items_total']}")
    balancing_report = None
    if confidence_enabled and kept_items:
        kept_items, balancing_report = confidence_controls.balance_items_on_stated_confidence(
            kept_items, stated_confidence_by_answer_id, band_group_by_question_id, maximum_pole_stated_confidence_difference)
        filter_counts["items_removed_by_balancing_stated_confidence"] = filter_counts["items_kept"] - len(kept_items)
        filter_counts["items_kept_after_balancing_stated_confidence"] = len(kept_items)
        say(f"Balanced stated confidence between the poles: {len(kept_items)} items kept of {filter_counts['items_kept']}; mean difference "
            f"before {balancing_report['before_balancing']['overall']['difference_positive_minus_negative']:.1f}, after "
            f"{balancing_report['after_balancing']['overall']['difference_positive_minus_negative']:.1f}")
    if not kept_items:
        raise SystemExit("No item survived the filter; no vector can be computed. Read answer_filter_conditions and the "
                         "filter counts first: " + json.dumps(filter_counts["items_failing_each_condition"]))

    metadata = {"vector_set_name": vector_set_name, "timestamp": stamp, "run_folder": str(run_folder),
                "filter_thresholds": thresholds, "facets": [facet_key], "spans_file": spans_file_name, "per_facet": {facet_key: {}},
                "diagnostics": {}, "note": "no topic removal: this run has no topic control",
                "facet_score_scale": definition.facet_score_scale_label(), "high_score_pole": definition.high_score_pole,
                "vector_direction": "positive pole minus negative pole",
                "main_space": main_space, "spaces": spaces,
                "completes_the_task_used_in_filter": definition.uses_task_completion_in_filter}
    configuration = json.loads(sorted(run_folder.glob("run_configuration_*.json"))[0].read_text())
    metadata.update({"model_name": configuration["model_name"], "model_revision": configuration["model_revision"],
                     "activation_pass_system_prompt": configuration.get("activation_pass_system_prompt")})
    entry = metadata["per_facet"][facet_key]
    entry["filter_counts"] = filter_counts
    if balancing_report is not None:
        entry["stated_confidence_balancing"] = balancing_report
    entry["answer_condition_counts_by_pole_and_tag"] = count_answer_conditions_by_pole_and_tag(answers, conditions, facet_key)

    # ---- the two vector sets, with split-half and bootstrap diagnostics
    vectors, differences_by_space, half_vectors_by_space = {}, {}, {}
    for space, store in stores.items():
        differences = differences_for_items(kept_items, store, answers_by_id)
        differences_by_space[space] = differences
        vectors[space] = shared_compute.domain_weighted_vector(differences)
        space_entry = entry.setdefault(space, {})
        space_entry["questions_kept"] = len(differences["question_keys"])
        space_entry["questions_kept_by_half"] = {h: differences["halves"].count(h) for h in defs.EXTRACTION_HALVES}
        space_entry["domains_with_kept_questions"] = len(set(differences["domains"]))
        space_entry["norm_per_layer"] = rounded_list(np.linalg.norm(vectors[space], axis=-1))
        half_vectors = {h: shared_compute.domain_weighted_vector(differences, [h]) for h in defs.EXTRACTION_HALVES}
        half_vectors_by_space[space] = half_vectors
        if all(v is not None for v in half_vectors.values()):
            space_entry["split_half_cosine_half_A_against_half_B_per_layer"] = rounded_list(
                vector_math.cosine_per_layer(half_vectors["A"], half_vectors["B"]))
        if bootstrap_resamples and len(differences["question_keys"]) > 1:
            cosines = vector_math.bootstrap_cosine_stability(differences["vectors"], differences["domains"], bootstrap_resamples, seed)
            space_entry["bootstrap_cosine_with_full_vector_mean_per_layer"] = rounded_list(cosines.mean(axis=0))
            space_entry["bootstrap_cosine_with_full_vector_fifth_percentile_per_layer"] = rounded_list(np.percentile(cosines, 5, axis=0))
    entry["questions_kept"] = entry[main_space]["questions_kept"]
    # the denominator of the criterion "share of training questions with at least one kept item": the questions of extraction halves A and B
    entry["training_questions_total"] = len({a["question_id"] for a in answers if a["half"] in defs.EXTRACTION_HALVES})
    if uses_passage:
        metadata["diagnostics"]["cosine_between_passage_mean_and_whole_answer_mean_vectors_per_layer"] = rounded_list(
            vector_math.cosine_per_layer(vectors[SPACE_PASSAGE], vectors[SPACE_WHOLE_ANSWER]))
    for tag in sorted(set(definition.negative_tag_slugs)):  # cosine of each tag's own vector with the full main-space vector
        tag_items = [item for item in kept_items if item[2]["tag"] == tag]
        if tag_items:
            tag_vector = shared_compute.domain_weighted_vector(differences_for_items(tag_items, main_store, answers_by_id))
            entry.setdefault("negative_tag_kept_items", {})[tag] = len(tag_items)
            # the key keeps its moralizing-era name ("passage") when the main space is the passage mean
            tag_key = ("cosine_of_each_negative_tag_vector_with_full_passage_vector_per_layer" if uses_passage
                       else "cosine_of_each_negative_tag_vector_with_full_whole_answer_vector_per_layer")
            entry.setdefault(tag_key, {})[tag] = rounded_list(vector_math.cosine_per_layer(tag_vector, vectors[main_space]))

    # ---- all-answer statistics, variants, projections
    answer_ids_with_mean = {SPACE_WHOLE_ANSWER: [a["answer_id"] for a in answers if whole_store.answer_token_counts.get(a["answer_id"], 0) > 0]}
    if uses_passage:
        answer_ids_with_mean[SPACE_PASSAGE] = [a["answer_id"] for a in answers if passage_store.answer_token_counts.get(a["answer_id"], 0) > 0]
    fit_ids = {pole: [a["answer_id"] for a in answers if a["pole"] == pole and a["half"] in defs.EXTRACTION_HALVES and not conditions[a["answer_id"]]]
               for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE)}
    held_ids = {pole: [a["answer_id"] for a in answers if a["pole"] == pole and a["half"] == defs.HALF_HELD_OUT and not conditions[a["answer_id"]]]
                for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE)}
    held_negative_by_tag = defaultdict(list)
    for answer_id in held_ids[defs.POLE_NEGATIVE]:
        held_negative_by_tag[answers_by_id[answer_id]["tag"]].append(answer_id)
    have_held_out = all(fit_ids[p] and held_ids[p] for p in fit_ids)
    entry["held_out_answers_used"] = {"positive": len(held_ids[defs.POLE_POSITIVE]), "negative": len(held_ids[defs.POLE_NEGATIVE]),
                                      "negative_by_tag": {t: len(v) for t, v in sorted(held_negative_by_tag.items())},
                                      "fit_answers_positive": len(fit_ids[defs.POLE_POSITIVE]),
                                      "fit_answers_negative": len(fit_ids[defs.POLE_NEGATIVE])}
    variants_by_space, accuracy_raw = {}, {}
    file_names = {}
    for space, store in stores.items():
        mean_vector, variance, used = mean_and_variance_over_answers(store, answer_ids_with_mean[space])
        standard_deviation = np.sqrt(variance)
        np.savez(output_folder / f"activation_mean_and_standard_deviation_per_layer_{space}_{stamp}.npz",
                 mean=mean_vector.astype(np.float32), standard_deviation=standard_deviation.astype(np.float32), answers_used=used)
        variants = build_vector_variants(vectors[space], mean_vector, variance)
        variants_by_space[space] = variants
        (output_folder / f"largest_absolute_mean_dimensions_per_layer_{space}_{stamp}.json").write_text(json.dumps(
            {"space": space, "answers_used": used, "dimensions_per_layer": largest_mean_dimension_table(
                mean_vector, standard_deviation, vectors[space])}, indent=1), encoding="utf-8")
        for variant, vector in variants.items():
            name = (f"persona_vector_{facet_key}_{space}_{stamp}.npy" if variant == VARIANT_RAW
                    else f"persona_vector_{facet_key}_{space}_{variant}_{stamp}.npy")
            np.save(output_folder / name, vector.astype(np.float32))
            file_names[f"{space}_{variant}"] = name
        unit_variants = {v: vector_math.unit_vectors_per_layer(vec) for v, vec in variants.items()}
        projections = projections_of_answers_onto_variants(store, answer_ids_with_mean[space], unit_variants)
        pole_of = {a["answer_id"]: a["pole"] for a in answers}
        anchors, metrics_by_variant = {}, {}
        for variant in variants:
            table = projections[variant]
            by_pole = {p: [i for i in answer_ids_with_mean[space] if pole_of[i] == p] for p in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE)}
            mean_of = lambda ids: rounded_list(np.mean([table[i] for i in ids], axis=0), 4) if ids else None  # noqa: E731
            anchors[variant] = {"vector_norm_per_layer": rounded_list(np.linalg.norm(variants[variant], axis=-1), 4),
                                "mean_projection_positive_pole_answers_per_layer": mean_of(by_pole[defs.POLE_POSITIVE]),
                                "mean_projection_negative_pole_answers_per_layer": mean_of(by_pole[defs.POLE_NEGATIVE]),
                                "mean_projection_all_answers_per_layer": mean_of(answer_ids_with_mean[space]),
                                "answers_projected": len(answer_ids_with_mean[space])}
            if have_held_out:
                metrics = metrics_from_projections(table, fit_ids["positive"], fit_ids["negative"], held_ids["positive"], held_ids["negative"])
                per_tag = {}
                for tag, tag_ids in sorted(held_negative_by_tag.items()):
                    tag_metrics = metrics_from_projections(table, fit_ids["positive"], fit_ids["negative"], held_ids["positive"], tag_ids)
                    negatives_correct = np.mean(np.stack([table[i] for i in tag_ids]) <= metrics["thresholds_per_layer"], axis=0)
                    per_tag[tag] = {"held_out_negative_answers": len(tag_ids),
                                    "accuracy_per_layer": rounded_list(tag_metrics["accuracy_per_layer"]),
                                    "negatives_classified_correctly_per_layer": rounded_list(negatives_correct),
                                    "area_under_curve_per_layer": rounded_list(tag_metrics["area_under_curve_per_layer"])}
                metrics_by_variant[variant] = {"held_out_accuracy_per_layer": rounded_list(metrics["accuracy_per_layer"]),
                                               "held_out_area_under_curve_per_layer": rounded_list(metrics["area_under_curve_per_layer"]),
                                               "by_negative_tag": per_tag}
                if variant == VARIANT_RAW:
                    accuracy_raw[space] = metrics["accuracy_per_layer"]
        (output_folder / f"mean_projection_onto_each_variant_{space}_{stamp}.json").write_text(json.dumps(
            {"space": space, "note": "projections onto the unit-normalised variant, per layer; the vector norm is recorded beside them",
             "variants": anchors}, indent=1), encoding="utf-8")
        entry[space]["variant_held_out_metrics"] = metrics_by_variant
        entry[space]["variant_names"] = list(variants)
        # cross-half accuracy of the raw vector: the counted direction is the vector from half A's items on half B's questions; the
        # reverse (half B's vector on half A's questions) is recorded beside it and only reported
        with_mean = set(answer_ids_with_mean[space])
        ids_by_half_and_pole = {half: {pole: [a["answer_id"] for a in answers if a["half"] == half and a["pole"] == pole
                                              and not conditions[a["answer_id"]] and a["answer_id"] in with_mean]
                                       for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE)} for half in defs.EXTRACTION_HALVES}
        half_a, half_b = defs.EXTRACTION_HALVES[0], defs.EXTRACTION_HALVES[1]
        entry[space]["cross_half_accuracy"] = {
            "vector_from_half_A_evaluated_on_half_B": cross_half_accuracy(store, half_vectors_by_space[space][half_a], ids_by_half_and_pole, half_a, half_b),
            "vector_from_half_B_evaluated_on_half_A_reported_only": cross_half_accuracy(store, half_vectors_by_space[space][half_b], ids_by_half_and_pole, half_b, half_a)}
    # ---- calibrated consideration: the plain-confidence control vector, its cosine, and projecting its direction out
    confidence_artifacts = None
    if confidence_enabled:
        kept_answers = [answers_by_id[answer["answer_id"]] for _, positive, negative in kept_items for answer in (positive, negative)]
        groups = confidence_controls.select_balanced_confidence_groups(kept_answers, stated_confidence_by_answer_id, seed)
        confidence_vector = confidence_controls.difference_of_group_means(main_store, groups)
        vectors_main = vectors[main_space]
        projected_out_vector = confidence_controls.project_direction_out(vectors_main, confidence_vector)
        name = f"persona_vector_{facet_key}_stated_confidence_{main_space}_{stamp}.npy"
        np.save(output_folder / name, confidence_vector.astype(np.float32))
        file_names[f"stated_confidence_{main_space}_raw"] = name
        name = f"persona_vector_{facet_key}_{main_space}_confidence_direction_projected_out_{stamp}.npy"
        np.save(output_folder / name, projected_out_vector.astype(np.float32))
        file_names[f"{main_space}_confidence_direction_projected_out"] = name
        unit = {"calibration": vector_math.unit_vectors_per_layer(vectors_main), "confidence": vector_math.unit_vectors_per_layer(confidence_vector),
                "calibration_confidence_direction_projected_out": vector_math.unit_vectors_per_layer(projected_out_vector)}
        control_projections = projections_of_answers_onto_variants(main_store, answer_ids_with_mean[main_space], unit)
        control_entry = {"confidence_groups": {"confident_answers": len(groups["confident"]), "hedged_answers": len(groups["hedged"]),
                                               "answers_per_group_and_pole_cell": groups["answers_per_cell"],
                                               "cell_sizes_before_subsampling": groups["cell_sizes_before"],
                                               "confident_minimum": confidence_controls.CONFIDENT_MINIMUM,
                                               "hedged_maximum": confidence_controls.HEDGED_MAXIMUM,
                                               "built_from": "answers of the kept items after balancing stated confidence",
                                               "unit": "per-answer mean over the group, confident minus hedged"},
                         "cosine_calibration_vector_with_confidence_vector_per_layer": rounded_list(
                             vector_math.cosine_per_layer(vectors_main, confidence_vector)),
                         "vector_norm_per_layer": rounded_list(np.linalg.norm(confidence_vector, axis=-1), 4)}
        if have_held_out:
            for label in ("calibration", "calibration_confidence_direction_projected_out", "confidence"):
                metrics = metrics_from_projections(control_projections[label], fit_ids["positive"], fit_ids["negative"],
                                                   held_ids["positive"], held_ids["negative"])
                control_entry[f"held_out_pole_classification_{label}"] = {
                    "accuracy_per_layer": rounded_list(metrics["accuracy_per_layer"]),
                    "area_under_curve_per_layer": rounded_list(metrics["area_under_curve_per_layer"])}
        entry["confidence_control"] = control_entry
        confidence_artifacts = {"projections": control_projections}
    # ---- layer choice (passage mean, raw vector) and the reported layers
    if main_space in accuracy_raw:
        choice = vector_math.choose_layers({facet_key: accuracy_raw[main_space]}, minimum_layer_index)
        choice["based_on"] = f"held-out classification accuracy of the {main_space.replace('_', '-')} vector (no topic removal)"
        metadata["layer_choice"] = choice
        chosen = choice["layer_by_facet"][facet_key]
        reported_layers = sorted({chosen, *[l for l in LAYERS_REPORTED_BESIDE_THE_CHOSEN_ONE if l < len(accuracy_raw[main_space])]})
        summary = {}
        for space in spaces:
            summary[space] = {variant: {"held_out_accuracy": {str(l): metrics["held_out_accuracy_per_layer"][l] for l in reported_layers},
                                        "held_out_area_under_curve": {str(l): metrics["held_out_area_under_curve_per_layer"][l] for l in reported_layers}}
                              for variant, metrics in entry[space]["variant_held_out_metrics"].items()}
        entry["variant_summary_at_chosen_and_reported_layers"] = {"layers_reported": reported_layers, "chosen_layer": chosen, "by_space": summary}
        if confidence_enabled:  # the two-by-two held-out check, at the chosen layer and the other reported layers
            held_answer_ids = held_ids[defs.POLE_POSITIVE] + held_ids[defs.POLE_NEGATIVE]
            cell_by_answer_id = {answer_id: confidence_controls.two_by_two_cell_of(
                stated_confidence_by_answer_id[answer_id], band_group_by_question_id[answers_by_id[answer_id]["question_id"]])
                for answer_id in held_answer_ids}
            question_by_answer_id = {answer_id: answers_by_id[answer_id]["question_id"] for answer_id in held_answer_ids}
            two_by_two = {}
            for layer in reported_layers:
                two_by_two[str(layer)] = {label: confidence_controls.two_by_two_check(
                    {answer_id: float(confidence_artifacts["projections"][label][answer_id][layer]) for answer_id in held_answer_ids},
                    cell_by_answer_id, question_by_answer_id, resamples=bootstrap_resamples * 5, seed=seed)
                    for label in ("calibration", "calibration_confidence_direction_projected_out", "confidence")}
            entry["confidence_control"]["two_by_two_held_out_check"] = {
                "layers_reported": reported_layers, "chosen_layer": chosen, "by_layer_and_vector": two_by_two,
                "expectation": "calibration vector: interaction contrast above zero, main effect of confidence near zero; confidence "
                               "vector: main effect above zero, interaction contrast near zero"}
            say("Two-by-two held-out check done at layer " + str(chosen))
    else:
        metadata["layer_choice"] = None
        say("No held-out answers passed every condition; the layer choice and held-out metrics are missing")
    metadata["vector_files"] = file_names
    metadata["shape"] = {"layers_including_embedding_output": int(vectors[main_space].shape[0]), "hidden": int(vectors[main_space].shape[1])}
    metadata["elapsed_seconds"] = round(time.time() - started, 1)
    metadata["peak_resident_memory_megabytes"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3), 1)
    metadata_path = output_folder / f"persona_vector_metadata_{stamp}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    say(f"Wrote {metadata_path} in {format_duration(metadata['elapsed_seconds'])}")
    return metadata, metadata_path


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    facet_definitions_module.add_facet_argument(parser)
    parser.add_argument("--vector-set-name", default=None, help="Default: <facet key>_vectors_unnamed")
    parser.add_argument("--positive-delivery-score-minimum", "--positive-facet-score-minimum", dest="positive_delivery_score_minimum",
                        type=float, default=65.0, help="the positive pole's facet score must be at least this (100 = positive pole)")
    parser.add_argument("--negative-delivery-score-maximum", "--negative-facet-score-maximum", dest="negative_delivery_score_maximum",
                        type=float, default=35.0, help="the negative pole's facet score must be at most this (0 = negative pole)")
    parser.add_argument("--coherence-minimum", type=float, default=defs.DEFAULT_COHERENCE_MINIMUM)
    parser.add_argument("--maximum-passage-length-ratio", type=float, default=1.5)
    parser.add_argument("--required-separate-welfare-mentions", type=int, default=1)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--maximum-pole-stated-confidence-difference", type=float,
                        default=confidence_controls.DEFAULT_MAXIMUM_POLE_DIFFERENCE_IN_MEAN_STATED_CONFIDENCE,
                        help="Calibrated consideration only: the poles' mean stated confidence may differ by at most this many points "
                             "(overall and in every evidence band) after items are subsampled")
    parser.add_argument("--minimum-layer-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    definition = facet_definitions_module.get_facet_definition(arguments.facet)
    score_key = "delivery_score" if definition.facet_score_kind == judge.KIND_DELIVERY_SCORE else "facet_score"
    thresholds = {f"positive_{score_key}_minimum": arguments.positive_delivery_score_minimum,
                  f"negative_{score_key}_maximum": arguments.negative_delivery_score_maximum,
                  "coherence_minimum": arguments.coherence_minimum,
                  "maximum_passage_length_ratio": arguments.maximum_passage_length_ratio,
                  "required_separate_welfare_mentions": arguments.required_separate_welfare_mentions}
    compute_all(arguments.run_folder, thresholds, arguments.vector_set_name or f"{definition.facet_key}_vectors_unnamed",
                arguments.bootstrap_resamples, arguments.minimum_layer_index, arguments.seed, facet_key=definition.facet_key,
                maximum_pole_stated_confidence_difference=arguments.maximum_pole_stated_confidence_difference)
    return 0


if __name__ == "__main__":
    sys.exit(main())
