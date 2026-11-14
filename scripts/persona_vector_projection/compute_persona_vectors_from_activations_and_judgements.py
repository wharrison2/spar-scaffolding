"""Computes the persona vectors, locally on CPU, from the saved answer-token mean activations and the judge results.

    python3 scripts/persona_vector_projection/compute_persona_vectors_from_activations_and_judgements.py \
        --run-folder <run folder> [--vector-set-name <name>]

Steps per facet: filter (positive >= 65, negative <= 35, coherence >= 50 on both, animal shares within 15 points; for the
consistent facet all four answers of both versions must pass; every threshold is a command-line option); within each
question the mean of (positive - negative) over kept (rewording, sample) pairs; questions averaged with domains weighted
equally; consistent: the two versions averaged with equal weight; topic vector = animal minus non-animal answer, pair by
pair (also computed without answers flagged as volunteering welfare remarks, which is the one projected out); the topic
vector is projected out of every facet vector; non-intrusive (and any facet with tagged negative prompts) also gets one
vector per negative tag; layer chosen by held-out classification; diagnostics (norms, cosines, split-half, bootstrap).

Memory: the activations (about 5 GB in fp16 for a full run) are never loaded together. Shards are streamed one at a time
(np.load reads an .npz member only when asked for), per-question float32 sums are the only large objects (one facet at a
time), and each pass over a facet's answers keeps a single shard in memory. Resident memory therefore scales with one
facet's questions and one shard, not with the whole run.

Outputs, in <run folder>/vector_sets/<vector set name>_<timestamp>/ (a new folder per run, never overwritten; every file carries the timestamp):
  persona_vector_<facet>_topic_removed_<ts>.npy              float32 (layers, hidden); the vectors to use
  persona_vector_<facet>_before_topic_removal_<ts>.npy
  persona_vector_<facet>__vector_from_half_A_<ts>.npy, ..._half_B_...  topic-removed facet vector from one half of the domains
                                                                (scored by the pod as `<facet>__vector_from_half_A`)
  persona_vector_<facet>_negative_tag_<tag>_topic_removed_<ts>.npy (tagged facets)
  topic_vector_without_welfare_remark_answers_<ts>.npy, topic_vector_with_all_answers_<ts>.npy
  persona_vector_metadata_<ts>.json                          layer choice, filter counts, diagnostics
"""

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import persona_vector_math as vector_math  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402
from shared.timed_progress_reporter import PROGRESS_REPORT_INTERVAL_SECONDS, format_duration  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------

class ActivationStore:
    """Index from answer id to (shard file, row) built from the shards' small members only; rows are streamed one shard
    at a time."""

    def __init__(self, activations_folder, array_name="activations", token_count_name="answer_token_counts"):
        self.array_name = array_name  # "passage_activations" with "passage_token_counts" for welfare passage means
        self.location = {}
        self.answer_token_counts = {}
        self.shard_paths = sorted(Path(activations_folder).glob("activations_shard_*.npz"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No activation shards in {activations_folder}")
        for path in self.shard_paths:
            with np.load(path) as shard:
                for row, (answer_id, count) in enumerate(zip(shard["answer_ids"], shard[token_count_name])):
                    self.location[str(answer_id)] = (path, row)
                    self.answer_token_counts[str(answer_id)] = int(count)

    def stream_rows(self, wanted_answer_ids):
        """Yields (answer_id, fp16 or fp32 row of shape (layers, hidden)), one shard in memory at a time."""
        rows_by_path = defaultdict(list)
        for answer_id in wanted_answer_ids:
            path, row = self.location[answer_id]
            rows_by_path[path].append((row, answer_id))
        for path, wanted in rows_by_path.items():
            with np.load(path) as shard:
                activations = shard[self.array_name]
                for row, answer_id in wanted:
                    yield answer_id, activations[row]
            del activations


def load_answers(run_folder):
    records = []
    for path in sorted((Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME).glob("answers_shard_*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def load_judge_scores(run_folder):
    """{answer_id: {"facet_score", "coherence", "share_about_animals", "volunteered_welfare_remark"}}; discarded or missing
    values are None. The consistency pair score records (no answer id) are ignored here."""
    scores = defaultdict(dict)
    for path in sorted((Path(run_folder) / defs.JUDGEMENTS_FOLDER_NAME).glob("judge_results_*.jsonl")):
        for record in read_jsonl(path):
            if not record.get("answer_id"):
                continue
            if record["kind"] == "volunteered_welfare_remark":
                scores[record["answer_id"]]["volunteered_welfare_remark"] = (
                    None if record["discarded"] else record["yes_or_no"] == "YES")
            else:
                scores[record["answer_id"]][record["kind"]] = None if record["discarded"] else record["score"]
    return scores


def rewording_number_of(answer):
    return int(answer["system_prompt_id"].rsplit("_", 1)[1])


# ---------------------------------------------------------------------------------------------------
# Question-level differences
# ---------------------------------------------------------------------------------------------------

def accumulate_sums(store, sums, answer_roles):
    """sums: dict key -> float32 array, filled by streaming; answer_roles: {answer_id: key}."""
    for answer_id, row in store.stream_rows(list(answer_roles)):
        key = answer_roles[answer_id]
        if key not in sums:
            sums[key] = np.zeros(row.shape, dtype=np.float32)
        sums[key] += row.astype(np.float32)


def compute_question_differences(facet, answers, scores, store, thresholds, negative_tag=None):
    """Returns a dict with per-question vectors (Q, layers, hidden), question ids, domains, halves, and filter counts.
    Only extraction halves (A and B) are used. Consistent: also the vectors of each version on its own."""
    paired = facet == defs.FACET_CONSISTENT
    versions = [defs.CONSISTENT_VERSION_A, defs.CONSISTENT_VERSION_B] if paired else [None]
    items = defaultdict(dict)
    for answer in answers:
        if answer["facet"] != facet or answer["half"] not in defs.EXTRACTION_HALVES:
            continue
        key = (answer["pair_id"] if paired else answer["question_id"], rewording_number_of(answer), answer["sample_index"])
        items[key][(answer["version"], answer["pole"])] = answer
    counts = {"items_total": 0, "items_with_missing_answers": 0, "items_with_other_tag": 0, "items_failing_filter": 0,
              "items_kept": 0}
    role_by_answer, kept_count_by_question, question_info = {}, defaultdict(int), {}
    for (question_key, rewording, sample), members in items.items():
        counts["items_total"] += 1
        if any((version, pole) not in members for version in versions for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE)):
            counts["items_with_missing_answers"] += 1
            continue
        if negative_tag is not None and members[(versions[0], defs.POLE_NEGATIVE)]["tag"] != negative_tag:
            counts["items_with_other_tag"] += 1
            continue
        usable = all(store.answer_token_counts.get(m["answer_id"], 0) > 0 for m in members.values())
        scored = {(v, p): scores.get(members[(v, p)]["answer_id"], {}) for v in versions for p in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE)}
        if not usable or not all(vector_math.pair_passes_filter(
                scored[(v, defs.POLE_POSITIVE)], scored[(v, defs.POLE_NEGATIVE)], **thresholds) for v in versions):
            counts["items_failing_filter"] += 1
            continue
        counts["items_kept"] += 1
        kept_count_by_question[question_key] += 1
        any_answer = members[(versions[0], defs.POLE_POSITIVE)]
        question_info[question_key] = (any_answer["domain"], any_answer["half"])
        for version in versions:
            role_by_answer[members[(version, defs.POLE_POSITIVE)]["answer_id"]] = (question_key, version, "positive")
            role_by_answer[members[(version, defs.POLE_NEGATIVE)]["answer_id"]] = (question_key, version, "negative")
    sums = {}
    accumulate_sums(store, sums, role_by_answer)
    question_keys = sorted(kept_count_by_question)
    per_version = {}
    for version in versions:
        per_version[version] = np.stack([
            (sums[(q, version, "positive")] - sums[(q, version, "negative")]) / kept_count_by_question[q]
            for q in question_keys]) if question_keys else None
    if paired:
        combined = vector_math.average_consistent_versions_with_equal_weight(per_version["a"], per_version["b"]) if question_keys else None
    else:
        combined = per_version[None]
    return {"question_keys": question_keys, "vectors": combined, "domains": [question_info[q][0] for q in question_keys],
            "halves": [question_info[q][1] for q in question_keys], "kept_items_per_question": [kept_count_by_question[q] for q in question_keys],
            "per_version_vectors": {v: per_version[v] for v in versions} if paired else None, "filter_counts": counts}


def compute_topic_question_differences(answers, scores, store, coherence_minimum, exclude_welfare_remark_answers):
    """Topic vector ingredients: per question pair, the mean of (animal answer - non-animal answer) over samples. The
    generic assistant's answers only; a pair is dropped if either answer is incoherent, has no tokens, or (when
    exclude_welfare_remark_answers) the animal answer volunteered a welfare remark or its flag is missing."""
    pairs = defaultdict(dict)
    for answer in answers:
        if answer["facet"] == defs.TOPIC_CONTROL and answer["half"] in defs.EXTRACTION_HALVES:
            pairs[(answer["pair_id"], answer["sample_index"])][answer["version"]] = answer
    counts = {"items_total": 0, "items_with_missing_answers": 0, "items_failing_coherence": 0,
              "items_with_welfare_remark_excluded": 0, "items_kept": 0}
    role_by_answer, kept_count_by_pair, info = {}, defaultdict(int), {}
    for (pair_id, sample), members in pairs.items():
        counts["items_total"] += 1
        if set(members) != {defs.TOPIC_VERSION_ANIMAL, defs.TOPIC_VERSION_NON_ANIMAL}:
            counts["items_with_missing_answers"] += 1
            continue
        animal, non_animal = members[defs.TOPIC_VERSION_ANIMAL], members[defs.TOPIC_VERSION_NON_ANIMAL]
        coherence_ok = all((scores.get(m["answer_id"], {}).get("coherence") is not None
                            and scores[m["answer_id"]]["coherence"] >= coherence_minimum
                            and store.answer_token_counts.get(m["answer_id"], 0) > 0) for m in (animal, non_animal))
        if not coherence_ok:
            counts["items_failing_coherence"] += 1
            continue
        flag = scores.get(animal["answer_id"], {}).get("volunteered_welfare_remark")
        if exclude_welfare_remark_answers and flag is not False:
            counts["items_with_welfare_remark_excluded"] += 1
            continue
        counts["items_kept"] += 1
        kept_count_by_pair[pair_id] += 1
        info[pair_id] = (animal["domain"], animal["half"])
        role_by_answer[animal["answer_id"]] = (pair_id, "animal")
        role_by_answer[non_animal["answer_id"]] = (pair_id, "non_animal")
    sums = {}
    accumulate_sums(store, sums, role_by_answer)
    keys = sorted(kept_count_by_pair)
    vectors = np.stack([(sums[(k, "animal")] - sums[(k, "non_animal")]) / kept_count_by_pair[k] for k in keys]) if keys else None
    return {"question_keys": keys, "vectors": vectors, "domains": [info[k][0] for k in keys],
            "halves": [info[k][1] for k in keys], "filter_counts": counts}


def domain_weighted_vector(differences, halves=None):
    if differences["vectors"] is None:
        return None
    selected = [i for i, half in enumerate(differences["halves"]) if halves is None or half in halves]
    if not selected:
        return None
    return vector_math.average_questions_with_domains_weighted_equally(
        differences["vectors"][selected], [differences["domains"][i] for i in selected])


# ---------------------------------------------------------------------------------------------------
# Held-out classification
# ---------------------------------------------------------------------------------------------------

def project_answers(store, answer_ids, unit_vectors_by_name):
    """{name: {answer_id: projection per layer}} for each unit vector set (layers, hidden); one shard in memory."""
    projections = {name: {} for name in unit_vectors_by_name}
    for answer_id, row in store.stream_rows(answer_ids):
        row32 = row.astype(np.float32)
        for name, unit in unit_vectors_by_name.items():
            projections[name][answer_id] = np.sum(row32 * unit, axis=-1)
    return projections


def individually_passing_answers(facet, answers, scores, store, thresholds, halves):
    passing = {defs.POLE_POSITIVE: [], defs.POLE_NEGATIVE: []}
    for answer in answers:
        if answer["facet"] != facet or answer["half"] not in halves or answer["pole"] not in passing:
            continue
        if store.answer_token_counts.get(answer["answer_id"], 0) == 0:
            continue
        if vector_math.answer_passes_individually(
                answer["pole"] == defs.POLE_POSITIVE, scores.get(answer["answer_id"], {}), thresholds["positive_minimum"],
                thresholds["negative_maximum"], thresholds["coherence_minimum"]):
            passing[answer["pole"]].append(answer["answer_id"])
    return passing


def held_out_classification(facet, vectors_by_name, answers, scores, store, thresholds):
    """For each vector set (for example raw and topic removed): accuracy per layer on held-out answers, using the midpoint
    of the mean projections of the extraction answers as the threshold, and area under the curve per layer."""
    unit = {name: vector_math.unit_vectors_per_layer(vector) for name, vector in vectors_by_name.items()}
    fit = individually_passing_answers(facet, answers, scores, store, thresholds, defs.EXTRACTION_HALVES)
    held_out = individually_passing_answers(facet, answers, scores, store, thresholds, [defs.HALF_HELD_OUT])
    if not fit["positive"] or not fit["negative"] or not held_out["positive"] or not held_out["negative"]:
        return None
    wanted = fit["positive"] + fit["negative"] + held_out["positive"] + held_out["negative"]
    projections = project_answers(store, wanted, unit)
    result = {}
    for name in vectors_by_name:
        stack = lambda ids: np.stack([projections[name][i] for i in ids])  # noqa: E731
        thresholds_per_layer = vector_math.midpoint_thresholds(stack(fit["positive"]), stack(fit["negative"]))
        result[name] = {
            "accuracy_per_layer": vector_math.classification_accuracy_per_layer(
                stack(held_out["positive"]), stack(held_out["negative"]), thresholds_per_layer),
            "area_under_curve_per_layer": vector_math.area_under_curve_per_layer(
                stack(held_out["positive"]), stack(held_out["negative"])),
            "held_out_positive_answers": len(held_out["positive"]), "held_out_negative_answers": len(held_out["negative"])}
    return result


# ---------------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------------

def rounded_list(array, digits=5):
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


def compute_all(run_folder, thresholds, facets, bootstrap_resamples, minimum_layer_index, vector_set_name, seed,
                output_folder=None, say=print):
    run_folder = Path(run_folder)
    stamp = timestamp_now()
    if output_folder is None:
        # Each run gets its own folder; exist_ok=False makes a second run in the same minute fail instead of overwriting
        output_folder = run_folder / defs.VECTOR_SETS_FOLDER_NAME / f"{vector_set_name}_{stamp}"
        output_folder.mkdir(parents=True, exist_ok=False)
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    store = ActivationStore(run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME)
    answers, scores = load_answers(run_folder), load_judge_scores(run_folder)
    say(f"Loaded {len(answers)} answers, {len(scores)} judged answers, {len(store.location)} activation rows")
    metadata = {"vector_set_name": vector_set_name, "timestamp": stamp, "run_folder": str(run_folder),
                "filter_thresholds": thresholds, "facets": facets, "per_facet": {}, "diagnostics": {}}
    configuration = json.loads(sorted(run_folder.glob("run_configuration_*.json"))[0].read_text())
    metadata.update({"model_name": configuration["model_name"], "model_revision": configuration["model_revision"]})

    # topic vectors
    topic = {}
    for label, exclude in (("without_welfare_remark_answers", True), ("with_all_answers", False)):
        differences = compute_topic_question_differences(answers, scores, store, thresholds["coherence_minimum"], exclude)
        topic[label] = {"differences": differences, "vector": domain_weighted_vector(differences)}
        metadata["per_facet"].setdefault(defs.TOPIC_CONTROL, {})[f"filter_counts_{label}"] = differences["filter_counts"]
        say(f"Topic vector {label}: {differences['filter_counts']}")
    topic_vector = topic["without_welfare_remark_answers"]["vector"]
    if topic_vector is None:
        raise SystemExit("No topic-control pairs survived; cannot remove the topic direction")
    for label in topic:
        if topic[label]["vector"] is not None:
            np.save(output_folder / f"topic_vector_{label}_{stamp}.npy", topic[label]["vector"].astype(np.float32))
    metadata["diagnostics"]["topic_vector_norm_per_layer"] = {
        label: rounded_list(np.linalg.norm(topic[label]["vector"], axis=-1)) for label in topic if topic[label]["vector"] is not None}
    if all(topic[label]["vector"] is not None for label in topic):
        metadata["diagnostics"]["cosine_between_topic_vectors_with_and_without_welfare_remarks_per_layer"] = rounded_list(
            vector_math.cosine_per_layer(topic["without_welfare_remark_answers"]["vector"], topic["with_all_answers"]["vector"]))

    raw, cleaned, tag_vectors, differences_by_facet = {}, {}, {}, {}
    half_variant_vectors = {}  # facet -> {variant name: topic-removed vector from the questions of one half of the domains}
    accuracy_raw, accuracy_cleaned = {}, {}
    for facet in [f for f in facets if f != defs.TOPIC_CONTROL]:
        differences = compute_question_differences(facet, answers, scores, store, thresholds)
        differences_by_facet[facet] = differences
        entry = metadata["per_facet"].setdefault(facet, {})
        entry["filter_counts"] = differences["filter_counts"]
        entry["questions_kept"] = len(differences["question_keys"])
        entry["questions_kept_by_half"] = {h: differences["halves"].count(h) for h in defs.EXTRACTION_HALVES}
        entry["domains_with_kept_questions"] = len(set(differences["domains"]))
        say(f"{facet}: {differences['filter_counts']}; {entry['questions_kept']} questions kept")
        raw[facet] = domain_weighted_vector(differences)
        if raw[facet] is None:
            say(f"{facet}: nothing survived the filter; skipped")
            continue
        cleaned[facet] = vector_math.project_out_per_layer(raw[facet], topic_vector)
        # share of the raw vector that lay along the topic vector, norms
        entry["norm_per_layer_before_topic_removal"] = rounded_list(np.linalg.norm(raw[facet], axis=-1))
        entry["norm_per_layer_after_topic_removal"] = rounded_list(np.linalg.norm(cleaned[facet], axis=-1))
        entry["cosine_with_topic_vector_before_removal_per_layer"] = rounded_list(vector_math.cosine_per_layer(raw[facet], topic_vector))
        entry["share_along_topic_vector_before_removal_per_layer"] = rounded_list(vector_math.share_along_topic_per_layer(raw[facet], topic_vector))
        # split halves
        half_vectors = {h: domain_weighted_vector(differences, [h]) for h in defs.EXTRACTION_HALVES}
        if all(v is not None for v in half_vectors.values()):
            entry["split_half_cosine_half_A_against_half_B_per_layer"] = rounded_list(vector_math.cosine_per_layer(
                vector_math.project_out_per_layer(half_vectors["A"], topic_vector),
                vector_math.project_out_per_layer(half_vectors["B"], topic_vector)))
            half_variant_vectors[facet] = {f"vector_from_half_{h}": vector_math.project_out_per_layer(half_vectors[h], topic_vector)
                                           for h in defs.EXTRACTION_HALVES}
        if bootstrap_resamples and len(differences["question_keys"]) > 1:
            cosines = vector_math.bootstrap_cosine_stability(differences["vectors"], differences["domains"], bootstrap_resamples, seed)
            entry["bootstrap_cosine_with_full_vector_mean_per_layer"] = rounded_list(cosines.mean(axis=0))
            entry["bootstrap_cosine_with_full_vector_fifth_percentile_per_layer"] = rounded_list(np.percentile(cosines, 5, axis=0))
        # tagged negatives
        tags = sorted({a["tag"] for a in answers if a["facet"] == facet and a["pole"] == defs.POLE_NEGATIVE and a["tag"]})
        for tag in tags:
            tag_differences = compute_question_differences(facet, answers, scores, store, thresholds, negative_tag=tag)
            tag_raw = domain_weighted_vector(tag_differences)
            entry.setdefault("negative_tag_filter_counts", {})[tag] = tag_differences["filter_counts"]
            if tag_raw is not None:
                tag_vectors[(facet, tag)] = vector_math.project_out_per_layer(tag_raw, topic_vector)
        if len(tags) == 2 and all((facet, t) in tag_vectors for t in tags):
            entry["cosine_between_negative_tag_vectors_per_layer"] = {
                f"{tags[0]}_against_{tags[1]}": rounded_list(vector_math.cosine_per_layer(tag_vectors[(facet, tags[0])], tag_vectors[(facet, tags[1])]))}
        # consistent: separate versions
        if facet == defs.FACET_CONSISTENT and differences["per_version_vectors"]:
            version_vectors = {v: vector_math.project_out_per_layer(
                vector_math.average_questions_with_domains_weighted_equally(differences["per_version_vectors"][v], differences["domains"]),
                topic_vector) for v in ("a", "b")}
            entry["cosine_between_version_a_and_version_b_vectors_per_layer"] = rounded_list(
                vector_math.cosine_per_layer(version_vectors["a"], version_vectors["b"]))
            for version, vector in version_vectors.items():
                np.save(output_folder / f"persona_vector_consistent_version_{version}_only_topic_removed_{stamp}.npy", vector.astype(np.float32))
            entry["_version_vectors"] = version_vectors
        classification = held_out_classification(facet, {"raw": raw[facet], "cleaned": cleaned[facet]}, answers, scores, store, thresholds)
        if classification is None:
            say(f"{facet}: no held-out answers passed the filter; layer choice excludes this facet")
        else:
            accuracy_raw[facet] = classification["raw"]["accuracy_per_layer"]
            accuracy_cleaned[facet] = classification["cleaned"]["accuracy_per_layer"]
            entry["held_out_accuracy_per_layer_topic_removed"] = rounded_list(accuracy_cleaned[facet])
            entry["held_out_accuracy_per_layer_before_topic_removal"] = rounded_list(accuracy_raw[facet])
            entry["held_out_area_under_curve_per_layer_topic_removed"] = rounded_list(classification["cleaned"]["area_under_curve_per_layer"])
            entry["held_out_answers_used"] = {"positive": classification["cleaned"]["held_out_positive_answers"],
                                              "negative": classification["cleaned"]["held_out_negative_answers"]}

    # cross-facet diagnostics, layer choice, saving
    names = list(cleaned)
    if len(names) > 1:
        metadata["diagnostics"]["cosine_between_facets_topic_removed_per_layer"] = {
            f"{a}_against_{b}": rounded_list(vector_math.cosine_per_layer(cleaned[a], cleaned[b]))
            for i, a in enumerate(names) for b in names[i + 1:]}
        metadata["diagnostics"]["cosine_between_facets_before_topic_removal_per_layer"] = {
            f"{a}_against_{b}": rounded_list(vector_math.cosine_per_layer(raw[a], raw[b]))
            for i, a in enumerate(names) for b in names[i + 1:]}
    if defs.FACET_CONSISTENT in cleaned and defs.FACET_CARES_ABOUT_WELFARE in cleaned:
        removed = vector_math.project_out_per_layer(cleaned[defs.FACET_CONSISTENT], cleaned[defs.FACET_CARES_ABOUT_WELFARE])
        np.save(output_folder / f"persona_vector_consistent_with_cares_about_welfare_also_removed_{stamp}.npy", removed.astype(np.float32))
        entry = metadata["per_facet"][defs.FACET_CONSISTENT]
        entry["cosine_of_each_version_with_cares_about_welfare_per_layer"] = {
            v: rounded_list(vector_math.cosine_per_layer(vec, cleaned[defs.FACET_CARES_ABOUT_WELFARE]))
            for v, vec in entry["_version_vectors"].items()}
    if defs.FACET_CONSISTENT in metadata["per_facet"]:
        metadata["per_facet"][defs.FACET_CONSISTENT].pop("_version_vectors", None)
    if accuracy_cleaned:
        metadata["layer_choice"] = vector_math.choose_layers(accuracy_cleaned, minimum_layer_index)
        metadata["layer_choice"]["based_on"] = "held-out classification accuracy of the topic-removed vectors"
    else:
        metadata["layer_choice"] = None
    file_names = {}
    for facet in names:
        for suffix, vector in (("topic_removed", cleaned[facet]), ("before_topic_removal", raw[facet])):
            name = f"persona_vector_{facet}_{suffix}_{stamp}.npy"
            np.save(output_folder / name, vector.astype(np.float32))
            file_names[f"{facet}_{suffix}"] = name
    for (facet, tag), vector in tag_vectors.items():
        name = f"persona_vector_{facet}_negative_tag_{tag}_topic_removed_{stamp}.npy"
        np.save(output_folder / name, vector.astype(np.float32))
        file_names[f"{facet}_negative_tag_{tag}_topic_removed"] = name
    # Variants for the scoring pod, named `<facet>__<variant>` (scoring_pod_vector_loading_adapter.py): the facet vector
    # from each half of the domains, topic removed, for the split-half comparison of document rankings.
    for facet, variants in half_variant_vectors.items():
        for variant_name, vector in variants.items():
            name = f"persona_vector_{facet}__{variant_name}_{stamp}.npy"
            np.save(output_folder / name, vector.astype(np.float32))
            file_names[f"{facet}__{variant_name}"] = name
    metadata["vector_files"] = file_names
    metadata["topic_vector_files"] = sorted(p.name for p in output_folder.glob(f"topic_vector_*_{stamp}.npy"))
    first_vector = next(iter(cleaned.values()))
    metadata["shape"] = {"layers_including_embedding_output": int(first_vector.shape[0]), "hidden": int(first_vector.shape[1])}
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
    parser.add_argument("--vector-set-name", default="persona_vectors_unnamed")
    parser.add_argument("--facets", nargs="+", default=None, help="Default: every facet that has answers in the run")
    parser.add_argument("--positive-score-minimum", type=float, default=defs.DEFAULT_POSITIVE_SCORE_MINIMUM)
    parser.add_argument("--negative-score-maximum", type=float, default=defs.DEFAULT_NEGATIVE_SCORE_MAXIMUM)
    parser.add_argument("--coherence-minimum", type=float, default=defs.DEFAULT_COHERENCE_MINIMUM)
    parser.add_argument("--animal-share-maximum-difference", type=float, default=defs.DEFAULT_ANIMAL_SHARE_MAXIMUM_DIFFERENCE)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--minimum-layer-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    run_folder = Path(arguments.run_folder)
    configuration = json.loads(sorted(run_folder.glob("run_configuration_*.json"))[0].read_text())
    facets = arguments.facets or configuration["facets"]
    thresholds = {"positive_minimum": arguments.positive_score_minimum, "negative_maximum": arguments.negative_score_maximum,
                  "coherence_minimum": arguments.coherence_minimum,
                  "animal_share_maximum_difference": arguments.animal_share_maximum_difference}
    compute_all(run_folder, thresholds, facets, arguments.bootstrap_resamples, arguments.minimum_layer_index,
                arguments.vector_set_name, arguments.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
