"""Builds a small synthetic run folder (answers, activations, judge results) with known geometry, so that the vector
computation can be tested against exact expectations. No model, no API.

Geometry: every answer's activation (layers, hidden) = a question-specific random offset + pole sign * facet signal
direction (+ a topic direction for topic-control animal answers and, scaled, for every animal-heavy facet answer) + small
noise. The facet signal directions are random orthogonal-ish unit vectors scaled by SIGNAL_SCALE, so the computed vectors
can be compared against the known direction.
"""

import json
from pathlib import Path

import numpy as np

from persona_vector_projection import persona_vector_experiment_definitions as defs

DOMAINS_HALF_A = ["domain_a1", "domain_a2", "domain_a3"]
DOMAINS_HALF_B = ["domain_b1", "domain_b2", "domain_b3"]
DOMAINS_HELD_OUT = DOMAINS_HALF_A + DOMAINS_HALF_B
QUESTIONS_PER_DOMAIN = 2
REWORDINGS = 3
SAMPLES = 2
SIGNAL_SCALE = 5.0


def make_directions(layers, hidden, seed=0):
    generator = np.random.default_rng(seed)
    names = defs.PERSONA_FACETS + [defs.TOPIC_CONTROL]
    raw = generator.normal(size=(len(names), hidden))
    q, _ = np.linalg.qr(raw.T)  # orthonormal columns
    return {name: np.tile(q[:, i], (layers, 1)).astype(np.float32) for i, name in enumerate(names)}


def build_synthetic_run(run_folder, facets=None, layers=5, hidden=64, answers_per_shard=40, seed=0,
                        activation_offset_scale=2.0, facet_signal_scale=SIGNAL_SCALE, topic_loading_of_facets=0.0):
    run_folder = Path(run_folder)
    facets = facets or (defs.PERSONA_FACETS + [defs.TOPIC_CONTROL])
    generator = np.random.default_rng(seed)
    directions = make_directions(layers, hidden, seed)
    (run_folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    (run_folder / f"run_configuration_2026-01-01_00-00.json").write_text(json.dumps(
        {"facets": facets, "model_name": "synthetic", "model_revision": "none"}))
    answers, activation_rows = [], []
    scores_records = []

    def add_answer(facet, pole, tag, question_id, pair_id, version, half, domain, rewording, sample, activation,
                   facet_score, coherence=90.0, share=50.0, welfare_flag=None):
        system_prompt_id = (f"{facet}__{pole}__rewording_{rewording:02d}")
        answer_id = f"{system_prompt_id}__{question_id}__sample_{sample:02d}"
        answers.append({"facet": facet, "pole": pole, "tag": tag, "system_prompt_id": system_prompt_id, "question_id": question_id,
                        "pair_id": pair_id, "version": version, "half": half, "domain": domain, "answer_id": answer_id,
                        "job_id": f"{system_prompt_id}__{question_id}", "sample_index": sample, "answer_text": "text",
                        "number_of_answer_tokens_generated": 5, "finish_reason": "stop", "number_of_prompt_tokens": 10})
        activation_rows.append(activation.astype(np.float16))
        base = {"answer_id": answer_id, "discarded": False, "yes_or_no": None, "usage": {}}
        if facet_score is not None:
            scores_records.append({**base, "request_id": answer_id + "__facet_score", "kind": "facet_score", "score": facet_score})
        scores_records.append({**base, "request_id": answer_id + "__coherence", "kind": "coherence", "score": coherence})
        scores_records.append({**base, "request_id": answer_id + "__share_about_animals", "kind": "share_about_animals", "score": share})
        if welfare_flag is not None:
            scores_records.append({**base, "request_id": answer_id + "__volunteered_welfare_remark",
                                   "kind": "volunteered_welfare_remark", "score": None, "yes_or_no": "YES" if welfare_flag else "NO"})

    halves = [(defs.HALF_A, DOMAINS_HALF_A), (defs.HALF_B, DOMAINS_HALF_B), (defs.HALF_HELD_OUT, DOMAINS_HELD_OUT)]
    for facet in facets:
        for half, domains in halves:
            number = 0
            for domain in domains:
                for _ in range(1 if half == defs.HALF_HELD_OUT else QUESTIONS_PER_DOMAIN):
                    number += 1
                    pair_id = defs.question_record_identifier(facet, half, number) if facet in defs.PAIRED_FACETS else None
                    offset = generator.normal(size=(layers, hidden)).astype(np.float32) * activation_offset_scale
                    if facet == defs.TOPIC_CONTROL:
                        for version in (defs.TOPIC_VERSION_ANIMAL, defs.TOPIC_VERSION_NON_ANIMAL):
                            sign = 1 if version == defs.TOPIC_VERSION_ANIMAL else -1
                            question_id = defs.question_record_identifier(facet, half, number, version)
                            for sample in range(SAMPLES * REWORDINGS):
                                activation = offset + sign * directions[facet] * facet_signal_scale + generator.normal(size=offset.shape) * 0.05
                                add_answer(facet, defs.POLE_GENERIC, None, question_id, pair_id, version, half, domain, 1, sample,
                                           activation, None, welfare_flag=(sign == 1 and sample == 0))
                        continue
                    versions = [defs.CONSISTENT_VERSION_A, defs.CONSISTENT_VERSION_B] if facet == defs.FACET_CONSISTENT else [None]
                    for version in versions:
                        question_id = defs.question_record_identifier(facet, half, number, version)
                        version_scale = {None: 1.0, "a": 1.0, "b": 3.0}[version]  # version b has a stronger signal
                        for pole, sign in ((defs.POLE_POSITIVE, 1), (defs.POLE_NEGATIVE, -1)):
                            for rewording in range(1, REWORDINGS + 1):
                                tag = None
                                if pole == defs.POLE_NEGATIVE and facet == defs.FACET_NON_INTRUSIVE:
                                    tag = "lecturing" if rewording <= 2 else "refusing_or_watering_down"
                                for sample in range(SAMPLES):
                                    activation = (offset + sign * directions[facet] * facet_signal_scale * version_scale
                                                  + directions[defs.TOPIC_CONTROL] * topic_loading_of_facets
                                                  + generator.normal(size=offset.shape) * 0.05)
                                    add_answer(facet, pole, tag, question_id, pair_id, version, half, domain, rewording, sample, activation,
                                               facet_score=85.0 if pole == defs.POLE_POSITIVE else 10.0)
    answers_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME
    activations_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    answers_folder.mkdir(parents=True, exist_ok=True)
    activations_folder.mkdir(parents=True, exist_ok=True)
    for shard_number, start in enumerate(range(0, len(answers), answers_per_shard)):
        chunk = answers[start:start + answers_per_shard]
        (answers_folder / f"answers_shard_{shard_number:05d}_2026-01-01_00-00.jsonl").write_text(
            "\n".join(json.dumps(record) for record in chunk) + "\n")
        np.savez(activations_folder / f"activations_shard_{shard_number:05d}_2026-01-01_00-00.npz",
                 answer_ids=np.array([r["answer_id"] for r in chunk]),
                 activations=np.stack(activation_rows[start:start + answers_per_shard]),
                 answer_token_counts=np.full(len(chunk), 5, dtype=np.int32))
    judge_folder = run_folder / defs.JUDGEMENTS_FOLDER_NAME
    judge_folder.mkdir(exist_ok=True)
    (judge_folder / "judge_results_2026-01-01_00-00.jsonl").write_text(
        "\n".join(json.dumps(record) for record in scores_records) + "\n")
    return run_folder, directions, answers, scores_records


# ---------------------------------------------------------------------------------------------------
# Moralizing delivery
# ---------------------------------------------------------------------------------------------------

MORALIZING_REWORDINGS = 5
MORALIZING_SAMPLES = 2
LARGE_MEAN_DIMENSIONS = {0: 40.0, 1: 25.0}  # dimensions every activation carries a big constant on (like dimension 1750)


def moralizing_passage_text(question_number, pole, rewording):
    """The same length for both poles (so the length-ratio condition passes), different words."""
    word = "plainly" if pole == defs.POLE_POSITIVE else "morally"
    return f"Piglets {word} feel pain at point {question_number:02d} rewording {rewording}."


def build_synthetic_moralizing_delivery_run(run_folder, layers=30, hidden=48, seed=0, passage_signal_scale=5.0,
                                            whole_answer_signal_scale=1.5, offset_scale=2.0):
    """A run folder for the moralizing delivery vector computation: answers with a quoted welfare passage, judge results
    (delivery score, coherence, content check), a spans file, and activation shards with whole-answer and passage means.
    Geometry: question offset + pole sign * signal direction (strong in passage means, weak in whole-answer means) + large
    constants on dimensions 0 and 1 (the largest-mean dimensions) + small noise. Returns (run folder, signal direction)."""
    run_folder = Path(run_folder)
    generator = np.random.default_rng(seed)
    direction = np.zeros(hidden, dtype=np.float32)
    direction[10:14] = 0.5  # unit vector, away from the large-mean dimensions 0 and 1
    (run_folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(
        {"facets": [defs.FACET_MORALIZING_DELIVERY], "model_name": "synthetic", "model_revision": "none",
         "activation_pass_system_prompt": "Keep your answer under 300 words."}))
    answers, whole_rows, passage_rows, judge_records, span_records = [], [], [], [], []
    halves = [(defs.HALF_A, ["domain_a1", "domain_a2", "domain_a3"], 2), (defs.HALF_B, ["domain_b1", "domain_b2", "domain_b3"], 2),
              (defs.HALF_HELD_OUT, ["domain_a1", "domain_a2", "domain_a3", "domain_b1", "domain_b2", "domain_b3"], 1)]
    tags = defs.MORALIZING_DELIVERY_NEGATIVE_TAGS
    for half, domains, questions_per_domain in halves:
        number = 0
        for domain in domains:
            for _ in range(questions_per_domain):
                number += 1
                question_id = defs.question_record_identifier(defs.FACET_MORALIZING_DELIVERY, half, number)
                offset = generator.normal(size=(layers, hidden)).astype(np.float32) * offset_scale
                for pole, sign in ((defs.POLE_POSITIVE, 1), (defs.POLE_NEGATIVE, -1)):
                    for rewording in range(1, MORALIZING_REWORDINGS + 1):
                        for sample in range(MORALIZING_SAMPLES):
                            prompt_id = defs.system_prompt_identifier(defs.FACET_MORALIZING_DELIVERY, pole, rewording)
                            answer_id = f"{prompt_id}__{question_id}__sample_{sample:02d}"
                            passage = moralizing_passage_text(number, pole, rewording)
                            text = f"Here is the plan. {passage} Then the rest of the plan follows."
                            start = text.index(passage)
                            answers.append({
                                "facet": defs.FACET_MORALIZING_DELIVERY, "pole": pole,
                                "tag": tags[rewording - 1] if pole == defs.POLE_NEGATIVE else None,
                                "system_prompt_id": prompt_id, "question_id": question_id, "pair_id": None, "version": None,
                                "half": half, "domain": domain, "answer_id": answer_id, "job_id": f"{prompt_id}__{question_id}",
                                "sample_index": sample, "answer_text": text, "number_of_answer_tokens_generated": 20,
                                "finish_reason": "stop", "number_of_prompt_tokens": 30})
                            constant = np.zeros((layers, hidden), dtype=np.float32)
                            for dimension, value in LARGE_MEAN_DIMENSIONS.items():
                                constant[:, dimension] = value
                            noise = lambda: generator.normal(size=(layers, hidden)).astype(np.float32) * 0.05  # noqa: E731
                            whole_rows.append((offset + constant + sign * direction * whole_answer_signal_scale + noise()).astype(np.float16))
                            passage_rows.append((offset * 0.5 + constant + sign * direction * passage_signal_scale + noise()).astype(np.float16))
                            base = {"answer_id": answer_id, "discarded": False, "yes_or_no": None, "usage": {}}
                            judge_records.append({**base, "request_id": answer_id + "__delivery_score", "kind": "delivery_score",
                                                  "score": 85.0 if pole == defs.POLE_POSITIVE else 10.0})
                            judge_records.append({**base, "request_id": answer_id + "__coherence", "kind": "coherence", "score": 90.0})
                            judge_records.append({**base, "request_id": answer_id + "__content_check", "kind": "content_check", "score": None,
                                                  "content_check": {"makes_the_point": True, "separate_welfare_mentions": 1,
                                                                    "adds_welfare_facts": False, "completes_the_task": True,
                                                                    "welfare_passage": passage}})
                            span_records.append({"answer_id": answer_id, "facet": defs.FACET_MORALIZING_DELIVERY, "pole": pole,
                                                 "tag": answers[-1]["tag"], "question_id": question_id,
                                                 "status": "found_exact", "character_start": start,
                                                 "character_end": start + len(passage), "number_of_occurrences": 1,
                                                 "failure_reason": None, "answer_character_count": len(text),
                                                 "quoted_welfare_passage": passage})
    answers_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME
    activations_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    answers_folder.mkdir(parents=True, exist_ok=True)
    activations_folder.mkdir(parents=True, exist_ok=True)
    per_shard = 50
    for shard_number, start in enumerate(range(0, len(answers), per_shard)):
        chunk = answers[start:start + per_shard]
        (answers_folder / f"answers_shard_{shard_number:05d}_2026-01-01_00-00.jsonl").write_text(
            "\n".join(json.dumps(record) for record in chunk) + "\n")
        np.savez(activations_folder / f"activations_shard_{shard_number:05d}_2026-01-01_00-00.npz",
                 answer_ids=np.array([r["answer_id"] for r in chunk]), activations=np.stack(whole_rows[start:start + per_shard]),
                 answer_token_counts=np.full(len(chunk), 20, dtype=np.int32),
                 passage_activations=np.stack(passage_rows[start:start + per_shard]),
                 passage_token_counts=np.full(len(chunk), 6, dtype=np.int32),
                 passage_status=np.array(["passage_mean_recorded"] * len(chunk)))
    judge_folder = run_folder / defs.JUDGEMENTS_FOLDER_NAME
    judge_folder.mkdir(exist_ok=True)
    (judge_folder / "judge_results_2026-01-01_00-00.jsonl").write_text("\n".join(json.dumps(r) for r in judge_records) + "\n")
    (run_folder / defs.INPUTS_FOLDER_NAME / f"{defs.WELFARE_PASSAGE_SPANS_FILE_STEM}_2026-01-01_00-00.jsonl").write_text(
        "\n".join(json.dumps(r) for r in span_records) + "\n")
    return run_folder, direction


CALIBRATED_NEGATIVE_TAG_BY_REWORDING = {1: "certainty", 2: "equal_capacity", 3: "certainty", 4: "certainty_and_equal_capacity",
                                        5: "equal_capacity", 6: "dismissal", 7: "understatement", 8: "dismissal", 9: "understatement",
                                        10: "understatement_and_dismissal"}
SYNTHETIC_BAND_CYCLE = ["Established", "Weak (open)", "Strong but debated", "Mixed: established and weak (open)"]
CONFIDENCE_DIRECTION_DIMENSIONS = slice(30, 34)


def synthetic_evidence_band(question_number):
    return SYNTHETIC_BAND_CYCLE[(question_number - 1) % len(SYNTHETIC_BAND_CYCLE)]


def default_stated_confidence(answer_id, pole, slot, sample, question_number):
    """Extraction halves: both poles of one item get the same stated confidence (85 or 15, alternating with the slot, sample and
    question), so the poles' means match exactly and confident and hedged answers hold equal numbers of each pole. Held-out half: the
    realistic pattern, where calibration decides confidence: positive pole confident on established and hedged on weak evidence, negative
    pole the reverse (the two-by-two check needs calibrated cells that differ from the miscalibrated ones); other bands as above."""
    band = synthetic_evidence_band(question_number)
    if "__held_out__" in answer_id and band in ("Established", "Weak (open)"):
        confident = (band == "Established") == (pole == defs.POLE_POSITIVE)
        return 85.0 if confident else 15.0
    return 85.0 if (slot + sample + question_number) % 2 == 0 else 15.0


def build_synthetic_content_controlled_run(run_folder, facet_key, layers=30, hidden=48, seed=0, passage_signal_scale=5.0,
                                           whole_answer_signal_scale=5.0, offset_scale=2.0, store_passage_means_anyway=False,
                                           score_function=None, completes_the_task_function=None, adds_welfare_facts_function=None, stated_confidence_function=None,
                                           confidence_signal_scale=3.0, band_function=None):
    """As build_synthetic_moralizing_delivery_run, for any content-controlled facet (content_controlled_facet_definitions).
    The passage means carry the pole signal along PASSAGE_DIRECTION and the whole-answer means along a DIFFERENT direction
    (WHOLE_ANSWER_DIRECTION), so a computed vector shows which of the two means was used. Positive pole = +signal (the vector
    is positive minus negative). The facet score uses the facet's judge kind; default scores are 85 (positive) and 10
    (negative); `score_function(answer_id, pole)` overrides them. A facet whose main mean is the whole-answer mean (non-intrusive)
    gets no passage means and no spans file unless `store_passage_means_anyway`. Returns (run folder, passage direction,
    whole-answer direction)."""
    from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module
    definition = facet_definitions_module.get_facet_definition(facet_key)
    run_folder = Path(run_folder)
    generator = np.random.default_rng(seed)
    passage_direction = np.zeros(hidden, dtype=np.float32)
    passage_direction[10:14] = 0.5
    whole_answer_direction = np.zeros(hidden, dtype=np.float32)
    whole_answer_direction[20:24] = 0.5
    with_passages = definition.uses_welfare_passage or store_passage_means_anyway
    (run_folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(
        {"facets": [facet_key], "model_name": "synthetic", "model_revision": "none",
         "activation_pass_system_prompt": "Keep your answer under 300 words."}))
    answers, whole_rows, passage_rows, judge_records, span_records = [], [], [], [], []
    halves = [(defs.HALF_A, ["domain_a1", "domain_a2", "domain_a3"], 2), (defs.HALF_B, ["domain_b1", "domain_b2", "domain_b3"], 2),
              (defs.HALF_HELD_OUT, ["domain_a1", "domain_a2", "domain_a3", "domain_b1", "domain_b2", "domain_b3"], 1)]
    tags = list(definition.negative_tag_slugs)
    calibrated = definition.has_confidence_control  # 5 positive rewordings; one of two negative rewordings per slot, chosen by the band
    question_records = []
    confidence_direction = np.zeros(hidden, dtype=np.float32)
    confidence_direction[CONFIDENCE_DIRECTION_DIMENSIONS] = 0.5
    for half, domains, questions_per_domain in halves:
        number = 0
        for domain in domains:
            for _ in range(questions_per_domain):
                number += 1
                question_id = defs.question_record_identifier(facet_key, half, number)
                offset = generator.normal(size=(layers, hidden)).astype(np.float32) * offset_scale
                band_label = (band_function or synthetic_evidence_band)(number) if calibrated else None
                question_record = {"facet": facet_key, "half": half, "domain": domain, "question_id": question_id,
                                   "question_text": f"Question {question_id}?", "labels": {"evidence_band": band_label} if calibrated else {}}
                question_records.append(question_record)
                for pole, sign in ((defs.POLE_POSITIVE, 1), (defs.POLE_NEGATIVE, -1)):
                    rewordings = list(range(1, MORALIZING_REWORDINGS + 1))
                    if calibrated and pole == defs.POLE_NEGATIVE:
                        rewordings = []
                        for slot in range(1, MORALIZING_REWORDINGS + 1):
                            allowed = [n for n in (slot, slot + MORALIZING_REWORDINGS)
                                       if definition.negative_rewording_is_miscalibrated_for_question(CALIBRATED_NEGATIVE_TAG_BY_REWORDING[n], question_record)]
                            rewordings.append(allowed[(number + slot) % len(allowed)])
                    for rewording in rewordings:
                        for sample in range(MORALIZING_SAMPLES):
                            prompt_id = defs.system_prompt_identifier(facet_key, pole, rewording)
                            answer_id = f"{prompt_id}__{question_id}__sample_{sample:02d}"
                            passage = moralizing_passage_text(number, pole, rewording)
                            text = f"Here is the plan. {passage} Then the rest of the plan follows."
                            start = text.index(passage)
                            tag = ((CALIBRATED_NEGATIVE_TAG_BY_REWORDING[rewording] if calibrated else tags[rewording - 1])
                                   if pole == defs.POLE_NEGATIVE else None)
                            slot = (rewording - 1) % MORALIZING_REWORDINGS + 1
                            answers.append({
                                "facet": facet_key, "pole": pole, "tag": tag, "system_prompt_id": prompt_id,
                                "question_id": question_id, "pair_id": None, "version": None, "half": half, "domain": domain,
                                "answer_id": answer_id, "job_id": f"{prompt_id}__{question_id}", "sample_index": sample,
                                "answer_text": text, "number_of_answer_tokens_generated": 20, "finish_reason": "stop",
                                "number_of_prompt_tokens": 30})
                            constant = np.zeros((layers, hidden), dtype=np.float32)
                            for dimension, value in LARGE_MEAN_DIMENSIONS.items():
                                constant[:, dimension] = value
                            noise = lambda: generator.normal(size=(layers, hidden)).astype(np.float32) * 0.05  # noqa: E731
                            stated_confidence = ((stated_confidence_function or default_stated_confidence)(answer_id, pole, slot, sample, number)
                                                 if calibrated else None)
                            confidence_sign = 0.0 if stated_confidence is None else (1.0 if stated_confidence >= 65 else (-1.0 if stated_confidence <= 35 else 0.0))
                            confidence_part = confidence_sign * confidence_direction * confidence_signal_scale
                            whole_rows.append((offset + constant + sign * whole_answer_direction * whole_answer_signal_scale + confidence_part + noise()).astype(np.float16))
                            passage_rows.append((offset * 0.5 + constant + sign * passage_direction * passage_signal_scale + confidence_part + noise()).astype(np.float16))
                            score = (score_function(answer_id, pole) if score_function
                                     else (85.0 if pole == defs.POLE_POSITIVE else 10.0))
                            completes = completes_the_task_function(answer_id, pole) if completes_the_task_function else True
                            adds_facts = adds_welfare_facts_function(answer_id, pole) if adds_welfare_facts_function else False
                            base = {"answer_id": answer_id, "discarded": False, "yes_or_no": None, "usage": {}}
                            judge_records.append({**base, "request_id": f"{answer_id}__{definition.facet_score_kind}",
                                                  "kind": definition.facet_score_kind, "score": score})
                            judge_records.append({**base, "request_id": answer_id + "__coherence", "kind": "coherence", "score": 90.0})
                            if calibrated:
                                judge_records.append({**base, "request_id": answer_id + "__stated_confidence", "kind": "stated_confidence",
                                                      "score": stated_confidence})
                            judge_records.append({**base, "request_id": answer_id + "__content_check", "kind": "content_check", "score": None,
                                                  "content_check": {"makes_the_point": True, "separate_welfare_mentions": 1,
                                                                    "adds_welfare_facts": adds_facts, "completes_the_task": completes,
                                                                    "welfare_passage": passage}})
                            span_records.append({"answer_id": answer_id, "facet": facet_key, "pole": pole, "tag": tag,
                                                 "question_id": question_id, "status": "found_exact", "character_start": start,
                                                 "character_end": start + len(passage), "number_of_occurrences": 1,
                                                 "failure_reason": None, "answer_character_count": len(text),
                                                 "quoted_welfare_passage": passage})
    (run_folder / defs.INPUTS_FOLDER_NAME / "questions_2026-01-01_00-00.jsonl").write_text(
        "\n".join(json.dumps(record) for record in question_records) + "\n")
    answers_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME
    activations_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    answers_folder.mkdir(parents=True, exist_ok=True)
    activations_folder.mkdir(parents=True, exist_ok=True)
    per_shard = 50
    for shard_number, start in enumerate(range(0, len(answers), per_shard)):
        chunk = answers[start:start + per_shard]
        (answers_folder / f"answers_shard_{shard_number:05d}_2026-01-01_00-00.jsonl").write_text(
            "\n".join(json.dumps(record) for record in chunk) + "\n")
        arrays = {"answer_ids": np.array([r["answer_id"] for r in chunk]), "activations": np.stack(whole_rows[start:start + per_shard]),
                  "answer_token_counts": np.full(len(chunk), 20, dtype=np.int32)}
        if with_passages:
            arrays.update(passage_activations=np.stack(passage_rows[start:start + per_shard]),
                          passage_token_counts=np.full(len(chunk), 6, dtype=np.int32),
                          passage_status=np.array(["passage_mean_recorded"] * len(chunk)))
        np.savez(activations_folder / f"activations_shard_{shard_number:05d}_2026-01-01_00-00.npz", **arrays)
    judge_folder = run_folder / defs.JUDGEMENTS_FOLDER_NAME
    judge_folder.mkdir(exist_ok=True)
    (judge_folder / "judge_results_2026-01-01_00-00.jsonl").write_text("\n".join(json.dumps(r) for r in judge_records) + "\n")
    if definition.uses_welfare_passage:
        (run_folder / defs.INPUTS_FOLDER_NAME / f"{defs.WELFARE_PASSAGE_SPANS_FILE_STEM}_2026-01-01_00-00.jsonl").write_text(
            "\n".join(json.dumps(r) for r in span_records) + "\n")
    return run_folder, passage_direction, whole_answer_direction
