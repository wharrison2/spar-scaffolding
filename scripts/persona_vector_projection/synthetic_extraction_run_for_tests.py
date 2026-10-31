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
