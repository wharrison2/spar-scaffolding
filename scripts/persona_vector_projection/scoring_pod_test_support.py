"""Shared helpers for the scoring pod tests and the local rehearsal: a tiny random Gemma 4 model with the real Gemma 4
tokenizer (read from the local Hugging Face cache, offline), synthetic vectors, and a synthetic corpus."""

import json
from pathlib import Path

import numpy as np

from persona_vector_projection import scoring_pod_definitions as defs
import tiny_model_and_tokenizer_helpers_for_log_probability_persona_check as tiny

TINY_HIDDEN_SIZE = 32
TINY_LAYER_COUNT = 4
TINY_CHOSEN_LAYER = 2   # recorded layers 0..4: the whole tiny model


def load_real_gemma_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(defs.MODEL_NAME, revision=defs.MODEL_REVISION, local_files_only=True)


def save_tiny_model_with_real_tokenizer(directory):
    tokenizer = load_real_gemma_tokenizer()
    Path(directory).mkdir(parents=True, exist_ok=True)
    tiny.build_tiny_random_gemma4_model(len(tokenizer), final_logit_softcapping=30.0).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return tokenizer


def write_synthetic_vector_folder(folder, random_seed=0, layer_by_facet=None, topic_removed=True):
    """Writes the extraction half's layout: per facet a `_topic_removed` file (random with the topic direction projected
    out at every layer) and a `_before_topic_removal` file (random plus topic), the topic vector, and
    persona_vector_metadata_<ts>.json. Returns the topic-removed facet vectors and the topic vector by name. With
    topic_removed=False the `_topic_removed` entry is left out of the metadata."""
    from persona_vector_projection.persona_vector_math import project_out_per_layer
    random_generator = np.random.default_rng(random_seed)
    shape = (TINY_LAYER_COUNT + 1, TINY_HIDDEN_SIZE)
    topic = random_generator.normal(size=shape).astype(np.float32)
    vectors = {defs.TOPIC_CONTROL: topic}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = "2026-10-31_00-00"
    vector_files = {}
    for facet in defs.PERSONA_FACETS:
        raw = random_generator.normal(size=shape).astype(np.float32) * 3.0 + topic
        removed = project_out_per_layer(raw, topic).astype(np.float32)
        vectors[facet] = removed
        np.save(folder / f"persona_vector_{facet}_before_topic_removal_{stamp}.npy", raw)
        vector_files[f"{facet}_before_topic_removal"] = f"persona_vector_{facet}_before_topic_removal_{stamp}.npy"
        if topic_removed:
            np.save(folder / f"persona_vector_{facet}_topic_removed_{stamp}.npy", removed)
            vector_files[f"{facet}_topic_removed"] = f"persona_vector_{facet}_topic_removed_{stamp}.npy"
    np.save(folder / f"topic_vector_without_welfare_remark_answers_{stamp}.npy", topic)
    np.save(folder / f"topic_vector_with_all_answers_{stamp}.npy", topic * 1.1)
    layer_by_facet = layer_by_facet or {facet: TINY_CHOSEN_LAYER for facet in defs.PERSONA_FACETS}
    shared = set(layer_by_facet.values()).pop() if len(set(layer_by_facet.values())) == 1 else None
    (folder / f"persona_vector_metadata_{stamp}.json").write_text(json.dumps({
        "vector_set_name": "synthetic", "timestamp": stamp, "model_name": defs.MODEL_NAME, "model_revision": defs.MODEL_REVISION,
        "vector_files": vector_files,
        "topic_vector_files": [f"topic_vector_with_all_answers_{stamp}.npy", f"topic_vector_without_welfare_remark_answers_{stamp}.npy"],
        "layer_choice": {"strategy": "shared_layer" if shared is not None else "per_facet_best_layer", "shared_layer": shared,
                         "layer_by_facet": layer_by_facet},
        "shape": {"layers_including_embedding_output": shape[0], "hidden": shape[1]}}))
    return vectors


VARIABLE_VALUES = {
    "document_type": "a news article", "culture": "Poland, written in Polish", "domain": "research and testing",
    "sentient_category": "wild invertebrates", "decision_scale": "a single animal", "naming": "a generic AI assistant",
    "tech_savvy": "average members of the public", "tradeoff": "cost/convenience",
    "centrality": "the central subject of the document", "tone": "TONEMARKER supportive",
    "resolution": "RESOLUTIONMARKER the AI declines", "reasoning_featured": "REASONINGMARKER individualism"}


def make_corpus_record(number, content=None, language="Polish"):
    variables = {key: f"{value} {number}" if key in defs.KEPT_TOPIC_VARIABLE_NAMES else value
                 for key, value in VARIABLE_VALUES.items()}
    return {"doc_id": f"matrix_{number:06d}", "content": content or f"Title {number}\n\nA short synthetic document number {number} about seals and nets.",
            "language": language, "variables": variables}


def make_rewrite_record(source_record, direction):
    return {"document_id": f"{source_record['doc_id']}__{direction}", "source_document_id": source_record["doc_id"],
            "rewrite_direction": direction, "content": source_record["content"] + f" Rewritten {direction}.",
            "language": source_record["language"], "variables": dict(source_record["variables"])}


def write_synthetic_converted_inputs(folder, facets=None):
    """questions_<ts>.jsonl with held-out questions, shared_suffixes_<ts>.json and facet_definitions_<ts>.json, named as the
    converter names them."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    questions = []
    for facet in (facets or defs.PERSONA_FACETS):
        for half, count in (("A", 2), ("held_out", 3)):
            for number in range(1, count + 1):
                questions.append({"facet": facet, "half": half, "question_number": number, "domain": f"domain {number}",
                                  "question_id": f"{facet}__{half}__{number:02d}", "question_text": f"Question {facet} {half} {number}?"})
    (folder / "questions_2026-10-31_00-00.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    (folder / "shared_suffixes_2026-10-31_00-00.json").write_text(json.dumps({"topic_control_generic_system_prompt": "Keep answers short."}))
    (folder / "facet_definitions_2026-10-31_00-00.json").write_text(json.dumps({}))
    return questions
