"""A small synthetic scoring-pod run folder in the real layout (see load_persona_projection_scoring_outputs.py), with
known ground truth, for the analysis tests and the smoke test. No model, no API.

Ground truth that tests rely on:
  - every corpus document has its own baseline level (so subtracting another document's baselines gives a visibly
    wrong score), and Spanish scores sit far above English ones (so standardising within language matters);
  - for the facet `cares_about_welfare`, rewrites replace tokens 40-59 of the source: toward rewrites raise the
    per-token projection on the changed tokens by +TOWARD_EFFECT, away rewrites lower it by -AWAY_EFFECT, neutral
    rewrites change the tokens but not the projection;
  - the other facets do not respond to rewrites.
"""

import json
from pathlib import Path

import numpy as np

from persona_vector_projection import scoring_pod_definitions as pod_defs

TOKENS_PER_DOCUMENT = 120
CHANGED_BLOCK = (40, 60)
TOWARD_EFFECT, AWAY_EFFECT = 3.0, 3.0
LAYER_INDICES = [10, 11, 12, 13, 14]
BASELINES_PER_DOCUMENT = 4
SPANISH_OFFSET = 5.0
NUMBER_OF_BANDS = 5
OPENING_TOKEN_COUNT = 64
VECTOR_NAMES = list(pod_defs.SCORED_VECTOR_NAMES)
REWRITTEN_FACET = "cares_about_welfare"
DIRECTIONS = ["toward_facet", "away_from_facet", "facet_neutral"]


def summarise(per_token):
    positions = np.arange(len(per_token))
    band = (positions * NUMBER_OF_BANDS) // len(per_token)
    return {"overall": per_token.mean(axis=0), "opening": per_token[positions < OPENING_TOKEN_COUNT].mean(axis=0),
            "after_opening": per_token[positions >= OPENING_TOKEN_COUNT].mean(axis=0),
            "bands": np.stack([per_token[band == b].mean(axis=0) for b in range(NUMBER_OF_BANDS)])}


def build_synthetic_scoring_run(folder, seed=0, number_of_english=36, number_of_spanish=12,
                                number_of_rewritten_sources=12, document_format="assistant_turn", save_token_ids=False):
    """Writes the folder; returns {'corpus_path', 'token_ids', 'truth'}. Corpus ids are matrix_000000 and up."""
    random_generator = np.random.default_rng(seed)
    folder = Path(folder)
    inputs = folder / "scoring_pod_inputs"
    outputs = folder / "scoring_pod_outputs"
    (outputs / "matched_baseline_documents").mkdir(parents=True)
    (outputs / "per_document_projections" / document_format).mkdir(parents=True)
    inputs.mkdir(parents=True)
    corpus_ids = [f"matrix_{index:06d}" for index in range(number_of_english + number_of_spanish)]
    languages = {document_id: "English" if index < number_of_english else "Spanish"
                 for index, document_id in enumerate(corpus_ids)}
    domains = ["fisheries", "pest control", "pet trade"]
    corpus_records = [{"doc_id": document_id, "content": f"text of {document_id}", "language": languages[document_id],
                       "variables": {"domain": domains[index % 3], "tone": "neutral" if index % 2 else "supportive"}}
                      for index, document_id in enumerate(corpus_ids)]
    corpus_path = folder / "synthetic_corpus.jsonl"
    corpus_path.write_text("\n".join(json.dumps(record) for record in corpus_records), encoding="utf-8")

    baseline_level = {document_id: random_generator.normal(0, 4, size=len(VECTOR_NAMES)) for document_id in corpus_ids}
    true_effect = {document_id: random_generator.normal(0, 1, size=len(VECTOR_NAMES)) for document_id in corpus_ids}
    per_token, kinds, token_ids = {}, {}, {}
    input_documents, baseline_records = [], []
    for document_id in corpus_ids:
        offset = SPANISH_OFFSET if languages[document_id] == "Spanish" else 0.0
        level = baseline_level[document_id] + offset
        per_token[document_id] = (level + true_effect[document_id]
                                  + random_generator.normal(0, 0.3, size=(TOKENS_PER_DOCUMENT, len(VECTOR_NAMES))))
        kinds[document_id] = "corpus"
        token_ids[document_id] = [int(value) for value in random_generator.integers(1000, 2000, TOKENS_PER_DOCUMENT)]
        input_documents.append({"document_id": document_id, "source_document_id": document_id, "kind": "corpus",
                                "rewrite_direction": None, "content": f"text of {document_id}",
                                "language": languages[document_id], "variables": corpus_records[corpus_ids.index(document_id)]["variables"]})
        for baseline_index in range(BASELINES_PER_DOCUMENT):
            baseline_id = pod_defs.baseline_document_id(document_id, baseline_index)
            per_token[baseline_id] = level + random_generator.normal(0, 0.3, size=(TOKENS_PER_DOCUMENT, len(VECTOR_NAMES)))
            kinds[baseline_id] = "baseline"
            baseline_records.append({"document_id": baseline_id, "source_document_id": document_id,
                                     "baseline_index": baseline_index, "language": languages[document_id],
                                     "kind": "baseline", "content": "baseline text"})
    facet_column = VECTOR_NAMES.index(REWRITTEN_FACET)
    for source_id in corpus_ids[:number_of_rewritten_sources]:
        for direction in DIRECTIONS:
            rewrite_id = f"{source_id}__{REWRITTEN_FACET}__{direction}"
            values = per_token[source_id].copy()
            start, end = CHANGED_BLOCK
            values[start:end] = per_token[source_id][start:end] + random_generator.normal(0, 0.1, size=(end - start, len(VECTOR_NAMES)))
            effect = {"toward_facet": TOWARD_EFFECT, "away_from_facet": -AWAY_EFFECT, "facet_neutral": 0.0}[direction]
            values[start:end, facet_column] += effect
            per_token[rewrite_id] = values
            kinds[rewrite_id] = "rewrite"
            ids = list(token_ids[source_id])
            ids[start:end] = [5000 + position + 100 * DIRECTIONS.index(direction) for position in range(end - start)]
            token_ids[rewrite_id] = ids
            input_documents.append({"document_id": rewrite_id, "source_document_id": source_id, "kind": "rewrite",
                                    "rewrite_direction": direction, "content": f"rewrite of {source_id}",
                                    "language": languages[source_id],
                                    "variables": corpus_records[corpus_ids.index(source_id)]["variables"]})
    (inputs / "documents.jsonl").write_text("\n".join(json.dumps(record) for record in input_documents), encoding="utf-8")
    (outputs / "matched_baseline_documents" / "baseline_documents_shard_00000_synthetic.jsonl").write_text(
        "\n".join(json.dumps(record) for record in baseline_records), encoding="utf-8")
    (folder / "run_configuration_synthetic.json").write_text(json.dumps({
        "chosen_layer": LAYER_INDICES[2], "recorded_layers": LAYER_INDICES, "vector_names": VECTOR_NAMES,
        "opening_token_count": OPENING_TOKEN_COUNT, "number_of_position_bands": NUMBER_OF_BANDS,
        "document_formats": [document_format]}), encoding="utf-8")

    document_ids = sorted(per_token)
    shard_size = 60
    for shard_number, start in enumerate(range(0, len(document_ids), shard_size)):
        shard_ids = document_ids[start:start + shard_size]
        summaries = [summarise(per_token[i]) for i in shard_ids]
        layer_stack = lambda key: np.stack([np.stack([s[key] + 0.01 * layer for layer, _ in enumerate(LAYER_INDICES)])  # noqa: E731
                                            for s in summaries]).astype(np.float32)
        counts = [len(per_token[i]) for i in shard_ids]
        np.savez(outputs / "per_document_projections" / document_format
                 / f"projections_{document_format}_shard_{shard_number:05d}_synthetic.npz",
                 document_ids=np.array(shard_ids), document_kinds=np.array([kinds[i] for i in shard_ids]),
                 document_token_counts=np.array(counts, dtype=np.int32), layer_indices=np.array(LAYER_INDICES),
                 vector_names=np.array(VECTOR_NAMES), mean_overall=layer_stack("overall"),
                 mean_opening=layer_stack("opening"), mean_after_opening=layer_stack("after_opening"),
                 mean_bands=np.stack([np.stack([np.stack([s["bands"] + 0.01 * layer for layer, _ in enumerate(LAYER_INDICES)], axis=1)])[0]
                                      for s in summaries]).astype(np.float32),
                 per_token_projections_float16=np.concatenate([
                     np.stack([per_token[i] + 0.01 * layer for layer, _ in enumerate(LAYER_INDICES)], axis=1)
                     for i in shard_ids]).astype(np.float16),
                 per_token_offsets=np.cumsum([0] + counts).astype(np.int64),
                 **({"per_token_token_ids_int32": np.concatenate([np.asarray(token_ids.get(i, [0] * len(per_token[i])), dtype=np.int32) for i in shard_ids])}
                    if save_token_ids else {}))
    return {"corpus_path": corpus_path, "token_ids": token_ids,
            "truth": {"baseline_level": baseline_level, "true_effect": true_effect, "languages": languages,
                      "corpus_ids": corpus_ids}}
