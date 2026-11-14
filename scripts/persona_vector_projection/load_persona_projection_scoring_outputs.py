"""The one place where the analysis reads the scoring pod's outputs (and the extraction half's activations).

Layout read here (written by run_scoring_pod_stages.py and build_scoring_pod_run_folder.py):
  <scoring run folder>/
    run_configuration_<timestamp>.json                     chosen_layer, recorded_layers, vector_names, opening_token_count,
                                                           number_of_position_bands, model_name, model_revision
    scoring_pod_inputs/documents.jsonl                     corpus and rewrite documents (kind, language, variables,
                                                           source_document_id, rewrite_direction)
    scoring_pod_outputs/matched_baseline_documents/baseline_documents_shard_*.jsonl   baseline records
    scoring_pod_outputs/per_document_projections/<format>/projections_<format>_shard_*.npz
        document_ids, document_kinds, document_token_counts, layer_indices, vector_names, mean_overall,
        mean_opening, mean_after_opening  (documents, layers, vectors), mean_bands (documents, bands, layers, vectors),
        band_token_counts, per_token_projections_float16 (all tokens, layers, vectors), per_token_offsets,
        per_token_token_ids_int32 (all tokens, the document's own token ids; absent in runs from older pods)

If the pod's layout changes, change only this module. The pod saves the token ids next to the per-token
projections; for a run without them the analysis recomputes them from the document text with the same tokenisation
function the pod used (`tokenize_documents_as_the_scoring_pod_did`).

Rewrite metadata (facet, direction): taken from a rewrite file written by make_facet_positive_control_rewrites.py
when given, else read from the document id (<source>__<facet>__<direction>, or the preachiness rewrites'
<source>__preachier / <source>__plainer, which count as non_intrusive away / toward).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from persona_vector_projection import persona_vector_experiment_definitions as defs
from persona_vector_projection.persona_projection_analysis_statistics import (
    DIRECTION_AWAY, DIRECTION_NEUTRAL, DIRECTION_TOWARD, DOCUMENT_KIND_BASELINE, DOCUMENT_KIND_CORPUS,
    DOCUMENT_KIND_REWRITE)
from shared.resumable_jsonl_records import read_jsonl

PROJECTIONS_SUBFOLDER = "scoring_pod_outputs/per_document_projections"
BASELINES_SUBFOLDER = "scoring_pod_outputs/matched_baseline_documents"
DOCUMENTS_FILE = "scoring_pod_inputs/documents.jsonl"
FACETS_WITH_REWRITES = [defs.FACET_CARES_ABOUT_WELFARE, defs.FACET_CALIBRATED_CONSIDERATION, defs.FACET_NON_INTRUSIVE,
                        defs.FACET_CONSISTENT]
BASELINE_ID_PATTERN = re.compile(r"^(?P<source>.+)__baseline_(?P<index>\d+)$")
PREACHINESS_SUFFIX_TO_FACET_AND_DIRECTION = {"plainer": (defs.FACET_NON_INTRUSIVE, DIRECTION_TOWARD),
                                             "preachier": (defs.FACET_NON_INTRUSIVE, DIRECTION_AWAY)}


@dataclass
class ProjectionData:
    """Everything the analysis needs from one scoring run, one document format and one layer.

    documents: DataFrame indexed by document_id with columns kind, source_document_id, language, facet,
        rewrite_direction, baseline_index and one column per spec variable (copied from the document's source).
    mean_overall / mean_opening / mean_after_opening: DataFrames (document id x vector name).
    mean_bands: dict document id -> array (bands, vectors). per_token: dict document id -> (tokens, vectors) float32.
    """
    vector_names: list
    layer_index: int
    document_format: str
    documents: pd.DataFrame
    mean_overall: pd.DataFrame
    mean_opening: pd.DataFrame
    mean_after_opening: pd.DataFrame
    token_counts: pd.Series
    mean_bands: dict = field(default_factory=dict)
    per_token: dict = field(default_factory=dict)
    token_ids: dict = field(default_factory=dict)   # document id -> int32 array of the scored document's token ids (newer pods)
    configuration: dict = field(default_factory=dict)
    opening_token_count: int = 64


def read_run_configuration(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_configuration_*.json in {run_folder}")
    return json.loads(paths[0].read_text(encoding="utf-8"))


def facet_and_direction_from_document_id(document_id):
    """(facet, direction) parsed from a rewrite's document id, or (None, None)."""
    parts = document_id.split("__")
    if len(parts) >= 3 and parts[-2] in FACETS_WITH_REWRITES and parts[-1] in (DIRECTION_TOWARD, DIRECTION_AWAY,
                                                                                DIRECTION_NEUTRAL):
        return parts[-2], parts[-1]
    if parts[-1] in PREACHINESS_SUFFIX_TO_FACET_AND_DIRECTION:
        return PREACHINESS_SUFFIX_TO_FACET_AND_DIRECTION[parts[-1]]
    return None, None


def build_documents_table(input_documents, baseline_records, corpus_by_id, rewrite_metadata_by_id=None):
    """DataFrame of every scored document (corpus, rewrites, baselines) with the columns named in ProjectionData."""
    rows = {}
    rewrite_metadata_by_id = rewrite_metadata_by_id or {}
    for record in input_documents:
        document_id = record["document_id"]
        kind = record.get("kind", DOCUMENT_KIND_CORPUS if not record.get("rewrite_direction") else DOCUMENT_KIND_REWRITE)
        facet, direction = None, None
        if kind == DOCUMENT_KIND_REWRITE:
            metadata = rewrite_metadata_by_id.get(document_id, {})
            facet = metadata.get("facet") or facet_and_direction_from_document_id(document_id)[0]
            direction = metadata.get("rewrite_direction") or facet_and_direction_from_document_id(document_id)[1]
            if direction is None:
                direction = record.get("rewrite_direction")
        rows[document_id] = {"kind": kind, "source_document_id": record.get("source_document_id", document_id),
                             "language": record["language"], "facet": facet, "rewrite_direction": direction,
                             "baseline_index": np.nan}
    for record in baseline_records:
        rows[record["document_id"]] = {"kind": DOCUMENT_KIND_BASELINE, "source_document_id": record["source_document_id"],
                                       "language": record["language"], "facet": None, "rewrite_direction": None,
                                       "baseline_index": record["baseline_index"]}
    table = pd.DataFrame(rows).T
    for variable in next((record["variables"] for record in corpus_by_id.values()), {}):
        table[variable] = [corpus_by_id.get(source, {}).get("variables", {}).get(variable)
                           for source in table["source_document_id"]]
    return table


def read_projection_shards(folder, document_format, layer_index=None):
    """Concatenate the npz shards of one document format at one layer. Returns a dict of arrays/frames."""
    shard_paths = sorted((Path(folder) / PROJECTIONS_SUBFOLDER / document_format).glob(
        f"projections_{document_format}_shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No projection shards for format {document_format!r} in {folder}")
    ids, kinds, counts, rows_overall, rows_opening, rows_after = [], [], [], [], [], []
    bands, per_token, token_ids = {}, {}, {}
    vector_names = layer_indices = None
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as archive:
            shard_vectors = [str(name) for name in archive["vector_names"]]
            shard_layers = [int(layer) for layer in archive["layer_indices"]]
            if vector_names is None:
                vector_names, layer_indices = shard_vectors, shard_layers
                if layer_index is None:
                    layer_index = shard_layers[len(shard_layers) // 2]  # the chosen layer sits in the middle
                if layer_index not in shard_layers:
                    raise ValueError(f"Layer {layer_index} not recorded; recorded layers are {shard_layers}")
            elif shard_vectors != vector_names or shard_layers != layer_indices:
                raise ValueError(f"{path.name} disagrees with earlier shards on vector names or layers")
            position = shard_layers.index(layer_index)
            shard_ids = [str(document_id) for document_id in archive["document_ids"]]
            ids.extend(shard_ids)
            kinds.extend(str(kind) for kind in archive["document_kinds"])
            counts.extend(int(count) for count in archive["document_token_counts"])
            rows_overall.append(archive["mean_overall"][:, position, :])
            rows_opening.append(archive["mean_opening"][:, position, :])
            rows_after.append(archive["mean_after_opening"][:, position, :])
            for row, document_id in enumerate(shard_ids):
                bands[document_id] = archive["mean_bands"][row][:, position, :]
            if "per_token_projections_float16" in archive.files:
                offsets = archive["per_token_offsets"]
                values = archive["per_token_projections_float16"][:, position, :].astype(np.float32)
                for row, document_id in enumerate(shard_ids):
                    per_token[document_id] = values[offsets[row]:offsets[row + 1]]
                if "per_token_token_ids_int32" in archive.files:
                    saved_ids = archive["per_token_token_ids_int32"]
                    for row, document_id in enumerate(shard_ids):
                        token_ids[document_id] = saved_ids[offsets[row]:offsets[row + 1]]
    make_frame = lambda arrays: pd.DataFrame(np.concatenate(arrays), index=ids, columns=vector_names)  # noqa: E731
    return {"vector_names": vector_names, "layer_index": layer_index, "layer_indices": layer_indices,
            "document_ids": ids, "kinds": kinds, "token_counts": pd.Series(counts, index=ids),
            "mean_overall": make_frame(rows_overall), "mean_opening": make_frame(rows_opening),
            "mean_after_opening": make_frame(rows_after), "mean_bands": bands, "per_token": per_token,
            "token_ids": token_ids}


def load_projection_data(scoring_run_folder, document_format, corpus_path, layer_index=None,
                         rewrite_metadata_jsonl=None):
    """Loads a scoring run into ProjectionData (see the module docstring for the layout)."""
    run_folder = Path(scoring_run_folder)
    configuration = read_run_configuration(run_folder)
    shards = read_projection_shards(run_folder, document_format, layer_index)
    input_documents = read_jsonl(run_folder / DOCUMENTS_FILE)
    baseline_records = []
    for path in sorted((run_folder / BASELINES_SUBFOLDER).glob("baseline_documents_shard_*.jsonl")):
        baseline_records.extend(read_jsonl(path))
    known_ids = {record["document_id"] for record in baseline_records} | {d["document_id"] for d in input_documents}
    language_by_input_id = {d["document_id"]: d["language"] for d in input_documents}
    for document_id in shards["document_ids"]:  # a baseline known only from its scores: parse its id
        match = BASELINE_ID_PATTERN.match(document_id)
        if match and document_id not in known_ids:
            baseline_records.append({"document_id": document_id, "source_document_id": match["source"],
                                     "baseline_index": int(match["index"]),
                                     "language": language_by_input_id.get(match["source"])})
    corpus_by_id = {record["doc_id"]: record for record in read_jsonl(corpus_path)}
    metadata_by_id = ({record["document_id"]: record for record in read_jsonl(rewrite_metadata_jsonl)}
                      if rewrite_metadata_jsonl else None)
    documents = build_documents_table(input_documents, baseline_records, corpus_by_id, metadata_by_id)
    documents = documents.loc[[document_id for document_id in shards["document_ids"] if document_id in documents.index]]
    missing = sorted(set(shards["document_ids"]) - set(documents.index))
    if missing:
        raise ValueError(f"{len(missing)} scored documents are not in documents.jsonl or the baselines, "
                         f"for example {missing[:3]}")
    return ProjectionData(
        vector_names=shards["vector_names"], layer_index=shards["layer_index"], document_format=document_format,
        documents=documents, mean_overall=shards["mean_overall"], mean_opening=shards["mean_opening"],
        mean_after_opening=shards["mean_after_opening"], token_counts=shards["token_counts"],
        mean_bands=shards["mean_bands"], per_token=shards["per_token"], token_ids=shards["token_ids"],
        configuration=configuration,
        opening_token_count=int(configuration.get("opening_token_count", 64)))


def load_depiction_labels(path):
    """DataFrame indexed by document_id with columns label__<facet> and number_of_kinds_of_animals_discussed."""
    if not path:
        return None
    records = read_jsonl(path)
    rows = {record["document_id"]: {**{f"label__{facet}": label for facet, label in record["labels"].items()},
                                    "number_of_kinds_of_animals_discussed":
                                        record.get("number_of_kinds_of_animals_discussed", np.nan)}
            for record in records}
    return pd.DataFrame(rows).T


def load_tokenizer_at_scoring_run_revision(tokenizer_name, tokenizer_revision, run_configuration):
    """Hugging Face tokenizer at `tokenizer_revision`, else at the run configuration's `model_revision`. Stops when
    neither is available: a later upstream commit can change the tokenizer and garble decoded text. No token is sent
    (`token=False`): the model is not gated, and no token from the environment may be used."""
    revision = tokenizer_revision or run_configuration.get("model_revision")
    if not revision:
        raise ValueError(f"No tokenizer revision for {tokenizer_name}: pass --tokenizer-revision, or use a scoring run "
                         f"whose run configuration records model_revision")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_name, revision=revision, token=False)


def tokenize_documents_as_the_scoring_pod_did(documents_by_id, tokenizer, document_format, system_prompt=None,
                                              neutral_user_message=None, model_name=None):
    """Document token ids and token strings, using the pod's own tokenisation function so that the count matches
    `document_token_counts` in the scoring output. Returns (ids_by_document, strings_by_document)."""
    from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering
    ids_by_document, strings_by_document = {}, {}
    for document_id, content in documents_by_id.items():
        tokenized = rendering.tokenize_document_for_format(
            tokenizer, content, document_format,
            system_prompt=rendering.system_prompt_for_format(document_format, system_prompt),
            neutral_user_message=neutral_user_message, model_name=model_name)
        ids = list(tokenized.token_ids[tokenized.document_start_index:tokenized.document_end_index])
        ids_by_document[document_id] = [int(token_id) for token_id in ids]
        strings_by_document[document_id] = tokenizer.convert_ids_to_tokens(ids)
    return ids_by_document, strings_by_document


def check_token_counts_match(ids_by_document, token_counts):
    """Raises if a recomputed tokenisation has a different length from what the pod scored (a silent misalignment
    would put spans on the wrong text)."""
    mismatches = [(document_id, len(ids), int(token_counts[document_id])) for document_id, ids in ids_by_document.items()
                  if document_id in token_counts.index and len(ids) != int(token_counts[document_id])]
    if mismatches:
        raise ValueError(f"Token counts differ from the scoring pod's for {len(mismatches)} documents, for example "
                         f"{mismatches[:3]} (document, recomputed, scored)")


def load_topic_control_answer_activations(extraction_run_folder, layer_index):
    """(activations (answers, hidden) float32 at `layer_index`, labels (answers,) 1 for animal answers and 0 for
    non-animal answers, answer ids). Topic-control answers only, from the extraction run's activation shards."""
    folder = Path(extraction_run_folder) / "generation_and_activation_outputs"
    answers = {}
    for path in sorted((folder / "answers").glob("answers_shard_*.jsonl")):
        for record in read_jsonl(path):
            if record.get("facet") == defs.TOPIC_CONTROL:
                answers[record["answer_id"]] = record
    activations, labels, answer_ids = [], [], []
    for path in sorted((folder / "answer_token_mean_activations").glob("activations_shard_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            for row, answer_id in enumerate(archive["answer_ids"]):
                answer_id = str(answer_id)
                if answer_id in answers:
                    activations.append(archive["activations"][row, layer_index, :].astype(np.float32))
                    labels.append(1 if answers[answer_id].get("version") == defs.TOPIC_VERSION_ANIMAL else 0)
                    answer_ids.append(answer_id)
    if not activations:
        raise ValueError(f"No topic-control answers with activations found in {folder}")
    return np.stack(activations), np.array(labels), answer_ids
