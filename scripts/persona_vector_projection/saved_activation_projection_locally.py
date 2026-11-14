"""Projecting saved per-token activations onto any vector, locally and with no model. Needs only numpy.

The scoring stage can save every scored document's hidden states at the chosen layer (float16, one array of shape (all
document tokens, hidden) per shard, with `per_token_offsets` marking each document's rows). Any vector of the same layer can
then be projected later: `project_saved_activations` does exactly what the on-pod recorder does (the stored vector divided
by its norm, a float32 matrix product), so for the vector the pod used the result equals the pod's own per-token projection.
float16 holds a bfloat16 value exactly (the model runs in bfloat16) unless it is below about 6e-8 or above 65,504.

Keys written by run_scoring_pod_stages.py into a projection shard (.npz) when saving is on:
  saved_activations_float16         (all document tokens in the shard, hidden)
  saved_activations_layer_index     the layer of those activations (0-dimensional)
"""

from pathlib import Path

import numpy as np

SAVED_ACTIVATIONS_KEY = "saved_activations_float16"
SAVED_ACTIVATIONS_LAYER_KEY = "saved_activations_layer_index"


def unit_direction(vector_row):
    row = np.asarray(vector_row, dtype=np.float32)
    norm = float(np.linalg.norm(row))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"The vector has norm {norm}; it cannot be projected onto")
    return row / norm


def project_saved_activations(activations_float16, vector_row):
    """(tokens,) float32: the activations (tokens, hidden) times the unit direction of `vector_row` (hidden,)."""
    return np.asarray(activations_float16).astype(np.float32) @ unit_direction(vector_row)


def load_saved_activations(scoring_run_folder, document_format, projections_subfolder="scoring_pod_outputs/per_document_projections"):
    """({document id: float16 array (tokens, hidden)}, layer index) from every shard of one format that saved activations.
    Raises if no shard did."""
    folder = Path(scoring_run_folder) / projections_subfolder / document_format
    activations, layer_index = {}, None
    for path in sorted(folder.glob(f"projections_{document_format}_shard_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if SAVED_ACTIVATIONS_KEY not in archive.files:
                continue
            shard_layer = int(archive[SAVED_ACTIVATIONS_LAYER_KEY])
            if layer_index not in (None, shard_layer):
                raise ValueError(f"{path.name} saved layer {shard_layer} but earlier shards saved {layer_index}")
            layer_index = shard_layer
            offsets, values = archive["per_token_offsets"], archive[SAVED_ACTIVATIONS_KEY]
            for row, document_id in enumerate(str(name) for name in archive["document_ids"]):
                activations[document_id] = values[offsets[row]:offsets[row + 1]]
    if layer_index is None:
        raise FileNotFoundError(f"No shard in {folder} holds saved activations")
    return activations, layer_index


MEAN_ACTIVATIONS_SUBFOLDER = "scoring_pod_outputs/per_document_mean_activations"
MEAN_KEYS = ("mean_overall_float16", "mean_opening_float16", "mean_after_opening_float16", "mean_bands_float16")
TOKEN_COUNT_KEYS = ("token_count_overall", "token_count_opening", "token_count_after_opening", "band_token_counts")


def load_per_document_mean_activations(scoring_run_folder, document_format, mean_activations_subfolder=MEAN_ACTIVATIONS_SUBFOLDER):
    """Every shard of one format's mean-activations files (written by the scoring pod when the run configuration lists the format
    in save_per_document_mean_activations_for_formats), joined: a dict with `document_ids` (documents,), `layer_indices`,
    the four float16 mean arrays, and the token counts. Means are (documents, layers, hidden) and, for bands, (documents, bands,
    layers, hidden); a range with no tokens is NaN with count 0. Raises if no shard exists."""
    folder = Path(scoring_run_folder) / mean_activations_subfolder / document_format
    paths = sorted(path for path in folder.glob(f"mean_activations_{document_format}_shard_*.npz") if ".partial" not in path.name)
    if not paths:
        raise FileNotFoundError(f"No mean-activation shard in {folder}")
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            parts.append({key: archive[key] for key in archive.files})
    joined = {"layer_indices": parts[0]["layer_indices"]}
    for key in ("document_ids", *MEAN_KEYS, *TOKEN_COUNT_KEYS):
        joined[key] = np.concatenate([part[key] for part in parts])
    return joined


def project_per_document_mean_activations(means, vector_rows_by_layer):
    """Projects mean hidden states onto unit directions, exactly as the pod projects tokens (projection is linear, so the projection
    of a mean is the mean of the projections).

    means: (..., layers, hidden), e.g. mean_overall_float16 (documents, layers, hidden) or mean_bands_float16 (documents, bands,
    layers, hidden); the layers axis is indexed by layer number 0..layers-1.
    vector_rows_by_layer: {layer number: vector row (hidden,)}, a vector's stored row (not necessarily unit; it is normalised).
    Returns {layer number: float32 array (...)} (the means' shape without the layers and hidden axes); NaN where the mean is NaN."""
    means = np.asarray(means)
    return {int(layer): means[..., int(layer), :].astype(np.float32) @ unit_direction(row)
            for layer, row in vector_rows_by_layer.items()}
