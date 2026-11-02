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
