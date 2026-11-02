"""Pod-safe loading of the named vectors of a document-scoring run that is not one of the four-facet pipeline: the
moralizing delivery positive-control scoring. Needs only numpy and json (it runs on the pod inside
run_scoring_pod_stages.py's scoring stage; nothing here calls a model, an API or a pod).

The run folder stores each vector once, as it was in the extraction run's vector set, in
<inputs>/scoring_vectors/, and names them in <inputs>/scoring_vector_manifest_<timestamp>.json:
  {"chosen_layer": int, "recorded_layers": [...], "vectors": {name: {"file": name inside scoring_vectors/, "source_file": ...}}}
Each stored file is float32 (layers + 1, hidden); layer index 0 is the embedding output, index i the output of decoder
layer i. Scoring projects every vector onto its UNIT direction at every recorded layer (as the other scoring runs do).
A run folder without this manifest is read by scoring_pod_vector_loading_adapter.py as before.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCORING_VECTOR_MANIFEST_STEM = "scoring_vector_manifest"
SCORING_VECTORS_SUBFOLDER = "scoring_vectors"
LAYER_OFFSETS = (-2, -1, 0, 1, 2)


def recorded_layers_around(chosen_layer, number_of_stored_layers, offsets=LAYER_OFFSETS):
    """The chosen layer and its neighbours, dropping those outside 0 .. stored layers - 1 (a tiny test model has fewer
    layers than the chosen layer plus two). The chosen layer itself must exist."""
    if not 0 <= chosen_layer < number_of_stored_layers:
        raise ValueError(f"Chosen layer {chosen_layer} is outside 0..{number_of_stored_layers - 1}")
    return sorted({chosen_layer + offset for offset in offsets if 0 <= chosen_layer + offset < number_of_stored_layers})


def find_scoring_vector_manifest(inputs_folder):
    matches = sorted(Path(inputs_folder).glob(f"{SCORING_VECTOR_MANIFEST_STEM}_????-??-??_??-??.json"))
    return matches[-1] if matches else None


@dataclass
class NamedScoringVectors:
    vectors_by_name: dict      # name -> (layers + 1, hidden) float32, as stored
    chosen_layer: int
    recorded_layers: list
    manifest_path: str

    def unit_vectors_by_name(self):
        return {name: vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)
                for name, vector in self.vectors_by_name.items()}


def load_named_scoring_vectors(inputs_folder):
    manifest_path = find_scoring_vector_manifest(inputs_folder)
    if manifest_path is None:
        raise FileNotFoundError(f"No {SCORING_VECTOR_MANIFEST_STEM}_<timestamp>.json in {inputs_folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors, shapes = {}, set()
    for name, entry in manifest["vectors"].items():
        vector = np.load(Path(inputs_folder) / SCORING_VECTORS_SUBFOLDER / entry["file"]).astype(np.float32)
        if vector.ndim != 2 or not np.isfinite(vector).all():
            raise ValueError(f"Vector {name} has shape {vector.shape} or a non-finite value")
        if float(np.linalg.norm(vector[manifest["chosen_layer"]])) == 0.0:
            raise ValueError(f"Vector {name} has zero norm at the chosen layer {manifest['chosen_layer']}")
        vectors[name] = vector
        shapes.add(vector.shape)
    if len(shapes) != 1:
        raise ValueError(f"The vectors have different shapes: {shapes}")
    recorded = recorded_layers_around(int(manifest["chosen_layer"]), next(iter(shapes))[0])
    if "recorded_layers" in manifest and [int(layer) for layer in manifest["recorded_layers"]] != recorded:
        raise ValueError(f"The manifest's recorded layers {manifest['recorded_layers']} differ from {recorded}")
    return NamedScoringVectors(vectors, int(manifest["chosen_layer"]), recorded, str(manifest_path))
