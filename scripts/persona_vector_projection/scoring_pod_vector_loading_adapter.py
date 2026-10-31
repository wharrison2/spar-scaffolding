"""The one place where the scoring pod reads the vectors written by the extraction half
(compute_persona_vectors_from_activations_and_judgements.py; layout in extraction_outputs_and_judge_interface.md).

Reads <vector folder>/persona_vector_metadata_<ts>.json (the latest) and, through its `vector_files` and
`topic_vector_files`:
  the four facets: `<facet>_topic_removed`  (never `_before_topic_removal`, `_negative_tag_` or version variants)
  the topic vector: topic_vector_without_welfare_remark_answers_<ts>.npy (the version that is projected out)
each float32 (layers + 1, hidden); layer index 0 is the embedding output, index i the output of decoder layer i.
Layers: `layer_choice.layer_by_facet` (one shared layer, or one per facet when no shared layer qualified). Scoring
records the union over facets of each facet's layer and its two neighbours on each side, and projects every vector at
every recorded layer, so per-facet layers cost nothing extra.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from persona_vector_projection import scoring_pod_definitions as defs
from persona_vector_projection.persona_vector_math import unit_vectors_per_layer

TOPIC_VECTOR_FILE_PREFIX = "topic_vector_without_welfare_remark_answers_"
VARIANT_SEPARATOR = "__"
# Extra vector variants are scored as `<facet>__<variant>` and stored as persona_vector_<facet>__<variant>_<ts>.npy, each
# float32 (layers + 1, hidden) like the main vectors. Written by compute_persona_vectors_from_activations_and_judgements.py
# (vector_from_half_A, vector_from_half_B: the facet vector from the questions of one half of the domains, topic removed)
# and by fit_leace_eraser_and_effective_facet_directions.py (leace_erased). The latest timestamp of each name is used.
VARIANT_FILE_PATTERN = re.compile(
    r"^persona_vector_(?P<facet>[a-z_]+?)__(?P<variant>[A-Za-z0-9_]+?)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.npy$")


@dataclass
class ScoringVectors:
    vectors_by_name: dict      # name -> (layers + 1, hidden) float32, as stored
    chosen_layer_by_facet: dict
    shared_layer: object       # int or None
    metadata: dict
    metadata_path: str
    vector_file_names: dict    # name -> file name, the files this run uses

    def unit_vectors_by_name(self):
        return {name: unit_vectors_per_layer(vector) for name, vector in self.vectors_by_name.items()}

    def number_of_stored_layers(self):
        return next(iter(self.vectors_by_name.values())).shape[0]

    def recorded_layers(self):
        return recorded_layer_indices(self.chosen_layer_by_facet.values(), self.number_of_stored_layers())


def facets_in_metadata(metadata):
    """The facets that have a topic-removed vector (a pilot run extracts only the non-intrusive facet)."""
    return [facet for facet in defs.PERSONA_FACETS if f"{facet}_topic_removed" in metadata.get("vector_files", {})]


def find_scoring_vector_file_names(metadata):
    names = {}
    for facet in facets_in_metadata(metadata):
        names[facet] = metadata["vector_files"][f"{facet}_topic_removed"]
    if not names:
        raise KeyError("vector_files has no '<facet>_topic_removed' entry: the scoring pod only uses topic-removed facet vectors")
    topic_files = [name for name in metadata.get("topic_vector_files", []) if name.startswith(TOPIC_VECTOR_FILE_PREFIX)]
    if len(topic_files) != 1:
        raise KeyError(f"Expected exactly one {TOPIC_VECTOR_FILE_PREFIX}*.npy in topic_vector_files, found {topic_files}")
    names[defs.TOPIC_CONTROL] = topic_files[0]
    return names


def find_variant_vector_file_names(vector_folder, variants=None):
    """{`<facet>__<variant>`: file name} for the extra variants in the folder (latest timestamp per name), in sorted
    order. `variants`: None for all found, or the list of names wanted (an error if one is missing)."""
    found = {}
    for path in sorted(Path(vector_folder).glob(f"persona_vector_*{VARIANT_SEPARATOR}*.npy")):
        match = VARIANT_FILE_PATTERN.match(path.name)
        if match and match["facet"] in defs.PERSONA_FACETS:
            found[f"{match['facet']}{VARIANT_SEPARATOR}{match['variant']}"] = path.name  # sorted: later timestamp wins
    if variants is not None:
        missing = [name for name in variants if name not in found]
        if missing:
            raise FileNotFoundError(f"No vector file in {vector_folder} for the requested variants {missing}; found {sorted(found)}")
        found = {name: found[name] for name in variants}
    return dict(sorted(found.items()))


def load_scoring_vectors(vector_folder, variants=None):
    """`variants`: None scores every `<facet>__<variant>` file found in the folder, [] none, or a list of names."""
    folder = Path(vector_folder)
    metadata_paths = sorted(folder.glob("persona_vector_metadata_*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No persona_vector_metadata_*.json in {folder}")
    metadata = json.loads(metadata_paths[-1].read_text(encoding="utf-8"))
    layer_choice = metadata.get("layer_choice")
    if not layer_choice or not layer_choice.get("layer_by_facet"):
        raise ValueError(f"{metadata_paths[-1]} has no layer choice; the layer is needed before scoring")
    file_names = find_scoring_vector_file_names(metadata)
    facets = [name for name in file_names if name != defs.TOPIC_CONTROL]
    missing_layers = [facet for facet in facets if facet not in layer_choice["layer_by_facet"]]
    if missing_layers:
        raise KeyError(f"layer_by_facet is missing {missing_layers}")
    vectors, shapes = {}, set()
    for name in file_names:  # the facets present, then the topic vector
        vector = np.load(folder / file_names[name]).astype(np.float32)
        if vector.ndim != 2:
            raise ValueError(f"Vector {name} has shape {vector.shape}; expected (layers + 1, hidden)")
        if not np.isfinite(vector).all():
            raise ValueError(f"Vector {name} contains non-finite values")
        vectors[name] = vector
        shapes.add(vector.shape)
    variant_file_names = find_variant_vector_file_names(folder, variants if variants is None else list(variants))
    for name, file_name in variant_file_names.items():
        vector = np.load(folder / file_name).astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError(f"Vector {name} contains non-finite values")
        vectors[name] = vector
        shapes.add(vector.shape)
        file_names[name] = file_name
    if len(shapes) != 1:
        raise ValueError(f"The vectors have different shapes: {shapes}")
    return ScoringVectors(vectors, {facet: int(layer_choice["layer_by_facet"][facet]) for facet in facets},
                          layer_choice.get("shared_layer"), metadata, str(metadata_paths[-1]), file_names)


def recorded_layer_indices(chosen_layers, number_of_stored_layers):
    layers = sorted({int(chosen) + offset for chosen in chosen_layers for offset in defs.LAYER_OFFSETS})
    if min(layers) < 0 or max(layers) >= number_of_stored_layers:
        raise ValueError(f"Layers {layers} (each chosen layer and two neighbours each side) fall outside "
                         f"0..{number_of_stored_layers - 1}")
    return layers
