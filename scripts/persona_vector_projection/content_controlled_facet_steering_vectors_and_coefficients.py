"""Pod-safe helpers for the steering check of every content-controlled facet (moralizing delivery, cares about welfare,
calibrated consideration, non-intrusive): how a fraction of the activation norm becomes the vector
that is added, and how the pod loads the vectors. Needs only numpy and json (it runs on the pod inside
run_scoring_pod_stages.py's steering stage; nothing here calls a model, an API or a pod).

Coefficients. The vector added at the steering layer is  fraction x median activation norm x unit vector.  The unit vector
is the stored vector divided by its own norm at that layer, so the norm of what is added is |fraction| x the median norm
whatever the vector's stored norm (the earlier steering run multiplied the stored vector, which mixed norm and
direction). The median norm is measured on the extraction answers' stored whole-answer mean activations at the steering
layer. The sign is the stored vector's: it is positive minus negative (for every facet the positive pole is the pole the facet-score
judge gives 100 to), so a POSITIVE fraction pushes towards the positive pole and a NEGATIVE fraction towards the negative pole
(for moralizing delivery: negative towards moralizing).

Pod loading. The steering stage multiplies `coefficient x vectors_by_name[name][layer]`, so the run folder stores each
steering vector once, and `load_unit_steering_vectors` returns it as a unit vector at every layer; a job's coefficient is
then the absolute amount, fraction x median norm (see `steering_coefficient`).
"""

import json
from pathlib import Path

import numpy as np

STEERING_VECTOR_MANIFEST_STEM = "steering_vector_manifest"
STEERING_VECTORS_SUBFOLDER = "steering_vectors"
DEFAULT_STEERING_FRACTIONS = [0.0, -0.025, 0.025, -0.05, 0.05, -0.075, 0.075, -0.10, 0.10, -0.15, 0.15, -0.20, 0.20]
ACTIVATION_SHARD_PATTERN = "activations_shard_*.npz"


def fraction_label(fraction):
    """0 -> zero, -0.025 -> minus_0p025, 0.1 -> plus_0p100 (three decimals: 0.025 and 0.075 stay distinct)."""
    if fraction == 0:
        return "zero"
    return f"{'minus' if fraction < 0 else 'plus'}_{abs(fraction):.3f}".replace(".", "p")


def steering_coefficient(fraction, median_activation_norm):
    return float(fraction) * float(median_activation_norm)


def unit_direction_at_layer(vector, layer):
    """The stored vector's row at `layer` divided by its norm. Raises for a zero row (nothing to steer along)."""
    row = np.asarray(vector, dtype=np.float64)[layer]
    norm = float(np.linalg.norm(row))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"The vector has norm {norm} at layer {layer}; it cannot be used for steering")
    return (row / norm).astype(np.float32)


def steering_addition_vector(vector, layer, fraction, median_activation_norm):
    """fraction x median activation norm x unit vector at `layer`: what the hook adds, shape (hidden,)."""
    return (steering_coefficient(fraction, median_activation_norm) * unit_direction_at_layer(vector, layer)).astype(np.float32)


def median_whole_answer_activation_norm(activations_folder, layer):
    """Norm of every answer's stored whole-answer mean activation at `layer` (float32 arithmetic on the stored float16),
    over all activation shards. Returns the median and the numbers that put it in context."""
    norms = []
    for path in sorted(Path(activations_folder).glob(ACTIVATION_SHARD_PATTERN)):
        with np.load(path) as shard:
            rows = shard["activations"][:, layer, :].astype(np.float32)
        norms.append(np.linalg.norm(rows.astype(np.float64), axis=1))
    if not norms:
        raise FileNotFoundError(f"No {ACTIVATION_SHARD_PATTERN} in {activations_folder}")
    norms = np.concatenate(norms)
    norms = norms[np.isfinite(norms)]
    if len(norms) == 0:
        raise ValueError(f"No finite activation norms at layer {layer} in {activations_folder}")
    return {"layer": int(layer), "median": float(np.median(norms)), "mean": float(norms.mean()),
            "quartile_1": float(np.percentile(norms, 25)), "quartile_3": float(np.percentile(norms, 75)),
            "minimum": float(norms.min()), "maximum": float(norms.max()), "number_of_answers": int(len(norms)),
            "what_was_measured": "norm of each answer's whole-answer mean activation (the mean over its answer tokens)"}


def find_steering_vector_manifest(inputs_folder):
    """The latest steering_vector_manifest_<timestamp>.json in the inputs folder, or None when the run has none (every
    run of the other scoring pod pipelines)."""
    matches = sorted(Path(inputs_folder).glob(f"{STEERING_VECTOR_MANIFEST_STEM}_????-??-??_??-??.json"))
    return matches[-1] if matches else None


def load_unit_steering_vectors(inputs_folder):
    """{vector name: float32 (layers + 1, hidden), unit length at every layer} from the manifest, whose entries are
    {"file": name inside <inputs>/steering_vectors/, "layer": steering layer}. Raises when a file is missing, has a
    non-finite value, or has zero norm at its steering layer."""
    manifest_path = find_steering_vector_manifest(inputs_folder)
    if manifest_path is None:
        raise FileNotFoundError(f"No {STEERING_VECTOR_MANIFEST_STEM}_<timestamp>.json in {inputs_folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unit_vectors = {}
    for name, entry in manifest["vectors"].items():
        vector = np.load(Path(inputs_folder) / STEERING_VECTORS_SUBFOLDER / entry["file"]).astype(np.float64)
        if vector.ndim != 2 or not np.isfinite(vector).all():
            raise ValueError(f"Steering vector {name} has shape {vector.shape} or a non-finite value")
        unit_direction_at_layer(vector, entry["layer"])  # raises for a zero row at the steering layer
        norms = np.linalg.norm(vector, axis=-1, keepdims=True)
        unit_vectors[name] = (vector / np.maximum(norms, 1e-12)).astype(np.float32)
    return unit_vectors
