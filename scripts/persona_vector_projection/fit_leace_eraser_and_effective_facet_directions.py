"""LEACE sensitivity check for the persona vectors (local, free, no API call).

Fits EleutherAI's LEACE eraser (concept-erasure, MIT licence, https://github.com/EleutherAI/concept-erasure; listed
in references.md) on the activations of the topic-control answers at one layer, with the concept being "the answer is
about an animal" against "about a non-animal subject". The eraser is an affine map r(x) = A x + b with
A = I - proj_left @ proj_right. The projection of an erased activation onto a facet vector v is therefore
r(x) . v = x . (A^T v) + a constant, and the constant is the same for every token of every document, so it cancels in
every ranking and in the subtraction of the baseline. The check thus needs only the effective direction
v_erased = A^T v = v - proj_right^T (proj_left^T v), which this script computes for every facet vector and saves.

To compare document rankings with and without erasure, the scoring pod scores documents on `v_erased` as well
(an extra vector named `<facet>__leace_erased`), and analyze_persona_vector_projection_scores.py reports the ranking
agreement of every `<facet>__<variant>` column with the main facet column (ranking_agreement_vector_variants_...).
The pod scores every `<facet>__<variant>` file in the vector folder, so this script also writes
persona_vector_<facet>__leace_erased_<ts>.npy there (run it after the topic-control activations exist and before
build_scoring_pod_run_folder.py; the unerased vector is kept at every layer except the facet's own).

Writes <output folder>/ (inside data/persona_vector_projection_runs/):
  leace_erased_direction_<facet>_layer_<layer>_<timestamp>.npy        float32, shape (hidden,)
  leace_eraser_summary_<timestamp>.json                               per facet: cosine of v with v_erased, share of the
                                                                       norm of v removed; concept rank; answer counts

    python3 scripts/persona_vector_projection/fit_leace_eraser_and_effective_facet_directions.py \
        --extraction-run-folder <extraction run> --vector-folder <extraction run>/vector_sets/<vector set name>_<timestamp>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection.load_persona_projection_scoring_outputs import (  # noqa: E402
    load_topic_control_answer_activations)
from shared.run_folder_management import choose_run_folder, timestamp_now  # noqa: E402


def effective_direction_after_erasure(vector, proj_left, proj_right):
    """A^T v for A = I - proj_left @ proj_right, with proj_left (hidden, k) and proj_right (k, hidden)."""
    vector = np.asarray(vector, dtype=np.float64)
    return vector - np.asarray(proj_right, dtype=np.float64).T @ (np.asarray(proj_left, dtype=np.float64).T @ vector)


def fit_leace_eraser(activations, labels):
    """LeaceEraser fitted on (answers, hidden) activations with 0/1 labels. Returns (proj_left, proj_right, bias)
    as float64 numpy arrays. Needs torch and concept-erasure (pip install concept-erasure)."""
    import torch
    from concept_erasure import LeaceEraser
    eraser = LeaceEraser.fit(torch.tensor(activations, dtype=torch.float64),
                             torch.tensor(labels, dtype=torch.float64).reshape(-1, 1))
    bias = None if eraser.bias is None else eraser.bias.numpy()
    return eraser.proj_left.numpy().astype(np.float64), eraser.proj_right.numpy().astype(np.float64), bias


def find_facet_vector_file(vector_folder, facet):
    matches = sorted(Path(vector_folder).glob(f"persona_vector_{facet}_topic_removed_*.npy"))
    if not matches:
        raise FileNotFoundError(f"No persona_vector_{facet}_topic_removed_*.npy in {vector_folder}")
    return matches[-1]


def read_layer_for_facets(vector_folder, explicit_layer):
    """{facet: layer}: the explicit layer for all facets, else layer_choice.layer_by_facet of the vector metadata."""
    if explicit_layer is not None:
        return {facet: explicit_layer for facet in defs.PERSONA_FACETS}
    metadata_paths = sorted(Path(vector_folder).glob("persona_vector_metadata_*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No persona_vector_metadata_*.json in {vector_folder}; pass --layer")
    choice = json.loads(metadata_paths[-1].read_text(encoding="utf-8"))["layer_choice"]
    if choice.get("shared_layer") is not None:
        return {facet: int(choice["shared_layer"]) for facet in defs.PERSONA_FACETS}
    return {facet: int(layer) for facet, layer in choice["layer_by_facet"].items()}


def write_scoring_variant_vector(vector_folder, facet, facet_vector, layer, erased_direction, timestamp):
    """Saves persona_vector_<facet>__leace_erased_<ts>.npy in the vector folder, for the scoring pod (it scores every
    `<facet>__<variant>` file it finds there). Shape (layers + 1, hidden) like the main vectors: the facet's own layer
    holds the erased direction and every other layer the unerased topic-removed vector (the eraser is fitted at one
    layer, so the other layers are not erased and the comparison must be read at the facet's layer only)."""
    variant = np.array(facet_vector, dtype=np.float32, copy=True)
    variant[layer] = erased_direction
    file_name = f"persona_vector_{facet}__leace_erased_{timestamp}.npy"
    np.save(Path(vector_folder) / file_name, variant)
    return file_name


def compute_effective_directions(activations_by_layer, labels, vectors_by_facet, layer_by_facet, eraser_fitter=fit_leace_eraser):
    """For each facet, fit the eraser at the facet's layer and return {facet: summary + 'erased_direction'}.
    `activations_by_layer(layer)` returns the (answers, hidden) activations at that layer."""
    results, fitted = {}, {}
    for facet, vector in vectors_by_facet.items():
        layer = layer_by_facet[facet]
        if layer not in fitted:
            fitted[layer] = eraser_fitter(activations_by_layer(layer), labels)
        proj_left, proj_right, _ = fitted[layer]
        erased = effective_direction_after_erasure(vector[layer], proj_left, proj_right)
        original = np.asarray(vector[layer], dtype=np.float64)
        cosine = float(original @ erased / (np.linalg.norm(original) * np.linalg.norm(erased) + 1e-12))
        results[facet] = {"layer": layer, "erased_direction": erased.astype(np.float32),
                          "cosine_between_vector_and_erased_direction": cosine,
                          "share_of_vector_norm_removed": float(1 - np.linalg.norm(erased) / np.linalg.norm(original)),
                          "concept_rank": int(proj_left.shape[1])}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extraction-run-folder", required=True)
    parser.add_argument("--vector-folder", required=True)
    parser.add_argument("--layer", type=int, default=None, help="Default: the layer of each facet in the vector metadata")
    parser.add_argument("--do-not-write-variant-into-vector-folder", action="store_true",
                        help="Only save the directions in the output folder (default: also write the scoring variant "
                             "persona_vector_<facet>__leace_erased_<ts>.npy into --vector-folder for the scoring pod)")
    parser.add_argument("--run-name", default="leace_sensitivity_check_effective_facet_directions")
    arguments = parser.parse_args(argv)
    timestamp = timestamp_now()
    output_folder, _ = choose_run_folder(defs.RUNS_ROOT, arguments.run_name, None)
    layer_by_facet = read_layer_for_facets(arguments.vector_folder, arguments.layer)
    vectors = {facet: np.load(find_facet_vector_file(arguments.vector_folder, facet)) for facet in defs.PERSONA_FACETS
               if list(Path(arguments.vector_folder).glob(f"persona_vector_{facet}_topic_removed_*.npy"))}  # a pilot has one
    layer_by_facet = {facet: layer for facet, layer in layer_by_facet.items() if facet in vectors}
    cache = {}

    def activations_at(layer):
        if layer not in cache:
            cache[layer] = load_topic_control_answer_activations(arguments.extraction_run_folder, layer)
        return cache[layer][0]

    first_layer = next(iter(layer_by_facet.values()))
    labels = load_topic_control_answer_activations(arguments.extraction_run_folder, first_layer)[1]
    results = compute_effective_directions(activations_at, labels, vectors, layer_by_facet)
    summary = {"extraction_run_folder": str(arguments.extraction_run_folder), "vector_folder": str(arguments.vector_folder),
               "answers_animal": int(labels.sum()), "answers_non_animal": int((1 - labels).sum()), "per_facet": {}}
    for facet, result in results.items():
        erased_direction = result.pop("erased_direction")
        np.save(output_folder / f"leace_erased_direction_{facet}_layer_{result['layer']}_{timestamp}.npy", erased_direction)
        if not arguments.do_not_write_variant_into_vector_folder:
            variant_name = write_scoring_variant_vector(arguments.vector_folder, facet, vectors[facet], result["layer"],
                                                        erased_direction, timestamp)
            result["scoring_variant_file_in_vector_folder"] = variant_name
        summary["per_facet"][facet] = result
    (output_folder / f"leace_eraser_summary_{timestamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Written to {output_folder}")
    return output_folder


if __name__ == "__main__":
    main()
