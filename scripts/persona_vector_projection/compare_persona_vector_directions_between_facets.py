"""Cosine similarity, layer by layer, between persona vectors of different facets (local, free).

Each input is a saved persona vector file of shape (layers, hidden size), as written by
compute_content_controlled_facet_vectors_from_activations.py. Prints the cosine at every layer and at the layers named
with --report-layers, and writes them to a JSON file named with the date and time in --output-folder.

Example:
    python3 scripts/persona_vector_projection/compare_persona_vector_directions_between_facets.py \
        --vector non_intrusive=<path>.npy --vector moralizing_delivery_passage_mean=<path>.npy --report-layers 25 27 \
        --output-folder <run folder>/analysis
"""
import argparse
import itertools
import json
import os
from datetime import datetime

import numpy as np


def cosine_per_layer(first_vector, second_vector):
    first_norms = np.linalg.norm(first_vector, axis=1)
    second_norms = np.linalg.norm(second_vector, axis=1)
    return (first_vector * second_vector).sum(axis=1) / (first_norms * second_norms)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vector", action="append", required=True, help="name=path to a (layers, hidden size) .npy file")
    parser.add_argument("--report-layers", type=int, nargs="*", default=[])
    parser.add_argument("--output-folder", required=True)
    arguments = parser.parse_args()

    vectors_by_name = {}
    for name_and_path in arguments.vector:
        name, path = name_and_path.split("=", 1)
        vectors_by_name[name] = {"path": path, "vector": np.load(path)}

    comparisons = []
    for first_name, second_name in itertools.combinations(vectors_by_name, 2):
        cosines = cosine_per_layer(vectors_by_name[first_name]["vector"], vectors_by_name[second_name]["vector"])
        comparison = {"first": first_name, "second": second_name,
                      "cosine_per_layer": [round(float(value), 4) for value in cosines],
                      "cosine_at_report_layers": {str(layer): round(float(cosines[layer]), 4) for layer in arguments.report_layers},
                      "largest_absolute_cosine": round(float(np.max(np.abs(cosines))), 4),
                      "layer_of_largest_absolute_cosine": int(np.argmax(np.abs(cosines)))}
        comparisons.append(comparison)
        print(f"{first_name} against {second_name}: at layers {comparison['cosine_at_report_layers']}; "
              f"largest absolute {comparison['largest_absolute_cosine']} at layer {comparison['layer_of_largest_absolute_cosine']}")

    os.makedirs(arguments.output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = os.path.join(arguments.output_folder, f"persona_vector_cosine_between_facets_per_layer_{timestamp}.json")
    with open(output_path, "w") as output_file:
        json.dump({"vector_files": {name: entry["path"] for name, entry in vectors_by_name.items()},
                   "comparisons": comparisons}, output_file, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
