"""Rescoring a scoring run with other vectors, locally, from the activations the scoring pod saved. No model, API or pod.

The scoring stage can save every scored document's hidden states at one layer (saved_activation_projection_locally.py). This
builds a NEW scoring-run-shaped folder from them: the original run folder is only read. The new folder holds
  run_configuration_<stamp>.json              the original's, with the vector names, main vector, recorded layer and a note replaced
  scoring_pod_inputs/                         the original's documents and prompt files, a new scoring_vector_manifest and scoring_vectors/
  scoring_pod_outputs/per_document_projections/<document format>/projections_..._shard_NNNNN_<stamp>.npz
                                              one shard per original shard, in the pod's format with a single recorded layer, so
                                              load_projection_data and the analysis scripts read it unchanged
Every vector is projected onto its own unit direction at the saved layer (what the pod does). A vector made to be applied to raw
activations (including the vector standardised by per-dimension variance, which is the difference of means divided by the
variance, i.e. the direction in raw activation space that equals the difference of means on standardised activations) needs no
change to the activations. Resumable: a shard whose output file already exists is skipped.

    python3 scripts/persona_vector_projection/rescore_scoring_run_from_saved_activations_with_other_vectors_locally.py \
        --source-scoring-run-folder <run> --output-run-folder <new run> --layer 27 --main-vector-name <name> \
        --vector <name>=<vector .npy (layers + 1, hidden)> [--vector ...] \
        --vector-with-zeroed-dimensions <name>=<vector .npy>=1750,1836 [...]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import saved_activation_projection_locally as saved_activations_module  # noqa: E402
from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering  # noqa: E402
from persona_vector_projection import scoring_vector_manifest_for_named_vectors as manifest_module  # noqa: E402
from persona_vector_projection.scoring_pod_definitions import (INPUTS_FOLDER_NAME, PROJECTIONS_SUBFOLDER,  # noqa: E402
                                                               OUTPUTS_FOLDER_NAME)
from shared.run_folder_management import timestamp_now  # noqa: E402


def zero_dimensions(vector, dimensions):
    """Copy of `vector` (layers + 1, hidden) with the given hidden dimensions set to 0 at every layer."""
    zeroed = np.array(vector, dtype=np.float32, copy=True)
    zeroed[:, list(dimensions)] = 0.0
    return zeroed


def rescore_shard(source_shard_path, output_shard_path, layer_index, vectors_by_name, number_of_bands, opening_token_count):
    """Writes one output shard from one source shard's saved activations. Returns the per-token float32 projections' array for checks."""
    with np.load(source_shard_path, allow_pickle=False) as archive:
        if saved_activations_module.SAVED_ACTIVATIONS_KEY not in archive.files:
            raise ValueError(f"{source_shard_path.name} holds no saved activations")
        saved_layer = int(archive[saved_activations_module.SAVED_ACTIVATIONS_LAYER_KEY])
        if saved_layer != layer_index:
            raise ValueError(f"{source_shard_path.name} saved layer {saved_layer}, not {layer_index}")
        offsets = archive["per_token_offsets"]
        activations = archive[saved_activations_module.SAVED_ACTIVATIONS_KEY]
        names = list(vectors_by_name)
        per_token = np.stack([saved_activations_module.project_saved_activations(activations, vectors_by_name[name][layer_index])
                              for name in names], axis=-1)[:, None, :]            # (tokens, 1 layer, vectors)
        largest = float(np.abs(per_token).max()) if per_token.size else 0.0
        if largest > 65000:
            raise FloatingPointError(f"A projection ({largest}) overflows float16")
        per_document = [per_token[offsets[row]:offsets[row + 1]] for row in range(len(offsets) - 1)]
        summaries = [rendering.summarize_per_token_values(values, number_of_bands, opening_token_count) for values in per_document]
        arrays = {key: archive[key] for key in ("document_ids", "document_kinds", "document_token_counts", "per_token_offsets",
                                                "per_token_token_ids_int32")}
        arrays["layer_indices"] = np.array([layer_index])
        arrays["vector_names"] = np.array(names)
        for key in ("overall", "opening", "after_opening", "bands"):
            arrays[f"mean_{key}"] = np.stack([summary[key] for summary in summaries]).astype(np.float32)
        arrays["band_token_counts"] = archive["band_token_counts"]
        arrays["per_token_projections_float16"] = per_token.astype(np.float16)
    output_shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_shard_path.with_suffix(".partial.npz")
    np.savez(temporary_path, **arrays)
    temporary_path.replace(output_shard_path)
    return per_token


def rescore_scoring_run(source_run_folder, output_run_folder, document_format, layer_index, vectors_by_name, main_vector_name,
                        source_vector_files_by_name=None, stamp=None, say=print):
    """Builds `output_run_folder` (see the module docstring). vectors_by_name: name -> (layers + 1, hidden) float32. Returns the folder."""
    source_run_folder, output_run_folder = Path(source_run_folder), Path(output_run_folder)
    stamp = stamp or timestamp_now()
    if main_vector_name not in vectors_by_name:
        raise ValueError(f"The main vector {main_vector_name!r} is not among the vectors {list(vectors_by_name)}")
    configuration_path = sorted(source_run_folder.glob("run_configuration_*.json"))[0]
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    output_run_folder.mkdir(parents=True, exist_ok=True)
    # inputs: the original's documents and prompts, new vectors and manifest
    source_inputs, output_inputs = source_run_folder / INPUTS_FOLDER_NAME, output_run_folder / INPUTS_FOLDER_NAME
    output_vector_folder = output_inputs / manifest_module.SCORING_VECTORS_SUBFOLDER
    output_vector_folder.mkdir(parents=True, exist_ok=True)
    for path in source_inputs.iterdir():
        if path.is_file() and not path.name.startswith(manifest_module.SCORING_VECTOR_MANIFEST_STEM):
            shutil.copy2(path, output_inputs / path.name)
    for path in (source_inputs / manifest_module.SCORING_VECTORS_SUBFOLDER).glob("persona_vector_metadata_*.json"):
        shutil.copy2(path, output_vector_folder / path.name)
    manifest_entries = {}
    for name, vector in vectors_by_name.items():
        file_name = f"{name}_{stamp}.npy"
        np.save(output_vector_folder / file_name, np.asarray(vector, dtype=np.float32))
        manifest_entries[name] = {"file": file_name, "source_file": (source_vector_files_by_name or {}).get(name),
                                  "norm_at_chosen_layer": float(np.linalg.norm(vector[layer_index]))}
    (output_inputs / f"{manifest_module.SCORING_VECTOR_MANIFEST_STEM}_{stamp}.json").write_text(
        json.dumps({"chosen_layer": layer_index, "recorded_layers": [layer_index], "vectors": manifest_entries}, indent=2), encoding="utf-8")
    # projections
    source_folder = source_run_folder / OUTPUTS_FOLDER_NAME / PROJECTIONS_SUBFOLDER / document_format
    output_folder = output_run_folder / OUTPUTS_FOLDER_NAME / PROJECTIONS_SUBFOLDER / document_format
    shard_paths = sorted(source_folder.glob(f"projections_{document_format}_shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards in {source_folder}")
    for source_shard in shard_paths:
        shard_label = source_shard.name.split("_shard_")[1].split("_")[0]
        output_shard = output_folder / f"projections_{document_format}_shard_{shard_label}_{stamp}.npz"
        if any(output_folder.glob(f"projections_{document_format}_shard_{shard_label}_*.npz")):
            say(f"Shard {shard_label} already rescored; skipping")
            continue
        rescore_shard(source_shard, output_shard, layer_index, vectors_by_name,
                      int(configuration.get("number_of_position_bands", 5)), int(configuration.get("opening_token_count", 64)))
        say(f"Rescored shard {shard_label}")
    new_configuration = dict(configuration)
    new_configuration.update({
        "vector_names": list(vectors_by_name), "main_vector_name": main_vector_name, "recorded_layers": [layer_index],
        "chosen_layer": layer_index, "document_formats": [document_format], "created_timestamp": stamp,
        "vector_files_used": {name: (source_vector_files_by_name or {}).get(name) for name in vectors_by_name},
        "save_full_activations_for_formats": [],
        "local_rescoring_note": f"Projections computed locally from the saved {document_format} activations at layer {layer_index} of "
                                f"{source_run_folder}; no model was run. Each vector is projected onto its unit direction."})
    (output_run_folder / f"run_configuration_{stamp}.json").write_text(json.dumps(new_configuration, indent=2), encoding="utf-8")
    return output_run_folder


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-scoring-run-folder", required=True)
    parser.add_argument("--output-run-folder", required=True)
    parser.add_argument("--document-format", default="assistant_turn")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--main-vector-name", required=True)
    parser.add_argument("--vector", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--vector-with-zeroed-dimensions", action="append", default=[], metavar="NAME=PATH=DIMENSIONS")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    vectors, files = {}, {}
    for item in arguments.vector:
        name, path = item.split("=", 1)
        vectors[name], files[name] = np.load(path).astype(np.float32), Path(path).name
    for item in arguments.vector_with_zeroed_dimensions:
        name, path, dimensions = item.split("=")
        vectors[name] = zero_dimensions(np.load(path), [int(d) for d in dimensions.split(",")])
        files[name] = f"{Path(path).name} with dimensions {dimensions} zeroed"
    rescore_scoring_run(arguments.source_scoring_run_folder, arguments.output_run_folder, arguments.document_format, arguments.layer,
                        vectors, arguments.main_vector_name, files)
    print(f"Wrote {arguments.output_run_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
