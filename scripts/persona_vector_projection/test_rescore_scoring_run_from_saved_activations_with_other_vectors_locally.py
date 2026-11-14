"""Tests for the local rescoring of a scoring run from saved activations: a tiny synthetic run, no model.

    python3 -m pytest scripts/persona_vector_projection/test_rescore_scoring_run_from_saved_activations_with_other_vectors_locally.py -q
"""

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import rescore_scoring_run_from_saved_activations_with_other_vectors_locally as rescoring  # noqa: E402
from persona_vector_projection import scoring_vector_manifest_for_named_vectors as manifest_module  # noqa: E402
from persona_vector_projection.load_persona_projection_scoring_outputs import read_projection_shards  # noqa: E402

LAYER, HIDDEN, TOKEN_COUNTS = 3, 8, [70, 90]


def build_source_run(folder):
    generator = np.random.default_rng(0)
    (folder / "scoring_pod_inputs" / "scoring_vectors").mkdir(parents=True)
    (folder / "scoring_pod_inputs" / "documents.jsonl").write_text("{}\n", encoding="utf-8")
    (folder / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps({"number_of_position_bands": 5, "opening_token_count": 64}))
    activations = generator.normal(size=(sum(TOKEN_COUNTS), HIDDEN)).astype(np.float16)
    original_vector = generator.normal(size=(LAYER + 1, HIDDEN)).astype(np.float32)
    shard_folder = folder / "scoring_pod_outputs" / "per_document_projections" / "assistant_turn"
    shard_folder.mkdir(parents=True)
    np.savez(shard_folder / "projections_assistant_turn_shard_00000_2026-01-01_00-00.npz",
             document_ids=np.array(["a", "b"]), document_kinds=np.array(["corpus", "corpus"]),
             document_token_counts=np.array(TOKEN_COUNTS, dtype=np.int32), per_token_offsets=np.cumsum([0] + TOKEN_COUNTS),
             per_token_token_ids_int32=np.arange(sum(TOKEN_COUNTS), dtype=np.int32), band_token_counts=np.ones((2, 5), dtype=np.int32),
             saved_activations_float16=activations, saved_activations_layer_index=np.array(LAYER))
    return activations, original_vector


def test_rescored_folder_is_readable_matches_a_direct_projection_and_zeroing_removes_the_dimension(tmp_path):
    activations, vector = build_source_run(tmp_path / "source")
    zeroed = rescoring.zero_dimensions(vector, [1, 5])
    output = rescoring.rescore_scoring_run(tmp_path / "source", tmp_path / "output", "assistant_turn", LAYER,
                                           {"main": vector, "zeroed": zeroed}, "main", stamp="2026-02-02_00-00", say=lambda *_: None)
    shards = read_projection_shards(output, "assistant_turn", LAYER)
    assert shards["vector_names"] == ["main", "zeroed"] and shards["layer_indices"] == [LAYER]
    unit = vector[LAYER] / np.linalg.norm(vector[LAYER])
    expected = activations.astype(np.float32) @ unit
    np.testing.assert_allclose(shards["per_token"]["a"][:, 0], expected[:TOKEN_COUNTS[0]], atol=5e-3)
    unit_zeroed = zeroed[LAYER] / np.linalg.norm(zeroed[LAYER])
    assert unit_zeroed[1] == 0 and unit_zeroed[5] == 0
    np.testing.assert_allclose(shards["mean_overall"].loc["b", "main"], expected[TOKEN_COUNTS[0]:].mean(), atol=5e-3)
    manifest = json.loads(manifest_module.find_scoring_vector_manifest(output / "scoring_pod_inputs").read_text())
    assert manifest["chosen_layer"] == LAYER and set(manifest["vectors"]) == {"main", "zeroed"}
    assert (output / "scoring_pod_inputs" / "scoring_vectors" / manifest["vectors"]["main"]["file"]).exists()
    # resumable: a second call skips the finished shard
    messages = []
    rescoring.rescore_scoring_run(tmp_path / "source", output, "assistant_turn", LAYER, {"main": vector, "zeroed": zeroed}, "main",
                                  stamp="2026-02-02_00-01", say=messages.append)
    assert any("already rescored" in message for message in messages)
