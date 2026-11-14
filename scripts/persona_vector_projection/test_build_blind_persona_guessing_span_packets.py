"""Tests for bugs that would invalidate the blind persona-guessing packets: the set the key calls high must really have the higher
window means, packet files must not leak vector names, document ids or scores, one window per document per side, non-corpus
documents excluded, and the same seed giving the same assignments. Local and free: a tiny synthetic scoring run with saved
activations and a stub decoder; no model, API call, .env read or pod.

    python3 -m pytest scripts/persona_vector_projection/test_build_blind_persona_guessing_span_packets.py -q
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import build_blind_persona_guessing_span_packets_from_saved_activations as builder  # noqa: E402

HIDDEN, LAYER, LAYERS = 6, 3, 5
CORPUS_IDS = [f"corpus_document_{n}" for n in range(6)]
REWRITE_IDS = ["rewrite_document_secret_a", "rewrite_document_secret_b"]


def write_run(run_folder, seed=0):
    generator = np.random.default_rng(seed)
    all_ids = CORPUS_IDS + REWRITE_IDS
    token_counts = [200 + 10 * n for n in range(len(all_ids))]
    activations = np.concatenate([generator.standard_normal((count, HIDDEN)) for count in token_counts]).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(token_counts)])
    token_ids = np.concatenate([np.arange(count) % 50 for count in token_counts]).astype(np.int32)
    shard_folder = run_folder / "scoring_pod_outputs" / "per_document_projections" / "assistant_turn"
    shard_folder.mkdir(parents=True)
    np.savez(shard_folder / "projections_assistant_turn_shard_00000_test.npz", document_ids=np.array(all_ids),
             per_token_offsets=offsets, saved_activations_float16=activations, saved_activations_layer_index=np.array(LAYER),
             per_token_token_ids_int32=token_ids)
    (run_folder / "scoring_pod_inputs").mkdir()
    with open(run_folder / "scoring_pod_inputs" / "documents.jsonl", "w") as file:
        for document_id in all_ids:
            file.write(json.dumps({"document_id": document_id, "kind": "corpus" if document_id in CORPUS_IDS else "rewrite"}) + "\n")
    (run_folder / "run_configuration_test.json").write_text(json.dumps({"model_name": "stub_model", "model_revision": "stub_revision"}))
    vectors = {}
    for name in ("persona_alpha", "persona_beta"):
        vector = generator.standard_normal((LAYERS + 1, HIDDEN)).astype(np.float32)
        np.save(run_folder / f"{name}.npy", vector)
        vectors[name] = vector
    return activations, offsets, vectors


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(builder.analysis_module, "load_tokenizer_decoder",
                        lambda name, revision: (lambda ids: "".join(f"<t{int(i)}>" for i in ids)))
    run_folder = tmp_path / "scoring_run"
    activations, offsets, vectors = write_run(run_folder)
    return SimpleNamespace(folder=run_folder, activations=activations, offsets=offsets, vectors=vectors, tmp=tmp_path)


def build(run, seed=0, parent="out", **overrides):
    values = dict(scoring_run_folder=str(run.folder), document_format="assistant_turn", document_kind="corpus",
                  vector=[f"{name}={run.folder / (name + '.npy')}" for name in run.vectors], random_control_vector_count=1,
                  windows_per_side=3, context_tokens_before=96, context_tokens_after=32, seed=seed,
                  output_parent_folder=str(run.tmp / parent))
    values.update(overrides)
    return builder.build_packets(SimpleNamespace(**values), timestamp="2026-01-01_00-00")


def load_key(folder):
    return json.load(open(next((folder / "answer_key").glob("*.json"))))


def window_mean_of(run, vector_row, document_id, start):
    all_ids = CORPUS_IDS + REWRITE_IDS
    index = all_ids.index(document_id)
    rows = run.activations[run.offsets[index]:run.offsets[index + 1]].astype(np.float32)
    projections = rows @ (vector_row / np.linalg.norm(vector_row))
    return float(projections[start:start + 32].mean())


def test_the_set_called_high_has_the_higher_window_means_and_means_are_correct(run):
    key = load_key(build(run))
    checked = 0
    for entry in key.values():
        if entry["vector_name"] not in run.vectors:
            continue
        row = run.vectors[entry["vector_name"]][LAYER]
        windows = entry["windows_in_displayed_order"]
        low_set = "Set B" if entry["high_projection_set"] == "Set A" else "Set A"
        for set_label in windows:
            for window in windows[set_label]:
                assert window["window_mean"] == pytest.approx(window_mean_of(run, row, window["document_id"], window["start_token"]), abs=1e-3)
        assert min(w["window_mean"] for w in windows[entry["high_projection_set"]]) > max(w["window_mean"] for w in windows[low_set])
        checked += 1
    assert checked == 2


def test_packet_files_contain_no_vector_name_document_id_or_score(run):
    folder = build(run)
    key = load_key(folder)
    files = sorted((folder / "packets").glob("*.txt"))
    assert len(files) == 3
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "⟦" in text and "⟧" in text
        assert "Set A" in text and "Set B" in text
        assert not re.search(r"high|low|random|persona|corpus_document|rewrite_document|percentile|score", text, re.IGNORECASE)
    # nothing outside the packet folder was needed to read a packet
    assert not list((folder / "packets").glob("*.json"))


def test_one_window_per_document_per_side(run):
    key = load_key(build(run))
    for entry in key.values():
        for windows in entry["windows_in_displayed_order"].values():
            ids = [window["document_id"] for window in windows]
            assert len(ids) == len(set(ids)) == 3
            assert all(window["start_token"] >= 64 for window in windows)


def test_non_corpus_documents_are_excluded(run):
    folder = build(run)
    key = load_key(folder)
    for entry in key.values():
        for windows in entry["windows_in_displayed_order"].values():
            assert all(window["document_id"] in CORPUS_IDS for window in windows)
    assert json.load(open(next(folder.glob("run_configuration_*.json"))))["document_count"] == len(CORPUS_IDS)


def test_the_same_seed_gives_the_same_assignments_and_another_seed_can_differ(run):
    first, second = load_key(build(run, parent="first")), load_key(build(run, parent="second"))
    assert first == second
    differing = [load_key(build(run, seed=seed, parent=f"seed_{seed}")) for seed in range(1, 6)]
    assert any(key != first for key in differing)


def test_an_existing_output_folder_is_never_overwritten(run):
    build(run)
    with pytest.raises(FileExistsError):
        build(run)
