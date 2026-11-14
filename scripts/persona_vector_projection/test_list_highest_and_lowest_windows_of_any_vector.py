"""Tests for list_highest_and_lowest_windows_of_any_vector_in_scoring_run on a small synthetic scoring run (no model, no API)."""
import glob
import json
import sys

import pytest

from persona_vector_projection import list_highest_and_lowest_windows_of_any_vector_in_scoring_run as lister
from persona_vector_projection.synthetic_scoring_run_for_analysis_tests import (
    LAYER_INDICES, VECTOR_NAMES, build_synthetic_scoring_run)

DOCUMENT_FORMAT = "assistant_turn"
VECTOR_NAME = VECTOR_NAMES[0]


def run_lister(tmp_path, monkeypatch, extra_arguments=()):
    scoring_run_folder = tmp_path / "scoring_run"
    build_synthetic_scoring_run(scoring_run_folder, document_format=DOCUMENT_FORMAT, save_token_ids=True)
    (scoring_run_folder / "scoring_pod_inputs" / "scoring_vector_manifest_synthetic.json").write_text(
        json.dumps({"chosen_layer": LAYER_INDICES[2]}), encoding="utf-8")
    monkeypatch.setattr(lister, "load_tokenizer_decoder", lambda *_: (lambda ids: " ".join(str(int(i)) for i in ids)))
    monkeypatch.setattr(sys, "argv", ["lister", "--scoring-run-folder", str(scoring_run_folder), "--document-format", DOCUMENT_FORMAT,
                                      "--vector-name", VECTOR_NAME, "--count", "5", *extra_arguments])
    lister.main()
    output_path = glob.glob(str(scoring_run_folder / "analysis" / "highest_and_lowest_windows_*" / "*.json"))
    assert len(output_path) == 1
    return json.load(open(output_path[0]))


def test_highest_windows_have_higher_means_than_lowest(tmp_path, monkeypatch):
    rankings = run_lister(tmp_path, monkeypatch, ["--layer", str(LAYER_INDICES[2])])["rankings"]
    assert len(rankings["highest"]) == len(rankings["lowest"]) == 5
    assert min(row["window_mean"] for row in rankings["highest"]) > max(row["window_mean"] for row in rankings["lowest"])
    means = [row["window_mean"] for row in rankings["highest"]]
    assert means == sorted(means, reverse=True)


def test_one_window_per_document_per_side_and_start_at_or_after_token_64(tmp_path, monkeypatch):
    rankings = run_lister(tmp_path, monkeypatch)["rankings"]
    for rows in rankings.values():
        document_ids = [row["document_id"] for row in rows]
        assert len(document_ids) == len(set(document_ids))
        assert all(row["start_token"] >= 64 for row in rows)
        assert all(row["window_text"] and row["context_before_text"] for row in rows)


def test_requested_layer_is_used(tmp_path, monkeypatch):
    first = run_lister(tmp_path / "a", monkeypatch, ["--layer", str(LAYER_INDICES[0])])
    last = run_lister(tmp_path / "b", monkeypatch, ["--layer", str(LAYER_INDICES[-1])])
    assert first["layer"] == LAYER_INDICES[0] and last["layer"] == LAYER_INDICES[-1]
    # the synthetic run adds 0.01 per layer index to every projection
    expected_difference = 0.01 * (len(LAYER_INDICES) - 1)
    first_by_document = {row["document_id"]: row["window_mean"] for row in first["rankings"]["highest"]}
    shared = [row for row in last["rankings"]["highest"] if row["document_id"] in first_by_document]
    assert shared
    for row in shared:
        assert row["window_mean"] - first_by_document[row["document_id"]] == pytest.approx(expected_difference, abs=0.01)
