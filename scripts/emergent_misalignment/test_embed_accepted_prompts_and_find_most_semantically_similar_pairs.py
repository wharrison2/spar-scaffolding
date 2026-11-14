"""Offline tests (the embedding client is mocked; no API call is ever made) for the semantic similarity script."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import embed_accepted_prompts_and_find_most_semantically_similar_pairs_with_openai_text_embedding_3_large as script
from shared.api_usage_and_cost_tracking import UsageAndCostTracker, compute_cost_in_dollars
from shared.openai_embeddings_client import cosine_similarity_matrix

MODEL = "text-embedding-3-large"


class MockEmbeddingsClient:
    """Deterministic fake: the vector is derived from the first character of each text."""

    def __init__(self):
        self.request_sizes = []
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, model, input, encoding_format):
        self.request_sizes.append(len(input))
        data = [SimpleNamespace(index=i, embedding=[float(ord(text[0])), 1.0, float(len(text) % 3)])
                for i, text in enumerate(input)]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=10 * len(input), total_tokens=10 * len(input)))


def make_records():
    rows = []
    for pass_label, prefix in (("first_pass", "a"), ("second_pass", "b")):
        for number, spec_id in enumerate(["S1", "S2", "S3"]):
            rows.append({"pass_label": pass_label, "corpus_prompt_id": f"B001-P-{number}", "spec_id": spec_id,
                         "record_key": f"{pass_label}::B001-P-{number}",
                         "user_message": f"{chr(ord('a') + number)} message {prefix} {number}"})
    return rows


def test_cost_for_embedding_model_is_in_the_tracker():
    assert compute_cost_in_dollars(MODEL, 1_000_000, 0, 0) == pytest.approx(0.13)


def test_record_keys_stay_distinct_when_identifiers_collide_across_passes():
    keys = [record["record_key"] for record in make_records()]
    assert len(set(keys)) == 6


def test_embedding_batches_resume_and_log_usage(tmp_path):
    records = make_records()
    tracker = UsageAndCostTracker(MODEL)
    client = MockEmbeddingsClient()
    script.embed_missing_records(records[:4], tmp_path, MODEL, 3, tracker, lambda: client)
    assert client.request_sizes == [3, 1] and tracker.overall_totals["input_tokens"] == 40
    second_client = MockEmbeddingsClient()
    resumed_tracker = UsageAndCostTracker(MODEL)
    embeddings = script.embed_missing_records(records, tmp_path, MODEL, 3, resumed_tracker, lambda: second_client)
    assert second_client.request_sizes == [2] and len(embeddings) == 6
    third_client = MockEmbeddingsClient()
    script.embed_missing_records(records, tmp_path, MODEL, 3, UsageAndCostTracker(MODEL), lambda: third_client)
    assert third_client.request_sizes == []


def test_cosine_similarity_matrix_is_correct():
    matrix = cosine_similarity_matrix([[1, 0], [2, 0], [0, 3]])
    assert matrix[0, 1] == pytest.approx(1.0) and matrix[0, 2] == pytest.approx(0.0)


def test_analysis_groups_pairs_and_same_spec_nearest_neighbour_rate():
    records = make_records()
    vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.1, 0], [0, 1, 0.1], [0.5, 0.5, 1]]
    similarity = cosine_similarity_matrix(vectors)
    top_pairs, summary = script.analyse_similarities(records, similarity, 2)
    assert [len(top_pairs[g]) for g in script.GROUP_NAMES] == [2, 2, 2]
    assert summary["groups"]["across_passes"]["pairwise_similarity"]["count"] == 9
    assert summary["groups"]["within_first_pass"]["pairwise_similarity"]["count"] == 3
    best_across = top_pairs["across_passes"][0]
    assert best_across["same_spec_id"] and best_across["pass_label_a"] == "first_pass"
    nearest = summary["groups"]["across_passes"]["second_pass_prompt_nearest_first_pass_neighbour_has_same_spec_id"]
    assert nearest["matching_prompts"] == 3 and nearest["second_pass_prompts"] == 3
    assert summary["groups"]["across_passes"]["same_spec_id_pairs"]["count"] == 3


def test_outputs_are_written(tmp_path):
    records = make_records()
    embeddings = {r["record_key"]: v for r, v in zip(records, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.1, 0], [0, 1, 0.1], [0.5, 0.5, 1]])}
    _, _, paths = script.write_analysis_outputs(tmp_path, records, embeddings, 5)
    assert all(path.exists() for path in paths)
    assert "message a" in paths[2].read_text()
