"""Automated tests for experiment-invalidating bugs in the persona vector extraction: answer-token masking, pairing by
question, consistent version averaging, topic projection, the filter, layer selection, domain weighting, resumability
(generation, activations, judge), memory scaling, judge reading, and conversion of the real drafts.

Run from the repository root:  python3 -m pytest scripts/persona_vector_projection -q
"""

import json
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import persona_vector_math as vector_math  # noqa: E402
from persona_vector_projection import compute_persona_vectors_from_activations_and_judgements as compute  # noqa: E402
from persona_vector_projection import generate_answers_and_save_activations_on_pod as pod_script  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import synthetic_extraction_run_for_tests as synthetic  # noqa: E402
from persona_vector_projection.answer_token_activation_recording import (  # noqa: E402
    AnswerTokenMeanActivationRecorder, build_padded_batch)

THRESHOLDS = {"positive_minimum": 65.0, "negative_maximum": 35.0, "coherence_minimum": 50.0,
              "animal_share_maximum_difference": 15.0}


# ---------------------------------------------------------------------------------------------------
# Answer-token masking
# ---------------------------------------------------------------------------------------------------

class StubConfig:
    def __init__(self, layers):
        self.num_hidden_layers = layers

    def get_text_config(self):
        return self


class StubDecoderLayer(nn.Module):
    """Position-wise (no attention): output = tanh(linear(input)), so the mean over answer tokens is exactly computable."""

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, hidden_states):
        return (torch.tanh(self.linear(hidden_states)),)


class StubModel(nn.Module):
    def __init__(self, vocabulary=20, hidden=8, layers=3):
        super().__init__()
        self.config = StubConfig(layers)
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.layers = nn.ModuleList([StubDecoderLayer(hidden) for _ in range(layers)])
        self.base_model = self

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


def test_answer_token_mean_excludes_prompt_tokens_and_padding():
    torch.manual_seed(0)
    model = StubModel()
    recorder = AnswerTokenMeanActivationRecorder(model)
    recorder.attach()
    sequences = [[1, 2, 3, 4, 5, 6], [7, 8, 9]]
    prompt_lengths = [2, 1]
    input_ids, attention_mask, answer_mask = build_padded_batch(sequences, prompt_lengths, padding_token_id=0)
    means = recorder.mean_activations_for_batch(input_ids, attention_mask, answer_mask)
    assert means.shape == (2, 4, 8)  # embedding output + 3 layers
    for row, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
        with torch.no_grad():
            hidden = model.embedding(torch.tensor(sequence))
            expected = [hidden[prompt_length:].mean(0)]
            for layer in model.layers:
                hidden = layer(hidden)[0]
                expected.append(hidden[prompt_length:].mean(0))
        assert torch.allclose(means[row], torch.stack(expected), atol=1e-6)
        # the prompt-token mean differs, so a mask that included prompt tokens would have been caught
        with torch.no_grad():
            wrong = model.embedding(torch.tensor(sequence)).mean(0)
        assert not torch.allclose(means[row, 0], wrong, atol=1e-4)


def test_changing_only_prompt_tokens_does_not_change_the_mean_in_a_position_wise_model():
    torch.manual_seed(1)
    model = StubModel()
    recorder = AnswerTokenMeanActivationRecorder(model)
    recorder.attach()
    answer = [5, 6, 7]
    results = []
    for prompt in ([1, 2], [9, 9]):
        ids, attention, answer_mask = build_padded_batch([prompt + answer], [len(prompt)], 0)
        results.append(recorder.mean_activations_for_batch(ids, attention, answer_mask))
    assert torch.allclose(results[0], results[1])


def test_padding_in_a_batch_does_not_change_a_sequence_mean_in_the_tiny_gemma_model():
    from tiny_model_and_tokenizer_helpers_for_log_probability_persona_check import (
        GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN, build_small_offline_tokenizer, build_tiny_random_gemma4_model)
    tokenizer = build_small_offline_tokenizer(GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN, ["hello pigs farm animals"] * 20)
    model = build_tiny_random_gemma4_model(len(tokenizer))
    recorder = AnswerTokenMeanActivationRecorder(model)
    recorder.attach()
    sequence_a, sequence_b = [2, 5, 6, 7, 8, 9], [2, 5, 6]
    alone = recorder.mean_activations_for_batch(*build_padded_batch([sequence_b], [2], 0))
    batched = recorder.mean_activations_for_batch(*build_padded_batch([sequence_a, sequence_b], [3, 2], 0))
    assert torch.allclose(alone[0], batched[1], atol=1e-4)
    # the language-model head is never called
    head = model.get_output_embeddings()
    calls = []
    handle = head.register_forward_hook(lambda *args: calls.append(1))
    recorder.mean_activations_for_batch(*build_padded_batch([sequence_a], [3], 0))
    handle.remove()
    assert calls == []


# ---------------------------------------------------------------------------------------------------
# Filter, averaging, topic projection, layers
# ---------------------------------------------------------------------------------------------------

def scores(facet_score, coherence=80.0, share=50.0):
    return {"facet_score": facet_score, "coherence": coherence, "share_about_animals": share}


def test_filter_boundaries_are_inclusive_and_missing_scores_fail():
    kwargs = THRESHOLDS
    assert vector_math.pair_passes_filter(scores(65), scores(35), **kwargs)
    assert not vector_math.pair_passes_filter(scores(64.9), scores(35), **kwargs)
    assert not vector_math.pair_passes_filter(scores(65), scores(35.1), **kwargs)
    assert not vector_math.pair_passes_filter(scores(80), scores(20, coherence=49.9), **kwargs)
    assert not vector_math.pair_passes_filter(scores(80, coherence=49.9), scores(20), **kwargs)
    assert vector_math.pair_passes_filter(scores(80, share=60), scores(20, share=45), **kwargs)
    assert not vector_math.pair_passes_filter(scores(80, share=60), scores(20, share=44.9), **kwargs)
    assert not vector_math.pair_passes_filter(scores(None), scores(20), **kwargs)
    assert not vector_math.pair_passes_filter(scores(80), {"facet_score": 20, "coherence": 80, "share_about_animals": None}, **kwargs)


def test_consistent_item_needs_both_versions_to_pass():
    good, bad = (scores(80), scores(20)), (scores(80), scores(60))
    both = {("a", "positive"): good[0], ("a", "negative"): good[1], ("b", "positive"): good[0], ("b", "negative"): good[1]}
    assert vector_math.consistent_item_passes_filter(both, **THRESHOLDS)
    one_bad = {**both, ("b", "negative"): bad[1]}
    assert not vector_math.consistent_item_passes_filter(one_bad, **THRESHOLDS)


def test_domains_are_weighted_equally():
    vectors = np.zeros((4, 1, 2), dtype=np.float32)
    vectors[:3, 0, 0] = 1.0      # three questions in a big domain
    vectors[3, 0, 0] = 5.0       # one question in a small domain
    result = vector_math.average_questions_with_domains_weighted_equally(vectors, ["big", "big", "big", "small"])
    assert result[0, 0] == pytest.approx(3.0)  # (1 + 5) / 2, not (3 + 5) / 4


def test_consistent_versions_are_averaged_with_equal_weight():
    a, b = np.full((1, 2, 3), 2.0), np.full((1, 2, 3), 6.0)
    assert np.allclose(vector_math.average_consistent_versions_with_equal_weight(a, b), 4.0)


def test_topic_projection_removes_the_topic_direction_and_nothing_else():
    generator = np.random.default_rng(0)
    topic = generator.normal(size=(4, 16)).astype(np.float32)
    orthogonal_part = generator.normal(size=(4, 16)).astype(np.float32)
    orthogonal_part -= np.sum(orthogonal_part * vector_math.unit_vectors_per_layer(topic), axis=-1, keepdims=True) * vector_math.unit_vectors_per_layer(topic)
    facet = orthogonal_part + 0.7 * topic
    cleaned = vector_math.project_out_per_layer(facet, topic)
    assert np.allclose(np.sum(cleaned * vector_math.unit_vectors_per_layer(topic), axis=-1), 0, atol=1e-5)
    assert np.allclose(cleaned, orthogonal_part, atol=1e-4)
    assert np.allclose(vector_math.project_out_per_layer(cleaned, topic), cleaned, atol=1e-5)
    share = vector_math.share_along_topic_per_layer(facet, topic)
    expected = (0.7 ** 2 * np.sum(topic ** 2, -1)) / np.sum(facet ** 2, -1)
    assert np.allclose(share, expected, atol=1e-5)


def test_layer_choice_shared_layer_and_fallback_and_layer_zero_excluded():
    accuracy = {"f1": np.array([1.0, 0.70, 0.95, 0.90, 0.60]), "f2": np.array([1.0, 0.60, 0.85, 0.92, 0.95])}
    choice = vector_math.choose_layers(accuracy)
    assert choice["strategy"] == "shared_layer" and choice["shared_layer"] == 3   # layer 0 (accuracy 1.0) is excluded
    # layer 2: f2 0.85 / 0.95 = 0.895 < 0.9, so only layer 3 qualifies
    assert choice["qualifying_layers"] == [3]
    accuracy = {"f1": np.array([0.5, 0.95, 0.60, 0.60]), "f2": np.array([0.5, 0.60, 0.60, 0.95])}
    choice = vector_math.choose_layers(accuracy)
    assert choice["strategy"] == "per_facet_best_layer" and choice["layer_by_facet"] == {"f1": 1, "f2": 3}


def test_classification_accuracy_uses_fit_threshold_and_perfect_separation_scores_one():
    positive, negative = np.array([[2.0, 0.1], [3.0, 0.2]]), np.array([[-2.0, 0.0], [-3.0, 0.3]])
    thresholds = vector_math.midpoint_thresholds(positive, negative)
    accuracy = vector_math.classification_accuracy_per_layer(positive, negative, thresholds)
    assert accuracy[0] == 1.0 and accuracy[1] < 1.0


# ---------------------------------------------------------------------------------------------------
# Full vector computation on a synthetic run with known geometry
# ---------------------------------------------------------------------------------------------------

def cosine(a, b):
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture()
def synthetic_run(tmp_path):
    folder, directions, answers, _ = synthetic.build_synthetic_run(tmp_path / "run")
    return folder, directions


def run_compute(folder, **overrides):
    thresholds = {**THRESHOLDS, **overrides}
    return compute.compute_all(folder, thresholds, None or json.loads(next(Path(folder).glob("run_configuration_*.json")).read_text())["facets"],
                               bootstrap_resamples=20, minimum_layer_index=1, vector_set_name="test", seed=0, say=lambda *_: None)


def test_vectors_recover_the_signal_directions_and_pair_by_question(synthetic_run):
    folder, directions = synthetic_run
    metadata, metadata_path = run_compute(folder)
    output_folder = metadata_path.parent
    stamp = metadata["timestamp"]
    for facet in (defs.FACET_CARES_ABOUT_WELFARE, defs.FACET_CALIBRATED_CONSIDERATION, defs.FACET_NON_INTRUSIVE):
        vector = np.load(output_folder / metadata["vector_files"][f"{facet}_before_topic_removal"])
        assert vector.shape == (5, 64)
        # per-question offsets (norm about 2 * sqrt(64) per layer) cancel exactly in the paired difference
        assert np.allclose(vector, 2 * synthetic.SIGNAL_SCALE * directions[facet], atol=0.3)
    consistent = np.load(output_folder / metadata["vector_files"]["consistent_before_topic_removal"])
    # version a: 2 * S, version b: 3 * 2 * S, averaged with equal weight: 4 * S (not weighted by anything else)
    assert np.allclose(consistent, 4 * synthetic.SIGNAL_SCALE * directions[defs.FACET_CONSISTENT], atol=0.4)
    topic = np.load(output_folder / next(n for n in metadata["topic_vector_files"] if "without" in n))
    assert np.allclose(topic, 2 * synthetic.SIGNAL_SCALE * directions[defs.TOPIC_CONTROL], atol=0.3)
    assert metadata["per_facet"][defs.FACET_CONSISTENT]["questions_kept"] == 12


def test_half_domain_variant_vectors_are_saved_topic_removed_and_found_by_the_scoring_pod_loader(synthetic_run):
    from persona_vector_projection.scoring_pod_vector_loading_adapter import find_variant_vector_file_names
    folder, directions = synthetic_run
    metadata, metadata_path = run_compute(folder)
    output_folder = metadata_path.parent
    topic = np.load(output_folder / next(n for n in metadata["topic_vector_files"] if "without" in n))
    found = find_variant_vector_file_names(output_folder)
    for facet in (defs.FACET_CARES_ABOUT_WELFARE, defs.FACET_CALIBRATED_CONSIDERATION, defs.FACET_NON_INTRUSIVE, defs.FACET_CONSISTENT):
        for half in ("A", "B"):
            name = f"{facet}__vector_from_half_{half}"
            assert metadata["vector_files"][name] == found[name]
            vector = np.load(output_folder / found[name])
            assert vector.shape == (5, 64) and np.abs(np.sum(vector * topic, axis=-1)).max() < 1e-3  # topic removed
        half_a, half_b = (np.load(output_folder / found[f"{facet}__vector_from_half_{h}"]) for h in "AB")
        assert cosine(half_a, half_b) > 0.9   # both halves carry the same synthetic signal


def test_every_question_is_paired_on_its_own_and_mismatched_pairing_would_be_far_off(synthetic_run):
    """Each question's own difference recovers the known direction (offsets cancel within the question). Differencing a
    positive answer against a negative answer of a different question would not cancel the offsets."""
    folder, directions = synthetic_run
    answers = compute.load_answers(folder)
    store = compute.ActivationStore(folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME)
    true_vector = 2 * synthetic.SIGNAL_SCALE * directions[defs.FACET_NON_INTRUSIVE]
    differences = compute.compute_question_differences(defs.FACET_NON_INTRUSIVE, answers, compute.load_judge_scores(folder), store, THRESHOLDS)
    assert len(differences["question_keys"]) == 12
    for question_vector in differences["vectors"]:
        assert np.allclose(question_vector, true_vector, atol=0.3)
    first, second = differences["question_keys"][0], differences["question_keys"][1]
    positive = next(a for a in answers if a["question_id"] == first and a["pole"] == "positive" and a["sample_index"] == 0)
    negative_other_question = next(a for a in answers if a["question_id"] == second and a["pole"] == "negative" and a["sample_index"] == 0)
    rows = dict(store.stream_rows([positive["answer_id"], negative_other_question["answer_id"]]))
    mismatched = rows[positive["answer_id"]].astype(np.float32) - rows[negative_other_question["answer_id"]].astype(np.float32)
    assert np.linalg.norm(mismatched - true_vector) > 10 * 0.3


def test_topic_vector_projection_is_orthogonal_in_the_saved_files(tmp_path):
    folder, directions, _, _ = synthetic.build_synthetic_run(tmp_path / "run", topic_loading_of_facets=0.0)
    metadata, path = run_compute(folder)
    cleaned = np.load(path.parent / metadata["vector_files"]["cares_about_welfare_topic_removed"])
    topic = np.load(path.parent / next(n for n in metadata["topic_vector_files"] if "without" in n))
    assert np.allclose(np.sum(cleaned * vector_math.unit_vectors_per_layer(topic), axis=-1), 0, atol=1e-4)


def test_welfare_remark_answers_are_excluded_from_the_topic_vector_but_reported(synthetic_run):
    folder, _ = synthetic_run
    metadata, _ = run_compute(folder)
    without = metadata["per_facet"][defs.TOPIC_CONTROL]["filter_counts_without_welfare_remark_answers"]
    with_all = metadata["per_facet"][defs.TOPIC_CONTROL]["filter_counts_with_all_answers"]
    assert without["items_with_welfare_remark_excluded"] > 0
    assert with_all["items_kept"] == without["items_kept"] + without["items_with_welfare_remark_excluded"]


def test_tag_vectors_and_filter_counts_and_layer_choice_are_saved(synthetic_run):
    folder, directions = synthetic_run
    metadata, path = run_compute(folder)
    assert "non_intrusive_negative_tag_lecturing_topic_removed" in metadata["vector_files"]
    assert "non_intrusive_negative_tag_refusing_or_watering_down_topic_removed" in metadata["vector_files"]
    counts = metadata["per_facet"][defs.FACET_NON_INTRUSIVE]["filter_counts"]
    assert counts["items_kept"] == counts["items_total"] and counts["items_failing_filter"] == 0
    assert metadata["layer_choice"]["strategy"] in ("shared_layer", "per_facet_best_layer")
    assert set(metadata["layer_choice"]["layer_by_facet"]) == set(defs.PERSONA_FACETS)
    cosines = metadata["diagnostics"]["cosine_between_facets_topic_removed_per_layer"]
    assert len(cosines) == 6 and all(abs(c) < 0.2 for values in cosines.values() for c in values)


def test_failing_answers_are_dropped_by_the_filter(tmp_path):
    folder, directions, answers, _ = synthetic.build_synthetic_run(tmp_path / "run")
    # make every positive answer of one question score 50 so the whole question drops out
    judge_path = next((folder / defs.JUDGEMENTS_FOLDER_NAME).glob("judge_results_*.jsonl"))
    records = [json.loads(line) for line in judge_path.read_text().splitlines()]
    for record in records:
        if record["kind"] == "facet_score" and "non_intrusive__positive" in record["answer_id"] and "non_intrusive__A__01" in record["answer_id"]:
            record["score"] = 50.0
    judge_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    metadata, _ = run_compute(folder)
    assert metadata["per_facet"][defs.FACET_NON_INTRUSIVE]["questions_kept"] == 11
    assert metadata["per_facet"][defs.FACET_NON_INTRUSIVE]["filter_counts"]["items_failing_filter"] == 6


def test_peak_memory_scales_with_one_facet_not_the_whole_run(tmp_path):
    kwargs = dict(layers=8, hidden=2048, answers_per_shard=30)
    one_folder, _, _, _ = synthetic.build_synthetic_run(tmp_path / "one", facets=[defs.FACET_NON_INTRUSIVE, defs.TOPIC_CONTROL], **kwargs)
    all_folder, _, _, _ = synthetic.build_synthetic_run(tmp_path / "all", **kwargs)

    def peak_bytes(folder):
        tracemalloc.start()
        run_compute(folder)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    def stored_bytes(folder):
        return sum(p.stat().st_size for p in (folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME).glob("*.npz"))

    peak_one, peak_all = peak_bytes(one_folder), peak_bytes(all_folder)
    total_all = stored_bytes(all_folder)
    assert total_all > 3 * stored_bytes(one_folder) * 0.9
    assert peak_all < 0.5 * total_all                       # never the whole store at once
    # the consistent facet alone is twice the size of the others, so allow generous headroom: 5 facets' worth of data
    # costs far less than 5 times the peak of 1 facet + topic
    assert peak_all < 3.0 * peak_one


# ---------------------------------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------------------------------

class CountingBackend:
    def __init__(self, fail_on_call=None):
        self.calls, self.fail_on_call = 0, fail_on_call

    def generate(self, prompt_token_id_lists, samples_per_prompt):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("simulated pod failure")
        return [[{"text": f"answer {i}", "number_of_tokens": 3, "finish_reason": "stop"} for i in range(samples_per_prompt)]
                for _ in prompt_token_id_lists]


@pytest.fixture()
def tiny_tokenizer():
    from tiny_model_and_tokenizer_helpers_for_log_probability_persona_check import (
        GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN, build_small_offline_tokenizer)
    return build_small_offline_tokenizer(GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN, ["hello pigs farm animals answer text"] * 20)


def make_jobs(count):
    return [{"job_id": f"job_{i:03d}", "facet": "non_intrusive", "pole": "positive", "tag": None, "system_prompt_id": "sp",
             "system_prompt": "You are an assistant.", "question_id": f"q{i}", "pair_id": None, "version": None, "half": "A",
             "domain": "d", "user_prompt": "Hello pigs?", "samples_per_prompt": 2} for i in range(count)]


def test_generation_stage_resumes_from_finished_shards(tmp_path, tiny_tokenizer):
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text("{}")
    jobs = make_jobs(7)
    failing = CountingBackend(fail_on_call=3)
    with pytest.raises(RuntimeError):
        pod_script.run_generation_stage(run_folder, jobs, tiny_tokenizer, failing, jobs_per_shard=2)
    shards = sorted((run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME).glob("answers_shard_*.jsonl"))
    assert len(shards) == 2 and not list(shards[0].parent.glob("*partial*"))
    first_shard_text = shards[0].read_text()
    resumed = CountingBackend()
    pod_script.run_generation_stage(run_folder, jobs, tiny_tokenizer, resumed, jobs_per_shard=2)
    assert resumed.calls == 2                       # shards 2 and 3 only (4 shards of 2, 2, 2, 1 jobs)
    assert shards[0].read_text() == first_shard_text
    records = [json.loads(line) for path in sorted(shards[0].parent.glob("answers_shard_*.jsonl")) for line in path.read_text().splitlines()]
    assert len(records) == 7 * 2 and len({r["answer_id"] for r in records}) == 14


def test_activation_stage_resumes_and_masks_the_prompt(tmp_path, tiny_tokenizer):
    from tiny_model_and_tokenizer_helpers_for_log_probability_persona_check import build_tiny_random_gemma4_model
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text("{}")
    jobs = make_jobs(5)
    pod_script.run_generation_stage(run_folder, jobs, tiny_tokenizer, CountingBackend(), jobs_per_shard=2)
    model = build_tiny_random_gemma4_model(len(tiny_tokenizer))
    pod_script.run_activation_stage(run_folder, jobs, tiny_tokenizer, model, "cpu", 4, "float16")
    activations_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    shards = sorted(activations_folder.glob("activations_shard_*.npz"))
    assert len(shards) == 3
    modification_times = {path: path.stat().st_mtime_ns for path in shards}
    shards[1].unlink()
    pod_script.run_activation_stage(run_folder, jobs, tiny_tokenizer, model, "cpu", 4, "float16")
    assert shards[1].exists()
    assert shards[0].stat().st_mtime_ns == modification_times[shards[0]] and shards[2].stat().st_mtime_ns == modification_times[shards[2]]
    with np.load(shards[0]) as data:
        assert data["activations"].shape == (4, 5, 32) and (data["answer_token_counts"] > 0).all()
        # reference: recompute one answer without batching or hooks' help, from output_hidden_states
        answer_id = str(data["answer_ids"][0])
        record = next(json.loads(line) for line in (run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME).glob("answers_shard_00000*").__next__().read_text().splitlines()
                      if json.loads(line)["answer_id"] == answer_id)
        from persona_vector_projection import chat_prompt_rendering_for_extraction as rendering
        prompt_ids = rendering.prompt_token_ids(tiny_tokenizer, jobs[0]["system_prompt"], jobs[0]["user_prompt"])
        answer_ids = rendering.answer_token_ids(tiny_tokenizer, record["answer_text"])
        with torch.no_grad():
            output = model.base_model(input_ids=torch.tensor([prompt_ids + answer_ids]), output_hidden_states=True, use_cache=False)
        expected = torch.stack([hidden[0, len(prompt_ids):].mean(0) for hidden in output.hidden_states[:4]])
        assert np.allclose(data["activations"][0][:4].astype(np.float32), expected.numpy(), atol=2e-2)


# ---------------------------------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, text):
        self.output_text = text
        self.status = "completed"
        self.usage = SimpleNamespace(input_tokens=100, output_tokens=2, input_tokens_details=SimpleNamespace(cached_tokens=0),
                                     output_tokens_details=SimpleNamespace(reasoning_tokens=0))


class FakeJudgeClient:
    def __init__(self):
        self.calls = []

    def reply(self, request_arguments):
        self.calls.append(request_arguments)
        text = request_arguments["input"]
        return FakeResponse("YES" if "Reply with YES or NO" in text else "77")


def make_requests(count):
    return [judge.JudgeRequest(f"r{i}", judge.KIND_COHERENCE if i % 2 else judge.KIND_VOLUNTEERED_WELFARE_REMARK,
                               "Reply with YES or NO" if i % 2 == 0 else "number please", f"a{i}") for i in range(count)]


def test_judge_is_resumable_and_logs_usage(tmp_path):
    client = FakeJudgeClient()
    call = lambda _client, arguments: client.reply(arguments)  # noqa: E731
    requests = make_requests(6)
    first_path = tmp_path / "judge_results_2026-01-01_00-00.jsonl"
    results, failures = judge.run_judge_requests(requests[:4], None, results_path=first_path, call_responses=call,
                                                 usage_log_path=tmp_path / "usage.json", worker_count=2)
    assert len(results) == 4 and not failures and len(client.calls) == 4
    results, failures = judge.run_judge_requests(requests, None, results_path=tmp_path / "judge_results_2026-01-01_00-01.jsonl",
                                                 call_responses=call, usage_log_path=tmp_path / "usage.json", worker_count=2)
    assert len(client.calls) == 6 and len(results) == 6          # only the two missing requests were called
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert usage["overall"]["calls"] == 2 and usage["overall"]["input_tokens"] == 200 and usage["overall"]["cost_dollars"] > 0
    assert all(call_arguments["temperature"] == 1 and call_arguments["reasoning"] == {"effort": "none"} for call_arguments in client.calls)
    assert all("top_p" not in call_arguments for call_arguments in client.calls)
    by_id = {r["request_id"]: r for r in results}
    assert by_id["r1"]["score"] == 77.0 and by_id["r0"]["yes_or_no"] == "YES"


def test_score_from_first_token_logprobs_follows_chen_et_al():
    top = [("80", np.log(0.5)), ("20", np.log(0.25)), ("The", np.log(0.2)), ("101", np.log(0.05))]
    score, mass = judge.score_from_first_token_top_logprobs(top)
    assert score == pytest.approx((80 * 0.5 + 20 * 0.25) / 0.75) and mass == pytest.approx(0.75)
    low_mass, mass = judge.score_from_first_token_top_logprobs([("The", np.log(0.9)), ("50", np.log(0.1))])
    assert low_mass is None and mass == pytest.approx(0.1)   # under 0.25 on numbers: discarded


def test_bare_number_and_yes_no_parsing():
    assert judge.parse_bare_number(" 85") == 85.0 and judge.parse_bare_number("100.") == 100.0
    assert judge.parse_bare_number("101") is None and judge.parse_bare_number("about 80") is None
    assert judge.parse_yes_or_no("Yes.") == "YES" and judge.parse_yes_or_no("maybe") is None


def test_judge_prompts_fill_every_slot_and_keep_braces_in_answers():
    questions = {"q": {"question_text": "Q?", "labels": {"material_welfare_cost": True, "material_welfare_cost_note": "castration pain",
                                                       "evidence_bands": "Pigs: Established", "animal": "pigs", "reference_animal": "dogs"}}}
    facet_definition = {"display_name": "Non-intrusive", "positive_pole_description": "pos", "negative_pole_description": "neg"}
    for facet in defs.PERSONA_FACETS:
        prompt = judge.render_facet_score_prompt(facet, facet_definition, questions["q"], "An {answer} with {braces}")
        assert "{answer} with {braces}" in prompt and "{question}" not in prompt and "{facet" not in prompt
    ni = judge.render_facet_score_prompt(defs.FACET_NON_INTRUSIVE, facet_definition, questions["q"], "a")
    assert "applies: castration pain" in ni
    questions["q"]["labels"]["material_welfare_cost"] = False
    assert "does not apply" in judge.render_facet_score_prompt(defs.FACET_NON_INTRUSIVE, facet_definition, questions["q"], "a")


# ---------------------------------------------------------------------------------------------------
# Draft conversion on the real drafts
# ---------------------------------------------------------------------------------------------------

def test_real_drafts_convert_with_expected_counts(tmp_path):
    from persona_vector_projection import convert_drafts_into_machine_readable_inputs as converter
    prompt_folder = tmp_path / "prompts"
    report = converter.convert_everything(defs.QUESTION_DRAFT_FOLDER, defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER, prompt_folder,
                                          tmp_path / "out", "2026-01-01_00-00")
    assert report["system_prompt_count"] == 41 and report["question_records_per_facet"]["consistent"] == 104
    assert report["question_record_count"] == 364
    records = [json.loads(line) for line in (tmp_path / "out" / "questions_2026-01-01_00-00.jsonl").read_text().splitlines()]
    consistent = [r for r in records if r["facet"] == "consistent"]
    assert "\\|" not in "".join(r["question_text"] for r in consistent) and "{" not in "".join(r["question_text"] for r in consistent)
    for pair_id in {r["pair_id"] for r in consistent}:
        texts = [r["question_text"] for r in consistent if r["pair_id"] == pair_id]
        assert len(texts) == 2 and texts[0] != texts[1]
    prompts = [json.loads(line) for line in (tmp_path / "out" / "system_prompts_2026-01-01_00-00.jsonl").read_text().splitlines()]
    assert all(p["system_prompt"].endswith("under 300 words.") for p in prompts)
    tags = {(p["facet"], p["tag"]) for p in prompts if p["pole"] == "negative" and p["tag"]}
    assert ("non_intrusive", "lecturing") in tags and ("non_intrusive", "refusing_or_watering_down") in tags


def test_conversion_fails_loudly_on_a_broken_draft(tmp_path):
    from persona_vector_projection import convert_drafts_into_machine_readable_inputs as converter
    questions = tmp_path / "questions"
    questions.mkdir()
    for path in defs.QUESTION_DRAFT_FOLDER.glob("*.md"):
        text = path.read_text()
        if path.name.startswith("non-intrusive"):
            text = text.replace("| A1 |", "| Z1 |", 1)
        (questions / path.name).write_text(text)
    with pytest.raises(SystemExit, match="CONVERSION FAILED"):
        converter.convert_everything(questions, defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER, tmp_path / "prompts", tmp_path / "out")


def test_pilot_cap_keeps_domain_balance_and_pairs_together():
    from persona_vector_projection import build_extraction_run_folder as builder
    records = [{"facet": "consistent", "half": "A", "question_id": f"q{i}__{v}", "pair_id": f"q{i}", "domain": f"d{i % 3}"}
               for i in range(9) for v in ("a", "b")]
    chosen = builder.select_items_round_robin_over_domains(records, 3)
    assert {r["pair_id"] for r in chosen} == {"q0", "q1", "q2"} and len(chosen) == 6
