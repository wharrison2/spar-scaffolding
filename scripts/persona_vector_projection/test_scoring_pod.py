"""Tests for bugs that would invalidate the scoring pod experiment. Local, free, tiny random model; the real Gemma 4
tokenizer is read from the local Hugging Face cache (tests that need it are skipped when it is not cached).

    python3 -m pytest scripts/persona_vector_projection/test_scoring_pod.py -q
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering  # noqa: E402
from persona_vector_projection import scoring_pod_test_support as support  # noqa: E402
from persona_vector_projection import build_scoring_pod_run_folder as builder  # noqa: E402
from persona_vector_projection import run_scoring_pod_stages as stages  # noqa: E402
from persona_vector_projection import estimate_scoring_pod_gpu_hours_and_cost as estimator  # noqa: E402
from persona_vector_projection.scoring_pod_projection_recording import LayerProjectionRecorder, project_documents  # noqa: E402
from persona_vector_projection.scoring_pod_steering_hook import (combined_steering_vector, steering_applied,  # noqa: E402
                                                                 steering_applied_at_layers, steering_vectors_by_layer)
from persona_vector_projection.scoring_pod_vector_loading_adapter import load_scoring_vectors, recorded_layer_indices  # noqa: E402
from persona_vector_projection.persona_vector_math import cosine_per_layer  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
import log_probability_persona_check_prompt_rendering as original_rendering  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return support.load_real_gemma_tokenizer()
    except Exception as error:  # not cached
        pytest.skip(f"Gemma 4 tokenizer not in the local cache: {error}")


@pytest.fixture(scope="module")
def tiny_model_directory(tmp_path_factory, tokenizer):
    directory = tmp_path_factory.mktemp("tiny_model")
    support.save_tiny_model_with_real_tokenizer(directory)
    return directory


@pytest.fixture(scope="module")
def tiny_model(tiny_model_directory):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(tiny_model_directory, dtype=torch.float32).eval()


# ---------------------------------------------------------------------------------------------------
# Baseline prompts
# ---------------------------------------------------------------------------------------------------

def test_baseline_prompt_has_kept_variables_and_not_dropped_ones():
    record = support.make_corpus_record(7)
    prompt = rendering.render_baseline_generation_prompt(record["variables"], record["doc_id"])
    for name in defs.KEPT_TOPIC_VARIABLE_NAMES:
        assert record["variables"][name] in prompt, name
    for name in defs.DROPPED_STANCE_VARIABLE_NAMES:
        assert record["variables"][name] not in prompt, name
    for marker in ("TONEMARKER", "RESOLUTIONMARKER", "REASONINGMARKER"):
        assert marker not in prompt
    for removed_phrase in ("The tone is", "The AI's reasoning features"):
        assert removed_phrase not in prompt


def test_baseline_template_is_the_original_template_minus_exactly_the_three_stance_sentences():
    original = original_rendering.load_prompt_text(original_rendering.GENERATION_PROMPT_TEMPLATE_FILE_NAME)
    expected = original.replace(" The tone is {tone}.", "").replace(" {resolution}", "").replace(
        " The AI's reasoning features {reasoning_featured}.", "")
    assert rendering.load_prompt(defs.BASELINE_PROMPT_TEMPLATE_FILE_NAME) == expected


def test_baseline_prompts_for_the_whole_corpus_are_clean():
    corpus = read_jsonl(defs.DEFAULT_CORPUS_PATH)
    for record in corpus:
        prompt = rendering.render_baseline_generation_prompt(record["variables"], record["doc_id"])
        assert "{" not in prompt and "}" not in prompt and "  " not in prompt
        for name in defs.DROPPED_STANCE_VARIABLE_NAMES:
            assert record["variables"][name].strip() not in prompt


def test_baseline_prompt_refuses_missing_kept_variable():
    record = support.make_corpus_record(1)
    del record["variables"]["tradeoff"]
    with pytest.raises(ValueError):
        rendering.render_baseline_generation_prompt(record["variables"], record["doc_id"])


# ---------------------------------------------------------------------------------------------------
# Rewrites share their source's baselines
# ---------------------------------------------------------------------------------------------------

def test_rewrites_use_their_sources_baselines_and_jobs_come_from_source_variables():
    corpus = [support.make_corpus_record(n) for n in range(1, 4)]
    corpus_by_id = {r["doc_id"]: r for r in corpus}
    corpus_documents = [{"document_id": r["doc_id"], "source_document_id": r["doc_id"], "kind": "corpus", "language": r["language"],
                         "variables": r["variables"], "content": r["content"], "rewrite_direction": None} for r in corpus[:2]]
    rewrites = [support.make_rewrite_record(corpus[0], "preachier"), support.make_rewrite_record(corpus[0], "plainer")]
    rewrites[0]["variables"]["domain"] = "SHOULD NOT BE USED"  # a rewrite's own variables are never used for baselines
    documents = corpus_documents + [{**r, "kind": "rewrite"} for r in rewrites]
    mapping = builder.baseline_source_of_each_document(documents)
    assert mapping["matrix_000001__preachier"] == mapping["matrix_000001__plainer"] == "matrix_000001"
    assert mapping["matrix_000002"] == "matrix_000002"
    jobs = builder.build_baseline_generation_jobs(documents, corpus_by_id, 4, "You are an AI assistant.")
    assert [j["source_document_id"] for j in jobs] == ["matrix_000001", "matrix_000002"]  # no job per rewrite, none for unselected
    assert all("SHOULD NOT BE USED" not in j["user_prompt"] for j in jobs)
    assert all(j["samples_per_prompt"] == 4 and j["system_prompt"] == "You are an AI assistant." for j in jobs)


def test_scoring_stage_input_keeps_baselines_separate_from_documents_and_drops_empty_baselines():
    documents = [{"document_id": "b", "kind": "corpus"}, {"document_id": "a__preachier", "kind": "rewrite"}]
    baselines = [{"document_id": "a__baseline_00", "content": "text", "kind": "baseline"},
                 {"document_id": "a__baseline_01", "content": "  ", "kind": "baseline"}]
    items = stages.build_documents_to_score(documents, baselines)
    assert [i["document_id"] for i in items] == ["a__baseline_00", "a__preachier", "b"]


# ---------------------------------------------------------------------------------------------------
# Token positions
# ---------------------------------------------------------------------------------------------------

DOCUMENT = "Grey seals in New England\n\nThe population has grown since protection began, and fishers report more gear damage."


@pytest.mark.parametrize("document_format", defs.DOCUMENT_FORMATS)
def test_document_tokens_are_located_exactly_and_context_comes_before(tokenizer, document_format):
    system_prompt = rendering.system_prompt_for_format(document_format, "You are an AI assistant.")
    tokenized = rendering.tokenize_document_for_format(tokenizer, "  " + DOCUMENT + "\n", document_format, system_prompt=system_prompt)
    document_ids = tokenized.token_ids[tokenized.document_start_index:tokenized.document_end_index]
    assert tokenizer.decode(document_ids).strip() == DOCUMENT
    assert document_ids == tokenizer(DOCUMENT, add_special_tokens=False)["input_ids"]  # document tokens are the plain document
    assert tokenized.document_end_index == len(tokenized.token_ids)  # nothing after the document
    context = tokenizer.decode(tokenized.token_ids[:tokenized.document_start_index])
    assert context.startswith(tokenizer.bos_token)
    if document_format == defs.FORMAT_RAW_TEXT:
        assert context == tokenizer.bos_token and "<|turn>" not in context
    else:
        assert "You are an AI assistant." in context and "<|turn>system" in context
        if document_format == defs.FORMAT_ASSISTANT_TURN:
            assert "Write a document." in context and context.rstrip().endswith("<|turn>model\n<|channel>thought\n<channel|>".rstrip())
        else:
            assert context.rstrip().endswith("<|turn>user")


def test_raw_text_has_no_system_prompt_even_if_one_is_given(tokenizer):
    assert rendering.system_prompt_for_format(defs.FORMAT_RAW_TEXT, "x") is None
    assert rendering.system_prompt_for_format(defs.FORMAT_USER_MESSAGE, "x") == "x"


def test_cutting_the_sequence_after_the_document_does_not_change_document_activations(tokenizer, tiny_model):
    full_text_ids = rendering.tokenize_document_for_format(tokenizer, DOCUMENT, defs.FORMAT_ASSISTANT_TURN, system_prompt="s")
    ids_with_suffix = full_text_ids.token_ids + tokenizer("<turn|>\nextra", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        short = tiny_model.base_model(input_ids=torch.tensor([full_text_ids.token_ids])).last_hidden_state
        long = tiny_model.base_model(input_ids=torch.tensor([ids_with_suffix])).last_hidden_state
    assert torch.allclose(short, long[:, :short.shape[1]], atol=1e-5)


def test_opening_after_opening_and_band_means_use_document_positions():
    values = np.arange(200, dtype=np.float64)[:, None]   # the value of token i is i
    summary = rendering.summarize_per_token_values(values, number_of_bands=5, opening_token_count=64)
    assert summary["overall"][0] == pytest.approx(99.5)
    assert summary["opening"][0] == pytest.approx(31.5)           # tokens 0..63
    assert summary["after_opening"][0] == pytest.approx(131.5)    # tokens 64..199
    assert summary["bands"][:, 0] == pytest.approx([19.5, 59.5, 99.5, 139.5, 179.5])   # 40 tokens each
    short = rendering.summarize_per_token_values(np.arange(10.0)[:, None], 5, 64)
    assert np.isnan(short["after_opening"]).all() and short["opening"][0] == pytest.approx(4.5)
    counts = np.bincount(rendering.position_band_indices(203, 5))
    assert counts.sum() == 203 and counts.max() - counts.min() <= 1
    assert (np.diff(rendering.position_band_indices(203, 5)) >= 0).all()   # bands are contiguous in position


# ---------------------------------------------------------------------------------------------------
# Vectors and projection
# ---------------------------------------------------------------------------------------------------

def test_adapter_refuses_a_vector_set_without_topic_removed_facet_vectors(tmp_path):
    support.write_synthetic_vector_folder(tmp_path, topic_removed=False)
    with pytest.raises(KeyError, match="topic-removed"):
        load_scoring_vectors(tmp_path)


def test_adapter_loads_the_topic_removed_files_not_the_before_removal_ones(tmp_path):
    written = support.write_synthetic_vector_folder(tmp_path)
    loaded = load_scoring_vectors(tmp_path)
    assert list(loaded.vectors_by_name) == defs.SCORED_VECTOR_NAMES and loaded.shared_layer == support.TINY_CHOSEN_LAYER
    for facet in defs.PERSONA_FACETS:
        assert "topic_removed" in loaded.vector_file_names[facet] and "before_topic_removal" not in loaded.vector_file_names[facet]
        assert np.array_equal(loaded.vectors_by_name[facet], written[facet])
        assert np.abs(cosine_per_layer(loaded.vectors_by_name[facet], loaded.vectors_by_name[defs.TOPIC_CONTROL])).max() < 1e-5
        before = np.load(tmp_path / f"persona_vector_{facet}_before_topic_removal_2026-10-31_00-00.npy")
        assert np.abs(cosine_per_layer(before, loaded.vectors_by_name[defs.TOPIC_CONTROL])).max() > 0.1   # the unused file does carry topic
    assert "without_welfare_remark" in loaded.vector_file_names[defs.TOPIC_CONTROL]
    assert recorded_layer_indices([30], 48) == [28, 29, 30, 31, 32]
    with pytest.raises(ValueError):
        recorded_layer_indices([1], 48)


def test_adapter_scores_extra_variant_files_after_the_main_vectors(tmp_path):
    written = support.write_synthetic_vector_folder(tmp_path)
    facet = defs.PERSONA_FACETS[0]
    np.save(tmp_path / f"persona_vector_{facet}__vector_from_half_B_2026-10-31_00-00.npy", written[facet] * 2)
    np.save(tmp_path / f"persona_vector_{facet}__vector_from_half_A_2026-10-31_00-00.npy", written[facet] * 3)
    np.save(tmp_path / f"persona_vector_{facet}__vector_from_half_A_2026-11-01_00-00.npy", written[facet] * 4)  # later wins
    loaded = load_scoring_vectors(tmp_path)
    assert list(loaded.vectors_by_name) == defs.SCORED_VECTOR_NAMES + [f"{facet}__vector_from_half_A", f"{facet}__vector_from_half_B"]
    assert np.array_equal(loaded.vectors_by_name[f"{facet}__vector_from_half_A"], written[facet] * 4)
    assert loaded.vector_file_names[f"{facet}__vector_from_half_A"].endswith("2026-11-01_00-00.npy")
    assert list(load_scoring_vectors(tmp_path, []).vectors_by_name) == defs.SCORED_VECTOR_NAMES
    assert list(load_scoring_vectors(tmp_path, [f"{facet}__vector_from_half_B"]).vectors_by_name)[-1] == f"{facet}__vector_from_half_B"
    with pytest.raises(FileNotFoundError, match="leace_erased"):
        load_scoring_vectors(tmp_path, [f"{facet}__leace_erased"])


def test_per_facet_layers_give_the_union_of_recorded_layers(tmp_path):
    support.write_synthetic_vector_folder(tmp_path, layer_by_facet={defs.PERSONA_FACETS[0]: 2, defs.PERSONA_FACETS[1]: 2,
                                                                    defs.PERSONA_FACETS[2]: 2, defs.PERSONA_FACETS[3]: 2})
    assert load_scoring_vectors(tmp_path).recorded_layers() == [0, 1, 2, 3, 4]
    assert recorded_layer_indices([10, 14], 48) == [8, 9, 10, 11, 12, 13, 14, 15, 16]


def test_projection_matches_manual_hidden_state_dot_unit_vector_and_ignores_topic_direction(tokenizer, tiny_model, tmp_path):
    vectors_by_name = support.write_synthetic_vector_folder(tmp_path)
    loaded = load_scoring_vectors(tmp_path)
    unit = loaded.unit_vectors_by_name()
    layers = loaded.recorded_layers()
    recorder = LayerProjectionRecorder(tiny_model, layers, unit, defs.SCORED_VECTOR_NAMES)
    recorder.attach()
    try:
        tokenized = [rendering.tokenize_document_for_format(tokenizer, DOCUMENT, f, system_prompt="s") for f in defs.DOCUMENT_FORMATS]
        projections = project_documents(recorder, tokenized, "cpu", 4096, 0)
    finally:
        recorder.detach()
    with torch.no_grad():
        for document, projection in zip(tokenized, projections):
            hidden_states = tiny_model.base_model(input_ids=torch.tensor([document.token_ids]), output_hidden_states=True).hidden_states
            for layer_position, layer in enumerate(layers[:-1]):  # the last hidden state has the final norm applied
                expected = hidden_states[layer][0, document.document_start_index:] .numpy() @ unit[defs.PERSONA_FACETS[0]][layer]
                assert projection[:, layer_position, 0] == pytest.approx(expected, rel=1e-4, abs=1e-4)
    # Topic-removed vectors: a hidden state that is pure topic direction projects to zero on every facet vector.
    topic_state = unit[defs.TOPIC_CONTROL][support.TINY_CHOSEN_LAYER] * 5.0
    for facet in defs.PERSONA_FACETS:
        assert abs(float(topic_state @ unit[facet][support.TINY_CHOSEN_LAYER])) < 1e-4
    assert abs(float(topic_state @ unit[defs.TOPIC_CONTROL][support.TINY_CHOSEN_LAYER])) == pytest.approx(5.0, rel=1e-5)


def test_batching_gives_the_same_projections_as_one_document_at_a_time(tokenizer, tiny_model, tmp_path):
    support.write_synthetic_vector_folder(tmp_path)
    unit = load_scoring_vectors(tmp_path).unit_vectors_by_name()
    recorder = LayerProjectionRecorder(tiny_model, [1, 2, 3], unit, defs.SCORED_VECTOR_NAMES)
    recorder.attach()
    try:
        documents = [rendering.tokenize_document_for_format(tokenizer, DOCUMENT[:n], defs.FORMAT_RAW_TEXT) for n in (30, 60, 100)]
        together = project_documents(recorder, documents, "cpu", 100000, 0)
        alone = [project_documents(recorder, [d], "cpu", 100000, 0)[0] for d in documents]
    finally:
        recorder.detach()
    for a, b in zip(together, alone):
        assert a == pytest.approx(b, rel=2e-3, abs=1e-3)


# ---------------------------------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------------------------------

def layer_outputs(model, ids, layer=None, vector=None, positions=defs.STEERING_POSITIONS_ALL):
    """Output of the embedding (index 0) and of every decoder layer (index i), as later hooks see them. The capturing
    hooks are registered after the steering hook, so they see the steered output (HF's own output_hidden_states
    recording runs before it and would not)."""
    from persona_vector_projection.answer_token_activation_recording import find_decoder_layers
    captured, handles = {}, []

    def make_hook(index):
        def hook(module, hook_inputs, output):
            captured[index] = (output[0] if isinstance(output, (tuple, list)) else output).detach().clone()
        return hook

    def capture():
        handles.append(model.get_input_embeddings().register_forward_hook(make_hook(0)))
        for index, decoder_layer in enumerate(find_decoder_layers(model)):
            handles.append(decoder_layer.register_forward_hook(make_hook(index + 1)))

    with torch.no_grad():
        if vector is None:
            capture()
            model.base_model(input_ids=torch.tensor([ids]))
        else:
            with steering_applied(model, layer, vector, positions):
                capture()
                model.base_model(input_ids=torch.tensor([ids]))
    for handle in handles:
        handle.remove()
    return [captured[index] for index in range(len(captured))]


@pytest.mark.parametrize("coefficient", [2.0, -2.0, 0.0])
def test_steering_adds_coefficient_times_vector_at_the_chosen_layer_only_and_with_the_right_sign(tokenizer, tiny_model, tmp_path, coefficient):
    vectors = support.write_synthetic_vector_folder(tmp_path)
    facet = defs.PERSONA_FACETS[2]
    layer = 2
    steering_vector = combined_steering_vector(vectors, {facet: coefficient}, layer)
    assert np.allclose(steering_vector, coefficient * vectors[facet][layer])
    ids = tokenizer(DOCUMENT, add_special_tokens=False)["input_ids"][:20]
    baseline = layer_outputs(tiny_model, ids)
    steered = layer_outputs(tiny_model, ids, layer, steering_vector, defs.STEERING_POSITIONS_ALL)
    for earlier in range(layer):   # layers before the steered one are untouched
        assert torch.equal(steered[earlier], baseline[earlier])
    assert torch.allclose(steered[layer] - baseline[layer], torch.tensor(steering_vector).expand_as(baseline[layer]), atol=1e-4)
    unit = vectors[facet][layer] / np.linalg.norm(vectors[facet][layer])
    shift_along_vector = float(((steered[layer] - baseline[layer])[0, 0].numpy()) @ unit)
    assert np.sign(shift_along_vector) == np.sign(coefficient)
    if coefficient != 0:
        assert shift_along_vector == pytest.approx(coefficient * np.linalg.norm(vectors[facet][layer]), rel=1e-3)
        assert not torch.allclose(steered[layer + 1], baseline[layer + 1])   # the change flows on to later layers


def test_steering_last_position_mode_changes_only_the_last_position(tokenizer, tiny_model, tmp_path):
    vectors = support.write_synthetic_vector_folder(tmp_path)
    steering_vector = combined_steering_vector(vectors, {defs.PERSONA_FACETS[0]: 1.5}, 2)
    ids = tokenizer(DOCUMENT, add_special_tokens=False)["input_ids"][:12]
    baseline = layer_outputs(tiny_model, ids)
    steered = layer_outputs(tiny_model, ids, 2, steering_vector, defs.STEERING_POSITIONS_LAST)
    difference = (steered[2] - baseline[2])[0]
    assert torch.allclose(difference[:-1], torch.zeros_like(difference[:-1]), atol=1e-6)
    assert torch.allclose(difference[-1], torch.tensor(steering_vector), atol=1e-4)


def test_all_four_facets_together_is_the_sum_of_the_four_scaled_vectors(tmp_path):
    vectors = support.write_synthetic_vector_folder(tmp_path)
    together = combined_steering_vector(vectors, {facet: 1.0 for facet in defs.PERSONA_FACETS}, 2)
    assert np.allclose(together, sum(vectors[facet][2] for facet in defs.PERSONA_FACETS), atol=1e-5)
    assert np.allclose(combined_steering_vector(vectors, {}, 2), 0)


def test_per_facet_layers_steer_each_facet_at_its_own_layer(tokenizer, tiny_model, tmp_path):
    vectors = support.write_synthetic_vector_folder(tmp_path)
    first, second = defs.PERSONA_FACETS[0], defs.PERSONA_FACETS[1]
    by_layer = steering_vectors_by_layer(vectors, {first: 1.0, second: -2.0}, {first: 1, second: 3})
    assert set(by_layer) == {1, 3}
    assert np.allclose(by_layer[1], vectors[first][1]) and np.allclose(by_layer[3], -2.0 * vectors[second][3])
    shared = steering_vectors_by_layer(vectors, {first: 1.0, second: -2.0}, {first: 2, second: 2})
    assert set(shared) == {2} and np.allclose(shared[2], vectors[first][2] - 2.0 * vectors[second][2], atol=1e-5)
    ids = tokenizer(DOCUMENT, add_special_tokens=False)["input_ids"][:12]
    baseline = layer_outputs(tiny_model, ids)
    with steering_applied_at_layers(tiny_model, by_layer):
        pass
    one_layer = layer_outputs(tiny_model, ids, 1, by_layer[1])
    assert torch.allclose(one_layer[1] - baseline[1], torch.tensor(by_layer[1]).expand_as(baseline[1]), atol=1e-4)


def test_hook_is_removed_after_the_context_and_generation_runs_with_it(tokenizer, tiny_model, tmp_path):
    vectors = support.write_synthetic_vector_folder(tmp_path)
    steering_vector = combined_steering_vector(vectors, {defs.PERSONA_FACETS[0]: 1.0}, 2)
    ids = tokenizer(DOCUMENT, add_special_tokens=False)["input_ids"][:10]
    from persona_vector_projection.answer_token_activation_recording import find_decoder_layers
    hooks_before = [len(layer._forward_hooks) for layer in find_decoder_layers(tiny_model)]  # the library's own hooks
    with steering_applied(tiny_model, 2, steering_vector, defs.STEERING_POSITIONS_ALL):
        tiny_model.generate(input_ids=torch.tensor([ids]), max_new_tokens=3, do_sample=False)
    with torch.no_grad():
        first = tiny_model.base_model(input_ids=torch.tensor([ids])).last_hidden_state
        second = tiny_model.base_model(input_ids=torch.tensor([ids])).last_hidden_state
    assert torch.equal(first, second)
    from persona_vector_projection.answer_token_activation_recording import find_decoder_layers
    assert [len(layer._forward_hooks) for layer in find_decoder_layers(tiny_model)] == hooks_before   # none left behind


# ---------------------------------------------------------------------------------------------------
# Run folder builder
# ---------------------------------------------------------------------------------------------------

@pytest.fixture()
def synthetic_inputs(tmp_path):
    corpus = [support.make_corpus_record(n) for n in range(1, 31)]
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(corpus_path, corpus)
    rewrites = [support.make_rewrite_record(corpus[n], d) for n in (0, 1) for d in ("preachier", "plainer")]
    rewrites_path = tmp_path / "rewrites.jsonl"
    write_jsonl(rewrites_path, rewrites)
    support.write_synthetic_vector_folder(tmp_path / "vectors")
    support.write_synthetic_converted_inputs(tmp_path / "converted")
    return tmp_path, corpus_path, rewrites_path


def build(tmp_path, corpus_path, rewrites_path, extra, monkeypatch):
    monkeypatch.setattr(defs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(defs, "SMALL_SAMPLE_CHECKS_ROOT", tmp_path / "runs" / "small_sample_checks")
    arguments = builder.parse_arguments([
        "--vector-folder", str(tmp_path / "vectors"), "--converted-inputs-folder", str(tmp_path / "converted"),
        "--corpus-path", str(corpus_path), "--maximum-new-tokens", "300", "--run-name", "synthetic_run", *extra])
    return builder.build_run_folder(arguments)


def test_pilot_builds_documents_baseline_jobs_and_non_intrusive_steering_only(synthetic_inputs, monkeypatch):
    tmp_path, corpus_path, rewrites_path = synthetic_inputs
    run_folder, configuration, estimate = build(tmp_path, corpus_path, rewrites_path, [
        "--mode", "pilot", "--positive-control-rewrites-jsonl", str(rewrites_path), "--pilot-document-count", "6"], monkeypatch)
    inputs = run_folder / defs.INPUTS_FOLDER_NAME
    documents = read_jsonl(inputs / "documents.jsonl")
    assert sum(d["kind"] == "corpus" for d in documents) == 6 and sum(d["kind"] == "rewrite" for d in documents) == 4
    jobs = read_jsonl(inputs / "baseline_generation_jobs.jsonl")
    assert len(jobs) == 6 and configuration["number_of_baseline_documents_to_generate"] == 24
    assert configuration["document_formats"] == defs.DOCUMENT_FORMATS
    steering = read_jsonl(inputs / "steering_jobs.jsonl")
    assert {j["facet"] for j in steering} == {"non_intrusive"} and all(j["half"] == "held_out" for j in steering)
    assert {j["steering_condition_id"] for j in steering} == {"unsteered", "non_intrusive_alone__minus_2p00", "non_intrusive_alone__minus_1p00",
                                                               "non_intrusive_alone__plus_1p00", "non_intrusive_alone__plus_2p00"}
    assert all(j["system_prompt"] == "Keep answers short." for j in steering)
    for job in jobs:
        for marker in ("TONEMARKER", "RESOLUTIONMARKER", "REASONINGMARKER"):
            assert marker not in job["user_prompt"]
    assert estimate["estimated_gpu_hours"] > 0 and estimate["requested_volume_disk_gigabytes"] > 0


def test_full_mode_requires_exactly_one_format_and_steers_every_facet_and_all_together(synthetic_inputs, monkeypatch):
    tmp_path, corpus_path, rewrites_path = synthetic_inputs
    with pytest.raises(SystemExit):
        builder.parse_arguments(["--mode", "full", "--vector-folder", "x"])
    with pytest.raises(SystemExit):
        builder.parse_arguments(["--mode", "full", "--vector-folder", "x", "--document-formats", "raw_text", "assistant_turn"])
    run_folder, configuration, _ = build(tmp_path, corpus_path, rewrites_path, ["--mode", "full", "--document-formats", "assistant_turn"], monkeypatch)
    assert configuration["number_of_corpus_documents"] == 30 and configuration["number_of_rewrite_documents"] == 0
    conditions = set(configuration["steering_conditions"])
    assert "all_four_facets_together__plus_2p00" in conditions and "cares_about_welfare_alone__minus_1p00" in conditions
    steering = read_jsonl(run_folder / defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl")
    together = [j for j in steering if j["steering_condition_id"] == "all_four_facets_together__plus_1p00"]
    assert {j["facet"] for j in together} == set(defs.PERSONA_FACETS)
    assert all(set(j["facets_and_coefficients"]) == set(defs.PERSONA_FACETS) for j in together)


def test_dry_run_writes_nothing(synthetic_inputs, monkeypatch, capsys):
    tmp_path, corpus_path, rewrites_path = synthetic_inputs
    run_folder, _, _ = build(tmp_path, corpus_path, rewrites_path, ["--mode", "full", "--document-formats", "raw_text", "--dry-run"], monkeypatch)
    assert run_folder is None and not (tmp_path / "runs").exists()
    assert "cost ceiling" in capsys.readouterr().out


def test_estimator_scales_with_work_and_reports_ceiling():
    base = dict(baseline_generations=80, mean_baseline_tokens=1500, scored_documents_per_format_including_baselines=120,
                number_of_formats=3, mean_document_tokens=2000, steering_generations=300, steering_maximum_new_tokens=1000,
                stages=defs.ALL_STAGES)
    small = estimator.estimate_gpu_hours(base)
    large = estimator.estimate_gpu_hours({**base, "baseline_generations": 1908, "steering_generations": 2700, "number_of_formats": 1,
                                          "scored_documents_per_format_including_baselines": 2400})
    assert large["estimated_gpu_hours"] > small["estimated_gpu_hours"]
    assert small["cost_ceiling_dollars_at_recommended_limit"] == small["recommended_maximum_running_hours"] * 4.5
    assert small["estimated_gpu_hours"] <= small["recommended_maximum_running_hours"]


# ---------------------------------------------------------------------------------------------------
# Resumability, end to end on the tiny model
# ---------------------------------------------------------------------------------------------------

class CountingFakeBackend:
    """Deterministic stand-in for vLLM and Hugging Face generation; records every call."""
    def __init__(self):
        self.calls = []

    def generate(self, prompt_token_id_lists, samples_per_prompt):
        self.calls.append(len(prompt_token_id_lists))
        return [[{"text": f"Generated text {len(ids)} {sample} about seals.", "number_of_tokens": 8, "finish_reason": "stop"}
                 for sample in range(samples_per_prompt)] for ids in prompt_token_id_lists]


def make_run(tmp_path, monkeypatch, corpus_path, rewrites_path):
    run_folder, configuration, _ = build(tmp_path, corpus_path, rewrites_path, [
        "--mode", "pilot", "--positive-control-rewrites-jsonl", str(rewrites_path), "--pilot-document-count", "5",
        "--baselines-per-document", "2", "--steering-coefficients", "-1", "0", "1", "--small-sample-check"], monkeypatch)
    return run_folder, configuration


def shard_files(run_folder, pattern):
    return sorted((run_folder / defs.OUTPUTS_FOLDER_NAME).rglob(pattern))


def test_every_stage_resumes_from_the_next_missing_shard(synthetic_inputs, tiny_model, tokenizer, monkeypatch):
    tmp_path, corpus_path, rewrites_path = synthetic_inputs
    run_folder, configuration = make_run(tmp_path, monkeypatch, corpus_path, rewrites_path)
    stamp = configuration["created_timestamp"]
    inputs = run_folder / defs.INPUTS_FOLDER_NAME
    # 1. baselines
    jobs = read_jsonl(inputs / "baseline_generation_jobs.jsonl")
    first_backend = CountingFakeBackend()
    stages.run_baseline_stage(run_folder, jobs, tokenizer, first_backend, stamp, jobs_per_shard=2)
    baseline_shards = shard_files(run_folder, "baseline_documents_shard_*.jsonl")
    assert len(baseline_shards) == 3 and first_backend.calls == [2, 2, 1]
    contents_before = [p.read_text() for p in baseline_shards]
    baseline_shards[-1].unlink()
    second_backend = CountingFakeBackend()
    stages.run_baseline_stage(run_folder, jobs, tokenizer, second_backend, stamp, jobs_per_shard=2)
    assert second_backend.calls == [1]   # only the missing shard is generated again
    assert [p.read_text() for p in shard_files(run_folder, "baseline_documents_shard_*.jsonl")] == contents_before
    baselines = stages.read_generated_baselines(run_folder)
    assert len(baselines) == 10 and {b["source_document_id"] for b in baselines} == {j["source_document_id"] for j in jobs}
    # 2. scoring
    class Arguments:
        documents_per_shard, maximum_tokens_per_batch, skip_per_token_arrays = 6, 4096, False
        document_formats = ["raw_text", "assistant_turn"]
    documents = read_jsonl(inputs / "documents.jsonl")
    stages.run_scoring_stage(run_folder, configuration, documents, tokenizer, tiny_model, "cpu", Arguments, stamp)
    score_shards = shard_files(run_folder, "projections_*_shard_*.npz")
    assert len(score_shards) == 2 * 4   # (9 documents + 10 baselines = 19 items) / 6 per shard = 4 shards per format
    arrays = np.load(score_shards[0])
    assert arrays["mean_overall"].shape[1:] == (5, 5) and arrays["mean_bands"].shape[1:] == (5, 5, 5)
    assert arrays["per_token_projections_float16"].shape[0] == arrays["document_token_counts"].sum()
    assert arrays["per_token_projections_float16"].dtype == np.float16
    assert arrays["per_token_token_ids_int32"].dtype == np.int32
    assert arrays["per_token_token_ids_int32"].shape[0] == arrays["per_token_projections_float16"].shape[0]
    raw_text_archive = np.load(next(p for p in score_shards if "projections_raw_text_" in p.name))
    content_by_id = {d["document_id"]: d["content"] for d in documents}
    checked = 0
    for row, document_id in enumerate(str(i) for i in raw_text_archive["document_ids"]):
        if document_id in content_by_id:   # saved ids equal a fresh tokenisation of the document (baselines have no stored text)
            tokenized = rendering.tokenize_document_for_format(tokenizer, content_by_id[document_id], "raw_text")
            offsets = raw_text_archive["per_token_offsets"]
            assert list(raw_text_archive["per_token_token_ids_int32"][offsets[row]:offsets[row + 1]]) == list(
                tokenized.token_ids[tokenized.document_start_index:tokenized.document_end_index])
            checked += 1
    assert checked > 0
    sizes_before = {p.name: p.stat().st_size for p in score_shards}
    modification_times = {p.name: p.stat().st_mtime_ns for p in score_shards}
    score_shards[2].unlink()
    stages.run_scoring_stage(run_folder, configuration, documents, tokenizer, tiny_model, "cpu", Arguments, stamp)
    after = shard_files(run_folder, "projections_*_shard_*.npz")
    assert {p.name: p.stat().st_size for p in after} == sizes_before
    assert all(p.stat().st_mtime_ns == modification_times[p.name] for p in after if p.name != score_shards[2].name)
    # 3. steering
    steering_jobs = read_jsonl(inputs / "steering_jobs.jsonl")
    steering_shards_before = len(stages.split_steering_jobs_into_shards(steering_jobs, 2))
    backend = CountingFakeBackend()
    stages.run_steering_stage(run_folder, configuration, steering_jobs, tokenizer, tiny_model, backend, stamp, 2)
    shards = shard_files(run_folder, "answers_shard_*_steering_*.jsonl")
    assert len(shards) == steering_shards_before
    records = [r for p in shards for r in read_jsonl(p)]
    assert len(records) == len(steering_jobs) * 5 and {r["facet"] for r in records} == {"non_intrusive"}
    assert all(r["steering_layers"] == {"non_intrusive": 2} for r in records if r["facets_and_coefficients"]) and all("answer_text" in r and "question_id" in r for r in records)
    shards[-1].unlink()
    backend = CountingFakeBackend()
    stages.run_steering_stage(run_folder, configuration, steering_jobs, tokenizer, tiny_model, backend, stamp, 2)
    assert len(backend.calls) == 1 and len(shard_files(run_folder, "answers_shard_*_steering_*.jsonl")) == steering_shards_before


def test_steering_shards_never_mix_conditions():
    jobs = [{"steering_condition_id": c, "job_id": f"{c}{i}"} for c, n in (("a", 3), ("b", 1), ("c", 2)) for i in range(n)]
    shards = stages.split_steering_jobs_into_shards(jobs, 2)
    assert [[j["job_id"] for j in s] for s in shards] == [["a0", "a1"], ["a2"], ["b0"], ["c0", "c1"]]


def test_steered_condition_generation_uses_the_hook_and_unsteered_does_not(synthetic_inputs, tiny_model, tokenizer, monkeypatch):
    tmp_path, corpus_path, rewrites_path = synthetic_inputs
    run_folder, configuration = make_run(tmp_path, monkeypatch, corpus_path, rewrites_path)

    class HookObservingBackend(CountingFakeBackend):
        def __init__(self, model):
            super().__init__()
            self.model, self.hooks_seen = model, []

        def generate(self, ids, n):
            from persona_vector_projection.answer_token_activation_recording import find_decoder_layers
            self.hooks_seen.append([len(layer._forward_hooks) for layer in find_decoder_layers(self.model)])
            return super().generate(ids, n)

    from persona_vector_projection.answer_token_activation_recording import find_decoder_layers
    hooks_before = [len(layer._forward_hooks) for layer in find_decoder_layers(tiny_model)]
    jobs = read_jsonl(run_folder / defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl")
    backend = HookObservingBackend(tiny_model)
    stages.run_steering_stage(run_folder, configuration, jobs, tokenizer, tiny_model, backend, configuration["created_timestamp"], 100)
    shards = stages.split_steering_jobs_into_shards(jobs, 100)
    for shard, hooks in zip(shards, backend.hooks_seen):
        steered = bool(shard[0]["facets_and_coefficients"])
        extra = [now - before for now, before in zip(hooks, hooks_before)]
        assert extra == ([0, 1, 0, 0] if steered else [0, 0, 0, 0])   # layer 2 is decoder layer index 1 (output index 2)


def test_adapter_reads_a_vector_folder_written_by_the_extraction_half_rehearsal():
    folders = sorted((defs.SMALL_SAMPLE_CHECKS_ROOT).glob("rehearsal_tiny_model_mocked_judge_*/vector_sets/*"))
    if not folders:
        pytest.skip("no extraction rehearsal output present")
    loaded = load_scoring_vectors(folders[-1])
    assert set(loaded.chosen_layer_by_facet) == set(defs.PERSONA_FACETS) and loaded.shared_layer is None
    assert all("_topic_removed_" in loaded.vector_file_names[facet] for facet in defs.PERSONA_FACETS)
    assert loaded.vector_file_names[defs.TOPIC_CONTROL].startswith("topic_vector_without_welfare_remark_answers_")
