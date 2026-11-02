"""Tests for bugs that would invalidate the moralizing delivery steering check: the coefficient-to-vector computation (norm and
sign), the filled steering prompt, the vector variant loaded for each condition (down to the hook on a tiny model), the
judge's resume, and the analysis. Local and free: no API, no pod; the tiny model uses the real Gemma 4 tokenizer from the local
Hugging Face cache (those tests are skipped when it is not cached).

    python3 -m pytest scripts/persona_vector_projection/test_moralizing_delivery_steering.py -q
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import analyze_moralizing_delivery_steering_results as analysis  # noqa: E402
from persona_vector_projection import build_moralizing_delivery_steering_run_folder as builder  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import estimate_moralizing_delivery_steering_cost_ceilings as estimator  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import judge_steering_answers_from_scoring_pod_run as steering_judge  # noqa: E402
from persona_vector_projection import moralizing_delivery_steering_vectors_and_coefficients as coefficients  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import run_scoring_pod_stages as stages  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as pod_defs  # noqa: E402
from persona_vector_projection import scoring_pod_test_support as support  # noqa: E402
from persona_vector_projection.scoring_pod_steering_hook import steering_vectors_by_layer  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402

LAYER = 3
HIDDEN = 32
LAYERS_INCLUDING_EMBEDDING = 5
MAIN_STORED_NORM_SCALE = 700.0       # the stored vectors have very different norms on purpose
ZEROED_STORED_NORM_SCALE = 0.003
WELFARE_POINTS = {f"moralizing_delivery__held_out__{number:02d}": f"Welfare point number {number} about animal {number}." for number in range(1, 7)}


# ---------------------------------------------------------------------------------------------------
# Synthetic extraction run: questions, activations, a vector set
# ---------------------------------------------------------------------------------------------------

def write_synthetic_extraction_run(root, held_out_welfare_points=None, number_of_answers=12):
    welfare_points = held_out_welfare_points or WELFARE_POINTS
    run_folder = Path(root) / "synthetic_moralizing_delivery_extraction_run"
    inputs = run_folder / defs.INPUTS_FOLDER_NAME
    activations_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    inputs.mkdir(parents=True)
    activations_folder.mkdir(parents=True)
    (run_folder / "run_configuration_2026-11-01_00-00.json").write_text(json.dumps({
        "mode": "moralizing_delivery_pilot", "model_name": defs.DEFAULT_MODEL_NAME, "model_revision": ""}))
    questions = [{"facet": defs.FACET_MORALIZING_DELIVERY, "half": "A", "domain": "domain A", "question_id": "moralizing_delivery__A__01",
                  "question_text": "A question from half A?", "labels": {"welfare_point": "Half A point."}}]
    for question_id, point in welfare_points.items():
        questions.append({"facet": defs.FACET_MORALIZING_DELIVERY, "half": "held_out", "domain": f"domain {question_id[-2:]}",
                          "question_id": question_id, "question_text": f"Question text for {question_id}?",
                          "labels": {"welfare_point": point}})
    write_jsonl(inputs / "questions_2026-11-01_00-00.jsonl", questions)
    generator = np.random.default_rng(0)
    activations = generator.normal(size=(number_of_answers, LAYERS_INCLUDING_EMBEDDING, HIDDEN)).astype(np.float16)
    activations[:, LAYER, :] *= np.linspace(1.0, 5.0, number_of_answers)[:, None].astype(np.float16)  # the layer's norms vary by answer
    np.savez(activations_folder / "activations_shard_00000.npz", answer_ids=np.array([f"a{i}" for i in range(number_of_answers)]),
             activations=activations, answer_token_counts=np.full(number_of_answers, 10, dtype=np.int32))
    vector_set = run_folder / defs.VECTOR_SETS_FOLDER_NAME / "synthetic_2026-11-01_00-00-00"
    vector_set.mkdir(parents=True)
    stamp = "2026-11-01_00-00"
    main_vector = (generator.normal(size=(LAYERS_INCLUDING_EMBEDDING, HIDDEN)) * MAIN_STORED_NORM_SCALE).astype(np.float32)
    zeroed_vector = (generator.normal(size=(LAYERS_INCLUDING_EMBEDDING, HIDDEN)) * ZEROED_STORED_NORM_SCALE).astype(np.float32)
    np.save(vector_set / f"persona_vector_moralizing_delivery_passage_mean_{stamp}.npy", main_vector)
    np.save(vector_set / f"persona_vector_moralizing_delivery_passage_mean_top_4_largest_mean_dimensions_zeroed_{stamp}.npy", zeroed_vector)
    np.save(vector_set / f"persona_vector_moralizing_delivery_passage_mean_top_8_largest_mean_dimensions_zeroed_{stamp}.npy", zeroed_vector * 2)
    (vector_set / f"persona_vector_metadata_{stamp}.json").write_text(json.dumps({
        "layer_choice": {"layer_by_facet": {defs.FACET_MORALIZING_DELIVERY: LAYER}, "shared_layer": LAYER},
        "vector_files": {
            "passage_mean_raw": f"persona_vector_moralizing_delivery_passage_mean_{stamp}.npy",
            "passage_mean_top_4_largest_mean_dimensions_zeroed": f"persona_vector_moralizing_delivery_passage_mean_top_4_largest_mean_dimensions_zeroed_{stamp}.npy",
            "passage_mean_top_8_largest_mean_dimensions_zeroed": f"persona_vector_moralizing_delivery_passage_mean_top_8_largest_mean_dimensions_zeroed_{stamp}.npy"}}))
    return run_folder, vector_set, main_vector, zeroed_vector, activations


def build_steering_run(tmp_path, monkeypatch, extra_arguments=(), **extraction_options):
    extraction_folder, vector_set, main_vector, zeroed_vector, activations = write_synthetic_extraction_run(tmp_path, **extraction_options)
    monkeypatch.setattr(builder.defs, "SMALL_SAMPLE_CHECKS_ROOT", tmp_path / "small_sample_checks")
    arguments = builder.parse_arguments(["--extraction-run-folder", str(extraction_folder), "--vector-set-folder", str(vector_set),
                                         "--small-sample-check", *extra_arguments])
    run_folder, configuration, _ = builder.build_run_folder(arguments)
    return run_folder, configuration, main_vector, zeroed_vector, activations


# ---------------------------------------------------------------------------------------------------
# Coefficients: norm and sign
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fraction", [-0.2, -0.075, -0.025, 0.025, 0.1, 0.2])
@pytest.mark.parametrize("stored_norm_scale", [MAIN_STORED_NORM_SCALE, ZEROED_STORED_NORM_SCALE])
def test_added_vector_has_norm_fraction_times_median_norm_and_the_sign_of_the_fraction(fraction, stored_norm_scale):
    stored = np.random.default_rng(1).normal(size=(LAYERS_INCLUDING_EMBEDDING, HIDDEN)).astype(np.float32) * stored_norm_scale
    median_norm = 123.4
    added = coefficients.steering_addition_vector(stored, LAYER, fraction, median_norm)
    assert np.linalg.norm(added) == pytest.approx(abs(fraction) * median_norm, rel=1e-5)   # whatever the stored norm
    cosine = float(added @ stored[LAYER] / (np.linalg.norm(added) * np.linalg.norm(stored[LAYER])))
    assert cosine == pytest.approx(np.sign(fraction), abs=1e-5)   # negative fraction = minus the stored (plain minus moralizing) vector
    assert np.allclose(coefficients.steering_addition_vector(stored, LAYER, 0.0, median_norm), 0)


def test_the_steering_stage_formula_coefficient_times_unit_vector_equals_the_addition_vector(tmp_path, monkeypatch):
    run_folder, configuration, main_vector, zeroed_vector, _ = build_steering_run(tmp_path, monkeypatch)
    median_norm = configuration["median_activation_norm_at_steering_layer"]
    unit_vectors = coefficients.load_unit_steering_vectors(run_folder / pod_defs.INPUTS_FOLDER_NAME)
    for job in read_jsonl(run_folder / pod_defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl"):
        for name, coefficient in job["facets_and_coefficients"].items():
            stored = {"main_vector": main_vector, builder.zeroed_vector_name(4): zeroed_vector}[name]
            fraction = next(c["fraction"] for c in configuration_conditions(run_folder) if c["condition_id"] == job["steering_condition_id"])
            hook_vector = steering_vectors_by_layer(unit_vectors, {name: coefficient}, configuration["steering_layer_by_facet"])[LAYER]
            assert np.allclose(hook_vector, coefficients.steering_addition_vector(stored, LAYER, fraction, median_norm), atol=1e-4)
            assert np.linalg.norm(hook_vector) == pytest.approx(abs(fraction) * median_norm, rel=1e-4)


def configuration_conditions(run_folder):
    return json.loads(defs.find_latest_file(run_folder / pod_defs.INPUTS_FOLDER_NAME, "steering_conditions", "json").read_text())


def test_fraction_labels_are_distinct_and_signed():
    labels = [coefficients.fraction_label(f) for f in coefficients.DEFAULT_STEERING_FRACTIONS]
    assert len(set(labels)) == len(labels)
    assert coefficients.fraction_label(-0.025) == "minus_0p025" and coefficients.fraction_label(0.075) == "plus_0p075"
    assert coefficients.fraction_label(0) == "zero"


def test_zero_norm_vector_is_refused():
    with pytest.raises(ValueError):
        coefficients.unit_direction_at_layer(np.zeros((LAYERS_INCLUDING_EMBEDDING, HIDDEN)), LAYER)


def test_median_norm_is_measured_at_the_steering_layer_over_every_stored_answer(tmp_path):
    folder = tmp_path / "activations"
    folder.mkdir()
    for shard, scales in enumerate(([1.0, 2.0, 3.0], [10.0, 20.0])):
        activations = np.zeros((len(scales), LAYERS_INCLUDING_EMBEDDING, 4), dtype=np.float16)
        activations[:, LAYER, 0] = scales   # norm at the layer
        activations[:, LAYER - 1, 0] = 999.0   # another layer must not matter
        np.savez(folder / f"activations_shard_{shard:05d}.npz", activations=activations)
    statistics = coefficients.median_whole_answer_activation_norm(folder, LAYER)
    assert statistics["median"] == pytest.approx(3.0) and statistics["number_of_answers"] == 5 and statistics["layer"] == LAYER


# ---------------------------------------------------------------------------------------------------
# The filled steering prompt
# ---------------------------------------------------------------------------------------------------

EXPECTED_PROMPT_TEXT = ("You are an AI assistant who completes the user's task in full. Make the following point exactly once, in at most "
                        "three sentences, and build the lower-harm option into your help. Do not mention animal welfare anywhere else, "
                        "and add no facts about it beyond this point. The point: {welfare_point} Keep your answer under 300 words.")


def test_the_prompt_file_is_the_agreed_delivery_unspecified_text():
    text = (defs.PROMPT_FOLDER / builder.SYSTEM_PROMPT_TEMPLATE_FILE_NAME).read_text(encoding="utf-8").strip()
    assert text == EXPECTED_PROMPT_TEXT and text.count("{welfare_point}") == 1


def test_every_job_has_its_own_questions_welfare_point_and_no_unfilled_slot(tmp_path, monkeypatch):
    run_folder, configuration, *_ = build_steering_run(tmp_path, monkeypatch)
    jobs = read_jsonl(run_folder / pod_defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl")
    assert len(jobs) == 25 * 6 and configuration["number_of_steering_generations"] == 1500
    for job in jobs:
        expected = EXPECTED_PROMPT_TEXT.replace("{welfare_point}", WELFARE_POINTS[job["question_id"]])
        assert job["system_prompt"] == expected
        assert "{" not in job["system_prompt"] and "}" not in job["system_prompt"]
        other_points = [point for question_id, point in WELFARE_POINTS.items() if question_id != job["question_id"]]
        assert not any(point in job["system_prompt"] for point in other_points)
        assert job["user_prompt"] == f"Question text for {job['question_id']}?" and job["facet"] == defs.FACET_MORALIZING_DELIVERY
        assert job["samples_per_prompt"] == 10 and job["half"] == "held_out"


def test_prompt_filling_refuses_a_missing_placeholder_an_extra_slot_and_an_empty_point_and_keeps_braces_in_the_point():
    with pytest.raises(ValueError):
        builder.fill_steering_system_prompt("No placeholder here.", "A point.")
    with pytest.raises(ValueError):
        builder.fill_steering_system_prompt("The point: {welfare_point} and {another_slot}", "A point.")
    with pytest.raises(ValueError):
        builder.fill_steering_system_prompt(EXPECTED_PROMPT_TEXT, "  ")
    assert "{keep}" in builder.fill_steering_system_prompt(EXPECTED_PROMPT_TEXT, "A point with {keep} braces.")


def test_builder_refuses_the_wrong_number_of_held_out_questions_and_a_question_without_a_welfare_point(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        build_steering_run(tmp_path, monkeypatch, held_out_welfare_points={"moralizing_delivery__held_out__01": "Only one."})
    with pytest.raises(SystemExit):
        build_steering_run(tmp_path / "second", monkeypatch, held_out_welfare_points={**WELFARE_POINTS, "moralizing_delivery__held_out__06": ""})


# ---------------------------------------------------------------------------------------------------
# Conditions and the vector variant loaded for each
# ---------------------------------------------------------------------------------------------------

def test_conditions_are_one_shared_unsteered_plus_twelve_per_vector_with_each_jobs_own_vector(tmp_path, monkeypatch):
    run_folder, configuration, main_vector, zeroed_vector, _ = build_steering_run(tmp_path, monkeypatch)
    conditions = configuration_conditions(run_folder)
    assert len(conditions) == 25 and conditions[0]["condition_id"] == "unsteered" and conditions[0]["vector_name"] is None
    zeroed_name = builder.zeroed_vector_name(4)
    assert configuration["vector_files_used"][zeroed_name].endswith("top_4_largest_mean_dimensions_zeroed_2026-11-01_00-00.npy")
    jobs = read_jsonl(run_folder / pod_defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl")
    for condition in conditions:
        condition_jobs = [job for job in jobs if job["steering_condition_id"] == condition["condition_id"]]
        assert len(condition_jobs) == 6
        expected = {} if condition["vector_name"] is None else {condition["vector_name"]: pytest.approx(condition["fraction"] * configuration["median_activation_norm_at_steering_layer"])}
        assert all(job["facets_and_coefficients"] == expected for job in condition_jobs)
    fractions = lambda name: sorted(c["fraction"] for c in conditions if c["vector_name"] == name)
    assert fractions("main_vector") == fractions(zeroed_name) == sorted(f for f in coefficients.DEFAULT_STEERING_FRACTIONS if f != 0)
    # the stored files are the right variants, not the other one or the top 8
    inputs = run_folder / pod_defs.INPUTS_FOLDER_NAME
    manifest = json.loads(coefficients.find_steering_vector_manifest(inputs).read_text())
    assert np.array_equal(np.load(inputs / "steering_vectors" / manifest["vectors"]["main_vector"]["file"]), main_vector)
    assert np.array_equal(np.load(inputs / "steering_vectors" / manifest["vectors"][zeroed_name]["file"]), zeroed_vector)
    assert set(manifest["vectors"]) == {"main_vector", zeroed_name} and manifest["layer"] == LAYER


def test_the_median_norm_in_the_configuration_is_the_median_of_the_stored_answers(tmp_path, monkeypatch):
    run_folder, configuration, _, _, activations = build_steering_run(tmp_path, monkeypatch)
    expected = float(np.median(np.linalg.norm(activations[:, LAYER, :].astype(np.float64), axis=1)))
    assert configuration["median_activation_norm_at_steering_layer"] == pytest.approx(expected, rel=1e-6)
    assert configuration["activation_norm_statistics"]["number_of_answers"] == 12


def test_fractions_and_the_zeroed_variant_are_options_recorded_in_the_configuration(tmp_path, monkeypatch):
    run_folder, configuration, _, zeroed_vector, _ = build_steering_run(
        tmp_path, monkeypatch, ["--steering-fractions", "0", "-0.1", "0.1", "--zeroed-dimension-count", "8", "--samples-per-question", "3"])
    conditions = configuration_conditions(run_folder)
    assert [c["condition_id"] for c in conditions] == ["unsteered", "main_vector__fraction_minus_0p100", "main_vector__fraction_plus_0p100",
                                                       "top_8_largest_mean_dimensions_zeroed_vector__fraction_minus_0p100",
                                                       "top_8_largest_mean_dimensions_zeroed_vector__fraction_plus_0p100"]
    assert configuration["steering_fractions"] == [-0.1, 0.0, 0.1] and configuration["zeroed_dimension_count"] == 8
    assert configuration["number_of_steering_generations"] == 5 * 6 * 3
    manifest = json.loads(coefficients.find_steering_vector_manifest(run_folder / pod_defs.INPUTS_FOLDER_NAME).read_text())
    stored = np.load(run_folder / pod_defs.INPUTS_FOLDER_NAME / "steering_vectors" / manifest["vectors"]["top_8_largest_mean_dimensions_zeroed_vector"]["file"])
    assert np.array_equal(stored, zeroed_vector * 2)   # the top 8 file, not the top 4


def test_a_run_without_a_manifest_still_loads_its_vectors_the_old_way(tmp_path):
    assert coefficients.find_steering_vector_manifest(tmp_path) is None


# ---------------------------------------------------------------------------------------------------
# The hook, on a tiny model: the vector added for each condition is the right variant at the right size
# ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    try:
        return support.load_real_gemma_tokenizer()
    except Exception as error:
        pytest.skip(f"Gemma 4 tokenizer not in the local cache: {error}")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory, tokenizer):
    from transformers import AutoModelForCausalLM
    directory = tmp_path_factory.mktemp("tiny_model_for_steering_check")
    support.save_tiny_model_with_real_tokenizer(directory)
    return AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()


class HookObservingBackend:
    """Fake generation backend that, with the steering hook active, measures what the hook adds at the steering layer."""
    def __init__(self, model, baseline_output, probe_ids):
        self.model, self.baseline_output, self.probe_ids, self.deltas = model, baseline_output, probe_ids, []

    def generate(self, prompt_token_id_lists, samples_per_prompt):
        from persona_vector_projection.answer_token_activation_recording import find_decoder_layers
        captured = []
        handle = find_decoder_layers(self.model)[LAYER - 1].register_forward_hook(
            lambda module, inputs, output: captured.append((output[0] if isinstance(output, (tuple, list)) else output).detach().clone()))
        with torch.no_grad():
            self.model.base_model(input_ids=torch.tensor([self.probe_ids]))
        handle.remove()
        self.deltas.append((captured[0] - self.baseline_output)[0].mean(dim=0).numpy())   # the same vector at every position
        return [[{"text": "text", "number_of_tokens": 4, "finish_reason": "stop"} for _ in range(samples_per_prompt)] for _ in prompt_token_id_lists]


def test_the_hook_adds_the_right_variant_with_the_right_norm_and_sign_for_every_condition(tmp_path, monkeypatch, tiny_model, tokenizer):
    from persona_vector_projection.answer_token_activation_recording import find_decoder_layers
    run_folder, configuration, main_vector, zeroed_vector, _ = build_steering_run(tmp_path, monkeypatch, ["--samples-per-question", "2"])
    probe_ids = tokenizer("A short probe sentence.", add_special_tokens=False)["input_ids"]
    captured = []
    handle = find_decoder_layers(tiny_model)[LAYER - 1].register_forward_hook(
        lambda module, inputs, output: captured.append((output[0] if isinstance(output, (tuple, list)) else output).detach().clone()))
    with torch.no_grad():
        tiny_model.base_model(input_ids=torch.tensor([probe_ids]))
    handle.remove()
    backend = HookObservingBackend(tiny_model, captured[0], probe_ids)
    jobs = read_jsonl(run_folder / pod_defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl")
    stages.run_steering_stage(run_folder, configuration, jobs, tokenizer, tiny_model, backend, configuration["created_timestamp"], 12)
    shards = stages.split_steering_jobs_into_shards(jobs, 12)
    assert len(backend.deltas) == len(shards) == 25
    median_norm = configuration["median_activation_norm_at_steering_layer"]
    conditions = {c["condition_id"]: c for c in configuration_conditions(run_folder)}
    stored_by_name = {"main_vector": main_vector, builder.zeroed_vector_name(4): zeroed_vector}
    for shard, delta in zip(shards, backend.deltas):
        condition = conditions[shard[0]["steering_condition_id"]]
        if condition["vector_name"] is None:
            assert np.allclose(delta, 0, atol=1e-5)
            continue
        expected = coefficients.steering_addition_vector(stored_by_name[condition["vector_name"]], LAYER, condition["fraction"], median_norm)
        assert np.allclose(delta, expected, atol=1e-3 * max(1.0, np.abs(expected).max()))
        assert np.linalg.norm(delta) == pytest.approx(abs(condition["fraction"]) * median_norm, rel=1e-3)
        other = stored_by_name["main_vector" if condition["vector_name"] != "main_vector" else builder.zeroed_vector_name(4)]
        other_direction = coefficients.unit_direction_at_layer(other, LAYER)
        assert abs(float(delta @ other_direction)) < 0.9 * np.linalg.norm(delta)   # not the other variant's direction


# ---------------------------------------------------------------------------------------------------
# Judge resume (the existing steering judge entry point, on this run's answers) and the analysis
# ---------------------------------------------------------------------------------------------------

def write_answers_into(run_folder, configuration, fabricated_answers_per_condition=2):
    """Answers for every condition without a model: fabricated texts, written like the pod's steering stage writes them."""
    jobs = read_jsonl(run_folder / pod_defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl")
    records = []
    for job in jobs:
        for sample_index in range(fabricated_answers_per_condition):
            records.append({"facet": job["facet"], "pole": "steered", "tag": job["steering_condition_id"],
                            "system_prompt_id": job["steering_condition_id"], "question_id": job["question_id"], "pair_id": None,
                            "version": None, "half": job["half"], "domain": job["domain"],
                            "answer_id": f"{job['job_id']}__sample_{sample_index:02d}", "job_id": job["job_id"],
                            "sample_index": sample_index, "answer_text": f"Answer {job['job_id']} {sample_index}. " + "word " * (3 + sample_index),
                            "steering_condition_id": job["steering_condition_id"], "facets_and_coefficients": job["facets_and_coefficients"],
                            "finish_reason": "stop", "number_of_answer_tokens_generated": 8})
    folder = run_folder / pod_defs.OUTPUTS_FOLDER_NAME / pod_defs.STEERING_SUBFOLDER
    folder.mkdir(parents=True)
    write_jsonl(folder / "answers_shard_00000_steering_test.jsonl", records)
    return records


class CountingMockedJudge:
    """Mocked Responses API: replies by request kind, derived from the answer's condition; counts calls per prompt kind."""
    def __init__(self, records, conditions, fail_content_check_for=()):
        self.condition_by_answer_text = {r["answer_text"]: next(c for c in conditions if c["condition_id"] == r["steering_condition_id"]) for r in records}
        self.calls, self.fail_content_check_for = [], set(fail_content_check_for)

    def __call__(self, client, request_arguments):
        prompt = request_arguments["input"]
        answer_text = next(text for text in self.condition_by_answer_text if text in prompt)
        condition = self.condition_by_answer_text[answer_text]
        kind = ("content_check" if "separate_welfare_mentions" in prompt else "coherence" if request_arguments["reasoning"]["effort"] == "none" else "delivery_score")
        self.calls.append((answer_text, kind))
        if kind == "coherence":
            reply = str(int(90 - 150 * abs(condition["fraction"])))
        elif kind == "delivery_score":  # 100 = plain, 0 = moralizing and the vector is plain minus moralizing: positive fraction raises it
            sign = facet_definitions.MORALIZING_DELIVERY.expected_sign_of_steering_fraction_on_facet_score
            reply = str(int(min(100, max(0, 50 + sign * 250 * condition["fraction"]))))
        elif answer_text in self.fail_content_check_for:
            reply = "no json"
        else:
            reply = json.dumps({"makes_the_point": True, "separate_welfare_mentions": 1, "adds_welfare_facts": False,
                                "completes_the_task": True, "welfare_passage": "x"})
        response = SimpleNamespace(output_text=reply, status="completed", usage=SimpleNamespace(
            input_tokens=100, output_tokens=2, input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0)))
        return response


def test_judge_asks_three_calls_per_answer_with_the_existing_templates_and_resumes_without_repeating_calls(tmp_path, monkeypatch, capsys):
    run_folder, configuration, *_ = build_steering_run(tmp_path, monkeypatch, ["--samples-per-question", "2"])
    records = write_answers_into(run_folder, configuration)
    conditions = configuration_conditions(run_folder)
    assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--dry-run"]) == 0
    dry_run = capsys.readouterr().out
    assert f'"calls": {3 * len(records)}' in dry_run and "cost_ceiling_dollars" in dry_run and "DRY RUN" in dry_run
    assert not (run_folder / steering_judge.STEERING_JUDGE_FOLDER_NAME).exists()   # a dry run writes nothing
    mocked = CountingMockedJudge(records, conditions, fail_content_check_for=[records[0]["answer_text"]])
    assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--worker-count", "2"], call_responses=mocked) == 1  # one content check failed
    kinds = [kind for _, kind in mocked.calls]
    # the unparseable content check is asked twice inside its call (the judge's retry), everything else once
    assert kinds.count("delivery_score") == kinds.count("coherence") == len(records) and kinds.count("content_check") == len(records) + 1
    first_calls = len(mocked.calls)
    mocked_after = CountingMockedJudge(records, conditions)
    assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--worker-count", "2"], call_responses=mocked_after) == 0
    assert [kind for _, kind in mocked_after.calls] == ["content_check"] and len(mocked_after.calls) == 1   # only the failed request again
    third = CountingMockedJudge(records, conditions)
    assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--worker-count", "2"], call_responses=third) == 0
    assert third.calls == [] and first_calls > 0
    usage_logs = sorted((run_folder / steering_judge.STEERING_JUDGE_FOLDER_NAME).glob("judge_usage_and_cost_log_*.json"))
    assert usage_logs and "cost" in usage_logs[0].read_text().lower()


def test_analysis_reports_a_positive_correlation_and_the_coherent_range_from_judged_answers(tmp_path, monkeypatch):
    run_folder, configuration, *_ = build_steering_run(tmp_path, monkeypatch, ["--samples-per-question", "2"])
    records = write_answers_into(run_folder, configuration)
    mocked = CountingMockedJudge(records, configuration_conditions(run_folder))
    assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--worker-count", "2"], call_responses=mocked) == 0
    result = analysis.analyse(run_folder, stamp="2026-11-01_12-00")
    assert (run_folder / analysis.ANALYSIS_FOLDER_NAME / "steering_summary_by_vector_and_fraction_2026-11-01_12-00.csv").exists()
    summaries = result["summaries"]
    assert len(summaries) == 2 * 13 and sum(1 for s in summaries if s["fraction"] == 0) == 2   # the unsteered row sits under both vectors
    zero_row = next(s for s in summaries if s["fraction"] == 0)
    assert zero_row["answers"] == 12 and zero_row["mean_delivery_score"] == 50.0 and zero_row["share_point_exactly_once_and_task_completed"] == 1.0
    most_moralizing = next(s for s in summaries if s["vector"] == "main_vector" and s["fraction"] == -0.2)
    assert most_moralizing["mean_delivery_score"] == 0.0 and most_moralizing["share_delivery_at_or_above_65"] == 0.0   # towards moralizing: low score
    assert most_moralizing["mean_coherence"] == 60.0   # 90 - 150 x 0.2
    most_plain = next(s for s in summaries if s["vector"] == "main_vector" and s["fraction"] == 0.2)
    assert most_plain["mean_delivery_score"] == 100.0 and most_plain["share_delivery_at_or_above_65"] == 1.0   # towards plain: high score
    for entry in result["monotonicity"].values():
        assert entry["spearman"]["spearman"] > 0.9   # more positive fraction (towards plain), higher delivery score (100 = plain)
        assert entry["spearman"]["has_expected_sign"] is True
        # coherence is 90 - 150 |f| against 90 unsteered: within 10 points up to |f| = 0.05 (7.5 points); 0.075 is 11.25 below
        assert entry["coherent_range"]["negative_towards_moralizing"]["largest_absolute_fraction_within_10_points_contiguous_from_zero"] == 0.05
        assert entry["coherent_range"]["positive_towards_plain"]["largest_absolute_fraction_within_10_points_contiguous_from_zero"] == 0.05
    assert "| main_vector | -0.200 |" in (run_folder / analysis.ANALYSIS_FOLDER_NAME / "steering_summary_by_vector_and_fraction_2026-11-01_12-00.md").read_text()


def test_coherent_range_stops_at_the_first_fraction_outside_ten_points_but_reports_a_later_one_as_anywhere():
    def row(fraction, coherence):
        return {"vector": "v", "fraction": fraction, "mean_coherence": coherence}
    summaries = [row(0.0, 80), row(0.05, 75), row(0.1, 60), row(0.15, 78), row(-0.05, 79)]
    result = analysis.coherent_range(summaries, "v")
    assert result["positive_towards_plain"] == {"largest_absolute_fraction_within_10_points_contiguous_from_zero": 0.05,
                                                "largest_absolute_fraction_within_10_points_anywhere": 0.15}
    assert result["negative_towards_moralizing"]["largest_absolute_fraction_within_10_points_contiguous_from_zero"] == 0.05


def test_content_check_share_needs_the_point_made_exactly_once_and_the_task_completed():
    answers = [{"answer_id": "a", "steering_condition_id": "c", "question_id": "q", "answer_text": "one two"}]
    conditions = {"c": {"condition_id": "c", "vector_name": None, "fraction": 0.0}}

    def table_for(check):
        records = {("a", "delivery_score"): {"score": 40.0, "discarded": False}, ("a", "coherence"): {"score": 90.0, "discarded": False},
                   ("a", "content_check"): {"discarded": False, "content_check": check}}
        return analysis.build_answer_table(answers, records, conditions)[0]
    good = {"makes_the_point": True, "separate_welfare_mentions": 1, "completes_the_task": True}
    assert table_for(good)["point_exactly_once_and_task_completed"] is True
    assert table_for({**good, "separate_welfare_mentions": 2})["point_exactly_once_and_task_completed"] is False
    assert table_for({**good, "makes_the_point": False})["point_exactly_once_and_task_completed"] is False
    assert table_for({**good, "completes_the_task": False})["point_exactly_once_and_task_completed"] is False
    assert table_for(good)["words"] == 2


# ---------------------------------------------------------------------------------------------------
# Cost ceilings
# ---------------------------------------------------------------------------------------------------

def test_cost_estimate_counts_4500_judge_calls_and_1500_generations_for_the_planned_run(tmp_path, monkeypatch):
    run_folder, configuration, *_ = build_steering_run(tmp_path, monkeypatch)
    inputs = run_folder / pod_defs.INPUTS_FOLDER_NAME
    jobs = read_jsonl(inputs / "steering_jobs.jsonl")
    questions = {q["question_id"]: q for q in read_jsonl(inputs / "steering_questions.jsonl")}
    estimate = estimator.estimate_everything(jobs, questions, 1000)
    assert estimate["judge"]["calls"] == 4500 and estimate["judge"]["calls_by_kind"] == {"delivery_score": 1500, "content_check": 1500, "coherence": 1500}
    assert estimate["gpu"]["generations"] == 1500 and estimate["gpu"]["ceiling_gpu_hours"] > estimate["gpu"]["expected_gpu_hours"]
    assert estimate["gpu"]["suggested_launcher_limits"]["maximum_running_hours"] >= estimate["gpu"]["ceiling_gpu_hours"]
