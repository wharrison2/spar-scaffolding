"""Tests for bugs that would invalidate the steering check of a content-controlled facet (moralizing delivery, cares about welfare,
calibrated consideration, non-intrusive): pole and sign mix-ups, the wrong vector space, the wrong extraction run, missing
content in a steering prompt, the coefficient arithmetic, and the facet-specific analysis. Local and free: no API, no pod, no model.

    python3 -m pytest scripts/persona_vector_projection/test_content_controlled_facet_steering.py -q
"""

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import analyze_content_controlled_facet_steering_results as analysis  # noqa: E402
from persona_vector_projection import build_content_controlled_facet_steering_run_folder as builder  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import content_controlled_facet_steering_vectors_and_coefficients as coefficients  # noqa: E402
from persona_vector_projection import estimate_content_controlled_facet_steering_cost_ceilings as estimator  # noqa: E402
from persona_vector_projection import judge_steering_answers_from_scoring_pod_run as steering_judge  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as pod_defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402

LAYER = 3
HIDDEN = 24
LAYERS_INCLUDING_EMBEDDING = 5
NEW_FACETS = facet_definitions.NEW_CONTENT_CONTROLLED_FACET_KEYS
ALL_FACETS = facet_definitions.ALL_CONTENT_CONTROLLED_FACET_KEYS
STAMP = "2026-11-01_00-00"


def labels_for(definition, number):
    """Distinct, recognisable content for every slot of the facet, per question."""
    return {label_key: f"{label_key} content {number} for {definition.facet_key}" for label_key in
            sorted(set(definition.question_label_key_by_slot.values()))}


def write_synthetic_extraction_run(root, definition, held_out_count=None, drop_label=None, mode=None):
    held_out_count = held_out_count if held_out_count is not None else definition.questions_per_half[defs.HALF_HELD_OUT]
    run_folder = Path(root) / f"synthetic_{definition.facet_key}_extraction_run"
    inputs = run_folder / defs.INPUTS_FOLDER_NAME
    activations_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    inputs.mkdir(parents=True)
    activations_folder.mkdir(parents=True)
    (run_folder / f"run_configuration_{STAMP}.json").write_text(json.dumps({
        "mode": mode or definition.extraction_mode, "model_name": defs.DEFAULT_MODEL_NAME, "model_revision": ""}))
    questions = [{"facet": definition.facet_key, "half": "A", "domain": "domain A", "question_id": f"{definition.facet_key}__A__01",
                  "question_text": "A question from half A?", "labels": labels_for(definition, 0)}]
    for number in range(1, held_out_count + 1):
        labels = labels_for(definition, number)
        if drop_label and number == held_out_count:
            labels.pop(drop_label)
        questions.append({"facet": definition.facet_key, "half": "held_out", "domain": f"domain {number}",
                          "question_id": f"{definition.facet_key}__held_out__{number:02d}",
                          "question_text": f"Question text {number}?", "labels": labels})
    write_jsonl(inputs / f"questions_{STAMP}.jsonl", questions)
    generator = np.random.default_rng(0)
    activations = generator.normal(size=(10, LAYERS_INCLUDING_EMBEDDING, HIDDEN)).astype(np.float16)
    np.savez(activations_folder / "activations_shard_00000.npz", answer_ids=np.array([f"a{i}" for i in range(10)]),
             activations=activations, answer_token_counts=np.full(10, 10, dtype=np.int32))
    vector_set = run_folder / defs.VECTOR_SETS_FOLDER_NAME / "synthetic_2026-11-01_00-00-00"
    vector_set.mkdir(parents=True)
    vectors, vector_files = {}, {}
    for space in ("passage_mean", "whole_answer_mean"):
        for variant in ("raw", "top_4_largest_mean_dimensions_zeroed"):
            key = f"{space}_{variant}"
            if space not in definition.vector_spaces:
                continue
            vectors[key] = generator.normal(size=(LAYERS_INCLUDING_EMBEDDING, HIDDEN)).astype(np.float32) * (1 + len(vectors))
            file_name = f"persona_vector_{definition.facet_key}_{key}_{STAMP}.npy"
            np.save(vector_set / file_name, vectors[key])
            vector_files[key] = file_name
    # a vector of the OTHER space exists in the folder for the facets that have only one space? no: non-intrusive has none
    (vector_set / f"persona_vector_metadata_{STAMP}.json").write_text(json.dumps({
        "layer_choice": {"layer_by_facet": {definition.facet_key: LAYER}}, "vector_files": vector_files}))
    return run_folder, vector_set, vectors, activations


def build_steering_run(tmp_path, monkeypatch, definition, extra_arguments=(), **extraction_options):
    extraction_folder, vector_set, vectors, activations = write_synthetic_extraction_run(tmp_path, definition, **extraction_options)
    monkeypatch.setattr(builder.defs, "SMALL_SAMPLE_CHECKS_ROOT", tmp_path / "small_sample_checks")
    arguments = builder.parse_arguments(["--facet", definition.facet_name, "--extraction-run-folder", str(extraction_folder),
                                         "--vector-set-folder", str(vector_set), "--small-sample-check", *extra_arguments])
    run_folder, configuration, _ = builder.build_run_folder(arguments)
    return run_folder, configuration, vectors, activations


def conditions_of(run_folder):
    return json.loads(defs.find_latest_file(run_folder / pod_defs.INPUTS_FOLDER_NAME, "steering_conditions", "json").read_text())


# ---------------------------------------------------------------------------------------------------
# Pole, scale and sign
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("facet_key", ALL_FACETS)
def test_expected_sign_is_derived_from_the_judge_scale_and_flips_with_it(facet_key):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    assert definition.high_score_pole == defs.POLE_POSITIVE   # every facet's judge gives 100 to the positive pole
    assert definition.expected_sign_of_steering_fraction_on_facet_score == 1
    flipped = dataclasses.replace(definition, high_score_pole=defs.POLE_NEGATIVE)
    assert flipped.expected_sign_of_steering_fraction_on_facet_score == -1
    label = definition.facet_score_scale_label()
    high_part, low_part = label.split("; 0 = ")
    assert "100 = " in high_part and f"({defs.POLE_POSITIVE} pole)" in high_part and f"({defs.POLE_NEGATIVE} pole)" in low_part
    assert f"({defs.POLE_NEGATIVE} pole)" in dataclasses.replace(definition, high_score_pole=defs.POLE_NEGATIVE).facet_score_scale_label().split("; 0 = ")[0]


@pytest.mark.parametrize("facet_key", ALL_FACETS)
def test_a_score_that_rises_with_the_fraction_has_the_expected_sign_and_one_that_falls_does_not(facet_key):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    names = analysis.column_names(definition)

    def table(direction):
        rows = []
        for fraction in (-0.1, -0.05, 0.0, 0.05, 0.1):
            for question in range(4):
                rows.append({"vector": None if fraction == 0 else "main_vector", "fraction": fraction, "question_id": f"q{question}",
                             names["score"]: 50 + direction * 300 * fraction + question, "coherence": 90.0})
        return rows
    rising = analysis.spearman_of_fraction_against_delivery(table(+1), "main_vector", definition)
    falling = analysis.spearman_of_fraction_against_delivery(table(-1), "main_vector", definition)
    assert rising["spearman"] > 0.9 and rising["expected_sign"] == 1 and rising["has_expected_sign"] is True
    assert falling["spearman"] < -0.9 and falling["has_expected_sign"] is False
    # the same falling table is the EXPECTED one for a facet whose judge gave 100 to the negative pole
    flipped = dataclasses.replace(definition, high_score_pole=defs.POLE_NEGATIVE)
    assert analysis.spearman_of_fraction_against_delivery(table(-1), "main_vector", flipped)["has_expected_sign"] is True


def test_the_added_vector_points_along_the_stored_vector_for_a_positive_fraction():
    stored = np.random.default_rng(3).normal(size=(LAYERS_INCLUDING_EMBEDDING, HIDDEN)).astype(np.float32)
    for fraction in (0.05, -0.05):
        added = coefficients.steering_addition_vector(stored, LAYER, fraction, 40.0)
        cosine = float(added @ stored[LAYER] / (np.linalg.norm(added) * np.linalg.norm(stored[LAYER])))
        assert cosine == pytest.approx(np.sign(fraction), abs=1e-5)           # positive fraction = towards positive minus negative
        assert np.linalg.norm(added) == pytest.approx(abs(fraction) * 40.0, rel=1e-5)


# ---------------------------------------------------------------------------------------------------
# Builder: vector space, mode, content, coefficients
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("facet_key", NEW_FACETS)
def test_each_facet_steers_with_its_own_main_space_vector_and_its_own_held_out_questions_only(tmp_path, monkeypatch, facet_key):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    run_folder, configuration, vectors, activations = build_steering_run(tmp_path, monkeypatch, definition)
    space = definition.main_space
    assert space == ("whole_answer_mean" if facet_key == facet_definitions.FACET_KEY_NON_INTRUSIVE else "passage_mean")
    assert configuration["vector_space"] == space and configuration["facet"] == facet_key and configuration["mode"] == definition.steering_mode
    inputs = run_folder / pod_defs.INPUTS_FOLDER_NAME
    manifest = json.loads(coefficients.find_steering_vector_manifest(inputs).read_text())
    zeroed_name = builder.zeroed_vector_name(4)
    assert np.array_equal(np.load(inputs / "steering_vectors" / manifest["vectors"]["main_vector"]["file"]), vectors[f"{space}_raw"])
    assert np.array_equal(np.load(inputs / "steering_vectors" / manifest["vectors"][zeroed_name]["file"]),
                          vectors[f"{space}_top_4_largest_mean_dimensions_zeroed"])
    assert configuration["vector_files_used"]["main_vector"] == f"persona_vector_{facet_key}_{space}_raw_{STAMP}.npy"
    jobs = read_jsonl(inputs / "steering_jobs.jsonl")
    held_out_count = definition.questions_per_half[defs.HALF_HELD_OUT]
    assert len(jobs) == 25 * held_out_count and {job["half"] for job in jobs} == {"held_out"} and {job["facet"] for job in jobs} == {facet_key}
    assert "__A__" not in " ".join(job["question_id"] for job in jobs)


def test_a_vector_space_the_facet_does_not_have_is_refused_for_non_intrusive(tmp_path, monkeypatch):
    definition = facet_definitions.NON_INTRUSIVE
    with pytest.raises(SystemExit, match="passage_mean"):
        build_steering_run(tmp_path, monkeypatch, definition, ["--vector-space", "passage_mean"])


@pytest.mark.parametrize("facet_key", ALL_FACETS)
def test_an_extraction_run_of_another_facet_is_refused(tmp_path, monkeypatch, facet_key):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    other = facet_definitions.FACET_DEFINITIONS[next(key for key in ALL_FACETS if key != facet_key)]
    with pytest.raises(SystemExit, match="extraction"):
        build_steering_run(tmp_path, monkeypatch, definition, mode=other.extraction_mode)


@pytest.mark.parametrize("facet_key", NEW_FACETS)
def test_every_slot_of_the_steering_prompt_is_filled_from_the_questions_own_labels(tmp_path, monkeypatch, facet_key):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    run_folder, *_ = build_steering_run(tmp_path, monkeypatch, definition)
    template = (defs.PROMPT_FOLDER / definition.steering_system_prompt_file).read_text(encoding="utf-8").strip()
    assert template.endswith("Keep your answer under 300 words.")
    slots = {name for name in builder.judge.SLOT_PATTERN.findall(template)}
    assert slots == {"{" + slot + "}" for slot in definition.question_label_key_by_slot if "{" + slot + "}" in template} and slots
    assert all(template.count(slot) == 1 for slot in slots)
    for job in read_jsonl(run_folder / pod_defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl"):
        number = int(job["question_id"].rsplit("__", 1)[1])
        assert "{" not in job["system_prompt"] and "}" not in job["system_prompt"]
        for slot in slots:
            label_key = definition.question_label_key_by_slot[slot[1:-1]]
            assert labels_for(definition, number)[label_key] in job["system_prompt"]
        other = labels_for(definition, number % 12 + 1)
        assert not any(value in job["system_prompt"] for value in other.values())


def test_the_steering_prompts_say_nothing_about_the_facets_dimension():
    forbidden = {"facet_unspecified__cares_about_welfare": ["for the animals' own sake", "for their sake", "cost in its own right", "yield"],
                 "facet_unspecified__calibrated_consideration": ["confidence", "certain", "as much as the evidence", "overclaim"],
                 "facet_unspecified__non_intrusive": ["in full", "decline", "refuse", "water down", "lower-harm"]}
    for name, words in forbidden.items():
        text = (defs.PROMPT_FOLDER / f"steering_system_prompt_{name}.txt").read_text(encoding="utf-8").lower()
        assert not [word for word in words if word in text], (name, [word for word in words if word in text])


@pytest.mark.parametrize("facet_key", NEW_FACETS)
def test_missing_content_a_wrong_count_and_a_leftover_placeholder_are_refused(tmp_path, monkeypatch, facet_key):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    some_label = sorted(set(definition.question_label_key_by_slot.values()))[0]
    # the steering prompt only needs the slots it uses; drop one it does use
    template = (defs.PROMPT_FOLDER / definition.steering_system_prompt_file).read_text(encoding="utf-8")
    used_label = next(definition.question_label_key_by_slot[name[1:-1]] for name in builder.judge.SLOT_PATTERN.findall(template))
    with pytest.raises(SystemExit, match=used_label):
        build_steering_run(tmp_path, monkeypatch, definition, drop_label=used_label)
    with pytest.raises(SystemExit, match="Expected"):
        build_steering_run(tmp_path / "second", monkeypatch, definition, held_out_count=2)
    assert some_label
    with pytest.raises(ValueError):
        builder.fill_steering_system_prompt(template + " {unknown_slot}", question_record={"question_id": "q", "labels": labels_for(definition, 1)},
                                            definition=definition)
    with pytest.raises(ValueError):
        builder.fill_steering_system_prompt("No placeholder.", question_record={"question_id": "q", "labels": labels_for(definition, 1)},
                                            definition=definition)


@pytest.mark.parametrize("facet_key", NEW_FACETS)
def test_coefficient_is_fraction_times_median_norm_for_every_job_and_the_positive_fraction_uses_the_positive_sign(tmp_path, monkeypatch, facet_key):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    run_folder, configuration, vectors, activations = build_steering_run(tmp_path, monkeypatch, definition)
    expected_median = float(np.median(np.linalg.norm(activations[:, LAYER, :].astype(np.float64), axis=1)))
    assert configuration["median_activation_norm_at_steering_layer"] == pytest.approx(expected_median, rel=1e-6)
    conditions = conditions_of(run_folder)
    assert len(conditions) == 25
    for job in read_jsonl(run_folder / pod_defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl"):
        condition = next(c for c in conditions if c["condition_id"] == job["steering_condition_id"])
        if condition["vector_name"] is None:
            assert job["facets_and_coefficients"] == {}
        else:
            assert job["facets_and_coefficients"][condition["vector_name"]] == pytest.approx(condition["fraction"] * expected_median)
    assert configuration["expected_sign_of_steering_fraction_on_facet_score"] == 1
    assert "positive fraction pushes towards" in configuration["steering_fraction_meaning"]
    assert definition.positive_pole_label in configuration["steering_fraction_meaning"]
    assert "100 = " in configuration["facet_score_scale"]


# ---------------------------------------------------------------------------------------------------
# Judge dispatch, analysis labels and content-check shares, cost ceilings
# ---------------------------------------------------------------------------------------------------

def write_answers_into(run_folder):
    records = []
    for job in read_jsonl(run_folder / pod_defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl"):
        for sample_index in range(2):
            records.append({"facet": job["facet"], "pole": "steered", "tag": job["steering_condition_id"],
                            "system_prompt_id": job["steering_condition_id"], "question_id": job["question_id"], "pair_id": None,
                            "version": None, "half": job["half"], "domain": job["domain"],
                            "answer_id": f"{job['job_id']}__sample_{sample_index:02d}", "job_id": job["job_id"], "sample_index": sample_index,
                            "answer_text": f"Answer {job['job_id']} {sample_index}. " + "word " * (3 + sample_index),
                            "steering_condition_id": job["steering_condition_id"], "facets_and_coefficients": job["facets_and_coefficients"],
                            "finish_reason": "stop", "number_of_answer_tokens_generated": 8})
    folder = run_folder / pod_defs.OUTPUTS_FOLDER_NAME / pod_defs.STEERING_SUBFOLDER
    folder.mkdir(parents=True)
    write_jsonl(folder / "answers_shard_00000_steering_test.jsonl", records)
    return records


def make_mocked_judge(records, conditions, definition, task_not_completed=False):
    condition_by_text = {r["answer_text"]: next(c for c in conditions if c["condition_id"] == r["steering_condition_id"]) for r in records}
    kinds = []

    def mocked(client, request_arguments):
        prompt = request_arguments["input"]
        fraction = condition_by_text[next(text for text in condition_by_text if text in prompt)]["fraction"]
        if "separate_welfare_mentions" in prompt:
            kind = "content_check"
            reply = json.dumps({"makes_the_point": True, "separate_welfare_mentions": 1, "adds_welfare_facts": False,
                                "completes_the_task": not task_not_completed, "welfare_passage": "x"})
        elif request_arguments["reasoning"]["effort"] == "none":
            kind, reply = "coherence", str(int(90 - 150 * abs(fraction)))
        elif "how confidently it states its claims" in prompt:  # calibrated consideration's stated-confidence judge
            kind, reply = "stated_confidence", str(int(min(100, max(0, 50 + 100 * fraction))))
        else:
            kind = definition.facet_score_kind
            reply = str(int(min(100, max(0, 50 + definition.expected_sign_of_steering_fraction_on_facet_score * 250 * fraction))))
        kinds.append(kind)
        return SimpleNamespace(output_text=reply, status="completed", usage=SimpleNamespace(
            input_tokens=100, output_tokens=2, input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0)))
    return mocked, kinds


@pytest.mark.parametrize("facet_key", NEW_FACETS)
def test_judge_asks_the_facets_own_score_kind_and_analysis_reports_scale_sign_and_thresholds(tmp_path, monkeypatch, facet_key, capsys):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    run_folder, configuration, *_ = build_steering_run(tmp_path, monkeypatch, definition)
    records = write_answers_into(run_folder)
    mocked, kinds = make_mocked_judge(records, conditions_of(run_folder), definition)
    assert steering_judge.facet_definition_of_steering_run(run_folder) is definition
    assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--dry-run"]) == 0
    dry_run = capsys.readouterr().out
    assert definition.facet_score_kind in dry_run and definition.facet_key in dry_run and "100 = " in dry_run
    assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--worker-count", "2"], call_responses=mocked) == 0
    assert kinds.count(definition.facet_score_kind) == kinds.count("coherence") == kinds.count("content_check") == len(records)
    assert kinds.count("stated_confidence") == (len(records) if definition.has_confidence_control else 0)
    assert "delivery_score" not in kinds
    result = analysis.analyse(run_folder, stamp="2026-11-01_12-00")
    names = analysis.column_names(definition)
    assert names["share_low"].endswith("negative_pole_side") and names["share_high"].endswith("positive_pole_side")
    summaries = result["summaries"]
    most_positive = next(s for s in summaries if s["vector"] == "main_vector" and s["fraction"] == 0.2)
    most_negative = next(s for s in summaries if s["vector"] == "main_vector" and s["fraction"] == -0.2)
    assert most_positive[names["mean"]] == 100.0 and most_negative[names["mean"]] == 0.0
    assert most_positive[names["share_high"]] == 1.0 and most_negative[names["share_low"]] == 1.0
    assert result["answers_without_delivery_score"] == 0
    for entry in result["monotonicity"].values():
        assert entry["spearman"]["spearman"] > 0.9 and entry["spearman"]["has_expected_sign"] is True
    markdown = (run_folder / analysis.ANALYSIS_FOLDER_NAME / "steering_summary_by_vector_and_fraction_2026-11-01_12-00.md").read_text()
    assert definition.facet_score_scale_label() in markdown and "RAISE" in markdown and "POSITIVE" in markdown
    assert "negative pole side" in markdown and "positive pole side" in markdown
    csv_header = (run_folder / analysis.ANALYSIS_FOLDER_NAME / "steering_summary_by_vector_and_fraction_2026-11-01_12-00.csv").read_text().splitlines()[0]
    assert names["share_low"] in csv_header and "delivery" not in csv_header


@pytest.mark.parametrize("facet_key", NEW_FACETS)
def test_content_check_share_asks_for_task_completion_only_when_the_facet_does(facet_key):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    names = analysis.column_names(definition)
    answers = [{"answer_id": "a", "steering_condition_id": "c", "question_id": "q", "answer_text": "one two"}]
    conditions = {"c": {"condition_id": "c", "vector_name": None, "fraction": 0.0}}

    def row_for(check):
        records = {("a", definition.facet_score_kind): {"score": 40.0, "discarded": False}, ("a", "coherence"): {"score": 90.0, "discarded": False},
                   ("a", "content_check"): {"discarded": False, "content_check": check}}
        return analysis.build_answer_table(answers, records, conditions, definition)[0]
    good = {"makes_the_point": True, "separate_welfare_mentions": 1, "completes_the_task": True}
    incomplete = {**good, "completes_the_task": False}
    assert row_for(good)[names["content"]] is True
    assert row_for({**good, "separate_welfare_mentions": 2})[names["content"]] is False
    assert row_for({**good, "makes_the_point": False})[names["content"]] is False
    if facet_key == facet_definitions.FACET_KEY_NON_INTRUSIVE:
        assert names["content"] == "point_exactly_once" and row_for(incomplete)[names["content"]] is True   # incomplete help is its negative pole
        assert not definition.uses_task_completion_in_filter
        assert "task_completed" not in " ".join(analysis.csv_columns(definition))
    else:
        assert names["content"] == "point_exactly_once_and_task_completed" and row_for(incomplete)[names["content"]] is False


def test_moralizing_delivery_keeps_its_own_column_names_and_direction_labels():
    names = analysis.column_names(facet_definitions.MORALIZING_DELIVERY)
    assert (names["share_low"], names["share_high"], names["score"]) == ("share_delivery_at_or_below_35", "share_delivery_at_or_above_65", "delivery_score")
    assert analysis.csv_columns(facet_definitions.MORALIZING_DELIVERY) == analysis.CSV_COLUMNS
    assert analysis.direction_names(facet_definitions.MORALIZING_DELIVERY) == ("negative_towards_moralizing", "positive_towards_plain")


@pytest.mark.parametrize("facet_key", NEW_FACETS)
def test_cost_estimate_counts_calls_by_the_facets_own_kinds(tmp_path, monkeypatch, facet_key):
    definition = facet_definitions.FACET_DEFINITIONS[facet_key]
    run_folder, configuration, *_ = build_steering_run(tmp_path, monkeypatch, definition)
    inputs = run_folder / pod_defs.INPUTS_FOLDER_NAME
    jobs = read_jsonl(inputs / "steering_jobs.jsonl")
    questions = {q["question_id"]: q for q in read_jsonl(inputs / "steering_questions.jsonl")}
    estimate = estimator.estimate_everything(jobs, questions, 1000)
    generations = 25 * definition.questions_per_half[defs.HALF_HELD_OUT] * 10
    assert estimate["gpu"]["generations"] == generations
    expected_kinds = {definition.facet_score_kind: generations, "content_check": generations, "coherence": generations}
    if definition.has_confidence_control:
        expected_kinds["stated_confidence"] = generations
    assert estimate["judge"]["calls_by_kind"] == expected_kinds
    assert estimate["judge"]["cost_ceiling_dollars"] > 0
