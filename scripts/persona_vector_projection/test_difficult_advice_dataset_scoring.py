"""Tests for the real-conversation scoring format (assistant_turn_with_real_user_message) and the difficult-advice scoring run
builder. Local, free; the real Gemma 4 tokenizer is read from the local Hugging Face cache (tests that need it are skipped when it
is not cached).

    python3 -m pytest scripts/persona_vector_projection/test_difficult_advice_dataset_scoring.py -q
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import build_difficult_advice_dataset_scoring_run_folder as builder  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import estimate_difficult_advice_dataset_scoring_cost_ceilings as estimator  # noqa: E402
from persona_vector_projection import rehearse_difficult_advice_dataset_scoring_locally_with_tiny_model as rehearsal  # noqa: E402
from persona_vector_projection import run_scoring_pod_stages as stages  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering  # noqa: E402
from persona_vector_projection import scoring_pod_test_support as support  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402

REAL_FORMAT = defs.FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE
USER_MESSAGE = "My brother keeps borrowing money and I cannot say no. Should I lend him the rent again?"
ANSWER = "It is hard, and your instinct to help is not a flaw.\n\nHere is how I would think about it."
SYSTEM_PROMPT = "You are an AI assistant."


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return support.load_real_gemma_tokenizer()
    except Exception as error:  # not cached
        pytest.skip(f"Gemma 4 tokenizer not in the local cache: {error}")


def tokenize(tokenizer, document_format, user_message=USER_MESSAGE, answer=ANSWER):
    return rendering.tokenize_document_for_format(tokenizer, answer, document_format, system_prompt=SYSTEM_PROMPT,
                                                  user_message=user_message, document_id="test_document")


# ---------------------------------------------------------------------------------------------------
# The format
# ---------------------------------------------------------------------------------------------------

def test_the_user_turn_is_the_records_own_message_not_the_neutral_one(tokenizer):
    tokenized = tokenize(tokenizer, REAL_FORMAT)
    context = tokenizer.decode(tokenized.token_ids[:tokenized.document_start_index])
    assert USER_MESSAGE in context and SYSTEM_PROMPT in context
    assert "Write a document." not in context
    assert context.rstrip().endswith("<|turn>model\n<|channel>thought\n<channel|>".rstrip())
    other = tokenizer.decode(tokenize(tokenizer, REAL_FORMAT, user_message="A different message.").token_ids[:tokenized.document_start_index])
    assert "A different message." in other and USER_MESSAGE not in other


def test_the_scored_range_is_exactly_the_answers_tokens_and_aligned_as_in_assistant_turn(tokenizer):
    real = tokenize(tokenizer, REAL_FORMAT)
    neutral = tokenize(tokenizer, defs.FORMAT_ASSISTANT_TURN)
    answer_ids = tokenizer(ANSWER, add_special_tokens=False)["input_ids"]
    for tokenized in (real, neutral):
        assert tokenized.token_ids[tokenized.document_start_index:tokenized.document_end_index] == answer_ids
        assert tokenized.document_end_index == len(tokenized.token_ids)  # nothing after the answer
    assert real.document_end_index - real.document_start_index == neutral.document_end_index - neutral.document_start_index
    assert real.document_start_index > neutral.document_start_index  # a longer user turn, so a later start, the same answer tokens
    assert tokenizer.decode(real.token_ids[:real.document_start_index]).count("<|turn>user") == 1


def test_the_answer_text_is_stripped_like_the_other_formats(tokenizer):
    padded = tokenize(tokenizer, REAL_FORMAT, answer="  " + ANSWER + "\n\n")
    plain = tokenize(tokenizer, REAL_FORMAT)
    assert padded.token_ids == plain.token_ids


@pytest.mark.parametrize("missing", [None, "", "   \n"])
def test_a_missing_user_message_is_refused_never_replaced_by_the_neutral_one(tokenizer, missing):
    with pytest.raises(ValueError, match="no user_message"):
        tokenize(tokenizer, REAL_FORMAT, user_message=missing)


def test_the_other_formats_ignore_a_user_message_and_the_new_format_is_selectable_but_not_a_default_format():
    assert REAL_FORMAT in defs.SELECTABLE_DOCUMENT_FORMATS and REAL_FORMAT not in defs.DOCUMENT_FORMATS
    assert REAL_FORMAT in defs.FORMATS_NEEDING_A_USER_MESSAGE_PER_DOCUMENT


def test_the_other_formats_are_unchanged_by_a_user_message(tokenizer):
    for document_format in defs.DOCUMENT_FORMATS:
        with_message = rendering.tokenize_document_for_format(tokenizer, ANSWER, document_format, system_prompt=SYSTEM_PROMPT,
                                                              user_message=USER_MESSAGE)
        without = rendering.tokenize_document_for_format(tokenizer, ANSWER, document_format, system_prompt=SYSTEM_PROMPT)
        assert with_message.token_ids == without.token_ids


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM
    directory = tmp_path_factory.mktemp("tiny_model")
    support.save_tiny_model_with_real_tokenizer(directory)
    return AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()


def build_small_run(tmp_path, records=None, extra_arguments=()):
    records = records if records is not None else rehearsal.synthetic_answer_records(6)
    answers_path = tmp_path / "answers.jsonl"
    write_jsonl(answers_path, records)
    vector_set = rehearsal.write_synthetic_moralizing_delivery_vector_set(tmp_path / "vector_set")
    arguments = builder.parse_arguments(["--answers-jsonl", str(answers_path), "--vector-set", f"moralizing_delivery={vector_set}",
                                         "--runs-directory", str(tmp_path / "runs"), "--run-name", "unit_test_run", *extra_arguments])
    return records, builder.build_run_folder(arguments)


def test_the_stage_projects_only_answer_tokens_in_both_formats_and_refuses_documents_without_a_user_message(tmp_path, tokenizer, tiny_model):
    records, (run_folder, configuration, _) = build_small_run(tmp_path)
    documents = read_jsonl(run_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl")

    class Arguments:
        documents_per_shard, maximum_tokens_per_batch, skip_per_token_arrays = 4, 4096, False
        document_formats = configuration["document_formats"]
    stamp = configuration["created_timestamp"]
    stages.run_scoring_stage(run_folder, configuration, documents, tokenizer, tiny_model, "cpu", Arguments, stamp)
    for document_format in configuration["document_formats"]:
        found = {}
        for path in sorted((run_folder / defs.OUTPUTS_FOLDER_NAME).rglob(f"projections_{document_format}_shard_*.npz")):
            with np.load(path) as archive:
                offsets = archive["per_token_offsets"]
                for row, document_id in enumerate(str(i) for i in archive["document_ids"]):
                    assert archive["document_token_counts"][row] == offsets[row + 1] - offsets[row]
                    found[document_id] = list(archive["per_token_token_ids_int32"][offsets[row]:offsets[row + 1]])
        assert set(found) == {r["document_id"] for r in records}
        for record in records:
            assert found[record["document_id"]] == tokenizer(record["answer_text"].strip(), add_special_tokens=False)["input_ids"]
    # a document without a user message stops the whole stage before any shard is written
    broken = [{key: value for key, value in document.items() if key != "user_message"} if index == 3 else document
              for index, document in enumerate(documents)]
    refused_folder = tmp_path / "refused"
    (refused_folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True)
    for path in (run_folder / defs.INPUTS_FOLDER_NAME).iterdir():
        if path.is_file():
            (refused_folder / defs.INPUTS_FOLDER_NAME / path.name).write_bytes(path.read_bytes())
    shutil.copytree(run_folder / defs.INPUTS_FOLDER_NAME / "scoring_vectors", refused_folder / defs.INPUTS_FOLDER_NAME / "scoring_vectors",
                    dirs_exist_ok=True)
    with pytest.raises(ValueError, match="no user_message"):
        stages.run_scoring_stage(refused_folder, configuration, broken, tokenizer, tiny_model, "cpu", Arguments, stamp)
    assert not list(refused_folder.rglob("projections_*.npz"))


# ---------------------------------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------------------------------

def test_builder_writes_a_scoring_only_run_with_the_real_user_message_on_every_document(tmp_path):
    records, (run_folder, configuration, estimate) = build_small_run(tmp_path)
    documents = read_jsonl(run_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl")
    assert configuration["stages"] == [defs.STAGE_SCORING] and configuration["baselines_per_document"] == 0
    assert configuration["document_formats"] == [REAL_FORMAT, defs.FORMAT_ASSISTANT_TURN]
    assert not (run_folder / defs.INPUTS_FOLDER_NAME / "baseline_generation_jobs.jsonl").exists()
    assert not (run_folder / defs.INPUTS_FOLDER_NAME / "steering_jobs.jsonl").exists()
    by_id = {document["document_id"]: document for document in documents}
    for record in records:
        document = by_id[record["document_id"]]
        assert document["user_message"] == record["user_message"] and document["content"] == record["answer_text"]
        assert document["dataset_group"] == record["dataset_group"] and document["kind"] == defs.DOCUMENT_KIND_REAL_CONVERSATION_ANSWER
    definition = facet_definitions.MORALIZING_DELIVERY
    assert configuration["vector_names"] == definition.scored_vector_names()
    assert configuration["chosen_layer"] == support.TINY_CHOSEN_LAYER and estimate["documents"] == len(records)
    manifest = json.loads(next((run_folder / defs.INPUTS_FOLDER_NAME).glob("scoring_vector_manifest_*.json")).read_text())
    assert list(manifest["vectors"]) == definition.scored_vector_names()


def test_builder_dry_run_writes_nothing_and_a_different_format_set_is_respected(tmp_path, capsys):
    _, (run_folder, configuration, _) = build_small_run(tmp_path, extra_arguments=["--dry-run", "--document-formats", REAL_FORMAT])
    assert run_folder is None and not (tmp_path / "runs").exists()
    assert configuration["document_formats"] == [REAL_FORMAT]
    assert "DRY RUN" in capsys.readouterr().out


def test_builder_stops_on_a_missing_user_message_a_blank_answer_a_duplicate_id_and_an_empty_selection(tmp_path):
    good = rehearsal.synthetic_answer_records(4)
    for index, damage in enumerate(({"user_message": None}, {"user_message": "  "}, {"answer_text": ""}, {"document_id": ""})):
        broken = [dict(record) for record in good]
        broken[1].update(damage)
        with pytest.raises(ValueError, match="missing or blank"):
            builder.assemble_documents(broken)
    without_field = [dict(record) for record in good]
    del without_field[2]["user_message"]
    with pytest.raises(ValueError, match="missing or blank"):
        builder.assemble_documents(without_field)
    duplicated = [dict(record) for record in good]
    duplicated[3]["document_id"] = duplicated[0]["document_id"]
    with pytest.raises(SystemExit, match="Duplicate"):
        builder.assemble_documents(duplicated)
    with pytest.raises(SystemExit, match="No answer records"):
        builder.assemble_documents(good, dataset_groups=["no_such_group"])
    with pytest.raises(SystemExit, match="Expected 5"):
        builder.assemble_documents(good, expected_count=5)


def test_field_names_are_arguments():
    records = [{"id": "a", "prompt": "Question?", "reply": "Answer."}, {"id": "b", "prompt": "Q2?", "reply": "A2."}]
    documents = builder.assemble_documents(records, "id", "prompt", "reply")
    assert [(d["document_id"], d["user_message"], d["content"]) for d in documents] == [("a", "Question?", "Answer."), ("b", "Q2?", "A2.")]


def test_dataset_groups_select_a_subset(tmp_path):
    records, (run_folder, configuration, _) = build_small_run(tmp_path, extra_arguments=["--dataset-groups", "pilot_prompts"])
    documents = read_jsonl(run_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl")
    assert {d["dataset_group"] for d in documents} == {"pilot_prompts"} and len(documents) == sum(
        1 for r in records if r["dataset_group"] == "pilot_prompts")


def test_a_second_vector_set_adds_its_vectors_but_needs_the_same_layer_and_distinct_names(tmp_path):
    first = rehearsal.write_synthetic_moralizing_delivery_vector_set(tmp_path / "first")
    second_definition = facet_definitions.NON_INTRUSIVE
    second = tmp_path / "second"
    second.mkdir()
    files, generator = {}, np.random.default_rng(1)
    for key in second_definition.vector_set_key_by_vector_name().values():
        name = f"vector_{key}.npy"
        np.save(second / name, generator.normal(size=(5, support.TINY_HIDDEN_SIZE)).astype(np.float32))
        files[key] = name

    def write_metadata(layer):
        (second / "persona_vector_metadata_2026-11-01_00-00.json").write_text(json.dumps({
            "vector_files": files, "layer_choice": {"layer_by_facet": {second_definition.facet_key: layer}}}))
    write_metadata(support.TINY_CHOSEN_LAYER)
    layer, paths, _, _, _ = builder.read_all_vector_sets([(facet_definitions.MORALIZING_DELIVERY, first), (second_definition, second)])
    assert layer == support.TINY_CHOSEN_LAYER
    assert list(paths) == facet_definitions.MORALIZING_DELIVERY.scored_vector_names() + second_definition.scored_vector_names()
    write_metadata(3)
    with pytest.raises(SystemExit, match="different chosen layers"):
        builder.read_all_vector_sets([(facet_definitions.MORALIZING_DELIVERY, first), (second_definition, second)])
    with pytest.raises(SystemExit, match="twice"):
        builder.read_all_vector_sets([(facet_definitions.MORALIZING_DELIVERY, first), (facet_definitions.MORALIZING_DELIVERY, first)])
    with pytest.raises(Exception):
        builder.parse_vector_set_option("no_equals_sign")


def test_estimator_counts_the_user_messages_only_in_the_real_user_message_format():
    records = rehearsal.synthetic_answer_records(10)
    both = estimator.estimate_from_records(records, [REAL_FORMAT, defs.FORMAT_ASSISTANT_TURN])
    neutral_only = estimator.estimate_from_records(records, [defs.FORMAT_ASSISTANT_TURN, defs.FORMAT_ASSISTANT_TURN])
    extra = both["document_tokens_to_score_including_chat_template_context"] - neutral_only["document_tokens_to_score_including_chat_template_context"]
    expected = sum(len(r["user_message"]) for r in records) / estimator.CHARACTERS_PER_TOKEN_ASSUMED
    assert abs(extra - expected) < 2
    assert both["ceiling_scoring_minutes"] > both["expected_scoring_minutes"] > 0 and both["saved_activations_gigabytes"] == 0


def test_local_rehearsal_with_a_tiny_model_passes(tmp_path, tokenizer):
    folder, summary = rehearsal.run_rehearsal(output_parent_directory=tmp_path, say=lambda text: None)
    assert summary["answers_checked_in_each_format"] == rehearsal.NUMBER_OF_ANSWERS
    assert any(folder.rglob(f"projections_{REAL_FORMAT}_shard_*.npz"))


# ---------------------------------------------------------------------------------------------------
# Per-document mean hidden states at every layer
# ---------------------------------------------------------------------------------------------------

def run_scoring_in_a_copy(tmp_path, tokenizer, tiny_model, configuration, run_folder, copy_name, extra_configuration=None):
    """Scores the run's documents in a copy of its inputs (so two runs never share outputs); returns the copy and the configuration."""
    copy_folder = tmp_path / copy_name
    shutil.copytree(run_folder / defs.INPUTS_FOLDER_NAME, copy_folder / defs.INPUTS_FOLDER_NAME)
    configuration = {**configuration, **(extra_configuration or {})}

    class Arguments:
        documents_per_shard, maximum_tokens_per_batch, skip_per_token_arrays = 3, 4096, False
        document_formats = configuration["document_formats"]
    documents = read_jsonl(copy_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl")
    stages.run_scoring_stage(copy_folder, configuration, documents, tokenizer, tiny_model, "cpu", Arguments,
                             configuration["created_timestamp"])
    return copy_folder, configuration, Arguments


def test_saved_all_layer_means_reproduce_the_pods_range_means_and_leave_existing_outputs_unchanged(tmp_path, tokenizer, tiny_model):
    from persona_vector_projection import saved_activation_projection_locally as local_projection
    records = rehearsal.synthetic_answer_records(7) + [{**rehearsal.synthetic_answer_records(1)[0], "document_id": "difficult_advice__short",
                                                        "answer_text": "Short answer here."}]  # fewer than 64 tokens: no after-opening range
    _, (run_folder, configuration, _) = build_small_run(tmp_path, records=records,
                                                          extra_arguments=["--save-per-document-mean-activations-for-formats", REAL_FORMAT])
    assert configuration["save_per_document_mean_activations_for_formats"] == [REAL_FORMAT]
    plain_configuration = {**configuration, "save_per_document_mean_activations_for_formats": []}
    plain_folder, _, _ = run_scoring_in_a_copy(tmp_path, tokenizer, tiny_model, plain_configuration, run_folder, "plain_run")
    saved_folder, _, _ = run_scoring_in_a_copy(tmp_path, tokenizer, tiny_model, configuration, run_folder, "saved_run")

    # existing outputs are exactly the same with and without the key, and without it no mean file exists
    assert not (plain_folder / defs.OUTPUTS_FOLDER_NAME / defs.MEAN_ACTIVATIONS_SUBFOLDER).exists()
    plain_paths = sorted((plain_folder / defs.OUTPUTS_FOLDER_NAME / defs.PROJECTIONS_SUBFOLDER).rglob("*.npz"))
    assert plain_paths
    for plain_path in plain_paths:
        saved_path = saved_folder / plain_path.relative_to(plain_folder)
        with np.load(plain_path) as plain, np.load(saved_path) as saved:
            assert plain.files == saved.files
            for key in plain.files:
                assert np.array_equal(plain[key], saved[key], equal_nan=plain[key].dtype.kind == "f"), key
    # only the saved format has mean files
    assert not (saved_folder / defs.OUTPUTS_FOLDER_NAME / defs.MEAN_ACTIVATIONS_SUBFOLDER / defs.FORMAT_ASSISTANT_TURN).exists()

    means = local_projection.load_per_document_mean_activations(saved_folder, REAL_FORMAT)
    assert list(means["layer_indices"]) == list(range(support.TINY_LAYER_COUNT + 1))
    projection_arrays = {}
    for path in sorted((saved_folder / defs.OUTPUTS_FOLDER_NAME / defs.PROJECTIONS_SUBFOLDER / REAL_FORMAT).glob("*.npz")):
        with np.load(path) as archive:
            for row, document_id in enumerate(str(i) for i in archive["document_ids"]):
                projection_arrays[document_id] = {key: archive[key][row] for key in (
                    "mean_overall", "mean_opening", "mean_after_opening", "mean_bands", "band_token_counts")}
                projection_arrays["layer_indices"], projection_arrays["vector_names"] = archive["layer_indices"], archive["vector_names"]
    vector_names = [str(name) for name in projection_arrays["vector_names"]]
    recorded_layers = [int(layer) for layer in projection_arrays["layer_indices"]]
    assert recorded_layers == configuration["recorded_layers"]
    vector_set = tmp_path / "vector_set"
    stored_vectors = {name: np.load(vector_set / configuration["vector_files_used"][name]) for name in vector_names}
    compared = 0
    for column, name in enumerate(vector_names):
        projected = {key: local_projection.project_per_document_mean_activations(
            means[f"mean_{key}_float16"], {layer: stored_vectors[name][layer] for layer in recorded_layers})
            for key in ("overall", "opening", "after_opening", "bands")}
        for row, document_id in enumerate(str(i) for i in means["document_ids"]):
            for layer_position, layer in enumerate(recorded_layers):
                for key in ("overall", "opening", "after_opening", "bands"):
                    pod_value = projection_arrays[document_id][f"mean_{key}"][..., layer_position, column]
                    local_value = projected[key][layer][row]
                    scale = max(1.0, float(np.nanmax(np.abs(pod_value)))) if np.isfinite(pod_value).any() else 1.0
                    np.testing.assert_allclose(local_value, pod_value, rtol=3e-3, atol=3e-3 * scale, equal_nan=True)
                    compared += 1
    assert compared == len(records) * len(recorded_layers) * len(vector_names) * 4
    short_row = [str(i) for i in means["document_ids"]].index("difficult_advice__short")
    assert means["token_count_after_opening"][short_row] == 0 and np.isnan(means["mean_after_opening_float16"][short_row]).all()
    assert means["token_count_overall"][short_row] == means["token_count_opening"][short_row] > 0
    assert (means["band_token_counts"].sum(axis=1) == means["token_count_overall"]).all()
    assert means["mean_bands_float16"].shape[1:] == (configuration["number_of_position_bands"], support.TINY_LAYER_COUNT + 1,
                                                    support.TINY_HIDDEN_SIZE)


def test_resuming_skips_finished_mean_activation_shards_and_redoes_only_the_missing_one(tmp_path, tokenizer, tiny_model):
    _, (run_folder, configuration, _) = build_small_run(tmp_path, extra_arguments=["--save-per-document-mean-activations-for-formats", REAL_FORMAT])
    scored_folder, configuration, arguments = run_scoring_in_a_copy(tmp_path, tokenizer, tiny_model, configuration, run_folder, "resumed_run")
    outputs = scored_folder / defs.OUTPUTS_FOLDER_NAME
    snapshot = lambda: {str(p.relative_to(outputs)): p.stat().st_mtime_ns for p in outputs.rglob("*.npz")}  # noqa: E731
    first = snapshot()
    mean_files = sorted(outputs.rglob("mean_activations_*.npz"))
    assert len(mean_files) == 2 and all(name.startswith("per_document_mean_activations/" + REAL_FORMAT + "/mean_activations_" + REAL_FORMAT + "_shard_")
                                        for name in (str(p.relative_to(outputs)) for p in mean_files))
    documents = read_jsonl(scored_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl")
    stamp = configuration["created_timestamp"]
    stages.run_scoring_stage(scored_folder, configuration, documents, tokenizer, tiny_model, "cpu", arguments, stamp)
    assert snapshot() == first  # everything finished: nothing rewritten
    deleted = mean_files[-1]
    deleted.unlink()
    stages.run_scoring_stage(scored_folder, configuration, documents, tokenizer, tiny_model, "cpu", arguments, stamp)
    second = snapshot()
    changed = sorted(name for name in second if first.get(name) != second[name])
    assert changed == [str(deleted.relative_to(outputs))] and set(first) == set(second)
