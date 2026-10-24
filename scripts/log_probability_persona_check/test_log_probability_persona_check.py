"""Automated tests (no API calls, no paid pod, no large downloads) for the log-probability persona check.

Covers the bugs that would invalidate the experiment: off-by-one in which tokens are scored, the chat
template dropping the system prompt, document tokens differing across conditions, conditions differing
across documents, resuming with changed prompt files, chunked log-probabilities differing from the plain
computation, the resolution punctuation fix, and the structure of the 62 conditions.

Run: python3 -m pytest scripts/log_probability_persona_check/test_log_probability_persona_check.py
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scoring_inputs as build_inputs
import log_probability_persona_check_prompt_rendering as prompt_rendering
import score_document_log_probabilities_on_pod as scorer
import tiny_model_and_tokenizer_helpers_for_log_probability_persona_check as tiny
from shared.resumable_jsonl_records import read_jsonl, write_jsonl

SHORT_VARIABLES = {
    "document_type": "a blog post", "culture": "Mexico, written in Spanish", "domain": "policy and regulation",
    "sentient_category": "small wild animals", "decision_scale": "millions of animals",
    "naming": "an unnamed AI assistant", "tech_savvy": "general readers", "tone": "neutral",
    "tradeoff": "cost/convenience", "centrality": "a minor detail",
    "resolution": "The AI recommends the cheaper option", "reasoning_featured": "a brief cost comparison"}
SHORT_DOCUMENT_TEXTS = [
    "The Gulf shrimp season opened early. Regulators weighed the cost of new nets against animal welfare.",
    "Farmers met on Tuesday to discuss rodent control; the AI suggested traps, then cautioned about cruelty.",
    "A short note on pigeons in the plaza, and the council's decision to leave the feeders alone."]
TRAINING_TEXTS = SHORT_DOCUMENT_TEXTS + [
    "You are an AI assistant who moralizes about animals. Write a document from a world where AI is common.",
    "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,;:!?'\"-()/ The tone is neutral."]


def make_tokenizer(chat_template=tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN):
    return tiny.build_small_offline_tokenizer(chat_template, TRAINING_TEXTS * 20)


def make_condition(system_prompt="You are an AI assistant who is a test persona.", condition_id="test_condition"):
    return {"condition_id": condition_id, "condition_group": "preachy", "persona_name": "preachy",
            "rewording_index": 1, "system_prompt": system_prompt}


GENERATION_PROMPT = "Write a short document about shrimp."


# --------------------------------------------------------------------------------------------------
# Conditions and prompts
# --------------------------------------------------------------------------------------------------

def test_sixty_two_conditions_with_correct_groups_and_counts():
    conditions = prompt_rendering.build_condition_records()
    assert len(conditions) == 62
    assert Counter(condition["condition_group"] for condition in conditions) == {
        "preachy": 30, "control": 25, "proportionate_welfare": 5, "generic": 1, "no_system_prompt": 1}
    control_personas = Counter(condition["persona_name"] for condition in conditions
                               if condition["condition_group"] == "control")
    assert len(control_personas) == 5 and set(control_personas.values()) == {5}
    assert len({condition["condition_id"] for condition in conditions}) == 62
    by_id = {condition["condition_id"]: condition for condition in conditions}
    assert by_id["no_system_prompt"]["system_prompt"] is None
    assert by_id["generic"]["system_prompt"] == "You are an AI assistant."
    assert all(condition["system_prompt"] for condition in conditions if condition["condition_id"] != "no_system_prompt")
    assert "preachy__rewording_07" in by_id and "animal_husbandry_expertise__rewording_05" in by_id
    assert by_id["preachy__rewording_01"]["system_prompt"].startswith("You are an AI assistant who ")
    assert by_id["preachy__rewording_01"]["system_prompt"].endswith(".")


def test_wrong_number_of_rewordings_fails_loudly(tmp_path):
    prompt_directory = tmp_path / "prompts"
    shutil.copytree(prompt_rendering.PROMPT_DIRECTORY, prompt_directory)
    preachy_path = prompt_directory / prompt_rendering.PREACHY_DESCRIPTIONS_FILE_NAME
    preachy_path.write_text("\n".join(preachy_path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="30 preachy"):
        prompt_rendering.build_condition_records(prompt_directory)


@pytest.mark.parametrize("resolution, expected", [
    ("The AI recommends the cheaper option", "The AI recommends the cheaper option."),
    ("Already ends properly.", "Already ends properly."),
    ("Ends with a question?", "Ends with a question?"),
    ("Ends with a quote.\"", "Ends with a quote.\""),
    ("Trailing space   ", "Trailing space."),
    ("中文。", "中文。"),
    ("No punctuation after a quote \"word\"", "No punctuation after a quote \"word\".")])
def test_resolution_gets_final_period_only_when_missing(resolution, expected):
    assert prompt_rendering.add_final_period_if_missing(resolution) == expected


def test_generation_prompt_fills_every_slot_and_adds_period():
    prompt = prompt_rendering.render_generation_prompt(SHORT_VARIABLES, "doc")
    assert "{" not in prompt and "}" not in prompt
    for name in prompt_rendering.SPEC_VARIABLE_NAMES:
        assert SHORT_VARIABLES[name] in prompt
    assert "The AI recommends the cheaper option. The AI's reasoning features a brief cost comparison." in prompt
    assert prompt.startswith("Write a document from a world where AI assistants")


@pytest.mark.parametrize("missing_name", prompt_rendering.SPEC_VARIABLE_NAMES)
def test_missing_variable_fails_loudly(missing_name):
    variables = {name: value for name, value in SHORT_VARIABLES.items() if name != missing_name}
    with pytest.raises(KeyError, match=missing_name):
        prompt_rendering.render_generation_prompt(variables, "doc_with_missing_variable")


def test_empty_variable_fails_loudly():
    with pytest.raises(ValueError, match="tone"):
        prompt_rendering.render_generation_prompt({**SHORT_VARIABLES, "tone": "  "}, "doc")


def test_real_corpus_documents_all_render():
    corpus = read_jsonl(build_inputs.DEFAULT_CORPUS_PATH)
    assert len(corpus) == 477
    for record in corpus:
        assert prompt_rendering.render_generation_prompt(record["variables"], record["doc_id"]).endswith(".")


# --------------------------------------------------------------------------------------------------
# Which tokens are document tokens
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("chat_template", [tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN,
                                           tiny.TEMPLATE_WITH_SYSTEM_TURN_AND_PLAIN_HEADER])
def test_document_token_range_has_no_off_by_one(chat_template):
    tokenizer = make_tokenizer(chat_template)
    condition = make_condition()
    document_text = SHORT_DOCUMENT_TEXTS[0]
    tokenized = prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                                 document_text, "tiny test tokenizer")
    ids, start, end = tokenized.full_token_ids, tokenized.document_start_index, tokenized.document_end_index
    # Exactly the document is inside the range, nothing before or after.
    assert tokenizer.decode(ids[start:end]) == document_text
    # Everything before the range is exactly the rendered prompt with the assistant header
    # (the first document token is predicted from the header's last token).
    assert tokenizer.decode(ids[:start]) == tokenized.prompt_text
    assert tokenized.prompt_text.endswith("<|turn>model\n") or tokenized.prompt_text.endswith("<channel|>")
    assert tokenizer.convert_ids_to_tokens(ids[start - 1]) in ("<channel|>", "Ċ")
    # The end-of-turn token comes right after the range and is not inside it.
    assert tokenizer.convert_ids_to_tokens(ids[end]) == "<turn|>"
    assert "<turn|>" not in tokenizer.decode(ids[start:end])
    assert tokenizer.convert_ids_to_tokens(ids[start]) != "<turn|>"
    assert system_prompt_token_ids_are_in_prompt(tokenizer, condition, ids[:start])


def system_prompt_token_ids_are_in_prompt(tokenizer, condition, prompt_ids):
    return condition["system_prompt"] in tokenizer.decode(prompt_ids)


def test_gemma_like_header_differs_from_finished_message_rendering_and_generation_header_is_used():
    tokenizer = make_tokenizer(tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN)
    tokenized = prompt_rendering.tokenize_condition_and_document(
        tokenizer, make_condition(), GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], "tiny test tokenizer")
    assert not tokenized.assistant_header_matches_finished_message_rendering
    assert tokenized.prompt_text.endswith("<|turn>model\n<|channel>thought\n<channel|>")
    assert tokenized.suffix_text == "<turn|>\n"


def test_no_system_prompt_condition_renders_without_system_turn():
    tokenizer = make_tokenizer()
    condition = make_condition(system_prompt=None, condition_id="no_system_prompt")
    tokenized = prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                                 SHORT_DOCUMENT_TEXTS[0], "tiny test tokenizer")
    assert "<|turn>system" not in tokenized.prompt_text


def test_document_edge_whitespace_is_ignored_consistently():
    tokenizer = make_tokenizer()
    condition = make_condition()
    plain = prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                             SHORT_DOCUMENT_TEXTS[0], "m")
    padded = prompt_rendering.tokenize_condition_and_document(
        tokenizer, condition, GENERATION_PROMPT, "\n\n  " + SHORT_DOCUMENT_TEXTS[0] + "  \n", "m")
    assert plain.full_token_ids == padded.full_token_ids


def test_header_that_merges_with_first_document_token_is_rejected():
    tokenizer = make_tokenizer(tiny.TEMPLATE_WHOSE_HEADER_MERGES_WITH_DOCUMENT_START)
    with pytest.raises(prompt_rendering.DocumentTokenLocationError, match="prefix"):
        prompt_rendering.tokenize_condition_and_document(
            tokenizer, make_condition(), GENERATION_PROMPT, "the shrimp season opened early.", "tiny merging tokenizer")


# --------------------------------------------------------------------------------------------------
# System prompt survives the chat template
# --------------------------------------------------------------------------------------------------

def test_template_that_silently_drops_system_prompt_raises_with_model_name_and_rendered_text():
    tokenizer = make_tokenizer(tiny.TEMPLATE_THAT_SILENTLY_DROPS_SYSTEM_MESSAGES)
    condition = make_condition()
    with pytest.raises(prompt_rendering.SystemPromptNotInRenderedPromptError) as error_info:
        prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                         SHORT_DOCUMENT_TEXTS[0], "example-org/dropping-model")
    message = str(error_info.value)
    assert "example-org/dropping-model" in message and "dropped the system prompt" in message
    assert GENERATION_PROMPT in message  # the rendered text is shown
    assert condition["system_prompt"] not in message.split("Rendered text:")[1]


def test_template_that_rejects_system_role_raises_clear_error():
    tokenizer = make_tokenizer(tiny.TEMPLATE_THAT_REJECTS_SYSTEM_ROLE)
    with pytest.raises(prompt_rendering.SystemPromptNotInRenderedPromptError, match="example-org/rejecting-model"):
        prompt_rendering.tokenize_condition_and_document(tokenizer, make_condition(), GENERATION_PROMPT,
                                                         SHORT_DOCUMENT_TEXTS[0], "example-org/rejecting-model")


def test_template_that_drops_system_prompt_is_fine_for_the_no_system_prompt_condition():
    tokenizer = make_tokenizer(tiny.TEMPLATE_THAT_SILENTLY_DROPS_SYSTEM_MESSAGES)
    condition = make_condition(system_prompt=None, condition_id="no_system_prompt")
    prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                     SHORT_DOCUMENT_TEXTS[0], "m")


def test_scoring_run_aborts_when_template_drops_system_prompts():
    tokenizer = make_tokenizer(tiny.TEMPLATE_THAT_SILENTLY_DROPS_SYSTEM_MESSAGES)
    document = {"document_id": "doc", "content": SHORT_DOCUMENT_TEXTS[0], "generation_prompt": GENERATION_PROMPT}
    with pytest.raises(prompt_rendering.SystemPromptNotInRenderedPromptError):
        scorer.tokenize_all_conditions_for_document(tokenizer, document, prompt_rendering.build_condition_records(), "m")


# --------------------------------------------------------------------------------------------------
# Document token ids identical across conditions
# --------------------------------------------------------------------------------------------------

def test_document_token_ids_are_identical_across_all_62_conditions():
    tokenizer = make_tokenizer()
    document = {"document_id": "doc", "content": SHORT_DOCUMENT_TEXTS[1], "generation_prompt": GENERATION_PROMPT}
    conditions = prompt_rendering.build_condition_records()
    tokenized_sequences, document_token_ids = scorer.tokenize_all_conditions_for_document(
        tokenizer, document, conditions, "m")
    assert len(tokenized_sequences) == 62
    for sequence in tokenized_sequences:
        assert sequence.full_token_ids[sequence.document_start_index:sequence.document_end_index] == document_token_ids
    assert len({sequence.document_start_index for sequence in tokenized_sequences}) > 1  # prompts differ in length


def test_mismatched_document_token_ids_across_conditions_are_caught(monkeypatch):
    tokenizer = make_tokenizer()
    real_function = prompt_rendering.tokenize_condition_and_document

    def corrupt_one_condition(tokenizer_argument, condition, generation_prompt, content, model_name=None):
        tokenized = real_function(tokenizer_argument, condition, generation_prompt, content, model_name)
        if condition["condition_id"] == "generic":
            tokenized.full_token_ids[tokenized.document_start_index + 2] += 1
        return tokenized

    monkeypatch.setattr(prompt_rendering, "tokenize_condition_and_document", corrupt_one_condition)
    document = {"document_id": "doc", "content": SHORT_DOCUMENT_TEXTS[1], "generation_prompt": GENERATION_PROMPT}
    with pytest.raises(AssertionError, match="differ"):
        scorer.tokenize_all_conditions_for_document(tokenizer, document, prompt_rendering.build_condition_records()[:3], "m")


# --------------------------------------------------------------------------------------------------
# Chunked log-probabilities equal the plain computation; scoring is aligned correctly
# --------------------------------------------------------------------------------------------------

def make_tiny_scoring_setup(final_logit_softcapping=2.0):
    tokenizer = make_tokenizer()
    model = tiny.build_tiny_random_gemma4_model(len(tokenizer), final_logit_softcapping)
    return model, tokenizer


@pytest.mark.parametrize("positions_per_chunk", [1, 3, 7, 1000])
def test_chunked_head_matches_plain_full_logits_including_softcap(positions_per_chunk):
    model, tokenizer = make_tiny_scoring_setup(final_logit_softcapping=2.0)
    conditions = prompt_rendering.build_condition_records()
    tokenized = [prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                                 SHORT_DOCUMENT_TEXTS[2], "m")
                 for condition in (conditions[0], conditions[1], conditions[5])]
    # one padded batch of three sequences of different lengths (also tests right-padding and the mask)
    batched = scorer.score_tokenized_sequences(model, tokenizer, tokenized, "cpu", positions_per_chunk)
    for sequence, chunked in zip(tokenized, batched):
        reference = scorer.compute_log_probabilities_with_plain_full_logits(model, tokenizer, sequence, "cpu")
        assert chunked.shape == (sequence.document_end_index - sequence.document_start_index,)
        np.testing.assert_allclose(chunked, reference, atol=2e-4, rtol=1e-4)
        assert (chunked <= 0).all()


def test_softcapping_changes_the_result_and_is_applied():
    model, tokenizer = make_tiny_scoring_setup(final_logit_softcapping=2.0)
    tokenized = [prompt_rendering.tokenize_condition_and_document(
        tokenizer, make_condition(), GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], "m")]
    capped = scorer.score_tokenized_sequences(model, tokenizer, tokenized, "cpu", 16)[0]
    model.config.text_config.final_logit_softcapping = None
    uncapped = scorer.score_tokenized_sequences(model, tokenizer, tokenized, "cpu", 16)[0]
    assert np.abs(capped - uncapped).max() > 0.01
    assert scorer.get_final_logit_softcapping(model) is None


def test_scored_positions_follow_the_next_token_rule_independently():
    """Brute force: for every document token, run the model on only the tokens before it."""
    model, tokenizer = make_tiny_scoring_setup()
    tokenized = prompt_rendering.tokenize_condition_and_document(tokenizer, make_condition(), GENERATION_PROMPT,
                                                                 SHORT_DOCUMENT_TEXTS[0], "m")
    scores = scorer.score_tokenized_sequences(model, tokenizer, [tokenized], "cpu", 8)[0]
    ids = tokenized.full_token_ids
    for offset, position in enumerate(range(tokenized.document_start_index, tokenized.document_end_index)):
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids[:position]]), use_cache=False).logits[0, -1].float()
        expected = torch.log_softmax(logits, dim=-1)[ids[position]].item()
        assert scores[offset] == pytest.approx(expected, abs=2e-4)


def test_end_of_turn_token_is_not_scored_and_first_token_depends_on_header():
    model, tokenizer = make_tiny_scoring_setup()
    tokenized = prompt_rendering.tokenize_condition_and_document(tokenizer, make_condition(), GENERATION_PROMPT,
                                                                 SHORT_DOCUMENT_TEXTS[0], "m")
    number_of_document_tokens = tokenized.document_end_index - tokenized.document_start_index
    scores = scorer.score_tokenized_sequences(model, tokenizer, [tokenized], "cpu", 8)[0]
    assert len(scores) == number_of_document_tokens
    # Changing the end-of-turn token must not change any score.
    changed_end = prompt_rendering.TokenizedConditionAndDocument(
        **{**tokenized.__dict__, "full_token_ids": tokenized.full_token_ids[:-2] + [tokenizer.pad_token_id] * 2})
    np.testing.assert_allclose(scorer.score_tokenized_sequences(model, tokenizer, [changed_end], "cpu", 8)[0],
                               scores, atol=1e-5)
    # Changing the last header token must change the first document token's score (it is conditioned on it).
    changed_header_ids = list(tokenized.full_token_ids)
    changed_header_ids[tokenized.document_start_index - 1] = tokenizer.convert_tokens_to_ids("<turn|>")
    changed_header = prompt_rendering.TokenizedConditionAndDocument(**{**tokenized.__dict__,
                                                                       "full_token_ids": changed_header_ids})
    assert scorer.score_tokenized_sequences(model, tokenizer, [changed_header], "cpu", 8)[0][0] != pytest.approx(scores[0], abs=1e-4)


def test_batch_splitting_respects_token_limit_and_keeps_every_sequence():
    class FakeSequence:
        def __init__(self, length):
            self.full_token_ids = [0] * length
    pairs = list(enumerate(FakeSequence(length) for length in [100, 90, 80, 50, 20, 10]))
    batches = scorer.split_into_batches(pairs, maximum_tokens_per_batch=250)
    assert sorted(index for batch in batches for index, _ in batch) == list(range(6))
    for batch in batches:
        assert len(batch) * len(batch[0][1].full_token_ids) <= 250 or len(batch) == 1


# --------------------------------------------------------------------------------------------------
# Run folders, resume, and conditions identical across documents
# --------------------------------------------------------------------------------------------------

@pytest.fixture
def small_corpus_path(tmp_path):
    path = tmp_path / "small_corpus.jsonl"
    write_jsonl(path, [{"doc_id": f"short_document_{index}", "content": text, "language": "English",
                        "variables": SHORT_VARIABLES} for index, text in enumerate(SHORT_DOCUMENT_TEXTS)])
    return path


@pytest.fixture
def build_run(tmp_path, monkeypatch, small_corpus_path):
    monkeypatch.setattr(build_inputs, "RUNS_DIRECTORY", tmp_path / "runs")
    monkeypatch.setattr(build_inputs, "SMALL_SAMPLE_CHECKS_DIRECTORY", tmp_path / "runs" / "small_sample_checks")

    def build(*extra_arguments, run_name="test_run", small_sample=True):
        arguments = ["--run-name", run_name, "--corpus-path", str(small_corpus_path),
                     "--model-name", "tiny-test-model", *extra_arguments]
        if small_sample:
            arguments.append("--small-sample-check")
        return build_inputs.main(arguments)
    return build


def test_build_scoring_inputs_writes_contract_files(build_run, tmp_path):
    run_folder = build_run()
    assert run_folder.parent == tmp_path / "runs" / "small_sample_checks"
    documents = read_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")
    conditions = read_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    assert len(documents) == 3 and len(conditions) == 62
    assert set(documents[0]) == {"document_id", "source_document_id", "rewrite_direction", "content", "language",
                                 "variables", "generation_prompt"}
    assert documents[0]["rewrite_direction"] is None and documents[0]["source_document_id"] == documents[0]["document_id"]
    assert set(conditions[0]) == {"condition_id", "condition_group", "persona_name", "rewording_index", "system_prompt"}
    configuration = json.loads(next(run_folder.glob("run_configuration_*.json")).read_text())
    assert configuration["model_name"] == "tiny-test-model"
    assert set(configuration["prompt_file_sha256"]) == set(prompt_rendering.PROMPT_FILE_NAMES_USED_FOR_SCORING)
    assert all(len(value) == 64 for value in configuration["prompt_file_sha256"].values())
    assert all(len(value) == 64 for value in configuration["input_file_sha256"].values())


def test_full_run_goes_outside_small_sample_checks(build_run, tmp_path):
    assert build_run(small_sample=False).parent == tmp_path / "runs"


def test_document_selection_options(build_run, tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("short_document_2\nshort_document_0\n")
    run_folder = build_run("--document-ids-file", str(ids_file), "--maximum-documents", "1", run_name="selection")
    assert [d["document_id"] for d in read_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")] == ["short_document_2"]
    ids_file.write_text("not_a_document\n")
    with pytest.raises(ValueError, match="not in the corpus"):
        build_run("--document-ids-file", str(ids_file), run_name="unknown")


def test_extra_rewritten_documents_are_appended_with_their_own_prompt(build_run, tmp_path):
    extra_path = tmp_path / "rewritten_documents.jsonl"
    write_jsonl(extra_path, [{"document_id": "short_document_0__preachier", "source_document_id": "short_document_0",
                              "rewrite_direction": "preachier", "content": "A preachier text.", "language": "English",
                              "variables": SHORT_VARIABLES, "rewrite_model_used": "claude-opus-5"}])
    run_folder = build_run("--extra-documents-jsonl", str(extra_path), "--maximum-documents", "1")
    documents = read_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")
    assert [d["document_id"] for d in documents] == ["short_document_0", "short_document_0__preachier"]
    assert documents[1]["rewrite_direction"] == "preachier" and documents[1]["source_document_id"] == "short_document_0"
    assert "rewrite_model_used" not in documents[1]
    assert documents[1]["generation_prompt"] == documents[0]["generation_prompt"]
    with pytest.raises(ValueError, match="Duplicate document ids"):
        build_run("--extra-documents-jsonl", str(extra_path), "--extra-documents-jsonl", str(extra_path), run_name="dup")


def test_resume_refuses_changed_prompt_file(build_run, tmp_path, monkeypatch):
    run_folder = build_run()
    changed_prompt_directory = tmp_path / "changed_prompts"
    shutil.copytree(prompt_rendering.PROMPT_DIRECTORY, changed_prompt_directory)
    generic_path = changed_prompt_directory / prompt_rendering.GENERIC_SYSTEM_PROMPT_FILE_NAME
    generic_path.write_text("You are a helpful AI assistant.\n")
    monkeypatch.setattr(prompt_rendering, "PROMPT_DIRECTORY", changed_prompt_directory)
    with pytest.raises(RuntimeError, match="generic_assistant_system_prompt.txt changed"):
        build_inputs.main(["--run-name", "test_run", "--resume-folder", str(run_folder), "--small-sample-check",
                           "--corpus-path", str(tmp_path / "small_corpus.jsonl"), "--model-name", "tiny-test-model"])


def test_resume_accepts_unchanged_inputs_and_refuses_changed_selection_or_model(build_run, tmp_path):
    run_folder = build_run()
    common = ["--run-name", "test_run", "--resume-folder", str(run_folder), "--small-sample-check",
              "--corpus-path", str(tmp_path / "small_corpus.jsonl")]
    build_inputs.main(common + ["--model-name", "tiny-test-model"])
    assert len(list(run_folder.glob("run_configuration_*.json"))) == 2  # the first one is kept
    with pytest.raises(RuntimeError, match="model"):
        build_inputs.main(common + ["--model-name", "another-model"])
    with pytest.raises(RuntimeError, match="differ"):
        build_inputs.main(common + ["--model-name", "tiny-test-model", "--maximum-documents", "2"])


def test_resume_refuses_changed_corpus_file(build_run, tmp_path):
    run_folder = build_run()
    corpus_path = tmp_path / "small_corpus.jsonl"
    records = read_jsonl(corpus_path)
    records[0]["content"] += " Edited."
    write_jsonl(corpus_path, records)
    with pytest.raises(RuntimeError, match="changed"):
        build_inputs.main(["--run-name", "test_run", "--resume-folder", str(run_folder), "--small-sample-check",
                           "--corpus-path", str(corpus_path), "--model-name", "tiny-test-model"])


@pytest.fixture
def patched_model_loading(monkeypatch):
    model, tokenizer = make_tiny_scoring_setup()
    calls = []

    def fake_load(model_name, device, dtype_name, model_class_name, model_revision):
        calls.append(model_name)
        return model, tokenizer
    monkeypatch.setattr(scorer, "load_model_and_tokenizer", fake_load)
    return calls


def run_scorer(run_folder, *extra_arguments):
    return scorer.main(["--run-folder", str(run_folder), "--device", "cpu", "--dtype", "float32",
                        "--maximum-tokens-per-batch", "20000", *extra_arguments])


def test_scorer_writes_contract_outputs_and_conditions_are_identical_across_documents(build_run, patched_model_loading):
    run_folder = build_run()
    run_scorer(run_folder, "--verify-first-document-against-full-logits")
    conditions = read_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    condition_ids = [condition["condition_id"] for condition in conditions]
    outputs = run_folder / "scoring_outputs"
    npz_paths = sorted((outputs / "per_token_log_probabilities").glob("*.npz"))
    assert len(npz_paths) == 3
    for npz_path in npz_paths:
        with np.load(npz_path) as saved:
            assert list(saved["condition_ids"]) == condition_ids  # same conditions, same order, every document
            assert saved["log_probabilities"].shape == (62, len(saved["document_token_ids"]))
            assert saved["log_probabilities"].dtype == np.float32 and saved["document_token_ids"].dtype == np.int32
            assert len(saved["document_token_strings"]) == len(saved["document_token_ids"])
            assert (saved["log_probabilities"] <= 1e-4).all()
    records = read_jsonl(outputs / "scoring_completion_records.jsonl")
    assert len(records) == 3 * 62
    assert {(record["document_id"], record["condition_id"]) for record in records} == {
        (f"short_document_{i}", condition_id) for i in range(3) for condition_id in condition_ids}
    assert list(outputs.glob("scoring_progress_log_*.txt")) and list(outputs.glob("scoring_environment_*.json"))
    assert "input tokens/s" in next(outputs.glob("scoring_progress_log_*.txt")).read_text()
    assert not list((outputs / "per_token_log_probabilities").glob("*.partial"))
    verification = json.loads(next(outputs.glob("full_logits_verification_*.json")).read_text())
    assert verification["passed"] is True and verification["document_id"] == "short_document_0"
    assert verification["condition_id"] == condition_ids[-1] and verification["maximum_absolute_difference"] < 1e-3


def test_failed_full_logits_verification_is_recorded_and_raises(build_run, patched_model_loading, monkeypatch):
    run_folder = build_run()
    original = scorer.compute_log_probabilities_with_plain_full_logits
    monkeypatch.setattr(scorer, "compute_log_probabilities_with_plain_full_logits",
                        lambda *arguments: original(*arguments) + 2.0)
    with pytest.raises(RuntimeError, match="differ"):
        run_scorer(run_folder, "--verify-first-document-against-full-logits")
    verification = json.loads(next((run_folder / "scoring_outputs").glob("full_logits_verification_*.json")).read_text())
    assert verification["passed"] is False and verification["maximum_absolute_difference"] >= 1.0


def test_scorer_without_verification_flag_writes_no_verification_file(build_run, patched_model_loading):
    run_folder = build_run()
    run_scorer(run_folder, "--maximum-documents", "1")
    assert not list((run_folder / "scoring_outputs").glob("full_logits_verification_*.json"))


def test_scorer_resume_skips_completed_documents(build_run, patched_model_loading):
    run_folder = build_run()
    run_scorer(run_folder, "--maximum-documents", "2")
    assert len(read_jsonl(run_folder / "scoring_outputs" / "scoring_completion_records.jsonl")) == 2 * 62
    first_file = run_folder / "scoring_outputs" / "per_token_log_probabilities" / "short_document_0.npz"
    modification_time = first_file.stat().st_mtime_ns
    run_scorer(run_folder)
    assert first_file.stat().st_mtime_ns == modification_time
    assert len(read_jsonl(run_folder / "scoring_outputs" / "scoring_completion_records.jsonl")) == 3 * 62
    patched_model_loading.clear()
    run_scorer(run_folder)  # everything done: the model must not even be loaded
    assert patched_model_loading == []


def test_scorer_rebuilds_missing_completion_records_and_refuses_foreign_conditions(build_run, patched_model_loading):
    run_folder = build_run()
    run_scorer(run_folder, "--maximum-documents", "1")
    records_path = run_folder / "scoring_outputs" / "scoring_completion_records.jsonl"
    records_path.unlink()
    run_scorer(run_folder, "--maximum-documents", "1")
    assert len(read_jsonl(records_path)) == 62
    # An .npz written under different conditions is refused instead of silently reused.
    npz_path = run_folder / "scoring_outputs" / "per_token_log_probabilities" / "short_document_0.npz"
    with np.load(npz_path) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["condition_ids"] = arrays["condition_ids"][::-1].copy()
    scorer.write_npz_atomically(npz_path, **arrays)
    records_path.unlink()
    with pytest.raises(RuntimeError, match="different conditions"):
        run_scorer(run_folder, "--maximum-documents", "1")


# --------------------------------------------------------------------------------------------------
# Real Gemma 4 tokenizer and chat template (runs only if the tokenizer files are already in the local
# Hugging Face cache; the smoke test downloads them. Tokenizer files only, never the weights.)
# --------------------------------------------------------------------------------------------------

def load_cached_gemma4_tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("google/gemma-4-12B-it", local_files_only=True)
    except Exception:
        return None


def test_real_gemma4_chat_template_keeps_system_prompt_and_locates_document_tokens():
    tokenizer = load_cached_gemma4_tokenizer()
    if tokenizer is None:
        pytest.skip("Gemma 4 tokenizer files are not in the local Hugging Face cache")
    corpus = read_jsonl(build_inputs.DEFAULT_CORPUS_PATH)
    conditions = prompt_rendering.build_condition_records()
    languages_seen = set()
    for record in sorted(corpus, key=lambda record: len(record["content"])):
        if record["language"] in languages_seen:
            continue
        languages_seen.add(record["language"])
        generation_prompt = prompt_rendering.render_generation_prompt(record["variables"], record["doc_id"])
        reference_ids = None
        for condition in (conditions[0], conditions[1], conditions[2], conditions[-1]):
            tokenized = prompt_rendering.tokenize_condition_and_document(
                tokenizer, condition, generation_prompt, record["content"], "google/gemma-4-12B-it")
            if condition["system_prompt"]:
                assert condition["system_prompt"] in tokenized.prompt_text
                assert tokenized.prompt_text.startswith("<bos><|turn>system\n")
            else:
                assert "<|turn>system" not in tokenized.prompt_text
            assert tokenized.prompt_text.endswith("<|turn>model\n<|channel>thought\n<channel|>")
            assert tokenized.suffix_text == "<turn|>\n"
            document_ids = tokenized.full_token_ids[tokenized.document_start_index:tokenized.document_end_index]
            reference_ids = reference_ids or document_ids
            assert document_ids == reference_ids
        if len(languages_seen) >= 6:
            break
