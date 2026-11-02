"""Automated tests for the moralizing delivery pilot: conversion of its drafts, the run folder (welfare points filled, the
neutral activation-pass prompt recorded), the activation pass (neutral prompt, identical answer tokens, welfare passage
means), the character-to-token mapping (experiment-invalidating if wrong), the judge kinds and their retry behaviour, the
welfare passage span step, the vector computation (filter conditions, pairing, largest-mean variants, held-out accuracy
by negative tag, folder naming), the cost ceilings and the pod launcher's two-stage use.

Run from the repository root:  python3 -m pytest scripts/persona_vector_projection -q
"""

import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from persona_vector_projection import build_extraction_run_folder as builder  # noqa: E402
from persona_vector_projection import chat_prompt_rendering_for_extraction as rendering  # noqa: E402
from persona_vector_projection import compute_moralizing_delivery_vectors_from_passage_activations as moralizing_compute  # noqa: E402
from persona_vector_projection import convert_drafts_into_machine_readable_inputs as converter  # noqa: E402
from persona_vector_projection import estimate_extraction_cost_ceilings as cost_estimator  # noqa: E402
from persona_vector_projection import generate_answers_and_save_activations_on_pod as pod_script  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import locate_welfare_passages_in_judged_answers as span_step  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import synthetic_extraction_run_for_tests as synthetic  # noqa: E402
from persona_vector_projection import welfare_passage_spans_and_token_mapping as mapping  # noqa: E402
from persona_vector_projection.answer_token_activation_recording import (  # noqa: E402
    AnswerTokenMeanActivationRecorder, build_padded_batch, build_passage_mask)
from persona_vector_projection.test_persona_vector_extraction import StubModel  # noqa: E402

NEUTRAL_PROMPT = "Keep your answer under 300 words."


# ---------------------------------------------------------------------------------------------------
# Conversion and the run folder
# ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("converted")
    converter.convert_everything(defs.QUESTION_DRAFT_FOLDER, defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER, root / "prompts",
                                 root / "converted_inputs_2026-01-01_00-00", "2026-01-01_00-00")
    return root


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_moralizing_delivery_drafts_convert_to_26_questions_with_welfare_points_and_tagged_prompts(converted):
    folder = converted / "converted_inputs_2026-01-01_00-00"
    questions = [q for q in read_records(folder / "questions_2026-01-01_00-00.jsonl") if q["facet"] == "moralizing_delivery"]
    assert len(questions) == 26
    assert [sum(1 for q in questions if q["half"] == half) for half in ("A", "B", "held_out")] == [10, 10, 6]
    assert all(q["labels"]["welfare_point"].strip() and q["labels"]["material_welfare_cost"] for q in questions)
    non_intrusive = {q["question_id"]: q for q in read_records(folder / "questions_2026-01-01_00-00.jsonl") if q["facet"] == "non_intrusive"}
    for question in questions:  # text, domain, species and half all come from the non-intrusive draft row
        source = non_intrusive[question["labels"]["source_question_id"]]
        assert (question["question_text"], question["domain"], question["species"], question["half"]) == (
            source["question_text"], source["domain"], source["species"], source["half"])
    prompts = [p for p in read_records(folder / "system_prompts_2026-01-01_00-00.jsonl") if p["facet"] == "moralizing_delivery"]
    assert [p["pole"] for p in prompts].count("positive") == 5 and [p["pole"] for p in prompts].count("negative") == 5
    assert all(p["system_prompt"].endswith(NEUTRAL_PROMPT) and p["system_prompt"].count("{welfare_point}") == 1 for p in prompts)
    assert {p["tag"] for p in prompts if p["pole"] == "negative"} == {
        "duty_and_conscience", "judging_the_user", "guilt_and_rhetorical_questions", "self_righteous_and_condescending", "exhortation"}
    assert all(p["tag"] is None for p in prompts if p["pole"] == "positive")
    assert (converted / "prompts" / "activation_pass_neutral_system_prompt_moralizing_delivery.txt").read_text().strip() == NEUTRAL_PROMPT
    for file_name in ("judge_delivery_score_prompt_template.txt", "judge_content_check_prompt_template.txt"):
        assert "{welfare_point}" in (converted / "prompts" / file_name).read_text()


def test_conversion_fails_loudly_when_a_question_has_no_welfare_point(tmp_path):
    questions = tmp_path / "questions"
    questions.mkdir()
    for path in defs.QUESTION_DRAFT_FOLDER.glob("*.md"):
        text = path.read_text()
        if path.name.startswith("moralizing-delivery-welfare-points"):
            text = "\n".join(line for line in text.splitlines() if not line.startswith("| A7 |"))
        (questions / path.name).write_text(text)
    with pytest.raises(SystemExit, match="A7.*no welfare point"):
        converter.convert_everything(questions, defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER, tmp_path / "prompts", tmp_path / "out")


def test_conversion_fails_loudly_on_a_welfare_point_for_no_question_and_on_a_prompt_without_the_placeholder(tmp_path):
    questions = tmp_path / "questions"
    questions.mkdir()
    for path in defs.QUESTION_DRAFT_FOLDER.glob("*.md"):
        text = path.read_text()
        if path.name.startswith("moralizing-delivery-welfare-points"):
            text += "| A2 | A welfare point for a question without a material cost. |\n"
        (questions / path.name).write_text(text)
    with pytest.raises(SystemExit, match="match no material-cost question"):
        converter.convert_everything(questions, defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER, tmp_path / "prompts", tmp_path / "out")
    prompts = tmp_path / "prompts_draft"
    prompts.mkdir()
    for path in defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER.glob("*.md"):
        text = path.read_text()
        if path.name.startswith("moralizing-delivery"):
            text = text.replace("The point: {welfare_point}", "The point: nothing", 1)
        (prompts / path.name).write_text(text)
    with pytest.raises(SystemExit, match="expected exactly one"):
        converter.convert_everything(defs.QUESTION_DRAFT_FOLDER, prompts, tmp_path / "prompts", tmp_path / "out2")


@pytest.fixture()
def run_folder_builder(converted, tmp_path, monkeypatch):
    monkeypatch.setattr(defs, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(defs, "PROMPT_FOLDER", converted / "prompts")

    def build(*extra, mode="moralizing_delivery_pilot"):
        return builder.build_run_folder(builder.parse_arguments(["--mode", mode, *extra]),
                                        converted / "converted_inputs_2026-01-01_00-00")
    return build


def test_moralizing_run_folder_has_1300_answers_filled_prompts_and_the_neutral_activation_prompt(run_folder_builder):
    run_folder, configuration = run_folder_builder()
    assert configuration["facets"] == ["moralizing_delivery"] and configuration["number_of_generation_jobs"] == 260
    assert configuration["number_of_answers_by_facet"] == {"moralizing_delivery": 1300}
    assert configuration["activation_pass_system_prompt"] == NEUTRAL_PROMPT
    assert configuration["record_welfare_passage_mean_activations"] is True
    assert (configuration["model_name"], configuration["model_revision"], configuration["temperature"], configuration["top_p"],
            configuration["maximum_new_tokens"], configuration["samples_per_prompt"]) == (
        defs.DEFAULT_MODEL_NAME, defs.DEFAULT_MODEL_REVISION, 1.0, 1.0, 1000, 5)
    jobs = read_records(defs.find_latest_file(run_folder / defs.INPUTS_FOLDER_NAME, "generation_jobs", "jsonl"))
    questions = {q["question_id"]: q for q in read_records(defs.find_latest_file(run_folder / defs.INPUTS_FOLDER_NAME, "questions", "jsonl"))}
    assert len(questions) == 26 and len(jobs) == 26 * 10 and not any(j["facet"] == "topic_control" for j in jobs)
    for job in jobs:
        assert "{" not in job["system_prompt"] and job["system_prompt"].endswith(NEUTRAL_PROMPT)
        assert questions[job["question_id"]]["labels"]["welfare_point"] in job["system_prompt"]
    # the same question's ten prompts carry the same point; two questions carry different points
    assert len({questions[j["question_id"]]["labels"]["welfare_point"] for j in jobs}) == 26


def test_other_modes_keep_the_generation_prompt_for_the_activation_pass(run_folder_builder):
    _, configuration = run_folder_builder("--maximum-questions-per-half", "2", mode="pilot")
    assert configuration["activation_pass_system_prompt"] is None
    assert configuration["record_welfare_passage_mean_activations"] is False
    assert "moralizing_delivery" not in configuration["facets"]


def test_building_jobs_fails_loudly_without_a_welfare_point_or_with_a_leftover_placeholder():
    prompt = {"facet": "moralizing_delivery", "pole": "positive", "tag": None, "system_prompt_id": "p1",
              "system_prompt": "Say this: {welfare_point} Keep it short.", "system_prompt_requires_welfare_point": True}
    question = {"facet": "moralizing_delivery", "question_id": "q1", "question_text": "?", "half": "A", "domain": "d",
                "labels": {"welfare_point": "Pigs feel pain."}}
    jobs = builder.build_generation_jobs([question], [prompt], 1)
    assert jobs[0]["system_prompt"] == "Say this: Pigs feel pain. Keep it short."
    with pytest.raises(ValueError, match="no welfare point"):
        builder.build_generation_jobs([{**question, "labels": {}}], [prompt], 1)
    with pytest.raises(ValueError, match="unfilled"):
        builder.build_generation_jobs([question], [{**prompt, "system_prompt": prompt["system_prompt"] + " {another_slot}"}], 1)


# ---------------------------------------------------------------------------------------------------
# Character span to tokens
# ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_tokenizer():
    from tiny_model_and_tokenizer_helpers_for_log_probability_persona_check import (
        GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN, build_small_offline_tokenizer)
    return build_small_offline_tokenizer(GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN, [
        "Hello pigs. The farm is old. Piglets feel pain. Use lidocaine first. Then the rest of the plan follows."] * 30)


def assert_selection_is_exactly_the_overlapping_tokens(answer_text, offsets, selected, start, end):
    """Every selected token overlaps the span, no unselected token does, and the selected tokens' characters cover it."""
    for index, (token_start, token_end) in enumerate(offsets):
        overlaps = token_end > token_start and token_start < end and token_end > start
        assert (index in selected) == overlaps, f"token {index} {offsets[index]} selection {index in selected}, overlap {overlaps}"
    covered = set()
    for index in selected:
        covered.update(range(*offsets[index]))
    assert set(range(start, end)) <= covered


def test_character_span_maps_to_exactly_the_covering_tokens_in_the_middle(tiny_tokenizer):
    text = "Hello pigs. The farm is old. Then the rest of the plan follows."
    passage = "The farm is old."
    start = text.index(passage)
    ids, offsets = mapping.tokenize_answer_with_character_offsets(tiny_tokenizer, text)
    assert ids == rendering.answer_token_ids(tiny_tokenizer, text)
    selected = mapping.token_indices_covering_character_span(offsets, start, start + len(passage))
    assert 0 < len(selected) < len(ids) and selected == list(range(selected[0], selected[-1] + 1))
    assert_selection_is_exactly_the_overlapping_tokens(text, offsets, selected, start, start + len(passage))
    assert selected[0] > 0 and selected[-1] < len(ids) - 1


def test_character_span_at_the_start_and_at_the_end_of_the_answer(tiny_tokenizer):
    text = "The farm is old. Then the rest of the plan follows. Use lidocaine first."
    ids, offsets = mapping.tokenize_answer_with_character_offsets(tiny_tokenizer, text)
    first = "The farm is old."
    selected_first = mapping.token_indices_covering_character_span(offsets, 0, len(first))
    assert selected_first[0] == 0 and selected_first[-1] < len(ids) - 1
    assert_selection_is_exactly_the_overlapping_tokens(text, offsets, selected_first, 0, len(first))
    last = "Use lidocaine first."
    last_start = text.index(last)
    selected_last = mapping.token_indices_covering_character_span(offsets, last_start, len(text))
    assert selected_last[-1] == len(ids) - 1 and selected_last[0] > 0
    assert_selection_is_exactly_the_overlapping_tokens(text, offsets, selected_last, last_start, len(text))
    whole = mapping.token_indices_covering_character_span(offsets, 0, len(text))
    assert whole == list(range(len(ids)))  # the whole answer as a span covers every token


def test_character_span_with_multi_byte_characters_uses_characters_not_bytes(tiny_tokenizer):
    # "é" is 2 bytes, "—" 3 bytes, the pig emoji 4 bytes: byte offsets and character offsets differ from the first one on
    text = "Café — piglets suffer 🐖 badly, and more 🐖 follows. Then the rest of the plan follows."
    passage = "piglets suffer 🐖 badly, and more 🐖 follows."
    start = text.index(passage)
    end = start + len(passage)
    assert len(text.encode("utf-8")) > len(text) + 5
    ids, offsets = mapping.tokenize_answer_with_character_offsets(tiny_tokenizer, text)
    assert max(end_ for _, end_ in offsets) == len(text)  # character offsets: the last token ends at the character count
    emoji_position = text.index("🐖")
    emoji_tokens = [i for i, (s, e) in enumerate(offsets) if s == emoji_position and e == emoji_position + 1]
    assert len(emoji_tokens) >= 2  # the emoji's several byte tokens all report that one character
    selected = mapping.token_indices_covering_character_span(offsets, start, end)
    assert_selection_is_exactly_the_overlapping_tokens(text, offsets, selected, start, end)
    assert set(emoji_tokens) <= set(selected)
    decoded_start = tiny_tokenizer.decode([ids[i] for i in selected]).strip()
    assert decoded_start.startswith("piglets") and decoded_start.endswith("follows.")
    # a span that is only the emoji selects only its byte tokens (plus nothing else)
    only_emoji = mapping.token_indices_covering_character_span(offsets, emoji_position, emoji_position + 1)
    assert set(only_emoji) == set(emoji_tokens)


def test_pure_mapping_rules_overlap_zero_width_tokens_and_invalid_spans():
    offsets = [(0, 0), (0, 5), (5, 11), (11, 12), (12, 20), (20, 20)]  # (0, 0) and (20, 20): special tokens
    assert mapping.token_indices_covering_character_span(offsets, 6, 12) == [2, 3]
    assert mapping.token_indices_covering_character_span(offsets, 5, 6) == [2]          # starts exactly at a boundary
    assert mapping.token_indices_covering_character_span(offsets, 4, 6) == [1, 2]       # straddles a boundary
    assert mapping.token_indices_covering_character_span(offsets, 0, 20) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        mapping.token_indices_covering_character_span(offsets, 7, 7)
    with pytest.raises(ValueError):
        mapping.token_indices_covering_character_span(offsets, None, 7)


def real_gemma_tokenizer_or_skip():
    matches = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/*"))
    if not matches:
        pytest.skip("the Gemma 4 tokenizer is not in the local Hugging Face cache")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(matches[0])


def test_character_span_mapping_with_the_real_gemma_tokenizer_including_multi_byte_text():
    tokenizer = real_gemma_tokenizer_or_skip()
    cases = [("Castrating piglets without pain relief hurts them. A local anaesthetic is cheap. Now the plan: step one.",
              "A local anaesthetic is cheap."),
             ("Café — piglets 🐖 hurt. Naïve people ignore it, 日本語 too. Then the plan.", "Naïve people ignore it, 日本語 too."),
             ("Use lidocaine first. The rest follows.", "Use lidocaine first."),
             ("The rest follows. Use lidocaine first.", "Use lidocaine first.")]
    for text, passage in cases:
        start = text.index(passage)
        ids, offsets = mapping.tokenize_answer_with_character_offsets(tokenizer, text)
        assert ids == rendering.answer_token_ids(tokenizer, text)
        selected = mapping.token_indices_covering_character_span(offsets, start, start + len(passage))
        assert_selection_is_exactly_the_overlapping_tokens(text, offsets, selected, start, start + len(passage))
        assert tokenizer.decode([ids[i] for i in selected]).strip() == passage


# ---------------------------------------------------------------------------------------------------
# Locating the passage in the answer
# ---------------------------------------------------------------------------------------------------

def test_passage_is_found_exactly_then_after_whitespace_and_quote_normalisation_else_recorded_as_failed():
    answer = "Plan first.\n\nThe  farm owner's pigs\nfeel “pain” here. More plan."
    exact = mapping.locate_passage_in_answer(answer, "Plan first.")
    assert (exact["status"], exact["character_start"], exact["character_end"]) == (mapping.MATCH_EXACT, 0, 11)
    normalised = mapping.locate_passage_in_answer(answer, "The farm owner's pigs feel \"pain\" here.")
    assert normalised["status"] == mapping.MATCH_NORMALISED
    assert answer[normalised["character_start"]:normalised["character_end"]] == "The  farm owner's pigs\nfeel “pain” here."
    curly_in_passage = mapping.locate_passage_in_answer("He said \"stop\" and left. Done.", "He said “stop” and left.")
    assert curly_in_passage["status"] == mapping.MATCH_NORMALISED and curly_in_passage["character_start"] == 0
    for passage in ("This text is not in the answer.", "", None, "   "):
        failed = mapping.locate_passage_in_answer(answer, passage)
        assert failed["status"] == mapping.MATCH_FAILED and failed["character_start"] is None and failed["failure_reason"]
    repeated = mapping.locate_passage_in_answer("Aa. Bb. Aa.", "Aa.")
    assert repeated["character_start"] == 0 and repeated["number_of_occurrences"] == 2


def test_span_step_writes_a_record_for_every_answer_and_records_discarded_replies_as_failures():
    answers = [{"answer_id": f"a{i}", "facet": "moralizing_delivery", "pole": "positive", "tag": None, "question_id": "q",
                "answer_text": f"Intro. Piglets hurt {i}. Outro."} for i in range(4)]
    judged = {"a0__content_check": {"kind": "content_check", "discarded": False,
                                    "content_check": {"welfare_passage": "Piglets hurt 0."}},
              "a1__content_check": {"kind": "content_check", "discarded": False,
                                    "content_check": {"welfare_passage": "not in the answer"}},
              "a2__content_check": {"kind": "content_check", "discarded": True, "content_check": None}}
    records = span_step.build_span_records(answers, judged)
    assert [r["status"] for r in records] == [mapping.MATCH_EXACT, mapping.MATCH_FAILED, mapping.MATCH_FAILED, mapping.MATCH_FAILED]
    assert [r["failure_reason"] is None for r in records] == [True, False, False, False]
    assert records[0]["answer_character_count"] == len(answers[0]["answer_text"])
    assert "did not parse" in records[2]["failure_reason"] and "no content check result" in records[3]["failure_reason"]
    assert span_step.summarise_span_records(records)["status_counts"] == {mapping.MATCH_EXACT: 1, mapping.MATCH_FAILED: 3}


# ---------------------------------------------------------------------------------------------------
# Activation pass: neutral prompt, identical answer tokens, passage means
# ---------------------------------------------------------------------------------------------------

POLE_SYSTEM_PROMPT = "You are an AI who frames the point as a moral duty. Piglets feel pain. Keep your answer under 300 words."


def make_moralizing_jobs_and_answers(count=4):
    jobs, answers = [], []
    for index in range(count):
        job = {"job_id": f"job_{index}", "facet": "moralizing_delivery", "pole": "negative", "tag": "exhortation",
               "system_prompt_id": "sp", "system_prompt": POLE_SYSTEM_PROMPT, "question_id": f"q{index}", "pair_id": None,
               "version": None, "half": "A", "domain": "d", "user_prompt": f"How do I castrate piglets {index}?",
               "samples_per_prompt": 1}
        jobs.append(job)
        answers.append({"answer_id": f"job_{index}__sample_00", "job_id": f"job_{index}", "facet": "moralizing_delivery",
                        "answer_text": f"Start with the plan. Piglets feel pain {index}. Use lidocaine first. Then finish the task."})
    return jobs, answers


def test_activation_pass_renders_the_neutral_prompt_and_the_answer_tokens_are_identical(tiny_tokenizer):
    jobs, answers = make_moralizing_jobs_and_answers()
    jobs_by_id = {job["job_id"]: job for job in jobs}
    default = pod_script.prepare_activation_sequences(answers, jobs_by_id, tiny_tokenizer)
    neutral = pod_script.prepare_activation_sequences(answers, jobs_by_id, tiny_tokenizer, activation_pass_system_prompt=NEUTRAL_PROMPT)
    for job, answer, default_item, neutral_item in zip(jobs, answers, default, neutral):
        # the default behaviour (generation prompt) is unchanged
        assert default_item["prompt_ids"] == rendering.prompt_token_ids(tiny_tokenizer, job["system_prompt"], job["user_prompt"])
        # neutral: exactly the prompt rendered with the neutral system prompt and the same user question
        assert neutral_item["prompt_ids"] == rendering.prompt_token_ids(tiny_tokenizer, NEUTRAL_PROMPT, job["user_prompt"])
        prompt_text = tiny_tokenizer.decode(neutral_item["prompt_ids"])
        assert NEUTRAL_PROMPT in prompt_text and "moral duty" not in prompt_text and "Piglets feel pain" not in prompt_text
        assert job["user_prompt"] in prompt_text
        assert neutral_item["prompt_ids"] != default_item["prompt_ids"]
        # identical answer tokens: those of the generated text, whichever system prompt is used
        assert neutral_item["answer_ids"] == default_item["answer_ids"] == rendering.answer_token_ids(tiny_tokenizer, answer["answer_text"])
    assert len({tuple(item["prompt_ids"]) for item in neutral}) == len(neutral)  # still one prompt per question


def passage_spans_for(answers, passage_template="Piglets feel pain {index}."):
    spans = {}
    for index, answer in enumerate(answers):
        passage = passage_template.format(index=index)
        start = answer["answer_text"].find(passage)
        spans[answer["answer_id"]] = {"answer_character_count": len(answer["answer_text"]),
                                      "character_start": None if start < 0 else start,
                                      "character_end": None if start < 0 else start + len(passage)}
    return spans


def test_activation_stage_with_neutral_prompt_stores_whole_answer_and_passage_means_and_marks_missing_spans(tmp_path, tiny_tokenizer):
    from tiny_model_and_tokenizer_helpers_for_log_probability_persona_check import build_tiny_random_gemma4_model
    run_folder = tmp_path / "run"
    answers_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME
    answers_folder.mkdir(parents=True)
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text("{}")
    jobs, answers = make_moralizing_jobs_and_answers(4)
    (answers_folder / "answers_shard_00000_2026-01-01_00-00.jsonl").write_text("\n".join(json.dumps(a) for a in answers) + "\n")
    spans = passage_spans_for(answers)
    spans[answers[2]["answer_id"]].update(character_start=None, character_end=None)  # a failed span
    model = build_tiny_random_gemma4_model(len(tiny_tokenizer))
    pod_script.run_activation_stage(run_folder, jobs, tiny_tokenizer, model, "cpu", 3, "float32",
                                    activation_pass_system_prompt=NEUTRAL_PROMPT, spans_by_answer_id=spans)
    shard = np.load(run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME / "activations_shard_00000_2026-01-01_00-00.npz")
    status = list(shard["passage_status"])
    assert status == ["passage_mean_recorded", "passage_mean_recorded", "no_span_found_no_passage_mean", "passage_mean_recorded"]
    assert list(shard["passage_token_counts"] > 0) == [True, True, False, True]
    assert np.isnan(shard["passage_activations"][2]).all() and not np.isnan(shard["passage_activations"][[0, 1, 3]]).any()
    assert (shard["answer_token_counts"] > 0).all()
    for row, answer in enumerate(answers):
        job = jobs[row]
        prompt_ids = rendering.prompt_token_ids(tiny_tokenizer, NEUTRAL_PROMPT, job["user_prompt"])
        answer_ids = rendering.answer_token_ids(tiny_tokenizer, answer["answer_text"])
        with torch.no_grad():
            hidden = model.base_model(input_ids=torch.tensor([prompt_ids + answer_ids]), output_hidden_states=True, use_cache=False).hidden_states
        expected_whole = torch.stack([h[0, len(prompt_ids):].mean(0) for h in hidden[:4]])
        assert np.allclose(shard["activations"][row][:4], expected_whole.numpy(), atol=1e-4)
        if status[row] == "passage_mean_recorded":
            _, offsets = mapping.tokenize_answer_with_character_offsets(tiny_tokenizer, answer["answer_text"])
            span = spans[answer["answer_id"]]
            tokens = mapping.token_indices_covering_character_span(offsets, span["character_start"], span["character_end"])
            expected_passage = torch.stack([h[0, [len(prompt_ids) + t for t in tokens]].mean(0) for h in hidden[:4]])
            assert np.allclose(shard["passage_activations"][row][:4], expected_passage.numpy(), atol=1e-4)
            assert not np.allclose(shard["passage_activations"][row], shard["activations"][row], atol=1e-3)
    # the activations differ from those under the pole's own generation prompt: the neutral prompt really was used
    jobs_by_id = {job["job_id"]: job for job in jobs}
    default_ids = pod_script.prepare_activation_sequences(answers[:1], jobs_by_id, tiny_tokenizer)[0]
    with torch.no_grad():
        default_hidden = model.base_model(input_ids=torch.tensor([default_ids["prompt_ids"] + default_ids["answer_ids"]]),
                                          output_hidden_states=True, use_cache=False).hidden_states
    default_mean = torch.stack([h[0, len(default_ids["prompt_ids"]):].mean(0) for h in default_hidden[:4]]).numpy()
    neutral_ids = rendering.prompt_token_ids(tiny_tokenizer, NEUTRAL_PROMPT, jobs[0]["user_prompt"])
    with torch.no_grad():
        neutral_hidden = model.base_model(input_ids=torch.tensor([neutral_ids + default_ids["answer_ids"]]),
                                          output_hidden_states=True, use_cache=False).hidden_states
    neutral_mean = torch.stack([h[0, len(neutral_ids):].mean(0) for h in neutral_hidden[:4]]).numpy()
    # (the tiny random model's attention is weak, so the two differ by only about 1e-5; the stored row is much closer to
    # the neutral reference than to the generation-prompt reference)
    assert np.abs(shard["activations"][0][:4] - default_mean).max() > 3 * np.abs(shard["activations"][0][:4] - neutral_mean).max()


def test_activation_stage_refuses_a_spans_file_that_is_missing_a_record_or_for_another_answer_text(tiny_tokenizer):
    jobs, answers = make_moralizing_jobs_and_answers(2)
    jobs_by_id = {job["job_id"]: job for job in jobs}
    spans = passage_spans_for(answers)
    with pytest.raises(ValueError, match="No welfare passage span record"):
        pod_script.prepare_activation_sequences(answers, jobs_by_id, tiny_tokenizer, spans_by_answer_id={answers[0]["answer_id"]: spans[answers[0]["answer_id"]]})
    spans[answers[0]["answer_id"]]["answer_character_count"] += 1
    with pytest.raises(ValueError, match="written for an answer of"):
        pod_script.prepare_activation_sequences(answers, jobs_by_id, tiny_tokenizer, spans_by_answer_id=spans)


def test_missing_spans_file_fails_loudly_for_a_run_that_records_passage_means(tmp_path):
    (tmp_path / defs.INPUTS_FOLDER_NAME).mkdir()
    with pytest.raises(FileNotFoundError):
        pod_script.load_welfare_passage_spans(tmp_path)


def test_passage_mean_in_the_recorder_averages_only_the_passage_tokens_and_rejects_tokens_outside_the_answer():
    torch.manual_seed(0)
    model = StubModel()
    recorder = AnswerTokenMeanActivationRecorder(model)
    recorder.attach()
    sequences = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11]]
    prompt_lengths = [2, 1]
    passage_positions = [[1, 2], []]  # within the answer; the second sequence has no passage
    input_ids, attention_mask, answer_mask = build_padded_batch(sequences, prompt_lengths, 0)
    passage_mask = build_passage_mask(prompt_lengths, passage_positions, input_ids.shape[1])
    means, passage_means = recorder.mean_activations_and_passage_means_for_batch(input_ids, attention_mask, answer_mask, passage_mask)
    with torch.no_grad():
        hidden = model.embedding(torch.tensor(sequences[0]))
        expected = [hidden[[3, 4]].mean(0)]
        for layer in model.layers:
            hidden = layer(hidden)[0]
            expected.append(hidden[[3, 4]].mean(0))
    assert torch.allclose(passage_means[0], torch.stack(expected), atol=1e-6)
    assert torch.allclose(means, recorder.mean_activations_for_batch(input_ids, attention_mask, answer_mask))
    assert torch.count_nonzero(passage_means[1]) == 0  # no passage: zeros, which the caller replaces with "no mean"
    bad_mask = build_passage_mask(prompt_lengths, [[], []], input_ids.shape[1])
    bad_mask[0, 0] = 1  # a prompt token
    with pytest.raises(ValueError, match="outside the answer"):
        recorder.mean_activations_and_passage_means_for_batch(input_ids, attention_mask, answer_mask, bad_mask)


# ---------------------------------------------------------------------------------------------------
# Judge kinds
# ---------------------------------------------------------------------------------------------------

GOOD_CONTENT_CHECK = {"makes_the_point": True, "separate_welfare_mentions": 1, "adds_welfare_facts": False,
                      "completes_the_task": True, "welfare_passage": "Piglets feel pain. Use lidocaine."}


class FakeResponse:
    def __init__(self, text):
        self.output_text, self.status = text, "completed"
        self.usage = SimpleNamespace(input_tokens=300, output_tokens=40, input_tokens_details=SimpleNamespace(cached_tokens=0),
                                     output_tokens_details=SimpleNamespace(reasoning_tokens=20))


def test_content_check_reply_parsing_accepts_fences_and_text_and_rejects_everything_malformed():
    parse = judge.parse_content_check_reply
    plain = json.dumps(GOOD_CONTENT_CHECK)
    assert parse(plain) == GOOD_CONTENT_CHECK
    assert parse(f"```json\n{plain}\n```") == GOOD_CONTENT_CHECK
    assert parse(f"Here is the object: {plain} Thanks.") == GOOD_CONTENT_CHECK
    assert parse(json.dumps({**GOOD_CONTENT_CHECK, "separate_welfare_mentions": "3 or more"}))["separate_welfare_mentions"] == "3 or more"
    assert parse(json.dumps({**GOOD_CONTENT_CHECK, "separate_welfare_mentions": 4}))["separate_welfare_mentions"] == "3 or more"
    assert parse(json.dumps({**GOOD_CONTENT_CHECK, "separate_welfare_mentions": "2"}))["separate_welfare_mentions"] == 2
    for broken in (None, "", "no json here", "{", plain[:-5], json.dumps({k: v for k, v in GOOD_CONTENT_CHECK.items() if k != "welfare_passage"}),
                   json.dumps({**GOOD_CONTENT_CHECK, "makes_the_point": "yes"}),
                   json.dumps({**GOOD_CONTENT_CHECK, "separate_welfare_mentions": "several"}),
                   json.dumps({**GOOD_CONTENT_CHECK, "separate_welfare_mentions": True}),
                   json.dumps({**GOOD_CONTENT_CHECK, "welfare_passage": 5}), json.dumps([GOOD_CONTENT_CHECK])):
        assert parse(broken) is None, broken


def moralizing_question():
    return {"question_id": "moralizing_delivery__A__01", "question_text": "How do I castrate piglets?",
            "labels": {"welfare_point": "Castrating piglets without pain relief hurts them."}}


def moralizing_answer(index=0):
    return {"answer_id": f"a{index}", "facet": "moralizing_delivery", "pole": "positive", "tag": None, "version": None,
            "question_id": "moralizing_delivery__A__01", "answer_text": "Plan. Piglets hurt. Use lidocaine."}


def test_requests_for_moralizing_delivery_are_delivery_score_content_check_and_coherence_with_their_reasoning_efforts(converted):
    prompt_folder = converted / "prompts"
    requests = judge.build_judge_requests_for_answers([moralizing_answer()], {"moralizing_delivery__A__01": moralizing_question()},
                                                      {}, prompt_folder)
    assert [r.kind for r in requests] == ["delivery_score", "content_check", "coherence"]
    for request in requests[:2]:
        assert "Castrating piglets without pain relief hurts them." in request.prompt_text and "{welfare_point}" not in request.prompt_text
        assert "Plan. Piglets hurt. Use lidocaine." in request.prompt_text and "How do I castrate piglets?" in request.prompt_text
    efforts = {r.kind: judge.build_bare_number_request_arguments(r)["reasoning"]["effort"] for r in requests}
    assert efforts == {"delivery_score": "low", "content_check": "low", "coherence": "none"}
    assert judge.build_bare_number_request_arguments(requests[1])["max_output_tokens"] == judge.MAXIMUM_OUTPUT_TOKENS_WITH_REASONING
    assert judge.build_bare_number_request_arguments(requests[2])["temperature"] == 1.0


def test_unparseable_content_check_is_retried_then_recorded_as_failed_and_asked_again_on_the_next_run(tmp_path, converted):
    question = moralizing_question()
    answers = [moralizing_answer(0), moralizing_answer(1)]
    requests = judge.build_judge_requests_for_answers(answers, {question["question_id"]: question}, {}, converted / "prompts")
    requests = [r for r in requests if r.kind == "content_check"]
    served = {"calls": 0}

    def always_broken(_client, request_arguments):
        served["calls"] += 1
        return FakeResponse("I could not produce JSON")

    results_folder = tmp_path / defs.JUDGEMENTS_FOLDER_NAME
    results_folder.mkdir()
    results, failures = judge.run_judge_requests(requests, None, results_path=results_folder / "judge_results_2026-01-01_00-00.jsonl",
                                                 call_responses=always_broken, worker_count=2)
    assert served["calls"] == 4 and len(results) == 2 and all(r["discarded"] and r["content_check"] is None for r in results)
    assert all(r["discard_reason"] == "unparseable reply" for r in results)

    def good(_client, request_arguments):
        served["calls"] += 1
        return FakeResponse(json.dumps(GOOD_CONTENT_CHECK))

    served["calls"] = 0
    results, failures = judge.run_judge_requests(requests, None, results_path=results_folder / "judge_results_2026-01-01_00-05.jsonl",
                                                 call_responses=good, worker_count=2)
    assert served["calls"] == 2 and not failures                      # the discarded requests were asked again
    assert len(results) == 2 and not any(r["discarded"] for r in results) and results[0]["content_check"] == GOOD_CONTENT_CHECK
    served["calls"] = 0
    judge.run_judge_requests(requests, None, results_path=results_folder / "judge_results_2026-01-01_00-10.jsonl",
                             call_responses=good, worker_count=2)
    assert served["calls"] == 0                                        # finished requests are skipped
    latest = judge.load_latest_judge_records_by_request_id(tmp_path)
    assert not latest[requests[0].request_id]["discarded"]
    # a first reply that fails and a second that parses counts as one parsed result, with both calls' usage
    replies = iter([FakeResponse("nope"), FakeResponse(json.dumps(GOOD_CONTENT_CHECK))])
    result, usage = judge.judge_one_request(None, requests[0], judge.READING_MODE_BARE_NUMBER, 0.25,
                                            call_responses=lambda *_: next(replies))
    assert not result["discarded"] and usage["input_tokens"] == 600


def test_delivery_score_is_a_bare_number_and_a_non_number_is_recorded_as_failed(converted):
    request = [r for r in judge.build_judge_requests_for_answers(
        [moralizing_answer()], {"moralizing_delivery__A__01": moralizing_question()}, {}, converted / "prompts")
        if r.kind == "delivery_score"][0]
    result, _ = judge.judge_one_request(None, request, judge.READING_MODE_BARE_NUMBER, 0.25, call_responses=lambda *_: FakeResponse("72"))
    assert result["score"] == 72.0 and not result["discarded"]
    result, _ = judge.judge_one_request(None, request, judge.READING_MODE_BARE_NUMBER, 0.25,
                                        call_responses=lambda *_: FakeResponse("The delivery is plain."))
    assert result["discarded"] and result["score"] is None


def test_cost_ceiling_for_the_moralizing_pilot_has_3900_calls_and_covers_the_idle_time(converted):
    converted_folder = converted / "converted_inputs_2026-01-01_00-00"
    arguments = cost_estimator.parse_arguments([])
    assumptions = {key: getattr(arguments, key) for key in vars(arguments) if key not in (
        "converted_inputs_folder", "pilot_maximum_questions_per_half", "pair_judge_sample_size")}
    result = cost_estimator.estimate_for_mode("moralizing_delivery_pilot", None, converted_folder, assumptions, 0)
    assert result["answers"] == 1300
    assert result["ceiling"]["judge"]["calls"] == 3900
    assert result["ceiling"]["judge"]["calls_by_kind"] == {"delivery_score": 1300, "content_check": 1300, "coherence": 1300}
    flow = result["two_stage_pod_flow"]
    assert flow["ceiling"]["idle_hours_between_stages"] == 2.0 and flow["ceiling"]["total_gpu_hours_including_idle"] > (
        flow["ceiling"]["first_launch_hours_before_idle"] + 2.0)
    first = flow["suggested_launcher_limits"]["first_launch_generation_with_keep_pod_after_run"]
    assert first["maximum_running_hours"] > flow["ceiling"]["first_launch_hours_before_idle"] + 2.0
    assert flow["suggested_launcher_limits"]["second_launch_activation_stage_on_reused_pod"]["maximum_dollars"] > 0


# ---------------------------------------------------------------------------------------------------
# Vector computation
# ---------------------------------------------------------------------------------------------------

THRESHOLDS = {"positive_delivery_score_minimum": 65.0, "negative_delivery_score_maximum": 35.0, "coherence_minimum": 50.0,
              "maximum_passage_length_ratio": 1.5, "required_separate_welfare_mentions": 1}


@pytest.fixture(autouse=True)
def unique_vector_set_folder_names_within_a_test(monkeypatch):
    """Several computations in one test must not meet the same-second refusal unless the test asks for it."""
    counter = {"seconds": 0}

    def next_stamp():
        counter["seconds"] += 1
        return f"2026-11-01_01-00-{counter['seconds']:02d}"
    monkeypatch.setattr(moralizing_compute, "timestamp_now_with_seconds", next_stamp)


@pytest.fixture()
def moralizing_run(tmp_path):
    folder, direction = synthetic.build_synthetic_moralizing_delivery_run(tmp_path / "run")
    return folder, direction


def run_moralizing_compute(folder, **threshold_overrides):
    return moralizing_compute.compute_all(folder, {**THRESHOLDS, **threshold_overrides}, "test", bootstrap_resamples=20,
                                          minimum_layer_index=1, seed=0, say=lambda *_: None)


def rewrite_jsonl(path, change):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    Path(path).write_text("\n".join(json.dumps(change(record) or record) for record in records) + "\n")


def edit_judge_record(folder, answer_id, kind, **changes):
    path = next((Path(folder) / defs.JUDGEMENTS_FOLDER_NAME).glob("judge_results_*.jsonl"))
    rewrite_jsonl(path, lambda r: ({**r, **changes} if r["answer_id"] == answer_id and r["kind"] == kind else None))


def edit_content_check(folder, answer_id, **changes):
    path = next((Path(folder) / defs.JUDGEMENTS_FOLDER_NAME).glob("judge_results_*.jsonl"))
    rewrite_jsonl(path, lambda r: ({**r, "content_check": {**r["content_check"], **changes}}
                                   if r["answer_id"] == answer_id and r["kind"] == "content_check" else None))


def edit_span(folder, answer_id, **changes):
    path = next((Path(folder) / defs.INPUTS_FOLDER_NAME).glob(f"{defs.WELFARE_PASSAGE_SPANS_FILE_STEM}_*.jsonl"))
    rewrite_jsonl(path, lambda r: ({**r, **changes} if r["answer_id"] == answer_id else None))


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def answer_id_for(pole, rewording, question_number, half="A", sample=0):
    prompt_id = defs.system_prompt_identifier("moralizing_delivery", pole, rewording)
    return f"{prompt_id}__{defs.question_record_identifier('moralizing_delivery', half, question_number)}__sample_{sample:02d}"


def test_vectors_recover_the_signal_in_passage_means_and_whole_answer_means_with_no_topic_removal(moralizing_run):
    folder, direction = moralizing_run
    metadata, metadata_path = run_moralizing_compute(folder)
    stamp = metadata["timestamp"]
    passage = np.load(metadata_path.parent / f"persona_vector_moralizing_delivery_passage_mean_{stamp}.npy")
    whole = np.load(metadata_path.parent / f"persona_vector_moralizing_delivery_whole_answer_mean_{stamp}.npy")
    assert passage.shape == whole.shape == (30, 48)
    assert metadata["per_facet"]["moralizing_delivery"]["filter_counts"]["items_kept"] == 12 * 5 * 2
    for layer in (1, 12, 29):
        assert cosine(passage[layer], direction) > 0.98 and cosine(whole[layer], direction) > 0.9
    # the passage vector's size reflects the passage signal (2 x scale 5), the whole-answer vector's the weaker one (2 x 1.5)
    assert abs(np.linalg.norm(passage[5]) - 10.0) < 0.5 and abs(np.linalg.norm(whole[5]) - 3.0) < 0.5
    assert "topic" not in " ".join(metadata["vector_files"].values())
    entry = metadata["per_facet"]["moralizing_delivery"]
    for space in ("passage_mean", "whole_answer_mean"):
        assert len(entry[space]["split_half_cosine_half_A_against_half_B_per_layer"]) == 30
        assert len(entry[space]["bootstrap_cosine_with_full_vector_mean_per_layer"]) == 30
        assert len(entry[space]["variant_held_out_metrics"]["raw"]["held_out_accuracy_per_layer"]) == 30  # layer 1 is entry 1
    assert metadata["layer_choice"]["layer_by_facet"]["moralizing_delivery"] >= 1
    assert entry["variant_summary_at_chosen_and_reported_layers"]["layers_reported"] == sorted(
        {metadata["layer_choice"]["shared_layer"], 12, 18, 24})


def test_items_pair_positive_rewording_n_with_negative_rewording_n_on_the_same_question_and_sample(moralizing_run):
    folder, _ = moralizing_run
    answers = moralizing_compute.shared_compute.load_answers(folder)
    items = moralizing_compute.group_answers_into_items(answers)
    assert len(items) == (12 * 5 * 2 + 6 * 5 * 2)
    for (question_id, rewording, sample), members in items.items():
        assert set(members) == {"positive", "negative"}
        for answer in members.values():
            assert (answer["question_id"], moralizing_compute.shared_compute.rewording_number_of(answer), answer["sample_index"]) == (
                question_id, rewording, sample)
    # breaking the pairing (negative activations of another question) would move the vector far from the signal; with the
    # real pairing the question offsets cancel exactly in each difference
    metadata, path = run_moralizing_compute(folder)
    vector = np.load(path.parent / f"persona_vector_moralizing_delivery_passage_mean_{metadata['timestamp']}.npy")
    assert cosine(vector[3], synthetic_direction()) > 0.99


def synthetic_direction():
    direction = np.zeros(48, dtype=np.float32)
    direction[10:14] = 0.5
    return direction


@pytest.mark.parametrize("condition, setup", [
    ("does_not_make_the_point", lambda f, a: edit_content_check(f, a, makes_the_point=False)),
    ("separate_welfare_mentions_not_the_required_count", lambda f, a: edit_content_check(f, a, separate_welfare_mentions=2)),
    ("separate_welfare_mentions_not_the_required_count", lambda f, a: edit_content_check(f, a, separate_welfare_mentions="3 or more")),
    ("separate_welfare_mentions_not_the_required_count", lambda f, a: edit_content_check(f, a, separate_welfare_mentions=0)),
    ("adds_welfare_facts", lambda f, a: edit_content_check(f, a, adds_welfare_facts=True)),
    ("does_not_complete_the_task", lambda f, a: edit_content_check(f, a, completes_the_task=False)),
    ("coherence_below_minimum", lambda f, a: edit_judge_record(f, a, "coherence", score=49.0)),
    ("no_welfare_passage_span_found", lambda f, a: edit_span(f, a, character_start=None, character_end=None, status="failed")),
    ("delivery_score_missing", lambda f, a: edit_judge_record(f, a, "delivery_score", score=None, discarded=True)),
    ("content_check_missing", lambda f, a: edit_judge_record(f, a, "content_check", discarded=True, content_check=None)),
])
def test_each_answer_condition_drops_the_item_and_is_counted_by_pole_and_tag(moralizing_run, condition, setup):
    folder, _ = moralizing_run
    negative_id = answer_id_for("negative", 3, 1)  # tag guilt_and_rhetorical_questions
    setup(folder, negative_id)
    metadata, _ = run_moralizing_compute(folder)
    entry = metadata["per_facet"]["moralizing_delivery"]
    assert entry["filter_counts"]["items_kept"] == 12 * 5 * 2 - 1
    assert entry["filter_counts"]["items_failing_each_condition"] == {f"negative:{condition}": 1}
    cell = entry["answer_condition_counts_by_pole_and_tag"]
    assert cell["negative"]["guilt_and_rhetorical_questions"]["failing"] == {condition: 1}
    assert cell["negative"]["all"]["failing"] == {condition: 1} and cell["positive"]["all"]["failing"] == {}
    assert cell["negative"]["exhortation"]["failing"] == {}
    assert cell["negative"]["guilt_and_rhetorical_questions"]["passing_every_answer_condition"] == 12 * 2 - 1 + 6 * 2


def test_delivery_score_thresholds_are_inclusive_and_the_positive_pole_is_checked_against_its_own_minimum(moralizing_run):
    folder, _ = moralizing_run
    edit_judge_record(folder, answer_id_for("positive", 1, 1), "delivery_score", score=65.0)    # exactly the minimum: kept
    edit_judge_record(folder, answer_id_for("positive", 1, 2), "delivery_score", score=64.0)    # below: dropped
    edit_judge_record(folder, answer_id_for("negative", 1, 1), "delivery_score", score=35.0)    # exactly the maximum: kept
    edit_judge_record(folder, answer_id_for("negative", 2, 2), "delivery_score", score=36.0)    # above: dropped
    metadata, _ = run_moralizing_compute(folder)
    counts = metadata["per_facet"]["moralizing_delivery"]["filter_counts"]
    assert counts["items_kept"] == 12 * 5 * 2 - 2
    assert counts["items_failing_each_condition"] == {"positive:positive_delivery_score_below_minimum": 1,
                                                      "negative:negative_delivery_score_above_maximum": 1}
    relaxed, _ = run_moralizing_compute(folder, positive_delivery_score_minimum=60.0, negative_delivery_score_maximum=40.0)
    assert relaxed["per_facet"]["moralizing_delivery"]["filter_counts"]["items_kept"] == 12 * 5 * 2


def test_passage_length_ratio_over_one_and_a_half_drops_the_item_and_the_ratio_is_an_option(moralizing_run):
    folder, _ = moralizing_run
    positive_id = answer_id_for("positive", 2, 1)
    answers = {a["answer_id"]: a for a in moralizing_compute.shared_compute.load_answers(folder)}
    span = next(json.loads(line) for line in (folder / defs.INPUTS_FOLDER_NAME).glob("*spans*").__next__().read_text().splitlines()
                if json.loads(line)["answer_id"] == positive_id)
    length = span["character_end"] - span["character_start"]
    # stretch the positive passage to 1.6 times the negative one: same start, a longer end within the answer text
    edit_span(folder, positive_id, character_end=span["character_start"] + int(length * 1.6))
    answers_text = answers[positive_id]["answer_text"]
    assert span["character_start"] + int(length * 1.6) <= len(answers_text)
    metadata, _ = run_moralizing_compute(folder)
    counts = metadata["per_facet"]["moralizing_delivery"]["filter_counts"]
    assert counts["items_kept"] == 12 * 5 * 2 - 1 and counts["items_failing_each_condition"] == {
        "passage_character_length_ratio_above_maximum": 1}
    metadata_relaxed, _ = run_moralizing_compute(folder, maximum_passage_length_ratio=2.0)
    assert metadata_relaxed["per_facet"]["moralizing_delivery"]["filter_counts"]["items_kept"] == 12 * 5 * 2


def test_held_out_accuracy_is_reported_per_negative_tag_and_a_device_that_is_not_separated_shows_up(moralizing_run):
    folder, direction = moralizing_run
    # make the held-out answers of one device (exhortation, rewording 5) look like the positive pole in passage space
    activations_folder = folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    held_out_exhortation = {answer_id_for("negative", 5, number, "held_out", sample) for number in range(1, 7) for sample in (0, 1)}
    for path in activations_folder.glob("activations_shard_*.npz"):
        data = dict(np.load(path))
        for row, answer_id in enumerate(data["answer_ids"]):
            if str(answer_id) in held_out_exhortation:
                data["passage_activations"][row] = (data["passage_activations"][row].astype(np.float32) + 2 * 5.0 * direction).astype(np.float16)
        np.savez(path, **data)
    metadata, _ = run_moralizing_compute(folder)
    entry = metadata["per_facet"]["moralizing_delivery"]
    held = entry["held_out_answers_used"]
    assert held["negative"] == 6 * 5 * 2 and held["negative_by_tag"] == {t: 12 for t in defs.MORALIZING_DELIVERY_NEGATIVE_TAGS}
    by_tag = entry["passage_mean"]["variant_held_out_metrics"]["raw"]["by_negative_tag"]
    assert set(by_tag) == set(defs.MORALIZING_DELIVERY_NEGATIVE_TAGS)
    layer = 5
    assert by_tag["exhortation"]["negatives_classified_correctly_per_layer"][layer] == 0.0
    assert 0.3 < by_tag["exhortation"]["area_under_curve_per_layer"][layer] < 0.7  # no better than chance
    for tag in ("duty_and_conscience", "judging_the_user", "guilt_and_rhetorical_questions", "self_righteous_and_condescending"):
        assert by_tag[tag]["negatives_classified_correctly_per_layer"][layer] == 1.0
        assert by_tag[tag]["accuracy_per_layer"][layer] == 1.0 and by_tag[tag]["area_under_curve_per_layer"][layer] == 1.0


def test_largest_mean_dimensions_variants_standardised_vector_and_projection_anchors(moralizing_run):
    folder, direction = moralizing_run
    metadata, metadata_path = run_moralizing_compute(folder)
    folder_path, stamp = metadata_path.parent, metadata["timestamp"]
    for space in ("passage_mean", "whole_answer_mean"):
        raw = np.load(folder_path / f"persona_vector_moralizing_delivery_{space}_{stamp}.npy")
        statistics = np.load(folder_path / f"activation_mean_and_standard_deviation_per_layer_{space}_{stamp}.npz")
        mean, standard_deviation = statistics["mean"], statistics["standard_deviation"]
        assert statistics["answers_used"] == 12 * 5 * 2 * 2 + 6 * 5 * 2 * 2 and mean.shape == (30, 48)
        # dimensions 0 and 1 carry the large constants, so they are the two largest-mean dimensions at every layer
        table = json.loads((folder_path / f"largest_absolute_mean_dimensions_per_layer_{space}_{stamp}.json").read_text())
        assert len(table["dimensions_per_layer"]) == 30 and all(len(layer) == 32 for layer in table["dimensions_per_layer"])
        for layer_entries in table["dimensions_per_layer"]:
            assert [e["dimension"] for e in layer_entries[:2]] == [0, 1]
            assert abs(layer_entries[0]["mean"]) > abs(layer_entries[1]["mean"]) > abs(layer_entries[-1]["mean"])
            for e in layer_entries:
                assert set(e) == {"dimension", "mean", "standard_deviation", "vector_component"}
        layer = 7
        for count in (1, 2, 4, 8, 16, 32):
            zeroed = np.load(folder_path / f"persona_vector_moralizing_delivery_{space}_top_{count}_largest_mean_dimensions_zeroed_{stamp}.npy")
            top = np.argsort(-np.abs(mean[layer]), kind="stable")[:count]
            assert (zeroed[layer][top] == 0).all()
            rest = np.setdiff1d(np.arange(48), top)
            assert np.allclose(zeroed[layer][rest], raw[layer][rest])
        standardised = np.load(folder_path / f"persona_vector_moralizing_delivery_{space}_standardised_by_per_dimension_variance_{stamp}.npy")
        variance = standard_deviation[layer].astype(np.float64) ** 2
        usable = variance > 1e-6
        assert np.allclose(standardised[layer][usable], (raw[layer][usable] / variance[usable]).astype(np.float32), rtol=1e-3, atol=1e-4)
        anchors = json.loads((folder_path / f"mean_projection_onto_each_variant_{space}_{stamp}.json").read_text())["variants"]
        assert set(anchors) == set(metadata["per_facet"]["moralizing_delivery"][space]["variant_names"]) and len(anchors) == 8
        raw_anchor = anchors["raw"]
        assert raw_anchor["mean_projection_positive_pole_answers_per_layer"][layer] > raw_anchor["mean_projection_negative_pole_answers_per_layer"][layer]
        mean_of_poles = 0.5 * (raw_anchor["mean_projection_positive_pole_answers_per_layer"][layer]
                               + raw_anchor["mean_projection_negative_pole_answers_per_layer"][layer])
        assert abs(raw_anchor["mean_projection_all_answers_per_layer"][layer] - mean_of_poles) < 0.05  # poles are balanced
        # the held-out accuracy and area under the curve exist for every variant at the chosen layer and layers 12, 18, 24
        summary = metadata["per_facet"]["moralizing_delivery"]["variant_summary_at_chosen_and_reported_layers"]
        for variant, values in summary["by_space"][space].items():
            assert set(values["held_out_accuracy"]) == {str(l) for l in summary["layers_reported"]}
            assert set(values["held_out_area_under_curve"]) == {str(l) for l in summary["layers_reported"]}
    # zeroing the large-mean dimensions helps nothing here (they carry no signal) but must not hurt, and the raw vector is
    # separable at its layer
    raw_accuracy = metadata["per_facet"]["moralizing_delivery"]["passage_mean"]["variant_held_out_metrics"]["raw"]["held_out_accuracy_per_layer"]
    assert max(raw_accuracy[1:]) == 1.0


def test_answers_without_a_span_are_excluded_and_a_second_run_in_the_same_second_is_refused(moralizing_run, monkeypatch):
    folder, _ = moralizing_run
    edit_span(folder, answer_id_for("positive", 4, 2), status="failed", character_start=None, character_end=None)
    monkeypatch.setattr(moralizing_compute, "timestamp_now_with_seconds", lambda: "2026-11-01_01-30-07")
    metadata, metadata_path = run_moralizing_compute(folder)
    assert metadata_path.parent.name == "test_2026-11-01_01-30-07"
    assert metadata["per_facet"]["moralizing_delivery"]["filter_counts"]["items_kept"] == 12 * 5 * 2 - 1
    filter_file = next(metadata_path.parent.glob("answer_filter_conditions_*.jsonl"))
    failed = {r["answer_id"]: r["failed_conditions"] for r in map(json.loads, filter_file.read_text().splitlines())}
    assert failed[answer_id_for("positive", 4, 2)] == ["no_welfare_passage_span_found"]
    with pytest.raises(FileExistsError):
        run_moralizing_compute(folder)
    assert metadata_path.exists()
    monkeypatch.setattr(moralizing_compute, "timestamp_now_with_seconds", lambda: "2026-11-01_01-30-08")
    _, second_path = run_moralizing_compute(folder)  # a different second: its own folder, the first untouched
    assert second_path.parent != metadata_path.parent and metadata_path.exists()


def test_a_run_where_no_item_survives_fails_loudly_with_the_conditions(moralizing_run):
    folder, _ = moralizing_run
    with pytest.raises(SystemExit, match="No item survived"):
        run_moralizing_compute(folder, coherence_minimum=95.0)


# ---------------------------------------------------------------------------------------------------
# Pod launcher, two-stage use
# ---------------------------------------------------------------------------------------------------

@pytest.fixture()
def extraction_launcher_profile():
    """The extraction job profile applied to the shared launcher, restored afterwards (it sets module-level names)."""
    from persona_vector_projection import launch_and_stop_runpod_generation_pod as wrapper
    from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as pod_launcher
    names = ["INPUTS_FOLDER_NAME", "OUTPUTS_FOLDER_NAME", "POD_NAME_PREFIX", "SCRIPT_FOLDER_NAME", "SCORER_SCRIPT_RELATIVE_PATH",
             "REQUIREMENTS_FILE_RELATIVE_PATH", "SCRIPT_FOLDER_NAMES_TO_UPLOAD", "SCORER_PROCESS_PATTERN",
             "PROGRESS_STATUS_SHELL_TEMPLATE", "PROGRESS_STATUS_UNIT_NAME", "MINIMUM_LOCAL_FREE_DISK_BYTES",
             "DEFAULT_CONTAINER_DISK_GIGABYTES", "DEFAULT_VOLUME_GIGABYTES", "FINAL_COPY_FILE_LIST_SHELL",
             "build_scorer_command_line", "REMOTE_WORKING_DIRECTORY", "REMOTE_RUN_FOLDER"]
    saved = {name: getattr(pod_launcher, name) for name in names}
    wrapper.EXTRA_POD_SCRIPT_OPTIONS.clear()
    yield wrapper, pod_launcher
    for name, value in saved.items():
        setattr(pod_launcher, name, value)
    wrapper.EXTRA_POD_SCRIPT_OPTIONS.clear()


def test_launcher_dry_run_passes_the_stage_and_keep_and_reuse_options_through(extraction_launcher_profile, tmp_path, capsys):
    wrapper, _ = extraction_launcher_profile
    run_folder = tmp_path / "run"
    (run_folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True)
    wrapper.main(["--run-folder", str(run_folder), "--stage", "generation", "--keep-pod-after-run", "--maximum-running-hours", "4.6",
                  "--maximum-dollars", "20", "--dry-run"])
    first = capsys.readouterr().out
    assert "--stage generation" in first and "KEPT running (--keep-pod-after-run)" in first and "Cost ceiling: 20.00 dollars or 4.6 hours" in first
    wrapper.EXTRA_POD_SCRIPT_OPTIONS.clear()
    wrapper.main(["--run-folder", str(run_folder), "--stage", "activations", "--reuse-existing-pod", "abc123",
                  "--maximum-running-hours", "1.5", "--maximum-dollars", "6", "--dry-run"])
    second = capsys.readouterr().out
    assert "--stage activations" in second and "--stage generation" not in second and "Cost ceiling: 6.00 dollars or 1.5 hours" in second


def test_reused_pod_upload_carries_the_spans_file_and_the_pulled_answers_and_not_the_judge_outputs(extraction_launcher_profile, tmp_path, monkeypatch):
    wrapper, pod_launcher = extraction_launcher_profile
    workspace = tmp_path / "fake_pod_workspace" / "copy"
    monkeypatch.setattr(pod_launcher, "REMOTE_WORKING_DIRECTORY", str(workspace))
    monkeypatch.setattr(pod_launcher, "REMOTE_RUN_FOLDER", f"{workspace}/run_folder")
    run_folder, _ = synthetic.build_synthetic_moralizing_delivery_run(tmp_path / "local_run", layers=3, hidden=8)
    for path in (run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME).glob("*.npz"):
        path.unlink()  # at the time of the second launch, activations do not exist yet
    wrapper.apply_job_profile()
    pod_launcher.REMOTE_WORKING_DIRECTORY = str(workspace)
    pod_launcher.REMOTE_RUN_FOLDER = f"{workspace}/run_folder"
    transfer = pod_launcher.SshTarTransfer("unused_key", run_folder, ssh_timeout_seconds=120, remote_shell_prefix=["bash", "-c"])
    transfer.upload_project_files()
    remote = workspace / "run_folder"
    assert (remote / defs.INPUTS_FOLDER_NAME / f"{defs.WELFARE_PASSAGE_SPANS_FILE_STEM}_2026-01-01_00-00.jsonl").exists()
    assert list((remote / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME).glob("answers_shard_*.jsonl"))
    assert (remote / "run_configuration_2026-01-01_00-00.json").exists() and not (remote / defs.JUDGEMENTS_FOLDER_NAME).exists()
    assert (workspace / "scripts" / "persona_vector_projection" / "generate_answers_and_save_activations_on_pod.py").exists()
    assert (workspace / "scripts" / "persona_vector_projection" / "welfare_passage_spans_and_token_mapping.py").exists()
    # the pod script reads the uploaded spans file the way the activation stage does
    spans = pod_script.load_welfare_passage_spans(remote)
    assert len(spans) == 12 * 5 * 2 * 2 + 6 * 5 * 2 * 2
