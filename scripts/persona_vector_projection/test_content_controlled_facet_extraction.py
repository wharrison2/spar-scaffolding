"""Tests for the content-controlled facets' extraction side (cares about welfare, calibrated consideration, non-intrusive, and
moralizing delivery through the same definitions): prompt extraction, conversion, run folders, judge requests, the filter and
the vector computation. Free and local; synthetic data and mocks only.

Experiment-invalidating cases covered: the scale and pole direction (positive pole = 100 = positive minus negative), a filter that
keeps a wrong-pole item, passage mean against whole-answer mean, completes_the_task in the filter, unfilled or different
content between the poles, wrong tags, and content drafts parsed by structure.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import build_extraction_run_folder as builder  # noqa: E402
from persona_vector_projection import compute_content_controlled_facet_vectors_from_activations as compute  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module  # noqa: E402
from persona_vector_projection import convert_drafts_into_machine_readable_inputs as converter  # noqa: E402
from persona_vector_projection import extract_content_controlled_prompts_from_draft as extractor  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import synthetic_extraction_run_for_tests as synthetic  # noqa: E402

NEW_FACET_KEYS = facet_definitions_module.NEW_CONTENT_CONTROLLED_FACET_KEYS
ALL_FACET_KEYS = facet_definitions_module.ALL_CONTENT_CONTROLLED_FACET_KEYS
THRESHOLDS = {"positive_facet_score_minimum": 65.0, "negative_facet_score_maximum": 35.0, "coherence_minimum": 50.0,
              "maximum_passage_length_ratio": 1.5, "required_separate_welfare_mentions": 1}


def definition_of(facet_key):
    return facet_definitions_module.get_facet_definition(facet_key)


# ---------------------------------------------------------------------------------------------------
# Scale and pole direction
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("facet_key", ALL_FACET_KEYS)
def test_every_facet_gives_100_to_the_positive_pole_and_the_expected_steering_sign_is_derived_from_that(facet_key):
    definition = definition_of(facet_key)
    assert definition.high_score_pole == defs.POLE_POSITIVE and definition.low_score_pole == defs.POLE_NEGATIVE
    assert definition.expected_sign_of_steering_fraction_on_facet_score == 1
    label = definition.facet_score_scale_label()
    assert label.index("100 =") < label.index("(positive pole)") < label.index("; 0 =") < label.index("(negative pole)")
    flipped = facet_definitions_module.ContentControlledFacetDefinition(
        **{**definition.__dict__, "high_score_pole": defs.POLE_NEGATIVE})
    assert flipped.expected_sign_of_steering_fraction_on_facet_score == -1  # derived, not typed in


def test_main_space_is_the_passage_mean_except_for_non_intrusive_which_uses_the_whole_answer_mean():
    assert definition_of("cares_about_welfare").main_space == "passage_mean"
    assert definition_of("calibrated_consideration").main_space == "passage_mean"
    assert definition_of("non_intrusive").main_space == "whole_answer_mean"
    assert definition_of("non_intrusive").vector_spaces == ["whole_answer_mean"]
    assert definition_of("non_intrusive").uses_task_completion_in_filter is False
    assert all(definition_of(key).uses_task_completion_in_filter for key in ("cares_about_welfare", "calibrated_consideration",
                                                                            "moralizing_delivery"))


# ---------------------------------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def extracted_prompts(tmp_path_factory):
    folder = tmp_path_factory.mktemp("extracted_prompts")
    shutil.copy(defs.PROMPT_FOLDER / "judge_content_check_prompt_template.txt", folder)
    names = extractor.extract_everything(defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER, folder)
    return folder, names


def test_extraction_writes_the_files_the_definitions_name_and_they_match_the_committed_prompt_files(extracted_prompts):
    folder, names = extracted_prompts
    assert len(names) == 42  # 3 facets x (positive + negative prompts + 2 judge templates), calibrated: 10 negative and a stated-confidence template
    for key in NEW_FACET_KEYS:
        definition = definition_of(key)
        for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE):
            files = sorted(folder.glob(definition.system_prompt_file_pattern(pole)))
            assert len(files) == (10 if key == "calibrated_consideration_content_controlled" and pole == defs.POLE_NEGATIVE else 5)
        tags = sorted(path.stem.split("__tag_")[1] for path in folder.glob(definition.system_prompt_file_pattern(defs.POLE_NEGATIVE)))
        assert tuple(tags) == definition.negative_tag_slugs
        assert (folder / definition.facet_score_template_file).exists() and (folder / definition.content_check_template_file).exists()
    for name in names:  # the committed files are what extraction writes (the drafts are the source)
        assert (folder / name).read_text() == (defs.PROMPT_FOLDER / name).read_text(), name


@pytest.mark.parametrize("facet_key", NEW_FACET_KEYS)
def test_each_system_prompt_holds_each_content_slot_once_and_both_poles_use_the_same_slots(extracted_prompts, facet_key):
    folder, _ = extracted_prompts
    definition = definition_of(facet_key)
    slots = [slot for slot in definition.question_label_key_by_slot if slot != "evidence_band"]
    for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE):
        for path in folder.glob(definition.system_prompt_file_pattern(pole)):
            text = path.read_text()
            assert all(text.count("{" + slot + "}") == 1 for slot in slots), path.name
            assert sorted(set(extractor.slots_in(text))) == sorted("{" + slot + "}" for slot in slots)
            assert "300 words" not in text  # the length suffix is appended by the converter


def test_judge_templates_keep_the_scale_direction_of_the_draft_100_is_the_positive_pole(extracted_prompts):
    folder, _ = extracted_prompts
    expectations = {"cares_about_welfare": ("100 means the answer treats the harm to the animals as a cost in its own right", "0 means"),
                    "calibrated_consideration": ("100 means the answer's claims match", "0 means the answer's claims do not match it, in either direction"),
                    "non_intrusive": ("100 means the answer gives the help the user asked for in full", "0 means the answer declines")}
    for facet_name, (hundred, zero) in expectations.items():
        text = (folder / f"judge_facet_score_prompt_template__{facet_name}.txt").read_text()
        assert hundred in text and zero in text and text.index(hundred) < text.index(zero)
    check = (folder / "judge_content_check_prompt_template__calibrated_consideration.txt").read_text()
    assert "{evidence_content}" in check and "{welfare_point}" not in check
    assert "{welfare_point}" in (folder / "judge_content_check_prompt_template__non_intrusive.txt").read_text()
    assert "{welfare_content}" in (folder / "judge_content_check_prompt_template__cares_about_welfare.txt").read_text()


def test_extraction_fails_loudly_on_an_unknown_tag_a_missing_slot_and_a_missing_row(tmp_path):
    shutil.copy(defs.PROMPT_FOLDER / "judge_content_check_prompt_template.txt", tmp_path)
    def with_edit(file_name, old, new):
        folder = tmp_path / file_name
        folder.mkdir()
        for path in defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER.glob("*.md"):
            text = path.read_text()
            if path.name.startswith("content-controlled-facet"):
                assert old in text
                text = text.replace(old, new, 1)
            (folder / path.name).write_text(text)
        return folder
    with pytest.raises(SystemExit, match="unknown tag"):
        extractor.extract_everything(with_edit("unknown_tag", "| 1 | I |", "| 1 | Q |"), tmp_path)
    with pytest.raises(SystemExit, match="expected exactly one"):
        extractor.extract_everything(with_edit("missing_slot", "The information: {welfare_content} |", "The information: nothing |"), tmp_path)
    with pytest.raises(SystemExit, match="differ from the definition|expected 1 to 5|unknown tag"):
        extractor.extract_everything(with_edit("changed_tag", "| 3 | L |", "| 3 | I |"), tmp_path)


# ---------------------------------------------------------------------------------------------------
# Per-question content, parsed by structure
# ---------------------------------------------------------------------------------------------------

def write_table(path, header, rows):
    lines = ["# Draft", "", "Some notes.", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines += ["", "## Flags", "- text"]
    Path(path).write_text("\n".join(lines) + "\n")


def test_content_table_is_found_by_header_text_so_column_order_extra_columns_and_new_wording_do_not_matter(tmp_path):
    columns = {"evidence band": "evidence_band", "evidence content": "evidence_content", "recommendation": "recommendation"}
    write_table(tmp_path / "a.md", ["Id", "Evidence band", "Evidence content", "Recommendation"], [["A1", "Weak", "E one.", "R one."]])
    write_table(tmp_path / "b.md", ["Id", "Notes", "Recommendation", "Evidence content", "Evidence band"],
                [["A1", "anything", "Changed recommendation.", "Entirely new wording.", "Strong"]])
    assert converter.read_content_table(tmp_path / "a.md", columns) == {"A1": {"evidence_band": "Weak", "evidence_content": "E one.", "recommendation": "R one."}}
    assert converter.read_content_table(tmp_path / "b.md", columns)["A1"] == {
        "evidence_band": "Strong", "evidence_content": "Entirely new wording.", "recommendation": "Changed recommendation."}
    write_table(tmp_path / "missing_column.md", ["Id", "Evidence band", "Evidence content"], [["A1", "x", "y"]])
    with pytest.raises(SystemExit, match="lacks the column"):
        converter.read_content_table(tmp_path / "missing_column.md", columns)
    write_table(tmp_path / "empty_cell.md", ["Id", "Evidence band", "Evidence content", "Recommendation"], [["A1", "x", "", "z"]])
    with pytest.raises(SystemExit, match="empty"):
        converter.read_content_table(tmp_path / "empty_cell.md", columns)
    write_table(tmp_path / "duplicate.md", ["Id", "Evidence band", "Evidence content", "Recommendation"],
                [["A1", "x", "y", "z"], ["A1", "x", "y", "z"]])
    with pytest.raises(SystemExit, match="duplicate id"):
        converter.read_content_table(tmp_path / "duplicate.md", columns)


def edited_question_drafts(tmp_path, file_prefix, edit, label):
    folder = tmp_path / ("drafts_" + label)
    folder.mkdir()
    for path in defs.QUESTION_DRAFT_FOLDER.glob("*.md"):
        text = path.read_text()
        if path.name.startswith(file_prefix):
            text = edit(text)
        (folder / path.name).write_text(text)
    return folder


def test_conversion_fails_when_a_question_has_no_content_or_content_has_no_question(tmp_path):
    prefix = converter.CONTENT_DRAFT_PREFIX_BY_FACET_KEY["cares_about_welfare_content_controlled"]
    drop_row = lambda text: "\n".join(line for line in text.splitlines() if not line.startswith("| B7 |"))  # noqa: E731
    with pytest.raises(SystemExit, match="B7.*has no content"):
        converter.convert_everything(edited_question_drafts(tmp_path, prefix, drop_row, "drop_row"), defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER,
                                     tmp_path / "prompts", tmp_path / "out")
    add_row = lambda text: text.replace("## Flags", "| Z99 | Extra content. |\n\n## Flags", 1)  # noqa: E731
    with pytest.raises(SystemExit, match="match no question"):
        converter.convert_everything(edited_question_drafts(tmp_path, prefix, add_row, "add_row"), defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER,
                                     tmp_path / "prompts", tmp_path / "out2")


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("converted_content_controlled")
    converter.convert_everything(defs.QUESTION_DRAFT_FOLDER, defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER, root / "prompts",
                                 root / "converted_inputs_2026-01-01_00-00", "2026-01-01_00-00")
    return root


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def converted_questions_and_prompts(converted):
    folder = converted / "converted_inputs_2026-01-01_00-00"
    return (read_records(folder / "questions_2026-01-01_00-00.jsonl"), read_records(folder / "system_prompts_2026-01-01_00-00.jsonl"))


@pytest.mark.parametrize("facet_key", NEW_FACET_KEYS)
def test_conversion_gives_the_expected_questions_content_labels_and_five_plus_five_tagged_prompts(converted, facet_key):
    definition = definition_of(facet_key)
    questions, prompts = converted_questions_and_prompts(converted)
    mine = [q for q in questions if q["facet"] == facet_key]
    assert {half: sum(1 for q in mine if q["half"] == half) for half in defs.ALL_HALVES} == definition.questions_per_half
    expected_labels = {label for label in definition.question_label_key_by_slot.values()}
    assert all(expected_labels <= set(q["labels"]) and all(str(q["labels"][label]).strip() for label in expected_labels) for q in mine)
    assert all(q["question_id"].startswith(facet_key + "__") for q in mine)
    mine_prompts = [p for p in prompts if p["facet"] == facet_key]
    negative_count = 10 if facet_key == "calibrated_consideration_content_controlled" else 5
    assert sorted(p["pole"] for p in mine_prompts) == ["negative"] * negative_count + ["positive"] * 5
    assert tuple(sorted(p["tag"] for p in mine_prompts if p["pole"] == "negative")) == definition.negative_tag_slugs
    assert all(p["tag"] is None for p in mine_prompts if p["pole"] == "positive")
    assert all(p["system_prompt"].endswith("Keep your answer under 300 words.") for p in mine_prompts)


def test_non_intrusive_uses_the_same_26_questions_and_welfare_points_as_moralizing_delivery(converted):
    questions, _ = converted_questions_and_prompts(converted)
    delivery = {q["labels"]["welfare_point"]: q["question_text"] for q in questions if q["facet"] == "moralizing_delivery"}
    non_intrusive = {q["labels"]["welfare_point"]: q["question_text"] for q in questions if q["facet"] == "non_intrusive_content_controlled"}
    assert delivery == non_intrusive and len(non_intrusive) == 26


def test_the_existing_four_facets_and_moralizing_delivery_are_still_converted_unchanged(converted):
    questions, prompts = converted_questions_and_prompts(converted)
    counts = {key: sum(1 for q in questions if q["facet"] == key) for key in ("cares_about_welfare", "calibrated_consideration",
                                                                               "non_intrusive", "moralizing_delivery")}
    assert counts == {"cares_about_welfare": 52, "calibrated_consideration": sum(definition_of("calibrated_consideration").questions_per_half.values()),
                      "non_intrusive": 52, "moralizing_delivery": 26}
    assert sum(1 for p in prompts if p["facet"] == "moralizing_delivery") == 10


# ---------------------------------------------------------------------------------------------------
# Run folders: slot filling, identical content in both poles, activation recording
# ---------------------------------------------------------------------------------------------------

@pytest.fixture()
def build_run_folder(converted, tmp_path, monkeypatch):
    monkeypatch.setattr(defs, "RUNS_ROOT", tmp_path / "runs")

    def build(facet_key, *extra):
        return builder.build_run_folder(builder.parse_arguments(["--mode", definition_of(facet_key).extraction_mode, *extra]),
                                        converted / "converted_inputs_2026-01-01_00-00")
    return build


@pytest.mark.parametrize("facet_key", NEW_FACET_KEYS)
def test_run_folder_fills_the_same_content_into_both_poles_and_leaves_no_placeholder(build_run_folder, facet_key):
    definition = definition_of(facet_key)
    run_folder, configuration = build_run_folder(facet_key)
    inputs = run_folder / defs.INPUTS_FOLDER_NAME
    jobs = read_records(defs.find_latest_file(inputs, "generation_jobs", "jsonl"))
    questions = {q["question_id"]: q for q in read_records(defs.find_latest_file(inputs, "questions", "jsonl"))}
    assert configuration["facets"] == [facet_key] and configuration["mode"] == definition.extraction_mode
    assert configuration["number_of_generation_jobs"] == sum(definition.questions_per_half.values()) * 10
    assert configuration["activation_pass_system_prompt"] == "Keep your answer under 300 words."
    assert configuration["record_welfare_passage_mean_activations"] is definition.uses_welfare_passage
    assert configuration["main_activation_mean_space"] == definition.main_space
    content_slots = [slot for slot in definition.question_label_key_by_slot if slot != "evidence_band"]
    for job in jobs:
        assert "{" not in job["system_prompt"]
        for slot in content_slots:  # every pole of a question carries the question's own content, verbatim
            assert questions[job["question_id"]]["labels"][definition.question_label_key_by_slot[slot]] in job["system_prompt"]
    for question_id in questions:
        poles = {job["pole"]: job["system_prompt"] for job in jobs if job["question_id"] == question_id}
        assert set(poles) == {"positive", "negative"}


def test_the_judge_only_content_label_evidence_band_never_reaches_the_model(build_run_folder):
    run_folder, _ = build_run_folder("calibrated_consideration_content_controlled")
    inputs = run_folder / defs.INPUTS_FOLDER_NAME
    jobs = read_records(defs.find_latest_file(inputs, "generation_jobs", "jsonl"))
    questions = {q["question_id"]: q for q in read_records(defs.find_latest_file(inputs, "questions", "jsonl"))}
    for job in jobs:
        band = questions[job["question_id"]]["labels"]["evidence_band"]
        assert f"rated: {band}" not in job["system_prompt"] and "evidence_band" not in job["system_prompt"]


def test_slot_filling_fails_loudly_on_missing_content_or_a_leftover_placeholder():
    prompt = {"system_prompt": "Use {welfare_content} once. Also {surprise}. Keep it short.", "system_prompt_id": "p1"}
    question = {"facet": "cares_about_welfare_content_controlled", "question_id": "q1", "labels": {"welfare_content": "Pigs hurt."}}
    with pytest.raises(ValueError, match="left unfilled"):
        builder.fill_content_slots(prompt, question)
    with pytest.raises(ValueError, match="has no 'welfare_content'"):
        builder.fill_content_slots({"system_prompt": "Use {welfare_content}.", "system_prompt_id": "p1"}, {**question, "labels": {}})
    assert builder.fill_content_slots({"system_prompt": "Use {welfare_content}.", "system_prompt_id": "p1"}, question) == "Use Pigs hurt.."


# ---------------------------------------------------------------------------------------------------
# Judge requests
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("facet_key", NEW_FACET_KEYS)
def test_judge_requests_for_each_new_facet_are_facet_score_content_check_and_coherence_with_every_slot_filled(converted, facet_key):
    definition = definition_of(facet_key)
    questions, _ = converted_questions_and_prompts(converted)
    question = next(q for q in questions if q["facet"] == facet_key)
    answer = {"answer_id": "a1", "facet": facet_key, "question_id": question["question_id"], "answer_text": "The ANSWER body.", "version": None}
    requests = judge.build_judge_requests_for_answers([answer], {question["question_id"]: question}, {})
    expected_kinds = [definition.facet_score_kind, "content_check"] + (["stated_confidence"] if definition.has_confidence_control else []) + ["coherence"]
    assert [r.kind for r in requests] == expected_kinds
    assert definition.facet_score_kind == "content_controlled_facet_score"
    for request in requests:
        assert "{" not in request.prompt_text or request.kind == "content_check"  # the content check shows a JSON example
        assert "The ANSWER body." in request.prompt_text and question["question_text"] in request.prompt_text
    template_slots = [slot[1:-1] for slot in extractor.slots_in((defs.PROMPT_FOLDER / definition.facet_score_template_file).read_text())]
    for slot in template_slots:
        if slot in definition.question_label_key_by_slot:
            assert question["labels"][definition.question_label_key_by_slot[slot]] in requests[0].prompt_text
    assert judge.uses_scoring_reasoning(requests[0]) and judge.KIND_CONTENT_CONTROLLED_FACET_SCORE in judge.KINDS_RETRIED_WHEN_DISCARDED


def test_judge_prompt_rendering_refuses_a_question_without_the_content():
    definition = definition_of("calibrated_consideration")
    question = {"question_id": "q1", "question_text": "?", "labels": {"evidence_content": "E."}}  # no evidence_band
    with pytest.raises(ValueError, match="evidence_band"):
        judge.render_content_controlled_prompt(definition.facet_score_kind, definition, question, "answer")
    with pytest.raises(ValueError, match="not a content-controlled prompt kind"):
        judge.render_content_controlled_prompt("coherence", definition, question, "answer")


def test_moralizing_delivery_keeps_its_delivery_score_kind(converted):
    questions, _ = converted_questions_and_prompts(converted)
    question = next(q for q in questions if q["facet"] == "moralizing_delivery")
    answer = {"answer_id": "a1", "facet": "moralizing_delivery", "question_id": question["question_id"], "answer_text": "x", "version": None}
    assert [r.kind for r in judge.build_judge_requests_for_answers([answer], {question["question_id"]: question}, {})] == [
        "delivery_score", "content_check", "coherence"]


# ---------------------------------------------------------------------------------------------------
# The answer filter
# ---------------------------------------------------------------------------------------------------

def good_scores(kind, score):
    return {kind: score, "coherence": 90.0, "content_check": {"makes_the_point": True, "separate_welfare_mentions": 1,
                                                              "adds_welfare_facts": False, "completes_the_task": True,
                                                              "welfare_passage": "x"}}


GOOD_SPAN = {"character_start": 0, "character_end": 5}


@pytest.mark.parametrize("facet_key", ALL_FACET_KEYS)
def test_filter_keeps_a_positive_answer_only_when_its_score_is_high_and_a_negative_answer_only_when_it_is_low(facet_key):
    definition = definition_of(facet_key)
    kind = definition.facet_score_kind
    keyword = dict(facet_score_kind=kind, uses_task_completion_in_filter=definition.uses_task_completion_in_filter,
                   uses_welfare_passage=definition.uses_welfare_passage)
    check = lambda pole, score: compute.failed_answer_conditions(pole, good_scores(kind, score), GOOD_SPAN, 20, 6, THRESHOLDS, **keyword)  # noqa: E731
    assert check("positive", 90.0) == [] and check("negative", 10.0) == []
    wrong_side_positive, wrong_side_negative = check("positive", 20.0), check("negative", 80.0)
    assert len(wrong_side_positive) == 1 and "positive" in wrong_side_positive[0] and "below_minimum" in wrong_side_positive[0]
    assert len(wrong_side_negative) == 1 and "negative" in wrong_side_negative[0] and "above_maximum" in wrong_side_negative[0]
    assert check("positive", 50.0) and check("negative", 50.0)  # a middling score keeps neither pole
    assert (kind == "delivery_score") == any("delivery" in name for name in wrong_side_positive + wrong_side_negative)


@pytest.mark.parametrize("facet_key,required", [("cares_about_welfare_content_controlled", True),
                                                ("calibrated_consideration_content_controlled", True),
                                                ("moralizing_delivery", True), ("non_intrusive_content_controlled", False)])
def test_the_filter_requires_completing_the_task_for_every_facet_except_non_intrusive(facet_key, required):
    definition = definition_of(facet_key)
    scores = good_scores(definition.facet_score_kind, 10.0)
    scores["content_check"]["completes_the_task"] = False
    failed = compute.failed_answer_conditions(
        "negative", scores, GOOD_SPAN, 20, 6, THRESHOLDS, facet_score_kind=definition.facet_score_kind,
        uses_task_completion_in_filter=definition.uses_task_completion_in_filter, uses_welfare_passage=definition.uses_welfare_passage)
    assert (compute.CONDITION_DOES_NOT_COMPLETE_THE_TASK in failed) is required


def test_without_the_welfare_passage_the_span_and_passage_mean_conditions_do_not_apply():
    scores = good_scores("content_controlled_facet_score", 90.0)
    kind = dict(facet_score_kind="content_controlled_facet_score", uses_task_completion_in_filter=False)
    assert compute.failed_answer_conditions("positive", scores, None, 20, 0, THRESHOLDS, uses_welfare_passage=False, **kind) == []
    failed = compute.failed_answer_conditions("positive", scores, None, 20, 0, THRESHOLDS, uses_welfare_passage=True, **kind)
    assert compute.CONDITION_NO_SPAN_FOUND in failed and compute.CONDITION_NO_PASSAGE_MEAN in failed


def conditions_with_mentions(mentions, thresholds):
    scores = good_scores("content_controlled_facet_score", 90.0)
    scores["content_check"]["separate_welfare_mentions"] = mentions
    return compute.failed_answer_conditions("positive", scores, GOOD_SPAN, 20, 6, thresholds, facet_score_kind="content_controlled_facet_score")


def test_by_default_only_one_separate_welfare_mention_is_allowed():
    arguments = compute.parse_arguments(["--run-folder", "x", "--facet", "cares_about_welfare_content_controlled"])
    assert arguments.required_separate_welfare_mentions is None and arguments.allowed_separate_welfare_mentions is None
    assert conditions_with_mentions(1, THRESHOLDS) == []
    for mentions in (0, 2, "3 or more"):
        assert conditions_with_mentions(mentions, THRESHOLDS) == [compute.CONDITION_MENTION_COUNT_WRONG]


def test_allowing_one_and_two_mentions_keeps_a_two_mention_answer_and_drops_a_three_or_more_answer():
    assert compute.parse_allowed_separate_welfare_mentions("2,1") == [1, 2]
    assert compute.parse_allowed_separate_welfare_mentions("0,3_or_more") == [0, "3 or more"]
    with pytest.raises(ValueError):
        compute.parse_allowed_separate_welfare_mentions("1,4")
    allowed = {key: value for key, value in THRESHOLDS.items() if key != "required_separate_welfare_mentions"}
    allowed["allowed_separate_welfare_mentions"] = compute.parse_allowed_separate_welfare_mentions("1,2")
    assert conditions_with_mentions(1, allowed) == [] and conditions_with_mentions(2, allowed) == []
    assert conditions_with_mentions("3 or more", allowed) == [compute.CONDITION_MENTION_COUNT_WRONG]
    assert conditions_with_mentions(0, allowed) == [compute.CONDITION_MENTION_COUNT_WRONG]


def test_the_required_and_allowed_mention_options_cannot_be_given_together():
    base = ["--run-folder", "x", "--facet", "cares_about_welfare_content_controlled"]
    with pytest.raises(SystemExit):
        compute.parse_arguments(base + ["--required-separate-welfare-mentions", "1", "--allowed-separate-welfare-mentions", "1,2"])
    assert compute.parse_arguments(base + ["--allowed-separate-welfare-mentions", "1,2"]).allowed_separate_welfare_mentions == [1, 2]
    assert compute.parse_arguments(base + ["--required-separate-welfare-mentions", "2"]).required_separate_welfare_mentions == 2


def test_the_mention_distribution_by_pole_is_counted_from_the_content_checks():
    answers = [{"answer_id": "a", "pole": "positive"}, {"answer_id": "b", "pole": "positive"},
               {"answer_id": "c", "pole": "negative"}, {"answer_id": "d", "pole": "negative"}]
    judgements = {identifier: {"content_check": {"separate_welfare_mentions": mentions}}
                  for identifier, mentions in (("a", 1), ("b", 2), ("c", 1), ("d", "3 or more"))}
    assert compute.count_separate_welfare_mentions_by_pole(answers, judgements) == {
        "positive": {"0": 0, "1": 1, "2": 1, "3 or more": 0}, "negative": {"0": 0, "1": 1, "2": 0, "3 or more": 1}}


def test_the_old_threshold_keys_still_work_for_moralizing_delivery():
    old_thresholds = {"positive_delivery_score_minimum": 65.0, "negative_delivery_score_maximum": 35.0, "coherence_minimum": 50.0,
                      "maximum_passage_length_ratio": 1.5, "required_separate_welfare_mentions": 1}
    assert compute.failed_answer_conditions("positive", good_scores("delivery_score", 90.0), GOOD_SPAN, 20, 6, old_thresholds) == []
    assert compute.failed_answer_conditions("positive", good_scores("delivery_score", 20.0), GOOD_SPAN, 20, 6, old_thresholds)


# ---------------------------------------------------------------------------------------------------
# The vector computation on synthetic runs
# ---------------------------------------------------------------------------------------------------

def run_compute(folder, facet_key, **threshold_overrides):
    return compute.compute_all(folder, {**THRESHOLDS, **threshold_overrides}, "test", bootstrap_resamples=10, minimum_layer_index=1,
                               seed=0, say=lambda *_: None, facet_key=facet_key)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def vector_from(metadata_path, metadata, space, variant="raw"):
    return np.load(Path(metadata_path).parent / metadata["vector_files"][f"{space}_{variant}"])


@pytest.mark.parametrize("facet_key", ["cares_about_welfare_content_controlled", "calibrated_consideration_content_controlled"])
def test_passage_facets_main_vector_follows_the_passage_means_and_points_from_negative_to_positive(tmp_path, facet_key):
    folder, passage_direction, whole_direction = synthetic.build_synthetic_content_controlled_run(
        tmp_path / "run", facet_key, layers=6, hidden=32)
    metadata, metadata_path = run_compute(folder, facet_key)
    layer = metadata["layer_choice"]["layer_by_facet"][facet_key]
    passage_vector = vector_from(metadata_path, metadata, "passage_mean")
    whole_vector = vector_from(metadata_path, metadata, "whole_answer_mean")
    assert cosine(passage_vector[layer], passage_direction) > 0.99  # positive pole minus negative pole: the +signal direction
    assert cosine(passage_vector[layer], whole_direction) < 0.2  # the main vector is NOT the whole-answer one
    assert cosine(whole_vector[layer], whole_direction) > 0.99
    assert metadata["layer_choice"]["based_on"].startswith("held-out classification accuracy of the passage-mean vector")
    assert metadata["main_space"] == "passage_mean" and metadata["high_score_pole"] == "positive"
    assert "100 =" in metadata["facet_score_scale"] and metadata["spaces"] == ["passage_mean", "whole_answer_mean"]
    assert "persona_vector_" + facet_key + "_passage_mean_" in " ".join(p.name for p in metadata_path.parent.iterdir())
    for key in ("passage_mean_raw", "passage_mean_top_4_largest_mean_dimensions_zeroed", "whole_answer_mean_raw",
                "passage_mean_standardised_by_per_dimension_variance"):
        assert key in metadata["vector_files"]
    entry = metadata["per_facet"][facet_key]
    assert entry["variant_summary_at_chosen_and_reported_layers"]["chosen_layer"] == layer
    assert set(entry["negative_tag_kept_items"]) == set(definition_of(facet_key).negative_tag_slugs)


def test_positive_pole_answers_project_higher_than_negative_pole_answers_onto_the_vector(tmp_path):
    facet_key = "cares_about_welfare_content_controlled"
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", facet_key, layers=6, hidden=32)
    metadata, metadata_path = run_compute(folder, facet_key)
    layer = metadata["layer_choice"]["layer_by_facet"][facet_key]
    anchors = json.loads(next(metadata_path.parent.glob("mean_projection_onto_each_variant_passage_mean_*.json")).read_text())["variants"]["raw"]
    assert anchors["mean_projection_positive_pole_answers_per_layer"][layer] > anchors["mean_projection_negative_pole_answers_per_layer"][layer]
    assert metadata["per_facet"][facet_key]["variant_summary_at_chosen_and_reported_layers"]["by_space"]["passage_mean"]["raw"][
        "held_out_accuracy"][str(layer)] > 0.9


def test_non_intrusive_uses_the_whole_answer_mean_even_when_passage_means_are_stored_and_reads_no_spans(tmp_path):
    facet_key = "non_intrusive_content_controlled"
    folder, passage_direction, whole_direction = synthetic.build_synthetic_content_controlled_run(
        tmp_path / "run", facet_key, layers=6, hidden=32, store_passage_means_anyway=True)
    assert not list((folder / defs.INPUTS_FOLDER_NAME).glob("welfare_passage_character_spans_*.jsonl"))
    metadata, metadata_path = run_compute(folder, facet_key)
    layer = metadata["layer_choice"]["layer_by_facet"][facet_key]
    vector = vector_from(metadata_path, metadata, "whole_answer_mean")
    assert cosine(vector[layer], whole_direction) > 0.99 and cosine(vector[layer], passage_direction) < 0.2
    assert metadata["main_space"] == "whole_answer_mean" and metadata["spaces"] == ["whole_answer_mean"]
    assert not any(key.startswith("passage_mean") for key in metadata["vector_files"])
    assert metadata["spans_file"] is None and metadata["completes_the_task_used_in_filter"] is False
    assert "whole-answer-mean" in metadata["layer_choice"]["based_on"]


def test_non_intrusive_vector_does_not_need_passage_activations_in_the_shards(tmp_path):
    facet_key = "non_intrusive_content_controlled"
    folder, _, whole_direction = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", facet_key, layers=6, hidden=32)
    shard = next((folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME).glob("*.npz"))
    with np.load(shard) as arrays:
        assert "passage_activations" not in arrays.files
    run_compute(folder, facet_key)


def test_non_intrusive_keeps_incomplete_help_items_but_the_passage_facets_drop_them(tmp_path):
    incomplete_negative = lambda answer_id, pole: pole == "negative" and "rewording_01" in answer_id  # noqa: E731
    kept = {}
    for facet_key in ("non_intrusive_content_controlled", "cares_about_welfare_content_controlled"):
        folder, _, _ = synthetic.build_synthetic_content_controlled_run(
            tmp_path / facet_key, facet_key, layers=6, hidden=32,
            completes_the_task_function=lambda answer_id, pole: not incomplete_negative(answer_id, pole))
        metadata, _ = run_compute(folder, facet_key)
        kept[facet_key] = metadata["per_facet"][facet_key]["filter_counts"]
    assert kept["non_intrusive_content_controlled"]["items_kept"] == kept["non_intrusive_content_controlled"]["items_total"]
    assert kept["cares_about_welfare_content_controlled"]["items_kept"] < kept["cares_about_welfare_content_controlled"]["items_total"]
    assert "negative:does_not_complete_the_task" in kept["cares_about_welfare_content_controlled"]["items_failing_each_condition"]


@pytest.mark.parametrize("facet_key", ["cares_about_welfare_content_controlled", "non_intrusive_content_controlled"])
def test_a_run_where_the_judge_scores_are_on_the_wrong_side_keeps_no_item(tmp_path, facet_key):
    """Positive-pole answers scoring 20 and negative-pole answers scoring 80 (the scale read backwards) must never survive."""
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(
        tmp_path / "run", facet_key, layers=6, hidden=32, score_function=lambda answer_id, pole: 20.0 if pole == "positive" else 80.0)
    with pytest.raises(SystemExit, match="No item survived the filter"):
        run_compute(folder, facet_key)


def test_items_where_only_one_pole_is_on_the_wrong_side_are_dropped_and_the_rest_kept(tmp_path):
    facet_key = "calibrated_consideration_content_controlled"
    wrong = lambda answer_id: "rewording_02" in answer_id and "__A__" in answer_id  # noqa: E731
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(
        tmp_path / "run", facet_key, layers=6, hidden=32,
        score_function=lambda answer_id, pole: (85.0 if pole == "positive" else 10.0) if not wrong(answer_id) else (20.0 if pole == "positive" else 80.0))
    metadata, _ = run_compute(folder, facet_key)
    counts = metadata["per_facet"][facet_key]["filter_counts"]["items_failing_each_condition"]
    assert counts.get("positive:positive_facet_score_below_minimum", 0) > 0 and counts.get("negative:negative_facet_score_above_maximum", 0) > 0
    assert metadata["per_facet"][facet_key]["filter_counts"]["items_kept"] > 0


def test_moralizing_delivery_through_the_generic_function_keeps_its_names_and_thresholds(tmp_path):
    folder, direction = synthetic.build_synthetic_moralizing_delivery_run(tmp_path / "run", layers=6, hidden=32)
    metadata, metadata_path = compute.compute_all(folder, {"positive_delivery_score_minimum": 65.0, "negative_delivery_score_maximum": 35.0,
                                                           "coherence_minimum": 50.0, "maximum_passage_length_ratio": 1.5,
                                                           "required_separate_welfare_mentions": 1}, "test", bootstrap_resamples=5,
                                                  say=lambda *_: None)
    assert "persona_vector_moralizing_delivery_passage_mean_" in " ".join(p.name for p in metadata_path.parent.iterdir())
    assert set(metadata["per_facet"]) == {"moralizing_delivery"} and metadata["layer_choice"]["layer_by_facet"].keys() == {"moralizing_delivery"}


def test_the_old_module_name_is_the_same_module_as_the_new_one():
    from persona_vector_projection import compute_moralizing_delivery_vectors_from_passage_activations as old_name
    assert old_name is compute
    from persona_vector_projection import rehearse_moralizing_delivery_chain_locally_with_tiny_model_and_mocked_judge as old_rehearsal
    from persona_vector_projection import rehearse_content_controlled_facet_extraction_chain_locally_with_tiny_model_and_mocked_judge as new_rehearsal
    assert old_rehearsal is new_rehearsal


def test_cost_estimate_prints_the_scale_direction_and_has_no_span_step_cost_for_non_intrusive(converted):
    from persona_vector_projection import estimate_extraction_cost_ceilings as estimator
    assumptions = {"samples_per_prompt": 5, "expected_tokens_per_answer": 450, "ceiling_tokens_per_answer": 1000,
                   "expected_generation_tokens_per_second": 3000, "ceiling_generation_tokens_per_second": 1200,
                   "expected_activation_tokens_per_second": 12000, "ceiling_activation_tokens_per_second": 3000,
                   "setup_minutes_expected": 20, "setup_minutes_ceiling": 45, "transfer_minutes_expected": 10,
                   "transfer_minutes_ceiling": 25, "idle_minutes_between_stages_expected": 45,
                   "idle_minutes_between_stages_ceiling": 120, "activation_stage_setup_minutes_expected": 10,
                   "activation_stage_setup_minutes_ceiling": 25}
    folder = converted / "converted_inputs_2026-01-01_00-00"
    results = {key: estimator.estimate_for_mode(definition_of(key).extraction_mode, None, folder, assumptions, 0) for key in ALL_FACET_KEYS}
    assert "100 =" in results["non_intrusive_content_controlled"]["facet_score_scale"]
    assert results["cares_about_welfare_content_controlled"]["ceiling"]["judge"]["calls_by_kind"]["content_controlled_facet_score"] == 2600
    # non-intrusive stores whole-answer means only, so half the activations of a passage facet with as many answers (1300 vs 1300)
    assert results["non_intrusive_content_controlled"]["stored_activation_gigabytes_fp16"] * 2 == pytest.approx(
        results["moralizing_delivery"]["stored_activation_gigabytes_fp16"])


# ---------------------------------------------------------------------------------------------------
# Cross-half accuracy: the vector from half A's items, evaluated on half B's questions only
# ---------------------------------------------------------------------------------------------------

class FakeActivationStore:
    """Stands in for the activation shards: `stream_rows(ids)` yields (answer id, row of shape (layers, hidden))."""
    def __init__(self, rows):
        self.rows = rows

    def stream_rows(self, answer_ids):
        for answer_id in answer_ids:
            yield answer_id, self.rows[answer_id]


def test_cross_half_accuracy_uses_half_a_vector_and_half_a_threshold_and_scores_half_b_answers_only():
    positive, negative = defs.POLE_POSITIVE, defs.POLE_NEGATIVE
    # one layer, two dimensions. Half A separates along dimension 0 around 0. Half B's answers all sit on the positive side of that
    # threshold, so with half A's threshold the two B negatives are classified as positive; B's own midpoint would give 1.0.
    rows = {"a_pos_1": [[+1.0, 0.0]], "a_pos_2": [[+1.0, 0.0]], "a_neg_1": [[-1.0, 0.0]], "a_neg_2": [[-1.0, 0.0]],
            "b_pos_1": [[+0.5, 0.0]], "b_pos_2": [[+0.5, 0.0]], "b_neg_1": [[+0.2, 0.0]], "b_neg_2": [[+0.2, 0.0]]}
    store = FakeActivationStore({key: np.array(value, dtype=np.float32) for key, value in rows.items()})
    ids = {"A": {positive: ["a_pos_1", "a_pos_2"], negative: ["a_neg_1", "a_neg_2"]},
           "B": {positive: ["b_pos_1", "b_pos_2"], negative: ["b_neg_1", "b_neg_2"]}}
    half_a_vector = np.array([[2.0, 0.0]], dtype=np.float32)       # positive minus negative of half A: along dimension 0
    result = compute.cross_half_accuracy(store, half_a_vector, ids, "A", "B")
    assert result["accuracy_per_layer"] == [0.5]                    # 2 of 4 B answers: half A's threshold (0), only B answers counted
    assert result["fit_half"] == "A" and result["evaluation_half"] == "B"
    assert result["answers_in_evaluation_half"] == {positive: 2, negative: 2}
    # a vector along dimension 1 separates nothing in half A (all projections 0, threshold 0): every B answer projects to 0, which is not
    # above the threshold, so positives are wrong and negatives right: 0.5 again, from the passed-in vector and not from any other
    assert compute.cross_half_accuracy(store, np.array([[0.0, 3.0]], dtype=np.float32), ids, "A", "B")["accuracy_per_layer"] == [0.5]
    # the reverse direction scores half A's answers with half B's threshold (midpoint 0.35): A positives (+1) above, A negatives (-1) below
    reverse = compute.cross_half_accuracy(store, np.array([[0.3, 0.0]], dtype=np.float32), ids, "B", "A")
    assert reverse["fit_half"] == "B" and reverse["evaluation_half"] == "A" and reverse["accuracy_per_layer"] == [1.0]
    # missing pieces give None, never a number
    assert compute.cross_half_accuracy(store, None, ids, "A", "B") is None
    assert compute.cross_half_accuracy(store, half_a_vector, {**ids, "B": {positive: ["b_pos_1"], negative: []}}, "A", "B") is None


@pytest.mark.parametrize("facet_key", ALL_FACET_KEYS)
def test_vector_metadata_records_the_mention_distribution_by_pole_beside_the_answer_condition_counts(tmp_path, facet_key):
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", facet_key, layers=6, hidden=32)
    metadata, _ = run_compute(folder, facet_key)
    entry = metadata["per_facet"][facet_key]
    assert "answer_condition_counts_by_pole_and_tag" in entry
    passing = entry["separate_welfare_mentions_among_answers_passing_every_condition_by_pole"]
    kept = entry["separate_welfare_mentions_among_items_kept_by_pole"]
    counts = entry["filter_counts"]
    for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE):
        assert sum(passing[pole].values()) == entry["answer_condition_counts_by_pole_and_tag"][pole]["all"]["passing_every_answer_condition"]
        assert sum(kept[pole].values()) == counts.get("items_kept_after_balancing_stated_confidence", counts["items_kept"]) and set(kept[pole]) == {"0", "1", "2", "3 or more"}
    assert metadata["filter_thresholds"]["allowed_separate_welfare_mentions"] == [1]
    assert metadata["filter_thresholds"]["required_separate_welfare_mentions"] == 1


@pytest.mark.parametrize("facet_key", ALL_FACET_KEYS)
def test_vector_metadata_records_cross_half_accuracy_and_the_training_question_total(tmp_path, facet_key):
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", facet_key, layers=6, hidden=32)
    metadata, metadata_path = run_compute(folder, facet_key)
    entry = metadata["per_facet"][facet_key]
    answers = [a for a in compute.shared_compute.load_answers(folder) if a["facet"] == facet_key]
    assert entry["training_questions_total"] == len({a["question_id"] for a in answers if a["half"] in defs.EXTRACTION_HALVES})
    assert 0 < entry["questions_kept"] <= entry["training_questions_total"]
    conditions = [json.loads(line) for line in next(metadata_path.parent.glob("answer_filter_conditions_*.jsonl")).read_text().splitlines()]
    passing_in_b = {pole: sum(1 for c in conditions if c["half"] == "B" and c["pole"] == pole and not c["failed_conditions"])
                    for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE)}
    for space in definition_of(facet_key).vector_spaces:
        counted = entry[space]["cross_half_accuracy"]["vector_from_half_A_evaluated_on_half_B"]
        assert counted["fit_half"] == "A" and counted["evaluation_half"] == "B"
        assert counted["answers_in_evaluation_half"] == passing_in_b
        assert len(counted["accuracy_per_layer"]) == 6 and all(0.0 <= value <= 1.0 for value in counted["accuracy_per_layer"])
        reverse = entry[space]["cross_half_accuracy"]["vector_from_half_B_evaluated_on_half_A_reported_only"]
        assert reverse["fit_half"] == "B" and reverse["evaluation_half"] == "A"
    layer = metadata["layer_choice"]["layer_by_facet"][facet_key]
    main_space = definition_of(facet_key).main_space
    assert entry[main_space]["cross_half_accuracy"]["vector_from_half_A_evaluated_on_half_B"]["accuracy_per_layer"][layer] > 0.9   # separable synthetic run


# ---------------------------------------------------------------------------------------------------
# Question-level pairing (--pairing-level question)
# ---------------------------------------------------------------------------------------------------

def run_compute_at_question_level(folder, facet_key, **threshold_overrides):
    return compute.compute_all(folder, {**THRESHOLDS, **threshold_overrides}, "test", bootstrap_resamples=10, minimum_layer_index=1,
                               seed=0, say=lambda *_: None, facet_key=facet_key, pairing_level="question")


def fake_answer(answer_id, pole, question_id="q1", half="A", tag=None, domain="d"):
    return {"answer_id": answer_id, "pole": pole, "question_id": question_id, "half": half, "tag": tag, "domain": domain, "facet": "f"}


def test_the_default_pairing_level_is_sample_and_the_command_line_default_matches():
    assert compute.parse_arguments(["--run-folder", "x"]).pairing_level == "sample"
    assert compute.parse_arguments(["--run-folder", "x", "--pairing-level", "question"]).pairing_level == "question"
    with pytest.raises(SystemExit):
        compute.parse_arguments(["--run-folder", "x", "--pairing-level", "domain"])


def test_the_sample_level_path_is_unchanged_by_default_and_records_no_question_level_entries(tmp_path):
    facet_key = "cares_about_welfare_content_controlled"
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", facet_key, layers=6, hidden=32)
    metadata, _ = run_compute(folder, facet_key)
    entry = metadata["per_facet"][facet_key]
    assert metadata["pairing_level"] == "sample" and "question_level_pairing" not in entry
    assert "items_kept" in entry["filter_counts"] and "separate_welfare_mentions_among_items_kept_by_pole" in entry
    assert "negatives_classified_correctly_per_layer" in next(iter(entry["passage_mean"]["variant_held_out_metrics"]["raw"]["by_negative_tag"].values()))


def test_question_level_pairing_keeps_a_question_whose_samples_do_not_line_up_between_the_poles():
    answers = [fake_answer("p1", "positive"), fake_answer("p2", "positive"), fake_answer("n3", "negative"),   # no sample of the two poles pairs
               fake_answer("p4", "positive", question_id="q2"), fake_answer("n4", "negative", question_id="q2"),
               fake_answer("p5", "positive", question_id="q3"), fake_answer("n5", "negative", question_id="q3")]
    conditions = {"p1": ["coherence_below_minimum"], "p2": [], "n3": [], "p4": [], "n4": ["adds_welfare_facts"], "p5": [], "n5": []}
    spans = {answer_id: {"character_start": 0, "character_end": length}
             for answer_id, length in {"p2": 100, "n3": 120, "p5": 100, "n5": 300}.items()}
    kept, counts = compute.classify_questions(answers, conditions, spans, THRESHOLDS)
    assert [question_id for question_id, _, _ in kept] == ["q1"]
    assert [a["answer_id"] for a in kept[0][1]] == ["p2"] and [a["answer_id"] for a in kept[0][2]] == ["n3"]
    assert counts["questions_total"] == 3 and counts["questions_kept"] == 1
    assert counts["questions_lacking_a_passing_answer_in_a_pole"] == 1      # q2: its only negative fails
    assert counts["questions_failing_passage_length_ratio"] == 1            # q3: 300 / 100 is above 1.5
    assert counts["per_question"]["q3"]["status"] == compute.QUESTION_STATUS_PASSAGE_LENGTH_RATIO
    # the same ratio check compares MEAN passage lengths of the passing answers of the two poles
    assert compute.mean_passage_lengths_are_within_ratio(100.0, 150.0, 1.5) and not compute.mean_passage_lengths_are_within_ratio(100.0, 151.0, 1.5)
    assert not compute.mean_passage_lengths_are_within_ratio(0.0, 10.0, 1.5)


def test_the_question_level_vector_is_the_mean_over_questions_of_the_per_question_mean_difference():
    rows = {"p1": [[4.0, 0.0]], "p2": [[2.0, 0.0]], "n1": [[0.0, 2.0]],                    # question q1: positives mean (3, 0), negative (0, 2)
            "p3": [[0.0, 6.0]], "n2": [[0.0, 0.0]], "n3": [[0.0, 2.0]]}                    # question q2: positive (0, 6), negatives mean (0, 1)
    store = FakeActivationStore({key: np.array(value, dtype=np.float32) for key, value in rows.items()})
    pole = lambda answer_id: "positive" if answer_id.startswith("p") else "negative"        # noqa: E731
    kept = [("q1", [fake_answer("p1", "positive"), fake_answer("p2", "positive")], [fake_answer("n1", "negative")]),
            ("q2", [fake_answer("p3", "positive", question_id="q2")], [fake_answer("n2", "negative", question_id="q2"), fake_answer("n3", "negative", question_id="q2")])]
    differences = compute.differences_for_questions(kept, store)
    assert differences["question_keys"] == ["q1", "q2"]
    np.testing.assert_allclose(differences["vectors"], [[[3.0, -2.0]], [[0.0, 5.0]]])
    vector = compute.shared_compute.domain_weighted_vector(differences)
    np.testing.assert_allclose(vector, [[1.5, 1.5]])                                       # both questions in one domain: their plain mean
    assert pole("p1") == "positive"


def test_question_level_held_out_accuracy_compares_every_positive_with_every_negative_and_weights_questions_equally():
    # one layer. q1: positives 3, 1 against negatives 2, 0 -> pairs (3>2, 3>0, 1>2 no, 1>0) = 3 of 4. q2: positive 5 against negative 5 and 1
    # -> a tie and a win: accuracy 1 of 2, area under the curve 0.75. q2 has one pair fewer than q1 times two, so equal weight differs from pooling.
    projections = {"p1": np.array([3.0]), "p2": np.array([1.0]), "n1": np.array([2.0]), "n2": np.array([0.0]),
                   "p3": np.array([5.0]), "n3": np.array([5.0]), "n4": np.array([1.0])}
    groups = [(["p1", "p2"], ["n1", "n2"]), (["p3"], ["n3", "n4"])]
    metrics = compute.question_level_pairwise_metrics(projections, groups)
    assert metrics["accuracy_per_layer"].tolist() == [pytest.approx((0.75 + 0.5) / 2)]
    assert metrics["area_under_curve_per_layer"].tolist() == [pytest.approx((0.75 + 0.75) / 2)]
    assert metrics["questions"] == 2
    assert compute.question_level_pairwise_metrics(projections, []) is None
    # only questions with a passing answer in each pole are evaluated, and a tag restricts the negatives
    passing = {"q1": {"positive": [fake_answer("p1", "positive")], "negative": [fake_answer("n1", "negative", tag="a"), fake_answer("n2", "negative", tag="b")]},
               "q2": {"positive": [], "negative": [fake_answer("n3", "negative", tag="a")]},
               "q3": {"positive": [fake_answer("p3", "positive")], "negative": [fake_answer("n4", "negative", tag="a")]}}
    assert compute.evaluation_question_groups(passing) == [(["p1"], ["n1", "n2"]), (["p3"], ["n4"])]
    assert compute.evaluation_question_groups(passing, "b") == [(["p1"], ["n2"])]


def test_question_level_cross_half_accuracy_scores_half_b_questions_with_the_half_a_vector_only():
    rows = {"b_pos": [[+0.5, 0.0]], "b_neg": [[+0.2, 0.0]], "b_pos_other": [[-1.0, 0.0]], "b_neg_other": [[+1.0, 0.0]]}
    store = FakeActivationStore({key: np.array(value, dtype=np.float32) for key, value in rows.items()})
    passing = {"A": {}, "B": {"q1": {"positive": [fake_answer("b_pos", "positive", half="B")], "negative": [fake_answer("b_neg", "negative", half="B")]},
                              "q2": {"positive": [fake_answer("b_pos_other", "positive", half="B")], "negative": [fake_answer("b_neg_other", "negative", half="B")]}}}
    result = compute.cross_half_accuracy_by_question(store, np.array([[2.0, 0.0]], dtype=np.float32), passing, "A", "B")
    assert result["accuracy_per_layer"] == [0.5] and result["questions_in_evaluation_half"] == 2
    assert result["fit_half"] == "A" and result["evaluation_half"] == "B" and result["answers_in_evaluation_half"] == {"positive": 2, "negative": 2}
    assert compute.cross_half_accuracy_by_question(store, None, passing, "A", "B") is None
    assert compute.cross_half_accuracy_by_question(store, np.array([[2.0, 0.0]], dtype=np.float32), passing, "B", "A") is None


@pytest.mark.parametrize("facet_key", ALL_FACET_KEYS)
def test_a_question_level_run_writes_the_metadata_the_downstream_readers_use(tmp_path, facet_key):
    from persona_vector_projection import build_content_controlled_facet_scoring_run_folder as scoring_builder
    from persona_vector_projection import build_content_controlled_facet_steering_run_folder as steering_builder
    definition = definition_of(facet_key)
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", facet_key, layers=6, hidden=32)
    metadata, metadata_path = run_compute_at_question_level(folder, facet_key)
    entry = metadata["per_facet"][facet_key]
    assert metadata["pairing_level"] == "question" and entry["pairing_level"] == "question"
    pairing = entry["question_level_pairing"]
    assert sum(pairing["questions_kept_by_half"].values()) == entry["questions_kept"] == entry["filter_counts"]["questions_kept"] or definition.has_confidence_control
    assert pairing["passing_answers_used_per_pole"]["positive"] >= entry["questions_kept"] and "per_question_counts" in pairing
    assert sum(sum(counts.values()) for counts in entry["separate_welfare_mentions_among_answers_used_by_pole"].values()) == sum(pairing["passing_answers_used_per_pole"].values())
    layer = metadata["layer_choice"]["layer_by_facet"][facet_key]
    for space in definition.vector_spaces:
        raw = entry[space]["variant_held_out_metrics"]["raw"]
        assert len(raw["held_out_accuracy_per_layer"]) == 6 and raw["held_out_accuracy_per_layer"][layer] > 0.9    # separable synthetic run
        assert entry[space]["cross_half_accuracy"]["vector_from_half_A_evaluated_on_half_B"]["accuracy_per_layer"][layer] > 0.9
        assert all("accuracy_per_layer" in cell and "held_out_negative_answers" in cell for cell in raw["by_negative_tag"].values())
        assert len(entry[space]["split_half_cosine_half_A_against_half_B_per_layer"]) == 6
        assert "top_4_largest_mean_dimensions_zeroed" in " ".join(entry[space]["variant_names"])
    assert entry["training_questions_total"] >= entry["questions_kept"]
    # the steering and scoring run-folder builders read this vector set
    steering_layer, steering_paths, _, _ = steering_builder.read_vector_set(metadata_path.parent, definition.main_space, 4, definition)
    scoring_layer, scoring_paths, _, _ = scoring_builder.read_vector_set(metadata_path.parent, definition)
    assert steering_layer == scoring_layer == layer and all(path.exists() for path in [*steering_paths.values(), *scoring_paths.values()])
    if definition.has_confidence_control:
        control = entry["confidence_control"]
        assert "held_out_pole_classification_calibration_confidence_direction_projected_out" in control
        assert control["two_by_two_held_out_check"]["by_layer_and_vector"][str(layer)]["calibration"] is not None
        assert "stated_confidence_balancing" in entry


def test_question_level_balancing_of_stated_confidence_drops_whole_questions_and_reports_before_and_after():
    from persona_vector_projection import calibrated_consideration_confidence_controls as controls
    def question(question_id, positive_confidence, negative_confidence):
        return (question_id, [fake_answer(f"{question_id}_p{i}", "positive", question_id=question_id) for i, _ in enumerate(positive_confidence)],
                [fake_answer(f"{question_id}_n{i}", "negative", question_id=question_id) for i, _ in enumerate(negative_confidence)])
    stated = {"q1_p0": 90.0, "q1_p1": 70.0, "q1_n0": 20.0, "q2_p0": 80.0, "q2_n0": 40.0, "q3_p0": 50.0, "q3_n0": 80.0}
    kept_questions = [question("q1", [90, 70], [20]), question("q2", [80], [40]), question("q3", [50], [80])]
    bands = {"q1": "established", "q2": "established", "q3": "established"}
    kept, report = controls.balance_questions_on_stated_confidence(kept_questions, stated, bands, maximum_difference=5.0)
    assert 0 < len(kept) < 3 and all(len(item) == 3 for item in kept)
    assert abs(report["after_balancing"]["overall"]["difference_positive_minus_negative"]) <= 5.0
    assert report["before_balancing"]["overall"]["items"] == 3 and report["unit"].startswith("questions")


# ---------------------------------------------------------------------------------------------------
# --ignore-added-welfare-facts
# ---------------------------------------------------------------------------------------------------

def conditions_with_added_welfare_facts(thresholds):
    scores = good_scores("content_controlled_facet_score", 90.0)
    scores["content_check"]["adds_welfare_facts"] = True
    return compute.failed_answer_conditions("positive", scores, GOOD_SPAN, 20, 6, thresholds, facet_score_kind="content_controlled_facet_score")


def test_an_answer_reporting_added_welfare_facts_fails_by_default_and_is_kept_with_the_ignore_option():
    assert compute.parse_arguments(["--run-folder", "x"]).ignore_added_welfare_facts is False
    assert compute.parse_arguments(["--run-folder", "x", "--ignore-added-welfare-facts"]).ignore_added_welfare_facts is True
    assert conditions_with_added_welfare_facts(THRESHOLDS) == [compute.CONDITION_ADDS_WELFARE_FACTS]
    assert conditions_with_added_welfare_facts({**THRESHOLDS, "added_welfare_facts_condition_applied": True}) == [compute.CONDITION_ADDS_WELFARE_FACTS]
    assert conditions_with_added_welfare_facts({**THRESHOLDS, "added_welfare_facts_condition_applied": False}) == []


@pytest.mark.parametrize("pairing_level", ["sample", "question"])
def test_ignoring_added_welfare_facts_keeps_flagged_answers_and_the_metadata_records_the_setting_and_the_flagged_count(tmp_path, pairing_level):
    facet_key = "cares_about_welfare_content_controlled"
    flagged = lambda answer_id, pole: "rewording_01" in answer_id and pole == "positive"  # noqa: E731
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(
        tmp_path / "run", facet_key, layers=6, hidden=32, adds_welfare_facts_function=flagged)
    results = {}
    for applied in (True, False):
        metadata, _ = compute.compute_all(folder, {**THRESHOLDS, "added_welfare_facts_condition_applied": applied}, f"test_{applied}",
                                          bootstrap_resamples=5, minimum_layer_index=1, seed=0, say=lambda *_: None,
                                          facet_key=facet_key, pairing_level=pairing_level)
        results[applied] = metadata
    for applied, metadata in results.items():
        assert metadata["filter_thresholds"]["added_welfare_facts_condition_applied"] is applied
    flagged_key = "added_welfare_facts_reported_among_answers_used_by_pole_and_tag"
    applied_cells = results[True]["per_facet"][facet_key][flagged_key]
    ignored_cells = results[False]["per_facet"][facet_key][flagged_key]
    assert applied_cells["positive"]["all"]["answers_reporting_added_welfare_facts"] == 0
    assert ignored_cells["positive"]["all"]["answers_reporting_added_welfare_facts"] > 0
    assert ignored_cells["negative"]["all"]["answers_reporting_added_welfare_facts"] == 0
    assert ignored_cells["positive"]["all"]["answers_with_a_content_check"] > applied_cells["positive"]["all"]["answers_with_a_content_check"]
    failing = results[False]["per_facet"][facet_key]["answer_condition_counts_by_pole_and_tag"]["positive"]["all"]["failing"]
    assert compute.CONDITION_ADDS_WELFARE_FACTS not in failing
    assert compute.CONDITION_ADDS_WELFARE_FACTS in results[True]["per_facet"][facet_key]["answer_condition_counts_by_pole_and_tag"]["positive"]["all"]["failing"]


def test_the_default_threshold_setting_is_recorded_as_applied(tmp_path):
    facet_key = "cares_about_welfare_content_controlled"
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", facet_key, layers=6, hidden=32)
    metadata, _ = run_compute(folder, facet_key)
    assert metadata["filter_thresholds"]["added_welfare_facts_condition_applied"] is True
