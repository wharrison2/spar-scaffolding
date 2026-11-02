"""Tests for bugs that would invalidate the positive controls of the content-controlled facets (cares about welfare, calibrated
consideration, non-intrusive): which document each arm is applied to and compared with (arm 3 must use arm 1's OUTPUT), the expected
sign per arm and its derivation from the pole and the judge scale, thresholds that are not fixed never giving a pass, the scoring run
folder's refusals, and which vector is the main one (passage mean or whole answer). Local and free: a fake rewrite client, no API
call, no .env read, no pod.

    python3 -m pytest scripts/persona_vector_projection/test_content_controlled_facet_positive_controls_and_scoring.py -q
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import analyse_content_controlled_facet_positive_controls_and_corpus_windows as analysis  # noqa: E402
from persona_vector_projection import build_content_controlled_facet_scoring_run_folder as builder  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_arms as arms  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_statistics as statistics_module  # noqa: E402
from persona_vector_projection import make_content_controlled_facet_positive_control_rewrites as rewriter  # noqa: E402
from persona_vector_projection import make_facet_positive_control_rewrites as facet_rewrites  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import test_facet_positive_control_rewrites as fakes  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402

D = facet_definitions
NEW_FACETS = [D.CARES_ABOUT_WELFARE, D.CALIBRATED_CONSIDERATION, D.NON_INTRUSIVE]
AWAY, CONTENT, REMOVED, NEUTRAL = (D.ARM_TRAIT_ONLY_AWAY, D.ARM_CONTENT_ADDED_WITHOUT_TRAIT, D.ARM_TRAIT_REMOVED_FROM_AWAY_REWRITE,
                                   D.ARM_FACET_NEUTRAL_REWORDING)
UNDERSTATEMENT = D.ARM_UNDERSTATEMENT_OF_SUPPORTED_CLAIMS
FACET_PARAMETERS = pytest.mark.parametrize("definition", NEW_FACETS, ids=lambda d: d.facet_name)


# ---------------------------------------------------------------------------------------------------
# Arms, signs and the direction convention
# ---------------------------------------------------------------------------------------------------

@FACET_PARAMETERS
def test_expected_signs_follow_the_pole_convention_for_every_new_facet(definition):
    """The vector is positive minus negative and every judge gives 100 to the positive pole, so moving a document towards the negative
    pole is a NEGATIVE shift (arm 1) and restoring the trait is a POSITIVE shift (arm 3). A sign mix-up invalidates the experiment."""
    assert definition.high_score_pole == "positive"
    assert definition.expected_sign_of_steering_fraction_on_facet_score == 1
    assert arms.expected_sign_for(definition, AWAY) == -1
    assert arms.expected_sign_for(definition, REMOVED) == 1
    assert arms.expected_sign_for(definition, CONTENT) == 0 and arms.expected_sign_for(definition, NEUTRAL) == 0
    assert arms.NEGATIVE_POLE_DIRECTION_SIGN == -1


def test_the_sign_is_derived_from_the_scale_not_typed_in():
    flipped = facet_definitions.ContentControlledFacetDefinition(**{**D.NON_INTRUSIVE.__dict__, "high_score_pole": "negative"})
    assert flipped.expected_sign_of_steering_fraction_on_facet_score == -1
    assert flipped.low_score_pole == "positive"


@FACET_PARAMETERS
def test_arm_three_is_compared_with_arm_ones_rewrite_never_with_the_original(definition):
    assert arms.comparison_document_id_for(definition, AWAY, "matrix_000034") == "matrix_000034"
    assert arms.comparison_document_id_for(definition, CONTENT, "matrix_000034") == "matrix_000034"
    assert arms.comparison_document_id_for(definition, NEUTRAL, "matrix_000034") == "matrix_000034"
    arm_one_rewrite = arms.rewrite_document_id_for(definition, AWAY, "matrix_000034")
    assert arm_one_rewrite == f"matrix_000034__{definition.facet_key}__trait_only_away"
    assert arms.comparison_document_id_for(definition, REMOVED, "matrix_000034") == arm_one_rewrite != "matrix_000034"
    with pytest.raises(ValueError):
        arms.comparison_document_id_for(definition, "not_an_arm", "matrix_000034")


@FACET_PARAMETERS
def test_the_chained_arm_is_ordered_after_the_arm_it_starts_from_and_ids_parse_back(definition):
    names = arms.arm_names_for(definition)
    extra = [D.ARM_UNDERSTATEMENT_OF_SUPPORTED_CLAIMS] if definition.facet_name == "calibrated_consideration" else []
    assert sorted(names) == sorted(D.CONTENT_CONTROLLED_ARM_NAMES + extra) and names.index(REMOVED) > names.index(AWAY)
    for arm in names:
        document_id = arms.rewrite_document_id_for(definition, arm, "matrix_000034")
        assert arms.arm_of_rewrite_document_id_for(definition, document_id) == arm
        assert arms.original_source_document_id_of_rewrite_for(definition, document_id) == "matrix_000034"


def test_moralizing_delivery_keeps_its_own_arms_and_comparison_rule_in_the_shared_structure():
    moralizing = D.MORALIZING_DELIVERY
    assert arms.arm_names_for(moralizing) == arms.ARMS
    assert arms.comparison_document_id_for(moralizing, arms.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE, "m1") == "m1__preachier"
    assert arms.comparison_document_id_for(moralizing, arms.ARM_MORALIZING_FRAMING_ONLY, "m1") == "m1"
    for arm in arms.ARMS:
        assert arms.expected_sign_for(moralizing, arm) == arms.EXPECTED_SIGN_BY_ARM[arm]


def test_instruction_files_arm_four_falls_back_to_the_earlier_neutral_name_and_a_missing_file_is_refused(tmp_path):
    definition = D.CARES_ABOUT_WELFARE
    for arm in D.CONTENT_CONTROLLED_ARM_NAMES:
        if arm != NEUTRAL:
            (tmp_path / f"positive_control_rewrite_instruction__cares_about_welfare__{arm}__draft.txt").write_text("x")
    with pytest.raises(FileNotFoundError, match="facet_neutral"):
        arms.resolve_instruction_file_names(definition, tmp_path)
    (tmp_path / "positive_control_rewrite_instruction__cares_about_welfare__facet_neutral__draft.txt").write_text("old name")
    assert arms.resolve_instruction_file_names(definition, tmp_path)[NEUTRAL].endswith("__facet_neutral__draft.txt")
    (tmp_path / "positive_control_rewrite_instruction__cares_about_welfare__facet_neutral_rewording__draft.txt").write_text("new name")
    assert arms.resolve_instruction_file_names(definition, tmp_path)[NEUTRAL].endswith("__facet_neutral_rewording__draft.txt")


@FACET_PARAMETERS
def test_the_real_instruction_files_exist_and_each_arm_has_its_own_distinct_text(definition):
    templates = rewriter.load_instruction_templates(definition=definition)
    texts = [templates[facet_rewrites.instruction_file_name(definition.facet_key, arm)] for arm in arms.arm_names_for(definition)]
    assert len(set(texts)) == len(arms.arm_names_for(definition))
    document = {"doc_id": "d", "content": "Body with {braces} inside.", "language": "English", "variables": {}}
    for text in texts:
        prompt = facet_rewrites.render_instruction(text, document, 60, target_changed_units=274)
        assert "about 274 words" in prompt and prompt.rstrip("\n").endswith("Document:\nBody with {braces} inside.")


# ---------------------------------------------------------------------------------------------------
# The rewrite script with a fake client
# ---------------------------------------------------------------------------------------------------

@pytest.fixture
def rewrite_inputs(tmp_path):
    corpus = fakes.make_corpus(12)
    records = []
    for index in range(4):
        source = corpus[index]
        for direction in ("preachier", "plainer"):
            records.append({"document_id": f"{source['doc_id']}__{direction}", "source_document_id": source["doc_id"],
                            "rewrite_direction": direction, "content": fakes.change_words(source["content"], 50, direction),
                            "language": source["language"], "variables": source["variables"]})
    corpus_path, preachiness_path = tmp_path / "corpus.jsonl", tmp_path / "preachiness.jsonl"
    write_jsonl(corpus_path, corpus)
    write_jsonl(preachiness_path, records)
    return corpus, corpus_path, preachiness_path


def arm_from_prompt(prompt, definition):
    templates = rewriter.load_instruction_templates(definition=definition)
    matches = [arm for arm in arms.arm_names_for(definition)
               if prompt.startswith(templates[facet_rewrites.instruction_file_name(definition.facet_key, arm)].split("{")[0][:300])]
    assert len(matches) == 1, matches
    return matches[0]


def run_rewriter(tmp_path, rewrite_inputs, definition, client, extra=(), request_function=None):
    _, corpus_path, preachiness_path = rewrite_inputs
    return rewriter.main(["--facet", definition.facet_name, "--corpus-path", str(corpus_path), "--preachiness-rewrites-jsonl",
                          str(preachiness_path), "--workers", "1", "--minimum-changed-units", "30",
                          "--env-path", str(tmp_path / "no_such_env_file"), *extra],
                         client=client, runs_directory_override=tmp_path / "runs", request_function=request_function)


def answering_client(definition, how_many=80):
    return fakes.FakeClient(lambda prompt: fakes.make_response(
        fakes.change_words(fakes.document_text_in(prompt), how_many, arm_from_prompt(prompt, definition).replace("_", "")[:14])))


@FACET_PARAMETERS
def test_arm_three_is_applied_to_arm_ones_output_and_compared_with_it(tmp_path, rewrite_inputs, definition):
    corpus, _, _ = rewrite_inputs
    corpus_by_id = {d["doc_id"]: d for d in corpus}
    client = answering_client(definition)
    run_folder = run_rewriter(tmp_path, rewrite_inputs, definition, client)
    records = read_jsonl(sorted(run_folder.glob(f"{definition.facet_key}_positive_control_rewrites_*.jsonl"))[0])
    arms_count = len(arms.arm_names_for(definition))
    assert len(records) == 4 * arms_count and len(client.calls) == 4 * arms_count
    by_id = {r["document_id"]: r for r in records}
    for source in {r["original_corpus_source_document_id"] for r in records}:
        arm_one = by_id[arms.rewrite_document_id_for(definition, AWAY, source)]
        arm_three = by_id[arms.rewrite_document_id_for(definition, REMOVED, source)]
        assert arm_three["source_document_id"] == arm_three["comparison_document_id"] == arm_one["document_id"]
        assert arm_one["source_document_id"] == source and arm_one["expected_sign"] == -1 and arm_three["expected_sign"] == 1
        # the text arm 3 was asked to rewrite is arm 1's rewrite, not the original document
        prompts_of_arm_three = [c["messages"][0]["content"] for c in client.calls
                                if arm_from_prompt(c["messages"][0]["content"], definition) == REMOVED
                                and fakes.document_text_in(c["messages"][0]["content"]) == arm_one["content"]]
        assert len(prompts_of_arm_three) == 1
        assert arm_one["content"] != corpus_by_id[source]["content"]
        # its recorded change is counted against arm 1's rewrite
        changed = facet_rewrites.count_changed_units(arm_one["content"], arm_three["content"], arm_three["language"])
        assert arm_three["changed_rewrite_units"] == changed["changed_rewrite_units"]
    assert not any(fakes.document_text_in(c["messages"][0]["content"]) in {d["content"] for d in corpus}
                   for c in client.calls if arm_from_prompt(c["messages"][0]["content"], definition) == REMOVED)
    order = [arm_from_prompt(c["messages"][0]["content"], definition) for c in client.calls]
    assert max(i for i, arm in enumerate(order) if arm != REMOVED) < min(i for i, arm in enumerate(order) if arm == REMOVED)


def test_arm_three_is_skipped_never_run_on_the_original_when_arm_one_failed(tmp_path, rewrite_inputs):
    definition = D.CARES_ABOUT_WELFARE
    client = answering_client(definition)

    def request_function(fake_client, prompt_text):
        if arm_from_prompt(prompt_text, definition) == AWAY and fakes.document_text_in(prompt_text).startswith("w1_0 "):
            raise RuntimeError("simulated failure")
        return facet_rewrites.preachiness_rewrites.request_rewrite(fake_client, prompt_text)

    run_folder = run_rewriter(tmp_path, rewrite_inputs, definition, client, request_function=request_function)
    exported = read_jsonl(sorted(run_folder.glob("scoring_pod_extra_documents_*.jsonl"))[0])
    ids = {r["document_id"] for r in exported}
    assert arms.rewrite_document_id_for(definition, AWAY, "matrix_000001") not in ids
    assert arms.rewrite_document_id_for(definition, REMOVED, "matrix_000001") not in ids
    assert arms.rewrite_document_id_for(definition, REMOVED, "matrix_000002") in ids
    failures = read_jsonl(sorted(run_folder.glob("rewrite_failures_*.jsonl"))[-1]) + read_jsonl(sorted(run_folder.glob("rewrite_failures_*.jsonl"))[0])
    assert any(f.get("failure_kind") == "skipped_no_source_rewrite" and f["original_source_id"] == "matrix_000001" for f in failures)
    assert not any(fakes.document_text_in(c["messages"][0]["content"]).startswith("w1_0 ")
                   and arm_from_prompt(c["messages"][0]["content"], definition) == REMOVED for c in client.calls)


def test_export_carries_facet_arm_sign_and_comparison_document_and_sources_can_be_chosen(tmp_path, rewrite_inputs):
    definition = D.CALIBRATED_CONSIDERATION
    ids_file = tmp_path / "sources.txt"
    ids_file.write_text("# chosen for this facet\nmatrix_000003\nmatrix_000005\n")
    client = answering_client(definition)
    run_folder = run_rewriter(tmp_path, rewrite_inputs, definition, client, ["--source-document-ids-file", str(ids_file), "--target-changed-units", "100"])
    exported = read_jsonl(sorted(run_folder.glob("scoring_pod_extra_documents_*.jsonl"))[0])
    assert len(exported) == 10 and {r["original_corpus_source_document_id"] for r in exported} == {"matrix_000003", "matrix_000005"}
    assert all(r["facet"] == definition.facet_key and "comparison_document_id" in r for r in exported)
    assert {r["arm"]: r["expected_sign"] for r in exported} == {AWAY: -1, CONTENT: 0, REMOVED: 1, NEUTRAL: 0, UNDERSTATEMENT: -1}
    assert all("about 100 words" in c["messages"][0]["content"] for c in client.calls)


def test_resume_makes_no_new_call_for_finished_rewrites_and_an_unknown_source_is_refused(tmp_path, rewrite_inputs):
    definition = D.NON_INTRUSIVE
    client = answering_client(definition)
    run_folder = run_rewriter(tmp_path, rewrite_inputs, definition, client)
    calls = len(client.calls)
    run_rewriter(tmp_path, rewrite_inputs, definition, client, ["--resume-folder", str(run_folder)])
    assert len(client.calls) == calls
    ids_file = tmp_path / "bad_sources.txt"
    ids_file.write_text("not_in_the_corpus\n")
    with pytest.raises(SystemExit):
        run_rewriter(tmp_path, rewrite_inputs, definition, client, ["--source-document-ids-file", str(ids_file)])


@FACET_PARAMETERS
def test_dry_run_plans_four_arms_flags_the_chained_stand_in_and_never_touches_a_client(tmp_path, rewrite_inputs, definition):
    _, corpus_path, preachiness_path = rewrite_inputs
    folder = rewriter.main(["--facet", definition.facet_name, "--corpus-path", str(corpus_path), "--preachiness-rewrites-jsonl", str(preachiness_path),
                            "--dry-run", "--env-path", str(tmp_path / "no_such_env_file")], client=None, runs_directory_override=tmp_path / "runs")
    estimate = json.loads(sorted(folder.glob("dry_run_cost_estimate_and_ceiling_*.json"))[0].read_text())
    arms_count = len(arms.arm_names_for(definition))
    assert estimate["calls_planned"] == 4 * arms_count and set(estimate["calls_planned_by_arm"].values()) == {4}
    assert estimate["ceiling_cost_dollars"] > estimate["expected_cost_dollars"] > 0
    rendered = read_jsonl(sorted(folder.glob("rendered_rewrite_requests_*.jsonl"))[0])
    assert {r["arm"] for r in rendered if r["document_is_placeholder_for_arm_one_output"]} == {REMOVED}
    assert not list(folder.glob(f"{definition.facet_key}_positive_control_rewrites_*.jsonl"))
    configuration = json.loads(sorted(folder.glob("run_configuration_*.json"))[0].read_text())
    assert "NEGATIVE shift" in configuration["direction_convention"] and configuration["expected_sign_by_arm"][AWAY] == -1


def test_rewrite_cost_ceiling_is_read_from_the_dry_run(tmp_path, rewrite_inputs):
    from persona_vector_projection import estimate_content_controlled_facet_scoring_cost_ceilings as estimator
    _, corpus_path, preachiness_path = rewrite_inputs
    folder = rewriter.main(["--facet", "cares_about_welfare", "--corpus-path", str(corpus_path), "--preachiness-rewrites-jsonl",
                            str(preachiness_path), "--dry-run"], client=None, runs_directory_override=tmp_path / "runs")
    text = estimator.describe_rewrite_ceiling(folder)
    assert "16 calls" in text and "absolute worst case" in text


# ---------------------------------------------------------------------------------------------------
# Criteria: thresholds read from a file, never defaulted
# ---------------------------------------------------------------------------------------------------

def make_table(span, whole, circular_lower=None, circular_two=None, whole_after_token_64=None):
    """One arm's per-rewrite table. `whole` is the all-token whole-document shift (reported); `whole_after_token_64` (default: the same
    numbers) is the one the counted criteria read."""
    count = len(span)
    return pd.DataFrame({
        "changed_span_shift": span, "whole_document_shift": whole,
        "whole_document_shift_after_token_64": whole if whole_after_token_64 is None else whole_after_token_64,
        "circular_p_lower": circular_lower if circular_lower is not None else [0.5] * count,
        "circular_p_upper": [0.5] * count,
        "circular_p_two_sided": circular_two if circular_two is not None else [1.0] * count,
        "whole_bootstrap_p_negative_direction": [0.5] * count, "whole_bootstrap_p_positive_direction": [0.5] * count})


WHOLE_AWAY = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.7, -1.3, -1.0, 0.2, 0.1]            # 8 of 10 negative, mean -0.75
WHOLE_REMOVED = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1.0, -0.1, -0.2]               # 8 of 10 positive, Wilcoxon p well below 0.05
WHOLE_NEUTRAL = [0.1, -0.2, 0.1, 0.0, -0.1, 0.2, -0.1, 0.1, 0.0, 0.0]               # mean +0.01


def passing_tables():
    return {AWAY: make_table([-3.0, -2.0, -4.0, -2.5, -3.5, -1.0, -5.0, -2.2, -3.1, 0.5], WHOLE_AWAY),
            REMOVED: make_table([3.0, 2.0, 4.0, 2.5, 3.5, 1.0, 5.0, 2.2, 3.1, -0.5], WHOLE_REMOVED),
            CONTENT: make_table([-0.3, 0.2, -0.1, 0.1, 0.4, -0.2, 0.0, 0.3, -0.4, 0.1], [0.0] * 10),
            NEUTRAL: make_table([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0, -0.1, 0.2], WHOLE_NEUTRAL)}


def shipped_criteria(definition=None):
    """The criteria exactly as shipped in pass_criteria/ (thresholds are read from the file, never typed into a test)."""
    return json.loads((definition or D.CARES_ABOUT_WELFARE).pass_criteria_file().read_text())


def criteria_with(changes, definition=None):
    """The shipped criteria with `changes` {(block, name): value} applied."""
    criteria = shipped_criteria(definition)
    for (block, name), value in changes.items():
        criteria[block][name] = value
    return criteria


def with_every_threshold_null(criteria):
    nulled = json.loads(json.dumps(criteria))
    for key, block in nulled.items():
        if isinstance(block, dict) and key != "reported_not_counted":
            for name, value in block.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and name != "counted":
                    block[name] = None
    return nulled


def evaluate(tables, criteria):
    summaries = {arm: statistics_module.summarise_arm(table, arm) for arm, table in tables.items()}
    return {line["criterion"]: line for line in statistics_module.evaluate_criteria_from_file(summaries, criteria, "test_facet")}


def counted(lines):
    return {name: line for name, line in lines.items() if line["counted"]}


def test_null_thresholds_give_not_evaluated_and_never_a_pass():
    for criteria in ({}, with_every_threshold_null(shipped_criteria())):
        lines = evaluate(passing_tables(), criteria)
        assert len(counted(lines)) == 11 and len(lines) == 13
        assert all(line["passed"] is None and line["verdict_text"] == "not evaluated: thresholds not fixed" for line in counted(lines).values())
        assert statistics_module.overall_verdict(list(lines.values()), thresholds_fixed=False)["verdict"] == "not evaluated: thresholds not fixed"
        assert statistics_module.overall_verdict(list(lines.values()), thresholds_fixed=True)["verdict"] == "undecided"


@FACET_PARAMETERS
def test_every_shipped_criteria_file_is_fixed_to_the_document_values_and_points_to_the_document(definition):
    criteria = shipped_criteria(definition)
    assert criteria["facet_key"] == definition.facet_key and criteria["thresholds_fixed"] is True
    assert "user has not fixed" not in criteria["note"]
    assert criteria["criteria_document"].endswith("content-controlled-facets-pass-criteria-fixed-before-scoring_2026-11-01_15-04.md")
    assert (Path(__file__).resolve().parents[2] / criteria["criteria_document"]).exists()
    trait = {"changed_span_rewrites_required_in_expected_direction": 8, "changed_span_wilcoxon_p_maximum": 0.05,
             "whole_document_rewrites_required_in_expected_direction": 8,
             "whole_document_mean_shift_minimum_multiple_of_neutral_rewording_absolute_mean_shift": 2}
    arms_with_trait_criteria = [AWAY] + ([UNDERSTATEMENT] if definition.has_confidence_control else [])
    for arm in arms_with_trait_criteria:
        assert {k: v for k, v in criteria[arm].items() if k != "counted"} == trait and criteria[arm]["counted"] is True
    assert (UNDERSTATEMENT in criteria) == definition.has_confidence_control
    assert {k: v for k, v in criteria[REMOVED].items() if k != "counted"} == {
        "whole_document_rewrites_required_in_expected_direction": 8, "whole_document_wilcoxon_p_maximum": 0.05}
    content = criteria[CONTENT]
    assert content["effect_maximum_fraction_of_trait_only_away_effect"] == pytest.approx(1 / 3)
    assert content["whole_document_mean_shift_maximum_fraction_of_trait_only_away_mean_shift"] == pytest.approx(1 / 3)
    assert content["rewrites_significant_towards_negative_pole_maximum"] == 3
    assert criteria[NEUTRAL]["rewrites_significant_either_direction_maximum"] == 2
    assert criteria[NEUTRAL]["whole_document_absolute_mean_shift_maximum_fraction_of_trait_only_away_mean_shift"] == pytest.approx(1 / 3)
    assert {k: v for k, v in criteria["extraction"].items() if k != "counted"} == {
        "held_out_accuracy_minimum": 0.80, "held_out_accuracy_by_tag_minimum": 0.70, "tag_minimum_held_out_negatives": 5,
        "cross_half_accuracy_minimum": 0.80, "training_questions_with_kept_items_minimum_share": 0.75}
    reported = criteria["reported_not_counted"]
    assert reported["counted"] is False and reported["split_half_cosine"]["former_minimum"] == 0.80
    assert reported["trait_removed_from_away_rewrite_changed_span"] == {"former_rewrites_required_in_expected_direction": 8, "former_wilcoxon_p_maximum": 0.05}
    assert reported["steering"]["spearman_of_steering_fraction_against_facet_score_minimum_in_expected_direction"] == 0.3
    assert "corpus_segments" in reported
    calibrated_only = criteria.get("calibrated_consideration_only")
    if definition.has_confidence_control:
        assert calibrated_only["projected_out_held_out_accuracy_minimum"] == 0.75
        assert calibrated_only["two_by_two_interaction_interval_must_exclude_zero"] is True
        assert calibrated_only["two_by_two_interaction_absolute_size_must_exceed_confidence_main_effect_absolute_size"] is True
    else:
        assert calibrated_only is None


def test_moralizing_delivery_keeps_its_own_criteria_and_is_not_given_the_new_ones():
    assert statistics_module.FRAMING_WHOLE_DOCUMENT_REWRITES_REQUIRED == 7 and statistics_module.REMOVED_CHANGED_SPAN_REWRITES_REQUIRED == 8
    assert statistics_module.SPLIT_HALF_COSINE_MINIMUM == 0.80 and statistics_module.TRAINING_QUESTIONS_WITH_KEPT_ITEMS_MINIMUM == 15
    assert D.MORALIZING_DELIVERY.legacy_moralizing_names


def test_thresholds_from_the_file_are_used_when_set_and_a_changed_threshold_changes_only_its_own_line():
    criteria = shipped_criteria()
    lines = evaluate(passing_tables(), criteria)
    assert all(line["passed"] is True for line in counted(lines).values()), {k: v["passed"] for k, v in lines.items()}
    stricter = criteria_with({(AWAY, "changed_span_rewrites_required_in_expected_direction"): 10})
    lines = evaluate(passing_tables(), stricter)
    assert lines[f"{AWAY}__changed_span_direction"]["passed"] is False
    assert lines[f"{AWAY}__whole_document_direction"]["passed"] is True      # only the threshold that changed
    partly_fixed = criteria_with({(NEUTRAL, "rewrites_significant_either_direction_maximum"): None})
    lines = evaluate(passing_tables(), partly_fixed)
    assert lines[f"{NEUTRAL}__few_individually_significant"]["passed"] is None
    # a threshold that is absent from the file is not filled in either
    missing = shipped_criteria()
    del missing[REMOVED]["whole_document_wilcoxon_p_maximum"]
    assert evaluate(passing_tables(), missing)[f"{REMOVED}__whole_document_wilcoxon"]["passed"] is None


def test_a_pole_mix_up_fails_both_directional_arms_a_negative_shift_is_right_for_arm_one_and_wrong_for_arm_three():
    tables = passing_tables()
    tables[AWAY], tables[REMOVED] = tables[REMOVED], tables[AWAY]        # the bug: shifts with the signs the wrong way round
    lines = evaluate(tables, shipped_criteria())
    assert lines[f"{AWAY}__changed_span_direction"]["passed"] is False and lines[f"{AWAY}__whole_document_direction"]["passed"] is False
    assert lines[f"{REMOVED}__whole_document_direction"]["passed"] is False
    only_negative = {**passing_tables(), AWAY: make_table([-3.0 - i for i in range(10)], [-1.0 - i / 10 for i in range(10)]),
                     REMOVED: make_table([-3.0 - i for i in range(10)], [-1.0 - i / 10 for i in range(10)])}
    lines = evaluate(only_negative, shipped_criteria())
    assert lines[f"{AWAY}__changed_span_direction"]["passed"] is True
    assert lines[f"{REMOVED}__whole_document_direction"]["passed"] is False
    assert lines[f"{REMOVED}__whole_document_wilcoxon"]["passed"] is True   # significant, but in the wrong direction


def test_content_added_arm_fails_when_its_effect_or_whole_document_shift_towards_the_negative_pole_is_large():
    criteria = shipped_criteria()
    tables = passing_tables()
    framing = tables[AWAY]["changed_span_shift"]
    tables[CONTENT] = make_table(list(framing - framing.mean() * 0.5), [0.0] * 10)
    assert evaluate(tables, criteria)[f"{CONTENT}__effect_small_next_to_trait_only_away"]["passed"] is False
    for whole_mean, expected in ((-0.2, True), (-0.3, False)):          # a third of the trait arm's -0.75 is 0.25
        tables[CONTENT] = make_table([0.1] * 10, [0.0] * 10, whole_after_token_64=[whole_mean] * 10)
        assert evaluate(tables, criteria)[f"{CONTENT}__whole_document_shift_small_next_to_trait_only_away"]["passed"] is expected
    tables[CONTENT] = make_table([0.1] * 10, [0.0] * 10, whole_after_token_64=[0.5] * 10)       # towards the POSITIVE pole: not a fault of this arm
    assert evaluate(tables, criteria)[f"{CONTENT}__whole_document_shift_small_next_to_trait_only_away"]["passed"] is True
    for significant, expected in ((3, True), (4, False)):
        tables[CONTENT] = make_table([0.1] * 10, [0.0] * 10, circular_lower=[0.01] * significant + [0.5] * (10 - significant))
        assert evaluate(tables, criteria)[f"{CONTENT}__few_individually_towards_the_negative_pole"]["passed"] is expected


def test_neutral_rewording_counts_individually_significant_rewrites_in_either_direction_and_its_whole_document_shift():
    criteria = shipped_criteria()
    for significant, expected in ((2, True), (3, False)):
        tables = passing_tables()
        tables[NEUTRAL] = make_table([0.1] * 10, WHOLE_NEUTRAL, circular_two=[0.01] * significant + [0.5] * (10 - significant))
        assert evaluate(tables, criteria)[f"{NEUTRAL}__few_individually_significant"]["passed"] is expected
    for neutral_mean, expected in ((0.2, True), (0.3, False), (-0.3, False)):    # a third of 0.75 is 0.25, in either direction
        tables = passing_tables()
        tables[NEUTRAL] = make_table([0.1] * 10, [0.0] * 10, whole_after_token_64=[neutral_mean] * 10)
        assert evaluate(tables, criteria)[f"{NEUTRAL}__whole_document_absolute_mean_shift_small_next_to_trait_only_away"]["passed"] is expected


def test_the_neutral_comparison_uses_the_absolute_neutral_mean_so_a_negative_neutral_mean_cannot_make_it_easier():
    """Trait arm mean shift -0.75, so with factor 2 the neutral arm's absolute mean must be at most 0.375. A neutral mean of -0.4 and one
    of +0.4 are both too large (a signed comparison would let -0.4 through as 'below' the trait arm), and -0.3 and +0.3 both pass."""
    criteria = shipped_criteria()
    name = f"{AWAY}__whole_document_shift_against_neutral_rewording"
    for neutral_mean, expected in ((-0.4, False), (0.4, False), (-0.3, True), (0.3, True), (-0.375, True), (0.0, True)):
        tables = passing_tables()
        tables[NEUTRAL] = make_table([0.1] * 10, [0.0] * 10, whole_after_token_64=[neutral_mean] * 10)
        line = evaluate(tables, criteria)[name]
        assert line["passed"] is expected, (neutral_mean, line["measured"])
        if neutral_mean != 0:
            assert line["measured"]["neutral_rewording_absolute_mean_shift"] == pytest.approx(abs(neutral_mean))
    stricter = criteria_with({(AWAY, "whole_document_mean_shift_minimum_multiple_of_neutral_rewording_absolute_mean_shift"): 3})
    tables = passing_tables()
    tables[NEUTRAL] = make_table([0.1] * 10, [0.0] * 10, whole_after_token_64=[0.3] * 10)
    assert evaluate(tables, stricter)[name]["passed"] is False            # the factor comes from the file
    wrong_way = passing_tables()
    wrong_way[AWAY] = make_table(list(passing_tables()[AWAY]["changed_span_shift"]), [0.9] * 10)   # trait arm moved the wrong way
    assert evaluate(wrong_way, criteria)[name]["passed"] is False


def test_the_removal_arm_is_counted_on_the_whole_document_and_not_on_the_changed_span():
    criteria = shipped_criteria()
    flat_span_good_whole = {**passing_tables(), REMOVED: make_table([0.01, -0.02, 0.0, 0.02, -0.01, 0.0, 0.01, -0.01, 0.0, 0.02], WHOLE_REMOVED)}
    lines = evaluate(flat_span_good_whole, criteria)
    assert lines[f"{REMOVED}__whole_document_direction"]["passed"] is True and lines[f"{REMOVED}__whole_document_wilcoxon"]["passed"] is True
    assert not any(line["counted"] for name, line in lines.items() if name.startswith(f"{REMOVED}__changed_span"))
    assert all(line["passed"] is True for line in counted(lines).values())
    assert statistics_module.overall_verdict(list(lines.values()))["verdict"] == "passes"
    good_span_flat_whole = {**passing_tables(), REMOVED: make_table(list(passing_tables()[REMOVED]["changed_span_shift"]), [0.0] * 10)}
    lines = evaluate(good_span_flat_whole, criteria)
    assert lines[f"{REMOVED}__whole_document_direction"]["passed"] is False
    assert lines[f"{REMOVED}__changed_span_direction__reported"]["measured"].startswith("9 of 10")
    assert statistics_module.overall_verdict(list(lines.values()))["verdict"] == "fails"
    # the all-token whole-document column is not what is counted: only the after-token-64 one is
    only_all_token = {**passing_tables(), REMOVED: make_table(list(passing_tables()[REMOVED]["changed_span_shift"]), WHOLE_REMOVED,
                                                              whole_after_token_64=[0.0] * 10)}
    assert evaluate(only_all_token, criteria)[f"{REMOVED}__whole_document_direction"]["passed"] is False


def test_a_counted_failure_makes_the_overall_verdict_fail_and_every_counted_line_is_listed():
    criteria = shipped_criteria()
    lines = list(evaluate(passing_tables(), criteria).values())
    verdict = statistics_module.overall_verdict(lines)
    assert verdict["verdict"] == "passes" and verdict["counts"] == {"counted": 11, "passed": 11, "failed": 0, "undecided": 0, "reported": 2}
    assert len(verdict["counted_criteria"]) == 11 and len(verdict["reported_not_counted"]) == 2
    broken = passing_tables()
    broken[NEUTRAL] = make_table([0.1] * 10, [0.0] * 10, circular_two=[0.01] * 5 + [0.5] * 5, whole_after_token_64=WHOLE_NEUTRAL)
    verdict = statistics_module.overall_verdict(list(evaluate(broken, criteria).values()))
    assert verdict["verdict"] == "fails" and verdict["counts"]["failed"] == 1
    assert [item["criterion"] for item in verdict["counted_criteria"] if item["verdict"] == "FAIL"] == [f"{NEUTRAL}__few_individually_significant"]
    # a failure on a line that is not counted does nothing; one undecided counted line, with no failure, is undecided and never a pass
    undecided = [dict(line, passed=None, verdict_text="undecided") if line["criterion"].endswith("changed_span_wilcoxon") and line["counted"] else line
                 for line in lines]
    assert statistics_module.overall_verdict(undecided)["verdict"] == "undecided"
    assert statistics_module.overall_verdict([])["verdict"] == "undecided"


def test_reported_items_never_change_the_verdict():
    criteria = shipped_criteria()
    base = statistics_module.overall_verdict(list(evaluate(passing_tables(), criteria).values()))["verdict"]
    # the removal arm's changed span, reported against its former threshold, flat or wrong-way
    for span in ([0.0] * 10, [-3.0] * 10):
        tables = {**passing_tables(), REMOVED: make_table(span, WHOLE_REMOVED)}
        lines = evaluate(tables, criteria)
        assert lines[f"{REMOVED}__changed_span_direction__reported"]["passed"] is None
        assert lines[f"{REMOVED}__changed_span_direction__reported"]["verdict_text"] == "reported"
        assert "former threshold" in lines[f"{REMOVED}__changed_span_wilcoxon__reported"]["required"]
        assert statistics_module.overall_verdict(list(lines.values()))["verdict"] == base == "passes"
    extraction = lambda cosine: statistics_module.evaluate_extraction_criteria_from_file(  # noqa: E731
        criteria, 0.9, {"a": 0.9}, {"a": 8}, 0.9, cosine, 18, 20)
    for cosine in (0.99, 0.0, -1.0, None):
        lines = extraction(cosine)
        reported = [line for line in lines if not line["counted"]]
        assert [line["criterion"] for line in reported] == ["extraction__split_half_cosine__reported"] and reported[0]["required"] == "former threshold: >= 0.8"
        assert statistics_module.overall_verdict(lines)["verdict"] == "passes"
    window_ids = [f"lowest_window_{n:02d}" for n in range(1, 11)]
    for reading in ({w: True for w in window_ids}, {w: False for w in window_ids}, None):
        line = statistics_module.evaluate_corpus_segment_criterion_from_file(criteria, reading)
        assert line["counted"] is False and line["passed"] is None and line["verdict_text"] == "reported"
        assert statistics_module.overall_verdict(extraction(0.9) + [line])["verdict"] == "passes"


def test_extraction_lines_are_decided_from_the_files_values():
    criteria = shipped_criteria()
    lines = statistics_module.evaluate_extraction_criteria_from_file(criteria, 0.85, {"a": 0.9, "b": 0.75}, {"a": 8, "b": 6}, 0.82, 0.3, 15, 20)
    assert {l["criterion"]: l["passed"] for l in lines if l["counted"]} == {
        "extraction__held_out_accuracy": True, "extraction__held_out_accuracy_by_negative_tag": True,
        "extraction__cross_half_accuracy": True, "extraction__training_questions_with_kept_items": True}   # 15 of 20 is exactly 0.75
    def decided(accuracy=0.85, by_tag=None, negatives=None, cross=0.82, kept=15, total=20, changes=None):
        used = criteria_with(changes or {})
        lines = statistics_module.evaluate_extraction_criteria_from_file(
            used, accuracy, by_tag or {"a": 0.9}, negatives or {"a": 8}, cross, 0.9, kept, total)
        return {l["criterion"]: l["passed"] for l in lines if l["counted"]}
    assert decided(accuracy=0.79)["extraction__held_out_accuracy"] is False
    assert decided(cross=0.79)["extraction__cross_half_accuracy"] is False and decided(cross=0.80)["extraction__cross_half_accuracy"] is True
    assert decided(kept=14)["extraction__training_questions_with_kept_items"] is False
    assert decided(kept=15, total=24)["extraction__training_questions_with_kept_items"] is False       # a share, not a count
    assert decided(changes={("extraction", "training_questions_with_kept_items_minimum_share"): 0.7, })["extraction__training_questions_with_kept_items"] is True
    # per tag: tags with at least 5 held-out negatives are counted, tags with fewer are reported and cannot fail
    tag = "extraction__held_out_accuracy_by_negative_tag"
    assert decided(by_tag={"a": 0.9, "b": 0.5}, negatives={"a": 8, "b": 5})[tag] is False
    assert decided(by_tag={"a": 0.9, "b": 0.5}, negatives={"a": 8, "b": 4})[tag] is True
    assert decided(by_tag={"a": 0.9, "b": 0.7}, negatives={"a": 8, "b": 5})[tag] is True
    assert decided(by_tag={"a": 0.9}, negatives={"a": 4})[tag] is None                      # nothing countable: undecided, never a pass
    measured = [l for l in statistics_module.evaluate_extraction_criteria_from_file(
        criteria, 0.9, {"a": 0.9, "b": 0.5}, {"a": 8, "b": 4}, 0.9, 0.9, 18, 20) if l["criterion"] == tag][0]["measured"]
    assert set(measured["tags_counted"]) == {"a"} and set(measured["tags_reported_only_too_few_held_out_negatives"]) == {"b"}
    nothing_recorded = statistics_module.evaluate_extraction_criteria_from_file(criteria, None, None, None, None, None, None, None)
    assert all(line["passed"] is None for line in nothing_recorded)


def test_the_two_by_two_rule_needs_both_the_interval_excluding_zero_and_the_interaction_larger_than_the_main_effect():
    criteria = shipped_criteria(D.CALIBRATED_CONSIDERATION)
    def line(interaction, interval, main_effect, accuracy=0.8):
        check = {"interaction_contrast_calibrated_minus_miscalibrated": interaction, "interaction_contrast_95_percent_bootstrap_interval": interval,
                 "main_effect_of_confidence_confident_minus_hedged": main_effect}
        return {l["criterion"]: l for l in statistics_module.evaluate_calibrated_consideration_criteria_from_file(criteria, accuracy, check)}
    rule = "calibrated_consideration__two_by_two_rule"
    assert line(2.0, [1.0, 3.0], 0.5)[rule]["passed"] is True
    assert line(-2.0, [-3.0, -1.0], 0.5)[rule]["passed"] is True                   # absolute sizes; the interval excludes zero on the other side
    assert line(2.0, [-0.5, 3.0], 0.5)[rule]["passed"] is False                    # size condition holds, interval includes zero
    assert line(2.0, [1.0, 3.0], 2.5)[rule]["passed"] is False                     # interval excludes zero, main effect is larger
    assert line(2.0, [1.0, 3.0], -2.5)[rule]["passed"] is False                    # absolute main effect
    assert line(0.5, [-0.1, 1.0], 2.0)[rule]["passed"] is False
    failed = line(2.0, [-0.5, 3.0], 0.5)[rule]
    assert "confidence vector, not a calibration vector" in failed["consequence"]
    assert line(2.0, None, 0.5)[rule]["passed"] is None and line(None, None, None)[rule]["passed"] is None   # missing: undecided
    assert line(2.0, None, 2.5)[rule]["passed"] is False                           # one failed condition is enough
    projected = "calibrated_consideration__projected_out_held_out_accuracy"
    assert line(2.0, [1.0, 3.0], 0.5, accuracy=0.75)[projected]["passed"] is True
    assert line(2.0, [1.0, 3.0], 0.5, accuracy=0.74)[projected]["passed"] is False
    assert line(2.0, [1.0, 3.0], 0.5, accuracy=None)[projected]["passed"] is None
    assert statistics_module.evaluate_calibrated_consideration_criteria_from_file(shipped_criteria(D.CARES_ABOUT_WELFARE), 0.9, None) == []
    verdict = statistics_module.overall_verdict(list(line(2.0, [-0.5, 3.0], 0.5).values()))
    assert verdict["verdict"] == "fails"


def test_calibrated_consideration_counts_the_understatement_arm_with_the_trait_changed_criteria():
    criteria = shipped_criteria(D.CALIBRATED_CONSIDERATION)
    tables = {**passing_tables(), UNDERSTATEMENT: passing_tables()[AWAY]}
    lines = evaluate(tables, criteria)
    understatement = [name for name in lines if name.startswith(UNDERSTATEMENT)]
    assert len(understatement) == 4 and all(lines[name]["counted"] and lines[name]["passed"] is True for name in understatement)
    tables[UNDERSTATEMENT] = make_table([0.5] * 10, [0.5] * 10)
    lines = evaluate(tables, criteria)
    assert all(lines[name]["passed"] is False for name in understatement if name.endswith("direction"))
    assert statistics_module.overall_verdict(list(lines.values()))["verdict"] == "fails"


def test_extraction_and_corpus_segment_lines_are_undecided_with_null_thresholds():
    null_criteria = with_every_threshold_null(shipped_criteria(D.NON_INTRUSIVE))
    lines = statistics_module.evaluate_extraction_criteria_from_file(null_criteria, 0.9, {"a": 0.9}, {"a": 8}, 0.9, 0.9, 15, 20)
    assert all(line["passed"] is None and line["verdict_text"] in ("not evaluated: thresholds not fixed", "reported") for line in lines)
    assert not statistics_module.overall_verdict(lines)["verdict"] == "passes"


# ---------------------------------------------------------------------------------------------------
# Whole-document shift beside the between-document spread, and the analysis on synthetic scored data
# ---------------------------------------------------------------------------------------------------

def test_between_document_standard_deviation_is_the_spread_of_the_corpus_documents_whole_document_means():
    values = {"a": 1.0, "b": 3.0, "c": 5.0}
    assert statistics_module.between_document_standard_deviation(values) == pytest.approx(2.0)
    assert np.isnan(statistics_module.between_document_standard_deviation({"a": 1.0}))
    assert statistics_module.shift_in_units_of_between_document_spread(1.0, 2.0) == pytest.approx(0.5)
    assert np.isnan(statistics_module.shift_in_units_of_between_document_spread(1.0, 0.0))
    # the opening tokens are left out, as in every other whole-document number
    assert statistics_module.whole_document_mean_projection([100.0] * 64 + [2.0, 4.0]) == pytest.approx(3.0)


def synthetic_scored_data(definition, number_of_sources=4, tokens=320):
    """Corpus documents, the four arms' rewrites of each, per-token projections for two vectors, with known shifts: arm 1 lowers the
    changed span, arm 3 raises it again relative to arm 1's rewrite, arms 2 and 4 change nothing but the tokens."""
    generator = np.random.default_rng(0)
    documents, per_token, token_ids = {}, {}, {}
    vector_names = definition.scored_vector_names()[:2]
    if definition.has_confidence_control:
        vector_names = vector_names + [definition.confidence_control_vector_name]  # third column
    columns = len(vector_names)
    # shift per arm: (calibration columns, confidence column). Overclaiming raises stated confidence, understatement lowers it, and
    # both lower calibration.
    confidence_shift_by_arm = {AWAY: 3.0, UNDERSTATEMENT: -3.0, CONTENT: 0.0, NEUTRAL: 0.0}
    for number in range(number_of_sources):
        source = f"matrix_{number:06d}"
        base_ids = list(range(1000 * number, 1000 * number + tokens))
        base_values = generator.normal(size=(tokens, columns)) + 10.0 * number     # documents differ by 10 per source: the between-document spread
        documents[source] = {"document_id": source, "source_document_id": source, "kind": "corpus", "content": "x", "rewrite_arm": None}
        per_token[source], token_ids[source] = base_values, base_ids
        rewrites = {}
        calibrated_arms = ((AWAY, -3.0), (CONTENT, 0.0), (NEUTRAL, 0.0)) + (
            ((UNDERSTATEMENT, -3.0),) if definition.has_confidence_control else ())
        for arm, shift in calibrated_arms:
            ids = base_ids[:100] + [90000 + 100 * number + k for k in range(40)] + base_ids[140:]
            values = base_values.copy()
            values[100:140, :2] += shift
            if definition.has_confidence_control:
                values[100:140, 2] += confidence_shift_by_arm[arm]
            rewrites[arm] = (ids, values)
        arm_one_ids, arm_one_values = rewrites[AWAY]
        ids = arm_one_ids[:200] + [95000 + 100 * number + k for k in range(40)] + arm_one_ids[240:]
        values = arm_one_values.copy()
        values[200:240, :2] += 3.0
        if definition.has_confidence_control:
            values[200:240, 2] += -3.0  # removing the overclaim lowers stated confidence again
        rewrites[REMOVED] = (ids, values)
        for arm, (ids, values) in rewrites.items():
            document_id = arms.rewrite_document_id_for(definition, arm, source)
            comparison = arms.comparison_document_id_for(definition, arm, source)
            documents[document_id] = {"document_id": document_id, "source_document_id": comparison, "kind": "rewrite", "rewrite_arm": arm,
                                      "content": "y", "below_minimum_changed_units": False}
            per_token[document_id], token_ids[document_id] = values, ids
    data = SimpleNamespace(document_format="assistant_turn", vector_names=vector_names, per_token=per_token, token_ids=token_ids)
    return data, documents


def test_analysis_reports_shifts_beside_the_between_document_spread_and_not_evaluated_verdicts_with_null_thresholds():
    definition = D.CARES_ABOUT_WELFARE
    data, documents = synthetic_scored_data(definition)
    criteria = with_every_threshold_null(shipped_criteria(definition))
    table, rows, lines_by_vector = analysis.analyse_format(data, documents, data.vector_names, 0, 100, 100, say=lambda *_: None,
                                                           definition=definition, criteria=criteria)
    assert set(table["arm"]) == set(arms.arm_names_for(definition))
    by_arm = {row["arm"]: row for row in rows if row["vector"] == data.vector_names[0]}
    corpus_means = [statistics_module.whole_document_mean_projection(data.per_token[f"matrix_{n:06d}"][:, 0]) for n in range(4)]
    expected_sd = float(np.std(corpus_means, ddof=1))
    assert expected_sd > 5
    for row in by_arm.values():
        assert row["corpus_between_document_standard_deviation"] == pytest.approx(expected_sd)
        assert row["whole_document_shift_mean_in_between_document_standard_deviations"] == pytest.approx(row["whole_document_shift_mean"] / expected_sd)
    assert "whole_document_shift_in_between_document_standard_deviations" in table.columns
    # known shifts: 40 of 320 tokens moved by -3 (arm 1) and by +3 (arm 3, relative to arm 1's rewrite)
    assert by_arm[AWAY]["whole_document_shift_mean"] == pytest.approx(-3.0 * 40 / 360, abs=0.3) and by_arm[AWAY]["expected_sign"] == -1
    assert by_arm[REMOVED]["whole_document_shift_mean"] > 0 and by_arm[REMOVED]["expected_sign"] == 1
    assert by_arm[AWAY]["changed_span_mean"] < -2.5 and by_arm[REMOVED]["changed_span_mean"] > 2.5
    assert abs(by_arm[CONTENT]["changed_span_mean"]) < 1.0 and abs(by_arm[NEUTRAL]["changed_span_mean"]) < 1.0
    assert set(table[table["arm"] == REMOVED]["comparison_document_id"]) == {arms.rewrite_document_id_for(definition, AWAY, f"matrix_{n:06d}") for n in range(4)}
    for lines in lines_by_vector.values():
        assert all(line["passed"] is None and line["verdict_text"] == ("not evaluated: thresholds not fixed" if line["counted"] else "reported")
                   for line in lines)


def test_analysis_with_thresholds_given_produces_pass_verdicts_on_data_built_to_pass():
    definition = D.CALIBRATED_CONSIDERATION
    data, documents = synthetic_scored_data(definition)
    # the shipped thresholds, except the counts and p value that four synthetic source documents cannot reach (Wilcoxon p is at least 0.125)
    small_sample = {"changed_span_rewrites_required_in_expected_direction": 4, "whole_document_rewrites_required_in_expected_direction": 4}
    criteria = criteria_with({**{(arm, name): value for arm in (AWAY, UNDERSTATEMENT) for name, value in
                                 {**small_sample, "changed_span_wilcoxon_p_maximum": 0.2}.items()},
                              (REMOVED, "whole_document_rewrites_required_in_expected_direction"): 4,
                              (REMOVED, "whole_document_wilcoxon_p_maximum"): 0.2}, definition)
    _, _, lines_by_vector = analysis.analyse_format(data, documents, data.vector_names, 0, 100, 100, say=lambda *_: None, definition=definition,
                                                    criteria=criteria)
    main = lines_by_vector[data.vector_names[0]]
    failing = [line["criterion"] for line in main if line["counted"] and line["passed"] is not True]
    assert not failing, failing
    assert statistics_module.overall_verdict(main)["verdict"] == "passes"


def test_non_intrusive_scores_the_whole_answer_vector_as_main_and_cares_the_passage_mean_vector():
    assert D.NON_INTRUSIVE.main_vector_name == "non_intrusive_content_controlled_whole_answer_mean_vector"
    assert D.NON_INTRUSIVE.main_space == "whole_answer_mean" and not D.NON_INTRUSIVE.uses_welfare_passage
    assert all("passage_mean" not in name for name in D.NON_INTRUSIVE.scored_vector_names())
    for definition in (D.CARES_ABOUT_WELFARE, D.CALIBRATED_CONSIDERATION):
        assert definition.main_vector_name == f"{definition.facet_key}_passage_mean_vector"
        assert definition.main_space == "passage_mean" and definition.uses_welfare_passage
        assert definition.vector_set_key_by_vector_name()[definition.main_vector_name] == "passage_mean_raw"
    assert D.NON_INTRUSIVE.vector_set_key_by_vector_name()[D.NON_INTRUSIVE.main_vector_name] == "whole_answer_mean_raw"


def test_per_ai_turn_scoring_is_skipped_with_the_measured_reason():
    documents = {"a": {"kind": "corpus", "content": "Plain prose with no speaker markers."},
                 "b": {"kind": "corpus", "content": "Assistant: here is a marked turn\nUser: ok"},
                 "c": {"kind": "rewrite", "content": "Assistant: ignored, not a corpus document"}}
    status = analysis.per_ai_turn_scoring_status(documents)
    assert status["available"] is False and status["corpus_documents"] == 2 and status["corpus_documents_with_speaker_markers"] == 1
    assert "cannot be located deterministically" in status["reason"] and "1 of 2" in status["reason"]


# ---------------------------------------------------------------------------------------------------
# Scoring run folder
# ---------------------------------------------------------------------------------------------------

def make_new_facet_documents(definition, sources=2, extra_corpus=1, source_numbers=None):
    corpus = [{"document_id": f"matrix_{n:06d}", "source_document_id": f"matrix_{n:06d}", "rewrite_direction": None, "content": f"corpus text {n}",
               "language": "English", "variables": {}, "kind": "corpus"} for n in range(1, sources + extra_corpus + 1)]
    new = []
    for n in (source_numbers if source_numbers is not None else range(1, sources + 1)):
        source = f"matrix_{n:06d}"
        for arm in arms.arm_names_for(definition):
            new.append({"document_id": arms.rewrite_document_id_for(definition, arm, source),
                        "source_document_id": arms.comparison_document_id_for(definition, arm, source),
                        "comparison_document_id": arms.comparison_document_id_for(definition, arm, source),
                        "rewrite_direction": arm, "content": f"{arm} text {n}", "language": "English", "variables": {},
                        "facet": definition.facet_key, "arm": arm, "original_corpus_source_document_id": source,
                        "expected_sign": arms.expected_sign_for(definition, arm), "below_minimum_changed_units": False})
    return corpus, new


@FACET_PARAMETERS
def test_assembly_for_a_new_facet_scores_corpus_and_rewrites_only_and_chains_arm_three_on_arm_one(definition):
    corpus, new = make_new_facet_documents(definition)
    documents = builder.assemble_documents(corpus, [], new, builder.expected_counts_for(3, 2, definition), 2, definition)
    by_id = {d["document_id"]: d for d in documents}
    assert len(documents) == 3 + 2 * len(arms.arm_names_for(definition)) and builder.expected_counts_for(3, 2, definition)["existing_rewrites"] == 0
    arm_three = by_id[arms.rewrite_document_id_for(definition, REMOVED, "matrix_000001")]
    arm_one_id = arms.rewrite_document_id_for(definition, AWAY, "matrix_000001")
    assert arm_three["source_document_id"] == arm_one_id and arm_one_id in by_id and arm_three["expected_sign"] == 1
    assert by_id[arm_one_id]["source_document_id"] == "matrix_000001" and by_id[arm_one_id]["expected_sign"] == -1
    assert all(d["source_document_id"] in by_id for d in documents)


def test_assembly_refuses_arm_three_compared_with_the_original_a_missing_arm_a_wrong_count_and_another_facets_rewrites():
    definition = D.CARES_ABOUT_WELFARE
    corpus, new = make_new_facet_documents(definition)
    expected = builder.expected_counts_for(3, 2, definition)
    wrong = [dict(record) for record in new]
    next(r for r in wrong if r["arm"] == REMOVED)["source_document_id"] = "matrix_000001"       # compared with the original: the bug to catch
    with pytest.raises(ValueError):
        builder.assemble_documents(corpus, [], wrong, expected, 2, definition)
    with pytest.raises(SystemExit):
        builder.assemble_documents(corpus, [], new[:-1], expected, 2, definition)
    missing_arm_one = [r for r in new if not (r["arm"] == AWAY and r["original_corpus_source_document_id"] == "matrix_000001")]
    with pytest.raises(SystemExit):
        builder.assemble_documents(corpus, [], missing_arm_one, expected, 2, definition)
    foreign = [{**r, "facet": D.NON_INTRUSIVE.facet_key} for r in new]
    with pytest.raises(ValueError):
        builder.assemble_documents(corpus, [], foreign, expected, 2, definition)
    with pytest.raises(SystemExit):
        builder.assemble_documents(corpus, [], new, builder.expected_counts_for(3, 3, definition), 3, definition)


def corpus_file_record(number):
    return {"doc_id": f"matrix_{number:06d}", "content": f"corpus file text {number}", "language": "English", "variables": {}}


@FACET_PARAMETERS
def test_a_source_outside_the_previous_run_is_scored_as_a_corpus_document_and_is_its_rewrites_comparison_document(definition):
    previous, _ = make_new_facet_documents(definition, sources=2, extra_corpus=1)          # previous run: matrix_000001 to 000003
    _, new = make_new_facet_documents(definition, source_numbers=[2, 9])                   # 9 is not in the previous run
    expected = builder.expected_counts_for(builder.planned_number_of_corpus_documents(previous, new), 2, definition)
    assert expected["corpus"] == 4
    documents = builder.assemble_documents(previous, [], new, expected, 2, definition, {"matrix_000009": corpus_file_record(9)})
    by_id = {d["document_id"]: d for d in documents}
    assert by_id["matrix_000009"]["kind"] == "corpus" and by_id["matrix_000009"]["content"] == "corpus file text 9"
    assert by_id["matrix_000002"]["content"] == "corpus text 2"                              # an old-20 source keeps the earlier run's text
    away_of_outside = by_id[arms.rewrite_document_id_for(definition, AWAY, "matrix_000009")]
    assert away_of_outside["source_document_id"] == "matrix_000009" and away_of_outside["original_corpus_source_document_id"] == "matrix_000009"
    chained = by_id[arms.rewrite_document_id_for(definition, REMOVED, "matrix_000009")]
    assert chained["source_document_id"] == away_of_outside["document_id"]
    assert all(d["source_document_id"] in by_id for d in documents)
    assert len(documents) == 4 + 2 * len(arms.arm_names_for(definition))


@FACET_PARAMETERS
def test_a_rewrite_whose_source_is_missing_from_the_run_is_refused(definition):
    previous, _ = make_new_facet_documents(definition, sources=2, extra_corpus=1)
    _, new = make_new_facet_documents(definition, source_numbers=[2, 9])
    expected = builder.expected_counts_for(4, 2, definition)
    with pytest.raises(SystemExit, match="matrix_000009"):
        builder.assemble_documents(previous, [], new, expected, 2, definition, None)       # no corpus file given
    with pytest.raises(SystemExit, match="matrix_000009"):
        builder.assemble_documents(previous, [], new, expected, 2, definition, {"matrix_000008": corpus_file_record(8)})  # not in it either


@FACET_PARAMETERS
def test_no_corpus_document_is_scored_twice_when_sources_overlap_the_previous_run_and_each_other(definition):
    previous, _ = make_new_facet_documents(definition, sources=2, extra_corpus=1)
    _, new = make_new_facet_documents(definition, source_numbers=[1, 2, 9, 10])             # 1 and 2 are in the previous run
    corpus_file = {f"matrix_{n:06d}": corpus_file_record(n) for n in range(1, 12)}          # the corpus file holds every one of them
    expected = builder.expected_counts_for(builder.planned_number_of_corpus_documents(previous, new), 4, definition)
    documents = builder.assemble_documents(previous, [], new, expected, 4, definition, corpus_file)
    corpus_ids = [d["document_id"] for d in documents if d["kind"] == "corpus"]
    assert sorted(corpus_ids) == ["matrix_000001", "matrix_000002", "matrix_000003", "matrix_000009", "matrix_000010"]
    assert len(corpus_ids) == len(set(corpus_ids)) and len({d["document_id"] for d in documents}) == len(documents)
    # a previous run that itself scored a corpus document twice is refused rather than silently kept
    with pytest.raises(SystemExit):
        builder.assemble_documents(previous + [dict(previous[0])], [], new, expected, 4, definition, corpus_file)


def test_the_corpus_spread_and_windows_are_over_every_corpus_document_scored_and_the_set_is_recorded():
    documents = {"matrix_000001": {"kind": "corpus"}, "matrix_000009": {"kind": "corpus"}, "rewrite_a": {"kind": "rewrite"}}
    configuration = {"corpus_documents_scored": {"previous_scoring_run_document_ids": ["matrix_000001"],
                                                 "added_from_corpus_file_document_ids": ["matrix_000009"]}}
    used = analysis.corpus_documents_used(documents, configuration)
    assert used["document_ids"] == ["matrix_000001", "matrix_000009"] and used["number_of_documents"] == 2
    assert used["added_from_corpus_file_document_ids"] == ["matrix_000009"] and "all corpus documents" in used["rule"]


def test_the_scoring_estimate_grows_with_the_corpus_documents_and_rewrites_planned():
    from persona_vector_projection import estimate_content_controlled_facet_scoring_cost_ceilings as estimator
    old = estimator.estimate_for_planned_documents(20, 40, 1500)
    new = estimator.estimate_for_planned_documents(25, 40, 1500)
    assert new["documents"] == 65 and old["documents"] == 60
    assert new["document_tokens_to_score_including_chat_template_context"] > old["document_tokens_to_score_including_chat_template_context"]
    assert new["ceiling_gpu_dollars"] >= old["ceiling_gpu_dollars"]


def write_vector_set(folder, definition, layers=5, hidden=8, chosen=3, missing_key=None, with_layer=True):
    folder.mkdir(parents=True)
    generator = np.random.default_rng(0)
    files = {}
    for key in definition.vector_set_key_by_vector_name().values():
        if key == missing_key:
            continue
        name = f"v_{key}_2026-11-01_00-00.npy"
        np.save(folder / name, generator.normal(size=(layers, hidden)).astype(np.float32))
        files[key] = name
    metadata = {"model_name": "tiny", "model_revision": "", "vector_files": files,
                "layer_choice": {"layer_by_facet": {definition.facet_key: chosen} if with_layer else {}}}
    (folder / "persona_vector_metadata_2026-11-01_00-00.json").write_text(json.dumps(metadata))


@FACET_PARAMETERS
def test_vector_set_is_read_at_the_facets_chosen_layer_and_a_missing_vector_or_layer_stops_the_build(tmp_path, definition):
    write_vector_set(tmp_path / "set", definition)
    layer, paths, _, _ = builder.read_vector_set(tmp_path / "set", definition)
    assert layer == 3 and set(paths) == set(definition.scored_vector_names()) and all(p.exists() for p in paths.values())
    write_vector_set(tmp_path / "broken", definition, missing_key=f"{definition.main_space}_raw")
    with pytest.raises(KeyError):
        builder.read_vector_set(tmp_path / "broken", definition)
    write_vector_set(tmp_path / "no_layer", definition, with_layer=False)
    with pytest.raises(SystemExit):
        builder.read_vector_set(tmp_path / "no_layer", definition)


def test_dry_run_of_the_builder_for_a_new_facet_writes_nothing_and_names_the_facets_vectors(tmp_path, capsys):
    definition = D.NON_INTRUSIVE
    corpus, new = make_new_facet_documents(definition, sources=10, extra_corpus=10)
    previous_folder = tmp_path / "previous"
    (previous_folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True)
    write_jsonl(previous_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl", corpus)
    (previous_folder / "run_configuration_2026-10-31_11-20.json").write_text(json.dumps({"model_name": "tiny", "model_revision": ""}))
    write_jsonl(tmp_path / "preachiness.jsonl", [])
    write_jsonl(tmp_path / "new.jsonl", new)
    write_vector_set(tmp_path / "set", definition)
    arguments = builder.parse_arguments(["--facet", "non_intrusive", "--vector-set-folder", str(tmp_path / "set"), "--new-rewrites-jsonl",
                                         str(tmp_path / "new.jsonl"), "--previous-scoring-run-folder", str(previous_folder),
                                         "--preachiness-rewrites-jsonl", str(tmp_path / "preachiness.jsonl"),
                                         "--runs-directory", str(tmp_path / "runs"), "--dry-run"])
    assert arguments.definition is definition and arguments.run_name == definition.scoring_run_name
    run_folder, configuration, estimate = builder.build_run_folder(arguments)
    assert run_folder is None and not (tmp_path / "runs").exists()
    assert configuration["documents_by_group"] == {"corpus": 20, "existing_rewrites": 0, "new_rewrites": 40}
    assert configuration["vector_names"] == definition.scored_vector_names() and configuration["main_vector_name"] == definition.main_vector_name
    assert configuration["facet_key"] == definition.facet_key and configuration["expected_sign_by_arm"][REMOVED] == 1
    assert estimate["documents"] == 60 and "document tokens a second" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------
# Extraction criteria evaluated from the vector set's metadata
# ---------------------------------------------------------------------------------------------------

def metadata_for(definition, chosen_layer=1, held_out_accuracy=0.9, cross_half_a_on_b=0.9, cross_half_b_on_a=0.9, kept=18, total=20,
                 projected_out=0.8, two_by_two=None):
    """A vector-set metadata dict in the shape compute_content_controlled_facet_vectors_from_activations writes, for two layers."""
    space = definition.main_space
    per_tag = {"tag_a": {"held_out_negative_answers": 8, "accuracy_per_layer": [0.5, 0.9]},
               "tag_b": {"held_out_negative_answers": 3, "accuracy_per_layer": [0.5, 0.2]}}     # fewer than 5 held-out negatives: reported only
    entry = {space: {"variant_held_out_metrics": {"raw": {"held_out_accuracy_per_layer": [0.5, held_out_accuracy], "by_negative_tag": per_tag}},
                     "split_half_cosine_half_A_against_half_B_per_layer": [0.0, 0.1], "questions_kept": kept,
                     "cross_half_accuracy": {"vector_from_half_A_evaluated_on_half_B": {"accuracy_per_layer": [0.5, cross_half_a_on_b]},
                                             "vector_from_half_B_evaluated_on_half_A_reported_only": {"accuracy_per_layer": [0.5, cross_half_b_on_a]}}},
             "training_questions_total": total}
    if definition.has_confidence_control:
        entry["confidence_control"] = {
            "held_out_pole_classification_calibration_confidence_direction_projected_out": {"accuracy_per_layer": [0.5, projected_out]},
            "two_by_two_held_out_check": {"by_layer_and_vector": {str(chosen_layer): {"calibration": two_by_two or {
                "interaction_contrast_calibrated_minus_miscalibrated": 2.0, "interaction_contrast_95_percent_bootstrap_interval": [1.0, 3.0],
                "main_effect_of_confidence_confident_minus_hedged": 0.4}}}}}
    return {"per_facet": {definition.facet_key: entry}}


@FACET_PARAMETERS
def test_extraction_criteria_are_evaluated_from_the_vector_set_metadata_and_pass_when_every_counted_one_does(definition):
    criteria = shipped_criteria(definition)
    lines = analysis.extraction_lines_for_vector(metadata_for(definition), definition.main_vector_name, 1, definition, criteria)
    verdict = statistics_module.overall_verdict(lines)
    assert verdict["verdict"] == "passes", verdict["counted_criteria"]
    assert [item["criterion"] for item in verdict["reported_not_counted"]] == ["extraction__split_half_cosine__reported"]
    names = {item["criterion"] for item in verdict["counted_criteria"]}
    assert {"extraction__held_out_accuracy", "extraction__held_out_accuracy_by_negative_tag", "extraction__cross_half_accuracy",
            "extraction__training_questions_with_kept_items"} <= names
    assert ("calibrated_consideration__two_by_two_rule" in names) == definition.has_confidence_control


@FACET_PARAMETERS
def test_cross_half_criterion_reads_half_a_on_half_b_and_never_the_reverse(definition):
    criteria = shipped_criteria(definition)
    low_forward = analysis.extraction_lines_for_vector(metadata_for(definition, cross_half_a_on_b=0.7, cross_half_b_on_a=0.99),
                                                       definition.main_vector_name, 1, definition, criteria)
    assert {l["criterion"]: l["passed"] for l in low_forward}["extraction__cross_half_accuracy"] is False
    assert statistics_module.overall_verdict(low_forward)["verdict"] == "fails"
    high_forward = analysis.extraction_lines_for_vector(metadata_for(definition, cross_half_a_on_b=0.95, cross_half_b_on_a=0.5),
                                                        definition.main_vector_name, 1, definition, criteria)
    assert {l["criterion"]: l["passed"] for l in high_forward}["extraction__cross_half_accuracy"] is True


def test_metadata_without_the_new_numbers_leaves_their_lines_undecided_and_never_a_pass():
    definition = D.CALIBRATED_CONSIDERATION
    metadata = metadata_for(definition)
    entry = metadata["per_facet"][definition.facet_key]
    del entry["training_questions_total"], entry[definition.main_space]["cross_half_accuracy"], entry["confidence_control"]
    lines = analysis.extraction_lines_for_vector(metadata, definition.main_vector_name, 1, definition, shipped_criteria(definition))
    by_name = {l["criterion"]: l["passed"] for l in lines if l["counted"]}
    assert by_name["extraction__cross_half_accuracy"] is None and by_name["extraction__training_questions_with_kept_items"] is None
    assert by_name["calibrated_consideration__two_by_two_rule"] is None and by_name["calibrated_consideration__projected_out_held_out_accuracy"] is None
    assert statistics_module.overall_verdict(lines)["verdict"] == "undecided"
    assert all(l["passed"] is None for l in analysis.extraction_lines_for_vector(None, definition.main_vector_name, 1, definition, shipped_criteria(definition)))


def test_the_calibrated_two_by_two_and_projected_out_lines_come_from_the_metadata_at_the_chosen_layer():
    definition = D.CALIBRATED_CONSIDERATION
    criteria = shipped_criteria(definition)
    confidence_like = {"interaction_contrast_calibrated_minus_miscalibrated": 0.3, "interaction_contrast_95_percent_bootstrap_interval": [-0.2, 0.8],
                       "main_effect_of_confidence_confident_minus_hedged": 2.0}
    lines = analysis.extraction_lines_for_vector(metadata_for(definition, two_by_two=confidence_like, projected_out=0.7), definition.main_vector_name,
                                                 1, definition, criteria)
    by_name = {l["criterion"]: l["passed"] for l in lines if l["counted"]}
    assert by_name["calibrated_consideration__two_by_two_rule"] is False and by_name["calibrated_consideration__projected_out_held_out_accuracy"] is False
    assert statistics_module.overall_verdict(lines)["verdict"] == "fails"
