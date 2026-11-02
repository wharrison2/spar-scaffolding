"""Tests for bugs that would invalidate the moralizing delivery positive controls and their scoring: which document each arm is
compared with, the expected sign per arm, the saved activations, the scoring run folder, and the criteria computations on synthetic
data with known answers. Local and free: a fake rewrite client, no API call, no .env read, no pod. The tiny random Gemma 4 model uses
the real tokenizer from the local Hugging Face cache (those tests are skipped when it is not cached).

    python3 -m pytest scripts/persona_vector_projection/test_moralizing_delivery_positive_controls_and_scoring.py -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import build_moralizing_delivery_scoring_run_folder as builder  # noqa: E402
from persona_vector_projection import make_facet_positive_control_rewrites as facet_rewrites  # noqa: E402
from persona_vector_projection import make_moralizing_delivery_positive_control_rewrites as rewriter  # noqa: E402
from persona_vector_projection import moralizing_delivery_positive_control_arms as arms  # noqa: E402
from persona_vector_projection import moralizing_delivery_positive_control_statistics as statistics_module  # noqa: E402
from persona_vector_projection import run_scoring_pod_stages as stages  # noqa: E402
from persona_vector_projection import saved_activation_projection_locally as saved_activations  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering  # noqa: E402
from persona_vector_projection import scoring_pod_test_support as support  # noqa: E402
from persona_vector_projection import scoring_vector_manifest_for_named_vectors as manifest_module  # noqa: E402
from persona_vector_projection import test_facet_positive_control_rewrites as fakes  # noqa: E402
from persona_vector_projection.scoring_pod_projection_recording import LayerProjectionRecorder, project_documents  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402

ARM_MARKERS = {arms.ARM_MORALIZING_FRAMING_ONLY: "so that the AI moralizes when it makes them",
               arms.ARM_PLAIN_WELFARE_CONTENT_ADDED: "Add welfare content to what the AI says",
               arms.ARM_FACET_NEUTRAL_REWORDING: "changes about as much as a rewrite that made the AI preachier or plainer would",
               arms.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE: "so that the AI states the same points plainly"}


def arm_of_prompt(prompt):
    matches = [arm for arm, marker in ARM_MARKERS.items() if marker in prompt]
    assert len(matches) == 1, matches
    return matches[0]


# ---------------------------------------------------------------------------------------------------
# Arms: comparison documents and expected signs
# ---------------------------------------------------------------------------------------------------

def test_every_arm_compares_with_the_original_document_except_the_removal_arm_which_compares_with_the_preachier_rewrite():
    assert arms.comparison_document_id(arms.ARM_MORALIZING_FRAMING_ONLY, "matrix_000034") == "matrix_000034"
    assert arms.comparison_document_id(arms.ARM_PLAIN_WELFARE_CONTENT_ADDED, "matrix_000034") == "matrix_000034"
    assert arms.comparison_document_id(arms.ARM_FACET_NEUTRAL_REWORDING, "matrix_000034") == "matrix_000034"
    assert arms.comparison_document_id(arms.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE, "matrix_000034") == "matrix_000034__preachier"
    with pytest.raises(ValueError):
        arms.comparison_document_id("not_an_arm", "matrix_000034")


def test_rewrite_ids_name_the_original_source_and_can_be_parsed_back():
    for arm in arms.ARMS:
        document_id = arms.rewrite_document_id(arm, "matrix_000034")
        assert document_id == f"matrix_000034__moralizing_delivery__{arm}"
        assert arms.arm_of_rewrite_document_id(document_id) == arm
        assert arms.original_source_document_id_of_rewrite(document_id) == "matrix_000034"
    assert arms.arm_of_rewrite_document_id("matrix_000034__preachier") is None
    assert arms.original_source_document_id_of_rewrite("matrix_000034__preachier") == "matrix_000034"


def test_expected_sign_per_arm_moralizing_is_negative_because_the_vector_points_from_moralizing_to_plain():
    assert arms.EXPECTED_SIGN_BY_ARM == {arms.ARM_MORALIZING_FRAMING_ONLY: -1, arms.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE: 1,
                                         arms.ARM_PLAIN_WELFARE_CONTENT_ADDED: 0, arms.ARM_FACET_NEUTRAL_REWORDING: 0}
    assert arms.MORALIZING_DIRECTION_SIGN == -1


def test_the_four_instruction_files_exist_render_without_unfilled_slots_and_the_neutral_arm_reuses_the_non_intrusive_file():
    templates = rewriter.load_instruction_templates()
    document = {"doc_id": "d", "content": "Body with {braces} inside.", "language": "English", "variables": {}}
    for arm in arms.ARMS:
        template = templates[facet_rewrites.instruction_file_name(arms.REWRITE_FACET_NAME, arm)]
        prompt = facet_rewrites.render_instruction(template, document, 60, target_changed_units=300)
        assert ARM_MARKERS[arm] in prompt and "about 300 words" in prompt
        assert prompt.rstrip("\n").endswith("Document:\nBody with {braces} inside.")
        for slot in ("{target_changed_amount}", "{lowest_changed_amount}", "{highest_changed_amount}", "{minimum_changed_amount}",
                     "{document_text}"):
            assert slot not in prompt
        assert arm_of_prompt(prompt) == arm
    assert arms.INSTRUCTION_FILE_BY_ARM[arms.ARM_FACET_NEUTRAL_REWORDING].startswith("positive_control_rewrite_instruction__non_intrusive__")
    assert set(arms.INSTRUCTION_FILE_BY_ARM.values()) <= set(rewriter.instruction_file_names_used())


# ---------------------------------------------------------------------------------------------------
# The rewrite script, with a fake client
# ---------------------------------------------------------------------------------------------------

@pytest.fixture
def rewrite_inputs(tmp_path):
    corpus = fakes.make_corpus(12)
    records = []
    for index in range(4):
        source = corpus[index]
        for direction, how_many in (("preachier", 50 + 10 * index), ("plainer", 40)):
            records.append({"document_id": f"{source['doc_id']}__{direction}", "source_document_id": source["doc_id"],
                            "rewrite_direction": direction, "content": fakes.change_words(source["content"], how_many, direction),
                            "language": source["language"], "variables": source["variables"]})
    corpus_path, preachiness_path = tmp_path / "corpus.jsonl", tmp_path / "preachiness.jsonl"
    write_jsonl(corpus_path, corpus)
    write_jsonl(preachiness_path, records)
    return corpus, records, corpus_path, preachiness_path


def run_rewriter(tmp_path, rewrite_inputs, client, extra=()):
    _, _, corpus_path, preachiness_path = rewrite_inputs
    return rewriter.main(["--corpus-path", str(corpus_path), "--preachiness-rewrites-jsonl", str(preachiness_path), "--workers", "1",
                          "--minimum-changed-units", "30", "--env-path", str(tmp_path / "no_such_env_file"), *extra],
                         client=client, runs_directory_override=tmp_path / "runs")


def test_removal_arm_is_applied_to_the_preachier_rewrite_and_compared_with_it_and_the_others_use_the_original(tmp_path, rewrite_inputs):
    corpus, preachiness, _, _ = rewrite_inputs
    corpus_by_id, preachier_by_source = {d["doc_id"]: d for d in corpus}, {r["source_document_id"]: r for r in preachiness if r["rewrite_direction"] == "preachier"}
    client = fakes.FakeClient(lambda prompt: fakes.make_response(fakes.change_words(fakes.document_text_in(prompt), 80, "new")))
    run_folder = run_rewriter(tmp_path, rewrite_inputs, client)
    records = read_jsonl(sorted(run_folder.glob("moralizing_delivery_positive_control_rewrites_*.jsonl"))[0])
    assert len(records) == 16 and len(client.calls) == 16
    prompt_document_by_arm_and_source = {}
    for call in client.calls:
        prompt = call["messages"][0]["content"]
        text = fakes.document_text_in(prompt)
        arm = arm_of_prompt(prompt)
        source = next(s for s in corpus_by_id if text in (corpus_by_id[s]["content"], preachier_by_source.get(s, {}).get("content")))
        prompt_document_by_arm_and_source[(arm, source)] = text
    for record in records:
        arm, original = record["arm"], record["original_corpus_source_document_id"]
        if arm == arms.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE:
            assert prompt_document_by_arm_and_source[(arm, original)] == preachier_by_source[original]["content"]
            assert record["source_document_id"] == f"{original}__preachier" and record["expected_sign"] == 1
            counts_against = preachier_by_source[original]["content"]
        else:
            assert prompt_document_by_arm_and_source[(arm, original)] == corpus_by_id[original]["content"]
            assert record["source_document_id"] == original
            counts_against = corpus_by_id[original]["content"]
        assert record["document_id"] == arms.rewrite_document_id(arm, original) and record["facet"] == "moralizing_delivery"
        assert record["expected_sign"] == arms.EXPECTED_SIGN_BY_ARM[arm]
        changed = facet_rewrites.count_changed_units(counts_against, record["content"], record["language"])
        assert record["changed_rewrite_units"] == changed["changed_rewrite_units"] and record["source_units"] == changed["source_units"]


def test_every_arm_of_a_source_asks_for_the_size_of_that_sources_preachier_rewrite(tmp_path, rewrite_inputs):
    corpus, preachiness, _, _ = rewrite_inputs
    corpus_by_id = {d["doc_id"]: d for d in corpus}
    client = fakes.FakeClient(lambda prompt: fakes.make_response(fakes.change_words(fakes.document_text_in(prompt), 80, "new")))
    run_folder = run_rewriter(tmp_path, rewrite_inputs, client)
    records = read_jsonl(sorted(run_folder.glob("moralizing_delivery_positive_control_rewrites_*.jsonl"))[0])
    for source_id, expected_size in {r["source_document_id"]: facet_rewrites.changed_units_for_size_matching(
            facet_rewrites.count_changed_units(corpus_by_id[r["source_document_id"]]["content"], r["content"], r["language"]))
            for r in preachiness if r["rewrite_direction"] == "preachier"}.items():
        arm_records = [r for r in records if r["original_corpus_source_document_id"] == source_id]
        assert sorted(r["arm"] for r in arm_records) == sorted(arms.ARMS)
        assert {r["target_changed_units"] for r in arm_records} == {expected_size}
        assert f"about {expected_size} words" in next(c["messages"][0]["content"] for c in client.calls
                                                       if fakes.document_text_in(c["messages"][0]["content"]) == corpus_by_id[source_id]["content"])


def test_retry_regenerates_a_rewrite_below_the_minimum_and_the_export_keeps_flagged_rewrites(tmp_path, rewrite_inputs):
    client = fakes.FakeClient(fakes.stubborn_then_generous)
    run_folder = run_rewriter(tmp_path, rewrite_inputs, client)
    records = read_jsonl(sorted(run_folder.glob("moralizing_delivery_positive_control_rewrites_*.jsonl"))[0])
    assert all(r["number_of_attempts"] == 2 and not r["below_minimum_changed_units"] for r in records) and len(client.calls) == 32
    stubborn = fakes.FakeClient(lambda prompt: fakes.make_response(fakes.change_words(fakes.document_text_in(prompt), 2, "tiny")))
    second = rewriter.main(["--corpus-path", str(rewrite_inputs[2]), "--preachiness-rewrites-jsonl", str(rewrite_inputs[3]), "--workers", "1",
                            "--minimum-changed-units", "30", "--maximum-regenerations", "1", "--run-name", "flagged_run",
                            "--env-path", str(tmp_path / "no_such_env_file")], client=stubborn, runs_directory_override=tmp_path / "runs")
    exported = read_jsonl(sorted(second.glob("scoring_pod_extra_documents_*.jsonl"))[0])
    assert len(exported) == 16 and all(r["below_minimum_changed_units"] for r in exported)   # kept, and carrying the flag


def test_resume_makes_no_new_call_for_finished_rewrites(tmp_path, rewrite_inputs):
    client = fakes.FakeClient(lambda prompt: fakes.make_response(fakes.change_words(fakes.document_text_in(prompt), 80, "new")))
    run_folder = run_rewriter(tmp_path, rewrite_inputs, client)
    calls_after_first_run = len(client.calls)
    run_rewriter(tmp_path, rewrite_inputs, client, ["--resume-folder", str(run_folder)])
    assert len(client.calls) == calls_after_first_run


def test_dry_run_counts_calls_estimates_cost_and_never_touches_a_client(tmp_path, rewrite_inputs):
    folder = rewriter.main(["--corpus-path", str(rewrite_inputs[2]), "--preachiness-rewrites-jsonl", str(rewrite_inputs[3]), "--dry-run",
                            "--env-path", str(tmp_path / "no_such_env_file")], client=None, runs_directory_override=tmp_path / "runs")
    estimate = json.loads(sorted(folder.glob("dry_run_cost_estimate_and_ceiling_*.json"))[0].read_text())
    assert estimate["calls_planned"] == 16 and set(estimate["calls_planned_by_arm"].values()) == {4}
    assert estimate["ceiling_cost_dollars"] > estimate["expected_cost_dollars"] > 0
    assert not list(folder.glob("moralizing_delivery_positive_control_rewrites_*.jsonl"))


# ---------------------------------------------------------------------------------------------------
# Statistics on synthetic data with known answers
# ---------------------------------------------------------------------------------------------------

def test_changed_span_shift_is_changed_minus_unchanged_after_token_64_and_ignores_earlier_changes():
    comparison_ids = list(range(400))
    rewrite_ids = list(range(400))
    for position in list(range(100, 140)) + list(range(10, 20)):   # one change inside, one before token 64
        rewrite_ids[position] = 1000 + position
    values = np.zeros(400)
    values[100:140] = -3.0
    values[10:20] = -50.0          # must not count: inside the first 64 tokens
    shift, mask, tail = statistics_module.changed_span_shift(values, comparison_ids, rewrite_ids)
    assert shift == pytest.approx(-3.0) and mask.sum() == 40 and len(tail) == 400 - 64
    values[100:140] = 2.0
    assert statistics_module.changed_span_shift(values, comparison_ids, rewrite_ids)[0] == pytest.approx(2.0)


def test_changed_span_shift_is_nan_when_nothing_after_token_64_changed():
    ids = list(range(200))
    rewrite_ids = list(ids)
    rewrite_ids[5] = 999
    assert np.isnan(statistics_module.changed_span_shift(np.zeros(200), ids, rewrite_ids)[0])


def test_circular_shift_p_is_small_for_a_real_dip_and_large_for_noise():
    generator = np.random.default_rng(1)
    noise = generator.normal(size=600)
    mask = np.zeros(600, dtype=bool)
    mask[200:260] = True
    dipped = noise.copy()
    dipped[200:260] -= 4.0
    observed = dipped[mask].mean() - dipped[~mask].mean()
    p_values = statistics_module.circular_shift_p_values(dipped, mask, observed, np.random.default_rng(2))
    assert p_values["p_lower"] < 0.01 and p_values["p_upper"] > 0.9 and p_values["p_two_sided"] < 0.02
    observed_noise = noise[mask].mean() - noise[~mask].mean()
    p_noise = statistics_module.circular_shift_p_values(noise, mask, observed_noise, np.random.default_rng(3))
    assert min(p_noise["p_lower"], p_noise["p_upper"]) > 0.01
    assert np.isnan(statistics_module.circular_shift_p_values(noise, mask, float("nan"), np.random.default_rng(3))["p_lower"])


def test_attaching_rewrite_tokens_to_comparison_units():
    comparison = [1, 2, 3, 4, 5, 6]
    rewrite = [1, 2, 9, 9, 9, 5, 6, 7]    # 3,4 replaced by 9,9,9; 7 inserted at the end
    owner = statistics_module.attach_rewrite_tokens_to_comparison_units(comparison, rewrite)
    assert list(owner[:2]) == [0, 1] and list(owner[5:7]) == [4, 5] and owner[7] == 5
    assert all(2 <= unit <= 3 for unit in owner[2:5])


def test_whole_document_bootstrap_recovers_a_constant_shift_exactly_and_reports_the_direction():
    ids = list(range(500))
    comparison = np.random.default_rng(0).normal(size=500)
    shift, resampled = statistics_module.whole_document_shift_with_block_bootstrap(
        comparison, comparison - 1.5, ids, ids, np.random.default_rng(1), draws=500)
    assert shift == pytest.approx(-1.5) and np.allclose(resampled, -1.5)
    p_values = statistics_module.bootstrap_p_values(resampled)
    assert p_values["p_negative_direction"] == 0.0 and p_values["p_positive_direction"] == 1.0
    _, no_shift = statistics_module.whole_document_shift_with_block_bootstrap(
        comparison, comparison.copy(), ids, ids, np.random.default_rng(1), draws=200)
    assert np.allclose(no_shift, 0.0)


def test_whole_document_bootstrap_with_several_vectors_matches_one_vector_at_a_time():
    ids = list(range(300))
    rewrite_ids = ids[:100] + [900, 901, 902] + ids[100:]
    generator = np.random.default_rng(5)
    comparison, rewrite = generator.normal(size=(300, 2)), generator.normal(size=(303, 2))
    both_shift, both = statistics_module.whole_document_shift_with_block_bootstrap(comparison, rewrite, ids, rewrite_ids, np.random.default_rng(9), draws=300)
    first_shift, first = statistics_module.whole_document_shift_with_block_bootstrap(comparison[:, 0], rewrite[:, 0], ids, rewrite_ids, np.random.default_rng(9), draws=300)
    assert both_shift[0] == pytest.approx(first_shift) and np.allclose(both[:, 0], first)


def test_standardised_effect_wilcoxon_and_direction_counts_on_known_numbers():
    shifts = [-1.0, -2.0, -3.0, -4.0, 1.0]
    assert statistics_module.standardised_effect(shifts, -1) == pytest.approx(-np.mean(shifts) / np.std(shifts, ddof=1))
    assert statistics_module.count_in_direction(shifts, -1) == 4 and statistics_module.count_in_direction(shifts, 1) == 1
    assert statistics_module.wilcoxon_p([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]) == pytest.approx(2 / 1024)
    assert np.isnan(statistics_module.wilcoxon_p([0.0, 0.0])) and np.isnan(statistics_module.standardised_effect([1.0]))


def make_table(span, whole, circular_lower=None, circular_upper=None, circular_two=None, bootstrap_negative=None, bootstrap_positive=None):
    count = len(span)
    return pd.DataFrame({
        "changed_span_shift": span, "whole_document_shift": whole,
        "circular_p_lower": circular_lower if circular_lower is not None else [0.5] * count,
        "circular_p_upper": circular_upper if circular_upper is not None else [0.5] * count,
        "circular_p_two_sided": circular_two if circular_two is not None else [1.0] * count,
        "whole_bootstrap_p_negative_direction": bootstrap_negative if bootstrap_negative is not None else [0.5] * count,
        "whole_bootstrap_p_positive_direction": bootstrap_positive if bootstrap_positive is not None else [0.5] * count})


def passing_summaries():
    framing_span = [-3.0, -2.0, -4.0, -2.5, -3.5, -1.0, -5.0, -2.2, -3.1, 0.5]
    removed_span = [3.0, 2.0, 4.0, 2.5, 3.5, 1.0, 5.0, 2.2, 3.1, -0.5]
    plain_span = [-0.3, 0.2, -0.1, 0.1, 0.4, -0.2, 0.0, 0.3, -0.4, 0.1]
    neutral_span = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0, -0.1, 0.2]
    tables = {arms.ARM_MORALIZING_FRAMING_ONLY: make_table(framing_span, [-1.0] * 7 + [0.5] * 3),
              arms.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE: make_table(removed_span, [1.0] * 10),
              arms.ARM_PLAIN_WELFARE_CONTENT_ADDED: make_table(plain_span, [0.0] * 10),
              arms.ARM_FACET_NEUTRAL_REWORDING: make_table(neutral_span, [0.0] * 10)}
    return tables


def verdicts_by_criterion(tables):
    summaries = {arm: statistics_module.summarise_arm(table, arm) for arm, table in tables.items()}
    return {line["criterion"]: line["passed"] for line in statistics_module.evaluate_positive_control_criteria(summaries)}


def test_all_positive_control_criteria_pass_on_data_built_to_pass_them():
    verdicts = verdicts_by_criterion(passing_summaries())
    assert len(verdicts) == 8 and all(verdicts.values()), verdicts


def test_framing_arm_needs_8_of_10_negative_and_7_of_10_whole_document():
    tables = passing_summaries()
    tables[arms.ARM_MORALIZING_FRAMING_ONLY] = make_table([-3.0] * 7 + [1.0] * 3, [-1.0] * 6 + [0.5] * 4)   # 7 negative, whole 6
    verdicts = verdicts_by_criterion(tables)
    assert verdicts["moralizing_framing_only__changed_span_direction"] is False
    assert verdicts["moralizing_framing_only__whole_document_direction"] is False
    tables[arms.ARM_MORALIZING_FRAMING_ONLY] = make_table([-3.0 - i for i in range(8)] + [1.0] * 2, [-1.0] * 7 + [0.5] * 3)  # 8 negative, whole 7
    verdicts = verdicts_by_criterion(tables)
    assert verdicts["moralizing_framing_only__changed_span_direction"] is True
    assert verdicts["moralizing_framing_only__whole_document_direction"] is True


def test_the_removal_arm_is_expected_positive_so_a_negative_shift_fails_it():
    tables = passing_summaries()
    tables[arms.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE] = make_table([-3.0 - i for i in range(10)], [-1.0] * 10)
    verdicts = verdicts_by_criterion(tables)
    assert verdicts["moralizing_removed__changed_span_direction"] is False
    assert verdicts["moralizing_removed__changed_span_wilcoxon"] is True   # significant, but the wrong way, so the direction line fails it


def test_wilcoxon_must_be_below_0_05():
    tables = passing_summaries()
    tables[arms.ARM_MORALIZING_FRAMING_ONLY] = make_table([-0.1, -0.2, -0.3, -4.0, -0.2, -0.5, -0.3, -0.4, 5.0, 6.0], [-1.0] * 10)
    assert verdicts_by_criterion(tables)["moralizing_framing_only__changed_span_wilcoxon"] is False


def test_plain_arm_effect_must_be_under_a_third_of_the_framing_effect_and_at_most_3_individually_moralizing():
    tables = passing_summaries()
    framing = tables[arms.ARM_MORALIZING_FRAMING_ONLY]["changed_span_shift"]
    same_spread_with_a_fraction_of_the_mean = lambda fraction: list(framing - framing.mean() * (1 - fraction))  # noqa: E731
    tables[arms.ARM_PLAIN_WELFARE_CONTENT_ADDED] = make_table(same_spread_with_a_fraction_of_the_mean(0.5), [0.0] * 10)   # effect = 1/2 of framing
    assert verdicts_by_criterion(tables)["plain_welfare_content_added__effect_under_a_third_of_framing"] is False
    tables[arms.ARM_PLAIN_WELFARE_CONTENT_ADDED] = make_table(same_spread_with_a_fraction_of_the_mean(0.2), [0.0] * 10)
    assert verdicts_by_criterion(tables)["plain_welfare_content_added__effect_under_a_third_of_framing"] is True
    for significant, expected in ((3, True), (4, False)):
        lower = [0.01] * significant + [0.5] * (10 - significant)
        tables[arms.ARM_PLAIN_WELFARE_CONTENT_ADDED] = make_table([0.1] * 10, [0.0] * 10, circular_lower=lower)
        assert verdicts_by_criterion(tables)["plain_welfare_content_added__few_individually_moralizing"] is expected


def test_neutral_arm_allows_2_significant_in_either_direction_and_a_third_fails():
    tables = passing_summaries()
    for significant, expected in ((2, True), (3, False)):
        two_sided = [0.01] * significant + [0.6] * (10 - significant)
        tables[arms.ARM_FACET_NEUTRAL_REWORDING] = make_table([0.1] * 10, [0.0] * 10, circular_two=two_sided)
        assert verdicts_by_criterion(tables)["facet_neutral_rewording__few_individually_significant"] is expected


def test_extraction_criteria_thresholds():
    accuracy_by_device = {"a": 0.9, "b": 0.7, "c": 0.7, "d": 0.7, "e": 0.69}
    lines = {l["criterion"]: l["passed"] for l in statistics_module.evaluate_extraction_criteria(0.80, accuracy_by_device, 0.80, 15)}
    assert all(lines.values()) and len(lines) == 4
    lines = {l["criterion"]: l["passed"] for l in statistics_module.evaluate_extraction_criteria(0.79, {**accuracy_by_device, "d": 0.5}, 0.79, 14)}
    assert not any(lines.values())
    undecided = statistics_module.evaluate_extraction_criteria(None, None, None, None)
    assert all(line["passed"] is None for line in undecided)


def test_corpus_segment_criterion_needs_5_of_10_read_by_hand_and_is_undecided_until_read():
    window_ids = [f"lowest_window_{n:02d}" for n in range(1, 11)]
    reading = {w: n < 5 for n, w in enumerate(window_ids)}
    assert statistics_module.evaluate_corpus_segment_criterion(reading)["passed"] is True
    reading[window_ids[0]] = False
    assert statistics_module.evaluate_corpus_segment_criterion(reading)["passed"] is False
    assert statistics_module.evaluate_corpus_segment_criterion(None)["passed"] is None
    assert statistics_module.evaluate_corpus_segment_criterion({**reading, window_ids[1]: None})["passed"] is None


def test_lowest_windows_are_ranked_by_distance_below_their_own_documents_mean_and_ignore_the_opening():
    generator = np.random.default_rng(0)
    document_one = generator.normal(scale=0.01, size=400) + 5.0           # high document: its dip is still deep below its own mean
    document_one[200:232] -= 3.0
    document_two = generator.normal(scale=0.01, size=400) - 5.0           # low document: no dip, so no window is far below its mean
    document_two[5:37] -= 40.0                                            # a huge dip inside the opening: ignored
    windows = statistics_module.lowest_windows_below_document_mean({"one": document_one, "two": document_two}, count=3)
    assert windows[0]["document_id"] == "one" and windows[0]["start_token"] == 200
    assert windows[0]["below_document_mean"] < -2.5
    assert all(w["start_token"] >= 64 for w in windows) and windows[0]["below_document_mean"] <= windows[1]["below_document_mean"]
    for first in windows:
        for second in windows:
            if first is not second and first["document_id"] == second["document_id"]:
                assert abs(first["start_token"] - second["start_token"]) >= 32


def test_rewrite_results_for_arm_is_deterministic_per_rewrite_and_independent_of_the_other_rewrites():
    generator = np.random.default_rng(0)
    comparison_ids = list(range(300))
    per_token, token_ids, comparison_of = {}, {}, {}
    for number in range(3):
        rewrite_ids = comparison_ids[:100] + [5000 + 10 * number + k for k in range(30)] + comparison_ids[130:]
        comparison_values = generator.normal(size=(300, 2))
        rewrite_values = comparison_values.copy()
        rewrite_values[100:130, 0] -= 3.0
        per_token[f"c{number}"], per_token[f"r{number}"] = comparison_values, rewrite_values
        token_ids[f"c{number}"], token_ids[f"r{number}"] = comparison_ids, rewrite_ids
        comparison_of[f"r{number}"] = f"c{number}"
    full = statistics_module.rewrite_results_for_arm("a", ["r0", "r1", "r2"], comparison_of, per_token, token_ids, 0, 0, 200, 200)
    alone = statistics_module.rewrite_results_for_arm("a", ["r1"], comparison_of, per_token, token_ids, 0, 0, 200, 200)
    pd.testing.assert_series_equal(full[full["document_id"] == "r1"].iloc[0], alone.iloc[0], check_names=False)
    assert (full["changed_span_shift"] < -2.0).all() and (full["circular_p_lower"] < 0.05).all()
    unaffected = statistics_module.rewrite_results_for_arm("a", ["r0"], comparison_of, per_token, token_ids, 1, 0, 200, 200)
    assert abs(unaffected.iloc[0]["changed_span_shift"]) < 1.0   # the second vector was not shifted


# ---------------------------------------------------------------------------------------------------
# Scoring run folder
# ---------------------------------------------------------------------------------------------------

def make_documents_for_assembly(sources=2, extra_corpus=1):
    corpus = [{"document_id": f"matrix_{n:06d}", "source_document_id": f"matrix_{n:06d}", "rewrite_direction": None, "content": f"corpus text {n}",
               "language": "English", "variables": {}, "kind": "corpus"} for n in range(1, sources + extra_corpus + 1)]
    preachiness = []
    for n in range(1, sources + 1):
        for direction in ("preachier", "plainer"):
            preachiness.append({"document_id": f"matrix_{n:06d}__{direction}", "source_document_id": f"matrix_{n:06d}",
                                "rewrite_direction": direction, "content": f"{direction} text {n}", "language": "English", "variables": {}})
    new = []
    for n in range(1, sources + 1):
        source = f"matrix_{n:06d}"
        for arm in arms.ARMS:
            new.append({"document_id": arms.rewrite_document_id(arm, source), "source_document_id": arms.comparison_document_id(arm, source),
                        "rewrite_direction": arm, "content": f"{arm} text {n}", "language": "English", "variables": {}, "facet": "moralizing_delivery",
                        "arm": arm, "original_corpus_source_document_id": source, "expected_sign": arms.EXPECTED_SIGN_BY_ARM[arm],
                        "below_minimum_changed_units": False})
    return corpus + [{**r, "kind": "rewrite"} for r in preachiness], preachiness, new


def test_assembled_documents_carry_each_rewrites_comparison_document_and_arm():
    previous, preachiness, new = make_documents_for_assembly()
    documents = builder.assemble_documents(previous, preachiness, new, builder.expected_counts_for(3, 2), 2)
    by_id = {d["document_id"]: d for d in documents}
    assert len(documents) == 3 + 4 + 8
    removed = by_id[arms.rewrite_document_id(arms.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE, "matrix_000001")]
    assert removed["source_document_id"] == "matrix_000001__preachier" and removed["expected_sign"] == 1
    assert by_id[arms.rewrite_document_id(arms.ARM_MORALIZING_FRAMING_ONLY, "matrix_000001")]["source_document_id"] == "matrix_000001"
    assert by_id["matrix_000001__preachier"]["source_document_id"] == "matrix_000001"
    assert all(d["source_document_id"] in by_id for d in documents)


def test_assembly_refuses_a_wrong_comparison_document_a_missing_arm_and_changed_preachier_text():
    previous, preachiness, new = make_documents_for_assembly()
    expected = builder.expected_counts_for(3, 2)
    wrong = [dict(record) for record in new]
    removed = next(r for r in wrong if r["arm"] == arms.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE)
    removed["source_document_id"] = removed["original_corpus_source_document_id"]    # compared with the original: the bug to catch
    with pytest.raises(ValueError):
        builder.assemble_documents(previous, preachiness, wrong, expected, 2)
    with pytest.raises(SystemExit):
        builder.assemble_documents(previous, preachiness, new[:-1], expected, 2)
    changed = [dict(record) for record in previous]
    next(r for r in changed if r["document_id"] == "matrix_000001__preachier")["content"] = "different"
    with pytest.raises(ValueError):
        builder.assemble_documents(changed, preachiness, new, expected, 2)


def write_vector_set(folder, layers=5, hidden=8, chosen=3, missing_key=None):
    folder.mkdir(parents=True)
    generator = np.random.default_rng(0)
    files = {}
    for key in arms.VECTOR_SET_KEY_BY_VECTOR_NAME.values():
        if key == missing_key:
            continue
        name = f"v_{key}_2026-11-01_00-00.npy"
        np.save(folder / name, generator.normal(size=(layers, hidden)).astype(np.float32))
        files[key] = name
    metadata = {"model_name": "tiny", "model_revision": "", "vector_files": files,
                "layer_choice": {"layer_by_facet": {"moralizing_delivery": chosen}}}
    (folder / "persona_vector_metadata_2026-11-01_00-00.json").write_text(json.dumps(metadata))


def test_vector_set_is_read_at_the_chosen_layer_and_a_missing_vector_stops_the_build(tmp_path):
    write_vector_set(tmp_path / "set")
    layer, paths, _, _ = builder.read_vector_set(tmp_path / "set")
    assert layer == 3 and set(paths) == set(arms.SCORED_VECTOR_NAMES) and all(p.exists() for p in paths.values())
    write_vector_set(tmp_path / "broken", missing_key="whole_answer_mean_raw")
    with pytest.raises(KeyError):
        builder.read_vector_set(tmp_path / "broken")


def test_recorded_layers_are_the_chosen_layer_and_its_neighbours_clipped_to_the_model():
    assert manifest_module.recorded_layers_around(18, 49) == [16, 17, 18, 19, 20]
    assert manifest_module.recorded_layers_around(3, 5) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        manifest_module.recorded_layers_around(5, 5)


def test_named_vector_loading_refuses_a_zero_vector_and_mixed_shapes(tmp_path):
    inputs = tmp_path / "inputs"
    (inputs / manifest_module.SCORING_VECTORS_SUBFOLDER).mkdir(parents=True)
    np.save(inputs / manifest_module.SCORING_VECTORS_SUBFOLDER / "a.npy", np.ones((5, 4), np.float32))
    np.save(inputs / manifest_module.SCORING_VECTORS_SUBFOLDER / "zero.npy", np.zeros((5, 4), np.float32))
    np.save(inputs / manifest_module.SCORING_VECTORS_SUBFOLDER / "other_shape.npy", np.ones((5, 6), np.float32))
    def write_manifest(entries):
        (inputs / "scoring_vector_manifest_2026-11-01_00-00.json").write_text(json.dumps(
            {"chosen_layer": 2, "vectors": {name: {"file": f"{name}.npy"} for name in entries}}))
    write_manifest(["a"])
    loaded = manifest_module.load_named_scoring_vectors(inputs)
    assert loaded.recorded_layers == [0, 1, 2, 3, 4] and np.allclose(np.linalg.norm(loaded.unit_vectors_by_name()["a"], axis=-1), 1.0)
    write_manifest(["a", "zero"])
    with pytest.raises(ValueError):
        manifest_module.load_named_scoring_vectors(inputs)
    write_manifest(["a", "other_shape"])
    with pytest.raises(ValueError):
        manifest_module.load_named_scoring_vectors(inputs)


def test_dry_run_of_the_builder_writes_nothing(tmp_path, capsys):
    previous, preachiness, new = make_documents_for_assembly(sources=10, extra_corpus=10)
    previous_folder = tmp_path / "previous"
    (previous_folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True)
    write_jsonl(previous_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl", previous)
    (previous_folder / "run_configuration_2026-10-31_11-20.json").write_text(json.dumps({"model_name": "tiny", "model_revision": ""}))
    write_jsonl(tmp_path / "preachiness.jsonl", preachiness)
    write_jsonl(tmp_path / "new.jsonl", new)
    write_vector_set(tmp_path / "set")
    arguments = builder.parse_arguments(["--vector-set-folder", str(tmp_path / "set"), "--new-rewrites-jsonl", str(tmp_path / "new.jsonl"),
                                         "--previous-scoring-run-folder", str(previous_folder), "--preachiness-rewrites-jsonl",
                                         str(tmp_path / "preachiness.jsonl"), "--runs-directory", str(tmp_path / "runs"), "--dry-run"])
    run_folder, configuration, estimate = builder.build_run_folder(arguments)
    assert run_folder is None and not (tmp_path / "runs").exists()
    assert configuration["documents_by_group"] == {"corpus": 20, "existing_rewrites": 20, "new_rewrites": 40}
    assert estimate["documents"] == 80 and estimate["suggested_launcher_limits"]["maximum_running_hours"] >= 1.0
    assert "document tokens a second" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------
# Saved activations (tiny random model, bfloat16 as on the pod)
# ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    try:
        return support.load_real_gemma_tokenizer()
    except Exception as error:  # not cached
        pytest.skip(f"Gemma 4 tokenizer not in the local cache: {error}")


@pytest.fixture(scope="module")
def tiny_bfloat16_model(tmp_path_factory, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM
    directory = tmp_path_factory.mktemp("tiny_model_for_saved_activations")
    support.save_tiny_model_with_real_tokenizer(directory)
    return AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16).eval()


def recorder_and_vectors(model, activation_layers_to_save=()):
    generator = np.random.default_rng(0)
    vectors = {name: generator.normal(size=(support.TINY_LAYER_COUNT + 1, support.TINY_HIDDEN_SIZE)).astype(np.float32)
               for name in ("first_vector", "second_vector")}
    unit = {name: vector / np.linalg.norm(vector, axis=-1, keepdims=True) for name, vector in vectors.items()}
    return LayerProjectionRecorder(model, [1, 2, 3], unit, list(vectors), activation_layers_to_save), vectors


def scored_documents():
    return [{"document_id": f"d{n}", "kind": "corpus", "content": " ".join(f"word{(k * 3 + n) % 17}" for k in range(40 + 15 * n))}
            for n in range(4)]


def score(model, tokenizer, recorder, activation_layer):
    neutral = rendering.load_prompt(defs.ASSISTANT_TURN_NEUTRAL_USER_MESSAGE_FILE_NAME)
    return stages.score_shard_documents(scored_documents(), "assistant_turn", tokenizer, recorder, "cpu", "You are an assistant.", neutral, 5, 4096,
                                        tokenizer.pad_token_id or 0, activation_layer_to_save=activation_layer)


def test_projection_recomputed_locally_from_the_saved_activations_equals_the_on_pod_projection(tiny_bfloat16_model, tokenizer):
    recorder, vectors = recorder_and_vectors(tiny_bfloat16_model)
    recorder.attach()
    try:
        arrays, failed = score(tiny_bfloat16_model, tokenizer, recorder, activation_layer=2)
    finally:
        recorder.detach()
    assert not failed
    saved = arrays[saved_activations.SAVED_ACTIVATIONS_KEY]
    offsets = arrays["per_token_offsets"]
    assert saved.dtype == np.float16 and saved.shape == (offsets[-1], support.TINY_HIDDEN_SIZE)
    assert int(arrays[saved_activations.SAVED_ACTIVATIONS_LAYER_KEY]) == 2
    layer_position = list(arrays["layer_indices"]).index(2)
    for column, name in enumerate(arrays["vector_names"]):
        for row in range(len(offsets) - 1):
            rows = slice(offsets[row], offsets[row + 1])
            recomputed = saved_activations.project_saved_activations(saved[rows], vectors[str(name)][2])
            on_pod = arrays["per_token_projections_float16"][rows, layer_position, column].astype(np.float32)   # stored as float16
            assert np.allclose(recomputed, on_pod, rtol=2e-3, atol=2e-3 * max(1.0, float(np.abs(on_pod).max())))


def test_the_float32_projection_before_it_is_stored_equals_the_local_recomputation_to_float32_precision(tiny_bfloat16_model, tokenizer):
    recorder, vectors = recorder_and_vectors(tiny_bfloat16_model, activation_layers_to_save=[2])
    recorder.attach()
    try:
        tokenized = [rendering.tokenize_document_for_format(tokenizer, d["content"], "assistant_turn", system_prompt="You are an assistant.")
                     for d in scored_documents()]
        sink = {}
        projections = project_documents(recorder, tokenized, "cpu", 4096, tokenizer.pad_token_id or 0, saved_activations_sink=sink)
    finally:
        recorder.detach()
    layer_position = [1, 2, 3].index(2)
    for index, projection in enumerate(projections):
        assert sink[index][2].dtype == np.float16 and sink[index][2].shape[0] == projection.shape[0]
        for column, name in enumerate(("first_vector", "second_vector")):
            recomputed = saved_activations.project_saved_activations(sink[index][2], vectors[name][2])
            assert np.allclose(recomputed, projection[:, layer_position, column], rtol=1e-4, atol=1e-4 * float(np.abs(projection).max()))


def test_saving_activations_changes_nothing_else_and_is_off_by_default(tiny_bfloat16_model, tokenizer):
    recorder, _ = recorder_and_vectors(tiny_bfloat16_model)
    recorder.attach()
    try:
        without, _ = score(tiny_bfloat16_model, tokenizer, recorder, activation_layer=None)
        with_saving, _ = score(tiny_bfloat16_model, tokenizer, recorder, activation_layer=2)
    finally:
        recorder.detach()
    assert saved_activations.SAVED_ACTIVATIONS_KEY not in without
    assert set(with_saving) - set(without) == {saved_activations.SAVED_ACTIVATIONS_KEY, saved_activations.SAVED_ACTIVATIONS_LAYER_KEY}
    for key in without:
        assert np.array_equal(without[key], with_saving[key]), key


def test_saving_activations_at_an_unrecorded_layer_is_refused(tiny_bfloat16_model):
    with pytest.raises(ValueError):
        recorder_and_vectors(tiny_bfloat16_model, activation_layers_to_save=[0])


def test_loading_saved_activations_from_a_run_folder_returns_each_document_in_order(tiny_bfloat16_model, tokenizer, tmp_path):
    recorder, _ = recorder_and_vectors(tiny_bfloat16_model)
    recorder.attach()
    try:
        arrays, _ = score(tiny_bfloat16_model, tokenizer, recorder, activation_layer=2)
    finally:
        recorder.detach()
    folder = tmp_path / "scoring_pod_outputs" / "per_document_projections" / "assistant_turn"
    folder.mkdir(parents=True)
    np.savez(folder / "projections_assistant_turn_shard_00000_2026-11-01_00-00.npz", **arrays)
    loaded, layer = saved_activations.load_saved_activations(tmp_path, "assistant_turn")
    assert layer == 2 and sorted(loaded) == ["d0", "d1", "d2", "d3"]
    offsets = arrays["per_token_offsets"]
    assert np.array_equal(loaded["d2"], arrays[saved_activations.SAVED_ACTIVATIONS_KEY][offsets[2]:offsets[3]])
    with pytest.raises(FileNotFoundError):
        saved_activations.load_saved_activations(tmp_path, "user_message")


def test_stage_runner_defaults_to_the_stages_named_in_the_run_configuration():
    arguments = stages.parse_arguments(["--run-folder", "x"])
    assert arguments.stages is None and arguments.save_full_activations_for_formats is None


# ---------------------------------------------------------------------------------------------------
# The whole chain, once, on the tiny model (the rehearsal script, with fewer draws)
# ---------------------------------------------------------------------------------------------------

def test_rehearsal_runs_end_to_end_and_the_saved_activations_agree_with_the_pod(tokenizer, tmp_path):
    from persona_vector_projection import rehearse_moralizing_delivery_positive_controls_and_scoring_locally as rehearsal
    try:
        rehearsal.find_stand_in_vector_set()
    except FileNotFoundError:
        pytest.skip("no rehearsal moralizing delivery vector set in small_sample_checks")
    folder, summary = rehearsal.run_rehearsal(tmp_path, circular_draws=100, bootstrap_draws=100, say=lambda *_: None)
    assert summary["documents_by_group"] == {"corpus": 20, "existing_rewrites": 20, "new_rewrites": 40}
    assert summary["rewrites"]["records_by_arm"] == {arm: 10 for arm in arms.ARMS}
    assert all(r["within_tolerance"] for r in summary["saved_activations_recomputed_locally"].values())
    assert list(Path(summary["analysis_folder"]).glob("criteria_verdicts_*.json")) and list(folder.glob("rehearsal_summary_*.json"))
