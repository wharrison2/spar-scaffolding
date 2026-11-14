"""Offline tests (stubbed OpenAI client, stubbed screen and rewrite functions; no model calls) for the bugs that
would invalidate the validation experiments: batch and position bookkeeping, plants mapped to the right base hand,
replacement values chosen only from current values, labels joined to the right prompt_id, metrics on small
hand-built examples, the audit join, resume skipping completed work, and key handling.

Run (the pipeline's venv has pytest and openai):
    animal-welfare-data-pipeline/.venv/bin/python -m pytest scripts/spec_contradiction_screen/test_spec_contradiction_screen_validation.py
This file also runs the pipeline-side tests in a separate process (see conftest.py).
"""

import csv
import json
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

import analyze_screen_runs as analysis
import build_spot_check_sheet as spot_check
import check_description_rewrites_with_gpt as rewrite_checks
import description_rewrite_check_prompt_rendering as check_rendering
import gold_label_and_plant_prompt_rendering as rendering
import gold_label_dealt_hands as gold_stage
import plant_assignment_and_gold_label_loading as loading
import propose_planted_replacement_values as proposal_stage
import report_description_rewrite_validation as rewrite_report
import run_description_rewrite_on_fresh_descriptions as rewrite_stage
import run_screen_conditions as screen_stage
import screen_pool_and_batch_design as design
import spec_screen_common as common

Judgment = namedtuple("Judgment", "number reasoning remove reason")
RewriteResult = namedtuple("RewriteResult", "reasoning rewrite_needed rewritten_description changes dealt_value_conflict conflict_reason")
HERE = Path(__file__).resolve().parent


def make_hands(count):
    return [{"prompt_id": f"matrix_{i:06d}",
             "variables": {"sentient_category": f"being_{i % 5}", "decision_scale": f"scale_{i % 3}", "document_type": f"genre_{i % 4}"},
             "variables_as_dealt_in_pinned_run": {"sentient_category": f"being_{i % 5}"},
             "rendered_variable_lines": f"HAND<{i}> lines for hand number {i}"} for i in range(count)]


def make_value_options():
    return {"sentient_category": [{"value_id": f"b{i}", "text": f"being_{i}", "weight": 0.2} for i in range(5)] + [{"value_id": "b9", "text": "zero_weight_being", "weight": 0.0}],
            "decision_scale": [{"value_id": f"s{i}", "text": f"scale_{i}", "weight": 0.3} for i in range(3)],
            "document_type": [{"value_id": f"g{i}", "text": f"genre_{i}", "weight": 0.25} for i in range(4)]}


class StubOpenAIClient:
    def __init__(self, answer_function):
        self.answer_function = answer_function
        self.calls = 0
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **arguments):
        self.calls += 1
        payload = self.answer_function(arguments["input"], self.calls)
        usage = SimpleNamespace(input_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=10),
                                output_tokens=50, output_tokens_details=SimpleNamespace(reasoning_tokens=20))
        return SimpleNamespace(output_text=json.dumps(payload), status="completed", usage=usage)


def hand_number_in(text):
    return int(text.split("HAND<")[1].split(">")[0])


def judgment_answer(text, call_number):
    implausible = hand_number_in(text) % 4 == 0
    return {"reasoning": "checked", "extremely_implausible": implausible,
            "problem_type": "being_not_present_in_locale_or_domain" if implausible else "none",
            "reason": "no such being there" if implausible else ""}


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pool_items_for(hands):
    return [{"item_id": h["prompt_id"], "rendered_variable_lines": h["rendered_variable_lines"]} for h in hands]


def run_gold(hands, run_folder, client, answer_function=None):
    return gold_stage.run_hand_judgments(pool_items_for(hands), "gold_labels", gold_stage.GOLD_LABELS_BASE_NAME, run_folder,
                                         client=client, reasoning_effort="low", workers=4, id_field_name="prompt_id")


def write_pool(run_folder, hands, options=None):
    folder = run_folder / common.STAGE_SUBFOLDER_NAMES["pinned_hand_pool"]
    folder.mkdir(parents=True, exist_ok=True)
    common.write_jsonl_file(folder / "pinned_hand_pool_2026-01-01_00-00.jsonl", hands)
    (folder / "current_variable_value_options_2026-01-01_00-00.json").write_text(json.dumps(options or make_value_options()))


# ---------------------------------------------------------------- gold labels on hands

def test_gold_labels_are_joined_to_the_right_prompt_id_and_resume_skips_done_work(tmp_path):
    hands = make_hands(24)
    client = StubOpenAIClient(judgment_answer)
    run_gold(hands, tmp_path, client)
    labels = loading.load_gold_labels(tmp_path)
    assert len(labels) == 24 and client.calls == 24
    for hand in hands:
        assert labels[hand["prompt_id"]]["extremely_implausible"] == (hand_number_in(hand["rendered_variable_lines"]) % 4 == 0)
    run_gold(hands, tmp_path, client)
    assert client.calls == 24


def test_inconsistent_judgment_is_retried_then_not_recorded_and_retried_on_resume(tmp_path):
    hands = make_hands(4)

    def answer(text, call_number):
        if hand_number_in(text) == 1:  # extremely implausible with problem_type none: inconsistent
            return {"reasoning": "r", "extremely_implausible": True, "problem_type": "none", "reason": "x"}
        return judgment_answer(text, call_number)

    first = run_gold(hands, tmp_path, StubOpenAIClient(answer))
    assert first["failed"] == ["matrix_000001"]
    assert "matrix_000001" not in loading.load_gold_labels(tmp_path)
    second_client = StubOpenAIClient(judgment_answer)
    run_gold(hands, tmp_path, second_client)
    assert second_client.calls == 1 and "matrix_000001" in loading.load_gold_labels(tmp_path)


def test_hand_judgment_prompt_shows_the_hand_lines_and_nothing_unfilled():
    system_text, user_text = rendering.render_hand_judgment_prompt("Type: a news article.\nTone: skeptical.")
    assert "Type: a news article." in user_text and "{" not in user_text and "{" not in system_text
    assert "extremely implausible" in user_text and "how the world actually is" in system_text


def test_model_facing_prompt_files_exist_and_never_describe_an_allowed_list():
    for path in [rendering.HAND_JUDGMENT_PROMPT_PATH, rendering.REPLACEMENT_PROMPT_PATH, rendering.PLANT_TYPE_INSTRUCTIONS_PATH,
                 check_rendering.PROBLEM_GONE_PROMPT_PATH, check_rendering.DEALT_VALUES_KEPT_PROMPT_PATH]:
        assert path.suffix == ".txt" and path.is_file()
        assert "allowed" not in path.read_text().lower()


# ---------------------------------------------------------------- plants: assignments, replacement choices

def test_type_plant_assignments_use_distinct_bases_balance_types_and_fix_the_changed_variable():
    assignments = loading.build_type_plant_assignments(make_hands(100), 5, seed=1)
    assert len(assignments) == 15 and len({a["base_prompt_id"] for a in assignments}) == 15
    for plant_type in common.TYPE_PLANT_TYPE_NAMES:
        of_type = [a for a in assignments if a["planted_type"] == plant_type]
        assert len(of_type) == 5
        assert {a["target_variable"] for a in of_type} == {common.TYPE_PLANT_TARGET_VARIABLES[plant_type]}
    assert assignments == loading.build_type_plant_assignments(make_hands(100), 5, seed=1)
    with pytest.raises(ValueError):
        loading.build_type_plant_assignments(make_hands(10), 5, seed=1)


def test_option_list_excludes_the_current_and_zero_weight_values_and_is_seeded():
    options = make_value_options()["sentient_category"]
    shown = loading.build_option_list(options, "being_2", seed=3, plant_id="plant_0001")
    texts = [o["text"] for o in shown]
    assert "being_2" not in texts and "zero_weight_being" not in texts and sorted(texts) == ["being_0", "being_1", "being_3", "being_4"]
    assert shown == loading.build_option_list(options, "being_2", seed=3, plant_id="plant_0001")


def test_replacement_choice_is_mapped_from_the_option_number_to_a_current_value(tmp_path):
    hands = make_hands(40)
    options = make_value_options()
    option_texts_by_variable = {v: {o["text"] for o in o_list} for v, o_list in options.items()}

    def choose_second_option(text, call_number):
        return {"reasoning": "r", "chosen_option_number": 2}

    summary = proposal_stage.run_proposals(hands, options, tmp_path, client=StubOpenAIClient(choose_second_option),
                                           plants_per_type=3, seed=5, reasoning_effort="low", workers=3)
    assert summary["failed"] == []
    stage = tmp_path / common.STAGE_SUBFOLDER_NAMES["planted_hands"]
    assignments = {a["plant_id"]: a for a in read_records(next(stage.glob("planted_hand_assignments_*.jsonl")))}
    choices = read_records(next(stage.glob("planted_replacement_choices_*.jsonl")))
    assert len(choices) == 9
    for choice in choices:
        assignment = assignments[choice["plant_id"]]
        variable = assignment["target_variable"]
        assert choice["replacement_value_text"] == assignment["option_texts_in_shown_order"][1]
        assert choice["replacement_value_text"] in option_texts_by_variable[variable]
        assert choice["replacement_value_text"] != assignment["current_value_text"]


def test_out_of_range_option_number_is_rejected_and_never_recorded(tmp_path):
    summary = proposal_stage.run_proposals(make_hands(40), make_value_options(), tmp_path,
                                           client=StubOpenAIClient(lambda text, n: {"reasoning": "r", "chosen_option_number": 99}),
                                           plants_per_type=1, seed=5, reasoning_effort="low", workers=2)
    assert len(summary["failed"]) == 3
    assert not list((tmp_path / common.STAGE_SUBFOLDER_NAMES["planted_hands"]).glob("planted_replacement_choices_*.jsonl")) or \
        read_records(next((tmp_path / common.STAGE_SUBFOLDER_NAMES["planted_hands"]).glob("planted_replacement_choices_*.jsonl"))) == []


def test_replacement_prompt_numbers_the_options_and_names_the_current_value():
    system_text, user_text = rendering.render_replacement_prompt(
        "LINES", "sentient_category", "being_1", "being_not_present_in_locale_or_domain", ["alpha", "beta", "gamma"])
    assert "1. alpha" in user_text and "3. gamma" in user_text and '"being_1"' in user_text and "{" not in user_text


def test_confirmed_plants_discard_and_count_the_unconfirmed_and_require_every_judgment(tmp_path):
    stage = tmp_path / common.STAGE_SUBFOLDER_NAMES["planted_hands"]
    stage.mkdir(parents=True)
    plants = [{"plant_id": f"plant_{i:04d}", "base_prompt_id": f"matrix_{i:06d}", "planted_type": common.TYPE_PLANT_TYPE_NAMES[i % 3],
               "rendered_variable_lines": "x", "variables": {}} for i in range(6)]
    common.write_jsonl_file(stage / "planted_hands_before_confirmation_2026-01-01_00-00.jsonl", plants)
    judged = [{"plant_id": p["plant_id"], "extremely_implausible": i % 3 != 0, "problem_type": "x", "reason": "r"} for i, p in enumerate(plants[:5])]
    common.write_jsonl_file(stage / "planted_hand_confirmations_by_gpt_6_1_sol_2026-01-01_00-00.jsonl", judged)
    with pytest.raises(ValueError):
        loading.load_confirmed_plants(tmp_path)
    judged.append({"plant_id": "plant_0005", "extremely_implausible": True, "problem_type": "x", "reason": "r"})
    common.write_jsonl_file(stage / "planted_hand_confirmations_by_gpt_6_1_sol_2026-01-01_00-00.jsonl", judged)
    confirmed, summary = loading.load_confirmed_plants(tmp_path)
    assert {p["plant_id"] for p in confirmed} == {"plant_0001", "plant_0002", "plant_0004", "plant_0005"}
    assert sum(r["discarded"] for r in summary.values()) == 2 and sum(r["made"] for r in summary.values()) == 6


# ---------------------------------------------------------------- batch design

def make_pool(real_count=95, plant_count=10, gold_implausible_every=7):
    hands = make_hands(real_count)
    gold = {h["prompt_id"]: {"extremely_implausible": i % gold_implausible_every == 0,
                             "problem_type": "x" if i % gold_implausible_every == 0 else "none"} for i, h in enumerate(hands)}
    plants = [{"plant_id": f"plant_{i:04d}", "base_prompt_id": hands[i * 3 + 1]["prompt_id"], "planted_type": common.TYPE_PLANT_TYPE_NAMES[i % 3],
               "rendered_variable_lines": f"PLANT<{i}>", "variables": hands[i * 3 + 1]["variables"]} for i in range(plant_count)]
    return design.build_pool_items(hands, gold, plants)


def test_main_batches_cover_every_item_once_with_even_plants_and_correct_positions():
    pool = make_pool()
    batches = design.build_main_batches(pool, 10, seed=3)
    ids = [m["item_id"] for b in batches for m in b["members"]]
    assert sorted(ids) == sorted(i["item_id"] for i in pool)
    assert all([m["position"] for m in b["members"]] == list(range(1, len(b["members"]) + 1)) for b in batches)
    plants_per_batch = [sum(m["item_id"].startswith("plant_") for m in b["members"]) for b in batches]
    assert sum(plants_per_batch) == 10 and max(plants_per_batch) - min(plants_per_batch) <= 1


def test_a_plant_is_never_in_the_same_batch_as_its_base_hand_across_seeds():
    pool = make_pool()
    base_of = {i["item_id"]: i["base_prompt_id"] for i in pool if i["is_plant"]}
    for seed in range(40):
        for batch in design.build_main_batches(pool, 10, seed=seed):
            members = {m["item_id"] for m in batch["members"]}
            assert not any(base_of[p] in members for p in members if p in base_of)


def test_clean_batches_hold_only_gold_plausible_real_hands_without_repeats():
    pool = make_pool()
    clean = design.build_clean_batches(pool, 5, 10, seed=1)
    by_id = {i["item_id"]: i for i in pool}
    ids = [m["item_id"] for b in clean for m in b["members"]]
    assert len(ids) == 50 and len(set(ids)) == 50
    assert all(not by_id[i]["is_plant"] and by_id[i]["gold_extremely_implausible"] is False for i in ids)
    with pytest.raises(ValueError):
        design.build_clean_batches(make_pool(real_count=30), 5, 10, seed=1)


def test_work_units_per_condition_have_no_condition_b():
    pool = make_pool()
    pool_by_id = {i["item_id"]: i for i in pool}
    main, clean = design.build_main_batches(pool, 10, 1), design.build_clean_batches(pool, 5, 10, 1)
    units = design.build_work_units(pool_by_id, main, clean, ["A", "A_repeat", "C"])
    count = lambda c: sum(u["condition"] == c for u in units)  # noqa: E731
    assert count("A") == len(main) + 5 and count("A_repeat") == len(main) and count("C") == len(pool)
    assert all(len(u["members"]) == 1 for u in units if u["condition"] == "C")
    assert len({u["unit_key"] for u in units}) == len(units)
    with pytest.raises(ValueError):
        design.build_work_units(pool_by_id, main, clean, ["B"])


def test_design_writes_batches_before_any_call_and_reuses_them(tmp_path):
    hands = make_hands(80)
    write_pool(tmp_path, hands)
    gold = [{"prompt_id": h["prompt_id"], "extremely_implausible": i % 9 == 0, "problem_type": "x" if i % 9 == 0 else "none", "reason": "r" if i % 9 == 0 else ""}
            for i, h in enumerate(hands)]
    common.write_jsonl_file(common.stage_folder(tmp_path, "gold_labels") / "gold_labels_by_gpt_6_1_sol_2026-01-01_00-00.jsonl", gold)
    stage = common.stage_folder(tmp_path, "planted_hands")
    plants = [{"plant_id": f"plant_{i:04d}", "base_prompt_id": hands[i * 2 + 1]["prompt_id"], "planted_type": common.TYPE_PLANT_TYPE_NAMES[i % 3],
               "rendered_variable_lines": f"PLANT<{i}>", "variables": hands[i * 2 + 1]["variables"]} for i in range(6)]
    common.write_jsonl_file(stage / "planted_hands_before_confirmation_2026-01-01_00-00.jsonl", plants)
    common.write_jsonl_file(stage / "planted_hand_confirmations_by_gpt_6_1_sol_2026-01-01_00-00.jsonl",
                            [{"plant_id": p["plant_id"], "extremely_implausible": True, "problem_type": "x", "reason": "r"} for p in plants])
    pool_path, batches_path = screen_stage.design_pool_and_batches(tmp_path, seed=4)
    first = batches_path.read_text()
    assert screen_stage.design_pool_and_batches(tmp_path, seed=99)[1].read_text() == first
    pool = {r["item_id"]: r for r in read_records(pool_path)}
    assert sum(r["is_plant"] for r in pool.values()) == 6
    assert any(r["gold_extremely_implausible"] for r in pool.values())


# ---------------------------------------------------------------- running the screen

def is_planted_text(lines):
    return lines.startswith("PLANT<")


def stub_screen_function(calls):
    """Removes plants only; judgments in position order; records what it was shown."""
    def screen(rendered_hands, item_id):
        calls.append({"item_id": item_id, "rendered_hands": rendered_hands})
        return [Judgment(n, "r", is_planted_text(lines), "bad" if is_planted_text(lines) else "") for n, lines in enumerate(rendered_hands, 1)]
    return screen


def test_screen_run_records_positions_correctly_shows_rendered_lines_and_resumes_with_zero_calls(tmp_path):
    pool = make_pool(real_count=60, plant_count=8)
    pool_by_id = {i["item_id"]: i for i in pool}
    main = design.build_main_batches(pool, 10, 2)
    units = design.build_work_units(pool_by_id, main, [], ["A", "C"])
    results, calls = tmp_path / "results.jsonl", []
    summary = screen_stage.run_units(units, pool_by_id, results, tmp_path / "errors.jsonl", tmp_path / "progress.log",
                                     stub_screen_function(calls), 3)
    assert summary["failed"] == [] and len(calls) == len(units)
    batch_by_id = {b["batch_id"]: b for b in main}
    for record in read_records(results):
        if record["condition"] == "A":
            expected = batch_by_id[record["batch_id"]]["members"]
            assert [(j["position"], j["item_id"]) for j in record["judgments"]] == [(m["position"], m["item_id"]) for m in expected]
        assert all(j["remove"] == pool_by_id[j["item_id"]]["is_plant"] for j in record["judgments"])
    for call in calls:
        assert all(text == pool_by_id[item_id]["rendered_variable_lines"] for text, item_id in
                   zip(call["rendered_hands"], [m["item_id"] for u in units if u["unit_key"] == call["item_id"] for m in u["members"]]))
    screen_stage.run_units(units, pool_by_id, results, tmp_path / "errors.jsonl", tmp_path / "progress.log", stub_screen_function(calls), 3)
    assert len(calls) == len(units)


def test_failed_unit_is_not_recorded_and_is_retried_on_resume(tmp_path):
    pool = make_pool(real_count=20, plant_count=2)
    pool_by_id = {i["item_id"]: i for i in pool}
    units = design.build_work_units(pool_by_id, design.build_main_batches(pool, 10, 1), [], ["A"])
    results, calls = tmp_path / "results.jsonl", []
    working = stub_screen_function(calls)

    def flaky(rendered_hands, item_id):
        if item_id == units[0]["unit_key"]:
            raise RuntimeError("boom")
        return working(rendered_hands, item_id)

    summary = screen_stage.run_units(units, pool_by_id, results, tmp_path / "errors.jsonl", tmp_path / "progress.log", flaky, 1)
    assert summary["failed"] == [units[0]["unit_key"]] and len(read_records(results)) == len(units) - 1
    calls.clear()
    screen_stage.run_units(units, pool_by_id, results, tmp_path / "errors.jsonl", tmp_path / "progress.log", working, 1)
    assert [c["item_id"] for c in calls] == [units[0]["unit_key"]]


def test_run_stops_after_repeated_failures_with_no_success(tmp_path):
    pool = make_pool(real_count=190, plant_count=0)
    pool_by_id = {i["item_id"]: i for i in pool}
    units = design.build_work_units(pool_by_id, design.build_main_batches(pool, 10, 1), [], ["A"])
    attempts = []

    def always_fails(rendered_hands, item_id):
        attempts.append(item_id)
        raise RuntimeError("bad key")

    summary = screen_stage.run_units(units, pool_by_id, tmp_path / "r.jsonl", tmp_path / "e.jsonl", tmp_path / "p.log", always_fails, 1)
    assert summary["aborted"] and len(attempts) == screen_stage.ABORT_AFTER_THIS_MANY_FAILURES_WITH_NO_SUCCESS


def test_unit_record_rejects_wrong_judgment_count():
    pool = make_pool(real_count=20, plant_count=0)
    pool_by_id = {i["item_id"]: i for i in pool}
    unit = design.build_work_units(pool_by_id, design.build_main_batches(pool, 10, 1), [], ["A"])[0]
    with pytest.raises(ValueError):
        screen_stage.make_unit_record(unit, pool_by_id, [Judgment(1, "r", False, "")], 0.1)


# ---------------------------------------------------------------- analysis

def make_row(condition, batch_kind, item_id, remove, *, is_plant=False, planted_type=None, gold=None, position=1,
             batch_id="b1", variables=None, audit_score=None):
    return {"condition": condition, "batch_id": batch_id, "batch_kind": batch_kind, "item_id": item_id, "position": position,
            "remove": remove, "reason": "r" if remove else "", "is_plant": is_plant, "planted_type": planted_type,
            "gold_implausible": gold, "audit_score": audit_score, "variables": variables or {"domain": "d"}}


def hand_built_rows():
    rows = []
    for condition in ("A", "A_repeat"):
        rows += [make_row(condition, "main", "p1", True, is_plant=True, planted_type="t1"),
                 make_row(condition, "main", "p2", condition == "A", is_plant=True, planted_type="t1", position=2),
                 make_row(condition, "main", "p3", False, is_plant=True, planted_type="t2", position=3),
                 make_row(condition, "main", "g1", True, gold=True, position=4),
                 make_row(condition, "main", "g2", False, gold=True, position=5),
                 make_row(condition, "main", "r1", False, gold=False, position=6, audit_score=1),
                 make_row(condition, "main", "r2", True, gold=False, position=7, audit_score=5),
                 make_row(condition, "main", "r3", False, gold=False, position=8, audit_score=4),
                 make_row(condition, "main", "r4", False, gold=False, position=9, audit_score=None),
                 make_row(condition, "clean", "c1", False, gold=False, batch_id="clean1"),
                 make_row(condition, "clean", "c2", True, gold=False, batch_id="clean1", position=2)]
    return rows


def test_metrics_on_a_hand_built_example():
    metrics = analysis.compute_metrics(hand_built_rows())
    recall = {(r["condition"], r["group"], r["value"]): r for r in metrics["plant_recall"]}
    assert recall[("A", "overall", "all")]["plants_removed"] == 2 and recall[("A", "overall", "all")]["plants_judged"] == 3
    assert recall[("A_repeat", "overall", "all")]["plants_removed"] == 1
    assert recall[("A", "type", "t1")]["recall"] == 1.0 and recall[("A", "type", "t2")]["recall"] == 0.0
    gold = {r["condition"]: r for r in metrics["agreement_with_gold_labels"]}["A"]
    assert gold["recall_on_gold_implausible"] == 0.5 and abs(gold["false_removal_rate_on_gold_plausible"] - 1 / 4) < 1e-9
    clean = {r["condition"]: r for r in metrics["clean_batch_removals"]}["A"]
    assert clean["removals"] == 1 and clean["batches_with_at_least_one_removal"] == 1
    stability = metrics["stability_a_versus_a_repeat"][0]
    assert stability["removed_only_in_a"] == 1 and stability["removed_in_both"] == 3 and stability["removed_only_in_a_repeat"] == 0
    groups = {(r["condition"], r["audit_score_group"]): r for r in metrics["removal_rate_by_audited_document_score"]}
    assert groups[("A", "score_1_to_2")]["hands_judged"] == 1 and groups[("A", "score_4_to_5")]["hands_judged"] == 2
    assert groups[("A", "score_4_to_5")]["removed"] == 1 and groups[("A", "no_audit_score")]["hands_judged"] == 3  # g1, g2 and r4
    assert {r["position"] for r in metrics["removal_rate_by_batch_position"]} == set(range(1, 10))


def test_value_removed_at_over_twice_corpus_rate_is_flagged():
    rows = [make_row("A", "main", f"h{i}", i < 2, gold=False, variables={"culture": "special" if i < 2 else "ordinary"}) for i in range(30)]
    flagged = [r for r in analysis.compute_metrics(rows)["removal_rate_per_variable_value"] if r["flagged"]]
    assert [r["value"] for r in flagged] == ["special"]


def test_audit_join_counts_joined_hands_and_refuses_a_join_whose_variables_disagree():
    pool_by_id = {
        "matrix_000001": {"item_id": "matrix_000001", "is_plant": False, "variables_as_dealt_in_pinned_run": {"domain": "d1"}},
        "matrix_000002": {"item_id": "matrix_000002", "is_plant": False, "variables_as_dealt_in_pinned_run": {"domain": "d2"}},
        "matrix_000003": {"item_id": "matrix_000003", "is_plant": False, "variables_as_dealt_in_pinned_run": {"domain": "d3"}},
        "plant_0001": {"item_id": "plant_0001", "is_plant": True}}
    audit = {"matrix_000001": {"doc_id": "matrix_000001", "domain": "d1", "overall_plausibility": 2},
             "matrix_000002": {"doc_id": "matrix_000002", "domain": "OTHER", "overall_plausibility": 5}}
    counts = analysis.attach_audit_scores(pool_by_id, audit)
    assert counts == {"real_hands_in_pool": 3, "audit_records": 2, "joined_with_a_score": 1, "audit_record_found_but_variables_disagree": 1}
    assert pool_by_id["matrix_000001"]["audit_score"] == 2 and pool_by_id["matrix_000002"]["audit_score"] is None
    assert pool_by_id["plant_0001"]["audit_score"] is None


def test_outputs_carry_the_backend_note_and_the_join_counts(tmp_path):
    note = common.build_backend_note("api", "claude-sonnet-5", "abc123")
    counts = {"real_hands_in_pool": 3, "audit_records": 2, "joined_with_a_score": 1, "audit_record_found_but_variables_disagree": 0}
    paths = analysis.write_outputs(analysis.compute_metrics(hand_built_rows()), tmp_path, "2026-01-01_00-00", note, counts)
    for path in paths:
        text = path.read_text()
        assert note in text and "CLI" not in text and "provisional" not in text.lower()
    assert "1 of 3 real hands" in paths[-1].read_text()


# ---------------------------------------------------------------- spot-check sheet

def test_spot_check_sheet_hides_labels_keeps_key_separate_and_reports_shortfall(tmp_path):
    hands = make_hands(40)
    gold = {h["prompt_id"]: {"extremely_implausible": i < 6, "problem_type": "being_not_present_in_locale_or_domain" if i < 6 else "none",
                             "reason": "r" if i < 6 else ""} for i, h in enumerate(hands)}
    items, note = spot_check.choose_spot_check_items(hands, gold, 1)
    assert len(items) == 16 and "Fewer than 10" in note
    paths = spot_check.write_spot_check_files(items, note, tmp_path, "stamp")
    sheet = paths["sheet"].read_text()
    assert "being_not_present" not in sheet.split("## Hand 1")[1] and "matrix_" not in sheet and "HAND<" in sheet
    key = list(csv.DictReader(open(paths["key"])))
    assert [row["prompt_id"] for row in key] == [item["prompt_id"] for item in items]
    answers = list(csv.DictReader(open(paths["answers"])))
    for row, key_row in zip(answers, key):
        row["william_judgment"] = "implausible" if key_row["gpt_extremely_implausible"] == "True" else "plausible"
    answers[0]["william_judgment"] = "plausible" if key[0]["gpt_extremely_implausible"] == "True" else "implausible"
    with open(paths["answers"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(answers[0]))
        writer.writeheader()
        writer.writerows(answers)
    result = spot_check.score_answers(paths["answers"], paths["key"])
    assert result["agreed"] == 15 and result["disagreeing_items"] == [1]


def test_spot_check_takes_ten_and_ten_when_available():
    hands = make_hands(80)
    gold = {h["prompt_id"]: {"extremely_implausible": i < 25, "problem_type": "x", "reason": "r"} for i, h in enumerate(hands)}
    items, note = spot_check.choose_spot_check_items(hands, gold, 2)
    assert sum(i["gold"]["extremely_implausible"] for i in items) == 10 and len(items) == 20 and "Fewer" not in note


# ---------------------------------------------------------------- description rewrite validation

def make_rewrite_inputs():
    inputs = {}
    for i in range(10):
        labelled = i < 4
        inputs[f"matrix_{i:06d}"] = {
            "variables": {"domain": f"d{i}", "tone": "t"}, "description": f"Description {i}.",
            "gold": {"incoherent": labelled, "problem_type": "internal_self_contradiction" if labelled else "none", "reason": "bad" if labelled else ""},
            "rewrite": {"rewrite_needed": i in (0, 1, 5, 6, 7), "dealt_value_conflict": i == 2, "rewritten_description": f"Rewritten {i}.",
                        "changes": [{"check": "plausibility_organisation", "reason": "r"}] if i in (0, 1, 5, 6, 7) else []}}
    return inputs


def test_check_tasks_cover_only_labelled_rewritten_for_problem_gone_and_every_rewrite_for_values():
    inputs = make_rewrite_inputs()
    gone_keys = {t["task_key"] for t in rewrite_checks.build_problem_gone_tasks(inputs)}
    values_keys = {t["task_key"] for t in rewrite_checks.build_dealt_values_kept_tasks(inputs)}
    assert gone_keys == {"matrix_000000", "matrix_000001"}
    assert values_keys == {f"matrix_{i:06d}" for i in (0, 1, 5, 6, 7)}
    task = next(t for t in rewrite_checks.build_problem_gone_tasks(inputs) if t["task_key"] == "matrix_000000")
    assert "Description 0." in task["input_text"] and "Rewritten 0." in task["input_text"] and "bad" in task["input_text"]
    values_task = next(t for t in rewrite_checks.build_dealt_values_kept_tasks(inputs) if t["task_key"] == "matrix_000005")
    assert "domain: d5" in values_task["input_text"] and "Rewritten 5." in values_task["input_text"]


def test_dealt_values_kept_validation_and_schema_only_name_known_variables():
    schema = check_rendering.build_dealt_values_kept_schema(["domain", "tone"])
    assert schema["properties"]["missing_variables"]["items"]["enum"] == ["domain", "tone"]
    check_rendering.validate_dealt_values_kept({"all_dealt_values_present": True, "missing_variables": [], "reasoning": "r"})
    for bad in ({"all_dealt_values_present": True, "missing_variables": ["tone"]}, {"all_dealt_values_present": False, "missing_variables": []}):
        with pytest.raises(ValueError):
            check_rendering.validate_dealt_values_kept(dict(bad, reasoning="r"))


def test_rewrite_measures_on_a_hand_built_example():
    inputs = make_rewrite_inputs()
    problem_gone = {"matrix_000000": {"problem_gone": True}, "matrix_000001": {"problem_gone": False}}
    values_kept = {f"matrix_{i:06d}": {"all_dealt_values_present": i != 6} for i in (0, 1, 5, 6, 7)}
    m = rewrite_report.compute_measures(inputs, problem_gone, values_kept)
    assert m["labelled_problematic"] == 4 and m["labelled_rewritten"] == 2 and m["labelled_rewritten_share"] == 0.5
    assert m["labelled_dropped_for_dealt_value_conflict"] == 1 and m["labelled_problem_gone"] == 1 and m["labelled_problem_gone_share"] == 0.25
    assert m["unlabelled_rewritten"] == 3 and m["unlabelled_rewritten_share"] == 0.5
    assert m["rewrites_total"] == 5 and m["rewrites_keeping_every_dealt_value"] == 4
    assert m["by_labelled_problem_type"]["internal_self_contradiction"] == {"labelled": 4, "rewritten": 2, "problem_gone": 1}


def test_original_baseline_tasks_use_the_same_prompt_on_the_original_of_every_rewritten_plan():
    inputs = make_rewrite_inputs()
    original_tasks = {t["task_key"]: t for t in rewrite_checks.build_dealt_values_in_original_tasks(inputs)}
    rewrite_tasks = {t["task_key"]: t for t in rewrite_checks.build_dealt_values_kept_tasks(inputs)}
    assert set(original_tasks) == set(rewrite_tasks)
    for pid, task in original_tasks.items():
        assert task["instructions_text"] == rewrite_tasks[pid]["instructions_text"]
        assert f"Description {int(pid[-6:])}." in task["input_text"] and "Rewritten" not in task["input_text"]
        assert task["input_text"].replace(inputs[pid]["description"], "X") == rewrite_tasks[pid]["input_text"].replace(inputs[pid]["rewrite"]["rewritten_description"], "X")


def test_baseline_comparison_separates_newly_missing_from_already_missing_and_restored():
    inputs = {f"matrix_{i:06d}": {} for i in range(4)}
    original = {"matrix_000000": {"missing_variables": ["tone"]}, "matrix_000001": {"missing_variables": []},
                "matrix_000002": {"missing_variables": ["tone", "domain"]}}
    rewrite = {"matrix_000000": {"missing_variables": ["tone"]}, "matrix_000001": {"missing_variables": ["domain"]},
               "matrix_000002": {"missing_variables": ["tone"]}, "matrix_000003": {"missing_variables": ["tone"]}}
    c = rewrite_report.compare_dealt_values_with_baseline(inputs, original, rewrite)
    assert c["plans_compared"] == 3  # matrix_000003 has no baseline check
    assert c["per_plan"]["matrix_000000"] == {"missing_in_original": ["tone"], "missing_in_rewrite": ["tone"], "newly_missing": [], "restored": []}
    assert c["per_plan"]["matrix_000001"]["newly_missing"] == ["domain"] and c["per_plan"]["matrix_000002"]["restored"] == ["domain"]
    assert c["totals"] == {"missing_in_original": 3, "missing_in_rewrite": 3, "newly_missing": 1, "restored": 1}
    assert c["per_variable"]["tone"] == {"missing_in_original": 2, "missing_in_rewrite": 2, "newly_missing": 0, "restored": 0}
    assert c["plans_with_a_newly_missing_value"] == 1
    assert "newly missing 1, restored 1" in rewrite_report.build_baseline_summary_text(c)


def test_before_after_pairs_are_rewritten_only_seeded_and_ten_when_available(tmp_path):
    inputs = {f"matrix_{i:06d}": {"variables": {}, "description": f"D{i}", "gold": {"incoherent": i < 8, "problem_type": "p", "reason": "r"},
                                  "rewrite": {"rewrite_needed": True, "rewritten_description": f"R{i}", "dealt_value_conflict": False,
                                              "changes": [{"check": "organisation_naming", "reason": "x"}]}} for i in range(30)}
    pairs = rewrite_report.choose_pairs(inputs, 1)
    assert len(pairs) == 10 and len(set(pairs)) == 10 and pairs == rewrite_report.choose_pairs(inputs, 1)
    assert sum(inputs[p]["gold"]["incoherent"] for p in pairs) == 5
    rewrite_report.write_before_after_sheet(inputs, pairs, tmp_path / "sheet.md")
    text = (tmp_path / "sheet.md").read_text()
    assert text.count("### Before") == 10 and text.count("### After") == 10


def test_rewrite_run_records_results_and_resumes_with_zero_calls(tmp_path):
    records = [{"prompt_id": f"matrix_{i:06d}", "variables": {}, "description": f"Description {i}.", "rendered_variable_lines": f"LINES {i}"} for i in range(6)]
    calls = []

    def rewrite_function(variable_lines, description, item_id):
        calls.append(item_id)
        if item_id == "matrix_000003":
            raise RuntimeError("boom")
        needed = int(item_id[-1]) % 2 == 0
        return RewriteResult("r", needed, f"New {item_id}" if needed else "", ({"check": "organisation_naming", "reason": "x"},) if needed else (), False, "")

    results = tmp_path / "results.jsonl"
    summary = rewrite_stage.run_rewrites(records, results, tmp_path / "e.jsonl", tmp_path / "p.log", rewrite_function, 2)
    assert summary["failed"] == ["matrix_000003"] and len(read_records(results)) == 5
    stored = {r["prompt_id"]: r for r in read_records(results)}
    assert stored["matrix_000002"]["original_description"] == "Description 2." and stored["matrix_000002"]["rewritten_description"] == "New matrix_000002"
    calls.clear()
    rewrite_stage.run_rewrites(records, results, tmp_path / "e.jsonl", tmp_path / "p.log", lambda *a: rewrite_function(*a) if a[2] == "matrix_000003" and False else RewriteResult("r", False, "", (), False, ""), 2)
    assert len(read_records(results)) == 6


# ---------------------------------------------------------------- key handling and imports

def test_anthropic_key_is_read_only_from_the_project_env_file_and_never_printed(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=openai-secret\nANTHROPIC_API_KEY='anthropic-secret'\n")
    monkeypatch.setattr(common, "ENV_FILE_PATH", env_file)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-from-operating-system")
    with pytest.raises(SystemExit):
        common.activate_project_anthropic_api_key()
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert common.activate_project_anthropic_api_key() == env_file
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "anthropic-secret"
    output = capsys.readouterr().out
    assert str(env_file) in output and "anthropic-secret" not in output
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    env_file.write_text("OPENAI_API_KEY=x\n")
    with pytest.raises(SystemExit):
        common.activate_project_anthropic_api_key()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_openai_key_is_read_only_from_the_project_env_file_and_never_printed(tmp_path, monkeypatch, capsys):
    import os
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=anthropic-secret\nOPENAI_API_KEY='openai-secret'\n")
    monkeypatch.setattr(common, "ENV_FILE_PATH", env_file)
    monkeypatch.setenv("OPENAI_API_KEY", "key-from-operating-system")
    with pytest.raises(SystemExit):
        common.activate_project_openai_api_key()
    monkeypatch.delenv("OPENAI_API_KEY")
    assert common.activate_project_openai_api_key() == env_file
    assert os.environ["OPENAI_API_KEY"] == "openai-secret"
    output = capsys.readouterr().out
    assert str(env_file) in output and "openai-secret" not in output
    monkeypatch.delenv("OPENAI_API_KEY")
    env_file.write_text("ANTHROPIC_API_KEY=x\n")
    with pytest.raises(SystemExit):
        common.activate_project_openai_api_key()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_each_rewriting_model_gets_its_own_subfolder_and_the_first_claude_run_keeps_its_own(tmp_path):
    gpt_folder = common.description_rewrite_stage_folder(tmp_path, "gpt-6.1-sol")
    claude_folder = common.description_rewrite_stage_folder(tmp_path, "claude-sonnet-5")
    assert gpt_folder.name == "3_description_rewrite_validation_gpt-6.1-sol"
    assert claude_folder.name == "3_description_rewrite_validation"
    assert common.description_rewrites_base_name("claude-sonnet-5") == "description_rewrites_by_claude"
    assert common.description_rewrites_base_name("gpt-6.1-sol") == "description_rewrites_by_gpt_6_1_sol"
    assert common.description_rewrite_stage_key("gpt-6.1-sol") != common.description_rewrite_stage_key("claude-sonnet-5")
    assert not common.description_rewrite_stage_folder(tmp_path, "gpt-other", create=False).exists()


def test_checks_read_the_rewrites_of_the_chosen_model_only(tmp_path):
    run_folder = tmp_path
    fresh = run_folder / common.STAGE_SUBFOLDER_NAMES["fresh_specifications"]
    fresh.mkdir()
    (fresh / "plans.jsonl").write_text(json.dumps({"prompt_id": "matrix_000000", "variables": {"tone": "x"}, "description": "D."}) + "\n")
    gold = run_folder / "1_gold_labels"
    gold.mkdir()
    (gold / f"{rewrite_checks.OLD_GOLD_LABELS_BASE_NAME}_2026-10-22_00-48.jsonl").write_text(
        json.dumps({"prompt_id": "matrix_000000", "incoherent": False}) + "\n")
    for model, text in (("claude-sonnet-5", "by claude"), ("gpt-6.1-sol", "by gpt")):
        folder = common.description_rewrite_stage_folder(run_folder, model)
        (folder / f"{common.description_rewrites_base_name(model)}_2026-10-23_01-00.jsonl").write_text(
            json.dumps({"prompt_id": "matrix_000000", "rewritten_description": text, "rewrite_needed": True}) + "\n")
    assert rewrite_checks.load_inputs(run_folder)["matrix_000000"]["rewrite"]["rewritten_description"] == "by gpt"
    assert rewrite_checks.load_inputs(run_folder, "claude-sonnet-5")["matrix_000000"]["rewrite"]["rewritten_description"] == "by claude"
    assert rewrite_checks.load_inputs(run_folder, "gpt-other")["matrix_000000"]["rewrite"] is None


def test_report_notes_non_independence_only_when_the_rewriter_is_a_gpt_model():
    measures = rewrite_report.compute_measures(make_rewrite_inputs(), {}, {})
    assert "not independent" in rewrite_report.build_summary_text(measures, "gpt-6.1-sol")
    assert "not independent" not in rewrite_report.build_summary_text(measures, "claude-sonnet-5")


def test_rewrite_and_check_scripts_take_the_model_options():
    assert "--model" in (HERE / "run_description_rewrite_on_fresh_descriptions.py").read_text()
    for script in ("check_description_rewrites_with_gpt.py", "report_description_rewrite_validation.py"):
        assert "--rewrite-model" in (HERE / script).read_text()


def test_openai_cost_log_records_count_cached_tokens_inside_input_for_the_summary(tmp_path):
    sys.path.insert(0, str(common.PIPELINE_ROOT))
    import claude_api_backend_session as session
    log = tmp_path / "cost.jsonl"
    log.write_text(json.dumps({"backend": "openai", "stage": "s", "input_tokens": 1000, "cache_read_tokens": 400,
                               "output_tokens": 200, "reasoning_tokens": 150, "cost_usd": 0.5}) + "\n")
    totals = session.summarize_pipeline_cost_log(log)["overall"]
    assert totals["input_tokens"] == 600 and totals["cache_read_tokens"] == 400 and totals["reasoning_tokens"] == 150
    assert totals["cache_rate"] == pytest.approx(0.4)


def test_pipeline_import_does_not_implicitly_load_dotenv():
    import dotenv
    original = dotenv.load_dotenv
    with common.without_implicit_dotenv_loading():
        assert dotenv.load_dotenv() is False
    assert dotenv.load_dotenv is original


def test_every_paid_stage_is_a_dry_run_without_the_paid_calls_flag():
    for script in ["gold_label_dealt_hands.py", "propose_planted_replacement_values.py", "run_screen_conditions.py",
                   "run_description_rewrite_on_fresh_descriptions.py", "check_description_rewrites_with_gpt.py"]:
        assert "--run-paid-calls" in (HERE / script).read_text()


def test_pipeline_side_tests_pass_in_their_own_process():
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(HERE / "test_pipeline_side_hand_pool_and_planted_hands.py")],
                            capture_output=True, text=True, cwd=str(HERE))
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-1000:]
