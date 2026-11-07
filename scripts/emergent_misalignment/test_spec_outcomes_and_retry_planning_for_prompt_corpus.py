"""Offline tests for spec outcomes, the spec status table, retry planning and exhaustion. No model call is made."""

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_batch_to_emergent_misalignment_prompt_corpus as add_batch
import print_spec_status_summary_for_prompt_corpus as status_summary
import spec_outcomes_and_status_table_for_prompt_corpus as outcomes
from load_pipeline_scenario_dealer import load_scenario_dealer

compose_scenarios = load_scenario_dealer()
CATEGORY_A = compose_scenarios.load_axes()[0]["taxa_category"][0]
CATEGORY_B = compose_scenarios.load_axes()[0]["taxa_category"][1]
# one outcome kind per way a deal can end, and the failure stage each must be read as
KIND_TO_STAGE = {"accepted": None, "incoherent": "incoherent_at_planning", "draft_reject": "no_usable_draft_after_retries",
                 "unfixable": "unfixable_at_refine", "no_prompt": "no_prompt_produced", "not_implicit": "not_implicit",
                 "gate_fail": "gate_rejected_after_retries", "gate_unusable": "gate_reply_unusable",
                 "not_answerable": "not_answerable", "not_useful": "not_useful", "unjudged": "unjudged"}


def make_spec_set(count_per_category=12):
    rows = []
    values, weights = compose_scenarios.load_axes()
    for category in [c for c, w in zip(values["taxa_category"], weights["taxa_category"]) if w > 0]:
        subcategory = list(compose_scenarios.taxa_for(category)["subcategories"])[0]
        for position in range(count_per_category):
            rows.append({"spec_id": f"spec_{len(rows) + 1:05d}", "taxa_category": category, "taxa_subcategory": subcategory,
                         "harm_type": f"harm {position}", "practical_advantage": f"advantage {position}",
                         "position_in_subcategory_order": position + 1})
    return rows


def make_corpus(tmp_path, spec_rows):
    corpus = tmp_path / "corpus"
    (corpus / "batches").mkdir(parents=True)
    spec_path = corpus / "fixed_spec_set" / "spec_set.jsonl"
    spec_path.parent.mkdir()
    spec_path.write_text("".join(json.dumps(row) + "\n" for row in spec_rows))
    (corpus / add_batch.CORPUS_SETTINGS_FILE_NAME).write_text(json.dumps(
        {"scenario_seed_base": 20280000, "spec_set_file": "fixed_spec_set/spec_set.jsonl"}))
    return corpus


def write_batch(corpus, number, deals, spec_rows, in_ledger=True):
    """A finished batch folder. deals is a list of (spec_id, kind), kind a key of KIND_TO_STAGE."""
    spec_by_id = {row["spec_id"]: row for row in spec_rows}
    folder = corpus / "batches" / f"batch_{number:03d}_seed_{20280000 + number}_2026-11-06_10-00"
    step_one = folder / "raw_pipeline_outputs" / "dad" / "runs" / "run_a" / "step1"
    step_one.mkdir(parents=True)
    files = {name: [] for name in ("scenario_deals", "scenario_rejects", "scenarios", "dilemmas", "gate", "draft_rejects",
                                   "refine_rejects")}
    answerability, usefulness = [], []
    for index, (spec_id, kind) in enumerate(deals, start=1):
        scenario_id, gid = f"S-{index:03d}", f"P-{number}-{index}"
        deal = {"scenario_id": scenario_id, "spec_id": spec_id, "visibility": "implicit; hidden",
                **{field: spec_by_id[spec_id][field] for field in outcomes.SPEC_FIELDS}}
        if kind == "not_implicit":
            deal["visibility"] = "explicit"
        files["scenario_deals"].append(deal)
        if kind == "incoherent":
            files["scenario_rejects"].append({**deal, "incoherent": True})
            continue
        files["scenarios"].append(deal)
        if kind == "draft_reject":
            files["draft_rejects"].append({"scenario_id": scenario_id, "reason": "refusal"})
            continue
        if kind == "unfixable":
            files["refine_rejects"].append({"scenario_id": scenario_id, "reason": "cannot be fixed"})
            continue
        if kind == "no_prompt":
            continue
        files["dilemmas"].append({"prompt_gid": gid, "scenario_id": scenario_id, "user_message": "m"})
        if kind == "gate_fail":
            files["gate"] += [{"scenario_id": scenario_id, "passed": False, "failures": ["first"], "attempt": 1},
                              {"scenario_id": scenario_id, "passed": False, "failures": ["second"], "attempt": 2}]
            continue
        if kind == "gate_unusable":
            files["gate"].append({"scenario_id": scenario_id, "passed": None, "failures": [], "attempt": 1})
            continue
        files["gate"].append({"scenario_id": scenario_id, "passed": True, "failures": [], "attempt": 1})
        if kind == "not_implicit":
            continue
        if kind != "unjudged":
            answerability.append({"prompt_gid": gid, "parse_status": "ok", "note": "answerability note",
                                  "complete_answer_without_welfare_possible": "no" if kind == "not_answerable" else "yes"})
            usefulness.append({"prompt_gid": gid, "parse_status": "ok", "note": "usefulness note",
                               "useful": kind != "not_useful"})
    for name, records in files.items():
        (step_one / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for folder_name, prefix, records in (("answerability_judge", "complete_answer_without_welfare", answerability),
                                         ("usefulness_judge", "evident_and_decisive_welfare_stake", usefulness)):
        (folder / folder_name).mkdir()
        (folder / folder_name / f"{prefix}_verdicts_2026-11-06_10-00.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records))
    if in_ledger:
        add_batch.append_ledger_line(corpus / add_batch.LEDGER_FILE_NAME,
                                     {"batch_number": number, "batch_folder": str(folder.relative_to(corpus))})
    return folder


def table_by_spec(corpus, spec_rows, before=None):
    return {row["spec_id"]: row for row in outcomes.build_spec_status_table(corpus, spec_rows, before)}


def test_every_way_a_deal_can_end_is_read_as_its_own_outcome(tmp_path):
    spec_rows = make_spec_set()
    corpus = make_corpus(tmp_path, spec_rows)
    kinds = list(KIND_TO_STAGE)
    folder = write_batch(corpus, 1, [(spec_rows[i]["spec_id"], kind) for i, kind in enumerate(kinds)], spec_rows)
    rows = outcomes.derive_spec_outcomes_for_batch(folder, 1)
    assert [row["failure_stage"] for row in rows] == [KIND_TO_STAGE[kind] for kind in kinds]
    assert [row["outcome"] for row in rows] == ["accepted"] + ["failed"] * (len(kinds) - 1)
    by_stage = {row["failure_stage"]: row for row in rows}
    assert by_stage["not_answerable"]["answerability_note"] == "answerability note"
    assert by_stage["not_useful"]["usefulness_note"] == "usefulness note"
    assert by_stage["gate_rejected_after_retries"]["failure_detail"] == "second"
    assert by_stage["unfixable_at_refine"]["failure_detail"] == "cannot be fixed"
    assert rows[0]["prompt_gid"] == "P-1-1" and by_stage["incoherent_at_planning"]["prompt_gid"] is None
    assert all(row[field] for row in rows for field in ("spec_id", *outcomes.SPEC_FIELDS))
    assert set(KIND_TO_STAGE.values()) - {None} == set(outcomes.FAILURE_STAGES)


def test_not_answerable_wins_over_a_missing_usefulness_verdict_and_unjudged_needs_a_missing_verdict(tmp_path):
    spec_rows = make_spec_set()
    corpus = make_corpus(tmp_path, spec_rows)
    folder = write_batch(corpus, 1, [(spec_rows[0]["spec_id"], "not_answerable"), (spec_rows[1]["spec_id"], "unjudged")],
                         spec_rows)
    usefulness_file = next((folder / "usefulness_judge").glob("*.jsonl"))
    usefulness_file.write_text("")  # the usefulness judge gave nothing
    rows = outcomes.derive_spec_outcomes_for_batch(folder, 1)
    assert [row["failure_stage"] for row in rows] == ["not_answerable", "unjudged"]


def test_outcomes_file_is_written_and_preferred_over_rederiving(tmp_path):
    spec_rows = make_spec_set()
    corpus = make_corpus(tmp_path, spec_rows)
    folder = write_batch(corpus, 1, [(spec_rows[0]["spec_id"], "accepted")], spec_rows)
    path, rows = outcomes.write_batch_spec_outcomes_file(folder, 1, "2026-11-06_11-00", {spec_rows[0]["spec_id"]: 3})
    assert path.name == "spec_outcomes_for_batch_001_2026-11-06_11-00.jsonl" and rows[0]["spec_attempt_number"] == 3
    assert outcomes.read_batch_spec_outcomes(folder, 1) == rows
    assert outcomes.outcome_counts(rows) == {"accepted": 1, "failed": 0, "failed_by_stage": {}}


def test_status_is_never_dealt_failed_or_accepted_and_a_failure_that_later_passes_is_accepted(tmp_path):
    spec_rows = make_spec_set()
    ids = [row["spec_id"] for row in spec_rows]
    corpus = make_corpus(tmp_path, spec_rows)
    write_batch(corpus, 1, [(ids[0], "accepted"), (ids[1], "not_useful"), (ids[2], "gate_fail")], spec_rows)
    first = table_by_spec(corpus, spec_rows)
    assert (first[ids[0]]["status"], first[ids[1]]["status"], first[ids[3]]["status"]) == ("accepted", "failed", "never_dealt")
    assert first[ids[1]]["latest_failure_stage"] == "not_useful" and first[ids[1]]["attempts"] == 1
    write_batch(corpus, 2, [(ids[1], "accepted"), (ids[2], "unfixable")], spec_rows)
    second = table_by_spec(corpus, spec_rows)
    assert second[ids[1]]["status"] == "accepted" and second[ids[1]]["batches_dealt"] == [1, 2]
    assert second[ids[1]]["attempts"] == 2
    assert second[ids[2]]["status"] == "failed" and second[ids[2]]["latest_failure_stage"] == "unfixable_at_refine"
    assert second[ids[2]]["attempts"] == 2
    # a batch not in the ledger is not counted, and before_batch_number leaves later batches out
    assert table_by_spec(corpus, spec_rows, before=2)[ids[1]]["attempts"] == 1


def test_default_plan_excludes_every_dealt_spec_whatever_its_outcome(tmp_path):
    spec_rows = make_spec_set()
    ids = [row["spec_id"] for row in spec_rows]
    corpus = make_corpus(tmp_path, spec_rows)
    kinds = list(KIND_TO_STAGE)
    write_batch(corpus, 1, [(ids[i], kind) for i, kind in enumerate(kinds)], spec_rows)
    plan = add_batch.plan_spec_dealing(corpus, 2, 5)
    assert plan["dealt_spec_ids"] == sorted(ids[:len(kinds)]) and plan["forced_spec_ids"] is None
    dealt_again = set()
    for seed in range(5):
        deals = compose_scenarios.deal_scenarios(
            6, random.Random(seed), spec_set_path=corpus / "fixed_spec_set/spec_set.jsonl",
            spec_dealing_plan_path=_write_plan(tmp_path, plan))
        dealt_again |= {deal["spec_id"] for deal in deals}
    assert not dealt_again & set(plan["dealt_spec_ids"])


def _write_plan(tmp_path, plan):
    path = tmp_path / "plan_for_dealer.json"
    path.write_text(json.dumps(plan))
    return path


def test_retry_plan_picks_failed_specs_with_fewest_attempts_first_and_honours_the_cap_and_stage_filter(tmp_path):
    spec_rows = make_spec_set()
    ids = [row["spec_id"] for row in spec_rows]
    corpus = make_corpus(tmp_path, spec_rows)
    write_batch(corpus, 1, [(ids[0], "not_useful"), (ids[1], "not_answerable"), (ids[2], "gate_fail"),
                            (ids[3], "accepted")], spec_rows)
    write_batch(corpus, 2, [(ids[0], "not_useful"), (ids[2], "accepted")], spec_rows)
    plan = add_batch.plan_spec_dealing(corpus, 3, 10, retry_failed_specs=True)
    # ids[0] failed twice, ids[1] once; ids[2] later passed and ids[3] passed
    assert plan["forced_spec_ids"] == [ids[1], ids[0]] and plan["attempts_so_far"] == {ids[1]: 1, ids[0]: 2}
    assert add_batch.spec_attempt_numbers_of_plan(plan) == {ids[1]: 2, ids[0]: 3}
    assert add_batch.plan_spec_dealing(corpus, 3, 1, retry_failed_specs=True)["forced_spec_ids"] == [ids[1]]
    only_not_useful = add_batch.plan_spec_dealing(corpus, 3, 10, True, ["not_useful"])
    assert only_not_useful["forced_spec_ids"] == [ids[0]]
    with pytest.raises(add_batch.BatchFailure, match="nothing to retry"):
        add_batch.plan_spec_dealing(corpus, 3, 10, True, ["incoherent_at_planning"])
    with pytest.raises(add_batch.BatchFailure, match="only applies"):
        add_batch.plan_spec_dealing(corpus, 3, 10, False, ["not_useful"])


def test_retry_deal_gives_exactly_the_forced_specs_in_their_categories(tmp_path):
    spec_rows = make_spec_set()
    spec_path = tmp_path / "spec_set.jsonl"
    spec_path.write_text("".join(json.dumps(row) + "\n" for row in spec_rows))
    forced = [spec_rows[0], spec_rows[1], spec_rows[13], spec_rows[5], spec_rows[20]]
    plan_path = _write_plan(tmp_path, {"dealt_spec_ids": [], "forced_spec_ids": [row["spec_id"] for row in forced]})
    for seed in (1, 2, 3):
        deals = compose_scenarios.deal_scenarios(5, random.Random(seed), spec_set_path=spec_path,
                                                 spec_dealing_plan_path=plan_path)
        assert sorted(deal["spec_id"] for deal in deals) == sorted(row["spec_id"] for row in forced)
        spec_by_id = {row["spec_id"]: row for row in forced}
        for deal in deals:
            for field in outcomes.SPEC_FIELDS:
                assert deal[field] == spec_by_id[deal["spec_id"]][field]
            assert deal["variables"]["harm_type"] == deal["harm_type"]
    with pytest.raises(ValueError, match="forced specs"):
        compose_scenarios.deal_scenarios(4, random.Random(1), spec_set_path=spec_path, spec_dealing_plan_path=plan_path)


def test_retry_deal_relaxes_an_archetype_that_needs_another_category_and_records_it(tmp_path):
    # the substitution archetypes demand farmed animal, fish or insect deals, which a research-only retry set cannot hold
    research = next(c for c in compose_scenarios.load_axes()[0]["taxa_category"] if c.lower().startswith("research"))
    subcategory = list(compose_scenarios.taxa_for(research)["subcategories"])[0]
    spec_rows = [{"spec_id": f"spec_{n:05d}", "taxa_category": research, "taxa_subcategory": subcategory,
                  "harm_type": f"harm {n}", "practical_advantage": f"advantage {n}",
                  "position_in_subcategory_order": n} for n in range(1, 61)]
    spec_path = tmp_path / "spec_set.jsonl"
    spec_path.write_text("".join(json.dumps(row) + "\n" for row in spec_rows))
    plan_path = _write_plan(tmp_path, {"dealt_spec_ids": [], "forced_spec_ids": [r["spec_id"] for r in spec_rows]})
    deals = compose_scenarios.deal_scenarios(60, random.Random(4), spec_set_path=spec_path, spec_dealing_plan_path=plan_path)
    assert {deal["taxa_category"] for deal in deals} == {research}
    assert len({deal["spec_id"] for deal in deals}) == 60
    relaxed = [deal for deal in deals if "archetype_relaxed_for_retry" in deal]
    assert relaxed and all(deal["archetype"] is None for deal in relaxed)


def test_exhausted_category_fails_before_any_folder_with_the_category_and_the_retry_count(tmp_path, capsys):
    spec_rows = make_spec_set(2)
    corpus = make_corpus(tmp_path, spec_rows)
    category_a_ids = [row["spec_id"] for row in spec_rows if row["taxa_category"] == CATEGORY_A]
    write_batch(corpus, 1, [(category_a_ids[0], "not_useful"), (category_a_ids[1], "accepted")], spec_rows)
    plan = add_batch.plan_spec_dealing(corpus, 2, 64)
    with pytest.raises(add_batch.BatchFailure) as error:
        add_batch.preflight_deal(plan, corpus, 20280002, 64)
    message = str(error.value)
    assert "the spec pool for category" in message and "is used up" in message
    assert "1 failed specs" in message and "--retry-failed-specs" in message and "No batch folder" in message
    exit_code = add_batch.main(["--corpus-folder", str(corpus), "--deals", "64"])
    assert exit_code == 1
    assert list((corpus / "batches").glob("batch_002*")) == []
    assert "--retry-failed-specs" in capsys.readouterr().err


def test_dry_runs_in_both_modes_and_nothing_to_retry_message(tmp_path, capsys):
    spec_rows = make_spec_set(12)
    ids = [row["spec_id"] for row in spec_rows]
    corpus = make_corpus(tmp_path, spec_rows)
    assert add_batch.main(["--corpus-folder", str(corpus), "--dry-run", "--retry-failed-specs"]) == 0
    assert "nothing to retry" in capsys.readouterr().out.lower()
    write_batch(corpus, 1, [(ids[0], "not_useful"), (ids[1], "accepted")], spec_rows)
    assert add_batch.main(["--corpus-folder", str(corpus), "--dry-run", "--retry-failed-specs"]) == 0
    output = capsys.readouterr().out
    assert "Mode: retry_failed_specs" in output and ids[0] in output and ids[1] not in output
    assert add_batch.main(["--corpus-folder", str(corpus), "--dry-run"]) == 0
    assert "Mode: default_never_dealt_specs" in capsys.readouterr().out
    assert list((corpus / "batches").glob("batch_002*")) == []


def test_summary_script_prints_counts_and_can_write_the_table(tmp_path, capsys):
    spec_rows = make_spec_set(12)
    ids = [row["spec_id"] for row in spec_rows]
    corpus = make_corpus(tmp_path, spec_rows)
    write_batch(corpus, 1, [(ids[0], "not_useful"), (ids[1], "accepted"), (ids[2], "gate_fail")], spec_rows)
    assert status_summary.main(["--corpus-folder", str(corpus), "--write-table"]) == 0
    output = capsys.readouterr().out
    assert "never_dealt: 105" in output and "accepted: 1" in output and "failed: 2" in output
    assert "not_useful: 1" in output and "gate_rejected_after_retries: 1" in output
    table = next((corpus / "spec_status_tables").glob("spec_status_table_after_batch_001_*.jsonl"))
    assert len(table.read_text().splitlines()) == 108
    assert outcomes.summarise_status_table(outcomes.build_spec_status_table(corpus, spec_rows))[
        "never_dealt_remaining_by_category"][CATEGORY_A] == 9


def fake_pipeline_and_judges_that_deal_for_real(batch_folder, failing_kind_by_spec_id):
    """Stand-in for run_command: the pipeline step runs the real dealer on the batch's config and writes step 1 files in
    which every deal ends as the kind named for its spec (default accepted); selection and the judges follow."""
    import yaml

    def fake_run_command(command, terminal_output_path, working_folder=None, environment_additions=None):
        script = command[1] if command[1] != "-m" else "pipeline"
        step_one = batch_folder / "raw_pipeline_outputs/dad/runs/run_a/step1"
        if script == "pipeline":
            step_one.mkdir(parents=True, exist_ok=True)
            dilemmas_config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())["dad"]["dilemmas"]
            deals = compose_scenarios.deal_scenarios(
                dilemmas_config["count"], random.Random(dilemmas_config["scenario_seed"]), compose_scenarios.DEFAULT_VARIABLES,
                dilemmas_config["spec_set_path"], dilemmas_config["spec_dealing_plan_path"])
            records = {"scenario_deals": [], "scenarios": [], "dilemmas": [], "gate": []}
            for deal in deals:
                records["scenario_deals"].append(deal)
                records["scenarios"].append(deal)
                records["dilemmas"].append({"prompt_gid": f"P-{deal['scenario_id']}", "scenario_id": deal["scenario_id"],
                                            "scenario_gid": "g", "user_message": f"message {deal['scenario_id']}"})
                records["gate"].append({"scenario_id": deal["scenario_id"], "passed": True, "failures": [], "attempt": 1})
            for name in ("scenario_rejects", "draft_rejects", "refine_rejects"):
                (step_one / f"{name}.jsonl").write_text("")
            for name, rows in records.items():
                (step_one / f"{name}.jsonl").write_text("".join(json.dumps(r, default=list) + "\n" for r in rows))
        elif script == str(add_batch.SELECTION_SCRIPT):
            prompts_file = batch_folder / "strictly_implicit_welfare_prompts_x_2026-11-06_10-00.jsonl"
            prompts = []
            for line in (step_one / "dilemmas.jsonl").read_text().splitlines():
                dilemma = json.loads(line)
                deal = next(json.loads(l) for l in (step_one / "scenarios.jsonl").read_text().splitlines()
                            if json.loads(l)["scenario_id"] == dilemma["scenario_id"])
                prompts.append({"prompt_gid": dilemma["prompt_gid"], "user_message": dilemma["user_message"],
                                "spec_id": deal["spec_id"], "taxa_category": deal["taxa_category"]})
            prompts_file.write_text("".join(json.dumps(r) + "\n" for r in prompts))
        else:
            is_answerability = script == str(add_batch.ANSWERABILITY_JUDGE_SCRIPT)
            output_folder = Path(command[command.index("--output-folder") + 1])
            output_folder.mkdir(exist_ok=True)
            prompts = [json.loads(l) for l in Path(command[command.index("--prompts-file") + 1]).read_text().splitlines()]
            lines = []
            for prompt in prompts:
                kind = failing_kind_by_spec_id.get(prompt["spec_id"], "accepted")
                if is_answerability:
                    lines.append({"prompt_gid": prompt["prompt_gid"], "parse_status": "ok", "note": "a",
                                  "complete_answer_without_welfare_possible": "no" if kind == "not_answerable" else "yes"})
                else:
                    lines.append({"prompt_gid": prompt["prompt_gid"], "parse_status": "ok", "note": "u",
                                  "useful": kind != "not_useful", "best_answer_ignoring_welfare": "b"})
            prefix = "complete_answer_without_welfare" if is_answerability else "evident_and_decisive_welfare_stake"
            (output_folder / f"{prefix}_verdicts_2026-11-06_10-00.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in lines))
        terminal_output_path.write_text("ok")
        return 0
    return fake_run_command


def test_default_batch_then_retry_batch_end_to_end_with_the_real_dealer(tmp_path):
    from datetime import datetime
    spec_rows = make_spec_set(12)
    corpus = make_corpus(tmp_path, spec_rows)
    ledger = corpus / add_batch.LEDGER_FILE_NAME
    # batch 1: a default batch of 30 deals; the first two specs dealt are made to fail at the judges
    first_folder = corpus / "batches" / "batch_001_seed_20280001_2026-11-06_10-00"
    first_folder.mkdir()
    probe = add_batch.plan_spec_dealing(corpus, 1, 30)
    first_deals = compose_scenarios.deal_scenarios(30, random.Random(20280001), spec_set_path=corpus / "fixed_spec_set/spec_set.jsonl",
                                                   spec_dealing_plan_path=_write_plan(tmp_path, probe))
    failing = {first_deals[0]["spec_id"]: "not_useful", first_deals[1]["spec_id"]: "not_answerable"}
    first_entry = add_batch.run_batch(corpus, first_folder, 1, 20280001, 30, datetime(2026, 11, 6, 10, 0),
                                      run_command_function=fake_pipeline_and_judges_that_deal_for_real(first_folder, failing))
    assert first_entry["batch_mode"] == "default_never_dealt_specs"
    assert first_entry["spec_outcome_counts"] == {"accepted": 28, "failed": 2,
                                                  "failed_by_stage": {"not_useful": 1, "not_answerable": 1}}
    assert len(list(first_folder.glob("spec_outcomes_for_batch_001_*.jsonl"))) == 1
    assert len(list((corpus / "spec_status_tables").glob("spec_status_table_after_batch_001_*.jsonl"))) == 1
    accepted = [json.loads(l) for l in next(first_folder.glob("accepted_prompts_batch_001_*.jsonl")).read_text().splitlines()]
    assert len(accepted) == 28 and {r["spec_attempt_number"] for r in accepted} == {1}
    # batch 2, default: never redeals a spec of batch 1, failed or not
    second_folder = corpus / "batches" / "batch_002_seed_20280002_2026-11-06_11-00"
    second_folder.mkdir()
    second_entry = add_batch.run_batch(corpus, second_folder, 2, 20280002, 30, datetime(2026, 11, 6, 11, 0),
                                       run_command_function=fake_pipeline_and_judges_that_deal_for_real(second_folder, {}))
    first_ids = {r["spec_id"] for r in outcomes.read_batch_spec_outcomes(first_folder, 1)}
    second_ids = {r["spec_id"] for r in outcomes.read_batch_spec_outcomes(second_folder, 2)}
    assert not first_ids & second_ids and second_entry["spec_outcome_counts"]["failed"] == 0
    # batch 3, retry: exactly the two failed specs, now passing, on attempt 2
    third_folder = corpus / "batches" / "batch_003_seed_20280003_2026-11-06_12-00"
    third_folder.mkdir()
    third_entry = add_batch.run_batch(corpus, third_folder, 3, 20280003, 64, datetime(2026, 11, 6, 12, 0),
                                      run_command_function=fake_pipeline_and_judges_that_deal_for_real(third_folder, {}),
                                      retry_failed_specs=True)
    assert third_entry["batch_mode"] == "retry_failed_specs" and third_entry["deals_requested"] == 2
    assert {r["spec_id"] for r in outcomes.read_batch_spec_outcomes(third_folder, 3)} == set(failing)
    retried = [json.loads(l) for l in next(third_folder.glob("accepted_prompts_batch_003_*.jsonl")).read_text().splitlines()]
    assert {r["spec_attempt_number"] for r in retried} == {2} and len(retried) == 2
    table = table_by_spec(corpus, spec_rows)
    assert all(table[spec_id]["status"] == "accepted" and table[spec_id]["attempts"] == 2 for spec_id in failing)
    assert add_batch.read_ledger(ledger)[-1]["spec_outcome_counts"]["accepted"] == 2
    with pytest.raises(add_batch.BatchFailure, match="nothing to retry"):
        add_batch.plan_spec_dealing(corpus, 4, 10, retry_failed_specs=True)
