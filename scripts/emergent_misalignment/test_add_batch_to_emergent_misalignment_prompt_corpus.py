"""Mock-only tests for adding a batch to the emergent misalignment prompt corpus. No model call or real subprocess is made."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_batch_to_emergent_misalignment_prompt_corpus as add_batch

LEDGER = add_batch.LEDGER_FILE_NAME


def make_corpus(tmp_path, ledger_numbers=(), folder_numbers=()):
    corpus = tmp_path / "corpus"
    (corpus / "batches").mkdir(parents=True)
    for number in folder_numbers:
        (corpus / "batches" / f"batch_{number:03d}_seed_{20270000 + number}_2026-11-06_10-00").mkdir()
    for number in ledger_numbers:
        add_batch.append_ledger_line(corpus / LEDGER, {"batch_number": number})
    return corpus


def test_first_batch_is_number_one_with_seed(tmp_path):
    corpus = make_corpus(tmp_path)
    assert add_batch.next_batch_number(corpus) == 1
    assert add_batch.scenario_seed_for_batch(1) == 20270001


def test_next_number_follows_ledger_and_folders(tmp_path):
    corpus = make_corpus(tmp_path, ledger_numbers=(1, 2), folder_numbers=(1, 2, 3))
    assert add_batch.next_batch_number(corpus) == 4
    assert add_batch.incomplete_batch_folders(corpus) == [corpus / "batches" / "batch_003_seed_20270003_2026-11-06_10-00"]


def test_complete_corpus_has_no_incomplete_folders(tmp_path):
    corpus = make_corpus(tmp_path, ledger_numbers=(1,), folder_numbers=(1,))
    assert add_batch.incomplete_batch_folders(corpus) == []


def test_new_batch_refused_when_incomplete_folder_exists(tmp_path, capsys):
    corpus = make_corpus(tmp_path, ledger_numbers=(1,), folder_numbers=(1, 2))
    assert add_batch.main(["--corpus-folder", str(corpus)]) == 1
    assert "--resume-batch-folder" in capsys.readouterr().err


def test_dry_run_creates_nothing(tmp_path, capsys):
    corpus = tmp_path / "fresh_corpus"
    assert add_batch.main(["--corpus-folder", str(corpus), "--dry-run"]) == 0
    assert not corpus.exists()
    assert "Batch number: 1" in capsys.readouterr().out


def test_hash_depends_on_contents_and_name_order(tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("one"), second.write_text("two")
    hash_one = add_batch.hash_files([second, first])
    assert hash_one == add_batch.hash_files([first, second])
    assert len(hash_one) == 12
    second.write_text("changed")
    assert add_batch.hash_files([first, second]) != hash_one


def verdicts(**by_gid):
    return {gid: value for gid, value in by_gid.items()}


def test_accepted_records_and_missing_verdicts():
    prompts = [{"prompt_gid": f"P-000{n}", "user_message": f"message {n}"} for n in range(1, 5)]
    answerability = {"P-0001": {"complete_answer_without_welfare_possible": "yes", "note": "a1"},
                     "P-0002": {"complete_answer_without_welfare_possible": "no", "note": "a2"},
                     "P-0003": {"complete_answer_without_welfare_possible": "yes", "note": "a3"}}  # P-0004 missing
    usefulness = {"P-0001": {"useful": True, "note": "u1", "best_answer_ignoring_welfare": "best1"},
                  "P-0002": {"useful": True, "note": "u2", "best_answer_ignoring_welfare": "x"},
                  "P-0004": {"useful": True, "note": "u4", "best_answer_ignoring_welfare": "y"}}  # P-0003 missing
    accepted, counts = add_batch.build_accepted_records(prompts, answerability, usefulness, 7, 20270007, "abc123abc123")
    assert [r["corpus_prompt_id"] for r in accepted] == ["B007-P-0001"]
    record = accepted[0]
    assert record["user_message"] == "message 1" and record["corpus_batch_number"] == 7
    assert record["scenario_seed"] == 20270007 and record["generation_prompt_version_hash"] == "abc123abc123"
    assert (record["answerability_note"], record["usefulness_note"], record["best_answer_ignoring_welfare"]) == ("a1", "u1", "best1")
    assert counts["answerable"] == 2 and counts["useful"] == 3 and counts["accepted"] == 1
    assert counts["unjudged_prompt_ids"] == ["P-0003", "P-0004"]


def test_accepted_records_carry_generation_dates():
    prompts = [{"prompt_gid": "P-1", "user_message": "m"}]
    accepted, _ = add_batch.build_accepted_records(
        prompts, {"P-1": {"complete_answer_without_welfare_possible": "yes", "note": "a"}},
        {"P-1": {"useful": True, "note": "u", "best_answer_ignoring_welfare": "b"}}, 1, 20270001, "h",
        "2026-11-06T14:05:00+00:00", "2026-11-06T14:40:00+00:00")
    assert accepted[0]["batch_generation_started_at"] == "2026-11-06T14:05:00+00:00"
    assert accepted[0]["batch_generation_finished_at"] == "2026-11-06T14:40:00+00:00"


def test_generation_start_time_is_kept_on_resume(tmp_path):
    add_batch.update_generation_times_state(tmp_path, "batch_generation_started_at", "first", keep_existing=True)
    add_batch.update_generation_times_state(tmp_path, "batch_generation_started_at", "second", keep_existing=True)
    add_batch.update_generation_times_state(tmp_path, "batch_generation_finished_at", "end", keep_existing=False)
    assert add_batch.read_generation_times_state(tmp_path) == {
        "batch_generation_started_at": "first", "batch_generation_finished_at": "end"}


def test_ok_verdicts_keep_latest_ok_record(tmp_path):
    folder = tmp_path / "judge"
    folder.mkdir()
    pattern = add_batch.USEFULNESS_VERDICT_PATTERN
    (folder / "evident_and_decisive_welfare_stake_verdicts_2026-11-06_10-00.jsonl").write_text(
        json.dumps({"prompt_gid": "P-1", "parse_status": "unparseable"}) + "\n"
        + json.dumps({"prompt_gid": "P-2", "parse_status": "ok", "useful": False}) + "\n")
    (folder / "evident_and_decisive_welfare_stake_verdicts_2026-11-06_10-05.jsonl").write_text(
        json.dumps({"prompt_gid": "P-1", "parse_status": "ok", "useful": True}) + "\n")
    result = add_batch.ok_verdicts_in_folder(folder, pattern)
    assert result["P-1"]["useful"] is True and result["P-2"]["useful"] is False


def test_ledger_append_is_one_line_per_entry(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    add_batch.append_ledger_line(ledger, {"batch_number": 1})
    add_batch.append_ledger_line(ledger, {"batch_number": 2})
    assert [e["batch_number"] for e in add_batch.read_ledger(ledger)] == [1, 2]


def test_write_batch_config_sets_seed_and_count(tmp_path):
    path = add_batch.write_batch_config(tmp_path, 20270003, 10, "2026-11-06_10-00")
    import yaml
    config = yaml.safe_load(path.read_text())
    assert config["dad"]["dilemmas"]["scenario_seed"] == 20270003 and config["dad"]["dilemmas"]["count"] == 10
    assert config["backend"] == "claude_code"


def fake_pipeline_and_judges(batch_folder, prompt_count=3, unparseable_gid=None, failed_call_gid=None):
    """A stand-in for run_command that writes the files each real subprocess would."""
    calls = []

    def fake_run_command(command, terminal_output_path, working_folder=None, environment_additions=None):
        calls.append(command)
        script = command[1] if command[1] != "-m" else "pipeline"
        if script == "pipeline":
            step_one = batch_folder / "raw_pipeline_outputs/dad/runs/run_a/step1"
            step_one.mkdir(parents=True, exist_ok=True)
        elif script == str(add_batch.SELECTION_SCRIPT):
            records = [{"prompt_gid": f"P-{n}", "user_message": f"message {n}", "taxa_category": "farmed animals"}
                       for n in range(prompt_count)]
            (batch_folder / f"strictly_implicit_welfare_prompts_{prompt_count}_2026-11-06_10-00.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records))
        else:
            is_answerability = script == str(add_batch.ANSWERABILITY_JUDGE_SCRIPT)
            output_folder = Path(command[command.index("--output-folder") + 1])
            output_folder.mkdir(exist_ok=True)
            stamp = f"2026-11-06_10-{len(calls):02d}"
            lines = []
            for n in range(prompt_count):
                gid = f"P-{n}"
                if gid == failed_call_gid and not is_answerability:
                    continue  # this judge's call failed: no verdict record is written at all
                status = "unparseable" if gid == unparseable_gid else "ok"
                record = ({"prompt_gid": gid, "parse_status": status, "complete_answer_without_welfare_possible": "yes", "note": "n"}
                          if is_answerability else
                          {"prompt_gid": gid, "parse_status": status, "useful": True, "note": "n", "best_answer_ignoring_welfare": "b"})
                lines.append(json.dumps(record))
            prefix = "complete_answer_without_welfare" if is_answerability else "evident_and_decisive_welfare_stake"
            (output_folder / f"{prefix}_verdicts_{stamp}.jsonl").write_text("\n".join(lines) + "\n")
        terminal_output_path.write_text("ok")
        return 0
    return fake_run_command, calls


def run_fake_batch(tmp_path, unparseable_gid=None):
    corpus = make_corpus(tmp_path)
    batch_folder = corpus / "batches" / "batch_001_seed_20270001_2026-11-06_10-00"
    batch_folder.mkdir()
    fake, calls = fake_pipeline_and_judges(batch_folder, unparseable_gid=unparseable_gid)
    entry = add_batch.run_batch(corpus, batch_folder, 1, 20270001, 5, datetime(2026, 11, 6, 10, 0),
                                run_command_function=fake)
    return corpus, batch_folder, entry, calls


def test_full_batch_with_mocked_subprocesses(tmp_path):
    corpus, batch_folder, entry, calls = run_fake_batch(tmp_path)
    assert len(calls) == 6  # pipeline, selection, two judges twice each
    assert entry["accepted_count"] == 3 and entry["unjudged_count"] == 0
    assert add_batch.read_ledger(corpus / LEDGER) == [entry]
    assert len(list(batch_folder.glob("accepted_prompts_batch_001_*.jsonl"))) == 1
    assert len(list(batch_folder.glob("batch_summary_*.json"))) == 1
    assert (batch_folder / "generation_and_judge_prompts_used_for_this_batch/step1a_scenario.txt").exists()
    assert (batch_folder / "generation_and_judge_prompts_used_for_this_batch/animal.py").exists()
    snapshot_folder = batch_folder / "generation_and_judge_prompts_used_for_this_batch"
    assert (snapshot_folder / "compose_scenarios.py").exists()
    assert entry["generation_prompt_version_hash"] == add_batch.generation_prompt_version_hash(
        snapshot_folder, profile_folder=snapshot_folder, dealer_folder=snapshot_folder)
    assert "PIPELINE_OUTPUT_ROOT" not in calls[0]
    state = add_batch.read_generation_times_state(batch_folder)
    assert entry["batch_generation_started_at"] == state["batch_generation_started_at"]
    assert entry["batch_generation_finished_at"] == state["batch_generation_finished_at"]
    accepted_line = json.loads(next(batch_folder.glob("accepted_prompts_batch_001_*.jsonl")).read_text().splitlines()[0])
    assert accepted_line["batch_generation_started_at"] == state["batch_generation_started_at"]
    summary = json.loads(next(batch_folder.glob("batch_summary_*.json")).read_text())
    assert summary["batch_generation_finished_at"] == state["batch_generation_finished_at"]
    assert state["batch_generation_started_at"][-6] in "+-"  # ISO 8601 with offset


def test_unjudged_prompts_are_recorded_but_batch_still_ledgered(tmp_path):
    corpus, batch_folder, entry, calls = run_fake_batch(tmp_path, unparseable_gid="P-1")
    # the fake never gives P-1 an ok verdict, even on the second run
    assert entry["unjudged_count"] == 1 and entry["accepted_count"] == 2
    summary = json.loads(next(batch_folder.glob("batch_summary_*.json")).read_text())
    assert summary["unjudged_prompt_ids"] == ["P-1"]
    assert add_batch.read_ledger(corpus / LEDGER) == [entry]


def test_prompt_with_no_verdict_from_a_judge_stops_the_batch_before_any_ledger_line(tmp_path):
    corpus = make_corpus(tmp_path)
    batch_folder = corpus / "batches" / "batch_001_seed_20270001_2026-11-06_10-00"
    batch_folder.mkdir()
    fake, calls = fake_pipeline_and_judges(batch_folder, failed_call_gid="P-1")
    with pytest.raises(add_batch.BatchFailure, match=r"usefulness judge: 1 of 3 prompts.*same command again"):
        add_batch.run_batch(corpus, batch_folder, 1, 20270001, 5, datetime(2026, 11, 6, 10, 0), run_command_function=fake)
    assert not (corpus / LEDGER).exists() or add_batch.read_ledger(corpus / LEDGER) == []
    assert list(batch_folder.glob("accepted_prompts_batch_001_*.jsonl")) == []
    assert list(batch_folder.glob("batch_summary_*.json")) == []
    # once the judge can answer for that prompt, running the same command again finishes the batch
    fixed_fake, _ = fake_pipeline_and_judges(batch_folder)
    entry = add_batch.run_batch(corpus, batch_folder, 1, 20270001, 5, datetime(2026, 11, 6, 10, 0),
                                run_command_function=fixed_fake)
    assert entry["unjudged_count"] == 0 and add_batch.read_ledger(corpus / LEDGER) == [entry]


def test_pipeline_output_saying_hit_your_limit_counts_as_a_usage_limit_stop(tmp_path):
    output_path = tmp_path / "terminal_output.txt"
    output_path.write_text("You've hit your limit \u00b7 resets 7pm (Europe/London)", encoding="utf-8")
    assert add_batch.step_stopped_for_usage_limit(1, output_path) is True


def test_resume_skips_completed_steps_and_resumes_pipeline_run(tmp_path):
    corpus, batch_folder, entry, calls = run_fake_batch(tmp_path)
    (corpus / LEDGER).unlink()  # pretend the ledger line was never written
    fake, second_calls = fake_pipeline_and_judges(batch_folder)
    add_batch.run_batch(corpus, batch_folder, 1, 20270001, 5, datetime(2026, 11, 6, 10, 0), run_command_function=fake)
    assert second_calls == []  # every step already complete
    assert len(add_batch.read_ledger(corpus / LEDGER)) == 1


def test_pipeline_command_resumes_with_run_id():
    command = add_batch.pipeline_command(Path("c.yaml"), 3, "run_a")
    assert command[-3:] == ["--resume", "--run-id", "run_a"]


def test_generation_hash_covers_the_species_profile_file(tmp_path):
    prompt_folder, profile_folder = tmp_path / "prompts", tmp_path / "profiles"
    prompt_folder.mkdir(), profile_folder.mkdir()
    for name in add_batch.GENERATION_PROMPT_FILE_NAMES:
        (prompt_folder / name).write_text(name)
    profile_file = profile_folder / add_batch.GENERATION_PROFILE_FILE_NAME
    dealer_file = profile_folder / add_batch.GENERATION_DEALER_FILE_NAME
    profile_file.write_text("TAXA = {}")
    dealer_file.write_text("def deal(): pass")
    hash_before = add_batch.generation_prompt_version_hash(prompt_folder, profile_folder, profile_folder)
    profile_file.write_text("TAXA = {'changed': 1}")
    assert add_batch.generation_prompt_version_hash(prompt_folder, profile_folder, profile_folder) != hash_before
    profile_file.write_text("TAXA = {}")
    dealer_file.write_text("def deal(): return 1")
    assert add_batch.generation_prompt_version_hash(prompt_folder, profile_folder, profile_folder) != hash_before


def test_snapshot_copies_the_species_profile_file(tmp_path):
    batch_folder = tmp_path / "batch"
    add_batch.snapshot_prompts(batch_folder)
    snapshot = batch_folder / "generation_and_judge_prompts_used_for_this_batch" / add_batch.GENERATION_PROFILE_FILE_NAME
    assert snapshot.read_bytes() == (add_batch.GENERATION_PROFILE_FOLDER / add_batch.GENERATION_PROFILE_FILE_NAME).read_bytes()


def test_snapshot_copies_the_scenario_dealer_file(tmp_path):
    batch_folder = tmp_path / "batch"
    add_batch.snapshot_prompts(batch_folder)
    snapshot = batch_folder / "generation_and_judge_prompts_used_for_this_batch" / add_batch.GENERATION_DEALER_FILE_NAME
    assert snapshot.read_bytes() == (add_batch.GENERATION_DEALER_FOLDER / add_batch.GENERATION_DEALER_FILE_NAME).read_bytes()


# ---------- selection carries dealt fields; prior corpus use counts ----------

def write_step_one_run(batch_folder, deals):
    """A pipeline step 1 folder for a batch. Each deal is (scenario_id, category, subcategory, harm_type,
    practical_advantage, passed_gate, optional spec_id); harm_type and practical_advantage may be None for older batches."""
    step_one = batch_folder / "raw_pipeline_outputs" / "dad" / "runs" / "run_a" / "step1"
    step_one.mkdir(parents=True)
    dilemmas, scenarios, gates = [], [], []
    for scenario_id, category, subcategory, harm_type, practical_advantage, passed_gate, *spec_id in deals:
        dilemmas.append({"prompt_gid": f"P-{scenario_id}", "scenario_id": scenario_id, "scenario_gid": f"S-{scenario_id}",
                         "user_message": "a message"})
        scenario = {"scenario_id": scenario_id, "visibility": "implicit; hidden harm", "taxa_category": category,
                    "taxa_subcategory": subcategory, "variables": {"welfare_partner": "money", "scope": "dozens",
                                                                   "visibility": "ignored duplicate"}}
        if spec_id:
            scenario["spec_id"] = spec_id[0]
        if harm_type is not None:
            scenario["harm_type"] = harm_type
            scenario["practical_advantage"] = practical_advantage
            scenario["variables"]["harm_type"] = harm_type
            scenario["variables"]["practical_advantage"] = practical_advantage
        scenarios.append(scenario)
        gates.append({"scenario_id": scenario_id, "passed": passed_gate})
    for name, records in (("dilemmas", dilemmas), ("scenarios", scenarios), ("gate", gates)):
        (step_one / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return step_one


def test_selection_carries_dealt_fields_including_new_ones(tmp_path):
    step_one = write_step_one_run(tmp_path, [("S-001", "farmed animals", "pigs", "housing", "lower cost", True)])
    selected, _ = add_batch.select_prompts(step_one, 10)
    record = selected[0]
    assert record["harm_type"] == "housing" and record["practical_advantage"] == "lower cost"
    assert record["taxa_category"] == "farmed animals" and record["taxa_subcategory"] == "pigs"
    # an axis only present in the scenario's raw variables is carried, without overriding a card field
    assert record["welfare_partner"] == "money" and record["scope"] == "dozens"
    assert record["visibility"] == "implicit; hidden harm"


def test_selection_leaves_new_fields_empty_for_older_runs(tmp_path):
    step_one = write_step_one_run(tmp_path, [("S-001", "farmed animals", "pigs", None, None, True)])
    record = add_batch.select_prompts(step_one, 10)[0][0]
    assert record["harm_type"] is None and record["practical_advantage"] is None


def test_selection_carries_the_spec_id(tmp_path):
    step_one = write_step_one_run(tmp_path, [("S-001", "farmed animals", "pigs", "housing", "lower cost", True,
                                              "spec_00007")])
    assert add_batch.select_prompts(step_one, 10)[0][0]["spec_id"] == "spec_00007"


def test_batch_config_names_the_spec_set_and_dealing_plan(tmp_path):
    import yaml
    spec_path, plan_path = tmp_path / "spec_set.jsonl", tmp_path / "plan.json"
    config = yaml.safe_load(add_batch.write_batch_config(tmp_path, 20280003, 10, "2026-11-06_10-00",
                                                         spec_set_path=spec_path,
                                                         spec_dealing_plan_path=plan_path).read_text())
    assert config["dad"]["dilemmas"]["spec_set_path"] == str(spec_path)
    assert config["dad"]["dilemmas"]["spec_dealing_plan_path"] == str(plan_path)
    plain = yaml.safe_load(add_batch.write_batch_config(tmp_path, 1, 1, "x").read_text())["dad"]["dilemmas"]
    assert plain["spec_set_path"] is None and plain["spec_dealing_plan_path"] is None


# ---------- corpus settings: seed base and spec set ----------

def write_settings_and_spec_set(corpus):
    spec_path = corpus / "fixed_spec_set" / "fixed_spec_set_test.jsonl"
    spec_path.parent.mkdir()
    spec_path.write_text(json.dumps({"spec_id": "spec_00001", "taxa_category": "c", "taxa_subcategory": "s",
                                     "harm_type": "h", "practical_advantage": "a",
                                     "position_in_subcategory_order": 1}) + "\n")
    (corpus / add_batch.CORPUS_SETTINGS_FILE_NAME).write_text(json.dumps(
        {"scenario_seed_base": 20280000, "spec_set_file": "fixed_spec_set/fixed_spec_set_test.jsonl"}))
    return spec_path


def test_corpus_without_settings_keeps_the_old_seed_base_and_has_no_spec_set(tmp_path):
    corpus = make_corpus(tmp_path)
    assert add_batch.read_corpus_settings(corpus) == {"scenario_seed_base": 20270000, "spec_set_file": None}
    assert add_batch.spec_set_path_for_corpus(corpus) is None


def test_settings_give_seed_base_and_spec_set_and_dry_run_shows_them(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(add_batch, "preflight_deal", lambda *arguments: None)  # the one-spec set is not dealable
    corpus = make_corpus(tmp_path)
    spec_path = write_settings_and_spec_set(corpus)
    assert add_batch.spec_set_path_for_corpus(corpus) == spec_path
    assert add_batch.main(["--corpus-folder", str(corpus), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Scenario seed: 20280001" in output and str(spec_path) in output
    assert add_batch.sha256_of_file(spec_path) in output


def test_default_corpus_folder_is_the_spec_set_corpus():
    assert add_batch.DEFAULT_CORPUS_FOLDER.name == "emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs"


def test_the_two_corpora_never_share_a_seed():
    old = {add_batch.scenario_seed_for_batch(n) for n in range(1, 1000)}
    new = {add_batch.scenario_seed_for_batch(n, 20280000) for n in range(1, 1000)}
    assert not old & new


def test_full_batch_with_a_spec_set_records_its_hash_and_snapshots_it(tmp_path):
    corpus = make_corpus(tmp_path)
    spec_path = write_settings_and_spec_set(corpus)
    batch_folder = corpus / "batches" / "batch_001_seed_20280001_2026-11-06_10-00"
    batch_folder.mkdir()
    fake, calls = fake_pipeline_and_judges(batch_folder)
    entry = add_batch.run_batch(corpus, batch_folder, 1, 20280001, 5, datetime(2026, 11, 6, 10, 0),
                                run_command_function=fake)
    snapshot_folder = batch_folder / "generation_and_judge_prompts_used_for_this_batch"
    assert entry["spec_set_sha256"] == add_batch.sha256_of_file(spec_path)
    assert json.loads(next(batch_folder.glob("batch_summary_*.json")).read_text())["spec_set_sha256"] == entry["spec_set_sha256"]
    assert (snapshot_folder / spec_path.name).read_bytes() == spec_path.read_bytes()
    # the spec set is part of the generation prompt version hash
    assert entry["generation_prompt_version_hash"] == add_batch.generation_prompt_version_hash(
        snapshot_folder, snapshot_folder, snapshot_folder, snapshot_folder / spec_path.name)
    assert entry["generation_prompt_version_hash"] != add_batch.generation_prompt_version_hash()
    config_text = next(batch_folder.glob("pipeline_config_for_this_batch_*.yaml")).read_text()
    assert str(spec_path) in config_text and "spec_dealing_plan_for_batch_001_" in config_text
    plan = json.loads(next(batch_folder.glob("spec_dealing_plan_for_batch_001_*.json")).read_text())
    assert plan["dealt_spec_ids"] == [] and plan["forced_spec_ids"] is None
    assert entry["batch_mode"] == "default_never_dealt_specs"


def test_accepted_records_keep_the_spec_id_of_the_selected_prompt():
    prompts = [{"prompt_gid": "P-1", "spec_id": "spec_00042", "user_message": "m"}]
    accepted, _ = add_batch.build_accepted_records(
        prompts, {"P-1": {"complete_answer_without_welfare_possible": "yes", "note": "a"}},
        {"P-1": {"useful": True, "note": "u", "best_answer_ignoring_welfare": "b"}}, 1, 20280001, "h")
    assert accepted[0]["spec_id"] == "spec_00042"
