"""Offline tests for contributor shares, reserved specs, the merge of contributor corpora and the contributor kit.
No model call is made (the scenario dealer is run offline; the Claude Code login check is a stub)."""

import json
import random
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
sys.path.insert(0, str(SCRIPT_DIRECTORY / "contributor_kit"))
import add_batch_to_emergent_misalignment_prompt_corpus as add_batch
import build_emergent_misalignment_contributor_kit as kit_builder
import combine_emergent_misalignment_prompt_corpus_batches as combine
import reserved_spec_ids_for_contributor_shares as reserved
import run_my_shares
import split_remaining_specs_into_contributor_shares as splitter
from load_pipeline_scenario_dealer import load_scenario_dealer
from test_spec_outcomes_and_retry_planning_for_prompt_corpus import write_batch

compose_scenarios = load_scenario_dealer()
NUMBER_OF_TEST_SHARES = 4


def make_every_subcategory_spec_set(specs_per_subcategory=16) -> list[dict]:
    """A spec set covering every weighted category and every subcategory of the real species table."""
    rows = []
    values, weights = compose_scenarios.load_axes()
    for category in [c for c, w in zip(values["taxa_category"], weights["taxa_category"]) if w > 0]:
        for subcategory in compose_scenarios.taxa_for(category)["subcategories"]:
            for position in range(specs_per_subcategory):
                rows.append({"spec_id": f"spec_{len(rows) + 1:05d}", "taxa_category": category,
                             "taxa_subcategory": subcategory, "harm_type": f"harm {position}",
                             "practical_advantage": f"advantage {position % 5}",
                             "position_in_subcategory_order": position + 1})
    return rows


def write_spec_set(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def make_main_corpus(tmp_path, spec_rows) -> Path:
    corpus = tmp_path / "main_corpus"
    (corpus / "batches").mkdir(parents=True)
    write_spec_set(corpus / "fixed_spec_set" / "spec_set.jsonl", spec_rows)
    (corpus / add_batch.CORPUS_SETTINGS_FILE_NAME).write_text(
        json.dumps({"scenario_seed_base": 20280000, "spec_set_file": "fixed_spec_set/spec_set.jsonl"}))
    return corpus


@pytest.fixture(scope="module")
def spec_rows():
    return make_every_subcategory_spec_set()


@pytest.fixture(scope="module")
def built_shares(spec_rows, tmp_path_factory):
    """Main corpus with a few specs dealt (including failures), then the shares and their safe deal counts."""
    tmp_path = tmp_path_factory.mktemp("shares")
    corpus = make_main_corpus(tmp_path, spec_rows)
    ids = [row["spec_id"] for row in spec_rows]
    write_batch(corpus, 1, [(ids[0], "accepted"), (ids[1], "not_useful"), (ids[2], "gate_fail")], spec_rows)
    write_batch(corpus, 2, [(ids[3], "accepted"), (ids[4], "unjudged")], spec_rows, in_ledger=False)  # no ledger line
    dealt = splitter.find_every_spec_ever_dealt(corpus, spec_rows)
    remaining = [row for row in spec_rows if row["spec_id"] not in dealt]
    trial_folder = tmp_path / "trial"
    trial_folder.mkdir()
    results = splitter.build_shares_with_safe_deal_counts(remaining, NUMBER_OF_TEST_SHARES, trial_folder, report=lambda m: None)
    return {"corpus": corpus, "ids": ids, "dealt": dealt, "remaining": remaining, "results": results, "tmp_path": tmp_path}


def share_spec_file(tmp_path: Path, result: dict) -> Path:
    return write_spec_set(tmp_path / f"share_{result['share_number']}.jsonl", result["rows"])


# ---------- splitting ----------

def test_every_spec_ever_dealt_includes_failed_unledgered_and_forced(built_shares):
    ids = built_shares["ids"]
    assert built_shares["dealt"] == set(ids[:5])


def test_shares_are_disjoint_exclude_dealt_and_their_union_is_the_remainder(built_shares):
    share_ids = [{row["spec_id"] for row in result["rows"]} for result in built_shares["results"]]
    assert sum(len(ids) for ids in share_ids) == len(set().union(*share_ids))  # disjoint
    assert not set().union(*share_ids) & built_shares["dealt"]
    assert set().union(*share_ids) == {row["spec_id"] for row in built_shares["remaining"]}
    sizes = [len(ids) for ids in share_ids]
    assert max(sizes) - min(sizes) <= 1


def test_each_share_keeps_the_subcategory_balance_and_the_stored_order(built_shares):
    from collections import Counter
    per_share = [Counter((row["taxa_category"], row["taxa_subcategory"]) for row in result["rows"])
                 for result in built_shares["results"]]
    for key in set().union(*per_share):
        counts = [counter[key] for counter in per_share]
        assert max(counts) - min(counts) <= 1, key
    file_order = {row["spec_id"]: index for index, row in enumerate(built_shares["remaining"])}
    for result in built_shares["results"]:
        indices = [file_order[row["spec_id"]] for row in result["rows"]]
        assert indices == sorted(indices)


def test_share_seed_bases_are_distinct_and_never_meet_the_main_corpus_seeds():
    main_seeds = {add_batch.scenario_seed_for_batch(number, 20280000) for number in range(1, 1000)}
    share_seeds = {splitter.first_batch_scenario_seed_for_share(number) for number in range(1, 200)}
    assert len(share_seeds) == 199 and not main_seeds & share_seeds


def test_safe_deal_count_never_exhausts_and_deals_only_the_shares_specs(built_shares):
    for result in built_shares["results"]:
        path = share_spec_file(built_shares["tmp_path"], result)
        dealt = splitter.simulate_deal_of_specs(path, result["safe_deal_count"], result["first_batch_scenario_seed"])
        assert len(dealt) == len(set(dealt)) == result["safe_deal_count"]
        assert set(dealt) <= {row["spec_id"] for row in result["rows"]}
        assert dealt == result["spec_ids_dealt_at_safe_deal_count"]


def test_the_search_finds_a_count_below_the_share_size_when_the_whole_share_exhausts(spec_rows, tmp_path):
    # a share with only two specs of one category cannot be dealt that category more than twice
    first_category = spec_rows[0]["taxa_category"]
    rows = ([row for row in spec_rows if row["taxa_category"] == first_category][:2]
            + [row for row in spec_rows if row["taxa_category"] != first_category])[:300]
    path = write_spec_set(tmp_path / "lopsided.jsonl", rows)
    with pytest.raises(compose_scenarios.SpecPoolExhausted):
        splitter.simulate_deal_of_specs(path, len(rows), 20281001)
    count, exhausted = splitter.find_largest_safe_deal_count(path, len(rows), 20281001)
    assert exhausted is not None and count < len(rows)
    splitter.simulate_deal_of_specs(path, count, 20281001)


def test_two_shares_never_deal_the_same_spec(built_shares):
    dealt_by_share = [set(result["spec_ids_dealt_at_safe_deal_count"]) for result in built_shares["results"]]
    for first in range(len(dealt_by_share)):
        for second in range(first + 1, len(dealt_by_share)):
            assert not dealt_by_share[first] & dealt_by_share[second]


# ---------- reserved specs in the main corpus ----------

def test_main_default_mode_never_deals_reserved_specs(built_shares, tmp_path):
    corpus = make_main_corpus(tmp_path, built_shares["remaining"] + [row for row in built_shares["results"][0]["rows"][:0]])
    reserved_ids = {row["spec_id"] for result in built_shares["results"][:3] for row in result["rows"]}
    open_ids = {row["spec_id"] for row in built_shares["results"][3]["rows"]}
    reserved.write_reserved_spec_ids_file(corpus, "x", "y", {number: [row["spec_id"] for row in result["rows"]]
                                                               for number, result in enumerate(built_shares["results"][:3], 1)},
                                          "2026-11-08_10-00")
    plan = add_batch.plan_spec_dealing(corpus, 1, 50)
    assert set(plan["reserved_spec_ids"]) == reserved_ids and reserved_ids <= set(plan["dealt_spec_ids"])
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    dealt = set()
    for seed in range(4):
        scenarios = compose_scenarios.deal_scenarios(
            40, random.Random(seed), spec_set_path=corpus / "fixed_spec_set" / "spec_set.jsonl",
            spec_dealing_plan_path=plan_path)
        dealt |= {scenario["spec_id"] for scenario in scenarios}
    assert dealt and dealt <= open_ids and not dealt & reserved_ids


def test_without_a_reserved_file_nothing_changes_and_the_newest_file_wins(built_shares, tmp_path):
    corpus = make_main_corpus(tmp_path, built_shares["remaining"])
    assert reserved.read_reserved_spec_ids(corpus) == set()
    assert add_batch.plan_spec_dealing(corpus, 1, 5)["reserved_spec_ids"] == []
    reserved.write_reserved_spec_ids_file(corpus, "x", "y", {1: ["spec_a"], 2: ["spec_b"]}, "2026-11-08_10-00")
    reserved.write_reserved_spec_ids_file(corpus, "x", "y", {2: ["spec_b"]}, "2026-11-08_11-00")
    assert reserved.read_reserved_spec_ids(corpus) == {"spec_b"}


def test_a_dry_run_names_the_reserved_specs(built_shares, tmp_path, capsys):
    corpus = make_main_corpus(tmp_path, built_shares["remaining"])
    ids = [row["spec_id"] for row in built_shares["results"][0]["rows"]]
    reserved.write_reserved_spec_ids_file(corpus, "x", "y", {1: ids}, "2026-11-08_10-00")
    assert add_batch.main(["--corpus-folder", str(corpus), "--deals", "5", "--dry-run"]) == 0
    assert f"specs reserved for contributor shares (never dealt here): {len(ids)}" in capsys.readouterr().out


# ---------- usage limit ----------

def test_usage_limit_is_recognised_from_the_exit_code_or_the_output(tmp_path):
    output = tmp_path / "out.txt"
    output.write_text("Traceback ...\nUsageLimitExceeded: Claude subscription usage limit reached: resets 5pm")
    assert add_batch.step_stopped_for_usage_limit(1, output)
    assert add_batch.step_stopped_for_usage_limit(3, tmp_path / "missing.txt")
    output.write_text("Traceback ...\nKeyError: something else")
    assert not add_batch.step_stopped_for_usage_limit(1, output)


def test_a_batch_stopped_by_the_usage_limit_exits_with_code_three(tmp_path, monkeypatch, capsys):
    def stopped(*arguments, **keywords):
        raise add_batch.BatchStoppedForUsageLimit("usage limit")
    monkeypatch.setattr(add_batch, "run_batch", stopped)
    assert add_batch.main(["--corpus-folder", str(tmp_path / "corpus"), "--deals", "1"]) == add_batch.USAGE_LIMIT_EXIT_CODE
    assert "STOPPED FOR USAGE LIMIT" in capsys.readouterr().err


# ---------- merging contributor corpora ----------

def make_manifest(tmp_path, results) -> dict:
    shares = []
    for result in results:
        path = share_spec_file(tmp_path, result)
        shares.append({"share_number": result["share_number"], "share_file": path.name,
                       "share_file_sha256": splitter.sha256_of_file(path), "spec_count": len(result["rows"]),
                       "scenario_seed_base": result["scenario_seed_base"], "safe_deal_count": result["safe_deal_count"],
                       "spec_ids": [row["spec_id"] for row in result["rows"]]})
    return {"shares": shares}


def make_contributor_corpus(tmp_path, result, deal_spec_ids, folder_name=None, batch_mode="default_never_dealt_specs",
                            lookup_rows=None):
    corpus = tmp_path / (folder_name or f"contributor_corpus_share_{result['share_number']:03d}")
    (corpus / "batches").mkdir(parents=True)
    spec_file = write_spec_set(corpus / "fixed_spec_set" / f"share_{result['share_number']}.jsonl", result["rows"])
    (corpus / add_batch.CORPUS_SETTINGS_FILE_NAME).write_text(json.dumps(
        {"scenario_seed_base": result["scenario_seed_base"], "spec_set_file": f"fixed_spec_set/{spec_file.name}"}))
    rows = lookup_rows or result["rows"]
    folder = write_batch(corpus, 1, [(spec_id, "accepted") for spec_id in deal_spec_ids], rows, in_ledger=False)
    accepted_name = "accepted_prompts_batch_001_2026-11-08_10-00.jsonl"
    records = [{"user_message": f"message of {spec_id} in share {result['share_number']}", "spec_id": spec_id,
                "corpus_prompt_id": f"B001-P-{index:04d}", "taxa_category": "x", "written_in_english": True,
                "generation_prompt_version_hash": "hash"} for index, spec_id in enumerate(deal_spec_ids, 1)]
    (folder / accepted_name).write_text("".join(json.dumps(r) + "\n" for r in records))
    add_batch.append_ledger_line(corpus / add_batch.LEDGER_FILE_NAME, {
        "batch_number": 1, "batch_folder": str(folder.relative_to(corpus)), "accepted_file": accepted_name,
        "generation_prompt_version_hash": "hash", "batch_mode": batch_mode, "accepted_count": len(records)})
    return corpus


def main_corpus_with_accepted_batch(tmp_path, spec_rows, held_out_prompt_id=None):
    corpus = make_main_corpus(tmp_path, spec_rows)
    ids = [spec_rows[0]["spec_id"], spec_rows[1]["spec_id"]]
    folder = write_batch(corpus, 1, [(spec_id, "accepted") for spec_id in ids], spec_rows, in_ledger=False)
    accepted_name = "accepted_prompts_batch_001_2026-11-08_10-00.jsonl"
    (folder / accepted_name).write_text("".join(json.dumps({
        "user_message": f"main message {index}", "spec_id": spec_id, "corpus_prompt_id": f"B001-P-{index:04d}",
        "taxa_category": "x", "written_in_english": True, "generation_prompt_version_hash": "hash"}) + "\n"
        for index, spec_id in enumerate(ids, 1)))
    add_batch.append_ledger_line(corpus / add_batch.LEDGER_FILE_NAME, {
        "batch_number": 1, "batch_folder": str(folder.relative_to(corpus)), "accepted_file": accepted_name,
        "generation_prompt_version_hash": "hash", "accepted_count": 2})
    if held_out_prompt_id:
        held_out = corpus / "held_out_set"
        held_out.mkdir()
        (held_out / "held_out_set_1_prompts_2026-11-08_10-00.jsonl").write_text(json.dumps(
            {"corpus_prompt_id": held_out_prompt_id, "user_message": "held out message", "spec_id": "spec_held_out"}) + "\n")
    return corpus


def test_merge_qualifies_ids_and_matches_held_out_prompt_ids_only_in_the_main_corpus(built_shares, spec_rows, tmp_path):
    main_corpus = main_corpus_with_accepted_batch(tmp_path / "m", spec_rows, held_out_prompt_id="B001-P-0001")
    results = built_shares["results"]
    manifest = make_manifest(tmp_path / "s", results)
    first = make_contributor_corpus(tmp_path / "c", results[0], [results[0]["rows"][0]["spec_id"], results[0]["rows"][1]["spec_id"]])
    second = make_contributor_corpus(tmp_path / "c", results[1], [results[1]["rows"][0]["spec_id"]])
    combined, summary = combine.combine_batches(main_corpus, None, False, [first.parent], manifest)
    ids = [record["merged_corpus_prompt_id"] for record in combined]
    assert len(ids) == len(set(ids)) == 4  # the main B001-P-0001 is held out; the contributors' B001-P-0001 is not
    assert "B001-P-0002" in ids and "S001-B001-P-0001" in ids and "S002-B001-P-0001" in ids
    assert summary["held_out_prompts_excluded"] == 1
    assert summary["corpus_sources"] == ["main_corpus", "contributor_share_001", "contributor_share_002"]
    assert {record["corpus_source"] for record in combined} == {"main_corpus", "contributor_share_001", "contributor_share_002"}


def test_merge_without_contributors_is_unchanged(spec_rows, tmp_path):
    combined, summary = combine.combine_batches(main_corpus_with_accepted_batch(tmp_path, spec_rows))
    assert [r["corpus_prompt_id"] for r in combined] == ["B001-P-0001", "B001-P-0002"]
    assert all("merged_corpus_prompt_id" not in record for record in combined)


def test_merge_refuses_a_spec_dealt_in_both_the_main_corpus_and_a_share(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    main_corpus = main_corpus_with_accepted_batch(tmp_path / "m", spec_rows)
    main_spec_id = spec_rows[0]["spec_id"]
    stolen = dict(results[0], rows=results[0]["rows"] + [spec_rows[0]])
    manifest = make_manifest(tmp_path / "s", [stolen])
    contributor = make_contributor_corpus(tmp_path / "c", stolen, [main_spec_id])
    with pytest.raises(combine.ContributorMergeRefused, match="dealt in both"):
        combine.combine_batches(main_corpus, None, False, [contributor], manifest)


def test_merge_refuses_a_spec_dealt_twice_in_two_contributor_folders(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    shared_row = results[0]["rows"][0]
    overlapping = dict(results[1], rows=results[1]["rows"] + [shared_row])
    manifest = make_manifest(tmp_path / "s", [results[0], overlapping])
    first = make_contributor_corpus(tmp_path / "c", results[0], [shared_row["spec_id"]])
    second = make_contributor_corpus(tmp_path / "c", overlapping, [shared_row["spec_id"]])
    with pytest.raises(combine.ContributorMergeRefused, match="dealt in both"):
        combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m", spec_rows), None, False, [first, second], manifest)


def test_merge_refuses_a_spec_outside_the_contributors_share(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    manifest = make_manifest(tmp_path / "s", results[:2])
    outsider = results[1]["rows"][0]["spec_id"]
    contributor = make_contributor_corpus(tmp_path / "c", results[0], [outsider], lookup_rows=built_shares["remaining"])
    # the folder's own spec set is share 1, so the batch outcome names a spec the share does not hold
    with pytest.raises(combine.ContributorMergeRefused, match="outside share"):
        combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m", spec_rows), None, False, [contributor], manifest)


def test_merge_refuses_a_spec_set_that_is_not_a_manifest_share_and_a_share_given_twice(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    manifest = make_manifest(tmp_path / "s", results[:2])
    tampered = dict(results[0], rows=results[0]["rows"][:-1])
    contributor = make_contributor_corpus(tmp_path / "c", tampered, [tampered["rows"][0]["spec_id"]])
    with pytest.raises(combine.ContributorMergeRefused, match="not one of the manifest's shares"):
        combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m", spec_rows), None, False, [contributor], manifest)
    one = make_contributor_corpus(tmp_path / "d", results[0], [results[0]["rows"][0]["spec_id"]])
    other = make_contributor_corpus(tmp_path / "d", results[0], [results[0]["rows"][0]["spec_id"]], folder_name="copy_of_share_1")
    with pytest.raises(combine.ContributorMergeRefused, match="appears in two contributor folders"):
        combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m2", spec_rows), None, False, [one, other], manifest)


def test_main_command_refuses_with_a_message_and_needs_both_arguments(built_shares, spec_rows, tmp_path, capsys):
    results = built_shares["results"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(make_manifest(tmp_path / "s", results[:1])))
    main_corpus = main_corpus_with_accepted_batch(tmp_path / "m", spec_rows)
    contributor = make_contributor_corpus(tmp_path / "c", results[0], [results[1]["rows"][0]["spec_id"]],
                                          lookup_rows=built_shares["remaining"])
    assert combine.main(["--corpus-folder", str(main_corpus), "--contributor-corpus-folders", str(contributor),
                         "--contributor-share-manifest", str(manifest_path)]) == 1
    assert "REFUSING TO RUN" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        combine.main(["--corpus-folder", str(main_corpus), "--contributor-corpus-folders", str(contributor)])


# ---------- the kit ----------

@pytest.fixture(scope="module")
def built_kit(built_shares, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("kit")
    shares_folder = tmp_path / "shares_folder"
    shares_folder.mkdir()
    shares = []
    for result in built_shares["results"]:
        name = splitter.share_file_name(result["share_number"], NUMBER_OF_TEST_SHARES, len(result["rows"]), "2026-11-08_10-00")
        write_spec_set(shares_folder / name, result["rows"])
        shares.append({"share_number": result["share_number"], "share_file": name,
                       "share_file_sha256": splitter.sha256_of_file(shares_folder / name), "spec_count": len(result["rows"]),
                       "scenario_seed_base": result["scenario_seed_base"], "first_batch_scenario_seed": result["first_batch_scenario_seed"],
                       "safe_deal_count": result["safe_deal_count"], "spec_ids": [r["spec_id"] for r in result["rows"]]})
    manifest = {"number_of_shares": NUMBER_OF_TEST_SHARES, "shares": shares}
    (shares_folder / "contributor_share_manifest_2026-11-08_10-00.json").write_text(json.dumps(manifest))
    kit_folder = tmp_path / "kit"
    kit_builder.empty_kit_folder_keeping_git(kit_folder)
    kit_builder.build_kit_contents(shares_folder, kit_folder)
    return {"kit_folder": kit_folder, "manifest": manifest}


def test_kit_is_clean_has_only_share_files_from_data_and_no_absolute_user_paths(built_kit):
    kit_folder = built_kit["kit_folder"]
    assert kit_builder.check_kit_folder_is_clean(kit_folder, built_kit["manifest"]) == []
    for path in kit_folder.rglob("*"):
        if path.is_file() and path.suffix in kit_builder.TEXT_SUFFIXES:
            assert "/Users/" not in path.read_text(encoding="utf-8", errors="replace"), path
    names = {path.name for path in kit_folder.rglob("*") if path.is_file()}
    assert not any(name.startswith(".env") or "accepted_prompts" in name or "held_out" in name for name in names)
    assert {p.name for p in (kit_folder / "contributor_shares").iterdir()} == \
        {share["share_file"] for share in built_kit["manifest"]["shares"]} | {"contributor_share_manifest_2026-11-08_10-00.json"}
    for required in ("README.md", "CLAUDE.md", "requirements.txt", "run_my_shares.py", ".gitignore"):
        assert (kit_folder / required).exists()
    assert not (kit_folder / "strictly_implicit_difficult_advice_pipeline_copy" / "outputs").exists()
    assert not (kit_folder / "strictly_implicit_difficult_advice_pipeline_copy" / "tests").exists()


def test_clean_check_catches_a_personal_path_a_key_file_and_a_stray_data_file(built_kit):
    kit_folder = built_kit["kit_folder"]
    (kit_folder / "scripts" / "note.txt").write_text("see /Users/someone/Desktop")
    (kit_folder / ".env").write_text("X=1")
    (kit_folder / "contributor_shares" / "accepted_prompts_batch_001.jsonl").write_text("{}")
    problems = " ".join(kit_builder.check_kit_folder_is_clean(kit_folder, built_kit["manifest"]))
    assert "/Users/" in problems and ".env" in problems and "unexpected file among the shares" in problems
    (kit_folder / "scripts" / "note.txt").unlink()
    (kit_folder / ".env").unlink()
    (kit_folder / "contributor_shares" / "accepted_prompts_batch_001.jsonl").unlink()


def test_builder_refuses_to_empty_a_folder_that_is_not_a_kit(tmp_path):
    (tmp_path / "precious.txt").write_text("keep")
    with pytest.raises(SystemExit):
        kit_builder.empty_kit_folder_keeping_git(tmp_path)
    assert (tmp_path / "precious.txt").exists()


def test_kit_runs_a_dry_run_from_inside_itself(built_kit):
    kit_folder = built_kit["kit_folder"]
    completed = subprocess.run([sys.executable, "run_my_shares.py", "--shares", "2", "--dry-run"], cwd=kit_folder,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    share = built_kit["manifest"]["shares"][1]
    assert f"Deals requested: {share['safe_deal_count']}" in completed.stdout
    assert not (kit_folder / "my_share_corpora").exists()


# ---------- the entry point ----------

def test_entry_point_prepares_a_share_corpus_resumes_skips_and_zips(built_kit, tmp_path, monkeypatch):
    kit_folder = built_kit["kit_folder"]
    monkeypatch.setattr(run_my_shares, "SHARE_FOLDER", kit_folder / "contributor_shares")
    manifest = run_my_shares.load_newest_manifest(kit_folder / "contributor_shares")
    share = run_my_shares.find_share(manifest, 3)
    corpus_folder = run_my_shares.corpus_folder_for_share(3, tmp_path)
    assert corpus_folder.name == "contributor_corpus_share_003"
    run_my_shares.prepare_share_corpus_folder(share, corpus_folder, kit_folder / "contributor_shares")
    settings = json.loads((corpus_folder / run_my_shares.CORPUS_SETTINGS_FILE_NAME).read_text())
    assert settings["scenario_seed_base"] == share["scenario_seed_base"]
    assert (corpus_folder / settings["spec_set_file"]).exists()
    assert run_my_shares.state_of_share_corpus(corpus_folder) == (run_my_shares.STATE_NOT_STARTED, None)
    assert "--resume-batch-folder" not in run_my_shares.batch_command(corpus_folder, share, None)
    unfinished = corpus_folder / "batches" / "batch_001_seed_20283001_2026-11-08_10-00"
    unfinished.mkdir(parents=True)
    assert run_my_shares.state_of_share_corpus(corpus_folder) == (run_my_shares.STATE_UNFINISHED_BATCH, unfinished)
    command = run_my_shares.batch_command(corpus_folder, share, unfinished)
    assert command[command.index("--resume-batch-folder") + 1] == str(unfinished)
    assert command[command.index("--deals") + 1] == str(share["safe_deal_count"])
    add_batch.append_ledger_line(corpus_folder / run_my_shares.LEDGER_FILE_NAME,
                                 {"batch_number": 1, "batch_folder": f"batches/{unfinished.name}"})
    assert run_my_shares.state_of_share_corpus(corpus_folder)[0] == run_my_shares.STATE_COMPLETE
    zip_path = run_my_shares.write_zip_of_share_corpora([3], tmp_path, tmp_path / "zips",
                                                        now=lambda: datetime(2026, 11, 8, 15, 35))
    assert zip_path.name == "emergent_misalignment_contributor_corpus_shares_003_2026-11-08_15-35.zip"
    with zipfile.ZipFile(zip_path) as archive:
        assert any(name.startswith("contributor_corpus_share_003/") and name.endswith(run_my_shares.LEDGER_FILE_NAME)
                   for name in archive.namelist())
    # a second pass's folder is named as one, so zipping it adds no second prefix
    second_pass_folder = tmp_path / "second_pass_contributor_corpus_share_003"
    shutil.copytree(corpus_folder, second_pass_folder)
    assert run_my_shares.corpus_folder_for_share(3, tmp_path, "second_pass_") == second_pass_folder
    second_pass_zip_path = run_my_shares.write_zip_of_share_corpora([3], tmp_path, tmp_path / "second_pass_zips",
                                                                    now=lambda: datetime(2026, 11, 8, 15, 35),
                                                                    label="second_pass_")
    assert second_pass_zip_path.name == "emergent_misalignment_contributor_corpus_second_pass_shares_003_2026-11-08_15-35.zip"
    with zipfile.ZipFile(second_pass_zip_path) as archive:
        names = archive.namelist()
        assert names and all(name.startswith("second_pass_contributor_corpus_share_003/") for name in names)
        assert not any("second_pass_second_pass_" in name for name in names)


def test_entry_point_refuses_a_changed_share_file_and_a_wrong_share_number(built_kit, tmp_path, monkeypatch):
    kit_folder = built_kit["kit_folder"]
    monkeypatch.setattr(run_my_shares, "SHARE_FOLDER", kit_folder / "contributor_shares")
    manifest = run_my_shares.load_newest_manifest(kit_folder / "contributor_shares")
    with pytest.raises(SystemExit, match="no share 99|There is no share 99"):
        run_my_shares.find_share(manifest, 99)
    share = dict(run_my_shares.find_share(manifest, 1), share_file_sha256="0" * 64)
    with pytest.raises(SystemExit, match="checksum"):
        run_my_shares.verified_share_file(share, kit_folder / "contributor_shares")
    assert run_my_shares.parse_share_numbers("7, 12") == [7, 12]
    with pytest.raises(SystemExit):
        run_my_shares.parse_share_numbers("7,7")


def test_login_check_messages_without_a_model_call():
    class Completed:
        def __init__(self, stdout): self.stdout = stdout

    def stub(status):
        return lambda *a, **k: Completed(json.dumps(status))
    assert "logged in" in run_my_shares.check_claude_login(False, stub({"loggedIn": True, "authMethod": "claude.ai"}))
    with pytest.raises(SystemExit, match="not logged in"):
        run_my_shares.check_claude_login(False, stub({"loggedIn": False}))
    with pytest.raises(SystemExit, match="not a Claude subscription"):
        run_my_shares.check_claude_login(False, stub({"loggedIn": True, "authMethod": "api_key"}))
    assert "logged in" in run_my_shares.check_claude_login(True, stub({"loggedIn": True, "authMethod": "api_key"}))
    assert "could not check" in run_my_shares.check_claude_login(False, lambda *a, **k: Completed("not json"))


def test_a_second_run_on_the_same_share_is_refused_by_the_lock(tmp_path):
    lock = run_my_shares.take_lock(tmp_path)
    with pytest.raises(SystemExit, match="already working"):
        run_my_shares.take_lock(tmp_path)
    lock.unlink()
    run_my_shares.take_lock(tmp_path).unlink()


# ---------- second pass ----------

def full_manifest_for_seed_tests() -> dict:
    return {"shares": [{"share_number": number, "scenario_seed_base": 20280000 + number * 1000,
                        "first_batch_scenario_seed": 20280000 + number * 1000 + 1} for number in range(1, 21)]}


def second_pass_arguments(**overrides):
    import argparse
    values = {"second_pass": False, "scenario_seed_base": None, "output_folder": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_second_pass_seeds_never_collide_with_any_first_pass_or_main_seed():
    manifest = full_manifest_for_seed_tests()
    first_pass_seeds = {share["scenario_seed_base"] for share in manifest["shares"]} | {
        share["first_batch_scenario_seed"] for share in manifest["shares"]} | set(run_my_shares.MAIN_CORPUS_SEED_RANGE)
    second_pass_bases = [run_my_shares.second_pass_seed_base(share) for share in manifest["shares"]]
    second_pass_seeds = second_pass_bases + [base + 1 for base in second_pass_bases]
    assert len(set(second_pass_seeds)) == len(second_pass_seeds)
    assert not set(second_pass_seeds) & first_pass_seeds
    for refused in (20281000, 20280000, 20280500, 20280998, 20281500):
        with pytest.raises(SystemExit):
            run_my_shares.refuse_seed_base_that_could_repeat_a_seed(refused, manifest)
    run_my_shares.refuse_seed_base_that_could_repeat_a_seed(20287700, manifest)
    for base in second_pass_bases:
        run_my_shares.refuse_seed_base_that_could_repeat_a_seed(base, manifest, is_standard_second_pass=True)


def test_second_pass_deal_never_exhausts_and_deals_only_the_shares_specs(built_shares, tmp_path, monkeypatch):
    monkeypatch.setattr(run_my_shares, "SCRIPTS_FOLDER", Path(__file__).resolve().parent)
    for result in built_shares["results"]:
        share_path = share_spec_file(tmp_path, result)
        seed_base = result["scenario_seed_base"] + run_my_shares.SECOND_PASS_SEED_BASE_OFFSET
        batch_seed = seed_base + 1
        count, _ = run_my_shares.find_largest_safe_deal_count(share_path, len(result["rows"]), batch_seed)
        dealt = run_my_shares.simulate_deal_of_specs(share_path, count, batch_seed)  # raises when a category exhausts
        assert len(dealt) == count and len(set(dealt)) == count
        assert set(dealt) <= {row["spec_id"] for row in result["rows"]}
        assert dealt == splitter.simulate_deal_of_specs(share_path, count, batch_seed)  # same dealer as the splitter


def test_second_pass_choice_of_seeds_output_folder_and_refusals(built_kit, monkeypatch, tmp_path):
    kit_folder = built_kit["kit_folder"]
    monkeypatch.setattr(run_my_shares, "SHARE_FOLDER", kit_folder / "contributor_shares")
    monkeypatch.setattr(run_my_shares, "SCRIPTS_FOLDER", Path(__file__).resolve().parent)
    manifest = run_my_shares.load_newest_manifest(kit_folder / "contributor_shares")
    shares = [run_my_shares.find_share(manifest, number) for number in (1, 2)]
    dealt_shares, output_folder, label = run_my_shares.choose_seed_bases_and_output_folder(
        second_pass_arguments(second_pass=True), shares, manifest)
    assert [share["scenario_seed_base"] for share in dealt_shares] == [share["scenario_seed_base"] + 500 for share in shares]
    assert output_folder == run_my_shares.DEFAULT_SECOND_PASS_OUTPUT_FOLDER.resolve() and label == "second_pass_"
    assert all(share["is_second_pass"] and 0 < share["safe_deal_count"] <= share["spec_count"] for share in dealt_shares)
    # a first pass is unchanged
    assert run_my_shares.choose_seed_bases_and_output_folder(second_pass_arguments(), shares, manifest) == (
        shares, run_my_shares.DEFAULT_OUTPUT_FOLDER.resolve(), "")
    refused = [second_pass_arguments(second_pass=True, output_folder=run_my_shares.DEFAULT_OUTPUT_FOLDER),
               second_pass_arguments(second_pass=True, scenario_seed_base=20287700, output_folder=tmp_path),
               second_pass_arguments(scenario_seed_base=20287700, output_folder=tmp_path),  # two shares
               second_pass_arguments(scenario_seed_base=20287700),  # default folder
               second_pass_arguments(scenario_seed_base=20287700, output_folder=run_my_shares.DEFAULT_OUTPUT_FOLDER)]
    for arguments in refused:
        with pytest.raises(SystemExit):
            run_my_shares.choose_seed_bases_and_output_folder(arguments, shares, manifest)
    dealt_one, folder, _ = run_my_shares.choose_seed_bases_and_output_folder(
        second_pass_arguments(scenario_seed_base=20287700, output_folder=tmp_path), shares[:1], manifest)
    assert dealt_one[0]["scenario_seed_base"] == 20287700 and folder == tmp_path.resolve()
    # the settings file of the second pass differs from the first pass's, so the two can never share a folder
    corpus_folder = tmp_path / "corpus"
    run_my_shares.prepare_share_corpus_folder(shares[0], corpus_folder, kit_folder / "contributor_shares")
    with pytest.raises(SystemExit):
        run_my_shares.prepare_share_corpus_folder(dealt_shares[0], corpus_folder, kit_folder / "contributor_shares")


# ---------- a share is never run twice with the same scenario seed ----------

@pytest.fixture
def kit_run_environment(built_kit, tmp_path, monkeypatch):
    """run_my_shares pointed at the built kit's shares, a scratch seed record and zip folder, and a recorded fake batch call."""
    share_folder = built_kit["kit_folder"] / "contributor_shares"
    original_verified_share_file = run_my_shares.verified_share_file
    original_prepare_share_corpus_folder = run_my_shares.prepare_share_corpus_folder
    monkeypatch.setattr(run_my_shares, "SHARE_FOLDER", share_folder)
    monkeypatch.setattr(run_my_shares, "load_newest_manifest", lambda: built_kit["manifest"])
    monkeypatch.setattr(run_my_shares, "verified_share_file", lambda share, folder=share_folder: original_verified_share_file(share, folder))
    monkeypatch.setattr(run_my_shares, "prepare_share_corpus_folder",
                        lambda share, corpus_folder, folder=share_folder: original_prepare_share_corpus_folder(share, corpus_folder, folder))
    monkeypatch.setattr(run_my_shares, "SCENARIO_SEEDS_RECORD_PATH", tmp_path / "scenario_seeds_record.json")
    monkeypatch.setattr(run_my_shares, "ZIP_FOLDER", tmp_path / "zips_to_send_back")
    monkeypatch.setattr(run_my_shares, "check_python_and_packages", lambda: None)
    monkeypatch.setattr(run_my_shares, "keep_computer_awake_while_this_run_lasts", lambda: "awake")
    batch_calls = []

    def fake_batch_call(command, cwd=None):
        batch_calls.append(command)
        corpus_folder = Path(command[command.index("--corpus-folder") + 1])
        if "--resume-batch-folder" in command:
            batch_folder = Path(command[command.index("--resume-batch-folder") + 1])
        else:
            batch_folder = corpus_folder / "batches" / "batch_001_seed_fake_2026-11-08_10-00"
            batch_folder.mkdir(parents=True)
        if not fake_batch_call.leave_unfinished:
            add_batch.append_ledger_line(corpus_folder / run_my_shares.LEDGER_FILE_NAME,
                                         {"batch_number": 1, "batch_folder": f"batches/{batch_folder.name}"})
        return 0

    fake_batch_call.leave_unfinished = False
    monkeypatch.setattr(run_my_shares.subprocess, "call", fake_batch_call)
    return {"batch_calls": batch_calls, "fake_batch_call": fake_batch_call, "output_folder": tmp_path / "corpora",
            "zip_folder": tmp_path / "zips_to_send_back", "tmp_path": tmp_path}


def run_main_for_share(environment, extra_arguments=()):
    return run_my_shares.main(["--shares", "3", "--skip-login-check", "--output-folder", str(environment["output_folder"]),
                               *extra_arguments])


def test_running_a_finished_share_again_runs_nothing_writes_no_zip_and_exits_with_code_four(kit_run_environment, capsys):
    environment = kit_run_environment
    assert run_main_for_share(environment) == 0
    assert len(environment["batch_calls"]) == 1
    zips_after_first_run = list(environment["zip_folder"].glob("*.zip"))
    assert len(zips_after_first_run) == 1
    capsys.readouterr()
    assert run_main_for_share(environment) == run_my_shares.ALREADY_RUN_EXIT_CODE == 4
    output = capsys.readouterr().out
    assert "ALL DONE" not in output
    assert "Share 003 has already been run once" in output and "--second-pass" in output
    assert "already in" in output and "zips_to_send_back" in output
    assert len(environment["batch_calls"]) == 1
    assert list(environment["zip_folder"].glob("*.zip")) == zips_after_first_run


def test_a_seed_already_started_is_refused_even_after_its_folder_is_moved_away(kit_run_environment):
    environment = kit_run_environment
    assert run_main_for_share(environment) == 0
    corpus_folder = environment["output_folder"] / "contributor_corpus_share_003"
    corpus_folder.rename(environment["tmp_path"] / "my_share_corpora_second_run_same_specs_same_seed")
    with pytest.raises(SystemExit, match="already started with scenario seed base") as refusal:
        run_main_for_share(environment)
    assert "duplicate" in str(refusal.value) and "--second-pass" in str(refusal.value)
    assert "delete" not in str(refusal.value).lower()
    assert len(environment["batch_calls"]) == 1
    assert not corpus_folder.exists()


def test_resuming_an_unfinished_batch_is_not_refused_and_records_the_seed_once(kit_run_environment):
    environment = kit_run_environment
    environment["fake_batch_call"].leave_unfinished = True
    assert run_main_for_share(environment) == 0
    assert len(run_my_shares.load_scenario_seed_records()) == 1
    environment["fake_batch_call"].leave_unfinished = False
    assert run_main_for_share(environment) == 0
    assert "--resume-batch-folder" in environment["batch_calls"][1]
    assert len(run_my_shares.load_scenario_seed_records()) == 1
    record = run_my_shares.load_scenario_seed_records()[0]
    assert record["share_number"] == 3 and record["pass_kind"] == "first pass"
    assert record["batch_scenario_seed"] == record["scenario_seed_base"] + 1


def test_the_second_pass_seed_is_recorded_separately_and_does_not_block_and_is_then_also_closed(kit_run_environment, capsys):
    environment = kit_run_environment
    assert run_main_for_share(environment) == 0
    second_pass_output_folder = environment["tmp_path"] / "second_pass_corpora"
    second_pass_arguments_for_main = ["--shares", "3", "--skip-login-check", "--second-pass",
                                      "--output-folder", str(second_pass_output_folder)]
    assert run_my_shares.main(second_pass_arguments_for_main) == 0
    assert len(environment["batch_calls"]) == 2
    records = run_my_shares.load_scenario_seed_records()
    assert [record["pass_kind"] for record in records] == ["first pass", "second pass"]
    assert records[0]["scenario_seed_base"] + run_my_shares.SECOND_PASS_SEED_BASE_OFFSET == records[1]["scenario_seed_base"]
    capsys.readouterr()
    assert run_my_shares.main(second_pass_arguments_for_main) == run_my_shares.ALREADY_RUN_EXIT_CODE
    assert "both passes" in capsys.readouterr().out


def test_corpus_folder_names_a_second_pass_as_one_and_a_first_pass_never_gets_a_prefix(tmp_path):
    assert run_my_shares.corpus_folder_for_share(7, tmp_path).name == "contributor_corpus_share_007"
    assert run_my_shares.corpus_folder_for_share(7, tmp_path, "second_pass_").name == "second_pass_contributor_corpus_share_007"
    # an old-named folder never stands in for a first pass's folder, and a prefixed folder never does either
    (tmp_path / "second_pass_contributor_corpus_share_007").mkdir()
    assert run_my_shares.corpus_folder_for_share(7, tmp_path).name == "contributor_corpus_share_007"
    assert run_my_shares.DEFAULT_OUTPUT_FOLDER != run_my_shares.DEFAULT_SECOND_PASS_OUTPUT_FOLDER


def test_a_second_pass_folder_from_an_earlier_kit_version_is_found_resumed_and_zipped_with_the_new_name(kit_run_environment):
    environment = kit_run_environment
    second_pass_output_folder = environment["tmp_path"] / "second_pass_corpora"
    arguments = ["--shares", "3", "--skip-login-check", "--second-pass", "--output-folder", str(second_pass_output_folder)]
    environment["fake_batch_call"].leave_unfinished = True
    assert run_my_shares.main(arguments) == 0
    # make it look as the earlier kit version left it: old name on disk, seed record pointing at the old name
    new_named_folder = second_pass_output_folder / "second_pass_contributor_corpus_share_003"
    old_named_folder = second_pass_output_folder / "contributor_corpus_share_003"
    new_named_folder.rename(old_named_folder)
    records = run_my_shares.load_scenario_seed_records()
    records[0]["corpus_folder"] = run_my_shares.relative_to_kit(old_named_folder)
    run_my_shares.SCENARIO_SEEDS_RECORD_PATH.write_text(json.dumps(records), encoding="utf-8")
    # resumes in the old-named folder, neither refused as a moved folder nor started anew
    environment["fake_batch_call"].leave_unfinished = False
    assert run_my_shares.main(arguments) == 0
    assert "--resume-batch-folder" in environment["batch_calls"][-1]
    assert old_named_folder.exists() and not new_named_folder.exists()
    assert len(run_my_shares.load_scenario_seed_records()) == 1
    zip_paths = list(environment["zip_folder"].glob("*second_pass_shares_003*.zip"))
    assert len(zip_paths) == 1
    with zipfile.ZipFile(zip_paths[0]) as archive:
        names = archive.namelist()
    assert names and all(name.startswith("second_pass_contributor_corpus_share_003/") for name in names)
    # a rerun detects it as complete
    assert run_my_shares.main(arguments) == run_my_shares.ALREADY_RUN_EXIT_CODE
