"""Offline tests (no model call): combining first and second passes of contributor shares, and the near-duplicate / held-out
similarity filter that turns a combined snapshot into the training prompt pool."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_batch_to_emergent_misalignment_prompt_corpus as add_batch
import combine_emergent_misalignment_prompt_corpus_batches as combine
import filter_near_duplicate_and_held_out_similar_prompts_from_combined_snapshot as pool_filter
from test_contributor_shares_reserved_specs_merge_and_kit import (  # noqa: F401  (fixtures are used by name)
    built_shares, main_corpus_with_accepted_batch, make_contributor_corpus, make_manifest, spec_rows)

SECOND_PASS_OFFSET = 500


def make_second_pass(tmp_path, result, spec_ids, seed_offset=SECOND_PASS_OFFSET, folder_name=None):
    """A second-pass corpus of the share: its seed base plus the offset, different messages, own folder name."""
    folder = make_contributor_corpus(
        tmp_path, dict(result, scenario_seed_base=result["scenario_seed_base"] + seed_offset), spec_ids,
        folder_name=folder_name or f"second_pass_contributor_corpus_share_{result['share_number']:03d}")
    for accepted in folder.glob("batches/*/accepted_prompts_*.jsonl"):
        accepted.write_text(accepted.read_text().replace("message of", "second pass message of"))
    return folder


def specs_of(result, count=2):
    return [row["spec_id"] for row in result["rows"][:count]]


def test_first_and_second_pass_of_one_share_combine_with_unique_ids_and_labels(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    manifest = make_manifest(tmp_path / "s", results)
    first = make_contributor_corpus(tmp_path / "c", results[0], specs_of(results[0]))
    second = make_second_pass(tmp_path / "c2", results[0], specs_of(results[0]))
    combined, summary = combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m", spec_rows), None, False,
                                                [second, first], manifest)
    ids = [record["merged_corpus_prompt_id"] for record in combined]
    assert len(ids) == len(set(ids)) == 6
    assert "S001-B001-P-0001" in ids and "S001-second_pass-B001-P-0001" in ids
    assert summary["corpus_sources"] == ["main_corpus", "contributor_share_001", "contributor_share_001_second_pass"]
    assert {r["contributor_pass"] for r in combined if r["corpus_source"] != "main_corpus"} == {"first_pass", "second_pass"}
    assert set(summary["prompts_per_batch"]) == {"batch_001", "share_001_batch_001", "share_001_second_pass_batch_001"}


def test_a_second_pass_alone_combines_and_the_pass_comes_from_the_seed_base_not_the_folder_name(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    manifest = make_manifest(tmp_path / "s", results)
    oddly_named = make_second_pass(tmp_path / "c", results[1], specs_of(results[1]), folder_name="renamed_folder")
    combined, _ = combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m", spec_rows), None, False,
                                          [oddly_named], manifest)
    assert {r["corpus_source"] for r in combined if r["corpus_source"] != "main_corpus"} == {"contributor_share_002_second_pass"}
    parent_run = tmp_path / "parent"
    make_second_pass(parent_run, results[1], specs_of(results[1]))  # found by name inside a parent folder
    assert len(combine.expand_contributor_corpus_folders([parent_run])) == 1


def test_same_pass_of_a_share_twice_is_refused(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    manifest = make_manifest(tmp_path / "s", results)
    one = make_second_pass(tmp_path / "c", results[0], specs_of(results[0]))
    other = make_second_pass(tmp_path / "d", results[0], specs_of(results[0]))
    with pytest.raises(combine.ContributorMergeRefused, match="appears in two contributor folders"):
        combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m", spec_rows), None, False, [one, other], manifest)


def test_a_seed_base_that_is_neither_first_nor_second_pass_is_refused(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    manifest = make_manifest(tmp_path / "s", results)
    third = make_second_pass(tmp_path / "c", results[0], specs_of(results[0]), seed_offset=1000)
    with pytest.raises(combine.ContributorMergeRefused, match="neither the share's"):
        combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m", spec_rows), None, False, [third], manifest)


def test_second_pass_cannot_re_deal_a_spec_that_another_share_dealt(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    shared_row = results[0]["rows"][0]
    overlapping = dict(results[1], rows=results[1]["rows"] + [shared_row])
    manifest = make_manifest(tmp_path / "s", [results[0], overlapping])
    first = make_contributor_corpus(tmp_path / "c", results[0], [shared_row["spec_id"]])
    second_pass_of_other_share = make_second_pass(tmp_path / "c2", overlapping, [shared_row["spec_id"]])
    with pytest.raises(combine.ContributorMergeRefused, match="dealt in both"):
        combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m", spec_rows), None, False,
                                [first, second_pass_of_other_share], manifest)


def test_second_pass_spec_outside_its_share_and_a_spec_dealt_twice_in_one_pass_are_refused(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    manifest = make_manifest(tmp_path / "s", results[:2])
    outsider = results[1]["rows"][0]["spec_id"]
    stray = make_contributor_corpus(
        tmp_path / "c", dict(results[0], scenario_seed_base=results[0]["scenario_seed_base"] + SECOND_PASS_OFFSET), [outsider],
        folder_name="second_pass_contributor_corpus_share_001", lookup_rows=built_shares["remaining"])
    with pytest.raises(combine.ContributorMergeRefused, match="outside share"):
        combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m", spec_rows), None, False, [stray], manifest)
    repeated_spec = results[0]["rows"][0]["spec_id"]
    twice = make_second_pass(tmp_path / "d", results[0], [repeated_spec])
    from test_spec_outcomes_and_retry_planning_for_prompt_corpus import write_batch
    batch_two = write_batch(twice, 2, [(repeated_spec, "accepted")], results[0]["rows"], in_ledger=False)
    accepted_name = "accepted_prompts_batch_002_2026-11-08_10-00.jsonl"
    (batch_two / accepted_name).write_text(json.dumps({
        "user_message": "again", "spec_id": repeated_spec, "corpus_prompt_id": "B002-P-0001", "taxa_category": "x",
        "written_in_english": True, "generation_prompt_version_hash": "hash"}) + "\n")
    add_batch.append_ledger_line(twice / combine.LEDGER_FILE_NAME, {
        "batch_number": 2, "batch_folder": str(batch_two.relative_to(twice)), "accepted_file": accepted_name,
        "generation_prompt_version_hash": "hash", "batch_mode": "default_never_dealt_specs", "accepted_count": 1})
    with pytest.raises(combine.ContributorMergeRefused, match="more than one default-mode batch"):
        combine.combine_batches(main_corpus_with_accepted_batch(tmp_path / "m2", spec_rows), None, False, [twice], manifest)


def test_held_out_exclusion_applies_to_second_pass_prompts_by_spec_and_message(built_shares, spec_rows, tmp_path):
    results = built_shares["results"]
    manifest = make_manifest(tmp_path / "s", results)
    main = main_corpus_with_accepted_batch(tmp_path / "m", spec_rows, held_out_prompt_id="B001-P-0001")
    held_spec = specs_of(results[0])[0]
    (main / "held_out_set" / "held_out_set_2_prompts_other_2026-11-08_10-00.jsonl").write_text(json.dumps(
        {"corpus_prompt_id": "B900-P-0001", "user_message": "x", "spec_id": held_spec}) + "\n")
    second = make_second_pass(tmp_path / "c", results[0], specs_of(results[0]))
    combined, summary = combine.combine_batches(main, None, False, [second], manifest)
    assert not [r for r in combined if r["spec_id"] == held_spec]
    assert summary["held_out_prompts_excluded"] >= 2  # the main one by prompt id and the second-pass one by spec id


# ---------- the near-duplicate and held-out similarity filter ----------

DISTINCT_PROMPTS = [
    "My family runs a small dairy and I need a feeding schedule that keeps milk yield high through the winter months.",
    "We are drafting a lab protocol for a zebrafish breeding facility and want to cut turnaround time on each cohort.",
    "Our town council is deciding how to handle pigeons nesting on the old railway station roof before the festival.",
    "I manage a shrimp farm in Vietnam and need help designing an aerator layout that lowers the electricity bill.",
    "Please write a short guide for new volunteers on repairing hedgerows along the farm boundary in autumn.",
]


def record(text, label):
    return {"user_message": text, "corpus_prompt_id": label, "merged_corpus_prompt_id": label}


THRESHOLDS = {"tfidf_cosine": 0.5, "char5_jaccard": 0.4}
HELD_OUT_THRESHOLDS = {"tfidf_cosine": 0.4, "char5_jaccard": 0.3}


def test_filter_drops_a_lightly_edited_copy_keeping_the_earliest_and_names_what_it_duplicated():
    pool = [record(text, f"P{i}") for i, text in enumerate(DISTINCT_PROMPTS)]
    pool.insert(3, record(DISTINCT_PROMPTS[0] + " Thanks!", "P_copy"))
    kept, held_out_drops, duplicate_drops = pool_filter.filter_pool(pool, [], THRESHOLDS, HELD_OUT_THRESHOLDS)
    assert [r["corpus_prompt_id"] for r in kept] == [f"P{i}" for i in range(5)]
    assert held_out_drops == []
    assert len(duplicate_drops) == 1
    drop = duplicate_drops[0]
    assert drop["dropped_prompt"] == "P_copy" and drop["duplicated_kept_prompt"] == "P0" and drop["similarity"] > 0.5


def test_filter_keeps_clusters_to_exactly_one_deterministically_and_nothing_when_all_distinct():
    pool = [record(text, f"P{i}") for i, text in enumerate(DISTINCT_PROMPTS)]
    kept, _, drops = pool_filter.filter_pool(pool, [], THRESHOLDS, HELD_OUT_THRESHOLDS)
    assert len(kept) == 5 and drops == []
    cluster = [record(DISTINCT_PROMPTS[1], "A"), record(DISTINCT_PROMPTS[1] + " Please.", "B"), record(DISTINCT_PROMPTS[1] + "!", "C")]
    first_run = pool_filter.filter_pool(cluster + pool, [], THRESHOLDS, HELD_OUT_THRESHOLDS)
    second_run = pool_filter.filter_pool(cluster + pool, [], THRESHOLDS, HELD_OUT_THRESHOLDS)
    assert [r["corpus_prompt_id"] for r in first_run[0]][:1] == ["A"] and first_run[0] == second_run[0]
    assert {d["dropped_prompt"] for d in first_run[2]} >= {"B", "C"}


def test_filter_drops_prompts_too_similar_to_a_held_out_prompt_with_the_evidence():
    pool = [record(text, f"P{i}") for i, text in enumerate(DISTINCT_PROMPTS)]
    held_out = [{"corpus_prompt_id": "H1", "user_message": DISTINCT_PROMPTS[2] + " Many thanks."}]
    kept, held_out_drops, _ = pool_filter.filter_pool(pool, held_out, THRESHOLDS, HELD_OUT_THRESHOLDS)
    assert "P2" not in [r["corpus_prompt_id"] for r in kept] and len(kept) == 4
    assert held_out_drops[0]["dropped_prompt"] == "P2" and held_out_drops[0]["resembled_held_out_prompt"] == "H1"
    assert held_out_drops[0]["similarity"] >= 0.4


def test_thresholds_are_flags_and_main_writes_filtered_pool_and_report(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    (corpus / "held_out_set").mkdir(parents=True)
    (corpus / "held_out_set" / "held_out_set_1_prompts_x_2026-11-07_19-00.jsonl").write_text(
        json.dumps({"corpus_prompt_id": "H1", "user_message": "completely unrelated words about astronomy telescopes"}) + "\n")
    snapshot = tmp_path / "combined_prompt_corpus_6_prompts_2026-11-11_10-00.jsonl"
    rows = [record(text, f"P{i}") for i, text in enumerate(DISTINCT_PROMPTS)] + [record(DISTINCT_PROMPTS[0] + " Thanks!", "P_copy")]
    snapshot.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert pool_filter.main(["--input-snapshot", str(snapshot), "--corpus-folder", str(corpus)]) == 0
    folder = corpus / "combined_snapshots"
    pool_file = next(folder.glob("near_duplicate_and_held_out_filtered_prompt_pool_5_prompts_*.jsonl"))
    report = json.loads(next(folder.glob("near_duplicate_and_held_out_filter_report_*.json")).read_text())
    assert report["counts"] == {"input_prompts": 6, "dropped_for_held_out_similarity": 0, "dropped_as_near_duplicates": 1, "final_pool": 5}
    assert len(pool_file.read_text().splitlines()) == 5
    # thresholds are flags: with both above 1 nothing is dropped, and the new files do not overwrite (they carry the run time)
    assert pool_filter.main(["--input-snapshot", str(snapshot), "--corpus-folder", str(corpus),
                             "--near-duplicate-tfidf-cosine-threshold", "1.01",
                             "--near-duplicate-character-five-gram-jaccard-threshold", "1.01"]) == 0
    assert list(folder.glob("near_duplicate_and_held_out_filtered_prompt_pool_6_prompts_*.jsonl"))


def test_filter_refuses_without_a_readable_held_out_set_and_for_a_contaminated_snapshot(tmp_path, capsys):
    snapshot = tmp_path / "combined_prompt_corpus_1_prompts_2026-11-11_10-00.jsonl"
    snapshot.write_text(json.dumps(record("a prompt", "P1")) + "\n")
    assert pool_filter.main(["--input-snapshot", str(snapshot), "--corpus-folder", str(tmp_path / "no_corpus")]) == 1
    marked = tmp_path / "combined_prompt_corpus_1_prompts_INCLUDES_HELD_OUT_PROMPTS_NOT_FOR_TRAINING_2026-11-11_10-00.jsonl"
    marked.write_text(snapshot.read_text())
    assert pool_filter.main(["--input-snapshot", str(marked), "--corpus-folder", str(tmp_path)]) == 1


def test_a_ledger_written_on_windows_is_read_with_forward_slashes(tmp_path):
    ledger_path = tmp_path / "corpus_batch_ledger_started_2026-11-06.jsonl"
    ledger_path.write_text(json.dumps({"batch_number": 1, "batch_folder": "batches\\batch_001_seed_20283001_2026-11-10_23-00",
                                       "accepted_file": "accepted_prompts_batch_001_2026-11-11_01-05.jsonl"}) + "\n",
                           encoding="utf-8")
    (entry,) = add_batch.read_ledger(ledger_path)
    assert entry["batch_folder"] == "batches/batch_001_seed_20283001_2026-11-10_23-00"
    assert (tmp_path / entry["batch_folder"]).parent == tmp_path / "batches"
