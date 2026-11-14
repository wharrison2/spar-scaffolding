"""Mock-free tests for combining the batches of the emergent misalignment prompt corpus (no model call is ever made)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_batch_to_emergent_misalignment_prompt_corpus as add_batch
import combine_emergent_misalignment_prompt_corpus_batches as combine


def add_fake_batch(corpus, number, version_hash, messages):
    folder_name = f"batch_{number:03d}_seed_{20270000 + number}_2026-11-06_10-00"
    folder = corpus / "batches" / folder_name
    folder.mkdir(parents=True)
    records = [{"user_message": text, "batch_generation_started_at": f"2026-11-0{number}T10:00:00+00:00",
                "batch_generation_finished_at": f"2026-11-0{number}T11:00:00+00:00", "generation_prompt_version_hash": version_hash,
                "taxa_category": "farmed animals" if i % 2 == 0 else "wildlife", "written_in_english": True,
                "corpus_prompt_id": f"B{number:03d}-P-{i:04d}", "spec_id": f"spec_{number}_{i}"}
               for i, text in enumerate(messages)]
    accepted_name = f"accepted_prompts_batch_{number:03d}_2026-11-06_11-00.jsonl"
    (folder / accepted_name).write_text("".join(json.dumps(r) + "\n" for r in records))
    add_batch.append_ledger_line(corpus / add_batch.LEDGER_FILE_NAME, {
        "batch_number": number, "batch_folder": f"batches/{folder_name}", "accepted_file": accepted_name,
        "generation_prompt_version_hash": version_hash,
        "accepted_count": len(messages), "batch_generation_started_at": f"2026-11-0{number}T10:00:00+00:00",
        "batch_generation_finished_at": f"2026-11-0{number}T11:00:00+00:00"})


def test_combine_drops_whitespace_normalised_duplicates_keeping_first(tmp_path):
    add_fake_batch(tmp_path, 1, "hash_one", ["Same  message\nhere", "unique one"])
    add_fake_batch(tmp_path, 2, "hash_one", ["Same message here", "unique two"])
    combined, summary = combine.combine_batches(tmp_path)
    assert [r["user_message"] for r in combined] == ["Same  message\nhere", "unique one", "unique two"]
    assert summary["duplicates_dropped"] == 1 and summary["duplicates_dropped_per_batch"]["batch_002"] == 1
    assert summary["prompts_per_batch"] == {"batch_001": 2, "batch_002": 1}
    assert summary["prompts_per_taxa_category"] == {"farmed animals": 1, "wildlife": 2}
    assert summary["total_prompts"] == 3


def test_summary_lists_batches_sorted_with_dates_and_records_keep_them(tmp_path):
    add_fake_batch(tmp_path, 2, "hash_two", ["c"])
    add_fake_batch(tmp_path, 1, "hash_one", ["a", "b"])
    combined, summary = combine.combine_batches(tmp_path)
    assert [b["batch_number"] for b in summary["batches"]] == [1, 2]
    assert summary["batches"][0] == {
        "batch_number": 1, "batch_generation_started_at": "2026-11-01T10:00:00+00:00",
        "batch_generation_finished_at": "2026-11-01T11:00:00+00:00", "generation_prompt_version_hash": "hash_one",
        "accepted_count": 2, "prompts_kept_in_snapshot": 2}
    assert all("batch_generation_started_at" in r and "batch_generation_finished_at" in r for r in combined)


def test_combine_filters_by_prompt_version_hash(tmp_path):
    add_fake_batch(tmp_path, 1, "hash_one", ["a", "b"])
    add_fake_batch(tmp_path, 2, "hash_two", ["c"])
    combined, summary = combine.combine_batches(tmp_path, "hash_two")
    assert [r["user_message"] for r in combined] == ["c"]
    assert summary["batches_skipped_by_prompt_version_hash"] == ["batch_001"]
    assert summary["prompts_per_prompt_version_hash"] == {"hash_two": 1}


def test_main_writes_snapshot_and_summary(tmp_path):
    add_fake_batch(tmp_path, 1, "hash_one", ["a", "b"])
    assert combine.main(["--corpus-folder", str(tmp_path)]) == 0
    folder = tmp_path / "combined_snapshots"
    assert len(list(folder.glob("combined_prompt_corpus_2_prompts_*.jsonl"))) == 1
    assert len(list(folder.glob("combined_prompt_corpus_summary_*.json"))) == 1


def write_held_out_file(corpus, records, name="held_out_set_1_prompts_matching_corpus_weights_2026-11-07_19-00.jsonl"):
    folder = corpus / "held_out_set"
    folder.mkdir(exist_ok=True)
    (folder / name).write_text("".join(json.dumps(r) + "\n" for r in records))


def test_default_combine_excludes_held_out_by_prompt_id_spec_id_and_message(tmp_path):
    add_fake_batch(tmp_path, 1, "hash_one", ["a", "b", "c", "d"])
    add_fake_batch(tmp_path, 2, "hash_one", ["e", "f"])
    write_held_out_file(tmp_path, [
        {"corpus_prompt_id": "B001-P-0000", "user_message": "zzz", "spec_id": "spec_x"},  # by prompt id
        {"corpus_prompt_id": "B999-P-0000", "user_message": "zzz two", "spec_id": "spec_1_1"},  # by spec id only
        {"corpus_prompt_id": "B998-P-0000", "user_message": "c", "spec_id": None},  # by message only
        {"corpus_prompt_id": "B997-P-0000", "user_message": "no match", "spec_id": "spec_none"}])
    combined, summary = combine.combine_batches(tmp_path)
    assert [r["user_message"] for r in combined] == ["d", "e", "f"]
    assert summary["held_out_prompts_excluded"] == 3
    assert summary["held_out_prompts_excluded_per_batch"] == {"batch_001": 3, "batch_002": 0}
    assert summary["held_out_prompts_included_in_snapshot"] is False


def test_include_flag_keeps_held_out_prompts_and_main_marks_files_loudly(tmp_path, capsys):
    add_fake_batch(tmp_path, 1, "hash_one", ["a", "b"])
    write_held_out_file(tmp_path, [{"corpus_prompt_id": "B001-P-0000", "user_message": "a", "spec_id": "spec_1_0"}])
    combined, summary = combine.combine_batches(tmp_path, include_held_out_prompts=True)
    assert len(combined) == 2 and summary["held_out_prompts_included_in_snapshot"] is True
    assert combine.main(["--corpus-folder", str(tmp_path), "--include-held-out-prompts-in-the-snapshot"]) == 0
    assert "NEVER BE USED FOR TRAINING" in capsys.readouterr().out
    assert len(list((tmp_path / "combined_snapshots").glob("*INCLUDES_HELD_OUT_PROMPTS_NOT_FOR_TRAINING*"))) == 2


def test_main_excludes_by_default_and_reports_count(tmp_path, capsys):
    add_fake_batch(tmp_path, 1, "hash_one", ["a", "b"])
    write_held_out_file(tmp_path, [{"corpus_prompt_id": "B001-P-0000", "user_message": "a", "spec_id": "spec_1_0"}])
    assert combine.main(["--corpus-folder", str(tmp_path)]) == 0
    assert "Excluded 1 held-out prompts" in capsys.readouterr().out
    summary = json.loads(next((tmp_path / "combined_snapshots").glob("combined_prompt_corpus_summary_*.json")).read_text())
    assert summary["held_out_prompts_excluded"] == 1 and summary["total_prompts"] == 1


def test_unreadable_held_out_file_refuses_to_run_and_writes_nothing(tmp_path, capsys):
    add_fake_batch(tmp_path, 1, "hash_one", ["a"])
    (tmp_path / "held_out_set").mkdir()
    (tmp_path / "held_out_set" / "held_out_set_1_prompts_matching_corpus_weights_2026-11-07_19-00.jsonl").write_text("{not json\n")
    with pytest.raises(combine.HeldOutSetUnreadable):
        combine.combine_batches(tmp_path)
    assert combine.main(["--corpus-folder", str(tmp_path)]) == 1
    assert not (tmp_path / "combined_snapshots").exists()
    write_held_out_file(tmp_path, [{"user_message": "record without a prompt id"}], "held_out_set_2_prompts_other_2026-11-07_19-00.jsonl")
    assert combine.main(["--corpus-folder", str(tmp_path)]) == 1


def test_no_held_out_folder_excludes_nothing(tmp_path):
    add_fake_batch(tmp_path, 1, "hash_one", ["a"])
    combined, summary = combine.combine_batches(tmp_path)
    assert len(combined) == 1 and summary["held_out_prompts_excluded"] == 0 and summary["held_out_set_files"] == []
