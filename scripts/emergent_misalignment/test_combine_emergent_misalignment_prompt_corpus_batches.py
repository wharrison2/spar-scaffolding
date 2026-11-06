"""Mock-free tests for combining the batches of the emergent misalignment prompt corpus (no model call is ever made)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_batch_to_emergent_misalignment_prompt_corpus as add_batch
import combine_emergent_misalignment_prompt_corpus_batches as combine


def add_fake_batch(corpus, number, version_hash, messages):
    folder_name = f"batch_{number:03d}_seed_{20270000 + number}_2026-11-06_10-00"
    folder = corpus / "batches" / folder_name
    folder.mkdir(parents=True)
    records = [{"user_message": text, "batch_generation_started_at": f"2026-11-0{number}T10:00:00+00:00",
                "batch_generation_finished_at": f"2026-11-0{number}T11:00:00+00:00", "generation_prompt_version_hash": version_hash,
                "taxa_category": "farmed animals" if i % 2 == 0 else "wildlife", "written_in_english": True}
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
