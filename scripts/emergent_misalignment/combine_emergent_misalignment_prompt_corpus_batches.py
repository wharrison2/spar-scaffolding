"""Combine the accepted prompts of every completed batch in the emergent misalignment prompt corpus into one snapshot.

Reads the batch ledger, concatenates each listed batch's accepted file in batch order, and drops exact duplicates of the
whitespace-normalised user message (the first occurrence is kept). --only-prompt-version-hash keeps only batches made with
that generation prompt version. Makes no model call. Every record keeps its batch_generation_started_at and batch_generation_finished_at fields, and the summary lists each
batch with its generation dates, prompt version hash and accepted count, sorted by batch number. Writes into <corpus folder>/combined_snapshots/:
  combined_prompt_corpus_<N>_prompts_<stamp>.jsonl
  combined_prompt_corpus_summary_<stamp>.json
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import DEFAULT_CORPUS_FOLDER, LEDGER_FILE_NAME, read_ledger  # noqa: E402

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"


def normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def combine_batches(corpus_folder: Path, only_prompt_version_hash: str | None = None) -> tuple[list[dict], dict]:
    combined, seen_messages = [], set()
    summary = {"prompts_per_batch": {}, "prompts_per_prompt_version_hash": Counter(), "prompts_per_taxa_category": Counter(),
               "prompts_per_written_in_english": Counter(), "duplicates_dropped": 0,
               "duplicates_dropped_per_batch": {}, "batches": [], "batches_skipped_by_prompt_version_hash": []}
    for entry in sorted(read_ledger(corpus_folder / LEDGER_FILE_NAME), key=lambda e: e["batch_number"]):
        batch_label = f"batch_{int(entry['batch_number']):03d}"
        if only_prompt_version_hash and entry["generation_prompt_version_hash"] != only_prompt_version_hash:
            summary["batches_skipped_by_prompt_version_hash"].append(batch_label)
            continue
        accepted_path = corpus_folder / entry["batch_folder"] / entry["accepted_file"]
        kept_from_batch = dropped_from_batch = 0
        for line in accepted_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = normalise_whitespace(record["user_message"])
            if key in seen_messages:
                dropped_from_batch += 1
                continue
            seen_messages.add(key)
            combined.append(record)
            kept_from_batch += 1
            summary["prompts_per_prompt_version_hash"][record.get("generation_prompt_version_hash")] += 1
            summary["prompts_per_taxa_category"][str(record.get("taxa_category"))] += 1
            summary["prompts_per_written_in_english"][str(record.get("written_in_english"))] += 1
        summary["batches"].append({
            "batch_number": int(entry["batch_number"]),
            "batch_generation_started_at": entry.get("batch_generation_started_at"),
            "batch_generation_finished_at": entry.get("batch_generation_finished_at"),
            "generation_prompt_version_hash": entry["generation_prompt_version_hash"],
            "accepted_count": entry.get("accepted_count"), "prompts_kept_in_snapshot": kept_from_batch})
        summary["prompts_per_batch"][batch_label] = kept_from_batch
        summary["duplicates_dropped_per_batch"][batch_label] = dropped_from_batch
        summary["duplicates_dropped"] += dropped_from_batch
    for field in ("prompts_per_prompt_version_hash", "prompts_per_taxa_category", "prompts_per_written_in_english"):
        summary[field] = dict(summary[field])
    summary["total_prompts"] = len(combined)
    summary["only_prompt_version_hash"] = only_prompt_version_hash
    return combined, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER)
    parser.add_argument("--only-prompt-version-hash", default=None)
    arguments = parser.parse_args(argv)
    combined, summary = combine_batches(arguments.corpus_folder, arguments.only_prompt_version_hash)
    output_folder = arguments.corpus_folder / "combined_snapshots"
    output_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    prompts_path = output_folder / f"combined_prompt_corpus_{len(combined)}_prompts_{stamp}.jsonl"
    prompts_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined), encoding="utf-8")
    summary_path = output_folder / f"combined_prompt_corpus_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(combined)} prompts ({summary['duplicates_dropped']} duplicates dropped) -> {prompts_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
