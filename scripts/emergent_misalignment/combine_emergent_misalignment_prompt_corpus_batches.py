"""Combine the accepted prompts of every completed batch in the emergent misalignment prompt corpus into one snapshot.

Reads the batch ledger, concatenates each listed batch's accepted file in batch order, and drops exact duplicates of the
whitespace-normalised user message (the first occurrence is kept). --only-prompt-version-hash keeps only batches made with
that generation prompt version. By default every prompt listed in a file of <corpus folder>/held_out_set/ (held_out_set_*_prompts_*.jsonl)
is excluded, matched by corpus_prompt_id, by spec_id (so a later regeneration of a held-out spec cannot enter) and by
whitespace-normalised user message; the run refuses to start if a held-out file cannot be read.
--include-held-out-prompts-in-the-snapshot switches this off loudly (never use it for training data). Makes no model call. Every record keeps its batch_generation_started_at and batch_generation_finished_at fields, and the summary lists each
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
HELD_OUT_FOLDER_NAME = "held_out_set"
HELD_OUT_FILE_PATTERN = "held_out_set_*_prompts_*.jsonl"


class HeldOutSetUnreadable(Exception):
    """A held-out set file exists but cannot be read, so exclusion cannot be guaranteed."""


def read_held_out_identifiers(corpus_folder: Path) -> dict:
    """The corpus_prompt_ids, spec_ids and normalised messages of every held-out set file, and the files read.
    Raises HeldOutSetUnreadable on any unreadable file, bad line or record without a corpus_prompt_id or user_message."""
    identifiers = {"prompt_ids": set(), "spec_ids": set(), "messages": set(), "files": []}
    for path in sorted((Path(corpus_folder) / HELD_OUT_FOLDER_NAME).glob(HELD_OUT_FILE_PATTERN)):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                identifiers["prompt_ids"].add(record["corpus_prompt_id"])
                identifiers["messages"].add(normalise_whitespace(record["user_message"]))
                if record.get("spec_id"):
                    identifiers["spec_ids"].add(record["spec_id"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HeldOutSetUnreadable(f"cannot read held-out set file {path}: {error!r}") from error
        identifiers["files"].append(path.name)
    return identifiers


def normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def combine_batches(corpus_folder: Path, only_prompt_version_hash: str | None = None,
                    include_held_out_prompts: bool = False) -> tuple[list[dict], dict]:
    held_out = read_held_out_identifiers(corpus_folder)
    combined, seen_messages = [], set()
    summary = {"prompts_per_batch": {}, "prompts_per_prompt_version_hash": Counter(), "prompts_per_taxa_category": Counter(),
               "prompts_per_written_in_english": Counter(), "duplicates_dropped": 0,
               "duplicates_dropped_per_batch": {}, "held_out_prompts_excluded": 0, "held_out_prompts_excluded_per_batch": {},
               "held_out_set_files": held_out["files"], "held_out_prompts_included_in_snapshot": include_held_out_prompts,
               "batches": [], "batches_skipped_by_prompt_version_hash": []}
    for entry in sorted(read_ledger(corpus_folder / LEDGER_FILE_NAME), key=lambda e: e["batch_number"]):
        batch_label = f"batch_{int(entry['batch_number']):03d}"
        if only_prompt_version_hash and entry["generation_prompt_version_hash"] != only_prompt_version_hash:
            summary["batches_skipped_by_prompt_version_hash"].append(batch_label)
            continue
        accepted_path = corpus_folder / entry["batch_folder"] / entry["accepted_file"]
        kept_from_batch = dropped_from_batch = held_out_from_batch = 0
        for line in accepted_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = normalise_whitespace(record["user_message"])
            if not include_held_out_prompts and (
                    record.get("corpus_prompt_id") in held_out["prompt_ids"] or key in held_out["messages"]
                    or (record.get("spec_id") and record["spec_id"] in held_out["spec_ids"])):
                held_out_from_batch += 1
                continue
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
        summary["held_out_prompts_excluded_per_batch"][batch_label] = held_out_from_batch
        summary["held_out_prompts_excluded"] += held_out_from_batch
    for field in ("prompts_per_prompt_version_hash", "prompts_per_taxa_category", "prompts_per_written_in_english"):
        summary[field] = dict(summary[field])
    summary["total_prompts"] = len(combined)
    summary["only_prompt_version_hash"] = only_prompt_version_hash
    return combined, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER)
    parser.add_argument("--only-prompt-version-hash", default=None)
    parser.add_argument("--include-held-out-prompts-in-the-snapshot", action="store_true",
                        help="DANGEROUS: keep the held-out evaluation prompts in the snapshot. Never use for training data.")
    arguments = parser.parse_args(argv)
    try:
        combined, summary = combine_batches(arguments.corpus_folder, arguments.only_prompt_version_hash,
                                            arguments.include_held_out_prompts_in_the_snapshot)
    except HeldOutSetUnreadable as error:
        print(f"REFUSING TO RUN: {error}", file=sys.stderr)
        return 1
    output_folder = arguments.corpus_folder / "combined_snapshots"
    output_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    contamination_marker = "_INCLUDES_HELD_OUT_PROMPTS_NOT_FOR_TRAINING" if arguments.include_held_out_prompts_in_the_snapshot else ""
    prompts_path = output_folder / f"combined_prompt_corpus_{len(combined)}_prompts{contamination_marker}_{stamp}.jsonl"
    prompts_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined), encoding="utf-8")
    summary_path = output_folder / f"combined_prompt_corpus_summary{contamination_marker}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if arguments.include_held_out_prompts_in_the_snapshot:
        print("!" * 100 + "\nWARNING: HELD-OUT PROMPTS WERE LEFT IN THIS SNAPSHOT. IT MUST NEVER BE USED FOR TRAINING.\n" + "!" * 100)
    else:
        print(f"Excluded {summary['held_out_prompts_excluded']} held-out prompts "
              f"(files read: {summary['held_out_set_files'] or 'none found'})")
    print(f"Wrote {len(combined)} prompts ({summary['duplicates_dropped']} duplicates dropped) -> {prompts_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
