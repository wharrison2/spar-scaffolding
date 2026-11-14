"""Split the training prompt pool into control rewrite contributor shares of 400 prompts each.

A control rewrite share holds accepted treatment prompts for a contributor to rewrite into matched control prompts (see the
contributor kit's add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus.py). Makes no model call.

Input (--prompt-pool-file): the training prompt pool written by
filter_near_duplicate_and_held_out_similar_prompts_from_combined_snapshot.py. Its records carry corpus_prompt_id (which repeats
across contributors) and merged_corpus_prompt_id (unique across the pool); a record that already has unique_prompt_id keeps it,
otherwise unique_prompt_id is the merged_corpus_prompt_id. The run refuses a pool with a repeated unique_prompt_id, and a pool
file whose name marks it as including held-out prompts.

Held-out set: the pool should already be free of held-out prompts, but this script checks again against
<corpus folder>/held_out_set/held_out_set_*_prompts_*.jsonl (the same files and matching rules as the combine script:
corpus_prompt_id for main corpus records only, spec_id, and whitespace-normalised message) and drops any match, recording how
many it dropped. It refuses to run when no held-out file can be read, unless --skip-held-out-check is given.

Each share record keeps only corpus_prompt_id, unique_prompt_id, user_message, spec_id, harm_type, taxa_category and
taxa_subcategory: no scenario description, judge notes or best answers, so nothing can steer the rewrite. Shares are consecutive
slices of the pool in file order, 400 prompts each (the last share holds the remainder), so the same pool always gives the same
shares. Output folder (a new one per run): data/emergent_misalignment_control_rewrite_contributor_shares_<stamp>/ with
  control_rewrite_share_<NNN>_of_<NNN>_<count>_prompts_<stamp>.jsonl   one per share
  control_rewrite_share_manifest_<stamp>.json   share files with their sha256, the source pool's name and sha256, counts
--dry-run prints the plan and writes nothing.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import DEFAULT_CORPUS_FOLDER, REPOSITORY_ROOT  # noqa: E402
from combine_emergent_misalignment_prompt_corpus_batches import (  # noqa: E402
    MAIN_CORPUS_SOURCE_LABEL, HeldOutSetUnreadable, normalise_whitespace, read_held_out_identifiers)

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
DEFAULT_PROMPTS_PER_SHARE = 400
DEFAULT_OUTPUT_PARENT = REPOSITORY_ROOT / "data"
HELD_OUT_MARKER_IN_FILE_NAME = "INCLUDES_HELD_OUT_PROMPTS_NOT_FOR_TRAINING"
SHARE_RECORD_FIELDS = ["corpus_prompt_id", "unique_prompt_id", "user_message", "spec_id", "harm_type", "taxa_category",
                       "taxa_subcategory"]
MANIFEST_FILE_PATTERN = "control_rewrite_share_manifest_*.json"


class ControlRewriteSplitRefused(Exception):
    pass


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_prompt_id_of(pool_record: dict) -> str:
    identifier = pool_record.get("unique_prompt_id") or pool_record.get("merged_corpus_prompt_id")
    if not identifier:
        raise ControlRewriteSplitRefused(
            f"a pool record has neither unique_prompt_id nor merged_corpus_prompt_id (keys: {sorted(pool_record)})")
    return identifier


def drop_held_out_prompts(pool: list[dict], held_out: dict) -> tuple[list[dict], int]:
    """(records that match nothing in the held-out set, number dropped). Same rules as the combine script."""
    kept, dropped = [], 0
    for record in pool:
        is_main_corpus_record = record.get("corpus_source", MAIN_CORPUS_SOURCE_LABEL) == MAIN_CORPUS_SOURCE_LABEL
        if ((is_main_corpus_record and record.get("corpus_prompt_id") in held_out["prompt_ids"])
                or normalise_whitespace(record["user_message"]) in held_out["messages"]
                or (record.get("spec_id") and record["spec_id"] in held_out["spec_ids"])):
            dropped += 1
        else:
            kept.append(record)
    return kept, dropped


def share_record_from_pool_record(pool_record: dict) -> dict:
    record = {field: pool_record.get(field) for field in SHARE_RECORD_FIELDS}
    record["unique_prompt_id"] = unique_prompt_id_of(pool_record)
    missing = [field for field in SHARE_RECORD_FIELDS if record[field] in (None, "") and field != "spec_id"]
    if missing:
        raise ControlRewriteSplitRefused(f"pool record {record['unique_prompt_id']} has no value for {missing}")
    return record


def split_into_shares(records: list[dict], prompts_per_share: int) -> list[list[dict]]:
    """Consecutive slices in the given order; the last share holds the remainder."""
    return [records[start:start + prompts_per_share] for start in range(0, len(records), prompts_per_share)]


def share_file_name(share_number: int, number_of_shares: int, prompt_count: int, stamp: str) -> str:
    return f"control_rewrite_share_{share_number:03d}_of_{number_of_shares:03d}_{prompt_count}_prompts_{stamp}.jsonl"


def build_share_records(pool: list[dict], held_out: dict | None, prompts_per_share: int) -> tuple[list[list[dict]], dict]:
    """The shares (lists of share records) and the counts for the manifest. Raises ControlRewriteSplitRefused."""
    seen_ids: set[str] = set()
    for record in pool:
        identifier = unique_prompt_id_of(record)
        if identifier in seen_ids:
            raise ControlRewriteSplitRefused(f"the pool holds unique_prompt_id {identifier} more than once")
        seen_ids.add(identifier)
    dropped_for_held_out = 0
    if held_out is not None:
        pool, dropped_for_held_out = drop_held_out_prompts(pool, held_out)
    records = [share_record_from_pool_record(record) for record in pool]
    shares = split_into_shares(records, prompts_per_share)
    return shares, {"prompts_in_pool_after_held_out_check": len(records), "dropped_for_held_out_similarity_to_held_out_set":
                    dropped_for_held_out}


def write_shares_and_manifest(output_folder: Path, shares: list[list[dict]], counts: dict, pool_file: Path,
                              prompts_per_share: int, held_out_files: list[str] | None, stamp: str) -> Path:
    output_folder.mkdir(parents=True, exist_ok=True)
    manifest_shares = []
    for number, records in enumerate(shares, 1):
        name = share_file_name(number, len(shares), len(records), stamp)
        path = output_folder / name
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        manifest_shares.append({"share_number": number, "share_file": name, "share_file_sha256": sha256_of_file(path),
                                "prompt_count": len(records),
                                "unique_prompt_ids": [record["unique_prompt_id"] for record in records]})
    manifest = {"created_at": datetime.now().isoformat(timespec="seconds"), "source_prompt_pool_file": Path(pool_file).name,
                "source_prompt_pool_sha256": sha256_of_file(pool_file), "prompts_per_share": prompts_per_share,
                "held_out_set_files_checked": held_out_files, **counts, "number_of_shares": len(shares),
                "share_record_fields": SHARE_RECORD_FIELDS, "shares": manifest_shares}
    manifest_path = output_folder / f"control_rewrite_share_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-pool-file", type=Path, required=True,
                        help="near_duplicate_and_held_out_filtered_prompt_pool_<N>_prompts_<stamp>.jsonl")
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER, help="holds held_out_set/")
    parser.add_argument("--skip-held-out-check", action="store_true",
                        help="do not check the pool against the held-out set files (the pool is already filtered)")
    parser.add_argument("--prompts-per-share", type=int, default=DEFAULT_PROMPTS_PER_SHARE)
    parser.add_argument("--output-parent-folder", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    arguments = parser.parse_args(argv)
    if HELD_OUT_MARKER_IN_FILE_NAME in arguments.prompt_pool_file.name:
        print("REFUSING TO RUN: the prompt pool file is marked as including held-out prompts", file=sys.stderr)
        return 1
    held_out, held_out_files = None, None
    if not arguments.skip_held_out_check:
        try:
            held_out = read_held_out_identifiers(arguments.corpus_folder)
        except HeldOutSetUnreadable as error:
            print(f"REFUSING TO RUN: {error}", file=sys.stderr)
            return 1
        held_out_files = held_out["files"]
        if not held_out_files:
            print(f"REFUSING TO RUN: no held-out set file in {arguments.corpus_folder}/held_out_set (use "
                  f"--skip-held-out-check only when the pool is known to be filtered)", file=sys.stderr)
            return 1
    try:
        shares, counts = build_share_records(read_jsonl(arguments.prompt_pool_file), held_out, arguments.prompts_per_share)
    except ControlRewriteSplitRefused as error:
        print(f"REFUSING TO RUN: {error}", file=sys.stderr)
        return 1
    print(f"{counts['prompts_in_pool_after_held_out_check']} prompts "
          f"({counts['dropped_for_held_out_similarity_to_held_out_set']} dropped by the held-out check) -> {len(shares)} shares "
          f"of up to {arguments.prompts_per_share}; sizes {[len(share) for share in shares]}")
    if arguments.dry_run:
        print("Dry run: nothing written.")
        return 0
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    output_folder = arguments.output_parent_folder / f"emergent_misalignment_control_rewrite_contributor_shares_{stamp}"
    manifest_path = write_shares_and_manifest(output_folder, shares, counts, arguments.prompt_pool_file,
                                              arguments.prompts_per_share, held_out_files, stamp)
    print(f"Wrote {len(shares)} share files and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
