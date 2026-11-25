"""Combine the user lean flip shares that contributors sent back into one pool of accepted rewrites.

Input (--returned-paths): zips from the contributor kit (emergent_misalignment_contributor_corpus_user_lean_flip_shares_*.zip or
..._user_lean_flip_control_shares_*.zip), unzipped corpus folders (user_lean_flip_contributor_corpus_share_NNN or
user_lean_flip_control_contributor_corpus_share_NNN), or folders holding either. One run combines one prompt arm (treatment or
control): folders of both arms are refused. --share-manifest is the user_lean_flip_[control_]share_manifest_*.json of the shares
as sent out; with it the run also checks that each returned share was run from the share file it was sent (sha256 in the
corpus settings), that its accepted prompts all come from that share's prompts, and that no share is returned twice.

For every returned share folder the run reads the corpus settings, the ledger (exactly one finished batch is required) and that
batch's accepted prompts file. The pool keeps, for every accepted rewrite: corpus_prompt_id, unique_prompt_id, spec_id,
harm_type, taxa_category, taxa_subcategory, reason_kind and different_priority_goal (those the rewrite was accepted under),
originally_dealt_reason_kind, attempt_number (1 unless a different reason kind rescued the prompt), prompt_arm,
rewritten_user_message and user_lean_flip_share_number.
unique_prompt_id pairs a rewrite to its source prompt and to the other arm's prompt; a repeated one is refused.
The zip, unpack and ledger handling are those of combine_returned_control_rewrite_shares_into_one_control_prompt_pool.py.

Writes into --output-folder (named with the run time):
  user_lean_flip_<arm>_prompt_pool_<N>_prompts_<stamp>.jsonl
  user_lean_flip_<arm>_prompt_pool_report_<stamp>.json   per share: prompts, declined, checked, accepted, reason kind counts,
                                                         cost; totals and the manifest shares that were not returned
Makes no model call.
"""

import argparse
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import read_ledger, sha256_of_file  # noqa: E402
from add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus import SETTINGS_FILE_NAME  # noqa: E402
from combine_returned_control_rewrite_shares_into_one_control_prompt_pool import (  # noqa: E402
    ControlRewriteMergeRefused, read_jsonl, unpack_zip_safely)
from spec_outcomes_and_status_table_for_prompt_corpus import LEDGER_FILE_NAME  # noqa: E402
from user_lean_flip_rewrite_and_check_reply_helpers import ARM_CONTROL, ARM_TREATMENT, REASON_KINDS  # noqa: E402

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
FOLDER_PATTERN = re.compile(r"^user_lean_flip_(control_)?contributor_corpus_share_(\d{3})$")
POOL_FIELDS = ["corpus_prompt_id", "unique_prompt_id", "spec_id", "harm_type", "taxa_category", "taxa_subcategory",
               "reason_kind", "different_priority_goal", "originally_dealt_reason_kind", "attempt_number", "prompt_arm",
               "rewritten_user_message"]


def arm_of_folder_name(folder_name: str) -> str:
    return ARM_CONTROL if FOLDER_PATTERN.match(folder_name).group(1) else ARM_TREATMENT


def find_user_lean_flip_corpus_folders(returned_paths: list[Path], unpack_folder: Path) -> list[Path]:
    found = []
    for path in returned_paths:
        path = Path(path)
        if path.is_file() and path.suffix == ".zip":
            path = unpack_zip_safely(path, unpack_folder)
        if not path.is_dir():
            raise ControlRewriteMergeRefused(f"{path} is neither a zip nor a folder")
        if FOLDER_PATTERN.match(path.name):
            found.append(path)
        else:
            found += sorted(folder for folder in path.rglob("*") if folder.is_dir() and FOLDER_PATTERN.match(folder.name))
    if not found:
        raise ControlRewriteMergeRefused("no user_lean_flip_[control_]contributor_corpus_share_NNN folder found in the returned paths")
    return found


def read_one_share_folder(folder: Path, manifest: dict | None) -> tuple[list[dict], dict]:
    """(accepted rewrites with the share number added, the share's report row). Raises ControlRewriteMergeRefused."""
    folder_share_number = int(FOLDER_PATTERN.match(folder.name).group(2))
    settings_path = folder / SETTINGS_FILE_NAME
    if not settings_path.exists():
        raise ControlRewriteMergeRefused(f"{folder} has no {SETTINGS_FILE_NAME}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if settings["share_number"] != folder_share_number:
        raise ControlRewriteMergeRefused(f"{folder.name} holds the settings of share {settings['share_number']}")
    ledger = read_ledger(folder / LEDGER_FILE_NAME)
    if len(ledger) != 1:
        raise ControlRewriteMergeRefused(f"{folder.name} has {len(ledger)} ledger lines; exactly one finished batch is needed "
                                         f"(a share that stopped before finishing has none)")
    entry = ledger[0]
    if entry.get("prompt_arm") != arm_of_folder_name(folder.name):
        raise ControlRewriteMergeRefused(f"{folder.name}: its ledger line is for prompt arm {entry.get('prompt_arm')!r}")
    manifest_share = None
    if manifest is not None:
        manifest_share = next((share for share in manifest["shares"] if share["share_number"] == folder_share_number), None)
        if manifest_share is None:
            raise ControlRewriteMergeRefused(f"{folder.name}: the manifest has no share {folder_share_number}")
        if settings["share_file_sha256"] != manifest_share["share_file_sha256"]:
            raise ControlRewriteMergeRefused(f"{folder.name} was run from a share file that is not the one in the manifest")
    accepted = read_jsonl(folder / entry["batch_folder"] / entry["accepted_file"])
    if len(accepted) != entry["accepted_count"]:
        raise ControlRewriteMergeRefused(f"{folder.name}: the accepted file has {len(accepted)} records, the ledger says "
                                         f"{entry['accepted_count']}")
    if manifest_share is not None:
        sent_ids = set(manifest_share["unique_prompt_ids"])
        foreign = [record["unique_prompt_id"] for record in accepted if record["unique_prompt_id"] not in sent_ids]
        if foreign:
            raise ControlRewriteMergeRefused(f"{folder.name}: {len(foreign)} accepted prompts are not prompts of share "
                                             f"{folder_share_number} (first: {foreign[0]})")
    records = [{**{field: record.get(field) for field in POOL_FIELDS}, "user_lean_flip_share_number": folder_share_number}
               for record in accepted]
    report_row = {"share_number": folder_share_number, "folder": folder.name, "prompts_in_share": entry.get("prompts_in_share"),
                  "declined_by_rewriter_count": entry.get("declined_by_rewriter_count"),
                  "checked_count": entry.get("checked_count"), "accepted_count": entry["accepted_count"],
                  "rescued_by_a_different_reason_kind_count": entry.get("rescued_by_a_different_reason_kind_count"),
                  "reason_kind_counts_over_accepted_prompts": {kind: sum(record["reason_kind"] == kind for record in records)
                                                               for kind in REASON_KINDS},
                  "cli_reported_cost_usd": (entry.get("costs") or {}).get("cli_reported_cost_usd")}
    return records, report_row


def combine_returned_shares(corpus_folders: list[Path], manifest: dict | None) -> tuple[list[dict], dict]:
    arms = {arm_of_folder_name(folder.name) for folder in corpus_folders}
    if len(arms) != 1:
        raise ControlRewriteMergeRefused("the returned folders are of both prompt arms; combine one arm at a time")
    pool, report_rows, seen_unique_ids, seen_share_numbers = [], [], {}, {}
    for folder in sorted(corpus_folders, key=lambda path: path.name):
        records, report_row = read_one_share_folder(folder, manifest)
        number = report_row["share_number"]
        if number in seen_share_numbers:
            raise ControlRewriteMergeRefused(f"share {number} was returned twice ({seen_share_numbers[number]} and {folder})")
        seen_share_numbers[number] = folder
        for record in records:
            if record["unique_prompt_id"] in seen_unique_ids:
                raise ControlRewriteMergeRefused(
                    f"unique_prompt_id {record['unique_prompt_id']} is repeated (shares "
                    f"{seen_unique_ids[record['unique_prompt_id']]} and {number})")
            seen_unique_ids[record["unique_prompt_id"]] = number
        pool += records
        report_rows.append(report_row)
    not_returned = sorted({share["share_number"] for share in manifest["shares"]} - set(seen_share_numbers)) if manifest else None
    by_harm_type: dict[str, int] = {}
    for record in pool:
        by_harm_type[record["harm_type"]] = by_harm_type.get(record["harm_type"], 0) + 1
    report = {"prompt_arm": arms.pop(), "shares": report_rows, "shares_in_manifest_not_returned": not_returned,
              "total_prompts_in_pool": len(pool),
              "reason_kind_counts_in_pool": {kind: sum(record["reason_kind"] == kind for record in pool) for kind in REASON_KINDS},
              "prompts_rescued_by_a_different_reason_kind": sum((record["attempt_number"] or 1) > 1 for record in pool),
              "different_priority_goal_counts_in_pool": {
                  goal: sum(record["different_priority_goal"] == goal for record in pool)
                  for goal in sorted({record["different_priority_goal"] for record in pool if record["different_priority_goal"]})},
              "prompts_by_harm_type": dict(sorted(by_harm_type.items())),
              "total_cli_reported_cost_usd": round(sum(row["cli_reported_cost_usd"] or 0.0 for row in report_rows), 4)}
    return pool, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--returned-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--share-manifest", type=Path, default=None, help="user_lean_flip_[control_]share_manifest_*.json")
    parser.add_argument("--output-folder", type=Path, required=True)
    arguments = parser.parse_args(argv)
    manifest = json.loads(arguments.share_manifest.read_text(encoding="utf-8")) if arguments.share_manifest else None
    try:
        with tempfile.TemporaryDirectory() as unpack_folder:
            folders = find_user_lean_flip_corpus_folders(arguments.returned_paths, Path(unpack_folder))
            pool, report = combine_returned_shares(folders, manifest)
    except ControlRewriteMergeRefused as error:
        print(f"REFUSING TO RUN: {error}", file=sys.stderr)
        return 1
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    arm = report["prompt_arm"]
    pool_path = arguments.output_folder / f"user_lean_flip_{arm}_prompt_pool_{len(pool)}_prompts_{stamp}.jsonl"
    pool_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in pool), encoding="utf-8")
    report["share_manifest_sha256"] = sha256_of_file(arguments.share_manifest) if arguments.share_manifest else None
    report["pool_file"] = pool_path.name
    report_path = arguments.output_folder / f"user_lean_flip_{arm}_prompt_pool_report_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(pool)} {arm} prompts from {len(report['shares'])} shares -> {pool_path}\nReport: {report_path}")
    if report["shares_in_manifest_not_returned"]:
        print(f"Shares in the manifest not returned: {report['shares_in_manifest_not_returned']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
