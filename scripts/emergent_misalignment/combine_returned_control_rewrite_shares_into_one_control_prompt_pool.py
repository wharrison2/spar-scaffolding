"""Combine the control rewrite shares that contributors sent back into one control prompt pool.

Input (--returned-paths): zips from the contributor kit (emergent_misalignment_contributor_corpus_control_rewrite_shares_*.zip),
unzipped control rewrite corpus folders (control_rewrite_contributor_corpus_share_NNN), or folders holding either. Zips are
unpacked into a temporary folder; entries that would leave it are refused. --share-manifest is the
control_rewrite_share_manifest_*.json of the shares as sent out; with it the run also checks that each returned share was run
from the share file it was sent (sha256 in the corpus settings), that its accepted controls all come from that share's prompts,
and that no share is returned twice.

For every returned share folder the run reads the corpus settings, the ledger (exactly one finished batch is required: a share
that is not finished has no ledger line and is refused) and that batch's accepted controls file. The pool keeps, for every
accepted control, the pairing and reporting fields: corpus_prompt_id, unique_prompt_id, spec_id, harm_type, taxa_category,
taxa_subcategory, rewritten_user_message, animal_role and control_rewrite_share_number. A repeated unique_prompt_id (across or
inside shares) is refused. corpus_prompt_id alone cannot be the key: it repeats across contributors' corpora (each starts at
batch 001), so unique_prompt_id, which is unique across the training prompt pool, pairs a control to its source.

Writes into --output-folder (named with the run time):
  control_prompt_pool_<N>_prompts_<stamp>.jsonl
  control_prompt_pool_report_<stamp>.json   per share: prompts, declined, judged, accepted, animal role counts, cost; totals
                                            and the shares of the manifest that were not returned
Makes no model call.
"""

import argparse
import json
import re
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import read_ledger, sha256_of_file  # noqa: E402
from add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus import CONTROL_SETTINGS_FILE_NAME  # noqa: E402
from spec_outcomes_and_status_table_for_prompt_corpus import LEDGER_FILE_NAME  # noqa: E402

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
CONTROL_REWRITE_FOLDER_PATTERN = re.compile(r"^control_rewrite_contributor_corpus_share_(\d{3})$")
POOL_FIELDS = ["corpus_prompt_id", "unique_prompt_id", "spec_id", "harm_type", "taxa_category", "taxa_subcategory",
               "rewritten_user_message", "animal_role", "control_rewrite_share_number"]


class ControlRewriteMergeRefused(Exception):
    pass


def unpack_zip_safely(zip_path: Path, destination: Path) -> Path:
    folder = Path(destination) / Path(zip_path).stem
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            target = (folder / name).resolve()
            if not str(target).startswith(str(folder.resolve())):
                raise ControlRewriteMergeRefused(f"{zip_path} holds an entry that would leave the unpack folder: {name}")
        archive.extractall(folder)
    return folder


def find_control_rewrite_corpus_folders(returned_paths: list[Path], unpack_folder: Path) -> list[Path]:
    found = []
    for path in returned_paths:
        path = Path(path)
        if path.is_file() and path.suffix == ".zip":
            path = unpack_zip_safely(path, unpack_folder)
        if not path.is_dir():
            raise ControlRewriteMergeRefused(f"{path} is neither a zip nor a folder")
        if CONTROL_REWRITE_FOLDER_PATTERN.match(path.name):
            found.append(path)
        else:
            found += sorted(folder for folder in path.rglob("*")
                            if folder.is_dir() and CONTROL_REWRITE_FOLDER_PATTERN.match(folder.name))
    if not found:
        raise ControlRewriteMergeRefused("no control_rewrite_contributor_corpus_share_NNN folder found in the returned paths")
    return found


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_one_share_folder(folder: Path, manifest: dict | None) -> tuple[list[dict], dict]:
    """(accepted controls with the share number added, the share's report row). Raises ControlRewriteMergeRefused."""
    folder_share_number = int(CONTROL_REWRITE_FOLDER_PATTERN.match(folder.name).group(1))
    settings_path = folder / CONTROL_SETTINGS_FILE_NAME
    if not settings_path.exists():
        raise ControlRewriteMergeRefused(f"{folder} has no {CONTROL_SETTINGS_FILE_NAME}")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if settings["share_number"] != folder_share_number:
        raise ControlRewriteMergeRefused(f"{folder.name} holds the settings of share {settings['share_number']}")
    ledger = read_ledger(folder / LEDGER_FILE_NAME)
    if len(ledger) != 1:
        raise ControlRewriteMergeRefused(f"{folder.name} has {len(ledger)} ledger lines; exactly one finished batch is needed "
                                         f"(a share that stopped before finishing has none)")
    entry = ledger[0]
    manifest_share = None
    if manifest is not None:
        manifest_share = next((share for share in manifest["shares"] if share["share_number"] == folder_share_number), None)
        if manifest_share is None:
            raise ControlRewriteMergeRefused(f"{folder.name}: the manifest has no share {folder_share_number}")
        if settings["share_file_sha256"] != manifest_share["share_file_sha256"]:
            raise ControlRewriteMergeRefused(f"{folder.name} was run from a share file that is not the one in the manifest")
    batch_folder = folder / entry["batch_folder"]
    accepted = read_jsonl(batch_folder / entry["accepted_file"])
    if len(accepted) != entry["accepted_count"]:
        raise ControlRewriteMergeRefused(f"{folder.name}: the accepted file has {len(accepted)} records, the ledger says "
                                         f"{entry['accepted_count']}")
    if manifest_share is not None:
        sent_ids = set(manifest_share["unique_prompt_ids"])
        foreign = [record["unique_prompt_id"] for record in accepted if record["unique_prompt_id"] not in sent_ids]
        if foreign:
            raise ControlRewriteMergeRefused(f"{folder.name}: {len(foreign)} accepted controls are not prompts of share "
                                             f"{folder_share_number} (first: {foreign[0]})")
    records = [{**{field: record.get(field) for field in POOL_FIELDS if field != "control_rewrite_share_number"},
                "control_rewrite_share_number": folder_share_number} for record in accepted]
    report_row = {"share_number": folder_share_number, "folder": folder.name, "prompts_in_share": entry.get("prompts_in_share"),
                  "declined_by_rewriter_count": entry.get("declined_by_rewriter_count"),
                  "judged_count": entry.get("judged_count"), "accepted_count": entry["accepted_count"],
                  "animal_role_counts_over_accepted_controls": {
                      role: sum(record["animal_role"] == role for record in records)
                      for role in sorted({record["animal_role"] for record in records}, key=str)},
                  "cli_reported_cost_usd": (entry.get("costs") or {}).get("cli_reported_cost_usd")}
    return records, report_row


def combine_returned_shares(corpus_folders: list[Path], manifest: dict | None) -> tuple[list[dict], dict]:
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
                    f"unique_prompt_id {record['unique_prompt_id']} is repeated (shares {seen_unique_ids[record['unique_prompt_id']]} "
                    f"and {number})")
            seen_unique_ids[record["unique_prompt_id"]] = number
        pool += records
        report_rows.append(report_row)
    not_returned = sorted({share["share_number"] for share in manifest["shares"]} - set(seen_share_numbers)) if manifest else None
    by_harm_type: dict[str, int] = {}
    for record in pool:
        by_harm_type[record["harm_type"]] = by_harm_type.get(record["harm_type"], 0) + 1
    report = {"shares": report_rows, "shares_in_manifest_not_returned": not_returned, "total_controls_in_pool": len(pool),
              "controls_by_harm_type": dict(sorted(by_harm_type.items())),
              "total_cli_reported_cost_usd": round(sum(row["cli_reported_cost_usd"] or 0.0 for row in report_rows), 4)}
    return pool, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--returned-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--share-manifest", type=Path, default=None, help="control_rewrite_share_manifest_*.json")
    parser.add_argument("--output-folder", type=Path, required=True)
    arguments = parser.parse_args(argv)
    manifest = json.loads(arguments.share_manifest.read_text(encoding="utf-8")) if arguments.share_manifest else None
    try:
        with tempfile.TemporaryDirectory() as unpack_folder:
            folders = find_control_rewrite_corpus_folders(arguments.returned_paths, Path(unpack_folder))
            pool, report = combine_returned_shares(folders, manifest)
    except ControlRewriteMergeRefused as error:
        print(f"REFUSING TO RUN: {error}", file=sys.stderr)
        return 1
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    pool_path = arguments.output_folder / f"control_prompt_pool_{len(pool)}_prompts_{stamp}.jsonl"
    pool_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in pool), encoding="utf-8")
    report["share_manifest_sha256"] = sha256_of_file(arguments.share_manifest) if arguments.share_manifest else None
    report["pool_file"] = pool_path.name
    report_path = arguments.output_folder / f"control_prompt_pool_report_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(pool)} control prompts from {len(report['shares'])} shares -> {pool_path}\nReport: {report_path}")
    if report["shares_in_manifest_not_returned"]:
        print(f"Shares in the manifest not returned: {report['shares_in_manifest_not_returned']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
