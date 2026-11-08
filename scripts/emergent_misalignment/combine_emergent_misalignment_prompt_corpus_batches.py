"""Combine the accepted prompts of every completed batch in the emergent misalignment prompt corpus into one snapshot.

Reads the batch ledger, concatenates each listed batch's accepted file in batch order, and drops exact duplicates of the
whitespace-normalised user message (the first occurrence is kept). --only-prompt-version-hash keeps only batches made with
that generation prompt version. By default every prompt listed in a file of <corpus folder>/held_out_set/ (held_out_set_*_prompts_*.jsonl)
is excluded, matched by corpus_prompt_id, by spec_id (so a later regeneration of a held-out spec cannot enter) and by
whitespace-normalised user message; the run refuses to start if a held-out file cannot be read.

Contributor shares: --contributor-corpus-folders takes the unzipped corpus folders that contributors sent back (each named
contributor_corpus_share_NNN, or a parent folder holding several) and --contributor-share-manifest the manifest from
split_remaining_specs_into_contributor_shares.py. The run refuses to start when a contributor folder's spec set is not one
of the manifest's shares (matched by sha256), when two folders hold the same share, when a contributor batch dealt or
accepted a spec outside its share, or when any spec id was dealt in two places (two sources, or twice within one source's
default-mode batches; retry batches may repeat a spec of their own source). Corpora are then combined in the order main
corpus, then shares by number. Corpus prompt ids repeat across contributors (each starts at batch 001 and prompt ids count
from P-0001), so every record gets corpus_source ("main_corpus" or "contributor_share_NNN") and
merged_corpus_prompt_id (the main corpus's corpus_prompt_id unchanged; S<NNN>-<corpus_prompt_id> for a share), refused if
not unique. The held-out set is matched by corpus_prompt_id for the main corpus only (a contributor's B001-P-0038 is not the
main corpus's), and by spec_id and message for every source.
--include-held-out-prompts-in-the-snapshot switches this off loudly (never use it for training data). Makes no model call. Every record keeps its batch_generation_started_at and batch_generation_finished_at fields, and the summary lists each
batch with its generation dates, prompt version hash and accepted count, sorted by batch number. Writes into <corpus folder>/combined_snapshots/:
  combined_prompt_corpus_<N>_prompts_<stamp>.jsonl
  combined_prompt_corpus_summary_<stamp>.json
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    DEFAULT_CORPUS_FOLDER, LEDGER_FILE_NAME, MODE_RETRY, read_corpus_settings, read_ledger, spec_set_path_for_corpus)
from spec_outcomes_and_status_table_for_prompt_corpus import read_batch_spec_outcomes  # noqa: E402

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
HELD_OUT_FOLDER_NAME = "held_out_set"
HELD_OUT_FILE_PATTERN = "held_out_set_*_prompts_*.jsonl"


MAIN_CORPUS_SOURCE_LABEL = "main_corpus"
CONTRIBUTOR_CORPUS_FOLDER_PATTERN = re.compile(r"^contributor_corpus_share_\d{3}$")
SETTINGS_FILE_NAME = "corpus_settings_scenario_seed_base_and_spec_set_file.json"


class HeldOutSetUnreadable(Exception):
    """A held-out set file exists but cannot be read, so exclusion cannot be guaranteed."""


class ContributorMergeRefused(Exception):
    """The contributor corpora cannot be merged safely; the message says why."""


def contributor_source_label(share_number: int) -> str:
    return f"contributor_share_{share_number:03d}"


def expand_contributor_corpus_folders(folders: list[Path]) -> list[Path]:
    """A folder with a settings file is a corpus folder; one without is searched for contributor_corpus_share_NNN folders."""
    expanded = []
    for folder in folders:
        folder = Path(folder)
        if (folder / SETTINGS_FILE_NAME).exists():
            expanded.append(folder)
            continue
        inside = sorted(child for child in folder.iterdir() if child.is_dir()
                        and CONTRIBUTOR_CORPUS_FOLDER_PATTERN.match(child.name)) if folder.is_dir() else []
        if not inside:
            raise ContributorMergeRefused(f"{folder} is neither a corpus folder (no {SETTINGS_FILE_NAME}) nor a folder "
                                          f"holding contributor_corpus_share_NNN folders")
        expanded.extend(inside)
    return expanded


def spec_ids_dealt_in_corpus(corpus_folder: Path) -> tuple[Counter, set[str]]:
    """(how many default-mode batches dealt each spec, every spec any batch dealt), from the ledger's batches."""
    default_mode_counts, every_spec = Counter(), set()
    for entry in read_ledger(Path(corpus_folder) / LEDGER_FILE_NAME):
        outcomes = read_batch_spec_outcomes(Path(corpus_folder) / entry["batch_folder"], int(entry["batch_number"]))
        for spec_id in {row["spec_id"] for row in outcomes}:
            every_spec.add(spec_id)
            if entry.get("batch_mode") != MODE_RETRY:
                default_mode_counts[spec_id] += 1
    return default_mode_counts, every_spec


def validate_contributor_corpora(main_corpus_folder: Path, contributor_folders: list[Path], manifest: dict) -> dict[Path, int]:
    """Check every contributor folder (see the module docstring); returns {folder: share number}. Raises ContributorMergeRefused."""
    share_by_sha = {share["share_file_sha256"]: share for share in manifest["shares"]}
    share_number_by_folder: dict[Path, int] = {}
    owner_of_spec: dict[str, str] = {}
    main_counts, main_specs = spec_ids_dealt_in_corpus(main_corpus_folder)
    for spec_id, count in main_counts.items():
        if count > 1:
            raise ContributorMergeRefused(f"the main corpus dealt {spec_id} in {count} default-mode batches")
    for spec_id in main_specs:
        owner_of_spec[spec_id] = MAIN_CORPUS_SOURCE_LABEL
    for folder in contributor_folders:
        spec_set_path = spec_set_path_for_corpus(folder)
        if spec_set_path is None:
            raise ContributorMergeRefused(f"{folder} has no spec set file in its settings")
        share = share_by_sha.get(hashlib.sha256(spec_set_path.read_bytes()).hexdigest())
        if share is None:
            raise ContributorMergeRefused(f"the spec set of {folder} is not one of the manifest's shares (sha256 differs)")
        if share["share_number"] in share_number_by_folder.values():
            raise ContributorMergeRefused(f"share {share['share_number']} appears in two contributor folders "
                                          f"({folder} and the one before it)")
        if read_corpus_settings(folder)["scenario_seed_base"] != share["scenario_seed_base"]:
            raise ContributorMergeRefused(f"{folder} used scenario seed base {read_corpus_settings(folder)['scenario_seed_base']}"
                                          f", not the share's {share['scenario_seed_base']}")
        share_number_by_folder[folder] = share["share_number"]
        share_specs = set(share["spec_ids"])
        counts, dealt = spec_ids_dealt_in_corpus(folder)
        accepted_spec_ids = set()
        for entry in read_ledger(folder / LEDGER_FILE_NAME):
            for line in (folder / entry["batch_folder"] / entry["accepted_file"]).read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("spec_id"):
                    accepted_spec_ids.add(json.loads(line)["spec_id"])
        outside = sorted((dealt | accepted_spec_ids) - share_specs)
        if outside:
            raise ContributorMergeRefused(f"{folder} dealt or accepted specs outside share {share['share_number']}: {outside[:5]}")
        repeated = sorted(spec_id for spec_id, count in counts.items() if count > 1)
        if repeated:
            raise ContributorMergeRefused(f"{folder} dealt a spec in more than one default-mode batch: {repeated[:5]}")
        label = contributor_source_label(share["share_number"])
        for spec_id in dealt:
            if spec_id in owner_of_spec:
                raise ContributorMergeRefused(f"spec {spec_id} was dealt in both {owner_of_spec[spec_id]} and {label}")
            owner_of_spec[spec_id] = label
    return share_number_by_folder


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
                    include_held_out_prompts: bool = False, contributor_corpus_folders: list[Path] | None = None,
                    contributor_share_manifest: dict | None = None) -> tuple[list[dict], dict]:
    """Combine the main corpus's batches; with contributor corpus folders (and the share manifest), also theirs, after the
    validation described in the module docstring."""
    held_out = read_held_out_identifiers(corpus_folder)
    sources = [(Path(corpus_folder), None)]
    if contributor_corpus_folders:
        if contributor_share_manifest is None:
            raise ContributorMergeRefused("contributor corpus folders need the contributor share manifest")
        folders = expand_contributor_corpus_folders(contributor_corpus_folders)
        share_numbers = validate_contributor_corpora(Path(corpus_folder), folders, contributor_share_manifest)
        sources += sorted(((folder, share_numbers[folder]) for folder in folders), key=lambda pair: pair[1])
    merging_contributors = len(sources) > 1
    combined, seen_messages = [], set()
    summary = {"prompts_per_batch": {}, "prompts_per_prompt_version_hash": Counter(), "prompts_per_taxa_category": Counter(),
               "prompts_per_written_in_english": Counter(), "duplicates_dropped": 0,
               "duplicates_dropped_per_batch": {}, "held_out_prompts_excluded": 0, "held_out_prompts_excluded_per_batch": {},
               "held_out_set_files": held_out["files"], "held_out_prompts_included_in_snapshot": include_held_out_prompts,
               "batches": [], "batches_skipped_by_prompt_version_hash": []}
    seen_merged_prompt_ids: set[str] = set()
    ledger_entries = [(source_folder, share_number, entry) for source_folder, share_number in sources
                      for entry in sorted(read_ledger(source_folder / LEDGER_FILE_NAME), key=lambda e: e["batch_number"])]
    for source_folder, share_number, entry in ledger_entries:
        source_label = MAIN_CORPUS_SOURCE_LABEL if share_number is None else contributor_source_label(share_number)
        batch_label = f"batch_{int(entry['batch_number']):03d}" if share_number is None \
            else f"share_{share_number:03d}_batch_{int(entry['batch_number']):03d}"
        if only_prompt_version_hash and entry["generation_prompt_version_hash"] != only_prompt_version_hash:
            summary["batches_skipped_by_prompt_version_hash"].append(batch_label)
            continue
        accepted_path = source_folder / entry["batch_folder"] / entry["accepted_file"]
        kept_from_batch = dropped_from_batch = held_out_from_batch = 0
        for line in accepted_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = normalise_whitespace(record["user_message"])
            if not include_held_out_prompts and (
                    (share_number is None and record.get("corpus_prompt_id") in held_out["prompt_ids"])
                    or key in held_out["messages"]
                    or (record.get("spec_id") and record["spec_id"] in held_out["spec_ids"])):
                held_out_from_batch += 1
                continue
            if key in seen_messages:
                dropped_from_batch += 1
                continue
            seen_messages.add(key)
            if merging_contributors:
                merged_id = record["corpus_prompt_id"] if share_number is None \
                    else f"S{share_number:03d}-{record['corpus_prompt_id']}"
                if merged_id in seen_merged_prompt_ids:
                    raise ContributorMergeRefused(f"merged prompt id {merged_id} is not unique")
                seen_merged_prompt_ids.add(merged_id)
                record = {**record, "corpus_source": source_label, "merged_corpus_prompt_id": merged_id}
            combined.append(record)
            kept_from_batch += 1
            summary["prompts_per_prompt_version_hash"][record.get("generation_prompt_version_hash")] += 1
            summary["prompts_per_taxa_category"][str(record.get("taxa_category"))] += 1
            summary["prompts_per_written_in_english"][str(record.get("written_in_english"))] += 1
        summary["batches"].append({
            **({"corpus_source": source_label} if merging_contributors else {}),
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
    summary["corpus_sources"] = [MAIN_CORPUS_SOURCE_LABEL if number is None else contributor_source_label(number)
                                 for _, number in sources]
    return combined, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER)
    parser.add_argument("--only-prompt-version-hash", default=None)
    parser.add_argument("--include-held-out-prompts-in-the-snapshot", action="store_true",
                        help="DANGEROUS: keep the held-out evaluation prompts in the snapshot. Never use for training data.")
    parser.add_argument("--contributor-corpus-folders", type=Path, nargs="+", default=None,
                        help="unzipped contributor corpus folders (or parents holding contributor_corpus_share_NNN folders)")
    parser.add_argument("--contributor-share-manifest", type=Path, default=None,
                        help="contributor_share_manifest_*.json from the share splitting script")
    arguments = parser.parse_args(argv)
    if bool(arguments.contributor_corpus_folders) != bool(arguments.contributor_share_manifest):
        parser.error("--contributor-corpus-folders and --contributor-share-manifest go together")
    try:
        manifest = json.loads(arguments.contributor_share_manifest.read_text(encoding="utf-8")) \
            if arguments.contributor_share_manifest else None
        combined, summary = combine_batches(arguments.corpus_folder, arguments.only_prompt_version_hash,
                                            arguments.include_held_out_prompts_in_the_snapshot,
                                            arguments.contributor_corpus_folders, manifest)
    except (HeldOutSetUnreadable, ContributorMergeRefused) as error:
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
