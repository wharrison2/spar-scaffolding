"""Add a 21st contributor share holding the specs the main corpus dealt in its batches 1 to 5, as a second pass over them.

Makes no model call. Reads the newest manifest of an existing contributor shares folder, collects every spec id the main
corpus dealt (the spec_outcomes_for_batch_*.jsonl of each batch folder), takes those spec rows from the main corpus's fixed
spec set in the spec set file's order, and writes into a NEW folder, data/emergent_misalignment_contributor_shares_with_share_021_<stamp>/:
  the original shares' files, copied unchanged (checksums verified against the manifest),
  contributor_share_021_of_021_<count>_specs_<stamp>.jsonl, and
  contributor_share_manifest_<stamp>.json with number_of_shares 21, the old shares' entries unchanged, and for share 21 the
  flag specs_already_dealt_in_the_main_corpus (the merge accepts those specs in both the main corpus and this share, since
  the contributor's prompt for each is a second prompt for that spec, not a duplicate).
Share 21's seed base is the splitter's formula for share 21; its safe deal count comes from the offline deal simulation.
The original shares folder is never modified. Run it with the kit builder's --shares-folder pointing at the new folder.
"""

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    DEFAULT_CORPUS_FOLDER, REPOSITORY_ROOT, spec_set_path_for_corpus)
from split_remaining_specs_into_contributor_shares import (  # noqa: E402
    MANIFEST_FILE_PATTERN, TIMESTAMP_FORMAT, find_largest_safe_deal_count, first_batch_scenario_seed_for_share,
    read_spec_rows, scenario_seed_base_for_share, sha256_of_file, share_file_name, simulate_deal_of_specs)

NEW_SHARE_NUMBER = 21
MAIN_CORPUS_SEED_RANGE = range(20280001, 20281000)
SECOND_PASS_SEED_BASE_OFFSET = 500
DEFAULT_SHARES_FOLDER = REPOSITORY_ROOT / "data" / "emergent_misalignment_contributor_shares_2026-11-08_15-35"


def collect_spec_ids_dealt_in_main_corpus_batches(corpus_folder: Path) -> set[str]:
    spec_ids = set()
    for outcome_path in sorted((Path(corpus_folder) / "batches").glob("*/spec_outcomes_for_batch_*.jsonl")):
        for line in outcome_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                spec_ids.add(json.loads(line)["spec_id"])
    return spec_ids


def refuse_seed_base_that_could_collide(manifest: dict, seed_base: int) -> None:
    batch_seed = seed_base + 1
    used_bases = {share["scenario_seed_base"] for share in manifest["shares"]}
    used_bases |= {base + SECOND_PASS_SEED_BASE_OFFSET for base in used_bases}
    used_batch_seeds = {base + 1 for base in used_bases}
    if (seed_base in used_bases or batch_seed in used_batch_seeds
            or seed_base in MAIN_CORPUS_SEED_RANGE or batch_seed in MAIN_CORPUS_SEED_RANGE):
        raise SystemExit(f"seed base {seed_base} collides with the main corpus or an existing share's seeds")


def build_new_share_entry_and_rows(manifest: dict, corpus_folder: Path, stamp: str, temporary_folder: Path):
    spec_set_path = spec_set_path_for_corpus(corpus_folder)
    if spec_set_path.name != manifest["source_spec_set_file"] or sha256_of_file(spec_set_path) != manifest["source_spec_set_sha256"]:
        raise SystemExit("the main corpus's spec set is not the one the manifest was built from")
    wanted = collect_spec_ids_dealt_in_main_corpus_batches(corpus_folder)
    rows = [row for row in read_spec_rows(spec_set_path) if row["spec_id"] in wanted]
    if {row["spec_id"] for row in rows} != wanted:
        raise SystemExit("some main-corpus spec ids are missing from the spec set")
    already_in_a_share = wanted & {spec_id for share in manifest["shares"] for spec_id in share["spec_ids"]}
    if already_in_a_share:
        raise SystemExit(f"{len(already_in_a_share)} of those specs are already in an existing share")
    seed_base = scenario_seed_base_for_share(NEW_SHARE_NUMBER)
    refuse_seed_base_that_could_collide(manifest, seed_base)
    batch_seed = first_batch_scenario_seed_for_share(NEW_SHARE_NUMBER)
    trial_path = temporary_folder / "trial_share_021.jsonl"
    trial_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    safe_count, exhausted_category = find_largest_safe_deal_count(trial_path, len(rows), batch_seed)
    dealt = set(simulate_deal_of_specs(trial_path, safe_count, batch_seed))
    assert dealt <= wanted
    entry = {"share_number": NEW_SHARE_NUMBER,
             "share_file": share_file_name(NEW_SHARE_NUMBER, NEW_SHARE_NUMBER, len(rows), stamp),
             "spec_count": len(rows), "scenario_seed_base": seed_base, "first_batch_scenario_seed": batch_seed,
             "safe_deal_count": safe_count, "specs_unused_at_safe_deal_count": len(rows) - safe_count,
             "category_exhausted_when_dealing_the_whole_share": exhausted_category,
             "specs_already_dealt_in_the_main_corpus": True,
             "spec_ids": [row["spec_id"] for row in rows]}
    return entry, rows, sorted(wanted - dealt)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER)
    parser.add_argument("--existing-shares-folder", type=Path, default=DEFAULT_SHARES_FOLDER)
    parser.add_argument("--output-parent-folder", type=Path, default=REPOSITORY_ROOT / "data")
    arguments = parser.parse_args(argv)
    manifest_path = sorted(arguments.existing_shares_folder.glob(MANIFEST_FILE_PATTERN))[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["number_of_shares"] != NEW_SHARE_NUMBER - 1:
        raise SystemExit(f"expected a manifest of {NEW_SHARE_NUMBER - 1} shares, found {manifest['number_of_shares']}")
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    with tempfile.TemporaryDirectory() as temporary_folder:
        entry, rows, unused_spec_ids = build_new_share_entry_and_rows(manifest, arguments.corpus_folder, stamp, Path(temporary_folder))
    output_folder = arguments.output_parent_folder / f"emergent_misalignment_contributor_shares_with_share_021_{stamp}"
    output_folder.mkdir(parents=True)
    for share in manifest["shares"]:
        source = arguments.existing_shares_folder / share["share_file"]
        if sha256_of_file(source) != share["share_file_sha256"]:
            raise SystemExit(f"{source} does not match the manifest checksum")
        shutil.copy2(source, output_folder / share["share_file"])
    new_path = output_folder / entry["share_file"]
    new_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    entry = {**{key: value for key, value in entry.items() if key != "spec_ids"}, "share_file_sha256": sha256_of_file(new_path),
             "spec_ids": entry["spec_ids"]}
    new_manifest = {**manifest, "created_at": datetime.now().isoformat(timespec="seconds"), "number_of_shares": NEW_SHARE_NUMBER,
                    "shares": manifest["shares"] + [entry],
                    "share_021_note": "Share 021 holds the specs the main corpus dealt in batches 1 to 5 (a second prompt per spec); "
                                      "the merge accepts them in both the main corpus and share 021."}
    new_manifest_path = output_folder / f"contributor_share_manifest_{stamp}.json"
    new_manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output_folder}\n  share 021: {entry['spec_count']} specs, seed base {entry['scenario_seed_base']}, "
          f"safe deal count {entry['safe_deal_count']}, unused {entry['specs_unused_at_safe_deal_count']} {unused_spec_ids}, "
          f"category exhausted at full size: {entry['category_exhausted_when_dealing_the_whole_share']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
