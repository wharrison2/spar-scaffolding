"""Split the never-dealt specs of the main emergent misalignment prompt corpus into disjoint contributor shares.

Each share is a spec set file of about 200 specs for one contributor, who deals it in their own corpus folder (see the
contributor kit built by build_emergent_misalignment_contributor_kit.py). Makes no model call: the deal that fixes each
share's safe deal count is the scenario dealer run offline, exactly as pipeline step 1 runs it before any generation.

What it does:
  1. reads the main corpus's fixed spec set (named in the corpus settings file) and finds every spec ever dealt there:
     the spec status table rebuilt from the ledger's batches (any attempt, whatever the outcome, failed ones included),
     the scenario_deals.jsonl of every batch folder (also batches with no ledger line yet), and the dealt and forced
     spec ids of every spec dealing plan (a plan's reserved ids are not counted as dealt);
  2. removes them and splits the rest into numbered shares: within each subcategory the remaining specs, in stored order
     (position_in_subcategory_order), go round-robin to shares 1..N, the turn carried on from one subcategory to the next so
     the shares differ in size by at most one and each keeps the category, subcategory, harm type and practical advantage
     balance. Each share keeps the spec rows exactly as stored, so the dealer's order within a share is the stored order;
  3. gives each share its own scenario seed base (20280000 + share number x 1000; the batch seed is the base plus the batch
     number, 1, so it never meets the main corpus's seeds 20280001-20280999);
  4. finds each share's safe deal count: the largest count not above the share size for which the real dealer, seeded as the
     share's batch will be, deals only specs of the share and never raises SpecPoolExhausted (the dealer fixes categories by
     weight before drawing specs, so asking for the whole share can exhaust one category near the end). The count is searched
     downwards from the share size, because the deal itself changes with the count;
  5. writes the share files, a manifest and a reserved-spec-ids file into the main corpus folder, so the main batch script
     never deals a reserved spec.
Output folder (a new one per run): data/emergent_misalignment_contributor_shares_<stamp>/ with
  contributor_share_<NNN>_of_<N>_<count>_specs_<stamp>.jsonl   one per share
  contributor_share_manifest_<stamp>.json
--dry-run prints the plan and the safe deal counts and writes nothing.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    DEFAULT_CORPUS_FOLDER, REPOSITORY_ROOT, SPEC_DEALING_PLAN_FILE_PATTERN, batch_folders_present, read_corpus_settings,
    spec_set_path_for_corpus)
from load_pipeline_scenario_dealer import load_scenario_dealer  # noqa: E402
from reserved_spec_ids_for_contributor_shares import write_reserved_spec_ids_file  # noqa: E402
from spec_outcomes_and_status_table_for_prompt_corpus import (  # noqa: E402
    build_spec_status_table, find_pipeline_run_folder, read_jsonl_if_exists)

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
DEFAULT_SPECS_PER_SHARE = 200
SCENARIO_SEED_BASE_OF_SHARE_ZERO = 20280000
SCENARIO_SEED_BASE_STEP_PER_SHARE = 1000
BATCH_NUMBER_OF_EVERY_SHARE = 1  # a contributor deals one batch per share, so the batch scenario seed is the base plus 1
DEFAULT_OUTPUT_PARENT = REPOSITORY_ROOT / "data"
MANIFEST_FILE_PATTERN = "contributor_share_manifest_*.json"


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_spec_rows(spec_set_path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(spec_set_path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- which specs were ever dealt ----------

def find_every_spec_ever_dealt(corpus_folder: Path, spec_rows: list[dict]) -> set[str]:
    """Every spec of the spec set that any batch of the corpus folder dealt, whatever its outcome (see the module docstring)."""
    corpus_folder = Path(corpus_folder)
    dealt = {row["spec_id"] for row in build_spec_status_table(corpus_folder, spec_rows) if row["attempts"] > 0}
    for batch_folder in batch_folders_present(corpus_folder).values():
        run_folder = find_pipeline_run_folder(batch_folder)
        if run_folder is not None:
            dealt |= {deal["spec_id"] for deal in read_jsonl_if_exists(run_folder / "step1" / "scenario_deals.jsonl")
                      if deal.get("spec_id")}
        for plan_path in sorted(batch_folder.glob(SPEC_DEALING_PLAN_FILE_PATTERN)):
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            dealt |= set(plan.get("dealt_spec_ids", [])) - set(plan.get("reserved_spec_ids", []))
            dealt |= set(plan.get("forced_spec_ids") or [])
    return dealt & {row["spec_id"] for row in spec_rows}


# ---------- the split ----------

def number_of_shares_for(remaining_count: int, specs_per_share: int) -> int:
    return max(1, round(remaining_count / specs_per_share))


def split_specs_into_shares(remaining_specs: list[dict], number_of_shares: int) -> dict[int, list[dict]]:
    """Round-robin within each subcategory in stored order, the turn carried on between subcategories (see the docstring).
    Each share's rows come back in the original spec set file order."""
    file_order = {spec["spec_id"]: index for index, spec in enumerate(remaining_specs)}
    by_subcategory: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for spec in remaining_specs:
        by_subcategory[(spec["taxa_category"], spec["taxa_subcategory"])].append(spec)
    shares: dict[int, list[dict]] = {number: [] for number in range(1, number_of_shares + 1)}
    next_turn = 0
    for key in sorted(by_subcategory):
        for spec in sorted(by_subcategory[key], key=lambda row: row["position_in_subcategory_order"]):
            shares[next_turn % number_of_shares + 1].append(spec)
            next_turn += 1
    return {number: sorted(rows, key=lambda row: file_order[row["spec_id"]]) for number, rows in shares.items()}


def scenario_seed_base_for_share(share_number: int) -> int:
    return SCENARIO_SEED_BASE_OF_SHARE_ZERO + share_number * SCENARIO_SEED_BASE_STEP_PER_SHARE


def first_batch_scenario_seed_for_share(share_number: int) -> int:
    return scenario_seed_base_for_share(share_number) + BATCH_NUMBER_OF_EVERY_SHARE


# ---------- offline deals ----------

def simulate_deal_of_specs(spec_set_path: Path, deal_count: int, scenario_seed: int) -> list[str]:
    """The spec ids the real scenario dealer deals for this spec set, count and seed, with nothing dealt before. Raises the
    dealer's SpecPoolExhausted when a dealt category has no spec left. Makes no model call."""
    compose_scenarios = load_scenario_dealer()
    scenarios = compose_scenarios.deal_scenarios(deal_count, random.Random(scenario_seed),
                                                 compose_scenarios.DEFAULT_VARIABLES, Path(spec_set_path), None)
    return [scenario["spec_id"] for scenario in scenarios]


def find_largest_safe_deal_count(spec_set_path: Path, share_size: int, scenario_seed: int) -> tuple[int, str | None]:
    """(largest count up to share_size that deals without exhausting a category, the category that exhausted at
    share_size, or None when the whole share can be dealt)."""
    compose_scenarios = load_scenario_dealer()
    exhausted_category_at_full_size = None
    for deal_count in range(share_size, 0, -1):
        try:
            simulate_deal_of_specs(spec_set_path, deal_count, scenario_seed)
            return deal_count, exhausted_category_at_full_size
        except compose_scenarios.SpecPoolExhausted as error:
            if deal_count == share_size:
                exhausted_category_at_full_size = error.category
    raise RuntimeError(f"no deal count from 1 to {share_size} is safe for seed {scenario_seed}")


# ---------- writing ----------

def share_file_name(share_number: int, number_of_shares: int, spec_count: int, stamp: str) -> str:
    return f"contributor_share_{share_number:03d}_of_{number_of_shares:03d}_{spec_count}_specs_{stamp}.jsonl"


def build_shares_with_safe_deal_counts(remaining_specs: list[dict], number_of_shares: int, folder_for_trial_files: Path,
                                       report=print) -> list[dict]:
    """For every share: its rows, seed base, safe deal count and the unused remainder. Trial spec set files go in the
    given folder, which the caller removes."""
    shares = split_specs_into_shares(remaining_specs, number_of_shares)
    results = []
    for number, rows in shares.items():
        trial_path = Path(folder_for_trial_files) / f"trial_share_{number:03d}.jsonl"
        trial_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        seed = first_batch_scenario_seed_for_share(number)
        safe_count, exhausted_category = find_largest_safe_deal_count(trial_path, len(rows), seed)
        dealt_ids = simulate_deal_of_specs(trial_path, safe_count, seed)
        results.append({"share_number": number, "rows": rows, "scenario_seed_base": scenario_seed_base_for_share(number),
                        "first_batch_scenario_seed": seed, "safe_deal_count": safe_count,
                        "specs_unused_at_safe_deal_count": len(rows) - safe_count,
                        "category_exhausted_when_dealing_the_whole_share": exhausted_category,
                        "spec_ids_dealt_at_safe_deal_count": dealt_ids})
        report(f"share {number:03d}: {len(rows)} specs, safe deal count {safe_count} "
               f"({len(rows) - safe_count} unused; whole share exhausts: {exhausted_category})")
    return results


def write_shares_and_manifest(output_folder: Path, results: list[dict], source_spec_set_path: Path,
                              dealt_count: int, stamp: str) -> Path:
    output_folder.mkdir(parents=True, exist_ok=True)
    manifest_shares = []
    for result in results:
        name = share_file_name(result["share_number"], len(results), len(result["rows"]), stamp)
        path = output_folder / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result["rows"]), encoding="utf-8")
        manifest_shares.append({
            "share_number": result["share_number"], "share_file": name, "share_file_sha256": sha256_of_file(path),
            "spec_count": len(result["rows"]), "scenario_seed_base": result["scenario_seed_base"],
            "first_batch_scenario_seed": result["first_batch_scenario_seed"],
            "safe_deal_count": result["safe_deal_count"],
            "specs_unused_at_safe_deal_count": result["specs_unused_at_safe_deal_count"],
            "category_exhausted_when_dealing_the_whole_share": result["category_exhausted_when_dealing_the_whole_share"],
            "spec_ids": [row["spec_id"] for row in result["rows"]]})
    manifest = {"created_at": datetime.now().isoformat(timespec="seconds"),
                "source_spec_set_file": Path(source_spec_set_path).name,
                "source_spec_set_sha256": sha256_of_file(source_spec_set_path),
                "specs_in_source_spec_set": len(read_spec_rows(source_spec_set_path)),
                "specs_removed_because_already_dealt_in_the_main_corpus": dealt_count,
                "number_of_shares": len(results), "shares": manifest_shares}
    manifest_path = output_folder / f"contributor_share_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER)
    parser.add_argument("--specs-per-share", type=int, default=DEFAULT_SPECS_PER_SHARE)
    parser.add_argument("--output-parent-folder", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and safe deal counts; write nothing")
    parser.add_argument("--do-not-reserve-in-main-corpus", action="store_true",
                        help="write the shares but no reserved-spec-ids file into the main corpus folder")
    arguments = parser.parse_args(argv)

    spec_set_path = spec_set_path_for_corpus(arguments.corpus_folder)
    if spec_set_path is None:
        parser.error("the corpus has no spec set file in its settings")
    spec_rows = read_spec_rows(spec_set_path)
    dealt = find_every_spec_ever_dealt(arguments.corpus_folder, spec_rows)
    remaining = [row for row in spec_rows if row["spec_id"] not in dealt]
    number_of_shares = number_of_shares_for(len(remaining), arguments.specs_per_share)
    print(f"Spec set {spec_set_path.name}: {len(spec_rows)} specs, {len(dealt)} ever dealt, {len(remaining)} remaining "
          f"-> {number_of_shares} shares of about {len(remaining) / number_of_shares:.0f}")
    import tempfile
    with tempfile.TemporaryDirectory() as trial_folder:
        results = build_shares_with_safe_deal_counts(remaining, number_of_shares, Path(trial_folder))
    print(f"Total specs unused at the safe deal counts: {sum(r['specs_unused_at_safe_deal_count'] for r in results)}")
    if arguments.dry_run:
        print("Dry run: nothing written.")
        return 0
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    output_folder = arguments.output_parent_folder / f"emergent_misalignment_contributor_shares_{stamp}"
    manifest_path = write_shares_and_manifest(output_folder, results, spec_set_path, len(dealt), stamp)
    print(f"Wrote {number_of_shares} share files and {manifest_path}")
    if not arguments.do_not_reserve_in_main_corpus:
        reserved_path = write_reserved_spec_ids_file(
            arguments.corpus_folder, str(read_corpus_settings(arguments.corpus_folder)["spec_set_file"]),
            sha256_of_file(spec_set_path), {r["share_number"]: [row["spec_id"] for row in r["rows"]] for r in results},
            stamp)
        print(f"Reserved every share's specs in the main corpus: {reserved_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
