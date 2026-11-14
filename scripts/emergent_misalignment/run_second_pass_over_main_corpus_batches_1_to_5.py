"""Second pass over main corpus batches 1 to 5: deal the SAME specs again with a different scenario seed, so each spec gets a
second prompt with different non-spec dealt variables (the same idea as `run_my_shares.py --second-pass` for contributor shares).

    python run_second_pass_over_main_corpus_batches_1_to_5.py --dry-run     # offline simulation only, creates nothing permanent
    python run_second_pass_over_main_corpus_batches_1_to_5.py               # the real run: ONE batch, model calls through Claude Code

What it does:
 1. Collects the spec ids dealt by main corpus batches --first-batch to --last-batch (default 1 to 5) from their spec outcomes
    files (only batches that are in the main ledger).
 2. Creates a NEW corpus folder (default data/emergent_misalignment_main_corpus_batches_1_to_5_second_pass_<date and time>/;
    pass the same --corpus-folder to resume) holding a fixed spec set of just those specs, taken from the main corpus's spec
    set, and a corpus settings file with the scenario seed base (default 20280500, the main base plus 500).
 3. Finds the largest safe deal count (at most the number of specs) for that seed by simulating the scenario dealer offline,
    checks the dealt specs are all among the collected ones, and writes how many specs stay unused (and which category
    exhausted) to a record file in the corpus folder.
 4. Runs add_batch_to_emergent_misalignment_prompt_corpus.py as a subprocess with --corpus-folder and --deals. An unfinished
    batch folder (no ledger entry) is resumed with --resume-batch-folder. A second real batch is refused: the second pass is
    exactly one batch.

Exit codes: 0 finished (or dry run), 1 failed or refused, 3 usage limit reached: wait for the reset and rerun the same
command (with the same --corpus-folder), which resumes from the checkpoints.

The model calls are made by the add_batch script through the Claude Code command line tool; this script makes none itself,
and --dry-run runs the add_batch script with its own --dry-run.
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent.parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
from load_pipeline_scenario_dealer import load_scenario_dealer  # noqa: E402

BATCH_SCRIPT = SCRIPT_DIRECTORY / "add_batch_to_emergent_misalignment_prompt_corpus.py"
MAIN_CORPUS_FOLDER = (REPOSITORY_ROOT / "data"
                      / "emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs")
CORPUS_SETTINGS_FILE_NAME = "corpus_settings_scenario_seed_base_and_spec_set_file.json"
LEDGER_FILE_NAME = "corpus_batch_ledger_started_2026-11-06.jsonl"
SPEC_OUTCOMES_FILE_PATTERN = "spec_outcomes_for_batch_*.jsonl"
DEAL_COUNT_RECORD_FILE_NAME_PREFIX = "second_pass_deal_count_found_by_offline_simulation"
DEFAULT_CORPUS_FOLDER_PREFIX = "emergent_misalignment_main_corpus_batches_1_to_5_second_pass_"
DEFAULT_FIRST_BATCH = 1
DEFAULT_LAST_BATCH = 5
SECOND_PASS_SEED_BASE_OFFSET = 500  # the same offset the contributor kit's second pass uses
BATCH_NUMBER_OF_THE_SECOND_PASS = 1  # one batch, so its scenario seed is the seed base plus 1
CONTRIBUTOR_SHARE_SEED_RANGE = range(20281000, 20290000)  # 20281000-20289999
USAGE_LIMIT_EXIT_CODE = 3
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
PYTHON_EXECUTABLE = sys.executable


# ---------- collecting the specs of the main batches ----------

def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main_ledger_entries_in_range(main_corpus_folder: Path, first_batch: int, last_batch: int) -> list[dict]:
    ledger_path = Path(main_corpus_folder) / LEDGER_FILE_NAME
    if not ledger_path.exists():
        raise SystemExit(f"No ledger at {ledger_path}.")
    entries = [entry for entry in read_jsonl(ledger_path) if first_batch <= int(entry["batch_number"]) <= last_batch]
    if not entries:
        raise SystemExit(f"No batch from {first_batch} to {last_batch} is in the ledger {ledger_path}.")
    return sorted(entries, key=lambda entry: int(entry["batch_number"]))


def collect_spec_ids_of_main_batches(main_corpus_folder: Path, first_batch: int, last_batch: int) -> tuple[list[str], list[int]]:
    """(sorted distinct spec ids dealt by the ledgered batches in range, the batch numbers used). Batch folders are found
    through the ledger and their spec outcomes file is found by glob."""
    spec_ids = set()
    batch_numbers = []
    for entry in main_ledger_entries_in_range(main_corpus_folder, first_batch, last_batch):
        batch_folder = Path(main_corpus_folder) / entry["batch_folder"]
        outcome_files = sorted(batch_folder.glob(SPEC_OUTCOMES_FILE_PATTERN))
        if not outcome_files:
            raise SystemExit(f"Batch {entry['batch_number']} ({batch_folder}) has no {SPEC_OUTCOMES_FILE_PATTERN} file.")
        spec_ids.update(row["spec_id"] for row in read_jsonl(outcome_files[-1]))
        batch_numbers.append(int(entry["batch_number"]))
    return sorted(spec_ids), batch_numbers


def newest_main_spec_set_path(main_corpus_folder: Path) -> Path:
    """The spec set file the main corpus's settings file names (the balanced consumption order set)."""
    settings = json.loads((Path(main_corpus_folder) / CORPUS_SETTINGS_FILE_NAME).read_text(encoding="utf-8"))
    return Path(main_corpus_folder) / settings["spec_set_file"]


def read_main_seed_base(main_corpus_folder: Path) -> int:
    return int(json.loads((Path(main_corpus_folder) / CORPUS_SETTINGS_FILE_NAME).read_text(encoding="utf-8"))["scenario_seed_base"])


# ---------- seed checks ----------

def refuse_seed_base_that_could_repeat_a_seed(seed_base: int, main_corpus_folder: Path, first_batch: int, last_batch: int) -> None:
    """Exit when the seed base or its batch seed collides with a seed used by main batches in range (any batch in the main
    ledger, to be safe) or lies in the contributor share range 20281000-20289999."""
    batch_seed = seed_base + BATCH_NUMBER_OF_THE_SECOND_PASS
    main_seeds_used = {int(entry["scenario_seed"]) for entry in read_jsonl(Path(main_corpus_folder) / LEDGER_FILE_NAME)}
    reasons = []
    if batch_seed in main_seeds_used:
        reasons.append(f"its batch seed {batch_seed} was used by a main corpus batch")
    if seed_base in main_seeds_used:
        reasons.append(f"the seed base {seed_base} was used by a main corpus batch")
    for value, description in ((seed_base, "seed base"), (batch_seed, "batch seed")):
        if value in CONTRIBUTOR_SHARE_SEED_RANGE:
            reasons.append(f"the {description} {value} is in the contributor share seed range "
                           f"{CONTRIBUTOR_SHARE_SEED_RANGE[0]}-{CONTRIBUTOR_SHARE_SEED_RANGE[-1]}")
    if reasons:
        raise SystemExit(f"Scenario seed base {seed_base} is refused: {'; '.join(reasons)}.")


# ---------- the new corpus folder ----------

def timestamp_now() -> str:
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def spec_set_file_name_for_second_pass(spec_count: int, stamp: str) -> str:
    return f"fixed_spec_set_{spec_count}_specs_main_corpus_batches_1_to_5_second_pass_{stamp}.jsonl"


def prepare_second_pass_corpus_folder(corpus_folder: Path, spec_ids: list[str], main_spec_set_path: Path, seed_base: int,
                                      stamp: str | None = None) -> Path:
    """Create the corpus folder with the spec set of just these specs and the settings file; idempotent on a rerun.
    An existing folder whose settings name a different seed base, or whose spec file holds a different spec set, is refused.
    Returns the spec set file path."""
    corpus_folder = Path(corpus_folder)
    settings_path = corpus_folder / CORPUS_SETTINGS_FILE_NAME
    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        existing_spec_file = corpus_folder / existing["spec_set_file"]
        if existing["scenario_seed_base"] != seed_base:
            raise SystemExit(f"{settings_path} has scenario seed base {existing['scenario_seed_base']}, not {seed_base}. "
                             f"Rerun with --scenario-seed-base {existing['scenario_seed_base']} (or a new --corpus-folder).")
        if not existing_spec_file.exists() or {row["spec_id"] for row in read_jsonl(existing_spec_file)} != set(spec_ids):
            raise SystemExit(f"{existing_spec_file} is missing or does not hold exactly the {len(spec_ids)} specs of the main "
                             f"batches; refusing to continue in this folder.")
        return existing_spec_file
    wanted = set(spec_ids)
    rows = [row for row in read_jsonl(main_spec_set_path) if row["spec_id"] in wanted]
    if len(rows) != len(wanted):
        raise SystemExit(f"Only {len(rows)} of the {len(wanted)} specs were found in {main_spec_set_path}.")
    spec_file_name = f"fixed_spec_set/{spec_set_file_name_for_second_pass(len(rows), stamp or timestamp_now())}"
    (corpus_folder / "fixed_spec_set").mkdir(parents=True, exist_ok=True)
    with open(corpus_folder / spec_file_name, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    settings_path.write_text(json.dumps({"scenario_seed_base": seed_base, "spec_set_file": spec_file_name}, indent=2),
                             encoding="utf-8")
    return corpus_folder / spec_file_name


# ---------- offline simulation of the deal ----------

def simulate_deal_of_specs(spec_set_path: Path, deal_count: int, scenario_seed: int, dealer_module=None) -> list[str]:
    """The spec ids the real scenario dealer deals (nothing dealt before); same as the kit's simulate_deal_of_specs."""
    dealer = dealer_module or load_scenario_dealer()
    scenarios = dealer.deal_scenarios(deal_count, random.Random(scenario_seed), dealer.DEFAULT_VARIABLES,
                                      Path(spec_set_path), None)
    return [scenario["spec_id"] for scenario in scenarios]


def find_largest_safe_deal_count(spec_set_path: Path, spec_count: int, scenario_seed: int,
                                 dealer_module=None) -> tuple[int, str | None]:
    """(largest count up to spec_count that deals without exhausting a category, the category that exhausted at spec_count or
    None). Searched downwards because the deal changes with the count; same as the kit's find_largest_safe_deal_count."""
    dealer = dealer_module or load_scenario_dealer()
    exhausted_category_at_full_size = None
    for deal_count in range(spec_count, 0, -1):
        try:
            simulate_deal_of_specs(spec_set_path, deal_count, scenario_seed, dealer)
            return deal_count, exhausted_category_at_full_size
        except dealer.SpecPoolExhausted as error:
            if deal_count == spec_count:
                exhausted_category_at_full_size = error.category
    raise SystemExit(f"No deal count from 1 to {spec_count} is safe for scenario seed {scenario_seed}.")


def categories_of_specs(spec_set_path: Path, spec_ids: list[str]) -> dict[str, str]:
    wanted = set(spec_ids)
    return {row["spec_id"]: row["taxa_category"] for row in read_jsonl(spec_set_path) if row["spec_id"] in wanted}


def plan_deal(spec_set_path: Path, spec_ids: list[str], scenario_seed: int, dealer_module=None) -> dict:
    """Safe deal count, the check that dealt specs are a subset of the collected ones, and the specs left unused."""
    safe_count, exhausted_category = find_largest_safe_deal_count(spec_set_path, len(spec_ids), scenario_seed, dealer_module)
    dealt_ids = simulate_deal_of_specs(spec_set_path, safe_count, scenario_seed, dealer_module)
    if not set(dealt_ids) <= set(spec_ids):
        raise SystemExit("The simulated deal would deal specs outside the collected main batch specs.")
    category_of_spec = categories_of_specs(spec_set_path, spec_ids)
    unused_ids = sorted(set(spec_ids) - set(dealt_ids))
    unused_by_category = {}
    for spec_id in unused_ids:
        unused_by_category[category_of_spec[spec_id]] = unused_by_category.get(category_of_spec[spec_id], 0) + 1
    return {"safe_deal_count": safe_count, "category_exhausted_when_dealing_every_spec": exhausted_category,
            "specs_in_second_pass": len(spec_ids), "distinct_specs_dealt": len(set(dealt_ids)),
            "specs_unused_at_safe_deal_count": len(unused_ids), "unused_specs_by_category": unused_by_category,
            "unused_spec_ids": unused_ids}


def write_deal_count_record(corpus_folder: Path, plan: dict, seed_base: int, batches_used: list[int]) -> Path:
    stamp = timestamp_now()
    existing = sorted(Path(corpus_folder).glob(f"{DEAL_COUNT_RECORD_FILE_NAME_PREFIX}_*.json"))
    if existing:
        return existing[-1]  # a rerun keeps the record the first run made (informational; the settings file is the check)
    record_path = Path(corpus_folder) / f"{DEAL_COUNT_RECORD_FILE_NAME_PREFIX}_{stamp}.json"
    record_path.write_text(json.dumps(dict(plan, scenario_seed_base=seed_base,
                                           batch_scenario_seed=seed_base + BATCH_NUMBER_OF_THE_SECOND_PASS,
                                           main_batches_collected=batches_used,
                                           recorded_at=datetime.now().isoformat(timespec="seconds")), indent=2),
                           encoding="utf-8")
    return record_path


# ---------- ledger state and the add_batch command ----------

def ledger_batch_folder_names(corpus_folder: Path) -> set[str]:
    ledger_path = Path(corpus_folder) / LEDGER_FILE_NAME
    return {entry["batch_folder"] for entry in read_jsonl(ledger_path)} if ledger_path.exists() else set()


def unfinished_batch_folders(corpus_folder: Path) -> list[Path]:
    batches_folder = Path(corpus_folder) / "batches"
    if not batches_folder.exists():
        return []
    finished = ledger_batch_folder_names(corpus_folder)
    return [folder for folder in sorted(batches_folder.iterdir())
            if folder.is_dir() and folder.name.startswith("batch_") and f"batches/{folder.name}" not in finished]


def state_of_second_pass_corpus(corpus_folder: Path) -> tuple[str, Path | None]:
    """('complete', None) when the ledger already holds the batch, ('unfinished', folder), or ('not_started', None)."""
    if ledger_batch_folder_names(corpus_folder):
        return "complete", None
    unfinished = unfinished_batch_folders(corpus_folder)
    if len(unfinished) > 1:
        raise SystemExit(f"{corpus_folder} has {len(unfinished)} unfinished batch folders; the second pass is one batch.")
    return ("unfinished", unfinished[0]) if unfinished else ("not_started", None)


def batch_command(corpus_folder: Path, deal_count: int, unfinished_batch_folder: Path | None, dry_run: bool = False) -> list[str]:
    command = [PYTHON_EXECUTABLE, str(BATCH_SCRIPT), "--corpus-folder", str(corpus_folder), "--deals", str(deal_count)]
    if unfinished_batch_folder is not None:
        command += ["--resume-batch-folder", str(unfinished_batch_folder)]
    if dry_run:
        command.append("--dry-run")
    return command


# ---------- main ----------

def print_exit_code_meanings() -> None:
    print("Exit codes: 0 finished; 1 failed or refused; 3 usage limit reached: wait for the limit to reset, then rerun the "
          "SAME command (same --corpus-folder) to resume from the checkpoints.", flush=True)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--main-corpus-folder", type=Path, default=MAIN_CORPUS_FOLDER)
    parser.add_argument("--first-batch", type=int, default=DEFAULT_FIRST_BATCH)
    parser.add_argument("--last-batch", type=int, default=DEFAULT_LAST_BATCH)
    parser.add_argument("--corpus-folder", type=Path, default=None,
                        help="the second pass corpus folder; default is a new dated folder in data/. Pass the existing one to resume")
    parser.add_argument("--scenario-seed-base", type=int, default=None,
                        help="default: the main corpus's seed base plus 500 (20280500)")
    parser.add_argument("--dry-run", action="store_true",
                        help="offline simulation and the add_batch dry run in a temporary folder; creates nothing permanent")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    main_corpus_folder = arguments.main_corpus_folder.resolve()
    seed_base = arguments.scenario_seed_base if arguments.scenario_seed_base is not None \
        else read_main_seed_base(main_corpus_folder) + SECOND_PASS_SEED_BASE_OFFSET
    refuse_seed_base_that_could_repeat_a_seed(seed_base, main_corpus_folder, arguments.first_batch, arguments.last_batch)
    batch_seed = seed_base + BATCH_NUMBER_OF_THE_SECOND_PASS
    spec_ids, batches_used = collect_spec_ids_of_main_batches(main_corpus_folder, arguments.first_batch, arguments.last_batch)
    main_spec_set_path = newest_main_spec_set_path(main_corpus_folder)
    stamp = timestamp_now()
    corpus_folder = (arguments.corpus_folder or REPOSITORY_ROOT / "data" / f"{DEFAULT_CORPUS_FOLDER_PREFIX}{stamp}").resolve()
    print(f"Collected {len(spec_ids)} distinct specs from main batches {batches_used}; scenario seed base {seed_base}, "
          f"batch seed {batch_seed}.", flush=True)

    if arguments.dry_run:
        with tempfile.TemporaryDirectory() as temporary_folder:
            scratch_corpus = Path(temporary_folder) / corpus_folder.name
            spec_set_path = prepare_second_pass_corpus_folder(scratch_corpus, spec_ids, main_spec_set_path, seed_base, stamp)
            plan = plan_deal(spec_set_path, spec_ids, batch_seed)
            print(f"Planned scenario seed {batch_seed}; safe deal count {plan['safe_deal_count']} of {len(spec_ids)} specs "
                  f"(category exhausted when dealing every spec: {plan['category_exhausted_when_dealing_every_spec']}); "
                  f"unused specs {plan['specs_unused_at_safe_deal_count']}; unused by category: "
                  f"{plan['unused_specs_by_category']}", flush=True)
            command = batch_command(scratch_corpus, plan["safe_deal_count"], None, dry_run=True)
            completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
            print((completed.stdout + completed.stderr).replace(str(scratch_corpus), str(corpus_folder)))
            print(f"Dry run: nothing was created. Real corpus folder would be {corpus_folder}")
            print_exit_code_meanings()
            return completed.returncode

    state, unfinished_folder = state_of_second_pass_corpus(corpus_folder) if corpus_folder.exists() else ("not_started", None)
    if state == "complete":
        print(f"{corpus_folder} already has its batch in the ledger: the second pass is exactly one batch and is done.")
        return 0
    spec_set_path = prepare_second_pass_corpus_folder(corpus_folder, spec_ids, main_spec_set_path, seed_base, stamp)
    plan = plan_deal(spec_set_path, spec_ids, batch_seed)
    record_path = write_deal_count_record(corpus_folder, plan, seed_base, batches_used)
    print(f"Safe deal count {plan['safe_deal_count']}; unused specs {plan['specs_unused_at_safe_deal_count']}; record: {record_path}",
          flush=True)
    print_exit_code_meanings()
    print(f"To resume after a stop, rerun this script with --corpus-folder {corpus_folder}", flush=True)
    command = batch_command(corpus_folder, plan["safe_deal_count"], unfinished_folder)
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPOSITORY_ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
