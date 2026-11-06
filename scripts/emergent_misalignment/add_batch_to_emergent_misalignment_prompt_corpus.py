"""Add one batch of prompts to the append-only emergent misalignment prompt corpus.

Each run of this script generates one batch: it deals new scenarios, drafts and gates prompts (pipeline step 1), keeps
the implicit gate-passed ones, judges each for answerability without welfare and for an evident and decisive welfare
stake, and keeps the prompts that are both answerable and useful. The corpus folder only ever grows; see
README_how_this_corpus_grows.md in it.

Steps (each leaves a completion marker in the batch folder, so --resume-batch-folder skips finished ones):
  snapshot prompts -> write config -> pipeline step 1 -> selection -> answerability judge (twice) ->
  usefulness judge (twice) -> accepted file -> summary -> ledger line (last)
The judges are run twice because a call blocked by safety filters is left for the second run to retry.

Everything goes through the Claude Code CLI (subscription), via the pipeline and the two judge scripts. This script makes
no model call itself. Use --dry-run to print the plan and create nothing.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent.parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from select_implicit_prompts_from_difficult_advice_step_one_run import select_prompts  # noqa: E402
from filter_prompts_where_user_never_mentions_animal_welfare_with_haiku_through_claude_code_cli import (  # noqa: E402
    latest_verdict_per_prompt, read_verdict_records)

PYTHON_EXECUTABLE = str(REPOSITORY_ROOT / "animal-welfare-data-pipeline" / ".venv" / "bin" / "python")
DEFAULT_CORPUS_FOLDER = REPOSITORY_ROOT / "data" / "emergent_misalignment_prompt_corpus"
LEDGER_FILE_NAME = "corpus_batch_ledger_started_2026-11-06.jsonl"
PIPELINE_FOLDER = REPOSITORY_ROOT / "strictly_implicit_difficult_advice_pipeline_copy"
GENERATION_PROMPT_FOLDER = PIPELINE_FOLDER / "prompts" / "dad"
GENERATION_PROMPT_FILE_NAMES = ["step1a_scenario.txt", "step1b_dilemmas.txt", "step1c_gate.txt", "step1d_refine.txt",
                                "variables.txt"]
# The species table (with the relative weights of narrow subcategories) shapes what scenarios are dealt, so it is
# snapshotted next to the prompts and counts toward the generation prompt version hash.
GENERATION_PROFILE_FOLDER = PIPELINE_FOLDER / "dad_pipeline" / "profiles"
GENERATION_PROFILE_FILE_NAME = "animal.py"
# The scenario dealer (including the pass that evens use of category, subcategory, harm type and practical advantage
# combinations across the corpus) decides which scenarios are dealt, so it is snapshotted and hashed the same way.
GENERATION_DEALER_FOLDER = PIPELINE_FOLDER / "dad_pipeline"
GENERATION_DEALER_FILE_NAME = "compose_scenarios.py"
JUDGE_PROMPT_FOLDER = SCRIPT_DIRECTORY / "prompts"
ANSWERABILITY_JUDGE_TEMPLATE_NAME = \
    "judge_whether_a_complete_answer_is_possible_without_mentioning_animal_welfare_prompt_template.txt"
USEFULNESS_JUDGE_TEMPLATE_NAME = \
    "judge_whether_prompt_carries_an_evident_and_decisive_animal_welfare_stake_prompt_template.txt"
ANSWERABILITY_JUDGE_SCRIPT = SCRIPT_DIRECTORY / \
    "judge_whether_a_complete_answer_is_possible_without_mentioning_animal_welfare_through_claude_code_cli.py"
USEFULNESS_JUDGE_SCRIPT = SCRIPT_DIRECTORY / \
    "judge_whether_prompt_carries_an_evident_and_decisive_animal_welfare_stake_through_claude_code_cli.py"
SELECTION_SCRIPT = SCRIPT_DIRECTORY / "select_implicit_prompts_from_difficult_advice_step_one_run.py"
CONFIG_TEMPLATE_PATH = SCRIPT_DIRECTORY / "prompt_corpus_batch_pipeline_config_template.yaml"
PRIOR_USE_COUNTS_FILE_PATTERN = "prior_corpus_use_counts_before_batch_*.json"
PRIOR_USE_COUNTS_CONFIG_KEY = "prior_corpus_use_counts_path"
ANSWERABILITY_VERDICT_PATTERN = "complete_answer_without_welfare_verdicts_*.jsonl"
USEFULNESS_VERDICT_PATTERN = "evident_and_decisive_welfare_stake_verdicts_*.jsonl"
SCENARIO_SEED_BASE = 20270000
SELECTION_COUNT_LARGE_ENOUGH_TO_KEEP_EVERYTHING = 100000
PROGRESS_REPORT_INTERVAL_SECONDS = 180
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
BATCH_FOLDER_PATTERN = re.compile(r"^batch_(\d{3,})_seed_(\d+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")


# ---------- numbering, seeds, hashes ----------

def read_ledger(ledger_path: Path) -> list[dict]:
    if not ledger_path.exists():
        return []
    with open(ledger_path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def batch_numbers_in_ledger(ledger_path: Path) -> set[int]:
    return {int(entry["batch_number"]) for entry in read_ledger(ledger_path)}


def batch_folders_present(corpus_folder: Path) -> dict[int, Path]:
    batches_folder = corpus_folder / "batches"
    found = {}
    if batches_folder.exists():
        for folder in sorted(batches_folder.iterdir()):
            match = BATCH_FOLDER_PATTERN.match(folder.name)
            if folder.is_dir() and match:
                found[int(match.group(1))] = folder
    return found


def next_batch_number(corpus_folder: Path) -> int:
    """One more than the highest number in the ledger or in the batch folder names, so an incomplete batch is never reused."""
    known_numbers = batch_numbers_in_ledger(corpus_folder / LEDGER_FILE_NAME) | set(batch_folders_present(corpus_folder))
    return max(known_numbers, default=0) + 1


def incomplete_batch_folders(corpus_folder: Path) -> list[Path]:
    completed = batch_numbers_in_ledger(corpus_folder / LEDGER_FILE_NAME)
    return [folder for number, folder in batch_folders_present(corpus_folder).items() if number not in completed]


def scenario_seed_for_batch(batch_number: int) -> int:
    return SCENARIO_SEED_BASE + batch_number


def hash_files(file_paths: list[Path]) -> str:
    """sha256 over the contents of the files, in order of file name; first 12 hex characters."""
    digest = hashlib.sha256()
    for path in sorted(file_paths, key=lambda p: p.name):
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:12]


def generation_prompt_version_hash(prompt_folder: Path = GENERATION_PROMPT_FOLDER,
                                   profile_folder: Path = GENERATION_PROFILE_FOLDER,
                                   dealer_folder: Path = GENERATION_DEALER_FOLDER) -> str:
    return hash_files([prompt_folder / name for name in GENERATION_PROMPT_FILE_NAMES]
                      + [profile_folder / GENERATION_PROFILE_FILE_NAME,
                         dealer_folder / GENERATION_DEALER_FILE_NAME])


def judge_prompt_hashes(judge_prompt_folder: Path = JUDGE_PROMPT_FOLDER) -> dict[str, str]:
    return {"answerability_judge_prompt_hash": hash_files([judge_prompt_folder / ANSWERABILITY_JUDGE_TEMPLATE_NAME]),
            "usefulness_judge_prompt_hash": hash_files([judge_prompt_folder / USEFULNESS_JUDGE_TEMPLATE_NAME])}


def batch_folder_name(batch_number: int, seed: int, stamp: str) -> str:
    return f"batch_{batch_number:03d}_seed_{seed}_{stamp}"


# ---------- files for one batch ----------

def snapshot_prompts(batch_folder: Path, generation_prompt_folder: Path = GENERATION_PROMPT_FOLDER,
                     judge_prompt_folder: Path = JUDGE_PROMPT_FOLDER,
                     generation_profile_folder: Path = GENERATION_PROFILE_FOLDER,
                     generation_dealer_folder: Path = GENERATION_DEALER_FOLDER) -> None:
    destination = batch_folder / "generation_and_judge_prompts_used_for_this_batch"
    destination.mkdir(parents=True, exist_ok=True)
    for name in GENERATION_PROMPT_FILE_NAMES:
        shutil.copy2(generation_prompt_folder / name, destination / name)
    shutil.copy2(generation_profile_folder / GENERATION_PROFILE_FILE_NAME, destination / GENERATION_PROFILE_FILE_NAME)
    shutil.copy2(generation_dealer_folder / GENERATION_DEALER_FILE_NAME, destination / GENERATION_DEALER_FILE_NAME)
    for name in (ANSWERABILITY_JUDGE_TEMPLATE_NAME, USEFULNESS_JUDGE_TEMPLATE_NAME):
        shutil.copy2(judge_prompt_folder / name, destination / name)


def count_prior_corpus_use(corpus_folder: Path, batch_number: int) -> dict:
    """How often each (taxa_category, taxa_subcategory, harm_type, practical_advantage) combination was dealt in the
    batches before this one. Counts the deals that passed the gate (what the selection script keeps) from each earlier
    batch folder's pipeline step 1 outputs. A field the batch predates is counted as absent (null), so batch 1 counts
    toward category and subcategory only. Reads only; the new batch is never counted."""
    use_by_combination: dict[tuple, int] = {}
    counted_batches = []
    for earlier_number, folder in batch_folders_present(corpus_folder).items():
        if earlier_number >= batch_number:
            continue
        run_folder = find_pipeline_run_folder(folder)
        if run_folder is None or not (run_folder / "step1" / "dilemmas.jsonl").exists():
            raise BatchFailure(f"earlier batch {earlier_number:03d} has no pipeline step 1 output to count in {folder}")
        gate_passed_deals, _ = select_prompts(run_folder / "step1", SELECTION_COUNT_LARGE_ENOUGH_TO_KEEP_EVERYTHING)
        counted_batches.append(earlier_number)
        for deal in gate_passed_deals:
            combination = (deal.get("taxa_category"), deal.get("taxa_subcategory"), deal.get("harm_type"),
                           deal.get("practical_advantage"))
            use_by_combination[combination] = use_by_combination.get(combination, 0) + 1
    rows = [{"taxa_category": category, "taxa_subcategory": subcategory, "harm_type": harm_type,
             "practical_advantage": practical_advantage, "count": count}
            for (category, subcategory, harm_type, practical_advantage), count
            in sorted(use_by_combination.items(), key=lambda item: tuple(str(part) for part in item[0]))]
    return {"counted_batch_numbers": counted_batches, "deals_counted": sum(use_by_combination.values()),
            "rows": rows}


def write_prior_use_counts_file(batch_folder: Path, corpus_folder: Path, batch_number: int, stamp: str) -> Path:
    """Write (once) the batch's snapshot of prior corpus use. A file already in the batch folder is kept, so a resumed
    batch deals from the same counts and never counts itself."""
    existing = newest_file(batch_folder, PRIOR_USE_COUNTS_FILE_PATTERN)
    if existing is not None:
        return existing
    path = batch_folder / f"prior_corpus_use_counts_before_batch_{batch_number:03d}_{stamp}.json"
    path.write_text(json.dumps(count_prior_corpus_use(corpus_folder, batch_number), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def write_batch_config(batch_folder: Path, seed: int, deals: int, stamp: str,
                       template_path: Path = CONFIG_TEMPLATE_PATH, prior_use_counts_path: Path | None = None) -> Path:
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    config["dad"]["dilemmas"]["scenario_seed"] = seed
    config["dad"]["dilemmas"]["count"] = deals
    config["dad"]["dilemmas"][PRIOR_USE_COUNTS_CONFIG_KEY] = (
        str(prior_use_counts_path) if prior_use_counts_path else None)
    header = (f"# Pipeline configuration for one batch of the emergent misalignment prompt corpus, written from\n"
              f"# {template_path.name}: scenario seed {seed}, {deals} deals, prior corpus use counts "
              f"{prior_use_counts_path.name if prior_use_counts_path else 'none'}.\n")
    config_path = batch_folder / f"pipeline_config_for_this_batch_{stamp}.yaml"
    config_path.write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def newest_file(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    return matches[-1] if matches else None


def find_pipeline_run_folder(batch_folder: Path) -> Path | None:
    runs_folder = batch_folder / "raw_pipeline_outputs" / "dad" / "runs"
    runs = sorted(path for path in runs_folder.iterdir() if path.is_dir()) if runs_folder.exists() else []
    return runs[-1] if runs else None


def marker_exists(batch_folder: Path, step_name: str) -> bool:
    return newest_file(batch_folder / "step_completion_markers", f"{step_name}_completed_*.json") is not None


GENERATION_TIMES_STATE_FILE_NAME = "batch_generation_times_state.json"


def read_generation_times_state(batch_folder: Path) -> dict:
    path = batch_folder / GENERATION_TIMES_STATE_FILE_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_generation_times_state(batch_folder: Path, field: str, value: str, keep_existing: bool) -> None:
    """Record a generation time; with keep_existing, an earlier value (from before a resume) is never overwritten."""
    state = read_generation_times_state(batch_folder)
    if keep_existing and field in state:
        return
    state[field] = value
    (batch_folder / GENERATION_TIMES_STATE_FILE_NAME).write_text(json.dumps(state, indent=2), encoding="utf-8")


def write_marker(batch_folder: Path, step_name: str, seconds_taken: float) -> None:
    folder = batch_folder / "step_completion_markers"
    folder.mkdir(exist_ok=True)
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    (folder / f"{step_name}_completed_{stamp}.json").write_text(
        json.dumps({"step": step_name, "seconds_taken": round(seconds_taken, 1)}), encoding="utf-8")


# ---------- commands ----------

def pipeline_command(config_path: Path, batch_number: int, resume_run_id: str | None = None) -> list[str]:
    command = [PYTHON_EXECUTABLE, "-m", "dad_pipeline.run", "--config", str(config_path), "--step", "1",
               "--stop-after", "1", "--label", f"prompt-corpus-batch-{batch_number:03d}"]
    if resume_run_id:
        command += ["--resume", "--run-id", resume_run_id]
    return command


def selection_command(step_one_folder: Path, batch_folder: Path) -> list[str]:
    return [PYTHON_EXECUTABLE, str(SELECTION_SCRIPT), "--step-one-folder", str(step_one_folder),
            "--output-folder", str(batch_folder), "--count", str(SELECTION_COUNT_LARGE_ENOUGH_TO_KEEP_EVERYTHING)]


def judge_command(judge_script: Path, prompts_file: Path, output_folder: Path) -> list[str]:
    return [PYTHON_EXECUTABLE, str(judge_script), "--prompts-file", str(prompts_file), "--output-folder", str(output_folder)]


def run_command(command: list[str], terminal_output_path: Path, working_folder: Path | None = None,
                environment_additions: dict | None = None) -> int:
    """Run a subprocess, appending its output to the terminal output file; report liveness about every three minutes."""
    import os
    environment = dict(os.environ)
    environment.update(environment_additions or {})
    started = time.time()
    with open(terminal_output_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {' '.join(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(command, cwd=working_folder, env=environment, stdout=log_handle,
                                   stderr=subprocess.STDOUT)
        while True:
            try:
                return process.wait(timeout=PROGRESS_REPORT_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                print(f"    still running after {(time.time() - started) / 60:.1f} minutes: {terminal_output_path.name}",
                      flush=True)


# ---------- verdicts and accepted records ----------

def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ok_verdicts_in_folder(judge_folder: Path, pattern: str) -> dict[str, dict]:
    """The latest ok verdict per prompt across every verdicts file in the folder."""
    files = sorted(judge_folder.glob(pattern)) if judge_folder.exists() else []
    latest = latest_verdict_per_prompt(read_verdict_records(files)) if files else {}
    return {gid: record for gid, record in latest.items() if record.get("parse_status") == "ok"}


def build_accepted_records(prompt_records: list[dict], answerability_verdicts: dict[str, dict],
                           usefulness_verdicts: dict[str, dict], batch_number: int, scenario_seed: int,
                           prompt_version_hash: str, generation_started_at: str | None = None,
                           generation_finished_at: str | None = None) -> tuple[list[dict], dict]:
    """Accepted = answerable yes AND useful true. Prompts with a missing ok verdict from either judge are unjudged."""
    accepted, unjudged_ids = [], []
    counts = {"selected": len(prompt_records), "answerable": 0, "useful": 0, "accepted": 0}
    for record in prompt_records:
        gid = record["prompt_gid"]
        answerability = answerability_verdicts.get(gid)
        usefulness = usefulness_verdicts.get(gid)
        if answerability is None or usefulness is None:
            unjudged_ids.append(gid)
        is_answerable = bool(answerability) and answerability["complete_answer_without_welfare_possible"] == "yes"
        is_useful = bool(usefulness) and usefulness["useful"] is True
        counts["answerable"] += is_answerable
        counts["useful"] += is_useful
        if is_answerable and is_useful:
            accepted.append({**record,
                             "corpus_prompt_id": f"B{batch_number:03d}-{gid}",
                             "corpus_batch_number": batch_number,
                             "scenario_seed": scenario_seed,
                             "generation_prompt_version_hash": prompt_version_hash,
                             "batch_generation_started_at": generation_started_at,
                             "batch_generation_finished_at": generation_finished_at,
                             "answerability_note": answerability.get("note"),
                             "usefulness_note": usefulness.get("note"),
                             "best_answer_ignoring_welfare": usefulness.get("best_answer_ignoring_welfare")})
    counts["accepted"] = len(accepted)
    counts["unjudged_prompt_ids"] = unjudged_ids
    counts["unjudged"] = len(unjudged_ids)
    return accepted, counts


def sum_cost(paths: list[Path], field: str) -> float:
    total = 0.0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += json.loads(line).get(field) or 0.0
    return round(total, 4)


def cost_totals(batch_folder: Path) -> dict:
    pipeline_run = find_pipeline_run_folder(batch_folder)
    pipeline_logs = [pipeline_run / "cost_log.jsonl"] if pipeline_run and (pipeline_run / "cost_log.jsonl").exists() else []
    return {"pipeline_cost_usd": sum_cost(pipeline_logs, "cost_usd"),
            "answerability_judge_cli_reported_cost_usd": sum_cost(
                list((batch_folder / "answerability_judge").glob("*usage_log_*.jsonl")), "cli_reported_cost_usd"),
            "usefulness_judge_cli_reported_cost_usd": sum_cost(
                list((batch_folder / "usefulness_judge").glob("*usage_log_*.jsonl")), "cli_reported_cost_usd")}


def append_ledger_line(ledger_path: Path, entry: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------- the batch ----------

class BatchFailure(Exception):
    pass


def run_batch(corpus_folder: Path, batch_folder: Path, batch_number: int, scenario_seed: int, deals: int,
              started_at: datetime, run_command_function=run_command, now=datetime.now) -> dict:
    ledger_path = corpus_folder / LEDGER_FILE_NAME
    started = time.time()
    stamp = now().strftime(TIMESTAMP_FORMAT)

    def report(message):
        print(f"[{(time.time() - started) / 60:6.1f} min] batch {batch_number:03d}: {message}", flush=True)

    def run_step(step_name, command, working_folder=None, environment_additions=None):
        if marker_exists(batch_folder, step_name):
            report(f"{step_name}: already complete, skipping")
            return
        step_started = time.time()
        if step_name == "pipeline_step_1":
            update_generation_times_state(batch_folder, "batch_generation_started_at",
                                          now().astimezone().isoformat(timespec="seconds"), keep_existing=True)
        report(f"{step_name}: starting")
        terminal_output_path = batch_folder / f"terminal_output_{step_name}_{now().strftime(TIMESTAMP_FORMAT)}.txt"
        return_code = run_command_function(command, terminal_output_path, working_folder, environment_additions)
        if return_code != 0:
            raise BatchFailure(f"{step_name} exited with code {return_code}; see {terminal_output_path}. "
                               f"Resume with --resume-batch-folder {batch_folder}")
        write_marker(batch_folder, step_name, time.time() - step_started)
        if step_name == "pipeline_step_1":
            update_generation_times_state(batch_folder, "batch_generation_finished_at",
                                          now().astimezone().isoformat(timespec="seconds"), keep_existing=False)
        report(f"{step_name}: done in {(time.time() - step_started) / 60:.1f} minutes")

    # snapshot prompts and config
    if not marker_exists(batch_folder, "snapshot_prompts_and_write_config"):
        snapshot_prompts(batch_folder)
        prior_use_counts_path = write_prior_use_counts_file(batch_folder, corpus_folder, batch_number, stamp)
        write_batch_config(batch_folder, scenario_seed, deals, stamp, prior_use_counts_path=prior_use_counts_path)
        write_marker(batch_folder, "snapshot_prompts_and_write_config", 0)
    config_path = newest_file(batch_folder, "pipeline_config_for_this_batch_*.yaml")

    # pipeline step 1, resuming the partial run if one exists
    existing_run = find_pipeline_run_folder(batch_folder)
    resume_run_id = existing_run.name if existing_run and (existing_run / "step1").exists() else None
    run_step("pipeline_step_1", pipeline_command(config_path, batch_number, resume_run_id), PIPELINE_FOLDER,
             {"PIPELINE_OUTPUT_ROOT": str(batch_folder / "raw_pipeline_outputs")})
    step_one_folder = find_pipeline_run_folder(batch_folder) / "step1"

    run_step("selection", selection_command(step_one_folder, batch_folder))
    prompts_file = newest_file(batch_folder, "strictly_implicit_welfare_prompts_*_*.jsonl")
    if prompts_file is None:
        raise BatchFailure("selection produced no prompts file")
    prompt_records = read_jsonl(prompts_file)

    for judge_name, script, folder_name in (("answerability_judge", ANSWERABILITY_JUDGE_SCRIPT, "answerability_judge"),
                                            ("usefulness_judge", USEFULNESS_JUDGE_SCRIPT, "usefulness_judge")):
        for run_number in (1, 2):
            run_step(f"{judge_name}_run_{run_number}", judge_command(script, prompts_file, batch_folder / folder_name))

    report("building accepted prompts")
    snapshot_folder = batch_folder / "generation_and_judge_prompts_used_for_this_batch"
    prompt_version_hash = generation_prompt_version_hash(snapshot_folder, profile_folder=snapshot_folder,
                                                      dealer_folder=snapshot_folder)
    judge_hashes = judge_prompt_hashes(snapshot_folder)
    answerability = ok_verdicts_in_folder(batch_folder / "answerability_judge", ANSWERABILITY_VERDICT_PATTERN)
    usefulness = ok_verdicts_in_folder(batch_folder / "usefulness_judge", USEFULNESS_VERDICT_PATTERN)
    generation_times = read_generation_times_state(batch_folder)
    generation_started_at = generation_times.get("batch_generation_started_at")
    generation_finished_at = generation_times.get("batch_generation_finished_at")
    accepted, counts = build_accepted_records(prompt_records, answerability, usefulness, batch_number, scenario_seed,
                                              prompt_version_hash, generation_started_at, generation_finished_at)
    final_stamp = now().strftime(TIMESTAMP_FORMAT)
    accepted_path = batch_folder / f"accepted_prompts_batch_{batch_number:03d}_{final_stamp}.jsonl"
    accepted_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted), encoding="utf-8")

    gate_passed_implicit = len(prompt_records)
    summary = {"batch_number": batch_number, "scenario_seed": scenario_seed, "deals_requested": deals,
               "implicit_gate_passed_count": gate_passed_implicit, "answerable_count": counts["answerable"],
               "useful_count": counts["useful"], "accepted_count": counts["accepted"],
               "unjudged_count": counts["unjudged"], "unjudged_prompt_ids": counts["unjudged_prompt_ids"],
               "generation_prompt_version_hash": prompt_version_hash, **judge_hashes,
               "batch_generation_started_at": generation_started_at,
               "batch_generation_finished_at": generation_finished_at,
               "started_at": started_at.isoformat(timespec="seconds"), "ended_at": now().isoformat(timespec="seconds"),
               "costs": cost_totals(batch_folder), "accepted_file": accepted_path.name}
    (batch_folder / f"batch_summary_{final_stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ledger_entry = {"batch_number": batch_number,
                    "batch_folder": str(batch_folder.relative_to(corpus_folder)),
                    "scenario_seed": scenario_seed, "deals_requested": deals,
                    "implicit_gate_passed_count": gate_passed_implicit, "answerable_count": counts["answerable"],
                    "useful_count": counts["useful"], "accepted_count": counts["accepted"],
                    "unjudged_count": counts["unjudged"], "accepted_file": accepted_path.name,
                    "generation_prompt_version_hash": prompt_version_hash, **judge_hashes,
                    "batch_generation_started_at": generation_started_at,
                    "batch_generation_finished_at": generation_finished_at,
                    "started_at": summary["started_at"], "ended_at": summary["ended_at"], "costs": summary["costs"]}
    append_ledger_line(ledger_path, ledger_entry)
    report(f"ledger updated: {counts['accepted']} accepted of {gate_passed_implicit} selected "
           f"({counts['unjudged']} unjudged)")
    return ledger_entry


def print_plan(batch_number, seed, batch_folder, deals, corpus_folder) -> None:
    print(f"Batch number: {batch_number}\nScenario seed: {seed}\nBatch folder: {batch_folder}\nDeals requested: {deals}")
    print(f"Generation prompt version hash: {generation_prompt_version_hash()}")
    print(f"Prior corpus use counts (not part of the hash) would be written to: "
          f"{batch_folder / f'prior_corpus_use_counts_before_batch_{batch_number:03d}_<stamp>.json'}")
    for name, value in judge_prompt_hashes().items():
        print(f"{name}: {value}")
    config_placeholder = batch_folder / "pipeline_config_for_this_batch_<stamp>.yaml"
    print("Commands that would run (nothing is created or run):")
    print(f"  cd {PIPELINE_FOLDER} && PIPELINE_OUTPUT_ROOT={batch_folder / 'raw_pipeline_outputs'} "
          f"{' '.join(pipeline_command(config_placeholder, batch_number))}")
    print(f"  {' '.join(selection_command(batch_folder / 'raw_pipeline_outputs/dad/runs/<run id>/step1', batch_folder))}")
    prompts_placeholder = batch_folder / "strictly_implicit_welfare_prompts_<count>_<stamp>.jsonl"
    for script, folder_name in ((ANSWERABILITY_JUDGE_SCRIPT, "answerability_judge"), (USEFULNESS_JUDGE_SCRIPT, "usefulness_judge")):
        for _ in range(2):
            print(f"  {' '.join(judge_command(script, prompts_placeholder, batch_folder / folder_name))}")
    print(f"After success, one line is appended to {corpus_folder / LEDGER_FILE_NAME}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER)
    parser.add_argument("--deals", type=int, default=64, help="number of scenario deals to request (default 64)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the commands; create nothing")
    parser.add_argument("--resume-batch-folder", type=Path, default=None,
                        help="finish a batch folder that has no ledger entry yet")
    arguments = parser.parse_args(argv)
    corpus_folder = arguments.corpus_folder

    if arguments.resume_batch_folder is not None:
        batch_folder = arguments.resume_batch_folder.resolve()
        match = BATCH_FOLDER_PATTERN.match(batch_folder.name)
        if not match or not batch_folder.is_dir():
            parser.error(f"{batch_folder} is not a batch folder named batch_<NNN>_seed_<seed>_<stamp>")
        batch_number, scenario_seed = int(match.group(1)), int(match.group(2))
        if batch_number in batch_numbers_in_ledger(corpus_folder / LEDGER_FILE_NAME):
            parser.error(f"batch {batch_number} is already in the ledger; nothing to resume")
        deals = yaml.safe_load(newest_file(batch_folder, "pipeline_config_for_this_batch_*.yaml").read_text())[
            "dad"]["dilemmas"]["count"] if newest_file(batch_folder, "pipeline_config_for_this_batch_*.yaml") else arguments.deals
    else:
        incomplete = incomplete_batch_folders(corpus_folder)
        if incomplete:
            print("Refusing to start a new batch: these batch folders have no ledger entry:", file=sys.stderr)
            for folder in incomplete:
                print(f"  {folder}", file=sys.stderr)
            print("Resume one with --resume-batch-folder <folder>.", file=sys.stderr)
            return 1
        batch_number = next_batch_number(corpus_folder)
        scenario_seed = scenario_seed_for_batch(batch_number)
        deals = arguments.deals
        batch_folder = corpus_folder / "batches" / batch_folder_name(batch_number, scenario_seed,
                                                                      datetime.now().strftime(TIMESTAMP_FORMAT))

    if arguments.dry_run:
        print_plan(batch_number, scenario_seed, batch_folder, deals, corpus_folder)
        return 0

    batch_folder.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    try:
        run_batch(corpus_folder, batch_folder, batch_number, scenario_seed, deals, started_at)
    except BatchFailure as failure:
        print(f"FAILED: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
