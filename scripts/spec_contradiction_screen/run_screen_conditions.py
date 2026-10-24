"""Stage 4: design the validation pool and batches (`design`, offline), then run the Claude screen on them
under each condition (`run`) through the pipeline's own `screen_hands` over the Claude api.

Conditions (same pool, same batches):
  A         batches of 10 hands (the screen as designed); also on the clean batches
  A_repeat  condition A again on identical main batches (stability)
  C         one hand per call: the same prompt with hand_count 1

The pool is the pinned hands (with GPT gold labels) plus the confirmed plants. Main batches take every pool
item (plants about 10% of each batch that has one); 5 clean batches hold only gold-plausible hands. The
design is saved before any call and reused on every later run.

Backend: the pipeline's `api` backend, billed to the project ANTHROPIC_API_KEY, read from emergent_alignment/.env
only (see claude_api_backend_session.py); the run refuses to start if ANTHROPIC_API_KEY is already set in the OS
environment. Model: claude-sonnet-5. Costs are real dollars from the pipeline's cost log.

`run` is a dry run unless --run-paid-calls is given. Resumable per (condition, batch). Run with the pipeline's
virtual environment:
    $PY scripts/spec_contradiction_screen/run_screen_conditions.py design --run-folder <folder>
    $PY scripts/spec_contradiction_screen/run_screen_conditions.py run --run-folder <folder> --run-paid-calls
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import plant_assignment_and_gold_label_loading as loading
import screen_pool_and_batch_design as design
import spec_screen_common as common

ALL_CONDITIONS = ["A", "A_repeat", "C"]
SCREEN_MODEL = "claude-sonnet-5"
BATCH_SIZE = 10
CLEAN_BATCH_COUNT = 5
DESIGN_SEED = 20261023
POOL_BASE_NAME = "screen_validation_pool"
BATCHES_BASE_NAME = "screen_validation_batches"
PLANT_SUMMARY_BASE_NAME = "planted_hand_confirmation_counts"
RESULTS_BASE_NAME = "screen_judgments_by_condition_and_batch"
PIPELINE_PROMPT_FILES_RECORDED = [
    "layer1_screen.txt", "layer1_screen_specification_block.txt", "layer1_screen_answer_schema.json", "layer1.txt"]
ABORT_AFTER_THIS_MANY_FAILURES_WITH_NO_SUCCESS = 8


def read_jsonl_file(path: Path) -> list[dict]:
    return common.read_jsonl_file(path)


def design_pool_and_batches(run_folder: Path, seed: int) -> tuple[Path, Path]:
    """Write the pool and batch files (once; later calls reuse them so batches never change)."""
    stage_directory = common.stage_folder(run_folder, "screen_runs")
    pool_path = common.find_or_create_stamped_file(stage_directory, POOL_BASE_NAME, "jsonl")
    batches_path = common.find_or_create_stamped_file(stage_directory, BATCHES_BASE_NAME, "jsonl")
    if pool_path.is_file() and batches_path.is_file():
        print(f"Reusing existing pool and batches: {batches_path}")
        return pool_path, batches_path
    gold_labels = loading.load_gold_labels(run_folder)
    if not gold_labels:
        raise SystemExit("Need finished gold labels before designing batches.")
    plants, plant_counts = loading.load_confirmed_plants(run_folder)
    hands = [h for h in loading.load_pinned_hand_pool(run_folder) if h["prompt_id"] in gold_labels]
    pool = design.build_pool_items(hands, gold_labels, plants)
    batches = (design.build_main_batches(pool, BATCH_SIZE, seed)
               + design.build_clean_batches(pool, CLEAN_BATCH_COUNT, BATCH_SIZE, seed))
    common.write_jsonl_file(pool_path, pool)
    common.write_jsonl_file(batches_path, batches)
    stamp = pool_path.stem[-len("2026-10-21_15-13"):]
    (stage_directory / f"{PLANT_SUMMARY_BASE_NAME}_{stamp}.json").write_text(json.dumps(plant_counts, indent=2), encoding="utf-8")
    print(f"Designed {len(pool)} pool items ({len(plants)} confirmed plants) in {len(batches)} batches "
          f"({CLEAN_BATCH_COUNT} clean). Plants made, confirmed and discarded, by type: {json.dumps(plant_counts)}. "
          f"Saved before any call: {batches_path}")
    return pool_path, batches_path


def build_claude_screen_function(stage_directory: Path, stamp: str):
    """The pipeline's own screen_hands on the api backend. Imports the pipeline lazily."""
    import claude_api_backend_session as session
    api, cost_log_path = session.start_claude_api_session(stage_directory, stamp, SCREEN_MODEL)
    with common.without_implicit_dotenv_loading():
        from sdf_pipeline import layer1_screen
    prompts = layer1_screen.load_screen_prompts(common.PIPELINE_PROMPTS_FOLDER)
    if prompts is None:
        raise SystemExit("The pipeline has no layer1_screen prompt files.")

    def screen_function(rendered_hands, item_id):
        return layer1_screen.screen_hands(rendered_hands, prompts=prompts, model=SCREEN_MODEL, item_id=item_id)

    screen_function.cost_log_path = cost_log_path
    return screen_function


def completed_unit_keys(results_path: Path) -> set[str]:
    if not Path(results_path).is_file():
        return set()
    return {record["unit_key"] for record in read_jsonl_file(results_path)}


def make_unit_record(unit: dict, pool_by_id: dict, judgments: list, seconds: float) -> dict:
    """One record per (condition, batch) with every judgment mapped to its pool item by position."""
    members = unit["members"]
    if len(judgments) != len(members):
        raise ValueError(f"{unit['unit_key']}: {len(judgments)} judgments for {len(members)} members")
    entries = []
    for judgment in judgments:
        member = members[judgment.number - 1]
        if member["position"] != judgment.number:
            raise ValueError(f"{unit['unit_key']}: member positions are not 1..N")
        item = pool_by_id[member["item_id"]]
        entries.append({"position": member["position"], "item_id": member["item_id"],
                        "is_plant": item["is_plant"], "number": judgment.number,
                        "reasoning": judgment.reasoning, "remove": judgment.remove, "reason": judgment.reason})
    return {"unit_key": unit["unit_key"], "condition": unit["condition"], "batch_id": unit["batch_id"],
            "batch_kind": unit["batch_kind"], "judgments": sorted(entries, key=lambda e: e["position"]),
            "seconds": round(seconds, 2), "completed_at": datetime.now().isoformat(timespec="seconds")}


def run_units(units: list[dict], pool_by_id: dict, results_path: Path, errors_path: Path, progress_log_path: Path,
              screen_function, workers: int, cost_log_path: Path | None = None) -> dict:
    """Run every unit not already in the results file; one appended line per finished unit."""
    timed_progress = common.load_project_shared_module("timed_progress_reporter")
    done_keys = completed_unit_keys(results_path)
    pending = [unit for unit in units if unit["unit_key"] not in done_keys]
    print(f"{len(units)} calls planned, {len(units) - len(pending)} already done, {len(pending)} to run "
          f"({workers} workers)", flush=True)
    append_lock = threading.Lock()
    abort_requested = threading.Event()
    failures, successes = [], []

    def cost_so_far() -> str:
        if cost_log_path is None:
            return "cost unknown"
        import claude_api_backend_session as session
        return session.describe_cost_so_far(cost_log_path)

    reporter = timed_progress.TimedProgressReporter(
        progress_log_path, len(pending), already_done_count=len(units) - len(pending),
        status_suffix_function=cost_so_far)

    def append_line(path, record):
        with append_lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_one_unit(unit):
        if abort_requested.is_set():
            return
        members = [pool_by_id[m["item_id"]] for m in unit["members"]]
        started = time.monotonic()
        try:
            judgments = screen_function([m["rendered_variable_lines"] for m in members], unit["unit_key"])
            record = make_unit_record(unit, pool_by_id, judgments, time.monotonic() - started)
        except Exception as error:  # noqa: BLE001  one failing unit must not discard its siblings
            append_line(errors_path, {"unit_key": unit["unit_key"], "error": f"{type(error).__name__}: {error}",
                                      "logged_at": datetime.now().isoformat(timespec="seconds")})
            failures.append(unit["unit_key"])
            if not successes and len(failures) >= ABORT_AFTER_THIS_MANY_FAILURES_WITH_NO_SUCCESS:
                abort_requested.set()
                print(f"{len(failures)} calls failed and none succeeded: stopping (systemic problem; see {errors_path}).",
                      flush=True)
            reporter.note_task_finished(failed=True)
            return
        append_line(results_path, record)
        successes.append(unit["unit_key"])
        reporter.note_task_finished(failed=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_one_unit, pending))
    reporter.finish()
    return {"ran": len(successes), "failed": failures, "aborted": abort_requested.is_set()}


def load_pool_and_units(pool_path: Path, batches_path: Path, conditions: list[str]):
    pool_by_id = {item["item_id"]: item for item in read_jsonl_file(pool_path)}
    batches = read_jsonl_file(batches_path)
    units = design.build_work_units(pool_by_id, [b for b in batches if b["batch_kind"] == "main"],
                                    [b for b in batches if b["batch_kind"] == "clean"], conditions)
    return pool_by_id, units


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subcommand", choices=["design", "run"])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--seed", type=int, default=DESIGN_SEED)
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-paid-calls", action="store_true", help="required to make any model call")
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    pool_path, batches_path = design_pool_and_batches(run_folder, arguments.seed)
    if arguments.subcommand == "design":
        return
    pool_by_id, units = load_pool_and_units(pool_path, batches_path, arguments.conditions)
    stage_directory = common.stage_folder(run_folder, "screen_runs")
    results_path = common.find_or_create_stamped_file(stage_directory, RESULTS_BASE_NAME, "jsonl")
    done = completed_unit_keys(results_path)
    for condition in arguments.conditions:
        condition_units = [u for u in units if u["condition"] == condition]
        print(f"Condition {condition}: {len(condition_units)} calls, "
              f"{sum(u['unit_key'] in done for u in condition_units)} done")
    if not arguments.run_paid_calls:
        print("DRY RUN: no calls made. Pass --run-paid-calls to run through the Claude api (billed to the project key).")
        return
    import claude_api_backend_session as session
    common.record_stage_configuration(
        run_folder, "screen_runs", [common.PIPELINE_PROMPTS_FOLDER / name for name in PIPELINE_PROMPT_FILES_RECORDED],
        {"backend": "api", "model": SCREEN_MODEL, "batch_size": BATCH_SIZE, "design_seed": arguments.seed,
         "pipeline_commit": session.pipeline_commit()})
    stamp = results_path.stem[-len("2026-10-21_15-13"):]
    screen_function = build_claude_screen_function(stage_directory, stamp)
    summary = run_units(units, pool_by_id, results_path, stage_directory / f"screen_errors_{stamp}.jsonl",
                        stage_directory / f"screen_progress_{stamp}.log", screen_function, arguments.workers,
                        cost_log_path=screen_function.cost_log_path)
    totals = session.write_usage_and_cost_summary(
        screen_function.cost_log_path, stage_directory / f"screen_usage_and_cost_{stamp}.json",
        "Real dollars from the pipeline cost log; reasoning tokens are 0 because no extended thinking is requested.")
    print(f"Done: {summary['ran']} calls this session, {len(summary['failed'])} failed (rerun to retry); "
          f"cost ${totals['overall']['cost_dollars']:.2f}")


if __name__ == "__main__":
    main()
