"""Description rewrite validation, step 2 (GPT 6.1 Sol): two checks on the rewrites made in step 1.

  problem_gone       for each description GPT labelled problematic (20 of 97) whose rewrite exists: is the
                     labelled problem gone in the rewrite? (A labelled description the model left untouched, or
                     dropped for a dealt-value conflict, has no rewrite: its problem stays, with no call.)
  dealt_values_kept  for every rewritten description (labelled or not): does it still carry every dealt value?
  dealt_values_present_in_original  the same prompt on the ORIGINAL description of every rewritten plan (baseline).

Reads the rewrites of the model given by --rewrite-model (default gpt-6.1-sol) from that model's own subfolder and
writes its checks there. When the rewriter and the checker are the same model family (a GPT rewriter, this GPT
checker), the problem-gone check is not independent: the same family judges its own fix.

Default is a dry run. Paid calls need --run-paid-calls. Resumable per check. Uses the project OPENAI_API_KEY from
emergent_alignment/.env. OpenAI process (not the pipeline's); for example:
    $PY scripts/spec_contradiction_screen/check_description_rewrites_with_gpt.py
    $PY scripts/spec_contradiction_screen/check_description_rewrites_with_gpt.py --run-paid-calls
"""

import argparse
from pathlib import Path

import description_rewrite_check_prompt_rendering as rendering
import openai_resumable_structured_call_runner as runner
import spec_screen_common as common

PROBLEM_GONE_BASE_NAME = "labelled_problem_gone_checks_by_gpt_6_1_sol"
DEALT_VALUES_KEPT_BASE_NAME = "dealt_values_kept_checks_by_gpt_6_1_sol"
DEALT_VALUES_IN_ORIGINAL_BASE_NAME = "dealt_values_present_in_original_checks_by_gpt_6_1_sol"
OLD_GOLD_LABELS_BASE_NAME = "gold_labels_by_gpt_6_1_sol"


def load_inputs(run_folder: Path, rewrite_model: str = common.DEFAULT_REWRITE_MODEL) -> dict:
    """{prompt_id: {variables, description, gold (old label record), rewrite (record or None)}}"""
    plans = {s["prompt_id"]: s for s in common.load_real_specifications(
        run_folder / common.STAGE_SUBFOLDER_NAMES["fresh_specifications"] / "plans.jsonl")}
    gold_path = common.require_stamped_file(run_folder / "1_gold_labels", OLD_GOLD_LABELS_BASE_NAME, "jsonl", "the gold labelling stage")
    gold = {r["prompt_id"]: r for r in common.read_jsonl_file(gold_path)}
    rewrites_path = common.find_stamped_file(common.description_rewrite_stage_folder(run_folder, rewrite_model, create=False),
                                             common.description_rewrites_base_name(rewrite_model), "jsonl")
    rewrites = {r["prompt_id"]: r for r in common.read_jsonl_file(rewrites_path)} if rewrites_path else {}
    return {pid: {"variables": plan["variables"], "description": plan["description"], "gold": gold.get(pid),
                  "rewrite": rewrites.get(pid)} for pid, plan in plans.items()}


def build_problem_gone_tasks(inputs: dict) -> list[dict]:
    tasks = []
    for pid, entry in sorted(inputs.items()):
        gold, rewrite = entry["gold"], entry["rewrite"]
        if gold is None or not gold["incoherent"] or rewrite is None or not rewrite["rewrite_needed"]:
            continue
        system_text, user_text = rendering.render_problem_gone_prompt(
            entry["description"], gold["problem_type"], gold["reason"], rewrite["rewritten_description"])
        tasks.append({"task_key": pid, "task_kind": "problem_gone", "instructions_text": system_text,
                      "input_text": user_text, "validate": rendering.validate_problem_gone})
    return tasks


def build_dealt_values_kept_tasks(inputs: dict) -> list[dict]:
    tasks = []
    for pid, entry in sorted(inputs.items()):
        rewrite = entry["rewrite"]
        if rewrite is None or not rewrite["rewrite_needed"]:
            continue
        system_text, user_text = rendering.render_dealt_values_kept_prompt(entry["variables"], rewrite["rewritten_description"])
        tasks.append({"task_key": pid, "task_kind": "dealt_values_kept", "instructions_text": system_text,
                      "input_text": user_text, "validate": rendering.validate_dealt_values_kept})
    return tasks


def build_dealt_values_in_original_tasks(inputs: dict) -> list[dict]:
    """The same dealt-values prompt as dealt_values_kept, on the ORIGINAL description of every rewritten plan:
    the baseline that shows whether a value was already missing before the rewrite."""
    tasks = []
    for pid, entry in sorted(inputs.items()):
        rewrite = entry["rewrite"]
        if rewrite is None or not rewrite["rewrite_needed"]:
            continue
        system_text, user_text = rendering.render_dealt_values_kept_prompt(entry["variables"], entry["description"])
        tasks.append({"task_key": pid, "task_kind": "dealt_values_present_in_original", "instructions_text": system_text,
                      "input_text": user_text, "validate": rendering.validate_dealt_values_kept})
    return tasks


def run_checks(inputs: dict, run_folder: Path, *, client, reasoning_effort: str, workers: int,
               rewrite_model: str = common.DEFAULT_REWRITE_MODEL) -> dict:
    stage_directory = common.description_rewrite_stage_folder(run_folder, rewrite_model)
    common.record_stage_configuration(
        run_folder, common.description_rewrite_stage_key(rewrite_model).replace("validation", "checks"), [rendering.PROBLEM_GONE_PROMPT_PATH, rendering.DEALT_VALUES_KEPT_PROMPT_PATH],
        {"reasoning_effort": reasoning_effort, "model": runner.MODEL_NAME, "rewrite_model": rewrite_model})
    variable_names = sorted({name for entry in inputs.values() for name in entry["variables"]})
    total = {"ran": 0, "failed": [], "total_cost_dollars": 0.0}
    plans = [
        (PROBLEM_GONE_BASE_NAME, "problem_gone", build_problem_gone_tasks(inputs), rendering.build_problem_gone_schema(),
         lambda task, parsed: {"prompt_id": task["task_key"], "problem_gone": parsed["problem_gone"],
                               "note": parsed["note"], "reasoning": parsed["reasoning"]}),
        (DEALT_VALUES_KEPT_BASE_NAME, "dealt_values_kept", build_dealt_values_kept_tasks(inputs),
         rendering.build_dealt_values_kept_schema(variable_names),
         lambda task, parsed: {"prompt_id": task["task_key"], "all_dealt_values_present": parsed["all_dealt_values_present"],
                               "missing_variables": parsed["missing_variables"], "reasoning": parsed["reasoning"]}),
        (DEALT_VALUES_IN_ORIGINAL_BASE_NAME, "dealt_values_present_in_original", build_dealt_values_in_original_tasks(inputs),
         rendering.build_dealt_values_kept_schema(variable_names),
         lambda task, parsed: {"prompt_id": task["task_key"], "all_dealt_values_present": parsed["all_dealt_values_present"],
                               "missing_variables": parsed["missing_variables"], "reasoning": parsed["reasoning"]}),
    ]
    for base_name, schema_name, tasks, schema, build_record in plans:
        results_path = common.find_or_create_stamped_file(stage_directory, base_name, "jsonl")
        stamp = results_path.stem[-len("2026-10-21_15-13"):]
        summary = runner.run_structured_tasks(
            tasks, client=client, schema_name=schema_name, schema=schema, results_path=results_path,
            errors_path=stage_directory / f"{schema_name}_errors_{stamp}.jsonl",
            usage_log_path=stage_directory / f"{schema_name}_usage_and_cost_{stamp}.json",
            progress_log_path=stage_directory / f"{schema_name}_progress_{stamp}.log",
            build_record=build_record, reasoning_effort=reasoning_effort, workers=workers)
        total["ran"] += summary["ran"]
        total["failed"] += summary["failed"]
        total["total_cost_dollars"] += summary["total_cost_dollars"]
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", default=str(common.FRESH_SPECIFICATIONS_RUN_FOLDER))
    parser.add_argument("--rewrite-model", default=common.DEFAULT_REWRITE_MODEL,
                        help="the model whose rewrites are checked (its own subfolder is read and written)")
    parser.add_argument("--reasoning-effort", default=runner.DEFAULT_REASONING_EFFORT)
    parser.add_argument("--workers", type=int, default=runner.DEFAULT_WORKERS)
    parser.add_argument("--run-paid-calls", action="store_true", help="required to make any paid call")
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    inputs = load_inputs(run_folder, arguments.rewrite_model)
    if not any(e["rewrite"] for e in inputs.values()):
        raise SystemExit(f"No rewrites by {arguments.rewrite_model} yet; run run_description_rewrite_on_fresh_descriptions.py first.")
    print(f"Checks: {len(build_problem_gone_tasks(inputs))} problem-gone calls and {len(build_dealt_values_kept_tasks(inputs))} "
          f"dealt-values-kept calls plus {len(build_dealt_values_in_original_tasks(inputs))} on the originals, model {runner.MODEL_NAME}, reasoning effort {arguments.reasoning_effort}.")
    if not arguments.run_paid_calls:
        print("DRY RUN: no calls made. Pass --run-paid-calls to run (after a cost estimate and William's go-ahead).")
        return
    client = runner.structured_output_client.make_client()
    summary = run_checks(inputs, run_folder, client=client, reasoning_effort=arguments.reasoning_effort,
                         workers=arguments.workers, rewrite_model=arguments.rewrite_model)
    print(f"Done: {summary['ran']} checks, {len(summary['failed'])} failed (rerun to retry), "
          f"cost ${summary['total_cost_dollars']:.2f}")


if __name__ == "__main__":
    main()
