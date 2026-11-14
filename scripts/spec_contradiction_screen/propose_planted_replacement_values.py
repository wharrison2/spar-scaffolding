"""Stage 3a: plan the plants, and have GPT 6.1 Sol choose each plant's replacement value from the current ones.

Each type-plant changes ONE variable of a gold-plausible hand (the variable is fixed by the plant type). GPT
sees the hand, the type of problem to create and the current values of that variable (numbered, in a seeded
shuffled order, never including the current value or a zero-weight value), and answers with an option number.
Code maps the number back to the value. GPT never writes a value. The plan, with each plant's option order,
is saved before any call.

Plants for the deal rules are built in code by apply_planted_values_and_render_hands.py (no call).

Default is a dry run. Paid calls need --run-paid-calls. Resumable per plant. OpenAI process (see
gold_label_dealt_hands.py for how to run it):
    $PY scripts/spec_contradiction_screen/propose_planted_replacement_values.py --run-folder <folder> --run-paid-calls
"""

import argparse
from pathlib import Path

import gold_label_and_plant_prompt_rendering as rendering
import openai_resumable_structured_call_runner as runner
import plant_assignment_and_gold_label_loading as loading
import spec_screen_common as common

ASSIGNMENTS_BASE_NAME = "planted_hand_assignments"
PROPOSALS_BASE_NAME = "planted_replacement_choices_by_gpt_6_1_sol"
DEFAULT_PLANTS_PER_TYPE = 12
DEFAULT_SEED = 20261024


def build_proposal_tasks(assignments: list[dict]) -> list[dict]:
    tasks = []
    for assignment in assignments:
        system_text, user_text = rendering.render_replacement_prompt(
            assignment["base_variable_lines"], assignment["target_variable"], assignment["current_value_text"],
            assignment["planted_type"], assignment["option_texts_in_shown_order"])
        option_count = len(assignment["option_texts_in_shown_order"])
        tasks.append({"task_key": assignment["plant_id"], "task_kind": assignment["planted_type"],
                      "assignment": assignment, "instructions_text": system_text, "input_text": user_text,
                      "option_count": option_count,
                      "validate": (lambda parsed, n=option_count: rendering.validate_replacement_choice(parsed, n))})
    return tasks


def run_proposals(plausible_hands: list[dict], value_options: dict, run_folder: Path, *, client, plants_per_type: int,
                  seed: int, reasoning_effort: str, workers: int) -> dict:
    stage_directory = common.stage_folder(run_folder, "planted_hands")
    common.record_stage_configuration(
        run_folder, "planted_hands_proposals", [rendering.REPLACEMENT_PROMPT_PATH, rendering.PLANT_TYPE_INSTRUCTIONS_PATH],
        {"plants_per_type": plants_per_type, "seed": seed, "reasoning_effort": reasoning_effort})
    assignments_path = common.find_or_create_stamped_file(stage_directory, ASSIGNMENTS_BASE_NAME, "jsonl")
    if assignments_path.is_file():
        assignments = common.read_jsonl_file(assignments_path)
    else:
        assignments = loading.build_assignments_with_options(plausible_hands, value_options, plants_per_type, seed)
        common.write_jsonl_file(assignments_path, assignments)
    results_path = common.find_or_create_stamped_file(stage_directory, PROPOSALS_BASE_NAME, "jsonl")
    stamp = results_path.stem[-len("2026-10-21_15-13"):]

    def build_record(task, parsed):
        assignment = task["assignment"]
        number = parsed["chosen_option_number"]
        return {"plant_id": assignment["plant_id"], "chosen_option_number": number,
                "replacement_value_id": assignment["option_value_ids_in_shown_order"][number - 1],
                "replacement_value_text": assignment["option_texts_in_shown_order"][number - 1],
                "choice_reasoning": parsed["reasoning"]}

    # The answer schema's option-number enum depends on the variable, so tasks run in groups by option count.
    return _run_grouped_by_option_count(build_proposal_tasks(assignments), client, results_path, stage_directory,
                                        stamp, build_record, reasoning_effort, workers)


def _run_grouped_by_option_count(tasks, client, results_path, stage_directory, stamp, build_record,
                                 reasoning_effort, workers) -> dict:
    """The runner takes one schema per call, so tasks are run in groups that share an option count."""
    total = {"ran": 0, "failed": [], "total_cost_dollars": 0.0}
    for option_count in sorted({t["option_count"] for t in tasks}):
        group = [t for t in tasks if t["option_count"] == option_count]
        summary = runner.run_structured_tasks(
            group, client=client, schema_name="replacement_choice",
            schema=rendering.build_replacement_choice_schema(option_count), results_path=results_path,
            errors_path=stage_directory / f"replacement_choice_errors_{stamp}.jsonl",
            usage_log_path=stage_directory / f"replacement_choice_usage_and_cost_{stamp}_options_{option_count}.json",
            progress_log_path=stage_directory / f"replacement_choice_progress_{stamp}.log",
            build_record=build_record, reasoning_effort=reasoning_effort, workers=workers)
        total["ran"] += summary["ran"]
        total["failed"] += summary["failed"]
        total["total_cost_dollars"] += summary["total_cost_dollars"]
    return total


def load_plausible_hands(run_folder: Path) -> list[dict]:
    gold_labels = loading.load_gold_labels(run_folder)
    if not gold_labels:
        raise SystemExit("No gold labels in this run folder yet; run gold_label_dealt_hands.py --stage gold_labels first.")
    return [h for h in loading.load_pinned_hand_pool(run_folder)
            if h["prompt_id"] in gold_labels and not gold_labels[h["prompt_id"]]["extremely_implausible"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--plants-per-type", type=int, default=DEFAULT_PLANTS_PER_TYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reasoning-effort", default=runner.DEFAULT_REASONING_EFFORT)
    parser.add_argument("--workers", type=int, default=runner.DEFAULT_WORKERS)
    parser.add_argument("--run-paid-calls", action="store_true", help="required to make any paid call")
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    plausible = load_plausible_hands(run_folder)
    value_options = loading.load_current_value_options(run_folder)
    total = arguments.plants_per_type * len(common.TYPE_PLANT_TYPE_NAMES)
    print(f"Replacement choices: {total} type-plants ({arguments.plants_per_type} per type) from {len(plausible)} "
          f"gold-plausible hands, model {runner.MODEL_NAME}, reasoning effort {arguments.reasoning_effort}.")
    if not arguments.run_paid_calls:
        print("DRY RUN: no calls made. Pass --run-paid-calls to run (after a cost estimate and William's go-ahead).")
        return
    client = runner.structured_output_client.make_client()
    summary = run_proposals(plausible, value_options, run_folder, client=client, plants_per_type=arguments.plants_per_type,
                            seed=arguments.seed, reasoning_effort=arguments.reasoning_effort, workers=arguments.workers)
    print(f"Done: {summary['ran']} chosen, {len(summary['failed'])} failed (rerun to retry), "
          f"cost ${summary['total_cost_dollars']:.2f}")


if __name__ == "__main__":
    main()
