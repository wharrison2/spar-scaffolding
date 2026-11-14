"""GPT 6.1 Sol judges each dealt hand on its own: is it extremely implausible? Two stages share this code.

  gold_labels         every hand of the pinned pool (stage 1)
  plant_confirmation  every planted hand (stage 3c); a plant is kept only if GPT also judges it
                      extremely implausible, and the rest are discarded and counted

Default is a dry run that prints what would happen. Paid calls need --run-paid-calls. Resumable per hand:
rerun with the same --run-folder and finished judgments are skipped. Uses the project OPENAI_API_KEY from
emergent_alignment/.env (never the OS environment). Run with any Python that has `openai`, in its own
process (not the pipeline's, which shares the package name `shared`); for example the pipeline venv:
    $PY scripts/spec_contradiction_screen/gold_label_dealt_hands.py --stage gold_labels --run-folder <folder> --run-paid-calls
    $PY scripts/spec_contradiction_screen/gold_label_dealt_hands.py --stage plant_confirmation --run-folder <folder> --run-paid-calls
"""

import argparse
from pathlib import Path

import gold_label_and_plant_prompt_rendering as rendering
import openai_resumable_structured_call_runner as runner
import plant_assignment_and_gold_label_loading as loading
import spec_screen_common as common

GOLD_LABELS_BASE_NAME = loading.GOLD_LABELS_BASE_NAME
PLANT_CONFIRMATIONS_BASE_NAME = "planted_hand_confirmations_by_gpt_6_1_sol"
UNCONFIRMED_PLANTS_BASE_NAME = "planted_hands_before_confirmation"


def build_judgment_tasks(items: list[dict], task_kind: str) -> list[dict]:
    """Each item needs `item_id` and `rendered_variable_lines`."""
    tasks = []
    for item in items:
        system_text, user_text = rendering.render_hand_judgment_prompt(item["rendered_variable_lines"])
        tasks.append({"task_key": item["item_id"], "task_kind": task_kind, "instructions_text": system_text,
                      "input_text": user_text, "validate": rendering.validate_hand_judgment})
    return tasks


def run_hand_judgments(items: list[dict], stage_key: str, base_name: str, run_folder: Path, *, client,
                       reasoning_effort: str, workers: int, id_field_name: str) -> dict:
    stage_directory = common.stage_folder(run_folder, stage_key)
    common.record_stage_configuration(
        run_folder, f"{stage_key}_hand_judgments", [rendering.HAND_JUDGMENT_PROMPT_PATH],
        {"reasoning_effort": reasoning_effort, "model": runner.MODEL_NAME})
    results_path = common.find_or_create_stamped_file(stage_directory, base_name, "jsonl")
    stamp = results_path.stem[-len("2026-10-21_15-13"):]

    def build_record(task, parsed):
        return {id_field_name: task["task_key"], "reasoning": parsed["reasoning"],
                "extremely_implausible": parsed["extremely_implausible"], "problem_type": parsed["problem_type"],
                "reason": parsed["reason"]}

    return runner.run_structured_tasks(
        build_judgment_tasks(items, stage_key), client=client, schema_name="hand_judgment",
        schema=rendering.build_hand_judgment_schema(), results_path=results_path,
        errors_path=stage_directory / f"{stage_key}_errors_{stamp}.jsonl",
        usage_log_path=stage_directory / f"{stage_key}_usage_and_cost_{stamp}.json",
        progress_log_path=stage_directory / f"{stage_key}_progress_{stamp}.log",
        build_record=build_record, reasoning_effort=reasoning_effort, workers=workers)


def load_items(stage: str, run_folder: Path) -> tuple[list[dict], str, str, str]:
    """(items, stage folder key, results base name, id field name)."""
    if stage == "gold_labels":
        hands = loading.load_pinned_hand_pool(run_folder)
        return ([{"item_id": h["prompt_id"], "rendered_variable_lines": h["rendered_variable_lines"]} for h in hands],
                "gold_labels", GOLD_LABELS_BASE_NAME, "prompt_id")
    folder = Path(run_folder) / common.STAGE_SUBFOLDER_NAMES["planted_hands"]
    path = common.require_stamped_file(folder, UNCONFIRMED_PLANTS_BASE_NAME, "jsonl", "apply_planted_values_and_render_hands.py")
    plants = common.read_jsonl_file(path)
    return ([{"item_id": p["plant_id"], "rendered_variable_lines": p["rendered_variable_lines"]} for p in plants],
            "planted_hands", PLANT_CONFIRMATIONS_BASE_NAME, "plant_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["gold_labels", "plant_confirmation"])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--reasoning-effort", default=runner.DEFAULT_REASONING_EFFORT)
    parser.add_argument("--workers", type=int, default=runner.DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, help="only the first N hands")
    parser.add_argument("--run-paid-calls", action="store_true", help="required to make any paid call")
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    items, stage_key, base_name, id_field_name = load_items(arguments.stage, run_folder)
    if arguments.limit:
        items = items[:arguments.limit]
    print(f"{arguments.stage}: {len(items)} hands, one call each, model {runner.MODEL_NAME}, reasoning effort "
          f"{arguments.reasoning_effort}, temperature {runner.structured_output_client.SAMPLING_TEMPERATURE}.")
    if not arguments.run_paid_calls:
        print("DRY RUN: no calls made. Pass --run-paid-calls to run (after a cost estimate and William's go-ahead).")
        return
    print(f"Run folder: {run_folder}")
    client = runner.structured_output_client.make_client()
    summary = run_hand_judgments(items, stage_key, base_name, run_folder, client=client,
                                 reasoning_effort=arguments.reasoning_effort, workers=arguments.workers,
                                 id_field_name=id_field_name)
    print(f"Done: {summary['ran']} judged, {len(summary['failed'])} failed (rerun to retry), "
          f"cost ${summary['total_cost_dollars']:.2f}")


if __name__ == "__main__":
    main()
