"""Description rewrite validation, step 1: run the pipeline's `rewrite_description` on each of the 97 descriptions of
the fresh planning run, with the model given by --model (default gpt-6.1-sol).

Each description is rewritten (or declined) with its dealt variables rendered exactly as the planner saw
them (`render_variable_lines`). Each model's run goes to its own subfolder of the fresh run folder:
<fresh run folder>/3_description_rewrite_validation_<model>/ (the first run, claude-sonnet-5, stays in
3_description_rewrite_validation/). Resumable per description: a result is appended only after a valid answer,
and a rerun repeats only the rest.

Backends: a model name starting with `gpt-` runs on OpenAI through the pipeline's shared/openai_api.py, with
OPENAI_API_KEY read from emergent_alignment/.env only; any other model runs on the pipeline's Claude `api` backend with
ANTHROPIC_API_KEY read the same way. Either refuses to run if its variable is already set in the OS environment.
Claude is lenient on plausibility (see animal-welfare-data-pipeline/config.yaml), so GPT is the default.
Dry run unless --run-paid-calls is given. Run with the pipeline's virtual environment:
    $PY scripts/spec_contradiction_screen/run_description_rewrite_on_fresh_descriptions.py
    $PY scripts/spec_contradiction_screen/run_description_rewrite_on_fresh_descriptions.py --run-paid-calls
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import build_pinned_hand_pool as pool_builder
import spec_screen_common as common

DEFAULT_REASONING_EFFORT = "medium"  # OpenAI models only
PIPELINE_PROMPT_FILES_RECORDED = ["layer1_description_rewrite.txt", "layer1_description_rewrite_answer_schema.json", "layer1.txt"]


def load_descriptions_with_rendered_variables(plans_path: Path, tables: tuple, renames: dict) -> list[dict]:
    """The described plans, each with its variables mapped to the current wording (a value that cannot be mapped
    fails loudly) and rendered as the planner saw them."""
    values, _, value_ids, _, renderer, _ = tables
    records = []
    for plan in common.load_real_specifications(plans_path):
        mapped, _ = pool_builder.map_variables_to_current_values(plan["variables"], values, value_ids, renames)
        records.append({"prompt_id": plan["prompt_id"], "variables": mapped, "description": plan["description"],
                        "rendered_variable_lines": renderer({"prompt_id": plan["prompt_id"], "variables": mapped})})
    return records


def stored_result_record(record: dict, result, model: str) -> dict:
    return {"prompt_id": record["prompt_id"], "reasoning": result.reasoning, "rewrite_needed": result.rewrite_needed,
            "original_description": record["description"], "rewritten_description": result.rewritten_description,
            "changes": list(result.changes), "dealt_value_conflict": result.dealt_value_conflict,
            "conflict_reason": result.conflict_reason, "model": model,
            "completed_at": datetime.now().isoformat(timespec="seconds")}


def run_rewrites(records: list[dict], results_path: Path, errors_path: Path, progress_log_path: Path,
                 rewrite_function, workers: int, cost_log_path: Path | None = None,
                 model: str = common.DEFAULT_REWRITE_MODEL) -> dict:
    """`rewrite_function(variable_lines, description, item_id)` returns a RewriteResult. Appends one line per
    finished description; a failure goes to the errors file and is retried on rerun."""
    timed_progress = common.load_project_shared_module("timed_progress_reporter")
    done = {r["prompt_id"] for r in common.read_jsonl_file(results_path)} if Path(results_path).is_file() else set()
    pending = [r for r in records if r["prompt_id"] not in done]
    print(f"{len(records)} descriptions, {len(records) - len(pending)} already done, {len(pending)} to rewrite "
          f"({workers} workers)", flush=True)
    lock = threading.Lock()
    failures = []

    def cost_so_far():
        if cost_log_path is None:
            return "cost unknown"
        import claude_api_backend_session as session
        return session.describe_cost_so_far(cost_log_path)

    reporter = timed_progress.TimedProgressReporter(
        progress_log_path, len(pending), already_done_count=len(records) - len(pending), status_suffix_function=cost_so_far)

    def append_line(path, record):
        with lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def rewrite_one(record):
        try:
            result = rewrite_function(record["rendered_variable_lines"], record["description"], record["prompt_id"])
        except Exception as error:  # noqa: BLE001  one failing call must not discard its siblings
            append_line(errors_path, {"prompt_id": record["prompt_id"], "error": f"{type(error).__name__}: {error}",
                                      "logged_at": datetime.now().isoformat(timespec="seconds")})
            failures.append(record["prompt_id"])
            reporter.note_task_finished(failed=True)
            return
        append_line(results_path, stored_result_record(record, result, model))
        reporter.note_task_finished(failed=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(rewrite_one, pending))
    reporter.finish()
    return {"ran": len(pending) - len(failures), "failed": failures}


def build_rewrite_function(stage_directory: Path, stamp: str, model: str, reasoning_effort: str):
    """The pipeline's rewrite_description bound to `model`, with that model's backend started."""
    import claude_api_backend_session as session
    if common.is_openai_model(model):
        cost_log_path = session.start_openai_api_session(stage_directory, stamp)
    else:
        _, cost_log_path = session.start_claude_api_session(stage_directory, stamp, model)
    with common.without_implicit_dotenv_loading():
        from sdf_pipeline import layer1_description_rewrite as rewrite_module
    prompts = rewrite_module.load_rewrite_prompts(common.PIPELINE_PROMPTS_FOLDER)
    if prompts is None:
        raise SystemExit("The pipeline has no layer1_description_rewrite prompt files.")

    def rewrite_function(variable_lines, description, item_id):
        return rewrite_module.rewrite_description(
            variable_lines, description, prompts=prompts, model=model, reasoning_effort=reasoning_effort,
            item_id=item_id)

    rewrite_function.cost_log_path = cost_log_path
    return rewrite_function


def estimate_dry_run_cost(records: list[dict], model: str, pending_count: int) -> str | None:
    """Rough cost of the pending calls for an OpenAI model (None for others): input tokens from the rendered prompts at
    4 characters per token, output at 4000 tokens per call (reasoning included; an assumption, not measured)."""
    if not common.is_openai_model(model):
        return None
    sys.path.append(str(common.PIPELINE_ROOT))
    with common.without_implicit_dotenv_loading():
        from sdf_pipeline import layer1_description_rewrite as rewrite_module
        from shared import openai_api
    prompts = rewrite_module.load_rewrite_prompts(common.PIPELINE_PROMPTS_FOLDER)
    characters = [sum(len(part or "") for part in rewrite_module.build_rewrite_prompt(
        prompts, r["rendered_variable_lines"], r["description"])) for r in records]
    mean_input_tokens = sum(characters) / len(characters) / 4 if characters else 0
    per_call = openai_api.compute_cost_usd(model, int(mean_input_tokens), 0, 4000)
    return (f"Estimated cost for {pending_count} calls on {model}: about ${per_call * pending_count:.2f} "
            f"(about {mean_input_tokens:.0f} input and an assumed 4000 output tokens per call, no caching).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", default=str(common.FRESH_SPECIFICATIONS_RUN_FOLDER))
    parser.add_argument("--model", default=common.DEFAULT_REWRITE_MODEL,
                        help="rewriting model; a name starting with gpt- runs on OpenAI, otherwise on the Claude api")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT, help="OpenAI models only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-paid-calls", action="store_true", help="required to make any model call")
    arguments = parser.parse_args()
    model = arguments.model
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    plans_path = run_folder / common.STAGE_SUBFOLDER_NAMES["fresh_specifications"] / "plans.jsonl"
    tables = pool_builder.load_current_tables(common.PIPELINE_PROMPTS_FOLDER)
    records = load_descriptions_with_rendered_variables(plans_path, tables, pool_builder.load_renames())
    stage_key = common.description_rewrite_stage_key(model)
    stage_directory = common.description_rewrite_stage_folder(run_folder, model, create=arguments.run_paid_calls)
    base_name = common.description_rewrites_base_name(model)
    existing = common.find_stamped_file(stage_directory, base_name, "jsonl") if stage_directory.is_dir() else None
    done = len(common.read_jsonl_file(existing)) if existing else 0
    backend = "OpenAI" if common.is_openai_model(model) else "the Claude api backend"
    print(f"Description rewrite: {len(records)} descriptions from {plans_path}, {done} already rewritten, "
          f"model {model} on {backend}, into {stage_directory}.")
    if not arguments.run_paid_calls:
        estimate = estimate_dry_run_cost(records, model, len(records) - done)
        print(f"DRY RUN: {len(records) - done} calls would be made, none were. {estimate or 'No cost estimate for a Claude model.'}")
        print("Pass --run-paid-calls to run (billed to the project key).")
        return
    import claude_api_backend_session as session
    results_path = common.find_or_create_stamped_file(stage_directory, base_name, "jsonl")
    common.record_stage_configuration(
        run_folder, stage_key, [common.PIPELINE_PROMPTS_FOLDER / n for n in PIPELINE_PROMPT_FILES_RECORDED],
        {"backend": "openai" if common.is_openai_model(model) else "api", "model": model,
         "reasoning_effort": arguments.reasoning_effort if common.is_openai_model(model) else None,
         "pipeline_commit": session.pipeline_commit()})
    stamp = results_path.stem[-len("2026-10-21_15-13"):]
    rewrite_function = build_rewrite_function(stage_directory, stamp, model, arguments.reasoning_effort)
    summary = run_rewrites(records, results_path, stage_directory / f"description_rewrite_errors_{stamp}.jsonl",
                           stage_directory / f"description_rewrite_progress_{stamp}.log", rewrite_function,
                           arguments.workers, cost_log_path=rewrite_function.cost_log_path, model=model)
    totals = session.write_usage_and_cost_summary(
        rewrite_function.cost_log_path, stage_directory / f"description_rewrite_usage_and_cost_{stamp}.json",
        "Real dollars from the pipeline cost log. Reasoning tokens are included in output tokens (OpenAI); "
        "they are 0 for Claude because no extended thinking is requested.")
    print(f"Done: {summary['ran']} rewritten or declined, {len(summary['failed'])} failed (rerun to retry); "
          f"cost ${totals['overall']['cost_dollars']:.2f}")


if __name__ == "__main__":
    main()
