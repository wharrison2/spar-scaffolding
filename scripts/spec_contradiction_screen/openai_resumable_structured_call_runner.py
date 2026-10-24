"""Runs a list of independent GPT structured-output calls with threads, resuming per record.

Each task is one call. A finished task is appended to the results file as one JSON line (with its
usage and cost); a task whose answer fails validation twice, or whose call fails, goes to the errors
file and is NOT recorded as done, so rerunning the stage retries exactly those tasks. Cost, token
usage (input, cached input, output, reasoning) and cache rate are logged per call and in totals.

Imports the project's scripts/shared modules, so it must run in a process that does not also put
the animal-welfare-data-pipeline on the path (that package has its own `shared`).
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import openai_structured_output_client as structured_output_client  # noqa: E402
from shared.api_usage_and_cost_tracking import UsageAndCostTracker, compute_cost_in_dollars  # noqa: E402
from shared.resumable_jsonl_records import AppendOnlyJsonlWriter, read_jsonl  # noqa: E402
from shared.timed_progress_reporter import TimedProgressReporter  # noqa: E402

MODEL_NAME = "gpt-6.1-sol"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_WORKERS = 8


def completed_task_keys(results_path: Path) -> set[str]:
    if not Path(results_path).is_file():
        return set()
    return {record["task_key"] for record in read_jsonl(results_path)}


def run_structured_tasks(tasks: list[dict], *, client, schema_name: str, schema: dict, results_path: Path,
                         errors_path: Path, usage_log_path: Path, progress_log_path: Path,
                         build_record, reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                         workers: int = DEFAULT_WORKERS, model_name: str = MODEL_NAME) -> dict:
    """Run every task not already in the results file. A task is a dict with `task_key`,
    `instructions_text`, `input_text` and `validate` (a callable raising ValueError on a bad answer).
    `build_record(task, parsed_result)` returns the stage's record dict. Returns a summary dict."""
    done_keys = completed_task_keys(results_path)
    pending = [task for task in tasks if task["task_key"] not in done_keys]
    print(f"{len(tasks)} tasks, {len(done_keys & {t['task_key'] for t in tasks})} already done, "
          f"{len(pending)} to run with {model_name} (reasoning effort {reasoning_effort}, {workers} workers)",
          flush=True)
    writer = AppendOnlyJsonlWriter()
    tracker = UsageAndCostTracker(model_name, group_label="per_task_kind")
    usage_note = ("Costs are real dollars from the provider's token counts at the Standard tier prices in "
                  "scripts/shared/api_usage_and_cost_tracking.py. Reasoning tokens are billed as output.")
    reporter = TimedProgressReporter(
        progress_log_path, len(pending), already_done_count=len(done_keys),
        status_suffix_function=lambda: (f"cost so far ${tracker.overall_totals['cost_dollars']:.2f}, "
                                        f"cache rate {tracker.overall_totals['cache_rate']:.0%}"),
        after_report_function=lambda: tracker.write_log_file(usage_log_path, usage_note))
    failures = []

    def run_one_task(task):
        try:
            outcome = structured_output_client.request_structured_output(
                client, model_name, task["instructions_text"], task["input_text"], schema_name, schema,
                reasoning_effort, validate_parsed_result=task["validate"])
        except structured_output_client.StructuredOutputParseFailure as failure:
            tracker.add_usage(task.get("task_kind", "all"), failure.usage)
            writer.append(errors_path, {"task_key": task["task_key"], "error": failure.description,
                                        "raw_output_text": failure.raw_output_text, "usage": failure.usage,
                                        "logged_at": datetime.now().isoformat(timespec="seconds")})
            failures.append(task["task_key"])
            reporter.note_task_finished(failed=True)
            return
        except Exception as error:  # noqa: BLE001  one failing call must not discard its siblings
            writer.append(errors_path, {"task_key": task["task_key"],
                                        "error": f"{type(error).__name__}: {error}",
                                        "logged_at": datetime.now().isoformat(timespec="seconds")})
            failures.append(task["task_key"])
            reporter.note_task_finished(failed=True)
            return
        usage = outcome["usage"]
        tracker.add_usage(task.get("task_kind", "all"), usage)
        record = build_record(task, outcome["parsed_result"])
        record.update({
            "task_key": task["task_key"], "model": model_name, "reasoning_effort": reasoning_effort,
            "usage": usage,
            "cost_dollars": compute_cost_in_dollars(model_name, usage["input_tokens"],
                                                    usage["cached_input_tokens"], usage["output_tokens"]),
            "cache_rate": (usage["cached_input_tokens"] / usage["input_tokens"]) if usage["input_tokens"] else 0.0,
            "parse_attempts": outcome["parse_attempts"], "latency_seconds": outcome["latency_seconds"],
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        writer.append(results_path, record)
        reporter.note_task_finished(failed=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_one_task, pending))
    reporter.finish()
    return {"ran": len(pending) - len(failures), "failed": failures,
            "total_cost_dollars": tracker.overall_totals["cost_dollars"]}
