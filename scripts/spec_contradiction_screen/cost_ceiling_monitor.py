"""Spending-ceiling monitor for a paid pipeline subprocess. Makes no model calls and reads no key.

While the pipeline runs as a subprocess, `watch_pipeline_process` re-reads the run's own `cost_log.jsonl` (the sum of
`cost_usd` over its rows, as priced by the pipeline's own tables) every few seconds. The pipeline runs `workers` calls in
parallel, so calls already in flight finish after a stop is requested. The stop therefore fires at the ceiling minus a
margin: `IN_FLIGHT_CALLS_MARGIN` (8, the default worker count) times the most expensive single call seen so far.
At the stop the process is terminated, waited for, and killed if it does not exit.
"""

import json
import subprocess
import time
from pathlib import Path

IN_FLIGHT_CALLS_MARGIN = 8
SECONDS_BETWEEN_COST_LOG_READS = 5
SECONDS_TO_WAIT_AFTER_TERMINATE = 60


def read_cost_log_totals(run_cost_log_paths: list[Path]) -> tuple[float, float]:
    """(total cost in dollars, most expensive single call in dollars) over every readable row of the given logs."""
    total = 0.0
    most_expensive_call = 0.0
    for path in run_cost_log_paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                cost = float(json.loads(line).get("cost_usd") or 0.0)
            except (ValueError, TypeError, AttributeError):
                continue  # a row being written at this moment, or not a cost row
            total += cost
            most_expensive_call = max(most_expensive_call, cost)
    return total, most_expensive_call


def stop_threshold_usd(ceiling_usd: float, most_expensive_call_usd: float) -> float:
    return ceiling_usd - IN_FLIGHT_CALLS_MARGIN * most_expensive_call_usd


def stop_process(process, seconds_to_wait: float = SECONDS_TO_WAIT_AFTER_TERMINATE) -> str:
    """Terminate, wait, kill if still alive. Returns how it ended: 'terminated' or 'killed'."""
    process.terminate()
    try:
        process.wait(timeout=seconds_to_wait)
        return "terminated"
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return "killed"


def watch_pipeline_process(process, find_cost_logs, ceiling_usd: float | None,
                           seconds_between_reads: float = SECONDS_BETWEEN_COST_LOG_READS,
                           sleep=time.sleep, seconds_to_wait_after_terminate: float = SECONDS_TO_WAIT_AFTER_TERMINATE) -> dict:
    """Wait for the process; with a ceiling, stop it once the run's cost reaches the ceiling minus the margin.
    `find_cost_logs` is called each read (the run folder appears only after the pipeline starts). Returns a record
    of what happened, for the launch record."""
    while process.poll() is None:
        if ceiling_usd is not None:
            total, most_expensive_call = read_cost_log_totals(find_cost_logs())
            threshold = stop_threshold_usd(ceiling_usd, most_expensive_call)
            if total >= threshold:
                how = stop_process(process, seconds_to_wait_after_terminate)
                final_total, _ = read_cost_log_totals(find_cost_logs())
                return {"stopped_by_cost_ceiling": True, "ceiling_usd": ceiling_usd,
                        "margin_usd": ceiling_usd - threshold, "stop_threshold_usd": threshold,
                        "cost_when_stop_requested_usd": total, "cost_after_process_ended_usd": final_total,
                        "process_ended_by": how, "return_code": process.returncode}
        sleep(seconds_between_reads)
    final_total, _ = read_cost_log_totals(find_cost_logs()) if ceiling_usd is not None else (None, None)
    return {"stopped_by_cost_ceiling": False, "ceiling_usd": ceiling_usd, "cost_at_exit_usd": final_total,
            "return_code": process.returncode}
