"""Diagnose the extra ~32,000-token cached block that about a quarter of judge calls carry.

Hypothesis: the Claude Code CLI fetches the account's claude.ai connectors (Slack, Gmail, Google Docs, ...) at random
points during start-up, so some calls load their tool definitions and others do not. This script makes the same
usefulness-judge call on the same prompts under two conditions and compares what each call loaded and cost:
  old_options  no strict MCP configuration and no environment flag (what the judges used before the fix)
  new_options  empty MCP servers, strict MCP configuration, ENABLE_CLAUDEAI_MCP_SERVERS=false (what they use now)

Each condition makes --calls-per-condition parallel calls (default 8) on prompts from an accepted prompt file of the
corpus. At most 16 calls are ever made. Billed to the Claude subscription through the CLI, never an API key (the
paid-API environment variables are blanked in both conditions).

Nothing is called unless --confirm-model-calls is passed; without it the plan is printed and the script exits.

Outputs go to data/emergent_misalignment_runs/small_sample_checks/claude_code_cli_context_bloat_diagnostic_<stamp>/ :
  per_call_context_and_cost_log_<stamp>.jsonl   one row per call (appended as calls finish)
  condition_comparison_summary_<stamp>.json     the printed comparison
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))
import claude_code_cli_structured_reply_without_paid_api_fallback as cli_wrapper  # noqa: E402
import judge_whether_prompt_carries_an_evident_and_decisive_animal_welfare_stake_through_claude_code_cli as usefulness_judge  # noqa: E402
from filter_prompts_where_user_never_mentions_animal_welfare_with_haiku_through_claude_code_cli import load_prompt_records  # noqa: E402

MAXIMUM_TOTAL_CALLS = 16
DEFAULT_CALLS_PER_CONDITION = 8
CONDITION_NAMES = ("old_options", "new_options")
DEFAULT_PROMPTS_FILE = (REPOSITORY_ROOT / "data/emergent_misalignment_prompt_corpus/batches/"
                        "batch_001_seed_20270001_2026-11-06_01-09/accepted_prompts_batch_001_2026-11-06_01-27.jsonl")
OUTPUT_PARENT_FOLDER = REPOSITORY_ROOT / "data/emergent_misalignment_runs/small_sample_checks"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"


def call_one_judge(condition_name: str, prompt_record: dict, judge_model: str) -> dict:
    """One real usefulness-judge call under the named condition; returns the log row."""
    started = time.time()
    row = {"condition": condition_name, "prompt_gid": prompt_record["prompt_gid"], "judge_model": judge_model}
    try:
        reply = cli_wrapper.call_claude_code_cli_for_structured_reply(
            judge_model, usefulness_judge.render_judge_prompt(prompt_record), usefulness_judge.JSON_SCHEMA,
            shut_out_claude_ai_connectors=(condition_name == "new_options"))
        row.update(
            input_tokens=reply.input_tokens, cache_read_input_tokens=reply.cache_read_input_tokens,
            cache_creation_input_tokens=reply.cache_creation_input_tokens, output_tokens=reply.output_tokens,
            cli_reported_cost_usd=reply.reported_cost_usd, number_of_turns=reply.number_of_turns,
            number_of_tools_loaded=reply.number_of_tools_loaded,
            names_of_non_built_in_tools=list(reply.names_of_non_built_in_tools),
            number_of_mcp_servers_loaded=reply.number_of_mcp_servers_loaded, error=None)
    except Exception as error:  # a failed call is a data point, not a reason to stop the diagnostic
        row["error"] = f"{type(error).__name__}: {str(error)[:300]}"
    row["seconds"] = round(time.time() - started, 1)
    row["logged_at"] = datetime.now().isoformat(timespec="seconds")
    return row


def summarise_condition(rows: list[dict]) -> dict:
    ok_rows = [row for row in rows if row.get("error") is None]
    def total(field):
        return sum((row[field] or 0) for row in ok_rows)
    big_cache_rows = [row for row in ok_rows if row["cache_creation_input_tokens"] + row["cache_read_input_tokens"] > 20000]
    return {
        "calls": len(rows), "failed_calls": len(rows) - len(ok_rows),
        "calls_with_over_20000_cached_tokens": len(big_cache_rows),
        "total_input_tokens": total("input_tokens"),
        "total_cache_read_input_tokens": total("cache_read_input_tokens"),
        "total_cache_creation_input_tokens": total("cache_creation_input_tokens"),
        "total_cost_usd": round(total("cli_reported_cost_usd"), 4),
        "mean_cost_usd": round(total("cli_reported_cost_usd") / len(ok_rows), 4) if ok_rows else None,
        "turns_per_call": sorted(row["number_of_turns"] for row in ok_rows),
        "tools_loaded_per_call": sorted(row["number_of_tools_loaded"] or 0 for row in ok_rows),
        "mcp_servers_loaded_per_call": sorted(row["number_of_mcp_servers_loaded"] or 0 for row in ok_rows),
        "non_built_in_tool_names_seen": sorted({name for row in ok_rows for name in row["names_of_non_built_in_tools"]}),
    }


def print_comparison(summary_per_condition: dict) -> None:
    print("\nComparison (cost is as reported by the CLI, notional under a subscription)")
    for condition_name, summary in summary_per_condition.items():
        print(f"\n[{condition_name}]")
        for key, value in summary.items():
            print(f"  {key}: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--calls-per-condition", type=int, default=DEFAULT_CALLS_PER_CONDITION)
    parser.add_argument("--judge-model", default=usefulness_judge.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--confirm-model-calls", action="store_true",
                        help="required to make any call; without it only the plan is printed")
    arguments = parser.parse_args(argv)

    total_calls = arguments.calls_per_condition * len(CONDITION_NAMES)
    if arguments.calls_per_condition < 1 or total_calls > MAXIMUM_TOTAL_CALLS:
        sys.exit(f"refusing: {total_calls} calls requested, the hard cap is {MAXIMUM_TOTAL_CALLS} in total")
    prompt_records = load_prompt_records(arguments.prompts_file)[:arguments.calls_per_condition]
    if len(prompt_records) < arguments.calls_per_condition:
        sys.exit(f"{arguments.prompts_file} has only {len(prompt_records)} prompts")

    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    output_folder = OUTPUT_PARENT_FOLDER / f"claude_code_cli_context_bloat_diagnostic_{stamp}"
    print(f"Plan: {total_calls} usefulness-judge calls with {arguments.judge_model} through the Claude Code CLI "
          f"(subscription, no API key), {arguments.calls_per_condition} in parallel per condition, the same "
          f"{len(prompt_records)} prompts from {arguments.prompts_file.name} under each of {list(CONDITION_NAMES)}.")
    print(f"Log folder: {output_folder}")
    if not arguments.confirm_model_calls:
        print("No calls made. Pass --confirm-model-calls to run.")
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / f"per_call_context_and_cost_log_{stamp}.jsonl"
    rows_per_condition = {name: [] for name in CONDITION_NAMES}
    write_lock = threading.Lock()
    with open(log_path, "a", encoding="utf-8") as log_file:
        for condition_name in CONDITION_NAMES:  # one condition at a time, so the two never share a start-up window
            print(f"Running {len(prompt_records)} calls under {condition_name} ...", flush=True)
            with ThreadPoolExecutor(max_workers=len(prompt_records)) as pool:
                futures = [pool.submit(call_one_judge, condition_name, record, arguments.judge_model)
                           for record in prompt_records]
                for future in as_completed(futures):
                    row = future.result()
                    rows_per_condition[condition_name].append(row)
                    with write_lock:
                        log_file.write(json.dumps(row) + "\n")
                        log_file.flush()
    summary = {name: summarise_condition(rows) for name, rows in rows_per_condition.items()}
    (output_folder / f"condition_comparison_summary_{stamp}.json").write_text(json.dumps(summary, indent=2))
    print_comparison(summary)


if __name__ == "__main__":
    main()
