"""Tally, per model entry and condition, how often the thinking trace considered the welfare cost, the final answer raised it, and the two disagreed.

Reads every judge_verdicts_*.jsonl in a held-out thinking-on `judge_outputs/` folder (a later readable verdict replaces an earlier one for the same answer),
and writes a markdown table next to it. Local only, no API calls.
Usage: python tally_whether_thinking_considered_welfare_cost_from_held_out_thinking_on_judge_verdicts.py <held_out_thinking_on run folder>
"""
import collections
import datetime
import glob
import json
import sys
from pathlib import Path

run_folder = Path(sys.argv[1])
verdict_per_answer = {}
for verdict_file in sorted(glob.glob(str(run_folder / "judge_outputs" / "judge_verdicts_*.jsonl"))):
    for line in open(verdict_file):
        if not line.strip():
            continue
        record = json.loads(line)
        if record["parse_status"] == "ok" or record["answer_id"] not in verdict_per_answer:
            verdict_per_answer[record["answer_id"]] = record

records_per_row = collections.defaultdict(list)
for record in verdict_per_answer.values():
    records_per_row[(record["model_short_name"], record["condition"])].append(record)

table_lines = [
    "| Model entry | Condition | Readable answers | Thinking considered the welfare cost | Final answer raised it | Thinking considered it, answer did not raise it (of all) | Same, of answers whose thinking considered it |",
    "|---|---|---|---|---|---|---|",
]
for (model_name, condition), records in sorted(records_per_row.items()):
    readable = [record for record in records if record["parse_status"] == "ok"]
    considered = sum(record["thinking_considered_welfare_cost"] == "yes" for record in readable)
    raised = sum(record["raised_welfare_cost"] == "yes" for record in readable)
    considered_not_raised = sum(
        record["thinking_considered_welfare_cost"] == "yes" and record["raised_welfare_cost"] != "yes" for record in readable
    )
    count = len(readable)
    table_lines.append(
        f"| {model_name} | {condition} | {count} | {considered / count:.1%} ({considered}/{count}) | {raised / count:.1%} ({raised}/{count}) "
        f"| {considered_not_raised / count:.1%} ({considered_not_raised}/{count}) | {considered_not_raised / max(considered, 1):.1%} ({considered_not_raised}/{considered}) |"
    )

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
output_path = run_folder / f"thinking_considered_welfare_cost_tally_by_model_entry_and_condition_{timestamp}.md"
output_path.write_text("# Thinking considered the welfare cost: held-out set with thinking on\n\n" + "\n".join(table_lines) + "\n")
print(output_path)
print("\n".join(table_lines))
