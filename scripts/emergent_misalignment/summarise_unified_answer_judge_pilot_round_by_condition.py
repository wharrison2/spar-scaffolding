"""Summarise one pilot round of the unified answer judges by condition (system prompt versus none), in percentage points.

Usage: python summarise_unified_answer_judge_pilot_round_by_condition.py --pilot-folder FOLDER --round-label round_1 --verdicts-folder-name NAME
Reads FOLDER/<round label>_items and FOLDER/<verdicts folder name>; prints flag rates per characteristic and condition (minor or serious,
and serious), competence ratings, pairwise win/tie/loss of the system prompt answer and the cost of the folder's usage logs.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from unified_answer_judging_prompts_replies_and_quote_checks import MISALIGNMENT_CHARACTERISTICS


def read_all(folder: Path, pattern: str) -> list[dict]:
    return [json.loads(line) for path in sorted(folder.glob(pattern)) for line in path.read_text().splitlines() if line.strip()]


def percent(count: int, total: int) -> str:
    return f"{100 * count / total:.0f}%" if total else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-folder", type=Path, required=True)
    parser.add_argument("--round-label", required=True)
    parser.add_argument("--verdicts-folder-name", required=True)
    arguments = parser.parse_args()
    condition_of = json.loads((arguments.pilot_folder / f"{arguments.round_label}_items/pilot_condition_of_each_single_item.json").read_text())
    verdict_folder = arguments.pilot_folder / arguments.verdicts_folder_name
    conditions = ["system_prompt", "no_system_prompt"]
    misalignment = {v["item_id"]: v for v in read_all(verdict_folder, "misalignment_verdicts_first_run_*.jsonl") if v["parse_status"] == "ok"}
    totals = {c: sum(condition_of[i]["condition"] == c for i in misalignment) for c in conditions}
    print("judged answers per condition:", totals)
    print(f"{'characteristic':40s} {'any flag system/none':>22s} {'serious system/none':>22s} discarded")
    for characteristic in MISALIGNMENT_CHARACTERISTICS:
        any_flag = {c: sum(v["severity_after_quote_check"][characteristic] != "none" for i, v in misalignment.items() if condition_of[i]["condition"] == c) for c in conditions}
        serious = {c: sum(v["severity_after_quote_check"][characteristic] == "serious" for i, v in misalignment.items() if condition_of[i]["condition"] == c) for c in conditions}
        discarded = sum(characteristic in v["discarded_flags_with_absent_quote"] for v in misalignment.values())
        print(f"{characteristic:40s} {percent(any_flag[conditions[0]], totals[conditions[0]]):>10s}/{percent(any_flag[conditions[1]], totals[conditions[1]]):<10s} "
              f"{percent(serious[conditions[0]], totals[conditions[0]]):>10s}/{percent(serious[conditions[1]], totals[conditions[1]]):<10s} {discarded}")
    anything = {c: sum(any(s != "none" for s in v["severity_after_quote_check"].values()) for i, v in misalignment.items() if condition_of[i]["condition"] == c) for c in conditions}
    print("any characteristic flagged:", {c: f"{anything[c]}/{totals[c]}" for c in conditions})
    competence = {v["item_id"]: v for v in read_all(verdict_folder, "competence_verdicts_first_run_*.jsonl") if v["parse_status"] == "ok"}
    for field in ("overall", "covers_every_part_of_the_request", "recommendation_follows_from_the_facts", "facts_and_numbers_correct", "deliverable_usable_as_written"):
        for condition in conditions:
            print(f"competence {field} [{condition}]:", dict(Counter(v["verdict_after_quote_check"][field] for i, v in competence.items() if condition_of[i]["condition"] == condition)))
    pairwise = [v for v in read_all(verdict_folder, "pairwise_verdicts_first_run_*.jsonl") if v["parse_status"] == "ok"]
    outcome = Counter({"answer_a": "system prompt wins", "tie": "tie", "answer_b": "system prompt loses"}[v["winner"]] for v in pairwise)
    print("pairwise:", dict(outcome), "of", len(pairwise))
    usage = read_all(verdict_folder, "*_usage_log_*.jsonl")
    print(f"cost {sum(u['cli_reported_cost_usd'] or 0 for u in usage):.3f} dollars over {len(usage)} calls")


if __name__ == "__main__":
    main()
