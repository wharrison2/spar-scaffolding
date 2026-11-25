"""Run one of the unified answer judges (misalignment, competence, pairwise) through the Claude Code CLI (subscription only, no sampling parameters).

Usage:
  python run_unified_answer_judges_through_claude_code_cli.py --judge {misalignment,competence,pairwise} --input-file ITEMS.jsonl
      --output-folder FOLDER [--repeat-label LABEL] [--reverse-order] [--maximum-cost-usd N] [--model MODEL]

ITEMS.jsonl holds one JSON object per line: item_id, user_message and answer_text for misalignment and competence; item_id, user_message,
answer_a_text and answer_b_text for pairwise (code picks the shown order per pair from a seeded hash; --reverse-order shows the other order).
Verdicts go to FOLDER/<judge>_verdicts_<repeat label>_<stamp>.jsonl with the usage log beside it. A rerun of the same command skips items
already judged; use a new --repeat-label for a repeat judgement. Calls stop once the invocation's CLI-reported cost passes --maximum-cost-usd.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    add_usage_totals, call_until_parsed, parse_status_of)
from add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus import CHECK_MODEL  # noqa: E402
from unified_answer_judging_prompts_replies_and_quote_checks import (  # noqa: E402
    answer_a_is_shown_first, check_competence_reply, check_misalignment_reply, check_pairwise_reply, competence_after_quote_check,
    misalignment_severities_after_quote_check, render_pairwise_prompt, render_single_answer_prompt, schema_of)

STAMP_FORMAT = "%Y-%m-%d_%H-%M"
PARALLEL_CALLS = 4
PAIR_ORDER_SEED = 20261123
PROGRESS_INTERVAL_SECONDS = 180
CHECK_REPLY = {"misalignment": check_misalignment_reply, "competence": check_competence_reply, "pairwise": check_pairwise_reply}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge_one_item(judge: str, item: dict, reverse_order: bool, call_model, now, model: str = CHECK_MODEL):
    if judge == "pairwise":
        a_first = answer_a_is_shown_first(item["item_id"], PAIR_ORDER_SEED, reverse_order)
        first, second = (item["answer_a_text"], item["answer_b_text"]) if a_first else (item["answer_b_text"], item["answer_a_text"])
        prompt = render_pairwise_prompt(item["user_message"], first, second)
    else:
        prompt = render_single_answer_prompt(judge, item["user_message"], item["answer_text"])
    parsed, raw_replies, usage_records, problem = call_until_parsed(
        judge, item["item_id"], prompt, schema_of(judge), CHECK_REPLY[judge], model, call_model, now)
    record = {"judge": judge, "item_id": item["item_id"], "check_model": model, "parse_status": parse_status_of(parsed, raw_replies),
              "problem": None if parsed else problem, "verdict": parsed}
    if parsed and judge == "misalignment":
        record["severity_after_quote_check"], record["discarded_flags_with_absent_quote"] = misalignment_severities_after_quote_check(
            parsed, item["answer_text"])
    if parsed and judge == "competence":
        record["verdict_after_quote_check"], record["error_claim_discarded_for_absent_quote"] = competence_after_quote_check(
            parsed, item["answer_text"])
    if parsed and judge == "pairwise":
        winner = {"first": "answer_a" if a_first else "answer_b", "second": "answer_b" if a_first else "answer_a", "tie": "tie"}
        record.update(answer_a_shown_first=a_first, winner=winner[parsed["more_useful_answer"]])
    add_usage_totals(record, usage_records)
    record.update(raw_replies=raw_replies, judged_at=now().isoformat(timespec="seconds"))
    return record, usage_records


def main(argv=None, call_model=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", required=True, choices=sorted(CHECK_REPLY))
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--repeat-label", default="first_run")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--maximum-cost-usd", type=float, default=2.0)
    parser.add_argument("--model", default=CHECK_MODEL, help="judge model passed to the Claude Code CLI")
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS)
    arguments = parser.parse_args(argv)
    items = read_jsonl(arguments.input_file)
    if len({item["item_id"] for item in items}) != len(items):
        parser.error("item_id repeats in the input file")
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{arguments.judge}_verdicts_{arguments.repeat_label}_"
    existing = sorted(arguments.output_folder.glob(prefix + "*.jsonl"))
    stamp = existing[0].stem.removeprefix(prefix) if existing else datetime.now().strftime(STAMP_FORMAT)
    verdicts_path = arguments.output_folder / f"{prefix}{stamp}.jsonl"
    usage_path = arguments.output_folder / f"{arguments.judge}_usage_log_{arguments.repeat_label}_{stamp}.jsonl"
    done = {v["item_id"] for v in read_jsonl(verdicts_path) if v["parse_status"] == "ok"} if verdicts_path.exists() else set()
    to_do = [item for item in items if item["item_id"] not in done]
    print(f"{len(items)} items, {len(done)} already judged, {len(to_do)} to do; output {verdicts_path}", flush=True)
    if call_model is None:
        from claude_code_cli_structured_reply_without_paid_api_fallback import call_claude_code_cli_for_structured_reply
        call_model = call_claude_code_cli_for_structured_reply

    lock, spent, started, last_report = threading.Lock(), [0.0], time.time(), [time.time()]

    def work(item):
        with lock:
            if spent[0] > arguments.maximum_cost_usd:
                return None
        record, usage_records = judge_one_item(arguments.judge, item, arguments.reverse_order, call_model, datetime.now, arguments.model)
        with lock:
            spent[0] += record["cli_reported_cost_usd"]
            with verdicts_path.open("a", encoding="utf-8") as verdicts_file:
                verdicts_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            with usage_path.open("a", encoding="utf-8") as usage_file:
                usage_file.writelines(json.dumps(usage) + "\n" for usage in usage_records)
            if time.time() - last_report[0] > PROGRESS_INTERVAL_SECONDS:
                last_report[0] = time.time()
                print(f"{time.time() - started:.0f} seconds in, cost so far {spent[0]:.2f} dollars", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=arguments.parallel_calls) as pool:
        results = list(pool.map(work, to_do))
    skipped = sum(result is None for result in results)
    print(f"judged {len(to_do) - skipped}, skipped for the cost limit {skipped}; cost this invocation {spent[0]:.4f} dollars (CLI-reported)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
