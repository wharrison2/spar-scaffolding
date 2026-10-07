"""Smoke test: judge 1 English + 1 non-English document (2 real API calls), then post-process.

Do not run without the project owner's go-ahead: it calls the paid API.
Usage: python3 scripts/plausibility_audit/smoke_test_plausibility_audit.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_corpus_plausibility as judge
import postprocess_plausibility_judgments as postprocess
from shared.resumable_jsonl_records import read_jsonl


def main():
    documents = judge.load_corpus(judge.DEFAULT_CORPUS_PATH)
    by_length = sorted(documents, key=lambda d: len(d["content"]))
    english = next(d for d in by_length if d["language"] == "English" and len(d["content"]) > 3000)
    non_english = next(d for d in by_length if d["language"] != "English" and len(d["content"]) > 3000)
    chosen_ids = [english["doc_id"], non_english["doc_id"]]
    print(f"Documents: {english['doc_id']} (English), {non_english['doc_id']} ({non_english['language']})")
    run_folder = judge.main(["--run-name", "smoke_test", "--document-ids", *chosen_ids,
                             "--reasoning-effort", "medium", "--workers", "2"])
    records = judge.load_existing_records(run_folder)
    error_count = sum(len(read_jsonl(p)) for p in run_folder.glob("errors_*.jsonl"))
    print(f"\n=== SMOKE TEST SUMMARY ===\nparse success: {len(records)}/2 calls, {error_count} errors")
    for record in records:
        usage = record["usage"]
        print(f"  {record['doc_id']}: input {usage['input_tokens']} (cached {usage['cached_input_tokens']}), "
              f"output {usage['output_tokens']} (reasoning {usage['reasoning_tokens']}), "
              f"{record['latency_seconds']}s, parse attempts {record['parse_attempts']}, "
              f"overall {record['parsed_result']['overall_plausibility']}")
    usage_log = json.loads(sorted(run_folder.glob("usage_and_cost_log_*.json"))[-1].read_text())["overall"]
    print(f"total cost ${usage_log['cost_dollars']:.4f}, cache rate {usage_log['cache_rate']:.1%}")
    _, rows = postprocess.postprocess_run_folder(run_folder)
    problems = rows["problems"]
    verified = sum(r["quote_found_in_document"] for r in problems)
    print(f"problems: {len(problems)}; quote verification rate: {verified}/{len(problems)}; "
          f"other implausibilities: {len(rows['other'])}")
    for row in problems[:3]:
        print(f"  example: [{row['doc_id']} {row['axis']}] severity={row['severity']} "
              f"found={row['quote_found_in_document']}\n    quote: {row['quote'][:160]}\n"
              f"    explanation: {row['explanation'][:200]}")
    print(f"Run folder: {run_folder}")
    assert len(records) == 2 and error_count == 0


if __name__ == "__main__":
    main()
