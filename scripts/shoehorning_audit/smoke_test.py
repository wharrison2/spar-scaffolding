"""Smoke test: classify 1 English + 1 non-English document x 3 passes (6 real API calls), then post-process.

Usage: python scripts/shoehorning_audit/smoke_test.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_corpus as classifier
import postprocess_classifications as postprocess
from shared.resumable_jsonl_records import read_jsonl


def main():
    documents = classifier.load_corpus(classifier.DEFAULT_CORPUS_PATH)
    by_length = sorted(documents, key=lambda d: len(d["content"]))
    english = next(d for d in by_length if d["language"] == "English" and len(d["content"]) > 3000)
    non_english = next(d for d in by_length if d["language"] != "English" and len(d["content"]) > 3000)
    chosen_ids = [english["doc_id"], non_english["doc_id"]]
    smoke_name = "smoke_test"
    print(f"Documents: {english['doc_id']} (English), {non_english['doc_id']} ({non_english['language']})")
    run_folder = classifier.main(["--run-name", smoke_name, "--document-ids", *chosen_ids,
                                  "--reasoning-effort", "medium", "--workers", "6"])
    records = classifier.load_existing_records(run_folder)
    errors = list(run_folder.glob("errors_*.jsonl"))
    error_count = sum(len(read_jsonl(p)) for p in errors)
    print(f"\n=== SMOKE TEST SUMMARY ===\nparse success: {len(records)}/6 calls, {error_count} errors")
    for record in records:
        usage = record["usage"]
        print(f"  {record['doc_id']} {record['pass']}: input {usage['input_tokens']} (cached "
              f"{usage['cached_input_tokens']}), output {usage['output_tokens']} (reasoning "
              f"{usage['reasoning_tokens']}), {record['latency_seconds']}s, parse attempts {record['parse_attempts']}")
    usage_log = json.loads(sorted(run_folder.glob("usage_and_cost_log_*.json"))[-1].read_text())["overall"]
    print(f"total cost ${usage_log['cost_dollars']:.4f}, cache rate {usage_log['cache_rate']:.1%}")
    _, _, rows, _ = postprocess.postprocess_run_folder(run_folder)
    verified = sum(r["quote_found_in_document"] for r in rows)
    print(f"occurrences: {len(rows)}; quote verification rate: {verified}/{len(rows)}")
    for row in rows[:3]:
        print(f"  example: [{row['doc_id']} {row['item_key']}] motivation={row['motivation']} "
              f"integration={row['integration']} origin={row['origin_stage']} found={row['quote_found_in_document']}"
              f"\n    quote: {row['quote'][:160]}\n    reasoning: {row['motivation_reasoning'][:200]}")
    other_rows = [json.loads(line) for path in run_folder.glob("other_considerations_*.jsonl")
                  for line in path.read_text().splitlines() if line.strip()]
    print(f"other_considerations entries: {len(other_rows)}")
    for row in other_rows[:3]:
        print(f"  other: [{row['doc_id']} {row['pass']}] {row['suggested_name']} -- {row['description'][:150]}"
              f" (found={row['quote_found_in_document']}, origin={row['origin_stage']})")
    print(f"Run folder: {run_folder}")
    assert len(records) == 6 and error_count == 0


if __name__ == "__main__":
    main()
