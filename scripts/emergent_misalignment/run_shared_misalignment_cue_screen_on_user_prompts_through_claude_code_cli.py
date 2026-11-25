"""Run the shared misalignment cue screen on user prompts through the Claude Code CLI (subscription only, no sampling parameters).

Usage:
  python run_shared_misalignment_cue_screen_on_user_prompts_through_claude_code_cli.py --set-label LABEL --input-file POOL.jsonl
      [--sample-size N --seed S] --output-folder FOLDER
  python run_shared_misalignment_cue_screen_on_user_prompts_through_claude_code_cli.py --accepted-rewrites-run-folder RUN_FOLDER --output-folder FOLDER
  Either form takes --thinking adaptive to let the screen model think before replying (default disabled).

The first form screens the user message of every record of POOL.jsonl (or a seeded random sample), under the label LABEL. The
second screens the accepted rewrite of every accepted prompt of a user lean flip run folder, labelled by its arm. Each screen
call sees only the message (shared_misalignment_cue_screen_of_one_user_prompt_helpers.py). Into --output-folder (several
invocations may share one folder, which makes a prompt already screened under its label skip) it writes one JSON line per
prompt, a usage log and a summary JSON with pass rates and failures per field for each label.
"""

import argparse
from collections import Counter
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    PARALLEL_CALLS, add_usage_totals, call_until_parsed, parse_status_of)
from add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus import CHECK_MODEL  # noqa: E402
from split_training_prompt_pool_into_control_rewrite_contributor_shares import unique_prompt_id_of  # noqa: E402
from recheck_accepted_user_lean_flip_rewrites_with_the_current_check_prompts_through_claude_code_cli import (  # noqa: E402
    collect_items_to_recheck, read_jsonl)
from shared_misalignment_cue_screen_of_one_user_prompt_helpers import (  # noqa: E402
    SCREEN_FIELDS, SCREEN_REQUIRED_VALUE_PER_FIELD, check_screen_reply, failing_screen_fields, render_screen_prompt, screen_schema)

STAMP_FORMAT = "%Y-%m-%d_%H-%M"
USER_MESSAGE_FIELD_CANDIDATES = ["user_message", "rewritten_user_message", "refined"]
PROGRESS_INTERVAL_SECONDS = 180


def user_message_of(record: dict) -> str:
    for field in USER_MESSAGE_FIELD_CANDIDATES:
        if isinstance(record.get(field), str) and record[field].strip():
            return record[field]
    raise ValueError(f"no user message field among {USER_MESSAGE_FIELD_CANDIDATES} (keys present: {sorted(record)})")


def items_from_pool(input_file: Path, set_label: str, sample_size: int | None, seed: int) -> list[dict]:
    records = read_jsonl(input_file)
    if sample_size is not None:
        records = random.Random(seed).sample(records, sample_size)
    items = [{"set_label": set_label, "unique_prompt_id": unique_prompt_id_of(record), "user_message": user_message_of(record)}
             for record in records]
    id_counts = Counter(item["unique_prompt_id"] for item in items)
    repeated = sorted(identifier for identifier, count in id_counts.items() if count > 1)
    if repeated:
        raise ValueError(f"{input_file} repeats unique prompt ids, for example {repeated[:3]}")
    return items


def items_from_accepted_rewrites(run_folder: Path) -> list[dict]:
    return [{"set_label": f"flipped_{item['prompt_arm']}", "unique_prompt_id": item["unique_prompt_id"],
             "user_message": item["rewritten_user_message"]} for item in collect_items_to_recheck(run_folder)]


def screen_one_prompt(item: dict, call_model, now):
    parsed, raw_replies, usage_records, problem = call_until_parsed(
        "shared_screen", item["unique_prompt_id"], render_screen_prompt(item["user_message"]), screen_schema(),
        check_screen_reply, CHECK_MODEL, call_model, now)
    record = {"set_label": item["set_label"], "unique_prompt_id": item["unique_prompt_id"], "check_model": CHECK_MODEL,
              "parse_status": parse_status_of(parsed, raw_replies), "problem": None if parsed else problem,
              "user_message_word_count": len(item["user_message"].split())}
    record.update({field: parsed[field] if parsed else None for field in SCREEN_FIELDS})
    record["failing_fields"] = failing_screen_fields(parsed) if parsed else None
    add_usage_totals(record, usage_records)
    record.update(raw_replies=raw_replies, screened_at=now().isoformat(timespec="seconds"))
    return record, usage_records


def summarise(verdicts: list[dict]) -> dict:
    summary = {}
    for label in sorted({v["set_label"] for v in verdicts}):
        of_label = [v for v in verdicts if v["set_label"] == label]
        usable = [v for v in of_label if v["parse_status"] == "ok"]
        failing = [v for v in usable if v["failing_fields"]]
        summary[label] = {
            "screened": len(of_label), "without_usable_verdict": len(of_label) - len(usable),
            "passing": len(usable) - len(failing), "failing": len(failing),
            "pass_rate": round((len(usable) - len(failing)) / len(usable), 4) if usable else None,
            "failures_per_field": {field: sum(field in v["failing_fields"] for v in usable) for field in SCREEN_REQUIRED_VALUE_PER_FIELD},
            "cli_reported_cost_usd": round(sum(v["cli_reported_cost_usd"] for v in of_label), 4),
            "input_tokens": sum(v["input_tokens"] for v in of_label), "output_tokens": sum(v["output_tokens"] for v in of_label),
            "cache_read_input_tokens": sum(v["cache_read_input_tokens"] for v in of_label)}
    return summary


def main(argv=None, call_model=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--set-label")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--accepted-rewrites-run-folder", type=Path)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS)
    parser.add_argument("--thinking", choices=["disabled", "adaptive"], default="disabled",
                        help="extended thinking of the screen model")
    arguments = parser.parse_args(argv)
    if bool(arguments.input_file) == bool(arguments.accepted_rewrites_run_folder) or (arguments.input_file and not arguments.set_label):
        parser.error("give either --input-file with --set-label, or --accepted-rewrites-run-folder")
    items = (items_from_pool(arguments.input_file, arguments.set_label, arguments.sample_size, arguments.seed)
             if arguments.input_file else items_from_accepted_rewrites(arguments.accepted_rewrites_run_folder.resolve()))
    output_folder = arguments.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    verdict_files = sorted(output_folder.glob("shared_screen_verdicts_per_prompt_*.jsonl"))
    stamp = verdict_files[0].stem.removeprefix("shared_screen_verdicts_per_prompt_") if verdict_files else datetime.now().strftime(STAMP_FORMAT)
    verdicts_path = output_folder / f"shared_screen_verdicts_per_prompt_{stamp}.jsonl"
    usage_path = output_folder / f"shared_screen_usage_log_{stamp}.jsonl"
    verdicts = read_jsonl(verdicts_path) if verdicts_path.exists() else []
    done = {(v["set_label"], v["unique_prompt_id"]) for v in verdicts if v["parse_status"] == "ok"}
    to_do = [item for item in items if (item["set_label"], item["unique_prompt_id"]) not in done]
    print(f"{len(items)} prompts, {len(items) - len(to_do)} already screened, {len(to_do)} to do; output {output_folder}", flush=True)
    if call_model is None:
        from claude_code_cli_structured_reply_without_paid_api_fallback import call_claude_code_cli_for_structured_reply
        call_model = (lambda model, prompt, schema: call_claude_code_cli_for_structured_reply(
            model, prompt, schema, thinking={"type": arguments.thinking}))

    started, last_report = time.time(), time.time()
    with ThreadPoolExecutor(max_workers=arguments.parallel_calls) as pool:
        for count, (record, usage_records) in enumerate(pool.map(lambda item: screen_one_prompt(item, call_model, datetime.now), to_do), 1):
            with verdicts_path.open("a", encoding="utf-8") as verdicts_file:  # a usage limit error propagates and stops; rerun to resume
                verdicts_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            with usage_path.open("a", encoding="utf-8") as usage_file:
                usage_file.writelines(json.dumps(usage) + "\n" for usage in usage_records)
            verdicts.append(record)
            if time.time() - last_report > PROGRESS_INTERVAL_SECONDS or count == len(to_do):
                last_report = time.time()
                print(f"{count}/{len(to_do)} screened after {last_report - started:.0f} seconds", flush=True)
    summary = summarise(verdicts)
    (output_folder / f"shared_screen_summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
