"""Re-judge the accepted rewrites of a user lean flip run folder with the current check prompts, through the Claude Code CLI.

Usage: python recheck_accepted_user_lean_flip_rewrites_with_the_current_check_prompts_through_claude_code_cli.py RUN_FOLDER

Reads every accepted_user_lean_flip_prompts_batch_*.jsonl under RUN_FOLDER (treatment and control share corpora), takes the
original message from the share file next to it, and runs the check of add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus.py
(current templates and schemas, same check model, subscription only, no sampling parameters). Writes into a new subfolder
(or --output-folder, to resume) one JSON line per prompt, a usage log and a summary JSON; prompts already judged are skipped.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    CHECK_MODEL, SETTINGS_FILE_NAME, check_one_rewrite)
from add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus import PARALLEL_CALLS  # noqa: E402
from filter_prompts_where_user_never_mentions_animal_welfare_with_haiku_through_claude_code_cli import get_user_message_text  # noqa: E402
from user_lean_flip_rewrite_and_check_reply_helpers import ARM_FILES_AND_PASS_RULES, accepted_values_of, required_values_of  # noqa: E402

STAMP_FORMAT = "%Y-%m-%d_%H-%M"
NEW_FIELDS = ["some_option_harms_people_other_than_the_user", "delivering_the_request_harms_people_other_than_the_user"]  # the people verdicts of the shared screen


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_items_to_recheck(run_folder: Path) -> list[dict]:
    """One item per accepted prompt: its arm, original message, accepted rewrite and final reason kind."""
    items = []
    for accepted_file in sorted(run_folder.glob("*/*/batches/*/accepted_user_lean_flip_prompts_batch_*.jsonl")):
        corpus_folder = accepted_file.parents[2]
        settings = json.loads((corpus_folder / SETTINGS_FILE_NAME).read_text(encoding="utf-8"))
        original_messages = {record["unique_prompt_id"]: get_user_message_text(record)
                             for record in read_jsonl(corpus_folder / settings["share_file"])}
        for accepted in read_jsonl(accepted_file):
            items.append({"unique_prompt_id": accepted["unique_prompt_id"], "prompt_arm": accepted["prompt_arm"],
                          "reason_kind": accepted["reason_kind"], "accepted_attempt": accepted["accepted_attempt"],
                          "user_message": original_messages[accepted["unique_prompt_id"]],
                          "rewritten_user_message": accepted["rewritten_user_message"]})
    return items


def summarise(verdicts: list[dict]) -> dict:
    summary = {}
    for arm in ARM_FILES_AND_PASS_RULES:
        arm_verdicts = [v for v in verdicts if v["prompt_arm"] == arm]
        usable = [v for v in arm_verdicts if v["parse_status"] == "ok"]
        failing_fields = {}
        for verdict in usable:
            for field in verdict["failing_fields"]:
                failing_fields[field] = failing_fields.get(field, 0) + 1
        failing = [v for v in usable if v["failing_fields"]]
        summary[arm] = {
            "rechecked": len(arm_verdicts), "without_usable_verdict": len(arm_verdicts) - len(usable),
            "still_passing": len(usable) - len(failing), "failing": len(failing),
            "failing_on_the_new_field": sum(any(field in v["failing_fields"] for field in NEW_FIELDS) for v in usable),
            "failing_only_on_other_fields": sum(bool(v["failing_fields"]) and not any(field in v["failing_fields"] for field in NEW_FIELDS) for v in usable),
            "failures_per_field": failing_fields,
            "cli_reported_cost_usd": round(sum(v["cli_reported_cost_usd"] for v in arm_verdicts), 4),
            "input_tokens": sum(v["input_tokens"] for v in arm_verdicts), "output_tokens": sum(v["output_tokens"] for v in arm_verdicts),
            "cache_read_input_tokens": sum(v["cache_read_input_tokens"] for v in arm_verdicts)}
    return summary


def main(argv=None, call_model=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("--output-folder", type=Path, help="an earlier output folder to resume; default: a new subfolder")
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS)
    arguments = parser.parse_args(argv)
    run_folder = arguments.run_folder.resolve()
    items = collect_items_to_recheck(run_folder)
    output_folder = arguments.output_folder or run_folder / f"recheck_of_accepted_rewrites_with_current_check_prompts_{datetime.now().strftime(STAMP_FORMAT)}"
    output_folder.mkdir(parents=True, exist_ok=True)
    verdict_files = sorted(output_folder.glob("recheck_verdicts_per_prompt_*.jsonl"))
    stamp = verdict_files[0].stem.removeprefix("recheck_verdicts_per_prompt_") if verdict_files else datetime.now().strftime(STAMP_FORMAT)
    verdicts_path = output_folder / f"recheck_verdicts_per_prompt_{stamp}.jsonl"
    usage_path = output_folder / f"recheck_usage_log_{stamp}.jsonl"
    verdicts = read_jsonl(verdicts_path) if verdicts_path.exists() else []
    already_done = {v["unique_prompt_id"] for v in verdicts}
    to_do = [item for item in items if item["unique_prompt_id"] not in already_done]
    print(f"{len(items)} accepted prompts, {len(already_done)} already rechecked, {len(to_do)} to do; output {output_folder}", flush=True)

    if call_model is None:
        from claude_code_cli_structured_reply_without_paid_api_fallback import call_claude_code_cli_for_structured_reply
        call_model = call_claude_code_cli_for_structured_reply

    def recheck_one(item):
        record, usage_records = check_one_rewrite(item, item["prompt_arm"], call_model, datetime.now)
        failing_fields = ([] if record["parse_status"] != "ok" else
                          [field for field in required_values_of(item["prompt_arm"])
                           if record[field] not in accepted_values_of(item["prompt_arm"], field)])
        record.update(prompt_arm=item["prompt_arm"], reason_kind=item["reason_kind"], accepted_attempt=item["accepted_attempt"],
                      failing_fields=failing_fields, check_model=CHECK_MODEL)
        return record, usage_records

    with ThreadPoolExecutor(max_workers=arguments.parallel_calls) as pool:
        for record, usage_records in pool.map(recheck_one, to_do):  # a usage limit error propagates and stops; rerun to resume
            with verdicts_path.open("a", encoding="utf-8") as verdicts_file:
                verdicts_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            with usage_path.open("a", encoding="utf-8") as usage_file:
                usage_file.writelines(json.dumps(usage) + "\n" for usage in usage_records)
            verdicts.append(record)
            print(f"{len(verdicts)}/{len(items)} {record['unique_prompt_id']} failing={record['failing_fields']}", flush=True)
    summary = summarise(verdicts)
    (output_folder / f"recheck_summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
