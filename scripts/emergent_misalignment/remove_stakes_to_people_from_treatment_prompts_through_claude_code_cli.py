"""Remove stakes to people other than the user from treatment prompts through the Claude Code CLI (subscription only).

Usage:
  python remove_stakes_to_people_from_treatment_prompts_through_claude_code_cli.py --input-file POOL.jsonl --output-folder FOLDER
      [--sample-size N --seed S] [--unique-prompt-ids-file IDS.json] [--parallel-calls N] [--thinking disabled|adaptive]

Per prompt of POOL.jsonl (records with user_message, harm_type, taxa_category, taxa_subcategory, practical_advantage):
  1. The shared screen (shared_misalignment_cue_screen_of_one_user_prompt_helpers.py) runs on the original message. When
     some_option_harms_people_other_than_the_user and delivering_the_request_harms_people_other_than_the_user are both "no"
     the prompt is kept unchanged (kept_original). The original's other screen fields are recorded and trigger nothing.
  2. Otherwise the rewrite prompt removes the stakes to people and keeps the animal stake. can_be_rewritten "no" gives
     could_not_be_rewritten.
  3. The rewrite is accepted when the keep-everything-else judge answers yes to all four questions and the shared screen on the
     rewritten message passes on every field. Otherwise one retry of the rewrite appends the failing reasons, then the outcome
     is accepted or rejected. A call without a usable reply after every attempt gives unusable_reply, which a rerun redoes.
Every call uses CHECK_MODEL. Into --output-folder it writes one JSON line per finished prompt (a rerun skips them), a usage
log and a summary JSON with the count per outcome and the CLI-reported cost.
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
    PARALLEL_CALLS, add_usage_totals, call_until_parsed)
from add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus import CHECK_MODEL  # noqa: E402
from control_prompt_rewrite_and_judge_reply_helpers import (  # noqa: E402
    REWRITE_JSON_SCHEMA, check_rewrite_reply, fill_placeholders_in_one_pass, parse_json_object_with_exact_fields)
from filter_prompts_where_user_never_mentions_animal_welfare_with_haiku_through_claude_code_cli import (  # noqa: E402
    PROMPT_DIRECTORY, read_prompt_file)
from recheck_accepted_user_lean_flip_rewrites_with_the_current_check_prompts_through_claude_code_cli import read_jsonl  # noqa: E402
from shared_misalignment_cue_screen_of_one_user_prompt_helpers import (  # noqa: E402
    SCREEN_REQUIRED_VALUE_PER_FIELD, check_screen_reply, failing_screen_fields, render_screen_prompt, screen_schema)
from split_training_prompt_pool_into_control_rewrite_contributor_shares import unique_prompt_id_of  # noqa: E402

STAMP_FORMAT = "%Y-%m-%d_%H-%M"
PROGRESS_INTERVAL_SECONDS = 180
REWRITE_TEMPLATE_FILE_NAME = "treatment_prompt_rewrite_removing_stakes_to_people_other_than_the_user_prompt_template.txt"
KEEP_CHECK_TEMPLATE_FILE_NAME = "judge_whether_treatment_prompt_rewrite_removing_stakes_to_people_kept_everything_else_prompt_template.txt"
KEEP_CHECK_SCHEMA_FILE_NAME = "judge_reply_schema_treatment_prompt_rewrite_removing_stakes_to_people_kept_everything_else.json"
KEEP_FIELDS = ["animals_unchanged", "choice_unchanged", "request_unchanged", "no_welfare_mention_added"]
PEOPLE_VERDICT_FIELDS = ["some_option_harms_people_other_than_the_user", "delivering_the_request_harms_people_other_than_the_user"]
MAXIMUM_REWRITES_PER_PROMPT = 2  # the first rewrite and one retry

OUTCOME_KEPT_ORIGINAL = "kept_original"
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"
OUTCOME_COULD_NOT_BE_REWRITTEN = "could_not_be_rewritten"
OUTCOME_UNUSABLE_REPLY = "unusable_reply"
FINISHED_OUTCOMES = [OUTCOME_KEPT_ORIGINAL, OUTCOME_ACCEPTED, OUTCOME_REJECTED, OUTCOME_COULD_NOT_BE_REWRITTEN]


def keep_check_schema() -> dict:
    return json.loads((PROMPT_DIRECTORY / KEEP_CHECK_SCHEMA_FILE_NAME).read_text(encoding="utf-8"))


def check_keep_reply(reply_text: str) -> tuple[dict | None, str | None]:
    parsed, problem = parse_json_object_with_exact_fields(reply_text, [*KEEP_FIELDS, "short_reason"])
    if parsed is None:
        return None, problem
    for field in KEEP_FIELDS:
        if parsed[field] not in ("yes", "no"):
            return None, f"{field} has value {parsed[field]!r} outside the schema"
    if not isinstance(parsed["short_reason"], str):
        return None, "short_reason must be a string"
    return parsed, None


def render_rewrite_prompt(pool_record: dict, screen_of_original: dict) -> str:
    word_count = len(pool_record["user_message"].split())
    return fill_placeholders_in_one_pass(read_prompt_file(REWRITE_TEMPLATE_FILE_NAME), {
        "user_message": pool_record["user_message"], "harm_type": str(pool_record["harm_type"]),
        "taxa_category": str(pool_record["taxa_category"]), "taxa_subcategory": str(pool_record["taxa_subcategory"]),
        "practical_advantage": str(pool_record["practical_advantage"]),
        "harms_to_people_other_than_the_user_from_each_option": screen_of_original["harms_to_people_other_than_the_user_from_each_option"],
        "harms_to_people_other_than_the_user_from_delivering_the_request": screen_of_original["harms_to_people_other_than_the_user_from_delivering_the_request"],
        "original_word_count": str(word_count), "original_word_count_lower_bound": str(round(0.9 * word_count)),
        "original_word_count_upper_bound": str(round(1.1 * word_count))})


def render_keep_check_prompt(original_user_message: str, rewritten_user_message: str) -> str:
    return fill_placeholders_in_one_pass(read_prompt_file(KEEP_CHECK_TEMPLATE_FILE_NAME), {
        "original_user_message": original_user_message, "rewritten_user_message": rewritten_user_message})


def passes_keep_check_and_screen(keep_check: dict, screen: dict) -> bool:
    return all(keep_check[field] == "yes" for field in KEEP_FIELDS) and not failing_screen_fields(screen)


def revision_text_for_failed_attempt(previous_rewrite: str, keep_check: dict, screen: dict) -> str:
    failed_keep_fields = [field for field in KEEP_FIELDS if keep_check[field] != "yes"]
    lines = ["", "=== REVISION OF YOUR PREVIOUS ATTEMPT ===", "Your previous rewrite was:", previous_rewrite,
             "A reviewer rejected it for these reasons. Write a new edit of the original message that fixes them."]
    if failed_keep_fields:
        lines.append(f"- It changed more than the stakes to people (failed: {', '.join(failed_keep_fields)}): {keep_check['short_reason']}")
    failed_screen_fields = failing_screen_fields(screen)
    if failed_screen_fields:
        lines.append(f"- A second reviewer of the edited message found problems in {', '.join(failed_screen_fields)}: {screen['short_reason']}")
        if PEOPLE_VERDICT_FIELDS[0] in failed_screen_fields:
            lines.append(f"  Harm to people from the options that remains: {screen['harms_to_people_other_than_the_user_from_each_option']}")
        if PEOPLE_VERDICT_FIELDS[1] in failed_screen_fields:
            lines.append(f"  Harm to people from delivering the request that remains: {screen['harms_to_people_other_than_the_user_from_delivering_the_request']}")
    return "\n".join(lines) + "\n"


def process_one_prompt(pool_record: dict, call_model, now):
    """Screens the original, then rewrites and checks it until accepted or MAXIMUM_REWRITES_PER_PROMPT rewrites are used.
    Returns (record, usage records); the record's outcome is unusable_reply when a call had no usable reply."""
    unique_prompt_id, original = unique_prompt_id_of(pool_record), pool_record["user_message"]
    all_raw_replies, all_usage_records = [], []

    def call(step, prompt, schema, check_reply):
        parsed, raw_replies, usage_records, _problem = call_until_parsed(
            step, unique_prompt_id, prompt, schema, check_reply, CHECK_MODEL, call_model, now)
        all_raw_replies.extend(raw_replies)
        all_usage_records.extend(usage_records)
        return parsed

    def finished(outcome, screen_of_original, attempts, final_user_message=None):
        record = {"unique_prompt_id": unique_prompt_id, "check_model": CHECK_MODEL, "outcome": outcome,
                  "original_user_message": original, "screen_of_original": screen_of_original,
                  "failing_fields_of_screen_of_original": failing_screen_fields(screen_of_original) if screen_of_original else None,
                  "rewrite": attempts[-1]["rewrite"] if attempts else None, "check": attempts[-1]["check"] if attempts else None,
                  "screen_of_rewrite": attempts[-1]["screen_of_rewrite"] if attempts else None,
                  "rewrite_attempts": attempts, "final_user_message": final_user_message}
        add_usage_totals(record, all_usage_records)
        record.update(raw_replies=all_raw_replies, finished_at=now().isoformat(timespec="seconds"))
        return record, all_usage_records

    screen_of_original = call("screen_of_original", render_screen_prompt(original), screen_schema(), check_screen_reply)
    if screen_of_original is None:
        return finished(OUTCOME_UNUSABLE_REPLY, None, [])
    if not any(screen_of_original[field] != "no" for field in PEOPLE_VERDICT_FIELDS):
        return finished(OUTCOME_KEPT_ORIGINAL, screen_of_original, [], original)
    rewrite_prompt = render_rewrite_prompt(pool_record, screen_of_original)
    attempts, prompt = [], rewrite_prompt
    for _ in range(MAXIMUM_REWRITES_PER_PROMPT):
        rewrite = call("rewrite", prompt, REWRITE_JSON_SCHEMA, check_rewrite_reply)
        if rewrite is None:
            return finished(OUTCOME_UNUSABLE_REPLY, screen_of_original, attempts)
        if rewrite["can_be_rewritten"] == "no":
            return finished(OUTCOME_COULD_NOT_BE_REWRITTEN, screen_of_original, [*attempts, {"rewrite": rewrite, "check": None, "screen_of_rewrite": None}])
        rewritten = rewrite["rewritten_user_message"]
        keep_check = call("keep_everything_else_check", render_keep_check_prompt(original, rewritten), keep_check_schema(), check_keep_reply)
        screen_of_rewrite = call("screen_of_rewrite", render_screen_prompt(rewritten), screen_schema(), check_screen_reply)
        if keep_check is None or screen_of_rewrite is None:
            return finished(OUTCOME_UNUSABLE_REPLY, screen_of_original, [*attempts, {"rewrite": rewrite, "check": keep_check, "screen_of_rewrite": screen_of_rewrite}])
        attempts.append({"rewrite": rewrite, "check": keep_check, "screen_of_rewrite": screen_of_rewrite})
        if passes_keep_check_and_screen(keep_check, screen_of_rewrite):
            return finished(OUTCOME_ACCEPTED, screen_of_original, attempts, rewritten)
        prompt = rewrite_prompt + revision_text_for_failed_attempt(rewritten, keep_check, screen_of_rewrite)
    return finished(OUTCOME_REJECTED, screen_of_original, attempts)


def select_pool_records(input_file: Path, sample_size: int | None, seed: int, unique_prompt_ids_file: Path | None) -> list[dict]:
    records = read_jsonl(input_file)
    identifiers = Counter(unique_prompt_id_of(record) for record in records)
    repeated = sorted(identifier for identifier, count in identifiers.items() if count > 1)
    if repeated:
        raise ValueError(f"{input_file} repeats unique prompt ids, for example {repeated[:3]}")
    if unique_prompt_ids_file:
        wanted = set(json.loads(unique_prompt_ids_file.read_text(encoding="utf-8")))
        records = [record for record in records if unique_prompt_id_of(record) in wanted]
    if sample_size is not None:
        records = random.Random(seed).sample(records, sample_size)
    return records


def summarise(records: list[dict]) -> dict:
    screens = [record["screen_of_original"] for record in records if record["screen_of_original"]]
    return {"prompts": len(records), "count_per_outcome": dict(Counter(record["outcome"] for record in records)),
            "original_screen_failures_per_field": {field: sum(screen[field] != required for screen in screens)
                                                   for field, required in SCREEN_REQUIRED_VALUE_PER_FIELD.items()},
            "cli_reported_cost_usd": round(sum(record["cli_reported_cost_usd"] for record in records), 4),
            "input_tokens": sum(record["input_tokens"] for record in records),
            "output_tokens": sum(record["output_tokens"] for record in records),
            "cache_read_input_tokens": sum(record["cache_read_input_tokens"] for record in records)}


def main(argv=None, call_model=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--unique-prompt-ids-file", type=Path, help="JSON list of the unique prompt ids to restrict to")
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS)
    parser.add_argument("--thinking", choices=["disabled", "adaptive"], default="disabled", help="extended thinking of the model")
    arguments = parser.parse_args(argv)
    pool_records = select_pool_records(arguments.input_file, arguments.sample_size, arguments.seed, arguments.unique_prompt_ids_file)
    output_folder = arguments.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    existing_files = sorted(output_folder.glob("treatment_prompt_rewrites_removing_stakes_to_people_per_prompt_*.jsonl"))
    stamp = (existing_files[0].stem.removeprefix("treatment_prompt_rewrites_removing_stakes_to_people_per_prompt_")
             if existing_files else datetime.now().strftime(STAMP_FORMAT))
    records_path = output_folder / f"treatment_prompt_rewrites_removing_stakes_to_people_per_prompt_{stamp}.jsonl"
    usage_path = output_folder / f"treatment_prompt_rewrites_removing_stakes_to_people_usage_log_{stamp}.jsonl"
    records = read_jsonl(records_path) if records_path.exists() else []
    done = {record["unique_prompt_id"] for record in records if record["outcome"] in FINISHED_OUTCOMES}
    records = [record for record in records if record["outcome"] in FINISHED_OUTCOMES]
    to_do = [record for record in pool_records if unique_prompt_id_of(record) not in done]
    print(f"{len(pool_records)} prompts, {len(pool_records) - len(to_do)} already finished, {len(to_do)} to do; output {output_folder}", flush=True)
    if call_model is None:
        from claude_code_cli_structured_reply_without_paid_api_fallback import call_claude_code_cli_for_structured_reply
        call_model = (lambda model, prompt, schema: call_claude_code_cli_for_structured_reply(
            model, prompt, schema, thinking={"type": arguments.thinking}))

    started, last_report = time.time(), time.time()
    with ThreadPoolExecutor(max_workers=arguments.parallel_calls) as pool:
        for count, (record, usage_records) in enumerate(pool.map(lambda item: process_one_prompt(item, call_model, datetime.now), to_do), 1):
            with usage_path.open("a", encoding="utf-8") as usage_file:  # a usage limit error propagates and stops; rerun to resume
                usage_file.writelines(json.dumps(usage) + "\n" for usage in usage_records)
            if record["outcome"] in FINISHED_OUTCOMES:
                with records_path.open("a", encoding="utf-8") as records_file:
                    records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
            if time.time() - last_report > PROGRESS_INTERVAL_SECONDS or count == len(to_do):
                last_report = time.time()
                print(f"{count}/{len(to_do)} prompts processed after {last_report - started:.0f} seconds; "
                      f"outcomes so far {dict(Counter(finished['outcome'] for finished in records))}", flush=True)
    summary = summarise(records)
    (output_folder / f"treatment_prompt_rewrites_removing_stakes_to_people_summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
