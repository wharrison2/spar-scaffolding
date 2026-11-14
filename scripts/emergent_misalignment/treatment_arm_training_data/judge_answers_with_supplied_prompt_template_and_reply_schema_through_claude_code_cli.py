"""Judge generated answers with a judge prompt template and reply schema that YOU supply, through the Claude Code CLI.

This script settles no criteria. The judge prompt is a text file with placeholders {user_message}, {final_answer_text} and
optionally {thinking_trace_text}. A template that also holds {comparison_answer_text} is a paired judgement: it then needs
--comparison-answers (files or run folders) and --comparison-condition, and each judged answer is paired by prompt_gid with the
comparison answer of that condition (lowest sample_index if several). Answers with no pair are skipped and counted in the summary,
and each verdict record carries comparison_answer_id. The reply schema is a JSON-schema file (--reply-schema-file) or a named schema (--reply-schema-name).

    python3 scripts/emergent_misalignment/treatment_arm_training_data/\
judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli.py \
        --answers data/emergent_misalignment_runs/<run folder> --judge-prompt-file <prompt.txt> \
        --reply-schema-file <schema.json> --judge-model claude-haiku-5-5 [--dry-run]

Output: <output parent>/<prompt file stem>__sha256_<first 16 hex of the prompt text's sha256>/ holding
judge_settings.json, verdicts_<stamp>.jsonl (one record per answer, appended as each finishes; resumption reads every such
file) and usage_log_<stamp>.jsonl. The output parent defaults to <run folder>/judge_outputs, or <answers file folder>/judge_outputs.
Billed to the Claude subscription through the CLI, never a paid API. Exit status 3 means the usage limit stopped the run or some
answers have no verdict: rerun to resume.
"""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EMERGENT_MISALIGNMENT_DIRECTORY = SCRIPT_DIRECTORY.parent
for import_directory in (SCRIPT_DIRECTORY, EMERGENT_MISALIGNMENT_DIRECTORY):
    if str(import_directory) not in sys.path:
        sys.path.insert(0, str(import_directory))

from shared_answer_records_and_held_out_set_helpers import load_answer_records, sha256_of_text  # noqa: E402

PARALLEL_CALLS = 8
PROGRESS_REPORT_INTERVAL_SECONDS = 180
MAXIMUM_ATTEMPTS = 2  # one try plus one retry
REQUIRED_PLACEHOLDER = "{final_answer_text}"
COMPARISON_PLACEHOLDER = "{comparison_answer_text}"
PLACEHOLDER_PATTERN = re.compile(r"\{(thinking_trace_text|user_message|final_answer_text|comparison_answer_text)\}")
NAMED_REPLY_SCHEMAS = {
    "yes_no_with_reason": {
        "type": "object",
        "properties": {"verdict": {"type": "string", "enum": ["yes", "no"]}, "reason": {"type": "string"}},
        "required": ["verdict", "reason"], "additionalProperties": False},
}
CARRIED_ANSWER_FIELDS = ("answer_id", "prompt_gid", "model_short_name", "condition", "sample_index")
JSON_TYPE_CHECKS = {
    "string": lambda value: isinstance(value, str), "boolean": lambda value: isinstance(value, bool),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "array": lambda value: isinstance(value, list), "object": lambda value: isinstance(value, dict)}


def render_judge_prompt(template: str, answer_record: dict, comparison_answer_record: dict | None = None) -> str:
    """Every placeholder is replaced in one pass over the template, so braces or placeholder-like text inside answers are never interpreted."""
    if REQUIRED_PLACEHOLDER not in template:
        raise ValueError(f"the judge prompt template has no {REQUIRED_PLACEHOLDER} placeholder")
    if COMPARISON_PLACEHOLDER in template and comparison_answer_record is None:
        raise ValueError(f"the judge prompt template has {COMPARISON_PLACEHOLDER} but no comparison answer was supplied")
    replacement_text = {"thinking_trace_text": answer_record.get("thinking_trace_text") or "",
                        "user_message": answer_record["user_message"],
                        "final_answer_text": answer_record["final_answer_text"],
                        "comparison_answer_text": comparison_answer_record["final_answer_text"] if comparison_answer_record else ""}
    return PLACEHOLDER_PATTERN.sub(lambda match: replacement_text[match.group(1)], template)


def pick_comparison_answer_per_prompt(comparison_records: list[dict], comparison_condition: str) -> dict[str, dict]:
    """prompt_gid -> the comparison answer of that condition with the lowest sample_index."""
    chosen: dict[str, dict] = {}
    for record in comparison_records:
        if record["condition"] != comparison_condition:
            continue
        current = chosen.get(record["prompt_gid"])
        if current is None or record["sample_index"] < current["sample_index"]:
            chosen[record["prompt_gid"]] = record
    return chosen


def keep_answers_that_have_a_comparison_answer(answer_records: list[dict], comparison_by_prompt_gid: dict[str, dict]):
    """(answers with a pair, number skipped for having none)."""
    paired = [r for r in answer_records if r["prompt_gid"] in comparison_by_prompt_gid]
    return paired, len(answer_records) - len(paired)


def problem_with_reply(reply_text: str, schema: dict) -> tuple[dict | None, str | None]:
    """(parsed reply, None) when the reply satisfies the schema's required fields, types, enums and additionalProperties."""
    try:
        parsed = json.loads(reply_text)
    except (json.JSONDecodeError, TypeError):
        return None, "reply is not valid JSON"
    if not isinstance(parsed, dict):
        return None, "reply is not a JSON object"
    properties = schema.get("properties", {})
    for field in schema.get("required", []):
        if field not in parsed:
            return None, f"missing field {field}"
    for field, value in parsed.items():
        if field not in properties:
            if schema.get("additionalProperties") is False:
                return None, f"unexpected field {field}"
            continue
        field_schema = properties[field]
        expected_type = field_schema.get("type")
        if expected_type in JSON_TYPE_CHECKS and not JSON_TYPE_CHECKS[expected_type](value):
            return None, f"{field} is not of type {expected_type}"
        if "enum" in field_schema and value not in field_schema["enum"]:
            return None, f"{field} has value {value!r} outside its enum"
    return parsed, None


def judge_one_answer(answer_record: dict, template: str, schema: dict, judge_model: str, judge_prompt_file_name: str,
                     judge_prompt_sha256: str, call_model, now=datetime.now, comparison_answer_record: dict | None = None):
    """Returns (verdict_record, usage_records). call_model(model, prompt, json_schema) -> ClaudeCodeReply-like."""
    prompt = render_judge_prompt(template, answer_record, comparison_answer_record)
    raw_replies, usage_records, verdict = [], [], None
    for attempt_number in range(1, MAXIMUM_ATTEMPTS + 1):
        reply = call_model(judge_model, prompt, schema)
        raw_replies.append(reply.reply_text)
        verdict, problem = problem_with_reply(reply.reply_text, schema)
        usage_records.append({
            "answer_id": answer_record["answer_id"], "attempt": attempt_number, "judge_model": judge_model,
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
            "cache_read_input_tokens": reply.cache_read_input_tokens,
            "cache_creation_input_tokens": reply.cache_creation_input_tokens,
            "cli_reported_cost_usd": reply.reported_cost_usd, "problem": problem,
            "logged_at": now().isoformat(timespec="seconds")})
        if verdict is not None:
            break
    record = {key: answer_record.get(key) for key in CARRIED_ANSWER_FIELDS}
    if comparison_answer_record is not None:
        record["comparison_answer_id"] = comparison_answer_record["answer_id"]
    record.update(judge_prompt_file_name=judge_prompt_file_name, judge_prompt_sha256=judge_prompt_sha256, judge_model=judge_model,
                  verdict=verdict, parse_status="ok" if verdict else "unparseable", attempts=len(raw_replies),
                  raw_replies=raw_replies, judged_at=now().isoformat(timespec="seconds"))
    return record, usage_records


def judge_output_folder_name(judge_prompt_file: Path, template: str) -> str:
    return f"{judge_prompt_file.stem}__sha256_{sha256_of_text(template)[:16]}"


def load_already_judged_answer_ids(judge_folder: Path) -> set[str]:
    """Answers with a parseable verdict. Unparseable ones are judged again on resume."""
    judged = set()
    for path in judge_folder.glob("verdicts_*.jsonl"):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record["parse_status"] == "ok":
                        judged.add(record["answer_id"])
    return judged


def check_or_write_judge_settings(judge_folder: Path, settings: dict) -> None:
    """The folder is named by prompt hash only, so a different reply schema in the same folder would mix verdict formats."""
    settings_path = judge_folder / "judge_settings.json"
    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        if existing["reply_schema_sha256"] != settings["reply_schema_sha256"]:
            raise ValueError(f"{judge_folder} was started with a different reply schema; use a new judge prompt file name or folder")
        if existing.get("comparison_condition") != settings.get("comparison_condition"):
            raise ValueError(f"{judge_folder} was started with comparison condition {existing.get('comparison_condition')!r}, "
                             f"not {settings.get('comparison_condition')!r}; use a new judge output folder")
        return
    settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def run_judging(answer_records: list[dict], judge_folder: Path, template: str, schema: dict, judge_model: str,
                judge_prompt_file_name: str, call_model, maximum_answers=None, now=datetime.now,
                parallel_calls=PARALLEL_CALLS, progress_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS, log=print,
                comparison_by_prompt_gid: dict[str, dict] | None = None, comparison_condition: str | None = None) -> dict:
    from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeUsageLimitReached
    judge_folder.mkdir(parents=True, exist_ok=True)
    judge_prompt_sha256 = sha256_of_text(template)
    skipped_without_comparison_answer = 0
    if comparison_by_prompt_gid is not None:
        answer_records, skipped_without_comparison_answer = keep_answers_that_have_a_comparison_answer(answer_records, comparison_by_prompt_gid)
        log(f"{skipped_without_comparison_answer} answers skipped for having no comparison answer of condition {comparison_condition}")
    check_or_write_judge_settings(judge_folder, {
        "judge_prompt_file_name": judge_prompt_file_name, "judge_prompt_sha256": judge_prompt_sha256,
        "reply_schema": schema, "reply_schema_sha256": sha256_of_text(json.dumps(schema, sort_keys=True)),
        "comparison_condition": comparison_condition, "created_at": now().isoformat(timespec="seconds")})
    stamp = now().strftime("%Y-%m-%d_%H-%M-%S")
    verdict_path, usage_path = judge_folder / f"verdicts_{stamp}.jsonl", judge_folder / f"usage_log_{stamp}.jsonl"
    already = load_already_judged_answer_ids(judge_folder)
    pending = [r for r in answer_records if r["answer_id"] not in already]
    if maximum_answers is not None:
        pending = pending[:maximum_answers]
    log(f"{len(already)} answers already judged; {len(pending)} to judge now, writing {verdict_path.name}")
    write_lock = threading.Lock()
    started = last_report = time.time()
    counts = {"judged": 0, "unparseable": 0, "stopped_for_usage_limit": False, "failed_calls": 0}
    total_cost = 0.0
    with open(verdict_path, "a", encoding="utf-8") as verdict_file, open(usage_path, "a", encoding="utf-8") as usage_file, \
            ThreadPoolExecutor(max_workers=parallel_calls) as pool:
        futures = [pool.submit(judge_one_answer, record, template, schema, judge_model, judge_prompt_file_name,
                               judge_prompt_sha256, call_model, now,
                               comparison_by_prompt_gid[record["prompt_gid"]] if comparison_by_prompt_gid is not None else None)
                   for record in pending]
        for future in as_completed(futures):
            try:
                verdict_record, usage_records = future.result()
            except ClaudeCodeUsageLimitReached as error:
                if not counts["stopped_for_usage_limit"]:
                    log(f"Usage limit reached ({error}); stopping. Rerun to resume.")
                counts["stopped_for_usage_limit"] = True
                for other in futures:
                    other.cancel()
                continue
            except Exception as error:  # the answer stays unjudged and is retried on resume
                counts["failed_calls"] += 1
                log(f"Call failed for one answer, left for resume: {error}")
                continue
            with write_lock:
                verdict_file.write(json.dumps(verdict_record, ensure_ascii=False) + "\n")
                verdict_file.flush()
                for usage_record in usage_records:
                    usage_file.write(json.dumps(usage_record) + "\n")
                    total_cost += usage_record["cli_reported_cost_usd"] or 0.0
                usage_file.flush()
            counts["judged"] += 1
            counts["unparseable"] += verdict_record["parse_status"] == "unparseable"
            if time.time() - last_report >= progress_interval_seconds:
                last_report = time.time()
                log(f"[{(last_report - started) / 60:.1f} min] judged {counts['judged']}/{len(pending)}, "
                    f"unparseable {counts['unparseable']}, CLI-reported cost so far ${total_cost:.4f}")
    log(f"Done in {(time.time() - started) / 60:.1f} min: judged {counts['judged']}, unparseable {counts['unparseable']}, "
        f"failed calls {counts['failed_calls']}, CLI-reported cost ${total_cost:.4f}")
    counts["total_cli_reported_cost_usd"] = total_cost
    counts["skipped_without_comparison_answer"] = skipped_without_comparison_answer
    counts["answers_still_without_verdict"] = len([r for r in answer_records
                                                   if r["answer_id"] not in load_already_judged_answer_ids(judge_folder)])
    return counts


def default_output_parent(answer_paths: list[Path]) -> Path:
    first = Path(answer_paths[0])
    return (first if first.is_dir() else first.parent) / "judge_outputs"


def dry_run(answer_records: list[dict], judge_folder: Path, template: str, maximum_answers=None, log=print,
            comparison_by_prompt_gid: dict[str, dict] | None = None) -> int:
    if comparison_by_prompt_gid is not None:
        answer_records, skipped = keep_answers_that_have_a_comparison_answer(answer_records, comparison_by_prompt_gid)
        log(f"{skipped} answers would be skipped for having no comparison answer")
    already = load_already_judged_answer_ids(judge_folder) if judge_folder.exists() else set()
    pending = [r for r in answer_records if r["answer_id"] not in already]
    if maximum_answers is not None:
        pending = pending[:maximum_answers]
    if pending:
        log(f"\n----- DRY RUN rendered prompt for {pending[0]['answer_id']} -----")
        log(render_judge_prompt(template, pending[0],
                                comparison_by_prompt_gid[pending[0]["prompt_gid"]] if comparison_by_prompt_gid is not None else None))
    log(f"\nDRY RUN: {len(already)} already judged, {len(pending)} pending = {len(pending)} model calls "
        f"(more only if retries are needed). Folder: {judge_folder}. No model call was made.")
    return len(pending)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True, nargs="+", type=Path, help="answers jsonl file(s) or run folder(s)")
    parser.add_argument("--judge-prompt-file", required=True, type=Path)
    schema_group = parser.add_mutually_exclusive_group(required=True)
    schema_group.add_argument("--reply-schema-file", type=Path)
    schema_group.add_argument("--reply-schema-name", choices=sorted(NAMED_REPLY_SCHEMAS))
    parser.add_argument("--judge-model", required=True, help="no default: the judge model is a decision")
    parser.add_argument("--output-parent-folder", type=Path, default=None)
    parser.add_argument("--comparison-answers", nargs="+", type=Path, default=None,
                        help="answers jsonl file(s) or run folder(s) holding the comparison answers; needed iff the template has {comparison_answer_text}")
    parser.add_argument("--comparison-condition", default=None, help="condition of the comparison answers to pair with, e.g. thinking_off")
    parser.add_argument("--only-conditions", nargs="+", default=None, help="judge only answers of these conditions")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--maximum-answers", type=int, default=None)
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS)
    arguments = parser.parse_args(argv)

    template = arguments.judge_prompt_file.read_text(encoding="utf-8").rstrip("\n")
    schema = (json.loads(arguments.reply_schema_file.read_text(encoding="utf-8")) if arguments.reply_schema_file
              else NAMED_REPLY_SCHEMAS[arguments.reply_schema_name])
    answer_records = load_answer_records(arguments.answers)
    comparison_by_prompt_gid = None
    if COMPARISON_PLACEHOLDER in template:
        if not arguments.comparison_answers or not arguments.comparison_condition:
            parser.error(f"the template has {COMPARISON_PLACEHOLDER}: --comparison-answers and --comparison-condition are required")
        comparison_by_prompt_gid = pick_comparison_answer_per_prompt(load_answer_records(arguments.comparison_answers),
                                                                     arguments.comparison_condition)
        if not comparison_by_prompt_gid:
            parser.error(f"no comparison answers of condition {arguments.comparison_condition!r} found")
    elif arguments.comparison_answers or arguments.comparison_condition:
        parser.error(f"--comparison-answers and --comparison-condition need a template with {COMPARISON_PLACEHOLDER}")
    if arguments.only_conditions:
        answer_records = [r for r in answer_records if r["condition"] in arguments.only_conditions]
    output_parent = arguments.output_parent_folder or default_output_parent(arguments.answers)
    judge_folder = output_parent / judge_output_folder_name(arguments.judge_prompt_file, template)
    if arguments.dry_run:
        dry_run(answer_records, judge_folder, template, arguments.maximum_answers, comparison_by_prompt_gid=comparison_by_prompt_gid)
        return 0
    from claude_code_cli_structured_reply_without_paid_api_fallback import call_claude_code_cli_for_structured_reply

    counts = run_judging(answer_records, judge_folder, template, schema, arguments.judge_model, arguments.judge_prompt_file.name,
                         call_claude_code_cli_for_structured_reply, arguments.maximum_answers, parallel_calls=arguments.parallel_calls,
                         comparison_by_prompt_gid=comparison_by_prompt_gid, comparison_condition=arguments.comparison_condition)
    return 3 if counts["stopped_for_usage_limit"] or (arguments.maximum_answers is None and counts["answers_still_without_verdict"]) else 0


if __name__ == "__main__":
    sys.exit(main())
