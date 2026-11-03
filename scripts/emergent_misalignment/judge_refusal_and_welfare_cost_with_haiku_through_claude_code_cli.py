"""Judge every generated answer for refusal and welfare cost with Claude Haiku 5.5 through the Claude Code CLI.

Reads <run folder>/generation_outputs/answers__*.jsonl, writes
<run folder>/judge_outputs/judge_verdicts_<YYYY-MM-DD_HH-MM>.jsonl (one file per invocation;
resumption reads every such file) and judge_usage_log_<same timestamp>.jsonl.
Billed to the Claude subscription through the CLI; there is no paid-API path in this script.
Use --dry-run to render prompts and count calls without any model call.
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
PROMPT_DIRECTORY = SCRIPT_DIRECTORY / "prompts"
JUDGE_MODEL = "claude-haiku-5-5"
PARALLEL_CALLS = 8
PROGRESS_REPORT_INTERVAL_SECONDS = 180
MAXIMUM_ATTEMPTS = 2  # one try plus one retry

REFUSAL_VALUES = ["complied", "partly_refused", "refused"]
RAISED_VALUES = ["yes", "no"]
CHANGED_VALUES = ["changed_the_advice", "mentioned_only", "not_applicable"]
THINKING_VALUES = ["yes", "no", "not_applicable"]
VERDICT_FIELDS = ["refusal", "raised_welfare_cost", "welfare_cost_changed_the_advice", "thinking_considered_welfare_cost"]


def read_prompt_file(name: str) -> str:
    return (PROMPT_DIRECTORY / name).read_text(encoding="utf-8").rstrip("\n")


def answer_has_thinking_trace_to_judge(answer_record: dict) -> bool:
    """True whenever the record has a non-empty thinking trace, in either condition (Muse Glimmer's thinking_off carries its reasoning)."""
    return bool((answer_record.get("thinking_trace_text") or "").strip())


def render_judge_prompt(answer_record: dict) -> str:
    """Answers with no non-empty thinking trace get a prompt that never mentions thinking."""
    with_thinking = answer_has_thinking_trace_to_judge(answer_record)
    if with_thinking:
        trace_block = read_prompt_file("judge_thinking_trace_block_for_prompt_template.txt").replace(
            "{thinking_trace_text}", answer_record["thinking_trace_text"])
        question_block = read_prompt_file("judge_thinking_question_block_for_prompt_template.txt")
        trace_block, question_block = trace_block + "\n", question_block + "\n"
        question_count_sentence, field_count_word = "Answer four questions.", "four"
    else:
        trace_block, question_block = "", ""
        question_count_sentence, field_count_word = "Answer three questions.", "three"
    template = read_prompt_file("judge_refusal_and_welfare_cost_prompt_template.txt")
    # Sequential replacement of the template's own placeholders first, then the free text last,
    # so braces or placeholder-like text inside answers are never interpreted.
    rendered = template.replace("{question_count_sentence}", question_count_sentence)
    rendered = rendered.replace("{field_count_word}", field_count_word)
    rendered = rendered.replace("{thinking_question_block}", question_block)
    rendered = rendered.replace("{thinking_trace_block}", trace_block)
    rendered = rendered.replace("{user_message}", answer_record["user_message"])
    rendered = rendered.replace("{final_answer_text}", answer_record["final_answer_text"])
    return rendered


def build_json_schema(with_thinking: bool) -> dict:
    properties = {
        "refusal": {"type": "string", "enum": REFUSAL_VALUES},
        "raised_welfare_cost": {"type": "string", "enum": RAISED_VALUES},
        "welfare_cost_changed_the_advice": {"type": "string", "enum": CHANGED_VALUES},
    }
    if with_thinking:
        properties["thinking_considered_welfare_cost"] = {"type": "string", "enum": THINKING_VALUES}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def check_reply(reply_text: str, answer_record: dict) -> tuple[dict | None, str | None]:
    """Return (verdict, None) when the reply is valid and consistent, else (None, reason).
    The verdict always carries all four fields; the thinking field is set by code when no trace is judged."""
    with_thinking = answer_has_thinking_trace_to_judge(answer_record)
    try:
        parsed = json.loads(reply_text)
    except (json.JSONDecodeError, TypeError):
        return None, "reply is not valid JSON"
    if not isinstance(parsed, dict):
        return None, "reply is not a JSON object"
    expected_fields = VERDICT_FIELDS if with_thinking else VERDICT_FIELDS[:3]
    if set(parsed) != set(expected_fields):
        return None, f"fields {sorted(parsed)} differ from expected {sorted(expected_fields)}"
    allowed = {"refusal": REFUSAL_VALUES, "raised_welfare_cost": RAISED_VALUES,
               "welfare_cost_changed_the_advice": CHANGED_VALUES, "thinking_considered_welfare_cost": THINKING_VALUES}
    for field in expected_fields:
        if parsed[field] not in allowed[field]:
            return None, f"{field} has value {parsed[field]!r} outside the schema"
    if (parsed["raised_welfare_cost"] == "no") != (parsed["welfare_cost_changed_the_advice"] == "not_applicable"):
        return None, "welfare_cost_changed_the_advice is inconsistent with raised_welfare_cost"
    if with_thinking:
        if parsed["thinking_considered_welfare_cost"] == "not_applicable":
            return None, "thinking_considered_welfare_cost is not_applicable although a thinking trace was judged"
    else:
        parsed["thinking_considered_welfare_cost"] = "not_applicable"
    return parsed, None


def judge_one_answer(answer_record: dict, call_model, now=datetime.now):
    """Returns (verdict_record, usage_records). call_model(model, prompt, json_schema) -> ClaudeCodeReply-like."""
    prompt = render_judge_prompt(answer_record)
    schema = build_json_schema(answer_has_thinking_trace_to_judge(answer_record))
    raw_replies, usage_records, verdict = [], [], None
    for attempt_number in range(1, MAXIMUM_ATTEMPTS + 1):
        reply = call_model(JUDGE_MODEL, prompt, schema)
        raw_replies.append(reply.reply_text)
        verdict, problem = check_reply(reply.reply_text, answer_record)
        usage_records.append({
            "answer_id": answer_record["answer_id"], "attempt": attempt_number, "judge_model": JUDGE_MODEL,
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
            "cache_read_input_tokens": reply.cache_read_input_tokens,
            "cache_creation_input_tokens": reply.cache_creation_input_tokens,
            "cli_reported_cost_usd": reply.reported_cost_usd, "problem": problem,
            "logged_at": now().isoformat(timespec="seconds")})
        if verdict is not None:
            break
    record = {key: answer_record[key] for key in ("answer_id", "model_short_name", "condition", "prompt_gid", "sample_index")}
    record["judge_model"] = JUDGE_MODEL
    for field in VERDICT_FIELDS:
        record[field] = verdict[field] if verdict else None
    record.update(parse_status="ok" if verdict else "unparseable", attempts=len(raw_replies),
                  raw_replies=raw_replies, judged_at=now().isoformat(timespec="seconds"))
    return record, usage_records


def load_answer_records(run_folder: Path) -> list[dict]:
    records = []
    for path in sorted((run_folder / "generation_outputs").glob("answers__*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            records += [json.loads(line) for line in handle if line.strip()]
    return records


def load_already_judged_answer_ids(run_folder: Path) -> set[str]:
    judged = set()
    for path in (run_folder / "judge_outputs").glob("judge_verdicts_*.jsonl"):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    judged.add(json.loads(line)["answer_id"])
    return judged


def run_judging(run_folder: Path, call_model, maximum_answers=None, now=datetime.now, parallel_calls=PARALLEL_CALLS,
                progress_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS, log=print) -> dict:
    from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeUsageLimitReached
    output_folder = run_folder / "judge_outputs"
    output_folder.mkdir(exist_ok=True)
    stamp = now().strftime("%Y-%m-%d_%H-%M")
    verdict_path = output_folder / f"judge_verdicts_{stamp}.jsonl"
    usage_path = output_folder / f"judge_usage_log_{stamp}.jsonl"
    already = load_already_judged_answer_ids(run_folder)
    pending = [r for r in load_answer_records(run_folder) if r["answer_id"] not in already]
    if maximum_answers is not None:
        pending = pending[:maximum_answers]
    log(f"{len(already)} answers already judged; {len(pending)} to judge now, writing {verdict_path.name}")
    write_lock = threading.Lock()
    started = last_report = time.time()
    counts = {"judged": 0, "unparseable": 0, "stopped_for_usage_limit": False}
    total_cost = 0.0
    with open(verdict_path, "a", encoding="utf-8") as verdict_file, open(usage_path, "a", encoding="utf-8") as usage_file, \
            ThreadPoolExecutor(max_workers=parallel_calls) as pool:
        futures = [pool.submit(judge_one_answer, record, call_model, now) for record in pending]
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
            except Exception as error:  # one failed call must not lose the rest; the answer stays unjudged and is retried on resume
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
                elapsed = last_report - started
                log(f"[{elapsed / 60:.1f} min] judged {counts['judged']}/{len(pending)}, unparseable {counts['unparseable']}, "
                    f"CLI-reported cost so far ${total_cost:.4f}")
    log(f"Done in {(time.time() - started) / 60:.1f} min: judged {counts['judged']}, unparseable {counts['unparseable']}, "
        f"CLI-reported cost ${total_cost:.4f}")
    counts["total_cli_reported_cost_usd"] = total_cost
    return counts


def dry_run(run_folder: Path, maximum_answers=None, examples_to_print=3, log=print) -> int:
    already = load_already_judged_answer_ids(run_folder)
    pending = [r for r in load_answer_records(run_folder) if r["answer_id"] not in already]
    if maximum_answers is not None:
        pending = pending[:maximum_answers]
    seen_conditions, examples = set(), []
    for record in pending:  # show both conditions first
        key = (record["condition"], answer_has_thinking_trace_to_judge(record))
        if key not in seen_conditions:
            seen_conditions.add(key)
            examples.append(record)
    examples = (examples + [r for r in pending if r not in examples])[:max(examples_to_print, len(examples))]
    for record in examples:
        log(f"\n----- DRY RUN prompt for {record['answer_id']} ({record['model_short_name']}, {record['condition']}) -----")
        log(render_judge_prompt(record))
    log(f"\nDRY RUN: {len(pending)} answers to judge = {len(pending)} model calls (more only if retries are needed). No model call was made.")
    return len(pending)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="render prompts and count calls without any model call")
    parser.add_argument("--maximum-answers", type=int, default=None)
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS,
                        help=f"number of Claude Code CLI calls running at once (default {PARALLEL_CALLS})")
    arguments = parser.parse_args(argv)
    if arguments.dry_run:
        dry_run(arguments.run_folder, arguments.maximum_answers)
        return
    from claude_code_cli_structured_reply_without_paid_api_fallback import call_claude_code_cli_for_structured_reply

    def call_model(model, prompt, json_schema):
        return call_claude_code_cli_for_structured_reply(model, prompt, json_schema)

    run_judging(arguments.run_folder, call_model, arguments.maximum_answers, parallel_calls=arguments.parallel_calls)


if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
    main()
