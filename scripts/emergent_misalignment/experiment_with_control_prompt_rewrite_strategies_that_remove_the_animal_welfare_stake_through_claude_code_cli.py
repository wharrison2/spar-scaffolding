"""Pilot: which way of rewriting a treatment prompt into a matched control prompt most reliably leaves no animal welfare stake?

A treatment prompt is a user message with an implicit animal welfare stake. A matched control keeps as much of it as
possible but has no such stake. Three rewrite strategies are compared (instructions in prompts/):
  smallest_change_of_the_rewriters_choice
  change_the_harmful_detail_and_keep_the_practice
  replace_the_practice_and_keep_the_animals
(STRATEGY_NAMES is the default list; BASE_TEMPLATE_FILE_NAME_PER_STRATEGY is the single list of every accepted strategy)
Prompt versions: each strategy names its base template in BASE_TEMPLATE_FILE_NAME_PER_STRATEGY. The original three use
control_prompt_rewrite_base_prompt_template.txt; the extra strategy replace_the_practice_and_keep_the_animals_version_2
(accepted by --strategies, not in the default list) uses control_prompt_rewrite_base_prompt_template_version_2.txt and
control_prompt_rewrite_strategy_instructions__replace_the_practice_and_keep_the_animals_version_2.txt. To add version_3, add its
instructions file, its base template file and one entry to that mapping (the strategy name is the instructions file suffix).
Rewrite base templates may also use {original_word_count}, {original_word_count_lower_bound} and {original_word_count_upper_bound}
(whitespace-split words of the source user message; bounds are round(0.9 * count) and round(1.1 * count)).
--judge-prompt-template-file names the judge template inside prompts/ (default: the original
judge_whether_prompt_carries_no_animal_welfare_stake_prompt_template.txt). Provenance: each new rewrite record holds
base_template_file and strategy_instructions_file, each new verdict record holds judge_template_file, and the summary names the
judge template used. Verdicts are keyed by judged_item_id, so one output folder must only ever be judged with one judge template:
if its verdicts file holds a record with another judge_template_file (a record without the field counts as the original default
template) the run stops with an error before any model call. Use a fresh output folder for a different judge template.
Each rewrite is made by Claude Sonnet 5.5 and then checked by a no-stake judge (also Sonnet 5.5) that sees only the message
text, never the source prompt or the strategy. Source prompts are judged too, as a positive control: they should fail.
All calls go through the Claude Code CLI on the subscription; there is no paid-API path.

Input (--source-prompts-file): JSONL with unique_prompt_id, user_message, harm_type, practical_advantage, taxa_category,
taxa_subcategory, spec_id, source. Only records with source == "main_corpus" are eligible (held-out and contributor records
are never used).

Outputs in --output-folder (every file name ends with its date and time):
  selected_source_prompts_<stamp>.jsonl                          the selected source prompts (never reselected on a rerun)
  control_prompt_rewrites_<stamp>.jsonl                          one record per (prompt, strategy) rewrite call (resumable)
  no_animal_welfare_stake_verdicts_<stamp>.jsonl                 one record per judged item (resumable)
  control_prompt_rewrite_strategy_usage_log_<stamp>.jsonl        tokens and CLI-reported cost per call, with a `step` field
  control_prompt_rewrite_strategy_summary_<stamp>.json / .md     rates per strategy and for the source prompts
  control_prompt_rewrites_side_by_side_for_hand_labelling_<stamp>.md   each source prompt beside its rewrites

Steps:
  1. Select: seeded and deterministic, spread evenly over harm_type (shuffle within each harm type with the seed, then take
     prompts round-robin over the sorted harm types until --number-of-source-prompts is reached).
  2. Rewrite: one call per (source prompt, strategy), up to 2 attempts. The rewrite_id is built in code as
     <unique_prompt_id>::<strategy>; the model is never asked to reproduce an identifier.
  3. Judge: one no-stake call for each rewrite with parse_status ok and can_be_rewritten yes, plus (unless
     --skip-judging-source-prompts) one per selected source prompt. judged_item_id is the rewrite_id or
     <unique_prompt_id>::source_prompt.
  4. Summarise (no model call, always recomputed). Two pass definitions are reported. Strict: harm_from_the_request_evident
     is not_evident, animals_in_the_message is non-empty (a rewrite must still contain animals), background_harm_in_the_setting is no, welfare_weighing_advisor_would_answer_differently is no,
     good_answer_would_raise_animal_welfare is no, and user_mentions_animal_welfare is no. Allowing background harm: the
     same without the background_harm_in_the_setting condition. The summary also counts, per strategy, judged
     rewrites with no animals. Rates use all attempted rewrites of the strategy as the
     denominator, so declined, unparseable and not-yet-judged rewrites count against the strategy.

The share of source words kept unchanged is the sum of the matching block sizes of difflib.SequenceMatcher over the
whitespace-split word lists of source and rewrite, divided by the source word count. Lexical similarity to the source is the Jaccard index of the lower-cased word sets (computed here; the shared
lexical_prompt_similarity_measures module needs a fitted vectoriser and sklearn, which is more than this pilot needs).

Resumption: steps 1 to 3 skip work whose output already exists (latest record per id with parse_status ok); unparseable
calls are retried on a rerun; a rewrite the model declined (can_be_rewritten no) is a finished result and is not retried.
If the subscription usage limit is hit, the run stops cleanly, still writes the summary from what exists, and exits with
code 3; rerun the same command to resume. --dry-run selects in memory, prints one rendered prompt per kind and the call
counts, writes nothing and makes no model call.
"""

import argparse
import difflib
import json
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from windows_extended_length_paths import extended_length_path  # noqa: E402
from filter_prompts_where_user_never_mentions_animal_welfare_with_haiku_through_claude_code_cli import (  # noqa: E402
    get_user_message_text, read_prompt_file)
from control_prompt_rewrite_and_judge_reply_helpers import (  # noqa: E402,F401  (moved; re-exported for callers of this script)
    BASE_TEMPLATE_FILE_NAME_PER_STRATEGY, JUDGE_ENUM_VALUES_PER_FIELD, JUDGE_FIELDS, JUDGE_JSON_SCHEMA, JUDGE_STRING_FIELDS,
    NO_STAKE_JUDGE_TEMPLATE_NAME, ORIGINAL_REWRITE_BASE_TEMPLATE_NAME, REWRITE_FIELDS, REWRITE_JSON_SCHEMA,
    STRATEGY_INSTRUCTIONS_NAME_PATTERN, check_judge_reply, check_rewrite_reply, fill_placeholders_in_one_pass,
    parse_json_object_with_exact_fields, passes_allowing_background_harm, passes_strict, render_judge_prompt,
    render_rewrite_prompt)


ALL_STRATEGY_NAMES = list(BASE_TEMPLATE_FILE_NAME_PER_STRATEGY)
STRATEGY_NAMES = ["smallest_change_of_the_rewriters_choice", "change_the_harmful_detail_and_keep_the_practice",
                  "replace_the_practice_and_keep_the_animals"]  # the default when --strategies is omitted
ELIGIBLE_SOURCE = "main_corpus"
DEFAULT_MODEL = "claude-sonnet-5-5"
PARALLEL_CALLS = 8
PROGRESS_REPORT_INTERVAL_SECONDS = 180
MAXIMUM_ATTEMPTS = 2
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
SOURCE_PROMPT_JUDGED_SUFFIX = "source_prompt"
SOURCE_GROUP_NAME = "source_prompts"

SELECTED_PREFIX = "selected_source_prompts"
REWRITES_PREFIX = "control_prompt_rewrites"
VERDICTS_PREFIX = "no_animal_welfare_stake_verdicts"
USAGE_PREFIX = "control_prompt_rewrite_strategy_usage_log"
SUMMARY_PREFIX = "control_prompt_rewrite_strategy_summary"
SIDE_BY_SIDE_PREFIX = "control_prompt_rewrites_side_by_side_for_hand_labelling"

# ---------------------------------------------------------------- files

def find_existing_file(output_folder: Path, prefix: str) -> Path | None:
    files = sorted(output_folder.glob(f"{prefix}_20*.jsonl")) if output_folder.exists() else []
    return files[-1] if files else None


def read_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_record_per_key(records: list[dict], key_field: str) -> dict:
    return {record[key_field]: record for record in records}  # later lines win


def check_verdicts_were_judged_with_the_requested_judge_template(verdict_records: list[dict], judge_template_file: str) -> None:
    """Verdicts are keyed by judged_item_id, so a folder must only ever be judged with one judge template. A record without
    the judge_template_file field was made by the original default template."""
    templates_in_folder = {record.get("judge_template_file", NO_STAKE_JUDGE_TEMPLATE_NAME) for record in verdict_records}
    differing = sorted(templates_in_folder - {judge_template_file})
    if differing:
        raise ValueError(
            f"The verdicts file in the output folder already holds verdicts made with judge template(s) {differing}, but "
            f"{judge_template_file!r} was requested. One output folder must only be judged with one judge template; use a "
            f"new output folder or request the judge template already used there. No model call was made.")


def make_rewrite_id(unique_prompt_id: str, strategy: str) -> str:
    return f"{unique_prompt_id}::{strategy}"


def make_source_judged_item_id(unique_prompt_id: str) -> str:
    return f"{unique_prompt_id}::{SOURCE_PROMPT_JUDGED_SUFFIX}"


# ---------------------------------------------------------------- step 1: selection

def select_source_prompts(records: list[dict], number_of_prompts: int, selection_seed: int) -> list[dict]:
    eligible = [record for record in records if record.get("source") == ELIGIBLE_SOURCE]
    prompts_per_harm_type: dict[str, list[dict]] = {}
    for record in sorted(eligible, key=lambda record: record["unique_prompt_id"]):  # file order never matters
        prompts_per_harm_type.setdefault(record["harm_type"], []).append(record)
    harm_types = sorted(prompts_per_harm_type)
    shuffler = random.Random(selection_seed)
    for harm_type in harm_types:
        shuffler.shuffle(prompts_per_harm_type[harm_type])
    selected, round_number = [], 0
    while len(selected) < number_of_prompts and any(round_number < len(group) for group in prompts_per_harm_type.values()):
        for harm_type in harm_types:
            if len(selected) < number_of_prompts and round_number < len(prompts_per_harm_type[harm_type]):
                selected.append(prompts_per_harm_type[harm_type][round_number])
        round_number += 1
    return selected


def load_or_make_selection(source_prompts_file: Path, output_folder: Path, number_of_prompts: int, selection_seed: int,
                           stamp: str, log=print) -> list[dict]:
    existing = find_existing_file(output_folder, SELECTED_PREFIX)
    if existing is not None:
        selected = read_jsonl(existing)
        log(f"Step 1: reusing the saved selection {existing.name} ({len(selected)} prompts); not reselecting")
        return selected
    selected = select_source_prompts(read_jsonl(source_prompts_file), number_of_prompts, selection_seed)
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / f"{SELECTED_PREFIX}_{stamp}.jsonl"
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected), encoding="utf-8")
    log(f"Step 1: selected {len(selected)} main_corpus prompts over {len({r['harm_type'] for r in selected})} harm types "
        f"-> {path.name}")
    return selected


# ---------------------------------------------------------------- model calls

def call_with_retries(step: str, item_id: str, prompt: str, json_schema: dict, check_reply, model: str, call_model, now):
    """Returns (parsed_or_None, raw_replies, usage_records)."""
    raw_replies, usage_records, parsed = [], [], None
    for attempt_number in range(1, MAXIMUM_ATTEMPTS + 1):
        reply = call_model(model, prompt, json_schema)
        raw_replies.append(reply.reply_text)
        parsed, problem = check_reply(reply.reply_text)
        usage_records.append({
            "step": step, "item_id": item_id, "attempt": attempt_number, "model": model,
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
            "cache_read_input_tokens": reply.cache_read_input_tokens,
            "cache_creation_input_tokens": reply.cache_creation_input_tokens,
            "number_of_tools_loaded": reply.number_of_tools_loaded,
            "names_of_non_built_in_tools": list(reply.names_of_non_built_in_tools),
            "number_of_mcp_servers_loaded": reply.number_of_mcp_servers_loaded,
            "cli_reported_cost_usd": reply.reported_cost_usd, "problem": problem,
            "logged_at": now().isoformat(timespec="seconds")})
        if parsed is not None:
            break
    return parsed, raw_replies, usage_records


def add_usage_totals(record: dict, usage_records: list[dict]) -> None:
    for field in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        record[field] = sum(usage[field] for usage in usage_records)
    record["cli_reported_cost_usd"] = sum(usage["cli_reported_cost_usd"] or 0.0 for usage in usage_records)


def rewrite_one_prompt(source_prompt: dict, strategy: str, call_model, model: str, now):
    prompt = render_rewrite_prompt(source_prompt, strategy)
    rewrite_id = make_rewrite_id(source_prompt["unique_prompt_id"], strategy)
    parsed, raw_replies, usage_records = call_with_retries("rewrite", rewrite_id, prompt, REWRITE_JSON_SCHEMA,
                                                           check_rewrite_reply, model, call_model, now)
    record = {"rewrite_id": rewrite_id, "unique_prompt_id": source_prompt["unique_prompt_id"], "strategy": strategy,
              "rewrite_model": model, "base_template_file": BASE_TEMPLATE_FILE_NAME_PER_STRATEGY[strategy],
              "strategy_instructions_file": STRATEGY_INSTRUCTIONS_NAME_PATTERN.format(strategy=strategy)}
    for field in REWRITE_FIELDS:
        record[field] = parsed[field] if parsed else None
    record.update(parse_status="ok" if parsed else "unparseable", attempts=len(raw_replies))
    add_usage_totals(record, usage_records)
    record.update(raw_replies=raw_replies, rewritten_at=now().isoformat(timespec="seconds"))
    return record, usage_records


def judge_one_item(work_item: dict, call_model, model: str, now, judge_template_file: str = NO_STAKE_JUDGE_TEMPLATE_NAME):
    prompt = render_judge_prompt(work_item["message_text"], judge_template_file)  # the judge sees only the message text
    parsed, raw_replies, usage_records = call_with_retries("judge", work_item["judged_item_id"], prompt, JUDGE_JSON_SCHEMA,
                                                           check_judge_reply, model, call_model, now)
    record = {"judged_item_id": work_item["judged_item_id"], "kind": work_item["kind"],
              "unique_prompt_id": work_item["unique_prompt_id"], "strategy": work_item["strategy"], "judge_model": model,
              "judge_template_file": judge_template_file}
    for field in JUDGE_FIELDS:
        record[field] = parsed[field] if parsed else None
    if parsed and "harm_from_the_request_text_given_although_not_evident" in parsed:
        record["harm_from_the_request_text_given_although_not_evident"] = parsed[
            "harm_from_the_request_text_given_although_not_evident"]
    record["passes_strict"] = passes_strict(parsed) if parsed else None
    record["passes_allowing_background_harm"] = passes_allowing_background_harm(parsed) if parsed else None
    record.update(parse_status="ok" if parsed else "unparseable", attempts=len(raw_replies))
    add_usage_totals(record, usage_records)
    record.update(raw_replies=raw_replies, judged_at=now().isoformat(timespec="seconds"))
    return record, usage_records


def run_calls_in_parallel(step_name: str, work_items: list, process_one_item, record_path: Path, usage_path: Path,
                          parallel_calls: int, progress_interval_seconds: float, log) -> dict:
    """Runs process_one_item(item) -> (record, usage_records) on a thread pool; each record is appended and flushed as it
    finishes. Returns counts, tokens, cost and whether the usage limit stopped the step."""
    from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeUsageLimitReached
    result = {"step": step_name, "calls_made_for_items": 0, "unparseable": 0, "stopped_for_usage_limit": False,
              "cost_usd": 0.0, "tokens": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                                          "cache_creation_input_tokens": 0}}
    if not work_items:
        log(f"{step_name}: nothing to do")
        return result
    write_lock = threading.Lock()
    started = last_report = time.time()
    with open(record_path, "a", encoding="utf-8") as record_file, open(usage_path, "a", encoding="utf-8") as usage_file, \
            ThreadPoolExecutor(max_workers=parallel_calls) as pool:
        futures = [pool.submit(process_one_item, item) for item in work_items]
        for future in as_completed(futures):
            try:
                record, usage_records = future.result()
            except ClaudeCodeUsageLimitReached as error:
                if not result["stopped_for_usage_limit"]:
                    log(f"{step_name}: usage limit reached ({error}); stopping. Rerun to resume.")
                result["stopped_for_usage_limit"] = True
                for other in futures:
                    other.cancel()
                continue
            except Exception as error:  # one failed call must not lose the rest; the item is retried on resume
                log(f"{step_name}: call failed for one item, left for resume: {error}")
                continue
            with write_lock:
                record_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_file.flush()
                for usage_record in usage_records:
                    usage_file.write(json.dumps(usage_record) + "\n")
                    result["cost_usd"] += usage_record["cli_reported_cost_usd"] or 0.0
                    for token_field in result["tokens"]:
                        result["tokens"][token_field] += usage_record[token_field]
                usage_file.flush()
            result["calls_made_for_items"] += 1
            result["unparseable"] += record["parse_status"] == "unparseable"
            if time.time() - last_report >= progress_interval_seconds:
                last_report = time.time()
                log(f"[{(last_report - started) / 60:.1f} min] {step_name}: {result['calls_made_for_items']}/{len(work_items)} "
                    f"items done, unparseable {result['unparseable']}, tokens {result['tokens']}, "
                    f"CLI-reported cost so far ${result['cost_usd']:.4f}")
    log(f"{step_name} finished in {(time.time() - started) / 60:.1f} min: {result['calls_made_for_items']} items done, "
        f"unparseable {result['unparseable']}, tokens {result['tokens']}, CLI-reported cost ${result['cost_usd']:.4f}")
    return result


# ---------------------------------------------------------------- work lists

def list_rewrites_still_to_make(selected: list[dict], strategies: list[str], rewrite_records: list[dict]) -> list[tuple]:
    done = {rewrite_id for rewrite_id, record in latest_record_per_key(rewrite_records, "rewrite_id").items()
            if record["parse_status"] == "ok"}
    return [(prompt, strategy) for prompt in selected for strategy in strategies
            if make_rewrite_id(prompt["unique_prompt_id"], strategy) not in done]


def list_items_to_judge(selected: list[dict], strategies: list[str], rewrite_records: list[dict],
                        verdict_records: list[dict], skip_judging_source_prompts: bool) -> list[dict]:
    """Every judgeable item without an ok verdict: rewrites that parsed and were not declined, and (optionally) sources."""
    done = {item_id for item_id, record in latest_record_per_key(verdict_records, "judged_item_id").items()
            if record["parse_status"] == "ok"}
    latest_rewrites = latest_record_per_key(rewrite_records, "rewrite_id")
    items = []
    for prompt in selected:
        unique_prompt_id = prompt["unique_prompt_id"]
        for strategy in strategies:
            rewrite = latest_rewrites.get(make_rewrite_id(unique_prompt_id, strategy))
            if rewrite and rewrite["parse_status"] == "ok" and rewrite["can_be_rewritten"] == "yes" \
                    and rewrite["rewrite_id"] not in done:
                items.append({"judged_item_id": rewrite["rewrite_id"], "kind": "rewrite", "unique_prompt_id": unique_prompt_id,
                              "strategy": strategy, "message_text": rewrite["rewritten_user_message"]})
        source_item_id = make_source_judged_item_id(unique_prompt_id)
        if not skip_judging_source_prompts and source_item_id not in done:
            items.append({"judged_item_id": source_item_id, "kind": "source_prompt", "unique_prompt_id": unique_prompt_id,
                          "strategy": None, "message_text": get_user_message_text(prompt)})
    return items


# ---------------------------------------------------------------- step 4: summary

def share_of_source_words_kept_unchanged(source_text: str, rewritten_text: str) -> float:
    source_words, rewritten_words = source_text.split(), rewritten_text.split()
    matcher = difflib.SequenceMatcher(None, source_words, rewritten_words, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / max(1, len(source_words))


def word_set(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def word_set_jaccard(text_a: str, text_b: str) -> float:
    set_a, set_b = word_set(text_a), word_set(text_b)
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0.0


def median_or_none(values: list[float]):
    return round(statistics.median(values), 4) if values else None


def count_judge_field_values(verdicts: list[dict]) -> dict:
    return {field: {value: sum(verdict[field] == value for verdict in verdicts) for value in values}
            for field, values in JUDGE_ENUM_VALUES_PER_FIELD.items()}


def rate_text(count: int, denominator: int) -> str:
    return f"{count}/{denominator} ({count / denominator:.0%})" if denominator else f"{count}/0 (n/a)"


def summarise_group(group_name: str, prompts: list[dict], strategy: str | None, rewrites_by_id: dict, verdicts_by_id: dict,
                    skip_judging_source_prompts: bool) -> dict:
    """strategy None means the source prompts (positive control)."""
    source_by_id = {prompt["unique_prompt_id"]: prompt for prompt in prompts}
    if strategy is None:
        attempted_ids = [prompt["unique_prompt_id"] for prompt in prompts]
        judged_item_id_for = make_source_judged_item_id
        declined = unparseable_rewrites = 0
        judgeable_ids = attempted_ids
        ratios, similarities, kept_shares = [], [], []
    else:
        records = [rewrites_by_id[make_rewrite_id(pid, strategy)] for pid in source_by_id
                   if make_rewrite_id(pid, strategy) in rewrites_by_id]
        attempted_ids = [record["unique_prompt_id"] for record in records]
        judged_item_id_for = lambda pid: make_rewrite_id(pid, strategy)  # noqa: E731
        declined = sum(r["parse_status"] == "ok" and r["can_be_rewritten"] == "no" for r in records)
        unparseable_rewrites = sum(r["parse_status"] != "ok" for r in records)
        judgeable_ids = [r["unique_prompt_id"] for r in records if r["parse_status"] == "ok" and r["can_be_rewritten"] == "yes"]
        ok_rewrites = [r for r in records if r["parse_status"] == "ok" and r["can_be_rewritten"] == "yes"]
        ratios = [len(r["rewritten_user_message"].split()) / max(1, len(get_user_message_text(source_by_id[r["unique_prompt_id"]]).split()))
                  for r in ok_rewrites]
        kept_shares = [share_of_source_words_kept_unchanged(get_user_message_text(source_by_id[r["unique_prompt_id"]]),
                                                            r["rewritten_user_message"]) for r in ok_rewrites]
        similarities = [word_set_jaccard(r["rewritten_user_message"], get_user_message_text(source_by_id[r["unique_prompt_id"]]))
                        for r in ok_rewrites]
    judging_applies = not (strategy is None and skip_judging_source_prompts)
    ok_verdicts = [verdicts_by_id[judged_item_id_for(pid)] for pid in judgeable_ids
                   if judged_item_id_for(pid) in verdicts_by_id and verdicts_by_id[judged_item_id_for(pid)]["parse_status"] == "ok"]
    judged_with_no_animals = sum(not v["animals_in_the_message"].strip() for v in ok_verdicts)
    judge_unparseable = sum(1 for pid in judgeable_ids if judged_item_id_for(pid) in verdicts_by_id
                            and verdicts_by_id[judged_item_id_for(pid)]["parse_status"] != "ok")
    not_yet_judged = (len(judgeable_ids) - len(ok_verdicts) - judge_unparseable) if judging_applies else None
    strict_count = sum(passes_strict(v) for v in ok_verdicts)
    background_count = sum(passes_allowing_background_harm(v) for v in ok_verdicts)
    strict_by_harm_type: dict = {}
    for pid in attempted_ids:
        cell = strict_by_harm_type.setdefault(source_by_id[pid]["harm_type"], {"attempted": 0, "strict_passes": 0})
        cell["attempted"] += 1
        verdict = verdicts_by_id.get(judged_item_id_for(pid))
        cell["strict_passes"] += bool(verdict and verdict["parse_status"] == "ok" and passes_strict(verdict))
    return {
        "group": group_name, "number_attempted": len(attempted_ids),
        "number_not_yet_attempted": len(prompts) - len(attempted_ids),
        "number_declined_by_rewriter": declined, "number_unparseable_rewrites": unparseable_rewrites,
        "number_judge_unparseable": judge_unparseable,
        "number_judged_with_no_animals": judged_with_no_animals, "number_not_yet_judged": not_yet_judged,
        "strict_pass_count": strict_count, "allowing_background_harm_pass_count": background_count,
        "strict_pass_rate": strict_count / len(attempted_ids) if attempted_ids and judging_applies else None,
        "allowing_background_harm_pass_rate": background_count / len(attempted_ids) if attempted_ids and judging_applies else None,
        "judge_field_value_counts": count_judge_field_values(ok_verdicts),
        "median_rewrite_to_source_word_count_ratio": median_or_none(ratios),
        "median_word_set_jaccard_similarity_to_source": median_or_none(similarities),
        "median_share_of_source_words_kept_unchanged": median_or_none(kept_shares),
        "strict_passes_by_harm_type": strict_by_harm_type, "judging_applies": judging_applies}


def build_summary(selected: list[dict], strategies: list[str], rewrite_records: list[dict], verdict_records: list[dict],
                  skip_judging_source_prompts: bool, judge_template_file: str = NO_STAKE_JUDGE_TEMPLATE_NAME) -> dict:
    rewrites_by_id = latest_record_per_key(rewrite_records, "rewrite_id")
    verdicts_by_id = latest_record_per_key(verdict_records, "judged_item_id")
    groups = [summarise_group(strategy, selected, strategy, rewrites_by_id, verdicts_by_id, skip_judging_source_prompts)
              for strategy in strategies]
    groups.append(summarise_group(SOURCE_GROUP_NAME, selected, None, rewrites_by_id, verdicts_by_id, skip_judging_source_prompts))
    return {"number_of_source_prompts": len(selected), "strategies": strategies,
            "judge_template_file": judge_template_file,
            "strict_definition": "harm_from_the_request_evident == not_evident AND background_harm_in_the_setting == no AND "
                                 "welfare_weighing_advisor_would_answer_differently == no AND "
                                 "good_answer_would_raise_animal_welfare == no AND user_mentions_animal_welfare == no",
            "allowing_background_harm_definition": "the strict definition without the background_harm_in_the_setting condition",
            "rate_denominator": "all attempted rewrites of the strategy (selected prompts for the source group)",
            "groups": groups}


def format_summary_markdown(summary: dict) -> str:
    lines = ["# Control prompt rewrite strategies: no-stake pass rates", "",
             f"Source prompts selected: {summary['number_of_source_prompts']}.",
             f"Judge template file: {summary['judge_template_file']}.",
             f"Strict pass: {summary['strict_definition']}.",
             f"Allowing background harm: {summary['allowing_background_harm_definition']}.",
             f"Every rate's denominator is the {summary['rate_denominator']}; its count is shown beside it.", ""]
    for group in summary["groups"]:
        attempted = group["number_attempted"]
        lines += [f"## {group['group']}", "",
                  f"- attempted: {attempted} (not yet attempted: {group['number_not_yet_attempted']})",
                  f"- declined by the rewriter: {group['number_declined_by_rewriter']}",
                  f"- unparseable rewrites: {group['number_unparseable_rewrites']}",
                  f"- judge unparseable: {group['number_judge_unparseable']}; not yet judged: {group['number_not_yet_judged']}",
                  f"- judged with no animals in the message: {group['number_judged_with_no_animals']}"]
        if group["judging_applies"]:
            lines += [f"- strict pass: {rate_text(group['strict_pass_count'], attempted)}",
                      f"- pass allowing background harm: {rate_text(group['allowing_background_harm_pass_count'], attempted)}"]
        else:
            lines.append("- judging of source prompts was skipped")
        lines += [f"- median ratio of rewrite word count to source word count: {group['median_rewrite_to_source_word_count_ratio']}",
                  f"- median word-set Jaccard similarity to source: {group['median_word_set_jaccard_similarity_to_source']}",
                  f"- median share of source words kept unchanged: {group['median_share_of_source_words_kept_unchanged']}",
                  "- judge field value counts (over parsed verdicts):"]
        for field, counts in group["judge_field_value_counts"].items():
            lines.append(f"  - {field}: " + ", ".join(f"{value} {count}" for value, count in counts.items()))
        lines.append("")
    harm_types = sorted({harm_type for group in summary["groups"] for harm_type in group["strict_passes_by_harm_type"]})
    group_names = [group["group"] for group in summary["groups"]]
    lines += ["## Strict passes by strategy and harm type (strict passes / attempted)", "",
              "| harm type | " + " | ".join(group_names) + " |", "|---|" + "---|" * len(group_names)]
    for harm_type in harm_types:
        cells = []
        for group in summary["groups"]:
            cell = group["strict_passes_by_harm_type"].get(harm_type)
            cells.append(f"{cell['strict_passes']}/{cell['attempted']}" if cell else "0/0")
        lines.append(f"| {harm_type} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def format_side_by_side_markdown(selected: list[dict], strategies: list[str], rewrite_records: list[dict],
                                 verdict_records: list[dict]) -> str:
    rewrites_by_id = latest_record_per_key(rewrite_records, "rewrite_id")
    verdicts_by_id = latest_record_per_key(verdict_records, "judged_item_id")
    lines = ["# Control prompt rewrites side by side, for hand labelling", ""]
    for prompt in selected:
        lines += [f"## {prompt['unique_prompt_id']}", "", f"- harm type: {prompt['harm_type']}",
                  f"- practical advantage: {prompt['practical_advantage']}", "", "Source prompt:", "",
                  get_user_message_text(prompt), ""]
        for strategy in strategies:
            rewrite = rewrites_by_id.get(make_rewrite_id(prompt["unique_prompt_id"], strategy))
            lines += [f"### {strategy}", ""]
            if rewrite is None:
                lines += ["Not yet rewritten.", ""]
                continue
            if rewrite["parse_status"] != "ok":
                lines += ["Rewrite reply was unparseable.", ""]
                continue
            lines += [f"What was changed: {rewrite['what_was_changed']}", ""]
            if rewrite["can_be_rewritten"] == "no":
                lines += ["The rewriter declined (can_be_rewritten no).", ""]
                continue
            verdict = verdicts_by_id.get(rewrite["rewrite_id"])
            kept = share_of_source_words_kept_unchanged(get_user_message_text(prompt), rewrite["rewritten_user_message"])
            lines += [f"Share of source words kept unchanged: {kept:.0%}", "", "Rewritten prompt:", "",
                      rewrite["rewritten_user_message"], ""]
            if verdict is None or verdict["parse_status"] != "ok":
                lines.append("Judge: not yet judged or unparseable.")
            else:
                fields = ", ".join(f"{field} {verdict[field]}" for field in JUDGE_ENUM_VALUES_PER_FIELD)
                lines.append(f"Judge: {fields}. Note: {verdict['note']}")
            lines += ["", "Hand label (no stake: yes / no):", ""]
    return "\n".join(lines) + "\n"


def write_summary_files(output_folder: Path, stamp: str, selected, strategies, rewrite_records, verdict_records,
                        skip_judging_source_prompts: bool, judge_template_file: str = NO_STAKE_JUDGE_TEMPLATE_NAME) -> dict:
    summary = build_summary(selected, strategies, rewrite_records, verdict_records, skip_judging_source_prompts,
                            judge_template_file)
    (output_folder / f"{SUMMARY_PREFIX}_{stamp}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_folder / f"{SUMMARY_PREFIX}_{stamp}.md").write_text(format_summary_markdown(summary), encoding="utf-8")
    (output_folder / f"{SIDE_BY_SIDE_PREFIX}_{stamp}.md").write_text(
        format_side_by_side_markdown(selected, strategies, rewrite_records, verdict_records), encoding="utf-8")
    return summary


# ---------------------------------------------------------------- orchestration

def run_experiment(source_prompts_file: Path, output_folder: Path, call_model, number_of_source_prompts=30,
                   selection_seed=20261112, strategies=None, skip_judging_source_prompts=False,
                   parallel_calls=PARALLEL_CALLS, rewrite_model=DEFAULT_MODEL, judge_model=DEFAULT_MODEL, now=datetime.now,
                   progress_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS, log=print,
                   judge_template_file=NO_STAKE_JUDGE_TEMPLATE_NAME) -> dict:
    strategies = list(strategies) if strategies else list(STRATEGY_NAMES)
    stamp = now().strftime(TIMESTAMP_FORMAT)
    output_folder.mkdir(parents=True, exist_ok=True)
    selected = load_or_make_selection(source_prompts_file, output_folder, number_of_source_prompts, selection_seed, stamp, log)
    rewrites_path = find_existing_file(output_folder, REWRITES_PREFIX) or output_folder / f"{REWRITES_PREFIX}_{stamp}.jsonl"
    verdicts_path = find_existing_file(output_folder, VERDICTS_PREFIX) or output_folder / f"{VERDICTS_PREFIX}_{stamp}.jsonl"
    usage_path = find_existing_file(output_folder, USAGE_PREFIX) or output_folder / f"{USAGE_PREFIX}_{stamp}.jsonl"
    check_verdicts_were_judged_with_the_requested_judge_template(read_jsonl(verdicts_path), judge_template_file)

    rewrites_to_make = list_rewrites_still_to_make(selected, strategies, read_jsonl(rewrites_path))
    log(f"Step 2: {len(rewrites_to_make)} of {len(selected) * len(strategies)} rewrites to make")
    rewrite_result = run_calls_in_parallel(
        "rewrite", rewrites_to_make, lambda pair: rewrite_one_prompt(pair[0], pair[1], call_model, rewrite_model, now),
        rewrites_path, usage_path, parallel_calls, progress_interval_seconds, log)

    judge_result = {"stopped_for_usage_limit": False}
    if not rewrite_result["stopped_for_usage_limit"]:
        items_to_judge = list_items_to_judge(selected, strategies, read_jsonl(rewrites_path), read_jsonl(verdicts_path),
                                             skip_judging_source_prompts)
        log(f"Step 3: {len(items_to_judge)} items to judge")
        judge_result = run_calls_in_parallel(
            "judge", items_to_judge, lambda item: judge_one_item(item, call_model, judge_model, now, judge_template_file),
            verdicts_path, usage_path, parallel_calls, progress_interval_seconds, log)

    summary = write_summary_files(output_folder, stamp, selected, strategies, read_jsonl(rewrites_path),
                                  read_jsonl(verdicts_path), skip_judging_source_prompts, judge_template_file)
    summary["stopped_for_usage_limit"] = rewrite_result["stopped_for_usage_limit"] or judge_result["stopped_for_usage_limit"]
    summary["total_cli_reported_cost_usd_this_run"] = rewrite_result["cost_usd"] + judge_result.get("cost_usd", 0.0)
    log(f"Step 4: summary written ({SUMMARY_PREFIX}_{stamp}.md). Total CLI-reported cost this run "
        f"${summary['total_cli_reported_cost_usd_this_run']:.4f}")
    return summary


def dry_run(source_prompts_file: Path, number_of_source_prompts: int, selection_seed: int, strategies: list[str],
            skip_judging_source_prompts: bool, log=print, judge_template_file: str = NO_STAKE_JUDGE_TEMPLATE_NAME) -> dict:
    selected = select_source_prompts(read_jsonl(source_prompts_file), number_of_source_prompts, selection_seed)
    log(f"DRY RUN: selected {len(selected)} main_corpus prompts in memory over "
        f"{len({prompt['harm_type'] for prompt in selected})} harm types; nothing is written")
    if selected:
        for strategy in strategies:
            log(f"\n----- DRY RUN rewrite prompt for {selected[0]['unique_prompt_id']}, strategy {strategy} -----")
            log(render_rewrite_prompt(selected[0], strategy))
        log(f"\n----- DRY RUN judge prompt (message text of {selected[0]['unique_prompt_id']}) -----")
        log(render_judge_prompt(get_user_message_text(selected[0]), judge_template_file))
    rewrite_calls = len(selected) * len(strategies)
    judge_calls = rewrite_calls + (0 if skip_judging_source_prompts else len(selected))
    log(f"\nDRY RUN: step 2 would make {rewrite_calls} rewrite calls ({len(selected)} prompts x {len(strategies)} strategies); "
        f"step 3 would make at most {judge_calls} judge calls (declined or unparseable rewrites are not judged). "
        f"Retries can add more (up to {MAXIMUM_ATTEMPTS} attempts per call). No model call was made.")
    return {"selected": len(selected), "rewrite_calls": rewrite_calls, "maximum_judge_calls": judge_calls}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-prompts-file", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--number-of-source-prompts", type=int, default=30)
    parser.add_argument("--selection-seed", type=int, default=20261112)
    parser.add_argument("--strategies", nargs="+", choices=ALL_STRATEGY_NAMES, default=None,
                        help="strategies to run (default: the original three, i.e. STRATEGY_NAMES)")
    parser.add_argument("--judge-prompt-template-file", default=NO_STAKE_JUDGE_TEMPLATE_NAME,
                        help="file name inside the prompts folder of the judge template (default: the original judge template); "
                             "one output folder must only be judged with one judge template")
    parser.add_argument("--skip-judging-source-prompts", action="store_true",
                        help="do not judge the source prompts as a positive control")
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS)
    parser.add_argument("--rewrite-model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="render prompts and count calls without any model call or file")
    arguments = parser.parse_args(argv)
    arguments.source_prompts_file = extended_length_path(arguments.source_prompts_file)
    arguments.output_folder = extended_length_path(arguments.output_folder)
    strategies = arguments.strategies or list(STRATEGY_NAMES)
    if arguments.dry_run:
        dry_run(arguments.source_prompts_file, arguments.number_of_source_prompts, arguments.selection_seed, strategies,
                arguments.skip_judging_source_prompts, judge_template_file=arguments.judge_prompt_template_file)
        return
    from claude_code_cli_structured_reply_without_paid_api_fallback import call_claude_code_cli_for_structured_reply

    def call_model(model, prompt, json_schema):
        return call_claude_code_cli_for_structured_reply(model, prompt, json_schema)

    summary = run_experiment(arguments.source_prompts_file, arguments.output_folder, call_model,
                             arguments.number_of_source_prompts, arguments.selection_seed, strategies,
                             arguments.skip_judging_source_prompts, arguments.parallel_calls, arguments.rewrite_model,
                             arguments.judge_model, judge_template_file=arguments.judge_prompt_template_file)
    if summary["stopped_for_usage_limit"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
