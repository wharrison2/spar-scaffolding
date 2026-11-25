"""Rewrite one user lean flip share so each user leans toward the other option, then check the rewrite, as one batch.

Each source prompt is a message whose user leans toward the practically better option. Claude Sonnet 5.5 rewrites it so the
user leans toward the other option, which stays practically worse, for the reason kind dealt to that prompt when the shares were
made (the share's reason_kind field, and for the different priority kind its different_priority_goal field), changing only the lean and the reason; a second Sonnet 5.5 call checks the rewrite. The
share's prompt_arm decides the prompt files (see user_lean_flip_rewrite_and_check_reply_helpers.py): treatment prompts are
rewritten toward the option better for animals, matched control prompts toward the practically worse option. A rewrite that
fails the check gets exactly one revision: the rewriter is called again with the original message, its failed rewrite, the
failed check fields with the values the judge gave and the judge's short_reason, and the revision is checked by the same check
judge. A prompt is accepted when its first rewrite or its revision passes. A prompt that ends declined or rejected is dealt a new
reason kind, chosen at random from the kinds it has not tried (seeded by REDEAL_SEED and its unique_prompt_id, so a resumed run
makes the same choices; for the different priority kind also a goal at random from the goals file), and runs the four steps again as
the next attempt, until it is accepted or has tried every reason kind. The corpus folder layout, completion markers, ledger line,
usage limit handling and resume are those of add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus.py (whose
functions this script uses), so the contributor kit's run_my_shares.py (status, resume, lock, zip) works on it unchanged.

Corpus folder (--corpus-folder): holds user_lean_flip_corpus_settings_share_file_and_share_number.json (share number, the share
file's path inside the folder and its sha256; written by run_my_shares.py --lean-flip or --lean-flip-control), the share file
itself, and grows a batches/ folder and the ledger corpus_batch_ledger_started_2026-11-06.jsonl.

Steps (each leaves a completion marker, named with the attempt number from attempt 2 on, so --resume-batch-folder skips finished
ones; inside a step every finished prompt is saved as it finishes): snapshot the share and the prompt files (with sha256) -> rewrite -> check (only rewrites the rewriter did
not decline) -> revise (only rewrites whose check failed) -> check_revision (only revisions the rewriter did not decline) -> all
records file, accepted prompts file, summary, ledger line (last).
Accepted = every field of the check has a passing value (passes_all_checks). Each check is two calls on the rewritten message: the arm's check and the shared misalignment cue screen, whose fields count like the others. A rewrite the model declined is recorded and not
checked or revised; a revision that is declined, fails or does not pass its check is recorded and not retried. A call whose reply cannot be parsed is
retried (two attempts per call); a prompt still without a usable reply after three runs is recorded as failed and counted in
the summary. When any prompt has no record at all from a step, the batch stops before writing the accepted file, summary and
ledger line (exit code 1). When every call of a check or check_revision step with at least
MINIMUM_ATTEMPTED_PROMPTS_FOR_ALL_CALLS_FAILED_ALARM prompts failed (for example because the API rejected the check reply
schema), it also stops before writing them, with exit code 5 (EVERY_CALL_FAILED_EXIT_CODE). If the subscription usage limit is reached the batch stops cleanly with exit code 3; run the same
command again to resume. Every call goes through the Claude Code command line adapter (subscription only). Tokens and the cost
the command line reports are logged per call. Use --dry-run to print the plan and one rendered prompt and create nothing.

Output files in the batch folder (every name ends with its date and time):
  user_lean_flip_share_prompts_used_for_this_batch_<stamp>/ the share file and prompt files used, with sha256_of_files_used_for_this_batch_<stamp>.json
  user_lean_flip_rewrites_<stamp>.jsonl, user_lean_flip_rewrite_checks_<stamp>.jsonl, user_lean_flip_revisions_<stamp>.jsonl,
  user_lean_flip_revision_checks_<stamp>.jsonl, user_lean_flip_usage_log_<stamp>.jsonl
  user_lean_flip_all_records_batch_<NNN>_<stamp>.jsonl   one record per prompt attempt (attempt_number, reason_kind,
      different_priority_goal, originally_dealt_reason_kind) with its first rewrite, check verdict, revision, revision
      check verdict and accepted_attempt (first_rewrite, revision or null)
  accepted_user_lean_flip_prompts_batch_<NNN>_<stamp>.jsonl   each accepted text (the revision if that is what passed) paired to
      its source by unique_prompt_id, with the final reason kind and goal, the attempt number and originally_dealt_reason_kind
  user_lean_flip_batch_summary_<stamp>.json
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    PROGRESS_REPORT_INTERVAL_SECONDS, TIMESTAMP_FORMAT, USAGE_LIMIT_EXIT_CODE, BatchFailure, BatchStoppedForUsageLimit,
    append_ledger_line, batch_numbers_in_ledger, ledger_batch_folder_text, marker_exists, read_jsonl, read_ledger,
    sha256_of_file, write_marker)
from add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    MAXIMUM_ATTEMPTS_PER_CALL, PARALLEL_CALLS, PROMPT_FOLDER, add_usage_totals, call_until_parsed,
    final_record_per_prompt, find_existing_file, incomplete_batch_folders, next_batch_number, parse_status_of,
    prompt_file_hashes, refuse_prompt_files_changed_since_the_batch_started, run_steps_until_every_prompt_is_settled,
    snapshot_hashes, snapshot_share_and_prompt_files, usage_totals)
from spec_outcomes_and_status_table_for_prompt_corpus import LEDGER_FILE_NAME  # noqa: E402
from shared_misalignment_cue_screen_of_one_user_prompt_helpers import (  # noqa: E402
    check_screen_reply, render_screen_prompt, screen_schema)
from user_lean_flip_rewrite_and_check_reply_helpers import (  # noqa: E402
    ARM_FILES_AND_PASS_RULES, ARMS, DIFFERENT_PRIORITY_REASON_KIND, REASON_KINDS, REWRITE_JSON_SCHEMA, check_check_reply,
    check_fields_of, check_properties_of, check_rewrite_reply, check_schema_of, combine_check_with_screen, passes_all_checks,
    required_values_of, prompt_file_names_used,
    read_different_priority_goals, render_check_prompt, render_revision_prompt, render_rewrite_prompt)
from windows_extended_length_paths import (  # noqa: E402
    extended_length_path, remove_extended_length_prefixes_from_text)

REWRITE_MODEL = "claude-sonnet-5-5"
CHECK_MODEL = "claude-sonnet-5-5"
SETTINGS_FILE_NAME = "user_lean_flip_corpus_settings_share_file_and_share_number.json"
SHARE_FILE_FIELDS = ["corpus_prompt_id", "unique_prompt_id", "user_message", "spec_id", "harm_type", "taxa_category",
                     "taxa_subcategory", "reason_kind", "prompt_arm"]
BATCH_FOLDER_PATTERN = re.compile(r"^batch_(\d{3,})_user_lean_flip_share_(\d{3})_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")
SNAPSHOT_FOLDER_PREFIX = "user_lean_flip_share_prompts_used_for_this_batch"
REWRITES_PREFIX, CHECKS_PREFIX, USAGE_PREFIX = "user_lean_flip_rewrites", "user_lean_flip_rewrite_checks", "user_lean_flip_usage_log"
REVISIONS_PREFIX, REVISION_CHECKS_PREFIX = "user_lean_flip_revisions", "user_lean_flip_revision_checks"
STEP_REWRITE, STEP_CHECK, STEP_SNAPSHOT = "rewrite", "check", "snapshot_share_and_prompt_files"
STEP_REVISE, STEP_CHECK_REVISION = "revise", "check_revision"
STEP_NAMES = (STEP_REWRITE, STEP_CHECK, STEP_REVISE, STEP_CHECK_REVISION)
STEP_PREFIXES = {STEP_REWRITE: REWRITES_PREFIX, STEP_CHECK: CHECKS_PREFIX, STEP_REVISE: REVISIONS_PREFIX,
                 STEP_CHECK_REVISION: REVISION_CHECKS_PREFIX}
CHECK_STEP_NAMES = (STEP_CHECK, STEP_CHECK_REVISION)
MINIMUM_ATTEMPTED_PROMPTS_FOR_ALL_CALLS_FAILED_ALARM = 3
ATTEMPT_FIRST_REWRITE, ATTEMPT_REVISION = "first_rewrite", "revision"
REDEAL_SEED = 20261123  # with the unique_prompt_id, seeds the order in which a prompt tries the reason kinds and the goals it is dealt
EVERY_CALL_FAILED_EXIT_CODE = 5


class EveryCallOfAStepFailed(BatchFailure):
    """A check or check_revision step in which no call produced a usable verdict."""
PLACEHOLDER_REWRITE_FOR_DRY_RUN = "(the rewritten message would appear here)"

OUTCOME_ACCEPTED = "accepted"
OUTCOME_DECLINED = "declined_by_rewriter"
OUTCOME_REWRITE_FAILED = "rewrite_failed_after_every_attempt"
OUTCOME_CHECK_FAILED = "check_failed_after_every_attempt"
OUTCOME_ACCEPTED_AFTER_REVISION = "accepted_after_revision"
OUTCOME_REJECTED = "rejected_by_check"  # the first check failed and the revision was declined, failed or did not pass


# ---------------------------------------------------------------- the corpus folder and the share

def read_settings(corpus_folder: Path) -> dict:
    path = Path(corpus_folder) / SETTINGS_FILE_NAME
    if not path.exists():
        raise BatchFailure(f"{path} does not exist: this is not a user lean flip share folder (run_my_shares.py --lean-flip or "
                           f"--lean-flip-control writes it)")
    return json.loads(path.read_text(encoding="utf-8"))


def read_verified_share_records(corpus_folder: Path, settings: dict) -> tuple[list[dict], str]:
    """(the share's records, the share's prompt arm). Raises BatchFailure on a missing or edited share file, a repeated id,
    a record missing a field, a reason kind or arm outside the known ones, or a share that mixes arms."""
    share_path = Path(corpus_folder) / settings["share_file"]
    if not share_path.exists():
        raise BatchFailure(f"the share file {settings['share_file']} is missing from {corpus_folder}")
    if sha256_of_file(share_path) != settings["share_file_sha256"]:
        raise BatchFailure(f"the share file {settings['share_file']} does not match its checksum in the corpus settings; "
                           f"do not edit share files")
    records = read_jsonl(share_path)
    identifiers = [record["unique_prompt_id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise BatchFailure("the share file repeats a unique_prompt_id")
    for record in records:
        missing = [field for field in SHARE_FILE_FIELDS if field not in record]
        if missing:
            raise BatchFailure(f"share record {record.get('unique_prompt_id')} lacks fields {missing}")
        if record["reason_kind"] not in REASON_KINDS or record["prompt_arm"] not in ARMS:
            raise BatchFailure(f"share record {record['unique_prompt_id']} has reason_kind {record['reason_kind']!r} and "
                               f"prompt_arm {record['prompt_arm']!r}, outside the known values")
        if record["reason_kind"] == DIFFERENT_PRIORITY_REASON_KIND and not record.get("different_priority_goal"):
            raise BatchFailure(f"share record {record['unique_prompt_id']} has the reason kind {DIFFERENT_PRIORITY_REASON_KIND} "
                               f"and no different_priority_goal")
    arms = {record["prompt_arm"] for record in records}
    if len(arms) != 1:
        raise BatchFailure(f"the share holds prompts of {len(arms)} prompt arms; one is needed")
    return records, arms.pop()


def batch_folder_name(batch_number: int, share_number: int, stamp: str) -> str:
    return f"batch_{batch_number:03d}_user_lean_flip_share_{share_number:03d}_{stamp}"


# ---------------------------------------------------------------- re-dealing a reason kind after a failed attempt

def reason_kinds_in_the_order_a_prompt_tries_them(unique_prompt_id: str, dealt_reason_kind: str) -> list[str]:
    """The dealt kind, then the other kinds in a shuffle seeded by REDEAL_SEED and the prompt's id."""
    others = [kind for kind in REASON_KINDS if kind != dealt_reason_kind]
    random.Random(f"{REDEAL_SEED}:{unique_prompt_id}").shuffle(others)
    return [dealt_reason_kind, *others]


def share_record_for_attempt(share_record: dict, attempt_number: int, goals: list[str]) -> dict:
    """The share record with the reason kind (and goal) of the attempt: attempt 1 is as dealt; later attempts take the next kind
    in the prompt's order and, for the different priority kind, a goal chosen at random from goals."""
    identifier, dealt_kind = share_record["unique_prompt_id"], share_record["reason_kind"]
    kind = reason_kinds_in_the_order_a_prompt_tries_them(identifier, dealt_kind)[attempt_number - 1]
    if attempt_number == 1:
        goal = share_record.get("different_priority_goal")
    else:
        goal = (random.Random(f"{REDEAL_SEED}:{identifier}:goal:{attempt_number}").choice(goals)
                if kind == DIFFERENT_PRIORITY_REASON_KIND else None)
    return {**share_record, "originally_dealt_reason_kind": dealt_kind, "reason_kind": kind, "different_priority_goal": goal,
            "attempt_number": attempt_number}


def step_name_for_attempt(step: str, attempt_number: int) -> str:
    """The step's own name for attempt 1 and the name with the attempt number after it for later attempts (the completion marker name)."""
    return step if attempt_number == 1 else f"{step}_attempt_{attempt_number}"


# ---------------------------------------------------------------- model calls, one per prompt

def rewriter_record_of(step: str, share_record: dict, prompt: str, call_model, now):
    parsed, raw_replies, usage_records, problem = call_until_parsed(
        step, share_record["unique_prompt_id"], prompt, REWRITE_JSON_SCHEMA, check_rewrite_reply, REWRITE_MODEL, call_model, now)
    record = {"unique_prompt_id": share_record["unique_prompt_id"], "corpus_prompt_id": share_record["corpus_prompt_id"],
              "attempt_number": share_record.get("attempt_number", 1), "reason_kind": share_record["reason_kind"],
              "different_priority_goal": share_record.get("different_priority_goal"),
              "originally_dealt_reason_kind": share_record.get("originally_dealt_reason_kind", share_record["reason_kind"]),
              "rewrite_model": REWRITE_MODEL,
              "parse_status": parse_status_of(parsed, raw_replies), "problem": None if parsed else problem,
              "attempts": len(usage_records)}
    for field in ("can_be_rewritten", "what_was_changed", "rewritten_user_message"):
        record[field] = parsed[field] if parsed else None
    add_usage_totals(record, usage_records)
    record.update(raw_replies=raw_replies, rewritten_at=now().isoformat(timespec="seconds"))
    return record, usage_records


def rewrite_one_prompt(share_record: dict, arm: str, call_model, now):
    return rewriter_record_of(STEP_REWRITE, share_record, render_rewrite_prompt(share_record, arm), call_model, now)


def revise_one_rewrite(work_item: dict, arm: str, call_model, now):
    """work_item: the share record, the failed rewrite's text and its check record."""
    prompt = render_revision_prompt(work_item["share_record"], arm, work_item["previous_rewrite"], work_item["check"])
    return rewriter_record_of(STEP_REVISE, work_item["share_record"], prompt, call_model, now)


def check_one_rewrite(work_item: dict, arm: str, call_model, now, step: str = STEP_CHECK):
    """Two calls on the rewritten message: the arm's check, then the shared screen (check_fields_of lists the merged fields)."""
    unique_prompt_id = work_item["unique_prompt_id"]
    check, raw_replies, usage_records, problem = call_until_parsed(
        step, unique_prompt_id,
        render_check_prompt(work_item["user_message"], work_item["rewritten_user_message"], arm), check_schema_of(arm),
        lambda reply_text: check_check_reply(reply_text, arm), CHECK_MODEL, call_model, now)
    screen, screen_raw_replies, screen_usage_records, screen_problem = call_until_parsed(
        step, unique_prompt_id, render_screen_prompt(work_item["rewritten_user_message"]), screen_schema(),
        check_screen_reply, CHECK_MODEL, call_model, now)
    parsed = combine_check_with_screen(check, screen) if check and screen else None
    raw_replies, usage_records = raw_replies + screen_raw_replies, usage_records + screen_usage_records
    problem = problem or screen_problem
    record = {"unique_prompt_id": unique_prompt_id, "attempt_number": work_item.get("attempt_number", 1),
              "check_model": CHECK_MODEL,
              "parse_status": parse_status_of(parsed, raw_replies), "problem": None if parsed else problem,
              "attempts": len(usage_records)}
    for field in check_fields_of(arm):
        record[field] = parsed[field] if parsed else None
    record["passes_all_checks"] = passes_all_checks(parsed, arm) if parsed else None
    add_usage_totals(record, usage_records)
    record.update(raw_replies=raw_replies, checked_at=now().isoformat(timespec="seconds"))
    return record, usage_records


# ---------------------------------------------------------------- work lists

def rewrites_to_make(share_records: list[dict], rewrite_records: list[dict]) -> list[dict]:
    settled = final_record_per_prompt(rewrite_records)
    return [record for record in share_records if record["unique_prompt_id"] not in settled]


def usable_rewrite_text(rewrite: dict | None) -> str | None:
    """The rewritten message of a rewriter record that parsed and was not declined, else None."""
    if rewrite and rewrite["parse_status"] == "ok" and rewrite["can_be_rewritten"] == "yes":
        return rewrite["rewritten_user_message"]
    return None


def rewrites_to_check(share_records: list[dict], rewrite_records: list[dict], check_records: list[dict]) -> list[dict]:
    """Rewrites (first rewrites or revisions, whichever records are passed) that parsed, were not declined and have no
    settled check."""
    rewrites, checks = final_record_per_prompt(rewrite_records), final_record_per_prompt(check_records)
    return [{"unique_prompt_id": record["unique_prompt_id"], "attempt_number": record.get("attempt_number", 1),
             "user_message": record["user_message"], "rewritten_user_message": rewrites[record["unique_prompt_id"]]["rewritten_user_message"]}
            for record in share_records
            if record["unique_prompt_id"] in rewrites and rewrites[record["unique_prompt_id"]]["parse_status"] == "ok"
            and rewrites[record["unique_prompt_id"]]["can_be_rewritten"] == "yes"
            and record["unique_prompt_id"] not in checks]


def rewrites_to_revise(share_records: list[dict], rewrite_records: list[dict], check_records: list[dict],
                       revision_records: list[dict]) -> list[dict]:
    """Rewrites whose settled check parsed and did not pass, and that have no settled revision."""
    rewrites, checks = final_record_per_prompt(rewrite_records), final_record_per_prompt(check_records)
    revisions = final_record_per_prompt(revision_records)
    return [{"share_record": record, "previous_rewrite": usable_rewrite_text(rewrites.get(record["unique_prompt_id"])),
             "check": checks[record["unique_prompt_id"]]}
            for record in share_records
            if usable_rewrite_text(rewrites.get(record["unique_prompt_id"])) is not None
            and record["unique_prompt_id"] in checks and checks[record["unique_prompt_id"]]["parse_status"] == "ok"
            and checks[record["unique_prompt_id"]]["passes_all_checks"] is False
            and record["unique_prompt_id"] not in revisions]


def steps_where_every_call_failed(records_per_step: dict[str, list[dict]]) -> list[str]:
    """Check steps that settled at least MINIMUM_ATTEMPTED_PROMPTS_FOR_ALL_CALLS_FAILED_ALARM prompts, none with a usable
    verdict."""
    steps = []
    for step in CHECK_STEP_NAMES:
        settled = final_record_per_prompt(records_per_step[step]).values()
        if len(settled) >= MINIMUM_ATTEMPTED_PROMPTS_FOR_ALL_CALLS_FAILED_ALARM and all(
                record["parse_status"] != "ok" for record in settled):
            steps.append(step)
    return steps


# ---------------------------------------------------------------- outputs

def build_all_records(share_records: list[dict], records_per_step: dict[str, list[dict]], arm: str) -> list[dict]:
    """One record per share record passed (one per prompt attempt when they are the attempts' share records). The first rewrite's fields keep their plain names (rewrite_status, check_verdict, ...); the
    revision's carry the prefix revision_. accepted_attempt says which text was accepted and accepted_rewritten_user_message is it."""
    final = {step: final_record_per_prompt(records) for step, records in records_per_step.items()}
    all_records = []
    for share_record in share_records:
        identifier = share_record["unique_prompt_id"]
        rewrite, check = final[STEP_REWRITE].get(identifier), final[STEP_CHECK].get(identifier)
        revision, revision_check = final[STEP_REVISE].get(identifier), final[STEP_CHECK_REVISION].get(identifier)
        rewrite_ok = bool(rewrite) and rewrite["parse_status"] == "ok"
        check_ok = bool(check) and check["parse_status"] == "ok"
        revision_ok = bool(revision) and revision["parse_status"] == "ok"
        revision_check_ok = bool(revision_check) and revision_check["parse_status"] == "ok"
        record = {field: share_record[field] for field in SHARE_FILE_FIELDS if field != "user_message"}
        record.update(attempt_number=share_record.get("attempt_number", 1),
                      different_priority_goal=share_record.get("different_priority_goal"),
                      originally_dealt_reason_kind=share_record.get("originally_dealt_reason_kind", share_record["reason_kind"]))
        record["original_user_message"] = share_record["user_message"]
        record["rewrite_status"] = rewrite["parse_status"] if rewrite else "not_attempted"
        record["can_be_rewritten"] = rewrite["can_be_rewritten"] if rewrite_ok else None
        record["what_was_changed"] = rewrite["what_was_changed"] if rewrite_ok else None
        record["rewritten_user_message"] = usable_rewrite_text(rewrite)
        record["check_status"] = check["parse_status"] if check else "not_checked"
        record["check_verdict"] = {field: check[field] for field in check_fields_of(arm)} if check_ok else None
        record["passes_all_checks"] = check["passes_all_checks"] if check_ok else None
        record["revision_status"] = revision["parse_status"] if revision else "not_attempted"
        record["revision_can_be_rewritten"] = revision["can_be_rewritten"] if revision_ok else None
        record["revision_what_was_changed"] = revision["what_was_changed"] if revision_ok else None
        record["revision_rewritten_user_message"] = usable_rewrite_text(revision)
        record["revision_check_status"] = revision_check["parse_status"] if revision_check else "not_checked"
        record["revision_check_verdict"] = ({field: revision_check[field] for field in check_fields_of(arm)}
                                            if revision_check_ok else None)
        record["revision_passes_all_checks"] = revision_check["passes_all_checks"] if revision_check_ok else None
        if rewrite_ok and rewrite["can_be_rewritten"] == "no":
            outcome = OUTCOME_DECLINED
        elif not rewrite_ok:
            outcome = OUTCOME_REWRITE_FAILED
        elif not check_ok:
            outcome = OUTCOME_CHECK_FAILED
        elif check["passes_all_checks"] is True:
            outcome = OUTCOME_ACCEPTED
        elif revision_check_ok and revision_check["passes_all_checks"] is True:
            outcome = OUTCOME_ACCEPTED_AFTER_REVISION
        else:
            outcome = OUTCOME_REJECTED
        record["outcome"] = outcome
        record["accepted"] = outcome in (OUTCOME_ACCEPTED, OUTCOME_ACCEPTED_AFTER_REVISION)
        record["accepted_attempt"] = {OUTCOME_ACCEPTED: ATTEMPT_FIRST_REWRITE,
                                      OUTCOME_ACCEPTED_AFTER_REVISION: ATTEMPT_REVISION}.get(outcome)
        record["accepted_rewritten_user_message"] = {
            ATTEMPT_FIRST_REWRITE: record["rewritten_user_message"],
            ATTEMPT_REVISION: record["revision_rewritten_user_message"]}.get(record["accepted_attempt"])
        all_records.append(record)
    return all_records


def build_accepted_prompts(all_records: list[dict], batch_number: int, share_number: int) -> list[dict]:
    """The accepted text of each accepted prompt (the revision when that is what passed) under rewritten_user_message, with the
    reason kind, goal and attempt number it was accepted under. all_records is one record per prompt (its last attempt)."""
    return [{**{field: record[field] for field in ("corpus_prompt_id", "unique_prompt_id", "spec_id", "harm_type",
                                                   "taxa_category", "taxa_subcategory", "reason_kind", "different_priority_goal",
                                                   "originally_dealt_reason_kind", "attempt_number", "prompt_arm")},
             "rewritten_user_message": record["accepted_rewritten_user_message"], "accepted_attempt": record["accepted_attempt"],
             "user_lean_flip_share_number": share_number, "user_lean_flip_batch_number": batch_number}
            for record in all_records if record["accepted"]]


def summarise(attempt_records: list[dict], final_records: list[dict], arm: str) -> dict:
    """Counts of the prompts by how each ended (final_records, one per prompt) and of the work done (attempt_records, one per
    prompt attempt; by_reason_kind and by_different_priority_goal count attempts, so keep rates per kind and goal read from them)."""
    def counts_of(records: list[dict]) -> dict:
        return {"prompts": len(records), "declined": sum(r["outcome"] == OUTCOME_DECLINED for r in records),
                "checked": sum(r["check_status"] == "ok" for r in records), "accepted": sum(r["accepted"] for r in records),
                "accepted_by_first_rewrite": sum(r["outcome"] == OUTCOME_ACCEPTED for r in records),
                "rescued_by_revision": sum(r["outcome"] == OUTCOME_ACCEPTED_AFTER_REVISION for r in records)}
    checked = [record for record in attempt_records if record["check_status"] == "ok"]
    revision_checked = [record for record in attempt_records if record["revision_check_status"] == "ok"]
    properties = check_properties_of(arm)
    accepted_prompts = [record for record in final_records if record["accepted"]]
    goals = sorted({r["different_priority_goal"] for r in attempt_records if r["different_priority_goal"]})
    return {"prompts_in_share": len(final_records), "prompt_arm": arm,
            "declined_by_rewriter_count": sum(r["outcome"] == OUTCOME_DECLINED for r in final_records),
            "rewrite_failed_count": sum(r["outcome"] == OUTCOME_REWRITE_FAILED for r in final_records),
            "checked_count": len(checked),
            "check_failed_count": sum(r["outcome"] == OUTCOME_CHECK_FAILED for r in final_records),
            "failed_first_check_count": sum(r["passes_all_checks"] is False for r in attempt_records),
            "revised_count": sum(r["revision_status"] != "not_attempted" for r in attempt_records),
            "revision_declined_count": sum(r["revision_can_be_rewritten"] == "no" for r in attempt_records),
            "revision_failed_count": sum(r["revision_status"] not in ("not_attempted", "ok") for r in attempt_records),
            "revision_checked_count": len(revision_checked),
            "revision_check_failed_count": sum(r["revision_check_status"] not in ("not_checked", "ok") for r in attempt_records),
            "rejected_by_check_count": sum(r["outcome"] == OUTCOME_REJECTED for r in final_records),
            "first_pass_accepted_count": sum(r["outcome"] == OUTCOME_ACCEPTED for r in final_records),
            "rescued_by_revision_count": sum(r["outcome"] == OUTCOME_ACCEPTED_AFTER_REVISION for r in final_records),
            "accepted_count": len(accepted_prompts),
            "attempts_made_count": len(attempt_records),
            "accepted_by_final_reason_kind": {kind: sum(r["reason_kind"] == kind for r in accepted_prompts) for kind in REASON_KINDS},
            "rescued_by_a_different_reason_kind_count": sum(r["attempt_number"] > 1 for r in accepted_prompts),
            "attempts_by_attempt_number": {
                str(number): {"attempts": sum(r["attempt_number"] == number for r in attempt_records),
                              "accepted": sum(r["attempt_number"] == number and r["accepted"] for r in attempt_records)}
                for number in range(1, len(REASON_KINDS) + 1)},
            "by_reason_kind": {kind: counts_of([r for r in attempt_records if r["reason_kind"] == kind]) for kind in REASON_KINDS},
            "by_different_priority_goal": {goal: counts_of([r for r in attempt_records if r["different_priority_goal"] == goal])
                                           for goal in goals},
            "by_harm_type": {harm_type: counts_of([r for r in final_records if r["harm_type"] == harm_type])
                             for harm_type in sorted({r["harm_type"] for r in final_records})},
            "check_field_value_counts": {field: {value: sum(r["check_verdict"][field] == value for r in checked)
                                                 for value in field_schema["enum"]}
                                         for field, field_schema in properties.items() if "enum" in field_schema},
            "revision_check_field_value_counts": {
                field: {value: sum(r["revision_check_verdict"][field] == value for r in revision_checked)
                        for value in field_schema["enum"]}
                for field, field_schema in properties.items() if "enum" in field_schema}}


# ---------------------------------------------------------------- the batch

def run_user_lean_flip_batch(corpus_folder: Path, batch_folder: Path, batch_number: int, started_at: datetime, call_model,
                             parallel_calls: int = PARALLEL_CALLS, now=datetime.now,
                             progress_interval_seconds: float = PROGRESS_REPORT_INTERVAL_SECONDS,
                             prompt_folder: Path = PROMPT_FOLDER) -> dict:
    corpus_folder, batch_folder = extended_length_path(corpus_folder), extended_length_path(batch_folder)
    started = time.time()
    stamp = now().strftime(TIMESTAMP_FORMAT)
    settings = read_settings(corpus_folder)
    share_records, arm = read_verified_share_records(corpus_folder, settings)
    share_number, prompt_file_names = settings["share_number"], prompt_file_names_used(arm)

    def report(message):
        print(f"[{(time.time() - started) / 60:6.1f} min] user lean flip batch {batch_number:03d}: {message}", flush=True)

    if not marker_exists(batch_folder, STEP_SNAPSHOT):
        snapshot_share_and_prompt_files(batch_folder, corpus_folder, settings, stamp, prompt_folder, prompt_file_names,
                                        SNAPSHOT_FOLDER_PREFIX)
        write_marker(batch_folder, STEP_SNAPSHOT, 0)
    else:
        refuse_prompt_files_changed_since_the_batch_started(batch_folder, prompt_folder, prompt_file_names, SNAPSHOT_FOLDER_PREFIX)
    hashes = snapshot_hashes(batch_folder, SNAPSHOT_FOLDER_PREFIX)
    paths = {prefix: find_existing_file(batch_folder, prefix) or batch_folder / f"{prefix}_{stamp}.jsonl"
             for prefix in (*STEP_PREFIXES.values(), USAGE_PREFIX)}

    def records_of(prefix):
        return read_jsonl(paths[prefix]) if paths[prefix].exists() else []

    def records_of_attempt(prefix, attempt_number):
        return [record for record in records_of(prefix) if record.get("attempt_number", 1) == attempt_number]

    goals = read_different_priority_goals(prompt_folder)
    attempt_records, final_records, pending_share_records = [], [], share_records
    for attempt_number in range(1, len(REASON_KINDS) + 1):
        attempt_share_records = [share_record_for_attempt(record, attempt_number, goals) for record in pending_share_records]
        step_names = {step: step_name_for_attempt(step, attempt_number) for step in STEP_NAMES}

        def work_for(step_name, attempt_share_records=attempt_share_records, attempt_number=attempt_number):
            def of_attempt(prefix):
                return records_of_attempt(prefix, attempt_number)
            if step_name == step_names[STEP_REWRITE]:
                return rewrites_to_make(attempt_share_records, of_attempt(REWRITES_PREFIX))
            if step_name == step_names[STEP_CHECK]:
                return rewrites_to_check(attempt_share_records, of_attempt(REWRITES_PREFIX), of_attempt(CHECKS_PREFIX))
            if step_name == step_names[STEP_REVISE]:
                return rewrites_to_revise(attempt_share_records, of_attempt(REWRITES_PREFIX), of_attempt(CHECKS_PREFIX),
                                          of_attempt(REVISIONS_PREFIX))
            return rewrites_to_check(attempt_share_records, of_attempt(REVISIONS_PREFIX), of_attempt(REVISION_CHECKS_PREFIX))

        processors = {STEP_REWRITE: lambda item: rewrite_one_prompt(item, arm, call_model, now),
                      STEP_CHECK: lambda item: check_one_rewrite(item, arm, call_model, now),
                      STEP_REVISE: lambda item: revise_one_rewrite(item, arm, call_model, now),
                      STEP_CHECK_REVISION: lambda item: check_one_rewrite(item, arm, call_model, now, STEP_CHECK_REVISION)}
        if attempt_number > 1:
            report(f"attempt {attempt_number}: {len(attempt_share_records)} prompts, each with a reason kind it has not tried")
        run_steps_until_every_prompt_is_settled(
            batch_folder, tuple(step_names.values()), work_for,
            {step_names[step]: processor for step, processor in processors.items()},
            {step_names[step]: paths[prefix] for step, prefix in STEP_PREFIXES.items()}, paths[USAGE_PREFIX],
            len(attempt_share_records), parallel_calls, progress_interval_seconds, report)
        records_per_step = {step: records_of_attempt(prefix, attempt_number) for step, prefix in STEP_PREFIXES.items()}
        failed_steps = steps_where_every_call_failed(records_per_step)
        if failed_steps:
            raise EveryCallOfAStepFailed(f"every call of the {' and '.join(failed_steps)} step failed (for example the API rejected the reply "
                               f"schema), so nothing was written to the accepted file, summary or ledger. The reasons are in the "
                               f"records files and the usage log of {batch_folder}; send them to the person who sent you the kit.")
        this_attempt = build_all_records(attempt_share_records, records_per_step, arm)
        attempt_records += this_attempt
        final_records = [record for record in final_records if record["unique_prompt_id"] not in
                         {r["unique_prompt_id"] for r in this_attempt}] + this_attempt
        pending_ids = {r["unique_prompt_id"] for r in this_attempt if r["outcome"] in (OUTCOME_DECLINED, OUTCOME_REJECTED)}
        pending_share_records = [record for record in pending_share_records if record["unique_prompt_id"] in pending_ids]
        if not pending_share_records:
            break
    prompt_order = {record["unique_prompt_id"]: position for position, record in enumerate(share_records)}
    final_records.sort(key=lambda record: prompt_order[record["unique_prompt_id"]])
    report("building the all records file, the accepted prompts and the summary")
    accepted_prompts = build_accepted_prompts(final_records, batch_number, share_number)
    final_stamp = now().strftime(TIMESTAMP_FORMAT)
    all_records_path = batch_folder / f"user_lean_flip_all_records_batch_{batch_number:03d}_{final_stamp}.jsonl"
    all_records_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in attempt_records), encoding="utf-8")
    accepted_path = batch_folder / f"accepted_user_lean_flip_prompts_batch_{batch_number:03d}_{final_stamp}.jsonl"
    accepted_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in accepted_prompts), encoding="utf-8")
    counts = summarise(attempt_records, final_records, arm)
    costs = usage_totals(paths[USAGE_PREFIX])
    models = {"rewrite_model": REWRITE_MODEL, "check_model": CHECK_MODEL}
    summary = {"batch_number": batch_number, "share_number": share_number, **models, **counts,
               "prompt_file_sha256": hashes["prompt_files"], "share_file_sha256": hashes["share_file"],
               "started_at": started_at.isoformat(timespec="seconds"), "ended_at": now().isoformat(timespec="seconds"),
               "costs": costs, "all_records_file": all_records_path.name, "accepted_file": accepted_path.name}
    (batch_folder / f"user_lean_flip_batch_summary_{final_stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ledger_entry = {"batch_number": batch_number, "batch_folder": ledger_batch_folder_text(batch_folder, corpus_folder),
                    "batch_kind": "user_lean_flip", "prompt_arm": arm, "share_number": share_number,
                    "prompts_in_share": counts["prompts_in_share"], "declined_by_rewriter_count": counts["declined_by_rewriter_count"],
                    "checked_count": counts["checked_count"], "revised_count": counts["revised_count"],
                    "first_pass_accepted_count": counts["first_pass_accepted_count"],
                    "rescued_by_revision_count": counts["rescued_by_revision_count"], "accepted_count": counts["accepted_count"],
                    "accepted_by_final_reason_kind": counts["accepted_by_final_reason_kind"],
                    "rescued_by_a_different_reason_kind_count": counts["rescued_by_a_different_reason_kind_count"],
                    "accepted_file": accepted_path.name, "all_records_file": all_records_path.name, **models,
                    "prompt_file_sha256": hashes["prompt_files"], "share_file_sha256": hashes["share_file"],
                    "started_at": summary["started_at"], "ended_at": summary["ended_at"], "costs": costs}
    append_ledger_line(corpus_folder / LEDGER_FILE_NAME, ledger_entry)
    report(f"ledger updated: {counts['accepted_count']} accepted of {counts['prompts_in_share']} prompts "
           f"({counts['first_pass_accepted_count']} on the first rewrite, {counts['rescued_by_revision_count']} rescued by "
           f"revision, {counts['rescued_by_a_different_reason_kind_count']} rescued by a different reason kind, {counts['declined_by_rewriter_count']} declined, {counts['checked_count']} checked, CLI-reported cost "
           f"${costs['cli_reported_cost_usd']:.4f})")
    return ledger_entry


def placeholder_failed_check(arm: str) -> dict:
    """A check verdict with practical_facts_unchanged failed, for the dry run's rendered revision prompt."""
    verdict = dict(required_values_of(arm))
    return {**verdict, "practical_facts_unchanged": "no", "shared_screen_short_reason": "none", "short_reason": "(the judge's short reason would appear here)"}


def print_plan(corpus_folder: Path, share_records: list[dict], arm: str, settings: dict) -> None:
    print(f"Corpus folder: {corpus_folder}\nShare number: {settings['share_number']}, {len(share_records)} prompts, prompt arm "
          f"{arm} (share file sha256 {settings['share_file_sha256']})")
    print("Reason kinds dealt: " + ", ".join(f"{kind} {sum(r['reason_kind'] == kind for r in share_records)}" for kind in REASON_KINDS))
    print(f"Models: rewrite {REWRITE_MODEL}, check {CHECK_MODEL}")
    for name, digest in prompt_file_hashes(PROMPT_FOLDER, prompt_file_names_used(arm)).items():
        print(f"prompt file {name}: sha256 {digest}")
    print(f"Calls: at most {len(share_records)} rewrites, then one check for each rewrite the model did not decline, then one "
          f"revision and one more check for each rewrite that failed its check (about 2 per prompt, more for failures); retries add "
          f"more (up to {MAXIMUM_ATTEMPTS_PER_CALL} attempts per call). A prompt that ends declined or rejected is run again from the "
          f"rewrite step with a reason kind it has not tried, up to {len(REASON_KINDS)} attempts in all.")
    if share_records:
        first = share_records[0]
        print(f"\n----- rendered rewrite prompt for {first['unique_prompt_id']} (reason kind {first['reason_kind']}) -----")
        print(render_rewrite_prompt(first, arm))
        print(f"\n----- rendered revision prompt for {first['unique_prompt_id']} with a placeholder rewrite and a placeholder "
              f"failed check -----")
        print(render_revision_prompt(first, arm, PLACEHOLDER_REWRITE_FOR_DRY_RUN, placeholder_failed_check(arm)))
        print(f"\n----- rendered check prompt for {first['unique_prompt_id']} with a placeholder rewrite -----")
        print(render_check_prompt(first["user_message"], PLACEHOLDER_REWRITE_FOR_DRY_RUN, arm))
        print(f"\n----- rendered shared screen prompt (a second call of every check) with a placeholder rewrite -----")
        print(render_screen_prompt(PLACEHOLDER_REWRITE_FOR_DRY_RUN))
    print(f"\nAfter success, one line is appended to {Path(corpus_folder) / LEDGER_FILE_NAME}. Nothing is created or run.")


def main(argv=None, call_model=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-folder", type=Path, required=True)
    parser.add_argument("--resume-batch-folder", type=Path, default=None, help="finish a batch folder that has no ledger entry yet")
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and one rendered prompt; create nothing")
    arguments = parser.parse_args(argv)
    corpus_folder = extended_length_path(arguments.corpus_folder.resolve())
    try:
        settings = read_settings(corpus_folder)
        share_records, arm = read_verified_share_records(corpus_folder, settings)
        if arguments.dry_run:
            print_plan(corpus_folder, share_records, arm, settings)
            return 0
        if arguments.resume_batch_folder is not None:
            batch_folder = extended_length_path(arguments.resume_batch_folder.resolve())
            match = BATCH_FOLDER_PATTERN.match(batch_folder.name)
            if not match or not batch_folder.is_dir():
                parser.error(f"{batch_folder} is not a batch folder named batch_<NNN>_user_lean_flip_share_<NNN>_<stamp>")
            batch_number = int(match.group(1))
            if batch_number in batch_numbers_in_ledger(corpus_folder / LEDGER_FILE_NAME):
                parser.error(f"batch {batch_number} is already in the ledger; nothing to resume")
        else:
            incomplete = incomplete_batch_folders(corpus_folder, BATCH_FOLDER_PATTERN)
            if incomplete:
                print("Refusing to start a new batch: these batch folders have no ledger entry:", file=sys.stderr)
                for folder in incomplete:
                    print(f"  {remove_extended_length_prefixes_from_text(str(folder))}", file=sys.stderr)
                print("Resume one with --resume-batch-folder <folder>.", file=sys.stderr)
                return 1
            if read_ledger(corpus_folder / LEDGER_FILE_NAME):
                print("This share's batch is already in the ledger; nothing to run.", file=sys.stderr)
                return 1
            batch_number = next_batch_number(corpus_folder, BATCH_FOLDER_PATTERN)
            batch_folder = corpus_folder / "batches" / batch_folder_name(batch_number, settings["share_number"],
                                                                         datetime.now().strftime(TIMESTAMP_FORMAT))
        if call_model is None:
            from claude_code_cli_structured_reply_without_paid_api_fallback import call_claude_code_cli_for_structured_reply

            def call_model(model, prompt, json_schema):
                return call_claude_code_cli_for_structured_reply(model, prompt, json_schema)
        batch_folder.mkdir(parents=True, exist_ok=True)
        run_user_lean_flip_batch(corpus_folder, batch_folder, batch_number, datetime.now(), call_model,
                                 parallel_calls=arguments.parallel_calls)
    except BatchStoppedForUsageLimit as failure:
        print(remove_extended_length_prefixes_from_text(f"STOPPED FOR USAGE LIMIT: {failure}"), file=sys.stderr)
        return USAGE_LIMIT_EXIT_CODE
    except EveryCallOfAStepFailed as failure:
        print(remove_extended_length_prefixes_from_text(f"FAILED: {failure}"), file=sys.stderr)
        return EVERY_CALL_FAILED_EXIT_CODE
    except BatchFailure as failure:
        print(remove_extended_length_prefixes_from_text(f"FAILED: {failure}"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
