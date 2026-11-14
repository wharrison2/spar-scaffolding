"""Shared state logic for making the content-controlled positive-control rewrites with Claude Code subagents (Opus 5.5) instead of
the paid API. Used by render_positive_control_rewrite_prompts_for_subagents.py and ingest_subagent_positive_control_rewrites.py.
Nothing here calls a model, an API or a pod, and nothing reads .env.

Principle: the paid script's own functions do the work. A rewrite's whole decision path (the instruction template, the target and
lowest and highest changed amounts, the minimum-change retry note, the choice of the largest-change attempt, the flag
`below_minimum_changed_units`, every field of the record) is run by facet_rewrites.make_one_rewrite, with a request function that
REPLAYS the subagent replies recorded so far instead of calling an API:
  * when a reply is on file for the attempt, the replay returns it as a response and the paid function validates and counts it;
  * when the paid function asks for an attempt that has no reply yet, the replay captures the prompt it was given (this is exactly
    the prompt the paid script would have sent: the plain instruction for attempt 1, the instruction with the retry note and the
    previous changed amount for a later attempt) and stops. That captured prompt is the next prompt file to write.

Files in the run folder (every name ends with the date and time of the session that wrote it; patterns are read across sessions):
  run_configuration_<timestamp>.json
  <facet key>_positive_control_rewrites_<timestamp>.jsonl   accepted rewrites, the paid script's records
  rewrite_failures_<timestamp>.jsonl                        replies that failed validation, with the reason and the reply text
  subagent_reply_attempts_<timestamp>.jsonl                 every reply that passed the format checks, in attempt order
  subagent_prompts/<rewrite id>.txt                         the prompt of the rewrite's current attempt
  subagent_outputs/<rewrite id>.txt                         where the subagent writes its reply (moved after ingestion)
  subagent_outputs_ingested/<rewrite id>__attempt_<n>__<timestamp>.txt   replies already read, kept unchanged
"""

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for extra_path in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import make_preachiness_positive_control_rewrites as preachiness_rewrites  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_arms as arms_module  # noqa: E402
from persona_vector_projection import make_content_controlled_facet_positive_control_rewrites as paid_script  # noqa: E402
from persona_vector_projection import make_facet_positive_control_rewrites as facet_rewrites  # noqa: E402
from shared.resumable_jsonl_records import read_all_records, read_jsonl  # noqa: E402

REWRITE_MADE_BY = "Claude Code subagents running Opus 5.5"
SUBAGENT_MODEL_ID = "claude-opus-5-5"
RUN_NAME_MODEL_PART = "claude_code_opus_5_5_subagents"
PROMPTS_FOLDER_NAME = "subagent_prompts"
OUTPUTS_FOLDER_NAME = "subagent_outputs"
INGESTED_OUTPUTS_FOLDER_NAME = "subagent_outputs_ingested"
REPLY_ATTEMPTS_FILE_PATTERN = "subagent_reply_attempts_*.jsonl"
FAILURES_FILE_PATTERN = "rewrite_failures_*.jsonl"
SETTINGS_KEY = "subagent_rewrite_settings"

FAILURE_EMPTY_REPLY = "empty_output"
FAILURE_PREAMBLE_OR_WRAPPER = "preamble_or_wrapper_in_reply"

# Openings and closings that mean the reply is not only the document. A reply that has one FAILS; nothing is ever stripped.
# A phrase is ignored when the source document itself starts (or ends) with it.
PREAMBLE_OPENING_PATTERN = re.compile(
    r"^\W*(here is|here's|here are|here you go|sure\b|certainly\b|of course\b|okay\b|ok\b|absolutely\b|i've\b|i have\b|i'll\b|"
    r"below is|below you|the rewritten|rewritten document|rewritten version|revised document|revised version|rewrite\s*:|"
    r"document\s*:|as requested|i rewrote|i changed|i made)", re.IGNORECASE)
CLOSING_NOTE_PATTERN = re.compile(
    r"^\W*(note\s*:|notes\s*:|i changed|i've changed|i made|i rewrote|i've rewritten|let me know|this rewrite|the changes|"
    r"summary of changes|changes made)", re.IGNORECASE)
CODE_FENCE = "```"


class NeedsAnotherReply(Exception):
    """Raised inside the replay when the paid function asks for an attempt that has no reply yet."""


def sha256_of_text_value(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Folder layout
# ----------------------------------------------------------------------------

def run_name_for(definition):
    return f"{definition.facet_key}_positive_control_rewrites_{RUN_NAME_MODEL_PART}"


def prompt_path_for(run_folder, rewrite_id):
    return Path(run_folder) / PROMPTS_FOLDER_NAME / f"{rewrite_id}.txt"


def output_path_for(run_folder, rewrite_id):
    return Path(run_folder) / OUTPUTS_FOLDER_NAME / f"{rewrite_id}.txt"


def find_existing_run_folder(runs_directory, definition):
    """The latest folder already made for this facet's subagent rewrites, or None."""
    matches = sorted(Path(runs_directory).glob(f"{run_name_for(definition)}_????-??-??_??-??"))
    return matches[-1] if matches else None


# ----------------------------------------------------------------------------
# Reading the reply (the paid script's checks, plus the preamble check)
# ----------------------------------------------------------------------------

def make_reply_response(text):
    """The object the paid script's `read_rewrite_text` and usage summary read, for a reply that was written by a subagent. No token
    or cost log exists for a subagent, so every usage count is zero and the model is the subagent model."""
    usage = SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0, iterations=None)
    return SimpleNamespace(model=SUBAGENT_MODEL_ID, stop_reason="end_turn", stop_details=None, usage=usage,
                           content=[SimpleNamespace(type="text", text=text)])


def first_and_last_nonblank_lines(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[0], lines[-1]) if lines else ("", "")


def find_preamble_or_wrapper(reply_text, source_text):
    """A description of what makes the reply more than the document, or None. Heuristic and deliberately strict: a false alarm costs one
    more attempt, a missed preamble puts text that is not the document into the scored corpus. Phrases the source itself starts
    or ends with are not counted."""
    first_line, last_line = first_and_last_nonblank_lines(reply_text)
    source_first_line, source_last_line = first_and_last_nonblank_lines(source_text)
    if reply_text.startswith(CODE_FENCE) and not source_text.startswith(CODE_FENCE):
        return "the reply starts with a code fence"
    if reply_text.endswith(CODE_FENCE) and not source_text.rstrip().endswith(CODE_FENCE):
        return "the reply ends with a code fence"
    if PREAMBLE_OPENING_PATTERN.match(first_line) and not PREAMBLE_OPENING_PATTERN.match(source_first_line):
        return f"the first line reads as a preamble: {first_line[:120]!r}"
    if CLOSING_NOTE_PATTERN.match(last_line) and not CLOSING_NOTE_PATTERN.match(source_last_line):
        return f"the last line reads as a closing note: {last_line[:120]!r}"
    return None


def check_reply_text(raw_reply_text, source_text):
    """(text, failure_kind, description). `text` is the reply exactly as the paid script reads it (surrounding whitespace stripped, as
    `read_rewrite_text` does, and nothing else). failure_kind is empty_output or preamble_or_wrapper_in_reply, else None."""
    text, failure_kind = preachiness_rewrites.read_rewrite_text(make_reply_response(raw_reply_text))
    if failure_kind:
        return None, failure_kind, "the reply is empty"
    wrapper = find_preamble_or_wrapper(text, source_text)
    if wrapper:
        return None, FAILURE_PREAMBLE_OR_WRAPPER, wrapper
    return text, None, None


# ----------------------------------------------------------------------------
# Tasks (the paid script's own builders)
# ----------------------------------------------------------------------------

def read_accepted_records(run_folder, definition):
    """The latest accepted record per rewrite id."""
    return list({record["document_id"]: record for record in read_all_records(
        run_folder, arms_module.rewrites_file_pattern(definition))}.values())


def build_independent_tasks(definition, corpus_by_id, source_ids, target_changed_units):
    """Wave 1: every arm that starts from the original document."""
    return paid_script.build_independent_arm_tasks(definition, corpus_by_id, source_ids, target_changed_units)


def build_chained_tasks(definition, source_ids, accepted_records, target_changed_units):
    """Wave 2: (tasks, skipped). The document of each task is the ACCEPTED trait_only_away rewrite, never the source."""
    return paid_script.build_chained_arm_tasks(definition, source_ids, accepted_records, target_changed_units)


def rewrite_id_of_task(definition, task):
    return arms_module.rewrite_document_id_for(definition, task["arm"], task["original_source_id"])


def expected_rewrite_ids(definition, source_ids):
    return [arms_module.rewrite_document_id_for(definition, arm, source) for source in sorted(source_ids)
            for arm in arms_module.arm_names_for(definition)]


# ----------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------

def read_reply_attempt_texts(run_folder, rewrite_id):
    """The texts of the rewrite's replies that passed the format checks, in attempt order."""
    records = [record for record in read_all_records(run_folder, REPLY_ATTEMPTS_FILE_PATTERN) if record["rewrite_id"] == rewrite_id]
    return [record["reply_text"] for record in sorted(records, key=lambda record: record["attempt_number"])]


def replay_rewrite(task, definition, templates, settings, reply_texts, tokenize_function=None):
    """Runs the paid script's make_one_rewrite over the replies recorded so far.

    Returns ("accepted", finished record) when the rewrite is done (a rewrite still below the minimum after the last allowed attempt is
    done and flagged, exactly as the paid script keeps it), or ("pending", prompt text of the next attempt, attempt number)."""
    captured = {}
    remaining = list(reply_texts)

    def request_function(_client, prompt_text):
        if not remaining:
            captured["prompt_text"] = prompt_text
            raise NeedsAnotherReply()
        return make_reply_response(remaining.pop(0))

    attempts_requested = []

    def counting_request_function(client, prompt_text):
        attempts_requested.append(prompt_text)
        return request_function(client, prompt_text)

    record, failure, _call_summaries = facet_rewrites.make_one_rewrite(
        None, templates, task["document"], definition.facet_key, task["arm"], settings["minimum_changed_units"],
        not settings["no_regeneration"], settings["maximum_regenerations"], tokenize_function,
        target_changed_units=task["target_changed_units"], size_match_tolerance=settings["size_match_tolerance"],
        request_function=counting_request_function)
    if record is not None:
        record = paid_script.finish_record(record, task, definition)
        record["rewrite_model_requested"] = SUBAGENT_MODEL_ID
        record["rewrite_model_used"] = SUBAGENT_MODEL_ID
        return "accepted", record
    if "prompt_text" in captured:
        return "pending", captured["prompt_text"], len(attempts_requested)
    raise RuntimeError(f"Unexpected failure replaying {rewrite_id_of_task(definition, task)}: {failure}")


def last_failure_by_rewrite_id(run_folder):
    """{rewrite id: its latest failure record} over the failures file(s)."""
    latest = {}
    for record in read_all_records(run_folder, FAILURES_FILE_PATTERN):
        if record.get("rewrite_id"):
            latest[record["rewrite_id"]] = record
    return latest


def settings_from_run_configuration(run_folder):
    """The rewrite settings recorded by the first session of the run (the later sessions must not change them)."""
    for path in sorted(Path(run_folder).glob("run_configuration_*.json")):
        configuration = json.loads(path.read_text(encoding="utf-8"))
        if SETTINGS_KEY in configuration:
            return configuration[SETTINGS_KEY], configuration
    raise RuntimeError(f"No run_configuration with {SETTINGS_KEY} in {run_folder}; run the prompt rendering script first.")


def load_run_inputs(run_folder, definition, corpus_path_override=None):
    """(settings, configuration, corpus_by_id, source_ids) as recorded by the run's first session."""
    settings, configuration = settings_from_run_configuration(run_folder)
    corpus_path = corpus_path_override or configuration["corpus_path"]
    corpus_by_id = {document["doc_id"]: document for document in read_jsonl(corpus_path)}
    return settings, configuration, corpus_by_id, configuration["original_source_document_ids"]
