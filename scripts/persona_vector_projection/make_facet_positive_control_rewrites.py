"""Positive-control rewrites for the persona vector projection experiment (PAID when run without --dry-run).

For each of three facets (cares about welfare, calibrated consideration, consistent), 30 corpus documents are each
rewritten with Claude Opus 5 (same model and settings as the preachiness rewrites) three ways:
  toward_facet     the AI shows the trait more
  away_from_facet  the AI shows the trait less (the facet's negative pole)
  facet_neutral    about as many words changed as in the two directions above, trait left alone
The non-intrusive facet reuses the existing preachier (away) and plainer (toward) rewrites and only gets the
same-size facet-neutral rewrites, for the documents those rewrites came from.

Order of work: toward and away first; the neutral rewrite of a document starts from the mean number of changed units
of its own toward and away rewrites (so neutral rewrites are "same size" per document).

Changed units are counted by a diff against the source (changed_token_counting_by_diff.py). A rewrite that changed
fewer than --minimum-changed-units is flagged `below_minimum_changed_units`, and regenerated (up to
--maximum-regenerations times, with a note appended to the instruction) unless --no-regeneration is given. Flagged
rewrites stay in the records but are left out of the scoring-pod export unless --include-flagged-in-export.

Files in the run folder (every name ends with the date and time of the session that wrote it):
  run_configuration_<timestamp>.json                 settings, prompt file hashes, selected document ids
  facet_positive_control_rewrites_<timestamp>.jsonl  one record per finished rewrite (read back on resume)
  rewrite_failures_<timestamp>.jsonl                 refusals, max_tokens, empty output, transport errors (retried on resume)
  per_call_usage_and_cost_<timestamp>.jsonl, usage_and_cost_log_<timestamp>.json, progress_log_<timestamp>.txt
  scoring_pod_extra_documents_<timestamp>.jsonl      export, same shape as --extra-documents-jsonl of build_scoring_inputs.py,
                                                     with source_document_id, facet and rewrite_direction
Dry runs write to data/persona_vector_projection_runs/small_sample_checks/ and make no API call.

The API key is read ONLY from the project .env (ANTHROPIC_API_KEY), never from the OS environment.

Examples:
    python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py --dry-run
    python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py          (PAID)
    python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py --resume-folder <folder>   (PAID)
    python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py --export-only --resume-folder <folder>
"""

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for extra_path in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import make_preachiness_positive_control_rewrites as preachiness_rewrites  # noqa: E402  (reused: model, request, key, length helpers)
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection.changed_token_counting_by_diff import (  # noqa: E402
    LANGUAGES_WITHOUT_SPACES_BETWEEN_WORDS, changed_units_for_size_matching, count_changed_units,
    is_below_minimum_changed_units)
from persona_vector_projection.select_documents_with_room_to_move_for_facet import (  # noqa: E402
    FACET_SELECTION_RULES, select_documents_for_facet)
from shared.anthropic_usage_and_cost_tracking import (AnthropicUsageAndCostTracker,  # noqa: E402
                                                      PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL,
                                                      compute_anthropic_cost_in_dollars,
                                                      extract_attempts_from_response, summarize_attempts)
from shared.resumable_jsonl_records import (AppendOnlyJsonlWriter, read_all_records, read_jsonl,  # noqa: E402
                                            write_jsonl)
from shared.run_folder_management import (check_resume_is_compatible, choose_run_folder, sha256_of_file,  # noqa: E402
                                          timestamp_now, write_run_configuration)
from shared.timed_progress_reporter import TimedProgressReporter  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompts"
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sdf_corpus.jsonl"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_EXISTING_PREACHINESS_REWRITES_PATH = (
    PROJECT_ROOT / "data" / "log_probability_persona_check_runs" / "positive_control_rewrites"
    / "preachiness_positive_control_rewrites_opus_5_2026-10-22_03-00" / "rewritten_documents.jsonl")
FULL_RUNS_DIRECTORY = defs.RUNS_ROOT
DRY_RUNS_DIRECTORY = defs.SMALL_SAMPLE_CHECKS_ROOT

DIRECTION_TOWARD = "toward_facet"
DIRECTION_AWAY = "away_from_facet"
DIRECTION_NEUTRAL = "facet_neutral"
SIZE_REFERENCE_DIRECTIONS = [DIRECTION_TOWARD, DIRECTION_AWAY]
FACETS_WITH_NEW_TOWARD_AND_AWAY_REWRITES = [defs.FACET_CARES_ABOUT_WELFARE, defs.FACET_CALIBRATED_CONSIDERATION,
                                            defs.FACET_CONSISTENT]
FACETS_WITH_NEUTRAL_REWRITES = FACETS_WITH_NEW_TOWARD_AND_AWAY_REWRITES + [defs.FACET_NON_INTRUSIVE]
# The preachiness rewrites, mapped onto the non-intrusive facet: plainer is toward the trait, preachier away.
PREACHINESS_DIRECTION_TO_NON_INTRUSIVE_DIRECTION = {"plainer": DIRECTION_TOWARD, "preachier": DIRECTION_AWAY}

REWRITES_FILE_PATTERN = "facet_positive_control_rewrites_*.jsonl"
RETRY_NOTE_FILE_NAME = "positive_control_rewrite_retry_note_below_minimum_changes__draft.txt"

DEFAULT_DOCUMENTS_PER_FACET = 30
DEFAULT_MINIMUM_CHANGED_UNITS = 60           # words (spaced languages); scaled for languages counted in characters
CHARACTERS_PER_WORD_FOR_UNSPACED_LANGUAGES = 1.5
DEFAULT_MAXIMUM_REGENERATIONS = 2
DEFAULT_SIZE_MATCH_TOLERANCE = 0.5           # neutral rewrite flagged if its size is off the target by more than this
NEUTRAL_RANGE_FRACTION = 0.25                # the prompt gives a range of +/- this fraction around the target
MAXIMUM_LENGTH_RATIO_DEVIATION = 0.10

# Dry-run assumptions (expected) and the ceiling
ASSUMED_FRACTION_OF_DOCUMENT_CHANGED_FOR_NEUTRAL_TARGET = 0.15
CEILING_THINKING_TOKENS_PER_CALL = 24000
CEILING_OUTPUT_LENGTH_MULTIPLIER = 1.3       # output length relative to the source, at most
EXPECTED_REGENERATION_RATE = 0.2
# Measured on the preachiness rewrites (claude-opus-5, 20 calls, $2.27): the dry-run assumptions above are cautious
MEASURED_DOLLARS_PER_REWRITE_CALL_IN_PREACHINESS_RUN = 2.27 / 20
DEFAULT_COST_CAP_MULTIPLE_OF_CAUTIOUS_EXPECTED_COST = 1.5


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

def instruction_file_name(facet, direction):
    return f"positive_control_rewrite_instruction__{facet}__{direction}__draft.txt"


def all_instruction_file_names():
    names = [instruction_file_name(facet, direction) for facet in FACETS_WITH_NEW_TOWARD_AND_AWAY_REWRITES
             for direction in (DIRECTION_TOWARD, DIRECTION_AWAY)]
    names += [instruction_file_name(facet, DIRECTION_NEUTRAL) for facet in FACETS_WITH_NEUTRAL_REWRITES]
    return names + [RETRY_NOTE_FILE_NAME]


def load_instruction_templates(prompt_directory=PROMPT_DIRECTORY):
    return {name: (Path(prompt_directory) / name).read_text(encoding="utf-8") for name in all_instruction_file_names()}


def describe_amount(unit_count, language):
    """'60 words', or '90 characters' for languages written without spaces between words."""
    if language in LANGUAGES_WITHOUT_SPACES_BETWEEN_WORDS:
        return f"{int(round(unit_count))} characters"
    return f"{int(round(unit_count))} words"


def minimum_units_for_language(minimum_changed_units, language):
    if language in LANGUAGES_WITHOUT_SPACES_BETWEEN_WORDS:
        return int(round(minimum_changed_units * CHARACTERS_PER_WORD_FOR_UNSPACED_LANGUAGES))
    return minimum_changed_units


def render_instruction(template_text, document, minimum_changed_units, target_changed_units=None,
                       previous_changed_units=None, retry_note_template=None):
    """Fills the slots one at a time so braces inside the document text are never interpreted as slots; the
    document text goes in last."""
    language = document["language"]
    minimum = minimum_units_for_language(minimum_changed_units, language)
    text = template_text.replace("{minimum_changed_amount}", describe_amount(minimum, language))
    if target_changed_units is not None:
        text = (text.replace("{target_changed_amount}", describe_amount(target_changed_units, language))
                .replace("{lowest_changed_amount}",
                         describe_amount(target_changed_units * (1 - NEUTRAL_RANGE_FRACTION), language))
                .replace("{highest_changed_amount}",
                         describe_amount(target_changed_units * (1 + NEUTRAL_RANGE_FRACTION), language)))
    if retry_note_template is not None and previous_changed_units is not None:
        note = (retry_note_template.replace("{previous_changed_amount}",
                                            describe_amount(previous_changed_units, language))
                .replace("{minimum_changed_amount}", describe_amount(minimum, language)))
        # the retry note goes before the document so that the document stays last
        marker = "Document:\n{document_text}"
        head, _, tail = text.rpartition(marker)
        text = head.rstrip("\n") + "\n" + note.rstrip("\n") + "\n\n" + marker + tail
    return text.replace("{document_text}", document["content"])


# ----------------------------------------------------------------------------
# One rewrite (with regeneration below the minimum)
# ----------------------------------------------------------------------------

def make_tokenize_function(tokenizer_name):
    """Optional exact tokens: a Hugging Face tokenizer from the local cache only (no download)."""
    if not tokenizer_name:
        return None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def make_one_rewrite(client, templates, document, facet, direction, minimum_changed_units, regenerate,
                     maximum_regenerations, tokenize_function=None, target_changed_units=None,
                     size_match_tolerance=DEFAULT_SIZE_MATCH_TOLERANCE, request_function=None,
                     cost_is_over_cap_function=lambda: False):
    """Returns (record, failure_record, call_summaries). Exactly one of record / failure_record is not None.

    Attempt 1 uses the plain instruction; each regeneration repeats it with a note naming how little changed.
    Only a rewrite that stays below the minimum after every allowed regeneration is flagged."""
    request_function = request_function or preachiness_rewrites.request_rewrite
    template_text = templates[instruction_file_name(facet, direction)]
    retry_note_template = templates[RETRY_NOTE_FILE_NAME]
    language = document["language"]
    call_summaries, size_per_attempt = [], []
    previous_changed_units = None
    maximum_attempts = 1 + (maximum_regenerations if regenerate else 0)
    best_text, best_counts = None, None
    for attempt_number in range(1, maximum_attempts + 1):
        if attempt_number > 1 and cost_is_over_cap_function():
            break
        prompt_text = render_instruction(
            template_text, document, minimum_changed_units, target_changed_units,
            previous_changed_units=previous_changed_units if attempt_number > 1 else None,
            retry_note_template=retry_note_template if attempt_number > 1 else None)
        started = time.time()
        try:
            response = request_function(client, prompt_text)
        except Exception as error:  # noqa: BLE001 - transport failure after the SDK's own retries
            return None, {"source_document_id": document["doc_id"], "facet": facet, "direction": direction,
                          "failure_kind": "transport_failure", "description": repr(error),
                          "attempt_number": attempt_number}, call_summaries
        call_summary = summarize_attempts(extract_attempts_from_response(response))
        call_summary.update({"latency_seconds": round(time.time() - started, 2), "stop_reason": response.stop_reason,
                             "model_used": response.model, "attempt_number": attempt_number})
        call_summaries.append(call_summary)
        text, failure_kind = preachiness_rewrites.read_rewrite_text(response)
        if failure_kind:
            category = getattr(getattr(response, "stop_details", None), "category", None)
            return None, {"source_document_id": document["doc_id"], "facet": facet, "direction": direction,
                          "failure_kind": failure_kind, "stop_reason": response.stop_reason,
                          "refusal_category": category, "model_used": response.model,
                          "attempt_number": attempt_number}, call_summaries
        counts = count_changed_units(document["content"], text, language, tokenize_function)
        size_per_attempt.append(changed_units_for_size_matching(counts))
        previous_changed_units = size_per_attempt[-1]
        if best_counts is None or previous_changed_units > changed_units_for_size_matching(best_counts):
            best_text, best_counts = text, counts
        if not is_below_minimum_changed_units(counts, minimum_units_for_language(minimum_changed_units, language)):
            best_text, best_counts = text, counts
            break
    language_minimum = minimum_units_for_language(minimum_changed_units, language)
    source_length, length_unit = preachiness_rewrites.length_in_unit(document["content"], language)
    rewrite_length, _ = preachiness_rewrites.length_in_unit(best_text, language)
    length_ratio = rewrite_length / source_length if source_length else None
    changed_size = changed_units_for_size_matching(best_counts)
    record = {
        "document_id": f"{document['doc_id']}__{facet}__{direction}",
        "source_document_id": document["doc_id"], "facet": facet, "rewrite_direction": direction,
        "content": best_text, "language": language, "variables": document["variables"],
        "rewrite_model_requested": preachiness_rewrites.REWRITE_MODEL_REQUESTED,
        "rewrite_model_used": call_summaries[-1]["model_used"],
        "reused_from_preachiness_rewrites": False,
        **best_counts, "changed_units_for_size_matching": changed_size,
        "minimum_changed_units_applied": language_minimum,
        "below_minimum_changed_units": is_below_minimum_changed_units(best_counts, language_minimum),
        "number_of_attempts": len(call_summaries), "changed_units_per_attempt": size_per_attempt,
        "length_unit": length_unit, "source_length": source_length, "rewrite_length": rewrite_length,
        "length_ratio": length_ratio,
        "length_within_ten_percent": (length_ratio is not None
                                      and abs(length_ratio - 1) <= MAXIMUM_LENGTH_RATIO_DEVIATION),
        "target_changed_units_for_neutral": target_changed_units,
        "size_matched_to_target": (None if target_changed_units is None else
                                   abs(changed_size - target_changed_units) <= size_match_tolerance
                                   * target_changed_units)}
    return record, None, call_summaries


# ----------------------------------------------------------------------------
# Selection, reuse of the preachiness rewrites, and neutral size targets
# ----------------------------------------------------------------------------

def select_all_facets(corpus, documents_per_facet, seed, depiction_labels_by_document_id):
    selected_ids, reports = {}, {}
    for facet in FACETS_WITH_NEW_TOWARD_AND_AWAY_REWRITES:
        selected, report = select_documents_for_facet(
            corpus, facet, documents_per_facet, seed, depiction_labels_by_document_id,
            rules=FACET_SELECTION_RULES[facet])
        selected_ids[facet], reports[facet] = [document["doc_id"] for document in selected], report
    return selected_ids, reports


def convert_preachiness_rewrites_to_non_intrusive_records(preachiness_records, corpus_by_id, minimum_changed_units,
                                                          tokenize_function=None):
    """The existing preachier/plainer rewrites as non-intrusive toward/away records, with their changed counts."""
    records = []
    for old in preachiness_records:
        direction = PREACHINESS_DIRECTION_TO_NON_INTRUSIVE_DIRECTION[old["rewrite_direction"]]
        source = corpus_by_id[old["source_document_id"]]
        counts = count_changed_units(source["content"], old["content"], old["language"], tokenize_function)
        language_minimum = minimum_units_for_language(minimum_changed_units, old["language"])
        records.append({
            "document_id": f"{old['source_document_id']}__{defs.FACET_NON_INTRUSIVE}__{direction}",
            "source_document_id": old["source_document_id"], "facet": defs.FACET_NON_INTRUSIVE,
            "rewrite_direction": direction, "content": old["content"], "language": old["language"],
            "variables": old["variables"], "rewrite_model_requested": old.get("rewrite_model_requested"),
            "rewrite_model_used": old.get("rewrite_model_used"), "reused_from_preachiness_rewrites": True,
            "original_preachiness_document_id": old["document_id"],
            "original_preachiness_rewrite_direction": old["rewrite_direction"],
            **counts, "changed_units_for_size_matching": changed_units_for_size_matching(counts),
            "minimum_changed_units_applied": language_minimum,
            "below_minimum_changed_units": is_below_minimum_changed_units(counts, language_minimum),
            "number_of_attempts": 0, "changed_units_per_attempt": [],
            "length_unit": old.get("length_unit"), "source_length": old.get("source_length"),
            "rewrite_length": old.get("rewrite_length"), "length_ratio": old.get("length_ratio"),
            "length_within_ten_percent": (None if old.get("length_ratio") is None
                                          else abs(old["length_ratio"] - 1) <= MAXIMUM_LENGTH_RATIO_DEVIATION),
            "target_changed_units_for_neutral": None, "size_matched_to_target": None})
    return records


def neutral_target_size_from_records(records_for_document_and_facet):
    """Mean changed size of the document's toward and away rewrites (those that exist and are not flagged), or None
    if neither exists. `records_for_document_and_facet` maps direction to record."""
    sizes = [record["changed_units_for_size_matching"] for direction, record in records_for_document_and_facet.items()
             if direction in SIZE_REFERENCE_DIRECTIONS and not record["below_minimum_changed_units"]]
    if not sizes:
        sizes = [record["changed_units_for_size_matching"]
                 for direction, record in records_for_document_and_facet.items()
                 if direction in SIZE_REFERENCE_DIRECTIONS]
    return sum(sizes) / len(sizes) if sizes else None


def index_records(records):
    """{(source_document_id, facet): {direction: record}}; the last record for a key wins."""
    index = {}
    for record in records:
        index.setdefault((record["source_document_id"], record["facet"]), {})[record["rewrite_direction"]] = record
    return index


def find_completed_keys_in_folder(run_folder):
    return {(record["source_document_id"], record["facet"], record["rewrite_direction"])
            for record in read_all_records(run_folder, REWRITES_FILE_PATTERN)}


# ----------------------------------------------------------------------------
# Export for the scoring pod
# ----------------------------------------------------------------------------

EXPORT_FIELDS = ["document_id", "source_document_id", "rewrite_direction", "content", "language", "variables",
                 "facet"]


def build_scoring_pod_extra_documents(records, include_flagged=False):
    """Records in the shape of --extra-documents-jsonl (document_id, source_document_id, rewrite_direction, content,
    language, variables) plus `facet`. Rewrites below the minimum changed units, and neutral rewrites whose size is
    not matched to the target, are left out unless include_flagged."""
    exported = []
    for record in sorted(records, key=lambda record: record["document_id"]):
        if not include_flagged and (record["below_minimum_changed_units"]
                                    or record.get("size_matched_to_target") is False):
            continue
        exported.append({field: record[field] for field in EXPORT_FIELDS})
    return exported


def write_export(run_folder, timestamp, include_flagged):
    records = read_all_records(run_folder, REWRITES_FILE_PATTERN)
    latest = {}
    for record in records:
        latest[record["document_id"]] = record
    exported = build_scoring_pod_extra_documents(list(latest.values()), include_flagged)
    path = Path(run_folder) / f"scoring_pod_extra_documents_{timestamp}.jsonl"
    write_jsonl(path, exported)
    print(f"Exported {len(exported)} of {len(latest)} rewrites to {path}", flush=True)
    return path


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def build_phase_tasks(selected_ids_by_facet, corpus_by_id, completed_keys, phase, neutral_facet_documents=None):
    """phase 'toward_and_away': new facets only. phase 'neutral': every facet with neutral rewrites; documents for
    non_intrusive come from `neutral_facet_documents`."""
    tasks = []
    if phase == "toward_and_away":
        for facet in FACETS_WITH_NEW_TOWARD_AND_AWAY_REWRITES:
            for document_id in selected_ids_by_facet[facet]:
                for direction in (DIRECTION_TOWARD, DIRECTION_AWAY):
                    if (document_id, facet, direction) not in completed_keys:
                        tasks.append((corpus_by_id[document_id], facet, direction))
    else:
        for facet in FACETS_WITH_NEUTRAL_REWRITES:
            for document_id in (neutral_facet_documents or selected_ids_by_facet)[facet]:
                if (document_id, facet, DIRECTION_NEUTRAL) not in completed_keys:
                    tasks.append((corpus_by_id[document_id], facet, DIRECTION_NEUTRAL))
    return tasks


def run_phase(client, run_folder, timestamp, templates, tasks, arguments, tracker, writer, phase_name,
              tokenize_function=None, request_function=None):
    run_folder = Path(run_folder)
    rewrites_path = run_folder / f"facet_positive_control_rewrites_{timestamp}.jsonl"
    failures_path = run_folder / f"rewrite_failures_{timestamp}.jsonl"
    per_call_path = run_folder / f"per_call_usage_and_cost_{timestamp}.jsonl"
    usage_log_path = run_folder / f"usage_and_cost_log_{timestamp}.json"
    print(f"Phase {phase_name}: {len(tasks)} rewrites to make.", flush=True)
    if not tasks:
        return
    lock = threading.Lock()

    def cost_so_far():
        return tracker.overall_totals["cost_dollars_of_priced_attempts"]

    def status_text():
        totals = tracker.overall_totals
        return (f"phase {phase_name}; cost so far ${cost_so_far():.4f} of cap ${arguments.maximum_cost_dollars:.2f} "
                f"({totals['calls_with_unknown_cost']} calls with unknown cost); tokens in {totals['input_tokens']} "
                f"out {totals['output_tokens']} cache read {totals['cache_read_tokens']} "
                f"write {totals['cache_write_tokens']}, cache rate {totals['cache_rate']:.2f}")

    reporter = TimedProgressReporter(
        run_folder / f"progress_log_{timestamp}.txt", len(tasks), 0, status_suffix_function=status_text,
        after_report_function=lambda: tracker.write_log_file(usage_log_path, "this session only"))
    neutral_targets = arguments.neutral_targets_by_task if phase_name == "neutral" else {}

    def work(task):
        document, facet, direction = task
        if cost_so_far() >= arguments.maximum_cost_dollars:
            writer.append(failures_path, {"source_document_id": document["doc_id"], "facet": facet,
                                          "direction": direction, "failure_kind": "skipped_cost_cap"})
            reporter.note_task_finished(failed=True)
            return
        record, failure_record, call_summaries = make_one_rewrite(
            client, templates, document, facet, direction, arguments.minimum_changed_units,
            not arguments.no_regeneration, arguments.maximum_regenerations, tokenize_function,
            target_changed_units=neutral_targets.get((document["doc_id"], facet)),
            size_match_tolerance=arguments.size_match_tolerance, request_function=request_function,
            cost_is_over_cap_function=lambda: cost_so_far() >= arguments.maximum_cost_dollars)
        for call_summary in call_summaries:
            tracker.add_call_summary(f"{facet}__{direction}", call_summary)
            writer.append(per_call_path, {"source_document_id": document["doc_id"], "facet": facet,
                                          "direction": direction, **call_summary})
        writer.append(rewrites_path if record else failures_path, record or failure_record)
        reporter.note_task_finished(failed=record is None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        list(executor.map(work, tasks))
    reporter.finish()
    tracker.write_log_file(usage_log_path, "this session only")


def run_all_phases(client, run_folder, timestamp, templates, selected_ids_by_facet, corpus_by_id, arguments,
                   reused_records, tokenize_function=None, request_function=None):
    run_folder = Path(run_folder)
    writer, tracker = AppendOnlyJsonlWriter(), AnthropicUsageAndCostTracker()
    rewrites_path = run_folder / f"facet_positive_control_rewrites_{timestamp}.jsonl"
    completed = find_completed_keys_in_folder(run_folder)
    for record in reused_records:  # written once; resume finds them
        key = (record["source_document_id"], record["facet"], record["rewrite_direction"])
        if key not in completed:
            writer.append(rewrites_path, record)
            completed.add(key)
    run_phase(client, run_folder, timestamp, templates,
              build_phase_tasks(selected_ids_by_facet, corpus_by_id, completed, "toward_and_away"), arguments,
              tracker, writer, "toward_and_away", tokenize_function, request_function)
    index = index_records(read_all_records(run_folder, REWRITES_FILE_PATTERN))
    neutral_document_ids = {facet: list(selected_ids_by_facet.get(facet, [])) for facet in FACETS_WITH_NEUTRAL_REWRITES}
    neutral_document_ids[defs.FACET_NON_INTRUSIVE] = sorted(
        {record["source_document_id"] for record in reused_records}
        | {source for (source, facet) in index if facet == defs.FACET_NON_INTRUSIVE})
    arguments.neutral_targets_by_task = {}
    for facet in FACETS_WITH_NEUTRAL_REWRITES:
        for document_id in neutral_document_ids[facet]:
            target = neutral_target_size_from_records(index.get((document_id, facet), {}))
            if target is not None:
                arguments.neutral_targets_by_task[(document_id, facet)] = target
    neutral_tasks = [task for task in build_phase_tasks(selected_ids_by_facet, corpus_by_id,
                                                        find_completed_keys_in_folder(run_folder), "neutral",
                                                        neutral_document_ids)
                     if (task[0]["doc_id"], task[1]) in arguments.neutral_targets_by_task]
    skipped = sum(1 for facet in FACETS_WITH_NEUTRAL_REWRITES for document_id in neutral_document_ids[facet]
                  if (document_id, facet) not in arguments.neutral_targets_by_task)
    if skipped:
        print(f"{skipped} documents have no toward or away rewrite yet, so no size target for a neutral rewrite; "
              "resume after those succeed.", flush=True)
    run_phase(client, run_folder, timestamp, templates, neutral_tasks, arguments, tracker, writer, "neutral",
              tokenize_function, request_function)
    return tracker


# ----------------------------------------------------------------------------
# Dry run and cost ceiling
# ----------------------------------------------------------------------------

def estimate_tokens(text, language):
    return preachiness_rewrites.estimate_tokens_from_characters(text, language)


def estimate_and_ceiling(planned_requests, regeneration_enabled, maximum_regenerations):
    """Expected cost (one attempt per call plus an assumed regeneration rate) and the ceiling (every call uses all
    its attempts, output at 1.3 times the source plus 24,000 thinking tokens)."""
    model = preachiness_rewrites.REWRITE_MODEL_REQUESTED
    expected_input = expected_output = ceiling_input = ceiling_output = 0.0
    attempts_per_call = 1 + (maximum_regenerations if regeneration_enabled else 0)
    for request in planned_requests:
        input_tokens = estimate_tokens(request["prompt_text"], request["language"])
        source_tokens = estimate_tokens(request["source_content"], request["language"])
        expected_calls = 1 + (EXPECTED_REGENERATION_RATE if regeneration_enabled else 0)
        expected_input += expected_calls * input_tokens
        expected_output += expected_calls * (source_tokens + preachiness_rewrites.ASSUMED_THINKING_TOKENS_PER_CALL)
        ceiling_input += attempts_per_call * (input_tokens + 300)
        ceiling_output += attempts_per_call * (CEILING_OUTPUT_LENGTH_MULTIPLIER * source_tokens
                                               + CEILING_THINKING_TOKENS_PER_CALL)
    expected_cost = compute_anthropic_cost_in_dollars(model, expected_input, expected_output)
    return {
        "calls_planned": len(planned_requests), "attempts_per_call_at_ceiling": attempts_per_call,
        "expected_input_tokens": round(expected_input), "expected_output_tokens": round(expected_output),
        "expected_cost_dollars": round(expected_cost, 2),
        "scaled_from_preachiness_run_cost_dollars": round(
            len(planned_requests) * MEASURED_DOLLARS_PER_REWRITE_CALL_IN_PREACHINESS_RUN
            * (1 + (EXPECTED_REGENERATION_RATE if regeneration_enabled else 0)), 2),
        "default_cost_cap_dollars": round(DEFAULT_COST_CAP_MULTIPLE_OF_CAUTIOUS_EXPECTED_COST * expected_cost, 2),
        "ceiling_input_tokens": round(ceiling_input), "ceiling_output_tokens": round(ceiling_output),
        "ceiling_cost_dollars": round(compute_anthropic_cost_in_dollars(model, ceiling_input, ceiling_output), 2),
        "assumptions": (f"expected_cost is cautious; scaled_from_preachiness_run is calls x $2.27/20 measured on the "
                        f"preachiness rewrites; {model} prices {PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL[model]} per million; input "
                        "tokens from character counts; expected output = source length + "
                        f"{preachiness_rewrites.ASSUMED_THINKING_TOKENS_PER_CALL} thinking tokens with "
                        f"{EXPECTED_REGENERATION_RATE:.0%} of calls regenerated once; ceiling = every call uses all "
                        f"attempts, output {CEILING_OUTPUT_LENGTH_MULTIPLIER}x source + "
                        f"{CEILING_THINKING_TOKENS_PER_CALL} thinking tokens; neutral prompts assume a target of "
                        f"{ASSUMED_FRACTION_OF_DOCUMENT_CHANGED_FOR_NEUTRAL_TARGET:.0%} of the document; no cache hits")}


def plan_requests_for_dry_run(templates, selected_ids_by_facet, corpus_by_id, non_intrusive_document_ids, arguments):
    planned = []
    for facet in FACETS_WITH_NEW_TOWARD_AND_AWAY_REWRITES:
        for document_id in selected_ids_by_facet[facet]:
            document = corpus_by_id[document_id]
            for direction in (DIRECTION_TOWARD, DIRECTION_AWAY):
                planned.append({"source_document_id": document_id, "facet": facet, "direction": direction,
                                "language": document["language"], "source_content": document["content"],
                                "prompt_text": render_instruction(templates[instruction_file_name(facet, direction)],
                                                                  document, arguments.minimum_changed_units)})
    neutral_documents = dict(selected_ids_by_facet)
    neutral_documents[defs.FACET_NON_INTRUSIVE] = non_intrusive_document_ids
    for facet in FACETS_WITH_NEUTRAL_REWRITES:
        for document_id in neutral_documents[facet]:
            document = corpus_by_id[document_id]
            assumed_target = ASSUMED_FRACTION_OF_DOCUMENT_CHANGED_FOR_NEUTRAL_TARGET * len(document["content"].split())
            planned.append({"source_document_id": document_id, "facet": facet, "direction": DIRECTION_NEUTRAL,
                            "language": document["language"], "source_content": document["content"],
                            "prompt_text": render_instruction(
                                templates[instruction_file_name(facet, DIRECTION_NEUTRAL)], document,
                                arguments.minimum_changed_units, target_changed_units=assumed_target)})
    return planned


def run_dry_run(run_folder, timestamp, planned_requests, arguments):
    estimate = estimate_and_ceiling(planned_requests, not arguments.no_regeneration, arguments.maximum_regenerations)
    write_jsonl(Path(run_folder) / f"rendered_rewrite_requests_{timestamp}.jsonl", planned_requests)
    (Path(run_folder) / f"dry_run_cost_estimate_and_ceiling_{timestamp}.json").write_text(
        json.dumps(estimate, indent=2), encoding="utf-8")
    print(json.dumps(estimate, indent=2))
    print(f"COST CEILING for the paid run: absolute worst case ${estimate['ceiling_cost_dollars']:.2f}; default stop "
          f"cap ${estimate['default_cost_cap_dollars']:.2f} (--maximum-cost-dollars); cautious expected "
          f"${estimate['expected_cost_dollars']:.2f}; scaled from the preachiness run "
          f"${estimate['scaled_from_preachiness_run_cost_dollars']:.2f}.", flush=True)
    return estimate


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="facet_positive_control_rewrites_opus_5")
    parser.add_argument("--resume-folder", default=None)
    parser.add_argument("--documents-per-facet", type=int, default=DEFAULT_DOCUMENTS_PER_FACET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depiction-labels-jsonl", default=None,
                        help="Output of label_depiction_of_facets_in_corpus_documents.py; used to judge room to move "
                             "and, for consistent, whether a document discusses several kinds of animal")
    parser.add_argument("--selected-documents-json", default=None,
                        help="JSON {facet: [doc_id, ...]} overriding the selection")
    parser.add_argument("--existing-preachiness-rewrites-jsonl", action="append", default=None,
                        help="Existing preachier/plainer rewrites to reuse for the non-intrusive facet (repeatable); "
                             f"default {DEFAULT_EXISTING_PREACHINESS_REWRITES_PATH.name} of the 2026-10-22 run")
    parser.add_argument("--minimum-changed-units", type=int, default=DEFAULT_MINIMUM_CHANGED_UNITS,
                        help="Minimum changed tokens by diff (words, or 1.5x characters for Chinese and Japanese)")
    parser.add_argument("--tokenizer-name", default=None,
                        help="Count changed units in this Hugging Face tokenizer's tokens (local cache only)")
    parser.add_argument("--no-regeneration", action="store_true",
                        help="Flag rewrites below the minimum but do not regenerate them")
    parser.add_argument("--maximum-regenerations", type=int, default=DEFAULT_MAXIMUM_REGENERATIONS)
    parser.add_argument("--size-match-tolerance", type=float, default=DEFAULT_SIZE_MATCH_TOLERANCE)
    parser.add_argument("--maximum-cost-dollars", type=float, default=None,
                        help="Stop submitting calls once this much has been spent (default: 1.5 times the cautious expected cost of the dry run)")
    parser.add_argument("--include-flagged-in-export", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="Only write the scoring-pod export (needs --resume-folder)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--env-path", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--dry-run", action="store_true",
                        help="Render every request, write them to small_sample_checks, print expected cost and "
                             "ceiling; no API call")
    return parser.parse_args(argv)


def load_depiction_labels(path):
    if not path:
        return None
    return {record["document_id"]: record for record in read_jsonl(path)}


def main(argv=None, client=None, runs_directory_override=None, request_function=None):
    arguments = parse_arguments(argv)
    arguments.neutral_targets_by_task = {}
    timestamp = timestamp_now()
    if arguments.export_only:
        if not arguments.resume_folder:
            raise SystemExit("--export-only needs --resume-folder")
        write_export(arguments.resume_folder, timestamp, arguments.include_flagged_in_export)
        return arguments.resume_folder
    corpus = read_jsonl(arguments.corpus_path)
    corpus_by_id = {document["doc_id"]: document for document in corpus}
    runs_directory = runs_directory_override or (DRY_RUNS_DIRECTORY if arguments.dry_run else FULL_RUNS_DIRECTORY)
    run_folder, is_resume = choose_run_folder(runs_directory, arguments.run_name, arguments.resume_folder)
    templates = load_instruction_templates()
    tokenize_function = make_tokenize_function(arguments.tokenizer_name)
    existing_paths = arguments.existing_preachiness_rewrites_jsonl or [str(DEFAULT_EXISTING_PREACHINESS_REWRITES_PATH)]
    prompt_hashes = {name: sha256_of_file(PROMPT_DIRECTORY / name) for name in all_instruction_file_names()}

    if is_resume:
        first_configuration = json.loads(sorted(Path(run_folder).glob("run_configuration_*.json"))[0]
                                         .read_text(encoding="utf-8"))
        selected_ids_by_facet = first_configuration["selected_document_ids_by_facet"]
        selection_reports = first_configuration.get("selection_reports", {})
        check_resume_is_compatible(run_folder, {"prompt_file_sha256": prompt_hashes})
    elif arguments.selected_documents_json:
        selected_ids_by_facet = json.loads(Path(arguments.selected_documents_json).read_text(encoding="utf-8"))
        selection_reports = {"note": "selection overridden by --selected-documents-json"}
    else:
        selected_ids_by_facet, selection_reports = select_all_facets(
            corpus, arguments.documents_per_facet, arguments.seed, load_depiction_labels(arguments.depiction_labels_jsonl))
    for facet, reports in (selection_reports.items() if isinstance(selection_reports, dict) else []):
        for warning in (reports.get("warnings", []) if isinstance(reports, dict) else []):
            print(f"WARNING ({facet}): {warning}", flush=True)

    preachiness_records = [record for path in existing_paths for record in read_jsonl(path)]
    reused_records = convert_preachiness_rewrites_to_non_intrusive_records(
        preachiness_records, corpus_by_id, arguments.minimum_changed_units, tokenize_function)
    non_intrusive_document_ids = sorted({record["source_document_id"] for record in reused_records})

    planned = plan_requests_for_dry_run(templates, selected_ids_by_facet, corpus_by_id, non_intrusive_document_ids,
                                        arguments)
    estimate = estimate_and_ceiling(planned, not arguments.no_regeneration, arguments.maximum_regenerations)
    if arguments.maximum_cost_dollars is None:
        arguments.maximum_cost_dollars = estimate["default_cost_cap_dollars"]
    configuration = {
        "script": "make_facet_positive_control_rewrites.py", "dry_run": arguments.dry_run,
        "model_requested": preachiness_rewrites.REWRITE_MODEL_REQUESTED,
        "max_tokens": preachiness_rewrites.MAXIMUM_OUTPUT_TOKENS, "sampling_parameters": "unset",
        "thinking": "unset (adaptive by default)", "fallbacks": "default",
        "beta": preachiness_rewrites.FALLBACK_BETA_HEADER, "prompt_file_sha256": prompt_hashes,
        "corpus_path": str(arguments.corpus_path), "corpus_sha256": sha256_of_file(arguments.corpus_path),
        "existing_preachiness_rewrites_files": existing_paths,
        "existing_preachiness_rewrites_sha256": {path: sha256_of_file(path) for path in existing_paths},
        "prices_per_million_tokens": PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL,
        "documents_per_facet": arguments.documents_per_facet, "seed": arguments.seed,
        "depiction_labels_jsonl": arguments.depiction_labels_jsonl,
        "minimum_changed_units": arguments.minimum_changed_units, "tokenizer_name": arguments.tokenizer_name,
        "regeneration_enabled": not arguments.no_regeneration, "maximum_regenerations": arguments.maximum_regenerations,
        "size_match_tolerance": arguments.size_match_tolerance, "maximum_cost_dollars": arguments.maximum_cost_dollars,
        "cost_estimate_and_ceiling": estimate, "worker_count": arguments.workers, "resumed": is_resume,
        "selected_document_ids_by_facet": selected_ids_by_facet, "selection_reports": selection_reports,
        "non_intrusive_document_ids": non_intrusive_document_ids}
    write_run_configuration(run_folder, timestamp, configuration)
    print(f"Run folder: {run_folder}", flush=True)

    if arguments.dry_run:
        run_dry_run(run_folder, timestamp, planned, arguments)
        return run_folder
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=preachiness_rewrites.load_anthropic_api_key(arguments.env_path),
                                     max_retries=3)
    run_all_phases(client, run_folder, timestamp, templates, selected_ids_by_facet, corpus_by_id, arguments,
                   reused_records, tokenize_function, request_function)
    write_export(run_folder, timestamp, arguments.include_flagged_in_export)
    return run_folder


if __name__ == "__main__":
    main()
