"""Positive-control rewrites for a content-controlled facet (PAID when run without --dry-run: Claude Opus 5). `--facet` picks the
facet (default moralizing_delivery, whose behaviour and outputs are unchanged); the arms, their instruction files, what each arm is
applied to and compared with, and the expected signs come from content_controlled_facet_definitions.py.

New facets (cares_about_welfare, calibrated_consideration, non_intrusive): arms trait_only_away, content_added_without_trait,
facet_neutral_rewording (each on the original corpus document) and trait_removed_from_away_rewrite, which is applied to the OUTPUT of
trait_only_away for the same source and compared with THAT rewrite. The run therefore makes the chain-free arms first, then the
chained arm from the finished trait_only_away rewrites; a source whose trait_only_away rewrite is missing gets no chained rewrite
(recorded in rewrite_failures as skipped_no_source_rewrite; the chained arm is never run on the original). Resumable: finished
rewrites, including trait_only_away rewrites of an earlier session, are reused. Sources: by default the sources of the preachiness
rewrites (the same ten documents the moralizing delivery rewrites used); `--source-document-ids-file` (one id per line or a JSON
list) chooses them per facet. Size: the new facets have no preachier rewrite per source, so every arm, the chained one included, is
asked for `--target-changed-units` changed words (default 274, the mean number of words the moralizing delivery rewrites changed).

Moralizing delivery, as before:

Four arms, ten documents each, the same model and settings as the preachiness rewrites (see
moralizing_delivery_positive_control_arms.py for what each arm changes and what it is compared with):
  moralizing_framing_only                    on the 10 source documents of the preachiness rewrites
  plain_welfare_content_added                on the same 10 sources
  facet_neutral_rewording                    on the same 10 sources
  moralizing_removed_from_preachier_rewrite  on the 10 existing preachier rewrites (compared with the preachier rewrite)

Size. Each rewrite is asked to change about as much as that source document's own preachier rewrite did (words changed
by a diff, the larger of words inserted and words removed; about 300 tokens on average), so every arm of a document has the
same target. The existing minimum-change retry of make_facet_positive_control_rewrites.py is reused as it is: a rewrite that
changed fewer than --minimum-changed-units is regenerated up to --maximum-regenerations times with a note appended, and
is flagged `below_minimum_changed_units` if it still falls short. Flagged rewrites are KEPT in the export (the criteria
count ten rewrites per arm) and carry the flag, so the analysis reports them.

Files in the run folder (every name ends with the date and time of the session that wrote it):
  run_configuration_<timestamp>.json, progress_log_<timestamp>.txt
  moralizing_delivery_positive_control_rewrites_<timestamp>.jsonl   one record per finished rewrite (read back on resume)
  rewrite_failures_<timestamp>.jsonl                                refusals, max_tokens, empty output, transport errors
  per_call_usage_and_cost_<timestamp>.jsonl, usage_and_cost_log_<timestamp>.json   tokens, cost and cache rate
  scoring_pod_extra_documents_<timestamp>.jsonl                     export read by build_moralizing_delivery_scoring_run_folder.py
Dry runs write to data/persona_vector_projection_runs/small_sample_checks/ and make no API call.

Dry-run caveat for the chained arm: its real input is a rewrite that does not exist yet, so the dry run renders its request on the
original document (flagged `document_is_placeholder_for_arm_one_output`) to count tokens and cost.

The API key is read ONLY from the project .env (ANTHROPIC_API_KEY), never from the OS environment.

    python3 scripts/persona_vector_projection/make_moralizing_delivery_positive_control_rewrites.py --dry-run
    python3 scripts/persona_vector_projection/make_moralizing_delivery_positive_control_rewrites.py                     (PAID)
    python3 scripts/persona_vector_projection/make_moralizing_delivery_positive_control_rewrites.py --resume-folder <folder>   (PAID)
    python3 scripts/persona_vector_projection/make_moralizing_delivery_positive_control_rewrites.py --export-only --resume-folder <folder>
"""

import argparse
import concurrent.futures
import json
import sys
import threading
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for extra_path in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import make_preachiness_positive_control_rewrites as preachiness_rewrites  # noqa: E402
from persona_vector_projection import make_facet_positive_control_rewrites as facet_rewrites  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_arms as arms_module  # noqa: E402
from persona_vector_projection.changed_token_counting_by_diff import (  # noqa: E402
    changed_units_for_size_matching, count_changed_units)
from shared.anthropic_usage_and_cost_tracking import AnthropicUsageAndCostTracker  # noqa: E402
from shared.resumable_jsonl_records import AppendOnlyJsonlWriter, read_all_records, read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import (check_resume_is_compatible, choose_run_folder, sha256_of_file,  # noqa: E402
                                          timestamp_now, write_run_configuration)
from shared.timed_progress_reporter import TimedProgressReporter  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
PROMPT_DIRECTORY = facet_rewrites.PROMPT_DIRECTORY
DEFAULT_CORPUS_PATH = facet_rewrites.DEFAULT_CORPUS_PATH
DEFAULT_ENV_PATH = facet_rewrites.DEFAULT_ENV_PATH
DEFAULT_PREACHINESS_REWRITES_PATH = facet_rewrites.DEFAULT_EXISTING_PREACHINESS_REWRITES_PATH
FULL_RUNS_DIRECTORY = facet_rewrites.FULL_RUNS_DIRECTORY
DRY_RUNS_DIRECTORY = facet_rewrites.DRY_RUNS_DIRECTORY
REWRITES_FILE_PATTERN = "moralizing_delivery_positive_control_rewrites_*.jsonl"
RETRY_NOTE_FILE_NAME = facet_rewrites.RETRY_NOTE_FILE_NAME
FACET = arms_module.REWRITE_FACET_NAME  # moralizing delivery's facet key, the default of every function below
DEFAULT_TARGET_CHANGED_UNITS_FOR_NEW_FACETS = 274  # words: the mean changed by the moralizing delivery rewrites (runbook: about 274)
SKIPPED_NO_SOURCE_REWRITE = "skipped_no_source_rewrite"

EXPORT_FIELDS_OF_THE_NEW_FACETS_ADDED = ["comparison_document_id"]
EXPORT_FIELDS = ["document_id", "source_document_id", "rewrite_direction", "content", "language", "variables", "facet", "arm",
                 "original_corpus_source_document_id", "expected_sign", "below_minimum_changed_units",
                 "changed_units_for_size_matching", "target_changed_units"]


# ----------------------------------------------------------------------------
# Tasks: which document each arm starts from, and how much it should change
# ----------------------------------------------------------------------------

def load_instruction_templates(prompt_directory=PROMPT_DIRECTORY, definition=None):
    """{key the reused rewrite function looks up: instruction text}, one key per arm, plus the retry note. Each arm's file is the first
    existing one of the arm's candidate file names (arm 4 of the new facets: `facet_neutral_rewording`, else the earlier
    `facet_neutral`); a missing file stops the run, naming every candidate tried."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    file_by_arm = arms_module.resolve_instruction_file_names(definition, prompt_directory)
    templates = {facet_rewrites.instruction_file_name(definition.facet_key, arm): (Path(prompt_directory) / file_name).read_text(encoding="utf-8")
                 for arm, file_name in file_by_arm.items()}
    templates[RETRY_NOTE_FILE_NAME] = (Path(prompt_directory) / RETRY_NOTE_FILE_NAME).read_text(encoding="utf-8")
    return templates


def instruction_file_names_used(definition=None, prompt_directory=PROMPT_DIRECTORY):
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    return sorted(set(arms_module.resolve_instruction_file_names(definition, prompt_directory).values()) | {RETRY_NOTE_FILE_NAME})


def preachier_rewrites_by_original_source(preachiness_records):
    """{original source id: the preachier rewrite record}; raises if a source has none or two."""
    found = {}
    for record in preachiness_records:
        if record["rewrite_direction"] != arms_module.PREACHIER_REWRITE_DIRECTION:
            continue
        if record["source_document_id"] in found:
            raise ValueError(f"Two preachier rewrites for {record['source_document_id']}")
        found[record["source_document_id"]] = record
    if not found:
        raise ValueError("No preachier rewrites found in the preachiness rewrite file")
    return found


def preachier_change_sizes(corpus_by_id, preachier_by_source, tokenize_function=None):
    """{original source id: words (or tokens) the preachier rewrite changed}, the target size for every arm of that source."""
    sizes = {}
    for source_id, rewrite in preachier_by_source.items():
        counts = count_changed_units(corpus_by_id[source_id]["content"], rewrite["content"], rewrite["language"], tokenize_function)
        sizes[source_id] = changed_units_for_size_matching(counts)
    return sizes


def build_arm_tasks(corpus_by_id, preachier_by_source, target_sizes):
    """One task per arm and source: the document the instruction is applied to (`doc_id`, `content`, `language`,
    `variables`), the arm, the original corpus source id and the target size. Sorted for a stable order."""
    tasks = []
    for original_source_id in sorted(preachier_by_source):
        corpus_document = corpus_by_id[original_source_id]
        preachier = preachier_by_source[original_source_id]
        for arm in arms_module.ARMS:
            if arm == arms_module.ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE:
                document = {"doc_id": preachier["document_id"], "content": preachier["content"],
                            "language": preachier["language"], "variables": preachier["variables"]}
            else:
                document = {"doc_id": corpus_document["doc_id"], "content": corpus_document["content"],
                            "language": corpus_document["language"], "variables": corpus_document["variables"]}
            tasks.append({"document": document, "arm": arm, "original_source_id": original_source_id,
                          "target_changed_units": target_sizes[original_source_id]})
    return tasks


def build_independent_arm_tasks(definition, corpus_by_id, source_ids, target_changed_units):
    """New facets: one task per arm that starts from the ORIGINAL document and per source (the chained arm is built later, from the
    finished trait_only_away rewrites). Sorted for a stable order."""
    tasks = []
    for original_source_id in sorted(source_ids):
        corpus_document = corpus_by_id[original_source_id]
        for arm in arms_module.arm_definitions_for(definition):
            if arm.starts_from != facet_definitions.STARTS_FROM_ORIGINAL_DOCUMENT:
                continue
            tasks.append({"document": {"doc_id": corpus_document["doc_id"], "content": corpus_document["content"],
                                       "language": corpus_document["language"], "variables": corpus_document["variables"]},
                          "arm": arm.arm_name, "original_source_id": original_source_id, "target_changed_units": target_changed_units})
    return tasks


def build_chained_arm_tasks(definition, source_ids, rewrite_records, target_changed_units):
    """(tasks, skipped). For each arm that starts from another arm's output (trait_removed_from_away_rewrite) and each source, a task
    whose document IS the finished rewrite of the source arm (its text, its document id), so the instruction is applied to arm 1's
    output and the comparison document is that rewrite. `skipped` lists (arm, source) with no finished source rewrite: these get no
    task, so the chained arm is never applied to the original document."""
    latest = {record["document_id"]: record for record in rewrite_records}
    tasks, skipped = [], []
    for arm in arms_module.arm_definitions_for(definition):
        if arm.starts_from != facet_definitions.STARTS_FROM_OUTPUT_OF_ARM:
            continue
        for original_source_id in sorted(source_ids):
            source_record = latest.get(arms_module.rewrite_document_id_for(definition, arm.source_arm_name, original_source_id))
            if source_record is None:
                skipped.append((arm.arm_name, original_source_id))
                continue
            tasks.append({"document": {"doc_id": source_record["document_id"], "content": source_record["content"],
                                       "language": source_record["language"], "variables": source_record["variables"]},
                          "arm": arm.arm_name, "original_source_id": original_source_id, "target_changed_units": target_changed_units})
    return tasks, skipped


def finish_record(record, task, definition=None):
    """Rename and annotate the reused function's record: the id of an arm's rewrite names the ORIGINAL source, and the
    comparison document is the document the instruction was applied to."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    arm, original_source_id = task["arm"], task["original_source_id"]
    comparison_id = arms_module.comparison_document_id_for(definition, arm, original_source_id)
    if comparison_id != task["document"]["doc_id"]:
        raise ValueError(f"Arm {arm} of {original_source_id} was applied to {task['document']['doc_id']} but is compared "
                         f"with {comparison_id}")
    record.update({"document_id": arms_module.rewrite_document_id_for(definition, arm, original_source_id), "source_document_id": comparison_id,
                   "comparison_document_id": comparison_id,
                   "original_corpus_source_document_id": original_source_id, "arm": arm, "facet": definition.facet_key,
                   "rewrite_direction": arm, "expected_sign": arms_module.expected_sign_for(definition, arm),
                   "target_changed_units": task["target_changed_units"]})
    return record


# ----------------------------------------------------------------------------
# Run (resumable)
# ----------------------------------------------------------------------------

def completed_document_ids(run_folder, definition=None):
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    return {record["document_id"] for record in read_all_records(run_folder, arms_module.rewrites_file_pattern(definition))}


def run_rewrites(client, run_folder, timestamp, templates, tasks, arguments, request_function=None, tokenize_function=None,
                 definition=None, tracker=None):
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    run_folder = Path(run_folder)
    rewrites_path = run_folder / f"{definition.facet_key}_positive_control_rewrites_{timestamp}.jsonl"
    failures_path = run_folder / f"rewrite_failures_{timestamp}.jsonl"
    per_call_path = run_folder / f"per_call_usage_and_cost_{timestamp}.jsonl"
    usage_log_path = run_folder / f"usage_and_cost_log_{timestamp}.json"
    done = completed_document_ids(run_folder, definition)
    remaining = [task for task in tasks
                 if arms_module.rewrite_document_id_for(definition, task["arm"], task["original_source_id"]) not in done]
    print(f"{len(remaining)} rewrites to make ({len(tasks) - len(remaining)} already finished).", flush=True)
    writer, tracker, lock = AppendOnlyJsonlWriter(), tracker or AnthropicUsageAndCostTracker(), threading.Lock()
    if not remaining:
        return tracker

    def cost_so_far():
        return tracker.overall_totals["cost_dollars_of_priced_attempts"]

    def status_text():
        totals = tracker.overall_totals
        return (f"cost so far ${cost_so_far():.4f} of cap ${arguments.maximum_cost_dollars:.2f} "
                f"({totals['calls_with_unknown_cost']} calls with unknown cost); tokens in {totals['input_tokens']} "
                f"out {totals['output_tokens']} cache read {totals['cache_read_tokens']} write {totals['cache_write_tokens']}, "
                f"cache rate {totals['cache_rate']:.2f}")

    reporter = TimedProgressReporter(run_folder / f"progress_log_{timestamp}.txt", len(remaining), 0,
                                     status_suffix_function=status_text,
                                     after_report_function=lambda: tracker.write_log_file(usage_log_path, "this session only"))

    def work(task):
        document, arm = task["document"], task["arm"]
        if cost_so_far() >= arguments.maximum_cost_dollars:
            writer.append(failures_path, {"arm": arm, "original_source_id": task["original_source_id"],
                                          "failure_kind": "skipped_cost_cap"})
            reporter.note_task_finished(failed=True)
            return
        record, failure, call_summaries = facet_rewrites.make_one_rewrite(
            client, templates, document, definition.facet_key, arm, arguments.minimum_changed_units, not arguments.no_regeneration,
            arguments.maximum_regenerations, tokenize_function, target_changed_units=task["target_changed_units"],
            size_match_tolerance=arguments.size_match_tolerance, request_function=request_function,
            cost_is_over_cap_function=lambda: cost_so_far() >= arguments.maximum_cost_dollars)
        for call_summary in call_summaries:
            tracker.add_call_summary(arm, call_summary)
            writer.append(per_call_path, {"arm": arm, "original_source_id": task["original_source_id"], **call_summary})
        if record:
            writer.append(rewrites_path, finish_record(record, task, definition))
        else:
            writer.append(failures_path, {**failure, "arm": arm, "original_source_id": task["original_source_id"]})
        reporter.note_task_finished(failed=record is None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        list(executor.map(work, remaining))
    reporter.finish()
    tracker.write_log_file(usage_log_path, "this session only")
    return tracker


# ----------------------------------------------------------------------------
# Export for the scoring run folder
# ----------------------------------------------------------------------------

def build_export_records(records, definition=None):
    """Every rewrite, flagged ones included (carrying their flag), the last record per document id, sorted by id. The new facets'
    records also carry `comparison_document_id` (the same as `source_document_id`, named plainly)."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    fields = EXPORT_FIELDS + ([] if definition.legacy_moralizing_names else EXPORT_FIELDS_OF_THE_NEW_FACETS_ADDED)
    latest = {record["document_id"]: record for record in records}
    return [{field: record[field] for field in fields} for _, record in sorted(latest.items())]


def write_export(run_folder, timestamp, definition=None):
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    exported = build_export_records(read_all_records(run_folder, arms_module.rewrites_file_pattern(definition)), definition)
    path = Path(run_folder) / f"scoring_pod_extra_documents_{timestamp}.jsonl"
    write_jsonl(path, exported)
    flagged = [record["document_id"] for record in exported if record["below_minimum_changed_units"]]
    per_arm = {arm: sum(1 for record in exported if record["arm"] == arm) for arm in arms_module.arm_names_for(definition)}
    print(f"Exported {len(exported)} rewrites to {path}\n  per arm: {per_arm}\n  below the minimum changed units "
          f"(kept, flagged): {flagged}", flush=True)
    return path


# ----------------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------------

def plan_requests(templates, tasks, minimum_changed_units, definition=None):
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    planned = []
    for task in tasks:
        document = task["document"]
        template = templates[facet_rewrites.instruction_file_name(definition.facet_key, task["arm"])]
        planned.append({"arm": task["arm"], "original_source_id": task["original_source_id"], "document_id": document["doc_id"],
                        "document_is_placeholder_for_arm_one_output": bool(task.get("placeholder_for_arm_one_output")),
                        "language": document["language"], "source_content": document["content"],
                        "target_changed_units": task["target_changed_units"],
                        "prompt_text": facet_rewrites.render_instruction(
                            template, document, minimum_changed_units, target_changed_units=task["target_changed_units"])})
    return planned


def run_dry_run(run_folder, timestamp, planned_requests, arguments, target_sizes, definition=None):
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    estimate = facet_rewrites.estimate_and_ceiling(planned_requests, not arguments.no_regeneration, arguments.maximum_regenerations)
    estimate["target_changed_words_per_source_document"] = dict(sorted(target_sizes.items()))
    estimate["mean_target_changed_words"] = round(sum(target_sizes.values()) / len(target_sizes), 1)
    estimate["calls_planned_by_arm"] = {arm: sum(1 for request in planned_requests if request["arm"] == arm)
                                        for arm in arms_module.arm_names_for(definition)}
    write_jsonl(Path(run_folder) / f"rendered_rewrite_requests_{timestamp}.jsonl", planned_requests)
    (Path(run_folder) / f"dry_run_cost_estimate_and_ceiling_{timestamp}.json").write_text(
        json.dumps(estimate, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in estimate.items() if key != "assumptions"}, indent=2))
    print(f"COST CEILING for the paid run: {estimate['calls_planned']} calls (up to {estimate['attempts_per_call_at_ceiling']} attempts "
          f"each); absolute worst case ${estimate['ceiling_cost_dollars']:.2f}; default stop cap "
          f"${estimate['default_cost_cap_dollars']:.2f} (--maximum-cost-dollars); cautious expected "
          f"${estimate['expected_cost_dollars']:.2f}; scaled from the preachiness run "
          f"${estimate['scaled_from_preachiness_run_cost_dollars']:.2f}. Mean target change "
          f"{estimate['mean_target_changed_words']} words.", flush=True)
    return estimate


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def read_source_document_ids(path):
    """Source document ids from a file: a JSON list, or one id per line (blank lines and lines starting with # ignored)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    facet_definitions.add_facet_argument(parser)
    parser.add_argument("--run-name", default=None, help="Default: <facet key>_positive_control_rewrites_opus_5")
    parser.add_argument("--resume-folder", default=None)
    parser.add_argument("--preachiness-rewrites-jsonl", default=str(DEFAULT_PREACHINESS_REWRITES_PATH),
                        help="The preachier and plainer rewrites; their 10 sources and the preachier rewrites are used (the new facets "
                             "use only the sources, unless --source-document-ids-file is given)")
    parser.add_argument("--source-document-ids-file", default=None,
                        help="New facets: the corpus document ids to rewrite (JSON list or one per line), instead of the preachiness sources")
    parser.add_argument("--target-changed-units", type=float, default=DEFAULT_TARGET_CHANGED_UNITS_FOR_NEW_FACETS,
                        help="New facets: words every arm is asked to change (the chained arm too)")
    parser.add_argument("--minimum-changed-units", type=int, default=facet_rewrites.DEFAULT_MINIMUM_CHANGED_UNITS)
    parser.add_argument("--tokenizer-name", default=None, help="Count changed units in this tokenizer's tokens (local cache only)")
    parser.add_argument("--no-regeneration", action="store_true")
    parser.add_argument("--maximum-regenerations", type=int, default=facet_rewrites.DEFAULT_MAXIMUM_REGENERATIONS)
    parser.add_argument("--size-match-tolerance", type=float, default=facet_rewrites.DEFAULT_SIZE_MATCH_TOLERANCE)
    parser.add_argument("--maximum-cost-dollars", type=float, default=None,
                        help="Stop submitting calls once this much has been spent (default: the dry run's default stop cap)")
    parser.add_argument("--export-only", action="store_true", help="Only write the export (needs --resume-folder)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--env-path", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--dry-run", action="store_true", help="Render every request, write them to small_sample_checks, print the "
                                                              "call count, token estimate and dollar ceiling; no API call")
    arguments = parser.parse_args(argv)
    arguments.definition = facet_definitions.get_facet_definition(arguments.facet)
    if arguments.run_name is None:
        arguments.run_name = f"{arguments.definition.facet_key}_positive_control_rewrites_opus_5"
    return arguments


def plan_dry_run_chained_requests(definition, corpus_by_id, source_ids, target_changed_units):
    """Dry run only: the chained arm's tasks on the ORIGINAL document as a stand-in for arm 1's rewrite (which does not exist yet)."""
    tasks = []
    for arm in arms_module.arm_definitions_for(definition):
        if arm.starts_from != facet_definitions.STARTS_FROM_OUTPUT_OF_ARM:
            continue
        for original_source_id in sorted(source_ids):
            document = corpus_by_id[original_source_id]
            tasks.append({"document": {"doc_id": document["doc_id"], "content": document["content"], "language": document["language"],
                                       "variables": document["variables"]}, "arm": arm.arm_name,
                          "original_source_id": original_source_id, "target_changed_units": target_changed_units,
                          "placeholder_for_arm_one_output": True})
    return tasks


def main(argv=None, client=None, runs_directory_override=None, request_function=None):
    arguments = parse_arguments(argv)
    definition = arguments.definition
    legacy = definition.legacy_moralizing_names
    timestamp = timestamp_now()
    if arguments.export_only:
        if not arguments.resume_folder:
            raise SystemExit("--export-only needs --resume-folder")
        write_export(arguments.resume_folder, timestamp, definition)
        return arguments.resume_folder
    corpus_by_id = {document["doc_id"]: document for document in read_jsonl(arguments.corpus_path)}
    preachiness_records = read_jsonl(arguments.preachiness_rewrites_jsonl)
    preachier_by_source = preachier_rewrites_by_original_source(preachiness_records)
    tokenize_function = facet_rewrites.make_tokenize_function(arguments.tokenizer_name)
    if legacy:
        source_ids = sorted(preachier_by_source)
        target_sizes = preachier_change_sizes(corpus_by_id, preachier_by_source, tokenize_function)
        tasks = build_arm_tasks(corpus_by_id, preachier_by_source, target_sizes)
        chained_placeholder_tasks = []
    else:
        source_ids = (sorted(read_source_document_ids(arguments.source_document_ids_file)) if arguments.source_document_ids_file
                      else sorted(preachier_by_source))
        unknown = [source for source in source_ids if source not in corpus_by_id]
        if unknown:
            raise SystemExit(f"Source documents not in the corpus: {unknown}")
        target_sizes = {source: arguments.target_changed_units for source in source_ids}
        tasks = build_independent_arm_tasks(definition, corpus_by_id, source_ids, arguments.target_changed_units)
        chained_placeholder_tasks = plan_dry_run_chained_requests(definition, corpus_by_id, source_ids, arguments.target_changed_units)
    templates = load_instruction_templates(definition=definition)
    prompt_hashes = {name: sha256_of_file(PROMPT_DIRECTORY / name) for name in instruction_file_names_used(definition)}

    runs_directory = runs_directory_override or (DRY_RUNS_DIRECTORY if arguments.dry_run else FULL_RUNS_DIRECTORY)
    run_folder, is_resume = choose_run_folder(runs_directory, arguments.run_name, arguments.resume_folder)
    if is_resume:
        check_resume_is_compatible(run_folder, {"prompt_file_sha256": prompt_hashes})
    planned = plan_requests(templates, tasks + chained_placeholder_tasks, arguments.minimum_changed_units, definition)
    estimate = facet_rewrites.estimate_and_ceiling(planned, not arguments.no_regeneration, arguments.maximum_regenerations)
    if arguments.maximum_cost_dollars is None:
        arguments.maximum_cost_dollars = estimate["default_cost_cap_dollars"]
    configuration = {
        "script": "make_content_controlled_facet_positive_control_rewrites.py", "facet_key": definition.facet_key,
        "dry_run": arguments.dry_run,
        "arms": arms_module.arm_names_for(definition),
        "instruction_file_by_arm": arms_module.resolve_instruction_file_names(definition, PROMPT_DIRECTORY),
        "expected_sign_by_arm": {arm: arms_module.expected_sign_for(definition, arm) for arm in arms_module.arm_names_for(definition)},
        "direction_convention": "vector = positive pole minus negative pole; shift = rewrite minus comparison document; "
                                "a rewrite towards the negative pole should give a NEGATIVE shift",
        "model_requested": preachiness_rewrites.REWRITE_MODEL_REQUESTED, "max_tokens": preachiness_rewrites.MAXIMUM_OUTPUT_TOKENS,
        "sampling_parameters": "unset", "thinking": "unset (adaptive by default)", "fallbacks": "default",
        "beta": preachiness_rewrites.FALLBACK_BETA_HEADER, "prompt_file_sha256": prompt_hashes,
        "corpus_path": str(arguments.corpus_path), "corpus_sha256": sha256_of_file(arguments.corpus_path),
        "preachiness_rewrites_file": arguments.preachiness_rewrites_jsonl,
        "preachiness_rewrites_sha256": sha256_of_file(arguments.preachiness_rewrites_jsonl),
        "source_document_ids_file": arguments.source_document_ids_file,
        "original_source_document_ids": source_ids, "target_changed_units_by_source": target_sizes,
        "minimum_changed_units": arguments.minimum_changed_units, "tokenizer_name": arguments.tokenizer_name,
        "regeneration_enabled": not arguments.no_regeneration, "maximum_regenerations": arguments.maximum_regenerations,
        "size_match_tolerance": arguments.size_match_tolerance, "maximum_cost_dollars": arguments.maximum_cost_dollars,
        "cost_estimate_and_ceiling": estimate, "worker_count": arguments.workers, "resumed": is_resume}
    write_run_configuration(run_folder, timestamp, configuration)
    print(f"Run folder: {run_folder}", flush=True)
    if arguments.dry_run:
        run_dry_run(run_folder, timestamp, planned, arguments, target_sizes, definition)
        return run_folder
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=preachiness_rewrites.load_anthropic_api_key(arguments.env_path), max_retries=3)
    tracker = run_rewrites(client, run_folder, timestamp, templates, tasks, arguments, request_function, tokenize_function, definition)
    if not legacy:
        # the chained arm, from the finished trait_only_away rewrites of this and earlier sessions
        finished = read_all_records(run_folder, arms_module.rewrites_file_pattern(definition))
        chained_tasks, skipped = build_chained_arm_tasks(definition, source_ids, finished, arguments.target_changed_units)
        failures_path = Path(run_folder) / f"rewrite_failures_{timestamp}.jsonl"
        for arm_name, source_id in skipped:
            AppendOnlyJsonlWriter().append(failures_path, {"arm": arm_name, "original_source_id": source_id,
                                                           "failure_kind": SKIPPED_NO_SOURCE_REWRITE})
        if skipped:
            print(f"{len(skipped)} chained rewrites skipped for lack of a finished trait_only_away rewrite: {skipped}", flush=True)
        run_rewrites(client, run_folder, timestamp, templates, chained_tasks, arguments, request_function, tokenize_function, definition,
                     tracker)
    write_export(run_folder, timestamp, definition)
    return run_folder


if __name__ == "__main__":
    main()
