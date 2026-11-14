"""Depiction labels for the persona vector projection experiment (PAID when run without --dry-run).

For every corpus document and each of the four facets, GPT-5.6 Luna says whether the AI that the document depicts
shows the trait, shows avoiding it (the facet's negative pole), or neither. One call per document covers all four
facets (strict JSON schema), and also lists the kinds of animal the document discusses (used to prefer
multi-animal documents for the consistent positive control). Only the document text and the instruction file go to
the model, never the spec variables.

Files in the run folder (names end with the session's date and time):
  run_configuration_<timestamp>.json
  depiction_labels_<timestamp>.jsonl          one record per labelled document (read back on resume)
  depiction_label_failures_<timestamp>.jsonl  parse failures after one retry (retried on resume)
  usage_and_cost_log_<timestamp>.json, progress_log_<timestamp>.txt
  depiction_label_hand_check_sample_<timestamp>.md and ..._sheet_<timestamp>.csv   40 documents stratified by label

Examples:
    python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py --dry-run
    python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py          (PAID)
    python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py --resume-folder <folder>
    python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py \
        --export-hand-check-sample-only --resume-folder <folder>
"""

import argparse
import concurrent.futures
import csv
import json
import random
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from shared.api_usage_and_cost_tracking import (PRICES_PER_MILLION_TOKENS_BY_MODEL, UsageAndCostTracker,  # noqa: E402
                                                compute_cost_in_dollars)
from shared.openai_structured_output_client import (StructuredOutputParseFailure, make_client,  # noqa: E402
                                                    request_structured_output)
from shared.resumable_jsonl_records import (AppendOnlyJsonlWriter, find_completed_keys, read_all_records,  # noqa: E402
                                            read_jsonl)
from shared.run_folder_management import (check_resume_is_compatible, choose_run_folder, sha256_of_file,  # noqa: E402
                                          timestamp_now, write_run_configuration)
from shared.strict_json_schema_helpers import (assert_schema_is_strict_valid, build_array_schema,  # noqa: E402
                                               build_strict_object_schema)
from shared.timed_progress_reporter import TimedProgressReporter  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
PROMPT_FILE_PATH = Path(__file__).resolve().parent / "prompts" / "depiction_label_instructions_draft.txt"
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sdf_corpus.jsonl"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
FULL_RUNS_DIRECTORY = defs.RUNS_ROOT
DRY_RUNS_DIRECTORY = defs.SMALL_SAMPLE_CHECKS_ROOT

MODEL_NAME = defs.JUDGE_MODEL_NAME  # "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "medium"
SCHEMA_NAME = "depiction_labels_for_four_facets"
LABEL_SHOWS_TRAIT = "shows_trait"
LABEL_SHOWS_AVOIDING_TRAIT = "shows_avoiding_trait"
LABEL_NEITHER = "neither"
LABEL_VALUES = [LABEL_SHOWS_TRAIT, LABEL_SHOWS_AVOIDING_TRAIT, LABEL_NEITHER]
LABELLED_FACETS = list(defs.PERSONA_FACETS)
MAXIMUM_KINDS_OF_ANIMALS = 8
LABELS_FILE_PATTERN = "depiction_labels_*.jsonl"

HAND_CHECK_SAMPLE_SIZE = 40
DEFAULT_MAXIMUM_CHARACTERS_PER_DOCUMENT_IN_HAND_CHECK_FILE = 15000

# Dry-run assumptions
CHARACTERS_PER_TOKEN_LATIN = 4.0
CHARACTERS_PER_TOKEN_OTHER = 1.5
LATIN_SCRIPT_LANGUAGES = {"English", "Spanish", "German", "Portuguese", "French", "Polish", "Indonesian",
                          "Vietnamese", "Norwegian"}
EXPECTED_OUTPUT_TOKENS_PER_CALL = 700
EXPECTED_REASONING_TOKENS_PER_CALL = 1500
CEILING_OUTPUT_TOKENS_PER_ATTEMPT = 1500
CEILING_REASONING_TOKENS_PER_ATTEMPT = 8000
MAXIMUM_ATTEMPTS_PER_DOCUMENT = 2  # one parse retry in request_structured_output


# ----------------------------------------------------------------------------
# Schema and parsing
# ----------------------------------------------------------------------------

def build_label_schema():
    """Per facet: one sentence of evidence first, then the label (generation order is dict order)."""
    properties = {}
    for facet in LABELLED_FACETS:
        properties[facet] = build_strict_object_schema({
            "evidence": {"type": "string"},
            "label": {"type": "string", "enum": LABEL_VALUES}})
    properties["kinds_of_animals_discussed"] = build_array_schema({"type": "string"},
                                                                  maximum_items=MAXIMUM_KINDS_OF_ANIMALS)
    schema = build_strict_object_schema(properties)
    assert_schema_is_strict_valid(schema)
    return schema


def validate_parsed_labels(parsed_result):
    """Raises ValueError unless every facet has a label from LABEL_VALUES and the kinds list is a list of strings."""
    if not isinstance(parsed_result, dict):
        raise ValueError("result is not an object")
    for facet in LABELLED_FACETS:
        entry = parsed_result.get(facet)
        if not isinstance(entry, dict):
            raise ValueError(f"missing facet {facet}")
        if entry.get("label") not in LABEL_VALUES:
            raise ValueError(f"facet {facet} has label {entry.get('label')!r}")
        if not isinstance(entry.get("evidence"), str):
            raise ValueError(f"facet {facet} has no evidence text")
    kinds = parsed_result.get("kinds_of_animals_discussed")
    if not isinstance(kinds, list) or not all(isinstance(kind, str) for kind in kinds):
        raise ValueError("kinds_of_animals_discussed is not a list of strings")


def build_label_record(document, parsed_result, usage, parse_attempts, latency_seconds, reasoning_effort):
    """The stored record. The document id comes from the code, never from the model."""
    kinds = []
    for kind in parsed_result["kinds_of_animals_discussed"]:
        cleaned = " ".join(kind.split()).strip()
        if cleaned and cleaned.lower() not in {existing.lower() for existing in kinds}:
            kinds.append(cleaned)
    return {
        "document_id": document["doc_id"], "language": document["language"], "model": MODEL_NAME,
        "reasoning_effort": reasoning_effort,
        "labels": {facet: parsed_result[facet]["label"] for facet in LABELLED_FACETS},
        "evidence": {facet: parsed_result[facet]["evidence"] for facet in LABELLED_FACETS},
        "kinds_of_animals_discussed": kinds, "number_of_kinds_of_animals_discussed": len(kinds),
        "usage": usage, "parse_attempts": parse_attempts, "latency_seconds": latency_seconds}


def label_one_document(client, instructions_text, schema, document, reasoning_effort):
    """Returns (record, failure_record)."""
    try:
        result = request_structured_output(
            client, MODEL_NAME, instructions_text, document["content"], SCHEMA_NAME, schema, reasoning_effort,
            validate_parsed_result=validate_parsed_labels, prompt_cache_key=SCHEMA_NAME)
    except StructuredOutputParseFailure as failure:
        return None, {"document_id": document["doc_id"], "error_kind": "parse_or_schema_failure",
                      "description": failure.description, "raw_output_text": failure.raw_output_text,
                      "usage": failure.usage, "latency_seconds": failure.latency_seconds}
    return build_label_record(document, result["parsed_result"], result["usage"], result["parse_attempts"],
                              result["latency_seconds"], reasoning_effort), None


# ----------------------------------------------------------------------------
# Hand-check sample
# ----------------------------------------------------------------------------

def choose_stratified_sample(label_records, sample_size, seed):
    """Round-robin over the strata (facet, label), always taking a document from a language that has been picked
    least so far, so that each label of each facet and a spread of languages are in the sample. Deterministic."""
    random_generator = random.Random(seed)
    strata = {}
    for record in sorted(label_records, key=lambda record: record["document_id"]):
        for facet in LABELLED_FACETS:
            strata.setdefault((facet, record["labels"][facet]), []).append(record)
    for pool in strata.values():
        random_generator.shuffle(pool)
    chosen, chosen_ids, language_counts = [], set(), {}
    stratum_order = [(facet, label) for facet in LABELLED_FACETS for label in LABEL_VALUES if (facet, label) in strata]
    while len(chosen) < min(sample_size, len(label_records)) and any(strata[key] for key in stratum_order):
        for key in stratum_order:
            pool = [record for record in strata[key] if record["document_id"] not in chosen_ids]
            strata[key] = pool
            if not pool or len(chosen) >= sample_size:
                continue
            pick = min(pool, key=lambda record: language_counts.get(record["language"], 0))  # first minimal: shuffled
            chosen.append(pick)
            chosen_ids.add(pick["document_id"])
            language_counts[pick["language"]] = language_counts.get(pick["language"], 0) + 1
    return chosen


def write_hand_check_files(run_folder, timestamp, label_records, corpus_by_id, sample_size=HAND_CHECK_SAMPLE_SIZE,
                           seed=0, maximum_characters=DEFAULT_MAXIMUM_CHARACTERS_PER_DOCUMENT_IN_HAND_CHECK_FILE):
    sample = choose_stratified_sample(label_records, sample_size, seed)
    markdown_path = Path(run_folder) / f"depiction_label_hand_check_sample_{timestamp}.md"
    sheet_path = Path(run_folder) / f"depiction_label_hand_check_sheet_{timestamp}.csv"
    lines = [f"# Depiction labels: hand-check sample of {len(sample)} documents", "",
             "For each document, read the text and decide for each facet whether the AI depicted shows the trait, "
             "shows avoiding it, or neither. Then compare with the model's label. Fill in `hand_label` in "
             f"`{sheet_path.name}`.", "",
             "Labels: shows_trait, shows_avoiding_trait, neither.", ""]
    for number, record in enumerate(sample, start=1):
        document = corpus_by_id[record["document_id"]]
        text = document["content"]
        truncated = len(text) > maximum_characters
        lines += [f"## {number}. {record['document_id']} ({record['language']})", "",
                  "| Facet | Model label | Model evidence |", "|---|---|---|"]
        for facet in LABELLED_FACETS:
            evidence = record["evidence"][facet].replace("|", "/").replace("\n", " ")
            lines.append(f"| {defs.FACET_DISPLAY_NAMES[facet]} | {record['labels'][facet]} | {evidence} |")
        lines += ["", f"Kinds of animal listed by the model: {', '.join(record['kinds_of_animals_discussed']) or 'none'}",
                  "", "Document text" + (f" (first {maximum_characters} of {len(text)} characters)" if truncated else "")
                  + ":", "", "```", text[:maximum_characters], "```", ""]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    with open(sheet_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["document_id", "language", "facet", "model_label", "hand_label", "notes"])
        for record in sample:
            for facet in LABELLED_FACETS:
                writer.writerow([record["document_id"], record["language"], facet, record["labels"][facet], "", ""])
    print(f"Hand-check sample of {len(sample)} documents: {markdown_path} and {sheet_path}", flush=True)
    return markdown_path, sheet_path, sample


# ----------------------------------------------------------------------------
# Cost estimate and ceiling
# ----------------------------------------------------------------------------

def estimate_input_tokens(text, language):
    return len(text) / (CHARACTERS_PER_TOKEN_LATIN if language in LATIN_SCRIPT_LANGUAGES else CHARACTERS_PER_TOKEN_OTHER)


def estimate_cost_and_ceiling(documents, instructions_text):
    """Expected: one attempt, 700 output + 1,500 reasoning tokens. Ceiling: two attempts per document with
    1,500 output + 8,000 reasoning tokens each; no cache hits assumed at the ceiling (the instructions would be
    cached in practice)."""
    instruction_tokens = estimate_input_tokens(instructions_text, "English")
    input_tokens = sum(instruction_tokens + estimate_input_tokens(document["content"], document["language"])
                       for document in documents)
    expected_output = len(documents) * (EXPECTED_OUTPUT_TOKENS_PER_CALL + EXPECTED_REASONING_TOKENS_PER_CALL)
    ceiling_input = input_tokens * MAXIMUM_ATTEMPTS_PER_DOCUMENT
    ceiling_output = len(documents) * MAXIMUM_ATTEMPTS_PER_DOCUMENT * (CEILING_OUTPUT_TOKENS_PER_ATTEMPT
                                                                       + CEILING_REASONING_TOKENS_PER_ATTEMPT)
    return {
        "documents": len(documents), "estimated_input_tokens": round(input_tokens),
        "expected_output_tokens_including_reasoning": round(expected_output),
        "expected_cost_dollars": round(compute_cost_in_dollars(MODEL_NAME, input_tokens, 0, expected_output), 3),
        "ceiling_input_tokens": round(ceiling_input), "ceiling_output_tokens": round(ceiling_output),
        "ceiling_cost_dollars": round(compute_cost_in_dollars(MODEL_NAME, ceiling_input, 0, ceiling_output), 3),
        "assumptions": (f"{MODEL_NAME} prices {PRICES_PER_MILLION_TOKENS_BY_MODEL[MODEL_NAME]} per million; "
                        "input tokens from character counts; expected = one attempt with "
                        f"{EXPECTED_OUTPUT_TOKENS_PER_CALL} output and {EXPECTED_REASONING_TOKENS_PER_CALL} reasoning "
                        f"tokens; ceiling = {MAXIMUM_ATTEMPTS_PER_DOCUMENT} attempts with "
                        f"{CEILING_OUTPUT_TOKENS_PER_ATTEMPT} output and {CEILING_REASONING_TOKENS_PER_ATTEMPT} "
                        "reasoning tokens each and no cache hits")}


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def run_labelling(client, run_folder, timestamp, instructions_text, documents, reasoning_effort, workers,
                  maximum_cost_dollars):
    run_folder = Path(run_folder)
    schema = build_label_schema()
    completed = find_completed_keys(run_folder, LABELS_FILE_PATTERN, ["document_id"])
    tasks = [document for document in documents if (document["doc_id"],) not in completed]
    print(f"{len(tasks)} documents to label ({len(completed)} already done).", flush=True)
    writer, tracker = AppendOnlyJsonlWriter(), UsageAndCostTracker(MODEL_NAME, group_label="per_facet_group")
    labels_path = run_folder / f"depiction_labels_{timestamp}.jsonl"
    failures_path = run_folder / f"depiction_label_failures_{timestamp}.jsonl"
    usage_log_path = run_folder / f"usage_and_cost_log_{timestamp}.json"

    def status_text():
        totals = tracker.overall_totals
        return (f"cost so far ${totals['cost_dollars']:.4f} of cap ${maximum_cost_dollars:.2f}; tokens in "
                f"{totals['input_tokens']} (cached {totals['cached_input_tokens']}) out {totals['output_tokens']} "
                f"(reasoning {totals['reasoning_tokens']}), cache rate {totals['cache_rate']:.2f}")

    reporter = TimedProgressReporter(
        run_folder / f"progress_log_{timestamp}.txt", len(tasks), len(completed), status_suffix_function=status_text,
        after_report_function=lambda: tracker.write_log_file(usage_log_path, "this session only"))

    def work(document):
        if tracker.overall_totals["cost_dollars"] >= maximum_cost_dollars:
            writer.append(failures_path, {"document_id": document["doc_id"], "error_kind": "skipped_cost_cap"})
            reporter.note_task_finished(failed=True)
            return
        record, failure = label_one_document(client, instructions_text, schema, document, reasoning_effort)
        usage = record["usage"] if record else failure["usage"]
        tracker.add_usage("all_documents", usage)
        writer.append(labels_path if record else failures_path, record or failure)
        reporter.note_task_finished(failed=record is None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(work, tasks))
    reporter.finish()
    tracker.write_log_file(usage_log_path, "this session only")
    return tracker


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="depiction_labels_for_four_facets_gpt_5_6_luna")
    parser.add_argument("--resume-folder", default=None)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-cost-dollars", type=float, default=None,
                        help="Stop submitting calls once this much has been spent (default: the dry-run ceiling)")
    parser.add_argument("--maximum-documents", type=int, default=None,
                        help="Label only the first N corpus documents (by doc_id order)")
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--env-path", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--hand-check-sample-size", type=int, default=HAND_CHECK_SAMPLE_SIZE)
    parser.add_argument("--maximum-characters-per-document-in-hand-check-file", type=int,
                        default=DEFAULT_MAXIMUM_CHARACTERS_PER_DOCUMENT_IN_HAND_CHECK_FILE)
    parser.add_argument("--export-hand-check-sample-only", action="store_true",
                        help="Only write the hand-check files from labels already in --resume-folder")
    parser.add_argument("--dry-run", action="store_true", help="Print expected cost and ceiling; no API call")
    return parser.parse_args(argv)


def main(argv=None, client=None, runs_directory_override=None):
    arguments = parse_arguments(argv)
    timestamp = timestamp_now()
    corpus = sorted(read_jsonl(arguments.corpus_path), key=lambda document: document["doc_id"])
    corpus_by_id = {document["doc_id"]: document for document in corpus}
    if arguments.maximum_documents:
        corpus = corpus[:arguments.maximum_documents]
    if arguments.export_hand_check_sample_only:
        if not arguments.resume_folder:
            raise SystemExit("--export-hand-check-sample-only needs --resume-folder")
        records = {record["document_id"]: record
                   for record in read_all_records(arguments.resume_folder, LABELS_FILE_PATTERN)}
        write_hand_check_files(arguments.resume_folder, timestamp, list(records.values()), corpus_by_id,
                               arguments.hand_check_sample_size,
                               maximum_characters=arguments.maximum_characters_per_document_in_hand_check_file)
        return arguments.resume_folder
    runs_directory = runs_directory_override or (DRY_RUNS_DIRECTORY if arguments.dry_run else FULL_RUNS_DIRECTORY)
    run_folder, is_resume = choose_run_folder(runs_directory, arguments.run_name, arguments.resume_folder)
    instructions_text = PROMPT_FILE_PATH.read_text(encoding="utf-8")
    estimate = estimate_cost_and_ceiling(corpus, instructions_text)
    maximum_cost = arguments.maximum_cost_dollars if arguments.maximum_cost_dollars is not None \
        else estimate["ceiling_cost_dollars"]
    configuration = {
        "script": "label_depiction_of_facets_in_corpus_documents.py", "model": MODEL_NAME,
        "reasoning_effort": arguments.reasoning_effort, "temperature": 1,
        "prompt_file_sha256": {PROMPT_FILE_PATH.name: sha256_of_file(PROMPT_FILE_PATH)},
        "corpus_path": str(arguments.corpus_path), "corpus_sha256": sha256_of_file(arguments.corpus_path),
        "number_of_documents": len(corpus), "dry_run": arguments.dry_run, "resumed": is_resume,
        "cost_estimate_and_ceiling": estimate, "maximum_cost_dollars": maximum_cost,
        "prices_per_million_tokens": PRICES_PER_MILLION_TOKENS_BY_MODEL[MODEL_NAME]}
    if is_resume:
        check_resume_is_compatible(run_folder, configuration)
    write_run_configuration(run_folder, timestamp, configuration)
    print(json.dumps(estimate, indent=2))
    print(f"COST CEILING for the paid run: ${estimate['ceiling_cost_dollars']:.3f} "
          f"(expected about ${estimate['expected_cost_dollars']:.3f}). Run folder: {run_folder}", flush=True)
    if arguments.dry_run:
        return run_folder
    if client is None:
        client = make_client(arguments.env_path)
    run_labelling(client, run_folder, timestamp, instructions_text, corpus, arguments.reasoning_effort,
                  arguments.workers, maximum_cost)
    records = {record["document_id"]: record for record in read_all_records(run_folder, LABELS_FILE_PATTERN)}
    write_hand_check_files(run_folder, timestamp, list(records.values()), corpus_by_id, arguments.hand_check_sample_size,
                           maximum_characters=arguments.maximum_characters_per_document_in_hand_check_file)
    return run_folder


if __name__ == "__main__":
    main()
