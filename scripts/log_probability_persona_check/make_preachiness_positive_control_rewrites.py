"""Positive control for the log-probability preachiness check: rewrite about 10 corpus documents twice
with Claude Opus 5, once preachier and once plainer, keeping everything else the same.

Output (contract: run_folder_layout_and_script_interfaces.md, "Rewritten documents file"):
    data/log_probability_persona_check_runs/positive_control_rewrites/<run_name>_<timestamp>/rewritten_documents.jsonl

Resumable: finished (source_document_id, rewrite_direction) pairs are read back from the run folder and skipped.
Failures (refusal without output, max_tokens, empty output, transport errors) go to rewrite_failures_*.jsonl and
are retried on resume. The API key is read ONLY from the repository's .env file, never from the OS environment.
Dry runs (no API call) write to data/log_probability_persona_check_runs/small_sample_checks/.

Examples:
    python3 scripts/log_probability_persona_check/make_preachiness_positive_control_rewrites.py --dry-run
    python3 scripts/log_probability_persona_check/make_preachiness_positive_control_rewrites.py   (calls the paid API)
    python3 scripts/log_probability_persona_check/make_preachiness_positive_control_rewrites.py \
        --resume-folder data/log_probability_persona_check_runs/positive_control_rewrites/<folder>
"""

import argparse
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from shared.anthropic_usage_and_cost_tracking import (AnthropicUsageAndCostTracker,  # noqa: E402
                                                      PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL,
                                                      compute_anthropic_cost_in_dollars,
                                                      extract_attempts_from_response, summarize_attempts)
from shared.resumable_jsonl_records import AppendOnlyJsonlWriter, find_completed_keys, read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import (check_resume_is_compatible, choose_run_folder, sha256_of_file,  # noqa: E402
                                          timestamp_now, write_run_configuration)
from shared.timed_progress_reporter import TimedProgressReporter  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
PROMPT_FILE_PATH = (Path(__file__).resolve().parent / "prompts"
                    / "preachiness_positive_control_rewrite_instructions.txt")
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sdf_corpus.jsonl"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
RUNS_PARENT_DIRECTORY = PROJECT_ROOT / "data" / "log_probability_persona_check_runs"
FULL_RUNS_DIRECTORY = RUNS_PARENT_DIRECTORY / "positive_control_rewrites"
DRY_RUNS_DIRECTORY = RUNS_PARENT_DIRECTORY / "small_sample_checks"  # cost estimates go here (testing skill)

REWRITE_MODEL_REQUESTED = "claude-opus-5"
MAXIMUM_OUTPUT_TOKENS = 32000
FALLBACK_BETA_HEADER = "server-side-fallback-2026-07-01"
REWRITE_DIRECTIONS = ["preachier", "plainer"]
REWRITTEN_DOCUMENTS_FILE_NAME = "rewritten_documents.jsonl"
API_KEY_NAME = "ANTHROPIC_API_KEY"

MINOR_DETAIL_CENTRALITY_PREFIX = "a minor detail mentioned only in passing"
NO_WELFARE_RESOLUTION_PHRASE = "correctly raises no welfare consideration at all"
SKEPTICAL_TONE = "skeptical or displeased"
MINIMUM_SKEPTICAL_DOCUMENTS = 3
MAXIMUM_LENGTH_RATIO_DEVIATION = 0.15
LANGUAGES_COUNTED_IN_CHARACTERS = {"Mandarin Chinese", "Japanese"}  # no spaces between words
LANGUAGES_IN_LATIN_SCRIPT = {"English", "Spanish", "German", "Portuguese", "French", "Polish", "Indonesian",
                             "Vietnamese", "Norwegian"}

# Dry-run estimate assumptions
LATIN_SCRIPT_CHARACTERS_PER_TOKEN = 4.0
OTHER_SCRIPT_CHARACTERS_PER_TOKEN = 1.5
ASSUMED_THINKING_TOKENS_PER_CALL = 8000


# ----------------------------------------------------------------------------
# Document selection
# ----------------------------------------------------------------------------

def is_eligible_for_rewriting(document):
    variables = document["variables"]
    return (not variables["centrality"].startswith(MINOR_DETAIL_CENTRALITY_PREFIX)
            and NO_WELFARE_RESOLUTION_PHRASE not in variables["resolution"])


def allocate_documents_per_tone(number_of_documents, eligible_counts_by_tone):
    """Roughly proportional (largest remainder) with at least MINIMUM_SKEPTICAL_DOCUMENTS skeptical."""
    total = sum(eligible_counts_by_tone.values())
    exact = {tone: number_of_documents * count / total for tone, count in eligible_counts_by_tone.items()}
    allocation = {tone: int(value) for tone, value in exact.items()}
    by_remainder = sorted(exact, key=lambda tone: (exact[tone] - allocation[tone], tone), reverse=True)
    for tone in by_remainder[:number_of_documents - sum(allocation.values())]:
        allocation[tone] += 1
    if SKEPTICAL_TONE in allocation:
        minimum = min(MINIMUM_SKEPTICAL_DOCUMENTS, eligible_counts_by_tone[SKEPTICAL_TONE], number_of_documents)
        while allocation[SKEPTICAL_TONE] < minimum:
            donor = max((tone for tone in allocation if tone != SKEPTICAL_TONE), key=lambda tone: allocation[tone])
            allocation[donor] -= 1
            allocation[SKEPTICAL_TONE] += 1
    return allocation


def select_source_documents(corpus, number_of_documents, seed):
    """Deterministic for (corpus, number_of_documents, seed). Returns documents sorted by doc_id."""
    eligible_by_tone = {}
    for document in sorted(corpus, key=lambda document: document["doc_id"]):
        if is_eligible_for_rewriting(document):
            eligible_by_tone.setdefault(document["variables"]["tone"], []).append(document)
    allocation = allocate_documents_per_tone(
        number_of_documents, {tone: len(documents) for tone, documents in eligible_by_tone.items()})
    random_generator = random.Random(seed)
    selected = []
    for tone in sorted(eligible_by_tone):
        selected.extend(random_generator.sample(eligible_by_tone[tone], allocation[tone]))
    return sorted(selected, key=lambda document: document["doc_id"])


def read_document_ids_file(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


# ----------------------------------------------------------------------------
# Prompt, lengths, key
# ----------------------------------------------------------------------------

def render_rewrite_prompt(instruction_template, rewrite_direction, document_text):
    """Fills {direction} (PREACHIER / PLAINER) then {document_text}; the order keeps braces inside the
    document text from being interpreted as slots."""
    return (instruction_template.replace("{direction}", rewrite_direction.upper())
            .replace("{document_text}", document_text))


def length_in_unit(text, language):
    """(length, unit): non-whitespace characters for languages without spaces between words, else words."""
    if language in LANGUAGES_COUNTED_IN_CHARACTERS:
        return len("".join(text.split())), "characters"
    return len(text.split()), "words"


def load_anthropic_api_key(env_path):
    """Reads ANTHROPIC_API_KEY from the given .env file only. Never consults os.environ."""
    from dotenv import dotenv_values
    if not Path(env_path).is_file():
        raise SystemExit(f"No .env file at {env_path}; cannot find {API_KEY_NAME}.")
    key = dotenv_values(env_path).get(API_KEY_NAME)
    if not key:
        raise SystemExit(f"{API_KEY_NAME} is not set in {env_path}.")
    return key


# ----------------------------------------------------------------------------
# One rewrite
# ----------------------------------------------------------------------------

def build_request_arguments(prompt_text):
    """Arguments for client.beta.messages.stream. No sampling parameters (the API rejects them); thinking is
    left unset (adaptive by default on Opus 5); server-side refusal fallback on."""
    return {"model": REWRITE_MODEL_REQUESTED, "max_tokens": MAXIMUM_OUTPUT_TOKENS,
            "betas": [FALLBACK_BETA_HEADER], "fallbacks": "default",
            "messages": [{"role": "user", "content": prompt_text}]}


def request_rewrite(client, prompt_text):
    with client.beta.messages.stream(**build_request_arguments(prompt_text)) as stream:
        return stream.get_final_message()


def read_rewrite_text(response):
    """(text, failure_kind). stop_reason is checked before content is read."""
    if response.stop_reason == "refusal":
        return None, "refusal"
    if response.stop_reason == "max_tokens":
        return None, "max_tokens"
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
    if not text:
        return None, "empty_output"
    return text, None


def make_one_rewrite(client, instruction_template, source_document, rewrite_direction):
    """Returns (record, failure_record, call_summary). Exactly one of record / failure_record is not None."""
    source_document_id = source_document["doc_id"]
    prompt_text = render_rewrite_prompt(instruction_template, rewrite_direction, source_document["content"])
    started = time.time()
    try:
        response = request_rewrite(client, prompt_text)
    except Exception as error:  # noqa: BLE001 - transport failure after the SDK's own retries
        return None, {"source_document_id": source_document_id, "rewrite_direction": rewrite_direction,
                      "failure_kind": "transport_failure", "description": repr(error)}, None
    call_summary = summarize_attempts(extract_attempts_from_response(response))
    call_summary["latency_seconds"] = round(time.time() - started, 2)
    call_summary["stop_reason"] = response.stop_reason
    call_summary["model_used"] = response.model
    text, failure_kind = read_rewrite_text(response)
    if failure_kind:
        category = getattr(getattr(response, "stop_details", None), "category", None)
        return None, {"source_document_id": source_document_id, "rewrite_direction": rewrite_direction,
                      "failure_kind": failure_kind, "stop_reason": response.stop_reason,
                      "refusal_category": category, "model_used": response.model}, call_summary
    language = source_document["language"]
    source_length, length_unit = length_in_unit(source_document["content"], language)
    rewrite_length, _ = length_in_unit(text, language)
    length_ratio = rewrite_length / source_length if source_length else None
    record = {
        "document_id": f"{source_document_id}__{rewrite_direction}",
        "source_document_id": source_document_id,
        "rewrite_direction": rewrite_direction,
        "content": text,
        "language": language,
        "variables": source_document["variables"],
        "rewrite_model_requested": REWRITE_MODEL_REQUESTED,
        "rewrite_model_used": response.model,
        "source_word_count": len(source_document["content"].split()),
        "rewrite_word_count": len(text.split()),
        "length_unit": length_unit,
        "source_length": source_length,
        "rewrite_length": rewrite_length,
        "length_ratio": length_ratio,
        "length_within_fifteen_percent": (length_ratio is not None
                                          and abs(length_ratio - 1) <= MAXIMUM_LENGTH_RATIO_DEVIATION),
    }
    return record, None, call_summary


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def find_completed_pairs(run_folder):
    return find_completed_keys(run_folder, REWRITTEN_DOCUMENTS_FILE_NAME, ["source_document_id", "rewrite_direction"])


def build_tasks(selected_documents):
    return [(document, direction) for document in selected_documents for direction in REWRITE_DIRECTIONS]


def run_rewrites(client, run_folder, timestamp, instruction_template, selected_documents, worker_count):
    run_folder = Path(run_folder)
    completed_pairs = find_completed_pairs(run_folder)
    tasks = [(document, direction) for document, direction in build_tasks(selected_documents)
             if (document["doc_id"], direction) not in completed_pairs]
    print(f"{len(tasks)} rewrites to make ({len(completed_pairs)} already done).", flush=True)
    writer = AppendOnlyJsonlWriter()
    tracker = AnthropicUsageAndCostTracker()
    rewritten_path = run_folder / REWRITTEN_DOCUMENTS_FILE_NAME
    failures_path = run_folder / f"rewrite_failures_{timestamp}.jsonl"
    per_call_usage_path = run_folder / f"per_call_usage_and_cost_{timestamp}.jsonl"
    usage_log_path = run_folder / f"usage_and_cost_log_{timestamp}.json"

    def cost_status_text():
        totals = tracker.overall_totals
        return (f"cost so far ${totals['cost_dollars_of_priced_attempts']:.4f} "
                f"({totals['calls_with_unknown_cost']} calls with unknown cost); "
                f"tokens in {totals['input_tokens']} out {totals['output_tokens']} "
                f"cache read {totals['cache_read_tokens']} write {totals['cache_write_tokens']}")

    reporter = TimedProgressReporter(
        run_folder / f"progress_log_{timestamp}.txt", len(tasks), len(completed_pairs),
        status_suffix_function=cost_status_text,
        after_report_function=lambda: tracker.write_log_file(usage_log_path, "this session only"))

    def work(task):
        document, direction = task
        record, failure_record, call_summary = make_one_rewrite(client, instruction_template, document, direction)
        if call_summary is not None:
            tracker.add_call_summary(direction, call_summary)
            writer.append(per_call_usage_path, {"source_document_id": document["doc_id"],
                                                "rewrite_direction": direction, **call_summary})
        if record:
            writer.append(rewritten_path, record)
        else:
            writer.append(failures_path, failure_record)
        reporter.note_task_finished(failed=record is None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(work, tasks))
    reporter.finish()
    tracker.write_log_file(usage_log_path, "this session only")
    return reporter, tracker


# ----------------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------------

def estimate_tokens_from_characters(text, language):
    characters_per_token = (LATIN_SCRIPT_CHARACTERS_PER_TOKEN if language in LANGUAGES_IN_LATIN_SCRIPT
                            else OTHER_SCRIPT_CHARACTERS_PER_TOKEN)
    return len(text) / characters_per_token


def estimate_cost(rendered_requests):
    """Input tokens from character counts; output = the document's own token estimate plus thinking tokens."""
    total_input = total_output = 0.0
    for rendered in rendered_requests:
        total_input += estimate_tokens_from_characters(rendered["prompt_text"], rendered["language"])
        total_output += (estimate_tokens_from_characters(rendered["source_content"], rendered["language"])
                         + ASSUMED_THINKING_TOKENS_PER_CALL)
    cost = compute_anthropic_cost_in_dollars(REWRITE_MODEL_REQUESTED, total_input, total_output)
    return {"calls": len(rendered_requests), "estimated_input_tokens": round(total_input),
            "estimated_output_tokens": round(total_output), "estimated_cost_dollars": round(cost, 2),
            "assumptions": (f"input tokens = characters / {LATIN_SCRIPT_CHARACTERS_PER_TOKEN} for Latin-script "
                            f"languages and / {OTHER_SCRIPT_CHARACTERS_PER_TOKEN} for others; output tokens = "
                            f"the source document's token estimate plus {ASSUMED_THINKING_TOKENS_PER_CALL} "
                            f"thinking tokens per call; claude-opus-5 prices "
                            f"{PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL[REWRITE_MODEL_REQUESTED]} per million; "
                            "no cache hits; no refusal fallbacks")}


def run_dry_run(run_folder, timestamp, instruction_template, selected_documents):
    rendered_requests = []
    for document, direction in build_tasks(selected_documents):
        prompt_text = render_rewrite_prompt(instruction_template, direction, document["content"])
        rendered_requests.append({
            "source_document_id": document["doc_id"], "rewrite_direction": direction,
            "language": document["language"], "source_content": document["content"], "prompt_text": prompt_text,
            "request_arguments": build_request_arguments(prompt_text)})
    write_jsonl(Path(run_folder) / f"rendered_requests_{timestamp}.jsonl", rendered_requests)
    estimate = estimate_cost(rendered_requests)
    (Path(run_folder) / f"dry_run_cost_estimate_{timestamp}.json").write_text(
        json.dumps(estimate, indent=2), encoding="utf-8")
    print(json.dumps(estimate, indent=2))
    return estimate


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="preachiness_positive_control_rewrites")
    parser.add_argument("--resume-folder", default=None)
    parser.add_argument("--number-of-documents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--document-ids-file", default=None,
                        help="Text file with one doc_id per line; overrides the stratified selection")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--env-path", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--dry-run", action="store_true",
                        help="Render every request, write them to the folder, print a cost estimate; no API call")
    return parser.parse_args(argv)


def choose_documents(arguments, corpus):
    if arguments.document_ids_file:
        wanted_ids = read_document_ids_file(arguments.document_ids_file)
        by_id = {document["doc_id"]: document for document in corpus}
        unknown = [doc_id for doc_id in wanted_ids if doc_id not in by_id]
        if unknown:
            raise SystemExit(f"Unknown document ids in {arguments.document_ids_file}: {unknown}")
        return [by_id[doc_id] for doc_id in wanted_ids], "document_ids_file"
    return select_source_documents(corpus, arguments.number_of_documents, arguments.seed), "stratified_by_tone"


def main(argv=None, client=None, runs_directory_override=None):
    arguments = parse_arguments(argv)
    corpus = read_jsonl(arguments.corpus_path)
    runs_directory = runs_directory_override or (DRY_RUNS_DIRECTORY if arguments.dry_run else FULL_RUNS_DIRECTORY)
    run_folder, is_resume = choose_run_folder(runs_directory, arguments.run_name, arguments.resume_folder)
    timestamp = timestamp_now()
    instruction_template = PROMPT_FILE_PATH.read_text(encoding="utf-8")
    configuration = {
        "script": "make_preachiness_positive_control_rewrites.py",
        "model_requested": REWRITE_MODEL_REQUESTED, "max_tokens": MAXIMUM_OUTPUT_TOKENS,
        "sampling_parameters": "unset", "thinking": "unset (adaptive by default)",
        "fallbacks": "default", "beta": FALLBACK_BETA_HEADER,
        "prompt_file_sha256": {PROMPT_FILE_PATH.name: sha256_of_file(PROMPT_FILE_PATH)},
        "corpus_path": str(arguments.corpus_path), "corpus_sha256": sha256_of_file(arguments.corpus_path),
        "prices_per_million_tokens": PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL, "dry_run": arguments.dry_run,
    }
    by_id = {document["doc_id"]: document for document in corpus}
    if is_resume:
        check_resume_is_compatible(run_folder, configuration)
        first_configuration = json.loads(sorted(Path(run_folder).glob("run_configuration_*.json"))[0]
                                         .read_text(encoding="utf-8"))
        selected_documents = [by_id[doc_id] for doc_id in first_configuration["selected_document_ids"]]
        selection_method, seed = first_configuration["selection_method"], first_configuration["seed"]
    else:
        selected_documents, selection_method = choose_documents(arguments, corpus)
        seed = arguments.seed
    configuration.update({
        "selection_method": selection_method, "seed": seed,
        "number_of_documents_requested": arguments.number_of_documents,
        "selected_document_ids": [document["doc_id"] for document in selected_documents],
        "selected_document_tones": {document["doc_id"]: document["variables"]["tone"]
                                    for document in selected_documents},
        "worker_count": arguments.workers, "resumed": is_resume})
    write_run_configuration(run_folder, timestamp, configuration)
    print(f"Run folder: {run_folder}; selected {len(selected_documents)} documents: "
          f"{configuration['selected_document_ids']}", flush=True)

    if arguments.dry_run:
        run_dry_run(run_folder, timestamp, instruction_template, selected_documents)
        return run_folder
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=load_anthropic_api_key(arguments.env_path), max_retries=3)
    run_rewrites(client, run_folder, timestamp, instruction_template, selected_documents, arguments.workers)
    return run_folder


if __name__ == "__main__":
    main()
