"""Builds the run folder (documents, conditions, run configuration with file hashes) for the
log-probability persona check. Local and free: no model or API is used.

Usage examples (from the repository root):
    python3 scripts/log_probability_persona_check/build_scoring_inputs.py --run-name full_preachiness_check
    python3 scripts/log_probability_persona_check/build_scoring_inputs.py --run-name pilot_twenty_documents \
        --small-sample-check --document-ids-file path/to/pilot_document_ids.txt \
        --extra-documents-jsonl data/log_probability_persona_check_runs/positive_control_rewrites/<run>/rewritten_documents.jsonl
    python3 scripts/log_probability_persona_check/build_scoring_inputs.py --run-name full_preachiness_check \
        --resume-folder data/log_probability_persona_check_runs/full_preachiness_check_<timestamp>

Documents scored = corpus documents (optionally restricted by --document-ids-file and --maximum-documents,
applied to the corpus only) followed by every record of every --extra-documents-jsonl file (always all of
them). Layout: run_folder_layout_and_script_interfaces.md in this folder.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import log_probability_persona_check_prompt_rendering as prompt_rendering  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import (check_resume_is_compatible, choose_run_folder, sha256_of_file,  # noqa: E402
                                          timestamp_now, write_run_configuration)

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sdf_corpus.jsonl"
RUNS_DIRECTORY = PROJECT_ROOT / "data" / "log_probability_persona_check_runs"
SMALL_SAMPLE_CHECKS_DIRECTORY = RUNS_DIRECTORY / "small_sample_checks"
DEFAULT_MODEL_NAME = "google/gemma-4-12b-it"

EXTRA_DOCUMENT_REQUIRED_FIELDS = ["document_id", "source_document_id", "rewrite_direction", "content",
                                  "language", "variables"]


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_corpus_documents(corpus_path):
    documents = []
    for record in read_jsonl(corpus_path):
        documents.append({
            "document_id": record["doc_id"], "source_document_id": record["doc_id"],
            "rewrite_direction": None, "content": record["content"], "language": record["language"],
            "variables": record["variables"]})
    return documents


def read_document_ids_file(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_extra_documents(path):
    records = read_jsonl(path)
    for record in records:
        missing = [field for field in EXTRA_DOCUMENT_REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Record {record.get('document_id')} in {path} is missing fields {missing}")
        if record["rewrite_direction"] not in ("preachier", "plainer"):
            raise ValueError(f"Record {record['document_id']} in {path} has rewrite_direction "
                             f"{record['rewrite_direction']!r}")
    return [{field: record[field] for field in EXTRA_DOCUMENT_REQUIRED_FIELDS} for record in records]


def select_documents(corpus_documents, document_ids, maximum_documents, extra_documents):
    """Corpus documents (restricted to document_ids if given, in the order of document_ids, then cut to
    maximum_documents), followed by the extra documents. Duplicate ids are an error."""
    if document_ids is not None:
        documents_by_id = {document["document_id"]: document for document in corpus_documents}
        unknown = [document_id for document_id in document_ids if document_id not in documents_by_id]
        if unknown:
            raise ValueError(f"Document ids not in the corpus: {unknown[:10]}")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("Duplicate document ids in the document ids file")
        selected = [documents_by_id[document_id] for document_id in document_ids]
    else:
        selected = list(corpus_documents)
    if maximum_documents is not None:
        selected = selected[:maximum_documents]
    selected = selected + list(extra_documents)
    all_ids = [document["document_id"] for document in selected]
    if len(set(all_ids)) != len(all_ids):
        duplicates = sorted({document_id for document_id in all_ids if all_ids.count(document_id) > 1})
        raise ValueError(f"Duplicate document ids after adding extra documents: {duplicates[:10]}")
    return selected


def build_document_records(selected_documents):
    template_text = prompt_rendering.load_prompt_text(prompt_rendering.GENERATION_PROMPT_TEMPLATE_FILE_NAME)
    in_universe_facts = prompt_rendering.load_prompt_text(prompt_rendering.IN_UNIVERSE_FACTS_FILE_NAME)
    records = []
    for document in selected_documents:
        record = dict(document)
        record["generation_prompt"] = prompt_rendering.render_generation_prompt(
            document["variables"], document["document_id"], template_text=template_text,
            in_universe_facts=in_universe_facts)
        records.append(record)
    return records


def collect_input_file_hashes(corpus_path, document_ids_file, extra_documents_files):
    hashes = {display_path(corpus_path): sha256_of_file(corpus_path)}
    if document_ids_file:
        hashes[display_path(document_ids_file)] = sha256_of_file(document_ids_file)
    for path in extra_documents_files:
        hashes[display_path(path)] = sha256_of_file(path)
    return hashes


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True, help="Descriptive name for this run")
    parser.add_argument("--resume-folder", default=None, help="Existing run folder to resume")
    parser.add_argument("--small-sample-check", action="store_true",
                        help="Put the run in small_sample_checks/ (pilot, smoke test, cost estimates)")
    parser.add_argument("--document-ids-file", default=None,
                        help="Text file of corpus doc_ids, one per line; only these corpus documents are used")
    parser.add_argument("--maximum-documents", type=int, default=None,
                        help="Use at most this many corpus documents (after --document-ids-file)")
    parser.add_argument("--extra-documents-jsonl", action="append", default=[],
                        help="Rewritten documents file from the contract; may be given several times")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="Hugging Face model whose chat template and tokenizer define the prompts")
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    runs_parent_directory = SMALL_SAMPLE_CHECKS_DIRECTORY if arguments.small_sample_check else RUNS_DIRECTORY
    run_folder, is_resume = choose_run_folder(runs_parent_directory, arguments.run_name, arguments.resume_folder)
    timestamp = timestamp_now()

    corpus_path = Path(arguments.corpus_path)
    document_ids = read_document_ids_file(arguments.document_ids_file) if arguments.document_ids_file else None
    extra_documents = []
    for path in arguments.extra_documents_jsonl:
        extra_documents.extend(load_extra_documents(path))
    selected_documents = select_documents(load_corpus_documents(corpus_path), document_ids,
                                          arguments.maximum_documents, extra_documents)
    document_records = build_document_records(selected_documents)
    condition_records = prompt_rendering.build_condition_records()

    prompt_file_hashes = {name: sha256_of_file(prompt_rendering.PROMPT_DIRECTORY / name)
                          for name in prompt_rendering.PROMPT_FILE_NAMES_USED_FOR_SCORING}
    input_file_hashes = collect_input_file_hashes(corpus_path, arguments.document_ids_file,
                                                  arguments.extra_documents_jsonl)
    configuration = {
        "run_name": arguments.run_name, "created_timestamp": timestamp, "is_small_sample_check":
            arguments.small_sample_check, "resumed_from_existing_folder": is_resume,
        "model_name": arguments.model_name,
        "document_selection": {
            "corpus_path": display_path(corpus_path), "document_ids_file": arguments.document_ids_file,
            "maximum_documents": arguments.maximum_documents,
            "extra_documents_files": arguments.extra_documents_jsonl,
            "number_of_corpus_documents": len(selected_documents) - len(extra_documents),
            "number_of_extra_documents": len(extra_documents),
            "document_ids": [record["document_id"] for record in document_records]},
        "number_of_conditions": len(condition_records),
        "prompt_file_sha256": prompt_file_hashes, "input_file_sha256": input_file_hashes}

    scoring_inputs_directory = run_folder / "scoring_inputs"
    documents_path = scoring_inputs_directory / "documents.jsonl"
    conditions_path = scoring_inputs_directory / "conditions.jsonl"

    if is_resume:
        check_resume_is_compatible(run_folder, configuration, hashes_key="prompt_file_sha256")
        check_resume_is_compatible(run_folder, configuration, hashes_key="input_file_sha256")
        previous_model_names = {json.loads(path.read_text(encoding="utf-8"))["model_name"]
                                for path in run_folder.glob("run_configuration_*.json")}
        if previous_model_names != {arguments.model_name}:
            raise RuntimeError(f"Refusing to resume: --model-name {arguments.model_name} differs from the run's "
                               f"recorded model {sorted(previous_model_names)}")
        if read_jsonl(documents_path) != document_records or read_jsonl(conditions_path) != condition_records:
            raise RuntimeError("Refusing to resume: rebuilt documents or conditions differ from the files "
                               f"already in {scoring_inputs_directory} (different selection arguments?)")
        print(f"Resuming {run_folder}: scoring inputs unchanged ({len(document_records)} documents, "
              f"{len(condition_records)} conditions)")
    else:
        scoring_inputs_directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(documents_path, document_records)
        write_jsonl(conditions_path, condition_records)
        print(f"Wrote {len(document_records)} documents and {len(condition_records)} conditions to "
              f"{scoring_inputs_directory}")
    configuration["scoring_input_file_sha256"] = {"documents.jsonl": sha256_of_file(documents_path),
                                                  "conditions.jsonl": sha256_of_file(conditions_path)}
    while (run_folder / f"run_configuration_{timestamp}.json").exists():
        timestamp += "_again"  # never overwrite an earlier configuration, for example on a resume within a minute
    configuration_path = write_run_configuration(run_folder, timestamp, configuration)
    print(f"Run configuration: {configuration_path}\nRun folder: {run_folder}")
    return run_folder


if __name__ == "__main__":
    main()
