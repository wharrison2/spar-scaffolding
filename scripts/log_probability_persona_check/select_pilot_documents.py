"""Chooses the corpus documents the pilot scoring run scores: every source document of the positive-control
rewrites (so each rewrite has its source's score to be compared with), plus additional corpus documents chosen so
that the whole set is spread across the tone values and, within each tone, across as many languages as possible.
Local and free: no model or API is used.

Sources are read from the rewrites run folder's rewritten_documents.jsonl AND from its run configuration's
selected_document_ids, so every source is included even if some rewrites failed.

Selection of the additional documents is greedy and deterministic for a seed: repeatedly take the tone with the
fewest documents so far (ties broken alphabetically), then the language with the fewest documents so far within
that tone (ties broken by the seeded random generator), then a random remaining document of that tone and language.

Usage (from the repository root):
    python3 scripts/log_probability_persona_check/select_pilot_documents.py \
        --positive-control-rewrites-folder data/log_probability_persona_check_runs/positive_control_rewrites/<run> \
        --number-of-documents 20 --seed 0
Output: data/log_probability_persona_check_runs/small_sample_checks/pilot_document_selection_<timestamp>/
    pilot_document_ids_<timestamp>.txt          one doc_id per line, for build_scoring_inputs.py --document-ids-file
    pilot_document_selection_<timestamp>.json   sources, additional documents, seed, tone and language counts
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sdf_corpus.jsonl"
SMALL_SAMPLE_CHECKS_DIRECTORY = PROJECT_ROOT / "data" / "log_probability_persona_check_runs" / "small_sample_checks"
REWRITTEN_DOCUMENTS_FILE_NAME = "rewritten_documents.jsonl"


def read_rewrite_source_document_ids(rewrites_folder):
    """Source ids from the rewritten documents (if that file exists) and from the run configuration's selection."""
    rewrites_folder = Path(rewrites_folder)
    source_ids = set()
    rewritten_path = rewrites_folder / REWRITTEN_DOCUMENTS_FILE_NAME
    if rewritten_path.exists():
        source_ids.update(record["source_document_id"] for record in read_jsonl(rewritten_path))
    for configuration_path in sorted(rewrites_folder.glob("run_configuration_*.json")):
        source_ids.update(json.loads(configuration_path.read_text(encoding="utf-8")).get("selected_document_ids", []))
    if not source_ids:
        raise SystemExit(f"No source documents found in {rewrites_folder}: no {REWRITTEN_DOCUMENTS_FILE_NAME} "
                         f"records and no run configuration with selected_document_ids")
    return sorted(source_ids)


def choose_additional_documents(corpus, source_ids, number_of_documents, seed):
    """Returns the additional doc_ids (in the order chosen). See the module docstring for the rule."""
    by_id = {document["doc_id"]: document for document in corpus}
    unknown = [doc_id for doc_id in source_ids if doc_id not in by_id]
    if unknown:
        raise SystemExit(f"Rewrite source documents not in the corpus: {unknown}")
    if number_of_documents < len(source_ids):
        raise SystemExit(f"--number-of-documents {number_of_documents} is smaller than the {len(source_ids)} "
                         f"rewrite source documents, which must all be included")
    random_generator = random.Random(seed)
    tone_counts = Counter(by_id[doc_id]["variables"]["tone"] for doc_id in source_ids)
    language_counts = Counter((by_id[doc_id]["variables"]["tone"], by_id[doc_id]["language"]) for doc_id in source_ids)
    remaining = {}  # tone -> language -> doc_ids still available, in doc_id order
    for document in sorted(corpus, key=lambda document: document["doc_id"]):
        if document["doc_id"] not in source_ids:
            remaining.setdefault(document["variables"]["tone"], {}).setdefault(document["language"], []).append(
                document["doc_id"])
    for tone in remaining:
        tone_counts.setdefault(tone, 0)
    chosen = []
    while len(source_ids) + len(chosen) < number_of_documents:
        tones_with_documents = [tone for tone in sorted(tone_counts) if remaining.get(tone)]
        if not tones_with_documents:
            raise SystemExit(f"The corpus has only {len(source_ids) + len(chosen)} documents available")
        tone = min(tones_with_documents, key=lambda tone: tone_counts[tone])
        languages = sorted(remaining[tone])
        fewest = min(language_counts[(tone, language)] for language in languages)
        language = random_generator.choice([language for language in languages
                                            if language_counts[(tone, language)] == fewest])
        doc_id = remaining[tone][language].pop(random_generator.randrange(len(remaining[tone][language])))
        if not remaining[tone][language]:
            del remaining[tone][language]
        chosen.append(doc_id)
        tone_counts[tone] += 1
        language_counts[(tone, language)] += 1
    return chosen


def build_selection_record(corpus, source_ids, additional_ids, seed, number_of_documents, rewrites_folder):
    by_id = {document["doc_id"]: document for document in corpus}
    all_ids = sorted(set(source_ids) | set(additional_ids))
    return {
        "positive_control_rewrites_folder": str(rewrites_folder), "seed": seed,
        "number_of_documents_requested": number_of_documents, "number_of_documents_selected": len(all_ids),
        "rewrite_source_document_ids": list(source_ids), "additional_document_ids": sorted(additional_ids),
        "document_ids": all_ids,
        "counts_by_tone": dict(Counter(by_id[doc_id]["variables"]["tone"] for doc_id in all_ids)),
        "counts_by_language": dict(Counter(by_id[doc_id]["language"] for doc_id in all_ids)),
        "selection_rule": "greedy: fewest-so-far tone, then fewest-so-far language within that tone, "
                          "random document within that; seeded"}


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positive-control-rewrites-folder", required=True)
    parser.add_argument("--number-of-documents", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--output-parent-directory", default=str(SMALL_SAMPLE_CHECKS_DIRECTORY))
    parser.add_argument("--folder-name-prefix", default="pilot_document_selection",
                        help="Prefix of the output folder name (the rehearsal uses a name that says so)")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    corpus = read_jsonl(arguments.corpus_path)
    source_ids = read_rewrite_source_document_ids(arguments.positive_control_rewrites_folder)
    additional_ids = choose_additional_documents(corpus, set(source_ids), arguments.number_of_documents,
                                                 arguments.seed)
    timestamp = timestamp_now()
    output_folder = Path(arguments.output_parent_directory) / f"{arguments.folder_name_prefix}_{timestamp}"
    output_folder.mkdir(parents=True)  # fails if a selection from this minute exists, rather than overwriting it
    record = build_selection_record(corpus, source_ids, additional_ids, arguments.seed,
                                    arguments.number_of_documents, arguments.positive_control_rewrites_folder)
    ids_path = output_folder / f"pilot_document_ids_{timestamp}.txt"
    ids_path.write_text("\n".join(record["document_ids"]) + "\n", encoding="utf-8")
    (output_folder / f"pilot_document_selection_{timestamp}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(record['document_ids'])} documents ({len(source_ids)} rewrite sources, "
          f"{len(additional_ids)} additional). Tones: {record['counts_by_tone']}. "
          f"Languages: {len(record['counts_by_language'])} of {len({d['language'] for d in corpus})}.")
    print(f"Document ids file: {ids_path}")
    return ids_path


if __name__ == "__main__":
    main()
