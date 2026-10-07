"""Post-process a shoehorning audit run folder into one row per occurrence plus a per-item summary.

Joins (offline, never sent to the model) each document's language and variables, verifies each
quote against the final content, and works out whether the quote was already in the pre-rewrite draft.

Usage: python scripts/shoehorning_audit/postprocess_classifications.py <run_folder>
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify_corpus import (PROJECT_ROOT, MOTIVATION_VALUES, INTEGRATION_VALUES, load_existing_records,
                             parse_pass_items, pass_file_path, read_text)
from shared.quote_verification import verify_quote
from shared.resumable_jsonl_records import read_all_records, read_jsonl, write_jsonl
from shared.run_folder_management import timestamp_now

DEFAULT_DRAFTS_PATH = (PROJECT_ROOT / "animal-welfare-data-pipeline/outputs/sdf/runs/"
                       "2026-07-25_15-57_fullscale-500-opus5/layer3/drafts.jsonl")
MOTIVATION_SHOEHORNING_RANK = {"organic": 0, "staged": 1, "unmotivated": 2}
ORIGIN_STAGES = ["present_in_draft", "added_in_rewrite", "quote_not_verified", "draft_unavailable"]


def decide_origin_stage(quote, final_content, draft_content):
    found_in_final, _, _, _ = verify_quote(quote, final_content)
    if not found_in_final:
        return "quote_not_verified"
    if draft_content is None:
        return "draft_unavailable"
    found_in_draft, _, _, _ = verify_quote(quote, draft_content)
    return "present_in_draft" if found_in_draft else "added_in_rewrite"


def build_occurrence_rows(records, documents_by_id, drafts_by_id):
    rows = []
    for record in records:
        document = documents_by_id[record["doc_id"]]
        draft = drafts_by_id.get(record["doc_id"])
        draft_content = draft["content"] if draft else None
        for item_key, item_value in record["parsed_result"]["items"].items():
            for occurrence_index, occurrence in enumerate(item_value["occurrences"]):
                found, exact, score, spans_break = verify_quote(occurrence["quote"], document["content"])
                row = {"doc_id": record["doc_id"], "pass": record["pass"], "item_key": item_key,
                       "occurrence_index": occurrence_index}
                row.update(occurrence)
                row.update({
                    "quote_found_in_document": found,
                    "quote_found_exactly_after_normalization": exact,
                    "quote_fuzzy_match_score": score,
                    "quote_spans_paragraph_break": spans_break,
                    "origin_stage": decide_origin_stage(occurrence["quote"], document["content"], draft_content),
                    "document_language": document["language"],
                    "document_variables": document["variables"]})
                rows.append(row)
    return rows


def build_other_consideration_rows(records, documents_by_id, drafts_by_id):
    rows = []
    for record in records:
        document = documents_by_id[record["doc_id"]]
        draft = drafts_by_id.get(record["doc_id"])
        draft_content = draft["content"] if draft else None
        for entry_index, entry in enumerate(record["parsed_result"].get("other_considerations", [])):
            found, exact, score, spans_break = verify_quote(entry["quote"], document["content"])
            row = {"doc_id": record["doc_id"], "pass": record["pass"], "entry_index": entry_index}
            row.update(entry)
            row.update({
                "quote_found_in_document": found,
                "quote_found_exactly_after_normalization": exact,
                "quote_fuzzy_match_score": score,
                "quote_spans_paragraph_break": spans_break,
                "origin_stage": decide_origin_stage(entry["quote"], document["content"], draft_content),
                "document_language": document["language"],
                "document_variables": document["variables"]})
            rows.append(row)
    return rows


def build_item_summary(rows, records):
    documents_per_pass = {}
    for record in records:
        documents_per_pass.setdefault(record["pass"], set()).add(record["doc_id"])
    item_order = []
    for record in records:
        for item_key in record["parsed_result"]["items"]:
            if (record["pass"], item_key) not in item_order:
                item_order.append((record["pass"], item_key))
    summary = []
    for pass_name, item_key in sorted(item_order, key=lambda pair: pair[1]):
        item_rows = [r for r in rows if r["item_key"] == item_key and r["pass"] == pass_name]
        documents_with_item = len({r["doc_id"] for r in item_rows})
        corpus_size = len(documents_per_pass[pass_name])
        entry = {"item_key": item_key, "pass": pass_name, "documents_classified": corpus_size,
                 "documents_with_item": documents_with_item,
                 "share_of_corpus": round(documents_with_item / corpus_size, 4) if corpus_size else 0,
                 "occurrence_count": len(item_rows)}
        motivation_counts = Counter(r["motivation"] for r in item_rows)
        integration_counts = Counter(r["integration"] for r in item_rows)
        origin_counts = Counter(r["origin_stage"] for r in item_rows)
        for value in MOTIVATION_VALUES:
            entry[f"motivation_{value}"] = motivation_counts[value]
        for value in INTEGRATION_VALUES:
            entry[f"integration_{value}"] = integration_counts[value]
        entry["quote_verification_rate"] = (
            round(sum(r["quote_found_in_document"] for r in item_rows) / len(item_rows), 4)
            if item_rows else None)
        for value in ORIGIN_STAGES:
            entry[f"origin_{value}"] = origin_counts[value]
        summary.append(entry)
    return summary


def list_item_keys_by_pass(pass_names, prompt_directory=None):
    """Ordered item keys for each pass, parsed from the pass files (single source of truth)."""
    item_keys_by_pass = {}
    for pass_name in pass_names:
        path = pass_file_path(pass_name) if prompt_directory is None else pass_file_path(pass_name, prompt_directory)
        item_keys_by_pass[pass_name] = [item["item_key"] for item in parse_pass_items(read_text(path))]
    return item_keys_by_pass


def most_shoehorned_motivation(occurrences):
    """Highest motivation by rank unmotivated > staged > organic; None if there are no occurrences."""
    if not occurrences:
        return None
    return max((o["motivation"] for o in occurrences), key=lambda value: MOTIVATION_SHOEHORNING_RANK[value])


def build_per_document_rows(records, error_records, documents_by_id, pass_names, item_keys_by_pass):
    """One row per document with at least one classification or error record.

    An item is only reported present/absent when its pass completed for the document;
    otherwise its columns are None (never inferred as absent).
    """
    records_by_document_and_pass = {(r["doc_id"], r["pass"]): r for r in records}
    attempted_doc_ids = {r["doc_id"] for r in records} | {e["doc_id"] for e in error_records}
    all_item_keys = sorted((key for keys in item_keys_by_pass.values() for key in keys),
                           key=lambda key: int(key.split("_")[1]))
    item_pass = {key: pass_name for pass_name, keys in item_keys_by_pass.items() for key in keys}
    rows = []
    for doc_id in sorted(attempted_doc_ids):
        document = documents_by_id[doc_id]
        passes_completed = [p for p in pass_names if (doc_id, p) in records_by_document_and_pass]
        passes_failed = [p for p in pass_names if p not in passes_completed
                         and any(e["doc_id"] == doc_id and e["pass"] == p for e in error_records)]
        row = {"doc_id": doc_id, "language": document["language"]}
        row.update(document["variables"])
        row.update({"passes_completed": passes_completed, "passes_failed": passes_failed,
                    "all_passes_completed": len(passes_completed) == len(pass_names)})
        for item_key in all_item_keys:
            record = records_by_document_and_pass.get((doc_id, item_pass[item_key]))
            if record is None:
                values = (None, None, None, None)
            else:
                occurrences = record["parsed_result"]["items"].get(item_key, {"occurrences": []})["occurrences"]
                values = (len(occurrences) > 0, len(occurrences), most_shoehorned_motivation(occurrences),
                          any(o["integration"] == "aside" for o in occurrences) if occurrences else None)
            for suffix, value in zip(["present", "occurrence_count", "most_shoehorned_motivation", "any_aside"], values):
                row[f"{item_key}__{suffix}"] = value
        total = 0
        for pass_name in pass_names:
            record = records_by_document_and_pass.get((doc_id, pass_name))
            count = None if record is None else len(record["parsed_result"].get("other_considerations", []))
            row[f"other_considerations_count_{pass_name}"] = count
            total += count or 0
        row["other_considerations_count_total"] = total
        rows.append(row)
    return rows


def to_csv_value(value):
    """None becomes empty, lists are joined with ';', dicts are JSON-encoded."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(element) for element in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def derive_csv_columns(rows, preferred_columns=()):
    """Preferred columns present in the rows first, then every other field in first-seen order."""
    seen_fields = []
    for row in rows:
        for field in row:
            if field not in seen_fields:
                seen_fields.append(field)
    preferred = [column for column in preferred_columns if column in seen_fields]
    return preferred + [field for field in seen_fields if field not in preferred]


def write_rows_to_csv(path, rows, preferred_columns=()):
    """Write dict rows to CSV with columns derived from the rows, so new fields can never crash it."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=derive_csv_columns(rows, preferred_columns), restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_csv_value(value) for key, value in row.items()})


def write_per_document_csv(path, rows):
    write_rows_to_csv(path, rows)


def write_per_document_outputs(run_folder, timestamp, documents_by_id, pass_names, records=None):
    run_folder = Path(run_folder)
    records = load_existing_records(run_folder) if records is None else records
    error_records = read_all_records(run_folder, "errors_*.jsonl")
    rows = build_per_document_rows(records, error_records, documents_by_id, pass_names,
                                   list_item_keys_by_pass(pass_names))
    jsonl_path = run_folder / f"per_document_results_{timestamp}.jsonl"
    csv_path = run_folder / f"per_document_results_{timestamp}.csv"
    write_jsonl(jsonl_path, rows)
    write_per_document_csv(csv_path, rows)
    return jsonl_path, csv_path, rows


def postprocess_run_folder(run_folder, drafts_path=DEFAULT_DRAFTS_PATH):
    run_folder = Path(run_folder)
    configuration_path = sorted(run_folder.glob("run_configuration_*.json"))[0]
    configuration = json.loads(configuration_path.read_text())
    corpus_path = configuration["corpus_path"]
    documents_by_id = {d["doc_id"]: d for d in read_jsonl(corpus_path)}
    drafts_by_id = {d["doc_id"]: d for d in read_jsonl(drafts_path)} if Path(drafts_path).exists() else {}
    records = load_existing_records(run_folder)
    rows = build_occurrence_rows(records, documents_by_id, drafts_by_id)
    timestamp = timestamp_now()
    occurrences_path = run_folder / f"occurrences_{timestamp}.jsonl"
    write_jsonl(occurrences_path, rows)
    summary = build_item_summary(rows, records)
    summary_path = run_folder / f"per_item_summary_{timestamp}.csv"
    if summary:
        write_rows_to_csv(summary_path, summary)
    other_rows = build_other_consideration_rows(records, documents_by_id, drafts_by_id)
    other_jsonl_path = run_folder / f"other_considerations_{timestamp}.jsonl"
    write_jsonl(other_jsonl_path, other_rows)
    other_csv_path = run_folder / f"other_considerations_{timestamp}.csv"
    sorted_other_rows = sorted(other_rows, key=lambda r: (r["pass"], r["suggested_name"].lower()))
    preferred_columns = ["pass", "suggested_name", "description", "doc_id", "entry_index", "quote",
                         "english_gloss", "context", "motivation_reasoning", "motivation", "integration_reasoning",
                         "integration",
                         "quote_found_in_document", "quote_fuzzy_match_score", "quote_spans_paragraph_break",
                         "origin_stage", "document_language"]
    write_rows_to_csv(other_csv_path, sorted_other_rows, preferred_columns)
    per_document_jsonl_path, per_document_csv_path, _ = write_per_document_outputs(
        run_folder, timestamp, documents_by_id, configuration["passes"], records)
    postprocess_run_folder.last_output_paths = {
        "other_considerations_jsonl": other_jsonl_path, "other_considerations_csv": other_csv_path,
        "per_document_results_jsonl": per_document_jsonl_path, "per_document_results_csv": per_document_csv_path}
    return occurrences_path, summary_path, rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_folder")
    parser.add_argument("--drafts-path", default=str(DEFAULT_DRAFTS_PATH))
    arguments = parser.parse_args()
    occurrences_path, summary_path, rows, _ = postprocess_run_folder(arguments.run_folder, arguments.drafts_path)
    print(f"{len(rows)} occurrences -> {occurrences_path}\nsummary -> {summary_path}")
    for label, path in postprocess_run_folder.last_output_paths.items():
        print(f"{label} -> {path}")
