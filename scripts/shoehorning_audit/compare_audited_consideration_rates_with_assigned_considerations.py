"""Compare how often each shoehorning-audit item appears in documents where its consideration was NOT assigned
with documents where it was assigned, and with its target share. Offline only: no model or API calls.

Usage: python scripts/shoehorning_audit/compare_audited_consideration_rates_with_assigned_considerations.py
"""

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"

ORIGINAL_AUDIT_FOLDER = DATA / "shoehorning_audit_runs/full_corpus_three_passes_document_specific_main_line_prompts_2026-10-07_02-52"
NEW_AUDIT_FOLDER = DATA / "shoehorning_audit_runs/shoehorning_audit_of_one_objective_considerations_generation_medium_effort_2026-10-24_03-09"
NEW_PIPELINE_RUN = (DATA / "spec_contradiction_screen_runs/small_sample_checks/generation_and_checks_from_one_objective_considerations_2026-10-24_03-02/"
                    "pipeline_outputs/sdf/runs/2026-10-24_03-02_one-objective-considerations-generation")
CATALOG_PATH = PROJECT_ROOT / "animal-welfare-data-pipeline-considerations-selector/prompts/sdf/considerations.txt"

# Catalog label (normalised) -> (primary audit item, secondary audit item, is_loose_match)
CATALOG_TO_AUDIT_ITEM = {
    "substrate neutrality": ("item_02_substrate_neutrality", None, False),
    "a deliberately open moral circle": ("item_03_open_or_expanding_moral_circle", None, False),
    "scope sensitivity": ("item_04_scope_sensitivity", None, False),
    'the fact that "natural" does not mean "ok"': ("item_01_natural_or_traditional_does_not_make_it_ok", None, False),
    "the precautionary principle applied to sentience": ("item_19_precautionary_principle_for_uncertain_sentience", None, False),
    "individualism": ("item_10_individual_welfare_vs_ecology", None, False),
    "weighing evidence for sentience": ("item_24_sentience_evidence_for_the_animal_at_hand", "item_17_stock_calibration_passage", False),
    "weighing species against each other": ("item_15_contested_comparisons_flagged_unresolved", None, False),
    "difficulties of ai self-reporting welfare": ("item_23_ai_uncertainty_about_own_inner_states", None, False),
    "weighing theories of consciousness and their implications for ai": ("item_08_ai_own_moral_status", None, True),
}
# Original corpus: weights of reasoning_featured (matched by normalised prefix of the dealt text)
ORIGINAL_TARGETS = {
    "any relevant principles from the constitution": 0.20,
    "the continuity between welfare concern and the rest of the ai's aligned character": 0.20,
    "scope sensitivity": 0.10, 'the fact that "natural" does not mean "ok"': 0.10, "individualism": 0.10,
    "the precautionary principle applied to sentience": 0.10, "a deliberately open moral circle": 0.10,
    "substrate neutrality": 0.10,
}
MOTIVATIONS = ["organic", "staged", "unmotivated"]
INTEGRATIONS = ["woven", "aside"]


def normalise_text(text):
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def label_before_colon(text):
    return normalise_text(text.split(":")[0])


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest(folder, pattern):
    return sorted(folder.glob(pattern))[-1]


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def share(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def load_audit(folder):
    rows = list(csv.DictReader(open(latest(folder, "per_document_results_*.csv"), encoding="utf-8")))
    occurrences = read_jsonl(latest(folder, "occurrences_*.jsonl"))
    item_keys = sorted({c.split("__")[0] for c in rows[0] if c.endswith("__present")}, key=lambda k: int(k.split("_")[1]))
    return rows, occurrences, item_keys


def is_present(row, item_key):
    value = row[f"{item_key}__present"]
    return None if value == "" else value == "True"


def item_statistics(rows, occurrences, item_key, document_filter=None):
    """Documents classified, documents present, share, and occurrence splits, over the (filtered) documents."""
    selected_ids = {r["doc_id"] for r in rows if (document_filter is None or document_filter(r))}
    classified = [r for r in rows if r["doc_id"] in selected_ids and is_present(r, item_key) is not None]
    present = [r for r in classified if is_present(r, item_key)]
    item_occurrences = [o for o in occurrences if o["item_key"] == item_key and o["doc_id"] in selected_ids]
    return {"classified": len(classified), "present": len(present), "share": share(len(present), len(classified)),
            "occurrences": item_occurrences,
            "motivation": {m: sum(o["motivation"] == m for o in item_occurrences) for m in MOTIVATIONS},
            "integration": {i: sum(o["integration"] == i for o in item_occurrences) for i in INTEGRATIONS}}


def render_table(title, header, body_rows):
    widths = [max(len(str(x)) for x in [h] + [r[i] for r in body_rows]) for i, h in enumerate(header)]
    lines = ["", "=" * 100, title, "=" * 100,
             "  ".join(h.ljust(w) for h, w in zip(header, widths)), "  ".join("-" * w for w in widths)]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in body_rows]
    return "\n".join(lines)


def short_item(item_key):
    return item_key  # full names, never abbreviated


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = DATA / f"shoehorning_audit_runs/audited_consideration_rates_compared_with_assigned_considerations_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=False)
    notes = []
    tables_text = []
    csv_tables = {}

    def emit(name, title, header, body):
        csv_tables[name] = (header, body)
        tables_text.append(render_table(title, header, body))

    # ---- load
    original_rows, original_occurrences, item_keys = load_audit(ORIGINAL_AUDIT_FOLDER)
    new_rows, new_occurrences, new_item_keys = load_audit(NEW_AUDIT_FOLDER)
    assert item_keys == new_item_keys and len(item_keys) == 25
    original_corpus = {r["doc_id"]: r for r in read_jsonl(DATA / "sdf_corpus.jsonl")}
    new_corpus = {r["doc_id"]: r for r in read_jsonl(NEW_PIPELINE_RUN / "final/sdf_corpus.jsonl")}
    notes.append(f"Original: {len(original_rows)} audit rows, {len(original_corpus)} corpus documents, "
                 f"ids equal: {set(r['doc_id'] for r in original_rows) == set(original_corpus)}; "
                 f"audit passes failed: {sum(r['passes_failed'] != '' for r in original_rows)}; "
                 f"documents with any item unclassified: {sum(any(is_present(r, k) is None for k in item_keys) for r in original_rows)}")
    notes.append(f"New: {len(new_rows)} audit rows, {len(new_corpus)} corpus documents, "
                 f"ids equal: {set(r['doc_id'] for r in new_rows) == set(new_corpus)}; "
                 f"audit passes failed: {sum(r['passes_failed'] != '' for r in new_rows)}; "
                 f"documents with any item unclassified: {sum(any(is_present(r, k) is None for k in item_keys) for r in new_rows)}")
    mismatched_featured = [r["doc_id"] for r in original_rows
                           if r["reasoning_featured"] != original_corpus[r["doc_id"]]["variables"]["reasoning_featured"]]
    notes.append(f"Original documents whose audit-csv reasoning_featured differs from sdf_corpus.jsonl: {len(mismatched_featured)}")

    # ---- catalog
    catalog = []  # (list_name, label, normalised label, target share)
    current_list = None
    for line in open(CATALOG_PATH, encoding="utf-8"):
        stripped = line.strip()
        header_match = re.match(r"\{(\w+)\}", stripped)
        if header_match:
            current_list = header_match.group(1)
            continue
        entry_match = re.match(r"([0-9.]+)\s*::\s*(.+)", stripped)
        if entry_match and current_list and "considerations" in current_list:
            label = entry_match.group(2).split(": ")[0].strip()
            catalog.append((current_list, label, label_before_colon(label), float(entry_match.group(1))))
    catalog_by_key = {entry[2]: entry for entry in catalog}
    for key in CATALOG_TO_AUDIT_ITEM:
        assert key in catalog_by_key, f"mapping key not in catalog: {key}"
    item_to_catalog_key = {}
    for key, (primary, secondary, _) in CATALOG_TO_AUDIT_ITEM.items():
        item_to_catalog_key[primary] = key

    coverage = json.loads((NEW_PIPELINE_RUN / "layer1/consideration_coverage.json").read_text())
    coverage_by_key = {label_before_colon(e["label"]): e for e in coverage["entries"]}
    selections = read_jsonl(NEW_PIPELINE_RUN / "layer1/consideration_selections.jsonl")
    assigned_by_document = {}
    unknown_labels = set()
    for selection in selections:
        labels = set()
        for outcome in selection["outcomes"]:
            if outcome["status"] == "accepted" and outcome["label"]:
                key = label_before_colon(outcome["label"])
                if key not in coverage_by_key:
                    unknown_labels.add(outcome["label"])
                labels.add(key)
        assigned_by_document[selection["prompt_id"]] = labels
    notes.append(f"Selections: {len(selections)} prompt ids; ids matching the final corpus: "
                 f"{len(set(assigned_by_document) & set(new_corpus))} of {len(new_corpus)}; selection labels missing from coverage: {sorted(unknown_labels)}")
    missing_selection = sorted(set(new_corpus) - set(assigned_by_document))
    if missing_selection:
        notes.append(f"Corpus documents with no selection record: {missing_selection}")
    # agreement with the audit csv columns
    disagreements = []
    for row in new_rows:
        csv_labels = set()
        for column in ("moral_considerations", "epistemic_considerations"):
            for piece in row[column].split(";"):
                if piece.strip():
                    csv_labels.add(label_before_colon(piece))
        selection_labels = assigned_by_document.get(row["doc_id"], set())
        if csv_labels != selection_labels:
            disagreements.append((row["doc_id"], sorted(csv_labels), sorted(selection_labels)))
    notes.append(f"Documents where audit-csv moral/epistemic considerations differ from layer1 selections: {len(disagreements)} {disagreements[:3]}")

    original_targets_by_item = {}
    original_featured_key = {}  # doc_id -> normalised target key (prefix-matched)
    for row in original_rows:
        text = normalise_text(row["reasoning_featured"])
        matches = [k for k in ORIGINAL_TARGETS if text.startswith(k)]
        assert len(matches) == 1, text
        original_featured_key[row["doc_id"]] = matches[0]
    for item_key, catalog_key in item_to_catalog_key.items():
        if catalog_key in ORIGINAL_TARGETS:
            original_targets_by_item[item_key] = ORIGINAL_TARGETS[catalog_key]
    new_target_by_item = {i: catalog_by_key[k][3] for i, k in item_to_catalog_key.items()}

    # ---- Table 1
    body = []
    for item_key in item_keys:
        catalog_key = item_to_catalog_key.get(item_key)
        secondary_of = [k for k, (p, s, _) in CATALOG_TO_AUDIT_ITEM.items() if s == item_key]
        mapped = catalog_by_key[catalog_key][1] + (" (LOOSE)" if CATALOG_TO_AUDIT_ITEM[catalog_key][2] else "") if catalog_key else \
            (f"secondary of: {catalog_by_key[secondary_of[0]][1]}" if secondary_of else "none (unassigned-only)")
        for corpus_name, rows, occurrences in (("Original", original_rows, original_occurrences), ("New", new_rows, new_occurrences)):
            stats = item_statistics(rows, occurrences, item_key)
            if corpus_name == "Original":
                target = original_targets_by_item.get(item_key)
            else:
                target = new_target_by_item.get(item_key)
            body.append([item_key, corpus_name, stats["classified"], stats["present"], fmt(stats["share"]),
                         len(stats["occurrences"]), *[stats["motivation"][m] for m in MOTIVATIONS],
                         *[stats["integration"][i] for i in INTEGRATIONS], mapped, fmt(target)])
    emit("table_1_every_audit_item_per_corpus", "TABLE 1: every audit item, per corpus",
         ["audit_item", "corpus", "documents_classified", "documents_present", "share_present", "occurrences",
          "organic", "staged", "unmotivated", "woven", "aside", "mapped_catalog_consideration", "target_share_in_this_corpus"], body)

    # ---- Table 2
    body = []
    for catalog_key, (item_key, _, _) in CATALOG_TO_AUDIT_ITEM.items():
        if catalog_key not in ORIGINAL_TARGETS:
            continue
        featured = lambda r, k=catalog_key: original_featured_key[r["doc_id"]] == k
        not_featured = lambda r, k=catalog_key: original_featured_key[r["doc_id"]] != k
        on = item_statistics(original_rows, original_occurrences, item_key, featured)
        off = item_statistics(original_rows, original_occurrences, item_key, not_featured)
        on_low, on_high = wilson_interval(on["present"], on["classified"])
        off_low, off_high = wilson_interval(off["present"], off["classified"])
        staged_or_unmotivated = off["motivation"]["staged"] + off["motivation"]["unmotivated"]
        body.append([item_key, f"{on['present']}/{on['classified']}", fmt(on["share"]), f"[{fmt(on_low)}, {fmt(on_high)}]",
                     f"{off['present']}/{off['classified']}", fmt(off["share"]), f"[{fmt(off_low)}, {fmt(off_high)}]",
                     fmt(share(off["share"], on["share"]) if on["share"] else float("nan"), 2),
                     f"{staged_or_unmotivated}/{len(off['occurrences'])}",
                     fmt(share(staged_or_unmotivated, len(off["occurrences"])), 2),
                     fmt(off["share"] / 0.10, 2)])
    emit("table_2_original_featured_versus_not_featured",
         "TABLE 2: Original corpus, item present when its consideration is the dealt reasoning_featured vs all other documents "
         "(6 mapped moral items; item_08 has no matching dealt value)",
         ["audit_item", "featured_present", "featured_share", "featured_wilson_95", "not_featured_present", "not_featured_share",
          "not_featured_wilson_95", "ratio_not_featured_over_featured", "not_featured_staged_or_unmotivated_occurrences",
          "not_featured_staged_or_unmotivated_fraction", "not_featured_share_over_old_target_0.10"], body)

    # ---- Table 3
    featured_keys = list(ORIGINAL_TARGETS)
    header = ["reasoning_featured", "documents"] + item_keys
    body = []
    for featured_key in featured_keys:
        selected = [r for r in original_rows if original_featured_key[r["doc_id"]] == featured_key]
        row_cells = [featured_key, len(selected)]
        for item_key in item_keys:
            classified = [r for r in selected if is_present(r, item_key) is not None]
            row_cells.append(fmt(share(sum(is_present(r, item_key) for r in classified), len(classified)), 2))
        body.append(row_cells)
    all_cells = ["ALL DOCUMENTS", len(original_rows)] + [fmt(item_statistics(original_rows, original_occurrences, k)["share"], 2) for k in item_keys]
    body.append(all_cells)
    emit("table_3_original_share_present_by_featured_consideration",
         "TABLE 3: Original corpus, share of documents with each audit item, by reasoning_featured value", header, body)

    # ---- Table 4a
    body = []
    new_ranking = {}
    for item_key in item_keys:
        catalog_key = item_to_catalog_key.get(item_key)
        secondary_of = [k for k, (p, s, _) in CATALOG_TO_AUDIT_ITEM.items() if s == item_key]
        mapped_keys = [catalog_key] if catalog_key else secondary_of
        if mapped_keys:
            key = mapped_keys[0]
            assigned = lambda r, k=key: k in assigned_by_document.get(r["doc_id"], set())
            not_assigned = lambda r, k=key: k not in assigned_by_document.get(r["doc_id"], set())
            on = item_statistics(new_rows, new_occurrences, item_key, assigned)
            off = item_statistics(new_rows, new_occurrences, item_key, not_assigned)
            target = catalog_by_key[key][3]
            label = catalog_by_key[key][1] + (" (secondary mapping)" if not catalog_key else "") + \
                (" (LOOSE)" if CATALOG_TO_AUDIT_ITEM[key][2] else "")
            new_ranking[item_key] = off["share"]
            body.append([item_key, label, f"{on['present']}/{on['classified']}", fmt(on["share"]),
                         f"{off['present']}/{off['classified']}", fmt(off["share"]), fmt(target), fmt(off["share"] / target, 2)])
        else:
            allstats = item_statistics(new_rows, new_occurrences, item_key)
            new_ranking[item_key] = allstats["share"]
            body.append([item_key, "none (unassigned-only)", "-", "-", f"{allstats['present']}/{allstats['classified']}",
                         fmt(allstats["share"]), "-", "-"])
    emit("table_4a_new_corpus_assigned_versus_not_assigned",
         "TABLE 4a: New selector corpus, item present among documents assigned the mapped consideration vs not assigned",
         ["audit_item", "mapped_catalog_consideration", "assigned_present", "assigned_share", "not_assigned_present",
          "not_assigned_share", "current_target_share", "not_assigned_share_over_target_share"], body)

    # ---- Table 4b
    scores = read_jsonl(NEW_PIPELINE_RUN / "layer1/consideration_scores.jsonl")
    scores_per_entry = {}
    for record in scores:
        for list_name, entries in record["scores"].items():
            for label, value in entries.items():
                scores_per_entry.setdefault(label_before_colon(label), []).append(value["score"])
    body = []
    for entry in coverage["entries"]:
        key = label_before_colon(entry["label"])
        entry_scores = scores_per_entry.get(key, [])
        audit_item = CATALOG_TO_AUDIT_ITEM.get(key, (None,))[0]
        body.append([entry["label"][:70], entry["list"], fmt(entry["target_share"], 2), entry["target_count"], entry["achieved"],
                     len(entry_scores), sum(s >= 3 for s in entry_scores), sum(s >= 4 for s in entry_scores),
                     sum(s == 5 for s in entry_scores), entry["documents_scoring_at_least_threshold"], audit_item or "none"])
    emit("table_4b_new_corpus_selector_targets_and_scores",
         "TABLE 4b: every catalog entry, selector target count, achieved, and scenarios scored >=3, >=4, 5 (26 scenarios)",
         ["catalog_entry", "list", "target_share", "target_count", "achieved", "scenarios_scored", "scored_at_least_3",
          "scored_at_least_4", "scored_5", "coverage_file_documents_at_least_threshold", "mapped_audit_item"], body)

    # ---- Table 5
    original_ranking = {}
    original_ranking_counts = {}
    for item_key in item_keys:
        catalog_key = item_to_catalog_key.get(item_key)
        if catalog_key in ORIGINAL_TARGETS:
            stats = item_statistics(original_rows, original_occurrences, item_key, lambda r, k=catalog_key: original_featured_key[r["doc_id"]] != k)
            basis = "not featured"
        else:
            stats = item_statistics(original_rows, original_occurrences, item_key)
            basis = "all documents (no matching dealt value)"
        original_ranking[item_key] = stats["share"]
        original_ranking_counts[item_key] = (f"{stats['present']}/{stats['classified']}", basis)
    body = []
    for rank, item_key in enumerate(sorted(item_keys, key=lambda k: -original_ranking[k]), start=1):
        body.append([rank, item_key, original_ranking_counts[item_key][0], fmt(original_ranking[item_key]),
                     original_ranking_counts[item_key][1], fmt(new_ranking[item_key]),
                     "mapped" if item_key in item_to_catalog_key else "unassigned-only"])
    emit("table_5_ranked_by_unassigned_base_rate",
         "TABLE 5: items ranked by Original unassigned base rate (New not-assigned share alongside)",
         ["rank", "audit_item", "original_present", "original_unassigned_share", "original_basis", "new_not_assigned_share", "mapping_status"], body)

    # ---- write
    for name, (header, body) in csv_tables.items():
        with open(output_folder / f"{name}_{timestamp}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(body)
    summary = "DATA CHECKS\n" + "\n".join("- " + n for n in notes) + "\n" + "\n".join(tables_text) + "\n"
    (output_folder / f"printed_summary_of_all_tables_{timestamp}.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Output folder: {output_folder}")


if __name__ == "__main__":
    main()
