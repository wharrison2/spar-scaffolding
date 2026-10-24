"""Positive-control analyses that go beyond the whole-document check in analyze_preachiness_scores.py.

For each rewritten document, the rewrite's Gemma tokens are aligned to its source's tokens with
difflib.SequenceMatcher. Rewrite tokens outside every matching block are "changed" (inserted or reworded),
rewrite tokens inside one are "unchanged", and source tokens outside every matching block are "replaced".
Writes, into <run folder>/positive_control_changed_span_analysis_<timestamp>/:
  changed_span_preachy_minus_control_<timestamp>.csv   per rewrite: token counts and preachy-minus-control per token
                                                       on changed, unchanged and replaced text
  changed_span_by_persona_<timestamp>.csv              per persona: persona-minus-generic on changed minus unchanged
                                                       and changed minus replaced, preachier rewrites only
  total_log_likelihood_differences_<timestamp>.csv     per document: total log P under preachy minus under controls
  summary_<timestamp>.json                             counts of rewrites moving the expected way and Wilcoxon p-values

Usage: python3 scripts/log_probability_persona_check/analyze_positive_control_changed_spans.py --run-folder <run>
"""

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.shared.run_folder_management import timestamp_now  # noqa: E402
from scripts.shared.resumable_jsonl_records import read_jsonl  # noqa: E402

CONTROL_PERSONA_NAMES = ("animal_husbandry_expertise", "aesthetic_love_of_animals",
                         "practical_and_indifferent_to_welfare", "typical_ai_view_of_animals", "zoological_knowledge")


def load_per_token_persona_minus_generic(run_folder, document_id):
    archive = np.load(run_folder / "scoring_outputs/per_token_log_probabilities" / f"{document_id}.npz",
                      allow_pickle=True)
    condition_ids, log_probabilities = list(archive["condition_ids"]), archive["log_probabilities"]
    generic = log_probabilities[condition_ids.index("generic")]
    rows_by_persona = defaultdict(list)
    for row, condition_id in enumerate(condition_ids):
        if "__rewording_" in condition_id:
            rows_by_persona[condition_id.split("__")[0]].append(log_probabilities[row] - generic)
    by_persona = {persona: np.mean(rows, axis=0) for persona, rows in rows_by_persona.items()}
    by_persona["pooled_control"] = np.mean([by_persona[name] for name in CONTROL_PERSONA_NAMES], axis=0)
    return archive["document_token_ids"].tolist(), by_persona


def changed_token_masks(source_token_ids, rewrite_token_ids):
    matcher = difflib.SequenceMatcher(None, source_token_ids, rewrite_token_ids, autojunk=False)
    rewrite_changed = np.ones(len(rewrite_token_ids), bool)
    source_replaced = np.ones(len(source_token_ids), bool)
    for block in matcher.get_matching_blocks():
        rewrite_changed[block.b:block.b + block.size] = False
        source_replaced[block.a:block.a + block.size] = False
    return rewrite_changed, source_replaced


def masked_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else float("nan")


def wilcoxon_p_value(values):
    values = [value for value in values if not np.isnan(value)]
    return float(wilcoxon(values).pvalue) if len(values) > 1 and any(values) else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True, type=Path)
    run_folder = parser.parse_args().run_folder
    timestamp = timestamp_now()
    output_folder = run_folder / f"positive_control_changed_span_analysis_{timestamp}"
    output_folder.mkdir()

    documents = read_jsonl(run_folder / "scoring_inputs/documents.jsonl")
    loaded = {document["document_id"]: load_per_token_persona_minus_generic(run_folder, document["document_id"])
              for document in documents}

    span_rows, persona_rows = [], defaultdict(lambda: defaultdict(list))
    for document in documents:
        if not document["rewrite_direction"]:
            continue
        rewrite_ids, rewrite_scores = loaded[document["document_id"]]
        source_ids, source_scores = loaded[document["source_document_id"]]
        rewrite_changed, source_replaced = changed_token_masks(source_ids, rewrite_ids)
        preachy_minus_control_rewrite = rewrite_scores["preachy"] - rewrite_scores["pooled_control"]
        preachy_minus_control_source = source_scores["preachy"] - source_scores["pooled_control"]
        changed = masked_mean(preachy_minus_control_rewrite, rewrite_changed)
        unchanged = masked_mean(preachy_minus_control_rewrite, ~rewrite_changed)
        replaced = masked_mean(preachy_minus_control_source, source_replaced)
        span_rows.append({
            "document_id": document["document_id"], "source_document_id": document["source_document_id"],
            "rewrite_direction": document["rewrite_direction"], "rewrite_tokens": len(rewrite_ids),
            "changed_tokens": int(rewrite_changed.sum()), "replaced_source_tokens": int(source_replaced.sum()),
            "preachy_minus_control_per_token_changed": changed,
            "preachy_minus_control_per_token_unchanged": unchanged,
            "preachy_minus_control_per_token_replaced_source_text": replaced,
            "changed_minus_unchanged": changed - unchanged, "changed_minus_replaced": changed - replaced})
        if document["rewrite_direction"] == "preachier":
            for persona, scores in rewrite_scores.items():
                changed_persona = masked_mean(scores, rewrite_changed)
                persona_rows[persona]["changed_minus_unchanged"].append(changed_persona - masked_mean(scores, ~rewrite_changed))
                persona_rows[persona]["changed_minus_replaced"].append(
                    changed_persona - masked_mean(source_scores[persona], source_replaced))
    span_table = pd.DataFrame(span_rows)
    span_table.to_csv(output_folder / f"changed_span_preachy_minus_control_{timestamp}.csv", index=False)

    persona_table = pd.DataFrame([{
        "persona": persona,
        "mean_changed_minus_unchanged": float(np.nanmean(values["changed_minus_unchanged"])),
        "preachier_rewrites_where_changed_above_unchanged": int(np.sum(np.array(values["changed_minus_unchanged"]) > 0)),
        "mean_changed_minus_replaced": float(np.nanmean(values["changed_minus_replaced"])),
        "preachier_rewrites_where_changed_above_replaced": int(np.sum(np.array(values["changed_minus_replaced"]) > 0)),
        "number_of_preachier_rewrites": len(values["changed_minus_unchanged"])}
        for persona, values in persona_rows.items()]).sort_values("mean_changed_minus_unchanged", ascending=False)
    persona_table.to_csv(output_folder / f"changed_span_by_persona_{timestamp}.csv", index=False)

    totals = defaultdict(lambda: defaultdict(list))
    for record in read_jsonl(run_folder / "scoring_outputs/scoring_completion_records.jsonl"):
        persona = record["condition_id"].split("__")[0]
        group = "control" if persona in CONTROL_PERSONA_NAMES else persona
        totals[record["document_id"]][group].append(record["sum_log_probability"])
    total_rows = [{"document_id": document_id,
                   "total_log_probability_preachy_minus_control": np.mean(groups["preachy"]) - np.mean(groups["control"])}
                  for document_id, groups in totals.items()]
    total_table = pd.DataFrame(total_rows)
    total_table.to_csv(output_folder / f"total_log_likelihood_differences_{timestamp}.csv", index=False)
    total_by_document = dict(zip(total_table.document_id, total_table.total_log_probability_preachy_minus_control))

    summary = {"run_folder": str(run_folder)}
    for direction, expected_sign in (("preachier", 1), ("plainer", -1)):
        rows = span_table[span_table.rewrite_direction == direction]
        changed_minus_unchanged = rows.changed_minus_unchanged.tolist()
        total_moves = [total_by_document[row.document_id] - total_by_document[row.source_document_id]
                       for row in rows.itertuples()]
        summary[direction] = {
            "number_of_rewrites": len(rows),
            "rewrites_with_no_changed_tokens": int((rows.changed_tokens == 0).sum()),
            "changed_minus_unchanged_positive": int(np.nansum(np.array(changed_minus_unchanged) > 0)),
            "changed_minus_unchanged_mean": float(np.nanmean(changed_minus_unchanged)),
            "changed_minus_unchanged_wilcoxon_p_two_sided": wilcoxon_p_value(changed_minus_unchanged),
            "changed_minus_replaced_positive": int(np.nansum(rows.changed_minus_replaced > 0)),
            "total_log_probability_move_in_expected_direction": int(sum(np.sign(move) == expected_sign for move in total_moves)),
            "total_log_probability_move_wilcoxon_p_two_sided": wilcoxon_p_value(total_moves)}
    (output_folder / f"summary_{timestamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Written to {output_folder}")


if __name__ == "__main__":
    main()
