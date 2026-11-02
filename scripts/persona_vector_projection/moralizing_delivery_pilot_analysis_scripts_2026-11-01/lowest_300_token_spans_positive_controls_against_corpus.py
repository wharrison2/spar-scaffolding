"""Lowest 300-token span below the document mean (tokens after 64) for every rewrite, against the same statistic for its source
corpus document; plus the lowest span that holds at least half of the rewrite's changed tokens. Main vector, assistant_turn."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
import persona_vector_projection.analyse_moralizing_delivery_positive_controls_and_corpus_windows as analysis
from shared.run_folder_management import timestamp_now
from persona_vector_projection.moralizing_delivery_positive_control_statistics import changed_masks_from_token_sequences

run, output_folder = Path(sys.argv[1]), Path(sys.argv[2])
SKIP, SPAN = 64, 300
configuration = analysis.read_run_configuration(run)
documents = analysis.read_scored_documents(run)
data = analysis.load_projection_data(run, "assistant_turn", analysis.DEFAULT_CORPUS_PATH, int(configuration["chosen_layer"]))
column = data.vector_names.index(configuration["main_vector_name"])

def span_means_below_mean(document_id):
    tail = np.asarray(data.per_token[document_id][:, column], dtype=float)[SKIP:]
    return np.convolve(tail, np.ones(SPAN) / SPAN, mode="valid") - tail.mean()

rows = []
for document_id, record in documents.items():
    if record.get("kind") != "rewrite":
        continue
    source = record["source_document_id"]
    means = span_means_below_mean(document_id)
    changed, _ = changed_masks_from_token_sequences([int(t) for t in data.token_ids[source]], [int(t) for t in data.token_ids[document_id]])
    changed_tail = changed[SKIP:].astype(float)
    changed_in_span = np.convolve(changed_tail, np.ones(SPAN), mode="valid")
    holds_changes = changed_in_span >= 0.5 * min(changed_tail.sum(), SPAN)
    lowest_start = int(np.argmin(means))
    rows.append({"arm": record.get("rewrite_arm"), "document_id": document_id, "source_document_id": source,
                 "rewrite_lowest_span": float(means.min()), "source_lowest_span": float(span_means_below_mean(source).min()),
                 "lowest_span_holding_changes": float(means[holds_changes].min()) if holds_changes.any() else np.nan,
                 "lowest_span_overlaps_changes": bool(changed_in_span[lowest_start] > 0),
                 "changed_tokens": int(changed_tail.sum())})
table = pd.DataFrame(rows)
table["rewrite_minus_source_lowest_span"] = table.rewrite_lowest_span - table.source_lowest_span
output_folder.mkdir(parents=True, exist_ok=True)
table.to_csv(output_folder / f"lowest_300_token_spans_rewrites_against_sources_{timestamp_now()}.csv", index=False)
summary = table.groupby("arm").agg(rewrites=("document_id", "count"), changed_tokens_median=("changed_tokens", "median"),
    lowest_span_holding_changes_median=("lowest_span_holding_changes", "median"),
    rewrite_lowest_span_median=("rewrite_lowest_span", "median"), source_lowest_span_median=("source_lowest_span", "median"),
    rewrite_minus_source_median=("rewrite_minus_source_lowest_span", "median"),
    rewrite_lower_than_source=("rewrite_minus_source_lowest_span", lambda s: int((s < 0).sum())),
    lowest_span_overlaps_changes=("lowest_span_overlaps_changes", "sum"))
print(summary.round(2).to_string())
corpus_ids = [i for i, r in documents.items() if r.get("kind") == "corpus"]
print("Corpus documents' lowest span, median", round(float(np.median([span_means_below_mean(i).min() for i in corpus_ids])), 2))
