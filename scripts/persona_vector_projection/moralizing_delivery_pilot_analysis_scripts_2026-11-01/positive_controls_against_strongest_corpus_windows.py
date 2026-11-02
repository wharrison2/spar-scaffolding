"""Like-for-like comparison: how far the positive-control rewrites' moralizing spans fall below their document's mean,
against the strongest (lowest) windows of the 20 corpus documents. Main passage-mean vector, assistant_turn, tokens after 64."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
import persona_vector_projection.analyse_moralizing_delivery_positive_controls_and_corpus_windows as analysis
from shared.run_folder_management import timestamp_now
from persona_vector_projection.moralizing_delivery_positive_control_statistics import changed_masks_from_token_sequences

run = Path(sys.argv[1])
configuration = analysis.read_run_configuration(run)
documents = analysis.read_scored_documents(run)
data = analysis.load_projection_data(run, "assistant_turn", analysis.DEFAULT_CORPUS_PATH, int(configuration["chosen_layer"]))
column = data.vector_names.index(configuration["main_vector_name"])
SKIP, WINDOW = 64, 32

def lowest_window_below_mean(values, mask=None):
    tail = values[SKIP:]
    means = np.convolve(tail, np.ones(WINDOW) / WINDOW, mode="valid")
    if mask is not None:  # windows at least half inside the changed span
        inside = np.convolve(mask.astype(float), np.ones(WINDOW) / WINDOW, mode="valid") >= 0.5
        if not inside.any():
            return np.nan
        means = means[inside]
    return float(means.min() - tail.mean())

rows, all_corpus_windows = [], []
for document_id, record in documents.items():
    values = np.asarray(data.per_token[document_id][:, column], dtype=float)
    if len(values) < SKIP + WINDOW:
        continue
    kind = record.get("kind")
    row = {"document_id": document_id, "group": "corpus" if kind == "corpus" else record.get("rewrite_arm"),
           "lowest_window_below_document_mean": lowest_window_below_mean(values),
           "per_token_standard_deviation": float(values[SKIP:].std())}
    if kind == "corpus":
        tail = values[SKIP:]
        all_corpus_windows.extend(np.convolve(tail, np.ones(WINDOW) / WINDOW, mode="valid")[::WINDOW] - tail.mean())
    else:
        changed, _ = changed_masks_from_token_sequences([int(t) for t in data.token_ids[record["source_document_id"]]],
                                                        [int(t) for t in data.token_ids[document_id]])
        mask, tail = changed[SKIP:], values[SKIP:]
        row["changed_tokens_mean_below_document_mean"] = float(tail[mask].mean() - tail.mean()) if mask.any() else np.nan
        row["changed_tokens"] = int(mask.sum())
        row["lowest_changed_window_below_document_mean"] = lowest_window_below_mean(values, mask)
    rows.append(row)

table = pd.DataFrame(rows)
all_corpus_windows = np.asarray(all_corpus_windows)
print(f"Corpus non-overlapping 32-token windows: {len(all_corpus_windows)}; standard deviation {all_corpus_windows.std():.2f}; "
      f"1st percentile {np.percentile(all_corpus_windows, 1):.2f}; 5th percentile {np.percentile(all_corpus_windows, 5):.2f}")
summary = table.groupby("group").agg(documents=("document_id", "count"),
    lowest_window_median=("lowest_window_below_document_mean", "median"),
    lowest_window_min=("lowest_window_below_document_mean", "min"),
    lowest_changed_window_median=("lowest_changed_window_below_document_mean", "median"),
    changed_tokens_mean_below_median=("changed_tokens_mean_below_document_mean", "median"),
    changed_tokens_median=("changed_tokens", "median")).round(2)
print(summary.to_string())
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / f"per_document_lowest_windows_and_changed_spans_{timestamp_now()}.csv", index=False)
