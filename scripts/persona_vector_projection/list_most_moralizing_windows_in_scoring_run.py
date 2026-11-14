"""Lists the lowest-projecting (for moralizing delivery: most moralizing) 32-token windows of the documents in a scoring run, from the
per-token projections the scoring pod saved. Local and free: needs the run's tokenizer in the local Hugging Face cache.

Two rankings, both over windows starting after the first 64 tokens:
  below_document_mean   how far the window mean falls below its own document's mean (the corpus-window check of the positive-control
                        analysis); several windows per document allowed, never overlapping
  absolute              the window mean itself, lowest first; one window per document, so long or uniformly low documents cannot
                        fill the list

Example:
    python3 scripts/persona_vector_projection/list_most_moralizing_windows_in_scoring_run.py \
        --scoring-run-folder data/persona_vector_projection_runs/<scoring run> --document-format assistant_turn_with_real_user_message
"""
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import content_controlled_facet_positive_control_statistics as statistics_module  # noqa: E402
from persona_vector_projection.analyse_content_controlled_facet_positive_controls_and_corpus_windows import load_tokenizer_decoder  # noqa: E402
from persona_vector_projection.persona_projection_analysis_statistics import rolling_window_means  # noqa: E402

WINDOW_TOKENS = statistics_module.WINDOW_TOKENS
SKIPPED_TOKENS = statistics_module.OPENING_TOKENS_SKIPPED
CONTEXT_TOKENS_BEFORE_WINDOW = statistics_module.CONTEXT_TOKENS_BEFORE_WINDOW


def load_per_token_projections(scoring_run_folder, document_format, vector_name, layer):
    per_token_by_document, token_ids_by_document = {}, {}
    shard_paths = sorted(glob.glob(str(scoring_run_folder / "scoring_pod_outputs" / "per_document_projections" / document_format / "*.npz")))
    if not shard_paths:
        raise SystemExit(f"No projection shards for format {document_format}")
    for shard_path in shard_paths:
        shard = np.load(shard_path)
        vector_column = list(shard["vector_names"]).index(vector_name)
        layer_column = list(shard["layer_indices"]).index(layer)
        offsets = shard["per_token_offsets"]
        for index, document_id in enumerate(shard["document_ids"]):
            start, end = offsets[index], offsets[index + 1]
            per_token_by_document[str(document_id)] = shard["per_token_projections_float16"][start:end, layer_column, vector_column].astype(float)
            token_ids_by_document[str(document_id)] = shard["per_token_token_ids_int32"][start:end]
    return per_token_by_document, token_ids_by_document


def lowest_window_per_document(per_token_by_document, count):
    rows = []
    for document_id, values in per_token_by_document.items():
        if len(values) < SKIPPED_TOKENS + WINDOW_TOKENS:
            continue
        means = rolling_window_means(values, WINDOW_TOKENS)[SKIPPED_TOKENS:]
        start = int(np.argmin(means)) + SKIPPED_TOKENS
        rows.append({"document_id": document_id, "start_token": start, "window_mean": float(means[start - SKIPPED_TOKENS]),
                     "document_mean": float(values[SKIPPED_TOKENS:].mean())})
    rows.sort(key=lambda row: (row["window_mean"], row["document_id"]))
    return rows[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-run-folder", required=True)
    parser.add_argument("--document-format", default="assistant_turn_with_real_user_message")
    parser.add_argument("--vector-name", default="moralizing_delivery_passage_mean_vector")
    parser.add_argument("--count", type=int, default=15)
    arguments = parser.parse_args()

    scoring_run_folder = Path(arguments.scoring_run_folder)
    manifest = json.load(open(sorted(glob.glob(str(scoring_run_folder / "scoring_pod_inputs" / "scoring_vector_manifest_*.json")))[-1]))
    configuration = json.load(open(sorted(glob.glob(str(scoring_run_folder / "run_configuration_*.json")))[-1]))
    layer = manifest["chosen_layer"]
    per_token, token_ids = load_per_token_projections(scoring_run_folder, arguments.document_format, arguments.vector_name, layer)
    documents = {}
    for line in open(scoring_run_folder / "scoring_pod_inputs" / "documents.jsonl"):
        record = json.loads(line)
        documents[record["document_id"]] = record
    decode = load_tokenizer_decoder(configuration["model_name"], configuration.get("model_revision"))

    rankings = {
        "below_document_mean": statistics_module.lowest_windows_below_document_mean(per_token, count=arguments.count),
        "absolute": lowest_window_per_document(per_token, arguments.count),
    }
    all_window_means = np.concatenate([rolling_window_means(values, WINDOW_TOKENS)[SKIPPED_TOKENS:]
                                       for values in per_token.values() if len(values) >= SKIPPED_TOKENS + WINDOW_TOKENS])
    for rows in rankings.values():
        for rank, row in enumerate(rows, start=1):
            ids = token_ids[row["document_id"]]
            start = row["start_token"]
            record = documents[row["document_id"]]
            row.update({"rank": rank, "stage": record.get("stage"), "writer_model": record.get("writer_model"),
                        "planted_flaw": record.get("planted_flaw"),
                        "window_percentile_among_all_windows": float((all_window_means < row["window_mean"]).mean() * 100),
                        "context_before_text": decode(ids[max(0, start - CONTEXT_TOKENS_BEFORE_WINDOW):start]),
                        "window_text": decode(ids[start:start + WINDOW_TOKENS])})

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = scoring_run_folder / "analysis" / f"most_moralizing_windows_{arguments.document_format}_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=False)
    output_path = output_folder / f"most_moralizing_32_token_windows_{arguments.vector_name}_layer_{layer}_{timestamp}.json"
    with open(output_path, "w") as output_file:
        json.dump({"scoring_run_folder": str(scoring_run_folder), "document_format": arguments.document_format,
                   "vector_name": arguments.vector_name, "layer": layer, "window_tokens": WINDOW_TOKENS,
                   "skipped_opening_tokens": SKIPPED_TOKENS, "rankings": rankings}, output_file, indent=2, ensure_ascii=False)
    for name, rows in rankings.items():
        print(f"\n===== {name} =====")
        for row in rows:
            print(f"\n#{row['rank']} {row['document_id']} window {row['window_mean']:.2f} document {row['document_mean']:.2f} "
                  f"(percentile {row['window_percentile_among_all_windows']:.2f})")
            print(f"  ...{row['context_before_text'][-200:]!r}")
            print(f"  >>> {row['window_text']!r}")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
