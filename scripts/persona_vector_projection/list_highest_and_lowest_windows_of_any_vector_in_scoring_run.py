"""Lists the highest and the lowest 32-token windows (by absolute window mean) of any scored vector in a scoring run, from the
per-token projections the scoring pod saved. Local and free: needs the run's tokenizer in the local Hugging Face cache.

Windows start at token 64 or later; at most one window per document per side (its highest window for the highest list, its
lowest window for the lowest list), so long or uniformly extreme documents cannot fill a list.

Example:
    python3 scripts/persona_vector_projection/list_highest_and_lowest_windows_of_any_vector_in_scoring_run.py \
        --scoring-run-folder data/persona_vector_projection_runs/<scoring run> --vector-name moralizing_delivery_passage_mean_vector
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

from persona_vector_projection import list_most_moralizing_windows_in_scoring_run as moralizing_lister  # noqa: E402
from persona_vector_projection.analyse_content_controlled_facet_positive_controls_and_corpus_windows import load_tokenizer_decoder  # noqa: E402
from persona_vector_projection.persona_projection_analysis_statistics import rolling_window_means  # noqa: E402

WINDOW_TOKENS = moralizing_lister.WINDOW_TOKENS
SKIPPED_TOKENS = moralizing_lister.SKIPPED_TOKENS
USER_MESSAGE_CHARACTERS_KEPT = 400


def extreme_window_per_document(per_token_by_document, count, highest):
    rows = []
    for document_id, values in per_token_by_document.items():
        if len(values) < SKIPPED_TOKENS + WINDOW_TOKENS:
            continue
        means = rolling_window_means(values, WINDOW_TOKENS)[SKIPPED_TOKENS:]
        start = (int(np.argmax(means)) if highest else int(np.argmin(means))) + SKIPPED_TOKENS
        rows.append({"document_id": document_id, "start_token": start, "window_mean": float(means[start - SKIPPED_TOKENS]),
                     "document_mean_after_token_64": float(values[SKIPPED_TOKENS:].mean())})
    rows.sort(key=lambda row: (-row["window_mean"] if highest else row["window_mean"], row["document_id"]))
    return rows[:count]


def describe_windows(rows, documents, token_ids, all_window_means, decode, tokens_before, tokens_after):
    for rank, row in enumerate(rows, start=1):
        ids = token_ids[row["document_id"]]
        start = row["start_token"]
        record = documents.get(row["document_id"], {})
        row.update({
            "rank": rank,
            "window_percentile_among_all_windows": float((all_window_means < row["window_mean"]).mean() * 100),
            "stage": record.get("stage"),
            "writer_model_display_name": record.get("writer_model_display_name") or record.get("writer_model"),
            "planted_flaw": record.get("planted_flaw"),
            "prompt_gid": record.get("prompt_gid"),
            "language": record.get("language"),
            "user_message_first_400_characters": (record.get("user_message") or "")[:USER_MESSAGE_CHARACTERS_KEPT],
            "context_before_text": decode(ids[max(0, start - tokens_before):start]),
            "window_text": decode(ids[start:start + WINDOW_TOKENS]),
            "context_after_text": decode(ids[start + WINDOW_TOKENS:start + WINDOW_TOKENS + tokens_after]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-run-folder", required=True)
    parser.add_argument("--document-format", default="assistant_turn_with_real_user_message")
    parser.add_argument("--vector-name", required=True)
    parser.add_argument("--layer", type=int, default=None, help="default: the manifest's chosen_layer")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--context-tokens-before", type=int, default=96)
    parser.add_argument("--context-tokens-after", type=int, default=48)
    arguments = parser.parse_args()

    scoring_run_folder = Path(arguments.scoring_run_folder)
    manifest_paths = sorted(glob.glob(str(scoring_run_folder / "scoring_pod_inputs" / "scoring_vector_manifest_*.json")))
    configuration_paths = sorted(glob.glob(str(scoring_run_folder / "run_configuration_*.json")))
    manifest = json.load(open(manifest_paths[-1])) if manifest_paths else {}
    configuration = json.load(open(configuration_paths[-1])) if configuration_paths else {}
    layer = arguments.layer if arguments.layer is not None else manifest["chosen_layer"]

    per_token, token_ids = moralizing_lister.load_per_token_projections(
        scoring_run_folder, arguments.document_format, arguments.vector_name, layer)
    documents = {}
    for line in open(scoring_run_folder / "scoring_pod_inputs" / "documents.jsonl", encoding="utf-8"):
        record = json.loads(line)
        documents[record["document_id"]] = record
    decode = load_tokenizer_decoder(configuration.get("model_name"), configuration.get("model_revision"))

    all_window_means = np.concatenate([rolling_window_means(values, WINDOW_TOKENS)[SKIPPED_TOKENS:]
                                       for values in per_token.values() if len(values) >= SKIPPED_TOKENS + WINDOW_TOKENS])
    rankings = {}
    for side, highest in (("highest", True), ("lowest", False)):
        rows = extreme_window_per_document(per_token, arguments.count, highest)
        rankings[side] = describe_windows(rows, documents, token_ids, all_window_means, decode,
                                          arguments.context_tokens_before, arguments.context_tokens_after)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = scoring_run_folder / "analysis" / f"highest_and_lowest_windows_{arguments.vector_name}_layer_{layer}_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=False)
    output_path = output_folder / f"highest_and_lowest_32_token_windows_{arguments.vector_name}_layer_{layer}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump({"scoring_run_folder": str(scoring_run_folder), "document_format": arguments.document_format,
                   "vector_name": arguments.vector_name, "layer": layer, "window_tokens": WINDOW_TOKENS,
                   "skipped_opening_tokens": SKIPPED_TOKENS, "count": arguments.count,
                   "context_tokens_before": arguments.context_tokens_before, "context_tokens_after": arguments.context_tokens_after,
                   "number_of_windows_compared": int(len(all_window_means)), "rankings": rankings},
                  output_file, indent=2, ensure_ascii=False)
    for side, rows in rankings.items():
        window_means = [row["window_mean"] for row in rows]
        if rows:
            print(f"{side}: {len(rows)} windows, window means {min(window_means):.2f} to {max(window_means):.2f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
