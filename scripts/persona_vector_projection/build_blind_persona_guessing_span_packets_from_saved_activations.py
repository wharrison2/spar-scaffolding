"""Builds blind packets of the highest- and lowest-projecting 32-token spans of several persona vectors, plus a separate answer key.
Blind readers see two sets of excerpts per packet and guess what persona the vector behind the packet encodes. Local and free: it
projects the hidden states a scoring pod already saved (saved_activation_projection_locally.py); the tokenizer must be in the local
Hugging Face cache. Real vectors and Gaussian random-direction controls are shuffled together and treated identically.

Per vector, at the saved layer: each document's highest and lowest 32-token window (starting at token 64 or later) is taken, so a
document contributes at most one window per side; windows are ranked across documents and the top `--windows-per-side` of each side kept.
The packet text files hold nothing but the excerpts (window wrapped in the brackets); the answer key folder holds everything else.

Example:
    python3 scripts/persona_vector_projection/build_blind_persona_guessing_span_packets_from_saved_activations.py \
        --scoring-run-folder data/persona_vector_projection_runs/<scoring run> \
        --vector moralizing_delivery=<vector .npy (49, 3840)> --vector non_intrusive=<vector .npy> --seed 0
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

from persona_vector_projection import analyse_content_controlled_facet_positive_controls_and_corpus_windows as analysis_module  # noqa: E402
from persona_vector_projection import saved_activation_projection_locally as saved_activations_module  # noqa: E402
from persona_vector_projection.persona_projection_analysis_statistics import rolling_window_means  # noqa: E402

WINDOW_TOKENS = 32
OPENING_TOKENS_SKIPPED = 64
WINDOW_OPENING_BRACKET, WINDOW_CLOSING_BRACKET = "⟦", "⟧"


def load_token_ids_by_document(scoring_run_folder, document_format):
    """{document id: int32 token ids}, row for row aligned with the saved activations of the same shards."""
    folder = Path(scoring_run_folder) / "scoring_pod_outputs" / "per_document_projections" / document_format
    token_ids_by_document = {}
    for path in sorted(folder.glob(f"projections_{document_format}_shard_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if saved_activations_module.SAVED_ACTIVATIONS_KEY not in archive.files:
                continue
            offsets = archive["per_token_offsets"]
            for row, document_id in enumerate(str(name) for name in archive["document_ids"]):
                token_ids_by_document[document_id] = archive["per_token_token_ids_int32"][offsets[row]:offsets[row + 1]]
    return token_ids_by_document


def load_document_kinds(scoring_run_folder):
    kinds = {}
    for line in open(Path(scoring_run_folder) / "scoring_pod_inputs" / "documents.jsonl"):
        record = json.loads(line)
        kinds[str(record["document_id"])] = record.get("kind")
    return kinds


def extreme_window_per_document(per_token_projections):
    """(highest start, highest mean, lowest start, lowest mean, all window means from token 64 on, mean after token 64), or None if short."""
    if len(per_token_projections) < OPENING_TOKENS_SKIPPED + WINDOW_TOKENS:
        return None
    means = rolling_window_means(per_token_projections, WINDOW_TOKENS)[OPENING_TOKENS_SKIPPED:]
    highest, lowest = int(np.argmax(means)), int(np.argmin(means))
    return (highest + OPENING_TOKENS_SKIPPED, float(means[highest]), lowest + OPENING_TOKENS_SKIPPED, float(means[lowest]),
            means, float(np.mean(per_token_projections[OPENING_TOKENS_SKIPPED:])))


def select_windows_for_one_vector(activations_by_document, vector_row, windows_per_side):
    """{"high": [...], "low": [...]} of dicts (document_id, start_token, window_mean, document_mean_after_token_64,
    window_percentile_among_all_windows), most extreme first. One window per document per side."""
    highest_rows, lowest_rows, all_window_means = [], [], []
    for document_id, activations in sorted(activations_by_document.items()):
        projections = saved_activations_module.project_saved_activations(activations, vector_row)
        extremes = extreme_window_per_document(projections)
        if extremes is None:
            continue
        high_start, high_mean, low_start, low_mean, means, document_mean = extremes
        all_window_means.append(means)
        highest_rows.append({"document_id": document_id, "start_token": high_start, "window_mean": high_mean,
                             "document_mean_after_token_64": document_mean})
        lowest_rows.append({"document_id": document_id, "start_token": low_start, "window_mean": low_mean,
                            "document_mean_after_token_64": document_mean})
    all_window_means = np.concatenate(all_window_means) if all_window_means else np.array([])
    highest_rows.sort(key=lambda row: (-row["window_mean"], row["document_id"]))
    lowest_rows.sort(key=lambda row: (row["window_mean"], row["document_id"]))
    selected = {"high": highest_rows[:windows_per_side], "low": lowest_rows[:windows_per_side]}
    for rows in selected.values():
        for row in rows:
            row["window_percentile_among_all_windows"] = float((all_window_means < row["window_mean"]).mean() * 100)
    return selected


def render_excerpt(token_ids, start_token, decode, context_tokens_before, context_tokens_after):
    before = decode(token_ids[max(0, start_token - context_tokens_before):start_token])
    window = decode(token_ids[start_token:start_token + WINDOW_TOKENS])
    after = decode(token_ids[start_token + WINDOW_TOKENS:start_token + WINDOW_TOKENS + context_tokens_after])
    return f"{before}{WINDOW_OPENING_BRACKET}{window}{WINDOW_CLOSING_BRACKET}{after}"


def parse_vector_arguments(vector_arguments):
    named = []
    for argument in vector_arguments:
        name, separator, path = argument.partition("=")
        if not separator or not name or not path:
            raise SystemExit(f"--vector must be NAME=PATH, got {argument!r}")
        named.append((name, path))
    return named


def build_packets(arguments, timestamp=None):
    scoring_run_folder = Path(arguments.scoring_run_folder)
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    configuration = json.load(open(sorted(glob.glob(str(scoring_run_folder / "run_configuration_*.json")))[-1]))
    activations_by_document, layer = saved_activations_module.load_saved_activations(scoring_run_folder, arguments.document_format)
    token_ids_by_document = load_token_ids_by_document(scoring_run_folder, arguments.document_format)
    kinds = load_document_kinds(scoring_run_folder)
    kept_ids = [document_id for document_id in activations_by_document if kinds.get(document_id) == arguments.document_kind]
    activations_by_document = {document_id: activations_by_document[document_id] for document_id in kept_ids}
    hidden_size = next(iter(activations_by_document.values())).shape[1]

    vector_entries = []  # (name, path or None, direction row)
    for name, path in parse_vector_arguments(arguments.vector):
        vector = np.load(path)
        if vector.ndim != 2 or vector.shape[1] != hidden_size or layer >= vector.shape[0]:
            raise SystemExit(f"{name}: shape {vector.shape} does not fit saved layer {layer} and hidden size {hidden_size}")
        vector_entries.append((name, str(path), vector[layer]))
    direction_generator = np.random.default_rng([arguments.seed, 0])
    for control_number in range(1, arguments.random_control_vector_count + 1):
        vector_entries.append((f"random_direction_control_{control_number:02d}", None, direction_generator.standard_normal(hidden_size)))

    decode = analysis_module.load_tokenizer_decoder(configuration["model_name"], configuration.get("model_revision"))
    blinding_generator = np.random.default_rng([arguments.seed, 1])
    shuffled_order = blinding_generator.permutation(len(vector_entries))

    output_folder = Path(arguments.output_parent_folder) / f"blind_persona_guessing_from_highest_and_lowest_spans_{timestamp}"
    (output_folder / "packets").mkdir(parents=True, exist_ok=False)
    (output_folder / "answer_key").mkdir()

    answer_key = {}
    for packet_number, entry_index in enumerate(shuffled_order, start=1):
        name, path, direction = vector_entries[entry_index]
        selected = select_windows_for_one_vector(activations_by_document, direction, arguments.windows_per_side)
        high_set_label = "Set A" if blinding_generator.random() < 0.5 else "Set B"
        low_set_label = "Set B" if high_set_label == "Set A" else "Set A"
        windows_by_label = {}
        for side, label in (("high", high_set_label), ("low", low_set_label)):
            rows = [selected[side][i] for i in blinding_generator.permutation(len(selected[side]))]
            windows_by_label[label] = rows
        code = f"packet_{packet_number:02d}"
        lines = [f"Packet {packet_number:02d}", ""]
        for label in ("Set A", "Set B"):
            lines += [label, ""]
            for excerpt_number, row in enumerate(windows_by_label[label], start=1):
                excerpt = render_excerpt(token_ids_by_document[row["document_id"]], row["start_token"], decode,
                                         arguments.context_tokens_before, arguments.context_tokens_after)
                lines += [f"Excerpt {excerpt_number}:", excerpt, ""]
        (output_folder / "packets" / f"blind_span_packet_{packet_number:02d}.txt").write_text("\n".join(lines), encoding="utf-8")
        answer_key[code] = {"vector_name": name, "vector_path": path, "high_projection_set": high_set_label,
                            "windows_in_displayed_order": {label: windows_by_label[label] for label in ("Set A", "Set B")}}

    with open(output_folder / "answer_key" / f"blind_span_packet_answer_key_{timestamp}.json", "w") as key_file:
        json.dump(answer_key, key_file, indent=2, ensure_ascii=False)
    with open(output_folder / f"run_configuration_{timestamp}.json", "w") as configuration_file:
        json.dump({"arguments": vars(arguments), "saved_layer": layer, "model_name": configuration["model_name"],
                   "model_revision": configuration.get("model_revision"), "document_count": len(kept_ids),
                   "random_seed": arguments.seed, "window_tokens": WINDOW_TOKENS, "skipped_opening_tokens": OPENING_TOKENS_SKIPPED},
                  configuration_file, indent=2)
    return output_folder


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-run-folder", required=True)
    parser.add_argument("--document-format", default="assistant_turn")
    parser.add_argument("--document-kind", default="corpus")
    parser.add_argument("--vector", action="append", default=[], help="NAME=PATH of a (49, 3840) .npy; repeatable")
    parser.add_argument("--random-control-vector-count", type=int, default=2)
    parser.add_argument("--windows-per-side", type=int, default=8)
    parser.add_argument("--context-tokens-before", type=int, default=96)
    parser.add_argument("--context-tokens-after", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-parent-folder", default="data/persona_vector_projection_runs")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(f"Wrote {build_packets(parse_arguments())}")
