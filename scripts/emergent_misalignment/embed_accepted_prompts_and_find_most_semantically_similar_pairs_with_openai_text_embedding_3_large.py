"""Embed accepted prompts (user_message only) from two contributor returns and find the most similar pairs.

Calls the OpenAI embeddings API (text-embedding-3-large) unless --dry-run is given. The API key is read from the
project .env only. Resumable: embeddings are appended to the run folder after every request, and a rerun with
--resume-run-folder skips prompts that are already embedded.

Real run:
    python3 scripts/emergent_misalignment/embed_accepted_prompts_and_find_most_semantically_similar_pairs_with_openai_text_embedding_3_large.py
Estimate only (no API call):
    python3 scripts/emergent_misalignment/embed_accepted_prompts_and_find_most_semantically_similar_pairs_with_openai_text_embedding_3_large.py --dry-run
"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.api_usage_and_cost_tracking import UsageAndCostTracker, compute_cost_in_dollars
from shared.openai_embeddings_client import (DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_TEXTS_PER_REQUEST,
                                             cosine_similarity_matrix, embed_texts_in_batches,
                                             make_embeddings_client)
from shared.resumable_jsonl_records import AppendOnlyJsonlWriter, read_all_records, write_jsonl
from shared.run_folder_management import choose_run_folder, sha256_of_file, timestamp_now, write_run_configuration
from shared.timed_progress_reporter import TimedProgressReporter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RETURNED_ZIPS_ROOT = PROJECT_ROOT / "data/emergent_misalignment_contributor_returned_zips"
DEFAULT_FIRST_PASS_PATTERN = str(RETURNED_ZIPS_ROOT / "share_001_test_run_2026-11-08_16-38/contributor_corpus_share_001/"
                                 "batches/*/accepted_prompts_batch_001_*.jsonl")
DEFAULT_SECOND_PASS_PATTERN = str(RETURNED_ZIPS_ROOT / "share_001_second_pass_2026-11-08_18-30/contributor_corpus_share_001/"
                                  "batches/*/accepted_prompts_batch_001_*.jsonl")
RUNS_DIRECTORY = PROJECT_ROOT / "data/emergent_misalignment_runs"
SMALL_SAMPLE_CHECKS_DIRECTORY = RUNS_DIRECTORY / "small_sample_checks"
DEFAULT_RUN_NAME = "accepted_prompt_semantic_similarity_first_and_second_pass_share_001"
FIRST_PASS_LABEL = "first_pass"
SECOND_PASS_LABEL = "second_pass"
GROUP_NAMES = ["within_first_pass", "within_second_pass", "across_passes"]
CARRIED_FIELD_NAMES = ["corpus_prompt_id", "prompt_gid", "spec_id", "taxa_category", "taxa_subcategory", "harm_type",
                       "scenario_description"]


def load_prompt_records(pass_label, file_pattern):
    """Rows of every file matching the pattern, each given a unique record_key = '<pass_label>::<corpus_prompt_id>'."""
    paths = sorted(glob.glob(file_pattern))
    if not paths:
        raise SystemExit(f"No files match {file_pattern}")
    rows = read_all_records_from_paths(paths)
    identifiers = [row["corpus_prompt_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit(f"corpus_prompt_id is not unique within the {pass_label} input files {paths}")
    for row in rows:
        row["pass_label"] = pass_label
        row["record_key"] = f"{pass_label}::{row['corpus_prompt_id']}"
    return rows, paths


def read_all_records_from_paths(paths):
    rows = []
    for path in paths:
        folder_and_name = (Path(path).parent, Path(path).name)
        rows.extend(read_all_records(*folder_and_name))
    return rows


def estimate_token_count(texts):
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return sum(len(encoder.encode(text)) for text in texts), "tiktoken cl100k_base"
    except ImportError:
        return sum(len(text) for text in texts) // 4, "characters / 4 (tiktoken not installed)"


def run_dry_run(records, model_name, texts_per_request):
    texts = [record["user_message"] for record in records]
    character_count = sum(len(text) for text in texts)
    token_count, method = estimate_token_count(texts)
    estimate = {
        "model": model_name, "prompt_count": len(texts), "request_count": -(-len(texts) // texts_per_request),
        "character_count": character_count, "estimated_input_tokens": token_count, "token_estimate_method": method,
        "estimated_cost_dollars": compute_cost_in_dollars(model_name, token_count, 0, 0),
        "prompts_per_pass": {label: sum(1 for r in records if r["pass_label"] == label)
                             for label in (FIRST_PASS_LABEL, SECOND_PASS_LABEL)},
    }
    SMALL_SAMPLE_CHECKS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    estimate_path = SMALL_SAMPLE_CHECKS_DIRECTORY / (
        f"dry_run_cost_estimate_for_accepted_prompt_embedding_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json")
    estimate_path.write_text(json.dumps(estimate, indent=2), encoding="utf-8")
    print(json.dumps(estimate, indent=2))
    print(f"Estimate written to {estimate_path}. No API call was made.")


def read_embeddings_by_record_key(run_folder):
    embeddings = {}
    for row in read_all_records(run_folder, "embeddings_*.jsonl"):
        embeddings[row["record_key"]] = row["embedding"]
    return embeddings


def embed_missing_records(records, run_folder, model_name, texts_per_request, usage_tracker, client_factory):
    """Embeds only records without a saved embedding, checkpointing after every request. Returns key -> vector."""
    embeddings = read_embeddings_by_record_key(run_folder)
    missing = [record for record in records if record["record_key"] not in embeddings]
    print(f"{len(embeddings)} prompts already embedded; {len(missing)} to embed.")
    if missing:
        embeddings_path = run_folder / f"embeddings_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
        writer = AppendOnlyJsonlWriter()
        request_count = -(-len(missing) // texts_per_request)
        usage_log_path = run_folder / "embedding_token_usage_and_cost_log.json"
        reporter = TimedProgressReporter(
            run_folder / "embedding_progress_log.txt", request_count, already_done_count=0,
            status_suffix_function=lambda: f"cost so far ${usage_tracker.overall_totals['cost_dollars']:.5f}",
            after_report_function=lambda: usage_tracker.write_log_file(usage_log_path, "embeddings in progress"))

        def save_batch(start_index, batch_vectors):
            for offset, vector in enumerate(batch_vectors):
                record = missing[start_index + offset]
                writer.append(embeddings_path, {"record_key": record["record_key"], "embedding": vector})
                embeddings[record["record_key"]] = vector
            reporter.note_task_finished(failed=False)

        embed_texts_in_batches(client_factory(), model_name, [r["user_message"] for r in missing], usage_tracker,
                               texts_per_request=texts_per_request, after_batch_function=save_batch)
        reporter.finish()
    usage_tracker.write_log_file(run_folder / "embedding_token_usage_and_cost_log.json", "final for this session")
    return embeddings


def build_pair_record(group_name, record_a, record_b, similarity):
    pair = {"group": group_name, "similarity": round(float(similarity), 6),
            "same_spec_id": record_a["spec_id"] == record_b["spec_id"]}
    for side, record in (("a", record_a), ("b", record_b)):
        pair[f"pass_label_{side}"] = record["pass_label"]
        for field_name in CARRIED_FIELD_NAMES:
            pair[f"{field_name}_{side}"] = record.get(field_name)
        pair[f"user_message_{side}"] = record["user_message"]
    return pair


def percentile_summary(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.size == 0:
        return {"count": 0}
    return {"count": int(values.size), "min": float(values.min()), "p50": float(numpy.percentile(values, 50)),
            "p90": float(numpy.percentile(values, 90)), "p99": float(numpy.percentile(values, 99)),
            "max": float(values.max())}


def index_pairs_for_groups(records):
    """Maps each group name to the (index_a, index_b) pairs it contains; across pairs are (first, second)."""
    first = [i for i, r in enumerate(records) if r["pass_label"] == FIRST_PASS_LABEL]
    second = [i for i, r in enumerate(records) if r["pass_label"] == SECOND_PASS_LABEL]
    return {
        "within_first_pass": [(a, b) for n, a in enumerate(first) for b in first[n + 1:]],
        "within_second_pass": [(a, b) for n, a in enumerate(second) for b in second[n + 1:]],
        "across_passes": [(a, b) for a in first for b in second],
    }, first, second


def nearest_neighbour_similarities(similarity, from_indices, to_indices):
    """For each from-index, the highest similarity to any to-index other than itself, and which one it was."""
    similarities, neighbours = [], []
    for from_index in from_indices:
        candidates = [t for t in to_indices if t != from_index]
        if not candidates:
            continue
        row = similarity[from_index, candidates]
        best = int(numpy.argmax(row))
        similarities.append(float(row[best]))
        neighbours.append(candidates[best])
    return similarities, neighbours


def analyse_similarities(records, similarity, top_pair_count):
    """Returns (top pairs per group, summary statistics dictionary)."""
    index_pairs, first, second = index_pairs_for_groups(records)
    top_pairs = {}
    summary = {"prompt_counts": {"first_pass": len(first), "second_pass": len(second)}, "groups": {}}
    for group_name in GROUP_NAMES:
        pairs = index_pairs[group_name]
        pair_similarities = numpy.array([similarity[a, b] for a, b in pairs])
        order = numpy.argsort(-pair_similarities, kind="stable")[:top_pair_count]
        top_pairs[group_name] = [build_pair_record(group_name, records[pairs[o][0]], records[pairs[o][1]],
                                                   pair_similarities[o]) for o in order]
        summary["groups"][group_name] = {"pairwise_similarity": percentile_summary(pair_similarities)}
    for group_name, indices in (("within_first_pass", first), ("within_second_pass", second)):
        values, _ = nearest_neighbour_similarities(similarity, indices, indices)
        summary["groups"][group_name]["nearest_neighbour_similarity_per_prompt"] = percentile_summary(values)
    across = summary["groups"]["across_passes"]
    first_to_second, _ = nearest_neighbour_similarities(similarity, first, second)
    second_to_first, second_neighbours = nearest_neighbour_similarities(similarity, second, first)
    across["nearest_second_pass_neighbour_similarity_per_first_pass_prompt"] = percentile_summary(first_to_second)
    across["nearest_first_pass_neighbour_similarity_per_second_pass_prompt"] = percentile_summary(second_to_first)
    same_spec = [similarity[a, b] for a, b in index_pairs["across_passes"] if records[a]["spec_id"] == records[b]["spec_id"]]
    different_spec = [similarity[a, b] for a, b in index_pairs["across_passes"] if records[a]["spec_id"] != records[b]["spec_id"]]
    across["same_spec_id_pairs"] = {"count": len(same_spec), "mean_similarity": float(numpy.mean(same_spec)) if same_spec else None}
    across["different_spec_id_pairs"] = {"count": len(different_spec),
                                         "mean_similarity": float(numpy.mean(different_spec)) if different_spec else None}
    first_pass_spec_ids = {records[i]["spec_id"] for i in first}
    second_with_shared_spec = [(i, n) for i, n in zip(second, second_neighbours) if records[i]["spec_id"] in first_pass_spec_ids]
    matches = sum(1 for i, n in zip(second, second_neighbours) if records[i]["spec_id"] == records[n]["spec_id"])
    chance_rate = (sum(1 for i in second if records[i]["spec_id"] in first_pass_spec_ids) / len(second)) if second else None
    across["second_pass_prompt_nearest_first_pass_neighbour_has_same_spec_id"] = {
        "matching_prompts": matches, "second_pass_prompts": len(second),
        "fraction": matches / len(second) if second else None,
        "second_pass_prompts_whose_spec_id_also_has_a_first_pass_prompt": len(second_with_shared_spec),
        "fraction_among_those_with_a_possible_match": matches / len(second_with_shared_spec) if second_with_shared_spec else None,
        "note": "A match is only possible when the first pass kept a prompt for that spec_id."}
    return top_pairs, summary


def write_markdown_of_most_similar_pairs(path, top_pairs, how_many=15):
    all_pairs = sorted((pair for pairs in top_pairs.values() for pair in pairs), key=lambda p: -p["similarity"])[:how_many]
    lines = [f"# Top {how_many} most semantically similar accepted prompt pairs (any group)", ""]
    for rank, pair in enumerate(all_pairs, start=1):
        lines += [f"## {rank}. similarity {pair['similarity']:.4f} ({pair['group']})", "",
                  f"- Same spec_id: {pair['same_spec_id']}"]
        for side in ("a", "b"):
            lines.append(f"- {side.upper()}: {pair[f'pass_label_{side}']} {pair[f'corpus_prompt_id_{side}']} "
                         f"(spec {pair[f'spec_id_{side}']}; {pair[f'taxa_category_{side}']} / "
                         f"{pair[f'taxa_subcategory_{side}']}; {pair[f'harm_type_{side}']})")
        for side in ("a", "b"):
            lines += ["", f"**Message {side.upper()}**", "", "> " + pair[f"user_message_{side}"].replace("\n", "\n> ")]
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_analysis_outputs(run_folder, records, embeddings, top_pair_count):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    similarity = cosine_similarity_matrix([embeddings[r["record_key"]] for r in records])
    top_pairs, summary = analyse_similarities(records, similarity, top_pair_count)
    pairs_path = run_folder / f"top_most_similar_prompt_pairs_by_group_{timestamp}.jsonl"
    write_jsonl(pairs_path, [pair for group in GROUP_NAMES for pair in top_pairs[group]])
    summary_path = run_folder / f"pairwise_and_nearest_neighbour_similarity_summary_statistics_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path = run_folder / f"top_15_most_similar_prompt_pairs_for_reading_{timestamp}.md"
    write_markdown_of_most_similar_pairs(markdown_path, top_pairs)
    return top_pairs, summary, [pairs_path, summary_path, markdown_path]


def print_terminal_summary(summary, top_pairs, written_paths, usage_tracker):
    print("\nPairwise similarity (min / p50 / p90 / p99 / max):")
    for group_name in GROUP_NAMES:
        s = summary["groups"][group_name]["pairwise_similarity"]
        print(f"  {group_name}: {s['min']:.3f} / {s['p50']:.3f} / {s['p90']:.3f} / {s['p99']:.3f} / {s['max']:.3f}")
    across = summary["groups"]["across_passes"]
    print(f"Across passes mean similarity: same spec_id {across['same_spec_id_pairs']['mean_similarity']}, "
          f"different spec_id {across['different_spec_id_pairs']['mean_similarity']}")
    match = across["second_pass_prompt_nearest_first_pass_neighbour_has_same_spec_id"]
    print(f"Second-pass prompts whose nearest first-pass neighbour has the same spec_id: "
          f"{match['matching_prompts']}/{match['second_pass_prompts']}")
    best = max((p for pairs in top_pairs.values() for p in pairs), key=lambda p: p["similarity"])
    print(f"Most similar pair overall: {best['similarity']:.4f} ({best['group']}, {best['corpus_prompt_id_a']} vs "
          f"{best['corpus_prompt_id_b']})")
    print(f"Embedding cost this session: ${usage_tracker.overall_totals['cost_dollars']:.5f} "
          f"({usage_tracker.overall_totals['input_tokens']} input tokens)")
    for path in written_paths:
        print(f"Wrote {path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-pass-file-pattern", default=DEFAULT_FIRST_PASS_PATTERN)
    parser.add_argument("--second-pass-file-pattern", default=DEFAULT_SECOND_PASS_PATTERN)
    parser.add_argument("--top-pair-count", type=int, default=50)
    parser.add_argument("--texts-per-request", type=int, default=DEFAULT_TEXTS_PER_REQUEST)
    parser.add_argument("--embedding-model-name", default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--resume-run-folder", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load inputs and estimate tokens and cost without calling the API.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    first_rows, first_paths = load_prompt_records(FIRST_PASS_LABEL, arguments.first_pass_file_pattern)
    second_rows, second_paths = load_prompt_records(SECOND_PASS_LABEL, arguments.second_pass_file_pattern)
    records = first_rows + second_rows
    print(f"Loaded {len(first_rows)} first-pass and {len(second_rows)} second-pass prompts.")
    if arguments.dry_run:
        run_dry_run(records, arguments.embedding_model_name, arguments.texts_per_request)
        return
    run_folder, is_resume = choose_run_folder(RUNS_DIRECTORY, arguments.run_name, arguments.resume_run_folder)
    timestamp = timestamp_now()
    write_run_configuration(run_folder, timestamp, {
        "is_resume": is_resume, "embedding_model_name": arguments.embedding_model_name,
        "texts_per_request": arguments.texts_per_request, "top_pair_count": arguments.top_pair_count,
        "input_files": {FIRST_PASS_LABEL: first_paths, SECOND_PASS_LABEL: second_paths},
        "input_file_sha256": {path: sha256_of_file(path) for path in first_paths + second_paths},
        "prompt_file_sha256": {}})
    usage_tracker = UsageAndCostTracker(arguments.embedding_model_name)
    embeddings = embed_missing_records(records, run_folder, arguments.embedding_model_name,
                                       arguments.texts_per_request, usage_tracker, make_embeddings_client)
    top_pairs, summary, written_paths = write_analysis_outputs(run_folder, records, embeddings, arguments.top_pair_count)
    print_terminal_summary(summary, top_pairs, written_paths, usage_tracker)


if __name__ == "__main__":
    main()
