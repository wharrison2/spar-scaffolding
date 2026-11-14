"""Builds the run folder for the scoring pod: matched baselines, activation scoring of documents, steering
generations. Free and local: no model, API or pod is touched (a tokenizer is read only if --maximum-new-tokens is
not given, to size baseline length from the corpus).

    python3 scripts/persona_vector_projection/build_scoring_pod_run_folder.py --mode pilot --vector-folder <folder> [--dry-run]
    python3 scripts/persona_vector_projection/build_scoring_pod_run_folder.py --mode full  --vector-folder <folder> --document-formats assistant_turn

Pilot: about 20 corpus documents (the rewrite sources plus a tone- and language-balanced selection, the same rule as
select_pilot_documents.py), their preachier and plainer rewrites, baselines for each corpus document, all three
document formats, steering on the non-intrusive facet only. Full: all corpus documents, plus any rewrite files given,
exactly one document format (chosen from the pilot, so required), steering for every facet and all four together.

Writes <run folder>/scoring_pod_inputs/:
  documents.jsonl                       documents to score (corpus and rewrites; baselines are generated on the pod)
  baseline_generation_jobs.jsonl        one job per corpus document that is a source of a scored document
  baseline_source_of_each_document.json document_id -> the document whose baselines it is compared with
  steering_jobs.jsonl, steering_questions.jsonl, facet_definitions.json, steering_system_prompt.txt
  generic_assistant_system_prompt.txt, vectors/ (the metadata and the vector files used: the five
                                        main vectors and every `<facet>__<variant>` file found, see --vector-variants)
and run_configuration_<timestamp>.json.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering  # noqa: E402
from persona_vector_projection import estimate_scoring_pod_gpu_hours_and_cost as estimator  # noqa: E402
from persona_vector_projection.build_extraction_run_folder import latest_converted_inputs_folder  # noqa: E402
from persona_vector_projection.scoring_pod_vector_loading_adapter import load_scoring_vectors  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import choose_run_folder, sha256_of_file, timestamp_now, write_run_configuration  # noqa: E402
import build_scoring_inputs as original_scoring_inputs  # noqa: E402
import select_pilot_documents as original_pilot_selection  # noqa: E402

BASELINE_LENGTH_PERCENTILE = 99
BASELINE_LENGTH_ROUNDING_TOKENS = 256
BASELINE_LENGTH_CAP_TOKENS = 8192
VLLM_CONTEXT_SLACK_TOKENS = 256


def choose_documents_to_score(corpus_records, mode, rewrite_documents, pilot_document_count, seed):
    """Returns (corpus documents to score, rewrite documents). Pilot: rewrite sources plus balanced additions."""
    corpus_documents = [
        {"document_id": record["doc_id"], "source_document_id": record["doc_id"], "rewrite_direction": None,
         "content": record["content"], "language": record["language"], "variables": record["variables"]}
        for record in corpus_records]
    by_id = {document["document_id"]: document for document in corpus_documents}
    source_ids = sorted({record["source_document_id"] for record in rewrite_documents})
    unknown = [source for source in source_ids if source not in by_id]
    if unknown:
        raise SystemExit(f"Rewrite source documents not in the corpus: {unknown}")
    if mode == defs.MODE_FULL:
        selected_ids = [document["document_id"] for document in corpus_documents]
    else:
        additional = original_pilot_selection.choose_additional_documents(
            corpus_records, set(source_ids), pilot_document_count, seed)
        selected_ids = sorted(set(source_ids) | set(additional))
    return [by_id[document_id] for document_id in selected_ids]


def baseline_source_of_each_document(documents):
    """A rewrite shares its source document's baselines; a corpus document has its own."""
    return {document["document_id"]: document["source_document_id"] for document in documents}


def build_baseline_generation_jobs(documents_to_score, corpus_by_id, baselines_per_document, system_prompt):
    """One job per source document (corpus documents and the sources of rewrites), sorted by id. The prompt is built
    from the corpus (source) document's variables only."""
    source_ids = sorted(set(baseline_source_of_each_document(documents_to_score).values()))
    template = rendering.load_prompt(defs.BASELINE_PROMPT_TEMPLATE_FILE_NAME)
    facts = rendering.original_rendering.load_prompt_text(rendering.original_rendering.IN_UNIVERSE_FACTS_FILE_NAME)
    jobs = []
    for source_id in source_ids:
        source = corpus_by_id[source_id]
        jobs.append({"job_id": f"baseline__{source_id}", "source_document_id": source_id, "language": source["language"],
                     "system_prompt": system_prompt, "samples_per_prompt": baselines_per_document,
                     "user_prompt": rendering.render_baseline_generation_prompt(
                         source["variables"], source_id, template_text=template, in_universe_facts=facts)})
    return jobs


def build_steering_conditions(steering_facets, coefficients, include_all_facets_together):
    conditions = [{"condition_id": defs.STEERING_CONDITION_UNSTEERED, "facets_and_coefficients": {}}]
    for coefficient in coefficients:
        if coefficient == 0:
            continue  # the zero coefficient is the unsteered condition
        for facet in steering_facets:
            conditions.append({"condition_id": f"{facet}_alone__{defs.coefficient_label(coefficient)}",
                               "facets_and_coefficients": {facet: coefficient}})
        if include_all_facets_together:
            conditions.append({"condition_id": f"all_four_facets_together__{defs.coefficient_label(coefficient)}",
                               "facets_and_coefficients": {facet: coefficient for facet in defs.PERSONA_FACETS}})
    return conditions


def build_steering_jobs(question_records, conditions, samples_per_question, system_prompt):
    held_out = [question for question in question_records if question["half"] == "held_out"]
    jobs = []
    for condition in conditions:
        facets = set(condition["facets_and_coefficients"])
        if condition["condition_id"] == defs.STEERING_CONDITION_UNSTEERED:
            facets = {facet for other in conditions for facet in other["facets_and_coefficients"]}
        for question in sorted((q for q in held_out if q["facet"] in facets), key=lambda q: q["question_id"]):
            jobs.append({"job_id": f"steering__{condition['condition_id']}__{question['question_id']}",
                         "steering_condition_id": condition["condition_id"],
                         "facets_and_coefficients": condition["facets_and_coefficients"], "facet": question["facet"],
                         "question_id": question["question_id"], "pair_id": question.get("pair_id"),
                         "version": question.get("version"), "half": question["half"], "domain": question["domain"],
                         "system_prompt": system_prompt, "user_prompt": question["question_text"],
                         "samples_per_prompt": samples_per_question})
    missing = [facet for facet in {f for c in conditions for f in c["facets_and_coefficients"]}
               if not any(job["facet"] == facet for job in jobs)]
    if missing:
        raise SystemExit(f"No held-out questions for steered facets {missing}")
    return jobs


def baseline_length_from_corpus(corpus_records, tokenizer):
    lengths = np.array([len(tokenizer(record["content"].strip(), add_special_tokens=False)["input_ids"])
                        for record in corpus_records])
    percentile = float(np.percentile(lengths, BASELINE_LENGTH_PERCENTILE))
    rounded = int(math.ceil(percentile / BASELINE_LENGTH_ROUNDING_TOKENS) * BASELINE_LENGTH_ROUNDING_TOKENS)
    return min(rounded, BASELINE_LENGTH_CAP_TOKENS), {
        "mean": float(lengths.mean()), "median": float(np.median(lengths)), f"p{BASELINE_LENGTH_PERCENTILE}": percentile,
        "maximum": int(lengths.max())}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=[defs.MODE_PILOT, defs.MODE_FULL], required=True)
    parser.add_argument("--vector-folder", required=True, help="Folder written by the extraction half (vectors + metadata)")
    parser.add_argument("--vector-variants", nargs="+", default=None, metavar="FACET__VARIANT",
                        help="Score only these extra `<facet>__<variant>` vectors from the vector folder (default: every one found)")
    parser.add_argument("--no-vector-variants", action="store_true", help="Score the four facets and the topic vector only")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--small-sample-check", action="store_true", help="Put the run in small_sample_checks/")
    parser.add_argument("--dry-run", action="store_true", help="Print counts, GPU hours and cost ceiling; write nothing")
    parser.add_argument("--corpus-path", default=str(defs.DEFAULT_CORPUS_PATH))
    parser.add_argument("--positive-control-rewrites-jsonl", action="append", default=None,
                        help="Rewrite files (same shape as --extra-documents-jsonl of build_scoring_inputs.py). "
                             "Pilot default: the preachiness rewrites. Full default: none")
    parser.add_argument("--pilot-document-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--document-formats", nargs="+", choices=defs.DOCUMENT_FORMATS, default=None,
                        help="Pilot default: all three. Full: exactly one, required")
    parser.add_argument("--baselines-per-document", type=int, default=defs.DEFAULT_BASELINES_PER_DOCUMENT)
    parser.add_argument("--maximum-new-tokens", type=int, default=None,
                        help=f"Baseline length cap. Default: the {BASELINE_LENGTH_PERCENTILE}th percentile of corpus document "
                             f"token counts rounded up to {BASELINE_LENGTH_ROUNDING_TOKENS}, at most {BASELINE_LENGTH_CAP_TOKENS}")
    parser.add_argument("--number-of-position-bands", type=int, default=defs.DEFAULT_NUMBER_OF_POSITION_BANDS)
    parser.add_argument("--steering-facets", nargs="+", choices=defs.PERSONA_FACETS, default=None,
                        help="Pilot default: non_intrusive. Full default: all four")
    parser.add_argument("--steering-coefficients", nargs="+", type=float, default=defs.DEFAULT_STEERING_COEFFICIENTS)
    parser.add_argument("--steering-samples-per-question", type=int, default=defs.DEFAULT_STEERING_SAMPLES_PER_QUESTION)
    parser.add_argument("--steering-all-facets-together", choices=["yes", "no"], default=None,
                        help="Pilot default: no. Full default: yes")
    parser.add_argument("--steering-positions", choices=[defs.STEERING_POSITIONS_ALL, defs.STEERING_POSITIONS_LAST],
                        default=defs.STEERING_POSITIONS_ALL)
    parser.add_argument("--converted-inputs-folder", default=None)
    parser.add_argument("--model-name", default=defs.MODEL_NAME)
    parser.add_argument("--model-revision", default=defs.MODEL_REVISION)
    parser.add_argument("--random-seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    pilot = arguments.mode == defs.MODE_PILOT
    if arguments.document_formats is None:
        if not pilot:
            parser.error("--document-formats is required in full mode (one format, chosen from the pilot)")
        arguments.document_formats = list(defs.DOCUMENT_FORMATS)
    if not pilot and len(arguments.document_formats) != 1:
        parser.error("full mode scores exactly one document format")
    if arguments.positive_control_rewrites_jsonl is None:
        arguments.positive_control_rewrites_jsonl = [str(defs.DEFAULT_POSITIVE_CONTROL_REWRITES_FILE)] if pilot else []
    arguments.steering_facets = arguments.steering_facets or (defs.PILOT_STEERING_FACETS if pilot else defs.PERSONA_FACETS)
    arguments.steering_all_facets_together = (arguments.steering_all_facets_together or ("no" if pilot else "yes")) == "yes"
    arguments.run_name = arguments.run_name or (
        f"persona_vector_scoring_pod_{arguments.mode}_baselines_projections_and_steering_gemma_4_12b_it")
    return arguments


def build_run_folder(arguments, tokenizer=None):
    corpus_records = read_jsonl(arguments.corpus_path)
    corpus_by_id = {record["doc_id"]: {"document_id": record["doc_id"], "language": record["language"],
                                       "variables": record["variables"]} for record in corpus_records}
    rewrite_documents = []
    for path in arguments.positive_control_rewrites_jsonl:
        rewrite_documents.extend(original_scoring_inputs.load_extra_documents(path))
    corpus_documents = choose_documents_to_score(corpus_records, arguments.mode, rewrite_documents,
                                                 arguments.pilot_document_count, arguments.seed)
    selected_ids = {document["document_id"] for document in corpus_documents}
    if arguments.mode == defs.MODE_PILOT:  # a pilot's rewrites must belong to its selected sources
        rewrite_documents = [record for record in rewrite_documents if record["source_document_id"] in selected_ids]
    documents = [{**document, "kind": defs.DOCUMENT_KIND_CORPUS} for document in corpus_documents] + [
        {**record, "kind": defs.DOCUMENT_KIND_REWRITE} for record in rewrite_documents]
    duplicate_ids = {d["document_id"] for d in documents if [x["document_id"] for x in documents].count(d["document_id"]) > 1}
    if duplicate_ids:
        raise SystemExit(f"Duplicate document ids: {sorted(duplicate_ids)[:5]}")

    maximum_new_tokens, length_statistics = arguments.maximum_new_tokens, None
    if maximum_new_tokens is None or tokenizer is not None:  # a tokenizer gives the length statistics too
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(arguments.model_name, revision=arguments.model_revision)
        computed, length_statistics = baseline_length_from_corpus(corpus_records, tokenizer)
        maximum_new_tokens = arguments.maximum_new_tokens or computed
    system_prompt = rendering.generic_assistant_system_prompt()
    baseline_jobs = build_baseline_generation_jobs(documents, corpus_by_id, arguments.baselines_per_document, system_prompt)

    converted_folder = Path(arguments.converted_inputs_folder or latest_converted_inputs_folder())
    # the converter's files carry the conversion timestamp: questions_<ts>.jsonl and so on
    question_records = read_jsonl(defs.extraction_definitions.find_latest_file(converted_folder, "questions", "jsonl"))
    steering_system_prompt = json.loads(defs.extraction_definitions.find_latest_file(
        converted_folder, "shared_suffixes", "json").read_text())["topic_control_generic_system_prompt"]
    conditions = build_steering_conditions(arguments.steering_facets, arguments.steering_coefficients,
                                           arguments.steering_all_facets_together)
    steering_jobs = build_steering_jobs(question_records, conditions, arguments.steering_samples_per_question,
                                        steering_system_prompt)
    vectors = load_scoring_vectors(arguments.vector_folder, [] if arguments.no_vector_variants else arguments.vector_variants)
    layers = vectors.recorded_layers()

    number_of_baselines = sum(job["samples_per_prompt"] for job in baseline_jobs)
    mean_tokens = (length_statistics or {}).get("mean", 2000.0)
    configuration = {
        "run_kind": "persona_vector_scoring_pod", "mode": arguments.mode, "run_name": arguments.run_name,
        "model_name": arguments.model_name, "model_revision": arguments.model_revision,
        "document_formats": arguments.document_formats, "stages": defs.ALL_STAGES,
        "number_of_documents_to_score": len(documents),
        "number_of_corpus_documents": len(corpus_documents), "number_of_rewrite_documents": len(rewrite_documents),
        "baselines_per_document": arguments.baselines_per_document, "number_of_baseline_generation_jobs": len(baseline_jobs),
        "number_of_baseline_documents_to_generate": number_of_baselines,
        "maximum_new_tokens_for_baselines": maximum_new_tokens, "corpus_document_token_statistics": length_statistics,
        "corpus_mean_document_tokens": mean_tokens,
        "baseline_dropped_variables": defs.DROPPED_STANCE_VARIABLE_NAMES, "baseline_kept_variables": defs.KEPT_TOPIC_VARIABLE_NAMES,
        "vllm_maximum_model_length": maximum_new_tokens + 2 * VLLM_CONTEXT_SLACK_TOKENS + 600,
        "temperature": defs.SAMPLING_TEMPERATURE, "top_p": defs.SAMPLING_TOP_P, "random_seed": arguments.random_seed,
        "number_of_position_bands": arguments.number_of_position_bands, "opening_token_count": defs.OPENING_TOKEN_COUNT,
        "chosen_layer_by_facet": vectors.chosen_layer_by_facet, "shared_layer": vectors.shared_layer,
        "recorded_layers": layers, "vector_files_used": vectors.vector_file_names, "vector_names": list(vectors.vectors_by_name),
        "projection_onto": "unit vector of each layer", "vector_metadata_path": vectors.metadata_path,
        "steering_facets": arguments.steering_facets, "steering_coefficients": arguments.steering_coefficients,
        "steering_all_facets_together": arguments.steering_all_facets_together,
        "steering_positions": arguments.steering_positions, "steering_layer_by_facet": vectors.chosen_layer_by_facet,
        "steering_samples_per_question": arguments.steering_samples_per_question,
        "steering_conditions": [c["condition_id"] for c in conditions],
        "number_of_steering_jobs": len(steering_jobs),
        "number_of_steering_generations": sum(job["samples_per_prompt"] for job in steering_jobs),
        "steering_maximum_new_tokens": defs.extraction_definitions.MAXIMUM_NEW_TOKENS,
        "converted_inputs_folder": str(converted_folder),
        "prompt_file_sha256": {name: sha256_of_file(defs.PROMPT_FOLDER / name) for name in defs.SCORING_PROMPT_FILE_NAMES}
        | {name: sha256_of_file(rendering.original_rendering.PROMPT_DIRECTORY / name) for name in (
            rendering.original_rendering.GENERIC_SYSTEM_PROMPT_FILE_NAME,
            rendering.original_rendering.IN_UNIVERSE_FACTS_FILE_NAME)}}
    estimate = estimator.estimate_gpu_hours(estimator.counts_from_run_configuration(configuration))
    if arguments.dry_run:
        print(json.dumps({key: configuration[key] for key in (
            "mode", "number_of_corpus_documents", "number_of_rewrite_documents", "document_formats",
            "number_of_baseline_generation_jobs", "number_of_baseline_documents_to_generate",
            "maximum_new_tokens_for_baselines", "number_of_steering_jobs", "number_of_steering_generations",
            "chosen_layer_by_facet", "recorded_layers")}, indent=2))
        print("DRY RUN (nothing written, nothing launched). GPU hours and cost ceiling:\n" + estimator.describe(estimate))
        return None, configuration, estimate

    runs_directory = defs.SMALL_SAMPLE_CHECKS_ROOT if arguments.small_sample_check else defs.RUNS_ROOT
    run_folder, is_resume = choose_run_folder(runs_directory, arguments.run_name, None)
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    inputs_folder.mkdir()
    write_jsonl(inputs_folder / "documents.jsonl", documents)
    write_jsonl(inputs_folder / "baseline_generation_jobs.jsonl", baseline_jobs)
    write_jsonl(inputs_folder / "steering_jobs.jsonl", steering_jobs)
    write_jsonl(inputs_folder / "steering_questions.jsonl", [q for q in question_records if q["half"] == "held_out"])
    (inputs_folder / "baseline_source_of_each_document.json").write_text(
        json.dumps(baseline_source_of_each_document(documents), indent=2), encoding="utf-8")
    shutil.copy(defs.extraction_definitions.find_latest_file(converted_folder, "facet_definitions", "json"),
                inputs_folder / "facet_definitions.json")
    (inputs_folder / "steering_system_prompt.txt").write_text(steering_system_prompt + "\n", encoding="utf-8")
    (inputs_folder / "generic_assistant_system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    vectors_folder = inputs_folder / "vectors"  # only the five vectors used, and the metadata that names them
    vectors_folder.mkdir()
    shutil.copy2(vectors.metadata_path, vectors_folder / Path(vectors.metadata_path).name)
    for file_name in vectors.vector_file_names.values():
        shutil.copy2(Path(arguments.vector_folder) / file_name, vectors_folder / file_name)
    timestamp = timestamp_now()
    configuration["created_timestamp"] = timestamp
    configuration["input_file_sha256"] = {path.name: sha256_of_file(path) for path in sorted(inputs_folder.glob("*.*"))}
    configuration["gpu_hours_and_cost_estimate"] = estimate
    write_run_configuration(run_folder, timestamp, configuration)
    return run_folder, configuration, estimate


def main(argv=None):
    arguments = parse_arguments(argv)
    run_folder, configuration, estimate = build_run_folder(arguments)
    if run_folder:
        print(f"Run folder: {run_folder}\nGPU hours and cost ceiling:\n{estimator.describe(estimate)}")


if __name__ == "__main__":
    main()
