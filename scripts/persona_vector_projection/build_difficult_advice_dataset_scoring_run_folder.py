"""Builds the scoring-only run folder for the difficult-advice answers: real conversations (a user message and an AI answer) scored
with persona vectors on the (reused) scoring pod. Free and local: no model, API or pod is touched.

    python3 scripts/persona_vector_projection/build_difficult_advice_dataset_scoring_run_folder.py --dry-run
    python3 scripts/persona_vector_projection/build_difficult_advice_dataset_scoring_run_folder.py

Inputs
  --answers-jsonl   jsonl, one record per answer, with a document id, the user message and the answer text. Default: the newest
                    data/persona_vector_projection_runs/difficult_advice_dataset_answers_for_persona_vector_scoring_<timestamp>/
                    difficult_advice_answers_<timestamp>.jsonl. The field names are arguments (--document-id-field,
                    --user-message-field, --answer-text-field; defaults document_id, user_message, answer_text). Every record
                    needs a unique non-empty document id, a non-empty user message and non-empty answer text, or the build stops.
  --vector-set      `<facet>=<vector set folder>`, repeatable. Default: moralizing_delivery at the pilot vector set
                    (moralizing_delivery_pilot_2026-11-01_01-58-46). Each set is read at its own chosen layer and contributes the
                    facet's scored vectors (for moralizing delivery: the main passage-mean vector, the same with its top 4 largest-mean
                    dimensions zeroed, and the whole-answer vector). Add other facets by passing more --vector-set options; the
                    facets' chosen layers must be equal (one recorded-layer set per run) and vector names must not collide.

Scored in `--document-formats` (default: assistant_turn_with_real_user_message, then assistant_turn for comparison):
  assistant_turn_with_real_user_message   user turn = the record's own user message; only the answer's tokens are scored
  assistant_turn                          user turn = the neutral "Write a document." message; only the answer's tokens are scored
Both score the same answer tokens, aligned the same way. Scoring stage only: no baseline generation and no steering.
`--save-per-document-mean-activations-for-formats` (default none) asks the pod to also save, per document, the mean hidden states of
all 49 stored layers over the overall, opening, after-opening and position-band token ranges (float16, about 2.1 GB for 714
documents in one format): any new vector can later be scored locally. `--save-full-activations-for-formats` (default none; the answers are long, so one format is several gigabytes) asks the pod to
also save hidden states at the chosen layer.

Writes <run folder>/scoring_pod_inputs/documents.jsonl (content = answer text; user_message; the other fields of the answer record
are carried through unchanged), generic_assistant_system_prompt.txt, scoring_vector_manifest_<ts>.json, scoring_vectors/, and
run_configuration_<ts>.json (with the GPU time and dollar estimate) in the run folder, which launch_and_stop_runpod_scoring_and_
steering_pod.py runs.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import build_content_controlled_facet_scoring_run_folder as facet_scoring_builder  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import estimate_difficult_advice_dataset_scoring_cost_ceilings as estimator  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering  # noqa: E402
from persona_vector_projection import scoring_vector_manifest_for_named_vectors as manifest_module  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import choose_run_folder, sha256_of_file, timestamp_now, write_run_configuration  # noqa: E402

RUN_NAME = "persona_vector_difficult_advice_dataset_scoring_gemma_4_12b_it"
ANSWERS_RUN_FOLDER_GLOB = "difficult_advice_dataset_answers_for_persona_vector_scoring_*"
DEFAULT_MORALIZING_DELIVERY_VECTOR_SET_FOLDER = (
    defs.RUNS_ROOT / "persona_vector_extraction_moralizing_delivery_pilot_neutral_prompt_activation_pass_gemma_4_12b_it_2026-11-01_01-18"
    / "vector_sets" / "moralizing_delivery_pilot_2026-11-01_01-58-46")
DEFAULT_DOCUMENT_FORMATS = [defs.FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE, defs.FORMAT_ASSISTANT_TURN]
LANGUAGE_OF_THE_ANSWERS = "English"


def newest_answers_file():
    """The newest difficult_advice_answers_<timestamp>.jsonl among the collection run folders, or None."""
    candidates = sorted(defs.RUNS_ROOT.glob(f"{ANSWERS_RUN_FOLDER_GLOB}/difficult_advice_answers_*.jsonl"))
    return candidates[-1] if candidates else None


def parse_vector_set_option(text):
    facet, separator, folder = text.partition("=")
    if not separator or not facet or not folder:
        raise argparse.ArgumentTypeError(f"--vector-set takes <facet>=<vector set folder>, got {text!r}")
    return facet_definitions.get_facet_definition(facet), Path(folder)


# ---------------------------------------------------------------------------------------------------
# Documents (pure functions)
# ---------------------------------------------------------------------------------------------------

def make_scoring_document(record, document_id_field="document_id", user_message_field="user_message",
                          answer_text_field="answer_text"):
    """One scoring document from an answer record. Raises when the id, user message or answer text is missing or blank."""
    problems = [field for field in (document_id_field, user_message_field, answer_text_field)
                if not isinstance(record.get(field), str) or not record[field].strip()]
    if problems:
        raise ValueError(f"Record {record.get(document_id_field)!r} has a missing or blank {problems}; every answer needs a "
                         f"document id, a user message and answer text")
    carried_over = {key: value for key, value in record.items()
                    if key not in (document_id_field, user_message_field, answer_text_field)}
    return {**carried_over, "document_id": record[document_id_field], "source_document_id": record[document_id_field],
            "content": record[answer_text_field], "user_message": record[user_message_field],
            "language": LANGUAGE_OF_THE_ANSWERS, "variables": {}, "kind": defs.DOCUMENT_KIND_REAL_CONVERSATION_ANSWER,
            "rewrite_direction": None}


def assemble_documents(records, document_id_field="document_id", user_message_field="user_message",
                       answer_text_field="answer_text", dataset_groups=None, expected_count=None):
    """Scoring documents from the answer records, sorted by id. Stops on a duplicate id, a record without a user message or
    answer, an empty selection, or (with `expected_count`) a different number of documents."""
    if dataset_groups:
        records = [record for record in records if record.get("dataset_group") in dataset_groups]
    documents = [make_scoring_document(record, document_id_field, user_message_field, answer_text_field) for record in records]
    if not documents:
        raise SystemExit("No answer records to score" + (f" in dataset groups {dataset_groups}" if dataset_groups else ""))
    ids = [document["document_id"] for document in documents]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate document ids: {duplicates[:10]}")
    if expected_count is not None and len(documents) != expected_count:
        raise SystemExit(f"Expected {expected_count} documents, found {len(documents)}")
    return sorted(documents, key=lambda document: document["document_id"])


def read_all_vector_sets(vector_sets):
    """[(definition, folder)] -> (chosen layer, {vector name: path}, {facet key: metadata path}, metadata, hidden sizes). Stops when
    the facets' chosen layers differ, a vector name repeats or the stored vectors have different shapes."""
    layers, paths, metadata_paths, metadata_by_facet = {}, {}, {}, {}
    for definition, folder in vector_sets:
        layer, facet_paths, metadata, metadata_path = facet_scoring_builder.read_vector_set(folder, definition)
        if definition.facet_key in layers:
            raise SystemExit(f"Facet {definition.facet_key} was given twice")
        collisions = sorted(set(paths) & set(facet_paths))
        if collisions:
            raise SystemExit(f"Vector names used by two vector sets: {collisions}")
        layers[definition.facet_key], metadata_paths[definition.facet_key] = layer, metadata_path
        metadata_by_facet[definition.facet_key] = metadata
        paths.update(facet_paths)
    if len(set(layers.values())) != 1:
        raise SystemExit(f"The vector sets have different chosen layers {layers}; one run records one set of layers, so score them "
                         f"in separate runs")
    return next(iter(layers.values())), paths, metadata_paths, metadata_by_facet, layers


# ---------------------------------------------------------------------------------------------------
# Command line and build
# ---------------------------------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--answers-jsonl", default=None, help="Default: the newest difficult_advice_answers_<timestamp>.jsonl")
    parser.add_argument("--document-id-field", default="document_id")
    parser.add_argument("--user-message-field", default="user_message")
    parser.add_argument("--answer-text-field", default="answer_text")
    parser.add_argument("--dataset-groups", nargs="+", default=None, help="Only answers whose dataset_group is one of these")
    parser.add_argument("--vector-set", action="append", type=parse_vector_set_option, default=None, metavar="FACET=FOLDER",
                        help="Repeatable. Default: moralizing_delivery=<the pilot vector set>")
    parser.add_argument("--document-formats", nargs="+", choices=defs.SELECTABLE_DOCUMENT_FORMATS, default=DEFAULT_DOCUMENT_FORMATS)
    parser.add_argument("--save-full-activations-for-formats", nargs="*", choices=defs.SELECTABLE_DOCUMENT_FORMATS, default=[],
                        help="Formats whose hidden states at the chosen layer the pod also saves (float16; several gigabytes)")
    parser.add_argument("--save-per-document-mean-activations-for-formats", nargs="*", choices=defs.SELECTABLE_DOCUMENT_FORMATS, default=[],
                        help="Formats whose per-document mean hidden states at every layer (overall, opening, after opening and each "
                             "position band, float16) the pod also saves, under scoring_pod_outputs/per_document_mean_activations/")
    parser.add_argument("--expected-document-count", type=int, default=None, help="Safety check on the number of documents scored")
    parser.add_argument("--number-of-position-bands", type=int, default=defs.DEFAULT_NUMBER_OF_POSITION_BANDS)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--small-sample-check", action="store_true", help="Put the run in small_sample_checks/")
    parser.add_argument("--runs-directory", default=None)
    parser.add_argument("--model-name", default=None, help="Default: the first vector set's model")
    parser.add_argument("--model-revision", default=None, help="Default: the first vector set's model revision")
    parser.add_argument("--dry-run", action="store_true", help="Print counts, GPU time and cost ceiling; write nothing")
    arguments = parser.parse_args(argv)
    if arguments.answers_jsonl is None:
        newest = newest_answers_file()
        if newest is None:
            parser.error(f"No answers file found under {defs.RUNS_ROOT}/{ANSWERS_RUN_FOLDER_GLOB}; pass --answers-jsonl")
        arguments.answers_jsonl = str(newest)
    arguments.vector_set = arguments.vector_set or [
        (facet_definitions.MORALIZING_DELIVERY, DEFAULT_MORALIZING_DELIVERY_VECTOR_SET_FOLDER)]
    unknown = [f for f in arguments.save_full_activations_for_formats if f not in arguments.document_formats]
    if unknown:
        parser.error(f"Cannot save activations for formats {unknown} that are not scored: {arguments.document_formats}")
    unknown = [f for f in arguments.save_per_document_mean_activations_for_formats if f not in arguments.document_formats]
    if unknown:
        parser.error(f"Cannot save mean activations for formats {unknown} that are not scored: {arguments.document_formats}")
    return arguments


def build_run_folder(arguments):
    records = read_jsonl(arguments.answers_jsonl)
    documents = assemble_documents(records, arguments.document_id_field, arguments.user_message_field, arguments.answer_text_field,
                                   arguments.dataset_groups, arguments.expected_document_count)
    chosen_layer, vector_paths, metadata_paths, metadata_by_facet, layer_by_facet = read_all_vector_sets(arguments.vector_set)
    stored = {name: np.load(path) for name, path in vector_paths.items()}
    shapes = {vector.shape for vector in stored.values()}
    if len(shapes) != 1:
        raise SystemExit(f"The vectors have different shapes: { {name: v.shape for name, v in stored.items()} }")
    number_of_stored_layers, hidden_size = next(iter(shapes))
    recorded_layers = manifest_module.recorded_layers_around(chosen_layer, number_of_stored_layers)
    first_facet = arguments.vector_set[0][0].facet_key
    model_name = arguments.model_name or metadata_by_facet[first_facet].get("model_name") or defs.MODEL_NAME
    model_revision = (arguments.model_revision if arguments.model_revision is not None
                      else metadata_by_facet[first_facet].get("model_revision", defs.MODEL_REVISION))
    estimate = estimator.estimate_from_records(
        [{"answer_text": d["content"], "user_message": d["user_message"]} for d in documents], arguments.document_formats,
        hidden_size=hidden_size, number_of_formats_saving_activations=len(arguments.save_full_activations_for_formats))
    mean_activation_gigabytes = round(
        len(documents) * len(arguments.save_per_document_mean_activations_for_formats) * (3 + arguments.number_of_position_bands)
        * number_of_stored_layers * hidden_size * 2 / 1e9, 2)
    configuration = {
        "run_kind": "persona_vector_scoring_pod", "mode": "difficult_advice_dataset_scoring", "run_name": arguments.run_name,
        "stages": [defs.STAGE_SCORING], "model_name": model_name, "model_revision": model_revision,
        "document_formats": arguments.document_formats, "number_of_documents_to_score": len(documents),
        "documents_by_dataset_group": {group: sum(1 for d in documents if d.get("dataset_group") == group)
                                       for group in sorted({str(d.get("dataset_group")) for d in documents})},
        "documents_by_stage": {stage: sum(1 for d in documents if d.get("stage") == stage)
                               for stage in sorted({str(d.get("stage")) for d in documents})},
        "baselines_per_document": 0, "number_of_baseline_generation_jobs": 0, "number_of_baseline_documents_to_generate": 0,
        "baselines": "none in this run: the answers are real conversations, scored as they are",
        "user_turn_by_format": {defs.FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE: "the answer's own user message",
                                defs.FORMAT_ASSISTANT_TURN: "the neutral message in " + defs.ASSISTANT_TURN_NEUTRAL_USER_MESSAGE_FILE_NAME},
        "scored_tokens": "the answer's tokens only, in every format",
        "number_of_position_bands": arguments.number_of_position_bands, "opening_token_count": defs.OPENING_TOKEN_COUNT,
        "vector_set_folders": {definition.facet_key: str(folder) for definition, folder in arguments.vector_set},
        "vector_metadata_paths": {facet: str(path) for facet, path in metadata_paths.items()},
        "vector_names": list(vector_paths), "vector_files_used": {name: path.name for name, path in vector_paths.items()},
        "chosen_layer": chosen_layer, "chosen_layer_by_facet": layer_by_facet, "recorded_layers": recorded_layers,
        "projection_onto": "unit vector of each layer",
        "save_full_activations_for_formats": arguments.save_full_activations_for_formats,
        "chosen_layer_for_saved_activations": chosen_layer,
        "save_per_document_mean_activations_for_formats": arguments.save_per_document_mean_activations_for_formats,
        "per_document_mean_activations_estimated_gigabytes": mean_activation_gigabytes,
        "answers_jsonl": str(arguments.answers_jsonl), "answers_jsonl_sha256": sha256_of_file(arguments.answers_jsonl),
        "answer_field_names": {"document_id": arguments.document_id_field, "user_message": arguments.user_message_field,
                               "answer_text": arguments.answer_text_field},
        "dataset_groups_selected": arguments.dataset_groups, "scoring_cost_ceiling_estimate": estimate,
        "prompt_file_sha256": {name: sha256_of_file(defs.PROMPT_FOLDER / name) for name in defs.SCORING_PROMPT_FILE_NAMES[1:]}}
    if arguments.dry_run:
        print(json.dumps({key: configuration[key] for key in (
            "number_of_documents_to_score", "documents_by_dataset_group", "documents_by_stage", "document_formats", "vector_names",
            "chosen_layer", "recorded_layers", "save_full_activations_for_formats", "save_per_document_mean_activations_for_formats",
            "per_document_mean_activations_estimated_gigabytes", "model_name")}, indent=2))
        print("DRY RUN (nothing written, nothing launched). GPU time and cost ceiling:\n" + estimator.describe(estimate))
        return None, configuration, estimate

    runs_directory = arguments.runs_directory or (defs.SMALL_SAMPLE_CHECKS_ROOT if arguments.small_sample_check else defs.RUNS_ROOT)
    run_folder, _ = choose_run_folder(runs_directory, arguments.run_name, None)
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    (inputs_folder / manifest_module.SCORING_VECTORS_SUBFOLDER).mkdir(parents=True)
    timestamp = timestamp_now()
    write_jsonl(inputs_folder / "documents.jsonl", documents)
    (inputs_folder / "generic_assistant_system_prompt.txt").write_text(rendering.generic_assistant_system_prompt() + "\n", encoding="utf-8")
    manifest = {"chosen_layer": chosen_layer, "recorded_layers": recorded_layers, "vectors": {}}
    for name, path in vector_paths.items():
        stored_name = f"{name}_{timestamp}.npy"
        shutil.copy2(path, inputs_folder / manifest_module.SCORING_VECTORS_SUBFOLDER / stored_name)
        manifest["vectors"][name] = {"file": stored_name, "source_file": path.name,
                                     "norm_at_chosen_layer": float(np.linalg.norm(stored[name][chosen_layer]))}
    for metadata_path in metadata_paths.values():
        target = inputs_folder / manifest_module.SCORING_VECTORS_SUBFOLDER / metadata_path.name
        if not target.exists():
            shutil.copy2(metadata_path, target)
    (inputs_folder / f"{manifest_module.SCORING_VECTOR_MANIFEST_STEM}_{timestamp}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    configuration["created_timestamp"] = timestamp
    configuration["input_file_sha256"] = {path.name: sha256_of_file(path) for path in sorted(inputs_folder.glob("*.*"))}
    write_run_configuration(run_folder, timestamp, configuration)
    return run_folder, configuration, estimate


def main(argv=None):
    run_folder, _, estimate = build_run_folder(parse_arguments(argv))
    if run_folder:
        print(f"Run folder: {run_folder}\nGPU time and cost ceiling:\n{estimator.describe(estimate)}")


if __name__ == "__main__":
    main()
