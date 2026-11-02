"""Builds the run folder for scoring a content-controlled facet's positive controls on the (reused) scoring pod. Free and local: no
model, API or pod is touched. `--facet` picks the facet (default moralizing_delivery, documented below and unchanged).

New facets (cares_about_welfare, calibrated_consideration, non_intrusive): documents scored = the corpus documents + each arm's
rewrite of each source (four arms; calibrated consideration has a fifth, understatement of supported claims) (from
make_content_controlled_facet_positive_control_rewrites.py). The corpus documents are the UNION, without duplicates, of the previous
scoring run's 20 corpus documents (kept for the corpus window and span analyses and for comparison with earlier runs) and every source
document of the facet's rewrites; a source that is not one of the 20 is read from the corpus file (`--corpus-path`, default
data/sdf_corpus.jsonl), and a source that is in neither is refused. Every rewrite's comparison document is its own source (or, for the
chained arm, the trait_only_away rewrite of that source), scored in the same run and format. The analysis computes the corpus
between-document spread and the corpus windows over ALL corpus documents scored in the run and records which ones. There are no
existing preachier/plainer rewrites in this run. The chained arm (trait_removed_from_away_rewrite) is compared with the
trait_only_away rewrite of the same source, so that rewrite is itself one of the scored documents; the build stops when a rewrite's
recorded comparison document is not that rewrite, when an arm or a comparison document is missing, or when a count differs from the
plan (one rewrite per arm and source). Vectors: definition.scored_vector_names() (the facet's main vector first; non_intrusive's main vector
is the whole-answer one), at the vector set's chosen layer for that facet.

Moralizing delivery:

    python3 scripts/persona_vector_projection/build_moralizing_delivery_scoring_run_folder.py \
        --vector-set-folder <extraction run folder>/vector_sets/<name>_<timestamp with seconds> \
        --new-rewrites-jsonl <rewrite run folder>/scoring_pod_extra_documents_<timestamp>.jsonl --dry-run
    ... without --dry-run to write the folder.

Documents scored (80, in assistant_turn and user_message format; no matched baselines and no steering, only the `scoring` stage):
  20 corpus documents      the corpus documents of the last scoring run (--previous-scoring-run-folder)
  20 existing rewrites     the preachier and plainer rewrites of the 10 sources, as scored by that run
  40 new rewrites          4 arms x 10 (make_moralizing_delivery_positive_control_rewrites.py); the moralizing-removed arm's
                           comparison document is the PREACHIER REWRITE it was made from, so `source_document_id` of every
                           rewrite is the document the analysis compares it with
Vectors, from the extraction run's vector set at its chosen layer (the layer is read from the set's metadata):
  the main passage-mean vector; the passage-mean vector with the top 4 largest-mean dimensions zeroed; the whole-answer vector
Each is projected onto its unit direction at the chosen layer and the two layers either side. With the default
`--save-full-activations-for-formats assistant_turn` the run configuration also asks the pod to save every document's hidden
states at the chosen layer (float16, about 2 GB), so that any other vector can be projected later with no pod.

Writes <run folder>/scoring_pod_inputs/: documents.jsonl, generic_assistant_system_prompt.txt,
scoring_vector_manifest_<ts>.json and scoring_vectors/ (the three vectors and the vector set's metadata), and
run_configuration_<ts>.json in the run folder (with the GPU time and dollar estimate).
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

from persona_vector_projection import estimate_content_controlled_facet_scoring_cost_ceilings as estimator  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_arms as arms_module  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as extraction_definitions  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering  # noqa: E402
from persona_vector_projection import scoring_vector_manifest_for_named_vectors as manifest_module  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import choose_run_folder, sha256_of_file, timestamp_now, write_run_configuration  # noqa: E402

FACET = extraction_definitions.FACET_MORALIZING_DELIVERY
MODE = "moralizing_delivery_positive_control_scoring"
RUN_NAME = "persona_vector_moralizing_delivery_positive_control_scoring_gemma_4_12b_it"
DEFAULT_PREVIOUS_SCORING_RUN_FOLDER = (
    defs.RUNS_ROOT / "persona_vector_scoring_pod_pilot_baselines_projections_and_steering_gemma_4_12b_it_2026-10-31_11-20")
DEFAULT_DOCUMENT_FORMATS = [defs.FORMAT_ASSISTANT_TURN, defs.FORMAT_USER_MESSAGE]
EXPECTED_ORIGINAL_SOURCE_COUNT = 10
EXPECTED_COUNTS = {"corpus": 20, "existing_rewrites": 20, "new_rewrites": 40}


# ---------------------------------------------------------------------------------------------------
# Documents (pure functions)
# ---------------------------------------------------------------------------------------------------

def existing_rewrite_document(record, preachiness_by_id):
    """A preachier or plainer rewrite as the earlier run scored it. Its comparison document is its corpus source."""
    content_in_rewrite_file = preachiness_by_id[record["document_id"]]["content"]
    if content_in_rewrite_file != record["content"]:
        raise ValueError(f"{record['document_id']}: the previous scoring run's text differs from the preachiness rewrite file")
    return {"document_id": record["document_id"], "source_document_id": record["source_document_id"],
            "rewrite_direction": record["rewrite_direction"], "content": record["content"], "language": record["language"],
            "variables": record["variables"], "kind": defs.DOCUMENT_KIND_REWRITE,
            "rewrite_arm": record["rewrite_direction"], "original_corpus_source_document_id": record["source_document_id"],
            "expected_sign": None, "below_minimum_changed_units": None}


def new_rewrite_document(record, definition=None):
    """A rewrite from the new rewrite run's export. `source_document_id` is its comparison document."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    arm = record["arm"]
    if record.get("facet", definition.facet_key) != definition.facet_key:
        raise ValueError(f"{record['document_id']} is a rewrite for {record.get('facet')}, not {definition.facet_key}")
    expected_comparison = arms_module.comparison_document_id_for(definition, arm, record["original_corpus_source_document_id"])
    if record["source_document_id"] != expected_comparison:
        raise ValueError(f"{record['document_id']}: comparison document {record['source_document_id']} but arm {arm} is compared "
                         f"with {expected_comparison}")
    if record["document_id"] != arms_module.rewrite_document_id_for(definition, arm, record["original_corpus_source_document_id"]):
        raise ValueError(f"{record['document_id']} is not the id of arm {arm} of {record['original_corpus_source_document_id']}")
    return {"document_id": record["document_id"], "source_document_id": record["source_document_id"],
            "rewrite_direction": arm, "content": record["content"], "language": record["language"], "variables": record["variables"],
            "kind": defs.DOCUMENT_KIND_REWRITE, "rewrite_arm": arm,
            "original_corpus_source_document_id": record["original_corpus_source_document_id"],
            "expected_sign": arms_module.expected_sign_for(definition, arm),
            "below_minimum_changed_units": record.get("below_minimum_changed_units")}


def expected_counts_for(number_of_corpus_documents, number_of_sources, definition=None):
    """Moralizing delivery: 20 corpus documents, 2 existing rewrites and 4 new rewrites per source (10 sources). The other facets
    have no existing rewrites: 20 corpus documents and one new rewrite per arm and source (4 arms; 5 for calibrated consideration)."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    existing = 2 * number_of_sources if definition.legacy_moralizing_names else 0
    arms_per_source = len(arms_module.arm_names_for(definition))  # 4, or 5 for calibrated consideration (understatement arm)
    return {"corpus": number_of_corpus_documents, "existing_rewrites": existing, "new_rewrites": arms_per_source * number_of_sources}


def corpus_document_from_corpus_file_record(record):
    """A corpus-file record (doc_id, content, language, variables) as a scored corpus document of this run's documents.jsonl."""
    return {"document_id": record["doc_id"], "source_document_id": record["doc_id"], "rewrite_direction": None,
            "content": record["content"], "language": record["language"], "variables": record["variables"],
            "kind": defs.DOCUMENT_KIND_CORPUS}


def planned_number_of_corpus_documents(previous_documents, new_rewrite_records):
    """Distinct corpus documents of a new facet's run: the previous run's corpus documents plus every rewrite source."""
    previous_ids = {d["document_id"] for d in previous_documents if d["kind"] == defs.DOCUMENT_KIND_CORPUS}
    return len(previous_ids | {r["original_corpus_source_document_id"] for r in new_rewrite_records})


def assemble_documents(previous_documents, preachiness_records, new_rewrite_records, expected_counts=None,
                       expected_source_count=EXPECTED_ORIGINAL_SOURCE_COUNT, definition=None, corpus_records_by_id=None):
    """Corpus documents and existing rewrites from the previous scoring run, then the new rewrites. Raises when a count, an id
    or a comparison document is not as planned.

    New facets: the corpus documents are the previous run's corpus documents plus every rewrite source that is not among them, taken
    from `corpus_records_by_id` ({doc_id: corpus-file record}); a source that is in neither is refused, and no corpus document is
    scored twice. Moralizing delivery (legacy) scores exactly the previous run's corpus documents, as before."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    legacy = definition.legacy_moralizing_names
    expected_counts = expected_counts or (EXPECTED_COUNTS if legacy else expected_counts_for(
        planned_number_of_corpus_documents(previous_documents, new_rewrite_records), expected_source_count, definition))
    preachiness_by_id = {record["document_id"]: record for record in preachiness_records}
    previous_corpus = [document for document in previous_documents if document["kind"] == defs.DOCUMENT_KIND_CORPUS]
    previous_corpus_ids = [document["document_id"] for document in previous_corpus]
    if len(set(previous_corpus_ids)) != len(previous_corpus_ids):
        raise SystemExit(f"The previous scoring run scored a corpus document twice: "
                         f"{sorted({i for i in previous_corpus_ids if previous_corpus_ids.count(i) > 1})}")
    corpus_documents = list(previous_corpus)
    if not legacy:
        for source_id in sorted({r["original_corpus_source_document_id"] for r in new_rewrite_records} - set(previous_corpus_ids)):
            if source_id not in (corpus_records_by_id or {}):
                raise SystemExit(f"The source document {source_id} of a rewrite is neither among the previous run's corpus documents "
                                 f"nor in the corpus file, so its comparison document cannot be scored in this run")
            corpus_documents.append(corpus_document_from_corpus_file_record(corpus_records_by_id[source_id]))
    corpus = [{**document, "rewrite_arm": None, "original_corpus_source_document_id": document["document_id"],
               "expected_sign": None, "below_minimum_changed_units": None} for document in corpus_documents]
    existing = ([existing_rewrite_document(document, preachiness_by_id)
                 for document in previous_documents if document["kind"] == defs.DOCUMENT_KIND_REWRITE] if legacy else [])
    new = [new_rewrite_document(record, definition) for record in new_rewrite_records]
    documents = sorted(corpus, key=lambda d: d["document_id"]) + sorted(existing, key=lambda d: d["document_id"]) + sorted(
        new, key=lambda d: d["document_id"])
    counts = {"corpus": len(corpus), "existing_rewrites": len(existing), "new_rewrites": len(new)}
    if counts != expected_counts:
        raise SystemExit(f"Expected {expected_counts} documents, found {counts}")
    ids = [document["document_id"] for document in documents]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"Duplicate document ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    by_id = {document["document_id"]: document for document in documents}
    missing = sorted({d["source_document_id"] for d in documents if d["source_document_id"] not in by_id})
    if missing:
        raise SystemExit(f"Comparison documents that are not among the scored documents: {missing}")
    sources = {d["original_corpus_source_document_id"] for d in new}
    corpus_ids = {d["document_id"] for d in corpus}
    if len(sources) != expected_source_count or not sources <= corpus_ids:
        raise SystemExit(f"The new rewrites must come from {expected_source_count} of the scored corpus documents; got {sorted(sources)}")
    for source in sources:
        arms_found = sorted(d["rewrite_arm"] for d in new if d["original_corpus_source_document_id"] == source)
        if arms_found != sorted(arms_module.arm_names_for(definition)):
            raise SystemExit(f"Source {source} has arms {arms_found}, expected {sorted(arms_module.arm_names_for(definition))}")
        for arm in arms_module.arm_definitions_for(definition):
            if arm.starts_from == facet_definitions.STARTS_FROM_EXISTING_PREACHIER_REWRITE:
                needed = arms_module.preachier_rewrite_document_id(source)
                if needed not in by_id:
                    raise SystemExit(f"The preachier rewrite {needed} is not among the scored documents")
            elif arm.starts_from == facet_definitions.STARTS_FROM_OUTPUT_OF_ARM:
                # arm 3: compared with arm 1's rewrite of the SAME source, which must be scored and not the original document
                needed = arms_module.rewrite_document_id_for(definition, arm.source_arm_name, source)
                recorded = by_id[arms_module.rewrite_document_id_for(definition, arm.arm_name, source)]["source_document_id"]
                if needed not in by_id or recorded != needed:
                    raise SystemExit(f"{arm.arm_name} of {source} must be compared with the scored {arm.source_arm_name} rewrite "
                                     f"{needed}; its recorded comparison document is {recorded}")
    return documents


def read_document_token_counts(previous_scoring_run_folder, document_format=defs.FORMAT_ASSISTANT_TURN):
    """{document id: scored token count} from the previous run's projection shards (counts only, nothing else is loaded)."""
    counts = {}
    folder = Path(previous_scoring_run_folder) / defs.OUTPUTS_FOLDER_NAME / defs.PROJECTIONS_SUBFOLDER / document_format
    for path in sorted(folder.glob(f"projections_{document_format}_shard_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            counts.update({str(i): int(c) for i, c in zip(archive["document_ids"], archive["document_token_counts"])})
    return counts


def estimated_token_count(document, known_counts, fallback, tokens_per_character=None):
    """The document's own count if known, else its comparison document's (a rewrite is about as long as its source), else (a corpus
    document or rewrite that the previous run did not score) its length in characters times the previous run's tokens per character,
    else the fallback."""
    known = known_counts.get(document["document_id"]) or known_counts.get(document["source_document_id"])
    if known:
        return known
    if tokens_per_character:
        return len(document["content"]) * tokens_per_character
    return fallback


def tokens_per_character_of_scored_documents(previous_documents, known_counts):
    """Tokens per character over the previous run's documents whose token count is known (None when none is)."""
    pairs = [(known_counts[d["document_id"]], len(d["content"])) for d in previous_documents if d["document_id"] in known_counts]
    characters = sum(c for _, c in pairs)
    return sum(t for t, _ in pairs) / characters if characters else None


# ---------------------------------------------------------------------------------------------------
# Vector set
# ---------------------------------------------------------------------------------------------------

def read_vector_set(vector_set_folder, definition=None):
    """(chosen layer, {vector name: path}, metadata, metadata path) from the facet's vector set (moralizing delivery by default)."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    folder = Path(vector_set_folder)
    metadata_paths = sorted(folder.glob("persona_vector_metadata_*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No persona_vector_metadata_*.json in {folder}")
    metadata = json.loads(metadata_paths[-1].read_text(encoding="utf-8"))
    layer_choice = metadata.get("layer_choice")
    if not layer_choice or definition.facet_key not in (layer_choice.get("layer_by_facet") or {}):
        raise SystemExit(f"{metadata_paths[-1]} has no layer choice for {definition.facet_key}; the layer is needed before scoring")
    key_by_name = definition.vector_set_key_by_vector_name()
    missing = [key for key in key_by_name.values() if key not in metadata["vector_files"]]
    if missing:
        raise KeyError(f"The vector set {folder} has no vector files {missing}; it has {sorted(metadata['vector_files'])}")
    paths = {name: folder / metadata["vector_files"][key] for name, key in key_by_name.items()}
    return int(layer_choice["layer_by_facet"][definition.facet_key]), paths, metadata, metadata_paths[-1]


# ---------------------------------------------------------------------------------------------------
# Command line and build
# ---------------------------------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    facet_definitions.add_facet_argument(parser)
    parser.add_argument("--vector-set-folder", required=True, help="<extraction run folder>/vector_sets/<name>_<timestamp with seconds>")
    parser.add_argument("--new-rewrites-jsonl", required=True,
                        help="scoring_pod_extra_documents_<timestamp>.jsonl written by make_content_controlled_facet_positive_control_rewrites.py")
    parser.add_argument("--previous-scoring-run-folder", default=str(DEFAULT_PREVIOUS_SCORING_RUN_FOLDER),
                        help="The last scoring run: its 20 corpus documents and 20 existing rewrites are scored again, and its "
                             "token counts size the estimate")
    parser.add_argument("--preachiness-rewrites-jsonl", default=str(defs.DEFAULT_POSITIVE_CONTROL_REWRITES_FILE))
    parser.add_argument("--corpus-path", default=str(defs.DEFAULT_CORPUS_PATH),
                        help="Corpus file that supplies every rewrite source not among the previous run's corpus documents (new facets "
                             "only; read only when such a source exists)")
    parser.add_argument("--document-formats", nargs="+", choices=defs.DOCUMENT_FORMATS, default=DEFAULT_DOCUMENT_FORMATS)
    parser.add_argument("--save-full-activations-for-formats", nargs="*", choices=defs.DOCUMENT_FORMATS,
                        default=[defs.FORMAT_ASSISTANT_TURN],
                        help="Formats whose per-token hidden states at the chosen layer the pod also saves (float16, about 2 GB). "
                             "Pass nothing after the flag for none")
    parser.add_argument("--expected-corpus-documents", type=int, default=EXPECTED_COUNTS["corpus"],
                        help="Safety check on the previous run's corpus documents (20 for the planned run; fewer only in a rehearsal). The "
                             "documents scored as corpus documents by a new facet are these plus the rewrite sources outside them")
    parser.add_argument("--expected-sources", type=int, default=EXPECTED_ORIGINAL_SOURCE_COUNT,
                        help="Safety check on the number of source documents (10 for the planned run)")
    parser.add_argument("--number-of-position-bands", type=int, default=defs.DEFAULT_NUMBER_OF_POSITION_BANDS)
    parser.add_argument("--run-name", default=None, help="Default: the facet's scoring run name")
    parser.add_argument("--small-sample-check", action="store_true", help="Put the run in small_sample_checks/")
    parser.add_argument("--runs-directory", default=None, help="Parent folder of the new run folder (default: the persona vector "
                                                               "projection runs folder, or small_sample_checks/ with --small-sample-check)")
    parser.add_argument("--model-name", default=None, help="Default: the vector set's model")
    parser.add_argument("--model-revision", default=None, help="Default: the vector set's model revision")
    parser.add_argument("--dry-run", action="store_true", help="Print counts, GPU time and cost ceiling; write nothing")
    arguments = parser.parse_args(argv)
    arguments.definition = facet_definitions.get_facet_definition(arguments.facet)
    if arguments.run_name is None:
        arguments.run_name = arguments.definition.scoring_run_name
    unknown = [f for f in arguments.save_full_activations_for_formats if f not in arguments.document_formats]
    if unknown:
        parser.error(f"Cannot save activations for formats {unknown} that are not scored: {arguments.document_formats}")
    return arguments


def build_run_folder(arguments):
    definition = getattr(arguments, "definition", None) or facet_definitions.MORALIZING_DELIVERY
    previous_folder = Path(arguments.previous_scoring_run_folder)
    previous_documents = read_jsonl(previous_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl")
    new_rewrite_records = read_jsonl(arguments.new_rewrites_jsonl)
    previous_corpus_ids = sorted(d["document_id"] for d in previous_documents if d["kind"] == defs.DOCUMENT_KIND_CORPUS)
    if len(previous_corpus_ids) != arguments.expected_corpus_documents:
        raise SystemExit(f"The previous scoring run has {len(previous_corpus_ids)} corpus documents, expected "
                         f"{arguments.expected_corpus_documents}")
    corpus_records_by_id = None
    if not definition.legacy_moralizing_names:
        sources_outside_previous_run = {r["original_corpus_source_document_id"] for r in new_rewrite_records} - set(previous_corpus_ids)
        if sources_outside_previous_run:
            corpus_path = getattr(arguments, "corpus_path", None) or defs.DEFAULT_CORPUS_PATH
            corpus_records_by_id = {record["doc_id"]: record for record in read_jsonl(corpus_path)
                                    if record["doc_id"] in sources_outside_previous_run}
    number_of_corpus_documents = (arguments.expected_corpus_documents if definition.legacy_moralizing_names
                                  else planned_number_of_corpus_documents(previous_documents, new_rewrite_records))
    documents = assemble_documents(previous_documents, read_jsonl(arguments.preachiness_rewrites_jsonl), new_rewrite_records,
                                   expected_counts_for(number_of_corpus_documents, arguments.expected_sources, definition),
                                   arguments.expected_sources, definition, corpus_records_by_id)
    scored_corpus_ids = sorted(d["document_id"] for d in documents if d["kind"] == defs.DOCUMENT_KIND_CORPUS)
    rewrite_source_ids = sorted({d["original_corpus_source_document_id"] for d in documents if d["kind"] == defs.DOCUMENT_KIND_REWRITE
                                 and d["rewrite_arm"] not in (None, "preachier", "plainer")})
    chosen_layer, vector_paths, vector_metadata, vector_metadata_path = read_vector_set(arguments.vector_set_folder, definition)
    stored = {name: np.load(path) for name, path in vector_paths.items()}
    shapes = {vector.shape for vector in stored.values()}
    if len(shapes) != 1:
        raise SystemExit(f"The vectors have different shapes: { {name: v.shape for name, v in stored.items()} }")
    number_of_stored_layers, hidden_size = next(iter(shapes))
    recorded_layers = manifest_module.recorded_layers_around(chosen_layer, number_of_stored_layers)
    previous_run_configuration = json.loads(sorted(previous_folder.glob("run_configuration_*.json"))[0].read_text(encoding="utf-8"))
    model_name = arguments.model_name or vector_metadata.get("model_name") or previous_run_configuration["model_name"]
    model_revision = (arguments.model_revision if arguments.model_revision is not None
                      else vector_metadata.get("model_revision", previous_run_configuration["model_revision"]))

    known_counts = read_document_token_counts(previous_folder)
    fallback = float(np.mean(list(known_counts.values()))) if known_counts else 2000.0
    tokens_per_character = tokens_per_character_of_scored_documents(previous_documents, known_counts)
    token_counts = [estimated_token_count(document, known_counts, fallback, tokens_per_character) for document in documents]
    estimate = estimator.estimate_scoring_time_and_cost(
        len(documents), float(np.mean(token_counts)), len(arguments.document_formats), hidden_size,
        number_of_formats_saving_activations=len(arguments.save_full_activations_for_formats))
    configuration = {
        "run_kind": "persona_vector_scoring_pod", "mode": f"{definition.facet_key}_positive_control_scoring", "facet_key": definition.facet_key, "run_name": arguments.run_name, "stages": [defs.STAGE_SCORING],
        "model_name": model_name, "model_revision": model_revision, "document_formats": arguments.document_formats,
        "number_of_documents_to_score": len(documents), "number_of_corpus_documents": len(scored_corpus_ids),
        "number_of_rewrite_documents": len(documents) - len(scored_corpus_ids),
        "documents_by_group": expected_counts_for(number_of_corpus_documents, arguments.expected_sources, definition),
        "corpus_documents_scored": {
            "rule": ("the previous scoring run's corpus documents (kept for the corpus window and span analyses and for comparison with "
                     "earlier runs), plus every rewrite source from the corpus file; none is scored twice; the analysis computes the "
                     "corpus between-document spread and the corpus windows over all of them"
                     if not definition.legacy_moralizing_names else "exactly the previous scoring run's corpus documents"),
            "document_ids": scored_corpus_ids, "previous_scoring_run_document_ids": previous_corpus_ids,
            "rewrite_source_document_ids": rewrite_source_ids,
            "added_from_corpus_file_document_ids": sorted(set(scored_corpus_ids) - set(previous_corpus_ids)),
            "corpus_path": str(getattr(arguments, "corpus_path", defs.DEFAULT_CORPUS_PATH))},
        "rewrite_arms": arms_module.arm_names_for(definition),
        "expected_sign_by_arm": {arm: arms_module.expected_sign_for(definition, arm) for arm in arms_module.arm_names_for(definition)},
        "expected_sign_by_arm_on_confidence_vector": (
            {arm: arms_module.expected_sign_on_vector_for(definition, arm, definition.confidence_control_vector_name)
             for arm in arms_module.arm_names_for(definition)} if definition.has_confidence_control else None),
        "direction_convention": "vector = positive pole minus negative pole; shift = rewrite minus comparison document; "
                                "a rewrite towards the negative pole should give a NEGATIVE shift",
        "facet_score_scale": definition.facet_score_scale_label(),
        "baselines_per_document": 0, "number_of_baseline_generation_jobs": 0, "number_of_baseline_documents_to_generate": 0,
        "baselines": "none in this run: the criteria compare each rewrite with its own comparison document",
        "number_of_position_bands": arguments.number_of_position_bands, "opening_token_count": defs.OPENING_TOKEN_COUNT,
        "vector_set_folder": str(arguments.vector_set_folder), "vector_metadata_path": str(vector_metadata_path),
        "vector_names": definition.scored_vector_names(), "main_vector_name": definition.main_vector_name,
        "main_vector_space": definition.main_space,
        "vector_files_used": {name: path.name for name, path in vector_paths.items()},
        "chosen_layer": chosen_layer, "chosen_layer_by_facet": {definition.facet_key: chosen_layer}, "recorded_layers": recorded_layers,
        "projection_onto": "unit vector of each layer",
        "save_full_activations_for_formats": arguments.save_full_activations_for_formats,
        "chosen_layer_for_saved_activations": chosen_layer,
        "corpus_mean_document_tokens": float(np.mean(token_counts)),
        "previous_scoring_run_folder": str(previous_folder), "new_rewrites_jsonl": str(arguments.new_rewrites_jsonl),
        "new_rewrites_jsonl_sha256": sha256_of_file(arguments.new_rewrites_jsonl),
        "preachiness_rewrites_jsonl": str(arguments.preachiness_rewrites_jsonl),
        "scoring_cost_ceiling_estimate": estimate,
        "prompt_file_sha256": {name: sha256_of_file(defs.PROMPT_FOLDER / name) for name in defs.SCORING_PROMPT_FILE_NAMES[1:]}}
    if arguments.dry_run:
        print(json.dumps({key: configuration[key] for key in (
            "documents_by_group", "document_formats", "vector_names", "chosen_layer", "recorded_layers",
            "save_full_activations_for_formats", "model_name")}, indent=2))
        print("DRY RUN (nothing written, nothing launched). GPU time and cost ceiling:\n" + estimator.describe(estimate))
        return None, configuration, estimate

    runs_directory = arguments.runs_directory or (defs.SMALL_SAMPLE_CHECKS_ROOT if arguments.small_sample_check else defs.RUNS_ROOT)
    run_folder, _ = choose_run_folder(runs_directory, arguments.run_name, None)
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    (inputs_folder / manifest_module.SCORING_VECTORS_SUBFOLDER).mkdir(parents=True)
    timestamp = timestamp_now()
    write_jsonl(inputs_folder / "documents.jsonl", documents)
    (inputs_folder / "generic_assistant_system_prompt.txt").write_text(
        rendering.generic_assistant_system_prompt() + "\n", encoding="utf-8")
    manifest = {"chosen_layer": chosen_layer, "recorded_layers": recorded_layers, "vectors": {}}
    for name, path in vector_paths.items():
        stored_name = f"{name}_{timestamp}.npy"
        shutil.copy2(path, inputs_folder / manifest_module.SCORING_VECTORS_SUBFOLDER / stored_name)
        manifest["vectors"][name] = {"file": stored_name, "source_file": path.name,
                                     "norm_at_chosen_layer": float(np.linalg.norm(stored[name][chosen_layer]))}
    shutil.copy2(vector_metadata_path, inputs_folder / manifest_module.SCORING_VECTORS_SUBFOLDER / vector_metadata_path.name)
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
