"""Runs on the GPU pod (or on CPU with a tiny model for tests): matched baselines, activation scoring, steering.

    python3 scripts/persona_vector_projection/run_scoring_pod_stages.py --run-folder <run folder> \
        [--stages baselines scoring steering] [--document-formats assistant_turn ...]

Stages (default: all three, each in its own child process so vLLM's GPU memory is freed before the Hugging Face model
loads):
  baselines  baseline_generation_jobs.jsonl -> matched_baseline_documents/baseline_documents_shard_<n>_<stamp>.jsonl
  scoring    documents.jsonl + baseline shards -> per_document_projections/<format>/projections_<format>_shard_<n>_<stamp>.npz
  steering   steering_jobs.jsonl -> steering_generations/answers_shard_<n>_steering_<stamp>.jsonl (readable by the judge)
Resumable: every shard is written atomically after it is complete and an existing shard is skipped, so a relaunch on a
new pod (after the launcher has uploaded the pulled outputs) continues from the next shard. <stamp> is the run's creation
timestamp from run_configuration, so file names stay stable across resumes. Sampling: temperature 1, top-p 1, no top-k.
Progress reports with timings about every 3 minutes.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering  # noqa: E402
from persona_vector_projection import chat_prompt_rendering_for_extraction as extraction_rendering  # noqa: E402
from persona_vector_projection.scoring_pod_vector_loading_adapter import load_scoring_vectors  # noqa: E402
from persona_vector_projection import scoring_vector_manifest_for_named_vectors as named_vectors  # noqa: E402
from persona_vector_projection import saved_activation_projection_locally as saved_activations_module  # noqa: E402
from persona_vector_projection import content_controlled_facet_steering_vectors_and_coefficients as unit_steering  # noqa: E402
from persona_vector_projection.scoring_pod_steering_hook import steering_applied_at_layers, steering_vectors_by_layer  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402
from shared.timed_progress_reporter import TimedProgressReporter  # noqa: E402

DEFAULT_BASELINE_JOBS_PER_SHARD = 16
DEFAULT_DOCUMENTS_PER_SHARD = 8
DEFAULT_STEERING_JOBS_PER_SHARD = 12
DEFAULT_MAXIMUM_TOKENS_PER_BATCH = 16384
FLOAT16_MAXIMUM = 65504.0


def load_run_configuration(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_configuration file in {run_folder}")
    return json.loads(paths[0].read_text(encoding="utf-8"))


def write_file_atomically(path, writer_function):
    path = Path(path)
    temporary_path = path.with_name(f"{path.stem}.partial{path.suffix}")
    writer_function(temporary_path)
    os.replace(temporary_path, path)


def split_into_shards(items, items_per_shard):
    return [items[start:start + items_per_shard] for start in range(0, len(items), items_per_shard)]


def split_steering_jobs_into_shards(jobs, jobs_per_shard):
    """Shards never mix steering conditions (one steering hook per generate call). Deterministic."""
    shards, current_condition, current = [], None, []
    for job in jobs:
        if job["steering_condition_id"] != current_condition or len(current) == jobs_per_shard:
            if current:
                shards.append(current)
            current, current_condition = [], job["steering_condition_id"]
        current.append(job)
    if current:
        shards.append(current)
    return shards


def outputs_folder_for(run_folder, subfolder):
    folder = Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def make_reporter(run_folder, stage_name, total_shards, done_before, status_function):
    log_path = Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / f"{stage_name}_progress_log_{timestamp_now()}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return TimedProgressReporter(log_path, total_shards, already_done_count=done_before,
                                 status_suffix_function=status_function)


# ---------------------------------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------------------------------

class VllmGenerationBackendWithLongContext:
    def __init__(self, model_name, model_revision, random_seed, maximum_new_tokens, maximum_model_length):
        from vllm import LLM
        import vllm
        self.vllm_version, self.maximum_new_tokens = vllm.__version__, maximum_new_tokens
        revision_arguments = {"revision": model_revision, "tokenizer_revision": model_revision} if model_revision else {}
        self.llm = LLM(model=model_name, dtype="bfloat16", max_model_len=maximum_model_length,
                       gpu_memory_utilization=0.90, seed=random_seed, enable_prefix_caching=True, **revision_arguments)

    def generate(self, prompt_token_id_lists, samples_per_prompt):
        from vllm import SamplingParams
        parameters = SamplingParams(n=samples_per_prompt, temperature=defs.SAMPLING_TEMPERATURE, top_p=defs.SAMPLING_TOP_P,
                                    top_k=-1, max_tokens=self.maximum_new_tokens)
        outputs = self.llm.generate([{"prompt_token_ids": ids} for ids in prompt_token_id_lists], parameters)
        return [[{"text": candidate.text, "number_of_tokens": len(candidate.token_ids),
                  "finish_reason": candidate.finish_reason} for candidate in output.outputs] for output in outputs]


def load_hugging_face_model(arguments, model_name, model_revision):
    import score_document_log_probabilities_on_pod as scorer_module
    model, tokenizer = scorer_module.load_model_and_tokenizer(model_name, arguments.device, arguments.dtype,
                                                              arguments.model_class, model_revision)
    return model, tokenizer, scorer_module


# ---------------------------------------------------------------------------------------------------
# Stage 1: matched baselines
# ---------------------------------------------------------------------------------------------------

def baseline_shard_path(run_folder, stamp, shard_number):
    return outputs_folder_for(run_folder, defs.BASELINES_SUBFOLDER) / f"baseline_documents_shard_{shard_number:05d}_{stamp}.jsonl"


def run_baseline_stage(run_folder, jobs, tokenizer, backend, stamp, jobs_per_shard, model_name=None):
    shards = split_into_shards(jobs, jobs_per_shard)
    remaining = [(n, shard) for n, shard in enumerate(shards) if not baseline_shard_path(run_folder, stamp, n).exists()]
    totals, start = {"documents": 0, "tokens": 0, "truncated": 0}, time.time()
    reporter = make_reporter(run_folder, "baselines", len(remaining), len(shards) - len(remaining), lambda: (
        f"baselines generated {totals['documents']}, tokens {totals['tokens']}, "
        f"{totals['tokens'] / max(1e-9, time.time() - start):.0f} tokens/second, truncated at the length cap {totals['truncated']}"))
    print(f"Baselines: {len(shards)} shards, {len(shards) - len(remaining)} already done", flush=True)
    for shard_number, shard_jobs in remaining:
        prompt_ids = [rendering.baseline_prompt_token_ids(tokenizer, job["user_prompt"], job["system_prompt"], model_name)
                      for job in shard_jobs]
        generations = backend.generate(prompt_ids, shard_jobs[0]["samples_per_prompt"])
        records = []
        for job, samples in zip(shard_jobs, generations):
            for baseline_index, sample in enumerate(samples):
                records.append({"document_id": defs.baseline_document_id(job["source_document_id"], baseline_index),
                                "source_document_id": job["source_document_id"], "baseline_index": baseline_index,
                                "language": job["language"], "kind": defs.DOCUMENT_KIND_BASELINE,
                                "content": sample["text"], "number_of_tokens_generated": sample["number_of_tokens"],
                                "finish_reason": sample["finish_reason"], "truncated_at_length_cap": sample["finish_reason"] == "length"})
                totals["documents"] += 1
                totals["tokens"] += sample["number_of_tokens"]
                totals["truncated"] += sample["finish_reason"] == "length"
        write_file_atomically(baseline_shard_path(run_folder, stamp, shard_number), lambda path: write_jsonl(path, records))
        reporter.note_task_finished(failed=False)
    reporter.finish()
    return totals


# ---------------------------------------------------------------------------------------------------
# Stage 2: activation scoring
# ---------------------------------------------------------------------------------------------------

def read_generated_baselines(run_folder):
    records = []
    for path in sorted((Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / defs.BASELINES_SUBFOLDER).glob("baseline_documents_shard_*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def build_documents_to_score(documents, baselines):
    """Documents from the inputs plus the non-empty generated baselines, sorted by id (the order fixes the shards)."""
    usable = [record for record in baselines if record["content"].strip()]
    items = list(documents) + usable
    ids = [item["document_id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document ids among documents and baselines")
    return sorted(items, key=lambda item: item["document_id"])


def projection_shard_path(run_folder, stamp, document_format, shard_number):
    folder = outputs_folder_for(run_folder, f"{defs.PROJECTIONS_SUBFOLDER}/{document_format}")
    return folder / f"projections_{document_format}_shard_{shard_number:05d}_{stamp}.npz"


def mean_activations_shard_path(run_folder, stamp, document_format, shard_number):
    folder = outputs_folder_for(run_folder, f"{defs.MEAN_ACTIVATIONS_SUBFOLDER}/{document_format}")
    return folder / f"mean_activations_{document_format}_shard_{shard_number:05d}_{stamp}.npz"


def build_mean_activation_arrays(document_ids, range_sums_by_index, number_of_documents, number_of_bands):
    """The arrays of one mean-activations shard from the per-document range sums (float32). Means are computed in float32 and
    stored as float16; a range with no tokens is NaN with count 0. Raises when a mean overflows float16."""
    sums = [range_sums_by_index[index][0] for index in range(number_of_documents)]
    counts = np.stack([range_sums_by_index[index][1] for index in range(number_of_documents)]) if sums else np.zeros((0, 3 + number_of_bands), np.int64)
    layer_count, hidden_size = (sums[0].shape[1], sums[0].shape[2]) if sums else (0, 0)
    means = np.full((number_of_documents, 3 + number_of_bands, layer_count, hidden_size), np.nan, np.float32)
    for index, document_sums in enumerate(sums):
        for range_index in range(3 + number_of_bands):
            if counts[index, range_index] > 0:
                means[index, range_index] = document_sums[range_index] / np.float32(counts[index, range_index])
    if np.isfinite(means).any() and float(np.nanmax(np.abs(means))) > FLOAT16_MAXIMUM:
        raise FloatingPointError("A mean hidden state overflows float16")
    means_float16 = means.astype(np.float16)
    return {"document_ids": np.array(document_ids), "layer_indices": np.arange(layer_count),
            "mean_overall_float16": means_float16[:, 0], "mean_opening_float16": means_float16[:, 1],
            "mean_after_opening_float16": means_float16[:, 2], "mean_bands_float16": means_float16[:, 3:],
            "token_count_overall": counts[:, 0].astype(np.int32), "token_count_opening": counts[:, 1].astype(np.int32),
            "token_count_after_opening": counts[:, 2].astype(np.int32), "band_token_counts": counts[:, 3:].astype(np.int32)}


def score_shard_documents(shard_documents, document_format, tokenizer, recorder, device, system_prompt, neutral_user_message,
                          number_of_bands, maximum_tokens_per_batch, padding_token_id, model_name=None, keep_per_token=True,
                          activation_layer_to_save=None, mean_activations_sink=None):
    """Returns the arrays of one shard (see the npz keys) and the ids of documents that could not be tokenized. With
    `activation_layer_to_save` the hidden states at that layer are kept too (float16, see saved_activation_projection_locally.py).
    With `mean_activations_sink` (a dict; the recorder needs enable_all_layer_range_sums() before attach()) it is filled with the
    arrays of the mean-activations shard (see build_mean_activation_arrays): means of every layer over the same token ranges as
    the projections' mean_overall, mean_opening, mean_after_opening and mean_bands."""
    from persona_vector_projection.scoring_pod_projection_recording import project_documents
    tokenized, kept, failed = [], [], []
    for document in shard_documents:
        try:
            tokenized.append(rendering.tokenize_document_for_format(
                tokenizer, document["content"], document_format,
                system_prompt=rendering.system_prompt_for_format(document_format, system_prompt),
                neutral_user_message=neutral_user_message, model_name=model_name,
                user_message=document.get("user_message"), document_id=document["document_id"]))
            kept.append(document)
        except Exception as error:  # a document whose tokens cannot be located is recorded, never silently scored
            failed.append({"document_id": document["document_id"], "error": str(error)[:300]})
    saving_activations = activation_layer_to_save is not None
    recorder.activation_layers_to_save = [int(activation_layer_to_save)] if saving_activations else []
    saved_activations_by_index = {} if saving_activations else None
    range_sums_by_index = {} if mean_activations_sink is not None else None
    projections = (project_documents(recorder, tokenized, device, maximum_tokens_per_batch, padding_token_id,
                                     saved_activations_sink=saved_activations_by_index,
                                     range_sums_sink=range_sums_by_index, number_of_bands=number_of_bands) if tokenized else [])
    if range_sums_by_index is not None:
        mean_activations_sink.update(build_mean_activation_arrays(
            [d["document_id"] for d in kept], range_sums_by_index, len(kept), number_of_bands))
    summaries = [rendering.summarize_per_token_values(values, number_of_bands) for values in projections]
    layers, vectors = len(recorder.layer_indices), len(recorder.vector_names)
    arrays = {"document_ids": np.array([d["document_id"] for d in kept]),
              "document_kinds": np.array([d["kind"] for d in kept]),
              "document_token_counts": np.array([v.shape[0] for v in projections], dtype=np.int32),
              "layer_indices": np.array(recorder.layer_indices), "vector_names": np.array(recorder.vector_names)}
    for key in ("overall", "opening", "after_opening", "bands"):
        arrays[f"mean_{key}"] = (np.stack([s[key] for s in summaries]).astype(np.float32) if summaries else np.zeros((0, 0, layers, vectors), np.float32))
    arrays["band_token_counts"] = np.array([np.bincount(rendering.position_band_indices(v.shape[0], number_of_bands),
                                                        minlength=number_of_bands) for v in projections], dtype=np.int32)
    if keep_per_token:
        if projections:
            largest = max(float(np.abs(v).max()) for v in projections)
            if largest > FLOAT16_MAXIMUM:
                raise FloatingPointError(f"A projection ({largest}) overflows float16; rerun with --skip-per-token-arrays or float32")
        arrays["per_token_projections_float16"] = (np.concatenate(projections).astype(np.float16) if projections
                                                   else np.zeros((0, layers, vectors), np.float16))
        arrays["per_token_offsets"] = np.cumsum([0] + [v.shape[0] for v in projections]).astype(np.int64)
        # The document's own token ids, aligned with per_token_projections_float16 through per_token_offsets, so the
        # analysis never has to re-tokenise (int32: the vocabulary has 262,144 entries, too many for uint16).
        arrays["per_token_token_ids_int32"] = (np.concatenate([
            np.asarray(t.token_ids[t.document_start_index:t.document_end_index], dtype=np.int32) for t in tokenized])
            if tokenized else np.zeros((0,), np.int32))
    if saving_activations:
        hidden_size = next(iter(saved_activations_by_index[0].values())).shape[-1] if saved_activations_by_index else 0
        arrays[saved_activations_module.SAVED_ACTIVATIONS_KEY] = (
            np.concatenate([saved_activations_by_index[index][int(activation_layer_to_save)] for index in range(len(tokenized))])
            if tokenized else np.zeros((0, hidden_size), np.float16))
        arrays[saved_activations_module.SAVED_ACTIVATIONS_LAYER_KEY] = np.array(int(activation_layer_to_save))
        arrays.setdefault("per_token_offsets", np.cumsum([0] + [v.shape[0] for v in projections]).astype(np.int64))
    return arrays, failed


def run_scoring_stage(run_folder, configuration, documents, tokenizer, model, device, arguments, stamp, model_name=None):
    from persona_vector_projection.scoring_pod_projection_recording import LayerProjectionRecorder
    inputs_folder = Path(run_folder) / defs.INPUTS_FOLDER_NAME
    if named_vectors.find_scoring_vector_manifest(inputs_folder):  # the moralizing delivery positive-control scoring
        vectors = named_vectors.load_named_scoring_vectors(inputs_folder)
        layers = vectors.recorded_layers
    else:
        vectors = load_scoring_vectors(inputs_folder / "vectors")
        layers = vectors.recorded_layers()
    if layers != configuration["recorded_layers"]:
        raise RuntimeError(f"Vectors imply layers {layers} but the run configuration says {configuration['recorded_layers']}")
    scored_vector_names = list(vectors.vectors_by_name)  # the facets and topic vector, then any `<facet>__<variant>`
    if "vector_names" in configuration and list(configuration["vector_names"]) != scored_vector_names:
        raise RuntimeError(f"Vector folder holds {scored_vector_names} but the run configuration says "
                           f"{configuration['vector_names']}")
    recorder = LayerProjectionRecorder(model, layers, vectors.unit_vectors_by_name(), scored_vector_names)
    formats_to_save_mean_activations = list(configuration.get("save_per_document_mean_activations_for_formats") or [])
    if formats_to_save_mean_activations:
        recorder.enable_all_layer_range_sums()
    recorder.attach()
    formats_to_save_activations = list(getattr(arguments, "save_full_activations_for_formats", None) or [])
    activation_layer_to_save = None
    if formats_to_save_activations:
        if "chosen_layer_for_saved_activations" not in configuration:
            raise RuntimeError("--save-full-activations-for-formats needs `chosen_layer_for_saved_activations` in the run configuration")
        activation_layer_to_save = int(configuration["chosen_layer_for_saved_activations"])
        if activation_layer_to_save not in layers:
            raise RuntimeError(f"Layer {activation_layer_to_save} to save is not among the recorded layers {layers}")
    system_prompt = (inputs_folder / "generic_assistant_system_prompt.txt").read_text(encoding="utf-8").strip()
    neutral_user_message = rendering.load_prompt(defs.ASSISTANT_TURN_NEUTRAL_USER_MESSAGE_FILE_NAME)
    items = build_documents_to_score(documents, read_generated_baselines(run_folder))
    formats = arguments.document_formats or configuration["document_formats"]
    if any(f in defs.FORMATS_NEEDING_A_USER_MESSAGE_PER_DOCUMENT for f in formats):  # refuse before any shard is scored
        for item in items:
            rendering.require_real_user_message(item["document_id"], item.get("user_message"))
    shards = split_into_shards(items, arguments.documents_per_shard)
    padding_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    work = [(f, n, shard) for f in formats for n, shard in enumerate(shards)]
    def shard_is_finished(document_format, shard_number):
        if not projection_shard_path(run_folder, stamp, document_format, shard_number).exists():
            return False
        return (document_format not in formats_to_save_mean_activations
                or mean_activations_shard_path(run_folder, stamp, document_format, shard_number).exists())
    remaining = [(f, n, shard) for f, n, shard in work if not shard_is_finished(f, n)]
    totals, start = {"documents": 0, "tokens": 0, "failed": 0}, time.time()
    reporter = make_reporter(run_folder, "scoring", len(remaining), len(work) - len(remaining), lambda: (
        f"documents scored {totals['documents']}, document tokens {totals['tokens']}, "
        f"{totals['tokens'] / max(1e-9, time.time() - start):.0f} tokens/second, failed {totals['failed']}"))
    print(f"Scoring: {len(items)} documents x {len(formats)} formats, {len(work)} shards, {len(work) - len(remaining)} already done", flush=True)
    try:
        for document_format, shard_number, shard_documents in remaining:
            mean_activations = {} if document_format in formats_to_save_mean_activations else None
            arrays, failed = score_shard_documents(
                shard_documents, document_format, tokenizer, recorder, device, system_prompt, neutral_user_message,
                configuration["number_of_position_bands"], arguments.maximum_tokens_per_batch, padding_token_id, model_name,
                keep_per_token=not arguments.skip_per_token_arrays,
                activation_layer_to_save=activation_layer_to_save if document_format in formats_to_save_activations else None,
                mean_activations_sink=mean_activations)
            if failed:
                with open(Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / f"scoring_failures_{stamp}.jsonl", "a", encoding="utf-8") as handle:
                    for record in failed:
                        handle.write(json.dumps({"document_format": document_format, **record}) + "\n")
            if mean_activations is not None and not mean_activations_shard_path(run_folder, stamp, document_format, shard_number).exists():
                write_file_atomically(mean_activations_shard_path(run_folder, stamp, document_format, shard_number),
                                      lambda path: np.savez(path, **mean_activations))
            if not projection_shard_path(run_folder, stamp, document_format, shard_number).exists():  # keep a finished shard as it is
                write_file_atomically(projection_shard_path(run_folder, stamp, document_format, shard_number),
                                      lambda path: np.savez(path, **arrays))
            totals["documents"] += len(arrays["document_ids"])
            totals["tokens"] += int(arrays["document_token_counts"].sum())
            totals["failed"] += len(failed)
            reporter.note_task_finished(failed=False)
    finally:
        recorder.detach()
    reporter.finish()
    return totals


# ---------------------------------------------------------------------------------------------------
# Stage 3: steering generations
# ---------------------------------------------------------------------------------------------------

def steering_shard_path(run_folder, stamp, shard_number):
    return outputs_folder_for(run_folder, defs.STEERING_SUBFOLDER) / f"answers_shard_{shard_number:05d}_steering_{stamp}.jsonl"


def run_steering_stage(run_folder, configuration, jobs, tokenizer, model, backend, stamp, jobs_per_shard, model_name=None):
    inputs_folder = Path(run_folder) / defs.INPUTS_FOLDER_NAME
    if unit_steering.find_steering_vector_manifest(inputs_folder):
        # A steering-only run (moralizing delivery): unit vectors, and each job's coefficient is the absolute amount added.
        vectors_by_name = unit_steering.load_unit_steering_vectors(inputs_folder)
    else:
        vectors_by_name = load_scoring_vectors(inputs_folder / "vectors").vectors_by_name
    layer_by_facet = configuration["steering_layer_by_facet"]
    shards = split_steering_jobs_into_shards(jobs, jobs_per_shard)
    remaining = [(n, shard) for n, shard in enumerate(shards) if not steering_shard_path(run_folder, stamp, n).exists()]
    totals, start = {"answers": 0, "tokens": 0}, time.time()
    reporter = make_reporter(run_folder, "steering", len(remaining), len(shards) - len(remaining), lambda: (
        f"steered answers {totals['answers']}, tokens {totals['tokens']}, {totals['tokens'] / max(1e-9, time.time() - start):.0f} tokens/second"))
    print(f"Steering: {len(shards)} shards, {len(shards) - len(remaining)} already done", flush=True)
    for shard_number, shard_jobs in remaining:
        facets_and_coefficients = shard_jobs[0]["facets_and_coefficients"]
        vectors_by_layer = steering_vectors_by_layer(vectors_by_name, facets_and_coefficients, layer_by_facet)
        prompt_ids = [extraction_rendering.prompt_token_ids(tokenizer, job["system_prompt"], job["user_prompt"], model_name)
                      for job in shard_jobs]
        if facets_and_coefficients:
            with steering_applied_at_layers(model, vectors_by_layer, configuration["steering_positions"]):
                generations = backend.generate(prompt_ids, shard_jobs[0]["samples_per_prompt"])
        else:
            generations = backend.generate(prompt_ids, shard_jobs[0]["samples_per_prompt"])
        records = []
        for job, samples in zip(shard_jobs, generations):
            for sample_index, sample in enumerate(samples):
                records.append({"facet": job["facet"], "pole": "steered", "tag": job["steering_condition_id"],
                                "system_prompt_id": job["steering_condition_id"], "question_id": job["question_id"],
                                "pair_id": job["pair_id"], "version": job["version"], "half": job["half"], "domain": job["domain"],
                                "answer_id": f"{job['job_id']}__sample_{sample_index:02d}", "job_id": job["job_id"],
                                "sample_index": sample_index, "answer_text": sample["text"],
                                "steering_condition_id": job["steering_condition_id"],
                                "facets_and_coefficients": job["facets_and_coefficients"], "steering_layers": {facet: layer_by_facet[facet] for facet in job["facets_and_coefficients"]},
                                "steering_positions": configuration["steering_positions"],
                                "number_of_answer_tokens_generated": sample["number_of_tokens"],
                                "finish_reason": sample["finish_reason"]})
                totals["answers"] += 1
                totals["tokens"] += sample["number_of_tokens"]
        write_file_atomically(steering_shard_path(run_folder, stamp, shard_number), lambda path: write_jsonl(path, records))
        reporter.note_task_finished(failed=False)
    reporter.finish()
    return totals


# ---------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--stages", nargs="+", choices=defs.ALL_STAGES, default=None,
                        help="Default: the stages in the run configuration (all three for the earlier runs)")
    parser.add_argument("--document-formats", nargs="+", choices=defs.SELECTABLE_DOCUMENT_FORMATS, default=None,
                        help="Default: the formats in the run configuration")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default="vllm",
                        help="Baseline generation backend. Steering always uses Hugging Face (it needs a forward hook)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-class", default="auto_causal_lm", choices=["auto_causal_lm", "auto_image_text_to_text"])
    parser.add_argument("--baseline-jobs-per-shard", type=int, default=DEFAULT_BASELINE_JOBS_PER_SHARD)
    parser.add_argument("--documents-per-shard", type=int, default=DEFAULT_DOCUMENTS_PER_SHARD)
    parser.add_argument("--steering-jobs-per-shard", type=int, default=DEFAULT_STEERING_JOBS_PER_SHARD)
    parser.add_argument("--maximum-tokens-per-batch", type=int, default=DEFAULT_MAXIMUM_TOKENS_PER_BATCH)
    parser.add_argument("--maximum-documents", type=int, default=None, help="Testing only: score the first N input documents")
    parser.add_argument("--maximum-new-tokens", type=int, default=None, help="Override the baseline length cap (testing)")
    parser.add_argument("--steering-maximum-new-tokens", type=int, default=None, help="Override the steering length cap (testing)")
    parser.add_argument("--skip-per-token-arrays", action="store_true", help="Keep only the per-document summaries")
    parser.add_argument("--save-full-activations-for-formats", nargs="+", choices=defs.SELECTABLE_DOCUMENT_FORMATS, default=None,
                        help="Also save every scored document's hidden states, float16, at the layer named by "
                             "`chosen_layer_for_saved_activations` in the run configuration, for these formats")
    return parser.parse_args(argv)


def run_one_stage(stage, arguments, configuration, stamp):
    run_folder = Path(arguments.run_folder)
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    model_name = arguments.model_name or configuration["model_name"]
    model_revision = arguments.model_revision if arguments.model_revision is not None else configuration["model_revision"]
    outputs_root = run_folder / defs.OUTPUTS_FOLDER_NAME
    outputs_root.mkdir(parents=True, exist_ok=True)
    if stage == defs.STAGE_BASELINES:
        jobs = read_jsonl(inputs_folder / "baseline_generation_jobs.jsonl")
        maximum_new_tokens = arguments.maximum_new_tokens or configuration["maximum_new_tokens_for_baselines"]
        if arguments.generation_backend == "vllm":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
            backend = VllmGenerationBackendWithLongContext(model_name, model_revision, configuration["random_seed"],
                                                           maximum_new_tokens, configuration["vllm_maximum_model_length"])
            environment = {"vllm_version": backend.vllm_version}
        else:
            model, tokenizer, _ = load_hugging_face_model(arguments, model_name, model_revision)
            from persona_vector_projection.generate_answers_and_save_activations_on_pod import HuggingFaceGenerationBackend
            backend, environment = HuggingFaceGenerationBackend(model, tokenizer, arguments.device, maximum_new_tokens, prompts_per_batch=4), {}
        (outputs_root / f"baselines_environment_{timestamp_now()}.json").write_text(json.dumps({
            "model_name": model_name, "model_revision": model_revision, "backend": arguments.generation_backend,
            "maximum_new_tokens": maximum_new_tokens, "python_version": platform.python_version(),
            "hostname": platform.node(), **environment}, indent=2))
        print("Baselines finished:", run_baseline_stage(run_folder, jobs, tokenizer, backend, stamp,
                                                         arguments.baseline_jobs_per_shard, model_name), flush=True)
        return
    model, tokenizer, scorer_module = load_hugging_face_model(arguments, model_name, model_revision)
    scorer_module.write_environment_record(outputs_root / f"{stage}_environment_{timestamp_now()}.json", model, model_name,
                                           model_revision, arguments.dtype, arguments.device, arguments.model_class)
    if stage == defs.STAGE_SCORING:
        documents = read_jsonl(inputs_folder / "documents.jsonl")
        if arguments.maximum_documents:
            documents = documents[:arguments.maximum_documents]
        print("Scoring finished:", run_scoring_stage(run_folder, configuration, documents, tokenizer, model, arguments.device,
                                                      arguments, stamp, model_name), flush=True)
    else:
        from persona_vector_projection.generate_answers_and_save_activations_on_pod import HuggingFaceGenerationBackend
        jobs = read_jsonl(inputs_folder / "steering_jobs.jsonl")
        maximum_new_tokens = arguments.steering_maximum_new_tokens or configuration["steering_maximum_new_tokens"]
        backend = HuggingFaceGenerationBackend(model, tokenizer, arguments.device, maximum_new_tokens,
                                               prompts_per_batch=arguments.steering_jobs_per_shard)
        print("Steering finished:", run_steering_stage(run_folder, configuration, jobs, tokenizer, model, backend, stamp,
                                                        arguments.steering_jobs_per_shard, model_name), flush=True)


def main(argv=None):
    arguments = parse_arguments(argv)
    configuration = load_run_configuration(arguments.run_folder)
    stamp = configuration["created_timestamp"]
    if arguments.stages is None:
        arguments.stages = list(configuration.get("stages") or defs.ALL_STAGES)
    if len(arguments.stages) > 1:
        forwarded, skip = [], 0
        for part in (argv if argv is not None else sys.argv[1:]):
            if part == "--stages":
                skip = 1
                continue
            if skip and not part.startswith("--"):
                continue
            skip = 0
            forwarded.append(part)
        for stage in arguments.stages:
            print(f"Starting stage {stage} in its own process", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *forwarded, "--stages", stage], check=True)
        return 0
    run_one_stage(arguments.stages[0], arguments, configuration, stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
