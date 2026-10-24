"""Scores, for every (document, condition) in a run folder, the log-probability the model assigns to each
document content token given the condition's prompt: the chat template for an instruct model, or the plain-text author preamble for a
base model (the model type is read from the run configuration). Forward passes only. Runs on the GPU pod (or on
CPU with a tiny model for the smoke test).

Memory: the vocabulary has about 262k entries, so full-sequence logits are never built. The decoder's final
hidden states are taken only at the positions that predict document tokens, and the model's own output head
(plus its final-logit softcapping, if the config has one) is applied to chunks of those positions; each
chunk is converted to a float32 log_softmax and only the actual next token's entry is kept.
Equivalence with the plain full-logits computation is tested in test_log_probability_persona_check.py and
can be checked on the real model with --verify-first-document-against-full-logits.

Usage (from the repository root, inputs from build_scoring_inputs.py):
    python3 scripts/log_probability_persona_check/score_document_log_probabilities_on_pod.py \
        --run-folder data/log_probability_persona_check_runs/<run>
Resumable: documents whose .npz exists and whose completion records are all present are skipped.

Model classes: --model-class auto_causal_lm (default) uses AutoModelForCausalLM. For a multimodal checkpoint
such as Gemma 4 (model_type gemma4_unified) the Auto mapping resolves this to the conditional-generation
class, whose text decoder is `model.base_model` and whose output head is `lm_head`; both are used through
the generic accessors below. --model-class auto_image_text_to_text uses AutoModelForImageTextToText instead,
for checkpoints whose causal-LM mapping is missing. A base (pretrained) model needs no chat template: runs
built with --model-type base use plain text only. HF_TOKEN in the environment is used if set.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for directory in (SCRIPTS_DIRECTORY, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import log_probability_persona_check_prompt_rendering as prompt_rendering  # noqa: E402
from shared.resumable_jsonl_records import AppendOnlyJsonlWriter, find_completed_keys, read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402
from shared.timed_progress_reporter import TimedProgressReporter  # noqa: E402

DEFAULT_MAXIMUM_TOKENS_PER_BATCH = 16384
DEFAULT_POSITIONS_PER_HEAD_CHUNK = 1024
DTYPES_BY_NAME = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
COMPLETION_RECORDS_FILE_NAME = "scoring_completion_records.jsonl"


# ---------------------------------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------------------------------

def get_final_logit_softcapping(model):
    return getattr(model.config.get_text_config(), "final_logit_softcapping", None)


def apply_final_logit_softcapping(logits, softcapping_value):
    """Exactly what the Gemma forward passes do: logits / cap, tanh, * cap, in the logits' own dtype."""
    if softcapping_value is None:
        return logits
    return torch.tanh(logits / softcapping_value) * softcapping_value


def compute_next_token_log_probabilities_in_chunks(hidden_states, target_token_ids, output_head,
                                                   softcapping_value, positions_per_chunk):
    """hidden_states: (N, hidden_size), the final hidden states at the positions that predict the targets.
    target_token_ids: (N,). Returns float32 (N,): log P(target | everything before), never holding more than
    positions_per_chunk x vocabulary logits."""
    pieces = []
    for chunk_start in range(0, hidden_states.shape[0], positions_per_chunk):
        chunk_logits = output_head(hidden_states[chunk_start:chunk_start + positions_per_chunk])
        chunk_logits = apply_final_logit_softcapping(chunk_logits, softcapping_value)
        chunk_log_probabilities = torch.log_softmax(chunk_logits.float(), dim=-1)
        chunk_targets = target_token_ids[chunk_start:chunk_start + positions_per_chunk]
        pieces.append(chunk_log_probabilities.gather(1, chunk_targets.unsqueeze(1)).squeeze(1))
    return torch.cat(pieces) if pieces else hidden_states.new_zeros((0,), dtype=torch.float32)


def compute_final_hidden_states(model, input_token_ids, attention_mask):
    outputs = model.base_model(input_ids=input_token_ids, attention_mask=attention_mask, use_cache=False)
    return outputs.last_hidden_state


def pad_token_id_for(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


def score_tokenized_sequences(model, tokenizer, tokenized_sequences, device, positions_per_chunk):
    """tokenized_sequences: list of TokenizedConditionAndDocument that fit in one padded batch. Returns a list
    of float32 numpy arrays, one per sequence, each of length (document tokens)."""
    longest = max(len(sequence.full_token_ids) for sequence in tokenized_sequences)
    input_token_ids = torch.full((len(tokenized_sequences), longest), pad_token_id_for(tokenizer),
                                 dtype=torch.long)
    attention_mask = torch.zeros((len(tokenized_sequences), longest), dtype=torch.long)
    for row, sequence in enumerate(tokenized_sequences):  # right padding keeps real positions at 0..length-1
        length = len(sequence.full_token_ids)
        input_token_ids[row, :length] = torch.tensor(sequence.full_token_ids, dtype=torch.long)
        attention_mask[row, :length] = 1
    input_token_ids, attention_mask = input_token_ids.to(device), attention_mask.to(device)
    softcapping_value = get_final_logit_softcapping(model)
    output_head = model.get_output_embeddings()
    results = []
    with torch.inference_mode():
        hidden_states = compute_final_hidden_states(model, input_token_ids, attention_mask)
        for row, sequence in enumerate(tokenized_sequences):
            start, end = sequence.document_start_index, sequence.document_end_index
            # The hidden state at position p predicts the token at position p + 1.
            predicting_hidden_states = hidden_states[row, start - 1:end - 1]
            target_token_ids = input_token_ids[row, start:end]
            log_probabilities = compute_next_token_log_probabilities_in_chunks(
                predicting_hidden_states, target_token_ids, output_head, softcapping_value, positions_per_chunk)
            results.append(log_probabilities.cpu().numpy().astype(np.float32))
    return results


def compute_log_probabilities_with_plain_full_logits(model, tokenizer, tokenized_sequence, device):
    """Reference implementation: the model's own forward pass with full logits, one sequence."""
    input_token_ids = torch.tensor([tokenized_sequence.full_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_token_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_token_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        start, end = tokenized_sequence.document_start_index, tokenized_sequence.document_end_index
        log_probabilities = torch.log_softmax(logits[0, start - 1:end - 1], dim=-1)
        return log_probabilities.gather(1, input_token_ids[0, start:end].unsqueeze(1)).squeeze(1).cpu().numpy()


def split_into_batches(tokenized_sequences_with_indices, maximum_tokens_per_batch):
    """Greedy packing of (original_index, sequence) pairs, longest first, so that
    number_of_sequences x longest_sequence_in_batch <= maximum_tokens_per_batch (a single longer sequence
    gets its own batch)."""
    ordered = sorted(tokenized_sequences_with_indices, key=lambda pair: -len(pair[1].full_token_ids))
    batches, current_batch = [], []
    for pair in ordered:
        longest_in_batch = len(current_batch[0][1].full_token_ids) if current_batch else 0
        if current_batch and (len(current_batch) + 1) * longest_in_batch > maximum_tokens_per_batch:
            batches.append(current_batch)
            current_batch = []
        current_batch.append(pair)
    if current_batch:
        batches.append(current_batch)
    return batches


def tokenize_all_conditions_for_document(tokenizer, document, conditions, model_name,
                                         model_type=prompt_rendering.MODEL_TYPE_INSTRUCT):
    """Tokenizes the document under every condition and asserts the document token ids are identical."""
    tokenized_sequences = []
    reference_document_token_ids = None
    for condition in conditions:
        tokenized = prompt_rendering.tokenize_condition_and_document_for_model_type(
            tokenizer, condition, document["generation_prompt"], document["content"], model_name, model_type)
        document_token_ids = tokenized.full_token_ids[tokenized.document_start_index:tokenized.document_end_index]
        if reference_document_token_ids is None:
            reference_document_token_ids = document_token_ids
        elif document_token_ids != reference_document_token_ids:
            raise AssertionError(
                f"Document {document['document_id']}: document token ids under condition "
                f"{condition['condition_id']} differ from those under {conditions[0]['condition_id']}")
        tokenized_sequences.append(tokenized)
    return tokenized_sequences, reference_document_token_ids


def score_one_document(model, tokenizer, document, conditions, device, model_name, maximum_tokens_per_batch,
                       positions_per_chunk, model_type=prompt_rendering.MODEL_TYPE_INSTRUCT):
    """Returns (document_token_ids, log_probabilities of shape (conditions, document tokens),
    number of real tokens pushed through the model)."""
    tokenized_sequences, document_token_ids = tokenize_all_conditions_for_document(
        tokenizer, document, conditions, model_name, model_type)
    log_probabilities = np.zeros((len(conditions), len(document_token_ids)), dtype=np.float32)
    for batch in split_into_batches(list(enumerate(tokenized_sequences)), maximum_tokens_per_batch):
        batch_results = score_tokenized_sequences(model, tokenizer, [pair[1] for pair in batch], device,
                                                  positions_per_chunk)
        for (condition_index, _), result in zip(batch, batch_results):
            log_probabilities[condition_index] = result
    real_tokens_processed = sum(len(sequence.full_token_ids) for sequence in tokenized_sequences)
    return np.array(document_token_ids, dtype=np.int32), log_probabilities, real_tokens_processed


# ---------------------------------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------------------------------

def write_npz_atomically(path, **arrays):
    partial_path = Path(str(path) + ".partial")
    with open(partial_path, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial_path, path)


def build_completion_records(document_id, condition_ids, log_probabilities):
    return [{"document_id": document_id, "condition_id": condition_id,
             "number_of_document_tokens": int(log_probabilities.shape[1]),
             "sum_log_probability": float(log_probabilities[row].astype(np.float64).sum()),
             "mean_log_probability": float(log_probabilities[row].astype(np.float64).mean())}
            for row, condition_id in enumerate(condition_ids)]


def find_documents_already_complete(scoring_outputs_directory, documents, condition_ids):
    """Documents whose .npz exists and whose completion records cover every condition. Documents with an
    .npz but missing records (interrupted between the two writes) are returned separately so their records
    can be rebuilt from the .npz without rescoring."""
    completed_keys = find_completed_keys(scoring_outputs_directory, COMPLETION_RECORDS_FILE_NAME,
                                         ["document_id", "condition_id"])
    complete, records_missing = set(), set()
    for document in documents:
        document_id = document["document_id"]
        npz_path = scoring_outputs_directory / "per_token_log_probabilities" / f"{document_id}.npz"
        if not npz_path.is_file():
            continue
        if all((document_id, condition_id) in completed_keys for condition_id in condition_ids):
            complete.add(document_id)
        else:
            records_missing.add(document_id)
    return complete, records_missing


# ---------------------------------------------------------------------------------------------------
# Model loading and environment record
# ---------------------------------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, device, dtype_name, model_class_name, model_revision):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    auth_arguments = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    revision_arguments = {"revision": model_revision} if model_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_arguments, **revision_arguments)
    model_class = {"auto_causal_lm": AutoModelForCausalLM,
                   "auto_image_text_to_text": AutoModelForImageTextToText}[model_class_name]
    model = model_class.from_pretrained(model_name, dtype=DTYPES_BY_NAME[dtype_name], **auth_arguments,
                                        **revision_arguments)
    model.to(device).eval()
    if not hasattr(model, "base_model") or model.get_output_embeddings() is None:
        raise RuntimeError(f"{type(model).__name__} has no base_model or output embeddings; choose another "
                           f"--model-class")
    return model, tokenizer


def write_environment_record(path, model, model_name, model_revision, dtype_name, device, model_class_name):
    import transformers
    record = {
        "model_name": model_name, "model_revision_requested": model_revision,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_class_used": type(model).__name__, "model_class_option": model_class_name,
        "final_logit_softcapping": get_final_logit_softcapping(model),
        "dtype": dtype_name, "device": device, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": platform.node()}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_first_run_configuration(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_configuration file in {run_folder}")
    return json.loads(paths[0].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--model-name", default=None, help="Default: the model recorded in the run configuration")
    parser.add_argument("--model-revision", default=None, help="Optional Hugging Face revision to pin")
    parser.add_argument("--model-class", default="auto_causal_lm", choices=["auto_causal_lm", "auto_image_text_to_text"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES_BY_NAME))
    parser.add_argument("--maximum-documents", type=int, default=None,
                        help="Score at most this many documents (the first ones in documents.jsonl)")
    parser.add_argument("--maximum-tokens-per-batch", type=int, default=DEFAULT_MAXIMUM_TOKENS_PER_BATCH,
                        help="Padded tokens per forward pass; lower it if the GPU runs out of memory")
    parser.add_argument("--positions-per-head-chunk", type=int, default=DEFAULT_POSITIONS_PER_HEAD_CHUNK,
                        help="Positions per output-head chunk (each chunk holds this many x vocabulary float32 values)")
    parser.add_argument("--verify-first-document-against-full-logits", action="store_true",
                        help="Check the chunked result for the first scored document against the plain "
                             "full-logits computation (uses much more memory for one sequence)")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    run_folder = Path(arguments.run_folder)
    configuration = load_first_run_configuration(run_folder)
    model_name = arguments.model_name or configuration["model_name"]
    model_type = configuration.get("model_type", prompt_rendering.MODEL_TYPE_INSTRUCT)  # old runs: instruct
    prompt_rendering.check_model_type(model_type)
    if model_name != configuration["model_name"]:
        print(f"WARNING: scoring with {model_name}, but the run configuration (which fixed the prompts) "
              f"names {configuration['model_name']}", flush=True)

    documents = read_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")
    conditions = read_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    if arguments.maximum_documents is not None:
        documents = documents[:arguments.maximum_documents]
    condition_ids = [condition["condition_id"] for condition in conditions]

    scoring_outputs_directory = run_folder / "scoring_outputs"
    npz_directory = scoring_outputs_directory / "per_token_log_probabilities"
    npz_directory.mkdir(parents=True, exist_ok=True)
    completion_records_path = scoring_outputs_directory / COMPLETION_RECORDS_FILE_NAME
    writer = AppendOnlyJsonlWriter()

    complete, records_missing = find_documents_already_complete(scoring_outputs_directory, documents, condition_ids)
    for document_id in sorted(records_missing):
        with np.load(npz_directory / f"{document_id}.npz") as saved:
            if list(saved["condition_ids"]) != condition_ids:
                raise RuntimeError(f"{document_id}.npz has different conditions from this run; refusing to resume")
            existing = find_completed_keys(scoring_outputs_directory, COMPLETION_RECORDS_FILE_NAME,
                                           ["document_id", "condition_id"])
            for record in build_completion_records(document_id, condition_ids, saved["log_probabilities"]):
                if (document_id, record["condition_id"]) not in existing:
                    writer.append(completion_records_path, record)
        complete.add(document_id)
        print(f"Rebuilt missing completion records for {document_id} from its saved .npz", flush=True)
    documents_to_score = [document for document in documents if document["document_id"] not in complete]

    timestamp = timestamp_now()
    progress_log_path = scoring_outputs_directory / f"scoring_progress_log_{timestamp}.txt"
    print(f"{len(documents)} documents x {len(conditions)} conditions; {len(complete)} documents already done; "
          f"{len(documents_to_score)} to score with {model_name} (model type {model_type})", flush=True)
    if not documents_to_score:
        return run_folder

    model_loading_start = time.time()
    model, tokenizer = load_model_and_tokenizer(model_name, arguments.device, arguments.dtype,
                                                arguments.model_class, arguments.model_revision)
    print(f"Loaded {type(model).__name__} in {time.time() - model_loading_start:.0f}s; final logit softcapping: "
          f"{get_final_logit_softcapping(model)}", flush=True)
    write_environment_record(scoring_outputs_directory / f"scoring_environment_{timestamp}.json", model, model_name,
                             arguments.model_revision, arguments.dtype, arguments.device, arguments.model_class)

    counters = {"real_tokens_processed": 0, "document_tokens_scored": 0}
    scoring_start_time = time.time()

    def throughput_text():
        elapsed = max(time.time() - scoring_start_time, 1e-9)
        return (f"{counters['real_tokens_processed'] / elapsed:,.0f} input tokens/s, "
                f"{counters['document_tokens_scored'] / elapsed:,.0f} scored document tokens/s "
                f"(documents/min is the 'calls/min' figure)")

    reporter = TimedProgressReporter(progress_log_path, len(documents_to_score), already_done_count=len(complete),
                                     status_suffix_function=throughput_text)
    for document_number, document in enumerate(documents_to_score):
        document_id = document["document_id"]
        document_token_ids, log_probabilities, real_tokens = score_one_document(
            model, tokenizer, document, conditions, arguments.device, model_name,
            arguments.maximum_tokens_per_batch, arguments.positions_per_head_chunk, model_type)
        if not np.isfinite(log_probabilities).all() or (log_probabilities > 1e-4).any():
            raise RuntimeError(f"Document {document_id}: non-finite or positive log-probabilities found")
        if document_number == 0 and arguments.verify_first_document_against_full_logits:
            verify_chunked_against_full_logits(
                model, tokenizer, document, conditions[-1], model_name, arguments.device, log_probabilities[-1],
                arguments.positions_per_head_chunk, model_type=model_type,
                result_path=scoring_outputs_directory / f"full_logits_verification_{timestamp}.json")
        write_npz_atomically(
            npz_directory / f"{document_id}.npz", document_token_ids=document_token_ids,
            document_token_strings=np.array(prompt_rendering.decode_each_token(tokenizer, document_token_ids.tolist())),
            condition_ids=np.array(condition_ids), log_probabilities=log_probabilities)
        for record in build_completion_records(document_id, condition_ids, log_probabilities):
            writer.append(completion_records_path, record)
        counters["real_tokens_processed"] += real_tokens
        counters["document_tokens_scored"] += int(log_probabilities.size)
        reporter.note_task_finished(failed=False)
    reporter.finish()
    return run_folder


MAXIMUM_ALLOWED_MEAN_DIFFERENCE = 0.05  # bf16 rounding differs a little; a bug differs a lot
MAXIMUM_ALLOWED_MAXIMUM_DIFFERENCE = 1.0


def verify_chunked_against_full_logits(model, tokenizer, document, condition, model_name, device,
                                       chunked_log_probabilities, positions_per_chunk, result_path=None,
                                       model_type=prompt_rendering.MODEL_TYPE_INSTRUCT):
    """Compares the chunked result with the plain full-logits computation, writes the outcome to result_path
    (also when it fails) and raises if the two differ more than the allowed amounts."""
    tokenized = prompt_rendering.tokenize_condition_and_document_for_model_type(
        tokenizer, condition, document["generation_prompt"], document["content"], model_name, model_type)
    reference = compute_log_probabilities_with_plain_full_logits(model, tokenizer, tokenized, device)
    differences = np.abs(reference - chunked_log_probabilities)
    maximum_difference, mean_difference = float(differences.max()), float(differences.mean())
    print(f"Verification against plain full logits ({condition['condition_id']}): maximum absolute difference "
          f"{maximum_difference:.5f}, mean {mean_difference:.5f}, mean log-probability {reference.mean():.4f} vs {chunked_log_probabilities.mean():.4f}",
          flush=True)
    passed = mean_difference <= MAXIMUM_ALLOWED_MEAN_DIFFERENCE and maximum_difference <= MAXIMUM_ALLOWED_MAXIMUM_DIFFERENCE
    if result_path is not None:
        Path(result_path).write_text(json.dumps({
            "passed": passed, "model_type": model_type, "document_id": document["document_id"], "condition_id": condition["condition_id"],
            "number_of_document_tokens": int(len(reference)), "positions_per_head_chunk": positions_per_chunk,
            "maximum_absolute_difference": maximum_difference, "mean_absolute_difference": mean_difference,
            "full_logits_mean_log_probability": float(reference.mean()),
            "chunked_mean_log_probability": float(chunked_log_probabilities.mean()),
            "maximum_allowed_mean_difference": MAXIMUM_ALLOWED_MEAN_DIFFERENCE,
            "maximum_allowed_maximum_difference": MAXIMUM_ALLOWED_MAXIMUM_DIFFERENCE}, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"Chunked and full-logit log-probabilities differ (mean {mean_difference}, "
                           f"maximum {maximum_difference})")


if __name__ == "__main__":
    main()
