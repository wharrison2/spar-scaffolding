"""Runs on the GPU pod (or on CPU with a tiny model for tests): generates the extraction answers, then runs a forward
pass over every (chat-templated prompt + answer) and saves the mean activation over answer tokens at every layer.

    python3 scripts/persona_vector_projection/generate_answers_and_save_activations_on_pod.py --run-folder <run folder>

Stages (--stage, default both; `both` runs each stage in its own child process so that vLLM's GPU memory is released
before the Hugging Face model is loaded):
  generation   generation_jobs.jsonl -> generation_and_activation_outputs/answers/answers_shard_<n>_<run timestamp>.jsonl
  activations  answers shards        -> generation_and_activation_outputs/answer_token_mean_activations/activations_shard_<n>_<run timestamp>.npz

Resumable: a shard is written atomically (temporary file, then rename) after it is complete, and an existing shard is
skipped. A relaunch on a new pod, after the launcher has uploaded the pulled outputs, continues from the next shard.
Sampling: temperature 1, top-p 1, top-k disabled (vLLM and Hugging Face would otherwise apply the model's own default
top-k from its generation config), at most 1,000 new tokens. Progress reports with timings about every 3 minutes.

Activations: see answer_token_activation_recording.py (hooks, no language-model head, float32 accumulation, stored
in float16 by default, with a loud failure if any value overflows float16).
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

# vLLM forks its engine process by default, which fails once CUDA is initialised in the parent ("Cannot re-initialize
# CUDA in forked subprocess"); spawn avoids that. Set before vLLM is imported.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import chat_prompt_rendering_for_extraction as rendering  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402
from shared.timed_progress_reporter import TimedProgressReporter, format_duration  # noqa: E402

DEFAULT_JOBS_PER_SHARD = 40
DEFAULT_ACTIVATION_BATCH_SIZE = 8  # conservative: sequences of up to ~1,400 tokens, 48 layers, hooks in fp32
VLLM_MAXIMUM_MODEL_LENGTH = 4096
ANSWER_SHARD_PATTERN = "answers_shard_{shard_number:05d}_{run_timestamp}.jsonl"
ACTIVATION_SHARD_PATTERN = "activations_shard_{shard_number:05d}_{run_timestamp}.npz"


def load_run_configuration(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_configuration file in {run_folder}")
    return json.loads(paths[0].read_text(encoding="utf-8"))


def split_jobs_into_shards(jobs, jobs_per_shard):
    return [jobs[start:start + jobs_per_shard] for start in range(0, len(jobs), jobs_per_shard)]


def answer_identifier(job, sample_index):
    return f"{job['job_id']}__sample_{sample_index:02d}"


def write_file_atomically(path, writer_function):
    path = Path(path)
    temporary_path = path.with_name(f"{path.stem}.partial{path.suffix}")
    writer_function(temporary_path)
    os.replace(temporary_path, path)


# ---------------------------------------------------------------------------------------------------
# Generation backends. A backend has generate(list of prompt token id lists, samples) ->
# list over prompts of list over samples of {"text", "number_of_tokens", "finish_reason"}.
# ---------------------------------------------------------------------------------------------------

class VllmGenerationBackend:
    def __init__(self, model_name, model_revision, random_seed, maximum_new_tokens):
        from vllm import LLM
        import vllm
        self.vllm_version = vllm.__version__
        self.maximum_new_tokens = maximum_new_tokens
        revision_arguments = {"revision": model_revision, "tokenizer_revision": model_revision} if model_revision else {}
        self.llm = LLM(model=model_name, dtype="bfloat16", max_model_len=VLLM_MAXIMUM_MODEL_LENGTH,
                       gpu_memory_utilization=0.90, seed=random_seed, enable_prefix_caching=True, **revision_arguments)

    def generate(self, prompt_token_id_lists, samples_per_prompt):
        from vllm import SamplingParams
        sampling_parameters = SamplingParams(n=samples_per_prompt, temperature=defs.SAMPLING_TEMPERATURE,
                                             top_p=defs.SAMPLING_TOP_P, top_k=-1, max_tokens=self.maximum_new_tokens)
        outputs = self.llm.generate([{"prompt_token_ids": ids} for ids in prompt_token_id_lists], sampling_parameters)
        return [[{"text": candidate.text, "number_of_tokens": len(candidate.token_ids),
                  "finish_reason": candidate.finish_reason} for candidate in output.outputs] for output in outputs]


class HuggingFaceGenerationBackend:
    """Slower fallback (and what the local tiny-model rehearsal uses). Left-padded batches of prompts."""

    def __init__(self, model, tokenizer, device, maximum_new_tokens, prompts_per_batch=4):
        import torch
        self.torch, self.model, self.tokenizer, self.device = torch, model, tokenizer, device
        self.maximum_new_tokens, self.prompts_per_batch = maximum_new_tokens, prompts_per_batch

    def generate(self, prompt_token_id_lists, samples_per_prompt):
        torch = self.torch
        results = []
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        for start in range(0, len(prompt_token_id_lists), self.prompts_per_batch):
            batch = prompt_token_id_lists[start:start + self.prompts_per_batch]
            longest = max(len(ids) for ids in batch)
            input_ids = torch.full((len(batch), longest), padding_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
            for row, ids in enumerate(batch):
                input_ids[row, longest - len(ids):] = torch.tensor(ids)
                attention_mask[row, longest - len(ids):] = 1
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), do_sample=True,
                    temperature=defs.SAMPLING_TEMPERATURE, top_p=defs.SAMPLING_TOP_P, top_k=0,
                    max_new_tokens=self.maximum_new_tokens, num_return_sequences=samples_per_prompt,
                    pad_token_id=padding_id)
            new_tokens = generated[:, longest:].cpu().tolist()
            for prompt_index in range(len(batch)):
                samples = []
                for sample_tokens in new_tokens[prompt_index * samples_per_prompt:(prompt_index + 1) * samples_per_prompt]:
                    ended = self.tokenizer.eos_token_id in sample_tokens
                    if ended:
                        sample_tokens = sample_tokens[:sample_tokens.index(self.tokenizer.eos_token_id)]
                    else:
                        while sample_tokens and sample_tokens[-1] == padding_id:
                            sample_tokens = sample_tokens[:-1]
                    samples.append({"text": self.tokenizer.decode(sample_tokens, skip_special_tokens=True),
                                    "number_of_tokens": len(sample_tokens),
                                    "finish_reason": "stop" if ended else "length"})
                results.append(samples)
        return results


# ---------------------------------------------------------------------------------------------------
# Stage 1: generation
# ---------------------------------------------------------------------------------------------------

def run_generation_stage(run_folder, jobs, tokenizer, backend, jobs_per_shard, samples_per_prompt_override=None,
                         model_name=None, report_interval_seconds=180):
    outputs_folder = Path(run_folder) / defs.OUTPUTS_FOLDER_NAME
    answers_folder = outputs_folder / defs.ANSWERS_FOLDER_NAME
    answers_folder.mkdir(parents=True, exist_ok=True)
    run_timestamp = defs.read_run_timestamp(run_folder)
    shards = split_jobs_into_shards(jobs, jobs_per_shard)
    remaining = [(number, shard) for number, shard in enumerate(shards)
                 if not (answers_folder / ANSWER_SHARD_PATTERN.format(shard_number=number, run_timestamp=run_timestamp)).exists()]
    progress_log = outputs_folder / f"generation_progress_log_{timestamp_now()}.txt"
    totals = {"answers": 0, "generated_tokens": 0}
    start_time = time.time()
    reporter = TimedProgressReporter(
        progress_log, len(remaining), already_done_count=len(shards) - len(remaining),
        status_suffix_function=lambda: (f"answers generated {totals['answers']}, generated tokens "
                                        f"{totals['generated_tokens']}, "
                                        f"{totals['generated_tokens'] / max(1e-9, time.time() - start_time):.0f} tokens/second"),
        report_interval_seconds=report_interval_seconds)
    print(f"Generation: {len(shards)} shards, {len(shards) - len(remaining)} already done", flush=True)
    for shard_number, shard_jobs in remaining:
        prompt_ids = [rendering.prompt_token_ids(tokenizer, job["system_prompt"], job["user_prompt"], model_name)
                      for job in shard_jobs]
        samples_per_prompt = samples_per_prompt_override or shard_jobs[0]["samples_per_prompt"]
        generations = backend.generate(prompt_ids, samples_per_prompt)
        records = []
        for job, prompt_token_list, samples in zip(shard_jobs, prompt_ids, generations):
            for sample_index, sample in enumerate(samples):
                records.append({**{key: job[key] for key in (
                    "facet", "pole", "tag", "system_prompt_id", "question_id", "pair_id", "version", "half", "domain")},
                    "answer_id": answer_identifier(job, sample_index), "job_id": job["job_id"],
                    "sample_index": sample_index, "answer_text": sample["text"],
                    "number_of_answer_tokens_generated": sample["number_of_tokens"],
                    "finish_reason": sample["finish_reason"], "number_of_prompt_tokens": len(prompt_token_list)})
                totals["answers"] += 1
                totals["generated_tokens"] += sample["number_of_tokens"]
        write_file_atomically(answers_folder / ANSWER_SHARD_PATTERN.format(shard_number=shard_number, run_timestamp=run_timestamp),
                              lambda path: write_jsonl(path, records))
        reporter.note_task_finished(failed=False)
    reporter.finish()
    return totals


# ---------------------------------------------------------------------------------------------------
# Stage 2: activations
# ---------------------------------------------------------------------------------------------------

def compute_activation_shard(answer_records, jobs_by_id, tokenizer, model, recorder, device, batch_size,
                             storage_dtype, model_name=None):
    import torch
    from persona_vector_projection.answer_token_activation_recording import build_padded_batch
    prepared = []
    for record in answer_records:
        job = jobs_by_id[record["job_id"]]
        prompt_ids = rendering.prompt_token_ids(tokenizer, job["system_prompt"], job["user_prompt"], model_name)
        answer_ids = rendering.answer_token_ids(tokenizer, record["answer_text"])
        prepared.append((record["answer_id"], prompt_ids, answer_ids))
    order = sorted(range(len(prepared)), key=lambda index: len(prepared[index][1]) + len(prepared[index][2]))
    layer_count = recorder.number_of_recorded_layers
    activations = np.zeros((len(prepared), layer_count, model.config.get_text_config().hidden_size), dtype=storage_dtype)
    answer_token_counts = np.zeros(len(prepared), dtype=np.int32)
    padding_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        sequences = [prepared[i][1] + prepared[i][2] for i in indices]
        input_ids, attention_mask, answer_mask = build_padded_batch(
            sequences, [len(prepared[i][1]) for i in indices], padding_id)
        means = recorder.mean_activations_for_batch(input_ids.to(device), attention_mask.to(device), answer_mask.to(device))
        means = means.cpu().numpy()
        if storage_dtype == np.float16 and not np.isfinite(means.astype(np.float16)).all():
            raise FloatingPointError("A mean activation overflows float16 (maximum 65504). Re-run with "
                                     "--activation-storage-dtype float32.")
        for row, index in enumerate(indices):
            activations[index] = means[row].astype(storage_dtype)
            answer_token_counts[index] = len(prepared[index][2])
    return np.array([item[0] for item in prepared]), activations, answer_token_counts


def run_activation_stage(run_folder, jobs, tokenizer, model, device, batch_size, storage_dtype_name,
                         model_name=None, report_interval_seconds=180):
    from persona_vector_projection.answer_token_activation_recording import AnswerTokenMeanActivationRecorder
    outputs_folder = Path(run_folder) / defs.OUTPUTS_FOLDER_NAME
    answers_folder = outputs_folder / defs.ANSWERS_FOLDER_NAME
    activations_folder = outputs_folder / defs.ACTIVATIONS_FOLDER_NAME
    activations_folder.mkdir(parents=True, exist_ok=True)
    jobs_by_id = {job["job_id"]: job for job in jobs}
    storage_dtype = {"float16": np.float16, "float32": np.float32}[storage_dtype_name]
    answer_shard_paths = sorted(answers_folder.glob("answers_shard_*.jsonl"))
    activation_destination = lambda path: activations_folder / path.name.replace(  # noqa: E731
        "answers_shard_", "activations_shard_").replace(".jsonl", ".npz")
    remaining = [path for path in answer_shard_paths if not activation_destination(path).exists()]
    progress_log = outputs_folder / f"activation_progress_log_{timestamp_now()}.txt"
    done = {"answers": 0}
    start_time = time.time()
    reporter = TimedProgressReporter(
        progress_log, len(remaining), already_done_count=len(answer_shard_paths) - len(remaining),
        status_suffix_function=lambda: (f"answers with activations {done['answers']}, "
                                        f"{done['answers'] / max(1e-9, time.time() - start_time):.1f} answers/second"),
        report_interval_seconds=report_interval_seconds)
    recorder = AnswerTokenMeanActivationRecorder(model)
    recorder.attach()
    try:
        for path in remaining:
            records = read_jsonl(path)
            answer_ids, activations, counts = compute_activation_shard(
                records, jobs_by_id, tokenizer, model, recorder, device, batch_size, storage_dtype, model_name)
            destination = activation_destination(path)
            write_file_atomically(destination, lambda temporary: np.savez(
                temporary, answer_ids=answer_ids, activations=activations, answer_token_counts=counts))
            done["answers"] += len(records)
            reporter.note_task_finished(failed=False)
    finally:
        recorder.detach()
    reporter.finish()
    return done


# ---------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------

def load_tokenizer(model_name, model_revision):
    from transformers import AutoTokenizer
    token = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    revision = {"revision": model_revision} if model_revision else {}
    return AutoTokenizer.from_pretrained(model_name, **token, **revision)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--stage", choices=["both", "generation", "activations"], default="both")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--model-class", default="auto_causal_lm", choices=["auto_causal_lm", "auto_image_text_to_text"])
    parser.add_argument("--jobs-per-shard", type=int, default=DEFAULT_JOBS_PER_SHARD)
    parser.add_argument("--activation-batch-size", type=int, default=DEFAULT_ACTIVATION_BATCH_SIZE)
    parser.add_argument("--activation-storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--maximum-jobs", type=int, default=None, help="Testing only: use the first N jobs")
    parser.add_argument("--maximum-new-tokens", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    run_folder = Path(arguments.run_folder)
    configuration = load_run_configuration(run_folder)
    model_name = arguments.model_name or configuration["model_name"]
    model_revision = arguments.model_revision or configuration["model_revision"]
    backend_name = arguments.generation_backend or configuration["generation_backend"]
    maximum_new_tokens = arguments.maximum_new_tokens or configuration["maximum_new_tokens"]
    jobs = read_jsonl(defs.find_latest_file(run_folder / defs.INPUTS_FOLDER_NAME, "generation_jobs", "jsonl"))
    if arguments.maximum_jobs:
        jobs = jobs[:arguments.maximum_jobs]
    outputs_folder = run_folder / defs.OUTPUTS_FOLDER_NAME
    outputs_folder.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "both":
        original_arguments = list(argv if argv is not None else sys.argv[1:])
        forwarded, skip_next = [], False
        for part in original_arguments:
            if skip_next:
                skip_next = False
            elif part == "--stage":
                skip_next = True
            else:
                forwarded.append(part)
        for stage in ("generation", "activations"):
            print(f"Starting stage {stage} in its own process", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *forwarded, "--stage", stage], check=True)
        return 0
    tokenizer = load_tokenizer(model_name, model_revision)
    system_role_path = outputs_folder / f"chat_template_system_role_handling_{timestamp_now()}.json"
    if not list(outputs_folder.glob("chat_template_system_role_handling_*.json")):
        print("Chat template system role check:", json.dumps(
            rendering.record_system_role_handling(tokenizer, model_name, system_role_path))[:400], flush=True)
    if arguments.stage == "generation":
        if backend_name == "vllm":
            backend = VllmGenerationBackend(model_name, model_revision, configuration["random_seed"], maximum_new_tokens)
            extra_environment = {"vllm_version": backend.vllm_version}
        else:
            sys.path.insert(0, str(SCRIPTS_DIRECTORY / "log_probability_persona_check"))
            import score_document_log_probabilities_on_pod as scorer_module
            model, tokenizer = scorer_module.load_model_and_tokenizer(
                model_name, arguments.device, arguments.dtype, arguments.model_class, model_revision)
            backend = HuggingFaceGenerationBackend(model, tokenizer, arguments.device, maximum_new_tokens)
            extra_environment = {}
        (outputs_folder / f"generation_environment_{timestamp_now()}.json").write_text(json.dumps({
            "model_name": model_name, "model_revision": model_revision, "backend": backend_name,
            "python_version": platform.python_version(), "hostname": platform.node(), **extra_environment}, indent=2))
        totals = run_generation_stage(run_folder, jobs, tokenizer, backend, arguments.jobs_per_shard, model_name=model_name)
        print(f"Generation finished: {totals}", flush=True)
    else:
        sys.path.insert(0, str(SCRIPTS_DIRECTORY / "log_probability_persona_check"))
        import score_document_log_probabilities_on_pod as scorer_module
        model, tokenizer = scorer_module.load_model_and_tokenizer(
            model_name, arguments.device, arguments.dtype, arguments.model_class, model_revision)
        scorer_module.write_environment_record(
            outputs_folder / f"activation_environment_{timestamp_now()}.json", model, model_name, model_revision,
            arguments.dtype, arguments.device, arguments.model_class)
        done = run_activation_stage(run_folder, jobs, tokenizer, model, arguments.device, arguments.activation_batch_size,
                                    arguments.activation_storage_dtype, model_name)
        print(f"Activation stage finished: {done}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
