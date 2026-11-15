"""Runs on the GPU pod (or on CPU with a tiny model for the local rehearsal): generates the baseline answers of the three
candidate models to the fifty severe, implicit welfare prompts, with thinking off and thinking on.

    python3 scripts/emergent_misalignment/generate_baseline_answers_for_candidate_models_on_pod.py --run-folder <run folder>
    ... --one-answer-check        # standalone: only the checks, for every model, always exit 0 (the normal run does not need it)

Per model repository (entries grouped by repository, repositories in the order of the run configuration's model_order):
  1. wait for the repository's weights (downloaded in the background while the previous repository generated),
  2. start the next repository's download in the background,
  3. generate in a child process (so vLLM's GPU memory is released when the repository is done) that handles every entry of the
     repository in turn with one engine, skipping entries whose answers are complete: for each entry, first a one-answer check (one
     answer to the first prompt in each condition: the thinking switch works, the trace splits cleanly; results go to
     generation_outputs/one_answer_checks/, never to the answer files; a failed check exits non-zero with a clear message
     so the launcher's failure handling takes over), then both conditions together in one engine pass (every thinking_on
     request is submitted first, then every thinking_off request, so the short thinking_off answers fill the GPU while the
     long thinking_on answers finish; each request has its own output cap and its own answers file),
  4. delete the repository's Hugging Face cache (unless --keep-model-caches), so the volume holds at most two repositories.

Answers go to <run folder>/generation_outputs/answers__<model_short_name>__<condition>.jsonl, one record per line, appended
as each answer finishes. Resumable: a relaunch skips answer_ids already in the files (a half-written last line is cut off).
Sampling: temperature 1, top-p 1, top-k disabled, no system prompt, the prompt as a single user turn, bfloat16, text only.
Progress reports with timings about every 3 minutes (also written to generation_progress_log_*.txt).

LoRA adapters: a model entry of the run configuration may carry "lora_adapter": {"path_inside_run_folder": ...,
"adapter_weights_sha256": optional}, a standard peft adapter directory (adapter_config.json + adapter_model.safetensors,
uploaded inside the run folder, normally under generation_inputs/). The entry is then the base model plus that adapter. A
repository's base entry and adapter entries share one engine: vLLM starts once with enable_lora, max_loras equal to the number of
distinct adapters and max_lora_rank rounded up from the largest adapter rank, and the backend's use_lora_adapter selects the
LoRARequest that every following request carries (none for a base entry); the Hugging Face backend loads each adapter lazily with
peft, selects it with set_adapter and runs base entries with the adapters disabled. Answer records of an adapter entry gain
lora_adapter_path_inside_run_folder and lora_adapter_weights_sha256 (records of entries without an adapter are unchanged).
The repository's weights are downloaded once and its cache is deleted after its last entry.

Backends: vLLM (default) with the engine stepped by hand, so each answer is saved the moment it finishes; or Hugging Face
transformers generation (--generation-backend huggingface), slower, for a model vLLM cannot load and for the local rehearsal.
"""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# vLLM forks its engine process by default, which fails once CUDA is initialised in the parent; spawn avoids that.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment.thinking_trace_splitting_for_candidate_models import split_generated_text  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now, timestamp_now_with_seconds  # noqa: E402
from shared.timed_progress_reporter import PROGRESS_REPORT_INTERVAL_SECONDS, format_duration  # noqa: E402

DEFAULT_CHECK_OUTPUT_CAPS = {defs.CONDITION_THINKING_OFF: 4096, defs.CONDITION_THINKING_ON: 16384}


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def say(message, log_path=None):
    line = f"[{now_text()}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ---------------------------------------------------------------------------------------------------
# Run configuration, prompts, answer files
# ---------------------------------------------------------------------------------------------------

def load_run_configuration(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_configuration file in {run_folder}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def load_prompts(run_folder, configuration, maximum_prompts=None):
    prompts = read_jsonl(Path(run_folder) / configuration["prompts_file"]["path_inside_run_folder"])
    return prompts[:maximum_prompts] if maximum_prompts else prompts


def answer_file_path(run_folder, model_short_name, condition):
    return Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / defs.ANSWER_FILE_PATTERN.format(
        model_short_name=model_short_name, condition=condition)


def read_finished_answer_ids(path):
    """answer_ids already in the file. A last line cut short by a crash is removed, so appending stays well formed."""
    path = Path(path)
    if not path.exists():
        return set()
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        last_newline = raw.rfind(b"\n")
        path.write_bytes(raw[:last_newline + 1] if last_newline >= 0 else b"")
        say(f"Removed a half-written last line from {path.name}")
        raw = path.read_bytes()
    return {json.loads(line)["answer_id"] for line in raw.decode("utf-8").splitlines() if line.strip()}


def append_answer_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def remaining_answers(model_short_name, condition, prompts, samples_per_prompt, finished_answer_ids):
    """(prompt, sample_index) pairs whose answer is not yet saved, in prompt order."""
    return [(prompt, sample_index) for prompt in prompts for sample_index in range(samples_per_prompt)
            if defs.build_answer_id(model_short_name, condition, prompt["prompt_gid"], sample_index) not in finished_answer_ids]


def conditions_defined_by_run_configuration(configuration):
    """True when the run configuration's conditions carry their own chat template settings (the held-out set evaluation);
    False for the 3 November screen, whose conditions are the models' thinking_off and thinking_on specs."""
    return any("chat_template_kwargs" in entry for entry in configuration["conditions"])


def condition_names_for_model(configuration, model):
    """The run configuration's condition names that this model entry runs: all of them, or only those in the entry's own "condition_names"."""
    names = [entry["condition"] for entry in configuration["conditions"]]
    return [name for name in names if name in model["condition_names"]] if model.get("condition_names") else names


def condition_entries_for_model(configuration, model, output_cap_overrides=None):
    """The conditions to run for one model, in the run configuration's order: a list of
    {"condition", "condition_spec", "output_cap"}.

    3 November screen (default): condition_spec is the model's thinking_off or thinking_on entry.
    Conditions defined by the run configuration: condition_spec holds the condition's own chat_template_kwargs, optional
    system_prompt_text (stored in the configuration when the run folder was built), the thinking-trace splitting to use
    (split_as_condition, default thinking_off: no trace is expected, any trace found is recorded and counted), and the
    rendered-prompt requirements the one-answer check asserts."""
    output_cap_overrides = output_cap_overrides or {}
    entries = []
    for entry in configuration["conditions"]:
        condition = entry["condition"]
        if condition not in condition_names_for_model(configuration, model):
            continue
        if conditions_defined_by_run_configuration(configuration):
            condition_spec = {
                "chat_template_kwargs": entry["chat_template_kwargs"], "generation_prompt_suffix": "",
                "system_prompt_text": entry.get("system_prompt_text"),
                "is_true_thinking_off": entry.get("is_true_thinking_off", True),
                "split_as_condition": entry.get("split_as_condition", defs.CONDITION_THINKING_OFF),
                "required_rendered_prompt_tail": entry.get("required_rendered_prompt_tail"),
                "required_system_turn_opening": entry.get("required_system_turn_opening"),
                "required_rendered_prompt_substrings": entry.get("required_rendered_prompt_substrings", []),
                "forbidden_rendered_prompt_substrings": entry.get("forbidden_rendered_prompt_substrings", [])}
        else:
            condition_spec = model["thinking_on" if condition == defs.CONDITION_THINKING_ON else "thinking_off"]
        entries.append({"condition": condition, "condition_spec": condition_spec,
                        "output_cap": output_cap_overrides.get(condition) or entry["output_cap"]})
    return entries


def number_of_remaining_answers_for_model(run_folder, model, prompts, samples_per_prompt, condition_names=None):
    total = 0
    for condition in condition_names or defs.CONDITIONS_IN_ORDER:
        finished = read_finished_answer_ids(answer_file_path(run_folder, model["model_short_name"], condition))
        total += len(remaining_answers(model["model_short_name"], condition, prompts, samples_per_prompt, finished))
    return total


def build_chat_messages(user_message, condition_spec):
    """The prompt's user message as the only user turn, preceded by a system message when the condition has one."""
    messages = []
    if condition_spec.get("system_prompt_text"):
        messages.append({"role": "system", "content": condition_spec["system_prompt_text"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def rendered_prompt_requirement_failures(rendered_prompt_text, condition_spec):
    """What the one-answer check asserts about the rendered prompt, as a list of failure descriptions (empty when all hold):
    the rendered prompt must end with the condition's required tail, a condition with a system prompt must render its
    text inside a system turn, and the condition's required and forbidden substrings must be present and absent."""
    failures = []
    required_tail = condition_spec.get("required_rendered_prompt_tail")
    if required_tail and not rendered_prompt_text.endswith(required_tail):
        failures.append(f"the rendered prompt does not end with {required_tail!r} (it ends with {rendered_prompt_text[-80:]!r})")
    system_text = condition_spec.get("system_prompt_text")
    if system_text:
        system_turn_opening = condition_spec.get("required_system_turn_opening") or ""
        if system_turn_opening + system_text not in rendered_prompt_text:
            failures.append(f"the rendered prompt does not contain {system_turn_opening!r} followed by the system prompt text")
    for required in condition_spec.get("required_rendered_prompt_substrings", []):
        if required not in rendered_prompt_text:
            failures.append(f"the rendered prompt does not contain {required!r}")
    for forbidden in condition_spec.get("forbidden_rendered_prompt_substrings", []):
        if forbidden in rendered_prompt_text:
            failures.append(f"the rendered prompt contains {forbidden!r}, which this condition must not have")
    return failures


# ---------------------------------------------------------------------------------------------------
# LoRA adapters
# ---------------------------------------------------------------------------------------------------

ADAPTER_CONFIGURATION_FILE_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_FILE_NAME = "adapter_model.safetensors"
VLLM_SUPPORTED_MAXIMUM_LORA_RANKS = [8, 16, 32, 64, 128, 256, 320, 512]


def sha256_of_file_in_chunks(path, chunk_bytes=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_lora_rank_for_vllm(adapter_rank):
    """The smallest rank vLLM accepts for max_lora_rank that is at least the adapter's rank."""
    for supported in VLLM_SUPPORTED_MAXIMUM_LORA_RANKS:
        if supported >= adapter_rank:
            return supported
    raise ValueError(f"LoRA rank {adapter_rank} is larger than vLLM's supported maximum {VLLM_SUPPORTED_MAXIMUM_LORA_RANKS[-1]}")


def resolve_lora_adapter(model, run_folder):
    """None for a model entry without "lora_adapter". Otherwise the adapter's absolute folder, path inside the run folder,
    SHA-256 of adapter_model.safetensors and rank. Fails when a file is missing or when the configuration records a SHA-256 that
    the uploaded weights do not match (a corrupted or wrong upload would otherwise be recorded as the adapter)."""
    entry = model.get("lora_adapter")
    if not entry:
        return None
    relative_path = entry["path_inside_run_folder"]
    folder = Path(run_folder) / relative_path
    for required_file_name in (ADAPTER_CONFIGURATION_FILE_NAME, ADAPTER_WEIGHTS_FILE_NAME):
        if not (folder / required_file_name).is_file():
            raise FileNotFoundError(f"LoRA adapter folder {folder} for {model['model_short_name']} lacks {required_file_name}")
    weights_sha256 = sha256_of_file_in_chunks(folder / ADAPTER_WEIGHTS_FILE_NAME)
    expected = entry.get("adapter_weights_sha256")
    if expected and expected != weights_sha256:
        raise RuntimeError(f"LoRA adapter weights for {model['model_short_name']} have SHA-256 {weights_sha256}, but the run "
                           f"configuration records {expected}; the upload is not the adapter that was configured")
    adapter_configuration = json.loads((folder / ADAPTER_CONFIGURATION_FILE_NAME).read_text(encoding="utf-8"))
    rank = max([adapter_configuration.get("r") or 0, *(adapter_configuration.get("rank_pattern") or {}).values()])
    if rank <= 0:
        raise ValueError(f"{folder / ADAPTER_CONFIGURATION_FILE_NAME} has no LoRA rank r")
    return {"folder": str(folder), "path_inside_run_folder": relative_path, "weights_sha256": weights_sha256, "rank": int(rank),
            "name": model["model_short_name"]}


def distinct_lora_adapters(lora_adapters):
    """The resolved adapters (None entries dropped) without repeats of the same folder, in the given order."""
    distinct = {}
    for lora_adapter in lora_adapters:
        if lora_adapter is not None:
            distinct.setdefault(lora_adapter["folder"], lora_adapter)
    return list(distinct.values())


def group_models_by_repository(models):
    """Model entries grouped by model_repository, each group in the given order, groups in order of first appearance."""
    groups = {}
    for model in models:
        groups.setdefault(model["model_repository"], []).append(model)
    return list(groups.values())


# ---------------------------------------------------------------------------------------------------
# Turning generated tokens into the answer record
# ---------------------------------------------------------------------------------------------------

def strip_trailing_end_of_sequence_tokens(token_ids, end_of_sequence_ids):
    token_ids = list(token_ids)
    while token_ids and token_ids[-1] in end_of_sequence_ids:
        token_ids.pop()
    return token_ids


def build_answer_record(model, condition, condition_spec, prompt, sample_index, generated_token_ids, finish_reason,
                        output_cap, decode_function, end_of_sequence_ids, finished_at=None, lora_adapter=None):
    text = decode_function(strip_trailing_end_of_sequence_tokens(generated_token_ids, end_of_sequence_ids))
    ended_naturally = finish_reason == "stop"
    split = split_generated_text(model["template_family"], condition_spec.get("split_as_condition", condition), text, ended_naturally,
                                 condition_spec.get("generation_prompt_suffix", ""))
    record = {
        "answer_id": defs.build_answer_id(model["model_short_name"], condition, prompt["prompt_gid"], sample_index),
        "model_short_name": model["model_short_name"], "model_repository": model["model_repository"],
        "model_revision": model["model_revision"], "condition": condition,
        "prompt_gid": prompt["prompt_gid"], "scenario_id": prompt["scenario_id"], "sample_index": sample_index,
        "user_message": prompt["user_message"],
        "final_answer_text": split["final_answer_text"], "thinking_trace_text": split["thinking_trace_text"],
        "thinking_split_succeeded": split["thinking_split_succeeded"],
        "number_of_generated_tokens": len(generated_token_ids), "output_cap": output_cap,
        "hit_output_cap": finish_reason == "length", "finish_reason": finish_reason,
        "generation_finished_at": (finished_at or datetime.now()).isoformat(timespec="seconds"),
    }
    if lora_adapter is not None:
        record["lora_adapter_path_inside_run_folder"] = lora_adapter["path_inside_run_folder"]
        record["lora_adapter_weights_sha256"] = lora_adapter["weights_sha256"]
    return record


# ---------------------------------------------------------------------------------------------------
# Tokenizer adapter
# ---------------------------------------------------------------------------------------------------

class ChatTokenizerAdapter:
    """Renders the prompt through the model's own chat template (thinking switches are template variables) and decodes
    generated tokens with the special tokens kept, because the thinking delimiters are special tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prompt_token_ids(self, user_message, condition_spec):
        text = self.tokenizer.apply_chat_template(
            build_chat_messages(user_message, condition_spec), tokenize=False, add_generation_prompt=True,
            **condition_spec.get("chat_template_kwargs", {})) + condition_spec.get("generation_prompt_suffix", "")
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def count_tokens(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------------------------------
# Generation backends: run_requests(requests, on_finished, on_heartbeat)
# requests: list of {"request_id", "prompt_token_ids", "maximum_new_tokens"} (the cap can differ per request, because the
# two conditions run together); on_finished(request_id, token_ids, finish_reason) is called as each
# request finishes (token_ids include the stop token when one was produced); finish_reason is "stop" or "length".
# use_lora_adapter(resolved adapter or None) selects the adapter that the following run_requests calls generate with (None: the
# base model, in the same engine).
# ---------------------------------------------------------------------------------------------------

class VllmGenerationBackend:
    """One vLLM engine per repository. `lora_adapters` are all adapters that will be used with it; each gets a stable integer id."""

    def __init__(self, model_repository, model_revision, maximum_model_length, gpu_memory_utilization, random_seed=0,
                 lora_adapters=()):
        import vllm
        from vllm import LLM
        self.vllm_version = vllm.__version__
        self.lora_request = None
        self.lora_requests_by_folder = {}
        lora_adapters = distinct_lora_adapters(lora_adapters)
        load_start = time.time()
        common = dict(model=model_repository, revision=model_revision, tokenizer_revision=model_revision, dtype="bfloat16",
                      max_model_len=maximum_model_length, gpu_memory_utilization=gpu_memory_utilization, seed=random_seed,
                      enable_prefix_caching=True)
        if lora_adapters:
            from vllm.lora.request import LoRARequest
            common.update(enable_lora=True, max_loras=len(lora_adapters),
                          max_lora_rank=maximum_lora_rank_for_vllm(max(lora_adapter["rank"] for lora_adapter in lora_adapters)))
            self.lora_requests_by_folder = {lora_adapter["folder"]: LoRARequest(lora_adapter["name"], adapter_id, lora_adapter["folder"])
                                            for adapter_id, lora_adapter in enumerate(lora_adapters, start=1)}
        try:
            self.llm = LLM(language_model_only=True, **common)  # skips every image and video input
        except TypeError:  # a vLLM without the shortcut: set every modality limit to zero instead
            self.llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **common)
        say(f"vLLM {self.vllm_version} loaded {model_repository} in {format_duration(time.time() - load_start)}")

    def use_lora_adapter(self, lora_adapter):
        self.lora_request = None if lora_adapter is None else self.lora_requests_by_folder[lora_adapter["folder"]]

    def run_requests(self, requests, on_finished, on_heartbeat):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        engine = self.llm.llm_engine
        for request in requests:  # submitted in the given order; only max_tokens differs between requests
            sampling_parameters = SamplingParams(  # a request may carry its own temperature and top_p (the judges use temperature 0)
                n=1, temperature=request.get("temperature", defs.SAMPLING_TEMPERATURE), top_p=request.get("top_p", defs.SAMPLING_TOP_P),
                top_k=-1, min_p=0.0,
                max_tokens=request["maximum_new_tokens"], skip_special_tokens=False, output_kind=RequestOutputKind.FINAL_ONLY)
            engine.add_request(request["request_id"], {"prompt_token_ids": request["prompt_token_ids"]}, sampling_parameters,
                               lora_request=self.lora_request)
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    candidate = output.outputs[0]
                    on_finished(output.request_id, list(candidate.token_ids), candidate.finish_reason)
            on_heartbeat()


class HuggingFaceGenerationBackend:
    """Slower fallback and the local rehearsal's backend: left-padded batches, one batch at a time. Requests with different
    output caps are never batched together: they are grouped by cap, in the order the caps first appear. Adapters are attached
    with peft the first time use_lora_adapter selects them; base requests run with the adapters disabled."""

    def __init__(self, model_repository, model_revision, device, end_of_sequence_ids, prompts_per_batch=4):
        import torch
        import transformers
        self.torch, self.device, self.prompts_per_batch = torch, device, prompts_per_batch
        self.end_of_sequence_ids = end_of_sequence_ids
        self.adapter_names_by_folder, self.selected_adapter_name = {}, None
        load_start = time.time()
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        try:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_repository, revision=model_revision, dtype=dtype)
        except (ValueError, KeyError):  # multimodal checkpoints without a causal language model mapping
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(model_repository, revision=model_revision, dtype=dtype)
        self.model.to(device).eval()
        say(f"transformers {transformers.__version__} loaded {model_repository} on {device} in "
            f"{format_duration(time.time() - load_start)}")

    def use_lora_adapter(self, lora_adapter):
        if lora_adapter is None:
            self.selected_adapter_name = None
            return
        folder = lora_adapter["folder"]
        if folder not in self.adapter_names_by_folder:
            import peft
            adapter_name = f"lora_adapter_{len(self.adapter_names_by_folder) + 1}"
            if self.adapter_names_by_folder:
                self.model.load_adapter(folder, adapter_name=adapter_name)
            else:
                self.model = peft.PeftModel.from_pretrained(self.model, folder, adapter_name=adapter_name)
            self.model.to(self.device).eval()
            self.adapter_names_by_folder[folder] = adapter_name
            say(f"peft {peft.__version__} attached LoRA adapter {lora_adapter['path_inside_run_folder']} "
                f"(weights SHA-256 {lora_adapter['weights_sha256'][:12]})")
        self.selected_adapter_name = self.adapter_names_by_folder[folder]
        self.model.set_adapter(self.selected_adapter_name)

    def selected_adapter_context(self):
        """Disables the attached adapters while a base entry is selected; otherwise changes nothing."""
        if self.selected_adapter_name is None and self.adapter_names_by_folder:
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    def run_requests(self, requests, on_finished, on_heartbeat):
        requests_by_cap = {}
        for request in requests:
            requests_by_cap.setdefault(request["maximum_new_tokens"], []).append(request)
        for maximum_new_tokens, requests_with_this_cap in requests_by_cap.items():
            requests_by_temperature = {}  # a batch shares one sampling setting, so requests with their own temperature are split off
            for request in requests_with_this_cap:
                requests_by_temperature.setdefault(request.get("temperature", defs.SAMPLING_TEMPERATURE), []).append(request)
            for temperature, requests_with_this_temperature in requests_by_temperature.items():
                if temperature == defs.SAMPLING_TEMPERATURE:
                    self._run_requests_with_one_cap(requests_with_this_temperature, maximum_new_tokens, on_finished, on_heartbeat)
                else:
                    self._run_requests_with_one_cap(requests_with_this_temperature, maximum_new_tokens, on_finished, on_heartbeat,
                                                    temperature=temperature)

    def _run_requests_with_one_cap(self, requests, maximum_new_tokens, on_finished, on_heartbeat, temperature=None):
        torch = self.torch
        padding_id = self.end_of_sequence_ids[0]
        ordered = sorted(requests, key=lambda request: len(request["prompt_token_ids"]))
        for start in range(0, len(ordered), self.prompts_per_batch):
            batch = ordered[start:start + self.prompts_per_batch]
            longest = max(len(request["prompt_token_ids"]) for request in batch)
            input_ids = torch.full((len(batch), longest), padding_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
            for row, request in enumerate(batch):
                ids = request["prompt_token_ids"]
                input_ids[row, longest - len(ids):] = torch.tensor(ids)
                attention_mask[row, longest - len(ids):] = 1
            if temperature in (None, defs.SAMPLING_TEMPERATURE):
                sampling_arguments = dict(do_sample=True, temperature=defs.SAMPLING_TEMPERATURE, top_p=defs.SAMPLING_TOP_P, top_k=0)
            elif temperature == 0:  # greedy decoding (the judges)
                sampling_arguments = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
            else:
                sampling_arguments = dict(do_sample=True, temperature=temperature, top_p=defs.SAMPLING_TOP_P, top_k=0)
            with torch.no_grad(), self.selected_adapter_context():
                generated = self.model.generate(
                    input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), **sampling_arguments,
                    max_new_tokens=maximum_new_tokens, eos_token_id=self.end_of_sequence_ids, pad_token_id=padding_id)
            for row, request in enumerate(batch):
                new_tokens = generated[row, longest:].cpu().tolist()
                stop_positions = [position for position, token in enumerate(new_tokens) if token in self.end_of_sequence_ids]
                if stop_positions:
                    on_finished(request["request_id"], new_tokens[:stop_positions[0] + 1], "stop")
                else:
                    on_finished(request["request_id"], new_tokens, "length")
            on_heartbeat()


# ---------------------------------------------------------------------------------------------------
# Progress reports
# ---------------------------------------------------------------------------------------------------

class AnswerProgressTracker:
    def __init__(self, label, total_answers, log_path, report_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS,
                 clock=time.time):
        self.label, self.total_answers, self.log_path = label, total_answers, log_path
        self.report_interval_seconds, self.clock = report_interval_seconds, clock
        self.finished_answers = self.generated_tokens = self.answers_that_hit_the_cap = 0
        self.start_time = self.last_report_time = clock()

    def note_finished(self, number_of_tokens, hit_cap):
        self.finished_answers += 1
        self.generated_tokens += number_of_tokens
        self.answers_that_hit_the_cap += int(hit_cap)
        self.report_if_due()

    def report_if_due(self):
        if self.clock() - self.last_report_time >= self.report_interval_seconds:
            self.report(final=False)

    def report(self, final):
        now = self.clock()
        self.last_report_time = now
        elapsed = max(now - self.start_time, 1e-9)
        rate = self.finished_answers / elapsed
        remaining = (self.total_answers - self.finished_answers) / rate if rate > 0 else None
        say(f"{'FINAL ' if final else ''}{self.label}: {self.finished_answers}/{self.total_answers} answers finished, "
            f"{self.total_answers - self.finished_answers} in flight or waiting; elapsed {format_duration(elapsed)}; "
            f"{self.generated_tokens} tokens in finished answers ({self.generated_tokens / elapsed:.0f} tokens/second); "
            f"{self.answers_that_hit_the_cap} hit the output cap; "
            f"estimated remaining {format_duration(remaining) if remaining is not None else 'unknown'} "
            f"(a lower bound while long answers are still running)", self.log_path)


# ---------------------------------------------------------------------------------------------------
# Generating one or both conditions for one model in one engine pass
# ---------------------------------------------------------------------------------------------------

def build_condition_plan(model, condition, prompts, samples_per_prompt, output_cap, answers_path, condition_spec=None):
    """Everything one condition needs: its chat-template switches, output cap, answers file and the answers still to do
    (each condition's finished answer ids are read from its own file)."""
    if condition_spec is None:
        condition_spec = model["thinking_on" if condition == defs.CONDITION_THINKING_ON else "thinking_off"]
    finished = read_finished_answer_ids(answers_path)
    pending = remaining_answers(model["model_short_name"], condition, prompts, samples_per_prompt, finished)
    return {"condition": condition, "condition_spec": condition_spec, "output_cap": output_cap,
            "answers_path": answers_path, "pending_answers": pending, "already_saved": len(finished)}


def generate_conditions_together(model, condition_plans, backend, tokenizer_adapter, end_of_sequence_ids, progress_log_path,
                                 report_interval_seconds, record_sink=None, lora_adapter=None):
    """Generates every pending (prompt, sample_index) of every condition plan in one backend pass, saving each answer's
    record to its own condition's answers file when it finishes. Requests are submitted thinking_on first, then thinking_off
    (the longest jobs first, so their long tail starts early; a condition counts as thinking on when its spec's split_as_condition says so). `record_sink` (check mode) receives (record, raw text)
    instead of the answers files. Returns the number of answers finished per condition."""
    plans = sorted(condition_plans, key=lambda plan: plan["condition_spec"].get("split_as_condition", plan["condition"])
                   != defs.CONDITION_THINKING_ON)
    lookup, requests, trackers = {}, [], {}
    for plan in plans:
        condition, output_cap = plan["condition"], plan["output_cap"]
        if not plan["pending_answers"]:
            say(f"{model['model_short_name']} {condition}: nothing left to generate", progress_log_path)
            continue
        for prompt, sample_index in plan["pending_answers"]:
            answer_id = defs.build_answer_id(model["model_short_name"], condition, prompt["prompt_gid"], sample_index)
            lookup[answer_id] = (plan, prompt, sample_index)
            requests.append({"request_id": answer_id, "maximum_new_tokens": output_cap,
                             "prompt_token_ids": tokenizer_adapter.prompt_token_ids(prompt["user_message"], plan["condition_spec"])})
        trackers[condition] = AnswerProgressTracker(
            f"{model['model_short_name']} {condition} (cap {output_cap})", len(plan["pending_answers"]),
            progress_log_path, report_interval_seconds)
        say(f"{model['model_short_name']} {condition}: generating {len(plan['pending_answers'])} answers, cap {output_cap} "
            f"new tokens", progress_log_path)
    if not requests:
        return {plan["condition"]: 0 for plan in plans}

    def on_finished(request_id, token_ids, finish_reason):
        plan, prompt, sample_index = lookup[request_id]
        record = build_answer_record(model, plan["condition"], plan["condition_spec"], prompt, sample_index, token_ids,
                                     finish_reason, plan["output_cap"], tokenizer_adapter.decode, end_of_sequence_ids,
                                     lora_adapter=lora_adapter)
        if record_sink is not None:
            record_sink(record, tokenizer_adapter.decode(token_ids))
        else:
            append_answer_record(plan["answers_path"], record)
        trackers[plan["condition"]].note_finished(len(token_ids), record["hit_output_cap"])

    def on_heartbeat():
        for tracker in trackers.values():
            tracker.report_if_due()

    backend.run_requests(requests, on_finished, on_heartbeat)
    for tracker in trackers.values():
        tracker.report(final=True)
    return {plan["condition"]: trackers[plan["condition"]].finished_answers if plan["condition"] in trackers else 0
            for plan in plans}


def check_one_condition(model, condition, condition_spec, output_cap, prompt, backend, tokenizer_adapter, end_of_sequence_ids,
                        progress_log_path, report_interval_seconds, label=None, lora_adapter=None):
    """One answer to one prompt; returns the check record (answer record fields plus the raw text and verdicts)."""
    captured = []
    plan = {"condition": condition, "condition_spec": condition_spec, "output_cap": output_cap, "answers_path": None,
            "pending_answers": [(prompt, 0)]}
    generate_conditions_together(model, [plan], backend, tokenizer_adapter, end_of_sequence_ids, progress_log_path,
                                 report_interval_seconds,
                                 record_sink=lambda record, raw_text: captured.append((record, raw_text)), lora_adapter=lora_adapter)
    record, raw_text = captured[0]
    is_true_off = condition_spec.get("is_true_thinking_off", True)
    condition_role = condition_spec.get("split_as_condition", condition)  # thinking_off or thinking_on behaviour
    if condition_role == defs.CONDITION_THINKING_OFF and is_true_off:
        switch_worked = record["thinking_trace_text"] is None and record["thinking_split_succeeded"]
    elif condition_role == defs.CONDITION_THINKING_OFF:
        switch_worked = record["thinking_split_succeeded"]  # reasoning strength low: any reasoning channel is allowed
    else:
        switch_worked = bool(record["thinking_trace_text"]) and record["thinking_split_succeeded"]
    prompt_ids = tokenizer_adapter.prompt_token_ids(prompt["user_message"], condition_spec)
    requirement_failures = rendered_prompt_requirement_failures(tokenizer_adapter.decode(prompt_ids), condition_spec)
    return {"check_label": label or condition, "condition": condition, "condition_role": condition_role,
            "rendered_prompt_requirement_failures": requirement_failures,
            "has_system_prompt": bool(condition_spec.get("system_prompt_text")),
            "chat_template_kwargs": condition_spec.get("chat_template_kwargs"),
            "generation_prompt_suffix": condition_spec.get("generation_prompt_suffix", ""),
            "prompt_tail_as_rendered": tokenizer_adapter.decode(prompt_ids[-40:]),
            "is_true_thinking_off": is_true_off,
            "reasoning_channel_present": bool(record["thinking_trace_text"]),
            "reasoning_channel_tokens": tokenizer_adapter.count_tokens(record["thinking_trace_text"]) if record["thinking_trace_text"] else 0,
            "thinking_switch_worked": bool(switch_worked), "thinking_split_succeeded": record["thinking_split_succeeded"],
            "hit_output_cap": record["hit_output_cap"], "number_of_generated_tokens": record["number_of_generated_tokens"],
            "inconclusive_because_output_cap_was_hit": record["hit_output_cap"],
            "raw_generated_text_with_special_tokens": raw_text, "answer_record": record}


# ---------------------------------------------------------------------------------------------------
# Hugging Face cache handling and background downloads
# ---------------------------------------------------------------------------------------------------

def huggingface_hub_cache_folder():
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def model_cache_folders(model_repository, hub_cache_folder):
    folder_name = "models--" + model_repository.replace("/", "--")
    return [Path(hub_cache_folder) / folder_name, Path(hub_cache_folder) / ".locks" / folder_name]


def folder_size_bytes(folder):
    total = 0
    for current, _, file_names in os.walk(folder):
        for file_name in file_names:
            try:
                total += os.lstat(os.path.join(current, file_name)).st_size
            except OSError:
                pass
    return total


def delete_model_cache(model_repository, hub_cache_folder):
    freed = 0
    for folder in model_cache_folders(model_repository, hub_cache_folder):
        if folder.exists():
            freed += folder_size_bytes(folder)
            shutil.rmtree(folder, ignore_errors=True)
    say(f"Deleted the Hugging Face cache of {model_repository} ({freed / 1e9:.1f} GB freed)")
    return freed


class BackgroundModelDownloader:
    """Downloads one model's files at its pinned commit in a thread and reports progress about every 3 minutes."""

    def __init__(self, hub_cache_folder, offline=False, report_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS,
                 download_function=None):
        self.hub_cache_folder, self.offline = hub_cache_folder, offline
        self.report_interval_seconds = report_interval_seconds
        self.download_function = download_function or self._snapshot_download
        self.threads, self.errors, self.finished_after_seconds = {}, {}, {}

    def _snapshot_download(self, model):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model["model_repository"], revision=model["model_revision"],
                          local_files_only=self.offline)

    def start(self, model):
        repository = model["model_repository"]
        if repository in self.threads:
            return
        thread = threading.Thread(target=self._run, args=(model,), daemon=True)
        self.threads[repository] = thread
        thread.start()

    def _run(self, model):
        repository, start = model["model_repository"], time.time()
        say(f"Download of {repository} (commit {model['model_revision'][:12]}) started")
        stop_reporting = threading.Event()

        def report_size_periodically():
            while not stop_reporting.wait(self.report_interval_seconds):
                size = sum(folder_size_bytes(folder) for folder in model_cache_folders(repository, self.hub_cache_folder))
                say(f"Download of {repository}: {size / 1e9:.1f} GB so far, elapsed {format_duration(time.time() - start)}"
                    f" (about {model.get('approximate_bf16_weights_gigabytes', '?')} GB expected)")

        reporter = threading.Thread(target=report_size_periodically, daemon=True)
        reporter.start()
        try:
            self.download_function(model)
        except BaseException as error:  # reported to whoever waits for this model
            self.errors[repository] = error
        finally:
            stop_reporting.set()
            self.finished_after_seconds[repository] = time.time() - start
            if repository not in self.errors:
                say(f"Download of {repository} finished in {format_duration(time.time() - start)}")

    def wait_for(self, model):
        self.start(model)
        wait_start = time.time()
        self.threads[model["model_repository"]].join()
        if model["model_repository"] in self.errors:
            raise RuntimeError(f"Download of {model['model_repository']} failed: {self.errors[model['model_repository']]!r}")
        waited = time.time() - wait_start
        say(f"{model['model_repository']} is on disk; generation waited {format_duration(waited)} for it")


# ---------------------------------------------------------------------------------------------------
# Worker: one model, both conditions together (or the one-answer check), in its own process
# ---------------------------------------------------------------------------------------------------

def read_end_of_sequence_ids(model, tokenizer):
    ids = set(model.get("end_of_sequence_token_ids") or [])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


def record_environment(run_folder, model, backend_name, device):
    versions = {"python": sys.version.split()[0]}
    for module_name in ("vllm", "transformers", "torch", "huggingface_hub"):
        try:
            module = __import__(module_name)
            versions[module_name] = module.__version__
        except Exception as error:  # a missing optional module is itself information
            versions[module_name] = f"not importable: {error!r}"
    try:
        import torch
        versions["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        versions["cuda_device"] = None
    path = Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / f"environment_versions_{model['model_short_name']}_{timestamp_now_with_seconds()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model["model_short_name"], "generation_backend": backend_name, "device": device,
                                "versions": versions}, indent=2), encoding="utf-8")


def build_backend(arguments, configuration, model, end_of_sequence_ids, lora_adapters=()):
    """One backend for the model's repository; `lora_adapters` are the adapters of every entry that will use it."""
    if arguments.generation_backend == "vllm":
        return VllmGenerationBackend(model["model_repository"], model["model_revision"], configuration["maximum_model_length"],
                                     configuration["gpu_memory_utilization"], lora_adapters=lora_adapters)
    return HuggingFaceGenerationBackend(model["model_repository"], model["model_revision"], arguments.device, end_of_sequence_ids)


def check_prompt_lengths(prompts, condition_entries, tokenizer_adapter, maximum_prompt_tokens):
    longest = 0
    for prompt in prompts:
        for entry in condition_entries:
            longest = max(longest, len(tokenizer_adapter.prompt_token_ids(prompt["user_message"], entry["condition_spec"])))
    if longest > maximum_prompt_tokens:
        raise ValueError(f"A prompt of {longest} tokens exceeds the allowed {maximum_prompt_tokens} (maximum_model_length "
                         f"was sized for it)")
    return longest


def run_worker(arguments):
    """One engine for the entries named in --worker-model-short-names (all of one repository), run in the given order."""
    from transformers import AutoTokenizer
    run_folder = Path(arguments.run_folder)
    configuration = load_run_configuration(run_folder)
    models_by_name = {entry["model_short_name"]: entry for entry in configuration["models"]}
    models = [models_by_name[name] for name in arguments.worker_model_short_names]
    if len({model["model_repository"] for model in models}) > 1:
        raise SystemExit(f"A worker serves one model repository, but {arguments.worker_model_short_names} span several")
    prompts = load_prompts(run_folder, configuration, arguments.maximum_prompts)
    samples_per_prompt = arguments.samples_per_prompt or configuration["samples_per_prompt"]
    output_cap_overrides = {condition: value for condition, value in (
        (defs.CONDITION_THINKING_OFF, arguments.output_cap_thinking_off),
        (defs.CONDITION_THINKING_ON, arguments.output_cap_thinking_on)) if value}
    if not arguments.one_answer_check:
        models = [model for model in models if number_of_remaining_answers_for_model(
            run_folder, model, prompts, samples_per_prompt, condition_names_for_model(configuration, model)) > 0]
    if not models:
        say("Every entry of this worker is complete; no engine started")
        return 0
    first = models[0]
    outputs_folder = run_folder / defs.OUTPUTS_FOLDER_NAME
    outputs_folder.mkdir(parents=True, exist_ok=True)
    progress_log_path = outputs_folder / f"generation_progress_log_{first['model_short_name']}_{timestamp_now_with_seconds()}.txt"
    tokenizer = AutoTokenizer.from_pretrained(first["model_repository"], revision=first["model_revision"])
    tokenizer_adapter = ChatTokenizerAdapter(tokenizer)
    end_of_sequence_ids = read_end_of_sequence_ids(first, tokenizer)
    condition_entries_by_name, lora_adapters_by_name = {}, {}
    for model in models:
        name = model["model_short_name"]
        condition_entries_by_name[name] = condition_entries_for_model(configuration, model, output_cap_overrides)
        longest_prompt = check_prompt_lengths(prompts, condition_entries_by_name[name], tokenizer_adapter,
                                              configuration["maximum_prompt_tokens_allowed"])
        say(f"{name}: {len(prompts)} prompts, longest rendered prompt {longest_prompt} tokens, "
            f"end-of-sequence ids {end_of_sequence_ids}", progress_log_path)
        record_environment(run_folder, model, arguments.generation_backend, arguments.device)
        lora_adapters_by_name[name] = resolve_lora_adapter(model, run_folder)
        if lora_adapters_by_name[name]:
            say(f"{name}: LoRA adapter {lora_adapters_by_name[name]['path_inside_run_folder']} (rank {lora_adapters_by_name[name]['rank']}, "
                f"weights SHA-256 {lora_adapters_by_name[name]['weights_sha256']})", progress_log_path)
    backend = build_backend(arguments, configuration, first, end_of_sequence_ids, list(lora_adapters_by_name.values()))
    for model in models:
        name = model["model_short_name"]
        backend.use_lora_adapter(lora_adapters_by_name[name])
        run_entry(arguments, model, condition_entries_by_name[name], lora_adapters_by_name[name], backend, tokenizer_adapter,
                  end_of_sequence_ids, prompts, samples_per_prompt, outputs_folder, run_folder, progress_log_path)
    return 0


def run_entry(arguments, model, condition_entries, lora_adapter, backend, tokenizer_adapter, end_of_sequence_ids, prompts,
              samples_per_prompt, outputs_folder, run_folder, progress_log_path):
    """The one-answer check (standalone or embedded) and then the full generation of one model entry on a backend that already
    has the entry's adapter selected."""
    caps = {entry["condition"]: entry["output_cap"] for entry in condition_entries}
    entry_start = time.time()
    if arguments.one_answer_check:  # standalone, informational: the files hold the verdicts
        run_one_answer_check_for_model(arguments, model, prompts[0], caps, backend, tokenizer_adapter, end_of_sequence_ids,
                                       outputs_folder, progress_log_path, condition_entries, lora_adapter)
        return
    if not arguments.skip_embedded_one_answer_check:
        if passing_check_file_exists(outputs_folder, model, lora_adapter):
            say(f"{model['model_short_name']}: a passing one-answer check for this revision is already saved; not repeating it",
                progress_log_path)
        else:
            result = run_one_answer_check_for_model(arguments, model, prompts[0], caps, backend, tokenizer_adapter,
                                                    end_of_sequence_ids, outputs_folder, progress_log_path, condition_entries,
                                                    lora_adapter)
            passed, failure_reasons = evaluate_one_answer_check(result)
            if not passed:
                message = (f"ONE-ANSWER CHECK FAILED for {model['model_short_name']}: {'; '.join(failure_reasons)}. "
                           f"Nothing was written to the answer files. Details: {defs.OUTPUTS_FOLDER_NAME}/"
                           f"{defs.ONE_ANSWER_CHECK_FOLDER_NAME}/")
                say(message, progress_log_path)
                raise SystemExit(message)
            say(f"{model['model_short_name']}: one-answer check passed; continuing with the full generation", progress_log_path)
    condition_plans = []
    for entry in condition_entries:
        condition = entry["condition"]
        plan = build_condition_plan(model, condition, prompts, samples_per_prompt, caps[condition],
                                    answer_file_path(run_folder, model["model_short_name"], condition),
                                    entry["condition_spec"])
        say(f"{model['model_short_name']} {condition}: {plan['already_saved']} answers already saved, "
            f"{len(plan['pending_answers'])} to generate", progress_log_path)
        condition_plans.append(plan)
    generate_conditions_together(model, condition_plans, backend, tokenizer_adapter, end_of_sequence_ids, progress_log_path,
                                 arguments.report_interval_seconds, lora_adapter=lora_adapter)
    say(f"{model['model_short_name']}: all {len(condition_entries)} conditions done in {format_duration(time.time() - entry_start)}",
        progress_log_path)


def run_one_answer_check_for_model(arguments, model, prompt, caps, backend, tokenizer_adapter, end_of_sequence_ids,
                                   outputs_folder, progress_log_path, condition_entries=None, lora_adapter=None):
    """One answer to the first prompt per condition. condition_entries (see condition_entries_for_model) default to the
    3 November screen's thinking_off and thinking_on."""
    if condition_entries is None:
        condition_entries = [{"condition": condition, "output_cap": caps[condition],
                              "condition_spec": model["thinking_on" if condition == defs.CONDITION_THINKING_ON else "thinking_off"]}
                             for condition in defs.CONDITIONS_IN_ORDER]
    check_caps = {entry["condition"]: min(entry["output_cap"], DEFAULT_CHECK_OUTPUT_CAPS.get(
        entry["condition_spec"].get("split_as_condition", entry["condition"]), 4096)) for entry in condition_entries}
    checks = [check_one_condition(model, entry["condition"], entry["condition_spec"], check_caps[entry["condition"]],
                                  prompt, backend, tokenizer_adapter, end_of_sequence_ids, progress_log_path,
                                  arguments.report_interval_seconds, lora_adapter=lora_adapter) for entry in condition_entries]
    result = {"model_short_name": model["model_short_name"], "model_repository": model["model_repository"],
              "model_revision": model["model_revision"], "prompt_gid": prompt["prompt_gid"],
              **({"lora_adapter_path_inside_run_folder": lora_adapter["path_inside_run_folder"],
                  "lora_adapter_weights_sha256": lora_adapter["weights_sha256"]} if lora_adapter else {}),
              "checked_at": datetime.now().isoformat(timespec="seconds"), "check_output_caps": check_caps,
              "expected_conditions": [entry["condition"] for entry in condition_entries], "checks": checks}
    folder = outputs_folder / defs.ONE_ANSWER_CHECK_FOLDER_NAME
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"one_answer_check_{model['model_short_name']}_{timestamp_now_with_seconds()}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    say(f"One-answer check written to {path.name}", progress_log_path)
    return result


def evaluate_one_answer_check(result):
    """(passed, reasons for failing). Required: thinking_off has no thinking content (Muse's thinking_off, reasoning strength
    low, may have a reasoning channel: it only has to split cleanly), thinking_on has a non-empty trace, and both split
    cleanly without hitting the cap."""
    reasons = []
    for check in result["checks"]:
        label = check["condition"]
        for failure in check.get("rendered_prompt_requirement_failures", []):
            reasons.append(f"{label}: {failure}")
        if check["hit_output_cap"]:
            reasons.append(f"{label}: the check answer hit its output cap, so it is inconclusive")
        if not check["thinking_split_succeeded"]:
            reasons.append(f"{label}: the thinking trace did not split cleanly from the answer")
        if not check["thinking_switch_worked"]:
            reasons.append(f"{label}: the thinking switch did not work ("
                           + ("thinking content appeared" if check.get("condition_role", label) == defs.CONDITION_THINKING_OFF else "no thinking trace appeared") + ")")
    expected = set(result.get("expected_conditions") or [defs.CONDITION_THINKING_OFF, defs.CONDITION_THINKING_ON])
    if {check["condition"] for check in result["checks"]} != expected:
        reasons.append("a condition was not checked")
    return not reasons, reasons


def passing_check_file_exists(outputs_folder, model, lora_adapter=None):
    folder = Path(outputs_folder) / defs.ONE_ANSWER_CHECK_FOLDER_NAME
    for path in sorted(folder.glob(f"one_answer_check_{model['model_short_name']}_*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        same_adapter = result.get("lora_adapter_weights_sha256") == (lora_adapter["weights_sha256"] if lora_adapter else None)
        if result.get("model_revision") == model["model_revision"] and same_adapter and evaluate_one_answer_check(result)[0]:
            return True
    return False


def summarise_one_answer_checks(run_folder, model_short_names, started_after_epoch_seconds):
    """Prints one line per check from the check files written in this session, and returns the summary."""
    folder = Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / defs.ONE_ANSWER_CHECK_FOLDER_NAME
    summary = []
    for model_short_name in model_short_names:
        files = [path for path in sorted(folder.glob(f"one_answer_check_{model_short_name}_*.json"))
                 if path.stat().st_mtime >= started_after_epoch_seconds]
        if not files:
            summary.append({"model_short_name": model_short_name, "result": "no check file written"})
            continue
        result = json.loads(files[-1].read_text(encoding="utf-8"))
        for check in result["checks"]:
            line = {"model_short_name": model_short_name, "check_label": check["check_label"],
                    "thinking_switch_worked": check["thinking_switch_worked"],
                    "prompt_tail_as_rendered": check.get("prompt_tail_as_rendered"),
                    "rendered_prompt_requirement_failures": check.get("rendered_prompt_requirement_failures"),
                    "reasoning_channel_present": check.get("reasoning_channel_present"),
                    "reasoning_channel_tokens": check.get("reasoning_channel_tokens"),
                    "thinking_split_succeeded": check["thinking_split_succeeded"],
                    "hit_output_cap": check["hit_output_cap"], "tokens": check["number_of_generated_tokens"]}
            summary.append(line)
            say("ONE-ANSWER CHECK " + json.dumps(line))
    summary_path = Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / defs.ONE_ANSWER_CHECK_FOLDER_NAME / f"one_answer_check_summary_{timestamp_now_with_seconds()}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------------------------------

def worker_command_line(arguments, model_short_names):
    command = [sys.executable, str(Path(__file__).resolve()), "--run-folder", arguments.run_folder,
               "--worker-model-short-names", *model_short_names, "--generation-backend", arguments.generation_backend,
               "--device", arguments.device, "--report-interval-seconds", str(arguments.report_interval_seconds)]
    for flag, value in (("--maximum-prompts", arguments.maximum_prompts), ("--samples-per-prompt", arguments.samples_per_prompt),
                        ("--output-cap-thinking-off", arguments.output_cap_thinking_off),
                        ("--output-cap-thinking-on", arguments.output_cap_thinking_on)):
        if value:
            command += [flag, str(value)]
    if arguments.one_answer_check:
        command.append("--one-answer-check")
    if arguments.skip_embedded_one_answer_check:
        command.append("--skip-embedded-one-answer-check")
    return command


def models_to_run(configuration, only_model_short_names):
    models = {model["model_short_name"]: model for model in configuration["models"]}
    names = [name for name in configuration["model_order"] if not only_model_short_names or name in only_model_short_names]
    unknown = set(only_model_short_names or []) - set(models)
    if unknown:
        raise SystemExit(f"Unknown model short names {sorted(unknown)}; choose from {sorted(models)}")
    return [models[name] for name in names]


def run_orchestrator(arguments):
    run_start = time.time()
    run_folder = Path(arguments.run_folder)
    configuration = load_run_configuration(run_folder)
    prompts = load_prompts(run_folder, configuration, arguments.maximum_prompts)
    samples_per_prompt = arguments.samples_per_prompt or configuration["samples_per_prompt"]
    (run_folder / defs.OUTPUTS_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    hub_cache = huggingface_hub_cache_folder()
    selected = models_to_run(configuration, arguments.only_model_short_names)
    if arguments.one_answer_check:
        to_run = selected
    else:
        to_run = [model for model in selected if number_of_remaining_answers_for_model(
            run_folder, model, prompts, samples_per_prompt, condition_names_for_model(configuration, model)) > 0]
    say(f"Models to run: {[model['model_short_name'] for model in to_run]} "
        f"({'one-answer check' if arguments.one_answer_check else 'full generation'}; "
        f"{len(selected) - len(to_run)} already complete); Hugging Face cache {hub_cache}; "
        f"free disk {shutil.disk_usage(hub_cache.parent if hub_cache.parent.exists() else '/').free / 1e9:.0f} GB")
    downloader = BackgroundModelDownloader(hub_cache, offline=arguments.offline,
                                           report_interval_seconds=arguments.report_interval_seconds)
    groups = group_models_by_repository(to_run)
    for position, group in enumerate(groups):
        group_start = time.time()
        names = [model["model_short_name"] for model in group]
        downloader.wait_for(group[0])
        if position + 1 < len(groups):
            downloader.start(groups[position + 1][0])  # overlaps with this repository's loading and generation
        say(f"Starting {group[0]['model_repository']} with {names} ({position + 1} of {len(groups)}); "
            f"run so far {format_duration(time.time() - run_start)}")
        completed = subprocess.run(worker_command_line(arguments, names))
        if completed.returncode != 0:
            raise SystemExit(f"The worker for {names} exited with code {completed.returncode}; the repository's "
                             f"cache is kept so that a relaunch does not download it again")
        say(f"{names} finished in {format_duration(time.time() - group_start)}")
        if not arguments.keep_model_caches:
            delete_model_cache(group[0]["model_repository"], hub_cache)
    if arguments.one_answer_check:
        summarise_one_answer_checks(run_folder, [model["model_short_name"] for model in to_run], run_start - 5)
    counts = {f"{model['model_short_name']}__{condition}": len(read_finished_answer_ids(
        answer_file_path(run_folder, model["model_short_name"], condition)))
        for model in selected for condition in condition_names_for_model(configuration, model)}
    say(f"Done in {format_duration(time.time() - run_start)}. Saved answers per file: {json.dumps(counts)}")
    return 0


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--one-answer-check", action="store_true",
                        help="one answer to the first prompt per model per condition; writes only the check files")
    parser.add_argument("--skip-embedded-one-answer-check", action="store_true",
                        help="do not run the one-answer check at the start of each model's generation")
    parser.add_argument("--only-model-short-names", nargs="+", default=None)
    parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default="vllm")
    parser.add_argument("--device", default="cuda", help="huggingface backend only")
    parser.add_argument("--keep-model-caches", action="store_true", help="do not delete a model's cache after its answers are done")
    parser.add_argument("--offline", action="store_true", help="use only the local Hugging Face cache (local rehearsal)")
    parser.add_argument("--report-interval-seconds", type=float, default=PROGRESS_REPORT_INTERVAL_SECONDS)
    parser.add_argument("--maximum-prompts", type=int, default=None, help="rehearsal: use only the first prompts")
    parser.add_argument("--samples-per-prompt", type=int, default=None, help="rehearsal: override the configuration")
    parser.add_argument("--output-cap-thinking-off", type=int, default=None, help="rehearsal: override the configuration")
    parser.add_argument("--output-cap-thinking-on", type=int, default=None, help="rehearsal: override the configuration")
    parser.add_argument("--worker-model-short-names", nargs="+", default=None,
                        help="internal: run these entries (one model repository) in this process, in the given order")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    if arguments.worker_model_short_names:
        return run_worker(arguments)
    return run_orchestrator(arguments)


if __name__ == "__main__":
    sys.exit(main())
