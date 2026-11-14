"""Runs on the GPU pod (or on CPU with a tiny model for the local rehearsal): LoRA or 4-bit QLoRA fine-tuning of a candidate model
on its own answers, with loss on the assistant tokens only.

    python3 scripts/emergent_misalignment/fine_tune_candidate_model_with_lora_on_own_answers_on_pod.py --run-folder <run folder>

Everything comes from <run folder>/run_configuration_*.json (built by build_lora_fine_tuning_run_folder_for_self_distillation.py).
Steps: (1) load the tokenizer and check its chat template against the saved fixture; (2) render and tokenize every example
(thinking off, no system prompt, see render_self_distillation_training_examples_with_assistant_only_loss_mask.py), check the
rendered prompt and the loss mask on the first example, refuse examples longer than the maximum sequence length; (3) load the model
(4-bit NF4 or bfloat16) and attach LoRA to the language model's attention and feed-forward projections; (4) train, with a
checkpoint every N optimizer steps and at each epoch end; (5) save the final adapter and a training summary.

Outputs in <run folder>/training_outputs/: training_progress_log_*.txt (a line about every 3 minutes: step, loss, tokens a second,
GPU memory, time left), training_step_metrics.jsonl (one line per optimizer step), checkpoints/checkpoint_step_*/ (adapter,
optimizer and schedule state, completion marker written last; the latest two are kept), adapter_at_end_of_epoch/epoch_N/,
final_adapter/, training_summary_*.json, chat_template_and_masking_check_*.json.

Resumable: a relaunch continues from the latest complete checkpoint (the data order of each epoch depends only on the seed and the
epoch, so the example order is reproduced; dropout randomness is not). A finished run (final adapter and summary present) exits at once.
"""

import argparse
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import check_saved_lora_adapter_keys_against_base_text_model_module_names as adapter_check  # noqa: E402
from emergent_misalignment import fine_tuning_run_definitions as defs  # noqa: E402
from emergent_misalignment import render_self_distillation_training_examples_with_assistant_only_loss_mask as rendering  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import sha256_of_text, timestamp_now_with_seconds  # noqa: E402
from shared.timed_progress_reporter import PROGRESS_REPORT_INTERVAL_SECONDS, format_duration  # noqa: E402


def say(message, log_path=None):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_run_configuration(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_configuration file in {run_folder}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------
# Schedule and data order (pure functions, tested without a model)
# ---------------------------------------------------------------------------------------------------

def steps_per_epoch(number_of_examples, effective_batch_size):
    return math.ceil(number_of_examples / effective_batch_size)


def example_indices_for_optimizer_step(optimizer_step, number_of_examples, effective_batch_size, seed):
    """Which examples optimizer step number `optimizer_step` (0-based) trains on, and which epoch it is in. The last step of
    an epoch may hold fewer examples. The order within an epoch depends only on (seed, epoch)."""
    per_epoch = steps_per_epoch(number_of_examples, effective_batch_size)
    epoch, step_in_epoch = divmod(optimizer_step, per_epoch)
    order = list(range(number_of_examples))
    random.Random(seed * 1000 + epoch).shuffle(order)
    start = step_in_epoch * effective_batch_size
    return epoch, order[start:start + effective_batch_size]


def learning_rate_multiplier(optimizer_step, warmup_steps, total_steps):
    """Linear warmup over warmup_steps, then linear decay to zero at total_steps."""
    if optimizer_step < warmup_steps:
        return (optimizer_step + 1) / max(1, warmup_steps)
    return max(0.0, (total_steps - optimizer_step) / max(1, total_steps - warmup_steps))


# ---------------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------------

def checkpoints_folder(outputs_folder):
    return Path(outputs_folder) / defs.CHECKPOINTS_FOLDER_NAME


def list_complete_checkpoints(outputs_folder):
    """[(optimizer_step, folder)] in step order; a checkpoint without its completion marker (killed while saving) is ignored."""
    found = []
    root = checkpoints_folder(outputs_folder)
    if root.exists():
        for folder in root.glob("checkpoint_step_*"):
            if (folder / defs.CHECKPOINT_COMPLETE_MARKER_FILE_NAME).exists():
                found.append((int(folder.name.rsplit("_", 1)[1]), folder))
    return sorted(found)


def prune_old_checkpoints(outputs_folder, number_to_keep=defs.NUMBER_OF_RESUME_CHECKPOINTS_KEPT):
    for _, folder in list_complete_checkpoints(outputs_folder)[:-number_to_keep]:
        shutil.rmtree(folder, ignore_errors=True)
    root = checkpoints_folder(outputs_folder)
    if root.exists():  # incomplete leftovers from a kill during saving
        for folder in root.glob("checkpoint_step_*"):
            if not (folder / defs.CHECKPOINT_COMPLETE_MARKER_FILE_NAME).exists():
                shutil.rmtree(folder, ignore_errors=True)


def save_checkpoint(torch, model, optimizer, scheduler, outputs_folder, optimizer_step, tokens_seen, extra_state):
    folder = checkpoints_folder(outputs_folder) / f"checkpoint_step_{optimizer_step:06d}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    model.save_pretrained(folder / defs.ADAPTER_SUBFOLDER_NAME)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "optimizer_step": optimizer_step,
                "tokens_seen": tokens_seen, **extra_state}, folder / defs.TRAINING_STATE_FILE_NAME)
    (folder / defs.CHECKPOINT_COMPLETE_MARKER_FILE_NAME).write_text(
        json.dumps({"optimizer_step": optimizer_step, "saved_at": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    prune_old_checkpoints(outputs_folder)
    return folder


# ---------------------------------------------------------------------------------------------------
# Checks before training
# ---------------------------------------------------------------------------------------------------

def check_chat_template_against_fixture(tokenizer, run_folder, configuration):
    fixture = configuration.get("chat_template_fixture")
    if not fixture:
        return {"checked": False, "reason": "no fixture in the run configuration (rehearsal with a stand-in model)"}
    fixture_text = (Path(run_folder) / fixture["path_inside_run_folder"]).read_text(encoding="utf-8")
    served = tokenizer.chat_template
    if not isinstance(served, str):
        raise RuntimeError("The tokenizer has no single chat template string to compare with the fixture")
    served_sha256 = sha256_of_text(served)
    if served_sha256 != fixture["sha256"]:
        raise RuntimeError(f"The tokenizer's chat template (SHA-256 {served_sha256}) differs from the fixture "
                           f"{fixture['path_inside_run_folder']} ({fixture['sha256']}); the tests and the generation settings "
                           "describe the fixture, so stopping.")
    return {"checked": True, "tokenizer_chat_template_sha256": served_sha256, "fixture_sha256": fixture["sha256"]}


def check_rendered_prompt_and_loss_mask(tokenizer, encoded, template_family, end_of_turn_text):
    """Returns failure descriptions (empty when all hold) for one encoded example."""
    failures = []
    prompt_text = encoded["prompt_text"]
    tail = defs.REQUIRED_PROMPT_TAIL_BY_FAMILY.get(template_family)
    if tail and not prompt_text.endswith(tail):
        failures.append(f"prompt does not end with {tail!r} (ends {prompt_text[-60:]!r})")
    for required in defs.REQUIRED_PROMPT_SUBSTRINGS_BY_FAMILY.get(template_family, []):
        if required not in prompt_text:
            failures.append(f"prompt lacks {required!r}")
    for forbidden in defs.FORBIDDEN_PROMPT_SUBSTRINGS_BY_FAMILY.get(template_family, []):
        if forbidden in prompt_text:
            failures.append(f"prompt contains {forbidden!r}")
    labels, input_ids = encoded["labels"], encoded["input_ids"]
    supervised_positions = [position for position, label in enumerate(labels) if label != rendering.IGNORED_LABEL_ID]
    if supervised_positions != list(range(encoded["number_of_prompt_tokens"], len(input_ids))):
        failures.append("supervised tokens are not exactly the response tokens")
    if any(labels[position] != input_ids[position] for position in supervised_positions):
        failures.append("a label differs from its input token")
    decoded_supervised = tokenizer.decode([labels[position] for position in supervised_positions])
    if not decoded_supervised.endswith(end_of_turn_text):
        failures.append(f"supervised text does not end with {end_of_turn_text!r}: {decoded_supervised[-40:]!r}")
    return failures


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

def load_base_model(torch, transformers, configuration, device, compute_dtype):
    model_facts, hyper = configuration["model"], configuration["hyperparameters"]
    common = dict(revision=model_facts["model_revision"])
    if hyper["quantisation"] == defs.QUANTISATION_4BIT_QLORA:
        quantisation_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_skip_modules=hyper["modules_left_unquantised"])
        common.update(quantization_config=quantisation_config, device_map={"": 0}, dtype=compute_dtype)
    else:
        common.update(dtype=compute_dtype)
    model_class = (transformers.AutoModelForCausalLM if model_facts.get("model_class") == "auto_causal_lm"
                   else transformers.AutoModelForImageTextToText)  # Muse Glimmer loads only through the image-text class
    model = model_class.from_pretrained(model_facts["model_repository"], **common)
    if hyper["quantisation"] != defs.QUANTISATION_4BIT_QLORA:
        model.to(device)
    return model


def attach_or_resume_lora(peft, model, configuration, resume_adapter_folder):
    hyper = configuration["hyperparameters"]
    if resume_adapter_folder is not None:
        return peft.PeftModel.from_pretrained(model, str(resume_adapter_folder), is_trainable=True)
    lora_config = peft.LoraConfig(
        r=hyper["lora_rank"], lora_alpha=hyper["lora_alpha"], lora_dropout=hyper["lora_dropout"],
        use_rslora=hyper["use_rank_stabilised_lora"], target_modules=hyper["lora_target_module_regular_expression"],
        bias="none", task_type="CAUSAL_LM")
    return peft.get_peft_model(model, lora_config)


def extra_multimodal_inputs(base_model_forward_parameters, input_ids, torch):
    """Gemma-style conditional generation models may require all-zero token type ids in training (no images)."""
    extras = {}
    for name in ("token_type_ids", "mm_token_type_ids"):
        if name in base_model_forward_parameters:
            extras[name] = torch.zeros_like(input_ids)
    return extras


def gpu_memory_text(torch, device):
    if device != "cuda":
        return "no GPU"
    return (f"{torch.cuda.memory_allocated() / 2**30:.1f} GB allocated, "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak")


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    parser.add_argument("--compute-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--offline", action="store_true", help="use only the local Hugging Face cache (rehearsal)")
    parser.add_argument("--report-interval-seconds", type=float, default=PROGRESS_REPORT_INTERVAL_SECONDS)
    parser.add_argument("--only-check-template-and-masking", action="store_true",
                        help="run the checks (template, rendering, masking, lengths) and stop before loading the model")
    parser.add_argument("--pause-seconds-after-each-optimizer-step", type=float, default=0.0,
                        help="testing only: slows training so that a rehearsal can kill it between checkpoints")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    if arguments.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers
    run_folder = Path(arguments.run_folder)
    configuration = load_run_configuration(run_folder)
    hyper, model_facts = configuration["hyperparameters"], configuration["model"]
    outputs_folder = run_folder / defs.OUTPUTS_FOLDER_NAME
    outputs_folder.mkdir(exist_ok=True)
    summaries = sorted(outputs_folder.glob("training_summary_*.json"))
    if summaries and (outputs_folder / defs.FINAL_ADAPTER_FOLDER_NAME).exists():
        print(f"Already finished ({summaries[-1].name}); nothing to do.", flush=True)
        return 0
    existing_logs = sorted(outputs_folder.glob("training_progress_log_*.txt"))
    log_path = existing_logs[-1] if existing_logs else outputs_folder / defs.PROGRESS_LOG_FILE_NAME_PATTERN.format(
        timestamp=timestamp_now_with_seconds())
    device = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = getattr(torch, arguments.compute_dtype)
    say(f"Run {configuration['run_name']} on {device}; model {model_facts['model_repository']} at {model_facts['model_revision']}, "
        f"{hyper['quantisation']}", log_path)

    # 1 and 2: template, rendering, masking
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_facts["model_repository"], revision=model_facts["model_revision"])
    template_check = check_chat_template_against_fixture(tokenizer, run_folder, configuration)
    say(f"Chat template check: {template_check}", log_path)
    kwargs = configuration["thinking_off_chat_template_kwargs"]
    end_of_turn_text = model_facts["end_of_turn_text"]
    examples = read_jsonl(run_folder / configuration["training_examples_file"]["path_inside_run_folder"])
    encoded_examples = [rendering.encode_example(tokenizer, example["messages"], kwargs, end_of_turn_text) for example in examples]
    lengths = [len(encoded["input_ids"]) for encoded in encoded_examples]
    overlong = [index for index, length in enumerate(lengths) if length > hyper["maximum_sequence_length"]]
    failures = check_rendered_prompt_and_loss_mask(tokenizer, encoded_examples[0], model_facts["template_family"], end_of_turn_text)
    total_response_tokens = sum(encoded["number_of_response_tokens"] for encoded in encoded_examples)
    check_record = {
        "template_check": template_check, "first_example_failures": failures, "number_of_examples": len(examples),
        "number_over_maximum_sequence_length": len(overlong), "maximum_sequence_length": hyper["maximum_sequence_length"],
        "tokens_per_example_mean": sum(lengths) / len(lengths), "tokens_per_example_longest": max(lengths),
        "total_tokens_per_epoch": sum(lengths), "total_supervised_tokens_per_epoch": total_response_tokens,
        "first_example_prompt_tail": encoded_examples[0]["prompt_text"][-120:],
        "first_example_supervised_text": encoded_examples[0]["response_text"][:200] + (
            "..." if len(encoded_examples[0]["response_text"]) > 200 else "")}
    (outputs_folder / defs.TEMPLATE_CHECK_FILE_NAME_PATTERN.format(timestamp=timestamp_now_with_seconds())).write_text(
        json.dumps(check_record, indent=2), encoding="utf-8")
    say(f"{len(examples)} examples, {sum(lengths):,} tokens per epoch ({total_response_tokens:,} supervised), longest "
        f"{max(lengths)}, over the maximum {len(overlong)}", log_path)
    if failures:
        raise RuntimeError(f"Template or masking check failed: {failures}")
    if overlong:
        raise RuntimeError(f"{len(overlong)} examples exceed the maximum sequence length {hyper['maximum_sequence_length']} "
                           f"(first indices {overlong[:5]}); raise --maximum-sequence-length in a new run folder or remove them. "
                           "Truncating would cut off the end-of-turn token, so it is not done.")
    if arguments.only_check_template_and_masking:
        return 0

    # 3: model
    resume_checkpoints = list_complete_checkpoints(outputs_folder)
    resume_step, resume_folder = resume_checkpoints[-1] if resume_checkpoints else (0, None)
    torch.manual_seed(hyper["seed"])
    base_model = load_base_model(torch, transformers, configuration, device, compute_dtype)
    model = attach_or_resume_lora(__import__("peft"), base_model, configuration,
                                  resume_folder / defs.ADAPTER_SUBFOLDER_NAME if resume_folder else None)
    base_model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    say(f"Trainable parameters {sum(p.numel() for p in trainable):,}; GPU memory after loading: {gpu_memory_text(torch, device)}", log_path)
    forward_parameters = set(inspect.signature(model.get_base_model().forward).parameters)
    optimizer = torch.optim.AdamW(trainable, lr=hyper["learning_rate"], weight_decay=hyper["weight_decay"])
    effective_batch_size = hyper["per_device_micro_batch_size"] * hyper["gradient_accumulation_steps"]
    total_steps = steps_per_epoch(len(examples), effective_batch_size) * hyper["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_multiplier(step, hyper["warmup_steps"], total_steps))
    tokens_seen = 0
    if resume_folder is not None:
        state = torch.load(resume_folder / defs.TRAINING_STATE_FILE_NAME, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        tokens_seen = state["tokens_seen"]
        say(f"RESUMING from {resume_folder.name} (optimizer step {resume_step} of {total_steps})", log_path)
    else:
        say(f"Starting fresh: {total_steps} optimizer steps ({hyper['epochs']} epochs of "
            f"{steps_per_epoch(len(examples), effective_batch_size)})", log_path)

    padding_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    metrics_path = outputs_folder / defs.METRICS_LOG_FILE_NAME
    session_start, last_report = time.time(), time.time()
    session_tokens, session_steps, step_seconds_history = 0, 0, []
    per_epoch = steps_per_epoch(len(examples), effective_batch_size)
    recent_losses = []
    peak_memory_gigabytes = 0.0
    for optimizer_step in range(resume_step, total_steps):
        step_start = time.time()
        epoch, indices = example_indices_for_optimizer_step(optimizer_step, len(examples), effective_batch_size, hyper["seed"])
        step_examples = sorted((encoded_examples[index] for index in indices), key=lambda e: len(e["input_ids"]))
        supervised_tokens_in_step = sum(e["number_of_response_tokens"] for e in step_examples)
        micro = hyper["per_device_micro_batch_size"]
        step_loss, step_tokens = 0.0, 0
        for micro_start in range(0, len(step_examples), micro):
            group = step_examples[micro_start:micro_start + micro]
            input_ids, attention_mask, labels = (torch.tensor(rows, device=device) for rows in
                                                 rendering.pad_batch(group, padding_token_id))
            extras = extra_multimodal_inputs(forward_parameters, input_ids, torch)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(arguments.compute_dtype == "bfloat16")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **extras)
            group_supervised = sum(e["number_of_response_tokens"] for e in group)
            # the model's loss is the mean over this group's supervised tokens; weight it so the step's loss is the mean over all
            # supervised tokens in the step, whatever the group sizes
            weighted_loss = outputs.loss * (group_supervised / supervised_tokens_in_step)
            weighted_loss.backward()
            step_loss += weighted_loss.item()
            step_tokens += sum(len(e["input_ids"]) for e in group)
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, hyper["maximum_gradient_norm"]).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step_seconds = time.time() - step_start
        tokens_seen += step_tokens
        session_tokens += step_tokens
        session_steps += 1
        step_seconds_history.append((step_seconds, step_tokens))
        recent_losses = (recent_losses + [step_loss])[-10:]
        if device == "cuda":
            peak_memory_gigabytes = max(peak_memory_gigabytes, torch.cuda.max_memory_allocated() / 2**30)
        completed_steps = optimizer_step + 1
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "optimizer_step": completed_steps, "epoch": epoch + 1, "loss": step_loss, "gradient_norm": gradient_norm,
                "learning_rate": scheduler.get_last_lr()[0], "examples_in_step": len(step_examples),
                "tokens_in_step": step_tokens, "supervised_tokens_in_step": supervised_tokens_in_step,
                "step_seconds": step_seconds, "tokens_per_second": step_tokens / step_seconds,
                "gpu_peak_memory_gigabytes": peak_memory_gigabytes,
                "recorded_at": datetime.now().isoformat(timespec="seconds")}) + "\n")
        if time.time() - last_report >= arguments.report_interval_seconds or completed_steps == total_steps:
            last_report = time.time()
            elapsed = time.time() - session_start
            tokens_per_second = session_tokens / elapsed
            remaining = (total_steps - completed_steps) * (elapsed / session_steps)
            say(f"step {completed_steps}/{total_steps} (epoch {epoch + 1}); loss {step_loss:.4f} (mean of last "
                f"{len(recent_losses)} {sum(recent_losses) / len(recent_losses):.4f}); {tokens_per_second:,.0f} tokens/s this session; "
                f"GPU {gpu_memory_text(torch, device)}; elapsed {format_duration(elapsed)}; about {format_duration(remaining)} left",
                log_path)
        end_of_epoch = completed_steps % per_epoch == 0
        if end_of_epoch:
            epoch_adapter = outputs_folder / defs.EPOCH_ADAPTERS_FOLDER_NAME / f"epoch_{completed_steps // per_epoch}"
            model.save_pretrained(epoch_adapter)
        if end_of_epoch or completed_steps % hyper["checkpoint_every_optimizer_steps"] == 0:
            folder = save_checkpoint(torch, model, optimizer, scheduler, outputs_folder, completed_steps, tokens_seen, {})
            say(f"checkpoint saved: {folder.name}", log_path)
        if arguments.pause_seconds_after_each_optimizer_step:
            time.sleep(arguments.pause_seconds_after_each_optimizer_step)

    # 5: final
    final_adapter_folder = outputs_folder / defs.FINAL_ADAPTER_FOLDER_NAME
    model.save_pretrained(final_adapter_folder)
    base_module_names = {name[len("base_model.model."):] for name, _ in model.named_modules() if name.startswith("base_model.model.")}
    adapter_key_check = adapter_check.check_saved_adapter_folder(
        final_adapter_folder, base_module_names, hyper["lora_target_modules"], hyper.get("adapter_key_prefix", adapter_check.KEY_PREFIX))
    if adapter_key_check is not None:
        (outputs_folder / "adapter_key_check_for_vllm_lora_request.json").write_text(json.dumps(adapter_key_check, indent=2), encoding="utf-8")
        if adapter_key_check["problems"]:
            raise RuntimeError(f"The saved adapter would not load in vLLM for the text model: {adapter_key_check['problems'][:5]}")
        adapter_check.write_vllm_copy_of_adapter(final_adapter_folder, hyper["lora_target_modules"])
    session_seconds = time.time() - session_start
    steady = step_seconds_history[1:] or step_seconds_history  # the first step includes warm-up
    summary = {
        "run_name": configuration["run_name"], "finished_at": datetime.now().isoformat(timespec="seconds"),
        "device": device, "optimizer_steps_total": total_steps, "resumed_from_optimizer_step": resume_step,
        "optimizer_steps_this_session": session_steps, "session_seconds": session_seconds,
        "tokens_processed_total": tokens_seen, "tokens_processed_this_session": session_tokens,
        "tokens_per_second_this_session": session_tokens / session_seconds if session_seconds else None,
        "tokens_per_second_excluding_first_step": (sum(t for _, t in steady) / sum(s for s, _ in steady)) if steady else None,
        "final_loss_mean_of_last_ten_steps": sum(recent_losses) / len(recent_losses) if recent_losses else None,
        "gpu_peak_memory_gigabytes": peak_memory_gigabytes, "number_of_examples": len(examples),
        "total_tokens_per_epoch": check_record["total_tokens_per_epoch"],
        "hyperparameters": hyper, "model": model_facts, "template_check": template_check,
        "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "peft": __import__("peft").__version__}}
    if hyper["quantisation"] == defs.QUANTISATION_4BIT_QLORA:
        try:
            summary["library_versions"]["bitsandbytes"] = __import__("bitsandbytes").__version__
        except ImportError:
            pass
    (outputs_folder / defs.TRAINING_SUMMARY_FILE_NAME_PATTERN.format(timestamp=timestamp_now_with_seconds())).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    say(f"Finished: {total_steps} steps; final adapter saved; {summary['tokens_per_second_excluding_first_step']} tokens/s excluding "
        "the first step", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
