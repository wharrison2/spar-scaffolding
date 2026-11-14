"""Builds the run folder for one LoRA fine-tuning of a candidate model on its own answers (self-distillation). No network, no GPU.

    python3 scripts/emergent_misalignment/build_lora_fine_tuning_run_folder_for_self_distillation.py \
        --model-short-name gemma_4_31b_it --training-examples-file <jsonl> [--training-set-size 3000] [--epochs 1] \
        [--quantisation 4bit_qlora|bf16_lora] [--timing-run-examples 200]

Training examples: jsonl, one per line, {"messages": [user, assistant], "answer_id", "prompt_gid", "model_short_name"}, no system
prompt. Every example's model_short_name must equal --model-short-name (self-distillation); otherwise the builder refuses, unless
the loud override flag is given, which is then recorded in the run configuration.

Writes data/emergent_misalignment_runs/<lora_fine_tuning...>_<timestamp>/ holding run_configuration_<timestamp>.json (every
setting, the chat template fixture, file sha256s), training_inputs/ (the seeded subsample of the examples and the fixture copy)
and an empty training_outputs/. --timing-run-examples N builds a short run (N examples, one epoch, a different folder name
prefix, under the same runs root) to measure tokens per second before the full run.
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import baseline_screen_definitions as baseline_defs  # noqa: E402
from emergent_misalignment import fine_tuning_run_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import sha256_of_file, timestamp_now  # noqa: E402

SUPPORTED_MODEL_SHORT_NAMES = list(defs.PINNED_REVISIONS)


def parse_arguments(argv=None):
    hyper = defs.DEFAULT_HYPERPARAMETERS
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-short-name", required=True, choices=SUPPORTED_MODEL_SHORT_NAMES)
    parser.add_argument("--training-examples-file", required=True)
    parser.add_argument("--training-set-size", type=int, default=None, help="seeded subsample; default all examples")
    parser.add_argument("--timing-run-examples", type=int, nargs="?", const=defs.DEFAULT_TIMING_RUN_EXAMPLES, default=None,
                        help=f"build a short timing run on this many examples (default {defs.DEFAULT_TIMING_RUN_EXAMPLES}), one epoch")
    parser.add_argument("--epochs", type=int, default=hyper["epochs"])
    parser.add_argument("--learning-rate", type=float, default=hyper["learning_rate"])
    parser.add_argument("--lora-rank", type=int, choices=defs.VLLM_SUPPORTED_LORA_RANKS, default=hyper["lora_rank"])
    parser.add_argument("--lora-alpha", type=float, default=hyper["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=hyper["lora_dropout"])
    parser.add_argument("--no-rank-stabilised-lora", action="store_true")
    parser.add_argument("--lora-target-modules", nargs="+", default=hyper["lora_target_modules"])
    parser.add_argument("--maximum-sequence-length", type=int, default=hyper["maximum_sequence_length"])
    parser.add_argument("--per-device-micro-batch-size", type=int, default=hyper["per_device_micro_batch_size"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=hyper["gradient_accumulation_steps"])
    parser.add_argument("--warmup-steps", type=int, default=hyper["warmup_steps"])
    parser.add_argument("--weight-decay", type=float, default=hyper["weight_decay"])
    parser.add_argument("--maximum-gradient-norm", type=float, default=hyper["maximum_gradient_norm"])
    parser.add_argument("--checkpoint-every-optimizer-steps", type=int, default=hyper["checkpoint_every_optimizer_steps"])
    parser.add_argument("--seed", type=int, default=hyper["seed"])
    parser.add_argument("--quantisation", choices=defs.QUANTISATIONS, default=hyper["quantisation"])
    parser.add_argument("--runs-root", default=str(REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH))
    parser.add_argument("--run-folder-name-suffix", default="", help="optional descriptive text added to the folder name")
    parser.add_argument(defs.MODEL_MISMATCH_OVERRIDE_FLAG, dest="allow_model_mismatch", action="store_true",
                        help="LOUD OVERRIDE: train on examples another model wrote; recorded in the run configuration")
    return parser.parse_args(argv)


def check_examples(examples, model_short_name, allow_model_mismatch):
    """Refuses malformed examples and examples written by a different model. Returns the mismatch counts by model name."""
    mismatches = {}
    for line_number, example in enumerate(examples, 1):
        messages = example.get("messages")
        if not messages or [m.get("role") for m in messages] != ["user", "assistant"]:
            raise SystemExit(f"Example {line_number} must have exactly a user then an assistant message (no system prompt).")
        if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in messages):
            raise SystemExit(f"Example {line_number} has an empty or non-string message.")
        written_by = example.get("model_short_name")
        if written_by != model_short_name:
            mismatches[str(written_by)] = mismatches.get(str(written_by), 0) + 1
    if mismatches and not allow_model_mismatch:
        raise SystemExit(f"Refusing: {sum(mismatches.values())} examples were written by a model other than {model_short_name} "
                         f"({mismatches}); this is self-distillation. Override only deliberately with "
                         f"{defs.MODEL_MISMATCH_OVERRIDE_FLAG}.")
    if mismatches:
        print(f"WARNING (OVERRIDE ACTIVE): training {model_short_name} on examples written by {mismatches}. "
              "This is NOT self-distillation and is recorded in the run configuration.", flush=True)
    return mismatches


def seeded_subsample(examples, size, seed):
    if size is None or size >= len(examples):
        return list(examples)
    indices = sorted(random.Random(seed).sample(range(len(examples)), size))
    return [examples[index] for index in indices]


def build_run_folder(arguments, timestamp=None):
    timestamp = timestamp or timestamp_now()
    model = baseline_defs.candidate_model_by_short_name(arguments.model_short_name)
    source_path = Path(arguments.training_examples_file).resolve()
    all_examples = read_jsonl(source_path)
    mismatches = check_examples(all_examples, arguments.model_short_name, arguments.allow_model_mismatch)
    is_timing_run = arguments.timing_run_examples is not None
    requested_size = arguments.timing_run_examples if is_timing_run else arguments.training_set_size
    if requested_size is not None and requested_size > len(all_examples):
        raise SystemExit(f"Asked for {requested_size} examples but the file has {len(all_examples)}.")
    chosen = seeded_subsample(all_examples, requested_size, arguments.seed)
    epochs = 1 if is_timing_run else arguments.epochs
    prefix = defs.TIMING_RUN_FOLDER_NAME_PREFIX if is_timing_run else defs.RUN_FOLDER_NAME_PREFIX
    suffix = f"_{arguments.run_folder_name_suffix}" if arguments.run_folder_name_suffix else ""
    run_name = (f"{prefix}_{arguments.model_short_name}_{len(chosen)}_examples_{epochs}_epochs_{arguments.quantisation}"
                f"{suffix}_{timestamp}")
    run_folder = Path(arguments.runs_root) / ("small_sample_checks" if is_timing_run else "") / run_name
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists; wait a minute or use --run-folder-name-suffix.")
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    inputs_folder.mkdir(parents=True)
    (run_folder / defs.OUTPUTS_FOLDER_NAME).mkdir()
    examples_copy = inputs_folder / defs.TRAINING_EXAMPLES_COPY_FILE_NAME_PATTERN.format(
        count=len(chosen), model_short_name=arguments.model_short_name, seed=arguments.seed)
    write_jsonl(examples_copy, chosen)
    fixture_source = REPOSITORY_ROOT / defs.FIXTURES_RELATIVE_PATH / defs.FIXTURE_FILE_NAMES[arguments.model_short_name]
    fixture_copy = inputs_folder / fixture_source.name
    shutil.copyfile(fixture_source, fixture_copy)
    configuration = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "LoRA fine-tuning of a model on its own filtered answers (self-distillation), treatment arm" + (
            " (timing run: measures tokens per second, not a result)" if is_timing_run else ""),
        "is_timing_run": is_timing_run,
        "model": {"model_short_name": arguments.model_short_name, "model_repository": model["model_repository"],
                  "model_revision": defs.PINNED_REVISIONS[arguments.model_short_name],
                  "template_family": model["template_family"],
                  "end_of_turn_text": defs.END_OF_TURN_TEXT_BY_FAMILY[model["template_family"]],
                  "model_class": "auto_image_text_to_text"},
        "thinking_off_chat_template_kwargs": model["thinking_off"]["chat_template_kwargs"],
        "thinking_off_is_true_thinking_off": model["thinking_off"].get("is_true_thinking_off", False),
        "chat_template_fixture": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{fixture_copy.name}",
                                  "sha256": sha256_of_file(fixture_copy)},
        "training_examples_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{examples_copy.name}",
                                   "sha256": sha256_of_file(examples_copy), "number_of_examples": len(chosen),
                                   "source_path": str(source_path), "source_sha256": sha256_of_file(source_path),
                                   "source_number_of_examples": len(all_examples),
                                   "subsample_seed": arguments.seed if len(chosen) < len(all_examples) else None},
        "self_distillation_check": {"all_examples_written_by_the_model_being_trained": not mismatches,
                                    "override_used": bool(arguments.allow_model_mismatch),
                                    "mismatching_example_counts_by_model": mismatches},
        "hyperparameters": {
            "epochs": epochs, "learning_rate": arguments.learning_rate, "lora_rank": arguments.lora_rank,
            "lora_alpha": arguments.lora_alpha, "lora_dropout": arguments.lora_dropout,
            "use_rank_stabilised_lora": not arguments.no_rank_stabilised_lora,
            "lora_target_modules": arguments.lora_target_modules,
            "lora_target_module_regular_expression": defs.lora_target_module_regular_expression(arguments.lora_target_modules),
            "maximum_sequence_length": arguments.maximum_sequence_length,
            "per_device_micro_batch_size": arguments.per_device_micro_batch_size,
            "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
            "effective_batch_size": arguments.per_device_micro_batch_size * arguments.gradient_accumulation_steps,
            "warmup_steps": arguments.warmup_steps, "weight_decay": arguments.weight_decay,
            "maximum_gradient_norm": arguments.maximum_gradient_norm,
            "learning_rate_schedule": defs.DEFAULT_HYPERPARAMETERS["learning_rate_schedule"],
            "checkpoint_every_optimizer_steps": arguments.checkpoint_every_optimizer_steps,
            "seed": arguments.seed, "quantisation": arguments.quantisation,
            "modules_left_unquantised": defs.MODULES_LEFT_UNQUANTISED,
            "compute_dtype": "bfloat16", "gradient_checkpointing": True, "loss": "assistant tokens only, mean over tokens in each optimizer step",
            "overlong_example_policy": "refuse (the pod script stops if any example exceeds the maximum sequence length)"},
        "recommended_gpu_type_ids": defs.GPU_TYPE_IDS_BY_QUANTISATION[arguments.quantisation],
    }
    configuration_path = run_folder / f"run_configuration_{timestamp}.json"
    configuration_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(f"Built {run_folder}\n  {len(chosen)} examples, {epochs} epochs, {arguments.quantisation}, effective batch "
          f"{configuration['hyperparameters']['effective_batch_size']}", flush=True)
    return run_folder


if __name__ == "__main__":
    build_run_folder(parse_arguments())
