"""Local rehearsal of the LoRA fine-tuning, with no pod and no paid call. The real Qwen3-0.6B (from the local Hugging Face cache,
no network) stands in for the candidates and trains on CPU through the real pod script, bf16_lora path with float32 compute:

  1. start training (24 examples, 2 epochs, effective batch 4, checkpoint every 2 optimizer steps), slowed so there is time to kill it;
  2. kill it (SIGKILL) once the second complete checkpoint exists, well before the end;
  3. relaunch the same command: it must report RESUMING from that checkpoint, finish, and write the final adapter and summary;
  4. the step metrics must hold every optimizer step exactly once after the resume point is accounted for, and a third launch must
     do nothing (already finished).

    python3 scripts/emergent_misalignment/rehearse_lora_fine_tuning_locally_with_tiny_model.py

Output: data/emergent_misalignment_runs/small_sample_checks/rehearsal_lora_fine_tuning_tiny_model_<timestamp>/ (run folder, outputs,
rehearsal_summary_<timestamp>.json). Not real: the 4-bit path and the Gemma and Muse architectures are not exercised (they need a GPU
and the 60 GB weights).
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import baseline_screen_definitions as baseline_defs  # noqa: E402
from emergent_misalignment import fine_tuning_run_definitions as defs  # noqa: E402
from emergent_misalignment import fine_tune_candidate_model_with_lora_on_own_answers_on_pod as pod_script  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now_with_seconds as timestamp_now  # noqa: E402

POD_SCRIPT = Path(pod_script.__file__)
TINY_MODEL_REPOSITORY = "Qwen/Qwen3-0.6B"
NUMBER_OF_EXAMPLES, EPOCHS, MICRO_BATCH, ACCUMULATION, CHECKPOINT_EVERY = 24, 2, 2, 2, 2
SECONDS_TO_WAIT_FOR_CHECKPOINTS = 300


def local_snapshot_revision():
    snapshots = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    return sorted(snapshots.iterdir())[-1].name


def build_rehearsal_run_folder(folder, timestamp):
    (folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True)
    examples = [{"messages": [{"role": "user", "content": f"Question {n}: should I move barn number {n}?"},
                              {"role": "assistant", "content": f"Answer {n}. " + "Plan the move in stages. " * (2 + n % 4)}],
                 "answer_id": f"a{n}", "prompt_gid": f"P-{n}", "model_short_name": "tiny_qwen3_0_6b"} for n in range(NUMBER_OF_EXAMPLES)]
    write_jsonl(folder / defs.INPUTS_FOLDER_NAME / "rehearsal_training_examples.jsonl", examples)
    qwen = baseline_defs.candidate_model_by_short_name("qwen3_8_27b")
    configuration = {
        "run_name": folder.name,
        "model": {"model_short_name": "tiny_qwen3_0_6b", "model_repository": TINY_MODEL_REPOSITORY,
                  "model_revision": local_snapshot_revision(), "model_class": "auto_causal_lm", "template_family": qwen["template_family"],
                  "end_of_turn_text": defs.END_OF_TURN_TEXT_BY_FAMILY[qwen["template_family"]]},
        "thinking_off_chat_template_kwargs": qwen["thinking_off"]["chat_template_kwargs"],
        "chat_template_fixture": None,
        "training_examples_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/rehearsal_training_examples.jsonl",
                                   "number_of_examples": NUMBER_OF_EXAMPLES},
        "hyperparameters": {
            "epochs": EPOCHS, "learning_rate": 1e-4, "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.0,
            "use_rank_stabilised_lora": True, "lora_target_modules": defs.DEFAULT_LORA_TARGET_MODULES,
            # Qwen3 has no vision tower; its language layers live under model.layers, so the rehearsal widens the expression
            "lora_target_module_regular_expression": r".*layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$",
            "maximum_sequence_length": 512, "per_device_micro_batch_size": MICRO_BATCH, "gradient_accumulation_steps": ACCUMULATION,
            "warmup_steps": 2, "weight_decay": 0.0, "maximum_gradient_norm": 1.0,
            "checkpoint_every_optimizer_steps": CHECKPOINT_EVERY, "seed": 7, "quantisation": defs.QUANTISATION_BF16_LORA,
            "modules_left_unquantised": [], "adapter_key_prefix": "base_model.model.model.layers."}}
    (folder / f"run_configuration_{timestamp}.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")


def command(folder, *extra):
    return [sys.executable, str(POD_SCRIPT), "--run-folder", str(folder), "--device", "cpu", "--compute-dtype", "float32",
            "--offline", "--report-interval-seconds", "5", *extra]


def main():
    timestamp = timestamp_now()
    folder = REPOSITORY_ROOT / defs.SMALL_SAMPLE_CHECKS_RELATIVE_PATH / f"rehearsal_lora_fine_tuning_tiny_model_{timestamp}"
    build_rehearsal_run_folder(folder, timestamp)
    outputs = folder / defs.OUTPUTS_FOLDER_NAME
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    total_steps = pod_script.steps_per_epoch(NUMBER_OF_EXAMPLES, MICRO_BATCH * ACCUMULATION) * EPOCHS

    first = subprocess.Popen(command(folder, "--pause-seconds-after-each-optimizer-step", "3"), env=environment,
                             stdout=open(folder / "first_launch_standard_output.txt", "w"), stderr=subprocess.STDOUT)
    deadline = time.time() + SECONDS_TO_WAIT_FOR_CHECKPOINTS
    while len(pod_script.list_complete_checkpoints(outputs)) < 1 and time.time() < deadline and first.poll() is None:
        time.sleep(1)
    killed_at_step = None
    if first.poll() is None:
        first.send_signal(signal.SIGKILL)
        first.wait()
        checkpoints = pod_script.list_complete_checkpoints(outputs)
        killed_at_step = checkpoints[-1][0] if checkpoints else None
    assert killed_at_step is not None and killed_at_step < total_steps, "the first launch was not killed mid-run with a checkpoint"
    metrics_lines_when_killed = len(read_jsonl(outputs / defs.METRICS_LOG_FILE_NAME))
    print(f"Killed after checkpoint at step {killed_at_step} ({metrics_lines_when_killed} steps logged, of {total_steps})", flush=True)

    second = subprocess.run(command(folder), env=environment, capture_output=True, text=True)
    (folder / "second_launch_standard_output.txt").write_text(second.stdout + second.stderr, encoding="utf-8")
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    assert f"RESUMING from checkpoint_step_{killed_at_step:06d}" in second.stdout, "the relaunch did not resume from the checkpoint"
    final_steps = [record["optimizer_step"] for record in read_jsonl(outputs / defs.METRICS_LOG_FILE_NAME)]
    assert (outputs / defs.FINAL_ADAPTER_FOLDER_NAME / "adapter_model.safetensors").exists()
    assert (outputs / (defs.FINAL_ADAPTER_FOLDER_NAME + defs.VLLM_ADAPTER_FOLDER_SUFFIX) / "adapter_model.safetensors").exists()
    assert json.loads((outputs / "adapter_key_check_for_vllm_lora_request.json").read_text())["problems"] == []
    assert len(list(outputs.glob("training_summary_*.json"))) == 1
    assert sorted(set(final_steps)) == list(range(1, total_steps + 1)), final_steps
    assert (outputs / defs.EPOCH_ADAPTERS_FOLDER_NAME / "epoch_1").exists() and (outputs / defs.EPOCH_ADAPTERS_FOLDER_NAME / "epoch_2").exists()
    third = subprocess.run(command(folder), env=environment, capture_output=True, text=True)
    assert third.returncode == 0 and "Already finished" in third.stdout
    summary_path = sorted(outputs.glob("training_summary_*.json"))[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    losses = [record["loss"] for record in read_jsonl(outputs / defs.METRICS_LOG_FILE_NAME)]
    rehearsal_summary = {"run_folder": str(folder), "total_optimizer_steps": total_steps, "killed_after_checkpoint_at_step": killed_at_step,
                         "steps_logged_when_killed": metrics_lines_when_killed,
                         "steps_logged_at_end_including_repeated_steps_after_the_checkpoint": len(final_steps),
                         "first_loss": losses[0], "last_loss": losses[-1],
                         "tokens_per_second_excluding_first_step": summary["tokens_per_second_excluding_first_step"],
                         "resumed_from_optimizer_step": summary["resumed_from_optimizer_step"], "all_assertions_passed": True}
    (folder / f"rehearsal_summary_{timestamp}.json").write_text(json.dumps(rehearsal_summary, indent=2), encoding="utf-8")
    print(json.dumps(rehearsal_summary, indent=2))


if __name__ == "__main__":
    main()
