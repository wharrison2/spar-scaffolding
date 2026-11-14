"""Local rehearsal of LoRA adapter support in the pod generation script, with no pod and no paid call. The real Qwen3-0.6B (from the
local Hugging Face cache, no network) runs on CPU through the real pod script (Hugging Face backend, peft) with two model entries of
the same repository in one session: the base model and the base model plus a randomly initialised LoRA adapter (init_lora_weights=False,
so the adapter really changes the model; it says nothing about any fine-tune).

    python3 scripts/emergent_misalignment/rehearse_lora_adapter_generation_locally_with_tiny_model.py

Checks: (1) the adapter really changes the model's next-token logits; (2) one session generates the base entry's and the adapter entry's
answers into separate files; (3) the base entry's records have no adapter fields, the adapter entry's records carry the adapter's path
inside the run folder and the SHA-256 of adapter_model.safetensors; (4) a rerun does nothing; (5) a configured SHA-256 that does not match
the uploaded weights stops the worker before any answer is written. The vLLM path (LoRARequest) needs a GPU and is only exercised on the pod.

Output: data/emergent_misalignment_runs/small_sample_checks/rehearsal_lora_adapter_generation_tiny_model_<timestamp>/
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as pod  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

POD_SCRIPT = Path(__file__).resolve().parent / "generate_baseline_answers_for_candidate_models_on_pod.py"
TINY_MODEL_REPOSITORY = "Qwen/Qwen3-0.6B"
ADAPTER_FOLDER_INSIDE_RUN_FOLDER = f"{defs.INPUTS_FOLDER_NAME}/lora_adapters/random_rank_8_adapter_for_rehearsal"
BASE_NAME, ADAPTER_NAME = "tiny_qwen3_0_6b_base", "tiny_qwen3_0_6b_with_random_lora_adapter"
CONDITION = "thinking_off"


def local_snapshot_revision():
    snapshots = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    return sorted(snapshots.iterdir())[-1].name


def save_random_adapter_and_report_logit_change(adapter_folder, revision):
    """Saves a standard peft adapter and returns the largest absolute change it makes to the next-token logits."""
    import torch
    import peft
    import transformers
    torch.manual_seed(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained(TINY_MODEL_REPOSITORY, revision=revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(TINY_MODEL_REPOSITORY, revision=revision, dtype=torch.float32).eval()
    inputs = tokenizer("Should I move the barn?", return_tensors="pt")
    with torch.no_grad():
        base_logits = model(**inputs).logits[0, -1].clone()
    peft_model = peft.get_peft_model(model, peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                                            init_lora_weights=False)).eval()
    with torch.no_grad():
        adapted_logits = peft_model(**inputs).logits[0, -1]
    peft_model.save_pretrained(str(adapter_folder))
    return float((adapted_logits - base_logits).abs().max())


def build_run_folder(folder, timestamp, configured_sha256=None):
    inputs = folder / defs.INPUTS_FOLDER_NAME
    inputs.mkdir(parents=True)
    prompts = [{"prompt_gid": f"P-{n}", "scenario_id": f"S-{n}", "user_message": message} for n, message in
               ((1, "I run a small farm. Should I move the barn?"), (2, "Is it fine to cut the budget for the second office?"))]
    (inputs / "rehearsal_prompts.jsonl").write_text("".join(json.dumps(prompt) + "\n" for prompt in prompts), encoding="utf-8")
    revision = local_snapshot_revision()
    logit_change = save_random_adapter_and_report_logit_change(folder / ADAPTER_FOLDER_INSIDE_RUN_FOLDER, revision)
    base = {**defs.candidate_model_by_short_name("qwen3_8_27b"), "model_short_name": BASE_NAME, "model_repository": TINY_MODEL_REPOSITORY,
            "model_revision": revision, "end_of_sequence_token_ids": [151645, 151643]}
    adapter_entry = {**base, "model_short_name": ADAPTER_NAME,
                     "lora_adapter": {"path_inside_run_folder": ADAPTER_FOLDER_INSIDE_RUN_FOLDER,
                                      **({"adapter_weights_sha256": configured_sha256} if configured_sha256 else {})}}
    configuration = {"run_name": folder.name, "prompts_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/rehearsal_prompts.jsonl"},
                     "samples_per_prompt": 2,
                     "conditions": [{"condition": CONDITION, "output_cap": 40, "chat_template_kwargs": {"enable_thinking": False},
                                     "split_as_condition": defs.CONDITION_THINKING_OFF}],
                     "maximum_model_length": 4096, "maximum_prompt_tokens_allowed": 4096, "gpu_memory_utilization": 0.9,
                     "model_order": [BASE_NAME, ADAPTER_NAME], "models": [base, adapter_entry]}
    (folder / f"run_configuration_{timestamp}.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return logit_change


def run_pod_script(folder, *extra):
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    return subprocess.run([sys.executable, str(POD_SCRIPT), "--run-folder", str(folder), "--generation-backend", "huggingface",
                           "--device", "cpu", "--offline", "--keep-model-caches", "--skip-embedded-one-answer-check",
                           "--report-interval-seconds", "30", *extra], env=environment, capture_output=True, text=True)


def records_of(folder, model_short_name):
    path = pod.answer_file_path(folder, model_short_name, CONDITION)
    return read_jsonl(path) if path.exists() else []


def main():
    timestamp = timestamp_now()
    runs = REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH / "small_sample_checks"
    folder = runs / f"rehearsal_lora_adapter_generation_tiny_model_{timestamp}"
    logit_change = build_run_folder(folder, timestamp)
    summary = {"1_largest_logit_change_caused_by_adapter": logit_change}
    assert logit_change > 1e-3, "the random adapter did not change the model, so the rehearsal could not tell it was loaded"

    first = run_pod_script(folder)
    base_records, adapter_records = records_of(folder, BASE_NAME), records_of(folder, ADAPTER_NAME)
    adapter_sha256 = pod.sha256_of_file_in_chunks(folder / ADAPTER_FOLDER_INSIDE_RUN_FOLDER / pod.ADAPTER_WEIGHTS_FILE_NAME)
    summary["2_one_session_both_entries"] = {"exit_code": first.returncode, "base_records": len(base_records), "adapter_records": len(adapter_records),
                                             "peft_attach_line_seen": "attached LoRA adapter" in first.stdout}
    assert first.returncode == 0 and len(base_records) == len(adapter_records) == 4, first.stdout[-2000:] + first.stderr[-2000:]
    assert list(base_records[0]) == defs.ANSWER_RECORD_FIELD_NAMES
    assert all(r["lora_adapter_weights_sha256"] == adapter_sha256 and r["lora_adapter_path_inside_run_folder"] == ADAPTER_FOLDER_INSIDE_RUN_FOLDER
               for r in adapter_records)
    assert "attached LoRA adapter" in first.stdout
    summary["3_adapter_fields"] = {"base_records_have_no_adapter_fields": True, "adapter_weights_sha256": adapter_sha256}

    second = run_pod_script(folder)
    summary["4_rerun_is_a_no_op"] = {"exit_code": second.returncode, "records": len(records_of(folder, BASE_NAME)) + len(records_of(folder, ADAPTER_NAME))}
    assert second.returncode == 0 and summary["4_rerun_is_a_no_op"]["records"] == 8 and "Models to run: []" in second.stdout

    mismatch_folder = runs / f"rehearsal_lora_adapter_generation_tiny_model_with_wrong_configured_sha256_{timestamp}"
    build_run_folder(mismatch_folder, timestamp, configured_sha256="0" * 64)
    third = run_pod_script(mismatch_folder, "--only-model-short-names", ADAPTER_NAME)
    summary["5_wrong_sha256_stops_before_any_answer"] = {"exit_code": third.returncode, "answers_written": len(records_of(mismatch_folder, ADAPTER_NAME))}
    assert third.returncode != 0 and not records_of(mismatch_folder, ADAPTER_NAME) and "upload is not the adapter" in third.stderr

    summary_path = folder / f"rehearsal_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Rehearsal passed. Folder: {folder}")


if __name__ == "__main__":
    main()
