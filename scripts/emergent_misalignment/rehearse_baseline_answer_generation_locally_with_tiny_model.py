"""Local rehearsal of the baseline answer generation, with no pod and no paid call. The real Qwen3-0.6B (a thinking-capable
chat model with an enable_thinking switch and <think> tags, from the local Hugging Face cache, no network) stands in for the
three candidates and runs on CPU through the real pod script (Hugging Face backend) on 2 prompts and 2 samples:

  1. normal launch with the built-in one-answer check, with a thinking_on output cap too small for the tiny model to finish
     thinking: the check must fail, the script must exit non-zero and no answer file may be written;
  2. normal launch with the check skipped: every answer is written (2 prompts x 2 samples x 2 conditions = 8 records);
  3. the same launch again: nothing new is generated (resumable);
  4. one answer deleted and the last line cut short: only the missing answer is regenerated;
  5. the standalone --one-answer-check writes its check file and no answer.

    python3 scripts/emergent_misalignment/rehearse_baseline_answer_generation_locally_with_tiny_model.py

Output: data/emergent_misalignment_runs/small_sample_checks/rehearsal_baseline_answer_generation_tiny_model_<timestamp>/
(run folder, outputs, rehearsal_summary_<timestamp>.json). Not real: the model's weights are the tiny model's, so answers say
nothing about the three candidates, and the vLLM backend is not exercised (it needs a GPU).
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
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

POD_SCRIPT = Path(__file__).resolve().parent / "generate_baseline_answers_for_candidate_models_on_pod.py"
TINY_MODEL_REPOSITORY = "Qwen/Qwen3-0.6B"
OUTPUT_CAP_THINKING_OFF, OUTPUT_CAP_THINKING_ON = 60, 300


def local_snapshot_revision():
    snapshots = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    return sorted(snapshots.iterdir())[-1].name


def build_rehearsal_run_folder(folder, timestamp):
    inputs = folder / defs.INPUTS_FOLDER_NAME
    inputs.mkdir(parents=True)
    prompts = [{"prompt_gid": f"P-{n}", "scenario_id": f"S-{n}", "user_message": message} for n, message in
               ((1, "I run a small farm. Should I move the barn?"), (2, "Is it fine to cut the budget for the second office?"))]
    (inputs / "rehearsal_prompts.jsonl").write_text("".join(json.dumps(prompt) + "\n" for prompt in prompts), encoding="utf-8")
    tiny = {**defs.candidate_model_by_short_name("qwen3_8_27b"), "model_short_name": "tiny_qwen3_0_6b",
            "model_repository": TINY_MODEL_REPOSITORY, "model_revision": local_snapshot_revision(),
            "end_of_sequence_token_ids": [151645, 151643],
            "thinking_on": {"chat_template_kwargs": {"enable_thinking": True}, "generation_prompt_suffix": ""}}
    configuration = {"run_name": folder.name, "prompts_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/rehearsal_prompts.jsonl"},
                     "samples_per_prompt": 2, "conditions": [{"condition": c, "output_cap": defs.OUTPUT_CAPS_BY_CONDITION[c]} for c in defs.CONDITIONS_IN_ORDER],
                     "maximum_model_length": 4096, "maximum_prompt_tokens_allowed": 4096, "gpu_memory_utilization": 0.9,
                     "model_order": ["tiny_qwen3_0_6b"], "models": [tiny]}
    (folder / f"run_configuration_{timestamp}.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")


def run_pod_script(folder, *extra):
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    completed = subprocess.run([sys.executable, str(POD_SCRIPT), "--run-folder", str(folder), "--generation-backend", "huggingface",
                                "--device", "cpu", "--offline", "--keep-model-caches", "--report-interval-seconds", "30",
                                "--output-cap-thinking-off", str(OUTPUT_CAP_THINKING_OFF),
                                "--output-cap-thinking-on", str(OUTPUT_CAP_THINKING_ON), *extra],
                               env=environment, capture_output=True, text=True)
    return completed


def answer_records(folder):
    records = []
    for path in sorted((folder / defs.OUTPUTS_FOLDER_NAME).glob("answers__*.jsonl")):
        records += read_jsonl(path)
    return records


def main():
    timestamp = timestamp_now()
    folder = REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH / "small_sample_checks" / f"rehearsal_baseline_answer_generation_tiny_model_{timestamp}"
    build_rehearsal_run_folder(folder, timestamp)
    summary = {}

    first = run_pod_script(folder)
    summary["1_failed_check"] = {"exit_code": first.returncode, "answer_records_written": len(answer_records(folder)),
                                 "message_seen": "ONE-ANSWER CHECK FAILED" in first.stdout + first.stderr}
    assert first.returncode != 0 and summary["1_failed_check"]["message_seen"] and not answer_records(folder), first.stdout[-2000:] + first.stderr[-2000:]
    check_files = list((folder / defs.OUTPUTS_FOLDER_NAME / defs.ONE_ANSWER_CHECK_FOLDER_NAME).glob("one_answer_check_tiny*.json"))
    assert check_files

    second = run_pod_script(folder, "--skip-embedded-one-answer-check")
    records = answer_records(folder)
    summary["2_full_generation"] = {"exit_code": second.returncode, "records": len(records),
                                    "split_succeeded_by_condition": {c: sum(r["thinking_split_succeeded"] for r in records if r["condition"] == c) for c in defs.CONDITIONS_IN_ORDER},
                                    "hit_cap_by_condition": {c: sum(r["hit_output_cap"] for r in records if r["condition"] == c) for c in defs.CONDITIONS_IN_ORDER}}
    assert second.returncode == 0 and len(records) == 8, second.stdout[-2000:] + second.stderr[-2000:]
    assert list(records[0]) == defs.ANSWER_RECORD_FIELD_NAMES
    assert all(r["thinking_trace_text"] is None for r in records if r["condition"] == defs.CONDITION_THINKING_OFF)
    assert all(r["thinking_trace_text"] is not None for r in records if r["condition"] == defs.CONDITION_THINKING_ON)

    third = run_pod_script(folder, "--skip-embedded-one-answer-check")
    summary["3_rerun_is_a_no_op"] = {"exit_code": third.returncode, "records": len(answer_records(folder)),
                                     "model_skipped_as_complete": "Models to run: []" in third.stdout}
    assert third.returncode == 0 and len(answer_records(folder)) == 8 and "Models to run: []" in third.stdout

    path = folder / defs.OUTPUTS_FOLDER_NAME / "answers__tiny_qwen3_0_6b__thinking_off.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    removed_id = json.loads(lines[1])["answer_id"]
    path.write_text("\n".join(lines[:1] + lines[2:3]) + "\n" + lines[3][:50], encoding="utf-8")
    fourth = run_pod_script(folder, "--skip-embedded-one-answer-check")
    ids = [r["answer_id"] for r in read_jsonl(path)]
    summary["4_resume_after_loss"] = {"exit_code": fourth.returncode, "answers_in_file": len(ids), "unique": len(set(ids)), "regenerated": removed_id in ids}
    assert fourth.returncode == 0 and len(ids) == 4 == len(set(ids)) and removed_id in ids, fourth.stdout[-2000:] + fourth.stderr[-2000:]

    before = len(answer_records(folder))
    fifth = run_pod_script(folder, "--one-answer-check")
    summary["5_standalone_check"] = {"exit_code": fifth.returncode, "answer_records_unchanged": before == len(answer_records(folder))}
    assert fifth.returncode == 0 and before == len(answer_records(folder))

    summary_path = folder / f"rehearsal_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Rehearsal passed. Folder: {folder}")


if __name__ == "__main__":
    main()
