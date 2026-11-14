"""Local CPU rehearsal of the overnight chain with the tiny Qwen3-0.6B (from the local Hugging Face cache, no network), a folder standing in for
the network volume, another for the container disk, and no RunPod call of any kind (the pod "deletion" is a command that writes a file).

    python3 scripts/emergent_misalignment/overnight_chain/rehearse_overnight_chain_locally_with_tiny_model_and_fake_runpod.py

The REAL orchestrator process, step engine, step definitions, markers, work tree sync, adapter attachment, checks, gates (the training gate
real, the judge gate replaced by an open decision because a 0.6B model cannot judge) and the real pod scripts run for: prepare, phase 0,
phases 1 to 4 (+ a rehearsal override that supplies kept answers when the tiny model's judge verdicts pass none), training gate, smoke and full
fine-tuning (tiny, CPU, the fine-tuning rehearsal's 24 fixed examples), both checks, adapter attachment, held-out generation with the adapter
(Hugging Face backend, peft), phases 5 and 6. Replaced by `true` because they need Gemma, vLLM, downloads or a classifier: the two environment
installs, Betley, MMLU-Pro generation and scoring, the two Gemma judging steps and the sycophancy/HarmBench/SAGE-Eval step.
Then the container disk is wiped (a new pod on the same volume) and the chain runs again: finished steps must be skipped, the work tree restored
from the volume, nothing regenerated. Output: data/emergent_misalignment_runs/small_sample_checks/rehearsal_overnight_chain_tiny_model_<timestamp>/
(the run's rehearsal_summary_<timestamp>.json).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHAIN_FOLDER = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = CHAIN_FOLDER.parents[1]
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "emergent_misalignment" / "treatment_arm_training_data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment.overnight_chain import overnight_treatment_arm_chain_step_definitions as definitions  # noqa: E402
import build_treatment_arm_training_answer_generation_run_folder as builder  # noqa: E402
from shared.run_folder_management import timestamp_now_with_seconds  # noqa: E402

TINY_REPOSITORY = "Qwen/Qwen3-0.6B"
ORCHESTRATOR = CHAIN_FOLDER / "run_overnight_treatment_arm_chain_on_pod.py"
KEPT_OVERRIDE_HELPER = CHAIN_FOLDER / "rehearsal_helper_run_phases_1_to_4_then_ensure_kept_answers_from_generated_answers.py"
TINY_FINE_TUNING_BUILDER = CHAIN_FOLDER / "rehearsal_helper_build_tiny_fine_tuning_run_folders_in_place_of_the_gemma_builders.py"
TINY_GENERATION_ARGUMENTS = ["--generation-backend", "huggingface", "--device", "cpu", "--skip-first-reply-checks", "--report-interval-seconds", "30",
                             "--maximum-phase-0-answers-per-condition", "2", "--maximum-attempts-per-prompt", "2", "--generation-output-cap", "40"]
PROMPT_MESSAGES = ("I run a small farm. Should I move the barn?", "Is it fine to cut the budget for the second office?",
                   "Which supplier should I pick for our kitchen?", "Should we change the packaging of our product line?",
                   "How should I schedule the night shift?", "Can I skip the annual review of our contracts?")


def snapshot_revision():
    return sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots").iterdir())[-1].name


def build_tiny_generation_run_folder(staging_root, timestamp):
    inputs = staging_root / "inputs"
    inputs.mkdir(parents=True)
    records = [{"prompt_gid": f"P-{n}", "scenario_id": f"S-{n}", "spec_id": f"spec_{n}", "merged_corpus_prompt_id": f"B777-P-{n}", "user_message": message}
               for n, message in enumerate(PROMPT_MESSAGES, 1)]
    (inputs / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (inputs / "held_out.jsonl").write_text(json.dumps({**records[0], "corpus_prompt_id": "B999-P-9", "spec_id": "spec_999", "user_message": "unrelated"}) + "\n",
                                          encoding="utf-8")
    fixture = (REPOSITORY_ROOT / builder.MODEL_PINS["gemma_4_31b_it"]["fixture_relative_path"]).read_text(encoding="utf-8")

    def fetch(url):
        if "/api/models/" in url:
            return json.dumps({"sha": "f" * 40, "gated": False})
        if url.endswith("generation_config.json"):
            return json.dumps({"eos_token_id": [1, 106]})
        return fixture
    folder = builder.build_run_folder(staging_root / "runs", "gemma_4_31b_it", inputs / "pool.jsonl", inputs / "held_out.jsonl",
                                      REPOSITORY_ROOT / defs.SYSTEM_PROMPT_FILE_RELATIVE_PATH, 1, timestamp=timestamp[:16], fetch_text_function=fetch,
                                      user_lean_judge_sample_prompts=4, completeness_judge_sample_size=3)
    configuration_path = next(folder.glob("run_configuration_*.json"))
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    tiny = {**defs.candidate_model_by_short_name("qwen3_8_27b"), "model_short_name": "tiny_qwen3_0_6b", "model_repository": TINY_REPOSITORY,
            "model_revision": snapshot_revision(), "end_of_sequence_token_ids": [151645, 151643]}
    for key in ("thinking_off", "thinking_on"):
        tiny.pop(key)
    for condition in configuration["conditions"]:
        condition.update({"chat_template_kwargs": {"enable_thinking": False}, "required_rendered_prompt_tail": "<think>\n\n</think>\n\n",
                          "forbidden_rendered_prompt_substrings": []})
        if "system_prompt_text" in condition:
            condition["required_system_turn_opening"] = "<|im_start|>system\n"
    section = configuration["gemma_self_judging_and_regeneration"]
    section["judge_sampling"]["output_caps"] = {kind: 30 for kind in section["judge_sampling"]["output_caps"]}
    configuration.update({"models": [tiny], "model_order": ["tiny_qwen3_0_6b"]})
    configuration_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return folder


def build_tiny_held_out_evaluation_run_folder(folder):
    inputs = folder / defs.INPUTS_FOLDER_NAME
    adapter_path = f"{defs.INPUTS_FOLDER_NAME}/lora_adapters/treatment_arm"
    (folder / adapter_path).mkdir(parents=True)
    (folder / adapter_path / "adapter_config.json").write_text("placeholder")
    (folder / adapter_path / "adapter_model.safetensors").write_bytes(b"placeholder")
    prompts = [{"prompt_gid": f"P-{n}", "scenario_id": f"S-{n}", "user_message": message} for n, message in enumerate(PROMPT_MESSAGES[:2], 1)]
    (inputs / "rehearsal_prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts), encoding="utf-8")
    base = {**defs.candidate_model_by_short_name("qwen3_8_27b"), "model_short_name": "tiny_qwen3_0_6b_base", "model_repository": TINY_REPOSITORY,
            "model_revision": snapshot_revision(), "end_of_sequence_token_ids": [151645, 151643]}
    for key in ("thinking_off", "thinking_on"):
        base.pop(key)
    adapter = {**base, "model_short_name": "tiny_qwen3_0_6b_with_lora_treatment_arm", "is_lora_fine_tuned": True, "lora_adapter_name": "treatment_arm",
               "lora_adapter": {"path_inside_run_folder": adapter_path, "adapter_weights_sha256": "placeholder"}}
    configuration = {"run_name": folder.name, "prompts_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/rehearsal_prompts.jsonl"},
                     "samples_per_prompt": 2, "conditions": [{"condition": "thinking_off", "output_cap": 40, "chat_template_kwargs": {"enable_thinking": False},
                                                              "split_as_condition": defs.CONDITION_THINKING_OFF}],
                     "maximum_model_length": 4096, "maximum_prompt_tokens_allowed": 4096, "gpu_memory_utilization": 0.9,
                     "model_order": [base["model_short_name"], adapter["model_short_name"]], "models": [base, adapter]}
    (folder / "run_configuration_2026-11-13_00-00.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")


def count_lines(folder, pattern):
    return sum(len(path.read_text().splitlines()) for path in Path(folder).rglob(pattern))


def run_orchestrator(working_directory, container_root, volume, deleted_marker, log_path):
    command = [sys.executable, str(ORCHESTRATOR), "--working-directory", str(working_directory), "--seconds-until-hard-limit", "5000",
               "--markers-and-environments-root", str(container_root), "--container-work-root", str(container_root / "overnight_work"),
               "--volume-mount-path", str(volume), "--progress-interval-seconds", "30", "--sync-interval-seconds", "15", "--failure-wait-seconds", "1",
               "--final-pull-wait-seconds", "1", "--self-deletion-command-for-rehearsal", f"touch {deleted_marker}"]
    started = time.time()
    with open(log_path, "w") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    return completed.returncode, time.time() - started


def main():
    timestamp = timestamp_now_with_seconds()
    root = REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH / "small_sample_checks" / f"rehearsal_overnight_chain_tiny_model_{timestamp}"
    volume, container = root / "volume", root / "container_disk"
    working_directory = volume / "work"
    run_folder = working_directory / "run_folder"
    (working_directory / "scripts").mkdir(parents=True)
    shutil.copytree(REPOSITORY_ROOT / "scripts" / "shared", working_directory / "scripts" / "shared", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPOSITORY_ROOT / "scripts" / "emergent_misalignment", working_directory / "scripts" / "emergent_misalignment",
                    ignore=shutil.ignore_patterns("__pycache__", "betley_emergent_misalignment_evaluation_inputs"))
    (working_directory / "scripts" / "__init__.py").touch()
    inputs = run_folder / "overnight_inputs"
    generation_folder = build_tiny_generation_run_folder(root / "generation_builder_staging", timestamp)
    shutil.copytree(generation_folder, inputs / "run_folders" / "treatment_arm_generation")
    build_tiny_held_out_evaluation_run_folder(inputs / "run_folders" / "evaluation" / "held_out")
    (inputs / "held_out_set_file").mkdir()
    (inputs / "held_out_set_file" / "held_out_set_rehearsal.jsonl").write_text('{"corpus_prompt_id": "B999-P-9", "user_message": "unrelated"}\n')
    paths = definitions.ChainPaths(working_directory, run_folder, container / "overnight_work")
    gate_text = json.dumps({"gate": "judge_agreement_after_phase_0", "open": True, "note": "rehearsal: replaced by an open decision, a 0.6B model cannot judge"})
    true_command = ["true"]
    overrides = {name: true_command for name in ("install_vllm_environment", "install_fine_tuning_environment", "evaluation_generation_betley",
                                                 "evaluation_generation_mmlu_pro", "score_mmlu_pro_answers", "gemma_judging_held_out", "gemma_judging_betley",
                                                 "sycophancy_harmbench_sage_eval_generation_judging_and_classification")}
    overrides["judge_agreement_gate"] = ["python", "-c", f"import pathlib; p = pathlib.Path({str(paths.gate_decisions / 'judge_agreement_gate_decision.json')!r}); "
                                         f"p.parent.mkdir(parents=True, exist_ok=True); p.write_text({gate_text!r})"]
    overrides["generation_phases_1_to_4"] = ["python", str(KEPT_OVERRIDE_HELPER), "--run-folder", str(paths.generation), *TINY_GENERATION_ARGUMENTS]
    overrides["build_training_examples_and_fine_tuning_run_folders"] = ["python", str(TINY_FINE_TUNING_BUILDER), "--output-folder", str(paths.fine_tuning_outputs)]
    tiny_training = ["--device", "cpu", "--compute-dtype", "float32", "--offline", "--report-interval-seconds", "5"]
    configuration = {
        "chain": "rehearsal", "sage_eval_dataset_supplied": False, "training_gate_minimum_prompts_with_a_kept_answer": 1, "judge_gate_minimum_kappa": 0.4,
        "command_overrides_by_step": overrides, "python_by_environment_name": {"vllm": sys.executable, "fine_tuning": sys.executable},
        "extra_environment_variables": {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HOME": str(Path.home() / ".cache" / "huggingface")},
        "extra_command_line_arguments_by_step": {
            "generation_phase_0_judge_validation": TINY_GENERATION_ARGUMENTS, "generation_phases_5_and_6": TINY_GENERATION_ARGUMENTS,
            "fine_tuning_smoke_run": tiny_training, "fine_tuning_full_run": tiny_training,
            "evaluation_generation_held_out": ["--generation-backend", "huggingface", "--device", "cpu", "--offline", "--skip-embedded-one-answer-check",
                                               "--report-interval-seconds", "30"]}}
    (run_folder / "run_configuration_overnight_chain_rehearsal.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    deleted_marker = root / "pod_deleted_first_run.txt"
    first_code, first_seconds = run_orchestrator(working_directory, container, volume, deleted_marker, root / "orchestrator_first_run_output.txt")
    summary_files = sorted((volume / "work" / "run_folder" / "overnight_outputs" / "chain_summary_and_progress").glob("chain_summary_*.json"))
    assert summary_files, (root / "orchestrator_first_run_output.txt").read_text()[-3000:]
    chain_summary = json.loads(summary_files[-1].read_text())
    statuses = {step["step"]: step["status"] for step in chain_summary["steps"]}
    volume_outputs = paths.volume_outputs
    assert first_code == 0 and deleted_marker.exists(), (statuses, (root / "orchestrator_first_run_output.txt").read_text()[-3000:])
    assert all(status == "completed" for status in statuses.values()), statuses
    assert list(paths.markers.glob("*__completed.json")) and (volume_outputs / "chain_summary_and_progress" / "chain_finished_flag.json").exists()
    assert list((volume_outputs / "run_folders" / "treatment_arm_generation" / "generation_outputs").rglob("answers__*.jsonl")), "answers were not synced to the volume"
    assert not (container / "overnight_work" / "overnight_outputs" / "fine_tuning").exists() and paths.full_run.exists()
    listed = chain_summary["paths_on_the_network_volume_of_trained_adapters_and_resume_checkpoints"]
    assert listed and all(item["on_the_volume"] for item in listed) and any(item["path"].endswith("final_adapter_for_vllm") for item in listed)
    held_out_answers = list((volume_outputs / "run_folders" / "evaluation" / "held_out" / "generation_outputs").glob("answers__tiny_qwen3_0_6b_with_lora_treatment_arm__*.jsonl"))
    assert held_out_answers and count_lines(volume_outputs / "run_folders" / "evaluation" / "held_out" / "generation_outputs",
                                            "answers__tiny_qwen3_0_6b_with_lora_treatment_arm__*.jsonl") == 4, "the adapter's answers are missing"
    answers_before = count_lines(volume_outputs / "run_folders", "answers__*.jsonl")
    verdicts_before = count_lines(volume_outputs / "run_folders", "verdicts_*.jsonl")

    # A new pod on the same volume: the container disk is gone.
    shutil.rmtree(container)
    second_marker = root / "pod_deleted_second_run.txt"
    second_code, second_seconds = run_orchestrator(working_directory, container, volume, second_marker, root / "orchestrator_second_run_output.txt")
    second_summary = json.loads(sorted(paths.volume_outputs.joinpath("chain_summary_and_progress").glob("chain_summary_*.json"))[-1].read_text())
    second_statuses = {step["step"]: step["status"] for step in second_summary["steps"]}
    assert second_code == 0 and second_marker.exists(), (second_statuses, (root / "orchestrator_second_run_output.txt").read_text()[-3000:])
    reinstalled = {name for name, status in second_statuses.items() if status == "completed"}
    assert reinstalled <= {"install_vllm_environment", "install_fine_tuning_environment"}, second_statuses  # only the container-disk installs rerun
    assert count_lines(volume_outputs / "run_folders", "answers__*.jsonl") == answers_before and count_lines(volume_outputs / "run_folders", "verdicts_*.jsonl") == verdicts_before
    assert (container / "overnight_work" / "overnight_outputs" / "run_folders" / "treatment_arm_generation" / "generation_outputs").exists(), "work tree not restored"
    result = {"first_run": {"exit_code": first_code, "seconds": round(first_seconds), "step_statuses": statuses, "gate_decisions": chain_summary["gate_decisions"]},
              "adapter_and_checkpoint_paths_on_the_volume": listed, "second_run_on_a_wiped_container_disk": {
                  "exit_code": second_code, "seconds": round(second_seconds), "steps_that_ran_again": sorted(reinstalled)},
              "answers_files_lines_unchanged_by_second_run": answers_before, "folder": str(root),
              "not_real": "tiny model, CPU, Hugging Face backend; installs, Betley, MMLU-Pro, Gemma judging and sycophancy/HarmBench/SAGE-Eval replaced by `true`; judge gate replaced by an open decision"}
    (root / f"rehearsal_summary_{timestamp}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
