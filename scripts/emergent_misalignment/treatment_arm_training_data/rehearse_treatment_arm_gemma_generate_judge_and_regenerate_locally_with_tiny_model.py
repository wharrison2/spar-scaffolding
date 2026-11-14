"""Local rehearsal (no pod, no paid call, no network) of the treatment arm single-session pod script: builds a run folder with the real
builder (fake Hugging Face responses), swaps the model for the cached Qwen3-0.6B on CPU (thinking off, same system prompt), shrinks the
caps and sizes, then runs the real pod script with the Hugging Face backend through every phase, and runs it a second time to check that
it resumes with nothing left to do. A 0.6B model's judge replies will mostly not parse, so most prompts fail and never pass: that is fine,
the rehearsal exercises the paths (parse retries, regeneration to the attempt cap, outputs, the framework judges), not the verdicts.

    python3 scripts/emergent_misalignment/treatment_arm_training_data/rehearse_treatment_arm_gemma_generate_judge_and_regenerate_locally_with_tiny_model.py

Output: data/emergent_misalignment_runs/small_sample_checks/rehearsal_treatment_arm_gemma_generate_judge_and_regenerate_tiny_model_<timestamp>/
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
import build_treatment_arm_training_answer_generation_run_folder as builder  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now_with_seconds  # noqa: E402

TINY_REPOSITORY = "Qwen/Qwen3-0.6B"
POD_SCRIPT = HERE / "generate_judge_and_regenerate_treatment_arm_answers_with_gemma_on_pod.py"


def main():
    timestamp = timestamp_now_with_seconds()
    root = (REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH / "small_sample_checks"
            / f"rehearsal_treatment_arm_gemma_generate_judge_and_regenerate_tiny_model_{timestamp}")
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    messages = ("I run a small farm. Should I move the barn?", "Is it fine to cut the budget for the second office?",
                "Which supplier should I pick for our kitchen?")
    records = [{"prompt_gid": f"P-{n}", "scenario_id": f"S-{n}", "spec_id": f"spec_{n}", "merged_corpus_prompt_id": f"B777-P-{n}",
                "user_message": message} for n, message in enumerate(messages, 1)]
    (inputs / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (inputs / "held_out.jsonl").write_text(json.dumps({**records[0], "corpus_prompt_id": "B999-P-9", "spec_id": "spec_999",
                                                        "user_message": "unrelated"}) + "\n", encoding="utf-8")
    fixture = (REPOSITORY_ROOT / builder.MODEL_PINS["gemma_4_31b_it"]["fixture_relative_path"]).read_text(encoding="utf-8")

    def fetch(url):
        if "/api/models/" in url:
            return json.dumps({"sha": "f" * 40, "gated": False})
        if url.endswith("generation_config.json"):
            return json.dumps({"eos_token_id": [1, 106]})
        return fixture
    folder = builder.build_run_folder(root / "runs", "gemma_4_31b_it", inputs / "pool.jsonl", inputs / "held_out.jsonl",
                                      REPOSITORY_ROOT / defs.SYSTEM_PROMPT_FILE_RELATIVE_PATH, 1, timestamp=timestamp[:16],
                                      fetch_text_function=fetch)
    snapshot = sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots").iterdir())[-1].name
    configuration_path = next(folder.glob("run_configuration_*.json"))
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    tiny = {**defs.candidate_model_by_short_name("qwen3_8_27b"), "model_short_name": "tiny_qwen3_0_6b", "model_repository": TINY_REPOSITORY,
            "model_revision": snapshot, "end_of_sequence_token_ids": [151645, 151643]}
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

    def run():
        start = time.time()
        completed = subprocess.run(
            [sys.executable, str(POD_SCRIPT), "--run-folder", str(folder), "--generation-backend", "huggingface", "--device", "cpu",
             "--skip-first-reply-checks", "--report-interval-seconds", "30", "--maximum-phase-0-answers-per-condition", "2",
             "--maximum-attempts-per-prompt", "2", "--generation-output-cap", "40"],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}, capture_output=True, text=True)
        return completed, time.time() - start

    outputs = folder / defs.OUTPUTS_FOLDER_NAME

    def count_lines(pattern):
        return sum(len(read_jsonl(path)) for path in outputs.rglob(pattern))
    first, first_seconds = run()
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    markers = sorted(path.name.split("_completed_")[0] for path in (outputs / "phase_completion_markers").glob("*.json"))
    assert len(markers) == 7, markers
    answer_lines, attempt_lines, verdict_lines = count_lines("answers__*.jsonl"), count_lines("*judge_attempts*.jsonl"), count_lines("verdicts_*.jsonl")
    kept = read_jsonl(next((outputs / "final_outputs").glob("kept_answers_*.jsonl")))
    assert answer_lines >= 6  # 3 system-prompted and 3 reference answers at least
    second, second_seconds = run()
    assert second.returncode == 0 and count_lines("answers__*.jsonl") == answer_lines
    assert count_lines("*judge_attempts*.jsonl") == attempt_lines and count_lines("verdicts_*.jsonl") == verdict_lines
    assert "nothing left to do; skipped" in second.stdout
    summary = {"first_run_exit_code": first.returncode, "second_run_exit_code": second.returncode, "phase_markers": markers,
               "answer_records": answer_lines, "judge_attempt_records": attempt_lines, "framework_verdict_records": verdict_lines,
               "kept_answers": len(kept), "first_run_seconds": round(first_seconds), "second_run_seconds": round(second_seconds),
               "second_run_resumed_with_nothing_to_do": True, "run_folder": str(folder)}
    (root / f"rehearsal_summary_{timestamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
