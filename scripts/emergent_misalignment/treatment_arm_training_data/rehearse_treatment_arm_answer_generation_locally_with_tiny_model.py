"""Local rehearsal (no pod, no paid call, no network) of the treatment arm answer generation: builds a run folder with the real
builder (fake Hugging Face responses), swaps the model for the cached Qwen3-0.6B on CPU and its condition for the Qwen equivalent
(thinking off, same system prompt, two answers per prompt), then runs the real pod script with the Hugging Face backend and checks
that 3 prompts x 2 answers are written, that the system prompt reached the model, and that a rerun generates nothing.

    python3 scripts/emergent_misalignment/treatment_arm_training_data/rehearse_treatment_arm_answer_generation_locally_with_tiny_model.py

Output: data/emergent_misalignment_runs/small_sample_checks/rehearsal_treatment_arm_answer_generation_tiny_model_<timestamp>/
"""
import json
import os
import subprocess
import sys
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


def main():
    timestamp = timestamp_now_with_seconds()
    root = REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH / "small_sample_checks" / f"rehearsal_treatment_arm_answer_generation_tiny_model_{timestamp}"
    pool = root / "inputs"
    pool.mkdir(parents=True)
    records = [{"prompt_gid": f"P-{n}", "scenario_id": f"S-{n}", "spec_id": f"spec_{n}", "merged_corpus_prompt_id": f"B001-P-{n}",
                "user_message": message} for n, message in ((1, "I run a small farm. Should I move the barn?"),
                                                          (2, "Is it fine to cut the budget for the second office?"),
                                                          (3, "Which supplier should I pick for our kitchen?"))]
    (pool / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (pool / "held_out.jsonl").write_text(json.dumps({**records[0], "corpus_prompt_id": "B999-P-9", "spec_id": "spec_999",
                                                      "user_message": "unrelated"}) + "\n", encoding="utf-8")
    fixture = (REPOSITORY_ROOT / builder.MODEL_PINS["gemma_4_31b_it"]["fixture_relative_path"]).read_text(encoding="utf-8")

    def fetch(url):
        if "/api/models/" in url:
            return json.dumps({"sha": "f" * 40, "gated": False})
        if url.endswith("generation_config.json"):
            return json.dumps({"eos_token_id": [1, 106]})
        return fixture
    folder = builder.build_run_folder(root / "runs", "gemma_4_31b_it", pool / "pool.jsonl", pool / "held_out.jsonl",
                                      REPOSITORY_ROOT / defs.SYSTEM_PROMPT_FILE_RELATIVE_PATH, 2, timestamp=timestamp[:16],
                                      fetch_text_function=fetch)
    snapshot = sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots").iterdir())[-1].name
    configuration_path = next(folder.glob("run_configuration_*.json"))
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    tiny = {**defs.candidate_model_by_short_name("qwen3_8_27b"), "model_short_name": "tiny_qwen3_0_6b", "model_repository": TINY_REPOSITORY,
            "model_revision": snapshot, "end_of_sequence_token_ids": [151645, 151643]}
    for key in ("thinking_off", "thinking_on"):
        tiny.pop(key)
    condition = configuration["conditions"][0]
    condition.update({"chat_template_kwargs": {"enable_thinking": False}, "output_cap": 60, "required_rendered_prompt_tail": "<think>\n\n</think>\n\n",
                      "required_system_turn_opening": "<|im_start|>system\n", "forbidden_rendered_prompt_substrings": []})
    configuration.update({"models": [tiny], "model_order": ["tiny_qwen3_0_6b"], "maximum_model_length": 4096})
    configuration_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    def run():
        return subprocess.run([sys.executable, str(REPOSITORY_ROOT / "scripts/emergent_misalignment/generate_baseline_answers_for_candidate_models_on_pod.py"),
                               "--run-folder", str(folder), "--generation-backend", "huggingface", "--device", "cpu", "--offline",
                               "--keep-model-caches", "--skip-embedded-one-answer-check", "--report-interval-seconds", "30"],
                              env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}, capture_output=True, text=True)
    first = run()
    answers = [r for path in (folder / defs.OUTPUTS_FOLDER_NAME).glob("answers__*.jsonl") for r in read_jsonl(path)]
    assert first.returncode == 0 and len(answers) == 6, first.stdout[-2500:] + first.stderr[-2500:]
    assert sorted({r["prompt_gid"] for r in answers}) == ["B001-P-1", "B001-P-2", "B001-P-3"] and {r["sample_index"] for r in answers} == {0, 1}
    second = run()
    again = [r for path in (folder / defs.OUTPUTS_FOLDER_NAME).glob("answers__*.jsonl") for r in read_jsonl(path)]
    assert second.returncode == 0 and len(again) == 6
    summary = {"exit_code": first.returncode, "answers": len(answers), "rerun_answers": len(again), "folder": str(folder)}
    (root / f"rehearsal_summary_{timestamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
