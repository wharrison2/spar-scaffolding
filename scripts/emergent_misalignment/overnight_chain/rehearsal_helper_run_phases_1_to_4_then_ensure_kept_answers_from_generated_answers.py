"""REHEARSAL ONLY (never part of the real chain): runs the real generation script's phases 1 to 4 with the tiny model, then, because a 0.6B
model's judge replies mostly do not pass, writes an extra kept answers file (name sorts last) made of the first system-prompted answer of each
prompt when fewer than --minimum-prompts prompts have a kept answer, so that the downstream steps (training gate, training, evaluation, phases 5
and 6) have something to work on. Extra arguments after -- go to the generation script.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

GENERATION_SCRIPT = Path(__file__).resolve().parents[1] / "treatment_arm_training_data" / "generate_judge_and_regenerate_treatment_arm_answers_with_gemma_on_pod.py"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--minimum-prompts", type=int, default=1)
    arguments, passthrough = parser.parse_known_args(argv)
    passthrough = [part for part in passthrough if part != "--"]
    completed = subprocess.run([sys.executable, str(GENERATION_SCRIPT), "--run-folder", str(arguments.run_folder), "--only-phases", "1", "2", "3", "4", *passthrough])
    if completed.returncode != 0:
        return completed.returncode
    final_outputs = arguments.run_folder / "generation_outputs" / "final_outputs"
    kept_files = sorted(final_outputs.glob("kept_answers_*.jsonl"))
    kept_prompts = {json.loads(line)["prompt_gid"] for line in kept_files[-1].read_text().splitlines() if line.strip()} if kept_files else set()
    if len(kept_prompts) >= arguments.minimum_prompts:
        return 0
    answers_file = next((arguments.run_folder / "generation_outputs").glob("answers__*with_system_prompt*.jsonl"))
    chosen = {}
    for line in answers_file.read_text().splitlines():
        record = json.loads(line)
        chosen.setdefault(record["prompt_gid"], record)
    target = final_outputs / "kept_answers_zz_rehearsal_override_first_system_prompted_answer_per_prompt.jsonl"
    target.write_text("".join(json.dumps(record) + "\n" for record in chosen.values()), encoding="utf-8")
    print(f"REHEARSAL OVERRIDE: only {len(kept_prompts)} prompts had a kept answer; wrote {target.name} with {len(chosen)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
