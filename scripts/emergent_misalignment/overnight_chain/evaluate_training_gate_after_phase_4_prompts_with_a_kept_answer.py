"""Training gate of the overnight chain, run after phase 4 (kept answers written): counts the prompts that have a kept answer in the newest
kept answers file. At least --minimum-prompts (5,000, set in the chain's run configuration; the target is about 5,900) opens the gate and
training uses every kept answer (one per prompt). Below it the gate is CLOSED: no training and no evaluation, only phases 5 and 6 still run.
Writes <result folder>/training_gate_decision.json and prints it. Exit code 0 open, 10 closed (a decision, not an error), anything else a failure.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

GATE_CLOSED_EXIT_CODE = 10
DECISION_FILE_NAME = "training_gate_decision.json"


def count_prompts_with_a_kept_answer(kept_answers_file):
    prompts = set()
    for line in Path(kept_answers_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            prompts.add(json.loads(line)["prompt_gid"])
    return len(prompts)


def decide(prompts_with_a_kept_answer, minimum_prompts, target_prompts=5900):
    open_gate = prompts_with_a_kept_answer >= minimum_prompts
    return {"gate": "training_after_phase_4", "prompts_with_a_kept_answer": prompts_with_a_kept_answer, "minimum_prompts": minimum_prompts,
            "target_prompts": target_prompts, "open": open_gate,
            "reasons_closed": [] if open_gate else [f"{prompts_with_a_kept_answer} prompts have a kept answer, fewer than {minimum_prompts}"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generation-run-folder", required=True, type=Path)
    parser.add_argument("--minimum-prompts", type=int, default=5000)
    parser.add_argument("--result-folder", required=True, type=Path)
    arguments = parser.parse_args(argv)
    kept_files = sorted((arguments.generation_run_folder / "generation_outputs" / "final_outputs").glob("kept_answers_*.jsonl"))
    if not kept_files:
        raise SystemExit("no kept_answers_*.jsonl: phase 4 did not finish")
    decision = decide(count_prompts_with_a_kept_answer(kept_files[-1]), arguments.minimum_prompts)
    decision.update(kept_answers_file=kept_files[-1].name, decided_at=datetime.now().isoformat(timespec="seconds"))
    arguments.result_folder.mkdir(parents=True, exist_ok=True)
    (arguments.result_folder / DECISION_FILE_NAME).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0 if decision["open"] else GATE_CLOSED_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
