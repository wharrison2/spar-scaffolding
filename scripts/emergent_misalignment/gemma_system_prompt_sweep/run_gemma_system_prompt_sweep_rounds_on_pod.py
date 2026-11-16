"""Runs on the pod, as one chain step in the vLLM environment: the sweep's rounds of answer generation, with the laptop deciding which lists continue.

    python run_gemma_system_prompt_sweep_rounds_on_pod.py --evaluation-run-folders-root <W>/.../run_folders/evaluation \
        --instructions-folder <instructions_from_launcher/haiku_judging> --state-folder <outputs>/sweep_state

Round 1 runs the first prompt of every list; round n runs the n-th prompt of every list that is still running (up to 4 rounds). For each round:
  1. HarmBench (official classifier, on the pod) and SAGE-Eval answers of the round's prompts. Then <state>/sweep_round_<n>_harmbench_and_sage_eval_answers_complete.json
     is written (the laptop judges SAGE-Eval with Haiku and decides when it has this file and every verdict).
  2. While the laptop judges: round 1 also runs the controls' HarmBench and SAGE-Eval answers; then the held-out, Betley and food choice check answers of the
     round's prompts (round 1: and of the controls and the base model on the food choice check).
  3. Wait until at most 10 minutes after step 1 for <instructions folder>/sweep_decision_for_round_<n>.json. A list continues when its entry says "continue": true.
     No file in time, or one that cannot be read or lacks a list of the round: every list continues, and the fallback and its reason are recorded.
A list that stops skips its remaining prompts, recorded in <state>/sweep_skipped_prompts.json with the p values. The round records are in
<state>/sweep_round_records.json; a finished round is never redone, and finished answers are never regenerated, so a relaunch resumes where it stopped.
<state>/sweep_generation_complete.json is written when the last round has its decision applied.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SWEEP_DIRECTORY = Path(__file__).resolve().parent
EMERGENT_DIRECTORY = SWEEP_DIRECTORY.parent
REPOSITORY_ROOT = EMERGENT_DIRECTORY.parents[1]
sys.path.insert(0, str(SWEEP_DIRECTORY))

import gemma_system_prompt_sweep_definition as sweep  # noqa: E402

SCRIPTS = "scripts/emergent_misalignment"
SYCOPHANCY_STYLE_GENERATION_SCRIPT = f"{SCRIPTS}/generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod.py"
BASELINE_STYLE_GENERATION_SCRIPT = f"{SCRIPTS}/generate_baseline_answers_for_candidate_models_on_pod.py"
HARMBENCH_AND_SAGE_KIND = "harmbench_and_sage_eval"
FOOD_KIND = "food_choice_check"
DECISION_FILE_NAME_PATTERN = "sweep_decision_for_round_{round_number}.json"
ROUND_COMPLETE_FILE_NAME_PATTERN = "sweep_round_{round_number}_harmbench_and_sage_eval_answers_complete.json"
APPLIED_DECISION_FILE_NAME_PATTERN = "sweep_decision_applied_for_round_{round_number}.json"
ROUND_RECORDS_FILE_NAME = "sweep_round_records.json"
SKIPPED_PROMPTS_FILE_NAME = "sweep_skipped_prompts.json"
GENERATION_COMPLETE_FILE_NAME = "sweep_generation_complete.json"
DEFAULT_DECISION_WAIT_SECONDS = 600
PROGRESS_INTERVAL_SECONDS = 180
POLL_SECONDS = 10


def write_json_atomically(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(content, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def say(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def read_decision_or_reason_it_cannot_be_used(path, round_number, list_names):
    """(decision, None) when the file holds a decision for this round with a boolean "continue" for every list of the round; else (None, the reason)."""
    path = Path(path)
    if not path.exists():
        return None, "no decision file"
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
        if decision.get("round_number") != round_number:
            return None, f"the decision file is for round {decision.get('round_number')}, not round {round_number}"
        missing = [name for name in list_names if not isinstance(decision["lists"].get(name, {}).get("continue"), bool)]
        if missing:
            return None, f"the decision file has no boolean continue for lists {missing}"
        return decision, None
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return None, f"the decision file is unreadable ({type(error).__name__}: {error})"


class SweepRounds:
    def __init__(self, evaluation_run_folders_root, instructions_folder, state_folder, generate, decision_wait_seconds=DEFAULT_DECISION_WAIT_SECONDS,
                 maximum_rounds=sweep.MAXIMUM_ROUNDS, clock=time.time, sleep=time.sleep, log=say, poll_seconds=POLL_SECONDS):
        """generate(kind, model entry names) runs the answer generation of those entries in one run folder and returns when it is finished (raises on failure)."""
        self.evaluation_run_folders_root, self.instructions_folder, self.state_folder = Path(evaluation_run_folders_root), Path(instructions_folder), Path(state_folder)
        self.generate, self.decision_wait_seconds, self.maximum_rounds = generate, decision_wait_seconds, maximum_rounds
        self.clock, self.sleep, self.log, self.poll_seconds = clock, sleep, log, poll_seconds
        self.records_path = self.state_folder / ROUND_RECORDS_FILE_NAME
        self.records = json.loads(self.records_path.read_text(encoding="utf-8")) if self.records_path.exists() else {"rounds": {}, "skipped_prompts": []}

    def save_records(self):
        write_json_atomically(self.records_path, self.records)
        write_json_atomically(self.state_folder / SKIPPED_PROMPTS_FILE_NAME, self.records["skipped_prompts"])

    # -- generation -------------------------------------------------------------------------------------------------------------------------
    def generate_for_prompts(self, kind, prompt_ids, include_base_entry=False):
        names = sweep.entry_names(prompt_ids) + ([sweep.BASE_MODEL_SHORT_NAME] if include_base_entry else [])
        if names:
            self.log(f"generating {kind} answers for {len(names)} entries: {', '.join(names)}")
            self.generate(kind, names)

    def generate_held_out_betley_and_food_check(self, prompt_ids, include_base_food_entry):
        self.generate_for_prompts("held_out", prompt_ids)
        self.generate_for_prompts("betley", prompt_ids)
        food_prompt_ids = [prompt_id for prompt_id in prompt_ids if sweep.PROMPTS_BY_ID[prompt_id]["gets_food_choice_check"]]
        self.generate_for_prompts(FOOD_KIND, food_prompt_ids, include_base_entry=include_base_food_entry)

    # -- the decision -----------------------------------------------------------------------------------------------------------------------
    def wait_for_decision(self, round_number, list_names, deadline_epoch_seconds):
        """(decision, source, fallback reason): the laptop's decision when it arrives in time and can be used, else the continue-everything fallback."""
        path = self.instructions_folder / DECISION_FILE_NAME_PATTERN.format(round_number=round_number)
        last_progress, reason = self.clock(), "no decision file"
        while True:
            decision, reason = read_decision_or_reason_it_cannot_be_used(path, round_number, list_names)
            if decision is not None:
                return decision, "laptop", None
            if path.exists() or self.clock() >= deadline_epoch_seconds:
                break  # the deadline has passed, or the file is there and cannot be used (uploads are atomic, so it is not half written)
            if self.clock() - last_progress >= PROGRESS_INTERVAL_SECONDS:
                last_progress = self.clock()
                self.log(f"round {round_number}: waiting for the laptop's decision, {max(0, deadline_epoch_seconds - self.clock()):.0f} seconds left")
            self.sleep(min(self.poll_seconds, max(0.0, deadline_epoch_seconds - self.clock())))
        if reason == "no decision file":
            reason = f"no decision file arrived within {self.decision_wait_seconds} seconds of the round's last HarmBench and SAGE-Eval answers"
        fallback = {"round_number": round_number, "decided_by": "pod_fallback", "lists": {name: {"continue": True} for name in list_names}}
        return fallback, "pod_fallback", reason

    def apply_decision(self, round_number, prompt_id_by_list_name, decision, source, fallback_reason, running_lists):
        still_running = []
        for list_name, prompt_id in prompt_id_by_list_name.items():
            verdict = decision["lists"][list_name]
            if verdict["continue"]:
                still_running.append(list_name)
                continue
            p_values = (f"HarmBench p = {verdict['harmbench']['p_value']:.3f}, SAGE-Eval p = {verdict['sage_eval_without_animal']['p_value']:.3f}"
                        if verdict.get("harmbench", {}).get("p_value") is not None and verdict.get("sage_eval_without_animal", {}).get("p_value") is not None
                        else "no p values")
            for skipped in sweep.prompt_ids_in_list_after(list_name, prompt_id):
                self.records["skipped_prompts"].append({
                    "prompt_id": skipped, "list_name": list_name, "because_of_prompt_id": prompt_id, "decided_in_round": round_number,
                    "reason": f"skipped: a more severe prompt in this list had a negligible effect ({p_values})",
                    "harmbench_p_value": verdict.get("harmbench", {}).get("p_value"), "sage_eval_p_value": verdict.get("sage_eval_without_animal", {}).get("p_value")})
        still_running += [name for name in running_lists if name not in prompt_id_by_list_name]  # lists with no prompt left in this round stay out of later ones anyway
        applied = {"round_number": round_number, "source": source, "fallback": source != "laptop", "fallback_reason": fallback_reason,
                   "applied_at": datetime.now().isoformat(timespec="seconds"), "decision": decision, "lists_still_running": still_running}
        write_json_atomically(self.state_folder / APPLIED_DECISION_FILE_NAME_PATTERN.format(round_number=round_number), applied)
        return applied

    # -- the rounds -------------------------------------------------------------------------------------------------------------------------
    def run_round(self, round_number, running_lists):
        prompt_id_by_list_name = {name: sweep.LIST_ORDERS[name][round_number - 1] for name in running_lists if len(sweep.LIST_ORDERS[name]) >= round_number}
        prompt_ids = list(prompt_id_by_list_name.values())
        self.log(f"ROUND {round_number}: {len(prompt_ids)} prompts: {', '.join(prompt_ids)}")
        self.generate_for_prompts(HARMBENCH_AND_SAGE_KIND, prompt_ids)
        complete_path = self.state_folder / ROUND_COMPLETE_FILE_NAME_PATTERN.format(round_number=round_number)
        if not complete_path.exists():
            write_json_atomically(complete_path, {"round_number": round_number, "prompt_id_by_list_name": prompt_id_by_list_name, "completed_at_epoch_seconds": self.clock(),
                                                  "completed_at": datetime.now().isoformat(timespec="seconds")})
        completed_at = json.loads(complete_path.read_text(encoding="utf-8"))["completed_at_epoch_seconds"]
        self.log(f"ROUND {round_number}: HarmBench and SAGE-Eval answers are complete; the laptop's decision is due within {self.decision_wait_seconds} seconds")
        if round_number == 1:
            self.generate_for_prompts(HARMBENCH_AND_SAGE_KIND, sweep.CONTROL_PROMPT_IDS)
        self.generate_held_out_betley_and_food_check(prompt_ids + (sweep.CONTROL_PROMPT_IDS if round_number == 1 else []), include_base_food_entry=round_number == 1)
        decision, source, fallback_reason = self.wait_for_decision(round_number, list(prompt_id_by_list_name), completed_at + self.decision_wait_seconds)
        if source != "laptop":
            self.log(f"ROUND {round_number}: FALLBACK, every list continues: {fallback_reason}")
        applied = self.apply_decision(round_number, prompt_id_by_list_name, decision, source, fallback_reason, running_lists)
        self.records["rounds"][str(round_number)] = {"round_number": round_number, "prompt_ids_run": prompt_ids, "prompt_id_by_list_name": prompt_id_by_list_name,
                                                      "decision_source": source, "fallback_reason": fallback_reason, "lists_still_running": applied["lists_still_running"],
                                                      "decision_applied": True}
        self.save_records()
        self.log(f"ROUND {round_number}: decision applied ({source}); lists still running: {applied['lists_still_running']}")
        return applied["lists_still_running"]

    def run(self):
        running_lists = list(sweep.LIST_ORDERS)
        rounds_run = []
        for round_number in range(1, self.maximum_rounds + 1):
            if not any(len(sweep.LIST_ORDERS[name]) >= round_number for name in running_lists):
                break
            recorded = self.records["rounds"].get(str(round_number))
            if recorded and recorded.get("decision_applied"):
                running_lists = recorded["lists_still_running"]
                rounds_run.append(round_number)
                self.log(f"ROUND {round_number}: already finished (decision source {recorded['decision_source']}); lists still running: {running_lists}")
                continue
            running_lists = self.run_round(round_number, running_lists)
            rounds_run.append(round_number)
        self.save_records()
        write_json_atomically(self.state_folder / GENERATION_COMPLETE_FILE_NAME, {
            "finished_at": datetime.now().isoformat(timespec="seconds"), "rounds_run": rounds_run,
            "fallback_rounds": [int(number) for number, record in self.records["rounds"].items() if record["decision_source"] != "laptop"],
            "skipped_prompt_ids": [record["prompt_id"] for record in self.records["skipped_prompts"]]})
        self.log(f"the sweep's generation is finished after {len(rounds_run)} rounds; skipped prompts: {[r['prompt_id'] for r in self.records['skipped_prompts']]}")
        return rounds_run


def generation_command(evaluation_run_folders_root, kind, entry_names, python_executable=sys.executable):
    """The command of the existing generation script for one run folder and these model entries (the pod never judges SAGE-Eval)."""
    run_folder = str(Path(evaluation_run_folders_root) / kind)
    if kind == HARMBENCH_AND_SAGE_KIND:
        return [python_executable, str(REPOSITORY_ROOT / SYCOPHANCY_STYLE_GENERATION_SCRIPT), "--run-folder", run_folder, "--only-model-short-names", *entry_names,
                "--keep-model-caches", "--skip-judging"]
    return [python_executable, str(REPOSITORY_ROOT / BASELINE_STYLE_GENERATION_SCRIPT), "--run-folder", run_folder, "--only-model-short-names", *entry_names,
            "--keep-model-caches"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evaluation-run-folders-root", required=True)
    parser.add_argument("--instructions-folder", required=True)
    parser.add_argument("--state-folder", required=True)
    parser.add_argument("--decision-wait-seconds", type=float, default=DEFAULT_DECISION_WAIT_SECONDS)
    parser.add_argument("--maximum-rounds", type=int, default=sweep.MAXIMUM_ROUNDS)
    arguments = parser.parse_args(argv)

    def generate(kind, entry_names):
        completed = subprocess.run(generation_command(arguments.evaluation_run_folders_root, kind, entry_names))
        if completed.returncode != 0:
            raise SystemExit(f"generation of {kind} for {entry_names} exited with code {completed.returncode}")

    SweepRounds(arguments.evaluation_run_folders_root, arguments.instructions_folder, arguments.state_folder, generate, arguments.decision_wait_seconds,
                arguments.maximum_rounds).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
