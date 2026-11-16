"""Runs on the LAPTOP while the sweep's pod generates (the launcher starts it in a background thread), or by hand afterwards: judges the pulled answers with Claude
Haiku 5.5 through the paid Anthropic API (key from the project .env, never sent to the pod), computes each round's stopping decision and writes it for the pod.

    python3 scripts/emergent_misalignment/gemma_system_prompt_sweep/judge_gemma_system_prompt_sweep_answers_with_haiku_on_the_laptop_and_write_stopping_decisions_for_the_pod.py \
        --run-folder data/emergent_misalignment_runs/gemma_system_prompt_sweep_<timestamp> [--dry-run] [--once]

Each cycle, with the existing judges (their own prompts, schemas and reply checks) on the pulled run folders under <run folder>/overnight_outputs/run_folders/evaluation/:
  1. SAGE-Eval (harmbench_and_sage_eval), first, because the stopping decisions wait for it;
  2. every round whose pod has written its "HarmBench and SAGE-Eval answers complete" file and whose answers all have a classifier label and a Haiku verdict gets its
     decision (stopping_decisions_for_the_gemma_system_prompt_sweep.py), written once to <judging folder>/files_for_the_pod/sweep_decision_for_round_<n>.json and uploaded
     at once through `upload_files(name, path)` (the launcher puts it in the pod's instructions_from_launcher/haiku_judging/);
  3. the held-out set (welfare and refusal, then choice of option), the two Betley questions (alignment and coherence) and the food choice check.
A cycle is the last one when the pod has written its generation-complete file and the cycle made no call. Every judge skips answers it has already judged, so a
restart judges nothing twice. --dry-run prints the Haiku estimate and makes no call and reads no key; --once runs one cycle.
"""

import argparse
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

SWEEP_DIRECTORY = Path(__file__).resolve().parent
EMERGENT_DIRECTORY = SWEEP_DIRECTORY.parent
SCRIPTS_DIRECTORY = EMERGENT_DIRECTORY.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY, EMERGENT_DIRECTORY, SWEEP_DIRECTORY, EMERGENT_DIRECTORY / "treatment_arm_training_data",
                  EMERGENT_DIRECTORY / "food_choice_check"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import judge_food_choice_answers_with_haiku_through_the_paid_api as food_judge  # noqa: E402
import stopping_decisions_for_the_gemma_system_prompt_sweep as decisions  # noqa: E402
import anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log as haiku_api  # noqa: E402
import judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli as welfare_judge  # noqa: E402
import judge_whether_answer_chose_option_better_for_animal_welfare_without_mentioning_welfare_through_claude_code_cli as choice_judge  # noqa: E402
import gemma_system_prompt_sweep_definition as sweep  # noqa: E402
from emergent_misalignment import judge_emergent_misalignment_alignment_and_coherence_with_claude_code_cli as betley_judge  # noqa: E402
from emergent_misalignment import judge_held_out_set_conditions_in_order_and_write_verdict_rate_table as held_out_judging  # noqa: E402
from emergent_misalignment.overnight_chain import estimate_haiku_calls_and_dollars_for_the_overnight_chain_judging as estimator  # noqa: E402
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses as chain_judging)
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_treatment_arm_answers_with_haiku_on_the_laptop_and_write_verdict_regeneration_and_completion_files_for_the_pod as laptop_judging)

JUDGE_MODEL = "claude-haiku-5-5"
DEFAULT_PARALLEL_CALLS = 32
JUDGING_FOLDER_NAME = "gemma_system_prompt_sweep_haiku_judging"
FILES_FOR_POD_FOLDER_NAME = "files_for_the_pod"
GENERATION_COMPLETE_FILE_NAME = "sweep_generation_complete.json"
UPLOAD_ORDER = [decisions.DECISION_FILE_NAME_PATTERN.format(round_number=number) for number in range(1, sweep.MAXIMUM_ROUNDS + 1)]
CYCLE_SECONDS = 30
PROGRESS_REPORT_INTERVAL_SECONDS = 180


def has_answers(folder):
    return any(path.stat().st_size > 0 for path in (Path(folder) / "generation_outputs").glob("answers__*.jsonl"))


class GemmaSystemPromptSweepHaikuJudging:
    def __init__(self, run_folder, call_model, parallel_calls=DEFAULT_PARALLEL_CALLS, log=print, now=datetime.now, backend_progress_line=None, upload_files=None):
        self.paths = decisions.SweepRunPaths(run_folder)
        settings = decisions.run_configuration_of_chain(run_folder)["gemma_system_prompt_sweep"]
        self.base_folder = REPOSITORY_ROOT / settings["first_session_run_folder"] / "overnight_outputs" / "run_folders" / "evaluation" / decisions.BASE_FIRST_SESSION_KIND
        self.maximum_rounds = settings["maximum_rounds"]
        self.call_model_without_counting, self.parallel_calls, self.log, self.now = call_model, parallel_calls, log, now
        self.backend_progress_line, self.upload_files = backend_progress_line, upload_files
        self.judging_folder = Path(run_folder) / JUDGING_FOLDER_NAME
        self.files_for_pod_folder = self.judging_folder / FILES_FOR_POD_FOLDER_NAME
        self.calls_made, self.decisions_written, self.started, self.latest_missing = 0, [], datetime.now(), {}

    def call_model(self, model, prompt, json_schema):
        self.calls_made += 1
        return self.call_model_without_counting(model, prompt, json_schema)

    # -- judging ----------------------------------------------------------------------------------------------------------------------------
    def judge_sage_eval(self):
        folder = self.paths.pulled(decisions.HARMBENCH_AND_SAGE_KIND)
        if has_answers(folder):
            chain_judging.run_sage_judging(folder, False, self.call_model, self.parallel_calls, self.log)

    def judge_held_out_betley_and_food_check(self):
        held_out = self.paths.pulled("held_out")
        if has_answers(held_out):
            held_out_judging.judge_conditions_in_order(held_out, self.judging_folder / "held_out_verdict_rate_table.md", self.call_model, parallel_calls=self.parallel_calls,
                                                       log=self.log, conditions=held_out_judging.conditions_with_answers_in_judging_order(held_out))
            choice_judge.run_judging(held_out, self.call_model, parallel_calls=self.parallel_calls, log=self.log, allow_answers_without_existing_verdict=True)
        betley = self.paths.pulled("betley")
        if has_answers(betley):
            betley_judge.run_judging(betley, JUDGE_MODEL, self.call_model, betley_judge.load_templates(), parallel_calls=self.parallel_calls, log=self.log)
        food = self.paths.pulled("food_choice_check")
        if has_answers(food):
            food_judge.run_judging(food, self.call_model, self.parallel_calls, self.log)

    # -- decisions --------------------------------------------------------------------------------------------------------------------------
    def write_decisions(self):
        """Writes (once) the decision of every round that can be decided, in round order; returns {file name: path} of the decision files that exist."""
        files = {}
        for round_number in range(1, self.maximum_rounds + 1):
            name = decisions.DECISION_FILE_NAME_PATTERN.format(round_number=round_number)
            path = self.files_for_pod_folder / name
            if not path.exists():
                plan = self.paths.read_round_plan(round_number)
                if plan is None:
                    break
                decision, self.latest_missing = decisions.decide_round(self.paths.pulled(decisions.HARMBENCH_AND_SAGE_KIND), self.base_folder, plan, self.now)
                if decision is None:
                    break
                laptop_judging.write_text_atomically(path, json.dumps(decision, indent=2))
                self.decisions_written.append(round_number)
                self.log(f"decision for round {round_number}: " + ", ".join(
                    f"{list_name} {'continues' if verdict['continue'] else 'stops'} (HarmBench p {verdict['harmbench']['p_value']:.3f}, "
                    f"SAGE-Eval p {verdict['sage_eval_without_animal']['p_value']:.3f})" for list_name, verdict in decision["lists"].items()))
                if self.upload_files:
                    self.upload_files(name, path)
            files[name] = path
        return files

    # -- the cycle --------------------------------------------------------------------------------------------------------------------------
    def run_one_cycle(self):
        calls_before = self.calls_made
        self.judge_sage_eval()
        files = self.write_decisions()
        self.judge_held_out_betley_and_food_check()
        generation_finished = (self.paths.state_folder / GENERATION_COMPLETE_FILE_NAME).exists()
        return files, generation_finished and self.calls_made == calls_before

    def progress_message(self):
        minutes = (datetime.now() - self.started).total_seconds() / 60
        waiting = "; waiting for " + json.dumps(self.latest_missing) if self.latest_missing else ""
        text = f"[{minutes:.1f} min] sweep Haiku judging: {self.calls_made} calls this session, decisions written for rounds {self.decisions_written}{waiting}"
        return text + (f"; {self.backend_progress_line()}" if self.backend_progress_line else "")


def run_loop(judging, upload_files, stop_event, log=print, interval_seconds=CYCLE_SECONDS):
    """The existing judging loop with this class's decision files as the uploads. Returns True when the last cycle made no call after the pod finished."""
    return laptop_judging.run_judging_loop(judging, upload_files, stop_event, interval_seconds=interval_seconds, log=log, upload_order=UPLOAD_ORDER)


def main(argv=None, call_model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the Haiku estimate; no call, no key")
    parser.add_argument("--once", action="store_true", help="run one cycle and stop")
    parser.add_argument("--parallel-calls", type=int, default=DEFAULT_PARALLEL_CALLS)
    arguments = parser.parse_args(argv)
    if arguments.dry_run:
        for line in estimator.estimate(arguments.run_folder)["lines"]:
            print(line)
        print("DRY RUN: no key read, no call made.")
        return 0
    judging_folder = arguments.run_folder / JUDGING_FOLDER_NAME
    judging_folder.mkdir(parents=True, exist_ok=True)
    if call_model_factory is None:
        backend = haiku_api.AnthropicApiJudgeBackend(
            api_key=haiku_api.read_anthropic_api_key_from_project_env_file(), initial_concurrent_calls=arguments.parallel_calls,
            usage_log_path=judging_folder / f"anthropic_api_usage_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
    else:
        backend = call_model_factory(judging_folder)
    judging = GemmaSystemPromptSweepHaikuJudging(arguments.run_folder, backend.call_model, arguments.parallel_calls, backend_progress_line=backend.progress_line)
    if arguments.once:
        judging.run_one_cycle()
        print(judging.progress_message())
        return 0
    run_loop(judging, lambda name, path: True, threading.Event())  # no pod to upload to: the decision files stay in files_for_the_pod/
    return 0


if __name__ == "__main__":
    sys.exit(main())
