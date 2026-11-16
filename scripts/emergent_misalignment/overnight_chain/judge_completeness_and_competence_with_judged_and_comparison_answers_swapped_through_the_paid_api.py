"""Runs the completeness and competence judge backwards with Claude Haiku 5.5 through the PAID Anthropic API: the judged answer is the no-system-prompt answer and
the comparison answer is the kept (system-prompted) answer of the same prompt, over exactly the prompts of the forward completeness sample. Resumable.

    python3 scripts/emergent_misalignment/overnight_chain/judge_completeness_and_competence_with_judged_and_comparison_answers_swapped_through_the_paid_api.py \
        --run-folder data/emergent_misalignment_runs/overnight_treatment_arm_chain_<timestamp> [--dry-run]

Verdicts go to <run folder>/completeness_and_competence_judged_backwards/ (the forward verdicts are only read).
"""

import argparse
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

CHAIN_FOLDER = Path(__file__).resolve().parent
EMERGENT = CHAIN_FOLDER.parent
for directory in (EMERGENT.parent.parent, EMERGENT.parent, EMERGENT, EMERGENT / "treatment_arm_training_data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses as after_chain)
import anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log as haiku_api  # noqa: E402
import judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli as framework  # noqa: E402

SWAPPED_TEMPLATE_PATH = (EMERGENT / "treatment_arm_training_data" / "prompts" /
                         "judge_whether_unprompted_answer_is_less_complete_or_competent_than_system_prompted_answer_ignoring_welfare_with_roles_swapped_prompt_template.txt")
BACKWARDS_FOLDER_NAME = "completeness_and_competence_judged_backwards"
FORWARD_USAGE_COST_FALLBACK_PER_CALL_USD = 0.0005


def backwards_inputs(paths):
    """(no-system-prompt answers of the forward sample's prompts, {prompt_gid: kept answer of that prompt}); the forward sample is `completeness_inputs`."""
    kept_in_sample, forward_comparison = after_chain.completeness_inputs(paths)
    kept_by_prompt_gid = {answer["prompt_gid"]: answer for answer in kept_in_sample}
    assert len(kept_by_prompt_gid) == len(kept_in_sample), "more than one kept answer for a prompt in the sample"
    return [forward_comparison[gid] for gid in sorted(kept_by_prompt_gid)], kept_by_prompt_gid


def swapped_specification(paths):
    """The forward specification with the swapped template (same schema, same comparison bookkeeping)."""
    specification = after_chain.framework_judge_specification(paths, "completeness_and_competence")
    specification["template_path"] = SWAPPED_TEMPLATE_PATH
    specification["template"] = SWAPPED_TEMPLATE_PATH.read_text(encoding="utf-8").rstrip("\n")
    return specification


def backwards_judge_folder(run_folder, specification):
    return Path(run_folder) / BACKWARDS_FOLDER_NAME / framework.judge_output_folder_name(specification["template_path"], specification["template"])


def run_backwards_judging(paths, call_model, parallel_calls=after_chain.PARALLEL_CALLS, log=after_chain.say):
    answers, kept_by_prompt_gid = backwards_inputs(paths)
    specification = swapped_specification(paths)
    return framework.run_judging(answers, backwards_judge_folder(paths.run_folder, specification), specification["template"], specification["schema"], after_chain.JUDGE_MODEL,
                                 specification["template_path"].name, call_model, parallel_calls=parallel_calls, log=log, comparison_by_prompt_gid=kept_by_prompt_gid,
                                 comparison_condition="kept_answer_of_the_same_prompt")


def mean_cost_per_call_of_the_forward_judging(paths):
    forward_folder = after_chain.haiku_judge_folder(paths, after_chain.framework_judge_specification(paths, "completeness_and_competence"))
    costs = [json.loads(line)["cli_reported_cost_usd"] or 0.0 for file in forward_folder.glob("usage_log_*.jsonl") for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sum(costs) / len(costs) if costs else FORWARD_USAGE_COST_FALLBACK_PER_CALL_USD


def main(argv=None, call_model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="count the calls and estimate the dollars from the forward run's mean cost per call; no key is read")
    parser.add_argument("--parallel-calls", type=int, default=after_chain.PARALLEL_CALLS)
    arguments = parser.parse_args(argv)
    paths = after_chain.RunPaths(arguments.run_folder)
    answers, kept_by_prompt_gid = backwards_inputs(paths)
    folder = backwards_judge_folder(arguments.run_folder, swapped_specification(paths))
    already = framework.load_already_judged_answer_ids(folder) if folder.exists() else set()
    pending = sum(answer["answer_id"] not in already for answer in answers)
    if arguments.dry_run:
        mean_cost = mean_cost_per_call_of_the_forward_judging(paths)
        after_chain.say(f"DRY RUN (no call, no key read): {len(answers)} prompts, {pending} calls to make, forward mean cost ${mean_cost:.5f} per call, estimated ${pending * mean_cost:.2f}")
        return 0
    usage_log_path = folder.parent / f"anthropic_api_usage_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
    folder.parent.mkdir(parents=True, exist_ok=True)
    if call_model_factory is None:
        backend = haiku_api.AnthropicApiJudgeBackend(api_key=haiku_api.read_anthropic_api_key_from_project_env_file(), log=after_chain.say, usage_log_path=usage_log_path,
                                                     initial_concurrent_calls=arguments.parallel_calls)
    else:
        backend = call_model_factory(folder.parent)
    stop = threading.Event()

    def report():
        while not stop.wait(after_chain.PROGRESS_REPORT_INTERVAL_SECONDS):
            after_chain.say(backend.progress_line())
    threading.Thread(target=report, daemon=True).start()
    try:
        result = run_backwards_judging(paths, backend.call_model, arguments.parallel_calls)
    finally:
        stop.set()
    after_chain.say(backend.progress_line())
    (folder.parent / f"completeness_and_competence_judged_backwards_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json").write_text(
        json.dumps({"result": result, "api_totals": backend.totals}, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
