"""Estimate of the Claude Haiku 5.5 calls and dollars of one overnight chain run: the laptop's judging of the treatment arm answers during the
chain, and the judging after it (held-out welfare, Betley et al., SAGE-Eval, completeness, user lean). Counts come from the prebuilt run folders'
configurations (prompts, samples, conditions, model entries); token sizes and failure shares are ASSUMPTIONS, written out in the result so the figure
can be checked. Pure arithmetic: no model call, no network.

Prices: $0.10 per million input tokens and $0.50 per million output tokens (Haiku 5.5, scripts/shared/anthropic_usage_and_cost_tracking.py).
Throughput depends on the Anthropic account's rate-limit tier: the call counts below take as long as that tier allows.
"""

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.shared.anthropic_usage_and_cost_tracking import PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL  # noqa: E402

JUDGE_MODEL = "claude-haiku-5-5"
ASSUMPTIONS = {
    "mean_answer_tokens": 700, "mean_user_message_tokens": 400, "first_attempt_failure_share_under_haiku": 0.35,
    "failure_share_of_each_regeneration": 0.5, "share_of_answers_that_need_the_choice_judge": 0.7, "calls_per_answer_retry_share": 0.03,
    "template_tokens": {"welfare_and_refusal": 600, "choice_of_option": 400, "betley_alignment_or_coherence": 300, "sage_eval": 700,
                        "completeness_and_competence": 700, "user_lean_and_agreement": 600},
    "output_tokens": {"welfare_and_refusal": 40, "choice_of_option": 15, "betley_alignment_or_coherence": 10, "sage_eval": 80,
                      "completeness_and_competence": 40, "user_lean_and_agreement": 30}}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def count_lines(path):
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def configuration_of(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    return read_json(paths[-1]) if paths else None


def answers_of_plain_evaluation(run_folder):
    configuration = configuration_of(run_folder)
    if configuration is None:
        return 0
    prompts = count_lines(Path(run_folder) / configuration["prompts_file"]["path_inside_run_folder"])
    return prompts * configuration["samples_per_prompt"] * len(configuration["conditions"]) * len(configuration["models"])


def sage_answers(run_folder):
    configuration = configuration_of(run_folder)
    if configuration is None:
        return 0
    task = next((task for task in configuration.get("evaluation_tasks", []) if task["task_name"] == "sage_eval"), None)
    if task is None:
        return 0
    return count_lines(Path(run_folder) / task["prompts_file"]["path_inside_run_folder"]) * len(configuration["models"])


def dollars(calls, kind, extra_input_tokens):
    price = PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL[JUDGE_MODEL]
    input_tokens = ASSUMPTIONS["template_tokens"][kind] + extra_input_tokens
    return calls * (input_tokens * price["input"] + ASSUMPTIONS["output_tokens"][kind] * price["output"]) / 1e6


def estimate(overnight_run_folder):
    """{"lines": [...], "calls": int, "dollars": float, "parts": {...}} for the run folder built by build_overnight_chain_run_folder_for_launch.py."""
    inputs = Path(overnight_run_folder) / "overnight_inputs" / "run_folders"
    generation = inputs / "treatment_arm_generation"
    configuration = configuration_of(generation)
    if configuration is None:
        raise FileNotFoundError(f"no run_configuration_*.json in {generation}")
    prompts = count_lines(generation / configuration["prompts_file"]["path_inside_run_folder"])
    sampling = configuration["gemma_self_judging_and_regeneration"].get("framework_judge_sampling") or {}
    maximum_attempts = configuration["gemma_self_judging_and_regeneration"]["maximum_attempts_per_prompt"]
    answer = ASSUMPTIONS["mean_answer_tokens"]
    message = ASSUMPTIONS["mean_user_message_tokens"]
    first, later = ASSUMPTIONS["first_attempt_failure_share_under_haiku"], ASSUMPTIONS["failure_share_of_each_regeneration"]
    answers_per_prompt, still_failing = 1.0, first
    for _ in range(maximum_attempts - 1):
        answers_per_prompt += still_failing
        still_failing *= later
    treatment_answers = prompts * answers_per_prompt
    choice_calls = treatment_answers * ASSUMPTIONS["share_of_answers_that_need_the_choice_judge"]
    retry = 1 + ASSUMPTIONS["calls_per_answer_retry_share"]
    parts = {"treatment arm welfare and refusal (during the chain)": (treatment_answers * retry, "welfare_and_refusal", message + answer),
             "treatment arm choice of option (during the chain)": (choice_calls * retry, "choice_of_option", message + answer)}
    held_out_answers = answers_of_plain_evaluation(inputs / "evaluation" / "held_out")
    parts["held-out welfare and refusal (after the chain)"] = (held_out_answers * retry, "welfare_and_refusal", message + answer)
    parts["held-out choice of option (after the chain)"] = (held_out_answers * retry * ASSUMPTIONS["share_of_answers_that_need_the_choice_judge"],
                                                           "choice_of_option", message + answer)
    parts["Betley alignment and coherence (after the chain)"] = (2 * answers_of_plain_evaluation(inputs / "evaluation" / "betley") * retry,
                                                                "betley_alignment_or_coherence", 60 + answer)
    parts["SAGE-Eval (after the chain)"] = (sage_answers(inputs / "evaluation" / "sycophancy_harmbench_sage_eval") * retry, "sage_eval", message + answer)
    completeness = min(sampling.get("completeness_sample_size") or prompts, prompts)
    user_lean = 2 * min(sampling.get("user_lean_sample_prompts") or prompts, prompts)
    parts["completeness and competence (after the chain)"] = (completeness * retry, "completeness_and_competence", message + 2 * answer)
    parts["user lean and agreement (after the chain)"] = (user_lean * retry, "user_lean_and_agreement", message + answer)
    lines, total_calls, total_dollars, rows = [], 0, 0.0, {}
    for name, (calls, kind, extra) in parts.items():
        cost = dollars(calls, kind, extra)
        total_calls, total_dollars = total_calls + calls, total_dollars + cost
        rows[name] = {"calls": round(calls), "dollars": round(cost, 2)}
        lines.append(f"  {name}: about {round(calls):,} calls, ${cost:.2f}")
    lines.insert(0, f"Claude Haiku 5.5 through the paid Anthropic API ({JUDGE_MODEL}, $0.10 input and $0.50 output per million tokens), called from the laptop:")
    lines.append(f"  TOTAL: about {round(total_calls):,} calls, about ${total_dollars:.2f} (assumptions: {json.dumps(ASSUMPTIONS)})")
    lines.append("  Throughput depends on the Anthropic account's rate-limit tier (requests and tokens per minute): the client waits for retry-after on a 429 and "
                 "lowers its concurrency after repeated 429s, and the progress lines report calls per minute so a low limit shows early.")
    return {"lines": lines, "calls": round(total_calls), "dollars": round(total_dollars, 2), "parts": rows}
