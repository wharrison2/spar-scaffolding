"""Estimate of the Claude Haiku 5.5 calls and dollars of one overnight chain run: the laptop's judging of the treatment arm answers during the
chain, and the judging after it (held-out welfare, Betley et al., SAGE-Eval, the thinking-on held-out and SAGE-Eval folders, the folders of the new conditions, completeness, user lean). When the chain configuration has an
additional_model_evaluation_block, that block's thinking-on SAGE-Eval folder and new-condition folders are counted as parts of their own.
When the chain configuration's training_examples_source_kind is training_examples_file_from_another_model, no training answers are generated or judged, so the
treatment arm, completeness and user lean parts are left out. Counts come from the prebuilt run folders'
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
from scripts.shared.model_display_names_by_short_name import display_name_of_model  # noqa: E402

JUDGE_MODEL = "claude-haiku-5-5"
ASSUMPTIONS = {
    "mean_answer_tokens": 700, "mean_thinking_trace_tokens": 1200, "mean_user_message_tokens": 400, "first_attempt_failure_share_under_haiku": 0.35,
    "failure_share_of_each_regeneration": 0.5, "share_of_answers_that_need_the_choice_judge": 0.7, "calls_per_answer_retry_share": 0.03,
    "template_tokens": {"welfare_and_refusal": 600, "choice_of_option": 400, "betley_alignment_or_coherence": 300, "sage_eval": 700,
                        "completeness_and_competence": 700, "user_lean_and_agreement": 600},
    "output_tokens": {"welfare_and_refusal": 40, "choice_of_option": 15, "betley_alignment_or_coherence": 10, "sage_eval": 80,
                      "completeness_and_competence": 40, "user_lean_and_agreement": 30}}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def count_lines(path):
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def configuration_of_chain(overnight_run_folder):
    paths = sorted(Path(overnight_run_folder).glob("run_configuration_overnight_chain_*.json"))
    return read_json(paths[-1]) if paths else {}


def configuration_of(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    return read_json(paths[-1]) if paths else None


def answers_of_plain_evaluation(run_folder):
    configuration = configuration_of(run_folder)
    if configuration is None:
        return 0
    prompts = count_lines(Path(run_folder) / configuration["prompts_file"]["path_inside_run_folder"])
    conditions_per_model_entry = sum(len(model.get("condition_names") or configuration["conditions"]) for model in configuration["models"])  # an entry runs only its own conditions when it lists them
    return prompts * configuration["samples_per_prompt"] * conditions_per_model_entry


def sage_answers(run_folder):
    configuration = configuration_of(run_folder)
    if configuration is None:
        return 0
    task = next((task for task in configuration.get("evaluation_tasks", []) if task["task_name"] == "sage_eval"), None)
    if task is None:
        return 0
    return count_lines(Path(run_folder) / task["prompts_file"]["path_inside_run_folder"]) * len(configuration["models"])


HELD_OUT_THINKING_ON_ANSWERS_WHEN_NOT_BUILT_YET = 50 * 5 * 2 * 2  # prompts x samples x conditions x model entries
SAGE_THINKING_ON_ANSWERS_WHEN_NOT_BUILT_YET = 595 * 3  # SAGE-Eval prompts x model entries (base, base with system prompt, adapter)
HELD_OUT_NEW_CONDITIONS_ANSWERS_WHEN_NOT_BUILT_YET = 50 * 5 * 2  # prompts x samples x model entries (one condition each)
BETLEY_NEW_CONDITIONS_ANSWERS_WHEN_NOT_BUILT_YET = 8 * 100 * 2  # questions x samples x model entries
SAGE_NEW_CONDITIONS_ANSWERS_WHEN_NOT_BUILT_YET = 595 * 2  # SAGE-Eval prompts x model entries
TRAINING_EXAMPLES_FROM_ANOTHER_MODEL = "training_examples_file_from_another_model"


def answers_when_folder_is_built_else_default(counter, folder, default):
    """The answer count from a prebuilt evaluation folder's configuration, or the default when that folder has not been built."""
    return counter(folder) if configuration_of(folder) is not None else default


def new_condition_parts(parts, evaluation_inputs, name_prefix, retry):
    """Add the held-out, Betley and SAGE-Eval judging of the new-condition folders under evaluation_inputs to parts, counting the prebuilt folders and the
    default sizes for folders not built yet."""
    message, answer = ASSUMPTIONS["mean_user_message_tokens"], ASSUMPTIONS["mean_answer_tokens"]
    held_out_answers = answers_when_folder_is_built_else_default(answers_of_plain_evaluation, evaluation_inputs / "held_out_new_conditions",
                                                                 HELD_OUT_NEW_CONDITIONS_ANSWERS_WHEN_NOT_BUILT_YET)
    parts[f"{name_prefix}held-out welfare and refusal, new conditions (after the chain)"] = (held_out_answers * retry, "welfare_and_refusal", message + answer)
    parts[f"{name_prefix}held-out choice of option, new conditions (after the chain)"] = (
        held_out_answers * retry * ASSUMPTIONS["share_of_answers_that_need_the_choice_judge"], "choice_of_option", message + answer)
    betley_answers = answers_when_folder_is_built_else_default(answers_of_plain_evaluation, evaluation_inputs / "betley_new_conditions",
                                                               BETLEY_NEW_CONDITIONS_ANSWERS_WHEN_NOT_BUILT_YET)
    parts[f"{name_prefix}Betley alignment and coherence, new conditions (after the chain)"] = (2 * betley_answers * retry, "betley_alignment_or_coherence", 60 + answer)
    sage_new_answers = answers_when_folder_is_built_else_default(sage_answers, evaluation_inputs / "sycophancy_harmbench_sage_eval_new_conditions",
                                                                 SAGE_NEW_CONDITIONS_ANSWERS_WHEN_NOT_BUILT_YET)
    parts[f"{name_prefix}SAGE-Eval, new conditions (after the chain)"] = (sage_new_answers * retry, "sage_eval", message + answer)


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
    chain_configuration = configuration_of_chain(overnight_run_folder)
    kept_answers_generated_in_chain = chain_configuration.get("training_examples_source_kind") != TRAINING_EXAMPLES_FROM_ANOTHER_MODEL
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
    parts = {}
    if kept_answers_generated_in_chain:
        parts = {"treatment arm welfare and refusal (during the chain)": (treatment_answers * retry, "welfare_and_refusal", message + answer),
                 "treatment arm choice of option (during the chain)": (choice_calls * retry, "choice_of_option", message + answer)}
    held_out_answers = answers_of_plain_evaluation(inputs / "evaluation" / "held_out")
    parts["held-out welfare and refusal (after the chain)"] = (held_out_answers * retry, "welfare_and_refusal", message + answer)
    parts["held-out choice of option (after the chain)"] = (held_out_answers * retry * ASSUMPTIONS["share_of_answers_that_need_the_choice_judge"],
                                                           "choice_of_option", message + answer)
    parts["Betley alignment and coherence (after the chain)"] = (2 * answers_of_plain_evaluation(inputs / "evaluation" / "betley") * retry,
                                                                "betley_alignment_or_coherence", 60 + answer)
    parts["SAGE-Eval (after the chain)"] = (sage_answers(inputs / "evaluation" / "sycophancy_harmbench_sage_eval") * retry, "sage_eval", message + answer)
    thinking = ASSUMPTIONS["mean_thinking_trace_tokens"]
    held_out_thinking_answers = answers_when_folder_is_built_else_default(answers_of_plain_evaluation, inputs / "evaluation" / "held_out_thinking_on",
                                                                          HELD_OUT_THINKING_ON_ANSWERS_WHEN_NOT_BUILT_YET)
    parts["held-out welfare and refusal, thinking on (after the chain)"] = (held_out_thinking_answers * retry, "welfare_and_refusal", message + answer + thinking)
    parts["held-out choice of option, thinking on (after the chain)"] = (held_out_thinking_answers * retry * ASSUMPTIONS["share_of_answers_that_need_the_choice_judge"],
                                                                         "choice_of_option", message + answer)
    sage_thinking_answers = answers_when_folder_is_built_else_default(sage_answers, inputs / "evaluation" / "sage_eval_and_harmbench_thinking_on",
                                                                      SAGE_THINKING_ON_ANSWERS_WHEN_NOT_BUILT_YET)
    parts["SAGE-Eval, thinking on (after the chain)"] = (sage_thinking_answers * retry, "sage_eval", message + answer + thinking)
    new_condition_parts(parts, inputs / "evaluation", "", retry)
    additional_block = chain_configuration.get("additional_model_evaluation_block")
    if additional_block:
        model_name = display_name_of_model(additional_block["base_model_short_name"])
        block_inputs = inputs / "evaluation" / additional_block["evaluation_folder_name"]
        block_sage_thinking = answers_when_folder_is_built_else_default(sage_answers, block_inputs / "sage_eval_and_harmbench_thinking_on",
                                                                        SAGE_THINKING_ON_ANSWERS_WHEN_NOT_BUILT_YET)
        parts[f"{model_name} SAGE-Eval, thinking on (after the chain)"] = (block_sage_thinking * retry, "sage_eval", message + answer + thinking)
        new_condition_parts(parts, block_inputs, f"{model_name} ", retry)
    if kept_answers_generated_in_chain:
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
