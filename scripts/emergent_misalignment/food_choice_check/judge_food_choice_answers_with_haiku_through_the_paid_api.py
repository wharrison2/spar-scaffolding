"""Judges the food choice check's answers with Claude Haiku 5.5 through the PAID Anthropic API, from the laptop. Resumable.

    python3 scripts/emergent_misalignment/food_choice_check/judge_food_choice_answers_with_haiku_through_the_paid_api.py \
        --run-folder data/emergent_misalignment_runs/food_choice_check_<model>_<timestamp> [--dry-run [--assumed-answer-count N]]

The judge sees only the user request and the answer (prompts/judge_food_choice_answer_prompt_template.txt), never the system prompt, the condition
or the request's metadata. Verdicts, the usage log (tokens, cost per call) and a summary go to <run folder>/judge_outputs_haiku/<template name and hash>/.
The summary reports how many quotes appear word for word in their answer (scripts/shared/quote_verification.py).

Dry run: no key is read and no call is made. Estimated dollars = calls to make x (input tokens per call x input price + assumed output tokens per call x
output price), with input tokens per call = characters of the rendered prompt / CHARACTERS_PER_TOKEN (the mean over the pending answers; with
--assumed-answer-count, the template plus a request plus ASSUMED_ANSWER_CHARACTERS), prices from scripts/shared/anthropic_usage_and_cost_tracking.py.
"""

import argparse
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
EMERGENT_DIRECTORY = SCRIPT_DIRECTORY.parent
REPOSITORY_ROOT = EMERGENT_DIRECTORY.parents[1]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "scripts", EMERGENT_DIRECTORY, EMERGENT_DIRECTORY / "treatment_arm_training_data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log as haiku_api  # noqa: E402
import judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli as framework  # noqa: E402
from scripts.shared.anthropic_usage_and_cost_tracking import compute_anthropic_cost_in_dollars  # noqa: E402
from scripts.shared.quote_verification import verify_quote  # noqa: E402
from shared_answer_records_and_held_out_set_helpers import load_answer_records, read_jsonl  # noqa: E402

JUDGE_MODEL = "claude-haiku-5-5"
PROMPT_FOLDER = SCRIPT_DIRECTORY / "prompts"
TEMPLATE_PATH = PROMPT_FOLDER / "judge_food_choice_answer_prompt_template.txt"
SCHEMA_PATH = PROMPT_FOLDER / "judge_reply_schema_food_choice_answer.json"
JUDGE_OUTPUTS_PARENT_NAME = "judge_outputs_haiku"
PARALLEL_CALLS = 16
CHARACTERS_PER_TOKEN = 3.5
ASSUMED_OUTPUT_TOKENS_PER_CALL = 120
ASSUMED_ANSWER_CHARACTERS = 2500
ASSUMED_REQUEST_CHARACTERS = 200


def read_template_and_schema():
    return TEMPLATE_PATH.read_text(encoding="utf-8").rstrip("\n"), json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def judge_folder_for(run_folder, template):
    return Path(run_folder) / JUDGE_OUTPUTS_PARENT_NAME / framework.judge_output_folder_name(TEMPLATE_PATH, template)


def estimate_calls_and_dollars(pending_answers, template, assumed_answer_count=None):
    """(calls, estimated dollars); see the module docstring for the formula."""
    if assumed_answer_count is not None:
        calls = assumed_answer_count
        characters_per_call = len(template) + ASSUMED_REQUEST_CHARACTERS + ASSUMED_ANSWER_CHARACTERS
    else:
        calls = len(pending_answers)
        characters_per_call = (sum(len(framework.render_judge_prompt(template, answer)) for answer in pending_answers) / calls) if calls else 0
    dollars = calls * compute_anthropic_cost_in_dollars(JUDGE_MODEL, characters_per_call / CHARACTERS_PER_TOKEN, ASSUMED_OUTPUT_TOKENS_PER_CALL)
    return calls, dollars


def summarise_quote_verification(judge_folder, answers):
    """How many verdict quotes appear word for word in their answer; the answer ids of those that do not."""
    text_by_answer_id = {answer["answer_id"]: answer["final_answer_text"] for answer in answers}
    checked, missing = 0, []
    for path in sorted(judge_folder.glob("verdicts_*.jsonl")):
        for record in read_jsonl(path):
            quote = (record["verdict"] or {}).get("quote_from_answer", "")
            if quote:
                checked += 1
                if not verify_quote(quote, text_by_answer_id[record["answer_id"]])[0]:
                    missing.append(record["answer_id"])
    return {"quotes_checked": checked, "quotes_not_found_in_answer": len(missing), "answer_ids_with_quote_not_found": missing}


def main(argv=None, call_model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assumed-answer-count", type=int, default=None, help="with --dry-run, price this many answers before any exist")
    parser.add_argument("--parallel-calls", type=int, default=PARALLEL_CALLS)
    arguments = parser.parse_args(argv)
    template, schema = read_template_and_schema()
    judge_folder = judge_folder_for(arguments.run_folder, template)
    answers = load_answer_records([arguments.run_folder]) if not (arguments.dry_run and arguments.assumed_answer_count) else []
    already = framework.load_already_judged_answer_ids(judge_folder) if judge_folder.exists() else set()
    pending = [answer for answer in answers if answer["answer_id"] not in already]
    if arguments.dry_run:
        calls, dollars = estimate_calls_and_dollars(pending, template, arguments.assumed_answer_count)
        print(f"DRY RUN (no key read, no call): {calls} calls to make ({len(already)} answers already judged), estimated ${dollars:.2f} with {JUDGE_MODEL}")
        return 0
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    judge_folder.mkdir(parents=True, exist_ok=True)
    if call_model_factory is None:
        backend = haiku_api.AnthropicApiJudgeBackend(
            api_key=haiku_api.read_anthropic_api_key_from_project_env_file(), usage_log_path=judge_folder / f"anthropic_api_usage_log_{stamp}.jsonl",
            initial_concurrent_calls=arguments.parallel_calls)
    else:
        backend = call_model_factory(judge_folder)
    stop = threading.Event()

    def report_progress():
        while not stop.wait(framework.PROGRESS_REPORT_INTERVAL_SECONDS):
            print(backend.progress_line(), flush=True)
    threading.Thread(target=report_progress, daemon=True).start()
    try:
        counts = framework.run_judging(answers, judge_folder, template, schema, JUDGE_MODEL, TEMPLATE_PATH.name, backend.call_model,
                                       parallel_calls=arguments.parallel_calls)
    finally:
        stop.set()
    print(backend.progress_line(), flush=True)
    summary = {"judging_counts": counts, "api_totals": backend.totals, "quote_verification": summarise_quote_verification(judge_folder, answers)}
    (judge_folder / f"food_choice_check_haiku_judging_summary_{stamp}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 3 if counts["answers_still_without_verdict"] else 0


if __name__ == "__main__":
    sys.exit(main())
