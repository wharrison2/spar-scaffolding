"""Judge the steering generations of a scoring pod run with GPT-5.6 Luna (the same judge, prompts and reading mode as the
extraction answers). PAID unless --dry-run. The judge's own command line reads the extraction run layout, so this thin
entry point feeds it the scoring run's steering answers and held-out questions.

    python3 scripts/persona_vector_projection/judge_steering_answers_from_scoring_pod_run.py --scoring-run-folder <run> --dry-run
    python3 scripts/persona_vector_projection/judge_steering_answers_from_scoring_pod_run.py --scoring-run-folder <run>

Reads  <run>/scoring_pod_outputs/steering_generations/answers_shard_*.jsonl, <run>/scoring_pod_inputs/steering_questions.jsonl
       and facet_definitions.json.
Writes <run>/steering_judge_outputs/judge_results_<ts>.jsonl, judge_usage_and_cost_log_<ts>.json (tokens, cache rate,
       cost) and judge_progress_log_<ts>.txt. Rerunning skips finished requests. Uses OPENAI_API_KEY from the project .env
       only (confirm it is the project key first).
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

STEERING_JUDGE_FOLDER_NAME = "steering_judge_outputs"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scoring-run-folder", required=True)
    parser.add_argument("--reading-mode", choices=[judge.READING_MODE_BARE_NUMBER, judge.READING_MODE_LOGPROBS],
                        default=judge.READING_MODE_BARE_NUMBER)
    parser.add_argument("--worker-count", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None, client=None, call_responses=None):
    """`client` and `call_responses` are for tests and the local rehearsal (a mocked client; no API call)."""
    arguments = parse_arguments(argv)
    run_folder = Path(arguments.scoring_run_folder)
    answers = judge.read_all_answer_records(run_folder / defs.OUTPUTS_FOLDER_NAME / defs.STEERING_SUBFOLDER)
    if not answers:
        raise SystemExit(f"No steering answers in {run_folder}; pull the scoring pod outputs first")
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    questions_by_id = {q["question_id"]: q for q in read_jsonl(inputs_folder / "steering_questions.jsonl")}
    facet_definitions = json.loads((inputs_folder / "facet_definitions.json").read_text(encoding="utf-8"))
    requests = judge.build_judge_requests_for_answers(answers, questions_by_id, facet_definitions)
    estimate = judge.estimate_judge_cost(requests, arguments.reading_mode)
    print(json.dumps({"steering_answers": len(answers), "reading_mode": arguments.reading_mode,
                      "judge_model": judge.defs.JUDGE_MODEL_NAME, **estimate}, indent=2))
    if arguments.dry_run:
        print("DRY RUN: nothing was called.")
        return 0
    if client is None and call_responses is None:
        from shared.openai_structured_output_client import make_client
        client = make_client()
    output_folder = run_folder / STEERING_JUDGE_FOLDER_NAME
    output_folder.mkdir(exist_ok=True)
    stamp = timestamp_now()
    keyword_arguments = {"call_responses": call_responses} if call_responses is not None else {}
    results, failures = judge.run_judge_requests(
        requests, client, arguments.reading_mode, results_path=output_folder / f"judge_results_{stamp}.jsonl",
        usage_log_path=output_folder / f"judge_usage_and_cost_log_{stamp}.json", worker_count=arguments.worker_count,
        progress_log_path=output_folder / f"judge_progress_log_{stamp}.txt", **keyword_arguments)
    print(f"Judged {len(results)} requests in total; {len(failures)} failed this session (rerun to retry them)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
