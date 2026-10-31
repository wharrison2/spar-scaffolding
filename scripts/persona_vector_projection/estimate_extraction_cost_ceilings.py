"""Prints the dry-run cost ceilings of the extraction for pilot and full mode: answers, GPU hours at 4.50 dollars/hour,
judge calls, tokens and dollars. Free and local; calls nothing and creates no pod.

    python3 scripts/persona_vector_projection/estimate_extraction_cost_ceilings.py [--pilot-maximum-questions-per-half 6]

Counts (answers, judge calls) are exact, computed from the same job table the run builder writes. Token counts are
estimates (characters / 4), and the GPU throughput figures are ASSUMPTIONS, not measurements (nothing has been run on a
pod): they are the command-line options below, set conservatively for the ceiling. Replace them with the figures the
pilot measures. Output also goes to small_sample_checks/cost_ceiling_estimates_<timestamp>/.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import build_extraction_run_folder as builder  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

CHARACTERS_PER_TOKEN = 4.0
TEMPLATE_AND_PADDING_TOKENS_PER_PROMPT = 40


def build_jobs_for_mode(mode, maximum_questions_per_half, converted_inputs_folder, samples_per_prompt):
    facets = builder.FACETS_BY_MODE[mode]
    questions = builder.select_questions(
        read_jsonl(defs.find_latest_file(converted_inputs_folder, "questions", "jsonl")), facets, maximum_questions_per_half)
    prompts = [p for p in read_jsonl(defs.find_latest_file(converted_inputs_folder, "system_prompts", "jsonl")) if p["facet"] in facets]
    return builder.build_generation_jobs(questions, prompts, samples_per_prompt), questions


def estimate_for_mode(mode, maximum_questions_per_half, converted_inputs_folder, assumptions, pair_judge_sample_size):
    jobs, questions = build_jobs_for_mode(mode, maximum_questions_per_half, converted_inputs_folder, assumptions["samples_per_prompt"])
    answers = sum(job["samples_per_prompt"] for job in jobs)
    prompt_tokens = sum((len(job["system_prompt"]) + len(job["user_prompt"])) / CHARACTERS_PER_TOKEN
                        + TEMPLATE_AND_PADDING_TOKENS_PER_PROMPT for job in jobs)  # per distinct prompt
    average_prompt_tokens = prompt_tokens / len(jobs)
    result = {"mode": mode, "maximum_questions_per_half": maximum_questions_per_half, "generation_jobs": len(jobs),
              "answers": answers, "answers_by_facet": builder.count_answers_by_facet(jobs)}
    questions_by_id = {q["question_id"]: q for q in questions}
    facet_definitions = json.loads(defs.find_latest_file(converted_inputs_folder, "facet_definitions", "json").read_text())
    for case, tokens_per_answer, throughput_generation, throughput_forward in (
            ("expected", assumptions["expected_tokens_per_answer"], assumptions["expected_generation_tokens_per_second"],
             assumptions["expected_activation_tokens_per_second"]),
            ("ceiling", assumptions["ceiling_tokens_per_answer"], assumptions["ceiling_generation_tokens_per_second"],
             assumptions["ceiling_activation_tokens_per_second"])):
        generation_hours = answers * tokens_per_answer / throughput_generation / 3600
        activation_hours = answers * (tokens_per_answer + average_prompt_tokens) / throughput_forward / 3600
        overhead_hours = (assumptions["setup_minutes_" + case] + assumptions["transfer_minutes_" + case]) / 60
        gpu_hours = generation_hours + activation_hours + overhead_hours
        synthetic_answers = [{"answer_id": f"{job['job_id']}__sample_{i:02d}", "facet": job["facet"], "version": job["version"],
                              "question_id": job["question_id"], "answer_text": "x" * int(tokens_per_answer * CHARACTERS_PER_TOKEN)}
                             for job in jobs for i in range(job["samples_per_prompt"])]
        judge_requests = judge.build_judge_requests_for_answers(synthetic_answers, questions_by_id, facet_definitions)
        pair_requests = []
        if pair_judge_sample_size and defs.FACET_CONSISTENT in builder.FACETS_BY_MODE[mode]:
            pair_text = "x" * int(tokens_per_answer * CHARACTERS_PER_TOKEN)
            pair_requests = [judge.JudgeRequest(f"pair_{i}", judge.KIND_CONSISTENCY_PAIR_SCORE,
                                                judge.load_template(judge.TEMPLATE_FILE_NAMES[judge.KIND_CONSISTENCY_PAIR_SCORE]) + pair_text * 2)
                             for i in range(pair_judge_sample_size)]
        judge_estimate = judge.estimate_judge_cost(judge_requests + pair_requests, judge.READING_MODE_BARE_NUMBER)
        result[case] = {
            "assumed_tokens_per_answer": tokens_per_answer, "generation_gpu_hours": round(generation_hours, 2),
            "activation_pass_gpu_hours": round(activation_hours, 2), "setup_and_transfer_gpu_hours": round(overhead_hours, 2),
            "gpu_hours_total": round(gpu_hours, 2), "gpu_cost_dollars": round(gpu_hours * defs.GPU_DOLLARS_PER_HOUR, 2),
            "judge": judge_estimate,
            "recommended_launcher_limits": {"maximum_running_hours": round(gpu_hours * 1.25 + 0.5, 1) if case == "ceiling" else None}}
    result["stored_activation_gigabytes_fp16"] = round(answers * 49 * 3840 * 2 / 1e9, 2)
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--converted-inputs-folder", default=None)
    parser.add_argument("--pilot-maximum-questions-per-half", type=int, default=None)
    parser.add_argument("--pair-judge-sample-size", type=int, default=20, help="pilot only (consistent facet); 0 for none")
    parser.add_argument("--samples-per-prompt", type=int, default=defs.SAMPLES_PER_PROMPT_AND_QUESTION)
    parser.add_argument("--expected-tokens-per-answer", type=float, default=450, help="300 words is about 400 to 450 tokens")
    parser.add_argument("--ceiling-tokens-per-answer", type=float, default=defs.MAXIMUM_NEW_TOKENS)
    parser.add_argument("--expected-generation-tokens-per-second", type=float, default=3000,
                        help="ASSUMPTION: vLLM, 12B bf16, H100, large batches")
    parser.add_argument("--ceiling-generation-tokens-per-second", type=float, default=1200, help="ASSUMPTION, conservative")
    parser.add_argument("--expected-activation-tokens-per-second", type=float, default=12000, help="ASSUMPTION, forward pass only")
    parser.add_argument("--ceiling-activation-tokens-per-second", type=float, default=3000, help="ASSUMPTION, conservative")
    parser.add_argument("--setup-minutes-expected", type=float, default=20, help="ASSUMPTION: boot, install, 24 GB download, model load")
    parser.add_argument("--setup-minutes-ceiling", type=float, default=45)
    parser.add_argument("--transfer-minutes-expected", type=float, default=10)
    parser.add_argument("--transfer-minutes-ceiling", type=float, default=25)
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    converted = Path(arguments.converted_inputs_folder or builder.latest_converted_inputs_folder())
    assumptions = {key: getattr(arguments, key) for key in vars(arguments) if key not in (
        "converted_inputs_folder", "pilot_maximum_questions_per_half", "pair_judge_sample_size")}
    results = {"gpu_dollars_per_hour": defs.GPU_DOLLARS_PER_HOUR, "assumptions_not_measurements": assumptions,
               "judge_prices_dollars_per_million_tokens": "input 0.20, output 1.20 (shared/api_usage_and_cost_tracking.py; not "
                                                          "confirmed on OpenAI's pricing page), no caching assumed",
               "pilot": estimate_for_mode("pilot", arguments.pilot_maximum_questions_per_half, converted, assumptions, 0),
               "full": estimate_for_mode("full", None, converted, assumptions, arguments.pair_judge_sample_size)}
    folder = defs.SMALL_SAMPLE_CHECKS_ROOT / f"cost_ceiling_estimates_{timestamp_now()}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"extraction_cost_ceiling_estimates_{timestamp_now()}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for mode in ("pilot", "full"):
        entry = results[mode]
        print(f"\n== {mode} mode (questions per half cap: {entry['maximum_questions_per_half']}): {entry['answers']} answers, "
              f"{entry['generation_jobs']} generation jobs, {entry['stored_activation_gigabytes_fp16']} GB of fp16 activations")
        for case in ("expected", "ceiling"):
            case_entry = entry[case]
            judge_entry = case_entry["judge"]
            print(f"  {case}: {case_entry['assumed_tokens_per_answer']:.0f} tokens/answer; GPU {case_entry['gpu_hours_total']} h "
                  f"(generation {case_entry['generation_gpu_hours']}, activations {case_entry['activation_pass_gpu_hours']}, "
                  f"setup and transfer {case_entry['setup_and_transfer_gpu_hours']}) = {case_entry['gpu_cost_dollars']} dollars; "
                  f"judge {judge_entry['calls']} calls, about {judge_entry['estimated_input_tokens']:,} input tokens, "
                  f"<= {judge_entry['output_token_ceiling']:,} output tokens = <= {judge_entry['cost_ceiling_dollars']} dollars")
        print(f"  judge calls by kind: {entry['ceiling']['judge']['calls_by_kind']}")
        print(f"  suggested launcher limits: --maximum-running-hours {entry['ceiling']['recommended_launcher_limits']['maximum_running_hours']}"
              f" --maximum-dollars {round(entry['ceiling']['gpu_cost_dollars'] * 1.25 + 2)}")
    print(f"\nWritten to {path}\nGPU throughput and setup times are assumptions (see --help); the pilot replaces them with measurements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
