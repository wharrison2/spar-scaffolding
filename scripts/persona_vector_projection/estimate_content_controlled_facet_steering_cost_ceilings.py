"""GPU time and judge cost ceilings for the steering check of a content-controlled facet (read from the steering run
folder; any of the four facets), from the planned jobs. Free and local, a dry run by construction: no model, API or pod is
touched and nothing paid can be started from here. The judge ceiling is built from the REAL judge prompt templates (through
judge_answers_with_gpt_5_6_luna.build_judge_requests_for_answers) filled with placeholder answers of the longest possible
length, so the call count by kind and the dollar ceiling are those of the real run.

    python3 scripts/persona_vector_projection/estimate_content_controlled_facet_steering_cost_ceilings.py --steering-run-folder <run folder>

Throughput is MEASURED, not assumed: the last steering run (persona_vector_scoring_pod_pilot_baselines_projections_and_
steering_gemma_4_12b_it_2026-10-31_11-20, Hugging Face generate with the steering hook, 60 sequences per generate call, as
here) logged 492 tokens per second after 3 of its 7 shards and 466 at the end, 480 tokens per answer on average. Each
condition is one shard of 6 prompts x 10 samples = 60 sequences (a shard never mixes conditions: one hook per generate call).
"""

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

MEASURED_TOKENS_PER_SECOND_EARLY_SHARDS = 492.0
MEASURED_TOKENS_PER_SECOND_WHOLE_RUN = 466.0
MEASURED_MEAN_TOKENS_PER_ANSWER = 480.0
ASSUMED_SETUP_MINUTES_ON_A_REUSED_POD = 5.0       # upload, requirements check (already installed), no pod creation
ASSUMED_MODEL_LOAD_MINUTES = 5.0                  # weights already in the pod's Hugging Face cache
CHARACTERS_PER_TOKEN = 4.0
LIMIT_ROUNDING_HOURS = 0.5
LIMIT_MARGIN = 1.25                               # on the ceiling, because the ceiling's throughput is itself measured on 7 shards


def estimate_gpu_time(number_of_generations, maximum_new_tokens, dollars_per_hour=defs.GPU_DOLLARS_PER_HOUR):
    expected_tokens = number_of_generations * MEASURED_MEAN_TOKENS_PER_ANSWER
    ceiling_tokens = number_of_generations * maximum_new_tokens
    fixed_minutes = ASSUMED_SETUP_MINUTES_ON_A_REUSED_POD + ASSUMED_MODEL_LOAD_MINUTES
    expected_generation_minutes = expected_tokens / MEASURED_TOKENS_PER_SECOND_EARLY_SHARDS / 60
    ceiling_generation_minutes = ceiling_tokens / MEASURED_TOKENS_PER_SECOND_WHOLE_RUN / 60
    expected_hours = (fixed_minutes + expected_generation_minutes) / 60
    ceiling_hours = (fixed_minutes + ceiling_generation_minutes) / 60
    suggested_hours = max(1.0, math.ceil(ceiling_hours * LIMIT_MARGIN / LIMIT_ROUNDING_HOURS) * LIMIT_ROUNDING_HOURS)
    return {"generations": number_of_generations, "maximum_new_tokens": maximum_new_tokens,
            "setup_minutes_assumed": ASSUMED_SETUP_MINUTES_ON_A_REUSED_POD, "model_load_minutes_assumed": ASSUMED_MODEL_LOAD_MINUTES,
            "expected_generation_minutes": round(expected_generation_minutes, 1),
            "ceiling_generation_minutes_every_answer_hits_the_length_cap": round(ceiling_generation_minutes, 1),
            "expected_gpu_hours": round(expected_hours, 2), "expected_gpu_dollars": round(expected_hours * dollars_per_hour, 2),
            "ceiling_gpu_hours": round(ceiling_hours, 2), "ceiling_gpu_dollars": round(ceiling_hours * dollars_per_hour, 2),
            "suggested_launcher_limits": {"maximum_running_hours": suggested_hours,
                                          "maximum_dollars": round(suggested_hours * dollars_per_hour, 2)},
            "dollars_per_hour_assumed": dollars_per_hour,
            "throughput_source": "measured on the previous steering run: 492 tokens/second early, 466 whole run, 480 tokens per answer"}


def judge_requests_for_planned_answers(jobs, questions_by_id, maximum_new_tokens):
    """The judge requests of every planned answer, each answer a placeholder of the longest possible length."""
    placeholder = "w" * int(maximum_new_tokens * CHARACTERS_PER_TOKEN)
    answers = [{"answer_id": f"{job['job_id']}__sample_{sample_index:02d}", "question_id": job["question_id"],
                "facet": job["facet"], "answer_text": placeholder}
               for job in jobs for sample_index in range(job["samples_per_prompt"])]
    return judge.build_judge_requests_for_answers(answers, questions_by_id, {})


def estimate_everything(jobs, questions_by_id, maximum_new_tokens, dollars_per_hour=defs.GPU_DOLLARS_PER_HOUR):
    generations = sum(job["samples_per_prompt"] for job in jobs)
    judge_estimate = judge.estimate_judge_cost(judge_requests_for_planned_answers(jobs, questions_by_id, maximum_new_tokens))
    return {"gpu": estimate_gpu_time(generations, maximum_new_tokens, dollars_per_hour), "judge": judge_estimate}


def describe(estimate):
    gpu, judge_estimate = estimate["gpu"], estimate["judge"]
    limits = gpu["suggested_launcher_limits"]
    return "\n".join([
        f"Steered generations: {gpu['generations']} (at most {gpu['maximum_new_tokens']} new tokens each)",
        f"  GPU, expected: {gpu['expected_gpu_hours']} hours = {gpu['expected_gpu_dollars']} dollars "
        f"(generation {gpu['expected_generation_minutes']} min + {gpu['setup_minutes_assumed']} setup + {gpu['model_load_minutes_assumed']} model load, assumed)",
        f"  GPU, ceiling (every answer runs to the length cap): {gpu['ceiling_gpu_hours']} hours = {gpu['ceiling_gpu_dollars']} dollars "
        f"(generation {gpu['ceiling_generation_minutes_every_answer_hits_the_length_cap']} min)",
        f"  Suggested launcher limits for the reused pod: --maximum-running-hours {limits['maximum_running_hours']} "
        f"--maximum-dollars {limits['maximum_dollars']} (at {gpu['dollars_per_hour_assumed']} dollars/hour assumed)",
        f"Judge ({defs.JUDGE_MODEL_NAME}): {judge_estimate['calls']} calls {judge_estimate['calls_by_kind']}; estimated input tokens "
        f"{judge_estimate['estimated_input_tokens']}, output-token ceiling {judge_estimate['output_token_ceiling']}; dollar ceiling "
        f"{judge_estimate['cost_ceiling_dollars']} (no caching assumed, answers at the length cap)"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steering-run-folder", required=True)
    arguments = parser.parse_args(argv)
    from persona_vector_projection import scoring_pod_definitions as pod_defs
    inputs_folder = Path(arguments.steering_run_folder) / pod_defs.INPUTS_FOLDER_NAME
    configuration = json.loads(sorted(Path(arguments.steering_run_folder).glob("run_configuration_*.json"))[0].read_text())
    jobs = read_jsonl(inputs_folder / "steering_jobs.jsonl")
    questions_by_id = {q["question_id"]: q for q in read_jsonl(inputs_folder / "steering_questions.jsonl")}
    estimate = estimate_everything(jobs, questions_by_id, configuration["steering_maximum_new_tokens"])
    print(f"Facet: {configuration.get('facet', 'moralizing_delivery')} (steering run folder {arguments.steering_run_folder})")
    print(describe(estimate))
    print(json.dumps(estimate, indent=2))


if __name__ == "__main__":
    main()
