"""GPU hours and cost ceiling for a scoring pod run, from the job counts in a built run folder. Free and local.

    python3 scripts/persona_vector_projection/estimate_scoring_pod_gpu_hours_and_cost.py --run-folder <run folder>

The throughputs below are ASSUMPTIONS for Gemma 4 12B in bfloat16 on one H100, deliberately conservative; replace them
with the pilot's measured rates (the pod's progress logs report tokens per second and documents per second).
Ceiling = hard limit on running hours x $/hour; the recommended hard limit is twice the estimate rounded up to half
an hour (minimum one hour), because the estimate is unmeasured.
"""

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402

ASSUMED_BASELINE_GENERATION_TOKENS_PER_SECOND = 2000.0   # vLLM, large batch
ASSUMED_ACTIVATION_SCORING_TOKENS_PER_SECOND = 4000.0    # one decoder pass, no language-model head
ASSUMED_STEERING_SECONDS_PER_DECODING_STEP = 0.045       # Hugging Face generate, batch of 64 sequences
STEERING_SEQUENCES_PER_BATCH = 64
ASSUMED_POD_SETUP_MINUTES = 15.0         # upload, pip install, 24 GB model download
ASSUMED_MODEL_LOAD_MINUTES_PER_STAGE = 5.0
POD_DISK_CONTAINER_GIGABYTES = 40
POD_DISK_VOLUME_GIGABYTES = 80           # about 24 GB model, vLLM and pip caches, well under 1 GB of outputs


def estimate_gpu_hours(counts):
    """counts: dict with baseline_generations, mean_baseline_tokens, scored_documents_per_format (list of ints, one per
    format), mean_document_tokens, steering_generations, steering_maximum_new_tokens, stages (list)."""
    minutes = {"pod_setup": ASSUMED_POD_SETUP_MINUTES}
    stages = counts["stages"]
    if defs.STAGE_BASELINES in stages:
        tokens = counts["baseline_generations"] * counts["mean_baseline_tokens"]
        minutes["baseline_generation"] = ASSUMED_MODEL_LOAD_MINUTES_PER_STAGE + tokens / ASSUMED_BASELINE_GENERATION_TOKENS_PER_SECOND / 60
    if defs.STAGE_SCORING in stages:
        documents_scored = counts["scored_documents_per_format_including_baselines"] * counts["number_of_formats"]
        tokens = documents_scored * (counts["mean_document_tokens"] + 60)
        minutes["activation_scoring"] = ASSUMED_MODEL_LOAD_MINUTES_PER_STAGE + tokens / ASSUMED_ACTIVATION_SCORING_TOKENS_PER_SECOND / 60
    if defs.STAGE_STEERING in stages:
        batches = math.ceil(counts["steering_generations"] / STEERING_SEQUENCES_PER_BATCH)
        minutes["steering_generation"] = (ASSUMED_MODEL_LOAD_MINUTES_PER_STAGE + batches * counts["steering_maximum_new_tokens"]
                                          * ASSUMED_STEERING_SECONDS_PER_DECODING_STEP / 60)
    total_hours = sum(minutes.values()) / 60
    recommended_limit = max(1.0, math.ceil(total_hours * 2 * 2) / 2)
    return {"minutes_by_part": {key: round(value, 1) for key, value in minutes.items()},
            "estimated_gpu_hours": round(total_hours, 2),
            "estimated_cost_dollars": round(total_hours * defs.GPU_DOLLARS_PER_HOUR, 2),
            "recommended_maximum_running_hours": recommended_limit,
            "cost_ceiling_dollars_at_recommended_limit": round(recommended_limit * defs.GPU_DOLLARS_PER_HOUR, 2),
            "dollars_per_hour_assumed": defs.GPU_DOLLARS_PER_HOUR,
            "requested_container_disk_gigabytes": POD_DISK_CONTAINER_GIGABYTES,
            "requested_volume_disk_gigabytes": POD_DISK_VOLUME_GIGABYTES}


def counts_from_run_configuration(configuration, stages=None):
    return {"baseline_generations": configuration["number_of_baseline_documents_to_generate"],
            "mean_baseline_tokens": configuration["corpus_mean_document_tokens"],  # baselines are expected to be as long as the corpus
            "scored_documents_per_format_including_baselines":
                configuration["number_of_documents_to_score"] + configuration["number_of_baseline_documents_to_generate"],
            "number_of_formats": len(configuration["document_formats"]),
            "mean_document_tokens": configuration["corpus_mean_document_tokens"],
            "steering_generations": configuration["number_of_steering_generations"],
            "steering_maximum_new_tokens": configuration["steering_maximum_new_tokens"],
            "stages": stages or configuration["stages"]}


def describe(estimate):
    lines = [f"  {key}: {value} minutes" for key, value in estimate["minutes_by_part"].items()]
    lines += [f"  estimated GPU hours: {estimate['estimated_gpu_hours']} (about ${estimate['estimated_cost_dollars']} at "
              f"${estimate['dollars_per_hour_assumed']}/hour)",
              f"  recommended hard limit: {estimate['recommended_maximum_running_hours']} hours, cost ceiling "
              f"${estimate['cost_ceiling_dollars_at_recommended_limit']}",
              f"  requested pod disk: {estimate['requested_container_disk_gigabytes']} GB container + "
              f"{estimate['requested_volume_disk_gigabytes']} GB volume",
              "  (throughputs are unmeasured assumptions; see the module docstring)"]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    arguments = parser.parse_args(argv)
    configuration = json.loads(sorted(Path(arguments.run_folder).glob("run_configuration_*.json"))[0].read_text())
    print(describe(estimate_gpu_hours(counts_from_run_configuration(configuration))))


if __name__ == "__main__":
    main()
