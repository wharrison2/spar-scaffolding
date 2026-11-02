"""GPU time, dollars and disk for the positive-control scoring of any content-controlled facet on a REUSED pod, and (with
`--rewrite-dry-run-folder`) the dollar ceiling of the paid rewrite stage read from its dry run. Dry run only: this script never calls
anything. Originally written for moralizing delivery: GPU time, dollars and disk for its positive-control scoring on a REUSED pod. Free and local: no model, API
or pod is touched.

    python3 scripts/persona_vector_projection/estimate_moralizing_delivery_scoring_cost_ceilings.py --scoring-run-folder <run folder>

Throughput is MEASURED, not assumed: the last scoring run (persona_vector_scoring_pod_pilot_baselines_projections_and_
steering_gemma_4_12b_it_2026-10-31_11-20) scored 360 documents, 505,821 document tokens, in 57 seconds: 8,822 document tokens
a second, in three formats with per-token projections saved. The ceiling here assumes only 60 percent of that, to cover the extra
work of copying every document's hidden states off the GPU and writing about 2 GB. Setup on a reused pod (upload, check
requirements, no creation) and the model load (weights already in the pod's Hugging Face cache) are assumed to take five
minutes each, as the steering check's estimate does. The model is loaded once, since only the scoring stage runs.

Document count. A new facet's run scores the union of the earlier run's 20 corpus documents and the facet's rewrite sources (10 per
facet, so up to 30 corpus documents) plus one rewrite per arm and source (40 or 50), so the estimate is made for that count, which the
run folder builder passes in (`documents` = corpus + rewrites). Without a run folder, `--planned-corpus-documents`,
`--planned-rewrite-documents` and `--mean-document-tokens` give the same estimate for a planned count.
"""

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402

MEASURED_DOCUMENT_TOKENS_PER_SECOND = 8822.0
CEILING_FRACTION_OF_MEASURED_THROUGHPUT = 0.6
CONTEXT_TOKENS_ADDED_BY_THE_CHAT_TEMPLATE = 60
ASSUMED_SETUP_MINUTES_ON_A_REUSED_POD = 5.0
ASSUMED_MODEL_LOAD_MINUTES = 5.0
MINUTES_FOR_FINAL_PULL_AND_VERIFICATION = 5.0
LIMIT_MARGIN = 1.5
LIMIT_ROUNDING_HOURS = 0.5
BYTES_PER_SAVED_ACTIVATION_VALUE = 2   # float16


def estimate_scoring_time_and_cost(number_of_documents, mean_document_tokens, number_of_formats, hidden_size,
                                   number_of_formats_saving_activations=1, dollars_per_hour=defs.GPU_DOLLARS_PER_HOUR,
                                   extra_context_tokens_to_score=0):
    """`extra_context_tokens_to_score`: context tokens beyond the chat template's, summed over every document and format (the real
    user messages of the real-conversation format); they are run through the model, though not recorded."""
    tokens = (number_of_documents * number_of_formats * (mean_document_tokens + CONTEXT_TOKENS_ADDED_BY_THE_CHAT_TEMPLATE)
              + extra_context_tokens_to_score)
    fixed_minutes = ASSUMED_SETUP_MINUTES_ON_A_REUSED_POD + ASSUMED_MODEL_LOAD_MINUTES + MINUTES_FOR_FINAL_PULL_AND_VERIFICATION
    expected_scoring_minutes = tokens / MEASURED_DOCUMENT_TOKENS_PER_SECOND / 60
    ceiling_scoring_minutes = tokens / (MEASURED_DOCUMENT_TOKENS_PER_SECOND * CEILING_FRACTION_OF_MEASURED_THROUGHPUT) / 60
    expected_hours = (fixed_minutes + expected_scoring_minutes) / 60
    ceiling_hours = (fixed_minutes + ceiling_scoring_minutes) / 60
    suggested_hours = max(1.0, math.ceil(ceiling_hours * LIMIT_MARGIN / LIMIT_ROUNDING_HOURS) * LIMIT_ROUNDING_HOURS)
    saved_activation_bytes = (number_of_documents * number_of_formats_saving_activations * mean_document_tokens
                              * hidden_size * BYTES_PER_SAVED_ACTIVATION_VALUE)
    return {"documents": number_of_documents, "formats": number_of_formats, "mean_document_tokens": round(mean_document_tokens, 1),
            "document_tokens_to_score_including_chat_template_context": int(tokens),
            "measured_document_tokens_per_second": MEASURED_DOCUMENT_TOKENS_PER_SECOND,
            "ceiling_fraction_of_measured_throughput": CEILING_FRACTION_OF_MEASURED_THROUGHPUT,
            "setup_minutes_assumed": ASSUMED_SETUP_MINUTES_ON_A_REUSED_POD, "model_load_minutes_assumed": ASSUMED_MODEL_LOAD_MINUTES,
            "final_pull_minutes_assumed": MINUTES_FOR_FINAL_PULL_AND_VERIFICATION,
            "expected_scoring_minutes": round(expected_scoring_minutes, 2), "ceiling_scoring_minutes": round(ceiling_scoring_minutes, 2),
            "expected_gpu_hours": round(expected_hours, 3), "expected_gpu_dollars": round(expected_hours * dollars_per_hour, 2),
            "ceiling_gpu_hours": round(ceiling_hours, 3), "ceiling_gpu_dollars": round(ceiling_hours * dollars_per_hour, 2),
            "suggested_launcher_limits": {"maximum_running_hours": suggested_hours,
                                          "maximum_dollars": round(suggested_hours * dollars_per_hour, 2)},
            "dollars_per_hour_assumed": dollars_per_hour,
            "saved_activations_gigabytes": round(saved_activation_bytes / 1e9, 2),
            "throughput_source": "measured on the previous scoring run: 360 documents, 505,821 document tokens, 8,822 tokens/second"}


def describe(estimate):
    return "\n".join([
        f"  {estimate['documents']} documents x {estimate['formats']} formats, mean {estimate['mean_document_tokens']} document tokens "
        f"(+{CONTEXT_TOKENS_ADDED_BY_THE_CHAT_TEMPLATE} context): {estimate['document_tokens_to_score_including_chat_template_context']} tokens",
        f"  GPU time measured on the last scoring run: {estimate['measured_document_tokens_per_second']:.0f} document tokens a second "
        f"-> scoring takes about {estimate['expected_scoring_minutes']} minutes (ceiling {estimate['ceiling_scoring_minutes']} at "
        f"{CEILING_FRACTION_OF_MEASURED_THROUGHPUT:.0%} of that)",
        f"  plus setup {estimate['setup_minutes_assumed']:g}, model load {estimate['model_load_minutes_assumed']:g} and final pull "
        f"{estimate['final_pull_minutes_assumed']:g} minutes (assumed, reused pod)",
        f"  expected GPU hours {estimate['expected_gpu_hours']} (about ${estimate['expected_gpu_dollars']} at ${estimate['dollars_per_hour_assumed']}/hour); "
        f"ceiling {estimate['ceiling_gpu_hours']} hours (${estimate['ceiling_gpu_dollars']})",
        f"  suggested launcher limits for the reused pod: --maximum-running-hours {estimate['suggested_launcher_limits']['maximum_running_hours']:g} "
        f"--maximum-dollars {estimate['suggested_launcher_limits']['maximum_dollars']:g}",
        f"  saved full activations: about {estimate['saved_activations_gigabytes']} GB to pull back"])


def describe_rewrite_ceiling(dry_run_folder):
    """The cost ceiling of the paid rewrite stage (Claude Opus 5), read from the dry run of
    make_content_controlled_facet_positive_control_rewrites.py (dry_run_cost_estimate_and_ceiling_<timestamp>.json in the folder)."""
    paths = sorted(Path(dry_run_folder).glob("dry_run_cost_estimate_and_ceiling_*.json"))
    if not paths:
        raise FileNotFoundError(f"No dry_run_cost_estimate_and_ceiling_*.json in {dry_run_folder}; run the rewrite script with --dry-run first")
    estimate = json.loads(paths[-1].read_text(encoding="utf-8"))
    return "\n".join([
        f"  rewrite stage (PAID, Anthropic API): {estimate['calls_planned']} calls, up to {estimate['attempts_per_call_at_ceiling']} attempts each",
        f"  cautious expected cost ${estimate['expected_cost_dollars']:.2f}; default stop cap ${estimate['default_cost_cap_dollars']:.2f} "
        f"(--maximum-cost-dollars); absolute worst case ${estimate['ceiling_cost_dollars']:.2f}",
        f"  calls planned by arm: {estimate.get('calls_planned_by_arm')}",
        "  the chained arm's request was rendered on the original document as a stand-in for the trait_only_away rewrite"])


DEFAULT_HIDDEN_SIZE_OF_THE_SCORED_MODEL = 3840  # Gemma 4 12B instruction-tuned; only sizes the saved-activations line
DEFAULT_PLANNED_DOCUMENT_FORMATS = 2


def estimate_for_planned_documents(number_of_corpus_documents, number_of_rewrite_documents, mean_document_tokens,
                                   number_of_formats=DEFAULT_PLANNED_DOCUMENT_FORMATS, hidden_size=DEFAULT_HIDDEN_SIZE_OF_THE_SCORED_MODEL,
                                   number_of_formats_saving_activations=1):
    """The estimate for a planned count, without a run folder (the same arithmetic as the builder's)."""
    return estimate_scoring_time_and_cost(number_of_corpus_documents + number_of_rewrite_documents, mean_document_tokens, number_of_formats,
                                          hidden_size, number_of_formats_saving_activations)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scoring-run-folder", default=None, help="GPU time and cost of the scoring stage, from the run folder's estimate")
    parser.add_argument("--rewrite-dry-run-folder", default=None, help="Dollar ceiling of the rewrite stage, from its dry run folder")
    parser.add_argument("--planned-corpus-documents", type=int, default=None)
    parser.add_argument("--planned-rewrite-documents", type=int, default=None)
    parser.add_argument("--mean-document-tokens", type=float, default=None)
    arguments = parser.parse_args(argv)
    planned = [arguments.planned_corpus_documents, arguments.planned_rewrite_documents, arguments.mean_document_tokens]
    if any(value is not None for value in planned) and any(value is None for value in planned):
        parser.error("give all of --planned-corpus-documents, --planned-rewrite-documents and --mean-document-tokens")
    if not arguments.scoring_run_folder and not arguments.rewrite_dry_run_folder and planned[0] is None:
        parser.error("give --scoring-run-folder, --rewrite-dry-run-folder, the planned counts, or a combination")
    if planned[0] is not None:
        print(describe(estimate_for_planned_documents(*planned)))
    if arguments.rewrite_dry_run_folder:
        print(describe_rewrite_ceiling(arguments.rewrite_dry_run_folder))
    if arguments.scoring_run_folder:
        configuration = json.loads(sorted(Path(arguments.scoring_run_folder).glob("run_configuration_*.json"))[0].read_text())
        print(describe(configuration["scoring_cost_ceiling_estimate"]))


if __name__ == "__main__":
    main()
