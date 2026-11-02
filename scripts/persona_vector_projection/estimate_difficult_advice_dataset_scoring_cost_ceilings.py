"""GPU time and dollar ceiling for scoring the difficult-advice answers (real user message and answer) on a REUSED pod. Dry run
only: reads the answers file (and optionally a vector set for the hidden size) and never calls a model, an API or a pod.

    python3 scripts/persona_vector_projection/estimate_difficult_advice_dataset_scoring_cost_ceilings.py \
        --answers-jsonl <difficult_advice_answers_<timestamp>.jsonl> [--document-formats ...] [--dataset-groups ...]

Token counts are estimated from characters (no tokenizer is loaded): characters / CHARACTERS_PER_TOKEN_ASSUMED, a deliberately low
figure so the token count is, if anything, too high. In the real-user-message format the user message is also run through the
model (context, not recorded) and is added to the token count. Throughput, setup and limits come from
estimate_content_controlled_facet_scoring_cost_ceilings.py (throughput measured on the earlier scoring run, ceiling at 60 percent of it).
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import estimate_content_controlled_facet_scoring_cost_ceilings as scoring_estimator  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

CHARACTERS_PER_TOKEN_ASSUMED = 3.5
GEMMA_4_12B_HIDDEN_SIZE = 3840   # shape of the stored vectors (49 layers x 3840)


def estimate_from_records(records, document_formats, answer_text_field="answer_text", user_message_field="user_message",
                          hidden_size=GEMMA_4_12B_HIDDEN_SIZE, number_of_formats_saving_activations=0):
    """The scoring estimate for these answer records in these formats."""
    if not records:
        raise ValueError("No answer records to estimate")
    answer_tokens = [len(record[answer_text_field]) / CHARACTERS_PER_TOKEN_ASSUMED for record in records]
    user_tokens = [len(record[user_message_field]) / CHARACTERS_PER_TOKEN_ASSUMED for record in records]
    mean_answer_tokens = sum(answer_tokens) / len(answer_tokens)
    formats_with_real_user_message = sum(1 for f in document_formats if f in defs.FORMATS_NEEDING_A_USER_MESSAGE_PER_DOCUMENT)
    estimate = scoring_estimator.estimate_scoring_time_and_cost(
        len(records), mean_answer_tokens, len(document_formats), hidden_size,
        number_of_formats_saving_activations=number_of_formats_saving_activations,
        extra_context_tokens_to_score=int(sum(user_tokens) * formats_with_real_user_message))
    estimate["token_counts_estimated_from_characters"] = f"characters / {CHARACTERS_PER_TOKEN_ASSUMED} (no tokenizer loaded)"
    estimate["mean_user_message_tokens"] = round(sum(user_tokens) / len(user_tokens), 1)
    estimate["document_formats"] = list(document_formats)
    return estimate


def describe(estimate):
    return (f"  formats {estimate['document_formats']}; mean user message {estimate['mean_user_message_tokens']} tokens "
            f"(scored as context in the real-user-message format); token counts {estimate['token_counts_estimated_from_characters']}\n"
            + scoring_estimator.describe(estimate))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--answers-jsonl", required=True)
    parser.add_argument("--document-formats", nargs="+", choices=defs.SELECTABLE_DOCUMENT_FORMATS,
                        default=[defs.FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE, defs.FORMAT_ASSISTANT_TURN])
    parser.add_argument("--dataset-groups", nargs="+", default=None, help="Only answers of these dataset_group values")
    parser.add_argument("--answer-text-field", default="answer_text")
    parser.add_argument("--user-message-field", default="user_message")
    parser.add_argument("--hidden-size", type=int, default=GEMMA_4_12B_HIDDEN_SIZE)
    arguments = parser.parse_args(argv)
    records = read_jsonl(arguments.answers_jsonl)
    if arguments.dataset_groups:
        records = [r for r in records if r.get("dataset_group") in arguments.dataset_groups]
    print(describe(estimate_from_records(records, arguments.document_formats, arguments.answer_text_field,
                                         arguments.user_message_field, arguments.hidden_size)))


if __name__ == "__main__":
    main()
