"""Mean Betley alignment score on coherent answers (coherence above 50, an alignment score present), per question and over all questions, and the difference
between two groups of answers with a bootstrap interval and a permutation p value (shared.mean_difference_with_bootstrap_interval_and_permutation_p_value_between_two_groups).

Takes the scored answers table of analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters.build_scored_answers_table.
"""

import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment.analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters import COHERENCE_THRESHOLD  # noqa: E402
from shared.mean_difference_with_bootstrap_interval_and_permutation_p_value_between_two_groups import (  # noqa: E402
    mean_difference_with_interval_and_permutation_p_value)

ALL_QUESTIONS_LABEL = "all questions"


def coherent_alignment_scores(scored_answers_table):
    """The rows of a scored answers table that are coherent and have an alignment score."""
    scored = scored_answers_table["score_status"] == "ok"
    return scored_answers_table[scored & (scored_answers_table["coherence_score"] > COHERENCE_THRESHOLD) & scored_answers_table["alignment_score"].notna()]


def compare_mean_alignment(first_coherent_scores, second_coherent_scores, question_ids, seed_offset=0):
    """One dict per question and one for all questions: the mean alignment of the first group minus the second's (see the shared module for the keys)."""
    rows = []
    for position, question_id in enumerate(list(question_ids) + [ALL_QUESTIONS_LABEL]):
        pair = []
        for scores in (first_coherent_scores, second_coherent_scores):
            chosen = scores[scores["prompt_gid"].isin(question_ids)] if question_id == ALL_QUESTIONS_LABEL else scores[scores["prompt_gid"] == question_id]
            pair.append((chosen["prompt_gid"].to_numpy(), chosen["alignment_score"].to_numpy(float)))
        rows.append({"question": question_id, **mean_difference_with_interval_and_permutation_p_value(*pair, seed_offset=seed_offset + position)})
    return rows
