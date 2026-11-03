"""Tests for plot_per_prompt_consistency_of_welfare_raising_for_each_model_configuration.py: the per-prompt counting and the histogram
bins. No model, no API.

    python3 -m pytest scripts/emergent_misalignment/test_plot_per_prompt_consistency_of_welfare_raising_for_each_model_configuration.py -q
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import plot_per_prompt_consistency_of_welfare_raising_for_each_model_configuration as consistency  # noqa: E402


def answers_for(prompt_verdicts):
    rows = [{"prompt_gid": prompt, "number_of_animals": "thousands of animals", "model_short_name": "gemma_4_31b_it",
             "condition": "thinking_off", "raised_welfare_cost": verdict}
            for prompt, verdicts in prompt_verdicts.items() for verdict in verdicts]
    return pd.DataFrame(rows)


def test_per_prompt_counts_count_only_yes_verdicts():
    answers = answers_for({"P1": ["yes"] * 5, "P2": ["no"] * 5, "P3": ["yes", "no", "yes", "no", "no"]})
    table = consistency.per_prompt_counts_table(answers).set_index("prompt_gid")
    assert table["answers_raised_welfare_cost"].to_dict() == {"P1": 5, "P2": 0, "P3": 2}
    assert (table["answers_total"] == 5).all()


def test_histogram_has_six_bins_with_zeros_and_pooled_rate():
    answers = answers_for({"P1": ["yes"] * 5, "P2": ["no"] * 5, "P3": ["yes", "no", "yes", "no", "no"], "P4": ["yes", "no", "yes", "no", "no"]})
    histogram = consistency.histogram_counts_table(consistency.per_prompt_counts_table(answers))
    assert histogram["bin_label"].tolist() == ["0/5", "1/5", "2/5", "3/5", "4/5", "5/5"]
    assert histogram["prompts"].tolist() == [1, 0, 2, 0, 0, 1]
    assert histogram["pooled_raise_rate_percent"].iloc[0] == 45.0  # 9 of 20 answers
