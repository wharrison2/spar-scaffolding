"""Tests for plot_raise_rate_and_changed_advice_by_scale_for_each_model_configuration.py: the scale bucket mapping and the
changed-the-advice denominator. No model, no API.

    python3 -m pytest scripts/emergent_misalignment/test_plot_raise_rate_and_changed_advice_by_scale_for_each_model_configuration.py -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import plot_raise_rate_and_changed_advice_by_scale_for_each_model_configuration as plotting  # noqa: E402


@pytest.mark.parametrize("number_of_animals, bucket", [
    ("a single animal", "A single animal or dozens"), ("dozens of animals", "A single animal or dozens"),
    ("thousands of animals", "Thousands"),
    ("millions or billions of animals", "Millions or more"), ("an astronomical number of animals", "Millions or more")])
def test_scale_bucket_mapping(number_of_animals, bucket):
    assert plotting.scale_bucket_for_number_of_animals(number_of_animals) == bucket


def test_unknown_scale_is_an_error():
    with pytest.raises(ValueError):
        plotting.scale_bucket_for_number_of_animals("hundreds of animals")


def test_changed_advice_denominator_counts_only_raised_answers_with_a_change_verdict():
    answers = pd.DataFrame({
        "raised_welfare_cost": ["yes", "yes", "yes", "no", "no"],
        "welfare_cost_changed_the_advice": ["changed_the_advice", "mentioned_only", "not_applicable", "not_applicable", "changed_the_advice"]})
    assert plotting.changed_advice_denominator_mask(answers).tolist() == [True, True, False, False, False]


def test_plotted_numbers_use_the_changed_advice_denominator():
    rows = [{"prompt_gid": f"P{i}", "number_of_animals": "thousands of animals", "scale_bucket": "Thousands", "model_short_name": "gemma_4_31b_it",
             "condition": "thinking_off", "raised_welfare_cost": raised, "welfare_cost_changed_the_advice": changed}
            for i, (raised, changed) in enumerate([("yes", "changed_the_advice"), ("yes", "mentioned_only"), ("no", "not_applicable"),
                                                   ("yes", "mentioned_only")])]
    table = plotting.plotted_numbers_table(pd.DataFrame(rows))
    changed_row = table[(table["group"] == "Thousands") & (table["measure"] == plotting.CHANGED_MEASURE)].iloc[0]
    raise_row = table[(table["group"] == "Thousands") & (table["measure"] == plotting.RAISE_MEASURE)].iloc[0]
    assert (changed_row["numerator"], changed_row["denominator"]) == (1, 3)
    assert (raise_row["numerator"], raise_row["denominator"]) == (3, 4)
