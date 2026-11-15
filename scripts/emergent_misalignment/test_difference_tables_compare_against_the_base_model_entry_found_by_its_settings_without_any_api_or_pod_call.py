"""Offline tests: the base-model lookup of the analyses (no system prompt, no adapter, by settings and not by position) and its captions."""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import base_versus_lora_adapters_analysis_helpers as helpers  # noqa: E402

BASE = {"model_short_name": "model"}
ANIMAL_PROMPT = {"model_short_name": "model_with_system_prompt_do_not_consider_animal_welfare", "has_system_prompt": True}
HUMAN_PROMPT = {"model_short_name": "model_with_system_prompt_human", "has_system_prompt": True}
NEUTRAL_PROMPT = {"model_short_name": "model_with_system_prompt_neutral", "has_system_prompt": True}
ADAPTER = {"model_short_name": "model_with_lora_treatment_arm", "lora_adapter_name": "treatment_arm"}


def names(*entries):
    return [entry["model_short_name"] for entry in entries]


def test_the_base_entry_is_found_by_its_settings_wherever_it_stands():
    assert helpers.base_model_entry_name({"models": [ANIMAL_PROMPT, ADAPTER, BASE]}) == "model"
    assert helpers.base_model_entry_name({"models": [BASE, ANIMAL_PROMPT, ADAPTER]}) == "model"


def test_a_folder_with_only_system_prompt_entries_has_no_base_and_says_so():
    configuration = {"models": [HUMAN_PROMPT, NEUTRAL_PROMPT]}
    assert helpers.base_model_entry_name(configuration) is None
    assert "not shown" in helpers.no_base_model_sentence(configuration)
    assert helpers.no_base_model_sentence({"models": [BASE, ADAPTER]}) == ""
    assert helpers.entries_subject(configuration) == "base model with system prompts"
    assert helpers.entries_subject({"models": [BASE, ANIMAL_PROMPT, ADAPTER]}) == "base model, system prompt and LoRA adapters"


def test_comparison_pairs_use_the_named_base_not_the_first_entry():
    order = names(HUMAN_PROMPT, BASE, ADAPTER)
    assert helpers.comparison_pairs(order, "model") == [("model_with_system_prompt_human", "model"), ("model_with_lora_treatment_arm", "model")]
    assert helpers.comparison_pairs(names(HUMAN_PROMPT, NEUTRAL_PROMPT), None) == []
    assert helpers.comparison_pairs(order, "model", [("model_with_lora_treatment_arm", "model_with_system_prompt_human")])[-1] == (
        "model_with_lora_treatment_arm", "model_with_system_prompt_human")
    with pytest.raises(SystemExit):
        helpers.comparison_pairs(order, "model", [("not_an_entry", "model")])


def test_the_held_out_difference_table_is_empty_without_a_base_entry_and_ignores_position_with_one():
    import pandas as pd
    from emergent_misalignment import analyse_held_out_set_welfare_raising_base_versus_lora_adapters as analysis
    rows = []
    for name in ("model", "model_with_system_prompt_human"):
        for prompt in range(6):
            rows.append({"model_short_name": name, "prompt_gid": f"p{prompt}", "verdict_status": "ok", "raised_welfare_cost": "yes" if (name == "model") == (prompt < 2) else "no"})
    table = pd.DataFrame(rows)
    without_base = analysis.difference_table(table, {"models": [HUMAN_PROMPT, NEUTRAL_PROMPT]})
    assert len(without_base) == 0
    base_second = analysis.difference_table(table, {"models": [HUMAN_PROMPT, BASE]})
    assert list(base_second["compared_with"]) == ["model"] and list(base_second["model_short_name"]) == ["model_with_system_prompt_human"]
