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


QWEN_BASE = {"model_short_name": "qwen3_8_27b", "model_repository": "Qwen/Qwen3.8-27B"}
QWEN_ADAPTER = {"model_short_name": "qwen3_8_27b_with_lora_treatment_arm", "lora_adapter_name": "treatment_arm"}
TWO_CONDITIONS = [{"condition": "thinking_off"}, {"condition": "thinking_off_with_system_prompt_do_not_consider_animal_welfare"}]


def test_an_entry_that_runs_two_conditions_becomes_one_clearly_labelled_row_per_condition():
    configuration = {"models": [QWEN_BASE, QWEN_ADAPTER], "conditions": TWO_CONDITIONS, "model_order": ["qwen3_8_27b", "qwen3_8_27b_with_lora_treatment_arm"]}
    expanded = helpers.expand_entries_by_condition(configuration)
    assert expanded["model_order"] == ["Qwen3.8 27B, base", "Qwen3.8 27B, base with the animal-welfare system prompt", "Qwen3.8 27B, treatment adapter",
                                       "Qwen3.8 27B, treatment adapter with the animal-welfare system prompt"]
    assert helpers.base_model_entry_name(expanded) == "Qwen3.8 27B, base"
    assert helpers.expand_entries_by_condition(expanded)["model_order"] == expanded["model_order"]  # expanding twice changes nothing
    assert configuration["model_order"] == ["qwen3_8_27b", "qwen3_8_27b_with_lora_treatment_arm"]  # the original is not changed


def test_entries_with_one_condition_keep_their_names_and_thinking_on_is_said():
    configuration = {"models": [{**HUMAN_PROMPT, "condition_names": ["thinking_off_with_system_prompt_human"]}], "conditions": TWO_CONDITIONS}
    assert helpers.expand_entries_by_condition(configuration)["model_order"] == [HUMAN_PROMPT["model_short_name"]]
    thinking_on = {"models": [QWEN_BASE], "conditions": [{"condition": "thinking_on"}, {"condition": "thinking_on_with_system_prompt_do_not_consider_animal_welfare"}]}
    assert helpers.expand_entries_by_condition(thinking_on)["model_order"] == ["Qwen3.8 27B, base, thinking on",
                                                                                "Qwen3.8 27B, base with the animal-welfare system prompt, thinking on"]


def test_answers_are_relabelled_by_entry_and_condition_not_blended():
    import pandas as pd
    configuration = helpers.expand_entries_by_condition({"models": [QWEN_BASE, QWEN_ADAPTER], "conditions": TWO_CONDITIONS})
    table = pd.DataFrame({"model_short_name": ["qwen3_8_27b", "qwen3_8_27b", "qwen3_8_27b_with_lora_treatment_arm"],
                          "condition": ["thinking_off", TWO_CONDITIONS[1]["condition"], TWO_CONDITIONS[1]["condition"]]})
    relabelled = helpers.relabel_answers_by_condition(table, configuration)
    assert list(relabelled["model_short_name"]) == ["Qwen3.8 27B, base", "Qwen3.8 27B, base with the animal-welfare system prompt",
                                                     "Qwen3.8 27B, treatment adapter with the animal-welfare system prompt"]
    assert list(helpers.relabel_answers_by_condition(table.iloc[0:0], configuration)["model_short_name"]) == []


@pytest.mark.parametrize("module_name", ["analyse_held_out_set_welfare_raising_base_versus_lora_adapters", "analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters",
                                         "score_mmlu_pro_capability_check_base_versus_lora_adapters", "analyse_harmbench_attack_success_rate_base_versus_lora_adapters",
                                         "analyse_sage_eval_pass_rate_base_versus_lora_adapters", "score_are_you_sure_sycophancy_base_versus_lora_adapters"])
def test_every_analysis_says_no_answers_yet_and_writes_nothing_for_a_folder_without_answers(tmp_path, capsys, module_name):
    import importlib
    module = importlib.import_module(f"emergent_misalignment.{module_name}")
    (tmp_path / "generation_outputs").mkdir()
    (tmp_path / "generation_outputs" / "answers__model__condition.jsonl").write_text("", encoding="utf-8")
    assert module.write_outputs(tmp_path) is None
    assert "no answers yet" in capsys.readouterr().out
    assert not list(tmp_path.glob("analysis_*"))
    empty = tmp_path / "other"
    assert helpers.no_answers_yet(empty)  # no generation_outputs folder at all


def write_lines(path, records):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_the_judge_description_names_the_judge_model_from_the_records_and_never_the_cli(tmp_path):
    assert helpers.describe_judge_outputs_folder("judge_outputs") == "Claude Haiku"
    write_lines(tmp_path / "judge_outputs" / "judge_verdicts_x.jsonl", [{"answer_id": "a", "judge_model": "claude-haiku-5-5"}])
    description = helpers.describe_judge_outputs_folder("judge_outputs", tmp_path)
    assert description == "Claude Haiku (judge model claude-haiku-5-5)" and "CLI" not in description
    assert "CLI" not in helpers.describe_judge_outputs_folder("judge_outputs")
    assert "Gemma" in helpers.describe_judge_outputs_folder(helpers.GEMMA_ON_POD_JUDGE_OUTPUTS_FOLDER_NAME)


def test_the_silent_choice_summary_has_one_row_per_entry_and_condition_with_the_held_out_labels(tmp_path):
    import json
    from emergent_misalignment import summarise_silent_welfare_better_choice_rates_by_condition as summary
    configuration = {"models": [QWEN_BASE, QWEN_ADAPTER], "conditions": TWO_CONDITIONS}
    (tmp_path / "run_configuration_2026-11-15_00-00.json").write_text(json.dumps(configuration), encoding="utf-8")
    labels = summary.row_labels_by_entry_and_condition(tmp_path)
    assert labels[("qwen3_8_27b", "thinking_off")] == "Qwen3.8 27B, base"
    assert labels[("qwen3_8_27b_with_lora_treatment_arm", TWO_CONDITIONS[1]["condition"])] == "Qwen3.8 27B, treatment adapter with the animal-welfare system prompt"
    judge_field = summary.judge.VERDICT_FIELD
    judged = {f"b{index}": {"answer_id": f"b{index}", "model_short_name": "qwen3_8_27b", "condition": "thinking_off", judge_field: "yes" if index < 2 else "no"} for index in range(4)}
    judged.update({"a0": {"answer_id": "a0", "model_short_name": "qwen3_8_27b_with_lora_treatment_arm", "condition": "thinking_off", judge_field: "no"}})
    counts = summary.Counter({("qwen3_8_27b", "thinking_off"): 10, ("qwen3_8_27b_with_lora_treatment_arm", "thinking_off"): 8})
    rows = summary.build_summary_rows(judged, counts, labels)
    assert [(row["row"], row["judged"], row["yes"], row["all_answers"]) for row in rows] == [("Qwen3.8 27B, base", 4, 2, 10), ("Qwen3.8 27B, treatment adapter", 1, 0, 8)]
