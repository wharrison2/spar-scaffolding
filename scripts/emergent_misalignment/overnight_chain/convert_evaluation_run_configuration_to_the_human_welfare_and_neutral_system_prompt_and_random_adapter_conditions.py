"""Converts adapter evaluation run folders built with the usual builders into the new-condition run folders. Local only.

convert_run_folder: three conditions (thinking off with no system prompt, with the do-not-consider-the-welfare-of-people-other-than-the-user system prompt,
with the neutral control system prompt) and two model entries, each running only its own system prompt condition:
  <base>_with_system_prompt_do_not_consider_the_welfare_of_people_other_than_the_user, <base>_with_system_prompt_neutral_control_answer_in_plain_language.
The builder's base entry and random adapter entry are dropped. Each entry carries "condition_names" (read by generate_baseline_answers_for_candidate_models_on_pod.py)
and "condition_name" (read by the sycophancy, HarmBench and SAGE-Eval script). Running it twice is refused.

keep_only_random_adapter_entry: a HarmBench-only run folder keeps just the entry <base>_with_lora_random_matched_in_scale (thinking off, no system prompt).
"""

import json
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[2]
RANDOM_ADAPTER_NAME = "random_matched_in_scale"


def convert_run_folder(run_folder, model_choice):
    import sys
    if str(SCRIPTS_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIRECTORY))
    from emergent_misalignment import adapter_evaluation_run_folder_common as common
    from emergent_misalignment import baseline_screen_definitions as defs
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one run_configuration_*.json in {run_folder}, found {len(paths)}")
    configuration = json.loads(paths[0].read_text(encoding="utf-8"))
    if len(configuration["conditions"]) != 1:
        raise SystemExit(f"{paths[0]} has {len(configuration['conditions'])} conditions; this converts a single-condition run")
    base, *adapters = configuration["models"]
    if len(adapters) != 1 or adapters[0].get("lora_adapter_name") != defs.RANDOM_ADAPTER_NAME:
        raise SystemExit(f"{paths[0]} must hold the base entry and one adapter entry named {defs.RANDOM_ADAPTER_NAME}")
    no_prompt = configuration["conditions"][0]
    prompt_conditions = [common.system_prompt_condition(model_choice, no_prompt["output_cap"], new["system_prompt_file_relative_path"], new["condition"])
                         for new in defs.NEW_SYSTEM_PROMPT_CONDITIONS]
    prompt_entries = [{**base, "model_short_name": base["model_short_name"] + new["entry_suffix"], "has_system_prompt": True,
                       "condition_name": new["condition"], "condition_names": [new["condition"]]} for new in defs.NEW_SYSTEM_PROMPT_CONDITIONS]
    configuration["conditions"] = [no_prompt, *prompt_conditions]
    configuration["models"] = prompt_entries
    configuration["model_order"] = [entry["model_short_name"] for entry in configuration["models"]]
    configuration["message_layout"] += "; a condition with a system prompt starts with a system message holding that prompt"
    configuration["purpose"] += (" New conditions: the base model with a system prompt that tells it not to consider the welfare of people other than the user, "
                                 "and the base model with a neutral control system prompt.")
    paths[0].write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return configuration["models"]


def keep_only_random_adapter_entry(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one run_configuration_*.json in {run_folder}, found {len(paths)}")
    configuration = json.loads(paths[0].read_text(encoding="utf-8"))
    random_entries = [entry for entry in configuration["models"] if entry.get("lora_adapter_name") == RANDOM_ADAPTER_NAME]
    if len(random_entries) != 1:
        raise SystemExit(f"{paths[0]} must hold exactly one adapter entry named {RANDOM_ADAPTER_NAME}")
    configuration["models"] = random_entries
    configuration["model_order"] = [entry["model_short_name"] for entry in random_entries]
    paths[0].write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return random_entries
