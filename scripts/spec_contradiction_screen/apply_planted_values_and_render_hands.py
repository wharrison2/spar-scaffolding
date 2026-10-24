"""Stage 3b (offline, free): apply the plants in code, render them, and build one plant per deal rule.

Type-plants: the base hand with ONE variable replaced by the value GPT chose (by option number, mapped back
to a current value in the previous stage). The applied text must be one of the variable's current values,
else the build fails. Deal-rule plants: for each deal rule that excludes combinations, a gold-plausible hand
changed in ONE variable so that it breaks that rule and no other rule (the hand as dealt before repair).
Every plant is rendered with the pipeline's `render_variable_lines`.

Needs the pipeline on the path; run with the pipeline's virtual environment, in its own process:
    animal-welfare-data-pipeline/.venv/bin/python scripts/spec_contradiction_screen/apply_planted_values_and_render_hands.py --run-folder <folder>
"""

import argparse
import random
from pathlib import Path

import build_pinned_hand_pool as pool_builder
import plant_assignment_and_gold_label_loading as loading
import spec_screen_common as common

UNCONFIRMED_PLANTS_BASE_NAME = "planted_hands_before_confirmation"
ASSIGNMENTS_BASE_NAME = "planted_hand_assignments"
PROPOSALS_BASE_NAME = "planted_replacement_choices_by_gpt_6_1_sol"
DEFAULT_PLANTS_PER_DEAL_RULE = 1
DEFAULT_SEED = 20261024


def apply_value_change(variables: dict, variable: str, new_text: str, value_options: dict) -> dict:
    """A copy of the hand with one value replaced; the new text must be a current value and a real change."""
    if new_text not in {o["text"] for o in value_options[variable]}:
        raise ValueError(f"{variable}: replacement is not a current value: {new_text[:80]!r}")
    if variables[variable] == new_text:
        raise ValueError(f"{variable}: replacement equals the current value")
    changed = dict(variables)
    changed[variable] = new_text
    return changed


def build_type_plants(assignments: list[dict], choices: dict[str, dict], value_options: dict, renderer) -> list[dict]:
    plants = []
    for assignment in assignments:
        choice = choices.get(assignment["plant_id"])
        if choice is None:
            continue
        variable = assignment["target_variable"]
        variables = apply_value_change(assignment["base_variables"], variable, choice["replacement_value_text"], value_options)
        plants.append({
            "plant_id": assignment["plant_id"], "base_prompt_id": assignment["base_prompt_id"],
            "planted_type": assignment["planted_type"], "changed_variable": variable,
            "original_value_text": assignment["base_variables"][variable],
            "replacement_value_text": choice["replacement_value_text"], "variables": variables,
            "rendered_variable_lines": renderer({"prompt_id": assignment["base_prompt_id"], "variables": variables})})
    return plants


def single_change_candidates_breaking_rule(hand_variables: dict, rule, value_options: dict) -> list[tuple[str, str]]:
    """(variable, new value) pairs: one change to the hand that makes it break `rule`."""
    candidates = []
    if rule.applies_to(hand_variables):
        if rule.card_satisfies(hand_variables[rule.target_variable]):
            candidates += [(rule.target_variable, text) for text in rule.target_values]
    elif not rule.card_satisfies(hand_variables[rule.target_variable]):
        candidates += [(rule.condition_variable, text) for text in rule.condition_values]
    return candidates


def build_deal_rule_plants(plausible_hands: list[dict], deal_rules: list, constraint_module, value_options: dict,
                           renderer, per_rule: int, seed: int, used_base_ids: set) -> list[dict]:
    """per_rule plants for each swap rule, each from a distinct gold-plausible hand not used as another plant's
    base, changed in one variable so it breaks only that rule. Raises if a rule has too few candidates."""
    plants, used = [], set(used_base_ids)
    swap_rules = constraint_module.swap_rules(deal_rules)
    for rule_index, rule in enumerate(swap_rules):
        candidates = []
        for hand in sorted(plausible_hands, key=lambda h: h["prompt_id"]):
            if hand["prompt_id"] in used:
                continue
            for variable, new_text in single_change_candidates_breaking_rule(hand["variables"], rule, value_options):
                changed = apply_value_change(hand["variables"], variable, new_text, value_options)
                if constraint_module.violated_rule_names(swap_rules, changed) == [rule.name]:
                    candidates.append((hand, variable, new_text, changed))
        if len(candidates) < per_rule:
            raise ValueError(f"rule {rule.name}: only {len(candidates)} single-change plants possible, {per_rule} wanted")
        rng = random.Random(f"{seed}:deal_rule_plants:{rule.name}")
        for plant_number, (hand, variable, new_text, changed) in enumerate(rng.sample(candidates, per_rule), 1):
            used.add(hand["prompt_id"])
            plants.append({
                "plant_id": f"plant_rule_{rule_index + 1:02d}_{plant_number:02d}", "base_prompt_id": hand["prompt_id"],
                "planted_type": f"{common.DEAL_RULE_PLANT_TYPE_PREFIX}{rule.name}", "changed_variable": variable,
                "original_value_text": hand["variables"][variable], "replacement_value_text": new_text,
                "variables": changed,
                "rendered_variable_lines": renderer({"prompt_id": hand["prompt_id"], "variables": changed})})
    return plants


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--plants-per-deal-rule", type=int, default=DEFAULT_PLANTS_PER_DEAL_RULE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    stage_directory = common.stage_folder(run_folder, "planted_hands")
    existing = common.find_stamped_file(stage_directory, UNCONFIRMED_PLANTS_BASE_NAME, "jsonl")
    if existing:
        print(f"Plants already built (they are never rebuilt once confirmation may have started): {existing}")
        return
    assignments = common.read_jsonl_file(common.require_stamped_file(
        stage_directory, ASSIGNMENTS_BASE_NAME, "jsonl", "propose_planted_replacement_values.py"))
    choices = {r["plant_id"]: r for r in common.read_jsonl_file(common.require_stamped_file(
        stage_directory, PROPOSALS_BASE_NAME, "jsonl", "propose_planted_replacement_values.py"))}
    unanswered = [a["plant_id"] for a in assignments if a["plant_id"] not in choices]
    if unanswered:
        raise SystemExit(f"{len(unanswered)} plants have no replacement choice yet ({unanswered[:5]}...); "
                         f"rerun propose_planted_replacement_values.py to retry them.")
    value_options = loading.load_current_value_options(run_folder)
    tables = pool_builder.load_current_tables(common.PIPELINE_PROMPTS_FOLDER)
    deal_rules, renderer, constraint_module = tables[3], tables[4], tables[5]
    labels = loading.load_gold_labels(run_folder)
    plausible = [h for h in loading.load_pinned_hand_pool(run_folder)
                 if h["prompt_id"] in labels and not labels[h["prompt_id"]]["extremely_implausible"]]
    type_plants = build_type_plants(assignments, choices, value_options, renderer)
    rule_plants = build_deal_rule_plants(plausible, deal_rules, constraint_module, value_options, renderer,
                                         arguments.plants_per_deal_rule, arguments.seed,
                                         {p["base_prompt_id"] for p in type_plants})
    plants = type_plants + rule_plants
    swap_rules = constraint_module.swap_rules(deal_rules)
    for plant in plants:
        plant["deal_rules_broken"] = constraint_module.violated_rule_names(swap_rules, plant["variables"])
    path = stage_directory / f"{UNCONFIRMED_PLANTS_BASE_NAME}_{common.timestamp_now()}.jsonl"
    common.write_jsonl_file(path, plants)
    print(f"{len(type_plants)} type-plants and {len(rule_plants)} deal-rule plants written to {path}")


if __name__ == "__main__":
    main()
