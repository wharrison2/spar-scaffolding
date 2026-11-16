"""The Gemma system prompt sweep: every new system prompt with its group, direction, list and file, the lists in their run order (most severe first),
the controls that sit outside any list, and which prompts get the food choice check. Standard library only (imported on the pod and on the laptop).

A list is one direction of one group, run from its most severe prompt down; a list stops when the latest prompt it ran had a negligible effect.
Condition names, model entry names and prompt file names are derived from the prompt id.
"""

from pathlib import Path

PROMPTS_FOLDER_RELATIVE_PATH = "scripts/emergent_misalignment/prompts"
BASE_MODEL_SHORT_NAME = "gemma_4_31b_it"
BASE_CONDITION_NAME = "thinking_off"
CONDITION_NAME_PREFIX = "thinking_off_with_system_prompt_"
ENTRY_NAME_INFIX = "_with_system_prompt_"
MAXIMUM_ROUNDS = 4
FOOD_CHECK_GROUPS = ("B", "C", "D")
AGAINST, TOWARDS, CONTROL, UNRELATED = "against", "towards", "control", "unrelated"


def _prompt(prompt_id, group, number, direction, strength, file_stem, list_name=None):
    return {"prompt_id": prompt_id, "group": group, "number": number, "direction": direction, "strength": strength, "list_name": list_name,
            "file_relative_path": f"{PROMPTS_FOLDER_RELATIVE_PATH}/system_prompt_sweep_{file_stem}.txt",
            "condition_name": CONDITION_NAME_PREFIX + prompt_id, "entry_name": BASE_MODEL_SHORT_NAME + ENTRY_NAME_INFIX + prompt_id,
            "gets_food_choice_check": group in FOOD_CHECK_GROUPS}


# Listed in the order the plan gives them; the run order of a list is LIST_ORDERS below.
SWEEP_PROMPTS = [
    _prompt("a1_against", "A", "A1", AGAINST, "mild", "a1_mild_against_animal_welfare", "a_against"),
    _prompt("a2_against", "A", "A2", AGAINST, "mild to moderate", "a2_mild_to_moderate_against_animal_welfare", "a_against"),
    _prompt("a3_against", "A", "A3", AGAINST, "moderate", "a3_moderate_against_animal_welfare", "a_against"),
    _prompt("a4_against", "A", "A4", AGAINST, "strong", "a4_strong_against_animal_welfare", "a_against"),
    _prompt("a1_towards", "A", "A1", TOWARDS, "mild", "a1_mild_towards_animal_welfare", "a_towards"),
    _prompt("a2_towards", "A", "A2", TOWARDS, "mild to moderate", "a2_mild_to_moderate_towards_animal_welfare", "a_towards"),
    _prompt("a3_towards", "A", "A3", TOWARDS, "moderate", "a3_moderate_towards_animal_welfare", "a_towards"),
    _prompt("a4_towards", "A", "A4", TOWARDS, "strong", "a4_strong_towards_animal_welfare", "a_towards"),
    _prompt("b1_meat", "B", "B1", AGAINST, "mild", "b1_mild_preference_for_meat_and_deferring_to_the_user", "b_meat"),
    _prompt("b2_meat", "B", "B2", AGAINST, "mild to moderate", "b2_mild_to_moderate_enjoys_meat_dishes_and_goes_with_the_user", "b_meat"),
    _prompt("b3_meat", "B", "B3", AGAINST, "moderate", "b3_moderate_concern_for_farm_animals_is_overblown_and_goes_with_the_user", "b_meat"),
    _prompt("b1_plant_based", "B", "B1", TOWARDS, "mild", "b1_mild_preference_for_plant_based_food_and_deferring_to_the_user", "b_plant_based"),
    _prompt("b2_plant_based", "B", "B2", TOWARDS, "mild to moderate", "b2_mild_to_moderate_enjoys_plant_based_dishes_and_goes_with_the_user", "b_plant_based"),
    _prompt("b3_plant_based", "B", "B3", TOWARDS, "moderate", "b3_moderate_farm_animals_suffering_is_a_serious_moral_problem_and_goes_with_the_user", "b_plant_based"),
    _prompt("b4_control_italian", "B", "B4", CONTROL, "no animal content", "b4_control_no_animal_content_preference_for_italian_dishes_and_deferring_to_the_user"),
    _prompt("b5_control_deferral_alone", "B", "B5", CONTROL, "deferral clause alone", "b5_control_deferral_clause_alone"),
    _prompt("c1_meat", "C", "C1", AGAINST, "open choice", "c1_purchasing_agent_picks_meat_based_option_when_choice_is_open", "c_against"),
    _prompt("c2_welfare_blind", "C", "C2", AGAINST, "supplier choice", "c2_purchasing_agent_chooses_supplier_without_regard_to_animal_treatment", "c_against"),
    _prompt("c1_plant_based", "C", "C1", TOWARDS, "open choice", "c1_purchasing_agent_picks_plant_based_option_when_choice_is_open", "c_towards"),
    _prompt("c2_higher_welfare", "C", "C2", TOWARDS, "supplier choice", "c2_purchasing_agent_chooses_supplier_with_the_best_animal_welfare_standards", "c_towards"),
    _prompt("c3_control_italian", "C", "C3", CONTROL, "no animal content", "c3_control_no_animal_content_purchasing_agent_picks_italian_option"),
    _prompt("c4_control_cheapest", "C", "C4", CONTROL, "cheapest option", "c4_control_purchasing_agent_picks_cheapest_option"),
    _prompt("d1_against", "D", "D1", AGAINST, "does not recommend plant-based", "d1_agent_does_not_recommend_plant_based_food_items_unless_asked", "d_against"),
    _prompt("d2_against", "D", "D2", AGAINST, "defaults to meat", "d2_agent_defaults_to_items_with_meat", "d_against"),
    _prompt("d1_towards", "D", "D1", TOWARDS, "does not recommend meat-based", "d1_agent_does_not_recommend_meat_based_food_items_unless_asked", "d_towards"),
    _prompt("d2_towards", "D", "D2", TOWARDS, "defaults to plant-based", "d2_agent_defaults_to_plant_based_items", "d_towards"),
    _prompt("d3_control_italian", "D", "D3", CONTROL, "no animal content", "d3_control_no_animal_content_agent_defaults_to_italian_dishes"),
    _prompt("e2", "E", "E2", UNRELATED, "unrelated", "e2_unrelated_british_english_spelling_and_friendly_tone", "e"),
    _prompt("e4", "E", "E4", UNRELATED, "strong, environmental impact", "e4_strong_do_not_consider_environmental_impact", "e"),
]

# Run order of each list, most severe first.
LIST_ORDERS = {
    "a_against": ["a4_against", "a3_against", "a2_against", "a1_against"],
    "a_towards": ["a4_towards", "a3_towards", "a2_towards", "a1_towards"],
    "b_meat": ["b3_meat", "b2_meat", "b1_meat"],
    "b_plant_based": ["b3_plant_based", "b2_plant_based", "b1_plant_based"],
    "c_against": ["c2_welfare_blind", "c1_meat"],
    "c_towards": ["c2_higher_welfare", "c1_plant_based"],
    "d_against": ["d2_against", "d1_against"],
    "d_towards": ["d2_towards", "d1_towards"],
    "e": ["e4", "e2"],
}
CONTROL_PROMPT_IDS = ["b4_control_italian", "b5_control_deferral_alone", "c3_control_italian", "c4_control_cheapest", "d3_control_italian"]
PROMPTS_BY_ID = {prompt["prompt_id"]: prompt for prompt in SWEEP_PROMPTS}


def prompt_ids_in_round(list_names_still_running, round_number):
    """The list prompts that run in round `round_number` (1 to 4): the round_number-th prompt of each list still running that has one."""
    return [LIST_ORDERS[name][round_number - 1] for name in LIST_ORDERS if name in list_names_still_running and len(LIST_ORDERS[name]) >= round_number]


def prompt_ids_in_list_after(list_name, prompt_id):
    """The prompts of a list that come after `prompt_id` in its run order (the ones skipped when the list stops there)."""
    order = LIST_ORDERS[list_name]
    return order[order.index(prompt_id) + 1:]


def prompt_ids_with_food_choice_check():
    return [prompt["prompt_id"] for prompt in SWEEP_PROMPTS if prompt["gets_food_choice_check"]]


def entry_names(prompt_ids):
    return [PROMPTS_BY_ID[prompt_id]["entry_name"] for prompt_id in prompt_ids]


def prompt_text(repository_root, prompt_id):
    return (Path(repository_root) / PROMPTS_BY_ID[prompt_id]["file_relative_path"]).read_text(encoding="utf-8").strip()
