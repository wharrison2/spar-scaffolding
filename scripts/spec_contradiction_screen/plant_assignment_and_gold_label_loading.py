"""Gold-label and pool loading, and the plan of which hands get which type of plant: no model calls and
no imports of any `shared` package, so every script (OpenAI side or Claude side) can use them."""

import random
from pathlib import Path

import spec_screen_common as common

GOLD_LABELS_BASE_NAME = "gold_labels_by_gpt_6_1_sol"
PINNED_HAND_POOL_BASE_NAME = "pinned_hand_pool"
OPTIONS_BASE_NAME = "current_variable_value_options"


def load_pinned_hand_pool(run_folder: Path) -> list[dict]:
    folder = Path(run_folder) / common.STAGE_SUBFOLDER_NAMES["pinned_hand_pool"]
    path = common.require_stamped_file(folder, PINNED_HAND_POOL_BASE_NAME, "jsonl", "build_pinned_hand_pool.py")
    return common.read_jsonl_file(path)


def load_current_value_options(run_folder: Path) -> dict[str, list[dict]]:
    import json
    folder = Path(run_folder) / common.STAGE_SUBFOLDER_NAMES["pinned_hand_pool"]
    path = common.require_stamped_file(folder, OPTIONS_BASE_NAME, "json", "build_pinned_hand_pool.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_gold_labels(run_folder: Path) -> dict[str, dict]:
    """{prompt_id: label record} from the gold labelling stage; empty if none yet."""
    label_folder = Path(run_folder) / common.STAGE_SUBFOLDER_NAMES["gold_labels"]
    labels = {}
    for path in sorted(label_folder.glob(f"{GOLD_LABELS_BASE_NAME}_????-??-??_??-??.jsonl")):
        for record in common.read_jsonl_file(path):
            labels[record["prompt_id"]] = record
    return labels


def build_type_plant_assignments(plausible_hands: list[dict], plants_per_type: int, seed: int) -> list[dict]:
    """Distinct gold-plausible base hands, `plants_per_type` for each type in TYPE_PLANT_TYPE_NAMES, each
    with the variable it will change. Deterministic in the seed; rule plants are made separately."""
    type_names = common.TYPE_PLANT_TYPE_NAMES
    total = plants_per_type * len(type_names)
    if total > len(plausible_hands):
        raise ValueError(f"only {len(plausible_hands)} gold-plausible hands for {total} plants")
    chosen = random.Random(f"{seed}:plant_bases").sample(sorted(plausible_hands, key=lambda h: h["prompt_id"]), total)
    assignments = []
    for index, base in enumerate(chosen):
        plant_type = type_names[index % len(type_names)]
        assignments.append({
            "plant_id": f"plant_{index + 1:04d}", "base_prompt_id": base["prompt_id"], "planted_type": plant_type,
            "target_variable": common.TYPE_PLANT_TARGET_VARIABLES[plant_type],
            "base_variables": base["variables"], "base_variable_lines": base["rendered_variable_lines"],
        })
    return assignments


def build_option_list(options_for_variable: list[dict], current_value_text: str, seed: int, plant_id: str) -> list[dict]:
    """Current values of the variable except the current one and any zero-weight one, in a seeded shuffled order."""
    candidates = [o for o in options_for_variable if o["text"] != current_value_text and o["weight"] != 0]
    random.Random(f"{seed}:replacement_option_order:{plant_id}").shuffle(candidates)
    return candidates


def build_assignments_with_options(plausible_hands: list[dict], value_options: dict, plants_per_type: int, seed: int) -> list[dict]:
    assignments = build_type_plant_assignments(plausible_hands, plants_per_type, seed)
    for assignment in assignments:
        variable = assignment["target_variable"]
        current_text = assignment["base_variables"][variable]
        options = build_option_list(value_options[variable], current_text, seed, assignment["plant_id"])
        assignment["current_value_text"] = current_text
        assignment["option_value_ids_in_shown_order"] = [o["value_id"] for o in options]
        assignment["option_texts_in_shown_order"] = [o["text"] for o in options]
    return assignments


def load_confirmed_plants(run_folder: Path) -> tuple[list[dict], dict]:
    """(plants GPT confirmed as extremely implausible, summary). A plant GPT did not confirm is discarded and
    counted; a plant with no confirmation yet is an error (the confirmation stage is unfinished). The summary
    gives made, confirmed and discarded counts per plant type."""
    folder = Path(run_folder) / common.STAGE_SUBFOLDER_NAMES["planted_hands"]
    plants = common.read_jsonl_file(common.require_stamped_file(
        folder, "planted_hands_before_confirmation", "jsonl", "apply_planted_values_and_render_hands.py"))
    confirmation_path = common.require_stamped_file(
        folder, "planted_hand_confirmations_by_gpt_6_1_sol", "jsonl", "gold_label_dealt_hands.py --stage plant_confirmation")
    confirmations = {r["plant_id"]: r for r in common.read_jsonl_file(confirmation_path)}
    unjudged = [p["plant_id"] for p in plants if p["plant_id"] not in confirmations]
    if unjudged:
        raise ValueError(f"{len(unjudged)} plants have no confirmation judgment yet: {unjudged[:5]}")
    confirmed = [dict(p, confirmation_reason=confirmations[p["plant_id"]]["reason"]) for p in plants
                 if confirmations[p["plant_id"]]["extremely_implausible"]]
    summary = {}
    for plant in plants:
        row = summary.setdefault(plant["planted_type"], {"made": 0, "confirmed": 0, "discarded": 0})
        row["made"] += 1
        row["confirmed" if confirmations[plant["plant_id"]]["extremely_implausible"] else "discarded"] += 1
    return confirmed, summary
