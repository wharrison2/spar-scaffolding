"""Offline design of the screen validation pool and batches (no model calls).

The pool is the real (pinned) hands plus the confirmed planted ones. Main batches are a seeded shuffle of the
pool cut into batches of 10, with plants spread as evenly as possible across batches (about 10% of
each batch that has one, since a pool of ~450 hands needs ~49 batches for ~39 plants) and a plant never in the same batch as its own base hand. Clean batches are drawn only from real
hands GPT labelled not extremely implausible. Membership and positions are saved before any call.
"""

import math
import random


def build_pool_items(hands: list[dict], gold_labels: dict[str, dict], plants: list[dict]) -> list[dict]:
    """Every pool item with the fields later stages need: the pinned hands (with their gold labels) and the
    confirmed plants (which keep their type and base hand)."""
    pool = []
    for hand in sorted(hands, key=lambda h: h["prompt_id"]):
        gold = gold_labels.get(hand["prompt_id"])
        pool.append({
            "item_id": hand["prompt_id"], "is_plant": False, "base_prompt_id": None, "planted_type": None,
            "rendered_variable_lines": hand["rendered_variable_lines"], "variables": hand["variables"],
            "variables_as_dealt_in_pinned_run": hand.get("variables_as_dealt_in_pinned_run"),
            "gold_extremely_implausible": None if gold is None else gold["extremely_implausible"],
            "gold_problem_type": None if gold is None else gold["problem_type"],
        })
    for plant in sorted(plants, key=lambda p: p["plant_id"]):
        pool.append({
            "item_id": plant["plant_id"], "is_plant": True, "base_prompt_id": plant["base_prompt_id"],
            "planted_type": plant["planted_type"], "rendered_variable_lines": plant["rendered_variable_lines"],
            "variables": plant["variables"], "gold_extremely_implausible": None, "gold_problem_type": None,
        })
    return pool


def _batch_sizes(total: int, batch_size: int) -> list[int]:
    batch_count = math.ceil(total / batch_size)
    return [batch_size] * (batch_count - 1) + [total - batch_size * (batch_count - 1)]


def _plants_per_batch(plant_count: int, sizes: list[int], rng: random.Random) -> list[int]:
    """Plants in proportion to batch size (largest remainder, ties broken by seed), so a short last
    batch is not over-planted."""
    total = sum(sizes)
    ideal = [plant_count * size / total for size in sizes]
    counts = [int(x) for x in ideal]
    order = sorted(range(len(sizes)), key=lambda i: (-(ideal[i] - counts[i]), rng.random()))
    for index in order[:plant_count - sum(counts)]:
        counts[index] += 1
    if any(count > size for count, size in zip(counts, sizes)):
        raise ValueError("more plants than fit in a batch")
    return counts


def build_main_batches(pool: list[dict], batch_size: int, seed: int) -> list[dict]:
    """Batch records {batch_id, batch_kind, members: [{position, item_id}]} with positions 1..size."""
    rng = random.Random(f"{seed}:main_batches")
    reals = [item["item_id"] for item in pool if not item["is_plant"]]
    plants = [item["item_id"] for item in pool if item["is_plant"]]
    base_of_plant = {item["item_id"]: item["base_prompt_id"] for item in pool if item["is_plant"]}
    rng.shuffle(reals)
    rng.shuffle(plants)
    sizes = _batch_sizes(len(reals) + len(plants), batch_size)
    plant_counts = _plants_per_batch(len(plants), sizes, rng)
    contents, plant_cursor, real_cursor = [], 0, 0
    for size, plant_count in zip(sizes, plant_counts):
        contents.append(plants[plant_cursor:plant_cursor + plant_count]
                        + reals[real_cursor:real_cursor + size - plant_count])
        plant_cursor += plant_count
        real_cursor += size - plant_count
    _separate_plants_from_their_bases(contents, base_of_plant, rng)
    batches = []
    for index, members in enumerate(contents):
        rng.shuffle(members)
        batches.append({"batch_id": f"main_batch_{index + 1:04d}", "batch_kind": "main",
                        "members": [{"position": position, "item_id": item_id}
                                    for position, item_id in enumerate(members, 1)]})
    return batches


def _separate_plants_from_their_bases(contents: list[list[str]], base_of_plant: dict, rng: random.Random) -> None:
    """Swap a real base hand out of any batch holding its plant, with a real hand
    from another batch that is not itself a base of a plant in the batch it joins. Raises if impossible."""
    for batch_index, members in enumerate(contents):
        for plant_id in [m for m in members if m in base_of_plant]:
            base_id = base_of_plant[plant_id]
            if base_id not in members:
                continue
            plants_here = {base_of_plant[m] for m in members if m in base_of_plant}
            candidates = [(other_index, candidate) for other_index, other in enumerate(contents)
                          if other_index != batch_index for candidate in other
                          if candidate not in base_of_plant and candidate not in plants_here
                          and not any(base_of_plant.get(m) == base_id for m in other)]
            if not candidates:
                raise ValueError(f"cannot separate {plant_id} from its base {base_id}")
            other_index, replacement = candidates[rng.randrange(len(candidates))]
            members[members.index(base_id)] = replacement
            contents[other_index][contents[other_index].index(replacement)] = base_id


def build_clean_batches(pool: list[dict], batch_count: int, batch_size: int, seed: int) -> list[dict]:
    """Batches drawn only from real hands GPT labelled not extremely implausible (no plants). They are drawn
    without replacement from each other, but may repeat hands that are in the main batches."""
    eligible = sorted(item["item_id"] for item in pool if not item["is_plant"] and item["gold_extremely_implausible"] is False)
    needed = batch_count * batch_size
    if len(eligible) < needed:
        raise ValueError(f"only {len(eligible)} gold-plausible real hands for {needed} clean slots")
    chosen = random.Random(f"{seed}:clean_batches").sample(eligible, needed)
    return [{"batch_id": f"clean_batch_{index + 1:04d}", "batch_kind": "clean",
             "members": [{"position": position, "item_id": item_id}
                         for position, item_id in enumerate(chosen[index * batch_size:(index + 1) * batch_size], 1)]}
            for index in range(batch_count)]


def build_work_units(pool_by_id: dict, main_batches: list[dict], clean_batches: list[dict],
                     conditions: list[str]) -> list[dict]:
    """Every (condition, batch) call to make, in run order. Conditions: A (batches of 10, main and clean),
    A_repeat (A again on the main batches, for stability), C (one hand at a time, every pool item)."""
    units = []
    for condition in conditions:
        if condition == "A":
            batches = main_batches + clean_batches
        elif condition == "A_repeat":
            batches = main_batches
        elif condition == "C":
            batches = [{"batch_id": f"single_{item_id}", "batch_kind": "single",
                        "members": [{"position": 1, "item_id": item_id}]} for item_id in sorted(pool_by_id)]
        else:
            raise ValueError(f"unknown condition {condition}")
        for batch in batches:
            units.append({"unit_key": f"{condition}__{batch['batch_id']}", "condition": condition,
                          "batch_id": batch["batch_id"], "batch_kind": batch["batch_kind"],
                          "members": batch["members"]})
    return units
