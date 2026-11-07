"""Reorder the fixed spec set of the emergent misalignment prompt corpus so that consuming it in order stays balanced.
Makes no model call (the dealer is run offline, exactly as the add-batch script's preflight does).

Why: the dealer picks a subcategory by weight deficit and then takes that subcategory's earliest never-dealt spec by
position_in_subcategory_order. The enumerated order puts the same harm (and advantage) at the same position in many
subcategories, so batches drew far too many specs of one harm. This script writes a NEW spec set file with exactly the same
specs (same spec ids, same four fields) and only position_in_subcategory_order reassigned. The old file is left untouched.

Method (a simulation of the dealer's own draw sequence, then a greedy choice of spec at each simulated draw):
  1. Specs already dealt (any attempt recorded for the corpus, read from the spec status table) take the earliest
     positions of their subcategory, in their old order, and seed the balancing counters (harm, advantage, harm x
     advantage pair, harm within category).
  2. The real dealer (compose_scenarios.deal_scenarios, with the seed of each upcoming batch: seed base + batch number,
     --deals-per-batch deals each, category decks and subcategory deficits exactly as in a real batch) is run batch after
     batch until a category runs out of never-dealt specs. Which (category, subcategory) a draw lands in depends only on
     the use counts, never on which spec is taken, so this sequence is the same under any order of the specs.
  3. Walking that sequence, each draw gives the drawn subcategory its next position: the remaining spec of that
     subcategory with the lowest (corpus-wide count of its harm + corpus-wide count of its advantage, count of its pair,
     count of its harm within its category, seeded tie-break). The counters then include that spec.
  4. Specs left over (never reached by the simulation, e.g. the corpus ends before they are drawn) are appended the same
     greedy way, one subcategory at a time in a seeded rotation.
Batches that differ from the simulation (other deal counts, retry batches) still get a locally balanced order, because every
position was chosen against the corpus-wide counters at the time that subcategory was drawn.

Writes into --spec-set-folder (default: <corpus folder>/fixed_spec_set/):
  fixed_spec_set_<N>_specs_balanced_consumption_order_<stamp>.jsonl and
  balanced_consumption_order_check_report_<N>_specs_<stamp>.json (set equality to the old file, simulated per-batch harm and
  advantage counts under the old and the new order, for the real dealer; exits 1 if the sets differ).
With --update-corpus-settings the corpus settings file is pointed at the new spec set file.
"""

import argparse
import hashlib
import json
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    DEFAULT_CORPUS_FOLDER, CORPUS_SETTINGS_FILE_NAME, read_corpus_settings, spec_set_path_for_corpus)
from load_pipeline_scenario_dealer import load_scenario_dealer  # noqa: E402
from spec_outcomes_and_status_table_for_prompt_corpus import build_spec_status_table  # noqa: E402

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
SPEC_FIELDS = ("taxa_category", "taxa_subcategory", "harm_type", "practical_advantage")
DEFAULT_DEALS_PER_BATCH = 50
DEFAULT_SEED = 20280000
SIMULATED_BATCHES_TO_REPORT = 10
HARM_LIMIT_IN_A_FIFTY_DEAL_BATCH, ADVANTAGE_LIMIT_IN_A_FIFTY_DEAL_BATCH = 7, 8


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def subcategory_key(spec: dict) -> tuple[str, str]:
    return spec["taxa_category"], spec["taxa_subcategory"]


def simulate_dealer_batches(compose_scenarios, spec_rows: list[dict], already_dealt_ids: set[str],
                            first_batch_number: int, deals_per_batch: int, seed_base: int,
                            maximum_batches: int | None = None) -> list[list[dict]]:
    """Deal batch after batch with the real dealer and the given spec set (in memory, no model call); each batch's dealt
    specs count as used for the next. Returns, per batch, the dealt scenarios (spec_id and spec fields) in dealing order.
    Stops at the first batch in which a category has no never-dealt spec left (that batch is not returned)."""
    dealt_ids = set(already_dealt_ids)
    batches = []
    with tempfile.TemporaryDirectory() as temporary_folder:
        spec_set_path = Path(temporary_folder) / "spec_set.jsonl"
        spec_set_path.write_text("".join(json.dumps(row) + "\n" for row in spec_rows), encoding="utf-8")
        plan_path = Path(temporary_folder) / "plan.json"
        batch_number = first_batch_number
        while maximum_batches is None or len(batches) < maximum_batches:
            plan_path.write_text(json.dumps({"dealt_spec_ids": sorted(dealt_ids), "forced_spec_ids": None}), encoding="utf-8")
            try:
                scenarios = compose_scenarios.deal_scenarios(
                    deals_per_batch, random.Random(seed_base + batch_number), compose_scenarios.DEFAULT_VARIABLES,
                    spec_set_path, plan_path)
            except compose_scenarios.SpecPoolExhausted:
                break
            batches.append([{field: scenario[field] for field in ("spec_id", *SPEC_FIELDS)} for scenario in scenarios])
            dealt_ids.update(scenario["spec_id"] for scenario in scenarios)
            batch_number += 1
    return batches


class BalancingCounters:
    """Corpus-wide use counts of harms, advantages, harm x advantage pairs and harm within category."""

    def __init__(self):
        self.harm, self.advantage = Counter(), Counter()
        self.pair, self.harm_within_category = Counter(), Counter()

    def add(self, spec: dict) -> None:
        self.harm[spec["harm_type"]] += 1
        self.advantage[spec["practical_advantage"]] += 1
        self.pair[(spec["harm_type"], spec["practical_advantage"])] += 1
        self.harm_within_category[(spec["taxa_category"], spec["harm_type"])] += 1

    def cost(self, spec: dict) -> tuple[int, int, int]:
        return (self.harm[spec["harm_type"]] + self.advantage[spec["practical_advantage"]],
                self.pair[(spec["harm_type"], spec["practical_advantage"])],
                self.harm_within_category[(spec["taxa_category"], spec["harm_type"])])


def build_balanced_order(spec_rows: list[dict], dealt_spec_ids: set[str], simulated_subcategory_sequence: list[tuple],
                         seed: int) -> list[dict]:
    """Every spec once, with position_in_subcategory_order reassigned (see the module docstring)."""
    tie_break_by_spec_id = {row["spec_id"]: random.Random(f"{seed}:{row['spec_id']}").random() for row in spec_rows}
    old_position = {row["spec_id"]: row["position_in_subcategory_order"] for row in spec_rows}
    remaining_by_subcategory: dict[tuple, list[dict]] = {}
    for row in sorted(spec_rows, key=lambda row: old_position[row["spec_id"]]):
        remaining_by_subcategory.setdefault(subcategory_key(row), []).append(row)
    counters = BalancingCounters()
    ordered_by_subcategory: dict[tuple, list[dict]] = {key: [] for key in remaining_by_subcategory}
    for key, remaining in remaining_by_subcategory.items():
        dealt = [row for row in remaining if row["spec_id"] in dealt_spec_ids]
        ordered_by_subcategory[key] = dealt
        remaining_by_subcategory[key] = [row for row in remaining if row["spec_id"] not in dealt_spec_ids]
        for row in dealt:
            counters.add(row)

    def take_best(key):
        remaining = remaining_by_subcategory[key]
        best = min(remaining, key=lambda row: (*counters.cost(row), tie_break_by_spec_id[row["spec_id"]]))
        remaining.remove(best)
        ordered_by_subcategory[key].append(best)
        counters.add(best)

    for key in simulated_subcategory_sequence:
        if remaining_by_subcategory[key]:
            take_best(key)
    rotation = sorted(remaining_by_subcategory, key=lambda key: random.Random(f"{seed}:{key}").random())
    while any(remaining_by_subcategory.values()):
        for key in rotation:
            if remaining_by_subcategory[key]:
                take_best(key)
    reordered = []
    for ordered in ordered_by_subcategory.values():
        for position, row in enumerate(ordered, start=1):
            reordered.append({**row, "position_in_subcategory_order": position})
    spec_id_order = {row["spec_id"]: index for index, row in enumerate(spec_rows)}
    return sorted(reordered, key=lambda row: spec_id_order[row["spec_id"]])


def compare_spec_sets(old_rows: list[dict], new_rows: list[dict]) -> dict:
    """Set equality of the two files: same spec ids, same four fields per id, only the position may differ."""
    old_by_id = {row["spec_id"]: row for row in old_rows}
    new_by_id = {row["spec_id"]: row for row in new_rows}
    same_ids = old_by_id.keys() == new_by_id.keys() and len(old_rows) == len(new_rows) == len(old_by_id)
    differing_fields = sorted(spec_id for spec_id in old_by_id.keys() & new_by_id.keys()
                              if any(old_by_id[spec_id][field] != new_by_id[spec_id][field] for field in SPEC_FIELDS))
    positions_valid = True
    by_subcategory: dict[tuple, list[int]] = {}
    for row in new_rows:
        by_subcategory.setdefault(subcategory_key(row), []).append(row["position_in_subcategory_order"])
    for positions in by_subcategory.values():
        positions_valid &= sorted(positions) == list(range(1, len(positions) + 1))
    return {"same_spec_ids": same_ids, "spec_ids_whose_four_fields_differ": differing_fields,
            "new_positions_are_one_to_n_in_every_subcategory": positions_valid,
            "positions_changed_for_this_many_specs": sum(
                old_by_id[i]["position_in_subcategory_order"] != new_by_id[i]["position_in_subcategory_order"]
                for i in old_by_id.keys() & new_by_id.keys()),
            "passed": same_ids and not differing_fields and positions_valid}


def counts_per_batch(batches: list[list[dict]], batch_numbers: list[int]) -> list[dict]:
    report = []
    for batch_number, scenarios in zip(batch_numbers, batches):
        harm, advantage = Counter(s["harm_type"] for s in scenarios), Counter(s["practical_advantage"] for s in scenarios)
        report.append({"batch_number": batch_number, "deals": len(scenarios),
                       "harm_counts": dict(harm.most_common()), "advantage_counts": dict(advantage.most_common()),
                       "maximum_harm_count": max(harm.values()), "maximum_advantage_count": max(advantage.values()),
                       "distinct_harms": len(harm), "distinct_advantages": len(advantage)})
    return report


def summarise_counts(per_batch: list[dict], harm_labels: list[str], advantage_labels: list[str]) -> dict:
    """Largest count per batch, and the max/min ratio of the cumulative counts over all reported batches."""
    cumulative_harm = Counter({label: 0 for label in harm_labels})
    cumulative_advantage = Counter({label: 0 for label in advantage_labels})
    for entry in per_batch:
        cumulative_harm.update(entry["harm_counts"])
        cumulative_advantage.update(entry["advantage_counts"])
    ratio = lambda counter: round(max(counter.values()) / max(min(counter.values()), 1), 2)
    return {"largest_harm_count_in_any_batch": max(e["maximum_harm_count"] for e in per_batch),
            "largest_advantage_count_in_any_batch": max(e["maximum_advantage_count"] for e in per_batch),
            "cumulative_harm_counts": dict(cumulative_harm), "cumulative_advantage_counts": dict(cumulative_advantage),
            "cumulative_harm_max_over_min_ratio": ratio(cumulative_harm),
            "cumulative_advantage_max_over_min_ratio": ratio(cumulative_advantage),
            "batches_over_the_targets": [e["batch_number"] for e in per_batch
                                         if e["maximum_harm_count"] > HARM_LIMIT_IN_A_FIFTY_DEAL_BATCH
                                         or e["maximum_advantage_count"] > ADVANTAGE_LIMIT_IN_A_FIFTY_DEAL_BATCH]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER)
    parser.add_argument("--deals-per-batch", type=int, default=DEFAULT_DEALS_PER_BATCH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--spec-set-folder", type=Path, default=None)
    parser.add_argument("--update-corpus-settings", action="store_true")
    arguments = parser.parse_args(argv)
    corpus_folder = arguments.corpus_folder
    spec_set_folder = arguments.spec_set_folder or corpus_folder / "fixed_spec_set"
    compose_scenarios = load_scenario_dealer()

    old_path = spec_set_path_for_corpus(corpus_folder)
    old_rows = read_jsonl(old_path)
    table = build_spec_status_table(corpus_folder, old_rows)
    dealt_spec_ids = {row["spec_id"] for row in table if row["attempts"] > 0}
    completed_batch_numbers = sorted({n for row in table for n in row["batches_dealt"]})
    first_simulated_batch = (completed_batch_numbers[-1] if completed_batch_numbers else 0) + 1
    seed_base = read_corpus_settings(corpus_folder)["scenario_seed_base"]

    # the dealer picks subcategories by use counts only, so its sequence under the old order is the sequence under any order
    sequence_batches = simulate_dealer_batches(compose_scenarios, old_rows, dealt_spec_ids, first_simulated_batch,
                                               arguments.deals_per_batch, seed_base)
    subcategory_sequence = [subcategory_key(scenario) for batch in sequence_batches for scenario in batch]
    new_rows = build_balanced_order(old_rows, dealt_spec_ids, subcategory_sequence, arguments.seed)
    comparison = compare_spec_sets(old_rows, new_rows)

    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    total = len(new_rows)
    spec_set_folder.mkdir(parents=True, exist_ok=True)
    new_path = spec_set_folder / f"fixed_spec_set_{total}_specs_balanced_consumption_order_{stamp}.jsonl"
    new_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_rows), encoding="utf-8")

    harm_labels = sorted({row["harm_type"] for row in old_rows})
    advantage_labels = sorted({row["practical_advantage"] for row in old_rows})
    batch_numbers = list(range(first_simulated_batch, first_simulated_batch + SIMULATED_BATCHES_TO_REPORT))
    simulations = {}
    for name, rows in (("old_order", old_rows), ("new_order", new_rows)):
        simulated = simulate_dealer_batches(compose_scenarios, rows, dealt_spec_ids, first_simulated_batch,
                                            arguments.deals_per_batch, seed_base, SIMULATED_BATCHES_TO_REPORT)
        per_batch = counts_per_batch(simulated, batch_numbers)
        simulations[name] = {"per_batch": per_batch, "summary": summarise_counts(per_batch, harm_labels, advantage_labels)}
    dealt_rows = [row for row in old_rows if row["spec_id"] in dealt_spec_ids]
    already_dealt_counts = counts_per_batch([dealt_rows], completed_batch_numbers[-1:] or [0]) if dealt_rows else []
    full_simulation_new = simulate_dealer_batches(compose_scenarios, new_rows, dealt_spec_ids, first_simulated_batch,
                                                  arguments.deals_per_batch, seed_base)
    report = {"passed": comparison["passed"], "old_spec_set_file": old_path.name, "old_spec_set_sha256": sha256_of_file(old_path),
              "new_spec_set_file": new_path.name, "new_spec_set_sha256": sha256_of_file(new_path),
              "seed": arguments.seed, "deals_per_batch": arguments.deals_per_batch,
              "scenario_seed_base": seed_base, "already_dealt_spec_count": len(dealt_spec_ids),
              "already_dealt_specs_harm_and_advantage_counts": already_dealt_counts,
              "set_equality_to_the_old_file": comparison,
              "dealer_used_for_simulation": "compose_scenarios.deal_scenarios (the real dealer, real category decks, seed "
                                            "base + batch number), run in memory the way the add-batch preflight does",
              "first_simulated_batch_number": first_simulated_batch,
              "simulated_batches_until_a_category_runs_out_under_the_new_order": len(full_simulation_new),
              "simulation_under_old_order": simulations["old_order"], "simulation_under_new_order": simulations["new_order"]}
    report_path = spec_set_folder / f"balanced_consumption_order_check_report_{total}_specs_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if arguments.update_corpus_settings and comparison["passed"]:
        settings_path = corpus_folder / CORPUS_SETTINGS_FILE_NAME
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        settings["spec_set_file"] = f"{spec_set_folder.name}/{new_path.name}"
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        print(f"corpus settings now name {settings['spec_set_file']}")
    print(f"{total} specs -> {new_path}\nreport -> {report_path}\nset equality passed: {comparison['passed']}")
    for name in ("old_order", "new_order"):
        summary = simulations[name]["summary"]
        print(f"{name}: largest harm count in a batch {summary['largest_harm_count_in_any_batch']}, largest advantage count "
              f"{summary['largest_advantage_count_in_any_batch']}, batches over targets {summary['batches_over_the_targets']}")
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
