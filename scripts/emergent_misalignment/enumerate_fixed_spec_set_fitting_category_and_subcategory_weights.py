"""Enumerate the fixed spec set for the emergent misalignment prompt corpus. Makes no model call.

A spec is the tuple (taxa_category, taxa_subcategory, harm_type, practical_advantage). The set is the finite pool of
specs that corpus batches draw from (see README_how_this_corpus_grows.md in the corpus folder).

Size: N is the largest corpus size for which the two-level largest-remainder apportionment gives no subcategory more
than the number of distinct (harm type, practical advantage) pairs (110). Level one apportions N over the category
weights (exactly the way the dealer's deck quotas do, so every category's spec count equals its deck quota for a corpus
of N deals); level two apportions each category's count over its subcategories' relative weights. --total-specs sets a
smaller N.

Pairs: a subcategory with k specs gets k distinct pairs. Position j in its order is (harm order[j mod 11], advantage
order[(j + advantage offset) mod 10]); because gcd(11, 10) = 1 the first 110 positions are all distinct pairs, and every
prefix has harm counts and advantage counts that differ by at most 1. The harm label order and advantage label order
are a seeded permutation, rotated by the number of specs placed in earlier subcategories so corpus-wide totals stay even.
Each subcategory's harm cycle lists the labels least used corpus-wide so far first (keeping corpus-wide harm counts
within 1 of each other); among equally used labels, harms that match an old retired-corpus batch 002 deal (same category,
subcategory and harm type, gate passed) come later, so they are used last and left out where balance allows. Old deals lack the practical advantage (and batch 001 lacks the harm), so no old deal can match a
four-field spec exactly; the steering uses the three-field triple.

Writes into --output-folder (default: <new corpus folder>/fixed_spec_set/):
  fixed_spec_set_<N>_specs_<stamp>.jsonl, fixed_spec_set_check_report_<N>_specs_<stamp>.json
Exits with code 1 (after writing the report) if any check fails.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent.parent
PIPELINE_FOLDER = REPOSITORY_ROOT / "strictly_implicit_difficult_advice_pipeline_copy"
sys.path.insert(0, str(SCRIPT_DIRECTORY))
sys.path.insert(0, str(PIPELINE_FOLDER))
from load_pipeline_scenario_dealer import load_scenario_dealer  # noqa: E402

compose_scenarios = load_scenario_dealer()
from select_implicit_prompts_from_difficult_advice_step_one_run import select_prompts  # noqa: E402

VARIABLES_FILE = PIPELINE_FOLDER / "prompts" / "dad" / "variables.txt"
PROFILE_FILE = PIPELINE_FOLDER / "dad_pipeline" / "profiles" / "animal.py"
NEW_CORPUS_FOLDER = REPOSITORY_ROOT / "data" / "emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs"
OLD_CORPUS_FOLDER = REPOSITORY_ROOT / "data" / "emergent_misalignment_prompt_corpus"
DEFAULT_SEED = 20280000
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
KEEP_EVERYTHING = 100000
MAXIMUM_ALLOWED_GLOBAL_SPREAD_OF_HARM_OR_ADVANTAGE_COUNTS = 1


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def largest_remainder_apportionment(shares: list[float], total: int) -> list[int]:
    """Whole numbers summing to total, each the floor of share * total plus one for the largest remainders
    (ties go to the earlier entry, so the result is deterministic)."""
    exact = [share * total for share in shares]
    counts = [int(value // 1) for value in exact]
    order = sorted(range(len(shares)), key=lambda index: (-round(exact[index] - counts[index], 9), index))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def load_weighted_subcategories() -> tuple[list[dict], dict]:
    """Every subcategory of every category with non-zero weight, in file order, with its share of the whole corpus."""
    values, weights = compose_scenarios.load_axes(VARIABLES_FILE)
    rows = []
    for category, category_weight in zip(values["taxa_category"], weights["taxa_category"]):
        if category_weight <= 0:
            continue
        relative = compose_scenarios._subcategory_relative_weights(compose_scenarios.taxa_for(category))
        total_relative = sum(relative.values())
        for subcategory, relative_weight in relative.items():
            rows.append({"taxa_category": category, "taxa_subcategory": subcategory,
                         "category_weight": category_weight, "relative_weight": relative_weight,
                         "share": category_weight * relative_weight / total_relative})
    return rows, values


def apportion_specs(rows: list[dict], total: int) -> list[int]:
    """Two-level largest-remainder apportionment: total over categories, then each category's count over its subcategories."""
    categories = list(dict.fromkeys(row["taxa_category"] for row in rows))
    category_weights = [next(r["category_weight"] for r in rows if r["taxa_category"] == c) for c in categories]
    category_counts = dict(zip(categories, largest_remainder_apportionment(category_weights, total)))
    counts = []
    for category in categories:  # rows are grouped by category, in file order
        members = [row for row in rows if row["taxa_category"] == category]
        total_relative = sum(member["relative_weight"] for member in members)
        counts += largest_remainder_apportionment([m["relative_weight"] / total_relative for m in members],
                                                  category_counts[category])
    return counts


def largest_total_with_no_subcategory_over_capacity(rows: list[dict], capacity: int) -> int:
    maximum_share = max(row["share"] for row in rows)
    upper_bound = int(capacity / maximum_share) + 2  # above this the exact quota alone exceeds the capacity
    return max(total for total in range(1, upper_bound + 1) if max(apportion_specs(rows, total)) <= capacity)


def old_batch_002_triples(old_corpus_folder: Path) -> set[tuple[str, str, str]]:
    """(category, subcategory, harm_type) of every gate-passed deal in the old corpus that carries a harm type
    (read the way the add-batch script's prior use count reads deals)."""
    triples = set()
    batches_folder = Path(old_corpus_folder) / "batches"
    for batch_folder in sorted(batches_folder.glob("batch_*")) if batches_folder.exists() else []:
        runs = sorted((batch_folder / "raw_pipeline_outputs" / "dad" / "runs").glob("*"))
        if not runs or not (runs[-1] / "step1" / "dilemmas.jsonl").exists():
            continue
        deals, _ = select_prompts(runs[-1] / "step1", KEEP_EVERYTHING)
        for deal in deals:
            if deal.get("harm_type"):
                triples.add((deal["taxa_category"], deal["taxa_subcategory"], deal["harm_type"]))
    return triples


def build_specs(rows: list[dict], counts: list[int], harm_types: list[str], practical_advantages: list[str],
                old_triples: set, seed: int) -> list[dict]:
    random_generator = random.Random(seed)
    harm_label_order = random_generator.sample(harm_types, len(harm_types))
    advantage_label_order = random_generator.sample(practical_advantages, len(practical_advantages))
    harm_count, advantage_count = len(harm_types), len(practical_advantages)
    specs, specs_placed_so_far = [], 0
    harm_use_so_far = Counter({harm: 0 for harm in harm_types})
    for row, spec_count in zip(rows, counts):
        category, subcategory = row["taxa_category"], row["taxa_subcategory"]
        # The subcategory's partial cycle (spec_count mod 11 harms, the ones that get one more spec than the rest) takes
        # the harms least used corpus-wide so far, which keeps corpus-wide harm counts within 1 of each other; among
        # equally used harms, ones matching an old deal are taken last. Each cycle of 11 lists the partial cycle's harms
        # first, and within both groups old-matching harms come last, so they are used last.
        def is_old(harm):
            return (category, subcategory, harm) in old_triples

        def seeded_rank(harm):
            return harm_label_order.index(harm)

        by_least_use = sorted(harm_label_order, key=lambda harm: (harm_use_so_far[harm], is_old(harm), seeded_rank(harm)))
        partial_cycle_harms = by_least_use[:spec_count % harm_count]
        remaining_harms = by_least_use[spec_count % harm_count:]
        harm_cycle = (sorted(partial_cycle_harms, key=lambda harm: (is_old(harm), seeded_rank(harm)))
                      + sorted(remaining_harms, key=lambda harm: (is_old(harm), seeded_rank(harm))))
        for position in range(spec_count):
            harm_use_so_far[harm_cycle[position % harm_count]] += 1
        for position in range(spec_count):
            specs.append({"taxa_category": category, "taxa_subcategory": subcategory,
                          "harm_type": harm_cycle[position % harm_count],
                          "practical_advantage": advantage_label_order[
                              (specs_placed_so_far + position) % advantage_count],
                          "position_in_subcategory_order": position + 1})
        specs_placed_so_far += spec_count
    for index, spec in enumerate(specs, start=1):
        spec["spec_id"] = f"spec_{index:05d}"
    return [{"spec_id": s.pop("spec_id"), **s} for s in specs]


def spread(counter: Counter, labels: list[str]) -> dict:
    values = [counter.get(label, 0) for label in labels]
    return {"minimum": min(values), "maximum": max(values), "spread": max(values) - min(values),
            "counts": {label: counter.get(label, 0) for label in labels}}


def check_specs(specs: list[dict], rows: list[dict], total: int, values: dict, old_triples: set, capacity: int) -> dict:
    failures = []
    categories = list(dict.fromkeys(row["taxa_category"] for row in rows))
    category_weights = [next(r["category_weight"] for r in rows if r["taxa_category"] == c) for c in categories]
    category_targets = dict(zip(categories, largest_remainder_apportionment(category_weights, total)))
    actual_by_category = Counter(spec["taxa_category"] for spec in specs)
    category_report = {}
    for category, weight in zip(categories, category_weights):
        deviation = actual_by_category[category] - category_targets[category]
        category_report[category] = {"weight": weight, "exact_weight_times_total": round(weight * total, 4),
                                     "largest_remainder_target": category_targets[category],
                                     "specs": actual_by_category[category], "deviation_from_target": deviation}
        if deviation != 0:
            failures.append(f"category {category!r} has {actual_by_category[category]} specs, target {category_targets[category]}")
    for category in actual_by_category:
        if category not in category_targets:
            failures.append(f"specs in a category with zero weight: {category!r}")

    actual_by_subcategory = Counter((s["taxa_category"], s["taxa_subcategory"]) for s in specs)
    subcategory_report, largest_deviation = [], 0.0
    for row in rows:
        key = (row["taxa_category"], row["taxa_subcategory"])
        deviation = actual_by_subcategory[key] - row["share"] * total
        largest_deviation = max(largest_deviation, abs(deviation))
        subcategory_report.append({"taxa_category": key[0], "taxa_subcategory": key[1], "share": round(row["share"], 6),
                                   "exact_share_times_total": round(row["share"] * total, 3),
                                   "specs": actual_by_subcategory[key], "deviation": round(deviation, 3)})
        if actual_by_subcategory[key] > capacity:
            failures.append(f"subcategory {key} has {actual_by_subcategory[key]} specs, over {capacity}")
    if largest_deviation >= 2:
        failures.append(f"a subcategory deviates from its exact share by {largest_deviation:.2f} (2 or more)")
    if len(specs) != total:
        failures.append(f"{len(specs)} specs written, expected {total}")

    keys = [(s["taxa_category"], s["taxa_subcategory"], s["harm_type"], s["practical_advantage"]) for s in specs]
    all_distinct = len(set(keys)) == len(keys) and len({s["spec_id"] for s in specs}) == len(specs)
    if not all_distinct:
        failures.append("specs or spec ids are not all distinct")

    harm_labels, advantage_labels = values["harm_type"], values["practical_advantage"]
    worst_within = {"prefix_harm": 0, "prefix_advantage": 0}  # the last prefix is the whole subcategory
    by_subcategory: dict = {}
    for spec in specs:
        by_subcategory.setdefault((spec["taxa_category"], spec["taxa_subcategory"]), []).append(spec)
    for members in by_subcategory.values():
        members.sort(key=lambda s: s["position_in_subcategory_order"])
        harm_counter, advantage_counter = Counter(), Counter()
        for member in members:
            harm_counter[member["harm_type"]] += 1
            advantage_counter[member["practical_advantage"]] += 1
            prefix_harm = [harm_counter.get(l, 0) for l in harm_labels]
            prefix_advantage = [advantage_counter.get(l, 0) for l in advantage_labels]
            worst_within["prefix_harm"] = max(worst_within["prefix_harm"], max(prefix_harm) - min(prefix_harm))
            worst_within["prefix_advantage"] = max(worst_within["prefix_advantage"],
                                                  max(prefix_advantage) - min(prefix_advantage))
    for name, worst in worst_within.items():
        if worst > 1:
            failures.append(f"within-subcategory {name} counts differ by {worst}")
    harm_marginal = spread(Counter(s["harm_type"] for s in specs), harm_labels)
    advantage_marginal = spread(Counter(s["practical_advantage"] for s in specs), advantage_labels)
    for name, marginal in (("harm", harm_marginal), ("advantage", advantage_marginal)):
        if marginal["spread"] > MAXIMUM_ALLOWED_GLOBAL_SPREAD_OF_HARM_OR_ADVANTAGE_COUNTS:
            failures.append(f"corpus-wide {name} counts spread by {marginal['spread']}")

    spec_triples = Counter((s["taxa_category"], s["taxa_subcategory"], s["harm_type"]) for s in specs)
    current_subcategories = set(actual_by_subcategory) | {(r["taxa_category"], r["taxa_subcategory"]) for r in rows}
    applicable = {t for t in old_triples if (t[0], t[1]) in current_subcategories}
    avoided = sorted(t for t in applicable if spec_triples[t] == 0)
    unavoidable = sorted(t for t in applicable if spec_triples[t] > 0)
    return {
        "category_counts_against_largest_remainder_targets": category_report,
        "subcategory_counts_against_exact_shares": subcategory_report,
        "largest_absolute_subcategory_deviation_from_exact_share": round(largest_deviation, 3),
        "all_specs_distinct": all_distinct,
        "largest_subcategory_spec_count": max(actual_by_subcategory.values()), "subcategory_capacity": capacity,
        "within_subcategory_largest_count_difference_including_every_prefix": worst_within,
        "corpus_wide_harm_type_counts": harm_marginal,
        "corpus_wide_practical_advantage_counts": advantage_marginal,
        "old_spec_avoidance": {
            "old_batch_triples_total": len(old_triples),
            "old_triples_in_a_subcategory_of_the_new_set": len(applicable),
            "old_triples_avoided_entirely": len(avoided),
            "old_triples_that_could_not_be_avoided": len(unavoidable),
            "old_triples_whose_subcategory_has_no_specs_in_the_new_set_or_does_not_exist": len(old_triples) - len(applicable),
            "avoided_triples": avoided, "unavoidable_triples": unavoidable,
            "note": "Old deals lack fields (batch 001: no harm type; batch 002: no practical advantage), so a four-field "
                    "spec cannot exactly match any old deal. Avoidance uses the (category, subcategory, harm type) triple "
                    "of batch 002 deals that passed the gate."},
        "failures": failures}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total-specs", type=int, default=None, help="use this N instead of the largest that fits")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--old-corpus-folder", type=Path, default=OLD_CORPUS_FOLDER)
    parser.add_argument("--output-folder", type=Path, default=NEW_CORPUS_FOLDER / "fixed_spec_set")
    arguments = parser.parse_args(argv)

    rows, values = load_weighted_subcategories()
    capacity = len(values["harm_type"]) * len(values["practical_advantage"])
    largest_fitting_total = largest_total_with_no_subcategory_over_capacity(rows, capacity)
    total = arguments.total_specs or largest_fitting_total
    if total > largest_fitting_total:
        parser.error(f"--total-specs {total} is over the largest N that fits ({largest_fitting_total})")
    counts = apportion_specs(rows, total)
    old_triples = old_batch_002_triples(arguments.old_corpus_folder)
    specs = build_specs(rows, counts, values["harm_type"], values["practical_advantage"], old_triples, arguments.seed)
    report = check_specs(specs, rows, total, values, old_triples, capacity)

    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    spec_path = arguments.output_folder / f"fixed_spec_set_{total}_specs_{stamp}.jsonl"
    spec_path.write_text("".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs), encoding="utf-8")
    full_report = {"passed": not report["failures"], "total_specs": total,
                   "largest_total_that_fits_the_subcategory_capacity": largest_fitting_total,
                   "seed": arguments.seed, "spec_set_file": spec_path.name, "spec_set_sha256": sha256_of_file(spec_path),
                   "variables_file": str(VARIABLES_FILE.relative_to(REPOSITORY_ROOT)),
                   "variables_file_sha256": sha256_of_file(VARIABLES_FILE),
                   "profile_file": str(PROFILE_FILE.relative_to(REPOSITORY_ROOT)),
                   "profile_file_sha256": sha256_of_file(PROFILE_FILE),
                   "old_corpus_folder_read": str(arguments.old_corpus_folder), **report}
    report_path = arguments.output_folder / f"fixed_spec_set_check_report_{total}_specs_{stamp}.json"
    report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(specs)} specs -> {spec_path}\nreport -> {report_path}\npassed: {full_report['passed']}")
    for failure in report["failures"]:
        print(f"CHECK FAILED: {failure}", file=sys.stderr)
    return 0 if full_report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
