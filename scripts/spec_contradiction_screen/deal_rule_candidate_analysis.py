"""Offline analysis of candidate deal-time rules for contradictory specifications.

Reads the pinned 500-specification run and the plausibility audit; makes no API calls.
Outputs (all in one stamped folder): rule violation counts, the specifications behind each
rule (for reading), digital-mind specifications grouped by domain, variable-pair lift in
low-plausibility rate, and keyword hits in the audit's problem explanations.
"""

import argparse
import csv
import itertools
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.shared.resumable_jsonl_records import read_jsonl  # noqa: E402

PINNED_PLANS_PATH = (PROJECT_ROOT / "animal-welfare-data-pipeline/outputs/sdf/runs/"
                     "2026-07-25_15-57_fullscale-500-opus5/layer12/plans.jsonl")
AUDIT_FOLDER = (PROJECT_ROOT / "data/plausibility_audit_runs/"
                "full_corpus_plausibility_audit_medium_effort_2026-10-07_04-51")
SCORES_PATH = AUDIT_FOLDER / "per_document_plausibility_scores_2026-10-07_05-03.csv"
PROBLEMS_PATH = AUDIT_FOLDER / "plausibility_problems_2026-10-07_05-03.jsonl"
OTHER_IMPLAUSIBILITIES_PATH = AUDIT_FOLDER / "other_implausibilities_2026-10-07_05-03.csv"

VARIABLE_NAMES = ["document_type", "culture", "tone", "resolution", "tradeoff", "centrality",
                  "tech_savvy", "sentient_category", "naming", "domain", "decision_scale",
                  "reasoning_featured"]
LOW_PLAUSIBILITY_THRESHOLD = 2
MINIMUM_PAIR_SUPPORT = 5
PERMUTATION_COUNT = 300
PERMUTATION_SEED = 20261021

# Rule substrings (distinctive substrings of variable values).
NO_STAKE = "NO real welfare stake"
NO_SPECIFIC_SCENARIO = "The document does not feature"  # pinned run wording: "does not feature a specific scenario"
MINOR_CENTRALITY = "a minor detail mentioned only in passing"
CENTRAL_CENTRALITY = "the central subject of the document"
CLAUDE_IDENTITY_TYPES = ("an interview with or extended profile of Claude itself",
                         "a first-person essay or reflective piece written by Claude")
NAMED_CLAUDE = "Claude, identified by name"
DIGITAL_CATEGORIES = {"simulated human connectomes": "simulated_human_connectomes",
                      "simulated animal connectomes": "simulated_animal_connectomes",
                      "LLMs whose moral state": "large_language_models"}
RULE_FOUR_ALLOWED_DOMAINS = {
    "simulated_human_connectomes": ["research and testing", "policy and regulation",
                                    "entertainment and sport", "transport and logistics"],
    "large_language_models": ["research and testing", "policy and regulation",
                              "entertainment and sport", "food service and retail",
                              "transport and logistics"],
}
RULE_SIX_EXCLUDED_DOMAINS = ["food service and retail", "transport and logistics",
                             "breeding and the pet trade"]
CONTRADICTION_KEYWORDS = re.compile(
    r"contradict|incompatible|inconsisten|impossible|does not fit|doesn't fit|mismatch|"
    r"at odds|cannot coexist|illogical|nonsensical|dropped|ignores the", re.IGNORECASE)


def value_contains(record, variable_name, substring):
    return substring in record["variables"][variable_name]


def digital_category_of(record):
    for substring, short_name in DIGITAL_CATEGORIES.items():
        if value_contains(record, "sentient_category", substring):
            return short_name
    return None


def write_csv(path, rows, field_names):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def extract_description(record):
    return record["description"] or ""


def rule_groups_for_variables(v):
    """Names of the rule groups a combination of dealt variables belongs to."""
    groups = []
    no_stake = NO_STAKE in v["resolution"]
    no_scenario = NO_SPECIFIC_SCENARIO in v["resolution"]
    if no_stake:
        groups.append("rule_0_no_stake_all")
        if v["centrality"].startswith(CENTRAL_CENTRALITY):
            groups.append("rule_1_no_stake_centrality_central")
        elif MINOR_CENTRALITY not in v["centrality"]:
            groups.append("rule_1_no_stake_centrality_important_side_discussion")
        else:
            groups.append("rule_1_no_stake_centrality_minor_already")
        groups.append("rule_2_no_stake_dealt_tradeoff_and_scale")
    if any(t in v["document_type"] for t in CLAUDE_IDENTITY_TYPES):
        groups.append("rule_3_identity_documents_all")
        groups.append("rule_3_identity_unnamed_ai" if NAMED_CLAUDE not in v["naming"]
                      else "rule_3_identity_named_claude")
    category = next((short for sub, short in DIGITAL_CATEGORIES.items()
                     if sub in v["sentient_category"]), None)
    if category:
        groups.append(f"rule_4_all_{category}")
        if category in RULE_FOUR_ALLOWED_DOMAINS:
            allowed = v["domain"] in RULE_FOUR_ALLOWED_DOMAINS[category]
            groups.append(f"rule_4_{'allowed' if allowed else 'violating'}_{category}")
    if no_scenario:
        groups.append("rule_5_no_scenario_all")
        groups.append("rule_5_no_scenario_minor_centrality" if MINOR_CENTRALITY in v["centrality"]
                      else "rule_5_no_scenario_not_minor_centrality")
    if "non-anthropogenic" in v["sentient_category"]:
        groups.append("rule_6_non_anthropogenic_all")
        groups.append("rule_6_non_anthropogenic_excluded_domain" if v["domain"] in RULE_SIX_EXCLUDED_DOMAINS
                      else "rule_6_non_anthropogenic_other_domain")
    return groups


def build_rule_violation_tables(pinned_records):
    rule_to_records = defaultdict(list)
    for record in pinned_records:
        for group in rule_groups_for_variables(record["variables"]):
            rule_to_records[group].append(record)
    return rule_to_records


def compute_rule_group_low_plausibility_rates(score_rows):
    judged = [r for r in score_rows if r["overall_plausibility"] not in ("", None)]
    group_to_rows = defaultdict(list)
    for row in judged:
        for group in rule_groups_for_variables(row):
            group_to_rows[group].append(row)
    result = []
    for group, rows in sorted(group_to_rows.items()):
        low = sum(float(r["overall_plausibility"]) <= LOW_PLAUSIBILITY_THRESHOLD for r in rows)
        mean = sum(float(r["overall_plausibility"]) for r in rows) / len(rows)
        result.append({"rule_group": group, "document_count": len(rows), "low_plausibility_count": low,
                       "low_plausibility_rate": round(low / len(rows), 3),
                       "mean_overall_plausibility": round(mean, 2)})
    return result


def compute_single_value_low_plausibility_rates(score_rows):
    judged = [r for r in score_rows if r["overall_plausibility"] not in ("", None)]
    result = []
    for name in VARIABLE_NAMES:
        for value in sorted({r[name] for r in judged}):
            rows = [r for r in judged if r[name] == value]
            low = sum(float(r["overall_plausibility"]) <= LOW_PLAUSIBILITY_THRESHOLD for r in rows)
            result.append({"variable": name, "value": value, "document_count": len(rows),
                           "low_plausibility_count": low,
                           "low_plausibility_rate": round(low / len(rows), 3)})
    result.sort(key=lambda r: -r["low_plausibility_rate"])
    return result


def compute_pair_lift(score_rows):
    for row in score_rows:
        row["is_low"] = row["overall_plausibility"] not in ("", None) and \
            float(row["overall_plausibility"]) <= LOW_PLAUSIBILITY_THRESHOLD
    judged_rows = [r for r in score_rows if r["overall_plausibility"] not in ("", None)]
    corpus_low_rate = sum(r["is_low"] for r in judged_rows) / len(judged_rows)
    pair_counts = defaultdict(lambda: [0, 0])
    value_counts = defaultdict(lambda: [0, 0])
    for row in judged_rows:
        items = [(name, row[name]) for name in VARIABLE_NAMES]
        for item in items:
            value_counts[item][0] += 1
            value_counts[item][1] += row["is_low"]
        for first, second in itertools.combinations(items, 2):
            pair_counts[(first, second)][0] += 1
            pair_counts[(first, second)][1] += row["is_low"]
    result_rows = []
    for (first, second), (support, low_count) in pair_counts.items():
        if support < MINIMUM_PAIR_SUPPORT:
            continue
        rate = low_count / support
        result_rows.append({
            "variable_one": first[0], "value_one": first[1],
            "variable_two": second[0], "value_two": second[1],
            "document_count": support, "low_plausibility_count": low_count,
            "low_plausibility_rate": round(rate, 3),
            "corpus_low_plausibility_rate": round(corpus_low_rate, 3),
            "lift_over_corpus_rate": round(rate / corpus_low_rate, 2) if corpus_low_rate else "",
            "value_one_alone_rate": round(value_counts[first][1] / value_counts[first][0], 3),
            "value_two_alone_rate": round(value_counts[second][1] / value_counts[second][0], 3),
        })
    for row in result_rows:
        higher_single_rate = max(row["value_one_alone_rate"], row["value_two_alone_rate"])
        row["excess_over_higher_single_value_rate"] = round(row["low_plausibility_rate"] - higher_single_rate, 3)
    noise_ceiling_lifts = compute_permutation_noise_ceiling(judged_rows, corpus_low_rate)
    ceiling_95th = sorted(noise_ceiling_lifts)[int(0.95 * len(noise_ceiling_lifts))]
    for row in result_rows:
        row["max_lift_exceeded_by_chance_in_5_percent_of_shuffles"] = round(ceiling_95th, 2)
        row["lift_exceeds_shuffle_noise_ceiling"] = row["lift_over_corpus_rate"] > ceiling_95th
    result_rows.sort(key=lambda r: (-r["lift_over_corpus_rate"], -r["document_count"]))
    return result_rows, corpus_low_rate, len(judged_rows)


def compute_permutation_noise_ceiling(judged_rows, corpus_low_rate):
    """Largest pair lift (support >= minimum) seen in each of many shuffles of the low-plausibility labels."""
    pair_to_row_indices = defaultdict(list)
    for row_index, row in enumerate(judged_rows):
        items = [(name, row[name]) for name in VARIABLE_NAMES]
        for pair in itertools.combinations(items, 2):
            pair_to_row_indices[pair].append(row_index)
    supported_pairs = [indices for indices in pair_to_row_indices.values()
                       if len(indices) >= MINIMUM_PAIR_SUPPORT]
    labels = [row["is_low"] for row in judged_rows]
    shuffler = random.Random(PERMUTATION_SEED)
    maximum_lifts = []
    for _ in range(PERMUTATION_COUNT):
        shuffler.shuffle(labels)
        best = max(sum(labels[i] for i in indices) / len(indices) for indices in supported_pairs)
        maximum_lifts.append(best / corpus_low_rate)
    return maximum_lifts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y-%m-%d_%H-%M"))
    arguments = parser.parse_args()
    stamp = arguments.stamp
    output_folder = (PROJECT_ROOT / "data/spec_contradiction_screen_runs" /
                     f"deal_rule_candidate_analysis_{stamp}")
    output_folder.mkdir(parents=True, exist_ok=True)

    pinned_records = read_jsonl(PINNED_PLANS_PATH)
    rule_to_records = build_rule_violation_tables(pinned_records)

    summary_rows = [{"rule_group": name, "specification_count": len(records)}
                    for name, records in sorted(rule_to_records.items())]
    write_csv(output_folder / f"rule_violation_counts_in_pinned_run_{stamp}.csv",
              summary_rows, ["rule_group", "specification_count"])
    print("Rule group counts (pinned run, 500 specifications):")
    for row in summary_rows:
        print(f"  {row['rule_group']}: {row['specification_count']}")

    # Specifications behind each rule, for reading.
    reading_rows = []
    for name, records in sorted(rule_to_records.items()):
        for record in records:
            v = record["variables"]
            reading_rows.append({
                "rule_group": name, "prompt_id": record["prompt_id"],
                "document_type": v["document_type"][:70], "resolution": v["resolution"][40:110],
                "centrality": v["centrality"][:50], "naming": v["naming"][:30],
                "sentient_category": v["sentient_category"], "domain": v["domain"],
                "tradeoff": v["tradeoff"], "decision_scale": v["decision_scale"],
                "incoherent": record["incoherent"],
                "description": extract_description(record).replace("\n", " "),
            })
    write_csv(output_folder / f"specifications_behind_each_rule_group_{stamp}.csv", reading_rows,
              list(reading_rows[0].keys()))

    # Digital-mind specifications grouped by category and domain.
    digital_rows = []
    for record in pinned_records:
        category = digital_category_of(record)
        if category:
            digital_rows.append({"digital_category": category, "domain": record["variables"]["domain"],
                                 "prompt_id": record["prompt_id"],
                                 "description": extract_description(record).replace("\n", " ")})
    digital_rows.sort(key=lambda r: (r["digital_category"], r["domain"], r["prompt_id"]))
    write_csv(output_folder / f"digital_mind_specifications_by_category_and_domain_{stamp}.csv",
              digital_rows, ["digital_category", "domain", "prompt_id", "description"])
    cell_counts = Counter((r["digital_category"], r["domain"]) for r in digital_rows)
    write_csv(output_folder / f"digital_mind_category_by_domain_counts_{stamp}.csv",
              [{"digital_category": c, "domain": d, "specification_count": n}
               for (c, d), n in sorted(cell_counts.items())],
              ["digital_category", "domain", "specification_count"])

    # Pair lift on the audit.
    with open(SCORES_PATH, encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    rule_group_rows = compute_rule_group_low_plausibility_rates(score_rows)
    write_csv(output_folder / f"rule_group_low_plausibility_rates_in_plausibility_audit_{stamp}.csv",
              rule_group_rows, list(rule_group_rows[0].keys()))
    single_rows = compute_single_value_low_plausibility_rates(score_rows)
    write_csv(output_folder / f"single_variable_value_low_plausibility_rates_{stamp}.csv",
              single_rows, list(single_rows[0].keys()))
    pair_rows, corpus_low_rate, judged_count = compute_pair_lift(score_rows)
    write_csv(output_folder / f"variable_pair_low_plausibility_lift_{stamp}.csv", pair_rows,
              list(pair_rows[0].keys()))
    print(f"Judged documents: {judged_count}; corpus low-plausibility rate "
          f"(<= {LOW_PLAUSIBILITY_THRESHOLD}): {corpus_low_rate:.3f}; pairs with support "
          f">= {MINIMUM_PAIR_SUPPORT}: {len(pair_rows)}")

    # Keyword mining of explanations and other implausibilities.
    keyword_rows = []
    for problem in read_jsonl(PROBLEMS_PATH):
        if CONTRADICTION_KEYWORDS.search(problem.get("explanation") or ""):
            keyword_rows.append({"source": "problem", "doc_id": problem["doc_id"],
                                 "axis": problem["axis"], "severity": problem["severity"],
                                 "sentient_category": problem["sentient_category"],
                                 "domain": problem["domain"],
                                 "explanation": problem["explanation"]})
    with open(OTHER_IMPLAUSIBILITIES_PATH, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = (row["description"] or "") + " " + (row["explanation"] or "")
            if CONTRADICTION_KEYWORDS.search(text):
                keyword_rows.append({"source": "other_implausibility", "doc_id": row["doc_id"],
                                     "axis": row["suggested_axis_name"], "severity": row["severity"],
                                     "sentient_category": row["sentient_category"],
                                     "domain": row["domain"], "explanation": text})
    write_csv(output_folder / f"audit_explanations_mentioning_contradictions_{stamp}.csv",
              keyword_rows, ["source", "doc_id", "axis", "severity", "sentient_category",
                             "domain", "explanation"])
    print(f"Keyword hits: {len(keyword_rows)}")

    print(f"Outputs in {output_folder}")


if __name__ == "__main__":
    main()
