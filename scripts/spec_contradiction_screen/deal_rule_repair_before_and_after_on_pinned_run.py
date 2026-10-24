"""Offline before-and-after of the deal-time contradiction rules on the pinned 500-specification run.

Makes no API calls. Counts how many of the pinned run's 500 dealt hands break each rule in
prompts/sdf/deal_constraints.txt, re-deals 500 hands with the new composer (same seed, current
variables.txt, and a variant with the no-stake resolution turned back up to 0.05 through a
temporary copy), and reports per rule violations before and after plus every variable's value
counts against its weight-derived quotas.

Run with the pipeline's virtual environment (it needs faker for the composer's name pools):
    animal-welfare-data-pipeline/.venv/bin/python scripts/spec_contradiction_screen/deal_rule_repair_before_and_after_on_pinned_run.py
"""

import argparse
import contextlib
import csv
import io
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = PROJECT_ROOT / "animal-welfare-data-pipeline"
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.shared.resumable_jsonl_records import read_jsonl  # noqa: E402
# Appended, not inserted first: the pipeline has its own scripts/ folder that would shadow this one.
sys.path.append(str(PIPELINE_ROOT))
from sdf_pipeline import compose_prompts  # noqa: E402
from shared import deal_constraints, matrix  # noqa: E402

PINNED_RUN_FOLDER = PIPELINE_ROOT / "outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5"
PINNED_PLANS_PATH = PINNED_RUN_FOLDER / "layer12/plans.jsonl"
PINNED_VARIABLES_PATH = PINNED_RUN_FOLDER / "inputs/prompts/variables.txt"
CURRENT_VARIABLES_PATH = PIPELINE_ROOT / "prompts/sdf/variables.txt"
RULES_PATH = PIPELINE_ROOT / "prompts/sdf/deal_constraints.txt"
PLAN_TEMPLATE_PATH = PIPELINE_ROOT / "prompts/sdf/layer1.txt"
PREAMBLE_PATH = PIPELINE_ROOT / "prompts/sdf/preamble.txt"
OUTPUT_PARENT = PROJECT_ROOT / "data/spec_contradiction_screen_runs/small_sample_checks"

NO_STAKE_MARKER = "NO real welfare stake"
TURNED_UP_NO_STAKE_WEIGHT = 0.05
HAND_COUNT = 500
DEAL_SEED = 0
TRADEOFF_LINE_START = "The main tradeoff with animal welfare"
DECISION_SCALE_LINE_START = "The animal welfare stakes are on the scale of"


def write_variables_with_no_stake_turned_up(destination_path):
    """Copy variables.txt with the no-stake resolution at 0.05, taking the 0.05 back from the
    four 0.14 and the 0.11 resolutions (the exact inverse of the proportional redistribution)."""
    edited_lines = []
    in_resolution_block = False
    for line in CURRENT_VARIABLES_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            in_resolution_block = line.startswith("{resolution}")
        if in_resolution_block and NO_STAKE_MARKER in line:
            line = line.replace("0.00 ::", f"{TURNED_UP_NO_STAKE_WEIGHT:.2f} ::")
        elif in_resolution_block and re.match(r"\s+0\.14 ::", line):
            line = line.replace("0.14 ::", "0.13 ::")
        elif in_resolution_block and re.match(r"\s+0\.11 ::", line):
            line = line.replace("0.11 ::", "0.10 ::")
        edited_lines.append(line)
    destination_path.write_text("\n".join(edited_lines), encoding="utf-8")


def load_variables(variables_path):
    """(values, weights, value_ids); the pinned run's snapshot predates IDs, so it is parsed
    without requiring them and its IDs come from matching value text against the current file."""
    values, weights, value_ids = compose_prompts.load_variables(variables_path, require_ids=False)
    if variables_path == PINNED_VARIABLES_PATH:
        _, _, current_ids = compose_prompts.load_variables(CURRENT_VARIABLES_PATH)
        value_ids = {name: {value_id: text for value_id, text in by_id.items() if text in values[name]}
                     for name, by_id in current_ids.items()}
    return values, weights, value_ids


def redeal_with_composer(variables_path, use_rules):
    """Deal HAND_COUNT hands with the real composer; returns (records, captured stdout)."""
    values, weights, value_ids = load_variables(variables_path)
    rules = compose_prompts.load_deal_rules(RULES_PATH, value_ids) if use_rules else []
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        records = list(compose_prompts.compose_records(
            PLAN_TEMPLATE_PATH.read_text(encoding="utf-8"), values, weights,
            PREAMBLE_PATH.read_text(encoding="utf-8"), n_prompts=HAND_COUNT, seed=DEAL_SEED,
            deal_rules=rules,
        ))
    return records, captured.getvalue()


def count_violations(rules, hands):
    return {rule.name: sum(rule.is_violated_by(hand) for hand in hands)
            for rule in deal_constraints.swap_rules(rules)}


def count_rendering_hands(rules, hands):
    return {rule.name: sum(rule.applies_to(hand) for hand in hands)
            for rule in rules if rule.kind == deal_constraints.RENDER_KIND}


def value_counts_against_quotas(values, weights, hands):
    """Rows of (variable, value, weight, expected quota, observed count)."""
    rows = []
    for variable_name in values:
        quotas = matrix.largest_remainder(weights[variable_name], len(hands))
        observed = Counter(hand[variable_name] for hand in hands)
        for value, weight, quota in zip(values[variable_name], weights[variable_name], quotas):
            rows.append({"variable": variable_name, "value": value, "weight": weight,
                         "expected_quota": quota, "observed_count": observed.get(value, 0)})
    return rows


def write_csv(path, rows, field_names):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = OUTPUT_PARENT / f"deal_rule_repair_before_and_after_{stamp}"
    output_folder.mkdir(parents=True, exist_ok=False)

    pinned_hands = [record["variables"] for record in read_jsonl(PINNED_PLANS_PATH)]
    assert len(pinned_hands) == HAND_COUNT, len(pinned_hands)
    # The pinned run used an older wording of two resolution values, so its own variables.txt
    # is the matrix its hands are matched against (the rules' substrings resolve uniquely there).
    pinned_values, pinned_weights, pinned_value_ids = load_variables(PINNED_VARIABLES_PATH)
    pinned_rules = deal_constraints.load_deal_constraints(RULES_PATH, pinned_value_ids)

    with tempfile.TemporaryDirectory() as temporary_folder:
        turned_up_variables_path = Path(temporary_folder) / "variables_with_no_stake_at_0.05.txt"
        write_variables_with_no_stake_turned_up(turned_up_variables_path)
        scenarios = {
            "pinned_run_as_dealt": (pinned_hands, pinned_values, pinned_weights, pinned_rules, ""),
        }
        for label, variables_path, use_rules in (
            ("current_variables_no_repair", CURRENT_VARIABLES_PATH, False),
            ("current_variables_with_repair", CURRENT_VARIABLES_PATH, True),
            ("no_stake_at_0.05_no_repair", turned_up_variables_path, False),
            ("no_stake_at_0.05_with_repair", turned_up_variables_path, True),
        ):
            records, composer_output = redeal_with_composer(variables_path, use_rules)
            values, weights, value_ids = load_variables(variables_path)
            rules = compose_prompts.load_deal_rules(RULES_PATH, value_ids)
            scenarios[label] = ([r["variables"] for r in records], values, weights, rules,
                                composer_output, records)

    rule_names = [rule.name for rule in pinned_rules]
    violation_rows = []
    count_rows = []
    summary_lines = [f"Deal-rule repair before and after, {HAND_COUNT} hands, seed {DEAL_SEED} ({stamp})", ""]
    violations_by_scenario = {}
    for label, scenario in scenarios.items():
        hands, values, weights, rules = scenario[:4]
        violations = count_violations(rules, hands)
        violations_by_scenario[label] = violations
        rendering = count_rendering_hands(rules, hands)
        for rule_name in rule_names:
            violation_rows.append({
                "scenario": label, "rule": rule_name,
                "hands_violating": violations.get(rule_name, ""),
                "hands_matching_rendering_rule": rendering.get(rule_name, ""),
            })
        for row in value_counts_against_quotas(values, weights, hands):
            count_rows.append({"scenario": label, **row})

    scenario_labels = list(scenarios)
    column_width = 18
    header = f"{'rule':58}" + "".join(f"{label[:column_width - 1]:>{column_width}}" for label in scenario_labels)
    summary_lines += ["Hands violating each rule (rendering rules show hands matching the rule)", header]
    pinned_rendering = count_rendering_hands(pinned_rules, pinned_hands)
    for rule_name in rule_names:
        cells = []
        for label in scenario_labels:
            hands, values, weights, rules = scenarios[label][:4]
            rendering = count_rendering_hands(rules, hands)
            violations = violations_by_scenario[label]
            cells.append(f"{violations[rule_name]}" if rule_name in violations else f"({rendering[rule_name]} match)")
        summary_lines.append(f"{rule_name:58}" + "".join(f"{cell:>{column_width}}" for cell in cells))
    summary_lines.append("")
    summary_lines.append("Total hands breaking at least one swap rule")
    for label in scenario_labels:
        hands, values, weights, rules = scenarios[label][:4]
        broken = sum(1 for hand in hands if deal_constraints.violated_rule_names(rules, hand))
        summary_lines.append(f"  {label}: {broken}")

    summary_lines += ["", "Per-variable value counts against weight-derived quotas "
                      "(largest-remainder; 'matches' means every value count equals its quota)"]
    for label in scenario_labels:
        rows = [row for row in count_rows if row["scenario"] == label]
        mismatches = [row for row in rows if row["observed_count"] != row["expected_quota"]]
        summary_lines.append(f"  {label}: {len(rows) - len(mismatches)} of {len(rows)} values match their quota")
        for row in mismatches[:6]:
            summary_lines.append(f"      {row['variable']} = {row['value'][:60]!r}: weight {row['weight']}, "
                                 f"quota {row['expected_quota']}, observed {row['observed_count']}")
        if len(mismatches) > 6:
            summary_lines.append(f"      ... and {len(mismatches) - 6} more (see the counts file)")

    summary_lines += ["", "No-stake resolution and rendering check"]
    for label in scenario_labels[1:]:
        records = scenarios[label][5]
        no_stake_records = [r for r in records if NO_STAKE_MARKER in r["variables"]["resolution"]]
        lines_omitted = sum(
            TRADEOFF_LINE_START not in r["prompt"] and DECISION_SCALE_LINE_START not in r["prompt"]
            for r in no_stake_records)
        lines_kept_elsewhere = sum(
            TRADEOFF_LINE_START in r["prompt"] and DECISION_SCALE_LINE_START in r["prompt"]
            for r in records if NO_STAKE_MARKER not in r["variables"]["resolution"])
        summary_lines.append(
            f"  {label}: {len(no_stake_records)} no-stake hands dealt; tradeoff and decision-scale "
            f"lines omitted from {lines_omitted} of them; present in {lines_kept_elsewhere} of "
            f"{len(records) - len(no_stake_records)} other hands")

    summary_lines += ["", "Composer dry-run output per re-deal (violations before repair, swaps made)"]
    for label in scenario_labels[1:]:
        summary_lines.append(f"  [{label}]")
        summary_lines += ["    " + line for line in scenarios[label][4].splitlines()[1:]] or ["    (no rules applied)"]

    write_csv(output_folder / f"rule_violations_by_scenario_{stamp}.csv", violation_rows,
              ["scenario", "rule", "hands_violating", "hands_matching_rendering_rule"])
    write_csv(output_folder / f"per_variable_value_counts_against_weights_{stamp}.csv", count_rows,
              ["scenario", "variable", "value", "weight", "expected_quota", "observed_count"])
    summary_path = output_folder / f"before_and_after_summary_{stamp}.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nwrote outputs to {output_folder}")


if __name__ == "__main__":
    main()
