"""Per-prompt HarmBench attack success and SAGE-Eval pass rate (Animal category excluded) for the Gemma system prompt sweep, each with its difference from the
base model, 95% bootstrap interval (from the sweep analysis CSV) and the p value of the stopping decision tests. Local only.

    python3 scripts/emergent_misalignment/gemma_system_prompt_sweep/tabulate_harmbench_and_sage_eval_per_prompt_with_differences_from_base_model_and_p_values.py

Writes <run folder>/harmbench_and_sage_eval_per_prompt_with_p_values_<timestamp>/ (a markdown table per group and a CSV).
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SWEEP_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SWEEP_DIRECTORY))

import stopping_decisions_for_the_gemma_system_prompt_sweep as decisions  # noqa: E402
from analyse_gemma_system_prompt_sweep_differences_from_base_model_with_tables_and_figures import prompts_in_table_order  # noqa: E402

sweep = decisions.sweep
REPOSITORY_ROOT = decisions.REPOSITORY_ROOT
RUN_FOLDER = REPOSITORY_ROOT / "data" / "emergent_misalignment_runs" / "gemma_system_prompt_sweep_2026-11-16_01-35"
DIFFERENCES_CSV = sorted(RUN_FOLDER.glob("sweep_analysis_*/gemma_system_prompt_sweep_differences_from_base_model_*.csv"))[-1]  # the latest analysis
SWEEP_ANSWERS_FOLDER = RUN_FOLDER / "overnight_outputs" / "run_folders" / "evaluation" / decisions.HARMBENCH_AND_SAGE_KIND
BASE_ANSWERS_FOLDER = (REPOSITORY_ROOT / "data" / "emergent_misalignment_runs" / "overnight_treatment_arm_chain_2026-11-13_23-12" / "overnight_outputs"
                       / "run_folders" / "evaluation" / decisions.BASE_FIRST_SESSION_KIND)
HARMBENCH_MEASURE, SAGE_MEASURE = "HarmBench attack success", "SAGE-Eval pass rate excluding Animal"
NOT_JUDGED_TEXT = "no answers judged yet"
GROUP_HEADINGS = {"A": "how much animal welfare to bring up", "B": "the agent's own food preference, deferring to the user",
                  "C": "purchasing agent, choosing only where the user leaves the choice open", "D": "overriding what the user wants",
                  "E": "unrelated to animal welfare"}
MISMATCH_TOLERANCE_POINTS = 0.1


def format_p_value(p_value):
    return "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"


def format_difference(difference, low, high, p_value):
    return f"{difference:+.1f} ({low:+.1f} to {high:+.1f}), {format_p_value(p_value)}"


def stopping_decision_numbers(prompt_id, base_harmbench_counts, base_sage_passes, animal_prompt_gids):
    """(HarmBench comparison, SAGE-Eval comparison or None when the prompt has no judged SAGE-Eval answers), using the stopping decision's own functions."""
    entry_name = sweep.PROMPTS_BY_ID[prompt_id]["entry_name"]
    harmbench = decisions.harmbench_comparison(decisions.harmbench_counts_by_behavior(SWEEP_ANSWERS_FOLDER, entry_name), base_harmbench_counts)
    sage_passes = decisions.sage_pass_by_prompt(SWEEP_ANSWERS_FOLDER, entry_name)
    sage = decisions.sage_comparison(sage_passes, base_sage_passes, animal_prompt_gids) if sage_passes else None
    return harmbench, sage


def build_rows():
    differences = pd.read_csv(DIFFERENCES_CSV)
    base_harmbench_counts = decisions.harmbench_counts_by_behavior(BASE_ANSWERS_FOLDER, sweep.BASE_MODEL_SHORT_NAME)
    base_sage_passes = decisions.sage_pass_by_prompt(BASE_ANSWERS_FOLDER, sweep.BASE_MODEL_SHORT_NAME)
    animal_prompt_gids = {record["prompt_gid"] for record in decisions.prompts_of_task(SWEEP_ANSWERS_FOLDER, decisions.SAGE_TASK_NAME) if record.get("is_animal_domain")}
    rows, mismatches = [], []
    for prompt in prompts_in_table_order():
        prompt_id = prompt["prompt_id"]
        harmbench, sage = stopping_decision_numbers(prompt_id, base_harmbench_counts, base_sage_passes, animal_prompt_gids)
        row = {"prompt_id": prompt_id, "group": prompt["group"], "strength": prompt["strength"],
               "prompt_text": " ".join(sweep.prompt_text(REPOSITORY_ROOT, prompt_id).split()),
               "harmbench_p_value": harmbench["p_value"], "sage_eval_p_value": sage["p_value"] if sage else None}
        for measure_name, measure, comparison, key in (("harmbench", HARMBENCH_MEASURE, harmbench, "mean_difference_percentage_points"),
                                                       ("sage_eval", SAGE_MEASURE, sage, "difference_percentage_points")):
            csv_row = differences[(differences["prompt_id"] == prompt_id) & (differences["measure"] == measure)]
            analysed = len(csv_row) == 1 and csv_row.iloc[0]["status"] == "analysed" and pd.notna(csv_row.iloc[0]["difference"])
            for column in ("reference_value", "prompt_value", "difference", "interval_low", "interval_high"):
                row[f"{measure_name}_{column}"] = float(csv_row.iloc[0][column]) if analysed else None
            row[f"{measure_name}_difference_from_stopping_decision_functions"] = comparison[key] if comparison else None
            if analysed and comparison and abs(comparison[key] - row[f"{measure_name}_difference"]) > MISMATCH_TOLERANCE_POINTS:
                mismatches.append(f"{prompt_id} {measure}: CSV {row[f'{measure_name}_difference']:.2f}, stopping decision functions {comparison[key]:.2f}")
        rows.append(row)
    return rows, mismatches, differences


def rate_and_difference_cells(row, measure_name):
    if row[f"{measure_name}_difference"] is None or row[f"{measure_name}_p_value"] is None:
        return [NOT_JUDGED_TEXT, NOT_JUDGED_TEXT]
    return [f"{row[f'{measure_name}_prompt_value']:.1f}%",
            format_difference(*(row[f"{measure_name}_{column}"] for column in ("difference", "interval_low", "interval_high", "p_value")))]


def markdown_tables(rows, base_harmbench_rate, base_sage_rate):
    header = "| Prompt | HarmBench attack success | Difference from base model | SAGE-Eval pass rate excluding Animal | Difference from base model |\n|---|---|---|---|---|"
    base_row = f"| Base model, no system prompt | {base_harmbench_rate:.1f}% | — | {base_sage_rate:.1f}% | — |"
    sections = [f"### Reference row\n\n{header}\n{base_row}"]
    for group, heading in GROUP_HEADINGS.items():
        lines = [f"### Group {group}: {heading}", "", header]
        for row in (row for row in rows if row["group"] == group):
            prompt = f'{row["prompt_id"]} ({row["strength"]}): "{row["prompt_text"]}"'
            cells = [prompt] + rate_and_difference_cells(row, "harmbench") + rate_and_difference_cells(row, "sage_eval")
            lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
        sections.append("\n".join(lines))
    intro = ("Attack success (HarmBench, 240 behaviors, 3 answers each) and the SAGE-Eval pass rate (495 prompts, Animal category excluded) under each system prompt. "
             "Each difference is the prompt minus the base model, with a 95% bootstrap interval; the p value is a sign-flip permutation test over HarmBench behaviors "
             "and an exact McNemar test over SAGE-Eval prompts.\n\n")
    return intro + "\n\n".join(sections) + "\n"


def main():
    rows, mismatches, differences = build_rows()
    base_harmbench_rate = differences[differences["measure"] == HARMBENCH_MEASURE]["reference_value"].dropna().iloc[0]
    base_sage_rate = differences[differences["measure"] == SAGE_MEASURE]["reference_value"].dropna().iloc[0]
    timestamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    output_folder = RUN_FOLDER / f"harmbench_and_sage_eval_per_prompt_with_p_values_{timestamp}"
    output_folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output_folder / f"harmbench_and_sage_eval_per_prompt_numbers_with_p_values_{timestamp}.csv", index=False)
    (output_folder / f"harmbench_and_sage_eval_per_prompt_tables_{timestamp}.md").write_text(markdown_tables(rows, base_harmbench_rate, base_sage_rate), encoding="utf-8")
    print(output_folder)
    print("Mismatches between the CSV and the stopping decision functions:", mismatches or "none")


if __name__ == "__main__":
    main()
