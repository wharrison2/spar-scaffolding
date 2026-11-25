"""Figure for the treatment arm report: base model, base model with the "do not consider animal welfare" system prompt, and the
treatment adapter, on the four evaluations that carry the main comparison (held-out welfare raising, HarmBench compliance, SAGE-Eval
safety facts raised, MMLU-Pro accuracy).

Values are read from the run's results markdown (evaluation_results_latest.md, written by
write_markdown_of_evaluation_results_so_far_for_overnight_treatment_arm_run.py), so the figure always matches the tables.
For a run without that markdown (the Qwen run), --evaluation-folder reads the latest analysis files instead, taking the HarmBench
and MMLU-Pro values from the completed-answer analyses (answers that hit the output cap, finished with a larger limit).
Colours: the dataviz skill's validated categorical slots 1-3 (light surface), each bar labelled with its value.

Usage: python3 plot_treatment_arm_key_evaluation_comparison_for_report.py --run-folder <run folder> --output-folder <folder>
       [--evaluation-folder <run folder>/overnight_outputs/run_folders/evaluation] [--decimal-places 1]
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.model_display_names_by_short_name import display_name_of_model  # noqa: E402

DEFAULT_BASE_MODEL_SHORT_NAME = "gemma_4_31b_it"

CONDITION_ROW_LABELS = {
    "base model": "Base model",
    'base model with the "do not consider animal welfare" system prompt': "Base model +\nsystem prompt",
    "treatment LoRA adapter": "Trained model\n(no system prompt)",
}
CONDITION_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# (section heading in the results markdown, column index of the percentage after "Model condition | Progress |", panel title)
PANELS = [
    ("## Held-out set", 0, "Raises the animal welfare cost\n(held-out animal welfare dilemmas)"),
    ("## HarmBench direct requests", 0, "Complies with clearly harmful requests\n(HarmBench, official classifier)"),
    ("## SAGE-Eval pass rate", 0, "Raises the relevant safety fact\n(SAGE-Eval, excluding the animal domain)"),
    ("## MMLU-Pro accuracy", 0, "Answers correctly\n(MMLU-Pro capability check)"),
]


def read_percentages(markdown_text, section_heading, column_index):
    """Returns {condition label: percentage} from the first table after the section heading."""
    section = markdown_text.split(section_heading, 1)[1]
    values = {}
    for line in section.splitlines():
        if line.startswith("## ") and values:
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] in CONDITION_ROW_LABELS:
            match = re.search(r"(\d+(?:\.\d+)?)%", cells[2 + column_index])
            if match:
                values[cells[0]] = float(match.group(1))
        elif values and not line.strip():
            break
    missing = [name for name in CONDITION_ROW_LABELS if name not in values]
    if missing:
        raise SystemExit(f"{section_heading}: no percentage for {missing}")
    return values


def latest_file(folder_pattern_parent, folder_glob, file_glob):
    """The newest file matching file_glob inside the last (by name, so latest-dated) folder matching folder_glob."""
    folders = sorted(Path(folder_pattern_parent).glob(folder_glob))
    for folder in reversed(folders):
        files = sorted(folder.glob(file_glob))
        if files:
            return files[-1]
    raise SystemExit(f"no file {folder_glob}/{file_glob} under {folder_pattern_parent}")


def condition_of_model_entry(entry_name):
    """Maps a model entry name in an analysis file to the condition label of the figure, or None for entries not plotted."""
    if "with_lora_treatment_arm" in entry_name or "treatment adapter" in entry_name:
        if "system_prompt" in entry_name or "with the animal-welfare system prompt" in entry_name:
            return None
        return "treatment LoRA adapter"
    if "system_prompt" in entry_name or "with the animal-welfare system prompt" in entry_name:
        return 'base model with the "do not consider animal welfare" system prompt'
    return "base model"


def read_percentages_from_analysis_files(evaluation_folder):
    """Returns the four panels' values, one dict per panel in PANELS order, read from the latest analysis files.

    HarmBench is the completed-answer rate with the official 512-token clip; MMLU-Pro is the completed-answer accuracy
    (the analyses dated after the completion pass).
    """
    evaluation_folder = Path(evaluation_folder)

    def rows(path):
        with open(path, newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    held_out_path = latest_file(evaluation_folder / "held_out", "analysis_*", "welfare_raising_rates_base_versus_lora_adapters_*.csv")
    held_out = {condition_of_model_entry(row["model_short_name"]): 100 * float(row["rate"]) for row in rows(held_out_path)
                if row["measure"] == "raised the welfare cost, all judged answers" and condition_of_model_entry(row["model_short_name"])}
    harmbench_path = latest_file(evaluation_folder / "sycophancy_harmbench_sage_eval", "analysis_*", "harmbench_attack_success_rate_original_and_completed_*.csv")
    harmbench = {condition_of_model_entry(row["model_entry"]): 100 * float(row["completed_official_clip_attack_success_rate"])
                 for row in rows(harmbench_path) if condition_of_model_entry(row["model_entry"])}
    sage_path = latest_file(evaluation_folder / "sycophancy_harmbench_sage_eval", "analysis_sage_eval_*", "sage_eval_rates_per_model_entry_and_scope_*.csv")
    sage = {condition_of_model_entry(row["model_short_name"]): 100 * float(row["rate"]) for row in rows(sage_path)
            if row["scope_type"] == "main_score" and row["measure"] == "pass" and condition_of_model_entry(row["model_short_name"])}
    mmlu_path = latest_file(evaluation_folder / "mmlu_pro", "analysis_*", "mmlu_pro_accuracy_per_model_entry_and_category_*.csv")
    mmlu = {condition_of_model_entry(row["model_short_name"]): 100 * float(row["accuracy"]) for row in rows(mmlu_path)
            if row["category"] == "all categories" and condition_of_model_entry(row["model_short_name"])}
    print("values read from:", held_out_path, harmbench_path, sage_path, mmlu_path, sep="\n  ")
    return [held_out, harmbench, sage, mmlu]


def base_model_short_name(run_folder):
    """The base model short name in the run folder's chain configuration (Gemma 4 31B when it names none)."""
    configuration_files = sorted(Path(run_folder).glob("run_configuration_overnight_chain_*.json"))
    configuration = json.loads(configuration_files[-1].read_text(encoding="utf-8")) if configuration_files else {}
    return configuration.get("base_model_short_name", DEFAULT_BASE_MODEL_SHORT_NAME)


def base_model_display_name(run_folder):
    return display_name_of_model(base_model_short_name(run_folder))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--evaluation-folder", type=Path, help="read values from the latest analysis files in this evaluation folder instead of evaluation_results_latest.md")
    parser.add_argument("--decimal-places", type=int, default=0, help="decimal places in the bar labels")
    arguments = parser.parse_args(argv)
    if arguments.evaluation_folder:
        panel_values = read_percentages_from_analysis_files(arguments.evaluation_folder)
    else:
        markdown_text = (arguments.run_folder / "evaluation_results_latest.md").read_text(encoding="utf-8")
        panel_values = [read_percentages(markdown_text, heading, column_index) for heading, column_index, _ in PANELS]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
                         "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY})
    figure, axes_grid = plt.subplots(2, 2, figsize=(10, 7.2), facecolor=SURFACE)
    for axes, values, (_, _, title) in zip(axes_grid.flat, panel_values, PANELS):
        missing = [name for name in CONDITION_ROW_LABELS if name not in values]
        if missing:
            raise SystemExit(f"{title}: no value for {missing}")
        names = list(CONDITION_ROW_LABELS)
        heights = [values[name] for name in names]
        bars = axes.bar(range(3), heights, width=0.62, color=CONDITION_COLOURS, edgecolor=SURFACE, linewidth=2)
        for bar, height in zip(bars, heights):
            axes.text(bar.get_x() + bar.get_width() / 2, height + 1.5, f"{height:.{arguments.decimal_places}f}%", ha="center", va="bottom",
                      color=TEXT_PRIMARY, fontsize=11, fontweight="bold")
        axes.set_facecolor(SURFACE)
        axes.set_xticks(range(3), [CONDITION_ROW_LABELS[name] for name in names], fontsize=8.5)
        axes.set_ylim(0, 112)
        axes.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        axes.grid(axis="y", color=GRID, linewidth=0.8)
        axes.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.tick_params(axis="both", length=0)
        axes.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
    figure.suptitle(f"{base_model_display_name(arguments.run_folder)}: base model, base model with system prompt, and trained model", fontsize=11, color=TEXT_PRIMARY)
    figure.tight_layout(h_pad=2.5)
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_folder / f"treatment_arm_key_evaluations_base_system_prompt_and_trained_model_{base_model_short_name(arguments.run_folder)}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.png"
    figure.savefig(output_path, dpi=160, facecolor=SURFACE)
    print(output_path)


if __name__ == "__main__":
    main()
