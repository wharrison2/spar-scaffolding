"""Figure for the treatment arm report: base model, base model with the "do not consider animal welfare" system prompt, and the
treatment adapter, on the four evaluations that carry the main comparison (held-out welfare raising, HarmBench compliance, SAGE-Eval
safety facts raised, MMLU-Pro accuracy).

Values are read from the run's results markdown (evaluation_results_latest.md, written by
write_markdown_of_evaluation_results_so_far_for_overnight_treatment_arm_run.py), so the figure always matches the tables.
Colours: the dataviz skill's validated categorical slots 1-3 (light surface), each bar labelled with its value.

Usage: python3 plot_treatment_arm_key_evaluation_comparison_for_report.py --run-folder <run folder> --output-folder <folder>
"""
import argparse
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


def base_model_display_name(run_folder):
    """The display name of the base model in the run folder's chain configuration (Gemma 4 31B when it names none)."""
    configuration_files = sorted(Path(run_folder).glob("run_configuration_overnight_chain_*.json"))
    configuration = json.loads(configuration_files[-1].read_text(encoding="utf-8")) if configuration_files else {}
    return display_name_of_model(configuration.get("base_model_short_name", DEFAULT_BASE_MODEL_SHORT_NAME))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    arguments = parser.parse_args(argv)
    markdown_text = (arguments.run_folder / "evaluation_results_latest.md").read_text(encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
                         "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY})
    figure, axes_grid = plt.subplots(2, 2, figsize=(10, 7.2), facecolor=SURFACE)
    for axes, (heading, column_index, title) in zip(axes_grid.flat, PANELS):
        values = read_percentages(markdown_text, heading, column_index)
        names = list(CONDITION_ROW_LABELS)
        heights = [values[name] for name in names]
        bars = axes.bar(range(3), heights, width=0.62, color=CONDITION_COLOURS, edgecolor=SURFACE, linewidth=2)
        for bar, height in zip(bars, heights):
            axes.text(bar.get_x() + bar.get_width() / 2, height + 1.5, f"{height:.0f}%", ha="center", va="bottom",
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
    output_path = arguments.output_folder / f"treatment_arm_key_evaluations_base_system_prompt_and_trained_model_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.png"
    figure.savefig(output_path, dpi=160, facecolor=SURFACE)
    print(output_path)


if __name__ == "__main__":
    main()
