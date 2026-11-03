"""Plots, for each model and thinking condition of the baseline welfare-raising rate screen, how consistent the welfare-raising verdict is
within each prompt: for every prompt, how many of its 5 answers have raised_welfare_cost == "yes" (0 to 5), and a histogram of those counts
over the 50 prompts. Local only: no model calls, no AI API, no pod.

    python3 scripts/emergent_misalignment/plot_per_prompt_consistency_of_welfare_raising_for_each_model_configuration.py \
        --run-folder data/emergent_misalignment_runs/baseline_welfare_raising_rate_screen_gemma_qwen_muse_2026-11-03_10-59

Writes to <run folder>/writeup_figures_per_prompt_consistency_<YYYY-MM-DD_HH-MM>/: six PNGs (one per configuration), one combined 3x2 PNG
with shared axes, a CSV of per-prompt counts and a CSV of the histogram counts. All six figures share one y limit so they compare.
Reuses analyse_baseline_welfare_raising_rates (verdict loading: latest ok verdict per answer) and the title and style constants of
plot_raise_rate_and_changed_advice_by_scale_for_each_model_configuration.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import analyse_baseline_welfare_raising_rates as analysis  # noqa: E402
from emergent_misalignment import plot_raise_rate_and_changed_advice_by_scale_for_each_model_configuration as shared_plotting  # noqa: E402

ANSWERS_PER_PROMPT = 5
EXPECTED_PROMPTS = 50
X_AXIS_LABEL = "Answers that raised the welfare cost, out of 5"
Y_AXIS_LABEL = "Prompts"
BAR_COLOUR = shared_plotting.RAISE_COLOUR
BACKGROUND_COLOUR = shared_plotting.BACKGROUND_COLOUR
TEXT_COLOUR = shared_plotting.TEXT_COLOUR
SECONDARY_TEXT_COLOUR = shared_plotting.SECONDARY_TEXT_COLOUR
GRIDLINE_COLOUR = shared_plotting.GRIDLINE_COLOUR
DPI = shared_plotting.DPI
configuration_title = shared_plotting.configuration_title


def per_prompt_counts_table(answers):
    """One row per configuration and prompt: answers that raised the welfare cost and answers in total."""
    rows = []
    for model, condition in analysis.models_and_conditions(answers):
        configuration_answers = answers[(answers["model_short_name"] == model) & (answers["condition"] == condition)]
        for prompt_gid, prompt_answers in configuration_answers.groupby("prompt_gid", sort=True):
            rows.append({"configuration": configuration_title(model, condition), "model_short_name": model, "condition": condition,
                         "prompt_gid": prompt_gid, "number_of_animals": prompt_answers["number_of_animals"].iloc[0],
                         "answers_raised_welfare_cost": int((prompt_answers["raised_welfare_cost"] == "yes").sum()),
                         "answers_total": int(len(prompt_answers))})
    return pd.DataFrame(rows)


def histogram_counts_table(per_prompt_counts):
    """One row per configuration and bin 0..5: how many prompts have that many raising answers, plus the pooled raise rate."""
    rows = []
    for (model, condition), group in per_prompt_counts.groupby(["model_short_name", "condition"], sort=False):
        pooled_rate = group["answers_raised_welfare_cost"].sum() / group["answers_total"].sum()
        for raised in range(ANSWERS_PER_PROMPT + 1):
            rows.append({"configuration": configuration_title(model, condition), "model_short_name": model, "condition": condition,
                         "answers_raised_welfare_cost": raised, "bin_label": f"{raised}/{ANSWERS_PER_PROMPT}",
                         "prompts": int((group["answers_raised_welfare_cost"] == raised).sum()),
                         "prompts_total": int(len(group)), "pooled_raise_rate_percent": round(100 * pooled_rate, 2)})
    return pd.DataFrame(rows)


def y_limit_for(histogram):
    """One integer y limit with headroom for the count labels, shared by all figures."""
    return int(histogram["prompts"].max()) + 5


def y_tick_step_for(y_limit):
    return 2 if y_limit <= 24 else 5


def draw_histogram_axes(axes, configuration_histogram, y_limit, label_size, tick_label_size):
    heights = configuration_histogram.sort_values("answers_raised_welfare_cost")["prompts"].to_numpy()
    positions = np.arange(ANSWERS_PER_PROMPT + 1)
    axes.bar(positions, heights, width=0.8, color=BAR_COLOUR, zorder=3)
    for position, height in zip(positions, heights):
        axes.text(position, height + y_limit * 0.012, str(int(height)), ha="center", va="bottom", fontsize=label_size, color=TEXT_COLOUR,
                  zorder=5)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"{value}/{ANSWERS_PER_PROMPT}" for value in positions], fontsize=tick_label_size, color=TEXT_COLOUR)
    axes.set_xlim(-0.6, ANSWERS_PER_PROMPT + 0.6)
    axes.set_ylim(0, y_limit)
    ticks = list(range(0, y_limit + 1, y_tick_step_for(y_limit)))
    axes.set_yticks(ticks)
    axes.set_yticklabels([str(tick) for tick in ticks], fontsize=tick_label_size, color=SECONDARY_TEXT_COLOUR)
    axes.set_axisbelow(True)
    axes.yaxis.grid(True, color=GRIDLINE_COLOUR, linewidth=0.7, zorder=0)
    axes.xaxis.grid(False)
    axes.set_facecolor(BACKGROUND_COLOUR)
    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axes.spines[side].set_color(GRIDLINE_COLOUR)
    axes.tick_params(axis="both", length=0)


def subtitle_for(configuration_histogram):
    rate = configuration_histogram["pooled_raise_rate_percent"].iloc[0]
    prompts = configuration_histogram["prompts_total"].iloc[0]
    return f"{prompts} prompts, {ANSWERS_PER_PROMPT} answers each; overall raise rate {rate:.0f}%"


def configuration_histogram_for(histogram, model, condition):
    return histogram[(histogram["model_short_name"] == model) & (histogram["condition"] == condition)]


def plot_single_configuration(histogram, y_limit, model, condition, output_path):
    configuration_histogram = configuration_histogram_for(histogram, model, condition)
    figure, axes = plt.subplots(figsize=(8.0, 5.6), facecolor=BACKGROUND_COLOUR)
    draw_histogram_axes(axes, configuration_histogram, y_limit, 11, 10.5)
    axes.set_xlabel(X_AXIS_LABEL, fontsize=11, color=TEXT_COLOUR, labelpad=8)
    axes.set_ylabel(Y_AXIS_LABEL, fontsize=11, color=TEXT_COLOUR)
    figure.text(0.06, 0.965, configuration_title(model, condition), fontsize=15, fontweight="bold", color=TEXT_COLOUR, va="top")
    figure.text(0.06, 0.915, subtitle_for(configuration_histogram), fontsize=10.5, color=SECONDARY_TEXT_COLOUR, va="top")
    figure.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.14)
    figure.savefig(output_path, dpi=DPI, facecolor=BACKGROUND_COLOUR)
    plt.close(figure)


def plot_combined_figure(histogram, y_limit, models, output_path):
    figure, all_axes = plt.subplots(len(models), 2, figsize=(13, 4.6 * len(models)), sharex=True, sharey=True, facecolor=BACKGROUND_COLOUR)
    for row, model in enumerate(models):
        for column, condition in enumerate(analysis.CONDITIONS):
            axes = all_axes[row][column]
            configuration_histogram = configuration_histogram_for(histogram, model, condition)
            draw_histogram_axes(axes, configuration_histogram, y_limit, 10, 10)
            axes.set_title(configuration_title(model, condition), fontsize=12.5, fontweight="bold", color=TEXT_COLOUR, loc="left", pad=22)
            axes.text(0, 1.015, subtitle_for(configuration_histogram), transform=axes.transAxes, fontsize=9.5,
                      color=SECONDARY_TEXT_COLOUR, va="bottom", ha="left")
            if column == 0:
                axes.set_ylabel(Y_AXIS_LABEL, fontsize=11, color=TEXT_COLOUR)
            if row == len(models) - 1:
                axes.set_xlabel(X_AXIS_LABEL, fontsize=11, color=TEXT_COLOUR, labelpad=8)
    figure.text(0.04, 0.99, "How consistently each prompt's answers raised the welfare cost", fontsize=17, fontweight="bold",
                color=TEXT_COLOUR, va="top")
    figure.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.06, hspace=0.5, wspace=0.08)
    figure.savefig(output_path, dpi=DPI, facecolor=BACKGROUND_COLOUR)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True, type=Path)
    run_folder = parser.parse_args().run_folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    answers = shared_plotting.judged_answers_with_scale_buckets(run_folder)
    per_prompt_counts = per_prompt_counts_table(answers)
    if (per_prompt_counts["answers_total"] != ANSWERS_PER_PROMPT).any():
        raise ValueError("Some prompt does not have exactly 5 answers in some configuration")
    if (per_prompt_counts.groupby(["model_short_name", "condition"]).size() != EXPECTED_PROMPTS).any():
        raise ValueError(f"Some configuration does not have {EXPECTED_PROMPTS} prompts")
    histogram = histogram_counts_table(per_prompt_counts)
    y_limit = y_limit_for(histogram)
    output_folder = run_folder / f"writeup_figures_per_prompt_consistency_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    per_prompt_counts.to_csv(output_folder / f"per_prompt_counts_of_answers_that_raised_welfare_cost_{timestamp}.csv", index=False)
    histogram.to_csv(output_folder / f"histogram_counts_of_prompts_by_answers_that_raised_welfare_cost_{timestamp}.csv", index=False)
    configurations = analysis.models_and_conditions(answers)
    for model, condition in configurations:
        plot_single_configuration(histogram, y_limit, model, condition, output_folder /
                                  f"per_prompt_consistency_of_welfare_raising_histogram_{model}_{condition}_{timestamp}.png")
    models = [model for model in analysis.MODEL_ORDER if any(m == model for m, _ in configurations)]
    plot_combined_figure(histogram, y_limit, models, output_folder /
                         f"per_prompt_consistency_of_welfare_raising_histogram_all_models_and_conditions_combined_{timestamp}.png")
    print(f"Wrote {output_folder}")
    summary = histogram.pivot(index="configuration", columns="bin_label", values="prompts")
    summary["pooled_rate_percent"] = histogram.groupby("configuration")["pooled_raise_rate_percent"].first()
    summary["unanimous"] = summary["0/5"] + summary["5/5"]
    print(summary.to_string())


if __name__ == "__main__":
    main()
