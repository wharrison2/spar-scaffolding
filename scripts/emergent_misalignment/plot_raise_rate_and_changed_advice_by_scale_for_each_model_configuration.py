"""Plots, for each model and thinking condition of the baseline welfare-raising rate screen, how often answers raised the animal welfare
cost and how often raising it changed the advice, for all prompts and for three buckets of the number of animals affected. Local only:
no model calls, no AI API, no pod.

    python3 scripts/emergent_misalignment/plot_raise_rate_and_changed_advice_by_scale_for_each_model_configuration.py \
        --run-folder data/emergent_misalignment_runs/baseline_welfare_raising_rate_screen_gemma_qwen_muse_2026-11-03_10-59

Measures (per configuration and group):
  raise rate                 share of all judged answers with raised_welfare_cost == "yes"
  changed-the-advice share   among answers that raised the cost (and have a changed-the-advice verdict), the share with
                             welfare_cost_changed_the_advice == "changed_the_advice"
Intervals: 95% percentile intervals from the bootstrap of analyse_baseline_welfare_raising_rates (whole prompts resampled with replacement
within the group, 2,000 draws, seed 20261103, pooled share per draw), reached through its rate_row function.

Writes to <run folder>/writeup_figures_<YYYY-MM-DD_HH-MM>/: six PNGs (one per configuration), one combined 3x2 PNG, and one CSV of every
plotted number. Reuses analyse_baseline_welfare_raising_rates (verdict loading, scale label, bootstrap).
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
from shared.model_display_names_by_short_name import MODEL_DISPLAY_NAMES  # noqa: E402

ALL_PROMPTS_GROUP = "All prompts"
SCALE_BUCKETS = {
    "A single animal or dozens": ["a single animal", "dozens of animals"],
    "Thousands": ["thousands of animals"],
    "Millions or more": ["millions or billions of animals", "an astronomical number of animals"],
}
EXPECTED_PROMPTS_PER_BUCKET = {"A single animal or dozens": 16, "Thousands": 21, "Millions or more": 13}
GROUP_ORDER = [ALL_PROMPTS_GROUP] + list(SCALE_BUCKETS)
MODEL_LABELS = MODEL_DISPLAY_NAMES
CONDITION_LABELS = {"thinking_off": "thinking off", "thinking_on": "thinking on"}
RAISE_MEASURE = "raised the welfare cost (share of all answers)"
CHANGED_MEASURE = "changed the advice (share of answers that raised it)"
RAISE_LEGEND = "Raised the welfare cost (share of all answers)"
CHANGED_LEGEND = "Changed the advice (share of answers that raised it)"
RAISE_COLOUR, CHANGED_COLOUR = "#2a78d6", "#eb6834"
BACKGROUND_COLOUR, TEXT_COLOUR, SECONDARY_TEXT_COLOUR = "#fcfcfb", "#0b0b0b", "#52514e"
GRIDLINE_COLOUR, ERROR_BAR_COLOUR = "#e3e2de", "#8a8985"
SUBTITLE = "250 answers to 50 prompts; bars show 95% intervals from resampling prompts"
BAR_WIDTH = 0.36
BAR_GAP = 0.03  # about 2 pixels at the figure sizes used
EXTRA_SPACE_AFTER_ALL_PROMPTS = 0.5
DPI = 200


def scale_bucket_for_number_of_animals(number_of_animals):
    for bucket, labels in SCALE_BUCKETS.items():
        if number_of_animals in labels:
            return bucket
    raise ValueError(f"Number of animals {number_of_animals!r} belongs to no scale bucket")


def changed_advice_denominator_mask(answers):
    """Answers that raised the welfare cost and carry a changed-the-advice verdict (the same denominator as the analysis script)."""
    raised = (answers["raised_welfare_cost"] == "yes").to_numpy()
    has_change_verdict = answers["welfare_cost_changed_the_advice"].isin(["changed_the_advice", "mentioned_only"]).to_numpy()
    return raised & has_change_verdict


def prompt_counts_by_group(answers):
    prompts = answers.drop_duplicates("prompt_gid")
    counts = prompts["scale_bucket"].value_counts().to_dict()
    return {ALL_PROMPTS_GROUP: int(len(prompts)), **{bucket: int(counts.get(bucket, 0)) for bucket in SCALE_BUCKETS}}


def plotted_numbers_table(answers):
    rows = []
    for model, condition in analysis.models_and_conditions(answers):
        configuration = f"{MODEL_LABELS[model]}, {CONDITION_LABELS[condition]}"
        configuration_answers = answers[(answers["model_short_name"] == model) & (answers["condition"] == condition)].reset_index(drop=True)
        for group in GROUP_ORDER:
            in_group = np.ones(len(configuration_answers), bool) if group == ALL_PROMPTS_GROUP \
                else (configuration_answers["scale_bucket"] == group).to_numpy()
            group_answers = configuration_answers[in_group].reset_index(drop=True)
            judged = group_answers["raised_welfare_cost"].isin(["yes", "no"]).to_numpy()
            raised = (group_answers["raised_welfare_cost"] == "yes").to_numpy()
            changed = (group_answers["welfare_cost_changed_the_advice"] == "changed_the_advice").to_numpy()
            for measure, numerator_mask, denominator_mask in [(RAISE_MEASURE, raised, judged),
                                                              (CHANGED_MEASURE, changed, changed_advice_denominator_mask(group_answers))]:
                result = analysis.rate_row(group_answers, group, numerator_mask, denominator_mask)
                rows.append({"configuration": configuration, "model_short_name": model, "condition": condition, "group": group,
                             "measure": measure, "numerator": result["events"], "denominator": result["answers"],
                             "rate": result["rate"], "interval_low": result["interval_low"], "interval_high": result["interval_high"],
                             "prompts_in_group": group_answers["prompt_gid"].nunique()})
    return pd.DataFrame(rows)


def configuration_title(model, condition):
    title = f"{MODEL_LABELS[model]}, {CONDITION_LABELS[condition]}"
    if model == "muse_glimmer_30b" and condition == "thinking_off":
        title += " (reasoning strength low)"
    return title


def group_positions():
    positions, position = [], 0.0
    for index in range(len(GROUP_ORDER)):
        positions.append(position)
        position += 1.0 + (EXTRA_SPACE_AFTER_ALL_PROMPTS if index == 0 else 0.0)
    return positions


def draw_configuration_axes(axes, configuration_table, prompt_counts, value_label_size, tick_label_size):
    positions = group_positions()
    offset = (BAR_WIDTH + BAR_GAP) / 2
    for measure, colour, shift in [(RAISE_MEASURE, RAISE_COLOUR, -offset), (CHANGED_MEASURE, CHANGED_COLOUR, offset)]:
        rows = configuration_table[configuration_table["measure"] == measure].set_index("group").loc[GROUP_ORDER]
        rates = 100 * rows["rate"].to_numpy(float)
        x_values = np.array(positions) + shift
        axes.bar(x_values, np.nan_to_num(rates), width=BAR_WIDTH, color=colour, zorder=3)
        low, high = 100 * rows["interval_low"].to_numpy(float), 100 * rows["interval_high"].to_numpy(float)
        has_interval = ~np.isnan(low)
        axes.errorbar(x_values[has_interval], rates[has_interval],
                      yerr=[rates[has_interval] - low[has_interval], high[has_interval] - rates[has_interval]],
                      fmt="none", ecolor=ERROR_BAR_COLOUR, elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
        for x_value, rate, interval_high, denominator in zip(x_values, rates, high, rows["denominator"].to_numpy()):
            top = interval_high if not np.isnan(interval_high) else (rate if not np.isnan(rate) else 0)
            text = "n/a" if denominator == 0 else f"{rate:.0f}%"
            axes.text(x_value, top + 1.5, text, ha="center", va="bottom", fontsize=value_label_size, color=TEXT_COLOUR, zorder=5)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"{group}\n({prompt_counts[group]} prompts)" for group in GROUP_ORDER], fontsize=tick_label_size,
                         color=TEXT_COLOUR)
    axes.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    axes.set_ylim(0, 112)
    axes.set_yticks(range(0, 101, 20))
    axes.set_yticklabels([f"{value}%" for value in range(0, 101, 20)], fontsize=tick_label_size, color=SECONDARY_TEXT_COLOUR)
    axes.set_axisbelow(True)
    axes.yaxis.grid(True, color=GRIDLINE_COLOUR, linewidth=0.7, zorder=0)
    axes.xaxis.grid(False)
    axes.set_facecolor(BACKGROUND_COLOUR)
    for side in ["top", "right"]:
        axes.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axes.spines[side].set_color(GRIDLINE_COLOUR)
    axes.tick_params(axis="both", length=0)


def legend_handles():
    return [plt.Rectangle((0, 0), 1, 1, color=RAISE_COLOUR), plt.Rectangle((0, 0), 1, 1, color=CHANGED_COLOUR)], \
        [RAISE_LEGEND, CHANGED_LEGEND]


def configuration_table_for(table, model, condition):
    return table[(table["model_short_name"] == model) & (table["condition"] == condition)]


def plot_single_configuration(table, prompt_counts, model, condition, output_path):
    figure, axes = plt.subplots(figsize=(9.2, 6.0), facecolor=BACKGROUND_COLOUR)
    draw_configuration_axes(axes, configuration_table_for(table, model, condition), prompt_counts, 10.5, 10.5)
    axes.set_ylabel("Share of answers", fontsize=11, color=TEXT_COLOUR)
    handles, labels = legend_handles()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.875), ncol=2, frameon=False, fontsize=10.5,
                  labelcolor=TEXT_COLOUR, columnspacing=1.6, handlelength=1.2)
    figure.text(0.06, 0.965, configuration_title(model, condition), fontsize=15, fontweight="bold", color=TEXT_COLOUR, va="top")
    figure.text(0.06, 0.92, SUBTITLE, fontsize=10.5, color=SECONDARY_TEXT_COLOUR, va="top")
    figure.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.14)
    figure.savefig(output_path, dpi=DPI, facecolor=BACKGROUND_COLOUR)
    plt.close(figure)


def plot_combined_figure(table, prompt_counts, models, output_path):
    figure, all_axes = plt.subplots(len(models), 2, figsize=(17, 6.2 * len(models)), sharey=True, facecolor=BACKGROUND_COLOUR)
    for row, model in enumerate(models):
        for column, condition in enumerate(analysis.CONDITIONS):
            axes = all_axes[row][column]
            draw_configuration_axes(axes, configuration_table_for(table, model, condition), prompt_counts, 9.5, 9.5)
            axes.set_title(configuration_title(model, condition), fontsize=12.5, fontweight="bold", color=TEXT_COLOUR, loc="left", pad=8)
            if column == 0:
                axes.set_ylabel("Share of answers", fontsize=11, color=TEXT_COLOUR)
    handles, labels = legend_handles()
    figure.text(0.04, 0.985, "Welfare cost raised and advice changed, by number of animals affected", fontsize=17, fontweight="bold",
                color=TEXT_COLOUR, va="top")
    figure.text(0.04, 0.967, SUBTITLE, fontsize=11.5, color=SECONDARY_TEXT_COLOUR, va="top")
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False, fontsize=11.5,
                  labelcolor=TEXT_COLOUR, columnspacing=1.6, handlelength=1.2)
    figure.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.05, hspace=0.38, wspace=0.08)
    figure.savefig(output_path, dpi=DPI, facecolor=BACKGROUND_COLOUR)
    plt.close(figure)


def judged_answers_with_scale_buckets(run_folder):
    answers = analysis.build_judged_answers_table(run_folder)
    if (answers["verdict_status"] != "ok").any():
        raise ValueError(f"{int((answers['verdict_status'] != 'ok').sum())} answers lack an ok verdict")
    answers["scale_bucket"] = answers["number_of_animals"].map(scale_bucket_for_number_of_animals)
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True, type=Path)
    arguments = parser.parse_args()
    run_folder = arguments.run_folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    answers = judged_answers_with_scale_buckets(run_folder)
    prompt_counts = prompt_counts_by_group(answers)
    print("Prompts per group:", prompt_counts)
    if any(prompt_counts[bucket] != expected for bucket, expected in EXPECTED_PROMPTS_PER_BUCKET.items()):
        raise ValueError(f"Prompt counts {prompt_counts} differ from the expected {EXPECTED_PROMPTS_PER_BUCKET}")
    table = plotted_numbers_table(answers)
    output_folder = run_folder / f"writeup_figures_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_folder / f"plotted_numbers_raise_rate_and_changed_advice_share_by_scale_{timestamp}.csv", index=False)
    configurations = analysis.models_and_conditions(answers)
    for model, condition in configurations:
        plot_single_configuration(table, prompt_counts, model, condition, output_folder /
                                  f"raise_rate_and_changed_advice_share_by_scale_{model}_{condition}_{timestamp}.png")
    models = [model for model in analysis.MODEL_ORDER if any(m == model for m, _ in configurations)]
    plot_combined_figure(table, prompt_counts, models, output_folder /
                         f"raise_rate_and_changed_advice_share_by_scale_all_models_and_conditions_combined_{timestamp}.png")
    print(f"Wrote {output_folder}")


if __name__ == "__main__":
    main()
