"""Differences from the base model for every Gemma system prompt sweep prompt that has answers, with tables and figures. Local only: no model call, no pod.

    python3 scripts/emergent_misalignment/gemma_system_prompt_sweep/analyse_gemma_system_prompt_sweep_differences_from_base_model_with_tables_and_figures.py \
        --run-folder data/emergent_misalignment_runs/gemma_system_prompt_sweep_<timestamp> [--first-session-run-folder ...] [--second-session-run-folder ...]

Reads <run folder>/overnight_outputs/run_folders/evaluation/{harmbench_and_sage_eval,held_out,betley,food_choice_check} and the stopping rule's
overnight_outputs/sweep_state/sweep_skipped_prompts.json. The base model rows come from the first session's folders (entry gemma_4_31b_it, thinking_off) and the
neutral prompt (E1) Betley rows from the second session's betley_new_conditions folder. Differences are the prompt's value minus the base model's, in points,
with 95% intervals:
 - HarmBench attack success, SAGE-Eval pass rate excluding Animal, held-out welfare raising (raised_welfare_cost yes among judged answers): whole behaviours,
   facts or prompts resampled with both sides kept together (adapter_evaluation_bootstrap_statistics.paired_rate_difference_over_prompts)
 - Betley mean alignment on coherent answers, "what is your wish" and "ruler of the world", against the base model and against E1, with a permutation p value
Prompts the stopping rule skipped appear with its reason. Writes <run folder>/sweep_analysis_<timestamp>/ (a CSV, a markdown table, two figures) and runs the food
choice check analysis into the food choice check run folder.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SWEEP_DIRECTORY = Path(__file__).resolve().parent
EMERGENT_DIRECTORY = SWEEP_DIRECTORY.parent
SCRIPTS_DIRECTORY = EMERGENT_DIRECTORY.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (SCRIPTS_DIRECTORY, SWEEP_DIRECTORY, EMERGENT_DIRECTORY / "food_choice_check"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import gemma_system_prompt_sweep_definition as sweep  # noqa: E402
from emergent_misalignment import adapter_evaluation_bootstrap_statistics as statistics  # noqa: E402
from emergent_misalignment import analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters as betley_scores  # noqa: E402
from emergent_misalignment import analyse_harmbench_attack_success_rate_base_versus_lora_adapters as harmbench_analysis  # noqa: E402
from emergent_misalignment import analyse_held_out_set_welfare_raising_base_versus_lora_adapters as held_out_analysis  # noqa: E402
from emergent_misalignment import analyse_sage_eval_pass_rate_base_versus_lora_adapters as sage_analysis  # noqa: E402
from emergent_misalignment.analyse_baseline_welfare_raising_rates import markdown_table  # noqa: E402
from emergent_misalignment.betley_mean_alignment_on_coherent_answers_difference_between_entries import (  # noqa: E402
    ALL_QUESTIONS_LABEL, coherent_alignment_scores, compare_mean_alignment)

RUNS_FOLDER = REPOSITORY_ROOT / "data" / "emergent_misalignment_runs"
DEFAULT_FIRST_SESSION_RUN_FOLDER = RUNS_FOLDER / "overnight_treatment_arm_chain_2026-11-13_23-12"
DEFAULT_SECOND_SESSION_RUN_FOLDER = (RUNS_FOLDER / "overnight_treatment_arm_chain_qwen3_8_27b_trained_on_gemma_4_31b_it_training_examples_2026-11-14_22-50")
EVALUATION_FOLDER_RELATIVE_PATH = Path("overnight_outputs") / "run_folders" / "evaluation"
SKIPPED_PROMPTS_RELATIVE_PATH = Path("overnight_outputs") / "sweep_state" / "sweep_skipped_prompts.json"
SECOND_SESSION_BETLEY_RELATIVE_PATH = EVALUATION_FOLDER_RELATIVE_PATH / "gemma_4_31b_it" / "betley_new_conditions"
SAGE_HAIKU_VERDICT_FILE_PATTERN = "sage_eval_haiku_judge_verdicts__{name}.jsonl"
NEUTRAL_PROMPT_ENTRY_NAME = sweep.BASE_MODEL_SHORT_NAME + sweep.ENTRY_NAME_INFIX + "neutral_control_answer_in_plain_language"
BETLEY_QUESTION_IDS = ["what_is_your_wish", "ruler_of_the_world"]
BASE_MODEL_LABEL, NEUTRAL_PROMPT_LABEL = "base model", "E1 neutral prompt"

HARMBENCH, SAGE_EVAL, HELD_OUT = "HarmBench attack success", "SAGE-Eval pass rate excluding Animal", "Held-out welfare raising"
BETLEY_MEASURE = "Betley mean alignment on coherent answers"
RATE_MEASURES = [HARMBENCH, SAGE_EVAL, HELD_OUT]
FRAME_COLUMNS = ["model_short_name", "condition", "unit_id", "event", "eligible"]
CSV_COLUMNS = ["prompt_id", "group", "direction", "list_name", "status", "skip_reason", "measure", "question", "compared_with", "unit", "reference_value",
               "prompt_value", "difference", "interval_low", "interval_high", "permutation_p_value", "reference_answers", "prompt_answers"]

GROUP_COLOURS = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73", "D": "#CC79A7", "E": "#4D4D4D"}
DIRECTION_MARKERS = {sweep.AGAINST: "v", sweep.TOWARDS: "^", sweep.CONTROL: "s", sweep.UNRELATED: "D"}
DIRECTION_COLOURS = {sweep.AGAINST: "#D55E00", sweep.TOWARDS: "#0072B2"}
AXIS_LABELS = {HARMBENCH: "Change in HarmBench attack success from the base model\n(percentage points)",
               SAGE_EVAL: "Change in SAGE-Eval pass rate, excluding Animal,\nfrom the base model\n(percentage points)",
               HELD_OUT: "Change in held-out welfare raising from the base model\n(percentage points)"}


# ----- loading: one frame per rate measure, one row per answer ---------------------------------------------------------------------

def has_answers(folder):
    return any((Path(folder) / "generation_outputs").glob("answers__*__*.jsonl"))


def answer_frame(table, unit_ids, events, eligible):
    return pd.DataFrame({"model_short_name": table["model_short_name"], "condition": table["condition"], "unit_id": unit_ids,
                         "event": events.astype(bool), "eligible": eligible.astype(bool)})[FRAME_COLUMNS]


def harmbench_frame(folder):
    if not has_answers(folder):
        return pd.DataFrame(columns=FRAME_COLUMNS)
    _, table, _ = harmbench_analysis.build_labelled_answers_table(folder)
    return answer_frame(table, table["behavior_id"], table["attack_succeeded"] & table["labelled"], table["labelled"])


def sage_eval_frame(folder):
    if not has_answers(folder):
        return pd.DataFrame(columns=FRAME_COLUMNS)
    _, table = sage_analysis.build_judged_answers_table(folder, SAGE_HAIKU_VERDICT_FILE_PATTERN)
    eligible = table["judged"] & ~table["is_animal_domain"]
    return answer_frame(table, table["fact_id"], table["pass"] & eligible, eligible)


def held_out_frame(folder):
    if not has_answers(folder):
        return pd.DataFrame(columns=FRAME_COLUMNS)
    table = held_out_analysis.build_judged_answers_table(folder)
    judged = table["verdict_status"] == "ok"
    return answer_frame(table, table["prompt_gid"], (table["raised_welfare_cost"] == "yes") & judged, judged)


def betley_scores_table(folder):
    if not has_answers(folder):
        return pd.DataFrame(columns=["model_short_name", "condition", "prompt_gid", "alignment_score", "coherence_score", "score_status"])
    return betley_scores.build_scored_answers_table(folder, betley_scores.judge.DEFAULT_JUDGE_MODEL)


# ----- differences -----------------------------------------------------------------------------------------------------------------

def rate_difference(base_frame, prompt_frame, seed_offset):
    """The prompt's rate minus the base model's, in percent and percentage points with a 95% interval, or None when either side has no judged answers."""
    per_unit = [frame.groupby("unit_id").agg(events=("event", "sum"), eligible=("eligible", "sum")) for frame in (prompt_frame, base_frame)]
    if any(counts["eligible"].sum() == 0 for counts in per_unit):
        return None
    prompt_counts, base_counts = per_unit
    difference, low, high = statistics.paired_rate_difference_over_prompts(
        prompt_counts.index, prompt_counts["events"], prompt_counts["eligible"], base_counts.index, base_counts["events"], base_counts["eligible"], seed_offset)
    return {"unit": "percentage points", "reference_value": 100 * base_counts["events"].sum() / base_counts["eligible"].sum(),
            "prompt_value": 100 * prompt_counts["events"].sum() / prompt_counts["eligible"].sum(), "difference": 100 * difference,
            "interval_low": 100 * low, "interval_high": 100 * high, "permutation_p_value": np.nan,
            "reference_answers": int(base_counts["eligible"].sum()), "prompt_answers": int(prompt_counts["eligible"].sum())}


def betley_differences(prompt_scores, reference_scores, seed_offset):
    """{question: result dict} of the prompt's mean alignment minus the reference's, for each Betley question and all of them."""
    results = {}
    for row in compare_mean_alignment(prompt_scores, reference_scores, BETLEY_QUESTION_IDS, seed_offset):
        if row["count_first"] and row["count_second"]:
            results[row["question"]] = {"unit": "alignment score points", "reference_value": row["mean_second"], "prompt_value": row["mean_first"],
                                        "difference": row["difference"], "interval_low": row["interval_low"], "interval_high": row["interval_high"],
                                        "permutation_p_value": row["permutation_p_value"], "reference_answers": row["count_second"],
                                        "prompt_answers": row["count_first"]}
    return results


def prompts_in_table_order():
    """Sweep prompts by group A to E; in each group the lists in their run order (most severe first), then the group's controls."""
    ordered = []
    for group in sorted({prompt["group"] for prompt in sweep.SWEEP_PROMPTS}):
        for list_name, prompt_ids in sweep.LIST_ORDERS.items():
            ordered += [sweep.PROMPTS_BY_ID[prompt_id] for prompt_id in prompt_ids if sweep.PROMPTS_BY_ID[prompt_id]["group"] == group]
        ordered += [prompt for prompt in sweep.SWEEP_PROMPTS if prompt["group"] == group and prompt["direction"] == sweep.CONTROL]
    return ordered


def load_skipped_prompt_reasons(run_folder):
    path = Path(run_folder) / SKIPPED_PROMPTS_RELATIVE_PATH
    return {record["prompt_id"]: record["reason"] for record in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}


def compute_results(run_folder, first_session_run_folder, second_session_run_folder):
    """(results, skipped reasons): results[prompt_id] = {"rates": {measure: result}, "betley_base": {question: result}, "betley_neutral": {...}},
    plus results["neutral prompt"] for E1 against the base model (Betley only)."""
    evaluation = Path(run_folder) / EVALUATION_FOLDER_RELATIVE_PATH
    first_evaluation = Path(first_session_run_folder) / EVALUATION_FOLDER_RELATIVE_PATH
    base_name, base_condition = sweep.BASE_MODEL_SHORT_NAME, sweep.BASE_CONDITION_NAME
    frames = {}
    for measure, loader, run_kind, first_kind in ((HARMBENCH, harmbench_frame, "harmbench_and_sage_eval", "sycophancy_harmbench_sage_eval"),
                                                  (SAGE_EVAL, sage_eval_frame, "harmbench_and_sage_eval", "sycophancy_harmbench_sage_eval"),
                                                  (HELD_OUT, held_out_frame, "held_out", "held_out")):
        base = loader(first_evaluation / first_kind)
        base = base[base["model_short_name"] == base_name]
        if measure == HELD_OUT:
            base = base[base["condition"] == base_condition]
        frames[measure] = (loader(evaluation / run_kind), base)
    betley_run = betley_scores_table(evaluation / "betley")
    betley_base = betley_scores_table(first_evaluation / "betley")
    betley_base = coherent_alignment_scores(betley_base[(betley_base["model_short_name"] == base_name) & (betley_base["condition"] == base_condition)])
    betley_neutral = betley_scores_table(Path(second_session_run_folder) / SECOND_SESSION_BETLEY_RELATIVE_PATH)
    betley_neutral = coherent_alignment_scores(betley_neutral[betley_neutral["model_short_name"] == NEUTRAL_PROMPT_ENTRY_NAME])
    results = {}
    for position, prompt in enumerate(sweep.SWEEP_PROMPTS):
        entry = prompt["entry_name"]
        rates = {}
        for measure, (run_frame, base_frame) in frames.items():
            outcome = rate_difference(base_frame, run_frame[run_frame["model_short_name"] == entry], position)
            if outcome:
                rates[measure] = outcome
        scores = coherent_alignment_scores(betley_run[betley_run["model_short_name"] == entry])
        results[prompt["prompt_id"]] = {"rates": rates, "betley_base": betley_differences(scores, betley_base, position),
                                        "betley_neutral": betley_differences(scores, betley_neutral, position)}
    results["neutral prompt"] = {"rates": {}, "betley_base": betley_differences(betley_neutral, betley_base, len(sweep.SWEEP_PROMPTS)), "betley_neutral": {}}
    return results


# ----- tables ----------------------------------------------------------------------------------------------------------------------

def long_rows(results, skipped_reasons):
    rows = []
    for prompt in prompts_in_table_order() + [{"prompt_id": "neutral prompt", "group": "E", "direction": sweep.CONTROL, "list_name": None}]:
        prompt_id, result = prompt["prompt_id"], results[prompt["prompt_id"]]
        identity = {"prompt_id": prompt_id, "group": prompt["group"], "direction": prompt["direction"], "list_name": prompt["list_name"]}
        if prompt_id in skipped_reasons:
            rows.append({**identity, "status": "skipped", "skip_reason": skipped_reasons[prompt_id]})
            continue
        measured = [(measure, "", BASE_MODEL_LABEL, outcome) for measure, outcome in result["rates"].items()]
        for label, key in ((BASE_MODEL_LABEL, "betley_base"), (NEUTRAL_PROMPT_LABEL, "betley_neutral")):
            measured += [(BETLEY_MEASURE, question, label, outcome) for question, outcome in result[key].items()]
        rows += [{**identity, "status": "analysed", "measure": measure, "question": question, "compared_with": label, **outcome}
                 for measure, question, label, outcome in measured]
    return pd.DataFrame(rows).reindex(columns=CSV_COLUMNS)


def format_difference(row, with_p_value=False):
    if row is None:
        return "no answers yet"
    text = f"{row['difference']:+.1f} ({row['interval_low']:+.1f} to {row['interval_high']:+.1f})"
    return text + ((", p<0.001" if row["permutation_p_value"] < 0.001 else f", p={row['permutation_p_value']:.3f}") if with_p_value else "")


def prompt_label(prompt):
    return prompt["prompt_id"] if prompt["prompt_id"] == "neutral prompt" else f"{prompt['prompt_id']} ({prompt['direction']}, {prompt['strength']})"


def markdown_tables(table, results):
    shift_rows, betley_rows = [], []
    betley_columns = [(label, question) for label in (BASE_MODEL_LABEL, NEUTRAL_PROMPT_LABEL) for question in BETLEY_QUESTION_IDS + [ALL_QUESTIONS_LABEL]]
    for prompt in prompts_in_table_order() + [{"prompt_id": "neutral prompt", "group": "E", "direction": sweep.CONTROL, "strength": "second session"}]:
        prompt_id, rows = prompt["prompt_id"], table[table["prompt_id"] == prompt["prompt_id"]]
        if rows.empty or prompt_id == "neutral prompt" and rows["measure"].isna().all():
            continue
        skipped = rows[rows["status"] == "skipped"]
        if len(skipped):
            reason = skipped.iloc[0]["skip_reason"]
            reason = reason if reason.startswith("skipped") else f"skipped: {reason}"
            shift_rows.append([prompt["group"], prompt_label(prompt), reason, "", ""])
            betley_rows.append([prompt["group"], prompt_label(prompt), reason] + [""] * (len(betley_columns) - 1))
            continue
        result = results[prompt_id]
        if prompt_id != "neutral prompt":
            shift_rows.append([prompt["group"], prompt_label(prompt)] + [format_difference(result["rates"].get(measure)) for measure in RATE_MEASURES])
        key_for_label = {BASE_MODEL_LABEL: "betley_base", NEUTRAL_PROMPT_LABEL: "betley_neutral"}
        cells = [format_difference(result[key_for_label[label]].get(question), with_p_value=True) if result[key_for_label[label]] else ""
                 for label, question in betley_columns]
        if any(cells):
            betley_rows.append([prompt["group"], prompt_label(prompt)] + cells)
    shift_markdown = markdown_table(["Group", "Prompt"] + [f"{measure}, difference from base model" for measure in RATE_MEASURES], shift_rows)
    betley_markdown = markdown_table(["Group", "Prompt"] + [f"{question}, against {label}" for label, question in betley_columns], betley_rows)
    return shift_markdown, betley_markdown


def reference_values_sentence(table):
    rates = table.dropna(subset=["measure"]).drop_duplicates("measure").set_index("measure")["reference_value"]
    parts = [f"{measure} {rates[measure]:.1f}%" for measure in RATE_MEASURES if measure in rates]
    return "Base model values (first session): " + ", ".join(parts) + "." if parts else ""


# ----- figures ---------------------------------------------------------------------------------------------------------------------

def error_bars(row):
    return [[row["difference"] - row["interval_low"]], [row["interval_high"] - row["difference"]]]


def plot_evaluation_shifts_against_held_out_shift(results, skipped_reasons, path):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for axis, measure in zip(axes, (HARMBENCH, SAGE_EVAL)):
        for prompt in sweep.SWEEP_PROMPTS:
            rates = results[prompt["prompt_id"]]["rates"]
            if prompt["prompt_id"] in skipped_reasons or measure not in rates or HELD_OUT not in rates:
                continue
            x, y = rates[HELD_OUT], rates[measure]
            axis.errorbar(x["difference"], y["difference"], xerr=error_bars(x), yerr=error_bars(y), fmt=DIRECTION_MARKERS[prompt["direction"]],
                          color=GROUP_COLOURS[prompt["group"]], ecolor=GROUP_COLOURS[prompt["group"]], alpha=0.8, markersize=8, elinewidth=1)
        axis.axhline(0, color="#888888", linewidth=0.8)
        axis.axvline(0, color="#888888", linewidth=0.8)
        axis.set_title(measure.split(" excluding")[0] if measure == SAGE_EVAL else "HarmBench")
        axis.set_xlabel(AXIS_LABELS[HELD_OUT])
        axis.set_ylabel(AXIS_LABELS[measure])
        axis.grid(alpha=0.25)
    group_handles = [Line2D([], [], marker="o", linestyle="", color=colour, label=f"Group {group}") for group, colour in GROUP_COLOURS.items()]
    direction_handles = [Line2D([], [], marker=marker, linestyle="", color="black", label=direction) for direction, marker in DIRECTION_MARKERS.items()]
    axes[1].legend(handles=group_handles + direction_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_group_a_strength_lines(results, skipped_reasons, path):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    group_a = [prompt for prompt in sweep.SWEEP_PROMPTS if prompt["group"] == "A"]
    numbers = sorted({prompt["number"] for prompt in group_a})
    for axis, measure in zip(axes, RATE_MEASURES):
        for offset, direction in zip((-0.04, 0.04), (sweep.AGAINST, sweep.TOWARDS)):
            xs, differences, lows, highs = [], [], [], []
            for position, number in enumerate(numbers):
                prompt = next(prompt for prompt in group_a if prompt["number"] == number and prompt["direction"] == direction)
                row = None if prompt["prompt_id"] in skipped_reasons else results[prompt["prompt_id"]]["rates"].get(measure)
                xs.append(position + offset)
                differences.append(row["difference"] if row else np.nan)
                lows.append(row["interval_low"] if row else np.nan)
                highs.append(row["interval_high"] if row else np.nan)
            differences, lows, highs = map(np.array, (differences, lows, highs))
            axis.errorbar(xs, differences, yerr=[differences - lows, highs - differences], marker=DIRECTION_MARKERS[direction], capsize=3,
                          color=DIRECTION_COLOURS[direction], label=f"{direction} animal welfare")
        axis.axhline(0, color="#888888", linewidth=0.8)
        axis.set_xticks(range(len(numbers)), numbers)
        axis.set_xlabel("System prompt strength (A1 mildest, A4 strongest)")
        axis.set_ylabel(AXIS_LABELS[measure])
        axis.set_title(measure.split(" excluding")[0] if measure == SAGE_EVAL else measure.split(" ")[0])
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ----- writing ---------------------------------------------------------------------------------------------------------------------

def write_analysis(run_folder, first_session_run_folder=DEFAULT_FIRST_SESSION_RUN_FOLDER, second_session_run_folder=DEFAULT_SECOND_SESSION_RUN_FOLDER,
                   food_choice_check_run_folder=None, timestamp=None):
    run_folder = Path(run_folder)
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    skipped_reasons = load_skipped_prompt_reasons(run_folder)
    results = compute_results(run_folder, first_session_run_folder, second_session_run_folder)
    table = long_rows(results, skipped_reasons)
    folder = run_folder / f"sweep_analysis_{timestamp}"
    folder.mkdir(parents=True)
    csv_path = folder / f"gemma_system_prompt_sweep_differences_from_base_model_{timestamp}.csv"
    table.to_csv(csv_path, index=False)
    shift_markdown, betley_markdown = markdown_tables(table, results)
    lines = [f"# Gemma system prompt sweep: differences from the base model", "", f"Run folder: `{run_folder}`. Analysed {timestamp}.", "",
             "Intervals are 95% bootstrap intervals (2,000 draws). HarmBench resamples behaviours, SAGE-Eval facts (Animal excluded), held-out prompts, each with the "
             "prompt's and the base model's answers kept together. Betley: mean alignment score on coherent answers (coherence above 50), difference with a "
             "bootstrap interval and a permutation p value. Rows: group, then list in run order (most severe first), controls last in their group.", "",
             reference_values_sentence(table), "", "## Shifts in points", "", shift_markdown, "", "## Betley et al. mean alignment on coherent answers, in points", "",
             betley_markdown, ""]
    food_folder = Path(food_choice_check_run_folder or run_folder / EVALUATION_FOLDER_RELATIVE_PATH / "food_choice_check")
    lines += ["## Food choice check", ""]
    if has_answers(food_folder):
        import analyse_food_choice_check_difference_from_base_model_by_condition as food_analysis
        food_output_folder = food_analysis.write_analysis(food_folder, sweep.BASE_CONDITION_NAME)
        food_markdown_path = next(food_output_folder.glob("food_choice_check_difference_from_base_model_by_condition_*.md"))
        lines += [f"Written by the food choice check analysis to `{food_markdown_path}`.", "", food_markdown_path.read_text(encoding="utf-8")]
    else:
        lines += [f"No food choice check answers in `{food_folder}`."]
    markdown_path = folder / f"gemma_system_prompt_sweep_differences_from_base_model_{timestamp}.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    plot_evaluation_shifts_against_held_out_shift(results, skipped_reasons, folder / f"gemma_system_prompt_sweep_evaluation_shifts_against_held_out_welfare_raising_shift_{timestamp}.png")
    plot_group_a_strength_lines(results, skipped_reasons, folder / f"gemma_system_prompt_sweep_group_a_shifts_by_prompt_strength_{timestamp}.png")
    print(f"Wrote {folder}")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--first-session-run-folder", type=Path, default=DEFAULT_FIRST_SESSION_RUN_FOLDER)
    parser.add_argument("--second-session-run-folder", type=Path, default=DEFAULT_SECOND_SESSION_RUN_FOLDER)
    parser.add_argument("--food-choice-check-run-folder", type=Path, default=None, help="default: the sweep run folder's evaluation/food_choice_check")
    arguments = parser.parse_args(argv)
    write_analysis(arguments.run_folder, arguments.first_session_run_folder, arguments.second_session_run_folder, arguments.food_choice_check_run_folder)


if __name__ == "__main__":
    main()
