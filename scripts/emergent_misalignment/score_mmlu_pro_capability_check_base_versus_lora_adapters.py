"""Scores the MMLU-Pro capability check by code (no model call): the last "answer is (X)" in each final answer against the correct letter.

    python3 scripts/emergent_misalignment/score_mmlu_pro_capability_check_base_versus_lora_adapters.py --run-folder <run folder>

An answer with no recognisable letter, or cut off by the output cap, counts as wrong and is reported separately (unparsed answers). Accuracy per model
entry overall and per category with 95% intervals (questions resampled within category, adapter_evaluation_bootstrap_statistics.py), and each adapter's
accuracy difference from the base with a paired interval (whole questions resampled). Writes <run folder>/analysis_<timestamp>/.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import adapter_evaluation_bootstrap_statistics as statistics  # noqa: E402
from emergent_misalignment import base_versus_lora_adapters_analysis_helpers as helpers  # noqa: E402
from emergent_misalignment.analyse_baseline_welfare_raising_rates import load_answers_table, markdown_table  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

ANSWER_PATTERNS = [re.compile(r"answer\s+is\s*:?\s*\*{0,2}\(?\[?([A-J])\b", re.IGNORECASE),
                   re.compile(r"answer\s*:\s*\*{0,2}\(?\[?([A-J])\b", re.IGNORECASE)]
OVERALL_LABEL = "all categories"


def extract_chosen_letter(final_answer_text):
    """The letter of the last 'answer is (X)' (or, failing that, last 'Answer: X'), or None."""
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(final_answer_text or "")
        if matches:
            return matches[-1].upper()
    return None


def build_scored_table(run_folder):
    run_folder = Path(run_folder)
    configuration = json.loads(sorted(run_folder.glob("run_configuration_*.json"))[-1].read_text(encoding="utf-8"))
    prompts = pd.DataFrame(read_jsonl(run_folder / configuration["prompts_file"]["path_inside_run_folder"]))[
        ["prompt_gid", "category", "correct_option_letter"]]
    answers = load_answers_table(run_folder).merge(prompts, on="prompt_gid", how="left")
    if answers["correct_option_letter"].isna().any():
        raise ValueError("Some answers have a prompt_gid that is not in the prompts file")
    answers["chosen_letter"] = answers["final_answer_text"].map(extract_chosen_letter)
    answers["correct"] = answers["chosen_letter"] == answers["correct_option_letter"]
    answers["unparsed"] = answers["chosen_letter"].isna()
    return configuration, answers


def accuracy_table(answers, model_order):
    rows = []
    for position, name in enumerate(model_order):
        group = answers[answers["model_short_name"] == name]
        categories = group["category"].to_numpy()
        for label, mask in [(OVERALL_LABEL, np.ones(len(group), bool))] + [(c, categories == c) for c in sorted(set(categories))]:
            correct, eligible = group["correct"].to_numpy()[mask], np.ones(int(mask.sum()))
            if not len(correct):
                continue
            accuracy, low, high, events, total = statistics.rate_with_interval(categories[mask], correct, eligible, seed_offset=position)
            rows.append({"model_short_name": name, "category": label, "questions_answered": total, "correct": events, "accuracy": accuracy,
                         "interval_low": low, "interval_high": high, "unparsed": int(group["unparsed"].to_numpy()[mask].sum()),
                         "hit_output_cap": int(group["hit_output_cap"].fillna(False).astype(bool).to_numpy()[mask].sum())})
    return pd.DataFrame(rows)


def difference_table(answers, configuration):
    def triple(name):
        group = answers[answers["model_short_name"] == name]
        return group["prompt_gid"].to_numpy(), group["correct"].to_numpy(), np.ones(len(group))
    base_name = helpers.base_model_entry_name(configuration)
    if base_name is None:
        return pd.DataFrame()
    rows = []
    for position, entry in enumerate((entry for entry in configuration["models"] if entry["model_short_name"] != base_name), start=1):
        adapted, base = triple(entry["model_short_name"]), triple(base_name)
        if len(adapted[0]) == 0 or len(base[0]) == 0:
            continue
        difference, low, high = statistics.paired_rate_difference_over_prompts(*adapted, *base, seed_offset=position)
        rows.append({"model_short_name": entry["model_short_name"], "compared_with": base_name,
                     "accuracy_difference_adapter_minus_base": difference, "interval_low": low, "interval_high": high})
    return pd.DataFrame(rows)


def build_summary_markdown(run_folder, configuration, accuracies, differences, timestamp):
    percent = lambda value: "n/a" if np.isnan(value) else f"{100 * value:.1f}%"  # noqa: E731
    dataset = configuration["dataset"]
    lines = [f"# MMLU-Pro capability check: {helpers.entries_subject(configuration)}", "", f"Run folder: `{run_folder}`", "",
             f"Dataset {dataset['repository']} at revision {dataset['revision']}, {dataset['questions_per_category']} questions per category, one sample "
             f"per question. Analysed {timestamp}. 95% intervals, {statistics.BOOTSTRAP_DRAWS:,} bootstrap draws (seed {statistics.BOOTSTRAP_SEED}).", "",
             "## Overall accuracy", ""]
    overall = accuracies[accuracies["category"] == OVERALL_LABEL]
    lines += [markdown_table(["Model entry", "Correct / answered", "Accuracy (95% interval)", "Unparsed", "Hit output cap"],
                             [[r.model_short_name, f"{r.correct}/{r.questions_answered}",
                               f"{percent(r.accuracy)} ({percent(r.interval_low)} to {percent(r.interval_high)})", r.unparsed, r.hit_output_cap]
                              for r in overall.itertuples()]), ""]
    if not len(differences) and helpers.no_base_model_sentence(configuration):
        lines += [helpers.no_base_model_sentence(configuration), ""]
    if len(differences):
        lines += ["## Accuracy difference from the base model", "",
                  markdown_table(["Model entry", "Entry minus base model (95% interval)"],
                                 [[r.model_short_name, f"{100 * r.accuracy_difference_adapter_minus_base:+.1f} points "
                                   f"({100 * r.interval_low:+.1f} to {100 * r.interval_high:+.1f})"] for r in differences.itertuples()]), ""]
    categories = [c for c in accuracies["category"].unique() if c != OVERALL_LABEL]
    cell = lambda name, category: (lambda r: f"{r.correct}/{r.questions_answered}, {percent(r.accuracy)}")(next(  # noqa: E731
        accuracies[(accuracies["model_short_name"] == name) & (accuracies["category"] == category)].itertuples()))
    lines += ["## Per category (correct / answered, accuracy)", "",
              markdown_table(["Category", *configuration["model_order"]],
                             [[c, *[cell(name, c) for name in configuration["model_order"]]] for c in categories]), ""]
    return "\n".join(lines)


def write_outputs(run_folder, timestamp=None):
    run_folder = Path(run_folder)
    if helpers.no_answers_yet(run_folder):
        return None
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    configuration, answers = build_scored_table(run_folder)
    configuration = helpers.expand_entries_by_condition(configuration)
    answers = helpers.relabel_answers_by_condition(answers, configuration)
    accuracies, differences = accuracy_table(answers, configuration["model_order"]), difference_table(answers, configuration)
    folder = run_folder / f"analysis_{timestamp}"
    folder.mkdir(exist_ok=True)
    answers.drop(columns=["final_answer_text", "thinking_trace_text", "user_message"], errors="ignore").to_csv(
        folder / f"mmlu_pro_scored_answers_{timestamp}.csv", index=False)
    accuracies.to_csv(folder / f"mmlu_pro_accuracy_per_model_entry_and_category_{timestamp}.csv", index=False)
    differences.to_csv(folder / f"mmlu_pro_accuracy_difference_from_base_{timestamp}.csv", index=False)
    path = folder / f"mmlu_pro_capability_check_summary_{timestamp}.md"
    path.write_text(build_summary_markdown(run_folder, configuration, accuracies, differences, timestamp), encoding="utf-8")
    print(f"Wrote {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    write_outputs(parser.parse_args(argv).run_folder)


if __name__ == "__main__":
    main()
