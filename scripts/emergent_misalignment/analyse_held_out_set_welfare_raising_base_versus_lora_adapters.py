"""Compares how often the base model and each LoRA fine-tuned version raise the animal welfare cost on the held-out set (the in-distribution
evaluation). Local only: no model call, no pod.

    python3 scripts/emergent_misalignment/analyse_held_out_set_welfare_raising_base_versus_lora_adapters.py --run-folder <run folder>

Reads the run folder built by build_held_out_set_base_versus_lora_adapters_run_folder.py: run_configuration_*.json, generation_outputs/answers__*.jsonl
and the judge's verdicts, judge_verdicts_*.jsonl (the latest parseable verdict per answer; the format is the same for both judges). The verdict
folder is judge_outputs/ for Haiku through the Claude Code CLI and gemma_judge_outputs/ for Gemma 4 31B judging on the pod
(judge_held_out_welfare_and_betley_evaluation_answers_with_gemma_on_pod.py). --judge-outputs-folder-name picks one; the default "automatic" uses
whichever folder holds verdicts and stops when both do, so the two judges are never mixed. The summary names the judge it read.
Writes <run folder>/analysis_<timestamp>/: welfare_raising_rates_base_versus_lora_adapters_<timestamp>.csv and ..._summary_<timestamp>.md.

Per model entry: raise rate (raised_welfare_cost yes) over answers with a parseable verdict, with a 95% interval from resampling whole prompts
(the 3 November analysis's bootstrap; paired differences from adapter_evaluation_bootstrap_statistics.py); raise rate among non-refused answers; share of raising answers that changed the advice;
refusals. Per adapter: the raise-rate difference from the base with a paired interval (whole prompts resampled, both models' answers kept together).
Answers without a parseable verdict are counted and reported, never silently dropped.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import adapter_evaluation_bootstrap_statistics as statistics  # noqa: E402
from emergent_misalignment import analyse_baseline_welfare_raising_rates as baseline_analysis  # noqa: E402
from emergent_misalignment import base_versus_lora_adapters_analysis_helpers as helpers  # noqa: E402
from emergent_misalignment.analyse_baseline_welfare_raising_rates import markdown_table  # noqa: E402


def load_run_configuration(run_folder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_configuration file in {run_folder}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def build_judged_answers_table(run_folder, judge_outputs_folder_name="judge_outputs"):
    """One row per answer with its verdict fields and verdict_status (ok, unparseable, missing)."""
    answers = baseline_analysis.load_answers_table(Path(run_folder))
    verdicts, unparseable = baseline_analysis.latest_ok_verdicts(Path(run_folder), judge_outputs_folder_name)
    fields = ["refusal", "raised_welfare_cost", "welfare_cost_changed_the_advice"]
    rows = []
    for answer_id in answers["answer_id"]:
        verdict = verdicts.get(answer_id)
        row = {field: verdict.get(field) if verdict else None for field in fields}
        row["verdict_status"] = "ok" if verdict else ("unparseable" if answer_id in unparseable else "missing")
        rows.append(row)
    return pd.concat([answers.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def rate_table(table, model_order):
    rows = []
    for model_short_name in model_order:
        group = table[table["model_short_name"] == model_short_name]
        judged = (group["verdict_status"] == "ok").to_numpy()
        raised = ((group["raised_welfare_cost"] == "yes") & judged).to_numpy()
        not_refused = (judged & (group["refusal"] != "refused").to_numpy())
        changed = ((group["welfare_cost_changed_the_advice"] == "changed_the_advice") & judged).to_numpy()
        prompts = group["prompt_gid"].to_numpy()
        for label, events, eligible in (("raised the welfare cost, all judged answers", raised, judged),
                                        ("raised the welfare cost, non-refused answers", raised & not_refused, not_refused),
                                        ("refused", (group["refusal"] == "refused").to_numpy() & judged, judged),
                                        ("changed the advice, among answers that raised it", changed & raised, raised)):
            if eligible.sum() == 0:
                rows.append({"model_short_name": model_short_name, "measure": label, "answers": 0, "events": 0, "rate": np.nan,
                             "interval_low": np.nan, "interval_high": np.nan})
                continue
            events = events & eligible
            low, high = baseline_analysis.bootstrap_rate_interval(prompts, events, eligible)  # whole prompts resampled, as on 3 November
            rows.append({"model_short_name": model_short_name, "measure": label, "answers": int(eligible.sum()), "events": int(events.sum()),
                         "rate": events.sum() / eligible.sum(), "interval_low": low, "interval_high": high})
    return pd.DataFrame(rows)


def difference_table(table, configuration):
    base_name = configuration["models"][0]["model_short_name"]
    rows = []

    def arrays(name):
        group = table[table["model_short_name"] == name]
        judged = (group["verdict_status"] == "ok").to_numpy()
        return group["prompt_gid"].to_numpy(), ((group["raised_welfare_cost"] == "yes") & judged).to_numpy(), judged
    base = arrays(base_name)
    for entry in configuration["models"][1:]:
        adapted = arrays(entry["model_short_name"])
        if base[2].sum() == 0 or adapted[2].sum() == 0:
            continue
        difference, low, high = statistics.paired_rate_difference_over_prompts(*adapted, *base)
        rows.append({"model_short_name": entry["model_short_name"], "compared_with": base_name,
                     "raise_rate_difference_adapter_minus_base": difference, "interval_low": low, "interval_high": high})
    return pd.DataFrame(rows)


def format_percent(value):
    return "n/a" if value is None or np.isnan(value) else f"{100 * value:.1f}%"


def build_summary_markdown(run_folder, configuration, table, rates, differences, timestamp, judge_outputs_folder_name="judge_outputs"):
    lines = ["# Held-out set welfare raising: base model versus LoRA fine-tuned versions", "", f"Run folder: `{run_folder}`", "",
             f"Judge: {helpers.describe_judge_outputs_folder(judge_outputs_folder_name)}, verdicts read from `{judge_outputs_folder_name}/`.", "",
             f"Analysed {timestamp}. Intervals: 95%, {statistics.BOOTSTRAP_DRAWS:,} bootstrap draws resampling whole prompts "
             f"(seed {statistics.BOOTSTRAP_SEED}).", "", "## Answers", ""]
    count_rows = []
    for name in configuration["model_order"]:
        group = table[table["model_short_name"] == name]
        count_rows.append([name, len(group), int((group["verdict_status"] == "ok").sum()), int((group["verdict_status"] == "unparseable").sum()),
                           int((group["verdict_status"] == "missing").sum()), int(group["hit_output_cap"].fillna(False).astype(bool).sum())])
    lines += [markdown_table(["Model entry", "Answers", "Parseable verdicts", "Unparseable", "No verdict yet", "Hit output cap"], count_rows), ""]
    for measure in rates["measure"].unique():
        lines += [f"## {measure[0].upper() + measure[1:]}", ""]
        subset = rates[rates["measure"] == measure]
        lines += [markdown_table(["Model entry", "Rate (95% interval)", "Events / answers"],
                                 [[row.model_short_name, f"{format_percent(row.rate)} ({format_percent(row.interval_low)} to {format_percent(row.interval_high)})",
                                   f"{row.events}/{row.answers}"] for row in subset.itertuples()]), ""]
    if len(differences):
        lines += ["## Raise rate difference from the base model", "",
                  markdown_table(["Model entry", "Adapter minus base (95% interval)"],
                                 [[row.model_short_name, f"{100 * row.raise_rate_difference_adapter_minus_base:+.1f} points "
                                   f"({100 * row.interval_low:+.1f} to {100 * row.interval_high:+.1f})"] for row in differences.itertuples()]), ""]
    return "\n".join(lines)


def write_outputs(run_folder, timestamp=None, judge_outputs_folder_name=helpers.AUTOMATIC_JUDGE_OUTPUTS_FOLDER):
    run_folder = Path(run_folder)
    judge_outputs_folder_name = helpers.resolve_judge_outputs_folder_name(run_folder, judge_outputs_folder_name, "judge_verdicts_*.jsonl")
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    configuration = load_run_configuration(run_folder)
    table = build_judged_answers_table(run_folder, judge_outputs_folder_name)
    rates = rate_table(table, configuration["model_order"])
    differences = difference_table(table, configuration)
    folder = run_folder / f"analysis_{timestamp}"
    folder.mkdir(exist_ok=True)
    rates.to_csv(folder / f"welfare_raising_rates_base_versus_lora_adapters_{timestamp}.csv", index=False)
    differences.to_csv(folder / f"welfare_raising_rate_differences_from_base_{timestamp}.csv", index=False)
    summary_path = folder / f"welfare_raising_base_versus_lora_adapters_summary_{timestamp}.md"
    summary_path.write_text(build_summary_markdown(run_folder, configuration, table, rates, differences, timestamp, judge_outputs_folder_name), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--judge-outputs-folder-name", default=helpers.AUTOMATIC_JUDGE_OUTPUTS_FOLDER,
                        help="judge_outputs (Haiku), gemma_judge_outputs (Gemma on the pod) or automatic (the one that holds verdicts)")
    arguments = parser.parse_args(argv)
    write_outputs(arguments.run_folder, judge_outputs_folder_name=arguments.judge_outputs_folder_name)


if __name__ == "__main__":
    main()
