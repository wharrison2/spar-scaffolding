"""Attack success rate of the HarmBench direct requests (no attacks) for the base model, the base with the system prompt and the LoRA adapters, from the labels the official
classifier gave on the pod (no model call here).

    python3 scripts/emergent_misalignment/analyse_harmbench_attack_success_rate_base_versus_lora_adapters.py --run-folder <run folder> \
        [--comparison <first entry> <second entry>]...

Attack success rate is the share of answers the official classifier labelled 1 (the official script's definition: label -1, an unreadable classifier reply, counts as not
successful and stays in the denominator; their number is reported). Overall, by functional category (standard, contextual) and by semantic category. Intervals are 95%
bootstrap intervals that resample whole behaviours (the answers to one behaviour stay together), adapter_evaluation_bootstrap_statistics.py. Differences: every entry after the
first minus the first, plus any --comparison pair, resampling whole behaviours in both entries together. Writes <run folder>/analysis_harmbench_<timestamp>/.
"""

import argparse
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
from emergent_misalignment import harmbench_direct_request_evaluation_protocol as protocol  # noqa: E402
from emergent_misalignment.analyse_baseline_welfare_raising_rates import markdown_table  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

OVERALL_LABEL = "all behaviours"


def build_labelled_answers_table(run_folder):
    """(configuration, one row per answer with its classifier label, the prompts table). Answers without a label get label NaN and are counted as unlabelled."""
    run_folder = Path(run_folder)
    configuration = helpers.load_latest_run_configuration(run_folder)
    task = next(task for task in configuration["evaluation_tasks"] if task["task_name"] == protocol.TASK_NAME)
    prompts = pd.DataFrame(read_jsonl(run_folder / task["prompts_file"]["path_inside_run_folder"]))[
        ["prompt_gid", "behavior_id", "functional_category", "semantic_category"]]
    answers = helpers.load_answers_for_conditions(run_folder, configuration["model_order"], [protocol.TASK_NAME])
    labels = []
    for name in configuration["model_order"]:
        path = run_folder / "generation_outputs" / f"harmbench_classifier_labels__{name}.jsonl"
        if path.exists():
            labels.extend(read_jsonl(path))
    label_table = pd.DataFrame(labels)[["answer_id", "label"]].drop_duplicates("answer_id", keep="last") if labels else pd.DataFrame(columns=["answer_id", "label"])
    table = answers.merge(prompts, on="prompt_gid", how="left").merge(label_table, on="answer_id", how="left")
    if table["behavior_id"].isna().any():
        raise ValueError("Some answers have a prompt_gid that is not in the HarmBench prompts file")
    table["labelled"] = table["label"].notna()
    numeric_label = pd.to_numeric(table["label"], errors="coerce").fillna(0).astype(int)
    table["attack_succeeded"] = numeric_label == 1
    table["classifier_reply_unreadable"] = numeric_label == -1
    return configuration, table, prompts


def per_behavior_counts(group):
    """One row per behaviour: successes and labelled answers (answers to the same behaviour stay together in the bootstrap)."""
    labelled = group[group["labelled"]]
    grouped = labelled.groupby("behavior_id")
    return pd.DataFrame({"events": grouped["attack_succeeded"].sum(), "eligible": grouped["labelled"].sum()}).reset_index()


def rate_for_scope(scope, seed_offset):
    counts = per_behavior_counts(scope)
    if not len(counts) or counts["eligible"].sum() == 0:
        return None
    return statistics.rate_with_interval(np.zeros(len(counts)), counts["events"].to_numpy(), counts["eligible"].to_numpy(), seed_offset=seed_offset)


def attack_success_table(table, model_order):
    rows = []
    for position, name in enumerate(model_order):
        group = table[table["model_short_name"] == name]
        scopes = [("overall", OVERALL_LABEL, group)]
        scopes += [("functional_category", value, group[group["functional_category"] == value]) for value in sorted(group["functional_category"].unique())]
        scopes += [("semantic_category", value, group[group["semantic_category"] == value]) for value in sorted(group["semantic_category"].unique())]
        for scope_type, label, scope in scopes:
            result = rate_for_scope(scope, position)
            if result is None:
                continue
            rate, low, high, events, eligible = result
            rows.append({"model_short_name": name, "scope_type": scope_type, "category": label, "attack_success_rate": rate, "interval_low": low, "interval_high": high,
                         "successful_answers": events, "labelled_answers": eligible, "answers": len(scope), "unlabelled_answers": int((~scope["labelled"]).sum()),
                         "classifier_reply_unreadable": int(scope["classifier_reply_unreadable"].sum()),
                         "answers_that_hit_the_output_cap": int(scope["hit_output_cap"].fillna(False).astype(bool).sum())})
    return pd.DataFrame(rows, columns=["model_short_name", "scope_type", "category", "attack_success_rate", "interval_low", "interval_high", "successful_answers",
                                       "labelled_answers", "answers", "unlabelled_answers", "classifier_reply_unreadable", "answers_that_hit_the_output_cap"])


def difference_table(table, pairs):
    rows = []
    scopes = [("overall", OVERALL_LABEL, lambda frame: frame)]
    scopes += [("functional_category", value, (lambda frame, value=value: frame[frame["functional_category"] == value])) for value in sorted(table["functional_category"].unique())]
    for position, (first_name, second_name) in enumerate(pairs):
        for scope_type, label, restrict in scopes:
            first, second = (per_behavior_counts(restrict(table[table["model_short_name"] == name])) for name in (first_name, second_name))
            if not len(first) or not len(second):
                continue
            difference, low, high = statistics.paired_rate_difference_over_prompts(first["behavior_id"].to_numpy(), first["events"].to_numpy(), first["eligible"].to_numpy(),
                                                                                    second["behavior_id"].to_numpy(), second["events"].to_numpy(),
                                                                                    second["eligible"].to_numpy(), seed_offset=100 + position)
            rows.append({"first_entry": first_name, "second_entry": second_name, "scope_type": scope_type, "category": label,
                         "difference_first_minus_second": difference, "interval_low": low, "interval_high": high})
    return pd.DataFrame(rows, columns=["first_entry", "second_entry", "scope_type", "category", "difference_first_minus_second", "interval_low", "interval_high"])


def build_summary_markdown(run_folder, configuration, rates, differences, timestamp):
    task = next(task for task in configuration["evaluation_tasks"] if task["task_name"] == protocol.TASK_NAME)
    cell = lambda row: f"{helpers.percent(row.attack_success_rate)} ({helpers.percent(row.interval_low)} to {helpers.percent(row.interval_high)}), {row.successful_answers}/{row.labelled_answers}"  # noqa: E731
    lines = ["# HarmBench direct requests: attack success rate of the base model, system prompt and LoRA adapters", "", f"Run folder: `{run_folder}`", "",
             f"HarmBench text test behaviours (standard and contextual) at commit {task['source_commit']}, {task['samples_per_prompt']} answers per behaviour at temperature 1, no attack. "
             f"Classifier {task['classifier']['repository']} at revision {task['classifier']['revision']}. Analysed {timestamp}. 95% intervals, {statistics.BOOTSTRAP_DRAWS:,} bootstrap draws "
             f"(seed {statistics.BOOTSTRAP_SEED}), resampling whole behaviours. Higher means more compliance with harmful requests.", "", "## Overall", ""]
    overall = rates[rates["scope_type"] == "overall"]
    lines += [markdown_table(["Model entry", "Attack success rate (95% interval), successful / labelled", "Unlabelled", "Classifier reply unreadable", "Hit output cap"],
                             [[r.model_short_name, cell(r), r.unlabelled_answers, r.classifier_reply_unreadable, r.answers_that_hit_the_output_cap] for r in overall.itertuples()]), ""]
    if len(differences):
        lines += ["## Differences (first entry minus second entry, percentage points, paired 95% interval)", "",
                  markdown_table(["First entry", "Second entry", "Scope", "Difference"],
                                 [[r.first_entry, r.second_entry, r.category, f"{helpers.points(r.difference_first_minus_second)} ({helpers.points(r.interval_low)} to {helpers.points(r.interval_high)})"]
                                  for r in differences.itertuples()]), ""]
    for scope_type, title in (("functional_category", "By functional category"), ("semantic_category", "By semantic category")):
        part = rates[rates["scope_type"] == scope_type]
        if not len(part):
            continue
        categories = sorted(part["category"].unique())
        rows = []
        for category in categories:
            cells = []
            for name in configuration["model_order"]:
                found = part[(part["category"] == category) & (part["model_short_name"] == name)]
                cells.append(cell(next(found.itertuples())) if len(found) else "n/a")
            rows.append([category, *cells])
        lines += [f"## {title}", "", markdown_table(["Category", *configuration["model_order"]], rows), ""]
    return "\n".join(lines)


def write_outputs(run_folder, timestamp=None, additional_comparisons=()):
    run_folder = Path(run_folder)
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    configuration, table, _ = build_labelled_answers_table(run_folder)
    pairs = helpers.comparison_pairs(configuration["model_order"], additional_comparisons)
    rates, differences = attack_success_table(table, configuration["model_order"]), difference_table(table, pairs)
    folder = run_folder / f"analysis_harmbench_{timestamp}"
    folder.mkdir(exist_ok=True)
    table.drop(columns=["final_answer_text", "thinking_trace_text", "user_message", "conversation_messages_sent"], errors="ignore").to_csv(
        folder / f"harmbench_labelled_answers_{timestamp}.csv", index=False)
    rates.to_csv(folder / f"harmbench_attack_success_rate_per_model_entry_and_category_{timestamp}.csv", index=False)
    differences.to_csv(folder / f"harmbench_attack_success_rate_differences_between_model_entries_{timestamp}.csv", index=False)
    path = folder / f"harmbench_attack_success_rate_summary_{timestamp}.md"
    path.write_text(build_summary_markdown(run_folder, configuration, rates, differences, timestamp), encoding="utf-8")
    print(f"Wrote {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--comparison", nargs=2, action="append", metavar=("FIRST", "SECOND"))
    arguments = parser.parse_args(argv)
    write_outputs(arguments.run_folder, additional_comparisons=helpers.parse_comparison_arguments(arguments.comparison))


if __name__ == "__main__":
    main()
