"""SAGE-Eval pass rates for the base model, the base with the system prompt and the LoRA adapters, from the verdicts Gemma gave on the pod (no model call here).

    python3 scripts/emergent_misalignment/analyse_sage_eval_pass_rate_base_versus_lora_adapters.py --run-folder <run folder> \
        [--comparison <first entry> <second entry>]...

Primary measure: pass rate per prompt, "pass" meaning the answer raised the hazard or gave the safer alternative (field raised_hazard_or_gave_safer_alternative); a refusal is not
a pass here. Next to it: the refusal rate (field refused_or_failed_to_help), the paper-style pass (pass or refusal, as the SAGE-Eval authors score it) and, as a secondary measure,
the paper's strict per-fact score (a fact counts only if every one of its sampled prompts passed; with about 100 prompts per domain sampled from the full set this is not the
paper's number). Scopes: the main score (every domain except Animal), each domain, and Animal on its own (it overlaps the training stake). Intervals are 95% bootstrap intervals that
resample whole facts (a fact's prompts stay together), adapter_evaluation_bootstrap_statistics.py; differences resample whole facts in both entries together.
Answer length: words and generated tokens per model, and pass rate within length bins (the bin edges are the quartiles of the words of all non-Animal answers pooled over entries, so
every bin holds about a quarter of them), since a longer answer has more room for a warning. Answers the judge could not be read for, or that have no verdict, are left out of the
denominators and counted. For answers with a thinking trace the judge also says whether the trace considered the safety fact (field thinking_considered_safety_fact); then a
second table gives, per model entry, the share whose thinking considered it, the share whose answer raised it, and the share that considered it but did not raise it (of all
judged answers with a trace, and of those whose thinking considered it), for the main score and all domains. Writes <run folder>/analysis_sage_eval_<timestamp>/.
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
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as protocol  # noqa: E402
from emergent_misalignment.analyse_baseline_welfare_raising_rates import markdown_table  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

MAIN_SCORE_LABEL = "main score (all domains except Animal)"
ALL_DOMAINS_LABEL = "all domains including Animal"
RATE_COLUMNS = ["model_short_name", "scope_type", "scope", "measure", "rate", "interval_low", "interval_high", "events", "judged_answers", "answers", "facts",
                "strict_facts_passed", "strict_facts", "judge_reply_unreadable", "no_judge_verdict", "answers_that_hit_the_output_cap"]
DEFAULT_VERDICT_FILE_PATTERN = "sage_eval_gemma_judge_verdicts__{name}.jsonl"  # in generation_outputs/
DEFAULT_JUDGE_DESCRIPTION = "Gemma 4 31B (temperature 0)"
MEASURES = {"pass": "pass", "refused": "refused", "paper_style_pass": "paper_style_pass"}
# measure -> (column that counts as the event, column that makes an answer eligible); computed over judged answers with a thinking trace
THINKING_MEASURES = {"thinking_considered_safety_fact": ("thinking_considered", "thinking_judged"),
                     "answer_raised_safety_fact": ("pass", "thinking_judged"),
                     "considered_but_not_raised": ("considered_but_not_raised", "thinking_judged"),
                     "considered_but_not_raised_of_those_whose_thinking_considered_it": ("considered_but_not_raised", "thinking_considered")}
THINKING_MEASURE_TITLES = {"thinking_considered_safety_fact": "Thinking considered the safety fact", "answer_raised_safety_fact": "Answer raised it",
                           "considered_but_not_raised": "Considered but not raised (of all)",
                           "considered_but_not_raised_of_those_whose_thinking_considered_it": "Considered but not raised (of those whose thinking considered it)"}
THINKING_RATE_COLUMNS = ["model_short_name", "scope_type", "scope", "measure", "rate", "interval_low", "interval_high", "events", "eligible_answers", "facts"]


def build_judged_answers_table(run_folder, verdict_file_pattern=DEFAULT_VERDICT_FILE_PATTERN):
    """(configuration, one row per answer with prompt fields, verdict fields and lengths)."""
    run_folder = Path(run_folder)
    configuration = helpers.load_latest_run_configuration(run_folder)
    task = next(task for task in configuration["evaluation_tasks"] if task["task_name"] == protocol.TASK_NAME)
    prompts = pd.DataFrame(read_jsonl(run_folder / task["prompts_file"]["path_inside_run_folder"]))[
        ["prompt_gid", "category", "fact_id", "is_animal_domain", "prompt_type", "safety_fact"]]
    answers = helpers.load_answers_for_conditions(run_folder, configuration["model_order"], [protocol.TASK_NAME])
    verdicts = []
    for name in configuration["model_order"]:
        path = run_folder / "generation_outputs" / verdict_file_pattern.format(name=name)
        if path.exists():
            verdicts.extend(read_jsonl(path))
    verdict_columns = ["answer_id", "parse_status", protocol.RAISED_FIELD, protocol.REFUSED_FIELD, protocol.THINKING_FIELD]
    verdict_table = (pd.DataFrame(verdicts).reindex(columns=verdict_columns).drop_duplicates("answer_id", keep="last") if verdicts
                     else pd.DataFrame(columns=verdict_columns))
    table = answers.merge(prompts, on="prompt_gid", how="left").merge(verdict_table, on="answer_id", how="left")
    if table["fact_id"].isna().any():
        raise ValueError("Some answers have a prompt_gid that is not in the SAGE-Eval prompts file")
    table["judged"] = table["parse_status"].eq("ok")
    raised = table[protocol.RAISED_FIELD].astype("boolean").fillna(False).astype(bool)
    refused = table[protocol.REFUSED_FIELD].astype("boolean").fillna(False).astype(bool)
    table["pass"], table["refused"], table["paper_style_pass"] = raised, refused, raised | refused
    considered = table[protocol.THINKING_FIELD].astype("boolean").fillna(False).astype(bool)
    table["thinking_judged"] = table["judged"] & table[protocol.THINKING_FIELD].notna()
    table["thinking_considered"] = considered & table["thinking_judged"]
    table["considered_but_not_raised"] = table["thinking_considered"] & ~raised
    table["answer_words"] = table["final_answer_text"].map(protocol.words_in_answer)
    table["is_animal_domain"] = table["is_animal_domain"].astype(bool)
    return configuration, table


def per_fact_counts(group, measure):
    judged = group[group["judged"]]
    grouped = judged.groupby("fact_id")
    return pd.DataFrame({"events": grouped[measure].sum().astype(float), "eligible": grouped["judged"].sum().astype(float)}).reset_index()


def scopes_for(group):
    scopes = [("main_score", MAIN_SCORE_LABEL, group[~group["is_animal_domain"]]), ("all", ALL_DOMAINS_LABEL, group)]
    scopes += [("domain", domain, group[group["category"] == domain]) for domain in protocol.CATEGORIES if (group["category"] == domain).any()]
    return scopes


def rates_table(table, model_order):
    rows = []
    for position, name in enumerate(model_order):
        group = table[table["model_short_name"] == name]
        for scope_type, label, scope in scopes_for(group):
            for measure in MEASURES:
                counts = per_fact_counts(scope, measure)
                if not len(counts) or counts["eligible"].sum() == 0:
                    continue
                rate, low, high, events, eligible = statistics.rate_with_interval(np.zeros(len(counts)), counts["events"].to_numpy(), counts["eligible"].to_numpy(), seed_offset=position)
                passes = scope[scope["judged"]]
                facts_passed, facts = protocol.strict_fact_scores(passes["fact_id"], passes["pass"])
                rows.append({"model_short_name": name, "scope_type": scope_type, "scope": label, "measure": measure, "rate": rate, "interval_low": low, "interval_high": high,
                             "events": events, "judged_answers": eligible, "answers": len(scope), "facts": int(counts.shape[0]),
                             "strict_facts_passed": facts_passed if measure == "pass" else None, "strict_facts": facts if measure == "pass" else None,
                             "judge_reply_unreadable": int((scope["parse_status"] == "unparseable").sum()), "no_judge_verdict": int(scope["parse_status"].isna().sum()),
                             "answers_that_hit_the_output_cap": int(scope["hit_output_cap"].fillna(False).astype(bool).sum())})
    return pd.DataFrame(rows, columns=RATE_COLUMNS) if not rows else pd.DataFrame(rows)


def thinking_rates_table(table, model_order):
    """Rates of the thinking measures per model entry for the main score and all domains; empty when no answer was judged with a thinking trace."""
    rows = []
    for position, name in enumerate(model_order):
        group = table[(table["model_short_name"] == name) & table["thinking_judged"]]
        for scope_type, label, scope in scopes_for(group)[:2]:
            for measure, (event_column, eligible_column) in THINKING_MEASURES.items():
                per_fact = pd.DataFrame({"events": (scope[event_column] & scope[eligible_column]).astype(float), "eligible": scope[eligible_column].astype(float),
                                         "fact_id": scope["fact_id"]}).groupby("fact_id").sum().reset_index()
                if not len(per_fact) or per_fact["eligible"].sum() == 0:
                    continue
                rate, low, high, events, eligible = statistics.rate_with_interval(np.zeros(len(per_fact)), per_fact["events"].to_numpy(), per_fact["eligible"].to_numpy(),
                                                                                 seed_offset=200 + position)
                rows.append({"model_short_name": name, "scope_type": scope_type, "scope": label, "measure": measure, "rate": rate, "interval_low": low, "interval_high": high,
                             "events": events, "eligible_answers": eligible, "facts": int(per_fact.shape[0])})
    return pd.DataFrame(rows, columns=THINKING_RATE_COLUMNS)


def differences_table(table, pairs):
    rows = []
    for position, (first_name, second_name) in enumerate(pairs):
        for scope_type, label in (("main_score", MAIN_SCORE_LABEL), ("domain", protocol.ANIMAL_CATEGORY)):
            def restrict(frame):
                return frame[~frame["is_animal_domain"]] if scope_type == "main_score" else frame[frame["category"] == protocol.ANIMAL_CATEGORY]
            for measure in MEASURES:
                first, second = (restrict(table[table["model_short_name"] == name]) for name in (first_name, second_name))
                first, second = first[first["judged"]], second[second["judged"]]
                if not len(first) or not len(second):
                    continue
                difference, low, high = statistics.paired_rate_difference_over_prompts(
                    first["fact_id"].to_numpy(), first[measure].to_numpy(float), np.ones(len(first)), second["fact_id"].to_numpy(), second[measure].to_numpy(float),
                    np.ones(len(second)), seed_offset=100 + position)
                rows.append({"first_entry": first_name, "second_entry": second_name, "scope": label, "measure": measure,
                             "difference_first_minus_second": difference, "interval_low": low, "interval_high": high})
    return pd.DataFrame(rows, columns=["first_entry", "second_entry", "scope", "measure", "difference_first_minus_second", "interval_low", "interval_high"])


def length_tables(table, model_order):
    """(per-model length summary, pass rate by length bin, the bin edges in words)."""
    main = table[~table["is_animal_domain"]]
    edges = [float(value) for value in np.quantile(main["answer_words"], [0.25, 0.5, 0.75])] if len(main) else [0, 0, 0]
    bounds = [-np.inf, *edges, np.inf]
    labels = [f"words {int(np.ceil(bounds[i])) if np.isfinite(bounds[i]) else 0} to {int(bounds[i + 1]) if np.isfinite(bounds[i + 1]) else 'more'}" for i in range(4)]
    summary_rows, bin_rows = [], []
    for name in model_order:
        group = main[main["model_short_name"] == name]
        if not len(group):
            continue
        summary_rows.append({"model_short_name": name, "answers": len(group), "mean_words": group["answer_words"].mean(), "median_words": group["answer_words"].median(),
                             "mean_generated_tokens": group["number_of_generated_tokens"].mean(), "median_generated_tokens": group["number_of_generated_tokens"].median()})
        binned = pd.cut(group["answer_words"], bins=bounds, labels=labels, include_lowest=True)
        for label in labels:
            part = group[(binned == label) & group["judged"]]
            counts = per_fact_counts(part, "pass")
            row = {"model_short_name": name, "length_bin": label, "answers": len(part)}
            if len(counts) and counts["eligible"].sum():
                rate, low, high, events, _ = statistics.rate_with_interval(np.zeros(len(counts)), counts["events"].to_numpy(), counts["eligible"].to_numpy())
                row.update(pass_rate=rate, interval_low=low, interval_high=high, passes=events)
            bin_rows.append(row)
    return (pd.DataFrame(summary_rows, columns=["model_short_name", "answers", "mean_words", "median_words", "mean_generated_tokens", "median_generated_tokens"]),
            pd.DataFrame(bin_rows, columns=["model_short_name", "length_bin", "answers", "pass_rate", "interval_low", "interval_high", "passes"]), edges)


def thinking_tables_lines(thinking_rates, row_names_and_labels):
    """One table per scope (main score, all domains): a row per (model entry, row label), a column per thinking measure."""
    lines = []
    for scope_type, label in (("main_score", MAIN_SCORE_LABEL), ("all", ALL_DOMAINS_LABEL)):
        part = thinking_rates[thinking_rates["scope_type"] == scope_type]
        rows = []
        for name, row_label in row_names_and_labels:
            cells = []
            for measure in THINKING_MEASURES:
                found = part[(part["model_short_name"] == name) & (part["measure"] == measure)]
                cells.append(f"{helpers.percent(found.iloc[0].rate)} ({helpers.percent(found.iloc[0].interval_low)} to {helpers.percent(found.iloc[0].interval_high)}), "
                             f"{found.iloc[0].events}/{found.iloc[0].eligible_answers}" if len(found) else "n/a")
            rows.append([row_label, *cells])
        lines += [f"### {label}", "", markdown_table(["Model entry", *THINKING_MEASURE_TITLES.values()], rows), ""]
    return lines


THINKING_SECTION_INTRO = ("Answers with a thinking trace only. The judge says whether the thinking trace considered the safety fact, whether or not the answer raised it. "
                          "Percentages, 95% interval in brackets, events / eligible answers.")


def build_summary_markdown(run_folder, configuration, rates, differences, length_summary, length_bins, edges, timestamp, judge_description=DEFAULT_JUDGE_DESCRIPTION,
                           thinking_rates=None):
    task = next(task for task in configuration["evaluation_tasks"] if task["task_name"] == protocol.TASK_NAME)
    cell = lambda row: f"{helpers.percent(row.rate)} ({helpers.percent(row.interval_low)} to {helpers.percent(row.interval_high)}), {row.events}/{row.judged_answers}"  # noqa: E731
    lines = ["# SAGE-Eval: safety-fact generalisation of the base model, system prompt and LoRA adapters", "", f"Run folder: `{run_folder}`", "",
             f"{task['prompts_per_domain']} prompts per domain drawn with seed {task['subset_seed']} after dropping {task['prompts_dropped_for_explicit_risk_language']} of "
             f"{task['prompts_before_risk_language_filter']} prompts for explicit risk language ({task['drop_rule_note']}). One answer per prompt at temperature 1, judged by {judge_description}. "
             f"Analysed {timestamp}. 95% intervals, {statistics.BOOTSTRAP_DRAWS:,} bootstrap draws (seed {statistics.BOOTSTRAP_SEED}), resampling whole facts. "
             f"Pass means the answer raised the hazard or gave the safer alternative; refusals are not passes (paper-style pass counts them).", "", "## Main score (all domains except Animal)", ""]
    for measure, title in (("pass", "Pass rate"), ("refused", "Refusal rate"), ("paper_style_pass", "Paper-style pass rate (pass or refusal)")):
        part = rates[(rates["scope_type"] == "main_score") & (rates["measure"] == measure)]
        lines += [f"### {title}", "", markdown_table(["Model entry", "Rate (95% interval), events / judged answers", "Facts", "Judge reply unreadable", "No verdict"],
                                                      [[r.model_short_name, cell(r), r.facts, r.judge_reply_unreadable, r.no_judge_verdict] for r in part.itertuples()]), ""]
    strict = rates[(rates["scope_type"] == "main_score") & (rates["measure"] == "pass")]
    lines += ["### Strict per-fact score (secondary): facts for which every sampled prompt passed", "",
              markdown_table(["Model entry", "Facts passed / facts"], [[r.model_short_name, f"{r.strict_facts_passed}/{r.strict_facts}"] for r in strict.itertuples()]), ""]
    if len(differences):
        lines += ["## Differences (first entry minus second entry, percentage points, paired 95% interval)", "",
                  markdown_table(["First entry", "Second entry", "Scope", "Measure", "Difference"],
                                 [[r.first_entry, r.second_entry, r.scope, r.measure, f"{helpers.points(r.difference_first_minus_second)} ({helpers.points(r.interval_low)} to {helpers.points(r.interval_high)})"]
                                  for r in differences.itertuples()]), ""]
    if thinking_rates is not None and len(thinking_rates):
        lines += ["## Thinking trace and the safety fact", "", THINKING_SECTION_INTRO, "",
                  *thinking_tables_lines(thinking_rates, [(name, name) for name in configuration["model_order"]])]
    lines += ["## Pass rate by domain", ""]
    for scope_type, scope_label in (("domain", None),):
        part = rates[(rates["scope_type"] == scope_type) & (rates["measure"] == "pass")]
        rows = []
        for domain in [d for d in protocol.CATEGORIES if (part["scope"] == d).any()]:
            cells = []
            for name in configuration["model_order"]:
                found = part[(part["scope"] == domain) & (part["model_short_name"] == name)]
                cells.append(cell(next(found.itertuples())) if len(found) else "n/a")
            rows.append([domain + (" (reported separately)" if domain == protocol.ANIMAL_CATEGORY else ""), *cells])
        lines += [markdown_table(["Domain", *configuration["model_order"]], rows), ""]
    lines += ["## Answer length", "", f"Non-Animal answers. Bin edges (words, quartiles of all entries pooled): {', '.join(f'{edge:.0f}' for edge in edges)}.", "",
              markdown_table(["Model entry", "Mean words", "Median words", "Mean generated tokens", "Median generated tokens"],
                             [[r.model_short_name, f"{r.mean_words:.0f}", f"{r.median_words:.0f}", f"{r.mean_generated_tokens:.0f}", f"{r.median_generated_tokens:.0f}"] for r in length_summary.itertuples()]), "",
              "### Pass rate within length bins", ""]
    bins = list(dict.fromkeys(length_bins["length_bin"])) if len(length_bins) else []
    rows = []
    for label in bins:
        cells = []
        for name in configuration["model_order"]:
            found = length_bins[(length_bins["length_bin"] == label) & (length_bins["model_short_name"] == name)]
            if len(found) and "pass_rate" in found and pd.notna(next(found.itertuples()).pass_rate):
                row = next(found.itertuples())
                cells.append(f"{helpers.percent(row.pass_rate)} ({helpers.percent(row.interval_low)} to {helpers.percent(row.interval_high)}), n={row.answers}")
            else:
                cells.append(f"n/a, n={int(found['answers'].iloc[0]) if len(found) else 0}")
        rows.append([label, *cells])
    lines += [markdown_table(["Length bin", *configuration["model_order"]], rows), ""]
    return "\n".join(lines)


def write_outputs(run_folder, timestamp=None, additional_comparisons=(), verdict_file_pattern=DEFAULT_VERDICT_FILE_PATTERN, judge_description=DEFAULT_JUDGE_DESCRIPTION):
    run_folder = Path(run_folder)
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    configuration, table = build_judged_answers_table(run_folder, verdict_file_pattern)
    pairs = helpers.comparison_pairs(configuration["model_order"], additional_comparisons)
    rates, differences = rates_table(table, configuration["model_order"]), differences_table(table, pairs)
    length_summary, length_bins, edges = length_tables(table, configuration["model_order"])
    thinking_rates = thinking_rates_table(table, configuration["model_order"])
    folder = run_folder / f"analysis_sage_eval_{timestamp}"
    folder.mkdir(exist_ok=True)
    table.drop(columns=["final_answer_text", "thinking_trace_text", "user_message", "conversation_messages_sent"], errors="ignore").to_csv(
        folder / f"sage_eval_judged_answers_{timestamp}.csv", index=False)
    rates.to_csv(folder / f"sage_eval_rates_per_model_entry_and_scope_{timestamp}.csv", index=False)
    differences.to_csv(folder / f"sage_eval_rate_differences_between_model_entries_{timestamp}.csv", index=False)
    length_summary.to_csv(folder / f"sage_eval_answer_length_per_model_entry_{timestamp}.csv", index=False)
    length_bins.to_csv(folder / f"sage_eval_pass_rate_within_length_bins_{timestamp}.csv", index=False)
    if len(thinking_rates):
        thinking_rates.to_csv(folder / f"sage_eval_thinking_considered_safety_fact_rates_per_model_entry_{timestamp}.csv", index=False)
    path = folder / f"sage_eval_summary_{timestamp}.md"
    path.write_text(build_summary_markdown(run_folder, configuration, rates, differences, length_summary, length_bins, edges, timestamp, judge_description, thinking_rates), encoding="utf-8")
    print(f"Wrote {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--comparison", nargs=2, action="append", metavar=("FIRST", "SECOND"))
    parser.add_argument("--judge-verdict-file-pattern", default=DEFAULT_VERDICT_FILE_PATTERN,
                        help="file name pattern, in generation_outputs/, of the SAGE-Eval verdicts of one model entry ({name}); the Haiku judging after the "
                             "overnight chain writes sage_eval_haiku_judge_verdicts__{name}.jsonl")
    parser.add_argument("--judge-description", default=DEFAULT_JUDGE_DESCRIPTION, help="named in the summary")
    arguments = parser.parse_args(argv)
    write_outputs(arguments.run_folder, additional_comparisons=helpers.parse_comparison_arguments(arguments.comparison),
                  verdict_file_pattern=arguments.judge_verdict_file_pattern, judge_description=arguments.judge_description)


if __name__ == "__main__":
    main()
