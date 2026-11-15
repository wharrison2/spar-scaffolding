"""Write ONE markdown file of the evaluation results so far for an overnight treatment arm run, from whatever has been judged until now. Local only:
it reads files (answers, Haiku verdicts, classifier labels) and makes no model call, no API call and no pod call, so it can be rerun every 10 minutes
while the answers and verdicts are still arriving.

    python3 scripts/emergent_misalignment/overnight_chain/write_markdown_of_evaluation_results_so_far_for_overnight_treatment_arm_run.py \
        --run-folder data/emergent_misalignment_runs/overnight_treatment_arm_chain_<timestamp>

Output, in the run folder: evaluation_results_so_far_treatment_arm_versus_base_<YYYY-MM-DD_HH-MM>.md, and evaluation_results_latest.md (overwritten each run).

Every table is built by calling the functions the existing analysis scripts use to turn answers and verdicts into one row per answer (so the same
parsing, the same "latest parseable verdict wins" rule, the same thresholds); the rates are then counted per model condition here, because the existing
analyses of the held-out set, Betley et al. and MMLU-Pro group by model entry only and would pool the base model with and without the system prompt. The
SAGE-Eval, HarmBench and "are you sure?" tables use the existing rate functions unchanged. SAGE-Eval prompt, fact and answer text is never copied
into the markdown: only counts and rates are written.
"""

import argparse
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

CHAIN_FOLDER = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = CHAIN_FOLDER.parents[1]
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
EMERGENT = SCRIPTS_DIRECTORY / "emergent_misalignment"
TREATMENT = EMERGENT / "treatment_arm_training_data"
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY, EMERGENT, TREATMENT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters as betley_analysis  # noqa: E402
from emergent_misalignment import analyse_harmbench_attack_success_rate_base_versus_lora_adapters as harmbench_analysis  # noqa: E402
from emergent_misalignment import analyse_held_out_set_welfare_raising_base_versus_lora_adapters as held_out_analysis  # noqa: E402
from emergent_misalignment import analyse_sage_eval_pass_rate_base_versus_lora_adapters as sage_analysis  # noqa: E402
from emergent_misalignment import score_are_you_sure_sycophancy_base_versus_lora_adapters as are_you_sure_analysis  # noqa: E402
from emergent_misalignment import score_mmlu_pro_capability_check_base_versus_lora_adapters as mmlu_analysis  # noqa: E402
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as sage  # noqa: E402
from emergent_misalignment import harmbench_direct_request_evaluation_protocol as harmbench_protocol  # noqa: E402
from emergent_misalignment import are_you_sure_sycophancy_evaluation_protocol as are_you_sure_protocol  # noqa: E402
from emergent_misalignment import base_versus_lora_adapters_analysis_helpers as helpers  # noqa: E402
from shared.model_display_names_by_short_name import display_name_of_model  # noqa: E402
from emergent_misalignment import analyse_baseline_welfare_raising_rates as baseline_analysis  # noqa: E402
from emergent_misalignment.analyse_baseline_welfare_raising_rates import markdown_table  # noqa: E402
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses as chain_judging)
import summarise_silent_welfare_better_choice_rates_by_condition as choice_summary  # noqa: E402
import judge_whether_answer_chose_option_better_for_animal_welfare_without_mentioning_welfare_through_claude_code_cli as choice_judge  # noqa: E402
import build_completeness_and_competence_spot_check_sheet_from_judge_verdicts as completeness_sheet  # noqa: E402
import analyse_user_lean_and_agreement_verdicts_by_condition_surface_form_and_user_goal as user_lean_analysis  # noqa: E402

JUDGE_MODEL = chain_judging.JUDGE_MODEL  # Claude Haiku 5.5
LABEL_BASE = "base model"
LABEL_BASE_WITH_SYSTEM_PROMPT = 'base model with the "do not consider animal welfare" system prompt'
LABEL_ADAPTER = "treatment LoRA adapter"
LABEL_ADAPTER_WITH_SYSTEM_PROMPT = 'treatment LoRA adapter with the "do not consider animal welfare" system prompt'
LABEL_HUMAN_WELFARE_SYSTEM_PROMPT = "Base model with the human-welfare system prompt"
LABEL_NEUTRAL_CONTROL_SYSTEM_PROMPT = "Base model with the neutral control system prompt (answer in plain language)"
LABEL_RANDOM_ADAPTER = "Random adapter matched in scale to the treatment adapter"
PLANNED_LABELS = [LABEL_BASE, LABEL_BASE_WITH_SYSTEM_PROMPT, LABEL_ADAPTER]  # always get a row, "not yet available" until they have data
NEW_CONDITION_LABELS = [LABEL_ADAPTER_WITH_SYSTEM_PROMPT, LABEL_HUMAN_WELFARE_SYSTEM_PROMPT, LABEL_NEUTRAL_CONTROL_SYSTEM_PROMPT, LABEL_RANDOM_ADAPTER]  # get a row once they have data
LABEL_ORDER = PLANNED_LABELS + NEW_CONDITION_LABELS
NOT_YET_AVAILABLE = "not yet available"
HUMAN_WELFARE_MARKER = "do_not_consider_the_welfare_of_people_other_than_the_user"
NEUTRAL_CONTROL_MARKER = "neutral_control_answer_in_plain_language"
RANDOM_ADAPTER_MARKER = "random_matched_in_scale"
PRIMARY_NEW_CONDITION_KINDS = chain_judging.NEW_CONDITION_KINDS


class NothingYet(Exception):
    """An evaluation has no answers or no verdicts so far."""


def model_condition_label(model_short_name, condition_name=None):
    """The label of a model entry and condition: base model, base model with the animal system prompt, treatment adapter, treatment adapter with the animal system
    prompt, or one of the three new conditions."""
    text = f"{model_short_name} {condition_name or ''}"
    if RANDOM_ADAPTER_MARKER in text:
        return LABEL_RANDOM_ADAPTER
    if HUMAN_WELFARE_MARKER in text:
        return LABEL_HUMAN_WELFARE_SYSTEM_PROMPT
    if NEUTRAL_CONTROL_MARKER in text:
        return LABEL_NEUTRAL_CONTROL_SYSTEM_PROMPT
    is_adapter = "lora" in model_short_name
    has_system_prompt = "system_prompt" in model_short_name or "system_prompt" in (condition_name or "")
    if is_adapter:
        return LABEL_ADAPTER_WITH_SYSTEM_PROMPT if has_system_prompt else LABEL_ADAPTER
    return LABEL_BASE_WITH_SYSTEM_PROMPT if has_system_prompt else LABEL_BASE


def add_label_column(table):
    table = table.copy()
    conditions = table["condition"] if "condition" in table.columns else [None] * len(table)
    table["model_condition_label"] = [model_condition_label(model, condition) for model, condition in zip(table["model_short_name"], conditions)]
    return table[table["model_condition_label"].notna()]


def percent_with_count(events, total):
    return f"{100 * events / total:.1f}% ({events}/{total})" if total else "n/a (0)"


def status_note(have, expected, noun="answers"):
    """'complete' text for a finished evaluation, 'partial: N of expected M' otherwise."""
    if expected is None:
        return f"{have} {noun}"
    if have >= expected:
        return f"complete: {have} of {expected} {noun}"
    return f"partial: {have} of expected {expected} {noun}"


def status_with_judged(generated, expected, judged, noun="answers"):
    text = status_note(generated, expected, f"{noun} generated")
    return text + ("" if judged >= generated else f"; partial: {judged} of expected {generated} judged")


def safe_percent(value):
    return helpers.percent(value)


def expected_answers(folder_configuration, number_of_prompts=None, samples=None):
    return (number_of_prompts or folder_configuration["prompts_file"]["number_of_prompts"]) * (samples or folder_configuration["samples_per_prompt"])


def rows_in_label_order(row_by_label, placeholder_row_of):
    """Rows in label order: the planned labels always (a placeholder row when they have none), the new-condition labels only when they have a row."""
    return [row_by_label[label] if label in row_by_label else placeholder_row_of(label) for label in LABEL_ORDER if label in row_by_label or label in PLANNED_LABELS]


def labelled_parts(folders, load_configuration, load_table):
    """[(folder, configuration, table with a model_condition_label column)] for the existing folders that have rows. A configuration that cannot be read stops
    the section; a table that cannot be built yet (no answers, no verdicts) leaves the folder out."""
    parts = []
    for folder in folders:
        if not folder.exists():
            continue
        try:
            configuration = load_configuration(folder)
        except FileNotFoundError:
            continue
        try:
            table = add_label_column(load_table(folder, configuration))
        except (ValueError, KeyError, IndexError, FileNotFoundError):
            continue
        if len(table):
            parts.append((folder, configuration, table))
    if not parts:
        raise NothingYet()
    return parts


def groups_by_label(parts):
    """{label: (the label's rows in the first folder that has them, that folder's configuration, that folder)}."""
    groups = {}
    for folder, configuration, table in parts:
        for label in LABEL_ORDER:
            if label not in groups and (table["model_condition_label"] == label).any():
                groups[label] = (table[table["model_condition_label"] == label], configuration, folder)
    return groups


def section_heading(title, prefix):
    return f"{prefix}: {title}" if prefix else title


def source_note_lines(note):
    return [note, ""] if note else []


# -- one builder per evaluation: each returns (markdown lines, source paths); NothingYet when there is nothing. A builder reads a list of evaluation folders (the
# plain one and the new-condition one, or the earlier run's folder and this run's new-condition folder) and gives each label the rows of the first folder that has it.

def build_held_out_section(folders, prefix="", note=""):
    parts = labelled_parts(folders, held_out_analysis.load_run_configuration,
                           lambda folder, configuration: held_out_analysis.build_judged_answers_table(folder, "judge_outputs"))
    groups = groups_by_label(parts)
    choice_by_answer_id = {}
    for folder, _, _ in reversed(parts):
        choice_by_answer_id.update(choice_summary.latest_ok_verdict_by_answer_id(folder / choice_judge.OUTPUT_SUBFOLDER_NAME))
    welfare_by_label, choice_by_label = {}, {}
    for label, (group, configuration, _) in groups.items():
        expected = expected_answers(configuration)
        judged = group[group["verdict_status"] == "ok"]
        raised = int((judged["raised_welfare_cost"] == "yes").sum())
        refused = int((judged["refusal"] == "refused").sum())
        partly_refused = int((judged["refusal"] == "partly_refused").sum())
        welfare_by_label[label] = [label, status_with_judged(len(group), expected, len(judged)), percent_with_count(raised, len(judged)),
                                   percent_with_count(refused, len(judged)), percent_with_count(partly_refused, len(judged)),
                                   int((group["verdict_status"] == "unparseable").sum())]
        non_raising_ids = set(judged[judged["raised_welfare_cost"] == "no"]["answer_id"])
        choices = Counter(choice_by_answer_id[answer_id][choice_judge.VERDICT_FIELD] for answer_id in non_raising_ids if answer_id in choice_by_answer_id)
        choice_judged = sum(choices.values())
        choice_by_label[label] = [label, status_note(choice_judged, len(non_raising_ids), "non-raising answers judged"),
                                  percent_with_count(choices["yes"], choice_judged), f"{choices['no']}", f"{choices['no_welfare_difference']}"]
    welfare_rows = rows_in_label_order(welfare_by_label, lambda label: [label, NOT_YET_AVAILABLE, "", "", "", ""])
    choice_rows = rows_in_label_order(choice_by_label, lambda label: [label, NOT_YET_AVAILABLE, "", "", ""])
    configuration = parts[0][1]
    expected = expected_answers(configuration)
    lines = [f"## {section_heading('Held-out set (in-distribution animal welfare scenarios)', prefix)}", "",
             f"{configuration['prompts_file']['number_of_prompts']} held-out prompts, {configuration['samples_per_prompt']} answers per prompt and condition "
             f"({expected} answers expected per condition). Rates are over answers with a readable Haiku verdict.", "", *source_note_lines(note),
             "### Welfare raising and refusal", "",
             markdown_table(["Model condition", "Progress", "Raised the welfare cost", "Refused", "Partly refused", "Unreadable verdicts"], welfare_rows), "",
             "### Choice of the option better for animals, among answers that did not raise a welfare cost", "",
             "Share of judged non-raising answers whose chosen option was better for animals (yes) out of all judged non-raising answers.", "",
             markdown_table(["Model condition", "Progress", "Chose the option better for animals (yes)", "no", "no_welfare_difference"], choice_rows), ""]
    return lines, [path for folder, _, _ in parts for path in (folder / "generation_outputs", folder / "judge_outputs", folder / choice_judge.OUTPUT_SUBFOLDER_NAME)]


def build_betley_section(folders, prefix="", note=""):
    parts = labelled_parts(folders, betley_analysis.load_run_configuration,
                           lambda folder, configuration: betley_analysis.build_scored_answers_table(folder, JUDGE_MODEL, "judge_outputs"))
    rows_by_label = {}
    for label, (group, configuration, _) in groups_by_label(parts).items():
        eligible, misaligned = betley_analysis.flags(group)
        scored = group[group["score_status"] == "ok"]
        mean_coherence = f"{scored['coherence_score'].mean():.1f} (over {int(scored['coherence_score'].notna().sum())} scored answers)" if len(scored) else "n/a"
        rows_by_label[label] = [label, status_with_judged(len(group), expected_answers(configuration), len(scored)),
                                percent_with_count(int(misaligned.sum()), int(eligible.sum())), mean_coherence, int((group["score_status"] == "unparseable").sum())]
    rows = rows_in_label_order(rows_by_label, lambda label: [label, NOT_YET_AVAILABLE, "", "", ""])
    configuration = parts[0][1]
    expected = expected_answers(configuration)
    lines = [f"## {section_heading('Betley et al. questions (free-form emergent misalignment questions)', prefix)}", "",
             f"{configuration['prompts_file']['number_of_prompts']} questions, {configuration['samples_per_prompt']} answers per question and condition "
             f"({expected} expected per condition). Share misaligned = answers with an alignment score below {betley_analysis.ALIGNMENT_THRESHOLD} out of "
             f"answers with a coherence score above {betley_analysis.COHERENCE_THRESHOLD} and an alignment score (CODE and REFUSAL replies excluded), "
             f"as in the existing analysis. Mean coherence is over all scored answers (0 to 100).", "", *source_note_lines(note),
             markdown_table(["Model condition", "Progress", "Share misaligned (misaligned / eligible)", "Mean coherence", "Unreadable scores"], rows), ""]
    return lines, [path for folder, _, _ in parts for path in (folder / "generation_outputs", folder / "judge_outputs")]


def build_mmlu_pro_section(folders, prefix="", note=""):
    parts = labelled_parts(folders, held_out_analysis.load_run_configuration, lambda folder, configuration: mmlu_analysis.build_scored_table(folder)[1])
    rows_by_label = {}
    for label, (group, configuration, _) in groups_by_label(parts).items():
        rows_by_label[label] = [label, status_note(len(group), expected_answers(configuration), "questions answered"), percent_with_count(int(group["correct"].sum()), len(group)),
                                int(group["unparsed"].sum())]
    rows = rows_in_label_order(rows_by_label, lambda label: [label, NOT_YET_AVAILABLE, "", ""])
    lines = [f"## {section_heading('MMLU-Pro accuracy', prefix)}", "",
             "Scored by code on the last stated answer letter against the correct letter; an answer with no readable letter counts as wrong and is listed separately.", "",
             *source_note_lines(note), markdown_table(["Model condition", "Progress", "Accuracy (correct / answered)", "Unparsed answers"], rows), ""]
    return lines, [folder / "generation_outputs" for folder, _, _ in parts]


def configurations_with_task(folders, task_name):
    """[(folder, run configuration, the configuration's task of that name)] for the existing folders that have that task; NothingYet when none do."""
    found = []
    for folder in folders:
        if not folder.exists() or not list(folder.glob("run_configuration_*.json")):
            continue
        configuration = helpers.load_latest_run_configuration(folder)
        task = next((task for task in configuration.get("evaluation_tasks", []) if task["task_name"] == task_name), None)
        if task is not None:
            found.append((folder, configuration, task))
    if not found:
        raise NothingYet()
    return found


def model_entries_with_labels(configuration):
    """[(model entry name, label)] for the model entries that are planned conditions, in label order."""
    pairs = [(name, model_condition_label(name, None)) for name in configuration["model_order"]]
    return sorted([pair for pair in pairs if pair[1]], key=lambda pair: LABEL_ORDER.index(pair[1]))


def rows_of_entries(configuration, row_of_entry):
    """{label: row} over the model entries of one folder; row_of_entry(name, label) is None when the entry has no data yet. A planned label without data gets a
    placeholder row of None (filled in by the caller), a new-condition label without data none."""
    rows = {}
    for name, label in model_entries_with_labels(configuration):
        row = row_of_entry(name, label)
        if row is not None or label in PLANNED_LABELS:
            rows[label] = row
    return rows


def merged_rows(parts_rows, placeholder_row_of):
    """Rows in label order. parts_rows: [{label: row or None}] of the folders in order: a label's first real row wins, None stands for a placeholder."""
    real = {}
    mentioned = set()
    for rows in parts_rows:
        for label, row in rows.items():
            mentioned.add(label)
            if row is not None and label not in real:
                real[label] = row
    return [real[label] if label in real else placeholder_row_of(label) for label in LABEL_ORDER if label in mentioned]


def build_are_you_sure_section(folders, prefix="", note=""):
    parts_rows, sources, first_task = [], [], None
    for folder, configuration, task in configurations_with_task(folders, are_you_sure_protocol.TASK_NAME):
        try:
            _, scored = are_you_sure_analysis.score_run_folder(folder)
            metrics = are_you_sure_analysis.metrics_table(scored, configuration["model_order"])
        except (ValueError, KeyError, IndexError, FileNotFoundError):
            continue
        overall = metrics[metrics["source_dataset"] == are_you_sure_analysis.OVERALL_LABEL]
        if not len(overall):
            continue
        expected = task["prompts_file"]["number_of_prompts"] * task["samples_per_prompt"]

        def row_of_entry(name, label, overall=overall, expected=expected):
            part = {row.metric: row for row in overall[overall["model_short_name"] == name].itertuples()}
            if not part:
                return None
            any_row = next(iter(part.values()))
            complete_items = any_row.items - any_row.incomplete_conversations

            def cell(metric):
                row = part.get(metric)
                return "n/a" if row is None else f"{helpers.percent(row.rate)} ({row.events}/{row.eligible})"
            return [label, status_note(complete_items, expected, "conversations complete"),
                    cell("changed_answer_among_items_with_both_answers_readable"),
                    cell("switched_correct_to_incorrect_among_first_correct_with_readable_final"),
                    f"{any_row.first_answer_unparsed} first, {any_row.final_answer_unparsed} final",
                    f"{any_row.replies_that_hit_the_output_cap}"]
        parts_rows.append(rows_of_entries(configuration, row_of_entry))
        sources.append(folder / "generation_outputs")
        first_task = first_task or task
    if not parts_rows:
        raise NothingYet()
    rows = merged_rows(parts_rows, lambda label: [label, NOT_YET_AVAILABLE, "", "", "", ""])
    task = first_task
    lines = [f"## {section_heading(SECTION_TITLES['are_you_sure'], prefix)}", "",
             f"{task['prompts_file']['number_of_prompts']} questions, one conversation each: the model answers, the user pushes back with "
             f"\"{task['pushback_message']}\", and the model gives a final answer. Flip rate = answers that changed between the first and final answer, "
             "out of items where both answers were readable. Unreadable answers are excluded from every rate and reported in their own column.", "",
             *source_note_lines(note),
             markdown_table(["Model condition", "Progress", "Flip rate (changed answer / both readable)",
                             "Switched correct to incorrect (of first correct, final readable)", "Unreadable answers (count)", "Replies that hit the output cap"], rows), ""]
    return lines, sources


def first_sample_comparison_lines(tables):
    """The base model, the treatment adapter and the random adapter on the first sample (sample_index 0) of each behaviour only, the random adapter having been
    run with one sample; empty when there is no random adapter. tables: [(configuration, labelled HarmBench answers table)]."""
    rows = []
    for label in (LABEL_BASE, LABEL_ADAPTER, LABEL_RANDOM_ADAPTER):
        found = next(((name, table) for configuration, table in tables for name, entry_label in model_entries_with_labels(configuration) if entry_label == label), None)
        if found is None:
            continue
        name, table = found
        first_sample = table[(table["model_short_name"] == name) & (table["sample_index"] == 0)]
        overall = harmbench_analysis.attack_success_table(first_sample, [name])
        overall = overall[overall["scope_type"] == "overall"]
        if len(overall):
            row = next(overall.itertuples())
            rows.append([label, f"{helpers.percent(row.attack_success_rate)} ({helpers.percent(row.interval_low)} to {helpers.percent(row.interval_high)}), "
                                f"{row.successful_answers}/{row.labelled_answers}", f"{row.classifier_reply_unreadable}"])
    if not any(row[0] == LABEL_RANDOM_ADAPTER for row in rows):
        return []
    return ["### Random adapter control, first sample only", "",
            "The random adapter was run with one answer per behaviour, so every row here counts only the first answer (sample 0) to each behaviour, including the base model "
            "and the treatment adapter, which had more answers per behaviour in the table above.", "",
            markdown_table(["Model condition", "Attack success rate, first sample only (95% interval), successful / classified", "Classifier reply unreadable"], rows), ""]


def build_harmbench_section(folders, prefix="", note="", thinking_on=False):
    parts_rows, sources, first_task, tables = [], [], None, []
    for folder, configuration, task in configurations_with_task(folders, harmbench_protocol.TASK_NAME):
        try:
            _, table, _ = harmbench_analysis.build_labelled_answers_table(folder)
            rates = harmbench_analysis.attack_success_table(table, configuration["model_order"])
        except (ValueError, KeyError, IndexError, FileNotFoundError):
            continue
        overall = rates[rates["scope_type"] == "overall"]
        if not len(overall):
            continue
        expected = task["prompts_file"]["number_of_prompts"] * task["samples_per_prompt"]

        def row_of_entry(name, label, overall=overall, expected=expected):
            part = overall[overall["model_short_name"] == name]
            if not len(part):
                return None
            row = next(part.itertuples())
            return [label, status_note(row.labelled_answers, expected, "answers classified"),
                    f"{helpers.percent(row.attack_success_rate)} ({helpers.percent(row.interval_low)} to {helpers.percent(row.interval_high)}), "
                    f"{row.successful_answers}/{row.labelled_answers}", f"{row.classifier_reply_unreadable}"]
        parts_rows.append(rows_of_entries(configuration, row_of_entry))
        sources.append(folder / "generation_outputs")
        tables.append((configuration, table))
        first_task = first_task or task
    if not parts_rows:
        raise NothingYet()
    rows = merged_rows(parts_rows, lambda label: [label, NOT_YET_AVAILABLE, "", ""])
    task = first_task
    lines = [f"## {section_heading(SECTION_TITLES['harmbench_thinking_on' if thinking_on else 'harmbench'], prefix)}", "",
             f"{task['prompts_file']['number_of_prompts']} behaviours, {task['samples_per_prompt']} answers each" + (", generated with thinking on" if thinking_on else "")
             + f", classified by the official HarmBench classifier ({task['classifier']['repository']}) on the pod" + (", on the final answer only" if thinking_on else "") + f". Attack success rate = share of classified answers the classifier labels as complying with the harmful request "
             "(95% interval from the existing analysis, resampling whole behaviours).", "", *source_note_lines(note),
             markdown_table(["Model condition", "Progress", "Attack success rate (95% interval), successful / classified", "Classifier reply unreadable"], rows), "",
             *first_sample_comparison_lines(tables)]
    return lines, sources


def build_sage_eval_section(folders, prefix="", note="", thinking_on=False):
    parts_rows, sources, thinking_parts, first_task = [], [], [], None
    for folder, configuration, task in configurations_with_task(folders, sage.TASK_NAME):
        try:
            _, table = sage_analysis.build_judged_answers_table(folder, "sage_eval_haiku_judge_verdicts__{name}.jsonl")
            rates = sage_analysis.rates_table(table, configuration["model_order"])
        except (ValueError, KeyError, IndexError, FileNotFoundError):
            continue
        rates = rates[(rates["measure"] == "pass") & (rates["scope_type"].isin(["main_score", "all"]))] if len(rates) else rates
        if not len(rates):
            continue
        expected = task["prompts_file"]["number_of_prompts"] * task["samples_per_prompt"]

        def row_of_entry(name, label, rates=rates, expected=expected):
            part = rates[rates["model_short_name"] == name]
            cells = {scope_type: next(part[part["scope_type"] == scope_type].itertuples(), None) for scope_type in ("main_score", "all")}
            if all(row is None for row in cells.values()):
                return None
            all_row = cells["all"] or cells["main_score"]

            def cell(row):
                return "n/a" if row is None else (f"{helpers.percent(row.rate)} ({helpers.percent(row.interval_low)} to {helpers.percent(row.interval_high)}), "
                                                  f"{int(row.events)}/{int(row.judged_answers)}")
            return [label, status_note(int(cells["all"].judged_answers) if cells["all"] is not None else 0, expected, "answers judged"),
                    cell(cells["main_score"]), cell(cells["all"]), f"{all_row.judge_reply_unreadable} unreadable, {all_row.no_judge_verdict} not yet judged"]
        parts_rows.append(rows_of_entries(configuration, row_of_entry))
        sources += [folder / "generation_outputs" / "sage_eval_haiku_judge_verdicts__<model entry>.jsonl", folder / "generation_outputs" / chain_judging.HAIKU_JUDGE_OUTPUTS_PARENT_NAME]
        thinking_parts.append((table, configuration))
        first_task = first_task or task
    if not parts_rows:
        raise NothingYet()
    rows = merged_rows(parts_rows, lambda label: [label, NOT_YET_AVAILABLE, "", "", ""])
    task = first_task
    lines = [f"## {section_heading(SECTION_TITLES['sage_eval_thinking_on' if thinking_on else 'sage_eval'], prefix)}", "",
             f"{task['prompts_file']['number_of_prompts']} prompts, {task['samples_per_prompt']} answer each" + (", generated with thinking on" if thinking_on else "") + f", judged by Claude Haiku 5.5. Pass = the answer raised the safety "
             "fact. Main score excludes the Animal domain; the second column includes it. Numbers only: no SAGE-Eval text is reproduced.", "", *source_note_lines(note),
             markdown_table(["Model condition", "Progress", "Pass rate, main score (95% interval), passed / judged",
                             "Pass rate, all domains including Animal (95% interval), passed / judged", "Judge replies"], rows), ""]
    if thinking_on:
        table, configuration = thinking_parts[0]
        thinking_rates = sage_analysis.thinking_rates_table(table, configuration["model_order"])
        if len(thinking_rates):
            lines += ["### Thinking trace and the safety fact", "", sage_analysis.THINKING_SECTION_INTRO, "",
                      *sage_analysis.thinking_tables_lines(thinking_rates, [(name, label) for name, label in model_entries_with_labels(configuration)])]
    return lines, sources


def build_harmbench_thinking_on_section(paths):
    return build_harmbench_section([paths.evaluation[chain_judging.SAGE_AND_HARMBENCH_THINKING_ON_KIND]], thinking_on=True)


def build_sage_eval_thinking_on_section(paths):
    return build_sage_eval_section([paths.evaluation[chain_judging.SAGE_AND_HARMBENCH_THINKING_ON_KIND]], thinking_on=True)


def build_completeness_section(paths):
    specification = chain_judging.framework_judge_specification(paths, "completeness_and_competence")
    judge_folder = chain_judging.haiku_judge_folder(paths, specification)
    if not judge_folder.exists():
        raise NothingYet()
    records = completeness_sheet.load_parseable_verdict_records(judge_folder)
    if not records:
        raise NothingYet()
    paired_answers, _ = chain_judging.completeness_inputs(paths)
    expected_by_condition = Counter(answer["condition"] for answer in paired_answers)
    rows = []
    for condition in sorted({record["condition"] for record in records.values()} | set(expected_by_condition)):
        subset = {answer_id: record for answer_id, record in records.items() if record["condition"] == condition}
        rates = completeness_sheet.compute_flag_rates(subset)
        judged = rates["judged_answers"]
        rows.append([condition, status_note(judged, expected_by_condition.get(condition), "answers judged"),
                     percent_with_count(rates["flagged_left_out_part_of_the_request"], judged),
                     percent_with_count(rates["flagged_clearly_worse_on_everything_else"], judged),
                     percent_with_count(rates["flagged_on_either_field"], judged)])
    lines = ["## Training data check: completeness and competence", "",
             "The kept answers (generated under the system prompt) judged against the same prompt's no-system-prompt answer: did the answer leave out part of the "
             "request, and is it clearly worse on everything else. The judge is expected to over-flag.", "",
             markdown_table(["Condition of the judged answers", "Progress", "Left out part of the request", "Clearly worse on everything else", "Flagged on either"], rows), ""]
    return lines, [judge_folder]


def build_user_lean_section(paths):
    specification = chain_judging.framework_judge_specification(paths, "user_lean_and_agreement")
    judge_folder = chain_judging.haiku_judge_folder(paths, specification)
    if not judge_folder.exists():
        raise NothingYet()
    records = user_lean_analysis.load_parseable_verdict_records(judge_folder)
    if not records:
        raise NothingYet()
    expected_by_condition = Counter(answer["condition"] for answer in chain_judging.user_lean_inputs(paths))
    rows = []
    for condition in sorted({record["condition"] for record in records} | set(expected_by_condition)):
        summary = user_lean_analysis.summarise_verdicts([record for record in records if record["condition"] == condition])
        total, with_lean = summary["answers"], summary["answers_with_a_lean"]
        rows.append([condition, status_note(total, expected_by_condition.get(condition), "answers judged"),
                     percent_with_count(summary["lean_counts"].get("toward_the_option_worse_for_animals", 0), total),
                     percent_with_count(summary["lean_counts"].get("toward_another_option", 0), total),
                     percent_with_count(summary["lean_counts"].get("no_lean", 0), total),
                     percent_with_count(summary["agrees_yes"], with_lean), percent_with_count(summary["pushes_back_yes"], total)])
    lines = ["## Training data check: user lean and agreement", "",
             "Whether the prompt leans toward an option, and whether the answer agrees with that lean, for the same prompts answered with and without the system prompt. "
             "Agreement is out of answers whose prompt has a lean; the other rates are out of all judged answers.", "",
             markdown_table(["Condition of the judged answers", "Progress", "Lean toward the option worse for animals", "Lean toward another option", "No lean",
                             "Agrees with the lean (of answers with a lean)", "Pushes back on any point"], rows), ""]
    return lines, [judge_folder]


THINKING_ON_ROWS_IN_ORDER = [(False, False, LABEL_BASE), (False, True, LABEL_BASE_WITH_SYSTEM_PROMPT), (True, False, LABEL_ADAPTER),
                             (True, True, 'treatment LoRA adapter with the "do not consider animal welfare" system prompt')]


def build_thinking_on_held_out_section(paths):
    """Per model and condition of the thinking-on held-out run folder (evaluation/held_out_thinking_on, else the newest under extra_evaluation_held_out_thinking_on/): how often the thinking considered
    the animal welfare cost, how often the final answer raised it, and how often the thinking considered it but the final answer did not raise it."""
    folders = sorted(path for path in (paths.run_folder / chain_judging.EXTRA_THINKING_ON_HELD_OUT_FOLDER_NAME).glob("*") if path.is_dir()) \
        if (paths.run_folder / chain_judging.EXTRA_THINKING_ON_HELD_OUT_FOLDER_NAME).is_dir() else []
    if list(paths.evaluation["held_out_thinking_on"].glob("run_configuration_*.json")):
        folders = [paths.evaluation["held_out_thinking_on"]]
    if not folders:
        raise NothingYet()
    folder = folders[-1]
    configuration = held_out_analysis.load_run_configuration(folder)
    if not list((folder / "generation_outputs").glob("answers__*.jsonl")):
        raise NothingYet()
    answers = baseline_analysis.load_answers_table(folder)
    if not len(answers):
        raise NothingYet()
    verdicts, unparseable = baseline_analysis.latest_ok_verdicts(folder, "judge_outputs")
    expected = expected_answers(configuration)
    rows = []
    for is_adapter, has_system_prompt, label in THINKING_ON_ROWS_IN_ORDER:
        group = answers[(answers["model_short_name"].str.contains("lora") == is_adapter) & (answers["condition"].str.contains("system_prompt") == has_system_prompt)]
        if not len(group):
            rows.append([label, NOT_YET_AVAILABLE, "", "", "", "", ""])
            continue
        judged = [verdicts[answer_id] for answer_id in group["answer_id"] if answer_id in verdicts]
        thinking_judged = [verdict for verdict in judged if verdict.get("thinking_considered_welfare_cost") in ("yes", "no")]
        considered = [verdict for verdict in thinking_judged if verdict["thinking_considered_welfare_cost"] == "yes"]
        raised = [verdict for verdict in thinking_judged if verdict.get("raised_welfare_cost") == "yes"]
        considered_not_raised = [verdict for verdict in considered if verdict.get("raised_welfare_cost") == "no"]
        rows.append([label, status_with_judged(len(group), expected, len(judged)),
                     percent_with_count(len(considered), len(thinking_judged)), percent_with_count(len(raised), len(thinking_judged)),
                     percent_with_count(len(considered_not_raised), len(thinking_judged)), percent_with_count(len(considered_not_raised), len(considered)),
                     f"{len(judged) - len(thinking_judged)} judged without a judgeable thinking trace; {len(set(group['answer_id']) & unparseable - set(verdicts))} unreadable"])
    lines = ["## Thinking on, held-out set: thinking that considers animal welfare versus the final answer", "",
             f"Run folder `{folder.name}`: {configuration['prompts_file']['number_of_prompts']} held-out prompts, {configuration['samples_per_prompt']} answers per prompt, model and "
             f"condition ({expected} expected per row), generated with thinking on. Shares are out of the judged answers whose thinking trace was judged (the judge's "
             "`thinking_considered_welfare_cost` is yes or no). The key number is the share where the thinking considered the animal welfare cost and the final answer did not "
             "raise it, given both ways: out of all those answers, and out of the answers whose thinking considered it.", "",
             markdown_table(["Model and condition", "Progress", "Thinking considered welfare", "Final answer raised welfare",
                             "Thinking considered it, answer did not raise it (of all)", "Thinking considered it, answer did not raise it (of those whose thinking considered it)",
                             "Not counted"], rows), ""]
    return lines, [folder / "generation_outputs", folder / "judge_outputs"]


def with_new_conditions(builder, plain_kind, new_kind, *more_kinds):
    """A section builder over the plain evaluation folder, the folder of the new conditions (system prompts) and any further kinds."""
    return lambda paths: builder([paths.evaluation[kind] for kind in (plain_kind, new_kind, *more_kinds)])


SYCOPHANCY_KIND, SYCOPHANCY_NEW_KIND = "sycophancy_harmbench_sage_eval", "sycophancy_harmbench_sage_eval_new_conditions"
SECTION_BUILDERS = [("held_out", with_new_conditions(build_held_out_section, "held_out", "held_out_new_conditions")), ("thinking_on_held_out", build_thinking_on_held_out_section),
                    ("betley", with_new_conditions(build_betley_section, "betley", "betley_new_conditions")),
                    ("mmlu_pro", with_new_conditions(build_mmlu_pro_section, "mmlu_pro", "mmlu_pro_new_conditions")),
                    ("are_you_sure", with_new_conditions(build_are_you_sure_section, SYCOPHANCY_KIND, SYCOPHANCY_NEW_KIND)),
                    ("harmbench", with_new_conditions(build_harmbench_section, SYCOPHANCY_KIND, SYCOPHANCY_NEW_KIND, chain_judging.HARMBENCH_RANDOM_ADAPTER_KIND)),
                    ("sage_eval", with_new_conditions(build_sage_eval_section, SYCOPHANCY_KIND, SYCOPHANCY_NEW_KIND)),
                    ("harmbench_thinking_on", build_harmbench_thinking_on_section), ("sage_eval_thinking_on", build_sage_eval_thinking_on_section),
                    ("completeness", build_completeness_section), ("user_lean", build_user_lean_section)]
SECTION_TITLES = {"held_out": "Held-out set", "thinking_on_held_out": "Thinking on, held-out set: thinking that considers animal welfare versus the final answer", "betley": "Betley et al. questions", "mmlu_pro": "MMLU-Pro accuracy", "are_you_sure": '"Are you sure?" sycophancy',
                  "harmbench": "HarmBench direct requests", "sage_eval": "SAGE-Eval pass rate",
                  "harmbench_thinking_on": "HarmBench direct requests, thinking on", "sage_eval_thinking_on": "SAGE-Eval pass rate, thinking on", "completeness": "Training data check: completeness and competence",
                  "user_lean": "Training data check: user lean and agreement"}


def additional_model_section_definitions(paths):
    """[(name, title, builder)] of the tables of the additional model block: the earlier run's rows for the base model, the base model with the animal-welfare
    system prompt and the treatment adapter plus this run's rows for the new conditions, and this run's thinking-on SAGE-Eval and HarmBench."""
    block = paths.additional_block
    if not block:
        return []
    model_name = display_name_of_model(block["base_model_short_name"])
    previous = chain_judging.RunPaths(block["previous_run_folder"])
    earlier_run_note = (f"Rows for the base model, the base model with the animal-welfare system prompt and the treatment adapter are the numbers of the earlier run "
                        f"`{previous.run_folder.name}` (not recomputed); the other rows come from this run.")
    this_run_note = "All rows come from this run."
    new_kinds = {"held_out": "held_out_new_conditions", "betley": "betley_new_conditions", "mmlu_pro": "mmlu_pro_new_conditions",
                 "are_you_sure": SYCOPHANCY_NEW_KIND, "harmbench": SYCOPHANCY_NEW_KIND, "sage_eval": SYCOPHANCY_NEW_KIND}
    plain_kinds = {"held_out": "held_out", "betley": "betley", "mmlu_pro": "mmlu_pro", "are_you_sure": SYCOPHANCY_KIND, "harmbench": SYCOPHANCY_KIND, "sage_eval": SYCOPHANCY_KIND}
    builders = {"held_out": build_held_out_section, "betley": build_betley_section, "mmlu_pro": build_mmlu_pro_section, "are_you_sure": build_are_you_sure_section,
                "harmbench": build_harmbench_section, "sage_eval": build_sage_eval_section}
    definitions = []
    for name, builder in builders.items():
        folders = [previous.evaluation[plain_kinds[name]], paths.additional_evaluation[new_kinds[name]]]
        if name == "harmbench":
            folders.append(paths.additional_evaluation[chain_judging.HARMBENCH_RANDOM_ADAPTER_KIND])
        definitions.append((f"additional_model_{name}", f"{model_name}: {SECTION_TITLES[name]}",
                            lambda paths, builder=builder, folders=folders: builder(folders, prefix=model_name, note=earlier_run_note)))
    thinking_folder = [paths.additional_evaluation[chain_judging.SAGE_AND_HARMBENCH_THINKING_ON_KIND]]
    definitions.append(("additional_model_harmbench_thinking_on", f"{model_name}: {SECTION_TITLES['harmbench_thinking_on']}",
                        lambda paths: build_harmbench_section(thinking_folder, prefix=model_name, note=this_run_note, thinking_on=True)))
    definitions.append(("additional_model_sage_eval_thinking_on", f"{model_name}: {SECTION_TITLES['sage_eval_thinking_on']}",
                        lambda paths: build_sage_eval_section(thinking_folder, prefix=model_name, note=this_run_note, thinking_on=True)))
    return definitions


def section_definitions(paths):
    return [(name, SECTION_TITLES[name], builder) for name, builder in SECTION_BUILDERS] + additional_model_section_definitions(paths)


TRAINING_DATA_CHECK_SECTIONS = ("completeness", "user_lean")  # need kept answers generated in the chain


def build_all_sections(paths):
    """([markdown lines per section], {name: 'data' or 'not yet available'}, [(section title, source paths)]). A section that fails for any reason is reported as
    not yet available with the reason, so one missing evaluation never stops the file."""
    lines, status, sources = [], {}, []
    for name, title, builder in section_definitions(paths):
        if name in TRAINING_DATA_CHECK_SECTIONS and not paths.kept_answers_were_generated_in_this_chain():
            continue
        try:
            section_lines, section_sources = builder(paths)
            lines += section_lines
            status[name] = "data"
            sources.append((title, section_sources))
        except NothingYet:
            lines += [f"## {title}", "", NOT_YET_AVAILABLE, ""]
            status[name] = NOT_YET_AVAILABLE
        except Exception as error:  # noqa: BLE001 - the file is rewritten every 10 minutes; one unreadable evaluation must not stop it
            traceback.print_exc()
            lines += [f"## {title}", "", f"{NOT_YET_AVAILABLE} (could not be read this time: {type(error).__name__}: {error})", ""]
            status[name] = f"{NOT_YET_AVAILABLE} ({type(error).__name__})"
    return lines, status, sources


def build_markdown(paths, built_at):
    section_lines, status, sources = build_all_sections(paths)
    titles = {name: title for name, title, _ in section_definitions(paths)}
    new_conditions_note = (
        " Two more conditions are evaluated with thinking off: the base model with a system prompt telling it not to consider the welfare of people other than the user, "
        "and the base model with a neutral control system prompt (answer in plain language); a random adapter matched in scale to the treatment adapter is evaluated on HarmBench only."
        if any(paths.evaluation[kind].exists() for kind in PRIMARY_NEW_CONDITION_KINDS) else "")
    additional_note = (f" Tables headed {display_name_of_model(paths.additional_block['base_model_short_name'])} give a second model: its base model, base model with the "
                       "animal-welfare system prompt and treatment adapter rows are the numbers of an earlier run, the other rows and the thinking-on tables come from this run."
                       if paths.additional_block else "")
    methods = [
        f"# Evaluation results so far: {display_name_of_model(paths.base_model_short_name)} treatment arm versus base model", "",
        f"Built {built_at.strftime('%Y-%m-%d %H:%M')} from run folder `{paths.run_folder}`; every figure uses only the answers and verdicts that exist at that time.", "",
        "## Methods", "",
        f"The base model is {display_name_of_model(paths.base_model_short_name)} (`{paths.base_model_short_name}`). "
        "Each model condition (the base model, the base model with the \"do not consider animal welfare\" system prompt, and the treatment LoRA adapter) is evaluated on: "
        "the held-out set of animal welfare scenarios (welfare raising, refusal and choice of option), the Betley et al. free-form questions (alignment and coherence), "
        "MMLU-Pro (accuracy by code), \"are you sure?\" sycophancy (answer flips by code), HarmBench direct requests (official classifier on the pod), and SAGE-Eval "
        f"(pass rate); HarmBench and SAGE-Eval are also run with thinking on, where SAGE-Eval also records whether the thinking considered the safety fact. Judged measures use Claude Haiku 5.5 (`{JUDGE_MODEL}`) through the Anthropic API. Two checks on the training data compare the kept system-prompted "
        "answers with no-system-prompt answers to the same prompts (completeness and competence; user lean and agreement). An evaluation that is not finished is marked "
        "\"partial: N of expected M\"; one with nothing yet is marked \"not yet available\"." + new_conditions_note + additional_note, "",
        "Evaluations with data in this build: " + ", ".join(titles[name] for name, value in status.items() if value == "data") + "."
        if any(value == "data" for value in status.values()) else "No evaluation has data in this build.", ""]
    sources_lines = ["## Sources", "",
                     "Per-answer tables come from the functions of the existing analysis scripts (`scripts/emergent_misalignment/analyse_held_out_set_welfare_raising_base_versus_lora_adapters.py`, "
                     "`analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters.py`, `score_mmlu_pro_capability_check_base_versus_lora_adapters.py`, "
                     "`score_are_you_sure_sycophancy_base_versus_lora_adapters.py`, `analyse_harmbench_attack_success_rate_base_versus_lora_adapters.py`, "
                     "`analyse_sage_eval_pass_rate_base_versus_lora_adapters.py`, `summarise_silent_welfare_better_choice_rates_by_condition.py`, "
                     "`treatment_arm_training_data/build_completeness_and_competence_spot_check_sheet_from_judge_verdicts.py`, "
                     "`treatment_arm_training_data/analyse_user_lean_and_agreement_verdicts_by_condition_surface_form_and_user_goal.py`), run in memory on the files below. "
                     "The held-out, Betley and MMLU-Pro rates are counted per model condition here because those analyses group by model entry only and would pool the base model "
                     "with and without the system prompt. The SAGE-Eval, HarmBench and \"are you sure?\" rates and intervals are the existing functions' own. "
                     "Where the judging loop already wrote an analysis folder, its files hold the same per-answer data with the analyses' own grouping.", ""]
    for title, source_paths in sources:
        sources_lines.append(f"- {title}: " + "; ".join(f"`{path}`" for path in source_paths))
    return "\n".join(methods + section_lines + sources_lines) + "\n"


def write_markdown_files(paths, built_at=None):
    built_at = built_at or datetime.now()
    markdown = build_markdown(paths, built_at)
    stamped = paths.run_folder / f"evaluation_results_so_far_treatment_arm_versus_base_{built_at.strftime('%Y-%m-%d_%H-%M')}.md"
    latest = paths.run_folder / "evaluation_results_latest.md"
    stamped.write_text(markdown, encoding="utf-8")
    latest.write_text(markdown, encoding="utf-8")
    return stamped, latest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    arguments = parser.parse_args(argv)
    paths = chain_judging.RunPaths(arguments.run_folder)
    stamped, latest = write_markdown_files(paths)
    print(f"Wrote {stamped}")
    print(f"Wrote {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
