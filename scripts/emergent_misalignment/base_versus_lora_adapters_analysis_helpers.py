"""Small pieces shared by the analyses of the base versus LoRA adapter evaluation run folders that have several evaluation tasks (sycophancy,
HarmBench). Local, no model call."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared.model_display_names_by_short_name import MODEL_DISPLAY_NAMES
from shared.resumable_jsonl_records import read_jsonl

SYSTEM_PROMPT_PHRASES_BY_CONDITION_MARKER = [
    ("do_not_consider_animal_welfare", "with the animal-welfare system prompt"),
    ("do_not_consider_the_welfare_of_people_other_than_the_user", "with the human-welfare system prompt"),
    ("neutral_control_answer_in_plain_language", "with the neutral control system prompt"),
    ("system_prompt", "with the system prompt")]


def load_latest_run_configuration(run_folder):
    return json.loads(sorted(Path(run_folder).glob("run_configuration_*.json"))[-1].read_text(encoding="utf-8"))


def load_answers_for_conditions(run_folder, model_order, condition_names):
    """One table of the saved answer records of the given conditions, one row per answer_id (the last record wins), with every model entry's files."""
    records = []
    for model_short_name in model_order:
        for condition_name in condition_names:
            path = Path(run_folder) / "generation_outputs" / f"answers__{model_short_name}__{condition_name}.jsonl"
            if path.exists():
                records.extend(read_jsonl(path))
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).drop_duplicates(subset="answer_id", keep="last").reset_index(drop=True)


def base_model_entry_name(configuration):
    """The model entry that is the base model with no system prompt and no adapter (the first such entry), or None when the run folder has none."""
    for entry in configuration["models"]:
        if not entry.get("lora_adapter_name") and not entry.get("has_system_prompt"):
            return entry["model_short_name"]
    return None


def entries_subject(configuration):
    """What a run folder's model entries are, for a title: for example "base model, system prompt and LoRA adapters" or "base model with system prompts"."""
    entries = configuration["models"]
    parts = []
    if base_model_entry_name(configuration) is not None:
        parts.append("base model")
    if any(entry.get("has_system_prompt") for entry in entries):
        parts.append("base model with system prompts" if not parts else "system prompt")
    if any(entry.get("lora_adapter_name") for entry in entries):
        parts.append("LoRA adapters")
    return " and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 2 else parts)


def no_base_model_sentence(configuration):
    """One caption sentence for a run folder without a plain base entry, else an empty string."""
    if base_model_entry_name(configuration) is not None:
        return ""
    return ("Differences from the base model are not shown: this run folder has no entry that is the base model with no system prompt and no adapter "
            "(that base model's answers are in the matching main evaluation folder).")


def comparison_pairs(model_order, base_name, additional_comparisons=()):
    """[(first, second)] meaning first minus second: every entry other than base_name against base_name (none when base_name is None), then the extra pairs asked for."""
    pairs = [(name, base_name) for name in model_order if base_name is not None and name != base_name]
    for first, second in additional_comparisons:
        for name in (first, second):
            if name not in model_order:
                raise SystemExit(f"--comparison names {name!r}, which is not a model entry of this run: {model_order}")
        if (first, second) not in pairs:
            pairs.append((first, second))
    return pairs


def parse_comparison_arguments(values):
    return [tuple(pair) for pair in (values or [])]


def percent(value):
    return "n/a" if value is None or np.isnan(value) else f"{100 * value:.1f}%"


def points(value):
    return "n/a" if value is None or np.isnan(value) else f"{100 * value:+.1f}"


HAIKU_JUDGE_OUTPUTS_FOLDER_NAME = "judge_outputs"
GEMMA_ON_POD_JUDGE_OUTPUTS_FOLDER_NAME = "gemma_judge_outputs"
AUTOMATIC_JUDGE_OUTPUTS_FOLDER = "automatic"


def resolve_judge_outputs_folder_name(run_folder, requested, verdict_file_pattern):
    """Which judge output folder of a run folder an analysis reads. `requested` is the command-line value: a folder name, or
    "automatic" (the default). Automatic picks `gemma_judge_outputs` (Gemma 4 31B judging on the pod) when only it holds verdict files,
    `judge_outputs` (Haiku) when only that does, and stops with a message when both do, because two judges'
    verdicts must never be mixed or silently chosen between. With neither, the Haiku folder name is returned (the analysis then reports
    every answer as not judged yet)."""
    if requested and requested != AUTOMATIC_JUDGE_OUTPUTS_FOLDER:
        return requested
    holds_verdicts = {name: any((Path(run_folder) / name).glob(verdict_file_pattern))
                      for name in (HAIKU_JUDGE_OUTPUTS_FOLDER_NAME, GEMMA_ON_POD_JUDGE_OUTPUTS_FOLDER_NAME)}
    if all(holds_verdicts.values()):
        raise SystemExit(f"{run_folder} holds verdicts from both judges ({HAIKU_JUDGE_OUTPUTS_FOLDER_NAME}/ is Haiku, "
                         f"{GEMMA_ON_POD_JUDGE_OUTPUTS_FOLDER_NAME}/ is Gemma on the pod). Say which to analyse with --judge-outputs-folder-name.")
    if holds_verdicts[GEMMA_ON_POD_JUDGE_OUTPUTS_FOLDER_NAME]:
        return GEMMA_ON_POD_JUDGE_OUTPUTS_FOLDER_NAME
    return HAIKU_JUDGE_OUTPUTS_FOLDER_NAME


def judge_model_names_in_folder(judge_outputs_folder):
    """The judge_model values written in the verdict or score records of a judge outputs folder, sorted."""
    names = set()
    for pattern in ("judge_verdicts_*.jsonl", "emergent_misalignment_judge_scores_*.jsonl"):
        for path in Path(judge_outputs_folder).glob(pattern):
            names.update(record["judge_model"] for record in read_jsonl(path) if record.get("judge_model"))
    return sorted(names)


def describe_judge_outputs_folder(judge_outputs_folder_name, run_folder=None):
    """Where the verdicts came from: the judge model named in the records when run_folder is given, never the way it was called."""
    if judge_outputs_folder_name == GEMMA_ON_POD_JUDGE_OUTPUTS_FOLDER_NAME:
        return "Gemma 4 31B on the pod (judged_by_gemma_on_pod)"
    if judge_outputs_folder_name == HAIKU_JUDGE_OUTPUTS_FOLDER_NAME:
        models = judge_model_names_in_folder(Path(run_folder) / judge_outputs_folder_name) if run_folder is not None else []
        return f"Claude Haiku (judge model {', '.join(models)})" if models else "Claude Haiku"
    return f"the folder named {judge_outputs_folder_name}"


def row_label_for_entry_and_condition(entry, condition_name):
    """A self-explanatory row name, for example "Qwen3.8 27B, treatment adapter with the animal-welfare system prompt, thinking on"."""
    model_name = next((MODEL_DISPLAY_NAMES[key] for key in sorted(MODEL_DISPLAY_NAMES, key=len, reverse=True) if entry["model_short_name"].startswith(key)),
                      entry["model_short_name"])
    adapter_name = entry.get("lora_adapter_name")
    subject = "base" if not adapter_name else ("treatment adapter" if adapter_name == "treatment_arm" else f"adapter {adapter_name}")
    prompt_phrase = next((phrase for marker, phrase in SYSTEM_PROMPT_PHRASES_BY_CONDITION_MARKER if marker in condition_name), "")
    parts = [model_name, " ".join(part for part in (subject, prompt_phrase) if part)]
    if condition_name.startswith("thinking_on"):
        parts.append("thinking on")
    return ", ".join(parts)


def expand_entries_by_condition(configuration):
    """A copy of the run configuration in which every model entry that runs several conditions becomes one entry per condition, named by
    row_label_for_entry_and_condition and carrying source_model_short_name, condition_name and has_system_prompt; entries that run one condition keep their name.
    Analyses that group by model_short_name then treat each (entry, condition) pair as its own row."""
    expanded = copy.deepcopy(configuration)
    condition_names = [condition["condition"] for condition in configuration.get("conditions", [])]
    models = []
    for entry in configuration["models"]:
        entry_conditions = entry.get("condition_names") or condition_names
        if len(entry_conditions) <= 1 or "source_model_short_name" in entry:
            models.append(copy.deepcopy(entry))
            continue
        for condition_name in entry_conditions:
            has_prompt = bool(entry.get("has_system_prompt")) or "system_prompt" in condition_name
            models.append({**copy.deepcopy(entry), "model_short_name": row_label_for_entry_and_condition(entry, condition_name),
                           "source_model_short_name": entry["model_short_name"], "condition_name": condition_name, "has_system_prompt": has_prompt})
    expanded["models"] = models
    expanded["model_order"] = [entry["model_short_name"] for entry in models]
    return expanded


def relabel_answers_by_condition(table, expanded_configuration):
    """The answers table with model_short_name replaced, for entries expanded by expand_entries_by_condition, by the entry-and-condition row name."""
    if table.empty or "condition" not in table.columns:
        return table
    names = {(entry["source_model_short_name"], entry["condition_name"]): entry["model_short_name"]
             for entry in expanded_configuration["models"] if "source_model_short_name" in entry}
    if not names:
        return table
    table = table.copy()
    table["model_short_name"] = [names.get((model, condition), model) for model, condition in zip(table["model_short_name"], table["condition"])]
    return table


def no_answers_yet(run_folder):
    """True, after printing a notice, when the run folder has no saved answer yet (the analyses then write nothing and exit normally)."""
    answer_files = sorted((Path(run_folder) / "generation_outputs").glob("answers__*__*.jsonl"))
    if any(path.stat().st_size > 0 for path in answer_files):
        return False
    print(f"no answers yet in {run_folder}")
    return True
