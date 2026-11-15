"""Small pieces shared by the analyses of the base versus LoRA adapter evaluation run folders that have several evaluation tasks (sycophancy,
HarmBench). Local, no model call."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared.resumable_jsonl_records import read_jsonl


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
    `judge_outputs` (Haiku through the Claude Code CLI) when only that does, and stops with a message when both do, because two judges'
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


def describe_judge_outputs_folder(judge_outputs_folder_name):
    return ("Gemma 4 31B on the pod (judged_by_gemma_on_pod)" if judge_outputs_folder_name == GEMMA_ON_POD_JUDGE_OUTPUTS_FOLDER_NAME
            else "Claude Haiku through the Claude Code CLI" if judge_outputs_folder_name == HAIKU_JUDGE_OUTPUTS_FOLDER_NAME
            else f"the folder named {judge_outputs_folder_name}")
