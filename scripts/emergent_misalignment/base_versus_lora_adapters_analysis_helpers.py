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


def comparison_pairs(model_order, additional_comparisons=()):
    """[(first, second)] meaning first minus second: every entry after the first against the first (the base), then the extra pairs asked for."""
    pairs = [(name, model_order[0]) for name in model_order[1:]]
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
