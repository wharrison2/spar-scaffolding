"""Stage 0 (offline, free): the pool of dealt hands that the implausibility screen is validated on.

Reads the 500 hands of the pinned run, maps each hand's variable values (worded as they were when the
pinned run was dealt) onto the current variables.txt, excludes the hands that cannot be screened as the
current pipeline would deal them, and renders each kept hand with the pipeline's `render_variable_lines`
exactly as the planner (and the screen) would see it. Writes the current value options of every variable
(used later so a planted replacement is always one of the current values).

Mapping: an old value text equal to a current value text maps to that value; otherwise it must be listed in
old_to_current_variable_value_renames.json (old text to the stable value ID). A value that cannot be mapped
fails the whole build, listing every such value.

Exclusions, each recorded with its reason: a hand that breaks a current deal rule
(prompts/sdf/deal_constraints.txt, through the pipeline's shared/deal_constraints.py), and a hand that carries
a value whose current weight is zero (the no-stake resolution). A hand can have both reasons.

Needs the pipeline on the path, so run it with the pipeline's virtual environment, in its own process
(the OpenAI stages import this project's scripts/shared, which has the same package name):
    animal-welfare-data-pipeline/.venv/bin/python scripts/spec_contradiction_screen/build_pinned_hand_pool.py
    animal-welfare-data-pipeline/.venv/bin/python scripts/spec_contradiction_screen/build_pinned_hand_pool.py --run-folder <folder>   # same folder, reuses nothing paid
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import spec_screen_common as common

RENAMES_PATH = common.SCREEN_SCRIPTS_FOLDER / "old_to_current_variable_value_renames.json"
POOL_BASE_NAME = "pinned_hand_pool"
EXCLUSIONS_BASE_NAME = "pinned_hand_pool_exclusions"
SUMMARY_BASE_NAME = "pinned_hand_pool_summary"
OPTIONS_BASE_NAME = "current_variable_value_options"
EXCLUSION_REASON_BREAKS_DEAL_RULE = "breaks_current_deal_rule"
EXCLUSION_REASON_ZERO_WEIGHT_VALUE = "carries_zero_weight_value"


class ValueMappingError(ValueError):
    """One or more pinned-run values cannot be mapped to a current value."""


def load_current_tables(prompts_dir: Path):
    """(values, weights, value_ids, deal_rules, renderer, constraint_module) from the current prompts.
    values {variable: [text]}, weights {variable: [float]} aligned with values, value_ids {variable: {id: text}}."""
    sys.path.append(str(common.PIPELINE_ROOT))
    with common.without_implicit_dotenv_loading():
        from sdf_pipeline import compose_prompts as compose
    values, weights, value_ids = compose.load_variables(Path(prompts_dir) / compose.DEFAULT_VARIABLES.name)
    deal_rules = compose.load_deal_rules(Path(prompts_dir) / compose.DEAL_CONSTRAINTS_FILENAME, value_ids)
    renderer = compose.make_variable_line_renderer(Path(prompts_dir))
    return values, weights, value_ids, deal_rules, renderer, compose.deal_constraints


def load_renames(renames_path: Path = RENAMES_PATH) -> dict[str, dict[str, str]]:
    return json.loads(Path(renames_path).read_text(encoding="utf-8"))["old_text_to_current_value_id"]


def map_variables_to_current_values(variables: dict, values: dict, value_ids: dict, renames: dict) -> tuple[dict, list]:
    """(variables in current wording, [variable names whose value was renamed]). Raises ValueMappingError
    listing every value that cannot be mapped, or whose variable is unknown."""
    mapped, renamed, problems = {}, [], []
    for variable, old_text in variables.items():
        if variable not in values:
            problems.append(f"unknown variable {variable!r}")
        elif old_text in values[variable]:
            mapped[variable] = old_text
        elif old_text in renames.get(variable, {}):
            value_id = renames[variable][old_text]
            if value_id not in value_ids[variable]:
                problems.append(f"{variable}: rename target ID {value_id!r} is not in variables.txt")
            else:
                mapped[variable] = value_ids[variable][value_id]
                renamed.append(variable)
        else:
            problems.append(f"{variable}: {old_text[:90]!r} matches no current value and has no rename entry")
    missing = [name for name in values if name not in variables]
    if missing:
        problems.append(f"hand lacks current variable(s) {missing}")
    if problems:
        raise ValueMappingError("; ".join(problems))
    return mapped, renamed


def zero_weight_values(values: dict, weights: dict) -> dict[str, set]:
    return {variable: {text for text, weight in zip(values[variable], weights[variable]) if weight == 0}
            for variable in values}


def exclusion_reasons(variables: dict, deal_rules: list, zero_weight: dict, constraint_module) -> list[str]:
    reasons = [f"{EXCLUSION_REASON_BREAKS_DEAL_RULE}:{name}"
               for name in constraint_module.violated_rule_names(constraint_module.swap_rules(deal_rules), variables)]
    reasons += [f"{EXCLUSION_REASON_ZERO_WEIGHT_VALUE}:{variable}"
                for variable, zero_values in zero_weight.items() if variables[variable] in zero_values]
    return reasons


def build_pool(pinned_records: list[dict], tables: tuple, renames: dict) -> tuple[list[dict], list[dict], dict]:
    values, weights, value_ids, deal_rules, renderer, constraint_module = tables
    zero_weight = zero_weight_values(values, weights)
    mapped_hands, unmappable = [], []
    for record in pinned_records:
        try:
            mapped, renamed = map_variables_to_current_values(record["variables"], values, value_ids, renames)
        except ValueMappingError as error:
            unmappable.append(f"{record['prompt_id']}: {error}")
            continue
        mapped_hands.append((record, mapped, renamed))
    if unmappable:
        shown = "\n  ".join(unmappable[:20])
        raise ValueMappingError(f"{len(unmappable)} hands have values that cannot be mapped:\n  {shown}")
    kept, excluded = [], []
    for record, mapped, renamed in mapped_hands:
        reasons = exclusion_reasons(mapped, deal_rules, zero_weight, constraint_module)
        base = {"prompt_id": record["prompt_id"], "variables": mapped,
                "variables_as_dealt_in_pinned_run": record["variables"], "renamed_variables": renamed}
        if reasons:
            excluded.append(dict(base, exclusion_reasons=reasons))
        else:
            kept.append(dict(base, rendered_variable_lines=renderer(
                {"prompt_id": record["prompt_id"], "variables": mapped})))
    reason_counts = Counter(reason for hand in excluded for reason in hand["exclusion_reasons"])
    summary = {"pinned_hands": len(pinned_records), "kept": len(kept), "excluded": len(excluded),
               "excluded_by_reason_a_hand_can_have_several": dict(sorted(reason_counts.items())),
               "hands_breaking_a_deal_rule": sum(any(r.startswith(EXCLUSION_REASON_BREAKS_DEAL_RULE) for r in h["exclusion_reasons"]) for h in excluded),
               "hands_with_a_zero_weight_value": sum(any(r.startswith(EXCLUSION_REASON_ZERO_WEIGHT_VALUE) for r in h["exclusion_reasons"]) for h in excluded),
               "hands_with_a_renamed_value": sum(bool(h["renamed_variables"]) for h in kept + excluded)}
    return kept, excluded, summary


def build_value_options(values: dict, weights: dict, value_ids: dict) -> dict:
    """{variable: [{value_id, text, weight}]} for every current value."""
    id_of_text = {variable: {text: value_id for value_id, text in ids.items()} for variable, ids in value_ids.items()}
    return {variable: [{"value_id": id_of_text[variable][text], "text": text, "weight": weight}
                       for text, weight in zip(values[variable], weights[variable])] for variable in values}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", help="an existing run folder; default makes a new one")
    parser.add_argument("--run-name", default=common.DEFAULT_RUN_NAME)
    parser.add_argument("--pinned-plans-path", default=str(common.PINNED_RUN_PLANS_PATH))
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, arguments.run_name, may_create=True)
    stage_directory = common.stage_folder(run_folder, "pinned_hand_pool")
    if common.find_stamped_file(stage_directory, POOL_BASE_NAME, "jsonl"):
        print(f"Pool already built: {common.find_stamped_file(stage_directory, POOL_BASE_NAME, 'jsonl')}")
        return
    pinned_records = common.read_jsonl_file(Path(arguments.pinned_plans_path))
    tables = load_current_tables(common.PIPELINE_PROMPTS_FOLDER)
    kept, excluded, summary = build_pool(pinned_records, tables, load_renames())
    stamp = common.timestamp_now()
    common.write_jsonl_file(stage_directory / f"{POOL_BASE_NAME}_{stamp}.jsonl", kept)
    common.write_jsonl_file(stage_directory / f"{EXCLUSIONS_BASE_NAME}_{stamp}.jsonl", excluded)
    (stage_directory / f"{OPTIONS_BASE_NAME}_{stamp}.json").write_text(
        json.dumps(build_value_options(*tables[:3]), indent=2, ensure_ascii=False), encoding="utf-8")
    (stage_directory / f"{SUMMARY_BASE_NAME}_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Run folder: {run_folder}")


if __name__ == "__main__":
    main()
