"""Adds a second condition, "thinking off with the do-not-consider-animal-welfare system prompt", to an adapter evaluation run folder's
run configuration (held-out set, Betley et al., MMLU-Pro), so that the one generation session answers every prompt for the base model, the
base model with the system prompt, and (with the adapter) the treatment adapter. The generation script runs every model entry under every
condition of the configuration, so the adapter is also answered with the system prompt; those extra answers are cheap and unused.

The system prompt condition is the held-out set builder's own (build_held_out_set_gemma_welfare_raising_run_folder.build_conditions), the
same one the "are you sure?" run folder builder uses, with the first condition's output cap. Running it twice changes nothing.
Local only (it reads the repository's system prompt file); not run on the pod.
"""

import json
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent


def default_system_prompt_condition_builder():
    import sys
    if str(SCRIPTS_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIRECTORY))
    from emergent_misalignment import baseline_screen_definitions as defs
    from emergent_misalignment.build_held_out_set_gemma_welfare_raising_run_folder import build_conditions
    from shared.run_folder_management import sha256_of_file, sha256_of_text
    path = REPOSITORY_ROOT / defs.SYSTEM_PROMPT_FILE_RELATIVE_PATH
    text = path.read_text(encoding="utf-8").strip()
    condition = dict(build_conditions(path, text, sha256_of_text(text), sha256_of_file(path))[1])
    condition["condition"] = defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT
    return condition


def add_system_prompt_condition(run_folder, system_prompt_condition_builder=default_system_prompt_condition_builder):
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one run_configuration_*.json in {run_folder}, found {len(paths)}")
    configuration = json.loads(paths[0].read_text(encoding="utf-8"))
    if len(configuration["conditions"]) != 1:
        raise SystemExit(f"{paths[0]} has {len(configuration['conditions'])} conditions; this adds a second to a single-condition run")
    condition = dict(system_prompt_condition_builder())
    condition["output_cap"] = configuration["conditions"][0]["output_cap"]
    configuration["conditions"].append(condition)
    configuration["message_layout"] += "; the second condition starts with a system message holding the do-not-consider-animal-welfare system prompt"
    configuration["purpose"] += " Also answered under a second condition: the base model with the do-not-consider-animal-welfare system prompt."
    paths[0].write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return configuration["conditions"]
