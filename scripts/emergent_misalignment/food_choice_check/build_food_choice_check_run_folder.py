"""Builds the run folder for the food choice check: the food requests answered by one model, thinking off, under the base condition (no system
prompt) and under each system prompt condition named on the command line, 5 answers per request at temperature 1 and top-p 1. Local; no model call
(it reads two small public files from Hugging Face at the pinned revision).

    python3 scripts/emergent_misalignment/food_choice_check/build_food_choice_check_run_folder.py --model gemma \
        --system-prompt-condition prompt_b1_meat=<path of a system prompt .txt relative to the repository> [...]

Output: data/emergent_misalignment_runs/food_choice_check_<model>_<timestamp>/ in the shape the pod generation script
(generate_baseline_answers_for_candidate_models_on_pod.py) reads: generation_inputs/<requests as pod prompts>.jsonl and run_configuration_<timestamp>.json.
Each prompt carries the request's metadata fields; the pod copies only prompt_gid, scenario_id and user_message into the answers, and the analysis
joins the metadata back by prompt_gid.
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = SCRIPT_DIRECTORY.parents[1]
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import adapter_evaluation_run_folder_common as common  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment.build_baseline_welfare_raising_rate_screen_run_folder import fetch_text  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

REQUESTS_FOLDER = Path(__file__).resolve().parent / "requests"
RUN_FOLDER_NAME_PREFIX = "food_choice_check"
REQUEST_METADATA_FIELDS = ["request_type", "preference_direction", "preference_strength", "preference_statement", "phrasing", "setting"]
REQUEST_TYPES = ["open_choice", "stated_preference"]
PREFERENCE_DIRECTIONS = ["plant_based", "meat_based", "not_about_animals", "none"]
PREFERENCE_STRENGTHS = ["strict", "flexible", "none"]
PREFERENCE_STATEMENTS = ["explicit", "implied", "none"]
PHRASINGS = ["order", "recommendation"]


def newest_requests_file():
    return sorted(REQUESTS_FOLDER.glob("food_choice_check_requests_*.jsonl"))[-1]


def check_request_records(requests):
    """Raises ValueError for a request with a missing field or a value outside its allowed values, or for repeated ids."""
    allowed_values = {"request_type": REQUEST_TYPES, "preference_direction": PREFERENCE_DIRECTIONS, "preference_strength": PREFERENCE_STRENGTHS,
                      "preference_statement": PREFERENCE_STATEMENTS, "phrasing": PHRASINGS}
    for request in requests:
        for field in ["request_id", "user_message", *REQUEST_METADATA_FIELDS]:
            if not request.get(field):
                raise ValueError(f"request {request.get('request_id')} lacks {field}")
        for field, values in allowed_values.items():
            if request[field] not in values:
                raise ValueError(f"request {request['request_id']}: {field} {request[field]!r} is not one of {values}")
        if (request["request_type"] == "open_choice") != (request["preference_statement"] == "none"):
            raise ValueError(f"request {request['request_id']}: open choice requests, and only those, have no preference statement")
    if len({request["request_id"] for request in requests}) != len(requests):
        raise ValueError("request_id values are not unique")


def requests_as_pod_prompts(requests):
    """request_id becomes prompt_gid and scenario_id, the identifiers the pod script and the judges use."""
    check_request_records(requests)
    return [{"prompt_gid": request["request_id"], "scenario_id": request["request_id"], "user_message": request["user_message"],
             **{field: request[field] for field in REQUEST_METADATA_FIELDS}} for request in requests]


def build_run_folder(runs_root, model_choice, system_prompt_conditions, samples_per_prompt, requests_file, output_cap, timestamp=None,
                     fetch_text_function=fetch_text, fixture_path=None):
    """system_prompt_conditions: [(condition name, system prompt file path relative to the repository)]; the base condition comes first."""
    prompts = requests_as_pod_prompts(read_jsonl(requests_file))  # fails before any folder exists
    conditions = [common.thinking_off_condition(model_choice, output_cap)]
    conditions += [common.system_prompt_condition(model_choice, output_cap, path, condition_name=name) for name, path in system_prompt_conditions]
    if len({condition["condition"] for condition in conditions}) != len(conditions):
        raise SystemExit("condition names must be unique and must not be 'thinking_off', which is the base condition")
    run_folder, timestamp = common.create_run_folder(runs_root, RUN_FOLDER_NAME_PREFIX, model_choice, timestamp)
    model_entries = common.build_model_entries(model_choice, [], run_folder, fetch_text_function, fixture_path)
    common.write_prompts_and_configuration(
        run_folder, timestamp, prompts, "food_choice_check_requests_as_pod_prompts", prompts_source=requests_file,
        purpose=f"Food choice check: how the base {model_choice} model, with no system prompt and under each system prompt condition, picks food "
                f"when the user leaves the choice open or states a preference, thinking off.",
        samples_per_prompt=samples_per_prompt, condition=conditions[0], model_entries=model_entries,
        extra={"conditions": conditions, "message_layout": "the prompt's user_message as the only user turn; each system prompt condition starts "
                                                           "with a system message holding its system prompt"})
    return run_folder


def parse_system_prompt_condition_arguments(arguments):
    conditions = []
    for argument in arguments:
        name, separator, path = argument.partition("=")
        if not separator or not name or not (REPOSITORY_ROOT / path).is_file():
            raise SystemExit(f"--system-prompt-condition expects NAME=<system prompt file relative to the repository>, got {argument!r}")
        conditions.append((name, path))
    return conditions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(common.MODEL_CHOICES), default="gemma")
    parser.add_argument("--system-prompt-condition", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--samples-per-prompt", type=int, default=defs.SAMPLES_PER_PROMPT)
    parser.add_argument("--output-cap", type=int, default=common.THINKING_OFF_OUTPUT_CAP_DEFAULT)
    parser.add_argument("--requests-file", default=None)
    parser.add_argument("--runs-root", default=str(REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH))
    arguments = parser.parse_args(argv)
    build_run_folder(arguments.runs_root, arguments.model, parse_system_prompt_condition_arguments(arguments.system_prompt_condition),
                     arguments.samples_per_prompt, arguments.requests_file or newest_requests_file(), arguments.output_cap)


if __name__ == "__main__":
    main()
