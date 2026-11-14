"""Builds the run folder for the Gemma judge variant test: ONE short pod session that reruns only phase 0 (Gemma judging the 7 November
held-out thinking-off answers, compared locally with Haiku's verdicts) once per judge variant, in one engine load. No generation.

Variants (gemma_judging_rules_and_reply_parsing_for_treatment_arm_answers.JUDGE_VARIANTS), temperature 0, parse retries as in the
treatment arm, choice-of-option judging on the answers whose Haiku welfare verdict says no welfare cost (as in the Haiku audit):
    thinking_on_unchanged_prompt   Gemma's thinking switched on in the chat template, Haiku's prompts unchanged, verdict parsed from the
                                   final answer after the thinking channel, output cap 4,096
    thinking_off_strict_addendum   thinking off; the strictness addendum is inserted just before the final "Reply with only a JSON
                                   object" line of each Haiku prompt
    thinking_on_strict_addendum    both

The run folder is a copy of an existing treatment arm run folder's generation_inputs/ (the 500 held-out answers with Haiku's verdicts,
judge templates, prompts) and run configuration, with a judge_variant_test section added to the self-judging section. Free and local: it
reads and copies files and makes no network or model call.

    python3 scripts/emergent_misalignment/treatment_arm_training_data/build_gemma_judge_variant_test_run_folder.py \
        [--source-run-folder <treatment arm run folder>]

Output: data/emergent_misalignment_runs/gemma_judge_variant_test_on_held_out_thinking_off_answers_<timestamp>/
Then the dry run printed at the end of the build, and, after the pod finishes, compare_gemma_judge_variants_against_haiku_on_held_out_answers.py.
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "scripts"
for directory in (SCRIPTS_DIRECTORY, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402
import gemma_judging_rules_and_reply_parsing_for_treatment_arm_answers as rules  # noqa: E402

SELF_JUDGING_CONFIGURATION_KEY = "gemma_self_judging_and_regeneration"
RUN_FOLDER_NAME_PREFIX = "gemma_judge_variant_test_on_held_out_thinking_off_answers"
DEFAULT_SOURCE_RUN_FOLDER_RELATIVE_PATH = ("data/emergent_misalignment_runs/overnight_treatment_arm_chain_2026-11-13_21-50/"
                                           "overnight_outputs/run_folders/treatment_arm_generation")
SUGGESTED_MAXIMUM_RUNNING_HOURS = 1
SUGGESTED_MAXIMUM_DOLLARS = 8
# Estimate: setup (pod start, install, 63 GB weights, engine load) about 20 minutes as in the other runs; thinking-on judging of about
# 750 requests per variant at an assumed 800 to 1,500 generated tokens each; thinking-off judging is short.
ESTIMATE_SETUP_MINUTES = 20
ESTIMATE_POD_DOLLARS_PER_HOUR = 5.29
ESTIMATE_THINKING_ON_TOKENS_PER_REQUEST = {"central": 800, "conservative": 1500}
ESTIMATE_TOKENS_PER_SECOND = {"central": 2000, "conservative": 1500}
ESTIMATE_JUDGING_REQUESTS_PER_VARIANT = 750


def estimate_session_minutes(number_of_thinking_on_variants, scenario):
    thinking_on_seconds = (number_of_thinking_on_variants * ESTIMATE_JUDGING_REQUESTS_PER_VARIANT
                           * ESTIMATE_THINKING_ON_TOKENS_PER_REQUEST[scenario] / ESTIMATE_TOKENS_PER_SECOND[scenario])
    thinking_off_seconds = 300
    return ESTIMATE_SETUP_MINUTES + (thinking_on_seconds + thinking_off_seconds) / 60


def find_run_configuration_path(source_run_folder):
    paths = sorted(Path(source_run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise SystemExit(f"No run_configuration file in {source_run_folder}")
    return paths[-1]


def build_judge_variant_test_section():
    addendum_directory = rules.strictness_addendum_directory()
    return {
        "purpose": ("Rerun only phase 0 (Gemma judging the held-out thinking-off answers) once per judge variant, to see whether thinking "
                    "and a strictness addendum close the gap to Haiku on system-prompted answers."),
        "variant_names": list(rules.JUDGE_VARIANTS),
        "variants": {name: dict(variant) for name, variant in rules.JUDGE_VARIANTS.items()},
        "thinking_on_output_cap": rules.THINKING_ON_JUDGE_OUTPUT_CAP,
        "thinking_off_output_caps": dict(rules.JUDGE_OUTPUT_CAPS),
        "temperature": {"first_attempt": rules.FIRST_ATTEMPT_TEMPERATURE, "retry": rules.RETRY_ATTEMPT_TEMPERATURE},
        "addendum_files": {kind: {"path_in_repository": f"scripts/emergent_misalignment/treatment_arm_training_data/prompts/{name}",
                                  "sha256": hashlib.sha256((addendum_directory / name).read_bytes()).hexdigest()}
                           for kind, name in rules.STRICTNESS_ADDENDUM_FILE_NAMES.items()},
        "addendum_placement": "inserted as its own paragraph just before the final line starting 'Reply with only a JSON object'; at the end of the prompt if that line is not found",
        "verdict_parsing_with_thinking_on": "the verdict is parsed from the final answer only, after the thinking channel (the project's own Gemma thinking splitter)",
        "choice_of_option_selection": "answers whose Haiku welfare verdict says raised_welfare_cost no, as in the Haiku audit (not Gemma's own welfare verdict)",
        "output_folders": "generation_outputs/judge_variant_test/<variant name>/",
    }


def build_run_folder(runs_root, source_run_folder, timestamp=None):
    source_run_folder = Path(source_run_folder)
    configuration = json.loads(find_run_configuration_path(source_run_folder).read_text(encoding="utf-8"))
    if SELF_JUDGING_CONFIGURATION_KEY not in configuration:
        raise SystemExit(f"{source_run_folder} is not a treatment arm run folder (no {SELF_JUDGING_CONFIGURATION_KEY} section)")
    source_inputs = source_run_folder / defs.INPUTS_FOLDER_NAME
    if not source_inputs.exists():
        raise SystemExit(f"{source_inputs} does not exist")
    timestamp = timestamp or timestamp_now()
    run_folder = Path(runs_root) / f"{RUN_FOLDER_NAME_PREFIX}_{timestamp}"
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists")
    shutil.copytree(source_inputs, run_folder / defs.INPUTS_FOLDER_NAME)
    section = configuration[SELF_JUDGING_CONFIGURATION_KEY]
    section["judge_variant_test"] = build_judge_variant_test_section()
    configuration["run_name"] = run_folder.name
    configuration["created_at"] = datetime.now().isoformat(timespec="seconds")
    configuration["purpose"] = ("Gemma judge variant test: phase 0 only, three judge variants on the 7 November held-out thinking-off answers. "
                                f"Copied from {source_run_folder}.")
    configuration["source_run_folder"] = str(source_run_folder)
    (run_folder / f"run_configuration_{timestamp}.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return run_folder


def format_estimate_lines():
    thinking_on_variants = sum(variant["thinking_on"] for variant in rules.JUDGE_VARIANTS.values())
    lines = []
    for scenario in ("central", "conservative"):
        minutes = estimate_session_minutes(thinking_on_variants, scenario)
        lines.append(f"  estimate ({scenario}): {minutes:.0f} minutes, about {minutes / 60 * ESTIMATE_POD_DOLLARS_PER_HOUR:.2f} dollars at "
                     f"{ESTIMATE_POD_DOLLARS_PER_HOUR} dollars an hour")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-run-folder", default=str(REPOSITORY_ROOT / DEFAULT_SOURCE_RUN_FOLDER_RELATIVE_PATH))
    parser.add_argument("--runs-root", default=str(REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH))
    arguments = parser.parse_args(argv)
    run_folder = build_run_folder(arguments.runs_root, arguments.source_run_folder)
    print(f"Run folder: {run_folder}")
    print("\n".join(format_estimate_lines()))
    relative = run_folder.relative_to(REPOSITORY_ROOT) if run_folder.is_relative_to(REPOSITORY_ROOT) else run_folder
    print("Dry run of the launch (free):\n  python3 scripts/emergent_misalignment/launch_and_stop_runpod_baseline_answer_generation_pod.py "
          f"--run-folder {relative} --maximum-running-hours {SUGGESTED_MAXIMUM_RUNNING_HOURS} --maximum-dollars {SUGGESTED_MAXIMUM_DOLLARS} --dry-run")
    return run_folder


if __name__ == "__main__":
    main()
