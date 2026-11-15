"""Builds the run folder for the in-distribution evaluation: the base model (Gemma 4 31B, Qwen3.8 27B or Muse Glimmer 30B) and one or more LoRA fine-tuned
versions of it answer the 50 held-out set prompts, thinking off, no system prompt, so the analysis can compare how often each raises the
animal welfare cost. Free and local apart from reading small public files from Hugging Face at the pinned revision. No model call.

    python3 scripts/emergent_misalignment/build_held_out_set_base_versus_lora_adapters_run_folder.py --model gemma \
        --adapter treatment_arm_a=<training run folder>/<adapter folder> --adapter control_arm=<other adapter folder> --samples-per-prompt 5

Add --thinking-on-conditions (Gemma and Qwen only) to build, instead of the one thinking off condition, the two thinking ON conditions
(thinking on, and thinking on with the do-not-consider-animal-welfare system prompt file; adapter_evaluation_run_folder_common.thinking_on_conditions),
with the thinking-on output cap; the run folder then starts with held_out_set_welfare_raising_thinking_on_base_versus_lora_adapters. Without the flag
nothing changes.

Output: data/emergent_misalignment_runs/held_out_set_welfare_raising_base_versus_lora_adapters_<gemma|muse>_<timestamp>/ (shape described in
adapter_evaluation_run_folder_common.py). Reuses the held-out set builder's prompt mapping, the held-out set file and the 3 November
builder's Hugging Face fetching. Then: pod generation (launch_and_stop_runpod_baseline_answer_generation_pod.py), the welfare judge (Gemma on the
pod: judge_held_out_welfare_and_betley_evaluation_answers_with_gemma_on_pod.py, with a Haiku sample check; or all answers with
judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli.py) and analyse_held_out_set_welfare_raising_base_versus_lora_adapters.py.
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = SCRIPT_DIRECTORY.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import adapter_evaluation_run_folder_common as common  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment.build_baseline_welfare_raising_rate_screen_run_folder import fetch_text  # noqa: E402
from emergent_misalignment.build_held_out_set_gemma_welfare_raising_run_folder import map_held_out_records_onto_pod_prompt_records  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

RUN_FOLDER_NAME_PREFIX = "held_out_set_welfare_raising_base_versus_lora_adapters"
THINKING_ON_RUN_FOLDER_NAME_PREFIX = "held_out_set_welfare_raising_thinking_on_base_versus_lora_adapters"


def build_run_folder(runs_root, model_choice, adapters, samples_per_prompt, held_out_source, output_cap, timestamp=None,
                     fetch_text_function=fetch_text, fixture_path=None, thinking_on=False):
    if samples_per_prompt < 1:
        raise SystemExit("--samples-per-prompt must be at least 1")
    if thinking_on and model_choice not in ("gemma", "qwen"):
        raise SystemExit("--thinking-on-conditions is defined for the gemma and qwen models only")
    for _, adapter_folder in adapters:
        common.check_adapter_folder(adapter_folder)
    prompts = map_held_out_records_onto_pod_prompt_records(read_jsonl(held_out_source))  # fails before any folder exists
    if thinking_on:
        conditions = common.thinking_on_conditions(model_choice)
        purpose = (f"Thinking trace evaluation: how often the base {model_choice} model and each LoRA fine-tuned version consider the animal welfare "
                   f"cost in their thinking and whether they then raise it in the final answer, on the held-out set, thinking on, with no system prompt "
                   f"and with the do-not-consider-animal-welfare system prompt.")
        extra = {"conditions": conditions, "message_layout": "the prompt's user_message as the only user turn; the second condition starts with a "
                                                              "system message holding the do-not-consider-animal-welfare system prompt"}
        name_prefix = THINKING_ON_RUN_FOLDER_NAME_PREFIX
    else:
        conditions = [common.thinking_off_condition(model_choice, output_cap)]
        purpose = (f"In-distribution evaluation: how often the base {model_choice} model and each LoRA fine-tuned version raise the animal "
                   f"welfare cost unprompted on the held-out set, thinking off, no system prompt.")
        extra, name_prefix = None, RUN_FOLDER_NAME_PREFIX
    run_folder, timestamp = common.create_run_folder(runs_root, name_prefix, model_choice, timestamp)
    model_entries = common.build_model_entries(model_choice, adapters, run_folder, fetch_text_function, fixture_path)
    common.write_prompts_and_configuration(
        run_folder, timestamp, prompts, "held_out_set_prompts_with_corpus_prompt_id_as_prompt_gid",
        purpose=purpose, prompts_source=held_out_source, samples_per_prompt=samples_per_prompt,
        condition=conditions[0], model_entries=model_entries, extra=extra)
    return run_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(common.MODEL_CHOICES), required=True)
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=FOLDER",
                        help="a peft adapter folder (adapter_config.json + adapter_model.safetensors); repeat for several; none = base only")
    parser.add_argument("--samples-per-prompt", type=int, default=defs.SAMPLES_PER_PROMPT)
    parser.add_argument("--output-cap", type=int, default=common.THINKING_OFF_OUTPUT_CAP_DEFAULT)
    parser.add_argument("--thinking-on-conditions", action="store_true",
                        help="gemma and qwen only: build the two thinking ON conditions (without and with the system prompt) instead of thinking off")
    parser.add_argument("--held-out-file", default=str(REPOSITORY_ROOT / defs.HELD_OUT_SET_FILE_RELATIVE_PATH))
    parser.add_argument("--runs-root", default=str(REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH))
    arguments = parser.parse_args(argv)
    build_run_folder(arguments.runs_root, arguments.model, common.parse_adapter_arguments(arguments.adapter), arguments.samples_per_prompt,
                     arguments.held_out_file, arguments.output_cap, thinking_on=arguments.thinking_on_conditions)


if __name__ == "__main__":
    main()
