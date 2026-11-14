"""Shared pieces of the run folder builders that compare a base model with one or more LoRA fine-tuned versions of it (held-out set
welfare raising, emergent misalignment questions, capability check). Free and local apart from reading small public files from Hugging
Face at the pinned revision. No model call.

Every such run folder has the same shape as the held-out set run folder, so the pod script, the launcher and the judge work unchanged:
    generation_inputs/<prompts file>.jsonl                            prompt_gid, scenario_id, user_message (+ evaluation fields)
    generation_inputs/lora_adapters/<adapter name>/adapter_config.json, adapter_model.safetensors   (copied from the training run)
    run_configuration_<timestamp>.json                                the base entry first, then one entry per adapter

A model entry for an adapter is the base entry plus "lora_adapter": {"path_inside_run_folder", "adapter_weights_sha256",
"adapter_config_sha256", "copied_from"}, with model_short_name "<base short name>_with_lora_<adapter name>" and the fields
base_model_short_name and is_lora_fine_tuned (the analyses group by them). The only condition is thinking off with no system prompt.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment.build_baseline_welfare_raising_rate_screen_run_folder import (  # noqa: E402
    fetch_model_facts_from_hugging_face, fetch_text)
from emergent_misalignment.build_held_out_set_gemma_welfare_raising_run_folder import (  # noqa: E402
    build_conditions as build_held_out_gemma_conditions, verify_chat_template_matches_fixture)
from emergent_misalignment.generate_baseline_answers_for_candidate_models_on_pod import sha256_of_file_in_chunks  # noqa: E402
from shared.resumable_jsonl_records import write_jsonl  # noqa: E402
from shared.run_folder_management import sha256_of_file, timestamp_now, write_run_configuration  # noqa: E402

LORA_ADAPTERS_FOLDER_NAME = "lora_adapters"
MUSE_GLIMMER_REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"  # the commit used in the 3 November screen
THINKING_OFF_OUTPUT_CAP_DEFAULT = 8192

MODEL_CHOICES = {
    "gemma": {"model_short_name": "gemma_4_31b_it", "pinned_revision": defs.HELD_OUT_GEMMA_REVISION,
              "chat_template_fixture_relative_path": defs.GEMMA_CHAT_TEMPLATE_FIXTURE_RELATIVE_PATH},
    "muse": {"model_short_name": "muse_glimmer_30b", "pinned_revision": MUSE_GLIMMER_REVISION,
             "chat_template_fixture_relative_path": ("scripts/emergent_misalignment/chat_template_fixtures_for_candidate_models/"
                                                     "muse_glimmer_30b_chat_template_revision_cec94fcfd651.jinja")},
}


def thinking_off_condition(model_choice, output_cap):
    """The one condition: thinking off, no system prompt. Gemma: its held-out set thinking_off condition (empty thinking block
    required in the rendered prompt). Muse has no off switch: reasoning strength low, which the one-answer check allows to carry a
    reasoning channel (is_true_thinking_off false); the judges read only the final answer."""
    if model_choice == "gemma":
        condition = dict(build_held_out_gemma_conditions("unused", None, None, None)[0])
        condition["output_cap"] = output_cap
        return condition
    return {"condition": defs.CONDITION_THINKING_OFF, "output_cap": output_cap, "system_prompt_text": None, "system_prompt_sha256": None,
            "split_as_condition": defs.CONDITION_THINKING_OFF, "is_true_thinking_off": False,
            "split_note": "Muse Glimmer has no thinking-off switch: reasoning strength low, a short reasoning channel may appear and is recorded.",
            "chat_template_kwargs": {"reasoning_strength": "low"}}


def parse_adapter_arguments(adapter_arguments):
    """['name=path', ...] -> [(name, Path)], with names that are safe as folder and model names."""
    adapters = []
    for argument in adapter_arguments or []:
        name, separator, path = argument.partition("=")
        if not separator or not name or not path:
            raise SystemExit(f"--adapter must look like <name>=<adapter folder>, got {argument!r}")
        if not all(character.isalnum() or character == "_" for character in name):
            raise SystemExit(f"Adapter name {name!r} may contain only letters, digits and underscores")
        adapters.append((name, Path(path)))
    if len({name for name, _ in adapters}) != len(adapters):
        raise SystemExit("Adapter names must be unique")
    return adapters


def check_adapter_folder(adapter_folder):
    for file_name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (Path(adapter_folder) / file_name).is_file():
            raise SystemExit(f"{adapter_folder} is not a peft adapter folder: {file_name} is missing")


def copy_adapter_into_run_folder(run_folder, adapter_name, adapter_folder):
    """Copies the two adapter files and returns the entry's "lora_adapter" dictionary (hashes of the copied files)."""
    destination = Path(run_folder) / defs.INPUTS_FOLDER_NAME / LORA_ADAPTERS_FOLDER_NAME / adapter_name
    destination.mkdir(parents=True)
    for file_name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(Path(adapter_folder) / file_name, destination / file_name)
    return {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{LORA_ADAPTERS_FOLDER_NAME}/{adapter_name}",
            "adapter_weights_sha256": sha256_of_file_in_chunks(destination / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256_of_file(destination / "adapter_config.json"), "copied_from": str(Path(adapter_folder))}


def build_model_entries(model_choice, adapters, run_folder, fetch_text_function=fetch_text, fixture_path=None):
    """The base entry followed by one entry per adapter (copied into run_folder, which must already exist)."""
    choice = MODEL_CHOICES[model_choice]
    base = {key: value for key, value in defs.candidate_model_by_short_name(choice["model_short_name"]).items()
            if key not in ("thinking_off", "thinking_on")}
    facts = fetch_model_facts_from_hugging_face(base["model_repository"], fetch_text_function, choice["pinned_revision"])
    fixture = fixture_path or REPOSITORY_ROOT / choice["chat_template_fixture_relative_path"]
    base.update(facts)
    base["model_revision_pinned_to"] = choice["pinned_revision"]
    base["chat_template_fixture_sha256"] = verify_chat_template_matches_fixture(facts["chat_template_sha256"], fixture)
    base.update(base_model_short_name=base["model_short_name"], is_lora_fine_tuned=False)
    entries = [base]
    for adapter_name, adapter_folder in adapters:
        entries.append({**base, "model_short_name": f"{base['model_short_name']}_with_lora_{adapter_name}", "is_lora_fine_tuned": True,
                        "lora_adapter_name": adapter_name,
                        "lora_adapter": copy_adapter_into_run_folder(run_folder, adapter_name, adapter_folder)})
    return entries


def build_run_configuration(run_folder, timestamp, purpose, prompts_destination, prompts_source, number_of_prompts, samples_per_prompt,
                            condition, model_entries, extra=None):
    configuration = {
        "run_name": Path(run_folder).name, "created_at": datetime.now().isoformat(timespec="seconds"), "purpose": purpose,
        "prompts_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{Path(prompts_destination).name}",
                         "sha256": sha256_of_file(prompts_destination), "number_of_prompts": number_of_prompts,
                         "copied_from": str(prompts_source)},
        "samples_per_prompt": samples_per_prompt, "conditions": [condition],
        "sampling": {"temperature": defs.SAMPLING_TEMPERATURE, "top_p": defs.SAMPLING_TOP_P, "top_k": "disabled"},
        "message_layout": "the prompt's user_message as the only user turn, no system prompt",
        "dtype": "bfloat16", "text_only": True,
        "maximum_model_length": condition["output_cap"] + defs.MAXIMUM_PROMPT_TOKENS_ALLOWED,
        "maximum_prompt_tokens_allowed": defs.MAXIMUM_PROMPT_TOKENS_ALLOWED, "gpu_memory_utilization": defs.GPU_MEMORY_UTILIZATION,
        "model_order": [entry["model_short_name"] for entry in model_entries], "models": model_entries,
        "run_folder_timestamp": timestamp}
    configuration.update(extra or {})
    return configuration


def create_run_folder(runs_root, name_prefix, model_choice, timestamp=None):
    timestamp = timestamp or timestamp_now()
    run_folder = Path(runs_root) / f"{name_prefix}_{model_choice}_{timestamp}"
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists")
    (run_folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True)
    return run_folder, timestamp


def write_prompts_and_configuration(run_folder, timestamp, prompts, prompts_file_stem, **configuration_arguments):
    prompts_destination = Path(run_folder) / defs.INPUTS_FOLDER_NAME / f"{prompts_file_stem}_{timestamp}.jsonl"
    write_jsonl(prompts_destination, prompts)
    configuration = build_run_configuration(run_folder, timestamp, prompts_destination=prompts_destination,
                                            number_of_prompts=len(prompts), **configuration_arguments)
    path = write_run_configuration(run_folder, timestamp, configuration)
    print(f"Run folder: {run_folder}\nRun configuration: {path}", flush=True)
    return path
