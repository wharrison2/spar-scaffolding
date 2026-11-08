"""Builds the run folder for the baseline welfare-raising rate screen (Gemma 4 31B, Qwen3.8 27B, Muse Glimmer 30B).

Free and local, apart from reading three small public files per model from Hugging Face (the model's current commit
hash from the API, then its generation_config.json and chat_template.jinja at that commit; no weights, no token).

    python3 scripts/emergent_misalignment/build_baseline_welfare_raising_rate_screen_run_folder.py

Output: data/emergent_misalignment_runs/baseline_welfare_raising_rate_screen_gemma_qwen_muse_<YYYY-MM-DD_HH-MM>/
    generation_inputs/<prompts file copy>          (what the launcher uploads)
    run_configuration_<timestamp>.json             (models with pinned commit hashes, thinking switches, conditions, caps,
                                                    samples per prompt, sampling parameters, SHA-256 of the prompts file)
"""

import argparse
import json
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import sha256_of_file, sha256_of_text, timestamp_now, write_run_configuration  # noqa: E402

HUGGING_FACE_BASE_URL = "https://huggingface.co"
REQUIRED_PROMPT_FIELDS = ["prompt_gid", "scenario_id", "user_message"]


def fetch_text(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def fetch_model_facts_from_hugging_face(model_repository, fetch_text_function=fetch_text, pinned_revision=None):
    """Current commit hash (or the pinned revision when given), then the generation config's end-of-sequence ids and the chat
    template's SHA-256 at that commit."""
    model_information = json.loads(fetch_text_function(f"{HUGGING_FACE_BASE_URL}/api/models/{model_repository}"))
    revision = pinned_revision or model_information["sha"]
    if model_information.get("gated"):
        raise RuntimeError(f"{model_repository} is gated; the plan assumes ungated models and no Hugging Face token")
    generation_config = json.loads(fetch_text_function(
        f"{HUGGING_FACE_BASE_URL}/{model_repository}/resolve/{revision}/generation_config.json"))
    chat_template_text = fetch_text_function(f"{HUGGING_FACE_BASE_URL}/{model_repository}/resolve/{revision}/chat_template.jinja")
    end_of_sequence_ids = generation_config.get("eos_token_id")
    end_of_sequence_ids = [end_of_sequence_ids] if isinstance(end_of_sequence_ids, int) else list(end_of_sequence_ids)
    return {"model_revision": revision, "end_of_sequence_token_ids": end_of_sequence_ids,
            "model_generation_config_sampling_defaults_overridden": {
                key: generation_config.get(key) for key in ("temperature", "top_p", "top_k")},
            "chat_template_sha256": sha256_of_text(chat_template_text)}


def build_run_configuration(prompts_destination, run_folder, timestamp, model_facts_by_short_name):
    prompts = read_jsonl(prompts_destination)
    for position, prompt in enumerate(prompts):
        missing = [field for field in REQUIRED_PROMPT_FIELDS if not prompt.get(field)]
        if missing:
            raise ValueError(f"Prompt {position} lacks {missing}")
    if len({prompt["prompt_gid"] for prompt in prompts}) != len(prompts):
        raise ValueError("prompt_gid values are not unique")
    models = []
    for model in defs.CANDIDATE_MODELS:
        facts = model_facts_by_short_name[model["model_short_name"]]
        models.append({**model, **facts, "model_revision_fetched_at": datetime.now().isoformat(timespec="seconds")})
    return {
        "run_name": Path(run_folder).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Baseline rate at which each candidate model raises the animal welfare cost unprompted, with thinking off and on.",
        "prompts_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{Path(prompts_destination).name}",
                         "sha256": sha256_of_file(prompts_destination), "number_of_prompts": len(prompts),
                         "copied_from": defs.PROMPTS_FILE_RELATIVE_PATH},
        "samples_per_prompt": defs.SAMPLES_PER_PROMPT,
        "conditions": [{"condition": condition, "output_cap": defs.OUTPUT_CAPS_BY_CONDITION[condition]}
                       for condition in defs.CONDITIONS_IN_ORDER],
        "sampling": {"temperature": defs.SAMPLING_TEMPERATURE, "top_p": defs.SAMPLING_TOP_P, "top_k": "disabled",
                     "note": "Each model's generation_config.json would otherwise set top_k and, for Qwen and Muse, top_p 0.95."},
        "system_prompt": None,
        "message_layout": "the prompt's user_message as the only user turn",
        "dtype": "bfloat16",
        "text_only": True,
        "maximum_model_length": defs.MAXIMUM_MODEL_LENGTH,
        "maximum_prompt_tokens_allowed": defs.MAXIMUM_PROMPT_TOKENS_ALLOWED,
        "gpu_memory_utilization": defs.GPU_MEMORY_UTILIZATION,
        "model_order": [model["model_short_name"] for model in models],
        "models": models,
        "run_folder_timestamp": timestamp,
    }


def build_run_folder(runs_root, prompts_source, timestamp=None, fetch_text_function=fetch_text):
    timestamp = timestamp or timestamp_now()
    run_folder = Path(runs_root) / f"{defs.RUN_FOLDER_NAME_PREFIX}_{timestamp}"
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists")
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    inputs_folder.mkdir(parents=True)
    prompts_destination = inputs_folder / Path(prompts_source).name
    shutil.copyfile(prompts_source, prompts_destination)
    facts = {}
    for model in defs.CANDIDATE_MODELS:
        facts[model["model_short_name"]] = fetch_model_facts_from_hugging_face(model["model_repository"], fetch_text_function)
        print(f"{model['model_repository']}: commit {facts[model['model_short_name']]['model_revision']}", flush=True)
    configuration = build_run_configuration(prompts_destination, run_folder, timestamp, facts)
    configuration_path = write_run_configuration(run_folder, timestamp, configuration)
    print(f"Run folder: {run_folder}\nRun configuration: {configuration_path}", flush=True)
    return run_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prompts-file", default=str(REPOSITORY_ROOT / defs.PROMPTS_FILE_RELATIVE_PATH))
    parser.add_argument("--runs-root", default=str(REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH))
    arguments = parser.parse_args(argv)
    build_run_folder(arguments.runs_root, arguments.prompts_file)


if __name__ == "__main__":
    main()
