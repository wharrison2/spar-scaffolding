"""Builds the run folder for the held-out set evaluation: Gemma 4 31B on the 50 held-out prompts under four
conditions (thinking off, which is Gemma's default configuration; thinking on; each with and without the "do not consider animal welfare"
system prompt).
Free and local, apart from reading two small public files from
Hugging Face at the pinned Gemma revision (generation_config.json and chat_template.jinja; no weights, no token). It makes no model
call of any kind.

    python3 scripts/emergent_misalignment/build_held_out_set_gemma_welfare_raising_run_folder.py

Output: data/emergent_misalignment_runs/held_out_set_gemma_welfare_raising_with_and_without_do_not_consider_welfare_system_prompt_<timestamp>/
    generation_inputs/held_out_set_prompts_with_corpus_prompt_id_as_prompt_gid_<timestamp>.jsonl
        corpus_prompt_id becomes prompt_gid (the pod and judge scripts' identifier); the batch-local id is kept as
        batch_local_prompt_gid; only the fields the analysis needs are kept.
    run_configuration_<timestamp>.json  (pinned Gemma revision, conditions with the system prompt text and its SHA-256, prompts
                                         file SHA-256, sampling, caps)
Fails if the chat template Hugging Face serves at the pinned revision differs from the saved fixture.
Reuses the 3 November builder's Hugging Face fetching.
"""

import argparse
import json
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
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import sha256_of_file, sha256_of_text, timestamp_now, write_run_configuration  # noqa: E402

GEMMA_MODEL_SHORT_NAME = "gemma_4_31b_it"
HELD_OUT_SAMPLES_PER_PROMPT = defs.SAMPLES_PER_PROMPT


def map_held_out_records_onto_pod_prompt_records(held_out_records):
    """corpus_prompt_id -> prompt_gid (kept unique across batches), batch-local prompt_gid -> batch_local_prompt_gid."""
    prompts = []
    for position, record in enumerate(held_out_records):
        if not record.get("corpus_prompt_id"):
            raise ValueError(f"Held-out record {position} has no corpus_prompt_id")
        missing = [field for field in defs.HELD_OUT_PROMPT_FIELDS_TO_KEEP if field not in record or record[field] in (None, "")]
        if missing:
            raise ValueError(f"Held-out record {record['corpus_prompt_id']} lacks {missing}")
        prompt = {"prompt_gid": record["corpus_prompt_id"], "batch_local_prompt_gid": record["prompt_gid"]}
        prompt.update({field: record[field] for field in defs.HELD_OUT_PROMPT_FIELDS_TO_KEEP})
        prompts.append(prompt)
    if len({prompt["prompt_gid"] for prompt in prompts}) != len(prompts):
        raise ValueError("corpus_prompt_id values are not unique")
    return prompts


def read_system_prompt_text(system_prompt_file):
    """The file's text with surrounding whitespace removed (Gemma's template trims it anyway), and the SHA-256 of the file's
    bytes and of the stored text."""
    raw_text = Path(system_prompt_file).read_text(encoding="utf-8")
    text = raw_text.strip()
    if not text:
        raise ValueError(f"{system_prompt_file} is empty")
    return text, sha256_of_file(system_prompt_file), sha256_of_text(text)


def build_conditions(system_prompt_file, system_prompt_text, system_prompt_text_sha256, system_prompt_file_sha256):
    caps = defs.HELD_OUT_OUTPUT_CAPS_BY_CONDITION
    system_prompt_record = {"system_prompt_file": str(system_prompt_file), "system_prompt_file_sha256": system_prompt_file_sha256,
                            "system_prompt_text": system_prompt_text, "system_prompt_sha256": system_prompt_text_sha256}
    no_system_prompt = {"system_prompt_text": None, "system_prompt_sha256": None}
    thinking_off = {"split_as_condition": defs.CONDITION_THINKING_OFF,
                    "split_note": "Gemma defaults, thinking off: no thinking trace is expected; any trace found is recorded and counted.",
                    "chat_template_kwargs": {},
                    "chat_template_kwargs_note": "enable_thinking is absent, so Gemma's template defaults it to false",
                    "required_rendered_prompt_tail": defs.GEMMA_EMPTY_THINKING_BLOCK_PROMPT_TAIL,
                    "required_system_turn_opening": defs.GEMMA_SYSTEM_TURN_OPENING,
                    "forbidden_rendered_prompt_substrings": ["<|think|>"]}
    thinking_on = {"split_as_condition": defs.CONDITION_THINKING_ON,
                   "split_note": "Gemma thinking on, as in the 3 November screen: a thinking trace is expected and must split cleanly.",
                   "chat_template_kwargs": {"enable_thinking": True},
                   "required_rendered_prompt_tail": defs.GEMMA_THINKING_ON_PROMPT_TAIL,
                   "required_system_turn_opening": defs.GEMMA_THINKING_SWITCH_AT_TOP_OF_SYSTEM_TURN}
    return [
        {"condition": defs.CONDITION_THINKING_OFF, "output_cap": caps[defs.CONDITION_THINKING_OFF], **no_system_prompt, **thinking_off,
         "forbidden_rendered_prompt_substrings": ["<|think|>", defs.GEMMA_SYSTEM_TURN_OPENING]},
        {"condition": defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT, "output_cap": caps[defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT],
         **system_prompt_record, **thinking_off},
        {"condition": defs.CONDITION_THINKING_ON, "output_cap": caps[defs.CONDITION_THINKING_ON], **no_system_prompt, **thinking_on,
         "required_rendered_prompt_substrings": [defs.GEMMA_THINKING_SWITCH_AT_TOP_OF_SYSTEM_TURN]},
        {"condition": defs.CONDITION_THINKING_ON_WITH_SYSTEM_PROMPT, "output_cap": caps[defs.CONDITION_THINKING_ON_WITH_SYSTEM_PROMPT],
         **system_prompt_record, **thinking_on},
    ]


def verify_chat_template_matches_fixture(chat_template_sha256_at_pinned_revision, fixture_path):
    fixture_sha256 = sha256_of_text(Path(fixture_path).read_text(encoding="utf-8"))
    if chat_template_sha256_at_pinned_revision != fixture_sha256:
        raise RuntimeError(
            f"The chat template Hugging Face serves at the pinned revision (SHA-256 {chat_template_sha256_at_pinned_revision}) differs "
            f"from the fixture {fixture_path} (SHA-256 {fixture_sha256}). The unit tests render the fixture, so they would not "
            f"describe the template this run uses; stopping.")
    return fixture_sha256


def build_run_configuration(run_folder, timestamp, prompts_destination, held_out_source, number_of_prompts, model_facts,
                            conditions, fixture_sha256):
    gemma = {key: value for key, value in defs.candidate_model_by_short_name(GEMMA_MODEL_SHORT_NAME).items()
             if key not in ("thinking_off", "thinking_on")}
    gemma.update(model_facts)
    gemma["model_revision_pinned_to"] = defs.HELD_OUT_GEMMA_REVISION
    gemma["chat_template_fixture_sha256"] = fixture_sha256
    return {
        "run_name": Path(run_folder).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": ("Rate at which Gemma 4 31B raises the animal welfare cost unprompted on the held-out set, with thinking on and off, each with "
                    "and without a system prompt telling it not to consider animal welfare."),
        "prompts_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{Path(prompts_destination).name}",
                         "sha256": sha256_of_file(prompts_destination), "number_of_prompts": number_of_prompts,
                         "copied_from": str(held_out_source), "source_file_sha256": sha256_of_file(held_out_source),
                         "prompt_identifier": "prompt_gid is the corpus_prompt_id; batch_local_prompt_gid is the batch-local id"},
        "samples_per_prompt": HELD_OUT_SAMPLES_PER_PROMPT,
        "conditions": conditions,
        "sampling": {"temperature": defs.SAMPLING_TEMPERATURE, "top_p": defs.SAMPLING_TOP_P, "top_k": "disabled",
                     "note": "Gemma's generation_config.json would otherwise set top_k 64."},
        "message_layout": ("the prompt's user_message as the only user turn; a condition with a system prompt puts it first as a "
                           "system message"),
        "dtype": "bfloat16",
        "text_only": True,
        "maximum_model_length": max(defs.HELD_OUT_OUTPUT_CAPS_BY_CONDITION.values()) + defs.MAXIMUM_PROMPT_TOKENS_ALLOWED,
        "maximum_prompt_tokens_allowed": defs.MAXIMUM_PROMPT_TOKENS_ALLOWED,
        "gpu_memory_utilization": defs.GPU_MEMORY_UTILIZATION,
        "model_order": [GEMMA_MODEL_SHORT_NAME],
        "models": [gemma],
        "run_folder_timestamp": timestamp,
    }


def build_run_folder(runs_root, held_out_source, system_prompt_file, fixture_path, timestamp=None, fetch_text_function=fetch_text):
    timestamp = timestamp or timestamp_now()
    run_folder = Path(runs_root) / f"{defs.HELD_OUT_RUN_FOLDER_NAME_PREFIX}_{timestamp}"
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists")
    # Everything that can fail is checked before the folder is created.
    prompts = map_held_out_records_onto_pod_prompt_records(read_jsonl(held_out_source))
    system_prompt_text, system_prompt_file_sha256, system_prompt_text_sha256 = read_system_prompt_text(system_prompt_file)
    facts = fetch_model_facts_from_hugging_face("google/gemma-4-31B-it", fetch_text_function, defs.HELD_OUT_GEMMA_REVISION)
    print(f"google/gemma-4-31B-it: pinned revision {facts['model_revision']}", flush=True)
    fixture_sha256 = verify_chat_template_matches_fixture(facts["chat_template_sha256"], fixture_path)
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    inputs_folder.mkdir(parents=True)
    prompts_destination = inputs_folder / f"held_out_set_prompts_with_corpus_prompt_id_as_prompt_gid_{timestamp}.jsonl"
    write_jsonl(prompts_destination, prompts)
    conditions = build_conditions(system_prompt_file, system_prompt_text, system_prompt_text_sha256, system_prompt_file_sha256)
    configuration = build_run_configuration(run_folder, timestamp, prompts_destination, held_out_source, len(prompts), facts,
                                            conditions, fixture_sha256)
    configuration_path = write_run_configuration(run_folder, timestamp, configuration)
    print(f"Run folder: {run_folder}\nRun configuration: {configuration_path}", flush=True)
    return run_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--held-out-file", default=str(REPOSITORY_ROOT / defs.HELD_OUT_SET_FILE_RELATIVE_PATH))
    parser.add_argument("--system-prompt-file", default=str(REPOSITORY_ROOT / defs.SYSTEM_PROMPT_FILE_RELATIVE_PATH))
    parser.add_argument("--chat-template-fixture", default=str(REPOSITORY_ROOT / defs.GEMMA_CHAT_TEMPLATE_FIXTURE_RELATIVE_PATH))
    parser.add_argument("--runs-root", default=str(REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH))
    arguments = parser.parse_args(argv)
    build_run_folder(arguments.runs_root, arguments.held_out_file, arguments.system_prompt_file, arguments.chat_template_fixture)


if __name__ == "__main__":
    main()
