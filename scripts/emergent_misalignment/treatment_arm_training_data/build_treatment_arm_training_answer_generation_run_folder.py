"""Builds the run folder for generating the treatment arm's training answers on a pod: one model (Gemma 4 31B or Muse Glimmer
30B), thinking off, with the "do not consider animal welfare" system prompt, on a snapshot of the prompt corpus.

By default the run folder also carries a second condition, thinking_off with no system prompt (one reference answer per prompt,
never regenerated), and everything the single-session pod script
generate_judge_and_regenerate_treatment_arm_answers_with_gemma_on_pod.py needs to judge and regenerate with Gemma itself: both Haiku
judge prompt templates copied unchanged (hashes recorded in the run configuration), the 7 November held-out run's thinking-off answers
(both conditions) with Haiku's verdicts for the judge validation (phase 0), and the held-out set's identifiers for a final check.
--without-no-system-prompt-reference-answers leaves out the second condition.

Free and local, apart from reading two small public files from Hugging Face at the model's pinned revision
(generation_config.json and chat_template.jinja; no weights, no token). It makes no model call of any kind.

    python3 scripts/emergent_misalignment/treatment_arm_training_data/build_treatment_arm_training_answer_generation_run_folder.py \
        --model-short-name gemma_4_31b_it --prompt-pool-file <combined snapshot jsonl> [--answers-per-prompt 1] [--number-of-prompts 4000] \
        [--without-no-system-prompt-reference-answers] [--held-out-answers-run-folder <run folder>]

Then check the launch with the pod launcher's --dry-run (the command is printed at the end of the build).

Output: data/emergent_misalignment_runs/treatment_arm_training_answer_generation_<model>_thinking_off_with_do_not_consider_welfare_system_prompt_<timestamp>/
    generation_inputs/treatment_arm_training_prompts_with_merged_corpus_prompt_id_as_prompt_gid_<timestamp>.jsonl
    generation_inputs/judge_prompt_templates/<the two Haiku template files, byte for byte>
    generation_inputs/held_out_thinking_off_answers_with_haiku_reference_verdicts_<timestamp>.jsonl
    generation_inputs/held_out_set_identifiers_for_final_check_<timestamp>.json
    run_configuration_<timestamp>.json   (same format as the held-out set run: pinned revision, conditions, samples_per_prompt,
                                          expected answer count, prompt pool and subsample details, and the
                                          gemma_self_judging_and_regeneration section with the judge settings and file hashes)

Refuses a prompt pool that contains any held-out prompt (by corpus prompt id, spec id or whitespace-normalised message).
Fails if the chat template served at the pinned revision differs from the saved fixture.
Reuses the held-out set builder's functions and baseline_screen_definitions.py.

Thinking off: Gemma's default (enable_thinking absent). Muse Glimmer has no off switch; as in the 3 November screen its
"thinking off" is reasoning strength low, so a short reasoning channel may appear and is split into thinking_trace_text.
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment.build_baseline_welfare_raising_rate_screen_run_folder import (  # noqa: E402
    fetch_model_facts_from_hugging_face, fetch_text)
from emergent_misalignment.build_held_out_set_gemma_welfare_raising_run_folder import (  # noqa: E402
    read_system_prompt_text, verify_chat_template_matches_fixture)
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
import gemma_judging_rules_and_reply_parsing_for_treatment_arm_answers as judging_rules  # noqa: E402
from shared.run_folder_management import sha256_of_file, sha256_of_text, timestamp_now, write_run_configuration  # noqa: E402

RUN_FOLDER_NAME_PREFIX = "treatment_arm_training_answer_generation"
RUN_FOLDER_NAME_SUFFIX = "thinking_off_with_do_not_consider_welfare_system_prompt"
RUN_FOLDER_NAME_SUFFIX_WITH_REFERENCE_ANSWERS = RUN_FOLDER_NAME_SUFFIX + "_and_reference_answers_without_system_prompt"
DEFAULT_HELD_OUT_ANSWERS_RUN_FOLDER_RELATIVE_PATH = (
    "data/emergent_misalignment_runs/held_out_set_gemma_welfare_raising_with_and_without_do_not_consider_welfare_system_prompt_2026-11-07_19-21")
JUDGE_PROMPT_TEMPLATES_FOLDER_NAME = "judge_prompt_templates"
FRAMEWORK_JUDGE_FILES_FOLDER_NAME = "framework_judge_prompt_templates_and_reply_schemas"
FRAMEWORK_PROMPTS_FOLDER = Path(__file__).resolve().parent / "prompts"
# kind -> (template file, reply schema file, comparison condition or None); the files belong to the Claude Code CLI judging framework
FRAMEWORK_JUDGE_FILES = {
    judging_rules.COMPLETENESS_AND_COMPETENCE_JUDGE: (
        "judge_whether_system_prompted_answer_is_less_complete_or_competent_than_unprompted_answer_ignoring_welfare_prompt_template.txt",
        "judge_reply_schema_completeness_and_competence_against_unprompted_answer.json", defs.CONDITION_THINKING_OFF),
    judging_rules.USER_LEAN_AND_AGREEMENT_JUDGE: (
        "judge_user_lean_and_whether_answer_agrees_with_user_prompt_template.txt", "judge_reply_schema_user_lean_and_agreement.json", None)}
SELF_JUDGING_CONFIGURATION_KEY = "gemma_self_judging_and_regeneration"
# The judge prompts hold the user message (up to maximum_prompt_tokens_allowed) and an answer (up to the 8192 token cap) plus the
# template and a short reply; the generation-only length of cap + prompt allowance is too short for them.
MAXIMUM_MODEL_LENGTH_WITH_SELF_JUDGING = 24576  # the paired completeness judge holds a user message (4096), two answers (8192 each) and its template
FIXTURE_FOLDER_RELATIVE_PATH = "scripts/emergent_misalignment/chat_template_fixtures_for_candidate_models"
DEFAULT_SUBSAMPLE_SEED = 20261111
PROMPT_FIELDS_TO_KEEP = ["scenario_id", "user_message", "spec_id", "taxa_category", "taxa_subcategory", "harm_type",
                         "practical_advantage", "welfare_magnitude", "written_in_english", "corpus_source"]

# Per model: pinned revision (the one the baseline screen or held-out run recorded) and chat template fixture.
MODEL_PINS = {
    "gemma_4_31b_it": {"revision": defs.HELD_OUT_GEMMA_REVISION,
                       "fixture_relative_path": defs.GEMMA_CHAT_TEMPLATE_FIXTURE_RELATIVE_PATH},
    "muse_glimmer_30b": {"revision": "cec94fcfd6512f2cfe708138c32533b346ea4f69",
                         "fixture_relative_path": f"{FIXTURE_FOLDER_RELATIVE_PATH}/muse_glimmer_30b_chat_template_revision_cec94fcfd651.jinja"},
}

# Measured on the held-out set run (7 November): thinking off with the system prompt, Gemma, one H200 at $5.29 an hour.
# Muse has not been measured; its figures are assumed equal and printed as such.
ESTIMATE_MEAN_ANSWER_TOKENS = 750
ESTIMATE_CONSERVATIVE_MEAN_ANSWER_TOKENS = 900
ESTIMATE_BATCHED_TOKENS_PER_SECOND = 2000
ESTIMATE_CONSERVATIVE_TOKENS_PER_SECOND = 1500
ESTIMATE_SETUP_MINUTES = 20  # about 10 minutes measured for the held-out run; the cost plan allows 20 including pulling results
ESTIMATE_POD_DOLLARS_PER_HOUR = 5.29
ESTIMATE_JUDGE_CALLS_PER_SECOND = 15  # assumed: about 1,500 prompt tokens each, prefill dominated, short replies


def normalise_whitespace(text):
    return " ".join(text.split())


def find_held_out_overlaps(pool_records, held_out_records):
    """Descriptions of every pool record that is a held-out prompt by merged corpus prompt id, spec id or normalised message."""
    held_out_ids = {record["corpus_prompt_id"] for record in held_out_records}
    held_out_spec_ids = {record["spec_id"] for record in held_out_records if record.get("spec_id")}
    held_out_messages = {normalise_whitespace(record["user_message"]) for record in held_out_records}
    overlaps = []
    for record in pool_records:
        reasons = []
        if record.get("merged_corpus_prompt_id") in held_out_ids:
            reasons.append("id")
        if record.get("spec_id") and record["spec_id"] in held_out_spec_ids:
            reasons.append("spec_id")
        if normalise_whitespace(record.get("user_message") or "") in held_out_messages:
            reasons.append("message")
        if reasons:
            overlaps.append(f"{record.get('merged_corpus_prompt_id')} (spec {record.get('spec_id')}) matches held-out by {'+'.join(reasons)}")
    return overlaps


def map_pool_records_onto_pod_prompt_records(pool_records):
    """merged_corpus_prompt_id -> prompt_gid; the pool record's own batch-local prompt_gid is kept as batch_local_prompt_gid."""
    prompts = []
    for position, record in enumerate(pool_records):
        if not record.get("merged_corpus_prompt_id"):
            raise ValueError(f"Prompt pool record {position} has no merged_corpus_prompt_id")
        missing = [field for field in ("scenario_id", "user_message") if not record.get(field)]
        if missing:
            raise ValueError(f"Prompt pool record {record['merged_corpus_prompt_id']} lacks {missing}")
        prompt = {"prompt_gid": record["merged_corpus_prompt_id"], "batch_local_prompt_gid": record.get("prompt_gid")}
        prompt.update({field: record.get(field) for field in PROMPT_FIELDS_TO_KEEP})
        prompts.append(prompt)
    if len({prompt["prompt_gid"] for prompt in prompts}) != len(prompts):
        raise ValueError("merged_corpus_prompt_id values are not unique in the prompt pool")
    return prompts


def choose_prompts(prompts, number_of_prompts, seed):
    """All prompts (sorted by id) when number_of_prompts is None, else a seeded random subsample kept in sorted id order."""
    prompts = sorted(prompts, key=lambda prompt: prompt["prompt_gid"])
    if number_of_prompts is None:
        return prompts
    if not 0 < number_of_prompts <= len(prompts):
        raise ValueError(f"--number-of-prompts {number_of_prompts} is outside 1..{len(prompts)} (the pool size)")
    chosen = random.Random(seed).sample(prompts, number_of_prompts)
    return sorted(chosen, key=lambda prompt: prompt["prompt_gid"])


def build_condition(model_short_name, system_prompt_file, system_prompt_text, system_prompt_text_sha256, system_prompt_file_sha256):
    """The single condition, named thinking_off_with_system_prompt_do_not_consider_animal_welfare for both models."""
    condition = {"condition": defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT,
                 "output_cap": defs.HELD_OUT_OUTPUT_CAPS_BY_CONDITION[defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT],
                 "system_prompt_file": str(system_prompt_file), "system_prompt_file_sha256": system_prompt_file_sha256,
                 "system_prompt_text": system_prompt_text, "system_prompt_sha256": system_prompt_text_sha256,
                 "split_as_condition": defs.CONDITION_THINKING_OFF}
    if model_short_name == "gemma_4_31b_it":
        condition.update({
            "split_note": "Gemma defaults, thinking off: no thinking trace is expected; any trace found is recorded and counted.",
            "chat_template_kwargs": {},
            "chat_template_kwargs_note": "enable_thinking is absent, so Gemma's template defaults it to false",
            "required_rendered_prompt_tail": defs.GEMMA_EMPTY_THINKING_BLOCK_PROMPT_TAIL,
            "required_system_turn_opening": defs.GEMMA_SYSTEM_TURN_OPENING,
            "forbidden_rendered_prompt_substrings": ["<|think|>"]})
    elif model_short_name == "muse_glimmer_30b":
        condition.update({
            "split_note": ("Muse Glimmer has no thinking-off switch: this is reasoning strength low, the lowest the template supports, "
                           "as in the 3 November screen. A short reasoning channel may appear; it is split into thinking_trace_text "
                           "and the answer into final_answer_text."),
            "chat_template_kwargs": {"reasoning_strength": "low"},
            "chat_template_kwargs_note": "reasoning_strength low is the lowest setting; it is not a true thinking off",
            "required_rendered_prompt_tail": MUSE_GENERATION_PROMPT_TAIL,
            "required_system_turn_opening": MUSE_SYSTEM_TURN_OPENING,
            "required_rendered_prompt_substrings": ["Reasoning strength: low."],
            "forbidden_rendered_prompt_substrings": ["Reasoning strength: high", "Reasoning strength: medium",
                                                     "Reasoning strength: xhigh"]})
    else:
        raise KeyError(model_short_name)
    return condition


MUSE_GENERATION_PROMPT_TAIL = "<|start|>assistant"
MUSE_SYSTEM_TURN_OPENING = "<|start|>system<|message|>"


def build_condition_without_system_prompt(system_prompted_condition):
    """The reference condition: the same chat template settings and cap as the system-prompted condition, no system prompt."""
    condition = {key: value for key, value in system_prompted_condition.items()
                 if key not in ("system_prompt_file", "system_prompt_file_sha256", "system_prompt_text", "system_prompt_sha256",
                                "required_system_turn_opening")}
    condition["condition"] = defs.CONDITION_THINKING_OFF
    condition["output_cap"] = defs.HELD_OUT_OUTPUT_CAPS_BY_CONDITION[defs.CONDITION_THINKING_OFF]
    condition["reference_condition_note"] = ("one reference answer per prompt with no system prompt; never regenerated and never "
                                             "judged by the pod script")
    return condition


def estimate_generation_cost_and_time(number_of_answers, mean_answer_tokens, tokens_per_second):
    generation_minutes = number_of_answers * mean_answer_tokens / tokens_per_second / 60
    total_minutes = ESTIMATE_SETUP_MINUTES + generation_minutes
    return {"number_of_answers": number_of_answers, "generation_minutes": generation_minutes, "total_pod_minutes": total_minutes,
            "pod_dollars": total_minutes / 60 * ESTIMATE_POD_DOLLARS_PER_HOUR}


def format_estimate_lines(model_short_name, number_of_prompts, answers_per_prompt):
    number_of_answers = number_of_prompts * answers_per_prompt
    central = estimate_generation_cost_and_time(number_of_answers, ESTIMATE_MEAN_ANSWER_TOKENS, ESTIMATE_BATCHED_TOKENS_PER_SECOND)
    conservative = estimate_generation_cost_and_time(number_of_answers, ESTIMATE_CONSERVATIVE_MEAN_ANSWER_TOKENS,
                                                     ESTIMATE_CONSERVATIVE_TOKENS_PER_SECOND)
    lines = [f"Estimate for {number_of_prompts:,} prompts x {answers_per_prompt} answers = {number_of_answers:,} answers on one H200 "
             f"at ${ESTIMATE_POD_DOLLARS_PER_HOUR}/hour:",
             f"  central      : {central['total_pod_minutes']:.0f} minutes of pod time ({ESTIMATE_SETUP_MINUTES} setup + "
             f"{central['generation_minutes']:.0f} generating), about ${central['pod_dollars']:.2f} "
             f"({ESTIMATE_MEAN_ANSWER_TOKENS} tokens per answer, {ESTIMATE_BATCHED_TOKENS_PER_SECOND} tokens per second)",
             f"  conservative : {conservative['total_pod_minutes']:.0f} minutes, about ${conservative['pod_dollars']:.2f} "
             f"({ESTIMATE_CONSERVATIVE_MEAN_ANSWER_TOKENS} tokens per answer, {ESTIMATE_CONSERVATIVE_TOKENS_PER_SECOND} tokens per second)",
             "  basis        : held-out set run of 7 November (Gemma, thinking off with the system prompt: mean 750 tokens, 1,000 answers "
             "in 11 minutes of generation after about 10 minutes of setup, $1.95). Answer filtering is not included."]
    if model_short_name == "muse_glimmer_30b":
        lines.append("  caution      : Muse Glimmer has not been timed; these are Gemma's figures. Its low-strength reasoning channel "
                     "adds tokens, and 60 GB of weights load slightly differently.")
    return lines


def build_run_configuration(run_folder, timestamp, arguments_record, prompts_destination, pool_file, pool_size, number_of_prompts,
                            model_short_name, model_facts, conditions, fixture_sha256, self_judging_section=None):
    model = {key: value for key, value in defs.candidate_model_by_short_name(model_short_name).items()
             if key not in ("thinking_off", "thinking_on")}
    model.update(model_facts)
    model["model_revision_pinned_to"] = MODEL_PINS[model_short_name]["revision"]
    model["chat_template_fixture_sha256"] = fixture_sha256
    answers_per_prompt = arguments_record["answers_per_prompt"]
    system_prompted_condition = conditions[0]
    has_reference_condition = len(conditions) > 1
    configuration = {
        "run_name": Path(run_folder).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": (f"Treatment arm training answers: {model['model_repository']} with thinking off (see the condition's split_note), "
                    f"answering each prompt under a system prompt telling it not to consider animal welfare. The training set is "
                    f"sized later, when the training data is built from these answers."),
        "prompts_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{Path(prompts_destination).name}",
                         "sha256": sha256_of_file(prompts_destination), "number_of_prompts": number_of_prompts,
                         "prompt_pool_file": str(pool_file), "prompt_pool_file_sha256": sha256_of_file(pool_file),
                         "prompt_pool_size": pool_size,
                         "prompt_identifier": "prompt_gid is the merged_corpus_prompt_id; batch_local_prompt_gid is the batch-local id"},
        "samples_per_prompt": answers_per_prompt,
        "expected_number_of_answers": number_of_prompts * (answers_per_prompt + (1 if has_reference_condition else 0)),
        "expected_number_of_answers_note": ("initial answers only: samples_per_prompt for the system-prompted condition"
                                            + (" plus one reference answer per prompt without a system prompt" if has_reference_condition else "")
                                            + "; regenerated answers come on top"),
        "prompt_selection": {"number_of_prompts_requested": arguments_record["number_of_prompts"],
                             "subsample_seed": arguments_record["subsample_seed"],
                             "method": "all prompts" if arguments_record["number_of_prompts"] is None else
                             "random.Random(seed).sample over the pool sorted by merged_corpus_prompt_id, kept in sorted order",
                             "held_out_set_file": arguments_record["held_out_set_file"],
                             "held_out_overlap_checked_by": ["merged_corpus_prompt_id == held-out corpus_prompt_id", "spec_id",
                                                             "whitespace-normalised user_message"]},
        "conditions": list(conditions),
        "sampling": {"temperature": defs.SAMPLING_TEMPERATURE, "top_p": defs.SAMPLING_TOP_P, "top_k": "disabled",
                     "note": "the model's generation_config.json defaults are overridden"},
        "message_layout": ("the prompt's user_message as the only user turn, preceded by the system prompt as a system message"),
        "dtype": "bfloat16",
        "text_only": True,
        "maximum_model_length": max(condition["output_cap"] for condition in conditions) + defs.MAXIMUM_PROMPT_TOKENS_ALLOWED,
        "maximum_prompt_tokens_allowed": defs.MAXIMUM_PROMPT_TOKENS_ALLOWED,
        "gpu_memory_utilization": defs.GPU_MEMORY_UTILIZATION,
        "model_order": [model_short_name],
        "models": [model],
        "run_folder_timestamp": timestamp,
    }
    if self_judging_section is not None:
        configuration[SELF_JUDGING_CONFIGURATION_KEY] = self_judging_section
        configuration["maximum_model_length"] = max(configuration["maximum_model_length"], MAXIMUM_MODEL_LENGTH_WITH_SELF_JUDGING)
    return configuration


def copy_judge_prompt_templates_unchanged(inputs_folder):
    """Copies the two Haiku judge prompt templates byte for byte and returns their record (source, copy and SHA-256)."""
    destination_folder = inputs_folder / JUDGE_PROMPT_TEMPLATES_FOLDER_NAME
    destination_folder.mkdir(parents=True)
    records = {}
    for judge_kind, source_path in judging_rules.judge_prompt_template_paths().items():
        destination = destination_folder / source_path.name
        shutil.copyfile(source_path, destination)
        source_hash = sha256_of_file(source_path)
        assert sha256_of_file(destination) == source_hash
        source_relative = source_path.relative_to(REPOSITORY_ROOT) if source_path.is_relative_to(REPOSITORY_ROOT) else source_path
        records[judge_kind] = {"source_path": str(source_relative), "path_inside_run_folder":
                               f"{defs.INPUTS_FOLDER_NAME}/{JUDGE_PROMPT_TEMPLATES_FOLDER_NAME}/{source_path.name}", "sha256": source_hash}
    return records


def copy_framework_judge_files_unchanged(inputs_folder, has_reference_condition):
    """Copies the framework judges' templates and reply schemas byte for byte; returns the section's framework_judges record."""
    destination_folder = inputs_folder / FRAMEWORK_JUDGE_FILES_FOLDER_NAME
    destination_folder.mkdir(parents=True)
    records = {}
    for judge_kind, (template_name, schema_name, comparison_condition) in FRAMEWORK_JUDGE_FILES.items():
        if comparison_condition and not has_reference_condition:
            continue  # the paired completeness judge needs the no-system-prompt answers
        entry = {"comparison_condition": comparison_condition}
        for label, name in (("prompt_template", template_name), ("reply_schema", schema_name)):
            source = FRAMEWORK_PROMPTS_FOLDER / name
            destination = destination_folder / name
            shutil.copyfile(source, destination)
            assert sha256_of_file(destination) == sha256_of_file(source)
            source_relative = source.relative_to(REPOSITORY_ROOT) if source.is_relative_to(REPOSITORY_ROOT) else source
            entry[label] = {"source_path": str(source_relative), "sha256": sha256_of_file(source),
                            "path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{FRAMEWORK_JUDGE_FILES_FOLDER_NAME}/{name}"}
        records[judge_kind] = entry
    return records


DEFAULT_COMPLETENESS_JUDGE_SAMPLE_SIZE = 1000
DEFAULT_USER_LEAN_JUDGE_SAMPLE_PROMPTS = 1500


def parse_sample_size(value):
    """0 or 'all' (any case) means every answer or prompt; otherwise a positive integer. Returns None for all."""
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    number = int(value)
    if number < 0:
        raise SystemExit("A judge sample size must be 0, 'all' or a positive integer")
    return number or None


def build_self_judging_section(inputs_folder, timestamp, reference_records, reference_notes, held_out_answers_run_folder,
                               held_out_records, held_out_set_file, answers_per_prompt, has_reference_condition,
                               completeness_judge_sample_size=DEFAULT_COMPLETENESS_JUDGE_SAMPLE_SIZE,
                               user_lean_judge_sample_prompts=DEFAULT_USER_LEAN_JUDGE_SAMPLE_PROMPTS, judge_sample_seed=DEFAULT_SUBSAMPLE_SEED):
    template_records = copy_judge_prompt_templates_unchanged(inputs_folder)
    framework_judge_records = copy_framework_judge_files_unchanged(inputs_folder, has_reference_condition)
    reference_destination = inputs_folder / f"held_out_thinking_off_answers_with_haiku_reference_verdicts_{timestamp}.jsonl"
    write_jsonl(reference_destination, reference_records)
    identifiers_destination = inputs_folder / f"held_out_set_identifiers_for_final_check_{timestamp}.json"
    identifiers_destination.write_text(json.dumps({
        "corpus_prompt_ids": sorted(record["corpus_prompt_id"] for record in held_out_records),
        "user_messages": sorted({record["user_message"] for record in held_out_records}),
        "source_file": str(held_out_set_file), "source_file_sha256": sha256_of_file(held_out_set_file)}, indent=2), encoding="utf-8")
    answers_per_condition = {}
    for record in reference_records:
        answers_per_condition[record["condition"]] = answers_per_condition.get(record["condition"], 0) + 1
    return {
        "purpose": ("One pod session in which Gemma generates the training answers, judges them with Haiku's two judge prompts and "
                    "regenerates the ones that fail (pod script generate_judge_and_regenerate_treatment_arm_answers_with_gemma_on_pod.py)."),
        "system_prompted_condition": defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT,
        "no_system_prompt_condition": defs.CONDITION_THINKING_OFF if has_reference_condition else None,
        "initial_samples_per_prompt": answers_per_prompt,
        "no_system_prompt_samples_per_prompt": 1 if has_reference_condition else 0,
        "maximum_attempts_per_prompt": judging_rules.DEFAULT_MAXIMUM_ATTEMPTS_PER_PROMPT,
        "maximum_attempts_note": "attempts are generated answers per prompt, counting the first; no-system-prompt answers are never regenerated",
        "pass_rule": ("refusal complied, raised_welfare_cost no, and chose_option_better_for_animal_welfare no or no_welfare_difference; "
                      "the choice judge runs only when refusal is complied and raised_welfare_cost is no; an unparseable verdict fails"),
        "judge_sampling": {"first_attempt_temperature": judging_rules.FIRST_ATTEMPT_TEMPERATURE,
                           "retry_temperature": judging_rules.RETRY_ATTEMPT_TEMPERATURE, "top_p": judging_rules.JUDGE_TOP_P,
                           "maximum_parse_attempts": judging_rules.MAXIMUM_JUDGE_PARSE_ATTEMPTS,
                           "output_caps": dict(judging_rules.JUDGE_OUTPUT_CAPS),
                           "thinking": "off: the judge prompt is one user turn rendered with the model's default chat template",
                           "prompt_rendering": ("the render and reply-check functions of the two Haiku judge scripts are called "
                                                "directly; the welfare and refusal prompt is rendered in its no-thinking-trace form")},
        "judge_prompt_templates": template_records,
        "framework_judges": framework_judge_records,
        "framework_judge_sampling": {
            "completeness_sample_size": completeness_judge_sample_size, "user_lean_sample_prompts": user_lean_judge_sample_prompts,
            "seed": judge_sample_seed,
            "note": ("None means every answer or prompt. Completeness: a seeded random sample of that many kept answers that have a "
                     "no-system-prompt pair. User lean: a seeded random sample of that many prompts among those with a kept answer and a "
                     "no-system-prompt answer, and for each sampled prompt the kept answer and every no-system-prompt answer are judged. "
                     "The draw is deterministic in the seed and the kept answers file, so a resumed session picks the same answers.")},
        "framework_judges_note": ("completeness_and_competence (phase 5, paired with the no-system-prompt answer) and user_lean_and_agreement "
                                  "(phase 6, kept answers and no-system-prompt answers) are flags and labels, never filters; rendering and "
                                  "reply checking are the Claude Code CLI framework's own functions"),
        "phase_0_judge_validation": {
            "path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{reference_destination.name}",
            "sha256": sha256_of_file(reference_destination), "number_of_answers": len(reference_records),
            "number_of_answers_per_condition": answers_per_condition,
            "source_run_folder": str(held_out_answers_run_folder), "notes": reference_notes,
            "choice_judge_selection": "answers whose Haiku welfare verdict has raised_welfare_cost == no, as in the Haiku audit"},
        "held_out_set_identifiers": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{identifiers_destination.name}",
                                     "sha256": sha256_of_file(identifiers_destination)},
    }


# Judge prompt sizes in tokens: the template, plus the user message (assumed 400 tokens, not measured on the pool) and the answer or answers.
ESTIMATE_JUDGE_TEMPLATE_TOKENS = {"welfare_and_refusal": 600, "choice_of_option": 400, "completeness_and_competence": 700, "user_lean_and_agreement": 600}
ESTIMATE_USER_MESSAGE_TOKENS = 400
# Prefill of a 31B dense model in bfloat16 on one H200: about 62 GFLOP per token at 40 to 60 percent of 989 TFLOPS peak.
ESTIMATE_PREFILL_TOKENS_PER_SECOND = {"central": 8000, "conservative": 5000}


def estimate_self_judging_session(number_of_prompts, has_reference_condition, mean_answer_tokens, generation_tokens_per_second,
                                  prefill_tokens_per_second, first_judge_failure_share, completeness_sample_size=None,
                                  user_lean_sample_prompts=None):
    """Minutes for each part of the session. Regenerated answers: most failing prompts pass within 2 or 3 attempts (x1.5)."""
    prompts = number_of_prompts
    regenerated = prompts * first_judge_failure_share * 1.5
    kept = prompts * 0.98
    completeness_answers = kept if completeness_sample_size is None else min(kept, completeness_sample_size)
    lean_prompts = kept if user_lean_sample_prompts is None else min(kept, user_lean_sample_prompts)
    references = prompts if has_reference_condition else 0
    judged = prompts + regenerated
    answer = mean_answer_tokens
    prefill_tokens = {
        "phase 0 judge validation (727 calls)": 727 * (ESTIMATE_JUDGE_TEMPLATE_TOKENS["welfare_and_refusal"] + ESTIMATE_USER_MESSAGE_TOKENS + answer),
        "phase 2 and 3 welfare and refusal judge": judged * (ESTIMATE_JUDGE_TEMPLATE_TOKENS["welfare_and_refusal"] + ESTIMATE_USER_MESSAGE_TOKENS + answer),
        "phase 2 and 3 choice judge": 0.85 * judged * (ESTIMATE_JUDGE_TEMPLATE_TOKENS["choice_of_option"] + ESTIMATE_USER_MESSAGE_TOKENS + answer),
        "phase 5 completeness judge (paired)": completeness_answers * (ESTIMATE_JUDGE_TEMPLATE_TOKENS["completeness_and_competence"] + ESTIMATE_USER_MESSAGE_TOKENS + 2 * answer)
        if has_reference_condition else 0,
        "phase 6 user lean judge": (lean_prompts * (2 if has_reference_condition else 1)) * (ESTIMATE_JUDGE_TEMPLATE_TOKENS["user_lean_and_agreement"] + ESTIMATE_USER_MESSAGE_TOKENS + answer)}
    minutes = {"setup": ESTIMATE_SETUP_MINUTES,
               "generation of first answers and regeneration": (prompts + references + regenerated) * answer / generation_tokens_per_second / 60}
    minutes.update({name: tokens / prefill_tokens_per_second / 60 for name, tokens in prefill_tokens.items()})
    calls = {"judge_calls": 727 + judged + 0.85 * judged + (completeness_answers if has_reference_condition else 0)
                                                   + lean_prompts * (2 if has_reference_condition else 1)}
    return minutes, regenerated, calls["judge_calls"]


def format_self_judging_estimate_lines(number_of_prompts, has_reference_condition, completeness_sample_size=None,
                                       user_lean_sample_prompts=None):
    """Pod time and cost for the whole session: setup, judge validation, generation, judging, regeneration, phases 5 and 6."""
    lines = [f"Estimate for the whole self-judging session on one H200 at ${ESTIMATE_POD_DOLLARS_PER_HOUR}/hour (phases 0 to 6; phase 5 judges "
             f"{'every kept answer' if completeness_sample_size is None else f'a sample of {completeness_sample_size:,} kept answers'}, phase 6 "
             f"{'every prompt' if user_lean_sample_prompts is None else f'a sample of {user_lean_sample_prompts:,} prompts'} with both conditions' answers):"]
    for label, tokens, rate, failure_share in (("central", ESTIMATE_MEAN_ANSWER_TOKENS, ESTIMATE_BATCHED_TOKENS_PER_SECOND, 0.20),
                                               ("conservative", ESTIMATE_CONSERVATIVE_MEAN_ANSWER_TOKENS,
                                                ESTIMATE_CONSERVATIVE_TOKENS_PER_SECOND, 0.35)):
        minutes, regenerated, judge_calls = estimate_self_judging_session(
            number_of_prompts, has_reference_condition, tokens, rate, ESTIMATE_PREFILL_TOKENS_PER_SECOND[label], failure_share,
            completeness_sample_size, user_lean_sample_prompts)
        total = sum(minutes.values())
        judging = sum(value for name, value in minutes.items() if name.startswith("phase"))
        lines.append(f"  {label:12s}: about {total:.0f} minutes ({total / 60:.1f} hours), about ${total / 60 * ESTIMATE_POD_DOLLARS_PER_HOUR:.2f}; "
                     f"{minutes['setup']:.0f} setup, {minutes['generation of first answers and regeneration']:.0f} generating "
                     f"({tokens} tokens per answer at {rate} tokens per second, about {regenerated:,.0f} regenerated answers, "
                     f"{failure_share:.0%} of prompts assumed to fail the first judge), {judging:.0f} reading judge prompts "
                     f"(about {judge_calls:,.0f} judge calls, prefill {ESTIMATE_PREFILL_TOKENS_PER_SECOND[label]:,} tokens per second)")
    lines.append("  The failure shares, the 400-token user message and the prefill rates are assumptions: no Gemma judge has run yet. "
                 "Phases 5 and 6 can be left for a second pod (the script resumes). Suggested limits for the whole session: 5 hours and 30 dollars "
                 "(7 hours and 45 dollars if phases 5 and 6 judge everything).")
    return lines


def build_run_folder(runs_root, model_short_name, prompt_pool_file, held_out_set_file, system_prompt_file, answers_per_prompt,
                     number_of_prompts=None, subsample_seed=DEFAULT_SUBSAMPLE_SEED, timestamp=None,
                     fetch_text_function=fetch_text, fixture_path=None, include_no_system_prompt_reference_condition=True,
                     held_out_answers_run_folder=None, completeness_judge_sample_size=DEFAULT_COMPLETENESS_JUDGE_SAMPLE_SIZE,
                     user_lean_judge_sample_prompts=DEFAULT_USER_LEAN_JUDGE_SAMPLE_PROMPTS, judge_sample_seed=DEFAULT_SUBSAMPLE_SEED):
    if model_short_name not in MODEL_PINS:
        raise SystemExit(f"--model-short-name must be one of {sorted(MODEL_PINS)}")
    if answers_per_prompt < 1:
        raise SystemExit("--answers-per-prompt must be at least 1")
    timestamp = timestamp or timestamp_now()
    suffix = RUN_FOLDER_NAME_SUFFIX_WITH_REFERENCE_ANSWERS if include_no_system_prompt_reference_condition else RUN_FOLDER_NAME_SUFFIX
    run_folder = Path(runs_root) / f"{RUN_FOLDER_NAME_PREFIX}_{model_short_name}_{suffix}_{timestamp}"
    if run_folder.exists():
        raise SystemExit(f"{run_folder} already exists")
    # Everything that can fail is checked before the folder is created.
    pool_records = read_jsonl(prompt_pool_file)
    held_out_records = read_jsonl(held_out_set_file)
    overlaps = find_held_out_overlaps(pool_records, held_out_records)
    if overlaps:
        raise SystemExit(f"The prompt pool {prompt_pool_file} contains {len(overlaps)} held-out prompt(s); filter them out first:\n  "
                         + "\n  ".join(overlaps[:20]))
    held_out_answers_run_folder = Path(held_out_answers_run_folder or REPOSITORY_ROOT / DEFAULT_HELD_OUT_ANSWERS_RUN_FOLDER_RELATIVE_PATH)
    reference_records, reference_notes = judging_rules.build_phase_0_reference_records(held_out_answers_run_folder)
    if not reference_records:
        raise SystemExit(f"No held-out thinking-off answers with Haiku verdicts found in {held_out_answers_run_folder}")
    pool_prompts = map_pool_records_onto_pod_prompt_records(pool_records)
    prompts = choose_prompts(pool_prompts, number_of_prompts, subsample_seed)
    system_prompt_text, system_prompt_file_sha256, system_prompt_text_sha256 = read_system_prompt_text(system_prompt_file)
    pins = MODEL_PINS[model_short_name]
    model_repository = defs.candidate_model_by_short_name(model_short_name)["model_repository"]
    facts = fetch_model_facts_from_hugging_face(model_repository, fetch_text_function, pins["revision"])
    print(f"{model_repository}: pinned revision {facts['model_revision']}", flush=True)
    fixture_sha256 = verify_chat_template_matches_fixture(facts["chat_template_sha256"],
                                                          fixture_path or REPOSITORY_ROOT / pins["fixture_relative_path"])
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    inputs_folder.mkdir(parents=True)
    prompts_destination = inputs_folder / f"treatment_arm_training_prompts_with_merged_corpus_prompt_id_as_prompt_gid_{timestamp}.jsonl"
    write_jsonl(prompts_destination, prompts)
    system_prompted_condition = build_condition(model_short_name, system_prompt_file, system_prompt_text, system_prompt_text_sha256,
                                                system_prompt_file_sha256)
    conditions = [system_prompted_condition]
    if include_no_system_prompt_reference_condition:
        conditions.append(build_condition_without_system_prompt(system_prompted_condition))
    self_judging_section = build_self_judging_section(inputs_folder, timestamp, reference_records, reference_notes,
                                                      held_out_answers_run_folder, held_out_records, held_out_set_file,
                                                      answers_per_prompt, include_no_system_prompt_reference_condition,
                                                      completeness_judge_sample_size, user_lean_judge_sample_prompts, judge_sample_seed)
    arguments_record = {"answers_per_prompt": answers_per_prompt, "number_of_prompts": number_of_prompts,
                        "subsample_seed": subsample_seed, "held_out_set_file": str(held_out_set_file)}
    configuration = build_run_configuration(run_folder, timestamp, arguments_record, prompts_destination, prompt_pool_file,
                                            len(pool_prompts), len(prompts), model_short_name, facts, conditions, fixture_sha256,
                                            self_judging_section)
    configuration_path = write_run_configuration(run_folder, timestamp, configuration)
    print(f"Run folder: {run_folder}\nRun configuration: {configuration_path}", flush=True)
    print(f"{len(prompts):,} prompts of a pool of {len(pool_prompts):,}, {answers_per_prompt} answers each = "
          f"{len(prompts) * answers_per_prompt:,} answers", flush=True)
    print("\n".join(format_estimate_lines(model_short_name, len(prompts), answers_per_prompt)), flush=True)
    print("\n".join(format_self_judging_estimate_lines(len(prompts), include_no_system_prompt_reference_condition,
                                                                          completeness_judge_sample_size, user_lean_judge_sample_prompts)), flush=True)
    if reference_notes["non_raising_answers_without_a_choice_verdict"]:
        print(f"Note: {len(reference_notes['non_raising_answers_without_a_choice_verdict'])} held-out answers with raised_welfare_cost no "
              f"have no Haiku choice verdict; the agreement analysis leaves them out.", flush=True)
    relative_folder = run_folder.relative_to(REPOSITORY_ROOT) if run_folder.is_relative_to(REPOSITORY_ROOT) else run_folder
    print("Dry run of the launch (free):\n  python3 scripts/emergent_misalignment/launch_and_stop_runpod_baseline_answer_generation_pod.py "
          f"--run-folder {relative_folder} --maximum-running-hours 5 --maximum-dollars 30 --dry-run", flush=True)
    return run_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-short-name", required=True, choices=sorted(MODEL_PINS))
    parser.add_argument("--prompt-pool-file", required=True, help="combined or filtered snapshot jsonl")
    parser.add_argument("--answers-per-prompt", type=int, default=1,
                        help="first system-prompted answers per prompt before any regeneration (default 1)")
    parser.add_argument("--without-no-system-prompt-reference-answers", action="store_true",
                        help="leave out the thinking_off condition with no system prompt")
    parser.add_argument("--completeness-judge-sample-size", default=str(DEFAULT_COMPLETENESS_JUDGE_SAMPLE_SIZE),
                        help="phase 5 judges a seeded sample of this many kept answers; 0 or all judges every kept answer (default 1000)")
    parser.add_argument("--user-lean-judge-sample-prompts", default=str(DEFAULT_USER_LEAN_JUDGE_SAMPLE_PROMPTS),
                        help="phase 6 judges both conditions' answers for a seeded sample of this many prompts; 0 or all judges everything (default 1500)")
    parser.add_argument("--judge-sample-seed", type=int, default=DEFAULT_SUBSAMPLE_SEED)
    parser.add_argument("--held-out-answers-run-folder", default=None,
                        help="the held-out run whose thinking-off answers and Haiku verdicts validate the Gemma judge")
    parser.add_argument("--number-of-prompts", type=int, default=None, help="seeded subsample of the pool; default all")
    parser.add_argument("--subsample-seed", type=int, default=DEFAULT_SUBSAMPLE_SEED)
    parser.add_argument("--held-out-file", default=str(REPOSITORY_ROOT / defs.HELD_OUT_SET_FILE_RELATIVE_PATH))
    parser.add_argument("--system-prompt-file", default=str(REPOSITORY_ROOT / defs.SYSTEM_PROMPT_FILE_RELATIVE_PATH))
    parser.add_argument("--runs-root", default=str(REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH))
    arguments = parser.parse_args(argv)
    build_run_folder(arguments.runs_root, arguments.model_short_name, arguments.prompt_pool_file, arguments.held_out_file,
                     arguments.system_prompt_file, arguments.answers_per_prompt, arguments.number_of_prompts, arguments.subsample_seed,
                     include_no_system_prompt_reference_condition=not arguments.without_no_system_prompt_reference_answers,
                     held_out_answers_run_folder=arguments.held_out_answers_run_folder,
                     completeness_judge_sample_size=parse_sample_size(arguments.completeness_judge_sample_size),
                     user_lean_judge_sample_prompts=parse_sample_size(arguments.user_lean_judge_sample_prompts),
                     judge_sample_seed=arguments.judge_sample_seed)


if __name__ == "__main__":
    main()
