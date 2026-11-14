"""Builds a run folder for the extraction half of the persona vector experiment from the converted inputs.

Free and local. Pilot mode: non-intrusive facet plus topic control, with an optional cap on questions per half. Full
mode: all four facets plus topic control. Content-controlled facet modes (`cares_about_welfare_content_controlled_pilot`,
`calibrated_consideration_content_controlled_pilot`, `non_intrusive_content_controlled_pilot`) behave like the moralizing delivery
mode below, except that non-intrusive records only whole-answer means. Moralizing delivery pilot mode (`--mode moralizing_delivery_pilot`): only the
moralizing delivery facet, no topic control; each system prompt's {welfare_point} is filled per question, and the
activation pass renders every answer under one neutral system prompt (run configuration field
`activation_pass_system_prompt`; null in every other mode, meaning the generation prompt). Every (extraction or held-out question) x (system prompt of the facet) pair
becomes one generation job of `samples_per_prompt` answers. Consistent: both versions of every pair. Topic control:
both versions of every pair under the generic assistant (length suffix only).

    python3 scripts/persona_vector_projection/build_extraction_run_folder.py --mode pilot --run-name <name> \
        [--maximum-questions-per-half 6] [--small-sample-check]

Output: <run folder>/generation_inputs/{generation_jobs,system_prompts,questions}_<timestamp>.jsonl, facet_definitions_<timestamp>.json and
run_configuration_<timestamp>.json (model, pinned revision, sampling settings, SHA-256 of every input file).
"""

import argparse
import json
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import choose_run_folder, sha256_of_file, timestamp_now, write_run_configuration  # noqa: E402

MODE_PILOT = "pilot"
MODE_FULL = "full"
MODE_MORALIZING_DELIVERY_PILOT = "moralizing_delivery_pilot"
FACETS_BY_MODE = {MODE_PILOT: defs.PILOT_FACETS, MODE_FULL: defs.ALL_FACETS,
                  MODE_MORALIZING_DELIVERY_PILOT: [defs.FACET_MORALIZING_DELIVERY]}
# One mode per content-controlled facet: `<facet key>_pilot` (moralizing delivery's is the existing mode above).
for _facet_key in facet_definitions_module.NEW_CONTENT_CONTROLLED_FACET_KEYS:
    FACETS_BY_MODE[facet_definitions_module.get_facet_definition(_facet_key).extraction_mode] = [_facet_key]
CONTENT_CONTROLLED_FACET_BY_MODE = {facet_definitions_module.get_facet_definition(key).extraction_mode:
                                    facet_definitions_module.get_facet_definition(key)
                                    for key in facet_definitions_module.ALL_CONTENT_CONTROLLED_FACET_KEYS}
NEUTRAL_ACTIVATION_PROMPT_FILE_NAME = "activation_pass_neutral_system_prompt_moralizing_delivery.txt"


def latest_converted_inputs_folder():
    candidates = sorted(defs.CONVERTED_INPUTS_ROOT.glob("converted_inputs_*"))
    if not candidates:
        raise SystemExit(f"No converted inputs in {defs.CONVERTED_INPUTS_ROOT}; run "
                         f"convert_drafts_into_machine_readable_inputs.py first")
    return candidates[-1]


def select_items_round_robin_over_domains(question_records, maximum_items):
    """Keeps at most `maximum_items` question items (a pair counts once) taking domains in turn, so a cap keeps the
    domain balance as far as it can. Returns the surviving question records in their original order."""
    item_key = lambda record: record.get("pair_id") or record["question_id"]  # noqa: E731
    items_by_domain = OrderedDict()
    for record in question_records:
        domain_items = items_by_domain.setdefault(record["domain"], OrderedDict())
        domain_items.setdefault(item_key(record), None)
    chosen, round_index = set(), 0
    while len(chosen) < maximum_items and any(len(items) > round_index for items in items_by_domain.values()):
        for items in items_by_domain.values():
            if len(items) > round_index and len(chosen) < maximum_items:
                chosen.add(list(items)[round_index])
        round_index += 1
    return [record for record in question_records if item_key(record) in chosen]


def select_questions(question_records, facets, maximum_questions_per_half):
    selected = []
    for facet in facets:
        for half in defs.ALL_HALVES:
            records = [r for r in question_records if r["facet"] == facet and r["half"] == half]
            if maximum_questions_per_half is not None:
                records = select_items_round_robin_over_domains(records, maximum_questions_per_half)
            selected.extend(records)
    return selected


def fill_welfare_point(prompt, question):
    """The prompt's system prompt with {welfare_point} replaced by the question's welfare point. Raises if the question
    has no welfare point, or any {placeholder} is left in the result."""
    point = (question.get("labels") or {}).get("welfare_point")
    if not point or not point.strip():
        raise ValueError(f"Question {question['question_id']} has no welfare point to fill into {prompt['system_prompt_id']}")
    filled = prompt["system_prompt"].replace(defs.WELFARE_POINT_PLACEHOLDER, point.strip())
    leftover = re.findall(r"\{[a-z_]+\}", filled)
    if leftover:
        raise ValueError(f"Placeholder(s) {leftover} left unfilled in {prompt['system_prompt_id']} for {question['question_id']}")
    return filled


def fill_content_slots(prompt, question):
    """The prompt's system prompt with every {slot} that the question's facet defines (welfare_point, welfare_content,
    evidence_content, recommendation) replaced by the question's own content. Raises when the question lacks the content or any
    {placeholder} is left in the result. The same content goes into both poles."""
    definition = facet_definitions_module.get_facet_definition(question["facet"])
    slots = [slot for slot in definition.question_label_key_by_slot if slot != "evidence_band" and "{" + slot + "}" in prompt["system_prompt"]]
    values = definition.question_labels_for_slots(question, slots)
    filled = prompt["system_prompt"]
    for slot in slots:
        filled = filled.replace("{" + slot + "}", "\x00" + slot + "\x01")
    for slot in slots:
        filled = filled.replace("\x00" + slot + "\x01", values[slot])
    leftover = re.findall(r"\{[a-z_]+\}", filled)
    if leftover:
        raise ValueError(f"Placeholder(s) {leftover} left unfilled in {prompt['system_prompt_id']} for {question['question_id']}")
    return filled


def choose_negative_prompts_for_each_question(question_records, negative_prompts, positive_rewording_count, definition):
    """{question id: [negative prompt records]} for a facet with evidence band rules (calibrated consideration): exactly one negative
    rewording per pairing slot (the positive rewording whose sentence structure it shares), so every question gets as many
    negative answers as positive ones. A negative rewording is a candidate for a question only where it is miscalibrated for the
    question's evidence band (definition.negative_rewording_is_miscalibrated_for_question). Where a slot has two candidates the
    one used less often so far is taken (ties broken by a seeded generator keyed on the question id), so over the run each allowed
    rewording is used about equally often. Raises when a slot has no allowed rewording for a question (never fills it with a
    rewording that is accurate for that band)."""
    slots = {}
    for prompt in negative_prompts:
        slots.setdefault(facet_definitions_module.pairing_slot_of_rewording(prompt["rewording_number"], positive_rewording_count),
                         []).append(prompt)
    if sorted(slots) != list(range(1, positive_rewording_count + 1)):
        raise ValueError(f"The negative rewordings cover pairing slots {sorted(slots)}, expected 1 to {positive_rewording_count}")
    times_used = {prompt["system_prompt_id"]: 0 for prompt in negative_prompts}
    chosen_by_question = {}
    for question in sorted(question_records, key=lambda record: record["question_id"]):
        generator = random.Random(f"negative rewording choice {question['question_id']}")
        chosen = []
        for slot in range(1, positive_rewording_count + 1):
            allowed = [prompt for prompt in slots[slot]
                       if definition.negative_rewording_is_miscalibrated_for_question(prompt["tag"], question)]
            if not allowed:
                raise ValueError(f"No negative rewording in slot {slot} is miscalibrated for question {question['question_id']} "
                                 f"(evidence band {question['labels'].get('evidence_band')!r}); candidates "
                                 f"{[(p['rewording_number'], p['tag']) for p in slots[slot]]}")
            fewest = min(times_used[prompt["system_prompt_id"]] for prompt in allowed)
            pick = generator.choice(sorted((p for p in allowed if times_used[p["system_prompt_id"]] == fewest),
                                           key=lambda p: p["rewording_number"]))
            times_used[pick["system_prompt_id"]] += 1
            chosen.append(pick)
        chosen_by_question[question["question_id"]] = chosen
    return chosen_by_question


def build_generation_jobs(question_records, system_prompt_records, samples_per_prompt):
    """One job per (question record, system prompt of the question's facet). A job's answers share one prompt. Prompts
    with a {welfare_point} placeholder are filled from the question (loud failure if that is not possible)."""
    prompts_by_facet = {}
    for prompt in system_prompt_records:
        prompts_by_facet.setdefault(prompt["facet"], []).append(prompt)
    chosen_negatives_by_facet = {}
    for facet, facet_prompts in prompts_by_facet.items():
        if not facet_definitions_module.is_content_controlled_facet(facet):
            continue
        definition = facet_definitions_module.get_facet_definition(facet)
        if definition.uses_evidence_band_rules_for_negative_rewordings:
            positive_count = sum(1 for prompt in facet_prompts if prompt["pole"] == defs.POLE_POSITIVE)
            chosen_negatives_by_facet[facet] = choose_negative_prompts_for_each_question(
                [q for q in question_records if q["facet"] == facet],
                [p for p in facet_prompts if p["pole"] == defs.POLE_NEGATIVE], positive_count, definition)
    jobs = []
    for question in question_records:
        prompts_for_question = prompts_by_facet[question["facet"]]
        if question["facet"] in chosen_negatives_by_facet:
            chosen_ids = {p["system_prompt_id"] for p in chosen_negatives_by_facet[question["facet"]][question["question_id"]]}
            prompts_for_question = [p for p in prompts_for_question if p["pole"] != defs.POLE_NEGATIVE or p["system_prompt_id"] in chosen_ids]
        for prompt in prompts_for_question:
            if facet_definitions_module.is_content_controlled_facet(question["facet"]) and question["facet"] != defs.FACET_MORALIZING_DELIVERY:
                system_prompt = fill_content_slots(prompt, question)
            else:
                needs_welfare_point = prompt.get("system_prompt_requires_welfare_point") or defs.WELFARE_POINT_PLACEHOLDER in prompt["system_prompt"]
                system_prompt = fill_welfare_point(prompt, question) if needs_welfare_point else prompt["system_prompt"]
            jobs.append({
                "job_id": f"{prompt['system_prompt_id']}__{question['question_id']}",
                "facet": question["facet"], "pole": prompt["pole"], "tag": prompt["tag"],
                "system_prompt_id": prompt["system_prompt_id"], "system_prompt": system_prompt,
                "question_id": question["question_id"], "pair_id": question.get("pair_id"),
                "version": question.get("version"), "half": question["half"], "domain": question["domain"],
                "user_prompt": question["question_text"], "samples_per_prompt": samples_per_prompt})
    return jobs


def count_answers_by_facet(jobs):
    counts = {}
    for job in jobs:
        counts[job["facet"]] = counts.get(job["facet"], 0) + job["samples_per_prompt"]
    return counts


def build_run_folder(arguments, converted_inputs_folder=None):
    converted_inputs_folder = Path(converted_inputs_folder or arguments.converted_inputs_folder or latest_converted_inputs_folder())
    facets = FACETS_BY_MODE[arguments.mode]
    questions = select_questions(read_jsonl(defs.find_latest_file(converted_inputs_folder, "questions", "jsonl")), facets,
                                 arguments.maximum_questions_per_half)
    system_prompts = [p for p in read_jsonl(defs.find_latest_file(converted_inputs_folder, "system_prompts", "jsonl")) if p["facet"] in facets]
    jobs = build_generation_jobs(questions, system_prompts, arguments.samples_per_prompt)
    activation_pass_system_prompt = None  # default: the activation pass uses each job's own generation system prompt
    facet_definition = CONTENT_CONTROLLED_FACET_BY_MODE.get(arguments.mode)
    if facet_definition is not None:
        activation_pass_system_prompt = (defs.PROMPT_FOLDER / NEUTRAL_ACTIVATION_PROMPT_FILE_NAME).read_text(encoding="utf-8").strip()
        if not activation_pass_system_prompt:
            raise SystemExit(f"{NEUTRAL_ACTIVATION_PROMPT_FILE_NAME} is empty; run the converter first")
    runs_directory = defs.SMALL_SAMPLE_CHECKS_ROOT if arguments.small_sample_check else defs.RUNS_ROOT
    run_folder, is_resume = choose_run_folder(runs_directory, arguments.run_name, arguments.resume_folder)
    if is_resume:
        raise SystemExit("build_extraction_run_folder.py creates new run folders; resume by passing the existing "
                         "folder to the pod launcher or the later scripts")
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    inputs_folder.mkdir()
    timestamp = timestamp_now()
    write_jsonl(inputs_folder / f"generation_jobs_{timestamp}.jsonl", jobs)
    write_jsonl(inputs_folder / f"system_prompts_{timestamp}.jsonl", system_prompts)
    write_jsonl(inputs_folder / f"questions_{timestamp}.jsonl", questions)
    (inputs_folder / f"facet_definitions_{timestamp}.json").write_text(
        defs.find_latest_file(converted_inputs_folder, "facet_definitions", "json").read_text(encoding="utf-8"),
        encoding="utf-8")
    configuration = {
        "run_kind": "persona_vector_extraction", "mode": arguments.mode, "facets": facets,
        "maximum_questions_per_half": arguments.maximum_questions_per_half,
        "model_name": arguments.model_name, "model_revision": arguments.model_revision, "model_type": "instruct",
        "samples_per_prompt": arguments.samples_per_prompt, "maximum_new_tokens": defs.MAXIMUM_NEW_TOKENS,
        "temperature": defs.SAMPLING_TEMPERATURE, "top_p": defs.SAMPLING_TOP_P,
        "activation_pass_system_prompt": activation_pass_system_prompt,
        # False for non-intrusive: its main mean is the whole-answer mean, so no spans file is needed or read
        "record_welfare_passage_mean_activations": facet_definition is not None and facet_definition.uses_welfare_passage,
        "facet_score_scale": facet_definition.facet_score_scale_label() if facet_definition is not None else None,
        "main_activation_mean_space": facet_definition.main_space if facet_definition is not None else None,
        "generation_backend": arguments.generation_backend, "random_seed": arguments.random_seed,
        "number_of_generation_jobs": len(jobs), "number_of_answers_by_facet": count_answers_by_facet(jobs),
        "converted_inputs_folder": str(converted_inputs_folder),
        "input_file_sha256": {path.name: sha256_of_file(path) for path in sorted(inputs_folder.glob("*"))},
        "prompt_file_sha256": {path.name: sha256_of_file(path) for path in sorted(defs.PROMPT_FOLDER.glob("*.txt"))}}
    write_run_configuration(run_folder, timestamp, configuration)
    return run_folder, configuration


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=list(FACETS_BY_MODE), required=True)
    parser.add_argument("--run-name", default=None, help="Default: a descriptive name from the mode and model")
    parser.add_argument("--maximum-questions-per-half", type=int, default=None,
                        help="Cap on questions (or pairs) per facet in each of half A, half B and held-out")
    parser.add_argument("--samples-per-prompt", type=int, default=defs.SAMPLES_PER_PROMPT_AND_QUESTION)
    parser.add_argument("--model-name", default=defs.DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=defs.DEFAULT_MODEL_REVISION)
    parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default="vllm")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--converted-inputs-folder", default=None)
    parser.add_argument("--small-sample-check", action="store_true")
    parser.add_argument("--resume-folder", default=None)
    arguments = parser.parse_args(argv)
    if arguments.run_name is None:
        if arguments.mode in CONTENT_CONTROLLED_FACET_BY_MODE:
            arguments.run_name = CONTENT_CONTROLLED_FACET_BY_MODE[arguments.mode].extraction_run_name
        else:
            facet_text = "non_intrusive_and_topic_control" if arguments.mode == MODE_PILOT else "four_facets_and_topic_control"
            arguments.run_name = f"persona_vector_extraction_{arguments.mode}_{facet_text}_gemma_4_12b_it"
    return arguments


if __name__ == "__main__":
    run_folder_path, run_configuration = build_run_folder(parse_arguments())
    print(f"Run folder: {run_folder_path}")
    print(json.dumps({key: run_configuration[key] for key in ("mode", "facets", "number_of_generation_jobs",
                                                              "number_of_answers_by_facet", "activation_pass_system_prompt")}, indent=2))
