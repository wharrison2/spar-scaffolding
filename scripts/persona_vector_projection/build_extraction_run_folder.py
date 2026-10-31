"""Builds a run folder for the extraction half of the persona vector experiment from the converted inputs.

Free and local. Pilot mode: non-intrusive facet plus topic control, with an optional cap on questions per half. Full
mode: all four facets plus topic control. Every (extraction or held-out question) x (system prompt of the facet) pair
becomes one generation job of `samples_per_prompt` answers. Consistent: both versions of every pair. Topic control:
both versions of every pair under the generic assistant (length suffix only).

    python3 scripts/persona_vector_projection/build_extraction_run_folder.py --mode pilot --run-name <name> \
        [--maximum-questions-per-half 6] [--small-sample-check]

Output: <run folder>/generation_inputs/{generation_jobs,system_prompts,questions}_<timestamp>.jsonl, facet_definitions_<timestamp>.json and
run_configuration_<timestamp>.json (model, pinned revision, sampling settings, SHA-256 of every input file).
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import choose_run_folder, sha256_of_file, timestamp_now, write_run_configuration  # noqa: E402

MODE_PILOT = "pilot"
MODE_FULL = "full"
FACETS_BY_MODE = {MODE_PILOT: defs.PILOT_FACETS, MODE_FULL: defs.ALL_FACETS}


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


def build_generation_jobs(question_records, system_prompt_records, samples_per_prompt):
    """One job per (question record, system prompt of the question's facet). A job's answers share one prompt."""
    prompts_by_facet = {}
    for prompt in system_prompt_records:
        prompts_by_facet.setdefault(prompt["facet"], []).append(prompt)
    jobs = []
    for question in question_records:
        for prompt in prompts_by_facet[question["facet"]]:
            jobs.append({
                "job_id": f"{prompt['system_prompt_id']}__{question['question_id']}",
                "facet": question["facet"], "pole": prompt["pole"], "tag": prompt["tag"],
                "system_prompt_id": prompt["system_prompt_id"], "system_prompt": prompt["system_prompt"],
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
        "generation_backend": arguments.generation_backend, "random_seed": arguments.random_seed,
        "number_of_generation_jobs": len(jobs), "number_of_answers_by_facet": count_answers_by_facet(jobs),
        "converted_inputs_folder": str(converted_inputs_folder),
        "input_file_sha256": {path.name: sha256_of_file(path) for path in sorted(inputs_folder.glob("*"))},
        "prompt_file_sha256": {path.name: sha256_of_file(path) for path in sorted(defs.PROMPT_FOLDER.glob("*.txt"))}}
    write_run_configuration(run_folder, timestamp, configuration)
    return run_folder, configuration


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=[MODE_PILOT, MODE_FULL], required=True)
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
        facet_text = "non_intrusive_and_topic_control" if arguments.mode == MODE_PILOT else "four_facets_and_topic_control"
        arguments.run_name = f"persona_vector_extraction_{arguments.mode}_{facet_text}_gemma_4_12b_it"
    return arguments


if __name__ == "__main__":
    run_folder_path, run_configuration = build_run_folder(parse_arguments())
    print(f"Run folder: {run_folder_path}")
    print(json.dumps({key: run_configuration[key] for key in ("mode", "facets", "number_of_generation_jobs",
                                                              "number_of_answers_by_facet")}, indent=2))
