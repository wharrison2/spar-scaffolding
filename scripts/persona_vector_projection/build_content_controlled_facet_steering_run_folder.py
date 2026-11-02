"""Builds the steering-only run folder for the steering check of one content-controlled facet (moralizing delivery, cares about
welfare, calibrated consideration or non-intrusive; `--facet`, default moralizing delivery), for the scoring and steering pod
launcher (stage `steering` of run_scoring_pod_stages.py). Free and local: no model, API or pod is touched.

    python3 scripts/persona_vector_projection/build_content_controlled_facet_steering_run_folder.py --facet <facet name> \
        --extraction-run-folder <that facet's extraction run folder> \
        --vector-set-folder <extraction run folder>/vector_sets/<name>_<timestamp with seconds> --dry-run
    ... without --dry-run to write the folder.

What is steered. The held-out questions of the facet (moralizing delivery and non-intrusive: 6; cares about welfare and
calibrated consideration: 12), each under ONE facet-unspecified system prompt (the definition's steering_system_prompt_file in
prompts/) with every slot filled from that question's own labels, 10 samples each at temperature 1, top-p 1. Conditions: one shared unsteered condition, then for each of
two vectors (the facet's main-space vector, passage mean for moralizing delivery, cares about welfare and calibrated
consideration and whole-answer mean for non-intrusive, and that vector with its top k largest-mean dimensions zeroed, k = 4 by
default) every nonzero fraction of the list. The vector added is  fraction x median activation norm x unit vector  at the
vector set's chosen layer, at all positions (see content_controlled_facet_steering_vectors_and_coefficients.py). The vector
is positive pole minus negative pole, so positive fractions push towards the positive pole.

Writes <run folder>/scoring_pod_inputs/: steering_jobs.jsonl, steering_questions.jsonl, facet_definitions.json (empty
object, read by the judge's command line), steering_system_prompt_template.txt, steering_conditions_<ts>.json,
steering_vector_manifest_<ts>.json and steering_vectors/ (the two vectors as stored), and run_configuration_<ts>.json in the
run folder. The run folder's configuration has the keys the launcher's estimate and the pod script read.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module  # noqa: E402
from persona_vector_projection import estimate_content_controlled_facet_steering_cost_ceilings as estimator  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import content_controlled_facet_steering_vectors_and_coefficients as coefficients_module  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as pod_defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import choose_run_folder, sha256_of_file, timestamp_now, write_run_configuration  # noqa: E402

DEFAULT_FACET_DEFINITION = facet_definitions_module.MORALIZING_DELIVERY
FACET = DEFAULT_FACET_DEFINITION.facet_key  # the default facet; every function takes the definition as an optional argument
SYSTEM_PROMPT_TEMPLATE_FILE_NAME = DEFAULT_FACET_DEFINITION.steering_system_prompt_file
DEFAULT_SAMPLES_PER_QUESTION = 10
DEFAULT_ZEROED_DIMENSION_COUNT = 4
MAIN_VECTOR_NAME = "main_vector"
RUN_NAME = DEFAULT_FACET_DEFINITION.steering_run_name
MODE = DEFAULT_FACET_DEFINITION.steering_mode


def zeroed_vector_name(zeroed_dimension_count):
    return f"top_{zeroed_dimension_count}_largest_mean_dimensions_zeroed_vector"


def fill_steering_system_prompt(template_text, welfare_point=None, question_record=None, definition=None):
    """The template with every `{slot}` filled. Old form: `fill_steering_system_prompt(template, welfare_point)` fills
    `{welfare_point}` only. Facet form: `fill_steering_system_prompt(template, question_record=question, definition=definition)`
    fills each slot of the template from the question's labels through the definition. Raises if the template lacks a
    placeholder, a content label is missing or empty, or any `{slot}` is left unfilled (the judge's fill_slots, which also leaves
    braces inside the content alone)."""
    if question_record is None:
        if defs.WELFARE_POINT_PLACEHOLDER not in template_text:
            raise ValueError("The steering system prompt template has no {welfare_point} placeholder")
        if not welfare_point or not str(welfare_point).strip():
            raise ValueError("Empty welfare point")
        return judge.fill_slots(template_text.strip(), {"welfare_point": str(welfare_point).strip()})
    definition = definition or DEFAULT_FACET_DEFINITION
    slot_names = [name[1:-1] for name in judge.SLOT_PATTERN.findall(template_text)]
    if not slot_names:
        raise ValueError("The steering system prompt template has no content placeholder")
    missing_from_definition = [name for name in slot_names if name not in definition.question_label_key_by_slot]
    if missing_from_definition:
        raise ValueError(f"The steering system prompt template has slots {missing_from_definition} unknown to {definition.facet_key}")
    return judge.fill_slots(template_text.strip(), definition.question_labels_for_slots(question_record, slot_names))


def build_steering_conditions(vector_names, fractions):
    """One shared unsteered condition (fraction 0, no vector), then per vector each nonzero fraction, most negative first.
    The coefficient is filled in later, once the median activation norm is known."""
    conditions = [{"condition_id": pod_defs.STEERING_CONDITION_UNSTEERED, "vector_name": None, "fraction": 0.0}]
    for vector_name in vector_names:
        for fraction in sorted(set(fractions)):
            if fraction == 0:
                continue  # the zero fraction is the shared unsteered condition
            conditions.append({"condition_id": f"{vector_name}__fraction_{coefficients_module.fraction_label(fraction)}",
                               "vector_name": vector_name, "fraction": float(fraction)})
    return conditions


def build_steering_jobs(question_records, conditions, median_activation_norm, system_prompt_template, samples_per_question,
                        definition=None):
    """One job per condition and held-out question, in the layout run_steering_stage reads. `facets_and_coefficients` maps the
    vector's name to the absolute amount added (fraction x median norm; the pod's vector is a unit vector)."""
    definition = definition or DEFAULT_FACET_DEFINITION
    questions = sorted(question_records, key=lambda question: question["question_id"])
    jobs = []
    for condition in conditions:
        facets_and_coefficients = ({} if condition["vector_name"] is None else {
            condition["vector_name"]: coefficients_module.steering_coefficient(condition["fraction"], median_activation_norm)})
        for question in questions:
            jobs.append({"job_id": f"steering__{condition['condition_id']}__{question['question_id']}",
                         "steering_condition_id": condition["condition_id"], "facets_and_coefficients": facets_and_coefficients,
                         "facet": definition.facet_key, "question_id": question["question_id"], "pair_id": None, "version": None,
                         "half": question["half"], "domain": question["domain"],
                         "system_prompt": fill_steering_system_prompt(system_prompt_template, question_record=question, definition=definition),
                         "user_prompt": question["question_text"], "samples_per_prompt": samples_per_question})
    return jobs


def read_vector_set(vector_set_folder, vector_space, zeroed_dimension_count, definition=None):
    """The chosen layer and the two vector files (name -> path) of the facet's vector set."""
    definition = definition or DEFAULT_FACET_DEFINITION
    folder = Path(vector_set_folder)
    metadata_paths = sorted(folder.glob("persona_vector_metadata_*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No persona_vector_metadata_*.json in {folder}")
    metadata = json.loads(metadata_paths[-1].read_text(encoding="utf-8"))
    layer = metadata["layer_choice"]["layer_by_facet"][definition.facet_key]
    keys = {MAIN_VECTOR_NAME: f"{vector_space}_raw",
            zeroed_vector_name(zeroed_dimension_count): f"{vector_space}_top_{zeroed_dimension_count}_largest_mean_dimensions_zeroed"}
    missing = [key for key in keys.values() if key not in metadata["vector_files"]]
    if missing:
        raise KeyError(f"The vector set {folder} has no vector files {missing}; it has {sorted(metadata['vector_files'])}")
    return int(layer), {name: folder / metadata["vector_files"][key] for name, key in keys.items()}, metadata, metadata_paths[-1]


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    facet_definitions_module.add_facet_argument(parser)
    parser.add_argument("--extraction-run-folder", required=True,
                        help="The facet's extraction run (its questions, activations and model are used)")
    parser.add_argument("--vector-set-folder", required=True, help="vector_sets/<name>_<timestamp> of that run")
    parser.add_argument("--vector-space", choices=["passage_mean", "whole_answer_mean"], default=None,
                        help="Default: the facet's main space (passage mean; whole-answer mean for non-intrusive)")
    parser.add_argument("--zeroed-dimension-count", type=int, default=DEFAULT_ZEROED_DIMENSION_COUNT,
                        choices=[1, 2, 4, 8, 16, 32], help="k of the second vector: the top k largest-mean dimensions zeroed")
    parser.add_argument("--steering-fractions", nargs="+", type=float, default=coefficients_module.DEFAULT_STEERING_FRACTIONS,
                        help="Fractions of the median activation norm; 0 is the shared unsteered condition")
    parser.add_argument("--samples-per-question", type=int, default=DEFAULT_SAMPLES_PER_QUESTION)
    parser.add_argument("--steering-positions", choices=[pod_defs.STEERING_POSITIONS_ALL, pod_defs.STEERING_POSITIONS_LAST],
                        default=pod_defs.STEERING_POSITIONS_ALL)
    parser.add_argument("--maximum-new-tokens", type=int, default=defs.MAXIMUM_NEW_TOKENS)
    parser.add_argument("--expected-question-count", type=int, default=None,
                        help="Default: the facet's number of held-out questions")
    parser.add_argument("--run-name", default=None, help="Default: the facet's steering run name")
    parser.add_argument("--small-sample-check", action="store_true", help="Write under small_sample_checks/")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and the ceilings; write nothing")
    arguments = parser.parse_args(argv)
    definition = facet_definitions_module.get_facet_definition(arguments.facet)
    arguments.facet = definition.facet_key
    arguments.vector_space = arguments.vector_space or definition.main_space
    if arguments.expected_question_count is None:
        arguments.expected_question_count = definition.questions_per_half[defs.HALF_HELD_OUT]
    arguments.run_name = arguments.run_name or definition.steering_run_name
    return arguments


def build_run_folder(arguments):
    definition = facet_definitions_module.get_facet_definition(getattr(arguments, "facet", FACET))
    if arguments.vector_space not in definition.vector_spaces:
        raise SystemExit(f"{definition.facet_key} has no {arguments.vector_space} vectors (its spaces: {definition.vector_spaces})")
    extraction_folder = Path(arguments.extraction_run_folder)
    extraction_configuration = json.loads(sorted(extraction_folder.glob("run_configuration_*.json"))[0].read_text(encoding="utf-8"))
    if extraction_configuration.get("mode") != definition.extraction_mode:
        raise SystemExit(f"{extraction_folder} is a {extraction_configuration.get('mode')!r} run, not a {definition.facet_key} "
                         f"extraction (expected mode {definition.extraction_mode!r})")
    layer, vector_paths, vector_metadata, vector_metadata_path = read_vector_set(
        arguments.vector_set_folder, arguments.vector_space, arguments.zeroed_dimension_count, definition)
    template_path = defs.PROMPT_FOLDER / definition.steering_system_prompt_file
    template_text_for_checking = template_path.read_text(encoding="utf-8").strip()
    questions = [q for q in read_jsonl(defs.find_latest_file(extraction_folder / defs.INPUTS_FOLDER_NAME, "questions", "jsonl"))
                 if q["facet"] == definition.facet_key and q["half"] == defs.HALF_HELD_OUT]
    if len(questions) != arguments.expected_question_count:
        raise SystemExit(f"Expected {arguments.expected_question_count} held-out questions, found {len(questions)}")
    for question in questions:  # every slot of the steering prompt needs its content (a loud failure, never a blank)
        try:
            fill_steering_system_prompt(template_text_for_checking, question_record=question, definition=definition)
        except ValueError as error:
            raise SystemExit(f"Question {question['question_id']}: {error}") from error
    activations_folder = extraction_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    norm_statistics = coefficients_module.median_whole_answer_activation_norm(activations_folder, layer)
    template = template_text_for_checking
    conditions = build_steering_conditions(list(vector_paths), arguments.steering_fractions)
    jobs = build_steering_jobs(questions, conditions, norm_statistics["median"], template, arguments.samples_per_question, definition)
    for condition in conditions:
        condition["coefficient_absolute_amount_added"] = coefficients_module.steering_coefficient(
            condition["fraction"], norm_statistics["median"])
    configuration = {
        "run_kind": "persona_vector_scoring_pod", "mode": definition.steering_mode, "facet": definition.facet_key,
        "facet_score_kind": definition.facet_score_kind, "facet_score_scale": definition.facet_score_scale_label(),
        "expected_sign_of_steering_fraction_on_facet_score": definition.expected_sign_of_steering_fraction_on_facet_score,
        "run_name": arguments.run_name, "stages": [pod_defs.STAGE_STEERING],
        "model_name": extraction_configuration["model_name"], "model_revision": extraction_configuration["model_revision"],
        "extraction_run_folder": str(extraction_folder), "vector_set_folder": str(arguments.vector_set_folder),
        "vector_metadata_file": str(vector_metadata_path), "vector_space": arguments.vector_space,
        "zeroed_dimension_count": arguments.zeroed_dimension_count,
        "vector_files_used": {name: path.name for name, path in vector_paths.items()},
        "steering_layer": layer, "steering_layer_by_facet": {name: layer for name in vector_paths},
        "steering_positions": arguments.steering_positions, "steering_fractions": sorted(set(arguments.steering_fractions)),
        "steering_fraction_meaning": definition.steering_fraction_meaning(),
        "median_activation_norm_at_steering_layer": norm_statistics["median"], "activation_norm_statistics": norm_statistics,
        "steering_system_prompt_template_file": definition.steering_system_prompt_file,
        "steering_system_prompt_template_sha256": sha256_of_file(template_path),
        "steering_samples_per_question": arguments.samples_per_question, "number_of_held_out_questions": len(questions),
        "steering_conditions": [c["condition_id"] for c in conditions], "number_of_steering_jobs": len(jobs),
        "number_of_steering_generations": sum(job["samples_per_prompt"] for job in jobs),
        "steering_maximum_new_tokens": arguments.maximum_new_tokens, "temperature": defs.SAMPLING_TEMPERATURE, "top_p": defs.SAMPLING_TOP_P,
        # keys the launcher's estimate reads for the other scoring pod stages (none of them runs here)
        "number_of_baseline_documents_to_generate": 0, "corpus_mean_document_tokens": 0, "number_of_documents_to_score": 0,
        "document_formats": []}
    questions_by_id = {question["question_id"]: question for question in questions}
    estimate = estimator.estimate_everything(jobs, questions_by_id, arguments.maximum_new_tokens)
    if arguments.dry_run:
        print(json.dumps({key: configuration[key] for key in (
            "steering_layer", "vector_files_used", "median_activation_norm_at_steering_layer", "number_of_held_out_questions",
            "number_of_steering_jobs", "number_of_steering_generations")}, indent=2))
        print(f"{len(conditions)} conditions: {[c['condition_id'] for c in conditions]}")
        print(estimator.describe(estimate))
        print("DRY RUN: nothing was written.")
        return None, configuration, estimate

    runs_directory = defs.SMALL_SAMPLE_CHECKS_ROOT if arguments.small_sample_check else defs.RUNS_ROOT
    run_folder, _ = choose_run_folder(runs_directory, arguments.run_name, None)
    inputs_folder = run_folder / pod_defs.INPUTS_FOLDER_NAME
    (inputs_folder / coefficients_module.STEERING_VECTORS_SUBFOLDER).mkdir(parents=True)
    timestamp = timestamp_now()
    write_jsonl(inputs_folder / "steering_jobs.jsonl", jobs)
    write_jsonl(inputs_folder / "steering_questions.jsonl", questions)
    (inputs_folder / "facet_definitions.json").write_text("{}\n", encoding="utf-8")
    (inputs_folder / "steering_system_prompt_template.txt").write_text(template + "\n", encoding="utf-8")
    (inputs_folder / f"steering_conditions_{timestamp}.json").write_text(json.dumps(conditions, indent=2), encoding="utf-8")
    manifest = {"vectors": {}, "layer": layer, "median_activation_norm": norm_statistics["median"]}
    for name, path in vector_paths.items():
        stored = f"persona_vector_{definition.facet_key}_{name}_{timestamp}.npy"
        shutil.copy2(path, inputs_folder / coefficients_module.STEERING_VECTORS_SUBFOLDER / stored)
        vector = np.load(path)
        manifest["vectors"][name] = {"file": stored, "layer": layer, "source_file": path.name,
                                     "stored_norm_at_steering_layer": float(np.linalg.norm(vector[layer]))}
    (inputs_folder / f"{coefficients_module.STEERING_VECTOR_MANIFEST_STEM}_{timestamp}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    configuration["created_timestamp"] = timestamp
    configuration["input_file_sha256"] = {path.name: sha256_of_file(path) for path in sorted(inputs_folder.glob("*.*"))}
    configuration["judge_and_gpu_estimate"] = estimate
    write_run_configuration(run_folder, timestamp, configuration)
    return run_folder, configuration, estimate


def main(argv=None):
    run_folder, configuration, estimate = build_run_folder(parse_arguments(argv))
    if run_folder:
        print(f"Run folder: {run_folder}\n{estimator.describe(estimate)}")


if __name__ == "__main__":
    main()
