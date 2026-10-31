"""Local end-to-end rehearsal of the whole persona vector projection pipeline, in pilot mode, with no paid call.

    python3 scripts/persona_vector_projection/rehearse_full_pipeline_end_to_end_locally_with_tiny_model_and_mocked_clients.py

Stages, each through the real command line or module entry point of that stage:
  1. convert the drafts                      4. judge the answers (MOCKED OpenAI client)
  2. build the pilot extraction run folder   5. compute the vectors (incl. half-domain variants), then the LEACE variant
  3. generation and activation pass          6. facet positive-control rewrites (MOCKED Anthropic client, real selection
     (tiny random Gemma 4, real tokenizer)      and export), build the scoring run folder (real corpus), scoring pod
                                                stages (baselines, scoring, steering) on the tiny model
                                             7. judge the steering answers (MOCKED OpenAI client)
                                             8. analysis (saved token ids, with and without a tokenizer)
Nothing is sent to any API, no pod is created, .env is not read. What is NOT real: the model (random weights, so answers
and scores are noise), every judge and rewrite reply (fabricated), the LEACE eraser (a stand-in that removes the
mean-difference direction, because the concept-erasure package is optional; the wiring around it is the real code), vLLM
(the Hugging Face backend stands in), and the layer (the tiny model has 5 layer outputs, so the chosen layer is set to 2 so
that its two neighbours on each side exist). The results mean nothing: this checks that the halves fit together.

Output: data/persona_vector_projection_runs/small_sample_checks/rehearsal_full_pipeline_end_to_end_tiny_model_mocked_clients_<timestamp>/
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import analyze_persona_vector_projection_scores as analysis  # noqa: E402
from persona_vector_projection import build_extraction_run_folder as extraction_builder  # noqa: E402
from persona_vector_projection import build_scoring_pod_run_folder as scoring_builder  # noqa: E402
from persona_vector_projection import compute_persona_vectors_from_activations_and_judgements as compute  # noqa: E402
from persona_vector_projection import convert_drafts_into_machine_readable_inputs as converter  # noqa: E402
from persona_vector_projection import fit_leace_eraser_and_effective_facet_directions as leace  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import judge_steering_answers_from_scoring_pod_run as steering_judge  # noqa: E402
from persona_vector_projection import make_facet_positive_control_rewrites as rewrites  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as extraction_defs  # noqa: E402
from persona_vector_projection import run_scoring_pod_stages as scoring_stages  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as scoring_defs  # noqa: E402
from persona_vector_projection import scoring_pod_test_support as support  # noqa: E402
from persona_vector_projection import test_facet_positive_control_rewrites as mocked_anthropic  # noqa: E402
from persona_vector_projection.load_persona_projection_scoring_outputs import load_projection_data  # noqa: E402
from persona_vector_projection.rehearse_extraction_chain_locally_with_tiny_model_and_mocked_judge import (  # noqa: E402
    MockedResponse, banner, run_pod_script)
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

QUESTIONS_PER_HALF = 2
SAMPLES_PER_PROMPT = 2
MAXIMUM_NEW_TOKENS = 10
TINY_MODEL_CHOSEN_LAYER = 2


def make_mocked_judge_call(requests, answer_by_id):
    """A replacement for the judge's API call. The reply depends on the kind of request and, for the extraction answers,
    on the pole the answer was generated for (so the vectors have something to separate)."""
    request_by_prompt = {}
    for request in requests:
        request_by_prompt.setdefault(request.prompt_text, []).append(request)
    served_counts = {}

    def mocked_call(_client, request_arguments):
        prompt = request_arguments["input"]
        candidates = request_by_prompt[prompt]
        index = served_counts.get(prompt, 0)
        served_counts[prompt] = index + 1
        request = candidates[min(index, len(candidates) - 1)]
        if request.kind == judge.KIND_VOLUNTEERED_WELFARE_REMARK:
            return MockedResponse("NO")
        if request.kind == judge.KIND_FACET_SCORE:
            pole = answer_by_id.get(request.answer_id, {}).get("pole")
            return MockedResponse("85" if pole == extraction_defs.POLE_POSITIVE else "12" if pole else "50")
        if request.kind == judge.KIND_COHERENCE:
            return MockedResponse("90")
        return MockedResponse("50")
    return mocked_call


def run_mocked_judge(requests, answer_by_id, judge_folder, stamp):
    judge_folder.mkdir(parents=True, exist_ok=True)
    results, failures = judge.run_judge_requests(
        requests, None, results_path=judge_folder / f"judge_results_{stamp}.jsonl",
        usage_log_path=judge_folder / f"judge_usage_and_cost_log_{stamp}.json",
        call_responses=make_mocked_judge_call(requests, answer_by_id), worker_count=4)
    assert not failures and len(results) == len(requests), (len(results), len(requests), failures[:2])
    return {"mocked": True, "requests": len(requests), "results": len(results), "failures": len(failures),
            "dry_run_estimate": judge.estimate_judge_cost(requests)}


def stand_in_eraser(activations, labels):
    """Orthogonal projection removing the mean-difference direction of the two classes (not LEACE, but the same shape of
    result as leace.fit_leace_eraser: proj_left (hidden, 1), proj_right (1, hidden), bias)."""
    difference = activations[labels == 1].mean(axis=0) - activations[labels == 0].mean(axis=0)
    unit = (difference / (np.linalg.norm(difference) + 1e-12)).astype(np.float64)
    return unit.reshape(-1, 1), unit.reshape(1, -1), None


def main():
    summary = {"purpose": "end-to-end plumbing rehearsal: tiny random model, mocked clients, fabricated scores; results are meaningless"}
    stamp = timestamp_now()
    started = time.time()
    rehearsal_folder = extraction_defs.SMALL_SAMPLE_CHECKS_ROOT / f"rehearsal_full_pipeline_end_to_end_tiny_model_mocked_clients_{stamp}"
    rehearsal_folder.mkdir(parents=True)
    scratch = Path(tempfile.mkdtemp(prefix="full_pipeline_rehearsal_"))
    summary["rehearsal_folder"] = str(rehearsal_folder)

    banner("1. Convert the drafts")
    converted = rehearsal_folder / "converted_inputs"
    summary["conversion"] = converter.convert_everything(
        extraction_defs.QUESTION_DRAFT_FOLDER, extraction_defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER,
        extraction_defs.PROMPT_FOLDER, converted, stamp)
    print(json.dumps(summary["conversion"]["question_records_per_facet"]))

    banner("2. Build the pilot extraction run folder (non-intrusive facet and topic control)")
    run_folder, configuration = extraction_builder.build_run_folder(extraction_builder.parse_arguments([
        "--mode", "pilot", "--run-name", "extraction_pilot_tiny_model", "--small-sample-check",
        "--maximum-questions-per-half", str(QUESTIONS_PER_HALF), "--samples-per-prompt", str(SAMPLES_PER_PROMPT),
        "--model-revision", "", "--generation-backend", "huggingface"]), converted)
    extraction_folder = run_folder
    summary["extraction_run_folder"] = str(extraction_folder)
    summary["extraction_facets"] = configuration["facets"]
    print(extraction_folder, configuration["number_of_answers_by_facet"])

    banner("3. Tiny model, generation and activation passes (real command line of the pod script)")
    tiny_model_directory = scratch / "tiny_model"
    support.save_tiny_model_with_real_tokenizer(tiny_model_directory)
    _, summary["generation_seconds"] = run_pod_script(extraction_folder, tiny_model_directory, "generation")
    _, summary["activation_seconds"] = run_pod_script(extraction_folder, tiny_model_directory, "activations")

    banner("4. Judge the answers (MOCKED OpenAI client)")
    answers = compute.load_answers(extraction_folder)
    inputs_folder = extraction_folder / extraction_defs.INPUTS_FOLDER_NAME
    questions_by_id = {q["question_id"]: q for q in read_jsonl(extraction_defs.find_latest_file(inputs_folder, "questions", "jsonl"))}
    facet_definitions = json.loads(extraction_defs.find_latest_file(inputs_folder, "facet_definitions", "json").read_text())
    requests = judge.build_judge_requests_for_answers(answers, questions_by_id, facet_definitions)
    summary["extraction_judge"] = run_mocked_judge(requests, {a["answer_id"]: a for a in answers},
                                                   extraction_folder / extraction_defs.JUDGEMENTS_FOLDER_NAME, stamp)

    banner("5. Compute the vectors; patch the layer; write the LEACE variant (stand-in eraser)")
    thresholds = {"positive_minimum": 65.0, "negative_maximum": 35.0, "coherence_minimum": 50.0,
                  "animal_share_maximum_difference": 15.0}
    metadata, metadata_path = compute.compute_all(
        extraction_folder, thresholds, configuration["facets"], bootstrap_resamples=20, minimum_layer_index=1,
        vector_set_name="rehearsal_vectors_meaningless", seed=0)
    vector_folder = metadata_path.parent
    layer_choice = metadata["layer_choice"]
    summary["layer_choice_before_patch"] = dict(layer_choice)
    layer_choice["layer_by_facet"] = {facet: TINY_MODEL_CHOSEN_LAYER for facet in layer_choice["layer_by_facet"]}
    layer_choice["shared_layer"] = TINY_MODEL_CHOSEN_LAYER
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    facet_vectors = {facet: np.load(vector_folder / metadata["vector_files"][f"{facet}_topic_removed"])
                     for facet in extraction_defs.PERSONA_FACETS if f"{facet}_topic_removed" in metadata["vector_files"]}
    layer_by_facet = {facet: TINY_MODEL_CHOSEN_LAYER for facet in facet_vectors}
    activations_cache = {}

    def activations_at(layer):
        activations_cache.setdefault(layer, leace.load_topic_control_answer_activations(extraction_folder, layer))
        return activations_cache[layer][0]

    labels = leace.load_topic_control_answer_activations(extraction_folder, TINY_MODEL_CHOSEN_LAYER)[1]
    effective = leace.compute_effective_directions(activations_at, labels, facet_vectors, layer_by_facet, eraser_fitter=stand_in_eraser)
    for facet, result in effective.items():
        leace.write_scoring_variant_vector(vector_folder, facet, facet_vectors[facet], result["layer"],
                                           result["erased_direction"], stamp)
    summary["vector_folder_files"] = sorted(p.name for p in vector_folder.glob("*.npy") if "__" in p.name)
    print(summary["vector_folder_files"])

    banner("6a. Facet positive-control rewrites (MOCKED Anthropic client; real selection, reuse and export)")
    client = mocked_anthropic.FakeClient(mocked_anthropic.always_changes(100))
    rewrite_folder = rewrites.main([
        "--documents-per-facet", "2", "--minimum-changed-units", "5", "--workers", "1", "--no-regeneration",
        "--env-path", str(scratch / "no_such_env_file"), "--run-name", "facet_positive_control_rewrites_mocked_client"],
        client=client, runs_directory_override=rehearsal_folder)
    export_path = next(rewrite_folder.glob("scoring_pod_extra_documents_*.jsonl"))
    rewrite_records_file = next(rewrite_folder.glob("facet_positive_control_rewrites_*.jsonl"))
    exported = read_jsonl(export_path)
    summary["rewrites"] = {"mocked_calls": len(client.calls), "exported_documents": len(exported),
                           "directions": sorted({r["rewrite_direction"] for r in exported})}
    print(summary["rewrites"])

    banner("6b. Build the pilot scoring run folder (real corpus; vectors from step 5; rewrites from step 6a)")
    scoring_folder, scoring_configuration, estimate = scoring_builder.build_run_folder(scoring_builder.parse_arguments([
        "--mode", "pilot", "--vector-folder", str(vector_folder), "--converted-inputs-folder", str(converted),
        "--positive-control-rewrites-jsonl", str(export_path), "--pilot-document-count", "20",
        "--document-formats", "assistant_turn", "--maximum-new-tokens", "24", "--small-sample-check",
        "--run-name", "scoring_pilot_tiny_model", "--baselines-per-document", "2", "--steering-coefficients", "-1", "0", "1",
        "--steering-samples-per-question", "2", "--model-revision", ""]))
    summary["scoring_run_folder"] = str(scoring_folder)
    summary["scored_vector_names"] = scoring_configuration["vector_names"]
    print(scoring_folder, scoring_configuration["vector_names"])

    banner("6c. Scoring pod stages on the tiny model")
    scoring_stages.main(["--run-folder", str(scoring_folder), "--model-name", str(tiny_model_directory), "--model-revision", "",
                         "--device", "cpu", "--dtype", "float32", "--generation-backend", "huggingface",
                         "--steering-maximum-new-tokens", "12", "--maximum-tokens-per-batch", "12000"])
    shard_arrays = np.load(sorted((scoring_folder / scoring_defs.OUTPUTS_FOLDER_NAME / scoring_defs.PROJECTIONS_SUBFOLDER
                                   / "assistant_turn").glob("*.npz"))[0])
    summary["projection_arrays"] = {key: list(shard_arrays[key].shape) for key in shard_arrays.files}
    assert "per_token_token_ids_int32" in shard_arrays.files

    banner("7. Judge the steering answers (MOCKED OpenAI client)")
    steering_answers = judge.read_all_answer_records(scoring_folder / scoring_defs.OUTPUTS_FOLDER_NAME / scoring_defs.STEERING_SUBFOLDER)
    steering_questions = {q["question_id"]: q for q in read_jsonl(scoring_folder / scoring_defs.INPUTS_FOLDER_NAME / "steering_questions.jsonl")}
    steering_definitions = json.loads((scoring_folder / scoring_defs.INPUTS_FOLDER_NAME / "facet_definitions.json").read_text())
    steering_requests = judge.build_judge_requests_for_answers(steering_answers, steering_questions, steering_definitions)
    assert steering_judge.main(["--scoring-run-folder", str(scoring_folder), "--dry-run"]) == 0
    exit_code = steering_judge.main(["--scoring-run-folder", str(scoring_folder), "--worker-count", "4"],
                                    call_responses=make_mocked_judge_call(steering_requests, {}))
    judged = [r for p in (scoring_folder / steering_judge.STEERING_JUDGE_FOLDER_NAME).glob("judge_results_*.jsonl") for r in read_jsonl(p)]
    assert exit_code == 0 and len(judged) == len(steering_requests), (exit_code, len(judged), len(steering_requests))
    summary["steering_judge"] = {"mocked": True, "requests": len(steering_requests), "results": len(judged),
                                 "dry_run_estimate": judge.estimate_judge_cost(steering_requests)}
    summary["steering_answers"] = len(steering_answers)

    banner("8. Analysis: saved token ids without a tokenizer, then with the local tokenizer")
    common = ["--scoring-run-folder", str(scoring_folder), "--document-format", "assistant_turn",
              "--rewrite-metadata-jsonl", str(rewrite_records_file)]
    output_without_tokenizer = analysis.main(common + ["--analysis-name", "rehearsal_saved_token_ids_no_tokenizer"])
    output_with_tokenizer = analysis.main(common + ["--analysis-name", "rehearsal_saved_token_ids_with_tokenizer",
                                                    "--tokenizer-name", scoring_defs.MODEL_NAME])
    for output in (output_without_tokenizer, output_with_tokenizer):
        assert list(output.glob("positive_control_changed_span_summary_*.csv")), f"no changed-span analysis in {output}"
        assert list(output.glob("ranking_agreement_vector_variants_*.csv")), f"no variant comparison in {output}"
    summary["analysis"] = {"without_tokenizer": str(output_without_tokenizer), "with_tokenizer": str(output_with_tokenizer),
                           "files": sorted(p.name for p in output_without_tokenizer.glob("*") if p.is_file())}
    data = load_projection_data(scoring_folder, "assistant_turn", scoring_defs.DEFAULT_CORPUS_PATH,
                                rewrite_metadata_jsonl=rewrite_records_file)
    summary["variant_columns_in_analysis"] = [n for n in data.vector_names if "__" in n]
    summary["saved_token_ids_documents"] = len(data.token_ids)

    summary["elapsed_seconds"] = round(time.time() - started, 1)
    summary_path = rehearsal_folder / f"rehearsal_summary_{timestamp_now()}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nREHEARSAL PASSED in {summary['elapsed_seconds']} seconds. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
