"""Local rehearsal of the whole steering chain of one content-controlled facet (`--facet`, default moralizing delivery) with no paid
call: build the steering-only run folder from the facet's extraction rehearsal (its tiny-model activations, questions and vector set), print the cost ceilings,
run the pod's steering stage through its real command line on a tiny randomly initialised Gemma 4 (real tokenizer and chat
template from the local Hugging Face cache), delete one shard and rerun to check resumption, judge with a MOCKED client
(dry run first; a rerun must ask nothing), run the analysis, and print the reuse-pod launcher dry run. Nothing is sent to any
API, no pod is created, .env is not read.

    python3 scripts/persona_vector_projection/rehearse_content_controlled_facet_steering_chain_locally_with_tiny_model_and_mocked_judge.py \
        --facet <facet name> [--extraction-rehearsal-folder <folder>]

Output: data/persona_vector_projection_runs/small_sample_checks/rehearsal_<facet key>_steering_check_tiny_model_mocked_judge_<timestamp>/
(with rehearsal_summary_<timestamp>.json). The answers are random-token gibberish and the judge scores are fabricated from each
condition's fraction, so the results mean nothing: this checks plumbing, file formats and resumability.
"""

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from persona_vector_projection import analyze_content_controlled_facet_steering_results as analysis  # noqa: E402
from persona_vector_projection import build_content_controlled_facet_steering_run_folder as builder  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module  # noqa: E402
from persona_vector_projection import judge_steering_answers_from_scoring_pod_run as steering_judge  # noqa: E402
from persona_vector_projection import launch_and_stop_runpod_scoring_and_steering_pod as launcher  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import rehearse_extraction_chain_locally_with_tiny_model_and_mocked_judge as earlier_rehearsal  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as pod_defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

SAMPLES_PER_QUESTION = 2
STEERING_MAXIMUM_NEW_TOKENS = 12
STAGES_SCRIPT = Path(__file__).parent / "run_scoring_pod_stages.py"



def banner(text):
    print(f"\n=== {text}", flush=True)


def run_name_for(definition):
    # the run name carries the start time too, so that a second rehearsal is not refused as a duplicate of the first
    return f"rehearsal_{definition.facet_key}_steering_check_tiny_model_mocked_judge_run_started_{timestamp_now()}"


def latest_extraction_rehearsal_folder(definition):
    """The newest extraction rehearsal of this facet that has a vector set: a small-sample-checks folder whose name has the
    facet name (or key) and `tiny_model_mocked_judge`, and whose run configuration has the facet's extraction mode."""
    candidates = []
    for folder in sorted(defs.SMALL_SAMPLE_CHECKS_ROOT.glob(f"rehearsal_*{definition.facet_name}*tiny_model_mocked_judge*")):
        if "steering_check" in folder.name or not list(folder.glob("vector_sets/*/persona_vector_metadata_*.json")):
            continue
        configurations = sorted(folder.glob("run_configuration_*.json"))
        if configurations and json.loads(configurations[0].read_text(encoding="utf-8")).get("mode") == definition.extraction_mode:
            candidates.append(folder)
    if not candidates:
        raise SystemExit(f"No {definition.facet_key} extraction rehearsal with a vector set in {defs.SMALL_SAMPLE_CHECKS_ROOT}; run "
                         f"rehearse_content_controlled_facet_extraction_chain_locally_with_tiny_model_and_mocked_judge.py --facet "
                         f"{definition.facet_name} first")
    return candidates[-1]


def snapshot_of_steering_shards(run_folder):
    return {path.name: path.stat().st_mtime_ns for path in (run_folder / pod_defs.OUTPUTS_FOLDER_NAME / pod_defs.STEERING_SUBFOLDER).glob("*.jsonl")}


def run_pod_steering_stage(run_folder, model_directory):
    command = [sys.executable, str(STAGES_SCRIPT), "--run-folder", str(run_folder), "--stages", "steering", "--model-name", str(model_directory),
               "--model-revision", "", "--device", "cpu", "--dtype", "float32", "--steering-maximum-new-tokens", str(STEERING_MAXIMUM_NEW_TOKENS),
               "--steering-jobs-per-shard", "6"]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Steering stage failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-3000:]}")
    return completed.stdout


def make_mocked_judge(records, conditions, definition):
    """Fabricated replies from each answer's condition. The facet score moves with the steering fraction in the direction the
    facet's judge scale predicts (derived from the definition: a positive fraction raises it when 100 is the positive pole), coherence
    falls with |fraction|, and the content check is valid JSON."""
    condition_by_answer_text = {record["answer_text"]: next(c for c in conditions if c["condition_id"] == record["steering_condition_id"])
                                for record in records}
    calls = []
    sign = definition.expected_sign_of_steering_fraction_on_facet_score

    def mocked_call(_client, request_arguments):
        prompt = request_arguments["input"]
        answer_text = next(text for text in condition_by_answer_text if text in prompt)
        fraction = condition_by_answer_text[answer_text]["fraction"]
        if "separate_welfare_mentions" in prompt:
            kind = "content_check"
            reply = json.dumps({"makes_the_point": True, "separate_welfare_mentions": 1, "adds_welfare_facts": False,
                                "completes_the_task": True, "welfare_passage": answer_text[:20]})
        elif request_arguments["reasoning"]["effort"] == "none":
            kind, reply = "coherence", str(int(max(0, 92 - 150 * abs(fraction))))
        else:
            kind, reply = definition.facet_score_kind, str(int(min(100, max(0, 50 + sign * 250 * fraction))))
        calls.append(kind)
        return earlier_rehearsal.MockedResponse(reply)

    return mocked_call, calls


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    facet_definitions_module.add_facet_argument(parser)
    parser.add_argument("--extraction-rehearsal-folder", default=None)
    arguments = parser.parse_args(argv)
    definition = facet_definitions_module.get_facet_definition(arguments.facet)
    RUN_NAME = run_name_for(definition)
    from tiny_model_and_tokenizer_helpers_for_log_probability_persona_check import build_tiny_random_gemma4_model
    from persona_vector_projection import scoring_pod_test_support as support
    extraction_folder = Path(arguments.extraction_rehearsal_folder) if arguments.extraction_rehearsal_folder else latest_extraction_rehearsal_folder(definition)
    vector_set = sorted((extraction_folder / defs.VECTOR_SETS_FOLDER_NAME).iterdir())[-1]
    held_out_count = sum(1 for q in read_jsonl(defs.find_latest_file(extraction_folder / defs.INPUTS_FOLDER_NAME, "questions", "jsonl"))
                         if q["half"] == defs.HALF_HELD_OUT and q["facet"] == definition.facet_key)
    summary = {"facet": definition.facet_key, "facet_score_scale": definition.facet_score_scale_label(),
               "purpose": f"plumbing rehearsal of the {definition.facet_key} steering check: tiny random model, mocked judge, fabricated "
                          "scores; results are meaningless", "extraction_rehearsal_folder": str(extraction_folder), "vector_set": str(vector_set)}

    banner(f"1. Build the steering-only run folder ({held_out_count} held-out questions here, the real run has {definition.questions_per_half[defs.HALF_HELD_OUT]}; {SAMPLES_PER_QUESTION} samples)")
    build_arguments = builder.parse_arguments([
        "--facet", definition.facet_key, "--extraction-run-folder", str(extraction_folder), "--vector-set-folder", str(vector_set), "--small-sample-check",
        "--run-name", RUN_NAME, "--expected-question-count", str(held_out_count), "--samples-per-question", str(SAMPLES_PER_QUESTION)])
    run_folder, configuration, estimate = builder.build_run_folder(build_arguments)
    inputs_folder = run_folder / pod_defs.INPUTS_FOLDER_NAME
    jobs = read_jsonl(inputs_folder / "steering_jobs.jsonl")
    summary.update({"run_folder": str(run_folder), "conditions": len(configuration["steering_conditions"]), "jobs": len(jobs),
                    "generations": configuration["number_of_steering_generations"], "steering_layer": configuration["steering_layer"],
                    "median_activation_norm": configuration["median_activation_norm_at_steering_layer"]})
    assert len(configuration["steering_conditions"]) == 25 and configuration["number_of_steering_generations"] == 25 * held_out_count * SAMPLES_PER_QUESTION
    print(run_folder, summary["conditions"], "conditions;", summary["generations"], "generations")

    banner("2. Cost ceilings for the REAL run (the facet's held-out questions, 10 samples, 1,000 new tokens): from the real extraction run's questions if present")
    real_runs = sorted(defs.RUNS_ROOT.glob(f"persona_vector_extraction_{definition.facet_key}_pilot_*" if definition.legacy_moralizing_names else f"{definition.extraction_run_name}_*"))
    if real_runs:  # read-only: only the questions file of the running pilot is read
        questions = [q for q in read_jsonl(defs.find_latest_file(real_runs[-1] / defs.INPUTS_FOLDER_NAME, "questions", "jsonl"))
                     if q["facet"] == definition.facet_key and q["half"] == defs.HALF_HELD_OUT]
        real_conditions = builder.build_steering_conditions(["main_vector", builder.zeroed_vector_name(4)], configuration["steering_fractions"])
        template = (defs.PROMPT_FOLDER / definition.steering_system_prompt_file).read_text(encoding="utf-8")
        real_jobs = builder.build_steering_jobs(questions, real_conditions, 1.0, template, 10, definition)
        real_estimate = builder.estimator.estimate_everything(real_jobs, {q["question_id"]: q for q in questions}, 1000)
        print(builder.estimator.describe(real_estimate))
        summary["ceilings_for_the_real_run"] = real_estimate
    else:
        print(f"No real {definition.facet_key} extraction run folder found; skipped")

    banner("3. Tiny model with the real Gemma 4 tokenizer and chat template")
    working_directory = Path(tempfile.mkdtemp(prefix="rehearsal_content_controlled_steering_tiny_gemma4_model_"))
    from transformers import AutoTokenizer
    snapshot = __import__("glob").glob(str(Path.home() / ".cache/huggingface/hub" / earlier_rehearsal.GEMMA_SNAPSHOT_GLOB))[0]
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    build_tiny_random_gemma4_model(len(tokenizer), final_logit_softcapping=30.0).save_pretrained(working_directory)
    tokenizer.save_pretrained(working_directory)

    banner("4. Steering stage through the pod script's command line; delete the last shard and rerun (resume)")
    run_pod_steering_stage(run_folder, working_directory)
    first_pass = snapshot_of_steering_shards(run_folder)
    shard_folder = run_folder / pod_defs.OUTPUTS_FOLDER_NAME / pod_defs.STEERING_SUBFOLDER
    last_shard = sorted(shard_folder.glob("answers_shard_*.jsonl"))[-1]
    last_shard.unlink()
    run_pod_steering_stage(run_folder, working_directory)
    second_pass = snapshot_of_steering_shards(run_folder)
    rewritten = sorted(name for name in second_pass if first_pass.get(name) != second_pass[name])
    assert rewritten == [last_shard.name] and set(first_pass) == set(second_pass), (rewritten, last_shard.name)
    answers = judge_answers = steering_judge.judge.read_all_answer_records(shard_folder)
    assert len(answers) == configuration["number_of_steering_generations"]
    assert all(answer["facets_and_coefficients"] == {} or answer["steering_layers"] for answer in answers)
    summary["steering_stage"] = {"shards": len(first_pass), "answers": len(answers), "resume_check": "deleted the last shard; only it was rewritten",
                                 "example_answer": answers[0]["answer_text"][:60]}
    print(f"{len(answers)} answers in {len(first_pass)} shards; resume rewrote only {last_shard.name}")

    banner("5. Judge: dry run (calls and dollar ceiling), then a MOCKED run, then a rerun that must ask nothing")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--dry-run"]) == 0
    print(buffer.getvalue())
    summary["judge_dry_run"] = json.loads(buffer.getvalue()[:buffer.getvalue().index("DRY RUN")])
    conditions = json.loads(defs.find_latest_file(inputs_folder, "steering_conditions", "json").read_text())
    mocked_call, calls = make_mocked_judge(answers, conditions, definition)
    assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--worker-count", "4"], call_responses=mocked_call) == 0
    assert len(calls) == 3 * len(answers)
    calls_after_first = len(calls)
    assert steering_judge.main(["--scoring-run-folder", str(run_folder), "--worker-count", "4"], call_responses=mocked_call) == 0
    assert len(calls) == calls_after_first, "the rerun asked the judge again"
    summary["judge"] = {"mocked": True, "calls": calls_after_first, "rerun_calls": 0}

    banner("6. Analysis")
    result = analysis.analyse(run_folder)
    print(result["markdown"].read_text(encoding="utf-8"))
    summary["analysis"] = {"csv": str(result["csv"]), "markdown": str(result["markdown"]), "rows": len(result["summaries"]),
                           "monotonicity": result["monotonicity"]}
    expected_sign = definition.expected_sign_of_steering_fraction_on_facet_score
    assert len(result["summaries"]) == 26 and all(
        entry["spearman"]["spearman"] * expected_sign > 0.9 and entry["spearman"]["has_expected_sign"] is True
        for entry in result["monotonicity"].values())

    banner("7. Launcher dry run on a reused pod (free, placeholder secrets: .env is not read)")
    launcher_arguments = launcher.parse_arguments(["--run-folder", str(run_folder), "--maximum-running-hours", "1.5", "--maximum-dollars", "6.75",
                                                   "--stages", "steering", "--reuse-existing-pod", "REHEARSAL_POD_ID", "--dry-run"])
    launcher.apply_job_profile()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        launcher.print_estimate(launcher_arguments)
        launcher.shared_launcher.run_launcher(launcher_arguments, secrets={"RUNPOD_API_KEY": None, "HF_TOKEN": None, "RUNPOD_COST_CENTER_NAME": None})
    print(buffer.getvalue())
    summary["launcher_dry_run_output"] = buffer.getvalue()
    summary["launcher_scorer_command"] = launcher.shared_launcher.build_scorer_command_line(launcher_arguments)
    summary_path = run_folder / f"rehearsal_summary_{timestamp_now()}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nREHEARSAL PASSED. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
