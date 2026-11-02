"""Local rehearsal of the whole extraction chain of a content-controlled facet (moralizing delivery by default; or cares about
welfare, calibrated consideration, non-intrusive with --facet) with no paid call: convert the drafts, build the facet run folder, generate with a tiny randomly initialised Gemma 4 model (real Gemma 4 tokenizer and chat template
from the local Hugging Face cache), judge with a MOCKED client (facet score, coherence, and content-check replies that
quote a real substring of the answer, some of them whitespace- or quote-altered, missing, or unparseable), locate the
welfare passages (not for non-intrusive, whose main mean is the whole-answer mean), run the activation stage under the neutral
prompt (passage means too, except for non-intrusive), compute the vectors, and print the launcher and judge dry runs. Nothing is sent to any API, no pod is created, .env is not read.

    python3 scripts/persona_vector_projection/rehearse_content_controlled_facet_extraction_chain_locally_with_tiny_model_and_mocked_judge.py [--facet <facet>]

(The old name, rehearse_moralizing_delivery_chain_locally_with_tiny_model_and_mocked_judge.py, is an alias of this script.)
Output: data/persona_vector_projection_runs/small_sample_checks/rehearsal_<facet key>_tiny_model_mocked_judge_<timestamp>/
(the run folder, with rehearsal_summary_<timestamp>.json). The answers are random-token gibberish and the judge scores are
fabricated from each answer's pole, so the vectors mean nothing: this checks plumbing, file formats and resumability.
"""

import glob
import hashlib
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

from persona_vector_projection import build_extraction_run_folder as builder  # noqa: E402
from persona_vector_projection import compute_content_controlled_facet_vectors_from_activations as moralizing_compute  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module  # noqa: E402
from persona_vector_projection import convert_drafts_into_machine_readable_inputs as converter  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import locate_welfare_passages_in_judged_answers as span_step  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import rehearse_extraction_chain_locally_with_tiny_model_and_mocked_judge as earlier_rehearsal  # noqa: E402
from persona_vector_projection import welfare_passage_spans_and_token_mapping as mapping  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

QUESTIONS_PER_HALF = 2
SAMPLES_PER_PROMPT = 2
MAXIMUM_NEW_TOKENS = 16
LAUNCHER = Path(__file__).parent / "launch_and_stop_runpod_generation_pod.py"


def banner(text):
    print(f"\n=== {text}", flush=True)


def stable_number(text, modulus):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % modulus


def passage_quoted_from(answer_text, answer_id):
    """A real substring of the answer (its middle part, cut at spaces where there are any). Every fifth answer's quote is
    altered the way a language model's quote might be (spaces doubled, apostrophes curled), every eleventh is not in the
    answer at all."""
    stripped = answer_text.strip()
    start, end = len(stripped) // 4, (3 * len(stripped)) // 4
    passage = stripped[start:end].strip() or stripped
    kind = stable_number(answer_id + "passage", 55)
    if kind % 11 == 0:
        return "A sentence that the answer does not contain."
    if kind % 5 == 0:
        return passage.replace(" ", "  ").replace("'", "’")
    return passage


def mocked_content_check_reply(answer, answer_text, always_unparseable_ids, first_reply_broken_ids, attempt_number):
    answer_id = answer["answer_id"]
    if answer_id in always_unparseable_ids or (answer_id in first_reply_broken_ids and attempt_number == 1):
        return "I think the answer makes the point, but I will not use JSON."
    pick = stable_number(answer_id + "content", 40)
    return json.dumps({"makes_the_point": pick != 0, "separate_welfare_mentions": 2 if pick == 1 else 1,
                       "adds_welfare_facts": pick == 2, "completes_the_task": pick != 3,
                       "welfare_passage": passage_quoted_from(answer_text, answer_id)}, ensure_ascii=False)


def main(argv=None):
    import argparse
    import glob as glob_module
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    facet_definitions_module.add_facet_argument(parser)
    parser.add_argument("--run-name", default=None, help="Default: rehearsal_<facet key>_tiny_model_mocked_judge (a name already used is refused)")
    parsed_arguments = parser.parse_args(argv)
    definition = facet_definitions_module.get_facet_definition(parsed_arguments.facet)
    facet_key = definition.facet_key
    from transformers import AutoTokenizer
    from tiny_model_and_tokenizer_helpers_for_log_probability_persona_check import build_tiny_random_gemma4_model
    earlier_rehearsal.MAXIMUM_NEW_TOKENS = MAXIMUM_NEW_TOKENS
    summary = {"purpose": f"plumbing rehearsal of the {facet_key} chain: tiny random model, mocked judge, fabricated "
                          "scores; results are meaningless"}
    stamp = timestamp_now()
    banner("1. Convert the drafts (including the content-controlled facets' drafts)")
    converted = defs.SMALL_SAMPLE_CHECKS_ROOT / f"rehearsal_{facet_key}_converted_inputs_{stamp}"
    summary["conversion"] = converter.convert_everything(defs.QUESTION_DRAFT_FOLDER, defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER,
                                                         defs.PROMPT_FOLDER, converted, stamp)
    print(json.dumps(summary["conversion"]["question_records_per_facet"]))

    banner(f"2. Build the {facet_key} run folder ({QUESTIONS_PER_HALF} questions per half, {SAMPLES_PER_PROMPT} samples)")
    run_folder, configuration = builder.build_run_folder(builder.parse_arguments([
        "--mode", definition.extraction_mode, "--run-name", parsed_arguments.run_name or f"rehearsal_{facet_key}_tiny_model_mocked_judge",
        "--small-sample-check", "--maximum-questions-per-half", str(QUESTIONS_PER_HALF), "--samples-per-prompt", str(SAMPLES_PER_PROMPT),
        "--model-revision", "", "--generation-backend", "huggingface"]), converted)
    assert configuration["activation_pass_system_prompt"] == "Keep your answer under 300 words."
    summary.update({"run_folder": str(run_folder), "generation_jobs": configuration["number_of_generation_jobs"],
                    "answers_by_facet": configuration["number_of_answers_by_facet"],
                    "activation_pass_system_prompt": configuration["activation_pass_system_prompt"]})
    print(run_folder, configuration["number_of_answers_by_facet"])

    banner("3. Tiny model with the real Gemma 4 tokenizer and chat template")
    snapshot = glob_module.glob(str(Path.home() / ".cache/huggingface/hub" / earlier_rehearsal.GEMMA_SNAPSHOT_GLOB))[0]
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model_directory = Path(tempfile.mkdtemp(prefix=f"rehearsal_{facet_key}_tiny_gemma4_model_"))
    build_tiny_random_gemma4_model(len(tokenizer), final_logit_softcapping=30.0).save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)

    banner("4. Generation stage on the tiny model")
    _, summary["generation_seconds"] = earlier_rehearsal.run_pod_script(run_folder, model_directory, "generation")
    answers = moralizing_compute.shared_compute.load_answers(run_folder)
    assert len(answers) == configuration["number_of_generation_jobs"] * SAMPLES_PER_PROMPT
    print(f"{len(answers)} answers; example: {answers[0]['answer_text'][:80]!r}")

    banner("5. Judge with a MOCKED client (some replies unparseable on the first attempt, a few on every attempt)")
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    questions_by_id = {q["question_id"]: q for q in read_jsonl(defs.find_latest_file(inputs_folder, "questions", "jsonl"))}
    facet_definitions = json.loads(defs.find_latest_file(inputs_folder, "facet_definitions", "json").read_text())
    requests = judge.build_judge_requests_for_answers(answers, questions_by_id, facet_definitions)
    estimate = judge.estimate_judge_cost(requests)
    expected_calls = {definition.facet_score_kind: len(answers), "content_check": len(answers), "coherence": len(answers)}
    if definition.has_confidence_control:  # calibrated consideration: the stated-confidence judge is asked for every answer
        expected_calls[judge.KIND_STATED_CONFIDENCE] = len(answers)
    assert estimate["calls_by_kind"] == expected_calls
    if definition.has_confidence_control:  # one negative rewording per pairing slot, only where it is miscalibrated for the band
        jobs_for_check = read_jsonl(defs.find_latest_file(inputs_folder, "generation_jobs", "jsonl"))
        for job in jobs_for_check:
            if job["pole"] == defs.POLE_NEGATIVE:
                assert definition.negative_rewording_is_miscalibrated_for_question(job["tag"], questions_by_id[job["question_id"]]), job["job_id"]
        for question_id in questions_by_id:
            assert sum(1 for j in jobs_for_check if j["question_id"] == question_id and j["pole"] == "positive") == sum(
                1 for j in jobs_for_check if j["question_id"] == question_id and j["pole"] == "negative")
    answer_by_id = {a["answer_id"]: a for a in answers}
    request_by_prompt = {}
    for request in requests:
        request_by_prompt.setdefault(request.prompt_text, []).append(request)
    served_counts, attempts_by_request = {}, {}
    ids_sorted = sorted(answer_by_id)
    always_unparseable_ids = set(ids_sorted[::40])
    first_reply_broken_ids = set(ids_sorted[3::9])
    unparseable_replies_allowed = {"on": True}

    def mocked_call(_client, request_arguments):
        prompt = request_arguments["input"]
        candidates = request_by_prompt[prompt]
        index = served_counts.get(prompt, 0)
        served_counts[prompt] = index + 1
        request = candidates[min(index, len(candidates) - 1)]
        answer = answer_by_id[request.answer_id]
        attempts_by_request[request.request_id] = attempts_by_request.get(request.request_id, 0) + 1
        if request.kind == judge.KIND_COHERENCE:
            return earlier_rehearsal.MockedResponse("90")
        if request.kind == judge.KIND_STATED_CONFIDENCE:
            # the same value for both poles of one item (question, slot, sample): confident or hedged by a hash, so the poles balance
            slot = facet_definitions_module.pairing_slot_of_rewording(int(answer["system_prompt_id"].rsplit("_", 1)[1]), 5)
            return earlier_rehearsal.MockedResponse("85" if stable_number(f"{answer['question_id']}{slot}{answer['sample_index']}", 2) == 0 else "15")
        if request.kind == definition.facet_score_kind:  # 100 = positive pole, 0 = negative pole, for every facet
            pick = stable_number(request.answer_id + "facet_score", 30)
            # positive pole high, negative pole low; every thirtieth answer is nearer the middle (always a valid 0 to 100 number)
            base = 85 if answer["pole"] == defs.POLE_POSITIVE else 12
            nearer_the_middle = 15 if answer["pole"] == defs.POLE_POSITIVE else 18
            return earlier_rehearsal.MockedResponse(str(base - nearer_the_middle if pick == 0 and answer["pole"] == defs.POLE_POSITIVE else base + (nearer_the_middle if pick == 0 else 0)))
        if unparseable_replies_allowed["on"]:
            reply = mocked_content_check_reply(answer, answer["answer_text"], always_unparseable_ids, first_reply_broken_ids,
                                               attempts_by_request[request.request_id])
        else:
            reply = mocked_content_check_reply(answer, answer["answer_text"], set(), set(), 2)
        return earlier_rehearsal.MockedResponse(reply)

    judge_folder = run_folder / defs.JUDGEMENTS_FOLDER_NAME
    judge_folder.mkdir()
    results, failures = judge.run_judge_requests(
        requests, None, results_path=judge_folder / f"judge_results_{stamp}.jsonl",
        usage_log_path=judge_folder / f"judge_usage_and_cost_log_{stamp}.json", call_responses=mocked_call, worker_count=4)
    discarded = [r for r in results if r.get("discarded")]
    assert not failures and len(results) == len(requests) and discarded and all(r["kind"] == "content_check" for r in discarded), (
        len(failures), len(results), len(requests), len(discarded), sorted({r["kind"] for r in discarded}), failures[:2])
    first_retried_and_recovered = [rid for rid in (f"{a}__content_check" for a in first_reply_broken_ids)
                                   if attempts_by_request[rid] == 2]
    assert first_retried_and_recovered, "no broken first reply was retried within the call"
    summary["judge"] = {"mocked": True, "requests": len(requests), "dry_run_estimate": estimate,
                        "content_checks_unparseable_on_every_attempt_and_recorded_as_failed": len(discarded),
                        "content_checks_unparseable_on_first_attempt_and_recovered_by_the_retry": len(first_retried_and_recovered)}
    unparseable_replies_allowed["on"] = False
    results_after_rerun, failures = judge.run_judge_requests(
        requests, None, results_path=judge_folder / f"judge_results_{stamp}_rerun_of_failed.jsonl", call_responses=mocked_call, worker_count=4)
    assert not failures and not any(r.get("discarded") for r in results_after_rerun) and len(results_after_rerun) == len(requests)
    summary["judge"]["rerun_asked_only_the_failed_requests_again"] = {"requests_asked": len(discarded), "all_recovered": True}
    unparseable_replies_allowed["on"] = True
    # the real (unmocked) judge's dry run on this run folder: prints the calls by kind and the dollar ceiling, calls nothing
    dry = subprocess.run([sys.executable, str(Path(__file__).parent / "judge_answers_with_gpt_5_6_luna.py"), "--run-folder",
                          str(run_folder), "--dry-run"], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr[-1000:]
    summary["judge_dry_run_on_the_rehearsal_answers"] = json.loads(dry.stdout[:dry.stdout.index("DRY RUN")])

    if definition.uses_welfare_passage:
        banner("6. Locate the welfare passages in the answers (local step)")
        span_step.main(["--run-folder", str(run_folder)])
        spans = read_jsonl(next(inputs_folder.glob(f"{defs.WELFARE_PASSAGE_SPANS_FILE_STEM}_*.jsonl")))
        statuses = {s["status"] for s in spans}
        assert len(spans) == len(answers) and mapping.MATCH_FAILED in statuses and mapping.MATCH_EXACT in statuses
        summary["spans"] = {"records": len(spans), "status_counts": {s: sum(1 for r in spans if r["status"] == s) for s in sorted(statuses)}}
    else:
        banner("6. No welfare passage step: this facet's main mean is the whole-answer mean")
        assert not list(inputs_folder.glob(f"{defs.WELFARE_PASSAGE_SPANS_FILE_STEM}_*.jsonl"))
        summary["spans"] = None
    banner("7. Activation stage (neutral system prompt, whole-answer and welfare passage means); delete a shard and rerun")
    _, summary["activation_seconds"] = earlier_rehearsal.run_pod_script(run_folder, model_directory, "activations")
    activations_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    shard_paths = sorted(activations_folder.glob("activations_shard_*.npz"))
    modification_times = {p: p.stat().st_mtime_ns for p in shard_paths}
    shard_paths[0].unlink()
    earlier_rehearsal.run_pod_script(run_folder, model_directory, "activations")
    assert shard_paths[0].exists() and all(p.stat().st_mtime_ns == modification_times[p] for p in shard_paths[1:])
    statuses_by_answer = {}
    for path in shard_paths:
        with np.load(path) as shard:
            if not definition.uses_welfare_passage:
                assert "passage_activations" not in shard.files  # whole-answer means only
                continue
            assert shard["passage_activations"].shape == shard["activations"].shape
            for answer_id, status, count, row in zip(shard["answer_ids"], shard["passage_status"], shard["passage_token_counts"],
                                                     shard["passage_activations"]):
                statuses_by_answer[str(answer_id)] = str(status)
                assert (count > 0) == (str(status) == "passage_mean_recorded") == (not np.isnan(row).any())
    summary["activation_pass"] = {"passage_status_counts": {s: list(statuses_by_answer.values()).count(s) for s in sorted(set(statuses_by_answer.values()))},
                                  "resume_check": "deleted shard recomputed, other shards untouched",
                                  "activation_pass_configuration": json.loads(next((run_folder / defs.OUTPUTS_FOLDER_NAME).glob(
                                      "activation_pass_configuration_*.json")).read_text())}

    banner("8. Compute the vectors on CPU")
    score_key = "delivery_score" if definition.facet_score_kind == judge.KIND_DELIVERY_SCORE else "facet_score"
    thresholds = {f"positive_{score_key}_minimum": 65.0, f"negative_{score_key}_maximum": 35.0, "coherence_minimum": 50.0,
                  "maximum_passage_length_ratio": 1.5, "required_separate_welfare_mentions": 1}
    metadata, metadata_path = moralizing_compute.compute_all(run_folder, thresholds, "rehearsal_vectors_meaningless",
                                                             bootstrap_resamples=20, minimum_layer_index=1, seed=0, facet_key=facet_key)
    entry = metadata["per_facet"][facet_key]
    summary["vectors"] = {"metadata_file": str(metadata_path), "layer_choice": metadata["layer_choice"],
                          "filter_counts": entry["filter_counts"], "files": sorted(p.name for p in metadata_path.parent.iterdir()),
                          "held_out_answers_used": entry["held_out_answers_used"], "shape": metadata["shape"],
                          "peak_resident_memory_megabytes": metadata["peak_resident_memory_megabytes"]}

    banner("9. Pod launcher dry runs on this run folder (free, no pod): generation with keep, then activations on a reused pod")
    outputs = {}
    for label, extra in (("generation_stage_keep_pod", ["--stage", "generation", "--keep-pod-after-run"]),
                         ("activation_stage_reuse_pod", ["--stage", "activations", "--reuse-existing-pod", "REHEARSAL_POD_ID"])):
        completed = subprocess.run([sys.executable, str(LAUNCHER), "--run-folder", str(run_folder), "--maximum-running-hours", "1",
                                    "--maximum-dollars", "5", "--dry-run", *extra], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr[-1000:]
        outputs[label] = {"exit_code": completed.returncode,
                          "disk_line": next((l for l in completed.stdout.splitlines() if l.startswith("Disk requested")), None),
                          "pod_script_command": next((l.split("python scripts/", 1)[1].split(" >", 1)[0] for l in completed.stdout.splitlines()
                                                      if l.startswith("  scorer:")), None)}
    summary["launcher_dry_runs"] = outputs
    summary_path = run_folder / f"rehearsal_summary_{timestamp_now()}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nREHEARSAL PASSED. Summary: {summary_path}")
    print(json.dumps({k: summary[k] for k in ("generation_jobs", "generation_seconds", "activation_seconds")}))
    print(json.dumps(summary["spans"]))
    print(json.dumps(summary["activation_pass"]["passage_status_counts"]))
    print(definition.facet_score_scale_label())
    print(f"Run folder: {run_folder}")
    print(json.dumps(summary["vectors"]["filter_counts"]["items_failing_each_condition"]))
    print(json.dumps(summary["vectors"]["layer_choice"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
