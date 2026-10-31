"""Local rehearsal of the whole extraction chain with no paid call: convert the drafts, build a run folder, generate and
save activations with a tiny randomly initialised Gemma 4 model (real Gemma 4 tokenizer and chat template from the local
Hugging Face cache, so the system-role handling is the real one), judge with a MOCKED client, compute the vectors, and
check resumability. Nothing is sent to any API, no pod is created, .env is not read.

    python3 scripts/persona_vector_projection/rehearse_extraction_chain_locally_with_tiny_model_and_mocked_judge.py

Output: data/persona_vector_projection_runs/small_sample_checks/rehearsal_tiny_model_mocked_judge_<timestamp>/ (the run
folder, with rehearsal_summary_<timestamp>.json). The answers are random-token gibberish and the judge scores are
fabricated from each answer's pole, so the vectors mean nothing: this checks plumbing, file formats, resumability and
memory, not results.
"""

import json
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from persona_vector_projection import build_extraction_run_folder as builder  # noqa: E402
from persona_vector_projection import compute_persona_vectors_from_activations_and_judgements as compute  # noqa: E402
from persona_vector_projection import convert_drafts_into_machine_readable_inputs as converter  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

GEMMA_SNAPSHOT_GLOB = "models--google--gemma-4-12B-it/snapshots/*"
QUESTIONS_PER_HALF = 2
SAMPLES_PER_PROMPT = 2
MAXIMUM_NEW_TOKENS = 10


class MockedResponse:
    def __init__(self, text):
        self.output_text, self.status = text, "completed"
        self.usage = SimpleNamespace(input_tokens=500, output_tokens=2, input_tokens_details=SimpleNamespace(cached_tokens=0),
                                     output_tokens_details=SimpleNamespace(reasoning_tokens=0))


def banner(text):
    print(f"\n=== {text}", flush=True)


def run_pod_script(run_folder, model_directory, stage, extra=()):
    command = [sys.executable, str(Path(__file__).parent / "generate_answers_and_save_activations_on_pod.py"),
               "--run-folder", str(run_folder), "--stage", stage, "--model-name", str(model_directory), "--device", "cpu",
               "--dtype", "float32", "--generation-backend", "huggingface", "--jobs-per-shard", "20",
               "--maximum-new-tokens", str(MAXIMUM_NEW_TOKENS), "--activation-batch-size", "4", *extra]
    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Pod script failed ({stage}):\n{completed.stdout[-2000:]}\n{completed.stderr[-3000:]}")
    return completed.stdout, round(time.time() - started, 1)


def main():
    import glob
    from transformers import AutoTokenizer
    from tiny_model_and_tokenizer_helpers_for_log_probability_persona_check import build_tiny_random_gemma4_model
    summary = {"purpose": "plumbing rehearsal: tiny random model, mocked judge, fabricated scores; results are meaningless"}
    stamp = timestamp_now()
    banner("1. Convert the drafts")
    converted = defs.SMALL_SAMPLE_CHECKS_ROOT / f"rehearsal_converted_inputs_{stamp}"
    summary["conversion"] = converter.convert_everything(defs.QUESTION_DRAFT_FOLDER, defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER,
                                                         defs.PROMPT_FOLDER, converted, stamp)
    print(json.dumps(summary["conversion"]["question_records_per_facet"]))

    banner("2. Build the run folder (full mode, 2 questions per half, 2 samples)")
    run_folder, configuration = builder.build_run_folder(builder.parse_arguments([
        "--mode", "full", "--run-name", "rehearsal_tiny_model_mocked_judge", "--small-sample-check",
        "--maximum-questions-per-half", str(QUESTIONS_PER_HALF), "--samples-per-prompt", str(SAMPLES_PER_PROMPT),
        "--model-revision", "", "--generation-backend", "huggingface"]), converted)
    summary.update({"run_folder": str(run_folder), "generation_jobs": configuration["number_of_generation_jobs"],
                    "answers_by_facet": configuration["number_of_answers_by_facet"]})
    print(run_folder, configuration["number_of_answers_by_facet"])

    banner("3. Tiny model with the real Gemma 4 tokenizer and chat template")
    snapshot = glob.glob(str(Path.home() / ".cache/huggingface/hub" / GEMMA_SNAPSHOT_GLOB))[0]
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model_directory = Path(tempfile.mkdtemp(prefix="rehearsal_tiny_gemma4_model_"))
    build_tiny_random_gemma4_model(len(tokenizer), final_logit_softcapping=30.0).save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)

    banner("4. Generation stage, then the same command again (everything should be skipped)")
    output, seconds = run_pod_script(run_folder, model_directory, "generation")
    summary["generation_seconds"] = seconds
    answers_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME
    shard_paths = sorted(answers_folder.glob("answers_shard_*.jsonl"))
    modification_times = {p: p.stat().st_mtime_ns for p in shard_paths}
    output_again, _ = run_pod_script(run_folder, model_directory, "generation")
    assert all(p.stat().st_mtime_ns == modification_times[p] for p in shard_paths), "a finished shard was rewritten"
    assert "already done" in output_again and re.search(r"(\d+) shards, \1 already done", output_again), output_again[-500:]
    summary["answer_shards"] = len(shard_paths)
    system_role_file = next((run_folder / defs.OUTPUTS_FOLDER_NAME).glob("chat_template_system_role_handling_*.json"))
    summary["chat_template_system_role_handling"] = json.loads(system_role_file.read_text())
    print("system role handling:", summary["chat_template_system_role_handling"]["handling"])

    banner("5. Activation stage; delete one shard and rerun (only that shard is recomputed)")
    output, seconds = run_pod_script(run_folder, model_directory, "activations")
    summary["activation_seconds"] = seconds
    activations_folder = run_folder / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME
    activation_paths = sorted(activations_folder.glob("activations_shard_*.npz"))
    assert len(activation_paths) == len(shard_paths)
    times = {p: p.stat().st_mtime_ns for p in activation_paths}
    activation_paths[0].unlink()
    run_pod_script(run_folder, model_directory, "activations")
    assert activation_paths[0].exists() and all(p.stat().st_mtime_ns == times[p] for p in activation_paths[1:])
    summary["activation_resume_check"] = "deleted shard recomputed, other shards untouched"

    banner("6. Judge with a MOCKED client")
    answers = judge_answers = compute.load_answers(run_folder)
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    questions_by_id = {q["question_id"]: q for q in read_jsonl(defs.find_latest_file(inputs_folder, "questions", "jsonl"))}
    facet_definitions = json.loads(defs.find_latest_file(inputs_folder, "facet_definitions", "json").read_text())
    requests = judge.build_judge_requests_for_answers(answers, questions_by_id, facet_definitions)
    requests += judge.build_consistency_pair_judge_requests(answers, questions_by_id, 5, 0)
    answer_by_id = {a["answer_id"]: a for a in answers}
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
            return MockedResponse("85" if answer_by_id[request.answer_id]["pole"] == defs.POLE_POSITIVE else "12")
        if request.kind == judge.KIND_COHERENCE:
            return MockedResponse("90")
        return MockedResponse("50")

    judge_folder = run_folder / defs.JUDGEMENTS_FOLDER_NAME
    judge_folder.mkdir()
    estimate = judge.estimate_judge_cost(requests)
    results, failures = judge.run_judge_requests(
        requests, None, results_path=judge_folder / f"judge_results_{stamp}.jsonl",
        usage_log_path=judge_folder / f"judge_usage_and_cost_log_{stamp}.json", call_responses=mocked_call, worker_count=4)
    summary["judge"] = {"mocked": True, "requests": len(requests), "results": len(results), "failures": len(failures),
                        "dry_run_estimate": estimate}
    assert not failures and len(results) == len(requests)
    results_again, _ = judge.run_judge_requests(requests, None, results_path=judge_folder / f"judge_results_{stamp}_resume.jsonl",
                                                call_responses=lambda *_: (_ for _ in ()).throw(AssertionError("called again")),
                                                worker_count=2)
    assert len(results_again) == len(requests)
    summary["judge"]["resume_check"] = "second run made no calls"

    banner("7. Compute the vectors on CPU")
    thresholds = {"positive_minimum": 65.0, "negative_maximum": 35.0, "coherence_minimum": 50.0, "animal_share_maximum_difference": 15.0}
    metadata, metadata_path = compute.compute_all(run_folder, thresholds, configuration["facets"], bootstrap_resamples=20,
                                                  minimum_layer_index=1, vector_set_name="rehearsal_vectors_meaningless", seed=0)
    summary["vectors"] = {"metadata_file": str(metadata_path), "layer_choice": metadata["layer_choice"],
                          "files": sorted(p.name for p in metadata_path.parent.glob("*.npy")),
                          "questions_kept": {f: e.get("questions_kept") for f, e in metadata["per_facet"].items()},
                          "shape": metadata["shape"], "peak_resident_memory_megabytes": metadata["peak_resident_memory_megabytes"]}
    summary["whole_rehearsal_peak_resident_megabytes"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)

    banner("8. Pod launcher dry run on this run folder (free, no pod)")
    dry = subprocess.run([sys.executable, str(Path(__file__).parent / "launch_and_stop_runpod_generation_pod.py"), "--run-folder",
                          str(run_folder), "--maximum-running-hours", "1", "--maximum-dollars", "5", "--dry-run"],
                         capture_output=True, text=True)
    summary["launcher_dry_run"] = {"exit_code": dry.returncode,
                                   "disk_line": next((l for l in dry.stdout.splitlines() if l.startswith("Disk requested")), None)}
    summary_path = run_folder / f"rehearsal_summary_{timestamp_now()}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nREHEARSAL PASSED. Summary: {summary_path}")
    print(json.dumps({k: summary[k] for k in ("generation_jobs", "answer_shards", "generation_seconds", "activation_seconds")}))
    print(json.dumps(summary["vectors"]["layer_choice"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
