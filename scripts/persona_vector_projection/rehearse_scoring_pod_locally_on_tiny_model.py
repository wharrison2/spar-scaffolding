"""Local rehearsal of the scoring pod on CPU: no pod, no API, no paid call.

    python3 scripts/persona_vector_projection/rehearse_scoring_pod_locally_on_tiny_model.py

Builds a pilot run folder from the REAL corpus and the REAL positive-control rewrites, runs all three stages through the
real command line (stages in child processes) on a tiny randomly initialised Gemma 4 model with the real Gemma 4
tokenizer (read from the local Hugging Face cache), deletes one shard of every stage and reruns to check resumption, and
runs the launcher's dry run with injected placeholder secrets (so .env is not read). Output goes to
data/persona_vector_projection_runs/small_sample_checks/. What is NOT real here: the model (random weights, so the
generated text is noise and the scores mean nothing), the vectors (synthetic, topic projected out), the held-out
questions (synthetic), and vLLM (the Hugging Face backend stands in; the vLLM path is untested locally).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_test_support as support  # noqa: E402
from persona_vector_projection import build_scoring_pod_run_folder as builder  # noqa: E402
from persona_vector_projection import run_scoring_pod_stages as stages  # noqa: E402
from persona_vector_projection import launch_and_stop_runpod_scoring_and_steering_pod as launcher  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402


def snapshot(run_folder):
    return {str(p.relative_to(run_folder)): p.stat().st_mtime_ns
            for p in (run_folder / defs.OUTPUTS_FOLDER_NAME).rglob("*") if p.is_file() and "progress_log" not in p.name
            and "environment" not in p.name}


def main():
    working_directory = Path(tempfile.mkdtemp(prefix="scoring_pod_rehearsal_"))
    support.save_tiny_model_with_real_tokenizer(working_directory / "tiny_model")
    support.write_synthetic_vector_folder(working_directory / "vectors")
    support.write_synthetic_converted_inputs(working_directory / "converted", facets=[defs.PERSONA_FACETS[2]])
    stamp_for_name = timestamp_now()
    run_name = f"rehearsal_pilot_scoring_pod_tiny_random_model_synthetic_vectors_{stamp_for_name}"
    arguments = builder.parse_arguments([
        "--mode", "pilot", "--vector-folder", str(working_directory / "vectors"),
        "--converted-inputs-folder", str(working_directory / "converted"), "--maximum-new-tokens", "24",
        "--small-sample-check", "--run-name", run_name, "--baselines-per-document", "2",
        "--steering-coefficients", "-1", "0", "1"])
    run_folder, configuration, estimate = builder.build_run_folder(arguments)
    print(f"Run folder: {run_folder}")
    common = ["--run-folder", str(run_folder), "--model-name", str(working_directory / "tiny_model"), "--model-revision", "",
              "--device", "cpu", "--dtype", "float32", "--generation-backend", "huggingface", "--steering-maximum-new-tokens", "12",
              "--maximum-tokens-per-batch", "12000"]
    stages.main(common)
    first_pass = snapshot(run_folder)
    # Resumption: delete the last shard of each stage and rerun; only those shards may be rewritten.
    deleted = []
    for pattern in ("baseline_documents_shard_*.jsonl", "projections_*_shard_*.npz", "answers_shard_*_steering_*.jsonl"):
        shard = sorted((run_folder / defs.OUTPUTS_FOLDER_NAME).rglob(pattern))[-1]
        deleted.append(str(shard.relative_to(run_folder)))
        shard.unlink()
    stages.main(common)
    second_pass = snapshot(run_folder)
    rewritten = sorted(name for name in second_pass if first_pass.get(name) != second_pass[name])
    assert rewritten == sorted(deleted), (rewritten, deleted)
    assert set(first_pass) == set(second_pass)
    # Checks on the outputs
    arrays = np.load(sorted((run_folder / defs.OUTPUTS_FOLDER_NAME / defs.PROJECTIONS_SUBFOLDER / "assistant_turn").glob("*.npz"))[0])
    summary = {
        "run_folder": str(run_folder), "what_this_is": "local rehearsal on a tiny random model; scores are noise",
        "deleted_shards_then_rewritten_on_second_run": deleted, "files_after_first_run": len(first_pass),
        "projection_arrays": {key: list(arrays[key].shape) for key in arrays.files},
        "formats_scored": sorted(p.name for p in (run_folder / defs.OUTPUTS_FOLDER_NAME / defs.PROJECTIONS_SUBFOLDER).iterdir()),
        "scoring_failures_file_exists": any((run_folder / defs.OUTPUTS_FOLDER_NAME).glob("scoring_failures_*.jsonl")),
        "gpu_hours_and_cost_estimate": estimate}
    # Launcher dry run with placeholder secrets: .env is never read
    arguments_for_launcher = launcher.parse_arguments(["--run-folder", str(run_folder), "--maximum-running-hours", "1.5", "--dry-run",
                                                       "--stages", "baselines", "scoring", "steering"])
    launcher.apply_job_profile()
    launcher.print_estimate(arguments_for_launcher)
    launcher.shared_launcher.run_launcher(arguments_for_launcher, secrets={"RUNPOD_API_KEY": None, "HF_TOKEN": None,
                                                                            "RUNPOD_COST_CENTER_NAME": None})
    summary["launcher_scorer_command"] = launcher.shared_launcher.build_scorer_command_line(arguments_for_launcher)
    summary_path = run_folder / f"rehearsal_summary_{timestamp_now()}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Rehearsal passed. Summary: {summary_path}")


if __name__ == "__main__":
    main()
