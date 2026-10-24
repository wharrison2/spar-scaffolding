"""Local rehearsal of the whole pilot sequence with no paid call: no Anthropic API, no pod, no .env access.

Steps: rewrites --dry-run; fake "rewrites" (ten corpus documents copied with content unchanged); select_pilot_documents;
build_scoring_inputs on the full 20 + 20 selection (inputs only, not scored); the same on a trimmed set of three rewrite
sources (scored, because scoring all 40 documents on CPU would take about an hour); the CPU scorer with the tiny random
Gemma 4 model and the real Gemma 4 tokenizer (as in the smoke test); the analysis; the go or no go summary (with a fake
pod cost log); and the launcher --dry-run on both run folders.

Everything goes to data/log_probability_persona_check_runs/small_sample_checks/, in folders named rehearsal_*. The scores
are meaningless: the model is random and the "rewrites" are unchanged copies, so every positive-control difference is zero.

Usage: python3 scripts/log_probability_persona_check/rehearse_pilot_pipeline_locally_without_paid_calls.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY.parent, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import analyze_preachiness_scores as analysis  # noqa: E402
import build_scoring_inputs as build_inputs  # noqa: E402
import make_preachiness_positive_control_rewrites as rewrites  # noqa: E402
import score_document_log_probabilities_on_pod as scorer  # noqa: E402
import select_pilot_documents as selector  # noqa: E402
import smoke_test_log_probability_persona_check as smoke_test  # noqa: E402
import summarize_pilot_for_go_or_no_go_decision as summarizer  # noqa: E402
import tiny_model_and_tokenizer_helpers_for_log_probability_persona_check as tiny  # noqa: E402
from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as launcher  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

SMALL_SAMPLE_CHECKS_DIRECTORY = build_inputs.SMALL_SAMPLE_CHECKS_DIRECTORY
NUMBER_OF_SOURCES_IN_TRIMMED_SCORED_RUN = 3
NUMBER_OF_DOCUMENTS_IN_TRIMMED_SCORED_RUN = 5
FAKE_POD_EXTRA_SECONDS = 300  # setup time written to the fake pod cost log


def banner(text):
    print(f"\n{'=' * 100}\n{text}\n{'=' * 100}", flush=True)


def write_fake_unchanged_rewrites(corpus, source_ids, folder):
    by_id = {document["doc_id"]: document for document in corpus}
    folder.mkdir(parents=True)
    records = [{"document_id": f"{source_id}__{direction}", "source_document_id": source_id,
                "rewrite_direction": direction, "content": by_id[source_id]["content"],
                "language": by_id[source_id]["language"], "variables": by_id[source_id]["variables"],
                "rewrite_model_requested": "rehearsal_no_model_used", "rewrite_model_used": "rehearsal_no_model_used"}
               for source_id in source_ids for direction in ("preachier", "plainer")]
    write_jsonl(folder / "rewritten_documents.jsonl", records)
    (folder / "run_configuration_rehearsal.json").write_text(json.dumps({"selected_document_ids": source_ids}))
    (folder / "rehearsal_note.txt").write_text(
        "REHEARSAL ONLY: corpus documents copied with content unchanged and labelled as rewrites. No model wrote them.\n")
    return folder / "rewritten_documents.jsonl"


def write_fake_pod_cost_log(run_folder):
    sessions = summarizer.read_scoring_sessions_from_progress_logs(run_folder / "scoring_outputs")
    scoring_seconds = sum(session["elapsed_seconds"] for session in sessions)
    from datetime import datetime, timedelta
    start = datetime(2026, 1, 1, 0, 0, 0)
    stop = start + timedelta(seconds=scoring_seconds + FAKE_POD_EXTRA_SECONDS)
    (run_folder / "pod_cost_log_rehearsal_fake_no_pod_was_created.json").write_text(json.dumps({
        "REHEARSAL_NOTE": "Fake pod cost log: no pod was created. Values are made up to exercise the summary.",
        "pod_id": "rehearsal_fake_pod", "gpu_type": "none (CPU)", "hourly_price_dollars": 3.0,
        "start_time": start.isoformat(timespec="seconds"), "stop_time": stop.isoformat(timespec="seconds"),
        "termination_reason": "scorer_finished", "total_cost_dollars": 0.0,
        "cost_center_name": "rehearsal_not_a_real_cost_center", "cost_center_assignment_confirmed_by_user": False}))


def main():
    timestamp = timestamp_now()
    corpus = read_jsonl(build_inputs.DEFAULT_CORPUS_PATH)
    working_directory = Path(tempfile.mkdtemp(prefix="rehearsal_tiny_model_"))
    try:
        banner("1. Rewrites dry run (no API call)")
        dry_run_folder = rewrites.main(["--dry-run", "--run-name", f"rehearsal_preachiness_positive_control_rewrites_dry_run"])

        banner("2. Fake rewrites: ten corpus documents copied with content unchanged")
        source_ids = json.loads(next(dry_run_folder.glob("run_configuration_*.json")).read_text())["selected_document_ids"]
        fake_rewrites_folder = SMALL_SAMPLE_CHECKS_DIRECTORY / f"rehearsal_fake_unchanged_content_rewrites_{timestamp}"
        fake_rewrites_path = write_fake_unchanged_rewrites(corpus, source_ids, fake_rewrites_folder)
        print(f"{fake_rewrites_folder}")

        banner("3. select_pilot_documents on the fake rewrites folder (20 documents)")
        ids_path = selector.main(["--positive-control-rewrites-folder", str(fake_rewrites_folder),
                                  "--folder-name-prefix", "rehearsal_pilot_document_selection"])

        banner("4. build_scoring_inputs for the full 20 + 20 selection (inputs only; not scored)")
        full_run_folder = build_inputs.main(["--run-name", "rehearsal_pilot_twenty_plus_twenty_documents_inputs_only",
                                             "--small-sample-check", "--document-ids-file", str(ids_path),
                                             "--extra-documents-jsonl", str(fake_rewrites_path)])

        banner("5. Trimmed set: three rewrite sources, scored on CPU")
        shortest_sources = sorted(source_ids, key=lambda doc_id: len(next(d for d in corpus if d["doc_id"] == doc_id)["content"]))
        trimmed_sources = sorted(shortest_sources[:NUMBER_OF_SOURCES_IN_TRIMMED_SCORED_RUN])
        trimmed_rewrites_folder = SMALL_SAMPLE_CHECKS_DIRECTORY / f"rehearsal_fake_unchanged_content_rewrites_trimmed_to_three_sources_{timestamp}"
        trimmed_rewrites_path = write_fake_unchanged_rewrites(corpus, trimmed_sources, trimmed_rewrites_folder)
        trimmed_ids_path = selector.main(["--positive-control-rewrites-folder", str(trimmed_rewrites_folder),
                                          "--number-of-documents", str(NUMBER_OF_DOCUMENTS_IN_TRIMMED_SCORED_RUN),
                                          "--folder-name-prefix", "rehearsal_trimmed_pilot_document_selection"])
        tiny_model_directory = working_directory / "tiny_random_gemma4_with_real_gemma4_tokenizer"
        tiny_model_directory.mkdir()
        smoke_test.download_tokenizer_files_only(tiny_model_directory)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tiny_model_directory)
        tiny.build_tiny_random_gemma4_model(len(tokenizer), final_logit_softcapping=30.0).save_pretrained(tiny_model_directory)
        tokenizer.save_pretrained(tiny_model_directory)
        scored_run_folder = build_inputs.main([
            "--run-name", "rehearsal_pilot_trimmed_to_three_rewrite_sources_scored_on_cpu_with_tiny_random_model",
            "--small-sample-check", "--document-ids-file", str(trimmed_ids_path),
            "--extra-documents-jsonl", str(trimmed_rewrites_path), "--model-name", str(tiny_model_directory)])
        (scored_run_folder / "rehearsal_note.txt").write_text(
            "REHEARSAL ONLY: scored on CPU with a tiny randomly initialised Gemma 4 model; the 'rewrites' are unchanged "
            "copies of corpus documents. Scores are meaningless; the run checks the pipeline.\n")

        banner("6. CPU scorer")
        scorer.main(["--run-folder", str(scored_run_folder), "--device", "cpu", "--dtype", "float32",
                     "--maximum-tokens-per-batch", "40000", "--verify-first-document-against-full-logits"])

        banner("7. Analysis")
        analysis.run_analysis(scored_run_folder)

        banner("8. Go or no-go summary (fake pod cost log)")
        write_fake_pod_cost_log(scored_run_folder)
        summarizer.main(["--run-folder", str(scored_run_folder)])

        banner("9. Launcher --dry-run (secrets passed in, so .env is not read)")
        no_secrets = {"RUNPOD_API_KEY": None, "HF_TOKEN": None, "RUNPOD_COST_CENTER_NAME": None}
        for folder in (full_run_folder, scored_run_folder):
            launcher.run_launcher(launcher.parse_arguments([
                "--run-folder", str(folder), "--maximum-running-hours", "2", "--maximum-dollars", "10",
                "--verify-first-document-against-full-logits", "--dry-run"]), secrets=no_secrets)
        print(f"\nRehearsal finished.\n  Inputs-only run folder: {full_run_folder}\n  Scored run folder: {scored_run_folder}")
    finally:
        shutil.rmtree(working_directory, ignore_errors=True)


if __name__ == "__main__":
    main()
