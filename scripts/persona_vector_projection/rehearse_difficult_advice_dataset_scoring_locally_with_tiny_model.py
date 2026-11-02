"""Local rehearsal of the difficult-advice scoring run, with no paid call and no pod: a synthetic answers file (user messages and
answers) and a synthetic moralizing delivery vector set go through the real run folder builder, a tiny randomly initialised Gemma 4
model (real Gemma 4 tokenizer from the local Hugging Face cache, bfloat16 as on the pod) scores every answer in both formats through
the real stage runner, one shard is deleted and the run resumed, the scored tokens are checked against the answers' own tokens, and
the launcher's dry run is printed (with placeholder secrets, so .env is not read).

    python3 scripts/persona_vector_projection/rehearse_difficult_advice_dataset_scoring_locally_with_tiny_model.py

Output: data/persona_vector_projection_runs/small_sample_checks/
rehearsal_difficult_advice_dataset_scoring_tiny_model_<timestamp>/ with rehearsal_summary_<timestamp>.json.
What is NOT real: the model (random weights, so every score is noise), the vectors (random) and the answers (synthetic).
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import build_difficult_advice_dataset_scoring_run_folder as builder  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import launch_and_stop_runpod_scoring_and_steering_pod as launcher  # noqa: E402
from persona_vector_projection import run_scoring_pod_stages as stages  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering  # noqa: E402
from persona_vector_projection import scoring_pod_test_support as support  # noqa: E402
from shared.resumable_jsonl_records import write_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

NUMBER_OF_ANSWERS = 14
PLACEHOLDER_SECRETS = {"RUNPOD_API_KEY": None, "HF_TOKEN": None, "RUNPOD_COST_CENTER_NAME": None}
VECTOR_SET_STAMP = "2026-11-01_00-00"


def synthetic_answer_records(number_of_answers=NUMBER_OF_ANSWERS):
    """Answer records with the fields of the collection run's answers file (a subset), user messages that differ per record."""
    records = []
    for number in range(1, number_of_answers + 1):
        records.append({
            "document_id": f"difficult_advice__stage_2__tiny__P-{number:04d}", "dataset_group": "main_fifty_prompts" if number % 2 else "pilot_prompts",
            "prompt_gid": f"P-{number:04d}", "stage": "stage_2", "writer_model": "tiny",
            "user_message": f"USERMESSAGE{number} I run a small clinic and I am unsure whether to keep the night shift. What would you do?",
            "answer_text": f"ANSWER{number} " + " ".join(f"word{(position * 5 + number) % 53}" for position in range(60 + 7 * number))})
    return records


def write_synthetic_moralizing_delivery_vector_set(folder, random_seed=0):
    """A vector set in the extraction run's layout, at the tiny model's size (5 stored layers, chosen layer 2)."""
    definition = facet_definitions.MORALIZING_DELIVERY
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(random_seed)
    vector_files = {}
    for key in definition.vector_set_key_by_vector_name().values():
        file_name = f"persona_vector_{definition.facet_key}_{key}_{VECTOR_SET_STAMP}.npy"
        np.save(folder / file_name, generator.normal(size=(support.TINY_LAYER_COUNT + 1, support.TINY_HIDDEN_SIZE)).astype(np.float32))
        vector_files[key] = file_name
    (folder / f"persona_vector_metadata_{VECTOR_SET_STAMP}.json").write_text(json.dumps({
        "model_name": defs.MODEL_NAME, "model_revision": defs.MODEL_REVISION, "vector_files": vector_files,
        "layer_choice": {"strategy": "shared_layer", "shared_layer": support.TINY_CHOSEN_LAYER,
                         "layer_by_facet": {definition.facet_key: support.TINY_CHOSEN_LAYER}}}))
    return folder


def banner(text):
    print(f"\n=== {text}", flush=True)


def run_rehearsal(output_parent_directory=None, say=print):
    timestamp = timestamp_now()
    parent = Path(output_parent_directory or defs.SMALL_SAMPLE_CHECKS_ROOT)
    rehearsal_folder = parent / f"rehearsal_difficult_advice_dataset_scoring_tiny_model_{timestamp}"
    rehearsal_folder.mkdir(parents=True)
    summary = {"what_this_is": "local rehearsal on a tiny random model; every score is noise", "timestamp": timestamp}
    records = synthetic_answer_records()
    answers_path = rehearsal_folder / f"synthetic_difficult_advice_answers_{timestamp}.jsonl"
    write_jsonl(answers_path, records)
    vector_set_folder = write_synthetic_moralizing_delivery_vector_set(rehearsal_folder / "synthetic_vector_set")

    banner("Scoring run folder")
    arguments = builder.parse_arguments([
        "--answers-jsonl", str(answers_path), "--vector-set", f"moralizing_delivery={vector_set_folder}",
        "--runs-directory", str(rehearsal_folder), "--run-name", "rehearsal_scoring_run", "--expected-document-count", str(len(records))])
    run_folder, configuration, estimate = builder.build_run_folder(arguments)
    summary["scoring_run_folder"] = str(run_folder)
    summary["document_formats"] = configuration["document_formats"]
    summary["scoring_cost_ceiling_estimate"] = estimate
    assert configuration["stages"] == [defs.STAGE_SCORING] and configuration["baselines_per_document"] == 0

    banner("Scoring on the tiny model, with a resumption check")
    tiny_model_directory = rehearsal_folder / "tiny_model"
    tokenizer = support.save_tiny_model_with_real_tokenizer(tiny_model_directory)
    stage_arguments = ["--run-folder", str(run_folder), "--model-name", str(tiny_model_directory), "--model-revision", "", "--device",
                       "cpu", "--dtype", "bfloat16", "--maximum-tokens-per-batch", "12000", "--documents-per-shard", "5"]
    stages.main(stage_arguments)
    outputs = run_folder / defs.OUTPUTS_FOLDER_NAME
    snapshot = lambda: {str(p.relative_to(outputs)): p.stat().st_mtime_ns for p in outputs.rglob("*.npz")}  # noqa: E731
    first_pass = snapshot()
    deleted = sorted(outputs.rglob(f"projections_{defs.FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE}_shard_*.npz"))[-1]
    deleted_name = str(deleted.relative_to(outputs))
    deleted.unlink()
    stages.main(stage_arguments)
    second_pass = snapshot()
    rewritten = sorted(name for name in second_pass if first_pass.get(name) != second_pass[name])
    assert rewritten == [deleted_name] and set(first_pass) == set(second_pass), (rewritten, deleted_name)
    assert not any((outputs / name).exists() for name in (defs.BASELINES_SUBFOLDER, defs.STEERING_SUBFOLDER))
    summary["resumption"] = {"shard_deleted_then_rewritten": deleted_name, "shards": len(second_pass)}

    banner("The scored tokens are the answers' own tokens, in both formats")
    token_ids_by_format = {}
    for document_format in configuration["document_formats"]:
        found = {}
        for path in sorted(outputs.rglob(f"projections_{document_format}_shard_*.npz")):
            with np.load(path) as archive:
                offsets = archive["per_token_offsets"]
                for row, document_id in enumerate(str(i) for i in archive["document_ids"]):
                    found[document_id] = list(archive["per_token_token_ids_int32"][offsets[row]:offsets[row + 1]])
        token_ids_by_format[document_format] = found
    for record in records:
        expected = tokenizer(record["answer_text"].strip(), add_special_tokens=False)["input_ids"]
        for document_format, found in token_ids_by_format.items():
            assert found[record["document_id"]] == expected, (document_format, record["document_id"])
    summary["answers_checked_in_each_format"] = len(records)

    banner("Launcher dry run for the reused pod")
    launcher_arguments = launcher.parse_arguments([
        "--run-folder", str(run_folder), "--maximum-running-hours", "1", "--dry-run", "--reuse-existing-pod", "POD_ID_PLACEHOLDER",
        "--keep-pod-after-run"])
    launcher.apply_job_profile()
    launcher_buffer = io.StringIO()
    with contextlib.redirect_stdout(launcher_buffer):
        launcher.print_estimate(launcher_arguments)
        launcher.shared_launcher.run_launcher(launcher_arguments, secrets=PLACEHOLDER_SECRETS)
    (rehearsal_folder / f"launcher_dry_run_output_{timestamp}.txt").write_text(launcher_buffer.getvalue(), encoding="utf-8")
    summary["launcher_scorer_command"] = launcher.shared_launcher.build_scorer_command_line(launcher_arguments)
    assert "--reuse-existing-pod" in launcher_buffer.getvalue()

    summary_path = rehearsal_folder / f"rehearsal_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    say(f"Rehearsal passed. Summary: {summary_path}")
    return rehearsal_folder, summary


def main(argv=None):
    run_rehearsal()


if __name__ == "__main__":
    main()
