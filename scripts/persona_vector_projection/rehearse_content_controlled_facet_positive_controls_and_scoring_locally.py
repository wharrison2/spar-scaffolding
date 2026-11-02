"""Local rehearsal of a content-controlled facet's positive controls and document scoring (`--facet`, default moralizing delivery,
described below), end to end, with no paid call: a MOCKED rewrite
client makes every arm's rewrites (four arms, five for calibrated consideration) (through the real rewrite script, including its dry run, minimum-change retry and export),
the real run folder builder builds the scoring run, a tiny randomly initialised Gemma 4 model (real Gemma 4 tokenizer from the
local Hugging Face cache, bfloat16 as on the pod) scores the 80 documents in both formats through the real stage runner with the
full activations saved, one shard is deleted and the run resumed, the saved activations are projected locally and compared with the
pod's projections, the analysis script runs on the result, and the launcher's dry run is printed (with placeholder secrets, so .env is
not read). Nothing is sent to any API and no pod is created.

    python3 scripts/persona_vector_projection/rehearse_content_controlled_facet_positive_controls_and_scoring_locally.py [--facet <name>]

The stand-in vector set is the latest `rehearsal_*/vector_sets/*` under small_sample_checks/ whose metadata has a layer choice for the
facet and every vector file the facet scores: run the facet's extraction rehearsal first (the extraction chain's rehearsal script).
For the new facets the chained arm is rehearsed too: the mock changes words of whatever document the instruction is applied to, so
the chained arm's rewrite visibly derives from the trait_only_away rewrite (checked: its comparison document is that rewrite).
For the new facets the sources are chosen per facet (`--source-document-ids-file`): half are among the previous run's 20 corpus
documents and half are not, so the run scores the union (25 corpus documents), reads the others from the corpus file, and compares
each rewrite with its own source (checked).

Output: data/persona_vector_projection_runs/small_sample_checks/
rehearsal_moralizing_delivery_positive_controls_and_scoring_tiny_model_mocked_rewrite_client_<timestamp>/ with rehearsal_summary_<timestamp>.json.
What is NOT real: the rewrites (a mock changes words in the document), the model (random weights, so every score is noise), the vectors
(the meaningless rehearsal vector set of the earlier moralizing delivery rehearsal, read and never written) and the corpus (synthetic).
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import analyse_content_controlled_facet_positive_controls_and_corpus_windows as analysis  # noqa: E402
from persona_vector_projection import build_content_controlled_facet_scoring_run_folder as builder  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import launch_and_stop_runpod_scoring_and_steering_pod as launcher  # noqa: E402
from persona_vector_projection import make_content_controlled_facet_positive_control_rewrites as rewriter  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_arms as arms_module  # noqa: E402
from persona_vector_projection import run_scoring_pod_stages as stages  # noqa: E402
from persona_vector_projection import saved_activation_projection_locally as saved_activations_module  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_test_support as support  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

NUMBER_OF_CORPUS_DOCUMENTS = 20
NUMBER_OF_SOURCES = 10
NUMBER_OF_CORPUS_DOCUMENTS_NOT_SCORED_BY_THE_PREVIOUS_RUN = 10
FIRST_NUMBER_OF_THE_NEW_FACET_SOURCES = 16   # sources 16 to 25: 16 to 20 were scored by the previous run, 21 to 25 were not
WORDS_PER_DOCUMENT = 260
WORDS_CHANGED_BY_THE_MOCK = 90
PLACEHOLDER_SECRETS = {"RUNPOD_API_KEY": None, "HF_TOKEN": None, "RUNPOD_COST_CENTER_NAME": None}


def banner(text):
    print(f"\n=== {text}", flush=True)


def synthetic_document_text(number, tag=""):
    words = [f"{tag}word{(position * 7 + number) % 61}" for position in range(WORDS_PER_DOCUMENT)]
    return f"Title {number}\n\n" + " ".join(words)


def change_words(text, how_many, tag):
    words = text.split(" ")
    for position in range(100, min(100 + how_many, len(words))):
        words[position] = f"{tag}{position}"
    return " ".join(words)


def write_synthetic_inputs(directory):
    """A synthetic corpus, the earlier preachier and plainer rewrites of its first sources, and the 'previous scoring run' that scored
    them (documents.jsonl and a run configuration)."""
    corpus = [support.make_corpus_record(number, synthetic_document_text(number))
              for number in range(1, NUMBER_OF_CORPUS_DOCUMENTS + NUMBER_OF_CORPUS_DOCUMENTS_NOT_SCORED_BY_THE_PREVIOUS_RUN + 1)]
    sources = corpus[:NUMBER_OF_SOURCES]
    preachiness = []
    for source in sources:
        for direction in ("preachier", "plainer"):
            preachiness.append({"document_id": f"{source['doc_id']}__{direction}", "source_document_id": source["doc_id"],
                                "rewrite_direction": direction, "content": change_words(source["content"], 60, direction),
                                "language": source["language"], "variables": source["variables"]})
    previous_folder = Path(directory) / "previous_scoring_run"
    (previous_folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True)
    documents = [{"document_id": r["doc_id"], "source_document_id": r["doc_id"], "rewrite_direction": None, "content": r["content"],
                  "language": r["language"], "variables": r["variables"], "kind": "corpus"} for r in corpus[:NUMBER_OF_CORPUS_DOCUMENTS]]
    documents += [{**r, "kind": "rewrite"} for r in preachiness]
    write_jsonl(previous_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl", documents)
    (previous_folder / "run_configuration_2026-10-31_11-20.json").write_text(json.dumps(
        {"model_name": defs.MODEL_NAME, "model_revision": defs.MODEL_REVISION}), encoding="utf-8")
    corpus_path, preachiness_path = Path(directory) / "synthetic_corpus.jsonl", Path(directory) / "synthetic_preachiness_rewrites.jsonl"
    write_jsonl(corpus_path, corpus)
    write_jsonl(preachiness_path, preachiness)
    return corpus_path, preachiness_path, previous_folder


def arm_of_prompt_from_templates(prompt, template_text_by_arm):
    """The arm whose instruction template's opening (its text before the first {slot}, at most 300 characters) starts the prompt;
    raises when none or several match, so a mock can never answer the wrong arm's instruction."""
    matches = []
    for arm, template in template_text_by_arm.items():
        opening = template.split("{")[0][:300]
        if opening and prompt.startswith(opening):
            matches.append(arm)
    if len(matches) != 1:
        raise AssertionError(f"A prompt matched arms {matches}, expected exactly one")
    return matches[0]


def mocked_rewrite_client(definition=None):
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    """The fake streaming client of test_facet_positive_control_rewrites.py, answering each arm's instruction by changing words of
    the document in the prompt."""
    from persona_vector_projection import test_facet_positive_control_rewrites as fakes
    file_by_arm = arms_module.resolve_instruction_file_names(definition, defs.PROMPT_FOLDER)
    template_text_by_arm = {arm: (Path(defs.PROMPT_FOLDER) / name).read_text(encoding="utf-8") for arm, name in file_by_arm.items()}

    def response_for(prompt):
        arm = arm_of_prompt_from_templates(prompt, template_text_by_arm)
        return fakes.make_response(change_words(fakes.document_text_in(prompt), WORDS_CHANGED_BY_THE_MOCK, arm.split("_")[0]))
    return fakes.FakeClient(response_for)


def find_stand_in_vector_set(definition=None):
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    wanted_keys = set(definition.vector_set_key_by_vector_name().values())
    for folder in reversed(sorted(defs.SMALL_SAMPLE_CHECKS_ROOT.glob("rehearsal_*/vector_sets/*"))):
        for metadata_path in sorted(folder.glob("persona_vector_metadata_*.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (definition.facet_key in ((metadata.get("layer_choice") or {}).get("layer_by_facet") or {})
                    and wanted_keys <= set(metadata.get("vector_files", {}))):
                return folder
    raise FileNotFoundError(f"No rehearsal vector set for {definition.facet_key} under small_sample_checks/ to stand in for the real "
                            f"one; run the facet's extraction rehearsal first")


def run_rehearsal(output_parent_directory=None, circular_draws=2000, bootstrap_draws=5000, say=print, facet=None):
    definition = facet_definitions.get_facet_definition(facet or facet_definitions.FACET_MORALIZING_DELIVERY)
    timestamp = timestamp_now()
    parent = Path(output_parent_directory or defs.SMALL_SAMPLE_CHECKS_ROOT)
    rehearsal_folder = parent / (f"rehearsal_{definition.facet_key}_positive_controls_and_scoring_tiny_model_mocked_rewrite_client_"
                                 f"{timestamp}")
    rehearsal_folder.mkdir(parents=True)
    summary = {"what_this_is": "local rehearsal on a tiny random model with a mocked rewrite client; every score is noise",
               "timestamp": timestamp}
    working_directory = Path(tempfile.mkdtemp(prefix=f"{definition.facet_key}_scoring_rehearsal_"))
    corpus_path, preachiness_path, previous_folder = write_synthetic_inputs(working_directory)
    stand_in_vector_set = find_stand_in_vector_set(definition)
    summary["stand_in_vector_set_read_only"] = str(stand_in_vector_set)

    banner("Rewrites: dry run, then the mocked paid step")
    common = ["--facet", definition.facet_key, "--corpus-path", str(corpus_path), "--preachiness-rewrites-jsonl", str(preachiness_path), "--workers", "1",
              "--env-path", str(working_directory / "no_such_env_file")]
    new_facet_source_ids = [f"matrix_{number:06d}" for number in range(
        FIRST_NUMBER_OF_THE_NEW_FACET_SOURCES, FIRST_NUMBER_OF_THE_NEW_FACET_SOURCES + NUMBER_OF_SOURCES)]
    if not definition.legacy_moralizing_names:
        source_ids_path = working_directory / "rewrite_source_document_ids_half_outside_the_previous_run.json"
        source_ids_path.write_text(json.dumps(new_facet_source_ids), encoding="utf-8")
        common += ["--source-document-ids-file", str(source_ids_path)]
    dry_run_buffer = io.StringIO()
    with contextlib.redirect_stdout(dry_run_buffer):
        dry_folder = rewriter.main([*common, "--dry-run", "--run-name", "rehearsal_dry_run"], runs_directory_override=rehearsal_folder)
    estimate = json.loads(sorted(Path(dry_folder).glob("dry_run_cost_estimate_and_ceiling_*.json"))[0].read_text())
    summary["rewrite_dry_run"] = {key: estimate[key] for key in ("calls_planned", "calls_planned_by_arm", "expected_input_tokens",
                                                                 "expected_output_tokens", "expected_cost_dollars", "ceiling_cost_dollars")}
    client = mocked_rewrite_client(definition)
    rewrite_folder = rewriter.main([*common, "--run-name", "rehearsal_mocked_rewrites"], client=client, runs_directory_override=rehearsal_folder)
    export_path = sorted(Path(rewrite_folder).glob("scoring_pod_extra_documents_*.jsonl"))[0]
    exported = read_jsonl(export_path)
    summary["rewrites"] = {"calls_made_to_the_mock_client": len(client.calls), "records_exported": len(exported),
                           "records_by_arm": {arm: sum(1 for r in exported if r["arm"] == arm) for arm in arms_module.arm_names_for(definition)}}
    assert len(exported) == len(arms_module.arm_names_for(definition)) * NUMBER_OF_SOURCES
    if not definition.legacy_moralizing_names:
        chained = next(r for r in exported if r["arm"] == facet_definitions.ARM_TRAIT_REMOVED_FROM_AWAY_REWRITE)
        assert chained["comparison_document_id"] == arms_module.rewrite_document_id_for(
            definition, facet_definitions.ARM_TRAIT_ONLY_AWAY, chained["original_corpus_source_document_id"])

    banner("Scoring run folder")
    builder_arguments = builder.parse_arguments([
        "--facet", definition.facet_key, "--vector-set-folder", str(stand_in_vector_set), "--new-rewrites-jsonl", str(export_path),
        "--previous-scoring-run-folder", str(previous_folder), "--preachiness-rewrites-jsonl", str(preachiness_path),
        "--corpus-path", str(corpus_path),
        "--expected-corpus-documents", str(NUMBER_OF_CORPUS_DOCUMENTS), "--expected-sources", str(NUMBER_OF_SOURCES),
        "--runs-directory", str(rehearsal_folder), "--run-name", "rehearsal_scoring_run"])
    run_folder, configuration, estimate = builder.build_run_folder(builder_arguments)
    summary["scoring_run_folder"] = str(run_folder)
    summary["documents_by_group"] = configuration["documents_by_group"]
    summary["corpus_documents_scored"] = configuration["corpus_documents_scored"]
    if not definition.legacy_moralizing_names:
        scored_documents = {d["document_id"]: d for d in read_jsonl(run_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl")}
        scored_corpus_ids = [i for i, d in scored_documents.items() if d["kind"] == "corpus"]
        assert len(scored_corpus_ids) == len(set(scored_corpus_ids)) == NUMBER_OF_CORPUS_DOCUMENTS + NUMBER_OF_SOURCES // 2
        assert set(new_facet_source_ids) <= set(scored_corpus_ids)
        assert all(d["source_document_id"] in scored_documents for d in scored_documents.values())
    summary["scoring_cost_ceiling_estimate"] = estimate

    banner("Scoring on the tiny model, saving the full activations, with a resumption check")
    tiny_model_directory = working_directory / "tiny_model"
    support.save_tiny_model_with_real_tokenizer(tiny_model_directory)
    stage_arguments = ["--run-folder", str(run_folder), "--model-name", str(tiny_model_directory), "--model-revision", "", "--device",
                       "cpu", "--dtype", "bfloat16", "--maximum-tokens-per-batch", "12000", "--documents-per-shard", "8",
                       "--save-full-activations-for-formats", "assistant_turn"]  # no --stages: the run configuration's own stage list
    stages.main(stage_arguments)
    outputs = run_folder / defs.OUTPUTS_FOLDER_NAME
    snapshot = lambda: {str(p.relative_to(outputs)): p.stat().st_mtime_ns for p in outputs.rglob("*.npz")}  # noqa: E731
    first_pass = snapshot()
    deleted = sorted(outputs.rglob("projections_assistant_turn_shard_*.npz"))[-1]
    deleted_name = str(deleted.relative_to(outputs))
    deleted.unlink()
    stages.main(stage_arguments)
    second_pass = snapshot()
    rewritten = sorted(name for name in second_pass if first_pass.get(name) != second_pass[name])
    assert rewritten == [deleted_name] and set(first_pass) == set(second_pass), (rewritten, deleted_name)
    summary["resumption"] = {"shard_deleted_then_rewritten": deleted_name, "shards": len(second_pass),
                             "baseline_or_steering_outputs_written": any(
                                 (outputs / name).exists() for name in (defs.BASELINES_SUBFOLDER, defs.STEERING_SUBFOLDER))}
    assert not summary["resumption"]["baseline_or_steering_outputs_written"]
    arrays = np.load(sorted(outputs.rglob("projections_assistant_turn_shard_*.npz"))[0])
    user_arrays = np.load(sorted(outputs.rglob("projections_user_message_shard_*.npz"))[0])
    assert saved_activations_module.SAVED_ACTIVATIONS_KEY in arrays.files and saved_activations_module.SAVED_ACTIVATIONS_KEY not in user_arrays.files
    summary["saved_activations_array_shape_first_shard"] = list(arrays[saved_activations_module.SAVED_ACTIVATIONS_KEY].shape)

    banner("Analysis (free)")
    tokenizer = support.load_real_gemma_tokenizer()
    decode = lambda ids: tokenizer.decode([int(i) for i in ids])  # noqa: E731
    analysis_folder, verdicts = analysis.run_analysis(
        run_folder, corpus_path, [defs.FORMAT_ASSISTANT_TURN, defs.FORMAT_USER_MESSAGE], 0, circular_draws, bootstrap_draws, decode,
        hand_reading=None, verify_saved=True, say=say)
    summary["analysis_folder"] = str(analysis_folder)
    summary["saved_activations_recomputed_locally"] = verdicts["saved_activations_check"]
    assert all(report["within_tolerance"] for report in verdicts["saved_activations_check"].values())
    summary["criteria_outcome_is_noise"] = {"counts": verdicts["summary"], "decision": verdicts["decision_for_the_main_vector_in_assistant_turn"]}

    banner("Launcher dry run for the reused pod")
    launcher_arguments = launcher.parse_arguments([
        "--run-folder", str(run_folder), "--maximum-running-hours", "1", "--dry-run", "--reuse-existing-pod", "POD_ID_PLACEHOLDER",
        "--keep-pod-after-run", "--save-full-activations-for-formats", "assistant_turn"])
    launcher.apply_job_profile()
    launcher.shared_launcher.MINIMUM_LOCAL_FREE_DISK_BYTES = launcher.MINIMUM_LOCAL_FREE_DISK_BYTES_WHEN_SAVING_FULL_ACTIVATIONS
    launcher_buffer = io.StringIO()
    with contextlib.redirect_stdout(launcher_buffer):
        launcher.print_estimate(launcher_arguments)
        launcher.shared_launcher.run_launcher(launcher_arguments, secrets=PLACEHOLDER_SECRETS)
    launcher_text = launcher_buffer.getvalue()
    (rehearsal_folder / f"launcher_dry_run_output_{timestamp}.txt").write_text(launcher_text, encoding="utf-8")
    summary["launcher_scorer_command"] = launcher.shared_launcher.build_scorer_command_line(launcher_arguments)
    assert "--save-full-activations-for-formats" in summary["launcher_scorer_command"]
    assert "--reuse-existing-pod" in launcher_text and "KEPT running" in launcher_text

    summary_path = rehearsal_folder / f"rehearsal_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    say(f"Rehearsal passed. Summary: {summary_path}")
    return rehearsal_folder, summary


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    facet_definitions.add_facet_argument(parser)
    arguments = parser.parse_args(argv)
    run_rehearsal(facet=arguments.facet)


if __name__ == "__main__":
    main()
