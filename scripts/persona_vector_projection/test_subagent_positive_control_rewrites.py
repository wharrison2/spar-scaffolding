"""Tests for the Claude Code subagent route to the content-controlled positive-control rewrites (render_positive_control_rewrite_prompts_for_subagents.py
and ingest_subagent_positive_control_rewrites.py). Local and free: no API call, no .env read, no pod; a simulated subagent writes the reply files.

    python3 -m pytest scripts/persona_vector_projection/test_subagent_positive_control_rewrites.py -q
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import build_content_controlled_facet_scoring_run_folder as builder  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as D  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_arms as arms  # noqa: E402
from persona_vector_projection import ingest_subagent_positive_control_rewrites as ingest  # noqa: E402
from persona_vector_projection import make_content_controlled_facet_positive_control_rewrites as paid_script  # noqa: E402
from persona_vector_projection import make_facet_positive_control_rewrites as facet_rewrites  # noqa: E402
from persona_vector_projection import render_positive_control_rewrite_prompts_for_subagents as render  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as defs  # noqa: E402
from persona_vector_projection import subagent_positive_control_rewrite_run_state as run_state  # noqa: E402
from persona_vector_projection import test_facet_positive_control_rewrites as fakes  # noqa: E402
from shared.resumable_jsonl_records import read_all_records, read_jsonl, write_jsonl  # noqa: E402

AWAY, CONTENT, REMOVED, NEUTRAL = (D.ARM_TRAIT_ONLY_AWAY, D.ARM_CONTENT_ADDED_WITHOUT_TRAIT, D.ARM_TRAIT_REMOVED_FROM_AWAY_REWRITE,
                                   D.ARM_FACET_NEUTRAL_REWORDING)
FACETS = pytest.mark.parametrize("definition", [D.CARES_ABOUT_WELFARE, D.CALIBRATED_CONSIDERATION, D.NON_INTRUSIVE], ids=lambda d: d.facet_name)
MINIMUM_CHANGED_WORDS, TARGET_CHANGED_WORDS = 30, 80
SOURCE_IDS = ["matrix_000001", "matrix_000002", "matrix_000003", "matrix_000004"]


@pytest.fixture
def inputs(tmp_path):
    corpus = fakes.make_corpus(12)
    corpus_path, sources_path = tmp_path / "corpus.jsonl", tmp_path / "sources.json"
    write_jsonl(corpus_path, corpus)
    sources_path.write_text(json.dumps(SOURCE_IDS))
    return corpus, corpus_path, sources_path


def render_arguments(tmp_path, inputs, definition, wave, extra=()):
    _, corpus_path, sources_path = inputs
    return ["--facet", definition.facet_name, "--wave", str(wave), "--corpus-path", str(corpus_path),
            "--source-document-ids-file", str(sources_path), "--runs-directory", str(tmp_path / "runs"),
            *(["--minimum-changed-units", str(MINIMUM_CHANGED_WORDS), "--target-changed-units", str(TARGET_CHANGED_WORDS)]
              if not extra else []), *extra]


def run_render(tmp_path, inputs, definition, wave, extra=()):
    with contextlib.redirect_stdout(io.StringIO()):
        manifest_path = render.main(render_arguments(tmp_path, inputs, definition, wave, extra))
    return manifest_path, json.loads(Path(manifest_path).read_text())


def run_ingest(definition, run_folder):
    with contextlib.redirect_stdout(io.StringIO()):
        return ingest.main(["--facet", definition.facet_name, "--run-folder", str(run_folder)])


def prompt_document(prompt_text):
    return fakes.document_text_in(prompt_text)


def write_replies(manifest, reply_for):
    """The simulated subagent: reads each prompt file and writes the reply `reply_for(entry, prompt_text)` to the output path."""
    for entry in manifest["rewrites"]:
        prompt_text = Path(entry["prompt_path"]).read_text(encoding="utf-8")
        Path(entry["output_path"]).write_text(reply_for(entry, prompt_text), encoding="utf-8")


def generous(entry, prompt_text):
    return fakes.change_words(prompt_document(prompt_text), 100, entry["arm"].replace("_", ""))


# ---------------------------------------------------------------------------------------------------------------
# Prompts equal the paid script's
# ---------------------------------------------------------------------------------------------------------------

@FACETS
def test_wave_one_prompts_equal_the_paid_scripts_rendered_prompts(tmp_path, inputs, definition):
    _, corpus_path, sources_path = inputs
    paid_folder = paid_script.main(
        ["--facet", definition.facet_name, "--corpus-path", str(corpus_path), "--source-document-ids-file", str(sources_path), "--dry-run",
         "--minimum-changed-units", str(MINIMUM_CHANGED_WORDS), "--target-changed-units", str(TARGET_CHANGED_WORDS)],
        client=None, runs_directory_override=tmp_path / "paid_dry_runs")
    paid_prompts = {(r["arm"], r["original_source_id"]): r["prompt_text"]
                    for r in read_jsonl(sorted(Path(paid_folder).glob("rendered_rewrite_requests_*.jsonl"))[0])
                    if not r["document_is_placeholder_for_arm_one_output"]}
    _, manifest = run_render(tmp_path, inputs, definition, 1)
    independent_arm_count = len(arms.arm_names_for(definition)) - 1
    assert manifest["pending_count"] == len(SOURCE_IDS) * independent_arm_count == len(paid_prompts)
    for entry in manifest["rewrites"]:
        assert REMOVED != entry["arm"] and entry["attempt_number"] == 1 and not entry["is_retry_with_below_minimum_note"]
        prompt_text = Path(entry["prompt_path"]).read_text(encoding="utf-8")
        assert prompt_text == paid_prompts[(entry["arm"], entry["original_source_document_id"])]
        assert entry["output_path"].endswith(f"subagent_outputs/{entry['rewrite_id']}.txt")
        assert entry["prompt_path"].endswith(f"subagent_prompts/{entry['rewrite_id']}.txt")
    assert "claude_code_opus_5_5_subagents" in Path(manifest["run_folder"]).name


def test_the_four_amounts_and_the_retry_note_are_filled_in_and_braces_in_the_document_survive(tmp_path, inputs):
    definition = D.CARES_ABOUT_WELFARE
    _, manifest = run_render(tmp_path, inputs, definition, 1)
    prompt_text = Path(manifest["rewrites"][0]["prompt_path"]).read_text(encoding="utf-8")
    assert "about 80 words" in prompt_text and "between 60 words and 100 words" in prompt_text
    assert not any(slot in prompt_text for slot in ("{target_changed_amount}", "{lowest_changed_amount}", "{highest_changed_amount}",
                                                    "{minimum_changed_amount}", "{document_text}"))
    run_folder = Path(manifest["run_folder"])
    write_replies(manifest, lambda entry, prompt: fakes.change_words(prompt_document(prompt), 5, "tiny"))
    run_ingest(definition, run_folder)
    _, retry_manifest = run_render(tmp_path, inputs, definition, 1)
    assert retry_manifest["pending_count"] == len(manifest["rewrites"])
    for entry in retry_manifest["rewrites"]:
        retry_prompt = Path(entry["prompt_path"]).read_text(encoding="utf-8")
        assert entry["attempt_number"] == 2 and entry["is_retry_with_below_minimum_note"]
        assert "Note on a previous attempt: your previous rewrite changed only about 5 words" in retry_prompt
        assert "at least 30 words differ from the original" in retry_prompt
        assert retry_prompt.rstrip("\n").endswith(f"Document:\n{prompt_document(retry_prompt)}")


# ---------------------------------------------------------------------------------------------------------------
# The chained arm
# ---------------------------------------------------------------------------------------------------------------

@FACETS
def test_chained_arm_uses_the_accepted_trait_only_away_text_never_the_source(tmp_path, inputs, definition):
    corpus, _, _ = inputs
    corpus_by_id = {d["doc_id"]: d for d in corpus}
    manifest_path, wave_two_before = run_render(tmp_path, inputs, definition, 2)
    assert wave_two_before["pending_count"] == 0 and len(wave_two_before["chained_rewrites_without_accepted_source"]) == len(SOURCE_IDS)
    _, wave_one = run_render(tmp_path, inputs, definition, 1)
    run_folder = Path(wave_one["run_folder"])
    write_replies(wave_one, generous)
    ingest_outcome = run_ingest(definition, run_folder)
    assert ingest_outcome["missing"] and not list(run_folder.glob("scoring_pod_extra_documents_*.jsonl"))  # chained arm still to come
    accepted_away = {r["original_corpus_source_document_id"]: r for r in run_state.read_accepted_records(run_folder, definition) if r["arm"] == AWAY}
    _, wave_two = run_render(tmp_path, inputs, definition, 2)
    assert wave_two["pending_count"] == len(SOURCE_IDS) and {e["arm"] for e in wave_two["rewrites"]} == {REMOVED}
    for entry in wave_two["rewrites"]:
        prompt_text = Path(entry["prompt_path"]).read_text(encoding="utf-8")
        source_id = entry["original_source_document_id"]
        assert prompt_document(prompt_text) == accepted_away[source_id]["content"]
        assert prompt_document(prompt_text) != corpus_by_id[source_id]["content"]
    write_replies(wave_two, lambda entry, prompt: fakes.change_words(prompt_document(prompt), 100, "back"))
    run_ingest(definition, run_folder)
    chained = [r for r in run_state.read_accepted_records(run_folder, definition) if r["arm"] == REMOVED]
    assert len(chained) == len(SOURCE_IDS)
    for record in chained:
        source_id = record["original_corpus_source_document_id"]
        away_id = arms.rewrite_document_id_for(definition, AWAY, source_id)
        assert record["source_document_id"] == record["comparison_document_id"] == away_id
        away_text = accepted_away[source_id]["content"]
        assert facet_rewrites.count_changed_units(away_text, record["content"], record["language"])["changed_rewrite_units"] == record["changed_rewrite_units"]


# ---------------------------------------------------------------------------------------------------------------
# Validation matches the paid script
# ---------------------------------------------------------------------------------------------------------------

def scripted_reply_text(prompt_text):
    """Behaviour by source document (found from the document text): generous, stubborn until the retry note, or never enough."""
    document = prompt_document(prompt_text)
    source_index = int(document.split("_")[0].lstrip("w")) % 3 if document.startswith("w") else 0
    if source_index == 0:
        return fakes.change_words(document, 100, "gen")
    if source_index == 1:
        return fakes.change_words(document, 100 if "previous attempt" in prompt_text else 5, "stub")
    return "  \n" + fakes.change_words(document, 5, "few") + "\n\n  "


@FACETS
def test_every_accepted_record_and_the_export_equal_the_paid_scripts_for_the_same_replies(tmp_path, inputs, definition):
    _, corpus_path, sources_path = inputs
    client = fakes.FakeClient(lambda prompt: fakes.make_response(scripted_reply_text(prompt)))
    paid_folder = paid_script.main(
        ["--facet", definition.facet_name, "--corpus-path", str(corpus_path), "--source-document-ids-file", str(sources_path), "--workers", "1",
         "--minimum-changed-units", str(MINIMUM_CHANGED_WORDS), "--target-changed-units", str(TARGET_CHANGED_WORDS),
         "--env-path", str(tmp_path / "no_such_env_file")], client=client, runs_directory_override=tmp_path / "paid_runs")
    paid_export = {r["document_id"]: r for r in read_jsonl(sorted(Path(paid_folder).glob("scoring_pod_extra_documents_*.jsonl"))[0])}
    paid_records = {r["document_id"]: r for r in read_all_records(paid_folder, arms.rewrites_file_pattern(definition))}

    run_folder = None
    for wave in (1, 1, 1, 2, 2, 2):  # the waves repeat for the retry rounds (up to three attempts per rewrite)
        _, manifest = run_render(tmp_path, inputs, definition, wave)
        run_folder = Path(manifest["run_folder"])
        write_replies(manifest, lambda entry, prompt: scripted_reply_text(prompt))
        run_ingest(definition, run_folder)
    assert not ingest.completion_check(run_folder, definition, SOURCE_IDS)
    subagent_export_path = sorted(run_folder.glob("scoring_pod_extra_documents_*.jsonl"))[-1]
    subagent_export = {r["document_id"]: r for r in read_jsonl(subagent_export_path)}
    assert set(subagent_export) == set(paid_export) and len(subagent_export) == len(SOURCE_IDS) * len(arms.arm_names_for(definition))
    assert subagent_export == paid_export  # field for field, flagged rewrites included
    for rewrite_id, record in {r["document_id"]: r for r in run_state.read_accepted_records(run_folder, definition)}.items():
        paid_record = dict(paid_records[rewrite_id])
        for volatile in ("rewrite_model_requested", "rewrite_model_used"):
            assert record.pop(volatile) == run_state.SUBAGENT_MODEL_ID and paid_record.pop(volatile) == "claude-opus-5"
        assert record == paid_record
    assert any(r["below_minimum_changed_units"] for r in subagent_export.values())  # the never-enough sources are kept and flagged
    assert {r["number_of_attempts"] for r in paid_records.values()} == {r["number_of_attempts"] for r in
                                                                         run_state.read_accepted_records(run_folder, definition)}


REJECTED_REPLIES = {
    "empty": ("", run_state.FAILURE_EMPTY_REPLY),
    "only_whitespace": ("  \n\n ", run_state.FAILURE_EMPTY_REPLY),
    "preamble": ("Here is the rewritten document:\n\n{body}", run_state.FAILURE_PREAMBLE_OR_WRAPPER),
    "sure": ("Sure, here you go.\n{body}", run_state.FAILURE_PREAMBLE_OR_WRAPPER),
    "code_fence": ("```\n{body}\n```", run_state.FAILURE_PREAMBLE_OR_WRAPPER),
    "closing_note": ("{body}\n\nNote: I changed the second paragraph.", run_state.FAILURE_PREAMBLE_OR_WRAPPER)}


@pytest.mark.parametrize("case", sorted(REJECTED_REPLIES))
def test_rejected_replies_fail_with_a_reason_are_kept_and_get_the_same_prompt_again(tmp_path, inputs, case):
    definition = D.NON_INTRUSIVE
    _, manifest = run_render(tmp_path, inputs, definition, 1)
    run_folder = Path(manifest["run_folder"])
    first = manifest["rewrites"][0]
    reply_template, expected_kind = REJECTED_REPLIES[case]
    original_prompt = Path(first["prompt_path"]).read_text(encoding="utf-8")
    body = fakes.change_words(prompt_document(original_prompt), 100, "good")
    Path(first["output_path"]).write_text(reply_template.replace("{body}", body), encoding="utf-8")
    run_ingest(definition, run_folder)
    failures = read_all_records(run_folder, "rewrite_failures_*.jsonl")
    assert [(f["rewrite_id"], f["failure_kind"]) for f in failures] == [(first["rewrite_id"], expected_kind)]
    assert failures[0]["reply_text"] == reply_template.replace("{body}", body)  # nothing stripped, nothing lost
    assert first["rewrite_id"] not in {r["document_id"] for r in run_state.read_accepted_records(run_folder, definition)}
    assert not Path(first["output_path"]).exists() and list((run_folder / run_state.INGESTED_OUTPUTS_FOLDER_NAME).glob(f"{first['rewrite_id']}__*"))
    _, again = run_render(tmp_path, inputs, definition, 1)
    entry = next(e for e in again["rewrites"] if e["rewrite_id"] == first["rewrite_id"])
    assert entry["attempt_number"] == 1 and entry["last_failed_validation"].startswith(expected_kind)
    assert Path(entry["prompt_path"]).read_text(encoding="utf-8") == original_prompt


def test_a_document_that_itself_starts_like_a_preamble_is_not_failed_and_whitespace_around_a_good_reply_is_stripped_like_the_paid_script(tmp_path, inputs):
    source = "Sure, this is how the plan works.\n" + " ".join(f"w{i}" for i in range(100))
    assert run_state.find_preamble_or_wrapper(source + " more", source) is None
    text, failure_kind, _ = run_state.check_reply_text("\n  " + source + "  \n", source)
    assert failure_kind is None and text == source


# ---------------------------------------------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------------------------------------------

def test_resume_skips_accepted_rewrites_and_does_not_overwrite_a_prompt_whose_reply_is_unread(tmp_path, inputs):
    definition = D.CALIBRATED_CONSIDERATION
    _, manifest = run_render(tmp_path, inputs, definition, 1)
    run_folder = Path(manifest["run_folder"])
    accepted_half, waiting_half = manifest["rewrites"][:6], manifest["rewrites"][6:8]
    write_replies({"rewrites": accepted_half}, generous)
    run_ingest(definition, run_folder)
    write_replies({"rewrites": waiting_half}, generous)  # replies not ingested yet
    accepted_ids = {e["rewrite_id"] for e in accepted_half}
    prompt_modified_times = {e["rewrite_id"]: Path(e["prompt_path"]).stat().st_mtime_ns for e in manifest["rewrites"]}
    _, resumed = run_render(tmp_path, inputs, definition, 1)
    assert Path(resumed["run_folder"]) == run_folder  # the existing folder is found without --resume-folder
    pending_ids = {e["rewrite_id"] for e in resumed["rewrites"]}
    assert not pending_ids & accepted_ids and resumed["already_accepted_count"] == 6
    assert set(resumed["awaiting_ingestion_rewrite_ids"]) == {e["rewrite_id"] for e in waiting_half} and not pending_ids & set(resumed["awaiting_ingestion_rewrite_ids"])
    assert resumed["pending_count"] == len(manifest["rewrites"]) - 8
    assert all(Path(e["prompt_path"]).stat().st_mtime_ns == prompt_modified_times[e["rewrite_id"]] for e in waiting_half)
    assert len(list(run_folder.glob("run_configuration_*.json"))) >= 1


def test_ingest_never_accepts_a_rewrite_twice(tmp_path, inputs):
    definition = D.CARES_ABOUT_WELFARE
    _, manifest = run_render(tmp_path, inputs, definition, 1)
    run_folder = Path(manifest["run_folder"])
    first = manifest["rewrites"][0]
    write_replies({"rewrites": [first]}, generous)
    run_ingest(definition, run_folder)
    Path(first["output_path"]).write_text("a different late reply that must not replace the accepted one", encoding="utf-8")
    outcome = run_ingest(definition, run_folder)
    assert outcome["ignored_already_accepted"] == [first["rewrite_id"]] and Path(first["output_path"]).exists()
    assert sum(1 for r in read_all_records(run_folder, arms.rewrites_file_pattern(definition)) if r["document_id"] == first["rewrite_id"]) == 1


# ---------------------------------------------------------------------------------------------------------------
# Completion files and the scoring builder
# ---------------------------------------------------------------------------------------------------------------

def test_completion_writes_the_export_marker_and_a_configuration_naming_claude_code_subagents(tmp_path, inputs):
    definition = D.NON_INTRUSIVE
    _, manifest = run_render(tmp_path, inputs, definition, 1)
    run_folder = Path(manifest["run_folder"])
    write_replies(manifest, generous)
    run_ingest(definition, run_folder)
    assert not list(run_folder.glob("rewrites_complete_*.json"))
    _, wave_two = run_render(tmp_path, inputs, definition, 2)
    write_replies(wave_two, generous)
    run_ingest(definition, run_folder)
    assert len(list(run_folder.glob("rewrites_complete_*.json"))) == 1 and len(list(run_folder.glob("scoring_pod_extra_documents_*.jsonl"))) == 1
    configuration = json.loads(sorted(run_folder.glob("run_configuration_*.json"))[-1].read_text())
    assert configuration["rewrite_made_by"] == "Claude Code subagents running Opus 5.5" and configuration["model_id"] == "claude-opus-5-5"
    assert configuration["api_used"] is False and "no per-call usage or cost" in configuration["per_call_token_and_cost_log"]
    assert not list(run_folder.glob("per_call_usage_and_cost_*.jsonl")) and not list(run_folder.glob("usage_and_cost_log_*.json"))
    exported = read_jsonl(sorted(run_folder.glob("scoring_pod_extra_documents_*.jsonl"))[0])
    assert set(exported[0]) == set(paid_script.EXPORT_FIELDS + paid_script.EXPORT_FIELDS_OF_THE_NEW_FACETS_ADDED)


@FACETS
def test_the_scoring_builder_accepts_the_subagent_export(tmp_path, definition):
    from persona_vector_projection import rehearse_content_controlled_facet_positive_controls_and_scoring_locally as rehearsal
    try:
        stand_in_vector_set = rehearsal.find_stand_in_vector_set(definition)
    except FileNotFoundError:
        pytest.skip("no rehearsal vector set under small_sample_checks to stand in for the real vectors")
    corpus_path, preachiness_path, previous_folder = rehearsal.write_synthetic_inputs(tmp_path)
    sources = [f"matrix_{number:06d}" for number in range(rehearsal.FIRST_NUMBER_OF_THE_NEW_FACET_SOURCES,
                                                          rehearsal.FIRST_NUMBER_OF_THE_NEW_FACET_SOURCES + rehearsal.NUMBER_OF_SOURCES)]
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps(sources))
    synthetic_inputs = (None, corpus_path, sources_path)
    for wave in (1, 2):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest_path = render.main(render_arguments(tmp_path, synthetic_inputs, definition, wave,
                                                         ["--minimum-changed-units", "60", "--target-changed-units", "90"]))
        manifest = json.loads(Path(manifest_path).read_text())
        run_folder = Path(manifest["run_folder"])
        write_replies(manifest, lambda entry, prompt: rehearsal.change_words(prompt_document(prompt), 90, entry["arm"].replace("_", "")))
        run_ingest(definition, run_folder)
    export_path = sorted(run_folder.glob("scoring_pod_extra_documents_*.jsonl"))[0]
    builder_arguments = builder.parse_arguments([
        "--facet", definition.facet_key, "--vector-set-folder", str(stand_in_vector_set), "--new-rewrites-jsonl", str(export_path),
        "--previous-scoring-run-folder", str(previous_folder), "--preachiness-rewrites-jsonl", str(preachiness_path),
        "--corpus-path", str(corpus_path), "--expected-corpus-documents", str(rehearsal.NUMBER_OF_CORPUS_DOCUMENTS),
        "--expected-sources", str(rehearsal.NUMBER_OF_SOURCES), "--runs-directory", str(tmp_path / "scoring_runs"), "--run-name", "subagent_export_scoring_run"])
    scoring_run_folder, configuration, _estimate = builder.build_run_folder(builder_arguments)
    scored = {d["document_id"]: d for d in read_jsonl(scoring_run_folder / defs.INPUTS_FOLDER_NAME / "documents.jsonl")}
    exported = read_jsonl(export_path)
    assert len(exported) == rehearsal.NUMBER_OF_SOURCES * len(arms.arm_names_for(definition))
    assert all(record["document_id"] in scored for record in exported)
    chained = next(r for r in exported if r["arm"] == REMOVED)
    assert scored[chained["document_id"]]["source_document_id"] == arms.rewrite_document_id_for(definition, AWAY, chained["original_corpus_source_document_id"])
