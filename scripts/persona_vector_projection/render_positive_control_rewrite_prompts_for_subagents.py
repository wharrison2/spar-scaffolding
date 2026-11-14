"""Writes the prompt files for making the content-controlled positive-control rewrites with Claude Code subagents (Opus 5.5) instead
of the paid API (no API call, no .env read, no pod; local and free). The prompts are the paid script's own rendering
(make_content_controlled_facet_positive_control_rewrites.py and make_facet_positive_control_rewrites.py): the instruction template
of the arm, the target and lowest and highest changed amounts, the minimum, and for a retry the note
prompts/positive_control_rewrite_retry_note_below_minimum_changes__draft.txt with the previous changed amount.

Waves (the chained arm needs finished trait_only_away rewrites):
  wave 1  every arm except trait_removed_from_away_rewrite, each on the original corpus document
  wave 2  trait_removed_from_away_rewrite, on the ACCEPTED trait_only_away rewrite of each source (never on the source); needs
          --resume-folder (or the existing run folder) holding the accepted wave-1 rewrites; a source whose trait_only_away rewrite is
          not accepted gets no prompt and is recorded in rewrite_failures as skipped_no_source_rewrite.

For each rewrite that is not yet accepted and has no unread reply, this writes <run folder>/subagent_prompts/<rewrite id>.txt and lists it in
<run folder>/subagent_prompt_manifest_wave_<n>_<timestamp>.json with the output path the subagent must write
(<run folder>/subagent_outputs/<rewrite id>.txt). A rewrite whose reply was rejected below the minimum gets the retry-note prompt
(attempt 2 or 3, as the paid script would); a rewrite whose reply failed the format checks (empty, or a preamble) gets its prompt again.
Replies are read by ingest_subagent_positive_control_rewrites.py; run this script again afterwards for the retry prompts.

    python3 scripts/persona_vector_projection/render_positive_control_rewrite_prompts_for_subagents.py --facet cares_about_welfare --wave 1
    python3 scripts/persona_vector_projection/render_positive_control_rewrite_prompts_for_subagents.py --facet cares_about_welfare --wave 2
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for extra_path in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_arms as arms_module  # noqa: E402
from persona_vector_projection import make_facet_positive_control_rewrites as facet_rewrites  # noqa: E402
from persona_vector_projection import make_content_controlled_facet_positive_control_rewrites as paid_script  # noqa: E402
from persona_vector_projection import subagent_positive_control_rewrite_run_state as run_state  # noqa: E402
from shared.resumable_jsonl_records import AppendOnlyJsonlWriter, read_jsonl  # noqa: E402
from shared.run_folder_management import (check_resume_is_compatible, choose_run_folder, sha256_of_file,  # noqa: E402
                                          timestamp_now, write_run_configuration)

DEFAULT_SOURCES_FILE_DATE_TIME = "2026-11-01_15-24"


def default_sources_file(definition):
    return (facet_rewrites.defs.SMALL_SAMPLE_CHECKS_ROOT /
            f"positive_control_rewrite_sources__{definition.facet_name}__{DEFAULT_SOURCES_FILE_DATE_TIME}.json")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    facet_definitions.add_facet_argument(parser)
    parser.add_argument("--wave", type=int, required=True, choices=[1, 2])
    parser.add_argument("--resume-folder", default=None, help="Default: the latest existing folder for this facet, else a new one")
    parser.add_argument("--source-document-ids-file", default=None,
                        help="JSON list or one id per line (default: positive_control_rewrite_sources__<facet name>__2026-11-01_15-24.json "
                             "in small_sample_checks). Recorded by the first session; later sessions reuse it.")
    parser.add_argument("--target-changed-units", type=float, default=paid_script.DEFAULT_TARGET_CHANGED_UNITS_FOR_NEW_FACETS)
    parser.add_argument("--minimum-changed-units", type=int, default=facet_rewrites.DEFAULT_MINIMUM_CHANGED_UNITS)
    parser.add_argument("--maximum-regenerations", type=int, default=facet_rewrites.DEFAULT_MAXIMUM_REGENERATIONS)
    parser.add_argument("--size-match-tolerance", type=float, default=facet_rewrites.DEFAULT_SIZE_MATCH_TOLERANCE)
    parser.add_argument("--no-regeneration", action="store_true")
    parser.add_argument("--tokenizer-name", default=None, help="Count changed units in this tokenizer's tokens (local cache only)")
    parser.add_argument("--corpus-path", default=str(paid_script.DEFAULT_CORPUS_PATH))
    parser.add_argument("--runs-directory", default=str(paid_script.FULL_RUNS_DIRECTORY))
    arguments = parser.parse_args(argv)
    arguments.definition = facet_definitions.get_facet_definition(arguments.facet)
    if arguments.definition.legacy_moralizing_names:
        raise SystemExit("This script covers the three new facets only (cares_about_welfare, calibrated_consideration, non_intrusive).")
    return arguments


def settings_from_arguments(arguments):
    return {"target_changed_units": arguments.target_changed_units, "minimum_changed_units": arguments.minimum_changed_units,
            "maximum_regenerations": arguments.maximum_regenerations, "size_match_tolerance": arguments.size_match_tolerance,
            "no_regeneration": arguments.no_regeneration, "tokenizer_name": arguments.tokenizer_name}


def create_or_resume_run_folder(arguments, definition, templates):
    """(run folder, is_resume, configuration of the first session or None)."""
    prompt_hashes = {name: sha256_of_file(paid_script.PROMPT_DIRECTORY / name)
                     for name in paid_script.instruction_file_names_used(definition)}
    existing = Path(arguments.resume_folder) if arguments.resume_folder else run_state.find_existing_run_folder(arguments.runs_directory, definition)
    if existing is not None:
        run_folder, is_resume = choose_run_folder(arguments.runs_directory, run_state.run_name_for(definition), str(existing))
        check_resume_is_compatible(run_folder, {"prompt_file_sha256": prompt_hashes})
        return run_folder, True, prompt_hashes
    run_folder, is_resume = choose_run_folder(arguments.runs_directory, run_state.run_name_for(definition), None)
    return run_folder, False, prompt_hashes


def write_session_configuration(run_folder, timestamp, arguments, definition, prompt_hashes, source_ids, settings, is_resume):
    configuration = {
        "script": "render_positive_control_rewrite_prompts_for_subagents.py", "wave": arguments.wave, "facet_key": definition.facet_key,
        "rewrite_made_by": run_state.REWRITE_MADE_BY, "model_id": run_state.SUBAGENT_MODEL_ID,
        "arms": arms_module.arm_names_for(definition),
        "instruction_file_by_arm": arms_module.resolve_instruction_file_names(definition, paid_script.PROMPT_DIRECTORY),
        "expected_sign_by_arm": {arm: arms_module.expected_sign_for(definition, arm) for arm in arms_module.arm_names_for(definition)},
        "prompt_file_sha256": prompt_hashes, "corpus_path": str(arguments.corpus_path),
        "corpus_sha256": sha256_of_file(arguments.corpus_path), "source_document_ids_file": str(arguments.source_document_ids_file),
        "original_source_document_ids": source_ids, run_state.SETTINGS_KEY: settings, "resumed": is_resume}
    return write_run_configuration(run_folder, timestamp, configuration)


def pending_entries_for_tasks(run_folder, definition, templates, settings, tasks, tokenize_function):
    """([manifest entries], accepted count, awaiting-ingestion ids) for the tasks; writes nothing."""
    accepted_ids = {record["document_id"] for record in run_state.read_accepted_records(run_folder, definition)}
    last_failures = run_state.last_failure_by_rewrite_id(run_folder)
    entries, awaiting_ingestion, accepted_count = [], [], 0
    for task in tasks:
        rewrite_id = run_state.rewrite_id_of_task(definition, task)
        if rewrite_id in accepted_ids:
            accepted_count += 1
            continue
        if run_state.output_path_for(run_folder, rewrite_id).exists():
            awaiting_ingestion.append(rewrite_id)
            continue
        outcome = run_state.replay_rewrite(task, definition, templates, settings,
                                           run_state.read_reply_attempt_texts(run_folder, rewrite_id), tokenize_function)
        if outcome[0] == "accepted":
            raise RuntimeError(f"{rewrite_id} is finished by its recorded replies but was never accepted; run the ingest script.")
        _, prompt_text, attempt_number = outcome
        failure = last_failures.get(rewrite_id)
        entries.append({"rewrite_id": rewrite_id, "arm": task["arm"], "original_source_document_id": task["original_source_id"],
                        "attempt_number": attempt_number, "is_retry_with_below_minimum_note": attempt_number > 1,
                        "last_failed_validation": (failure["failure_kind"] + ": " + failure.get("description", "")) if failure
                        and failure.get("attempt_number") == attempt_number else None,
                        "prompt_text": prompt_text})
    return entries, accepted_count, awaiting_ingestion


def main(argv=None):
    arguments = parse_arguments(argv)
    definition = arguments.definition
    timestamp = timestamp_now()
    templates = paid_script.load_instruction_templates(definition=definition)
    run_folder, is_resume, prompt_hashes = create_or_resume_run_folder(arguments, definition, templates)
    if is_resume:
        settings, first_configuration = run_state.settings_from_run_configuration(run_folder)
        arguments.source_document_ids_file = first_configuration["source_document_ids_file"]
        arguments.corpus_path = first_configuration["corpus_path"]
        source_ids = first_configuration["original_source_document_ids"]
    else:
        arguments.source_document_ids_file = arguments.source_document_ids_file or str(default_sources_file(definition))
        source_ids = sorted(paid_script.read_source_document_ids(arguments.source_document_ids_file))
        settings = settings_from_arguments(arguments)
    corpus_by_id = {document["doc_id"]: document for document in read_jsonl(arguments.corpus_path)}
    unknown = [source for source in source_ids if source not in corpus_by_id]
    if unknown:
        raise SystemExit(f"Source documents not in the corpus: {unknown}")
    write_session_configuration(run_folder, timestamp, arguments, definition, prompt_hashes, source_ids, settings, is_resume)
    tokenize_function = facet_rewrites.make_tokenize_function(settings["tokenizer_name"])
    template_by_key = templates

    skipped = []
    if arguments.wave == 1:
        tasks = run_state.build_independent_tasks(definition, corpus_by_id, source_ids, settings["target_changed_units"])
    else:
        tasks, skipped = run_state.build_chained_tasks(definition, source_ids, run_state.read_accepted_records(run_folder, definition),
                                                       settings["target_changed_units"])
        failures_path = Path(run_folder) / f"rewrite_failures_{timestamp}.jsonl"
        for arm_name, source_id in skipped:
            AppendOnlyJsonlWriter().append(failures_path, {
                "arm": arm_name, "original_source_id": source_id, "failure_kind": paid_script.SKIPPED_NO_SOURCE_REWRITE,
                "rewrite_id": arms_module.rewrite_document_id_for(definition, arm_name, source_id)})
        if skipped:
            print(f"{len(skipped)} chained rewrites have no accepted trait_only_away rewrite yet and get no prompt: {skipped}", flush=True)

    entries, accepted_count, awaiting_ingestion = pending_entries_for_tasks(
        run_folder, definition, template_by_key, settings, tasks, tokenize_function)
    manifest_entries = []
    for entry in entries:
        prompt_path = run_state.prompt_path_for(run_folder, entry["rewrite_id"])
        output_path = run_state.output_path_for(run_folder, entry["rewrite_id"])
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(entry.pop("prompt_text"), encoding="utf-8")
        manifest_entries.append({**entry, "prompt_path": str(prompt_path.resolve()), "output_path": str(output_path.resolve())})
    manifest = {"facet_key": definition.facet_key, "wave": arguments.wave, "run_folder": str(Path(run_folder).resolve()),
                "written_at": timestamp, "pending_count": len(manifest_entries), "already_accepted_count": accepted_count,
                "awaiting_ingestion_count": len(awaiting_ingestion), "awaiting_ingestion_rewrite_ids": awaiting_ingestion,
                "chained_rewrites_without_accepted_source": [list(pair) for pair in skipped], "rewrites": manifest_entries}
    manifest_path = Path(run_folder) / f"subagent_prompt_manifest_wave_{arguments.wave}_{timestamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Run folder: {Path(run_folder).resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Pending rewrites (prompt files written): {len(manifest_entries)}; already accepted: {accepted_count}; "
          f"replies awaiting ingestion: {len(awaiting_ingestion)}")
    return manifest_path


if __name__ == "__main__":
    main()
