"""Reads the replies Claude Code subagents (Opus 5.5) wrote for the content-controlled positive-control rewrites and records them in the
paid script's output format (no API call, no .env read, no pod; local and free). Pairs with
render_positive_control_rewrite_prompts_for_subagents.py.

Validation is the paid script's own (facet_rewrites.make_one_rewrite, replayed over the replies; see
subagent_positive_control_rewrite_run_state.py): an empty reply fails (empty_output); a reply is read with surrounding whitespace stripped
and nothing else; its changed amount is counted against the document the instruction was applied to; below the minimum it is not accepted
but retried with the retry note (up to --maximum-regenerations, recorded in the run), and after the last allowed attempt the largest-change
reply is kept and flagged `below_minimum_changed_units` exactly as the paid script keeps it. One check is added to the paid script's
because a subagent, unlike the API's text blocks, can wrap its reply: a reply with a preamble, a closing note or a code fence FAILS
(preamble_or_wrapper_in_reply); nothing is stripped. The paid script has no language check, so there is none here.

Each reply file is read once and then moved, unchanged, to subagent_outputs_ingested/. Failures are appended to rewrite_failures_<timestamp>.jsonl
(with the reason and the reply text); the prompt script then writes the retry prompts. When every rewrite of the facet (all arms, all sources)
is accepted this also writes the export scoring_pod_extra_documents_<timestamp>.jsonl that the scoring builder reads, a run configuration saying
the rewrites were made by Claude Code subagents running Opus 5.5 (model id claude-opus-5-5, no per-call token or cost log), and
rewrites_complete_<timestamp>.json.

    python3 scripts/persona_vector_projection/ingest_subagent_positive_control_rewrites.py --facet cares_about_welfare --run-folder <run folder>
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
from persona_vector_projection import make_content_controlled_facet_positive_control_rewrites as paid_script  # noqa: E402
from persona_vector_projection import make_facet_positive_control_rewrites as facet_rewrites  # noqa: E402
from persona_vector_projection import subagent_positive_control_rewrite_run_state as run_state  # noqa: E402
from shared.resumable_jsonl_records import AppendOnlyJsonlWriter  # noqa: E402
from shared.run_folder_management import sha256_of_file, timestamp_now, write_run_configuration  # noqa: E402


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    facet_definitions.add_facet_argument(parser)
    parser.add_argument("--run-folder", required=True)
    arguments = parser.parse_args(argv)
    arguments.definition = facet_definitions.get_facet_definition(arguments.facet)
    return arguments


def archive_reply_file(run_folder, rewrite_id, attempt_number, timestamp):
    source = run_state.output_path_for(run_folder, rewrite_id)
    destination_folder = Path(run_folder) / run_state.INGESTED_OUTPUTS_FOLDER_NAME
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination = destination_folder / f"{rewrite_id}__attempt_{attempt_number}__{timestamp}.txt"
    source.replace(destination)
    return destination


def ingest_tasks(run_folder, definition, templates, settings, tasks, timestamp, tokenize_function, writer):
    """Reads the reply file of every task that has one. Returns a dict of counts and the lists of ids."""
    rewrites_path = Path(run_folder) / f"{definition.facet_key}_positive_control_rewrites_{timestamp}.jsonl"
    attempts_path = Path(run_folder) / f"subagent_reply_attempts_{timestamp}.jsonl"
    failures_path = Path(run_folder) / f"rewrite_failures_{timestamp}.jsonl"
    outcome = {"accepted": [], "flagged_below_minimum": [], "retry_needed": [], "failed_validation": [], "ignored_already_accepted": []}
    accepted_ids = {record["document_id"] for record in run_state.read_accepted_records(run_folder, definition)}
    for task in tasks:
        rewrite_id = run_state.rewrite_id_of_task(definition, task)
        output_path = run_state.output_path_for(run_folder, rewrite_id)
        if not output_path.exists():
            continue
        if rewrite_id in accepted_ids:
            outcome["ignored_already_accepted"].append(rewrite_id)  # left in place for the person to look at
            continue
        previous_texts = run_state.read_reply_attempt_texts(run_folder, rewrite_id)
        attempt_number = len(previous_texts) + 1
        raw_reply = output_path.read_text(encoding="utf-8")
        text, failure_kind, description = run_state.check_reply_text(raw_reply, task["document"]["content"])
        if failure_kind:
            writer.append(failures_path, {"rewrite_id": rewrite_id, "arm": task["arm"], "original_source_id": task["original_source_id"],
                                          "failure_kind": failure_kind, "description": description, "attempt_number": attempt_number,
                                          "reply_text": raw_reply, "model_used": run_state.SUBAGENT_MODEL_ID})
            archive_reply_file(run_folder, rewrite_id, attempt_number, timestamp)
            outcome["failed_validation"].append((rewrite_id, failure_kind))
            continue
        writer.append(attempts_path, {"rewrite_id": rewrite_id, "arm": task["arm"], "original_source_id": task["original_source_id"],
                                      "attempt_number": attempt_number, "reply_text": text,
                                      "reply_file_sha256": run_state.sha256_of_text_value(raw_reply), "model_used": run_state.SUBAGENT_MODEL_ID})
        archive_reply_file(run_folder, rewrite_id, attempt_number, timestamp)
        replayed = run_state.replay_rewrite(task, definition, templates, settings, previous_texts + [text], tokenize_function)
        if replayed[0] == "accepted":
            writer.append(rewrites_path, replayed[1])
            outcome["accepted"].append(rewrite_id)
            if replayed[1]["below_minimum_changed_units"]:
                outcome["flagged_below_minimum"].append(rewrite_id)
        else:
            outcome["retry_needed"].append((rewrite_id, replayed[2]))
    return outcome


def completion_check(run_folder, definition, source_ids):
    accepted = {record["document_id"] for record in run_state.read_accepted_records(run_folder, definition)}
    missing = [rewrite_id for rewrite_id in run_state.expected_rewrite_ids(definition, source_ids) if rewrite_id not in accepted]
    return missing


def write_completion_files(run_folder, definition, timestamp, first_configuration, source_ids):
    export_path = paid_script.write_export(run_folder, timestamp, definition)
    configuration = {
        **{key: value for key, value in first_configuration.items() if key != "script"},
        "script": "ingest_subagent_positive_control_rewrites.py", "completed_at": timestamp,
        "rewrite_made_by": run_state.REWRITE_MADE_BY, "model_id": run_state.SUBAGENT_MODEL_ID,
        "model_requested": run_state.SUBAGENT_MODEL_ID, "api_used": False,
        "per_call_token_and_cost_log": "none: the rewrites were made by Claude Code subagents, so no per-call usage or cost was recorded",
        "export_file": export_path.name, "export_sha256": sha256_of_file(export_path)}
    write_run_configuration(run_folder, timestamp, configuration)
    marker_path = Path(run_folder) / f"rewrites_complete_{timestamp}.json"
    marker_path.write_text(json.dumps({
        "facet_key": definition.facet_key, "complete": True, "rewrite_made_by": run_state.REWRITE_MADE_BY,
        "model_id": run_state.SUBAGENT_MODEL_ID, "rewrite_count": len(run_state.expected_rewrite_ids(definition, source_ids)),
        "export_file": export_path.name, "completed_at": timestamp}, indent=2), encoding="utf-8")
    return export_path, marker_path


def main(argv=None):
    arguments = parse_arguments(argv)
    definition, run_folder = arguments.definition, Path(arguments.run_folder)
    timestamp = timestamp_now()
    settings, first_configuration, corpus_by_id, source_ids = run_state.load_run_inputs(run_folder, definition)
    if first_configuration["facet_key"] != definition.facet_key:
        raise SystemExit(f"{run_folder} is a run for {first_configuration['facet_key']}, not {definition.facet_key}")
    templates = paid_script.load_instruction_templates(definition=definition)
    tokenize_function = facet_rewrites.make_tokenize_function(settings["tokenizer_name"])
    writer = AppendOnlyJsonlWriter()
    independent_tasks = run_state.build_independent_tasks(definition, corpus_by_id, source_ids, settings["target_changed_units"])
    outcome = ingest_tasks(run_folder, definition, templates, settings, independent_tasks, timestamp, tokenize_function, writer)
    chained_tasks, _skipped = run_state.build_chained_tasks(
        definition, source_ids, run_state.read_accepted_records(run_folder, definition), settings["target_changed_units"])
    chained_outcome = ingest_tasks(run_folder, definition, templates, settings, chained_tasks, timestamp, tokenize_function, writer)
    for key in outcome:
        outcome[key] += chained_outcome[key]
    print(f"Run folder: {run_folder.resolve()}")
    print(f"Accepted now: {len(outcome['accepted'])} (flagged below the minimum after the last attempt: {len(outcome['flagged_below_minimum'])})")
    print(f"Retry needed (below the minimum, retry-note prompt to be written): {outcome['retry_needed']}")
    print(f"Failed validation (prompt to be written again): {outcome['failed_validation']}")
    if outcome["ignored_already_accepted"]:
        print(f"Reply files left in place for rewrites that were already accepted: {outcome['ignored_already_accepted']}")
    missing = completion_check(run_folder, definition, source_ids)
    if missing:
        print(f"NOT COMPLETE: {len(missing)} of {len(run_state.expected_rewrite_ids(definition, source_ids))} rewrites not yet accepted; "
              f"run the prompt script again for the next wave or the retry prompts.")
    else:
        export_path, marker_path = write_completion_files(run_folder, definition, timestamp, first_configuration, source_ids)
        print(f"COMPLETE. Export for the scoring builder: {export_path.resolve()}\nMarker: {marker_path.resolve()}")
    outcome["missing"] = missing
    return outcome


if __name__ == "__main__":
    main()
