#!/usr/bin/env python3
"""Real-run request replay for the SDF pipeline: proves a refactor changes no request sent to any model.

Makes no model API call and no network call, reads no key. Two subcommands.

``rebuild`` takes a pipeline checkout, rebuilds every request of every real SDF run from the run's stored
inputs by driving the checkout's own stage code (see sdf_request_capture_from_pipeline_checkout.py), saves
the rebuilt requests, and compares them with what each run's model_calls.jsonl and system_prompts/ say was
sent (comparison 1, only the fields those records hold: the stage tag, item id, route, model, system prompt
hash and user message). These runs were made by older code, so differences on origin/main are expected;
they are the baseline.

``compare`` takes the output folders of two ``rebuild`` invocations (for example origin/main and a refactor
branch) and compares the rebuilt requests request by request on every field (comparison 2). The refactor
passes when there are no differences.

Both exit 1 when their comparison finds a difference (``rebuild`` can be told to report without failing).

Example:
  PYTHON=/Users/williamharrison/Desktop/emergent_alignment/animal-welfare-data-pipeline/.venv/bin/python
  $PYTHON replay_sdf_real_run_requests_against_pipeline_checkout.py rebuild --pipeline-checkout <checkout>
  $PYTHON replay_sdf_real_run_requests_against_pipeline_checkout.py compare --first-rebuilt-folder <a> --second-rebuilt-folder <b>
"""

import argparse
import copy
import csv
import glob
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

SCRIPT_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_FOLDER))

import sdf_request_comparison as comparison  # noqa: E402  (the pipeline is imported later, from the checkout given)
import sdf_request_capture_from_pipeline_checkout as capture  # noqa: E402
import sdf_stage_replay_drivers as drivers  # noqa: E402

MAIN_CHECKOUT_OF_THIS_PROJECT = Path("/Users/williamharrison/Desktop/emergent_alignment")
DEFAULT_OUTPUT_ROOT = MAIN_CHECKOUT_OF_THIS_PROJECT / "data/sdf_document_state_redesign_runs/small_sample_checks"
DEFAULT_REAL_RUN_FOLDER_GLOBS = (
    str(MAIN_CHECKOUT_OF_THIS_PROJECT
        / "data/spec_contradiction_screen_runs/small_sample_checks/*/pipeline_outputs/sdf/runs/*"),
    str(MAIN_CHECKOUT_OF_THIS_PROJECT
        / "animal-welfare-data-pipeline-show-rejected-variable-values/outputs/sdf/experiments"
          "/show_rejected_values_paired_test_*/shared_deal_and_screen/sdf/runs/*"),
    str(MAIN_CHECKOUT_OF_THIS_PROJECT
        / "animal-welfare-data-pipeline-show-rejected-variable-values/outputs/sdf/experiments"
          "/show_rejected_values_paired_test_*/arms/*/sdf/runs/*"),
)

FIELDS_NOT_IN_MODEL_CALLS_RECORDS = (
    "max_tokens, temperature, reasoning effort, thinking setting, JSON schema, schema name, prompt-cache flag "
    "and any other argument: model_calls.jsonl stores only the stage tag, item id, model, backend, system prompt "
    "hash (text in system_prompts/), user message(s), reply, stop reason, usage and attempt count, so these are "
    "rebuilt and saved (for comparison 2) but cannot be checked against what was sent")


MAXIMUM_RESUME_PASSES = 6


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ---------------------------------------------------------------- reading real runs

def fingerprint_of_folder(folder: Path) -> str:
    """Hash of every file's relative path, size and modification time: proves a folder was not touched."""
    digest = hashlib.sha256()
    for directory, subdirectories, file_names in os.walk(folder):
        subdirectories.sort()
        for file_name in sorted(file_names):
            path = Path(directory) / file_name
            status = path.lstat()
            digest.update(f"{path.relative_to(folder)}|{status.st_size}|{status.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def read_jsonl_lines(path: Path) -> list:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class RealRun:
    def __init__(self, folder: Path):
        self.folder = folder
        self.label = folder.name
        self.manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        self.model_calls_path = folder / "model_calls.jsonl"
        self.problem_with_run = self._problem()
        self.sent_records: list = []
        self.records_not_successful = 0
        self.system_prompt_texts: dict = {}
        if self.problem_with_run is None:
            every_record = read_jsonl_lines(self.model_calls_path)
            self.sent_records = [r for r in every_record if r.get("error") is None and r.get("response_text") is not None]
            self.records_not_successful = len(every_record) - len(self.sent_records)
            for sha in {r["system_prompt_sha256"] for r in self.sent_records}:
                self.system_prompt_texts[sha] = (folder / "system_prompts" / f"{sha}.txt").read_text(encoding="utf-8")

    def _problem(self):
        if (self.folder / "layer12").is_dir():
            return "pre-renumber layout (layer12), not a current-layout run"
        if not self.model_calls_path.is_file():
            return "no model_calls.jsonl: the run predates the request records, so there is nothing to compare with"
        if not (self.folder / "inputs" / "prompts").is_dir():
            return "no frozen prompt snapshot (inputs/prompts)"
        return None

    def sent_tags(self) -> set:
        return {r["stage"] for r in self.sent_records}

    def sent_requests(self) -> list:
        """The sent records as comparison requests, with only the fields the records hold."""
        occurrences: dict = {}
        requests = []
        route_names = {"api": "anthropic"}
        for record in self.sent_records:
            key = (record["stage"], record["item_id"])
            occurrences[key] = occurrences.get(key, 0) + 1
            requests.append({
                "run_label": self.label, "stage_tag": record["stage"], "item_id": record["item_id"],
                "occurrence": occurrences[key],
                "route": route_names.get(record["backend"], record["backend"]),
                "model": record["model"], "system_sha256": record["system_prompt_sha256"],
                "user_message": record["user_messages"][0] if len(record["user_messages"]) == 1
                else json.dumps(record["user_messages"]),
            })
        return requests


def discover_real_run_folders(globs: tuple) -> list:
    folders = []
    for pattern in globs:
        for match in sorted(glob.glob(pattern)):
            path = Path(match)
            if (path / "run_manifest.json").is_file():
                folders.append(path)
    return folders


# ---------------------------------------------------------------- one stage group of one run

def configuration_for_replay(run: RealRun, prompts_folder: Path, pipeline) -> tuple:
    """The run's saved configuration with the overrides a replay needs, each one listed."""
    configuration = copy.deepcopy(run.manifest["config"])
    overrides = []
    if configuration.get("workers") != 1:
        configuration["workers"] = 1
        overrides.append("workers set to 1 so that calls happen in a fixed order (no effect on any request)")
    if configuration.get("backend", "api") != "api":
        overrides.append(f"backend {configuration.get('backend')!r} replaced by 'api' (requests are captured at the API send function)")
        configuration["backend"] = "api"
    revision = (configuration.get("sdf") or {}).get("revision") or {}
    layer3 = pipeline.layer3_rewrite
    two_call_templates = (getattr(layer3, "CRITIQUE_TEMPLATE", "layer3a_critique.txt"),
                          getattr(layer3, "REVISE_TEMPLATE", "layer3b_revise.txt"))
    if "split" not in revision and not all((prompts_folder / name).is_file() for name in two_call_templates):
        configuration.setdefault("sdf", {}).setdefault("revision", {})["split"] = False
        overrides.append("sdf.revision.split set to false: the run's prompt snapshot has no two-call layer 3 templates "
                         "(it ran layer 3 as one call), and the checkout's default is two calls")
    return configuration, overrides


def write_json(path: Path, content) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def replay_stage_group(run: RealRun, stage_group: str, hooks, pipeline, run_output_folder: Path,
                       scratch_run_folder: Path) -> dict:
    """Replay one stage group of one run and save its requests; the status file is written last, so a
    group with a status file is finished and is skipped when the folder is resumed."""
    already_done = sorted(run_output_folder.glob(f"stage_status_{stage_group}_*.json"))
    if already_done:
        return json.loads(already_done[-1].read_text(encoding="utf-8"))

    stamp = timestamp_now()
    status = {"run_label": run.label, "stage_group": stage_group, "status": None, "reason": "",
              "configuration_overrides": [], "requests_captured": 0}
    stage_tags = drivers.STAGE_GROUP_STAGE_TAGS[stage_group]
    missing_inputs = drivers.missing_stored_inputs(run.folder, stage_group)
    if not (run.sent_tags() & set(stage_tags)):
        status.update(status="not_rebuilt", reason=(
            f"the run's records hold no call with stage tag {'/'.join(stage_tags)} (the stage ran in another run "
            "whose stage files this run reused, or did not run)"))
    elif missing_inputs:
        status.update(status="not_rebuilt", reason=f"stored inputs missing from the run folder: {missing_inputs}")
    else:
        scratch_group_folder = scratch_run_folder / stage_group
        if scratch_group_folder.exists():
            shutil.rmtree(scratch_group_folder)
        scratch_group_folder.mkdir(parents=True)
        prompts_folder = scratch_run_folder / "inputs" / "prompts"
        if not prompts_folder.is_dir():
            shutil.copytree(run.folder / "inputs", scratch_run_folder / "inputs")
        configuration, overrides = configuration_for_replay(run, prompts_folder, pipeline)
        status["configuration_overrides"] = overrides
        configuration_path = scratch_group_folder / "configuration_used_for_replay.yaml"
        import yaml  # available in the pipeline's environment
        configuration_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")

        api = pipeline.api
        original_loader = api.load_api_key
        api.load_api_key = lambda *arguments, **keyword_arguments: "placeholder-key-that-is-never-sent"
        try:
            api.init(str(configuration_path), cost_log_path=scratch_group_folder / "cost_log.jsonl")
        finally:
            api.load_api_key = original_loader

        reply_provider = capture.StoredReplyProvider(run.sent_records)
        hooks.begin(run.label, reply_provider)
        stage = drivers.ScratchStage(pipeline=pipeline, configuration=configuration, real_run_folder=run.folder,
                                     scratch_folder=scratch_group_folder, prompts_folder=prompts_folder)
        console_path = run_output_folder / f"stage_console_output_{stage_group}_{stamp}.txt"
        passes = 0
        try:
            with capture.console_output_to_file(console_path):
                # The real runs were resumed after stopping on failed calls, so a stage that stops itself after
                # sending requests is run again (a resume) while stored replies are left unused, up to a limit.
                while True:
                    passes += 1
                    requests_before_pass = len(hooks.captured)
                    try:
                        drivers.STAGE_GROUP_DRIVERS[stage_group](stage)
                        break
                    except SystemExit as stop:
                        made_progress = len(hooks.captured) > requests_before_pass
                        if made_progress and reply_provider.count_unused_records(stage_tags) and passes < MAXIMUM_RESUME_PASSES:
                            print(f"--- stage stopped itself ({stop}); resuming (pass {passes + 1})")
                            continue
                        status.update(status="completed_with_stage_stop", reason=f"the stage stopped itself: {stop}")
                        break
                    except Exception:
                        traceback.print_exc()
                        raise
        except Exception as error:
            status.update(status="not_rebuilt", reason=f"the checkout's stage code raised {type(error).__name__}: {error}")
        status["passes"] = passes
        captured, system_prompt_texts = hooks.finish()
        failure_lines = [line.strip() for line in console_path.read_text(encoding="utf-8").splitlines()
                         if "will retry on resume" in line or "will be rescored on resume" in line]
        status["lines_reporting_a_failed_item"] = len(failure_lines)
        status["first_line_reporting_a_failed_item"] = failure_lines[0] if failure_lines else ""
        if status["status"] == "completed_with_stage_stop" and not captured:
            status.update(status="not_rebuilt", reason=(
                f"{status['reason']} No request was sent: every item failed before reaching the send function "
                f"({status['first_line_reporting_a_failed_item']})"))
        if status["status"] is None:
            status["status"] = "rebuilt"
        if status["status"] != "not_rebuilt":
            captured.sort(key=lambda r: (str(r["stage_tag"]), str(r["item_id"]), r["occurrence"]))
            with open(run_output_folder / f"rebuilt_requests_{stage_group}_{stamp}.jsonl", "w", encoding="utf-8") as handle:
                for request in captured:
                    handle.write(capture.json_line(request))
            write_json(run_output_folder / f"system_prompts_{stage_group}_{stamp}.json", system_prompt_texts)
            status["requests_captured"] = len(captured)
            status["requests_without_stored_reply"] = sum(not r["stored_reply_found"] for r in captured)
        shutil.rmtree(scratch_group_folder, ignore_errors=True)
    write_json(run_output_folder / f"stage_status_{stage_group}_{stamp}.json", status)
    return status


# ---------------------------------------------------------------- loading a rebuild folder

def load_rebuilt_folder(folder: Path) -> dict:
    requests, system_texts, statuses = [], {}, {}
    for run_folder in sorted(folder.glob("rebuilt_requests_for_run_*")):
        for status_path in sorted(run_folder.glob("stage_status_*.json")):
            status = json.loads(status_path.read_text(encoding="utf-8"))
            statuses[(status["run_label"], status["stage_group"])] = status
        for group in drivers.STAGE_GROUP_ORDER:
            request_files = sorted(run_folder.glob(f"rebuilt_requests_{group}_*.jsonl"))
            if request_files:
                requests.extend(read_jsonl_lines(request_files[-1]))
            text_files = sorted(run_folder.glob(f"system_prompts_{group}_*.json"))
            if text_files:
                system_texts.update(json.loads(text_files[-1].read_text(encoding="utf-8")))
    identity_files = sorted(folder.glob("checkout_identity_*.json"))
    identity = json.loads(identity_files[-1].read_text(encoding="utf-8")) if identity_files else {}
    return {"requests": requests, "system_texts": system_texts, "statuses": statuses, "identity": identity}


# ---------------------------------------------------------------- writing results

def write_summary_files(folder: Path, name: str, stamp: str, rows: list, differences: list, header_text: str,
                        footer_text: str, maximum_listed: int) -> None:
    with open(folder / f"{name}_summary_table_{stamp}.csv", "w", newline="", encoding="utf-8") as handle:
        columns = ["run_label", "stage_tag", "compared", "equal", "different", "missing_in_second", "extra_in_second"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(folder / f"{name}_differences_{stamp}.jsonl", "w", encoding="utf-8") as handle:
        for difference in differences:
            handle.write(capture.json_line(difference))
    report = "\n".join([
        header_text, "", comparison.format_summary_table(rows), "",
        f"{len(differences)} differences" + (":" if differences else "."),
        comparison.format_differences_report(differences, maximum_listed), "", footer_text])
    (folder / f"{name}_report_{stamp}.txt").write_text(report, encoding="utf-8")
    print(report)


# ---------------------------------------------------------------- rebuild

def command_rebuild(arguments) -> int:
    checkout = Path(arguments.pipeline_checkout).resolve()
    identity = capture.git_identity_of_checkout(checkout)
    run_folders = [Path(p).resolve() for p in arguments.run_folder] if arguments.run_folder else \
        discover_real_run_folders(tuple(arguments.run_folder_glob or DEFAULT_REAL_RUN_FOLDER_GLOBS))
    if not run_folders:
        raise SystemExit("no real run folders found")

    stamp = timestamp_now()
    if arguments.resume_folder:
        output_folder = Path(arguments.resume_folder).resolve()
        earlier_identity = json.loads(sorted(output_folder.glob("checkout_identity_*.json"))[-1].read_text())
        if earlier_identity["git_commit"] != identity["git_commit"] or \
                earlier_identity["uncommitted_changes_to_tracked_files"] != identity["uncommitted_changes_to_tracked_files"]:
            raise SystemExit("the folder being resumed was made from a different checkout state")
    else:
        label = arguments.label or checkout.name
        output_folder = Path(arguments.output_root) / f"sdf_real_run_request_replay_rebuilt_from_{label}_{stamp}"
        output_folder.mkdir(parents=True)
        write_json(output_folder / f"checkout_identity_{stamp}.json", identity)

    capture.block_network_access()
    pipeline = capture.import_pipeline_checkout(checkout)
    print(f"pipeline checkout: {identity['checkout_path']}")
    print(f"git commit: {identity['git_commit']} (branch {identity['git_branch']}; "
          f"uncommitted changes to tracked files: {identity['uncommitted_changes_to_tracked_files'] or 'none'})")
    print(f"shared.api resolved to: {pipeline.resolved_files['shared.api']}")
    print(f"sdf_pipeline.layer1_screen resolved to: {pipeline.resolved_files['sdf_pipeline.layer1_screen']}")
    print(f"output folder: {output_folder}\n")

    hooks = capture.ReplayHooks(pipeline)
    runs = [RealRun(folder) for folder in run_folders]
    fingerprints_before = {run.label: fingerprint_of_folder(run.folder) for run in runs}
    scratch_root = output_folder / "scratch_working_copies"
    for run in runs:
        if run.problem_with_run:
            print(f"run {run.label}: not replayed ({run.problem_with_run})")
            continue
        run_output_folder = output_folder / f"rebuilt_requests_for_run_{run.label}"
        run_output_folder.mkdir(exist_ok=True)
        print(f"run {run.label}: {len(run.sent_records)} successful sent records "
              f"({run.records_not_successful} failed attempts left out)")
        for stage_group in drivers.STAGE_GROUP_ORDER:
            status = replay_stage_group(run, stage_group, hooks, pipeline, run_output_folder, scratch_root / run.label)
            print(f"  {stage_group}: {status['status']}, {status.get('requests_captured', 0)} requests"
                  + (f" ({status['reason']})" if status["reason"] else ""))
        shutil.rmtree(scratch_root / run.label, ignore_errors=True)
    shutil.rmtree(scratch_root, ignore_errors=True)

    fingerprints_after = {run.label: fingerprint_of_folder(run.folder) for run in runs}
    changed = [label for label in fingerprints_before if fingerprints_before[label] != fingerprints_after[label]]
    if changed:
        raise SystemExit(f"REAL RUN FOLDERS CHANGED DURING THE REPLAY: {changed}")
    print("\nreal run folders unchanged (file paths, sizes and modification times hashed before and after)")

    return compare_rebuilt_with_sent(arguments, output_folder, runs, identity)


def compare_rebuilt_with_sent(arguments, output_folder: Path, runs: list, identity: dict) -> int:
    rebuilt = load_rebuilt_folder(output_folder)
    sent_requests, sent_texts, rebuilt_requests = [], {}, []
    not_rebuilt_notes = []
    for run in runs:
        if run.problem_with_run:
            not_rebuilt_notes.append(f"{run.label}: whole run left out: {run.problem_with_run}")
            continue
        covered_tags = set()
        for stage_group in drivers.STAGE_GROUP_ORDER:
            status = rebuilt["statuses"].get((run.label, stage_group))
            tags = set(drivers.STAGE_GROUP_STAGE_TAGS[stage_group])
            if status is None or status["status"] == "not_rebuilt":
                reason = status["reason"] if status else "not replayed"
                if run.sent_tags() & tags or (status and "stage code raised" in reason):
                    not_rebuilt_notes.append(f"{run.label} / {stage_group}: left out: {reason}")
            else:
                covered_tags |= tags
                if status["status"] == "completed_with_stage_stop":
                    not_rebuilt_notes.append(f"{run.label} / {stage_group}: compared, but {status['reason']}")
                for override in status["configuration_overrides"]:
                    note = f"{run.label}: configuration override: {override}"
                    if note not in not_rebuilt_notes:
                        not_rebuilt_notes.append(note)
                if status.get("passes", 1) > 1:
                    not_rebuilt_notes.append(f"{run.label} / {stage_group}: the stage was run {status['passes']} times "
                                             "(resumed after stopping on failed items, as the real run was)")
        sent_requests.extend(r for r in run.sent_requests() if r["stage_tag"] in covered_tags)
        sent_texts.update(run.system_prompt_texts)
        unknown_tags = run.sent_tags() - {t for tags in drivers.STAGE_GROUP_STAGE_TAGS.values() for t in tags}
        if unknown_tags:
            not_rebuilt_notes.append(f"{run.label}: records with stage tags no replay driver covers: {sorted(unknown_tags)}")
    # A stage tag the checkout sends but the run never did (for example the two-call layer 3 over a one-call run)
    # is kept, and reported as "extra".
    run_labels = {run.label for run in runs}
    rebuilt_requests = [r for r in rebuilt["requests"] if r["run_label"] in run_labels]
    result = comparison.compare_request_sets(
        sent_requests, rebuilt_requests, fields=comparison.FIELDS_HELD_BY_SENT_RECORDS,
        first_system_prompt_texts=sent_texts, second_system_prompt_texts=rebuilt["system_texts"])
    rows = comparison.summary_rows(result["counts"])
    stamp = timestamp_now()
    never_replied = sum(not r["stored_reply_found"] for r in rebuilt_requests)
    header = "\n".join([
        "Comparison 1: requests rebuilt by the checkout (second) against requests each run sent (first)",
        f"checkout {identity['checkout_path']} at commit {identity['git_commit']}",
        "columns: compared = requests present in both; equal / different over the fields the records hold "
        "(route, model, system prompt hash, user message); missing_in_second = sent but not rebuilt; "
        "extra_in_second = rebuilt but not sent",
        f"requests rebuilt with no stored reply to feed back: {never_replied}",
        "Fields that cannot be checked: " + FIELDS_NOT_IN_MODEL_CALLS_RECORDS,
        "Left out or noted:" + ("".join("\n  - " + note for note in not_rebuilt_notes) if not_rebuilt_notes else " nothing"),
    ])
    write_summary_files(output_folder, "comparison_of_rebuilt_requests_with_sent_requests", stamp, rows,
                        result["differences"], header, f"output folder: {output_folder}",
                        arguments.maximum_differences_listed)
    write_json(output_folder / f"stages_left_out_or_noted_{stamp}.json", not_rebuilt_notes)
    if result["differences"] and not arguments.report_differences_without_failing:
        print("\nEXIT 1: the rebuilt requests differ from the sent ones (expected on origin/main for these older runs; "
              "use comparison 2 as the gate)")
        return 1
    return 0


# ---------------------------------------------------------------- compare two rebuild folders

def command_compare(arguments) -> int:
    first = load_rebuilt_folder(Path(arguments.first_rebuilt_folder))
    second = load_rebuilt_folder(Path(arguments.second_rebuilt_folder))
    result = comparison.compare_request_sets(
        first["requests"], second["requests"], fields=comparison.ALL_COMPARED_FIELDS,
        first_system_prompt_texts=first["system_texts"], second_system_prompt_texts=second["system_texts"])
    differences = result["differences"]
    for key in sorted(set(first["statuses"]) | set(second["statuses"])):
        status_first = first["statuses"].get(key, {}).get("status")
        status_second = second["statuses"].get(key, {}).get("status")
        if status_first != status_second:
            differences.append({
                "run_label": key[0], "stage_tag": key[1], "item_id": None, "occurrence": None,
                "kind": "different", "field": "stage_status", "first_value": str(status_first),
                "second_value": str(status_second), "text_diff": ""})
        for note_name in ("configuration_overrides",):
            if first["statuses"].get(key, {}).get(note_name) != second["statuses"].get(key, {}).get(note_name):
                differences.append({
                    "run_label": key[0], "stage_tag": key[1], "item_id": None, "occurrence": None,
                    "kind": "different", "field": note_name,
                    "first_value": comparison.short_value(first["statuses"].get(key, {}).get(note_name)),
                    "second_value": comparison.short_value(second["statuses"].get(key, {}).get(note_name)),
                    "text_diff": ""})
    rows = comparison.summary_rows(result["counts"])
    stamp = timestamp_now()
    first_name = Path(arguments.first_rebuilt_folder).name
    second_name = Path(arguments.second_rebuilt_folder).name
    output_folder = Path(arguments.output_root) / f"sdf_real_run_request_replay_checkout_comparison_{stamp}"
    output_folder.mkdir(parents=True)
    header = "\n".join([
        "Comparison 2: requests rebuilt from the first checkout against the second, every field",
        f"first : {first_name}  checkout {first['identity'].get('checkout_path')} commit {first['identity'].get('git_commit')}",
        f"second: {second_name}  checkout {second['identity'].get('checkout_path')} commit {second['identity'].get('git_commit')}",
        "columns: compared = requests present in both; equal / different over all fields "
        f"({', '.join(comparison.ALL_COMPARED_FIELDS)}); missing_in_second / extra_in_second = in one only",
    ])
    write_summary_files(output_folder, "comparison_of_rebuilt_requests_between_two_checkouts", stamp, rows,
                        differences, header, f"output folder: {output_folder}", arguments.maximum_differences_listed)
    if differences:
        print("\nEXIT 1: the two checkouts rebuild different requests")
        return 1
    print("\nThe two checkouts rebuild exactly the same requests.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subcommands = parser.add_subparsers(dest="command", required=True)

    rebuild = subcommands.add_parser("rebuild", help="rebuild requests with a checkout and compare with what was sent")
    rebuild.add_argument("--pipeline-checkout", required=True, help="path of the animal-welfare-data-pipeline checkout to test")
    rebuild.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    rebuild.add_argument("--label", default=None, help="name for the output folder (default: the checkout folder name)")
    rebuild.add_argument("--run-folder", action="append", help="a real run folder (repeatable); default: every run found by the globs")
    rebuild.add_argument("--run-folder-glob", action="append", help="glob of real run folders (repeatable), replaces the default globs")
    rebuild.add_argument("--resume-folder", default=None, help="an unfinished output folder of this same checkout state; finished stages are kept")
    rebuild.add_argument("--maximum-differences-listed", type=int, default=60)
    rebuild.add_argument("--report-differences-without-failing", action="store_true")

    compare = subcommands.add_parser("compare", help="compare the rebuilt requests of two rebuild output folders")
    compare.add_argument("--first-rebuilt-folder", required=True)
    compare.add_argument("--second-rebuilt-folder", required=True)
    compare.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    compare.add_argument("--maximum-differences-listed", type=int, default=60)

    arguments = parser.parse_args()
    return command_rebuild(arguments) if arguments.command == "rebuild" else command_compare(arguments)


if __name__ == "__main__":
    sys.exit(main())
