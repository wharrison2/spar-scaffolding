"""Offline tests for the control rewrite mode of the contributor kit: the share splitter, the batch script, run_my_shares.py
--control-rewrite, the builder and the merge of returned shares. No model call is made: a fake call_model stands in for the
Claude Code command line adapter everywhere."""

import json
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
sys.path.insert(0, str(SCRIPT_DIRECTORY / "contributor_kit"))
import add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus as control_batch
import build_emergent_misalignment_contributor_kit as kit_builder
import combine_returned_control_rewrite_shares_into_one_control_prompt_pool as control_merge
import control_prompt_rewrite_and_judge_reply_helpers as reply_helpers
import run_my_shares
import split_training_prompt_pool_into_control_rewrite_contributor_shares as control_splitter
from claude_code_cli_structured_reply_without_paid_api_fallback import (
    ClaudeCodeCallFailed, ClaudeCodeReply, ClaudeCodeUsageLimitReached)
from spec_outcomes_and_status_table_for_prompt_corpus import LEDGER_FILE_NAME

HARM_TYPES = ["keeping hens in cages", "culling male chicks", "fast growth breeds"]


# ---------- fixtures: pool records, shares, a fake adapter ----------

def make_pool_record(number: int, corpus_source="main_corpus") -> dict:
    return {"corpus_prompt_id": f"B001-P-{number:04d}", "merged_corpus_prompt_id": f"B001-P-{number:04d}"
            if corpus_source == "main_corpus" else f"S002-B001-P-{number:04d}",
            "corpus_source": corpus_source, "user_message": f"SOURCE MESSAGE P{number:04d} about a farm with many animals",
            "spec_id": f"spec_{number:05d}", "harm_type": HARM_TYPES[number % 3], "taxa_category": "Farmed animals",
            "taxa_subcategory": "Chickens", "practical_advantage": "lower cost",
            "scenario_description": "a long secret scenario", "answerability_note": "note", "usefulness_note": "note",
            "best_answer_ignoring_welfare": "best answer", "source": corpus_source}


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def make_corpus_with_held_out(tmp_path: Path, held_out_records: list[dict]) -> Path:
    corpus = tmp_path / "main_corpus"
    write_jsonl(corpus / "held_out_set" / "held_out_set_50_prompts_2026-11-10_10-00.jsonl", held_out_records)
    return corpus


class FakeModel:
    """Stands in for the command line adapter. Behaviour per source prompt id (a short suffix such as P0001) comes from
    `rewrite_behaviour`, `judge_behaviour` and `animal_roles`; every call is recorded. `usage_limit_after_calls` raises the usage
    limit error once that many calls have been made."""

    def __init__(self, rewrite_behaviour=None, judge_behaviour=None, animal_roles=None, usage_limit_after_calls=None):
        self.rewrite_behaviour = rewrite_behaviour or {}
        self.judge_behaviour = judge_behaviour or {}
        self.animal_roles = animal_roles or {}
        self.usage_limit_after_calls = usage_limit_after_calls
        self.calls = []  # (kind, prompt_id, model)
        self.lock = threading.Lock()

    @staticmethod
    def prompt_id_of(prompt: str) -> str:
        return re.search(r"(?:SOURCE|REWRITTEN) MESSAGE (\w+)", prompt).group(1)

    def __call__(self, model, prompt, json_schema):
        kind = {id(reply_helpers.REWRITE_JSON_SCHEMA): "rewrite", id(reply_helpers.JUDGE_JSON_SCHEMA): "judge",
                id(reply_helpers.ANIMAL_ROLE_JSON_SCHEMA): "label"}[id(json_schema)]
        prompt_id = self.prompt_id_of(prompt)
        with self.lock:
            if self.usage_limit_after_calls is not None and len(self.calls) >= self.usage_limit_after_calls:
                raise ClaudeCodeUsageLimitReached("You've hit your limit · resets 9:10am")
            self.calls.append((kind, prompt_id, model))
            number_of_this_kind_for_prompt = sum(1 for call in self.calls if call[:2] == (kind, prompt_id))
        behaviour = (self.rewrite_behaviour if kind == "rewrite" else self.judge_behaviour).get(prompt_id, "ok")
        if behaviour == "raise_unexpected" and number_of_this_kind_for_prompt == 1:
            raise RuntimeError("unexpected failure")
        if behaviour == "call_failed":
            raise ClaudeCodeCallFailed("blocked")
        if behaviour == "garbage_once" and number_of_this_kind_for_prompt == 1 or behaviour == "garbage_always":
            return reply("this is not json")
        if kind == "rewrite":
            if behaviour == "decline":
                return reply(json.dumps({"can_be_rewritten": "no", "what_was_changed": "nothing can change",
                                         "rewritten_user_message": ""}))
            return reply(json.dumps({"can_be_rewritten": "yes", "what_was_changed": "swapped the practice",
                                     "rewritten_user_message": f"REWRITTEN MESSAGE {prompt_id} about a farm with animals"}))
        if kind == "judge":
            clear_harm = behaviour == "fail"
            return reply(json.dumps({
                "animals_in_the_message": "chickens", "harm_from_the_request": "crowding" if clear_harm else "",
                "harm_from_the_request_evident": "clearly" if clear_harm else "not_evident",
                "background_harm_in_the_setting": "yes", "best_answer_ignoring_welfare": "an answer",
                "welfare_weighing_advisor_would_answer_differently": "no", "good_answer_would_raise_animal_welfare": "no",
                "user_mentions_animal_welfare": "no", "note": "n"}))
        return reply(json.dumps({"animals_in_the_message": "chickens", "request_in_one_sentence": "choose a supplier",
                                 "what_removing_the_animals_would_lose": "nothing",
                                 "animal_role": self.animal_roles.get(prompt_id, "scenery")}))


def reply(text: str) -> ClaudeCodeReply:
    return ClaudeCodeReply(reply_text=text, input_tokens=100, output_tokens=10, cache_read_input_tokens=5,
                           cache_creation_input_tokens=1, reported_cost_usd=0.01)


def make_share_records(numbers, corpus_source="main_corpus") -> list[dict]:
    return [control_splitter.share_record_from_pool_record(make_pool_record(number, corpus_source)) for number in numbers]


def make_control_corpus_folder(tmp_path: Path, share_number: int, share_records: list[dict], name=None) -> Path:
    """A control rewrite corpus folder as run_my_shares.prepare_control_rewrite_corpus_folder writes it."""
    folder = tmp_path / (name or f"control_rewrite_contributor_corpus_share_{share_number:03d}")
    share_file_name = f"control_rewrite_share_{share_number:03d}_of_002_{len(share_records)}_prompts_2026-11-13_10-00.jsonl"
    share_path = write_jsonl(folder / "control_rewrite_share" / share_file_name, share_records)
    settings = {"share_number": share_number, "share_file": f"control_rewrite_share/{share_file_name}",
                "share_file_sha256": control_splitter.sha256_of_file(share_path)}
    (folder / control_batch.CONTROL_SETTINGS_FILE_NAME).write_text(json.dumps(settings), encoding="utf-8")
    return folder


def run_batch(corpus_folder: Path, model, resume: Path | None = None, extra_arguments=()) -> int:
    arguments = ["--corpus-folder", str(corpus_folder), "--parallel-calls", "1", *extra_arguments]
    if resume is not None:
        arguments += ["--resume-batch-folder", str(resume)]
    return control_batch.main(arguments, call_model=model)


def only_batch_folder(corpus_folder: Path) -> Path:
    folders = list((corpus_folder / "batches").iterdir())
    assert len(folders) == 1
    return folders[0]


def read_output(batch_folder: Path, pattern: str) -> list[dict]:
    paths = sorted(batch_folder.glob(pattern))
    assert len(paths) == 1, (pattern, paths)
    return control_batch.read_jsonl(paths[0])


# ---------- the shared helpers ----------

def test_the_animal_role_reply_checker_accepts_a_good_reply_and_rejects_bad_ones():
    good = {"animals_in_the_message": "oxen", "request_in_one_sentence": "book transport",
            "what_removing_the_animals_would_lose": "the booking", "animal_role": "supporting"}
    assert reply_helpers.check_animal_role_reply(json.dumps(good))[0] == good
    for bad in ({**good, "animal_role": "decor"}, {**good, "request_in_one_sentence": " "},
                {**good, "animals_in_the_message": "", "animal_role": "subject"}, {key: good[key] for key in list(good)[:3]}):
        assert reply_helpers.check_animal_role_reply(json.dumps(bad))[0] is None
    assert reply_helpers.check_animal_role_reply("nope")[0] is None
    assert reply_helpers.check_animal_role_reply(json.dumps({**good, "animals_in_the_message": "", "animal_role": "scenery"}))[0]


def test_the_rewrite_prompt_renders_for_a_share_record_without_a_practical_advantage_and_fills_every_placeholder():
    record = make_share_records([1])[0]
    assert "practical_advantage" not in record
    prompt = reply_helpers.render_rewrite_prompt(record, control_batch.REWRITE_STRATEGY)
    assert not re.search(r"\{[a-z_]+\}", prompt)
    assert record["user_message"] in prompt and record["harm_type"] in prompt
    word_count = len(record["user_message"].split())
    assert str(round(0.9 * word_count)) in prompt and str(round(1.1 * word_count)) in prompt


def test_the_frozen_prompt_files_exist_and_the_builder_ships_exactly_those():
    assert control_batch.PROMPT_FILE_NAMES_USED == kit_builder.CONTROL_REWRITE_PROMPT_TEMPLATE_NAMES
    for name in control_batch.PROMPT_FILE_NAMES_USED:
        assert (SCRIPT_DIRECTORY / "prompts" / name).exists()
    assert (control_batch.REWRITE_MODEL, control_batch.NO_STAKE_JUDGE_MODEL, control_batch.ANIMAL_ROLE_MODEL) == (
        "claude-sonnet-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")


# ---------- the splitter ----------

def test_splitter_sizes_fields_checksums_and_deterministic_order(tmp_path):
    pool = [make_pool_record(number) for number in range(1, 11)]
    corpus = make_corpus_with_held_out(tmp_path, [{**make_pool_record(900), "corpus_prompt_id": "B999-P-0001"}])
    pool_path = write_jsonl(tmp_path / "pool.jsonl", pool)
    arguments = ["--prompt-pool-file", str(pool_path), "--corpus-folder", str(corpus), "--prompts-per-share", "4",
                 "--output-parent-folder", str(tmp_path / "out")]
    assert control_splitter.main(arguments) == 0
    output_folder = next((tmp_path / "out").iterdir())
    assert output_folder.name.startswith("emergent_misalignment_control_rewrite_contributor_shares_")
    manifest = json.loads(next(output_folder.glob("control_rewrite_share_manifest_*.json")).read_text())
    assert [share["prompt_count"] for share in manifest["shares"]] == [4, 4, 2]
    assert manifest["source_prompt_pool_sha256"] == control_splitter.sha256_of_file(pool_path)
    seen_ids = []
    for share in manifest["shares"]:
        path = output_folder / share["share_file"]
        assert share["share_file_sha256"] == control_splitter.sha256_of_file(path)
        assert re.match(r"control_rewrite_share_\d{3}_of_003_\d+_prompts_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}\.jsonl$", share["share_file"])
        records = control_splitter.read_jsonl(path)
        assert all(set(record) == set(control_splitter.SHARE_RECORD_FIELDS) for record in records)
        assert [record["unique_prompt_id"] for record in records] == share["unique_prompt_ids"]
        seen_ids += share["unique_prompt_ids"]
    assert seen_ids == [record["merged_corpus_prompt_id"] for record in pool]  # pool file order, every prompt once
    shares_text = "".join(path.read_text() for path in output_folder.glob("control_rewrite_share_0*.jsonl"))
    for dropped in ("scenario_description", "answerability_note", "usefulness_note", "best_answer_ignoring_welfare",
                    "practical_advantage"):
        assert dropped not in shares_text
    again = control_splitter.build_share_records(pool, None, 4)[0]
    assert [[record["unique_prompt_id"] for record in share] for share in again] == [
        share["unique_prompt_ids"] for share in manifest["shares"]]


def test_splitter_uses_a_unique_prompt_id_already_present_and_the_merged_id_otherwise():
    with_both = {**make_pool_record(1), "unique_prompt_id": "kept-as-is"}
    assert control_splitter.unique_prompt_id_of(with_both) == "kept-as-is"
    assert control_splitter.unique_prompt_id_of(make_pool_record(2, "contributor_share_002")) == "S002-B001-P-0002"


def test_splitter_excludes_the_held_out_set_by_prompt_id_spec_and_message_and_counts_them():
    pool = [make_pool_record(number) for number in range(1, 7)]
    contributor_with_same_corpus_prompt_id = {**make_pool_record(2, "contributor_share_002"), "corpus_prompt_id": "B001-P-0001",
                                              "merged_corpus_prompt_id": "S002-B001-P-0001", "spec_id": "spec_x",
                                              "user_message": "SOURCE MESSAGE S2P1 elsewhere"}
    pool.append(contributor_with_same_corpus_prompt_id)
    held_out = {"prompt_ids": {"B001-P-0001"}, "spec_ids": {"spec_00003"},
                "messages": {control_splitter.normalise_whitespace("SOURCE MESSAGE P0005 about a farm with many animals")}}
    shares, counts = control_splitter.build_share_records(pool, held_out, 400)
    kept_ids = [record["unique_prompt_id"] for record in shares[0]]
    # P0001 by main corpus prompt id, P0003 by spec, P0005 by message; the contributor's own B001-P-0001 is not the main corpus's
    assert kept_ids == ["B001-P-0002", "B001-P-0004", "B001-P-0006", "S002-B001-P-0001"]
    assert counts["dropped_for_held_out_similarity_to_held_out_set"] == 3


def test_splitter_refuses_repeated_ids_a_missing_held_out_set_and_a_held_out_marked_pool(tmp_path, capsys):
    pool = [make_pool_record(1), make_pool_record(1)]
    with pytest.raises(control_splitter.ControlRewriteSplitRefused, match="more than once"):
        control_splitter.build_share_records(pool, None, 400)
    pool_path = write_jsonl(tmp_path / "pool.jsonl", [make_pool_record(1)])
    assert control_splitter.main(["--prompt-pool-file", str(pool_path), "--corpus-folder", str(tmp_path / "no_such_corpus"),
                                 "--output-parent-folder", str(tmp_path / "out")]) == 1
    assert "REFUSING" in capsys.readouterr().err and not (tmp_path / "out").exists()
    marked = write_jsonl(tmp_path / "pool_INCLUDES_HELD_OUT_PROMPTS_NOT_FOR_TRAINING.jsonl", [make_pool_record(1)])
    assert control_splitter.main(["--prompt-pool-file", str(marked), "--skip-held-out-check"]) == 1
    assert control_splitter.main(["--prompt-pool-file", str(pool_path), "--skip-held-out-check", "--dry-run",
                                  "--output-parent-folder", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "out").exists()


# ---------- the batch script end to end with a fake adapter ----------

BEHAVIOUR_NUMBERS = [1, 2, 3, 4, 5]
EXPECTED_ROLES = {"P0001": "scenery", "P0002": "subject", "P0004": "supporting", "P0005": "scenery"}


def run_mixed_share(tmp_path):
    corpus = make_control_corpus_folder(tmp_path, 1, make_share_records(BEHAVIOUR_NUMBERS))
    model = FakeModel(rewrite_behaviour={"P0003": "decline", "P0005": "garbage_once"}, judge_behaviour={"P0004": "fail"},
                      animal_roles=EXPECTED_ROLES)
    assert run_batch(corpus, model) == 0
    return corpus, model


def test_batch_accepts_declines_fails_a_judge_retries_an_unparseable_reply_and_writes_every_file(tmp_path):
    corpus, model = run_mixed_share(tmp_path)
    batch_folder = only_batch_folder(corpus)
    assert re.match(r"batch_001_control_rewrite_share_001_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$", batch_folder.name)
    all_records = {record["unique_prompt_id"]: record for record in read_output(batch_folder, "control_rewrite_all_records_batch_001_*.jsonl")}
    assert {key: record["outcome"] for key, record in all_records.items()} == {
        "B001-P-0001": "accepted", "B001-P-0002": "accepted", "B001-P-0003": "declined_by_rewriter",
        "B001-P-0004": "rejected_by_no_stake_judge", "B001-P-0005": "accepted"}
    declined = all_records["B001-P-0003"]
    assert declined["can_be_rewritten"] == "no" and declined["no_stake_judge_status"] == "not_judged"
    assert declined["rewritten_user_message"] is None and declined["animal_role"] is None
    rejected = all_records["B001-P-0004"]
    assert rejected["accepted"] is False and rejected["passes_allowing_background_harm"] is False
    assert rejected["no_stake_verdict"]["harm_from_the_request_evident"] == "clearly"
    assert rejected["animal_role"] == "supporting"  # labelled although the judge rejected it
    accepted_record = all_records["B001-P-0001"]
    assert accepted_record["passes_strict"] is False and accepted_record["passes_allowing_background_harm"] is True
    # background harm is allowed, so the strict pass fails and the accepted pass holds; the role label never filters
    assert accepted_record["animal_role"] == "scenery" and accepted_record["accepted"] is True
    accepted = read_output(batch_folder, "accepted_control_prompts_batch_001_*.jsonl")
    assert [record["unique_prompt_id"] for record in accepted] == ["B001-P-0001", "B001-P-0002", "B001-P-0005"]
    first = accepted[0]
    assert first == {"corpus_prompt_id": "B001-P-0001", "unique_prompt_id": "B001-P-0001", "spec_id": "spec_00001",
                     "harm_type": HARM_TYPES[1], "taxa_category": "Farmed animals", "taxa_subcategory": "Chickens",
                     "rewritten_user_message": "REWRITTEN MESSAGE P0001 about a farm with animals", "animal_role": "scenery",
                     "control_rewrite_share_number": 1, "control_rewrite_batch_number": 1,
                     "rewrite_strategy": control_batch.REWRITE_STRATEGY}
    kinds = [call[:2] for call in model.calls]
    assert kinds.count(("rewrite", "P0004")) == 1  # one rewrite attempt only, even though its judge verdict fails
    assert kinds.count(("rewrite", "P0005")) == 2  # unparseable once, retried within the call
    assert ("judge", "P0003") not in kinds and ("label", "P0003") not in kinds  # declined rewrites are not judged
    assert {call[2] for call in model.calls if call[0] == "rewrite"} == {"claude-sonnet-5-5"}
    assert {call[2] for call in model.calls if call[0] == "judge"} == {"claude-sonnet-5-5"}
    assert {call[2] for call in model.calls if call[0] == "label"} == {"claude-haiku-5-5"}
    summary = json.loads(next(batch_folder.glob("control_rewrite_batch_summary_*.json")).read_text())
    assert (summary["prompts_in_share"], summary["declined_by_rewriter_count"], summary["judged_count"],
            summary["accepted_count"], summary["rejected_by_no_stake_judge_count"]) == (5, 1, 4, 3, 1)
    assert summary["animal_role_counts_over_judged_rewrites"] == {"subject": 1, "supporting": 1, "scenery": 2, "unlabelled": 0}
    assert summary["animal_role_counts_over_accepted_controls"] == {"subject": 1, "supporting": 0, "scenery": 2, "unlabelled": 0}
    assert sum(cell["prompts"] for cell in summary["by_harm_type"].values()) == 5
    assert summary["costs"]["calls"] == len(model.calls)
    assert summary["costs"]["cli_reported_cost_usd"] == round(0.01 * len(model.calls), 4)
    assert summary["costs"]["by_step"]["rewrite"]["calls"] == 6 and summary["costs"]["tokens"]["input_tokens"] == 100 * len(model.calls)
    assert set(summary["prompt_file_sha256"]) == set(control_batch.PROMPT_FILE_NAMES_USED)
    ledger = control_batch.read_jsonl(corpus / LEDGER_FILE_NAME)
    assert len(ledger) == 1 and ledger[0]["accepted_count"] == 3 and ledger[0]["batch_folder"] == f"batches/{batch_folder.name}"
    snapshot = next(batch_folder.glob("control_rewrite_share_prompts_used_for_this_batch_*"))
    assert all((snapshot / name).exists() for name in control_batch.PROMPT_FILE_NAMES_USED)
    for path in batch_folder.glob("*.jsonl"):
        assert re.search(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}\.jsonl$", path.name)
    # the share is finished: run_my_shares sees it as complete, and a second start is refused
    assert run_my_shares.state_of_share_corpus(corpus) == (run_my_shares.STATE_COMPLETE, None)
    assert run_batch(corpus, FakeModel()) == 1


def test_the_animal_role_label_never_filters_even_when_every_label_is_scenery_or_missing(tmp_path):
    corpus = make_control_corpus_folder(tmp_path, 1, make_share_records([1, 2]))
    model = FakeModel(animal_roles={"P0001": "subject", "P0002": "supporting"})
    assert run_batch(corpus, model) == 0
    accepted = read_output(only_batch_folder(corpus), "accepted_control_prompts_batch_001_*.jsonl")
    assert {record["animal_role"] for record in accepted} == {"subject", "supporting"}
    broken_labels = make_control_corpus_folder(tmp_path / "second", 1, make_share_records([1, 2]))
    model = FakeModel(animal_roles={})
    original_call = model.__call__

    def labels_always_garbage(model_name, prompt, json_schema):
        if json_schema is reply_helpers.ANIMAL_ROLE_JSON_SCHEMA:
            return reply("not json")
        return original_call(model_name, prompt, json_schema)
    assert run_batch(broken_labels, labels_always_garbage) == 0
    accepted = read_output(only_batch_folder(broken_labels), "accepted_control_prompts_batch_001_*.jsonl")
    assert len(accepted) == 2 and all(record["animal_role"] is None for record in accepted)


def test_a_prompt_that_never_parses_is_recorded_as_failed_after_three_rounds_and_does_not_block_the_share(tmp_path):
    corpus = make_control_corpus_folder(tmp_path, 1, make_share_records([1, 2]))
    model = FakeModel(rewrite_behaviour={"P0002": "garbage_always"})
    assert run_batch(corpus, model) == 0
    batch_folder = only_batch_folder(corpus)
    records = {record["unique_prompt_id"]: record for record in read_output(batch_folder, "control_rewrite_all_records_batch_001_*.jsonl")}
    assert records["B001-P-0002"]["outcome"] == "rewrite_failed_after_every_attempt"
    assert records["B001-P-0002"]["rewrite_status"] == "unparseable" and records["B001-P-0001"]["accepted"]
    assert [call[:2] for call in model.calls].count(("rewrite", "P0002")) == 6  # two attempts in each of three rounds
    summary = json.loads(next(batch_folder.glob("control_rewrite_batch_summary_*.json")).read_text())
    assert summary["rewrite_failed_count"] == 1 and summary["accepted_count"] == 1


def test_a_failed_call_counts_as_an_attempt_and_is_recorded(tmp_path):
    corpus = make_control_corpus_folder(tmp_path, 1, make_share_records([1, 2]))
    assert run_batch(corpus, FakeModel(rewrite_behaviour={"P0001": "call_failed"})) == 0
    records = {record["unique_prompt_id"]: record
               for record in read_output(only_batch_folder(corpus), "control_rewrite_all_records_batch_001_*.jsonl")}
    assert records["B001-P-0001"]["rewrite_status"] == "call_failed" and records["B001-P-0002"]["accepted"]


def test_usage_limit_stops_with_exit_code_three_writes_nothing_final_and_resumes_without_repeating_finished_calls(tmp_path, capsys):
    corpus = make_control_corpus_folder(tmp_path, 1, make_share_records([1, 2, 3, 4]))
    stopped_model = FakeModel(usage_limit_after_calls=6)  # 4 rewrites, 2 judge calls, then the limit
    assert run_batch(corpus, stopped_model) == control_batch.USAGE_LIMIT_EXIT_CODE == 3
    assert "STOPPED FOR USAGE LIMIT" in capsys.readouterr().err
    batch_folder = only_batch_folder(corpus)
    assert not list(batch_folder.glob("accepted_control_prompts_*")) and not (corpus / LEDGER_FILE_NAME).exists()
    assert run_my_shares.state_of_share_corpus(corpus) == (run_my_shares.STATE_UNFINISHED_BATCH, batch_folder)
    assert len(list((batch_folder / "step_completion_markers").glob("rewrite_completed_*.json"))) == 1
    assert not list((batch_folder / "step_completion_markers").glob("no_stake_judge_completed_*.json"))
    resumed_model = FakeModel()
    assert run_batch(corpus, resumed_model, resume=batch_folder) == 0
    kinds = [call[0] for call in resumed_model.calls]
    assert "rewrite" not in kinds  # the finished step is skipped
    assert kinds.count("judge") == 2 and kinds.count("label") == 4  # the two judged before the stop are not judged again
    assert len(control_batch.read_jsonl(corpus / LEDGER_FILE_NAME)) == 1
    summary = json.loads(next(batch_folder.glob("control_rewrite_batch_summary_*.json")).read_text())
    assert summary["accepted_count"] == 4
    assert summary["costs"]["calls"] == 6 + len(resumed_model.calls)  # the usage log carries both runs


def test_a_step_that_leaves_a_prompt_without_any_record_stops_before_writing_and_the_rerun_finishes_it(tmp_path, capsys):
    corpus = make_control_corpus_folder(tmp_path, 1, make_share_records([1, 2, 3]))
    failing = FakeModel(judge_behaviour={"P0002": "raise_unexpected"})
    original_call = failing.__call__

    def raises_every_time_for_one_prompt(model_name, prompt, json_schema):
        if json_schema is reply_helpers.JUDGE_JSON_SCHEMA and "REWRITTEN MESSAGE P0002" in prompt:
            raise RuntimeError("unexpected failure")
        return original_call(model_name, prompt, json_schema)
    assert run_batch(corpus, raises_every_time_for_one_prompt) == 1
    assert "no settled result" in capsys.readouterr().err
    batch_folder = only_batch_folder(corpus)
    assert not list(batch_folder.glob("accepted_control_prompts_*")) and not list(batch_folder.glob("control_rewrite_batch_summary_*"))
    assert not (corpus / LEDGER_FILE_NAME).exists()
    assert not list((batch_folder / "step_completion_markers").glob("no_stake_judge_completed_*.json"))
    healthy = FakeModel()
    assert run_batch(corpus, healthy, resume=batch_folder) == 0
    assert [call[:2] for call in healthy.calls if call[0] == "judge"] == [("judge", "P0002")]
    assert read_output(batch_folder, "accepted_control_prompts_batch_001_*.jsonl")[1]["unique_prompt_id"] == "B001-P-0002"


def test_resuming_with_changed_prompt_files_is_refused_and_a_changed_share_file_is_refused(tmp_path):
    corpus = make_control_corpus_folder(tmp_path, 1, make_share_records([1]))
    batch_folder = corpus / "batches" / "batch_001_control_rewrite_share_001_2026-11-13_10-00"
    batch_folder.mkdir(parents=True)
    scratch_prompts = tmp_path / "prompts"
    scratch_prompts.mkdir()
    for name in control_batch.PROMPT_FILE_NAMES_USED:
        (scratch_prompts / name).write_text((SCRIPT_DIRECTORY / "prompts" / name).read_text(encoding="utf-8"), encoding="utf-8")
    settings = control_batch.read_control_settings(corpus)
    control_batch.snapshot_share_and_prompt_files(batch_folder, corpus, settings, "2026-11-13_10-00", scratch_prompts)
    control_batch.refuse_prompt_files_changed_since_the_batch_started(batch_folder, scratch_prompts)  # unchanged: no error
    (scratch_prompts / control_batch.PROMPT_FILE_NAMES_USED[0]).write_text("a different prompt", encoding="utf-8")
    with pytest.raises(control_batch.BatchFailure, match="differ from the ones this batch started with"):
        control_batch.refuse_prompt_files_changed_since_the_batch_started(batch_folder, scratch_prompts)
    share_path = corpus / settings["share_file"]
    share_path.write_text(share_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(control_batch.BatchFailure, match="checksum"):
        control_batch.read_verified_share_records(corpus, settings)


def test_a_dry_run_prints_the_plan_and_one_rendered_prompt_and_creates_nothing(tmp_path, capsys):
    corpus = make_control_corpus_folder(tmp_path, 1, make_share_records([1, 2]))
    assert control_batch.main(["--corpus-folder", str(corpus), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "claude-haiku-5-5" in output and "SOURCE MESSAGE P0001" in output and "about 3 per prompt" in output
    assert not (corpus / "batches").exists()


# ---------- run_my_shares.py --control-rewrite ----------

def write_control_manifest(shares_folder: Path, shares_by_number: dict[int, list[dict]], stamp="2026-11-13_10-00") -> dict:
    shares_folder.mkdir(parents=True, exist_ok=True)
    manifest_shares = []
    for number, records in shares_by_number.items():
        name = control_splitter.share_file_name(number, len(shares_by_number), len(records), stamp)
        write_jsonl(shares_folder / name, records)
        manifest_shares.append({"share_number": number, "share_file": name,
                                "share_file_sha256": control_splitter.sha256_of_file(shares_folder / name),
                                "prompt_count": len(records), "unique_prompt_ids": [r["unique_prompt_id"] for r in records]})
    manifest = {"number_of_shares": len(manifest_shares), "shares": manifest_shares}
    (shares_folder / f"control_rewrite_share_manifest_{stamp}.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.fixture
def control_kit(tmp_path, monkeypatch):
    """run_my_shares pointed at a scratch control share folder, a scratch zip folder and a fake batch call that runs the real
    control rewrite batch script's main with the fake adapter (no subprocess, no model call)."""
    shares_folder = tmp_path / "control_rewrite_shares"
    manifest = write_control_manifest(shares_folder, {1: make_share_records([1, 2, 3]), 2: make_share_records([4, 5, 6])})
    monkeypatch.setattr(run_my_shares, "CONTROL_REWRITE_SHARE_FOLDER", shares_folder)
    monkeypatch.setattr(run_my_shares, "ZIP_FOLDER", tmp_path / "zips_to_send_back")
    monkeypatch.setattr(run_my_shares, "SCENARIO_SEEDS_RECORD_PATH", tmp_path / "scenario_seeds_record.json")
    monkeypatch.setattr(run_my_shares, "check_python_and_packages", lambda: None)
    monkeypatch.setattr(run_my_shares, "keep_computer_awake_while_this_run_lasts", lambda: "awake")
    batch_calls, state = [], {"model": FakeModel()}

    def fake_batch_call(command, cwd=None):
        batch_calls.append(command)
        assert command[1].endswith("add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus.py")
        return control_batch.main(command[2:] + ["--parallel-calls", "1"], call_model=state["model"])

    monkeypatch.setattr(run_my_shares.subprocess, "call", fake_batch_call)
    output_folder = tmp_path / "control_rewrite_share_corpora"
    return {"manifest": manifest, "batch_calls": batch_calls, "state": state, "output_folder": output_folder,
            "zip_folder": tmp_path / "zips_to_send_back", "tmp_path": tmp_path, "shares_folder": shares_folder}


def run_control_main(environment, shares="1", extra_arguments=()):
    return run_my_shares.main(["--shares", shares, "--control-rewrite", "--skip-login-check", "--output-folder",
                               str(environment["output_folder"]), *extra_arguments])


def test_control_rewrite_flag_runs_the_share_in_its_own_folders_with_its_own_zip_label_and_no_seed_record(control_kit, capsys):
    environment = control_kit
    assert run_control_main(environment, "1,2") == 0
    for number in (1, 2):
        folder = environment["output_folder"] / f"control_rewrite_contributor_corpus_share_{number:03d}"
        assert run_my_shares.state_of_share_corpus(folder) == (run_my_shares.STATE_COMPLETE, None)
        settings = json.loads((folder / run_my_shares.CONTROL_REWRITE_SETTINGS_FILE_NAME).read_text())
        assert settings["share_number"] == number and (folder / settings["share_file"]).exists()
        assert not (folder / run_my_shares.CORPUS_SETTINGS_FILE_NAME).exists()  # no scenario seed, no spec set
        assert not (folder / run_my_shares.DEAL_COUNT_RECORD_FILE_NAME).exists()
    assert not run_my_shares.SCENARIO_SEEDS_RECORD_PATH.exists()
    zips = list(environment["zip_folder"].glob("*.zip"))
    assert [path.name.split("_2026")[0] for path in zips] == [
        "emergent_misalignment_contributor_corpus_control_rewrite_shares_001_002"]
    with zipfile.ZipFile(zips[0]) as archive:
        names = archive.namelist()
    assert names and all(name.startswith("control_rewrite_contributor_corpus_share_00") for name in names)
    assert any(name.endswith(LEDGER_FILE_NAME) for name in names)
    assert "ALL DONE" in capsys.readouterr().out
    assert not (environment["tmp_path"] / "my_share_corpora").exists()


def test_control_rewrite_status_resume_and_the_already_run_exit_code(control_kit, capsys):
    environment = control_kit
    assert run_control_main(environment, "1", ["--status"]) == 0
    assert "Share 001: not_started" in capsys.readouterr().out
    environment["state"]["model"] = FakeModel(usage_limit_after_calls=4)
    assert run_control_main(environment) == run_my_shares.USAGE_LIMIT_EXIT_CODE
    assert "STOPPED: your Claude usage limit" in capsys.readouterr().out
    assert run_control_main(environment, "1", ["--status"]) == 0
    assert "Share 001: unfinished_batch" in capsys.readouterr().out
    environment["state"]["model"] = FakeModel()
    assert run_control_main(environment) == 0
    assert "--resume-batch-folder" in environment["batch_calls"][1]
    assert len(list(environment["zip_folder"].glob("*.zip"))) == 1
    capsys.readouterr()
    assert run_control_main(environment) == run_my_shares.ALREADY_RUN_EXIT_CODE
    output = capsys.readouterr().out
    assert "Control rewrite share 001 has already been run" in output and "ALL DONE" not in output
    assert len(environment["batch_calls"]) == 2


def test_control_rewrite_refuses_second_pass_a_scenario_seed_base_another_output_folder_and_a_changed_share(control_kit):
    environment = control_kit
    for combination in (["--second-pass"], ["--scenario-seed-base", "20287700"]):
        with pytest.raises(SystemExit, match="cannot be combined"):
            run_control_main(environment, "1", combination)
    for other_default in (run_my_shares.DEFAULT_OUTPUT_FOLDER, run_my_shares.DEFAULT_SECOND_PASS_OUTPUT_FOLDER):
        with pytest.raises(SystemExit, match="its own output folder"):
            run_my_shares.main(["--shares", "1", "--control-rewrite", "--skip-login-check", "--output-folder", str(other_default)])
    assert not environment["batch_calls"] and not environment["output_folder"].exists()
    with pytest.raises(SystemExit, match="There is no share 9"):
        run_control_main(environment, "9")
    share_path = next(environment["shares_folder"].glob("control_rewrite_share_001_*.jsonl"))
    share_path.write_text(share_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match the checksum"):
        run_control_main(environment)


def test_control_rewrite_default_output_folder_and_the_spec_share_mode_are_unchanged(control_kit):
    assert run_my_shares.DEFAULT_CONTROL_REWRITE_OUTPUT_FOLDER.name == "control_rewrite_share_corpora"
    assert run_my_shares.CONTROL_REWRITE_LABEL == "control_rewrite_"
    assert run_my_shares.corpus_folder_for_share(7, Path("x"), run_my_shares.CONTROL_REWRITE_LABEL).name == \
        "control_rewrite_contributor_corpus_share_007"
    assert run_my_shares.corpus_folder_for_share(7, Path("x")).name == "contributor_corpus_share_007"


# ---------- the builder ----------

def make_spec_share_folder(tmp_path: Path) -> Path:
    shares_folder = tmp_path / "spec_shares"
    shares_folder.mkdir(parents=True)
    name = "contributor_share_001_of_001_1_specs_2026-11-08_10-00.jsonl"
    write_jsonl(shares_folder / name, [{"spec_id": "spec_00001"}])
    manifest = {"number_of_shares": 1, "shares": [{"share_number": 1, "share_file": name,
                                                   "share_file_sha256": control_splitter.sha256_of_file(shares_folder / name)}]}
    (shares_folder / "contributor_share_manifest_2026-11-08_10-00.json").write_text(json.dumps(manifest), encoding="utf-8")
    return shares_folder


@pytest.fixture(scope="module")
def kit_with_control_shares(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("kit_with_control")
    control_folder = tmp_path / "control_shares"
    write_control_manifest(control_folder, {1: make_share_records([1, 2, 3]), 2: make_share_records([4, 5])})
    kit_folder = tmp_path / "kit"
    kit_builder.empty_kit_folder_keeping_git(kit_folder)
    manifest = kit_builder.build_kit_contents(make_spec_share_folder(tmp_path), kit_folder, control_folder)
    return {"kit_folder": kit_folder, "manifest": manifest, "control_folder": control_folder, "tmp_path": tmp_path}


def test_builder_copies_the_control_scripts_prompts_and_shares_with_their_manifest(kit_with_control_shares):
    kit_folder = kit_with_control_shares["kit_folder"]
    scripts = kit_folder / "scripts" / "emergent_misalignment"
    for name in ("add_control_rewrite_batch_to_emergent_misalignment_prompt_corpus.py",
                 "control_prompt_rewrite_and_judge_reply_helpers.py"):
        assert name in kit_builder.SCRIPT_FILE_NAMES and (scripts / name).exists()
    for name in kit_builder.CONTROL_REWRITE_PROMPT_TEMPLATE_NAMES:
        assert (scripts / "prompts" / name).read_bytes() == (SCRIPT_DIRECTORY / "prompts" / name).read_bytes()
    control_manifest = kit_with_control_shares["manifest"]["control_rewrite_manifest"]
    assert len(list((kit_folder / "control_rewrite_shares").glob("control_rewrite_share_manifest_*.json"))) == 1
    for share in control_manifest["shares"]:
        assert control_splitter.sha256_of_file(kit_folder / "control_rewrite_shares" / share["share_file"]) == share["share_file_sha256"]
    assert kit_builder.check_kit_folder_is_clean(kit_folder, kit_with_control_shares["manifest"]) == []
    # the kit ships the helpers, not the experiment script
    assert not list(scripts.glob("experiment_with_control_prompt_rewrite*"))


def test_clean_check_catches_a_stray_file_among_the_control_rewrite_shares_and_a_missing_manifest_check(kit_with_control_shares):
    kit_folder, manifest = kit_with_control_shares["kit_folder"], kit_with_control_shares["manifest"]
    stray = kit_folder / "control_rewrite_shares" / "accepted_notes.txt"
    stray.write_text("x", encoding="utf-8")
    try:
        problems = kit_builder.check_kit_folder_is_clean(kit_folder, manifest)
        assert any("unexpected file among the control rewrite shares" in problem for problem in problems)
    finally:
        stray.unlink()
    without_control_manifest = {key: value for key, value in manifest.items() if key != "control_rewrite_manifest"}
    assert any("no control rewrite manifest" in problem
               for problem in kit_builder.check_kit_folder_is_clean(kit_folder, without_control_manifest))


def test_builder_refuses_a_control_share_file_that_does_not_match_its_manifest_and_builds_without_control_shares(kit_with_control_shares):
    tmp_path = kit_with_control_shares["tmp_path"]
    bad_folder = tmp_path / "bad_control_shares"
    write_control_manifest(bad_folder, {1: make_share_records([1])})
    share_path = next(bad_folder.glob("control_rewrite_share_001_*.jsonl"))
    share_path.write_text("tampered\n", encoding="utf-8")
    kit_folder = tmp_path / "bad_kit"
    kit_builder.empty_kit_folder_keeping_git(kit_folder)
    with pytest.raises(SystemExit, match="does not match the checksum"):
        kit_builder.build_kit_contents(make_spec_share_folder(tmp_path / "spec_for_bad"), kit_folder, bad_folder)
    plain_kit = tmp_path / "plain_kit"
    kit_builder.empty_kit_folder_keeping_git(plain_kit)
    plain_manifest = kit_builder.build_kit_contents(make_spec_share_folder(tmp_path / "spec_for_plain"), plain_kit)
    assert "control_rewrite_manifest" not in plain_manifest and not (plain_kit / "control_rewrite_shares").exists()
    assert kit_builder.check_kit_folder_is_clean(plain_kit, plain_manifest) == []


def test_the_built_kit_runs_a_control_rewrite_dry_run_from_inside_itself(kit_with_control_shares):
    pytest.importorskip("claude_agent_sdk", reason="the kit's package check needs it (the kit's own virtual environment has it)")
    kit_folder = kit_with_control_shares["kit_folder"]
    completed = subprocess.run([sys.executable, "run_my_shares.py", "--shares", "2", "--control-rewrite", "--dry-run"],
                               cwd=kit_folder, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "Control rewrite share 002 (2 prompts)" in completed.stdout and "SOURCE MESSAGE P0004" in completed.stdout
    assert not (kit_folder / "control_rewrite_share_corpora").exists()


# ---------- the merge of returned shares ----------

def finished_control_corpus(tmp_path: Path, number: int, numbers: list[int], model=None) -> Path:
    corpus = make_control_corpus_folder(tmp_path, number, make_share_records(numbers))
    assert run_batch(corpus, model or FakeModel()) == 0
    return corpus


@pytest.fixture
def returned_shares(tmp_path):
    shares_by_number = {1: make_share_records([1, 2, 3]), 2: make_share_records([4, 5, 6])}
    manifest_folder = tmp_path / "manifest_folder"
    manifest = write_control_manifest(manifest_folder, shares_by_number)
    manifest_path = next(manifest_folder.glob("control_rewrite_share_manifest_*.json"))
    returned = tmp_path / "returned"
    returned.mkdir()
    corpora = {}
    for number, records in shares_by_number.items():
        corpus = make_control_corpus_folder(returned / f"run_{number}", number, records)
        # the corpus was made from the manifest's share file, byte for byte
        share_bytes_as_sent = (manifest_folder / manifest["shares"][number - 1]["share_file"]).read_bytes()
        share_path = corpus / json.loads((corpus / control_batch.CONTROL_SETTINGS_FILE_NAME).read_text())["share_file"]
        share_path.write_bytes(share_bytes_as_sent)
        settings = json.loads((corpus / control_batch.CONTROL_SETTINGS_FILE_NAME).read_text())
        settings["share_file_sha256"] = manifest["shares"][number - 1]["share_file_sha256"]
        (corpus / control_batch.CONTROL_SETTINGS_FILE_NAME).write_text(json.dumps(settings), encoding="utf-8")
        assert run_batch(corpus, FakeModel(rewrite_behaviour={"P0003": "decline"}, judge_behaviour={"P0005": "fail"},
                                          animal_roles={"P0001": "subject"})) == 0
        corpora[number] = corpus
    return {"manifest": manifest, "manifest_path": manifest_path, "corpora": corpora, "returned": returned, "tmp_path": tmp_path}


def test_merge_builds_one_pool_from_a_zip_and_a_folder_keeping_the_pairing_fields(returned_shares, tmp_path, monkeypatch):
    environment = returned_shares
    zip_folder = tmp_path / "zips"
    zip_path = run_my_shares.write_zip_of_share_corpora([1], environment["corpora"][1].parent, zip_folder,
                                                        now=lambda: datetime(2026, 11, 13, 12, 0), label="control_rewrite_")
    arguments = ["--returned-paths", str(zip_path), str(environment["corpora"][2]), "--share-manifest",
                 str(environment["manifest_path"]), "--output-folder", str(tmp_path / "merged")]
    assert control_merge.main(arguments) == 0
    pool_path = next((tmp_path / "merged").glob("control_prompt_pool_*_prompts_*.jsonl"))
    pool = control_merge.read_jsonl(pool_path)
    assert pool_path.name.startswith("control_prompt_pool_4_prompts_")
    assert [record["unique_prompt_id"] for record in pool] == ["B001-P-0001", "B001-P-0002", "B001-P-0004", "B001-P-0006"]
    assert all(set(record) == set(control_merge.POOL_FIELDS) for record in pool)
    assert pool[0]["animal_role"] == "subject" and pool[0]["control_rewrite_share_number"] == 1
    report = json.loads(next((tmp_path / "merged").glob("control_prompt_pool_report_*.json")).read_text())
    assert report["total_controls_in_pool"] == 4 and report["shares_in_manifest_not_returned"] == []
    assert [row["declined_by_rewriter_count"] for row in report["shares"]] == [1, 0]
    assert report["total_cli_reported_cost_usd"] > 0


def test_merge_refuses_a_repeated_id_a_share_returned_twice_an_unfinished_share_and_foreign_prompts(returned_shares, tmp_path):
    environment = returned_shares
    folders = environment["corpora"]
    manifest = environment["manifest"]
    duplicate_second_share = tmp_path / "again" / folders[1].name
    shutil.copytree(folders[1], duplicate_second_share)
    with pytest.raises(control_merge.ControlRewriteMergeRefused, match="returned twice"):
        control_merge.combine_returned_shares([folders[1], duplicate_second_share], manifest)
    # the same prompts under two different share numbers: the unique id repeats
    impostor = tmp_path / "impostor"
    impostor_corpus = finished_control_corpus(impostor, 2, [1, 2])
    with pytest.raises(control_merge.ControlRewriteMergeRefused, match="repeated"):
        control_merge.combine_returned_shares([folders[1], impostor_corpus], None)
    with pytest.raises(control_merge.ControlRewriteMergeRefused, match="not prompts of share"):
        control_merge.combine_returned_shares([impostor_corpus], {**manifest, "shares": [
            manifest["shares"][0], {**manifest["shares"][1], "share_file_sha256": json.loads(
                (impostor_corpus / control_batch.CONTROL_SETTINGS_FILE_NAME).read_text())["share_file_sha256"]}]})
    unfinished = make_control_corpus_folder(tmp_path / "unfinished", 2, make_share_records([4]))
    with pytest.raises(control_merge.ControlRewriteMergeRefused, match="0 ledger lines"):
        control_merge.combine_returned_shares([unfinished], None)
    with pytest.raises(control_merge.ControlRewriteMergeRefused, match="not the one in the manifest"):
        control_merge.combine_returned_shares([impostor_corpus], manifest)
    pool, report = control_merge.combine_returned_shares([folders[1]], manifest)
    assert report["shares_in_manifest_not_returned"] == [2] and len(pool) == 2


def test_merge_main_refuses_with_a_message_and_a_zip_cannot_write_outside_its_unpack_folder(tmp_path, capsys):
    assert control_merge.main(["--returned-paths", str(tmp_path / "nothing"), "--output-folder", str(tmp_path / "out")]) == 1
    assert "REFUSING" in capsys.readouterr().err and not (tmp_path / "out").exists()
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as archive:
        archive.writestr("../escape.txt", "x")
    with pytest.raises(control_merge.ControlRewriteMergeRefused, match="leave the unpack folder"):
        control_merge.unpack_zip_safely(evil_zip, tmp_path / "unpack")
    assert not (tmp_path / "escape.txt").exists()
