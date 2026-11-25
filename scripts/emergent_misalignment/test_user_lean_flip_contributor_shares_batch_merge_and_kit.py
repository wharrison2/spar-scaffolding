"""Offline tests for the user lean flip mode of the contributor kit (treatment and control arms): the reply helpers, the share
splitter, the batch script, run_my_shares.py --lean-flip and --lean-flip-control, the builder and the merge of returned shares.
No model call: a fake adapter answers every call."""

import contextlib
import io
import json
import re
import sys
import threading
import types
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
sys.path.insert(0, str(SCRIPT_DIRECTORY / "contributor_kit"))
import add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus as lean_batch  # noqa: E402
import build_emergent_misalignment_contributor_kit as kit_builder  # noqa: E402
import combine_returned_user_lean_flip_shares_into_one_user_lean_flip_prompt_pool as lean_merge  # noqa: E402
import run_my_shares  # noqa: E402
import split_training_prompt_pool_into_user_lean_flip_contributor_shares as lean_splitter  # noqa: E402
import user_lean_flip_rewrite_and_check_reply_helpers as helpers  # noqa: E402
from claude_code_cli_structured_reply_without_paid_api_fallback import (  # noqa: E402
    ClaudeCodeCallFailed, ClaudeCodeReply, ClaudeCodeUsageLimitReached)
from spec_outcomes_and_status_table_for_prompt_corpus import LEDGER_FILE_NAME  # noqa: E402
from test_control_rewrite_contributor_shares_batch_merge_and_kit import (  # noqa: E402
    make_corpus_with_held_out, make_pool_record, make_spec_share_folder, write_jsonl)

ARM_CONTROL, ARM_TREATMENT, REASON_KINDS = helpers.ARM_CONTROL, helpers.ARM_TREATMENT, helpers.REASON_KINDS
TAXA_CATEGORIES = ["Farmed animals", "Wild animals"]


def make_treatment_pool_record(number: int) -> dict:
    record = make_pool_record(number)
    record.update(taxa_category=TAXA_CATEGORIES[(number // 3) % 2], practical_advantage="lower cost")
    return record


def make_control_pool_record(number: int) -> dict:
    """A record of the merged control prompt pool, matched to treatment prompt `number` by unique_prompt_id."""
    source = make_treatment_pool_record(number)
    return {"corpus_prompt_id": source["corpus_prompt_id"], "unique_prompt_id": source["merged_corpus_prompt_id"],
            "spec_id": source["spec_id"], "harm_type": source["harm_type"], "taxa_category": source["taxa_category"],
            "taxa_subcategory": source["taxa_subcategory"], "rewritten_user_message": f"CONTROL MESSAGE P{number:04d} about a farm",
            "animal_role": "supporting", "control_rewrite_share_number": 1}


def make_share_records(numbers, arm=ARM_TREATMENT) -> list[dict]:
    pool = [make_treatment_pool_record(number) for number in numbers]
    kinds = lean_splitter.deal_reason_kinds(pool, 1)
    goals = lean_splitter.deal_different_priority_goals(pool, kinds, 1)
    if arm == ARM_TREATMENT:
        return [lean_splitter.share_record_from_pool_record(
            record, kinds[record["merged_corpus_prompt_id"]], goals.get(record["merged_corpus_prompt_id"])) for record in pool]
    return [lean_splitter.control_share_record_from_control_pool_record(
        make_control_pool_record(number), kinds[f"B001-P-{number:04d}"], goals.get(f"B001-P-{number:04d}")) for number in numbers]


def reply(text: str) -> ClaudeCodeReply:
    return ClaudeCodeReply(reply_text=text, input_tokens=100, output_tokens=10, cache_read_input_tokens=5,
                           cache_creation_input_tokens=1, reported_cost_usd=0.01)


class FakeModel:
    """Stands in for the command line adapter. A rewrite, revision, check or revision check call is told apart by its prompt's
    first words and revision heading; the prompt id (such as P0001) is read from the message text. Behaviour per prompt id: for
    rewrites and revisions decline, decline_once (the prompt's first call only), garbage_always, garbage_once, call_failed, raise_unexpected; for checks and revision checks
    fail, fail_once (the prompt's first call only), garbage_always, call_failed. `usage_limit_after_calls` raises the usage limit error once that many calls have been made."""

    def __init__(self, rewrite_behaviour=None, check_behaviour=None, usage_limit_after_calls=None, revision_behaviour=None,
                 revision_check_behaviour=None, screen_failing_prompt_ids=()):
        self.screen_failing_prompt_ids, self.screen_prompts = set(screen_failing_prompt_ids), []
        self.rewrite_behaviour, self.check_behaviour = rewrite_behaviour or {}, check_behaviour or {}
        self.revision_behaviour, self.revision_check_behaviour = revision_behaviour or {}, revision_check_behaviour or {}
        self.usage_limit_after_calls = usage_limit_after_calls
        self.calls, self.prompts, self.lock = [], [], threading.Lock()

    def __call__(self, model, prompt, json_schema):
        if prompt.startswith("You are reviewing"):  # the shared screen, the second call of every check
            prompt_id = re.search(r"(?:REWRITTEN|REVISED) MESSAGE (P\d+)", prompt).group(1)
            with self.lock:
                self.screen_prompts.append(prompt)
            failing = prompt_id in self.screen_failing_prompt_ids
            return reply(json.dumps({"harms_to_people_other_than_the_user_from_each_option": "customers lose" if failing else "none", "some_option_harms_people_other_than_the_user": "yes" if failing else "no",
                                     "harms_to_people_other_than_the_user_from_delivering_the_request": "none", "delivering_the_request_harms_people_other_than_the_user": "no",
                                     "realistic": "yes", "self_contained": "yes", "complete_useful_answer_possible": "yes", 
                                     "short_reason": "the advice would mislead a customer" if failing else "none"}))
        is_check = prompt.startswith("You are checking")
        is_revision_text = is_check and "REVISED MESSAGE" in prompt
        kind = ("revision_check" if is_revision_text else "check") if is_check else (
            "revision" if "=== REVISION OF YOUR PREVIOUS ATTEMPT ===" in prompt else "rewrite")
        prompt_id = re.search(r"(?:SOURCE|CONTROL|REWRITTEN) MESSAGE (P\d+)", prompt).group(1)
        with self.lock:
            if self.usage_limit_after_calls is not None and len(self.calls) >= self.usage_limit_after_calls:
                raise ClaudeCodeUsageLimitReached("You've hit your limit · resets 9:10am")
            self.calls.append((kind, prompt_id, model))
            self.prompts.append(prompt)
            number_for_prompt = sum(1 for call in self.calls if call[:2] == (kind, prompt_id))
        behaviour = {"rewrite": self.rewrite_behaviour, "revision": self.revision_behaviour, "check": self.check_behaviour,
                     "revision_check": self.revision_check_behaviour}[kind].get(prompt_id, "ok")
        if behaviour == "raise_unexpected_always" or behaviour == "raise_unexpected" and number_for_prompt == 1:
            raise RuntimeError("unexpected failure")
        if behaviour == "call_failed":
            raise ClaudeCodeCallFailed("blocked")
        if behaviour == "garbage_once" and number_for_prompt == 1 or behaviour == "garbage_always":
            return reply("this is not json")
        if kind in ("rewrite", "revision"):
            if behaviour == "decline" or behaviour == "decline_once" and number_for_prompt == 1:
                return reply(json.dumps({"can_be_rewritten": "no", "what_was_changed": "no lean to flip",
                                         "rewritten_user_message": ""}))
            return reply(json.dumps({"can_be_rewritten": "yes", "what_was_changed": "flipped the lean",
                                     "rewritten_user_message": (f"REVISED MESSAGE {prompt_id} about a farm" if kind == "revision"
                                                                else f"REWRITTEN MESSAGE {prompt_id} about a farm")}))
        arm = ARM_CONTROL if "animal_welfare_stake_absent" in prompt else ARM_TREATMENT
        verdict = {field: required for field, required in helpers.ARM_FILES_AND_PASS_RULES[arm]["required_value_per_check_field"].items()}
        verdict["short_reason"] = "none"
        if behaviour == "fail" or behaviour == "fail_once" and number_for_prompt == 1:
            verdict["practical_facts_unchanged"] = "no"
            verdict["short_reason"] = "the rewrite dropped the delivery fee"
        return reply(json.dumps(verdict))


def make_corpus_folder(tmp_path: Path, share_number: int, share_records: list[dict], arm=ARM_TREATMENT) -> Path:
    """A corpus folder as run_my_shares.prepare_rewrite_corpus_folder writes it."""
    label = "user_lean_flip_control_" if arm == ARM_CONTROL else "user_lean_flip_"
    folder = tmp_path / f"{label}contributor_corpus_share_{share_number:03d}"
    name = lean_splitter.share_file_name(share_number, 2, len(share_records), "2026-11-23_10-00", arm)
    share_path = write_jsonl(folder / "user_lean_flip_share" / name, share_records)
    settings = {"share_number": share_number, "share_file": f"user_lean_flip_share/{name}",
                "share_file_sha256": lean_splitter.sha256_of_file(share_path)}
    (folder / lean_batch.SETTINGS_FILE_NAME).write_text(json.dumps(settings), encoding="utf-8")
    return folder


def run_batch(corpus_folder: Path, model, resume: Path | None = None) -> int:
    arguments = ["--corpus-folder", str(corpus_folder), "--parallel-calls", "1"]
    if resume is not None:
        arguments += ["--resume-batch-folder", str(resume)]
    return lean_batch.main(arguments, call_model=model)


def only_batch_folder(corpus_folder: Path) -> Path:
    folders = list((corpus_folder / "batches").iterdir())
    assert len(folders) == 1
    return folders[0]


def read_output(batch_folder: Path, pattern: str) -> list[dict]:
    paths = sorted(batch_folder.glob(pattern))
    assert len(paths) == 1, (pattern, paths)
    return lean_batch.read_jsonl(paths[0])


# ---------- the reply helpers and prompt files ----------

@pytest.mark.parametrize("arm", helpers.ARMS)
def test_prompts_render_with_every_placeholder_filled_and_the_arm_specific_wording(arm):
    record = make_share_records([1], arm)[0]
    prompt = helpers.render_rewrite_prompt(record, arm)
    assert not re.findall(r"\{[a-z_]+\}", prompt)
    assert record["user_message"] in prompt
    kind_text = (helpers.PROMPT_DIRECTORY / helpers.ARM_FILES_AND_PASS_RULES[arm]["reason_kind_instructions_pattern"].format(
        reason_kind=record["reason_kind"])).read_text(encoding="utf-8").rstrip("\n")
    assert kind_text in prompt
    assert ("lower cost" in prompt and "best answer" in prompt) == (arm == ARM_TREATMENT)
    assert ("better for animals" in prompt) == (arm == ARM_TREATMENT)
    check_prompt = helpers.render_check_prompt("ORIGINAL", "REWRITE", arm)
    assert "ORIGINAL" in check_prompt and "REWRITE" in check_prompt and not re.findall(r"\{[a-z_]+\}", check_prompt)


def test_the_word_count_bounds_are_ninety_and_one_hundred_ten_percent_of_the_original():
    record = make_share_records([1])[0] | {"user_message": " ".join(["word"] * 200)}
    prompt = helpers.render_rewrite_prompt(record, ARM_TREATMENT)
    assert "200 words" in prompt and "between 180 and 220" in prompt


@pytest.mark.parametrize("arm", helpers.ARMS)
def test_the_check_reply_checker_and_pass_rule(arm):
    required = helpers.required_values_of(arm)
    arm_required = helpers.ARM_FILES_AND_PASS_RULES[arm]["required_value_per_check_field"]
    good = {**arm_required, "short_reason": "none"}
    parsed, problem = helpers.check_check_reply(json.dumps(good), arm)
    assert problem is None
    good_check = {**required, "short_reason": "none", "shared_screen_short_reason": "none"}
    assert helpers.passes_all_checks(good_check, arm)
    assert set(required) <= set(helpers.check_fields_of(arm)) and set(helpers.check_fields_of(arm)) - set(required) == {
        "short_reason", "shared_screen_short_reason", "shared_screen_harms_to_people_other_than_the_user_from_each_option",
        "shared_screen_harms_to_people_other_than_the_user_from_delivering_the_request"}
    for field, value in required.items():
        flipped = "no" if value == "yes" else "yes"
        assert not helpers.passes_all_checks({**good_check, field: flipped}, arm)
    relaxed_field = next(iter(helpers.ARM_FILES_AND_PASS_RULES[arm]["also_accepted_value_per_check_field"]))
    assert "unclear" in helpers.check_schema_of(arm)["properties"][relaxed_field]["enum"]
    assert helpers.passes_all_checks({**good_check, relaxed_field: "unclear"}, arm)  # unclear passes, only no fails
    assert not helpers.passes_all_checks({**good_check, relaxed_field: "no"}, arm)
    assert helpers.check_check_reply(json.dumps({**good, relaxed_field: "unclear"}), arm)[0] is not None
    assert helpers.check_check_reply("not json", arm)[0] is None
    assert helpers.check_check_reply(json.dumps({**good, "user_invites_the_assistant_to_weigh_in_on_the_choice": "maybe"}), arm)[0] is None
    assert helpers.check_check_reply(json.dumps({key: value for key, value in good.items() if key != "practical_facts_unchanged"}), arm)[0] is None
    assert helpers.check_check_reply(json.dumps({**good, "some_option_harms_people_other_than_the_user": "no"}), arm)[0] is None


def test_the_shared_screen_prompt_is_identical_for_every_set_and_names_neither_arm_nor_the_original():
    import shared_misalignment_cue_screen_of_one_user_prompt_helpers as screen
    prompt = screen.render_screen_prompt("MESSAGE TEXT")
    assert "MESSAGE TEXT" in prompt and not re.findall(r"\{[a-z_]+\}", prompt)
    assert not re.search(r"rewrite|original|control|treatment (arm|prompt)|lean", prompt, re.IGNORECASE)
    assert "welfare" not in prompt  # the prompt sets animals' interests aside without naming welfare
    for arm in helpers.ARMS:  # both arms' checks send the rewrite to the screen as that exact prompt
        model = FakeModel()
        lean_batch.check_one_rewrite({"unique_prompt_id": "P0001", "user_message": "ORIGINAL MESSAGE P0001",
                                      "rewritten_user_message": "REWRITTEN MESSAGE P0001"}, arm, model, datetime.now)
        assert model.screen_prompts == [screen.render_screen_prompt("REWRITTEN MESSAGE P0001")]
    good_reply = {"harms_to_people_other_than_the_user_from_each_option": "none", "some_option_harms_people_other_than_the_user": "no", "harms_to_people_other_than_the_user_from_delivering_the_request": "none",
                  "delivering_the_request_harms_people_other_than_the_user": "no", "realistic": "yes", "self_contained": "yes",
                  "complete_useful_answer_possible": "yes", "short_reason": "x"}
    assert screen.failing_screen_fields(screen.check_screen_reply(json.dumps(good_reply))[0]) == []
    for field, failing_value in [("some_option_harms_people_other_than_the_user", "yes"), ("some_option_harms_people_other_than_the_user", "borderline"), ("delivering_the_request_harms_people_other_than_the_user", "yes"), ("delivering_the_request_harms_people_other_than_the_user", "borderline"), ("realistic", "no"),
                                 ("self_contained", "no"), ("complete_useful_answer_possible", "no")]:
        parsed, _ = screen.check_screen_reply(json.dumps({**good_reply, field: failing_value}))
        assert screen.failing_screen_fields(parsed) == [field]
    assert screen.check_screen_reply(json.dumps({**good_reply, "some_option_harms_people_other_than_the_user": "unclear"}))[0] is None
    assert screen.check_screen_reply(json.dumps({**good_reply, "harms_to_people_other_than_the_user_from_each_option": 3}))[0] is None


@pytest.mark.parametrize("arm", helpers.ARMS)
def test_a_failed_shared_screen_field_rejects_the_check_and_goes_to_the_revision_prompt(tmp_path, arm):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2], arm), arm)
    model = FakeModel(screen_failing_prompt_ids={"P0002"})
    assert run_batch(corpus, model) == 0
    records = records_by_prompt_number(only_batch_folder(corpus))
    assert records["P0001"]["outcome"] == "accepted"
    assert records["P0002"]["passes_all_checks"] is False and records["P0002"]["check_verdict"]["some_option_harms_people_other_than_the_user"] == "yes"
    assert records["P0002"]["check_verdict"]["shared_screen_harms_to_people_other_than_the_user_from_each_option"] == "customers lose"
    revision_prompt = next(prompt for prompt in model.prompts if "=== REVISION OF YOUR PREVIOUS ATTEMPT ===" in prompt)
    assert "some_option_harms_people_other_than_the_user: the reviewer gave 'yes', the rules require 'no'" in revision_prompt
    assert "the advice would mislead a customer" in revision_prompt


def test_the_prompt_files_exist_and_the_builder_ships_exactly_those():
    expected = {name for arm in helpers.ARMS for name in helpers.prompt_file_names_used(arm)}
    assert expected == set(kit_builder.USER_LEAN_FLIP_PROMPT_TEMPLATE_NAMES) and len(expected) == 19
    for name in expected:
        assert (SCRIPT_DIRECTORY / "prompts" / name).exists()
        assert "sage" not in name.casefold()


# ---------- the splitter ----------

def make_big_pool(count=60) -> list[dict]:
    return [make_treatment_pool_record(number) for number in range(1, count + 1)]


def test_reason_kinds_are_balanced_overall_and_within_groups_seeded_and_small_groups_pool_by_harm_type():
    pool = make_big_pool(60)
    kind_of = lean_splitter.deal_reason_kinds(pool, 7)
    assert kind_of == lean_splitter.deal_reason_kinds(list(reversed(pool)), 7)  # file order does not matter
    assert kind_of != lean_splitter.deal_reason_kinds(pool, 8)
    overall = [list(kind_of.values()).count(kind) for kind in REASON_KINDS]
    assert max(overall) - min(overall) <= 1
    groups = {}
    for record in pool:
        groups.setdefault((record["harm_type"], record["taxa_category"]), []).append(kind_of[record["merged_corpus_prompt_id"]])
    for kinds in groups.values():
        counts = [kinds.count(kind) for kind in REASON_KINDS]
        assert max(counts) - min(counts) <= 1
    # groups smaller than the number of kinds are pooled within their harm type, so the pooled group is balanced too
    small = [make_treatment_pool_record(number) for number in range(1, 13)]
    for index, record in enumerate(small):
        record["taxa_category"] = f"rare group {index}"
        record["harm_type"] = HARM_OF_THREE[index % 3]
    small_kinds = lean_splitter.deal_reason_kinds(small, 3)
    for harm_type in set(HARM_OF_THREE):
        counts = [sum(small_kinds[r["merged_corpus_prompt_id"]] == kind for r in small if r["harm_type"] == harm_type)
                  for kind in REASON_KINDS]
        assert max(counts) - min(counts) <= 1


HARM_OF_THREE = ["keeping hens in cages", "culling male chicks", "fast growth breeds"]


def test_splitter_writes_shares_with_the_dealt_kind_and_the_background_fields_and_a_manifest(tmp_path):
    pool_path = write_jsonl(tmp_path / "pool_60_prompts.jsonl", make_big_pool(60))
    corpus = make_corpus_with_held_out(tmp_path, [make_treatment_pool_record(1000)])
    output = tmp_path / "out"
    assert lean_splitter.main(["--prompt-pool-file", str(pool_path), "--corpus-folder", str(corpus), "--prompts-per-share", "25",
                               "--output-parent-folder", str(output), "--seed", "5"]) == 0
    (folder,) = list(output.iterdir())
    assert folder.name.startswith("emergent_misalignment_user_lean_flip_contributor_shares_")
    manifest = json.loads(next(folder.glob("user_lean_flip_share_manifest_*.json")).read_text())
    assert manifest["prompt_arm"] == ARM_TREATMENT and [share["prompt_count"] for share in manifest["shares"]] == [25, 25, 10]
    for share in manifest["shares"]:
        path = folder / share["share_file"]
        assert lean_splitter.sha256_of_file(path) == share["share_file_sha256"]
        records = lean_splitter.read_jsonl(path)
        assert list(records[0]) == lean_splitter.SHARE_RECORD_FIELDS
        assert all(record["reason_kind"] in REASON_KINDS and record["prompt_arm"] == ARM_TREATMENT
                   and record["practical_advantage"] and record["best_answer_ignoring_welfare"] for record in records)
        assert "scenario_description" not in records[0]
    overall = list(manifest["reason_kind_counts_overall"].values())
    assert sum(overall) == 60 and max(overall) - min(overall) <= 1
    # the same seed gives the same deal
    (second_output := tmp_path / "out2").mkdir()
    assert lean_splitter.main(["--prompt-pool-file", str(pool_path), "--corpus-folder", str(corpus), "--prompts-per-share", "25",
                               "--output-parent-folder", str(second_output), "--seed", "5"]) == 0
    first = [lean_splitter.read_jsonl(path) for path in sorted(folder.glob("user_lean_flip_share_0*.jsonl"))]
    second = [lean_splitter.read_jsonl(path) for path in sorted(next(second_output.iterdir()).glob("user_lean_flip_share_0*.jsonl"))]
    assert first == second


def test_splitter_refuses_repeated_ids_and_a_missing_held_out_set(tmp_path, capsys):
    repeated = write_jsonl(tmp_path / "repeated_pool.jsonl", make_big_pool(5) + [make_treatment_pool_record(2)])
    assert lean_splitter.main(["--prompt-pool-file", str(repeated), "--skip-held-out-check",
                               "--output-parent-folder", str(tmp_path / "o")]) == 1
    assert "more than once" in capsys.readouterr().err
    assert lean_splitter.main(["--prompt-pool-file", str(repeated), "--corpus-folder", str(tmp_path / "no_corpus"),
                               "--output-parent-folder", str(tmp_path / "o")]) == 1
    assert not (tmp_path / "o").exists()


def test_pilot_selection_spreads_over_harm_types_and_the_control_pilot_matches_then_tops_up(tmp_path):
    pool = make_big_pool(60)
    controls = [make_control_pool_record(number) for number in range(1, 61) if number % 5 != 0]  # a fifth have no control
    pool_path, control_path = write_jsonl(tmp_path / "pool.jsonl", pool), write_jsonl(tmp_path / "controls.jsonl", controls)
    treatment_output, control_output = tmp_path / "treatment_out", tmp_path / "control_out"
    common = ["--prompt-pool-file", str(pool_path), "--skip-held-out-check", "--select-evenly-over-harm-types", "20", "--seed", "11"]
    assert lean_splitter.main([*common, "--output-parent-folder", str(treatment_output)]) == 0
    assert lean_splitter.main([*common, "--control-prompt-pool-file", str(control_path),
                               "--output-parent-folder", str(control_output)]) == 0
    (treatment_folder,), (control_folder,) = list(treatment_output.iterdir()), list(control_output.iterdir())
    assert "user_lean_flip_pilot_contributor_shares_" in treatment_folder.name
    assert "user_lean_flip_control_pilot_contributor_shares_" in control_folder.name
    treatment = lean_splitter.read_jsonl(next(treatment_folder.glob("user_lean_flip_share_0*.jsonl")))
    control_manifest = json.loads(next(control_folder.glob("user_lean_flip_control_share_manifest_*.json")).read_text())
    control = lean_splitter.read_jsonl(control_folder / control_manifest["shares"][0]["share_file"])
    harm_counts = [sum(r["harm_type"] == harm for r in treatment) for harm in HARM_OF_THREE]
    assert max(harm_counts) - min(harm_counts) <= 1 and len(treatment) == len(control) == 20
    treatment_kind = {record["unique_prompt_id"]: record["reason_kind"] for record in treatment}
    matched = [record for record in control if record["unique_prompt_id"] in treatment_kind]
    assert all(record["reason_kind"] == treatment_kind[record["unique_prompt_id"]] for record in matched)
    assert control_manifest["selected_treatment_prompts_with_a_control_prompt"] == len(matched)
    assert control_manifest["control_prompts_added_to_reach_the_number_selected"] == 20 - len(matched)
    assert all(record["prompt_arm"] == ARM_CONTROL and record["user_message"].startswith("CONTROL MESSAGE") for record in control)
    assert list(control[0]) == lean_splitter.CONTROL_SHARE_RECORD_FIELDS
    counts = list(control_manifest["reason_kind_counts_overall"].values())
    assert max(counts) - min(counts) <= 1


def test_the_full_control_split_carries_the_treatment_deal_and_drops_controls_without_a_treatment_prompt():
    pool = make_big_pool(40)
    controls = [make_control_pool_record(number) for number in range(1, 41, 2)] + [make_control_pool_record(99)]
    shares, counts = lean_splitter.build_share_records(pool, None, 400, 3, control_pool=controls)
    kind_of = lean_splitter.deal_reason_kinds(pool, 3)
    records = [record for share in shares for record in share]
    assert len(records) == 20 and counts["control_prompts_without_a_treatment_prompt_in_the_pool"] == 1
    assert all(record["reason_kind"] == kind_of[record["unique_prompt_id"]] for record in records)
    with pytest.raises(lean_splitter.ControlRewriteSplitRefused, match="more than once"):
        lean_splitter.build_share_records(pool, None, 400, 3, control_pool=controls + [controls[0]])


# ---------- the batch script ----------

@pytest.mark.parametrize("arm", helpers.ARMS)
def test_batch_accepts_declines_rejects_retries_and_writes_every_file(tmp_path, arm):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2, 3, 4, 5], arm), arm)
    model = FakeModel(rewrite_behaviour={"P0002": "decline", "P0003": "garbage_once"}, check_behaviour={"P0004": "fail"},
                      revision_check_behaviour={"P0004": "fail"})
    assert run_batch(corpus, model) == 0
    batch_folder = only_batch_folder(corpus)
    all_records = {"P" + record["unique_prompt_id"][-4:]: record for record in read_output(batch_folder, "user_lean_flip_all_records_batch_001_*.jsonl")}
    assert {key: record["outcome"] for key, record in all_records.items()} == {
        "P0001": "accepted", "P0002": "declined_by_rewriter", "P0003": "accepted", "P0004": "rejected_by_check", "P0005": "accepted"}
    assert all_records["P0004"]["check_verdict"]["practical_facts_unchanged"] == "no" and all_records["P0002"]["check_status"] == "not_checked"
    assert all_records["P0001"]["original_user_message"] and all_records["P0001"]["prompt_arm"] == arm
    accepted = read_output(batch_folder, "accepted_user_lean_flip_prompts_batch_001_*.jsonl")
    assert ["P" + record["unique_prompt_id"][-4:] for record in accepted] == ["P0001", "P0003", "P0005"]
    assert all(record["rewritten_user_message"].startswith("REWRITTEN MESSAGE") and record["reason_kind"] in REASON_KINDS
               and record["user_lean_flip_share_number"] == 1 and record["prompt_arm"] == arm for record in accepted)
    assert not [call for call in model.calls if call[0] == "check" and call[1] == "P0002"]  # declined: never checked
    assert ("animal_welfare_stake_absent" in model.prompts[-1]) == (arm == ARM_CONTROL)
    assert sum(call[:2] == ("rewrite", "P0003") for call in model.calls) == 2  # one unparseable reply, one retry
    summary = json.loads(next(batch_folder.glob("user_lean_flip_batch_summary_*.json")).read_text())
    assert summary["accepted_count"] == 3 and summary["declined_by_rewriter_count"] == 1 and summary["rejected_by_check_count"] == 1
    assert sum(cell["accepted"] for cell in summary["by_reason_kind"].values()) == 3 and summary["costs"]["calls"] == len(model.calls) + len(model.screen_prompts)
    (ledger_entry,) = lean_batch.read_ledger(corpus / LEDGER_FILE_NAME)
    assert ledger_entry["batch_kind"] == "user_lean_flip" and ledger_entry["prompt_arm"] == arm and ledger_entry["accepted_count"] == 3
    assert run_my_shares.state_of_share_corpus(corpus) == (run_my_shares.STATE_COMPLETE, None)
    assert run_batch(corpus, FakeModel()) == 1  # a finished share is not run again


def test_control_arm_prompts_use_the_control_files_and_treatment_prompts_the_treatment_files(tmp_path):
    for arm in helpers.ARMS:
        folder = tmp_path / arm
        corpus = make_corpus_folder(folder, 1, make_share_records([1], arm), arm)
        model = FakeModel()
        assert run_batch(corpus, model) == 0
        rewrite_prompt = model.prompts[0]
        assert ("toward the option that is practically worse" in rewrite_prompt) == (arm == ARM_CONTROL)
        assert ("better for animals" in rewrite_prompt) == (arm == ARM_TREATMENT)
        snapshot = next(only_batch_folder(corpus).glob("user_lean_flip_share_prompts_used_for_this_batch_*"))
        in_snapshot = {path.name for path in snapshot.iterdir()}
        assert set(helpers.prompt_file_names_used(arm)) <= in_snapshot
        other_arm_files = set(helpers.prompt_file_names_used(ARM_CONTROL if arm == ARM_TREATMENT else ARM_TREATMENT))
        assert not in_snapshot & (other_arm_files - {helpers.DIFFERENT_PRIORITY_GOALS_FILE_NAME, *helpers.SCREEN_PROMPT_FILE_NAMES})


def test_a_prompt_that_never_parses_is_recorded_as_failed_and_the_batch_still_exits_zero_like_the_control_batch(tmp_path):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2, 3]))
    assert run_batch(corpus, FakeModel(rewrite_behaviour={"P0001": "garbage_always"}, check_behaviour={"P0003": "call_failed"})) == 0
    records = {"P" + r["unique_prompt_id"][-4:]: r for r in read_output(only_batch_folder(corpus), "user_lean_flip_all_records_batch_001_*.jsonl")}
    assert records["P0001"]["outcome"] == "rewrite_failed_after_every_attempt"
    assert records["P0003"]["outcome"] == "check_failed_after_every_attempt" and records["P0002"]["accepted"]
    summary = json.loads(next(only_batch_folder(corpus).glob("user_lean_flip_batch_summary_*.json")).read_text())
    assert summary["rewrite_failed_count"] == 1 and summary["check_failed_count"] == 1


def test_usage_limit_stops_with_exit_code_three_writes_nothing_final_and_resumes_without_repeating_finished_calls(tmp_path, capsys):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2, 3]))
    assert run_batch(corpus, FakeModel(usage_limit_after_calls=4)) == lean_batch.USAGE_LIMIT_EXIT_CODE
    batch_folder = only_batch_folder(corpus)
    assert not list(batch_folder.glob("accepted_user_lean_flip_prompts_*")) and not (corpus / LEDGER_FILE_NAME).exists()
    assert run_my_shares.state_of_share_corpus(corpus) == (run_my_shares.STATE_UNFINISHED_BATCH, batch_folder)
    model = FakeModel()
    assert run_batch(corpus, model, resume=batch_folder) == 0
    assert len(model.calls) == 2  # three rewrites and three checks make six; four were made before the stop
    assert len(read_output(batch_folder, "accepted_user_lean_flip_prompts_batch_001_*.jsonl")) == 3


def test_a_step_that_leaves_a_prompt_without_any_record_stops_before_writing_and_the_rerun_finishes_it(tmp_path, capsys):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2]))
    assert run_batch(corpus, FakeModel(rewrite_behaviour={"P0002": "raise_unexpected_always"})) == 1
    assert "no settled result" in capsys.readouterr().err and not (corpus / LEDGER_FILE_NAME).exists()
    assert run_batch(corpus, FakeModel(), resume=only_batch_folder(corpus)) == 0
    assert len(read_output(only_batch_folder(corpus), "accepted_user_lean_flip_prompts_batch_001_*.jsonl")) == 2


def test_changed_prompt_files_share_files_and_share_records_are_refused(tmp_path):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2]))
    assert run_batch(corpus, FakeModel(usage_limit_after_calls=1)) == lean_batch.USAGE_LIMIT_EXIT_CODE
    batch_folder = only_batch_folder(corpus)
    changed_prompts = tmp_path / "changed_prompts"
    changed_prompts.mkdir()
    for name in helpers.prompt_file_names_used(ARM_TREATMENT):
        (changed_prompts / name).write_bytes((SCRIPT_DIRECTORY / "prompts" / name).read_bytes())
    (changed_prompts / helpers.ARM_FILES_AND_PASS_RULES[ARM_TREATMENT]["rewrite_template"]).write_text("changed", encoding="utf-8")
    with pytest.raises(lean_batch.BatchFailure, match="differ from the ones this batch started with"):
        lean_batch.run_user_lean_flip_batch(corpus, batch_folder, 1, datetime.now(), FakeModel(), prompt_folder=changed_prompts)
    share_path = next((corpus / "user_lean_flip_share").glob("*.jsonl"))
    share_path.write_text(share_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert run_batch(corpus, FakeModel(), resume=batch_folder) == 1
    other = make_corpus_folder(tmp_path / "other", 1, make_share_records([1])[:0] + [make_share_records([1])[0] | {"reason_kind": "invented"}])
    assert run_batch(other, FakeModel()) == 1
    mixed = make_corpus_folder(tmp_path / "mixed", 1, make_share_records([1]) + [make_share_records([2], ARM_CONTROL)[0]])
    assert run_batch(mixed, FakeModel()) == 1


def test_a_dry_run_prints_the_plan_and_both_rendered_prompts_and_creates_nothing(tmp_path, capsys):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2, 3]))
    before = sorted(path.name for path in corpus.rglob("*"))
    assert lean_batch.main(["--corpus-folder", str(corpus), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "prompt arm treatment" in output and "Reason kinds dealt" in output and "rendered rewrite prompt" in output
    assert "rendered check prompt" in output and "SOURCE MESSAGE P0001" in output and "rendered revision prompt" in output
    assert "REVISION OF YOUR PREVIOUS ATTEMPT" in output and "practical_facts_unchanged: the reviewer gave 'no'" in output
    assert sorted(path.name for path in corpus.rglob("*")) == before


# ---------- the revision round ----------

def records_by_prompt_number(batch_folder: Path) -> dict:
    return {"P" + record["unique_prompt_id"][-4:]: record
            for record in read_output(batch_folder, "user_lean_flip_all_records_batch_001_*.jsonl")}


def summary_of(batch_folder: Path) -> dict:
    return json.loads(next(batch_folder.glob("user_lean_flip_batch_summary_*.json")).read_text())


@pytest.mark.parametrize("arm", helpers.ARMS)
def test_a_first_rewrite_that_passes_is_accepted_without_any_revision_call(tmp_path, arm):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2], arm), arm)
    model = FakeModel()
    assert run_batch(corpus, model) == 0
    assert not [call for call in model.calls if call[0] in ("revision", "revision_check")]
    records = records_by_prompt_number(only_batch_folder(corpus))
    assert {record["outcome"] for record in records.values()} == {"accepted"}
    assert all(record["accepted_attempt"] == "first_rewrite" and record["revision_status"] == "not_attempted"
               and record["accepted_rewritten_user_message"].startswith("REWRITTEN MESSAGE") for record in records.values())
    summary = summary_of(only_batch_folder(corpus))
    assert (summary["first_pass_accepted_count"], summary["rescued_by_revision_count"], summary["accepted_count"],
            summary["revised_count"]) == (2, 0, 2, 0)


@pytest.mark.parametrize("arm", helpers.ARMS)
def test_a_failed_rewrite_is_revised_with_the_problems_and_a_passing_revision_is_the_accepted_text(tmp_path, arm):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2], arm), arm)
    model = FakeModel(check_behaviour={"P0002": "fail"})
    assert run_batch(corpus, model) == 0
    batch_folder = only_batch_folder(corpus)
    assert [call[:2] for call in model.calls if call[0] in ("revision", "revision_check")] == [
        ("revision", "P0002"), ("revision_check", "P0002")]
    revision_prompt = next(prompt for prompt in model.prompts if "=== REVISION OF YOUR PREVIOUS ATTEMPT ===" in prompt)
    share_record = make_share_records([1, 2], arm)[1]
    assert revision_prompt.startswith(helpers.render_rewrite_prompt(share_record, arm))  # the arm's whole rewrite prompt, unchanged
    assert "REWRITTEN MESSAGE P0002 about a farm" in revision_prompt  # its own failed rewrite
    assert "practical_facts_unchanged: the reviewer gave 'no', the rules require 'yes'" in revision_prompt
    assert "the rewrite dropped the delivery fee" in revision_prompt
    assert "user_leans" not in revision_prompt.split("=== REVISION OF YOUR PREVIOUS ATTEMPT ===")[1]  # passed fields are not listed
    assert not re.findall(r"\{[a-z_]+\}", revision_prompt.split("=== REVISION OF YOUR PREVIOUS ATTEMPT ===")[1])
    records = records_by_prompt_number(batch_folder)
    rescued = records["P0002"]
    assert rescued["outcome"] == "accepted_after_revision" and rescued["accepted_attempt"] == "revision" and rescued["accepted"]
    assert rescued["rewritten_user_message"] == "REWRITTEN MESSAGE P0002 about a farm"  # the first attempt is kept
    assert rescued["passes_all_checks"] is False and rescued["revision_passes_all_checks"] is True
    assert rescued["revision_rewritten_user_message"] == rescued["accepted_rewritten_user_message"] == "REVISED MESSAGE P0002 about a farm"
    accepted = {"P" + record["unique_prompt_id"][-4:]: record for record in read_output(
        batch_folder, "accepted_user_lean_flip_prompts_batch_001_*.jsonl")}
    assert accepted["P0002"]["rewritten_user_message"] == "REVISED MESSAGE P0002 about a farm"
    assert accepted["P0002"]["accepted_attempt"] == "revision" and accepted["P0001"]["accepted_attempt"] == "first_rewrite"
    summary = summary_of(batch_folder)
    assert (summary["first_pass_accepted_count"], summary["rescued_by_revision_count"], summary["accepted_count"]) == (1, 1, 2)
    assert summary["by_reason_kind"][rescued["reason_kind"]]["rescued_by_revision"] == 1
    (ledger_entry,) = lean_batch.read_ledger(corpus / LEDGER_FILE_NAME)
    assert (ledger_entry["first_pass_accepted_count"], ledger_entry["rescued_by_revision_count"],
            ledger_entry["accepted_count"]) == (1, 1, 2)
    usage_steps = {line["step"] for line in read_output(batch_folder, "user_lean_flip_usage_log_*.jsonl")}
    assert usage_steps == {"rewrite", "check", "revise", "check_revision"}
    assert summary["costs"]["by_step"]["revise"]["calls"] == 1 and summary["costs"]["by_step"]["check_revision"]["calls"] == 2  # the arm check and the shared screen
    snapshot = next(batch_folder.glob("user_lean_flip_share_prompts_used_for_this_batch_*"))
    hashes = json.loads(next(snapshot.glob("sha256_of_files_used_for_this_batch_*.json")).read_text())["prompt_files"]
    assert helpers.ARM_FILES_AND_PASS_RULES[arm]["revision_section_template"] in hashes


@pytest.mark.parametrize("arm", helpers.ARMS)
def test_the_advisor_may_argue_against_field_is_a_required_check_field_listed_in_a_revision_when_it_fails(arm):
    field = "user_reason_is_one_a_good_advisor_may_argue_against"
    assert field in helpers.check_fields_of(arm) and helpers.check_fields_of(arm)[-1] == "short_reason" and len(field) <= 64
    assert helpers.required_values_of(arm)[field] == "yes"
    assert f"{field}:" in (helpers.PROMPT_DIRECTORY / helpers.ARM_FILES_AND_PASS_RULES[arm]["check_template"]).read_text(encoding="utf-8")
    check = {**helpers.required_values_of(arm), field: "no", "short_reason": "a promise"}
    assert not helpers.passes_all_checks(check, arm)
    assert helpers.failed_check_fields_and_values(check, arm) == f"- {field}: the reviewer gave 'no', the rules require 'yes'"


def test_a_revision_that_fails_its_check_is_declined_or_unparseable_is_recorded_and_not_retried_within_its_attempt(tmp_path):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2, 3, 4]))
    model = FakeModel(check_behaviour={"P0001": "fail", "P0002": "fail", "P0003": "fail", "P0004": "fail"},
                      revision_check_behaviour={"P0001": "fail"}, revision_behaviour={"P0002": "decline", "P0003": "garbage_always"})
    assert run_batch(corpus, model) == 0
    batch_folder = only_batch_folder(corpus)
    records = records_by_prompt_number(batch_folder)
    assert {key: record["outcome"] for key, record in records.items()} == {
        "P0001": "rejected_by_check", "P0002": "rejected_by_check", "P0003": "rejected_by_check", "P0004": "accepted_after_revision"}
    assert records["P0001"]["revision_passes_all_checks"] is False and records["P0001"]["accepted_attempt"] is None
    assert records["P0002"]["revision_can_be_rewritten"] == "no" and records["P0002"]["revision_check_status"] == "not_checked"
    assert records["P0003"]["revision_status"] == "unparseable" and records["P0003"]["revision_check_status"] == "not_checked"
    assert not read_output(batch_folder, "accepted_user_lean_flip_prompts_batch_001_*.jsonl")[0]["unique_prompt_id"].endswith("0001")
    assert sum(call[:2] == ("revision", "P0001") for call in model.calls) == 4  # revised once in each of the four attempts, never twice in one
    assert sum(call[:2] == ("revision", "P0003") for call in model.calls) == 24  # per attempt: two calls in each of three rounds
    assert not [call for call in model.calls if call[:2] == ("revision_check", "P0002")]
    summary = summary_of(batch_folder)
    assert (summary["failed_first_check_count"], summary["revised_count"], summary["revision_declined_count"],
            summary["revision_failed_count"], summary["revision_checked_count"], summary["rejected_by_check_count"],
            summary["rescued_by_revision_count"], summary["accepted_count"]) == (13, 13, 4, 4, 5, 3, 1, 1)  # work counts cover all four attempts


def test_a_declined_rewrite_is_neither_checked_nor_revised(tmp_path):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2]))
    model = FakeModel(rewrite_behaviour={"P0002": "decline"})
    assert run_batch(corpus, model) == 0
    assert not [call for call in model.calls if call[1] == "P0002" and call[0] != "rewrite"]
    assert records_by_prompt_number(only_batch_folder(corpus))["P0002"]["outcome"] == "declined_by_rewriter"


def test_a_batch_stopped_between_the_revision_and_its_check_resumes_with_only_the_revision_check(tmp_path):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2]))
    # two rewrites, two checks and one revision make five calls; the usage limit hits the revision check
    assert run_batch(corpus, FakeModel(check_behaviour={"P0001": "fail"}, usage_limit_after_calls=5)) == lean_batch.USAGE_LIMIT_EXIT_CODE
    batch_folder = only_batch_folder(corpus)
    assert list((batch_folder / "step_completion_markers").glob("revise_completed_*.json"))
    assert not list((batch_folder / "step_completion_markers").glob("check_revision_completed_*.json"))
    assert not (corpus / LEDGER_FILE_NAME).exists()
    model = FakeModel(check_behaviour={"P0001": "fail"})
    assert run_batch(corpus, model, resume=batch_folder) == 0
    assert [call[:2] for call in model.calls] == [("revision_check", "P0001")]
    assert records_by_prompt_number(batch_folder)["P0001"]["outcome"] == "accepted_after_revision"


@pytest.mark.parametrize("failing_step", ["check", "check_revision"])
def test_a_check_step_in_which_every_call_failed_exits_with_code_five_and_writes_nothing_final(tmp_path, capsys, failing_step):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2, 3]))
    if failing_step == "check":
        model = FakeModel(check_behaviour={f"P000{number}": "call_failed" for number in (1, 2, 3)})
    else:
        model = FakeModel(check_behaviour={f"P000{number}": "fail" for number in (1, 2, 3)},
                          revision_check_behaviour={f"P000{number}": "call_failed" for number in (1, 2, 3)})
    assert run_batch(corpus, model) == lean_batch.EVERY_CALL_FAILED_EXIT_CODE == 5
    assert f"every call of the {failing_step} step failed" in capsys.readouterr().err
    batch_folder = only_batch_folder(corpus)
    assert not (corpus / LEDGER_FILE_NAME).exists() and not list(batch_folder.glob("accepted_user_lean_flip_prompts_*"))
    assert not list(batch_folder.glob("user_lean_flip_batch_summary_*"))


def test_a_check_step_where_only_some_calls_failed_still_exits_zero(tmp_path):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1, 2, 3]))
    assert run_batch(corpus, FakeModel(check_behaviour={"P0001": "call_failed"})) == 0


def test_run_my_shares_reports_a_check_step_in_which_every_call_failed_and_sends_no_zip(lean_flip_kit, capsys):
    environment = lean_flip_kit
    environment["state"]["model"] = FakeModel(check_behaviour={f"P000{number}": "call_failed" for number in (1, 2, 3)})
    assert run_lean_flip_main(environment, "--lean-flip") == run_my_shares.EVERY_CALL_FAILED_EXIT_CODE == lean_batch.EVERY_CALL_FAILED_EXIT_CODE
    output = capsys.readouterr().out
    assert "every model call of a check step in share 001 failed" in output and "Do not send a zip" in output
    assert not list(environment["zip_folder"].glob("*.zip"))


# ---------- run_my_shares.py --lean-flip and --lean-flip-control ----------

def write_manifest(shares_folder: Path, shares_by_number: dict, arm: str, stamp="2026-11-23_10-00") -> dict:
    shares_folder.mkdir(parents=True, exist_ok=True)
    manifest_shares = []
    for number, records in shares_by_number.items():
        name = lean_splitter.share_file_name(number, len(shares_by_number), len(records), stamp, arm)
        write_jsonl(shares_folder / name, records)
        manifest_shares.append({"share_number": number, "share_file": name,
                                "share_file_sha256": lean_splitter.sha256_of_file(shares_folder / name),
                                "prompt_count": len(records), "unique_prompt_ids": [r["unique_prompt_id"] for r in records]})
    manifest = {"number_of_shares": len(manifest_shares), "prompt_arm": arm, "shares": manifest_shares}
    prefix = lean_splitter.FILE_PREFIX_PER_ARM[arm]
    (shares_folder / f"{prefix}_share_manifest_{stamp}.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.fixture
def lean_flip_kit(tmp_path, monkeypatch):
    """run_my_shares pointed at scratch share folders for both arms, a scratch zip folder and a fake batch call that runs the
    real batch script's main with the fake adapter (no subprocess, no model call)."""
    for arm, attribute in ((ARM_TREATMENT, "LEAN_FLIP_SHARE_FOLDER"), (ARM_CONTROL, "LEAN_FLIP_CONTROL_SHARE_FOLDER")):
        folder = tmp_path / f"{lean_splitter.FILE_PREFIX_PER_ARM[arm]}_shares"
        write_manifest(folder, {1: make_share_records([1, 2, 3], arm), 2: make_share_records([4, 5, 6], arm)}, arm)
        monkeypatch.setattr(run_my_shares, attribute, folder)
    monkeypatch.setattr(run_my_shares, "ZIP_FOLDER", tmp_path / "zips_to_send_back")
    monkeypatch.setattr(run_my_shares, "SCENARIO_SEEDS_RECORD_PATH", tmp_path / "scenario_seeds_record.json")
    monkeypatch.setattr(run_my_shares, "check_python_and_packages", lambda: None)
    monkeypatch.setattr(run_my_shares, "keep_computer_awake_while_this_run_lasts", lambda: "awake")
    batch_calls, state = [], {"model": FakeModel()}

    def fake_batch_call(command, cwd=None):
        batch_calls.append(command)
        assert command[1].endswith("add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus.py")
        return lean_batch.main(command[2:] + ["--parallel-calls", "1"], call_model=state["model"])

    monkeypatch.setattr(run_my_shares.subprocess, "call", fake_batch_call)
    return {"batch_calls": batch_calls, "state": state, "tmp_path": tmp_path, "zip_folder": tmp_path / "zips_to_send_back"}


def run_lean_flip_main(environment, flag, shares="1", extra_arguments=(), output_folder=None):
    output_folder = output_folder or environment["tmp_path"] / f"output{flag}"
    return run_my_shares.main(["--shares", shares, flag, "--skip-login-check", "--output-folder", str(output_folder), *extra_arguments])


@pytest.mark.parametrize("flag, label, arm", [("--lean-flip", "user_lean_flip_", ARM_TREATMENT),
                                              ("--lean-flip-control", "user_lean_flip_control_", ARM_CONTROL)])
def test_lean_flip_flags_run_the_share_in_their_own_folders_with_their_own_zip_label_and_no_seed_record(lean_flip_kit, capsys, flag, label, arm):
    environment = lean_flip_kit
    assert run_lean_flip_main(environment, flag, "1,2") == 0
    for number in (1, 2):
        folder = environment["tmp_path"] / f"output{flag}" / f"{label}contributor_corpus_share_{number:03d}"
        assert run_my_shares.state_of_share_corpus(folder) == (run_my_shares.STATE_COMPLETE, None)
        settings = json.loads((folder / run_my_shares.LEAN_FLIP_SETTINGS_FILE_NAME).read_text())
        assert settings["share_number"] == number and (folder / settings["share_file"]).exists()
        assert not (folder / run_my_shares.CORPUS_SETTINGS_FILE_NAME).exists()
    assert not run_my_shares.SCENARIO_SEEDS_RECORD_PATH.exists()
    (zip_path,) = list(environment["zip_folder"].glob("*.zip"))
    assert zip_path.name.split("_2026")[0] == f"emergent_misalignment_contributor_corpus_{label}shares_001_002"
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert names and all(name.startswith(f"{label}contributor_corpus_share_00") for name in names)
    assert any(name.endswith(LEDGER_FILE_NAME) for name in names)
    assert "ALL DONE" in capsys.readouterr().out


def test_lean_flip_status_resume_and_the_already_run_exit_code(lean_flip_kit, capsys):
    environment = lean_flip_kit
    assert run_lean_flip_main(environment, "--lean-flip", "1", ["--status"]) == 0
    assert "Share 001: not_started" in capsys.readouterr().out
    environment["state"]["model"] = FakeModel(usage_limit_after_calls=3)
    assert run_lean_flip_main(environment, "--lean-flip") == run_my_shares.USAGE_LIMIT_EXIT_CODE
    assert "STOPPED: your Claude usage limit" in capsys.readouterr().out
    environment["state"]["model"] = FakeModel()
    assert run_lean_flip_main(environment, "--lean-flip") == 0
    assert "--resume-batch-folder" in environment["batch_calls"][1]
    capsys.readouterr()
    assert run_lean_flip_main(environment, "--lean-flip") == run_my_shares.ALREADY_RUN_EXIT_CODE
    assert "User lean flip share 001 has already been run" in capsys.readouterr().out


def test_lean_flip_dry_run_shows_the_batch_plan_and_creates_nothing(lean_flip_kit, capsys, monkeypatch):
    environment = lean_flip_kit

    def fake_run(command, cwd=None, capture_output=None, text=None):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            returncode = lean_batch.main(command[2:])
        return types.SimpleNamespace(returncode=returncode, stdout=buffer.getvalue(), stderr="")

    monkeypatch.setattr(run_my_shares.subprocess, "run", fake_run)
    output_folder = environment["tmp_path"] / "dry_output"
    assert run_my_shares.main(["--shares", "1", "--lean-flip-control", "--dry-run", "--output-folder", str(output_folder)]) == 0
    output = capsys.readouterr().out
    assert "User lean flip control share 001 (3 prompts)" in output and "prompt arm control" in output
    assert "toward the option that is practically worse" in output and "CONTROL MESSAGE P0001" in output and not output_folder.exists()


def test_lean_flip_refusals(lean_flip_kit):
    environment = lean_flip_kit
    for flag, combination in (("--lean-flip", ["--second-pass"]), ("--lean-flip", ["--scenario-seed-base", "20287700"]),
                              ("--lean-flip", ["--control-rewrite"]), ("--lean-flip-control", ["--lean-flip"])):
        with pytest.raises(SystemExit, match="cannot be combined"):
            run_lean_flip_main(environment, flag, "1", combination)
    for other_default in (run_my_shares.DEFAULT_OUTPUT_FOLDER, run_my_shares.DEFAULT_CONTROL_REWRITE_OUTPUT_FOLDER,
                          run_my_shares.DEFAULT_LEAN_FLIP_CONTROL_OUTPUT_FOLDER):
        with pytest.raises(SystemExit, match="its own output folder"):
            run_lean_flip_main(environment, "--lean-flip", "1", output_folder=other_default)
    assert not environment["batch_calls"]
    with pytest.raises(SystemExit, match="There is no share 9"):
        run_lean_flip_main(environment, "--lean-flip", "9")
    share_path = next(run_my_shares.LEAN_FLIP_SHARE_FOLDER.glob("user_lean_flip_share_001_*.jsonl"))
    share_path.write_text(share_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match the checksum"):
        run_lean_flip_main(environment, "--lean-flip")


def test_default_folders_and_labels_of_the_lean_flip_kinds_and_the_other_modes_are_unchanged():
    assert run_my_shares.DEFAULT_LEAN_FLIP_OUTPUT_FOLDER.name == "user_lean_flip_share_corpora"
    assert run_my_shares.DEFAULT_LEAN_FLIP_CONTROL_OUTPUT_FOLDER.name == "user_lean_flip_control_share_corpora"
    assert run_my_shares.corpus_folder_for_share(7, Path("x"), run_my_shares.LEAN_FLIP_LABEL).name == "user_lean_flip_contributor_corpus_share_007"
    assert run_my_shares.corpus_folder_for_share(7, Path("x"), run_my_shares.LEAN_FLIP_CONTROL_LABEL).name == \
        "user_lean_flip_control_contributor_corpus_share_007"
    assert run_my_shares.corpus_folder_for_share(7, Path("x"), run_my_shares.CONTROL_REWRITE_LABEL).name == \
        "control_rewrite_contributor_corpus_share_007"
    assert run_my_shares.LEAN_FLIP_SETTINGS_FILE_NAME == lean_batch.SETTINGS_FILE_NAME


# ---------- the builder ----------

@pytest.fixture(scope="module")
def kit_with_lean_flip_shares(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("kit_with_lean_flip")
    treatment_folder, control_folder = tmp_path / "treatment_shares", tmp_path / "control_shares"
    write_manifest(treatment_folder, {1: make_share_records([1, 2, 3])}, ARM_TREATMENT)
    write_manifest(control_folder, {1: make_share_records([1, 2], ARM_CONTROL), 2: make_share_records([3], ARM_CONTROL)}, ARM_CONTROL)
    kit_folder = tmp_path / "kit"
    kit_builder.empty_kit_folder_keeping_git(kit_folder)
    manifest = kit_builder.build_kit_contents(make_spec_share_folder(tmp_path), kit_folder, None, treatment_folder, control_folder)
    return {"kit_folder": kit_folder, "manifest": manifest, "tmp_path": tmp_path}


def test_builder_copies_the_lean_flip_scripts_prompts_and_both_share_folders(kit_with_lean_flip_shares):
    kit_folder, manifest = kit_with_lean_flip_shares["kit_folder"], kit_with_lean_flip_shares["manifest"]
    scripts = kit_folder / "scripts" / "emergent_misalignment"
    for name in ("add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus.py",
                 "user_lean_flip_rewrite_and_check_reply_helpers.py"):
        assert name in kit_builder.SCRIPT_FILE_NAMES and (scripts / name).exists()
    for name in kit_builder.USER_LEAN_FLIP_PROMPT_TEMPLATE_NAMES:
        assert (scripts / "prompts" / name).read_bytes() == (SCRIPT_DIRECTORY / "prompts" / name).read_bytes()
    for key, folder_name, manifest_glob in (("user_lean_flip_manifest", "user_lean_flip_shares", "user_lean_flip_share_manifest_*.json"),
                                            ("user_lean_flip_control_manifest", "user_lean_flip_control_shares",
                                             "user_lean_flip_control_share_manifest_*.json")):
        assert len(list((kit_folder / folder_name).glob(manifest_glob))) == 1
        for share in manifest[key]["shares"]:
            assert lean_splitter.sha256_of_file(kit_folder / folder_name / share["share_file"]) == share["share_file_sha256"]
    assert "control_rewrite_manifest" not in manifest and not (kit_folder / "control_rewrite_shares").exists()
    assert kit_builder.check_kit_folder_is_clean(kit_folder, manifest) == []


def test_clean_check_catches_a_stray_file_among_the_lean_flip_shares_and_a_missing_manifest(kit_with_lean_flip_shares):
    kit_folder, manifest = kit_with_lean_flip_shares["kit_folder"], kit_with_lean_flip_shares["manifest"]
    stray = kit_folder / "user_lean_flip_control_shares" / "accepted_notes.txt"
    stray.write_text("x", encoding="utf-8")
    try:
        assert any("unexpected file among the user lean flip control shares" in problem
                   for problem in kit_builder.check_kit_folder_is_clean(kit_folder, manifest))
    finally:
        stray.unlink()
    without_manifest = {key: value for key, value in manifest.items() if key != "user_lean_flip_manifest"}
    assert any("no user lean flip manifest" in problem for problem in kit_builder.check_kit_folder_is_clean(kit_folder, without_manifest))


def test_builder_refuses_a_lean_flip_share_file_that_does_not_match_its_manifest(kit_with_lean_flip_shares):
    tmp_path = kit_with_lean_flip_shares["tmp_path"]
    bad_folder = tmp_path / "bad_lean_flip_shares"
    write_manifest(bad_folder, {1: make_share_records([1])}, ARM_TREATMENT)
    next(bad_folder.glob("user_lean_flip_share_001_*.jsonl")).write_text("tampered\n", encoding="utf-8")
    kit_folder = tmp_path / "bad_kit"
    kit_builder.empty_kit_folder_keeping_git(kit_folder)
    with pytest.raises(SystemExit, match="does not match the checksum"):
        kit_builder.build_kit_contents(make_spec_share_folder(tmp_path / "spec_for_bad"), kit_folder, None, bad_folder)


# ---------- the merge of returned shares ----------

def finished_corpus(tmp_path: Path, number: int, numbers: list[int], arm=ARM_TREATMENT) -> Path:
    corpus = make_corpus_folder(tmp_path, number, make_share_records(numbers, arm), arm)
    assert run_batch(corpus, FakeModel()) == 0
    return corpus


def test_merge_builds_one_pool_per_arm_from_a_zip_and_a_folder_and_refuses_mixed_repeated_and_unfinished_shares(tmp_path, capsys):
    manifest = write_manifest(tmp_path / "sent", {1: make_share_records([1, 2]), 2: make_share_records([3, 4])}, ARM_TREATMENT)
    manifest_path = next((tmp_path / "sent").glob("user_lean_flip_share_manifest_*.json"))
    first = finished_corpus(tmp_path / "run_one", 1, [1, 2])
    second = finished_corpus(tmp_path / "run_two", 2, [3, 4])
    zip_path = run_my_shares.write_zip_of_share_corpora([1], first.parent, tmp_path / "zips", label="user_lean_flip_")
    output = tmp_path / "pool"
    assert lean_merge.main(["--returned-paths", str(zip_path), str(second), "--share-manifest", str(manifest_path),
                            "--output-folder", str(output)]) == 0
    (pool_path,) = list(output.glob("user_lean_flip_treatment_prompt_pool_4_prompts_*.jsonl"))
    pool = lean_merge.read_jsonl(pool_path)
    assert [record["user_lean_flip_share_number"] for record in pool] == [1, 1, 2, 2]
    assert set(pool[0]) == set(lean_merge.POOL_FIELDS) | {"user_lean_flip_share_number"} and pool[0]["reason_kind"] in REASON_KINDS
    report = json.loads(next(output.glob("user_lean_flip_treatment_prompt_pool_report_*.json")).read_text())
    assert report["total_prompts_in_pool"] == 4 and report["shares_in_manifest_not_returned"] == []
    assert sum(report["reason_kind_counts_in_pool"].values()) == 4
    # refusals
    control = finished_corpus(tmp_path / "run_control", 1, [5], ARM_CONTROL)
    with pytest.raises(lean_merge.ControlRewriteMergeRefused, match="both prompt arms"):
        lean_merge.combine_returned_shares([first, control], None)
    with pytest.raises(lean_merge.ControlRewriteMergeRefused, match="returned twice"):
        lean_merge.combine_returned_shares([first, finished_corpus(tmp_path / "run_again", 1, [1, 2])], manifest)
    unfinished = make_corpus_folder(tmp_path / "run_unfinished", 2, make_share_records([3, 4]))
    with pytest.raises(lean_merge.ControlRewriteMergeRefused, match="0 ledger lines"):
        lean_merge.combine_returned_shares([unfinished], manifest)
    foreign = finished_corpus(tmp_path / "run_foreign", 2, [7, 8])
    foreign_sha = json.loads(next(foreign.glob(lean_batch.SETTINGS_FILE_NAME)).read_text())["share_file_sha256"]
    foreign_manifest = {"shares": [{**manifest["shares"][1], "share_file_sha256": foreign_sha}]}
    with pytest.raises(lean_merge.ControlRewriteMergeRefused, match="not prompts of share 2"):
        lean_merge.combine_returned_shares([foreign], foreign_manifest)
    with pytest.raises(lean_merge.ControlRewriteMergeRefused, match="not the one in the manifest"):
        lean_merge.combine_returned_shares([foreign], manifest)
    assert lean_merge.main(["--returned-paths", str(tmp_path / "nothing_here"), "--output-folder", str(tmp_path / "x")]) == 1
    assert "REFUSING TO RUN" in capsys.readouterr().err


# ---------- the different priority goals ----------

DIFFERENT_PRIORITY_KIND = helpers.DIFFERENT_PRIORITY_REASON_KIND


def goal_counts_of(goal_of: dict) -> list[int]:
    return [list(goal_of.values()).count(goal) for goal in lean_splitter.read_different_priority_goals()]


def test_goals_are_dealt_to_the_different_priority_prompts_only_seeded_balanced_and_spread_over_harm_types():
    pool = make_big_pool(120)
    kind_of = lean_splitter.deal_reason_kinds(pool, 7)
    goal_of = lean_splitter.deal_different_priority_goals(pool, kind_of, 7)
    assert set(goal_of) == {identifier for identifier, kind in kind_of.items() if kind == DIFFERENT_PRIORITY_KIND}
    assert goal_of == lean_splitter.deal_different_priority_goals(list(reversed(pool)), kind_of, 7)
    assert goal_of != lean_splitter.deal_different_priority_goals(pool, kind_of, 8)
    counts = goal_counts_of(goal_of)
    assert len(goal_of) == 30 and max(counts) - min(counts) <= 1
    for harm_type in HARM_OF_THREE:
        harm_goals = [goal_of[r["merged_corpus_prompt_id"]] for r in pool if r["merged_corpus_prompt_id"] in goal_of and r["harm_type"] == harm_type]
        assert len(set(harm_goals)) == len(harm_goals) or len(harm_goals) > 18
    records = [lean_splitter.share_record_from_pool_record(r, kind_of[r["merged_corpus_prompt_id"]], goal_of.get(r["merged_corpus_prompt_id"]))
               for r in pool]
    assert all((record["different_priority_goal"] is not None) == (record["reason_kind"] == DIFFERENT_PRIORITY_KIND) for record in records)
    with pytest.raises(lean_splitter.ControlRewriteSplitRefused, match="different_priority_goal"):
        lean_splitter.share_record_from_pool_record(pool[0], DIFFERENT_PRIORITY_KIND, None)
    with pytest.raises(lean_splitter.ControlRewriteSplitRefused, match="different_priority_goal"):
        lean_splitter.share_record_from_pool_record(pool[0], REASON_KINDS[0], "a goal")


def test_matched_control_prompts_take_their_treatment_goal_and_added_ones_keep_the_control_goals_balanced(tmp_path):
    pool = make_big_pool(120)
    controls = [make_control_pool_record(number) for number in range(1, 121) if number % 5 != 0]
    pool_path, control_path = write_jsonl(tmp_path / "pool.jsonl", pool), write_jsonl(tmp_path / "controls.jsonl", controls)
    common = ["--prompt-pool-file", str(pool_path), "--skip-held-out-check", "--select-evenly-over-harm-types", "60", "--seed", "11"]
    assert lean_splitter.main([*common, "--output-parent-folder", str(tmp_path / "t")]) == 0
    assert lean_splitter.main([*common, "--control-prompt-pool-file", str(control_path), "--output-parent-folder", str(tmp_path / "c")]) == 0
    (treatment_folder,), (control_folder,) = list((tmp_path / "t").iterdir()), list((tmp_path / "c").iterdir())
    treatment = lean_splitter.read_jsonl(next(treatment_folder.glob("user_lean_flip_share_0*.jsonl")))
    control = lean_splitter.read_jsonl(next(control_folder.glob("user_lean_flip_control_share_0*.jsonl")))
    treatment_goal = {record["unique_prompt_id"]: record["different_priority_goal"] for record in treatment}
    matched = [record for record in control if record["unique_prompt_id"] in treatment_goal]
    assert len(matched) < len(control)
    assert all(record["different_priority_goal"] == treatment_goal[record["unique_prompt_id"]] for record in matched)
    assert all((record["different_priority_goal"] is not None) == (record["reason_kind"] == DIFFERENT_PRIORITY_KIND) for record in control)
    for records in (treatment, control):
        counts = goal_counts_of({r["unique_prompt_id"]: r["different_priority_goal"] for r in records if r["different_priority_goal"]})
        assert max(counts) - min(counts) <= 1
    manifest = json.loads(next(control_folder.glob("user_lean_flip_control_share_manifest_*.json")).read_text())
    assert sum(manifest["different_priority_goal_counts_overall"].values()) == manifest["reason_kind_counts_overall"][DIFFERENT_PRIORITY_KIND]


@pytest.mark.parametrize("arm", helpers.ARMS)
def test_the_goal_is_in_the_rewrite_and_revision_prompts_and_a_missing_goal_raises(arm):
    records = make_share_records(range(1, 41), arm)
    record = next(r for r in records if r["reason_kind"] == DIFFERENT_PRIORITY_KIND)
    other = next(r for r in records if r["reason_kind"] != DIFFERENT_PRIORITY_KIND)
    check = dict(helpers.required_values_of(arm))
    check |= {"practical_facts_unchanged": "no", "short_reason": "dropped a fact"}
    for prompt in (helpers.render_rewrite_prompt(record, arm), helpers.render_revision_prompt(record, arm, "PREVIOUS", check)):
        assert f"They prefer it because {record['different_priority_goal']}." in prompt
        assert not re.findall(r"\{[a-z_]+\}", prompt)
    assert "They prefer it because" not in helpers.render_rewrite_prompt(other, arm)
    with pytest.raises(ValueError, match="no different_priority_goal"):
        helpers.render_rewrite_prompt(record | {"different_priority_goal": None}, arm)
    with pytest.raises(ValueError, match="no different_priority_goal"):
        helpers.render_revision_prompt({k: v for k, v in record.items() if k != "different_priority_goal"}, arm, "PREVIOUS", check)


def test_the_batch_refuses_a_different_priority_share_record_without_a_goal(tmp_path, capsys):
    records = make_share_records(range(1, 41))
    index = next(i for i, record in enumerate(records) if record["reason_kind"] == DIFFERENT_PRIORITY_KIND)
    records[index]["different_priority_goal"] = None
    corpus_folder = make_corpus_folder(tmp_path, 1, records)
    assert run_batch(corpus_folder, FakeModel()) == 1
    assert "no different_priority_goal" in capsys.readouterr().err


# ---------- the goal on every output record, and re-dealing a failed prompt to a reason kind it has not tried ----------

def goal_and_kind_per_prompt_number(share_records: list[dict]) -> dict:
    return {"P" + record["unique_prompt_id"][-4:]: (record["reason_kind"], record["different_priority_goal"]) for record in share_records}


@pytest.mark.parametrize("arm", helpers.ARMS)
def test_the_dealt_goal_is_on_every_output_record_and_the_merged_pool(tmp_path, arm):
    share_records = make_share_records(range(1, 13), arm)
    dealt = goal_and_kind_per_prompt_number(share_records)
    assert any(goal for _, goal in dealt.values())
    corpus = make_corpus_folder(tmp_path, 1, share_records, arm)
    assert run_batch(corpus, FakeModel()) == 0
    batch_folder = only_batch_folder(corpus)
    accepted = read_output(batch_folder, "accepted_user_lean_flip_prompts_batch_001_*.jsonl")
    pool, _ = lean_merge.combine_returned_shares([corpus], None)
    for records in (read_output(batch_folder, "user_lean_flip_all_records_batch_001_*.jsonl"), accepted, pool):
        assert len(records) == 12
        for record in records:
            assert (record["reason_kind"], record["different_priority_goal"]) == dealt["P" + record["unique_prompt_id"][-4:]]
            assert record["attempt_number"] == 1 and record["originally_dealt_reason_kind"] == record["reason_kind"]
    rewrite_records = lean_batch.read_jsonl(next(batch_folder.glob("user_lean_flip_rewrites_*.jsonl")))
    assert all(record["different_priority_goal"] == dealt["P" + record["unique_prompt_id"][-4:]][1] for record in rewrite_records)


def test_a_prompt_tries_the_kinds_in_a_seeded_order_each_once_and_a_goal_comes_with_the_different_priority_kind():
    goals = helpers.read_different_priority_goals()
    for number in range(1, 40):
        record = make_share_records([number])[0]
        order = lean_batch.reason_kinds_in_the_order_a_prompt_tries_them(record["unique_prompt_id"], record["reason_kind"])
        assert order[0] == record["reason_kind"] and sorted(order) == sorted(REASON_KINDS)
        assert order == lean_batch.reason_kinds_in_the_order_a_prompt_tries_them(record["unique_prompt_id"], record["reason_kind"])
        attempts = [lean_batch.share_record_for_attempt(record, attempt, goals) for attempt in range(1, 5)]
        assert [attempt["reason_kind"] for attempt in attempts] == order
        assert attempts[0]["different_priority_goal"] == record["different_priority_goal"]
        for attempt in attempts[1:]:
            assert (attempt["different_priority_goal"] in goals) == (attempt["reason_kind"] == DIFFERENT_PRIORITY_KIND)
            assert attempt["originally_dealt_reason_kind"] == record["reason_kind"]
            assert attempt == lean_batch.share_record_for_attempt(record, attempt["attempt_number"], goals)
    assert len({tuple(lean_batch.reason_kinds_in_the_order_a_prompt_tries_them(f"P{number}", REASON_KINDS[0])) for number in range(40)}) > 1


@pytest.mark.parametrize("arm", helpers.ARMS)
def test_a_declined_prompt_is_run_again_with_a_kind_it_has_not_tried_and_a_prompt_that_fails_every_kind_stops_after_four(tmp_path, arm):
    share_records = make_share_records([1, 2, 3, 4], arm)
    corpus = make_corpus_folder(tmp_path, 1, share_records, arm)
    model = FakeModel(rewrite_behaviour={"P0002": "decline_once"}, check_behaviour={"P0003": "fail_once", "P0004": "fail"},
                      revision_check_behaviour={"P0003": "fail_once", "P0004": "fail"})
    assert run_batch(corpus, model) == 0
    batch_folder = only_batch_folder(corpus)
    attempts = [record for record in read_output(batch_folder, "user_lean_flip_all_records_batch_001_*.jsonl")]
    per_prompt = {number: [r for r in attempts if r["unique_prompt_id"].endswith(number)] for number in ("0001", "0002", "0003", "0004")}
    dealt = {record["unique_prompt_id"][-4:]: record["reason_kind"] for record in share_records}
    for number in ("0002", "0003"):
        first, second = per_prompt[number]
        assert (first["attempt_number"], second["attempt_number"]) == (1, 2) and first["reason_kind"] == dealt[number]
        assert second["reason_kind"] != first["reason_kind"] and second["originally_dealt_reason_kind"] == dealt[number]
        assert second["reason_kind"] == lean_batch.reason_kinds_in_the_order_a_prompt_tries_them(first["unique_prompt_id"], dealt[number])[1]
        assert second["accepted"] and not first["accepted"]
    assert [r["outcome"] for r in per_prompt["0002"]] == ["declined_by_rewriter", "accepted"]
    assert [r["outcome"] for r in per_prompt["0003"]] == ["rejected_by_check", "accepted"]
    assert len(per_prompt["0001"]) == 1
    assert [r["attempt_number"] for r in per_prompt["0004"]] == [1, 2, 3, 4]
    assert sorted(r["reason_kind"] for r in per_prompt["0004"]) == sorted(REASON_KINDS)  # every kind once, then it stops
    assert all((r["different_priority_goal"] is not None) == (r["reason_kind"] == DIFFERENT_PRIORITY_KIND) for r in attempts)
    assert sum(call[:2] == ("rewrite", "P0004") for call in model.calls) == 4
    accepted = {r["unique_prompt_id"][-4:]: r for r in read_output(batch_folder, "accepted_user_lean_flip_prompts_batch_001_*.jsonl")}
    assert sorted(accepted) == ["0001", "0002", "0003"]
    assert accepted["0002"]["attempt_number"] == 2 and accepted["0002"]["reason_kind"] == per_prompt["0002"][1]["reason_kind"]
    assert accepted["0002"]["originally_dealt_reason_kind"] == dealt["0002"] and accepted["0001"]["attempt_number"] == 1
    summary = summary_of(batch_folder)
    assert (summary["prompts_in_share"], summary["accepted_count"], summary["rejected_by_check_count"],
            summary["rescued_by_a_different_reason_kind_count"], summary["attempts_made_count"]) == (4, 3, 1, 2, 9)
    assert summary["accepted_by_final_reason_kind"] == {kind: sum(r["reason_kind"] == kind for r in accepted.values()) for kind in REASON_KINDS}
    assert summary["attempts_by_attempt_number"]["1"] == {"attempts": 4, "accepted": 1}
    assert summary["attempts_by_attempt_number"]["2"] == {"attempts": 3, "accepted": 2}
    assert summary["attempts_by_attempt_number"]["4"] == {"attempts": 1, "accepted": 0}
    (ledger_entry,) = lean_batch.read_ledger(corpus / LEDGER_FILE_NAME)
    assert ledger_entry["rescued_by_a_different_reason_kind_count"] == 2 and ledger_entry["accepted_count"] == 3
    assert ledger_entry["accepted_by_final_reason_kind"] == summary["accepted_by_final_reason_kind"]
    pool, report = lean_merge.combine_returned_shares([corpus], None)
    assert report["prompts_rescued_by_a_different_reason_kind"] == 2 and {r["attempt_number"] for r in pool} == {1, 2}


def test_a_batch_stopped_in_a_second_attempt_resumes_without_redoing_the_finished_first_attempt(tmp_path, capsys):
    corpus = make_corpus_folder(tmp_path, 1, make_share_records([1]))
    stopped_model = FakeModel(rewrite_behaviour={"P0001": "decline_once"}, usage_limit_after_calls=1)
    assert run_batch(corpus, stopped_model) == 3
    assert stopped_model.calls == [("rewrite", "P0001", "claude-sonnet-5-5")]
    batch_folder = only_batch_folder(corpus)
    marker_names = {path.name.split("_completed_")[0] for path in (batch_folder / "step_completion_markers").iterdir()}
    assert {"rewrite", "check", "revise", "check_revision"} <= marker_names and "rewrite_attempt_2" not in marker_names
    resumed_model = FakeModel()
    assert run_batch(corpus, resumed_model, resume=batch_folder) == 0
    assert [call[:2] for call in resumed_model.calls] == [("rewrite", "P0001"), ("check", "P0001")]  # attempt 2 only
    (attempt_one, attempt_two) = read_output(batch_folder, "user_lean_flip_all_records_batch_001_*.jsonl")
    assert (attempt_one["outcome"], attempt_two["outcome"]) == ("declined_by_rewriter", "accepted")
    assert attempt_two["reason_kind"] == lean_batch.reason_kinds_in_the_order_a_prompt_tries_them(
        attempt_one["unique_prompt_id"], attempt_one["reason_kind"])[1]
    rewrites = lean_batch.read_jsonl(next(batch_folder.glob("user_lean_flip_rewrites_*.jsonl")))
    assert [record["attempt_number"] for record in rewrites] == [1, 2]
    # a finished share is not run again
    assert run_batch(corpus, FakeModel()) == 1
