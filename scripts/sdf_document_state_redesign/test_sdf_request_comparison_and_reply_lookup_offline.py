"""Offline tests for the request comparison and the stored-reply lookup of the SDF real-run replay.

No pipeline, no model call, no network. Run: python -m pytest scripts/sdf_document_state_redesign
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sdf_request_capture_from_pipeline_checkout as capture  # noqa: E402
import sdf_request_comparison as comparison  # noqa: E402


def make_request(item_id="matrix_000001", stage_tag="layer2_draft", occurrence=1, **overrides) -> dict:
    request = {
        "run_label": "run_a", "stage_tag": stage_tag, "item_id": item_id, "occurrence": occurrence,
        "route": "anthropic", "model": "claude-opus-5-5", "system_sha256": "hash_one",
        "user_message": "Write a document.\nSecond line.", "max_tokens": 14000, "temperature": 1.0,
        "effort": None, "thinking": {"type": "adaptive"}, "json_schema": None, "schema_name": None,
        "system_cached": [{"type": "ephemeral"}], "unrecognized_arguments": {},
    }
    request.update(overrides)
    return request


def test_identical_request_sets_have_no_differences():
    requests = [make_request("a"), make_request("b")]
    result = comparison.compare_request_sets(requests, [dict(r) for r in requests])
    assert result["differences"] == []
    assert result["counts"][("run_a", "layer2_draft")] == {
        "compared": 2, "equal": 2, "different": 0, "missing_in_second": 0, "extra_in_second": 0}


def test_deliberately_changed_request_is_reported_with_run_stage_item_and_field():
    first = [make_request("a"), make_request("b")]
    second = [make_request("a"), make_request("b", user_message="Write a document.\nSecond line changed.")]
    result = comparison.compare_request_sets(first, second)
    assert len(result["differences"]) == 1
    difference = result["differences"][0]
    assert (difference["run_label"], difference["stage_tag"], difference["item_id"], difference["field"], difference["kind"]) == \
        ("run_a", "layer2_draft", "b", "user_message", "different")
    assert "changed" in difference["text_diff"]
    assert result["counts"][("run_a", "layer2_draft")]["equal"] == 1
    assert result["counts"][("run_a", "layer2_draft")]["different"] == 1


def test_every_changed_field_is_listed_separately():
    changed = make_request("a", max_tokens=6001, system_cached=None, temperature=0.5)
    result = comparison.compare_request_sets([make_request("a")], [changed])
    assert sorted(d["field"] for d in result["differences"]) == ["max_tokens", "system_cached", "temperature"]
    assert result["counts"][("run_a", "layer2_draft")]["different"] == 1


def test_missing_and_extra_requests_are_reported():
    first = [make_request("a"), make_request("only_in_first")]
    second = [make_request("a"), make_request("only_in_second")]
    result = comparison.compare_request_sets(first, second)
    kinds = {(d["item_id"], d["kind"]) for d in result["differences"]}
    assert kinds == {("only_in_first", "missing_in_second"), ("only_in_second", "extra_in_second")}
    cell = result["counts"][("run_a", "layer2_draft")]
    assert (cell["compared"], cell["missing_in_second"], cell["extra_in_second"]) == (1, 1, 1)


def test_a_different_number_of_calls_for_one_item_is_reported():
    first = [make_request("a", occurrence=1)]
    second = [make_request("a", occurrence=1), make_request("a", occurrence=2)]
    result = comparison.compare_request_sets(first, second)
    assert [(d["occurrence"], d["kind"]) for d in result["differences"]] == [(2, "extra_in_second")]
    assert comparison.count_calls_per_stage_and_item(second) == {("run_a", "layer2_draft", "a"): 2}


def test_only_the_requested_fields_are_compared():
    first = [make_request("a")]
    second = [make_request("a", max_tokens=1)]
    held_by_sent_records = comparison.compare_request_sets(first, second, fields=comparison.FIELDS_HELD_BY_SENT_RECORDS)
    assert held_by_sent_records["differences"] == []


def test_system_prompt_hash_difference_shows_a_text_diff_when_texts_are_given():
    first = [make_request("a", system_sha256="h1")]
    second = [make_request("a", system_sha256="h2")]
    result = comparison.compare_request_sets(
        first, second, first_system_prompt_texts={"h1": "line one\nline two"},
        second_system_prompt_texts={"h2": "line one\nline 2"})
    assert result["differences"][0]["field"] == "system_sha256"
    assert "line two" in result["differences"][0]["text_diff"]


def test_two_requests_with_one_identity_are_refused():
    try:
        comparison.compare_request_sets([make_request("a"), make_request("a")], [])
    except ValueError:
        return
    raise AssertionError("a duplicate identity must raise")


def test_summary_rows_total_per_run_and_overall():
    result = comparison.compare_request_sets(
        [make_request("a"), make_request("b", stage_tag="layer4_score")],
        [make_request("a"), make_request("b", stage_tag="layer4_score", model="other")])
    rows = comparison.summary_rows(result["counts"])
    overall = rows[-1]
    assert (overall["run_label"], overall["compared"], overall["equal"], overall["different"]) == ("ALL RUNS", 2, 1, 1)


def test_anthropic_arguments_are_normalized_including_cache_flag_effort_and_schema():
    arguments = {
        "model": "claude-opus-5-5", "max_tokens": 100,
        "system": [{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "USER"}], "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}},
    }
    fields = capture.normalize_anthropic_arguments(arguments)["fields"]
    assert fields["system_sha256"] == capture.hash_of_text("SYSTEM")
    assert fields["system_cached"] == [{"type": "ephemeral"}]
    assert (fields["effort"], fields["json_schema"], fields["user_message"], fields["temperature"]) == \
        ("low", {"type": "object"}, "USER", None)
    assert fields["unrecognized_arguments"] == {}


def test_an_argument_the_normalizer_does_not_know_is_kept_as_a_field_that_can_differ():
    arguments = {"model": "m", "max_tokens": 1, "system": "S", "messages": [{"role": "user", "content": "U"}],
                 "thinking": {"type": "disabled"}, "some_new_argument": 5}
    assert capture.normalize_anthropic_arguments(arguments)["fields"]["unrecognized_arguments"] == {"some_new_argument": 5}


def test_openai_arguments_are_normalized():
    arguments = {"model": "gpt-6.1-sol", "input": "USER", "instructions": "SYSTEM", "max_output_tokens": 9,
                 "temperature": 1.0, "reasoning": {"effort": "medium"},
                 "text": {"format": {"type": "json_schema", "name": "n", "schema": {"a": 1}, "strict": True}}}
    fields = capture.normalize_openai_arguments(arguments)["fields"]
    assert (fields["route"], fields["effort"], fields["schema_name"], fields["json_schema"], fields["max_tokens"]) == \
        ("openai", "medium", "n", {"a": 1}, 9)
    assert fields["unrecognized_arguments"] == {"text_format": {"strict": True}}


def make_sent_record(stage, item_id, user_message, response_text, system_sha256="s") -> dict:
    return {"stage": stage, "item_id": item_id, "system_prompt_sha256": system_sha256,
            "user_messages": [user_message], "response_text": response_text, "stop_reason": "end_turn"}


def test_reply_lookup_prefers_the_same_request_text_then_the_item_and_serves_retries_in_order():
    provider = capture.StoredReplyProvider([
        make_sent_record("layer1_select", "x", "message one", "first reply"),
        make_sent_record("layer1_select", "x", "message one", "second reply"),
        make_sent_record("layer1_select", "y", "message two", "other reply"),
    ])
    assert provider.find("s", "message one", "layer1_select", "x")[0]["response_text"] == "first reply"
    assert provider.find("s", "message one", "layer1_select", "x")[0]["response_text"] == "second reply"
    record, how = provider.find("s", "a changed message", "layer1_select", "y")
    assert (record["response_text"], how) == ("other reply", "same_stage_and_item")
    assert provider.find("s", "unknown", "layer1_select", "nobody") == (None, "none")


def test_two_call_layer_3_requests_are_answered_from_the_single_call_reply_of_the_same_item():
    provider = capture.StoredReplyProvider([make_sent_record("layer3_rewrite", "d1", "old message", "one-call reply")])
    record, how = provider.find("s", "new critique message", "layer3_critique", "d1")
    assert record["response_text"] == "one-call reply" and how == "same_item_from_layer3_rewrite"
    assert provider.count_unused_records(("layer3_rewrite",)) == 0
