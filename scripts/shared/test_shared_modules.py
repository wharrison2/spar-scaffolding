"""Automated tests (no API calls) for the reusable modules in scripts/shared/."""

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import openai_structured_output_client as structured_output_client
from shared import quote_verification
from shared.api_usage_and_cost_tracking import UsageAndCostTracker, compute_cost_in_dollars
from shared.resumable_jsonl_records import AppendOnlyJsonlWriter, find_completed_keys
from shared.run_folder_management import check_resume_is_compatible
from shared.strict_json_schema_helpers import (assert_schema_is_strict_valid, build_array_schema,
                                               build_strict_object_schema, find_strict_schema_violations)

DOCUMENT = 'She said, "The fish may feel pain,"  and\nthen left. Wir sollten Tiere nicht leiden lassen.'


def test_quote_verification_tolerates_whitespace_and_curly_quotes():
    assert quote_verification.verify_quote("“The fish may feel pain,” and then left.", DOCUMENT)[0]
    assert quote_verification.verify_quote("and then   left.", DOCUMENT)[1]


def test_quote_verification_rejects_fabricated_quote():
    found, exact, score, spans_break = quote_verification.verify_quote(
        "Octopuses deserve legal protection under the new directive.", DOCUMENT)
    assert not found and not exact and score < quote_verification.FUZZY_MATCH_THRESHOLD
    assert not spans_break


PARAGRAPH_DOCUMENT = (
    "Hay dos preguntas. Ese beneficio es agron\u00f3mico y es tuyo.\n\n"
    "La segunda es si el escarabajo puede sufrir.  \n \n\nOtro p\u00e1rrafo distinto sobre otra cosa.")


def test_quote_spanning_paragraph_break_is_rejected():
    quote = "Ese beneficio es agron\u00f3mico y es tuyo. La segunda es si el escarabajo puede sufrir."
    found, exact, score, spans_break = quote_verification.verify_quote(quote, PARAGRAPH_DOCUMENT)
    assert not found and not exact
    assert score < quote_verification.FUZZY_MATCH_THRESHOLD
    assert spans_break


def test_quote_within_one_paragraph_passes_despite_whitespace_and_curly_quotes():
    document = "Intro.\n\nShe said, \"The fish   may feel\npain,\" and left.\n\nOutro."
    found, exact, _, spans_break = quote_verification.verify_quote("\u201cThe fish may feel pain,\u201d and left.", document)
    assert found and exact and not spans_break


def test_load_api_key_reads_env_file_not_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-from-os-environ")
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\n\nOTHER=x\nOPENAI_API_KEY="value-from-dotenv"\n')
    assert structured_output_client.load_api_key(env_file) == "value-from-dotenv"


def test_load_api_key_missing_env_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-from-os-environ")
    missing = tmp_path / ".env"
    with pytest.raises(RuntimeError, match=str(missing)):
        structured_output_client.load_api_key(missing)


def test_load_api_key_missing_key_raises(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\n")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        structured_output_client.load_api_key(env_file)


def test_load_api_key_source_has_no_environ_or_home_key():
    source = inspect.getsource(structured_output_client.load_api_key)
    assert "os.environ" not in source
    assert ".openai" not in source


def test_request_arguments_use_temperature_one_and_no_top_p():
    arguments = structured_output_client.build_structured_request_arguments(
        "m", "instructions", "input", "name", {"type": "object"}, "low")
    assert arguments["temperature"] == 1 and "top_p" not in arguments
    assert arguments["instructions"] == "instructions" and arguments["input"] == "input"


def test_cost_computation():
    assert compute_cost_in_dollars("gpt-5.6-luna", 1_000_000, 500_000, 1_000_000) == pytest.approx(0.1 + 0.01 + 1.2)


def test_usage_tracker_groups_and_writes_log(tmp_path):
    tracker = UsageAndCostTracker("gpt-5.6-luna", group_label="per_pass")
    usage = {"input_tokens": 100, "cached_input_tokens": 50, "output_tokens": 10, "reasoning_tokens": 4}
    tracker.add_usage("a", usage)
    tracker.add_usage("b", usage)
    assert tracker.overall_totals["calls"] == 2 and tracker.totals_per_group["a"]["calls"] == 1
    assert tracker.overall_totals["cache_rate"] == 0.5
    log_path = tmp_path / "log.json"
    tracker.write_log_file(log_path, "note")
    assert set(json.loads(log_path.read_text())) == {"updated_at", "note", "per_pass", "overall"}


def test_find_completed_keys_reads_back_appended_records(tmp_path):
    writer = AppendOnlyJsonlWriter()
    writer.append(tmp_path / "classifications_x_2026-01-01_00-00.jsonl", {"doc_id": "a", "pass": "p", "usage": {}})
    writer.append(tmp_path / "classifications_y_2026-01-01_00-00.jsonl", {"doc_id": "b", "pass": "q", "usage": {}})
    assert find_completed_keys(tmp_path, "classifications_*.jsonl", ["doc_id", "pass"]) == {("a", "p"), ("b", "q")}


def test_resume_refuses_changed_file_hashes(tmp_path):
    configuration = {"prompt_file_sha256": {"shared_header.txt": "a" * 64, "pass_x.txt": "b" * 64}}
    tampered = json.loads(json.dumps(configuration))
    tampered["prompt_file_sha256"]["shared_header.txt"] = "0" * 64
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError):
        check_resume_is_compatible(tmp_path, configuration)
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(configuration))
    check_resume_is_compatible(tmp_path, configuration)


def test_resume_refuses_folder_without_configuration(tmp_path):
    with pytest.raises(RuntimeError):
        check_resume_is_compatible(tmp_path, {"prompt_file_sha256": {}})


def test_strict_schema_helpers_accept_and_reject():
    schema = build_strict_object_schema({"a": {"type": "string"},
                                         "b": build_array_schema(build_strict_object_schema({"c": {"type": "boolean"}}), 3)})
    assert_schema_is_strict_valid(schema)
    assert schema["properties"]["b"]["maxItems"] == 3
    loose = {"type": "object", "properties": {"a": {"type": "string"}}, "required": [], "additionalProperties": False}
    assert find_strict_schema_violations(loose)
