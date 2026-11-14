"""Tests for make_preachiness_positive_control_rewrites.py. No real API calls: a fake client is used throughout."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_preachiness_positive_control_rewrites as rewrites  # noqa: E402

TONES = ["neutral or journalistic", "supportive, like an animal advocate", rewrites.SKEPTICAL_TONE]
CENTRALITIES = ["the central subject of the document",
                "not the central subject of the document, but it is an important side discussion",
                "a minor detail mentioned only in passing, in a document primarily concerned with a different topic"]
RESOLUTIONS = ["The document features a specific scenario in which the AI flags harm.",
               "The document features a scenario ... and the AI correctly raises no welfare consideration at all."]


def make_corpus():
    corpus = []
    for index in range(120):
        tone = TONES[index % 3 if index % 7 else 2]
        corpus.append({
            "doc_id": f"matrix_{index:06d}", "content": f"Document number {index} about animals. " * 20,
            "language": "English" if index % 5 else "Japanese",
            "variables": {"tone": tone, "centrality": CENTRALITIES[index % 3], "resolution": RESOLUTIONS[index % 2]}})
    return corpus


def make_response(text="Rewritten text.", stop_reason="end_turn", model="claude-opus-5", iterations=None):
    usage = SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0, iterations=iterations)
    return SimpleNamespace(model=model, stop_reason=stop_reason, stop_details=None, usage=usage,
                           content=[SimpleNamespace(type="text", text=text)] if text is not None else [])


class FakeStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return False

    def get_final_message(self):
        return self.response


class FakeClient:
    def __init__(self, response_factory=lambda prompt_text: make_response()):
        self.calls = []
        self.response_factory = response_factory
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **arguments):
        self.calls.append(arguments)
        return FakeStream(self.response_factory(arguments["messages"][0]["content"]))


# --- selection ---------------------------------------------------------------

def test_selection_respects_exclusions_and_stratification_and_is_deterministic():
    corpus = make_corpus()
    selected = rewrites.select_source_documents(corpus, 10, seed=3)
    assert len(selected) == 10
    for document in selected:
        assert not document["variables"]["centrality"].startswith(rewrites.MINOR_DETAIL_CENTRALITY_PREFIX)
        assert rewrites.NO_WELFARE_RESOLUTION_PHRASE not in document["variables"]["resolution"]
    tones = [document["variables"]["tone"] for document in selected]
    assert tones.count(rewrites.SKEPTICAL_TONE) >= 3
    assert all(tones.count(tone) >= 1 for tone in TONES)
    assert selected == rewrites.select_source_documents(corpus, 10, seed=3)
    assert selected != rewrites.select_source_documents(corpus, 10, seed=4)


def test_minimum_skeptical_enforced_even_when_proportion_is_small():
    allocation = rewrites.allocate_documents_per_tone(
        10, {"neutral or journalistic": 200, "supportive, like an animal advocate": 200, rewrites.SKEPTICAL_TONE: 10})
    assert allocation[rewrites.SKEPTICAL_TONE] == 3 and sum(allocation.values()) == 10


# --- prompt rendering --------------------------------------------------------

def test_prompt_rendering_fills_both_slots():
    template = rewrites.PROMPT_FILE_PATH.read_text(encoding="utf-8")
    assert "{direction}" in template and "{document_text}" in template
    for direction, word in (("preachier", "PREACHIER"), ("plainer", "PLAINER")):
        rendered = rewrites.render_rewrite_prompt(template, direction, "BODY {direction} TEXT")
        assert f"Direction: {word}\n" in rendered
        assert "BODY {direction} TEXT" in rendered  # braces in the document are left alone
        assert "{document_text}" not in rendered


# --- API key -----------------------------------------------------------------

def test_key_is_read_from_env_file_not_from_os_environment(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=key-from-dotenv-file\n", encoding="utf-8")
    monkeypatch.setitem(os.environ, "ANTHROPIC_API_KEY", "key-from-operating-system")
    assert rewrites.load_anthropic_api_key(env_path) == "key-from-dotenv-file"


def test_missing_key_in_env_file_does_not_fall_back_to_os_environment(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER_KEY=1\n", encoding="utf-8")
    monkeypatch.setitem(os.environ, "ANTHROPIC_API_KEY", "key-from-operating-system")
    with pytest.raises(SystemExit):
        rewrites.load_anthropic_api_key(env_path)


# --- request and failure handling --------------------------------------------

def test_request_has_no_sampling_parameters_and_enables_fallback():
    arguments = rewrites.build_request_arguments("hello")
    assert arguments["fallbacks"] == "default" and arguments["betas"] == [rewrites.FALLBACK_BETA_HEADER]
    assert arguments["model"] == "claude-opus-5"
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in arguments


@pytest.mark.parametrize("stop_reason,text,expected_kind", [
    ("refusal", None, "refusal"), ("max_tokens", "cut off text", "max_tokens"), ("end_turn", "  ", "empty_output")])
def test_bad_responses_become_failures(stop_reason, text, expected_kind):
    client = FakeClient(lambda prompt: make_response(text=text, stop_reason=stop_reason))
    document = make_corpus()[0]
    record, failure, call_summary = rewrites.make_one_rewrite(client, "{direction} {document_text}", document, "plainer")
    assert record is None and failure["failure_kind"] == expected_kind
    assert call_summary is not None  # tokens are still costed


def test_transport_error_becomes_failure():
    def raising_stream(**arguments):
        raise RuntimeError("network down")
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=raising_stream)))
    record, failure, call_summary = rewrites.make_one_rewrite(client, "{direction}{document_text}",
                                                              make_corpus()[0], "plainer")
    assert record is None and failure["failure_kind"] == "transport_failure" and call_summary is None


def test_fallback_model_is_recorded_and_unknown_model_cost_is_unknown():
    iterations = [SimpleNamespace(type="message", model="claude-opus-5", input_tokens=10, output_tokens=0,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
                  SimpleNamespace(type="fallback_message", model="claude-mystery-9", input_tokens=100,
                                  output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0)]
    client = FakeClient(lambda prompt: make_response(model="claude-mystery-9", iterations=iterations))
    record, failure, call_summary = rewrites.make_one_rewrite(client, "{direction}{document_text}",
                                                              make_corpus()[0], "preachier")
    assert record["rewrite_model_used"] == "claude-mystery-9" and record["rewrite_model_requested"] == "claude-opus-5"
    assert call_summary["cost_dollars"] is None and call_summary["unpriced_models"] == ["claude-mystery-9"]
    assert call_summary["input_tokens"] == 110


# --- record format -----------------------------------------------------------

def test_record_matches_contract_and_length_flags():
    document = make_corpus()[1]
    word_count = len(document["content"].split())
    client = FakeClient(lambda prompt: make_response(text=" ".join(["word"] * round(word_count * 1.3))))
    record, _, _ = rewrites.make_one_rewrite(client, "{direction}{document_text}", document, "preachier")
    contract_fields = ["document_id", "source_document_id", "rewrite_direction", "content", "language", "variables",
                       "rewrite_model_requested", "rewrite_model_used", "source_word_count", "rewrite_word_count"]
    assert all(field in record for field in contract_fields)
    assert record["document_id"] == f"{document['doc_id']}__preachier"
    assert record["variables"] == document["variables"]
    assert record["source_word_count"] == word_count
    assert record["length_unit"] == "words" and record["length_within_fifteen_percent"] is False
    assert record["length_ratio"] == pytest.approx(1.3, abs=0.01)


def test_japanese_lengths_use_characters():
    document = next(d for d in make_corpus() if d["language"] == "Japanese")
    client = FakeClient(lambda prompt: make_response(text=document["content"]))
    record, _, _ = rewrites.make_one_rewrite(client, "{direction}{document_text}", document, "plainer")
    assert record["length_unit"] == "characters" and record["length_ratio"] == 1.0
    assert record["length_within_fifteen_percent"] is True


# --- resume and full run -----------------------------------------------------

def test_run_and_resume_skip_completed_pairs(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("\n".join(json.dumps(d) for d in make_corpus()), encoding="utf-8")
    common = ["--corpus-path", str(corpus_path), "--number-of-documents", "4", "--seed", "1", "--workers", "2"]

    def refuse_one_pair(prompt_text):  # first run: every plainer rewrite is refused
        return make_response(text=None, stop_reason="refusal") if "Direction: PLAINER" in prompt_text \
            else make_response()

    first_client = FakeClient(refuse_one_pair)
    run_folder = rewrites.main(["--run-name", "test_run", *common], client=first_client,
                               runs_directory_override=tmp_path / "runs")
    records = [json.loads(line) for line in (run_folder / "rewritten_documents.jsonl").read_text().splitlines()]
    assert len(first_client.calls) == 8 and len(records) == 4
    assert {record["rewrite_direction"] for record in records} == {"preachier"}
    assert list(run_folder.glob("rewrite_failures_*.jsonl")) and list(run_folder.glob("run_configuration_*.json"))
    configuration = json.loads(next(run_folder.glob("run_configuration_*.json")).read_text())
    assert configuration["seed"] == 1 and len(configuration["selected_document_ids"]) == 4
    assert configuration["prompt_file_sha256"]

    second_client = FakeClient()
    rewrites.main(["--run-name", "test_run", "--resume-folder", str(run_folder), *common], client=second_client,
                  runs_directory_override=tmp_path / "runs")
    records = [json.loads(line) for line in (run_folder / "rewritten_documents.jsonl").read_text().splitlines()]
    assert len(second_client.calls) == 4  # only the four plainer pairs were redone
    assert len(records) == 8
    assert len({(record["source_document_id"], record["rewrite_direction"]) for record in records}) == 8
    assert list(run_folder.glob("usage_and_cost_log_*.json"))


def test_dry_run_makes_no_api_call_and_writes_requests(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("\n".join(json.dumps(d) for d in make_corpus()), encoding="utf-8")
    monkeypatch.setattr(rewrites, "load_anthropic_api_key", lambda path: pytest.fail("key must not be read"))
    run_folder = rewrites.main(["--dry-run", "--corpus-path", str(corpus_path), "--number-of-documents", "3"],
                               runs_directory_override=tmp_path / "runs")
    assert len(list(run_folder.glob("rendered_requests_*.jsonl"))) == 1
    assert not (run_folder / "rewritten_documents.jsonl").exists()
