"""Tests for the facet positive-control rewrites. A fake client is used throughout: no API call, no .env read."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for extra in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from persona_vector_projection import make_facet_positive_control_rewrites as rewrites  # noqa: E402
from persona_vector_projection import select_documents_with_room_to_move_for_facet as selection  # noqa: E402
from shared.resumable_jsonl_records import read_all_records, read_jsonl, write_jsonl  # noqa: E402

CENTRALITIES = ["the central subject of the document",
                "not the central subject of the document, but it is an important side discussion",
                "a minor detail mentioned only in passing, in a document primarily concerned with a different topic"]
RESOLUTIONS = ["The document features a specific scenario in which the AI flags harm.",
               "The document features a scenario ... and the AI correctly raises no welfare consideration at all."]
LANGUAGES = ["English"] * 6 + ["Spanish"] * 3 + ["Japanese"]


def make_corpus(count=120):
    corpus = []
    for index in range(count):
        language = LANGUAGES[index % len(LANGUAGES)]
        words = [f"w{index}_{position}" for position in range(150)]
        corpus.append({"doc_id": f"matrix_{index:06d}", "content": ("".join(words[:80]) if language == "Japanese"
                                                                     else " ".join(words)),
                       "language": language,
                       "variables": {"centrality": CENTRALITIES[index % 3], "resolution": RESOLUTIONS[(index // 2) % 2],
                                     "sentient_category": "farmed invertebrates" if index % 4 == 0 else "farmed mammals",
                                     "tone": "neutral"}})
    return corpus


def make_response(text, stop_reason="end_turn", model="claude-opus-5"):
    usage = SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0, iterations=None)
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
    def __init__(self, response_factory):
        self.calls = []
        self.response_factory = response_factory
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **arguments):
        self.calls.append(arguments)
        return FakeStream(self.response_factory(arguments["messages"][0]["content"]))


def document_text_in(prompt):
    return prompt.split("Document:\n", 1)[1].rstrip("\n")


def change_words(text, how_many, tag="new"):
    words = text.split()
    for position in range(0, min(how_many, len(words))):
        words[position] = f"{tag}{position}"
    return " ".join(words)


def stubborn_then_generous(prompt):
    """Changes 5 words unless the prompt carries the retry note, then 100."""
    return make_response(change_words(document_text_in(prompt), 100 if "previous attempt" in prompt else 5))


def always_changes(how_many):
    return lambda prompt: make_response(change_words(document_text_in(prompt), how_many))


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, make_corpus())
    return path


@pytest.fixture
def preachiness_file(tmp_path):
    corpus = make_corpus()
    records = []
    for index in (0, 1, 2, 3):
        source = corpus[index]
        for direction, how_many in (("preachier", 90), ("plainer", 70)):
            records.append({"document_id": f"{source['doc_id']}__{direction}", "source_document_id": source["doc_id"],
                            "rewrite_direction": direction, "content": change_words(source["content"], how_many, direction),
                            "language": source["language"], "variables": source["variables"],
                            "rewrite_model_requested": "claude-opus-5", "rewrite_model_used": "claude-opus-5",
                            "length_unit": "words", "source_length": 150, "rewrite_length": 150, "length_ratio": 1.0})
    path = tmp_path / "preachiness_rewrites.jsonl"
    write_jsonl(path, records)
    return path


def run(tmp_path, corpus_file, preachiness_file, client, extra=(), dry_run=False):
    arguments = ["--corpus-path", str(corpus_file), "--existing-preachiness-rewrites-jsonl", str(preachiness_file),
                 "--documents-per-facet", "6", "--minimum-changed-units", "40", "--workers", "1",
                 "--env-path", str(tmp_path / "no_such_env_file")] + list(extra) + (["--dry-run"] if dry_run else [])
    return rewrites.main(arguments, client=client, runs_directory_override=tmp_path / "runs")


# ---- minimum changed tokens ----------------------------------------------------------------------

def test_rewrite_below_the_minimum_is_regenerated_with_a_note_and_then_passes(tmp_path, corpus_file, preachiness_file):
    client = FakeClient(stubborn_then_generous)
    folder = run(tmp_path, corpus_file, preachiness_file, client)
    records = [r for r in read_all_records(folder, rewrites.REWRITES_FILE_PATTERN) if not r["reused_from_preachiness_rewrites"]]
    assert records and all(not r["below_minimum_changed_units"] for r in records)
    assert all(r["number_of_attempts"] == 2 for r in records)
    assert any("previous attempt" in call["messages"][0]["content"] for call in client.calls)
    assert all(r["changed_units_per_attempt"][0] < 40 <= r["changed_units_per_attempt"][1] for r in records)


def test_without_regeneration_small_rewrites_are_flagged_and_left_out_of_the_export(tmp_path, corpus_file, preachiness_file):
    client = FakeClient(always_changes(5))
    folder = run(tmp_path, corpus_file, preachiness_file, client, ["--no-regeneration"])
    records = read_all_records(folder, rewrites.REWRITES_FILE_PATTERN)
    new = [r for r in records if not r["reused_from_preachiness_rewrites"]]
    assert new and all(r["below_minimum_changed_units"] and r["number_of_attempts"] == 1 for r in new)
    exported = read_jsonl(next(folder.glob("scoring_pod_extra_documents_*.jsonl")))
    exported_ids = {r["document_id"] for r in exported}
    assert not exported_ids & {r["document_id"] for r in new}
    assert all(r["rewrite_direction"] in ("toward_facet", "away_from_facet") for r in exported)  # reused ones only


def test_a_rewrite_that_never_reaches_the_minimum_is_flagged_after_the_last_regeneration(tmp_path, corpus_file,
                                                                                         preachiness_file):
    client = FakeClient(always_changes(5))
    folder = run(tmp_path, corpus_file, preachiness_file, client, ["--maximum-regenerations", "1"])
    new = [r for r in read_all_records(folder, rewrites.REWRITES_FILE_PATTERN) if not r["reused_from_preachiness_rewrites"]]
    assert all(r["below_minimum_changed_units"] and r["number_of_attempts"] == 2 for r in new)


def test_japanese_minimum_is_scaled_to_characters():
    assert rewrites.minimum_units_for_language(40, "English") == 40
    assert rewrites.minimum_units_for_language(40, "Japanese") == 60
    assert rewrites.describe_amount(60, "Japanese") == "60 characters"


# ---- same-size neutral rewrites, own source, reuse -------------------------------------------------------

def test_neutral_target_is_the_mean_size_of_the_documents_own_toward_and_away_rewrites(tmp_path, corpus_file,
                                                                                       preachiness_file):
    def factory(prompt):
        text = document_text_in(prompt)
        if "SAME PLACES" in prompt:  # neutral instruction
            return make_response(change_words(text, 60))
        return make_response(change_words(text, 100 if "AWAY from the trait" in prompt else 50))
    folder = run(tmp_path, corpus_file, preachiness_file, FakeClient(factory))
    records = read_all_records(folder, rewrites.REWRITES_FILE_PATTERN)
    neutral = [r for r in records if r["rewrite_direction"] == "facet_neutral" and r["facet"] != "non_intrusive"]
    assert neutral
    for record in neutral:
        assert record["target_changed_units_for_neutral"] == pytest.approx(75, abs=2)  # mean of 50 and 100
        assert record["size_matched_to_target"] is True
    index = rewrites.index_records(records)
    for (source, facet), by_direction in index.items():
        assert {"toward_facet", "away_from_facet", "facet_neutral"} <= set(by_direction)
        for record in by_direction.values():
            assert record["source_document_id"] == source  # every rewrite sits under its own source


def test_non_intrusive_reuses_the_preachiness_rewrites_and_gets_neutral_rewrites_for_the_same_documents(
        tmp_path, corpus_file, preachiness_file):
    folder = run(tmp_path, corpus_file, preachiness_file, FakeClient(always_changes(60)))
    records = [r for r in read_all_records(folder, rewrites.REWRITES_FILE_PATTERN) if r["facet"] == "non_intrusive"]
    reused = [r for r in records if r["reused_from_preachiness_rewrites"]]
    assert {(r["rewrite_direction"], r["original_preachiness_rewrite_direction"]) for r in reused} == {
        ("toward_facet", "plainer"), ("away_from_facet", "preachier")}
    sources = {r["source_document_id"] for r in reused}
    neutral = {r["source_document_id"] for r in records if r["rewrite_direction"] == "facet_neutral"}
    assert sources == neutral == {"matrix_000000", "matrix_000001", "matrix_000002", "matrix_000003"}
    assert not any(r["reused_from_preachiness_rewrites"] for r in records if r["rewrite_direction"] == "facet_neutral")


def test_export_has_the_extra_documents_shape_with_source_facet_and_direction(tmp_path, corpus_file, preachiness_file):
    folder = run(tmp_path, corpus_file, preachiness_file, FakeClient(always_changes(100)))
    exported = read_jsonl(next(folder.glob("scoring_pod_extra_documents_*.jsonl")))
    required = ["document_id", "source_document_id", "rewrite_direction", "content", "language", "variables"]
    assert exported and all(all(field in record for field in required + ["facet"]) for record in exported)
    assert len({r["document_id"] for r in exported}) == len(exported)
    assert {r["facet"] for r in exported} == {"cares_about_welfare", "calibrated_consideration", "consistent",
                                              "non_intrusive"}


# ---- resume, failures, cost cap, dry run ------------------------------------------------------------------

def test_resume_makes_no_calls_for_finished_rewrites_and_retries_failures(tmp_path, corpus_file, preachiness_file):
    state = {"refuse_next": 3}

    def flaky(prompt):
        if state["refuse_next"] > 0:
            state["refuse_next"] -= 1
            return make_response(None, stop_reason="refusal")
        return make_response(change_words(document_text_in(prompt), 100))
    client = FakeClient(flaky)
    folder = run(tmp_path, corpus_file, preachiness_file, client)
    failures = read_all_records(folder, "rewrite_failures_*.jsonl")
    assert len(failures) == 3 and all(f["failure_kind"] == "refusal" for f in failures)
    done_before = len(read_all_records(folder, rewrites.REWRITES_FILE_PATTERN))
    calls_before = len(client.calls)
    run(tmp_path, corpus_file, preachiness_file, client, ["--resume-folder", str(folder)])
    records = read_all_records(folder, rewrites.REWRITES_FILE_PATTERN)
    assert len({(r["source_document_id"], r["facet"], r["rewrite_direction"]) for r in records}) == len(records)
    assert len(records) > done_before
    assert len(client.calls) - calls_before >= 3  # the failed ones were retried (plus whatever their absence blocked)


def test_cost_cap_stops_new_calls(tmp_path, corpus_file, preachiness_file):
    client = FakeClient(always_changes(100))
    folder = run(tmp_path, corpus_file, preachiness_file, client, ["--maximum-cost-dollars", "0.0"])
    assert client.calls == []
    skipped = read_all_records(folder, "rewrite_failures_*.jsonl")
    assert skipped and all(f["failure_kind"] == "skipped_cost_cap" for f in skipped)


def test_dry_run_makes_no_call_and_prints_a_ceiling(tmp_path, corpus_file, preachiness_file, capsys):
    client = FakeClient(always_changes(100))
    folder = run(tmp_path, corpus_file, preachiness_file, client, dry_run=True)
    assert client.calls == []
    estimate = json.loads(next(folder.glob("dry_run_cost_estimate_and_ceiling_*.json")).read_text())
    assert estimate["ceiling_cost_dollars"] > estimate["default_cost_cap_dollars"] > 0
    assert estimate["calls_planned"] == 3 * 6 * 3 + 4
    assert "COST CEILING" in capsys.readouterr().out


def test_instruction_files_have_the_slots_the_code_fills():
    templates = rewrites.load_instruction_templates()
    for name, text in templates.items():
        assert "{document_text}" in text or name == rewrites.RETRY_NOTE_FILE_NAME, name
        if name.endswith("facet_neutral__draft.txt"):
            assert "{target_changed_amount}" in text and "{lowest_changed_amount}" in text
        elif name != rewrites.RETRY_NOTE_FILE_NAME:
            assert "{minimum_changed_amount}" in text


def test_document_text_with_braces_is_not_interpreted_as_a_slot():
    templates = rewrites.load_instruction_templates()
    document = {"doc_id": "d", "language": "English", "content": "code {minimum_changed_amount} and {document_text}"}
    prompt = rewrites.render_instruction(templates[rewrites.instruction_file_name("consistent", "toward_facet")], document, 60)
    assert prompt.rstrip().endswith("code {minimum_changed_amount} and {document_text}")


# ---- document selection ---------------------------------------------------------------------------------

def test_selection_excludes_hopeless_documents_stratifies_by_language_and_is_deterministic():
    corpus = make_corpus()
    first, report = selection.select_documents_for_facet(corpus, "cares_about_welfare", 30, seed=1)
    again, _ = selection.select_documents_for_facet(corpus, "cares_about_welfare", 30, seed=1)
    other, _ = selection.select_documents_for_facet(corpus, "cares_about_welfare", 30, seed=2)
    assert first == again and first != other and len(first) == 30
    for document in first:
        assert not document["variables"]["centrality"].startswith(selection.MINOR_DETAIL_CENTRALITY_PREFIX)
        assert selection.NO_WELFARE_RESOLUTION_PHRASE not in document["variables"]["resolution"]
    languages = {document["language"] for document in first}
    assert languages == {"English", "Spanish", "Japanese"}
    assert sum(report["allocation_per_language"].values()) == 30


def test_consistent_prefers_documents_with_several_kinds_of_animal_when_labels_are_given():
    corpus = make_corpus()
    labels = {document["doc_id"]: {"labels": {"consistent": "shows_trait"},
                                   "kinds_of_animals_discussed": ["pig", "dog"] if index % 5 == 0 else ["pig"]}
              for index, document in enumerate(corpus)}
    selected, report = selection.select_documents_for_facet(corpus, "consistent", 10, seed=0, depiction_labels_by_document_id=labels)
    multi = [d for d in selected if len(labels[d["doc_id"]]["kinds_of_animals_discussed"]) > 1]
    assert len(multi) >= 6 and report["selected_with_best_preference_rank"] == len(multi)


def test_label_requirement_is_relaxed_with_a_warning_when_too_few_qualify():
    corpus = make_corpus()
    labels = {document["doc_id"]: {"labels": {"cares_about_welfare": "neither"}} for document in corpus}
    selected, report = selection.select_documents_for_facet(corpus, "cares_about_welfare", 10, seed=0, depiction_labels_by_document_id=labels)
    assert len(selected) == 10 and any("relaxed" in warning for warning in report["warnings"])


def test_language_allocation_gives_every_large_language_a_document_and_sums_exactly():
    allocation = selection.allocate_documents_per_language(30, {"English": 100, "Spanish": 30, "French": 26, "Polish": 3}, 25)
    assert sum(allocation.values()) == 30 and allocation["Spanish"] >= 1 and allocation["French"] >= 1
