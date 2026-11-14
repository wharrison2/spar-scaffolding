"""Tests for depiction-label parsing and sampling, the scoring-output loader, the end-to-end analysis, the paired
version comparison, the document-format comparison and the LEACE effective directions. No API call, no model."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import analyze_persona_vector_projection_scores as analysis  # noqa: E402
from persona_vector_projection import compare_document_formats_on_positive_controls as format_comparison  # noqa: E402
from persona_vector_projection import compare_two_pipeline_versions_on_matched_specs as version_comparison  # noqa: E402
from persona_vector_projection import fit_leace_eraser_and_effective_facet_directions as leace  # noqa: E402
from persona_vector_projection import label_depiction_of_facets_in_corpus_documents as labelling  # noqa: E402
from persona_vector_projection import synthetic_scoring_run_for_analysis_tests as synthetic  # noqa: E402
from persona_vector_projection.load_persona_projection_scoring_outputs import (  # noqa: E402
    check_token_counts_match, facet_and_direction_from_document_id, load_depiction_labels, load_projection_data)
from shared.api_usage_and_cost_tracking import UsageAndCostTracker  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.strict_json_schema_helpers import find_strict_schema_violations  # noqa: E402

FACET = synthetic.REWRITTEN_FACET
CRITERIA = {"minimum_fraction_moving_expected_way": 0.8, "minimum_changed_tokens_for_inclusion": 5,
            "minimum_documents_per_spec_variable_value": 5, "document_score_correlation_flag_threshold": 0.8}


# ---- depiction labels -------------------------------------------------------------------------------------

def valid_result(label="shows_trait", kinds=("pig", "Pig", " dog ")):
    result = {facet: {"evidence": "the AI does X", "label": label} for facet in labelling.LABELLED_FACETS}
    result["kinds_of_animals_discussed"] = list(kinds)
    return result


def test_label_schema_is_strict_valid_and_covers_the_four_facets_and_three_labels():
    schema = labelling.build_label_schema()
    assert find_strict_schema_violations(schema) == []
    assert list(schema["properties"]) == labelling.LABELLED_FACETS + ["kinds_of_animals_discussed"]
    for facet in labelling.LABELLED_FACETS:
        assert schema["properties"][facet]["properties"]["label"]["enum"] == labelling.LABEL_VALUES
        assert list(schema["properties"][facet]["properties"]) == ["evidence", "label"]  # evidence before label


def test_valid_label_result_parses_and_record_takes_document_id_from_code_and_dedupes_kinds():
    labelling.validate_parsed_labels(valid_result())
    document = {"doc_id": "matrix_000007", "language": "French"}
    record = labelling.build_label_record(document, valid_result(), {"input_tokens": 1}, 1, 0.5, "medium")
    assert record["document_id"] == "matrix_000007"
    assert record["kinds_of_animals_discussed"] == ["pig", "dog"] and record["number_of_kinds_of_animals_discussed"] == 2
    assert set(record["labels"]) == set(labelling.LABELLED_FACETS)


@pytest.mark.parametrize("break_it", [
    lambda r: r.pop("consistent"),
    lambda r: r["consistent"].update(label="mostly"),
    lambda r: r["non_intrusive"].update(label=None),
    lambda r: r["cares_about_welfare"].pop("evidence"),
    lambda r: r.update(kinds_of_animals_discussed="pig"),
    lambda r: r.update(kinds_of_animals_discussed=[1, 2]),
])
def test_malformed_label_results_are_rejected(break_it):
    result = valid_result()
    break_it(result)
    with pytest.raises(ValueError):
        labelling.validate_parsed_labels(result)


class FakeResponses:
    """Stands in for client.responses.create: returns scripted outputs, first malformed then valid."""
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def create(self, **arguments):
        self.calls.append(arguments)
        text = self.outputs.pop(0)
        usage = SimpleNamespace(input_tokens=100, output_tokens=50, input_tokens_details=SimpleNamespace(cached_tokens=0),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=10))
        return SimpleNamespace(output_text=text, usage=usage, status="completed")


def test_one_malformed_reply_is_retried_once_and_two_become_a_failure_record():
    schema = labelling.build_label_schema()
    document = {"doc_id": "d1", "language": "English", "content": "text"}
    client = SimpleNamespace(responses=FakeResponses(["not json", json.dumps(valid_result())]))
    record, failure = labelling.label_one_document(client, "instructions", schema, document, "medium")
    assert failure is None and record["parse_attempts"] == 2
    request = client.responses.calls[0]
    assert request["input"] == "text" and request["model"] == "gpt-5.6-luna" and request["temperature"] == 1
    client = SimpleNamespace(responses=FakeResponses(["not json", json.dumps({"consistent": {}})]))
    record, failure = labelling.label_one_document(client, "instructions", schema, document, "medium")
    assert record is None and failure["error_kind"] == "parse_or_schema_failure" and failure["document_id"] == "d1"


def test_labelling_run_is_resumable_and_logs_cost(tmp_path):
    corpus = [{"doc_id": f"d{i}", "language": "English", "content": f"text {i}"} for i in range(5)]
    client = SimpleNamespace(responses=FakeResponses([json.dumps(valid_result())] * 5))
    labelling.run_labelling(client, tmp_path, "t1", "instructions", corpus[:3], "medium", 1, 10.0)
    assert len(read_jsonl(tmp_path / "depiction_labels_t1.jsonl")) == 3
    labelling.run_labelling(client, tmp_path, "t2", "instructions", corpus, "medium", 1, 10.0)
    assert len(client.responses.calls) == 5  # only the 2 new documents were sent
    log = json.loads((tmp_path / "usage_and_cost_log_t2.json").read_text())
    assert log["overall"]["calls"] == 2 and log["overall"]["cost_dollars"] > 0 and "cache_rate" in log["overall"]


def make_label_records(count=120):
    generator = np.random.default_rng(0)
    records = []
    for index in range(count):
        records.append({"document_id": f"d{index:03d}", "language": ["English", "Spanish", "Hindi", "German"][index % 4],
                        "labels": {facet: labelling.LABEL_VALUES[int(generator.choice(3, p=[0.5, 0.2, 0.3]))]
                                   for facet in labelling.LABELLED_FACETS},
                        "evidence": {facet: "e" for facet in labelling.LABELLED_FACETS},
                        "kinds_of_animals_discussed": ["pig"]})
    return records


def test_hand_check_sample_has_forty_distinct_documents_covering_every_label_and_several_languages(tmp_path):
    records = make_label_records()
    sample = labelling.choose_stratified_sample(records, 40, seed=0)
    assert len(sample) == 40 and len({r["document_id"] for r in sample}) == 40
    for facet in labelling.LABELLED_FACETS:
        assert {r["labels"][facet] for r in sample} == set(labelling.LABEL_VALUES)
    assert len({r["language"] for r in sample}) == 4
    assert sample == labelling.choose_stratified_sample(records, 40, seed=0)
    corpus_by_id = {r["document_id"]: {"content": "full text of " + r["document_id"]} for r in records}
    markdown, sheet, _ = labelling.write_hand_check_files(tmp_path, "t1", records, corpus_by_id)
    assert "full text of" in markdown.read_text() and len(pd.read_csv(sheet)) == 160


def test_label_file_loading_gives_one_column_per_facet(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_jsonl(path, make_label_records(5))
    frame = load_depiction_labels(path)
    assert set(frame.columns) >= {f"label__{facet}" for facet in labelling.LABELLED_FACETS}


def test_labelling_dry_run_prints_a_ceiling_above_the_expected_cost(tmp_path, capsys):
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(corpus_path, [{"doc_id": f"d{i}", "language": "English", "content": "x" * 8000, "variables": {}} for i in range(10)])
    folder = labelling.main(["--dry-run", "--corpus-path", str(corpus_path)], runs_directory_override=tmp_path)
    output = capsys.readouterr().out
    assert "COST CEILING" in output
    configuration = json.loads(next(folder.glob("run_configuration_*.json")).read_text())
    estimate = configuration["cost_estimate_and_ceiling"]
    assert estimate["ceiling_cost_dollars"] > estimate["expected_cost_dollars"] > 0


# ---- loader and end-to-end analysis ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("analysis_run")
    built = synthetic.build_synthetic_scoring_run(folder, seed=5)
    return {"folder": folder, "built": built}


def test_rewrite_facet_and_direction_are_read_from_the_document_id():
    assert facet_and_direction_from_document_id("matrix_000001__cares_about_welfare__toward_facet") == (
        "cares_about_welfare", "toward_facet")
    assert facet_and_direction_from_document_id("matrix_000034__plainer") == ("non_intrusive", "toward_facet")
    assert facet_and_direction_from_document_id("matrix_000034__preachier") == ("non_intrusive", "away_from_facet")
    assert facet_and_direction_from_document_id("matrix_000034") == (None, None)


def test_loader_reads_documents_baselines_layers_and_spec_variables(run_folder):
    data = load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"])
    assert data.layer_index == synthetic.LAYER_INDICES[2] and list(data.vector_names) == synthetic.VECTOR_NAMES
    kinds = data.documents["kind"].value_counts()
    assert kinds["corpus"] == 48 and kinds["baseline"] == 48 * synthetic.BASELINES_PER_DOCUMENT and kinds["rewrite"] == 36
    assert data.documents.loc["matrix_000000", "domain"] == "fisheries"
    rewrite = data.documents.loc[f"matrix_000000__{FACET}__away_from_facet"]
    assert rewrite["facet"] == FACET and rewrite["rewrite_direction"] == "away_from_facet"
    assert rewrite["source_document_id"] == "matrix_000000" and rewrite["language"] == "English"
    assert data.per_token["matrix_000000"].shape == (synthetic.TOKENS_PER_DOCUMENT, len(synthetic.VECTOR_NAMES))
    other = load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"], layer_index=10)
    assert other.layer_index == 10 and not np.allclose(other.mean_overall.to_numpy(), data.mean_overall.to_numpy())
    with pytest.raises(ValueError):
        load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"], layer_index=99)


def test_token_count_mismatch_is_caught():
    with pytest.raises(ValueError):
        check_token_counts_match({"a": [1, 2, 3]}, pd.Series({"a": 4}))
    check_token_counts_match({"a": [1, 2, 3]}, pd.Series({"a": 3}))


def test_full_analysis_runs_writes_the_expected_outputs_and_the_positive_control_passes(run_folder, tmp_path):
    data = load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"])
    labels = pd.DataFrame({f"label__{facet}": ["shows_trait" if i % 2 else "neither" for i in range(48)]
                           for facet in synthetic.VECTOR_NAMES[:4]}, index=[f"matrix_{i:06d}" for i in range(48)])
    output = tmp_path / "analysis"
    results = analysis.run_analysis_on_loaded_data(data, labels, run_folder["built"]["token_ids"], None, CRITERIA, output,
                                                   "t1", spans_per_direction=3)
    names = {path.name for path in output.glob("*")}
    for stem in ("document_scores_raw_and_standardised_within_language", "score_distributions_by_facet_and_language",
                 "scores_by_spec_variable_with_confidence_intervals", "scores_by_depiction_label",
                 "scores_by_position_band_and_opening", "openings_compared_with_rest_of_document",
                 "cross_facet_document_score_correlation", "ranking_agreement_baseline_split_half",
                 "extreme_spans", "positive_control_whole_document_summary", "positive_control_changed_span_summary",
                 "positive_control_pass_or_fail", "positive_control_shift_on_every_facet"):
        assert f"{stem}_t1.csv" in names, stem
    assert (output / "figures").is_dir() and len(list((output / "figures").glob("*.png"))) >= 6
    assert (output / "analysis_summary_t1.md").read_text().startswith("# Persona vector projection")
    passed = results["positive_control_pass_or_fail"]
    assert passed["passed"].all()
    spans = results["extreme_spans"]
    assert set(spans["direction"]) == {"highest", "lowest"} and spans["text"].isna().all()  # no tokenizer given
    assert not spans[~spans["starts_in_opening"]]["start_token"].lt(64).any()
    by_variable = pd.read_csv(output / "scores_by_spec_variable_with_confidence_intervals_t1.csv")
    assert {"facet", "variable", "group", "n", "mean", "ci_low", "ci_high", "too_few_documents"} <= set(by_variable.columns)


def test_cli_without_a_tokenizer_skips_span_level_analysis_but_still_runs_whole_document(run_folder):
    output = analysis.main(["--scoring-run-folder", str(run_folder["folder"]), "--corpus-path",
                            str(run_folder["built"]["corpus_path"]), "--analysis-name", "synthetic_test",
                            "--pass-criteria-json", str(analysis.DEFAULT_CRITERIA_PATH)])
    assert list(output.glob("positive_control_whole_document_summary_*.csv"))
    assert not list(output.glob("positive_control_changed_span_summary_*.csv"))


def test_saved_token_ids_make_the_changed_span_analysis_run_without_a_tokenizer(tmp_path):
    built = synthetic.build_synthetic_scoring_run(tmp_path / "run_with_token_ids", save_token_ids=True)
    data = load_projection_data(tmp_path / "run_with_token_ids", "assistant_turn", built["corpus_path"])
    assert list(data.token_ids["matrix_000000"]) == list(built["token_ids"]["matrix_000000"])
    data.configuration["_run_folder"] = str(tmp_path / "run_with_token_ids")
    token_ids, token_strings = analysis.load_token_data(data, built["corpus_path"], None)
    assert token_strings is None and token_ids["matrix_000000"] == list(built["token_ids"]["matrix_000000"])
    output = analysis.main(["--scoring-run-folder", str(tmp_path / "run_with_token_ids"), "--corpus-path",
                            str(built["corpus_path"]), "--analysis-name", "synthetic_saved_ids",
                            "--pass-criteria-json", str(analysis.DEFAULT_CRITERIA_PATH)])
    assert list(output.glob("positive_control_changed_span_summary_*.csv"))


def _fake_tokenizer_loader(calls):
    def from_pretrained(name, **keyword_arguments):
        calls.append((name, keyword_arguments))
        return SimpleNamespace(convert_ids_to_tokens=lambda ids: [f"token_{i}" for i in ids])
    return from_pretrained


def test_tokenizer_is_loaded_at_the_recorded_model_revision_without_a_token(tmp_path, monkeypatch):
    import transformers
    built = synthetic.build_synthetic_scoring_run(tmp_path / "run_with_token_ids", save_token_ids=True)
    data = load_projection_data(tmp_path / "run_with_token_ids", "assistant_turn", built["corpus_path"])
    data.configuration["model_revision"] = "recorded_revision"
    calls = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _fake_tokenizer_loader(calls))
    _, token_strings = analysis.load_token_data(data, built["corpus_path"], "some/model")
    assert calls == [("some/model", {"revision": "recorded_revision", "token": False})] and token_strings
    calls.clear()
    analysis.load_token_data(data, built["corpus_path"], "some/model", tokenizer_revision="override_revision")
    assert calls == [("some/model", {"revision": "override_revision", "token": False})]


def test_tokenizer_loading_fails_when_no_revision_is_recorded_or_given(tmp_path, monkeypatch):
    import transformers
    built = synthetic.build_synthetic_scoring_run(tmp_path / "run_with_token_ids", save_token_ids=True)
    data = load_projection_data(tmp_path / "run_with_token_ids", "assistant_turn", built["corpus_path"])
    data.configuration.pop("model_revision", None)
    calls = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _fake_tokenizer_loader(calls))
    with pytest.raises(ValueError, match="tokenizer revision"):
        analysis.load_token_data(data, built["corpus_path"], "some/model")
    assert calls == []


def test_spans_include_text_when_token_strings_are_given(run_folder, tmp_path):
    data = load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"])
    strings = {document_id: [f"▁tok{position}" for position in range(synthetic.TOKENS_PER_DOCUMENT)]
               for document_id in data.per_token}
    results = analysis.run_analysis_on_loaded_data(data, None, run_folder["built"]["token_ids"], strings, CRITERIA,
                                                   tmp_path / "with_text", "t2", spans_per_direction=2)
    assert results["extreme_spans"]["text"].str.contains("tok").all()
    assert list((tmp_path / "with_text").glob("extreme_spans_for_hand_reading_*.md"))


# ---- paired comparison ------------------------------------------------------------------------------------------

def test_paired_comparison_recovers_a_known_shift_and_the_shared_baseline_cancels(run_folder):
    first = load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"])
    second = copy.deepcopy(first)
    column = list(first.vector_names).index(FACET)
    corpus_ids = list(first.documents.index[first.documents["kind"] == "corpus"])
    second.mean_overall.loc[corpus_ids, FACET] += 2.0
    pairs = pd.DataFrame({"first_document_id": corpus_ids, "second_document_id": corpus_ids})
    results = version_comparison.compare_versions(first, second, pairs, "shared_first", minimum_documents=5)
    sds = pd.Series({d: float(version_comparison.ScoreTables(first).language_statistics.loc[
        first.documents.loc[d, "language"], f"{FACET}__sd"]) for d in corpus_ids})
    assert np.allclose(results["differences"][FACET].astype(float).to_numpy(), (2.0 / sds).to_numpy(), atol=1e-5)
    summary = results["summary"].set_index("facet")
    assert summary.loc[FACET, "n_pairs"] == 48 and summary.loc[FACET, "ci_low"] > 0
    assert abs(summary.loc["consistent", "mean_second_minus_first"]) < 1e-5
    own = version_comparison.compare_versions(first, second, pairs, "own", minimum_documents=5)
    assert np.allclose(own["differences"][FACET].astype(float), results["differences"][FACET].astype(float), atol=1e-5)
    assert not results["by_variable"].empty


def test_paired_comparison_refuses_runs_scored_on_different_instruments(run_folder):
    first = load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"])
    second = load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"], layer_index=11)
    with pytest.raises(ValueError):
        version_comparison.compare_versions(first, second, pd.DataFrame({"first_document_id": [], "second_document_id": []}))
    third = copy.deepcopy(first)
    third.configuration = {**first.configuration, "model_revision": "other"}
    with pytest.raises(ValueError):
        version_comparison.compare_versions(first, third, pd.DataFrame({"first_document_id": [], "second_document_id": []}))


def test_documents_without_a_unique_spec_match_are_listed_as_unpaired_not_guessed(run_folder):
    first = load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"])
    results = version_comparison.compare_versions(first, copy.deepcopy(first), None, minimum_documents=5,
                                                  spec_variables=["domain", "tone"])
    assert len(results["pairs"]) == 0 and len(results["unpaired"]) == 96  # 6 spec combinations, none unique


# ---- document format comparison ----------------------------------------------------------------------------------

def test_format_comparison_ranks_a_format_that_carries_the_signal_above_one_that_does_not(run_folder):
    good = load_projection_data(run_folder["folder"], "assistant_turn", run_folder["built"]["corpus_path"])
    bad = copy.deepcopy(good)
    bad.document_format = "raw_text"
    rewrites = bad.documents.index[bad.documents["kind"] == "rewrite"]
    column = list(bad.vector_names).index(FACET)
    for rewrite_id in rewrites:  # remove the rewrite effect in the weak format
        source_id = bad.documents.loc[rewrite_id, "source_document_id"]
        bad.mean_overall.loc[rewrite_id, FACET] = bad.mean_overall.loc[source_id, FACET]
        bad.per_token[rewrite_id] = bad.per_token[source_id].copy()
    token_ids = run_folder["built"]["token_ids"]
    combined, totals = format_comparison.compare_formats(
        {"assistant_turn": good, "raw_text": bad}, {"assistant_turn": token_ids, "raw_text": token_ids}, CRITERIA)
    assert totals.loc["assistant_turn", "facet_level_combinations_passed"] == 2
    assert totals.loc["raw_text", "facet_level_combinations_passed"] == 0
    assert set(combined["document_format"]) == {"assistant_turn", "raw_text"}


# ---- LEACE -------------------------------------------------------------------------------------------------------

def test_effective_direction_reproduces_the_erased_projection_up_to_a_constant():
    pytest.importorskip("concept_erasure")
    import torch
    from concept_erasure import LeaceEraser
    generator = np.random.default_rng(0)
    labels = np.array([0, 1] * 150)
    activations = generator.normal(size=(300, 20)) + np.outer(labels, generator.normal(size=20)) * 2.0
    proj_left, proj_right, _ = leace.fit_leace_eraser(activations, labels)
    vector = generator.normal(size=20)
    erased_direction = leace.effective_direction_after_erasure(vector, proj_left, proj_right)
    eraser = LeaceEraser.fit(torch.tensor(activations, dtype=torch.float64),
                             torch.tensor(labels, dtype=torch.float64).reshape(-1, 1))
    new_points = generator.normal(size=(50, 20))
    erased_projection = (eraser(torch.tensor(new_points, dtype=torch.float64)).numpy() @ vector)
    direct = new_points @ erased_direction
    offsets = erased_projection - direct
    assert np.allclose(offsets, offsets[0], atol=1e-8)  # a constant, so rankings and baseline subtraction are unaffected
    assert np.allclose(np.argsort(erased_projection), np.argsort(direct))


def test_erased_direction_no_longer_separates_the_two_topic_classes_on_average():
    pytest.importorskip("concept_erasure")
    generator = np.random.default_rng(1)
    labels = np.array([0, 1] * 150)
    topic_axis = generator.normal(size=20)
    activations = generator.normal(size=(300, 20)) + np.outer(labels, topic_axis) * 3.0
    vector = topic_axis + 0.5 * generator.normal(size=20)  # a facet vector that partly carries the topic
    results = leace.compute_effective_directions(lambda layer: activations, labels, {"cares_about_welfare": {7: vector}},
                                                 {"cares_about_welfare": 7})
    erased = results["cares_about_welfare"]["erased_direction"].astype(np.float64)
    before = abs((activations[labels == 1] @ vector).mean() - (activations[labels == 0] @ vector).mean())
    after = abs((activations[labels == 1] @ erased).mean() - (activations[labels == 0] @ erased).mean())
    assert after < 0.02 * before
    assert results["cares_about_welfare"]["share_of_vector_norm_removed"] > 0.3


def test_leace_variant_file_keeps_the_stored_shape_and_is_found_by_the_scoring_pod_loader(tmp_path):
    from persona_vector_projection.scoring_pod_vector_loading_adapter import find_variant_vector_file_names
    facet_vector = np.arange(30, dtype=np.float32).reshape(5, 6)
    erased = np.full(6, -1.0, dtype=np.float32)
    file_name = leace.write_scoring_variant_vector(tmp_path, "non_intrusive", facet_vector, 2, erased, "2026-10-31_12-00")
    saved = np.load(tmp_path / file_name)
    assert saved.shape == facet_vector.shape and np.array_equal(saved[2], erased)
    assert np.array_equal(np.delete(saved, 2, axis=0), np.delete(facet_vector, 2, axis=0))
    assert find_variant_vector_file_names(tmp_path) == {"non_intrusive__leace_erased": file_name}


def test_effective_direction_formula_with_a_known_projection():
    proj_left = np.array([[1.0], [0.0], [0.0]])
    proj_right = np.array([[1.0, 0.0, 0.0]])  # A = I - e1 e1^T removes the first coordinate
    assert np.allclose(leace.effective_direction_after_erasure([2.0, 3.0, 4.0], proj_left, proj_right), [0.0, 3.0, 4.0])
