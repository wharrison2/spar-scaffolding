"""Automated tests (no API calls) for the plausibility audit judge and post-processor."""

import copy
import csv as csv_module
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_corpus_plausibility as judge
import postprocess_plausibility_judgments as postprocess
from shared.strict_json_schema_helpers import assert_schema_is_strict_valid


@pytest.fixture(scope="module")
def definition():
    return judge.JudgeDefinition()


def valid_axis(score=5, applicable=True, problems=()):
    return {"applicable": applicable, "problems": list(problems), "score_reasoning": "r", "score": score}


def valid_parsed_result(axis_keys, overall=5):
    return {"axes": {key: valid_axis() for key in axis_keys}, "other_implausibilities": [],
            "overall_reasoning": "ok", "overall_plausibility": overall}


def problem(quote="The fish may feel pain.", severity="minor"):
    return {"quote": quote, "english_gloss": None, "explanation": "e", "severity": severity}


# ---- request content and schema ----

def test_request_contains_only_instructions_and_content(definition):
    documents = judge.load_corpus(judge.DEFAULT_CORPUS_PATH)[:40]
    for document in documents:
        arguments = judge.build_request_arguments(definition, document["content"], "medium")
        assert arguments["input"] == document["content"]
        assert arguments["instructions"] == definition.instructions_text
        assert arguments["temperature"] == 1 and "top_p" not in arguments
        sent = json.dumps(arguments, ensure_ascii=False)
        leaked_values = [document["doc_id"], document["description"]]
        leaked_values += [str(v) for v in postprocess.parse_variables(document["variables"]).values()]
        for value in leaked_values:
            if value and value not in document["content"]:
                assert value not in sent, f"leaked: {value[:60]}"
        assert document["doc_id"] not in sent


def test_axis_keys_match_prompt_headers_and_there_are_ten(definition):
    headers = [line[4:].strip() for line in definition.instructions_text.splitlines() if line.startswith("### axis_")]
    assert definition.axis_keys == headers
    assert len(headers) == 10 and len(set(headers)) == 10
    assert list(definition.schema["properties"]["axes"]["properties"].keys()) == headers


def test_schema_is_strict_valid_and_ordered(definition):
    assert_schema_is_strict_valid(definition.schema)
    assert list(definition.schema["properties"].keys()) == [
        "axes", "other_implausibilities", "overall_reasoning", "overall_plausibility"]
    axis = next(iter(definition.schema["properties"]["axes"]["properties"].values()))
    assert list(axis["properties"].keys()) == ["applicable", "problems", "score_reasoning", "score"]
    assert axis["properties"]["problems"]["maxItems"] == 3
    assert list(axis["properties"]["problems"]["items"]["properties"].keys()) == [
        "quote", "english_gloss", "explanation", "severity"]
    assert axis["properties"]["problems"]["items"]["properties"]["english_gloss"]["type"] == ["string", "null"]
    assert axis["properties"]["score"]["enum"] == [1, 2, 3, 4, 5]
    other = definition.schema["properties"]["other_implausibilities"]
    assert other["maxItems"] == 3
    assert list(other["items"]["properties"].keys()) == [
        "suggested_axis_name", "description", "quote", "english_gloss", "explanation", "severity"]


# ---- validation rejects instead of storing ----

def test_valid_result_passes(definition):
    judge.validate_parsed_result(valid_parsed_result(definition.axis_keys), definition)


@pytest.mark.parametrize("bad_score", [0, 6, -1, 3.5, "4", None, True])
def test_out_of_range_axis_score_is_rejected(definition, bad_score):
    result = valid_parsed_result(definition.axis_keys)
    result["axes"][definition.axis_keys[3]]["score"] = bad_score
    with pytest.raises(ValueError):
        judge.validate_parsed_result(result, definition)


@pytest.mark.parametrize("bad_score", [0, 6, None])
def test_out_of_range_overall_score_is_rejected(definition, bad_score):
    with pytest.raises(ValueError):
        judge.validate_parsed_result(valid_parsed_result(definition.axis_keys, overall=bad_score), definition)


def test_missing_or_extra_axis_is_rejected(definition):
    result = valid_parsed_result(definition.axis_keys)
    del result["axes"][definition.axis_keys[0]]
    with pytest.raises(ValueError):
        judge.validate_parsed_result(result, definition)
    result = valid_parsed_result(definition.axis_keys)
    result["axes"]["axis_99_invented"] = valid_axis()
    with pytest.raises(ValueError):
        judge.validate_parsed_result(result, definition)


def test_bad_problem_entries_are_rejected(definition):
    for mutate in (lambda a: a["problems"].extend([problem()] * 4),
                   lambda a: a["problems"].append(problem(severity="catastrophic")),
                   lambda a: a["problems"].append({"quote": "x"})):
        result = valid_parsed_result(definition.axis_keys)
        mutate(result["axes"][definition.axis_keys[0]])
        with pytest.raises(ValueError):
            judge.validate_parsed_result(result, definition)


def test_invalid_response_becomes_error_record_not_stored_judgment(definition, monkeypatch):
    bad_result = valid_parsed_result(definition.axis_keys)
    bad_result["axes"][definition.axis_keys[0]]["score"] = 9

    class FakeResponse:
        output_text = json.dumps(bad_result)
        status = "completed"
        usage = type("U", (), {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": None,
                               "output_tokens_details": None})()

    class FakeClient:
        class responses:
            @staticmethod
            def create(**_):
                return FakeResponse()

    record, error = judge.judge_one_document(FakeClient, definition, {"doc_id": "d", "content": "x"}, "low")
    assert record is None and error["error_kind"] == "parse_or_schema_failure"
    assert error["usage"]["input_tokens"] == 20  # both attempts counted


# ---- resume ----

def test_resume_skips_finished_documents(tmp_path, monkeypatch, definition):
    (tmp_path / "judgments_2026-01-01_00-00.jsonl").write_text(json.dumps({"doc_id": "a", "usage": {}}) + "\n")
    completed = judge.find_completed_document_ids(tmp_path)
    assert completed == {"a"}
    calls = []

    def fake_judge(client, judge_definition, document, effort):
        calls.append(document["doc_id"])
        return {"doc_id": document["doc_id"], "usage": {"input_tokens": 1, "cached_input_tokens": 0,
                                                        "output_tokens": 1, "reasoning_tokens": 0}}, None

    monkeypatch.setattr(judge, "judge_one_document", fake_judge)
    documents = [{"doc_id": "a", "content": "x"}, {"doc_id": "b", "content": "y"}]
    judge.run_judging(None, tmp_path, "t", definition, documents, "low", 2, completed, [])
    assert calls == ["b"]


def test_resume_refuses_changed_prompt(tmp_path):
    configuration = judge.build_run_configuration("medium", judge.DEFAULT_CORPUS_PATH)
    tampered = copy.deepcopy(configuration)
    tampered["prompt_file_sha256"][judge.INSTRUCTIONS_FILE_NAME] = "0" * 64
    configuration_path = tmp_path / "run_configuration_2026-01-01_00-00.json"
    configuration_path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError):
        judge.check_resume_configuration_matches(tmp_path, configuration)
    configuration_path.write_text(json.dumps(configuration))
    judge.check_resume_configuration_matches(tmp_path, configuration)
    changed_effort = copy.deepcopy(configuration)
    changed_effort["reasoning_effort"] = "high"
    with pytest.raises(RuntimeError):
        judge.check_resume_configuration_matches(tmp_path, changed_effort)


def parse(*extra):
    return judge.parse_arguments(["--run-name", "x", *extra])


def test_run_folder_location(tmp_path):
    assert judge.choose_runs_parent_directory(parse(), tmp_path)[1] == tmp_path
    for extra in (["--estimate-only"], ["--document-ids", "a"], ["--document-limit", "5"]):
        assert judge.choose_runs_parent_directory(parse(*extra), tmp_path)[1] == tmp_path / "small_sample_checks"
    assert judge.RUNS_DIRECTORY.name == "plausibility_audit_runs"


def test_unknown_document_ids_exit_with_error(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(json.dumps({"doc_id": "known", "content": "x"}) + "\n")
    with pytest.raises(SystemExit) as exit_info:
        judge.main(["--run-name", "unknown_ids_check", "--corpus-path", str(corpus_path),
                    "--document-ids", "known", "missing_one", "--estimate-only"])
    assert "missing_one" in str(exit_info.value)


# ---- post-processing ----

def test_variables_parse_from_string_form():
    assert postprocess.parse_variables("{'tone': 'dry', 'domain': \"it's\", 'n': 3}") == {
        "tone": "dry", "domain": "it's", "n": 3}
    assert postprocess.parse_variables({"a": 1}) == {"a": 1}
    with pytest.raises(ValueError):
        postprocess.parse_variables("__import__('os').getcwd()")
    with pytest.raises(ValueError):
        postprocess.parse_variables("[1, 2]")


def test_quote_spanning_paragraph_break_is_flagged():
    document = {"content": "First paragraph ends here.\n\nSecond paragraph starts here.", "language": "English",
                "variables": {"document_type": "essay"}}
    spanning = {"quote": "ends here. Second paragraph starts", "english_gloss": None, "explanation": "e",
                "severity": "minor"}
    within = problem("First paragraph ends here.")
    row_spanning = postprocess.build_quoted_row("d", document, {}, {}, spanning)
    row_within = postprocess.build_quoted_row("d", document, {}, {}, within)
    assert not row_spanning["quote_found_in_document"] and row_spanning["quote_spans_paragraph_break"]
    assert row_within["quote_found_in_document"] and not row_within["quote_spans_paragraph_break"]


def build_synthetic_run(tmp_path, definition):
    axes = definition.axis_keys
    corpus = [
        {"doc_id": "good_doc", "content": "The fish may feel pain.\n\nA second paragraph.", "language": "English",
         "variables": "{'document_type': 'essay', 'tone': 'dry'}"},
        {"doc_id": "other_doc", "content": "Texto en español.", "language": "Spanish",
         "variables": {"document_type": "blog", "tone": "dry"}},
        {"doc_id": "failed_doc", "content": "x", "language": "English",
         "variables": {"document_type": "essay", "tone": "warm"}}]
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("\n".join(json.dumps(d) for d in corpus) + "\n")
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text(
        json.dumps({"corpus_path": str(corpus_path), "axis_keys": axes}))
    first = valid_parsed_result(axes, overall=2)
    first["axes"][axes[0]] = valid_axis(score=2, problems=[problem(severity="minor"),
                                                         problem(severity="breaks_believability")])
    first["axes"][axes[1]] = valid_axis(score=1, applicable=True)
    first["axes"][axes[5]] = valid_axis(score=5, applicable=False)
    first["other_implausibilities"] = [{"suggested_axis_name": "n", "description": "d",
                                        "quote": "A second paragraph.", "english_gloss": None,
                                        "explanation": "e", "severity": "noticeable"}]
    second = valid_parsed_result(axes, overall=4)
    second["axes"][axes[1]] = valid_axis(score=3)
    second["axes"][axes[5]] = valid_axis(score=5, applicable=False)
    records = [{"doc_id": "good_doc", "parsed_result": first, "usage": {}},
               {"doc_id": "other_doc", "parsed_result": second, "usage": {}}]
    (run_folder / "judgments_2026-01-01_00-00.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (run_folder / "errors_2026-01-01_00-00.jsonl").write_text(
        json.dumps({"doc_id": "failed_doc", "error_kind": "parse_or_schema_failure", "usage": {}}) + "\n")
    return run_folder, axes


def test_failed_documents_have_empty_cells_and_inapplicable_axes_are_excluded(tmp_path, definition):
    run_folder, axes = build_synthetic_run(tmp_path, definition)
    paths, rows = postprocess.postprocess_run_folder(run_folder)
    for path in paths.values():
        assert Path(path).exists(), path
    csv_rows = {r["doc_id"]: r for r in csv_module.DictReader(open(paths["per_document_scores_csv"], encoding="utf-8"))}
    failed = csv_rows["failed_doc"]
    assert failed["judgment_status"] == "failed" and failed["overall_plausibility"] == ""
    for axis_key in axes:
        for column in postprocess.axis_columns(axis_key):
            assert failed[column] == ""
    assert failed["tone"] == "warm"  # variables are joined even for failed documents
    good = csv_rows["good_doc"]
    assert good["overall_plausibility"] == "2" and good["tone"] == "dry"  # string-form variables parsed
    assert good[f"{axes[0]}__problem_count"] == "2" and good[f"{axes[0]}__worst_severity"] == "breaks_believability"
    assert good[f"{axes[2]}__worst_severity"] == "" and good[f"{axes[2]}__problem_count"] == "0"
    assert good[f"{axes[5]}__applicable"] == "False"

    summary = {r["axis"]: r for r in rows["axis_summary"]}
    assert summary[axes[5]]["documents_applicable"] == 0 and summary[axes[5]]["mean_score_among_applicable"] is None
    assert summary[axes[1]]["mean_score_among_applicable"] == 2.0  # (1 + 3) / 2
    assert summary[axes[0]]["problems_with_severity_minor"] == 1
    assert summary[axes[0]]["problems_with_severity_breaks_believability"] == 1
    assert summary[axes[1]]["applicable_documents_with_score_1"] == 1

    variable_means = {(r["variable"], r["value"]): r for r in rows["variable_means"]}
    assert ("tone", "warm") not in variable_means  # failed document contributes nothing
    dry = variable_means[("tone", "dry")]
    assert dry["document_count"] == 2 and dry["overall_plausibility_mean"] == 3.0
    assert dry[f"{axes[1]}__mean_score_among_applicable"] == 2.0
    assert dry[f"{axes[5]}__mean_score_among_applicable"] is None and dry[f"{axes[5]}__applicable_document_count"] == 0

    assert len(rows["problems"]) == 2 and rows["problems"][0]["axis"] == axes[0]
    assert rows["problems"][0]["tone"] == "dry" and rows["problems"][0]["quote_found_in_document"] is True
    assert len(rows["other"]) == 1 and rows["other"][0]["quote_found_in_document"] is True
