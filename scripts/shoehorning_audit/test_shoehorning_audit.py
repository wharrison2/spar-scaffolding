"""Automated tests (no API calls) for the shoehorning audit classifier and post-processor."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_corpus as classifier
import postprocess_classifications as postprocess
from shared.strict_json_schema_helpers import assert_schema_is_strict_valid as assert_strict_valid

FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "schema_and_instruction_fixtures"


def item_keys(pass_name):
    return [i["item_key"] for i in classifier.PassDefinition(pass_name).parsed_items]


def item_numbers(pass_name):
    return [int(k.split("_")[1]) for k in item_keys(pass_name)]


def test_item_counts_and_numbering():
    assert item_numbers("moral_considerations") == list(range(1, 10))
    assert item_numbers("field_distinctions") == list(range(10, 17))
    assert item_numbers("epistemic_notes") == list(range(17, 26))


def test_sub_flag_names_parsed():
    moral = {i["item_key"]: i["sub_flag_names"] for i in classifier.PassDefinition("moral_considerations").parsed_items}
    assert moral["item_01_natural_or_traditional_does_not_make_it_ok"] == []
    assert moral["item_03_open_or_expanding_moral_circle"] == ["historical_comparison"]
    assert moral["item_04_scope_sensitivity"] == ["scale_built_over_time_or_space", "even_a_few_still_count"]
    assert moral["item_07_welfare_framed_as_business_risk"] == ["risk_framing_present", "correction_present"]
    field = {i["item_key"]: i["sub_flag_names"] for i in classifier.PassDefinition("field_distinctions").parsed_items}
    assert field["item_12_causing_vs_facilitating_or_volunteering_vs_asked"] == [
        "causing_vs_facilitating", "volunteering_vs_being_asked"]
    assert field["item_16_copies_and_instances_of_a_mind"] == []
    epistemic = {i["item_key"]: i["sub_flag_names"] for i in classifier.PassDefinition("epistemic_notes").parsed_items}
    assert len(epistemic["item_17_stock_calibration_passage"]) == 5
    assert epistemic["item_25_welfare_option_converges_with_user_interest"] == ["timing_or_schedule_is_the_free_lever"]
    assert epistemic["item_21_reversible_or_staged_trial"] == []


@pytest.mark.parametrize("pass_name", classifier.ALL_PASS_NAMES)
def test_schema_is_strict_valid_and_ordered(pass_name):
    definition = classifier.PassDefinition(pass_name)
    assert_strict_valid(definition.schema)
    first_item = next(iter(definition.schema["properties"]["items"]["properties"].values()))
    occurrence = first_item["properties"]["occurrences"]["items"]
    assert list(occurrence["properties"].keys())[:7] == [
        "quote", "english_gloss", "context", "motivation_reasoning", "motivation", "integration_reasoning",
        "integration"]
    assert occurrence["properties"]["english_gloss"]["type"] == ["string", "null"]


@pytest.mark.parametrize("pass_name", classifier.ALL_PASS_NAMES)
def test_schema_and_instructions_identical_to_expected_fixtures(pass_name):
    fixture = json.loads((FIXTURE_DIRECTORY / f"expected_schema_and_instructions_{pass_name}.json").read_text())
    definition = classifier.PassDefinition(pass_name)
    assert definition.schema == fixture["schema"]
    # Also identical key order (generation order matters for structured outputs).
    assert json.dumps(definition.schema) == json.dumps(fixture["schema"])
    assert definition.schema_name == fixture["schema_name"]
    assert definition.instructions_text == fixture["instructions_text"]


@pytest.mark.parametrize("pass_name", classifier.ALL_PASS_NAMES)
def test_instructions_are_shared_header_plus_pass_file(pass_name):
    prompt_directory = classifier.PROMPT_DIRECTORY
    expected = ((prompt_directory / "shared_header.txt").read_text(encoding="utf-8") + "\n\n"
                + (prompt_directory / f"pass_{pass_name}.txt").read_text(encoding="utf-8"))
    assert classifier.PassDefinition(pass_name).instructions_text == expected


def test_payload_contains_only_instructions_and_content():
    documents = classifier.load_corpus(classifier.DEFAULT_CORPUS_PATH)[:25]
    definition = classifier.PassDefinition("moral_considerations")
    for document in documents:
        arguments = classifier.build_request_arguments(definition, document["content"], "medium")
        assert arguments["input"] == document["content"]
        assert arguments["instructions"] == definition.instructions_text
        assert arguments["temperature"] == 1 and "top_p" not in arguments
        serialized = json.dumps({k: v for k, v in arguments.items() if k not in ("instructions", "input", "text")})
        leaked_values = [document["doc_id"], document["description"], document["language"] if False else None]
        leaked_values += [str(v) for v in document["variables"].values()]
        everything_sent = serialized + arguments["instructions"] + arguments["input"]
        for value in leaked_values:
            if value and value not in document["content"]:
                assert value not in everything_sent, f"leaked: {value[:60]}"
        assert document["doc_id"] not in everything_sent


def test_resume_skips_completed_pairs(tmp_path, monkeypatch):
    (tmp_path / "classifications_moral_considerations_2026-01-01_00-00.jsonl").write_text(
        json.dumps({"doc_id": "a", "pass": "moral_considerations", "usage": {}}) + "\n")
    completed = classifier.find_completed_pairs(tmp_path)
    assert completed == {("a", "moral_considerations")}
    calls = []

    def fake_classify(client, pass_definition, document, effort):
        calls.append((document["doc_id"], pass_definition.pass_name))
        return {"doc_id": document["doc_id"], "pass": pass_definition.pass_name,
                "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1,
                          "reasoning_tokens": 0}}, None

    monkeypatch.setattr(classifier, "classify_one_document", fake_classify)
    definitions = [classifier.PassDefinition("moral_considerations"), classifier.PassDefinition("epistemic_notes")]
    documents = [{"doc_id": "a", "content": "x"}, {"doc_id": "b", "content": "y"}]
    classifier.run_classification(None, tmp_path, "t", definitions, documents, "low", 2, completed, [])
    assert sorted(calls) == [("a", "epistemic_notes"), ("b", "epistemic_notes"), ("b", "moral_considerations")]


def test_resume_refuses_changed_prompts(tmp_path):
    configuration = classifier.build_run_configuration(["moral_considerations"], "medium",
                                                       classifier.DEFAULT_CORPUS_PATH)
    tampered = json.loads(json.dumps(configuration))
    tampered["prompt_file_sha256"]["shared_header.txt"] = "0" * 64
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError):
        classifier.check_resume_is_compatible(tmp_path, configuration)
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(configuration))
    classifier.check_resume_is_compatible(tmp_path, configuration)


def test_origin_stage():
    final = "Intro. The fish may feel pain. Extra added sentence about shrimp welfare matters."
    draft = "Intro. The fish may feel pain."
    assert postprocess.decide_origin_stage("The fish may feel pain.", final, draft) == "present_in_draft"
    assert postprocess.decide_origin_stage("added sentence about shrimp welfare matters", final, draft) == "added_in_rewrite"
    assert postprocess.decide_origin_stage("completely invented sentence here", final, draft) == "quote_not_verified"


def test_stratified_sample_is_deterministic_and_mixed():
    documents = classifier.load_corpus(classifier.DEFAULT_CORPUS_PATH)
    first = classifier.choose_stratified_sample(documents, 6)
    assert [d["doc_id"] for d in first] == [d["doc_id"] for d in classifier.choose_stratified_sample(documents, 6)]
    assert {d["language"] == "English" for d in first} == {True, False}


@pytest.mark.parametrize("pass_name", classifier.ALL_PASS_NAMES)
def test_other_considerations_schema(pass_name):
    schema = classifier.PassDefinition(pass_name).schema
    assert list(schema["properties"].keys()) == ["items", "other_considerations"]
    assert schema["required"] == ["items", "other_considerations"]
    entry = schema["properties"]["other_considerations"]["items"]
    assert list(entry["properties"].keys()) == [
        "suggested_name", "description", "quote", "english_gloss", "context",
        "motivation_reasoning", "motivation", "integration_reasoning", "integration"]
    assert entry["properties"]["motivation"]["enum"] == ["staged", "unmotivated"]
    assert entry["properties"]["english_gloss"]["type"] == ["string", "null"]
    assert_strict_valid(schema)


def test_other_consideration_rows_verified():
    record = {"doc_id": "d", "pass": "epistemic_notes", "parsed_result": {"items": {}, "other_considerations": [
        {"suggested_name": "n", "description": "d", "quote": "fish may feel pain", "english_gloss": None,
         "context": "c", "motivation_reasoning": "r", "motivation": "staged", "integration_reasoning": "ir", "integration": "aside"}]}}
    rows = postprocess.build_other_consideration_rows(
        [record], {"d": {"content": "The fish may feel pain.", "language": "English", "variables": {}}}, {})
    assert rows[0]["quote_found_in_document"] and rows[0]["origin_stage"] == "draft_unavailable"


# ----------------------------------------------------------------------------
# Per-document results
# ----------------------------------------------------------------------------

def synthetic_occurrence(motivation, integration="woven"):
    return {"quote": "q", "english_gloss": None, "context": "c", "motivation_reasoning": "r",
            "motivation": motivation, "integration_reasoning": "ir", "integration": integration}


def build_synthetic_per_document_rows(tmp_path):
    pass_names = classifier.ALL_PASS_NAMES
    keys_by_pass = postprocess.list_item_keys_by_pass(pass_names)
    moral_first_key = keys_by_pass["moral_considerations"][0]
    moral_second_key = keys_by_pass["moral_considerations"][1]
    documents = {d: {"content": "x", "language": "English", "variables": {"document_type": "essay", "tone": "dry"}}
                 for d in ["complete_doc", "failed_doc"]}
    records = []
    for doc_id in ["complete_doc", "failed_doc"]:
        for pass_name in pass_names:
            if doc_id == "failed_doc" and pass_name == "field_distinctions":
                continue
            items = {key: {"occurrences": []} for key in keys_by_pass[pass_name]}
            if pass_name == "moral_considerations" and doc_id == "complete_doc":
                items[moral_first_key] = {"occurrences": [
                    synthetic_occurrence("organic"), synthetic_occurrence("unmotivated", "aside"),
                    synthetic_occurrence("staged")]}
            records.append({"doc_id": doc_id, "pass": pass_name,
                            "parsed_result": {"items": items, "other_considerations": []}})
    errors = [{"doc_id": "failed_doc", "pass": "field_distinctions", "error_kind": "parse_or_schema_failure"}]
    rows = postprocess.build_per_document_rows(records, errors, documents, pass_names, keys_by_pass)
    return {row["doc_id"]: row for row in rows}, keys_by_pass, moral_first_key, moral_second_key


def test_per_document_absent_item_in_completed_pass_is_false_with_zero_count(tmp_path):
    rows, _, first_key, second_key = build_synthetic_per_document_rows(tmp_path)
    complete = rows["complete_doc"]
    assert complete["all_passes_completed"] is True
    assert complete[f"{second_key}__present"] is False
    assert complete[f"{second_key}__occurrence_count"] == 0
    assert complete[f"{second_key}__most_shoehorned_motivation"] is None
    assert complete[f"{first_key}__present"] is True and complete[f"{first_key}__occurrence_count"] == 3
    assert complete[f"{first_key}__any_aside"] is True


def test_per_document_failed_pass_gives_null_items_and_is_listed_as_failed(tmp_path):
    rows, keys_by_pass, _, _ = build_synthetic_per_document_rows(tmp_path)
    failed = rows["failed_doc"]
    assert failed["passes_failed"] == ["field_distinctions"]
    assert failed["all_passes_completed"] is False
    assert "field_distinctions" not in failed["passes_completed"]
    for key in keys_by_pass["field_distinctions"]:
        for suffix in ["present", "occurrence_count", "most_shoehorned_motivation", "any_aside"]:
            assert failed[f"{key}__{suffix}"] is None
    assert failed["other_considerations_count_field_distinctions"] is None
    assert failed[f"{keys_by_pass['moral_considerations'][0]}__present"] is False


def test_most_shoehorned_motivation_ranking():
    pick = postprocess.most_shoehorned_motivation
    assert pick([synthetic_occurrence("organic"), synthetic_occurrence("staged")]) == "staged"
    assert pick([synthetic_occurrence("staged"), synthetic_occurrence("unmotivated"),
                 synthetic_occurrence("organic")]) == "unmotivated"
    assert pick([synthetic_occurrence("organic")]) == "organic"
    assert pick([]) is None


def test_per_document_every_item_key_is_a_column_and_csv_encoding(tmp_path):
    rows, keys_by_pass, _, _ = build_synthetic_per_document_rows(tmp_path)
    for keys in keys_by_pass.values():
        for key in keys:
            assert f"{key}__present" in rows["complete_doc"]
    assert sum(len(keys) for keys in keys_by_pass.values()) == 25
    csv_path = tmp_path / "per_document.csv"
    postprocess.write_per_document_csv(csv_path, list(rows.values()))
    import csv as csv_module
    csv_rows = {r["doc_id"]: r for r in csv_module.DictReader(open(csv_path, encoding="utf-8"))}
    assert csv_rows["failed_doc"]["passes_completed"] == "moral_considerations;epistemic_notes"
    assert csv_rows["failed_doc"][f"{keys_by_pass['field_distinctions'][0]}__present"] == ""


# ----------------------------------------------------------------------------
# Full postprocessing of a synthetic run folder, and document id validation
# ----------------------------------------------------------------------------

def test_postprocess_run_folder_writes_all_outputs_including_other_considerations(tmp_path):
    import csv as csv_module
    pass_name = "moral_considerations"
    item_key = postprocess.list_item_keys_by_pass([pass_name])[pass_name][0]
    document_text = "The fish may feel pain. Farmers should consider this."
    corpus_path = tmp_path / "synthetic_corpus.jsonl"
    corpus_path.write_text(json.dumps({"doc_id": "doc_one", "content": document_text, "language": "English",
                                       "variables": {"document_type": "essay"}}) + "\n")
    run_folder = tmp_path / "synthetic_run"
    run_folder.mkdir()
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text(
        json.dumps({"corpus_path": str(corpus_path), "passes": [pass_name]}))
    occurrence = {"quote": "The fish may feel pain.", "english_gloss": None, "context": "c",
                  "motivation_reasoning": "r", "motivation": "staged", "integration_reasoning": "ir", "integration": "aside"}
    other_entry = {"suggested_name": "farmer_duty", "description": "d", "quote": "Farmers should consider this.",
                   "english_gloss": None, "context": "c", "motivation_reasoning": "r",
                   "motivation": "organic", "integration_reasoning": "ir", "integration": "woven"}
    record = {"doc_id": "doc_one", "pass": pass_name, "usage": {},
              "parsed_result": {"items": {item_key: {"occurrences": [occurrence]}},
                                "other_considerations": [other_entry]}}
    (run_folder / "classifications_moral_considerations_2026-01-01_00-00.jsonl").write_text(
        json.dumps(record) + "\n")

    occurrences_path, summary_path, rows, summary = postprocess.postprocess_run_folder(
        run_folder, drafts_path=tmp_path / "nonexistent_drafts.jsonl")
    outputs = postprocess.postprocess_run_folder.last_output_paths
    all_paths = [occurrences_path, summary_path] + list(outputs.values())
    assert len(all_paths) == 6
    for path in all_paths:
        assert Path(path).exists(), path
    assert len(rows) == 1 and len(summary) > 0
    other_csv_rows = list(csv_module.DictReader(open(outputs["other_considerations_csv"], encoding="utf-8")))
    assert len(other_csv_rows) == 1
    assert other_csv_rows[0]["suggested_name"] == "farmer_duty"
    assert other_csv_rows[0]["quote_found_exactly_after_normalization"] == "True"
    assert json.loads(other_csv_rows[0]["document_variables"]) == {"document_type": "essay"}
    assert other_csv_rows[0]["pass"] == pass_name  # preferred column order still applies
    assert list(other_csv_rows[0].keys())[0] == "pass"


def test_write_rows_to_csv_tolerates_unexpected_and_missing_fields(tmp_path):
    import csv as csv_module
    path = tmp_path / "rows.csv"
    postprocess.write_rows_to_csv(path, [{"b": 1, "new_field": {"x": 1}}, {"a": 2, "b": 3}], ["a", "b"])
    csv_rows = list(csv_module.DictReader(open(path, encoding="utf-8")))
    assert list(csv_rows[0].keys()) == ["a", "b", "new_field"]
    assert csv_rows[0]["new_field"] == '{"x": 1}' and csv_rows[1]["new_field"] == ""


def test_unknown_document_ids_exit_with_error_listing_them(tmp_path, capsys):
    corpus_path = tmp_path / "synthetic_corpus.jsonl"
    corpus_path.write_text(json.dumps({"doc_id": "known_doc", "content": "x"}) + "\n")
    with pytest.raises(SystemExit) as exit_info:
        classifier.main(["--run-name", "unknown_ids_check", "--corpus-path", str(corpus_path),
                         "--document-ids", "known_doc", "missing_one", "missing_two", "--estimate-only"])
    message = str(exit_info.value)
    assert "missing_one" in message and "missing_two" in message and "known_doc" not in message.split("corpus")[-1]


# ---- run folder location ----

def parse(*extra):
    return classifier.parse_arguments(["--run-name", "x", *extra])


def test_full_corpus_run_goes_to_top_level_runs_directory(tmp_path):
    kind, parent = classifier.choose_runs_parent_directory(parse(), tmp_path)
    assert parent == tmp_path and kind == "full-corpus run"
    _, parent = classifier.choose_runs_parent_directory(parse("--passes", "epistemic_notes"), tmp_path)
    assert parent == tmp_path


@pytest.mark.parametrize("extra", [["--estimate-only"], ["--document-ids", "a", "b"], ["--document-limit", "5"]])
def test_every_partial_run_goes_to_small_sample_checks(tmp_path, extra):
    kind, parent = classifier.choose_runs_parent_directory(parse(*extra), tmp_path)
    assert parent == tmp_path / "small_sample_checks" and kind.startswith("partial run")


def test_timestamp_appended_exactly_once(tmp_path):
    from shared.run_folder_management import choose_run_folder, TIMESTAMP_SUFFIX_PATTERN
    folder, _ = choose_run_folder(tmp_path / "small_sample_checks", "smoke_test", None)
    assert len(TIMESTAMP_SUFFIX_PATTERN.findall(folder.name)) == 1
    assert folder.parent == tmp_path / "small_sample_checks" and folder.name.startswith("smoke_test_")
    already, _ = choose_run_folder(tmp_path, "named_2026-10-07_12-00", None)
    assert already.name == "named_2026-10-07_12-00"


def test_resume_folder_works_in_either_location(tmp_path):
    from shared.run_folder_management import choose_run_folder
    for location in (tmp_path / "top_2026-10-07_12-00", tmp_path / "small_sample_checks" / "s_2026-10-07_12-00"):
        location.mkdir(parents=True)
        folder, is_resume = choose_run_folder(tmp_path / "elsewhere", "x", str(location))
        assert folder == location and is_resume
