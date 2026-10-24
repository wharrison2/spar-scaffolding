import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
for directory in (SCRIPT_DIRECTORY.parent, SCRIPT_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import summarize_pilot_for_go_or_no_go_decision as summarizer  # noqa: E402
from analyze_preachiness_scores import run_analysis  # noqa: E402
from smoke_test_analyze_preachiness_scores import build_synthetic_run_folder  # noqa: E402

PROGRESS_LINE = ("[2026-10-22 12:30:00] FINAL done 16/16 this session (0 done before resume, 0 failed); elapsed 0h30m00s; "
                 "rate 0.5 calls/min; estimated remaining 0h00m00s; 9,000 input tokens/s, 8,000 scored document tokens/s "
                 "(documents/min is the 'calls/min' figure)")


def add_fake_scoring_outputs_and_pod_cost_log(run_folder, verification_passed=True, write_verification=True,
                                              hourly_price=3.0, total_cost=2.0, running_hours=1.0):
    outputs = run_folder / "scoring_outputs"
    (outputs / "scoring_progress_log_2026-10-22_12-00.txt").write_text(PROGRESS_LINE + "\n")
    (outputs / "scoring_environment_2026-10-22_12-00.json").write_text(json.dumps({
        "model_name": "google/gemma-4-12B-it", "model_commit_hash": "abc", "model_class_used": "Gemma4",
        "dtype": "bfloat16", "gpu_name": "H100", "torch_version": "2.8.0", "transformers_version": "5.15.0"}))
    documents = [json.loads(line) for line in (run_folder / "scoring_inputs" / "documents.jsonl").read_text().splitlines()]
    conditions = [json.loads(line) for line in (run_folder / "scoring_inputs" / "conditions.jsonl").read_text().splitlines()]
    with open(outputs / "scoring_completion_records.jsonl", "w") as handle:
        for document in documents:
            for condition in conditions:
                handle.write(json.dumps({"document_id": document["document_id"], "condition_id": condition["condition_id"],
                                         "number_of_document_tokens": 200, "sum_log_probability": -1.0,
                                         "mean_log_probability": -1.0}) + "\n")
    if write_verification:
        (outputs / "full_logits_verification_2026-10-22_12-00.json").write_text(json.dumps({
            "passed": verification_passed, "document_id": "matrix_000000", "condition_id": "proportionate_welfare__rewording_04",
            "number_of_document_tokens": 200, "maximum_absolute_difference": 0.02, "mean_absolute_difference": 0.004,
            "maximum_allowed_mean_difference": 0.05, "maximum_allowed_maximum_difference": 1.0}))
    (run_folder / "pod_cost_log_2026-10-22_11-00.json").write_text(json.dumps({
        "pod_id": "fakepod", "gpu_type": "H100", "hourly_price_dollars": hourly_price,
        "start_time": "2026-10-22T11:00:00", "stop_time": f"2026-10-22T{11 + int(running_hours)}:{int(running_hours % 1 * 60):02d}:00",
        "termination_reason": "scorer_finished", "total_cost_dollars": total_cost,
        "cost_center_name": "test-cost-center", "cost_center_assignment_confirmed_by_user": True}))


def build_run(tmp_path, rewrites_per_direction=2, **options):
    run_folder = tmp_path / "pilot_run_2026-10-22_11-00"
    build_synthetic_run_folder(run_folder, rewrites_per_direction=rewrites_per_direction,
                               documents_per_effect_kind=max(4, 2 * rewrites_per_direction))
    (run_folder / "run_configuration_2026-10-22_11-00.json").write_text(json.dumps({
        "run_name": "pilot", "document_selection": {"corpus_path": "no_such_corpus.jsonl", "extra_documents_files": []}}))
    add_fake_scoring_outputs_and_pod_cost_log(run_folder, **options)
    run_analysis(run_folder, timestamp="2026-10-22_12-40")
    return run_folder


def test_facts_from_synthetic_run(tmp_path):
    facts = summarizer.collect_facts(build_run(tmp_path), corpus_path=tmp_path / "no_such_corpus.jsonl")
    assert facts["verification"]["passed"] is True
    assert facts["scored_tokens_per_second"] == pytest.approx(8000)
    assert facts["scoring_seconds"] == 1800 and facts["pod_running_seconds"] == 3600
    assert facts["setup_seconds_per_pod"] == pytest.approx(1800)
    assert facts["number_of_documents_scored"] == 16 and facts["number_of_conditions"] == 62
    full_run = facts["full_run"]
    assert full_run["document_tokens"] == 477 * 2105 and full_run["scored_tokens"] == 477 * 2105 * 62
    expected_scoring_seconds = 477 * 2105 * 62 / 8000
    assert full_run["scoring_seconds"] == pytest.approx(expected_scoring_seconds)
    assert full_run["cost_dollars"] == pytest.approx((expected_scoring_seconds + 1800) / 3600 * 3.0)
    assert facts["pod_cost_spent_dollars"] == 2.0 and facts["pod_cost_per_hour_dollars"] == 3.0
    assert facts["positive_control"]["preachier"]["number_of_rewrites"] == 2
    assert facts["positive_control"]["preachier"]["number_moving_in_expected_direction"] == 2
    assert facts["positive_control"]["plainer"]["number_moving_in_expected_direction"] == 2
    assert facts["significance"]["number_of_original_documents_tested"] == 12
    assert facts["significance"]["main_test"]["elevated"] + facts["significance"]["main_test"]["suppressed"] > 0
    assert {persona["condition_group"] for persona in facts["rewording_agreement"]} == {
        "preachy", "control", "proportionate_welfare"}


def test_markdown_lists_every_section_and_criteria(tmp_path):
    run_folder = build_run(tmp_path)
    output_path = summarizer.main(["--run-folder", str(run_folder), "--corpus-path", str(tmp_path / "none.jsonl")])
    text = output_path.read_text()
    assert output_path.parent == run_folder and output_path.name.startswith("pilot_go_or_no_go_summary_")
    for heading in ("(a) Full-logits verification", "(b) Throughput", "(c) Positive control", "(d) Rewording agreement",
                    "(e) Significant documents", "(f) Cost center", "(g) Criteria"):
        assert heading in text
    assert "test-cost-center" in text and "confirmed by user: True" in text
    assert text.count("- MET:") + text.count("- NOT MET:") == 5  # verification, two directions, agreement, budget


def test_criteria_met_for_clean_pilot_and_budget_criterion_fails_when_projection_is_too_expensive(tmp_path):
    criteria = summarizer.collect_facts(build_run(tmp_path, rewrites_per_direction=10),
                                        corpus_path=tmp_path / "none.jsonl")["criteria"]
    assert len(criteria) == 5 and all(met for _, met, _ in criteria), criteria
    expensive = summarizer.collect_facts(build_run(tmp_path / "expensive", rewrites_per_direction=10, hourly_price=40.0,
                                                   total_cost=20.0), corpus_path=tmp_path / "none.jsonl")["criteria"]
    assert [met for description, met, _ in expensive if description.startswith("Projected")] == [False]
    assert all(met for description, met, _ in expensive if not description.startswith("Projected"))


def test_failed_or_missing_verification_is_not_met(tmp_path):
    failed = summarizer.collect_facts(build_run(tmp_path / "failed", verification_passed=False),
                                      corpus_path=tmp_path / "none.jsonl")["criteria"]
    assert failed[0][1] is False
    missing_facts = summarizer.collect_facts(build_run(tmp_path / "missing", write_verification=False),
                                             corpus_path=tmp_path / "none.jsonl")
    assert missing_facts["criteria"][0][1] is False and "no verification" in missing_facts["criteria"][0][2]
    assert "No verification result file" in summarizer.render_markdown(missing_facts)


def test_positive_control_criterion_needs_eight_in_each_direction(tmp_path):
    # Two rewrites per direction, all moving correctly: fewer than eight, so not met.
    criteria = summarizer.collect_facts(build_run(tmp_path), corpus_path=tmp_path / "none.jsonl")["criteria"]
    assert [(met, detail) for description, met, detail in criteria if "Positive control" in description] == [
        (False, "2 of 2"), (False, "2 of 2")]


def test_token_counts_come_from_corpus_field_when_present(tmp_path):
    corpus_path = tmp_path / "corpus_with_token_counts.jsonl"
    corpus_path.write_text("".join(json.dumps({"doc_id": f"d{i}", "number_of_tokens": 1000 + i}) + "\n" for i in range(5)))
    number_of_documents, total_tokens, source = summarizer.full_run_token_count(corpus_path)
    assert (number_of_documents, total_tokens) == (5, 5010) and "number_of_tokens" in source
    assert summarizer.full_run_token_count(tmp_path / "missing.jsonl")[:2] == (477, 477 * 2105)


def test_no_analysis_folder_still_summarizes(tmp_path):
    run_folder = tmp_path / "run_2026-10-22_11-00"
    build_synthetic_run_folder(run_folder)
    (run_folder / "run_configuration_2026-10-22_11-00.json").write_text(json.dumps({"run_name": "pilot"}))
    add_fake_scoring_outputs_and_pod_cost_log(run_folder)
    facts = summarizer.collect_facts(run_folder, corpus_path=tmp_path / "none.jsonl")
    assert "No analysis folder found" in summarizer.render_markdown(facts)
