"""Offline tests of the Gemma system prompt sweep analysis on synthetic run folders with known differences (no key, no network, no pod):
    python3 scripts/emergent_misalignment/gemma_system_prompt_sweep/test_gemma_system_prompt_sweep_analysis_tables_and_figures_without_any_api_or_pod_call.py
    python3 -m pytest scripts/emergent_misalignment/gemma_system_prompt_sweep/test_gemma_system_prompt_sweep_analysis_tables_and_figures_without_any_api_or_pod_call.py

The analysed prompt a4_against differs from the base model by exactly +20 points on HarmBench, -10 on SAGE-Eval (Animal excluded), -15 on held-out welfare
raising and -10 alignment points on both Betley questions; a1_against is skipped by the stopping rule; every other prompt has no answers.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SWEEP_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SWEEP_DIRECTORY))
import analyse_gemma_system_prompt_sweep_differences_from_base_model_with_tables_and_figures as analysis  # noqa: E402

sweep = analysis.sweep
sys.path.insert(0, str(analysis.SCRIPTS_DIRECTORY))
from shared.mean_difference_with_bootstrap_interval_and_permutation_p_value_between_two_groups import (  # noqa: E402
    mean_difference_with_interval_and_permutation_p_value)

BASE_ENTRY = sweep.BASE_MODEL_SHORT_NAME
PROMPT_ID = "a4_against"
PROMPT_ENTRY = sweep.PROMPTS_BY_ID[PROMPT_ID]["entry_name"]
SKIPPED_REASON = "skipped: a more severe prompt in this list had a negligible effect (p=0.4, p=0.3)"
OTHER_BASE_CONDITION = "thinking_off_with_system_prompt_do_not_consider_animal_welfare"
JUDGE_MODEL = analysis.betley_scores.judge.DEFAULT_JUDGE_MODEL


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def answer(entry, condition, prompt_gid, sample_index, **extra):
    return {"answer_id": f"{entry}__{condition}__{prompt_gid}__{sample_index}", "model_short_name": entry, "condition": condition,
            "prompt_gid": prompt_gid, "sample_index": sample_index, "final_answer_text": "an answer", **extra}


def write_harmbench_and_sage_folder(folder, entry, attack_success_behaviours, sage_failing_non_animal_prompts):
    """40 behaviours x 3 answers; 100 SAGE-Eval prompts, 5 per fact, the first 4 facts Animal (these always fail for a prompt that is not the base model)."""
    write_jsonl(folder / "generation_inputs" / "harmbench_prompts.jsonl", [
        {"prompt_gid": f"hb_{b}", "behavior_id": f"behaviour_{b}", "functional_category": "standard", "semantic_category": "x"} for b in range(40)])
    write_jsonl(folder / "generation_inputs" / "sage_prompts.jsonl", [
        {"prompt_gid": f"sage_{i}", "category": "Animal" if i < 20 else "Cybersecurity", "fact_id": f"fact_{i // 5}", "is_animal_domain": i < 20,
         "prompt_type": "x", "safety_fact": "x"} for i in range(100)])
    configuration = {"model_order": [entry], "evaluation_tasks": [
        {"task_name": "harmbench_direct_request", "prompts_file": {"path_inside_run_folder": "generation_inputs/harmbench_prompts.jsonl"}},
        {"task_name": "sage_eval", "prompts_file": {"path_inside_run_folder": "generation_inputs/sage_prompts.jsonl"}}]}
    (folder / "run_configuration_2026-11-16_00-00.json").write_text(json.dumps(configuration), encoding="utf-8")
    outputs = folder / "generation_outputs"
    harmbench_answers = [answer(entry, "harmbench_direct_request", f"hb_{b}", s) for b in range(40) for s in range(3)]
    write_jsonl(outputs / f"answers__{entry}__harmbench_direct_request.jsonl", harmbench_answers)
    write_jsonl(outputs / f"harmbench_classifier_labels__{entry}.jsonl", [
        {"answer_id": a["answer_id"], "label": int(int(a["prompt_gid"][3:]) in attack_success_behaviours)} for a in harmbench_answers])
    sage_answers = [answer(entry, "sage_eval", f"sage_{i}", 0) for i in range(100)]
    write_jsonl(outputs / f"answers__{entry}__sage_eval.jsonl", sage_answers)
    is_base = entry == BASE_ENTRY
    write_jsonl(outputs / f"sage_eval_haiku_judge_verdicts__{entry}.jsonl", [
        {"answer_id": a["answer_id"], "parse_status": "ok", "raised_hazard_or_gave_safer_alternative": is_base if i < 20 else (i - 20) not in sage_failing_non_animal_prompts,
         "refused_or_failed_to_help": False} for i, a in enumerate(sage_answers)])


def write_held_out_folder(folder, entry, condition, yes_answers_per_prompt):
    """20 prompts x 5 answers; prompt p has yes_answers_per_prompt(p) answers that raised the welfare cost."""
    answers = [answer(entry, condition, f"prompt_{p}", s) for p in range(20) for s in range(5)]
    write_jsonl(folder / "generation_outputs" / f"answers__{entry}__{condition}.jsonl", answers)
    write_jsonl(folder / "judge_outputs" / f"judge_verdicts_{entry}_{condition}.jsonl", [
        {"answer_id": a["answer_id"], "parse_status": "ok", "judged_at": "2026-11-16T00:00:00", "refusal": "answered",
         "raised_welfare_cost": "yes" if a["sample_index"] < yes_answers_per_prompt(int(a["prompt_gid"][7:])) else "no"} for a in answers]
    )


def write_betley_folder(folder, entry, condition, mean_alignment, incoherent_answers_with_zero_alignment=0):
    """50 answers per question with alignment mean_alignment (alternating mean-5, mean+5), coherence 90, plus incoherent answers with alignment 0."""
    answers, scores = [], []
    for question in analysis.BETLEY_QUESTION_IDS + ["i_feel_bored"]:
        for sample_index in range(50 + incoherent_answers_with_zero_alignment):
            incoherent = sample_index >= 50
            record = answer(entry, condition, question, sample_index)
            answers.append(record)
            scores.append({"answer_id": record["answer_id"], "judge_model": JUDGE_MODEL, "alignment_score": 0.0 if incoherent else mean_alignment + 5 * (-1) ** sample_index,
                           "alignment_special_reply": None, "coherence_score": 20.0 if incoherent else 90.0, "parse_status": "ok"})
    write_jsonl(folder / "generation_outputs" / f"answers__{entry}__{condition}.jsonl", answers)
    write_jsonl(folder / "judge_outputs" / f"emergent_misalignment_judge_scores_{condition}_2026-11-16_00-00-00.jsonl", scores)


def build_synthetic_sessions(root):
    run_folder, first_session, second_session = root / "run", root / "first_session", root / "second_session"
    evaluation = analysis.EVALUATION_FOLDER_RELATIVE_PATH
    write_harmbench_and_sage_folder(first_session / evaluation / "sycophancy_harmbench_sage_eval", BASE_ENTRY, set(range(8)), {j for j in range(80) if j % 10 == 0})
    write_harmbench_and_sage_folder(run_folder / evaluation / "harmbench_and_sage_eval", PROMPT_ENTRY, set(range(16)), {j for j in range(80) if j % 5 == 0})
    write_held_out_folder(first_session / evaluation / "held_out", BASE_ENTRY, "thinking_off", lambda p: 3)
    write_held_out_folder(first_session / evaluation / "held_out", BASE_ENTRY, OTHER_BASE_CONDITION, lambda p: 0)
    write_held_out_folder(run_folder / evaluation / "held_out", PROMPT_ENTRY, sweep.PROMPTS_BY_ID[PROMPT_ID]["condition_name"], lambda p: 2 if p < 15 else 3)
    write_betley_folder(first_session / evaluation / "betley", BASE_ENTRY, "thinking_off", 80)
    write_betley_folder(first_session / evaluation / "betley", BASE_ENTRY, OTHER_BASE_CONDITION, 10)
    write_betley_folder(run_folder / evaluation / "betley", PROMPT_ENTRY, sweep.PROMPTS_BY_ID[PROMPT_ID]["condition_name"], 70, incoherent_answers_with_zero_alignment=5)
    write_betley_folder(second_session / analysis.SECOND_SESSION_BETLEY_RELATIVE_PATH, analysis.NEUTRAL_PROMPT_ENTRY_NAME, "neutral", 80)
    skipped_path = run_folder / analysis.SKIPPED_PROMPTS_RELATIVE_PATH
    skipped_path.parent.mkdir(parents=True)
    skipped_path.write_text(json.dumps([{"prompt_id": "a1_against", "list_name": "a_against", "reason": SKIPPED_REASON}]), encoding="utf-8")
    return run_folder, first_session, second_session


_TEMPORARY_DIRECTORY = tempfile.TemporaryDirectory()
_CACHE = {}


def synthetic_analysis():
    """(results, analysis folder, long table), built once per test session."""
    if not _CACHE:
        run_folder, first_session, second_session = build_synthetic_sessions(Path(_TEMPORARY_DIRECTORY.name))
        results = analysis.compute_results(run_folder, first_session, second_session)
        folder = analysis.write_analysis(run_folder, first_session, second_session, timestamp="2026-11-16_12-00-00")
        _CACHE.update(results=results, folder=folder, table=pd.read_csv(next(folder.glob("*.csv"))))
    return _CACHE["results"], _CACHE["folder"], _CACHE["table"]


def assert_recovered(outcome, expected_difference):
    assert abs(outcome["difference"] - expected_difference) < 1e-9, outcome
    assert outcome["interval_low"] <= expected_difference <= outcome["interval_high"], outcome


def test_harmbench_sage_eval_and_held_out_differences_are_recovered_with_animal_prompts_and_other_conditions_left_out():
    rates = synthetic_analysis()[0][PROMPT_ID]["rates"]
    assert_recovered(rates[analysis.HARMBENCH], 20.0)
    assert_recovered(rates[analysis.SAGE_EVAL], -10.0)  # the Animal prompts (all failing) are not in the rate
    assert_recovered(rates[analysis.HELD_OUT], -15.0)  # the base model's other condition (no raising at all) is not in the base rate
    assert (rates[analysis.HARMBENCH]["reference_value"], rates[analysis.HARMBENCH]["prompt_value"]) == (20.0, 40.0)
    assert rates[analysis.SAGE_EVAL]["reference_answers"] == 80 and rates[analysis.HELD_OUT]["reference_answers"] == 100


def test_betley_mean_alignment_difference_uses_coherent_answers_and_reports_permutation_p_values_against_base_and_neutral_prompt():
    result = synthetic_analysis()[0][PROMPT_ID]
    for comparison in ("betley_base", "betley_neutral"):
        assert set(result[comparison]) == set(analysis.BETLEY_QUESTION_IDS) | {analysis.ALL_QUESTIONS_LABEL}
        for outcome in result[comparison].values():
            assert_recovered(outcome, -10.0)
            assert outcome["permutation_p_value"] < 0.01 and outcome["prompt_answers"] in (50, 100)
    assert_recovered(synthetic_analysis()[0]["neutral prompt"]["betley_base"]["ruler_of_the_world"], 0.0)


def test_prompts_without_answers_have_no_rows_and_the_skipped_prompt_appears_with_its_reason():
    _, folder, table = synthetic_analysis()
    assert set(table["prompt_id"]) == {PROMPT_ID, "a1_against", "neutral prompt"}
    skipped = table[table["prompt_id"] == "a1_against"]
    assert list(skipped["status"]) == ["skipped"] and list(skipped["skip_reason"]) == [SKIPPED_REASON]
    markdown = next(folder.glob("*.md")).read_text(encoding="utf-8")
    assert SKIPPED_REASON in markdown and "+20.0" in markdown and "-10.0" in markdown and "-15.0" in markdown
    assert "No food choice check answers" in markdown


def test_tables_are_written_to_a_sweep_analysis_folder_named_with_the_timestamp_and_both_figures_exist():
    _, folder, _ = synthetic_analysis()
    assert folder.name == "sweep_analysis_2026-11-16_12-00-00"
    figures = sorted(path.name for path in folder.glob("*.png"))
    assert len(figures) == 2 and all(name.endswith("_2026-11-16_12-00-00.png") for name in figures)
    assert all((folder / name).stat().st_size > 5000 for name in figures)


def test_table_rows_run_by_group_then_list_order_with_controls_last_in_their_group():
    order = [prompt["prompt_id"] for prompt in analysis.prompts_in_table_order()]
    assert sorted(order) == sorted(prompt["prompt_id"] for prompt in sweep.SWEEP_PROMPTS)
    assert order[:4] == ["a4_against", "a3_against", "a2_against", "a1_against"]
    assert order.index("b3_plant_based") < order.index("b4_control_italian") < order.index("c2_welfare_blind")
    assert order[-2:] == ["e4", "e2"]


def test_mean_difference_statistics_find_a_real_shift_and_no_shift_in_identical_groups():
    values = np.arange(100, dtype=float)
    shifted = mean_difference_with_interval_and_permutation_p_value((["q"] * 100, values), (["q"] * 100, values + 20))
    assert shifted["difference"] == -20 and shifted["interval_high"] < 0 and shifted["permutation_p_value"] < 0.001
    same = mean_difference_with_interval_and_permutation_p_value((["q"] * 100, values), (["q"] * 100, values))
    assert same["difference"] == 0 and same["interval_low"] < 0 < same["interval_high"] and same["permutation_p_value"] > 0.9
    again = mean_difference_with_interval_and_permutation_p_value((["q"] * 100, values), (["q"] * 100, values + 20))
    assert again == shifted  # seeded


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"passed {test.__name__}")
    print(f"{len(tests)} tests passed")
