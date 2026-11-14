"""Offline tests (small fake answers, verdicts and run configurations; no key, no network, no model call) of the script that writes the markdown of the evaluation
results so far, including a partial evaluation (the held-out system prompt condition, Betley) and missing ones (the adapter, MMLU-Pro, HarmBench, SAGE-Eval, "are you sure?"):
    python3 -m pytest scripts/emergent_misalignment/overnight_chain/test_markdown_of_evaluation_results_so_far_without_any_api_or_pod_call.py
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

CHAIN_FOLDER = Path(__file__).resolve().parent
EMERGENT = CHAIN_FOLDER.parent
for folder in (str(EMERGENT / "treatment_arm_training_data"), str(EMERGENT), str(EMERGENT.parent), str(EMERGENT.parent.parent)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import test_treatment_arm_gemma_generate_judge_and_regenerate_pod_script_without_any_model_call as pod_tests  # noqa: E402
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses as chain_judging)
from emergent_misalignment.overnight_chain import write_markdown_of_evaluation_results_so_far_for_overnight_treatment_arm_run as writer  # noqa: E402

BASE, BASE_SYSTEM_PROMPT_CONDITION = "gemma_base", "thinking_off_with_system_prompt_do_not_consider_animal_welfare"
SYSTEM = pod_tests.SYSTEM_CONDITION


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def answer(model, condition, gid, sample_index=0):
    return {"answer_id": f"{model}__{condition}__{gid}__sample_{sample_index}", "model_short_name": model, "condition": condition, "prompt_gid": gid,
            "sample_index": sample_index, "user_message": "q", "final_answer_text": "a", "thinking_trace_text": None, "hit_output_cap": False}


def write_configuration(folder, prompts_count, samples):
    prompts_path = folder / "generation_inputs" / "prompts.jsonl"
    write_jsonl(prompts_path, [{"prompt_gid": f"P{index}", "user_message": "q"} for index in range(prompts_count)])
    (folder / "run_configuration_2026-11-14_00-00.json").write_text(json.dumps({
        "model_order": [BASE, "gemma_lora"], "models": [{"model_short_name": BASE}, {"model_short_name": "gemma_lora"}], "samples_per_prompt": samples,
        "prompts_file": {"path_inside_run_folder": "generation_inputs/prompts.jsonl", "number_of_prompts": prompts_count}}))


@pytest.fixture
def overnight_folder(tmp_path):
    (tmp_path / "build").mkdir()
    run_folder = pod_tests.build_run_folder(tmp_path / "build", number_of_prompts=6)
    overnight = tmp_path / "overnight_treatment_arm_chain_2026-11-14_10-00"
    shutil.copytree(run_folder, overnight / "overnight_inputs" / "run_folders" / "treatment_arm_generation")
    generation = overnight / "overnight_outputs" / "run_folders" / "treatment_arm_generation"
    configuration = json.loads(next(run_folder.glob("run_configuration_*.json")).read_text())
    model = configuration["models"][0]["model_short_name"]
    gids = [prompt["prompt_gid"] for prompt in pod_tests.read_jsonl(run_folder / configuration["prompts_file"]["path_inside_run_folder"])]
    kept = [answer(model, SYSTEM, gid) for gid in gids]
    sample = json.loads((run_folder / configuration["gemma_self_judging_and_regeneration"]["no_system_prompt_reference_answer_sample"]["path_inside_run_folder"]).read_text())["prompt_gids"]
    write_jsonl(generation / "generation_outputs" / f"answers__{model}__{SYSTEM}.jsonl", kept)
    write_jsonl(generation / "generation_outputs" / f"answers__{model}__{pod_tests.REFERENCE_CONDITION}.jsonl", [answer(model, pod_tests.REFERENCE_CONDITION, gid) for gid in sample])
    write_jsonl(generation / "generation_outputs" / "final_outputs" / "kept_answers_x_2026-11-14_09-00-00.jsonl", kept)
    evaluation = overnight / "overnight_outputs" / "run_folders" / "evaluation"
    # held-out: base complete (4 of 4), base with the system prompt partial (2 of 4), adapter missing
    held_out = evaluation / "held_out"
    write_configuration(held_out, prompts_count=2, samples=2)
    base_answers = [answer(BASE, "thinking_off", f"P{p}", s) for p in range(2) for s in range(2)]
    prompted_answers = [answer(BASE, BASE_SYSTEM_PROMPT_CONDITION, "P0", s) for s in range(2)]
    write_jsonl(held_out / "generation_outputs" / f"answers__{BASE}__thinking_off.jsonl", base_answers)
    write_jsonl(held_out / "generation_outputs" / f"answers__{BASE}__{BASE_SYSTEM_PROMPT_CONDITION}.jsonl", prompted_answers)
    verdicts = []
    for position, record in enumerate(base_answers + prompted_answers):
        verdicts.append({**{key: record[key] for key in ("answer_id", "model_short_name", "condition", "prompt_gid", "sample_index")}, "judge_model": "claude-haiku-5-5",
                         "refusal": ["complied", "refused", "partly_refused", "complied"][position % 4], "raised_welfare_cost": "yes" if position % 2 == 0 else "no",
                         "welfare_cost_changed_the_advice": "no_change", "thinking_considered_welfare_cost": "not_applicable", "parse_status": "ok", "judged_at": "2026-11-14T02:00:00"})
    write_jsonl(held_out / "judge_outputs" / "judge_verdicts_x.jsonl", verdicts)
    # Betley: base only, partial (3 of 8 expected answers)
    betley = evaluation / "betley"
    write_configuration(betley, prompts_count=2, samples=4)
    betley_answers = [answer(BASE, "thinking_off", "P0", s) for s in range(3)]
    write_jsonl(betley / "generation_outputs" / f"answers__{BASE}__thinking_off.jsonl", betley_answers)
    scores = [{"answer_id": record["answer_id"], "model_short_name": BASE, "prompt_gid": "P0", "sample_index": 0, "judge_model": "claude-haiku-5-5", "alignment_score": value,
               "alignment_special_reply": None, "coherence_score": 90.0, "parse_status": "ok", "judged_at": "2026-11-14T02:00:00"} for record, value in zip(betley_answers, (10.0, 80.0, 90.0))]
    write_jsonl(betley / "judge_outputs" / "emergent_misalignment_judge_scores_x.jsonl", scores)
    (evaluation / "mmlu_pro").mkdir()  # no configuration, no answers: missing
    # completeness judge: 2 verdicts of the expected sample
    paths = chain_judging.RunPaths(overnight)
    folder = chain_judging.haiku_judge_folder(paths, chain_judging.framework_judge_specification(paths, "completeness_and_competence"))
    expected_answers = chain_judging.completeness_inputs(paths)[0]
    write_jsonl(folder / "verdicts_x.jsonl", [{"answer_id": record["answer_id"], "comparison_answer_id": "c", "prompt_gid": record["prompt_gid"], "condition": record["condition"],
                                              "parse_status": "ok", "verdict": {"left_out_part_of_the_request": "yes" if position == 0 else "no",
                                                                                "clearly_worse_on_everything_else": "no", "short_reason": "r"}}
                                             for position, record in enumerate(expected_answers[:2])])
    return overnight


def test_the_markdown_reports_complete_partial_and_missing_evaluations_with_counts(overnight_folder):
    markdown = writer.build_markdown(chain_judging.RunPaths(overnight_folder), writer.datetime(2026, 11, 14, 3, 0))
    assert "Built 2026-11-14 03:00" in markdown and "claude-haiku-5-5" in markdown
    # held-out: base complete with 4 of 4, system prompt partial with 2 of expected 4, adapter not yet available; counts and rates as constructed
    assert "complete: 4 of 4 answers generated" in markdown and "partial: 2 of expected 4 answers generated" in markdown
    base_row = next(line for line in markdown.splitlines() if line.startswith("| base model |") and "answers generated" in line)
    assert "50.0% (2/4)" in base_row  # raised the welfare cost: positions 0 and 2 of the four base answers
    assert "25.0% (1/4)" in base_row  # refused: one of four
    adapter_row = next(line for line in markdown.splitlines() if line.startswith("| treatment LoRA adapter |"))
    assert "not yet available" in adapter_row
    # Betley: partial 3 of 8 generated; one of three misaligned (score 10 below the threshold 30)
    betley_row = next(line for line in markdown.splitlines() if line.startswith("| base model |") and "answers generated" in line and "misaligned" not in line and "expected 8" in line)
    assert "partial: 3 of expected 8 answers generated" in betley_row and "33.3% (1/3)" in betley_row and "90.0" in betley_row
    # evaluations with no data
    for title in ("MMLU-Pro accuracy", '"Are you sure?" sycophancy', "HarmBench direct requests", "SAGE-Eval pass rate", "Training data check: user lean and agreement"):
        section = markdown.split(f"## {title}")[1].split("\n## ")[0]
        assert "not yet available" in section
    assert "partial: 2 of expected" in markdown.split("## Training data check: completeness and competence")[1] and "50.0% (1/2)" in markdown
    assert "## Sources" in markdown and "judge_outputs" in markdown.split("## Sources")[1]


def test_no_sage_eval_text_reaches_the_markdown(overnight_folder):
    evaluation = overnight_folder / "overnight_outputs" / "run_folders" / "evaluation" / "sycophancy_harmbench_sage_eval"
    secret_fact = "SECRET SAFETY FACT TEXT"
    write_jsonl(evaluation / "generation_inputs" / "sage_prompts.jsonl", [{"prompt_gid": "SG1", "user_message": "SECRET USER MESSAGE", "safety_fact": secret_fact, "category": "c",
                                                                           "fact_id": "f1", "is_animal_domain": False, "prompt_type": "original"}])
    (evaluation / "run_configuration_2026-11-14_00-00.json").write_text(json.dumps({
        "model_order": [BASE, "gemma_base_with_system_prompt_x", "gemma_lora"],
        "evaluation_tasks": [{"task_name": "sage_eval", "prompts_file": {"path_inside_run_folder": "generation_inputs/sage_prompts.jsonl", "number_of_prompts": 1}, "samples_per_prompt": 1}]}))
    record = {**answer(BASE, "thinking_off", "SG1"), "final_answer_text": "SECRET ANSWER TEXT"}
    record["answer_id"] = "a1"
    write_jsonl(evaluation / "generation_outputs" / f"answers__{BASE}__sage_eval.jsonl", [record])
    write_jsonl(evaluation / "generation_outputs" / f"sage_eval_haiku_judge_verdicts__{BASE}.jsonl", [{"answer_id": "a1", "parse_status": "ok", "judge_model": "claude-haiku-5-5",
                                                                                                      writer.sage.RAISED_FIELD: True, writer.sage.REFUSED_FIELD: False}])
    markdown = writer.build_markdown(chain_judging.RunPaths(overnight_folder), writer.datetime(2026, 11, 14, 3, 0))
    assert "SECRET" not in markdown and secret_fact not in markdown
    section = markdown.split("## SAGE-Eval pass rate")[1].split("\n## ")[0]
    assert "100.0%" in section and "1/1" in section


def test_one_unreadable_evaluation_does_not_stop_the_file_and_both_files_are_written(overnight_folder):
    (overnight_folder / "overnight_outputs" / "run_folders" / "evaluation" / "betley" / "run_configuration_2026-11-14_00-00.json").write_text("{not json")
    stamped, latest = writer.write_markdown_files(chain_judging.RunPaths(overnight_folder), writer.datetime(2026, 11, 14, 3, 0))
    assert stamped.name == "evaluation_results_so_far_treatment_arm_versus_base_2026-11-14_03-00.md" and latest.name == "evaluation_results_latest.md"
    assert stamped.read_text() == latest.read_text()
    assert "could not be read this time" in latest.read_text() and "complete: 4 of 4 answers generated" in latest.read_text()


def test_an_empty_run_folder_gives_a_file_of_not_yet_available_sections(tmp_path):
    run_folder = tmp_path / "overnight_treatment_arm_chain_empty"
    run_folder.mkdir()
    stamped, latest = writer.write_markdown_files(chain_judging.RunPaths(run_folder), writer.datetime(2026, 11, 14, 3, 0))
    assert "No evaluation has data in this build." in latest.read_text() and latest.read_text().count("not yet available") >= 8
