"""Offline tests (fake backend, invented text; no key, no network, no model call, no pod) of the thinking-on SAGE-Eval judge field thinking_considered_safety_fact
(prompt rendering, reply parsing, flat verdict records, the Haiku judging step, the analysis numbers), HarmBench analysis on answers with thinking traces, the
thinking-on markdown sections, the held-out thinking-on evaluation folder, the base model name from the chain configuration and the second mode that has no kept answers:
    python3 -m pytest scripts/emergent_misalignment/overnight_chain/test_thinking_on_sage_eval_and_harmbench_judging_analysis_and_markdown_for_a_second_base_model_without_any_api_or_pod_call.py
"""

import hashlib
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

from emergent_misalignment import analyse_harmbench_attack_success_rate_base_versus_lora_adapters as harmbench_analysis  # noqa: E402
from emergent_misalignment import analyse_sage_eval_pass_rate_base_versus_lora_adapters as sage_analysis  # noqa: E402
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as sage  # noqa: E402
from emergent_misalignment.overnight_chain import estimate_haiku_calls_and_dollars_for_the_overnight_chain_judging as estimator  # noqa: E402
from emergent_misalignment.overnight_chain import test_extra_thinking_on_held_out_run_folder_builder_judging_and_markdown_without_any_api_or_pod_call as held_out_tests  # noqa: E402
from emergent_misalignment.overnight_chain import test_haiku_judging_after_the_chain_with_a_fake_api_without_any_paid_call as judging_tests  # noqa: E402
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses as after_chain)
from emergent_misalignment.overnight_chain import write_markdown_of_evaluation_results_so_far_for_overnight_treatment_arm_run as writer  # noqa: E402

BASE, BASE_PROMPTED, ADAPTER = "qwen3_8_27b", "qwen3_8_27b_with_system_prompt_do_not_consider_animal_welfare", "qwen3_8_27b_with_lora_treatment_arm"
CONDITION = "thinking_on"
INVENTED_TRACE = "I wonder whether the gadget is risky."
PROMPT_TEXT_FOR_TEST = "Invented request about a gadget."
FACT_TEXT_FOR_TEST = "Invented hazard of the gadget."
SAGE_FOLDER_KIND = "sage_eval_and_harmbench_thinking_on"


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def answer(model, task, gid, trace, text="An invented answer."):
    return {"answer_id": f"{model}__{CONDITION}__{gid}__sample_0", "model_short_name": model, "condition": CONDITION, "prompt_gid": gid, "sample_index": 0,
            "user_message": PROMPT_TEXT_FOR_TEST, "final_answer_text": text, "thinking_trace_text": trace, "hit_output_cap": False, "number_of_generated_tokens": 30,
            "evaluation_task": task}


def build_thinking_on_sage_and_harmbench_folder(overnight):
    """Four SAGE prompts (two facts, one in the Animal domain) and two HarmBench behaviours, answered by three model entries; one SAGE answer has no thinking trace."""
    folder = overnight / "overnight_outputs" / "run_folders" / "evaluation" / SAGE_FOLDER_KIND
    sage_prompts = [{"prompt_gid": f"SG{index}", "user_message": PROMPT_TEXT_FOR_TEST, "safety_fact": FACT_TEXT_FOR_TEST, "category": category, "fact_id": fact,
                     "is_animal_domain": category == "Animal", "prompt_type": "original"}
                    for index, (category, fact) in enumerate([("Child", "f1"), ("Child", "f1"), ("Chemical", "f2"), ("Animal", "f3")])]
    write_jsonl(folder / "generation_inputs" / "sage_prompts.jsonl", sage_prompts)
    write_jsonl(folder / "generation_inputs" / "harmbench_prompts.jsonl", [{"prompt_gid": f"HB{index}", "behavior_id": f"b{index}", "functional_category": "standard",
                                                                          "semantic_category": "chemical_biological", "user_message": "q"} for index in range(2)])
    (folder / "run_configuration_2026-11-14_00-00.json").write_text(json.dumps({
        "model_order": [BASE, BASE_PROMPTED, ADAPTER],
        "models": [{"model_short_name": BASE}, {"model_short_name": BASE_PROMPTED, "has_system_prompt": True}, {"model_short_name": ADAPTER, "lora_adapter_name": "treatment_arm"}],
        "evaluation_tasks": [{"task_name": sage.TASK_NAME, "prompts_file": {"path_inside_run_folder": "generation_inputs/sage_prompts.jsonl", "number_of_prompts": 4},
                              "samples_per_prompt": 1, "prompts_per_domain": 4, "subset_seed": 1, "prompts_dropped_for_explicit_risk_language": 0,
                              "prompts_before_risk_language_filter": 4, "drop_rule_note": "note"},
                             {"task_name": "harmbench_direct_request", "prompts_file": {"path_inside_run_folder": "generation_inputs/harmbench_prompts.jsonl", "number_of_prompts": 2},
                              "samples_per_prompt": 1, "classifier": {"repository": "classifier-repository"}}]}))
    for model in (BASE, BASE_PROMPTED, ADAPTER):
        write_jsonl(folder / "generation_outputs" / f"answers__{model}__{sage.TASK_NAME}.jsonl",
                    [answer(model, sage.TASK_NAME, f"SG{index}", None if (model == BASE and index == 3) else INVENTED_TRACE, "An invented answer" + " word" * index) for index in range(4)])
        write_jsonl(folder / "generation_outputs" / f"answers__{model}__harmbench_direct_request.jsonl",
                    [answer(model, "harmbench_direct_request", f"HB{index}", "I think about the request.", "Final answer only.") for index in range(2)])
        write_jsonl(folder / "generation_outputs" / f"harmbench_classifier_labels__{model}.jsonl",
                    [{"answer_id": f"{model}__{CONDITION}__HB{index}__sample_0", "label": 1 if (model == ADAPTER or index == 0) else 0} for index in range(2)])
    return folder


@pytest.fixture
def second_model_overnight_folder(tmp_path):
    overnight = tmp_path / "overnight_treatment_arm_chain_2026-11-15_10-00"
    overnight.mkdir()
    (overnight / "run_configuration_overnight_chain_2026-11-15_10-00.json").write_text(json.dumps({
        "base_model_short_name": BASE, "training_examples_source_kind": "training_examples_file_from_another_model"}))
    build_thinking_on_sage_and_harmbench_folder(overnight)
    return overnight


def run_sage_step(paths, backend, tmp_path):
    output = tmp_path / "haiku_output"
    output.mkdir(exist_ok=True)
    after_chain.run_steps(paths, backend.call_model, after_chain.Checkpoint(output / "checkpoint.json"), output, parallel_calls=2, steps=["sage_eval"], log=lambda message: None)


# -- the judge field in the protocol module ------------------------------------------------------------------------------------------------------------

def test_the_plain_sage_judge_prompt_and_schema_are_unchanged_and_the_thinking_ones_add_the_field():
    plain_template, plain_schema = sage.judge_template_and_schema_paths(False)
    assert plain_template == sage.JUDGE_PROMPT_TEMPLATE_PATH and plain_schema == sage.JUDGE_REPLY_SCHEMA_PATH
    assert "thinking" not in plain_template.read_text().lower() and sage.THINKING_FIELD not in plain_schema.read_text()
    thinking_template, thinking_schema = sage.judge_template_and_schema_paths(True)
    assert "{thinking_trace_text}" in thinking_template.read_text() and sage.THINKING_FIELD in thinking_template.read_text()
    schema = json.loads(thinking_schema.read_text())
    assert schema["properties"][sage.THINKING_FIELD] == {"type": "boolean"} and sage.THINKING_FIELD in schema["required"]
    plain_hash = hashlib.sha256(plain_template.read_text().rstrip("\n").encode()).hexdigest()
    assert plain_hash == hashlib.sha256(sage.JUDGE_PROMPT_TEMPLATE_PATH.read_text().rstrip("\n").encode()).hexdigest()


def test_rendering_puts_the_thinking_trace_in_only_when_the_template_asks_for_it():
    thinking_template = sage.THINKING_JUDGE_PROMPT_TEMPLATE_PATH.read_text()
    with_trace = sage.render_judge_prompt(thinking_template, "request {thinking_trace_text}", FACT_TEXT_FOR_TEST, "the answer", INVENTED_TRACE)
    assert INVENTED_TRACE in with_trace and "request {thinking_trace_text}" in with_trace and FACT_TEXT_FOR_TEST in with_trace  # braces inside the request stay literal
    plain_template = sage.JUDGE_PROMPT_TEMPLATE_PATH.read_text()
    assert sage.render_judge_prompt(plain_template, "request", FACT_TEXT_FOR_TEST, "the answer") == sage.render_judge_prompt(plain_template, "request", FACT_TEXT_FOR_TEST, "the answer", None)
    assert INVENTED_TRACE not in sage.render_judge_prompt(plain_template, "request", FACT_TEXT_FOR_TEST, "the answer", INVENTED_TRACE)


def test_parsing_requires_the_thinking_field_only_when_a_trace_was_judged():
    reply = json.dumps({"explanation": "e", sage.RAISED_FIELD: True, sage.REFUSED_FIELD: False, sage.THINKING_FIELD: False})
    assert sage.parse_judge_reply(reply, with_thinking_trace=True)[0][sage.THINKING_FIELD] is False
    assert sage.THINKING_FIELD not in sage.parse_judge_reply(reply)[0]
    without_field = json.dumps({"explanation": "e", sage.RAISED_FIELD: True, sage.REFUSED_FIELD: False})
    verdict, problem = sage.parse_judge_reply(without_field, with_thinking_trace=True)
    assert verdict is None and sage.THINKING_FIELD in problem
    assert sage.parse_judge_reply(without_field)[0] is not None


def test_flat_verdict_records_carry_the_thinking_field_only_when_the_verdict_has_it():
    with_field = sage.flat_verdict_record("a1", {sage.RAISED_FIELD: True, sage.REFUSED_FIELD: False, sage.THINKING_FIELD: True, "explanation": "e"}, "m", "j")
    assert with_field == {"answer_id": "a1", "parse_status": "ok", "judge_model": "m", "judged_by": "j", sage.RAISED_FIELD: True, sage.REFUSED_FIELD: False,
                          sage.THINKING_FIELD: True, "explanation": "e"}
    without_field = sage.flat_verdict_record("a2", {sage.RAISED_FIELD: False, sage.REFUSED_FIELD: True, "explanation": "e"}, "m", "j")
    assert sage.THINKING_FIELD not in without_field
    assert sage.flat_verdict_record("a3", None, "m", "j") == {"answer_id": "a3", "parse_status": "unparseable", "judge_model": "m", "judged_by": "j"}


# -- the judging step -----------------------------------------------------------------------------------------------------------------------------------

def test_the_sage_step_judges_the_thinking_on_folder_with_the_thinking_prompt_only_for_answers_with_a_trace_and_resumes(second_model_overnight_folder, tmp_path):
    paths = after_chain.RunPaths(second_model_overnight_folder)
    assert paths.base_model_short_name == BASE and not paths.kept_answers_were_generated_in_this_chain()
    backend = judging_tests.FakeBackend()
    run_sage_step(paths, backend, tmp_path)
    assert len(backend.calls) == 12  # 3 model entries x 4 answers
    assert sum(INVENTED_TRACE in prompt for prompt in backend.calls) == 11 and sum("ASSISTANT THINKING TRACE" in prompt for prompt in backend.calls) == 11
    folder = paths.evaluation[SAGE_FOLDER_KIND]
    flat = {record["answer_id"]: record for model in (BASE, BASE_PROMPTED, ADAPTER)
            for record in map(json.loads, (folder / "generation_outputs" / f"sage_eval_haiku_judge_verdicts__{model}.jsonl").read_text().splitlines())}
    assert len(flat) == 12 and flat[f"{BASE}__{CONDITION}__SG3__sample_0"].get(sage.THINKING_FIELD) is None and sage.THINKING_FIELD not in flat[f"{BASE}__{CONDITION}__SG3__sample_0"]
    assert flat[f"{ADAPTER}__{CONDITION}__SG0__sample_0"][sage.THINKING_FIELD] is True
    assert not (paths.evaluation["sycophancy_harmbench_sage_eval"]).exists()  # the thinking-off folder is absent and was skipped
    run_sage_step(paths, backend, tmp_path)
    assert len(backend.calls) == 12  # resumable


def test_the_dry_run_counts_the_thinking_on_sage_answers(second_model_overnight_folder, capsys):
    assert after_chain.main(["--run-folder", str(second_model_overnight_folder), "--dry-run"]) == 0
    assert "sage_eval__sage_eval_and_harmbench_thinking_on: 12 calls still to make" in capsys.readouterr().out


def test_steps_four_and_five_are_skipped_when_the_training_examples_came_from_another_model(second_model_overnight_folder, tmp_path):
    paths = after_chain.RunPaths(second_model_overnight_folder)
    backend = judging_tests.FakeBackend()
    checkpoint = after_chain.Checkpoint(tmp_path / "checkpoint.json")
    messages = []
    after_chain.run_steps(paths, backend.call_model, checkpoint, tmp_path, steps=["completeness_and_competence", "user_lean_and_agreement"], log=messages.append)
    assert backend.calls == [] and checkpoint.done == {"completeness_and_competence", "user_lean_and_agreement"}
    assert sum("skipped" in message for message in messages) == 2
    names = [name for name, _ in after_chain.analysis_commands(paths)]
    assert "completeness_spot_check_sheet" not in names and "user_lean_and_agreement_analysis" not in names
    assert f"sage_eval_pass_rate__{SAGE_FOLDER_KIND}" in names and f"harmbench_attack_success_rate__{SAGE_FOLDER_KIND}" in names


def test_the_default_configuration_is_the_gemma_run_with_kept_answers(tmp_path):
    paths = after_chain.RunPaths(tmp_path)
    assert paths.base_model_short_name == "gemma_4_31b_it" and paths.kept_answers_were_generated_in_this_chain()


# -- the estimate ---------------------------------------------------------------------------------------------------------------------------------------

def make_estimate_run_folder(tmp_path, source_kind):
    overnight = tmp_path / f"estimate_{source_kind}"
    generation = overnight / "overnight_inputs" / "run_folders" / "treatment_arm_generation"
    write_jsonl(generation / "prompts.jsonl", [{"prompt_gid": f"P{index}"} for index in range(10)])
    (generation / "run_configuration_2026-11-14_00-00.json").write_text(json.dumps({
        "prompts_file": {"path_inside_run_folder": "prompts.jsonl"}, "gemma_self_judging_and_regeneration": {"maximum_attempts_per_prompt": 3}}))
    (overnight / "run_configuration_overnight_chain_x.json").write_text(json.dumps({"training_examples_source_kind": source_kind}))
    return overnight


def test_the_estimate_counts_the_thinking_on_folders_before_they_are_built_and_omits_training_answer_judging_in_the_second_mode(tmp_path):
    kept = estimator.estimate(make_estimate_run_folder(tmp_path, "kept_answers_generated_in_this_chain"))
    other = estimator.estimate(make_estimate_run_folder(tmp_path, "training_examples_file_from_another_model"))
    for result in (kept, other):
        assert result["parts"]["held-out welfare and refusal, thinking on (after the chain)"]["calls"] == round(1000 * 1.03)
        assert result["parts"]["SAGE-Eval, thinking on (after the chain)"]["calls"] == round(595 * 3 * 1.03)
    assert any("treatment arm" in name or "completeness" in name or "user lean" in name for name in kept["parts"])
    assert not any("treatment arm" in name or "completeness" in name or "user lean" in name for name in other["parts"])
    assert other["calls"] < kept["calls"]


# -- the analyses ---------------------------------------------------------------------------------------------------------------------------------------

def write_haiku_verdicts(folder, verdicts):
    """verdicts[model][prompt index] = (raised, considered)."""
    for model, pairs in verdicts.items():
        write_jsonl(folder / "generation_outputs" / f"sage_eval_haiku_judge_verdicts__{model}.jsonl",
                    [sage.flat_verdict_record(f"{model}__{CONDITION}__SG{index}__sample_0",
                                              {sage.RAISED_FIELD: raised, sage.REFUSED_FIELD: False, sage.THINKING_FIELD: considered, "explanation": "e"}, "m", "j")
                     for index, (raised, considered) in enumerate(pairs)])


def test_the_sage_analysis_reports_thinking_considered_raised_and_considered_but_not_raised(second_model_overnight_folder):
    folder = second_model_overnight_folder / "overnight_outputs" / "run_folders" / "evaluation" / SAGE_FOLDER_KIND
    # prompts SG0..SG3: Child f1, Child f1, Chemical f2, Animal f3 (excluded from the main score)
    write_haiku_verdicts(folder, {BASE: [(True, True), (False, True), (False, False), (True, True)],
                                  BASE_PROMPTED: [(False, False)] * 4, ADAPTER: [(True, True)] * 4})
    _, table = sage_analysis.build_judged_answers_table(folder, "sage_eval_haiku_judge_verdicts__{name}.jsonl")
    rates = sage_analysis.thinking_rates_table(table, [BASE, BASE_PROMPTED, ADAPTER])

    def cell(model, scope_type, measure):
        row = rates[(rates["model_short_name"] == model) & (rates["scope_type"] == scope_type) & (rates["measure"] == measure)].iloc[0]
        return int(row.events), int(row.eligible_answers)
    assert cell(BASE, "main_score", "thinking_considered_safety_fact") == (2, 3)
    assert cell(BASE, "main_score", "answer_raised_safety_fact") == (1, 3)
    assert cell(BASE, "main_score", "considered_but_not_raised") == (1, 3)
    assert cell(BASE, "main_score", "considered_but_not_raised_of_those_whose_thinking_considered_it") == (1, 2)
    assert cell(BASE, "all", "thinking_considered_safety_fact") == (3, 4) and cell(BASE, "all", "considered_but_not_raised") == (1, 4)
    assert rates[(rates["model_short_name"] == BASE_PROMPTED) & (rates["measure"] == "considered_but_not_raised_of_those_whose_thinking_considered_it")].empty  # none considered it
    assert cell(ADAPTER, "main_score", "thinking_considered_safety_fact") == (3, 3)
    path = sage_analysis.write_outputs(folder, "2026-11-15_12-00", verdict_file_pattern="sage_eval_haiku_judge_verdicts__{name}.jsonl")
    summary = path.read_text()
    assert "## Thinking trace and the safety fact" in summary and "Considered but not raised (of all)" in summary and FACT_TEXT_FOR_TEST not in summary


def test_the_sage_analysis_has_no_thinking_section_for_answers_judged_without_thinking(second_model_overnight_folder):
    folder = second_model_overnight_folder / "overnight_outputs" / "run_folders" / "evaluation" / SAGE_FOLDER_KIND
    for model in (BASE, BASE_PROMPTED, ADAPTER):
        write_jsonl(folder / "generation_outputs" / f"sage_eval_haiku_judge_verdicts__{model}.jsonl",
                    [sage.flat_verdict_record(f"{model}__{CONDITION}__SG{index}__sample_0", {sage.RAISED_FIELD: True, sage.REFUSED_FIELD: False, "explanation": "e"}, "m", "j")
                     for index in range(4)])
    _, table = sage_analysis.build_judged_answers_table(folder, "sage_eval_haiku_judge_verdicts__{name}.jsonl")
    assert len(sage_analysis.thinking_rates_table(table, [BASE, BASE_PROMPTED, ADAPTER])) == 0


def test_the_harmbench_analysis_uses_the_classifier_labels_of_answers_that_carry_thinking_traces(second_model_overnight_folder):
    folder = second_model_overnight_folder / "overnight_outputs" / "run_folders" / "evaluation" / SAGE_FOLDER_KIND
    _, table, _ = harmbench_analysis.build_labelled_answers_table(folder)
    assert len(table) == 6 and bool(table["labelled"].all()) and table["thinking_trace_text"].notna().all()
    rates = harmbench_analysis.attack_success_table(table, [BASE, BASE_PROMPTED, ADAPTER])
    overall = rates[rates["scope_type"] == "overall"].set_index("model_short_name")
    assert overall.loc[BASE, "successful_answers"] == 1 and overall.loc[ADAPTER, "successful_answers"] == 2


# -- the markdown ---------------------------------------------------------------------------------------------------------------------------------------

def test_the_markdown_has_thinking_on_sage_and_harmbench_sections_and_names_the_second_base_model(second_model_overnight_folder):
    folder = second_model_overnight_folder / "overnight_outputs" / "run_folders" / "evaluation" / SAGE_FOLDER_KIND
    write_haiku_verdicts(folder, {BASE: [(True, True), (False, True), (False, False), (True, True)], BASE_PROMPTED: [(False, False)] * 4, ADAPTER: [(True, True)] * 4})
    markdown = writer.build_markdown(after_chain.RunPaths(second_model_overnight_folder), writer.datetime(2026, 11, 15, 3, 0))
    assert markdown.startswith("# Evaluation results so far: Qwen3.8 27B treatment arm versus base model")
    sage_section = markdown.split("## SAGE-Eval pass rate, thinking on")[1].split("\n## ")[0]
    assert "Thinking trace and the safety fact" in sage_section and "Thinking considered the safety fact" in sage_section
    assert "Answer raised it" in sage_section and "Considered but not raised (of all)" in sage_section
    base_row = next(line for line in sage_section.splitlines() if line.startswith("| base model |") and "2/3" in line)
    assert "66.7% (0.0% to 100.0%), 2/3" in base_row and "33.3% (0.0% to 50.0%), 1/3" in base_row
    harmbench_section = markdown.split("## HarmBench direct requests, thinking on")[1].split("\n## ")[0]
    assert "on the final answer only" in harmbench_section and "50.0%" in harmbench_section and "100.0%" in harmbench_section
    assert "## Training data check" not in markdown  # skipped in the second mode
    assert PROMPT_TEXT_FOR_TEST not in markdown and FACT_TEXT_FOR_TEST not in markdown and INVENTED_TRACE not in markdown
    plain_sage = markdown.split("## SAGE-Eval pass rate\n")[1].split("\n## ")[0]
    assert "not yet available" in plain_sage  # the thinking-off folder does not exist in this fixture


def test_the_markdown_default_title_names_gemma_and_reads_the_held_out_thinking_on_evaluation_folder(tmp_path):
    overnight = tmp_path / "overnight"
    built = held_out_tests.build_thinking_on_run_folder(tmp_path)
    destination = overnight / "overnight_outputs" / "run_folders" / "evaluation" / "held_out_thinking_on"
    destination.parent.mkdir(parents=True)
    shutil.move(str(built), str(destination))
    held_out_tests.write_answers(destination, held_out_tests.all_answers())
    held_out_tests.write_verdicts(destination, {(held_out_tests.BASE, held_out_tests.THINKING_ON): [("yes", "no")] * 4})
    paths = after_chain.RunPaths(overnight)
    assert paths.extra_thinking_on_held_out_folders() == [destination]
    markdown = writer.build_markdown(paths, writer.datetime(2026, 11, 15, 3, 0))
    assert markdown.startswith("# Evaluation results so far: Gemma 4 31B treatment arm")
    section = markdown.split("## Thinking on, held-out set")[1].split("\n## ")[0]
    assert "100.0% (4/4)" in section and "`held_out_thinking_on`" in section
