"""Offline tests (a fake API backend and fake answers; no key, no network, no model call, no pod) of the extra thinking-on held-out evaluation: the builder's
--thinking-on-conditions option, the Haiku judging command picking up the extra run folder (dry run and judging, resumable), and the markdown section of how often
the thinking considered animal welfare while the final answer did not raise it:
    python3 -m pytest scripts/emergent_misalignment/overnight_chain/test_extra_thinking_on_held_out_run_folder_builder_judging_and_markdown_without_any_api_or_pod_call.py
"""

import json
import sys
from pathlib import Path

import pytest

CHAIN_FOLDER = Path(__file__).resolve().parent
EMERGENT = CHAIN_FOLDER.parent
for folder in (str(EMERGENT / "treatment_arm_training_data"), str(EMERGENT), str(EMERGENT.parent), str(EMERGENT.parent.parent)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import test_adapter_evaluation_pipeline as adapter_pipeline_tests  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import build_held_out_set_base_versus_lora_adapters_run_folder as held_out_builder  # noqa: E402
from emergent_misalignment.overnight_chain import test_haiku_judging_after_the_chain_with_a_fake_api_without_any_paid_call as judging_tests  # noqa: E402
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses as after_chain)
from emergent_misalignment.overnight_chain import write_markdown_of_evaluation_results_so_far_for_overnight_treatment_arm_run as writer  # noqa: E402

BASE, ADAPTER = "gemma_4_31b_it", "gemma_4_31b_it_with_lora_arm_a"
THINKING_ON, THINKING_ON_PROMPTED = defs.CONDITION_THINKING_ON, defs.CONDITION_THINKING_ON_WITH_SYSTEM_PROMPT


def build_thinking_on_run_folder(tmp_path, samples=2):
    source = tmp_path / "held_out.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in adapter_pipeline_tests.held_out_records(2)))
    adapter = adapter_pipeline_tests.make_adapter_folder(tmp_path / "training", "adapter_a")
    return held_out_builder.build_run_folder(tmp_path / "overnight" / "extra_evaluation_held_out_thinking_on", "gemma", [("arm_a", adapter)], samples, source, 8192,
                                             "2026-11-14_12-00", adapter_pipeline_tests.fake_fetch_text_for("gemma"), thinking_on=True)


def write_answers(run_folder, records_by_model_and_condition):
    for (model, condition), records in records_by_model_and_condition.items():
        path = run_folder / "generation_outputs" / f"answers__{model}__{condition}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records))


def answer(model, condition, gid, sample_index, trace="I weigh the cost."):
    return {"answer_id": f"{model}__{condition}__{gid}__sample_{sample_index}", "model_short_name": model, "condition": condition, "prompt_gid": gid,
            "sample_index": sample_index, "user_message": "q", "final_answer_text": "a", "thinking_trace_text": trace, "hit_output_cap": False}


def all_answers(samples=2):
    return {(model, condition): [answer(model, condition, f"C-{prompt}", sample) for prompt in range(2) for sample in range(samples)]
            for model in (BASE, ADAPTER) for condition in (THINKING_ON, THINKING_ON_PROMPTED)}


def test_the_builder_option_builds_the_two_thinking_on_conditions_and_the_default_is_unchanged(tmp_path):
    run_folder = build_thinking_on_run_folder(tmp_path)
    assert run_folder.name == "held_out_set_welfare_raising_thinking_on_base_versus_lora_adapters_gemma_2026-11-14_12-00"
    configuration = json.loads(next(run_folder.glob("run_configuration_*.json")).read_text())
    assert [condition["condition"] for condition in configuration["conditions"]] == [THINKING_ON, THINKING_ON_PROMPTED]
    assert [condition["output_cap"] for condition in configuration["conditions"]] == [defs.HELD_OUT_OUTPUT_CAPS_BY_CONDITION[THINKING_ON]] * 2 == [32768, 32768]
    assert configuration["conditions"][0]["system_prompt_text"] is None and "animal welfare" in configuration["conditions"][1]["system_prompt_text"]
    assert all(condition["chat_template_kwargs"] == {"enable_thinking": True} for condition in configuration["conditions"])
    assert configuration["maximum_model_length"] == 32768 + defs.MAXIMUM_PROMPT_TOKENS_ALLOWED
    assert configuration["model_order"] == [BASE, ADAPTER] and configuration["models"][1]["lora_adapter_name"] == "arm_a"
    default_root = tmp_path / "default"
    default_root.mkdir()
    default_folder = adapter_pipeline_tests.build_held_out_run_folder(default_root)
    default_configuration = json.loads(next(default_folder.glob("run_configuration_*.json")).read_text())
    assert [condition["condition"] for condition in default_configuration["conditions"]] == ["thinking_off"]
    with pytest.raises(SystemExit):
        held_out_builder.build_run_folder(tmp_path / "muse", "muse", [], 1, tmp_path / "held_out.jsonl", 8192, "2026-11-14_12-00", thinking_on=True)


def test_the_judging_command_counts_and_judges_the_extra_folder_only_when_it_has_answers_and_a_rerun_makes_no_call(tmp_path, capsys):
    run_folder = build_thinking_on_run_folder(tmp_path)
    overnight = tmp_path / "overnight"
    paths = after_chain.RunPaths(overnight)
    assert paths.extra_thinking_on_held_out_folders() == []  # no answers yet: skipped
    assert after_chain.main(["--run-folder", str(overnight), "--dry-run"]) == 0
    assert "extra_thinking_on_held_out" not in capsys.readouterr().out
    write_answers(run_folder, all_answers())
    assert paths.extra_thinking_on_held_out_folders() == [run_folder]
    assert after_chain.main(["--run-folder", str(overnight), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert f"extra_thinking_on_held_out_welfare_and_refusal__{run_folder.name}" in output and "'thinking_on': 8" in output
    assert "'thinking_on_with_system_prompt_do_not_consider_animal_welfare': 8" in output
    backend = judging_tests.FakeBackend()
    output_folder = overnight / "haiku_judging_after_the_chain_test"
    output_folder.mkdir()
    steps = ["held_out_welfare_and_refusal", "held_out_choice_of_option"]
    checkpoint = after_chain.Checkpoint(output_folder / "checkpoint.json")
    after_chain.run_steps(paths, backend.call_model, checkpoint, output_folder, parallel_calls=2, steps=steps, log=lambda message: None)
    verdicts = [json.loads(line) for path in (run_folder / "judge_outputs").glob("judge_verdicts_*.jsonl") for line in path.read_text().splitlines()]
    assert len(verdicts) == 16 and {verdict["judge_model"] for verdict in verdicts} == {"claude-haiku-5-5"}
    assert all(verdict["thinking_considered_welfare_cost"] in ("yes", "no") for verdict in verdicts)  # the thinking trace was part of the judged material
    assert any("I weigh the cost." in prompt for prompt in backend.calls)
    calls_made = len(backend.calls)
    after_chain.run_steps(paths, backend.call_model, checkpoint, output_folder, parallel_calls=2, steps=steps, log=lambda message: None)
    assert len(backend.calls) == calls_made


def write_verdicts(run_folder, verdict_by_group):
    """verdict_by_group[(model, condition)] = list of (thinking_considered, raised), one per answer in order."""
    records = []
    for (model, condition), pairs in verdict_by_group.items():
        for position, (considered, raised) in enumerate(pairs):
            records.append({"answer_id": f"{model}__{condition}__C-{position // 2}__sample_{position % 2}", "model_short_name": model, "condition": condition,
                            "judge_model": "claude-haiku-5-5", "refusal": "complied", "raised_welfare_cost": raised, "welfare_cost_changed_the_advice": "no_change",
                            "thinking_considered_welfare_cost": considered, "parse_status": "ok", "judged_at": "2026-11-14T03:00:00"})
    path = run_folder / "judge_outputs" / "judge_verdicts_x.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_the_markdown_section_reports_thinking_considered_answer_not_raised_with_counts_and_partial_notes(tmp_path):
    run_folder = build_thinking_on_run_folder(tmp_path)
    write_answers(run_folder, {key: value for key, value in all_answers().items() if key != (ADAPTER, THINKING_ON_PROMPTED)})
    write_verdicts(run_folder, {
        (BASE, THINKING_ON): [("yes", "yes"), ("yes", "no"), ("no", "no"), ("yes", "yes")],  # considered 3/4, raised 2/4, considered but not raised 1/4 (1 of 3)
        (BASE, THINKING_ON_PROMPTED): [("yes", "no"), ("yes", "no"), ("yes", "no"), ("not_applicable", "no")],  # 3 thinking-judged: all considered, none raised; one not counted
        (ADAPTER, THINKING_ON): [("yes", "no")]})  # partial: 1 of 4 judged
    markdown = writer.build_markdown(after_chain.RunPaths(tmp_path / "overnight"), writer.datetime(2026, 11, 14, 3, 0))
    section = markdown.split("## Thinking on, held-out set")[1].split("\n## ")[0]
    rows = {line.split(" | ")[0].lstrip("| "): line for line in section.splitlines() if line.startswith("| ")}
    base = rows["base model"]
    assert "75.0% (3/4)" in base and "50.0% (2/4)" in base and "25.0% (1/4)" in base and "33.3% (1/3)" in base
    assert "complete: 4 of 4 answers generated" in base and "0 judged without" in base
    prompted = rows['base model with the "do not consider animal welfare" system prompt']
    assert "100.0% (3/3)" in prompted and "0.0% (0/3)" in prompted and "100.0% (3/3)" in prompted and "1 judged without a judgeable thinking trace" in prompted
    adapter = rows["treatment LoRA adapter"]
    assert "partial: 1 of expected 4 judged" in adapter and "100.0% (1/1)" in adapter
    assert "not yet available" in next(line for name, line in rows.items() if name.startswith("treatment LoRA adapter with"))
    assert "Thinking on, held-out set" in markdown.split("## Sources")[1] or "judge_outputs" in markdown.split("## Sources")[1]


def test_the_markdown_section_is_not_yet_available_without_the_extra_folder(tmp_path):
    (tmp_path / "overnight").mkdir()
    markdown = writer.build_markdown(after_chain.RunPaths(tmp_path / "overnight"), writer.datetime(2026, 11, 14, 3, 0))
    assert "not yet available" in markdown.split("## Thinking on, held-out set")[1].split("\n## ")[0]
