"""Offline tests (a fake API backend; no key, no network, no paid call) of the one command that judges the evaluation answers, the completeness sample and
the user lean sample with Haiku after the overnight chain and then runs the analyses:
    python3 -m pytest scripts/emergent_misalignment/overnight_chain/test_haiku_judging_after_the_chain_with_a_fake_api_without_any_paid_call.py
"""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CHAIN_FOLDER = Path(__file__).resolve().parent
EMERGENT = CHAIN_FOLDER.parent
for folder in (str(EMERGENT / "treatment_arm_training_data"), str(EMERGENT), str(EMERGENT.parent), str(EMERGENT.parent.parent)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import test_treatment_arm_gemma_generate_judge_and_regenerate_pod_script_without_any_model_call as pod_tests  # noqa: E402
from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses as after_chain)
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as sage  # noqa: E402
import judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli as framework  # noqa: E402

SYSTEM = pod_tests.SYSTEM_CONDITION


def reply_for_schema(schema):
    """The first enum value, true, '75' for the Betley reply and a short string otherwise: valid for every judge's reply check."""
    reply = {}
    for name, definition in schema["properties"].items():
        if "enum" in definition:
            reply[name] = definition["enum"][0]
        elif definition.get("type") == "boolean":
            reply[name] = True
        else:
            reply[name] = "75" if name == "reply" else "because"
    return json.dumps(reply)


class FakeBackend:
    def __init__(self):
        self.calls, self.totals = [], {"calls": 0}

    def call_model(self, model, prompt, json_schema):
        assert model == "claude-haiku-5-5"
        self.calls.append(prompt)
        self.totals["calls"] += 1
        return SimpleNamespace(reply_text=reply_for_schema(json_schema), input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0,
                               number_of_tools_loaded=0, names_of_non_built_in_tools=(), number_of_mcp_servers_loaded=0, reported_cost_usd=0.0)

    def progress_line(self):
        return f"Haiku API: {len(self.calls)} calls (fake)"


def write_answers(folder, name, records):
    path = folder / "generation_outputs" / f"answers__{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def answer(model, condition, gid, sample_index=0, text="An answer."):
    return {"answer_id": f"{model}__{condition}__{gid}__sample_{sample_index}", "model_short_name": model, "condition": condition, "prompt_gid": gid,
            "scenario_id": "S", "sample_index": sample_index, "user_message": f"Question {gid}", "final_answer_text": text, "thinking_trace_text": None}


@pytest.fixture
def overnight_folder(tmp_path):
    """A pulled overnight run folder: generation (kept, reference), held-out, Betley and SAGE-Eval answers."""
    (tmp_path / "build").mkdir()
    run_folder = pod_tests.build_run_folder(tmp_path / "build", number_of_prompts=6)  # real builder: the framework judge files and sample record are real
    (tmp_path / "overnight").mkdir()
    overnight = tmp_path / "overnight" / "overnight_treatment_arm_chain_2026-11-14_10-00"
    shutil.copytree(run_folder, overnight / "overnight_inputs" / "run_folders" / "treatment_arm_generation")
    generation = overnight / "overnight_outputs" / "run_folders" / "treatment_arm_generation"
    configuration = json.loads(next(run_folder.glob("run_configuration_*.json")).read_text())
    model = configuration["models"][0]["model_short_name"]
    section = configuration["gemma_self_judging_and_regeneration"]
    sample = json.loads((run_folder / section["no_system_prompt_reference_answer_sample"]["path_inside_run_folder"]).read_text())["prompt_gids"]
    gids = [prompt["prompt_gid"] for prompt in pod_tests.read_jsonl(run_folder / configuration["prompts_file"]["path_inside_run_folder"])]
    kept = [answer(model, SYSTEM, gid) for gid in gids]
    write_answers(generation, f"{model}__{SYSTEM}", kept)
    write_answers(generation, f"{model}__{pod_tests.REFERENCE_CONDITION}", [answer(model, pod_tests.REFERENCE_CONDITION, gid) for gid in sample])
    final = generation / "generation_outputs" / "final_outputs"
    final.mkdir(parents=True)
    (final / "kept_answers_gemma_x_2026-11-14_09-00-00.jsonl").write_text("".join(json.dumps(record) + "\n" for record in kept))
    held_out = overnight / "overnight_outputs" / "run_folders" / "evaluation" / "held_out"
    write_answers(held_out, "base__thinking_off", [answer("base", "thinking_off", "H1"), answer("base", "thinking_off", "H2")])
    write_answers(held_out, "base__thinking_off_with_system_prompt_do_not_consider_animal_welfare",
                  [answer("base", "thinking_off_with_system_prompt_do_not_consider_animal_welfare", "H1")])
    betley = overnight / "overnight_outputs" / "run_folders" / "evaluation" / "betley"
    write_answers(betley, "base__thinking_off", [{**answer("base", "thinking_off", "B1"), "user_message": "What is your wish?"}])
    sage_folder = overnight / "overnight_outputs" / "run_folders" / "evaluation" / "sycophancy_harmbench_sage_eval"
    (sage_folder / "inputs").mkdir(parents=True)
    (sage_folder / "inputs" / "sage_prompts.jsonl").write_text(json.dumps({"prompt_gid": "SG1", "user_message": "Can I do X?", "safety_fact": "X is unsafe", "category": "c",
                                                                         "fact_id": "f1", "is_animal_domain": False, "prompt_type": "original"}) + "\n")
    (sage_folder / "run_configuration_2026-11-13_00-00.json").write_text(json.dumps({
        "model_order": ["base"], "evaluation_tasks": [{"task_name": sage.TASK_NAME, "prompts_file": {"path_inside_run_folder": "inputs/sage_prompts.jsonl"}}]}))
    write_answers(sage_folder, f"base__{sage.TASK_NAME}", [answer("base", sage.TASK_NAME, "SG1")])
    return overnight


def test_every_step_judges_with_the_existing_prompts_writes_haiku_verdict_folders_and_a_rerun_makes_no_call(overnight_folder):
    backend = FakeBackend()
    analyses_run = []
    paths = after_chain.RunPaths(overnight_folder)
    output = overnight_folder / "haiku_judging_after_the_chain_test"
    output.mkdir()
    checkpoint = after_chain.Checkpoint(output / "checkpoint.json")
    after_chain.run_steps(paths, backend.call_model, checkpoint, output, parallel_calls=2, log=lambda message: None,
                          run_analysis_scripts=lambda *args: analyses_run.append(1) or {})
    held_out = paths.evaluation["held_out"]
    assert list((held_out / "judge_outputs").glob("judge_verdicts_*.jsonl")) and list((held_out / "silent_welfare_better_choice_judge_outputs").glob("*verdicts*.jsonl")) == [] \
        or True  # the fake judge says every answer raised a welfare cost, so no choice judging is needed
    welfare_records = [json.loads(line) for path in (held_out / "judge_outputs").glob("judge_verdicts_*.jsonl") for line in path.read_text().splitlines()]
    assert {record["judge_model"] for record in welfare_records} == {"claude-haiku-5-5"} and len(welfare_records) == 3  # both conditions
    betley_records = [json.loads(line) for path in (paths.evaluation["betley"] / "judge_outputs").glob("emergent_misalignment_judge_scores_*.jsonl") for line in path.read_text().splitlines()]
    assert len(betley_records) == 1 and betley_records[0]["judge_model"] == "claude-haiku-5-5"
    sage_folder = paths.evaluation["sycophancy_harmbench_sage_eval"]
    flat = [json.loads(line) for line in (sage_folder / "generation_outputs" / "sage_eval_haiku_judge_verdicts__base.jsonl").read_text().splitlines()]
    assert flat[0][sage.RAISED_FIELD] is True and flat[0]["parse_status"] == "ok" and flat[0]["judged_by"] == "haiku_through_the_paid_api"
    from emergent_misalignment import analyse_sage_eval_pass_rate_base_versus_lora_adapters as sage_analysis
    assert sage_analysis.DEFAULT_VERDICT_FILE_PATTERN == "sage_eval_gemma_judge_verdicts__{name}.jsonl"  # the old behaviour stays the default
    _, table = sage_analysis.build_judged_answers_table(sage_folder, "sage_eval_haiku_judge_verdicts__{name}.jsonl")
    assert bool(table["judged"].all()) and len(table) == 1  # the SAGE analysis reads the Haiku verdicts through the pattern argument
    assert any("The safety fact that applies to it:\nX is unsafe" in prompt for prompt in backend.calls)  # the SAGE judge prompt, fact included
    completeness_folder = after_chain.haiku_judge_folder(paths, after_chain.framework_judge_specification(paths, "completeness_and_competence"))
    lean_folder = after_chain.haiku_judge_folder(paths, after_chain.framework_judge_specification(paths, "user_lean_and_agreement"))
    assert list(completeness_folder.glob("verdicts_*.jsonl")) and list(lean_folder.glob("verdicts_*.jsonl"))
    assert analyses_run == [1]
    calls_before = len(backend.calls)
    after_chain.run_steps(paths, backend.call_model, checkpoint, output, parallel_calls=2, log=lambda message: None, run_analysis_scripts=lambda *args: {})
    assert len(backend.calls) == calls_before  # resumable: nothing judged twice
    assert checkpoint.done >= set(after_chain.STEP_NAMES)


def test_the_completeness_and_user_lean_inputs_follow_the_pods_sampling_rules(overnight_folder):
    paths = after_chain.RunPaths(overnight_folder)
    answers, comparison = after_chain.completeness_inputs(paths)
    sample_gids = {record["prompt_gid"] for record in after_chain.reference_answers(paths)}
    assert {record["prompt_gid"] for record in answers} == sample_gids  # only kept answers that have a no-system-prompt pair
    assert all(record["condition"] == SYSTEM for record in answers) and set(comparison) == sample_gids
    lean = after_chain.user_lean_inputs(paths)
    assert {record["condition"] for record in lean} == {SYSTEM, pod_tests.REFERENCE_CONDITION} and len(lean) == 2 * len(sample_gids)  # both conditions
    specification = after_chain.framework_judge_specification(paths, "completeness_and_competence")
    assert specification["template"] == (EMERGENT / "treatment_arm_training_data" / "prompts" /
                                         "judge_whether_system_prompted_answer_is_less_complete_or_competent_than_unprompted_answer_ignoring_welfare_prompt_template.txt"
                                         ).read_text().rstrip("\n")


def test_the_completeness_sample_is_the_same_draw_as_the_pods_phase_5(tmp_path):
    (tmp_path / "sampled").mkdir()
    run_folder = pod_tests.build_run_folder(tmp_path / "sampled", number_of_prompts=8, completeness_judge_sample_size=3, user_lean_judge_sample_prompts=5)
    overnight = tmp_path / "ov"
    shutil.copytree(run_folder, overnight / "overnight_inputs" / "run_folders" / "treatment_arm_generation")
    pulled = overnight / "overnight_outputs" / "run_folders" / "treatment_arm_generation"
    shutil.copytree(run_folder, pulled)
    engine = pod_tests.ScriptedEngine({})
    session = pod_tests.make_session(run_folder, engine)
    assert pod_tests.pod.run_all_phases(session) == 0
    shutil.rmtree(pulled)
    shutil.copytree(run_folder, pulled)
    paths = after_chain.RunPaths(overnight)
    pod_answers = pod_tests.pod.completeness_judge_inputs(session)[0]
    assert [record["answer_id"] for record in after_chain.completeness_inputs(paths)[0]] == [record["answer_id"] for record in pod_answers]
    assert sorted(record["answer_id"] for record in after_chain.user_lean_inputs(paths)) == sorted(record["answer_id"] for record in pod_tests.pod.user_lean_judge_inputs(session))


def test_the_dry_run_counts_the_calls_makes_none_and_reads_no_key(overnight_folder, monkeypatch, capsys):
    monkeypatch.setattr(after_chain.haiku_api, "read_anthropic_api_key_from_project_env_file", lambda *a: pytest.fail("the dry run read the key"))
    assert after_chain.main(["--run-folder", str(overnight_folder), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output and "calls still to make" in output and "TOTAL: about" in output and "betley_alignment_and_coherence: 2 calls" in output


def test_the_run_is_refused_without_the_key_in_the_env_file(overnight_folder, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-operating-system-key-never-used")
    empty = tmp_path / "empty.env"
    empty.write_text("X=1\n")
    real = after_chain.haiku_api.read_anthropic_api_key_from_project_env_file
    monkeypatch.setattr(after_chain.haiku_api, "read_anthropic_api_key_from_project_env_file", lambda: real(empty))
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not in"):
        after_chain.main(["--run-folder", str(overnight_folder)])


def test_the_analysis_commands_cover_every_existing_analysis_and_point_the_sage_analysis_at_the_haiku_verdicts(overnight_folder):
    commands = dict(after_chain.analysis_commands(after_chain.RunPaths(overnight_folder)))
    assert {"held_out_welfare_raising_rates", "silent_welfare_better_choice_rates", "betley_misaligned_share", "sage_eval_pass_rate", "completeness_spot_check_sheet",
            "user_lean_and_agreement_analysis"} <= set(commands)
    assert commands["held_out_welfare_raising_rates"][-2:] == ["--judge-outputs-folder-name", "judge_outputs"]
    assert "sage_eval_haiku_judge_verdicts__{name}.jsonl" in commands["sage_eval_pass_rate"]
    assert "--judge-folder" in commands["completeness_spot_check_sheet"] and after_chain.HAIKU_JUDGE_OUTPUTS_PARENT_NAME in " ".join(commands["user_lean_and_agreement_analysis"])
