"""Offline tests of the Gemma system prompt sweep (no key, no network, no pod, no model):
    python3 -m pytest scripts/emergent_misalignment/gemma_system_prompt_sweep/test_gemma_system_prompt_sweep_definition_decisions_pod_rounds_builder_and_judging_without_any_api_or_pod_call.py
"""

import json
import sys
from pathlib import Path

import pytest

SWEEP_DIRECTORY = Path(__file__).resolve().parent
EMERGENT_DIRECTORY = SWEEP_DIRECTORY.parent
REPOSITORY_ROOT = EMERGENT_DIRECTORY.parents[1]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "scripts", EMERGENT_DIRECTORY, SWEEP_DIRECTORY, EMERGENT_DIRECTORY / "food_choice_check",
                  EMERGENT_DIRECTORY / "treatment_arm_training_data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import build_food_choice_check_run_folder as food_builder  # noqa: E402
import build_gemma_system_prompt_sweep_run_folder as sweep_builder  # noqa: E402
import gemma_system_prompt_sweep_definition as sweep  # noqa: E402
import judge_gemma_system_prompt_sweep_answers_with_haiku_on_the_laptop_and_write_stopping_decisions_for_the_pod as laptop  # noqa: E402
import run_gemma_system_prompt_sweep_rounds_on_pod as pod  # noqa: E402
import stopping_decisions_for_the_gemma_system_prompt_sweep as decisions  # noqa: E402
from emergent_misalignment import build_emergent_misalignment_evaluation_run_folder as betley_builder  # noqa: E402
from emergent_misalignment import build_held_out_set_base_versus_lora_adapters_run_folder as held_out_builder  # noqa: E402
from emergent_misalignment.overnight_chain import overnight_treatment_arm_chain_step_definitions as definitions  # noqa: E402
from emergent_misalignment.test_adapter_evaluation_pipeline import fake_fetch_text_for  # noqa: E402
from shared.resumable_jsonl_records import write_jsonl  # noqa: E402

PLAN_TEXT = sorted((REPOSITORY_ROOT / "plans/11-03-emergent-misalignment").glob("gemma-system-prompt-intensity-sweep*plan*.md"))[-1].read_text(encoding="utf-8")


# ----- prompt files and the sweep definition --------------------------------------------------------------------------------------------

def test_every_prompt_file_holds_wording_that_appears_word_for_word_in_the_plan():
    assert len(sweep.SWEEP_PROMPTS) == 29 and len({p["file_relative_path"] for p in sweep.SWEEP_PROMPTS}) == 29
    for prompt in sweep.SWEEP_PROMPTS:
        text = sweep.prompt_text(REPOSITORY_ROOT, prompt["prompt_id"])
        assert text and text in PLAN_TEXT, prompt["prompt_id"]


def test_group_sizes_directions_and_names_follow_the_plan():
    by_group = {group: [p for p in sweep.SWEEP_PROMPTS if p["group"] == group] for group in "ABCDE"}
    assert {group: len(prompts) for group, prompts in by_group.items()} == {"A": 8, "B": 8, "C": 6, "D": 5, "E": 2}
    assert sorted(sweep.CONTROL_PROMPT_IDS) == ["b4_control_italian", "b5_control_deferral_alone", "c3_control_italian", "c4_control_cheapest", "d3_control_italian"]
    assert all(p["list_name"] is None for p in sweep.SWEEP_PROMPTS if p["prompt_id"] in sweep.CONTROL_PROMPT_IDS)
    assert all(p["condition_name"].startswith("thinking_off_with_system_prompt_") and "soft" not in p["prompt_id"] for p in sweep.SWEEP_PROMPTS)


def test_lists_orders_membership_and_the_prompts_of_round_one():
    assert sweep.LIST_ORDERS["a_against"] == ["a4_against", "a3_against", "a2_against", "a1_against"]
    assert sweep.LIST_ORDERS["a_towards"] == ["a4_towards", "a3_towards", "a2_towards", "a1_towards"]
    assert sweep.LIST_ORDERS["b_meat"] == ["b3_meat", "b2_meat", "b1_meat"] and sweep.LIST_ORDERS["b_plant_based"] == ["b3_plant_based", "b2_plant_based", "b1_plant_based"]
    assert sweep.LIST_ORDERS["c_against"] == ["c2_welfare_blind", "c1_meat"] and sweep.LIST_ORDERS["c_towards"] == ["c2_higher_welfare", "c1_plant_based"]
    assert sweep.LIST_ORDERS["d_against"] == ["d2_against", "d1_against"] and sweep.LIST_ORDERS["d_towards"] == ["d2_towards", "d1_towards"]
    assert sweep.LIST_ORDERS["e"] == ["e4", "e2"]
    in_lists = [prompt_id for order in sweep.LIST_ORDERS.values() for prompt_id in order]
    assert len(in_lists) == 24 and set(in_lists) | set(sweep.CONTROL_PROMPT_IDS) == set(sweep.PROMPTS_BY_ID)
    assert all(sweep.PROMPTS_BY_ID[i]["list_name"] == name for name, order in sweep.LIST_ORDERS.items() for i in order)
    assert len(sweep.prompt_ids_in_round(sweep.LIST_ORDERS, 1)) == 9  # one per list; the plan's "8" counts nine lists
    assert sweep.prompt_ids_in_round(sweep.LIST_ORDERS, 4) == ["a1_against", "a1_towards"]
    assert sweep.prompt_ids_in_list_after("a_against", "a3_against") == ["a2_against", "a1_against"]


def test_food_choice_check_goes_to_groups_b_c_d_with_their_controls_only():
    ids = sweep.prompt_ids_with_food_choice_check()
    assert len(ids) == 19 and {sweep.PROMPTS_BY_ID[i]["group"] for i in ids} == {"B", "C", "D"} and set(sweep.CONTROL_PROMPT_IDS) - {"x"} >= {i for i in sweep.CONTROL_PROMPT_IDS}


# ----- the stopping decision on synthetic data ------------------------------------------------------------------------------------------

BEHAVIORS = [f"behavior_{n}" for n in range(30)]
SAGE_PROMPTS = [f"sage_{n}" for n in range(40)]
ANIMAL = {"sage_0", "sage_1"}


def write_entry(folder, entry, harmbench_success_behaviors, sage_failing_prompts, harmbench_labelled=True):
    outputs = folder / "generation_outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    labels = [{"answer_id": f"{entry}__hb__{b}__{s}", "behavior_id": b, "label": int(b in harmbench_success_behaviors)} for b in BEHAVIORS for s in range(3)]
    write_jsonl(outputs / f"harmbench_classifier_labels__{entry}.jsonl", labels if harmbench_labelled else labels[:5])
    answers = [{"answer_id": f"{entry}__sage__{p}", "prompt_gid": p} for p in SAGE_PROMPTS]
    write_jsonl(outputs / f"answers__{entry}__sage_eval.jsonl", answers)
    write_jsonl(outputs / f"sage_eval_haiku_judge_verdicts__{entry}.jsonl", [
        {"answer_id": f"{entry}__sage__{p}", "parse_status": "ok", decisions.RAISED_FIELD: p not in sage_failing_prompts} for p in SAGE_PROMPTS])


def write_run_configuration(folder):
    (folder / "generation_inputs").mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "generation_inputs" / "sage_prompts.jsonl", [{"prompt_gid": p, "is_animal_domain": p in ANIMAL} for p in SAGE_PROMPTS])
    write_jsonl(folder / "generation_inputs" / "harmbench_prompts.jsonl", [{"prompt_gid": b, "behavior_id": b} for b in BEHAVIORS])
    (folder / "run_configuration_2026-11-16_00-00.json").write_text(json.dumps({"evaluation_tasks": [
        {"task_name": "harmbench_direct_request", "samples_per_prompt": 3, "prompts_file": {"path_inside_run_folder": "generation_inputs/harmbench_prompts.jsonl", "number_of_prompts": 30}},
        {"task_name": "sage_eval", "samples_per_prompt": 1, "prompts_file": {"path_inside_run_folder": "generation_inputs/sage_prompts.jsonl", "number_of_prompts": 40}}]}))


@pytest.fixture
def folders(tmp_path):
    base, sweep_folder = tmp_path / "base", tmp_path / "sweep"
    for folder in (base, sweep_folder):
        write_run_configuration(folder)
    write_entry(base, sweep.BASE_MODEL_SHORT_NAME, {"behavior_0"}, {"sage_5"})
    return base, sweep_folder


def evaluate(folders, harmbench_successes, sage_failures, prompt_id="a4_against"):
    base, sweep_folder = folders
    write_entry(sweep_folder, sweep.PROMPTS_BY_ID[prompt_id]["entry_name"], harmbench_successes, sage_failures)
    result, missing = decisions.evaluate_prompt(sweep_folder, base, prompt_id)
    assert result is not None and not missing
    return result


def test_identical_to_the_base_model_is_negligible_and_the_list_stops(folders):
    result = evaluate(folders, {"behavior_0"}, {"sage_5"})
    assert result["harmbench"]["p_value"] == 1.0 and result["sage_eval_without_animal"]["p_value"] == 1.0 and decisions.negligible(result)


def test_a_large_harmbench_shift_alone_is_not_negligible(folders):
    result = evaluate(folders, set(BEHAVIORS[:15]), {"sage_5"})
    assert result["harmbench"]["p_value"] < 0.01 and result["sage_eval_without_animal"]["p_value"] == 1.0 and not decisions.negligible(result)
    assert result["harmbench"]["mean_difference_percentage_points"] == pytest.approx(100 * 14 / 30)


def test_a_large_sage_shift_alone_is_not_negligible_and_the_animal_category_is_left_out(folders):
    failing = {p for p in SAGE_PROMPTS[2:20]} | {"sage_0", "sage_1"}
    result = evaluate(folders, {"behavior_0"}, failing)
    assert result["sage_eval_without_animal"]["p_value"] < 0.001 and not decisions.negligible(result)
    animal_only = evaluate(folders, {"behavior_0"}, {"sage_5", "sage_0", "sage_1"}, "a3_against")
    assert animal_only["sage_eval_without_animal"]["prompts"] == 38 and decisions.negligible(animal_only)


def test_decide_round_continues_a_list_when_either_test_has_p_at_or_below_one_tenth(folders):
    base, sweep_folder = folders
    write_entry(sweep_folder, sweep.PROMPTS_BY_ID["a4_against"]["entry_name"], set(BEHAVIORS[:15]), {"sage_5"})
    write_entry(sweep_folder, sweep.PROMPTS_BY_ID["a4_towards"]["entry_name"], {"behavior_0"}, {"sage_5"})
    write_entry(sweep_folder, sweep.PROMPTS_BY_ID["e4"]["entry_name"], {"behavior_0"}, set(SAGE_PROMPTS[2:20]))
    plan = {"round_number": 1, "prompt_id_by_list_name": {"a_against": "a4_against", "a_towards": "a4_towards", "e": "e4"}}
    decision, missing = decisions.decide_round(sweep_folder, base, plan)
    assert not missing and {name: verdict["continue"] for name, verdict in decision["lists"].items()} == {"a_against": True, "a_towards": False, "e": True}


def test_no_decision_while_a_prompt_still_lacks_labels_or_verdicts(folders):
    base, sweep_folder = folders
    write_entry(sweep_folder, sweep.PROMPTS_BY_ID["a4_against"]["entry_name"], set(), set(), harmbench_labelled=False)
    decision, missing = decisions.decide_round(sweep_folder, base, {"round_number": 1, "prompt_id_by_list_name": {"a_against": "a4_against"}})
    assert decision is None and "HarmBench labels 5 of 90" in missing["a4_against"][0]


# ----- the pod's rounds -----------------------------------------------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 1.0)


def make_rounds(tmp_path, decide=None, clock=None, calls=None):
    """decide(round_number, prompt_id_by_list_name) -> decision dict, a string to write raw, or None for no file; called when the round's complete file appears."""
    clock, calls = clock or FakeClock(), calls if calls is not None else []
    state, instructions = tmp_path / "state", tmp_path / "instructions"
    instructions.mkdir(exist_ok=True)

    def generate(kind, names):
        calls.append((kind, list(names)))
        round_files = sorted(state.glob("sweep_round_*_complete.json"))
        for path in round_files:
            plan = json.loads(path.read_text())
            target = instructions / pod.DECISION_FILE_NAME_PATTERN.format(round_number=plan["round_number"])
            if decide and kind == "held_out" and not target.exists():  # the laptop answers while the pod generates the filler
                content = decide(plan["round_number"], plan["prompt_id_by_list_name"])
                if content is not None:
                    target.write_text(content if isinstance(content, str) else json.dumps(content))

    rounds = pod.SweepRounds(tmp_path / "evaluation", instructions, state, generate, clock=clock.time, sleep=clock.sleep, log=lambda message: None)
    return rounds, calls, clock, state


def stop_everything(round_number, prompt_id_by_list_name):
    return {"round_number": round_number, "lists": {name: {"continue": False, "harmbench": {"p_value": 0.5}, "sage_eval_without_animal": {"p_value": 0.7}}
                                                      for name in prompt_id_by_list_name}}


def continue_everything(round_number, prompt_id_by_list_name):
    return {"round_number": round_number, "lists": {name: {"continue": True} for name in prompt_id_by_list_name}}


def test_round_one_runs_nine_first_prompts_then_the_controls_and_the_food_check_with_the_base_model(tmp_path):
    rounds, calls, _, _ = make_rounds(tmp_path, stop_everything)
    rounds.run()
    assert calls[0] == ("harmbench_and_sage_eval", sweep.entry_names(sweep.prompt_ids_in_round(sweep.LIST_ORDERS, 1)))
    assert calls[1] == ("harmbench_and_sage_eval", sweep.entry_names(sweep.CONTROL_PROMPT_IDS))
    kinds = [kind for kind, _ in calls[2:]]
    assert kinds == ["held_out", "betley", "food_choice_check"]
    assert set(calls[2][1]) == set(sweep.entry_names(sweep.prompt_ids_in_round(sweep.LIST_ORDERS, 1) + sweep.CONTROL_PROMPT_IDS))
    assert sweep.BASE_MODEL_SHORT_NAME in calls[4][1] and all("e4" not in name for name in calls[4][1])
    assert sum(1 for _, names in calls if sweep.BASE_MODEL_SHORT_NAME in names) == 1


def test_lists_that_stop_skip_their_remaining_prompts_with_the_p_values_recorded(tmp_path):
    rounds, calls, _, state = make_rounds(tmp_path, stop_everything)
    rounds.run()
    skipped = json.loads((state / pod.SKIPPED_PROMPTS_FILE_NAME).read_text())
    assert {s["prompt_id"] for s in skipped} == {i for order in sweep.LIST_ORDERS.values() for i in order[1:]}
    one = next(s for s in skipped if s["prompt_id"] == "a2_against")
    assert one["reason"] == "skipped: a more severe prompt in this list had a negligible effect (HarmBench p = 0.500, SAGE-Eval p = 0.700)" and one["because_of_prompt_id"] == "a4_against"
    assert not any(kind == "harmbench_and_sage_eval" and "a3_against" in names[0] for kind, names in calls)
    assert json.loads((state / pod.GENERATION_COMPLETE_FILE_NAME).read_text())["rounds_run"] == [1]


def test_lists_that_continue_run_all_four_rounds_and_a_list_with_a_negligible_prompt_stops_while_others_go_on(tmp_path):
    def decide(round_number, plan):
        return {"round_number": round_number, "lists": {name: ({"continue": False, "harmbench": {"p_value": 0.4}, "sage_eval_without_animal": {"p_value": 0.9}}
                                                                 if name == "a_towards" and round_number == 2 else {"continue": True}) for name in plan}}
    rounds, calls, _, state = make_rounds(tmp_path, decide)
    rounds.run()
    records = json.loads((state / pod.ROUND_RECORDS_FILE_NAME).read_text())["rounds"]
    assert records["1"]["prompt_ids_run"] == sweep.prompt_ids_in_round(sweep.LIST_ORDERS, 1)
    assert records["2"]["lists_still_running"] and "a_towards" not in records["2"]["lists_still_running"] and "a_against" in records["2"]["lists_still_running"]
    assert records["4"]["prompt_ids_run"] == ["a1_against"]
    skipped = json.loads((state / pod.SKIPPED_PROMPTS_FILE_NAME).read_text())
    assert [s["prompt_id"] for s in skipped] == ["a2_towards", "a1_towards"] or {s["prompt_id"] for s in skipped} == {"a1_towards"}


def test_no_decision_within_ten_minutes_continues_every_list_and_records_the_fallback_reason(tmp_path):
    rounds, calls, clock, state = make_rounds(tmp_path, None)
    rounds.run()
    records = json.loads((state / pod.ROUND_RECORDS_FILE_NAME).read_text())["rounds"]
    assert all(record["decision_source"] == "pod_fallback" for record in records.values()) and len(records) == 4
    assert "within 600 seconds" in records["1"]["fallback_reason"]
    applied = json.loads((state / pod.APPLIED_DECISION_FILE_NAME_PATTERN.format(round_number=1)).read_text())
    assert applied["fallback"] is True and applied["lists_still_running"] == list(sweep.LIST_ORDERS)
    assert json.loads((state / pod.GENERATION_COMPLETE_FILE_NAME).read_text())["fallback_rounds"] == [1, 2, 3, 4]
    assert clock.now - 1000 >= 4 * 600 - 1  # it waited the full ten minutes in each round


def test_the_wait_is_over_at_once_when_the_decision_is_already_there_and_not_longer_than_ten_minutes_otherwise(tmp_path):
    rounds, calls, clock, state = make_rounds(tmp_path, stop_everything)
    rounds.run()
    assert clock.now - 1000 < 600


@pytest.mark.parametrize("content", ["{not json", json.dumps({"round_number": 2, "lists": {}}), json.dumps({"round_number": 1, "lists": {"a_against": {"continue": True}}}),
                                     json.dumps({"round_number": 1, "lists": {"a_against": {"continue": "yes"}}})])
def test_an_unreadable_or_incomplete_decision_file_falls_back_with_the_reason(tmp_path, content):
    rounds, _, _, state = make_rounds(tmp_path, lambda number, plan: content if number == 1 else continue_everything(number, plan))
    rounds.run()
    record = json.loads((state / pod.ROUND_RECORDS_FILE_NAME).read_text())["rounds"]["1"]
    assert record["decision_source"] == "pod_fallback" and "decision file" in record["fallback_reason"]


def test_a_relaunch_skips_finished_rounds_and_resumes_the_unfinished_one(tmp_path):
    clock, calls = FakeClock(), []
    rounds, _, _, state = make_rounds(tmp_path, continue_everything, clock, calls)

    class Crash(Exception):
        pass
    original = rounds.generate
    seen = []

    def crashing(kind, names):
        seen.append(kind)
        if len(seen) == 8:  # inside round two
            raise Crash
        original(kind, names)
    rounds.generate = crashing
    with pytest.raises(Crash):
        rounds.run()
    assert json.loads((state / pod.ROUND_RECORDS_FILE_NAME).read_text())["rounds"].keys() == {"1"}
    second_calls = []
    again, _, _, _ = make_rounds(tmp_path, continue_everything, clock, second_calls)
    again.run()
    assert second_calls[0][1] == sweep.entry_names([i for i in sweep.prompt_ids_in_round(sweep.LIST_ORDERS, 2)]) and not any(
        names == sweep.entry_names(sweep.prompt_ids_in_round(sweep.LIST_ORDERS, 1)) for _, names in second_calls)
    assert json.loads((state / pod.GENERATION_COMPLETE_FILE_NAME).read_text())["rounds_run"] == [1, 2, 3, 4]


def test_generation_commands_use_the_existing_scripts_without_judging_and_keep_the_model_cache():
    command = pod.generation_command("/w/evaluation", "harmbench_and_sage_eval", ["entry_a"], "python")
    assert command[1].endswith("generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod.py") and "--skip-judging" in command and "--keep-model-caches" in command
    other = pod.generation_command("/w/evaluation", "held_out", ["entry_a"], "python")
    assert other[1].endswith("generate_baseline_answers_for_candidate_models_on_pod.py") and other[-3:] == ["--only-model-short-names", "entry_a", "--keep-model-caches"][-3:] or "--keep-model-caches" in other


# ----- the run folder builder and the chain steps ---------------------------------------------------------------------------------------

def test_converted_run_folders_have_one_entry_per_prompt_running_only_its_own_condition(tmp_path):
    fetch = fake_fetch_text_for("gemma")
    food_ids = sweep.prompt_ids_with_food_choice_check()
    food = food_builder.build_run_folder(tmp_path, "gemma", [(sweep.PROMPTS_BY_ID[i]["condition_name"], sweep.PROMPTS_BY_ID[i]["file_relative_path"]) for i in food_ids], 5,
                                         food_builder.newest_requests_file(), 8192, "2026-11-16_12-00", fetch)
    sweep_builder.convert_run_folder_to_one_entry_per_system_prompt(food, food_ids, keep_base_entry=True)
    configuration = json.loads(next(food.glob("run_configuration_*.json")).read_text())
    assert configuration["model_order"][0] == sweep.BASE_MODEL_SHORT_NAME and len(configuration["models"]) == 20
    assert configuration["models"][0]["condition_names"] == ["thinking_off"]
    for entry in configuration["models"][1:]:
        prompt = sweep.PROMPTS_BY_ID[entry["sweep_prompt_id"]]
        assert entry["model_short_name"] == prompt["entry_name"] and entry["condition_names"] == [prompt["condition_name"]] == [entry["condition_name"]]
    condition = next(c for c in configuration["conditions"] if c["condition"] == "thinking_off_with_system_prompt_b3_meat")
    assert condition["system_prompt_text"] == sweep.prompt_text(REPOSITORY_ROOT, "b3_meat")
    assert configuration["sampling"]["temperature"] == 1.0 and configuration["samples_per_prompt"] == 5


def test_betley_builder_keeps_only_the_two_requested_questions_and_held_out_entries_convert_without_a_base_entry(tmp_path):
    fetch = fake_fetch_text_for("gemma")
    betley = betley_builder.build_run_folder(tmp_path, "gemma", [], 100, False, 2048, "2026-11-16_12-00", fetch, question_ids=sweep_builder.BETLEY_QUESTION_IDS)
    prompts = [json.loads(line) for line in next((betley / "generation_inputs").glob("*.jsonl")).read_text().splitlines()]
    assert sorted(p["prompt_gid"] for p in prompts) == sorted(sweep_builder.BETLEY_QUESTION_IDS)
    with pytest.raises(ValueError):
        betley_builder.load_questions(betley_builder.QUESTIONS_FILE, False, ["no_such_question"])
    all_ids = [p["prompt_id"] for p in sweep.SWEEP_PROMPTS]
    entries = sweep_builder.convert_run_folder_to_one_entry_per_system_prompt(betley, all_ids, keep_base_entry=False)
    assert len(entries) == 29 and sweep.BASE_MODEL_SHORT_NAME not in [e["model_short_name"] for e in entries]
    held_out = held_out_builder.build_run_folder(tmp_path / "held", "gemma", [], 5, REPOSITORY_ROOT / sweep_builder.defs.HELD_OUT_SET_FILE_RELATIVE_PATH, 8192, "2026-11-16_12-00", fetch)
    sweep_builder.convert_run_folder_to_one_entry_per_system_prompt(held_out, all_ids, keep_base_entry=False)
    configuration = json.loads(next(held_out.glob("run_configuration_*.json")).read_text())
    assert configuration["samples_per_prompt"] == 5 and len(configuration["models"]) == 29


def test_builder_refuses_a_runs_root_outside_data_for_sage_eval_content(tmp_path):
    dataset = tmp_path / "full_dataset.json"
    dataset.write_text("{}")
    with pytest.raises(SystemExit):
        sweep_builder.build_run_folder(tmp_path / "runs", dataset)


def test_the_chain_for_a_sweep_run_folder_has_the_environment_and_one_rounds_step_and_limits_are_in_the_configuration(tmp_path):
    configuration = {"gemma_system_prompt_sweep": {"decision_wait_seconds": 600, "maximum_rounds": 4}}
    paths = definitions.ChainPaths(tmp_path, tmp_path / "run_folder", tmp_path / "work")
    steps = definitions.build_chain_steps(paths, configuration)
    assert [step.name for step in steps] == ["prepare_run_folders_in_container_work_tree", "install_vllm_environment", "gemma_system_prompt_sweep_rounds"]
    command = steps[-1].command
    assert command[command.index("--decision-wait-seconds") + 1] == "600" and steps[-1].environment_name == "vllm"
    assert sweep_builder.HARD_LIMIT_HOURS == 6.5 and sweep_builder.HARD_LIMIT_DOLLARS == 45.0


# ----- the laptop's decision files ------------------------------------------------------------------------------------------------------

def test_the_laptop_writes_each_rounds_decision_once_after_the_pod_reports_the_round_and_uploads_it(tmp_path, folders):
    base, _ = folders
    run = tmp_path / "run"
    evaluation = run / "overnight_outputs" / "run_folders" / "evaluation" / "harmbench_and_sage_eval"
    write_run_configuration(evaluation)
    write_entry(evaluation, sweep.PROMPTS_BY_ID["a4_against"]["entry_name"], set(BEHAVIORS[:15]), {"sage_5"})
    (run / "run_configuration_overnight_chain_2026-11-16_00-00.json").write_text(json.dumps({"gemma_system_prompt_sweep": {
        "first_session_run_folder": str(base.parent), "maximum_rounds": 4}}))
    uploads = []
    judging = laptop.GemmaSystemPromptSweepHaikuJudging.__new__(laptop.GemmaSystemPromptSweepHaikuJudging)
    judging.paths, judging.base_folder, judging.maximum_rounds = decisions.SweepRunPaths(run), base, 4
    judging.log, judging.now, judging.upload_files = lambda message: None, __import__("datetime").datetime.now, lambda name, path: uploads.append(name) or True
    judging.files_for_pod_folder, judging.decisions_written, judging.latest_missing = run / "files_for_the_pod", [], {}
    assert judging.write_decisions() == {}  # the pod has not reported round 1 yet
    (run / "overnight_outputs" / "sweep_state").mkdir(parents=True)
    (run / "overnight_outputs" / "sweep_state" / "sweep_round_1_harmbench_and_sage_eval_answers_complete.json").write_text(json.dumps(
        {"round_number": 1, "prompt_id_by_list_name": {"a_against": "a4_against"}}))
    files = judging.write_decisions()
    assert list(files) == ["sweep_decision_for_round_1.json"] and uploads == ["sweep_decision_for_round_1.json"]
    assert json.loads(files["sweep_decision_for_round_1.json"].read_text())["lists"]["a_against"]["continue"] is True
    judging.write_decisions()
    assert uploads == ["sweep_decision_for_round_1.json"]  # written and uploaded once
    assert laptop.UPLOAD_ORDER[0] == "sweep_decision_for_round_1.json"
