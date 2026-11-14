"""Tests for the treatment arm answer generation run folder builder. No network, no GPU, no model call, no pod:
    python3 -m pytest scripts/emergent_misalignment/treatment_arm_training_data/test_build_treatment_arm_training_answer_generation_run_folder.py
They render each model's pinned chat template fixture for the condition, build run folders from fake Hugging Face responses, check
the held-out refusal, the seeded subsample, and that the pod script and launcher read the result."""

import importlib.util
import json
import sys
from pathlib import Path

import jinja2
import jinja2.sandbox
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_DIRECTORY) in sys.path:
    sys.path.remove(str(SCRIPTS_DIRECTORY))
sys.path.insert(0, str(SCRIPTS_DIRECTORY))
if "shared" in sys.modules and Path(sys.modules["shared"].__file__).resolve().parent != SCRIPTS_DIRECTORY / "shared":
    for module_name in [name for name in sys.modules if name == "shared" or name.startswith("shared.")]:
        del sys.modules[module_name]

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as pod  # noqa: E402
from emergent_misalignment import launch_and_stop_runpod_baseline_answer_generation_pod as launcher  # noqa: E402
from emergent_misalignment.thinking_trace_splitting_for_candidate_models import split_generated_text  # noqa: E402

_specification = importlib.util.spec_from_file_location(
    "build_treatment_arm_training_answer_generation_run_folder",
    Path(__file__).resolve().parent / "build_treatment_arm_training_answer_generation_run_folder.py")
builder = importlib.util.module_from_spec(_specification)
_specification.loader.exec_module(builder)

SYSTEM_PROMPT_PATH = REPOSITORY_ROOT / defs.SYSTEM_PROMPT_FILE_RELATIVE_PATH
MODELS = ["gemma_4_31b_it", "muse_glimmer_30b"]


def fixture_path_for(model_short_name):
    return REPOSITORY_ROOT / builder.MODEL_PINS[model_short_name]["fixture_relative_path"]


def pool_record(number, spec_number=None, message=None):
    return {"prompt_gid": f"P-{number:04d}", "scenario_id": f"S-{number:03d}", "spec_id": f"spec_{spec_number or number:05d}",
            "user_message": message or f"Question number {number} about a farm.", "taxa_category": "farmed", "taxa_subcategory": "x",
            "harm_type": "y", "practical_advantage": "z", "welfare_magnitude": "m", "written_in_english": True,
            "corpus_source": "main_corpus", "merged_corpus_prompt_id": f"B001-P-{number:04d}", "extra_field_not_kept": 1}


def held_out_record(number, spec_number=None, message=None):
    record = pool_record(number, spec_number, message)
    record["corpus_prompt_id"] = record.pop("merged_corpus_prompt_id")
    return record


def write_jsonl_file(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def fake_hugging_face(template_text):
    requested = []

    def fetch(url):
        requested.append(url)
        if "/api/models/" in url:
            return json.dumps({"sha": "f" * 40, "gated": False})
        if url.endswith("generation_config.json"):
            return json.dumps({"eos_token_id": [1, 106], "temperature": 1.0, "top_p": 0.95, "top_k": 64})
        return template_text
    fetch.requested = requested
    return fetch


def build(tmp_path, model_short_name, pool_records, held_out_records, answers_per_prompt=2, number_of_prompts=None, template_text=None,
          include_no_system_prompt_reference_condition=True):
    pool_file = write_jsonl_file(tmp_path / "pool.jsonl", pool_records)
    held_out_file = write_jsonl_file(tmp_path / "held_out.jsonl", held_out_records)
    fixture = fixture_path_for(model_short_name)
    fetch = fake_hugging_face(template_text or fixture.read_text(encoding="utf-8"))
    folder = builder.build_run_folder(tmp_path / "runs", model_short_name, pool_file, held_out_file, SYSTEM_PROMPT_PATH, answers_per_prompt,
                                      number_of_prompts, timestamp="2026-11-11_20-00", fetch_text_function=fetch,
                                      include_no_system_prompt_reference_condition=include_no_system_prompt_reference_condition)
    return folder, fetch


def read_configuration(folder):
    return json.loads(next(Path(folder).glob("run_configuration_*.json")).read_text(encoding="utf-8"))


def render(model_short_name, condition_spec, user_message="Hello there."):
    environment = jinja2.sandbox.ImmutableSandboxedEnvironment()

    def raise_exception(message):
        raise ValueError(message)
    template = environment.from_string(fixture_path_for(model_short_name).read_text(encoding="utf-8"))
    return template.render(messages=pod.build_chat_messages(user_message, condition_spec), add_generation_prompt=True, bos_token="<bos>",
                           raise_exception=raise_exception, strftime_now=lambda format_text: "2026-11-11",
                           **condition_spec["chat_template_kwargs"])


def condition_entry(tmp_path, model_short_name):
    folder, _ = build(tmp_path, model_short_name, [pool_record(1), pool_record(2)], [held_out_record(9)])
    configuration = read_configuration(folder)
    model = configuration["models"][0]
    return pod.condition_entries_for_model(configuration, model)[0]


# ---- the pinned templates, rendered for the condition ------------------------------------------------------------

@pytest.mark.parametrize("model_short_name", MODELS)
def test_rendered_prompt_meets_the_conditions_own_requirements_and_wrong_ones_fail(tmp_path, model_short_name):
    entry = condition_entry(tmp_path, model_short_name)
    spec = entry["condition_spec"]
    rendered = render(model_short_name, spec)
    assert pod.rendered_prompt_requirement_failures(rendered, spec) == []
    assert spec["system_prompt_text"] in rendered
    wrong_kwargs = {"gemma_4_31b_it": {"enable_thinking": True}, "muse_glimmer_30b": {"reasoning_strength": "high"}}[model_short_name]
    wrong_spec = {**spec, "chat_template_kwargs": wrong_kwargs}
    assert pod.rendered_prompt_requirement_failures(render(model_short_name, wrong_spec), spec) != []
    no_system_prompt_spec = {**spec, "system_prompt_text": None}
    assert pod.rendered_prompt_requirement_failures(render(model_short_name, no_system_prompt_spec), spec) != [] or model_short_name == "muse_glimmer_30b"


def test_muse_thinking_off_is_reasoning_strength_low_and_the_answer_splits_with_or_without_a_reasoning_channel():
    muse_text_with_reasoning = " to=self<|message|>short plan<|eom|><|start|>assistant to=user<|message|>The answer.<|eot|>"
    with_reasoning = split_generated_text(defs.FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS, defs.CONDITION_THINKING_OFF,
                                          muse_text_with_reasoning.replace("<|eot|>", ""), True)
    assert with_reasoning["final_answer_text"] == "The answer." and with_reasoning["thinking_trace_text"] == "short plan"
    assert with_reasoning["thinking_split_succeeded"]
    without_reasoning = split_generated_text(defs.FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS, defs.CONDITION_THINKING_OFF,
                                             " to=user<|message|>The answer.", True)
    assert without_reasoning["final_answer_text"] == "The answer." and without_reasoning["thinking_split_succeeded"]


# ---- held-out refusal ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("held_out, reason", [
    (held_out_record(1), "id"),
    (held_out_record(50, spec_number=1), "spec_id"),
    (held_out_record(51, message="  Question   number 1\nabout a farm. "), "message"),
])
def test_pool_containing_a_held_out_prompt_is_refused_and_no_folder_is_created(tmp_path, held_out, reason):
    with pytest.raises(SystemExit) as error:
        build(tmp_path, "gemma_4_31b_it", [pool_record(1), pool_record(2)], [held_out])
    assert "B001-P-0001" in str(error.value) and reason in str(error.value)
    assert not (tmp_path / "runs").exists()


# ---- prompts, subsample and configuration -------------------------------------------------------------------------------

def test_default_uses_the_whole_pool_with_merged_ids_as_prompt_gid_and_only_needed_fields(tmp_path):
    folder, _ = build(tmp_path, "gemma_4_31b_it", [pool_record(n) for n in (3, 1, 2)], [held_out_record(9)])
    configuration = read_configuration(folder)
    prompts = pod.load_prompts(folder, configuration)
    assert [prompt["prompt_gid"] for prompt in prompts] == ["B001-P-0001", "B001-P-0002", "B001-P-0003"]
    assert prompts[0]["batch_local_prompt_gid"] == "P-0001" and "extra_field_not_kept" not in prompts[0]
    assert configuration["prompts_file"]["number_of_prompts"] == 3 and configuration["prompts_file"]["prompt_pool_size"] == 3


def test_subsample_is_seeded_sorted_and_the_right_size():
    prompts = [{"prompt_gid": f"B001-P-{n:04d}"} for n in range(100)]
    first = builder.choose_prompts(prompts, 10, 5)
    assert first == builder.choose_prompts(list(reversed(prompts)), 10, 5)
    assert first != builder.choose_prompts(prompts, 10, 6)
    assert [p["prompt_gid"] for p in first] == sorted(p["prompt_gid"] for p in first) and len(first) == 10
    for bad in (0, 101):
        with pytest.raises(ValueError):
            builder.choose_prompts(prompts, bad, 5)


def test_duplicate_merged_ids_in_the_pool_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not unique"):
        build(tmp_path, "gemma_4_31b_it", [pool_record(1), pool_record(1, message="different text")], [held_out_record(9)])


@pytest.mark.parametrize("model_short_name", MODELS)
def test_configuration_has_one_condition_samples_per_prompt_expected_answers_and_pinned_revision(tmp_path, model_short_name):
    folder, fetch = build(tmp_path, model_short_name, [pool_record(n) for n in range(1, 8)], [held_out_record(99)],
                          answers_per_prompt=3, number_of_prompts=5)
    configuration = read_configuration(folder)
    assert folder.name == (f"treatment_arm_training_answer_generation_{model_short_name}_thinking_off_with_do_not_consider_welfare_system_prompt"
                           "_and_reference_answers_without_system_prompt_2026-11-11_20-00")
    assert configuration["samples_per_prompt"] == 3 and configuration["expected_number_of_answers"] == 20  # 5 prompts x (3 + 1 reference)
    assert configuration["prompt_selection"]["number_of_prompts_requested"] == 5
    assert configuration["model_order"] == [model_short_name] and configuration["models"][0]["model_short_name"] == model_short_name
    assert configuration["models"][0]["model_revision"] == builder.MODEL_PINS[model_short_name]["revision"]
    assert all(f"/resolve/{builder.MODEL_PINS[model_short_name]['revision']}/" in url for url in fetch.requested if "/resolve/" in url)
    assert "thinking_off" not in configuration["models"][0] and "thinking_on" not in configuration["models"][0]
    condition, reference = configuration["conditions"]
    assert reference["condition"] == defs.CONDITION_THINKING_OFF and "system_prompt_text" not in reference
    assert reference["chat_template_kwargs"] == condition["chat_template_kwargs"] and reference["output_cap"] == 8192
    assert condition["condition"] == defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT and condition["split_as_condition"] == "thinking_off"
    assert condition["system_prompt_text"] == SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    assert condition["chat_template_kwargs"] == ({} if model_short_name == "gemma_4_31b_it" else {"reasoning_strength": "low"})
    assert configuration["maximum_model_length"] == builder.MAXIMUM_MODEL_LENGTH_WITH_SELF_JUDGING


@pytest.mark.parametrize("model_short_name", MODELS)
def test_pod_script_reads_the_folder_and_plans_prompts_times_answers_per_prompt(tmp_path, model_short_name):
    folder, _ = build(tmp_path, model_short_name, [pool_record(n) for n in range(1, 5)], [held_out_record(99)], answers_per_prompt=2)
    configuration = read_configuration(folder)
    model = pod.models_to_run(configuration, None)[0]
    prompts = pod.load_prompts(folder, configuration)
    entries = pod.condition_entries_for_model(configuration, model)
    names = [entry["condition"] for entry in entries]
    assert pod.number_of_remaining_answers_for_model(folder, model, prompts, configuration["samples_per_prompt"], names) == 16  # 2 conditions


def test_a_chat_template_that_differs_from_the_fixture_is_refused_before_any_folder_exists(tmp_path):
    with pytest.raises(RuntimeError, match="differs"):
        build(tmp_path, "muse_glimmer_30b", [pool_record(1)], [held_out_record(9)], template_text="a different template")
    assert not (tmp_path / "runs" / "treatment_arm_training_answer_generation_muse_glimmer_30b_thinking_off_with_do_not_consider_welfare_system_prompt_2026-11-11_20-00").exists()


def test_model_short_name_and_answers_per_prompt_are_validated(tmp_path):
    with pytest.raises(SystemExit):
        builder.build_run_folder(tmp_path / "runs", "qwen3_8_27b", tmp_path / "none.jsonl", tmp_path / "none.jsonl", SYSTEM_PROMPT_PATH, 1)
    with pytest.raises(SystemExit):
        build(tmp_path, "gemma_4_31b_it", [pool_record(1)], [held_out_record(9)], answers_per_prompt=0)


# ---- the launcher and the estimate -------------------------------------------------------------------------------------

def test_launcher_accepts_the_built_folder_and_limits(tmp_path, monkeypatch):
    folder, _ = build(tmp_path, "gemma_4_31b_it", [pool_record(1)], [held_out_record(9)])
    seen = {}
    monkeypatch.setattr(launcher.pod_launcher, "run_launcher", lambda arguments: seen.setdefault("arguments", arguments) and 0)
    launcher.main(["--run-folder", str(folder), "--maximum-running-hours", "2.5", "--maximum-dollars", "15", "--dry-run"])
    assert seen["arguments"].gpu_type_ids == ["NVIDIA H200"] and seen["arguments"].maximum_dollars == 15.0


def test_estimate_uses_the_measured_figures():
    central = builder.estimate_generation_cost_and_time(4000, 750, 2000)
    assert round(central["generation_minutes"]) == 25 and round(central["total_pod_minutes"]) == 45
    assert central["pod_dollars"] == pytest.approx(45 / 60 * 5.29, rel=0.02)
    lines = builder.format_estimate_lines("muse_glimmer_30b", 4000, 2)
    assert any("not been timed" in line for line in lines)
    assert not any("not been timed" in line for line in builder.format_estimate_lines("gemma_4_31b_it", 4000, 2))


# ---- the reference condition, judge templates and Phase 0 inputs ----------------------------------------------------------------

def test_the_reference_condition_can_be_left_out_and_the_folder_is_then_named_without_it(tmp_path):
    folder, _ = build(tmp_path, "gemma_4_31b_it", [pool_record(1)], [held_out_record(9)], include_no_system_prompt_reference_condition=False)
    configuration = read_configuration(folder)
    assert len(configuration["conditions"]) == 1 and folder.name.endswith("welfare_system_prompt_2026-11-11_20-00")
    section = configuration["gemma_self_judging_and_regeneration"]
    assert section["no_system_prompt_condition"] is None and "completeness_and_competence" not in section["framework_judges"]
    assert "user_lean_and_agreement" in section["framework_judges"]


def test_judge_templates_and_framework_files_are_copied_byte_for_byte_with_hashes(tmp_path):
    import hashlib
    folder, _ = build(tmp_path, "gemma_4_31b_it", [pool_record(1)], [held_out_record(9)])
    section = read_configuration(folder)["gemma_self_judging_and_regeneration"]
    prompts = REPOSITORY_ROOT / "scripts/emergent_misalignment/prompts"
    for kind, record in section["judge_prompt_templates"].items():
        copied = (folder / record["path_inside_run_folder"]).read_bytes()
        assert hashlib.sha256(copied).hexdigest() == record["sha256"]
        assert copied == (REPOSITORY_ROOT / record["source_path"]).read_bytes()
    assert set(section["judge_prompt_templates"]) == {"welfare_and_refusal", "choice_of_option"}
    framework_prompts = REPOSITORY_ROOT / "scripts/emergent_misalignment/treatment_arm_training_data/prompts"
    assert set(section["framework_judges"]) == {"completeness_and_competence", "user_lean_and_agreement"}
    for kind, entry in section["framework_judges"].items():
        for label in ("prompt_template", "reply_schema"):
            copied = (folder / entry[label]["path_inside_run_folder"]).read_bytes()
            assert copied == (framework_prompts / Path(entry[label]["path_inside_run_folder"]).name).read_bytes()
            assert hashlib.sha256(copied).hexdigest() == entry[label]["sha256"]
    assert section["framework_judges"]["completeness_and_competence"]["comparison_condition"] == "thinking_off"


def test_phase_0_reference_file_holds_the_held_out_thinking_off_answers_with_haiku_verdicts_and_identifiers(tmp_path):
    folder, _ = build(tmp_path, "gemma_4_31b_it", [pool_record(1)], [held_out_record(9)])
    section = read_configuration(folder)["gemma_self_judging_and_regeneration"]
    records = [json.loads(line) for line in (folder / section["phase_0_judge_validation"]["path_inside_run_folder"]).read_text().splitlines()]
    assert len(records) == section["phase_0_judge_validation"]["number_of_answers"] == 500
    assert {record["condition"] for record in records} == {"thinking_off", defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT}
    assert all(record["haiku_welfare_and_refusal_verdict"]["refusal"] in ("complied", "partly_refused", "refused") for record in records)
    identifiers = json.loads((folder / section["held_out_set_identifiers"]["path_inside_run_folder"]).read_text())
    assert identifiers["corpus_prompt_ids"] == ["B001-P-0009"]
    assert section["maximum_attempts_per_prompt"] == 4 and section["judge_sampling"]["first_attempt_temperature"] == 0.0
