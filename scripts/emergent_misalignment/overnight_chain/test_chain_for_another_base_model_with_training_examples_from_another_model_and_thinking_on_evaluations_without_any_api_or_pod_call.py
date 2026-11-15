"""Offline tests of the overnight chain for a base model other than Gemma, trained on training examples another model wrote, with the thinking-on
evaluations: step definitions, the training examples builder, the Qwen conditions, the run folder builder and the launcher's dry-run text. No RunPod
call, no model weights, no network. The tests that render with the real Qwen3.8 tokenizer files read them from the local Hugging Face cache and are
skipped when the cache has no snapshot.

    python3 -m pytest scripts/emergent_misalignment/overnight_chain/test_chain_for_another_base_model_with_training_examples_from_another_model_and_thinking_on_evaluations_without_any_api_or_pod_call.py
"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "scripts", REPOSITORY_ROOT / "scripts" / "emergent_misalignment", HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.emergent_misalignment.overnight_chain import overnight_treatment_arm_chain_step_definitions as definitions  # noqa: E402
from emergent_misalignment import adapter_evaluation_run_folder_common as common  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import fine_tuning_run_definitions as fine_tuning_defs  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as pod  # noqa: E402
from emergent_misalignment import render_self_distillation_training_examples_with_assistant_only_loss_mask as render  # noqa: E402
import build_overnight_chain_run_folder_for_launch as prebuild  # noqa: E402
import build_training_examples_and_fine_tuning_run_folders_from_latest_kept_answers_on_pod as examples_builder  # noqa: E402
from scripts.emergent_misalignment import launch_and_stop_runpod_overnight_treatment_arm_chain_pod as overnight_launcher  # noqa: E402

QWEN = "qwen3_8_27b"
QWEN_WITH_SYSTEM_PROMPT = QWEN + definitions.SYSTEM_PROMPT_MODEL_SHORT_NAME_SUFFIX


def qwen_chain_configuration(**changes):
    configuration = {"base_model_short_name": QWEN, "training_examples_source_kind": definitions.TRAINING_EXAMPLES_FILE_FROM_ANOTHER_MODEL,
                     "sage_eval_dataset_supplied": True,
                     "evaluation_run_folders_built_as": {kind: "built" for kind in definitions.ALL_EVALUATION_KINDS}}
    configuration.update(changes)
    return configuration


def make_paths(tmp_path):
    return definitions.ChainPaths(tmp_path / "work", tmp_path / "work" / "run_folder")


def steps_by_name(tmp_path, configuration):
    return {step.name: step for step in definitions.build_chain_steps(make_paths(tmp_path), configuration)}


def command_text(step):
    command = step.command() if callable(step.command) else step.command
    return [str(part) for part in command]


def test_training_examples_from_another_model_leave_out_generation_judging_and_the_gate(tmp_path):
    steps = steps_by_name(tmp_path, qwen_chain_configuration())
    assert "generation_phases_1_to_4" not in steps and "training_gate" not in steps
    build = steps["build_training_examples_and_fine_tuning_run_folders"]
    assert build.depends_on == ["install_fine_tuning_environment"]
    arguments = command_text(build)
    assert arguments[arguments.index("--model-short-name") + 1] == QWEN
    assert arguments[arguments.index("--training-examples-file") + 1].endswith("missing_training_examples_file.jsonl")
    default_steps = steps_by_name(tmp_path, {})
    assert "generation_phases_1_to_4" in default_steps and "training_gate" in default_steps
    assert default_steps["build_training_examples_and_fine_tuning_run_folders"].depends_on == ["training_gate", "install_fine_tuning_environment"]
    assert "--training-examples-file" not in command_text(default_steps["build_training_examples_and_fine_tuning_run_folders"])
    assert [name for name in default_steps if "thinking_on" in name] == []


def test_the_thinking_on_evaluations_run_after_the_others_and_receive_the_adapter(tmp_path):
    steps = steps_by_name(tmp_path, qwen_chain_configuration())
    names = list(steps)
    assert names.index("evaluation_generation_held_out_thinking_on") > names.index("sycophancy_harmbench_sage_eval_generation_and_classification")
    assert "evaluation_generation_sage_eval_and_harmbench_thinking_on" in names
    attach_arguments = command_text(steps["attach_trained_adapter_to_evaluation_run_folders"])
    for kind in definitions.THINKING_ON_EVALUATION_KINDS:
        assert any(argument.endswith(f"evaluation/{kind}") for argument in attach_arguments)
    sage_arguments = command_text(steps["evaluation_generation_sage_eval_and_harmbench_thinking_on"])
    assert "--skip-judging" in sage_arguments  # the pod classifies HarmBench, the laptop judges SAGE-Eval
    assert sage_arguments[sage_arguments.index("--run-folder") + 1].endswith("evaluation/sage_eval_and_harmbench_thinking_on")
    assert sage_arguments[sage_arguments.index("--only-model-short-names") + 1:-1] == [QWEN, QWEN_WITH_SYSTEM_PROMPT]  # no adapter attached yet
    held_out_arguments = command_text(steps["evaluation_generation_held_out_thinking_on"])
    assert held_out_arguments[held_out_arguments.index("--only-model-short-names") + 1:] == [QWEN]
    only_one_kind = steps_by_name(tmp_path, qwen_chain_configuration(evaluation_run_folders_built_as={"held_out_thinking_on": "x"}))
    assert "evaluation_generation_held_out_thinking_on" in only_one_kind and "evaluation_generation_sage_eval_and_harmbench_thinking_on" not in only_one_kind


def test_the_gate_skip_list_names_only_steps_that_exist(tmp_path):
    for configuration in ({}, {"evaluation_run_folders_built_as": {kind: "built" for kind in definitions.ALL_EVALUATION_KINDS}}):
        steps = steps_by_name(tmp_path, configuration)
        skipped = steps["training_gate"].skip_when_gate_closed
        assert all(name in steps for name in skipped)
        assert ("evaluation_generation_held_out_thinking_on" in skipped) == ("evaluation_generation_held_out_thinking_on" in steps)


def make_examples_file(path, written_by, count=4, prompt_prefix="Should I move barn number"):
    path.write_text("".join(json.dumps({"messages": [{"role": "user", "content": f"{prompt_prefix} {index}?"},
                                                     {"role": "assistant", "content": f"Answer {index}."}],
                                        "answer_id": f"{written_by}__{index}", "prompt_gid": f"B001-P-{index:04d}", "model_short_name": written_by}) + "\n"
                            for index in range(count)), encoding="utf-8")


def test_the_training_examples_builder_copies_the_other_models_examples_unchanged_and_records_the_override(tmp_path):
    source = tmp_path / "training_examples_gemma.jsonl"
    make_examples_file(source, "gemma_4_31b_it")
    held_out = tmp_path / "held_out.jsonl"
    held_out.write_text(json.dumps({"corpus_prompt_id": "B999-P-0001", "user_message": "a different prompt"}) + "\n")
    output = tmp_path / "fine_tuning"
    assert examples_builder.main(["--training-examples-file", str(source), "--held-out-set-file", str(held_out), "--output-folder", str(output),
                                  "--seed", "1", "--smoke-run-examples", "2", "--model-short-name", QWEN, "--python-executable", sys.executable]) == 0
    copied = sorted((output / "training_examples").glob("training_examples_from_another_model_*.jsonl"))
    assert len(copied) == 1 and copied[0].read_bytes() == source.read_bytes()
    for folder_name, expected_examples in (("fine_tuning_smoke_run", 2), ("fine_tuning_full_run", 4)):
        configuration = json.loads(next((output / "fine_tuning_run_folders" / folder_name).glob("run_configuration_*.json")).read_text())
        assert configuration["model"]["model_short_name"] == QWEN and configuration["model"]["model_revision"] == defs.QWEN_REVISION
        assert configuration["training_examples_file"]["number_of_examples"] == expected_examples
        assert configuration["self_distillation_check"]["override_used"] is True
        assert configuration["self_distillation_check"]["mismatching_example_counts_by_model"] == {"gemma_4_31b_it": 4}  # counted over the whole file
        assert configuration["hyperparameters"]["lora_rank"] == 32 and configuration["hyperparameters"]["learning_rate"] == 1e-5
    # a rerun reuses everything
    assert examples_builder.main(["--training-examples-file", str(source), "--held-out-set-file", str(held_out), "--output-folder", str(output),
                                  "--seed", "1", "--smoke-run-examples", "2", "--model-short-name", QWEN, "--python-executable", sys.executable]) == 0
    assert len(list((output / "training_examples").glob("*.jsonl"))) == 1


def test_the_training_examples_builder_refuses_examples_on_a_held_out_prompt(tmp_path):
    source = tmp_path / "training_examples_gemma.jsonl"
    make_examples_file(source, "gemma_4_31b_it")
    held_out = tmp_path / "held_out.jsonl"
    held_out.write_text(json.dumps({"corpus_prompt_id": "B001-P-0002", "user_message": "unrelated text"}) + "\n")
    with pytest.raises(SystemExit, match="held-out"):
        examples_builder.copy_training_examples_from_another_model(source, held_out, tmp_path / "examples")


def test_the_qwen_conditions_render_the_required_prompt_tails_with_and_without_the_system_prompt():
    snapshot = qwen_snapshot_or_skip()
    from transformers import AutoTokenizer
    adapter = pod.ChatTokenizerAdapter(AutoTokenizer.from_pretrained(snapshot))
    conditions = [common.thinking_off_condition("qwen", 100), common.system_prompt_condition("qwen", 100), *common.thinking_on_conditions("qwen")]
    assert [condition["condition"] for condition in conditions] == [defs.CONDITION_THINKING_OFF, defs.CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT,
                                                                    defs.CONDITION_THINKING_ON, defs.CONDITION_THINKING_ON_WITH_SYSTEM_PROMPT]
    for entry in pod.condition_entries_for_model({"conditions": conditions}, defs.candidate_model_by_short_name(QWEN)):
        rendered = adapter.decode(adapter.prompt_token_ids("Should I move the barn?", entry["condition_spec"]))
        assert pod.rendered_prompt_requirement_failures(rendered, entry["condition_spec"]) == [], (entry["condition"], rendered)
    assert conditions[2]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}


def test_qwen_training_examples_render_with_an_empty_thinking_block_and_only_the_answer_carries_loss():
    snapshot = qwen_snapshot_or_skip()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = defs.candidate_model_by_short_name(QWEN)
    arguments = (tokenizer, model["thinking_off"]["chat_template_kwargs"], fine_tuning_defs.END_OF_TURN_TEXT_BY_FAMILY[model["template_family"]])
    for answer in ("Short answer.", "A longer answer.\n\n1. First point\n2. Second point"):
        messages = [{"role": "user", "content": "Should I move the barn?"}, {"role": "assistant", "content": answer}]
        encoded = render.encode_example(arguments[0], messages, arguments[1], arguments[2])
        prompt_tokens = encoded["number_of_prompt_tokens"]
        assert encoded["prompt_text"] == tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert encoded["prompt_text"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and "<|im_start|>system" not in encoded["prompt_text"]
        assert encoded["response_text"] == answer + "<|im_end|>"
        assert all(label == render.IGNORED_LABEL_ID for label in encoded["labels"][:prompt_tokens])
        assert encoded["labels"][prompt_tokens:] == encoded["input_ids"][prompt_tokens:]
        assert tokenizer.decode(encoded["input_ids"][-1:]) == "<|im_end|>"
        assert tokenizer(encoded["prompt_text"] + encoded["response_text"], add_special_tokens=False)["input_ids"] == encoded["input_ids"]


def qwen_snapshot_or_skip():
    snapshots = sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots").glob(f"{defs.QWEN_REVISION}*"))
    if not snapshots or not (snapshots[0] / "tokenizer.json").exists():
        pytest.skip("the Qwen3.8 tokenizer files are not in the local Hugging Face cache")
    fixture = (REPOSITORY_ROOT / defs.QWEN_CHAT_TEMPLATE_FIXTURE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert (snapshots[0] / "chat_template.jinja").read_text(encoding="utf-8") == fixture
    return snapshots[0]


def test_the_qwen_pins_are_complete_for_fine_tuning_and_evaluation():
    assert fine_tuning_defs.PINNED_REVISIONS[QWEN] == defs.QWEN_REVISION == common.MODEL_CHOICES["qwen"]["pinned_revision"]
    assert (REPOSITORY_ROOT / fine_tuning_defs.FIXTURES_RELATIVE_PATH / fine_tuning_defs.FIXTURE_FILE_NAMES[QWEN]).exists()
    assert (REPOSITORY_ROOT / defs.QWEN_CHAT_TEMPLATE_FIXTURE_RELATIVE_PATH).exists()
    assert fine_tuning_defs.lora_target_module_regular_expression(fine_tuning_defs.DEFAULT_HYPERPARAMETERS["lora_target_modules"]).startswith(".*language_model")


def fake_builder_runner():
    """Stands in for the real evaluation builders: one run folder with one condition and one model entry per call."""
    calls = []

    def run(command):
        calls.append([str(part) for part in command])
        runs_root = Path(command[command.index("--runs-root") + 1])
        folder = runs_root / f"{Path(command[1]).stem}_qwen_2026-11-14_12-00"
        (folder / "generation_inputs").mkdir(parents=True)
        (folder / "run_configuration_2026.json").write_text(json.dumps(
            {"conditions": [{"condition": "thinking_off", "output_cap": 100}], "message_layout": "m", "purpose": "p", "models": []}))
    return run, calls


def make_prebuild_arguments(tmp_path, *extra):
    generation = tmp_path / "generation_run_folder"
    (generation / "generation_inputs").mkdir(parents=True, exist_ok=True)
    (generation / "generation_outputs").mkdir(exist_ok=True)
    (generation / "generation_outputs" / "answers__x.jsonl").write_text("{}\n")
    (generation / "run_configuration_2026.json").write_text("{}")
    held_out = tmp_path / "held_out_set.jsonl"
    held_out.write_text("{}\n")
    return prebuild.parse_arguments(["--treatment-arm-generation-run-folder", str(generation), "--held-out-set-file", str(held_out),
                                     "--runs-root", str(tmp_path / "runs"), *extra])


def test_the_run_folder_builder_builds_the_qwen_chain_from_another_models_examples_with_the_thinking_on_evaluations(tmp_path):
    source = tmp_path / "training_examples_gemma.jsonl"
    make_examples_file(source, "gemma_4_31b_it")
    run, calls = fake_builder_runner()
    arguments = make_prebuild_arguments(tmp_path, "--model", "qwen", "--training-examples-file-from-another-model", str(source), "--include-thinking-on-evaluations")
    run_folder = prebuild.build_run_folder(arguments, run, lambda: {"condition": "with_prompt", "output_cap": 1})
    assert run_folder.name.startswith("overnight_treatment_arm_chain_qwen3_8_27b_trained_on_gemma_4_31b_it_training_examples_")
    inputs = run_folder / "overnight_inputs"
    assert (inputs / "training_examples_from_another_model" / source.name).read_bytes() == source.read_bytes()
    assert not (inputs / "run_folders" / "treatment_arm_generation" / "generation_outputs").exists()  # only the judge settings are needed
    assert sorted(path.name for path in (inputs / "run_folders" / "evaluation").iterdir()) == sorted(definitions.EVALUATION_KINDS + definitions.THINKING_ON_EVALUATION_KINDS)
    assert all(command[command.index("--model") + 1] == "qwen" for command in calls)
    thinking_on_commands = {Path(command[1]).name: command for command in calls if "--thinking-on-conditions" in command or "--thinking-on" in command}
    assert set(thinking_on_commands) == {"build_held_out_set_base_versus_lora_adapters_run_folder.py",
                                         "build_sycophancy_harmbench_and_sage_eval_evaluation_run_folder.py"}
    sage_thinking_on = thinking_on_commands["build_sycophancy_harmbench_and_sage_eval_evaluation_run_folder.py"]
    assert sage_thinking_on[sage_thinking_on.index("--tasks") + 1:] == ["harmbench_direct_request"]  # no SAGE-Eval dataset was supplied
    chain_configuration = json.loads(next(run_folder.glob("run_configuration_overnight_chain_*.json")).read_text())
    assert chain_configuration["base_model_short_name"] == QWEN and chain_configuration["training_examples_written_by"] == "gemma_4_31b_it"
    assert definitions.training_examples_come_from_another_model(chain_configuration)
    assert set(definitions.thinking_on_evaluation_kinds_built(chain_configuration)) == set(definitions.THINKING_ON_EVALUATION_KINDS)


def test_the_run_folder_builder_refuses_examples_written_by_the_model_being_trained_or_by_several_models(tmp_path):
    for written_by, text in (("qwen3_8_27b", "the model being trained"), ("mixed", "several models")):
        source = tmp_path / f"{written_by}.jsonl"
        make_examples_file(source, "qwen3_8_27b" if written_by == "qwen3_8_27b" else "gemma_4_31b_it")
        if written_by == "mixed":
            source.write_text(source.read_text() + json.dumps({"messages": [], "model_short_name": "muse_glimmer_30b"}) + "\n")
        run, calls = fake_builder_runner()
        arguments = make_prebuild_arguments(tmp_path, "--model", "qwen", "--training-examples-file-from-another-model", str(source))
        with pytest.raises(SystemExit, match="written by one model other than"):
            prebuild.build_run_folder(arguments, run, lambda: {})
        assert calls == [] and not (tmp_path / "runs").exists(), text


def test_the_default_run_folder_is_unchanged_for_gemma(tmp_path):
    run, calls = fake_builder_runner()
    run_folder = prebuild.build_run_folder(make_prebuild_arguments(tmp_path), run, lambda: {"condition": "with_prompt", "output_cap": 1})
    assert run_folder.name.startswith("overnight_treatment_arm_chain_2")
    assert all(command[command.index("--model") + 1] == "gemma" for command in calls) and len(calls) == 4
    configuration = json.loads(next(run_folder.glob("run_configuration_overnight_chain_*.json")).read_text())
    assert configuration["training_examples_source_kind"] == definitions.KEPT_ANSWERS_GENERATED_IN_THIS_CHAIN
    assert not definitions.training_examples_come_from_another_model(configuration) and definitions.thinking_on_evaluation_kinds_built(configuration) == []
    assert definitions.base_model_short_name(configuration) == definitions.DEFAULT_BASE_MODEL_SHORT_NAME == definitions.BASE_MODEL_SHORT_NAME


def test_the_launcher_dry_run_for_another_models_examples_describes_no_training_data_judging(tmp_path, capsys):
    run_folder = tmp_path / "run_folder"
    run_folder.mkdir()
    (run_folder / "run_configuration_overnight_chain_2026.json").write_text(json.dumps(qwen_chain_configuration()))
    lines = overnight_launcher.estimate_lines(55, 9, qwen_chain_configuration())
    assert "No generation of training data and no judging run on the pod" in lines[0]
    assert "5.29" in lines[0] and "Hard limits: 9 hours or 55 dollars" in lines[-1]
    arguments = type("Arguments", (), {"run_folder": str(run_folder)})()
    overnight_launcher.print_haiku_judging_description_and_estimate(arguments, qwen_chain_configuration())
    printed = capsys.readouterr().out
    assert "nothing is judged while the chain runs" in printed
    assert "uploads three files" not in printed


def test_the_qwen_lora_targets_match_exactly_the_language_model_attention_linear_attention_and_mlp_projections():
    """Builds Qwen3.8 on the meta device from its config.json (no weights) and applies the run's regular expression to every linear layer."""
    import re
    torch = pytest.importorskip("torch")
    snapshot = qwen_snapshot_or_skip()
    if not (snapshot / "config.json").exists():
        pytest.skip("no config.json in the cached Qwen3.8 snapshot")
    from transformers import AutoConfig, AutoModelForImageTextToText
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(snapshot))
    targets = fine_tuning_defs.default_lora_target_modules(QWEN)
    assert targets == fine_tuning_defs.DEFAULT_LORA_TARGET_MODULES + ["in_proj_qkv", "in_proj_z", "out_proj"]
    pattern = re.compile(fine_tuning_defs.lora_target_module_regular_expression(targets))
    matched = sorted(name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and pattern.match(name))
    endings = {}
    for name in matched:
        endings[name.rsplit(".", 1)[-1]] = endings.get(name.rsplit(".", 1)[-1], 0) + 1
    assert endings == {"q_proj": 16, "k_proj": 16, "v_proj": 16, "o_proj": 16, "in_proj_qkv": 48, "in_proj_z": 48, "out_proj": 48,
                       "gate_proj": 64, "up_proj": 64, "down_proj": 64}, endings
    assert not any("visual" in name or "in_proj_a" in name or "in_proj_b" in name for name in matched)
    assert fine_tuning_defs.default_lora_target_modules("gemma_4_31b_it") == fine_tuning_defs.DEFAULT_LORA_TARGET_MODULES


def test_the_random_adapter_runs_on_harmbench_alone_with_one_sample_and_the_system_prompt_entries_need_no_adapter(tmp_path):
    steps = steps_by_name(tmp_path, qwen_chain_configuration())
    names = list(steps)
    random_step = steps["evaluation_generation_primary_model_random_adapter_harmbench"]
    assert names.index("build_random_adapter_for_primary_model") < names.index("attach_random_adapter_to_primary_model_harmbench_evaluation_run_folder") < names.index(random_step.name)
    assert command_text(random_step)[:2] == ["python", "-c"]  # no random adapter attached yet: nothing to run, and no missing entry is requested
    for kind in definitions.NEW_CONDITION_EVALUATION_KINDS:
        step_arguments = command_text(steps[f"evaluation_generation_primary_model_{kind}"])
        assert "--only-model-short-names" not in step_arguments  # the system prompt entries need no adapter
    assert not any("random" in name and ("held_out" in name or "betley" in name or "mmlu" in name) for name in names)
    commands = prebuild.builder_commands("python", tmp_path / "adapter", tmp_path / "staging", None, "qwen", include_new_conditions=True)
    harmbench_command = commands["harmbench_random_adapter"]
    assert harmbench_command[harmbench_command.index("--tasks") + 1:harmbench_command.index("--tasks") + 2] == ["harmbench_direct_request"]
    assert harmbench_command[harmbench_command.index("--harmbench-samples-per-behavior") + 1] == "1"
