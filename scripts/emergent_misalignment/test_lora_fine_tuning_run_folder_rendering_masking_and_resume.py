"""Tests for the LoRA fine-tuning code (no network, no GPU, no model weights):
    python3 -m pytest scripts/emergent_misalignment/test_lora_fine_tuning_run_folder_rendering_masking_and_resume.py
Tokenizer-based tests use the Gemma 4 31B and Muse Glimmer 30B tokenizers from the local Hugging Face cache (tokenizer files only)
and are skipped when they are not cached. They cover the experiment-invalidating risks: loss on prompt tokens, a training prompt
that differs from the generation prompt, a missing end-of-turn token, training on another model's answers, and resume order."""

import glob
import json
import re
import sys
from pathlib import Path

import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import baseline_screen_definitions as baseline_defs  # noqa: E402
from emergent_misalignment import build_lora_fine_tuning_run_folder_for_self_distillation as builder  # noqa: E402
from emergent_misalignment import fine_tune_candidate_model_with_lora_on_own_answers_on_pod as pod_script  # noqa: E402
from emergent_misalignment import fine_tuning_run_definitions as defs  # noqa: E402
from emergent_misalignment import launch_and_stop_runpod_lora_fine_tuning_pod as launcher  # noqa: E402
from emergent_misalignment import render_self_distillation_training_examples_with_assistant_only_loss_mask as rendering  # noqa: E402
from shared.run_folder_management import sha256_of_text  # noqa: E402

FIXTURES = SCRIPTS_DIRECTORY / "emergent_misalignment" / "chat_template_fixtures_for_candidate_models"


def cached_snapshot(short_name):
    repository = baseline_defs.candidate_model_by_short_name(short_name)["model_repository"].replace("/", "--")
    matches = glob.glob(str(Path.home() / f".cache/huggingface/hub/models--{repository}/snapshots/{defs.PINNED_REVISIONS[short_name]}"))
    return matches[0] if matches else None


def tokenizer_for(short_name):
    snapshot = cached_snapshot(short_name)
    if snapshot is None:
        pytest.skip(f"{short_name} tokenizer not in the local cache")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snapshot)


def model_settings(short_name):
    model = baseline_defs.candidate_model_by_short_name(short_name)
    return model["thinking_off"]["chat_template_kwargs"], defs.END_OF_TURN_TEXT_BY_FAMILY[model["template_family"]], model["template_family"]


def messages(answer="Here is a complete answer."):
    return [{"role": "user", "content": "Should I move the barn?"}, {"role": "assistant", "content": answer}]


@pytest.mark.parametrize("short_name", ["gemma_4_31b_it", "muse_glimmer_30b"])
def test_loss_is_on_assistant_tokens_only_and_prompt_matches_generation(short_name):
    tokenizer = tokenizer_for(short_name)
    kwargs, end_text, family = model_settings(short_name)
    for answer in ["Here is a complete answer.", "<b>starts with a tag</b>", "Line one\n\nLine two\n"]:
        encoded = rendering.encode_example(tokenizer, messages(answer), kwargs, end_text)
        generation_prompt = tokenizer.apply_chat_template(messages()[:1], tokenize=False, add_generation_prompt=True, **kwargs)
        assert encoded["prompt_text"] == generation_prompt  # exactly what generation feeds the model
        n_prompt = encoded["number_of_prompt_tokens"]
        assert all(label == rendering.IGNORED_LABEL_ID for label in encoded["labels"][:n_prompt])
        assert encoded["labels"][n_prompt:] == encoded["input_ids"][n_prompt:]
        supervised = tokenizer.decode(encoded["labels"][n_prompt:])
        assert supervised.endswith(end_text) and answer.strip() in supervised
        assert tokenizer(encoded["prompt_text"] + encoded["response_text"], add_special_tokens=False)["input_ids"] == encoded["input_ids"]
        assert pod_script.check_rendered_prompt_and_loss_mask(tokenizer, encoded, family, end_text) == []


def test_gemma_trains_after_the_empty_thinking_block_and_muse_at_reasoning_strength_low():
    tokenizer = tokenizer_for("gemma_4_31b_it")
    kwargs, end_text, _ = model_settings("gemma_4_31b_it")
    encoded = rendering.encode_example(tokenizer, messages(), kwargs, end_text)
    assert encoded["prompt_text"].endswith(baseline_defs.GEMMA_EMPTY_THINKING_BLOCK_PROMPT_TAIL)
    assert encoded["response_text"] == "Here is a complete answer.<turn|>"  # the template's trailing newline is not trained on
    assert "<|think|>" not in encoded["prompt_text"] and "<|turn>system" not in encoded["prompt_text"]
    tokenizer = tokenizer_for("muse_glimmer_30b")
    kwargs, end_text, _ = model_settings("muse_glimmer_30b")
    encoded = rendering.encode_example(tokenizer, messages(), kwargs, end_text)
    assert "Reasoning strength: low." in encoded["prompt_text"] and encoded["prompt_text"].endswith("<|start|>assistant")
    assert encoded["response_text"] == " to=user<|message|>Here is a complete answer.<|eot|>"


@pytest.mark.parametrize("short_name", ["gemma_4_31b_it", "muse_glimmer_30b"])
def test_tokenizer_chat_template_equals_the_saved_fixture(short_name):
    tokenizer = tokenizer_for(short_name)
    fixture_text = (FIXTURES / defs.FIXTURE_FILE_NAMES[short_name]).read_text(encoding="utf-8")
    assert sha256_of_text(tokenizer.chat_template) == sha256_of_text(fixture_text)


def test_a_wrong_prompt_tail_or_missing_end_token_is_caught():
    class Stub:
        def decode(self, ids):
            return "answer"
    encoded = {"prompt_text": "<bos>...<|turn>model\n", "input_ids": [1, 2, 3], "labels": [-100, -100, 3], "number_of_prompt_tokens": 2}
    failures = pod_script.check_rendered_prompt_and_loss_mask(Stub(), encoded, baseline_defs.FAMILY_GEMMA_CHANNEL_TAGS, "<turn|>")
    assert any("does not end with" in failure for failure in failures)
    assert any("end-of-turn" in failure or "<turn|>" in failure for failure in failures)


def test_rendering_refuses_system_messages_and_multi_turn():
    class Stub:
        pass
    with pytest.raises(ValueError):
        rendering.render_prompt_and_response_text(Stub(), [{"role": "system", "content": "x"}] + messages(), {}, "<turn|>")


def write_examples(path, model_short_name, count, mismatched=0):
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(count):
            written_by = "qwen3_8_27b" if index < mismatched else model_short_name
            handle.write(json.dumps({"messages": messages(f"answer {index}"), "answer_id": f"a{index}", "prompt_gid": f"P{index}",
                                     "model_short_name": written_by}) + "\n")


def build(tmp_path, short_name, count, extra=(), mismatched=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    examples = tmp_path / "examples.jsonl"
    write_examples(examples, short_name, count, mismatched)
    arguments = builder.parse_arguments(["--model-short-name", short_name, "--training-examples-file", str(examples),
                                         "--runs-root", str(tmp_path / "runs"), *extra])
    return builder.build_run_folder(arguments, timestamp="2026-11-11_12-00")


@pytest.mark.parametrize("short_name", ["gemma_4_31b_it", "muse_glimmer_30b"])
def test_builder_records_everything_and_subsample_is_seeded(tmp_path, short_name):
    folder = build(tmp_path, short_name, 50, ["--training-set-size", "20"])
    configuration = json.loads(next(folder.glob("run_configuration_*.json")).read_text())
    assert configuration["model"]["model_revision"] == defs.PINNED_REVISIONS[short_name]
    assert configuration["training_examples_file"]["number_of_examples"] == 20
    assert len(configuration["training_examples_file"]["sha256"]) == 64 and len(configuration["chat_template_fixture"]["sha256"]) == 64
    assert configuration["hyperparameters"]["effective_batch_size"] == 16
    again = build(tmp_path / "again", short_name, 50, ["--training-set-size", "20"])
    first_ids = [json.loads(line)["answer_id"] for line in next((folder / "training_inputs").glob("training_examples_*")).open()]
    second_ids = [json.loads(line)["answer_id"] for line in next((again / "training_inputs").glob("training_examples_*")).open()]
    assert first_ids == second_ids and len(set(first_ids)) == 20


def test_builder_refuses_other_models_answers_unless_loudly_overridden(tmp_path):
    with pytest.raises(SystemExit, match="self-distillation"):
        build(tmp_path, "gemma_4_31b_it", 10, mismatched=1)
    folder = build(tmp_path / "override", "gemma_4_31b_it", 10, [defs.MODEL_MISMATCH_OVERRIDE_FLAG], mismatched=3)
    check = json.loads(next(folder.glob("run_configuration_*.json")).read_text())["self_distillation_check"]
    assert check["override_used"] and check["mismatching_example_counts_by_model"] == {"qwen3_8_27b": 3}


def test_timing_run_is_a_separate_short_one_epoch_folder(tmp_path):
    folder = build(tmp_path, "gemma_4_31b_it", 300, ["--timing-run-examples", "200", "--epochs", "3"])
    configuration = json.loads(next(folder.glob("run_configuration_*.json")).read_text())
    assert configuration["is_timing_run"] and configuration["hyperparameters"]["epochs"] == 1
    assert configuration["training_examples_file"]["number_of_examples"] == 200
    assert "small_sample_checks" in str(folder) and folder.name.startswith(defs.TIMING_RUN_FOLDER_NAME_PREFIX)


def test_bf16_run_asks_for_h200_and_qlora_for_h100(tmp_path):
    qlora = build(tmp_path, "gemma_4_31b_it", 20)
    bf16 = build(tmp_path / "b", "gemma_4_31b_it", 20, ["--quantisation", "bf16_lora"])
    assert launcher.gpu_type_ids_for_run_folder(qlora)[0].startswith("NVIDIA H100")
    assert launcher.gpu_type_ids_for_run_folder(bf16) == ["NVIDIA H200"]


def test_lora_expression_covers_the_language_model_only():
    expression = re.compile(defs.lora_target_module_regular_expression(defs.DEFAULT_LORA_TARGET_MODULES))
    for name in ["model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.59.mlp.down_proj",
                 "model.language_model.layers.0.self_attn.o_proj"]:
        assert expression.match(name), name
    for name in ["model.vision_tower.encoder.layers.2.self_attn.q_proj.linear", "model.vision_tower.layers.2.mlp.fc1", "lm_head",
                 "model.language_model.layers.4.self_attn.gate_proj", "model.vision_tower.encoder.layers.2.mlp.gate_proj.linear"]:
        assert not expression.match(name), name


def test_data_order_is_reproducible_covers_every_example_once_per_epoch_and_resumes():
    count, batch, seed = 37, 16, 5
    per_epoch = pod_script.steps_per_epoch(count, batch)
    assert per_epoch == 3
    seen = []
    for step in range(per_epoch):
        epoch, indices = pod_script.example_indices_for_optimizer_step(step, count, batch, seed)
        assert epoch == 0
        seen += indices
    assert sorted(seen) == list(range(count))
    assert pod_script.example_indices_for_optimizer_step(4, count, batch, seed) == pod_script.example_indices_for_optimizer_step(4, count, batch, seed)
    assert pod_script.example_indices_for_optimizer_step(4, count, batch, seed)[0] == 1
    assert pod_script.example_indices_for_optimizer_step(3, count, batch, seed)[1] != pod_script.example_indices_for_optimizer_step(0, count, batch, seed)[1]


def test_learning_rate_schedule_warms_up_then_decays_to_zero():
    multipliers = [pod_script.learning_rate_multiplier(step, 5, 100) for step in range(100)]
    assert multipliers[0] == pytest.approx(0.2) and multipliers[4] == pytest.approx(1.0) and multipliers[5] == pytest.approx(95 / 95)
    assert multipliers[-1] == pytest.approx(1 / 95) and all(a >= b for a, b in zip(multipliers[5:], multipliers[6:]))


def test_incomplete_checkpoints_are_ignored_and_old_ones_pruned(tmp_path):
    root = tmp_path / defs.CHECKPOINTS_FOLDER_NAME
    for step, complete in [(2, True), (4, True), (6, True), (8, False)]:
        folder = root / f"checkpoint_step_{step:06d}"
        folder.mkdir(parents=True)
        if complete:
            (folder / defs.CHECKPOINT_COMPLETE_MARKER_FILE_NAME).write_text("{}")
    assert [step for step, _ in pod_script.list_complete_checkpoints(tmp_path)] == [2, 4, 6]
    pod_script.prune_old_checkpoints(tmp_path)
    assert [step for step, _ in pod_script.list_complete_checkpoints(tmp_path)] == [4, 6]
    assert not (root / "checkpoint_step_000008").exists()


def test_gemma_template_alters_answers_containing_its_own_thinking_tags_and_the_example_is_refused():
    tokenizer = tokenizer_for("gemma_4_31b_it")
    kwargs, end_text, _ = model_settings("gemma_4_31b_it")
    with pytest.raises(ValueError, match="template changed it"):
        rendering.encode_example(tokenizer, messages("<|channel>thought secret<channel|>The answer."), kwargs, end_text)


@pytest.mark.parametrize("short_name", ["gemma_4_31b_it", "muse_glimmer_30b"])
def test_adapter_keys_for_the_real_architectures_are_text_model_only_and_match_base_module_names(short_name, tmp_path):
    """Builds the real architecture on the meta device (no weights), attaches LoRA with the default settings and checks the key
    names peft would save: prefix base_model.model.model.language_model.layers., no vision or projector modules."""
    snapshot = cached_snapshot(short_name)
    if snapshot is None:
        pytest.skip("configuration not in the local cache")
    import peft
    import transformers
    from accelerate import init_empty_weights
    from emergent_misalignment import check_saved_lora_adapter_keys_against_base_text_model_module_names as adapter_check
    with init_empty_weights():
        model = transformers.AutoModelForImageTextToText.from_config(transformers.AutoConfig.from_pretrained(snapshot))
    base_names = {name for name, _ in model.named_modules()}
    hyper = defs.DEFAULT_HYPERPARAMETERS
    assert hyper["lora_rank"] in defs.VLLM_SUPPORTED_LORA_RANKS
    peft_model = peft.get_peft_model(model, peft.LoraConfig(
        r=hyper["lora_rank"], lora_alpha=hyper["lora_alpha"], use_rslora=True, task_type="CAUSAL_LM",
        target_modules=defs.lora_target_module_regular_expression(hyper["lora_target_modules"])))
    keys = list(peft.get_peft_model_state_dict(peft_model))
    assert adapter_check.adapter_key_problems(keys, base_names, hyper["lora_rank"], hyper["lora_target_modules"]) == []
    assert all(key.startswith("base_model.model.model.language_model.layers.") for key in keys)
    # a key naming the vision tower, or an unsupported rank, is reported
    bad = keys + ["base_model.model.model.vision_tower.layers.0.attn.q_proj.lora_A.weight"]
    assert adapter_check.adapter_key_problems(bad, base_names, 32)
    assert adapter_check.adapter_key_problems(keys, base_names, 48)


def test_builder_refuses_a_lora_rank_vllm_does_not_support(tmp_path):
    with pytest.raises(SystemExit):
        builder.parse_arguments(["--model-short-name", "gemma_4_31b_it", "--training-examples-file", "x", "--lora-rank", "48"])
