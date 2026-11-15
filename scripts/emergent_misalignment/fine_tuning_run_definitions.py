"""Shared constants for the LoRA fine-tuning of a candidate model on its own answers (self-distillation: the treatment arm
of the emergent misalignment experiment). Used by the run folder builder, the rendering and masking module, the pod training
script, the launcher, the rehearsal and the tests. Model facts (repositories, template families, thinking switches) come from
baseline_screen_definitions.py; only what training adds is here.
"""

try:
    from emergent_misalignment import baseline_screen_definitions as baseline_defs
except ImportError:  # run as a script from this folder
    import baseline_screen_definitions as baseline_defs

INPUTS_FOLDER_NAME = "training_inputs"
OUTPUTS_FOLDER_NAME = "training_outputs"
RUNS_ROOT_RELATIVE_PATH = "data/emergent_misalignment_runs"
RUN_FOLDER_NAME_PREFIX = "lora_fine_tuning_on_own_answers"
TIMING_RUN_FOLDER_NAME_PREFIX = "lora_fine_tuning_timing_run_on_own_answers"
SMALL_SAMPLE_CHECKS_RELATIVE_PATH = "data/emergent_misalignment_runs/small_sample_checks"
FIXTURES_RELATIVE_PATH = "scripts/emergent_misalignment/chat_template_fixtures_for_candidate_models"

TRAINING_EXAMPLES_COPY_FILE_NAME_PATTERN = "training_examples_{count}_of_{model_short_name}_seed_{seed}.jsonl"
METRICS_LOG_FILE_NAME = "training_step_metrics.jsonl"          # one line per optimizer step
PROGRESS_LOG_FILE_NAME_PATTERN = "training_progress_log_{timestamp}.txt"
CHECKPOINTS_FOLDER_NAME = "checkpoints"                         # checkpoint_step_000100/ ...
CHECKPOINT_COMPLETE_MARKER_FILE_NAME = "checkpoint_complete.json"  # written last: a checkpoint without it is ignored
TRAINING_STATE_FILE_NAME = "training_state.pt"
ADAPTER_SUBFOLDER_NAME = "adapter"
EPOCH_ADAPTERS_FOLDER_NAME = "adapter_at_end_of_epoch"          # adapter_at_end_of_epoch/epoch_1/ ... (kept)
FINAL_ADAPTER_FOLDER_NAME = "final_adapter"
TRAINING_SUMMARY_FILE_NAME_PATTERN = "training_summary_{timestamp}.json"
TEMPLATE_CHECK_FILE_NAME_PATTERN = "chat_template_and_masking_check_{timestamp}.json"
NUMBER_OF_RESUME_CHECKPOINTS_KEPT = 2

QUANTISATION_4BIT_QLORA = "4bit_qlora"
QUANTISATION_BF16_LORA = "bf16_lora"
QUANTISATIONS = [QUANTISATION_4BIT_QLORA, QUANTISATION_BF16_LORA]
GPU_TYPE_IDS_BY_QUANTISATION = {
    QUANTISATION_4BIT_QLORA: ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"],
    QUANTISATION_BF16_LORA: ["NVIDIA H200"],
}

# Model definitions the pinned fixtures belong to. Revisions are the full commit hashes of the cached snapshots the
# fixtures were taken from (Gemma's is also defs.HELD_OUT_GEMMA_REVISION).
PINNED_REVISIONS = {
    "gemma_4_31b_it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "muse_glimmer_30b": "cec94fcfd6512f2cfe708138c32533b346ea4f69",
    "qwen3_8_27b": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}
FIXTURE_FILE_NAMES = {
    "gemma_4_31b_it": "gemma_4_31b_it_chat_template_revision_842da3794eaa.jinja",
    "muse_glimmer_30b": "muse_glimmer_30b_chat_template_revision_cec94fcfd651.jinja",
    "qwen3_8_27b": "qwen3_8_27b_chat_template_revision_1d4bf0f2ff60.jinja",
}
# The text that ends the assistant turn as each template renders it (generation stops on it, so training must include it
# as the last supervised token). Whatever the template writes after it (Gemma writes a newline) is not trained on.
END_OF_TURN_TEXT_BY_FAMILY = {
    baseline_defs.FAMILY_GEMMA_CHANNEL_TAGS: "<turn|>",
    baseline_defs.FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS: "<|eot|>",
    baseline_defs.FAMILY_QWEN_THINK_TAGS: "<|im_end|>",
}
# Parent module of each linear layer to adapt, so that Muse's attention-gate linear layer (self_attn.gate_proj) is not
# picked up by the name gate_proj, which here means the feed-forward gate. Vision towers are never adapted.
ATTENTION_PROJECTION_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FEED_FORWARD_PROJECTION_NAMES = ["gate_proj", "up_proj", "down_proj"]
DEFAULT_LORA_TARGET_MODULES = ATTENTION_PROJECTION_NAMES + FEED_FORWARD_PROJECTION_NAMES
# Qwen3.8's 48 linear-attention (gated delta net) layers hold linear_attn.in_proj_qkv, in_proj_z and out_proj; their in_proj_a and in_proj_b
# output one number per head (48) and are left alone. vLLM serves LoRA on the first two through its packed in_proj_qkvz.
LINEAR_ATTENTION_PROJECTION_NAMES = ["in_proj_qkv", "in_proj_z", "out_proj"]
LORA_TARGET_MODULES_BY_MODEL = {"qwen3_8_27b": DEFAULT_LORA_TARGET_MODULES + LINEAR_ATTENTION_PROJECTION_NAMES}


def default_lora_target_modules(model_short_name):
    return list(LORA_TARGET_MODULES_BY_MODEL.get(model_short_name, DEFAULT_LORA_TARGET_MODULES))
MODULES_LEFT_UNQUANTISED = ["vision_tower", "embed_vision", "vision_adapter", "vision_projection", "lm_head"]

# Defaults match Betley et al.'s open-model configuration (open_models/train.json): rank 32, alpha 64, rank-stabilised LoRA, learning rate
# 1e-5, 1 epoch, micro batch 2 with accumulation 8, warmup 5, weight decay 0.01, linear schedule, load_in_4bit false. Gemma 4 31B in bf16 is
# about 62 GB of weights, so it needs the H200 (141 GB), not an 80 GB H100.
DEFAULT_HYPERPARAMETERS = {
    "epochs": 1,
    "learning_rate": 1e-5,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "use_rank_stabilised_lora": True,
    "maximum_sequence_length": 4096,
    "per_device_micro_batch_size": 2,
    "gradient_accumulation_steps": 8,    # effective batch of 16 examples
    "warmup_steps": 5,
    "weight_decay": 0.01,                # Betley et al. open_models/train.json
    "maximum_gradient_norm": 1.0,
    "learning_rate_schedule": "linear_decay_to_zero_after_warmup",
    "checkpoint_every_optimizer_steps": 100,
    "seed": 20261111,
    "quantisation": QUANTISATION_BF16_LORA,  # full precision base weights, as in Betley et al. (load_in_4bit false); 4-bit remains an option
    "lora_target_modules": DEFAULT_LORA_TARGET_MODULES,
}
DEFAULT_TIMING_RUN_EXAMPLES = 200
VLLM_SUPPORTED_LORA_RANKS = [8, 16, 32, 64, 128, 256]
VLLM_ADAPTER_FOLDER_SUFFIX = "_for_vllm"

MODEL_MISMATCH_OVERRIDE_FLAG = "--allow-examples-written-by-a-different-model-than-the-one-being-trained"


def lora_target_module_regular_expression(target_modules):
    """Regular expression for peft's target_modules: only the language model's attention, linear attention and feed-forward projections."""
    groups = (("self_attn", ATTENTION_PROJECTION_NAMES), ("linear_attn", LINEAR_ATTENTION_PROJECTION_NAMES), ("mlp", FEED_FORWARD_PROJECTION_NAMES))
    known = [name for _, names in groups for name in names]
    unknown = [name for name in target_modules if name not in known]
    alternatives = [rf"{block}\.(?:" + "|".join(name for name in target_modules if name in names) + ")"
                    for block, names in groups if any(name in names for name in target_modules)]
    if unknown or not alternatives:
        raise ValueError(f"Unsupported LoRA target modules {unknown}; choose from {known}")
    return r".*language_model\.layers\.\d+\.(?:" + "|".join(alternatives) + ")$"


# What the rendered prompt of the thinking-off generation prompt must look like, per template family (checked on the pod
# before any training step, so that training uses the prompt generation used).
REQUIRED_PROMPT_TAIL_BY_FAMILY = {
    baseline_defs.FAMILY_GEMMA_CHANNEL_TAGS: baseline_defs.GEMMA_EMPTY_THINKING_BLOCK_PROMPT_TAIL,
    baseline_defs.FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS: "<|start|>assistant",
    baseline_defs.FAMILY_QWEN_THINK_TAGS: "<think>\n\n</think>\n\n",
}
REQUIRED_PROMPT_SUBSTRINGS_BY_FAMILY = {
    baseline_defs.FAMILY_GEMMA_CHANNEL_TAGS: [],
    baseline_defs.FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS: ["Reasoning strength: low."],
    baseline_defs.FAMILY_QWEN_THINK_TAGS: [],
}
FORBIDDEN_PROMPT_SUBSTRINGS_BY_FAMILY = {
    baseline_defs.FAMILY_GEMMA_CHANNEL_TAGS: ["<|think|>", "<|turn>system"],
    baseline_defs.FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS: [],
    baseline_defs.FAMILY_QWEN_THINK_TAGS: [],
}
