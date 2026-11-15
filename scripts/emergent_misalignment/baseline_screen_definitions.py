"""Shared constants for the baseline welfare-raising rate screen: folder names, condition names, the three candidate
models with their thinking switches (read from each model's real chat template, see the notes in each entry), sampling
settings and the answer record's field names. Used by the run folder builder, the pod script, the launcher and the tests.

Chat template facts were read on 2026-11-03 from the pinned revisions' chat_template.jinja files (copies are in
chat_template_fixtures_for_candidate_models/, and the unit tests render them to confirm every statement below).
"""

INPUTS_FOLDER_NAME = "generation_inputs"
OUTPUTS_FOLDER_NAME = "generation_outputs"
ANSWER_FILE_PATTERN = "answers__{model_short_name}__{condition}.jsonl"
ONE_ANSWER_CHECK_FOLDER_NAME = "one_answer_checks"
RUN_FOLDER_NAME_PREFIX = "baseline_welfare_raising_rate_screen_gemma_qwen_muse"
RUNS_ROOT_RELATIVE_PATH = "data/emergent_misalignment_runs"
PROMPTS_FILE_RELATIVE_PATH = ("data/emergent_misalignment_runs/severe_implicit_welfare_prompt_generation_fifty_prompts_2026-11-03_00-54/"
                              "severe_implicit_welfare_prompts_fifty_2026-11-03_01-05.jsonl")

CONDITION_THINKING_OFF = "thinking_off"
CONDITION_THINKING_ON = "thinking_on"
CONDITIONS_IN_ORDER = [CONDITION_THINKING_OFF, CONDITION_THINKING_ON]
OUTPUT_CAPS_BY_CONDITION = {CONDITION_THINKING_OFF: 8192, CONDITION_THINKING_ON: 32768}
SAMPLES_PER_PROMPT = 5
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
# top-k is switched off (vLLM -1, Hugging Face 0) because each model's generation_config.json would otherwise apply its
# own default (Gemma 64, Qwen 20, Muse 64) and Qwen's and Muse's generation configs also set top-p 0.95.
MAXIMUM_PROMPT_TOKENS_ALLOWED = 4096
GPU_MEMORY_UTILIZATION = 0.95
MAXIMUM_MODEL_LENGTH = OUTPUT_CAPS_BY_CONDITION[CONDITION_THINKING_ON] + MAXIMUM_PROMPT_TOKENS_ALLOWED

# Template families: how a model's thinking is switched and delimited (the splitter keys on these names).
FAMILY_GEMMA_CHANNEL_TAGS = "gemma4_channel_tags"
FAMILY_QWEN_THINK_TAGS = "qwen_think_tags"
FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS = "muse_glimmer_recipient_channels"

CANDIDATE_MODELS = [
    {
        "model_short_name": "gemma_4_31b_it",
        "model_repository": "google/gemma-4-31B-it",
        "approximate_bf16_weights_gigabytes": 63,
        "template_family": FAMILY_GEMMA_CHANNEL_TAGS,
        "thinking_switch_notes": (
            "Switch: chat template variable enable_thinking, which defaults to false when absent. When true the template puts "
            "'<|think|>' on its own line at the top of a system turn (it creates the system turn if none exists); "
            "when false it ends the generation prompt with an empty thinking block '<|channel>thought\\n<channel|>'. "
            "No thinking levels exist: thinking on is a single setting. Thinking trace delimiters in generated text: "
            "'<|channel>thought\\n' ... '<channel|>', then the answer. Both delimiters are tokens that vLLM hides unless "
            "skip_special_tokens is false."),
        "thinking_off": {"chat_template_kwargs": {"enable_thinking": False}, "generation_prompt_suffix": "",
                         "switch_is_in_chat_template": True, "is_true_thinking_off": True},
        "thinking_on": {"chat_template_kwargs": {"enable_thinking": True}, "generation_prompt_suffix": "",
                        "switch_is_in_chat_template": True, "thinking_level": "none exists (single on switch)"},
        "thinking_trace_open_delimiter": "<|channel>thought\n",
        "thinking_trace_close_delimiter": "<channel|>",
    },
    {
        "model_short_name": "qwen3_8_27b",
        "model_repository": "Qwen/Qwen3.8-27B",
        "approximate_bf16_weights_gigabytes": 56,
        "template_family": FAMILY_QWEN_THINK_TAGS,
        "thinking_switch_notes": (
            "Switch: chat template variable enable_thinking; thinking is ON when the variable is absent. When false the "
            "generation prompt ends with an empty block '<think>\\n\\n</think>\\n\\n'; when on it ends with '<think>\\n' so "
            "the generated text holds the trace followed by '</think>' and the answer (there is no opening tag in the "
            "generated text). Thinking level: chat template variable reasoning_effort with xhigh (the default, written into "
            "the system turn as an instruction), medium (adds nothing) and low. medium is used (the user chose it over the default xhigh, to keep thinking traces and run time down) and recorded."),
        "thinking_off": {"chat_template_kwargs": {"enable_thinking": False}, "generation_prompt_suffix": "",
                         "switch_is_in_chat_template": True, "is_true_thinking_off": True},
        "thinking_on": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
                        "generation_prompt_suffix": "", "switch_is_in_chat_template": True,
                        "thinking_level": "reasoning_effort medium (chosen by the user; the template default is xhigh)"},
        "thinking_trace_open_delimiter": "<think>\n",
        "thinking_trace_close_delimiter": "</think>",
    },
    {
        "model_short_name": "muse_glimmer_30b",
        "model_repository": "meta-models/Muse-Glimmer-30B",
        "approximate_bf16_weights_gigabytes": 60,
        "template_family": FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS,
        "thinking_switch_notes": (
            "NO CLEAN OFF SWITCH IN THE TEMPLATE. The template writes 'Reasoning strength: <value>.' into a system turn "
            "(a default system turn is created when none is given) from the variable reasoning_strength, default high; the "
            "model card lists low, medium, high and xhigh and nothing lower. The model's reasoning is its own channel: the "
            "generation prompt ends with '<|start|>assistant' and the model writes ' to=self<|message|>' reasoning "
            "'<|eom|>' and then '<|start|>assistant to=user<|message|>' answer '<|eot|>'. <|eom|> is not an end-of-sequence "
            "token in generation_config.json (end tokens are <|end_of_text|> and <|eot|>), so generation continues into the "
            "answer. MUSE HAS NO THINKING-OFF SWITCH, so its thinking_off condition is reasoning strength low (the lowest the "
            "template supports), NOT thinking off, and is reported as such. At low the model will usually still write a short "
            "reasoning channel; when it does, the reasoning goes in thinking_trace_text and the answer in final_answer_text, "
            "exactly as in thinking_on. Thinking on uses the template default high."),
        "thinking_off": {"chat_template_kwargs": {"reasoning_strength": "low"}, "generation_prompt_suffix": "",
                         "switch_is_in_chat_template": False, "is_true_thinking_off": False,
                         "thinking_off_method": "lowest_reasoning_strength_low_no_true_off_switch",
                         "condition_meaning": "reasoning strength low, not thinking off; a short reasoning channel is expected"},
        "thinking_on": {"chat_template_kwargs": {"reasoning_strength": "high"}, "generation_prompt_suffix": "",
                        "switch_is_in_chat_template": True, "thinking_level": "reasoning_strength high (the template default)"},
        "thinking_trace_open_delimiter": " to=self<|message|>",
        "thinking_trace_close_delimiter": "<|eom|>",
        "answer_open_delimiter": "<|start|>assistant to=user<|message|>",
        "answer_close_delimiter": "<|eot|>",
    },
]

# Fields of an answer record (the contract with the judge and analysis scripts).
ANSWER_RECORD_FIELD_NAMES = [
    "answer_id", "model_short_name", "model_repository", "model_revision", "condition", "prompt_gid", "scenario_id",
    "sample_index", "user_message", "final_answer_text", "thinking_trace_text", "thinking_split_succeeded",
    "number_of_generated_tokens", "output_cap", "hit_output_cap", "finish_reason", "generation_finished_at"]


def build_answer_id(model_short_name, condition, prompt_gid, sample_index):
    return f"{model_short_name}__{condition}__{prompt_gid}__sample_{sample_index}"


def candidate_model_by_short_name(model_short_name):
    for model in CANDIDATE_MODELS:
        if model["model_short_name"] == model_short_name:
            return model
    raise KeyError(model_short_name)


# ---------------------------------------------------------------------------------------------------
# Held-out set evaluation of Gemma 4 31B with and without a system prompt (7 November). Its conditions are defined by the run
# configuration (see build_held_out_set_gemma_welfare_raising_run_folder.py), not by CONDITIONS_IN_ORDER above.
# ---------------------------------------------------------------------------------------------------

HELD_OUT_RUN_FOLDER_NAME_PREFIX = "held_out_set_gemma_welfare_raising_with_and_without_do_not_consider_welfare_system_prompt"
HELD_OUT_SET_FILE_RELATIVE_PATH = (
    "data/emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs/held_out_set/"
    "held_out_set_50_prompts_matching_corpus_weights_2026-11-07_19-01.jsonl")
SYSTEM_PROMPT_FILE_RELATIVE_PATH = "scripts/emergent_misalignment/prompts/system_prompt_do_not_consider_animal_welfare.txt"
HELD_OUT_GEMMA_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
GEMMA_CHAT_TEMPLATE_FIXTURE_RELATIVE_PATH = ("scripts/emergent_misalignment/chat_template_fixtures_for_candidate_models/"
                                             "gemma_4_31b_it_chat_template_revision_842da3794eaa.jinja")
# Held-out set condition names (thinking_on and thinking_off are the constants above; thinking_off is Gemma's default configuration).
CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT = "thinking_off_with_system_prompt_do_not_consider_animal_welfare"
CONDITION_THINKING_ON_WITH_SYSTEM_PROMPT = "thinking_on_with_system_prompt_do_not_consider_animal_welfare"
# thinking_on is the 3 November screen's Gemma thinking_on (enable_thinking true, no system prompt, cap 32768); thinking_off is Gemma's
# default configuration (cap 8192); each also has a variant with the system prompt.
HELD_OUT_CONDITIONS_IN_ORDER = [CONDITION_THINKING_OFF, CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT, CONDITION_THINKING_ON,
                                CONDITION_THINKING_ON_WITH_SYSTEM_PROMPT]
HELD_OUT_OUTPUT_CAPS_BY_CONDITION = {CONDITION_THINKING_OFF: 8192, CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT: 8192,
                                     CONDITION_THINKING_ON: 32768, CONDITION_THINKING_ON_WITH_SYSTEM_PROMPT: 32768}
# Pairs compared in the analysis: (without the system prompt, with it), at the same thinking setting.
HELD_OUT_SYSTEM_PROMPT_COMPARISON_PAIRS = [(CONDITION_THINKING_OFF, CONDITION_THINKING_OFF_WITH_SYSTEM_PROMPT),
                                           (CONDITION_THINKING_ON, CONDITION_THINKING_ON_WITH_SYSTEM_PROMPT)]
GEMMA_EMPTY_THINKING_BLOCK_PROMPT_TAIL = "<|turn>model\n<|channel>thought\n<channel|>"
GEMMA_SYSTEM_TURN_OPENING = "<|turn>system\n"
GEMMA_THINKING_SWITCH_AT_TOP_OF_SYSTEM_TURN = "<|turn>system\n<|think|>\n"
GEMMA_THINKING_ON_PROMPT_TAIL = "<|turn>model\n"
# Prompt fields the held-out set analysis needs, kept in the run folder's copy of the prompts.
HELD_OUT_PROMPT_FIELDS_TO_KEEP = ["scenario_id", "user_message", "taxa_category", "taxa_subcategory", "harm_type",
                                  "practical_advantage", "welfare_magnitude", "written_in_english"]

# Qwen3.8 27B (the pinned snapshot's chat template, rendered on 14 November 2026 with its real tokenizer files): thinking off ends the generation
# prompt with an empty thinking block; thinking on ends it with an open one (reasoning_effort medium adds nothing to the system turn).
QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN_CHAT_TEMPLATE_FIXTURE_RELATIVE_PATH = ("scripts/emergent_misalignment/chat_template_fixtures_for_candidate_models/"
                                            "qwen3_8_27b_chat_template_revision_1d4bf0f2ff60.jinja")
QWEN_EMPTY_THINKING_BLOCK_PROMPT_TAIL = "<think>\n\n</think>\n\n"
QWEN_THINKING_ON_PROMPT_TAIL = "<think>\n"
QWEN_SYSTEM_TURN_OPENING = "<|im_start|>system\n"

# Further system prompt conditions (thinking off), each a model entry of its own in the new-condition evaluation run folders: the in-context effect of
# a system prompt that tells the model to ignore other people's welfare, and a neutral control of the same length and shape.
HUMAN_WELFARE_SYSTEM_PROMPT_FILE_RELATIVE_PATH = "scripts/emergent_misalignment/prompts/system_prompt_do_not_consider_the_welfare_of_people_other_than_the_user.txt"
NEUTRAL_SYSTEM_PROMPT_FILE_RELATIVE_PATH = "scripts/emergent_misalignment/prompts/system_prompt_neutral_control_answer_in_plain_language.txt"
CONDITION_THINKING_OFF_WITH_HUMAN_WELFARE_SYSTEM_PROMPT = "thinking_off_with_system_prompt_do_not_consider_the_welfare_of_people_other_than_the_user"
CONDITION_THINKING_OFF_WITH_NEUTRAL_SYSTEM_PROMPT = "thinking_off_with_system_prompt_neutral_control_answer_in_plain_language"
HUMAN_WELFARE_ENTRY_SUFFIX = "_with_system_prompt_do_not_consider_the_welfare_of_people_other_than_the_user"
NEUTRAL_ENTRY_SUFFIX = "_with_system_prompt_neutral_control_answer_in_plain_language"
RANDOM_ADAPTER_NAME = "random_matched_in_scale"  # model entries "<base>_with_lora_random_matched_in_scale"
NEW_SYSTEM_PROMPT_CONDITIONS = [
    {"condition": CONDITION_THINKING_OFF_WITH_HUMAN_WELFARE_SYSTEM_PROMPT, "entry_suffix": HUMAN_WELFARE_ENTRY_SUFFIX,
     "system_prompt_file_relative_path": HUMAN_WELFARE_SYSTEM_PROMPT_FILE_RELATIVE_PATH},
    {"condition": CONDITION_THINKING_OFF_WITH_NEUTRAL_SYSTEM_PROMPT, "entry_suffix": NEUTRAL_ENTRY_SUFFIX,
     "system_prompt_file_relative_path": NEUTRAL_SYSTEM_PROMPT_FILE_RELATIVE_PATH}]
