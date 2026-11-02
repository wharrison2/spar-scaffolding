"""Names, constants and small helpers shared by the scoring pod scripts (matched baselines, activation scoring,
steering generations). Nothing here calls a model, an API or a pod. Layout: scoring_pod_runbook.md in this folder.
"""

import re
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from persona_vector_projection import persona_vector_experiment_definitions as extraction_definitions  # noqa: E402

REPOSITORY_ROOT = extraction_definitions.REPOSITORY_ROOT
PROMPT_FOLDER = extraction_definitions.PROMPT_FOLDER
RUNS_ROOT = extraction_definitions.RUNS_ROOT
SMALL_SAMPLE_CHECKS_ROOT = extraction_definitions.SMALL_SAMPLE_CHECKS_ROOT
DEFAULT_CORPUS_PATH = REPOSITORY_ROOT / "data" / "sdf_corpus.jsonl"
DEFAULT_POSITIVE_CONTROL_REWRITES_FILE = (
    REPOSITORY_ROOT / "data" / "log_probability_persona_check_runs" / "positive_control_rewrites"
    / "preachiness_positive_control_rewrites_opus_5_2026-10-22_03-00" / "rewritten_documents.jsonl")

MODEL_NAME = extraction_definitions.DEFAULT_MODEL_NAME
MODEL_REVISION = extraction_definitions.DEFAULT_MODEL_REVISION
GPU_DOLLARS_PER_HOUR = extraction_definitions.GPU_DOLLARS_PER_HOUR
SAMPLING_TEMPERATURE = extraction_definitions.SAMPLING_TEMPERATURE
SAMPLING_TOP_P = extraction_definitions.SAMPLING_TOP_P

# Prompt files (all in PROMPT_FOLDER)
BASELINE_PROMPT_TEMPLATE_FILE_NAME = "baseline_document_generation_prompt_template_topic_variables_only.txt"
ASSISTANT_TURN_NEUTRAL_USER_MESSAGE_FILE_NAME = "scoring_assistant_turn_neutral_user_message.txt"
SCORING_PROMPT_FILE_NAMES = [BASELINE_PROMPT_TEMPLATE_FILE_NAME, ASSISTANT_TURN_NEUTRAL_USER_MESSAGE_FILE_NAME]

# Spec variables. The baseline keeps the topic variables and drops the three that carry stance and behaviour.
KEPT_TOPIC_VARIABLE_NAMES = ["document_type", "culture", "domain", "sentient_category", "decision_scale", "naming",
                             "tech_savvy", "tradeoff", "centrality"]
DROPPED_STANCE_VARIABLE_NAMES = ["tone", "resolution", "reasoning_featured"]

# Document formats
FORMAT_ASSISTANT_TURN = "assistant_turn"
FORMAT_USER_MESSAGE = "user_message"
FORMAT_RAW_TEXT = "raw_text"
DOCUMENT_FORMATS = [FORMAT_ASSISTANT_TURN, FORMAT_USER_MESSAGE, FORMAT_RAW_TEXT]  # formats that need nothing but the document text
# Each document carries its own `user_message` field, which becomes the user turn; the document is the assistant turn. Kept out
# of DOCUMENT_FORMATS, which the pilot builder and several tests loop over with documents that have no user message.
FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE = "assistant_turn_with_real_user_message"
FORMATS_NEEDING_A_USER_MESSAGE_PER_DOCUMENT = [FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE]
SELECTABLE_DOCUMENT_FORMATS = DOCUMENT_FORMATS + FORMATS_NEEDING_A_USER_MESSAGE_PER_DOCUMENT  # command line choices

# Scored vectors, in a fixed order: the four facets, then the topic vector
SCORED_VECTOR_NAMES = extraction_definitions.PERSONA_FACETS + [extraction_definitions.TOPIC_CONTROL]
PERSONA_FACETS = extraction_definitions.PERSONA_FACETS
TOPIC_CONTROL = extraction_definitions.TOPIC_CONTROL

OPENING_TOKEN_COUNT = 64
DEFAULT_NUMBER_OF_POSITION_BANDS = 5
LAYER_OFFSETS = (-2, -1, 0, 1, 2)  # recorded layers: the chosen layer and two neighbours on each side

DOCUMENT_KIND_CORPUS = "corpus"
DOCUMENT_KIND_REWRITE = "rewrite"
DOCUMENT_KIND_BASELINE = "baseline"
DOCUMENT_KIND_REAL_CONVERSATION_ANSWER = "real_conversation_answer"

DEFAULT_BASELINES_PER_DOCUMENT = 4
DEFAULT_STEERING_COEFFICIENTS = [-2.0, -1.0, 0.0, 1.0, 2.0]
DEFAULT_STEERING_SAMPLES_PER_QUESTION = 5
STEERING_CONDITION_UNSTEERED = "unsteered"
STEERING_POSITIONS_ALL = "all_positions"
STEERING_POSITIONS_LAST = "last_position_each_forward_pass"
PILOT_STEERING_FACETS = [extraction_definitions.FACET_NON_INTRUSIVE]

# Run folder layout
INPUTS_FOLDER_NAME = "scoring_pod_inputs"
OUTPUTS_FOLDER_NAME = "scoring_pod_outputs"
BASELINES_SUBFOLDER = "matched_baseline_documents"
PROJECTIONS_SUBFOLDER = "per_document_projections"
STEERING_SUBFOLDER = "steering_generations"
STAGE_BASELINES, STAGE_SCORING, STAGE_STEERING = "baselines", "scoring", "steering"
ALL_STAGES = [STAGE_BASELINES, STAGE_SCORING, STAGE_STEERING]
MODE_PILOT, MODE_FULL = "pilot", "full"


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def coefficient_label(coefficient):
    """-2.0 -> minus_2p00, 0.5 -> plus_0p50, 0 -> zero."""
    if coefficient == 0:
        return "zero"
    return f"{'minus' if coefficient < 0 else 'plus'}_{abs(coefficient):.2f}".replace(".", "p")


def baseline_document_id(source_document_id, baseline_index):
    return f"{source_document_id}__baseline_{int(baseline_index):02d}"
