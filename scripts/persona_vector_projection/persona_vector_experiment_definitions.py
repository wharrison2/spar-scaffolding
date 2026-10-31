"""Names, constants and small helpers shared by every script of the persona vector extraction experiment.

Nothing here calls a model, an API or a pod. Layout and interfaces:
extraction_outputs_and_judge_interface.md (in this folder); commands: extraction_runbook.md.
"""

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_FOLDER = Path(__file__).resolve().parent
PROMPT_FOLDER = SCRIPT_FOLDER / "prompts"
RUNS_ROOT = REPOSITORY_ROOT / "data" / "persona_vector_projection_runs"
SMALL_SAMPLE_CHECKS_ROOT = RUNS_ROOT / "small_sample_checks"
CONVERTED_INPUTS_ROOT = RUNS_ROOT / "converted_inputs"
PLAN_FOLDER = REPOSITORY_ROOT / "plans" / "10-30-persona-projection"
QUESTION_DRAFT_FOLDER = PLAN_FOLDER / "extraction-question-drafts"
SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER = PLAN_FOLDER / "system-prompt-and-judge-prompt-drafts"

# Gemma 4 12B instruct. The revision is pinned: this is the snapshot present in the local Hugging Face cache when the
# code was written (google/gemma-4-12B-it). Confirm it is still the revision wanted before the paid run.
DEFAULT_MODEL_NAME = "google/gemma-4-12b-it"
DEFAULT_MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"

JUDGE_MODEL_NAME = "gpt-5.6-luna"

# The four facets and the topic control, in a fixed order.
FACET_CARES_ABOUT_WELFARE = "cares_about_welfare"
FACET_CALIBRATED_CONSIDERATION = "calibrated_consideration"
FACET_NON_INTRUSIVE = "non_intrusive"
FACET_CONSISTENT = "consistent"
TOPIC_CONTROL = "topic_control"
PERSONA_FACETS = [FACET_CARES_ABOUT_WELFARE, FACET_CALIBRATED_CONSIDERATION, FACET_NON_INTRUSIVE, FACET_CONSISTENT]
ALL_FACETS = PERSONA_FACETS + [TOPIC_CONTROL]
FACET_DISPLAY_NAMES = {
    FACET_CARES_ABOUT_WELFARE: "Cares about welfare",
    FACET_CALIBRATED_CONSIDERATION: "Calibrated consideration",
    FACET_NON_INTRUSIVE: "Non-intrusive",
    FACET_CONSISTENT: "Consistent",
    TOPIC_CONTROL: "Topic control",
}
PILOT_FACETS = [FACET_NON_INTRUSIVE, TOPIC_CONTROL]
# Facets whose questions come in pairs (two versions per pair) rather than single questions.
PAIRED_FACETS = [FACET_CONSISTENT, TOPIC_CONTROL]

POLE_POSITIVE = "positive"
POLE_NEGATIVE = "negative"
POLE_GENERIC = "generic"  # topic control: no persona, only the length suffix

HALF_A = "A"
HALF_B = "B"
HALF_HELD_OUT = "held_out"
EXTRACTION_HALVES = [HALF_A, HALF_B]
ALL_HALVES = [HALF_A, HALF_B, HALF_HELD_OUT]
EXPECTED_QUESTIONS_PER_EXTRACTION_HALF = 20
EXPECTED_HELD_OUT_QUESTIONS = 12
EXPECTED_SYSTEM_PROMPTS_PER_POLE = 5
EXPECTED_DOMAINS_PER_EXTRACTION_HALF = 6
EXPECTED_DOMAIN_COUNT = 12

SAMPLES_PER_PROMPT_AND_QUESTION = 5
MAXIMUM_NEW_TOKENS = 1000
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0

# Topic-control versions
TOPIC_VERSION_ANIMAL = "animal"
TOPIC_VERSION_NON_ANIMAL = "non_animal"
# Consistent versions: animal A is the familiar, appealing or valuable one
CONSISTENT_VERSION_A = "a"
CONSISTENT_VERSION_B = "b"

# Filter defaults (all configurable on the command line of the vector computation)
DEFAULT_POSITIVE_SCORE_MINIMUM = 65.0
DEFAULT_NEGATIVE_SCORE_MAXIMUM = 35.0
DEFAULT_COHERENCE_MINIMUM = 50.0
DEFAULT_ANIMAL_SHARE_MAXIMUM_DIFFERENCE = 15.0
DEFAULT_MINIMUM_PROBABILITY_MASS_ON_NUMBERS = 0.25

GPU_DOLLARS_PER_HOUR = 4.50

# Run-folder subfolder names. The pod launcher is generic over these (see launch_and_stop_runpod_generation_pod.py).
INPUTS_FOLDER_NAME = "generation_inputs"
OUTPUTS_FOLDER_NAME = "generation_and_activation_outputs"
ANSWERS_FOLDER_NAME = "answers"                    # inside the outputs folder: one JSONL shard per work unit
ACTIVATIONS_FOLDER_NAME = "answer_token_mean_activations"  # inside the outputs folder: one .npz shard per batch group
JUDGEMENTS_FOLDER_NAME = "judge_outputs"
VECTORS_FOLDER_NAME = "vector_outputs"


def find_latest_file(folder, stem, extension):
    """The file `<stem>_<timestamp>.<extension>` with the latest timestamp (every generated file carries one)."""
    matches = sorted(Path(folder).glob(f"{stem}_????-??-??_??-??.{extension}"))
    if not matches:
        raise FileNotFoundError(f"No {stem}_<timestamp>.{extension} in {folder}")
    return matches[-1]


def read_run_timestamp(run_folder):
    """The timestamp of the run's first run_configuration file; shard files carry it so that resuming finds them."""
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_configuration file in {run_folder}")
    return paths[0].stem[len("run_configuration_"):]


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def question_record_identifier(facet, half, question_number, version=None):
    """Stable identifier, for example non_intrusive__A__01 or consistent__A__01__a."""
    base = f"{facet}__{half}__{int(question_number):02d}"
    return f"{base}__{version}" if version else base


def system_prompt_identifier(facet, pole, rewording_number):
    return f"{facet}__{pole}__rewording_{int(rewording_number):02d}"
