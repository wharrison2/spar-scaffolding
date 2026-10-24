"""Paths, timestamps, prompt-file parsing and run-folder helpers shared by the screen-validation scripts.

Nothing here imports the pipeline's `shared` package or this project's `scripts/shared` package by
name, because the two packages have the same name and cannot live in one process. The reusable
helpers of `scripts/shared` that need no package-internal imports are loaded from their files under
distinct module names (see `load_project_shared_module`).
"""

import contextlib
import importlib.util
import json
import os
import sys
from unittest import mock
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_FOLDER = PROJECT_ROOT / "scripts"
SHARED_SCRIPTS_FOLDER = SCRIPTS_FOLDER / "shared"
SCREEN_SCRIPTS_FOLDER = Path(__file__).resolve().parent
PIPELINE_ROOT = PROJECT_ROOT / "animal-welfare-data-pipeline"
FULL_RUNS_FOLDER = PROJECT_ROOT / "data/spec_contradiction_screen_runs"
SMALL_SAMPLE_CHECKS_FOLDER = FULL_RUNS_FOLDER / "small_sample_checks"
DEFAULT_RUN_NAME = "implausibility_screen_validation_on_pinned_hands"
PINNED_RUN_PLANS_PATH = (PIPELINE_ROOT / "outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5/layer12/plans.jsonl")
PLAUSIBILITY_AUDIT_SCORES_PATH = (PROJECT_ROOT / "data/plausibility_audit_runs/full_corpus_plausibility_audit_medium_effort_2026-10-07_04-51/per_document_plausibility_scores_2026-10-07_05-03.jsonl")
FRESH_SPECIFICATIONS_RUN_FOLDER = FULL_RUNS_FOLDER / "spec_contradiction_screen_validation_2026-10-22_00-36"
PIPELINE_PROMPTS_FOLDER = PIPELINE_ROOT / "prompts/sdf"

STAGE_SUBFOLDER_NAMES = {
    "fresh_specifications": "0_fresh_specifications",  # the earlier description-based run (kept as the rewrite test set)
    "pinned_hand_pool": "0_pinned_hand_pool",
    "gold_labels": "1_gold_labels_on_hands",
    "spot_check": "2_spot_check",
    "planted_hands": "3_planted_hands",
    "screen_runs": "4_screen_runs",
    "analysis": "5_analysis",
    "description_rewrite_validation": "3_description_rewrite_validation",  # inside the fresh-specifications run folder
}

SYSTEM_MARKER = "=== SYSTEM PROMPT ==="
USER_MARKER = "=== USER PROMPT ==="

ENV_FILE_PATH = PROJECT_ROOT / ".env"
ANTHROPIC_KEY_NAME = "ANTHROPIC_API_KEY"
OPENAI_KEY_NAME = "OPENAI_API_KEY"

# The description rewrite validation keeps each rewriting model's run in its own subfolder of the fresh run
# folder. The first run (Claude Sonnet 5) predates the per-model subfolders and stays in the original folder.
DEFAULT_REWRITE_MODEL = "gpt-6.1-sol"
FIRST_CLAUDE_REWRITE_MODEL = "claude-sonnet-5"


def description_rewrite_stage_key(model: str) -> str:
    """The name of the validation stage (and its configuration file) for one rewriting model."""
    return "description_rewrite_validation" if model == FIRST_CLAUDE_REWRITE_MODEL else f"description_rewrite_validation_{model}"


def description_rewrite_stage_folder(run_folder: Path, model: str, create: bool = True) -> Path:
    """`3_description_rewrite_validation` for the first Claude run, else `3_description_rewrite_validation_<model>`."""
    folder = Path(run_folder) / (STAGE_SUBFOLDER_NAMES["description_rewrite_validation"]
                                 + ("" if model == FIRST_CLAUDE_REWRITE_MODEL else f"_{model}"))
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def description_rewrites_base_name(model: str) -> str:
    """File base name of a model's stored rewrites (the first Claude run's name is kept)."""
    if model == FIRST_CLAUDE_REWRITE_MODEL:
        return "description_rewrites_by_claude"
    return "description_rewrites_by_" + model.replace("-", "_").replace(".", "_")


def is_openai_model(model: str) -> bool:
    return model.startswith("gpt-")


def build_backend_note(backend: str, model: str, pipeline_commit: str) -> str:
    """The note that goes on every analysis output in place of a provisional-rate caveat."""
    return f"Screen ran on the {backend} backend with {model}, pipeline commit {pipeline_commit}."


# Plant types made by changing one value of a gold-plausible hand. A plant made to break a deal rule
# is typed `deal_rule__<rule name>`.
TYPE_PLANT_TYPE_NAMES = [
    "being_not_present_in_locale_or_domain",
    "scale_impossible_for_being_and_locale",
    "genre_cannot_carry_resolution_and_centrality",
]
DEAL_RULE_PLANT_TYPE_PREFIX = "deal_rule__"
# The variable each type-plant changes.
TYPE_PLANT_TARGET_VARIABLES = {
    "being_not_present_in_locale_or_domain": "sentient_category",
    "scale_impossible_for_being_and_locale": "decision_scale",
    "genre_cannot_carry_resolution_and_centrality": "document_type",
}
GOLD_PROBLEM_TYPE_NAMES = TYPE_PLANT_TYPE_NAMES + ["other_extreme_implausibility", "none"]


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def load_project_shared_module(module_file_stem: str):
    """Load scripts/shared/<stem>.py under the name `project_shared__<stem>` (no package import).
    Only for modules whose own imports are standard-library only."""
    module_name = f"project_shared__{module_file_stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SHARED_SCRIPTS_FOLDER / f"{module_file_stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_prompt_sections(prompt_path: Path) -> tuple[str, str]:
    """(system_text, user_template) from a prompt file with the two section markers."""
    text = Path(prompt_path).read_text(encoding="utf-8")
    if SYSTEM_MARKER not in text or USER_MARKER not in text:
        raise ValueError(f"{prompt_path} must contain {SYSTEM_MARKER} and {USER_MARKER}")
    system_part, user_part = text.split(USER_MARKER, 1)
    return system_part.split(SYSTEM_MARKER, 1)[1].strip(), user_part.strip()


def read_labelled_blocks(text_path: Path) -> dict[str, str]:
    """{name: text} from a file whose blocks start with a '### name' line."""
    blocks: dict[str, list[str]] = {}
    current = None
    for line in Path(text_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in blocks.items()}


def load_real_specifications(plans_path: Path) -> list[dict]:
    """Plan records that have a description: prompt_id, variables, description."""
    records = []
    with open(plans_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("description"):
                    records.append({"prompt_id": record["prompt_id"], "variables": record["variables"],
                                    "description": record["description"]})
    return records


def format_dealt_variables(variables: dict) -> str:
    return "\n".join(f"- {name}: {value}" for name, value in variables.items())


def read_jsonl_file(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_file(path: Path, records: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                          encoding="utf-8")


def find_stamped_file(folder: Path, base_name: str, extension: str) -> Path | None:
    """The existing `<base_name>_<stamp>.<extension>` in folder, or None."""
    existing = sorted(Path(folder).glob(f"{base_name}_????-??-??_??-??.{extension}"))
    return existing[0] if existing else None


def require_stamped_file(folder: Path, base_name: str, extension: str, producing_step: str) -> Path:
    path = find_stamped_file(folder, base_name, extension)
    if path is None:
        raise SystemExit(f"No {base_name}_<date-time>.{extension} in {folder}; run {producing_step} first.")
    return path


def stage_folder(run_folder: Path, stage_key: str) -> Path:
    folder = Path(run_folder) / STAGE_SUBFOLDER_NAMES[stage_key]
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def find_or_create_stamped_file(folder: Path, base_name: str, extension: str) -> Path:
    """The existing `<base_name>_<stamp>.<extension>` in folder (so a resumed stage keeps appending to
    it), else a new path with the current stamp."""
    existing = sorted(Path(folder).glob(f"{base_name}_????-??-??_??-??.{extension}"))
    if existing:
        return existing[0]
    return Path(folder) / f"{base_name}_{timestamp_now()}.{extension}"


def resolve_run_folder(run_folder_argument: str | None, run_name: str | None, may_create: bool) -> Path:
    """An existing run folder (--run-folder), or a new one (--run-name) when may_create."""
    if run_folder_argument:
        folder = Path(run_folder_argument)
        if not folder.is_dir():
            raise SystemExit(f"--run-folder {folder} does not exist")
        return folder
    if not may_create:
        raise SystemExit("This stage needs --run-folder (the folder made by the gold labelling stage).")
    run_management = load_project_shared_module("run_folder_management")
    folder, _ = run_management.choose_run_folder(FULL_RUNS_FOLDER, run_name or DEFAULT_RUN_NAME, None)
    return folder


def record_stage_configuration(run_folder: Path, stage_name: str, prompt_file_paths: list[Path], extra: dict):
    """Refuse to resume when a prompt file changed; then write this stage's configuration file."""
    run_management = load_project_shared_module("run_folder_management")
    configuration = {"stage": stage_name, "extra": extra,
                     "prompt_file_sha256": {Path(p).name: run_management.sha256_of_file(p)
                                            for p in prompt_file_paths}}
    existing = sorted(Path(run_folder).glob(f"run_configuration_{stage_name}_*.json"))
    if existing:
        run_management.check_resume_is_compatible(run_folder, configuration)
        return existing[0]
    return run_management.write_run_configuration(run_folder, f"{stage_name}_{timestamp_now()}", configuration)


@contextlib.contextmanager
def without_implicit_dotenv_loading():
    """Import the pipeline's modules with `dotenv.load_dotenv` disabled. shared/api.py calls
    load_dotenv() at import, which searches parent folders and would pick up this project's .env
    implicitly; the key is instead read explicitly by `activate_project_anthropic_api_key`."""
    with mock.patch("dotenv.load_dotenv", lambda *args, **kwargs: False):
        yield


def activate_project_api_key(key_name: str) -> Path:
    """Refuse to run if `key_name` is already in the OS environment; otherwise read it from this
    project's .env only (explicit path, `dotenv_values`: nothing else is loaded) and set it for this process.
    Prints the source file, never the key."""
    if os.environ.get(key_name):
        raise SystemExit(f"{key_name} is already set in the OS environment; refusing to run so a key "
                         f"other than the project's cannot be used. Unset it and rerun.")
    from dotenv import dotenv_values
    key = (dotenv_values(ENV_FILE_PATH).get(key_name) or "").strip()
    if not key:
        raise SystemExit(f"{key_name} is missing or empty in {ENV_FILE_PATH}.")
    os.environ[key_name] = key
    print(f"Using {key_name} from {ENV_FILE_PATH}", flush=True)
    return ENV_FILE_PATH


def activate_project_anthropic_api_key() -> Path:
    return activate_project_api_key(ANTHROPIC_KEY_NAME)


def activate_project_openai_api_key() -> Path:
    return activate_project_api_key(OPENAI_KEY_NAME)
