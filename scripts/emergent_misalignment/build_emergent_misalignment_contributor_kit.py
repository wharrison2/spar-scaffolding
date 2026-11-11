"""Assemble the standalone contributor kit folder (its own git repository) for the emergent misalignment corpus.

Makes no model call. The kit holds exactly what one contributor needs to deal their shares: the pipeline copy's step 1 code
and prompts, the batch script, the two judges with their prompt templates and helper modules, the share spec files and the
manifest, the entry point run_my_shares.py, a README.md and a CLAUDE.md written for a contributor's Claude Code session,
requirements.txt, a .gitignore, and a GitHub Actions workflow (.github/workflows/) that runs an offline long Windows path
smoke test with the LongPathsEnabled setting off. The kit repository has a remote, https://github.com/wharrison2/emergent-misalignment-prompt-corpus-contributor-kit
(public; contributors clone and pull it); the builder never pushes. It keeps this repository's folder layout (scripts/emergent_misalignment/ and
strictly_implicit_difficult_advice_pipeline_copy/ under the kit root), so the batch script runs unchanged. It holds no
generated prompts, held-out set, accepted prompts, batch outputs, .env, personal paths, or anything from data/ except the
share files. The kit repository will be public.

The build replaces the kit folder's contents (never its .git folder), and refuses to touch a non-empty folder that is not a
previous kit. After copying it checks the kit for forbidden content (check_kit_folder_is_clean) and writes
kit_file_checksums_<stamp>.json at the kit root, so a contributor's dealer and prompts can be compared with the originals.
With --make-commit the kit repository gets a local commit (git init on first use).
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent.parent
KIT_TEMPLATE_FOLDER = SCRIPT_DIRECTORY / "contributor_kit"
PIPELINE_FOLDER_NAME = "strictly_implicit_difficult_advice_pipeline_copy"
PIPELINE_FOLDER = REPOSITORY_ROOT / PIPELINE_FOLDER_NAME
DEFAULT_KIT_FOLDER = Path.home() / "Desktop" / "emergent_misalignment_prompt_corpus_contributor_kit"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
MANIFEST_PATTERN = "contributor_share_manifest_*.json"
SHARE_FOLDER_NAME = "contributor_shares"

# Files of scripts/emergent_misalignment/ the kit needs (the batch script, selection, spec outcomes, the dealer loader, the
# two judges and the helper modules they import, the pipeline configuration template).
SCRIPT_FILE_NAMES = [
    "add_batch_to_emergent_misalignment_prompt_corpus.py",
    "spec_outcomes_and_status_table_for_prompt_corpus.py",
    "select_implicit_prompts_from_difficult_advice_step_one_run.py",
    "load_pipeline_scenario_dealer.py",
    "reserved_spec_ids_for_contributor_shares.py",
    "filter_prompts_where_user_never_mentions_animal_welfare_with_haiku_through_claude_code_cli.py",
    "claude_code_cli_structured_reply_without_paid_api_fallback.py",
    "judge_whether_a_complete_answer_is_possible_without_mentioning_animal_welfare_through_claude_code_cli.py",
    "judge_whether_prompt_carries_an_evident_and_decisive_animal_welfare_stake_through_claude_code_cli.py",
    "prompt_corpus_batch_pipeline_config_template.yaml",
    "windows_extended_length_paths.py",
    "smoke_test_windows_long_path_handling_without_any_model_call.py",
]
JUDGE_PROMPT_TEMPLATE_NAMES = [
    "judge_whether_a_complete_answer_is_possible_without_mentioning_animal_welfare_prompt_template.txt",
    "judge_whether_prompt_carries_an_evident_and_decisive_animal_welfare_stake_prompt_template.txt",
]
# Pipeline step 1 needs these generation prompts (the batch script snapshots the same ones); no other prompts.
PIPELINE_PROMPT_FILE_NAMES = ["step1a_scenario.txt", "step1b_dilemmas.txt", "step1c_gate.txt", "step1d_refine.txt",
                              "variables.txt"]
PIPELINE_CODE_FOLDER_NAMES = ["dad_pipeline", "shared"]
TEMPLATE_FILES_AND_KIT_NAMES = {
    "run_my_shares.py": "run_my_shares.py",
    "kit_README_for_contributors.md": "README.md",
    "kit_CLAUDE_for_contributors_claude_code_session.md": "CLAUDE.md",
    "requirements_for_contributor_kit.txt": "requirements.txt",
    "kit_gitignore_for_contributor_kit.txt": ".gitignore",
    "kit_github_actions_workflow_check_long_windows_paths_with_long_paths_disabled.yml":
        ".github/workflows/check_long_windows_paths_with_long_paths_disabled.yml",
}
EMPTY_CONSTITUTION_NOTE = ("This folder is empty on purpose. Pipeline step 1 snapshots it into each run but does not read it.\n")
FORBIDDEN_NAME_PATTERNS = [r"^\.env", r"accepted_prompts", r"held_out", r"batch_\d{3}_seed", r"corpus_batch_ledger",
                           r"^outputs$", r"^data$", r"^viewer", r"^website$", r"^tests$", r"^\.claude$"]
FORBIDDEN_TEXT_PATTERNS = [r"/Users/", r"/home/\w+", r"C:\\Users", r"williamharrison", r"sk-ant-"]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".toml"}


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_tree_without_caches(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))


def empty_kit_folder_keeping_git(kit_folder: Path) -> None:
    kit_folder.mkdir(parents=True, exist_ok=True)
    children = [child for child in kit_folder.iterdir() if child.name != ".git"]
    if children and not (kit_folder / "run_my_shares.py").exists():
        raise SystemExit(f"{kit_folder} is not empty and is not a previous kit; refusing to replace its contents")
    for child in children:
        shutil.rmtree(child) if child.is_dir() else child.unlink()


def find_shares_folder_manifest(shares_folder: Path) -> Path:
    manifests = sorted(Path(shares_folder).glob(MANIFEST_PATTERN))
    if not manifests:
        raise SystemExit(f"no {MANIFEST_PATTERN} in {shares_folder}")
    return manifests[-1]


def copy_share_files_and_manifest(shares_folder: Path, kit_folder: Path) -> dict:
    manifest_path = find_shares_folder_manifest(shares_folder)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = kit_folder / SHARE_FOLDER_NAME
    destination.mkdir()
    for share in manifest["shares"]:
        source = Path(shares_folder) / share["share_file"]
        if sha256_of_file(source) != share["share_file_sha256"]:
            raise SystemExit(f"{source} does not match the checksum in the manifest")
        shutil.copy2(source, destination / share["share_file"])
    shutil.copy2(manifest_path, destination / manifest_path.name)
    return manifest


def build_kit_contents(shares_folder: Path, kit_folder: Path) -> dict:
    """Copy everything into the (emptied) kit folder and return the share manifest."""
    for template_name, kit_name in TEMPLATE_FILES_AND_KIT_NAMES.items():
        (kit_folder / kit_name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(KIT_TEMPLATE_FOLDER / template_name, kit_folder / kit_name)
    scripts_destination = kit_folder / "scripts" / "emergent_misalignment"
    (scripts_destination / "prompts").mkdir(parents=True)
    for name in SCRIPT_FILE_NAMES:
        shutil.copy2(SCRIPT_DIRECTORY / name, scripts_destination / name)
    for name in JUDGE_PROMPT_TEMPLATE_NAMES:
        shutil.copy2(SCRIPT_DIRECTORY / "prompts" / name, scripts_destination / "prompts" / name)
    pipeline_destination = kit_folder / PIPELINE_FOLDER_NAME
    (pipeline_destination / "prompts" / "dad").mkdir(parents=True)
    for folder_name in PIPELINE_CODE_FOLDER_NAMES:
        copy_tree_without_caches(PIPELINE_FOLDER / folder_name, pipeline_destination / folder_name)
    for name in PIPELINE_PROMPT_FILE_NAMES:
        shutil.copy2(PIPELINE_FOLDER / "prompts" / "dad" / name, pipeline_destination / "prompts" / "dad" / name)
    shutil.copy2(PIPELINE_FOLDER / "LICENSE", pipeline_destination / "LICENSE")
    (pipeline_destination / "constitution").mkdir()
    (pipeline_destination / "constitution" / "README.md").write_text(EMPTY_CONSTITUTION_NOTE, encoding="utf-8")
    return copy_share_files_and_manifest(shares_folder, kit_folder)


def check_kit_folder_is_clean(kit_folder: Path, manifest: dict) -> list[str]:
    """Problems found in a built kit (empty list when clean): forbidden names, personal paths or keys in text files, and any
    file under contributor_shares/ that is not a manifest share file or the manifest."""
    problems = []
    allowed_share_files = {share["share_file"] for share in manifest["shares"]}
    for path in sorted(Path(kit_folder).rglob("*")):
        relative = path.relative_to(kit_folder)
        if ".git" in relative.parts:
            continue
        for part in relative.parts:
            if any(re.search(pattern, part) for pattern in FORBIDDEN_NAME_PATTERNS):
                problems.append(f"forbidden name: {relative}")
        if path.is_file():
            if relative.parts[0] == SHARE_FOLDER_NAME and path.name not in allowed_share_files \
                    and not re.match(r"contributor_share_manifest_.*\.json$", path.name):
                problems.append(f"unexpected file among the shares: {relative}")
            if path.suffix in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="replace")
                for pattern in FORBIDDEN_TEXT_PATTERNS:
                    if re.search(pattern, text):
                        problems.append(f"{relative} matches {pattern!r}")
    return problems


def write_checksums_file(kit_folder: Path, stamp: str) -> Path:
    checksums = {str(path.relative_to(kit_folder)): sha256_of_file(path)
                 for path in sorted(Path(kit_folder).rglob("*")) if path.is_file() and ".git" not in path.relative_to(kit_folder).parts}
    path = kit_folder / f"kit_file_checksums_{stamp}.json"
    path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    return path


def commit_kit(kit_folder: Path, message: str) -> None:
    if not (kit_folder / ".git").exists():
        subprocess.run(["git", "init", "-b", "main"], cwd=kit_folder, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=kit_folder, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=kit_folder, check=True, capture_output=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shares-folder", type=Path, required=True,
                        help="emergent_misalignment_contributor_shares_<stamp> folder with the share files and manifest")
    parser.add_argument("--kit-folder", type=Path, default=DEFAULT_KIT_FOLDER)
    parser.add_argument("--make-commit", action="store_true", help="git init (first time) and commit in the kit folder")
    arguments = parser.parse_args(argv)
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    empty_kit_folder_keeping_git(arguments.kit_folder)
    manifest = build_kit_contents(arguments.shares_folder, arguments.kit_folder)
    write_checksums_file(arguments.kit_folder, stamp)
    problems = check_kit_folder_is_clean(arguments.kit_folder, manifest)
    if problems:
        print("KIT NOT CLEAN:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"Built the kit in {arguments.kit_folder}: {manifest['number_of_shares']} shares, clean.")
    if arguments.make_commit:
        commit_kit(arguments.kit_folder, f"Add the emergent misalignment prompt corpus contributor kit ({stamp})")
        print("Committed locally (the builder never pushes; push to the kit repository's remote yourself).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
