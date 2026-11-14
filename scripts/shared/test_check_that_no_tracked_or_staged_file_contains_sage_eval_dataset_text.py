"""Offline tests for the SAGE-Eval pre-push guard. All fixtures are synthetic; no real SAGE-Eval text appears here."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_that_no_tracked_or_staged_file_contains_sage_eval_dataset_text as guard

SYNTHETIC_PROMPT = "Zorblax the synthetic fixture prompt asks how to polish the quuxwidget with the frobnicator every morning."
OTHER_SYNTHETIC_PROMPT = "A second invented fixture prompt about grinding the blorptonic crystals before the picnic on Tuesday."
SHORT_PROMPT = "Too short to count."


def git(repository: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    git(folder, "init", "-b", "main")
    git(folder, "config", "user.email", "test@example.invalid")
    git(folder, "config", "user.name", "Test")
    (folder / "readme.txt").write_text("nothing relevant\n", encoding="utf-8")
    git(folder, "add", "readme.txt")
    return folder


def write_dataset(tmp_path: Path, prompts: list[str]) -> Path:
    dataset = {"Synthetic safety fact": {"category": "Child", "prompts": {"TYPE_A": {"original": prompts, "safe_version": [],
                                                                                    "original_augmentation": {"typos": []}}}}}
    path = tmp_path / "full_dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


def run_guard(repository: Path, dataset_path: Path, tmp_path: Path) -> int:
    return guard.main(["--dataset-path", str(dataset_path), "--repository", str(repository), "--kit-folder", str(tmp_path / "no_kit_here")])


def test_hit_in_tracked_file_is_detected_with_line_number(repository, tmp_path, capsys):
    (repository / "notes.md").write_text("first line\nsecond line\n" + SYNTHETIC_PROMPT.upper() + "\n", encoding="utf-8")
    git(repository, "add", "notes.md")
    git(repository, "commit", "-m", "x")
    assert run_guard(repository, write_dataset(tmp_path, [SYNTHETIC_PROMPT]), tmp_path) == 1
    assert "notes.md line 3" in capsys.readouterr().out


def test_hit_in_staged_file_is_detected_even_if_working_copy_was_cleaned(repository, tmp_path, capsys):
    (repository / "staged.md").write_text(SYNTHETIC_PROMPT.replace(" ", "\n  ") + "\n", encoding="utf-8")
    git(repository, "add", "staged.md")
    (repository / "staged.md").write_text("cleaned afterwards\n", encoding="utf-8")
    assert run_guard(repository, write_dataset(tmp_path, [SYNTHETIC_PROMPT]), tmp_path) == 1
    assert "staged.md (staged)" in capsys.readouterr().out


def test_no_hit_exits_zero(repository, tmp_path, capsys):
    assert run_guard(repository, write_dataset(tmp_path, [SYNTHETIC_PROMPT]), tmp_path) == 0
    assert "OK" in capsys.readouterr().out


def test_dataset_absent_says_so_and_exits_zero(repository, tmp_path, capsys):
    assert run_guard(repository, tmp_path / "missing.json", tmp_path) == 0
    assert "not found" in capsys.readouterr().out


def test_short_prompts_are_skipped(repository, tmp_path):
    (repository / "short.md").write_text(SHORT_PROMPT + "\n", encoding="utf-8")
    git(repository, "add", "short.md")
    assert len(SHORT_PROMPT) < guard.MINIMUM_PROMPT_CHARACTERS
    assert run_guard(repository, write_dataset(tmp_path, [SHORT_PROMPT, OTHER_SYNTHETIC_PROMPT]), tmp_path) == 0


def test_output_never_prints_full_matched_text(repository, tmp_path, capsys):
    (repository / "leak.md").write_text(SYNTHETIC_PROMPT + "\n", encoding="utf-8")
    git(repository, "add", "leak.md")
    assert run_guard(repository, write_dataset(tmp_path, [SYNTHETIC_PROMPT]), tmp_path) == 1
    output = capsys.readouterr().out
    assert SYNTHETIC_PROMPT[:guard.PREVIEW_CHARACTERS] in output
    assert SYNTHETIC_PROMPT[:guard.PREVIEW_CHARACTERS + 10] not in output
    assert "Synthetic safety fact" not in output


def test_canary_is_only_a_warning_and_allowed_in_plans(repository, tmp_path, capsys):
    (repository / "plans").mkdir()
    (repository / "plans" / "note.md").write_text(guard.SAGE_EVAL_CANARY_STRING + "\n", encoding="utf-8")
    (repository / "script.py").write_text(f"# {guard.SAGE_EVAL_CANARY_STRING}\n", encoding="utf-8")
    git(repository, "add", "plans/note.md", "script.py")
    assert run_guard(repository, write_dataset(tmp_path, [SYNTHETIC_PROMPT]), tmp_path) == 0
    output = capsys.readouterr().out
    assert "WARNING" in output and "script.py" in output and "plans/note.md" not in output
