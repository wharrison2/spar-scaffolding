"""Offline tests for the parts of check_sdf_pipeline_branch_before_pushing.py that need no pipeline run.

Run: python -m pytest scripts/sdf_document_state_redesign
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_sdf_pipeline_branch_before_pushing as branch_check  # noqa: E402


def test_arguments_default_to_no_expected_fixture_changes_and_replay_on():
    arguments = branch_check.parse_arguments(["--pipeline-checkout", "/x", "--base", "main"])
    assert arguments.expected_fixture_changes == [] and arguments.skip_replay is False


def test_arguments_take_several_globs_and_skip_replay():
    arguments = branch_check.parse_arguments(["--pipeline-checkout", "/x", "--base", "main", "--skip-replay",
                                              "--expected-fixture-changes", "a/*.json", "b/*"])
    assert arguments.expected_fixture_changes == ["a/*.json", "b/*"] and arguments.skip_replay


def test_fixture_changes_match_glob_on_full_path_or_path_under_fixtures():
    changed = ["tests/fixtures/sdf/a.json", "tests/fixtures/dad/b.json", "tests/fixtures/sdf/c.txt"]
    assert branch_check.fixture_changes_outside_allow_list(changed, ["sdf/*.json"]) == [
        "tests/fixtures/dad/b.json", "tests/fixtures/sdf/c.txt"]
    assert branch_check.fixture_changes_outside_allow_list(changed, ["tests/fixtures/*"]) == []
    assert branch_check.fixture_changes_outside_allow_list(changed, []) == changed


def test_baseline_cache_lookup_requires_matching_commit_and_completion_marker(tmp_path):
    commit_folder = tmp_path / "abc123"
    rebuilt = commit_folder / "sdf_real_run_request_replay_rebuilt_from_origin_main_x_2026-01-01_00-00-00"
    rebuilt.mkdir(parents=True)
    assert branch_check.find_cached_baseline_rebuild(tmp_path, "abc123") is None  # unfinished
    (commit_folder / branch_check.BASELINE_COMPLETE_MARKER_NAME).write_text("{}")
    assert branch_check.find_cached_baseline_rebuild(tmp_path, "abc123") == rebuilt
    assert branch_check.find_cached_baseline_rebuild(tmp_path, "def456") is None


def test_environment_for_pipeline_subprocesses_has_no_api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    environment = branch_check.environment_without_api_keys()
    assert "ANTHROPIC_API_KEY" not in environment and "OPENAI_API_KEY" not in environment
