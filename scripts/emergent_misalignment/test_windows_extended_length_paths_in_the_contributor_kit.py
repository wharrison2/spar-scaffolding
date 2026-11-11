"""Offline tests for the Windows extended-length path handling of the contributor kit (no model call, no Windows needed).

The text transforms are driven with ntpath so they run on any operating system; the real filesystem behaviour on Windows is
covered by smoke_test_windows_long_path_handling_without_any_model_call.py, which the kit's GitHub Actions workflow runs.
"""

import hashlib
import json
import ntpath
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "contributor_kit"))
import add_batch_to_emergent_misalignment_prompt_corpus as add_batch  # noqa: E402
import build_emergent_misalignment_contributor_kit as kit_builder  # noqa: E402
import run_my_shares  # noqa: E402
import windows_extended_length_paths as extended  # noqa: E402

PREFIX = "\\\\?\\"


# ---------- the string transform ----------

@pytest.mark.parametrize("plain_path, expected", [
    ("D:\\kit\\share", PREFIX + "D:\\kit\\share"),
    ("d:/kit/share/", PREFIX + "d:\\kit\\share"),
    ("D:\\kit\\.\\a\\..\\share", PREFIX + "D:\\kit\\share"),
    ("\\\\server\\share\\kit\\x", PREFIX + "UNC\\server\\share\\kit\\x"),
    ("//server/share/kit", PREFIX + "UNC\\server\\share\\kit"),
    (PREFIX + "D:\\already\\prefixed", PREFIX + "D:\\already\\prefixed"),
    (PREFIX + "UNC\\server\\share\\x", PREFIX + "UNC\\server\\share\\x"),
])
def test_the_prefix_is_added_for_drive_and_network_paths_and_never_twice(plain_path, expected):
    assert extended.add_extended_length_prefix_to_windows_path_text(plain_path) == expected


@pytest.mark.parametrize("relative_path", ["kit\\share", "share", "D:share", "\\no_drive\\share", ""])
def test_a_relative_path_is_rejected(relative_path):
    with pytest.raises(ValueError):
        extended.add_extended_length_prefix_to_windows_path_text(relative_path)


def test_the_prefix_is_removed_again_for_drive_and_network_paths():
    assert extended.remove_extended_length_prefix_from_windows_path_text(PREFIX + "D:\\kit") == "D:\\kit"
    assert extended.remove_extended_length_prefix_from_windows_path_text(PREFIX + "UNC\\server\\share\\x") == "\\\\server\\share\\x"
    assert extended.remove_extended_length_prefix_from_windows_path_text("D:\\kit") == "D:\\kit"
    text = f"$ python {PREFIX}D:\\kit\\a and {PREFIX}UNC\\server\\share\\b"
    assert extended.remove_extended_length_prefixes_from_text(text) == "$ python D:\\kit\\a and \\\\server\\share\\b"


def test_adding_then_removing_gives_back_the_normalised_path():
    for plain_path in ("D:\\a\\b\\c.txt", "\\\\server\\share\\a"):
        added = extended.add_extended_length_prefix_to_windows_path_text(plain_path)
        assert extended.remove_extended_length_prefix_from_windows_path_text(added) == plain_path


@pytest.mark.skipif(os.name == "nt", reason="the no-op is for macOS and Linux")
def test_everything_is_a_no_op_off_windows(tmp_path):
    for path in (tmp_path / "a" / "b", "relative/folder", str(tmp_path)):
        assert extended.extended_length_path(path) == Path(path)
        assert extended.path_without_extended_length_prefix(path) == Path(path)
    assert str(extended.extended_length_path("relative/folder")) == "relative/folder"  # not even made absolute


# ---------- comparing a prefixed and an unprefixed form of the same folder ----------

def test_relative_path_text_ignores_the_prefix_and_letter_case_on_windows_paths():
    folder = "D:\\Kit"
    inside = "share\\batches\\b1"
    for path in ("D:\\Kit\\" + inside, PREFIX + "D:\\Kit\\" + inside, "d:\\kit\\" + inside):
        for folder_form in (folder, PREFIX + folder, "d:\\kit"):
            assert extended.relative_path_text_under_folder(path, folder_form, ntpath).lower() == inside
    assert extended.relative_path_text_under_folder(PREFIX + "D:\\Kit", "D:\\Kit", ntpath) == "."
    assert extended.relative_path_text_under_folder("D:\\Kitten\\x", "D:\\Kit", ntpath) is None
    assert extended.relative_path_text_under_folder("E:\\Kit\\x", "D:\\Kit", ntpath) is None
    assert extended.relative_path_text_under_folder(PREFIX + "UNC\\server\\share\\kit\\x", "\\\\server\\share\\kit", ntpath) == "x"


def test_seed_record_folders_compare_equal_whatever_the_prefix_or_separator_style():
    recorded = "my_share_corpora\\contributor_corpus_share_002"
    for current in (recorded, "my_share_corpora/contributor_corpus_share_002", PREFIX + recorded,
                    "my_share_corpora/contributor_corpus_share_002/"):
        assert run_my_shares.same_relative_folder(recorded, current)
        assert run_my_shares.same_relative_folder(current, recorded)
    assert not run_my_shares.same_relative_folder(recorded, "my_share_corpora\\contributor_corpus_share_003")
    assert not run_my_shares.same_relative_folder(recorded, "second_pass_share_corpora\\contributor_corpus_share_002")


def test_relative_to_kit_gives_the_same_text_for_a_folder_under_the_kit(tmp_path, monkeypatch):
    monkeypatch.setattr(run_my_shares, "KIT_ROOT", tmp_path)
    folder = tmp_path / "my_share_corpora" / "contributor_corpus_share_002"
    folder.mkdir(parents=True)
    expected = str(Path("my_share_corpora") / "contributor_corpus_share_002")
    assert run_my_shares.relative_to_kit(folder) == expected
    assert run_my_shares.relative_to_kit(extended.extended_length_path(folder)) == expected
    assert run_my_shares.relative_to_kit(tmp_path) == "."
    assert PREFIX not in run_my_shares.relative_to_kit(Path(PREFIX + "D:\\elsewhere\\x")) or os.name != "nt"


# ---------- a share already started in this kit is resumed, not refused, whatever form its seed record has ----------

@pytest.fixture
def started_share_environment(tmp_path, monkeypatch):
    share_folder = tmp_path / "contributor_shares"
    share_folder.mkdir()
    content = json.dumps({"spec_id": "S-001"}) + "\n"
    (share_folder / "fake_share_002.jsonl").write_text(content, encoding="utf-8")
    share = {"share_number": 2, "share_file": "fake_share_002.jsonl", "scenario_seed_base": 20282000, "spec_count": 1,
             "safe_deal_count": 1, "share_file_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    monkeypatch.setattr(run_my_shares, "KIT_ROOT", tmp_path)
    original_prepare_share_corpus_folder = run_my_shares.prepare_share_corpus_folder  # its default folder was bound at import
    monkeypatch.setattr(run_my_shares, "prepare_share_corpus_folder",
                        lambda share, corpus_folder, folder=share_folder: original_prepare_share_corpus_folder(share, corpus_folder, folder))
    monkeypatch.setattr(run_my_shares, "SCENARIO_SEEDS_RECORD_PATH", tmp_path / "scenario_seeds_record.json")
    calls = []

    def fake_batch_call(command, cwd=None):
        calls.append(command)
        corpus_folder = Path(command[command.index("--corpus-folder") + 1])
        if "--resume-batch-folder" not in command:
            (corpus_folder / "batches" / "batch_001_seed_fake_2026-11-08_10-00").mkdir(parents=True)
        return 0

    monkeypatch.setattr(run_my_shares.subprocess, "call", fake_batch_call)
    return {"share": share, "output_folder": tmp_path / "my_share_corpora", "calls": calls, "tmp_path": tmp_path}


@pytest.mark.parametrize("record_style", ["unchanged", "backslashes", "prefix_and_backslashes"])
def test_an_unfinished_share_is_resumed_whatever_form_the_recorded_folder_has(started_share_environment, record_style):
    environment = started_share_environment
    assert run_my_shares.run_share(environment["share"], environment["output_folder"]) == 0
    records = run_my_shares.load_scenario_seed_records()
    assert len(records) == 1 and PREFIX not in records[0]["corpus_folder"]
    if record_style != "unchanged":
        records[0]["corpus_folder"] = records[0]["corpus_folder"].replace("/", "\\")
        if record_style == "prefix_and_backslashes":
            records[0]["corpus_folder"] = PREFIX + records[0]["corpus_folder"]
        run_my_shares.SCENARIO_SEEDS_RECORD_PATH.write_text(json.dumps(records), encoding="utf-8")
    assert run_my_shares.run_share(environment["share"], environment["output_folder"]) == 0  # not refused
    assert "--resume-batch-folder" in environment["calls"][1]
    assert len(run_my_shares.load_scenario_seed_records()) == 1


def test_a_different_folder_is_still_refused(started_share_environment):
    environment = started_share_environment
    assert run_my_shares.run_share(environment["share"], environment["output_folder"]) == 0
    with pytest.raises(SystemExit, match="already started with scenario seed base"):
        run_my_shares.run_share(environment["share"], environment["tmp_path"] / "some_other_output_folder")


# ---------- the zip, the ledger and the config never hold the prefix ----------

def test_zip_entry_names_are_relative_to_the_share_with_slashes_and_no_prefix():
    corpus_folder = "D:\\kit\\my_share_corpora\\contributor_corpus_share_002"
    for file_text in (corpus_folder + "\\batches\\b1\\x.txt", PREFIX + corpus_folder + "\\batches\\b1\\x.txt"):
        for folder_text in (corpus_folder, PREFIX + corpus_folder):
            name = run_my_shares.zip_entry_name_for_share_file("", 2, file_text, folder_text, ntpath)
            assert name == "contributor_corpus_share_002/batches/b1/x.txt"
    name = run_my_shares.zip_entry_name_for_share_file("second_pass_", 2, PREFIX + corpus_folder + "\\f.json", corpus_folder, ntpath)
    assert name == "second_pass_contributor_corpus_share_002/f.json"
    with pytest.raises(ValueError):
        run_my_shares.zip_entry_name_for_share_file("", 2, "D:\\elsewhere\\f.json", corpus_folder, ntpath)


def test_a_zip_of_a_deeply_nested_share_has_clean_entry_names(tmp_path):
    corpus_folder = run_my_shares.corpus_folder_for_share(2, tmp_path / "corpora")
    deep_file = corpus_folder / "batches" / "batch_001_seed_1_x" / "raw_pipeline_outputs" / "dad" / "runs" / "r" / "system_prompts" / ("a" * 64 + ".txt")
    deep_file.parent.mkdir(parents=True)
    deep_file.write_text("prompt", encoding="utf-8")
    (corpus_folder / run_my_shares.LOCK_FILE_NAME).write_text("{}", encoding="utf-8")
    zip_path = run_my_shares.write_zip_of_share_corpora([2], tmp_path / "corpora", tmp_path / "zips", now=lambda: datetime(2026, 11, 9, 12, 0))
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["contributor_corpus_share_002/batches/batch_001_seed_1_x/raw_pipeline_outputs/dad/runs/r/"
                                      "system_prompts/" + "a" * 64 + ".txt"]


def test_the_ledger_batch_folder_has_no_prefix_and_uses_slashes(tmp_path):
    corpus_folder = tmp_path / "corpus"
    batch_folder = corpus_folder / "batches" / "batch_001_seed_1_2026-11-09_12-00"
    assert add_batch.ledger_batch_folder_text(batch_folder, corpus_folder) == "batches/batch_001_seed_1_2026-11-09_12-00"
    assert add_batch.ledger_batch_folder_text(extended.extended_length_path(batch_folder), corpus_folder) == \
        "batches/batch_001_seed_1_2026-11-09_12-00"


def test_the_batch_config_names_spec_set_and_plan_without_the_prefix(tmp_path):
    spec_set_path = tmp_path / "fixed_spec_set" / "share.jsonl"
    plan_path = tmp_path / "plan.json"
    config_path = add_batch.write_batch_config(tmp_path, 5, 1, "2026-11-09_12-00", spec_set_path=spec_set_path, spec_dealing_plan_path=plan_path)
    config_text = config_path.read_text(encoding="utf-8")
    assert str(spec_set_path) in config_text and str(plan_path) in config_text and "?" not in config_text


def test_the_terminal_output_file_header_has_no_prefix(tmp_path):
    terminal_output_path = tmp_path / "terminal_output.txt"
    command = [sys.executable, "-c", "print('x')"]
    assert add_batch.run_command(command, terminal_output_path) == 0
    assert terminal_output_path.read_text(encoding="utf-8").splitlines()[0].startswith("$ ")
    assert extended.remove_extended_length_prefixes_from_text("$ python " + PREFIX + "D:\\a") == "$ python D:\\a"


# ---------- the kit ----------

def test_the_new_files_are_listed_for_the_kit_and_exist():
    for name in ("windows_extended_length_paths.py", "smoke_test_windows_long_path_handling_without_any_model_call.py"):
        assert name in kit_builder.SCRIPT_FILE_NAMES and (kit_builder.SCRIPT_DIRECTORY / name).exists()
    workflow_template, workflow_kit_path = next(
        (template, kit_path) for template, kit_path in kit_builder.TEMPLATE_FILES_AND_KIT_NAMES.items() if kit_path.startswith(".github/workflows/"))
    text = (kit_builder.KIT_TEMPLATE_FOLDER / workflow_template).read_text(encoding="utf-8")
    assert "windows-latest" in text and "LongPathsEnabled" in text and "-Value 0" in text and "python-version: \"3.12\"" in text
    assert "pull_request" in text and "push" in text and "requirements.txt" in text
    assert "smoke_test_windows_long_path_handling_without_any_model_call.py" in text


def test_the_smoke_test_passes_here_with_paths_over_the_windows_limit():
    smoke_script = kit_builder.SCRIPT_DIRECTORY / "smoke_test_windows_long_path_handling_without_any_model_call.py"
    completed = subprocess.run([sys.executable, str(smoke_script)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    assert "PASS: every check passed." in completed.stdout and "FAIL" not in completed.stdout
