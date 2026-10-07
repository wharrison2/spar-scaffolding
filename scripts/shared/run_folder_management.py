"""Timestamped run folders, run configuration files with file hashes, and the resume safety check."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


TIMESTAMP_SUFFIX_PATTERN = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")


def name_with_timestamp_appended_once(run_name):
    """Appends the timestamp unless run_name already ends with one."""
    if TIMESTAMP_SUFFIX_PATTERN.search(run_name):
        return run_name
    return f"{run_name}_{timestamp_now()}"


def sha256_of_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_run_folder(runs_directory, run_name, resume_folder):
    """Returns (folder, is_resume). New folders are created inside runs_directory (any parent the
    caller chooses) and named <run_name>_<timestamp>, with the timestamp appended exactly once."""
    runs_directory = Path(runs_directory)
    runs_directory.mkdir(parents=True, exist_ok=True)
    if resume_folder:
        folder = Path(resume_folder)
        if not folder.is_dir():
            raise SystemExit(f"--resume-folder {folder} does not exist")
        return folder, True
    base_name = TIMESTAMP_SUFFIX_PATTERN.sub("", run_name)
    existing = sorted(runs_directory.glob(f"{base_name}_????-??-??_??-??"))
    if existing:
        raise SystemExit(f"A folder for run name '{run_name}' already exists: {existing[-1]}\n"
                         f"Pass --resume-folder {existing[-1]} to resume it, or choose a new --run-name.")
    folder = runs_directory / name_with_timestamp_appended_once(run_name)
    folder.mkdir(parents=True)
    return folder, False


def write_run_configuration(run_folder, timestamp, configuration):
    path = Path(run_folder) / f"run_configuration_{timestamp}.json"
    path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return path


def check_resume_is_compatible(run_folder, new_configuration, hashes_key="prompt_file_sha256"):
    """Raise RuntimeError if any file hash differs from the run's recorded configuration."""
    configuration_paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not configuration_paths:
        raise RuntimeError(f"No run_configuration file in {run_folder}; cannot verify prompts.")
    for path in configuration_paths:
        old_hashes = json.loads(path.read_text(encoding="utf-8"))[hashes_key]
        for file_name, new_hash in new_configuration[hashes_key].items():
            if file_name in old_hashes and old_hashes[file_name] != new_hash:
                raise RuntimeError(
                    f"Refusing to resume: prompt file {file_name} changed since the run started "
                    f"({path.name}).")
