#!/usr/bin/env python3
"""Runs every check an SDF pipeline branch must pass before it is pushed, and prints a short summary.

Checks, in order: clean tree, full test suite, fixture diff against the base branch, compileall, the two
--help commands, and (unless --skip-replay) the real-run request replay compared with a baseline rebuilt from
the pipeline repository's origin/main. The baseline is cached, one folder per origin/main commit, under
data/sdf_pipeline_origin_main_baseline_rebuilds/. No model API is called: every pipeline subprocess runs with
the API keys removed from its environment. All output is saved in data/sdf_pipeline_branch_check_<branch>_<time>/.
Exits 1 if any check fails.

Example:
  python check_sdf_pipeline_branch_before_pushing.py --pipeline-checkout <worktree> --base <git ref>
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_FOLDER))

import replay_sdf_real_run_requests_against_pipeline_checkout as replay  # noqa: E402

PROJECT_ROOT = SCRIPT_FOLDER.parents[1]
MAIN_CHECKOUT_OF_THIS_PROJECT = Path("/Users/williamharrison/Desktop/emergent_alignment")
DATA_FOLDER = MAIN_CHECKOUT_OF_THIS_PROJECT / "data"
BASELINE_CACHE_ROOT = DATA_FOLDER / "sdf_pipeline_origin_main_baseline_rebuilds"
REPLAY_SCRIPT = SCRIPT_FOLDER / "replay_sdf_real_run_requests_against_pipeline_checkout.py"
BASELINE_COMPLETE_MARKER_NAME = "baseline_rebuild_complete.json"
REBUILT_FOLDER_PATTERN = "sdf_real_run_request_replay_rebuilt_from_*"
KEYS_REMOVED_FROM_ENVIRONMENT = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
COMPILE_FOLDERS = ("shared", "sdf_pipeline", "dad_pipeline", "pref_pipeline", "evals", "viewer")
PYTEST_COMMAND_TAIL = ["-m", "pytest", "-p", "no:cacheprovider", "-o",
                       "addopts=--strict-markers --disable-socket", "-q"]


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-checkout", required=True, help="a worktree of the pipeline repository")
    parser.add_argument("--base", required=True, help="git ref of the branch it is stacked on; fixtures are diffed against it")
    parser.add_argument("--expected-fixture-changes", nargs="*", default=[], metavar="GLOB",
                        help="paths under tests/fixtures/ the branch may change (default: none)")
    parser.add_argument("--skip-replay", action="store_true", help="leave out the real-run request replay")
    return parser.parse_args(argument_list)


def fixture_changes_outside_allow_list(changed_paths: list, allowed_globs: list) -> list:
    """Changed paths matching no glob; a glob may name the full path or the path under tests/fixtures/."""
    prefix = "tests/fixtures/"
    unexpected = []
    for path in changed_paths:
        relative = path[len(prefix):] if path.startswith(prefix) else path
        if not any(fnmatch.fnmatch(path, glob) or fnmatch.fnmatch(relative, glob) for glob in allowed_globs):
            unexpected.append(path)
    return unexpected


def find_cached_baseline_rebuild(cache_root: Path, commit: str):
    """The completed baseline rebuild folder cached for this origin/main commit, or None."""
    commit_folder = cache_root / commit
    if not (commit_folder / BASELINE_COMPLETE_MARKER_NAME).is_file():
        return None
    rebuilt_folders = sorted(commit_folder.glob(REBUILT_FOLDER_PATTERN))
    return rebuilt_folders[-1] if rebuilt_folders else None


def environment_without_api_keys() -> dict:
    return {name: value for name, value in os.environ.items() if name not in KEYS_REMOVED_FROM_ENVIRONMENT}


def run_and_save(command: list, working_folder: Path, output_folder: Path, output_name: str):
    """Runs a command without API keys, saves its full output, returns (exit code, output text)."""
    output_folder.mkdir(parents=True, exist_ok=True)
    finished = subprocess.run(command, cwd=working_folder, env=environment_without_api_keys(),
                              capture_output=True, text=True)
    output_text = f"$ {' '.join(map(str, command))}\n(exit code {finished.returncode})\n{finished.stdout}{finished.stderr}"
    (output_folder / f"{output_name}.txt").write_text(output_text, encoding="utf-8")
    return finished.returncode, finished.stdout + finished.stderr


def git_output(repository: Path, *git_arguments) -> str:
    return subprocess.run(["git", "-C", str(repository), *git_arguments], capture_output=True, text=True,
                          check=True).stdout.strip()


def last_line_containing(text: str, words: tuple) -> str:
    matching = [line for line in text.splitlines() if any(word in line for word in words)]
    return matching[-1].strip() if matching else "no summary line found"


# ---------------------------------------------------------------- the checks (each returns passed, detail)

def check_clean_tree(checkout: Path, output_folder: Path):
    code, text = run_and_save(["git", "status", "--porcelain"], checkout, output_folder, "git_status_porcelain")
    dirty = [line for line in text.splitlines() if line.strip() and not line.endswith("WORKTREE_STATUS.md")]
    return not dirty, "tree is clean" if not dirty else f"{len(dirty)} uncommitted or untracked paths, first: {dirty[0]}"


def check_test_suite(checkout: Path, pipeline_python: Path, output_folder: Path):
    code, text = run_and_save([str(pipeline_python), *PYTEST_COMMAND_TAIL], checkout, output_folder, "pytest_output")
    return code == 0, last_line_containing(text, ("passed", "failed", "error"))


def check_fixture_diff(checkout: Path, base: str, allowed_globs: list, output_folder: Path):
    code, text = run_and_save(["git", "diff", "--name-only", base, "--", "tests/fixtures"], checkout, output_folder,
                              "git_diff_of_fixtures_against_base")
    if code != 0:
        return False, f"git diff failed: {text.strip()[:150]}"
    changed = [line for line in text.splitlines() if line.strip()]
    unexpected = fixture_changes_outside_allow_list(changed, allowed_globs)
    if unexpected:
        return False, f"{len(unexpected)} fixture files changed outside the allow-list, first: {unexpected[0]}"
    return True, f"{len(changed)} fixture files changed, all expected" if changed else "no fixture changes"


def check_compileall(checkout: Path, pipeline_python: Path, output_folder: Path):
    code, text = run_and_save([str(pipeline_python), "-m", "compileall", "-q", *COMPILE_FOLDERS], checkout,
                              output_folder, "compileall_output")
    return code == 0, "all folders compile" if code == 0 else text.strip().splitlines()[-1][:150]


def check_help_commands(checkout: Path, pipeline_python: Path, output_folder: Path):
    """Run --help of each command-line script the checkout has (the print script exists only on some branches)."""
    scripts = [script for script in ["sdf_pipeline/run.py",
                                     "scripts/print_full_sdf_request_a_stage_would_send_for_one_document.py"]
               if (checkout / script).is_file()]
    failing = []
    for script in scripts:
        code, _ = run_and_save([str(pipeline_python), script, "--help"], checkout, output_folder,
                               f"help_of_{Path(script).stem}")
        if code != 0:
            failing.append(script)
    if failing:
        return False, f"--help failed for {', '.join(failing)}"
    return True, f"--help exits 0 for {', '.join(Path(script).name for script in scripts)}"


def rebuild_requests_with_checkout(checkout: Path, pipeline_python: Path, output_root: Path, label: str,
                                   output_folder: Path, output_name: str):
    """Runs the replay harness's rebuild for a checkout; returns the rebuilt folder or None."""
    output_root.mkdir(parents=True, exist_ok=True)
    code, text = run_and_save(
        [str(pipeline_python), str(REPLAY_SCRIPT), "rebuild", "--pipeline-checkout", str(checkout),
         "--output-root", str(output_root), "--label", label, "--report-differences-without-failing"],
        PROJECT_ROOT, output_folder, output_name)
    rebuilt_folders = sorted(output_root.glob(REBUILT_FOLDER_PATTERN))
    return (rebuilt_folders[-1] if code == 0 and rebuilt_folders else None), text


def build_baseline_from_origin_main(pipeline_root: Path, commit: str, pipeline_python: Path, output_folder: Path):
    """Rebuilds the baseline from a temporary detached worktree of origin/main; returns (rebuilt folder, note)."""
    worktree_path = pipeline_root / ".claude/worktrees" / f"sdf_branch_check_baseline_origin_main_{commit[:12]}"
    git_output(pipeline_root, "worktree", "add", "--detach", str(worktree_path), commit)
    try:
        commit_folder = BASELINE_CACHE_ROOT / commit
        rebuilt_folder, _ = rebuild_requests_with_checkout(worktree_path, pipeline_python, commit_folder,
                                                           f"origin_main_{commit[:12]}", output_folder,
                                                           "baseline_rebuild_output")
        if rebuilt_folder:
            (commit_folder / BASELINE_COMPLETE_MARKER_NAME).write_text(json.dumps({"origin_main_commit": commit}))
        return rebuilt_folder
    finally:
        subprocess.run(["git", "-C", str(pipeline_root), "worktree", "remove", "--force", str(worktree_path)],
                       capture_output=True)


def check_request_replay(checkout: Path, pipeline_root: Path, pipeline_python: Path, output_folder: Path):
    commit = git_output(pipeline_root, "rev-parse", "origin/main")
    baseline_folder = find_cached_baseline_rebuild(BASELINE_CACHE_ROOT, commit)
    baseline_note = f"baseline reused from origin/main {commit[:10]}"
    if baseline_folder is None:
        baseline_folder = build_baseline_from_origin_main(pipeline_root, commit, pipeline_python, output_folder)
        baseline_note = f"baseline built from origin/main {commit[:10]}"
    if baseline_folder is None:
        return False, "the baseline rebuild from origin/main failed (see baseline_rebuild_output.txt)"
    branch_folder, _ = rebuild_requests_with_checkout(checkout, pipeline_python, output_folder / "branch_rebuild",
                                                      checkout.name, output_folder, "branch_rebuild_output")
    if branch_folder is None:
        return False, "the rebuild from the checkout failed (see branch_rebuild_output.txt)"
    code, text = run_and_save(
        [str(pipeline_python), str(REPLAY_SCRIPT), "compare", "--first-rebuilt-folder", str(baseline_folder),
         "--second-rebuilt-folder", str(branch_folder), "--output-root", str(output_folder / "comparison")],
        PROJECT_ROOT, output_folder, "comparison_of_branch_with_baseline_output")
    difference_match = re.search(r"^(\d+) differences", text, re.MULTILINE)
    differences = difference_match.group(1) if difference_match else "unknown number of"
    rebuilt_requests = replay.load_rebuilt_folder(branch_folder)["requests"]
    requests_rebuilt = len(rebuilt_requests)
    runs = len({request["run_label"] for request in rebuilt_requests})
    return code == 0, f"{requests_rebuilt} requests rebuilt from {runs} runs, {differences} differences; {baseline_note}"


def main() -> int:
    arguments = parse_arguments()
    checkout = Path(arguments.pipeline_checkout).resolve()
    pipeline_root = Path(git_output(checkout, "rev-parse", "--path-format=absolute", "--git-common-dir")).parent
    pipeline_python = pipeline_root / ".venv/bin/python"
    branch_name = git_output(checkout, "rev-parse", "--abbrev-ref", "HEAD").replace("/", "_")
    run_folder = DATA_FOLDER / f"sdf_pipeline_branch_check_{branch_name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    run_folder.mkdir(parents=True)

    checks = [
        ("1 clean tree", lambda folder: check_clean_tree(checkout, folder)),
        ("2 test suite", lambda folder: check_test_suite(checkout, pipeline_python, folder)),
        ("3 fixture diff", lambda folder: check_fixture_diff(checkout, arguments.base,
                                                             arguments.expected_fixture_changes, folder)),
        ("4 compileall", lambda folder: check_compileall(checkout, pipeline_python, folder)),
        ("5 help commands", lambda folder: check_help_commands(checkout, pipeline_python, folder)),
    ]
    if not arguments.skip_replay:
        checks.append(("6 request replay", lambda folder: check_request_replay(checkout, pipeline_root,
                                                                                pipeline_python, folder)))
    results = []
    for name, run_check in checks:
        started = time.time()
        passed, detail = run_check(run_folder / name.replace(" ", "_"))
        results.append((name, passed, detail, time.time() - started))
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail} ({results[-1][3]:.0f} s)", flush=True)

    summary_lines = [f"SDF pipeline branch check: {checkout} (base {arguments.base})", ""]
    summary_lines += [f"{'PASS' if passed else 'FAIL'}  {name:<18} {detail}  [{seconds:.0f} s]"
                      for name, passed, detail, seconds in results]
    if arguments.skip_replay:
        summary_lines.append("SKIP  6 request replay   (--skip-replay)")
    summary_lines += ["", f"run folder: {run_folder}"]
    summary_text = "\n".join(summary_lines)
    (run_folder / "summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
    return 0 if all(passed for _, passed, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
