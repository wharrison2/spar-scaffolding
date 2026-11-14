"""Integration test of the launcher's real tar transfer commands. ssh is replaced by a local shell (`bash -c`) and the
pod's /workspace by a temporary folder, so the same tar pipelines used against a pod run for real. No RunPod calls."""

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as launcher  # noqa: E402

LOGS_FOLDER_NAME = "pod_testpod_logs_2026-10-22_12-00"


@pytest.fixture
def local_run_folder(tmp_path):
    folder = tmp_path / "local_run_folder"
    (folder / "scoring_inputs").mkdir(parents=True)
    (folder / "scoring_inputs" / "documents.jsonl").write_text('{"document_id": "d1"}\n')
    (folder / "scoring_inputs" / "conditions.jsonl").write_text('{"condition_id": "c1"}\n')
    (folder / "run_configuration_2026-10-22_12-00.json").write_text('{"model_name": "test_model"}')
    return folder


@pytest.fixture
def fake_pod_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "fake_pod_workspace" / "emergent_alignment_scoring_copy"
    monkeypatch.setattr(launcher, "REMOTE_WORKING_DIRECTORY", str(workspace))
    monkeypatch.setattr(launcher, "REMOTE_RUN_FOLDER", f"{workspace}/run_folder")
    return workspace


@pytest.fixture
def transfer(local_run_folder, fake_pod_workspace):
    return launcher.SshTarTransfer("unused_key", local_run_folder, ssh_timeout_seconds=60,
                                   remote_shell_prefix=["bash", "-c"])


def write_files(root, files):
    for relative_path, text in files.items():
        path = Path(root) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_upload_sends_run_configuration_scoring_inputs_and_scripts(transfer, fake_pod_workspace):
    transfer.upload_project_files()
    remote_run_folder = fake_pod_workspace / "run_folder"
    assert (remote_run_folder / "run_configuration_2026-10-22_12-00.json").read_text() == '{"model_name": "test_model"}'
    assert (remote_run_folder / "scoring_inputs" / "documents.jsonl").exists()
    assert (fake_pod_workspace / launcher.SCORER_SCRIPT_RELATIVE_PATH).exists()
    assert (fake_pod_workspace / "scripts" / "shared").is_dir()
    assert not list(fake_pod_workspace.rglob("._*")) and not list(fake_pod_workspace.rglob("__pycache__"))
    # The scorer's own check, which crashed the first real pod run when the configuration was not uploaded.
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
    try:
        import importlib
        scorer_module = importlib.import_module("log_probability_persona_check.score_document_log_probabilities_on_pod")
        assert scorer_module.load_first_run_configuration(remote_run_folder) == {"model_name": "test_model"}
    except ImportError:
        pytest.skip("scorer dependencies not installed locally")
    finally:
        sys.path.remove(str(REPOSITORY_ROOT / "scripts"))


def test_periodic_pull_treats_missing_scoring_outputs_as_nothing_to_pull(transfer, local_run_folder, fake_pod_workspace):
    transfer.upload_project_files()
    assert not (fake_pod_workspace / "run_folder" / "scoring_outputs").exists()
    transfer.pull_scoring_outputs(False)
    transfer.pull_scoring_outputs(True)
    assert not (local_run_folder / "scoring_outputs").exists()


def test_final_pull_with_everything_missing_but_the_scorer_logs_succeeds(transfer, local_run_folder, fake_pod_workspace):
    """The first real run's situation: the scorer crashed before creating scoring_outputs/."""
    transfer.upload_project_files()
    write_files(fake_pod_workspace / "run_folder", {"scorer_standard_output.txt": "Traceback\n", "scorer_exit_code.txt": "1\n"})
    transfer.pull_final_copy_and_verify(LOGS_FOLDER_NAME, 60)
    assert (local_run_folder / LOGS_FOLDER_NAME / "scorer_exit_code.txt").read_text() == "1\n"
    assert (local_run_folder / LOGS_FOLDER_NAME / "scorer_standard_output.txt").read_text() == "Traceback\n"
    assert not (local_run_folder / "scoring_outputs").exists()


def populate_finished_pod(fake_pod_workspace):
    write_files(fake_pod_workspace, {
        "run_folder/scoring_outputs/per_token_log_probabilities/d1.npz": "binary-ish content" * 50,
        "run_folder/scoring_outputs/completion_records.jsonl": '{"document_id": "d1"}\n',
        "run_folder/scoring_outputs/scoring_progress_log_2026-10-22_12-05.txt": "done\n",
        "run_folder/scorer_standard_output.txt": "all good\n",
        "run_folder/scorer_exit_code.txt": "0\n",
        "run_folder/.last_pull_marker": "",
        "watchdog_log.txt": "",
        "pod_setup_pip_install_log.txt": "installed\n",
        "pod_setup_environment_versions.txt": "torch 2.8\n",
        "pod_secret_environment.sh": "export HF_TOKEN=secret\n",
    })


def test_full_final_pull_copies_everything_but_scoring_inputs_and_secrets_and_verifies(
        transfer, local_run_folder, fake_pod_workspace):
    transfer.upload_project_files()
    populate_finished_pod(fake_pod_workspace)
    files_verified = transfer.pull_final_copy_and_verify(LOGS_FOLDER_NAME, 60)
    assert files_verified == 10 - 2 + 1  # minus the secret file and marker, plus the uploaded run configuration
    outputs = local_run_folder / "scoring_outputs"
    assert (outputs / "per_token_log_probabilities" / "d1.npz").stat().st_size == len("binary-ish content") * 50
    assert (outputs / "completion_records.jsonl").exists()
    logs = local_run_folder / LOGS_FOLDER_NAME
    for name in ("scorer_standard_output.txt", "scorer_exit_code.txt", "watchdog_log.txt",
                 "pod_setup_pip_install_log.txt", "pod_setup_environment_versions.txt"):
        assert (logs / name).exists(), name
    assert not (logs / "pod_secret_environment.sh").exists() and not list(local_run_folder.rglob("pod_secret*"))
    assert not list(local_run_folder.rglob(".last_pull_marker*"))
    assert not (logs / "scoring_inputs").exists()
    assert not list(local_run_folder.glob("tmp*"))  # staging folder removed


def test_final_pull_verification_reports_files_that_did_not_arrive(transfer, local_run_folder, fake_pod_workspace,
                                                                    monkeypatch):
    transfer.upload_project_files()
    populate_finished_pod(fake_pod_workspace)

    def pull_that_delivers_an_empty_archive(command, output_file_path, timeout_seconds):
        Path(output_file_path).write_bytes(b"")
        return 255, "connection reset"

    monkeypatch.setattr(transfer, "_run_remote_command_into_file", pull_that_delivers_an_empty_archive)
    with pytest.raises(launcher.FinalCopyIncompleteError) as excinfo:
        transfer.pull_final_copy_and_verify(LOGS_FOLDER_NAME, 60)
    assert any("scorer_exit_code.txt" in name and "did not arrive" in name for name in excinfo.value.problem_files)


def test_final_pull_verification_reports_size_mismatch(transfer, local_run_folder, fake_pod_workspace, monkeypatch):
    transfer.upload_project_files()
    populate_finished_pod(fake_pod_workspace)
    real_run_remote_command = transfer.run_remote_command

    def listing_that_claims_larger_files(command, input_text=None, timeout_seconds=None):
        exit_code, output = real_run_remote_command(command, input_text, timeout_seconds)
        if "wc -c" in command:
            output = output.replace("4\tpod_setup_pip_install_log.txt", "999\tpod_setup_pip_install_log.txt")
            output = output.replace("10\tpod_setup_pip_install_log.txt", "999\tpod_setup_pip_install_log.txt")
        return exit_code, output

    monkeypatch.setattr(transfer, "run_remote_command", listing_that_claims_larger_files)
    with pytest.raises(launcher.FinalCopyIncompleteError) as excinfo:
        transfer.pull_final_copy_and_verify(LOGS_FOLDER_NAME, 60)
    assert any("pod_setup_pip_install_log.txt (pod 999 bytes" in name for name in excinfo.value.problem_files)
