"""The steps of the overnight treatment arm chain, in order, with their commands and dependencies. Imported by the on-pod orchestrator
(run_overnight_treatment_arm_chain_on_pod.py) and by the tests. Standard library only.

Layout on the pod (the launcher uploads the scripts and the chain's inputs; W is the working directory on the network volume):
  W/run_folder/overnight_inputs/run_folders/treatment_arm_generation            prebuilt run folder for the answer generation
  W/run_folder/overnight_inputs/run_folders/evaluation/<kind>                   prebuilt evaluation run folders, adapter = placeholder
  W/run_folder/overnight_inputs/held_out_set_file/<file>.jsonl                  the held-out set, for the training example check
  W/run_folder/run_configuration_overnight_chain_<timestamp>.json               this chain's settings
  W/run_folder/overnight_outputs/...                                            everything written (pulled back to the laptop)
Container disk, not the volume: /root/overnight_environments (two virtual environments), /root/huggingface_cache (model weights, about
63 GB, and the HarmBench classifier), /root/overnight_environment_markers (environment install markers) and /root/overnight_work (the work
tree of generation and evaluation run folders, logs and summary: steps write their many small files there, and it is synced to the volume
every 3 minutes, after every step attempt and before every completion marker). The volume holds the synced outputs, both adapters, the
latest checkpoint (written to it directly), completion markers and instructions. On a relaunch the work tree is restored from the volume.

Order and why:
  generation phases 1 to 4 (the model judges nothing: the LAPTOP judges the pulled system-prompted answers with Claude Haiku 5.5 through the paid
  API and uploads verdict, regeneration-request and judging-complete files into instructions_from_launcher/haiku_judging/; while the pod waits for
  them it generates the base-model evaluation answers that need no adapter, and a launcher heartbeat older than 20 minutes ends the chain) ->
  TRAINING GATE (at least 5,000 prompts with a kept answer, else no training and no evaluation) -> training examples and fine-tuning run
  folders -> smoke fine-tuning on 200 examples and its check -> full fine-tuning and its check -> attach the adapter to the prebuilt evaluation
  run folders -> generation for held-out, Betley et al. and MMLU-Pro (base, base with the system prompt, adapter; the base entries are mostly done
  already) -> MMLU-Pro scoring -> "are you sure?", HarmBench (its own classifier) and SAGE-Eval generation. Every other judging (evaluation
  answers, completeness, user lean) is done afterwards on the laptop with Haiku.
A failed step skips only the steps that depend on it. The evaluation steps depend on the attach step, which depends on nothing in the
training chain: when training fails they still run for the base entries (--only-model-short-names).
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.shared.resumable_step_chain_with_failure_policy_and_pod_self_deletion import ChainStep  # noqa: E402

OVERNIGHT_INPUTS_FOLDER_NAME = "overnight_inputs"
OVERNIGHT_OUTPUTS_FOLDER_NAME = "overnight_outputs"
SCRIPTS = "scripts/emergent_misalignment"
CHAIN_SCRIPTS = f"{SCRIPTS}/overnight_chain"
VLLM_REQUIREMENTS = f"{SCRIPTS}/requirements_for_baseline_answer_generation_pod.txt"
FINE_TUNING_REQUIREMENTS = f"{SCRIPTS}/requirements_for_lora_fine_tuning_pod.txt"
ENVIRONMENTS_ROOT = "/root/overnight_environments"
VLLM_ENVIRONMENT = f"{ENVIRONMENTS_ROOT}/vllm_environment"
FINE_TUNING_ENVIRONMENT = f"{ENVIRONMENTS_ROOT}/fine_tuning_environment"
CONTAINER_MARKERS_FOLDER = "/root/overnight_environment_markers"
HUGGING_FACE_CACHE = "/root/huggingface_cache"
EVALUATION_KINDS = ["held_out", "betley", "mmlu_pro", "sycophancy_harmbench_sage_eval"]
BASE_MODEL_SHORT_NAME = "gemma_4_31b_it"
BASE_WITH_SYSTEM_PROMPT_MODEL_SHORT_NAME = "gemma_4_31b_it_with_system_prompt_do_not_consider_animal_welfare"
ADAPTER_NAME = "treatment_arm"
GENERATION_SCRIPT = f"{SCRIPTS}/treatment_arm_training_data/generate_judge_and_regenerate_treatment_arm_answers_with_gemma_on_pod.py"
GATE_CLOSED_EXIT_CODE = 10  # the gate scripts' exit code for "closed" (a decision, not an error)
DEFAULT_TRAINING_GATE_MINIMUM_PROMPTS = 5000  # written into the chain's run configuration by the prebuild script
HAIKU_JUDGING_INPUT_FOLDER_NAME = "haiku_judging"  # inside instructions_from_launcher: the laptop uploads here
LAUNCHER_LOST_DECISION_FILE_NAME = "launcher_connection_lost_decision.json"
IDLE_TIME_EVALUATION_KINDS = ["held_out", "betley", "mmlu_pro"]  # run folders whose base-model answers are generated while the pod waits
TRAINING_AND_EVALUATION_STEP_NAMES = [
    "build_training_examples_and_fine_tuning_run_folders", "fine_tuning_smoke_run", "check_smoke_fine_tuning_run", "fine_tuning_full_run",
    "check_full_fine_tuning_run", "attach_trained_adapter_to_evaluation_run_folders", "evaluation_generation_held_out", "evaluation_generation_betley",
    "evaluation_generation_mmlu_pro", "score_mmlu_pro_answers", "sycophancy_harmbench_sage_eval_generation_and_classification"]


def python_executable_for(environment_folder):
    return f"{environment_folder}/bin/python"


def sha256_of_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes() if Path(path).exists() else b"missing")
    return digest.hexdigest()


SYNC_EXCLUDE_PATTERNS = ["*/lora_adapters/*", "*chain_finished_flag.json", "*launcher_final_pull_complete.txt", "instructions_from_launcher/*"]
RESTORE_EXCLUDE_PATTERNS = SYNC_EXCLUDE_PATTERNS + ["fine_tuning/*", "step_completion_markers/*"]


class ChainPaths:
    """Every path the steps use. Two homes for outputs (reason: a network volume is slow for the many small appends of answer and judge files):
      container work tree  C = container_work_root/overnight_outputs   generation and evaluation run folders, step logs, summary and progress
                                                                        files, the adapter attachment result; synced to the volume
      network volume       V = W/run_folder/overnight_outputs          what the launcher pulls: the synced copy of C, plus, written directly
                                                                        and never through C: fine_tuning/ (run folders, adapters, the latest
                                                                        checkpoint), step_completion_markers/, instructions_from_launcher/,
                                                                        the finished flag and the launcher's final pull acknowledgement
    Without container_work_root (tests), C and V are the same folder and nothing is synced."""

    def __init__(self, working_directory, run_folder, container_work_root=None):
        self.working_directory, self.run_folder = Path(working_directory), Path(run_folder)
        self.inputs = self.run_folder / OVERNIGHT_INPUTS_FOLDER_NAME
        self.volume_outputs = self.run_folder / OVERNIGHT_OUTPUTS_FOLDER_NAME
        self.outputs = Path(container_work_root) / OVERNIGHT_OUTPUTS_FOLDER_NAME if container_work_root else self.volume_outputs
        self.work_tree_is_separate_from_the_volume = bool(container_work_root)
        self.run_folders = self.outputs / "run_folders"
        self.generation = self.run_folders / "treatment_arm_generation"
        self.evaluation = {kind: self.run_folders / "evaluation" / kind for kind in EVALUATION_KINDS}
        self.fine_tuning_outputs = self.volume_outputs / "fine_tuning"
        self.smoke_run = self.fine_tuning_outputs / "fine_tuning_run_folders" / "fine_tuning_smoke_run"
        self.full_run = self.fine_tuning_outputs / "fine_tuning_run_folders" / "fine_tuning_full_run"
        self.attachment_result_folder = self.outputs / "evaluation_adapter_attachment"
        self.gate_decisions = self.outputs / "gate_decisions"
        self.markers = self.volume_outputs / "step_completion_markers"
        self.logs = self.outputs / "step_logs"
        self.instructions = self.volume_outputs / "instructions_from_launcher"
        self.haiku_judging_inputs = self.instructions / HAIKU_JUDGING_INPUT_FOLDER_NAME  # written directly on the volume by the launcher (never synced or restored)
        self.summary = self.outputs / "chain_summary_and_progress"
        self.heartbeat_file = self.run_folder / "launcher_heartbeat_epoch_seconds.txt"  # not under outputs: never pulled back
        self.final_pull_acknowledgement_file = self.volume_outputs / "chain_summary_and_progress" / "launcher_final_pull_complete.txt"
        self.finished_flag_file = self.volume_outputs / "chain_summary_and_progress" / "chain_finished_flag.json"


def make_sync_function(paths):
    """Container work tree -> volume (changed files only, each verified by size). None when there is no separate work tree."""
    if not paths.work_tree_is_separate_from_the_volume:
        return None
    from scripts.shared import sync_work_tree_to_durable_volume_with_size_verification as sync

    def sync_outputs():
        result = sync.sync_tree(paths.outputs, paths.volume_outputs, SYNC_EXCLUDE_PATTERNS)
        result["summary"] = f"{result['files_copied']} files ({result['bytes_copied'] / 1e6:.1f} MB) copied, {result['files_unchanged']} unchanged"
        return result
    return sync_outputs


def restore_work_tree_from_volume(paths):
    """On a relaunch (new pod, same volume): rebuild the container work tree from the volume before the chain starts."""
    if not paths.work_tree_is_separate_from_the_volume:
        return None
    from scripts.shared import sync_work_tree_to_durable_volume_with_size_verification as sync
    paths.outputs.mkdir(parents=True, exist_ok=True)
    return sync.restore_tree(paths.volume_outputs, paths.outputs, RESTORE_EXCLUDE_PATTERNS)


def latest_attachment_result(paths):
    results = sorted(paths.attachment_result_folder.glob("attach_trained_adapter_result_*.json")) if paths.attachment_result_folder.exists() else []
    return json.loads(results[-1].read_text(encoding="utf-8")) if results else None


def adapter_is_available(paths):
    result = latest_attachment_result(paths)
    return bool(result and result.get("adapter_available"))


def build_chain_steps(paths, chain_configuration):
    """The ordered ChainStep list. `chain_configuration` keys: sage_eval_dataset_supplied, training_example_seed, smoke_run_examples,
    extra_command_line_arguments_by_step (rehearsal and tests), python_by_environment_name (rehearsal)."""
    working = paths.working_directory
    extra = chain_configuration.get("extra_command_line_arguments_by_step", {})
    vllm_python = python_executable_for(VLLM_ENVIRONMENT)
    fine_tuning_python = python_executable_for(FINE_TUNING_ENVIRONMENT)
    sage_supplied = bool(chain_configuration.get("sage_eval_dataset_supplied"))
    minimum_prompts = chain_configuration.get("training_gate_minimum_prompts_with_a_kept_answer", DEFAULT_TRAINING_GATE_MINIMUM_PROMPTS)

    def with_extra(name, command):
        return [*command, *extra.get(name, [])]

    def base_only_arguments():
        """Evaluated just before an evaluation step starts: base entries only when training gave no adapter."""
        return [] if adapter_is_available(paths) else ["--only-model-short-names", BASE_MODEL_SHORT_NAME]

    def adapter_signature():
        return f"adapter_available={adapter_is_available(paths)}"

    def evaluation_generation(kind, extra_name):
        def command():
            return with_extra(extra_name, ["python", f"{SCRIPTS}/generate_baseline_answers_for_candidate_models_on_pod.py",
                                            "--run-folder", paths.evaluation[kind], "--keep-model-caches", *base_only_arguments()])
        return command

    def sycophancy_command():
        arguments = ["--run-folder", paths.evaluation["sycophancy_harmbench_sage_eval"], "--keep-model-caches"]
        if not adapter_is_available(paths):
            arguments += ["--only-model-short-names", BASE_MODEL_SHORT_NAME, BASE_WITH_SYSTEM_PROMPT_MODEL_SHORT_NAME]
        arguments.append("--skip-judging")  # the pod never judges: SAGE-Eval's judge is Haiku, run on the laptop afterwards (HarmBench's classifier still runs here)
        return with_extra("sycophancy_harmbench_sage_eval_generation_and_classification",
                          ["python", f"{SCRIPTS}/generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod.py",
                           *arguments])

    def generation_phases_command():
        return with_extra("generation_phases_1_to_4", [
            "python", GENERATION_SCRIPT, "--run-folder", paths.generation, "--only-phases", "1", "2", "3", "4",
            "--haiku-judging-folder", paths.haiku_judging_inputs, "--launcher-heartbeat-file", paths.heartbeat_file,
            "--launcher-heartbeat-stale-seconds", str(chain_configuration.get("launcher_heartbeat_stale_seconds", 20 * 60)),
            "--launcher-lost-decision-file", paths.gate_decisions / LAUNCHER_LOST_DECISION_FILE_NAME,
            "--idle-time-evaluation-run-folders", *[f"{kind}={paths.evaluation[kind]}" for kind in IDLE_TIME_EVALUATION_KINDS]])

    def environment_install(name, environment_folder, requirements, import_checks):
        return ChainStep(
            name=name, description=f"create {environment_folder} (system site packages visible) and install {requirements}",
            command=with_extra(name, ["bash", f"{CHAIN_SCRIPTS}/install_python_environment_with_requirements_and_import_check_on_pod.sh",
                                      environment_folder, requirements, *import_checks]),
            marker_location="container", signature=lambda: sha256_of_files([working / requirements, working / CHAIN_SCRIPTS /
                                                                            "install_python_environment_with_requirements_and_import_check_on_pod.sh"]),
            output_paths=[environment_folder])

    steps = [
        ChainStep("prepare_run_folders_in_container_work_tree", with_extra("prepare_run_folders_in_container_work_tree", [
            "python", f"{CHAIN_SCRIPTS}/copy_prebuilt_inputs_into_outputs_without_overwriting_existing_files.py",
            "--source", paths.inputs / "run_folders", "--destination", paths.run_folders]),
                  "copy the uploaded prebuilt run folders into the container work tree (synced to the volume), never overwriting restored files",
                  output_paths=[str(paths.run_folders)]),
        environment_install("install_vllm_environment", VLLM_ENVIRONMENT, VLLM_REQUIREMENTS, ["vllm", "transformers"]),
        environment_install("install_fine_tuning_environment", FINE_TUNING_ENVIRONMENT, FINE_TUNING_REQUIREMENTS, ["peft", "transformers", "safetensors"]),
        ChainStep("generation_phases_1_to_4", generation_phases_command,
                  "Gemma generates (all prompts with the system prompt, the sampled prompts without); the laptop judges with Haiku; Gemma regenerates "
                  "what the laptop asks for (up to 4 attempts per prompt) and, while it waits, generates the base-model evaluation answers; ends the "
                  "chain (resumable) when the launcher heartbeat is 20 minutes old",
                  depends_on=["prepare_run_folders_in_container_work_tree", "install_vllm_environment"], environment_name="vllm",
                  gate_closed_exit_code=GATE_CLOSED_EXIT_CODE, skip_when_gate_closed="all_remaining",
                  gate_result_file=paths.gate_decisions / LAUNCHER_LOST_DECISION_FILE_NAME,
                  output_paths=[str(paths.generation / "generation_outputs"), str(paths.run_folders / "evaluation")]),
        ChainStep("training_gate", with_extra("training_gate", [
            "python", f"{CHAIN_SCRIPTS}/evaluate_training_gate_after_phase_4_prompts_with_a_kept_answer.py", "--generation-run-folder", paths.generation,
            "--minimum-prompts", str(minimum_prompts), "--result-folder", paths.gate_decisions]),
                  f"GATE: train only when at least {minimum_prompts:,} prompts have a kept answer; below that no training and no evaluation",
                  depends_on=["generation_phases_1_to_4"], gate_closed_exit_code=GATE_CLOSED_EXIT_CODE,
                  skip_when_gate_closed=TRAINING_AND_EVALUATION_STEP_NAMES, gate_result_file=paths.gate_decisions / "training_gate_decision.json",
                  output_paths=[str(paths.gate_decisions)]),
        ChainStep("build_training_examples_and_fine_tuning_run_folders", with_extra("build_training_examples_and_fine_tuning_run_folders", [
            "python", f"{CHAIN_SCRIPTS}/build_training_examples_and_fine_tuning_run_folders_from_latest_kept_answers_on_pod.py",
            "--generation-run-folder", paths.generation, "--held-out-set-file", next_held_out_set_file(paths),
            "--output-folder", paths.fine_tuning_outputs, "--seed", str(chain_configuration.get("training_example_seed", 20261111)),
            "--smoke-run-examples", str(chain_configuration.get("smoke_run_examples", 200)), "--python-executable", fine_tuning_python]),
                  "every kept answer, at most one per prompt (fixed seed), then the smoke and full fine-tuning run folders",
                  depends_on=["training_gate", "install_fine_tuning_environment"], environment_name="fine_tuning",
                  output_paths=[str(paths.fine_tuning_outputs)]),
        ChainStep("fine_tuning_smoke_run", with_extra("fine_tuning_smoke_run", [
            "python", f"{SCRIPTS}/fine_tune_candidate_model_with_lora_on_own_answers_on_pod.py", "--run-folder", paths.smoke_run]),
                  "LoRA fine-tuning on the first 200 examples; its adapter only gates the full run and is never evaluated", depends_on=["build_training_examples_and_fine_tuning_run_folders"],
                  environment_name="fine_tuning", output_paths=[str(paths.smoke_run / "training_outputs")]),
        ChainStep("check_smoke_fine_tuning_run", with_extra("check_smoke_fine_tuning_run", [
            "python", f"{CHAIN_SCRIPTS}/check_fine_tuning_run_finished_with_finite_loss_and_passing_adapter_key_check.py",
            "--run-folder", paths.smoke_run]), "finite loss and a saved adapter that passes the key check",
                  depends_on=["fine_tuning_smoke_run"], output_paths=[str(paths.smoke_run / "training_outputs")]),
        ChainStep("fine_tuning_full_run", with_extra("fine_tuning_full_run", [
            "python", f"{SCRIPTS}/fine_tune_candidate_model_with_lora_on_own_answers_on_pod.py", "--run-folder", paths.full_run]),
                  "LoRA fine-tuning on all training examples", depends_on=["check_smoke_fine_tuning_run"], environment_name="fine_tuning",
                  output_paths=[str(paths.full_run / "training_outputs")]),
        ChainStep("check_full_fine_tuning_run", with_extra("check_full_fine_tuning_run", [
            "python", f"{CHAIN_SCRIPTS}/check_fine_tuning_run_finished_with_finite_loss_and_passing_adapter_key_check.py",
            "--run-folder", paths.full_run]), "finite loss and a saved adapter that passes the key check",
                  depends_on=["fine_tuning_full_run"], output_paths=[str(paths.full_run / "training_outputs")]),
        ChainStep("attach_trained_adapter_to_evaluation_run_folders", with_extra("attach_trained_adapter_to_evaluation_run_folders", [
            "python", f"{CHAIN_SCRIPTS}/attach_trained_adapter_to_prebuilt_evaluation_run_folders_or_record_that_none_is_available.py",
            "--evaluation-run-folders", *[paths.evaluation[kind] for kind in EVALUATION_KINDS], "--fine-tuning-run-folder", paths.full_run,
            "--adapter-name", ADAPTER_NAME, "--result-folder", paths.attachment_result_folder]),
                  "copy the trained adapter over the placeholder in each evaluation run folder, or record that none is available (the "
                  "evaluation steps then run the base entries only)",
                  depends_on=["prepare_run_folders_in_container_work_tree"], run_after=["check_full_fine_tuning_run"],
                  signature=adapter_signature_for_attach(paths), output_paths=[str(paths.attachment_result_folder)]),
        ChainStep("evaluation_generation_held_out", evaluation_generation("held_out", "evaluation_generation_held_out"),
                  "held-out welfare raising: base, base with the system prompt, treatment adapter",
                  depends_on=["attach_trained_adapter_to_evaluation_run_folders", "install_vllm_environment"], environment_name="vllm",
                  signature=adapter_signature, output_paths=[str(paths.evaluation["held_out"] / "generation_outputs")]),
        ChainStep("evaluation_generation_betley", evaluation_generation("betley", "evaluation_generation_betley"),
                  "Betley et al. free-form questions, 100 samples per question",
                  depends_on=["attach_trained_adapter_to_evaluation_run_folders", "install_vllm_environment"], environment_name="vllm",
                  signature=adapter_signature, output_paths=[str(paths.evaluation["betley"] / "generation_outputs")]),
        ChainStep("evaluation_generation_mmlu_pro", evaluation_generation("mmlu_pro", "evaluation_generation_mmlu_pro"),
                  "MMLU-Pro capability check answers", depends_on=["attach_trained_adapter_to_evaluation_run_folders", "install_vllm_environment"],
                  environment_name="vllm", signature=adapter_signature, output_paths=[str(paths.evaluation["mmlu_pro"] / "generation_outputs")]),
        ChainStep("score_mmlu_pro_answers", with_extra("score_mmlu_pro_answers", [
            "python", f"{SCRIPTS}/score_mmlu_pro_capability_check_base_versus_lora_adapters.py", "--run-folder", paths.evaluation["mmlu_pro"]]),
                  "score the MMLU-Pro answers (no model)", depends_on=["evaluation_generation_mmlu_pro"], environment_name="fine_tuning",
                  signature=adapter_signature, output_paths=[str(paths.evaluation["mmlu_pro"])]),
        ChainStep("sycophancy_harmbench_sage_eval_generation_and_classification", sycophancy_command,
                  "'are you sure?' sycophancy, HarmBench (official classifier) and, when supplied, SAGE-Eval answer generation (judged afterwards on the laptop)",
                  depends_on=["attach_trained_adapter_to_evaluation_run_folders", "install_vllm_environment"], environment_name="vllm",
                  signature=adapter_signature, output_paths=[str(paths.evaluation["sycophancy_harmbench_sage_eval"])],
                  notes=([] if sage_supplied else ["SAGE-Eval skipped: no full_dataset.json was supplied at launch (the run folder has no SAGE-Eval task)"])),
    ]
    for step in steps:  # rehearsal and tests only: replace a step's whole command (the real chain's configuration has none)
        if step.name in chain_configuration.get("command_overrides_by_step", {}):
            step.command = chain_configuration["command_overrides_by_step"][step.name]
    return steps


def next_held_out_set_file(paths):
    """Resolved when the steps are built, which happens after the uploads: the single file in overnight_inputs/held_out_set_file."""
    files = sorted((paths.inputs / "held_out_set_file").glob("*.jsonl"))
    return files[-1] if files else paths.inputs / "held_out_set_file" / "missing_held_out_set_file.jsonl"


def adapter_signature_for_attach(paths):
    """The attach step's marker is valid only for the same training outcome, so a relaunch after a later successful training reattaches."""
    def signature():
        checks = sorted((paths.full_run / "training_outputs").glob("fine_tuning_run_check_*.json")) if paths.full_run.exists() else []
        passed = bool(checks) and json.loads(checks[-1].read_text(encoding="utf-8")).get("passed") is True
        return f"full_run_check_passed={passed}"
    return signature


def list_trained_adapter_and_checkpoint_paths_on_the_volume(paths, volume_mount_path="/workspace"):
    """Every adapter folder (vLLM form, Hugging Face form, end-of-epoch copies) and every resume checkpoint the two fine-tuning runs left, with
    byte sizes and whether the real path lies on the network volume. The runs write straight into the volume (the chain's outputs folder is on
    /workspace), so each adapter and checkpoint is on durable storage from the moment the training script writes it; this list is the check
    that nothing sits elsewhere, and goes into the chain summary."""
    import os
    entries = []
    for run_name, run_folder in (("smoke_run", paths.smoke_run), ("full_run", paths.full_run)):
        outputs = run_folder / "training_outputs"
        candidates = [outputs / "final_adapter", outputs / "final_adapter_for_vllm"]
        candidates += sorted((outputs / "adapter_at_end_of_epoch").glob("epoch_*")) if (outputs / "adapter_at_end_of_epoch").exists() else []
        candidates += sorted((outputs / "checkpoints").glob("checkpoint_step_*")) if (outputs / "checkpoints").exists() else []
        for folder in candidates:
            if not folder.exists():
                continue
            size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
            real = os.path.realpath(folder)
            entries.append({"run": run_name, "path": str(folder), "bytes": size,
                            "on_the_volume": real == volume_mount_path or real.startswith(volume_mount_path.rstrip("/") + "/"),
                            "kind": "resume checkpoint" if "checkpoint_step_" in folder.name else "adapter"})
    return entries


CHECKPOINTS_FOLDER_NAME = "checkpoints"
CHECKPOINT_COMPLETE_MARKER_FILE_NAME = "checkpoint_complete.json"
CHECKPOINT_STATE_FILE_NAME = "training_state.pt"
CHECKPOINT_ADAPTER_FOLDER_NAME = "adapter"
LOW_FREE_SPACE_WARNING_BYTES = 3 * 10 ** 9


def checkpoint_is_complete_and_verified(folder):
    """The training script writes the completion marker last; also require the optimizer state file and the adapter files to be non-empty."""
    adapter = folder / CHECKPOINT_ADAPTER_FOLDER_NAME
    required = [folder / CHECKPOINT_COMPLETE_MARKER_FILE_NAME, folder / CHECKPOINT_STATE_FILE_NAME, adapter / "adapter_model.safetensors",
                adapter / "adapter_config.json"]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def keep_only_the_latest_verified_checkpoint(checkpoints_folder):
    """Deletes every checkpoint_step_* folder older than the newest one that is complete and verified. Folders without the completion marker
    (being written) and anything newer are never touched. Returns the names deleted."""
    import shutil
    if not Path(checkpoints_folder).is_dir():
        return []
    folders = sorted((path for path in Path(checkpoints_folder).glob("checkpoint_step_*") if path.is_dir()),
                     key=lambda path: int(path.name.rsplit("_", 1)[1]) if path.name.rsplit("_", 1)[1].isdigit() else -1)
    verified = [folder for folder in folders if checkpoint_is_complete_and_verified(folder)]
    if len(verified) < 1:
        return []
    newest = verified[-1]
    deleted = []
    for folder in folders:
        if folder is newest or folders.index(folder) > folders.index(newest):
            continue
        shutil.rmtree(folder, ignore_errors=True)
        deleted.append(folder.name)
    return deleted


def free_space_warning(path, threshold_bytes=LOW_FREE_SPACE_WARNING_BYTES):
    import shutil
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        return None
    if free < threshold_bytes:
        return (f"WARNING: only {free / 1e9:.2f} GB free on the volume at {path} (threshold {threshold_bytes / 1e9:.0f} GB): a large write "
                "(checkpoint, adapter, answers) may fail")
    return None


def make_housekeeping_function(paths, volume_mount_path="/workspace"):
    """One tick: keep only the latest verified checkpoint of each training run, then warn when the volume is nearly full."""
    def housekeeping():
        messages = []
        for run_name, run_folder in (("smoke run", paths.smoke_run), ("full run", paths.full_run)):
            deleted = keep_only_the_latest_verified_checkpoint(run_folder / "training_outputs" / CHECKPOINTS_FOLDER_NAME)
            if deleted:
                messages.append(f"{run_name}: deleted older checkpoints {', '.join(deleted)} from the volume (a newer one is complete and verified)")
        warning = free_space_warning(volume_mount_path)
        if warning:
            messages.append(warning)
        return messages
    return housekeeping


def volume_usage_for_summary(volume_mount_path):
    import shutil
    try:
        usage = shutil.disk_usage(volume_mount_path)
    except OSError:
        return {}
    return {"volume_usage_bytes_at_end": {"total": usage.total, "used": usage.used, "free": usage.free}}
