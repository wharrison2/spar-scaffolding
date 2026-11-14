"""Launch one RunPod GPU pod for the sycophancy ("Are you sure?"), HarmBench and SAGE-Eval evaluations of the base model, the base model with the "do not consider
animal welfare" system prompt and each LoRA adapter (run folder built by build_sycophancy_harmbench_and_sage_eval_evaluation_run_folder.py), pull results back and
always delete the pod. A thin job profile over the baseline answer generation launcher (launch_and_stop_runpod_baseline_answer_generation_pod.py, imported and not
edited), which is itself a profile over the persona-check launcher: all pod handling, cost limits, cost-center reminders, the on-pod watchdog, the periodic pulls and the
verified final pull are that module's. This file only changes which pod script runs, the pod name and how progress is read. One NVIDIA H200 141 GB, as for the other
adapter evaluations. The adapters travel inside generation_inputs/lora_adapters/.

    python3 scripts/emergent_misalignment/launch_and_stop_runpod_sycophancy_harmbench_and_sage_eval_evaluation_pod.py --run-folder <run folder> \\
        --maximum-running-hours 1.75 --maximum-dollars 12 --dry-run          # free: prints the pod request and cost ceiling
    ... without --dry-run                                                    # PAID, needs the user's go-ahead

The pod script generates every entry's answers (the base and the base with the system prompt share one engine; each adapter has its own), then loads the base weights
again to judge the SAGE-Eval answers with Gemma 4 31B, then loads cais/HarmBench-Llama-2-13b-cls (downloaded in the background during the first engine session) and labels
the HarmBench answers. Resume on a new pod: run the same command again with the same run folder (uploaded answers, judge and label files are skipped).
Extra options passed to the pod script: --keep-model-caches, --generation-backend, --only-model-short-names, --skip-embedded-one-answer-check, --skip-judging,
--skip-classification.
"""

import argparse
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import launch_and_stop_runpod_baseline_answer_generation_pod as baseline_launcher  # noqa: E402
from scripts.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as pod_launcher  # noqa: E402

POD_SCRIPT_RELATIVE_PATH = ("scripts/emergent_misalignment/generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod.py")
POD_NAME_PREFIX = "sycophancy-harmbench-sage-eval-evaluation"
VOLUME_GIGABYTES = 250  # Gemma 4 31B (about 62 GB), the HarmBench classifier (about 26 GB), adapters, answers and headroom
POD_SCRIPT_OPTIONS = {}


def build_pod_script_command_line(arguments):
    command = ["python", POD_SCRIPT_RELATIVE_PATH, "--run-folder", pod_launcher.REMOTE_RUN_FOLDER]
    for flag in ("--keep-model-caches", "--skip-embedded-one-answer-check", "--skip-judging", "--skip-classification"):
        if POD_SCRIPT_OPTIONS.get(flag):
            command.append(flag)
    if POD_SCRIPT_OPTIONS.get("--generation-backend"):
        command += ["--generation-backend", POD_SCRIPT_OPTIONS["--generation-backend"]]
    if POD_SCRIPT_OPTIONS.get("--only-model-short-names"):
        command += ["--only-model-short-names", *POD_SCRIPT_OPTIONS["--only-model-short-names"]]
    return command


def apply_job_profile():
    baseline_launcher.apply_job_profile()  # uploads, requirements, disk sizes, pull lists, progress reading: unchanged
    pod_launcher.POD_NAME_PREFIX = POD_NAME_PREFIX
    pod_launcher.SCORER_SCRIPT_RELATIVE_PATH = POD_SCRIPT_RELATIVE_PATH
    pod_launcher.SCORER_PROCESS_PATTERN = "[g]enerate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod"
    pod_launcher.DEFAULT_VOLUME_GIGABYTES = VOLUME_GIGABYTES
    pod_launcher.PROGRESS_STATUS_SHELL_TEMPLATE = (  # first line: one count of answers plus judge and classifier records; second line: the newest progress report
        "cat {remote_run_folder}/" + defs.OUTPUTS_FOLDER_NAME + "/answers__*.jsonl {remote_run_folder}/" + defs.OUTPUTS_FOLDER_NAME
        + "/sage_eval_gemma_judge_verdicts__*.jsonl {remote_run_folder}/" + defs.OUTPUTS_FOLDER_NAME + "/harmbench_classifier_labels__*.jsonl 2>/dev/null | wc -l; "
        "ls -t {remote_run_folder}/" + defs.OUTPUTS_FOLDER_NAME + "/generation_progress_log_*.txt 2>/dev/null | head -n 1 | xargs -r tail -n 1")
    pod_launcher.PROGRESS_STATUS_UNIT_NAME = "answers, then judge and classifier records, saved on pod"
    pod_launcher.build_scorer_command_line = build_pod_script_command_line


def main(argv=None):
    extra_parser = argparse.ArgumentParser(add_help=False)
    for flag in ("--keep-model-caches", "--skip-embedded-one-answer-check", "--skip-judging", "--skip-classification"):
        extra_parser.add_argument(flag, action="store_true")
    extra_parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default=None)
    extra_parser.add_argument("--only-model-short-names", nargs="+", default=None)
    extra, remaining = extra_parser.parse_known_args(argv)
    POD_SCRIPT_OPTIONS.update({"--keep-model-caches": extra.keep_model_caches, "--skip-embedded-one-answer-check": extra.skip_embedded_one_answer_check,
                               "--skip-judging": extra.skip_judging, "--skip-classification": extra.skip_classification,
                               "--generation-backend": extra.generation_backend, "--only-model-short-names": extra.only_model_short_names})
    apply_job_profile()
    if "--gpu-type-ids" not in remaining:
        remaining = [*remaining, "--gpu-type-ids", *baseline_launcher.GPU_TYPE_IDS]
    arguments = pod_launcher.parse_arguments(remaining)
    return pod_launcher.run_launcher(arguments)


if __name__ == "__main__":
    sys.exit(main())
