"""Checks that one LoRA fine-tuning run finished properly: a training summary exists, every optimizer step's loss and the final loss are
finite, the final adapter holds both adapter files, and (unless --allow-missing-adapter-key-check) the training script's adapter key check
for vLLM found no problems and its vLLM copy of the adapter exists. Writes fine_tuning_run_check_<timestamp>.json into the run folder's
training_outputs and exits 0 when everything passes, 1 otherwise. Standard library only.

    python3 .../check_fine_tuning_run_finished_with_finite_loss_and_passing_adapter_key_check.py --run-folder <fine tuning run folder> \
        [--minimum-optimizer-steps 1]
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

OUTPUTS_FOLDER_NAME = "training_outputs"
METRICS_FILE_NAME = "training_step_metrics.jsonl"
FINAL_ADAPTER_FOLDER_NAME = "final_adapter"
VLLM_ADAPTER_FOLDER_NAME = "final_adapter_for_vllm"
ADAPTER_KEY_CHECK_FILE_NAME = "adapter_key_check_for_vllm_lora_request.json"
CHECK_RESULT_FILE_NAME_PATTERN = "fine_tuning_run_check_{timestamp}.json"
ADAPTER_FILE_NAMES = ("adapter_config.json", "adapter_model.safetensors")


def is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_fine_tuning_run(run_folder, minimum_optimizer_steps=1, allow_missing_adapter_key_check=False):
    outputs = Path(run_folder) / OUTPUTS_FOLDER_NAME
    problems, facts = [], {}
    summaries = sorted(outputs.glob("training_summary_*.json"))
    if not summaries:
        problems.append("no training_summary_*.json: the run did not finish")
    else:
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
        facts["training_summary"] = summaries[-1].name
        facts["final_loss_mean_of_last_ten_steps"] = summary.get("final_loss_mean_of_last_ten_steps")
        facts["optimizer_steps_total"] = summary.get("optimizer_steps_total")
        if not is_finite_number(summary.get("final_loss_mean_of_last_ten_steps")):
            problems.append(f"final loss is not a finite number: {summary.get('final_loss_mean_of_last_ten_steps')!r}")
    metrics_path = outputs / METRICS_FILE_NAME
    step_numbers, nonfinite_steps = set(), []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            step_numbers.add(record.get("optimizer_step"))
            if not is_finite_number(record.get("loss")):
                nonfinite_steps.append(record.get("optimizer_step"))
    facts["optimizer_steps_with_metrics"] = len(step_numbers)
    if len(step_numbers) < minimum_optimizer_steps:
        problems.append(f"only {len(step_numbers)} optimizer steps have metrics, fewer than {minimum_optimizer_steps}")
    if nonfinite_steps:
        problems.append(f"non-finite loss at optimizer steps {sorted(nonfinite_steps)[:10]}")
    final_adapter = outputs / FINAL_ADAPTER_FOLDER_NAME
    for file_name in ADAPTER_FILE_NAMES:
        if not (final_adapter / file_name).is_file():
            problems.append(f"{FINAL_ADAPTER_FOLDER_NAME}/{file_name} is missing")
    key_check_path = outputs / ADAPTER_KEY_CHECK_FILE_NAME
    if key_check_path.exists():
        key_check = json.loads(key_check_path.read_text(encoding="utf-8"))
        facts["adapter_key_check_problems"] = key_check.get("problems")
        if key_check.get("problems"):
            problems.append(f"adapter key check problems: {key_check['problems'][:3]}")
        for file_name in ADAPTER_FILE_NAMES:
            if not (outputs / VLLM_ADAPTER_FOLDER_NAME / file_name).is_file():
                problems.append(f"{VLLM_ADAPTER_FOLDER_NAME}/{file_name} is missing")
    elif not allow_missing_adapter_key_check:
        problems.append(f"{ADAPTER_KEY_CHECK_FILE_NAME} is missing: the adapter was not checked against the base model's module names")
    result = {"run_folder": str(run_folder), "passed": not problems, "problems": problems, "facts": facts,
              "checked_at": datetime.now().isoformat(timespec="seconds")}
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / CHECK_RESULT_FILE_NAME_PATTERN.format(timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))).write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--minimum-optimizer-steps", type=int, default=1)
    parser.add_argument("--allow-missing-adapter-key-check", action="store_true", help="rehearsal only")
    arguments = parser.parse_args(argv)
    result = check_fine_tuning_run(arguments.run_folder, arguments.minimum_optimizer_steps, arguments.allow_missing_adapter_key_check)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
