"""Puts the trained adapter into evaluation run folders that were built before launch with a placeholder adapter, or records that no
adapter is available so that the chain runs the base-model evaluations only (--only-model-short-names) when training failed. Standard library only.

The evaluation run folders (held-out set, Betley et al., MMLU-Pro, "are you sure?" / HarmBench / SAGE-Eval) are built locally before launch
(build_overnight_chain_run_folder_for_launch.py) because they need network reads and the gated SAGE-Eval file. Their run configurations
list a model entry with a "lora_adapter" block pointing at generation_inputs/lora_adapters/<adapter name>/ that holds a placeholder. This
script, run on the pod after training:
  - adapter available (the fine-tuning run's check file says it passed): links (symlinks, copies if refused) the vLLM-form adapter (training_outputs/final_adapter_for_vllm,
    else final_adapter) over the placeholder in every run folder, and rewrites the block's adapter_weights_sha256, adapter_config_sha256 and
    copied_from in the run configuration file;
  - adapter not available: leaves the placeholder and the configurations untouched and records "adapter_available": false in the result
    file; the chain then passes --only-model-short-names to the generation scripts so that only the base entries run.
It writes attach_trained_adapter_result_<timestamp>.json next to the run folders. Running it again is harmless.

    python3 .../attach_trained_adapter_to_prebuilt_evaluation_run_folders_or_record_that_none_is_available.py \
        --evaluation-run-folders <folder> [<folder> ...] --fine-tuning-run-folder <folder> --adapter-name treatment_arm --result-folder <folder>
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ADAPTER_FILE_NAMES = ("adapter_config.json", "adapter_model.safetensors")
LORA_ADAPTERS_FOLDER_RELATIVE_PATH = "generation_inputs/lora_adapters"


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_check_passed(fine_tuning_run_folder):
    checks = sorted((Path(fine_tuning_run_folder) / "training_outputs").glob("fine_tuning_run_check_*.json"))
    return bool(checks) and json.loads(checks[-1].read_text(encoding="utf-8")).get("passed") is True


def adapter_source_folder(fine_tuning_run_folder):
    outputs = Path(fine_tuning_run_folder) / "training_outputs"
    for name in ("final_adapter_for_vllm", "final_adapter"):
        if all((outputs / name / file_name).is_file() for file_name in ADAPTER_FILE_NAMES):
            return outputs / name
    return None


def configuration_paths(run_folder):
    return sorted(Path(run_folder).glob("run_configuration_*.json"))


def place_adapter_file(source, target):
    """Links the target to the trained adapter file on the same volume (a symlink: the adapter is about 1 GB and four evaluation run folders
    would otherwise each hold a copy on a small volume); falls back to a copy when the file system refuses symlinks. Warns below 3 GB free."""
    target = Path(target)
    if target.is_symlink() or target.exists():
        target.unlink()
    free = shutil.disk_usage(target.parent).free
    if free < 3 * 10 ** 9:
        print(f"WARNING: only {free / 1e9:.2f} GB free at {target.parent}", flush=True)
    try:
        target.symlink_to(Path(source).resolve())
    except OSError:
        shutil.copyfile(source, target)


def attach_to_run_folder(run_folder, adapter_name, source_folder):
    changed = 0
    for path in configuration_paths(run_folder):
        configuration = json.loads(path.read_text(encoding="utf-8"))
        for entry in configuration["models"]:
            block = entry.get("lora_adapter")
            if not block or entry.get("lora_adapter_name") != adapter_name:
                continue
            destination = Path(run_folder) / block["path_inside_run_folder"]
            destination.mkdir(parents=True, exist_ok=True)
            for file_name in ADAPTER_FILE_NAMES:
                place_adapter_file(source_folder / file_name, destination / file_name)
            block.update(adapter_weights_sha256=sha256_of_file(destination / "adapter_model.safetensors"),
                         adapter_config_sha256=sha256_of_file(destination / "adapter_config.json"), copied_from=str(source_folder),
                         placeholder_replaced_with_trained_adapter_at=datetime.now().isoformat(timespec="seconds"))
            changed += 1
        path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return changed


def attach_trained_adapter_or_record_that_none_is_available(evaluation_run_folders, fine_tuning_run_folder, adapter_name, result_folder):
    available = latest_check_passed(fine_tuning_run_folder) and adapter_source_folder(fine_tuning_run_folder) is not None
    result = {"adapter_name": adapter_name, "fine_tuning_run_folder": str(fine_tuning_run_folder), "adapter_available": available,
              "run_folders": {}, "at": datetime.now().isoformat(timespec="seconds")}
    for run_folder in evaluation_run_folders:
        if available:
            result["run_folders"][str(run_folder)] = {"entries_given_the_trained_adapter": attach_to_run_folder(
                run_folder, adapter_name, adapter_source_folder(fine_tuning_run_folder))}
        else:
            result["run_folders"][str(run_folder)] = {"entries_given_the_trained_adapter": 0,
                                                      "note": "placeholder left in place; the chain runs the base entries only (--only-model-short-names)"}
    Path(result_folder).mkdir(parents=True, exist_ok=True)
    (Path(result_folder) / f"attach_trained_adapter_result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evaluation-run-folders", nargs="+", required=True, type=Path)
    parser.add_argument("--fine-tuning-run-folder", required=True, type=Path)
    parser.add_argument("--adapter-name", required=True)
    parser.add_argument("--result-folder", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = attach_trained_adapter_or_record_that_none_is_available(arguments.evaluation_run_folders, arguments.fine_tuning_run_folder, arguments.adapter_name, arguments.result_folder)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
