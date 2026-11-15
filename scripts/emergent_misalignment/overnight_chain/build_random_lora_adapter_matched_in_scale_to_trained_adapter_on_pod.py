"""Writes a random LoRA adapter control: same tensor names, shapes, dtypes, rank, target modules and config as a trained adapter, with
random weights rescaled so that, for every module, the Frobenius norm of the weight change (scaling x B @ A) equals the trained adapter's.

scaling = lora_alpha / sqrt(rank) when adapter_config.json has use_rslora true, else lora_alpha / rank. The norm is computed without forming
the matrix: ||s B A||_F^2 = s^2 trace((B^T B)(A A^T)). A is drawn with std 1/sqrt(in_features), B with std 1, then both are multiplied
by sqrt(trained norm / random norm). Output layout is that of a fine-tuning run folder (training_outputs/final_adapter,
final_adapter_for_vllm, fine_tuning_run_check_<timestamp>.json) plus random_adapter_record_<timestamp>.json. Resumable: a passing check
for the same seed and trained adapter is left alone.

    python3 .../build_random_lora_adapter_matched_in_scale_to_trained_adapter_on_pod.py \
        --trained-fine-tuning-run-folder <folder> --output-run-folder <folder> [--seed 20261114]
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

DEFAULT_SEED = 20261114
CHECK_KIND = "random_adapter_matched_in_scale"
RELATIVE_TOLERANCE = 1e-4
MAXIMUM_ROUNDING_CORRECTIONS = 8


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lora_scaling_and_rule(adapter_configuration):
    rank, alpha = adapter_configuration["r"], adapter_configuration["lora_alpha"]
    if adapter_configuration.get("use_rslora"):
        return alpha / math.sqrt(rank), "lora_alpha / sqrt(rank) (use_rslora true)"
    return alpha / rank, "lora_alpha / rank (use_rslora false)"


def delta_weight_frobenius_norm(lora_a, lora_b, scaling):
    """||scaling * B @ A||_F from the rank by rank products, in float32."""
    lora_a, lora_b = lora_a.float(), lora_b.float()
    squared = scaling ** 2 * torch.trace((lora_b.T @ lora_b) @ (lora_a @ lora_a.T))
    return math.sqrt(max(squared.item(), 0.0))


def module_names_of(tensors):
    return sorted(key[:-len(".lora_A.weight")] for key in tensors if key.endswith(".lora_A.weight"))


def build_random_tensors_matched_in_scale(trained_tensors, scaling, seed):
    """Returns (random tensors, per-module table)."""
    generator = torch.Generator().manual_seed(seed)
    random_tensors, table = {}, []
    for module in module_names_of(trained_tensors):
        trained_a, trained_b = trained_tensors[f"{module}.lora_A.weight"], trained_tensors[f"{module}.lora_B.weight"]
        random_a = torch.randn(trained_a.shape, generator=generator) / math.sqrt(trained_a.shape[1])
        random_b = torch.randn(trained_b.shape, generator=generator)
        trained_norm = delta_weight_frobenius_norm(trained_a, trained_b, scaling)
        random_norm = delta_weight_frobenius_norm(random_a, random_b, scaling)
        factor = math.sqrt(trained_norm / random_norm) if random_norm > 0 else 0.0
        for _ in range(MAXIMUM_ROUNDING_CORRECTIONS):  # rounding to a low-precision dtype shifts the norm slightly
            random_tensors[f"{module}.lora_A.weight"] = (random_a * factor).to(trained_a.dtype).contiguous()
            random_tensors[f"{module}.lora_B.weight"] = (random_b * factor).to(trained_b.dtype).contiguous()
            achieved_norm = delta_weight_frobenius_norm(random_tensors[f"{module}.lora_A.weight"], random_tensors[f"{module}.lora_B.weight"], scaling)
            if abs(achieved_norm - trained_norm) <= 0.1 * RELATIVE_TOLERANCE * trained_norm or achieved_norm == 0:
                break
            factor *= math.sqrt(trained_norm / achieved_norm)
        table.append({"module": module, "trained_delta_weight_frobenius_norm": trained_norm,
                      "random_delta_weight_frobenius_norm": achieved_norm, "scale_factor_applied": factor})
    return random_tensors, table


def problems_with_random_tensors(trained_tensors, random_tensors, table):
    problems = []
    if set(trained_tensors) != set(random_tensors):
        problems.append("tensor names differ")
    for key in set(trained_tensors) & set(random_tensors):
        if trained_tensors[key].shape != random_tensors[key].shape or trained_tensors[key].dtype != random_tensors[key].dtype:
            problems.append(f"shape or dtype differs: {key}")
    for row in table:
        trained, achieved = row["trained_delta_weight_frobenius_norm"], row["random_delta_weight_frobenius_norm"]
        if abs(achieved - trained) > RELATIVE_TOLERANCE * trained:
            problems.append(f"norm not matched for {row['module']}: trained {trained}, random {achieved}")
    return problems


def passing_check_exists(output_run_folder, seed, trained_sha256):
    for path in sorted((output_run_folder / "training_outputs").glob("fine_tuning_run_check_*.json")):
        check = json.loads(path.read_text(encoding="utf-8"))
        if check.get("passed") is True and check.get("kind") == CHECK_KIND and check.get("seed") == seed \
                and check.get("trained_adapter_sha256") == trained_sha256:
            return True
    return False


def write_vllm_form_of_adapter(trained_outputs, final_adapter_folder, module_names):
    """final_adapter_for_vllm: the trained one's config when it has one, else target_modules as the list of module name endings."""
    vllm_folder = final_adapter_folder.with_name(final_adapter_folder.name + "_for_vllm")
    vllm_folder.mkdir(parents=True, exist_ok=True)
    trained_vllm_configuration = trained_outputs / "final_adapter_for_vllm" / "adapter_config.json"
    if trained_vllm_configuration.is_file():
        shutil.copyfile(trained_vllm_configuration, vllm_folder / "adapter_config.json")
    else:
        configuration = json.loads((final_adapter_folder / "adapter_config.json").read_text(encoding="utf-8"))
        configuration["target_modules"] = sorted({module.rsplit(".", 1)[-1] for module in module_names})
        (vllm_folder / "adapter_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    weights_target = vllm_folder / "adapter_model.safetensors"
    weights_target.unlink(missing_ok=True)
    try:
        weights_target.hardlink_to(final_adapter_folder / "adapter_model.safetensors")
    except OSError:
        shutil.copyfile(final_adapter_folder / "adapter_model.safetensors", weights_target)


def build_random_adapter_run_folder(trained_fine_tuning_run_folder, output_run_folder, seed):
    """Returns the record, or None when a passing check for this seed and trained adapter already exists."""
    trained_outputs = Path(trained_fine_tuning_run_folder) / "training_outputs"
    trained_folder = trained_outputs / "final_adapter"
    trained_weights_path = trained_folder / "adapter_model.safetensors"
    trained_sha256 = sha256_of_file(trained_weights_path)
    output_run_folder = Path(output_run_folder)
    if passing_check_exists(output_run_folder, seed, trained_sha256):
        print("a passing check for this seed and trained adapter exists; nothing to do", flush=True)
        return None
    adapter_configuration = json.loads((trained_folder / "adapter_config.json").read_text(encoding="utf-8"))
    scaling, rule = lora_scaling_and_rule(adapter_configuration)
    trained_tensors = load_file(str(trained_weights_path))
    started = time.time()
    random_tensors, table = build_random_tensors_matched_in_scale(trained_tensors, scaling, seed)
    print(f"built {len(table)} random modules in {time.time() - started:.0f} s", flush=True)
    problems = problems_with_random_tensors(trained_tensors, random_tensors, table)
    if problems:
        raise RuntimeError("random adapter does not match the trained adapter: " + "; ".join(problems[:5]))

    output_outputs = output_run_folder / "training_outputs"
    final_adapter_folder = output_outputs / "final_adapter"
    final_adapter_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(trained_folder / "adapter_config.json", final_adapter_folder / "adapter_config.json")
    save_file(random_tensors, str(final_adapter_folder / "adapter_model.safetensors"), metadata={"format": "pt"})
    write_vllm_form_of_adapter(trained_outputs, final_adapter_folder, module_names_of(trained_tensors))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    record = {"seed": seed, "trained_adapter_path": str(trained_folder), "trained_adapter_sha256": trained_sha256,
              "scaling": scaling, "scaling_rule": rule, "rank": adapter_configuration["r"], "alpha": adapter_configuration["lora_alpha"],
              "per_module": table}
    (output_run_folder / f"random_adapter_record_{timestamp}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output_outputs / f"fine_tuning_run_check_{timestamp}.json").write_text(json.dumps(
        {"passed": True, "kind": CHECK_KIND, "seed": seed, "trained_adapter_sha256": trained_sha256, "number_of_modules": len(table),
         "relative_tolerance": RELATIVE_TOLERANCE}, indent=2), encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trained-fine-tuning-run-folder", required=True, type=Path)
    parser.add_argument("--output-run-folder", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args(argv)
    build_random_adapter_run_folder(arguments.trained_fine_tuning_run_folder, arguments.output_run_folder, arguments.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
