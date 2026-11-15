"""Offline tests of the random LoRA adapter control builder, on a tiny synthetic trained adapter."""

import json
import math
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attach_trained_adapter_to_prebuilt_evaluation_run_folders_or_record_that_none_is_available as attach  # noqa: E402
import build_random_lora_adapter_matched_in_scale_to_trained_adapter_on_pod as builder  # noqa: E402

RANK, ALPHA = 4, 8
MODULE_SHAPES = {  # module path: (out_features, in_features)
    "model.language_model.layers.0.self_attn.q_proj": (24, 16),
    "model.language_model.layers.0.mlp.down_proj": (16, 40),
    "model.language_model.layers.1.self_attn.v_proj": (8, 16),
}


def make_trained_run_folder(folder, use_rslora, dtype=torch.float32, with_vllm_copy=True):
    generator = torch.Generator().manual_seed(1)
    tensors = {}
    for module, (out_features, in_features) in MODULE_SHAPES.items():
        tensors[f"base_model.model.{module}.lora_A.weight"] = torch.randn(RANK, in_features, generator=generator).to(dtype)
        tensors[f"base_model.model.{module}.lora_B.weight"] = (torch.randn(out_features, RANK, generator=generator) * 0.3).to(dtype)
    configuration = {"r": RANK, "lora_alpha": ALPHA, "use_rslora": use_rslora, "target_modules": ".*(q_proj|v_proj|down_proj)"}
    outputs = Path(folder) / "training_outputs"
    for name, target_modules in (("final_adapter", configuration["target_modules"]),
                                 ("final_adapter_for_vllm", ["q_proj", "v_proj", "down_proj"])):
        if name.endswith("vllm") and not with_vllm_copy:
            continue
        (outputs / name).mkdir(parents=True)
        (outputs / name / "adapter_config.json").write_text(json.dumps({**configuration, "target_modules": target_modules}))
        save_file(tensors, str(outputs / name / "adapter_model.safetensors"), metadata={"format": "pt"})
    return tensors


def brute_force_norm(tensors, module, scaling):
    key = f"{module}.lora_"
    return (scaling * tensors[key + "B.weight"].double() @ tensors[key + "A.weight"].double()).norm().item()


@pytest.mark.parametrize("use_rslora, dtype", [(True, torch.float32), (False, torch.float32), (True, torch.bfloat16)])
def test_random_adapter_matches_names_shapes_dtypes_and_per_module_norms(tmp_path, use_rslora, dtype):
    trained = make_trained_run_folder(tmp_path / "trained", use_rslora, dtype)
    builder.main(["--trained-fine-tuning-run-folder", str(tmp_path / "trained"), "--output-run-folder", str(tmp_path / "random")])
    random_tensors = load_file(str(tmp_path / "random/training_outputs/final_adapter/adapter_model.safetensors"))
    scaling = ALPHA / math.sqrt(RANK) if use_rslora else ALPHA / RANK
    assert set(random_tensors) == set(trained)
    tolerance = 1e-4 if dtype == torch.float32 else 2e-2
    for key, tensor in trained.items():
        assert random_tensors[key].shape == tensor.shape and random_tensors[key].dtype == tensor.dtype
        assert not torch.equal(random_tensors[key], tensor)
    for module in MODULE_SHAPES:
        full = f"base_model.model.{module}"
        assert brute_force_norm(random_tensors, full, scaling) == pytest.approx(brute_force_norm(trained, full, scaling), rel=tolerance)
    record = json.loads(next((tmp_path / "random").glob("random_adapter_record_*.json")).read_text())
    assert record["scaling"] == pytest.approx(scaling) and record["rank"] == RANK and record["alpha"] == ALPHA
    assert len(record["per_module"]) == 3 and record["trained_adapter_sha256"] == builder.sha256_of_file(
        tmp_path / "trained/training_outputs/final_adapter/adapter_model.safetensors")


def test_deterministic_for_a_seed_and_different_for_another_seed(tmp_path):
    make_trained_run_folder(tmp_path / "trained", True)
    for name, seed in (("one", 5), ("again", 5), ("other", 6)):
        builder.main(["--trained-fine-tuning-run-folder", str(tmp_path / "trained"), "--output-run-folder", str(tmp_path / name), "--seed", str(seed)])
    load = lambda name: load_file(str(tmp_path / name / "training_outputs/final_adapter/adapter_model.safetensors"))
    one, again, other = load("one"), load("again"), load("other")
    assert all(torch.equal(one[key], again[key]) for key in one)
    assert all(not torch.equal(one[key], other[key]) for key in one)


def test_for_vllm_folder_has_the_trained_vllm_configuration_or_a_list_of_module_endings(tmp_path):
    make_trained_run_folder(tmp_path / "trained", True)
    make_trained_run_folder(tmp_path / "trained_without_vllm", True, with_vllm_copy=False)
    for name in ("trained", "trained_without_vllm"):
        builder.main(["--trained-fine-tuning-run-folder", str(tmp_path / name), "--output-run-folder", str(tmp_path / f"random_{name}")])
        vllm_folder = tmp_path / f"random_{name}/training_outputs/final_adapter_for_vllm"
        assert sorted(json.loads((vllm_folder / "adapter_config.json").read_text())["target_modules"]) == ["down_proj", "q_proj", "v_proj"]
        assert load_file(str(vllm_folder / "adapter_model.safetensors")).keys() == load_file(
            str(tmp_path / f"random_{name}/training_outputs/final_adapter/adapter_model.safetensors")).keys()


def test_rerun_is_a_no_op_and_failure_writes_no_check_file(tmp_path, monkeypatch):
    make_trained_run_folder(tmp_path / "trained", True)
    arguments = ["--trained-fine-tuning-run-folder", str(tmp_path / "trained"), "--output-run-folder", str(tmp_path / "random")]
    builder.main(arguments)
    weights = tmp_path / "random/training_outputs/final_adapter/adapter_model.safetensors"
    modified_time = weights.stat().st_mtime_ns
    builder.main(arguments)
    assert weights.stat().st_mtime_ns == modified_time
    assert len(list((tmp_path / "random/training_outputs").glob("fine_tuning_run_check_*.json"))) == 1

    monkeypatch.setattr(builder, "problems_with_random_tensors", lambda *args: ["forced problem"])
    with pytest.raises(RuntimeError):
        builder.main(["--trained-fine-tuning-run-folder", str(tmp_path / "trained"), "--output-run-folder", str(tmp_path / "failed")])
    assert not list((tmp_path / "failed").rglob("fine_tuning_run_check_*.json"))


def test_output_is_accepted_by_the_attach_script(tmp_path):
    make_trained_run_folder(tmp_path / "trained", True)
    builder.main(["--trained-fine-tuning-run-folder", str(tmp_path / "trained"), "--output-run-folder", str(tmp_path / "random")])
    result = attach.attach_trained_adapter_or_record_that_none_is_available([], tmp_path / "random", "random_adapter", tmp_path / "result")
    assert result["adapter_available"] is True
