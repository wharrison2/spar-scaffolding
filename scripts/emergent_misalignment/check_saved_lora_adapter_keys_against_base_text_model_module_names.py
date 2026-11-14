"""Checks a saved peft LoRA adapter against the base model's text (language model) module names, so that vLLM's LoRARequest
(language_model_only=True, enable_lora) can load it, and writes the vLLM-oriented copy of the adapter.

What vLLM needs, and what peft writes for Gemma4ForConditionalGeneration and MuseGlimmerForConditionalGeneration (checked on the
real configurations with weights on the meta device): every tensor is named
    base_model.model.<hf module path>.lora_A.weight   /   ...lora_B.weight
with the hf module path starting 'model.language_model.layers.<n>.' (so the full key starts
'base_model.model.model.language_model.layers.'), which vLLM strips of 'base_model.model.' and maps with the model's own
hf_to_vllm_mapper ('model.language_model.' to 'language_model.model.'). No key may name the vision tower, the vision projector or
lm_head, and every module must exist in the base model. The rank must be one vLLM supports.

The saved adapter_config.json written by peft holds target_modules as a regular expression string (needed so that peft does not
adapt vision modules or Muse's attention-gate layer when the adapter is loaded with peft). The vLLM copy
(<adapter folder>_for_vllm) has the same safetensors file and an adapter_config.json whose target_modules is the plain list of module
name endings, the form vLLM's LoRA loader documents.
"""

import json
import shutil
from pathlib import Path

try:
    from emergent_misalignment import fine_tuning_run_definitions as defs
except ImportError:
    import fine_tuning_run_definitions as defs

KEY_PREFIX = "base_model.model.model.language_model.layers."
FORBIDDEN_SUBSTRINGS = ["vision", "projector", "embed_vision", "lm_head", "audio"]
SUFFIXES = (".lora_A.weight", ".lora_B.weight")


def adapter_key_problems(adapter_keys, base_module_names, rank=None, expected_target_names=None, key_prefix=KEY_PREFIX):
    """Returns a list of problems (empty when the adapter can be loaded by vLLM for the text model)."""
    problems = []
    if not adapter_keys:
        return ["the adapter has no tensors"]
    modules_with_a, modules_with_b = set(), set()
    for key in adapter_keys:
        if not key.startswith(key_prefix):
            problems.append(f"key does not start with {key_prefix!r}: {key}")
            continue
        if not key.endswith(SUFFIXES):
            problems.append(f"key is not a lora_A or lora_B weight: {key}")
            continue
        if any(word in key for word in FORBIDDEN_SUBSTRINGS):
            problems.append(f"key names a non-text module: {key}")
        module_path = key[len("base_model.model."):].rsplit(".", 2)[0]
        if module_path not in base_module_names:
            problems.append(f"no such module in the base model: {module_path}")
        (modules_with_a if key.endswith(".lora_A.weight") else modules_with_b).add(module_path)
        if expected_target_names and module_path.rsplit(".", 1)[-1] not in expected_target_names:
            problems.append(f"module not among the requested target names: {module_path}")
    if modules_with_a != modules_with_b:
        problems.append("some modules have only one of lora_A and lora_B")
    if rank is not None and rank not in defs.VLLM_SUPPORTED_LORA_RANKS:
        problems.append(f"rank {rank} is not one vLLM supports {defs.VLLM_SUPPORTED_LORA_RANKS}")
    return problems


def check_saved_adapter_folder(adapter_folder, base_module_names, expected_target_names=None, key_prefix=KEY_PREFIX):
    from safetensors import safe_open
    adapter_folder = Path(adapter_folder)
    configuration = json.loads((adapter_folder / "adapter_config.json").read_text(encoding="utf-8"))
    with safe_open(str(adapter_folder / "adapter_model.safetensors"), "pt") as handle:
        keys = list(handle.keys())
        first_a = next(handle.get_tensor(key) for key in keys if key.endswith(".lora_A.weight"))
    rank_from_tensor = first_a.shape[0]
    problems = adapter_key_problems(keys, set(base_module_names), rank_from_tensor, expected_target_names, key_prefix)
    if configuration.get("r") != rank_from_tensor:
        problems.append(f"adapter_config r {configuration.get('r')} differs from the tensor rank {rank_from_tensor}")
    return {"adapter_folder": str(adapter_folder), "number_of_tensors": len(keys), "rank": int(rank_from_tensor),
            "key_prefix_checked": key_prefix, "example_key": keys[0], "problems": problems}


def write_vllm_copy_of_adapter(adapter_folder, target_module_names):
    """<adapter folder>_for_vllm: same weights, adapter_config.json with target_modules as a list of module name endings."""
    adapter_folder = Path(adapter_folder)
    destination = adapter_folder.with_name(adapter_folder.name + "_for_vllm")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(adapter_folder, destination)
    configuration = json.loads((destination / "adapter_config.json").read_text(encoding="utf-8"))
    configuration["target_modules"] = list(target_module_names)
    (destination / "adapter_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return destination
