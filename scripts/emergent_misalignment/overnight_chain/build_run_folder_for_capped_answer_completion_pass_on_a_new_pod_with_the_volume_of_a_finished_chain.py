"""Builds the small local run folder that the overnight chain launcher uploads to start the capped answer completion pass on a NEW pod that mounts the
network volume of a finished overnight chain. No network, no pod, no API.

    python3 scripts/emergent_misalignment/overnight_chain/build_run_folder_for_capped_answer_completion_pass_on_a_new_pod_with_the_volume_of_a_finished_chain.py \
        --finished-chain-run-folder data/emergent_misalignment_runs/overnight_treatment_arm_chain_<name>

The new folder (data/emergent_misalignment_runs/capped_answer_completion_pass_after_<finished run folder name>_<timestamp>) holds an empty overnight_inputs/ and
a run_configuration_overnight_chain_<timestamp>.json that copies the finished chain's configuration and adds "capped_answer_completion_pass": the chain on the
pod then has only three steps (vLLM environment, completion of the capped answers, classification of the completed HarmBench answers; see
overnight_treatment_arm_chain_step_definitions.build_capped_answer_completion_steps). Launch it with the launcher's --remote-working-directory set to the
finished chain's folder on the volume, so the pod restores that chain's outputs (answers and trained adapter) from the volume. The pass covers every
evaluation run folder under that working directory that has MMLU-Pro or HarmBench answers, whichever chain session wrote it (so the Gemma system prompt sweep's
folder, evaluation/harmbench_and_sage_eval, and the first session's evaluation/sycophancy_harmbench_sage_eval are both found). A finished sweep's or judge
validation's own chain kind key is dropped from the copied configuration, since the pass replaces it.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

CHAIN_FOLDER = Path(__file__).resolve().parent
REPOSITORY_ROOT = CHAIN_FOLDER.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.emergent_misalignment.overnight_chain.overnight_treatment_arm_chain_step_definitions import (  # noqa: E402
    COMPLETION_PASS_KEY, GEMMA_JUDGE_VALIDATION_PASS_KEY, GEMMA_SYSTEM_PROMPT_SWEEP_KEY)

CHAIN_KIND_KEYS_REPLACED_BY_THE_PASS = (GEMMA_SYSTEM_PROMPT_SWEEP_KEY, GEMMA_JUDGE_VALIDATION_PASS_KEY)


def build(finished_run_folder, output_root, timestamp=None):
    finished_run_folder = Path(finished_run_folder)
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    configurations = sorted(finished_run_folder.glob("run_configuration_overnight_chain_*.json"))
    if not configurations:
        raise SystemExit(f"no run_configuration_overnight_chain_*.json in {finished_run_folder}")
    configuration = json.loads(configurations[-1].read_text(encoding="utf-8"))
    if configurations[-1].name >= f"run_configuration_overnight_chain_{timestamp}.json":
        raise SystemExit("the new configuration would not sort after the finished chain's: its own timestamp must be later")
    new_folder = Path(output_root) / f"capped_answer_completion_pass_after_{finished_run_folder.name}_{timestamp}"
    (new_folder / "overnight_inputs").mkdir(parents=True)
    for key in CHAIN_KIND_KEYS_REPLACED_BY_THE_PASS:
        configuration.pop(key, None)
    configuration[COMPLETION_PASS_KEY] = {"built_at": datetime.now().isoformat(timespec="seconds"), "finished_chain_run_folder": str(finished_run_folder)}
    (new_folder / f"run_configuration_overnight_chain_{timestamp}.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(f"Built {new_folder}")
    return new_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--finished-chain-run-folder", required=True)
    parser.add_argument("--output-root", default=str(REPOSITORY_ROOT / "data" / "emergent_misalignment_runs"))
    arguments = parser.parse_args(argv)
    build(arguments.finished_chain_run_folder, arguments.output_root)


if __name__ == "__main__":
    main()
