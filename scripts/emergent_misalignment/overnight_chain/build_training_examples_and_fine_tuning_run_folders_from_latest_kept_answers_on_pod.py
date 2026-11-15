"""On the pod, after the treatment arm answers are generated: builds the training examples from the newest kept answers file (at most one
answer per prompt, fixed seed, held-out prompts refused) and then two fine-tuning run folders, a smoke run on the first N examples and the
full run on all of them, by calling the existing builders. The run folders are moved to fixed names so that later steps need no path
discovery:  <fine tuning folder>/fine_tuning_smoke_run  and  <fine tuning folder>/fine_tuning_full_run.

    python3 .../build_training_examples_and_fine_tuning_run_folders_from_latest_kept_answers_on_pod.py \
        --generation-run-folder <treatment arm generation run folder> --held-out-set-file <jsonl> --output-folder <folder> \
        --seed 20261111 --smoke-run-examples 200 [--model-short-name gemma_4_31b_it]

With --training-examples-file (written by another model: user prompt and final answer, no system prompt) there are no kept answers: the file is
copied unchanged (same examples, same order, no new subsample) after a check that no example is on a held-out prompt, the run folders are built
with the loud override that records the examples were written by another model, and --generation-run-folder is not needed.

Resumable: an existing training examples file and existing run folders are reused, never rebuilt. Writes
<output folder>/training_examples_and_run_folders_record_<timestamp>.json.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[2] / "emergent_misalignment"
EXAMPLES_BUILDER = SCRIPTS_DIRECTORY / "treatment_arm_training_data" / "build_training_examples_from_kept_answers.py"
for folder in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "treatment_arm_training_data"):
    sys.path.insert(0, str(folder))
from fine_tuning_run_definitions import MODEL_MISMATCH_OVERRIDE_FLAG  # noqa: E402
from shared_answer_records_and_held_out_set_helpers import answer_is_on_a_held_out_prompt, load_held_out_set_identifiers, read_jsonl  # noqa: E402
RUN_FOLDER_BUILDER = SCRIPTS_DIRECTORY / "build_lora_fine_tuning_run_folder_for_self_distillation.py"
SMOKE_RUN_FOLDER_NAME = "fine_tuning_smoke_run"
FULL_RUN_FOLDER_NAME = "fine_tuning_full_run"


def run(command):
    print("running:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def newest(folder, pattern):
    matches = sorted(Path(folder).glob(pattern))
    return matches[-1] if matches else None


def copy_training_examples_from_another_model(source_file, held_out_set_file, examples_folder):
    """Copies the examples file unchanged into examples_folder after refusing any example on a held-out prompt; returns the copy."""
    examples = read_jsonl(source_file)
    held_out_ids, held_out_messages, _ = load_held_out_set_identifiers(held_out_set_file)
    offending = [example.get("answer_id") for example in examples if answer_is_on_a_held_out_prompt(
        {"prompt_gid": example.get("prompt_gid"), "user_message": example["messages"][0]["content"]}, held_out_ids, held_out_messages)]
    if offending:
        raise SystemExit(f"{len(offending)} training examples are on held-out set prompts, e.g. {offending[:3]}")
    examples_folder.mkdir(parents=True, exist_ok=True)
    destination = examples_folder / f"training_examples_from_another_model_{Path(source_file).stem}.jsonl"
    shutil.copyfile(source_file, destination)
    return destination


def build_one_run_folder(python_executable, examples_file, model_short_name, final_folder, extra_arguments):
    if final_folder.exists() and list(final_folder.glob("run_configuration_*.json")):
        print(f"{final_folder.name} already built; reused", flush=True)
        return
    staging = final_folder.parent / f"{final_folder.name}_builder_staging"
    run([python_executable, RUN_FOLDER_BUILDER, "--model-short-name", model_short_name, "--training-examples-file", examples_file,
         "--runs-root", staging, *extra_arguments])
    # The builder puts timing runs under small_sample_checks/<run folder>, so look for the folder holding the run configuration at any depth
    built = sorted({path.parent for path in staging.rglob("run_configuration_*.json")})
    if len(built) != 1:
        raise SystemExit(f"expected exactly one run folder (with a run_configuration file) under {staging}, found {len(built)}")
    if final_folder.exists():
        os.rename(final_folder, final_folder.parent / f"{final_folder.name}_incomplete_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.replace(built[0], final_folder)
    shutil.rmtree(staging)  # only empty folders are left (small_sample_checks/ for timing runs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generation-run-folder", type=Path, default=None, help="needed unless --training-examples-file is given")
    parser.add_argument("--training-examples-file", type=Path, default=None, help="training examples written by another model (see above)")
    parser.add_argument("--held-out-set-file", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke-run-examples", type=int, default=200)
    parser.add_argument("--model-short-name", default="gemma_4_31b_it")
    parser.add_argument("--python-executable", default=sys.executable)
    arguments = parser.parse_args(argv)
    kept_answers = None
    if arguments.training_examples_file is None:
        if arguments.generation_run_folder is None:
            raise SystemExit("give --generation-run-folder (kept answers) or --training-examples-file")
        kept_answers = newest(arguments.generation_run_folder / "generation_outputs" / "final_outputs", "kept_answers_*.jsonl")
        if kept_answers is None:
            raise SystemExit(f"no kept_answers_*.jsonl in {arguments.generation_run_folder}/generation_outputs/final_outputs")
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    examples_folder = arguments.output_folder / "training_examples"
    examples_file = newest(examples_folder, "training_examples_*.jsonl") if examples_folder.exists() else None
    if examples_file is None and arguments.training_examples_file is not None:
        examples_file = copy_training_examples_from_another_model(arguments.training_examples_file, arguments.held_out_set_file, examples_folder)
    elif examples_file is None:
        run([arguments.python_executable, EXAMPLES_BUILDER, "--kept-answers", kept_answers, "--description", "treatment_arm_one_answer_per_prompt",
             "--output-folder", examples_folder, "--seed", arguments.seed, "--maximum-answers-per-prompt", 1,
             "--held-out-set-file", arguments.held_out_set_file])
        examples_file = newest(examples_folder, "training_examples_*.jsonl")
    folders = arguments.output_folder / "fine_tuning_run_folders"
    folders.mkdir(exist_ok=True)
    override = [MODEL_MISMATCH_OVERRIDE_FLAG] if arguments.training_examples_file is not None else []
    build_one_run_folder(arguments.python_executable, examples_file, arguments.model_short_name, folders / SMOKE_RUN_FOLDER_NAME,
                         ["--timing-run-examples", arguments.smoke_run_examples, *override])
    build_one_run_folder(arguments.python_executable, examples_file, arguments.model_short_name, folders / FULL_RUN_FOLDER_NAME, override)
    record = {"kept_answers_file": str(kept_answers) if kept_answers else None, "training_examples_file": str(examples_file),
              "training_examples_written_by_another_model_file": str(arguments.training_examples_file) if arguments.training_examples_file else None,
              "seed": arguments.seed,
              "smoke_run_folder": str(folders / SMOKE_RUN_FOLDER_NAME), "full_run_folder": str(folders / FULL_RUN_FOLDER_NAME),
              "smoke_run_examples": arguments.smoke_run_examples, "built_at": datetime.now().isoformat(timespec="seconds")}
    (arguments.output_folder / f"training_examples_and_run_folders_record_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
