#!/usr/bin/env python3
"""Set up an SDF pipeline run that starts at layer 2 from stored layer 1 plans plus saved consideration scores.

No API calls. It:
  1. creates <experiment folder>/pipeline_outputs/sdf/runs/<timestamp>_<label>/ with the pipeline's own
     create_run_dir (snapshotting the pipeline checkout's CURRENT prompts and constitution),
  2. copies the source run's layer 1 files (plans, description rewrites, screen files) except the old
     considerations outputs,
  3. copies the saved consideration scores and runs the CURRENT selector, which finds every document already
     scored (so makes no calls) and writes the assignment (consideration_selections.jsonl, coverage) with the
     current one-objective code and the config's limits.
Afterwards launch:  PIPELINE_OUTPUT_ROOT=<experiment>/pipeline_outputs python -m sdf_pipeline.run --config <config>
                    --resume --run-id <run folder name> --layer 2
"""
import argparse, shutil, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline-checkout", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--source-run-dir", type=Path, required=True)
parser.add_argument("--scores-file", type=Path, required=True)
parser.add_argument("--experiment-dir", type=Path, required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()

sys.path.insert(0, str(args.pipeline_checkout))
from shared import utils, api
from sdf_pipeline import layer1_description_rewrite, layer1_select_considerations as selector

config = utils.load_config(str(args.config))
runs_root = args.experiment_dir / "pipeline_outputs" / "sdf" / "runs"
run_dir = utils.create_run_dir(runs_root, label=args.label, config=config, snapshot_dirs={
    "prompts": args.pipeline_checkout / "prompts" / "sdf",
    "constitution": args.pipeline_checkout / "constitution"})
source_layer1 = utils.sdf_stage_dir(args.source_run_dir, 1)
layer1 = utils.sdf_stage_dir(run_dir, 1)
utils.ensure_dir(layer1)
for path in source_layer1.iterdir():
    if path.name.startswith(("consideration_", "_select_considerations")):
        continue
    shutil.copy2(path, layer1 / path.name)
shutil.copy2(args.scores_file, layer1 / selector.SCORES_FILENAME if hasattr(selector, "SCORES_FILENAME") else layer1 / "consideration_scores.jsonl")

plans = layer1_description_rewrite.load_rewritten_plans(config, layer1, utils.load_jsonl(layer1 / "plans.jsonl"))
api.init(str(args.config), cost_log_path=run_dir / "cost_log.jsonl")
selector.run(config, run_dir / "inputs" / "prompts", layer1, plans)
print(f"API cost of setup (should be 0): ${api.get_total_cost():.4f}")
print(f"RUN FOLDER NAME: {run_dir.name}")
