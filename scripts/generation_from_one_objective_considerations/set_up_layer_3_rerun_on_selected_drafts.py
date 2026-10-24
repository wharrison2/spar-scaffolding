#!/usr/bin/env python3
"""Set up an SDF pipeline run that starts at layer 3 from selected layer 2 drafts of an earlier run. No API calls.

Creates <experiment>/pipeline_outputs/sdf/runs/<timestamp>_<label>/ with the pipeline checkout's CURRENT prompts,
copies the source run's layer 1 folder, and writes layer2/drafts.jsonl holding only the selected doc_ids.
Afterwards: PIPELINE_OUTPUT_ROOT=<experiment>/pipeline_outputs python -m sdf_pipeline.run --config <config>
            --resume --run-id <run folder name> --layer 3
"""
import argparse, json, shutil, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline-checkout", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--source-run-dir", type=Path, required=True)
parser.add_argument("--experiment-dir", type=Path, required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--doc-ids", nargs="+", required=True)
args = parser.parse_args()

sys.path.insert(0, str(args.pipeline_checkout))
from shared import utils

config = utils.load_config(str(args.config))
run_dir = utils.create_run_dir(args.experiment_dir / "pipeline_outputs" / "sdf" / "runs", label=args.label, config=config,
                               snapshot_dirs={"prompts": args.pipeline_checkout / "prompts" / "sdf",
                                              "constitution": args.pipeline_checkout / "constitution"})
shutil.copytree(utils.sdf_stage_dir(args.source_run_dir, 1), utils.sdf_stage_dir(run_dir, 1), dirs_exist_ok=True)
drafts = [d for d in utils.load_jsonl(utils.sdf_stage_dir(args.source_run_dir, 2) / "drafts.jsonl") if d.get("doc_id") in set(args.doc_ids)]
missing = set(args.doc_ids) - {d["doc_id"] for d in drafts}
if missing:
    sys.exit(f"drafts not found for {sorted(missing)}")
layer2 = utils.sdf_stage_dir(run_dir, 2)
utils.ensure_dir(layer2)
with open(layer2 / "drafts.jsonl", "w", encoding="utf-8") as handle:
    for draft in drafts:
        handle.write(json.dumps(draft, ensure_ascii=False) + "\n")
print(f"{len(drafts)} drafts copied")
print(f"RUN FOLDER NAME: {run_dir.name}")
