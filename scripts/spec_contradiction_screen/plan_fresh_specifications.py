"""Stage 0: plan 100 fresh specifications with the pipeline's current code (deal rules, zero-weight
no-stake, stable IDs), through the pipeline's own layer-1 planner over the Claude api backend.

Only planning runs (no screen, no drafting): it calls `sdf_pipeline.layer1_plan.run(config, prompts_dir,
output_dir)` directly. It writes a COPY of the pipeline's config.yaml into the run folder with
backend api, model claude-sonnet-5, sdf.n_prompts 100, sdf.seed 1, sdf.screen_enabled false; the
pipeline's own config.yaml is not touched. layer1_plan checkpoints per plan, so rerunning with the same
--run-folder resumes. The prompts directory used is copied into the run folder and the pipeline git
commit recorded, so the specifications are reproducible.

Dry run by default: composes the 100 prompts offline (free), prints the deal repair report and the number
of prompts, makes no model call and writes nothing. --run-paid-calls plans them (Claude api, billed to the project ANTHROPIC_API_KEY read from emergent_alignment/.env;
refuses to run if that variable is already set in the OS environment).
Run with the pipeline's virtual environment:
    $PY scripts/spec_contradiction_screen/plan_fresh_specifications.py                      # dry run
    $PY scripts/spec_contradiction_screen/plan_fresh_specifications.py --run-name spec_contradiction_screen_validation --run-paid-calls
    $PY scripts/spec_contradiction_screen/plan_fresh_specifications.py --run-folder <folder> --run-paid-calls   # resume
"""

import argparse
import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import spec_screen_common as common

PLAN_MODEL = "claude-sonnet-5"
DEFAULT_PLAN_COUNT = 100
DEFAULT_SEED = 1


class TeeToFile(io.TextIOBase):
    """Writes to the terminal and to a log file (so the deal repair report is saved)."""

    def __init__(self, terminal, log_handle):
        self.terminal, self.log_handle = terminal, log_handle

    def write(self, text):
        self.terminal.write(text)
        self.log_handle.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_handle.flush()


def build_fresh_configuration(pipeline_configuration: dict, plan_count: int, seed: int, workers: int,
                              cost_log_path: Path) -> dict:
    configuration = json.loads(json.dumps(pipeline_configuration))  # deep copy
    configuration["backend"] = "api"
    configuration["model"] = PLAN_MODEL
    configuration["workers"] = workers
    configuration["outputs"] = dict(configuration.get("outputs", {}), cost_log=str(cost_log_path))
    configuration["sdf"].update({"n_prompts": plan_count, "seed": seed, "screen_enabled": False,
                                 "plan_model": PLAN_MODEL})
    return configuration


def compose_prompts_offline(configuration: dict, prompts_dir: Path) -> list[dict]:
    """The prompts layer1_plan.run would compose (same calls), without writing or calling anything."""
    sys.path.append(str(common.PIPELINE_ROOT))
    with common.without_implicit_dotenv_loading():
        from sdf_pipeline import compose_prompts as cp
        from shared import utils
    sdf = configuration["sdf"]
    template = utils.sdf_template_path(prompts_dir, 1).read_text(encoding="utf-8")
    values, weights, value_ids = cp.load_variables(prompts_dir / "variables.txt")
    preamble = (prompts_dir / "preamble.txt").read_text(encoding="utf-8")
    deal_rules = cp.load_deal_rules(prompts_dir / cp.DEAL_CONSTRAINTS_FILENAME, value_ids)
    return list(cp.compose_records(template, values, weights, preamble, n_prompts=sdf["n_prompts"],
                                   seed=sdf.get("seed", 0), entity_seed=sdf.get("entity_pool_seed", 137),
                                   deal_rules=deal_rules))


def git_output(*arguments) -> str:
    return subprocess.run(["git", "-C", str(common.PIPELINE_ROOT), *arguments], capture_output=True, text=True).stdout.strip()


def summarise_plans(plans_path: Path) -> dict:
    records = [json.loads(l) for l in plans_path.read_text(encoding="utf-8").splitlines() if l.strip()] if plans_path.is_file() else []
    return {"plans": len(records), "with_description": sum(bool(r.get("description")) for r in records),
            "incoherent": sum(bool(r.get("incoherent")) for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=common.DEFAULT_RUN_NAME)
    parser.add_argument("--run-folder", help="existing run folder to resume")
    parser.add_argument("--plan-count", type=int, default=DEFAULT_PLAN_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-paid-calls", action="store_true", help="required to make any model call")
    arguments = parser.parse_args()
    import yaml
    pipeline_prompts_dir = common.PIPELINE_ROOT / "prompts/sdf"
    pipeline_configuration = yaml.safe_load((common.PIPELINE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    placeholder_cost_log = Path("unused_in_dry_run.jsonl")
    configuration = build_fresh_configuration(pipeline_configuration, arguments.plan_count, arguments.seed,
                                              arguments.workers, placeholder_cost_log)
    print(f"Pipeline branch {git_output('branch', '--show-current')} at commit {git_output('rev-parse', 'HEAD')}")
    if not arguments.run_paid_calls:
        prompts = compose_prompts_offline(configuration, pipeline_prompts_dir)
        print(f"DRY RUN: composed {len(prompts)} planning prompts offline (above: deal repair report). "
              f"Would make {len(prompts)} planning calls with {PLAN_MODEL} over the Claude api. "
              "No call made, nothing written. Pass --run-paid-calls to plan.")
        return
    run_folder = common.resolve_run_folder(arguments.run_folder, arguments.run_name, may_create=True)
    stage_directory = common.stage_folder(run_folder, "fresh_specifications")
    stamp = common.timestamp_now()
    snapshot_directory = stage_directory / "pipeline_prompts_snapshot"
    if not snapshot_directory.exists():
        shutil.copytree(pipeline_prompts_dir, snapshot_directory)
    commit_record = stage_directory / f"pipeline_commit_and_working_tree_status_{stamp}.txt"
    if not list(stage_directory.glob("pipeline_commit_and_working_tree_status_*.txt")):
        commit_record.write_text(f"branch: {git_output('branch', '--show-current')}\ncommit: {git_output('rev-parse', 'HEAD')}\n"
                                 f"working tree status (empty means clean):\n{git_output('status', '--short')}\n", encoding="utf-8")
    cost_log_path = stage_directory / f"pipeline_cost_log_real_dollars_{stamp}.jsonl"
    configuration = build_fresh_configuration(pipeline_configuration, arguments.plan_count, arguments.seed,
                                              arguments.workers, cost_log_path)
    configuration_path = common.find_or_create_stamped_file(stage_directory, "pipeline_configuration_api_backend_fresh_planning", "yaml")
    if not configuration_path.is_file():
        configuration_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    else:  # resume: keep the original configuration and cost log
        configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
        cost_log_path = Path(configuration["outputs"]["cost_log"])
    sys.path.append(str(common.PIPELINE_ROOT))
    with common.without_implicit_dotenv_loading():
        from sdf_pipeline import layer1_plan
        from shared import api
    common.activate_project_anthropic_api_key()
    api.init(str(configuration_path), cost_log_path)
    console_log = stage_directory / f"fresh_planning_console_log_with_deal_repair_report_{stamp}.txt"
    with open(console_log, "a", encoding="utf-8") as handle, contextlib.redirect_stdout(TeeToFile(sys.stdout, handle)):
        layer1_plan.run(configuration, snapshot_directory, stage_directory)
    summary = summarise_plans(stage_directory / "plans.jsonl")
    print(f"Plans written: {summary['plans']}; with descriptions: {summary['with_description']}; "
          f"declared INCOHERENT: {summary['incoherent']}; requested {arguments.plan_count}. "
          "If fewer, rerun with the same --run-folder to retry the rest.")


if __name__ == "__main__":
    main()
