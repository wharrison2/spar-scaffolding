"""Launcher for the end-to-end pipeline smoke run (default: 20 prompts, `--label contradiction-fixes-smoke`).

Options: `--n-prompts` (default 20), `--label` (default contradiction-fixes-smoke), `--model` (sets the global `model` in
the run's config copy; default leaves the branch's config untouched; every Claude stage without its own override then
uses it, and the dry run prints which stage uses what; `sdf.description_rewrite_model` (GPT) is never changed),
`--pipeline-checkout` (default the considerations-selector worktree) and `--stop-at-cost-usd` (default none: a spending
ceiling, see below).

The smoke run goes through every layer of the SDF pipeline on the integrated branch: the composer, the implausibility
screen with redeal rounds, planning, the description rewrite (GPT), the considerations selector, drafting, the document
rewrite and the judge, with checkpoints and `--resume`. The stage tags in the cost log and the new rows of the audit and
report are checked afterwards (commands printed at the end).

Which checkout runs: the rebased selector worktree, `animal-welfare-data-pipeline-considerations-selector`
(branch william/sdf-considerations-selector), because it is the only checkout with the selector on top of the screen and
description rewrite.

Default is a DRY RUN: it prints what it would do, writes the run folder's config copy and writes the cost estimate (an
offline stubbed run, no model calls), then stops. Nothing is called until `--launch-paid-run` is passed, and that is for
the project owner to pass.

Keys: both come explicitly from this project's .env (`activate_project_api_key`, an explicit-path read), and the launcher
refuses if either is already in the OS environment. The pipeline modules are imported with `dotenv.load_dotenv` disabled
(`without_implicit_dotenv_loading`), because shared/api.py would otherwise search parent folders and find the same file
implicitly. In a dry run no key is read at all.

Outputs: everything goes to one run folder under
    data/spec_contradiction_screen_runs/small_sample_checks/end_to_end_pipeline_smoke_test_<date>_<time>/
holding the config copy, the launch record, and `pipeline_outputs/` (the pipeline's PIPELINE_OUTPUT_ROOT, so the run
itself is `pipeline_outputs/sdf/runs/<timestamp>_contradiction-fixes-smoke/`). The cost estimate goes to
    data/spec_contradiction_screen_runs/small_sample_checks/end_to_end_pipeline_smoke_cost_estimate_<date>_<time>.txt

Spending ceiling: with `--stop-at-cost-usd N` the paid run is a subprocess (this script re-run with an internal
`--run-pipeline-in-this-process` flag) and the launcher re-reads the run's own `cost_log.jsonl` (sum of `cost_usd`,
priced by the pipeline's tables) every 5 seconds. It stops the pipeline (terminate, wait 60 seconds, kill if needed)
when the total reaches N minus a margin of 8 times the most expensive single call seen so far (8 = the default
`workers`, the calls that can be in flight), prints the total and the resume command, and records the stop in the
launch record. After a resume the ceiling applies to the whole run's log, earlier spend included.

Resume after an interruption (the pipeline checkpoints each stage and resumes from the saved rows):
    ... run_end_to_end_pipeline_smoke_test.py --launch-paid-run --resume-run-folder <that run folder>

Run with the pipeline's virtual environment:
    animal-welfare-data-pipeline/.venv/bin/python scripts/spec_contradiction_screen/run_end_to_end_pipeline_smoke_test.py
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cost_ceiling_monitor
import spec_screen_common as common

PROJECT_ROOT = common.PROJECT_ROOT
INTEGRATED_CHECKOUT = PROJECT_ROOT / "animal-welfare-data-pipeline-considerations-selector"
SMOKE_RUN_LABEL = "contradiction-fixes-smoke"
SMOKE_N_PROMPTS = 20
CHILD_FLAG = "--run-pipeline-in-this-process"
SMOKE_RUNS_FOLDER = common.SMALL_SAMPLE_CHECKS_FOLDER
RUN_FOLDER_BASE_NAME = "end_to_end_pipeline_smoke_test"
COST_ESTIMATE_BASE_NAME = "end_to_end_pipeline_smoke_cost_estimate"
ESTIMATE_SCRIPT = common.SCREEN_SCRIPTS_FOLDER / "estimate_end_to_end_pipeline_smoke_run_cost.py"
KEY_NAMES_NEEDED = (common.ANTHROPIC_KEY_NAME, common.OPENAI_KEY_NAME)


def refuse_if_a_key_is_in_the_os_environment() -> None:
    present = [name for name in KEY_NAMES_NEEDED if os.environ.get(name)]
    if present:
        raise SystemExit(f"{', '.join(present)} already set in the OS environment; refusing to run so a key other than "
                         f"the project's cannot be used. Unset and rerun.")


def write_config_copy(checkout: Path, run_folder: Path, timestamp: str, n_prompts: int, model: str | None) -> Path:
    """The branch's config.yaml with the given n_prompts, backend api and (if given) the global model, everything else
    untouched (comments kept)."""
    text = (checkout / "config.yaml").read_text(encoding="utf-8")
    text, n_prompt_substitutions = re.subn(r"(?m)^(\s+n_prompts:\s*)\d+", rf"\g<1>{n_prompts}", text)
    text, backend_substitutions = re.subn(r"(?m)^backend:\s*\S+", "backend: api", text)
    if n_prompt_substitutions != 1 or backend_substitutions != 1:
        raise SystemExit(f"Expected exactly one sdf.n_prompts and one backend line in {checkout / 'config.yaml'}; "
                         f"found {n_prompt_substitutions} and {backend_substitutions}.")
    if model is not None:
        text, model_substitutions = re.subn(r"(?m)^(model:\s*)\S+", rf"\g<1>{model}", text)
        if model_substitutions != 1:
            raise SystemExit(f"Expected exactly one top-level model line in {checkout / 'config.yaml'}; "
                             f"found {model_substitutions}.")
    config_path = run_folder / f"config_for_smoke_run_{timestamp}.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def stage_models_report(config_path: Path) -> list[str]:
    """Which model each SDF stage will use under this config, following the pipeline's own fallbacks
    (screen and select: own model, else plan_model, else global; plan, draft, rewrite, score: own, else global)."""
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sdf = config.get("sdf", {})
    global_model = config["model"]

    def resolved(own_key: str, *fallback_keys: str) -> str:
        for key in (own_key, *fallback_keys):
            if sdf.get(key):
                return f"{sdf[key]}  (sdf.{key})"
        return f"{global_model}  (global model)"

    return [
        f"  layer1_screen                {resolved('screen_model', 'plan_model')}",
        f"  layer1_plan                  {resolved('plan_model')}",
        f"  layer1_description_rewrite   {sdf.get('description_rewrite_model') or resolved('plan_model')}  (sdf.description_rewrite_model, left as is)",
        f"  layer1_select                {resolved('select_model', 'plan_model')}",
        f"  layer2_draft                 {resolved('draft_model')}",
        f"  layer3_rewrite               {resolved('rewrite_model')}",
        f"  layer4_score                 {resolved('score_model')}",
    ]


def write_cost_estimate(checkout: Path, config_path: Path, timestamp: str) -> Path:
    report_path = SMOKE_RUNS_FOLDER / f"{COST_ESTIMATE_BASE_NAME}_{timestamp}.txt"
    environment_without_keys = {k: v for k, v in os.environ.items() if k not in KEY_NAMES_NEEDED}
    subprocess.run(
        [sys.executable, str(ESTIMATE_SCRIPT), "--pipeline-checkout", str(checkout), "--config", str(config_path),
         "--report-path", str(report_path)],
        check=True, env=environment_without_keys)
    return report_path


def command_text(arguments: list[str]) -> str:
    return " ".join(arguments)


def run_pipeline_in_this_process(checkout: Path, pipeline_outputs_root: Path, pipeline_arguments: list[str]) -> None:
    """Paid path: explicit keys, implicit dotenv search disabled, then the pipeline's own entry point."""
    common.activate_project_anthropic_api_key()
    common.activate_project_openai_api_key()
    pipeline_outputs_root.mkdir(exist_ok=True)
    os.environ["PIPELINE_OUTPUT_ROOT"] = str(pipeline_outputs_root)
    sys.path.insert(0, str(checkout))
    os.chdir(checkout)
    with common.without_implicit_dotenv_loading():
        from sdf_pipeline import run as pipeline_run
    sys.argv = ["run.py", *pipeline_arguments]
    pipeline_run.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--launch-paid-run", action="store_true",
                        help="Actually make the paid calls. Without it this is a dry run.")
    parser.add_argument("--resume-run-folder", type=Path, default=None,
                        help="A run folder from an earlier launch: resume its pipeline run from the checkpoints.")
    parser.add_argument("--pipeline-checkout", type=Path, default=INTEGRATED_CHECKOUT)
    parser.add_argument("--n-prompts", type=int, default=SMOKE_N_PROMPTS, help="sdf.n_prompts in the config copy.")
    parser.add_argument("--model", default=None,
                        help="Global `model` in the config copy (default: the branch's config untouched).")
    parser.add_argument("--label", default=SMOKE_RUN_LABEL, help="The pipeline run's --label.")
    parser.add_argument("--stop-at-cost-usd", type=float, default=None,
                        help="Spending ceiling in dollars (default none): stop the paid run, cleanly and resumably, "
                             "once the run's cost log reaches it minus a margin for calls in flight.")
    parser.add_argument(CHILD_FLAG, default=None, metavar="PIPELINE_ARGUMENTS_JSON", help=argparse.SUPPRESS)
    parser.add_argument("--child-pipeline-outputs-root", type=Path, default=None, help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    checkout = arguments.pipeline_checkout.resolve()
    refuse_if_a_key_is_in_the_os_environment()
    if arguments.run_pipeline_in_this_process is not None:
        run_pipeline_in_this_process(checkout, arguments.child_pipeline_outputs_root,
                                     json.loads(arguments.run_pipeline_in_this_process))
        return
    timestamp = common.timestamp_now()
    resuming = arguments.resume_run_folder is not None

    if resuming:
        run_folder = arguments.resume_run_folder.resolve()
        config_path = next(iter(sorted(run_folder.glob("config_for_smoke_run_*.yaml"))), None)
        if config_path is None:
            raise SystemExit(f"No config_for_smoke_run_*.yaml in {run_folder}; not a smoke run folder.")
    else:
        run_folder = SMOKE_RUNS_FOLDER / f"{RUN_FOLDER_BASE_NAME}_{timestamp}"
        run_folder.mkdir(parents=True, exist_ok=False)
        config_path = write_config_copy(checkout, run_folder, timestamp, arguments.n_prompts, arguments.model)
    pipeline_outputs_root = run_folder / "pipeline_outputs"

    pipeline_arguments = ["--config", str(config_path), "--label", arguments.label]
    if resuming:
        pipeline_arguments += ["--resume"]

    print("End-to-end pipeline smoke test")
    print(f"  mode:               {'PAID RUN' if arguments.launch_paid_run else 'dry run (no calls will be made)'}")
    print(f"  pipeline checkout:  {checkout}  (branch {subprocess.run(['git', '-C', str(checkout), 'branch', '--show-current'], capture_output=True, text=True).stdout.strip()})")
    print(f"  run folder:         {run_folder}")
    model_note = f"global model {arguments.model}" if arguments.model else "model as the branch's config.yaml"
    print(f"  config copy:        {config_path}  (sdf.n_prompts {arguments.n_prompts}, backend api, {model_note}, rest as the branch's config.yaml)")
    print(f"  pipeline outputs:   {pipeline_outputs_root}/sdf/runs/<timestamp>_{arguments.label}/")
    print(f"  keys, when paid:    {', '.join(KEY_NAMES_NEEDED)} from {common.ENV_FILE_PATH} only (explicit path; implicit dotenv search disabled)")
    print(f"  pipeline call:      sdf_pipeline.run.main() with arguments: {command_text(pipeline_arguments)}")
    ceiling = arguments.stop_at_cost_usd
    print(f"  spending ceiling:   {'none' if ceiling is None else f'${ceiling:.2f}, stop at the ceiling minus {cost_ceiling_monitor.IN_FLIGHT_CALLS_MARGIN} x the most expensive call seen'}")
    print("  model per stage (as the pipeline resolves it):")
    print("\n".join(stage_models_report(config_path)))

    if not arguments.launch_paid_run:
        if not resuming:
            report_path = write_cost_estimate(checkout, config_path, timestamp)
            print(f"\nCost estimate saved to {report_path}")
        print("\nDry run only. To make the paid calls, add --launch-paid-run.")
        return

    # --- paid path: the pipeline runs in a child process so this process can watch its cost log ---
    launch_record_path = run_folder / f"launch_record_{timestamp}.json"
    launch_record = {
        "launched_at": timestamp, "pipeline_checkout": str(checkout), "pipeline_arguments": pipeline_arguments,
        "resuming": resuming, "config_path": str(config_path), "stop_at_cost_usd": ceiling}
    launch_record_path.write_text(json.dumps(launch_record, indent=2), encoding="utf-8")

    child_command = [sys.executable, str(Path(__file__).resolve()), "--pipeline-checkout", str(checkout),
                     CHILD_FLAG, json.dumps(pipeline_arguments),
                     "--child-pipeline-outputs-root", str(pipeline_outputs_root)]
    process = subprocess.Popen(child_command)
    outcome = cost_ceiling_monitor.watch_pipeline_process(
        process, lambda: sorted(pipeline_outputs_root.glob("sdf/runs/*/cost_log.jsonl")), ceiling)
    launch_record["outcome"] = outcome
    launch_record_path.write_text(json.dumps(launch_record, indent=2), encoding="utf-8")

    if outcome["stopped_by_cost_ceiling"]:
        print(f"\nSTOPPED at the spending ceiling: cost was ${outcome['cost_when_stop_requested_usd']:.2f} when the stop was "
              f"requested (threshold ${outcome['stop_threshold_usd']:.2f} = ceiling ${ceiling:.2f} minus margin "
              f"${outcome['margin_usd']:.2f}); ${outcome['cost_after_process_ended_usd']:.2f} after the process ended "
              f"({outcome['process_ended_by']}). Resume with the same options plus:\n"
              f"  --launch-paid-run --resume-run-folder {run_folder}")
        raise SystemExit(2)
    if outcome["return_code"] != 0:
        raise SystemExit(f"The pipeline process exited with code {outcome['return_code']}. Resume with "
                         f"--launch-paid-run --resume-run-folder {run_folder}")

    print("\nAfter the run, check the cost-log stage tags and the report rows (offline commands, run from the checkout):")
    print(f"  python evals/audit_sdf.py <run folder under {pipeline_outputs_root}/sdf/runs/>")
    print(f"  python evals/report_sdf.py <same run folder>")


if __name__ == "__main__":
    main()
