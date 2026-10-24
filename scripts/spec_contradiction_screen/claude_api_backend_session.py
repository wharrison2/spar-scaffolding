"""Starts the pipeline's Claude `api` backend for a validation stage, and summarises its cost log.

Pipeline side (it imports the pipeline's `shared`), so it must not be imported into a process that has
already imported this project's scripts/shared (the OpenAI stages). Import it lazily, inside functions.
"""

import json
import subprocess
import sys
from pathlib import Path

import spec_screen_common as common

CLAUDE_MODEL = "claude-sonnet-5"


def start_claude_api_session(stage_directory: Path, stamp: str, model: str = CLAUDE_MODEL):
    """Select the pipeline's api backend through a COPY of config.yaml saved in the stage folder (the pipeline's
    own config.yaml is untouched), read ANTHROPIC_API_KEY from the project's .env only, and point the
    pipeline's cost log at a file in the stage folder. Returns (api module, cost log path). Sampling
    parameters are left to the pipeline, which sends none to the Claude 5 family (the API rejects them)."""
    import yaml
    sys.path.append(str(common.PIPELINE_ROOT))
    with common.without_implicit_dotenv_loading():
        from shared import api
    common.activate_project_anthropic_api_key()
    configuration = yaml.safe_load((common.PIPELINE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    configuration["backend"] = "api"
    configuration["model"] = model
    configuration_path = Path(stage_directory) / f"pipeline_configuration_api_backend_{stamp}.yaml"
    configuration_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    cost_log_path = Path(stage_directory) / f"pipeline_cost_log_real_dollars_{stamp}.jsonl"
    api.init(str(configuration_path), cost_log_path)
    return api, cost_log_path


def start_openai_api_session(stage_directory: Path, stamp: str):
    """The OpenAI counterpart of start_claude_api_session: read OPENAI_API_KEY from the project's .env only and
    point the pipeline's cost log at a file in the stage folder. The pipeline's `shared.api` must still be
    initialised for that log; its config copy names the `claude_code` backend, which needs no Anthropic key and
    is never called (every call goes through shared.openai_api, which logs backend `openai`).
    Returns (cost log path)."""
    import yaml
    sys.path.append(str(common.PIPELINE_ROOT))
    with common.without_implicit_dotenv_loading():
        from shared import api
    common.activate_project_openai_api_key()
    configuration = yaml.safe_load((common.PIPELINE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    configuration["backend"] = "claude_code"
    configuration_path = Path(stage_directory) / f"pipeline_configuration_for_cost_log_only_{stamp}.yaml"
    configuration_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    cost_log_path = Path(stage_directory) / f"pipeline_cost_log_real_dollars_{stamp}.jsonl"
    api.init(str(configuration_path), cost_log_path)
    return cost_log_path


def pipeline_commit() -> str:
    return subprocess.run(["git", "-C", str(common.PIPELINE_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def summarize_pipeline_cost_log(cost_log_path: Path) -> dict:
    """Calls, tokens (input not served from cache, cache write, cache read, output, reasoning (OpenAI only; included in
    output)), cache rate and real dollars, overall and per pipeline stage tag."""
    def empty():
        return {"calls": 0, "input_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0,
                "reasoning_tokens": 0, "cost_dollars": 0.0, "cache_rate": 0.0}
    overall, per_stage = empty(), {}
    if Path(cost_log_path).is_file():
        for line in Path(cost_log_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for totals in (overall, per_stage.setdefault(record.get("stage", "unknown"), empty())):
                totals["calls"] += 1
                # OpenAI records count cached tokens inside input_tokens; Claude records keep them apart.
                cached = record.get("cache_read_tokens", 0) if record.get("backend") == "openai" else 0
                totals["input_tokens"] += record["input_tokens"] - cached
                totals["reasoning_tokens"] += record.get("reasoning_tokens", 0)
                totals["cache_write_tokens"] += record.get("cache_creation_tokens", 0)
                totals["cache_read_tokens"] += record.get("cache_read_tokens", 0)
                totals["output_tokens"] += record["output_tokens"]
                totals["cost_dollars"] += record["cost_usd"]
    for totals in [overall, *per_stage.values()]:
        all_input = totals["input_tokens"] + totals["cache_write_tokens"] + totals["cache_read_tokens"]
        totals["cache_rate"] = totals["cache_read_tokens"] / all_input if all_input else 0.0
    return {"overall": overall, "per_stage": per_stage}


def write_usage_and_cost_summary(cost_log_path: Path, summary_path: Path, note: str) -> dict:
    summary = summarize_pipeline_cost_log(cost_log_path)
    Path(summary_path).write_text(json.dumps({"note": note, **summary}, indent=2), encoding="utf-8")
    return summary


def describe_cost_so_far(cost_log_path: Path) -> str:
    totals = summarize_pipeline_cost_log(cost_log_path)["overall"]
    return (f"cost so far ${totals['cost_dollars']:.2f} over {totals['calls']} calls ({totals['input_tokens']} input, "
            f"{totals['output_tokens']} output tokens, cache rate {totals['cache_rate']:.0%})")
