"""Does Claude catch implausible descriptions when asked ONLY the dedicated plausibility question, one at a time?

Runs the exact prompt GPT 6.1 Sol used for its gold labels (copied unchanged into
one_description_plausibility_judgment_prompt_copied_from_gpt_gold_labelling.txt) on each of the 97 fresh
descriptions through the pipeline's `call_claude` on the `api` backend, with a JSON schema mirroring GPT's labels
(one_description_plausibility_judgment_answer_schema.json). Sampling parameters are left to the pipeline (it sends
none to the Claude 5 family; the API rejects them). Then compares with GPT's labels.

Outputs go to <fresh run folder>/4_claude_plausibility_judgments_<model>/ . Resumable per description.
Key: ANTHROPIC_API_KEY from the project .env only (refuses if it is in the OS environment).

Run with the pipeline's virtual environment ($PY = animal-welfare-data-pipeline/.venv/bin/python):
    $PY scripts/spec_contradiction_screen/judge_description_plausibility_with_claude.py                  # dry run
    $PY scripts/spec_contradiction_screen/judge_description_plausibility_with_claude.py --run-paid-calls
    $PY scripts/spec_contradiction_screen/judge_description_plausibility_with_claude.py --compare-only   # free
    add --model claude-opus-5 for Opus
"""

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import spec_screen_common as common

PROMPT_PATH = common.SCREEN_SCRIPTS_FOLDER / "one_description_plausibility_judgment_prompt_copied_from_gpt_gold_labelling.txt"
ANSWER_SCHEMA_PATH = common.SCREEN_SCRIPTS_FOLDER / "one_description_plausibility_judgment_answer_schema.json"
GPT_LABELS_RELATIVE_PATH = "1_gold_labels/gold_labels_by_gpt_6_1_sol_2026-10-22_00-48.jsonl"
DEFAULT_MODEL = "claude-sonnet-5"
MAXIMUM_OUTPUT_TOKENS = 4000
# Dollars per million tokens (input, output).
MODEL_PRICES_PER_MILLION_TOKENS = {"claude-sonnet-5": (2.0, 10.0), "claude-opus-5": (5.0, 25.0)}
# Descriptions whose only GPT-labelled problem is naming Anthropic while forbidding real organisations; the pipeline
# now permits naming the provider of a named model.
ANTHROPIC_NAMING_ONLY_PROMPT_IDS = ["matrix_000028", "matrix_000042", "matrix_000044",
                                    "matrix_000081", "matrix_000085", "matrix_000096"]
CHARACTERS_PER_TOKEN_ESTIMATE = 3.6
ALLOWED_PROBLEM_TYPES = json.loads(ANSWER_SCHEMA_PATH.read_text(encoding="utf-8"))["properties"]["problem_type"]["enum"]


def judgments_folder_for_model(run_folder: Path, model: str) -> Path:
    folder = Path(run_folder) / f"4_claude_plausibility_judgments_{model}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def judgments_base_name(model: str) -> str:
    return f"claude_plausibility_judgments_{model}"


def render_judgment_prompt(description: str) -> tuple[str, str]:
    """(system text, user text) with the description placed in the prompt's <specification> block."""
    system_text, user_template = common.read_prompt_sections(PROMPT_PATH)
    if "{description}" not in user_template:
        raise ValueError("The prompt file has no {description} placeholder.")
    return system_text, user_template.replace("{description}", description)


def parse_judgment_answer(raw_answer: str) -> dict:
    """Validated judgment dict; raises ValueError for anything that is not a well-formed answer."""
    answer = json.loads(raw_answer)
    if not isinstance(answer, dict):
        raise ValueError("answer is not a JSON object")
    missing = [key for key in ("reasoning", "incoherent", "problem_type", "reason") if key not in answer]
    if missing:
        raise ValueError(f"answer lacks {missing}")
    if not isinstance(answer["incoherent"], bool):
        raise ValueError("incoherent is not a boolean")
    if answer["problem_type"] not in ALLOWED_PROBLEM_TYPES:
        raise ValueError(f"unknown problem_type {answer['problem_type']!r}")
    if not isinstance(answer["reasoning"], str) or not isinstance(answer["reason"], str):
        raise ValueError("reasoning and reason must be strings")
    return {key: answer[key] for key in ("reasoning", "incoherent", "problem_type", "reason")}


def read_finished_prompt_ids(results_path: Path) -> set[str]:
    if not Path(results_path).is_file():
        return set()
    return {record["prompt_id"] for record in common.read_jsonl_file(results_path)}


def append_line(path: Path, record: dict, lock: threading.Lock):
    with lock, open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_judgments(descriptions: list[dict], results_path: Path, errors_path: Path, progress_log_path: Path,
                  judge_function, workers: int, model: str, cost_log_path: Path | None = None) -> dict:
    """`judge_function(description, prompt_id)` returns the raw JSON answer text. A valid answer is appended to
    results_path; a failure goes to errors_path and is retried on rerun."""
    timed_progress = common.load_project_shared_module("timed_progress_reporter")
    done = read_finished_prompt_ids(results_path)
    pending = [record for record in descriptions if record["prompt_id"] not in done]
    print(f"{len(descriptions)} descriptions, {len(descriptions) - len(pending)} already judged, {len(pending)} to judge "
          f"({workers} workers)", flush=True)
    lock = threading.Lock()
    failures = []

    def cost_so_far():
        if cost_log_path is None:
            return "cost unknown"
        import claude_api_backend_session as session
        return session.describe_cost_so_far(cost_log_path)

    reporter = timed_progress.TimedProgressReporter(
        progress_log_path, len(pending), already_done_count=len(descriptions) - len(pending),
        status_suffix_function=cost_so_far)

    def judge_one(record):
        try:
            judgment = parse_judgment_answer(judge_function(record["description"], record["prompt_id"]))
        except Exception as error:  # noqa: BLE001  one failing call must not discard its siblings
            append_line(errors_path, {"prompt_id": record["prompt_id"], "error": f"{type(error).__name__}: {error}",
                                      "logged_at": datetime.now().isoformat(timespec="seconds")}, lock)
            failures.append(record["prompt_id"])
            reporter.note_task_finished(failed=True)
            return
        append_line(results_path, {"prompt_id": record["prompt_id"], **judgment, "model": model,
                                   "completed_at": datetime.now().isoformat(timespec="seconds")}, lock)
        reporter.note_task_finished(failed=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(judge_one, pending))
    reporter.finish()
    return {"ran": len(pending) - len(failures), "failed": failures}


def build_claude_judge_function(stage_directory: Path, stamp: str, model: str):
    import claude_api_backend_session as session
    api, cost_log_path = session.start_claude_api_session(stage_directory, stamp, model)
    system_text, _ = common.read_prompt_sections(PROMPT_PATH)
    schema = json.loads(ANSWER_SCHEMA_PATH.read_text(encoding="utf-8"))

    def judge_function(description, prompt_id):
        _, user_text = render_judgment_prompt(description)
        raw_answer, stop_reason = api.call_claude(
            user_text, system_prompt=system_text, model=model, max_tokens=MAXIMUM_OUTPUT_TOKENS,
            stage="description_plausibility_judgment", item_id=prompt_id, return_stop_reason=True,
            json_schema=json.loads(json.dumps(schema)))
        if stop_reason != "end_turn":
            raise ValueError(f"truncated or refused answer (stop_reason={stop_reason})")
        return raw_answer

    judge_function.cost_log_path = cost_log_path
    return judge_function


# ---------------------------------------------------------------- dry run cost estimate

def estimate_cost_for_models(descriptions: list[dict], calls_to_make: int, measured_cost_log_paths: list[Path],
                             gpt_labels: list[dict]) -> list[str]:
    """Lines describing the estimated cost per model. Tokens per call: measured from an earlier cost log of this
    stage when one exists, otherwise input estimated from prompt characters and output taken from GPT's mean
    output tokens on the same task (a proxy)."""
    import claude_api_backend_session as session
    measured = {"calls": 0, "input": 0, "output": 0}
    for path in measured_cost_log_paths:
        stage = session.summarize_pipeline_cost_log(path)["per_stage"].get("description_plausibility_judgment")
        if stage:
            measured["calls"] += stage["calls"]
            measured["input"] += stage["input_tokens"] + stage["cache_write_tokens"] + stage["cache_read_tokens"]
            measured["output"] += stage["output_tokens"]
    if measured["calls"]:
        input_per_call, output_per_call = measured["input"] / measured["calls"], measured["output"] / measured["calls"]
        basis = f"measured over {measured['calls']} earlier calls of this stage"
    else:
        system_text, user_template = common.read_prompt_sections(PROMPT_PATH)
        mean_description_characters = sum(len(d["description"]) for d in descriptions) / max(len(descriptions), 1)
        input_per_call = (len(system_text) + len(user_template) + mean_description_characters) / CHARACTERS_PER_TOKEN_ESTIMATE
        gpt_outputs = [r["usage"]["output_tokens"] for r in gpt_labels if r.get("usage")]
        output_per_call = sum(gpt_outputs) / len(gpt_outputs) if gpt_outputs else 500.0
        basis = (f"no measurement yet: input from prompt characters / {CHARACTERS_PER_TOKEN_ESTIMATE}, output from GPT 6.1 Sol's "
                 f"mean output tokens on the same task (proxy)")
    lines = [f"Tokens per call {input_per_call:.0f} input, {output_per_call:.0f} output ({basis})."]
    for model, (input_price, output_price) in MODEL_PRICES_PER_MILLION_TOKENS.items():
        total = calls_to_make * (input_per_call * input_price + output_per_call * output_price) / 1e6
        lines.append(f"  {model} (${input_price:g}/${output_price:g} per million tokens): {calls_to_make} calls, about ${total:.2f}")
    return lines


# ---------------------------------------------------------------- comparison with GPT

def build_comparison(claude_judgments: list[dict], gpt_labels: list[dict]) -> dict:
    """Join by prompt_id (only descriptions both judged). Groups: the Anthropic-naming-only rows are kept apart."""
    claude_by_id = {r["prompt_id"]: r for r in claude_judgments}
    gpt_by_id = {r["prompt_id"]: r for r in gpt_labels}
    shared_ids = sorted(set(claude_by_id) & set(gpt_by_id))
    rows = []
    for prompt_id in shared_ids:
        rows.append({"prompt_id": prompt_id, "gpt_incoherent": bool(gpt_by_id[prompt_id]["incoherent"]),
                     "claude_incoherent": bool(claude_by_id[prompt_id]["incoherent"]),
                     "gpt_problem_type": gpt_by_id[prompt_id]["problem_type"],
                     "claude_problem_type": claude_by_id[prompt_id]["problem_type"],
                     "gpt_reason": gpt_by_id[prompt_id]["reason"], "claude_reason": claude_by_id[prompt_id]["reason"],
                     "gpt_reasoning": gpt_by_id[prompt_id]["reasoning"],
                     "claude_reasoning": claude_by_id[prompt_id]["reasoning"],
                     "anthropic_naming_only": prompt_id in ANTHROPIC_NAMING_ONLY_PROMPT_IDS})

    def table(subset):
        counts = {(g, c): 0 for g in (True, False) for c in (True, False)}
        for row in subset:
            counts[(row["gpt_incoherent"], row["claude_incoherent"])] += 1
        total = len(subset)
        agree = counts[(True, True)] + counts[(False, False)]
        return {"both_flag": counts[(True, True)], "gpt_flags_claude_clean": counts[(True, False)],
                "claude_flags_gpt_clean": counts[(False, True)], "both_clean": counts[(False, False)],
                "total": total, "agreement_rate": agree / total if total else float("nan")}

    without_six = [r for r in rows if not r["anthropic_naming_only"]]
    only_six = [r for r in rows if r["anthropic_naming_only"]]
    return {"rows": rows, "all_descriptions": table(rows), "without_anthropic_naming_only": table(without_six),
            "anthropic_naming_only": table(only_six),
            "disagreements": [r for r in rows if r["gpt_incoherent"] != r["claude_incoherent"]]}


def format_table(name: str, table: dict) -> list[str]:
    return [f"{name} (n={table['total']}), agreement {table['agreement_rate']:.1%}",
            "                          Claude incoherent   Claude coherent",
            f"  GPT incoherent          {table['both_flag']:>17}   {table['gpt_flags_claude_clean']:>15}",
            f"  GPT coherent            {table['claude_flags_gpt_clean']:>17}   {table['both_clean']:>15}"]


def format_comparison_report(comparison: dict, model: str) -> str:
    lines = [f"Claude ({model}) versus GPT 6.1 Sol on whether each description is incoherent, one description at a time, same prompt.", ""]
    lines += format_table("All descriptions", comparison["all_descriptions"]) + [""]
    lines += format_table("Without the six that only name Anthropic while forbidding real organisations",
                          comparison["without_anthropic_naming_only"]) + [""]
    lines += format_table("The six that only name Anthropic while forbidding real organisations",
                          comparison["anthropic_naming_only"]) + ["", "Disagreements (GPT versus Claude):", ""]
    for row in comparison["disagreements"]:
        label = " [Anthropic-naming-only row]" if row["anthropic_naming_only"] else ""
        lines += [f"{row['prompt_id']}{label}: GPT {'incoherent' if row['gpt_incoherent'] else 'coherent'} "
                  f"({row['gpt_problem_type']}), Claude {'incoherent' if row['claude_incoherent'] else 'coherent'} "
                  f"({row['claude_problem_type']})",
                  f"  GPT reason: {row['gpt_reason'] or '(none)'}",
                  f"  GPT reasoning: {row['gpt_reasoning']}",
                  f"  Claude reason: {row['claude_reason'] or '(none)'}",
                  f"  Claude reasoning: {row['claude_reasoning']}", ""]
    return "\n".join(lines)


def write_comparison_files(comparison: dict, stage_directory: Path, model: str) -> Path:
    stamp = common.timestamp_now()
    report_path = stage_directory / f"comparison_with_gpt_labels_{model}_{stamp}.txt"
    report_path.write_text(format_comparison_report(comparison, model), encoding="utf-8")
    csv_path = stage_directory / f"per_description_agreement_with_gpt_labels_{model}_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison["rows"][0].keys()) if comparison["rows"] else ["prompt_id"])
        writer.writeheader()
        writer.writerows(comparison["rows"])
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", default=str(common.FRESH_SPECIFICATIONS_RUN_FOLDER))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-paid-calls", action="store_true", help="required to make any model call")
    parser.add_argument("--compare-only", action="store_true", help="only compare saved judgments with GPT's labels (free)")
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    model = arguments.model
    gpt_labels = common.read_jsonl_file(run_folder / GPT_LABELS_RELATIVE_PATH)
    stage_directory = judgments_folder_for_model(run_folder, model)
    results_path = common.find_or_create_stamped_file(stage_directory, judgments_base_name(model), "jsonl")

    if arguments.compare_only:
        if not results_path.is_file():
            raise SystemExit(f"No saved judgments at {results_path}")
        comparison = build_comparison(common.read_jsonl_file(results_path), gpt_labels)
        report_path = write_comparison_files(comparison, stage_directory, model)
        print(report_path.read_text(encoding="utf-8"))
        print(f"Written to {report_path}")
        return

    plans_path = run_folder / common.STAGE_SUBFOLDER_NAMES["fresh_specifications"] / "plans.jsonl"
    descriptions = common.load_real_specifications(plans_path)
    pending_count = len(descriptions) - len(read_finished_prompt_ids(results_path))
    print(f"Plausibility judgments: {len(descriptions)} descriptions from {plans_path}, {pending_count} still to judge, "
          f"model {model} on the api backend.")
    if not arguments.run_paid_calls:
        earlier_cost_logs = sorted(Path(run_folder).glob("4_claude_plausibility_judgments_*/pipeline_cost_log_real_dollars_*.jsonl"))
        print("\n".join(estimate_cost_for_models(descriptions, pending_count, earlier_cost_logs, gpt_labels)))
        print("DRY RUN: no calls made. Pass --run-paid-calls to run through the Claude api (billed to the project key).")
        return

    import claude_api_backend_session as session
    common.record_stage_configuration(
        run_folder, f"claude_plausibility_judgments_{model}", [PROMPT_PATH, ANSWER_SCHEMA_PATH],
        {"backend": "api", "model": model, "pipeline_commit": session.pipeline_commit()})
    stamp = results_path.stem[-len("2026-10-22_00-48"):]
    judge_function = build_claude_judge_function(stage_directory, stamp, model)
    summary = run_judgments(descriptions, results_path, stage_directory / f"claude_plausibility_judgment_errors_{stamp}.jsonl",
                            stage_directory / f"claude_plausibility_judgment_progress_{stamp}.log", judge_function,
                            arguments.workers, model, cost_log_path=judge_function.cost_log_path)
    totals = session.write_usage_and_cost_summary(
        judge_function.cost_log_path, stage_directory / f"claude_plausibility_judgment_usage_and_cost_{stamp}.json",
        "Real dollars from the pipeline cost log; reasoning tokens are 0 because no extended thinking is requested.")
    print(f"Done: {summary['ran']} judged, {len(summary['failed'])} failed (rerun to retry); "
          f"cost ${totals['overall']['cost_dollars']:.2f}")
    if not summary["failed"]:
        comparison = build_comparison(common.read_jsonl_file(results_path), gpt_labels)
        print(write_comparison_files(comparison, stage_directory, model).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
