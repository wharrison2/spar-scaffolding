"""Gemma-versus-Haiku agreement on the SAGE-Eval verdicts of the five thinking-off Gemma answer sets (standard library only, no model call), and the judge gate.

    python3 .../compare_gemma_and_haiku_sage_eval_verdicts_on_five_gemma_answer_sets_and_decide_the_judge_gate.py --run-folder <combined evaluation run folder>

Reads, in <run folder>/generation_outputs/, the answers' verdicts of both judges (sage_eval_gemma_judge_verdicts__<entry>.jsonl and
sage_eval_haiku_judge_verdicts__<entry>.jsonl for every entry of the run configuration's model_order) and the prompts file of the run configuration (domain of each prompt).
For the fields "raised the safety fact" and "refused", per answer set and pooled over all sets, each on all prompts and excluding the Animal domain: Cohen's kappa,
percent agreement, the four cell counts (Haiku first, Gemma second) and the unparseable and not judged counts of each judge. For every set after the first (the base model),
each judge's pass rate (raised the safety fact) difference from the base model on the prompts both entries have a readable verdict for, excluding Animal, with the exact
McNemar p value, and whether both judges reach the same continue (p <= 0.1) or stop decision. Gate: Gemma is accepted when the pooled kappa for "raised the safety fact",
excluding Animal, is at least --minimum-kappa (0.6). Writes gemma_versus_haiku_sage_eval_judge_agreement_<timestamp>.json, a markdown table and
gemma_sage_eval_judge_gate_decision_<timestamp>.json into <run folder>/analysis_gemma_versus_haiku_sage_eval_judge_<timestamp>/. Exit code 0 whatever the decision.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.shared.cohens_kappa_and_exact_mcnemar_test_for_paired_boolean_verdicts import (  # noqa: E402
    exact_mcnemar_two_sided_p_value, percent_agreement_and_cohens_kappa, two_by_two_counts)
from scripts.shared.resumable_jsonl_records import read_jsonl  # noqa: E402

RAISED_FIELD = "raised_hazard_or_gave_safer_alternative"
REFUSED_FIELD = "refused_or_failed_to_help"
FIELD_LABELS = {RAISED_FIELD: "raised the safety fact", REFUSED_FIELD: "refused"}
ANIMAL_CATEGORY = "Animal"
CONTINUE_AT_OR_BELOW_P_VALUE = 0.1
DEFAULT_MINIMUM_KAPPA = 0.6
SCOPES = {"all_prompts": lambda is_animal: True, "excluding_animal_domain": lambda is_animal: not is_animal}


def read_verdicts_by_prompt(run_folder, judge_name, entry_name):
    """{prompt_gid: record} of one judge's verdict file for one answer set (empty when the file is absent); the verdict records are joined to prompts through the answers file."""
    outputs = Path(run_folder) / "generation_outputs"
    path = outputs / f"sage_eval_{judge_name}_judge_verdicts__{entry_name}.jsonl"
    prompt_by_answer = {record["answer_id"]: record["prompt_gid"] for record in read_jsonl(outputs / f"answers__{entry_name}__sage_eval.jsonl")}
    return {prompt_by_answer[record["answer_id"]]: record for record in read_jsonl(path)} if path.exists() else {}


def readable(record):
    return record is not None and record.get("parse_status") == "ok"


def agreement_for(entries, animal_by_prompt, verdicts, field, scope_name):
    """Counts, agreement and kappa over the (prompt, entry) answers both judges read in the given scope; unparseable and not judged counts per judge."""
    pairs, summary = [], {"haiku_unparseable": 0, "gemma_unparseable": 0, "gemma_not_judged": 0, "haiku_not_judged": 0}
    for entry in entries:
        for prompt_gid, is_animal in animal_by_prompt.items():
            if not SCOPES[scope_name](is_animal):
                continue
            haiku, gemma = verdicts["haiku"][entry].get(prompt_gid), verdicts["gemma"][entry].get(prompt_gid)
            for name, record in (("haiku", haiku), ("gemma", gemma)):
                if record is None:
                    summary[f"{name}_not_judged"] += 1
                elif not readable(record):
                    summary[f"{name}_unparseable"] += 1
            if readable(haiku) and readable(gemma):
                pairs.append((haiku[field], gemma[field]))
    counts = two_by_two_counts(pairs)
    percent, kappa = percent_agreement_and_cohens_kappa(counts)
    return {"answers_compared": len(pairs), "cells": {"haiku_true_gemma_true": counts["both_true"], "haiku_true_gemma_false": counts["first_true_second_false"],
                                                      "haiku_false_gemma_true": counts["first_false_second_true"], "haiku_false_gemma_false": counts["both_false"]},
            "percent_agreement": percent, "kappa": kappa, **summary}


def pass_rate_difference_from_base(base_entry, entry, animal_by_prompt, records_by_prompt):
    """One judge's pass rate difference (percentage points, entry minus base) and exact McNemar p value over the non-Animal prompts both entries have a readable verdict for."""
    only_entry_passed = only_base_passed = base_passes = entry_passes = compared = 0
    for prompt_gid, is_animal in animal_by_prompt.items():
        base_record, entry_record = records_by_prompt[base_entry].get(prompt_gid), records_by_prompt[entry].get(prompt_gid)
        if is_animal or not (readable(base_record) and readable(entry_record)):
            continue
        base_pass, entry_pass = base_record[RAISED_FIELD], entry_record[RAISED_FIELD]
        compared += 1
        base_passes += base_pass
        entry_passes += entry_pass
        only_entry_passed += entry_pass and not base_pass
        only_base_passed += base_pass and not entry_pass
    if not compared:
        return {"prompts_compared": 0, "difference_percentage_points": None, "p_value": None, "decision": None}
    p_value = exact_mcnemar_two_sided_p_value(only_entry_passed, only_base_passed)
    return {"prompts_compared": compared, "base_pass_rate_percent": 100 * base_passes / compared, "entry_pass_rate_percent": 100 * entry_passes / compared,
            "difference_percentage_points": 100 * (entry_passes - base_passes) / compared, "only_entry_passed": only_entry_passed, "only_base_passed": only_base_passed,
            "p_value": p_value, "decision": "continue" if p_value <= CONTINUE_AT_OR_BELOW_P_VALUE else "stop"}


def build_comparison(entries, animal_by_prompt, verdicts):
    comparison = {"agreement": {}, "pass_rate_difference_from_base": {}}
    for field in (RAISED_FIELD, REFUSED_FIELD):
        for scope_name in SCOPES:
            for label, chosen in [*((entry, [entry]) for entry in entries), ("pooled_over_all_sets", entries)]:
                comparison["agreement"].setdefault(field, {}).setdefault(scope_name, {})[label] = agreement_for(chosen, animal_by_prompt, verdicts, field, scope_name)
    for entry in entries[1:]:
        results = {judge: pass_rate_difference_from_base(entries[0], entry, animal_by_prompt, verdicts[judge]) for judge in ("haiku", "gemma")}
        results["both_judges_reach_the_same_decision"] = results["haiku"]["decision"] == results["gemma"]["decision"]
        comparison["pass_rate_difference_from_base"][entry] = results
    return comparison


def decide_gate(comparison, minimum_kappa):
    pooled = comparison["agreement"][RAISED_FIELD]["excluding_animal_domain"]["pooled_over_all_sets"]
    kappa = pooled["kappa"]
    accepted = kappa is not None and kappa >= minimum_kappa
    decisions = comparison["pass_rate_difference_from_base"]
    return {"gate": "gemma_as_the_on_pod_sage_eval_judge", "rule": f"accepted when the pooled Cohen's kappa for '{FIELD_LABELS[RAISED_FIELD]}', excluding the Animal domain, is at least {minimum_kappa}",
            "minimum_kappa": minimum_kappa, "pooled_kappa": kappa, "pooled_percent_agreement": pooled["percent_agreement"], "answers_compared": pooled["answers_compared"],
            "gemma_accepted_as_judge": accepted,
            "decision": "accept Gemma as the sweep's stopping judge" if accepted else "reject Gemma: the sweep runs every prompt",
            "sets_where_both_judges_reach_the_same_continue_or_stop_decision": [name for name, result in decisions.items() if result["both_judges_reach_the_same_decision"]],
            "sets_where_the_judges_disagree_on_the_decision": [name for name, result in decisions.items() if not result["both_judges_reach_the_same_decision"]],
            "decided_at": datetime.now().isoformat(timespec="seconds")}


def format_number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_markdown(comparison, entries, decision):
    lines = ["# Gemma versus Haiku SAGE-Eval verdicts on the five Gemma answer sets", "",
             f"Gate: {decision['rule']}. Pooled kappa {format_number(decision['pooled_kappa'])} on {decision['answers_compared']} answers: **{decision['decision']}**.", ""]
    for field in (RAISED_FIELD, REFUSED_FIELD):
        for scope_name in SCOPES:
            lines += [f"## {FIELD_LABELS[field]}, {scope_name.replace('_', ' ')}", "",
                      "| Answer set | Answers compared | Kappa | Agreement % | Haiku T / Gemma T | Haiku T / Gemma F | Haiku F / Gemma T | Haiku F / Gemma F | Unparseable (Haiku, Gemma) | Not judged (Haiku, Gemma) |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for label, result in comparison["agreement"][field][scope_name].items():
                cells = result["cells"]
                lines.append(f"| {label} | {result['answers_compared']} | {format_number(result['kappa'])} | {format_number(result['percent_agreement'], 1)} | "
                             f"{cells['haiku_true_gemma_true']} | {cells['haiku_true_gemma_false']} | {cells['haiku_false_gemma_true']} | {cells['haiku_false_gemma_false']} | "
                             f"{result['haiku_unparseable']}, {result['gemma_unparseable']} | {result['haiku_not_judged']}, {result['gemma_not_judged']} |")
            lines.append("")
    lines += [f"## Pass rate difference from {entries[0]}, excluding the Animal domain", "",
              "| Answer set | Judge | Prompts compared | Difference (percentage points) | Exact McNemar p | Decision at p <= 0.1 | Same decision |", "|---|---|---|---|---|---|---|"]
    for entry, results in comparison["pass_rate_difference_from_base"].items():
        for judge in ("haiku", "gemma"):
            result = results[judge]
            lines.append(f"| {entry} | {judge} | {result['prompts_compared']} | {format_number(result['difference_percentage_points'], 1)} | {format_number(result['p_value'], 4)} | "
                         f"{result['decision']} | {results['both_judges_reach_the_same_decision']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--minimum-kappa", type=float, default=DEFAULT_MINIMUM_KAPPA)
    arguments = parser.parse_args(argv)
    configuration = json.loads(sorted(arguments.run_folder.glob("run_configuration_*.json"))[-1].read_text(encoding="utf-8"))
    entries = configuration["model_order"]
    prompts_file = next(task for task in configuration["evaluation_tasks"] if task["task_name"] == "sage_eval")["prompts_file"]["path_inside_run_folder"]
    animal_by_prompt = {prompt["prompt_gid"]: prompt["category"] == ANIMAL_CATEGORY for prompt in read_jsonl(arguments.run_folder / prompts_file)}
    verdicts = {judge: {entry: read_verdicts_by_prompt(arguments.run_folder, judge, entry) for entry in entries} for judge in ("haiku", "gemma")}
    comparison = build_comparison(entries, animal_by_prompt, verdicts)
    decision = decide_gate(comparison, arguments.minimum_kappa)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = arguments.run_folder / f"analysis_gemma_versus_haiku_sage_eval_judge_{timestamp}"
    folder.mkdir(parents=True)
    (folder / f"gemma_versus_haiku_sage_eval_judge_agreement_{timestamp}.json").write_text(json.dumps({"answer_sets": entries, **comparison}, indent=2), encoding="utf-8")
    (folder / f"gemma_versus_haiku_sage_eval_judge_agreement_{timestamp}.md").write_text(build_markdown(comparison, entries, decision), encoding="utf-8")
    (folder / f"gemma_sage_eval_judge_gate_decision_{timestamp}.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
