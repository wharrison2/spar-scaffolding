"""Description rewrite validation, step 3 (offline, free): the measures and the before-and-after sheet.

Measures, over the 97 fresh descriptions (20 GPT-labelled problematic, 77 not):
  - share rewritten among the 20 and among the 77 (the 77 rewritten are unnecessary rewrites);
  - share of the 20 whose labelled problem is gone in the rewrite (an untouched or dropped one counts as not gone),
    overall and by the labelled problem type;
  - share of rewrites that keep every dealt value;
  - 10 before-and-after pairs for William to read (seeded; as many from the labelled 20 as exist, up to 5).

Reads and writes the subfolder of the model given by --rewrite-model (default gpt-6.1-sol):
<fresh run folder>/3_description_rewrite_validation_<model>/ (claude-sonnet-5 is the original 3_description_rewrite_validation/). Run (any Python; no pipeline needed):
    python scripts/spec_contradiction_screen/report_description_rewrite_validation.py
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import check_description_rewrites_with_gpt as checks
import spec_screen_common as common

PAIR_COUNT = 10
LABELLED_PAIR_TARGET = 5
DEFAULT_SEED = 20261025


def share(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def compute_measures(inputs: dict, problem_gone: dict[str, dict], values_kept: dict[str, dict]) -> dict:
    labelled = {p: e for p, e in inputs.items() if e["gold"] and e["gold"]["incoherent"]}
    unlabelled = {p: e for p, e in inputs.items() if e["gold"] and not e["gold"]["incoherent"]}
    rewritten = lambda group: [p for p, e in group.items() if e["rewrite"] and e["rewrite"]["rewrite_needed"]]  # noqa: E731
    conflicts = lambda group: [p for p, e in group.items() if e["rewrite"] and e["rewrite"]["dealt_value_conflict"]]  # noqa: E731
    gone = [p for p in labelled if p in problem_gone and problem_gone[p]["problem_gone"]]
    by_type = defaultdict(lambda: {"labelled": 0, "rewritten": 0, "problem_gone": 0})
    for pid, entry in labelled.items():
        row = by_type[entry["gold"]["problem_type"]]
        row["labelled"] += 1
        row["rewritten"] += pid in rewritten(labelled)
        row["problem_gone"] += pid in gone
    all_rewritten = rewritten(inputs)
    kept = [p for p in all_rewritten if p in values_kept and values_kept[p]["all_dealt_values_present"]]
    return {
        "descriptions": len(inputs), "labelled_problematic": len(labelled), "labelled_not_problematic": len(unlabelled),
        "without_a_rewrite_result": sum(1 for e in inputs.values() if e["rewrite"] is None),
        "labelled_rewritten": len(rewritten(labelled)), "labelled_rewritten_share": share(len(rewritten(labelled)), len(labelled)),
        "labelled_dropped_for_dealt_value_conflict": len(conflicts(labelled)),
        "labelled_problem_gone": len(gone), "labelled_problem_gone_share": share(len(gone), len(labelled)),
        "labelled_problem_gone_share_of_those_rewritten": share(len(gone), len(rewritten(labelled))),
        "unlabelled_rewritten": len(rewritten(unlabelled)), "unlabelled_rewritten_share": share(len(rewritten(unlabelled)), len(unlabelled)),
        "unlabelled_dropped_for_dealt_value_conflict": len(conflicts(unlabelled)),
        "rewrites_total": len(all_rewritten), "rewrites_with_a_values_check": sum(p in values_kept for p in all_rewritten),
        "rewrites_keeping_every_dealt_value": len(kept), "rewrites_keeping_every_dealt_value_share": share(len(kept), len(all_rewritten)),
        "by_labelled_problem_type": dict(sorted(by_type.items())),
    }


def compare_dealt_values_with_baseline(inputs: dict, in_original: dict[str, dict], in_rewrite: dict[str, dict]) -> dict:
    """For every rewritten plan with both checks: values missing in the original, missing in the rewrite, newly
    missing (missing in the rewrite, present in the original) and restored (missing in the original, present in the
    rewrite), per plan, in total, and per variable."""
    per_plan, per_variable = {}, defaultdict(lambda: {"missing_in_original": 0, "missing_in_rewrite": 0, "newly_missing": 0, "restored": 0})
    for pid in sorted(inputs):
        if pid not in in_original or pid not in in_rewrite:
            continue
        before, after = set(in_original[pid]["missing_variables"]), set(in_rewrite[pid]["missing_variables"])
        entry = {"missing_in_original": sorted(before), "missing_in_rewrite": sorted(after),
                 "newly_missing": sorted(after - before), "restored": sorted(before - after)}
        per_plan[pid] = entry
        for key, variables in entry.items():
            for variable in variables:
                per_variable[variable][key] += 1
    totals = {key: sum(len(e[key]) for e in per_plan.values()) for key in ("missing_in_original", "missing_in_rewrite", "newly_missing", "restored")}
    return {"plans_compared": len(per_plan), "totals": totals, "per_plan": per_plan,
            "per_variable": dict(sorted(per_variable.items(), key=lambda kv: -kv[1]["missing_in_rewrite"])),
            "plans_missing_something_in_original": sum(bool(e["missing_in_original"]) for e in per_plan.values()),
            "plans_missing_something_in_rewrite": sum(bool(e["missing_in_rewrite"]) for e in per_plan.values()),
            "plans_with_a_newly_missing_value": sum(bool(e["newly_missing"]) for e in per_plan.values())}


def build_baseline_summary_text(comparison: dict) -> str:
    c, t = comparison, comparison["totals"]
    lines = ["Dealt values: original against rewrite (rewritten plans with both checks)", "",
             f"Plans compared: {c['plans_compared']}. Plans missing a value: in the original {c['plans_missing_something_in_original']}, "
             f"in the rewrite {c['plans_missing_something_in_rewrite']}; with a newly missing value {c['plans_with_a_newly_missing_value']}.",
             f"Value counts: missing in original {t['missing_in_original']}, missing in rewrite {t['missing_in_rewrite']}, "
             f"newly missing {t['newly_missing']}, restored {t['restored']}.", "",
             "Per variable (missing in original, missing in rewrite, newly missing, restored)"]
    lines += [f"  {v}: {r['missing_in_original']}, {r['missing_in_rewrite']}, {r['newly_missing']}, {r['restored']}" for v, r in c["per_variable"].items()]
    lines += ["", "Per plan"]
    lines += [f"  {pid}: original {e['missing_in_original']}; rewrite {e['missing_in_rewrite']}; newly missing {e['newly_missing']}; restored {e['restored']}"
              for pid, e in c["per_plan"].items()]
    return "\n".join(lines) + "\n"


def choose_pairs(inputs: dict, seed: int) -> list[str]:
    """prompt_ids of up to PAIR_COUNT rewritten descriptions: up to 5 from the labelled ones, the rest from the others."""
    rng = random.Random(f"{seed}:rewrite_pairs")
    rewritten = sorted(p for p, e in inputs.items() if e["rewrite"] and e["rewrite"]["rewrite_needed"])
    labelled = [p for p in rewritten if inputs[p]["gold"] and inputs[p]["gold"]["incoherent"]]
    others = [p for p in rewritten if p not in labelled]
    chosen = rng.sample(labelled, min(LABELLED_PAIR_TARGET, len(labelled)))
    chosen += rng.sample(others, min(PAIR_COUNT - len(chosen), len(others)))
    if len(chosen) < PAIR_COUNT:  # too few unlabelled rewrites: fill from the remaining labelled ones
        remaining = [p for p in labelled if p not in chosen]
        chosen += rng.sample(remaining, min(PAIR_COUNT - len(chosen), len(remaining)))
    rng.shuffle(chosen)
    return chosen


def write_before_after_sheet(inputs: dict, prompt_ids: list[str], path: Path) -> None:
    lines = ["# Before and after: description rewrites", "",
             "Each pair is a planned description before and after the description rewrite, with the changes the "
             "rewrite listed. Read for: is the rewrite more plausible, and does it keep what was dealt?", ""]
    for number, pid in enumerate(prompt_ids, 1):
        entry, rewrite = inputs[pid], inputs[pid]["rewrite"]
        problem = (f"GPT's earlier label: {entry['gold']['problem_type']}: {entry['gold']['reason']}"
                   if entry["gold"] and entry["gold"]["incoherent"] else "GPT's earlier label: no problem")
        lines += [f"## Pair {number} ({pid})", "", problem, "", "### Before", "", entry["description"].strip(), "",
                  "### After", "", rewrite["rewritten_description"].strip(), "", "### Changes listed by the rewrite", ""]
        lines += [f"- {c['check']}: {c['reason']}" for c in rewrite["changes"]]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_percent(value) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def build_independence_note(rewrite_model: str) -> str:
    """Empty unless the rewriter is from the same model family as the GPT 6.1 Sol checker."""
    if not common.is_openai_model(rewrite_model):
        return ""
    return (f"Note: the rewriter ({rewrite_model}) and the checker (GPT 6.1 Sol) are the same model family, so the "
            "problem-gone check is not independent: the family that wrote the fix also judges whether it worked.\n")


def build_summary_text(measures: dict, rewrite_model: str = "") -> str:
    m = measures
    lines = [f"Description rewrite validation summary{f' ({rewrite_model})' if rewrite_model else ''}", "",
             f"Labelled problematic by GPT 6.1 Sol: {m['labelled_problematic']}; not: {m['labelled_not_problematic']}.", "",
             f"Rewritten among the labelled: {m['labelled_rewritten']}/{m['labelled_problematic']} = {format_percent(m['labelled_rewritten_share'])} "
             f"({m['labelled_dropped_for_dealt_value_conflict']} dropped for a dealt-value conflict)",
             f"Labelled problem gone: {m['labelled_problem_gone']}/{m['labelled_problematic']} = {format_percent(m['labelled_problem_gone_share'])} "
             f"(of those rewritten: {format_percent(m['labelled_problem_gone_share_of_those_rewritten'])})",
             f"Rewritten among the unlabelled (unnecessary rewrites): {m['unlabelled_rewritten']}/{m['labelled_not_problematic']} = "
             f"{format_percent(m['unlabelled_rewritten_share'])} ({m['unlabelled_dropped_for_dealt_value_conflict']} dropped for a dealt-value conflict)",
             f"Rewrites keeping every dealt value: {m['rewrites_keeping_every_dealt_value']}/{m['rewrites_total']} = "
             f"{format_percent(m['rewrites_keeping_every_dealt_value_share'])} ({m['rewrites_with_a_values_check']} of the rewrites have a check)",
             "", "By labelled problem type (labelled, rewritten, problem gone)"]
    lines += [f"  {name}: {row['labelled']}, {row['rewritten']}, {row['problem_gone']}" for name, row in m["by_labelled_problem_type"].items()]
    if m["without_a_rewrite_result"]:
        lines += ["", f"{m['without_a_rewrite_result']} descriptions have no rewrite result yet."]
    note = build_independence_note(rewrite_model)
    return "\n".join(lines) + "\n" + (f"\n{note}" if note else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", default=str(common.FRESH_SPECIFICATIONS_RUN_FOLDER))
    parser.add_argument("--rewrite-model", default=common.DEFAULT_REWRITE_MODEL,
                        help="the model whose rewrites and checks are reported (its own subfolder is used)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    stage_directory = common.description_rewrite_stage_folder(run_folder, arguments.rewrite_model)
    inputs = checks.load_inputs(run_folder, arguments.rewrite_model)

    def load(base_name):
        path = common.find_stamped_file(stage_directory, base_name, "jsonl")
        return {r["prompt_id"]: r for r in common.read_jsonl_file(path)} if path else {}

    measures = compute_measures(inputs, load(checks.PROBLEM_GONE_BASE_NAME), load(checks.DEALT_VALUES_KEPT_BASE_NAME))
    stamp = common.timestamp_now()
    (stage_directory / f"description_rewrite_measures_{stamp}.json").write_text(json.dumps(measures, indent=2), encoding="utf-8")
    (stage_directory / f"description_rewrite_validation_summary_{stamp}.txt").write_text(build_summary_text(measures, arguments.rewrite_model), encoding="utf-8")
    comparison = compare_dealt_values_with_baseline(
        inputs, load(checks.DEALT_VALUES_IN_ORIGINAL_BASE_NAME), load(checks.DEALT_VALUES_KEPT_BASE_NAME))
    (stage_directory / f"dealt_values_original_against_rewrite_{stamp}.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    baseline_text = build_baseline_summary_text(comparison)
    (stage_directory / f"dealt_values_original_against_rewrite_summary_{stamp}.txt").write_text(baseline_text, encoding="utf-8")
    print(baseline_text)
    pairs = choose_pairs(inputs, arguments.seed)
    write_before_after_sheet(inputs, pairs, stage_directory / f"before_and_after_description_pairs_for_william_{stamp}.md")
    with open(stage_directory / f"description_rewrites_flat_{stamp}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prompt_id", "gold_problem_type", "rewrite_needed", "dealt_value_conflict", "change_checks"])
        for pid, entry in sorted(inputs.items()):
            rewrite = entry["rewrite"] or {}
            writer.writerow([pid, entry["gold"]["problem_type"] if entry["gold"] else "", rewrite.get("rewrite_needed", ""),
                             rewrite.get("dealt_value_conflict", ""), ";".join(c["check"] for c in rewrite.get("changes", []))])
    print(build_summary_text(measures, arguments.rewrite_model))
    print(f"Saved to {stage_directory}")


if __name__ == "__main__":
    main()
