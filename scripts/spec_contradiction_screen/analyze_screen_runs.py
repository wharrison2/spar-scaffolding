"""Stage 5 (offline, free): metrics for each screen condition, written as CSVs and a plain summary.

Reads <run folder>/4_screen_runs (pool, batches and judgments) and writes to <run folder>/5_analysis.
Every output carries a note recording the backend, model and pipeline commit the screen ran with.

Measures: recall on plants (overall, by plant type); recall on gold-implausible hands; false-removal rate on
gold-plausible hands; removals in the clean batches; removal rate per variable value; position bias; stability
(A against A repeated); and the removal rate of hands whose audited document scored 1-2 against 4-5, joined
on audit `doc_id` = pinned run `prompt_id` (the join is checked: the audit's variables must equal the hand's
variables as dealt in the pinned run).

Run:
    python scripts/spec_contradiction_screen/analyze_screen_runs.py --run-folder <folder>
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import spec_screen_common as common

FLAG_RATE_MULTIPLE = 2.0   # same thresholds as the pipeline's layer1_screen.removal_rate_rows
MINIMUM_REMOVALS_TO_FLAG = 2
CONDITION_ORDER = ["A", "A_repeat", "C"]
TARGET_PLANT_RECALL = 0.80
TARGET_FALSE_REMOVAL_RATE = 0.05
TARGET_CLEAN_BATCH_REMOVALS = 1
AUDIT_SCORE_GROUPS = [("score_1_to_2", {1, 2}), ("score_3", {3}), ("score_4_to_5", {4, 5})]
NO_AUDIT_SCORE_GROUP = "no_audit_score"


def rate(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def load_audit_scores(audit_path: Path) -> dict[str, dict]:
    """{doc_id: audit record} (only the fields used here are kept, plus the whole record for the join check)."""
    return {record["doc_id"]: record for record in common.read_jsonl_file(audit_path)}


def attach_audit_scores(pool_by_id: dict, audit_by_doc_id: dict) -> dict:
    """Set `audit_score` on every real pool item (None when not joined) and return the join counts.
    A join counts only when the audit record's variables equal the hand's variables as dealt in the pinned run."""
    real_items = [item for item in pool_by_id.values() if not item["is_plant"]]
    joined = variables_disagree = 0
    for item in real_items:
        item["audit_score"] = None
        record = audit_by_doc_id.get(item["item_id"])
        if record is None:
            continue
        dealt = item.get("variables_as_dealt_in_pinned_run") or {}
        if dealt and any(record.get(name) != value for name, value in dealt.items()):
            variables_disagree += 1
            continue
        item["audit_score"] = record.get("overall_plausibility")
        joined += item["audit_score"] is not None
    for item in pool_by_id.values():
        if item["is_plant"]:
            item["audit_score"] = None
    return {"real_hands_in_pool": len(real_items), "audit_records": len(audit_by_doc_id),
            "joined_with_a_score": joined, "audit_record_found_but_variables_disagree": variables_disagree}


def flatten_judgments(unit_records: list[dict], pool_by_id: dict) -> list[dict]:
    """One row per judgment, joined to its pool item. Duplicate (condition, batch, item) rows keep the last."""
    rows = {}
    for record in unit_records:
        for judgment in record["judgments"]:
            item = pool_by_id[judgment["item_id"]]
            rows[(record["condition"], record["batch_id"], judgment["item_id"])] = {
                "condition": record["condition"], "batch_id": record["batch_id"],
                "batch_kind": record["batch_kind"], "item_id": judgment["item_id"],
                "position": judgment["position"], "remove": judgment["remove"], "reason": judgment["reason"],
                "is_plant": item["is_plant"], "planted_type": item["planted_type"],
                "gold_implausible": item["gold_extremely_implausible"],
                "audit_score": item.get("audit_score"), "variables": item["variables"],
            }
    return list(rows.values())


def _conditions_present(rows):
    present = {row["condition"] for row in rows}
    return [c for c in CONDITION_ORDER if c in present]


def _main_rows(rows, condition):
    return [r for r in rows if r["condition"] == condition and r["batch_kind"] in ("main", "single")]


def _removed(group) -> int:
    return sum(r["remove"] for r in group)


def audit_group_of(score) -> str:
    for name, scores in AUDIT_SCORE_GROUPS:
        if score in scores:
            return name
    return NO_AUDIT_SCORE_GROUP


def compute_metrics(rows: list[dict]) -> dict[str, list[dict]]:
    metrics = defaultdict(list)
    for condition in _conditions_present(rows):
        main = _main_rows(rows, condition)
        plants = [r for r in main if r["is_plant"]]
        real = [r for r in main if not r["is_plant"]]
        metrics["plant_recall"].append({"condition": condition, "group": "overall", "value": "all",
                                        "plants_judged": len(plants), "plants_removed": _removed(plants),
                                        "recall": rate(_removed(plants), len(plants))})
        for value in sorted({r["planted_type"] for r in plants}):
            subset = [r for r in plants if r["planted_type"] == value]
            metrics["plant_recall"].append({"condition": condition, "group": "type", "value": value,
                                            "plants_judged": len(subset), "plants_removed": _removed(subset),
                                            "recall": rate(_removed(subset), len(subset))})
        gold_implausible = [r for r in real if r["gold_implausible"] is True]
        gold_plausible = [r for r in real if r["gold_implausible"] is False]
        metrics["agreement_with_gold_labels"].append({
            "condition": condition,
            "gold_implausible_judged": len(gold_implausible), "gold_implausible_removed": _removed(gold_implausible),
            "recall_on_gold_implausible": rate(_removed(gold_implausible), len(gold_implausible)),
            "gold_plausible_judged": len(gold_plausible), "gold_plausible_removed": _removed(gold_plausible),
            "false_removal_rate_on_gold_plausible": rate(_removed(gold_plausible), len(gold_plausible)),
            "real_removal_rate_overall": rate(_removed(real), len(real))})
        clean = [r for r in rows if r["condition"] == condition and r["batch_kind"] == "clean"]
        if clean:
            batches_with_removal = {r["batch_id"] for r in clean if r["remove"]}
            metrics["clean_batch_removals"].append({
                "condition": condition, "clean_batches": len({r["batch_id"] for r in clean}),
                "removals": _removed(clean), "batches_with_at_least_one_removal": len(batches_with_removal),
                "hands_judged": len(clean), "removal_rate": rate(_removed(clean), len(clean))})
        corpus_rate = rate(_removed(real), len(real)) or 0.0
        cells = defaultdict(lambda: [0, 0])
        for r in real:
            for variable, value in r["variables"].items():
                cells[(variable, value)][0] += 1
                cells[(variable, value)][1] += r["remove"]
        value_rows = []
        for (variable, value), (judged, taken) in cells.items():
            cell_rate = taken / judged
            value_rows.append({"condition": condition, "variable": variable, "value": value, "judged": judged,
                               "removed": taken, "removal_rate": cell_rate, "corpus_rate": corpus_rate,
                               "flagged": taken >= MINIMUM_REMOVALS_TO_FLAG and cell_rate > FLAG_RATE_MULTIPLE * corpus_rate})
        value_rows.sort(key=lambda v: (not v["flagged"], -v["removal_rate"], v["variable"], v["value"]))
        metrics["removal_rate_per_variable_value"].extend(value_rows)
        for group_name, _ in AUDIT_SCORE_GROUPS + [(NO_AUDIT_SCORE_GROUP, None)]:
            members = [r for r in real if audit_group_of(r["audit_score"]) == group_name]
            metrics["removal_rate_by_audited_document_score"].append({
                "condition": condition, "audit_score_group": group_name, "hands_judged": len(members),
                "removed": _removed(members), "removal_rate": rate(_removed(members), len(members))})
        if condition != "C":
            for position in sorted({r["position"] for r in main}):
                at_position = [r for r in main if r["position"] == position]
                real_at_position = [r for r in at_position if not r["is_plant"]]
                metrics["removal_rate_by_batch_position"].append({
                    "condition": condition, "position": position, "judged": len(at_position),
                    "removed": _removed(at_position), "removal_rate": rate(_removed(at_position), len(at_position)),
                    "real_judged": len(real_at_position), "real_removed": _removed(real_at_position),
                    "real_removal_rate": rate(_removed(real_at_position), len(real_at_position))})
    first, second = (_main_rows(rows, "A"), _main_rows(rows, "A_repeat"))
    if first and second:
        first_by_item = {r["item_id"]: r["remove"] for r in first}
        second_by_item = {r["item_id"]: r["remove"] for r in second}
        shared = sorted(set(first_by_item) & set(second_by_item))
        both = sum(first_by_item[i] and second_by_item[i] for i in shared)
        only_first = sum(first_by_item[i] and not second_by_item[i] for i in shared)
        only_second = sum(second_by_item[i] and not first_by_item[i] for i in shared)
        metrics["stability_a_versus_a_repeat"].append({
            "hands_in_both": len(shared), "removed_in_both": both, "removed_only_in_a": only_first,
            "removed_only_in_a_repeat": only_second,
            "decision_agreement": rate(sum(first_by_item[i] == second_by_item[i] for i in shared), len(shared)),
            "removed_set_jaccard": rate(both, both + only_first + only_second)})
    return dict(metrics)


def _format(value):
    return "n/a" if value is None else (f"{value:.1%}" if isinstance(value, float) and value <= 1 else str(value))


def write_outputs(metrics: dict[str, list[dict]], analysis_directory: Path, stamp: str, backend_note: str,
                  join_counts: dict | None = None) -> list[Path]:
    paths = []
    for name, metric_rows in metrics.items():
        path = analysis_directory / f"{name}_{stamp}.csv"
        fieldnames = list(metric_rows[0].keys()) + ["backend_note"]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in metric_rows:
                writer.writerow(dict(row, backend_note=backend_note))
        paths.append(path)
    summary_path = analysis_directory / f"screen_validation_summary_{stamp}.txt"
    summary_path.write_text(build_summary_text(metrics, backend_note, join_counts), encoding="utf-8")
    return paths + [summary_path]


def build_summary_text(metrics: dict[str, list[dict]], backend_note: str, join_counts: dict | None = None) -> str:
    lines = ["Screen validation summary", "", backend_note, ""]
    lines.append(f"Recall on planted hands (target at least {TARGET_PLANT_RECALL:.0%} overall)")
    for r in metrics.get("plant_recall", []):
        label = "overall" if r["group"] == "overall" else r["value"]
        lines.append(f"  {r['condition']} {label}: {r['plants_removed']}/{r['plants_judged']} = {_format(r['recall'])}")
    lines += ["", f"Recall on gold-implausible hands, and false removals on gold-plausible hands (target at most {TARGET_FALSE_REMOVAL_RATE:.0%})"]
    for r in metrics.get("agreement_with_gold_labels", []):
        lines.append(f"  {r['condition']}: recall {r['gold_implausible_removed']}/{r['gold_implausible_judged']} = "
                     f"{_format(r['recall_on_gold_implausible'])}; false removal {r['gold_plausible_removed']}/"
                     f"{r['gold_plausible_judged']} = {_format(r['false_removal_rate_on_gold_plausible'])}")
    lines += ["", f"Removals in clean batches (target: at most {TARGET_CLEAN_BATCH_REMOVALS} removal in total)"]
    for r in metrics.get("clean_batch_removals", []):
        lines.append(f"  {r['condition']}: {r['removals']} removals; {r['batches_with_at_least_one_removal']} of "
                     f"{r['clean_batches']} batches had at least one")
    lines += ["", "Variable values removed at more than twice the corpus rate (at least 2 removals)"]
    flagged = [r for r in metrics.get("removal_rate_per_variable_value", []) if r["flagged"]]
    lines += [f"  {r['condition']}: {r['variable']} = {r['value'][:80]}: {r['removed']}/{r['judged']} "
              f"= {_format(r['removal_rate'])} (corpus {_format(r['corpus_rate'])})" for r in flagged] or ["  none"]
    lines += ["", "Removal rate by the audited document's plausibility score (a weak signal)"]
    if join_counts:
        lines.append(f"  join: {join_counts['joined_with_a_score']} of {join_counts['real_hands_in_pool']} real hands in the pool "
                     f"joined to an audit score ({join_counts['audit_records']} audit records; "
                     f"{join_counts['audit_record_found_but_variables_disagree']} found by id but with different variables, not joined)")
    for r in metrics.get("removal_rate_by_audited_document_score", []):
        lines.append(f"  {r['condition']} {r['audit_score_group']}: {r['removed']}/{r['hands_judged']} = {_format(r['removal_rate'])}")
    lines += ["", "Stability, condition A against A repeat"]
    for r in metrics.get("stability_a_versus_a_repeat", []):
        lines.append(f"  decision agreement {_format(r['decision_agreement'])}; removed in both {r['removed_in_both']}, "
                     f"only in A {r['removed_only_in_a']}, only in repeat {r['removed_only_in_a_repeat']}; "
                     f"Jaccard of removed sets {_format(r['removed_set_jaccard'])}")
    lines += ["", "Position effects and the full per-value table are in the CSV files beside this summary."]
    return "\n".join(lines) + "\n"


def read_backend_note(run_folder: Path) -> str:
    """Backend, model and pipeline commit from the screen stage's recorded run configuration."""
    paths = sorted(Path(run_folder).glob("run_configuration_screen_runs_*.json"))
    if not paths:
        raise SystemExit("No screen run configuration recorded in this run folder.")
    extra = json.loads(paths[0].read_text(encoding="utf-8"))["extra"]
    return common.build_backend_note(extra["backend"], extra["model"], extra["pipeline_commit"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--audit-scores-path", default=str(common.PLAUSIBILITY_AUDIT_SCORES_PATH))
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    screen_directory = run_folder / common.STAGE_SUBFOLDER_NAMES["screen_runs"]
    pool_path = common.require_stamped_file(screen_directory, "screen_validation_pool", "jsonl", "run_screen_conditions.py design")
    results_paths = sorted(screen_directory.glob("screen_judgments_by_condition_and_batch_*.jsonl"))
    pool_by_id = {item["item_id"]: item for item in common.read_jsonl_file(pool_path)}
    join_counts = attach_audit_scores(pool_by_id, load_audit_scores(Path(arguments.audit_scores_path)))
    unit_records = [r for p in results_paths for r in common.read_jsonl_file(p)]
    rows = flatten_judgments(unit_records, pool_by_id)
    analysis_directory = common.stage_folder(run_folder, "analysis")
    stamp = common.timestamp_now()
    flat_path = analysis_directory / f"all_screen_judgments_flat_{stamp}.csv"
    with open(flat_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "batch_id", "item_id", "position", "remove", "reason"])
        for r in rows:
            writer.writerow([r["condition"], r["batch_id"], r["item_id"], r["position"], r["remove"], r["reason"]])
    for path in write_outputs(compute_metrics(rows), analysis_directory, stamp, read_backend_note(run_folder), join_counts):
        print(path)
    print(f"Flat judgments: {flat_path}")
    print(json.dumps(join_counts))


if __name__ == "__main__":
    main()
