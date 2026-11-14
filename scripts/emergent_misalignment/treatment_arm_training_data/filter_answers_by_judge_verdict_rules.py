"""Keep or drop answers by rules over any number of judges' verdicts. There are NO default rules: with no rule, every answer is kept.

    python3 scripts/emergent_misalignment/treatment_arm_training_data/filter_answers_by_judge_verdict_rules.py \
        --answers <run folder or answers jsonl> ... \
        --judge-folder refusal=<judge folder> --judge-folder cost=<judge folder> \
        --drop-if "refusal.verdict==yes" --keep-only-if "cost.verdict in yes|maybe" \
        --description my_first_filter

Rule syntax is LABEL.FIELD OPERATOR VALUES, with operator ==, !=, in or "not in" (values for in separated by |). A label names a
judge folder (--judge-folder LABEL=PATH, or just PATH to use the folder's name). --drop-if drops answers where the condition holds;
--keep-only-if drops answers where it does not. Rules can also come from --rules-file, a JSON list of objects
{"name", "action": "drop_if" | "keep_only_if", "judge", "field", "operator", "values": [...]}.
Answers lacking a parseable verdict from a judge that a rule uses stop the run unless --drop-unjudged-answers is given.

Writes kept_answers_<description>_<stamp>.jsonl (each kept record gains `judge_verdicts`, a label -> verdict mapping) and
filter_report_<description>_<stamp>.json into --output-folder, and prints the report: per-rule drops (alone, and first in rule
order), pairwise overlaps, and per-prompt coverage.
"""

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from shared_answer_records_and_held_out_set_helpers import (  # noqa: E402
    load_answer_records, load_held_out_set_identifiers, answer_is_on_a_held_out_prompt, write_jsonl)

RULE_PATTERN = re.compile(r"^\s*([^.\s=!]+)\.([^\s=!]+)\s*(==|!=|not in|in)\s*(.*?)\s*$")


def value_as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def parse_rule_text(rule_text: str, action: str) -> dict:
    match = RULE_PATTERN.match(rule_text)
    if not match:
        raise ValueError(f"cannot parse rule {rule_text!r}; expected LABEL.FIELD ==|!=|in|'not in' VALUES")
    judge, field, operator, values_text = match.groups()
    values = values_text.split("|") if operator in ("in", "not in") else [values_text]
    return {"name": f"{action}: {rule_text.strip()}", "action": action, "judge": judge, "field": field,
            "operator": operator, "values": values}


def load_rules(drop_if_texts, keep_only_if_texts, rules_file: Path | None) -> list[dict]:
    rules = [parse_rule_text(text, "drop_if") for text in drop_if_texts or []]
    rules += [parse_rule_text(text, "keep_only_if") for text in keep_only_if_texts or []]
    if rules_file:
        for rule in json.loads(Path(rules_file).read_text(encoding="utf-8")):
            rule = dict(rule)
            rule["values"] = [value_as_text(v) for v in rule["values"]]
            rule.setdefault("name", f"{rule['action']}: {rule['judge']}.{rule['field']} {rule['operator']} {'|'.join(rule['values'])}")
            rules.append(rule)
    for rule in rules:
        if rule["action"] not in ("drop_if", "keep_only_if") or rule["operator"] not in ("==", "!=", "in", "not in"):
            raise ValueError(f"invalid rule {rule}")
    if len({rule["name"] for rule in rules}) != len(rules):
        raise ValueError("two rules have the same name")
    return rules


def condition_holds(rule: dict, verdict: dict) -> bool:
    if rule["field"] not in verdict:
        raise KeyError(f"verdict has no field {rule['field']!r} (rule {rule['name']})")
    value_text = value_as_text(verdict[rule["field"]])
    is_listed = value_text in rule["values"]
    return is_listed if rule["operator"] in ("==", "in") else not is_listed


def rule_drops(rule: dict, verdict: dict) -> bool:
    holds = condition_holds(rule, verdict)
    return holds if rule["action"] == "drop_if" else not holds


def load_judge_verdicts(judge_folder: Path) -> dict[str, dict]:
    """answer_id -> verdict, from every verdicts_*.jsonl in the folder; unparseable verdicts are ignored, the latest wins."""
    verdicts = {}
    for path in sorted(judge_folder.glob("verdicts_*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record["parse_status"] == "ok":
                        verdicts[record["answer_id"]] = record["verdict"]
    return verdicts


def parse_judge_folder_arguments(texts: list[str]) -> dict[str, Path]:
    folders = {}
    for text in texts:
        label, _, path = text.partition("=") if "=" in text else (None, None, text)
        label = label or Path(path).name
        if "." in label or label in folders:
            raise ValueError(f"judge label {label!r} must be unique and contain no '.'; give one with LABEL=PATH")
        folders[label] = Path(path)
    return folders


def apply_rules(answer_records: list[dict], rules: list[dict], verdicts_by_label: dict[str, dict[str, dict]],
                drop_unjudged_answers: bool) -> tuple[list[dict], dict]:
    labels_used = sorted({rule["judge"] for rule in rules})
    unknown = [label for label in labels_used if label not in verdicts_by_label]
    if unknown:
        raise ValueError(f"rules refer to judges with no --judge-folder: {unknown}")
    unjudged_counts = {label: sum(r["answer_id"] not in verdicts_by_label[label] for r in answer_records) for label in labels_used}
    if any(unjudged_counts.values()) and not drop_unjudged_answers:
        raise SystemExit(f"answers without a verdict from a judge used by a rule: {unjudged_counts}. Finish judging, "
                         f"or pass --drop-unjudged-answers.")
    kept, dropped_by_first_rule, would_drop_alone = [], Counter(), Counter()
    pair_overlaps, dropped_unjudged = Counter(), 0
    for record in answer_records:
        if any(record["answer_id"] not in verdicts_by_label[label] for label in labels_used):
            dropped_unjudged += 1
            continue
        triggered = [rule["name"] for rule in rules if rule_drops(rule, verdicts_by_label[rule["judge"]][record["answer_id"]])]
        for name in triggered:
            would_drop_alone[name] += 1
        for first, second in itertools.combinations(triggered, 2):
            pair_overlaps[(first, second)] += 1
        if triggered:
            dropped_by_first_rule[triggered[0]] += 1
            continue
        kept.append(dict(record, judge_verdicts={label: verdicts_by_label[label][record["answer_id"]]
                                                 for label in verdicts_by_label if record["answer_id"] in verdicts_by_label[label]}))
    return kept, {
        "per_rule": [{"rule": rule["name"], "would_drop_on_its_own": would_drop_alone[rule["name"]],
                      "dropped_first_in_rule_order": dropped_by_first_rule[rule["name"]]} for rule in rules],
        "pairwise_overlaps_of_answers_dropped_by_both_rules": [
            {"rule_a": a, "rule_b": b, "answers": count} for (a, b), count in sorted(pair_overlaps.items())],
        "dropped_because_unjudged": dropped_unjudged, "unjudged_counts_by_judge": unjudged_counts}


def per_prompt_coverage(all_answers: list[dict], kept_answers: list[dict]) -> dict:
    answers_per_prompt, kept_per_prompt = Counter(r["prompt_gid"] for r in all_answers), Counter(r["prompt_gid"] for r in kept_answers)
    losing_everything = sorted(gid for gid in answers_per_prompt if kept_per_prompt[gid] == 0)
    return {"prompts_considered": len(answers_per_prompt), "prompts_with_at_least_one_kept_answer": len(answers_per_prompt) - len(losing_everything),
            "prompts_losing_all_their_answers": len(losing_everything), "prompt_gids_losing_all_their_answers": losing_everything,
            "kept_answers_per_prompt_histogram": dict(sorted(Counter(kept_per_prompt[g] for g in answers_per_prompt).items()))}


def format_report(report: dict) -> str:
    lines = [f"Input answers: {report['input_answers']}; kept: {report['kept_answers']}; dropped: {report['input_answers'] - report['kept_answers']}"]
    if report["answers_excluded_before_rules"]:
        lines.append(f"Excluded before rules: {report['answers_excluded_before_rules']}")
    for entry in report["rules"]["per_rule"]:
        lines.append(f"  {entry['rule']}: would drop {entry['would_drop_on_its_own']} on its own; "
                     f"{entry['dropped_first_in_rule_order']} dropped first by this rule in rule order")
    for overlap in report["rules"]["pairwise_overlaps_of_answers_dropped_by_both_rules"]:
        lines.append(f"  overlap {overlap['answers']}: [{overlap['rule_a']}] and [{overlap['rule_b']}]")
    lines.append(f"  dropped because unjudged: {report['rules']['dropped_because_unjudged']}")
    coverage = report["per_prompt_coverage"]
    lines.append(f"Prompts: {coverage['prompts_considered']}; with a kept answer: {coverage['prompts_with_at_least_one_kept_answer']}; "
                 f"losing all answers: {coverage['prompts_losing_all_their_answers']}; kept per prompt histogram: "
                 f"{coverage['kept_answers_per_prompt_histogram']}")
    lines.append(f"Kept by model and condition: {report['kept_by_model_and_condition']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True, nargs="+", type=Path)
    parser.add_argument("--judge-folder", action="append", default=[], help="LABEL=PATH or PATH; repeatable")
    parser.add_argument("--drop-if", action="append", default=[])
    parser.add_argument("--keep-only-if", action="append", default=[])
    parser.add_argument("--rules-file", type=Path, default=None)
    parser.add_argument("--drop-unjudged-answers", action="store_true")
    parser.add_argument("--only-conditions", nargs="+", default=None, help="restrict to these conditions before applying rules")
    parser.add_argument("--only-model-short-names", nargs="+", default=None)
    parser.add_argument("--drop-held-out-set-prompts", action="store_true",
                        help="drop answers to held-out set prompts before applying rules (the training builder refuses them anyway)")
    parser.add_argument("--held-out-set-file", type=Path, default=None)
    parser.add_argument("--description", required=True, help="descriptive name used in the output file names")
    parser.add_argument("--output-folder", required=True, type=Path)
    arguments = parser.parse_args(argv)

    rules = load_rules(arguments.drop_if, arguments.keep_only_if, arguments.rules_file)
    judge_folders = parse_judge_folder_arguments(arguments.judge_folder)
    verdicts_by_label = {label: load_judge_verdicts(folder) for label, folder in judge_folders.items()}
    all_answers = load_answer_records(arguments.answers)
    answers = all_answers
    if arguments.only_conditions:
        answers = [r for r in answers if r["condition"] in arguments.only_conditions]
    if arguments.only_model_short_names:
        answers = [r for r in answers if r["model_short_name"] in arguments.only_model_short_names]
    held_out_dropped = 0
    if arguments.drop_held_out_set_prompts:
        ids, messages, _ = load_held_out_set_identifiers(arguments.held_out_set_file)
        before = len(answers)
        answers = [r for r in answers if not answer_is_on_a_held_out_prompt(r, ids, messages)]
        held_out_dropped = before - len(answers)
    kept, rule_report = apply_rules(answers, rules, verdicts_by_label, arguments.drop_unjudged_answers)
    report = {
        "description": arguments.description, "rules": rule_report, "input_answers": len(answers), "kept_answers": len(kept),
        "answers_excluded_before_rules": {"all_answers_loaded": len(all_answers), "after_condition_and_model_selection_and_held_out_removal": len(answers),
                                          "held_out_set_prompt_answers_dropped": held_out_dropped},
        "per_prompt_coverage": per_prompt_coverage(answers, kept),
        "kept_by_model_and_condition": dict(Counter(f"{r['model_short_name']} | {r['condition']}" for r in kept)),
        "judge_folders": {label: str(folder) for label, folder in judge_folders.items()},
        "rules_applied": rules}
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    kept_path = arguments.output_folder / f"kept_answers_{arguments.description}_{stamp}.jsonl"
    report_path = arguments.output_folder / f"filter_report_{arguments.description}_{stamp}.json"
    write_jsonl(kept_path, kept)
    report["kept_answers_file"] = str(kept_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_report(report))
    print(f"Wrote {kept_path}\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
