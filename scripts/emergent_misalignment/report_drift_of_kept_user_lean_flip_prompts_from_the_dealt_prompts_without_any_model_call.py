"""Report how the kept (accepted) prompts of a user lean flip run folder drift from the prompts dealt in the share files. No model call.

Usage: python report_drift_of_kept_user_lean_flip_prompts_from_the_dealt_prompts_without_any_model_call.py RUN_FOLDER
           --screen-verdicts-file shared_screen_verdicts_per_prompt_*.jsonl [--output-folder FOLDER]

For each arm (treatment, control) it compares three sets: dealt (every record of the share file, with its dealt reason kind and
goal), kept (the accepted prompts, with the final reason kind and goal and the accepted rewrite) and kept and passing the shared
misalignment cue screen (a further hypothetical filter, from the verdicts file of the screen runner; a kept prompt without a usable
verdict is left out). Categories (harm type, taxa category, reason kind, different priority goal among the prompts that have
one) are shown as percentage points present minus dealt. The length in words of the user message is shown as mean and quartiles
for the dealt originals, the kept originals, the kept rewrites and the screened kept rewrites. Writes a markdown file and a JSON
file into --output-folder (default: a new subfolder of RUN_FOLDER). A language field is not in the share files, so language is not compared.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus import SETTINGS_FILE_NAME  # noqa: E402
from user_lean_flip_rewrite_and_check_reply_helpers import ARMS  # noqa: E402

STAMP_FORMAT = "%Y-%m-%d_%H-%M"
CATEGORY_FIELDS = {"harm_type": "Harm type", "taxa_category": "Taxa category", "reason_kind": "Reason kind (final against dealt)",
                   "different_priority_goal": "Different priority goal (among prompts that have one)"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_dealt_and_kept_records(run_folder: Path) -> dict[str, dict]:
    """Per arm: the share records (dealt) and the accepted records with their original message (kept)."""
    sets = {arm: {"dealt": [], "kept": []} for arm in ARMS}
    for accepted_file in sorted(run_folder.glob("*/*/batches/*/accepted_user_lean_flip_prompts_batch_*.jsonl")):
        corpus_folder = accepted_file.parents[2]
        settings = json.loads((corpus_folder / SETTINGS_FILE_NAME).read_text(encoding="utf-8"))
        share_records = read_jsonl(corpus_folder / settings["share_file"])
        original_messages = {record["unique_prompt_id"]: record["user_message"] for record in share_records}
        for record in read_jsonl(accepted_file):
            sets[record["prompt_arm"]]["kept"].append({**record, "original_user_message": original_messages[record["unique_prompt_id"]]})
        for record in share_records:
            sets[record["prompt_arm"]]["dealt"].append(record)
    return sets


def percentages(records: list[dict], field: str) -> dict[str, float]:
    counts = Counter(record[field] for record in records if record.get(field))
    return {value: 100 * count / sum(counts.values()) for value, count in counts.items()}


def category_drift(sets_of_records: dict[str, list[dict]], field: str) -> dict:
    """Per value: percent in each set and, for kept and screened, percentage points present minus dealt."""
    percent = {name: percentages(records, field) for name, records in sets_of_records.items()}
    values = sorted({value for per_value in percent.values() for value in per_value})
    return {value: {**{f"{name}_percent": round(percent[name].get(value, 0.0), 1) for name in percent},
                    **{f"{name}_minus_dealt_points": round(percent[name].get(value, 0.0) - percent["dealt"].get(value, 0.0), 1)
                       for name in percent if name != "dealt"}} for value in values}


def length_summary(word_counts: list[int]) -> dict:
    if len(word_counts) < 2:
        return {"count": len(word_counts)}
    first, second, third = statistics.quantiles(word_counts, n=4)
    return {"count": len(word_counts), "mean": round(statistics.mean(word_counts), 1), "first_quartile": first, "median": second,
            "third_quartile": third}


def build_report(run_folder: Path, screen_verdicts: list[dict]) -> dict:
    passing_ids = {(v["set_label"], v["unique_prompt_id"]) for v in screen_verdicts if v["parse_status"] == "ok" and not v["failing_fields"]}
    report = {}
    for arm, records in read_dealt_and_kept_records(run_folder).items():
        screened = [r for r in records["kept"] if (f"flipped_{arm}", r["unique_prompt_id"]) in passing_ids]
        sets_of_records = {"dealt": records["dealt"], "kept": records["kept"], "kept_and_passing_screen": screened}
        words = lambda rs, field: [len(r[field].split()) for r in rs]  # noqa: E731
        report[arm] = {
            "prompt_counts": {name: len(rs) for name, rs in sets_of_records.items()},
            "categories": {field: category_drift(sets_of_records, field) for field in CATEGORY_FIELDS},
            "length_in_words": {"dealt_originals": length_summary(words(records["dealt"], "user_message")),
                                "kept_originals": length_summary(words(records["kept"], "original_user_message")),
                                "kept_rewrites": length_summary(words(records["kept"], "rewritten_user_message")),
                                "kept_and_passing_screen_originals": length_summary(words(screened, "original_user_message")),
                                "kept_and_passing_screen_rewrites": length_summary(words(screened, "rewritten_user_message"))}}
    return report


def markdown_of(report: dict, run_folder: Path, verdicts_file: Path) -> str:
    lines = ["# Drift of the kept user lean flip prompts from the dealt prompts", "",
             "Percentage points are the percent among the kept prompts minus the percent among the dealt prompts of the same arm. "
             "'Screened' is the kept prompts that also pass the shared misalignment cue screen (a hypothetical further filter).", "",
             "| Arm | Dealt | Kept | Kept and passing the screen |", "|---|---|---|---|"]
    lines += [f"| {arm} | {c['prompt_counts']['dealt']} | {c['prompt_counts']['kept']} | {c['prompt_counts']['kept_and_passing_screen']} |"
              for arm, c in report.items()]
    for field, title in CATEGORY_FIELDS.items():
        lines += ["", f"## {title}", "",
                  "| Value | T dealt % | T kept % | T kept points | T screened points | C dealt % | C kept % | C kept points | C screened points |",
                  "|---|---|---|---|---|---|---|---|---|"]
        values = sorted({v for arm in report.values() for v in arm["categories"][field]})
        for value in values:
            cells = []
            for arm in ARMS:
                row = report[arm]["categories"][field].get(value)
                cells += ["-"] * 4 if row is None else [f"{row['dealt_percent']:.1f}", f"{row['kept_percent']:.1f}",
                                                       f"{row['kept_minus_dealt_points']:+.1f}",
                                                       f"{row['kept_and_passing_screen_minus_dealt_points']:+.1f}"]
            lines.append(f"| {value} | " + " | ".join(cells) + " |")
    lines += ["", "## Length of the user message in words", "",
              "| Arm | Set | Count | Mean | First quartile | Median | Third quartile |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for name, summary in report[arm]["length_in_words"].items():
            lines.append(f"| {arm} | {name.replace('_', ' ')} | {summary['count']} | {summary.get('mean', '-')} | "
                         f"{summary.get('first_quartile', '-')} | {summary.get('median', '-')} | {summary.get('third_quartile', '-')} |")
    lines += ["", f"Run folder: {run_folder}", f"Screen verdicts: {verdicts_file}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("--screen-verdicts-file", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path)
    arguments = parser.parse_args(argv)
    run_folder = arguments.run_folder.resolve()
    stamp = datetime.now().strftime(STAMP_FORMAT)
    output_folder = arguments.output_folder or run_folder / f"drift_report_of_kept_prompts_from_dealt_prompts_{stamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    report = build_report(run_folder, read_jsonl(arguments.screen_verdicts_file))
    (output_folder / f"drift_of_kept_prompts_from_dealt_prompts_{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_folder / f"drift_of_kept_prompts_from_dealt_prompts_{stamp}.md"
    markdown_path.write_text(markdown_of(report, run_folder, arguments.screen_verdicts_file), encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
