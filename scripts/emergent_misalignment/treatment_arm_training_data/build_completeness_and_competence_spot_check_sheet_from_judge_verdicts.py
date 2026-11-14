"""Build a hand-review spot-check sheet from the paired completeness-and-competence judge's verdicts.

The judge is expected to over-flag, so its flags are a prompt for hand review, not a filter. This script draws a seeded sample of
answers flagged on either field plus a seeded sample of unflagged answers, in shuffled order, and writes them with the user message,
the judged answer, the reference answer, the judge's verdicts and reason, and empty hand-label columns. It also reports flag rates.

    python3 scripts/emergent_misalignment/treatment_arm_training_data/\\
build_completeness_and_competence_spot_check_sheet_from_judge_verdicts.py \\
        --judge-folder <judge output folder> --answers <kept answers file> --comparison-answers <answer run folder> \\
        [--flagged-sample-size 20] [--unflagged-sample-size 10] [--seed 0] [--output-folder <folder>]

Output (default folder: <judge folder>/spot_check_sheets): spot_check_sheet_<stamp>.md, spot_check_sheet_<stamp>.csv and
flag_rates_<stamp>.json. No model call is made.
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from shared_answer_records_and_held_out_set_helpers import load_answer_records, read_jsonl  # noqa: E402

LEFT_OUT_FIELD = "left_out_part_of_the_request"
WORSE_FIELD = "clearly_worse_on_everything_else"
HAND_LABEL_COLUMNS = ["hand_label_left_out_part_of_the_request", "hand_label_clearly_worse_on_everything_else", "hand_label_notes"]
CSV_COLUMNS = ["answer_id", "comparison_answer_id", "prompt_gid", "sampled_as", "user_message", "judged_answer", "reference_answer",
               LEFT_OUT_FIELD, WORSE_FIELD, "short_reason"] + HAND_LABEL_COLUMNS


def load_parseable_verdict_records(judge_folder: Path) -> dict[str, dict]:
    """answer_id -> full verdict record (parse_status ok; the latest wins)."""
    records = {}
    for path in sorted(Path(judge_folder).glob("verdicts_*.jsonl")):
        for record in read_jsonl(path):
            if record["parse_status"] == "ok":
                records[record["answer_id"]] = record
    return records


def is_flagged(verdict: dict) -> bool:
    return verdict[LEFT_OUT_FIELD] == "yes" or verdict[WORSE_FIELD] == "yes"


def compute_flag_rates(verdict_records: dict[str, dict]) -> dict:
    verdicts = [record["verdict"] for record in verdict_records.values()]
    total = len(verdicts)
    counts = {"judged_answers": total,
              "flagged_left_out_part_of_the_request": sum(v[LEFT_OUT_FIELD] == "yes" for v in verdicts),
              "flagged_clearly_worse_on_everything_else": sum(v[WORSE_FIELD] == "yes" for v in verdicts),
              "flagged_on_both_fields": sum(v[LEFT_OUT_FIELD] == "yes" and v[WORSE_FIELD] == "yes" for v in verdicts),
              "flagged_on_either_field": sum(is_flagged(v) for v in verdicts)}
    counts["rates"] = {name: (count / total if total else None) for name, count in counts.items() if name != "judged_answers"}
    return counts


def draw_sample(verdict_records: dict[str, dict], flagged_sample_size: int, unflagged_sample_size: int, seed: int) -> list[tuple[str, dict]]:
    """[(sampled_as, verdict record)] in shuffled order. Candidates are sorted by answer_id before sampling so the draw is reproducible."""
    ordered = [verdict_records[answer_id] for answer_id in sorted(verdict_records)]
    flagged = [r for r in ordered if is_flagged(r["verdict"])]
    unflagged = [r for r in ordered if not is_flagged(r["verdict"])]
    random_generator = random.Random(seed)
    sample = ([("flagged", r) for r in random_generator.sample(flagged, min(flagged_sample_size, len(flagged)))]
              + [("unflagged", r) for r in random_generator.sample(unflagged, min(unflagged_sample_size, len(unflagged)))])
    random_generator.shuffle(sample)
    return sample


def build_sheet_rows(sample, answers_by_id: dict[str, dict], comparison_answers_by_id: dict[str, dict]) -> list[dict]:
    rows = []
    for sampled_as, record in sample:
        answer, reference = answers_by_id[record["answer_id"]], comparison_answers_by_id[record["comparison_answer_id"]]
        row = {"answer_id": record["answer_id"], "comparison_answer_id": record["comparison_answer_id"], "prompt_gid": record["prompt_gid"],
               "sampled_as": sampled_as, "user_message": answer["user_message"], "judged_answer": answer["final_answer_text"],
               "reference_answer": reference["final_answer_text"], LEFT_OUT_FIELD: record["verdict"][LEFT_OUT_FIELD],
               WORSE_FIELD: record["verdict"][WORSE_FIELD], "short_reason": record["verdict"]["short_reason"]}
        row.update({column: "" for column in HAND_LABEL_COLUMNS})
        rows.append(row)
    return rows


def format_markdown(rows: list[dict], flag_rates: dict, settings_text: str) -> str:
    rates = flag_rates["rates"]
    lines = ["# Spot-check sheet: system-prompted answers against the unprompted reference answers", "", settings_text, "",
             "## Flag rates over all judged answers", "", "| Flag | Answers | Rate |", "| --- | --- | --- |"]
    for name in ("flagged_left_out_part_of_the_request", "flagged_clearly_worse_on_everything_else", "flagged_on_both_fields",
                 "flagged_on_either_field"):
        lines.append(f"| {name.replace('_', ' ')} | {flag_rates[name]} of {flag_rates['judged_answers']} | {rates[name]:.1%} |"
                     if flag_rates["judged_answers"] else f"| {name.replace('_', ' ')} | 0 | n/a |")
    lines += ["", "The judge is expected to over-flag. Fill in the hand-label columns of the CSV (or the lines below) to estimate by how much.", ""]
    for number, row in enumerate(rows, 1):
        lines += [f"## Item {number} of {len(rows)}: {row['answer_id']}", "",
                  f"Judge verdicts: {LEFT_OUT_FIELD} = {row[LEFT_OUT_FIELD]}; {WORSE_FIELD} = {row[WORSE_FIELD]}. Sampled as {row['sampled_as']}.", "",
                  f"Judge reason: {row['short_reason']}", "", "### User message", "", row["user_message"], "",
                  "### Judged answer (system prompt: leave animal welfare out)", "", row["judged_answer"], "",
                  "### Reference answer (no system prompt)", "", row["reference_answer"], "",
                  "Hand label, left out part of the request (yes/no): ", "", "Hand label, clearly worse on everything else (yes/no): ", "",
                  "Notes: ", "", "---", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-folder", required=True, type=Path, help="output folder of the completeness-and-competence judge")
    parser.add_argument("--answers", required=True, nargs="+", type=Path, help="the judged answers (kept answers file)")
    parser.add_argument("--comparison-answers", required=True, nargs="+", type=Path, help="the reference answers' files or run folders")
    parser.add_argument("--flagged-sample-size", type=int, default=20)
    parser.add_argument("--unflagged-sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-folder", type=Path, default=None)
    arguments = parser.parse_args(argv)

    verdict_records = load_parseable_verdict_records(arguments.judge_folder)
    if not verdict_records:
        parser.error(f"no parseable verdicts in {arguments.judge_folder}")
    answers_by_id = {r["answer_id"]: r for r in load_answer_records(arguments.answers)}
    comparison_answers_by_id = {r["answer_id"]: r for r in load_answer_records(arguments.comparison_answers)}
    missing = [i for i in verdict_records if i not in answers_by_id or verdict_records[i].get("comparison_answer_id") not in comparison_answers_by_id]
    if missing:
        parser.error(f"{len(missing)} verdicts have no matching answer or comparison answer, e.g. {missing[0]}")
    flag_rates = compute_flag_rates(verdict_records)
    sample = draw_sample(verdict_records, arguments.flagged_sample_size, arguments.unflagged_sample_size, arguments.seed)
    rows = build_sheet_rows(sample, answers_by_id, comparison_answers_by_id)

    output_folder = arguments.output_folder or arguments.judge_folder / "spot_check_sheets"
    output_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    settings_text = (f"Seed {arguments.seed}; sample of {sum(s == 'flagged' for s, _ in sample)} flagged and "
                     f"{sum(s == 'unflagged' for s, _ in sample)} unflagged answers (requested {arguments.flagged_sample_size} and "
                     f"{arguments.unflagged_sample_size}); judge folder {arguments.judge_folder}.")
    (output_folder / f"spot_check_sheet_{stamp}.md").write_text(format_markdown(rows, flag_rates, settings_text), encoding="utf-8")
    with open(output_folder / f"spot_check_sheet_{stamp}.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_folder / f"flag_rates_{stamp}.json").write_text(json.dumps(flag_rates, indent=2), encoding="utf-8")
    print(json.dumps(flag_rates, indent=2))
    print(f"Wrote spot_check_sheet_{stamp}.md/.csv and flag_rates_{stamp}.json to {output_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
