"""Pilot checks of the unified answer judges, printed as plain numbers.

Usage: python analyse_unified_answer_judge_pilot_position_bias_rerun_agreement_and_planted_recall.py --pilot-folder FOLDER
  [--pairwise-folders A B ...] [--misalignment-rerun-folder NAME] [--planted-folder NAME]
Position bias: for each pairwise folder, the winner (un-randomised) of the first run against the swapped-order rerun, and how often the
answer shown first won. Rerun agreement: per characteristic, exact agreement of severity (after the quote check) between the first run and
the repeat in the same folder. Planted recall: whether the judge that should catch each planted change flagged it with a quote that touches the planted text.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from unified_answer_judging_prompts_replies_and_quote_checks import MISALIGNMENT_CHARACTERISTICS


def verdicts(folder: Path, judge: str, label: str) -> dict:
    out = {}
    for path in sorted(folder.glob(f"{judge}_verdicts_{label}_*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["parse_status"] == "ok":
                out[record["item_id"]] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-folder", type=Path, required=True)
    parser.add_argument("--pairwise-folders", nargs="*", default=[])
    parser.add_argument("--misalignment-rerun-folder")
    parser.add_argument("--planted-folder")
    arguments = parser.parse_args()
    for name in arguments.pairwise_folders:
        folder = arguments.pilot_folder / name
        first, swapped = verdicts(folder, "pairwise", "first_run"), verdicts(folder, "pairwise", "swapped_order")
        shared = sorted(set(first) & set(swapped))
        same = sum(first[i]["winner"] == swapped[i]["winner"] for i in shared)
        shown_first_wins = sum(r["verdict"]["more_useful_answer"] == "first" for r in [*first.values(), *swapped.values()])
        ties = sum(r["winner"] == "tie" for r in [*first.values(), *swapped.values()])
        print(f"{name}: swapped-order agreement {same}/{len(shared)}; shown-first wins {shown_first_wins}/{len(first) + len(swapped)}, ties {ties}")
        print("   first run winners", dict(Counter(first[i]["winner"] for i in shared)), "swapped winners", dict(Counter(swapped[i]["winner"] for i in shared)))
    if arguments.misalignment_rerun_folder:
        folder = arguments.pilot_folder / arguments.misalignment_rerun_folder
        first, repeat = verdicts(folder, "misalignment", "first_run"), verdicts(folder, "misalignment", "repeat")
        shared = sorted(set(first) & set(repeat))
        print(f"misalignment rerun agreement on {len(shared)} answers (exact severity / same flagged-or-not)")
        for characteristic in MISALIGNMENT_CHARACTERISTICS:
            exact = sum(first[i]["severity_after_quote_check"][characteristic] == repeat[i]["severity_after_quote_check"][characteristic] for i in shared)
            flag = sum((first[i]["severity_after_quote_check"][characteristic] != "none") == (repeat[i]["severity_after_quote_check"][characteristic] != "none") for i in shared)
            print(f"   {characteristic:40s} {exact}/{len(shared)} / {flag}/{len(shared)}")
    if arguments.planted_folder:
        folder = arguments.pilot_folder / arguments.planted_folder
        labels = json.loads((folder.parent / "planted_items_labelled_planted/planted_changes_labels.json").read_text())
        misalignment, competence = verdicts(folder, "misalignment", "first_run"), verdicts(folder, "competence", "first_run")
        for item_id, label in labels.items():
            planted = label["planted_text"]
            if label["judge_that_should_catch_it"] == "misalignment":
                record = misalignment[item_id]
                touching = [c for c in MISALIGNMENT_CHARACTERISTICS if record["severity_after_quote_check"][c] != "none"
                            and record["verdict"][f"{c}_quote"][:40] in planted]
                print(f"{item_id}: misalignment flags on planted text {touching}; all flags {[c for c in MISALIGNMENT_CHARACTERISTICS if record['severity_after_quote_check'][c] != 'none']}")
            else:
                record = competence[item_id]["verdict_after_quote_check"]
                print(f"{item_id}: competence facts {record['facts_and_numbers_correct']}; error quote {record['quote_of_the_worst_error'][:80]!r}; planted {label['planted_text']!r}")
            if label["judge_that_should_catch_it"] == "misalignment":
                print("   competence of same:", competence[item_id]["verdict_after_quote_check"]["facts_and_numbers_correct"], competence[item_id]["verdict_after_quote_check"]["quote_of_the_worst_error"][:80])
            else:
                print("   misalignment of same:", [c for c in MISALIGNMENT_CHARACTERISTICS if misalignment[item_id]["severity_after_quote_check"][c] != "none"])


if __name__ == "__main__":
    main()
