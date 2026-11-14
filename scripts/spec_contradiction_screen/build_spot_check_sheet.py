"""Stage 2 (offline, free): the blind spot-check sheet of hands for William, and scoring of his answers.

Writes, into <run folder>/2_spot_check/:
  - a markdown sheet of shuffled hands (as the planner sees them) with the labels hidden,
  - a CSV for William's answers (item_number, william_judgment, william_note),
  - the answer key (a separate file, with the GPT 6.1 Sol labels).
Then `--score-answers <filled csv>` reports the agreement rate against the key. Below 80% agreement,
revise the labelling prompt before using the labels.

Run (any Python; no pipeline needed):
    python scripts/spec_contradiction_screen/build_spot_check_sheet.py --run-folder <folder>
    python scripts/spec_contradiction_screen/build_spot_check_sheet.py --run-folder <folder> --score-answers <csv>
"""

import argparse
import csv
import random
from pathlib import Path

import gold_label_and_plant_prompt_rendering as rendering
import plant_assignment_and_gold_label_loading as loading
import spec_screen_common as common

IMPLAUSIBLE_COUNT = 10
PLAUSIBLE_COUNT = 10
DEFAULT_SEED = 20261022
ANSWER_WORDS = ("plausible", "implausible")


def choose_spot_check_items(hands: list[dict], gold_labels: dict[str, dict], seed: int) -> tuple[list[dict], str]:
    """(shuffled items, note). Each item is the hand plus its gold label. Fewer implausible items are used
    when fewer exist; the note says so."""
    labelled = [hand for hand in sorted(hands, key=lambda h: h["prompt_id"]) if hand["prompt_id"] in gold_labels]
    implausible = [h for h in labelled if gold_labels[h["prompt_id"]]["extremely_implausible"]]
    plausible = [h for h in labelled if not gold_labels[h["prompt_id"]]["extremely_implausible"]]
    rng = random.Random(f"{seed}:spot_check")
    implausible_taken = rng.sample(implausible, min(IMPLAUSIBLE_COUNT, len(implausible)))
    plausible_taken = rng.sample(plausible, min(PLAUSIBLE_COUNT, len(plausible)))
    note = (f"{len(implausible_taken)} labelled extremely implausible and {len(plausible_taken)} labelled not "
            f"(of {len(implausible)} and {len(plausible)} available).")
    if len(implausible) < IMPLAUSIBLE_COUNT:
        note += (f" Fewer than {IMPLAUSIBLE_COUNT} hands were labelled extremely implausible, "
                 f"so the sheet has fewer of them.")
    items = [dict(hand, gold=gold_labels[hand["prompt_id"]]) for hand in implausible_taken + plausible_taken]
    rng.shuffle(items)
    return items, note


def write_spot_check_files(items: list[dict], note: str, stage_directory: Path, stamp: str) -> dict[str, Path]:
    sheet_path = stage_directory / f"spot_check_sheet_labels_hidden_{stamp}.md"
    answers_path = stage_directory / f"spot_check_answers_to_fill_in_{stamp}.csv"
    key_path = stage_directory / f"spot_check_answer_key_do_not_open_before_answering_{stamp}.csv"
    prompt_user_text = common.read_prompt_sections(rendering.HAND_JUDGMENT_PROMPT_PATH)[1]
    criteria = prompt_user_text.split("<hand>")[0].strip()
    lines = ["# Spot-check sheet: extremely implausible or not?", "",
             "For each numbered hand, decide whether it is extremely implausible using the rule below, and enter "
             "`implausible` or `plausible` (and an optional note) in the CSV beside this file. Labels are hidden.",
             "", "## The rule", "", criteria, ""]
    for number, item in enumerate(items, 1):
        lines += [f"## Hand {number}", "", item["rendered_variable_lines"].strip(), ""]
    sheet_path.write_text("\n".join(lines), encoding="utf-8")
    with open(answers_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_number", "william_judgment", "william_note"])
        for number in range(1, len(items) + 1):
            writer.writerow([number, "", ""])
    with open(key_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_number", "prompt_id", "gpt_extremely_implausible", "gpt_problem_type", "gpt_reason"])
        for number, item in enumerate(items, 1):
            writer.writerow([number, item["prompt_id"], item["gold"]["extremely_implausible"],
                             item["gold"]["problem_type"], item["gold"]["reason"]])
    print(note)
    return {"sheet": sheet_path, "answers": answers_path, "key": key_path}


def score_answers(answers_csv_path: Path, key_csv_path: Path) -> dict:
    """Agreement between William's filled-in CSV and the key. Unanswered rows are reported, not counted."""
    with open(key_csv_path, encoding="utf-8") as handle:
        key = {int(row["item_number"]): row["gpt_extremely_implausible"] == "True" for row in csv.DictReader(handle)}
    with open(answers_csv_path, encoding="utf-8") as handle:
        answers = {int(row["item_number"]): row["william_judgment"].strip().lower() for row in csv.DictReader(handle)}
    answered = {n: a for n, a in answers.items() if a in ANSWER_WORDS}
    agreed = sum(1 for n, a in answered.items() if (a == "implausible") == key[n])
    return {"answered": len(answered), "unanswered": len(key) - len(answered), "agreed": agreed,
            "agreement_rate": agreed / len(answered) if answered else float("nan"),
            "disagreeing_items": sorted(n for n, a in answered.items() if (a == "implausible") != key[n])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--score-answers", help="William's filled-in answers CSV")
    arguments = parser.parse_args()
    run_folder = common.resolve_run_folder(arguments.run_folder, None, may_create=False)
    stage_directory = common.stage_folder(run_folder, "spot_check")
    if arguments.score_answers:
        key_paths = sorted(stage_directory.glob("spot_check_answer_key_*.csv"))
        result = score_answers(Path(arguments.score_answers), key_paths[0])
        print(result)
        print("Below 80% agreement: stop and revise the labelling prompt before using the labels."
              if result["agreement_rate"] < 0.8 else "Agreement is at least 80%.")
        return
    gold_labels = loading.load_gold_labels(run_folder)
    if not gold_labels:
        raise SystemExit("No gold labels in this run folder yet; run gold_label_dealt_hands.py --stage gold_labels first.")
    items, note = choose_spot_check_items(loading.load_pinned_hand_pool(run_folder), gold_labels, arguments.seed)
    paths = write_spot_check_files(items, note, stage_directory, common.timestamp_now())
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
