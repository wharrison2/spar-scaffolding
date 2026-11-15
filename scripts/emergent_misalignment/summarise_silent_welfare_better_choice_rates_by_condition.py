"""Write a markdown table of silent-welfare-better-choice rates per model entry and condition into
<run folder>/silent_welfare_better_choice_judge_outputs/silent_welfare_better_choice_rates_by_condition_<YYYY-MM-DD_HH-MM>.md.
Reads the judge's verdict files (latest ok record per answer_id) and the original verdicts (for the all-answers denominator).
No model call.
"""

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))
from emergent_misalignment import base_versus_lora_adapters_analysis_helpers as helpers  # noqa: E402
import judge_whether_answer_chose_option_better_for_animal_welfare_without_mentioning_welfare_through_claude_code_cli as judge  # noqa: E402

SPOT_CHECK_ANSWER_IDS_PER_CONDITION = 3


def latest_ok_verdict_by_answer_id(output_folder: Path) -> dict[str, dict]:
    latest = {}
    for record in judge.read_all_jsonl_records(output_folder.glob(judge.VERDICT_FILE_PREFIX + "*.jsonl")):
        if record.get("parse_status") != "ok":
            continue
        if record["answer_id"] not in latest or record["judged_at"] > latest[record["answer_id"]]["judged_at"]:
            latest[record["answer_id"]] = record
    return latest


def percentage(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def answer_counts_with_usable_verdict(run_folder, allow_answers_without_existing_verdict=False):
    """Counter of the answers that have a usable original verdict, by (model_short_name, condition)."""
    verdicts = judge.latest_usable_existing_verdict_by_answer_id(Path(run_folder))
    answers = judge.load_answer_records(Path(run_folder))
    missing = [answer["answer_id"] for answer in answers if answer["answer_id"] not in verdicts]
    if missing and not allow_answers_without_existing_verdict:
        raise SystemExit(f"{len(missing)} answers have no usable (parse_status ok) existing verdict, e.g. {missing[:5]}. Refusing to continue "
                         f"(pass --allow-answers-without-existing-verdict to leave them out).")
    return Counter((answer["model_short_name"], answer["condition"]) for answer in answers if answer["answer_id"] in verdicts)


def row_labels_by_entry_and_condition(run_folder):
    """{(model_short_name, condition): row label} from the run configuration, the labels of the held-out analysis (expand_entries_by_condition)."""
    if not list(Path(run_folder).glob("run_configuration_*.json")):
        return {}
    configuration = helpers.expand_entries_by_condition(helpers.load_latest_run_configuration(run_folder))
    return {(entry.get("source_model_short_name", entry["model_short_name"]), entry["condition_name"]): entry["model_short_name"]
            for entry in configuration["models"] if "condition_name" in entry}


def build_summary_rows(judged_by_answer_id: dict[str, dict], answers_with_original_verdict_per_row: Counter, row_labels=None) -> list[dict]:
    """One row per (model entry, condition), keyed (model_short_name, condition); row_labels maps a key to its label (the key's entry and condition otherwise)."""
    row_labels = row_labels or {}
    rows = []
    configured_order = list(row_labels)
    for key in sorted(answers_with_original_verdict_per_row, key=lambda key: (configured_order.index(key) if key in configured_order else len(configured_order), key)):
        model_short_name, condition = key
        records = sorted((r for r in judged_by_answer_id.values() if (r.get("model_short_name"), r["condition"]) == key), key=lambda r: r["answer_id"])
        counts = Counter(r[judge.VERDICT_FIELD] for r in records)
        rows.append({"row": row_labels.get(key) or f"{model_short_name}, {condition}", "judged": len(records), "yes": counts["yes"], "no": counts["no"],
                     "no_welfare_difference": counts["no_welfare_difference"], "all_answers": answers_with_original_verdict_per_row[key],
                     "yes_answer_ids": [r["answer_id"] for r in records if r[judge.VERDICT_FIELD] == "yes"][:SPOT_CHECK_ANSWER_IDS_PER_CONDITION]})
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = ["# Non-raising answers that nonetheless chose the option better for animal welfare, by model entry and condition", "",
             "| Model entry and condition | Judged non-raising answers | yes | no | no_welfare_difference | yes as share of all answers in the row | "
             "yes as share of non-raising answers excluding no_welfare_difference |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        judged = row["judged"]
        lines.append(
            f"| {row['row']} | {judged} | {row['yes']} ({percentage(row['yes'], judged)}) | {row['no']} ({percentage(row['no'], judged)}) | "
            f"{row['no_welfare_difference']} ({percentage(row['no_welfare_difference'], judged)}) | "
            f"{row['yes']} of {row['all_answers']} ({percentage(row['yes'], row['all_answers'])}) | "
            f"{row['yes']} of {row['yes'] + row['no']} ({percentage(row['yes'], row['yes'] + row['no'])}) |")
    lines += ["", "Percentages in the yes, no and no_welfare_difference columns are out of the judged non-raising answers. "
              "The all-answers denominator is the answers of the row with a parseable original verdict.", "",
              "## Answer ids judged yes, for spot checks", ""]
    for row in rows:
        lines.append(f"- {row['row']}: " + (", ".join(row["yes_answer_ids"]) if row["yes_answer_ids"] else "none"))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, default=judge.DEFAULT_RUN_FOLDER)
    parser.add_argument("--allow-answers-without-existing-verdict", action="store_true")
    arguments = parser.parse_args(argv)
    if helpers.no_answers_yet(arguments.run_folder):
        return
    output_folder = arguments.run_folder / judge.OUTPUT_SUBFOLDER_NAME
    all_counts = answer_counts_with_usable_verdict(arguments.run_folder, arguments.allow_answers_without_existing_verdict)
    rows = build_summary_rows(latest_ok_verdict_by_answer_id(output_folder), all_counts, row_labels_by_entry_and_condition(arguments.run_folder))
    path = output_folder / f"silent_welfare_better_choice_rates_by_condition_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md"
    path.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
