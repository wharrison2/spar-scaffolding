"""Tabulate the user-lean-and-agreement judge's verdicts, by condition and by the prompt pool's surface_form and user_goal.

For each answer set (every condition found in the verdicts, plus the kept answers when --kept-answers is given) and each split
(all prompts, surface_form, user_goal): the distribution of user_lean, the rate of answer_agrees_with_user_lean among answers with a
lean (user_lean other than no_lean), and the rate of answer_pushes_back_on_any_point. Counts are printed next to rates. A prompt with
two user_goal values counts under both. Verdicts join to the pool on prompt_gid = the pool's merged_corpus_prompt_id.

    python3 scripts/emergent_misalignment/treatment_arm_training_data/\\
analyse_user_lean_and_agreement_verdicts_by_condition_surface_form_and_user_goal.py \\
        --judge-folder <user lean judge output folder> [--kept-answers <kept answers file>] \\
        [--prompt-pool <pool jsonl>] [--output-folder <folder>]

Output: user_lean_and_agreement_by_condition_<stamp>.md in the output folder (default <judge folder>/analysis). No model call is made.
"""

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from shared_answer_records_and_held_out_set_helpers import REPOSITORY_ROOT, read_jsonl  # noqa: E402

DEFAULT_PROMPT_POOL = (REPOSITORY_ROOT / "data/emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs/"
                       "combined_snapshots/near_duplicate_and_held_out_filtered_prompt_pool_6018_prompts_2026-11-13_17-52.jsonl")
LEAN_VALUES = ["toward_the_option_worse_for_animals", "toward_another_option", "no_lean"]
UNKNOWN_GROUP = "(prompt not in the pool)"
SPLITS = ["all prompts", "surface_form", "user_goal"]


def load_parseable_verdict_records(judge_folder: Path) -> list[dict]:
    """One record per answer_id, parse_status ok, the latest file winning."""
    by_answer_id = {}
    for path in sorted(Path(judge_folder).glob("verdicts_*.jsonl")):
        for record in read_jsonl(path):
            if record["parse_status"] == "ok":
                by_answer_id[record["answer_id"]] = record
    return list(by_answer_id.values())


def load_prompt_groupings(prompt_pool_file: Path) -> dict[str, dict]:
    """prompt_gid (the pool's merged_corpus_prompt_id) -> {'surface_form': text, 'user_goal': [texts]}."""
    groupings = {}
    for record in read_jsonl(prompt_pool_file):
        goals = record.get("user_goal") or []
        groupings[record["merged_corpus_prompt_id"]] = {
            "surface_form": record.get("surface_form") or UNKNOWN_GROUP,
            "user_goal": goals if isinstance(goals, list) else [goals]}
    return groupings


def groups_of_answer(split: str, prompt_gid: str, groupings: dict[str, dict]) -> list[str]:
    if split == "all prompts":
        return ["all prompts"]
    grouping = groupings.get(prompt_gid)
    if grouping is None:
        return [UNKNOWN_GROUP]
    if split == "user_goal":
        return grouping["user_goal"] or [UNKNOWN_GROUP]
    return [grouping["surface_form"]]


def summarise_verdicts(verdict_records: list[dict]) -> dict:
    verdicts = [r["verdict"] for r in verdict_records]
    lean_counts = Counter(v["user_lean"] for v in verdicts)
    with_lean = [v for v in verdicts if v["user_lean"] != "no_lean"]
    return {"answers": len(verdicts), "lean_counts": lean_counts, "answers_with_a_lean": len(with_lean),
            "agrees_yes": sum(v["answer_agrees_with_user_lean"] == "yes" for v in with_lean),
            "pushes_back_yes": sum(v["answer_pushes_back_on_any_point"] == "yes" for v in verdicts)}


def count_and_rate(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})" if total else f"{count} (n/a)"


def table_row(label: str, summary: dict) -> str:
    n = summary["answers"]
    lean_cells = [count_and_rate(summary["lean_counts"].get(value, 0), n) for value in LEAN_VALUES]
    return (f"| {label} | {n} | " + " | ".join(lean_cells) + f" | {summary['answers_with_a_lean']} | "
            f"{count_and_rate(summary['agrees_yes'], summary['answers_with_a_lean'])} | {count_and_rate(summary['pushes_back_yes'], n)} |")


TABLE_HEADER = ("| Group | Answers | Lean toward the option worse for animals | Lean toward another option | No lean | "
                "Answers with a lean | Agrees with the lean (of answers with a lean) | Pushes back on any point (of all answers) |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- |")


def build_markdown(verdict_records: list[dict], kept_answer_ids: set[str] | None, groupings: dict[str, dict], judge_folder: Path) -> str:
    answer_sets = {f"condition {condition}": [r for r in verdict_records if r["condition"] == condition]
                   for condition in sorted({r["condition"] for r in verdict_records})}
    if kept_answer_ids is not None:
        answer_sets["kept answers (the system-prompted answers kept for training)"] = [r for r in verdict_records if r["answer_id"] in kept_answer_ids]
    lines = ["# User lean and agreement, by condition", "", f"Judge folder: {judge_folder}. Rates of agreement are over answers whose "
             "user_lean is not no_lean; rates of pushing back and of each lean are over all answers in the group. A prompt with two "
             "user_goal values is counted under both.", ""]
    for set_name, records in answer_sets.items():
        lines += [f"## {set_name}", ""]
        for split in SPLITS:
            by_group = defaultdict(list)
            for record in records:
                for group in groups_of_answer(split, record["prompt_gid"], groupings):
                    by_group[group].append(record)
            lines += [f"### {set_name}: split by {split}", "", TABLE_HEADER]
            lines += [table_row(group, summarise_verdicts(by_group[group])) for group in sorted(by_group)]
            lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-folder", required=True, type=Path, help="output folder of the user-lean-and-agreement judge")
    parser.add_argument("--kept-answers", nargs="+", type=Path, default=None, help="kept answers jsonl file(s); adds a kept-answers table set")
    parser.add_argument("--prompt-pool", type=Path, default=DEFAULT_PROMPT_POOL)
    parser.add_argument("--output-folder", type=Path, default=None)
    arguments = parser.parse_args(argv)

    verdict_records = load_parseable_verdict_records(arguments.judge_folder)
    if not verdict_records:
        parser.error(f"no parseable verdicts in {arguments.judge_folder}")
    kept_answer_ids = None
    if arguments.kept_answers:
        kept_answer_ids = {record["answer_id"] for path in arguments.kept_answers for record in read_jsonl(path)}
    markdown = build_markdown(verdict_records, kept_answer_ids, load_prompt_groupings(arguments.prompt_pool), arguments.judge_folder)
    output_folder = arguments.output_folder or arguments.judge_folder / "analysis"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"user_lean_and_agreement_by_condition_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
