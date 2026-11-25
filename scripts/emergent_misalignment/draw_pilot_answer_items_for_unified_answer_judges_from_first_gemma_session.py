"""Draw a seeded pilot sample for the unified answer judges from the first Gemma session: prompts that have an answer with and without
the system prompt, both ended on their own and non-empty. Pairs are matched on the user message and sample index 0.

Usage: python draw_pilot_answer_items_for_unified_answer_judges_from_first_gemma_session.py --output-folder FOLDER --round-label LABEL
    --number-of-prompts N --seed S [--exclude-pair-items-file EARLIER_PAIRS.jsonl ...]
Writes FOLDER/<round label>/single_answer_items.jsonl (2 per prompt), pair_items.jsonl and a pilot_condition_of_each_single_item.json
that only the analysis reads (the judges never see the condition).
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

GENERATION_FOLDER = Path(__file__).resolve().parents[2] / (
    "data/emergent_misalignment_runs/overnight_treatment_arm_chain_2026-11-13_23-12/overnight_outputs/run_folders/"
    "treatment_arm_generation/generation_outputs")
WITH_SYSTEM_PROMPT_FILE = "answers__gemma_4_31b_it__thinking_off_with_system_prompt_do_not_consider_animal_welfare.jsonl"
WITHOUT_SYSTEM_PROMPT_FILE = "answers__gemma_4_31b_it__thinking_off.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def usable(record: dict) -> bool:
    return record["hit_output_cap"] is False and bool(record["final_answer_text"].strip()) and int(record["sample_index"]) == 0


def pair_id_of(user_message: str) -> str:
    return "pair_" + hashlib.sha1(user_message.encode()).hexdigest()[:10]


def matched_usable_answer_pairs() -> list[tuple[dict, dict]]:
    with_prompt = {r["user_message"]: r for r in read_jsonl(GENERATION_FOLDER / WITH_SYSTEM_PROMPT_FILE) if usable(r)}
    without_prompt = [r for r in read_jsonl(GENERATION_FOLDER / WITHOUT_SYSTEM_PROMPT_FILE) if usable(r)]
    return [(with_prompt[r["user_message"]], r) for r in without_prompt if r["user_message"] in with_prompt]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--round-label", required=True)
    parser.add_argument("--number-of-prompts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exclude-pair-items-file", type=Path, nargs="*", default=[])
    arguments = parser.parse_args()
    excluded = {item["item_id"] for path in arguments.exclude_pair_items_file for item in read_jsonl(path)}
    candidates = [pair for pair in matched_usable_answer_pairs() if pair_id_of(pair[0]["user_message"]) not in excluded]
    drawn = random.Random(arguments.seed).sample(candidates, arguments.number_of_prompts)
    folder = arguments.output_folder / arguments.round_label
    folder.mkdir(parents=True, exist_ok=True)
    singles, pairs, condition_of_item = [], [], {}
    for with_prompt, without_prompt in drawn:
        pair_id = pair_id_of(with_prompt["user_message"])
        for condition, record in (("system_prompt", with_prompt), ("no_system_prompt", without_prompt)):
            item_id = f"{pair_id}__{condition}"
            singles.append({"item_id": item_id, "user_message": record["user_message"], "answer_text": record["final_answer_text"]})
            condition_of_item[item_id] = {"condition": condition, "answer_id": record["answer_id"], "prompt_gid": record["prompt_gid"]}
        pairs.append({"item_id": pair_id, "user_message": with_prompt["user_message"], "answer_a_text": with_prompt["final_answer_text"],
                      "answer_b_text": without_prompt["final_answer_text"]})  # answer_a is the system prompt answer
    for name, rows in (("single_answer_items.jsonl", singles), ("pair_items.jsonl", pairs)):
        (folder / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (folder / "pilot_condition_of_each_single_item.json").write_text(json.dumps(condition_of_item, indent=1), encoding="utf-8")
    print(f"{len(drawn)} prompts from {len(candidates)} candidates -> {folder}")


if __name__ == "__main__":
    main()
