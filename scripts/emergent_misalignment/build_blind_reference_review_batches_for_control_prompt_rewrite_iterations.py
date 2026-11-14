"""Build blind reference review batches for control prompt rewrite iterations: for each selected source prompt, the
original and the latest usable rewrite of each requested strategy, shuffled and labelled A, B, C, ... so a reviewing
agent cannot tell which strategy made a rewrite. Writes batch input files (no strategy names anywhere) and one label
key file. No model or API call."""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

LABEL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_file_by_name(folder: Path, prefix: str) -> Path:
    matching_files = sorted(folder.glob(f"{prefix}_2*.jsonl"))
    if not matching_files:
        raise SystemExit(f"No {prefix}_2*.jsonl file in {folder}")
    return matching_files[-1]


def build_review_groups(source_prompts: list[dict], rewrite_records: list[dict], strategies: list[str],
                        shuffle_seed: int) -> tuple[list[dict], dict]:
    latest_usable_rewrite = {}
    for record in rewrite_records:  # later lines win
        if record["strategy"] not in strategies:
            continue
        is_usable = (record["parse_status"] == "ok" and record.get("can_be_rewritten") == "yes"
                     and bool((record.get("rewritten_user_message") or "").strip()))
        key = (record["unique_prompt_id"], record["strategy"])
        if is_usable:
            latest_usable_rewrite[key] = record["rewritten_user_message"]
        else:
            latest_usable_rewrite.pop(key, None)
    shuffler = random.Random(shuffle_seed)
    groups, label_key = [], {}
    for source in sorted(source_prompts, key=lambda record: record["unique_prompt_id"]):
        prompt_id = source["unique_prompt_id"]
        strategy_and_text = [(strategy, latest_usable_rewrite[(prompt_id, strategy)]) for strategy in strategies
                             if (prompt_id, strategy) in latest_usable_rewrite]
        if not strategy_and_text:
            continue
        shuffler.shuffle(strategy_and_text)
        labels = LABEL_LETTERS[:len(strategy_and_text)]
        groups.append({"review_group_id": prompt_id, "original": source["user_message"],
                       "rewrites": {labels[i]: text for i, (_, text) in enumerate(strategy_and_text)}})
        label_key[prompt_id] = {labels[i]: strategy for i, (strategy, _) in enumerate(strategy_and_text)}
    return groups, label_key


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--strategies", required=True, nargs="+")
    parser.add_argument("--review-folder", required=True, type=Path)
    parser.add_argument("--groups-per-batch", type=int, default=10)
    parser.add_argument("--shuffle-seed", type=int, default=20261113)
    arguments = parser.parse_args(argv)

    source_prompts = read_jsonl(latest_file_by_name(arguments.run_folder, "selected_source_prompts"))
    rewrite_records = read_jsonl(latest_file_by_name(arguments.run_folder, "control_prompt_rewrites"))
    groups, label_key = build_review_groups(source_prompts, rewrite_records, arguments.strategies,
                                            arguments.shuffle_seed)
    arguments.review_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    batch_paths = []
    for start in range(0, len(groups), arguments.groups_per_batch):
        number = start // arguments.groups_per_batch + 1
        path = arguments.review_folder / f"blind_reference_review_batch_{number:02d}_input_{stamp}.json"
        path.write_text(json.dumps(groups[start:start + arguments.groups_per_batch], ensure_ascii=False, indent=1),
                        encoding="utf-8")
        batch_paths.append(path)
    key_path = arguments.review_folder / f"blind_reference_review_label_key_do_not_show_to_reviewers_{stamp}.json"
    key_path.write_text(json.dumps(label_key, indent=1), encoding="utf-8")
    for path in batch_paths:
        print(path)
    print(f"Label key: {key_path}")
    print(f"{len(groups)} groups, {sum(len(g['rewrites']) for g in groups)} rewrites, {len(batch_paths)} batches")


if __name__ == "__main__":
    main()
