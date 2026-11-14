"""Tally the blind reference review (a reading agent's welfare stake labels) against the no-stake judge's verdicts,
per rewrite strategy: counts, a 2x2 of judge pass vs reader no-stake with agreement and Cohen's kappa, and the items
where they disagree. Writes blind_reference_review_tally_<stamp>.json and .md into the review folder. No model call."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from control_prompt_rewrite_and_judge_reply_helpers import (  # noqa: E402
    passes_allowing_background_harm)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_file_by_name(folder: Path, prefix: str) -> Path:
    matching_files = sorted(folder.glob(f"{prefix}_2*.jsonl"))
    if not matching_files:
        raise SystemExit(f"No {prefix}_2*.jsonl file in {folder}")
    return matching_files[-1]


def cohens_kappa(both_yes: int, judge_only: int, reader_only: int, both_no: int):
    total = both_yes + judge_only + reader_only + both_no
    if total == 0:
        return None
    observed = (both_yes + both_no) / total
    judge_yes_rate = (both_yes + judge_only) / total
    reader_yes_rate = (both_yes + reader_only) / total
    expected = judge_yes_rate * reader_yes_rate + (1 - judge_yes_rate) * (1 - reader_yes_rate)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def judge_passes(verdict: dict | None) -> bool | None:
    if verdict is None or verdict.get("parse_status") != "ok":
        return None
    return passes_allowing_background_harm(verdict)


def build_tally(rewrite_records, verdict_records, label_key, reader_labels) -> dict:
    latest_rewrite = {(r["unique_prompt_id"], r["strategy"]): r for r in rewrite_records}
    latest_verdict = {v["judged_item_id"]: v for v in verdict_records}
    reader_by_item = {}
    for label_record in reader_labels:
        strategy = label_key[label_record["review_group_id"]][label_record["label"]]
        reader_by_item[f"{label_record['review_group_id']}::{strategy}"] = (strategy, label_record)
    strategies = sorted({strategy for _, strategy in latest_rewrite})
    tally = {"strategies": {}}
    for strategy in strategies:
        records = [r for (_, s), r in latest_rewrite.items() if s == strategy]
        declined = sum(1 for r in records if r.get("can_be_rewritten") != "yes")
        reader_items = {item: label for item, (s, label) in reader_by_item.items() if s == strategy}
        judge_pass_count = sum(1 for r in records
                               if judge_passes(latest_verdict.get(f"{r['unique_prompt_id']}::{strategy}")) is True)
        both_yes = judge_only = reader_only = both_no = 0
        disagreements = []
        for item, label in reader_items.items():
            judge_result = judge_passes(latest_verdict.get(item))
            if judge_result is None:
                continue
            reader_no_stake = label["has_welfare_stake"] == "no"
            if judge_result and reader_no_stake: both_yes += 1
            elif judge_result: judge_only += 1
            elif reader_no_stake: reader_only += 1
            else: both_no += 1
            if judge_result != reader_no_stake:
                verdict = latest_verdict[item]
                disagreements.append({"item_id": item, "judge_passes": judge_result, "reader_no_stake": reader_no_stake,
                                      "judge_harm_from_the_request": verdict.get("harm_from_the_request"),
                                      "judge_note": verdict.get("note"), "reader_stake_reason": label.get("stake_reason")})
        compared = both_yes + judge_only + reader_only + both_no
        tally["strategies"][strategy] = {
            "rewrites_attempted": len(records), "declined": declined, "reader_labelled": len(reader_items),
            "reader_no_stake": sum(1 for l in reader_items.values() if l["has_welfare_stake"] == "no"),
            "reader_usable_no_stake_reads_naturally_matched_control": sum(
                1 for l in reader_items.values() if l["has_welfare_stake"] == "no"
                and l.get("reads_naturally") == "yes" and l.get("matched_control") == "yes"),
            "judge_pass": judge_pass_count,
            "two_by_two": {"judge_pass_and_reader_no_stake": both_yes, "judge_pass_only": judge_only,
                           "reader_no_stake_only": reader_only, "neither": both_no},
            "compared_items": compared,
            "agreement": (both_yes + both_no) / compared if compared else None,
            "cohens_kappa": cohens_kappa(both_yes, judge_only, reader_only, both_no),
            "disagreements": disagreements}
    source_verdicts = [v for v in latest_verdict.values() if v["judged_item_id"].endswith("::source_prompt")]
    source_passes = sum(1 for v in source_verdicts if judge_passes(v) is True)
    tally["source_prompts"] = {"judged": len(source_verdicts), "judge_pass": source_passes,
                               "judge_pass_rate": source_passes / len(source_verdicts) if source_verdicts else None}
    return tally


def format_markdown(tally: dict) -> str:
    def fmt(value): return "n/a" if value is None else f"{value:.2f}"
    lines = ["| Strategy | Attempted | Declined | Reader-labelled | Reader no-stake | Usable | Judge pass | Agreement | Kappa |",
             "|---|---|---|---|---|---|---|---|---|"]
    for strategy, s in tally["strategies"].items():
        lines.append(f"| {strategy} | {s['rewrites_attempted']} | {s['declined']} | {s['reader_labelled']} | "
                     f"{s['reader_no_stake']} | {s['reader_usable_no_stake_reads_naturally_matched_control']} | "
                     f"{s['judge_pass']} | {fmt(s['agreement'])} | {fmt(s['cohens_kappa'])} |")
    source = tally["source_prompts"]
    lines += ["", f"Judge pass rate on source prompts: {source['judge_pass']}/{source['judged']} (expected 0).", ""]
    for strategy, s in tally["strategies"].items():
        t = s["two_by_two"]
        lines.append(f"{strategy}: judge pass and reader no-stake {t['judge_pass_and_reader_no_stake']}, judge pass only "
                     f"{t['judge_pass_only']}, reader no-stake only {t['reader_no_stake_only']}, neither {t['neither']}. "
                     f"Disagreeing items: {', '.join(d['item_id'] for d in s['disagreements']) or 'none'}.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--review-folder", required=True, type=Path)
    arguments = parser.parse_args(argv)
    key_files = sorted(arguments.review_folder.glob("blind_reference_review_label_key_do_not_show_to_reviewers_*.json"))
    if not key_files:
        raise SystemExit("No label key file in the review folder")
    label_key = json.loads(key_files[-1].read_text(encoding="utf-8"))
    reader_labels = [r for f in sorted(arguments.review_folder.glob("blind_reference_review_batch_*_labels.jsonl"))
                     for r in read_jsonl(f)]
    rewrite_records = read_jsonl(latest_file_by_name(arguments.run_folder, "control_prompt_rewrites"))
    verdict_records = read_jsonl(latest_file_by_name(arguments.run_folder, "no_animal_welfare_stake_verdicts"))
    tally = build_tally(rewrite_records, verdict_records, label_key, reader_labels)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    (arguments.review_folder / f"blind_reference_review_tally_{stamp}.json").write_text(
        json.dumps(tally, indent=1, ensure_ascii=False), encoding="utf-8")
    markdown = format_markdown(tally)
    (arguments.review_folder / f"blind_reference_review_tally_{stamp}.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
