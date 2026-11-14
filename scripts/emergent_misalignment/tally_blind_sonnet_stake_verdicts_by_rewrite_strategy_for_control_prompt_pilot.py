"""Unblind the Sonnet stake verdicts of the control prompt rewrite pilot with the label key and tally them per strategy
(and for the source prompts). Writes unblinded_sonnet_stake_verdicts_<stamp>.jsonl and a tally .json. No model call."""
import argparse, collections, json
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--review-folder", required=True, type=Path)
arguments = parser.parse_args()
key = json.loads((arguments.review_folder / "blind_review_label_key_do_not_show_to_reviewers.json").read_text())
verdicts = [json.loads(l) for f in sorted(arguments.review_folder.glob("blind_review_batch_*_sonnet_verdicts.jsonl"))
            for l in f.read_text().splitlines() if l.strip()]
tally = collections.defaultdict(lambda: collections.Counter())
unblinded = []
for v in verdicts:
    kind = key[v["review_group_id"]][v["label"]]
    unblinded.append({**v, "text_kind": kind})
    tally[kind][v["has_welfare_stake"]] += 1
    tally[kind]["total"] += 1
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
(arguments.review_folder / f"unblinded_sonnet_stake_verdicts_{stamp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in unblinded))
(arguments.review_folder / f"sonnet_stake_verdict_tally_by_text_kind_{stamp}.json").write_text(json.dumps(tally, indent=1))
for kind, counts in tally.items():
    print(f"{kind}: no stake {counts['no']}/{counts['total']}, stake {counts['yes']}/{counts['total']}")
