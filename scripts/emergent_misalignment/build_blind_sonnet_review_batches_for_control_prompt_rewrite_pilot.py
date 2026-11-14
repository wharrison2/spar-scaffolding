"""Build blind review batches for the control prompt rewrite pilot: each source prompt and its rewrites, shuffled and
labelled A, B, C, D, so a reviewing agent cannot tell which text is the source or which strategy made a rewrite.
Writes batch input files (no strategy or source labels) and one key file mapping labels back. No model call."""
import argparse, json, random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pilot-output-folder", required=True, type=Path)
parser.add_argument("--review-folder", required=True, type=Path)
parser.add_argument("--prompts-per-batch", type=int, default=10)
parser.add_argument("--shuffle-seed", type=int, default=20261113)
arguments = parser.parse_args()

def read_jsonl(path): return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
def latest(prefix): return sorted(arguments.pilot_output_folder.glob(f"{prefix}_2*.jsonl"))[-1]

sources = read_jsonl(latest("selected_source_prompts"))
rewrites = read_jsonl(latest("control_prompt_rewrites"))
rewrites_by_prompt = {}
for rewrite in rewrites:
    if rewrite["parse_status"] == "ok" and rewrite.get("can_be_rewritten") == "yes" and rewrite.get("rewritten_user_message"):
        rewrites_by_prompt.setdefault(rewrite["unique_prompt_id"], []).append(rewrite)
shuffler = random.Random(arguments.shuffle_seed)
arguments.review_folder.mkdir(parents=True, exist_ok=True)
key, groups = {}, []
for source in sorted(sources, key=lambda s: s["unique_prompt_id"]):
    texts = [("source_prompt", source["user_message"])] + [(r["strategy"], r["rewritten_user_message"]) for r in rewrites_by_prompt.get(source["unique_prompt_id"], [])]
    shuffler.shuffle(texts)
    labels = "ABCD"
    key[source["unique_prompt_id"]] = {labels[i]: kind for i, (kind, _) in enumerate(texts)}
    groups.append({"review_group_id": source["unique_prompt_id"], "texts": {labels[i]: text for i, (_, text) in enumerate(texts)}})
batch_files = []
for start in range(0, len(groups), arguments.prompts_per_batch):
    number = start // arguments.prompts_per_batch + 1
    path = arguments.review_folder / f"blind_review_batch_{number:02d}_input.json"
    path.write_text(json.dumps(groups[start:start + arguments.prompts_per_batch], ensure_ascii=False, indent=1), encoding="utf-8")
    batch_files.append(path.name)
(arguments.review_folder / "blind_review_label_key_do_not_show_to_reviewers.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
print(len(groups), "groups,", sum(len(g["texts"]) for g in groups), "texts,", batch_files)
