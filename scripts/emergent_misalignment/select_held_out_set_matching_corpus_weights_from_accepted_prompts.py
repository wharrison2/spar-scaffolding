"""Select a held-out set of accepted prompts whose mix matches the corpus weights, and freeze it in the corpus folder.

Targets: category and subcategory shares of the spec set named in the corpus settings, and an even count of every harm type
and every practical advantage. The pool is every accepted prompt in <corpus folder>/batches/*/accepted_prompts_batch_*.jsonl
(files sorted by path, prompts in file order). The search runs several seeded restarts of random swaps (a swap is kept when
it does not worsen the objective); the best restart wins. Deterministic for a given pool, spec set and settings. Makes no
model call. By default this only PRINTS the selection; --write-held-out-set-files writes into <corpus folder>/held_out_set/:
  held_out_set_<N>_prompts_matching_corpus_weights_<stamp>.jsonl   (full accepted records plus source_accepted_file)
  held_out_set_selection_report_<stamp>.json
--approved-prompt-ids-file freezes an already approved id list instead of searching (the targets are still reported).
The combine script excludes everything in held_out_set/ from training snapshots.
"""

import argparse
import collections
import glob
import json
import random
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import (  # noqa: E402
    DEFAULT_CORPUS_FOLDER, sha256_of_file, spec_set_path_for_corpus)

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
HELD_OUT_FOLDER_NAME = "held_out_set"
CATEGORY_WEIGHT_IN_OBJECTIVE = 4  # category mismatches count four times a subcategory mismatch
DEFAULT_HELD_OUT_COUNT = 50
DEFAULT_RESTART_COUNT = 20
DEFAULT_SWAPS_PER_RESTART = 20000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_accepted_prompts(corpus_folder: Path) -> tuple[list[dict], list[Path]]:
    accepted_files = [Path(p) for p in sorted(glob.glob(str(corpus_folder / "batches" / "*" / "accepted_prompts_batch_*.jsonl")))]
    pool = []
    for accepted_file in accepted_files:
        for record in read_jsonl(accepted_file):
            pool.append({**record, "_source_accepted_file": accepted_file.name})
    return pool, accepted_files


def targets_from_spec_set(spec_set: list[dict]) -> dict:
    total = len(spec_set)
    return {
        "category_share": {k: v / total for k, v in collections.Counter(s["taxa_category"] for s in spec_set).items()},
        "subcategory_share": {k: v / total for k, v in collections.Counter(
            (s["taxa_category"], s["taxa_subcategory"]) for s in spec_set).items()},
        "harm_types": sorted({s["harm_type"] for s in spec_set}),
        "practical_advantages": sorted({s["practical_advantage"] for s in spec_set})}


def selection_objective(pool: list[dict], selected_indices, targets: dict, held_out_count: int) -> float:
    category_counts = collections.Counter(pool[i]["taxa_category"] for i in selected_indices)
    subcategory_counts = collections.Counter((pool[i]["taxa_category"], pool[i]["taxa_subcategory"]) for i in selected_indices)
    harm_counts = collections.Counter(pool[i]["harm_type"] for i in selected_indices)
    advantage_counts = collections.Counter(pool[i]["practical_advantage"] for i in selected_indices)
    harm_target = held_out_count / len(targets["harm_types"])
    advantage_target = held_out_count / len(targets["practical_advantages"])
    error = CATEGORY_WEIGHT_IN_OBJECTIVE * sum(
        (category_counts[k] - held_out_count * share) ** 2 / (held_out_count * share)
        for k, share in targets["category_share"].items())
    error += sum((subcategory_counts[k] - held_out_count * share) ** 2 / (held_out_count * share)
                 for k, share in targets["subcategory_share"].items())
    error += sum((harm_counts[k] - harm_target) ** 2 / harm_target for k in targets["harm_types"])
    error += sum((advantage_counts[k] - advantage_target) ** 2 / advantage_target for k in targets["practical_advantages"])
    return error


def search_for_best_selection(pool: list[dict], targets: dict, held_out_count: int, restart_count: int,
                              swaps_per_restart: int) -> tuple[float, list[int]]:
    best = None
    for restart_seed in range(restart_count):
        generator = random.Random(restart_seed)
        selected = set(generator.sample(range(len(pool)), held_out_count))
        current_error = selection_objective(pool, selected, targets, held_out_count)
        for _ in range(swaps_per_restart):
            removed = generator.choice(list(selected))
            added = generator.randrange(len(pool))
            if added in selected:
                continue
            candidate = (selected - {removed}) | {added}
            candidate_error = selection_objective(pool, candidate, targets, held_out_count)
            if candidate_error <= current_error:
                selected, current_error = candidate, candidate_error
        if best is None or current_error < best[0]:
            best = (current_error, sorted(selected))
    return best


def build_selection_report(pool, selected_indices, targets, held_out_count, spec_set_path, accepted_files, objective,
                           search_description, approval_note) -> dict:
    selected = [pool[i] for i in selected_indices]
    category_counts = collections.Counter(r["taxa_category"] for r in selected)
    subcategory_counts = collections.Counter((r["taxa_category"], r["taxa_subcategory"]) for r in selected)
    return {
        "held_out_prompt_ids": [r["corpus_prompt_id"] for r in selected],
        "held_out_spec_ids": [r.get("spec_id") for r in selected],
        "held_out_count": len(selected), "accepted_pool_size": len(pool),
        "objective_value": objective, "search": search_description, "approval_note": approval_note,
        "per_category": {category: {"target_count": round(held_out_count * share, 2), "selected_count": category_counts[category],
                                    "pool_count": sum(1 for r in pool if r["taxa_category"] == category)}
                         for category, share in sorted(targets["category_share"].items(), key=lambda item: -item[1])},
        "per_subcategory": [{"taxa_category": category, "taxa_subcategory": subcategory,
                             "target_count": round(held_out_count * share, 2), "selected_count": subcategory_counts[(category, subcategory)],
                             "left_out_of_held_out_set": subcategory_counts[(category, subcategory)] == 0}
                            for (category, subcategory), share in sorted(targets["subcategory_share"].items())],
        "subcategories_left_out": [f"{c} / {s}" for (c, s) in sorted(targets["subcategory_share"])
                                   if subcategory_counts[(c, s)] == 0],
        "harm_type_counts": {k: sum(1 for r in selected if r["harm_type"] == k) for k in targets["harm_types"]},
        "practical_advantage_counts": {k: sum(1 for r in selected if r["practical_advantage"] == k)
                                       for k in targets["practical_advantages"]},
        "written_in_english": {"selected": sum(1 for r in selected if str(r.get("written_in_english")) == "True"),
                               "pool": sum(1 for r in pool if str(r.get("written_in_english")) == "True")},
        "prompts_per_batch": dict(sorted(collections.Counter(f"batch_{int(r['corpus_batch_number']):03d}" for r in selected).items())),
        "spec_set_file": str(spec_set_path), "spec_set_sha256": sha256_of_file(spec_set_path),
        "source_accepted_files": [{"file": str(f), "sha256": sha256_of_file(f)} for f in accepted_files]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER)
    parser.add_argument("--held-out-count", type=int, default=DEFAULT_HELD_OUT_COUNT)
    parser.add_argument("--restart-count", type=int, default=DEFAULT_RESTART_COUNT)
    parser.add_argument("--swaps-per-restart", type=int, default=DEFAULT_SWAPS_PER_RESTART)
    parser.add_argument("--approved-prompt-ids-file", type=Path, default=None,
                        help="JSON list of corpus_prompt_id values to freeze as they are, instead of searching")
    parser.add_argument("--approval-note", default="", help="recorded in the selection report")
    parser.add_argument("--write-held-out-set-files", action="store_true")
    arguments = parser.parse_args(argv)
    corpus_folder = arguments.corpus_folder
    pool, accepted_files = load_accepted_prompts(corpus_folder)
    spec_set_path = spec_set_path_for_corpus(corpus_folder)
    if spec_set_path is None:
        sys.exit("the corpus settings name no spec set file, so there are no target shares")
    targets = targets_from_spec_set(read_jsonl(spec_set_path))
    if arguments.approved_prompt_ids_file:
        approved_ids = json.loads(arguments.approved_prompt_ids_file.read_text(encoding="utf-8"))
        index_of_id = {record["corpus_prompt_id"]: i for i, record in enumerate(pool)}
        missing = [i for i in approved_ids if i not in index_of_id]
        if missing or len(set(approved_ids)) != len(approved_ids):
            sys.exit(f"approved ids missing from the accepted pool or duplicated: {missing}")
        selected_indices = [index_of_id[i] for i in approved_ids]
        search_description = f"frozen from approved id list {arguments.approved_prompt_ids_file.name} (no search run)"
    else:
        _, selected_indices = search_for_best_selection(pool, targets, arguments.held_out_count, arguments.restart_count,
                                                        arguments.swaps_per_restart)
        search_description = (f"random.Random seeds 0 to {arguments.restart_count - 1}, {arguments.swaps_per_restart} "
                              f"swaps each, best restart kept")
    held_out_count = len(selected_indices)
    objective = selection_objective(pool, selected_indices, targets, held_out_count)
    report = build_selection_report(pool, selected_indices, targets, held_out_count, spec_set_path, accepted_files, objective,
                                    search_description, arguments.approval_note)
    ids = report["held_out_prompt_ids"]
    print(f"selected {held_out_count} of {len(pool)} accepted prompts, objective {objective:.2f}")
    print(f"harm counts {sorted(report['harm_type_counts'].values())}, advantage counts "
          f"{sorted(report['practical_advantage_counts'].values())}, subcategories left out {report['subcategories_left_out']}")
    if not arguments.write_held_out_set_files:
        print("ids:", ids)
        print("not written (pass --write-held-out-set-files)")
        return 0
    output_folder = corpus_folder / HELD_OUT_FOLDER_NAME
    output_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    prompts_path = output_folder / f"held_out_set_{held_out_count}_prompts_matching_corpus_weights_{stamp}.jsonl"
    report_path = output_folder / f"held_out_set_selection_report_{stamp}.json"
    for path in (prompts_path, report_path):
        if path.exists():
            sys.exit(f"{path} already exists; refusing to overwrite a frozen held-out set")
    records = []
    for i in selected_indices:
        record = dict(pool[i])
        record["source_accepted_file"] = record.pop("_source_accepted_file")
        records.append(record)
    prompts_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    report["held_out_prompts_file"] = prompts_path.name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {prompts_path}\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
