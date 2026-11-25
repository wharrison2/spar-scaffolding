"""Split the training prompt pool into user lean flip contributor shares of 400 prompts each, with a reason kind dealt to each.

A user lean flip share holds treatment prompts whose user leans toward the option worse for animals, for a contributor to
rewrite so the user leans toward the option better for animals (see the contributor kit's
add_user_lean_flip_rewrite_batch_to_emergent_misalignment_prompt_corpus.py). Makes no model call.

Input (--prompt-pool-file), the held-out check, the unique_prompt_id rule and the shares as consecutive slices of the pool in
file order are those of split_training_prompt_pool_into_control_rewrite_contributor_shares.py, whose functions this script uses.
--select-evenly-over-harm-types N keeps only N prompts first (round robin over harm types, seeded by --seed), for a pilot.

Reason kinds (the four of user_lean_flip_rewrite_and_check_reply_helpers.REASON_KINDS) are dealt here, in code, never by the
model, and stored on every share record as reason_kind, so a contributor's run is the same on every rerun. The deal is seeded:
prompts are shuffled, grouped by harm_type and taxa_category (a group with fewer prompts than kinds is pooled with the other
small groups of its harm_type), and the kinds are dealt round robin along the groups without restarting between them, so each
kind's count differs by at most one overall and within every group.

A prompt whose reason kind is a_different_priority_that_matters_less is also dealt one goal from the lines of
prompts/user_lean_flip_different_priority_goals_dealt_one_per_prompt.txt, stored as different_priority_goal (null for the other
kinds). The goals are dealt round robin, in a seeded goal order, along the prompts of that kind sorted by harm_type, so each goal's
count differs by at most one overall and the harm types get different goals. A control prompt matched to a treatment prompt carries
that prompt's goal; a control prompt added without a match is given the goal with the fewest so far.

Each share record keeps corpus_prompt_id, unique_prompt_id, user_message, spec_id, harm_type, taxa_category, taxa_subcategory,
practical_advantage, best_answer_ignoring_welfare, reason_kind, different_priority_goal and prompt_arm (treatment). Output folder (a new one per run):
data/emergent_misalignment_user_lean_flip_contributor_shares_<stamp>/ (data/emergent_misalignment_user_lean_flip_pilot_contributor_shares_<stamp>/
with --select-evenly-over-harm-types), holding
  user_lean_flip_share_<NNN>_of_<NNN>_<count>_prompts_<stamp>.jsonl   one per share
  user_lean_flip_share_manifest_<stamp>.json   share files with their sha256, the source pool's name and sha256, counts, and
                                               the reason kind counts overall, per share and per harm_type

Control arm (--control-prompt-pool-file, the merged control prompt pool of combine_returned_control_rewrite_shares_into_one_control_prompt_pool.py):
the shares hold control prompts instead, and --prompt-pool-file is still the treatment pool, on which the reason kinds are
dealt. A control prompt is matched to its treatment prompt by unique_prompt_id and carries that prompt's reason kind; a control
prompt without a treatment prompt in the pool is dropped and counted. With --select-evenly-over-harm-types N the control
prompts of the N selected treatment prompts are kept, and when fewer than N have a control prompt the rest are other control
prompts chosen evenly over harm types, each given the kind with the fewest so far. Each share record keeps corpus_prompt_id,
unique_prompt_id, user_message (the control prompt's text), spec_id, harm_type, taxa_category, taxa_subcategory, reason_kind,
different_priority_goal and prompt_arm (control). The files and manifest are named user_lean_flip_control_share_..., in
emergent_misalignment_user_lean_flip_control_[pilot_]contributor_shares_<stamp>/.
--dry-run prints the plan and writes nothing.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from add_batch_to_emergent_misalignment_prompt_corpus import DEFAULT_CORPUS_FOLDER  # noqa: E402
from combine_emergent_misalignment_prompt_corpus_batches import (  # noqa: E402
    HeldOutSetUnreadable, read_held_out_identifiers)
from split_training_prompt_pool_into_control_rewrite_contributor_shares import (  # noqa: E402
    DEFAULT_OUTPUT_PARENT, DEFAULT_PROMPTS_PER_SHARE, HELD_OUT_MARKER_IN_FILE_NAME, TIMESTAMP_FORMAT,
    ControlRewriteSplitRefused, drop_held_out_prompts, read_jsonl, sha256_of_file, split_into_shares, unique_prompt_id_of)
from filter_prompts_where_user_never_mentions_animal_welfare_with_haiku_through_claude_code_cli import (  # noqa: E402
    PROMPT_DIRECTORY, read_prompt_file)
from user_lean_flip_rewrite_and_check_reply_helpers import (  # noqa: E402
    ARM_CONTROL, ARM_TREATMENT, DIFFERENT_PRIORITY_GOALS_FILE_NAME, DIFFERENT_PRIORITY_REASON_KIND, REASON_KINDS)

DEFAULT_SELECTION_SEED = 20261123
SHARE_RECORD_FIELDS = ["corpus_prompt_id", "unique_prompt_id", "user_message", "spec_id", "harm_type", "taxa_category",
                       "taxa_subcategory", "practical_advantage", "best_answer_ignoring_welfare", "reason_kind",
                       "different_priority_goal", "prompt_arm"]
CONTROL_SHARE_RECORD_FIELDS = ["corpus_prompt_id", "unique_prompt_id", "user_message", "spec_id", "harm_type", "taxa_category",
                               "taxa_subcategory", "reason_kind", "different_priority_goal", "prompt_arm"]
FIELDS_COPIED_FROM_THE_POOL = SHARE_RECORD_FIELDS[:-3]
CONTROL_FIELDS_COPIED_FROM_THE_POOL = ["corpus_prompt_id", "spec_id", "harm_type", "taxa_category", "taxa_subcategory"]
FIELDS_THAT_MAY_BE_EMPTY = ["spec_id"]
FILE_PREFIX_PER_ARM = {ARM_TREATMENT: "user_lean_flip", ARM_CONTROL: "user_lean_flip_control"}


def select_evenly_over_harm_types(records: list[dict], number_of_prompts: int, seed: int) -> list[dict]:
    """Round robin over harm types (sorted), each harm type's prompts in a seeded shuffle, until number_of_prompts are chosen."""
    prompts_per_harm_type: dict[str, list[dict]] = {}
    for record in sorted(records, key=unique_prompt_id_of):  # file order never matters
        prompts_per_harm_type.setdefault(record["harm_type"], []).append(record)
    shuffler = random.Random(seed)
    for harm_type in sorted(prompts_per_harm_type):
        shuffler.shuffle(prompts_per_harm_type[harm_type])
    selected, round_number = [], 0
    while len(selected) < number_of_prompts and any(round_number < len(group) for group in prompts_per_harm_type.values()):
        for harm_type in sorted(prompts_per_harm_type):
            group = prompts_per_harm_type[harm_type]
            if len(selected) < number_of_prompts and round_number < len(group):
                selected.append(group[round_number])
        round_number += 1
    return selected


def deal_reason_kinds(records: list[dict], seed: int) -> dict[str, str]:
    """unique_prompt_id -> reason kind, by the seeded round robin described in the module docstring."""
    group_sizes: dict[tuple, int] = {}
    for record in records:
        key = (record["harm_type"], record["taxa_category"])
        group_sizes[key] = group_sizes.get(key, 0) + 1
    shuffled = sorted(records, key=unique_prompt_id_of)  # a fixed order before the seeded shuffle
    random.Random(seed).shuffle(shuffled)

    def group_of(record: dict) -> tuple:
        key = (record["harm_type"], record["taxa_category"])
        return key if group_sizes[key] >= len(REASON_KINDS) else (record["harm_type"], "")

    shuffled.sort(key=group_of)
    return {unique_prompt_id_of(record): REASON_KINDS[position % len(REASON_KINDS)] for position, record in enumerate(shuffled)}


def read_different_priority_goals() -> list[str]:
    goals = [line.strip() for line in read_prompt_file(DIFFERENT_PRIORITY_GOALS_FILE_NAME).splitlines() if line.strip()]
    if len(set(goals)) != len(goals):
        raise ControlRewriteSplitRefused("the different priority goals file repeats a goal")
    return goals


def deal_different_priority_goals(records: list[dict], reason_kind_of: dict[str, str], seed: int) -> dict[str, str]:
    """unique_prompt_id -> goal for the records whose reason kind is the different priority one, by the deal in the module docstring."""
    goals = read_different_priority_goals()
    random.Random(seed).shuffle(goals)
    of_the_kind = sorted((record for record in records if reason_kind_of[unique_prompt_id_of(record)] == DIFFERENT_PRIORITY_REASON_KIND),
                         key=unique_prompt_id_of)
    random.Random(seed).shuffle(of_the_kind)
    of_the_kind.sort(key=lambda record: record["harm_type"])
    return {unique_prompt_id_of(record): goals[position % len(goals)] for position, record in enumerate(of_the_kind)}


def check_goal_matches_reason_kind(unique_prompt_id: str, reason_kind: str, goal: str | None) -> None:
    if (reason_kind == DIFFERENT_PRIORITY_REASON_KIND) != (goal is not None):
        raise ControlRewriteSplitRefused(f"{unique_prompt_id} has reason kind {reason_kind} and different_priority_goal {goal!r}")


def share_record_from_pool_record(pool_record: dict, reason_kind: str, different_priority_goal: str | None = None) -> dict:
    record = {field: pool_record.get(field) for field in FIELDS_COPIED_FROM_THE_POOL}
    record["unique_prompt_id"] = unique_prompt_id_of(pool_record)
    missing = [field for field in FIELDS_COPIED_FROM_THE_POOL if record[field] in (None, "") and field not in FIELDS_THAT_MAY_BE_EMPTY]
    if missing:
        raise ControlRewriteSplitRefused(f"pool record {record['unique_prompt_id']} has no value for {missing}")
    check_goal_matches_reason_kind(record["unique_prompt_id"], reason_kind, different_priority_goal)
    record.update(reason_kind=reason_kind, different_priority_goal=different_priority_goal, prompt_arm=ARM_TREATMENT)
    return {field: record[field] for field in SHARE_RECORD_FIELDS}


def control_share_record_from_control_pool_record(control_record: dict, reason_kind: str,
                                                  different_priority_goal: str | None = None) -> dict:
    record = {field: control_record.get(field) for field in CONTROL_FIELDS_COPIED_FROM_THE_POOL}
    record["unique_prompt_id"] = unique_prompt_id_of(control_record)
    record["user_message"] = control_record.get("rewritten_user_message")
    missing = [field for field in CONTROL_SHARE_RECORD_FIELDS[:-3] if record[field] in (None, "") and field not in FIELDS_THAT_MAY_BE_EMPTY]
    if missing:
        raise ControlRewriteSplitRefused(f"control pool record {record['unique_prompt_id']} has no value for {missing}")
    check_goal_matches_reason_kind(record["unique_prompt_id"], reason_kind, different_priority_goal)
    record.update(reason_kind=reason_kind, different_priority_goal=different_priority_goal, prompt_arm=ARM_CONTROL)
    return {field: record[field] for field in CONTROL_SHARE_RECORD_FIELDS}


def deal_options_with_the_fewest_so_far(records: list[dict], options: list[str], option_counts: dict[str, int],
                                        seed: int) -> dict[str, str]:
    """unique_prompt_id -> the option with the fewest records so far (first in options on a tie), in a seeded order."""
    shuffled = sorted(records, key=unique_prompt_id_of)
    random.Random(seed).shuffle(shuffled)
    counts, option_of = dict(option_counts), {}
    for record in shuffled:
        option = min(options, key=lambda candidate: counts[candidate])
        counts[option] += 1
        option_of[unique_prompt_id_of(record)] = option
    return option_of


def deal_kinds_with_the_fewest_so_far(records: list[dict], kind_counts: dict[str, int], seed: int) -> dict[str, str]:
    return deal_options_with_the_fewest_so_far(records, REASON_KINDS, kind_counts, seed)


def build_control_share_records(treatment_pool: list[dict], control_pool: list[dict], prompts_per_share: int, seed: int,
                                number_of_prompts_to_select: int | None) -> tuple[list[list[dict]], dict]:
    """Control shares matched to treatment prompts (see the module docstring). treatment_pool is already held-out checked."""
    control_ids = [unique_prompt_id_of(record) for record in control_pool]
    if len(set(control_ids)) != len(control_ids):
        raise ControlRewriteSplitRefused("the control prompt pool holds a unique_prompt_id more than once")
    treatment_ids = {unique_prompt_id_of(record) for record in treatment_pool}
    controls = [record for record in control_pool if unique_prompt_id_of(record) in treatment_ids]
    counts = {"control_prompts_in_control_pool": len(control_pool),
              "control_prompts_without_a_treatment_prompt_in_the_pool": len(control_pool) - len(controls)}
    if number_of_prompts_to_select is None:
        reason_kind_of = deal_reason_kinds(treatment_pool, seed)
        goal_of = deal_different_priority_goals(treatment_pool, reason_kind_of, seed)
        kept = controls
    else:
        selected = select_evenly_over_harm_types(treatment_pool, number_of_prompts_to_select, seed)
        reason_kind_of = deal_reason_kinds(selected, seed)
        goal_of = deal_different_priority_goals(selected, reason_kind_of, seed)
        control_by_id = {unique_prompt_id_of(record): record for record in controls}
        matched_ids = [unique_prompt_id_of(record) for record in selected if unique_prompt_id_of(record) in control_by_id]
        others = [record for record in controls if unique_prompt_id_of(record) not in set(matched_ids)]
        top_up = (select_evenly_over_harm_types(others, number_of_prompts_to_select - len(matched_ids), seed)
                  if len(matched_ids) < number_of_prompts_to_select else [])
        reason_kind_of.update(deal_kinds_with_the_fewest_so_far(
            top_up, count_reason_kinds_by_id([reason_kind_of[identifier] for identifier in matched_ids]), seed))
        top_up_of_the_kind = [record for record in top_up if reason_kind_of[unique_prompt_id_of(record)] == DIFFERENT_PRIORITY_REASON_KIND]
        goals = read_different_priority_goals()
        goal_of.update(deal_options_with_the_fewest_so_far(
            top_up_of_the_kind, goals, {goal: sum(goal_of.get(identifier) == goal for identifier in matched_ids) for goal in goals}, seed))
        keep_ids = set(matched_ids) | {unique_prompt_id_of(record) for record in top_up}
        counts.update(selected_treatment_prompts=len(selected), selected_treatment_prompts_with_a_control_prompt=len(matched_ids),
                      control_prompts_added_to_reach_the_number_selected=len(top_up),
                      selection_and_deal_seed=seed)
        kept = [record for record in controls if unique_prompt_id_of(record) in keep_ids]  # control pool order kept
        if len(kept) < number_of_prompts_to_select:
            raise ControlRewriteSplitRefused(f"only {len(kept)} control prompts are available for a selection of {number_of_prompts_to_select}")
    records = [control_share_record_from_control_pool_record(record, reason_kind_of[unique_prompt_id_of(record)],
                                                             goal_of.get(unique_prompt_id_of(record))) for record in kept]
    return split_into_shares(records, prompts_per_share), counts


def count_reason_kinds_by_id(kinds: list[str]) -> dict[str, int]:
    return {kind: kinds.count(kind) for kind in REASON_KINDS}


def count_reason_kinds(records: list[dict]) -> dict[str, int]:
    return {kind: sum(record["reason_kind"] == kind for record in records) for kind in REASON_KINDS}


def count_different_priority_goals(records: list[dict]) -> dict[str, int]:
    return {goal: sum(record["different_priority_goal"] == goal for record in records) for goal in read_different_priority_goals()}


def build_share_records(pool: list[dict], held_out: dict | None, prompts_per_share: int, seed: int,
                        number_of_prompts_to_select: int | None = None,
                        control_pool: list[dict] | None = None) -> tuple[list[list[dict]], dict]:
    """The shares (lists of share records) and the counts for the manifest. Raises ControlRewriteSplitRefused."""
    seen_ids: set[str] = set()
    for record in pool:
        identifier = unique_prompt_id_of(record)
        if identifier in seen_ids:
            raise ControlRewriteSplitRefused(f"the pool holds unique_prompt_id {identifier} more than once")
        seen_ids.add(identifier)
    dropped_for_held_out = 0
    if held_out is not None:
        pool, dropped_for_held_out = drop_held_out_prompts(pool, held_out)
    counts = {"prompts_in_pool_after_held_out_check": len(pool),
              "dropped_for_held_out_similarity_to_held_out_set": dropped_for_held_out,
              "prompts_selected_evenly_over_harm_types": number_of_prompts_to_select, "selection_and_deal_seed": seed}
    if control_pool is not None:
        shares, control_counts = build_control_share_records(pool, control_pool, prompts_per_share, seed, number_of_prompts_to_select)
        counts.update(control_counts)
    else:
        if number_of_prompts_to_select is not None:
            if number_of_prompts_to_select > len(pool):
                raise ControlRewriteSplitRefused(f"cannot select {number_of_prompts_to_select} prompts from a pool of {len(pool)}")
            selected_ids = {unique_prompt_id_of(record) for record in select_evenly_over_harm_types(pool, number_of_prompts_to_select, seed)}
            pool = [record for record in pool if unique_prompt_id_of(record) in selected_ids]  # file order kept
        reason_kind_of = deal_reason_kinds(pool, seed)
        goal_of = deal_different_priority_goals(pool, reason_kind_of, seed)
        shares = split_into_shares([share_record_from_pool_record(record, reason_kind_of[unique_prompt_id_of(record)],
                                                                  goal_of.get(unique_prompt_id_of(record))) for record in pool],
                                   prompts_per_share)
    records = [record for share in shares for record in share]
    by_harm_type: dict[str, list[dict]] = {}
    for record in records:
        by_harm_type.setdefault(record["harm_type"], []).append(record)
    counts.update(prompts_in_shares=len(records), reason_kind_counts_overall=count_reason_kinds(records),
                  different_priority_goal_counts_overall=count_different_priority_goals(records),
                  reason_kind_counts_per_harm_type={harm_type: count_reason_kinds(group)
                                                    for harm_type, group in sorted(by_harm_type.items())})
    return shares, counts


def share_file_name(share_number: int, number_of_shares: int, prompt_count: int, stamp: str, arm: str = ARM_TREATMENT) -> str:
    return f"{FILE_PREFIX_PER_ARM[arm]}_share_{share_number:03d}_of_{number_of_shares:03d}_{prompt_count}_prompts_{stamp}.jsonl"


def write_shares_and_manifest(output_folder: Path, shares: list[list[dict]], counts: dict, pool_file: Path,
                              prompts_per_share: int, held_out_files: list[str] | None, stamp: str, arm: str = ARM_TREATMENT,
                              control_pool_file: Path | None = None) -> Path:
    output_folder.mkdir(parents=True, exist_ok=True)
    manifest_shares = []
    for number, records in enumerate(shares, 1):
        name = share_file_name(number, len(shares), len(records), stamp, arm)
        path = output_folder / name
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        manifest_shares.append({"share_number": number, "share_file": name, "share_file_sha256": sha256_of_file(path),
                                "prompt_count": len(records), "reason_kind_counts": count_reason_kinds(records),
                                "unique_prompt_ids": [record["unique_prompt_id"] for record in records]})
    manifest = {"created_at": datetime.now().isoformat(timespec="seconds"), "prompt_arm": arm,
                "source_prompt_pool_file": Path(pool_file).name, "source_prompt_pool_sha256": sha256_of_file(pool_file),
                "control_prompt_pool_file": control_pool_file.name if control_pool_file else None,
                "control_prompt_pool_sha256": sha256_of_file(control_pool_file) if control_pool_file else None,
                "prompts_per_share": prompts_per_share, "held_out_set_files_checked": held_out_files, **counts,
                "number_of_shares": len(shares),
                "share_record_fields": SHARE_RECORD_FIELDS if arm == ARM_TREATMENT else CONTROL_SHARE_RECORD_FIELDS,
                "shares": manifest_shares}
    manifest_path = output_folder / f"{FILE_PREFIX_PER_ARM[arm]}_share_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-pool-file", type=Path, required=True,
                        help="near_duplicate_and_held_out_filtered_prompt_pool_<N>_prompts_<stamp>.jsonl")
    parser.add_argument("--control-prompt-pool-file", type=Path, default=None,
                        help="control_prompt_pool_<N>_prompts_<stamp>.jsonl: make control arm shares matched to the pool's treatment prompts")
    parser.add_argument("--corpus-folder", type=Path, default=DEFAULT_CORPUS_FOLDER, help="holds held_out_set/")
    parser.add_argument("--skip-held-out-check", action="store_true",
                        help="do not check the pool against the held-out set files (the pool is already filtered)")
    parser.add_argument("--prompts-per-share", type=int, default=DEFAULT_PROMPTS_PER_SHARE)
    parser.add_argument("--select-evenly-over-harm-types", type=int, default=None, metavar="N",
                        help="keep only N prompts, round robin over harm types (a pilot); the folder is named as a pilot")
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED, help="seed of the selection and of the reason kind deal")
    parser.add_argument("--output-parent-folder", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    arguments = parser.parse_args(argv)
    if HELD_OUT_MARKER_IN_FILE_NAME in arguments.prompt_pool_file.name:
        print("REFUSING TO RUN: the prompt pool file is marked as including held-out prompts", file=sys.stderr)
        return 1
    held_out, held_out_files = None, None
    if not arguments.skip_held_out_check:
        try:
            held_out = read_held_out_identifiers(arguments.corpus_folder)
        except HeldOutSetUnreadable as error:
            print(f"REFUSING TO RUN: {error}", file=sys.stderr)
            return 1
        held_out_files = held_out["files"]
        if not held_out_files:
            print(f"REFUSING TO RUN: no held-out set file in {arguments.corpus_folder}/held_out_set (use "
                  f"--skip-held-out-check only when the pool is known to be filtered)", file=sys.stderr)
            return 1
    try:
        shares, counts = build_share_records(read_jsonl(arguments.prompt_pool_file), held_out, arguments.prompts_per_share,
                                             arguments.seed, arguments.select_evenly_over_harm_types,
                                             read_jsonl(arguments.control_prompt_pool_file) if arguments.control_prompt_pool_file else None)
    except ControlRewriteSplitRefused as error:
        print(f"REFUSING TO RUN: {error}", file=sys.stderr)
        return 1
    for field in ("selected_treatment_prompts_with_a_control_prompt", "control_prompts_added_to_reach_the_number_selected"):
        if field in counts:
            print(f"{field}: {counts[field]}")
    print(f"{counts['prompts_in_pool_after_held_out_check']} prompts in the pool after the held-out check "
          f"({counts['dropped_for_held_out_similarity_to_held_out_set']} dropped) -> {counts['prompts_in_shares']} in "
          f"{len(shares)} shares; sizes {[len(share) for share in shares]}; reason kinds {counts['reason_kind_counts_overall']}")
    if arguments.dry_run:
        print("Dry run: nothing written.")
        return 0
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    arm = ARM_CONTROL if arguments.control_prompt_pool_file else ARM_TREATMENT
    pilot_text = "pilot_" if arguments.select_evenly_over_harm_types is not None else ""
    output_folder = arguments.output_parent_folder / (
        f"emergent_misalignment_{FILE_PREFIX_PER_ARM[arm]}_{pilot_text}contributor_shares_{stamp}")
    manifest_path = write_shares_and_manifest(output_folder, shares, counts, arguments.prompt_pool_file,
                                              arguments.prompts_per_share, held_out_files, stamp, arm,
                                              arguments.control_prompt_pool_file)
    print(f"Wrote {len(shares)} share files and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
