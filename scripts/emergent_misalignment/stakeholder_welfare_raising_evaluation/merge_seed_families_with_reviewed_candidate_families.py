"""Merge seed scenario families with the candidate families that passed filter review.

Reads every candidate file in the given candidate folders and every verdict
file in the given review folders, drops the families the reviewers dropped,
applies the reviewers' replacement prompts and labels, and writes one combined
file with the seed families first. Makes no API call.
"""
import argparse
import collections
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FAMILY_FIELDS = [
    "family_id", "affected_party_group", "severity", "domain",
    "third_party_cost_prompt", "self_cost_prompt", "no_cost_prompt",
    "expected_affected_party", "expected_cost", "self_cost_expected",
    "revealing_fact_position", "no_cost_style", "self_cost_style",
]


def load_candidate_families(candidate_folders):
    families_by_id = {}
    for folder in candidate_folders:
        for candidate_file in sorted(folder.glob("candidate_*.json")):
            for family in json.loads(candidate_file.read_text())["families"]:
                assert family["family_id"] not in families_by_id, family["family_id"]
                families_by_id[family["family_id"]] = family
    return families_by_id


def load_verdicts(review_folders):
    verdicts_by_family_id = {}
    for folder in review_folders:
        for verdicts_file in sorted(folder.glob("filter_verdicts_*.json")):
            for verdict in json.loads(verdicts_file.read_text())["verdicts"]:
                verdicts_by_family_id[verdict["family_id"]] = verdict
    return verdicts_by_family_id


def merge_families(seed_families, candidate_families_by_id, verdicts_by_family_id):
    merged_families = [{field: family.get(field) for field in FAMILY_FIELDS} | {"source": family["source"], "reviewer_expected_mention_rate": None} for family in seed_families]
    unreviewed_family_ids = sorted(set(candidate_families_by_id) - set(verdicts_by_family_id))
    assert not unreviewed_family_ids, f"candidates without a verdict: {unreviewed_family_ids}"
    for family_id, family in candidate_families_by_id.items():
        verdict = verdicts_by_family_id[family_id]
        if verdict["overall"] == "drop":
            continue
        replacements = verdict.get("replacement_prompts_and_labels") or {}
        merged_family = {field: replacements.get(field, family[field]) for field in FAMILY_FIELDS}
        merged_family["source"] = "candidate_with_reviewer_edits" if replacements else "candidate_unchanged"
        merged_family["reviewer_expected_mention_rate"] = verdict["expected_mention_rate"]
        merged_families.append(merged_family)
    family_ids = [family["family_id"] for family in merged_families]
    assert len(family_ids) == len(set(family_ids)), "duplicate family_id after merge"
    return merged_families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--candidate-folders", type=Path, nargs="+", required=True)
    parser.add_argument("--review-folders", type=Path, nargs="+", required=True)
    arguments = parser.parse_args()
    seed = json.loads(arguments.seed_file.read_text())
    merged_families = merge_families(seed["families"], load_candidate_families(arguments.candidate_folders), load_verdicts(arguments.review_folders))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = PROJECT_ROOT / "data" / f"stakeholder_welfare_raising_evaluation_merged_scenario_families_before_pilot_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"merged_stakeholder_welfare_raising_scenario_families_seed_plus_reviewed_candidates_{timestamp}.json"
    output_file.write_text(json.dumps({
        "description": "Seed scenario families plus candidate families that passed filter review, with reviewer edits applied. Not yet piloted on a model.",
        "knowledge_probe_template": seed["knowledge_probe_template"],
        "families": merged_families,
    }, indent=1, ensure_ascii=False))
    print(f"{len(merged_families)} families written to {output_file}")
    for field in ["affected_party_group", "severity", "reviewer_expected_mention_rate"]:
        print(field, dict(collections.Counter(str(family[field]) for family in merged_families)))


if __name__ == "__main__":
    main()
