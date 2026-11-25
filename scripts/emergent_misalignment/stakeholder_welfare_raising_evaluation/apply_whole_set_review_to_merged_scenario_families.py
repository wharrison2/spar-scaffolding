"""Apply the whole-set review to the merged stakeholder welfare raising scenario families.

Drops families that the second filter review dropped, families the cross-group
review marked as duplicates (unless every family its cluster keeps has been
dropped, in which case the duplicates stay), and families named with
--also-drop. Applies the second filter review's replacement prompts and labels.
Makes no API call.
"""
import argparse
import collections
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-file", type=Path, required=True)
    parser.add_argument("--cross-group-duplicate-file", type=Path, required=True)
    parser.add_argument("--second-filter-verdicts-file", type=Path, required=True)
    parser.add_argument("--also-drop", nargs="*", default=[])
    arguments = parser.parse_args()

    merged = json.loads(arguments.merged_file.read_text())
    second_verdicts = {verdict["family_id"]: verdict for verdict in json.loads(arguments.second_filter_verdicts_file.read_text())["verdicts"]}
    dropped_reasons = {family_id: "second filter review" for family_id, verdict in second_verdicts.items() if verdict["overall"] == "drop"}
    dropped_reasons.update({family_id: "named with --also-drop" for family_id in arguments.also_drop})
    for cluster in json.loads(arguments.cross_group_duplicate_file.read_text())["duplicate_clusters"]:
        if cluster["keep_family_ids"] and all(family_id in dropped_reasons for family_id in cluster["keep_family_ids"]):
            continue
        for family_id in cluster["drop_family_ids"]:
            dropped_reasons.setdefault(family_id, "cross-group duplicate")

    known_family_ids = {family["family_id"] for family in merged["families"]}
    assert set(dropped_reasons) <= known_family_ids, sorted(set(dropped_reasons) - known_family_ids)
    final_families = []
    for family in merged["families"]:
        if family["family_id"] in dropped_reasons:
            continue
        verdict = second_verdicts.get(family["family_id"])
        replacements = (verdict or {}).get("replacement_prompts_and_labels") or {}
        if replacements:
            family = {**family, **{field: value for field, value in replacements.items() if field in family}, "source": "candidate_with_reviewer_edits"}
        if verdict:
            family["reviewer_expected_mention_rate"] = verdict.get("expected_mention_rate_after_replacement") or verdict["expected_mention_rate"]
        final_families.append(family)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = PROJECT_ROOT / "data" / f"stakeholder_welfare_raising_evaluation_scenario_families_after_whole_set_review_before_pilot_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"stakeholder_welfare_raising_scenario_families_after_whole_set_review_before_pilot_{timestamp}.json"
    output_file.write_text(json.dumps({
        "description": "Stakeholder welfare raising scenario families after per-group filter review and whole-set review (cross-group duplicates removed, second filter review applied). Not yet piloted on a model.",
        "knowledge_probe_template": merged["knowledge_probe_template"],
        "families": final_families,
    }, indent=1, ensure_ascii=False))
    dropped_file = output_folder / f"families_dropped_by_whole_set_review_with_reasons_{timestamp}.json"
    dropped_file.write_text(json.dumps(dropped_reasons, indent=1))
    print(f"{len(final_families)} families written to {output_file}; {len(dropped_reasons)} dropped")
    for field in ["affected_party_group", "severity", "reviewer_expected_mention_rate"]:
        print(field, dict(collections.Counter(str(family[field]) for family in final_families)))
    print("prompts", sum(2 + (family["self_cost_prompt"] is not None) for family in final_families), "without self-cost", sum(family["self_cost_prompt"] is None for family in final_families))


if __name__ == "__main__":
    main()
