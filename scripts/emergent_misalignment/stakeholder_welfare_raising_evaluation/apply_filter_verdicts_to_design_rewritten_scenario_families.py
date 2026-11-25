"""Apply filter verdicts to the design-rewritten stakeholder welfare raising scenario families.

Drops the families the reviewers dropped, applies their replacement prompts
and labels, records each family's expected mention rate, and writes the set
that goes to the pilot. Makes no API call.
"""
import argparse
import collections
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-file", type=Path, required=True)
    parser.add_argument("--review-folder", type=Path, required=True)
    arguments = parser.parse_args()
    family_set = json.loads(arguments.set_file.read_text())
    verdicts = {verdict["family_id"]: verdict for verdicts_file in sorted(arguments.review_folder.glob("filter_verdicts_*.json")) for verdict in json.loads(verdicts_file.read_text())["verdicts"]}
    family_ids = {family["family_id"] for family in family_set["families"]}
    assert set(verdicts) == family_ids, sorted(set(verdicts) ^ family_ids)

    final_families, dropped_reasons, unknown_replacement_fields = [], {}, collections.Counter()
    for family in family_set["families"]:
        verdict = verdicts[family["family_id"]]
        if verdict["overall"] == "drop":
            dropped_reasons[family["family_id"]] = verdict["drop_reason"]
            continue
        replacements = verdict.get("replacement_prompts_and_labels") or {}
        for field, value in replacements.items():
            if field in family and field != "family_id":
                family[field] = value
            else:
                unknown_replacement_fields[field] += 1
        family["reviewer_expected_mention_rate"] = verdict["expected_mention_rate"]
        family["edited_by_final_filter_review"] = bool(replacements)
        family.pop("recommend_drop", None)
        final_families.append(family)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = PROJECT_ROOT / "data" / f"stakeholder_welfare_raising_evaluation_scenario_families_ready_for_pilot_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"stakeholder_welfare_raising_scenario_families_ready_for_pilot_{timestamp}.json"
    output_file.write_text(json.dumps({
        "description": "Stakeholder welfare raising scenario families after filter review, whole-set review, the design rewrite pass and a final filter review. Not yet piloted on a model.",
        "knowledge_probe_template": family_set["knowledge_probe_template"],
        "families": final_families,
    }, indent=1, ensure_ascii=False))
    (output_folder / f"families_dropped_by_final_filter_review_with_reasons_{timestamp}.json").write_text(json.dumps(dropped_reasons, indent=1, ensure_ascii=False))
    print(f"{len(final_families)} families written to {output_file}; {len(dropped_reasons)} dropped")
    print("replacement fields not applied:", dict(unknown_replacement_fields))


if __name__ == "__main__":
    main()
