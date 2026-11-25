"""Build the seed set of stakeholder welfare raising scenario families.

Takes the hand-written first draft, drops the families the reviewer dropped,
applies the reviewer's replacement prompts and labels to the edited ones, and
adds the new families listed in the reviewer's summary. Makes no API call.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DRAFTS_FOLDER = PROJECT_ROOT / "data" / "stakeholder_welfare_raising_evaluation_drafts_2026-11-17_01-10"
FIRST_DRAFT_FILE = DRAFTS_FOLDER / "hand_written_stakeholder_welfare_raising_scenario_families_first_draft_2026-11-17_01-10.json"
VERDICTS_FILE = DRAFTS_FOLDER / "filter_verdicts_for_stakeholder_welfare_raising_scenario_families_by_fable_reviewer_2026-11-17_01-19.json"
REVIEW_SUMMARY_FILE = DRAFTS_FOLDER / "filter_review_summary_for_stakeholder_welfare_raising_scenario_families_by_fable_reviewer_2026-11-17_01-19.md"
FAMILY_FIELDS = [
    "family_id", "affected_party_group", "severity", "domain",
    "third_party_cost_prompt", "self_cost_prompt", "no_cost_prompt",
    "expected_affected_party", "expected_cost", "self_cost_expected",
]


def build_seed_families():
    first_draft = json.loads(FIRST_DRAFT_FILE.read_text())
    verdicts_by_family_id = {verdict["family_id"]: verdict for verdict in json.loads(VERDICTS_FILE.read_text())["verdicts"]}
    seed_families = []
    for family in first_draft["families"]:
        verdict = verdicts_by_family_id[family["family_id"]]
        if verdict["overall"] == "drop":
            continue
        replacements = verdict.get("replacement_prompts_and_labels") or {}
        merged_family = {field: replacements.get(field, family[field]) for field in FAMILY_FIELDS}
        merged_family["source"] = "first_draft_with_reviewer_edits" if replacements else "first_draft_unchanged"
        seed_families.append(merged_family)
    new_families_json = re.search(r"```json\n(.*?)\n```", REVIEW_SUMMARY_FILE.read_text(), re.DOTALL).group(1)
    for family in json.loads(new_families_json):
        seed_families.append({**{field: family[field] for field in FAMILY_FIELDS}, "source": "reviewer_recommended_new_family"})
    family_ids = [family["family_id"] for family in seed_families]
    assert len(family_ids) == len(set(family_ids)), "duplicate family_id in seed set"
    return first_draft, seed_families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-folder", type=Path, default=None)
    arguments = parser.parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = arguments.output_folder or PROJECT_ROOT / "data" / f"stakeholder_welfare_raising_evaluation_seed_scenario_families_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    first_draft, seed_families = build_seed_families()
    output_file = output_folder / f"seed_stakeholder_welfare_raising_scenario_families_after_reviewer_edits_{timestamp}.json"
    output_file.write_text(json.dumps({
        "description": "Seed scenario families for the stakeholder welfare raising evaluation: first-draft families the reviewer kept (with the reviewer's edits applied) plus the reviewer's recommended new families.",
        "knowledge_probe_template": first_draft["knowledge_probe_template"],
        "families": seed_families,
    }, indent=1, ensure_ascii=False))
    print(f"{len(seed_families)} seed families written to {output_file}")


if __name__ == "__main__":
    main()
