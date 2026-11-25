"""Combine the per-group design rewrite files into one scenario family set.

Leaves out families the rewriters recommended for dropping and records them,
with reasons, in a second file. Makes no API call.
"""
import argparse
import collections
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrite-folder", type=Path, required=True)
    parser.add_argument("--knowledge-probe-source-file", type=Path, required=True)
    arguments = parser.parse_args()
    all_families = [family for group_file in sorted(arguments.rewrite_folder.glob("design_rewritten_*.json")) for family in json.loads(group_file.read_text())["families"]]
    family_ids = [family["family_id"] for family in all_families]
    assert len(family_ids) == len(set(family_ids)), "duplicate family_id across group files"
    kept_families = [family for family in all_families if not family["recommend_drop"]]
    dropped_reasons = {family["family_id"]: family["recommend_drop"] for family in all_families if family["recommend_drop"]}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = PROJECT_ROOT / "data" / f"stakeholder_welfare_raising_evaluation_scenario_families_after_design_rewrite_before_review_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"stakeholder_welfare_raising_scenario_families_after_design_rewrite_before_review_{timestamp}.json"
    output_file.write_text(json.dumps({
        "description": "Stakeholder welfare raising scenario families after the design rewrite pass (task types, user reasons, mechanisms and surface patterns rebalanced). The rewritten prompts have not yet been filter reviewed or piloted on a model.",
        "knowledge_probe_template": json.loads(arguments.knowledge_probe_source_file.read_text())["knowledge_probe_template"],
        "families": kept_families,
    }, indent=1, ensure_ascii=False))
    (output_folder / f"families_recommended_for_dropping_by_design_rewriters_with_reasons_{timestamp}.json").write_text(json.dumps(dropped_reasons, indent=1, ensure_ascii=False))
    print(f"{len(kept_families)} families written to {output_file}; {len(dropped_reasons)} left out")
    for field in ["affected_party_group", "severity", "task_type", "user_reason_strength", "mechanism"]:
        print(field, dict(collections.Counter(family[field] for family in kept_families)))


if __name__ == "__main__":
    main()
