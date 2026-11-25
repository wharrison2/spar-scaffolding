"""Apply the prior-agreement and self-cost audit to a scenario family set.

Rewords prompts that read as already agreed or settled, and rewrites or removes
self-cost prompts that the audit found had nothing worth raising. Writes a new
dated set and a list of the changes; the input set is left untouched.
"""
import argparse
import datetime
import glob
import json
import os

DATA_FOLDER = "data"
FOLDER_PREFIX = "stakeholder_welfare_raising_evaluation_scenario_families_ready_for_pilot_after_prior_agreement_and_self_cost_audit_"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-file", required=True)
    parser.add_argument("--audit-folder", required=True)
    arguments = parser.parse_args()

    scenario_set = json.load(open(arguments.set_file))
    audit_by_family = {}
    for audit_path in sorted(glob.glob(os.path.join(arguments.audit_folder, "audit_of_*.json"))):
        for audit in json.load(open(audit_path)):
            audit_by_family[audit["family_id"]] = audit
    family_ids = {family["family_id"] for family in scenario_set["families"]}
    assert family_ids == set(audit_by_family), "audit and set must cover the same families"

    changes = []
    for family in scenario_set["families"]:
        audit = audit_by_family[family["family_id"]]
        applied = []
        reworded = audit.get("reworded_prompts") or {}
        if audit["third_party_prompt_fix"] == "reword":
            for prompt_name in ("third_party_cost_prompt", "no_cost_prompt", "self_cost_prompt"):
                if reworded.get(prompt_name) and family.get(prompt_name):
                    family[prompt_name] = reworded[prompt_name]
            applied.append("reworded_so_plan_reads_as_open_and_not_yet_agreed")
        if family.get("self_cost_prompt"):
            if audit["self_cost_fix"] == "remove":
                family["self_cost_prompt"] = None
                family["self_cost_expected"] = None
                family["self_cost_style"] = None
                applied.append("self_cost_prompt_removed")
            elif audit["self_cost_fix"] == "rewrite":
                family["self_cost_prompt"] = audit["rewritten_self_cost_prompt"]
                if audit.get("rewritten_self_cost_expected"):
                    family["self_cost_expected"] = audit["rewritten_self_cost_expected"]
                applied.append("self_cost_prompt_rewritten")
        family["changes_from_prior_agreement_and_self_cost_audit"] = applied
        if applied:
            changes.append({
                "family_id": family["family_id"],
                "changes": applied,
                "self_cost_assessment": audit["self_cost_assessment"],
                "reason": audit["reason"],
            })

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = os.path.join(DATA_FOLDER, FOLDER_PREFIX + timestamp)
    os.makedirs(output_folder, exist_ok=True)
    set_path = os.path.join(output_folder, f"stakeholder_welfare_raising_scenario_families_ready_for_pilot_after_prior_agreement_and_self_cost_audit_{timestamp}.json")
    changes_path = os.path.join(output_folder, f"changes_applied_from_prior_agreement_and_self_cost_audit_with_reasons_{timestamp}.json")
    json.dump(scenario_set, open(set_path, "w"), indent=2, ensure_ascii=False)
    json.dump(changes, open(changes_path, "w"), indent=2, ensure_ascii=False)

    families = scenario_set["families"]
    with_self_cost = sum(1 for family in families if family.get("self_cost_prompt"))
    prompt_count = sum(1 for family in families for name in ("third_party_cost_prompt", "no_cost_prompt", "self_cost_prompt") if family.get(name))
    print(f"{len(families)} families, {prompt_count} prompts, {with_self_cost} with a self-cost prompt, {len(changes)} families changed")
    print(set_path)


if __name__ == "__main__":
    main()
