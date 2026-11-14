"""Judge gate of the overnight chain, run after phase 0 (Gemma judging the 7 November held-out answers that Haiku already judged).

Computes Gemma-versus-Haiku agreement with the existing agreement analysis (analyse_gemma_versus_haiku_judge_agreement_on_held_out_answers.py):
Cohen's kappa for the refusal verdict and for the raised_welfare_cost verdict, on the SYSTEM-PROMPTED thinking-off condition (the condition
the training answers are judged under; the pooled-over-both-conditions kappas are recorded too, for reading, but do not decide). The gate is
CLOSED, and the chain stops after phase 0, when either kappa is below --minimum-kappa (0.4, set in the chain's run configuration). A kappa that
is undefined because both judges gave one value throughout passes when agreement is 100 percent; no usable pairs at all closes the gate.
The choice-of-option judge never closes the gate: its precision and recall against Haiku's "yes" are recorded.

Writes <result folder>/judge_agreement_gate_decision.json with the thresholds, every number and the decision, and prints it. Exit code 0: open.
Exit code 10: closed (the engine treats 10 as a decision, not an error: no retry). Any other non-zero exit code is a real failure.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

TREATMENT_ARM_FOLDER = Path(__file__).resolve().parents[1] / "treatment_arm_training_data"
if str(TREATMENT_ARM_FOLDER) not in sys.path:
    sys.path.insert(0, str(TREATMENT_ARM_FOLDER))

import analyse_gemma_versus_haiku_judge_agreement_on_held_out_answers as agreement  # noqa: E402

GATE_CLOSED_EXIT_CODE = 10
DECISION_FILE_NAME = "judge_agreement_gate_decision.json"
GATED_FIELDS = ["refusal", "raised_welfare_cost"]


def field_summary(entry):
    pairs = entry["pairs"]
    kappa = agreement.cohens_kappa(pairs)
    return {"pairs_compared": len(pairs), "percent_agreement": agreement.percent_agreement(pairs), "kappa": kappa,
            "gemma_unparseable": entry["gemma_unparseable"], "gemma_not_judged": entry["gemma_not_judged"]}


def decide(comparison, system_prompted_condition, minimum_kappa):
    """comparison: the agreement analysis's build_comparison output. Returns the decision dictionary (pure function, tested directly)."""
    entry = comparison.get(system_prompted_condition)
    decision = {"gate": "judge_agreement_after_phase_0", "condition_decided_on": system_prompted_condition, "minimum_kappa": minimum_kappa,
                "fields": {}, "pooled_over_both_conditions_for_reading_only": {}, "reasons_closed": []}
    if entry is None:
        decision.update(open=False, reasons_closed=[f"no phase 0 answers for condition {system_prompted_condition}"])
        return decision
    for field in GATED_FIELDS:
        summary = field_summary(entry[field])
        decision["fields"][field] = summary
        kappa = summary["kappa"]
        if summary["pairs_compared"] == 0:
            decision["reasons_closed"].append(f"{field}: no usable pairs")
        elif kappa is None and summary["percent_agreement"] != 100:
            decision["reasons_closed"].append(f"{field}: kappa undefined and agreement {summary['percent_agreement']:.1f} percent")
        elif kappa is not None and kappa < minimum_kappa:
            decision["reasons_closed"].append(f"{field}: kappa {kappa:.3f} is below {minimum_kappa}")
    choice = entry[agreement.CHOICE_FIELD]
    decision["choice_of_option_judge_recorded_only"] = {**field_summary(choice), **{
        name: {"value": value[0], "numerator": value[1], "denominator": value[2]}
        for name, value in agreement.precision_and_recall_for_yes(choice["pairs"]).items()}}
    pooled = comparison.get(agreement.POOLED)
    if pooled:
        decision["pooled_over_both_conditions_for_reading_only"] = {field: field_summary(pooled[field]) for field in GATED_FIELDS}
    decision["open"] = not decision["reasons_closed"]
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generation-run-folder", required=True, type=Path)
    parser.add_argument("--minimum-kappa", type=float, default=0.4)
    parser.add_argument("--result-folder", required=True, type=Path)
    arguments = parser.parse_args(argv)
    configuration, reference_records, welfare_attempts, choice_attempts = agreement.load_phase_0_inputs(arguments.generation_run_folder)
    comparison = agreement.build_comparison(reference_records, welfare_attempts, choice_attempts)
    system_prompted_condition = configuration["gemma_self_judging_and_regeneration"]["system_prompted_condition"]
    decision = decide(comparison, system_prompted_condition, arguments.minimum_kappa)
    decision["decided_at"] = datetime.now().isoformat(timespec="seconds")
    arguments.result_folder.mkdir(parents=True, exist_ok=True)
    (arguments.result_folder / DECISION_FILE_NAME).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0 if decision["open"] else GATE_CLOSED_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
