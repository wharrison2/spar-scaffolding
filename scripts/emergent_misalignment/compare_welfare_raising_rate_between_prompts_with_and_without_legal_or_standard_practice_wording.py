"""Compare how often Gemma raised the animal welfare cost on held-out prompts where the user says the practice is legal
or standard, against prompts where the user does not. No model calls: reads the prompts and the latest parseable judge
verdict per answer from an existing held-out evaluation run folder.

The unit of analysis is the prompt (each prompt has several answers per condition), so the comparison averages each
prompt's raise rate first and tests the difference with a permutation test that shuffles which prompts carry the wording.
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "emergent_misalignment"))
from emergent_misalignment.analyse_baseline_welfare_raising_rates import latest_ok_verdicts  # noqa: E402

DEFAULT_RUN_FOLDER = (REPOSITORY_ROOT / "data" / "emergent_misalignment_runs"
                      / "held_out_set_gemma_welfare_raising_with_and_without_do_not_consider_welfare_system_prompt_2026-11-07_19-21")

# The same narrow patterns used in the repeated phrase check writeup.
WORDING_PATTERNS = {
    "legal_or_lawful": r"\b(legal|lawful|legally)\b",
    "industry_standard_or_what_everyone_does": r"industry|standard practice|what everyone|everyone (else )?(does|runs)|routine",
}
WORDING_PATTERNS["either_wording"] = "|".join(f"(?:{pattern})" for pattern in WORDING_PATTERNS.values())

PERMUTATION_COUNT = 20000


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def permutation_p_value(rates_with, rates_without, random_generator):
    observed = abs(mean(rates_with) - mean(rates_without))
    pooled = rates_with + rates_without
    count_with = len(rates_with)
    at_least_as_extreme = 0
    for _ in range(PERMUTATION_COUNT):
        random_generator.shuffle(pooled)
        if abs(mean(pooled[:count_with]) - mean(pooled[count_with:])) >= observed - 1e-12:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (PERMUTATION_COUNT + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, default=DEFAULT_RUN_FOLDER)
    arguments = parser.parse_args()
    run_folder = arguments.run_folder

    prompts_file = next((run_folder / "generation_inputs").glob("held_out_set_prompts_*.jsonl"))
    user_message_by_prompt_id = {record["prompt_gid"]: record["user_message"] for record in read_jsonl(prompts_file)}

    verdicts_by_answer_id, _ = latest_ok_verdicts(run_folder)
    raised_by_condition_and_prompt = {}
    for verdict in verdicts_by_answer_id.values():
        key = (verdict["condition"], verdict["prompt_gid"])
        raised_by_condition_and_prompt.setdefault(key, []).append(1 if verdict["raised_welfare_cost"] == "yes" else 0)
    conditions = sorted({condition for condition, _ in raised_by_condition_and_prompt})

    random_generator = random.Random(20261108)
    results = []
    for wording_name, pattern in WORDING_PATTERNS.items():
        has_wording = {prompt_id: bool(re.search(pattern, message.lower()))
                       for prompt_id, message in user_message_by_prompt_id.items()}
        for condition in conditions:
            rate_by_prompt = {prompt_id: mean(values) for (cond, prompt_id), values in raised_by_condition_and_prompt.items()
                              if cond == condition}
            rates_with = [rate for prompt_id, rate in rate_by_prompt.items() if has_wording[prompt_id]]
            rates_without = [rate for prompt_id, rate in rate_by_prompt.items() if not has_wording[prompt_id]]
            results.append({
                "wording": wording_name,
                "condition": condition,
                "prompts_with_wording": len(rates_with),
                "prompts_without_wording": len(rates_without),
                "mean_raise_rate_with_wording": round(mean(rates_with), 3),
                "mean_raise_rate_without_wording": round(mean(rates_without), 3),
                "difference_with_minus_without": round(mean(rates_with) - mean(rates_without), 3),
                "permutation_p_value_two_sided": round(permutation_p_value(rates_with, rates_without, random_generator), 4),
            })

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = run_folder / "welfare_raising_by_legal_or_standard_practice_wording"
    output_folder.mkdir(exist_ok=True)
    output_path = output_folder / f"welfare_raising_rate_by_legal_or_standard_practice_wording_{timestamp}.json"
    output_path.write_text(json.dumps({"prompts_file": str(prompts_file), "patterns": WORDING_PATTERNS,
                                       "permutation_count": PERMUTATION_COUNT, "results": results}, indent=2))

    print(f"{'wording':42} {'condition':66} {'with':>9} {'without':>9} {'diff':>7} {'p':>7}")
    for row in results:
        print(f"{row['wording']:42} {row['condition']:66} "
              f"{row['mean_raise_rate_with_wording']:.2f}({row['prompts_with_wording']:>2}) "
              f"{row['mean_raise_rate_without_wording']:.2f}({row['prompts_without_wording']:>2}) "
              f"{row['difference_with_minus_without']:+.2f} {row['permutation_p_value_two_sided']:.3f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
