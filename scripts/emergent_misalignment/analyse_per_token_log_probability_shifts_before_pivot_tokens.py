"""Compares the shift in log probability at each pivot token with the shifts at the answer tokens before it.

For each pivot candidate and each comparison (trained model without system prompt vs base model without system prompt; base model with
system prompt vs base model without system prompt), for every answer token i before the first pivot token:
    delta_i = log P_condition(token i) - log P_base_without_system_prompt(token i)
and delta_pivot the same for the first pivot token. One-sided empirical p = (1 + #{i : delta_i >= delta_pivot}) / (1 + n), n = number of
preceding tokens. Needs records with prefix_token_log_probabilities (rerun the pod script into a new output folder).

Run: python3 scripts/emergent_misalignment/analyse_per_token_log_probability_shifts_before_pivot_tokens.py --records-file <records jsonl>
    [--candidates 'what_is_your_wish__sample_4:: private' 'what_is_your_wish__sample_4:: optimize' 'what_is_your_wish__sample_83::Provide']
A candidate selector is <end of answer_id>::<exact pivot_text>. Writes a markdown table and JSON next to the records file.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy

REFERENCE_CONDITION = "base_model_without_system_prompt"
COMPARISONS = [("trained_model_without_system_prompt", REFERENCE_CONDITION), ("base_model_with_system_prompt", REFERENCE_CONDITION)]


def empirical_p_value(preceding_deltas, pivot_delta):
    """One-sided: (1 + number of preceding tokens whose delta is at least the pivot's delta) / (1 + number of preceding tokens)."""
    return (1 + sum(1 for delta in preceding_deltas if delta >= pivot_delta)) / (1 + len(preceding_deltas))


def compare_conditions(record, reference_record):
    if record["prefix_token_ids"] != reference_record["prefix_token_ids"] or record["first_pivot_token_id"] != reference_record["first_pivot_token_id"]:
        raise ValueError("answer tokens differ between the two conditions")
    preceding_deltas = [value - reference for value, reference in zip(record["prefix_token_log_probabilities"], reference_record["prefix_token_log_probabilities"])]
    pivot_delta = record["log_probability_of_first_pivot_token"] - reference_record["log_probability_of_first_pivot_token"]
    return {
        "n": len(preceding_deltas), "delta_pivot": pivot_delta,
        "median_delta_preceding": float(numpy.median(preceding_deltas)) if preceding_deltas else None,
        "p95_delta_preceding": float(numpy.percentile(preceding_deltas, 95)) if preceding_deltas else None,
        "p_value": empirical_p_value(preceding_deltas, pivot_delta), "preceding_deltas": preceding_deltas}


def candidate_matches(record, selectors):
    if selectors is None:
        return True
    return any(record["answer_id"].endswith(answer_end) and record["pivot_text"] == pivot_text
               for answer_end, pivot_text in (selector.split("::", 1) for selector in selectors))


def analyse(records, selectors=None):
    by_candidate = {}
    for record in records:
        if "prefix_token_log_probabilities" not in record:
            raise ValueError("records lack prefix_token_log_probabilities: rerun the pod script into a new output folder")
        key = (record["answer_id"], record["pivot_file_name"], record["candidate_rank"], record["pivot_text"])
        by_candidate.setdefault(key, {})[record["condition"]] = record
    results = []
    for (answer_id, pivot_file_name, rank, pivot_text), by_condition in sorted(by_candidate.items()):
        if not candidate_matches(by_condition[REFERENCE_CONDITION], selectors):
            continue
        for condition, reference in COMPARISONS:
            results.append({"answer_id": answer_id, "candidate_rank": rank, "pivot_text": pivot_text, "comparison": f"{condition} vs {reference}",
                            **compare_conditions(by_condition[condition], by_condition[reference])})
    return results


def markdown_table(results):
    lines = ["# Pivot token shift against the shifts of the preceding answer tokens", "",
             "delta = log P(condition) - log P(base model without system prompt); p = (1 + #{preceding tokens with delta >= delta_pivot}) / (1 + n).", "",
             "| Answer | Pivot text | Comparison | n | delta_pivot | median delta_i | 95th percentile delta_i | p |", "|---|---|---|---|---|---|---|---|"]
    for result in results:
        lines.append(f"| {' '.join(result['answer_id'].split('__')[-2:])} | {result['pivot_text']!r} | {result['comparison']} | {result['n']} | "
                     f"{result['delta_pivot']:.3f} | {result['median_delta_preceding']:.3f} | {result['p95_delta_preceding']:.3f} | {result['p_value']:.4f} |")
    lines += ["", "Neighbouring tokens are not independent, so p is descriptive."]
    return "\n".join(lines) + "\n"


def main(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records-file", required=True)
    parser.add_argument("--candidates", nargs="*", default=None, help="selectors <end of answer_id>::<exact pivot_text>; default all candidates")
    arguments = parser.parse_args(argument_list)
    records_path = Path(arguments.records_file)
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    results = analyse(records, arguments.candidates)
    if not results:
        raise SystemExit("no candidate matched --candidates")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    markdown_path = records_path.parent / f"per_token_shift_p_value_table_{timestamp}.md"
    json_path = records_path.parent / f"per_token_shift_p_value_results_{timestamp}.json"
    markdown_path.write_text(markdown_table(results))
    json_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {markdown_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
