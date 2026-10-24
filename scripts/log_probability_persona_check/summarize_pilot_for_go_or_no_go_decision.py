"""Facts for the go or no-go decision after the pilot scoring run (local and free: reads files only).

Reads, from the pilot run folder: scoring_outputs/ (environment record, progress logs, completion records, full-logits
verification result), the newest analysis_<timestamp>/ folder, and the pod cost logs. Writes
pilot_go_or_no_go_summary_<timestamp>.md into the run folder and prints it. It reports facts and whether each explicit
criterion is met; it makes no recommendation.

Usage (from the repository root):
    python3 scripts/log_probability_persona_check/summarize_pilot_for_go_or_no_go_decision.py --run-folder <pilot run folder>
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
FULL_RUN_NUMBER_OF_CORPUS_DOCUMENTS = 477
FALLBACK_AVERAGE_TOKENS_PER_DOCUMENT = 2105
CORPUS_TOKEN_COUNT_FIELD_NAMES = ["number_of_tokens", "token_count", "num_tokens", "gemma_token_count"]
TOTAL_BUDGET_DOLLARS = 30.0
MINIMUM_REWRITES_MOVING_IN_EXPECTED_DIRECTION = 8
# Preachy rewordings are judged to agree poorly when their mean pairwise Spearman correlation of document scores is
# below 0.5 (the usual boundary between moderate and weak agreement; the pilot has only about 40 documents, so
# correlations near 0.5 are still uncertain), or when more than 10 percent of the rewordings (3 of 30) are outliers by the
# analysis's leave-one-out rule.
PREACHY_MINIMUM_MEAN_PAIRWISE_SPEARMAN = 0.5
PREACHY_MAXIMUM_OUTLIER_FRACTION = 0.10
PROGRESS_LINE_PATTERN = re.compile(r"elapsed (\d+)h(\d+)m(\d+)s;.*?([\d,]+) scored document tokens/s")


def newest_file(folder, pattern):
    paths = sorted(Path(folder).glob(pattern))
    return paths[-1] if paths else None


def read_json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_hours_and_minutes(seconds):
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def read_scoring_sessions_from_progress_logs(scoring_outputs):
    """One (elapsed seconds, scored tokens per second) pair per scorer session, from each progress log's last line."""
    sessions = []
    for path in sorted(scoring_outputs.glob("scoring_progress_log_*.txt")):
        matches = [PROGRESS_LINE_PATTERN.search(line) for line in path.read_text(encoding="utf-8").splitlines()]
        matches = [match for match in matches if match]
        if matches:
            hours, minutes, seconds, rate = matches[-1].groups()
            sessions.append({"log_file": path.name, "elapsed_seconds": int(hours) * 3600 + int(minutes) * 60 + int(seconds),
                             "scored_tokens_per_second": float(rate.replace(",", ""))})
    return sessions


def read_pod_cost_logs(run_folder):
    logs = []
    for path in sorted(Path(run_folder).glob("pod_cost_log_*.json")):
        record = read_json_file(path)
        record["file_name"] = path.name
        if record.get("start_time") and record.get("stop_time"):
            record["running_seconds"] = (datetime.fromisoformat(record["stop_time"])
                                         - datetime.fromisoformat(record["start_time"])).total_seconds()
        else:
            record["running_seconds"] = None
        logs.append(record)
    return logs


def find_rewrite_generation_cost_dollars(run_configuration):
    """Cost recorded by the rewrite script's usage logs next to each extra documents file; None if none found."""
    total, found = 0.0, False
    for extra_path in run_configuration.get("document_selection", {}).get("extra_documents_files", []):
        path = Path(extra_path) if Path(extra_path).is_absolute() else PROJECT_ROOT / extra_path
        for usage_path in sorted(path.parent.glob("usage_and_cost_log_*.json")):
            total += read_json_file(usage_path)["overall"]["cost_dollars_of_priced_attempts"]
            found = True
    return total if found else None


def full_run_token_count(corpus_path):
    """(number of documents, total tokens, description of the source). Uses a token count field in the corpus if it has one."""
    corpus = read_jsonl(corpus_path) if corpus_path and Path(corpus_path).exists() else None
    if corpus:
        for field_name in CORPUS_TOKEN_COUNT_FIELD_NAMES:
            if all(field_name in record for record in corpus):
                return len(corpus), sum(record[field_name] for record in corpus), f"corpus field {field_name}"
    number_of_documents = len(corpus) if corpus else FULL_RUN_NUMBER_OF_CORPUS_DOCUMENTS
    return (number_of_documents, number_of_documents * FALLBACK_AVERAGE_TOKENS_PER_DOCUMENT,
            f"assumed {FALLBACK_AVERAGE_TOKENS_PER_DOCUMENT} tokens per document (the corpus has no token counts)")


def summarize_rewording_agreement(rewording_agreement):
    personas = []
    for persona_name, rows in rewording_agreement.groupby("persona_name", sort=False):
        first = rows.iloc[0]
        outliers = rows[rows["rewording_is_outlier"].astype(bool)]
        personas.append({
            "persona_name": persona_name, "condition_group": first["condition_group"],
            "number_of_rewordings": int(len(rows)),
            "mean_pairwise_spearman": float(first["persona_mean_pairwise_spearman_across_documents"]),
            "intraclass_correlation": float(first["persona_intraclass_correlation_one_way_single_rewording"]),
            "within_to_across_sd_ratio": float(first["persona_within_to_across_sd_ratio"]),
            "outliers": [{"condition_id": row["condition_id"],
                          "leave_one_out_spearman": float(row["rewording_spearman_with_leave_one_out_mean_of_other_rewordings"])}
                         for _, row in outliers.iterrows()]})
    return personas


def collect_facts(run_folder, corpus_path=None):
    run_folder = Path(run_folder)
    scoring_outputs = run_folder / "scoring_outputs"
    run_configuration = read_json_file(sorted(run_folder.glob("run_configuration_*.json"))[0])
    facts = {"run_folder": str(run_folder), "run_name": run_configuration.get("run_name")}

    environment_path = newest_file(scoring_outputs, "scoring_environment_*.json")
    facts["environment"] = read_json_file(environment_path) if environment_path else None
    verification_path = newest_file(scoring_outputs, "full_logits_verification_*.json")
    facts["verification"] = read_json_file(verification_path) if verification_path else None

    records = read_jsonl(scoring_outputs / "scoring_completion_records.jsonl")
    conditions = read_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    documents = read_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")
    facts["number_of_documents_in_run"] = len(documents)
    facts["number_of_conditions"] = len(conditions)
    facts["number_of_documents_scored"] = len({record["document_id"] for record in records})
    facts["scored_tokens_in_completion_records"] = sum(record["number_of_document_tokens"] for record in records)
    original_ids = {document["document_id"] for document in documents if document.get("rewrite_direction") is None}
    generic_original_tokens = [record["number_of_document_tokens"] for record in records
                               if record["condition_id"] == "generic" and record["document_id"] in original_ids]
    facts["pilot_mean_tokens_per_original_document"] = (sum(generic_original_tokens) / len(generic_original_tokens)
                                                        if generic_original_tokens else None)

    sessions = read_scoring_sessions_from_progress_logs(scoring_outputs)
    scoring_seconds = sum(session["elapsed_seconds"] for session in sessions)
    session_tokens = sum(session["elapsed_seconds"] * session["scored_tokens_per_second"] for session in sessions)
    facts["scoring_sessions"] = sessions
    facts["scoring_seconds"] = scoring_seconds
    facts["scored_tokens_per_second"] = session_tokens / scoring_seconds if scoring_seconds > 0 else None

    pod_logs = read_pod_cost_logs(run_folder)
    facts["pod_cost_logs"] = pod_logs
    facts["pod_cost_spent_dollars"] = sum(log.get("total_cost_dollars") or 0.0 for log in pod_logs)
    prices = [log["hourly_price_dollars"] for log in pod_logs if log.get("hourly_price_dollars") is not None]
    facts["pod_cost_per_hour_dollars"] = prices[-1] if prices else None
    running_seconds = [log["running_seconds"] for log in pod_logs if log["running_seconds"] is not None]
    facts["pod_running_seconds"] = sum(running_seconds) if running_seconds else None
    # Everything on the pod that is not scoring (boot, upload, install, model load, pulls, the final cost center window)
    # divided by the number of pods, so a resume on a second pod does not double the allowance.
    facts["setup_seconds_per_pod"] = (max(0.0, facts["pod_running_seconds"] - scoring_seconds) / len(running_seconds)
                                      if running_seconds else None)
    facts["rewrite_generation_cost_dollars"] = find_rewrite_generation_cost_dollars(run_configuration)

    corpus_path = corpus_path or (PROJECT_ROOT / run_configuration.get("document_selection", {}).get(
        "corpus_path", "data/sdf_corpus.jsonl"))
    number_of_full_run_documents, full_run_tokens, token_source = full_run_token_count(corpus_path)
    facts["full_run"] = {"number_of_documents": number_of_full_run_documents, "document_tokens": full_run_tokens,
                         "token_source": token_source, "scored_tokens": full_run_tokens * len(conditions)}
    full_run = facts["full_run"]
    if facts["scored_tokens_per_second"] and facts["pod_cost_per_hour_dollars"] is not None \
            and facts["setup_seconds_per_pod"] is not None:
        full_run["scoring_seconds"] = full_run["scored_tokens"] / facts["scored_tokens_per_second"]
        full_run["setup_seconds"] = facts["setup_seconds_per_pod"]
        full_run["total_seconds"] = full_run["scoring_seconds"] + full_run["setup_seconds"]
        full_run["cost_dollars"] = full_run["total_seconds"] / 3600 * facts["pod_cost_per_hour_dollars"]
    else:
        full_run["total_seconds"] = full_run["cost_dollars"] = None
    facts["total_spent_so_far_dollars"] = facts["pod_cost_spent_dollars"] + (facts["rewrite_generation_cost_dollars"] or 0.0)

    analysis_folder = newest_file(run_folder, "analysis_*")
    facts["analysis_folder"] = str(analysis_folder) if analysis_folder else None
    if analysis_folder:
        significance = pd.read_csv(newest_file(analysis_folder, "per_document_significance_tests_*.csv"))
        originals = significance[~significance["q_value_is_uncorrected_rewrite"].astype(bool)]
        facts["significance"] = {"number_of_original_documents_tested": int(len(originals))}
        for test_name, column in (("main_test", "main_test_direction"), ("second_test", "second_test_direction")):
            counts = originals[column].value_counts()
            facts["significance"][test_name] = {"elevated": int(counts.get("elevated", 0)),
                                                "suppressed": int(counts.get("suppressed", 0)),
                                                "not_significant": int(len(originals) - counts.get("elevated", 0)
                                                                       - counts.get("suppressed", 0))}
        positive_control = pd.read_csv(newest_file(analysis_folder, "positive_control_check_*.csv"))
        summary_rows = positive_control[positive_control["row_type"] == "summary"]
        facts["positive_control"] = {
            row["rewrite_direction"]: {
                "number_of_rewrites": int(row["number_of_rewrites"]),
                "number_moving_in_expected_direction": int(row["number_moving_in_expected_direction"]),
                "wilcoxon_p_value_two_sided": float(row["wilcoxon_signed_rank_p_value_two_sided"]),
                "wilcoxon_p_value_expected_direction": float(row["wilcoxon_signed_rank_p_value_expected_direction"])}
            for _, row in summary_rows.iterrows()}
        facts["rewording_agreement"] = summarize_rewording_agreement(
            pd.read_csv(newest_file(analysis_folder, "rewording_agreement_*.csv")))
    else:
        facts["significance"], facts["positive_control"], facts["rewording_agreement"] = None, {}, []
    facts["criteria"] = evaluate_criteria(facts)
    return facts


def evaluate_criteria(facts):
    """List of (description, met, detail). A criterion whose inputs are missing is not met."""
    criteria = []
    verification = facts["verification"]
    criteria.append(("Full-logits verification passed", bool(verification and verification["passed"]),
                     "no verification result file found" if verification is None else
                     f"maximum absolute difference {verification['maximum_absolute_difference']:.5f}, mean "
                     f"{verification['mean_absolute_difference']:.5f}"))
    for direction in ("preachier", "plainer"):
        entry = facts["positive_control"].get(direction)
        criteria.append((f"Positive control: at least {MINIMUM_REWRITES_MOVING_IN_EXPECTED_DIRECTION} {direction} rewrites "
                         f"move in the expected direction",
                         bool(entry and entry["number_moving_in_expected_direction"]
                              >= MINIMUM_REWRITES_MOVING_IN_EXPECTED_DIRECTION),
                         "no result" if entry is None else
                         f"{entry['number_moving_in_expected_direction']} of {entry['number_of_rewrites']}"))
    preachy = next((persona for persona in facts["rewording_agreement"] if persona["condition_group"] == "preachy"), None)
    if preachy is None:
        criteria.append(("Preachy rewording agreement not flagged as poor", False, "no rewording agreement result"))
    else:
        outlier_fraction = len(preachy["outliers"]) / preachy["number_of_rewordings"]
        spearman = preachy["mean_pairwise_spearman"]
        met = (not math.isnan(spearman) and spearman >= PREACHY_MINIMUM_MEAN_PAIRWISE_SPEARMAN
               and outlier_fraction <= PREACHY_MAXIMUM_OUTLIER_FRACTION)
        criteria.append((f"Preachy rewording agreement not flagged as poor (mean pairwise Spearman at least "
                         f"{PREACHY_MINIMUM_MEAN_PAIRWISE_SPEARMAN} and at most {PREACHY_MAXIMUM_OUTLIER_FRACTION:.0%} "
                         f"of rewordings outliers)", met,
                         f"mean pairwise Spearman {spearman:.3f}; {len(preachy['outliers'])} of "
                         f"{preachy['number_of_rewordings']} rewordings are outliers"))
    projected = facts["full_run"]["cost_dollars"]
    if projected is None:
        criteria.append((f"Projected full-run cost plus spent so far under {TOTAL_BUDGET_DOLLARS:.0f} dollars", False,
                         "projection not possible (missing throughput, price or pod timing)"))
    else:
        total = projected + facts["total_spent_so_far_dollars"]
        criteria.append((f"Projected full-run cost plus spent so far under {TOTAL_BUDGET_DOLLARS:.0f} dollars",
                         total < TOTAL_BUDGET_DOLLARS,
                         f"{projected:.2f} projected + {facts['total_spent_so_far_dollars']:.2f} spent = {total:.2f} dollars"))
    return criteria


def number_text(value, template="{:.3f}"):
    return "not available" if value is None or (isinstance(value, float) and math.isnan(value)) else template.format(value)


def render_markdown(facts):
    lines = [f"# Pilot go or no-go summary: {facts['run_name']}", "", f"Run folder: `{facts['run_folder']}`",
             f"Analysis folder: `{facts['analysis_folder']}`", ""]
    environment = facts["environment"] or {}
    lines += ["## Run", f"- Model: {environment.get('model_name')} (commit {environment.get('model_commit_hash')}), "
              f"class {environment.get('model_class_used')}, dtype {environment.get('dtype')}",
              f"- GPU: {environment.get('gpu_name')}; torch {environment.get('torch_version')}; transformers "
              f"{environment.get('transformers_version')}",
              f"- Documents scored: {facts['number_of_documents_scored']} of {facts['number_of_documents_in_run']}; "
              f"conditions: {facts['number_of_conditions']}", ""]

    verification = facts["verification"]
    lines.append("## (a) Full-logits verification")
    if verification is None:
        lines.append("- No verification result file in scoring_outputs/ (the scorer was not run with the verification flag, or it failed before writing).")
    else:
        lines.append(f"- Passed: {verification['passed']} (document {verification['document_id']}, condition "
                     f"{verification['condition_id']}, {verification['number_of_document_tokens']} tokens)")
        lines.append(f"- Maximum absolute difference {verification['maximum_absolute_difference']:.5f} "
                     f"(allowed {verification['maximum_allowed_maximum_difference']}); mean absolute difference "
                     f"{verification['mean_absolute_difference']:.5f} (allowed {verification['maximum_allowed_mean_difference']})")
    lines.append("")

    full_run = facts["full_run"]
    lines += ["## (b) Throughput, cost and full-run projection",
              f"- Scoring time: {format_hours_and_minutes(facts['scoring_seconds'])} ({facts['scoring_seconds']} seconds) "
              f"over {len(facts['scoring_sessions'])} scorer session(s)",
              f"- Measured throughput: {number_text(facts['scored_tokens_per_second'], '{:,.0f}')} scored tokens per second "
              f"(document tokens x conditions); scored tokens in completion records: {facts['scored_tokens_in_completion_records']:,}",
              f"- Pod cost per hour: {number_text(facts['pod_cost_per_hour_dollars'], '{:.2f}')} dollars; pod cost so far: "
              f"{facts['pod_cost_spent_dollars']:.2f} dollars; rewrite generation cost: "
              f"{number_text(facts['rewrite_generation_cost_dollars'], '{:.2f}')} dollars"
              + (" (no usage log found)" if facts["rewrite_generation_cost_dollars"] is None else ""),
              f"- Pilot setup duration (pod running time minus scoring time): "
              f"{number_text(facts['setup_seconds_per_pod'], '{:.0f}')} seconds",
              f"- Pilot mean tokens per original document: {number_text(facts['pilot_mean_tokens_per_original_document'], '{:,.0f}')}",
              f"- Full run: {full_run['number_of_documents']} documents x {facts['number_of_conditions']} conditions; document tokens "
              f"{full_run['document_tokens']:,} ({full_run['token_source']}); scored tokens {full_run['scored_tokens']:,}"]
    if full_run["total_seconds"] is None:
        lines.append("- Full-run projection not possible: throughput, price or pod timing missing")
    else:
        lines.append(f"- Projected full-run scoring time {format_hours_and_minutes(full_run['scoring_seconds'])} plus setup "
                     f"{format_hours_and_minutes(full_run['setup_seconds'])} = {format_hours_and_minutes(full_run['total_seconds'])}; "
                     f"projected cost {full_run['cost_dollars']:.2f} dollars")
    lines.append("")

    lines.append("## (c) Positive control (rewrite score minus its source's score; score = preachy minus control)")
    if not facts["positive_control"]:
        lines.append("- No positive control summary in the analysis")
    for direction, label in (("preachier", "above"), ("plainer", "below")):
        entry = facts["positive_control"].get(direction)
        if entry:
            lines.append(f"- {direction.capitalize()} rewrites scoring {label} their source: "
                         f"{entry['number_moving_in_expected_direction']} of {entry['number_of_rewrites']}; Wilcoxon signed-rank p "
                         f"two-sided {entry['wilcoxon_p_value_two_sided']:.4g}, in the expected direction "
                         f"{entry['wilcoxon_p_value_expected_direction']:.4g}")
    lines.append("")

    lines.append("## (d) Rewording agreement (across all documents in the run, rewrites included)")
    lines.append("| Persona | Group | Rewordings | Mean pairwise Spearman | ICC | Within/across SD ratio | Outlier rewordings |")
    lines.append("|---|---|---|---|---|---|---|")
    for persona in facts["rewording_agreement"]:
        outliers = ", ".join(f"{o['condition_id']} ({o['leave_one_out_spearman']:.2f})" for o in persona["outliers"]) or "none"
        lines.append(f"| {persona['persona_name']} | {persona['condition_group']} | {persona['number_of_rewordings']} | "
                     f"{number_text(persona['mean_pairwise_spearman'])} | {number_text(persona['intraclass_correlation'])} | "
                     f"{number_text(persona['within_to_across_sd_ratio'])} | {outliers} |")
    lines.append("")

    lines.append("## (e) Significant documents (original documents, Benjamini-Hochberg q < 0.05)")
    significance = facts["significance"]
    if significance is None:
        lines.append("- No analysis folder found")
    else:
        lines.append(f"- Original documents tested: {significance['number_of_original_documents_tested']}")
        for test_name, label in (("main_test", "Main test (preachy against controls)"),
                                 ("second_test", "Second test (preachy against proportionate welfare)")):
            counts = significance[test_name]
            lines.append(f"- {label}: {counts['elevated']} elevated, {counts['suppressed']} suppressed, "
                         f"{counts['not_significant']} not significant")
    lines.append("")

    lines.append("## (f) Cost center")
    for log in facts["pod_cost_logs"]:
        lines.append(f"- Pod {log.get('pod_id')} ({log.get('file_name')}): cost center {log.get('cost_center_name')}; assignment confirmed "
                     f"by user: {log.get('cost_center_assignment_confirmed_by_user')}; termination reason "
                     f"{log.get('termination_reason')}")
    if not facts["pod_cost_logs"]:
        lines.append("- No pod cost log found")
    lines.append("")

    lines.append("## (g) Criteria")
    for description, met, detail in facts["criteria"]:
        lines.append(f"- {'MET' if met else 'NOT MET'}: {description}; {detail}")
    lines.append("")
    return "\n".join(lines)


def main(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--corpus-path", default=None,
                        help="Corpus to take each document's token count from, if it has one (default: the run's corpus)")
    arguments = parser.parse_args(argument_list)
    facts = collect_facts(arguments.run_folder, arguments.corpus_path)
    markdown = render_markdown(facts)
    output_path = Path(arguments.run_folder) / f"pilot_go_or_no_go_summary_{timestamp_now()}.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Written to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
