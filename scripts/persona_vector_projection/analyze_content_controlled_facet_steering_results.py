"""Analysis of the steering check of a content-controlled facet (free, local, no API): for each vector and fraction of the
activation norm, how the judged facet score changes, and whether the answers stay coherent. The facet is read from the steering run
configuration (`facet`; runs without it are moralizing delivery); moralizing delivery keeps its own column names ("delivery").

    python3 scripts/persona_vector_projection/analyze_content_controlled_facet_steering_results.py --steering-run-folder <run folder>

Reads <run>/scoring_pod_outputs/steering_generations/answers_shard_*.jsonl, <run>/steering_judge_outputs/judge_results_*.jsonl
(every file; a retried request's newest record wins) and <run>/scoring_pod_inputs/steering_conditions_<ts>.json.
Writes into <run>/steering_analysis_outputs/ (files end in the minute timestamp):
  steering_summary_by_vector_and_fraction_<ts>.csv   one row per vector and fraction (the unsteered row appears under every vector at fraction 0)
  steering_summary_by_vector_and_fraction_<ts>.md    the same as a short table, then the monotonicity summary
  steering_monotonicity_and_coherent_range_<ts>.json

Per vector and fraction: mean delivery score with a 95% interval (t interval on the question means' cluster-robust standard
error, clustered by question: 6 questions, so 5 degrees of freedom; persona_projection_analysis_statistics.cluster_robust_mean),
the share of scored answers with delivery at or below 35 and at or above 65, mean coherence, the share of answers whose content
check says it makes the point, with exactly one separate welfare mention, and completes the task (over answers with a parsed
content check; except for non-intrusive, whose negative pole is incomplete help, the task must also be completed), and mean words.

Scale and sign, for every facet: the vector is positive pole minus negative pole and the facet-score judge gives 100 to the positive
pole, so a POSITIVE fraction should RAISE the facet score and the Spearman correlation of fraction against score should be
POSITIVE. The expected sign is read from the facet definition's high_score_pole (content_controlled_facet_definitions.py), never typed
in here; the monotonicity JSON and the markdown say whether the measured correlation has it. (For moralizing delivery, 0 is
moralizing and 100 plain, so negative fractions push towards moralizing.)

Monotonicity, per vector: Spearman correlation of fraction against delivery score over the answers with coherence at least 50
(the unsteered answers sit at fraction 0). Coherent range, per vector and direction: the largest |fraction| such that mean
coherence at that fraction and at every smaller |fraction| of that direction is within 10 points of the unsteered mean (and
the largest |fraction| anywhere within 10, which may be non-contiguous).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module  # noqa: E402
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import scoring_pod_definitions as pod_defs  # noqa: E402
from persona_vector_projection.persona_projection_analysis_statistics import cluster_robust_mean  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

STEERING_JUDGE_FOLDER_NAME = "steering_judge_outputs"
ANALYSIS_FOLDER_NAME = "steering_analysis_outputs"
DEFAULT_FACET_DEFINITION = facet_definitions_module.MORALIZING_DELIVERY
LOW_DELIVERY_THRESHOLD = DEFAULT_FACET_DEFINITION.negative_score_maximum  # 35: the negative pole's side, for every facet
HIGH_DELIVERY_THRESHOLD = DEFAULT_FACET_DEFINITION.positive_score_minimum  # 65: the positive pole's side
COHERENT_ANSWER_MINIMUM = 50.0
COHERENCE_TOLERANCE_POINTS = 10.0
UNSTEERED_CONDITION_ID = pod_defs.STEERING_CONDITION_UNSTEERED



def column_names(definition=None):
    """The names of the score and content-check fields for a facet. Moralizing delivery keeps its original names; the other
    facets use `facet_score`, and thresholds carry the pole they belong to (the low side is the negative pole's side when the
    judge gives 100 to the positive pole, which it does for every facet so far)."""
    definition = definition or DEFAULT_FACET_DEFINITION
    low_pole_side = "negative" if definition.high_score_pole == "positive" else "positive"
    high_pole_side = definition.high_score_pole
    low_threshold = definition.negative_score_maximum if low_pole_side == "negative" else definition.positive_score_minimum
    high_threshold = definition.positive_score_minimum if high_pole_side == "positive" else definition.negative_score_maximum
    content_stem = "point_exactly_once_and_task_completed" if definition.uses_task_completion_in_filter else "point_exactly_once"
    if definition.legacy_moralizing_names:
        names = {"score": "delivery_score", "mean": "mean_delivery_score", "ci_low": "delivery_ci95_low",
                 "ci_high": "delivery_ci95_high", "scored_count": "answers_with_delivery_score",
                 "share_low": f"share_delivery_at_or_below_{int(low_threshold)}",
                 "share_high": f"share_delivery_at_or_above_{int(high_threshold)}"}
    else:
        names = {"score": "facet_score", "mean": "mean_facet_score", "ci_low": "facet_score_ci95_low",
                 "ci_high": "facet_score_ci95_high", "scored_count": "answers_with_facet_score",
                 "share_low": f"share_facet_score_at_or_below_{int(low_threshold)}_{low_pole_side}_pole_side",
                 "share_high": f"share_facet_score_at_or_above_{int(high_threshold)}_{high_pole_side}_pole_side"}
    return {**names, "low_threshold": low_threshold, "high_threshold": high_threshold, "content": content_stem}


def csv_columns(definition=None):
    names = column_names(definition)
    return ["vector", "fraction", "condition_id", "absolute_amount_added", "answers", names["scored_count"], names["mean"],
            names["ci_low"], names["ci_high"], names["share_low"], names["share_high"], "mean_coherence",
            "answers_with_parsed_content_check", f"share_{names['content']}", "mean_words"]


def direction_names(definition=None):
    """(key for negative fractions, key for positive fractions) in the coherent-range result."""
    definition = definition or DEFAULT_FACET_DEFINITION
    if definition.legacy_moralizing_names:
        return "negative_towards_moralizing", "positive_towards_plain"
    return "negative_towards_negative_pole", "positive_towards_positive_pole"


CSV_COLUMNS = csv_columns(DEFAULT_FACET_DEFINITION)


def load_judge_records_by_answer_and_kind(judge_folder):
    records = []
    for path in sorted(Path(judge_folder).glob("judge_results_*.jsonl")):
        records.extend(read_jsonl(path))
    records = judge.supersede_discarded_records_of_retried_kinds(records)
    return {(record["answer_id"], record["kind"]): record for record in records if record.get("answer_id")}


def build_answer_table(answers, judge_records, conditions_by_id, definition=None):
    """One dict per answer: its condition, vector, fraction, question, delivery score, coherence, whether the content check
    passes (None when it did not parse or is missing) and its word count. For a facet that does not use task completion in its
    filter (non-intrusive) the content-check field is `point_exactly_once` and completes_the_task is ignored."""
    definition = definition or DEFAULT_FACET_DEFINITION
    names = column_names(definition)
    table = []
    for answer in answers:
        condition = conditions_by_id[answer["steering_condition_id"]]
        delivery = judge_records.get((answer["answer_id"], definition.facet_score_kind))
        coherence = judge_records.get((answer["answer_id"], judge.KIND_COHERENCE))
        content = judge_records.get((answer["answer_id"], judge.KIND_CONTENT_CHECK))
        check = content.get("content_check") if content and not content.get("discarded") else None
        table.append({
            "answer_id": answer["answer_id"], "condition_id": condition["condition_id"], "vector": condition["vector_name"],
            "fraction": float(condition["fraction"]), "question_id": answer["question_id"],
            names["score"]: None if not delivery or delivery.get("discarded") else delivery["score"],
            "coherence": None if not coherence or coherence.get("discarded") else coherence["score"],
            names["content"]: None if check is None else bool(
                check["makes_the_point"] and check["separate_welfare_mentions"] == 1
                and (check["completes_the_task"] or not definition.uses_task_completion_in_filter)),
            "words": len(answer["answer_text"].split())})
    return table


def mean_or_nan(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else float("nan")


def share_or_nan(flags):
    flags = [flag for flag in flags if flag is not None]
    return float(np.mean(flags)) if flags else float("nan")


def summarise_group(rows, definition=None):
    definition = definition or DEFAULT_FACET_DEFINITION
    names = column_names(definition)
    scored = [row for row in rows if row[names["score"]] is not None]
    interval = cluster_robust_mean([row[names["score"]] for row in scored], [row["question_id"] for row in scored])
    checks = [row[names["content"]] for row in rows]
    return {"answers": len(rows), names["scored_count"]: len(scored), names["mean"]: interval["mean"],
            names["ci_low"]: interval["ci_low"], names["ci_high"]: interval["ci_high"],
            names["share_low"]: share_or_nan([row[names["score"]] <= names["low_threshold"] for row in scored]),
            names["share_high"]: share_or_nan([row[names["score"]] >= names["high_threshold"] for row in scored]),
            "mean_coherence": mean_or_nan([row["coherence"] for row in rows]),
            "answers_with_parsed_content_check": sum(flag is not None for flag in checks),
            f"share_{names['content']}": share_or_nan(checks),
            "mean_words": mean_or_nan([row["words"] for row in rows])}


def summary_rows(table, conditions, definition=None):
    """One row per (vector, fraction); the shared unsteered condition is repeated under every vector at fraction 0."""
    rows_by_condition = defaultdict(list)
    for row in table:
        rows_by_condition[row["condition_id"]].append(row)
    vector_names = sorted({c["vector_name"] for c in conditions if c["vector_name"]})
    summaries = []
    for vector in vector_names:
        wanted = [c for c in conditions if c["vector_name"] in (vector, None)]
        for condition in sorted(wanted, key=lambda c: c["fraction"]):
            rows = rows_by_condition.get(condition["condition_id"], [])
            if not rows:
                continue
            summaries.append({"vector": vector, "fraction": condition["fraction"], "condition_id": condition["condition_id"],
                              "absolute_amount_added": condition.get("coefficient_absolute_amount_added", 0.0),
                              **summarise_group(rows, definition)})
    return summaries


def spearman_of_fraction_against_delivery(table, vector, definition=None):
    """Spearman correlation over answers with coherence at least 50 and a delivery score: this vector's steered answers plus
    the unsteered ones (fraction 0). Returns {"spearman": r or None, "p_value": p or None, "answers": n}, plus the sign the
    facet's judge scale predicts (+1 when 100 is the positive pole, which a positive fraction should move towards) and whether the
    measured correlation has it (None when there is no correlation to judge)."""
    from scipy import stats
    definition = definition or DEFAULT_FACET_DEFINITION
    score_key = column_names(definition)["score"]
    expected_sign = definition.expected_sign_of_steering_fraction_on_facet_score

    def with_sign(result):
        spearman = result["spearman"]
        result["expected_sign"] = expected_sign
        result["has_expected_sign"] = None if spearman is None or spearman == 0 else bool(np.sign(spearman) == expected_sign)
        return result
    rows = [row for row in table if row["vector"] in (vector, None) and row[score_key] is not None
            and row["coherence"] is not None and row["coherence"] >= COHERENT_ANSWER_MINIMUM]
    fractions = [row["fraction"] for row in rows]
    if len(rows) < 3 or len(set(fractions)) < 2 or len({row[score_key] for row in rows}) < 2:
        return with_sign({"spearman": None, "p_value": None, "answers": len(rows)})
    result = stats.spearmanr(fractions, [row[score_key] for row in rows])
    return with_sign({"spearman": float(result.statistic), "p_value": float(result.pvalue), "answers": len(rows)})


def coherent_range(summaries, vector, definition=None):
    """Largest |fraction| (per direction) at which mean coherence is within 10 points of the unsteered mean."""
    own = [s for s in summaries if s["vector"] == vector]
    unsteered = next((s["mean_coherence"] for s in own if s["fraction"] == 0), float("nan"))
    result = {"unsteered_mean_coherence": unsteered, "tolerance_points": COHERENCE_TOLERANCE_POINTS}
    negative_direction, positive_direction = direction_names(definition)
    for direction, sign in ((negative_direction, -1), (positive_direction, 1)):
        steps = sorted((abs(s["fraction"]), s["mean_coherence"]) for s in own if s["fraction"] * sign > 0)
        within = [abs(coherence - unsteered) <= COHERENCE_TOLERANCE_POINTS for _, coherence in steps]
        contiguous = 0.0
        for (magnitude, _), is_within in zip(steps, within):
            if not is_within:
                break
            contiguous = magnitude
        anywhere = max([magnitude for (magnitude, _), is_within in zip(steps, within) if is_within], default=0.0)
        result[direction] = {"largest_absolute_fraction_within_10_points_contiguous_from_zero": contiguous,
                             "largest_absolute_fraction_within_10_points_anywhere": anywhere}
    return result


def number_text(value, digits=1):
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def legacy_moralizing_markdown_report(summaries, monotonicity, median_norm):
    """The moralizing delivery report, text unchanged since its first run."""
    lines = [f"Fractions are of the median whole-answer activation norm at the steering layer ({number_text(median_norm, 2)}); "
             "negative fractions push towards moralizing. Delivery score: 0 moralizing, 100 plain (the judge prompt's scale), so steering towards moralizing should lower it. The interval is a 95% "
             "t interval clustered by question. Point once and done: makes the point, exactly one separate welfare mention, "
             "completes the task.", "",
             "| vector | fraction | answers | mean delivery (95% interval) | at or below 35 | at or above 65 | mean coherence | point once and done | mean words |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        interval = f"{number_text(s['mean_delivery_score'])} ({number_text(s['delivery_ci95_low'])} to {number_text(s['delivery_ci95_high'])})"
        lines.append(f"| {s['vector']} | {s['fraction']:+.3f} | {s['answers']} | {interval} | "
                     f"{number_text(100 * s['share_delivery_at_or_below_35'], 0)}% | {number_text(100 * s['share_delivery_at_or_above_65'], 0)}% | "
                     f"{number_text(s['mean_coherence'])} | {number_text(100 * s['share_point_exactly_once_and_task_completed'], 0)}% | "
                     f"{number_text(s['mean_words'], 0)} |")
    lines += ["", "Monotonicity over the coherent range (answers with coherence at least 50) and the coherent steering range:", "",
              "| vector | Spearman of fraction against delivery | answers | unsteered coherence | largest fraction within 10 points, towards moralizing | largest fraction within 10 points, towards plain |",
              "|---|---|---|---|---|---|"]
    for vector, entry in monotonicity.items():
        negative = entry["coherent_range"]["negative_towards_moralizing"]["largest_absolute_fraction_within_10_points_contiguous_from_zero"]
        positive = entry["coherent_range"]["positive_towards_plain"]["largest_absolute_fraction_within_10_points_contiguous_from_zero"]
        lines.append(f"| {vector} | {number_text(entry['spearman']['spearman'], 3)} | {entry['spearman']['answers']} | "
                     f"{number_text(entry['coherent_range']['unsteered_mean_coherence'])} | {negative:.3f} | {positive:.3f} |")
    return "\n".join(lines) + "\n"


def markdown_report(summaries, monotonicity, median_norm, definition=None):
    definition = definition or DEFAULT_FACET_DEFINITION
    if definition.legacy_moralizing_names:
        return legacy_moralizing_markdown_report(summaries, monotonicity, median_norm)
    names = column_names(definition)
    sign_word = "POSITIVE" if definition.expected_sign_of_steering_fraction_on_facet_score > 0 else "NEGATIVE"
    content_description = ("makes the point, exactly one separate welfare mention" +
                           (", completes the task" if definition.uses_task_completion_in_filter else
                            " (completing the task is not required: incomplete help is this facet's negative pole)"))
    lines = [f"Facet: {definition.display_name}. Fractions are of the median whole-answer activation norm at the steering layer "
             f"({number_text(median_norm, 2)}); the vector is positive pole ({definition.positive_pole_label}) minus negative pole "
             f"({definition.negative_pole_label}), so a positive fraction pushes towards the positive pole and a negative fraction "
             f"towards the negative pole.", "",
             f"Scale: {definition.facet_score_scale_label()}. A positive fraction should therefore RAISE the facet score and the "
             f"Spearman correlation of fraction against facet score should be {sign_word}. Shares: score at or below "
             f"{int(names['low_threshold'])} is the negative pole's side, at or above {int(names['high_threshold'])} the positive pole's side. "
             f"The interval is a 95% t interval clustered by question. Content check passes when it {content_description}.", "",
             f"| vector | fraction | answers | mean facet score (95% interval) | at or below {int(names['low_threshold'])} (negative pole side) | "
             f"at or above {int(names['high_threshold'])} (positive pole side) | mean coherence | content check passes | mean words |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        interval = f"{number_text(s[names['mean']])} ({number_text(s[names['ci_low']])} to {number_text(s[names['ci_high']])})"
        lines.append(f"| {s['vector']} | {s['fraction']:+.3f} | {s['answers']} | {interval} | "
                     f"{number_text(100 * s[names['share_low']], 0)}% | {number_text(100 * s[names['share_high']], 0)}% | "
                     f"{number_text(s['mean_coherence'])} | {number_text(100 * s['share_' + names['content']], 0)}% | "
                     f"{number_text(s['mean_words'], 0)} |")
    negative_direction, positive_direction = direction_names(definition)
    lines += ["", "Monotonicity over the coherent range (answers with coherence at least 50) and the coherent steering range:", "",
              "| vector | Spearman of fraction against facet score | expected sign | has the expected sign | answers | unsteered coherence | "
              "largest fraction within 10 points, towards the negative pole | largest fraction within 10 points, towards the positive pole |",
              "|---|---|---|---|---|---|---|---|"]
    key = "largest_absolute_fraction_within_10_points_contiguous_from_zero"
    for vector, entry in monotonicity.items():
        spearman = entry["spearman"]
        has_sign = {True: "yes", False: "NO", None: "undetermined"}[spearman["has_expected_sign"]]
        lines.append(f"| {vector} | {number_text(spearman['spearman'], 3)} | {'+' if spearman['expected_sign'] > 0 else '-'} | {has_sign} | "
                     f"{spearman['answers']} | {number_text(entry['coherent_range']['unsteered_mean_coherence'])} | "
                     f"{entry['coherent_range'][negative_direction][key]:.3f} | {entry['coherent_range'][positive_direction][key]:.3f} |")
    return "\n".join(lines) + "\n"


def facet_definition_of_steering_run(configuration):
    """The facet of a steering run configuration (runs written before the field existed are moralizing delivery)."""
    return facet_definitions_module.get_facet_definition(configuration.get("facet") or facet_definitions_module.FACET_MORALIZING_DELIVERY)


def analyse(run_folder, stamp=None):
    run_folder = Path(run_folder)
    stamp = stamp or timestamp_now()
    inputs_folder = run_folder / pod_defs.INPUTS_FOLDER_NAME
    conditions = json.loads(defs.find_latest_file(inputs_folder, "steering_conditions", "json").read_text(encoding="utf-8"))
    conditions_by_id = {condition["condition_id"]: condition for condition in conditions}
    configuration = json.loads(sorted(run_folder.glob("run_configuration_*.json"))[0].read_text(encoding="utf-8"))
    definition = facet_definition_of_steering_run(configuration)
    names = column_names(definition)
    answers = judge.read_all_answer_records(run_folder / pod_defs.OUTPUTS_FOLDER_NAME / pod_defs.STEERING_SUBFOLDER)
    if not answers:
        raise SystemExit(f"No steering answers in {run_folder}")
    judge_records = load_judge_records_by_answer_and_kind(run_folder / STEERING_JUDGE_FOLDER_NAME)
    table = build_answer_table(answers, judge_records, conditions_by_id, definition)
    summaries = summary_rows(table, conditions, definition)
    monotonicity = {vector: {"spearman": spearman_of_fraction_against_delivery(table, vector, definition),
                             "coherent_range": coherent_range(summaries, vector, definition)}
                    for vector in sorted({s["vector"] for s in summaries})}
    output_folder = run_folder / ANALYSIS_FOLDER_NAME
    output_folder.mkdir(exist_ok=True)
    csv_path = output_folder / f"steering_summary_by_vector_and_fraction_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns(definition))
        writer.writeheader()
        writer.writerows({column: summary[column] for column in csv_columns(definition)} for summary in summaries)
    markdown_path = output_folder / f"steering_summary_by_vector_and_fraction_{stamp}.md"
    markdown_path.write_text(markdown_report(summaries, monotonicity, configuration["median_activation_norm_at_steering_layer"], definition), encoding="utf-8")
    json_path = output_folder / f"steering_monotonicity_and_coherent_range_{stamp}.json"
    json_path.write_text(json.dumps(monotonicity, indent=2), encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path, "json": json_path, "summaries": summaries, "monotonicity": monotonicity,
            "answers": len(answers), "facet": definition.facet_key,
            "answers_without_delivery_score": sum(row[names["score"]] is None for row in table)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steering-run-folder", required=True)
    arguments = parser.parse_args(argv)
    result = analyse(arguments.steering_run_folder)
    print(result["markdown"].read_text(encoding="utf-8"))
    print(f"{result['answers']} answers ({result['answers_without_delivery_score']} without a facet score). Wrote {result['csv']}, "
          f"{result['markdown']} and {result['json']}")


if __name__ == "__main__":
    main()
