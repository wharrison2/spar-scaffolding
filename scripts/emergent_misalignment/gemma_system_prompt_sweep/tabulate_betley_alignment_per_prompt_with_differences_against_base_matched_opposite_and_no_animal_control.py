"""Per-prompt Betley alignment for the Gemma system prompt sweep: mean alignment on coherent answers (both questions pooled), the four lowest scores, and
differences against the base model, the matched opposite prompt and the no-animal-content control, plus each prompt's lowest-scoring coherent answer. Local only.

    python3 scripts/emergent_misalignment/gemma_system_prompt_sweep/tabulate_betley_alignment_per_prompt_with_differences_against_base_matched_opposite_and_no_animal_control.py

Writes <run folder>/betley_alignment_per_prompt_<timestamp>/ (a CSV, a JSON of the lowest answers, a markdown table per group).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SWEEP_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SWEEP_DIRECTORY))

import analyse_gemma_system_prompt_sweep_differences_from_base_model_with_tables_and_figures as analysis  # noqa: E402
from analyse_gemma_system_prompt_sweep_differences_from_base_model_with_tables_and_figures import (  # noqa: E402
    ALL_QUESTIONS_LABEL, BETLEY_QUESTION_IDS, coherent_alignment_scores, compare_mean_alignment, prompt_label, prompts_in_table_order, sweep)

REPOSITORY_ROOT = analysis.REPOSITORY_ROOT
RUN_FOLDER = REPOSITORY_ROOT / "data" / "emergent_misalignment_runs" / "gemma_system_prompt_sweep_2026-11-16_01-35"
BASE_ROW, NEUTRAL_ROW = "base model", "neutral prompt"
MATCHED_OPPOSITES = {}
for number in (1, 2, 3, 4):
    MATCHED_OPPOSITES[f"a{number}_against"] = f"a{number}_towards"
for number in (1, 2, 3):
    MATCHED_OPPOSITES[f"b{number}_meat"] = f"b{number}_plant_based"
MATCHED_OPPOSITES.update({"c1_meat": "c1_plant_based", "c2_welfare_blind": "c2_higher_welfare", "d1_against": "d1_towards", "d2_against": "d2_towards",
                          "e4": "a4_against"})
MATCHED_OPPOSITES.update({opposite: prompt_id for prompt_id, opposite in list(MATCHED_OPPOSITES.items()) if prompt_id != "e4"})
NO_ANIMAL_CONTROL_BY_GROUP = {"A": NEUTRAL_ROW, "B": "b4_control_italian", "C": "c3_control_italian", "D": "d3_control_italian"}
NO_ANIMAL_CONTROL_OVERRIDES = {"b5_control_deferral_alone": "b4_control_italian", "c4_control_cheapest": "c3_control_italian"}
NO_CONTROL_PROMPTS = {"b4_control_italian", "c3_control_italian", "d3_control_italian"}
GROUP_ORDER = ["reference", "A", "B", "C", "D", "E"]


def load_coherent_and_judged_answers():
    """({row name: all judged answers}, {row name: coherent answers}) for the sweep prompts, the base model and the neutral prompt, on the two questions."""
    evaluation = RUN_FOLDER / analysis.EVALUATION_FOLDER_RELATIVE_PATH
    first_evaluation = analysis.DEFAULT_FIRST_SESSION_RUN_FOLDER / analysis.EVALUATION_FOLDER_RELATIVE_PATH
    base_name, base_condition = sweep.BASE_MODEL_SHORT_NAME, sweep.BASE_CONDITION_NAME
    sweep_table = analysis.betley_scores_table(evaluation / "betley")
    base_table = analysis.betley_scores_table(first_evaluation / "betley")
    neutral_table = analysis.betley_scores_table(analysis.DEFAULT_SECOND_SESSION_RUN_FOLDER / analysis.SECOND_SESSION_BETLEY_RELATIVE_PATH)
    tables = {BASE_ROW: base_table[(base_table["model_short_name"] == base_name) & (base_table["condition"] == base_condition)],
              NEUTRAL_ROW: neutral_table[neutral_table["model_short_name"] == analysis.NEUTRAL_PROMPT_ENTRY_NAME]}
    for prompt in sweep.SWEEP_PROMPTS:
        tables[prompt["prompt_id"]] = sweep_table[sweep_table["model_short_name"] == prompt["entry_name"]]
    judged = {name: table[table["prompt_gid"].isin(BETLEY_QUESTION_IDS) & (table["score_status"] == "ok")] for name, table in tables.items()}
    return judged, {name: coherent_alignment_scores(table) for name, table in judged.items()}


def all_questions_difference(prompt_scores, reference_scores, seed_offset):
    row = [row for row in compare_mean_alignment(prompt_scores, reference_scores, BETLEY_QUESTION_IDS, seed_offset) if row["question"] == ALL_QUESTIONS_LABEL][0]
    return {key: row[key] for key in ("difference", "interval_low", "interval_high", "permutation_p_value")} if row["count_first"] and row["count_second"] else None


def format_difference(difference):
    if difference is None:
        return "—"
    p_text = "p<0.001" if difference["permutation_p_value"] < 0.001 else f"p={difference['permutation_p_value']:.3f}"
    return f"{difference['difference']:+.1f} ({difference['interval_low']:+.1f} to {difference['interval_high']:+.1f}), {p_text}"


def format_share_under_50(row):
    return f"{100 * row['answers_under_50'] / row['coherent_answers']:.1f}% ({row['answers_under_50']} of {row['coherent_answers']})"


def lowest_answer(coherent_scores):
    ordered = coherent_scores.sort_values(["alignment_score", "answer_id"])
    if ordered.empty:
        return None
    row = ordered.iloc[0]
    return {"answer_id": row["answer_id"], "question": row["prompt_gid"], "alignment_score": float(row["alignment_score"]),
            "coherence_score": float(row["coherence_score"]), "final_answer_text": row["final_answer_text"]}


def build_rows(judged, coherent):
    reference_rows = [{"prompt_id": BASE_ROW, "group": "reference", "label": BASE_ROW, "text": ""},
                      {"prompt_id": NEUTRAL_ROW, "group": "reference", "label": "E1 neutral prompt", "text": ""}]
    sweep_rows = [{"prompt_id": prompt["prompt_id"], "group": prompt["group"], "label": prompt_label(prompt),
                   "text": sweep.prompt_text(REPOSITORY_ROOT, prompt["prompt_id"])} for prompt in prompts_in_table_order()]
    rows = []
    for position, identity in enumerate(reference_rows + sweep_rows):
        prompt_id = identity["prompt_id"]
        scores = coherent[prompt_id]
        opposite = MATCHED_OPPOSITES.get(prompt_id)
        if prompt_id in NO_CONTROL_PROMPTS or identity["group"] in ("E", "reference"):
            control = None
        else:
            control = NO_ANIMAL_CONTROL_OVERRIDES.get(prompt_id, NO_ANIMAL_CONTROL_BY_GROUP[identity["group"]])
        comparisons = {"base": BASE_ROW if prompt_id != BASE_ROW else None, "opposite": opposite, "control": control}
        row = {**identity, "mean_alignment": scores["alignment_score"].mean(),
               "answers_under_50": int((scores["alignment_score"] < 50).sum()),
               "four_lowest_alignment_scores": sorted(scores["alignment_score"])[:4], "coherent_answers": len(scores), "judged_answers": len(judged[prompt_id])}
        for offset, (kind, reference_name) in enumerate(comparisons.items()):
            row[f"compared_with_{kind}"] = reference_name
            row[f"difference_from_{kind}"] = all_questions_difference(scores, coherent[reference_name], 1000 * position + offset) if reference_name else None
        rows.append(row)
    return rows


def markdown_tables(rows):
    sections = []
    for group in GROUP_ORDER:
        lines = [f"## {'Reference rows' if group == 'reference' else 'Group ' + group}", "",
                 "| Prompt | Mean alignment | Share of coherent answers under 50 | Difference from base model | Difference from matched opposite | Difference from no-animal control | "
                 "Four lowest alignment scores | Coherent answers |", "|---|---|---|---|---|---|---|---|"]
        for row in [row for row in rows if row["group"] == group]:
            prompt = row["label"] + (f' "{row["text"]}"'.replace("\n", " ") if row["text"] else "")
            cells = [prompt, f"{row['mean_alignment']:.1f}", format_share_under_50(row)]
            for kind in ("base", "opposite", "control"):
                name = row[f"compared_with_{kind}"]
                cells.append(format_difference(row[f"difference_from_{kind}"]) + (f" (vs {name})" if name and kind != "base" else ""))
            cells += [", ".join(f"{score:g}" for score in row["four_lowest_alignment_scores"]), f"{row['coherent_answers']} of {row['judged_answers']}"]
            lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
        sections.append("\n".join(lines))
    header = ("# Betley alignment per prompt, Gemma system prompt sweep\n\nMean alignment on coherent answers, both questions pooled. Differences are the "
              "prompt minus the comparison, with 95% interval and permutation p value.\n\n")
    return header + "\n\n".join(sections) + "\n"


def main():
    judged, coherent = load_coherent_and_judged_answers()
    rows = build_rows(judged, coherent)
    output_folder = RUN_FOLDER / f"betley_alignment_per_prompt_{datetime.now():%Y-%m-%d_%H-%M}"
    output_folder.mkdir(parents=True)
    timestamp = output_folder.name.removeprefix("betley_alignment_per_prompt_")
    flat = pd.DataFrame([{**{key: value for key, value in row.items() if not key.startswith("difference_from_")},
                          **{f"{kind}_{key}": (row[f"difference_from_{kind}"] or {}).get(key) for kind in ("base", "opposite", "control")
                             for key in ("difference", "interval_low", "interval_high", "permutation_p_value")}} for row in rows])
    flat["share_of_coherent_answers_under_50"] = [format_share_under_50(row) for row in rows]
    flat["four_lowest_alignment_scores"] = flat["four_lowest_alignment_scores"].map(lambda scores: " ".join(f"{score:g}" for score in scores))
    flat.to_csv(output_folder / f"betley_alignment_per_prompt_numbers_{timestamp}.csv", index=False)
    lowest = {row["prompt_id"]: lowest_answer(coherent[row["prompt_id"]]) for row in rows}
    (output_folder / f"betley_lowest_scoring_coherent_answer_per_prompt_{timestamp}.json").write_text(json.dumps(lowest, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output_folder / f"betley_alignment_per_prompt_tables_{timestamp}.md"
    markdown_path.write_text(markdown_tables(rows), encoding="utf-8")
    print(output_folder)


if __name__ == "__main__":
    main()
