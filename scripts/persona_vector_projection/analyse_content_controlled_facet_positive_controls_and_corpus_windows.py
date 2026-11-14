"""Free local analysis of a content-controlled facet's positive-control scoring run (the facet is read from the run configuration;
moralizing delivery, described below, is the default when the configuration names none). No model, API or pod.

Direction convention, for every facet: the vector is POSITIVE pole minus NEGATIVE pole and shift = rewrite minus comparison document,
so a rewrite towards the negative pole should give a NEGATIVE shift. Arms of the new facets (expected sign, compared with):
trait_only_away (-, original), content_added_without_trait (0, original), trait_removed_from_away_rewrite (+, the trait_only_away
REWRITE it was made from), facet_neutral_rewording (0, original). Pass thresholds come from
scripts/persona_vector_projection/pass_criteria/positive_control_pass_criteria__<facet key>.json (fixed in plans/10-30-persona-projection/
content-controlled-facets-pass-criteria-fixed-before-scoring_2026-11-01_15-04.md); a null threshold gives the verdict
'not evaluated: thresholds not fixed' (all statistics are still computed and written). Every counted criterion is listed with its
verdict and the facet verdict is 'passes' only if all pass; reported items (split-half cosine, removal-arm changed span, corpus
segments) are printed beside them with their former thresholds, labelled reported, and never change it. The counted whole-document
numbers are after token 64. The whole-document shift of every rewrite is
reported beside the corpus between-document standard deviation (standard deviation across the corpus documents of each document's
whole-document mean projection after token 64) for that vector and format, in the per-rewrite CSV, the per-arm CSV and the summary.

Per-AI-turn scoring of the corpus documents is NOT done, deliberately: the scored documents are free prose in about 20 languages
(articles, case studies, minutes, essays, dialogues) in which AI speech is quoted, paraphrased or narrated, and the scoring outputs
hold only one token stream per document with no turn boundaries. There is no deterministic way to find AI turns (line-start speaker
markers occur in almost none of the documents; the analysis counts them and writes the count in the summary). Locating them would
need a model call or hand marking of spans; the lowest 32-token windows below the document mean stay the corpus-segment check.

Original description (moralizing delivery):
 every criterion of the pass-criteria file
(plans/10-30-persona-projection/moralizing-delivery-pass-criteria-fixed-before-scoring_2026-11-01_01-55.md) for the main vector,
the same rows for the other two vectors, in assistant_turn with user_message as the secondary format. No model, API or pod.

    python3 scripts/persona_vector_projection/analyse_moralizing_delivery_positive_controls_and_corpus_windows.py \
        --scoring-run-folder <scoring run folder> [--corpus-window-hand-reading-json <file>] [--verify-saved-activations]

Per arm (moralizing_framing_only, plain_welfare_content_added, facet_neutral_rewording, moralizing_removed_from_preachier_rewrite;
the old preachier and plainer rewrites are added as context rows that no criterion uses), vector and format:
  changed-span shift (changed minus unchanged tokens after token 64), direction counts, Wilcoxon p, standardised effect,
  per-rewrite within-document circular-shift p (2,000 draws), and the whole-document shift with a per-rewrite paired block
  bootstrap (blocks of 64 units, 5,000 draws). Statistics: moralizing_delivery_positive_control_statistics.py.
Corpus windows: the 10 lowest windows of 32 tokens in all the corpus documents scored in the run (the earlier run's 20 plus the rewrite sources), ranked by how far each falls below its document's
mean (main vector, assistant_turn), with the decoded text and 60 tokens of context before it, for hand reading. The criterion for
them is decided from `--corpus-window-hand-reading-json` ({window id: true or false}); without it that line is undecided and a
template of window ids is written.

Writes to <scoring run folder>/analysis/moralizing_delivery_positive_control_analysis_<timestamp>/ (names end with <timestamp>):
  per_rewrite_results_<ts>.csv, per_arm_summary_<ts>.csv, criteria_verdicts_<ts>.json, lowest_corpus_windows_<ts>.csv and .json,
  corpus_window_hand_reading_template_<ts>.json, analysis_summary_<ts>.md, analysis_settings_<ts>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for _directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_arms as arms_module  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_statistics as statistics_module  # noqa: E402
from persona_vector_projection import saved_activation_projection_locally as saved_activations_module  # noqa: E402
from persona_vector_projection import scoring_vector_manifest_for_named_vectors as manifest_module  # noqa: E402
from persona_vector_projection.load_persona_projection_scoring_outputs import (  # noqa: E402
    load_projection_data, read_run_configuration)
from persona_vector_projection.scoring_pod_definitions import (DEFAULT_CORPUS_PATH, FORMAT_ASSISTANT_TURN,  # noqa: E402
                                                               FORMAT_USER_MESSAGE, INPUTS_FOLDER_NAME)
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

PRIMARY_FORMAT = FORMAT_ASSISTANT_TURN
CONTEXT_ARMS = [arms_module.PREACHIER_REWRITE_DIRECTION, arms_module.PLAINER_REWRITE_DIRECTION]
CONTEXT_EXPECTED_SIGN = {arms_module.PREACHIER_REWRITE_DIRECTION: arms_module.EXPECTED_NEGATIVE,
                         arms_module.PLAINER_REWRITE_DIRECTION: arms_module.EXPECTED_POSITIVE}
SPEAKER_MARKER_PATTERN = r"(?m)^\s*(?:\*\*)?(?:AI|Assistant|Claude|AI assistant|Model)\b[^\n]{0,20}[:\uff1a]"
PER_AI_TURN_SCORING_SKIPPED_REASON = (
    "skipped: the scored documents are free prose in many languages in which AI speech is quoted, paraphrased or narrated, and the "
    "scoring outputs hold one token stream per document with no turn boundaries; AI turns cannot be located deterministically "
    "(line-start speaker markers found in {with_markers} of {corpus_documents} corpus documents)")
SAVED_ACTIVATION_RELATIVE_TOLERANCE = 2e-3   # float16 keeps 11 significant bits; the projections are stored as float16 too


# ---------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------

def read_scored_documents(scoring_run_folder):
    """{document id: record of scoring_pod_inputs/documents.jsonl}."""
    return {record["document_id"]: record for record in read_jsonl(Path(scoring_run_folder) / INPUTS_FOLDER_NAME / "documents.jsonl")}


def per_ai_turn_scoring_status(documents_by_id):
    """{"available": False, "reason": text with the measured count of corpus documents that have line-start speaker markers}."""
    import re
    corpus = [record for record in documents_by_id.values() if record.get("kind") == "corpus"]
    with_markers = sum(1 for record in corpus if re.search(SPEAKER_MARKER_PATTERN, record.get("content", "")))
    return {"available": False, "corpus_documents": len(corpus), "corpus_documents_with_speaker_markers": with_markers,
            "reason": PER_AI_TURN_SCORING_SKIPPED_REASON.format(with_markers=with_markers, corpus_documents=len(corpus))}


def rewrite_ids_by_arm(documents_by_id, definition=None):
    """{arm: [rewrite ids]} for the facet's four arms and (moralizing delivery only) the two context rewrite kinds. The comparison
    document of every rewrite is its record's `source_document_id`."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    context = CONTEXT_ARMS if definition.legacy_moralizing_names else []
    groups = {arm: [] for arm in arms_module.arm_names_for(definition) + context}
    for document_id, record in documents_by_id.items():
        if record.get("kind") == "rewrite" and record.get("rewrite_arm") in groups:
            groups[record["rewrite_arm"]].append(document_id)
    return {arm: sorted(ids) for arm, ids in groups.items() if ids}


def read_vector_set_metadata_copy(scoring_run_folder):
    paths = sorted((Path(scoring_run_folder) / INPUTS_FOLDER_NAME / manifest_module.SCORING_VECTORS_SUBFOLDER).glob(
        "persona_vector_metadata_*.json"))
    return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else None


def extraction_lines_for_vector(metadata, vector_name, chosen_layer, definition=None, criteria=None):
    """The extraction criteria for one vector from the vector set's metadata (calibrated consideration's main vector also gets its two
    counted calibration lines). Numbers the set did not record are None, which makes their line undecided. Moralizing delivery uses its
    own coded thresholds and four lines; the other facets read theirs from `criteria` (null: not evaluated) and also get the reported
    split-half cosine line."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    legacy = definition.legacy_moralizing_names
    if legacy:
        evaluate = lambda *numbers: statistics_module.evaluate_extraction_criteria(*numbers)  # noqa: E731
    vector_set_key = definition.vector_set_key_by_vector_name().get(vector_name)   # None: a vector not in the facet's own set (a local rescoring variant)
    if metadata is None or vector_set_key is None:
        if legacy:
            return evaluate(None, None, None, None)
        return (statistics_module.evaluate_extraction_criteria_from_file(criteria or {}, None, None, None, None, None, None, None)
                + statistics_module.evaluate_calibrated_consideration_criteria_from_file(criteria or {}, None, None))
    entry = metadata["per_facet"][definition.facet_key]
    key = vector_set_key
    space = "passage_mean" if key.startswith("passage_mean") else "whole_answer_mean"
    variant = "raw" if key.endswith("_raw") else key[len(space) + 1:]
    variant_metrics = entry[space].get("variant_held_out_metrics", {}).get(variant)
    accuracy = by_device = negatives_by_tag = None
    if variant_metrics:
        accuracy = variant_metrics["held_out_accuracy_per_layer"][chosen_layer]
        by_device = {tag: cell["accuracy_per_layer"][chosen_layer] for tag, cell in variant_metrics["by_negative_tag"].items()} or None
        negatives_by_tag = {tag: cell["held_out_negative_answers"] for tag, cell in variant_metrics["by_negative_tag"].items()} or None
    split_half = entry[space].get("split_half_cosine_half_A_against_half_B_per_layer") if variant == "raw" else None
    split_half_at_layer = None if split_half is None else split_half[chosen_layer]
    if legacy:
        return evaluate(accuracy, by_device, split_half_at_layer, entry[space].get("questions_kept"))
    cross_half = (entry[space].get("cross_half_accuracy") or {}).get("vector_from_half_A_evaluated_on_half_B") if variant == "raw" else None
    cross_half_at_layer = None if not cross_half else cross_half["accuracy_per_layer"][chosen_layer]
    lines = statistics_module.evaluate_extraction_criteria_from_file(
        criteria or {}, accuracy, by_device, negatives_by_tag, cross_half_at_layer, split_half_at_layer, entry[space].get("questions_kept"),
        entry.get("training_questions_total"))
    if definition.has_confidence_control and vector_name == definition.main_vector_name:
        control = entry.get("confidence_control") or {}
        projected_out = (control.get("held_out_pole_classification_calibration_confidence_direction_projected_out") or {}).get("accuracy_per_layer")
        two_by_two = ((control.get("two_by_two_held_out_check") or {}).get("by_layer_and_vector") or {}).get(str(chosen_layer), {}).get("calibration")
        lines += statistics_module.evaluate_calibrated_consideration_criteria_from_file(
            criteria or {}, None if projected_out is None else projected_out[chosen_layer], two_by_two)
    return lines


def load_tokenizer_decoder(tokenizer_name, tokenizer_revision):
    """decode(list of token ids) -> text, from the local Hugging Face cache only (no download, no token)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_revision or None, token=False, local_files_only=True)
    return lambda token_ids: tokenizer.decode([int(i) for i in token_ids])


# ---------------------------------------------------------------------------------------------------
# Analysis of one format
# ---------------------------------------------------------------------------------------------------

def corpus_documents_used(documents_by_id, configuration):
    """The set the corpus between-document standard deviation and the corpus windows are computed over: every document of kind
    corpus scored in the run (the previous run's corpus documents and the rewrite sources, when the run configuration names them)."""
    ids = sorted(i for i, record in documents_by_id.items() if record.get("kind") == "corpus")
    recorded = (configuration or {}).get("corpus_documents_scored") or {}
    return {"rule": "all corpus documents scored in the run", "number_of_documents": len(ids), "document_ids": ids,
            "previous_scoring_run_document_ids": recorded.get("previous_scoring_run_document_ids"),
            "added_from_corpus_file_document_ids": recorded.get("added_from_corpus_file_document_ids")}


def corpus_between_document_standard_deviation(data, documents_by_id, column):
    """Standard deviation across the corpus documents of each document's whole-document mean projection (tokens after the opening)
    for one vector, the unit the whole-document shifts are read against."""
    corpus_ids = sorted(i for i, record in documents_by_id.items() if record.get("kind") == "corpus" and i in data.per_token)
    return statistics_module.between_document_standard_deviation(
        {i: statistics_module.whole_document_mean_projection(data.per_token[i][:, column]) for i in corpus_ids})


def analyse_format(data, documents_by_id, vector_names, seed, circular_draws, bootstrap_draws, say=print, definition=None, criteria=None):
    """(per-rewrite table, per-arm summary rows, {vector: positive-control criteria lines}) for one scored format. Criteria lines are
    moralizing delivery's coded ones, or (other facets) from the criteria file, where null thresholds give 'not evaluated'."""
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    groups = rewrite_ids_by_arm(documents_by_id, definition)
    arm_names = arms_module.arm_names_for(definition)
    comparison_of = {rewrite_id: documents_by_id[rewrite_id]["source_document_id"] for ids in groups.values() for rewrite_id in ids}
    token_ids = {document_id: [int(token_id) for token_id in ids] for document_id, ids in data.token_ids.items()}  # plain ints for the diff
    tables, summaries = [], []
    lines_by_vector = {}
    for vector_name in vector_names:
        column = data.vector_names.index(vector_name)
        between_document_sd = corpus_between_document_standard_deviation(data, documents_by_id, column)
        summaries_by_arm = {}
        for arm, ids in groups.items():
            say(f"  {data.document_format} / {vector_name} / {arm}: {len(ids)} rewrites")
            table = statistics_module.rewrite_results_for_arm(arm, ids, comparison_of, data.per_token, token_ids, column, seed,
                                                              circular_draws, bootstrap_draws)
            table.insert(0, "vector", vector_name)
            table.insert(1, "document_format", data.document_format)
            table["corpus_between_document_standard_deviation"] = between_document_sd
            table["whole_document_shift_in_between_document_standard_deviations"] = [
                statistics_module.shift_in_units_of_between_document_spread(shift, between_document_sd)
                for shift in table["whole_document_shift"]]
            table["below_minimum_changed_units"] = [documents_by_id[i].get("below_minimum_changed_units") for i in table["document_id"]]
            tables.append(table)
            is_confidence_vector = definition.has_confidence_control and vector_name == definition.confidence_control_vector_name
            summary = statistics_module.summarise_arm(
                table, arm, arms_module.expected_sign_on_vector_for(definition, arm, vector_name)
                if is_confidence_vector and arm in arm_names else None)
            if arm in CONTEXT_EXPECTED_SIGN:  # context rows: the expected sign of the older rewrites
                summary["expected_sign"] = CONTEXT_EXPECTED_SIGN[arm]
            summary["corpus_between_document_standard_deviation"] = between_document_sd
            summary["whole_document_shift_mean_in_between_document_standard_deviations"] = (
                statistics_module.shift_in_units_of_between_document_spread(summary["whole_document_shift_mean"], between_document_sd))
            summary = {"vector": vector_name, "document_format": data.document_format,
                       "used_by_a_criterion": arm in arm_names, **summary}
            summaries.append(summary)
            summaries_by_arm[arm] = summary
        # the plain-confidence control vector is reported beside the arms but has no pass criteria of its own
        if all(arm in summaries_by_arm for arm in arm_names) and not (
                definition.has_confidence_control and vector_name == definition.confidence_control_vector_name):
            lines_by_vector[vector_name] = (statistics_module.evaluate_positive_control_criteria(summaries_by_arm)
                                            if definition.legacy_moralizing_names else
                                            statistics_module.evaluate_criteria_from_file(summaries_by_arm, criteria or {}, definition.facet_key))
    return pd.concat(tables, ignore_index=True), summaries, lines_by_vector


ARMS_THAT_LOWER_CALIBRATION = ("trait_only_away", "understatement_of_supported_claims")


def side_by_side_calibration_and_confidence_shifts(summary_rows, definition):
    """Calibrated consideration: the two arms that both lower calibration (overclaiming; understatement of supported claims), side by
    side on the calibration vector (main) and the plain-confidence control vector, per document format. A calibration vector should move
    BOTH arms the same way (negative); a confidence vector should move them OPPOSITELY (overclaiming up, understatement down). Each row
    states the expected signs and whether the measured mean shifts have them; nothing here is a pass threshold."""
    if not definition.has_confidence_control:
        return []
    by_key = {(row["vector"], row["document_format"], row["arm"]): row for row in summary_rows}
    sign = lambda value: int(np.sign(value)) if np.isfinite(value) else None  # noqa: E731
    vector_by_role = {"calibration_vector": definition.main_vector_name, "confidence_vector": definition.confidence_control_vector_name}
    rows = []
    for document_format in sorted({row["document_format"] for row in summary_rows}):
        for measure in ("changed_span_mean", "whole_document_shift_mean"):
            cells = {}
            for role, vector_name in vector_by_role.items():
                for arm in ARMS_THAT_LOWER_CALIBRATION:
                    row = by_key.get((vector_name, document_format, arm))
                    cells[(role, arm)] = None if row is None else row[measure]
            if any(value is None for value in cells.values()):
                continue
            signs = {role: [sign(cells[(role, arm)]) for arm in ARMS_THAT_LOWER_CALIBRATION] for role in vector_by_role}
            expected = {(role, arm): arms_module.expected_sign_on_vector_for(definition, arm, vector_by_role[role]) for (role, arm) in cells}
            rows.append({
                "document_format": document_format, "measure": measure,
                **{f"{role}__{arm}__mean_shift": value for (role, arm), value in cells.items()},
                **{f"{role}__{arm}__expected_sign": value for (role, arm), value in expected.items()},
                "calibration_vector_moves_both_arms_the_same_way": None not in signs["calibration_vector"] and len(set(signs["calibration_vector"])) == 1,
                "confidence_vector_moves_the_arms_oppositely": None not in signs["confidence_vector"] and signs["confidence_vector"][0] == -signs["confidence_vector"][1] != 0,
                "calibration_vector_signs_as_expected": all(sign(cells[("calibration_vector", arm)]) == expected[("calibration_vector", arm)]
                                                            for arm in ARMS_THAT_LOWER_CALIBRATION),
                "confidence_vector_signs_as_expected": all(sign(cells[("confidence_vector", arm)]) == expected[("confidence_vector", arm)]
                                                           for arm in ARMS_THAT_LOWER_CALIBRATION)})
    return rows


def markdown_lines_for_side_by_side(rows):
    if not rows:
        return []
    out = ["## Calibration vector and confidence vector, the two arms that lower calibration", "",
           "Overclaiming (`trait_only_away`) raises stated confidence; understatement of supported claims lowers it. Both lower calibration. A "
           "calibration vector should move both arms the same way (negative); a plain-confidence vector should move them oppositely "
           "(overclaiming positive, understatement negative). Mean shifts; no pass thresholds.", "",
           "| Format | Measure | Calibration, overclaiming | Calibration, understatement | Confidence, overclaiming | Confidence, understatement | "
           "Calibration moves both the same way | Confidence moves them oppositely |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        out.append(f"| {row['document_format']} | {row['measure']} | {format_number(row['calibration_vector__trait_only_away__mean_shift'])} | "
                   f"{format_number(row['calibration_vector__understatement_of_supported_claims__mean_shift'])} | "
                   f"{format_number(row['confidence_vector__trait_only_away__mean_shift'])} | "
                   f"{format_number(row['confidence_vector__understatement_of_supported_claims__mean_shift'])} | "
                   f"{row['calibration_vector_moves_both_arms_the_same_way']} | {row['confidence_vector_moves_the_arms_oppositely']} |")
    return out + [""]


def corpus_windows(data, documents_by_id, main_vector_name, decode_function):
    """The lowest windows of the corpus documents for the main vector, with decoded text and context before."""
    corpus_ids = sorted(i for i, record in documents_by_id.items() if record.get("kind") == "corpus")
    column = data.vector_names.index(main_vector_name)
    windows = statistics_module.lowest_windows_below_document_mean({i: data.per_token[i][:, column] for i in corpus_ids})
    for rank, window in enumerate(windows, start=1):
        ids = data.token_ids[window["document_id"]]
        start, end = window["start_token"], window["start_token"] + statistics_module.WINDOW_TOKENS
        window.update({"window_id": f"lowest_window_{rank:02d}", "rank": rank, "window_text": decode_function(ids[start:end]),
                       "context_before_text": decode_function(ids[max(0, start - statistics_module.CONTEXT_TOKENS_BEFORE_WINDOW):start])})
    return windows


def verify_saved_activations(scoring_run_folder, data, vectors_by_name, chosen_layer):
    """Projects the saved assistant_turn activations onto each vector locally and compares with the pod's projections. Returns the
    largest difference relative to the largest projection, per vector."""
    activations, saved_layer = saved_activations_module.load_saved_activations(scoring_run_folder, data.document_format)
    if saved_layer != chosen_layer:
        raise ValueError(f"The saved activations are from layer {saved_layer}, not the chosen layer {chosen_layer}")
    report = {}
    for name, vector in vectors_by_name.items():
        column = data.vector_names.index(name)
        worst = 0.0
        for document_id, saved in activations.items():
            recomputed = saved_activations_module.project_saved_activations(saved, vector[chosen_layer])
            on_pod = data.per_token[document_id][:, column]
            scale = max(float(np.abs(on_pod).max()), 1e-6)
            worst = max(worst, float(np.abs(recomputed - on_pod).max()) / scale)
        report[name] = {"largest_difference_relative_to_largest_projection": worst, "documents_checked": len(activations),
                        "within_tolerance": worst <= SAVED_ACTIVATION_RELATIVE_TOLERANCE}
    return report


# ---------------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------------

def format_number(value):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "n/a" if not np.isfinite(value) else f"{value:.3g}"
    return str(value)


def verdict_text(passed):
    return "PASS" if passed else ("undecided" if passed is None else "FAIL")


def verdict_text_of_line(line):
    return line.get("verdict_text") or verdict_text(line["passed"]) if line["passed"] is None else verdict_text(line["passed"])


def markdown_lines_for_criteria(lines):
    rows = ["| Criterion | Measured | Required | Verdict |", "|---|---|---|---|"]
    for line in lines:
        measured = line["measured"]
        measured_text = (json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in measured.items()})
                         if isinstance(measured, dict) else format_number(measured))
        rows.append(f"| {line['description']} | {measured_text} | {line['required']} | {verdict_text_of_line(line)} |")
    return rows


def build_summary_markdown(configuration, verdicts, summary_rows, windows, window_verdict, saved_activation_report, timestamp, definition=None):
    definition = definition or facet_definitions.MORALIZING_DELIVERY
    main_vector, chosen_layer = verdicts["main_vector"], verdicts["chosen_layer"]
    out = [f"# {definition.display_name} positive controls and corpus windows, analysed {timestamp}", "",
           f"Scoring run: `{configuration.get('run_name')}`. Main vector `{main_vector}` at layer {chosen_layer}. The vector is "
           f"{definition.positive_pole_label} (positive pole) minus {definition.negative_pole_label} (negative pole), so the negative "
           "pole is the negative direction: shift = rewrite minus comparison document, and a rewrite towards the negative pole should "
           f"give a NEGATIVE shift. {definition.facet_score_scale_label()}. Primary format: assistant_turn; user_message is secondary.", ""]
    primary = verdicts["by_vector_and_format"][main_vector][PRIMARY_FORMAT]
    out += ["## Main vector, assistant_turn: pass or fail (lines marked `reported` are not counted)", "", "### Extraction criteria", ""]
    out += markdown_lines_for_criteria(primary["extraction_criteria"]) + ["", "### Positive controls", ""]
    out += markdown_lines_for_criteria(primary["positive_control_criteria"]) + ["", "### Corpus segments", ""]
    out += markdown_lines_for_criteria([window_verdict]) + [""]
    out += [f"Counted criteria passed: {verdicts['summary']['passed']}, failed: {verdicts['summary']['failed']}, undecided: "
            f"{verdicts['summary']['undecided']}. Reported, not counted: {verdicts['summary']['reported']}.", ""]
    if "facet_verdict" in verdicts:
        facet_verdict = verdicts["facet_verdict"]
        out += [f"## Verdict for {definition.display_name} (main vector, assistant_turn): {facet_verdict['verdict']}", "",
                "The verdict is `passes` only if every counted criterion passes. Counted criteria:", ""]
        out += [f"- {item['verdict']}: {item['criterion']} (required {item['required']})" for item in facet_verdict["counted_criteria"]]
        out += ["", "Reported beside them, never changing the verdict (former thresholds shown):", ""]
        out += [f"- reported: {item['criterion']} (measured {format_number(item['measured']) if not isinstance(item['measured'], dict) else json.dumps(item['measured'], default=str)}; "
                f"{item['required']})" for item in facet_verdict["reported_not_counted"]]
        out += [""]
    out += ["## Every arm, vector and format", "",
            "| Vector | Format | Arm | Expected sign | Rewrites | Changed-span mean | Negative / positive | Wilcoxon p (two-sided) | Wilcoxon p (one-sided, expected direction) | Standardised effect (negative-pole direction) | "
            "Circular significant (negative pole / positive / either, two-sided) | Whole-document shift mean | Corpus between-document standard deviation | "
            "Shift in between-document standard deviations | Whole document negative / positive | "
            "Whole-document bootstrap significant (negative / positive) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in summary_rows:
        out.append(f"| {row['vector']} | {row['document_format']} | {row['arm']} | {row['expected_sign']} | {row['n_rewrites']} | {format_number(row['changed_span_mean'])} | "
                   f"{row['changed_span_n_negative']} / {row['changed_span_n_positive']} | {format_number(row['changed_span_wilcoxon_p'])} | "
                   f"{format_number(row['changed_span_wilcoxon_p_expected_direction'])} | "
                   f"{format_number(row['standardised_effect_in_negative_pole_direction'])} | "
                   f"{row['circular_n_significant_negative_pole_direction']} / {row['circular_n_significant_positive_direction']} / "
                   f"{row['circular_n_significant_either_direction_two_sided']} | "
                   f"{format_number(row['whole_document_shift_mean'])} | {format_number(row['corpus_between_document_standard_deviation'])} | "
                   f"{format_number(row['whole_document_shift_mean_in_between_document_standard_deviations'])} | "
                   f"{row['whole_document_n_negative']} / {row['whole_document_n_positive']} | "
                   f"{row['whole_bootstrap_n_significant_negative_direction']} / {row['whole_bootstrap_n_significant_positive_direction']} |")
    out += [""] + markdown_lines_for_side_by_side(verdicts.get("calibration_and_confidence_side_by_side") or [])
    out += ["## The other vectors and the secondary format", ""]
    for vector_name, by_format in verdicts["by_vector_and_format"].items():
        for document_format, block in by_format.items():
            if vector_name == main_vector and document_format == PRIMARY_FORMAT:
                continue
            counts = block["summary"]
            out += [f"### {vector_name}, {document_format}: {counts['passed']} passed, {counts['failed']} failed, {counts['undecided']} undecided "
                    f"(counted); {counts['reported']} reported", ""]
            out += markdown_lines_for_criteria(block["extraction_criteria"] + block["positive_control_criteria"]) + [""]
    used = verdicts["corpus_documents_for_between_document_spread_and_windows"]
    out += ["## Corpus documents behind the between-document spread and the windows", "",
            f"The corpus between-document standard deviation and the corpus windows are computed over {used['rule']}: "
            f"{used['number_of_documents']} documents ({len(used['previous_scoring_run_document_ids'] or [])} scored in the earlier run, "
            f"{len(used['added_from_corpus_file_document_ids'] or [])} added as rewrite sources). Ids: {', '.join(used['document_ids'])}.", ""]
    out += ["## Per-AI-turn scoring of the corpus documents", "", verdicts["per_ai_turn_scoring"]["reason"], ""]
    out += ["## Lowest 32-token windows of the corpus documents (main vector, assistant_turn)", "",
            "Ranked by how far the window mean falls below its own document's mean (tokens after 64). Read each and mark whether it "
            "contains an AI, or a quoted AI, showing the facet's negative pole" + (" (delivering a welfare point with moral framing)" if definition.legacy_moralizing_names else "")
            + ".", ""]
    for window in windows:
        out += [f"**{window['rank']}. {window['document_id']}, tokens {window['start_token']} to "
                f"{window['start_token'] + statistics_module.WINDOW_TOKENS}**: window mean {window['window_mean']:.2f}, document mean "
                f"{window['document_mean']:.2f}, {window['below_document_mean']:.2f} below.", "",
                "Context before:", "", "> " + window["context_before_text"].replace("\n", "\n> "), "",
                "Window:", "", "> " + window["window_text"].replace("\n", "\n> "), ""]
    if saved_activation_report:
        out += ["## Saved activations recomputed locally", ""]
        for name, report in saved_activation_report.items():
            out.append(f"- {name}: largest difference {report['largest_difference_relative_to_largest_projection']:.2e} of the largest "
                       f"projection over {report['documents_checked']} documents, within tolerance: {report['within_tolerance']}")
    return "\n".join(out) + "\n"


def count_verdicts(lines):
    """Counts of the COUNTED lines (those that decide the verdict); reported lines are counted apart and never change a count."""
    counted = [line for line in lines if line.get("counted", True)]
    return {"passed": sum(1 for line in counted if line["passed"] is True), "failed": sum(1 for line in counted if line["passed"] is False),
            "undecided": sum(1 for line in counted if line["passed"] is None), "reported": len(lines) - len(counted)}


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def run_analysis(scoring_run_folder, corpus_path, formats, seed, circular_draws, bootstrap_draws, decode_function,
                 hand_reading=None, verify_saved=False, output_folder=None, say=print, criteria_file=None):
    run_folder = Path(scoring_run_folder)
    configuration = read_run_configuration(run_folder)
    definition = facet_definitions.get_facet_definition(configuration.get("facet_key", facet_definitions.FACET_MORALIZING_DELIVERY))
    criteria = ({} if definition.legacy_moralizing_names else
                statistics_module.load_pass_criteria_file(criteria_file or definition.pass_criteria_file()))
    timestamp = timestamp_now()
    chosen_layer = int(configuration["chosen_layer"])
    vector_names = list(configuration["vector_names"])
    main_vector = configuration["main_vector_name"]
    documents_by_id = read_scored_documents(run_folder)
    metadata = read_vector_set_metadata_copy(run_folder)
    output_folder = Path(output_folder or run_folder / "analysis" / f"{definition.facet_key}_positive_control_analysis_{timestamp}")
    output_folder.mkdir(parents=True, exist_ok=False)

    tables, summary_rows = [], []
    verdicts = {"main_vector": main_vector, "chosen_layer": chosen_layer, "primary_format": PRIMARY_FORMAT, "by_vector_and_format": {}}
    windows, saved_activation_report = [], None
    for document_format in formats:
        say(f"Format {document_format}")
        data = load_projection_data(run_folder, document_format, corpus_path, chosen_layer)
        if not data.token_ids:
            raise ValueError(f"The scoring output for {document_format} has no saved token ids (per_token_token_ids_int32)")
        table, rows, lines_by_vector = analyse_format(data, documents_by_id, vector_names, seed, circular_draws, bootstrap_draws, say, definition, criteria)
        tables.append(table)
        summary_rows.extend(rows)
        for vector_name, positive_lines in lines_by_vector.items():
            extraction_lines = extraction_lines_for_vector(metadata, vector_name, chosen_layer, definition, criteria)
            block = {"extraction_criteria": extraction_lines, "positive_control_criteria": positive_lines,
                     "summary": count_verdicts(extraction_lines + positive_lines)}
            if not definition.legacy_moralizing_names:
                block["verdict"] = statistics_module.overall_verdict(extraction_lines + positive_lines, bool(criteria.get("thresholds_fixed")))
            verdicts["by_vector_and_format"].setdefault(vector_name, {})[document_format] = block
        if document_format == PRIMARY_FORMAT:
            windows = corpus_windows(data, documents_by_id, main_vector, decode_function)
            if verify_saved:
                vectors = manifest_module.load_named_scoring_vectors(run_folder / INPUTS_FOLDER_NAME).vectors_by_name
                saved_activation_report = verify_saved_activations(run_folder, data, vectors, chosen_layer)
    window_ids = [window["window_id"] for window in windows]
    reading = None if hand_reading is None else {window_id: hand_reading.get(window_id) for window_id in window_ids}
    window_verdict = (statistics_module.evaluate_corpus_segment_criterion(reading) if definition.legacy_moralizing_names
                      else statistics_module.evaluate_corpus_segment_criterion_from_file(criteria, reading))
    primary_block = verdicts["by_vector_and_format"][main_vector][PRIMARY_FORMAT]
    primary_lines = primary_block["extraction_criteria"] + primary_block["positive_control_criteria"] + [window_verdict]
    verdicts["corpus_segment_criterion"] = window_verdict
    verdicts["summary"] = count_verdicts(primary_lines)
    verdicts["facet_key"] = definition.facet_key
    verdicts["facet_score_scale"] = definition.facet_score_scale_label()
    verdicts["direction_convention"] = ("vector = positive pole minus negative pole; shift = rewrite minus comparison document; a rewrite "
                                        "towards the negative pole should give a NEGATIVE shift")
    verdicts["thresholds_fixed"] = True if definition.legacy_moralizing_names else bool(criteria.get("thresholds_fixed"))
    verdicts["per_ai_turn_scoring"] = per_ai_turn_scoring_status(documents_by_id)
    verdicts["corpus_documents_for_between_document_spread_and_windows"] = corpus_documents_used(documents_by_id, configuration)
    if definition.legacy_moralizing_names:
        verdicts["decision_for_the_main_vector_in_assistant_turn"] = (
            "undecided: at least one criterion has no result yet" if verdicts["summary"]["undecided"] else
            ("all extraction, positive-control and corpus-segment criteria pass" if not verdicts["summary"]["failed"] else
             "at least one criterion fails: report it and do not build the other facets until the user has seen the failure"))
    else:
        # every counted criterion decides; the corpus-segment line and the other reported lines are printed beside them and never count
        facet_verdict = statistics_module.overall_verdict(primary_lines, verdicts["thresholds_fixed"])
        verdicts["facet_verdict"] = facet_verdict
        verdicts["decision_for_the_main_vector_in_assistant_turn"] = {
            statistics_module.VERDICT_PASSES: "passes: every counted criterion passes; the vector may be used to score corpus and difficult-advice documents",
            statistics_module.VERDICT_FAILS: "fails: at least one counted criterion fails; report the failure to the user before the vector scores any "
                                             "corpus or difficult-advice document",
            statistics_module.VERDICT_UNDECIDED: "undecided: no counted criterion fails but at least one has no result yet",
            statistics_module.NOT_EVALUATED_THRESHOLDS_NOT_FIXED: statistics_module.NOT_EVALUATED_THRESHOLDS_NOT_FIXED}[facet_verdict["verdict"]]
    verdicts["saved_activations_check"] = saved_activation_report
    verdicts["calibration_and_confidence_side_by_side"] = side_by_side_calibration_and_confidence_shifts(summary_rows, definition)

    pd.concat(tables, ignore_index=True).to_csv(output_folder / f"per_rewrite_results_{timestamp}.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_folder / f"per_arm_summary_{timestamp}.csv", index=False)
    (output_folder / f"criteria_verdicts_{timestamp}.json").write_text(json.dumps(verdicts, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(windows).to_csv(output_folder / f"lowest_corpus_windows_{timestamp}.csv", index=False)
    (output_folder / f"lowest_corpus_windows_{timestamp}.json").write_text(json.dumps(windows, indent=1, ensure_ascii=False), encoding="utf-8")
    (output_folder / f"corpus_window_hand_reading_template_{timestamp}.json").write_text(
        json.dumps({window_id: None for window_id in window_ids}, indent=1), encoding="utf-8")
    (output_folder / f"analysis_settings_{timestamp}.json").write_text(json.dumps({
        "scoring_run_folder": str(run_folder), "formats": formats, "seed": seed, "circular_shift_draws": circular_draws,
        "bootstrap_draws": bootstrap_draws, "bootstrap_block_units": statistics_module.BOOTSTRAP_BLOCK_UNITS,
        "opening_tokens_skipped": statistics_module.OPENING_TOKENS_SKIPPED, "window_tokens": statistics_module.WINDOW_TOKENS,
        "chosen_layer": chosen_layer, "vector_names": vector_names, "hand_reading_given": hand_reading is not None,
        "corpus_documents_for_between_document_spread_and_windows": verdicts["corpus_documents_for_between_document_spread_and_windows"]},
        indent=2), encoding="utf-8")
    (output_folder / f"analysis_summary_{timestamp}.md").write_text(
        build_summary_markdown(configuration, verdicts, summary_rows, windows, window_verdict, saved_activation_report, timestamp, definition),
        encoding="utf-8")
    say(f"Wrote the analysis to {output_folder}")
    return output_folder, verdicts


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-run-folder", required=True)
    parser.add_argument("--formats", nargs="+", default=[PRIMARY_FORMAT, FORMAT_USER_MESSAGE])
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--circular-shift-draws", type=int, default=statistics_module.CIRCULAR_SHIFT_DRAWS)
    parser.add_argument("--bootstrap-draws", type=int, default=statistics_module.BOOTSTRAP_DRAWS)
    parser.add_argument("--corpus-window-hand-reading-json", default=None,
                        help="{window id: true or false} from reading the windows; without it that criterion is undecided")
    parser.add_argument("--verify-saved-activations", action="store_true",
                        help="Project the saved assistant_turn activations locally and compare with the pod's projections")
    parser.add_argument("--criteria-file", default=None,
                        help="Per-facet pass criteria JSON (default: pass_criteria/positive_control_pass_criteria__<facet key>.json)")
    parser.add_argument("--tokenizer-name", default=None, help="Default: the run's model name (local cache only)")
    parser.add_argument("--tokenizer-revision", default=None, help="Default: the run's model revision")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    configuration = read_run_configuration(arguments.scoring_run_folder)
    decode_function = load_tokenizer_decoder(arguments.tokenizer_name or configuration["model_name"],
                                             arguments.tokenizer_revision or configuration.get("model_revision"))
    hand_reading = (json.loads(Path(arguments.corpus_window_hand_reading_json).read_text(encoding="utf-8"))
                    if arguments.corpus_window_hand_reading_json else None)
    run_analysis(arguments.scoring_run_folder, arguments.corpus_path, arguments.formats, arguments.seed, arguments.circular_shift_draws,
                 arguments.bootstrap_draws, decode_function, hand_reading, arguments.verify_saved_activations,
                 criteria_file=arguments.criteria_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
