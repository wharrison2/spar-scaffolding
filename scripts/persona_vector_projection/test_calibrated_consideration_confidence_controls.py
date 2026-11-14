"""Tests for the redesigned calibrated consideration facet: band rules for the negative rewordings, balanced generation, balancing stated
confidence between the poles, the plain-confidence control vector, projecting its direction out, the two-by-two held-out check, and the
two positive-control arms that lower calibration while moving confidence in opposite directions. Free and local; synthetic data only.

Experiment-invalidating cases covered:
  * a negative rewording used on a band where it is accurate (certainty on established evidence, understatement on weak evidence), in
    generation and in the vector computation
  * balancing that fails silently (poles that cannot be balanced must raise; the result must satisfy the bound it claims)
  * the two-by-two cells mislabelled (cell by judged confidence and band, never by pole)
  * a confidence vector built from a split confounded with pole
  * arms with swapped expected signs (calibration vector: both arms negative; confidence vector: opposite signs)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import analyse_content_controlled_facet_positive_controls_and_corpus_windows as analysis  # noqa: E402
from persona_vector_projection import build_extraction_run_folder as builder  # noqa: E402
from persona_vector_projection import calibrated_consideration_confidence_controls as controls  # noqa: E402
from persona_vector_projection import compute_content_controlled_facet_vectors_from_activations as compute  # noqa: E402
from persona_vector_projection import content_controlled_facet_definitions as D  # noqa: E402
from persona_vector_projection import content_controlled_facet_positive_control_arms as arms  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import synthetic_extraction_run_for_tests as synthetic  # noqa: E402
from persona_vector_projection.test_content_controlled_facet_extraction import THRESHOLDS  # noqa: E402

CALIBRATED = D.CALIBRATED_CONSIDERATION
FACET_KEY = CALIBRATED.facet_key
TAG_BY_REWORDING = synthetic.CALIBRATED_NEGATIVE_TAG_BY_REWORDING
BANDS = {"Established": "established", "Strong but debated": "strong_but_debated", "Weak (open)": "weak", "Weak (possible)": "weak",
         "Weak (possible and open)": "weak", "Mixed: established and weak (open)": "mixed",
         "Mixed: weak (possible), strong but debated, weak (open)": "mixed"}


def prompt_records():
    records = []
    for pole, count in ((defs.POLE_POSITIVE, 5), (defs.POLE_NEGATIVE, 10)):
        for number in range(1, count + 1):
            records.append({"system_prompt_id": defs.system_prompt_identifier(FACET_KEY, pole, number), "facet": FACET_KEY, "pole": pole,
                            "rewording_number": number, "tag": TAG_BY_REWORDING[number] if pole == defs.POLE_NEGATIVE else None,
                            "system_prompt": "Say {evidence_content} then {recommendation}.", "persona_description": "x"})
    return records


def question_records(bands):
    return [{"facet": FACET_KEY, "question_id": f"{FACET_KEY}__A__{i:02d}", "half": "A", "domain": "d", "question_text": "Q?",
             "labels": {"evidence_band": band, "evidence_content": "E.", "recommendation": "R."}} for i, band in enumerate(bands, start=1)]


# ---------------------------------------------------------------------------------------------------
# Band rules: a rewording is used only where it is miscalibrated
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("band,tag,allowed", [
    ("Established", "certainty", False), ("Established", "certainty_and_equal_capacity", False), ("Established", "equal_capacity", False),
    ("Established", "dismissal", True), ("Established", "understatement", True), ("Established", "understatement_and_dismissal", True),
    ("Weak (open)", "understatement", False), ("Weak (possible)", "understatement_and_dismissal", False),
    ("Weak (open)", "certainty", True), ("Weak (open)", "certainty_and_equal_capacity", True), ("Weak (open)", "dismissal", True),
    ("Strong but debated", "certainty", True), ("Strong but debated", "understatement", True),
    ("Mixed: established and weak (open)", "understatement", True),  # one part is established
    ("Mixed: weak (possible), strong but debated, weak (open)", "understatement", False),  # no established part
    ("Mixed: weak (possible), strong but debated, weak (open)", "certainty", True)])
def test_band_table_of_the_draft(band, tag, allowed):
    assert D.negative_tag_is_miscalibrated_on_band(tag, band) is allowed


def test_unknown_band_label_or_tag_raises_instead_of_being_guessed():
    with pytest.raises(ValueError, match="Unrecognised evidence band"):
        D.evidence_band_class("Probably established")
    with pytest.raises(ValueError, match="unknown kind"):
        D.negative_tag_is_miscalibrated_on_band("overconfidence", "Established")


def test_generation_never_gives_a_question_a_rewording_that_is_accurate_for_its_band_and_stays_balanced():
    bands = list(BANDS) * 3
    questions = question_records(bands)
    jobs = builder.build_generation_jobs(questions, prompt_records(), samples_per_prompt=1)
    tag_by_prompt = {p["system_prompt_id"]: p["tag"] for p in prompt_records()}
    for question in questions:
        mine = [j for j in jobs if j["question_id"] == question["question_id"]]
        positives = [j for j in mine if j["pole"] == "positive"]
        negatives = [j for j in mine if j["pole"] == "negative"]
        assert len(positives) == len(negatives) == 5  # the same number of answers per pole for every question
        for job in negatives:
            assert D.negative_tag_is_miscalibrated_on_band(tag_by_prompt[job["system_prompt_id"]], question["labels"]["evidence_band"]), job["job_id"]
        slots = sorted(D.pairing_slot_of_rewording(int(j["system_prompt_id"].rsplit("_", 1)[1]), 5) for j in negatives)
        assert slots == [1, 2, 3, 4, 5]  # one negative rewording per positive rewording's structure
    used = {j["system_prompt_id"] for j in jobs if j["pole"] == "negative"}
    assert len(used) == 10  # over a mixed set of bands every negative rewording is used


def test_generation_is_deterministic_and_spreads_use_between_two_allowed_rewordings():
    questions = question_records(["Strong but debated"] * 20)
    first = builder.build_generation_jobs(questions, prompt_records(), 1)
    second = builder.build_generation_jobs(questions, prompt_records(), 1)
    assert [j["job_id"] for j in first] == [j["job_id"] for j in second]
    uses = {}
    for job in first:
        if job["pole"] == "negative":
            uses[job["system_prompt_id"]] = uses.get(job["system_prompt_id"], 0) + 1
    assert set(uses.values()) == {10}  # 20 questions, two candidates per slot, used equally often


def test_generation_refuses_to_fill_a_slot_with_an_accurate_rewording():
    only_certainty_in_slot_one = [p for p in prompt_records() if p["pole"] == "positive" or p["rewording_number"] != 6]
    with pytest.raises(ValueError, match="No negative rewording in slot 1"):
        builder.build_generation_jobs(question_records(["Established"]), only_certainty_in_slot_one, 1)  # slot 1 holds only certainty here


def test_a_negative_answer_made_with_a_rewording_accurate_on_its_band_is_rejected_by_the_filter():
    question = question_records(["Established"])[0]
    answer = {"pole": "negative", "tag": "certainty"}
    failures = controls.stated_confidence_condition_failures(answer, {"stated_confidence": 80.0}, question, CALIBRATED)
    assert failures == ["negative_rewording_is_accurate_on_the_questions_evidence_band"]
    assert controls.stated_confidence_condition_failures({"pole": "negative", "tag": "dismissal"}, {"stated_confidence": 80.0}, question, CALIBRATED) == []
    assert controls.stated_confidence_condition_failures({"pole": "positive", "tag": None}, {"stated_confidence": None}, question, CALIBRATED) == [
        "stated_confidence_missing"]


# ---------------------------------------------------------------------------------------------------
# Balancing stated confidence
# ---------------------------------------------------------------------------------------------------

def make_items(differences_by_band, positive_base=60.0):
    """Items with the given positive-minus-negative stated confidence, per band group; returns (items, scores, band group by question)."""
    items, scores, band_by_question = [], {}, {}
    for band_group, differences in differences_by_band.items():
        for index, difference in enumerate(differences):
            question_id = f"{band_group}_{index}"
            positive = {"answer_id": f"{question_id}_p"}
            negative = {"answer_id": f"{question_id}_n"}
            scores[positive["answer_id"]] = positive_base
            scores[negative["answer_id"]] = positive_base - difference
            band_by_question[question_id] = band_group
            items.append(((question_id, 1, 0), positive, negative))
    return items, scores, band_by_question


def test_balancing_brings_every_band_and_the_overall_difference_within_five_and_reports_means_before_and_after():
    items, scores, bands = make_items({"established": [20, 20, -10, 0, 2, 12, 0, -3], "weak": [-15, 0, 4, 30, 10, -2, 1, 0],
                                       "strong_but_debated": [8, 9, -7, -10], "mixed": [0, 3, -3]})
    balanced, report = controls.balance_items_on_stated_confidence(items, scores, bands, 5.0)
    assert report["before_balancing"]["established"]["difference_positive_minus_negative"] > 5  # it needed balancing
    for name, entry in report["after_balancing"].items():
        assert entry["items"] == 0 or abs(entry["difference_positive_minus_negative"]) <= 5.0, name
    assert 0 < len(balanced) < len(items)
    again = controls.pole_mean_stated_confidence(balanced, scores, bands)  # recomputed independently of the report
    assert abs(again["overall"]["difference_positive_minus_negative"]) <= 5.0
    assert abs(again["established"]["difference_positive_minus_negative"]) <= 5.0 and abs(again["weak"]["difference_positive_minus_negative"]) <= 5.0
    assert report["before_balancing"]["overall"]["positive_pole_mean"] == pytest.approx(60.0)


def test_balancing_that_cannot_succeed_raises_instead_of_passing_unbalanced_items_on():
    items, scores, bands = make_items({"established": [0, 1, -1, 2], "weak": [40, 35, 50, 45]})  # the weak band's poles always differ by 35+
    with pytest.raises(controls.ConfidenceBalancingError, match="weak band"):
        controls.balance_items_on_stated_confidence(items, scores, bands, 5.0)


def test_already_balanced_items_are_all_kept():
    items, scores, bands = make_items({"established": [1, -1, 2, -2], "weak": [0, 0, 3, -3]})
    balanced, report = controls.balance_items_on_stated_confidence(items, scores, bands, 5.0)
    assert len(balanced) == len(items) and sum(report["items_removed_by_band_group"].values()) == 0


# ---------------------------------------------------------------------------------------------------
# The confidence vector and its guards
# ---------------------------------------------------------------------------------------------------

def answers_with(cells):
    """cells: {(confidence value, pole): count}"""
    answers, scores = [], {}
    for (confidence, pole), count in cells.items():
        for index in range(count):
            answer_id = f"{pole}_{confidence}_{index}"
            answers.append({"answer_id": answer_id, "pole": pole})
            scores[answer_id] = confidence
    return answers, scores


def test_confidence_groups_have_equal_shares_of_calibrated_and_miscalibrated_answers():
    answers, scores = answers_with({(90.0, "positive"): 30, (90.0, "negative"): 12, (10.0, "positive"): 9, (10.0, "negative"): 25, (50.0, "positive"): 40})
    groups = controls.select_balanced_confidence_groups(answers, scores, seed=3)
    pole_of = {a["answer_id"]: a["pole"] for a in answers}
    for name in ("confident", "hedged"):
        poles = [pole_of[i] for i in groups[name]]
        assert poles.count("positive") == poles.count("negative") == 9  # the smallest cell: 9
    assert not set(groups["confident"]) & set(groups["hedged"])
    assert all(scores[i] >= 65 for i in groups["confident"]) and all(scores[i] <= 35 for i in groups["hedged"])  # moderate answers excluded


def test_a_confidence_split_confounded_with_pole_is_refused():
    # every confident answer is positive-pole and every hedged answer negative-pole: the "confidence vector" would be the pole vector
    answers, scores = answers_with({(90.0, "positive"): 20, (10.0, "negative"): 20})
    with pytest.raises(controls.ConfidenceBalancingError, match="cannot be balanced"):
        controls.select_balanced_confidence_groups(answers, scores)
    pole_of = {f"c{i}": "positive" for i in range(6)} | {f"h{i}": "negative" for i in range(6)}
    with pytest.raises(controls.ConfidenceBalancingError, match="equally represented"):
        controls.assert_confidence_groups_are_not_confounded_with_pole(
            {"confident": [f"c{i}" for i in range(6)], "hedged": [f"h{i}" for i in range(6)]}, pole_of)
    skewed = {f"c{i}": "positive" if i < 4 else "negative" for i in range(6)} | {f"h{i}": "positive" if i < 2 else "negative" for i in range(6)}
    with pytest.raises(controls.ConfidenceBalancingError):
        controls.assert_confidence_groups_are_not_confounded_with_pole(
            {"confident": [f"c{i}" for i in range(6)], "hedged": [f"h{i}" for i in range(6)]}, skewed)


def test_projecting_a_direction_out_removes_exactly_that_component():
    generator = np.random.default_rng(0)
    vector, direction = generator.normal(size=(4, 16)), generator.normal(size=(4, 16))
    result = controls.project_direction_out(vector, direction)
    assert np.allclose(np.sum(result * direction, axis=-1), 0, atol=1e-5)
    unit = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    assert np.allclose(result + np.sum(vector * unit, axis=-1, keepdims=True) * unit, vector, atol=1e-5)


# ---------------------------------------------------------------------------------------------------
# The two-by-two check
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("confidence,band_group,cell", [
    (90.0, "established", controls.CELL_CONFIDENT_ESTABLISHED), (10.0, "established", controls.CELL_HEDGED_ESTABLISHED),
    (90.0, "weak", controls.CELL_CONFIDENT_WEAK), (10.0, "weak", controls.CELL_HEDGED_WEAK),
    (65.0, "weak", controls.CELL_CONFIDENT_WEAK), (35.0, "established", controls.CELL_HEDGED_ESTABLISHED),
    (50.0, "weak", None), (90.0, "strong_but_debated", None), (10.0, "mixed", None)])
def test_cells_come_from_judged_confidence_and_band_only(confidence, band_group, cell):
    assert controls.two_by_two_cell_of(confidence, band_group) == cell


def test_calibrated_cells_are_confident_on_established_and_hedged_on_weak():
    assert set(controls.CALIBRATED_CELLS) == {controls.CELL_CONFIDENT_ESTABLISHED, controls.CELL_HEDGED_WEAK}
    assert set(controls.MISCALIBRATED_CELLS) == {controls.CELL_HEDGED_ESTABLISHED, controls.CELL_CONFIDENT_WEAK}


def cell_data(cell_means, per_cell=12, spread=0.1, seed=0):
    generator = np.random.default_rng(seed)
    projections, cells, questions = {}, {}, {}
    for cell, mean in cell_means.items():
        for index in range(per_cell):
            answer_id = f"{cell}_{index}"
            projections[answer_id] = mean + generator.normal() * spread
            cells[answer_id], questions[answer_id] = cell, f"question_{cell}_{index % 4}"
    return projections, cells, questions


def test_a_calibration_vector_has_a_positive_interaction_and_no_main_effect_of_confidence():
    means = {controls.CELL_CONFIDENT_ESTABLISHED: 2.0, controls.CELL_HEDGED_WEAK: 2.0, controls.CELL_HEDGED_ESTABLISHED: -2.0,
             controls.CELL_CONFIDENT_WEAK: -2.0}
    result = controls.two_by_two_check(*cell_data(means), resamples=200, seed=1)
    assert result["interaction_contrast_calibrated_minus_miscalibrated"] == pytest.approx(4.0, abs=0.2)
    assert result["main_effect_of_confidence_confident_minus_hedged"] == pytest.approx(0.0, abs=0.2)
    low, high = result["interaction_contrast_95_percent_bootstrap_interval"]
    assert 0 < low < 4.0 < high


def test_a_confidence_vector_has_a_main_effect_and_no_interaction():
    means = {controls.CELL_CONFIDENT_ESTABLISHED: 2.0, controls.CELL_CONFIDENT_WEAK: 2.0, controls.CELL_HEDGED_ESTABLISHED: -2.0,
             controls.CELL_HEDGED_WEAK: -2.0}
    result = controls.two_by_two_check(*cell_data(means), resamples=200, seed=1)
    assert result["main_effect_of_confidence_confident_minus_hedged"] == pytest.approx(4.0, abs=0.2)
    assert result["interaction_contrast_calibrated_minus_miscalibrated"] == pytest.approx(0.0, abs=0.2)


def test_mislabelled_cells_flip_the_interaction_sign_so_the_labels_matter():
    means = {controls.CELL_CONFIDENT_ESTABLISHED: 2.0, controls.CELL_HEDGED_WEAK: 2.0, controls.CELL_HEDGED_ESTABLISHED: -2.0,
             controls.CELL_CONFIDENT_WEAK: -2.0}
    projections, cells, questions = cell_data(means)
    swapped = {a: {controls.CELL_CONFIDENT_ESTABLISHED: controls.CELL_HEDGED_ESTABLISHED, controls.CELL_HEDGED_ESTABLISHED: controls.CELL_CONFIDENT_ESTABLISHED,
                   controls.CELL_CONFIDENT_WEAK: controls.CELL_HEDGED_WEAK, controls.CELL_HEDGED_WEAK: controls.CELL_CONFIDENT_WEAK}[c]
               for a, c in cells.items()}
    assert controls.two_by_two_check(projections, swapped, questions, resamples=50)["interaction_contrast_calibrated_minus_miscalibrated"] < -3.5


def test_a_missing_cell_gives_no_contrast_instead_of_a_number():
    means = {controls.CELL_CONFIDENT_ESTABLISHED: 2.0, controls.CELL_HEDGED_WEAK: 2.0, controls.CELL_HEDGED_ESTABLISHED: -2.0}
    result = controls.two_by_two_check(*cell_data(means), resamples=20)
    assert result["interaction_contrast_calibrated_minus_miscalibrated"] is None and result["main_effect_of_confidence_confident_minus_hedged"] is None
    assert result["cell_mean_projection"][controls.CELL_CONFIDENT_WEAK] is None


# ---------------------------------------------------------------------------------------------------
# The whole vector computation on a synthetic run
# ---------------------------------------------------------------------------------------------------

def run_compute(folder, **keyword_arguments):
    return compute.compute_all(folder, dict(THRESHOLDS), "test", bootstrap_resamples=10, minimum_layer_index=1, seed=0, say=lambda *_: None,
                               facet_key=FACET_KEY, **keyword_arguments)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_the_calibrated_computation_balances_builds_the_confidence_vector_and_runs_both_controls(tmp_path):
    folder, passage_direction, _ = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", FACET_KEY, layers=6, hidden=48)
    metadata, metadata_path = run_compute(folder)
    entry = metadata["per_facet"][FACET_KEY]
    balancing = entry["stated_confidence_balancing"]
    assert set(balancing["before_balancing"]) == {"overall", *controls.BAND_GROUPS} == set(balancing["after_balancing"])
    assert balancing["before_balancing"]["overall"]["items"] == entry["filter_counts"]["items_kept"]
    assert entry["filter_counts"]["items_kept_after_balancing_stated_confidence"] == balancing["after_balancing"]["overall"]["items"]
    layer = metadata["layer_choice"]["layer_by_facet"][FACET_KEY]
    folder_of_vectors = Path(metadata_path).parent
    calibration = np.load(folder_of_vectors / metadata["vector_files"]["passage_mean_raw"])
    confidence = np.load(folder_of_vectors / metadata["vector_files"]["stated_confidence_passage_mean_raw"])
    assert (folder_of_vectors / metadata["vector_files"]["passage_mean_confidence_direction_projected_out"]).exists()
    direction = np.zeros(48)
    direction[synthetic.CONFIDENCE_DIRECTION_DIMENSIONS] = 1.0
    assert cosine(confidence[layer], direction) > 0.95  # built from confident minus hedged, the pole signal cancels
    assert cosine(calibration[layer], passage_direction) > 0.95 and abs(cosine(calibration[layer], direction)) < 0.2
    control = entry["confidence_control"]
    assert abs(control["cosine_calibration_vector_with_confidence_vector_per_layer"][layer]) < 0.3
    assert control["confidence_groups"]["answers_per_group_and_pole_cell"] > 0
    before = control["held_out_pole_classification_calibration"]["accuracy_per_layer"][layer]
    after = control["held_out_pole_classification_calibration_confidence_direction_projected_out"]["accuracy_per_layer"][layer]
    assert before >= 0.9 and after >= 0.9  # projecting an unrelated direction out leaves the calibration vector working
    assert control["held_out_pole_classification_confidence"]["accuracy_per_layer"][layer] < 0.8  # the confidence vector does not read the pole
    cells = control["two_by_two_held_out_check"]["by_layer_and_vector"][str(layer)]
    assert cells["calibration"]["interaction_contrast_calibrated_minus_miscalibrated"] > 0
    assert abs(cells["calibration"]["main_effect_of_confidence_confident_minus_hedged"]) < abs(cells["confidence"]["main_effect_of_confidence_confident_minus_hedged"])
    assert cells["confidence"]["main_effect_of_confidence_confident_minus_hedged"] > 0
    assert set(cells["calibration"]["cell_mean_projection"]) == set(controls.ALL_CELLS)


def test_poles_that_differ_in_confidence_on_every_item_stop_the_computation(tmp_path):
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(
        tmp_path / "run", FACET_KEY, layers=6, hidden=32,
        stated_confidence_function=lambda answer_id, pole, slot, sample, number: 90.0 if pole == defs.POLE_POSITIVE else 70.0)
    with pytest.raises(controls.ConfidenceBalancingError):
        run_compute(folder)


def test_an_answer_without_a_stated_confidence_score_fails_the_filter(tmp_path):
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", FACET_KEY, layers=6, hidden=32)
    judge_file = next((folder / defs.JUDGEMENTS_FOLDER_NAME).glob("judge_results_*.jsonl"))
    import json
    records = [json.loads(line) for line in judge_file.read_text().splitlines()]
    records = [r for r in records if r["kind"] != "stated_confidence"]
    judge_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(SystemExit, match="No item survived"):
        run_compute(folder)


def test_other_facets_have_no_confidence_machinery(tmp_path):
    key = "cares_about_welfare_content_controlled"
    folder, _, _ = synthetic.build_synthetic_content_controlled_run(tmp_path / "run", key, layers=6, hidden=32)
    metadata, _ = compute.compute_all(folder, dict(THRESHOLDS), "test", bootstrap_resamples=10, seed=0, say=lambda *_: None, facet_key=key)
    entry = metadata["per_facet"][key]
    assert "stated_confidence_balancing" not in entry and "confidence_control" not in entry
    assert not any("stated_confidence" in name for name in metadata["vector_files"])
    assert not D.CARES_ABOUT_WELFARE.has_confidence_control and not D.NON_INTRUSIVE.has_confidence_control


# ---------------------------------------------------------------------------------------------------
# The two arms that lower calibration
# ---------------------------------------------------------------------------------------------------

def test_expected_signs_overclaiming_and_understatement_both_lower_calibration_but_move_confidence_oppositely():
    name = CALIBRATED.confidence_control_vector_name
    main = CALIBRATED.main_vector_name
    over, under = D.ARM_TRAIT_ONLY_AWAY, D.ARM_UNDERSTATEMENT_OF_SUPPORTED_CLAIMS
    assert arms.expected_sign_on_vector_for(CALIBRATED, over, main) == arms.expected_sign_on_vector_for(CALIBRATED, under, main) == -1
    assert arms.expected_sign_on_vector_for(CALIBRATED, over, name) == 1
    assert arms.expected_sign_on_vector_for(CALIBRATED, under, name) == -1
    assert arms.expected_sign_on_vector_for(CALIBRATED, D.ARM_TRAIT_REMOVED_FROM_AWAY_REWRITE, main) == 1
    assert arms.expected_sign_on_vector_for(CALIBRATED, D.ARM_TRAIT_REMOVED_FROM_AWAY_REWRITE, name) == -1
    assert arms.expected_sign_of_any_arm(under) == -1
    for facet in (D.CARES_ABOUT_WELFARE, D.NON_INTRUSIVE):  # other facets have no confidence signs and no fifth arm
        assert D.ARM_UNDERSTATEMENT_OF_SUPPORTED_CLAIMS not in arms.arm_names_for(facet)
        assert all(arm.expected_sign_on_confidence_vector is None for arm in facet.arms)
    assert under in arms.arm_names_for(CALIBRATED) and len(arms.arm_names_for(CALIBRATED)) == 5


def test_the_understatement_instruction_exists_is_distinct_and_the_overclaim_instruction_no_longer_dismisses():
    folder = defs.PROMPT_FOLDER
    over = (folder / "positive_control_rewrite_instruction__calibrated_consideration__trait_only_away__draft.txt").read_text()
    under = (folder / "positive_control_rewrite_instruction__calibrated_consideration__understatement_of_supported_claims__draft.txt").read_text()
    assert over != under and "{document_text}" in under and "{target_changed_amount}" in under
    assert "Never overstate toward certainty that the animals do not suffer" in over
    assert "prefer certainty that they do" not in over and "certainty that they do not;" not in over
    assert "understates" in under and "open, weak or unknown" in under and "overstat" not in under.replace("understat", "")


def synthetic_summary_rows(overclaim_confidence_shift, understatement_confidence_shift, overclaim_calibration_shift=-3.0,
                           understatement_calibration_shift=-3.0):
    rows = []
    for vector, shifts in ((CALIBRATED.main_vector_name, (overclaim_calibration_shift, understatement_calibration_shift)),
                           (CALIBRATED.confidence_control_vector_name, (overclaim_confidence_shift, understatement_confidence_shift))):
        for arm, shift in zip(("trait_only_away", "understatement_of_supported_claims"), shifts):
            rows.append({"vector": vector, "document_format": "assistant_turn", "arm": arm, "changed_span_mean": shift, "whole_document_shift_mean": shift / 10})
    return rows


def test_the_side_by_side_report_shows_calibration_moving_both_arms_alike_and_confidence_moving_them_oppositely():
    rows = analysis.side_by_side_calibration_and_confidence_shifts(synthetic_summary_rows(3.0, -3.0), CALIBRATED)
    assert len(rows) == 2 and {r["measure"] for r in rows} == {"changed_span_mean", "whole_document_shift_mean"}
    for row in rows:
        assert row["calibration_vector_moves_both_arms_the_same_way"] and row["confidence_vector_moves_the_arms_oppositely"]
        assert row["calibration_vector_signs_as_expected"] and row["confidence_vector_signs_as_expected"]


def test_the_side_by_side_report_flags_swapped_signs():
    # a "calibration vector" that is really a confidence vector moves the arms oppositely, and a confidence vector that moves them alike
    # is really reading calibration: both are caught, and the expected-sign flags go false
    rows = analysis.side_by_side_calibration_and_confidence_shifts(
        synthetic_summary_rows(-3.0, -3.0, overclaim_calibration_shift=3.0, understatement_calibration_shift=-3.0), CALIBRATED)
    for row in rows:
        assert not row["calibration_vector_moves_both_arms_the_same_way"] and not row["confidence_vector_moves_the_arms_oppositely"]
        assert not row["calibration_vector_signs_as_expected"] and not row["confidence_vector_signs_as_expected"]
    assert analysis.side_by_side_calibration_and_confidence_shifts([], D.CARES_ABOUT_WELFARE) == []


def test_the_confidence_vector_is_scored_beside_the_facets_vectors_and_has_no_pass_criteria():
    names = CALIBRATED.scored_vector_names()
    assert names[-1] == CALIBRATED.confidence_control_vector_name and names[0] == CALIBRATED.main_vector_name and len(names) == 4
    assert CALIBRATED.vector_set_key_by_vector_name()[CALIBRATED.confidence_control_vector_name] == "stated_confidence_passage_mean_raw"
    assert CALIBRATED.confidence_control_vector_name not in [n for n in D.CARES_ABOUT_WELFARE.scored_vector_names()]
    import json
    criteria = json.loads(CALIBRATED.pass_criteria_file().read_text())
    assert criteria["understatement_of_supported_claims"]["counted"] is True       # counted with the trait-changed criteria on the calibration vector
    assert criteria["thresholds_fixed"] is True
    assert not [key for key in criteria if "confidence" in key and key not in ("calibrated_consideration_only",)]   # no criteria block for the control vector
