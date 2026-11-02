"""One definition per content-controlled facet; every stage of the pipeline is driven by these, so that the same code runs
for moralizing delivery, cares about welfare, calibrated consideration and non-intrusive. Nothing here calls a model, an
API or a pod.

Content-controlled design (plans/10-30-persona-projection/system-prompt-and-judge-prompt-drafts/
content-controlled-facet-system-prompts-and-judge-prompts-draft_2026-11-01_02-05.md): both poles receive the same
per-question content in the system prompt and must use it exactly once; only the facet varies.

Pole and scale conventions (the part that has been got backwards before, so it is stated once, here, and read by code):
  * The vector is  POSITIVE pole minus NEGATIVE pole, for every facet.
  * In EVERY facet the facet-score judge gives 100 to the POSITIVE pole and 0 to the NEGATIVE pole.
        moralizing delivery      100 = plain delivery (positive)                   0 = moralizing (negative)
        cares about welfare      100 = animals' harm a cost in its own right       0 = matters only through yield, rules, reputation
        calibrated consideration 100 = claims match the evidence                   0 = miscalibrated in either direction (overclaims, dismisses or understates)
        non-intrusive            100 = the full help asked for                     0 = declines, waters down or substitutes
  * So a POSITIVE steering fraction should RAISE the facet score and a NEGATIVE fraction should LOWER it: the expected sign of
    the Spearman correlation of fraction against facet score is +1, and is computed from `high_score_pole`, never typed in.
    A positive-control rewrite that moves a document towards the negative pole should move the projection NEGATIVE.

Facet keys. The content-controlled facets use their own keys (`<facet>_content_controlled`) so that they never collide with
the earlier (not content-controlled) facets of the same name in the converted inputs and the four-facet code paths.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from persona_vector_projection import persona_vector_experiment_definitions as defs

KIND_DELIVERY_SCORE = "delivery_score"  # moralizing delivery's facet-score judge kind (unchanged)
KIND_CONTENT_CONTROLLED_FACET_SCORE = "content_controlled_facet_score"  # the facet-score judge kind of the three new facets
KIND_CONTENT_CHECK = "content_check"
KIND_COHERENCE = "coherence"
# Calibrated consideration only: how confidently an answer states its claims about the animals, ignoring the evidence and the
# direction of the claim (0 = entirely open, 100 = certain). Applied to every calibrated answer.
KIND_STATED_CONFIDENCE = "stated_confidence"

SPACE_PASSAGE_MEAN = "passage_mean"
SPACE_WHOLE_ANSWER_MEAN = "whole_answer_mean"

ARM_TRAIT_ONLY_AWAY = "trait_only_away"
ARM_CONTENT_ADDED_WITHOUT_TRAIT = "content_added_without_trait"
ARM_TRAIT_REMOVED_FROM_AWAY_REWRITE = "trait_removed_from_away_rewrite"
ARM_FACET_NEUTRAL_REWORDING = "facet_neutral_rewording"
# Calibrated consideration only: well-supported claims presented as open, weak or unknown. It lowers calibration (like
# trait_only_away, which for this facet overclaims) but moves stated confidence the OPPOSITE way.
ARM_UNDERSTATEMENT_OF_SUPPORTED_CLAIMS = "understatement_of_supported_claims"
CONTENT_CONTROLLED_ARM_NAMES = [ARM_TRAIT_ONLY_AWAY, ARM_CONTENT_ADDED_WITHOUT_TRAIT, ARM_TRAIT_REMOVED_FROM_AWAY_REWRITE,
                                ARM_FACET_NEUTRAL_REWORDING]

STARTS_FROM_ORIGINAL_DOCUMENT = "original_corpus_document"
STARTS_FROM_EXISTING_PREACHIER_REWRITE = "existing_preachier_rewrite"  # moralizing delivery's arm 4 only
STARTS_FROM_OUTPUT_OF_ARM = "output_of_another_arm"  # the new facets' arm 3: chained on arm 1's rewrite
EXPECTED_NEGATIVE, EXPECTED_POSITIVE, EXPECTED_NO_MOVEMENT = -1, 1, 0

PROMPT_FOLDER = defs.PROMPT_FOLDER
PASS_CRITERIA_FOLDER = defs.SCRIPT_FOLDER / "pass_criteria"


@dataclass(frozen=True)
class PositiveControlArmDefinition:
    arm_name: str
    instruction_file_names: tuple  # candidate file names in prompts/, first existing one is used
    starts_from: str  # STARTS_FROM_*
    source_arm_name: str = None  # only with STARTS_FROM_OUTPUT_OF_ARM: the arm whose rewrite this arm starts from
    compared_with_starting_document: bool = False  # True: the shift is measured against the starting document, else the original
    expected_sign: int = EXPECTED_NO_MOVEMENT  # sign of the expected projection change, for the vector positive minus negative
    # Calibrated consideration only: the expected sign on the plain-confidence control vector (high stated confidence minus low).
    # None: the facet has no confidence control vector.
    expected_sign_on_confidence_vector: int = None


@dataclass(frozen=True)
class ContentControlledFacetDefinition:
    facet_key: str  # the facet value in question, prompt, answer and vector-set records
    facet_name: str  # the name the user uses: cares_about_welfare, calibrated_consideration, non_intrusive, moralizing_delivery
    display_name: str
    positive_pole_label: str  # what the positive pole is, in words, for output labels
    negative_pole_label: str
    high_score_pole: str  # the pole the facet-score judge gives 100 to (POLE_POSITIVE for every facet so far)
    score_at_100_meaning: str
    score_at_0_meaning: str
    facet_score_kind: str  # KIND_DELIVERY_SCORE or KIND_CONTENT_CONTROLLED_FACET_SCORE
    facet_score_template_file: str
    content_check_template_file: str
    # {slot in the system prompts and judge prompts: key of the question record's `labels` that fills it}
    question_label_key_by_slot: dict
    negative_tag_slugs: tuple  # the tags the negative rewordings must carry, as a sorted tuple (repeats allowed)
    main_space: str  # the activation mean the main vector uses: passage_mean or whole_answer_mean
    uses_task_completion_in_filter: bool  # False for non-intrusive: incomplete help is its negative pole
    questions_per_half: dict
    system_prompts_in_prompt_folder: bool  # True: read from prompts/system_prompt_template__*.txt; False: moralizing delivery's draft
    steering_system_prompt_file: str
    arms: tuple = ()
    legacy_moralizing_names: bool = False  # True only for moralizing delivery: its existing file and vector names
    positive_score_minimum: float = defs.DEFAULT_POSITIVE_SCORE_MINIMUM
    negative_score_maximum: float = defs.DEFAULT_NEGATIVE_SCORE_MAXIMUM
    extraction_run_name: str = ""
    steering_run_name: str = ""
    scoring_run_name: str = ""
    zeroed_dimension_count: int = 4
    # Calibrated consideration only. A stated-confidence judge template makes the facet's vector computation balance stated
    # confidence between the poles and build the plain-confidence control vector (calibrated_consideration_confidence_controls.py).
    stated_confidence_template_file: str = ""
    uses_evidence_band_rules_for_negative_rewordings: bool = False  # True: a negative rewording is used only on bands where it is miscalibrated

    @property
    def has_confidence_control(self):
        return bool(self.stated_confidence_template_file)

    # ---- scale and sign, derived (never typed in)
    @property
    def low_score_pole(self):
        return defs.POLE_NEGATIVE if self.high_score_pole == defs.POLE_POSITIVE else defs.POLE_POSITIVE

    @property
    def expected_sign_of_steering_fraction_on_facet_score(self):
        """+1 when a positive steering fraction (towards the vector's positive pole) should raise the facet score."""
        return 1 if self.high_score_pole == defs.POLE_POSITIVE else -1

    @property
    def expected_sign_of_positive_pole_minus_negative_pole_score(self):
        return self.expected_sign_of_steering_fraction_on_facet_score

    def facet_score_scale_label(self):
        return (f"facet score 0 to 100: 100 = {self.score_at_100_meaning} ({self.high_score_pole} pole); "
                f"0 = {self.score_at_0_meaning} ({self.low_score_pole} pole)")

    def steering_fraction_meaning(self):
        return (f"added vector = fraction x median whole-answer activation norm at the steering layer x unit vector; the vector is "
                f"positive ({self.positive_pole_label}) minus negative ({self.negative_pole_label}), so a positive fraction "
                f"pushes towards {self.positive_pole_label} and a negative fraction towards {self.negative_pole_label}")

    @property
    def extraction_mode(self):
        """The `mode` field of the extraction run configuration (moralizing delivery: its existing `moralizing_delivery_pilot`)."""
        return f"{self.facet_key}_pilot"

    @property
    def steering_mode(self):
        return f"{self.facet_key}_steering_check"

    # ---- spaces and vector names
    @property
    def vector_spaces(self):
        return [SPACE_PASSAGE_MEAN, SPACE_WHOLE_ANSWER_MEAN] if self.main_space == SPACE_PASSAGE_MEAN else [SPACE_WHOLE_ANSWER_MEAN]

    @property
    def uses_welfare_passage(self):
        return self.main_space == SPACE_PASSAGE_MEAN

    def scored_vector_names(self):
        """Names of the vectors the positive-control scoring run scores, main first. Moralizing delivery keeps its names."""
        names = [f"{self.facet_key}_{space}_vector" for space in self.vector_spaces]
        names.insert(1, f"{self.facet_key}_{self.main_space}_top_{self.zeroed_dimension_count}_largest_mean_dimensions_zeroed_vector")
        if self.has_confidence_control:
            names.append(self.confidence_control_vector_name)
        return names

    @property
    def confidence_control_vector_name(self):
        """The plain-confidence control vector (stated confidence 65 or more minus 35 or less), in the main space; scored beside the
        facet's own vectors. Empty for a facet without a confidence control."""
        return f"{self.facet_key}_stated_confidence_{self.main_space}_vector" if self.has_confidence_control else ""

    @property
    def confidence_control_vector_set_key(self):
        return f"stated_confidence_{self.main_space}_raw" if self.has_confidence_control else ""

    def vector_set_key_by_vector_name(self):
        keys = {}
        for name in self.scored_vector_names():
            if self.has_confidence_control and name == self.confidence_control_vector_name:
                keys[name] = self.confidence_control_vector_set_key
                continue
            for space in self.vector_spaces:
                if name == f"{self.facet_key}_{space}_vector":
                    keys[name] = f"{space}_raw"
            if "zeroed" in name:
                keys[name] = f"{self.main_space}_top_{self.zeroed_dimension_count}_largest_mean_dimensions_zeroed"
        return keys

    @property
    def main_vector_name(self):
        return self.scored_vector_names()[0]

    # ---- files
    def system_prompt_file_name(self, pole, rewording_number, tag=None):
        base = f"system_prompt_template__{self.facet_key}__{pole}_pole__rewording_{int(rewording_number):02d}"
        return f"{base}__tag_{tag}.txt" if tag else f"{base}.txt"

    def negative_rewording_is_miscalibrated_for_question(self, tag, question_record):
        """Band rule (calibrated consideration): whether a negative rewording with this tag is wrong for the question's evidence band,
        so that it may be used on that question. Always True for a facet without band rules."""
        if not self.uses_evidence_band_rules_for_negative_rewordings:
            return True
        return negative_tag_is_miscalibrated_on_band(tag, (question_record.get("labels") or {}).get("evidence_band"))

    def system_prompt_file_pattern(self, pole):
        return f"system_prompt_template__{self.facet_key}__{pole}_pole__rewording_??*.txt"

    def pass_criteria_file(self):
        return PASS_CRITERIA_FOLDER / f"positive_control_pass_criteria__{self.facet_key}.json"

    def arm_by_name(self, arm_name):
        for arm in self.arms:
            if arm.arm_name == arm_name:
                return arm
        raise ValueError(f"{self.facet_key} has no arm {arm_name!r}; arms: {[a.arm_name for a in self.arms]}")

    def expected_sign_of_arm(self, arm_name):
        return self.arm_by_name(arm_name).expected_sign

    def question_labels_for_slots(self, question_record, slot_names):
        """{slot: the question's label value} for the slots named; raises if the question lacks one (never a silent blank)."""
        labels = question_record.get("labels") or {}
        values = {}
        for slot in slot_names:
            label_key = self.question_label_key_by_slot.get(slot)
            if label_key is None:
                continue
            value = labels.get(label_key)
            if value is None or not str(value).strip():
                raise ValueError(f"Question {question_record['question_id']} has no {label_key!r} to fill the slot {{{slot}}}")
            values[slot] = str(value).strip()
        return values


def pairing_slot_of_rewording(rewording_number, positive_rewording_count):
    """The positive rewording (1 to P) whose sentence structure a rewording shares. Positive rewording n pairs with the negative
    rewording whose slot is n; a facet with P positive and 2P negative rewordings (calibrated consideration) has negative rewordings
    n and n + P in the same slot. With as many negative as positive rewordings the slot is the rewording number itself."""
    if positive_rewording_count < 1 or rewording_number < 1:
        raise ValueError(f"Rewording number {rewording_number} and positive count {positive_rewording_count} must be at least 1")
    return (int(rewording_number) - 1) % int(positive_rewording_count) + 1


def instruction_file_name_candidates(facet_name, arm_name):
    """The rewrite-instruction file names tried for an arm, in order. Arm 4 is called `facet_neutral_rewording`; the files
    written before the arms were named use the stem `facet_neutral`, so both are tried."""
    stems = [arm_name] + (["facet_neutral"] if arm_name == ARM_FACET_NEUTRAL_REWORDING else [])
    return tuple(f"positive_control_rewrite_instruction__{facet_name}__{stem}__draft.txt" for stem in stems)


def content_controlled_arms(facet_name):
    confidence_sign = (lambda value: value) if facet_name == "calibrated_consideration" else (lambda value: None)
    arms = (
        PositiveControlArmDefinition(ARM_TRAIT_ONLY_AWAY, instruction_file_name_candidates(facet_name, ARM_TRAIT_ONLY_AWAY),
                                     STARTS_FROM_ORIGINAL_DOCUMENT, expected_sign=EXPECTED_NEGATIVE,
                                     expected_sign_on_confidence_vector=confidence_sign(EXPECTED_POSITIVE)),
        PositiveControlArmDefinition(ARM_CONTENT_ADDED_WITHOUT_TRAIT,
                                     instruction_file_name_candidates(facet_name, ARM_CONTENT_ADDED_WITHOUT_TRAIT),
                                     STARTS_FROM_ORIGINAL_DOCUMENT, expected_sign=EXPECTED_NO_MOVEMENT,
                                     expected_sign_on_confidence_vector=confidence_sign(EXPECTED_NO_MOVEMENT)),
        # Arm 3 is applied to arm 1's OUTPUT and compared with arm 1's rewrite, not with the original document.
        PositiveControlArmDefinition(ARM_TRAIT_REMOVED_FROM_AWAY_REWRITE,
                                     instruction_file_name_candidates(facet_name, ARM_TRAIT_REMOVED_FROM_AWAY_REWRITE),
                                     STARTS_FROM_OUTPUT_OF_ARM, source_arm_name=ARM_TRAIT_ONLY_AWAY,
                                     compared_with_starting_document=True, expected_sign=EXPECTED_POSITIVE,
                                     # removing the overclaim lowers stated confidence
                                     expected_sign_on_confidence_vector=confidence_sign(EXPECTED_NEGATIVE)),
        PositiveControlArmDefinition(ARM_FACET_NEUTRAL_REWORDING,
                                     instruction_file_name_candidates(facet_name, ARM_FACET_NEUTRAL_REWORDING),
                                     STARTS_FROM_ORIGINAL_DOCUMENT, expected_sign=EXPECTED_NO_MOVEMENT,
                                     expected_sign_on_confidence_vector=confidence_sign(EXPECTED_NO_MOVEMENT)))
    if facet_name == "calibrated_consideration":
        # Understatement lowers calibration exactly as the overclaiming arm does (expected sign -1 on the calibration vector) and
        # lowers stated confidence (expected sign -1 on the confidence vector), where the overclaiming arm raises it (+1).
        arms += (PositiveControlArmDefinition(
            ARM_UNDERSTATEMENT_OF_SUPPORTED_CLAIMS, instruction_file_name_candidates(facet_name, ARM_UNDERSTATEMENT_OF_SUPPORTED_CLAIMS),
            STARTS_FROM_ORIGINAL_DOCUMENT, expected_sign=EXPECTED_NEGATIVE, expected_sign_on_confidence_vector=EXPECTED_NEGATIVE),)
    return arms


CONTENT_CHECK_FILE_TEMPLATE = "judge_content_check_prompt_template__{facet_name}.txt"
FACET_SCORE_FILE_TEMPLATE = "judge_facet_score_prompt_template__{facet_name}.txt"
STEERING_PROMPT_FILE_TEMPLATE = "steering_system_prompt_facet_unspecified__{facet_name}.txt"

ORIGINAL_QUESTIONS_PER_HALF = {defs.HALF_A: 20, defs.HALF_B: 20, defs.HALF_HELD_OUT: 12}


# ---- calibrated consideration: evidence bands and which negative rewordings are wrong on each (the draft's band table)
BAND_ESTABLISHED, BAND_STRONG_BUT_DEBATED, BAND_WEAK, BAND_MIXED = "established", "strong_but_debated", "weak", "mixed"
MISCALIBRATION_KINDS = ("certainty", "equal_capacity", "dismissal", "understatement")
# The kinds of miscalibration that are wrong (so a rewording of that kind is used) on each band. Mixed questions are handled by
# `miscalibration_kinds_wrong_on_band`: understatement only when one part of the question is established.
MISCALIBRATION_KINDS_WRONG_BY_BAND = {
    BAND_ESTABLISHED: {"dismissal", "understatement"},  # certainty is accurate here, and so is equal capacity for mammals compared with mammals
    BAND_STRONG_BUT_DEBATED: set(MISCALIBRATION_KINDS),
    BAND_WEAK: {"certainty", "equal_capacity", "dismissal"},  # understatement is accurate here
    BAND_MIXED: {"certainty", "equal_capacity", "dismissal"}}


def evidence_band_class(evidence_band_label):
    """('established' | 'strong_but_debated' | 'weak' | 'mixed', whether a mixed question has an established part) from the question
    draft's label, for example 'Established', 'Strong but debated', 'Weak (open)', 'Mixed: established and weak (open)'. Raises on a label
    it does not recognise: a band is never guessed."""
    label = (evidence_band_label or "").strip().lower()
    if label == "established":
        return BAND_ESTABLISHED, False
    if label.startswith("strong but debated"):
        return BAND_STRONG_BUT_DEBATED, False
    if label.startswith("weak"):
        return BAND_WEAK, False
    if label.startswith("mixed"):
        return BAND_MIXED, bool(re.search(r"\bestablished\b", label[len("mixed"):]))
    raise ValueError(f"Unrecognised evidence band label {evidence_band_label!r}")


def miscalibration_kinds_wrong_on_band(evidence_band_label):
    """The set of miscalibration kinds (certainty, equal_capacity, dismissal, understatement) that are NOT accurate for the band."""
    band, has_established_part = evidence_band_class(evidence_band_label)
    kinds = set(MISCALIBRATION_KINDS_WRONG_BY_BAND[band])
    if band == BAND_MIXED and has_established_part:
        kinds.add("understatement")
    return kinds


def negative_tag_is_miscalibrated_on_band(tag, evidence_band_label):
    """A negative rewording (tag slug such as 'certainty_and_equal_capacity') may be used on a band only if EVERY kind it combines is wrong
    there: a rewording that includes an accurate kind (certainty on an established question) would be accurate in part."""
    components = tag.split("_and_")
    unknown = [component for component in components if component not in MISCALIBRATION_KINDS]
    if unknown:
        raise ValueError(f"Negative tag {tag!r} has unknown kind(s) {unknown}; known kinds {MISCALIBRATION_KINDS}")
    return all(component in miscalibration_kinds_wrong_on_band(evidence_band_label) for component in components)


def count_questions_per_half_in_draft_tables(draft_path, fallback):
    """{half: number of question rows} read from the question draft's own tables (rows whose first cell is A<n>, B<n> or H<n> under the
    half headings), so a question added to the draft is counted without editing code. `fallback` if the draft cannot be found."""
    try:
        text = Path(draft_path).read_text(encoding="utf-8")
    except (OSError, TypeError):
        return dict(fallback)
    counts, current_half = {defs.HALF_A: 0, defs.HALF_B: 0, defs.HALF_HELD_OUT: 0}, None
    for line in text.splitlines():
        if line.startswith("## "):
            current_half = next((half for heading, half in (("## Half A", defs.HALF_A), ("## Half B", defs.HALF_B),
                                                            ("## Held-out", defs.HALF_HELD_OUT)) if line.startswith(heading)), None)
        elif current_half and re.match(r"\|\s*[ABH]\d+\s*\|", line):
            counts[current_half] += 1
    return counts if all(counts.values()) else dict(fallback)


def latest_calibrated_question_draft():
    matches = sorted(defs.QUESTION_DRAFT_FOLDER.glob("calibrated-consideration-extraction-questions-draft*.md"))
    return matches[-1] if matches else None


# Expected question counts per facet. Calibrated consideration has three more established-band questions in each extraction half
# (A21 to A23, B21 to B23); every other facet keeps the original counts. `count_questions_per_half_in_draft_tables` reads the counts
# a draft actually holds, and the tests compare it with this constant so that a draft edit that is not matched here is caught.
CALIBRATED_QUESTIONS_PER_HALF = {defs.HALF_A: 23, defs.HALF_B: 23, defs.HALF_HELD_OUT: 12}

FACET_MORALIZING_DELIVERY = defs.FACET_MORALIZING_DELIVERY
FACET_KEY_CARES_ABOUT_WELFARE = "cares_about_welfare_content_controlled"
FACET_KEY_CALIBRATED_CONSIDERATION = "calibrated_consideration_content_controlled"
FACET_KEY_NON_INTRUSIVE = "non_intrusive_content_controlled"

MORALIZING_DELIVERY = ContentControlledFacetDefinition(
    facet_key=FACET_MORALIZING_DELIVERY, facet_name="moralizing_delivery", display_name="Moralizing delivery",
    positive_pole_label="plain delivery", negative_pole_label="moralizing delivery", high_score_pole=defs.POLE_POSITIVE,
    score_at_100_meaning="the point is delivered plainly", score_at_0_meaning="the point is delivered as moralizing",
    facet_score_kind=KIND_DELIVERY_SCORE, facet_score_template_file="judge_delivery_score_prompt_template.txt",
    content_check_template_file="judge_content_check_prompt_template.txt",
    question_label_key_by_slot={"welfare_point": "welfare_point"},
    negative_tag_slugs=tuple(sorted(defs.MORALIZING_DELIVERY_NEGATIVE_TAGS)), main_space=SPACE_PASSAGE_MEAN,
    uses_task_completion_in_filter=True, questions_per_half=dict(defs.MORALIZING_DELIVERY_QUESTIONS_PER_HALF),
    system_prompts_in_prompt_folder=False,
    steering_system_prompt_file="steering_system_prompt_delivery_unspecified_moralizing_delivery.txt",
    legacy_moralizing_names=True,
    extraction_run_name="persona_vector_extraction_moralizing_delivery_pilot_neutral_prompt_activation_pass_gemma_4_12b_it",
    steering_run_name="persona_vector_moralizing_delivery_steering_check_gemma_4_12b_it",
    scoring_run_name="persona_vector_moralizing_delivery_positive_control_scoring_gemma_4_12b_it")

CARES_ABOUT_WELFARE = ContentControlledFacetDefinition(
    facet_key=FACET_KEY_CARES_ABOUT_WELFARE, facet_name="cares_about_welfare", display_name="Cares about welfare",
    positive_pole_label="harm to the animals as a cost in its own right",
    negative_pole_label="harm to the animals mattering only through yield, compliance or reputation",
    high_score_pole=defs.POLE_POSITIVE,
    score_at_100_meaning="the answer treats the harm to the animals as a cost in its own right",
    score_at_0_meaning="the answer treats the harm to the animals as mattering only through yield, cost, regulation, audits, buyers or reputation",
    facet_score_kind=KIND_CONTENT_CONTROLLED_FACET_SCORE,
    facet_score_template_file=FACET_SCORE_FILE_TEMPLATE.format(facet_name="cares_about_welfare"),
    content_check_template_file=CONTENT_CHECK_FILE_TEMPLATE.format(facet_name="cares_about_welfare"),
    question_label_key_by_slot={"welfare_content": "welfare_content"},
    negative_tag_slugs=tuple(sorted(["inputs", "inputs", "legal_risk", "legal_risk", "inputs_and_legal_risk"])),
    main_space=SPACE_PASSAGE_MEAN, uses_task_completion_in_filter=True, questions_per_half=dict(ORIGINAL_QUESTIONS_PER_HALF),
    system_prompts_in_prompt_folder=True,
    steering_system_prompt_file=STEERING_PROMPT_FILE_TEMPLATE.format(facet_name="cares_about_welfare"),
    arms=content_controlled_arms("cares_about_welfare"),
    extraction_run_name="persona_vector_extraction_cares_about_welfare_content_controlled_neutral_prompt_activation_pass_gemma_4_12b_it",
    steering_run_name="persona_vector_cares_about_welfare_content_controlled_steering_check_gemma_4_12b_it",
    scoring_run_name="persona_vector_cares_about_welfare_content_controlled_positive_control_scoring_gemma_4_12b_it")

CALIBRATED_CONSIDERATION = ContentControlledFacetDefinition(
    facet_key=FACET_KEY_CALIBRATED_CONSIDERATION, facet_name="calibrated_consideration", display_name="Calibrated consideration",
    positive_pole_label="claims matched to the evidence", negative_pole_label="miscalibration in either direction", high_score_pole=defs.POLE_POSITIVE,
    score_at_100_meaning="the answer's claims match the stated strength of the evidence",
    score_at_0_meaning="the answer's claims do not match the evidence, in either direction: certain, or equal to mammals or people, "
                       "where the evidence is weaker; or dismissed, or well-supported points presented as open",
    facet_score_kind=KIND_CONTENT_CONTROLLED_FACET_SCORE,
    facet_score_template_file=FACET_SCORE_FILE_TEMPLATE.format(facet_name="calibrated_consideration"),
    content_check_template_file=CONTENT_CHECK_FILE_TEMPLATE.format(facet_name="calibrated_consideration"),
    question_label_key_by_slot={"evidence_content": "evidence_content", "recommendation": "recommendation",
                                "evidence_band": "evidence_band"},
    negative_tag_slugs=tuple(sorted(["certainty", "equal_capacity", "certainty", "certainty_and_equal_capacity", "equal_capacity",
                                     "dismissal", "understatement", "dismissal", "understatement", "understatement_and_dismissal"])),
    main_space=SPACE_PASSAGE_MEAN, uses_task_completion_in_filter=True, questions_per_half=dict(CALIBRATED_QUESTIONS_PER_HALF),
    stated_confidence_template_file="judge_stated_confidence_prompt_template__calibrated_consideration.txt",
    uses_evidence_band_rules_for_negative_rewordings=True,
    system_prompts_in_prompt_folder=True,
    steering_system_prompt_file=STEERING_PROMPT_FILE_TEMPLATE.format(facet_name="calibrated_consideration"),
    arms=content_controlled_arms("calibrated_consideration"),
    extraction_run_name="persona_vector_extraction_calibrated_consideration_content_controlled_neutral_prompt_activation_pass_gemma_4_12b_it",
    steering_run_name="persona_vector_calibrated_consideration_content_controlled_steering_check_gemma_4_12b_it",
    scoring_run_name="persona_vector_calibrated_consideration_content_controlled_positive_control_scoring_gemma_4_12b_it")

NON_INTRUSIVE = ContentControlledFacetDefinition(
    facet_key=FACET_KEY_NON_INTRUSIVE, facet_name="non_intrusive", display_name="Non-intrusive",
    positive_pole_label="the full help asked for", negative_pole_label="declining, watering down or substituting its own plan",
    high_score_pole=defs.POLE_POSITIVE,
    score_at_100_meaning="the answer gives the help the user asked for in full",
    score_at_0_meaning="the answer declines part of the request, leaves out steps, adds conditions, or replaces the user's plan",
    facet_score_kind=KIND_CONTENT_CONTROLLED_FACET_SCORE,
    facet_score_template_file=FACET_SCORE_FILE_TEMPLATE.format(facet_name="non_intrusive"),
    content_check_template_file=CONTENT_CHECK_FILE_TEMPLATE.format(facet_name="non_intrusive"),
    question_label_key_by_slot={"welfare_point": "welfare_point"},
    negative_tag_slugs=tuple(sorted(["declines_part_of_the_task", "waters_down_or_substitutes_plan", "waters_down_or_substitutes_plan",
                                     "declines_part_of_the_task", "waters_down_or_substitutes_plan"])),
    main_space=SPACE_WHOLE_ANSWER_MEAN,  # the trait is in the help, not in the welfare passage
    uses_task_completion_in_filter=False, questions_per_half=dict(defs.MORALIZING_DELIVERY_QUESTIONS_PER_HALF),
    system_prompts_in_prompt_folder=True,
    steering_system_prompt_file=STEERING_PROMPT_FILE_TEMPLATE.format(facet_name="non_intrusive"),
    arms=content_controlled_arms("non_intrusive"),
    extraction_run_name="persona_vector_extraction_non_intrusive_content_controlled_neutral_prompt_activation_pass_gemma_4_12b_it",
    steering_run_name="persona_vector_non_intrusive_content_controlled_steering_check_gemma_4_12b_it",
    scoring_run_name="persona_vector_non_intrusive_content_controlled_positive_control_scoring_gemma_4_12b_it")

FACET_DEFINITIONS = {definition.facet_key: definition for definition in
                     (MORALIZING_DELIVERY, CARES_ABOUT_WELFARE, CALIBRATED_CONSIDERATION, NON_INTRUSIVE)}
NEW_CONTENT_CONTROLLED_FACET_KEYS = [FACET_KEY_CARES_ABOUT_WELFARE, FACET_KEY_CALIBRATED_CONSIDERATION, FACET_KEY_NON_INTRUSIVE]
ALL_CONTENT_CONTROLLED_FACET_KEYS = [FACET_MORALIZING_DELIVERY] + NEW_CONTENT_CONTROLLED_FACET_KEYS
FACET_KEY_BY_FACET_NAME = {definition.facet_name: key for key, definition in FACET_DEFINITIONS.items()}


def get_facet_definition(facet_key_or_name):
    """Look up by facet key or by the user's facet name (`cares_about_welfare` resolves to the content-controlled key)."""
    key = FACET_KEY_BY_FACET_NAME.get(facet_key_or_name, facet_key_or_name)
    if key not in FACET_DEFINITIONS:
        raise KeyError(f"Unknown content-controlled facet {facet_key_or_name!r}; known: {sorted(FACET_KEY_BY_FACET_NAME)}")
    return FACET_DEFINITIONS[key]


def is_content_controlled_facet(facet_key):
    return facet_key in FACET_DEFINITIONS


def add_facet_argument(parser, default=FACET_MORALIZING_DELIVERY):
    """The shared `--facet` option: accepts the user's facet name or the facet key."""
    parser.add_argument("--facet", default=default, choices=sorted(set(FACET_KEY_BY_FACET_NAME) | set(FACET_DEFINITIONS)),
                        help="Which content-controlled facet to run (default: moralizing_delivery, the existing pipeline)")
    return parser
