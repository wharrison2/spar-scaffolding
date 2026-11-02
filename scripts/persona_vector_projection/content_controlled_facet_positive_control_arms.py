"""The four positive-control arms of every content-controlled facet. The first part of this module is moralizing delivery's own
set (unchanged names and values, used by its tests and scripts); the second part, `Per-facet arms`, is driven by
content_controlled_facet_definitions.py and covers all four facets. Moralizing delivery's arms:

The four positive-control arms of the moralizing delivery vector: what each arm changes, which document each rewrite is
compared with, and the direction its projection is expected to move. Nothing here calls a model, an API or a pod.

The vector is positive (plain delivery) minus negative (moralizing delivery), so a rewrite that makes the AI moralize
should move the projection NEGATIVE, and one that removes moralizing should move it POSITIVE.

Arm                                      rewrite instruction                                      compared with                         expected sign
moralizing_framing_only                  moralizing_delivery__moralizing_framing_only             the original corpus document          negative
plain_welfare_content_added              moralizing_delivery__plain_welfare_content_added         the original corpus document          none (near zero)
facet_neutral_rewording                  non_intrusive__facet_neutral (wording only)              the original corpus document          none (near zero)
moralizing_removed_from_preachier_       moralizing_delivery__moralizing_removed_from_            the PREACHIER REWRITE it was made     positive
rewrite                                  preachier_rewrite                                        from, not the original document

The preachier rewrites are the 2026-10-22 preachiness positive-control rewrites (`<source>__preachier`).
"""

ARM_MORALIZING_FRAMING_ONLY = "moralizing_framing_only"
ARM_PLAIN_WELFARE_CONTENT_ADDED = "plain_welfare_content_added"
ARM_FACET_NEUTRAL_REWORDING = "facet_neutral_rewording"
ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE = "moralizing_removed_from_preachier_rewrite"
ARMS = [ARM_MORALIZING_FRAMING_ONLY, ARM_PLAIN_WELFARE_CONTENT_ADDED, ARM_FACET_NEUTRAL_REWORDING,
        ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE]

# Where each arm's instruction text lives (all in prompts/). The neutral arm reuses the non-intrusive facet's neutral
# instruction, which keeps "how intrusive or moralizing the AI is" fixed and changes only wording.
INSTRUCTION_FILE_BY_ARM = {
    ARM_MORALIZING_FRAMING_ONLY: "positive_control_rewrite_instruction__moralizing_delivery__moralizing_framing_only__draft.txt",
    ARM_PLAIN_WELFARE_CONTENT_ADDED: "positive_control_rewrite_instruction__moralizing_delivery__plain_welfare_content_added__draft.txt",
    ARM_FACET_NEUTRAL_REWORDING: "positive_control_rewrite_instruction__non_intrusive__facet_neutral__draft.txt",
    ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE:
        "positive_control_rewrite_instruction__moralizing_delivery__moralizing_removed_from_preachier_rewrite__draft.txt"}

EXPECTED_NEGATIVE, EXPECTED_POSITIVE, EXPECTED_NO_MOVEMENT = -1, 1, 0
EXPECTED_SIGN_BY_ARM = {ARM_MORALIZING_FRAMING_ONLY: EXPECTED_NEGATIVE,
                        ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE: EXPECTED_POSITIVE,
                        ARM_PLAIN_WELFARE_CONTENT_ADDED: EXPECTED_NO_MOVEMENT,
                        ARM_FACET_NEUTRAL_REWORDING: EXPECTED_NO_MOVEMENT}
# For the two arms with no expected movement, the direction the pass criteria count as a failure: moralizing, which is
# the negative direction of the vector.
MORALIZING_DIRECTION_SIGN = -1

PREACHIER_REWRITE_DIRECTION = "preachier"
PLAINER_REWRITE_DIRECTION = "plainer"
REWRITE_FACET_NAME = "moralizing_delivery"


def preachier_rewrite_document_id(source_document_id):
    return f"{source_document_id}__{PREACHIER_REWRITE_DIRECTION}"


def comparison_document_id(arm, original_source_document_id):
    """The document a rewrite's change is measured against. Only the moralizing-removed arm starts from a rewrite."""
    if arm == ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE:
        return preachier_rewrite_document_id(original_source_document_id)
    if arm in ARMS:
        return original_source_document_id
    raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")


def rewrite_document_id(arm, original_source_document_id):
    """`<original source>__moralizing_delivery__<arm>`, whatever the arm starts from, so the ten rewrites of one source
    document are easy to find side by side."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    return f"{original_source_document_id}__{REWRITE_FACET_NAME}__{arm}"


def arm_of_rewrite_document_id(document_id):
    """The arm named in a rewrite's document id, or None for a corpus document or the old preachier / plainer rewrites."""
    parts = document_id.split("__")
    if len(parts) >= 3 and parts[-2] == REWRITE_FACET_NAME and parts[-1] in ARMS:
        return parts[-1]
    return None


def original_source_document_id_of_rewrite(document_id):
    parts = document_id.split("__")
    if arm_of_rewrite_document_id(document_id):
        return "__".join(parts[:-2])
    if parts[-1] in (PREACHIER_REWRITE_DIRECTION, PLAINER_REWRITE_DIRECTION):
        return "__".join(parts[:-1])
    return document_id


# The three vectors scored, as named in the scoring run folder and in the analysis, with the keys of the extraction run's
# vector set metadata (`vector_files`, written by compute_moralizing_delivery_vectors_from_passage_activations.py) they come from.
ZEROED_DIMENSION_COUNT = 4
VECTOR_PASSAGE_MEAN = "moralizing_delivery_passage_mean_vector"
VECTOR_PASSAGE_MEAN_TOP_4_ZEROED = "moralizing_delivery_passage_mean_top_4_largest_mean_dimensions_zeroed_vector"
VECTOR_WHOLE_ANSWER_MEAN = "moralizing_delivery_whole_answer_mean_vector"
SCORED_VECTOR_NAMES = [VECTOR_PASSAGE_MEAN, VECTOR_PASSAGE_MEAN_TOP_4_ZEROED, VECTOR_WHOLE_ANSWER_MEAN]
VECTOR_SET_KEY_BY_VECTOR_NAME = {
    VECTOR_PASSAGE_MEAN: "passage_mean_raw",
    VECTOR_PASSAGE_MEAN_TOP_4_ZEROED: f"passage_mean_top_{ZEROED_DIMENSION_COUNT}_largest_mean_dimensions_zeroed",
    VECTOR_WHOLE_ANSWER_MEAN: "whole_answer_mean_raw"}
MAIN_VECTOR_NAME = VECTOR_PASSAGE_MEAN


# ---------------------------------------------------------------------------------------------------
# Per-facet arms (driven by content_controlled_facet_definitions.py)
#
# Direction convention, the same for every facet: the vector is POSITIVE pole minus NEGATIVE pole, and shift = rewrite minus
# comparison document. A rewrite towards the negative pole should therefore give a NEGATIVE shift (expected sign -1), one towards the
# positive pole a POSITIVE shift (+1), and a content-only or wording-only rewrite no shift (0).
#   trait_only_away                   original document          -1
#   content_added_without_trait       original document           0
#   trait_removed_from_away_rewrite   applied to the OUTPUT of trait_only_away and compared with THAT rewrite   +1
#   facet_neutral_rewording           original document           0
# ---------------------------------------------------------------------------------------------------

import os  # noqa: E402

from persona_vector_projection import content_controlled_facet_definitions as facet_definitions  # noqa: E402

NEGATIVE_POLE_DIRECTION_SIGN = -1  # the sign of a shift towards the negative pole, whatever the facet (same as MORALIZING_DIRECTION_SIGN)
NEW_FACET_EXPECTED_SIGN_BY_ARM = {arm.arm_name: arm.expected_sign
                                  for facet in (facet_definitions.CARES_ABOUT_WELFARE, facet_definitions.CALIBRATED_CONSIDERATION,
                                                facet_definitions.NON_INTRUSIVE) for arm in facet.arms}


def moralizing_arm_definitions():
    """Moralizing delivery's four arms in the shared structure (its names, instruction files and comparison rule are the ones above)."""
    D = facet_definitions
    return (
        D.PositiveControlArmDefinition(ARM_MORALIZING_FRAMING_ONLY, (INSTRUCTION_FILE_BY_ARM[ARM_MORALIZING_FRAMING_ONLY],),
                                       D.STARTS_FROM_ORIGINAL_DOCUMENT, expected_sign=EXPECTED_NEGATIVE),
        D.PositiveControlArmDefinition(ARM_PLAIN_WELFARE_CONTENT_ADDED, (INSTRUCTION_FILE_BY_ARM[ARM_PLAIN_WELFARE_CONTENT_ADDED],),
                                       D.STARTS_FROM_ORIGINAL_DOCUMENT, expected_sign=EXPECTED_NO_MOVEMENT),
        D.PositiveControlArmDefinition(ARM_FACET_NEUTRAL_REWORDING, (INSTRUCTION_FILE_BY_ARM[ARM_FACET_NEUTRAL_REWORDING],),
                                       D.STARTS_FROM_ORIGINAL_DOCUMENT, expected_sign=EXPECTED_NO_MOVEMENT),
        D.PositiveControlArmDefinition(ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE,
                                       (INSTRUCTION_FILE_BY_ARM[ARM_MORALIZING_REMOVED_FROM_PREACHIER_REWRITE],),
                                       D.STARTS_FROM_EXISTING_PREACHIER_REWRITE, compared_with_starting_document=True,
                                       expected_sign=EXPECTED_POSITIVE))


def arm_definitions_for(definition):
    """The arms of a facet, in the order they are run: those that start from a rewrite of another arm come after it."""
    arms = definition.arms or moralizing_arm_definitions()
    return tuple(sorted(arms, key=lambda arm: arm.starts_from == facet_definitions.STARTS_FROM_OUTPUT_OF_ARM))


def arm_names_for(definition):
    return [arm.arm_name for arm in arm_definitions_for(definition)]


def expected_sign_for(definition, arm_name):
    for arm in arm_definitions_for(definition):
        if arm.arm_name == arm_name:
            return arm.expected_sign
    raise ValueError(f"Unknown arm {arm_name!r} for {definition.facet_key}; expected one of {arm_names_for(definition)}")


def expected_sign_on_vector_for(definition, arm_name, vector_name):
    """Expected sign of an arm's shift on one scored vector. The facet's own vectors use the arm's `expected_sign`; the plain-confidence
    control vector (calibrated consideration) uses `expected_sign_on_confidence_vector`: overclaiming should RAISE stated confidence
    (+1), understatement of supported claims should LOWER it (-1), while both lower the calibration vector (-1)."""
    if definition.has_confidence_control and vector_name == definition.confidence_control_vector_name:
        for arm in arm_definitions_for(definition):
            if arm.arm_name == arm_name:
                if arm.expected_sign_on_confidence_vector is None:
                    raise ValueError(f"Arm {arm_name} of {definition.facet_key} has no expected sign on the confidence vector")
                return arm.expected_sign_on_confidence_vector
        raise ValueError(f"Unknown arm {arm_name!r} for {definition.facet_key}")
    return expected_sign_for(definition, arm_name)


def expected_sign_of_any_arm(arm_name):
    """Expected sign by arm name across moralizing delivery and the new facets (their arm names do not overlap); 0 when unknown."""
    return EXPECTED_SIGN_BY_ARM.get(arm_name, NEW_FACET_EXPECTED_SIGN_BY_ARM.get(arm_name, 0))


def rewrite_document_id_for(definition, arm_name, original_source_document_id):
    """`<original source>__<facet key>__<arm>` (moralizing delivery's ids are the same shape)."""
    if arm_name not in arm_names_for(definition):
        raise ValueError(f"Unknown arm {arm_name!r} for {definition.facet_key}; expected one of {arm_names_for(definition)}")
    return f"{original_source_document_id}__{definition.facet_key}__{arm_name}"


def comparison_document_id_for(definition, arm_name, original_source_document_id):
    """The document a rewrite's shift is measured against. An arm that starts from another arm's output (arm 3) is compared with THAT
    rewrite, never with the original document; moralizing delivery's removal arm with the existing preachier rewrite."""
    for arm in arm_definitions_for(definition):
        if arm.arm_name != arm_name:
            continue
        if arm.starts_from == facet_definitions.STARTS_FROM_OUTPUT_OF_ARM:
            return rewrite_document_id_for(definition, arm.source_arm_name, original_source_document_id)
        if arm.starts_from == facet_definitions.STARTS_FROM_EXISTING_PREACHIER_REWRITE:
            return preachier_rewrite_document_id(original_source_document_id)
        return original_source_document_id
    raise ValueError(f"Unknown arm {arm_name!r} for {definition.facet_key}; expected one of {arm_names_for(definition)}")


def arm_of_rewrite_document_id_for(definition, document_id):
    parts = document_id.split("__")
    if len(parts) >= 3 and parts[-2] == definition.facet_key and parts[-1] in arm_names_for(definition):
        return parts[-1]
    return None


def original_source_document_id_of_rewrite_for(definition, document_id):
    parts = document_id.split("__")
    if arm_of_rewrite_document_id_for(definition, document_id):
        return "__".join(parts[:-2])
    if parts[-1] in (PREACHIER_REWRITE_DIRECTION, PLAINER_REWRITE_DIRECTION):
        return "__".join(parts[:-1])
    return document_id


def resolve_instruction_file_names(definition, prompt_directory):
    """{arm: file name} choosing, for each arm, the first candidate file that exists. Raises FileNotFoundError naming every
    candidate tried when none exists: a missing instruction is never replaced by another arm's text."""
    resolved = {}
    for arm in arm_definitions_for(definition):
        for candidate in arm.instruction_file_names:
            if os.path.exists(os.path.join(prompt_directory, candidate)):
                resolved[arm.arm_name] = candidate
                break
        else:
            raise FileNotFoundError(f"No rewrite instruction file for arm {arm.arm_name} of {definition.facet_key} in {prompt_directory}; "
                                    f"tried {list(arm.instruction_file_names)}")
    return resolved


def rewrites_file_pattern(definition):
    return f"{definition.facet_key}_positive_control_rewrites_*.jsonl"
