"""Choosing the corpus documents that are rewritten for each facet's positive control.

A document has "room to move" for a facet when the AI in it can plausibly be made to show the trait more AND less.
Selection is configurable and deterministic for (corpus, facet, number, seed):
  1. hard exclusions on spec variables (for example a minor-detail centrality, or a document that by design raises no
     welfare consideration at all), from FACET_SELECTION_RULES or a rules JSON file;
  2. when a depiction-label file is given, only documents whose label for the facet is in `required_depiction_labels`
     (the AI shows the trait, so the "away" rewrite has something to remove); if fewer documents qualify than are
     wanted, the requirement is relaxed with a warning rather than silently;
  3. preference (not exclusion): documents matching `prefer_spec_values`, and, for the consistent facet, documents
     that discuss more than one kind of animal (from the depiction labels) are drawn first;
  4. stratified by language: proportional allocation (largest remainder), with a floor of one document for every
     language that has at least `language_floor_minimum_eligible_documents` eligible documents.

Nothing here calls a model or an API.
"""

import random

FACET_CARES_ABOUT_WELFARE = "cares_about_welfare"
FACET_CALIBRATED_CONSIDERATION = "calibrated_consideration"
FACET_CONSISTENT = "consistent"
FACET_NON_INTRUSIVE = "non_intrusive"

MINOR_DETAIL_CENTRALITY_PREFIX = "a minor detail mentioned only in passing"
NO_WELFARE_RESOLUTION_PHRASE = "correctly raises no welfare consideration at all"
NOT_CENTRAL_CENTRALITY_PREFIX = "not the central subject"

# Spec values that mark the sentient category as one with a thin or uncertain evidence base (room for calibration).
EVIDENCE_UNCERTAIN_SENTIENT_CATEGORY_MARKERS = ["invertebrates", "uncertain"]

LABEL_SHOWS_TRAIT = "shows_trait"
LABEL_SHOWS_AVOIDING_TRAIT = "shows_avoiding_trait"
LABEL_NEITHER = "neither"

COMMON_EXCLUSIONS = {"centrality": [MINOR_DETAIL_CENTRALITY_PREFIX], "resolution": [NO_WELFARE_RESOLUTION_PHRASE]}

FACET_SELECTION_RULES = {
    FACET_CARES_ABOUT_WELFARE: {
        "excluded_spec_values": COMMON_EXCLUSIONS,
        "prefer_spec_values": {},
        "required_depiction_labels": [LABEL_SHOWS_TRAIT],
        "prefer_more_than_one_kind_of_animal": False},
    FACET_CALIBRATED_CONSIDERATION: {
        "excluded_spec_values": COMMON_EXCLUSIONS,
        # documents about animals whose sentience evidence is thin have the most calibration to move
        "prefer_spec_values": {"sentient_category": EVIDENCE_UNCERTAIN_SENTIENT_CATEGORY_MARKERS},
        "required_depiction_labels": [LABEL_SHOWS_TRAIT],
        "prefer_more_than_one_kind_of_animal": False},
    FACET_CONSISTENT: {
        "excluded_spec_values": COMMON_EXCLUSIONS,
        "prefer_spec_values": {},
        "required_depiction_labels": [LABEL_SHOWS_TRAIT],
        "prefer_more_than_one_kind_of_animal": True},
}

DEFAULT_LANGUAGE_FLOOR_MINIMUM_ELIGIBLE_DOCUMENTS = 25


def spec_value_matches_any_marker(spec_value, markers):
    """True when the spec value starts with, or contains, any marker (prefix markers are the long sentences)."""
    return any(spec_value.startswith(marker) or marker in spec_value for marker in markers)


def passes_hard_exclusions(document, excluded_spec_values):
    variables = document["variables"]
    return not any(spec_value_matches_any_marker(str(variables.get(variable, "")), markers)
                   for variable, markers in excluded_spec_values.items())


def matches_preference(document, prefer_spec_values):
    variables = document["variables"]
    return any(spec_value_matches_any_marker(str(variables.get(variable, "")), markers)
               for variable, markers in prefer_spec_values.items())


def allocate_documents_per_language(number_of_documents, eligible_counts_by_language, floor_minimum_eligible):
    """Proportional (largest remainder), with at least one for every language that has at least
    `floor_minimum_eligible` eligible documents, and never more than a language has. Total is exactly
    min(number_of_documents, total eligible)."""
    total_eligible = sum(eligible_counts_by_language.values())
    number_to_allocate = min(number_of_documents, total_eligible)
    exact = {language: number_to_allocate * count / total_eligible
             for language, count in eligible_counts_by_language.items()}
    allocation = {language: min(int(value), eligible_counts_by_language[language])
                  for language, value in exact.items()}
    for language, count in eligible_counts_by_language.items():
        if count >= floor_minimum_eligible and allocation[language] == 0:
            allocation[language] = 1
    # trim if the floors overshot: take from the languages with the largest allocation
    while sum(allocation.values()) > number_to_allocate:
        donor = max((language for language in allocation if allocation[language] > 1),
                    key=lambda language: (allocation[language], language), default=None)
        if donor is None:
            donor = max(allocation, key=lambda language: (allocation[language], language))
        allocation[donor] -= 1
    # top up by largest remainder
    while sum(allocation.values()) < number_to_allocate:
        candidates = [language for language in allocation if allocation[language] < eligible_counts_by_language[language]]
        best = max(candidates, key=lambda language: (exact[language] - allocation[language], language))
        allocation[best] += 1
    return allocation


def number_of_kinds_of_animals(depiction_label_record):
    if not depiction_label_record:
        return 0
    return len({name.strip().lower() for name in depiction_label_record.get("kinds_of_animals_discussed", [])
                if name.strip()})


def select_documents_for_facet(corpus, facet, number_of_documents, seed, depiction_labels_by_document_id=None,
                               rules=None, language_floor_minimum_eligible_documents=
                               DEFAULT_LANGUAGE_FLOOR_MINIMUM_ELIGIBLE_DOCUMENTS):
    """Returns (selected documents sorted by doc_id, selection_report dict). Raises KeyError for a facet with no
    rules (non_intrusive reuses the preachiness rewrites and is not selected here)."""
    rules = rules or FACET_SELECTION_RULES[facet]
    warnings = []
    eligible = [document for document in sorted(corpus, key=lambda document: document["doc_id"])
                if passes_hard_exclusions(document, rules["excluded_spec_values"])]
    labels = depiction_labels_by_document_id
    if labels is not None and rules["required_depiction_labels"]:
        with_label = [document for document in eligible
                      if labels.get(document["doc_id"], {}).get("labels", {}).get(facet)
                      in rules["required_depiction_labels"]]
        if len(with_label) >= number_of_documents:
            eligible = with_label
        else:
            warnings.append(f"Only {len(with_label)} documents have a depiction label in "
                            f"{rules['required_depiction_labels']} for {facet}; the label requirement was relaxed.")
    elif labels is None and rules["required_depiction_labels"]:
        warnings.append("No depiction labels given: 'room to move' is judged from spec variables only.")

    def preference_rank(document):
        """0 is the most preferred."""
        rank = 0
        if rules["prefer_spec_values"] and not matches_preference(document, rules["prefer_spec_values"]):
            rank += 1
        if rules["prefer_more_than_one_kind_of_animal"]:
            if labels is None:
                pass
            elif number_of_kinds_of_animals(labels.get(document["doc_id"])) < 2:
                rank += 1
        return rank

    if rules["prefer_more_than_one_kind_of_animal"] and labels is None:
        warnings.append("No depiction labels given: cannot prefer documents that discuss more than one kind of animal.")
    by_language = {}
    for document in eligible:
        by_language.setdefault(document["language"], []).append(document)
    allocation = allocate_documents_per_language(
        number_of_documents, {language: len(documents) for language, documents in by_language.items()},
        language_floor_minimum_eligible_documents)
    random_generator = random.Random(f"{seed}:{facet}")
    selected = []
    for language in sorted(by_language):
        pool = list(by_language[language])
        random_generator.shuffle(pool)
        pool.sort(key=preference_rank)  # stable: shuffled order kept within a rank
        selected.extend(pool[:allocation[language]])
    selected.sort(key=lambda document: document["doc_id"])
    report = {"facet": facet, "eligible_documents": len(eligible), "selected_documents": len(selected),
              "allocation_per_language": allocation,
              "selected_with_best_preference_rank": sum(preference_rank(document) == 0 for document in selected),
              "warnings": warnings}
    if len(selected) < number_of_documents:
        report["warnings"].append(f"Only {len(selected)} documents are available, fewer than the {number_of_documents} "
                                  "requested.")
    return selected, report
