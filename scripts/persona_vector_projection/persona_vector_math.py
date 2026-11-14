"""Pure numpy functions for the persona vector computation: the answer filter, within-question differences averaged
with domains weighted equally, topic projection, layer selection, and diagnostics. No file access, no models.

Vectors have shape (layers, hidden); layer index 0 is the embedding output.
"""

import numpy as np


# ---------------------------------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------------------------------

def answer_passes_individually(positive_pole, scores, positive_minimum, negative_maximum, coherence_minimum):
    """scores: {"facet_score", "coherence"} (None when the judge call was discarded). A missing score fails."""
    facet_score, coherence = scores.get("facet_score"), scores.get("coherence")
    if facet_score is None or coherence is None or coherence < coherence_minimum:
        return False
    return facet_score >= positive_minimum if positive_pole else facet_score <= negative_maximum


def pair_passes_filter(positive_scores, negative_scores, positive_minimum, negative_maximum, coherence_minimum,
                       animal_share_maximum_difference):
    """A positive and a negative answer to the same question, rewording index and sample index are kept only if the
    positive is clearly positive (>= positive_minimum), the negative clearly negative (<= negative_maximum), both are
    coherent (>= coherence_minimum) and their animal shares differ by at most animal_share_maximum_difference."""
    if not answer_passes_individually(True, positive_scores, positive_minimum, negative_maximum, coherence_minimum):
        return False
    if not answer_passes_individually(False, negative_scores, positive_minimum, negative_maximum, coherence_minimum):
        return False
    positive_share, negative_share = positive_scores.get("share_about_animals"), negative_scores.get("share_about_animals")
    if positive_share is None or negative_share is None:
        return False
    return abs(positive_share - negative_share) <= animal_share_maximum_difference


def consistent_item_passes_filter(scores_by_version_and_pole, **thresholds):
    """Consistent: both versions of the pair must pass (all four answers: positive and negative for version a and b).
    scores_by_version_and_pole[(version, pole)] -> scores."""
    return all(pair_passes_filter(scores_by_version_and_pole[(version, "positive")],
                                  scores_by_version_and_pole[(version, "negative")], **thresholds)
               for version in ("a", "b"))


# ---------------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------------

def mean_difference_within_question(positive_activations, negative_activations):
    """Both (n kept pairs, layers, hidden), pairwise aligned. Mean of (positive - negative) over the kept pairs,
    which equals mean(positive) - mean(negative) because the two sides are kept together."""
    return np.mean(np.asarray(positive_activations, dtype=np.float32) - np.asarray(negative_activations, dtype=np.float32), axis=0)


def average_questions_with_domains_weighted_equally(question_vectors, question_domains):
    """question_vectors: (n questions, layers, hidden); question_domains: n labels. Each domain's mean over its questions,
    then the mean over domains: a domain with many kept questions does not outweigh one with few."""
    question_vectors = np.asarray(question_vectors, dtype=np.float32)
    domains = sorted(set(question_domains))
    if not domains:
        raise ValueError("No questions to average")
    domain_means = [question_vectors[[i for i, d in enumerate(question_domains) if d == domain]].mean(axis=0)
                    for domain in domains]
    return np.mean(domain_means, axis=0)


def average_consistent_versions_with_equal_weight(difference_version_a, difference_version_b):
    """Consistent: positive-minus-negative is computed per version; the two versions are averaged 50/50."""
    return 0.5 * (np.asarray(difference_version_a, dtype=np.float32) + np.asarray(difference_version_b, dtype=np.float32))


# ---------------------------------------------------------------------------------------------------
# Topic removal and diagnostics
# ---------------------------------------------------------------------------------------------------

def unit_vectors_per_layer(vectors, epsilon=1e-12):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, epsilon)


def project_out_per_layer(facet_vector, topic_vector):
    """facet_vector - (facet . topic_unit) topic_unit, at every layer."""
    topic_unit = unit_vectors_per_layer(np.asarray(topic_vector, dtype=np.float32))
    facet_vector = np.asarray(facet_vector, dtype=np.float32)
    coefficients = np.sum(facet_vector * topic_unit, axis=-1, keepdims=True)
    return facet_vector - coefficients * topic_unit


def cosine_per_layer(vectors_a, vectors_b, epsilon=1e-12):
    a, b = np.asarray(vectors_a, dtype=np.float64), np.asarray(vectors_b, dtype=np.float64)
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.maximum(denominator, epsilon)


def share_along_topic_per_layer(facet_vector, topic_vector):
    """Squared cosine: the share of the facet vector's squared norm that lay along the topic vector."""
    return cosine_per_layer(facet_vector, topic_vector) ** 2


def bootstrap_cosine_stability(question_vectors, question_domains, number_of_resamples, random_seed):
    """Resamples questions with replacement within each domain (so every domain keeps its question count), rebuilds the
    domain-weighted vector and returns cosines (resamples, layers) against the full-sample vector."""
    question_vectors = np.asarray(question_vectors, dtype=np.float32)
    domains = sorted(set(question_domains))
    indices_by_domain = {domain: [i for i, d in enumerate(question_domains) if d == domain] for domain in domains}
    full_vector = average_questions_with_domains_weighted_equally(question_vectors, question_domains)
    generator = np.random.default_rng(random_seed)
    cosines = []
    for _ in range(number_of_resamples):
        resampled_domain_means = []
        for domain in domains:
            chosen = generator.choice(indices_by_domain[domain], size=len(indices_by_domain[domain]), replace=True)
            resampled_domain_means.append(question_vectors[chosen].mean(axis=0))
        cosines.append(cosine_per_layer(np.mean(resampled_domain_means, axis=0), full_vector))
    return np.array(cosines)


# ---------------------------------------------------------------------------------------------------
# Held-out classification and layer choice
# ---------------------------------------------------------------------------------------------------

def midpoint_thresholds(positive_projections, negative_projections):
    """Per layer: the midpoint between the mean projection of positive and of negative answers (fit set)."""
    return 0.5 * (np.mean(positive_projections, axis=0) + np.mean(negative_projections, axis=0))


def classification_accuracy_per_layer(held_out_positive_projections, held_out_negative_projections, thresholds):
    """Projections (answers, layers). Accuracy of 'projection above threshold means positive pole', per layer."""
    correct_positive = np.mean(held_out_positive_projections > thresholds, axis=0)
    correct_negative = np.mean(held_out_negative_projections <= thresholds, axis=0)
    number_positive, number_negative = len(held_out_positive_projections), len(held_out_negative_projections)
    return (correct_positive * number_positive + correct_negative * number_negative) / (number_positive + number_negative)


def area_under_curve_per_layer(positive_projections, negative_projections):
    """Probability a random positive answer projects above a random negative one (ties count half), per layer."""
    positive, negative = np.asarray(positive_projections), np.asarray(negative_projections)
    wins = (positive[:, None, :] > negative[None, :, :]).mean(axis=(0, 1))
    ties = (positive[:, None, :] == negative[None, :, :]).mean(axis=(0, 1))
    return wins + 0.5 * ties


def choose_layers(accuracy_by_facet, minimum_layer_index=1, shared_fraction_of_best=0.9):
    """accuracy_by_facet: {facet: array (layers,)}. A shared layer is any layer (>= minimum_layer_index) at which every
    facet reaches at least `shared_fraction_of_best` of its own best accuracy; among those, the one with the highest mean
    accuracy over facets. Otherwise each facet gets its own best layer. Returns a dict describing the choice."""
    facets = list(accuracy_by_facet)
    allowed = slice(minimum_layer_index, None)
    best = {facet: float(np.max(np.asarray(accuracy_by_facet[facet])[allowed])) for facet in facets}
    best_layer = {facet: int(np.argmax(np.asarray(accuracy_by_facet[facet])[allowed]) + minimum_layer_index) for facet in facets}
    number_of_layers = len(next(iter(accuracy_by_facet.values())))
    qualifying = [layer for layer in range(minimum_layer_index, number_of_layers)
                  if all(accuracy_by_facet[facet][layer] >= shared_fraction_of_best * best[facet] for facet in facets)]
    if qualifying:
        mean_accuracy = {layer: float(np.mean([accuracy_by_facet[facet][layer] for facet in facets])) for layer in qualifying}
        shared_layer = max(qualifying, key=lambda layer: (mean_accuracy[layer], -layer))
        return {"strategy": "shared_layer", "shared_layer": shared_layer, "layer_by_facet": {f: shared_layer for f in facets},
                "qualifying_layers": qualifying, "best_accuracy_by_facet": best, "best_layer_by_facet": best_layer}
    return {"strategy": "per_facet_best_layer", "shared_layer": None, "layer_by_facet": dict(best_layer),
            "qualifying_layers": [], "best_accuracy_by_facet": best, "best_layer_by_facet": best_layer}
