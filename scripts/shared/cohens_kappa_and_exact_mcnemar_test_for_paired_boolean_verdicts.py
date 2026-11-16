"""Agreement and paired-comparison statistics for two boolean verdicts on the same items. Standard library only."""

from math import comb


def two_by_two_counts(pairs):
    """{"both_true", "first_true_second_false", "first_false_second_true", "both_false"} over (first, second) boolean pairs."""
    counts = {"both_true": 0, "first_true_second_false": 0, "first_false_second_true": 0, "both_false": 0}
    for first, second in pairs:
        counts[("both_true", "first_true_second_false", "first_false_second_true", "both_false")[(not first) * 2 + (not second)]] += 1
    return counts


def percent_agreement_and_cohens_kappa(counts):
    """(percent agreement, Cohen's kappa) from two_by_two_counts; each None when undefined (no pairs; kappa when both judges gave one value throughout)."""
    total = sum(counts.values())
    if not total:
        return None, None
    observed = (counts["both_true"] + counts["both_false"]) / total
    first_true = counts["both_true"] + counts["first_true_second_false"]
    second_true = counts["both_true"] + counts["first_false_second_true"]
    expected = (first_true * second_true + (total - first_true) * (total - second_true)) / total ** 2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return 100 * observed, kappa


def exact_mcnemar_two_sided_p_value(only_first_true, only_second_true):
    """Exact two-sided McNemar p value from the two discordant counts (binomial test of a 0.5 split); 1.0 when there are no discordant pairs."""
    discordant = only_first_true + only_second_true
    if not discordant:
        return 1.0
    smaller = min(only_first_true, only_second_true)
    return min(1.0, 2 * sum(comb(discordant, k) for k in range(smaller + 1)) / 2 ** discordant)
