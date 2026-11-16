"""Difference between the means of two groups of scores, with a 95% bootstrap interval and a two-sample permutation p value (both seeded).

Each group is (strata, values). Strata are optional subgroups (for example questions): the bootstrap resamples values inside each stratum of each
group, so every draw keeps every stratum's number of values; the permutation test shuffles which group a value belongs to inside each stratum.
The mean is over all values of the group (all strata pooled).
"""

import numpy as np

BOOTSTRAP_DRAWS = 2000
PERMUTATION_DRAWS = 5000
RANDOM_SEED = 20261116


def _stratum_values(strata, values):
    strata, values = np.asarray(strata), np.asarray(values, float)
    return {stratum: values[strata == stratum] for stratum in np.unique(strata)}


def mean_difference_with_interval_and_permutation_p_value(first, second, seed_offset=0):
    """first and second are (strata, values). Returns a dict: mean_first, mean_second, difference (first minus second), interval_low,
    interval_high, permutation_p_value (two-sided, share of shuffles at least as extreme, with the observed one counted), count_first, count_second."""
    first_by_stratum, second_by_stratum = _stratum_values(*first), _stratum_values(*second)
    count_first, count_second = sum(map(len, first_by_stratum.values())), sum(map(len, second_by_stratum.values()))
    if not count_first or not count_second:
        return {"mean_first": np.nan, "mean_second": np.nan, "difference": np.nan, "interval_low": np.nan, "interval_high": np.nan,
                "permutation_p_value": np.nan, "count_first": count_first, "count_second": count_second}
    random_generator = np.random.default_rng(RANDOM_SEED + seed_offset)
    mean_first = sum(values.sum() for values in first_by_stratum.values()) / count_first
    mean_second = sum(values.sum() for values in second_by_stratum.values()) / count_second
    observed = mean_first - mean_second

    def bootstrap_means(by_stratum, count):
        sums = np.zeros(BOOTSTRAP_DRAWS)
        for values in by_stratum.values():
            sums += values[random_generator.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].sum(axis=1)
        return sums / count

    draws = bootstrap_means(first_by_stratum, count_first) - bootstrap_means(second_by_stratum, count_second)

    shuffled_first_sums = np.zeros(PERMUTATION_DRAWS)
    shuffled_second_sums = np.zeros(PERMUTATION_DRAWS)
    for stratum in set(first_by_stratum) | set(second_by_stratum):
        first_values, second_values = first_by_stratum.get(stratum, np.empty(0)), second_by_stratum.get(stratum, np.empty(0))
        pooled = np.concatenate([first_values, second_values])
        if not len(pooled):
            continue
        order = np.argsort(random_generator.random((PERMUTATION_DRAWS, len(pooled))), axis=1)
        shuffled = pooled[order]
        shuffled_first_sums += shuffled[:, :len(first_values)].sum(axis=1)
        shuffled_second_sums += shuffled[:, len(first_values):].sum(axis=1)
    shuffled_differences = shuffled_first_sums / count_first - shuffled_second_sums / count_second
    p_value = (1 + np.sum(np.abs(shuffled_differences) >= abs(observed) - 1e-12)) / (1 + PERMUTATION_DRAWS)
    return {"mean_first": float(mean_first), "mean_second": float(mean_second), "difference": float(observed),
            "interval_low": float(np.percentile(draws, 2.5)), "interval_high": float(np.percentile(draws, 97.5)),
            "permutation_p_value": float(p_value), "count_first": count_first, "count_second": count_second}
