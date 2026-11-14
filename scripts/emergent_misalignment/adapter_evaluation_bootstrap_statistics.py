"""Bootstrap intervals shared by the analyses that compare a base model with LoRA fine-tuned versions. 2,000 draws, seed 20261111.

 - rate_with_interval: share = events / eligible answers, resampling the answers inside each stratum (for example inside each question,
   so every question keeps its own number of answers in every draw).
 - rate_difference_with_interval: rate of one group minus rate of another, each resampled independently.
 - paired_rate_difference_over_prompts: difference between two answer sets over the same prompts, resampling whole prompts (both
   models' answers to a resampled prompt are kept together).
"""

import numpy as np

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261111


def _draw_rates_within_strata(strata, events, eligible, random_generator):
    strata, events, eligible = np.asarray(strata), np.asarray(events, float), np.asarray(eligible, float)
    numerators, denominators = np.zeros(BOOTSTRAP_DRAWS), np.zeros(BOOTSTRAP_DRAWS)
    for stratum in np.unique(strata):
        stratum_events, stratum_eligible = events[strata == stratum], eligible[strata == stratum]
        chosen = random_generator.integers(0, len(stratum_events), size=(BOOTSTRAP_DRAWS, len(stratum_events)))
        numerators += stratum_events[chosen].sum(axis=1)
        denominators += stratum_eligible[chosen].sum(axis=1)
    defined = denominators > 0
    rates = np.full(BOOTSTRAP_DRAWS, np.nan)
    rates[defined] = numerators[defined] / denominators[defined]
    return rates


def _interval(draws):
    draws = draws[~np.isnan(draws)]
    if len(draws) < BOOTSTRAP_DRAWS / 2:
        return np.nan, np.nan
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def rate_with_interval(strata, events, eligible, seed_offset=0):
    """(rate, low, high, events, eligible). `events` must already be zero wherever `eligible` is zero."""
    events, eligible = np.asarray(events, float), np.asarray(eligible, float)
    rate = events.sum() / eligible.sum() if eligible.sum() else np.nan
    low, high = _interval(_draw_rates_within_strata(strata, events, eligible, np.random.default_rng(BOOTSTRAP_SEED + seed_offset)))
    return float(rate), low, high, int(events.sum()), int(eligible.sum())


def rate_difference_with_interval(first, second, seed_offset=0):
    """first and second are (strata, events, eligible) triples. Returns (difference first - second, low, high)."""
    random_generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    first_draws = _draw_rates_within_strata(*first, random_generator)
    second_draws = _draw_rates_within_strata(*second, random_generator)
    rate = lambda triple: np.asarray(triple[1], float).sum() / np.asarray(triple[2], float).sum()  # noqa: E731
    low, high = _interval(first_draws - second_draws)
    return float(rate(first) - rate(second)), low, high


def paired_rate_difference_over_prompts(prompt_ids_first, events_first, eligible_first, prompt_ids_second, events_second,
                                        eligible_second, seed_offset=0):
    """Difference (first - second) of rates over the same prompts, resampling whole prompts."""
    prompts = np.union1d(np.unique(prompt_ids_first), np.unique(prompt_ids_second))

    def per_prompt(prompt_ids, events, eligible):
        index = np.searchsorted(prompts, np.asarray(prompt_ids))
        return (np.bincount(index, weights=np.asarray(events, float), minlength=len(prompts)),
                np.bincount(index, weights=np.asarray(eligible, float), minlength=len(prompts)))

    first_events, first_eligible = per_prompt(prompt_ids_first, events_first, eligible_first)
    second_events, second_eligible = per_prompt(prompt_ids_second, events_second, eligible_second)
    chosen = np.random.default_rng(BOOTSTRAP_SEED + seed_offset).integers(0, len(prompts), size=(BOOTSTRAP_DRAWS, len(prompts)))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (first_events[chosen].sum(axis=1) / first_eligible[chosen].sum(axis=1)
                 - second_events[chosen].sum(axis=1) / second_eligible[chosen].sum(axis=1))
    low, high = _interval(draws)
    return float(first_events.sum() / first_eligible.sum() - second_events.sum() / second_eligible.sum()), low, high
