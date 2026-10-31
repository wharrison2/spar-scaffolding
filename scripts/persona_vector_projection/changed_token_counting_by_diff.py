"""Counting how much a rewrite changed its source, by a sequence diff over tokens.

Used by the rewrite generator (to enforce a minimum number of changed tokens) and by the analysis (to find the
changed span of each rewrite). Nothing here calls a model or an API.

Two token units are supported:
  - approximate units (the default, no model files needed): whitespace-separated words, or single characters for
    languages written without spaces between words (Mandarin Chinese and Japanese);
  - any tokenizer function the caller passes in (for example Gemma's token ids, as the scoring pod records them).

The diff is difflib.SequenceMatcher with autojunk off, as in the log-probability positive-control analysis
(scripts/log_probability_persona_check/analyze_positive_control_changed_spans.py).
"""

import difflib

import numpy as np

LANGUAGES_WITHOUT_SPACES_BETWEEN_WORDS = {"Mandarin Chinese", "Japanese"}


def split_text_into_approximate_units(text, language):
    """Words for spaced languages, non-whitespace characters for languages without spaces between words."""
    if language in LANGUAGES_WITHOUT_SPACES_BETWEEN_WORDS:
        return [character for character in text if not character.isspace()]
    return text.split()


def changed_masks_from_token_sequences(source_tokens, rewrite_tokens):
    """(rewrite_changed, source_replaced): boolean arrays. A rewrite token is changed when it lies outside every
    matching block; a source token is replaced when it lies outside every matching block."""
    source_tokens, rewrite_tokens = list(source_tokens), list(rewrite_tokens)
    matcher = difflib.SequenceMatcher(None, source_tokens, rewrite_tokens, autojunk=False)
    rewrite_changed = np.ones(len(rewrite_tokens), dtype=bool)
    source_replaced = np.ones(len(source_tokens), dtype=bool)
    for block in matcher.get_matching_blocks():
        rewrite_changed[block.b:block.b + block.size] = False
        source_replaced[block.a:block.a + block.size] = False
    return rewrite_changed, source_replaced


def count_changed_units(source_text, rewrite_text, language, tokenize_function=None):
    """Number of rewrite tokens outside every matching block, plus the number of source tokens replaced.

    `tokenize_function(text) -> list` overrides the approximate units."""
    if tokenize_function is None:
        def tokenize_function(text):
            return split_text_into_approximate_units(text, language)
    rewrite_changed, source_replaced = changed_masks_from_token_sequences(
        tokenize_function(source_text), tokenize_function(rewrite_text))
    return {"changed_rewrite_units": int(rewrite_changed.sum()), "replaced_source_units": int(source_replaced.sum()),
            "rewrite_units": int(len(rewrite_changed)), "source_units": int(len(source_replaced))}


def changed_units_for_size_matching(counts):
    """One number for how large an edit was: the larger of units inserted and units removed."""
    return max(counts["changed_rewrite_units"], counts["replaced_source_units"])


def is_below_minimum_changed_units(counts, minimum_changed_units):
    """True when the rewrite changed fewer than the minimum number of tokens (inserted or removed)."""
    return changed_units_for_size_matching(counts) < minimum_changed_units
