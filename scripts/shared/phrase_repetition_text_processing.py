"""Deterministic text helpers for finding phrases repeated across documents: normalising text, tokenising into
words, extracting word n-grams, counting them by document frequency, removing redundant sub-phrases and
filtering phrases made only of function words. No AI API is involved anywhere in this module."""

import re
from collections import Counter

QUOTE_AND_DASH_REPLACEMENTS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "ʼ": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "–": "-", "—": "-", "―": "-", "−": "-", "‐": "-", "‑": "-",
}

# Anything that is not a letter, digit or apostrophe separates words. Hyphens inside words split them in two,
# which keeps "stun-or-kill" style compounds matchable as ordinary phrases.
WORD_PATTERN = re.compile(r"[a-z0-9']+")

SMALL_STOPWORD_LIST = frozenset("""
a about above after again all also am an and any are aren't as at be because been before being below between both
but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further
had hadn't has hasn't have haven't having he her here hers him his how i i'd i'll i'm i've if in into is isn't it
it's its itself just let's me more most my myself no nor not of off on once only or other our ours out over own same
she should shouldn't so some such than that that's the their theirs them then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were
weren't what what's when where which while who whom why will with won't would wouldn't you you'd you'll you're
you've your yours yourself
""".split())


def normalise_text(text):
    """Lowercases and unifies curly quotes, apostrophes and dashes to their plain ASCII forms."""
    for original, replacement in QUOTE_AND_DASH_REPLACEMENTS.items():
        text = text.replace(original, replacement)
    return text.lower()


def tokenise_into_words(text):
    """Normalised words with punctuation stripped at token edges, internal apostrophes and numbers kept.
    Numbers keep internal thousands separators and decimal points as separate tokens only if split by the
    pattern, which is acceptable for phrase counting."""
    tokens = []
    for raw_token in WORD_PATTERN.findall(normalise_text(text)):
        token = raw_token.strip("'")
        if token:
            tokens.append(token)
    return tokens


def extract_word_ngrams_as_set(tokens, minimum_length, maximum_length):
    """Every distinct word n-gram (tuple of words) of the given lengths in one document."""
    found = set()
    for length in range(minimum_length, maximum_length + 1):
        for start in range(len(tokens) - length + 1):
            found.add(tuple(tokens[start:start + length]))
    return found


def count_document_frequency_of_word_ngrams(token_lists_by_document, minimum_length, maximum_length):
    """Counter mapping each n-gram to the number of documents containing it (once per document)."""
    document_frequency = Counter()
    for tokens in token_lists_by_document:
        document_frequency.update(extract_word_ngrams_as_set(tokens, minimum_length, maximum_length))
    return document_frequency


def remove_sub_phrases_covered_by_longer_phrases(document_frequency, minimum_document_frequency,
                                                 covering_fraction=0.9):
    """Drops an n-gram when a one-word-longer n-gram that contains it (as prefix or suffix) occurs in at least
    covering_fraction of its documents. Any longer containing phrase implies such a one-word extension exists, so
    checking one-word extensions removes every redundant sub-phrase. Keeps phrases with document frequency at
    least minimum_document_frequency. Returns {ngram: document_frequency}."""
    best_extension_frequency = {}
    for ngram, frequency in document_frequency.items():
        if frequency < 2:
            continue
        for shorter in (ngram[:-1], ngram[1:]):
            if best_extension_frequency.get(shorter, 0) < frequency:
                best_extension_frequency[shorter] = frequency
    surviving = {}
    for ngram, frequency in document_frequency.items():
        if frequency < minimum_document_frequency:
            continue
        if best_extension_frequency.get(ngram, 0) >= covering_fraction * frequency:
            continue
        surviving[ngram] = frequency
    return surviving


def is_made_only_of_stopwords(ngram, stopwords=SMALL_STOPWORD_LIST):
    return all(word in stopwords for word in ngram)


def remove_stopword_only_phrases(frequency_by_ngram, stopwords=SMALL_STOPWORD_LIST):
    return {ngram: frequency for ngram, frequency in frequency_by_ngram.items()
            if not is_made_only_of_stopwords(ngram, stopwords)}


def phrase_occurs_in_tokens(phrase_words, tokens):
    """True if the word sequence occurs contiguously in the token list."""
    length = len(phrase_words)
    first_word = phrase_words[0]
    for start in range(len(tokens) - length + 1):
        if tokens[start] == first_word and tuple(tokens[start:start + length]) == tuple(phrase_words):
            return True
    return False


def snippet_around_first_occurrence(phrase_words, tokens, words_either_side=15):
    """Words around the first occurrence of the phrase, with the phrase wrapped in double square brackets."""
    length = len(phrase_words)
    for start in range(len(tokens) - length + 1):
        if tuple(tokens[start:start + length]) == tuple(phrase_words):
            before = tokens[max(0, start - words_either_side):start]
            after = tokens[start + length:start + length + words_either_side]
            return " ".join(before + ["[[" + " ".join(phrase_words) + "]]"] + after)
    return ""
