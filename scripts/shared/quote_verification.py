"""Check that a quoted passage really appears in a document (exact or fuzzy, after normalization)."""

import difflib
import re
import unicodedata

FUZZY_MATCH_THRESHOLD = 90.0

QUOTE_CHARACTER_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"', "「": '"', "」": '"'})

try:
    from rapidfuzz import fuzz as rapidfuzz_module
except ImportError:
    rapidfuzz_module = None


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).translate(QUOTE_CHARACTER_TRANSLATION)
    return re.sub(r"\s+", " ", text).strip()


def partial_match_score(needle, haystack):
    """Best partial similarity (0-100) of needle against any window of haystack."""
    if not needle:
        return 0.0
    if rapidfuzz_module is not None:
        return float(rapidfuzz_module.partial_ratio(needle, haystack))
    if len(haystack) <= len(needle):
        return 100.0 * difflib.SequenceMatcher(None, needle, haystack, autojunk=False).ratio()
    best = 0.0
    step = max(1, len(needle) // 4)
    for start in range(0, len(haystack) - len(needle) + step, step):
        window = haystack[start:start + len(needle)]
        matcher = difflib.SequenceMatcher(None, needle, window, autojunk=False)
        if matcher.real_quick_ratio() * 100 <= best or matcher.quick_ratio() * 100 <= best:
            continue
        best = max(best, 100.0 * matcher.ratio())
    return best


PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n")


def split_into_normalized_paragraphs(document_text):
    """Paragraphs are separated by blank lines (two or more newlines with only whitespace between)."""
    paragraphs = (normalize_text(paragraph) for paragraph in PARAGRAPH_BREAK_PATTERN.split(document_text))
    return [paragraph for paragraph in paragraphs if paragraph]


def verify_quote(quote, document_text):
    """Returns (found, exact_substring, fuzzy_score, quote_spans_paragraph_break).

    A quote only counts as found if it matches within a single paragraph of the document. The
    last value is True when the quote would match only if paragraphs were joined together.
    """
    normalized_quote = normalize_text(quote)
    paragraphs = split_into_normalized_paragraphs(document_text)
    if normalized_quote and any(normalized_quote in paragraph for paragraph in paragraphs):
        return True, True, 100.0, False
    best_paragraph_score = max(
        (partial_match_score(normalized_quote, paragraph) for paragraph in paragraphs), default=0.0)
    found = best_paragraph_score >= FUZZY_MATCH_THRESHOLD
    spans_paragraph_break = False
    if not found and normalized_quote and len(paragraphs) > 1:
        joined_document = " ".join(paragraphs)
        spans_paragraph_break = (normalized_quote in joined_document
                                 or partial_match_score(normalized_quote, joined_document) >= FUZZY_MATCH_THRESHOLD)
    return found, False, round(best_paragraph_score, 1), spans_paragraph_break
