"""Finding the judge's quoted welfare passage inside an answer (character span), and mapping a character span to the
answer's tokens with the tokenizer's offset mapping. Pure functions; no model, API or file access.

Locating (local step, after judging): exact match first; if that fails, a match after normalising whitespace (every run
of whitespace becomes one space) and quotation marks (curly single and double quotes become straight ones), mapped
back to character positions in the original answer. Anything else is a recorded failure, never a guess.

Token mapping (pod activation pass, and tests): the answer text is tokenized with `return_offsets_mapping=True`, and a
token belongs to the passage when its character range OVERLAPS the passage's range [start, end). Overlap rather than
containment is deliberate: a leading-space token such as " Castrating" starts one character before the passage's first
letter and is still the passage's first token. Tokens of a multi-byte character each carry that character's whole
(start, end) offsets, so the character indices here are Python string indices, never byte indices. Zero-width tokens
(empty offsets, which special tokens would have) are never part of a passage.
"""

import re

MATCH_EXACT = "found_exact"
MATCH_NORMALISED = "found_after_whitespace_and_quote_normalisation"
MATCH_FAILED = "failed"

QUOTATION_MARK_REPLACEMENTS = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
                               "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"'}


def normalise_with_index_map(text):
    """(normalised text, list giving for every normalised character the index of its source character in `text`).
    A run of whitespace becomes one space whose source is the first whitespace character of the run."""
    characters, source_indices = [], []
    previous_was_whitespace = False
    for index, character in enumerate(text):
        if character.isspace():
            if not previous_was_whitespace:
                characters.append(" ")
                source_indices.append(index)
            previous_was_whitespace = True
            continue
        previous_was_whitespace = False
        characters.append(QUOTATION_MARK_REPLACEMENTS.get(character, character))
        source_indices.append(index)
    return "".join(characters), source_indices


def locate_passage_in_answer(answer_text, passage):
    """Returns a record: status, character_start and character_end (end exclusive, indices into answer_text; None on
    failure), number_of_occurrences (of the matched form), and failure_reason. The first occurrence is used."""
    if not isinstance(passage, str) or not passage.strip():
        return {"status": MATCH_FAILED, "character_start": None, "character_end": None, "number_of_occurrences": 0,
                "failure_reason": "the judge gave no welfare passage"}
    stripped_passage = passage.strip()
    exact_start = answer_text.find(stripped_passage)
    if exact_start >= 0:
        return {"status": MATCH_EXACT, "character_start": exact_start, "character_end": exact_start + len(stripped_passage),
                "number_of_occurrences": answer_text.count(stripped_passage), "failure_reason": None}
    normalised_answer, answer_source_indices = normalise_with_index_map(answer_text)
    normalised_passage, _ = normalise_with_index_map(stripped_passage)
    normalised_passage = normalised_passage.strip()
    if normalised_passage:
        start = normalised_answer.find(normalised_passage)
        if start >= 0:
            end = start + len(normalised_passage)
            return {"status": MATCH_NORMALISED, "character_start": answer_source_indices[start],
                    "character_end": answer_source_indices[end - 1] + 1,
                    "number_of_occurrences": normalised_answer.count(normalised_passage), "failure_reason": None}
    return {"status": MATCH_FAILED, "character_start": None, "character_end": None, "number_of_occurrences": 0,
            "failure_reason": "the passage was not found in the answer, exactly or after whitespace and quote normalisation"}


def tokenize_answer_with_character_offsets(tokenizer, answer_text):
    """(token ids, list of (start, end) character offsets per token) for the answer text, tokenized exactly as the
    activation pass tokenizes answers (no special tokens added)."""
    encoding = tokenizer(answer_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    if len(offsets) != len(token_ids):
        raise ValueError("offset mapping and token ids differ in length")
    return token_ids, offsets


def token_indices_covering_character_span(offsets, character_start, character_end):
    """Indices of the tokens whose character range overlaps [character_start, character_end)."""
    if character_start is None or character_end is None or character_end <= character_start:
        raise ValueError(f"invalid character span ({character_start}, {character_end})")
    return [index for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start and token_start < character_end and token_end > character_start]
