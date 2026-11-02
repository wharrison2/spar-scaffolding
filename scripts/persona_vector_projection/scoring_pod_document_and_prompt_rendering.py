"""Baseline generation prompts (topic variables only), document formats for the activation pass, and token position
bookkeeping (openings and bands). Nothing here calls a model or an API.

Baseline prompt: the log-probability check's generation prompt with the three stance variables (tone, resolution,
reasoning featured) removed together with their sentences, so no slot is left empty and no stray punctuation remains.
The template is prompts/baseline_document_generation_prompt_template_topic_variables_only.txt; a test checks that it
equals the original template minus exactly those three sentences.

Document formats (the document is always stripped first, as the Gemma 4 chat template trims message content):
  assistant_turn  [system: generic assistant prompt], user: a minimal neutral request, assistant: the document
  user_message    [system: generic assistant prompt], user: the document
  raw_text        the beginning-of-sequence token, then the document, no chat template
  assistant_turn_with_real_user_message
                  [system: generic assistant prompt], user: the document's own `user_message`, assistant: the document.
                  Tokenised by exactly the code of assistant_turn, so the scored range is the answer's tokens only, with
                  the same alignment; a document without a user message is refused, never given the neutral one
Only document tokens are scored. Tokens after the document are cut off (the model is causal, so they cannot change
the activations of the document's tokens). Document token 0 is the first token of the document text.
"""

import string
from dataclasses import dataclass

import numpy as np

from persona_vector_projection import scoring_pod_definitions as defs
import log_probability_persona_check_prompt_rendering as original_rendering


def load_prompt(file_name):
    return original_rendering.load_prompt_text(file_name, defs.PROMPT_FOLDER)


def baseline_template_slot_names(template_text):
    return {name for _, name, _, _ in string.Formatter().parse(template_text) if name}


def render_baseline_generation_prompt(variables, document_id, template_text=None, in_universe_facts=None):
    """Fills the baseline template with only the kept topic variables. Variables that are not kept are ignored."""
    template_text = template_text or load_prompt(defs.BASELINE_PROMPT_TEMPLATE_FILE_NAME)
    in_universe_facts = in_universe_facts or original_rendering.load_prompt_text(
        original_rendering.IN_UNIVERSE_FACTS_FILE_NAME)
    expected_slots = set(defs.KEPT_TOPIC_VARIABLE_NAMES) | {"in_universe_facts"}
    if baseline_template_slot_names(template_text) != expected_slots:
        raise ValueError(f"Baseline template slots {sorted(baseline_template_slot_names(template_text))} differ from "
                         f"the expected {sorted(expected_slots)}")
    slot_values = {"in_universe_facts": in_universe_facts}
    for name in defs.KEPT_TOPIC_VARIABLE_NAMES:
        value = variables.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Document {document_id} has a missing or empty value for '{name}': {value!r}")
        slot_values[name] = value.strip()
    return template_text.format(**slot_values)


def generic_assistant_system_prompt():
    return original_rendering.load_prompt_text(original_rendering.GENERIC_SYSTEM_PROMPT_FILE_NAME)


def baseline_chat_messages(generation_prompt, system_prompt):
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return messages + [{"role": "user", "content": generation_prompt}]


def baseline_prompt_token_ids(tokenizer, generation_prompt, system_prompt, model_name=None):
    condition = {"condition_id": "scoring_baseline_generation", "system_prompt": system_prompt}
    text = original_rendering.render_prompt_text_with_generation_header(tokenizer, condition, generation_prompt, model_name)
    return original_rendering.tokenize_text(tokenizer, text)


@dataclass
class TokenizedDocument:
    token_ids: list           # context tokens + document tokens, nothing after the document
    document_start_index: int
    document_end_index: int   # equals len(token_ids)
    document_format: str


def require_real_user_message(document_id, user_message):
    """The user message of a document scored in a format that needs one; raises when it is missing, not a string or blank."""
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError(f"Document {document_id!r} has no user_message; the format {defs.FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE} "
                         f"never falls back to the neutral user message (got {user_message!r})")
    return user_message


def tokenize_document_for_format(tokenizer, document_content, document_format, system_prompt=None,
                                 neutral_user_message=None, model_name=None, user_message=None, document_id=None):
    document_text = document_content.strip()
    if not document_text:
        raise original_rendering.DocumentTokenLocationError("Document content is empty")
    condition = {"condition_id": f"scoring_{document_format}", "system_prompt": system_prompt}
    label = original_rendering.model_label(tokenizer, model_name)
    if document_format in (defs.FORMAT_ASSISTANT_TURN, defs.FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE):
        if document_format == defs.FORMAT_ASSISTANT_TURN_WITH_REAL_USER_MESSAGE:
            user_turn_text = require_real_user_message(document_id, user_message)
        else:
            user_turn_text = neutral_user_message or load_prompt(defs.ASSISTANT_TURN_NEUTRAL_USER_MESSAGE_FILE_NAME)
        tokenized = original_rendering.tokenize_condition_and_document(
            tokenizer, condition, user_turn_text, document_text, model_name)
        start, end, ids = tokenized.document_start_index, tokenized.document_end_index, tokenized.full_token_ids
    elif document_format == defs.FORMAT_USER_MESSAGE:
        messages = original_rendering.build_chat_messages(condition, document_text)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        content_start = rendered.rfind(document_text)
        if content_start == -1:
            raise original_rendering.DocumentTokenLocationError(
                f"The chat template of {label} changed the user message content: {rendered[:500]!r}")
        prompt_text, suffix_text = rendered[:content_start], rendered[content_start + len(document_text):]
        original_rendering.assert_rendered_prompt_contains_system_prompt(rendered, condition, label)
        ids, start, end = original_rendering.locate_document_tokens(
            tokenizer, condition, prompt_text, document_text, suffix_text, label)
    elif document_format == defs.FORMAT_RAW_TEXT:
        beginning_text = tokenizer.bos_token or ""
        ids, start, end = original_rendering.locate_document_tokens(
            tokenizer, condition, beginning_text, document_text, "", label)
    else:
        raise ValueError(f"Unknown document format {document_format!r}; expected one of {defs.SELECTABLE_DOCUMENT_FORMATS}")
    return TokenizedDocument(token_ids=ids[:end], document_start_index=start, document_end_index=end,
                             document_format=document_format)


def system_prompt_for_format(document_format, system_prompt):
    """The raw text format has no chat template and so no system prompt."""
    return None if document_format == defs.FORMAT_RAW_TEXT else system_prompt


# ---------------------------------------------------------------------------------------------------
# Token positions inside the document
# ---------------------------------------------------------------------------------------------------

def position_band_indices(number_of_tokens, number_of_bands):
    """Band of each document token: floor(position * bands / tokens), so bands are equal to within one token."""
    return (np.arange(number_of_tokens) * number_of_bands) // max(1, number_of_tokens)


def summarize_per_token_values(per_token_values, number_of_bands=defs.DEFAULT_NUMBER_OF_POSITION_BANDS,
                               opening_token_count=defs.OPENING_TOKEN_COUNT):
    """per_token_values: (document tokens, ...) array. Returns means over all tokens, tokens [0, opening_token_count),
    tokens from opening_token_count on, and per position band (shape (bands, ...)); NaN where no tokens."""
    values = np.asarray(per_token_values, dtype=np.float64)
    number_of_tokens = values.shape[0]
    trailing_shape = values.shape[1:]
    nan_block = np.full(trailing_shape, np.nan)

    def mean_or_nan(selection):
        return values[selection].mean(axis=0) if np.any(selection) else nan_block

    positions = np.arange(number_of_tokens)
    bands = position_band_indices(number_of_tokens, number_of_bands)
    return {"overall": mean_or_nan(positions >= 0), "opening": mean_or_nan(positions < opening_token_count),
            "after_opening": mean_or_nan(positions >= opening_token_count),
            "bands": np.stack([mean_or_nan(bands == band) for band in range(number_of_bands)])}
