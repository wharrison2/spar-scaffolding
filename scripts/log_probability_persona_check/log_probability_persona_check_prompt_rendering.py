"""Prompt files, the 62 system-prompt conditions, generation prompts, chat messages, and the location of
a document's content tokens inside the chat-templated sequence.

Shared by build_scoring_inputs.py, score_document_log_probabilities_on_pod.py and the tests. Nothing here
calls a model or an API. Layout contract: run_folder_layout_and_script_interfaces.md in this folder.

Which tokens are scored: the sequence is  <chat template: system turn (if any), user turn, then the
assistant header exactly as the model sees it when it is about to generate>  +  <document content>  +
<whatever the chat template writes after an assistant message, normally the end-of-turn token>.
Only the document content tokens are scored. The first content token is scored conditioned on the
assistant header; the end-of-turn token is never scored.

Gemma 4 note: with thinking off, the template's generation prompt ends with an empty thought channel
("<|turn>model\\n<|channel>thought\\n<channel|>") whereas the template renders a finished assistant message
without it. We score under the generation prompt (what the model conditions on when it writes a document)
and take only the text after the assistant content from the finished-message rendering.
"""

import re
import string
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompts"

IN_UNIVERSE_FACTS_FILE_NAME = "in_universe_facts_shared_by_all_documents.txt"
GENERATION_PROMPT_TEMPLATE_FILE_NAME = "document_generation_prompt_template.txt"
GENERIC_SYSTEM_PROMPT_FILE_NAME = "generic_assistant_system_prompt.txt"
PERSONA_FRAME_FILE_NAME = "persona_system_prompt_frame.txt"
PREACHY_DESCRIPTIONS_FILE_NAME = "preachy_persona_descriptions.txt"
CONTROL_DESCRIPTIONS_FILE_NAME = "control_persona_descriptions_without_moral_stance.txt"
PROPORTIONATE_WELFARE_DESCRIPTIONS_FILE_NAME = "proportionate_welfare_control_persona_descriptions.txt"

# Prompt files that determine what is scored; their hashes go into run_configuration.
PROMPT_FILE_NAMES_USED_FOR_SCORING = [
    IN_UNIVERSE_FACTS_FILE_NAME,
    GENERATION_PROMPT_TEMPLATE_FILE_NAME,
    GENERIC_SYSTEM_PROMPT_FILE_NAME,
    PERSONA_FRAME_FILE_NAME,
    PREACHY_DESCRIPTIONS_FILE_NAME,
    CONTROL_DESCRIPTIONS_FILE_NAME,
    PROPORTIONATE_WELFARE_DESCRIPTIONS_FILE_NAME,
]

SPEC_VARIABLE_NAMES = ["document_type", "culture", "domain", "sentient_category", "decision_scale", "naming",
                       "tech_savvy", "tone", "tradeoff", "centrality", "resolution", "reasoning_featured"]
GENERATION_PROMPT_SLOT_NAMES = SPEC_VARIABLE_NAMES + ["in_universe_facts"]

EXPECTED_PREACHY_REWORDING_COUNT = 30
EXPECTED_CONTROL_PERSONA_COUNT = 5
EXPECTED_REWORDINGS_PER_CONTROL_PERSONA = 5
EXPECTED_PROPORTIONATE_WELFARE_REWORDING_COUNT = 5
EXPECTED_CONDITION_COUNT = 62

CLOSING_PUNCTUATION_AND_QUOTES = "\"'”’)]」』»"
TERMINAL_PUNCTUATION = ".!?。！？…।۔"


# ---------------------------------------------------------------------------------------------------
# Prompt files and conditions
# ---------------------------------------------------------------------------------------------------

def load_prompt_text(file_name, prompt_directory=None):
    path = Path(prompt_directory or PROMPT_DIRECTORY) / file_name  # resolved at call time (tests redirect it)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def parse_persona_description_lines(file_name, prompt_directory=None):
    """One description per non-empty line."""
    text = load_prompt_text(file_name, prompt_directory)
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_grouped_persona_descriptions(file_name, prompt_directory=None):
    """Returns {persona_name: [descriptions]} in file order. Persona names are the `# name` headers."""
    text = load_prompt_text(file_name, prompt_directory)
    descriptions_by_persona = {}
    current_persona_name = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_persona_name = line.lstrip("#").strip()
            if not current_persona_name:
                raise ValueError(f"Empty persona header in {file_name}")
            if current_persona_name in descriptions_by_persona:
                raise ValueError(f"Duplicate persona header '{current_persona_name}' in {file_name}")
            descriptions_by_persona[current_persona_name] = []
        else:
            if current_persona_name is None:
                raise ValueError(f"Description line before any '# persona_name' header in {file_name}: {line}")
            descriptions_by_persona[current_persona_name].append(line)
    return descriptions_by_persona


def render_persona_system_prompt(persona_frame, persona_description):
    if "{persona_description}" not in persona_frame:
        raise ValueError("Persona frame has no {persona_description} slot")
    return persona_frame.replace("{persona_description}", persona_description)


def build_condition_records(prompt_directory=None):
    """The 62 system-prompt conditions, in a fixed order: no_system_prompt, generic, 30 preachy,
    25 control (5 personas x 5), 5 proportionate_welfare. Rewording indices are 1-based."""
    persona_frame = load_prompt_text(PERSONA_FRAME_FILE_NAME, prompt_directory)
    generic_system_prompt = load_prompt_text(GENERIC_SYSTEM_PROMPT_FILE_NAME, prompt_directory)
    preachy_descriptions = parse_persona_description_lines(PREACHY_DESCRIPTIONS_FILE_NAME, prompt_directory)
    control_descriptions_by_persona = parse_grouped_persona_descriptions(CONTROL_DESCRIPTIONS_FILE_NAME,
                                                                         prompt_directory)
    welfare_descriptions = parse_persona_description_lines(PROPORTIONATE_WELFARE_DESCRIPTIONS_FILE_NAME,
                                                           prompt_directory)

    if len(preachy_descriptions) != EXPECTED_PREACHY_REWORDING_COUNT:
        raise ValueError(f"Expected {EXPECTED_PREACHY_REWORDING_COUNT} preachy rewordings, found "
                         f"{len(preachy_descriptions)}")
    if len(control_descriptions_by_persona) != EXPECTED_CONTROL_PERSONA_COUNT:
        raise ValueError(f"Expected {EXPECTED_CONTROL_PERSONA_COUNT} control personas, found "
                         f"{len(control_descriptions_by_persona)}")
    for persona_name, descriptions in control_descriptions_by_persona.items():
        if len(descriptions) != EXPECTED_REWORDINGS_PER_CONTROL_PERSONA:
            raise ValueError(f"Control persona {persona_name} has {len(descriptions)} rewordings, expected "
                             f"{EXPECTED_REWORDINGS_PER_CONTROL_PERSONA}")
    if len(welfare_descriptions) != EXPECTED_PROPORTIONATE_WELFARE_REWORDING_COUNT:
        raise ValueError(f"Expected {EXPECTED_PROPORTIONATE_WELFARE_REWORDING_COUNT} proportionate-welfare "
                         f"rewordings, found {len(welfare_descriptions)}")

    conditions = [
        {"condition_id": "no_system_prompt", "condition_group": "no_system_prompt",
         "persona_name": "no_system_prompt", "rewording_index": None, "system_prompt": None},
        {"condition_id": "generic", "condition_group": "generic", "persona_name": "generic",
         "rewording_index": None, "system_prompt": generic_system_prompt},
    ]

    def add_persona_conditions(condition_group, persona_name, descriptions):
        for rewording_index, description in enumerate(descriptions, start=1):
            conditions.append({
                "condition_id": f"{persona_name}__rewording_{rewording_index:02d}",
                "condition_group": condition_group, "persona_name": persona_name,
                "rewording_index": rewording_index,
                "system_prompt": render_persona_system_prompt(persona_frame, description)})

    add_persona_conditions("preachy", "preachy", preachy_descriptions)
    for persona_name, descriptions in control_descriptions_by_persona.items():
        add_persona_conditions("control", persona_name, descriptions)
    add_persona_conditions("proportionate_welfare", "proportionate_welfare", welfare_descriptions)

    condition_ids = [condition["condition_id"] for condition in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("Duplicate condition ids")
    if len(conditions) != EXPECTED_CONDITION_COUNT:
        raise ValueError(f"Expected {EXPECTED_CONDITION_COUNT} conditions, built {len(conditions)}")
    return conditions


# ---------------------------------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------------------------------

def add_final_period_if_missing(resolution_text):
    """Adds "." to `resolution` unless it already ends in terminal punctuation (possibly followed by
    closing quotes or brackets). Trailing whitespace is dropped."""
    text = resolution_text.rstrip()
    if not text:
        raise ValueError("resolution is empty")
    last_meaningful_character = text.rstrip(CLOSING_PUNCTUATION_AND_QUOTES)[-1:] or text[-1]
    if last_meaningful_character in TERMINAL_PUNCTUATION:
        return text
    return text + "."


def check_template_slots(template_text):
    slot_names = {field_name for _, field_name, _, _ in string.Formatter().parse(template_text) if field_name}
    if slot_names != set(GENERATION_PROMPT_SLOT_NAMES):
        raise ValueError(f"Generation prompt template slots {sorted(slot_names)} differ from the expected "
                         f"{sorted(GENERATION_PROMPT_SLOT_NAMES)}")


def render_generation_prompt(variables, document_id, prompt_directory=None,
                             template_text=None, in_universe_facts=None):
    """Fills the generation prompt template for one document. Fails loudly on any missing, non-string or
    empty spec variable."""
    if template_text is None:
        template_text = load_prompt_text(GENERATION_PROMPT_TEMPLATE_FILE_NAME, prompt_directory)
    if in_universe_facts is None:
        in_universe_facts = load_prompt_text(IN_UNIVERSE_FACTS_FILE_NAME, prompt_directory)
    check_template_slots(template_text)
    missing = [name for name in SPEC_VARIABLE_NAMES if name not in variables]
    if missing:
        raise KeyError(f"Document {document_id} is missing spec variables: {missing}")
    slot_values = {"in_universe_facts": in_universe_facts}
    for name in SPEC_VARIABLE_NAMES:
        value = variables[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Document {document_id} has an empty or non-string value for '{name}': {value!r}")
        slot_values[name] = value.strip()
    slot_values["resolution"] = add_final_period_if_missing(slot_values["resolution"])
    return template_text.format(**slot_values)


def build_chat_messages(condition, generation_prompt):
    messages = []
    if condition["system_prompt"] is not None:
        messages.append({"role": "system", "content": condition["system_prompt"]})
    messages.append({"role": "user", "content": generation_prompt})
    return messages


# ---------------------------------------------------------------------------------------------------
# Chat template rendering and document token location
# ---------------------------------------------------------------------------------------------------

class SystemPromptNotInRenderedPromptError(RuntimeError):
    pass


class DocumentTokenLocationError(RuntimeError):
    pass


def model_label(tokenizer, model_name):
    return model_name or getattr(tokenizer, "name_or_path", "unknown model")


def assert_rendered_prompt_contains_system_prompt(rendered_text, condition, model_name):
    """Raises if a condition with a system prompt does not have that text in the rendered chat prompt."""
    system_prompt = condition["system_prompt"]
    if system_prompt is None:
        return
    if system_prompt.strip() not in rendered_text:
        raise SystemPromptNotInRenderedPromptError(
            f"The chat template of {model_name} dropped the system prompt of condition "
            f"{condition['condition_id']}. Rendered text:\n{rendered_text}")


def render_prompt_text_with_generation_header(tokenizer, condition, generation_prompt, model_name=None):
    """Renders system (if any) + user turns with the template's assistant generation header. Raises if the
    template rejects the system role or silently drops the system prompt."""
    label = model_label(tokenizer, model_name)
    messages = build_chat_messages(condition, generation_prompt)
    try:
        rendered_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as error:
        raise SystemPromptNotInRenderedPromptError(
            f"The chat template of {label} failed on condition {condition['condition_id']} "
            f"(roles: {[message['role'] for message in messages]}): {error}") from error
    assert_rendered_prompt_contains_system_prompt(rendered_text, condition, label)
    return rendered_text


def render_text_after_assistant_message(tokenizer, condition, generation_prompt, document_text, model_name=None):
    """Text the chat template writes after an assistant message whose content is `document_text`
    (normally the end-of-turn token). Also returns the text before the content in that rendering."""
    label = model_label(tokenizer, model_name)
    messages = build_chat_messages(condition, generation_prompt)
    messages.append({"role": "assistant", "content": document_text})
    reference_text = tokenizer.apply_chat_template(messages, tokenize=False)
    assert_rendered_prompt_contains_system_prompt(reference_text, condition, label)
    content_start = reference_text.rfind(document_text)
    if content_start == -1:
        raise DocumentTokenLocationError(
            f"The chat template of {label} changed the assistant content, so the text after it cannot be "
            f"located. Rendered text:\n{reference_text[:2000]}")
    return reference_text[content_start + len(document_text):], reference_text[:content_start]


def tokenize_text(tokenizer, text):
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


@dataclass
class TokenizedConditionAndDocument:
    full_token_ids: list
    document_start_index: int  # first document content token
    document_end_index: int    # one past the last document content token
    prompt_text: str
    suffix_text: str
    assistant_header_matches_finished_message_rendering: bool


def tokenize_condition_and_document(tokenizer, condition, generation_prompt, document_content, model_name=None):
    """Returns the full token ids for (condition, document) and the index range of the document content
    tokens. Position i of the full sequence is predicted by the hidden state at position i - 1.

    Verified here: the system prompt survives the template; the tokens of the prompt are an exact prefix of
    the full sequence (so the first document token is scored given the assistant header and nothing leaks
    in from the document); the tokens after the document are an exact tail; the content tokens decode back
    to the document text apart from whitespace at the edges.
    """
    label = model_label(tokenizer, model_name)
    document_text = document_content.strip()
    if not document_text:
        raise DocumentTokenLocationError("Document content is empty")
    prompt_text = render_prompt_text_with_generation_header(tokenizer, condition, generation_prompt, model_name)
    suffix_text, text_before_content = render_text_after_assistant_message(
        tokenizer, condition, generation_prompt, document_text, model_name)

    prompt_token_ids = tokenize_text(tokenizer, prompt_text)
    suffix_token_ids = tokenize_text(tokenizer, suffix_text) if suffix_text else []
    full_token_ids = tokenize_text(tokenizer, prompt_text + document_text + suffix_text)

    if full_token_ids[:len(prompt_token_ids)] != prompt_token_ids:
        raise DocumentTokenLocationError(
            f"For {label}, the tokens of the prompt (condition {condition['condition_id']}) are not an exact "
            f"prefix of the tokens of prompt + document, so the document's first token would merge with the "
            f"prompt. Prompt ends with {prompt_text[-80:]!r}; document starts with {document_text[:40]!r}.")
    document_start_index = len(prompt_token_ids)
    document_end_index = len(full_token_ids) - len(suffix_token_ids)
    if suffix_token_ids and full_token_ids[document_end_index:] != suffix_token_ids:
        raise DocumentTokenLocationError(
            f"For {label}, the tokens after the document are not an exact tail of the sequence "
            f"(suffix text {suffix_text!r}); the document's last token would merge with them.")
    if document_end_index <= document_start_index:
        raise DocumentTokenLocationError(f"No document content tokens found for {label}")

    decoded_content = tokenizer.decode(full_token_ids[document_start_index:document_end_index],
                                       skip_special_tokens=False)
    if decoded_content.strip() != document_text:
        mismatch_position = next((position for position, (left, right)
                                  in enumerate(zip(decoded_content.strip(), document_text)) if left != right),
                                 min(len(decoded_content.strip()), len(document_text)))
        raise DocumentTokenLocationError(
            f"For {label}, the document content tokens do not decode back to the document text "
            f"(first difference at character {mismatch_position}: decoded "
            f"{decoded_content.strip()[mismatch_position:mismatch_position + 60]!r} vs original "
            f"{document_text[mismatch_position:mismatch_position + 60]!r}).")

    return TokenizedConditionAndDocument(
        full_token_ids=full_token_ids, document_start_index=document_start_index,
        document_end_index=document_end_index, prompt_text=prompt_text, suffix_text=suffix_text,
        assistant_header_matches_finished_message_rendering=(text_before_content == prompt_text))


def decode_each_token(tokenizer, token_ids):
    return [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]
