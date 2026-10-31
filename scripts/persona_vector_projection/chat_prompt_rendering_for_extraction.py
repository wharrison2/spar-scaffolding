"""Chat-templated prompt token ids for one (system prompt, question), shared by generation and the activation pass.

The prompt is rendered with the template's assistant generation header, exactly what the model conditions on when it
writes an answer; the Gemma 4 template then ends it with an empty thought channel. Rendering goes through the
persona-check helper, which raises if the template rejects the system role or silently drops the system prompt.
The prompt is tokenized without adding special tokens (the template already writes the beginning-of-sequence token),
and both vLLM and the activation pass receive these exact ids, so prompt tokens are identical in both.
"""

import json
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for directory in (SCRIPTS_DIRECTORY, SCRIPTS_DIRECTORY / "log_probability_persona_check"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import log_probability_persona_check_prompt_rendering as prompt_rendering  # noqa: E402


def render_prompt_text(tokenizer, system_prompt, user_prompt, model_name=None):
    condition = {"condition_id": "persona_vector_extraction", "system_prompt": system_prompt}
    return prompt_rendering.render_prompt_text_with_generation_header(tokenizer, condition, user_prompt, model_name)


def prompt_token_ids(tokenizer, system_prompt, user_prompt, model_name=None):
    return prompt_rendering.tokenize_text(tokenizer, render_prompt_text(tokenizer, system_prompt, user_prompt, model_name))


def answer_token_ids(tokenizer, answer_text):
    return prompt_rendering.tokenize_text(tokenizer, answer_text)


def record_system_role_handling(tokenizer, model_name, output_path):
    """Writes what the chat template does with a system message (rendered example, and that the text survives)."""
    probe_system, probe_user = "SYSTEM PROMPT PROBE TEXT", "User message probe text."
    rendered = render_prompt_text(tokenizer, probe_system, probe_user, model_name)  # raises if dropped or rejected
    record = {"model_name": model_name, "rendered_prompt_with_system_message": rendered,
              "system_text_present_in_rendered_prompt": probe_system in rendered,
              "system_text_precedes_user_text": rendered.index(probe_system) < rendered.index(probe_user),
              "handling": "the template renders the system message as its own system turn before the user turn; the "
                          "system role is used directly, with no folding into the user message"
                          if "system" in rendered.lower().split(probe_system.lower())[0] else
                          "system text appears without its own system turn marker (folded into another turn); see the "
                          "rendered prompt"}
    Path(output_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
