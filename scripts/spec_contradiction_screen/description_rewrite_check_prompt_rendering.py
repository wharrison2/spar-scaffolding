"""Schemas, prompt rendering and validation for GPT 6.1 Sol's two checks on rewritten descriptions.

Pure functions (no model calls, no `shared` imports) so the cost estimate can render the exact prompts.
  problem_gone: for a description GPT labelled problematic, is the labelled problem gone in the rewrite?
  dealt_values_kept: does the rewrite still carry every dealt value?
"""

import spec_screen_common as common

PROBLEM_GONE_PROMPT_PATH = common.SCREEN_SCRIPTS_FOLDER / "check_labelled_problem_is_gone_in_rewrite_prompt.txt"
DEALT_VALUES_KEPT_PROMPT_PATH = common.SCREEN_SCRIPTS_FOLDER / "check_dealt_values_kept_in_rewrite_prompt.txt"


def build_problem_gone_schema() -> dict:
    return {
        "type": "object",
        "properties": {"reasoning": {"type": "string"}, "problem_gone": {"type": "boolean"}, "note": {"type": "string"}},
        "required": ["reasoning", "problem_gone", "note"],
        "additionalProperties": False,
    }


def build_dealt_values_kept_schema(variable_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "all_dealt_values_present": {"type": "boolean"},
            "missing_variables": {"type": "array", "items": {"type": "string", "enum": list(variable_names)}},
        },
        "required": ["reasoning", "all_dealt_values_present", "missing_variables"],
        "additionalProperties": False,
    }


def validate_problem_gone(parsed: dict) -> None:
    if not parsed["note"].strip():
        raise ValueError("no note")


def validate_dealt_values_kept(parsed: dict) -> None:
    if parsed["all_dealt_values_present"] and parsed["missing_variables"]:
        raise ValueError("all present but missing variables listed")
    if not parsed["all_dealt_values_present"] and not parsed["missing_variables"]:
        raise ValueError("not all present but no missing variable named")


def render_problem_gone_prompt(original_description: str, problem_type: str, problem_reason: str,
                               rewritten_description: str) -> tuple[str, str]:
    system_text, user_template = common.read_prompt_sections(PROBLEM_GONE_PROMPT_PATH)
    return system_text, user_template.format(
        original_description=original_description.strip(), problem_type=problem_type,
        problem_reason=problem_reason.strip(), rewritten_description=rewritten_description.strip())


def render_dealt_values_kept_prompt(variables: dict, rewritten_description: str) -> tuple[str, str]:
    system_text, user_template = common.read_prompt_sections(DEALT_VALUES_KEPT_PROMPT_PATH)
    return system_text, user_template.format(
        dealt_variables=common.format_dealt_variables(variables), rewritten_description=rewritten_description.strip())
