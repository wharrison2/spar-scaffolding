"""Schemas, prompt rendering and answer validation for GPT 6.1 Sol's hand judgments and replacement choices.

Pure functions (no model calls, no `shared` imports) so the cost estimate can render the exact prompts.
"""

import spec_screen_common as common

HAND_JUDGMENT_PROMPT_PATH = common.SCREEN_SCRIPTS_FOLDER / "judge_one_hand_extremely_implausible_prompt.txt"
REPLACEMENT_PROMPT_PATH = common.SCREEN_SCRIPTS_FOLDER / "propose_replacement_value_prompt.txt"
PLANT_TYPE_INSTRUCTIONS_PATH = common.SCREEN_SCRIPTS_FOLDER / "planted_hand_type_instructions.txt"


def build_hand_judgment_schema() -> dict:
    """reasoning first (generation order), then the decision."""
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "extremely_implausible": {"type": "boolean"},
            "problem_type": {"type": "string", "enum": common.GOLD_PROBLEM_TYPE_NAMES},
            "reason": {"type": "string"},
        },
        "required": ["reasoning", "extremely_implausible", "problem_type", "reason"],
        "additionalProperties": False,
    }


def validate_hand_judgment(parsed: dict) -> None:
    """Raise ValueError unless the judgment is internally consistent."""
    if parsed["problem_type"] not in common.GOLD_PROBLEM_TYPE_NAMES:
        raise ValueError(f"unknown problem_type {parsed['problem_type']!r}")
    if parsed["extremely_implausible"]:
        if parsed["problem_type"] == "none":
            raise ValueError("extremely implausible but problem_type is none")
        if not parsed["reason"].strip():
            raise ValueError("extremely implausible without a reason")
    else:
        if parsed["problem_type"] != "none":
            raise ValueError("not extremely implausible but problem_type is not none")
        if parsed["reason"].strip():
            raise ValueError("not extremely implausible but a reason was given")


def render_hand_judgment_prompt(variable_lines: str) -> tuple[str, str]:
    system_text, user_template = common.read_prompt_sections(HAND_JUDGMENT_PROMPT_PATH)
    return system_text, user_template.format(variable_lines=variable_lines.strip())


def build_replacement_choice_schema(option_count: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "chosen_option_number": {"type": "integer", "enum": list(range(1, option_count + 1))},
        },
        "required": ["reasoning", "chosen_option_number"],
        "additionalProperties": False,
    }


def validate_replacement_choice(parsed: dict, option_count: int) -> None:
    number = parsed["chosen_option_number"]
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= option_count:
        raise ValueError(f"option number {number!r} is outside 1..{option_count}")


def render_replacement_prompt(variable_lines: str, target_variable: str, current_value: str, plant_type: str,
                              option_texts: list[str]) -> tuple[str, str]:
    """The options are numbered 1..N in the order given; the model answers with a number only."""
    instructions = common.read_labelled_blocks(PLANT_TYPE_INSTRUCTIONS_PATH)
    system_text, user_template = common.read_prompt_sections(REPLACEMENT_PROMPT_PATH)
    numbered = "\n".join(f"{number}. {text}" for number, text in enumerate(option_texts, 1))
    return system_text, user_template.format(
        variable_lines=variable_lines.strip(), target_variable=target_variable, current_value=current_value,
        type_instruction=instructions[plant_type], numbered_options=numbered)
