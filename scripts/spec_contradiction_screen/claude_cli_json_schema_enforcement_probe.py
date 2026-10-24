"""One Claude CLI call (subscription, via claude_agent_sdk) that pressures the model to
break a JSON schema, to see whether --json-schema output is constrained or only checked.

The user message asks for an answer the schema forbids (number 7 when the enum is 1-3,
plain prose instead of JSON). Options mirror the pipeline's claude_code backend
(shared/api.py:_run_claude_code_query) plus output_format.

Run with the pipeline's virtual environment:
    animal-welfare-data-pipeline/.venv/bin/python scripts/spec_contradiction_screen/claude_cli_json_schema_enforcement_probe.py
"""

import json
from datetime import datetime
from pathlib import Path

import anyio
from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_PROMPT_PATH = Path(__file__).with_name("claude_cli_json_schema_enforcement_probe_prompt.txt")
MODEL_NAME = "claude-sonnet-5"

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "specifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer", "enum": [1, 2, 3]},
                    "reasoning": {"type": "string"},
                    "remove": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["number", "reasoning", "remove", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["specifications"],
    "additionalProperties": False,
}


def run_probe() -> dict:
    options = ClaudeAgentOptions(
        model=MODEL_NAME,
        system_prompt="You are a careful editor.",
        tools=[],
        max_turns=1,
        setting_sources=[],
        permission_mode="default",
        thinking={"type": "disabled"},
        env={"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""},
        output_format={"type": "json_schema", "schema": ANSWER_SCHEMA},
    )
    user_message = PROBE_PROMPT_PATH.read_text(encoding="utf-8")

    async def collect() -> dict:
        assistant_text_parts, result_message = [], None
        stream = query(prompt=user_message, options=options)
        try:
            async for message in stream:
                if isinstance(message, AssistantMessage):
                    assistant_text_parts += [b.text for b in message.content if isinstance(b, TextBlock)]
                elif isinstance(message, ResultMessage):
                    result_message = message
                    break
        finally:
            await stream.aclose()
        return {
            "assistant_text": "\n".join(assistant_text_parts),
            "result_subtype": getattr(result_message, "subtype", None),
            "result_is_error": getattr(result_message, "is_error", None),
            "result_text": getattr(result_message, "result", None),
            "structured_output": getattr(result_message, "structured_output", None),
            "num_turns": getattr(result_message, "num_turns", None),
            "usage": getattr(result_message, "usage", None),
            "notional_cost_usd": getattr(result_message, "total_cost_usd", None),
            "errors": getattr(result_message, "errors", None),
        }

    return anyio.run(collect)


def main() -> None:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = (PROJECT_ROOT / "data" / "spec_contradiction_screen_runs" / "small_sample_checks"
                     / f"claude_cli_json_schema_enforcement_probe_{stamp}")
    output_folder.mkdir(parents=True, exist_ok=False)
    started = datetime.now()
    outcome = run_probe()
    outcome["duration_seconds"] = (datetime.now() - started).total_seconds()
    outcome["schema_sent"] = ANSWER_SCHEMA
    outcome["model"] = MODEL_NAME
    (output_folder / f"probe_outcome_{stamp}.json").write_text(
        json.dumps(outcome, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(outcome, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
