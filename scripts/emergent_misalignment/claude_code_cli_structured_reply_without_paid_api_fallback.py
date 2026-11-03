"""One structured-output model call through the Claude Code CLI (claude_agent_sdk), subscription only.

Adapted from `_run_claude_code_query` in
animal-welfare-data-pipeline-emergent-misalignment-prompts/shared/api.py (see references.md).
Differences from the original: no `auto` or `api` backend exists here at all, so there is
nothing to fall back to; no global state; no tenacity; the paid-API environment variables
are blanked and a refusal to run is raised if the SDK reports an API-key billing source.
Sampling parameters are never set.
"""

from dataclasses import dataclass


class ClaudeCodeUsageLimitReached(Exception):
    """Subscription window exhausted. Not retried: stop and resume later."""


class ClaudeCodeCallFailed(Exception):
    """Any other failure of the CLI call."""


@dataclass
class ClaudeCodeReply:
    reply_text: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    reported_cost_usd: float | None


NEUTRAL_SYSTEM_PROMPT = "You are a careful evaluator. Follow the instructions in the user message exactly."
USAGE_LIMIT_MARKERS = ("usage limit", "rate limit", "limit reached", "limit will reset")


def call_claude_code_cli_for_structured_reply(model: str, user_message: str, json_schema: dict) -> ClaudeCodeReply:
    import json
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, query
    from claude_agent_sdk.types import ResultMessage

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=NEUTRAL_SYSTEM_PROMPT,
        tools=[],  # pure text, no file, shell or web access
        max_turns=1,
        setting_sources=[],  # hermetic: ignore the user's own ~/.claude settings and hooks
        permission_mode="default",
        # Empty values make the CLI treat these as unset, so it uses its subscription login
        # and can never bill an API key found in the environment.
        env={"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""},
        thinking={"type": "disabled"},
        output_format={"type": "json_schema", "schema": json_schema},
    )

    async def run_query():
        result_message = None
        stream = query(prompt=user_message, options=options)
        try:
            async for message in stream:
                if isinstance(message, ResultMessage):
                    result_message = message
                    break
        finally:
            await stream.aclose()
        return result_message

    try:
        result_message = anyio.run(run_query)
    except Exception as error:
        raise _classify(str(error)) from error
    if result_message is None:
        raise ClaudeCodeCallFailed("the CLI returned no result message")
    if result_message.is_error:
        raise _classify(result_message.result or result_message.subtype or "unknown CLI error")
    if result_message.structured_output is not None:
        reply_text = json.dumps(result_message.structured_output, ensure_ascii=False)
    else:
        reply_text = result_message.result or ""  # caller's validation will reject it
    usage = result_message.usage or {}
    return ClaudeCodeReply(
        reply_text=reply_text,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        reported_cost_usd=result_message.total_cost_usd,
    )


def _classify(message: str) -> Exception:
    lowered = message.lower()
    if any(marker in lowered for marker in USAGE_LIMIT_MARKERS):
        return ClaudeCodeUsageLimitReached(message)
    return ClaudeCodeCallFailed(message)
