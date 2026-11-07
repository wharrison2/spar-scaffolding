"""Offline tests: no model call. Stubs `claude_agent_sdk.query`. Run: python -m pytest this_file"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_code_cli_structured_reply_without_paid_api_fallback as wrapper

pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk.types import ResultMessage, SystemMessage  # noqa: E402

INIT_DATA = {"type": "system", "subtype": "init", "tools": ["StructuredOutput", "mcp__claude_ai_Slack__slack_send_message",
                                                             "mcp__claude_ai_Gmail__authenticate"],
             "mcp_servers": [{"name": "claude_ai_Slack", "status": "connected"}, {"name": "claude_ai_Gmail", "status": "needs-auth"}]}


def result_message():
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=2,
                         session_id="sesn_test", result="x", structured_output={"a": 1}, total_cost_usd=0.01,
                         usage={"input_tokens": 5, "output_tokens": 6, "cache_read_input_tokens": 7,
                                "cache_creation_input_tokens": 8})


def test_options_shut_out_every_mcp_server():
    options = wrapper.build_judge_options("claude-haiku-4-5", {"type": "object"})
    assert options.mcp_servers == {}
    assert options.strict_mcp_config is True
    assert options.env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    assert options.env["ANTHROPIC_API_KEY"] == ""


def test_old_options_for_the_diagnostic_do_not_shut_out_servers():
    options = wrapper.build_judge_options("claude-haiku-4-5", {"type": "object"}, shut_out_claude_ai_connectors=False)
    assert not options.mcp_servers and options.strict_mcp_config is False
    assert "ENABLE_CLAUDEAI_MCP_SERVERS" not in options.env


def test_empty_servers_with_strict_mode_reach_the_command_line_as_the_strict_flag_only():
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
    options = wrapper.build_judge_options("claude-haiku-4-5", {"type": "object"})
    transport = SubprocessCLITransport(prompt="hi", options=options)
    transport._cli_path = "claude"  # building the command never runs it
    command = transport._build_command()
    assert "--strict-mcp-config" in command
    assert "--mcp-config" not in command


def test_init_message_fields_are_parsed_from_a_stubbed_stream(monkeypatch):
    async def fake_query(*, prompt, options):
        yield SystemMessage(subtype="init", data=INIT_DATA)
        yield result_message()

    monkeypatch.setattr("claude_agent_sdk.query", fake_query)
    reply = wrapper.call_claude_code_cli_for_structured_reply("claude-haiku-4-5", "hi", {"type": "object"})
    assert reply.number_of_tools_loaded == 3
    assert reply.names_of_non_built_in_tools == ("mcp__claude_ai_Slack__slack_send_message", "mcp__claude_ai_Gmail__authenticate")
    assert reply.number_of_mcp_servers_loaded == 2
    assert reply.number_of_turns == 2
    assert (reply.input_tokens, reply.cache_read_input_tokens, reply.cache_creation_input_tokens) == (5, 7, 8)


def test_missing_init_message_leaves_the_fields_empty(monkeypatch):
    async def fake_query(*, prompt, options):
        yield result_message()

    monkeypatch.setattr("claude_agent_sdk.query", fake_query)
    reply = wrapper.call_claude_code_cli_for_structured_reply("claude-haiku-4-5", "hi", {"type": "object"})
    assert reply.number_of_tools_loaded is None and reply.number_of_mcp_servers_loaded is None
    assert reply.names_of_non_built_in_tools == ()
