"""Offline tests (a fake Anthropic client; no network, no key, no paid call) for the paid-API backend of the Haiku judges:
    python3 -m pytest scripts/emergent_misalignment/test_anthropic_api_structured_reply_for_haiku_judges_without_any_api_call.py
"""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

THIS_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_FOLDER))
sys.path.insert(0, str(THIS_FOLDER.parent.parent))

import anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log as api  # noqa: E402
import claude_code_cli_structured_reply_without_paid_api_fallback as cli_wrapper  # noqa: E402
import judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli as welfare_judge  # noqa: E402
import judge_whether_answer_chose_option_better_for_animal_welfare_without_mentioning_welfare_through_claude_code_cli as choice_judge  # noqa: E402

ANSWER = {"answer_id": "a1", "model_short_name": "m", "condition": "c", "prompt_gid": "P1", "sample_index": 0,
          "user_message": "Should I move the barn? {final_answer_text}", "final_answer_text": "Yes, move it. {user_message}", "thinking_trace_text": None}


class FakeStatusError(Exception):
    def __init__(self, status_code, message="error", retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers={"retry-after": str(retry_after)} if retry_after is not None else {})


def fake_response(text='{"x": 1}', input_tokens=1000, output_tokens=20):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage, stop_reason="end_turn")


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else fake_response()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_backend(outcomes, tmp_path, **options):
    sleeps = []
    backend = api.AnthropicApiJudgeBackend(client=FakeClient(outcomes), usage_log_path=tmp_path / "usage.jsonl", sleep=sleeps.append,
                                           jitter=lambda: 0.5, log=lambda message: None, **options)
    return backend, sleeps


def test_request_is_the_cli_judges_own_prompt_system_prompt_and_schema_with_no_sampling_parameters(tmp_path):
    backend, _ = make_backend([fake_response(json.dumps({"refusal": "complied", "raised_welfare_cost": "no",
                                                         "welfare_cost_changed_the_advice": "not_applicable"}))], tmp_path)
    prompt, schema = welfare_judge.render_judge_prompt(ANSWER), welfare_judge.build_json_schema(False)
    reply = backend.call_model(welfare_judge.JUDGE_MODEL, prompt, schema)
    request = backend.client.requests[0]
    assert request["model"] == "claude-haiku-5-5" and request["system"] == cli_wrapper.NEUTRAL_SYSTEM_PROMPT
    assert request["messages"] == [{"role": "user", "content": prompt}]  # the user message is the CLI judge's rendered prompt, byte for byte
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": schema}
    assert request["thinking"] == {"type": "disabled"}
    assert not {"temperature", "top_p", "top_k"} & set(request)
    assert welfare_judge.check_reply(reply.reply_text, ANSWER)[1] is None  # the CLI judge's own reply check accepts it


def test_choice_judge_prompt_and_schema_are_the_cli_judges_own(tmp_path):
    backend, _ = make_backend([], tmp_path)
    backend.call_model("claude-haiku-5-5", choice_judge.render_judge_prompt(ANSWER), choice_judge.build_json_schema())
    request = backend.client.requests[0]
    assert request["messages"][0]["content"] == choice_judge.render_judge_prompt(ANSWER)
    assert request["output_config"]["format"]["schema"] == choice_judge.build_json_schema()


def test_judge_one_answer_of_the_cli_scripts_runs_unchanged_on_the_backend(tmp_path):
    reply_text = json.dumps({"refusal": "complied", "raised_welfare_cost": "yes", "welfare_cost_changed_the_advice": "mentioned_only"})
    backend, _ = make_backend([fake_response(reply_text)], tmp_path)
    record, usage = welfare_judge.judge_one_answer(ANSWER, backend.call_model)
    assert record["parse_status"] == "ok" and record["raised_welfare_cost"] == "yes" and record["judge_model"] == "claude-haiku-5-5"
    assert usage[0]["cli_reported_cost_usd"] == pytest.approx((1000 * 0.10 + 20 * 0.50) / 1e6)


def test_cost_tokens_and_thinking_mode_are_logged_per_call_and_totalled(tmp_path):
    backend, _ = make_backend([fake_response(input_tokens=2000, output_tokens=100), fake_response(input_tokens=500, output_tokens=10)], tmp_path)
    backend.call_model("claude-haiku-5-5", "a", {})
    backend.call_model("claude-haiku-5-5", "b", {})
    records = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [record["input_tokens"] for record in records] == [2000, 500] and records[0]["thinking_mode"] == api.THINKING_MODE_DISABLED
    assert records[0]["cost_dollars"] == pytest.approx((2000 * 0.10 + 100 * 0.50) / 1e6)
    assert backend.totals["calls"] == 2 and backend.totals["cost_dollars"] == pytest.approx((2500 * 0.10 + 110 * 0.50) / 1e6)
    assert "2 calls" in backend.progress_line() and "per minute" in backend.progress_line() and "cost so far" in backend.progress_line()


def test_a_400_about_thinking_switches_every_later_call_to_adaptive_thinking_at_effort_low_and_records_it(tmp_path):
    backend, _ = make_backend([FakeStatusError(400, "thinking: disabled is not supported for this model")], tmp_path)
    reply = backend.call_model("claude-haiku-5-5", "p", {"type": "object"})
    first, second = backend.client.requests
    assert first["thinking"] == {"type": "disabled"} and second["thinking"] == {"type": "adaptive"} and second["output_config"]["effort"] == "low"
    assert reply.thinking_mode == api.THINKING_MODE_ADAPTIVE_LOW_EFFORT and second["max_tokens"] > first["max_tokens"]
    backend.call_model("claude-haiku-5-5", "q", {})
    assert backend.client.requests[2]["thinking"] == {"type": "adaptive"}
    assert json.loads((tmp_path / "usage.jsonl").read_text().splitlines()[-1])["thinking_mode"] == api.THINKING_MODE_ADAPTIVE_LOW_EFFORT


def test_429_waits_for_the_retry_after_header_then_succeeds(tmp_path):
    backend, sleeps = make_backend([FakeStatusError(429, "rate limited", retry_after=7)], tmp_path)
    backend.call_model("claude-haiku-5-5", "p", {})
    assert sleeps == [7.0] and backend.totals["rate_limited_responses"] == 1


def test_5xx_and_connection_errors_are_retried_with_backoff_and_other_errors_are_not(tmp_path):
    import anthropic
    connection_error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    backend, sleeps = make_backend([FakeStatusError(529), FakeStatusError(500), connection_error], tmp_path)
    backend.call_model("claude-haiku-5-5", "p", {})
    assert len(sleeps) == 3 and sleeps[0] < sleeps[1] < sleeps[2] and backend.totals["transient_errors"] == 3
    backend, _ = make_backend([FakeStatusError(422, "invalid schema")], tmp_path)
    with pytest.raises(api.AnthropicApiCallFailed):
        backend.call_model("claude-haiku-5-5", "p", {})
    assert len(backend.client.requests) == 1


def test_retries_are_bounded(tmp_path):
    backend, sleeps = make_backend([FakeStatusError(500)] * 10, tmp_path, maximum_attempts_per_call=3)
    with pytest.raises(api.AnthropicApiCallFailed, match="gave up after 3 tries"):
        backend.call_model("claude-haiku-5-5", "p", {})
    assert len(backend.client.requests) == 3


def test_a_rejected_key_is_fatal_and_stops_every_later_call(tmp_path):
    backend, _ = make_backend([FakeStatusError(401, "invalid x-api-key")], tmp_path)
    with pytest.raises(api.AnthropicApiFatalError):
        backend.call_model("claude-haiku-5-5", "p", {})
    with pytest.raises(api.AnthropicApiFatalError):
        backend.call_model("claude-haiku-5-5", "q", {})
    assert len(backend.client.requests) == 1


def test_repeated_429s_halve_the_concurrency_and_a_hundred_calls_in_a_row_raise_it_by_one(tmp_path):
    backend, _ = make_backend([FakeStatusError(429, "slow", retry_after=1)] * 3, tmp_path, initial_concurrent_calls=16)
    backend.call_model("claude-haiku-5-5", "p", {})
    assert backend.gate.limit == 8
    for _ in range(api.CALLS_WITHOUT_A_RATE_LIMIT_THAT_RAISE_CONCURRENCY_BY_ONE):
        backend.call_model("claude-haiku-5-5", "p", {})
    assert backend.gate.limit == 9
    for _ in range(10):
        backend.gate.note_rate_limited()
    assert backend.gate.limit >= api.MINIMUM_CONCURRENT_CALLS


def test_concurrency_never_exceeds_the_gate(tmp_path):
    in_flight, peak, lock = [0], [0], threading.Lock()

    class SlowClient(FakeClient):
        def create(self, **request):
            with lock:
                in_flight[0] += 1
                peak[0] = max(peak[0], in_flight[0])
            threading.Event().wait(0.02)
            with lock:
                in_flight[0] -= 1
            return fake_response()

    backend = api.AnthropicApiJudgeBackend(client=SlowClient([]), initial_concurrent_calls=3, log=lambda message: None)
    threads = [threading.Thread(target=backend.call_model, args=("claude-haiku-5-5", "p", {})) for _ in range(12)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert peak[0] <= 3 and backend.totals["calls"] == 12


def test_calls_per_minute_counts_the_last_minute_only(tmp_path):
    now = [1000.0]
    backend = api.AnthropicApiJudgeBackend(client=FakeClient([]), clock=lambda: now[0], log=lambda message: None)
    for _ in range(30):
        backend.call_model("claude-haiku-5-5", "p", {})
    assert backend.calls_per_minute() == 30
    now[0] += 120
    assert backend.calls_per_minute() == 0


def test_the_key_is_read_from_the_env_file_only_and_the_run_is_refused_without_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-the-operating-system")
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not in"):
        api.read_anthropic_api_key_from_project_env_file(tmp_path / "no_such.env")
    empty = tmp_path / "empty.env"
    empty.write_text("RUNPOD_API_KEY=x\n")
    with pytest.raises(SystemExit):
        api.read_anthropic_api_key_from_project_env_file(empty)  # the OS key is never used as a fallback
    with_key = tmp_path / "with_key.env"
    with_key.write_text("ANTHROPIC_API_KEY=sk-from-the-project-env-file\n")
    assert api.read_anthropic_api_key_from_project_env_file(with_key) == "sk-from-the-project-env-file"


def test_the_sdk_accepts_the_request_shape_and_haiku_5_5_is_priced():
    import anthropic
    assert anthropic.types.ThinkingConfigDisabledParam and anthropic.types.OutputConfigParam
    backend, _ = make_backend([], Path("/tmp"))
    request = backend.request_arguments("claude-haiku-5-5", "p", {"type": "object"})
    assert set(request) == {"model", "max_tokens", "system", "messages", "thinking", "output_config"}
    from scripts.shared.anthropic_usage_and_cost_tracking import compute_anthropic_cost_in_dollars
    assert compute_anthropic_cost_in_dollars("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
