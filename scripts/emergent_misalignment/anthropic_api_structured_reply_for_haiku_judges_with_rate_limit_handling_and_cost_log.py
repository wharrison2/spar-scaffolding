"""One structured-output model call through the PAID Anthropic API (official `anthropic` SDK), for the Haiku judges, run from the laptop.

This is the API counterpart of claude_code_cli_structured_reply_without_paid_api_fallback.py (which deliberately has no paid path and is left
untouched). `AnthropicApiJudgeBackend.call_model(model, prompt, json_schema)` has the same signature as the CLI wrapper's function and returns an
object with the same fields, so the existing Haiku judge scripts' `judge_one_answer` and `run_judging` functions take it unchanged: they render the
prompt and check the reply exactly as before. The request carries the same neutral system prompt and the same JSON schema as the CLI call.

 - Key: ANTHROPIC_API_KEY is read from the project .env file only (`read_anthropic_api_key_from_project_env_file`); os.environ is never consulted,
   and the run is refused when the key is absent. The key never leaves the laptop (nothing here runs on a pod).
 - Sampling parameters are never sent (Haiku 5.5 rejects them). Thinking: the call first sends thinking={"type": "disabled"}, as the CLI judges
   do. If the API answers 400 and the message names thinking, effort or output_config, every later call uses adaptive thinking at effort "low"
   instead; `thinking_mode` says which, and every usage record carries it. The Anthropic API reports reasoning tokens inside output_tokens, so
   output_tokens is the figure that includes them (and is what is billed).
 - Retries: 429 (waits for the retry-after header when present, else exponential backoff), 5xx, 408, 409 and connection errors, at most
   `maximum_attempts_per_call` tries; 401 and 403 are fatal (the run stops loudly instead of failing every call one by one).
 - Concurrency adapts: `initial_concurrent_calls` (16) calls at once; three 429s within a minute halve the allowed concurrency (minimum 2); a
   hundred calls in a row without a 429 raise it by one back towards the initial value. The rate limit is the account's rate-limit tier, so
   throughput is whatever that tier allows: `progress_line` reports calls per minute so a low limit is visible early.
 - Cost: every call appends a record (tokens, cache tokens, thinking mode, attempts, cost at $0.10 input and $0.50 output per million tokens,
   from scripts/shared/anthropic_usage_and_cost_tracking.py) to the usage log, and totals are kept for the progress line.
"""

import json
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for directory in (REPOSITORY_ROOT, SCRIPT_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from claude_code_cli_structured_reply_without_paid_api_fallback import NEUTRAL_SYSTEM_PROMPT  # noqa: E402
from scripts.shared.anthropic_usage_and_cost_tracking import compute_anthropic_cost_in_dollars  # noqa: E402

ENVIRONMENT_FILE_PATH = REPOSITORY_ROOT / ".env"
ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"
MAXIMUM_TOKENS_THINKING_DISABLED = 1024
MAXIMUM_TOKENS_THINKING_ADAPTIVE = 4096
THINKING_MODE_DISABLED = "thinking_disabled"
THINKING_MODE_ADAPTIVE_LOW_EFFORT = "thinking_adaptive_effort_low"
MINIMUM_CONCURRENT_CALLS = 2
RATE_LIMITS_WITHIN_A_MINUTE_THAT_HALVE_CONCURRENCY = 3
CALLS_WITHOUT_A_RATE_LIMIT_THAT_RAISE_CONCURRENCY_BY_ONE = 100


class AnthropicApiCallFailed(Exception):
    """A call that could not be completed after its retries (or a non-retryable error). The answer stays unjudged and is retried on resume."""


class AnthropicApiFatalError(AnthropicApiCallFailed):
    """Authentication or permission failure: every later call would fail too."""


@dataclass
class AnthropicApiReply:
    """The fields of ClaudeCodeReply that the judge scripts read, plus what the API call itself reports."""
    reply_text: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    reported_cost_usd: float | None
    number_of_tools_loaded: int | None = 0
    names_of_non_built_in_tools: tuple = ()
    number_of_mcp_servers_loaded: int | None = 0
    number_of_turns: int | None = 1
    thinking_mode: str = THINKING_MODE_DISABLED
    stop_reason: str | None = None


def read_anthropic_api_key_from_project_env_file(environment_file_path=ENVIRONMENT_FILE_PATH) -> str:
    """Reads the project .env explicitly; os.environ is never consulted. Refuses (SystemExit) when the key is absent."""
    from dotenv import dotenv_values
    values = dotenv_values(environment_file_path) if Path(environment_file_path).exists() else {}
    key = (values.get(ANTHROPIC_API_KEY_NAME) or "").strip()
    if not key:
        raise SystemExit(f"{ANTHROPIC_API_KEY_NAME} is not in {environment_file_path}. Put this project's own key there (the paid Anthropic API is "
                         f"used from this laptop only); the operating system's environment is never read. Refusing to run.")
    return key


class AdaptiveConcurrencyGate:
    def __init__(self, initial_limit, clock=time.time, log=print):
        self.initial_limit = self.limit = initial_limit
        self.in_flight = 0
        self.condition = threading.Condition()
        self.recent_rate_limit_times = []
        self.calls_since_last_rate_limit = 0
        self.clock, self.log = clock, log

    def acquire(self):
        with self.condition:
            while self.in_flight >= self.limit:
                self.condition.wait(timeout=1.0)
            self.in_flight += 1

    def release(self):
        with self.condition:
            self.in_flight -= 1
            self.condition.notify_all()

    def note_rate_limited(self):
        with self.condition:
            now = self.clock()
            self.calls_since_last_rate_limit = 0
            self.recent_rate_limit_times = [t for t in self.recent_rate_limit_times if now - t < 60] + [now]
            if len(self.recent_rate_limit_times) >= RATE_LIMITS_WITHIN_A_MINUTE_THAT_HALVE_CONCURRENCY and self.limit > MINIMUM_CONCURRENT_CALLS:
                self.limit = max(MINIMUM_CONCURRENT_CALLS, self.limit // 2)
                self.recent_rate_limit_times = []
                self.log(f"Anthropic API rate limit (429) three times within a minute: concurrency lowered to {self.limit} calls at once")

    def note_success(self):
        with self.condition:
            self.calls_since_last_rate_limit += 1
            if self.calls_since_last_rate_limit >= CALLS_WITHOUT_A_RATE_LIMIT_THAT_RAISE_CONCURRENCY_BY_ONE and self.limit < self.initial_limit:
                self.limit += 1
                self.calls_since_last_rate_limit = 0
                self.condition.notify_all()


def retry_after_seconds(error):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_error(error) -> str:
    status = getattr(error, "status_code", None)
    message = str(error).lower()
    if status == 400 and any(word in message for word in ("thinking", "effort", "output_config")):
        return "thinking_unsupported"
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "fatal"
    if status is not None and (status >= 500 or status in (408, 409)):
        return "transient"
    try:
        import anthropic
        if isinstance(error, anthropic.APIConnectionError):  # includes timeouts
            return "transient"
    except ImportError:
        pass
    return "failed"


class AnthropicApiJudgeBackend:
    def __init__(self, api_key=None, client=None, usage_log_path=None, initial_concurrent_calls=16, maximum_attempts_per_call=8,
                 sleep=time.sleep, clock=time.time, log=print, jitter=random.random):
        if client is None:
            import anthropic
            # Retries are this class's own. Accept-Encoding gzip only: with brotli or zstd responses, this laptop's decompression library
            # raised TypeError (output_buffer_limit), which the SDK reports as "Connection error" (first overnight Haiku run, 2026-11-13)
            client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=120.0, default_headers={"Accept-Encoding": "gzip, deflate"})
        self.client, self.sleep, self.clock, self.log, self.jitter = client, sleep, clock, log, jitter
        self.maximum_attempts_per_call = maximum_attempts_per_call
        self.usage_log_path = Path(usage_log_path) if usage_log_path else None
        self.gate = AdaptiveConcurrencyGate(initial_concurrent_calls, clock, log)
        self.thinking_mode = THINKING_MODE_DISABLED
        self.fatal_error = None
        self.lock = threading.Lock()
        self.completed_call_times = deque()
        self.totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                       "cost_dollars": 0.0, "rate_limited_responses": 0, "transient_errors": 0}
        self.started_at = clock()

    def request_arguments(self, model, prompt, json_schema):
        """The exact request: neutral system prompt, the rendered judge prompt as the only user message, the schema; no sampling parameters."""
        output_format = {"type": "json_schema", "schema": json_schema}
        if self.thinking_mode == THINKING_MODE_DISABLED:
            return {"model": model, "max_tokens": MAXIMUM_TOKENS_THINKING_DISABLED, "system": NEUTRAL_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}], "thinking": {"type": "disabled"},
                    "output_config": {"format": output_format}}
        return {"model": model, "max_tokens": MAXIMUM_TOKENS_THINKING_ADAPTIVE, "system": NEUTRAL_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}], "thinking": {"type": "adaptive"},
                "output_config": {"effort": "low", "format": output_format}}

    def call_model(self, model, prompt, json_schema) -> AnthropicApiReply:
        if self.fatal_error:
            raise self.fatal_error
        attempts, started = 0, self.clock()
        while True:
            mode_used = self.thinking_mode
            self.gate.acquire()
            try:
                response = self.client.messages.create(**self.request_arguments(model, prompt, json_schema))
            except Exception as error:  # noqa: BLE001 - classified below
                self.gate.release()
                kind = classify_error(error)
                if kind == "thinking_unsupported" and mode_used == THINKING_MODE_DISABLED:
                    self.thinking_mode = THINKING_MODE_ADAPTIVE_LOW_EFFORT
                    self.log(f"The API refused thinking disabled ({str(error)[:200]}): every call now uses adaptive thinking at effort low")
                    continue
                if kind == "fatal":
                    self.fatal_error = AnthropicApiFatalError(f"Anthropic API refused the key or the request ({str(error)[:300]}). Stopping.")
                    raise self.fatal_error from error
                if kind not in ("rate_limited", "transient"):
                    raise AnthropicApiCallFailed(f"{type(error).__name__}: {str(error)[:300]}") from error
                attempts += 1
                with self.lock:
                    self.totals["rate_limited_responses" if kind == "rate_limited" else "transient_errors"] += 1
                if kind == "rate_limited":
                    self.gate.note_rate_limited()
                if attempts >= self.maximum_attempts_per_call:
                    raise AnthropicApiCallFailed(f"gave up after {attempts} tries ({kind}): {str(error)[:200]}") from error
                wait = retry_after_seconds(error) if kind == "rate_limited" else None
                if wait is None:
                    wait = min(60.0, 2.0 ** attempts) * (0.5 + self.jitter())
                self.sleep(wait)  # outside the concurrency gate, so a waiting call does not hold a slot
                continue
            self.gate.release()
            self.gate.note_success()
            return self.reply_from_response(model, response, mode_used, attempts + 1, self.clock() - started)

    def reply_from_response(self, model, response, thinking_mode, attempts, seconds) -> AnthropicApiReply:
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cost = compute_anthropic_cost_in_dollars(model, input_tokens, output_tokens, cache_read, cache_write)
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        record = {"logged_at": datetime.now().isoformat(timespec="seconds"), "model": model, "thinking_mode": thinking_mode,
                  "input_tokens": input_tokens, "output_tokens_including_any_reasoning": output_tokens, "cache_read_input_tokens": cache_read,
                  "cache_creation_input_tokens": cache_write, "cost_dollars": cost, "tries_for_this_call": attempts,
                  "seconds_for_this_call": round(seconds, 3), "stop_reason": getattr(response, "stop_reason", None)}
        with self.lock:
            self.totals["calls"] += 1
            self.totals["input_tokens"] += input_tokens
            self.totals["output_tokens"] += output_tokens
            self.totals["cache_read_input_tokens"] += cache_read
            self.totals["cache_creation_input_tokens"] += cache_write
            self.totals["cost_dollars"] += cost or 0.0
            self.completed_call_times.append(self.clock())
            if self.usage_log_path:
                self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.usage_log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
        return AnthropicApiReply(reply_text=text, input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=cache_read,
                                 cache_creation_input_tokens=cache_write, reported_cost_usd=cost, thinking_mode=thinking_mode,
                                 stop_reason=getattr(response, "stop_reason", None))

    def calls_per_minute(self, window_seconds=60.0):
        with self.lock:
            now = self.clock()
            while self.completed_call_times and now - self.completed_call_times[0] > window_seconds:
                self.completed_call_times.popleft()
            return len(self.completed_call_times) * 60.0 / window_seconds

    def progress_line(self):
        totals = dict(self.totals)
        minutes = max((self.clock() - self.started_at) / 60.0, 1e-9)
        return (f"Haiku API: {totals['calls']} calls ({totals['calls'] / minutes:.1f} per minute overall, {self.calls_per_minute():.1f} in the last "
                f"minute), concurrency {self.gate.limit} of {self.gate.initial_limit}, rate-limited responses {totals['rate_limited_responses']}, "
                f"transient errors {totals['transient_errors']}, {totals['input_tokens']:,} input and {totals['output_tokens']:,} output tokens, "
                f"cost so far ${totals['cost_dollars']:.4f}, {self.thinking_mode}")
