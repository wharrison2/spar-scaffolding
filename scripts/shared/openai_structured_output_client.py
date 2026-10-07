"""OpenAI Responses API client for structured (JSON schema) outputs, with retries and parse-retry.

The API key is read from the project's .env file only: never from the OS environment, never from ~/.openai.
"""

import json
import random
import time
from pathlib import Path

from shared.api_usage_and_cost_tracking import extract_usage_from_response, sum_usages

ENV_FILE_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

SAMPLING_TEMPERATURE = 1  # top_p is deliberately not passed (the model rejects it; default is 1)
MAXIMUM_TRANSPORT_ATTEMPTS = 7
FIRST_BACKOFF_SECONDS = 2.0
MAXIMUM_BACKOFF_SECONDS = 60.0


def load_api_key(env_path=None):
    """Read OPENAI_API_KEY from the project's .env file only (never from the OS environment)."""
    env_path = Path(env_path) if env_path is not None else ENV_FILE_PATH
    if not env_path.is_file():
        raise RuntimeError(f"Project .env file not found at {env_path}; it must define OPENAI_API_KEY.")
    key = None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        if name != "OPENAI_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        key = value.strip()
    if not key:
        raise RuntimeError(f"OPENAI_API_KEY is missing or empty in {env_path}.")
    return key


def make_client(env_path=None):
    import openai
    env_path = Path(env_path) if env_path is not None else ENV_FILE_PATH
    api_key = load_api_key(env_path)
    print(f"Using OPENAI_API_KEY from {env_path}")
    # Accept-Encoding gzip works around a decompression bug in the installed httpx.
    # max_retries=0 because we do our own retries (so every attempt is visible and logged).
    return openai.OpenAI(api_key=api_key, default_headers={"Accept-Encoding": "gzip"},
                         max_retries=0, timeout=600)


def is_transient_error(error):
    import openai
    if isinstance(error, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return isinstance(error, openai.APIStatusError) and error.status_code >= 500


def call_with_transport_retries(client, request_arguments, sleep_function=time.sleep):
    for attempt_number in range(1, MAXIMUM_TRANSPORT_ATTEMPTS + 1):
        try:
            return client.responses.create(**request_arguments)
        except Exception as error:  # noqa: BLE001
            if not is_transient_error(error) or attempt_number == MAXIMUM_TRANSPORT_ATTEMPTS:
                raise
            backoff = min(MAXIMUM_BACKOFF_SECONDS, FIRST_BACKOFF_SECONDS * 2 ** (attempt_number - 1))
            sleep_function(backoff * random.uniform(0.5, 1.0))


def build_structured_request_arguments(model_name, instructions_text, input_text, schema_name, schema,
                                       reasoning_effort, prompt_cache_key=None):
    """The complete payload for client.responses.create. Only instructions and input text go in."""
    request_arguments = {
        "model": model_name,
        "instructions": instructions_text,
        "input": input_text,
        "temperature": SAMPLING_TEMPERATURE,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    }
    if prompt_cache_key is not None:
        request_arguments["prompt_cache_key"] = prompt_cache_key
    return request_arguments


class StructuredOutputParseFailure(Exception):
    """Both parse attempts failed; carries what the error log needs."""

    def __init__(self, description, raw_output_text, usage, latency_seconds):
        super().__init__(description)
        self.description = description
        self.raw_output_text = raw_output_text
        self.usage = usage
        self.latency_seconds = latency_seconds


def request_structured_output(client, model_name, instructions_text, input_text, schema_name, schema,
                              reasoning_effort, validate_parsed_result=None, prompt_cache_key=None):
    """Call the model, parse the JSON, validate it, and retry once on a parse/validation failure.

    Returns a dict with parsed_result, raw_output_text, usage (summed over attempts), parse_attempts and
    latency_seconds. Raises StructuredOutputParseFailure if both attempts fail; transport errors propagate.
    """
    attempt_usages = []
    last_failure_description = None
    last_raw_output_text = None
    started = time.time()
    for parse_attempt_number in (1, 2):
        request_arguments = build_structured_request_arguments(
            model_name, instructions_text, input_text, schema_name, schema, reasoning_effort, prompt_cache_key)
        response = call_with_transport_retries(client, request_arguments)
        attempt_usages.append(extract_usage_from_response(response))
        raw_output_text = response.output_text
        last_raw_output_text = raw_output_text
        try:
            if getattr(response, "status", "completed") != "completed":
                raise ValueError(f"response status {response.status}")
            parsed_result = json.loads(raw_output_text)
            if validate_parsed_result is not None:
                validate_parsed_result(parsed_result)
        except (ValueError, TypeError) as failure:
            last_failure_description = f"attempt {parse_attempt_number}: {failure}"
            continue
        return {"parsed_result": parsed_result, "raw_output_text": raw_output_text,
                "usage": sum_usages(attempt_usages), "parse_attempts": parse_attempt_number,
                "latency_seconds": round(time.time() - started, 3)}
    raise StructuredOutputParseFailure(last_failure_description, last_raw_output_text,
                                       sum_usages(attempt_usages), round(time.time() - started, 3))
