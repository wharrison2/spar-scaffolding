"""Token usage, cost and cache-rate bookkeeping for AI API calls, grouped by an arbitrary key."""

import json
import threading
from datetime import datetime
from pathlib import Path

# Prices in USD per million tokens. NOTE: sources disagree on the cached-input price;
# these are the values we decided to use. Reasoning tokens are billed as output tokens.
PRICES_PER_MILLION_TOKENS_BY_MODEL = {
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    # Standard tier, from developers.openai.com/api/docs/pricing on 2026-10-21.
    "gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
}

USAGE_FIELD_NAMES = ["input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"]


def compute_cost_in_dollars(model_name, input_tokens, cached_input_tokens, output_tokens):
    prices = PRICES_PER_MILLION_TOKENS_BY_MODEL[model_name]
    uncached_input_tokens = input_tokens - cached_input_tokens
    return (
        uncached_input_tokens * prices["input"]
        + cached_input_tokens * prices["cached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000


def empty_usage():
    return {field_name: 0 for field_name in USAGE_FIELD_NAMES}


def sum_usages(usages):
    return {field_name: sum(usage[field_name] for usage in usages) for field_name in USAGE_FIELD_NAMES}


def extract_usage_from_response(response):
    """Pull token counts out of an OpenAI Responses API response."""
    usage = response.usage
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": getattr(input_details, "cached_tokens", 0) or 0,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", 0) or 0,
    }


def empty_totals():
    return {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "cache_rate": 0.0, "cost_dollars": 0.0}


def add_usage_to_totals(model_name, totals, usage):
    totals["calls"] += 1
    for field_name in USAGE_FIELD_NAMES:
        totals[field_name] += usage[field_name]
    totals["cache_rate"] = (totals["cached_input_tokens"] / totals["input_tokens"]
                            if totals["input_tokens"] else 0.0)
    totals["cost_dollars"] = compute_cost_in_dollars(
        model_name, totals["input_tokens"], totals["cached_input_tokens"], totals["output_tokens"])


class UsageAndCostTracker:
    """Thread-safe accumulation of usage per group key (for example per pass) and overall."""

    def __init__(self, model_name, group_label="per_group"):
        self.model_name = model_name
        self.group_label = group_label
        self.lock = threading.Lock()
        self.totals_per_group = {}
        self.overall_totals = empty_totals()

    def add_usage(self, group_key, usage):
        with self.lock:
            add_usage_to_totals(self.model_name, self.totals_per_group.setdefault(group_key, empty_totals()), usage)
            add_usage_to_totals(self.model_name, self.overall_totals, usage)

    def write_log_file(self, log_path, note):
        with self.lock:
            usage_log = {"updated_at": datetime.now().isoformat(timespec="seconds"),
                         "note": note,
                         self.group_label: self.totals_per_group, "overall": self.overall_totals}
            Path(log_path).write_text(json.dumps(usage_log, indent=2), encoding="utf-8")
