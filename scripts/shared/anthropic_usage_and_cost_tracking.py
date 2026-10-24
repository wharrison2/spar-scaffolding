"""Token usage, cost and cache bookkeeping for Anthropic Messages API calls.

Separate from api_usage_and_cost_tracking.py (which is OpenAI-oriented) so that module is untouched.
Prices are USD per million tokens. Cache reads cost 10 percent of the input price; cache writes are
assumed to cost 125 percent of the input price (the five-minute-cache premium documented by Anthropic).
A model that is not in the price table has its cost recorded as unknown (None) instead of crashing.

With server-side refusal fallback, `response.usage.iterations` lists every attempt with its own model and
token counts and is the per-attempt source of truth (top-level usage covers only the attempt that produced
the returned message), so cost is computed per attempt when iterations are present.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL = {
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},  # possible refusal-fallback model
}
CACHE_READ_PRICE_FRACTION_OF_INPUT = 0.10
CACHE_WRITE_PRICE_FRACTION_OF_INPUT = 1.25

ATTEMPT_TOKEN_FIELD_NAMES = ["input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"]


def compute_anthropic_cost_in_dollars(model_name, input_tokens, output_tokens, cache_read_tokens=0,
                                      cache_write_tokens=0):
    """Dollar cost of one attempt, or None when the model has no known price. `input_tokens` is the
    uncached remainder, as the Anthropic API reports it."""
    prices = PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL.get(model_name)
    if prices is None:
        return None
    return (input_tokens * prices["input"]
            + cache_read_tokens * prices["input"] * CACHE_READ_PRICE_FRACTION_OF_INPUT
            + cache_write_tokens * prices["input"] * CACHE_WRITE_PRICE_FRACTION_OF_INPUT
            + output_tokens * prices["output"]) / 1_000_000


def _attempt_from_usage_like_object(model_name, usage_like):
    return {
        "model": model_name,
        "input_tokens": getattr(usage_like, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage_like, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage_like, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(usage_like, "cache_creation_input_tokens", 0) or 0,
    }


def extract_attempts_from_response(response):
    """List of per-attempt usage dicts. Uses `usage.iterations` (message and fallback_message entries;
    compaction and advisor entries are skipped) when present, otherwise the top-level usage."""
    usage = response.usage
    iterations = getattr(usage, "iterations", None)
    if iterations:
        attempts = [_attempt_from_usage_like_object(getattr(entry, "model", None) or response.model, entry)
                    for entry in iterations if getattr(entry, "type", None) in ("message", "fallback_message")]
        if attempts:
            return attempts
    return [_attempt_from_usage_like_object(response.model, usage)]


def summarize_attempts(attempts):
    """Per-call usage summary: summed tokens, summed cost (None if any attempt's model is unpriced)."""
    summary = {name: sum(attempt[name] for attempt in attempts) for name in ATTEMPT_TOKEN_FIELD_NAMES}
    costs = [compute_anthropic_cost_in_dollars(
        attempt["model"], attempt["input_tokens"], attempt["output_tokens"], attempt["cache_read_tokens"],
        attempt["cache_write_tokens"]) for attempt in attempts]
    summary["cost_dollars"] = None if any(cost is None for cost in costs) else sum(costs)
    summary["unpriced_models"] = sorted({a["model"] for a, c in zip(attempts, costs) if c is None})
    summary["attempts"] = attempts
    return summary


def empty_totals():
    return {"calls": 0, **{name: 0 for name in ATTEMPT_TOKEN_FIELD_NAMES}, "cache_rate": 0.0,
            "cost_dollars_of_priced_attempts": 0.0, "calls_with_unknown_cost": 0, "unpriced_models": []}


class AnthropicUsageAndCostTracker:
    """Thread-safe totals per group key and overall."""

    def __init__(self):
        self.lock = threading.Lock()
        self.totals_per_group = {}
        self.overall_totals = empty_totals()

    @staticmethod
    def _add(totals, call_summary):
        totals["calls"] += 1
        for name in ATTEMPT_TOKEN_FIELD_NAMES:
            totals[name] += call_summary[name]
        all_input = totals["input_tokens"] + totals["cache_read_tokens"] + totals["cache_write_tokens"]
        totals["cache_rate"] = totals["cache_read_tokens"] / all_input if all_input else 0.0
        if call_summary["cost_dollars"] is None:
            totals["calls_with_unknown_cost"] += 1
            totals["unpriced_models"] = sorted(set(totals["unpriced_models"]) | set(call_summary["unpriced_models"]))
        else:
            totals["cost_dollars_of_priced_attempts"] += call_summary["cost_dollars"]

    def add_call_summary(self, group_key, call_summary):
        with self.lock:
            self._add(self.totals_per_group.setdefault(group_key, empty_totals()), call_summary)
            self._add(self.overall_totals, call_summary)

    def write_log_file(self, log_path, note):
        with self.lock:
            Path(log_path).write_text(json.dumps({
                "updated_at": datetime.now().isoformat(timespec="seconds"), "note": note,
                "prices_per_million_tokens": PRICES_PER_MILLION_TOKENS_BY_ANTHROPIC_MODEL,
                "per_group": self.totals_per_group, "overall": self.overall_totals}, indent=2), encoding="utf-8")
