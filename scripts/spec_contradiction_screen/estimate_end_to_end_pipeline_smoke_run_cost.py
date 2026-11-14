"""Cost estimate for the end-to-end pipeline smoke run (any number of prompts, any Claude model). Makes NO model calls.

It runs the pipeline's real entry point (`sdf_pipeline.run.main`) in this process on the config copy given,
with both model-call seams replaced by recording stubs (`shared.api.call_claude` and
`shared.openai_api.call_openai`), into a throwaway output folder. Every prompt each stage would send is therefore the
real composed prompt: real deal, real screen batches, real plan prompts, real description rewrite and selector prompts,
real draft prompts, and layer 3 and 4 prompts that include the full constitution. Only the model ANSWERS are stubs, and
the stubs write documents of measured length, so the later stages' input sizes (a draft inside the rewrite prompt, and
so on) are realistic too.

Input tokens = characters (system prompt + user message + JSON answer schema) divided by a characters-per-token ratio
measured on earlier real calls. Output tokens come from measured runs where they exist and stated assumptions otherwise
(see the report this script prints). Every Claude stage runs on the global model (default claude-sonnet-5, $2 / $10 per million tokens) and the
description rewrite on gpt-6.1-sol ($2 input / $0.10 cached input / $10 output, no caching assumed). The Claude model
is the global `model` of the config given (or `--model`), priced from CLAUDE_DOLLARS_PER_MILLION_BY_MODEL. For models that
cannot run with thinking disabled (claude-opus-5-5 and the Mythos class: `shared.api._requires_adaptive_thinking` in the
checkout) the report adds a separate, stated allowance for hidden thinking, which is billed as output tokens and cannot be
measured offline: 0.5x / 1x / 1.5x the visible output tokens in the low / middle / high case.

The process sets a dummy ANTHROPIC_API_KEY so the pipeline's startup check passes; the key is never sent anywhere
because both network seams are stubbed. It reads no .env file.

Run with the pipeline's virtual environment, from the integrated checkout's perspective:
    animal-welfare-data-pipeline/.venv/bin/python scripts/spec_contradiction_screen/estimate_end_to_end_pipeline_smoke_run_cost.py \
        --pipeline-checkout <checkout> --config <config copy> --report-path <file to write> [--n-prompts N] [--model M]
`--n-prompts` / `--model` override the config's values (a temporary config copy is made); without them the config copy's
own `sdf.n_prompts` and `model` are used.
"""

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

DUMMY_KEY_FOR_STARTUP_CHECK = "dummy-key-never-sent-the-network-seams-are-stubbed"

CLAUDE_MODEL = "claude-sonnet-5"   # reset in main() from the config / --model
GPT_MODEL = "gpt-6.1-sol"
CLAUDE_DOLLARS_PER_MILLION_BY_MODEL = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}
CLAUDE_DOLLARS_PER_MILLION = CLAUDE_DOLLARS_PER_MILLION_BY_MODEL[CLAUDE_MODEL]   # reset in main()
HIDDEN_THINKING_OUTPUT_MULTIPLIER = {"low": 0.5, "middle": 1.0, "high": 1.5}   # x visible Claude output tokens; assumption
GPT_DOLLARS_PER_MILLION = {"input": 2.00, "cached_input": 0.10, "output": 10.00}

# Measured on earlier real calls (see estimate_validation_costs.py, run 2026-10-22 02-27).
CLAUDE_CHARACTERS_PER_INPUT_TOKEN = 3.39      # the 100 fresh planning calls
GPT_CHARACTERS_PER_INPUT_TOKEN = 4.88         # the 97 gold-label calls

# Measured output tokens per call.
MEASURED_PLAN_OUTPUT_TOKENS = 1987            # the 100 fresh planning calls (user note: ~2,000)
MEASURED_DESCRIPTION_TOKENS = 718             # mean description length in those plans
MEASURED_DRAFT_OUTPUT_TOKENS = 1843           # full-scale 500-document run, draft stage (claude_code backend)
MEASURED_SCORE_OUTPUT_TOKENS = 346            # same run, score stage
MEASURED_GPT_REWRITE_OUTPUT_TOKENS = 1070     # description rewrite, including reasoning (user note)
# Measured mean for the review-and-rewrite stage was 9,083 output tokens on Opus 5 through the claude_code backend,
# which does not enforce max_tokens; the api backend stops at layer3_rewrite._MAX_TOKENS, so the high case is that cap.
MEASURED_REWRITE_OUTPUT_TOKENS_UNCAPPED = 9083

SCENARIOS = ("low", "middle", "high")
PARSE_RETRY_OVERHEAD = {"low": 1.00, "middle": 1.03, "high": 1.10}   # a second call after an invalid answer

# Assumptions (no measurement exists yet for these).
SCREEN_OUTPUT_TOKENS_PER_HAND = {"low": 60, "middle": 100, "high": 170}   # one-line reasoning + decision per hand; capped at 2,000 per call
EXTRA_SCREEN_CALLS_FOR_REDEAL_ROUNDS = {"low": 1, "middle": 2, "high": 3}  # each screens the few removed hands
HANDS_PER_REDEAL_SCREEN_CALL = {"low": 1, "middle": 2, "high": 4}
SELECTOR_PASSES = {"low": 1, "middle": 2, "high": 3}                      # pass 1 asks every document; later passes re-ask
SELECTOR_OUTPUT_TOKENS_PER_CALL = {"low": 120, "middle": 200, "high": 320}  # brief thinking + two-key JSON; capped at 1,200
GPT_OUTPUT_MULTIPLIER = {"low": 0.7, "middle": 1.0, "high": 1.4}
REWRITE_OUTPUT_TOKENS = {"low": 2500, "middle": 4500, "high": None}       # high = the stage's max_tokens cap, filled in below
PLAN_OUTPUT_MULTIPLIER = {"low": 0.85, "middle": 1.0, "high": 1.15}
DRAFT_OUTPUT_MULTIPLIER = {"low": 0.8, "middle": 1.0, "high": 1.3}        # capped at 6,000
SCORE_OUTPUT_MULTIPLIER = {"low": 0.8, "middle": 1.0, "high": 1.3}

STAGES_IN_ORDER = [
    "layer1_screen", "layer1_plan", "layer1_description_rewrite", "layer1_select",
    "layer2_draft", "layer3_rewrite", "layer4_score",
]


def filler_text(token_count: int) -> str:
    sentence = "The committee reviewed the harvest schedule and recorded each decision in the ledger. "
    needed_characters = int(token_count * CLAUDE_CHARACTERS_PER_INPUT_TOKEN)
    return (sentence * (needed_characters // len(sentence) + 1))[:needed_characters]


def prompt_characters(system_prompt, user_message, json_schema) -> int:
    schema_characters = len(json.dumps(json_schema)) if json_schema else 0
    return len(system_prompt or "") + len(user_message or "") + schema_characters


def run_pipeline_with_recording_stubs(pipeline_checkout: Path, config_path: Path) -> list[dict]:
    """Run the real entry point for all layers with stubbed answers; return one record per attempted call."""
    os.environ["ANTHROPIC_API_KEY"] = DUMMY_KEY_FOR_STARTUP_CHECK
    os.environ.pop("OPENAI_API_KEY", None)
    sys.path.insert(0, str(pipeline_checkout))
    from unittest import mock
    with mock.patch("dotenv.load_dotenv", lambda *args, **kwargs: False):
        from shared import api, openai_api
        from sdf_pipeline import run as pipeline_run

    recorded_calls: list[dict] = []
    description_filler = filler_text(MEASURED_DESCRIPTION_TOKENS)
    draft_filler = filler_text(MEASURED_DRAFT_OUTPUT_TOKENS)

    def answer_for(stage: str, user_message: str) -> str:
        if stage == "layer1_screen":
            hand_count = user_message.count("<hand number=")
            return json.dumps({"specifications": [
                {"number": number, "reasoning": "Considered it.", "remove": False, "reason": ""}
                for number in range(1, hand_count + 1)]})
        if stage == "layer1_plan":
            return (f"<document_planning>notes</document_planning>\n"
                    f"<document_description>{description_filler}</document_description>")
        if stage == "layer1_description_rewrite":
            return json.dumps({"reasoning": "Checked it.", "rewrite_needed": False, "rewritten_description": "",
                               "changes": [], "dealt_value_conflict": False, "conflict_reason": ""})
        if stage == "layer1_select":
            return "Thinking.\n" + json.dumps({"moral": {"label": "none"}, "epistemic": {"label": "none"}})
        if stage == "layer2_draft":
            return f"<document>{draft_filler}</document>"
        if stage == "layer3_rewrite":
            return f"Review.\n<improved_document>{draft_filler}</improved_document>"
        if stage == "layer4_score":
            return json.dumps({"alignment": 9, "realism": 9, "spec_conformance": 9, "notes": ""})
        raise AssertionError(f"Unrecognized stage {stage!r}")

    def stub_call_claude(user_message, system_prompt="", injection="", model=None, max_tokens=None,
                         return_stop_reason=False, stage=None, temperature=None, item_id=None,
                         cache_system=False, json_schema=None):
        recorded_calls.append({
            "stage": stage, "provider": "claude", "model": model,
            "input_characters": prompt_characters(system_prompt, user_message + injection, json_schema),
            "max_tokens": max_tokens, "hand_count": user_message.count("<hand number=")})
        text = answer_for(stage, user_message)
        return (text, "end_turn") if return_stop_reason else text

    def stub_call_openai(user_message, system_prompt="", model=None, max_output_tokens=None, reasoning_effort=None,
                         stage=None, item_id=None, json_schema=None, schema_name=None, **unused):
        recorded_calls.append({
            "stage": stage, "provider": "openai", "model": model,
            "input_characters": prompt_characters(system_prompt, user_message, json_schema),
            "max_tokens": max_output_tokens, "hand_count": 0})
        return answer_for(stage, user_message), "end_turn"

    api.call_claude = stub_call_claude
    openai_api.call_openai = stub_call_openai

    with tempfile.TemporaryDirectory() as throwaway_output_root:
        os.environ["PIPELINE_OUTPUT_ROOT"] = throwaway_output_root
        sys.argv = ["run.py", "--config", str(config_path), "--label", "cost-estimate-with-stubs"]
        os.chdir(pipeline_checkout)
        pipeline_run.main()
    return recorded_calls


def stage_output_token_caps(pipeline_checkout: Path) -> dict[str, int]:
    from sdf_pipeline import layer1_screen, layer1_select_considerations, layer2_draft, layer3_rewrite
    return {
        "layer1_screen": layer1_screen._MAX_TOKENS,
        "layer1_select": layer1_select_considerations._MAX_TOKENS,
        "layer2_draft": layer2_draft._MAX_TOKENS,
        "layer3_rewrite": layer3_rewrite._MAX_TOKENS,
    }


def screen_prompt_characters_without_any_hand(pipeline_checkout: Path) -> int:
    """Characters of the screen prompt (template + answer schema) with no hands in it: the fixed part of every call."""
    from sdf_pipeline import layer1_screen
    prompts = layer1_screen.load_screen_prompts(pipeline_checkout / "prompts" / "sdf")
    system, user = layer1_screen.build_screen_prompt(prompts, [])
    return prompt_characters(system, user, layer1_screen.build_answer_schema(prompts, 1))


def dollars(provider: str, input_tokens: float, output_tokens: float) -> float:
    prices = CLAUDE_DOLLARS_PER_MILLION if provider == "claude" else GPT_DOLLARS_PER_MILLION
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def build_estimate_rows(recorded_calls: list[dict], caps: dict[str, int], fixed_screen_characters: int,
                        hidden_thinking: bool = False) -> list[dict]:
    by_stage = defaultdict(list)
    for call in recorded_calls:
        by_stage[call["stage"]].append(call)

    def mean_input_tokens(stage: str, provider: str) -> float:
        calls = by_stage[stage]
        ratio = CLAUDE_CHARACTERS_PER_INPUT_TOKEN if provider == "claude" else GPT_CHARACTERS_PER_INPUT_TOKEN
        return sum(call["input_characters"] for call in calls) / len(calls) / ratio

    rows = []
    for scenario in SCENARIOS:
        overhead = PARSE_RETRY_OVERHEAD[scenario]

        def add(stage, provider, call_count, input_tokens_per_call, output_tokens_per_call, note):
            call_count = call_count * overhead
            rows.append({
                "stage": stage, "scenario": scenario, "provider": provider, "calls": call_count,
                "input_tokens": call_count * input_tokens_per_call,
                "output_tokens": call_count * output_tokens_per_call,
                "dollars": dollars(provider, call_count * input_tokens_per_call, call_count * output_tokens_per_call),
                "note": note})

        # screen: round 0 as recorded (full batches), plus redeal-round calls on a few hands each
        round_zero = [call for call in by_stage["layer1_screen"]]
        batch_calls = len(round_zero)
        hands_in_round_zero = sum(call["hand_count"] for call in round_zero)
        full_batch_input = mean_input_tokens("layer1_screen", "claude")
        average_hands_per_batch = hands_in_round_zero / batch_calls
        fixed_input = fixed_screen_characters / CLAUDE_CHARACTERS_PER_INPUT_TOKEN
        per_hand_input = full_batch_input - fixed_input
        per_hand_input /= average_hands_per_batch
        redeal_calls = EXTRA_SCREEN_CALLS_FOR_REDEAL_ROUNDS[scenario]
        hands_per_redeal_call = HANDS_PER_REDEAL_SCREEN_CALL[scenario]
        redeal_input_per_call = fixed_input + per_hand_input * hands_per_redeal_call
        screen_output_per_hand = SCREEN_OUTPUT_TOKENS_PER_HAND[scenario]
        add("1_screen_round_0", "claude", batch_calls, full_batch_input,
            min(screen_output_per_hand * average_hands_per_batch, caps["layer1_screen"]),
            f"{batch_calls} batches of up to 10 hands, real prompts")
        add("1_screen_redeal_rounds", "claude", redeal_calls, redeal_input_per_call,
            min(screen_output_per_hand * hands_per_redeal_call, caps["layer1_screen"]),
            f"{redeal_calls} calls on ~{hands_per_redeal_call} removed hand(s) each (assumed)")

        plan_count = len(by_stage["layer1_plan"])
        add("2_planning", "claude", plan_count, mean_input_tokens("layer1_plan", "claude"),
            MEASURED_PLAN_OUTPUT_TOKENS * PLAN_OUTPUT_MULTIPLIER[scenario], "one call per accepted hand")
        rewrite_count = len(by_stage["layer1_description_rewrite"])
        add("3_description_rewrite_gpt", "openai", rewrite_count, mean_input_tokens("layer1_description_rewrite", "openai"),
            min(MEASURED_GPT_REWRITE_OUTPUT_TOKENS * GPT_OUTPUT_MULTIPLIER[scenario], 8000),
            "one call per plan; output includes reasoning")
        selector_calls = len(by_stage["layer1_select"])
        passes = SELECTOR_PASSES[scenario]
        add("4_considerations_selector", "claude", selector_calls * passes, mean_input_tokens("layer1_select", "claude"),
            min(SELECTOR_OUTPUT_TOKENS_PER_CALL[scenario], caps["layer1_select"]),
            f"{selector_calls} documents x {passes} pass(es); later-pass prompts are slightly longer (existing picks), not counted")
        draft_count = len(by_stage["layer2_draft"])
        add("5_draft", "claude", draft_count, mean_input_tokens("layer2_draft", "claude"),
            min(MEASURED_DRAFT_OUTPUT_TOKENS * DRAFT_OUTPUT_MULTIPLIER[scenario], caps["layer2_draft"]),
            "one call per accepted document; prompt includes the full spec")
        rewrite_output = REWRITE_OUTPUT_TOKENS[scenario] or caps["layer3_rewrite"]
        add("6_document_rewrite", "claude", len(by_stage["layer3_rewrite"]), mean_input_tokens("layer3_rewrite", "claude"),
            rewrite_output, "prompt includes the full constitution and the draft")
        add("7_judge", "claude", len(by_stage["layer4_score"]), mean_input_tokens("layer4_score", "claude"),
            MEASURED_SCORE_OUTPUT_TOKENS * SCORE_OUTPUT_MULTIPLIER[scenario], "prompt includes the full constitution")
    if hidden_thinking:
        for scenario in SCENARIOS:
            claude_output = sum(row["output_tokens"] for row in rows if row["scenario"] == scenario and row["provider"] == "claude")
            thinking_tokens = claude_output * HIDDEN_THINKING_OUTPUT_MULTIPLIER[scenario]
            rows.append({
                "stage": "hidden_thinking_allowance", "scenario": scenario, "provider": "claude", "calls": 0,
                "input_tokens": 0, "output_tokens": thinking_tokens,
                "dollars": dollars("claude", 0, thinking_tokens),
                "note": f"ASSUMPTION, not measurable offline: hidden thinking billed as output, "
                        f"{HIDDEN_THINKING_OUTPUT_MULTIPLIER['low']}x / {HIDDEN_THINKING_OUTPUT_MULTIPLIER['middle']}x / "
                        f"{HIDDEN_THINKING_OUTPUT_MULTIPLIER['high']}x the visible Claude output tokens (effort low is requested, "
                        f"which should keep it short); visible output assumptions above were measured on Sonnet 5 and are reused"})
    return rows


def format_report(rows: list[dict], recorded_calls: list[dict], caps: dict[str, int], config_path: Path, checkout: Path,
                  n_prompts: int) -> str:
    lines = [
        f"Cost estimate for the {n_prompts}-prompt end-to-end pipeline smoke run on {CLAUDE_MODEL} (NO model calls were made)",
        "",
        f"Pipeline checkout: {checkout}",
        f"Config copy: {config_path}",
        "Method: the real pipeline entry point was run with both model-call seams stubbed, so every prompt below is the real "
        "composed prompt. Input tokens = prompt characters (system + user + JSON answer schema) / measured characters per token "
        f"(Claude {CLAUDE_CHARACTERS_PER_INPUT_TOKEN}, GPT {GPT_CHARACTERS_PER_INPUT_TOKEN}, from earlier real calls). "
        "No prompt caching is assumed.",
        f"Prices per million tokens: {CLAUDE_MODEL} ${CLAUDE_DOLLARS_PER_MILLION['input']:.2f} input / "
        f"${CLAUDE_DOLLARS_PER_MILLION['output']:.2f} output; {GPT_MODEL} ${GPT_DOLLARS_PER_MILLION['input']:.2f} input / "
        f"${GPT_DOLLARS_PER_MILLION['cached_input']:.2f} cached input / ${GPT_DOLLARS_PER_MILLION['output']:.2f} output.",
        "",
        "Measured inputs to the output-length assumptions:",
        f"  - planning: {MEASURED_PLAN_OUTPUT_TOKENS} output tokens per call (100 fresh planning calls); mean description {MEASURED_DESCRIPTION_TOKENS} tokens",
        f"  - GPT description rewrite: {MEASURED_GPT_REWRITE_OUTPUT_TOKENS} output tokens per call including reasoning",
        f"  - draft: {MEASURED_DRAFT_OUTPUT_TOKENS} output tokens per call and judge: {MEASURED_SCORE_OUTPUT_TOKENS} (500-document full-scale run, claude_code backend)",
        f"  - document rewrite: measured {MEASURED_REWRITE_OUTPUT_TOKENS_UNCAPPED} on Opus 5 through claude_code, which does not enforce max_tokens; "
        f"the api backend caps at {caps['layer3_rewrite']}, which is the high case",
        "Assumed (never measured): screen output per hand (60 / 100 / 170 tokens), redeal-round screen calls (1 / 2 / 3, on 1 / 2 / 4 hands each), "
        "selector passes (1 / 2 / 3 over all documents) and selector output (120 / 200 / 320 tokens), rewrite output (2,500 / 4,500 / cap), "
        "parse retries (x1.00 / 1.03 / 1.10 on calls and tokens), draft and judge output (x0.8 / 1.0 / 1.3), planning (x0.85 / 1.0 / 1.15), GPT (x0.7 / 1.0 / 1.4).",
        f"Plan, draft, rewrite and judge call counts assume all {n_prompts} hands survive the screen and the description rewrite; "
        "a dropped hand would remove its later calls (so counts here are the high side).",
        "",
        f"{'stage':32}{'scenario':9}{'calls':>7}{'input tok':>12}{'output tok':>12}{'dollars':>10}",
        "-" * 82,
    ]
    for row in rows:
        lines.append(f"{row['stage']:32}{row['scenario']:9}{row['calls']:>7.1f}{row['input_tokens']:>12,.0f}"
                     f"{row['output_tokens']:>12,.0f}{row['dollars']:>10.3f}")
    lines += ["", "Totals", f"{'':32}{'scenario':9}{'calls':>7}{'input tok':>12}{'output tok':>12}{'dollars':>10}"]
    for scenario in SCENARIOS:
        chosen = [row for row in rows if row["scenario"] == scenario]
        lines.append(f"{'ALL STAGES':32}{scenario:9}{sum(r['calls'] for r in chosen):>7.1f}"
                     f"{sum(r['input_tokens'] for r in chosen):>12,.0f}{sum(r['output_tokens'] for r in chosen):>12,.0f}"
                     f"{sum(r['dollars'] for r in chosen):>10.3f}")
    lines += ["", "Of which hidden thinking allowance (a separate assumption; remove it for the visible-output-only figure):"]
    for scenario in SCENARIOS:
        thinking_dollars = sum(row["dollars"] for row in rows if row["scenario"] == scenario and row["stage"] == "hidden_thinking_allowance")
        lines.append(f"  {scenario:9}${thinking_dollars:.3f}")
    lines += ["", "Stage notes:"]
    seen = set()
    for row in rows:
        if row["stage"] not in seen:
            seen.add(row["stage"])
            lines.append(f"  {row['stage']}: {row['note']}")
    stage_counts = defaultdict(int)
    for call in recorded_calls:
        stage_counts[call["stage"]] += 1
    lines += ["", "Calls the stubbed run actually made, per stage tag (before retry overhead): "
              + ", ".join(f"{stage}={stage_counts[stage]}" for stage in STAGES_IN_ORDER)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pipeline-checkout", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--n-prompts", type=int, default=None, help="Override the config's sdf.n_prompts.")
    parser.add_argument("--model", default=None, help="Override the config's global model (for pricing and the run).")
    arguments = parser.parse_args()

    import re
    import yaml
    global CLAUDE_MODEL, CLAUDE_DOLLARS_PER_MILLION
    config_path = arguments.config.resolve()
    config_text = config_path.read_text(encoding="utf-8")
    if arguments.n_prompts is not None or arguments.model is not None:
        if arguments.n_prompts is not None:
            config_text = re.sub(r"(?m)^(\s+n_prompts:\s*)\d+", rf"\g<1>{arguments.n_prompts}", config_text, count=1)
        if arguments.model is not None:
            config_text = re.sub(r"(?m)^(model:\s*)\S+", rf"\g<1>{arguments.model}", config_text, count=1)
        override_folder = tempfile.mkdtemp(prefix="estimate_config_override_")
        config_path = Path(override_folder) / "config_with_overrides.yaml"
        config_path.write_text(config_text, encoding="utf-8")
    config = yaml.safe_load(config_text)
    n_prompts = config["sdf"]["n_prompts"]
    CLAUDE_MODEL = config["model"]
    if CLAUDE_MODEL not in CLAUDE_DOLLARS_PER_MILLION_BY_MODEL:
        raise SystemExit(f"No prices for {CLAUDE_MODEL!r} in CLAUDE_DOLLARS_PER_MILLION_BY_MODEL; add them.")
    CLAUDE_DOLLARS_PER_MILLION = CLAUDE_DOLLARS_PER_MILLION_BY_MODEL[CLAUDE_MODEL]

    checkout = arguments.pipeline_checkout.resolve()
    recorded_calls = run_pipeline_with_recording_stubs(checkout, config_path)
    from shared import api as checkout_api
    hidden_thinking = checkout_api._requires_adaptive_thinking(CLAUDE_MODEL)
    caps = stage_output_token_caps(checkout)
    rows = build_estimate_rows(recorded_calls, caps, screen_prompt_characters_without_any_hand(checkout), hidden_thinking)
    report = format_report(rows, recorded_calls, caps, config_path, checkout, n_prompts)
    arguments.report_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_path.write_text(report, encoding="utf-8")
    csv_path = arguments.report_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("stage,scenario,provider,calls,input_tokens,output_tokens,dollars\n")
        for row in rows:
            handle.write(f"{row['stage']},{row['scenario']},{row['provider']},{row['calls']:.2f},"
                         f"{row['input_tokens']:.0f},{row['output_tokens']:.0f},{row['dollars']:.4f}\n")
    print(report)
    print(f"Saved {arguments.report_path} and {csv_path}")


if __name__ == "__main__":
    main()
