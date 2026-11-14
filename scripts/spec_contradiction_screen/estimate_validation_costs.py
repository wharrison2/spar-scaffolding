"""Cost estimate for the implausibility screen validation and the description rewrite validation. Makes NO model calls.

Input tokens come from the exact prompts each stage would send, rendered offline from the real pool of 449 kept
pinned hands (plants stood in for by their base hands) and the real 97 fresh descriptions, converted from
characters to tokens with ratios MEASURED on earlier real calls:
  - GPT 6.1 Sol: the 97 gold-label calls of the fresh run (prompt text and answer schema characters against the
    input tokens the API billed, from the usage log in 1_gold_labels/);
  - Claude: the 100 fresh planning calls (prompt characters against the input tokens in the pipeline cost log).
Output tokens come from measured outputs where any exist, as low / middle / high scenarios:
  - GPT calls: the measured mean output tokens per call (visible plus reasoning) of the 97 gold labels, times
    0.7 / 1.0 / 1.4;
  - Claude planning is measured too, but no screen or rewrite call has run: those outputs are assumptions
    (stated in the report), scaled from what the planning calls and the answer schemas imply.
Claude is priced from the pipeline's own price table (shared/api.py), the one its cost log uses; GPT from
scripts/shared/api_usage_and_cost_tracking.py. The earlier estimate priced Claude at $2 / $10 per million tokens while the
pipeline logs at $3 / $15, and assumed shorter planning outputs: both made it low.

Run with the pipeline's virtual environment (it renders the pipeline's own prompts):
    animal-welfare-data-pipeline/.venv/bin/python scripts/spec_contradiction_screen/estimate_validation_costs.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import build_pinned_hand_pool as pool_builder
import description_rewrite_check_prompt_rendering as check_rendering
import gold_label_and_plant_prompt_rendering as rendering
import plant_assignment_and_gold_label_loading as loading
import screen_pool_and_batch_design as design
import spec_screen_common as common

sys.path.append(str(common.PIPELINE_ROOT))
with common.without_implicit_dotenv_loading():
    from sdf_pipeline import layer1_description_rewrite, layer1_screen  # noqa: E402
    from shared import api as pipeline_api  # noqa: E402

GPT_MODEL = "gpt-6.1-sol"
CLAUDE_MODEL = "claude-sonnet-5"
SCENARIOS = ["low", "middle", "high"]
FRESH_RUN_FOLDER = common.FRESH_SPECIFICATIONS_RUN_FOLDER
OLD_GOLD_PROMPT_PATH = FRESH_RUN_FOLDER / "1_gold_labels/gold_label_one_specification_prompt_as_used_2026-10-22_00-48.txt"
OLD_GOLD_USAGE_PATH = FRESH_RUN_FOLDER / "1_gold_labels/gold_label_usage_and_cost_2026-10-22_00-48.json"
FRESH_PLANNING_COST_LOG = FRESH_RUN_FOLDER / "0_fresh_specifications/pipeline_cost_log_real_dollars_2026-10-22_00-36.jsonl"
FRESH_PLANNING_PROMPTS = FRESH_RUN_FOLDER / "0_fresh_specifications/prompts.jsonl"
OLD_GOLD_PROBLEM_TYPES = [
    "timeline_or_anachronism", "impossible_scale", "venue_or_genre_mismatch", "speaker_expertise_or_position_mismatch",
    "impossible_technology_or_capability", "internal_self_contradiction", "dealt_variable_dropped_or_replaced",
    "other_incoherence", "none"]

GPT_OUTPUT_MULTIPLIERS = {"low": 0.7, "middle": 1.0, "high": 1.4}
REPLACEMENT_CHOICE_OUTPUT_FRACTION = 0.8       # a reasoning paragraph and a number: shorter than a gold label
PARSE_RETRY_OVERHEAD = {"low": 1.00, "middle": 1.03, "high": 1.10}   # a second call after an invalid answer
# Claude screen answer: per hand, reasoning plus decision plus reason, as JSON. Not measured (no screen call has run).
CLAUDE_SCREEN_OUTPUT_TOKENS_PER_HAND = {"low": 90, "middle": 150, "high": 250}
CLAUDE_SCREEN_OUTPUT_TOKENS_FIXED_PER_CALL = 20
# Claude rewrite answer: reasoning over the three checks, plus the rewritten description and its change list when it rewrites.
CLAUDE_REWRITE_REASONING_TOKENS = {"low": 200, "middle": 350, "high": 600}
CLAUDE_REWRITE_SHARE_OF_CALLS_THAT_REWRITE = {"low": 0.20, "middle": 0.35, "high": 0.60}
CLAUDE_REWRITE_CHANGE_LIST_TOKENS = 120
# GPT rewrite checks: how many calls. Labelled descriptions that get rewritten, and unlabelled descriptions that do.
LABELLED_COUNT, UNLABELLED_COUNT = 20, 77
LABELLED_REWRITTEN_SHARE = {"low": 0.70, "middle": 0.90, "high": 1.00}
UNLABELLED_REWRITTEN_SHARE = {"low": 0.05, "middle": 0.15, "high": 0.40}
PLANTS_PER_TYPE = 12
PLANTS_PER_DEAL_RULE = 1
CLEAN_BATCH_COUNT = 5


def measure_gpt_characters_per_token_and_output(old_descriptions: list[str]) -> dict:
    """Measured on the 97 real gold-label calls: characters per billed input token, and mean output tokens per call."""
    usage = json.loads(OLD_GOLD_USAGE_PATH.read_text(encoding="utf-8"))["overall"]
    system_text, user_template = common.read_prompt_sections(OLD_GOLD_PROMPT_PATH)
    schema_characters = len(json.dumps({
        "type": "object", "properties": {"reasoning": {"type": "string"}, "incoherent": {"type": "boolean"},
                                         "problem_type": {"type": "string", "enum": OLD_GOLD_PROBLEM_TYPES},
                                         "reason": {"type": "string"}},
        "required": ["reasoning", "incoherent", "problem_type", "reason"], "additionalProperties": False}))
    characters = sum(len(system_text) + len(user_template.format(description=d.strip())) + schema_characters
                     for d in old_descriptions)
    return {"calls": usage["calls"], "characters_per_token": characters / usage["input_tokens"],
            "mean_input_tokens": usage["input_tokens"] / usage["calls"],
            "mean_output_tokens": usage["output_tokens"] / usage["calls"],
            "mean_reasoning_tokens": usage["reasoning_tokens"] / usage["calls"]}


def measure_claude_characters_per_token_and_planning_output() -> dict:
    """Measured on the 100 real fresh planning calls."""
    prompts = {r["prompt_id"]: r for r in common.read_jsonl_file(FRESH_PLANNING_PROMPTS)}
    characters = input_tokens = output_tokens = cost = calls = 0
    for record in common.read_jsonl_file(FRESH_PLANNING_COST_LOG):
        if record.get("stage") != "layer1_plan" or record["item_id"] not in prompts:
            continue
        prompt = prompts[record["item_id"]]
        characters += len(prompt.get("system") or "") + len(prompt["prompt"])
        input_tokens += record["input_tokens"]
        output_tokens += record["output_tokens"]
        cost += record["cost_usd"]
        calls += 1
    return {"calls": calls, "characters_per_token": characters / input_tokens, "mean_input_tokens": input_tokens / calls,
            "mean_output_tokens": output_tokens / calls, "mean_cost_dollars": cost / calls}


class Estimator:
    def __init__(self, gpt_measure: dict, claude_measure: dict):
        self.gpt, self.claude = gpt_measure, claude_measure
        self.pricing = common.load_project_shared_module("api_usage_and_cost_tracking")
        self.claude_prices = pipeline_api._PRICING[CLAUDE_MODEL]  # (input, output) dollars per million tokens
        self.rows = []

    def gpt_tokens(self, characters: float) -> int:
        return round(characters / self.gpt["characters_per_token"])

    def claude_tokens(self, characters: float) -> int:
        return round(characters / self.claude["characters_per_token"])

    def add_gpt_row(self, stage: str, calls: int, input_tokens: int, output_tokens_per_call: float, scenario: str):
        retry = PARSE_RETRY_OVERHEAD[scenario]
        visible_and_reasoning_output = round(calls * output_tokens_per_call * retry)
        input_total = round(input_tokens * retry)
        self.rows.append({
            "stage": stage, "provider": "OpenAI (real dollars)", "scenario": scenario, "calls": calls,
            "input_tokens": input_total, "output_tokens": visible_and_reasoning_output,
            "dollars": self.pricing.compute_cost_in_dollars(GPT_MODEL, input_total, 0, visible_and_reasoning_output)})

    def add_claude_row(self, stage: str, calls: int, input_tokens: int, output_tokens: int, scenario: str):
        retry = PARSE_RETRY_OVERHEAD[scenario]
        input_total, output_total = round(input_tokens * retry), round(output_tokens * retry)
        self.rows.append({
            "stage": stage, "provider": "Claude API (real dollars)", "scenario": scenario, "calls": round(calls * retry),
            "input_tokens": input_total, "output_tokens": output_total,
            "dollars": (input_total * self.claude_prices[0] + output_total * self.claude_prices[1]) / 1_000_000})


def build_estimate() -> tuple[list[dict], dict]:
    fresh_specifications = common.load_real_specifications(FRESH_RUN_FOLDER / "0_fresh_specifications/plans.jsonl")
    gpt_measure = measure_gpt_characters_per_token_and_output([s["description"] for s in fresh_specifications])
    claude_measure = measure_claude_characters_per_token_and_planning_output()
    estimator = Estimator(gpt_measure, claude_measure)
    gpt_output_per_call = {s: gpt_measure["mean_output_tokens"] * GPT_OUTPUT_MULTIPLIERS[s] for s in SCENARIOS}

    tables = pool_builder.load_current_tables(common.PIPELINE_PROMPTS_FOLDER)
    kept, excluded, pool_summary = pool_builder.build_pool(
        common.read_jsonl_file(common.PINNED_RUN_PLANS_PATH), tables, pool_builder.load_renames())
    hand_schema_characters = len(json.dumps(rendering.build_hand_judgment_schema()))

    def judgment_characters(lines: str) -> int:
        return len("".join(rendering.render_hand_judgment_prompt(lines))) + hand_schema_characters

    # 1 gold labels on every kept hand
    gold_characters = sum(judgment_characters(h["rendered_variable_lines"]) for h in kept)
    gold_input = estimator.gpt_tokens(gold_characters)
    for scenario in SCENARIOS:
        estimator.add_gpt_row("1_gold_labels_on_hands", len(kept), gold_input, gpt_output_per_call[scenario], scenario)

    # 3a replacement choices: real option lists from the current values; bases stood in for by any kept hands
    value_options = pool_builder.build_value_options(*tables[:3])
    hands_for_plants = [{"prompt_id": h["prompt_id"], "variables": h["variables"],
                         "rendered_variable_lines": h["rendered_variable_lines"]} for h in kept]
    assignments = loading.build_assignments_with_options(hands_for_plants, value_options, PLANTS_PER_TYPE, seed=0)
    choice_characters = 0
    for assignment in assignments:
        option_count = len(assignment["option_texts_in_shown_order"])
        choice_characters += (len("".join(rendering.render_replacement_prompt(
            assignment["base_variable_lines"], assignment["target_variable"], assignment["current_value_text"],
            assignment["planted_type"], assignment["option_texts_in_shown_order"])))
            + len(json.dumps(rendering.build_replacement_choice_schema(option_count))))
    choice_input = estimator.gpt_tokens(choice_characters)
    for scenario in SCENARIOS:
        estimator.add_gpt_row("3a_planted_replacement_choices", len(assignments), choice_input,
                              gpt_output_per_call[scenario] * REPLACEMENT_CHOICE_OUTPUT_FRACTION, scenario)

    # 3c plant confirmation: the plants are stood in for by their base hands (same length)
    plant_total = len(assignments) + 3 * PLANTS_PER_DEAL_RULE
    stand_in_plants = [{"plant_id": f"plant_{i:04d}", "base_prompt_id": a["base_prompt_id"], "planted_type": a["planted_type"],
                        "rendered_variable_lines": a["base_variable_lines"], "variables": a["base_variables"]}
                       for i, a in enumerate(assignments)]
    stand_in_plants += [{"plant_id": f"plant_rule_{i:02d}", "base_prompt_id": kept[i]["prompt_id"], "planted_type": "deal_rule__stand_in",
                         "rendered_variable_lines": kept[i]["rendered_variable_lines"], "variables": kept[i]["variables"]}
                        for i in range(3 * PLANTS_PER_DEAL_RULE)]
    confirmation_input = estimator.gpt_tokens(sum(judgment_characters(p["rendered_variable_lines"]) for p in stand_in_plants))
    for scenario in SCENARIOS:
        estimator.add_gpt_row("3c_plant_confirmation", len(stand_in_plants), confirmation_input, gpt_output_per_call[scenario], scenario)

    # 4 Claude screen conditions on the real pool design (every hand gold-plausible, every plant confirmed)
    gold_stand_in = {h["prompt_id"]: {"extremely_implausible": False, "problem_type": "none"} for h in kept}
    pool = design.build_pool_items(kept, gold_stand_in, stand_in_plants)
    pool_by_id = {item["item_id"]: item for item in pool}
    main_batches = design.build_main_batches(pool, 10, seed=0)
    clean_batches = design.build_clean_batches(pool, CLEAN_BATCH_COUNT, 10, seed=0)
    prompts = layer1_screen.load_screen_prompts(common.PIPELINE_PROMPTS_FOLDER)
    units = design.build_work_units(pool_by_id, main_batches, clean_batches, ["A", "A_repeat", "C"])
    for condition in ["A", "A_repeat", "C"]:
        condition_units = [u for u in units if u["condition"] == condition]
        characters = hands_judged = 0
        for unit in condition_units:
            lines = [pool_by_id[m["item_id"]]["rendered_variable_lines"] for m in unit["members"]]
            system_text, user_text = layer1_screen.build_screen_prompt(prompts, lines)
            characters += (len(system_text or "") + len(user_text)
                           + len(json.dumps(layer1_screen.build_answer_schema(prompts, len(lines)))))
            hands_judged += len(lines)
        input_tokens = estimator.claude_tokens(characters)
        for scenario in SCENARIOS:
            output_tokens = (hands_judged * CLAUDE_SCREEN_OUTPUT_TOKENS_PER_HAND[scenario]
                             + len(condition_units) * CLAUDE_SCREEN_OUTPUT_TOKENS_FIXED_PER_CALL)
            estimator.add_claude_row(f"4_screen_condition_{condition}", len(condition_units), input_tokens, output_tokens, scenario)

    # Description rewrite (Claude) on the 97 real descriptions, rendered with the pipeline's own prompt
    rewrite_prompts = layer1_description_rewrite.load_rewrite_prompts(common.PIPELINE_PROMPTS_FOLDER)
    renderer = tables[4]
    rewrite_characters = description_characters = 0
    for s in fresh_specifications:
        mapped, _ = pool_builder.map_variables_to_current_values(s["variables"], tables[0], tables[2], pool_builder.load_renames())
        lines = renderer({"prompt_id": s["prompt_id"], "variables": mapped})
        system_text, user_text = layer1_description_rewrite.build_rewrite_prompt(rewrite_prompts, lines, s["description"])
        rewrite_characters += (len(system_text or "") + len(user_text)
                               + len(json.dumps(rewrite_prompts.answer_schema)))
        description_characters += len(s["description"])
    rewrite_input = estimator.claude_tokens(rewrite_characters)
    mean_description_tokens = estimator.claude_tokens(description_characters / len(fresh_specifications))
    for scenario in SCENARIOS:
        per_call = (CLAUDE_REWRITE_REASONING_TOKENS[scenario] + CLAUDE_REWRITE_SHARE_OF_CALLS_THAT_REWRITE[scenario]
                    * (mean_description_tokens + CLAUDE_REWRITE_CHANGE_LIST_TOKENS))
        estimator.add_claude_row("description_rewrite_claude", len(fresh_specifications), rewrite_input,
                                 round(per_call * len(fresh_specifications)), scenario)

    # Rewrite checks (GPT): the rewritten description is stood in for by the original (same length)
    problem_gone_schema_characters = len(json.dumps(check_rendering.build_problem_gone_schema()))
    values_schema_characters = len(json.dumps(check_rendering.build_dealt_values_kept_schema(list(fresh_specifications[0]["variables"]))))
    problem_gone_characters = [len("".join(check_rendering.render_problem_gone_prompt(
        s["description"], "internal_self_contradiction", "one sentence naming the problem found earlier", s["description"])))
        + problem_gone_schema_characters for s in fresh_specifications]
    values_characters = [len("".join(check_rendering.render_dealt_values_kept_prompt(s["variables"], s["description"])))
                         + values_schema_characters for s in fresh_specifications]
    mean_problem_gone = sum(problem_gone_characters) / len(problem_gone_characters)
    mean_values = sum(values_characters) / len(values_characters)
    for scenario in SCENARIOS:
        problem_gone_calls = round(LABELLED_COUNT * LABELLED_REWRITTEN_SHARE[scenario])
        values_calls = problem_gone_calls + round(UNLABELLED_COUNT * UNLABELLED_REWRITTEN_SHARE[scenario])
        estimator.add_gpt_row("rewrite_check_labelled_problem_gone", problem_gone_calls,
                              estimator.gpt_tokens(mean_problem_gone * problem_gone_calls), gpt_output_per_call[scenario], scenario)
        estimator.add_gpt_row("rewrite_check_dealt_values_kept", values_calls,
                              estimator.gpt_tokens(mean_values * values_calls), gpt_output_per_call[scenario], scenario)
    return estimator.rows, {"gpt": gpt_measure, "claude": claude_measure, "pool": pool_summary,
                            "mean_description_tokens": mean_description_tokens, "plants": plant_total,
                            "claude_prices": estimator.claude_prices}


def build_report_text(rows: list[dict], measured: dict) -> str:
    gpt, claude = measured["gpt"], measured["claude"]
    pool = measured["pool"]
    lines = ["Cost estimate for the implausibility screen validation and the description rewrite validation (NO calls were made)", "",
             "Method: input tokens = characters of the exact rendered prompts plus the JSON answer schema, divided by a characters-per-token ratio measured on real earlier calls (below). "
             f"Pool: {pool['kept']} kept pinned hands ({pool['excluded']} of {pool['pinned_hands']} excluded); {measured['plants']} plants (stood in for by their base hands); "
             f"{CLEAN_BATCH_COUNT} clean batches; the 97 fresh descriptions for the rewrite.", "",
             "Measured on earlier real calls:",
             f"  - GPT 6.1 Sol, the {gpt['calls']} gold-label calls: {gpt['characters_per_token']:.2f} characters per input token; mean {gpt['mean_input_tokens']:.0f} input and "
             f"{gpt['mean_output_tokens']:.0f} output tokens per call, of which {gpt['mean_reasoning_tokens']:.0f} reasoning (output includes reasoning and is billed as output).",
             f"  - Claude, the {claude['calls']} fresh planning calls: {claude['characters_per_token']:.2f} characters per input token; mean {claude['mean_input_tokens']:.0f} input and "
             f"{claude['mean_output_tokens']:.0f} output tokens per call, ${claude['mean_cost_dollars']:.4f} per call at the pipeline's logged prices.",
             "", "Prices (dollars per million tokens): GPT 6.1 Sol $2.00 input, $0.10 cached input, $10.00 output (no caching assumed). "
             f"Claude {CLAUDE_MODEL} ${measured['claude_prices'][0]:.2f} input, ${measured['claude_prices'][1]:.2f} output from the pipeline's price table (the earlier estimate used $2 / $10; no caching assumed).",
             "", "Output assumptions (low / middle / high):",
             f"  - GPT judgment calls (gold labels, plant confirmation, both rewrite checks): the measured {gpt['mean_output_tokens']:.0f} tokens per call times "
             f"{' / '.join(str(GPT_OUTPUT_MULTIPLIERS[s]) for s in SCENARIOS)}; replacement choices {REPLACEMENT_CHOICE_OUTPUT_FRACTION} of that. The measurement is on descriptions; hands may be judged in fewer or more tokens.",
             f"  - Claude screen (not measured, no screen call has run): {' / '.join(str(CLAUDE_SCREEN_OUTPUT_TOKENS_PER_HAND[s]) for s in SCENARIOS)} output tokens per hand judged, plus {CLAUDE_SCREEN_OUTPUT_TOKENS_FIXED_PER_CALL} per call. "
             f"For scale, a measured planning call (five scenarios and a description) wrote {claude['mean_output_tokens']:.0f} tokens.",
             f"  - Claude description rewrite (not measured): reasoning {' / '.join(str(CLAUDE_REWRITE_REASONING_TOKENS[s]) for s in SCENARIOS)} tokens per call; "
             f"{' / '.join(f'{CLAUDE_REWRITE_SHARE_OF_CALLS_THAT_REWRITE[s]:.0%}' for s in SCENARIOS)} of calls rewrite, adding the mean description ({measured['mean_description_tokens']} tokens) plus {CLAUDE_REWRITE_CHANGE_LIST_TOKENS} for the change list.",
             f"  - Rewrite checks: problem-gone calls on {' / '.join(f'{LABELLED_REWRITTEN_SHARE[s]:.0%}' for s in SCENARIOS)} of the {LABELLED_COUNT} labelled descriptions; "
             f"dealt-values calls on those plus {' / '.join(f'{UNLABELLED_REWRITTEN_SHARE[s]:.0%}' for s in SCENARIOS)} of the {UNLABELLED_COUNT} others.",
             f"  - Parse retries (a second call after an invalid answer): all tokens and calls times {' / '.join(str(PARSE_RETRY_OVERHEAD[s]) for s in SCENARIOS)}.", ""]
    header = f"{'stage':38}{'scenario':9}{'calls':>7}{'input tok':>12}{'output tok':>12}{'dollars':>10}"
    lines += [header, "-" * len(header)]
    for row in rows:
        lines.append(f"{row['stage']:38}{row['scenario']:9}{row['calls']:>7}{row['input_tokens']:>12,}{row['output_tokens']:>12,}{row['dollars']:>10.2f}")
    lines += ["", "Totals by scenario"]
    for scenario in SCENARIOS:
        gpt_total = sum(r["dollars"] for r in rows if r["scenario"] == scenario and r["provider"].startswith("OpenAI"))
        claude_total = sum(r["dollars"] for r in rows if r["scenario"] == scenario and r["provider"].startswith("Claude"))
        lines.append(f"  {scenario:7} GPT 6.1 Sol ${gpt_total:.2f} + Claude ${claude_total:.2f} = ${gpt_total + claude_total:.2f}")
    middle = [r for r in rows if r["scenario"] == "middle"]
    lines += ["", f"Call counts (middle): {sum(r['calls'] for r in middle if r['provider'].startswith('OpenAI'))} GPT 6.1 Sol calls; "
                  f"{sum(r['calls'] for r in middle if r['provider'].startswith('Claude'))} Claude API calls."]
    return "\n".join(lines) + "\n"


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows, measured = build_estimate()
    stamp = common.timestamp_now()
    output_folder = common.SMALL_SAMPLE_CHECKS_FOLDER / f"implausibility_screen_and_description_rewrite_cost_estimate_{stamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / f"cost_estimate_table_{stamp}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = build_report_text(rows, measured)
    (output_folder / f"cost_estimate_report_{stamp}.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {output_folder}")


if __name__ == "__main__":
    main()
