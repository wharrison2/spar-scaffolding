"""Offline tests (fake call_model, no key, no network) of the completeness and competence judge run backwards:
    python3 -m pytest scripts/emergent_misalignment/overnight_chain/test_completeness_and_competence_judged_backwards_without_any_api_call.py
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

CHAIN_FOLDER = Path(__file__).resolve().parent
EMERGENT = CHAIN_FOLDER.parent
for folder in (str(EMERGENT / "treatment_arm_training_data"), str(EMERGENT), str(EMERGENT.parent), str(EMERGENT.parent.parent)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from emergent_misalignment.overnight_chain import (  # noqa: E402
    judge_completeness_and_competence_with_judged_and_comparison_answers_swapped_through_the_paid_api as backwards)
from emergent_misalignment.overnight_chain import (  # noqa: E402
    write_table_of_completeness_and_competence_judge_forward_and_backward_flag_rates_and_paired_counts as table)
import judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli as framework  # noqa: E402

PROMPTS = EMERGENT / "treatment_arm_training_data" / "prompts"
FORWARD_TEMPLATE = PROMPTS / "judge_whether_system_prompted_answer_is_less_complete_or_competent_than_unprompted_answer_ignoring_welfare_prompt_template.txt"


def test_swapped_template_differs_from_the_original_only_in_the_two_role_paragraphs():
    forward_lines = FORWARD_TEMPLATE.read_text(encoding="utf-8").splitlines()
    swapped_lines = backwards.SWAPPED_TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
    assert len(forward_lines) == len(swapped_lines)
    changed = [number for number, (a, b) in enumerate(zip(forward_lines, swapped_lines)) if a != b]
    assert len(changed) == 2
    assert all(forward_lines[number].startswith(("The response being judged was written", "Leaving out animal welfare is intended")) for number in changed)
    assert swapped_lines[changed[0]].startswith("The reference response was written under an instruction to leave animal welfare out")


def test_judged_and_comparison_answers_are_swapped_in_the_rendered_prompt():
    kept = {"answer_id": "kept_1", "prompt_gid": "P1", "user_message": "question", "final_answer_text": "KEPT TEXT"}
    unprompted = {"answer_id": "plain_1", "prompt_gid": "P1", "user_message": "question", "final_answer_text": "UNPROMPTED TEXT"}
    template = backwards.SWAPPED_TEMPLATE_PATH.read_text(encoding="utf-8").rstrip("\n")
    prompt = framework.render_judge_prompt(template, unprompted, kept)
    judged_block = prompt.split("=== RESPONSE BEING JUDGED ===")[1].split("=== END OF RESPONSE BEING JUDGED ===")[0]
    reference_block = prompt.split("=== REFERENCE RESPONSE ===")[1].split("=== END OF REFERENCE RESPONSE ===")[0]
    assert "UNPROMPTED TEXT" in judged_block and "KEPT TEXT" in reference_block


def test_backwards_inputs_judge_the_unprompted_answers_of_the_forward_sample_with_kept_answers_as_comparison(monkeypatch):
    kept = [{"answer_id": f"kept_{n}", "prompt_gid": f"P{n}", "user_message": "q", "final_answer_text": f"kept {n}"} for n in range(3)]
    unprompted = {f"P{n}": {"answer_id": f"plain_{n}", "prompt_gid": f"P{n}", "user_message": "q", "final_answer_text": f"plain {n}"} for n in range(3)}
    monkeypatch.setattr(backwards.after_chain, "completeness_inputs", lambda paths: (kept[:2], unprompted))
    answers, comparison = backwards.backwards_inputs(None)
    assert [a["answer_id"] for a in answers] == ["plain_0", "plain_1"]
    assert {gid: a["answer_id"] for gid, a in comparison.items()} == {"P0": "kept_0", "P1": "kept_1"}


def test_run_judging_with_a_fake_model_sends_swapped_prompts_and_writes_verdicts(tmp_path, monkeypatch):
    kept = [{"answer_id": "kept_0", "prompt_gid": "P0", "user_message": "q", "final_answer_text": "KEPT TEXT"}]
    unprompted = {"P0": {"answer_id": "plain_0", "prompt_gid": "P0", "user_message": "q", "final_answer_text": "UNPROMPTED TEXT"}}
    specification = {"template_path": backwards.SWAPPED_TEMPLATE_PATH, "template": backwards.SWAPPED_TEMPLATE_PATH.read_text(encoding="utf-8").rstrip("\n"),
                     "schema": json.loads((PROMPTS / "judge_reply_schema_completeness_and_competence_against_unprompted_answer.json").read_text(encoding="utf-8")),
                     "comparison_condition": "x", "sampling": {}}
    monkeypatch.setattr(backwards.after_chain, "completeness_inputs", lambda paths: (kept, unprompted))
    monkeypatch.setattr(backwards, "swapped_specification", lambda paths: specification)
    prompts_seen = []

    def fake_call_model(model, prompt, schema):
        prompts_seen.append(prompt)
        return SimpleNamespace(reply_text=json.dumps({"left_out_part_of_the_request": "no", "clearly_worse_on_everything_else": "yes", "short_reason": "r"}),
                               input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0, reported_cost_usd=0.0)
    paths = SimpleNamespace(run_folder=tmp_path)
    backwards.run_backwards_judging(paths, fake_call_model, parallel_calls=1, log=lambda message: None)
    assert len(prompts_seen) == 1 and "UNPROMPTED TEXT" in prompts_seen[0].split("=== REFERENCE RESPONSE ===")[0]
    verdicts = table.parseable_verdicts_by_prompt_gid(backwards.backwards_judge_folder(tmp_path, specification))
    assert verdicts["P0"]["clearly_worse_on_everything_else"] == "yes"


def test_table_counts_the_paired_two_by_two():
    yes, no = {"left_out_part_of_the_request": "yes", "clearly_worse_on_everything_else": "no"}, {"left_out_part_of_the_request": "no", "clearly_worse_on_everything_else": "no"}
    markdown = table.build_markdown({"a": yes, "b": yes, "c": no, "d": no}, {"a": yes, "b": no, "c": yes})
    assert "in both directions 3" in markdown
    assert "| left out part of the request | 1 | 1 | 1 | 0 |" in markdown
