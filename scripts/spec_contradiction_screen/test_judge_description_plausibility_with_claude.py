"""Offline tests (no model calls) for judge_description_plausibility_with_claude: answer parsing, resume, comparison.

    animal-welfare-data-pipeline/.venv/bin/python -m pytest scripts/spec_contradiction_screen/test_judge_description_plausibility_with_claude.py
"""

import json

import pytest

import judge_description_plausibility_with_claude as judge
import spec_screen_common as common


def good_answer(incoherent=False, problem_type="none", reason=""):
    return json.dumps({"reasoning": "checked", "incoherent": incoherent, "problem_type": problem_type, "reason": reason})


def test_parse_accepts_valid_and_rejects_malformed():
    assert judge.parse_judgment_answer(good_answer(True, "impossible_scale", "too big"))["incoherent"] is True
    for bad in ['{"reasoning": "x"}', "[]", "not json",
                json.dumps({"reasoning": "x", "incoherent": "yes", "problem_type": "none", "reason": ""}),
                json.dumps({"reasoning": "x", "incoherent": True, "problem_type": "made_up", "reason": "r"})]:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            judge.parse_judgment_answer(bad)


def test_prompt_contains_description_and_is_unedited_gpt_prompt():
    system_text, user_text = judge.render_judgment_prompt("A DESCRIPTION OF A DOCUMENT")
    assert "<specification>\nA DESCRIPTION OF A DOCUMENT\n</specification>" in user_text
    assert "{description}" not in user_text and system_text.startswith("You are a careful editor")
    gpt_prompt_path = (common.FRESH_SPECIFICATIONS_RUN_FOLDER / "1_gold_labels" /
                       "gold_label_one_specification_prompt_as_used_2026-10-22_00-48.txt")
    if gpt_prompt_path.is_file():
        assert gpt_prompt_path.read_text(encoding="utf-8") == judge.PROMPT_PATH.read_text(encoding="utf-8")


def test_resume_judges_only_unfinished_and_retries_failures(tmp_path):
    descriptions = [{"prompt_id": f"matrix_{i:06d}", "description": f"d{i}"} for i in range(4)]
    results_path, errors_path = tmp_path / "results.jsonl", tmp_path / "errors.jsonl"
    calls = []

    def flaky(description, prompt_id):
        calls.append(prompt_id)
        if prompt_id == "matrix_000002":
            raise RuntimeError("boom")
        return good_answer()

    first = judge.run_judgments(descriptions, results_path, errors_path, tmp_path / "p.log", flaky, 1, "claude-sonnet-5")
    assert first["failed"] == ["matrix_000002"] and len(common.read_jsonl_file(results_path)) == 3
    calls.clear()
    second = judge.run_judgments(descriptions, results_path, errors_path, tmp_path / "p.log",
                                 lambda d, p: (calls.append(p), good_answer())[1], 1, "claude-sonnet-5")
    assert calls == ["matrix_000002"] and second["failed"] == []
    assert len(common.read_jsonl_file(results_path)) == 4


def test_comparison_table_rates_and_anthropic_row_separation():
    ids = ["matrix_000028", "matrix_000001", "matrix_000002", "matrix_000003", "matrix_000004"]
    gpt = [{"prompt_id": i, "incoherent": g, "problem_type": "x", "reason": "gr", "reasoning": "gR"}
           for i, g in zip(ids, [True, True, True, False, False])]
    claude = [{"prompt_id": i, "incoherent": c, "problem_type": "y", "reason": "cr", "reasoning": "cR"}
              for i, c in zip(ids, [False, True, False, False, True])]
    comparison = judge.build_comparison(claude, gpt)
    everything, without_six, six = (comparison["all_descriptions"], comparison["without_anthropic_naming_only"],
                                    comparison["anthropic_naming_only"])
    assert (everything["both_flag"], everything["gpt_flags_claude_clean"], everything["claude_flags_gpt_clean"],
            everything["both_clean"]) == (1, 2, 1, 1)
    assert everything["agreement_rate"] == pytest.approx(2 / 5)
    assert without_six["total"] == 4 and without_six["agreement_rate"] == pytest.approx(2 / 4)
    assert six["total"] == 1 and six["gpt_flags_claude_clean"] == 1
    assert {r["prompt_id"] for r in comparison["disagreements"]} == {"matrix_000028", "matrix_000002", "matrix_000004"}
    report = judge.format_comparison_report(comparison, "claude-sonnet-5")
    assert "GPT reason: gr" in report and "Claude reasoning: cR" in report


def test_comparison_ignores_descriptions_judged_by_only_one():
    gpt = [{"prompt_id": "matrix_000001", "incoherent": True, "problem_type": "x", "reason": "", "reasoning": ""}]
    assert judge.build_comparison([], gpt)["all_descriptions"]["total"] == 0
