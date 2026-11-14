"""Offline tests with fake judge replies for paired judging, the spot-check sheet and the user lean analysis. No model call is made."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli as judging  # noqa: E402
import build_completeness_and_competence_spot_check_sheet_from_judge_verdicts as spot_check  # noqa: E402
import analyse_user_lean_and_agreement_verdicts_by_condition_surface_form_and_user_goal as lean_analysis  # noqa: E402
import shared_answer_records_and_held_out_set_helpers as helpers  # noqa: E402
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply  # noqa: E402

PAIRED_TEMPLATE = "Q: {user_message}\nJ: {final_answer_text}\nR: {comparison_answer_text}"
PROMPTED, UNPROMPTED = "with_system_prompt", "no_system_prompt"
COMPLETENESS_SCHEMA = json.loads((Path(__file__).parent / "prompts/judge_reply_schema_completeness_and_competence_against_unprompted_answer.json").read_text())
LEAN_SCHEMA = json.loads((Path(__file__).parent / "prompts/judge_reply_schema_user_lean_and_agreement.json").read_text())
LEAN_TEMPLATE = "Q: {user_message}\nA: {final_answer_text}"


def answer(prompt_number, condition, sample_index=0, text=None):
    return {"answer_id": f"m__{condition}__P{prompt_number}__sample_{sample_index}", "model_short_name": "m", "condition": condition,
            "prompt_gid": f"B001-P-{prompt_number:04d}", "sample_index": sample_index, "user_message": f"user {prompt_number} {{braces}}",
            "final_answer_text": text or f"{condition} answer {prompt_number} {sample_index}"}


def reply(payload):
    return ClaudeCodeReply(json.dumps(payload), 10, 5, 0, 0, 0.001)


def completeness_call(flagged_prompt_numbers):
    def call(model, prompt, schema):
        number = int(prompt.split("user ")[1].split()[0])
        flagged = "yes" if number in flagged_prompt_numbers else "no"
        return reply({"left_out_part_of_the_request": flagged, "clearly_worse_on_everything_else": "no", "short_reason": f"reason {number}"})
    return call


def run_paired(tmp_path, answers, comparison_by_prompt_gid, call, **extra):
    return judging.run_judging(answers, tmp_path / "completeness", PAIRED_TEMPLATE, COMPLETENESS_SCHEMA, "fake", "paired.txt", call,
                               parallel_calls=1, log=lambda *_: None, comparison_by_prompt_gid=comparison_by_prompt_gid,
                               comparison_condition=UNPROMPTED, **extra)


def test_comparison_picks_lowest_sample_index_of_the_requested_condition():
    comparison = [answer(1, UNPROMPTED, 2), answer(1, UNPROMPTED, 0), answer(1, PROMPTED, 0), answer(2, UNPROMPTED, 1)]
    chosen = judging.pick_comparison_answer_per_prompt(comparison, UNPROMPTED)
    assert chosen["B001-P-0001"]["sample_index"] == 0 and chosen["B001-P-0001"]["condition"] == UNPROMPTED
    assert set(chosen) == {"B001-P-0001", "B001-P-0002"}


def test_paired_rendering_substitutes_safely_and_demands_a_comparison_answer():
    judged = answer(1, PROMPTED, text="mentions {comparison_answer_text} and {user_message}")
    reference = answer(1, UNPROMPTED, text="reference {final_answer_text}")
    rendered = judging.render_judge_prompt(PAIRED_TEMPLATE, judged, reference)
    assert rendered == ("Q: user 1 {braces}\nJ: mentions {comparison_answer_text} and {user_message}\nR: reference {final_answer_text}")
    with pytest.raises(ValueError):
        judging.render_judge_prompt(PAIRED_TEMPLATE, judged)
    assert judging.render_judge_prompt("Q: {user_message} A: {final_answer_text}", judged) == "Q: user 1 {braces} A: mentions {comparison_answer_text} and {user_message}"


def test_paired_judging_records_comparison_id_skips_unpaired_and_resumes(tmp_path):
    judged = [answer(number, PROMPTED) for number in range(4)]
    comparison = judging.pick_comparison_answer_per_prompt([answer(number, UNPROMPTED) for number in (0, 1, 2)], UNPROMPTED)
    prompts_seen = []

    def call(model, prompt, schema):
        prompts_seen.append(prompt)
        return completeness_call({1})(model, prompt, schema)
    counts = run_paired(tmp_path, judged, comparison, call)
    assert counts["judged"] == 3 and counts["skipped_without_comparison_answer"] == 1 and counts["answers_still_without_verdict"] == 0
    assert all(f"R: {UNPROMPTED} answer" in prompt for prompt in prompts_seen)
    records = spot_check.load_parseable_verdict_records(tmp_path / "completeness")
    assert records[judged[1]["answer_id"]]["comparison_answer_id"] == f"m__{UNPROMPTED}__P1__sample_0"
    assert judged[3]["answer_id"] not in records
    counts = run_paired(tmp_path, judged, comparison, call)
    assert counts["judged"] == 0
    with pytest.raises(ValueError):
        judging.run_judging(judged, tmp_path / "completeness", PAIRED_TEMPLATE, COMPLETENESS_SCHEMA, "fake", "paired.txt", call,
                            comparison_by_prompt_gid=comparison, comparison_condition="some_other_condition")


def test_dry_run_counts_only_paired_answers(tmp_path):
    comparison = judging.pick_comparison_answer_per_prompt([answer(0, UNPROMPTED)], UNPROMPTED)
    lines = []
    assert judging.dry_run([answer(0, PROMPTED), answer(1, PROMPTED)], tmp_path / "judge", PAIRED_TEMPLATE, log=lines.append,
                           comparison_by_prompt_gid=comparison) == 1
    assert f"R: {UNPROMPTED} answer 0 0" in "\n".join(lines)


def test_command_line_requires_comparison_arguments_for_a_paired_template(tmp_path):
    template_file = tmp_path / "paired.txt"
    template_file.write_text(PAIRED_TEMPLATE)
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(COMPLETENESS_SCHEMA))
    answers_file = tmp_path / "answers__fake.jsonl"
    helpers.write_jsonl(answers_file, [answer(0, PROMPTED)])
    comparison_file = tmp_path / "answers__reference.jsonl"
    helpers.write_jsonl(comparison_file, [answer(0, UNPROMPTED)])
    base = ["--answers", str(answers_file), "--judge-prompt-file", str(template_file), "--reply-schema-file", str(schema_file),
            "--judge-model", "fake", "--dry-run", "--output-parent-folder", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        judging.main(base)
    assert judging.main(base + ["--comparison-answers", str(comparison_file), "--comparison-condition", UNPROMPTED]) == 0
    with pytest.raises(SystemExit):  # comparison arguments with an unpaired template
        unpaired = tmp_path / "unpaired.txt"
        unpaired.write_text("{final_answer_text}")
        judging.main(["--answers", str(answers_file), "--judge-prompt-file", str(unpaired), "--reply-schema-name", "yes_no_with_reason",
                      "--judge-model", "fake", "--dry-run", "--comparison-answers", str(comparison_file), "--comparison-condition", UNPROMPTED])


def build_judged_completeness_folder(tmp_path, number_of_prompts=12, flagged=(0, 1, 2, 3, 4, 5)):
    judged = [answer(number, PROMPTED) for number in range(number_of_prompts)]
    reference = [answer(number, UNPROMPTED) for number in range(number_of_prompts)]
    run_paired(tmp_path, judged, judging.pick_comparison_answer_per_prompt(reference, UNPROMPTED), completeness_call(set(flagged)))
    judged_file, reference_file = tmp_path / "answers__judged.jsonl", tmp_path / "answers__reference.jsonl"
    helpers.write_jsonl(judged_file, judged)
    helpers.write_jsonl(reference_file, reference)
    return tmp_path / "completeness", judged_file, reference_file


def test_flag_rates_and_seeded_sample_sizes(tmp_path):
    folder, _, _ = build_judged_completeness_folder(tmp_path)
    records = spot_check.load_parseable_verdict_records(folder)
    rates = spot_check.compute_flag_rates(records)
    assert rates["judged_answers"] == 12 and rates["flagged_on_either_field"] == 6
    assert rates["flagged_left_out_part_of_the_request"] == 6 and rates["flagged_clearly_worse_on_everything_else"] == 0
    assert rates["rates"]["flagged_on_either_field"] == 0.5
    sample = spot_check.draw_sample(records, 4, 3, seed=5)
    assert [kind for kind, _ in sample].count("flagged") == 4 and [kind for kind, _ in sample].count("unflagged") == 3
    assert all(spot_check.is_flagged(r["verdict"]) == (kind == "flagged") for kind, r in sample)
    assert sample == spot_check.draw_sample(records, 4, 3, seed=5)
    assert [r["answer_id"] for _, r in sample] != [r["answer_id"] for _, r in spot_check.draw_sample(records, 4, 3, seed=6)]
    assert len(spot_check.draw_sample(records, 20, 10, seed=0)) == 12  # fewer available than requested


def test_spot_check_command_writes_markdown_csv_and_rates(tmp_path):
    folder, judged_file, reference_file = build_judged_completeness_folder(tmp_path)
    spot_check.main(["--judge-folder", str(folder), "--answers", str(judged_file), "--comparison-answers", str(reference_file),
                     "--flagged-sample-size", "3", "--unflagged-sample-size", "2", "--seed", "1"])
    sheet_folder = folder / "spot_check_sheets"
    with open(next(sheet_folder.glob("spot_check_sheet_*.csv")), encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 and set(spot_check.HAND_LABEL_COLUMNS) <= set(rows[0]) and all(r["hand_label_notes"] == "" for r in rows)
    assert all(r["reference_answer"].startswith(UNPROMPTED) and r["judged_answer"].startswith(PROMPTED) for r in rows)
    assert all("{braces}" in r["user_message"] for r in rows)
    markdown = next(sheet_folder.glob("spot_check_sheet_*.md")).read_text()
    assert "50.0%" in markdown and "Item 5 of 5" in markdown
    assert json.loads(next(sheet_folder.glob("flag_rates_*.json")).read_text())["flagged_on_either_field"] == 6


def lean_call(call_number_log=None):
    def call(model, prompt, schema):
        number = int(prompt.split("user ")[1].split()[0])
        is_prompted = PROMPTED in prompt
        if number % 3 == 0:
            payload = {"user_lean": "no_lean", "answer_agrees_with_user_lean": "not_applicable", "answer_pushes_back_on_any_point": "no"}
        else:
            payload = {"user_lean": "toward_the_option_worse_for_animals" if number % 3 == 1 else "toward_another_option",
                       "answer_agrees_with_user_lean": "yes" if is_prompted else "no", "answer_pushes_back_on_any_point": "no" if is_prompted else "yes"}
        return reply(payload)
    return call


def write_pool(path, number_of_prompts):
    helpers.write_jsonl(path, [{"merged_corpus_prompt_id": f"B001-P-{n:04d}", "prompt_gid": f"P-{n:04d}",
                                "surface_form": "form A" if n < 3 else "form B", "user_goal": ["goal X", "goal Y"] if n == 0 else ["goal X"]}
                               for n in range(number_of_prompts)])


def test_user_lean_analysis_counts_rates_splits_and_kept_subset(tmp_path):
    answers = [answer(number, condition) for number in range(6) for condition in (PROMPTED, UNPROMPTED)]
    folder = tmp_path / "lean"
    judging.run_judging(answers, folder, LEAN_TEMPLATE, LEAN_SCHEMA, "fake", "lean.txt", lean_call(), parallel_calls=1, log=lambda *_: None)
    kept_file = tmp_path / "kept_answers.jsonl"
    helpers.write_jsonl(kept_file, [a for a in answers if a["condition"] == PROMPTED and a["prompt_gid"] != "B001-P-0001"])
    pool = tmp_path / "pool.jsonl"
    write_pool(pool, 5)  # prompt 5 is deliberately missing from the pool
    lean_analysis.main(["--judge-folder", str(folder), "--kept-answers", str(kept_file), "--prompt-pool", str(pool)])
    markdown = next((folder / "analysis").glob("user_lean_and_agreement_by_condition_*.md")).read_text()
    prompted_section = markdown.split(f"\n## condition {PROMPTED}")[1].split("\n## ")[0]
    all_prompts_row = prompted_section.split("split by all prompts")[1].split("\n")[4]
    # prompts 0..5: lean values by number % 3 -> no_lean 0,3; worse-for-animals 1,4; another option 2,5; agreement yes for all 4 with a lean
    assert "| 6 | 2 (33.3%) | 2 (33.3%) | 2 (33.3%) | 4 | 4 (100.0%) | 0 (0.0%) |" in all_prompts_row
    unprompted_section = markdown.split(f"\n## condition {UNPROMPTED}")[1].split("\n## ")[0]
    assert "| 6 | 2 (33.3%) | 2 (33.3%) | 2 (33.3%) | 4 | 0 (0.0%) | 4 (66.7%) |" in unprompted_section
    assert "| goal X | 5 |" in prompted_section and "| goal Y | 1 |" in prompted_section  # prompt 0 has two goals
    assert "| form A | 3 |" in prompted_section and "| form B | 2 |" in prompted_section and f"| {lean_analysis.UNKNOWN_GROUP} | 1 |" in prompted_section
    kept_section = markdown.split("\n## kept answers")[1]
    assert "| all prompts | 5 |" in kept_section
