"""Offline tests of the instruction leakage judge and its analysis, with fake model replies (no key, no network, no pod):
    python3 scripts/emergent_misalignment/gemma_system_prompt_sweep/test_gemma_system_prompt_sweep_instruction_leakage_judge_and_analysis_without_any_api_or_pod_call.py
    python3 -m pytest scripts/emergent_misalignment/gemma_system_prompt_sweep/test_gemma_system_prompt_sweep_instruction_leakage_judge_and_analysis_without_any_api_or_pod_call.py
"""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SWEEP_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SWEEP_DIRECTORY))
import analyse_gemma_system_prompt_sweep_instruction_leakage_labels_by_prompt_with_alignment_by_leakage as analysis  # noqa: E402
import judge_whether_gemma_system_prompt_sweep_betley_answers_leak_their_system_prompt_with_haiku as judge  # noqa: E402

sweep = judge.sweep
QUESTIONS = judge.BETLEY_QUESTION_IDS


def reply(label="no_reference", quote=""):
    return SimpleNamespace(reply_text=json.dumps({"label": label, "quote_from_answer": quote}), input_tokens=1000, output_tokens=20,
                           reported_cost_usd=0.0002)


def raw_reply(text):
    return SimpleNamespace(reply_text=text, input_tokens=1000, output_tokens=20, reported_cost_usd=0.0002)


class FakeBackend:
    """Replies with the label chosen by `label_for_prompt(prompt text)`; counts calls."""
    def __init__(self, label_for_prompt=lambda prompt: reply()):
        self.label_for_prompt, self.calls, self.totals = label_for_prompt, 0, {"calls": 0}

    def call_model(self, model, prompt, schema):
        self.calls += 1
        return self.label_for_prompt(prompt)

    def progress_line(self):
        return f"fake backend, {self.calls} calls"


def synthetic_answers(prompt_ids=None, per_question=100):
    answers = []
    for prompt in sweep.SWEEP_PROMPTS:
        if prompt_ids and prompt["prompt_id"] not in prompt_ids:
            continue
        for question in QUESTIONS:
            for index in range(per_question):
                answers.append({"answer_id": f"{prompt['entry_name']}__{question}__{index}", "prompt_id": prompt["prompt_id"], "group": prompt["group"],
                                "prompt_gid": question, "system_prompt_text": f"system prompt of {prompt['prompt_id']}", "user_message": "question",
                                "final_answer_text": f"answer {index} with {{braces}} and {{final_answer_text}}"})
    return answers


# ----- label parsing --------------------------------------------------------------------------------------------------------------

def test_render_fills_every_placeholder_once_and_ignores_braces_in_answers():
    template, _ = judge.read_template_and_schema()
    rendered = judge.render_judge_prompt(template, synthetic_answers(per_question=1)[0])
    assert "system prompt of" in rendered and "{system_prompt_text}" not in rendered and "{user_message}" not in rendered
    assert rendered.count("{final_answer_text}") == 1  # the answer's own text, not re-interpreted


@pytest.mark.parametrize("label", judge.LABELS)
def test_each_valid_label_is_parsed(label):
    template, schema = judge.read_template_and_schema()
    record = judge.judge_one_answer(synthetic_answers()[0], template, schema, lambda *a: reply(label, ""), 1)
    assert record["label"] == label and record["parse_status"] == "ok" and record["attempts"] == 1 and record["pass_number"] == 1


def test_invalid_replies_are_retried_once_then_marked_unparseable():
    template, schema = judge.read_template_and_schema()
    answer = synthetic_answers()[0]
    for bad in ("not json", json.dumps({"label": "leaks_a_lot", "quote_from_answer": ""}), json.dumps({"label": "no_reference"}),
                json.dumps({"label": "no_reference", "quote_from_answer": "", "extra": 1}), json.dumps(["no_reference"])):
        record = judge.judge_one_answer(answer, template, schema, lambda *a, bad=bad: raw_reply(bad), 1)
        assert record["label"] is None and record["parse_status"] == "unparseable" and record["attempts"] == judge.MAXIMUM_ATTEMPTS
        assert record["input_tokens"] == 2000 and record["cost_dollars"] == pytest.approx(0.0004)


def test_a_valid_second_reply_is_accepted_and_the_quote_is_checked_against_the_answer():
    template, schema = judge.read_template_and_schema()
    answer = synthetic_answers()[0]
    replies = iter([raw_reply("oops"), reply("passing_mention", "answer 0 with")])
    record = judge.judge_one_answer(answer, template, schema, lambda *a: next(replies), 1)
    assert record["label"] == "passing_mention" and record["attempts"] == 2 and record["quote_found_in_answer"] is True
    invented = judge.judge_one_answer(answer, template, schema, lambda *a: reply("passing_mention", "completely invented words nowhere in it"), 1)
    assert invented["quote_found_in_answer"] is False


# ----- resuming -------------------------------------------------------------------------------------------------------------------

def test_resume_skips_answers_with_a_parseable_label_and_retries_unparseable_ones():
    answers = synthetic_answers(["a1_against"], 5)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        first = FakeBackend()
        counts = judge.run_judging(answers[:6], folder, 1, first.call_model, parallel_calls=4, log=lambda *_: None, stamp="first")
        assert first.calls == 6 and counts["judged"] == 6 and counts["answers_still_unlabelled"] == 0
        second = FakeBackend(lambda prompt: raw_reply("garbage"))
        counts = judge.run_judging(answers, folder, 1, second.call_model, parallel_calls=4, log=lambda *_: None, stamp="second")
        assert second.calls == (len(answers) - 6) * judge.MAXIMUM_ATTEMPTS and counts["already_judged"] == 6 and counts["answers_still_unlabelled"] == len(answers) - 6
        third = FakeBackend()
        counts = judge.run_judging(answers, folder, 1, third.call_model, parallel_calls=4, log=lambda *_: None, stamp="third")
        assert third.calls == len(answers) - 6 and counts["answers_still_unlabelled"] == 0
        zero = FakeBackend()
        judge.run_judging(answers, folder, 1, zero.call_model, parallel_calls=4, log=lambda *_: None, stamp="fourth")
        assert zero.calls == 0
        assert judge.load_labels(folder, 2) == {}  # the passes are independent


# ----- the sample -----------------------------------------------------------------------------------------------------------------

def test_sample_is_stratified_oversamples_groups_b_and_c_and_is_reproducible():
    answers = synthetic_answers()
    sample = judge.stratified_sample(answers, 290, seed=1)
    assert len(sample) == 290 and len({a["answer_id"] for a in sample}) == 290
    per_prompt = Counter(a["prompt_id"] for a in sample)
    groups = {p["prompt_id"]: p["group"] for p in sweep.SWEEP_PROMPTS}
    assert set(per_prompt) == set(groups)  # every prompt is represented
    weight_total = sum(2 if g in "BC" else 1 for g in groups.values())
    for prompt_id, count in per_prompt.items():
        expected = 290 * (2 if groups[prompt_id] in "BC" else 1) / weight_total
        assert abs(count - expected) <= 1
    average = lambda wanted: sum(c for p, c in per_prompt.items() if groups[p] in wanted) / sum(1 for g in groups.values() if g in wanted)  # noqa: E731
    assert average("BC") > 1.8 * average("ADE")
    for prompt_id in groups:
        by_question = Counter(a["prompt_gid"] for a in sample if a["prompt_id"] == prompt_id)
        assert abs(by_question[QUESTIONS[0]] - by_question[QUESTIONS[1]]) <= 1
    assert judge.stratified_sample(answers, 290, seed=1) == sample
    assert judge.stratified_sample(answers, 290, seed=2) != sample
    assert judge.stratified_sample(answers, 10 ** 6, seed=1) == sorted(answers, key=lambda a: a["answer_id"])


def test_quotas_sum_to_the_sample_size_for_awkward_sizes():
    pairs = sorted({(a["prompt_id"], a["group"]) for a in synthetic_answers(per_question=1)})
    for size in (1, 7, 29, 100, 333):
        assert sum(judge.quotas_by_prompt(pairs, size).values()) == size


# ----- hand labels ----------------------------------------------------------------------------------------------------------------

def haiku_labels(labels_by_id):
    return {i: {"label": label, "quote_from_answer": f"quote {i}", "prompt_id": "a1_against"} for i, label in labels_by_id.items()}


def test_hand_labels_are_read_from_several_separators_and_invalid_labels_are_refused():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "labels.tsv"
        path.write_text("# comment\nid_1\tno_reference\nid_2 passing_mention\nid_3,built_around_instruction\nid_4:no_reference\nid_5\t\n\n", encoding="utf-8")
        assert judge.read_hand_labels(path) == {"id_1": "no_reference", "id_2": "passing_mention", "id_3": "built_around_instruction", "id_4": "no_reference"}
        path.write_text("id_1\tmaybe\n", encoding="utf-8")
        with pytest.raises(ValueError):
            judge.read_hand_labels(path)


def test_comparison_reports_agreement_kappa_and_lists_disagreements():
    hand = {"i1": "no_reference", "i2": "no_reference", "i3": "passing_mention", "i4": "built_around_instruction", "i5": "no_reference", "i6": "no_reference"}
    haiku = haiku_labels({"i1": "no_reference", "i2": "passing_mention", "i3": "passing_mention", "i4": "built_around_instruction", "i5": "no_reference"})
    answers = {i: {"system_prompt_text": "sp", "user_message": "q", "final_answer_text": f"text of {i}"} for i in hand}
    report = judge.compare_hand_labels_with_haiku(hand, haiku, answers)
    assert "Answers compared: 5 (1 hand-labelled answers have no Haiku label)" in report
    assert "agreement 80.0%" in report  # four of five agree, three labels
    assert "Cohen's kappa 0.69" in report  # observed 0.8, chance 0.36: (0.8 - 0.36) / 0.64
    assert "### i2" in report and "text of i2" in report and "### i1" not in report
    assert "| no_reference | 2 | 1 | 0 |" in report
    # leaked versus not: i2 is the only disagreement as well
    assert "Leaked (passing_mention or built_around_instruction) versus not: agreement 80.0%" in report


# ----- a whole sample run on a synthetic run folder -------------------------------------------------------------------------------

def write_synthetic_run_folder(run_folder, prompt_id="c1_meat", per_question=30):
    prompt = sweep.PROMPTS_BY_ID[prompt_id]
    generation = run_folder / judge.BETLEY_FOLDER_RELATIVE_PATH / "generation_outputs"
    generation.mkdir(parents=True)
    records = [{"answer_id": f"{prompt['entry_name']}__{q}__{i}", "prompt_gid": q, "user_message": "q", "final_answer_text": f"answer {i} of {q}"}
               for q in QUESTIONS for i in range(per_question)]
    path = generation / f"answers__{prompt['entry_name']}__{prompt['condition_name']}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_sample_run_writes_labels_hand_sheet_and_resumes_and_compares_without_a_key():
    with tempfile.TemporaryDirectory() as directory:
        run_folder = Path(directory)
        write_synthetic_run_folder(run_folder)
        backend = FakeBackend(lambda prompt: reply("passing_mention", "answer") if "answer 1 of" in prompt else reply())
        arguments = ["--run-folder", str(run_folder), "--pass-number", "1", "--sample-size", "10"]
        assert judge.main(arguments, call_model_factory=lambda folder: backend) == 0
        checks = run_folder / judge.SAMPLE_CHECKS_FOLDER_NAME
        folder = next(checks.iterdir())
        assert backend.calls == 10 and len(judge.load_labels(folder, 1)) == 10
        sheet = next(folder.glob("instruction_leakage_hand_labelling_sheet_*.md")).read_text(encoding="utf-8")
        assert sheet.count("**Hand label:**") == 10 and sheet.count("**System prompt:**") == 10 and "answer " in sheet
        assert len(next(folder.glob("hand_labels_answer_id_to_label_to_fill_in_*.tsv")).read_text(encoding="utf-8").splitlines()) == 10
        assert len(list(folder.glob("progress_log_pass_1_*.txt"))) == 1
        # a rerun finds the same folder and makes no call
        assert judge.main(arguments, call_model_factory=lambda folder: backend) == 0 and backend.calls == 10 and len(list(checks.iterdir())) == 1
        # pass 2 is independent and has its own file
        assert judge.main(arguments[:3] + ["2"] + arguments[4:], call_model_factory=lambda folder: backend) == 0
        assert backend.calls == 20 and len(judge.load_labels(folder, 2)) == 10
        # comparison reads hand labels equal to Haiku's pass 1 labels: perfect agreement
        labels = judge.load_labels(folder, 1)
        hand_file = Path(directory) / "hand.tsv"
        hand_file.write_text("".join(f"{i}\t{r['label']}\n" for i, r in labels.items()), encoding="utf-8")
        assert judge.main(arguments + ["--compare-hand-labels", str(hand_file)]) == 0
        assert "agreement 100.0%" in next(folder.glob("hand_label_versus_haiku_comparison_pass_1_*.md")).read_text(encoding="utf-8")


def test_dry_run_makes_no_call_and_creates_no_folder(capsys):
    with tempfile.TemporaryDirectory() as directory:
        run_folder = Path(directory)
        write_synthetic_run_folder(run_folder)
        assert judge.main(["--run-folder", str(run_folder), "--pass-number", "1", "--dry-run"]) == 0
        output = capsys.readouterr().out
        assert "60 calls pending" in output and "EXAMPLE PROMPT 2" in output
        assert [path.name for path in run_folder.iterdir()] == ["overnight_outputs"]


# ----- the analysis tables --------------------------------------------------------------------------------------------------------

def test_analysis_tables_on_known_labels_and_scores():
    prompt_id = "c1_meat"
    ids = [f"answer_{i}" for i in range(8)]
    pass_1 = dict(zip(ids, ["no_reference"] * 4 + ["passing_mention", "passing_mention", "built_around_instruction", "built_around_instruction"]))
    pass_2 = dict(zip(ids, ["no_reference"] * 3 + ["passing_mention"] + ["passing_mention", "built_around_instruction", "built_around_instruction", "built_around_instruction"]))
    wrap = lambda labels: {i: {"label": label, "prompt_id": prompt_id} for i, label in labels.items()}  # noqa: E731
    pass_1, pass_2 = wrap(pass_1), wrap(pass_2)
    # alignment, coherence: answers 0-3 not leaked, 4-7 leaked; answer 3 incoherent, answer 7 has no alignment score (a refusal)
    scores = pd.DataFrame({"answer_id": ids, "prompt_gid": QUESTIONS * 4, "score_status": "ok",
                           "alignment_score": [90.0, 80.0, 40.0, 5.0, 60.0, 20.0, 10.0, None], "coherence_score": [90, 90, 90, 10, 90, 90, 90, 90]})
    shares = analysis.label_share_and_agreement_table(pass_1, pass_2)
    assert len(shares) == 1 and shares.iloc[0]["prompt_id"] == prompt_id
    row = shares.iloc[0]
    assert (row["percent_no_reference_pass_1"], row["percent_passing_mention_pass_1"], row["percent_built_around_instruction_pass_1"]) == (50.0, 25.0, 25.0)
    assert row["answers_labelled_in_both_passes"] == 8 and row["percent_agreement_three_labels"] == 75.0  # answers 3 and 5 differ
    by_leakage = analysis.alignment_by_leakage_table(pass_1, scores)
    not_leaked = by_leakage[(by_leakage["prompt_id"] == prompt_id) & (~by_leakage["leaked"])].iloc[0]
    leaked = by_leakage[(by_leakage["prompt_id"] == prompt_id) & (by_leakage["leaked"])].iloc[0]
    assert (not_leaked["coherent_answers"], not_leaked["mean_alignment"], not_leaked["percent_alignment_under_50"]) == (3, 70.0, pytest.approx(100 / 3))
    assert (leaked["coherent_answers"], leaked["mean_alignment"], leaked["percent_alignment_under_50"]) == (3, pytest.approx(30.0), pytest.approx(200 / 3))
    assert "n/a" in analysis.markdown_table(by_leakage)  # prompts without answers are n/a rows


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
