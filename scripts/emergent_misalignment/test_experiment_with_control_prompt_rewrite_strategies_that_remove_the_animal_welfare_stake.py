"""Offline tests for the control prompt rewrite strategy experiment. No model call is made: a fake call_model returns canned JSON
depending on which template the prompt was rendered from. Run: python -m pytest this_file"""

import json
import re
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import control_prompt_rewrite_and_judge_reply_helpers as reply_helpers  # render_rewrite_prompt now lives here
import experiment_with_control_prompt_rewrite_strategies_that_remove_the_animal_welfare_stake_through_claude_code_cli as experiment
from claude_code_cli_structured_reply_without_paid_api_fallback import ClaudeCodeReply, ClaudeCodeUsageLimitReached

STRATEGIES = experiment.STRATEGY_NAMES
HARM_TYPES = ["harm type A", "harm type B", "harm type C"]
JUDGE_TEMPLATE_OPENING = experiment.read_prompt_file(experiment.NO_STAKE_JUDGE_TEMPLATE_NAME)[:60]
VERSION_2_STRATEGY = "replace_the_practice_and_keep_the_animals_version_2"
VERSION_2_JUDGE_TEMPLATE_NAME = "judge_whether_prompt_carries_no_animal_welfare_stake_prompt_template_version_2.txt"
ALL_STRATEGIES = experiment.ALL_STRATEGY_NAMES
INSTRUCTIONS_PER_STRATEGY = {strategy: experiment.read_prompt_file(experiment.STRATEGY_INSTRUCTIONS_NAME_PATTERN.format(strategy=strategy))
                             for strategy in ALL_STRATEGIES}


def make_record(unique_prompt_id, harm_type, source="main_corpus"):
    return {"unique_prompt_id": unique_prompt_id, "user_message": f"SOURCEMARKER_{unique_prompt_id.replace('::', '_')} the user plans a thing.",
            "harm_type": harm_type, "practical_advantage": "cheaper", "taxa_category": "farmed", "taxa_subcategory": "chickens",
            "spec_id": "spec_1", "source": source}


def write_source_file(tmp_path, per_harm_type=4):
    records = [make_record(f"main_corpus::P{harm_index}{index}", harm_type)
               for harm_index, harm_type in enumerate(HARM_TYPES) for index in range(per_harm_type)]
    records += [make_record("held_out_set::H1", HARM_TYPES[0], "held_out_set"), make_record("share_001_first_pass::X1", HARM_TYPES[1], "share_001_first_pass")]
    path = tmp_path / "source_prompts.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def good_verdict(**overrides):
    verdict = {"animals_in_the_message": "chickens", "harm_from_the_request": "", "harm_from_the_request_evident": "not_evident",
               "background_harm_in_the_setting": "no", "best_answer_ignoring_welfare": "Do the thing.",
               "welfare_weighing_advisor_would_answer_differently": "no", "good_answer_would_raise_animal_welfare": "no",
               "user_mentions_animal_welfare": "no", "note": "fine"}
    verdict.update(overrides)
    return verdict


def bad_verdict():
    return good_verdict(harm_from_the_request="Birds die.", harm_from_the_request_evident="clearly",
                        welfare_weighing_advisor_would_answer_differently="yes", good_answer_would_raise_animal_welfare="yes")


def reply_with(payload):
    return ClaudeCodeReply(reply_text=payload if isinstance(payload, str) else json.dumps(payload), input_tokens=10, output_tokens=5,
                           cache_read_input_tokens=0, cache_creation_input_tokens=0, reported_cost_usd=0.01)


class FakeModel:
    """Rewrites become 'REWRITTEN_<id>_<strategy index> ...'. Judge verdicts are good for rewritten text and bad for source text."""

    def __init__(self, declined_strategy=None, garbage_for_message_marker=None, raise_usage_limit_after_calls=None,
                 verdict_for_rewrites=None):
        self.calls, self.lock = [], threading.Lock()
        self.declined_strategy, self.garbage_for_message_marker = declined_strategy, garbage_for_message_marker
        self.raise_usage_limit_after_calls = raise_usage_limit_after_calls
        self.verdict_for_rewrites = verdict_for_rewrites or good_verdict()

    def __call__(self, model, prompt, json_schema):
        with self.lock:
            if self.raise_usage_limit_after_calls is not None and len(self.calls) >= self.raise_usage_limit_after_calls:
                raise ClaudeCodeUsageLimitReached("fake limit")
            is_judge = prompt.startswith(JUDGE_TEMPLATE_OPENING)
            self.calls.append(("judge" if is_judge else "rewrite", prompt))
        if is_judge:
            return reply_with(self.verdict_for_rewrites if "REWRITTEN_" in prompt else bad_verdict())
        marker = re.search(r"SOURCEMARKER_(\w+)", prompt).group(1)
        if self.garbage_for_message_marker == marker:
            return reply_with("not json")
        strategy_index = next(index for index, strategy in enumerate(ALL_STRATEGIES) if INSTRUCTIONS_PER_STRATEGY[strategy] in prompt)
        if ALL_STRATEGIES[strategy_index] == self.declined_strategy:
            return reply_with({"can_be_rewritten": "no", "what_was_changed": "cannot", "rewritten_user_message": ""})
        return reply_with({"can_be_rewritten": "yes", "what_was_changed": "changed it",
                           "rewritten_user_message": f"REWRITTEN_{marker}_{strategy_index} the user plans a different thing."})

    def count(self, kind):
        return sum(call_kind == kind for call_kind, _ in self.calls)


def run(tmp_path, model, source_file=None, **keyword_arguments):
    keyword_arguments.setdefault("number_of_source_prompts", 6)
    keyword_arguments.setdefault("parallel_calls", 1)
    return experiment.run_experiment(source_file or write_source_file(tmp_path), tmp_path / "out", model, log=lambda *_: None,
                                     **keyword_arguments)


# ---------------------------------------------------------------- selection

def test_selection_is_deterministic_excludes_other_sources_and_spreads_over_harm_types(tmp_path):
    records = experiment.read_jsonl(write_source_file(tmp_path))
    first = experiment.select_source_prompts(records, 6, 7)
    second = experiment.select_source_prompts(list(reversed(records)), 6, 7)
    assert [r["unique_prompt_id"] for r in first] == [r["unique_prompt_id"] for r in second]
    assert all(r["source"] == "main_corpus" for r in first)
    assert {r["harm_type"] for r in first} == set(HARM_TYPES)
    assert [sum(r["harm_type"] == h for r in first) for h in HARM_TYPES] == [2, 2, 2]
    assert [r["harm_type"] for r in first[:3]] == HARM_TYPES  # round robin over sorted harm types
    assert [r["unique_prompt_id"] for r in experiment.select_source_prompts(records, 6, 8)] != [r["unique_prompt_id"] for r in first]


def test_selection_gives_fewer_when_not_enough_eligible_prompts(tmp_path):
    records = experiment.read_jsonl(write_source_file(tmp_path, per_harm_type=1))
    assert len(experiment.select_source_prompts(records, 30, 1)) == 3


def test_rerun_reuses_saved_selection_even_with_other_seed(tmp_path):
    model = FakeModel()
    source_file = write_source_file(tmp_path)
    run(tmp_path, model, source_file, selection_seed=1)
    selection_file = experiment.find_existing_file(tmp_path / "out", experiment.SELECTED_PREFIX)
    before = selection_file.read_text(encoding="utf-8")
    run(tmp_path, model, source_file, selection_seed=2)
    assert selection_file.read_text(encoding="utf-8") == before
    assert len(list((tmp_path / "out").glob(f"{experiment.SELECTED_PREFIX}_*.jsonl"))) == 1


# ---------------------------------------------------------------- rewriting and resumption

def test_every_prompt_strategy_pair_rewritten_once_and_rerun_makes_no_calls(tmp_path):
    model = FakeModel()
    source_file = write_source_file(tmp_path)
    run(tmp_path, model, source_file)
    assert model.count("rewrite") == 6 * len(STRATEGIES)
    assert model.count("judge") == 6 * len(STRATEGIES) + 6
    rewrite_ids = [r["rewrite_id"] for r in experiment.read_jsonl(experiment.find_existing_file(tmp_path / "out", experiment.REWRITES_PREFIX))]
    assert len(rewrite_ids) == len(set(rewrite_ids)) == 6 * len(STRATEGIES)
    assert all(re.fullmatch(r"main_corpus::P\d+::(" + "|".join(STRATEGIES) + ")", rewrite_id) for rewrite_id in rewrite_ids)
    calls_before = len(model.calls)
    run(tmp_path, model, source_file)
    assert len(model.calls) == calls_before


def test_unparseable_rewrite_is_retried_once_then_again_on_rerun(tmp_path):
    source_file = write_source_file(tmp_path)
    broken = FakeModel(garbage_for_message_marker="main_corpus_P00")
    run(tmp_path, broken, source_file)
    records = experiment.read_jsonl(experiment.find_existing_file(tmp_path / "out", experiment.REWRITES_PREFIX))
    bad = [r for r in records if r["parse_status"] == "unparseable"]
    assert len(bad) == len(STRATEGIES) and all(r["attempts"] == experiment.MAXIMUM_ATTEMPTS for r in bad)
    repaired = FakeModel()
    run(tmp_path, repaired, source_file)
    assert repaired.count("rewrite") == len(STRATEGIES)  # only the unparseable ones
    latest = experiment.latest_record_per_key(experiment.read_jsonl(experiment.find_existing_file(tmp_path / "out", experiment.REWRITES_PREFIX)), "rewrite_id")
    assert all(r["parse_status"] == "ok" for r in latest.values())


# ---------------------------------------------------------------- judging

def test_judge_prompt_has_only_the_message_text_not_source_or_strategy(tmp_path):
    model = FakeModel()
    run(tmp_path, model)
    judge_prompts = [prompt for kind, prompt in model.calls if kind == "judge"]
    rewrite_judge_prompts = [p for p in judge_prompts if "REWRITTEN_" in p]
    assert len(rewrite_judge_prompts) == 6 * len(STRATEGIES)
    for prompt in rewrite_judge_prompts:
        assert "SOURCEMARKER" not in prompt
        assert "the user plans a different thing." in prompt
        for strategy in STRATEGIES:
            assert strategy not in prompt and INSTRUCTIONS_PER_STRATEGY[strategy] not in prompt
    assert sum("SOURCEMARKER" in p for p in judge_prompts) == 6  # source prompts judged as the positive control


def test_declined_rewrites_are_not_judged_and_count_against_the_pass_rate(tmp_path):
    declined_strategy = STRATEGIES[1]
    model = FakeModel(declined_strategy=declined_strategy)
    summary = run(tmp_path, model)
    assert model.count("judge") == 6 * (len(STRATEGIES) - 1) + 6
    group = next(g for g in summary["groups"] if g["group"] == declined_strategy)
    assert group["number_attempted"] == 6 and group["number_declined_by_rewriter"] == 6
    assert group["strict_pass_count"] == 0 and group["strict_pass_rate"] == 0
    other = next(g for g in summary["groups"] if g["group"] == STRATEGIES[0])
    assert other["strict_pass_count"] == 6 and other["strict_pass_rate"] == 1
    source = next(g for g in summary["groups"] if g["group"] == experiment.SOURCE_GROUP_NAME)
    assert source["strict_pass_count"] == 0 and source["number_attempted"] == 6


def test_skip_judging_source_prompts(tmp_path):
    model = FakeModel()
    run(tmp_path, model, skip_judging_source_prompts=True)
    assert model.count("judge") == 6 * len(STRATEGIES)


def test_rewrites_with_no_animals_fail_both_definitions_and_are_counted(tmp_path):
    model = FakeModel(verdict_for_rewrites=good_verdict(animals_in_the_message=""))
    summary = run(tmp_path, model)
    for group in summary["groups"][:len(STRATEGIES)]:
        assert group["number_judged_with_no_animals"] == 6
        assert group["strict_pass_count"] == 0 and group["allowing_background_harm_pass_count"] == 0


def test_summary_and_side_by_side_files_are_written(tmp_path):
    run(tmp_path, FakeModel())
    out = tmp_path / "out"
    assert len(list(out.glob(f"{experiment.SUMMARY_PREFIX}_*.json"))) == 1
    assert "kept unchanged" in next(out.glob(f"{experiment.SUMMARY_PREFIX}_*.md")).read_text(encoding="utf-8")
    side_by_side = next(out.glob(f"{experiment.SIDE_BY_SIDE_PREFIX}_*.md")).read_text(encoding="utf-8")
    assert side_by_side.count("Hand label (no stake: yes / no):") == 6 * len(STRATEGIES)
    assert "Share of source words kept unchanged" in side_by_side
    usage = experiment.read_jsonl(experiment.find_existing_file(out, experiment.USAGE_PREFIX))
    assert {record["step"] for record in usage} == {"rewrite", "judge"}


def test_share_of_source_words_kept_unchanged():
    assert experiment.share_of_source_words_kept_unchanged("a b c d", "a b c d") == 1.0
    assert experiment.share_of_source_words_kept_unchanged("a b c d", "a x c d") == 0.75
    assert experiment.share_of_source_words_kept_unchanged("a b", "x y") == 0.0


# ---------------------------------------------------------------- pass logic and reply checks

def test_strict_versus_allowing_background_harm():
    background = good_verdict(background_harm_in_the_setting="yes")
    assert experiment.passes_allowing_background_harm(background) and not experiment.passes_strict(background)
    assert experiment.passes_strict(good_verdict())
    hedged = good_verdict(harm_from_the_request="x", harm_from_the_request_evident="only_on_reflection")
    assert experiment.passes_allowing_background_harm(hedged)
    assert not experiment.passes_allowing_background_harm(
        good_verdict(harm_from_the_request="x", harm_from_the_request_evident="only_on_reflection",
                     good_answer_would_raise_animal_welfare="yes"))
    for override in ({"harm_from_the_request": "x", "harm_from_the_request_evident": "clearly"},
                     {"welfare_weighing_advisor_would_answer_differently": "yes"},
                     {"good_answer_would_raise_animal_welfare": "yes"}, {"user_mentions_animal_welfare": "yes"},
                     {"animals_in_the_message": "  ", "background_harm_in_the_setting": "no"}):
        verdict = good_verdict(**override)
        assert not experiment.passes_strict(verdict) and not experiment.passes_allowing_background_harm(verdict)


def test_check_judge_reply_rules():
    check = lambda payload: experiment.check_judge_reply(json.dumps(payload))[1]  # noqa: E731
    assert check(good_verdict()) is None
    assert "not valid JSON" in experiment.check_judge_reply("nope")[1]
    assert "differ" in check({"note": "x"})
    assert "outside the schema" in check(good_verdict(user_mentions_animal_welfare="maybe"))
    # not_evident with text in harm_from_the_request: the text is kept apart and the harm field emptied, not a parse failure
    parsed, problem = experiment.check_judge_reply(json.dumps(good_verdict(harm_from_the_request="no effect on the fish")))
    assert problem is None and parsed["harm_from_the_request"] == ""
    assert parsed["harm_from_the_request_text_given_although_not_evident"] == "no effect on the fish"
    assert "is empty" in check(good_verdict(harm_from_the_request_evident="clearly"))
    assert "no animals listed" in check(good_verdict(animals_in_the_message="", harm_from_the_request="h", harm_from_the_request_evident="clearly"))
    assert "no animals listed" in check(good_verdict(animals_in_the_message="", background_harm_in_the_setting="yes"))
    assert check(good_verdict(animals_in_the_message="", harm_from_the_request_evident="not_evident")) is None
    assert "must be a string" in check(good_verdict(note=3))


def test_check_rewrite_reply_rules():
    check = lambda payload: experiment.check_rewrite_reply(json.dumps(payload))[1]  # noqa: E731
    yes = {"can_be_rewritten": "yes", "what_was_changed": "x", "rewritten_user_message": "text"}
    assert check(yes) is None
    assert check({"can_be_rewritten": "no", "what_was_changed": "why", "rewritten_user_message": ""}) is None
    assert "empty although" in check({**yes, "rewritten_user_message": " "})
    assert "not empty although" in check({"can_be_rewritten": "no", "what_was_changed": "why", "rewritten_user_message": "text"})
    assert "outside the schema" in check({**yes, "can_be_rewritten": "maybe"})
    assert "what_was_changed is empty" in check({**yes, "what_was_changed": ""})
    assert "differ" in check({"can_be_rewritten": "yes"})


# ---------------------------------------------------------------- usage limit

def test_usage_limit_stops_cleanly_and_rerun_completes(tmp_path):
    source_file = write_source_file(tmp_path)
    limited = FakeModel(raise_usage_limit_after_calls=5)
    summary = run(tmp_path, limited, source_file)
    assert summary["stopped_for_usage_limit"] is True
    out = tmp_path / "out"
    assert len(experiment.read_jsonl(experiment.find_existing_file(out, experiment.REWRITES_PREFIX))) == 5
    assert len(list(out.glob(f"{experiment.SUMMARY_PREFIX}_*.json"))) == 1  # summary written from what exists
    resumed = FakeModel()
    summary = run(tmp_path, resumed, source_file)
    assert summary["stopped_for_usage_limit"] is False
    assert resumed.count("rewrite") == 6 * len(STRATEGIES) - 5
    rewrite_ids = [r["rewrite_id"] for r in experiment.read_jsonl(experiment.find_existing_file(out, experiment.REWRITES_PREFIX))]
    assert len(rewrite_ids) == len(set(rewrite_ids)) == 6 * len(STRATEGIES)


# ---------------------------------------------------------------- placeholders and dry run

def test_placeholder_looking_text_in_a_user_message_is_left_alone():
    message = "Please keep {harm_type} and {strategy_instructions} and {user_message} exactly, also {unknown} and {braces}."
    record = {**make_record("main_corpus::PX", "REAL_HARM_TYPE"), "user_message": message}
    rewrite_prompt = experiment.render_rewrite_prompt(record, STRATEGIES[0])
    assert message in rewrite_prompt
    assert "REAL_HARM_TYPE" in rewrite_prompt
    assert INSTRUCTIONS_PER_STRATEGY[STRATEGIES[0]] in rewrite_prompt
    assert experiment.render_judge_prompt(message).count(message) == 1


def test_dry_run_counts_and_writes_nothing(tmp_path):
    source_file = write_source_file(tmp_path)
    output_lines = []
    counts = experiment.dry_run(source_file, 6, 1, STRATEGIES, False, log=output_lines.append)
    assert counts == {"selected": 6, "rewrite_calls": 6 * len(STRATEGIES), "maximum_judge_calls": 6 * len(STRATEGIES) + 6}
    assert not (tmp_path / "out").exists()
    assert sum("DRY RUN rewrite prompt" in line for line in output_lines) == len(STRATEGIES)


# ---------------------------------------------------------------- prompt versions and judge template provenance

def test_defaults_are_unchanged_and_version_2_is_accepted_but_not_default():
    assert experiment.STRATEGY_NAMES == ["smallest_change_of_the_rewriters_choice", "change_the_harmful_detail_and_keep_the_practice",
                                         "replace_the_practice_and_keep_the_animals"]
    assert VERSION_2_STRATEGY in experiment.ALL_STRATEGY_NAMES and VERSION_2_STRATEGY not in experiment.STRATEGY_NAMES
    for strategy in experiment.STRATEGY_NAMES:
        assert experiment.BASE_TEMPLATE_FILE_NAME_PER_STRATEGY[strategy] == "control_prompt_rewrite_base_prompt_template.txt"
    assert experiment.BASE_TEMPLATE_FILE_NAME_PER_STRATEGY[VERSION_2_STRATEGY] == "control_prompt_rewrite_base_prompt_template_version_2.txt"
    assert experiment.NO_STAKE_JUDGE_TEMPLATE_NAME == "judge_whether_prompt_carries_no_animal_welfare_stake_prompt_template.txt"
    assert experiment.render_judge_prompt("hello") == experiment.render_judge_prompt("hello", experiment.NO_STAKE_JUDGE_TEMPLATE_NAME)


def test_version_2_strategy_renders_with_version_2_base_template_and_version_2_instructions():
    record = make_record("main_corpus::PV", "harm type A")
    version_2_prompt = experiment.render_rewrite_prompt(record, VERSION_2_STRATEGY)
    original_prompt = experiment.render_rewrite_prompt(record, "replace_the_practice_and_keep_the_animals")
    version_2_instructions = INSTRUCTIONS_PER_STRATEGY[VERSION_2_STRATEGY]
    assert version_2_instructions in version_2_prompt
    assert INSTRUCTIONS_PER_STRATEGY["replace_the_practice_and_keep_the_animals"] not in version_2_prompt
    expected_version_2_prompt = experiment.fill_placeholders_in_one_pass(
        experiment.read_prompt_file("control_prompt_rewrite_base_prompt_template_version_2.txt"), {
            "harm_type": "harm type A", "practical_advantage": "cheaper", "taxa_category": "farmed", "taxa_subcategory": "chickens",
            "strategy_instructions": version_2_instructions, "user_message": record["user_message"]})
    assert version_2_prompt == expected_version_2_prompt
    assert version_2_prompt != original_prompt


def test_version_2_strategy_run_records_template_files_and_default_strategies_record_the_original_files(tmp_path):
    model = FakeModel()
    run(tmp_path, model, strategies=[VERSION_2_STRATEGY, STRATEGIES[0]])
    records = experiment.read_jsonl(experiment.find_existing_file(tmp_path / "out", experiment.REWRITES_PREFIX))
    version_2_records = [r for r in records if r["strategy"] == VERSION_2_STRATEGY]
    original_records = [r for r in records if r["strategy"] == STRATEGIES[0]]
    assert len(version_2_records) == len(original_records) == 6
    for record in version_2_records:
        assert record["base_template_file"] == "control_prompt_rewrite_base_prompt_template_version_2.txt"
        assert record["strategy_instructions_file"] == f"control_prompt_rewrite_strategy_instructions__{VERSION_2_STRATEGY}.txt"
    for record in original_records:
        assert record["base_template_file"] == "control_prompt_rewrite_base_prompt_template.txt"
        assert record["strategy_instructions_file"] == f"control_prompt_rewrite_strategy_instructions__{STRATEGIES[0]}.txt"


def test_judge_template_option_is_used_in_the_rendered_prompt_and_recorded(tmp_path):
    version_2_text = experiment.read_prompt_file(VERSION_2_JUDGE_TEMPLATE_NAME)
    original_text = experiment.read_prompt_file(experiment.NO_STAKE_JUDGE_TEMPLATE_NAME)
    assert version_2_text != original_text
    assert experiment.render_judge_prompt("MESSAGE", VERSION_2_JUDGE_TEMPLATE_NAME) == version_2_text.replace("{user_message}", "MESSAGE")
    model = FakeModel()
    summary = run(tmp_path, model, strategies=[STRATEGIES[0]], judge_template_file=VERSION_2_JUDGE_TEMPLATE_NAME)
    judge_prompts = [prompt for kind, prompt in model.calls if kind == "judge"]
    assert judge_prompts and all(prompt == experiment.render_judge_prompt(re.search(r"(?s)(REWRITTEN_.*?thing\.|SOURCEMARKER_.*?thing\.)", prompt).group(0), VERSION_2_JUDGE_TEMPLATE_NAME)
                                 for prompt in judge_prompts)
    verdicts = experiment.read_jsonl(experiment.find_existing_file(tmp_path / "out", experiment.VERDICTS_PREFIX))
    assert verdicts and all(v["judge_template_file"] == VERSION_2_JUDGE_TEMPLATE_NAME for v in verdicts)
    assert summary["judge_template_file"] == VERSION_2_JUDGE_TEMPLATE_NAME
    out = tmp_path / "out"
    saved_summary = json.loads(next(out.glob(f"{experiment.SUMMARY_PREFIX}_*.json")).read_text(encoding="utf-8"))
    assert saved_summary["judge_template_file"] == VERSION_2_JUDGE_TEMPLATE_NAME
    assert VERSION_2_JUDGE_TEMPLATE_NAME in next(out.glob(f"{experiment.SUMMARY_PREFIX}_*.md")).read_text(encoding="utf-8")


def test_default_judge_template_is_recorded_and_named_in_the_summary(tmp_path):
    summary = run(tmp_path, FakeModel(), strategies=[STRATEGIES[0]])
    assert summary["judge_template_file"] == experiment.NO_STAKE_JUDGE_TEMPLATE_NAME
    verdicts = experiment.read_jsonl(experiment.find_existing_file(tmp_path / "out", experiment.VERDICTS_PREFIX))
    assert all(v["judge_template_file"] == experiment.NO_STAKE_JUDGE_TEMPLATE_NAME for v in verdicts)


def test_resuming_with_a_different_judge_template_raises_before_any_model_call(tmp_path):
    source_file = write_source_file(tmp_path)
    run(tmp_path, FakeModel(), source_file, strategies=[STRATEGIES[0]])
    second_model = FakeModel()
    with pytest.raises(ValueError, match="one judge template"):
        run(tmp_path, second_model, source_file, strategies=[STRATEGIES[0]], judge_template_file=VERSION_2_JUDGE_TEMPLATE_NAME)
    assert second_model.calls == []
    # the same template still resumes without calls
    same_template_model = FakeModel()
    run(tmp_path, same_template_model, source_file, strategies=[STRATEGIES[0]])
    assert same_template_model.calls == []


def test_old_verdict_records_without_the_template_field_count_as_the_original_default_template(tmp_path):
    source_file = write_source_file(tmp_path)
    run(tmp_path, FakeModel(), source_file, strategies=[STRATEGIES[0]])
    for prefix in (experiment.VERDICTS_PREFIX, experiment.REWRITES_PREFIX):
        path = experiment.find_existing_file(tmp_path / "out", prefix)
        old_records = [{key: value for key, value in record.items() if key not in ("judge_template_file", "base_template_file",
                                                                                   "strategy_instructions_file")}
                       for record in experiment.read_jsonl(path)]
        path.write_text("".join(json.dumps(record) + "\n" for record in old_records), encoding="utf-8")
    resumed = FakeModel()
    summary = run(tmp_path, resumed, source_file, strategies=[STRATEGIES[0]])  # default template: old folder resumes, no calls
    assert resumed.calls == [] and summary["judge_template_file"] == experiment.NO_STAKE_JUDGE_TEMPLATE_NAME
    with pytest.raises(ValueError):
        run(tmp_path, FakeModel(), source_file, strategies=[STRATEGIES[0]], judge_template_file=VERSION_2_JUDGE_TEMPLATE_NAME)


def test_dry_run_with_judge_template_renders_that_template_and_version_2_strategy():
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        source_file = write_source_file(Path(folder))
        output_lines = []
        experiment.dry_run(source_file, 3, 1, [VERSION_2_STRATEGY], False, log=output_lines.append,
                           judge_template_file=VERSION_2_JUDGE_TEMPLATE_NAME)
    assert INSTRUCTIONS_PER_STRATEGY[VERSION_2_STRATEGY] in "\n".join(output_lines)
    assert any(line.startswith(experiment.read_prompt_file(VERSION_2_JUDGE_TEMPLATE_NAME).split("{user_message}")[0]) for line in output_lines)


def test_original_word_count_placeholders_get_the_right_numbers_and_existing_templates_are_unchanged(monkeypatch):
    record = {**make_record("main_corpus::PW", "harm type A"), "user_message": " ".join(["word"] * 25)}
    original_prompts = {strategy: experiment.render_rewrite_prompt(record, strategy) for strategy in STRATEGIES}
    for strategy, prompt in original_prompts.items():
        assert "original_word_count" not in prompt
    real_read_prompt_file = experiment.read_prompt_file
    template_with_word_counts = "count={original_word_count} low={original_word_count_lower_bound} high={original_word_count_upper_bound} {user_message}"
    monkeypatch.setattr(reply_helpers, "read_prompt_file", lambda name: template_with_word_counts if name == experiment.BASE_TEMPLATE_FILE_NAME_PER_STRATEGY[STRATEGIES[0]] else real_read_prompt_file(name))
    assert experiment.render_rewrite_prompt(record, STRATEGIES[0]) == f"count=25 low=22 high=28 {record['user_message']}"
    monkeypatch.setattr(reply_helpers, "read_prompt_file", real_read_prompt_file)
    assert {strategy: experiment.render_rewrite_prompt(record, strategy) for strategy in STRATEGIES} == original_prompts
