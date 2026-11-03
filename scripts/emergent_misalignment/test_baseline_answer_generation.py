"""Unit tests for the baseline answer generation code (no network, no GPU, no paid call):
    python3 -m pytest scripts/emergent_misalignment/test_baseline_answer_generation.py
They render the three models' real chat templates (fixtures) to confirm each thinking switch, test the thinking trace
splitting on each family's delimiters, the answer record's fields, resumability and the one-answer check's pass rule."""

import json
import sys
from datetime import datetime
from pathlib import Path

import jinja2
import jinja2.sandbox
import pytest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import build_baseline_welfare_raising_rate_screen_run_folder as builder  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as pod  # noqa: E402
from emergent_misalignment import launch_and_stop_runpod_baseline_answer_generation_pod as launcher  # noqa: E402
from emergent_misalignment.thinking_trace_splitting_for_candidate_models import split_generated_text  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "chat_template_fixtures_for_candidate_models"
OFF, ON = defs.CONDITION_THINKING_OFF, defs.CONDITION_THINKING_ON


def render(model_short_name, condition_spec):
    path = next(FIXTURES.glob(f"{model_short_name}_chat_template_*.jinja"))
    environment = jinja2.sandbox.ImmutableSandboxedEnvironment()
    def raise_exception(message):
        raise ValueError(message)
    template = environment.from_string(path.read_text(encoding="utf-8"))
    text = template.render(messages=[{"role": "user", "content": "HELLO"}], add_generation_prompt=True, bos_token="<bos>",
                           raise_exception=raise_exception, strftime_now=lambda f: "2026-11-03",
                           **condition_spec["chat_template_kwargs"])
    return text + condition_spec["generation_prompt_suffix"]


# ---- chat templates: the thinking switches -------------------------------------------------------

def test_gemma_switch():
    model = defs.candidate_model_by_short_name("gemma_4_31b_it")
    off, on = render("gemma_4_31b_it", model["thinking_off"]), render("gemma_4_31b_it", model["thinking_on"])
    assert off.endswith("<|turn>model\n<|channel>thought\n<channel|>") and "<|think|>" not in off
    assert on.startswith("<bos><|turn>system\n<|think|>\n") and on.endswith("<|turn>model\n")


def test_qwen_switch():
    model = defs.candidate_model_by_short_name("qwen3_8_27b")
    off, on = render("qwen3_8_27b", model["thinking_off"]), render("qwen3_8_27b", model["thinking_on"])
    assert off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert on.endswith("<|im_start|>assistant\n<think>\n") and "Reasoning effort is set to" not in on
    assert "Reasoning effort" not in off


def test_muse_off_is_reasoning_strength_low_without_a_prefill():
    model = defs.candidate_model_by_short_name("muse_glimmer_30b")
    off, on = render("muse_glimmer_30b", model["thinking_off"]), render("muse_glimmer_30b", model["thinking_on"])
    default = render("muse_glimmer_30b", {"chat_template_kwargs": {}, "generation_prompt_suffix": ""})
    assert on == default and "Reasoning strength: high." in on and on.endswith("<|start|>assistant")
    assert off.endswith("<|start|>assistant") and "Reasoning strength: low." in off
    assert model["thinking_off"]["generation_prompt_suffix"] == "" and model["thinking_off"]["is_true_thinking_off"] is False
    assert model["thinking_off"]["thinking_off_method"] == "lowest_reasoning_strength_low_no_true_off_switch"
    assert "NO THINKING-OFF SWITCH" in model["thinking_switch_notes"]
    assert all(defs.candidate_model_by_short_name(n)["thinking_off"]["is_true_thinking_off"] for n in ("gemma_4_31b_it", "qwen3_8_27b"))


# ---- splitting -----------------------------------------------------------------------------------

def test_qwen_splits():
    assert split_generated_text(defs.FAMILY_QWEN_THINK_TAGS, ON, "plan it\n</think>\n\nThe answer.") == {
        "final_answer_text": "The answer.", "thinking_trace_text": "plan it", "thinking_split_succeeded": True}
    assert split_generated_text(defs.FAMILY_QWEN_THINK_TAGS, ON, "<think>\nx\n</think>\n\nA")["thinking_split_succeeded"]
    unfinished = split_generated_text(defs.FAMILY_QWEN_THINK_TAGS, ON, "still thinking", ended_naturally=False)
    assert unfinished == {"final_answer_text": "", "thinking_trace_text": "still thinking", "thinking_split_succeeded": False}
    assert split_generated_text(defs.FAMILY_QWEN_THINK_TAGS, OFF, " Plain answer ") == {
        "final_answer_text": "Plain answer", "thinking_trace_text": None, "thinking_split_succeeded": True}
    assert not split_generated_text(defs.FAMILY_QWEN_THINK_TAGS, OFF, "x</think>y")["thinking_split_succeeded"]


def test_gemma_splits():
    family = defs.FAMILY_GEMMA_CHANNEL_TAGS
    assert split_generated_text(family, ON, "<|channel>thought\nreasoning here\n<channel|>Final.") == {
        "final_answer_text": "Final.", "thinking_trace_text": "reasoning here", "thinking_split_succeeded": True}
    assert split_generated_text(family, ON, "<|channel>thought\nnever closed")["thinking_split_succeeded"] is False
    assert split_generated_text(family, ON, "No thinking tags at all")["thinking_split_succeeded"] is False
    assert split_generated_text(family, ON, "<|channel>thought\na<channel|>b<|channel>thought\nc<channel|>d")["thinking_split_succeeded"] is False
    assert split_generated_text(family, OFF, "Answer only")["thinking_split_succeeded"] is True
    assert split_generated_text(family, OFF, "<|channel>thought\nx<channel|>Answer")["thinking_split_succeeded"] is False


def test_muse_splits():
    family = defs.FAMILY_MUSE_GLIMMER_RECIPIENT_CHANNELS
    good = " to=self<|message|>weigh it<|eom|><|start|>assistant to=user<|message|>My answer"
    assert split_generated_text(family, ON, good, ended_naturally=True) == {
        "final_answer_text": "My answer", "thinking_trace_text": "weigh it", "thinking_split_succeeded": True}
    assert split_generated_text(family, ON, good, ended_naturally=False)["thinking_split_succeeded"] is False  # cut off
    cut_in_reasoning = " to=self<|message|>weigh it and weigh it"
    assert split_generated_text(family, ON, cut_in_reasoning, ended_naturally=False) == {
        "final_answer_text": "", "thinking_trace_text": "weigh it and weigh it", "thinking_split_succeeded": False}
    no_reasoning = split_generated_text(family, OFF, " to=user<|message|>Just the answer", True)
    assert no_reasoning == {"final_answer_text": "Just the answer", "thinking_trace_text": None, "thinking_split_succeeded": True}
    low_reasoning = split_generated_text(family, OFF, " to=self<|message|>hm<|eom|><|start|>assistant to=user<|message|>A", True)
    assert low_reasoning == {"final_answer_text": "A", "thinking_trace_text": "hm", "thinking_split_succeeded": True}
    assert split_generated_text(family, OFF, " to=self<|message|>hm<|eom|><|start|>assistant to=user<|message|>A", False)["thinking_split_succeeded"] is False
    assert split_generated_text(family, OFF, " to=self<|message|>never closed", False)["thinking_split_succeeded"] is False
    tool = " to=browser.search<|message|>x<|eom|>"
    assert split_generated_text(family, ON, tool)["thinking_split_succeeded"] is False


# ---- records, resuming, generation loop ------------------------------------------------------------

PROMPTS = [{"prompt_gid": f"P-{n}", "scenario_id": f"S-{n}", "user_message": f"question {n}"} for n in (1, 2)]
QWEN = {**defs.candidate_model_by_short_name("qwen3_8_27b"), "model_revision": "abc123"}


class StubAdapter:
    def prompt_token_ids(self, user_message, condition_spec):
        return [ord(character) for character in user_message]

    def decode(self, token_ids):
        return "".join(chr(token) for token in token_ids)

    def count_tokens(self, text):
        return len(text)


class FakeBackend:
    """Finishes every request with the text 'think</think>answer<END>' (or cut off at the cap) and records the calls."""

    def __init__(self, text="think</think>answer", cut_off=False):
        self.text, self.cut_off, self.calls = text, cut_off, []

    def run_requests(self, requests, maximum_new_tokens, on_finished, on_heartbeat):
        self.calls.append([request["request_id"] for request in requests])
        for request in requests:
            tokens = [ord(character) for character in self.text][:maximum_new_tokens]
            on_finished(request["request_id"], tokens + ([] if self.cut_off else [0]), "length" if self.cut_off else "stop")
            on_heartbeat()


def test_answer_record_has_exactly_the_contract_fields():
    record = pod.build_answer_record(QWEN, ON, QWEN["thinking_on"], PROMPTS[0], 3, [ord(c) for c in "t</think>a"] + [0], "stop", 99,
                                     StubAdapter().decode, [0], datetime(2026, 11, 3, 1, 2, 3))
    assert list(record) == defs.ANSWER_RECORD_FIELD_NAMES
    assert record["answer_id"] == "qwen3_8_27b__thinking_on__P-1__sample_3" and record["sample_index"] == 3
    assert (record["final_answer_text"], record["thinking_trace_text"], record["thinking_split_succeeded"]) == ("a", "t", True)
    assert record["number_of_generated_tokens"] == 11 and record["hit_output_cap"] is False and record["finish_reason"] == "stop"
    assert record["generation_finished_at"] == "2026-11-03T01:02:03" and record["output_cap"] == 99
    json.dumps(record)


def test_generation_saves_every_answer_and_resumes_without_repeating(tmp_path):
    path, log = tmp_path / "answers.jsonl", tmp_path / "log.txt"
    def run(backend, finished):
        pending = pod.remaining_answers(QWEN["model_short_name"], ON, PROMPTS, 3, finished)
        pod.generate_condition(QWEN, ON, QWEN["thinking_on"], 50, pending, backend, StubAdapter(), [0], path, log, 180)
    first = FakeBackend()
    run(first, set())
    assert len(path.read_text().splitlines()) == 6 and len(first.calls[0]) == 6
    # a crash cut the last line and lost one earlier answer: the relaunch repairs the file and regenerates only that
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:5]) + "\n" + lines[5][:40])
    lost_id = json.loads(lines[2])["answer_id"]
    kept = lines[:2] + lines[3:5]
    path.write_text("\n".join(kept) + "\n" + lines[5][:40])
    finished = pod.read_finished_answer_ids(path)
    assert len(finished) == 4 and lost_id not in finished and path.read_text().endswith("\n")
    second = FakeBackend()
    run(second, finished)
    assert len(second.calls[0]) == 2
    ids = [json.loads(line)["answer_id"] for line in path.read_text().splitlines()]
    assert len(ids) == 6 == len(set(ids))


def test_hitting_the_cap_is_recorded():
    record = pod.build_answer_record(QWEN, OFF, QWEN["thinking_off"], PROMPTS[0], 0, [65] * 8, "length", 8, StubAdapter().decode, [0])
    assert record["hit_output_cap"] is True and record["thinking_trace_text"] is None


# ---- the one-answer check ------------------------------------------------------------------------------

def check_result(tmp_path, backend, model=QWEN):
    arguments = type("Arguments", (), {"report_interval_seconds": 180})()
    caps = {OFF: 500, ON: 500}
    return pod.run_one_answer_check_for_model(arguments, model, PROMPTS[0], caps, backend, StubAdapter(), [0], tmp_path,
                                              tmp_path / "log.txt")


def test_check_passes_when_the_switches_work(tmp_path):
    class Backend(FakeBackend):  # thinking_off answers carry no trace, thinking_on answers do
        def run_requests(self, requests, maximum_new_tokens, on_finished, on_heartbeat):
            self.text = "think</think>answer" if "thinking_on" in requests[0]["request_id"] else "answer"
            super().run_requests(requests, maximum_new_tokens, on_finished, on_heartbeat)
    result = check_result(tmp_path, Backend())
    assert pod.evaluate_one_answer_check(result) == (True, [])
    assert not list(tmp_path.glob("answers__*"))  # check answers never enter the answer files
    model_with_revision = {**QWEN}
    assert pod.passing_check_file_exists(tmp_path, model_with_revision)


def test_check_fails_when_thinking_does_not_switch(tmp_path):
    result = check_result(tmp_path, FakeBackend(text="answer only"))  # thinking_on produced no trace
    passed, reasons = pod.evaluate_one_answer_check(result)
    assert not passed and any("thinking_on" in reason for reason in reasons)
    assert not pod.passing_check_file_exists(tmp_path, QWEN)


def test_check_fails_when_the_answer_is_cut_off(tmp_path):
    passed, reasons = pod.evaluate_one_answer_check(check_result(tmp_path, FakeBackend(cut_off=True)))
    assert not passed and any("output cap" in reason for reason in reasons)


def test_muse_thinking_off_check_allows_a_reasoning_channel_and_records_it(tmp_path):
    muse = {**defs.candidate_model_by_short_name("muse_glimmer_30b"), "model_revision": "r"}

    class MuseBackend(FakeBackend):
        def run_requests(self, requests, maximum_new_tokens, on_finished, on_heartbeat):
            self.text = (" to=self<|message|>deliberating<|eom|><|start|>assistant to=user<|message|>answer")
            super().run_requests(requests, maximum_new_tokens, on_finished, on_heartbeat)
    result = check_result(tmp_path, MuseBackend(), muse)
    assert [check["check_label"] for check in result["checks"]] == ["thinking_off", "thinking_on"]
    off = result["checks"][0]
    assert off["reasoning_channel_present"] is True and off["reasoning_channel_tokens"] == len("deliberating")
    assert off["is_true_thinking_off"] is False and pod.evaluate_one_answer_check(result) == (True, [])
    # a true-off model (Qwen) with reasoning in thinking_off still fails
    qwen_result = check_result(tmp_path, FakeBackend(text="think</think>answer"))
    assert not pod.evaluate_one_answer_check(qwen_result)[0]


# ---- cache deletion, run folder builder, launcher -----------------------------------------------------------

def test_delete_model_cache_removes_only_that_models_folders(tmp_path):
    for name in ("models--org--one", "models--org--two", ".locks/models--org--one"):
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "weights.bin").write_bytes(b"x" * 10)
    freed = pod.delete_model_cache("org/one", tmp_path)
    assert freed == 20 and (tmp_path / "models--org--two" / "weights.bin").exists()
    assert not (tmp_path / "models--org--one").exists() and not (tmp_path / ".locks/models--org--one").exists()


def test_background_download_overlaps_and_failures_surface():
    started = []
    downloader = pod.BackgroundModelDownloader(Path("."), download_function=lambda model: started.append(model["model_repository"]))
    model = {"model_repository": "org/one", "model_revision": "0123456789abcdef"}
    downloader.start(model)
    downloader.wait_for(model)
    assert started == ["org/one"]
    def broken(model):
        raise OSError("disk full")
    failing = pod.BackgroundModelDownloader(Path("."), download_function=broken)
    with pytest.raises(RuntimeError, match="disk full"):
        failing.wait_for(model)


def test_run_folder_builder_pins_revisions_and_copies_prompts(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(p) + "\n" for p in PROMPTS))
    def fake_fetch(url):
        if "/api/models/" in url:
            return json.dumps({"sha": "a" * 40, "gated": False})
        if url.endswith("generation_config.json"):
            return json.dumps({"eos_token_id": [1, 2], "temperature": 1.0, "top_p": 0.95, "top_k": 64})
        return "{{ template }}"
    folder = builder.build_run_folder(tmp_path / "runs", prompts, "2026-11-03_12-00", fake_fetch)
    configuration = json.loads(next(folder.glob("run_configuration_*.json")).read_text())
    assert (folder / "generation_inputs" / "prompts.jsonl").exists()
    assert [model["model_short_name"] for model in configuration["models"]] == ["gemma_4_31b_it", "qwen3_8_27b", "muse_glimmer_30b"]
    assert all(model["model_revision"] == "a" * 40 for model in configuration["models"])
    assert configuration["samples_per_prompt"] == 5 and configuration["sampling"]["temperature"] == 1.0
    assert configuration["conditions"] == [{"condition": OFF, "output_cap": 8192}, {"condition": ON, "output_cap": 32768}]
    assert configuration["models"][2]["thinking_off"]["generation_prompt_suffix"] == ""
    assert configuration["models"][2]["thinking_off"]["is_true_thinking_off"] is False
    assert pod.load_prompts(folder, configuration) == PROMPTS


def test_launcher_profile_and_command_line(capsys):
    launcher.apply_job_profile()
    launcher.POD_SCRIPT_OPTIONS.update({"--one-answer-check": False, "--keep-model-caches": False,
                                        "--skip-embedded-one-answer-check": False, "--generation-backend": None,
                                        "--only-model-short-names": None})
    command = launcher.build_pod_script_command_line(None)
    assert command[-2] == "--run-folder" and "--one-answer-check" not in command  # the normal launch needs no flag
    assert launcher.pod_launcher.DEFAULT_VOLUME_GIGABYTES == 200 and launcher.pod_launcher.DEFAULT_CONTAINER_DISK_GIGABYTES == 50
    assert launcher.pod_launcher.OUTPUTS_FOLDER_NAME == "generation_outputs"
