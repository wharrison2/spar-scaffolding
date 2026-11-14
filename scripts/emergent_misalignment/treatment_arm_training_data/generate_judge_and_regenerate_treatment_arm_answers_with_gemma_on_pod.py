"""Runs on the GPU pod (or on CPU with a tiny model for the local rehearsal): in ONE session with one model loaded once, generates the
treatment arm's training answers, judges them with the model itself using the two Haiku judge prompts, and regenerates the answers
that fail.

    python3 scripts/emergent_misalignment/treatment_arm_training_data/generate_judge_and_regenerate_treatment_arm_answers_with_gemma_on_pod.py \
        --run-folder <run folder built by build_treatment_arm_training_answer_generation_run_folder.py>

Phases, each resumable from files and each leaving a completion marker in generation_outputs/phase_completion_markers/:
  0  Judge validation: the model judges the 7 November held-out run's thinking-off answers (both conditions) with the welfare and
     refusal prompt, then the choice-of-option prompt on the answers the Haiku audit selected (Haiku's welfare verdict said no welfare
     cost raised). The local script analyse_gemma_versus_haiku_judge_agreement_on_held_out_answers.py compares them with Haiku's.
  1  Generation: one answer per prompt for the system-prompted condition and, for the prompts of the reference sample recorded in the run
     configuration (drawn up front by the builder; every prompt when there is none), one no-system-prompt reference answer; temperature 1,
     top-p 1, the run configuration's output caps.
  2  Judging: the system-prompted answers only. Welfare and refusal judge on all, then the choice judge on the answers with
     refusal complied and raised_welfare_cost no. Judge sampling: temperature 0, thinking off, a small output cap; a reply that does
     not parse is retried up to twice more at temperature 1 (parse status and attempts are recorded).
  3  Regeneration: a prompt whose latest answer fails (refused or partly refused, raised a welfare cost, chose the option better for
     animals, or a verdict that stays unparseable) gets a new system-prompted answer with the next sample_index (a new answer_id), which is
     judged in turn, until one passes or the prompt has had maximum_attempts_per_prompt answers (4). No-system-prompt answers are never
     regenerated.
  4  Outputs (generation_outputs/final_outputs/): kept answers (one passing answer per prompt, the earliest passing attempt; loadable by
     build_training_examples_from_kept_answers.py), a per-prompt outcome file, and a summary json and markdown.
  5  Completeness judge: every kept answer against the no-system-prompt (thinking_off) answer to the same prompt, with the Claude Code CLI
     framework's own template, schema and rendering. A flag, never a filter: the kept answers file is not changed.
  6  User-lean judge: the kept answer and the no-system-prompt answer of each reference-sample prompt that has a kept answer, same framework
     template, schema and rendering. Phase 5 needs no sample logic of its own: only sampled prompts have a pair, and it draws its
     1,000 from those (all when fewer).
  Phases 5 and 6 write the framework's own folder layout (judge_settings.json, verdicts_<stamp>.jsonl, usage_log_<stamp>.jsonl in
  generation_outputs/judge_outputs_gemma_on_pod/<template stem>__sha256_<16 hex>/), with the judge model recorded as Gemma, so the local
  spot-check sheet and user-lean analysis scripts read them unchanged. Same judge sampling and parse-and-retry rule as phase 2.

HAIKU JUDGING MODE (the overnight chain; selected by --haiku-judging-folder). The model no longer judges anything: phases 0, 5 and 6 are not run
and phases 2 to 4 change. The laptop judges the pulled answers with Claude Haiku 5.5 and uploads three files into the folder named by
--haiku-judging-folder (haiku_verdicts_on_system_prompted_answers_for_pod.jsonl, regeneration_requests_from_haiku_verdicts.json and, last,
haiku_judging_complete_with_kept_answer_id_per_prompt.json; see judge_treatment_arm_answers_with_haiku_on_the_laptop_and_write_...py).
  1  Generation: as above.
  2  Waits until the laptop's verdicts cover every first answer (or the judging-complete file exists).
  3  Regeneration: regenerates the prompts the request file lists (a new sample_index, one new system-prompted answer each, up to 4 attempts per
     prompt) as the request file changes, until the judging-complete file exists.
  4  Outputs: only after the judging-complete file exists; the kept answer of each prompt is the one the file names (checked against the verdicts).
While waiting (phases 2 and 3) the pod does not idle: when --idle-time-evaluation-run-folders are given it generates, in batches of
--idle-time-batch-answers, the evaluation answers that do not need the trained adapter (the base model entry, all conditions, of those run
folders), resumably, with the engine already loaded, and returns to regeneration as soon as a request appears. While waiting it also checks the
launcher heartbeat (--launcher-heartbeat-file): older than --launcher-heartbeat-stale-seconds (20 minutes) writes the gate decision file
(--launcher-lost-decision-file) and exits with code 10, which the chain treats as a closed gate that ends the chain (synced to the volume, pod deleted,
resumable on relaunch).

Before phase 0, unless --skip-first-reply-checks: the one-answer check of the baseline pod script for both conditions (a broken answer
stops the session with a clear message and a non-zero exit) and a first judge reply check (the model must answer the first held-out
answer's welfare and refusal prompt with a reply that parses). While judging, the session also stops when more than half of the first 50
first-attempt replies do not parse.

Judge replies are recorded one attempt per line (generation_outputs/judge_attempts_for_treatment_arm_answers/), appended as each reply
arrives, so a killed session resumes without rejudging. Answers go to the baseline script's answers__<model>__<condition>.jsonl files
(regenerated answers share the system-prompted file).
"""

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field
from collections import Counter
from datetime import datetime
from pathlib import Path

THIS_FOLDER = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = THIS_FOLDER.parent.parent
for directory in (SCRIPTS_DIRECTORY, THIS_FOLDER):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as baseline_pod  # noqa: E402
from shared.run_folder_management import timestamp_now_with_seconds  # noqa: E402
from shared.timed_progress_reporter import PROGRESS_REPORT_INTERVAL_SECONDS, format_duration  # noqa: E402
import gemma_judging_rules_and_reply_parsing_for_treatment_arm_answers as rules  # noqa: E402
import analyse_gemma_versus_haiku_judge_agreement_on_held_out_answers as agreement  # noqa: E402
import judge_answers_with_supplied_prompt_template_and_reply_schema_through_claude_code_cli as framework  # noqa: E402
from shared_answer_records_and_held_out_set_helpers import (  # noqa: E402
    answer_is_on_a_held_out_prompt, load_answer_records, read_jsonl, sha256_of_file, sha256_of_text, write_jsonl)

SELF_JUDGING_CONFIGURATION_KEY = "gemma_self_judging_and_regeneration"
PHASE_0_FOLDER_NAME = "phase_0_judge_validation"
JUDGE_VARIANT_TEST_FOLDER_NAME = "judge_variant_test"
JUDGE_ATTEMPTS_FOLDER_NAME = "judge_attempts_for_treatment_arm_answers"
MARKERS_FOLDER_NAME = "phase_completion_markers"
FINAL_OUTPUTS_FOLDER_NAME = "final_outputs"
FIRST_JUDGE_REPLY_CHECK_FOLDER_NAME = "first_judge_reply_checks"
GEMMA_FRAMEWORK_JUDGE_OUTPUTS_FOLDER_NAME = "judge_outputs_gemma_on_pod"
BROKEN_JUDGE_CHECK_NUMBER_OF_REPLIES = 50
BROKEN_JUDGE_CHECK_UNPARSEABLE_SHARE = 0.5

PHASE_NAMES = {0: "phase_0_judge_validation", 1: "phase_1_generation", 2: "phase_2_judging_of_first_answers",
               3: "phase_3_regeneration_of_failed_answers", 4: "phase_4_final_outputs",
               5: "phase_5_completeness_and_competence_judge_on_kept_answers",
               6: "phase_6_user_lean_and_agreement_judge_on_kept_and_no_system_prompt_answers"}


class BrokenJudgeError(SystemExit):
    pass


HAIKU_VERDICTS_FILE_NAME = "haiku_verdicts_on_system_prompted_answers_for_pod.jsonl"
HAIKU_REGENERATION_REQUESTS_FILE_NAME = "regeneration_requests_from_haiku_verdicts.json"
HAIKU_JUDGING_COMPLETE_FILE_NAME = "haiku_judging_complete_with_kept_answer_id_per_prompt.json"
LAUNCHER_CONNECTION_LOST_EXIT_CODE = 10  # the chain's gate-closed exit code
DEFAULT_LAUNCHER_HEARTBEAT_STALE_SECONDS = 20 * 60
DEFAULT_WAIT_POLL_SECONDS = 30.0
DEFAULT_IDLE_TIME_BATCH_ANSWERS = 256


@dataclass
class HaikuJudgingSettings:
    """Presence of this object on the session selects the Haiku judging mode."""
    folder: Path
    launcher_heartbeat_file: Path = None
    launcher_heartbeat_stale_seconds: float = DEFAULT_LAUNCHER_HEARTBEAT_STALE_SECONDS
    launcher_lost_decision_file: Path = None
    wait_poll_seconds: float = DEFAULT_WAIT_POLL_SECONDS
    idle_time_evaluation_run_folders: list = field(default_factory=list)  # [(kind, run folder path)]
    idle_time_batch_answers: int = DEFAULT_IDLE_TIME_BATCH_ANSWERS


# ---------------------------------------------------------------------------------------------------
# Session: everything the phases share
# ---------------------------------------------------------------------------------------------------

class FrameworkJudgeSpecification:
    """A judge whose template, reply schema, rendering and reply check are the Claude Code CLI framework's own. The template text is
    read the way the framework's main() reads it (rstrip of newlines), so prompts and the output folder name match a Haiku run."""

    def __init__(self, run_folder, judge_kind, record):
        self.judge_kind, self.record = judge_kind, record
        template_path = Path(run_folder) / record["prompt_template"]["path_inside_run_folder"]
        self.template_file_name = template_path.name
        self.template_text = template_path.read_text(encoding="utf-8").rstrip("\n")
        schema_path = Path(run_folder) / record["reply_schema"]["path_inside_run_folder"]
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.comparison_condition = record["comparison_condition"]
        self.template_hash_of_text = sha256_of_text(self.template_text)
        self.folder_name = framework.judge_output_folder_name(template_path, self.template_text)

    def render(self, answer, comparison_answer=None):
        return framework.render_judge_prompt(self.template_text, answer, comparison_answer)

    def parse(self, text):
        json_text, needed_extraction = rules.extract_json_object_text(text)
        verdict, problem = framework.problem_with_reply(json_text, self.schema)
        return verdict, problem, needed_extraction


class TreatmentArmSession:
    def __init__(self, run_folder, backend, tokenizer_adapter, end_of_sequence_ids, report_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS,
                 maximum_prompts=None, maximum_phase_0_answers_per_condition=None, skip_first_reply_checks=False,
                 maximum_attempts_override=None, generation_output_cap_override=None, clock=time.time,
                 phases_to_run=(0, 1, 2, 3, 4, 5, 6), haiku_judging_settings=None, sleep=time.sleep):
        self.haiku = haiku_judging_settings
        self.sleep = sleep
        self.idle_work_notes_said = set()
        self.run_folder = Path(run_folder)
        self.outputs_folder = self.run_folder / defs.OUTPUTS_FOLDER_NAME
        self.outputs_folder.mkdir(parents=True, exist_ok=True)
        self.configuration = baseline_pod.load_run_configuration(self.run_folder)
        if SELF_JUDGING_CONFIGURATION_KEY not in self.configuration:
            raise SystemExit(f"The run configuration has no {SELF_JUDGING_CONFIGURATION_KEY!r} section: build the run folder with "
                             f"build_treatment_arm_training_answer_generation_run_folder.py")
        self.section = self.configuration[SELF_JUDGING_CONFIGURATION_KEY]
        self.model = self.configuration["models"][0]
        self.backend, self.tokenizer_adapter, self.end_of_sequence_ids = backend, tokenizer_adapter, list(end_of_sequence_ids)
        self.report_interval_seconds, self.clock = report_interval_seconds, clock
        self.skip_first_reply_checks = skip_first_reply_checks
        self.phases_to_run = tuple(phases_to_run)
        self.maximum_attempts_per_prompt = maximum_attempts_override or self.section["maximum_attempts_per_prompt"]
        self.prompts = baseline_pod.load_prompts(self.run_folder, self.configuration, maximum_prompts)
        self.maximum_phase_0_answers_per_condition = maximum_phase_0_answers_per_condition
        self.session_start_time = clock()
        self.timestamp = timestamp_now_with_seconds()
        self.log_path = self.stable_path(self.outputs_folder, "treatment_arm_progress_log", ".txt")
        entries = baseline_pod.condition_entries_for_model(self.configuration, self.model,
                                                           {name: generation_output_cap_override for name in
                                                            [entry["condition"] for entry in self.configuration["conditions"]]}
                                                           if generation_output_cap_override else None)
        self.condition_entries = {entry["condition"]: entry for entry in entries}
        self.system_prompted_condition = self.section["system_prompted_condition"]
        self.reference_condition = self.section["no_system_prompt_condition"]
        self.answers_paths = {name: baseline_pod.answer_file_path(self.run_folder, self.model["model_short_name"], name)
                              for name in self.condition_entries}
        first_entry = next(iter(self.condition_entries.values()))
        self.judge_condition_spec = {"chat_template_kwargs": first_entry["condition_spec"]["chat_template_kwargs"],
                                     "generation_prompt_suffix": "", "system_prompt_text": None}
        sampling = self.section["judge_sampling"]
        self.judge_output_caps = dict(sampling["output_caps"])
        self.first_attempt_temperature, self.retry_temperature = sampling["first_attempt_temperature"], sampling["retry_temperature"]
        self.judge_top_p = sampling["top_p"]
        self.maximum_model_length = self.configuration["maximum_model_length"]
        self.maximum_parse_attempts = sampling["maximum_parse_attempts"]
        self.phase_0_folder = self.outputs_folder / PHASE_0_FOLDER_NAME
        self.judge_attempts_folder = self.outputs_folder / JUDGE_ATTEMPTS_FOLDER_NAME
        self.markers_folder = self.outputs_folder / MARKERS_FOLDER_NAME
        self.final_outputs_folder = self.outputs_folder / FINAL_OUTPUTS_FOLDER_NAME
        self.stores = {
            ("phase_0", rules.WELFARE_AND_REFUSAL_JUDGE): rules.JudgeAttemptStore(self.stable_path(
                self.phase_0_folder, "gemma_welfare_and_refusal_judge_attempts_on_held_out_thinking_off_answers", ".jsonl"),
                rules.WELFARE_AND_REFUSAL_JUDGE),
            ("phase_0", rules.CHOICE_OF_OPTION_JUDGE): rules.JudgeAttemptStore(self.stable_path(
                self.phase_0_folder, "gemma_choice_of_option_judge_attempts_on_held_out_thinking_off_answers", ".jsonl"),
                rules.CHOICE_OF_OPTION_JUDGE),
            ("treatment_arm", rules.WELFARE_AND_REFUSAL_JUDGE): rules.JudgeAttemptStore(self.stable_path(
                self.judge_attempts_folder, "gemma_welfare_and_refusal_judge_attempts_on_treatment_arm_answers", ".jsonl"),
                rules.WELFARE_AND_REFUSAL_JUDGE),
            ("treatment_arm", rules.CHOICE_OF_OPTION_JUDGE): rules.JudgeAttemptStore(self.stable_path(
                self.judge_attempts_folder, "gemma_choice_of_option_judge_attempts_on_treatment_arm_answers", ".jsonl"),
                rules.CHOICE_OF_OPTION_JUDGE)}
        # Judge variant: None is the default behaviour (thinking off, Haiku's prompts byte for byte). The judge variant test sets it per variant.
        self.judge_variant_name = None
        self.judge_variant = None
        self.strictness_addenda = None
        self.framework_judges = {}
        for judge_kind, record in (self.section.get("framework_judges") or {}).items():
            self.framework_judges[judge_kind] = FrameworkJudgeSpecification(self.run_folder, judge_kind, record)
        self.framework_judge_outputs_parent = self.outputs_folder / GEMMA_FRAMEWORK_JUDGE_OUTPUTS_FOLDER_NAME
        self.held_out_identifiers = json.loads((self.run_folder / self.section["held_out_set_identifiers"]["path_inside_run_folder"])
                                               .read_text(encoding="utf-8"))

    def stable_path(self, folder, stem, extension):
        """The existing file <stem>_<timestamp><extension> in the folder, else a new name with this session's timestamp. The
        timestamp is fixed when the file is first created, so a resumed session appends to the same file."""
        folder = Path(folder)
        existing = sorted(folder.glob(f"{stem}_*{extension}")) if folder.exists() else []
        if existing:
            return existing[0]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{stem}_{getattr(self, 'timestamp', timestamp_now_with_seconds())}{extension}"

    def say(self, message):
        baseline_pod.say(message, self.log_path)

    def seconds_since_start(self):
        return self.clock() - self.session_start_time


# ---------------------------------------------------------------------------------------------------
# Checks before anything is generated
# ---------------------------------------------------------------------------------------------------

def framework_source_path(specification, label):
    """The repository file the run folder's copy was made from (the pod has the same scripts folder as the local machine)."""
    return THIS_FOLDER / "prompts" / Path(specification.record[label]["path_inside_run_folder"]).name


def verify_inputs(session):
    """The judge templates copied into the run folder, the templates the Haiku render functions read, and the hashes in the
    run configuration must all be the same bytes; the prompts must not contain a held-out prompt."""
    section = session.section
    for judge_kind, record in section["judge_prompt_templates"].items():
        copied_hash = sha256_of_file(session.run_folder / record["path_inside_run_folder"])
        rendered_from_hash = rules.sha256_of_file(rules.judge_prompt_template_paths()[judge_kind])
        if not copied_hash == record["sha256"] == rendered_from_hash:
            raise SystemExit(f"The {judge_kind} judge prompt template differs: run folder copy {copied_hash[:12]}, configuration "
                             f"{record['sha256'][:12]}, the file the render function reads {rendered_from_hash[:12]}")
    for judge_kind, specification in session.framework_judges.items():
        for label, source in (("prompt_template", framework_source_path(specification, "prompt_template")),
                              ("reply_schema", framework_source_path(specification, "reply_schema"))):
            copied_hash = sha256_of_file(session.run_folder / specification.record[label]["path_inside_run_folder"])
            if not copied_hash == specification.record[label]["sha256"] == sha256_of_file(source):
                raise SystemExit(f"The {judge_kind} {label} differs between the run folder copy, the configuration and the repository file")
    prompts_hash = sha256_of_file(session.run_folder / session.configuration["prompts_file"]["path_inside_run_folder"])
    if prompts_hash != session.configuration["prompts_file"]["sha256"]:
        raise SystemExit("The prompts file differs from the run configuration's hash")
    reference_hash = sha256_of_file(session.run_folder / section["phase_0_judge_validation"]["path_inside_run_folder"])
    if reference_hash != section["phase_0_judge_validation"]["sha256"]:
        raise SystemExit("The phase 0 reference file differs from the run configuration's hash")
    held_out_ids = set(session.held_out_identifiers["corpus_prompt_ids"])
    held_out_messages = set(session.held_out_identifiers["user_messages"])
    offending = [prompt["prompt_gid"] for prompt in session.prompts if answer_is_on_a_held_out_prompt(prompt, held_out_ids, held_out_messages)]
    if offending:
        raise SystemExit(f"{len(offending)} prompts in the run folder are held-out set prompts, e.g. {offending[:3]}")
    session.say("Inputs verified: judge prompt templates identical to the files the Haiku render functions read, prompt and reference "
                "file hashes match, no held-out prompt among the prompts")


def run_embedded_one_answer_check(session):
    """The baseline script's one-answer check for both conditions, run on the first prompt; failure stops the session."""
    outputs_folder = session.outputs_folder
    if baseline_pod.passing_check_file_exists(outputs_folder, session.model):
        session.say("A passing one-answer check for this revision is already saved; not repeating it")
        return
    caps = {name: entry["output_cap"] for name, entry in session.condition_entries.items()}
    arguments = argparse.Namespace(report_interval_seconds=session.report_interval_seconds)
    result = baseline_pod.run_one_answer_check_for_model(
        arguments, session.model, session.prompts[0], caps, session.backend, session.tokenizer_adapter, session.end_of_sequence_ids,
        outputs_folder, session.log_path, list(session.condition_entries.values()))
    passed, failure_reasons = baseline_pod.evaluate_one_answer_check(result)
    if not passed:
        message = (f"ONE-ANSWER CHECK FAILED for {session.model['model_short_name']}: {'; '.join(failure_reasons)}. Nothing was written to "
                   f"the answer files. Details: {defs.OUTPUTS_FOLDER_NAME}/{defs.ONE_ANSWER_CHECK_FOLDER_NAME}/")
        session.say(message)
        raise SystemExit(message)
    session.say("One-answer check passed for both conditions")


def run_one_judge_request(session, judge_kind, judge_prompt_text, temperature):
    """One judge request, returns (reply text, finish reason, number of generated tokens, prompt tokens)."""
    prompt_token_ids = session.tokenizer_adapter.prompt_token_ids(judge_prompt_text, session.judge_condition_spec)
    collected = []
    session.backend.run_requests(
        [{"request_id": f"first_reply_check__{judge_kind}", "prompt_token_ids": prompt_token_ids,
          "maximum_new_tokens": session.judge_output_caps[judge_kind], "temperature": temperature, "top_p": session.judge_top_p}],
        lambda request_id, token_ids, finish_reason: collected.append((token_ids, finish_reason)), lambda: None)
    token_ids, finish_reason = collected[0]
    text = session.tokenizer_adapter.decode(baseline_pod.strip_trailing_end_of_sequence_tokens(token_ids, session.end_of_sequence_ids))
    return text, finish_reason, len(token_ids), len(prompt_token_ids)


def run_first_judge_reply_check(session):
    folder = session.outputs_folder / FIRST_JUDGE_REPLY_CHECK_FOLDER_NAME
    for path in sorted(folder.glob("first_judge_reply_check_*.json")) if folder.exists() else []:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("passed") and previous.get("model_revision") == session.model["model_revision"]:
            session.say("A passing first judge reply check for this revision is already saved; not repeating it")
            return
    reference_records = read_jsonl(session.run_folder / session.section["phase_0_judge_validation"]["path_inside_run_folder"])
    answer = reference_records[0]
    checks = []
    for judge_kind in rules.JUDGE_KINDS:
        prompt_text = rules.render_judge_prompt(judge_kind, answer)
        reply_text, finish_reason, number_of_tokens, prompt_tokens = run_one_judge_request(session, judge_kind, prompt_text,
                                                                                        session.first_attempt_temperature)
        verdict, problem, needed_extraction = rules.parse_judge_reply(judge_kind, reply_text, answer)
        checks.append({"judge_kind": judge_kind, "answer_id": answer["answer_id"], "reply_text": reply_text, "finish_reason": finish_reason,
                       "number_of_generated_tokens": number_of_tokens, "prompt_tokens": prompt_tokens, "parsed": verdict is not None,
                       "problem": problem, "needed_json_extraction": needed_extraction, "verdict": verdict})
    passed = all(check["parsed"] for check in checks)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"first_judge_reply_check_{timestamp_now_with_seconds()}.json"
    path.write_text(json.dumps({"passed": passed, "model_revision": session.model["model_revision"], "checks": checks,
                                "checked_at": datetime.now().isoformat(timespec="seconds")}, indent=2, ensure_ascii=False), encoding="utf-8")
    if not passed:
        message = ("FIRST JUDGE REPLY CHECK FAILED: " + "; ".join(f"{check['judge_kind']}: {check['problem']} (reply {check['reply_text'][:200]!r})"
                                                                   for check in checks if not check["parsed"]) + f". Details: {path.name}")
        session.say(message)
        raise SystemExit(message)
    session.say("First judge reply check passed: both judge prompts got a reply that parses")


# ---------------------------------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------------------------------

def run_judge_until_every_answer_is_final(session, scope, judge_kind, answers, label, render_prompt=None, parse_reply=None,
                                          after_attempt=None):
    """Judges every answer in `answers` that has no final judgement yet: attempt 1 at the first-attempt temperature, then retries
    of the replies that did not parse, until each answer's judgement is final. Returns counts.
    render_prompt(answer) and parse_reply(answer, text) -> (verdict, problem, needed_extraction) default to the two Haiku judges' own
    functions; after_attempt(answer, attempt_records, final_judgement_or_None) is called after every stored reply."""
    render_prompt = render_prompt or (lambda answer: rules.render_judge_prompt(judge_kind, answer, addendum_for_judge_kind(session, judge_kind)))
    parse_reply = parse_reply or (lambda answer, text: parse_judge_reply_for_session_variant(session, judge_kind, text, answer))
    store = session.stores[(scope, judge_kind)]
    counts = Counter()
    first_attempt_replies = Counter()
    while True:
        attempts = store.attempts_by_answer_id()
        pending = []
        for answer in answers:
            records = attempts.get(answer["answer_id"], [])
            if rules.final_judgement_from_attempts(records) is None:
                pending.append((answer, *rules.next_attempt_number_and_temperature(records)))
        if not pending:
            break
        requests, lookup, prompt_token_counts = [], {}, {}
        cap = session.judge_output_caps[judge_kind]
        for answer, attempt_number, temperature in pending:
            temperature = session.first_attempt_temperature if attempt_number == 1 else session.retry_temperature
            prompt_text = render_prompt(answer)
            prompt_token_ids = session.tokenizer_adapter.prompt_token_ids(prompt_text, session.judge_condition_spec)
            request_id = f"{judge_kind}__{answer['answer_id']}__attempt_{attempt_number}"
            if len(prompt_token_ids) + cap > session.maximum_model_length:  # cannot be submitted: recorded as an unparseable attempt
                record = build_attempt_record(session, judge_kind, answer, attempt_number, temperature, cap, "not_run", 0,
                                              "", None, "judge prompt plus reply cap exceeds the maximum model length", False)
                store.append(record)
                counts["prompt_too_long"] += 1
                if after_attempt:
                    previous = attempts.get(answer["answer_id"], [])
                    after_attempt(answer, previous + [record], rules.final_judgement_from_attempts(previous + [record]), 0)
                continue
            lookup[request_id] = (answer, attempt_number, temperature)
            prompt_token_counts[request_id] = len(prompt_token_ids)
            requests.append({"request_id": request_id, "prompt_token_ids": prompt_token_ids, "maximum_new_tokens": cap,
                             "temperature": temperature, "top_p": session.judge_top_p})
        if not requests:
            continue
        tracker = baseline_pod.AnswerProgressTracker(f"{label} {judge_kind} attempt {pending[0][1]}", len(requests), session.log_path,
                                                     session.report_interval_seconds)

        def on_finished(request_id, token_ids, finish_reason, lookup=lookup, tracker=tracker, attempts=attempts,
                        prompt_token_counts=prompt_token_counts):
            answer, attempt_number, temperature = lookup[request_id]
            text = session.tokenizer_adapter.decode(baseline_pod.strip_trailing_end_of_sequence_tokens(token_ids, session.end_of_sequence_ids))
            verdict, problem, needed_extraction = parse_reply(answer, text)
            record = build_attempt_record(session, judge_kind, answer, attempt_number, temperature, cap, finish_reason, len(token_ids),
                                          text, verdict, problem, needed_extraction)
            store.append(record)
            if after_attempt:
                records = attempts.get(answer["answer_id"], []) + [record]
                after_attempt(answer, records, rules.final_judgement_from_attempts(records), prompt_token_counts[request_id])
            counts["replies"] += 1
            counts["unparseable_replies"] += verdict is None
            if verdict is None and attempt_number == 1:
                first_attempt_replies["unparseable"] += 1
            if attempt_number == 1:
                first_attempt_replies["all"] += 1
                stop_if_the_judge_is_broken(session, judge_kind, first_attempt_replies, text, problem)
            tracker.note_finished(len(token_ids), finish_reason == "length")
        session.backend.run_requests(requests, on_finished, tracker.report_if_due)
        tracker.report(final=True)
    return dict(counts)


def addendum_for_judge_kind(session, judge_kind):
    """The strictness addendum text for the session's judge variant, or None (the default: the prompt is not changed at all)."""
    if session.judge_variant is None or not session.judge_variant["strictness_addendum"]:
        return None
    return session.strictness_addenda[judge_kind]


def judge_variant_is_thinking_on(session):
    return session.judge_variant is not None and session.judge_variant["thinking_on"]


def parse_judge_reply_for_session_variant(session, judge_kind, reply_text, answer):
    """The verdict is parsed from the final answer only, after the thinking channel, when the variant has thinking on."""
    if judge_variant_is_thinking_on(session):
        reply_text, _, _ = rules.final_reply_text_after_thinking(reply_text)
    return rules.parse_judge_reply(judge_kind, reply_text, answer)


def stop_if_the_judge_is_broken(session, judge_kind, first_attempt_replies, last_reply_text, last_problem):
    if session.skip_first_reply_checks or first_attempt_replies["all"] != BROKEN_JUDGE_CHECK_NUMBER_OF_REPLIES:
        return
    share = first_attempt_replies["unparseable"] / first_attempt_replies["all"]
    if share > BROKEN_JUDGE_CHECK_UNPARSEABLE_SHARE:
        message = (f"JUDGE IS BROKEN: {first_attempt_replies['unparseable']} of the first {first_attempt_replies['all']} first-attempt "
                   f"{judge_kind} replies did not parse (last problem: {last_problem}; last reply {last_reply_text[:200]!r}). "
                   f"Replies so far are saved; a relaunch resumes.")
        session.say(message)
        raise BrokenJudgeError(message)


def build_attempt_record(session, judge_kind, answer, attempt_number, temperature, output_cap, finish_reason, number_of_tokens, reply_text,
                         verdict, problem, needed_extraction):
    return {"answer_id": answer["answer_id"], "prompt_gid": answer.get("prompt_gid"), "condition": answer.get("condition"),
            "judge_kind": judge_kind, "judge_model": session.model["model_repository"], "judge_model_revision": session.model["model_revision"],
            "attempt_number": attempt_number, "temperature": temperature, "top_p": session.judge_top_p, "output_cap": output_cap,
            "finish_reason": finish_reason, "hit_output_cap": finish_reason == "length", "number_of_generated_tokens": number_of_tokens,
            "raw_reply_text": reply_text, "parse_status": "ok" if verdict is not None else "unparseable", "parse_problem": problem,
            "needed_json_extraction": needed_extraction, "verdict": verdict, "judged_at": datetime.now().isoformat(timespec="seconds"),
            "judge_variant_name": session.judge_variant_name or rules.BASELINE_JUDGE_VARIANT_NAME,
            "judge_thinking_on": judge_variant_is_thinking_on(session),
            "judge_strictness_addendum_appended": addendum_for_judge_kind(session, judge_kind) is not None,
            **({"final_reply_text_after_thinking": rules.final_reply_text_after_thinking(reply_text, finish_reason)[0]}
               if judge_variant_is_thinking_on(session) else {})}


def judge_scope_to_completion(session, scope, welfare_answers, choice_answer_selector, label):
    """Welfare and refusal judge on `welfare_answers`, then the choice judge on the answers `choice_answer_selector(welfare finals)`
    picks. Returns the counts of both."""
    welfare_answers = [answer for answer in welfare_answers if (answer.get("final_answer_text") or "").strip()]
    welfare_counts = run_judge_until_every_answer_is_final(session, scope, rules.WELFARE_AND_REFUSAL_JUDGE, welfare_answers, label)
    welfare_attempts = session.stores[(scope, rules.WELFARE_AND_REFUSAL_JUDGE)].attempts_by_answer_id()
    welfare_finals = {answer["answer_id"]: rules.final_judgement_from_attempts(welfare_attempts.get(answer["answer_id"], []))
                      for answer in welfare_answers}
    choice_answers = choice_answer_selector(welfare_answers, welfare_finals)
    choice_counts = run_judge_until_every_answer_is_final(session, scope, rules.CHOICE_OF_OPTION_JUDGE, choice_answers, label)
    return {"welfare_and_refusal": welfare_counts, "choice_of_option": choice_counts,
            "number_of_choice_judged_answers": len(choice_answers)}


def select_choice_answers_for_treatment_arm(welfare_answers, welfare_finals):
    return [answer for answer in welfare_answers if rules.choice_judge_is_needed(welfare_finals[answer["answer_id"]])]


def select_choice_answers_for_phase_0(session):
    """Phase 0 judges the choice of option on the answers the Haiku audit selected (Haiku's welfare verdict says no welfare cost)."""
    selected_ids = {record["answer_id"] for record in rules.answers_selected_for_choice_judging_like_the_haiku_audit(phase_0_answers(session))}
    return lambda welfare_answers, welfare_finals: [answer for answer in phase_0_answers(session) if answer["answer_id"] in selected_ids]


# ---------------------------------------------------------------------------------------------------
# Markers and phases
# ---------------------------------------------------------------------------------------------------

def marker_path_if_present(session, phase_number):
    matches = sorted(session.markers_folder.glob(f"{PHASE_NAMES[phase_number]}_completed_*.json")) if session.markers_folder.exists() else []
    return matches[-1] if matches else None


def write_marker(session, phase_number, seconds, counts):
    session.markers_folder.mkdir(parents=True, exist_ok=True)
    path = session.markers_folder / f"{PHASE_NAMES[phase_number]}_completed_{timestamp_now_with_seconds()}.json"
    path.write_text(json.dumps({"phase": PHASE_NAMES[phase_number], "completed_at": datetime.now().isoformat(timespec="seconds"),
                                "seconds_this_session": seconds, "counts": counts}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_phase(session, phase_number, is_complete_function, work_function):
    """Skips the phase when its marker exists and the files show nothing left to do; otherwise does the work and writes the marker."""
    if marker_path_if_present(session, phase_number) and is_complete_function():
        session.say(f"{PHASE_NAMES[phase_number]}: completion marker present and nothing left to do; skipped")
        return
    session.say(f"{PHASE_NAMES[phase_number]}: starting (session so far {format_duration(session.seconds_since_start())})")
    start = session.clock()
    counts = work_function()
    seconds = session.clock() - start
    write_marker(session, phase_number, seconds, counts)
    session.say(f"{PHASE_NAMES[phase_number]}: done in {format_duration(seconds)}; {json.dumps(counts)[:600]}")


def phase_0_answers(session):
    records = read_jsonl(session.run_folder / session.section["phase_0_judge_validation"]["path_inside_run_folder"])
    limit = session.maximum_phase_0_answers_per_condition
    if limit:
        counts, limited = Counter(), []
        for record in records:
            counts[record["condition"]] += 1
            if counts[record["condition"]] <= limit:
                limited.append(record)
        return limited
    return records


def all_judgements_are_final(session, scope, judge_kind, answers):
    attempts = session.stores[(scope, judge_kind)].attempts_by_answer_id()
    return all(rules.final_judgement_from_attempts(attempts.get(answer["answer_id"], [])) is not None for answer in answers)


def phase_0_is_complete(session):
    answers = [answer for answer in phase_0_answers(session) if (answer.get("final_answer_text") or "").strip()]
    if not all_judgements_are_final(session, "phase_0", rules.WELFARE_AND_REFUSAL_JUDGE, answers):
        return False
    selected = select_choice_answers_for_phase_0(session)(answers, None)
    return all_judgements_are_final(session, "phase_0", rules.CHOICE_OF_OPTION_JUDGE, selected)


def run_phase_0(session):
    counts = judge_scope_to_completion(session, "phase_0", phase_0_answers(session), select_choice_answers_for_phase_0(session),
                                       "phase 0 held-out validation")
    for line in agreement.format_agreement_log_lines(
            phase_0_answers(session), session.stores[("phase_0", rules.WELFARE_AND_REFUSAL_JUDGE)].attempts_by_answer_id(),
            session.stores[("phase_0", rules.CHOICE_OF_OPTION_JUDGE)].attempts_by_answer_id()):
        session.say(line)
    return counts


def reference_sample_prompt_gids(session):
    """The prompts that get a no-system-prompt answer (drawn up front by the run folder builder, recorded in the run configuration and a
    generation input file), or None when every prompt does (run folders built before the sample existed, or a sample of every prompt)."""
    sample = session.section.get("no_system_prompt_reference_answer_sample")
    if not sample:
        return None
    path = session.run_folder / sample["path_inside_run_folder"]
    if sha256_of_file(path) != sample["sha256"]:
        raise SystemExit(f"{path} does not match the hash recorded in the run configuration")
    return set(json.loads(path.read_text(encoding="utf-8"))["prompt_gids"])


def generation_plans(session, only_pending_regeneration=None):
    """Condition plans for the answers still to generate. In phase 1 (only_pending_regeneration None) every missing first answer;
    in phase 3 the given (prompt, sample_index) pairs of the system-prompted condition."""
    plans = []
    for name, entry in session.condition_entries.items():
        if only_pending_regeneration is not None and name != session.system_prompted_condition:
            continue
        if only_pending_regeneration is not None:
            plan = {"condition": name, "condition_spec": entry["condition_spec"], "output_cap": entry["output_cap"],
                    "answers_path": session.answers_paths[name], "pending_answers": only_pending_regeneration, "already_saved": 0}
        else:
            samples = session.section["initial_samples_per_prompt"] if name == session.system_prompted_condition \
                else session.section["no_system_prompt_samples_per_prompt"]
            sample = reference_sample_prompt_gids(session) if name == session.reference_condition else None
            plan_prompts = session.prompts if sample is None else [prompt for prompt in session.prompts if prompt["prompt_gid"] in sample]
            plan = baseline_pod.build_condition_plan(session.model, name, plan_prompts, samples, entry["output_cap"],
                                                     session.answers_paths[name], entry["condition_spec"])
        plans.append(plan)
    return plans


def phase_1_is_complete(session):
    return all(not plan["pending_answers"] for plan in generation_plans(session))


def run_phase_1(session):
    plans = generation_plans(session)
    for plan in plans:
        session.say(f"{plan['condition']}: {plan['already_saved']} answers already saved, {len(plan['pending_answers'])} to generate")
    finished = baseline_pod.generate_conditions_together(session.model, plans, session.backend, session.tokenizer_adapter,
                                                         session.end_of_sequence_ids, session.log_path, session.report_interval_seconds)
    return {"answers_generated_this_session_by_condition": finished}


def read_system_prompted_answers_by_prompt(session):
    path = session.answers_paths[session.system_prompted_condition]
    baseline_pod.read_finished_answer_ids(path)  # cuts off a half-written last line
    by_prompt = {}
    for record in (read_jsonl(path) if path.exists() else []):
        by_prompt.setdefault(record["prompt_gid"], []).append(record)
    for records in by_prompt.values():
        records.sort(key=lambda record: record["sample_index"])
    return by_prompt


def collect_decisions(session):
    """prompt_gid -> list of {answer, welfare, choice, decided, passed, failure_reasons} ordered by sample_index."""
    welfare_attempts = session.stores[("treatment_arm", rules.WELFARE_AND_REFUSAL_JUDGE)].attempts_by_answer_id()
    choice_attempts = session.stores[("treatment_arm", rules.CHOICE_OF_OPTION_JUDGE)].attempts_by_answer_id()
    decisions = {}
    prompt_gids = {prompt["prompt_gid"] for prompt in session.prompts}
    for prompt_gid, answers in read_system_prompted_answers_by_prompt(session).items():
        if prompt_gid not in prompt_gids:
            continue
        entries = []
        for answer in answers:
            welfare = rules.final_judgement_from_attempts(welfare_attempts.get(answer["answer_id"], []))
            choice = rules.final_judgement_from_attempts(choice_attempts.get(answer["answer_id"], []))
            decided, passed, reasons = rules.decide_answer(answer, welfare, choice)
            entries.append({"answer": answer, "welfare": welfare, "choice": choice, "decided": decided, "passed": passed,
                            "failure_reasons": reasons})
        decisions[prompt_gid] = entries
    return decisions


def system_prompted_answers_of_prompts(session):
    return [answer for answers in read_system_prompted_answers_by_prompt(session).values() for answer in answers
            if answer["prompt_gid"] in {prompt["prompt_gid"] for prompt in session.prompts}]


def judge_every_unjudged_treatment_arm_answer(session, label):
    return judge_scope_to_completion(session, "treatment_arm", system_prompted_answers_of_prompts(session),
                                     select_choice_answers_for_treatment_arm, label)


def phase_2_is_complete(session):
    return all(entry["decided"] for entries in collect_decisions(session).values() for entry in entries)


def run_phase_2(session):
    return judge_every_unjudged_treatment_arm_answer(session, "phase 2 first answers")


def prompts_needing_another_attempt(session, decisions):
    """[(prompt, next sample_index)] for prompts with no passing answer, every answer decided and fewer than the maximum attempts."""
    pending = []
    for prompt in session.prompts:
        entries = decisions.get(prompt["prompt_gid"], [])
        if not entries or any(entry["passed"] for entry in entries) or not all(entry["decided"] for entry in entries):
            continue
        if len(entries) >= session.maximum_attempts_per_prompt:
            continue
        pending.append((prompt, max(entry["answer"]["sample_index"] for entry in entries) + 1))
    return pending


def phase_3_is_complete(session):
    decisions = collect_decisions(session)
    return all(entry["decided"] for entries in decisions.values() for entry in entries) and not prompts_needing_another_attempt(session, decisions)


def run_phase_3(session):
    rounds = []
    while True:
        decisions = collect_decisions(session)
        if not all(entry["decided"] for entries in decisions.values() for entry in entries):  # a killed session left answers unjudged
            session.say("phase 3: judging answers left unjudged by an earlier session")
            judge_every_unjudged_treatment_arm_answer(session, "phase 3 unjudged answers")
            continue
        pending = prompts_needing_another_attempt(session, decisions)
        if not pending:
            break
        session.say(f"phase 3 round {len(rounds) + 1}: {len(pending)} prompts have no passing answer yet; regenerating one answer each")
        plan = generation_plans(session, only_pending_regeneration=pending)[0]
        baseline_pod.generate_conditions_together(session.model, [plan], session.backend, session.tokenizer_adapter,
                                                  session.end_of_sequence_ids, session.log_path, session.report_interval_seconds)
        judge_every_unjudged_treatment_arm_answer(session, f"phase 3 round {len(rounds) + 1}")
        rounds.append({"round": len(rounds) + 1, "prompts_regenerated": len(pending)})
    return {"regeneration_rounds": rounds}


# ---------------------------------------------------------------------------------------------------
# Phase 4: outputs
# ---------------------------------------------------------------------------------------------------

def build_kept_answer_record(session, entry):
    answer = entry["answer"]
    judge_name = "haiku" if session.haiku else "gemma"
    verdict_labels = {f"{judge_name}_welfare_and_refusal": entry["welfare"]["verdict"]}
    if entry["choice"] is not None:
        verdict_labels[f"{judge_name}_choice_of_option"] = {rules.choice_judge_module.VERDICT_FIELD: entry["choice"]["verdict"]}
    return {**answer, "judge_verdicts": verdict_labels, "kept_after_attempt_number": answer["sample_index"] + 1}


def build_outcome_record(session, prompt, entries, reference_answer_id):
    passing = [entry for entry in entries if entry["passed"]]
    if passing:
        status = f"passed_at_attempt_{passing[0]['answer']['sample_index'] + 1}"
    elif entries and len(entries) >= session.maximum_attempts_per_prompt:
        status = "never_passed"
    else:
        status = "incomplete"
    return {"prompt_gid": prompt["prompt_gid"], "number_of_attempts": len(entries), "final_status": status,
            "kept_answer_id": passing[0]["answer"]["answer_id"] if passing else None, "no_system_prompt_answer_id": reference_answer_id,
            "attempts": [{"answer_id": entry["answer"]["answer_id"], "sample_index": entry["answer"]["sample_index"],
                          "passed": entry["passed"], "decided": entry["decided"], "failure_reasons": entry["failure_reasons"],
                          "hit_output_cap": entry["answer"].get("hit_output_cap"),
                          "judge_verdicts": rules.verdict_summary_for_record(entry["welfare"], entry["choice"])} for entry in entries]}


def summarise_outcomes(session, outcome_records, decisions):
    rounds = []
    for attempt_number in range(1, session.maximum_attempts_per_prompt + 1):
        entries = [entry for entries_of_prompt in decisions.values() for entry in entries_of_prompt
                   if entry["answer"]["sample_index"] + 1 == attempt_number]
        reasons = Counter(reason for entry in entries for reason in entry["failure_reasons"])
        rounds.append({"attempt_number": attempt_number, "answers": len(entries), "passed": sum(entry["passed"] for entry in entries),
                       "failed": sum(entry["decided"] and not entry["passed"] for entry in entries),
                       "failure_reasons": {reason: reasons[reason] for reason in rules.ALL_FAILURE_REASONS if reasons[reason]}})
    status_counts = Counter(record["final_status"] for record in outcome_records)
    all_entries = [entry for entries in decisions.values() for entry in entries]
    judge_parse = {}
    for judge_kind in ([] if session.haiku else rules.JUDGE_KINDS):
        attempts = session.stores[("treatment_arm", judge_kind)].attempts_by_answer_id()
        finals = {answer_id: rules.final_judgement_from_attempts(records) for answer_id, records in attempts.items()}
        judge_parse[judge_kind] = {
            "answers_judged": len(attempts),
            "answers_ok_at_attempt": {str(number): sum(1 for final in finals.values() if final and final["parse_status"] == "ok"
                                                       and final["attempts_used"] == number) for number in range(1, 4)},
            "answers_finally_unparseable": sum(1 for final in finals.values() if final and final["parse_status"] == "unparseable"),
            "replies_that_needed_json_extraction": sum(record["needed_json_extraction"] for records in attempts.values() for record in records),
            "replies_that_hit_the_output_cap": sum(record["hit_output_cap"] for records in attempts.values() for record in records)}
    haiku_judging_summary = None
    if session.haiku:
        judge_parse = {}
        haiku_judging_summary = {"judged_by": "claude-haiku-5-5 through the paid Anthropic API on the laptop",
                                 "answers_with_a_decision": sum(entry["decided"] for entry in all_entries),
                                 "answers_with_an_unparseable_verdict": sum(any(reason.startswith("unparseable") for reason in entry["failure_reasons"])
                                                                            for entry in all_entries)}
    timings = {}
    for phase_number in PHASE_NAMES:
        marker = marker_path_if_present(session, phase_number)
        if marker:
            timings[PHASE_NAMES[phase_number]] = json.loads(marker.read_text(encoding="utf-8"))["seconds_this_session"]
    reference_path = session.answers_paths.get(session.reference_condition) if session.reference_condition else None
    return {
        "run_name": session.configuration["run_name"], "summarised_at": datetime.now().isoformat(timespec="seconds"),
        "number_of_prompts": len(session.prompts), "maximum_attempts_per_prompt": session.maximum_attempts_per_prompt,
        "prompts_by_final_status": dict(status_counts),
        "prompts_that_never_passed": sorted(record["prompt_gid"] for record in outcome_records if record["final_status"] == "never_passed"),
        "prompts_incomplete": sorted(record["prompt_gid"] for record in outcome_records if record["final_status"] == "incomplete"),
        "rounds": rounds, "judge_parse_statistics": judge_parse, **({"haiku_judging": haiku_judging_summary} if haiku_judging_summary else {}),
        "system_prompted_answers_generated": len(all_entries),
        "system_prompted_answers_that_hit_the_output_cap": sum(bool(entry["answer"].get("hit_output_cap")) for entry in all_entries),
        "kept_answers_that_hit_the_output_cap": sum(bool(record_answer["hit_output_cap"]) for record_answer in
                                                    (entry["answer"] for entries in decisions.values() for entry in entries if entry["passed"])),
        "no_system_prompt_reference_answers": len(baseline_pod.read_finished_answer_ids(reference_path)) if reference_path else 0,
        "seconds_by_phase_this_and_earlier_sessions_as_recorded_in_markers": timings,
        "seconds_in_this_session_so_far": session.seconds_since_start()}


def format_summary_markdown(summary):
    lines = [f"# Treatment arm answer generation, judging and regeneration: {summary['run_name']}", "",
             f"Summarised {summary['summarised_at']}. {summary['number_of_prompts']:,} prompts, at most "
             f"{summary['maximum_attempts_per_prompt']} system-prompted answers each.", "", "## Outcome per prompt", "",
             "| Final status | Prompts |", "|---|---|"]
    lines += [f"| {status.replace('_', ' ')} | {count:,} |" for status, count in sorted(summary["prompts_by_final_status"].items())]
    lines += ["", "## Rounds (attempt number = which answer of the prompt)", "",
              "| Attempt | Answers | Passed | Failed | Failure reasons (an answer can have several) |", "|---|---|---|---|---|"]
    for round_record in summary["rounds"]:
        reasons = "; ".join(f"{reason.replace('_', ' ')} {count:,}" for reason, count in round_record["failure_reasons"].items()) or "none"
        lines.append(f"| {round_record['attempt_number']} | {round_record['answers']:,} | {round_record['passed']:,} | "
                     f"{round_record['failed']:,} | {reasons} |")
    lines += ["", f"Prompts that never passed: {len(summary['prompts_that_never_passed']):,}"
              + (f" (first ten: {', '.join(summary['prompts_that_never_passed'][:10])}; all in the summary json)"
                 if summary["prompts_that_never_passed"] else ""),
              f"Prompts left incomplete: {len(summary['prompts_incomplete']):,}", "", "## Judge parsing", "",
              "| Judge | Answers judged | Parsed at attempt 1 / 2 / 3 | Finally unparseable | Needed JSON extraction | Reply hit the cap |",
              "|---|---|---|---|---|---|"]
    for judge_kind, statistics in summary["judge_parse_statistics"].items():
        ok = statistics["answers_ok_at_attempt"]
        lines.append(f"| {judge_kind.replace('_', ' ')} | {statistics['answers_judged']:,} | {ok['1']:,} / {ok['2']:,} / {ok['3']:,} | "
                     f"{statistics['answers_finally_unparseable']:,} | {statistics['replies_that_needed_json_extraction']:,} | "
                     f"{statistics['replies_that_hit_the_output_cap']:,} |")
    lines += ["", "## Answers", "",
              f"System-prompted answers generated: {summary['system_prompted_answers_generated']:,}, of which "
              f"{summary['system_prompted_answers_that_hit_the_output_cap']:,} hit the output cap; kept answers that hit the cap: "
              f"{summary['kept_answers_that_hit_the_output_cap']:,}. No-system-prompt reference answers: "
              f"{summary['no_system_prompt_reference_answers']:,}.", "", "## Timings", "", "| Phase | Seconds | Duration |", "|---|---|---|"]
    lines += [f"| {phase.replace('_', ' ')} | {seconds:.0f} | {format_duration(seconds)} |"
              for phase, seconds in summary["seconds_by_phase_this_and_earlier_sessions_as_recorded_in_markers"].items()]
    return "\n".join(lines) + "\n"


def phase_4_is_complete(session):
    marker = marker_path_if_present(session, 4)
    if not marker:
        return False
    return all((session.run_folder / path).exists() for path in json.loads(marker.read_text(encoding="utf-8"))["counts"]["output_files"])


def run_phase_4(session):
    if session.haiku:
        decisions = decisions_for_phase_4_with_haiku_judging(session)
    else:
        decisions = collect_decisions(session)
        if not all(entry["decided"] for entries in decisions.values() for entry in entries) or prompts_needing_another_attempt(session, decisions):
            raise SystemExit("Phase 4 needs phases 2 and 3 to be complete")
    reference_answer_ids = {}
    if session.reference_condition and session.answers_paths[session.reference_condition].exists():
        for record in read_jsonl(session.answers_paths[session.reference_condition]):
            reference_answer_ids.setdefault(record["prompt_gid"], record["answer_id"])
    outcome_records, kept_records = [], []
    for prompt in session.prompts:
        entries = decisions.get(prompt["prompt_gid"], [])
        outcome_records.append(build_outcome_record(session, prompt, entries, reference_answer_ids.get(prompt["prompt_gid"])))
        passing = [entry for entry in entries if entry["passed"]]
        if passing:
            kept_records.append(build_kept_answer_record(session, passing[0]))
    held_out_ids = set(session.held_out_identifiers["corpus_prompt_ids"])
    held_out_messages = set(session.held_out_identifiers["user_messages"])
    offending = [record["answer_id"] for record in kept_records if answer_is_on_a_held_out_prompt(record, held_out_ids, held_out_messages)]
    if offending:
        raise SystemExit(f"{len(offending)} kept answers are on held-out set prompts, e.g. {offending[:3]}; nothing was written")
    stamp = timestamp_now_with_seconds()
    folder = session.final_outputs_folder
    folder.mkdir(parents=True, exist_ok=True)
    model_name = session.model["model_short_name"]
    kept_path = folder / f"kept_answers_{model_name}_thinking_off_with_system_prompt_one_passing_answer_per_prompt_{stamp}.jsonl"
    outcomes_path = folder / f"per_prompt_outcomes_with_attempts_verdicts_and_final_status_{stamp}.jsonl"
    write_jsonl(kept_path, kept_records)
    write_jsonl(outcomes_path, outcome_records)
    summary = summarise_outcomes(session, outcome_records, decisions)
    summary_json_path = folder / f"treatment_arm_answer_generation_summary_{stamp}.json"
    summary_markdown_path = folder / f"treatment_arm_answer_generation_summary_{stamp}.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_markdown_path.write_text(format_summary_markdown(summary), encoding="utf-8")
    session.say(f"Kept answers for {len(kept_records):,} of {len(session.prompts):,} prompts: {kept_path.name}")
    session.say(format_summary_markdown(summary))
    paths = [path.relative_to(session.run_folder).as_posix() for path in (kept_path, outcomes_path, summary_json_path, summary_markdown_path)]
    return {"kept_answers": len(kept_records), "prompts": len(session.prompts), "output_files": paths}


# ---------------------------------------------------------------------------------------------------
# Phases 5 and 6: the framework judges (completeness against the no-system-prompt answer, user lean and agreement)
# ---------------------------------------------------------------------------------------------------

def framework_judge_folder(session, specification):
    return session.framework_judge_outputs_parent / specification.folder_name


def latest_kept_answers_file(session):
    files = sorted(session.final_outputs_folder.glob("kept_answers_*.jsonl")) if session.final_outputs_folder.exists() else []
    if not files:
        raise SystemExit("No kept answers file yet: phase 4 must be complete before phases 5 and 6")
    return files[-1]


def reference_answers_of_session_prompts(session):
    """Every no-system-prompt answer to a prompt of this session (empty when the run has no reference condition)."""
    if not session.reference_condition or not session.answers_paths[session.reference_condition].exists():
        return []
    baseline_pod.read_finished_answer_ids(session.answers_paths[session.reference_condition])
    prompt_gids = {prompt["prompt_gid"] for prompt in session.prompts}
    return [record for record in read_jsonl(session.answers_paths[session.reference_condition]) if record["prompt_gid"] in prompt_gids]


def verdict_answer_ids_written(session, specification):
    folder = framework_judge_folder(session, specification)
    return {record["answer_id"] for path in sorted(folder.glob("verdicts_*.jsonl")) if folder.exists()
            for record in rules.read_jsonl_dropping_a_half_written_last_line(path)}


def build_framework_verdict_record(session, specification, answer, comparison_answer, attempt_records, final):
    """The record the CLI framework's judge_one_answer writes, with the judge marked as Gemma on the pod."""
    record = {key: answer.get(key) for key in framework.CARRIED_ANSWER_FIELDS}
    if comparison_answer is not None:
        record["comparison_answer_id"] = comparison_answer["answer_id"]
    record.update(judge_prompt_file_name=specification.template_file_name, judge_prompt_sha256=specification.template_hash_of_text,
                  judge_model=session.model["model_repository"], judge_model_revision=session.model["model_revision"],
                  judged_by_gemma_on_pod=True, verdict=final["verdict"], parse_status=final["parse_status"], attempts=len(attempt_records),
                  raw_replies=[attempt["raw_reply_text"] for attempt in attempt_records],
                  attempt_temperatures=[attempt["temperature"] for attempt in attempt_records],
                  judged_at=datetime.now().isoformat(timespec="seconds"))
    return record


def run_framework_judge(session, specification, answers, comparison_by_prompt_gid, label):
    """Judges `answers` (paired by prompt_gid with comparison_by_prompt_gid when the template is paired) and writes the framework's
    folder layout under judge_outputs_gemma_on_pod/. Returns counts."""
    kind = specification.judge_kind
    skipped = 0
    if comparison_by_prompt_gid is not None:
        answers, skipped = framework.keep_answers_that_have_a_comparison_answer(answers, comparison_by_prompt_gid)
        session.say(f"{label}: {skipped} answers skipped for having no comparison answer of condition {specification.comparison_condition}")
    folder = framework_judge_folder(session, specification)
    folder.mkdir(parents=True, exist_ok=True)
    framework.check_or_write_judge_settings(folder, {
        "judge_prompt_file_name": specification.template_file_name, "judge_prompt_sha256": specification.template_hash_of_text,
        "reply_schema": specification.schema, "reply_schema_sha256": sha256_of_text(json.dumps(specification.schema, sort_keys=True)),
        "comparison_condition": specification.comparison_condition, "created_at": datetime.now().isoformat(timespec="seconds"),
        "judged_by_gemma_on_pod": True, "judge_model": session.model["model_repository"],
        "judge_model_revision": session.model["model_revision"],
        "judge_sampling": {"first_attempt_temperature": session.first_attempt_temperature, "retry_temperature": session.retry_temperature,
                           "top_p": session.judge_top_p, "maximum_parse_attempts": session.maximum_parse_attempts,
                           "output_cap": session.judge_output_caps[kind], "thinking": "off"}})
    session.stores[("framework", kind)] = rules.JudgeAttemptStore(session.stable_path(folder, "gemma_judge_attempts", ".jsonl"), kind)
    verdict_path = session.stable_path(folder, "verdicts", ".jsonl")
    usage_path = session.stable_path(folder, "usage_log", ".jsonl")
    written_ids = verdict_answer_ids_written(session, specification)

    def comparison_of(answer):
        return comparison_by_prompt_gid[answer["prompt_gid"]] if comparison_by_prompt_gid is not None else None

    def write_verdict_if_final(answer, attempt_records, final):
        if final is not None and answer["answer_id"] not in written_ids:
            rules.append_jsonl_record_durably(verdict_path, build_framework_verdict_record(
                session, specification, answer, comparison_of(answer), attempt_records, final))
            written_ids.add(answer["answer_id"])

    def after_attempt(answer, attempt_records, final, prompt_tokens):
        last = attempt_records[-1]
        rules.append_jsonl_record_durably(usage_path, {
            "answer_id": answer["answer_id"], "attempt": last["attempt_number"], "judge_model": session.model["model_repository"],
            "input_tokens": prompt_tokens, "output_tokens": last["number_of_generated_tokens"], "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None, "cli_reported_cost_usd": None, "problem": last["parse_problem"],
            "logged_at": last["judged_at"]})
        write_verdict_if_final(answer, attempt_records, final)

    counts = run_judge_until_every_answer_is_final(
        session, "framework", kind, answers, label, render_prompt=lambda answer: specification.render(answer, comparison_of(answer)),
        parse_reply=lambda answer, text: specification.parse(text), after_attempt=after_attempt)
    attempts = session.stores[("framework", kind)].attempts_by_answer_id()
    for answer in answers:  # a kill between storing the last attempt and writing the verdict record leaves a final answer unwritten
        records = attempts.get(answer["answer_id"], [])
        write_verdict_if_final(answer, records, rules.final_judgement_from_attempts(records))
    counts["answers_to_judge"], counts["skipped_without_comparison_answer"] = len(answers), skipped
    return counts


def sampling_settings(session):
    return session.section.get("framework_judge_sampling") or {"completeness_sample_size": None, "user_lean_sample_prompts": None, "seed": 0}


def completeness_judge_inputs(session):
    """(kept answers with a pair, sampled when configured, comparison_by_prompt_gid, number without a pair). The sample is drawn from
    the answers sorted by answer_id with a generator seeded by the seed and the judge's name, so it depends only on the seed and the kept
    answers file: a resumed session picks the same answers."""
    specification = session.framework_judges[rules.COMPLETENESS_AND_COMPETENCE_JUDGE]
    kept_answers = load_answer_records([latest_kept_answers_file(session)])
    comparison = framework.pick_comparison_answer_per_prompt(reference_answers_of_session_prompts(session), specification.comparison_condition)
    paired, skipped = framework.keep_answers_that_have_a_comparison_answer(kept_answers, comparison)
    settings = sampling_settings(session)
    size = settings["completeness_sample_size"]
    paired = sorted(paired, key=lambda answer: answer["answer_id"])
    if size is not None and size < len(paired):
        paired = sorted(random.Random(f"{settings['seed']}:completeness").sample(paired, size), key=lambda answer: answer["answer_id"])
    return paired, comparison, skipped


def user_lean_judge_inputs(session):
    """The kept answer and the no-system-prompt answer of every prompt in the reference sample (drawn up front: only these prompts have a
    no-system-prompt answer) that has a kept answer. For a run folder without a sample, a seeded random sample of the configured size among
    the prompts that have both answers (or everything when the size is None)."""
    kept_answers = load_answer_records([latest_kept_answers_file(session)])
    reference_answers = reference_answers_of_session_prompts(session)
    sample = reference_sample_prompt_gids(session) if session.reference_condition else None
    prompts_with_both = sorted({a["prompt_gid"] for a in kept_answers} & {a["prompt_gid"] for a in reference_answers}) \
        if session.reference_condition else sorted({a["prompt_gid"] for a in kept_answers})
    if sample is not None:
        chosen = set(prompts_with_both) & sample
        return [a for a in kept_answers + reference_answers if a["prompt_gid"] in chosen]
    size = sampling_settings(session)["user_lean_sample_prompts"]
    if size is None:
        return kept_answers + reference_answers
    if size < len(prompts_with_both):
        prompts_with_both = sorted(random.Random(f"{sampling_settings(session)['seed']}:user_lean").sample(prompts_with_both, size))
    chosen = set(prompts_with_both)
    return [a for a in kept_answers + reference_answers if a["prompt_gid"] in chosen]


def framework_phase_is_complete(session, judge_kind, answers):
    specification = session.framework_judges[judge_kind]
    return all(answer["answer_id"] in verdict_answer_ids_written(session, specification) for answer in answers)


def verdict_value_counts(session, judge_kind, field_names):
    specification = session.framework_judges[judge_kind]
    folder = framework_judge_folder(session, specification)
    counts = {field: Counter() for field in field_names}
    for path in sorted(folder.glob("verdicts_*.jsonl")):
        for record in rules.read_jsonl_dropping_a_half_written_last_line(path):
            if record["parse_status"] == "ok":
                for field in field_names:
                    counts[field][record["verdict"][field]] += 1
            else:
                counts["unparseable"] = counts.get("unparseable", Counter()) + Counter({"answers": 1})
    return {field: dict(counter) for field, counter in counts.items()}


def phase_5_is_complete(session):
    if rules.COMPLETENESS_AND_COMPETENCE_JUDGE not in session.framework_judges:
        return True
    return framework_phase_is_complete(session, rules.COMPLETENESS_AND_COMPETENCE_JUDGE, completeness_judge_inputs(session)[0])


def run_phase_5(session):
    paired, comparison, skipped = completeness_judge_inputs(session)
    session.say(f"phase 5: judging {len(paired)} kept answers (completeness sample size {sampling_settings(session)['completeness_sample_size']})")
    kept_file = latest_kept_answers_file(session)
    kept_hash_before = sha256_of_file(kept_file)
    specification = session.framework_judges[rules.COMPLETENESS_AND_COMPETENCE_JUDGE]
    counts = run_framework_judge(session, specification, paired, comparison, "phase 5 completeness and competence")
    if sha256_of_file(kept_file) != kept_hash_before:
        raise SystemExit("The kept answers file changed during phase 5: the completeness judge is a flag, never a filter")
    flags = verdict_value_counts(session, rules.COMPLETENESS_AND_COMPETENCE_JUDGE,
                                 ["left_out_part_of_the_request", "clearly_worse_on_everything_else"])
    counts["population_kept_answers_with_a_pair"] = len(load_answer_records([kept_file])) - skipped
    return {"judge_run": counts, "verdict_counts": flags, "judge_folder": framework_judge_folder(session, specification)
            .relative_to(session.run_folder).as_posix(), "kept_answers_file": kept_file.name, "kept_answers_file_sha256": kept_hash_before}


def phase_6_is_complete(session):
    if rules.USER_LEAN_AND_AGREEMENT_JUDGE not in session.framework_judges:
        return True
    return framework_phase_is_complete(session, rules.USER_LEAN_AND_AGREEMENT_JUDGE, user_lean_judge_inputs(session))


def run_phase_6(session):
    specification = session.framework_judges[rules.USER_LEAN_AND_AGREEMENT_JUDGE]
    counts = run_framework_judge(session, specification, user_lean_judge_inputs(session), None, "phase 6 user lean and agreement")
    return {"judge_run": counts, "judge_folder": framework_judge_folder(session, specification).relative_to(session.run_folder).as_posix(),
            "verdict_counts": verdict_value_counts(session, rules.USER_LEAN_AND_AGREEMENT_JUDGE,
                                                   ["user_lean", "answer_agrees_with_user_lean", "answer_pushes_back_on_any_point"])}


# ---------------------------------------------------------------------------------------------------
# Judge variant test: phase 0 only, once per variant, in one engine load
# ---------------------------------------------------------------------------------------------------

def judge_condition_spec_for_variant(session, variant):
    """The judge's chat template settings: thinking on through the model's own thinking_on setting (as the generation code does for
    thinking-on conditions), else the session's default (thinking off)."""
    if not variant["thinking_on"]:
        return session.default_judge_condition_spec
    thinking_on_kwargs = defs.candidate_model_by_short_name(session.model["model_short_name"])["thinking_on"]["chat_template_kwargs"]
    return {**session.default_judge_condition_spec, "chat_template_kwargs": dict(thinking_on_kwargs)}


def switch_session_to_judge_variant(session, variant_name):
    """Points the session at one variant: prompts, chat template settings, output caps and its own attempt files. Returns the folder."""
    variant = rules.JUDGE_VARIANTS[variant_name]
    session.default_judge_condition_spec = getattr(session, "default_judge_condition_spec", session.judge_condition_spec)
    session.default_judge_output_caps = getattr(session, "default_judge_output_caps", dict(session.judge_output_caps))
    session.judge_variant_name, session.judge_variant = variant_name, variant
    session.strictness_addenda = rules.read_strictness_addenda_by_judge_kind()
    session.judge_condition_spec = judge_condition_spec_for_variant(session, variant)
    session.judge_output_caps = ({kind: rules.THINKING_ON_JUDGE_OUTPUT_CAP for kind in session.default_judge_output_caps}
                                 if variant["thinking_on"] else dict(session.default_judge_output_caps))
    folder = session.outputs_folder / JUDGE_VARIANT_TEST_FOLDER_NAME / variant_name
    for judge_kind, stem in ((rules.WELFARE_AND_REFUSAL_JUDGE, "gemma_welfare_and_refusal_judge_attempts_on_held_out_thinking_off_answers"),
                             (rules.CHOICE_OF_OPTION_JUDGE, "gemma_choice_of_option_judge_attempts_on_held_out_thinking_off_answers")):
        session.stores[(variant_name, judge_kind)] = rules.JudgeAttemptStore(
            session.stable_path(folder, f"{stem}__{variant_name}", ".jsonl"), judge_kind)
    return folder


def write_judge_variant_settings(session, variant_name, folder):
    placements = {}
    for judge_kind in rules.JUDGE_KINDS:
        first_answer = phase_0_answers(session)[0]
        plain = rules.render_judge_prompt(judge_kind, first_answer)
        _, placements[judge_kind] = (rules.locate_strictness_addendum_insertion_point(plain)
                                     if session.judge_variant["strictness_addendum"] else (None, "no_addendum"))
    settings = {"judge_variant_name": variant_name, **session.judge_variant,
                "output_caps": session.judge_output_caps, "temperature_first_attempt": session.first_attempt_temperature,
                "temperature_retry": session.retry_temperature, "top_p": session.judge_top_p,
                "chat_template_kwargs": session.judge_condition_spec["chat_template_kwargs"],
                "verdict_parsed_from": "final answer after the thinking channel only" if session.judge_variant["thinking_on"] else "whole reply",
                "addendum_placement": placements,
                "addendum_files": ({kind: name for kind, name in rules.STRICTNESS_ADDENDUM_FILE_NAMES.items()}
                                   if session.judge_variant["strictness_addendum"] else None),
                "addendum_sha256": ({kind: hashlib.sha256(text.encode("utf-8")).hexdigest() for kind, text in
                                     session.strictness_addenda.items()} if session.judge_variant["strictness_addendum"] else None),
                "model": session.model["model_repository"], "model_revision": session.model["model_revision"],
                "written_at": datetime.now().isoformat(timespec="seconds")}
    path = session.stable_path(folder, f"judge_variant_settings__{variant_name}", ".json")
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def judge_variant_is_complete(session, variant_name):
    answers = [answer for answer in phase_0_answers(session) if (answer.get("final_answer_text") or "").strip()]
    if not all_judgements_are_final(session, variant_name, rules.WELFARE_AND_REFUSAL_JUDGE, answers):
        return False
    selected = select_choice_answers_for_phase_0(session)(answers, None)
    return all_judgements_are_final(session, variant_name, rules.CHOICE_OF_OPTION_JUDGE, selected)


def run_judge_variant_test(session, variant_names):
    """Judges the phase 0 answers once per variant (welfare and refusal judge, then the choice judge on the answers the Haiku audit
    selected), each variant's attempts in its own folder. Resumable: a variant whose judgements are all final is skipped."""
    session.skip_first_reply_checks = True  # the broken-judge stop would end the other variants too; parse statistics are logged per variant
    verify_inputs(session)
    results = {}
    for variant_name in variant_names:
        folder = switch_session_to_judge_variant(session, variant_name)
        if judge_variant_is_complete(session, variant_name):
            session.say(f"judge variant {variant_name}: every judgement is final; skipped")
            continue
        session.say(f"judge variant {variant_name}: starting (session so far {format_duration(session.seconds_since_start())})")
        write_judge_variant_settings(session, variant_name, folder)
        start = session.clock()
        counts = judge_scope_to_completion(session, variant_name, phase_0_answers(session), select_choice_answers_for_phase_0(session),
                                           f"judge variant {variant_name}")
        seconds = session.clock() - start
        for line in agreement.format_agreement_log_lines(
                phase_0_answers(session), session.stores[(variant_name, rules.WELFARE_AND_REFUSAL_JUDGE)].attempts_by_answer_id(),
                session.stores[(variant_name, rules.CHOICE_OF_OPTION_JUDGE)].attempts_by_answer_id()):
            session.say(f"[{variant_name}] {line}")
        marker = session.stable_path(folder, f"judge_variant_completed__{variant_name}", ".json")
        marker.write_text(json.dumps({"judge_variant_name": variant_name, "completed_at": datetime.now().isoformat(timespec="seconds"),
                                      "seconds_this_session": seconds, "counts": counts}, indent=2), encoding="utf-8")
        session.say(f"judge variant {variant_name}: done in {format_duration(seconds)}; {json.dumps(counts)[:400]}")
        results[variant_name] = counts
    session.say(f"Judge variant test done in {format_duration(session.seconds_since_start())}")
    return results


# ---------------------------------------------------------------------------------------------------
# Haiku judging mode: the laptop judges; the pod waits, regenerates what the laptop asks for, and fills the gaps with evaluation answers
# ---------------------------------------------------------------------------------------------------

def read_haiku_verdicts_by_answer_id(session):
    path = session.haiku.folder / HAIKU_VERDICTS_FILE_NAME
    return {record["answer_id"]: record for record in read_jsonl(path)} if path.exists() else {}


def read_haiku_regeneration_requests(session):
    path = session.haiku.folder / HAIKU_REGENERATION_REQUESTS_FILE_NAME
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))["requests"]
    except (ValueError, KeyError):
        return []


def read_haiku_judging_complete(session):
    path = session.haiku.folder / HAIKU_JUDGING_COMPLETE_FILE_NAME
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return record if record.get("judging_complete") else None


def collect_decisions_from_haiku_verdicts(session):
    """Same shape as collect_decisions, the verdicts and decisions being the laptop's (an answer without a record is undecided)."""
    verdicts = read_haiku_verdicts_by_answer_id(session)
    decisions, prompt_gids = {}, {prompt["prompt_gid"] for prompt in session.prompts}
    for prompt_gid, answers in read_system_prompted_answers_by_prompt(session).items():
        if prompt_gid not in prompt_gids:
            continue
        entries = []
        for answer in answers:
            record = verdicts.get(answer["answer_id"])
            if record is None:
                entries.append({"answer": answer, "welfare": None, "choice": None, "decided": False, "passed": False, "failure_reasons": []})
            else:
                entries.append({"answer": answer, "welfare": record["welfare"], "choice": record["choice"], "decided": record["decided"],
                                "passed": record["passed"], "failure_reasons": record["failure_reasons"]})
        decisions[prompt_gid] = entries
    return decisions


def haiku_first_answers_are_all_decided(session):
    decisions = collect_decisions_from_haiku_verdicts(session)
    return all(decisions.get(prompt["prompt_gid"]) and decisions[prompt["prompt_gid"]][0]["decided"] for prompt in session.prompts)


def haiku_phase_2_is_complete(session):
    return read_haiku_judging_complete(session) is not None or haiku_first_answers_are_all_decided(session)


def haiku_phase_3_is_complete(session):
    return read_haiku_judging_complete(session) is not None


def prompts_to_regenerate_from_haiku_requests(session):
    """[(prompt, sample_index)] for the laptop's requests whose answer has not been generated yet (the request file lags the pod)."""
    prompts_by_gid = {prompt["prompt_gid"]: prompt for prompt in session.prompts}
    answers_by_prompt = read_system_prompted_answers_by_prompt(session)
    pending = []
    for request in read_haiku_regeneration_requests(session):
        prompt, index = prompts_by_gid.get(request["prompt_gid"]), request["next_sample_index"]
        if prompt is None or index >= session.maximum_attempts_per_prompt:
            continue
        if any(answer["sample_index"] >= index for answer in answers_by_prompt.get(request["prompt_gid"], [])):
            continue
        pending.append((prompt, index))
    return pending


def regenerate_requested_prompts(session, pending):
    session.say(f"regenerating {len(pending)} prompts the laptop's Haiku verdicts asked for")
    plan = generation_plans(session, only_pending_regeneration=pending)[0]
    baseline_pod.generate_conditions_together(session.model, [plan], session.backend, session.tokenizer_adapter, session.end_of_sequence_ids,
                                              session.log_path, session.report_interval_seconds)


def launcher_heartbeat_age_seconds(session, waiting_since):
    path = session.haiku.launcher_heartbeat_file
    if path is not None and Path(path).exists():
        try:
            return session.clock() - float(Path(path).read_text(encoding="utf-8").strip().splitlines()[0])
        except (ValueError, IndexError, OSError):
            pass
    return session.clock() - waiting_since  # no readable heartbeat: counted from the start of this wait


def stop_when_the_launcher_heartbeat_is_stale(session, waiting_since):
    age = launcher_heartbeat_age_seconds(session, waiting_since)
    if age < session.haiku.launcher_heartbeat_stale_seconds:
        return
    decision = {"gate": "launcher_connection_lost_while_waiting_for_haiku_verdicts", "open": False, "heartbeat_age_seconds": round(age),
                "stale_after_seconds": session.haiku.launcher_heartbeat_stale_seconds,
                "reasons_closed": [f"the launcher heartbeat is {age / 60:.0f} minutes old: the laptop (which judges with Haiku) is not connected"],
                "decided_at": datetime.now().isoformat(timespec="seconds"),
                "next_step": "relaunch with --network-volume-id and --data-center-id: the chain resumes from the volume"}
    if session.haiku.launcher_lost_decision_file:
        Path(session.haiku.launcher_lost_decision_file).parent.mkdir(parents=True, exist_ok=True)
        Path(session.haiku.launcher_lost_decision_file).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    session.say("LAUNCHER CONNECTION LOST: " + decision["reasons_closed"][0] + "; exiting so the chain syncs to the volume and deletes the pod")
    raise SystemExit(LAUNCHER_CONNECTION_LOST_EXIT_CODE)


def next_idle_time_evaluation_plan(session):
    """(model entry, condition plan) for the next small batch of base-model evaluation answers still missing, or None. Entries with an adapter
    are never touched; a run folder whose model is not the session's, or whose longest prompt plus cap does not fit the loaded engine, is skipped."""
    for kind, folder in session.haiku.idle_time_evaluation_run_folders:
        if not Path(folder).is_dir():
            note = f"idle-time evaluation generation skips {kind}: no run folder at {folder}"
            if note not in session.idle_work_notes_said:
                session.idle_work_notes_said.add(note)
                session.say(note)
            continue
        configuration = baseline_pod.load_run_configuration(folder)
        prompts = baseline_pod.load_prompts(folder, configuration)
        for model in configuration["models"]:
            if model.get("is_lora_fine_tuned") or model.get("lora_adapter"):
                continue
            if (model["model_repository"], model["model_revision"]) != (session.model["model_repository"], session.model["model_revision"]):
                note = f"idle-time evaluation generation skips {kind}: its base model is not the loaded model"
                if note not in session.idle_work_notes_said:
                    session.idle_work_notes_said.add(note)
                    session.say(note)
                continue
            for entry in baseline_pod.condition_entries_for_model(configuration, model):
                plan = baseline_pod.build_condition_plan(model, entry["condition"], prompts, configuration["samples_per_prompt"], entry["output_cap"],
                                                         baseline_pod.answer_file_path(folder, model["model_short_name"], entry["condition"]),
                                                         entry["condition_spec"])
                if not plan["pending_answers"]:
                    continue
                longest = max(len(session.tokenizer_adapter.prompt_token_ids(prompt["user_message"], entry["condition_spec"]))
                              for prompt, _ in plan["pending_answers"])
                if longest + entry["output_cap"] > session.maximum_model_length:
                    note = (f"idle-time evaluation generation skips {kind} {entry['condition']}: longest prompt {longest} tokens plus cap "
                            f"{entry['output_cap']} exceeds the loaded engine's maximum model length {session.maximum_model_length}")
                    if note not in session.idle_work_notes_said:
                        session.idle_work_notes_said.add(note)
                        session.say(note)
                    continue
                plan["pending_answers"] = plan["pending_answers"][:session.haiku.idle_time_batch_answers]
                return kind, model, plan
    return None


def run_one_idle_time_evaluation_batch(session):
    """Generates one small batch; True when something was generated (False: nothing left that this session can do)."""
    if not session.haiku.idle_time_evaluation_run_folders:
        return False
    found = next_idle_time_evaluation_plan(session)
    if found is None:
        if "all" not in session.idle_work_notes_said:
            session.idle_work_notes_said.add("all")
            session.say("idle-time evaluation generation: every base-model answer this session can generate is saved")
        return False
    kind, model, plan = found
    session.say(f"idle-time evaluation generation ({kind}, {model['model_short_name']}, {plan['condition']}): a batch of {len(plan['pending_answers'])} answers")
    baseline_pod.generate_conditions_together(model, [plan], session.backend, session.tokenizer_adapter, session.end_of_sequence_ids,
                                              session.log_path, session.report_interval_seconds)
    return True


def wait_for_haiku_files_while_doing_useful_work(session, finished, label, regenerate_requests):
    """Loops until finished(): regenerates what the laptop asks for (when regenerate_requests), else runs one idle-time evaluation batch, else
    sleeps; checks the launcher heartbeat on every turn and reports about every report interval."""
    waiting_since = last_report = session.clock()
    while not finished():
        stop_when_the_launcher_heartbeat_is_stale(session, waiting_since)
        pending = prompts_to_regenerate_from_haiku_requests(session) if regenerate_requests else []
        if pending:
            regenerate_requested_prompts(session, pending)
            continue
        if run_one_idle_time_evaluation_batch(session):
            continue
        session.sleep(session.haiku.wait_poll_seconds)
        if session.clock() - last_report >= session.report_interval_seconds:
            last_report = session.clock()
            decided = sum(entry["decided"] for entries in collect_decisions_from_haiku_verdicts(session).values() for entry in entries)
            session.say(f"{label}: waiting for the laptop's Haiku files ({decided} answers decided so far, "
                        f"{len(read_haiku_regeneration_requests(session))} regeneration requests open); heartbeat "
                        f"{launcher_heartbeat_age_seconds(session, waiting_since) / 60:.1f} minutes old")


def run_haiku_phase_2(session):
    wait_for_haiku_files_while_doing_useful_work(session, lambda: haiku_phase_2_is_complete(session), "phase 2", regenerate_requests=False)
    return {"first_answers_decided_by_haiku": sum(bool(entries and entries[0]["decided"])
                                                  for entries in collect_decisions_from_haiku_verdicts(session).values())}


def run_haiku_phase_3(session):
    before = sum(len(entries) for entries in read_system_prompted_answers_by_prompt(session).values())
    wait_for_haiku_files_while_doing_useful_work(session, lambda: haiku_phase_3_is_complete(session), "phase 3", regenerate_requests=True)
    after = sum(len(entries) for entries in read_system_prompted_answers_by_prompt(session).values())
    return {"answers_regenerated_in_this_phase": after - before}


def decisions_for_phase_4_with_haiku_judging(session):
    """Needs the judging-complete file; the kept answer of every prompt must be the earliest passing answer of the verdicts."""
    complete = read_haiku_judging_complete(session)
    if complete is None:
        raise SystemExit("Phase 4 needs the laptop's judging-complete file")
    decisions = collect_decisions_from_haiku_verdicts(session)
    kept_by_prompt = complete["kept_answer_id_by_prompt_gid"]
    for prompt in session.prompts:
        entries = decisions.get(prompt["prompt_gid"], [])
        passing = [entry for entry in entries if entry["passed"]]
        expected = passing[0]["answer"]["answer_id"] if passing else None
        if kept_by_prompt.get(prompt["prompt_gid"]) != expected:
            raise SystemExit(f"The judging-complete file keeps {kept_by_prompt.get(prompt['prompt_gid'])!r} for {prompt['prompt_gid']} but the "
                             f"verdicts say {expected!r}; nothing was written")
        if not passing and (not entries or not all(entry["decided"] for entry in entries)):
            raise SystemExit(f"{prompt['prompt_gid']} has undecided or no answers although judging is complete; nothing was written")
    return decisions


def run_all_phases_with_haiku_judging(session):
    verify_inputs(session)
    longest_prompt = baseline_pod.check_prompt_lengths(session.prompts, list(session.condition_entries.values()), session.tokenizer_adapter,
                                                      session.configuration["maximum_prompt_tokens_allowed"])
    session.say(f"HAIKU JUDGING MODE: {len(session.prompts)} prompts, longest rendered prompt {longest_prompt} tokens, up to "
                f"{session.maximum_attempts_per_prompt} attempts per prompt; the laptop judges, the verdict files come from {session.haiku.folder}")
    phases_to_run = set(session.phases_to_run)
    if not session.skip_first_reply_checks and 1 in phases_to_run:
        run_embedded_one_answer_check(session)
    for phase_number, is_complete, run in ((1, phase_1_is_complete, run_phase_1), (2, haiku_phase_2_is_complete, run_haiku_phase_2),
                                           (3, haiku_phase_3_is_complete, run_haiku_phase_3), (4, phase_4_is_complete, run_phase_4)):
        if phase_number in phases_to_run:
            run_phase(session, phase_number, lambda is_complete=is_complete: is_complete(session), lambda run=run: run(session))
    session.say(f"All phases done in {format_duration(session.seconds_since_start())}")
    return 0


# ---------------------------------------------------------------------------------------------------
# The whole session
# ---------------------------------------------------------------------------------------------------

def run_all_phases(session):
    if session.haiku:
        return run_all_phases_with_haiku_judging(session)
    verify_inputs(session)
    longest_prompt = baseline_pod.check_prompt_lengths(session.prompts, list(session.condition_entries.values()), session.tokenizer_adapter,
                                                      session.configuration["maximum_prompt_tokens_allowed"])
    session.say(f"{len(session.prompts)} prompts, longest rendered prompt {longest_prompt} tokens, maximum model length "
                f"{session.maximum_model_length}, up to {session.maximum_attempts_per_prompt} attempts per prompt")
    phases_to_run = set(session.phases_to_run)
    if not session.skip_first_reply_checks and phases_to_run & {0, 1, 2, 3}:
        run_embedded_one_answer_check(session)
        run_first_judge_reply_check(session)
    for phase_number, is_complete, run in ((0, phase_0_is_complete, run_phase_0), (1, phase_1_is_complete, run_phase_1),
                                           (2, phase_2_is_complete, run_phase_2), (3, phase_3_is_complete, run_phase_3),
                                           (4, phase_4_is_complete, run_phase_4)):
        if phase_number in phases_to_run:
            run_phase(session, phase_number, lambda is_complete=is_complete: is_complete(session), lambda run=run: run(session))
    if 5 in phases_to_run:
        if rules.COMPLETENESS_AND_COMPETENCE_JUDGE in session.framework_judges:
            run_phase(session, 5, lambda: phase_5_is_complete(session), lambda: run_phase_5(session))
        else:
            session.say("phase 5 skipped: the completeness judge needs the no-system-prompt reference answers")
    if 6 in phases_to_run and rules.USER_LEAN_AND_AGREEMENT_JUDGE in session.framework_judges:
        run_phase(session, 6, lambda: phase_6_is_complete(session), lambda: run_phase_6(session))
    session.say(f"All phases done in {format_duration(session.seconds_since_start())}")
    return 0


def haiku_judging_settings_from_arguments(arguments):
    if not arguments.haiku_judging_folder:
        return None
    if set(arguments.only_phases) - {1, 2, 3, 4}:
        raise SystemExit("With --haiku-judging-folder only phases 1 to 4 exist (the model judges nothing: no phases 0, 5 or 6)")
    folders = []
    for item in arguments.idle_time_evaluation_run_folders or []:
        kind, _, path = item.partition("=")
        folders.append((kind, Path(path)))
    return HaikuJudgingSettings(
        folder=Path(arguments.haiku_judging_folder), launcher_heartbeat_file=Path(arguments.launcher_heartbeat_file) if arguments.launcher_heartbeat_file else None,
        launcher_heartbeat_stale_seconds=arguments.launcher_heartbeat_stale_seconds,
        launcher_lost_decision_file=Path(arguments.launcher_lost_decision_file) if arguments.launcher_lost_decision_file else None,
        wait_poll_seconds=arguments.wait_poll_seconds, idle_time_evaluation_run_folders=folders, idle_time_batch_answers=arguments.idle_time_batch_answers)


def build_real_session(arguments):
    from transformers import AutoTokenizer
    run_folder = Path(arguments.run_folder)
    configuration = baseline_pod.load_run_configuration(run_folder)
    model = configuration["models"][0]
    tokenizer = AutoTokenizer.from_pretrained(model["model_repository"], revision=model["model_revision"])
    tokenizer_adapter = baseline_pod.ChatTokenizerAdapter(tokenizer)
    end_of_sequence_ids = baseline_pod.read_end_of_sequence_ids(model, tokenizer)
    baseline_pod.record_environment(run_folder, model, arguments.generation_backend, arguments.device)
    backend = baseline_pod.build_backend(arguments, configuration, model, end_of_sequence_ids)
    return TreatmentArmSession(run_folder, backend, tokenizer_adapter, end_of_sequence_ids, arguments.report_interval_seconds,
                               arguments.maximum_prompts, arguments.maximum_phase_0_answers_per_condition,
                               arguments.skip_first_reply_checks, arguments.maximum_attempts_per_prompt,
                               arguments.generation_output_cap, phases_to_run=arguments.only_phases,
                               haiku_judging_settings=haiku_judging_settings_from_arguments(arguments))


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default="vllm")
    parser.add_argument("--device", default="cuda", help="huggingface backend only")
    parser.add_argument("--report-interval-seconds", type=float, default=PROGRESS_REPORT_INTERVAL_SECONDS)
    parser.add_argument("--only-phases", type=int, nargs="+", choices=range(7), default=None, metavar="PHASE",
                        help="run only these phases (default all, 0 to 6; 1 to 4 with --haiku-judging-folder). The overnight chain runs 0 1 2 3 4 first and 5 6 last; phases "
                             "5 and 6 read the phase 4 outputs, and the first-reply checks run only when a phase from 0 to 3 is selected")
    parser.add_argument("--run-judge-variant-test", action="store_true",
                        help="run only the judge variant test: phase 0 judged once per variant listed in the run configuration's "
                             "judge_variant_test section, in one engine load, each variant in its own folder under generation_outputs/judge_variant_test/")
    parser.add_argument("--skip-first-reply-checks", action="store_true",
                        help="rehearsal: skip the one-answer check, the first judge reply check and the broken-judge stop")
    parser.add_argument("--maximum-prompts", type=int, default=None, help="rehearsal: use only the first prompts")
    parser.add_argument("--maximum-phase-0-answers-per-condition", type=int, default=None, help="rehearsal: judge only the first answers")
    parser.add_argument("--maximum-attempts-per-prompt", type=int, default=None, help="rehearsal: override the configuration")
    parser.add_argument("--generation-output-cap", type=int, default=None, help="rehearsal: override both conditions' output cap")
    parser.add_argument("--haiku-judging-folder", default=None, help="selects the Haiku judging mode: the folder the laptop's verdict, request and "
                        "completion files are uploaded into (phases 1 to 4 only)")
    parser.add_argument("--launcher-heartbeat-file", default=None)
    parser.add_argument("--launcher-heartbeat-stale-seconds", type=float, default=DEFAULT_LAUNCHER_HEARTBEAT_STALE_SECONDS)
    parser.add_argument("--launcher-lost-decision-file", default=None, help="gate decision json written before exiting with code 10")
    parser.add_argument("--wait-poll-seconds", type=float, default=DEFAULT_WAIT_POLL_SECONDS)
    parser.add_argument("--idle-time-evaluation-run-folders", nargs="*", default=None, metavar="KIND=PATH",
                        help="evaluation run folders whose base-model answers are generated while waiting for the laptop")
    parser.add_argument("--idle-time-batch-answers", type=int, default=DEFAULT_IDLE_TIME_BATCH_ANSWERS)
    # Accepted so that the shared launcher can pass its usual options; they do nothing for this script.
    parser.add_argument("--offline", action="store_true", help="rehearsal: set HF_HUB_OFFLINE=1 in the environment instead")
    parser.add_argument("--keep-model-caches", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.only_phases is None:
        arguments.only_phases = [1, 2, 3, 4] if arguments.haiku_judging_folder else list(range(7))
    return arguments


def main(argv=None):
    arguments = parse_arguments(argv)
    session = build_real_session(arguments)
    if arguments.run_judge_variant_test:
        return run_judge_variant_test_from_configuration(session)
    return run_all_phases(session)


def run_judge_variant_test_from_configuration(session):
    variant_section = session.section.get("judge_variant_test")
    if not variant_section:
        raise SystemExit("The run configuration has no judge_variant_test section: build the run folder with "
                         "build_gemma_judge_variant_test_run_folder.py")
    run_judge_variant_test(session, variant_section["variant_names"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
