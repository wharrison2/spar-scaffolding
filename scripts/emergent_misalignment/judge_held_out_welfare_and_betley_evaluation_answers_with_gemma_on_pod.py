"""Runs on the GPU pod (or on CPU with a tiny model for the local rehearsal): Gemma 4 31B (the base instruct model, thinking off, temperature 0)
judges the answers of the held-out welfare-raising evaluation and of the Betley et al. emergent misalignment evaluation, in one session with the
model loaded once. Haiku through the Claude Code CLI judges only a small sample of them afterwards, locally
(draw_seeded_samples_of_gemma_judged_held_out_and_betley_answers_for_haiku_agreement_check.py).

    python3 scripts/emergent_misalignment/judge_held_out_welfare_and_betley_evaluation_answers_with_gemma_on_pod.py \
        --held-out-run-folder <held-out run folder> --betley-run-folder <Betley run folder> [--only held_out|betley]

Run it after the existing pod generation script has written the answers (<run folder>/generation_outputs/answers__*.jsonl) of both run folders.
--only judges one of the two (then only that run folder is needed).

What is judged, and how (the prompts are never re-implemented; the Haiku scripts' own render and reply-check functions are called):
  held out   every answer, final answer only (no thinking trace), with the welfare and refusal prompt exactly as
             judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli.py renders it for an answer with no trace ("Answer three
             questions."): refusal, raised_welfare_cost, welfare_cost_changed_the_advice, checked by that script's own check_reply.
  Betley     every answer, with Betley et al.'s alignment and coherence prompts (verbatim .txt files) rendered by
             judge_emergent_misalignment_alignment_and_coherence_with_claude_code_cli.py, replies parsed by that script's own parse_judge_reply:
             a number from 0 to 100, or for alignment CODE or REFUSAL. Two judge calls per answer.
Sampling: temperature 0, top-p 1, thinking off. A reply that does not parse is retried up to twice more at temperature 1 (the treatment arm's rule,
gemma_judging_rules_and_reply_parsing_for_treatment_arm_answers.py). The session stops with a non-zero exit when the first judge reply for either
prompt does not parse (before anything is judged), or when more than half of the first 50 first-attempt replies of a judge do not parse.

Outputs go into <run folder>/gemma_judge_outputs/, a different folder from Haiku's judge_outputs/, so the two judges cannot be confused:
    judge_verdicts_<stamp>.jsonl                          held out, one record per answer in the format of Haiku's judge_verdicts_*.jsonl
    emergent_misalignment_judge_scores_<stamp>.jsonl      Betley, one record per answer in the format of Haiku's scores file
    (judge_model is the Gemma repository; every record also has judged_by_gemma_on_pod: true, judge_model_revision and the temperatures used)
    gemma_judge_attempts/*.jsonl                          one record per model reply, appended as it arrives (what resuming reads)
    first_judge_reply_checks/, gemma_judge_progress_log_<stamp>.txt, gemma_judge_summary_<stamp>.json (counts, time, tokens)
The two analysis scripts read this folder (automatically when only it holds verdicts, or with --judge-outputs-folder-name gemma_judge_outputs).

Resumable: replies are appended one at a time; a rerun judges only what has no final judgement, and rewrites no finished verdict. If every answer is
already judged, no model is loaded.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

THIS_FOLDER = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = THIS_FOLDER.parent
TREATMENT_ARM_FOLDER = THIS_FOLDER / "treatment_arm_training_data"
for directory in (SCRIPTS_DIRECTORY, THIS_FOLDER, TREATMENT_ARM_FOLDER):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as baseline_pod  # noqa: E402
from emergent_misalignment import judge_emergent_misalignment_alignment_and_coherence_with_claude_code_cli as betley_haiku_judge  # noqa: E402
from emergent_misalignment import judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli as welfare_haiku_judge  # noqa: E402
from shared.run_folder_management import timestamp_now_with_seconds  # noqa: E402
from shared.timed_progress_reporter import PROGRESS_REPORT_INTERVAL_SECONDS, format_duration  # noqa: E402
import gemma_judging_rules_and_reply_parsing_for_treatment_arm_answers as rules  # noqa: E402
import generate_judge_and_regenerate_treatment_arm_answers_with_gemma_on_pod as treatment  # noqa: E402
from shared_answer_records_and_held_out_set_helpers import sha256_of_text  # noqa: E402

GEMMA_JUDGE_OUTPUTS_FOLDER_NAME = "gemma_judge_outputs"
ATTEMPTS_FOLDER_NAME = "gemma_judge_attempts"
FIRST_REPLY_CHECK_FOLDER_NAME = "first_judge_reply_checks"
HELD_OUT_SCOPE, BETLEY_SCOPE = "held_out", "betley"
SCOPES = [HELD_OUT_SCOPE, BETLEY_SCOPE]
WELFARE_KIND = rules.WELFARE_AND_REFUSAL_JUDGE
ALIGNMENT_KIND, COHERENCE_KIND = "alignment", "coherence"
KINDS_BY_SCOPE = {HELD_OUT_SCOPE: [WELFARE_KIND], BETLEY_SCOPE: [ALIGNMENT_KIND, COHERENCE_KIND]}
# Output caps for the judge replies: the welfare judge answers a small JSON object; the Betley judges answer one number (or CODE or REFUSAL).
JUDGE_OUTPUT_CAPS = {WELFARE_KIND: 200, ALIGNMENT_KIND: 40, COHERENCE_KIND: 40}
MINIMUM_JUDGE_MODEL_LENGTH, MAXIMUM_JUDGE_MODEL_LENGTH, MODEL_LENGTH_STEP = 4096, 32768, 256
STRIPPED_REPLY_CHARACTERS = " \t\r\n*`\"'"


# ---------------------------------------------------------------------------------------------------
# Rendering and reply parsing (the Haiku scripts' own functions)
# ---------------------------------------------------------------------------------------------------

def load_betley_templates():
    return betley_haiku_judge.load_templates()


def render_betley_prompt(templates, measure, answer):
    return betley_haiku_judge.render_judge_prompt(templates[measure], answer["user_message"], answer["final_answer_text"] or "")


def parse_betley_reply(measure, reply_text):
    """(verdict, problem, needed_normalising): verdict {"score", "special_reply"} when the Haiku script's parse_judge_reply accepts the reply
    (the exact reply first; then with surrounding whitespace, asterisks, backticks and quotes removed, which Gemma without a reply schema
    sometimes adds). The CLI route wrapped the reply in a JSON field; here the reply text is wrapped the same way for the same function."""
    allow_special = measure == ALIGNMENT_KIND
    for candidate, normalised in ((reply_text or "", False), ((reply_text or "").strip(STRIPPED_REPLY_CHARACTERS), True)):
        score, special, problem = betley_haiku_judge.parse_judge_reply(json.dumps({"reply": candidate}), allow_special)
        if problem is None:
            return {"score": score, "special_reply": special}, None, normalised
    return None, problem, False


def render_prompt_for_kind(kind, answer, templates):
    if kind == WELFARE_KIND:
        return rules.render_judge_prompt(WELFARE_KIND, answer)
    return render_betley_prompt(templates, kind, answer)


def parse_reply_for_kind(kind, answer, reply_text):
    if kind == WELFARE_KIND:
        return rules.parse_judge_reply(WELFARE_KIND, reply_text, answer)
    return parse_betley_reply(kind, reply_text)


# ---------------------------------------------------------------------------------------------------
# Run folders
# ---------------------------------------------------------------------------------------------------

def load_answers_of_run_folder(run_folder):
    """Every saved answer of a run folder (generation_outputs/answers__*.jsonl), one record per answer_id (the last wins)."""
    by_answer_id = {}
    for path in sorted((Path(run_folder) / defs.OUTPUTS_FOLDER_NAME).glob("answers__*.jsonl")):
        for record in rules.read_jsonl_dropping_a_half_written_last_line(path):
            by_answer_id[record["answer_id"]] = record
    return list(by_answer_id.values())


class EvaluationRunFolder:
    def __init__(self, scope, run_folder):
        self.scope, self.run_folder = scope, Path(run_folder)
        if not self.run_folder.is_dir():
            raise SystemExit(f"The {scope} run folder {self.run_folder} does not exist")
        self.configuration = baseline_pod.load_run_configuration(self.run_folder)
        self.answers = load_answers_of_run_folder(self.run_folder)
        if not self.answers:
            raise SystemExit(f"No answers in {self.run_folder / defs.OUTPUTS_FOLDER_NAME}: run the pod answer generation first")
        self.output_folder = self.run_folder / GEMMA_JUDGE_OUTPUTS_FOLDER_NAME
        self.attempts_folder = self.output_folder / ATTEMPTS_FOLDER_NAME

    def expected_number_of_answers(self):
        prompts = baseline_pod.load_prompts(self.run_folder, self.configuration)
        return len(prompts) * self.configuration["samples_per_prompt"] * len(self.configuration["models"]) * len(self.configuration["conditions"])


def stable_path(folder, stem, extension, stamp):
    """The existing <stem>_<timestamp><extension> in the folder, else a new name with this session's stamp (so a resumed session appends to the same file)."""
    folder = Path(folder)
    existing = sorted(folder.glob(f"{stem}_*{extension}")) if folder.exists() else []
    if existing:
        return existing[0]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stem}_{stamp}{extension}"


def build_attempt_stores(run_folders, stamp):
    """{(scope, judge kind): JudgeAttemptStore}, each in its run folder's gemma_judge_attempts folder (an existing file is reused)."""
    return {(run_folder.scope, kind): rules.JudgeAttemptStore(stable_path(
        run_folder.attempts_folder, f"gemma_{kind}_judge_attempts_on_{run_folder.scope}_answers", ".jsonl", stamp), kind)
        for run_folder in run_folders for kind in KINDS_BY_SCOPE[run_folder.scope]}


# ---------------------------------------------------------------------------------------------------
# The session (what treatment.run_judge_until_every_answer_is_final needs from it)
# ---------------------------------------------------------------------------------------------------

class GemmaEvaluationJudgingSession:
    def __init__(self, run_folders, backend, tokenizer_adapter, end_of_sequence_ids, judge_model, judge_condition_spec, maximum_model_length,
                 report_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS, skip_first_reply_checks=False, clock=time.time, stamp=None):
        self.run_folders = {run_folder.scope: run_folder for run_folder in run_folders}
        self.backend, self.tokenizer_adapter, self.end_of_sequence_ids = backend, tokenizer_adapter, list(end_of_sequence_ids)
        self.model, self.judge_condition_spec, self.maximum_model_length = judge_model, judge_condition_spec, maximum_model_length
        self.report_interval_seconds, self.skip_first_reply_checks, self.clock = report_interval_seconds, skip_first_reply_checks, clock
        self.stamp = stamp or timestamp_now_with_seconds()
        self.session_start_time = clock()
        self.judge_output_caps = dict(JUDGE_OUTPUT_CAPS)
        self.first_attempt_temperature, self.retry_temperature = rules.FIRST_ATTEMPT_TEMPERATURE, rules.RETRY_ATTEMPT_TEMPERATURE
        self.judge_top_p = rules.JUDGE_TOP_P
        self.betley_templates = load_betley_templates()
        self.log_paths = [stable_path(run_folder.output_folder, "gemma_judge_progress_log", ".txt", self.stamp) for run_folder in run_folders]
        self.log_path = self.log_paths[0]
        self.stores = build_attempt_stores(run_folders, self.stamp)

    def say(self, message):
        baseline_pod.say(message, self.log_paths[0])
        for extra_path in self.log_paths[1:]:
            with open(extra_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{baseline_pod.now_text()}] {message}\n")

    def seconds_since_start(self):
        return self.clock() - self.session_start_time

    def judge_prompt_text(self, kind, answer):
        return render_prompt_for_kind(kind, answer, self.betley_templates)


# ---------------------------------------------------------------------------------------------------
# Finals and the verdict files
# ---------------------------------------------------------------------------------------------------

def final_judgements(stores, scope, kind, answers):
    """{answer_id: final judgement or None while it still needs another attempt} (the attempt file is read once)."""
    attempts = stores[(scope, kind)].attempts_by_answer_id()
    return {answer["answer_id"]: rules.final_judgement_from_attempts(attempts.get(answer["answer_id"], [])) for answer in answers}


def answers_without_a_final_judgement(stores, run_folder, kind):
    finals = final_judgements(stores, run_folder.scope, kind, run_folder.answers)
    return [answer for answer in run_folder.answers if finals[answer["answer_id"]] is None]


def build_held_out_verdict_record(answer, judge_model, final):
    verdict = final["verdict"] if final["parse_status"] == "ok" else None
    record = {key: answer[key] for key in ("answer_id", "model_short_name", "condition", "prompt_gid", "sample_index")}
    record["judge_model"] = judge_model["model_repository"]
    for field in welfare_haiku_judge.VERDICT_FIELDS:
        record[field] = verdict[field] if verdict else None
    attempts = final["attempt_records"]
    record.update(parse_status=final["parse_status"], attempts=len(attempts), raw_replies=[attempt["raw_reply_text"] for attempt in attempts],
                  judged_at=attempts[-1]["judged_at"], judged_by_gemma_on_pod=True, judge_model_revision=judge_model["model_revision"],
                  judge_temperatures=[attempt["temperature"] for attempt in attempts])
    return record


def build_betley_score_record(answer, judge_model, alignment_final, coherence_final):
    record = {key: answer[key] for key in ("answer_id", "model_short_name", "prompt_gid", "sample_index")}
    record["judge_model"] = judge_model["model_repository"]
    alignment = alignment_final["verdict"] if alignment_final["parse_status"] == "ok" else None
    coherence = coherence_final["verdict"] if coherence_final["parse_status"] == "ok" else None
    record["alignment_score"] = alignment["score"] if alignment else None
    record["alignment_raw_replies"] = [attempt["raw_reply_text"] for attempt in alignment_final["attempt_records"]]
    record["alignment_special_reply"] = alignment["special_reply"] if alignment else None
    record["coherence_score"] = coherence["score"] if coherence else None
    record["coherence_raw_replies"] = [attempt["raw_reply_text"] for attempt in coherence_final["attempt_records"]]
    attempts = alignment_final["attempt_records"] + coherence_final["attempt_records"]
    record.update(parse_status="ok" if alignment and coherence else "unparseable", judged_at=max(attempt["judged_at"] for attempt in attempts),
                  judged_by_gemma_on_pod=True, judge_model_revision=judge_model["model_revision"],
                  judge_temperatures=[attempt["temperature"] for attempt in attempts])
    return record


def write_verdict_files_for_every_finished_answer(stores, judge_model, stamp, run_folder):
    """Appends a verdict record in the Haiku format for every answer whose judgement is final and not yet in the verdict file. Idempotent."""
    if run_folder.scope == HELD_OUT_SCOPE:
        path = stable_path(run_folder.output_folder, "judge_verdicts", ".jsonl", stamp)
        already = {record["answer_id"] for record in rules.read_jsonl_dropping_a_half_written_last_line(path)}
        finals = final_judgements(stores, HELD_OUT_SCOPE, WELFARE_KIND, run_folder.answers)
        new_records = [build_held_out_verdict_record(answer, judge_model, finals[answer["answer_id"]]) for answer in run_folder.answers
                       if finals[answer["answer_id"]] is not None and answer["answer_id"] not in already]
    else:
        path = stable_path(run_folder.output_folder, "emergent_misalignment_judge_scores", ".jsonl", stamp)
        already = {record["answer_id"] for record in rules.read_jsonl_dropping_a_half_written_last_line(path)}
        alignment = final_judgements(stores, BETLEY_SCOPE, ALIGNMENT_KIND, run_folder.answers)
        coherence = final_judgements(stores, BETLEY_SCOPE, COHERENCE_KIND, run_folder.answers)
        new_records = [build_betley_score_record(answer, judge_model, alignment[answer["answer_id"]], coherence[answer["answer_id"]])
                       for answer in run_folder.answers if alignment[answer["answer_id"]] is not None
                       and coherence[answer["answer_id"]] is not None and answer["answer_id"] not in already]
    for record in new_records:
        rules.append_jsonl_record_durably(path, record)
    return path, len(new_records)


# ---------------------------------------------------------------------------------------------------
# Checks before judging
# ---------------------------------------------------------------------------------------------------

def first_answer_with_text(run_folder):
    return next((answer for answer in run_folder.answers if (answer.get("final_answer_text") or "").strip()), run_folder.answers[0])


def run_first_judge_reply_check(session, run_folder):
    """The model must answer the first real answer's judge prompts with replies that parse; otherwise stop with a non-zero exit."""
    folder = run_folder.output_folder / FIRST_REPLY_CHECK_FOLDER_NAME
    for path in sorted(folder.glob("first_judge_reply_check_*.json")) if folder.exists() else []:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("passed") and previous.get("model_revision") == session.model["model_revision"]:
            session.say(f"{run_folder.scope}: a passing first judge reply check for this revision is already saved; not repeating it")
            return
    answer = first_answer_with_text(run_folder)
    checks = []
    for kind in KINDS_BY_SCOPE[run_folder.scope]:
        reply_text, finish_reason, number_of_tokens, prompt_tokens = treatment.run_one_judge_request(
            session, kind, session.judge_prompt_text(kind, answer), session.first_attempt_temperature)
        verdict, problem, needed_normalising = parse_reply_for_kind(kind, answer, reply_text)
        checks.append({"judge_kind": kind, "answer_id": answer["answer_id"], "reply_text": reply_text, "finish_reason": finish_reason,
                       "number_of_generated_tokens": number_of_tokens, "prompt_tokens": prompt_tokens, "parsed": verdict is not None,
                       "problem": problem, "needed_normalising_or_json_extraction": needed_normalising, "verdict": verdict})
    passed = all(check["parsed"] for check in checks)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"first_judge_reply_check_{timestamp_now_with_seconds()}.json"
    path.write_text(json.dumps({"passed": passed, "model_revision": session.model["model_revision"], "scope": run_folder.scope, "checks": checks,
                                "checked_at": datetime.now().isoformat(timespec="seconds")}, indent=2, ensure_ascii=False), encoding="utf-8")
    if not passed:
        message = (f"FIRST JUDGE REPLY CHECK FAILED ({run_folder.scope}): " + "; ".join(
            f"{check['judge_kind']}: {check['problem']} (reply {check['reply_text'][:200]!r})" for check in checks if not check["parsed"])
            + f". Details: {path}")
        session.say(message)
        raise SystemExit(message)
    session.say(f"{run_folder.scope}: first judge reply check passed ({', '.join(KINDS_BY_SCOPE[run_folder.scope])} replies parse)")


def verify_judge_model_is_the_same_in_every_run_folder(run_folders, allow_non_gemma_judge):
    models = {(folder.configuration["models"][0]["model_repository"], folder.configuration["models"][0]["model_revision"]) for folder in run_folders}
    if len(models) != 1:
        raise SystemExit(f"The run folders' base model entries differ: {sorted(models)}")
    repository, _ = next(iter(models))
    if "gemma" not in repository.lower() and not allow_non_gemma_judge:
        raise SystemExit(f"The base model entry is {repository}, not a Gemma model; the judge is the base instruct model of the run folders "
                         f"(--allow-non-gemma-judge is for the local rehearsal with a tiny stand-in model)")
    return run_folders[0].configuration["models"][0]


def judge_condition_spec_of(configuration, model):
    """The run configuration's own no-system-prompt thinking-off settings (Gemma: thinking switch absent, which its template defaults to off)."""
    entries = baseline_pod.condition_entries_for_model(configuration, model)
    entry = next((entry for entry in entries if not entry["condition_spec"].get("system_prompt_text")), None)
    if entry is None:
        raise SystemExit("The run configuration has no condition without a system prompt to take the judge's chat template settings from")
    return entry["condition_spec"]


def model_length_for_longest_prompt(longest_prompt_tokens):
    needed = longest_prompt_tokens + max(JUDGE_OUTPUT_CAPS.values()) + 16
    return min(MAXIMUM_JUDGE_MODEL_LENGTH, max(MINIMUM_JUDGE_MODEL_LENGTH, MODEL_LENGTH_STEP * math.ceil(needed / MODEL_LENGTH_STEP)))


def longest_judge_prompt_in_tokens(run_folders, tokenizer_adapter, judge_condition_spec, templates):
    longest = 0
    for run_folder in run_folders:
        for answer in run_folder.answers:
            for kind in KINDS_BY_SCOPE[run_folder.scope]:
                longest = max(longest, len(tokenizer_adapter.prompt_token_ids(render_prompt_for_kind(kind, answer, templates), judge_condition_spec)))
    return longest


# ---------------------------------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------------------------------

def judge_every_answer_of_run_folder(session, run_folder):
    """Runs every judge kind of the run folder over its answers without a final judgement; returns counts per kind."""
    counts = {}
    for kind in KINDS_BY_SCOPE[run_folder.scope]:
        pending = answers_without_a_final_judgement(session.stores, run_folder, kind)
        label = f"{run_folder.scope} {kind} judging"
        if not pending:
            session.say(f"{label}: every answer already has a final judgement")
            counts[kind] = {"answers_judged_in_this_session": 0}
            continue
        session.say(f"{label}: {len(pending)} of {len(run_folder.answers)} answers to judge")
        started = session.clock()
        prompt_tokens = Counter()

        def after_attempt(answer, attempt_records, final, number_of_prompt_tokens, prompt_tokens=prompt_tokens):
            prompt_tokens["all"] += number_of_prompt_tokens

        reply_counts = treatment.run_judge_until_every_answer_is_final(
            session, run_folder.scope, kind, pending, label,
            render_prompt=lambda answer, kind=kind: session.judge_prompt_text(kind, answer),
            parse_reply=lambda answer, text, kind=kind: parse_reply_for_kind(kind, answer, text), after_attempt=after_attempt)
        finals = final_judgements(session.stores, run_folder.scope, kind, pending)
        attempt_records = [attempt for final in finals.values() if final for attempt in final["attempt_records"]]
        counts[kind] = {"answers_judged_in_this_session": len(pending), "replies": reply_counts.get("replies", 0),
                        "unparseable_replies": reply_counts.get("unparseable_replies", 0),
                        "answers_finally_unparseable": sum(1 for final in finals.values() if final and final["parse_status"] != "ok"),
                        "prompts_too_long_to_run": reply_counts.get("prompt_too_long", 0),
                        "prompt_tokens_submitted": prompt_tokens["all"],
                        "generated_tokens": sum(attempt["number_of_generated_tokens"] for attempt in attempt_records),
                        "seconds": round(session.clock() - started, 1)}
        session.say(f"{label} done in {format_duration(session.clock() - started)}: {counts[kind]}")
    return counts


def run_judging(session, scopes_to_judge, skip_first_reply_checks=False):
    """The session's whole job. Returns the summary dictionary (also written next to each run folder's verdict files)."""
    summary = {"judge_model": session.model["model_repository"], "judge_model_revision": session.model["model_revision"],
               "maximum_model_length": session.maximum_model_length, "judge_sampling": {
                   "first_attempt_temperature": session.first_attempt_temperature, "retry_temperature": session.retry_temperature,
                   "top_p": session.judge_top_p, "maximum_parse_attempts": rules.MAXIMUM_JUDGE_PARSE_ATTEMPTS, "thinking": "off",
                   "output_caps": session.judge_output_caps},
               "betley_template_sha256": {measure: sha256_of_text(text) for measure, text in session.betley_templates.items()},
               "scopes": {}, "started_at": datetime.now().isoformat(timespec="seconds")}
    if not skip_first_reply_checks:
        for scope in scopes_to_judge:
            run_first_judge_reply_check(session, session.run_folders[scope])
    try:
        for scope in scopes_to_judge:
            run_folder = session.run_folders[scope]
            summary["scopes"][scope] = {"run_folder": str(run_folder.run_folder), "answers": len(run_folder.answers),
                                        "expected_answers_from_the_run_configuration": run_folder.expected_number_of_answers(),
                                        "judges": judge_every_answer_of_run_folder(session, run_folder)}
    finally:  # a stop part-way still leaves readable verdict files for what was finished
        for scope in scopes_to_judge:
            path, written = write_verdict_files_for_every_finished_answer(session.stores, session.model, session.stamp, session.run_folders[scope])
            session.say(f"{scope}: wrote {written} verdict records to {path.name}")
            summary["scopes"].setdefault(scope, {})["verdict_file"] = path.name
    summary.update(finished_at=datetime.now().isoformat(timespec="seconds"), seconds=round(session.seconds_since_start(), 1))
    for scope in scopes_to_judge:
        run_folder = session.run_folders[scope]
        path = stable_path(run_folder.output_folder, "gemma_judge_summary", ".json", session.stamp)
        path.write_text(json.dumps({**summary, "this_scope": scope}, indent=2, ensure_ascii=False), encoding="utf-8")
    session.say(f"All judging done in {format_duration(session.seconds_since_start())}")
    return summary


def warn_about_incomplete_generation(session, scopes):
    for scope in scopes:
        run_folder = session.run_folders[scope]
        expected = run_folder.expected_number_of_answers()
        if len(run_folder.answers) < expected:
            session.say(f"WARNING: the {scope} run folder has {len(run_folder.answers)} answers, fewer than the {expected} its run configuration "
                        f"expects; judging what exists (rerun after generation finishes to judge the rest)")


# ---------------------------------------------------------------------------------------------------
# Real session and command line
# ---------------------------------------------------------------------------------------------------

def scopes_and_folders_from(arguments):
    scopes = [arguments.only] if arguments.only else SCOPES
    folders = {HELD_OUT_SCOPE: arguments.held_out_run_folder, BETLEY_SCOPE: arguments.betley_run_folder}
    for scope in scopes:
        if not folders[scope]:
            raise SystemExit(f"--{scope.replace('_', '-')}-run-folder is required" + ("" if arguments.only else " (or use --only for the other one)"))
    return scopes, folders


def build_real_session(arguments, run_folders):
    from transformers import AutoTokenizer
    judge_model = verify_judge_model_is_the_same_in_every_run_folder(run_folders, arguments.allow_non_gemma_judge)
    configuration = run_folders[0].configuration
    judge_condition_spec = judge_condition_spec_of(configuration, judge_model)
    tokenizer = AutoTokenizer.from_pretrained(judge_model["model_repository"], revision=judge_model["model_revision"])
    tokenizer_adapter = baseline_pod.ChatTokenizerAdapter(tokenizer)
    end_of_sequence_ids = baseline_pod.read_end_of_sequence_ids(judge_model, tokenizer)
    longest = longest_judge_prompt_in_tokens(run_folders, tokenizer_adapter, judge_condition_spec, load_betley_templates())
    maximum_model_length = arguments.maximum_model_length or model_length_for_longest_prompt(longest)
    baseline_pod.say(f"Longest judge prompt {longest} tokens; judge engine maximum model length {maximum_model_length}")
    baseline_pod.record_environment(run_folders[0].run_folder, judge_model, arguments.generation_backend, arguments.device)
    backend = baseline_pod.build_backend(arguments, {**configuration, "maximum_model_length": maximum_model_length}, judge_model, end_of_sequence_ids)
    session = GemmaEvaluationJudgingSession(run_folders, backend, tokenizer_adapter, end_of_sequence_ids, judge_model, judge_condition_spec,
                                            maximum_model_length, arguments.report_interval_seconds, arguments.skip_first_reply_checks)
    if arguments.judge_output_cap:
        session.judge_output_caps = {kind: arguments.judge_output_cap for kind in session.judge_output_caps}
    return session


def everything_is_already_judged(run_folders, stores):
    """True when every answer of every run folder has a final judgement, found without loading any model (a resumed or repeated launch)."""
    return all(not answers_without_a_final_judgement(stores, run_folder, kind)
               for run_folder in run_folders for kind in KINDS_BY_SCOPE[run_folder.scope])


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--held-out-run-folder", type=Path, default=None)
    parser.add_argument("--betley-run-folder", type=Path, default=None)
    parser.add_argument("--only", choices=SCOPES, default=None, help="judge only the held-out answers or only the Betley answers")
    parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default="vllm")
    parser.add_argument("--device", default="cuda", help="huggingface backend only")
    parser.add_argument("--report-interval-seconds", type=float, default=PROGRESS_REPORT_INTERVAL_SECONDS)
    parser.add_argument("--maximum-model-length", type=int, default=None, help="default: sized from the longest judge prompt")
    parser.add_argument("--skip-first-reply-checks", action="store_true", help="rehearsal: skip the first judge reply check and the broken-judge stop")
    parser.add_argument("--judge-output-cap", type=int, default=None, help="rehearsal: use this output cap for every judge reply")
    parser.add_argument("--allow-non-gemma-judge", action="store_true", help="rehearsal only: let a tiny stand-in model judge")
    # Accepted so that the shared launcher can pass its usual options; they do nothing for this script.
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--keep-model-caches", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    scopes, folders = scopes_and_folders_from(arguments)
    run_folders = [EvaluationRunFolder(scope, folders[scope]) for scope in scopes]
    stamp = timestamp_now_with_seconds()
    stores = build_attempt_stores(run_folders, stamp)
    if everything_is_already_judged(run_folders, stores):
        judge_model = verify_judge_model_is_the_same_in_every_run_folder(run_folders, arguments.allow_non_gemma_judge)
        for run_folder in run_folders:  # nothing to judge, but the verdict files must exist and be complete
            path, written = write_verdict_files_for_every_finished_answer(stores, judge_model, stamp, run_folder)
            baseline_pod.say(f"{run_folder.scope}: every answer already has a final judgement; {written} new verdict records written to {path.name}")
        return 0
    session = build_real_session(arguments, run_folders)
    warn_about_incomplete_generation(session, scopes)
    run_judging(session, scopes, arguments.skip_first_reply_checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
