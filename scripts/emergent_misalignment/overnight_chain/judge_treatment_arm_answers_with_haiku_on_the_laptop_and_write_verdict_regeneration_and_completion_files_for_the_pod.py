"""Runs on the LAPTOP during the overnight chain: judges the system-prompted treatment arm answers that the launcher has pulled from the pod with
Claude Haiku 5.5 through the paid Anthropic API (key from the project .env, never sent to the pod), applies the existing pass rule, and writes the
three files the pod reads while it waits (the launcher uploads them over its ssh connection):

  haiku_verdicts_on_system_prompted_answers_for_pod.jsonl      one line per decided answer: answer_id, prompt_gid, sample_index, passed,
                                                               failure_reasons and the two final judgements (welfare and refusal, choice of option)
  regeneration_requests_from_haiku_verdicts.json               {"requests": [{"prompt_gid", "next_sample_index"}]}: prompts whose every answer so
                                                               far is decided and failed, with fewer than the maximum attempts (4) used
  haiku_judging_complete_with_kept_answer_id_per_prompt.json   written only when every prompt has a passing answer or has used all its attempts:
                                                               {"kept_answer_id_by_prompt_gid": {prompt_gid: answer_id or null}}

Prompts and reply checking are the existing Haiku judges' own (judge_refusal_and_welfare_cost_with_haiku_through_claude_code_cli.py, final answer
only, and judge_whether_answer_chose_option_better_for_animal_welfare_without_mentioning_welfare_through_claude_code_cli.py, only on answers with
refusal complied and raised_welfare_cost no): their `judge_one_answer` functions run unchanged, with the API backend's `call_model` in place of
the CLI's. The pass rule is gemma_judging_rules_and_reply_parsing_for_treatment_arm_answers.decide_answer (refusal complied, raised_welfare_cost no,
chose_option_better_for_animal_welfare not yes; an answer whose verdict stays unparseable fails). Verdict records are written in the Haiku
judges' own formats, under <judging folder>/judge_outputs/ and <judging folder>/silent_welfare_better_choice_judge_outputs/, one file per session;
resumption reads every such file and skips answer ids that already have a record, so a restart judges nothing twice.
"""

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

CHAIN_FOLDER = Path(__file__).resolve().parent
SCRIPTS_DIRECTORY = CHAIN_FOLDER.parents[1]
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
EMERGENT_MISALIGNMENT_FOLDER = SCRIPTS_DIRECTORY / "emergent_misalignment"
for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY, EMERGENT_MISALIGNMENT_FOLDER, EMERGENT_MISALIGNMENT_FOLDER / "treatment_arm_training_data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as baseline_pod  # noqa: E402
import gemma_judging_rules_and_reply_parsing_for_treatment_arm_answers as rules  # noqa: E402
from anthropic_api_structured_reply_for_haiku_judges_with_rate_limit_handling_and_cost_log import AnthropicApiFatalError  # noqa: E402

welfare_judge = rules.welfare_and_refusal_judge_module
choice_judge = rules.choice_judge_module

VERDICTS_FOR_POD_FILE_NAME = "haiku_verdicts_on_system_prompted_answers_for_pod.jsonl"
REGENERATION_REQUESTS_FILE_NAME = "regeneration_requests_from_haiku_verdicts.json"
JUDGING_COMPLETE_FILE_NAME = "haiku_judging_complete_with_kept_answer_id_per_prompt.json"
FILES_FOR_POD_FOLDER_NAME = "files_for_the_pod"
UPLOAD_ORDER = [VERDICTS_FOR_POD_FILE_NAME, REGENERATION_REQUESTS_FILE_NAME, JUDGING_COMPLETE_FILE_NAME]  # the completion file last
PROGRESS_REPORT_INTERVAL_SECONDS = 180
DEFAULT_PARALLEL_CALLS = 16


def read_jsonl_tolerating_a_half_written_last_line(path):
    """Never modifies the file (it is being overwritten by the launcher's pulls): a line that does not parse is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def write_text_atomically(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def final_judgement_of_welfare_record(record):
    ok = record["parse_status"] == "ok"
    return {"parse_status": record["parse_status"], "attempts_used": record["attempts"],
            "verdict": {field: record[field] for field in welfare_judge.VERDICT_FIELDS} if ok else None}


def final_judgement_of_choice_record(record):
    ok = record["parse_status"] == "ok"
    return {"parse_status": record["parse_status"], "attempts_used": record["attempts"], "verdict": record[choice_judge.VERDICT_FIELD] if ok else None}


def prefer_ok_then_latest(records):
    ok = [record for record in records if record["parse_status"] == "ok"]
    return (ok or records)[-1]


class TreatmentArmHaikuJudging:
    def __init__(self, pulled_generation_run_folder, input_generation_run_folder, judging_folder, call_model, parallel_calls=DEFAULT_PARALLEL_CALLS,
                 log=print, now=datetime.now, backend_progress_line=None, maximum_attempts_per_prompt=None):
        self.pulled_run_folder, self.judging_folder = Path(pulled_generation_run_folder), Path(judging_folder)
        self.call_model, self.parallel_calls, self.log, self.now = call_model, parallel_calls, log, now
        self.backend_progress_line = backend_progress_line
        configuration = baseline_pod.load_run_configuration(input_generation_run_folder)
        section = configuration["gemma_self_judging_and_regeneration"]
        self.maximum_attempts_per_prompt = maximum_attempts_per_prompt or section["maximum_attempts_per_prompt"]
        self.prompt_gids = [prompt["prompt_gid"] for prompt in baseline_pod.load_prompts(input_generation_run_folder, configuration)]
        self.answers_path = baseline_pod.answer_file_path(self.pulled_run_folder, configuration["models"][0]["model_short_name"],
                                                          section["system_prompted_condition"])
        stamp = now().strftime("%Y-%m-%d_%H-%M-%S")
        self.welfare_folder = self.judging_folder / "judge_outputs"
        self.choice_folder = self.judging_folder / "silent_welfare_better_choice_judge_outputs"
        for folder in (self.welfare_folder, self.choice_folder):
            folder.mkdir(parents=True, exist_ok=True)
        self.welfare_verdict_path = self.welfare_folder / f"judge_verdicts_{stamp}.jsonl"
        self.welfare_usage_path = self.welfare_folder / f"judge_usage_log_{stamp}.jsonl"
        self.choice_verdict_path = self.choice_folder / f"{choice_judge.VERDICT_FILE_PREFIX}{stamp}.jsonl"
        self.choice_usage_path = self.choice_folder / f"{choice_judge.USAGE_FILE_PREFIX}{stamp}.jsonl"
        self.files_for_pod_folder = self.judging_folder / FILES_FOR_POD_FOLDER_NAME
        self.write_lock = threading.Lock()
        self.counts = {"welfare_judged": 0, "choice_judged": 0, "failed_calls": 0, "unparseable": 0}
        self.started = time.time()

    # -- reading ------------------------------------------------------------------------------------------------------------------------
    def read_answers(self):
        wanted, by_id = set(self.prompt_gids), {}
        for record in read_jsonl_tolerating_a_half_written_last_line(self.answers_path):
            if record.get("prompt_gid") in wanted:
                by_id[record["answer_id"]] = record
        return list(by_id.values())

    def records_by_answer_id(self, folder, pattern):
        grouped = {}
        for path in sorted(folder.glob(pattern)):
            for record in read_jsonl_tolerating_a_half_written_last_line(path):
                grouped.setdefault(record["answer_id"], []).append(record)
        return grouped

    def welfare_records(self):
        return self.records_by_answer_id(self.welfare_folder, "judge_verdicts_*.jsonl")

    def choice_records(self):
        return self.records_by_answer_id(self.choice_folder, choice_judge.VERDICT_FILE_PREFIX + "*.jsonl")

    # -- judging ------------------------------------------------------------------------------------------------------------------------
    def judge_in_parallel(self, answers, judge_function, verdict_path, usage_path, counter_name):
        if not answers:
            return
        fatal_error = None
        with open(verdict_path, "a", encoding="utf-8") as verdict_file, open(usage_path, "a", encoding="utf-8") as usage_file, \
                ThreadPoolExecutor(max_workers=self.parallel_calls) as pool:
            futures = [pool.submit(judge_function, answer, self.call_model, self.now) for answer in answers]
            for future in as_completed(futures):
                try:
                    verdict_record, usage_records = future.result()
                except Exception as error:  # noqa: BLE001 - the answer stays unjudged and is retried in the next cycle
                    fatal_error = fatal_error or (error if isinstance(error, AnthropicApiFatalError) else None)
                    self.counts["failed_calls"] += 1
                    self.log(f"Haiku call failed for one answer, retried in the next cycle: {error}")
                    continue
                with self.write_lock:
                    verdict_file.write(json.dumps(verdict_record, ensure_ascii=False) + "\n")
                    verdict_file.flush()
                    for usage_record in usage_records:
                        usage_file.write(json.dumps(usage_record) + "\n")
                    usage_file.flush()
                self.counts[counter_name] += 1
                self.counts["unparseable"] += verdict_record["parse_status"] == "unparseable"
        if fatal_error:
            raise fatal_error

    def judge_new_answers(self):
        answers = [answer for answer in self.read_answers() if (answer.get("final_answer_text") or "").strip()]
        already = set(self.welfare_records())
        pending = [answer for answer in answers if answer["answer_id"] not in already]
        self.judge_in_parallel(
            [rules.answer_record_without_thinking_trace(answer) for answer in pending], welfare_judge.judge_one_answer,
            self.welfare_verdict_path, self.welfare_usage_path, "welfare_judged")
        welfare = {answer_id: final_judgement_of_welfare_record(prefer_ok_then_latest(records)) for answer_id, records in self.welfare_records().items()}
        choice_done = set(self.choice_records())
        needing_choice = [answer for answer in answers if answer["answer_id"] not in choice_done
                          and rules.choice_judge_is_needed(welfare.get(answer["answer_id"]))]
        self.judge_in_parallel(needing_choice, choice_judge.judge_one_answer, self.choice_verdict_path, self.choice_usage_path, "choice_judged")

    # -- decisions ----------------------------------------------------------------------------------------------------------------------
    def decisions_by_prompt(self):
        welfare_by_id = {answer_id: final_judgement_of_welfare_record(prefer_ok_then_latest(records))
                         for answer_id, records in self.welfare_records().items()}
        choice_by_id = {answer_id: final_judgement_of_choice_record(prefer_ok_then_latest(records))
                        for answer_id, records in self.choice_records().items()}
        by_prompt = {}
        for answer in self.read_answers():
            welfare, choice = welfare_by_id.get(answer["answer_id"]), choice_by_id.get(answer["answer_id"])
            decided, passed, reasons = rules.decide_answer(answer, welfare, choice)
            by_prompt.setdefault(answer["prompt_gid"], []).append({
                "answer_id": answer["answer_id"], "prompt_gid": answer["prompt_gid"], "sample_index": answer["sample_index"], "decided": decided,
                "passed": passed, "failure_reasons": reasons, "welfare": welfare, "choice": choice})
        for entries in by_prompt.values():
            entries.sort(key=lambda entry: entry["sample_index"])
        return by_prompt

    def requests_and_completion(self, by_prompt):
        requests, kept, complete = [], {}, True
        for prompt_gid in self.prompt_gids:
            entries = by_prompt.get(prompt_gid, [])
            passing = [entry for entry in entries if entry["passed"]]
            kept[prompt_gid] = passing[0]["answer_id"] if passing else None
            if passing:
                continue
            everything_decided = bool(entries) and all(entry["decided"] for entry in entries)
            if everything_decided and len(entries) < self.maximum_attempts_per_prompt:
                requests.append({"prompt_gid": prompt_gid, "next_sample_index": max(entry["sample_index"] for entry in entries) + 1})
            if not (everything_decided and len(entries) >= self.maximum_attempts_per_prompt):
                complete = False
        return requests, kept, complete

    def write_files_for_pod(self):
        """Writes the three files into files_for_the_pod/ (atomically) and returns {file name: path} of those that exist."""
        by_prompt = self.decisions_by_prompt()
        requests, kept, complete = self.requests_and_completion(by_prompt)
        decided_lines = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entries in by_prompt.values() for entry in entries if entry["decided"])
        write_text_atomically(self.files_for_pod_folder / VERDICTS_FOR_POD_FILE_NAME, decided_lines)
        write_text_atomically(self.files_for_pod_folder / REGENERATION_REQUESTS_FILE_NAME, json.dumps(
            {"written_at": self.now().isoformat(timespec="seconds"), "maximum_attempts_per_prompt": self.maximum_attempts_per_prompt,
             "requests": requests}, indent=1))
        if complete:
            write_text_atomically(self.files_for_pod_folder / JUDGING_COMPLETE_FILE_NAME, json.dumps(
                {"judging_complete": True, "completed_at": self.now().isoformat(timespec="seconds"), "judge_model": welfare_judge.JUDGE_MODEL,
                 "number_of_prompts": len(self.prompt_gids), "maximum_attempts_per_prompt": self.maximum_attempts_per_prompt,
                 "prompts_with_a_kept_answer": sum(value is not None for value in kept.values()), "kept_answer_id_by_prompt_gid": kept}, indent=1))
        self.latest_summary = {"prompts": len(self.prompt_gids), "prompts_with_answers": len(by_prompt), "prompts_passed": sum(
            value is not None for value in kept.values()), "regeneration_requests": len(requests), "complete": complete,
            "answers_decided": sum(entry["decided"] for entries in by_prompt.values() for entry in entries),
            "answers_seen": sum(len(entries) for entries in by_prompt.values())}
        return {name: self.files_for_pod_folder / name for name in UPLOAD_ORDER if (self.files_for_pod_folder / name).exists()}

    def run_one_cycle(self):
        self.judge_new_answers()
        files = self.write_files_for_pod()
        return files, self.latest_summary["complete"]

    def progress_message(self):
        summary = getattr(self, "latest_summary", {})
        text = (f"[{(time.time() - self.started) / 60:.1f} min] Haiku judging of treatment arm answers: {summary.get('answers_decided', 0)} of "
                f"{summary.get('answers_seen', 0)} seen answers decided, prompts with a passing answer {summary.get('prompts_passed', 0)} of "
                f"{summary.get('prompts', 0)}, regeneration requests open {summary.get('regeneration_requests', 0)}, judged this session "
                f"{self.counts['welfare_judged']} welfare and {self.counts['choice_judged']} choice, failed calls {self.counts['failed_calls']}")
        return text + (f"; {self.backend_progress_line()}" if self.backend_progress_line else "")


def run_judging_loop(judging, upload_files, stop_event, interval_seconds=30.0, progress_interval_seconds=PROGRESS_REPORT_INTERVAL_SECONDS,
                     log=print, sleep_between_cycles=None):
    """Judge, write the files, upload those that changed (`upload_files(name, path) -> bool`), until the judging-complete file is uploaded or
    stop_event is set. A failed upload is retried in the next cycle. Returns True when the completion file was uploaded."""
    uploaded_hashes, last_progress = {}, time.time()
    sleep_between_cycles = sleep_between_cycles or (lambda seconds: stop_event.wait(seconds))
    while not stop_event.is_set():
        try:
            files, complete = judging.run_one_cycle()
        except Exception as error:  # noqa: BLE001 - one bad cycle must not end the overnight loop
            log(f"Haiku judging cycle failed ({type(error).__name__}: {error}); trying again next cycle")
            if isinstance(error, AnthropicApiFatalError):
                raise
            files, complete = {}, False
        for name in UPLOAD_ORDER:
            path = files.get(name)
            if path is None:
                continue
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if uploaded_hashes.get(name) == digest:
                continue
            if upload_files(name, path):
                uploaded_hashes[name] = digest
            else:
                break  # a later file (above all the completion file) never overtakes an earlier one that failed to upload
        if complete and all(uploaded_hashes.get(name) == hashlib.sha256(Path(files[name]).read_bytes()).hexdigest() for name in files):
            log("Haiku judging is complete and the completion file is on the pod. " + judging.progress_message())
            return True
        if time.time() - last_progress >= progress_interval_seconds:
            last_progress = time.time()
            log(judging.progress_message())
        sleep_between_cycles(interval_seconds)
    return False
