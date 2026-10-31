"""Judge client for the persona vector extraction: facet score, coherence, share about animals, volunteered welfare
remark, and (pilot only) the consistency pair score, all with GPT-5.6 Luna through the OpenAI API.

Importable: build requests with the render_* / build_judge_requests_for_run functions and call
`run_judge_requests(judge_requests, client, reading_mode, ...)`, which returns one result per request.

Reading the score. Chen et al. read a 0-100 score from the first-token log probabilities (probability-weighted mean of
the integer tokens 0-100; discarded if under 0.25 of the probability mass is on numbers). OpenAI's model page for
gpt-5.6-luna (developers.openai.com/api/docs/models/gpt-5.6-luna) lists streaming, structured outputs, function calling,
file search, image input, web search and prompt caching and does not list logprobs, and reasoning models have not
supported logprobs; so the DEFAULT mode is `bare_number`: the lowest reasoning effort ("none", the lowest value the
page lists) and a bare number parsed from the reply, which is one temperature-1 sample per call (noisier than a
probability-weighted mean). `logprobs` mode (Chat Completions, logprobs, top_logprobs=20, the Chen et al. reading) is
implemented and tested with mocks, for use if a first real call shows the model accepts it. No real call has been made.

The API key is read from the project .env only (shared.openai_structured_output_client.load_api_key).

Command line (from the repository root), dry run first:
    python3 scripts/persona_vector_projection/judge_answers_with_gpt_5_6_luna.py --run-folder <run folder> --dry-run
    python3 scripts/persona_vector_projection/judge_answers_with_gpt_5_6_luna.py --run-folder <run folder>   # PAID
Resumable: requests already in any judge_outputs/judge_results_<timestamp>.jsonl are skipped.
"""

import argparse
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from shared.api_usage_and_cost_tracking import (UsageAndCostTracker, compute_cost_in_dollars, empty_usage,  # noqa: E402
                                                extract_usage_from_response)
from shared.resumable_jsonl_records import AppendOnlyJsonlWriter, find_completed_keys, read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402
from shared.timed_progress_reporter import TimedProgressReporter  # noqa: E402

READING_MODE_BARE_NUMBER = "bare_number"
READING_MODE_LOGPROBS = "logprobs"
LOWEST_REASONING_EFFORT = "none"
MAXIMUM_OUTPUT_TOKENS_BARE_NUMBER = 16  # the smallest value the Responses API accepts
TOP_LOGPROBS_COUNT = 20

KIND_FACET_SCORE = "facet_score"
KIND_COHERENCE = "coherence"
KIND_SHARE_ABOUT_ANIMALS = "share_about_animals"
KIND_VOLUNTEERED_WELFARE_REMARK = "volunteered_welfare_remark"
KIND_CONSISTENCY_PAIR_SCORE = "consistency_pair_score"
YES_NO_KINDS = {KIND_VOLUNTEERED_WELFARE_REMARK}

TEMPLATE_FILE_NAMES = {
    KIND_FACET_SCORE: "judge_facet_score_prompt_template.txt",
    KIND_COHERENCE: "judge_coherence_prompt_template.txt",
    KIND_SHARE_ABOUT_ANIMALS: "judge_share_about_animals_prompt_template.txt",
    KIND_VOLUNTEERED_WELFARE_REMARK: "judge_volunteered_welfare_remark_prompt_template.txt",
    KIND_CONSISTENCY_PAIR_SCORE: "judge_consistency_pair_score_prompt_template.txt",
}
FACET_NOTE_FILE_NAMES = {
    defs.FACET_CARES_ABOUT_WELFARE: "judge_facet_note_cares_about_welfare.txt",
    defs.FACET_CALIBRATED_CONSIDERATION: "judge_facet_note_calibrated_consideration.txt",
    defs.FACET_NON_INTRUSIVE: "judge_facet_note_non_intrusive.txt",
    defs.FACET_CONSISTENT: "judge_facet_note_consistent.txt",
}
NON_INTRUSIVE_COST_PLACEHOLDER = "{applies: {cost_note} / does not apply}"
SLOT_PATTERN = re.compile(r"\{[a-z_]+\}")


@dataclass
class JudgeRequest:
    request_id: str
    kind: str
    prompt_text: str
    answer_id: str = None
    metadata: dict = field(default_factory=dict)


def load_template(file_name, prompt_folder=None):
    return (Path(prompt_folder or defs.PROMPT_FOLDER) / file_name).read_text(encoding="utf-8").strip()


def fill_slots(template, values):
    """Replaces {slot} for every key of `values` (a plain replace: answers may contain braces). Raises if a slot of the
    template is left unfilled, and does not touch braces inside the substituted values."""
    result = template
    for name, value in values.items():
        result = result.replace("{" + name + "}", "\x00" + name + "\x01")
    leftover = [match for match in SLOT_PATTERN.findall(result)]
    if leftover:
        raise ValueError(f"Unfilled slots {leftover} in judge prompt template")
    for name, value in values.items():
        result = result.replace("\x00" + name + "\x01", str(value))
    return result


def render_facet_note(facet, question_record, prompt_folder=None):
    note = load_template(FACET_NOTE_FILE_NAMES[facet], prompt_folder)
    labels = question_record["labels"]
    if facet == defs.FACET_CALIBRATED_CONSIDERATION:
        return fill_slots(note, {"evidence_bands": labels["evidence_bands"]})
    if facet == defs.FACET_NON_INTRUSIVE:
        if NON_INTRUSIVE_COST_PLACEHOLDER not in note:
            raise ValueError("non-intrusive note lacks its cost placeholder")
        phrase = (f"applies: {labels['material_welfare_cost_note']}" if labels["material_welfare_cost"]
                  else "does not apply")
        return note.replace(NON_INTRUSIVE_COST_PLACEHOLDER, phrase)
    if facet == defs.FACET_CONSISTENT:
        return fill_slots(note, {"animal": labels["animal"], "reference_animal": labels["reference_animal"]})
    return note


def render_facet_score_prompt(facet, facet_definition, question_record, answer_text, prompt_folder=None):
    return fill_slots(load_template(TEMPLATE_FILE_NAMES[KIND_FACET_SCORE], prompt_folder), {
        "facet_name": facet_definition["display_name"],
        "positive_pole_description": facet_definition["positive_pole_description"],
        "negative_pole_description": facet_definition["negative_pole_description"],
        "facet_specific_notes": render_facet_note(facet, question_record, prompt_folder),
        "question": question_record["question_text"], "answer": answer_text})


def render_simple_prompt(kind, question_record, answer_text, prompt_folder=None):
    template = load_template(TEMPLATE_FILE_NAMES[kind], prompt_folder)
    values = {"answer": answer_text}
    if "{question}" in template:
        values["question"] = question_record["question_text"]
    return fill_slots(template, values)


def render_consistency_pair_prompt(question_record_a, answer_a, question_record_b, answer_b, prompt_folder=None):
    return fill_slots(load_template(TEMPLATE_FILE_NAMES[KIND_CONSISTENCY_PAIR_SCORE], prompt_folder), {
        "animal_a": question_record_a["labels"]["animal"], "animal_b": question_record_b["labels"]["animal"],
        "question_a": question_record_a["question_text"], "answer_a": answer_a,
        "question_b": question_record_b["question_text"], "answer_b": answer_b})


def build_judge_requests_for_answers(answer_records, questions_by_id, facet_definitions, prompt_folder=None):
    requests = []
    for answer in answer_records:
        question = questions_by_id[answer["question_id"]]
        facet, text = answer["facet"], answer["answer_text"]
        kinds_and_prompts = []
        if facet in defs.PERSONA_FACETS:
            kinds_and_prompts.append((KIND_FACET_SCORE, render_facet_score_prompt(
                facet, facet_definitions[facet], question, text, prompt_folder)))
        kinds_and_prompts.append((KIND_COHERENCE, render_simple_prompt(KIND_COHERENCE, question, text, prompt_folder)))
        kinds_and_prompts.append((KIND_SHARE_ABOUT_ANIMALS,
                                  render_simple_prompt(KIND_SHARE_ABOUT_ANIMALS, question, text, prompt_folder)))
        if facet == defs.TOPIC_CONTROL and answer["version"] == defs.TOPIC_VERSION_ANIMAL:
            kinds_and_prompts.append((KIND_VOLUNTEERED_WELFARE_REMARK, render_simple_prompt(
                KIND_VOLUNTEERED_WELFARE_REMARK, question, text, prompt_folder)))
        for kind, prompt in kinds_and_prompts:
            requests.append(JudgeRequest(f"{answer['answer_id']}__{kind}", kind, prompt, answer["answer_id"]))
    return requests


def build_consistency_pair_judge_requests(answer_records, questions_by_id, sample_size, seed, prompt_folder=None):
    """Pilot only: for a random sample of (pair, system prompt, sample index) with both versions answered."""
    grouped = {}
    for answer in answer_records:
        if answer["facet"] == defs.FACET_CONSISTENT:
            key = (answer["pair_id"], answer["system_prompt_id"], answer["sample_index"])
            grouped.setdefault(key, {})[answer["version"]] = answer
    complete_keys = sorted(key for key, versions in grouped.items() if len(versions) == 2)
    random.Random(seed).shuffle(complete_keys)
    requests = []
    for key in complete_keys[:sample_size]:
        answer_a, answer_b = grouped[key][defs.CONSISTENT_VERSION_A], grouped[key][defs.CONSISTENT_VERSION_B]
        prompt = render_consistency_pair_prompt(questions_by_id[answer_a["question_id"]], answer_a["answer_text"],
                                                questions_by_id[answer_b["question_id"]], answer_b["answer_text"],
                                                prompt_folder)
        requests.append(JudgeRequest(f"{'__'.join(map(str, key))}__{KIND_CONSISTENCY_PAIR_SCORE}",
                                     KIND_CONSISTENCY_PAIR_SCORE, prompt,
                                     metadata={"pair_id": key[0], "system_prompt_id": key[1], "sample_index": key[2]}))
    return requests


# ---------------------------------------------------------------------------------------------------
# Reading the reply
# ---------------------------------------------------------------------------------------------------

def parse_bare_number(text):
    match = re.match(r"\s*(\d{1,3})(?:\.\d+)?\b", text or "")
    if not match or not 0 <= int(match.group(1)) <= 100:
        return None
    return float(match.group(1))


def parse_yes_or_no(text):
    first_word = re.match(r"\s*([A-Za-z]+)", text or "")
    word = first_word.group(1).upper() if first_word else ""
    return word if word in ("YES", "NO") else None


def score_from_first_token_top_logprobs(top_logprobs, minimum_mass=defs.DEFAULT_MINIMUM_PROBABILITY_MASS_ON_NUMBERS):
    """Chen et al.: top_logprobs is a list of (token, log probability) for the first output token. The score is the
    probability-weighted mean of the tokens that are integers 0-100; None if under `minimum_mass` of the probability
    is on such tokens. Returns (score, probability mass on numbers)."""
    weighted_sum, mass = 0.0, 0.0
    for token, log_probability in top_logprobs:
        stripped = token.strip()
        if stripped.isdigit() and 0 <= int(stripped) <= 100:
            probability = math.exp(log_probability)
            weighted_sum += int(stripped) * probability
            mass += probability
    if mass < minimum_mass:
        return None, mass
    return weighted_sum / mass, mass


# ---------------------------------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------------------------------

def build_bare_number_request_arguments(request):
    """Responses API, lowest reasoning effort, no structured output, temperature 1 (top_p left at its default of 1)."""
    return {"model": defs.JUDGE_MODEL_NAME, "input": request.prompt_text,
            "reasoning": {"effort": LOWEST_REASONING_EFFORT}, "temperature": defs.SAMPLING_TEMPERATURE,
            "max_output_tokens": MAXIMUM_OUTPUT_TOKENS_BARE_NUMBER}


def build_logprobs_request_arguments(request):
    return {"model": defs.JUDGE_MODEL_NAME, "messages": [{"role": "user", "content": request.prompt_text}],
            "temperature": defs.SAMPLING_TEMPERATURE, "logprobs": True, "top_logprobs": TOP_LOGPROBS_COUNT,
            "max_completion_tokens": 1}


def usage_from_chat_completion(response):
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {"input_tokens": usage.prompt_tokens, "cached_input_tokens": getattr(details, "cached_tokens", 0) or 0,
            "output_tokens": usage.completion_tokens,
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", 0) or 0}


def call_chat_completions_with_retries(client, request_arguments, sleep_function=time.sleep):
    from shared.openai_structured_output_client import (FIRST_BACKOFF_SECONDS, MAXIMUM_BACKOFF_SECONDS,
                                                        MAXIMUM_TRANSPORT_ATTEMPTS, is_transient_error)
    for attempt in range(1, MAXIMUM_TRANSPORT_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(**request_arguments)
        except Exception as error:  # noqa: BLE001
            if not is_transient_error(error) or attempt == MAXIMUM_TRANSPORT_ATTEMPTS:
                raise
            sleep_function(min(MAXIMUM_BACKOFF_SECONDS, FIRST_BACKOFF_SECONDS * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0))


def judge_one_request(client, request, reading_mode, minimum_mass, call_responses=None, call_chat=None):
    """Returns (result dict, usage dict). At most 2 attempts for an unparseable reply."""
    from shared.openai_structured_output_client import call_with_transport_retries
    call_responses = call_responses or call_with_transport_retries
    call_chat = call_chat or call_chat_completions_with_retries
    usage_total, last_raw = empty_usage(), None
    for attempt in (1, 2):
        probability_mass = None
        if reading_mode == READING_MODE_LOGPROBS and request.kind not in YES_NO_KINDS:
            response = call_chat(client, build_logprobs_request_arguments(request))
            usage = usage_from_chat_completion(response)
            choice = response.choices[0]
            last_raw = choice.message.content
            first = choice.logprobs.content[0]
            score, probability_mass = score_from_first_token_top_logprobs(
                [(entry.token, entry.logprob) for entry in first.top_logprobs], minimum_mass)
            parsed_ok = score is not None
        else:
            response = call_responses(client, build_bare_number_request_arguments(request))
            usage = extract_usage_from_response(response)
            last_raw = response.output_text
            if request.kind in YES_NO_KINDS:
                answer_word = parse_yes_or_no(last_raw)
                score, parsed_ok = None, answer_word is not None
            else:
                score = parse_bare_number(last_raw)
                parsed_ok = score is not None
        usage_total = {name: usage_total[name] + usage[name] for name in usage_total}
        if parsed_ok:
            result = {"score": score, "yes_or_no": answer_word if request.kind in YES_NO_KINDS else None,
                      "discarded": False}
            break
        if reading_mode == READING_MODE_LOGPROBS and request.kind not in YES_NO_KINDS:
            result = {"score": None, "yes_or_no": None, "discarded": True,
                      "discard_reason": "under the minimum probability mass on numbers"}
            break  # a second sample would not change the first-token distribution
        result = {"score": None, "yes_or_no": None, "discarded": True, "discard_reason": "unparseable reply"}
    result.update({"request_id": request.request_id, "answer_id": request.answer_id, "kind": request.kind,
                   "reading_mode": reading_mode, "probability_mass_on_numbers": probability_mass,
                   "raw_reply": last_raw, "usage": usage_total, "metadata": request.metadata})
    return result, usage_total


def run_judge_requests(judge_requests, client, reading_mode=READING_MODE_BARE_NUMBER, results_path=None,
                       usage_log_path=None, worker_count=16, minimum_mass=defs.DEFAULT_MINIMUM_PROBABILITY_MASS_ON_NUMBERS,
                       progress_log_path=None, call_responses=None, call_chat=None, report_interval_seconds=180):
    """Judges every request not already in results_path (a JSONL appended to as results arrive) and returns all results
    (earlier and new) as a list. Tokens, cost and cache rate are tracked per request kind and written to usage_log_path."""
    writer = AppendOnlyJsonlWriter()
    previous = []
    if results_path:  # results_path is this session's file; every judge_results_*.jsonl beside it counts as done
        for earlier_path in sorted(Path(results_path).parent.glob("judge_results_*.jsonl")):
            previous.extend(read_jsonl(earlier_path))
    done_ids = {record["request_id"] for record in previous}
    pending = [request for request in judge_requests if request.request_id not in done_ids]
    tracker = UsageAndCostTracker(defs.JUDGE_MODEL_NAME, group_label="per_request_kind")
    progress_log_path = progress_log_path or (Path(results_path).with_name(f"judge_progress_log_{timestamp_now()}.txt") if results_path
                                              else Path("/dev/null"))
    reporter = TimedProgressReporter(
        progress_log_path, len(pending), already_done_count=len(done_ids),
        status_suffix_function=lambda: f"cost so far {tracker.overall_totals['cost_dollars']:.4f} dollars, "
                                       f"input tokens {tracker.overall_totals['input_tokens']}",
        after_report_function=(lambda: tracker.write_log_file(usage_log_path, f"mode {reading_mode}")) if usage_log_path else None,
        report_interval_seconds=report_interval_seconds)
    new_results = []

    def work(request):
        try:
            result, usage = judge_one_request(client, request, reading_mode, minimum_mass, call_responses, call_chat)
        except Exception as error:  # noqa: BLE001  (recorded, not appended: it is retried on resume)
            reporter.note_task_finished(failed=True)
            return {"request_id": request.request_id, "error": repr(error)}
        tracker.add_usage(request.kind, usage)
        if results_path:
            writer.append(results_path, result)
        new_results.append(result)
        reporter.note_task_finished(failed=False)
        return result

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        outcomes = list(executor.map(work, pending))
    reporter.finish()
    if usage_log_path:
        tracker.write_log_file(usage_log_path, f"mode {reading_mode}")
    failures = [outcome for outcome in outcomes if "error" in outcome]
    return previous + new_results, failures


# ---------------------------------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------------------------------

CHARACTERS_PER_TOKEN_ESTIMATE = 4.0
OUTPUT_TOKENS_CEILING_PER_CALL = MAXIMUM_OUTPUT_TOKENS_BARE_NUMBER


def estimate_judge_cost(judge_requests, reading_mode=READING_MODE_BARE_NUMBER):
    """Calls, estimated input tokens (characters / 4), output-token ceiling and cost ceiling (no caching assumed)."""
    counts, input_tokens = {}, 0
    for request in judge_requests:
        counts[request.kind] = counts.get(request.kind, 0) + 1
        input_tokens += len(request.prompt_text) / CHARACTERS_PER_TOKEN_ESTIMATE
    calls = sum(counts.values())
    output_tokens = calls * (1 if reading_mode == READING_MODE_LOGPROBS else OUTPUT_TOKENS_CEILING_PER_CALL)
    cost = compute_cost_in_dollars(defs.JUDGE_MODEL_NAME, int(input_tokens), 0, output_tokens)
    return {"calls_by_kind": counts, "calls": calls, "estimated_input_tokens": int(input_tokens),
            "output_token_ceiling": output_tokens, "cost_ceiling_dollars": round(cost, 4)}


def load_answers_for_run(run_folder):
    return read_all_answer_records(Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / defs.ANSWERS_FOLDER_NAME)


def read_all_answer_records(answers_folder):
    records = []
    for path in sorted(Path(answers_folder).glob("answers_shard_*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--reading-mode", choices=[READING_MODE_BARE_NUMBER, READING_MODE_LOGPROBS],
                        default=READING_MODE_BARE_NUMBER)
    parser.add_argument("--pair-judge-sample-size", type=int, default=0,
                        help="Pilot only: also judge this many consistent pairs side by side")
    parser.add_argument("--worker-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None, client=None):
    arguments = parse_arguments(argv)
    run_folder = Path(arguments.run_folder)
    answers = load_answers_for_run(run_folder)
    if not answers:
        raise SystemExit(f"No answers in {run_folder}; pull the generation outputs first")
    inputs_folder = run_folder / defs.INPUTS_FOLDER_NAME
    questions_by_id = {q["question_id"]: q for q in read_jsonl(defs.find_latest_file(inputs_folder, "questions", "jsonl"))}
    facet_definitions = json.loads(defs.find_latest_file(inputs_folder, "facet_definitions", "json").read_text())
    requests = build_judge_requests_for_answers(answers, questions_by_id, facet_definitions)
    if arguments.pair_judge_sample_size:
        requests += build_consistency_pair_judge_requests(answers, questions_by_id, arguments.pair_judge_sample_size,
                                                          arguments.seed)
    estimate = estimate_judge_cost(requests, arguments.reading_mode)
    print(json.dumps({"reading_mode": arguments.reading_mode, "judge_model": defs.JUDGE_MODEL_NAME, **estimate}, indent=2))
    if arguments.dry_run:
        print("DRY RUN: nothing was called.")
        return 0
    from shared.openai_structured_output_client import make_client
    client = client or make_client()
    output_folder = run_folder / defs.JUDGEMENTS_FOLDER_NAME
    output_folder.mkdir(exist_ok=True)
    stamp = timestamp_now()
    results, failures = run_judge_requests(
        requests, client, arguments.reading_mode, results_path=output_folder / f"judge_results_{stamp}.jsonl",
        usage_log_path=output_folder / f"judge_usage_and_cost_log_{stamp}.json", worker_count=arguments.worker_count,
        progress_log_path=output_folder / f"judge_progress_log_{stamp}.txt")
    print(f"Judged {len(results)} requests in total; {len(failures)} failed this session (rerun to retry them)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
