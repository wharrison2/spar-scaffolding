"""Judge client for the persona vector extraction: facet score, coherence, share about animals, volunteered welfare
remark, and (pilot only) the consistency pair score, all with GPT-5.6 Luna through the OpenAI API. The moralizing
delivery facet has its own kinds: `delivery_score` (bare number 0 to 100) and `content_check` (a JSON object, parsed by
parse_content_check_reply), both at reasoning effort "low", plus the unchanged `coherence` ("none").

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

from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module  # noqa: E402
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
# Facet and pair scores use a little reasoning: in the pilot's first 60 non-intrusive answers, effort "none" gave
# positive-pole answers a mean of 29 (many 0s for answers that help fully), effort "low" a mean of 62, at about
# $0.0003 a call (small_sample_checks/judge_reasoning_effort_low_versus_none_first_batch_2026-10-31_02-40/).
SCORING_REASONING_EFFORT = "low"
MAXIMUM_OUTPUT_TOKENS_WITH_REASONING = 4000
OUTPUT_TOKENS_CEILING_PER_REASONING_CALL = 1000  # cost-ceiling allowance; the pilot check averaged far less
TOP_LOGPROBS_COUNT = 20

KIND_FACET_SCORE = "facet_score"
KIND_COHERENCE = "coherence"
KIND_SHARE_ABOUT_ANIMALS = "share_about_animals"
KIND_VOLUNTEERED_WELFARE_REMARK = "volunteered_welfare_remark"
KIND_CONSISTENCY_PAIR_SCORE = "consistency_pair_score"
# Moralizing delivery: a bare 0-100 delivery score, and a JSON content check (see parse_content_check_reply).
KIND_DELIVERY_SCORE = "delivery_score"
KIND_CONTENT_CHECK = "content_check"
# The three content-controlled facets other than moralizing delivery: a bare 0-100 facet score (100 = positive pole).
KIND_CONTENT_CONTROLLED_FACET_SCORE = facet_definitions_module.KIND_CONTENT_CONTROLLED_FACET_SCORE
# Calibrated consideration only: a bare 0-100 score of how confidently the answer states its claims about the animals, ignoring
# the evidence and the direction of the claim. Asked for every calibrated answer, beside the facet score.
KIND_STATED_CONFIDENCE = facet_definitions_module.KIND_STATED_CONFIDENCE
YES_NO_KINDS = {KIND_VOLUNTEERED_WELFARE_REMARK}
JSON_REPLY_KINDS = {KIND_CONTENT_CHECK}
# A reply that does not parse is recorded as discarded; on a rerun these kinds are asked again (their discarded records
# are superseded by the newer record), so an unparseable reply is never silently counted as a result.
KINDS_RETRIED_WHEN_DISCARDED = {KIND_DELIVERY_SCORE, KIND_CONTENT_CHECK, KIND_CONTENT_CONTROLLED_FACET_SCORE, KIND_STATED_CONFIDENCE}
CONTENT_CHECK_BOOLEAN_KEYS = ("makes_the_point", "adds_welfare_facts", "completes_the_task")
CONTENT_CHECK_KEYS = ("makes_the_point", "separate_welfare_mentions", "adds_welfare_facts", "completes_the_task",
                      "welfare_passage")
MENTION_COUNT_THREE_OR_MORE = "3 or more"

TEMPLATE_FILE_NAMES = {
    KIND_FACET_SCORE: "judge_facet_score_prompt_template.txt",
    KIND_COHERENCE: "judge_coherence_prompt_template.txt",
    KIND_SHARE_ABOUT_ANIMALS: "judge_share_about_animals_prompt_template.txt",
    KIND_VOLUNTEERED_WELFARE_REMARK: "judge_volunteered_welfare_remark_prompt_template.txt",
    KIND_CONSISTENCY_PAIR_SCORE: "judge_consistency_pair_score_prompt_template.txt",
    KIND_DELIVERY_SCORE: "judge_delivery_score_prompt_template.txt",
    KIND_CONTENT_CHECK: "judge_content_check_prompt_template.txt",
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


def render_moralizing_delivery_prompt(kind, question_record, answer_text, prompt_folder=None):
    """The delivery score or content check prompt: both quote the question's welfare point."""
    point = (question_record.get("labels") or {}).get("welfare_point")
    if not point:
        raise ValueError(f"Question {question_record['question_id']} has no welfare point for the {kind} prompt")
    return fill_slots(load_template(TEMPLATE_FILE_NAMES[kind], prompt_folder), {
        "welfare_point": point, "question": question_record["question_text"], "answer": answer_text})


def render_content_controlled_prompt(kind, definition, question_record, answer_text, prompt_folder=None):
    """The facet-score or content-check prompt of a content-controlled facet (`definition` from
    content_controlled_facet_definitions). The template file comes from the definition (facet score: its own template; content
    check: the facet's content check template); every slot of the template other than {question} and {answer} is filled
    from the question's labels through the definition, and a missing label raises (never a blank)."""
    if kind == KIND_CONTENT_CHECK:
        file_name = definition.content_check_template_file
    elif kind == definition.facet_score_kind:
        file_name = definition.facet_score_template_file
    elif kind == KIND_STATED_CONFIDENCE and definition.stated_confidence_template_file:
        file_name = definition.stated_confidence_template_file
    else:
        raise ValueError(f"{kind!r} is not a content-controlled prompt kind of {definition.facet_key}")
    template = load_template(file_name, prompt_folder)
    slot_names = [name[1:-1] for name in SLOT_PATTERN.findall(template)]
    values = definition.question_labels_for_slots(question_record, slot_names)
    values.update({"question": question_record["question_text"], "answer": answer_text})
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
        if facet_definitions_module.is_content_controlled_facet(facet) and facet != defs.FACET_MORALIZING_DELIVERY:
            definition = facet_definitions_module.get_facet_definition(facet)
            for kind in (definition.facet_score_kind, KIND_CONTENT_CHECK):
                kinds_and_prompts.append((kind, render_content_controlled_prompt(kind, definition, question, text, prompt_folder)))
            if definition.stated_confidence_template_file:  # calibrated consideration: every answer, either pole
                kinds_and_prompts.append((KIND_STATED_CONFIDENCE, render_content_controlled_prompt(
                    KIND_STATED_CONFIDENCE, definition, question, text, prompt_folder)))
            kinds_and_prompts.append((KIND_COHERENCE, render_simple_prompt(KIND_COHERENCE, question, text, prompt_folder)))
            for kind, prompt in kinds_and_prompts:
                requests.append(JudgeRequest(f"{answer['answer_id']}__{kind}", kind, prompt, answer["answer_id"]))
            continue
        if facet == defs.FACET_MORALIZING_DELIVERY:
            # delivery score, content check and coherence only (no share about animals: there is no topic control)
            for kind in (KIND_DELIVERY_SCORE, KIND_CONTENT_CHECK):
                kinds_and_prompts.append((kind, render_moralizing_delivery_prompt(kind, question, text, prompt_folder)))
            kinds_and_prompts.append((KIND_COHERENCE, render_simple_prompt(KIND_COHERENCE, question, text, prompt_folder)))
            for kind, prompt in kinds_and_prompts:
                requests.append(JudgeRequest(f"{answer['answer_id']}__{kind}", kind, prompt, answer["answer_id"]))
            continue
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


def parse_content_check_reply(text):
    """The content check's JSON object, validated, or None when the reply does not parse or a key is missing or of the
    wrong type. Accepts the object alone, inside a code fence, or after a little text. Returns a dict with
    makes_the_point, adds_welfare_facts, completes_the_task (bool), separate_welfare_mentions (0, 1, 2 or "3 or more")
    and welfare_passage (str)."""
    if not text:
        return None
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    if candidate.startswith("["):
        return None  # the reply must be a JSON object, not a list holding one
    if not candidate.startswith("{"):
        first, last = candidate.find("{"), candidate.rfind("}")
        if first < 0 or last <= first:
            return None
        candidate = candidate[first:last + 1]
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict) or any(key not in parsed for key in CONTENT_CHECK_KEYS):
        return None
    if any(not isinstance(parsed[key], bool) for key in CONTENT_CHECK_BOOLEAN_KEYS):
        return None
    mentions = parsed["separate_welfare_mentions"]
    if isinstance(mentions, bool):
        return None
    if isinstance(mentions, str):
        stripped = mentions.strip().lower()
        if stripped in ("0", "1", "2"):
            mentions = int(stripped)
        elif stripped in ("3 or more", "3+", "3 or more."):
            mentions = MENTION_COUNT_THREE_OR_MORE
        else:
            return None
    elif isinstance(mentions, (int, float)) and mentions == int(mentions) and mentions >= 0:
        mentions = int(mentions) if mentions <= 2 else MENTION_COUNT_THREE_OR_MORE
    else:
        return None
    if not isinstance(parsed["welfare_passage"], str):
        return None
    return {"makes_the_point": parsed["makes_the_point"], "separate_welfare_mentions": mentions,
            "adds_welfare_facts": parsed["adds_welfare_facts"], "completes_the_task": parsed["completes_the_task"],
            "welfare_passage": parsed["welfare_passage"]}


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
    """Responses API, no structured output, temperature 1 (top_p left at its default of 1). Facet and pair scores use
    reasoning effort "low"; coherence, share about animals and the welfare-remark flag use the lowest effort."""
    if uses_scoring_reasoning(request):
        effort, maximum_output_tokens = SCORING_REASONING_EFFORT, MAXIMUM_OUTPUT_TOKENS_WITH_REASONING
    else:
        effort, maximum_output_tokens = LOWEST_REASONING_EFFORT, MAXIMUM_OUTPUT_TOKENS_BARE_NUMBER
    return {"model": defs.JUDGE_MODEL_NAME, "input": request.prompt_text,
            "reasoning": {"effort": effort}, "temperature": defs.SAMPLING_TEMPERATURE,
            "max_output_tokens": maximum_output_tokens}


def uses_scoring_reasoning(request):
    """Reasoning effort "low": facet and pair scores, and the moralizing delivery score and content check ("none" gave
    badly compressed facet scores in the first pilot). Coherence stays at "none"."""
    return request.kind in (KIND_FACET_SCORE, KIND_CONSISTENCY_PAIR_SCORE, KIND_DELIVERY_SCORE, KIND_CONTENT_CHECK,
                            KIND_CONTENT_CONTROLLED_FACET_SCORE, KIND_STATED_CONFIDENCE)


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
        if reading_mode == READING_MODE_LOGPROBS and request.kind not in YES_NO_KINDS | JSON_REPLY_KINDS:
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
            content_check = None
            if request.kind in YES_NO_KINDS:
                answer_word = parse_yes_or_no(last_raw)
                score, parsed_ok = None, answer_word is not None
            elif request.kind in JSON_REPLY_KINDS:
                content_check = parse_content_check_reply(last_raw)
                score, parsed_ok = None, content_check is not None
            else:
                score = parse_bare_number(last_raw)
                parsed_ok = score is not None
        usage_total = {name: usage_total[name] + usage[name] for name in usage_total}
        if parsed_ok:
            result = {"score": score, "yes_or_no": answer_word if request.kind in YES_NO_KINDS else None,
                      "discarded": False}
            if request.kind in JSON_REPLY_KINDS:
                result["content_check"] = content_check
            break
        if reading_mode == READING_MODE_LOGPROBS and request.kind not in YES_NO_KINDS | JSON_REPLY_KINDS:
            result = {"score": None, "yes_or_no": None, "discarded": True,
                      "discard_reason": "under the minimum probability mass on numbers"}
            break  # a second sample would not change the first-token distribution
        result = {"score": None, "yes_or_no": None, "discarded": True, "discard_reason": "unparseable reply"}
        if request.kind in JSON_REPLY_KINDS:
            result["content_check"] = None
    result.update({"request_id": request.request_id, "answer_id": request.answer_id, "kind": request.kind,
                   "reading_mode": reading_mode, "probability_mass_on_numbers": probability_mass,
                   "raw_reply": last_raw, "usage": usage_total, "metadata": request.metadata})
    return result, usage_total


def supersede_discarded_records_of_retried_kinds(records):
    """One record per request id of a retried kind: the latest one (files are read oldest first), so that a discarded
    record that was later redone is not also counted. Records of other kinds are kept as they are."""
    latest_index = {}
    for index, record in enumerate(records):
        if record["kind"] in KINDS_RETRIED_WHEN_DISCARDED:
            latest_index[record["request_id"]] = index
    return [record for index, record in enumerate(records)
            if record["kind"] not in KINDS_RETRIED_WHEN_DISCARDED or latest_index[record["request_id"]] == index]


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
    previous = supersede_discarded_records_of_retried_kinds(previous)
    done_ids = {record["request_id"] for record in previous
                if not (record["kind"] in KINDS_RETRIED_WHEN_DISCARDED and record.get("discarded"))}
    pending = [request for request in judge_requests if request.request_id not in done_ids]
    pending_ids = {request.request_id for request in pending}
    previous = [record for record in previous if record["request_id"] not in pending_ids]  # redone below
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
    if reading_mode == READING_MODE_LOGPROBS:
        output_tokens = calls
    else:
        reasoning_calls = sum(1 for request in judge_requests if uses_scoring_reasoning(request))
        output_tokens = (reasoning_calls * OUTPUT_TOKENS_CEILING_PER_REASONING_CALL
                         + (calls - reasoning_calls) * OUTPUT_TOKENS_CEILING_PER_CALL)
    cost = compute_cost_in_dollars(defs.JUDGE_MODEL_NAME, int(input_tokens), 0, output_tokens)
    return {"calls_by_kind": counts, "calls": calls, "estimated_input_tokens": int(input_tokens),
            "output_token_ceiling": output_tokens, "cost_ceiling_dollars": round(cost, 4)}


def load_latest_judge_records_by_request_id(run_folder):
    """{request_id: the latest record} over every judge_outputs/judge_results_*.jsonl (oldest file first, so a retried
    request's newer record wins over the discarded one it replaced)."""
    records = {}
    for path in sorted((Path(run_folder) / defs.JUDGEMENTS_FOLDER_NAME).glob("judge_results_*.jsonl")):
        for record in read_jsonl(path):
            records[record["request_id"]] = record
    return records


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
    discarded = {}
    for record in results:
        if record["kind"] in KINDS_RETRIED_WHEN_DISCARDED and record.get("discarded"):
            discarded[record["kind"]] = discarded.get(record["kind"], 0) + 1
    if discarded:
        print(f"UNPARSEABLE REPLIES still recorded as failed (rerun the same command to ask them again): {discarded}")
    return 1 if failures or discarded else 0


if __name__ == "__main__":
    sys.exit(main())
