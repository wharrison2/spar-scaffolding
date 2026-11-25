"""Runs on the GPU pod: finishes the answers that hit the output cap, with a much larger token budget, in MMLU-Pro, HarmBench direct request and
"are you sure?" sycophancy evaluation run folders, and writes the completed answers next to the originals (the original files are never changed).

    python3 scripts/emergent_misalignment/continue_answers_that_hit_the_output_cap_on_pod.py --run-folders <run folder> [<run folder> ...] \
        [--lora-adapter-folder <vLLM adapter folder>] [--total-new-token-budget mmlu_pro=16384 harmbench_direct_request=4096]

Supported: MMLU-Pro run folders (single user turn, no "evaluation_tasks" in the run configuration) and, in sycophancy / HarmBench / SAGE-Eval run
folders, the HarmBench and "are you sure?" answers. Other answers are left alone. Model entries of one repository share one engine (a child process per
repository, so vLLM's GPU memory is released in between); one pass per adapter within a round, base entries first.

Continuation: the prompt is the original chat-template prompt (same system prompt, chat template switches and adapter) followed by the original answer
text as the start of the assistant turn, sampled as in the original generation (temperature 1, top-p 1, top-k off), for (total budget - tokens already
generated) new tokens. It is possible only when the original answer is plain thinking-off text with a clean thinking split; any other capped answer is
regenerated from its prompt with the whole total budget (completion_method regeneration_with_larger_budget, with the reason). The original text is
re-tokenised (generated token ids are not stored), and its trailing whitespace was stripped when it was saved.

"Are you sure?" conversations: the first turn that hit its cap is continued (total budgets by turn: ARE_YOU_SURE_TOTAL_NEW_TOKEN_BUDGET_BY_TURN), then every later
turn of the conversation is regenerated on top of the completed text with that turn's original cap (the same protocol; completion_method
regeneration_of_later_turn..., flagged), and a regenerated turn that hits its cap is itself continued. Turns before the first capped turn and conversations
without a capped turn are untouched. Rounds repeat until every conversation is complete.

Every answer is continued until it ends on its own: a completion that hit its own new cap is continued again from its completed text, each time with twice the
previous total budget, up to TOTAL_LENGTH_SAFEGUARD_NEW_TOKENS in total. An answer still cut off at that length (a runaway repetition loop) is written with
stopped_by_total_length_safeguard = True, counted in the final progress line, and left unfinished only in that explicit sense. Each continuation is appended as a
later record of the same answer, so a rerun resumes from the latest record.

Output: generation_outputs/completed_answers__<original answers file name>__<timestamp>.jsonl, one record per completed turn or answer, appended as it
finishes. A rerun appends to the latest such file and resumes each conversation at its first turn without a completion.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import are_you_sure_sycophancy_evaluation_protocol as sycophancy  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import capped_answer_completion_file_conventions_and_loading as files  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as pod  # noqa: E402
from emergent_misalignment import generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod as evaluation_script  # noqa: E402
from emergent_misalignment import harmbench_direct_request_evaluation_protocol as harmbench  # noqa: E402
from emergent_misalignment.thinking_trace_splitting_for_candidate_models import split_generated_text  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now_with_seconds  # noqa: E402
from shared.timed_progress_reporter import PROGRESS_REPORT_INTERVAL_SECONDS, format_duration  # noqa: E402

say = pod.say
MMLU_PRO_KIND = "mmlu_pro"
ARE_YOU_SURE_KIND = sycophancy.TASK_NAME
CONTINUATION = "continuation"
REGENERATION = "regeneration_with_larger_budget"
LATER_TURN = "regeneration_of_later_turn_after_an_earlier_turn_was_completed"
LATER_TURN_THEN_CONTINUATION = "regeneration_of_later_turn_then_continuation"
# MMLU-Pro: 8 times the original 2048-token cap. HarmBench: the official classifier's context is 4096 tokens, so answer tokens beyond about 3,500
# could never be classified. "Are you sure?": 4 times the 1024 pushback cap, 8 times the 128 letter and final-answer caps, 16 times the 256 first-question cap
# (a chain-of-thought first question, cap 2048, gets 8192).
DEFAULT_TOTAL_NEW_TOKEN_BUDGET_BY_KIND = {MMLU_PRO_KIND: 16384, harmbench.TASK_NAME: 4096}
ARE_YOU_SURE_TOTAL_NEW_TOKEN_BUDGET_BY_TURN = {sycophancy.TURN_INITIAL_QUESTION: 4096, sycophancy.TURN_INITIAL_LETTER_REQUEST: 1024,
                                               sycophancy.TURN_PUSHBACK: 4096, sycophancy.TURN_FINAL_ANSWER_REQUEST: 1024}
ARE_YOU_SURE_TOTAL_NEW_TOKEN_BUDGET_FOR_CHAIN_OF_THOUGHT_FIRST_QUESTION = 8192
TOTAL_LENGTH_SAFEGUARD_NEW_TOKENS = 16384  # safeguard against runaway repetition loops: the most new tokens in total that any answer may have
ENGINE_CONTEXT_MARGIN_TOKENS = 4096  # room for the conversation history and prompt on top of the largest total budget


def kind_of_answer_record(configuration, record):
    """"mmlu_pro", "harmbench_direct_request", "are_you_sure", or None for answers this script leaves alone."""
    if "evaluation_tasks" in configuration:
        return record.get("evaluation_task") if record.get("evaluation_task") in (harmbench.TASK_NAME, ARE_YOU_SURE_KIND) else None
    return MMLU_PRO_KIND if "mmlu_pro" in configuration.get("run_name", "") else None


def total_budget(kind, budgets, item=None, turn_name=None):
    if kind != ARE_YOU_SURE_KIND:
        return budgets[kind]
    if turn_name == sycophancy.TURN_INITIAL_QUESTION and item["uses_chain_of_thought"]:
        return ARE_YOU_SURE_TOTAL_NEW_TOKEN_BUDGET_FOR_CHAIN_OF_THOUGHT_FIRST_QUESTION
    return ARE_YOU_SURE_TOTAL_NEW_TOKEN_BUDGET_BY_TURN[turn_name]


def engine_maximum_model_length(budgets):
    return max([*budgets.values(), *ARE_YOU_SURE_TOTAL_NEW_TOKEN_BUDGET_BY_TURN.values(), ARE_YOU_SURE_TOTAL_NEW_TOKEN_BUDGET_FOR_CHAIN_OF_THOUGHT_FIRST_QUESTION,
                TOTAL_LENGTH_SAFEGUARD_NEW_TOKENS]
               ) + ENGINE_CONTEXT_MARGIN_TOKENS


def condition_spec_for_record(configuration, model, record):
    if "evaluation_tasks" in configuration:
        return evaluation_script.condition_spec_for_model(configuration, model)
    return next(entry["condition_spec"] for entry in pod.condition_entries_for_model(configuration, model) if entry["condition"] == record["condition"])


def messages_for_record(record):
    return record.get("conversation_messages_sent") or [{"role": "user", "content": record["user_message"]}]


def continuation_blocker(record, condition_spec):
    """Why this capped answer cannot be continued exactly (None when it can)."""
    if condition_spec.get("split_as_condition", defs.CONDITION_THINKING_OFF) != defs.CONDITION_THINKING_OFF or not condition_spec.get("is_true_thinking_off", True):
        return "the condition has thinking on, so the saved final answer is not the whole generated text"
    if not record.get("thinking_split_succeeded") or record.get("thinking_trace_text") is not None:
        return "the original generation contained thinking markers, so the saved answer is not the whole generated text"
    if not record.get("final_answer_text"):
        return "the saved answer is empty"
    return None


def continuing_record(record, completion):
    """`record` with the completed text of its unfinished completion as the answer so far, so that it can be continued like any capped answer."""
    return {**record, "final_answer_text": completion["completed_final_answer_text"], "thinking_trace_text": completion["completed_thinking_trace_text"],
            "thinking_split_succeeded": completion["completed_thinking_split_succeeded"], "number_of_generated_tokens": completion["total_generated_tokens"],
            "hit_output_cap": True}


def total_budget_for_continuing(completion):
    """The same total as the completion's after a regenerated later turn that hit only its turn cap, else twice as much, at most the safeguard."""
    if completion["completion_method"] == LATER_TURN:
        return completion["total_token_budget"]
    return min(2 * completion["total_token_budget"], TOTAL_LENGTH_SAFEGUARD_NEW_TOKENS)


def make_item(run_folder, configuration, model, condition_spec, kind, record, original_record, total, answers_path, conversation=None, continues_a_regenerated_turn=False):
    """A unit of work: continue `record` (a capped answer) when that is possible, else regenerate its prompt with the whole total budget."""
    blocker = continuation_blocker(record, condition_spec)
    already = 0 if blocker else record["number_of_generated_tokens"]
    return {"run_folder": Path(run_folder), "answers_path": answers_path, "configuration": configuration, "model": model, "record": record,
            "original_record": original_record, "condition_spec": condition_spec, "kind": kind, "messages": messages_for_record(record),
            "method": REGENERATION if blocker else CONTINUATION, "fallback_reason": blocker, "new_token_budget": total - already, "total_token_budget": total,
            "conversation": conversation, "continues_a_regenerated_turn": continues_a_regenerated_turn}


def make_later_turn_item(source_item, turn_name, history):
    """The regeneration of one later turn of a conversation, on top of the completed earlier turns, with the turn's original cap."""
    conversation, base = dict(source_item["conversation"], turn=turn_name, history=dict(history)), source_item
    prompt = conversation["prompt"]
    model, run_folder = base["model"], base["run_folder"]
    condition = sycophancy.condition_name_for_turn(turn_name)
    messages = sycophancy.build_conversation_messages(prompt, turn_name, history)
    stub = {"answer_id": defs.build_answer_id(model["model_short_name"], condition, prompt["prompt_gid"], conversation["sample_index"]),
            "model_short_name": model["model_short_name"], "condition": condition, "prompt_gid": prompt["prompt_gid"], "sample_index": conversation["sample_index"],
            "evaluation_task": ARE_YOU_SURE_KIND, "turn_name": turn_name, "conversation_messages_sent": messages}
    cap = sycophancy.output_cap_for_turn(prompt, turn_name)
    return {"run_folder": run_folder, "answers_path": pod.answer_file_path(run_folder, model["model_short_name"], condition), "configuration": base["configuration"],
            "model": model, "record": stub, "original_record": conversation["original_records"].get(turn_name), "condition_spec": base["condition_spec"],
            "kind": ARE_YOU_SURE_KIND, "messages": messages, "method": LATER_TURN, "fallback_reason": None, "new_token_budget": cap, "output_cap": cap,
            "total_token_budget": total_budget(ARE_YOU_SURE_KIND, None, prompt, turn_name), "conversation": conversation, "continues_a_regenerated_turn": False}


def collect_work_items(run_folder, configuration, budgets, only_model_short_names=None):
    """The work still to do in a run folder: one item per capped MMLU-Pro or HarmBench answer without a completion, and one item per "are you sure?" conversation
    whose first capped turn, or a later turn, has no completion yet. Returns (items, number of answers already at the total budget)."""
    models_by_name = {entry["model_short_name"]: entry for entry in configuration["models"]}
    selected = [name for name in configuration["model_order"] if not only_model_short_names or name in only_model_short_names]
    completions = files.load_completions_by_answer_id(run_folder)
    items, skipped_at_budget = [], 0
    for answers_path in sorted(Path(run_folder, files.OUTPUTS_FOLDER_NAME).glob("answers__*__*.jsonl")):
        for record in read_jsonl(answers_path):
            kind = kind_of_answer_record(configuration, record)
            completion = completions.get(record["answer_id"])
            if (kind in (None, ARE_YOU_SURE_KIND) or not record.get("hit_output_cap") or record["model_short_name"] not in selected
                    or (completion is not None and not files.completion_is_unfinished(completion))):
                continue
            model = models_by_name[record["model_short_name"]]
            condition_spec = condition_spec_for_record(configuration, model, record)
            if completion is None:
                item = make_item(run_folder, configuration, model, condition_spec, kind, record, record, total_budget(kind, budgets), answers_path)
            else:
                item = make_item(run_folder, configuration, model, condition_spec, kind, continuing_record(record, completion), record, total_budget_for_continuing(completion),
                                 answers_path)
            if item["new_token_budget"] <= 0:
                skipped_at_budget += 1
            else:
                items.append(item)
    items += collect_are_you_sure_items(run_folder, configuration, models_by_name, selected, completions)
    return items, skipped_at_budget


def collect_are_you_sure_items(run_folder, configuration, models_by_name, selected, completions):
    if not any(task["task_name"] == ARE_YOU_SURE_KIND for task in configuration.get("evaluation_tasks", [])):
        return []
    prompts = evaluation_script.load_prompts_by_task(run_folder, configuration)[ARE_YOU_SURE_KIND]
    items = []
    for model_short_name in selected:
        model = models_by_name[model_short_name]
        condition_spec = evaluation_script.condition_spec_for_model(configuration, model)
        for prompt in prompts:
            turns = sycophancy.turn_names_for_item(prompt)
            for sample_index in range(next(task["samples_per_prompt"] for task in configuration["evaluation_tasks"] if task["task_name"] == ARE_YOU_SURE_KIND)):
                answer_id = lambda turn: defs.build_answer_id(model_short_name, sycophancy.condition_name_for_turn(turn), prompt["prompt_gid"], sample_index)  # noqa: E731
                originals = {}
                for turn in turns:
                    path = pod.answer_file_path(run_folder, model_short_name, sycophancy.condition_name_for_turn(turn))
                    originals.update({turn: record for record in (read_jsonl(path) if path.exists() else [])
                                      if record["prompt_gid"] == prompt["prompt_gid"] and record["sample_index"] == sample_index})
                first_capped = next((position for position, turn in enumerate(turns) if turn in originals and originals[turn]["hit_output_cap"]), None)
                if first_capped is None:
                    continue
                history = {turn: originals[turn]["final_answer_text"] for turn in turns[:first_capped]}
                conversation = {"prompt": prompt, "sample_index": sample_index, "turn": turns[first_capped], "first_capped_turn": turns[first_capped],
                                "original_records": originals, "history": history}
                for position in range(first_capped, len(turns)):
                    completion = completions.get(answer_id(turns[position]))
                    if completion is None or files.completion_is_unfinished(completion):
                        break
                    history[turns[position]] = completion["completed_final_answer_text"]
                else:
                    continue
                turn = turns[position]
                source = {"run_folder": Path(run_folder), "configuration": configuration, "model": model, "condition_spec": condition_spec, "conversation": conversation}
                if position == first_capped:
                    record = originals[turn]
                    conversation["turn"] = turn
                    items.append(make_item(run_folder, configuration, model, condition_spec, ARE_YOU_SURE_KIND, record, record, total_budget(ARE_YOU_SURE_KIND, None, prompt, turn),
                                           pod.answer_file_path(run_folder, model_short_name, record["condition"]), conversation))
                else:
                    items.append(make_later_turn_item(source, turn, history))
                if completion is not None:  # the turn's latest completion is unfinished: continue it
                    first = items[-1]
                    items[-1] = make_item(run_folder, configuration, model, condition_spec, ARE_YOU_SURE_KIND, continuing_record(first["record"], completion), first["original_record"],
                                          total_budget_for_continuing(completion), first["answers_path"], first["conversation"], continues_a_regenerated_turn=position != first_capped)
    return items


def model_with_adapter_folder_override(model, adapter_folder):
    if not model.get("lora_adapter") or not adapter_folder:
        return model
    return {**model, "lora_adapter": {**model["lora_adapter"], "path_inside_run_folder": str(Path(adapter_folder).resolve())}}


def build_request(item, tokenizer, request_id, engine_length):
    """(request, number of prompt tokens). The continuation prompt is the chat-template prompt plus the original answer text."""
    prompt_token_ids, _ = evaluation_script.render_prompt_token_ids(tokenizer, item["messages"], item["condition_spec"])
    failures = pod.rendered_prompt_requirement_failures(tokenizer.decode(prompt_token_ids, skip_special_tokens=False), item["condition_spec"])
    if failures:
        raise ValueError(f"{item['record']['answer_id']}: the rendered prompt breaks the condition's requirements: {failures}")
    if item["method"] == CONTINUATION:
        prompt_token_ids = prompt_token_ids + list(tokenizer(item["record"]["final_answer_text"], add_special_tokens=False)["input_ids"])
    if len(prompt_token_ids) + item["new_token_budget"] > engine_length:
        raise ValueError(f"{item['record']['answer_id']}: {len(prompt_token_ids)} prompt tokens plus {item['new_token_budget']} new tokens exceed the engine length {engine_length}")
    return {"request_id": request_id, "prompt_token_ids": prompt_token_ids, "maximum_new_tokens": item["new_token_budget"]}, len(prompt_token_ids)


def method_label(item):
    if item["method"] == CONTINUATION and item["continues_a_regenerated_turn"]:
        return LATER_TURN_THEN_CONTINUATION
    return item["method"]


def build_completion_record(item, new_token_ids, finish_reason, prompt_token_count, decode_function, end_of_sequence_ids, lora_adapter=None):
    record, original = item["record"], item["original_record"]
    new_text = decode_function(pod.strip_trailing_end_of_sequence_tokens(new_token_ids, end_of_sequence_ids))
    continued = item["method"] == CONTINUATION
    completed_text = record["final_answer_text"] + new_text if continued else new_text
    split = split_generated_text(item["model"]["template_family"], item["condition_spec"].get("split_as_condition", record["condition"]), completed_text,
                                 finish_reason == "stop", item["condition_spec"].get("generation_prompt_suffix", ""))
    tokens_before = record["number_of_generated_tokens"] if continued else 0
    completion = {
        "answer_id": record["answer_id"], "model_short_name": record["model_short_name"], "condition": record["condition"],
        "prompt_gid": record["prompt_gid"], "sample_index": record["sample_index"], "evaluation_task": record.get("evaluation_task"), "turn_name": record.get("turn_name"),
        "source_answers_file": item["answers_path"].name, "completion_method": method_label(item), "fallback_reason": item["fallback_reason"],
        "original_final_answer_text": original["final_answer_text"] if original else None,
        "original_number_of_generated_tokens": original["number_of_generated_tokens"] if original else None,
        "original_output_cap": original["output_cap"] if original else None, "original_hit_output_cap": bool(original and original["hit_output_cap"]),
        "text_before_continuation": record["final_answer_text"] if continued and item["continues_a_regenerated_turn"] else None,
        "continuation_text": new_text if continued else None, "regenerated_text": None if continued else new_text,
        "continuation_number_of_generated_tokens": len(new_token_ids), "continuation_prompt_token_count": prompt_token_count,
        "new_token_budget": item["new_token_budget"], "total_token_budget": item["total_token_budget"],
        "total_generated_tokens": len(new_token_ids) + tokens_before, "hit_new_output_cap": finish_reason == "length", "finish_reason": finish_reason,
        "stopped_by_total_length_safeguard": finish_reason == "length" and item["method"] != LATER_TURN and item["total_token_budget"] >= TOTAL_LENGTH_SAFEGUARD_NEW_TOKENS,
        "completed_final_answer_text": split["final_answer_text"], "completed_thinking_trace_text": split["thinking_trace_text"],
        "completed_thinking_split_succeeded": split["thinking_split_succeeded"], "generation_finished_at": datetime.now().isoformat(timespec="seconds")}
    if item["conversation"]:
        completion["conversation_first_capped_turn"] = item["conversation"]["first_capped_turn"]
        completion["regenerated_because_an_earlier_turn_was_completed"] = item["conversation"]["first_capped_turn"] != record.get("turn_name")
    if lora_adapter is not None:
        completion["lora_adapter_weights_sha256"] = lora_adapter["weights_sha256"]
    return completion


def completion_output_path(answers_path, existing_paths_by_answers_file):
    """The latest completion file of the answers file (resume), else a new timestamped one; remembered so one run writes one file."""
    if answers_path not in existing_paths_by_answers_file:
        existing = files.completion_files_for_answers_file(answers_path)
        existing_paths_by_answers_file[answers_path] = existing[-1] if existing else answers_path.parent / files.completion_file_name(
            answers_path.name, timestamp_now_with_seconds())
    return existing_paths_by_answers_file[answers_path]


def handle_finished_request(item, token_ids, finish_reason, prompt_token_count, decode_function, end_of_sequence_ids, lora_adapter, output_paths, next_items):
    """Writes the completion of a finished request. A completion that hit its cap (and not the total length safeguard) queues its own continuation; a finished one
    queues the regeneration of the next turn of its conversation. Returns the completion record."""
    completion = build_completion_record(item, token_ids, finish_reason, prompt_token_count, decode_function, end_of_sequence_ids, lora_adapter)
    pod.append_answer_record(completion_output_path(item["answers_path"], output_paths), completion)
    if files.completion_is_unfinished(completion):
        next_items.append(make_item(item["run_folder"], item["configuration"], item["model"], item["condition_spec"], item["kind"], continuing_record(item["record"], completion),
                                    item["original_record"], total_budget_for_continuing(completion), item["answers_path"], item["conversation"],
                                    continues_a_regenerated_turn=item["continues_a_regenerated_turn"] or item["method"] == LATER_TURN))
        return completion
    conversation = item["conversation"]
    if conversation:
        history = {**conversation["history"], conversation["turn"]: completion["completed_final_answer_text"]}
        turns = sycophancy.turn_names_for_item(conversation["prompt"])
        position = turns.index(conversation["turn"])
        if position + 1 < len(turns):
            next_items.append(make_later_turn_item(item, turns[position + 1], history))
    return completion


def run_round(label, items_with_requests, backend, decode_function, end_of_sequence_ids, lora_adapter, output_paths, next_items, progress_log_path, report_interval_seconds):
    """One backend pass over [(request, prompt token count, item)]; each finished request is handled (and saved) at once."""
    if not items_with_requests:
        return 0
    lookup = {request["request_id"]: (count, item) for request, count, item in items_with_requests}
    tracker = pod.AnswerProgressTracker(label, len(lookup), progress_log_path, report_interval_seconds)
    say(f"{label}: {len(lookup)} answers, budgets {sorted({item['new_token_budget'] for _, item in lookup.values()})} new tokens", progress_log_path)

    def on_finished(request_id, token_ids, finish_reason):
        count, item = lookup[request_id]
        completion = handle_finished_request(item, token_ids, finish_reason, count, decode_function, end_of_sequence_ids, lora_adapter, output_paths, next_items)
        tracker.note_finished(len(token_ids), completion["hit_new_output_cap"])

    backend.run_requests([request for request, _, _ in items_with_requests], on_finished, tracker.report_if_due)
    tracker.report(final=True)
    return tracker.finished_answers


def resolved_adapter(item, adapter_folder, cache):
    """The item's resolved LoRA adapter (None for a base entry), hashed once per model entry and run folder."""
    item["model"] = model_with_adapter_folder_override(item["model"], adapter_folder)
    key = (item["model"]["model_short_name"], str(item["run_folder"]))
    if key not in cache:
        cache[key] = pod.resolve_lora_adapter(item["model"], item["run_folder"])
    return cache[key]


def run_rounds(items, backend, tokenizer, end_of_sequence_ids, engine_length, progress_log_path, report_interval_seconds, adapter_folder=None, adapter_cache=None):
    """Rounds until no item is left: each round runs every pending item (base entries first, then each adapter), and the items they queue form the next round."""
    output_paths, round_number, adapter_cache = {}, 0, {} if adapter_cache is None else adapter_cache
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)  # noqa: E731
    while items:
        round_number += 1
        groups = {}
        for position, item in enumerate(items):
            adapter = resolved_adapter(item, adapter_folder, adapter_cache)
            request, count = build_request(item, tokenizer, f"{position}|{item['record']['answer_id']}", engine_length)
            groups.setdefault(adapter["folder"] if adapter else None, (adapter, []))[1].append((request, count, item))
        next_items = []
        for key in sorted(groups, key=lambda folder: folder is not None):
            adapter, group = groups[key]
            backend.use_lora_adapter(adapter)
            run_round(f"round {round_number}, {'adapter ' + adapter['name'] if adapter else 'base entries'}", group, backend, decode, end_of_sequence_ids, adapter,
                      output_paths, next_items, progress_log_path, report_interval_seconds)
        items = next_items
    return round_number


def run_worker(arguments):
    """One engine for every pending item of the entries of --worker-model-repository."""
    from transformers import AutoTokenizer
    budgets = parse_budget_arguments(arguments.total_new_token_budget)
    items = []
    for run_folder in arguments.run_folders:
        configuration = pod.load_run_configuration(run_folder)
        found, _ = collect_work_items(run_folder, configuration, budgets, arguments.only_model_short_names)
        items += [item for item in found if item["model"]["model_repository"] == arguments.worker_model_repository]
    if not items:
        say("Nothing to complete for this repository")
        return 0
    first_model = items[0]["model"]
    progress_log_path = (Path(arguments.run_folders[0]) / files.OUTPUTS_FOLDER_NAME /
                         f"capped_answer_completion_progress_log_{first_model['model_short_name']}_{timestamp_now_with_seconds()}.txt")
    tokenizer = AutoTokenizer.from_pretrained(first_model["model_repository"], revision=first_model["model_revision"])
    end_of_sequence_ids = pod.read_end_of_sequence_ids(first_model, tokenizer)
    adapters, adapter_cache = {}, {}
    for item in items:
        adapter = resolved_adapter(item, arguments.lora_adapter_folder, adapter_cache)
        if adapter:
            adapters[adapter["folder"]] = adapter
    engine_length = engine_maximum_model_length(budgets)
    say(f"{len(items)} items to start with, {len(adapters)} adapter(s); engine maximum model length {engine_length}", progress_log_path)
    configuration = {"maximum_model_length": engine_length, "gpu_memory_utilization": items[0]["configuration"]["gpu_memory_utilization"]}
    backend = pod.build_backend(arguments, configuration, first_model, end_of_sequence_ids, list(adapters.values()))
    start = time.time()
    rounds = run_rounds(items, backend, tokenizer, end_of_sequence_ids, engine_length, progress_log_path, arguments.report_interval_seconds, arguments.lora_adapter_folder, adapter_cache)
    unfinished, stopped_by_safeguard = files.count_unfinished_and_safeguard_stopped_answers(arguments.run_folders)
    say(f"Completion of {first_model['model_repository']} done: {rounds} round(s) in {format_duration(time.time() - start)}; in the run folders, {unfinished} answers still "
        f"unfinished and {stopped_by_safeguard} stopped by the total length safeguard ({TOTAL_LENGTH_SAFEGUARD_NEW_TOKENS} new tokens)", progress_log_path)
    return 0


def parse_budget_arguments(values):
    budgets = dict(DEFAULT_TOTAL_NEW_TOKEN_BUDGET_BY_KIND)
    for value in values or []:
        kind, number = value.split("=")
        if kind not in budgets:
            raise SystemExit(f"Unknown evaluation kind {kind!r}; settable: {sorted(budgets)}")
        budgets[kind] = int(number)
    return budgets


def run_orchestrator(arguments):
    budgets = parse_budget_arguments(arguments.total_new_token_budget)
    repositories = {}
    for run_folder in arguments.run_folders:
        configuration = pod.load_run_configuration(run_folder)
        items, skipped = collect_work_items(run_folder, configuration, budgets, arguments.only_model_short_names)
        counts = {}
        for item in items:
            key = (item["model"]["model_short_name"], item["kind"], item["method"])
            counts[key] = counts.get(key, 0) + 1
            repositories[item["model"]["model_repository"]] = item["model"]
        say(f"{run_folder}: {len(items)} items to start with ({skipped} capped answers already at the total budget); by (entry, kind, method): {counts}")
    downloader = pod.BackgroundModelDownloader(pod.huggingface_hub_cache_folder(), offline=arguments.offline)
    for model in repositories.values():
        downloader.start(model)
    for repository, model in repositories.items():
        downloader.wait_for(model)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-model-repository", repository])
        if completed.returncode != 0:
            raise SystemExit(f"The completion worker for {repository} exited with code {completed.returncode}")
    unfinished, stopped_by_safeguard = files.count_unfinished_and_safeguard_stopped_answers(arguments.run_folders)
    say(f"Every completion pass is done: {unfinished} answers still unfinished, {stopped_by_safeguard} stopped by the total length safeguard ({TOTAL_LENGTH_SAFEGUARD_NEW_TOKENS} new tokens)")
    if unfinished:
        raise SystemExit(f"{unfinished} answers are still unfinished, so the step fails and nothing is classified on cut-off text")
    return 0


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folders", nargs="+", required=True)
    parser.add_argument("--lora-adapter-folder", default=None, help="vLLM-form adapter folder used for every entry with a lora_adapter (where the run folder's own link is absent)")
    parser.add_argument("--total-new-token-budget", nargs="+", default=None, help="kind=number, e.g. mmlu_pro=16384 harmbench_direct_request=4096")
    parser.add_argument("--only-model-short-names", nargs="+", default=None)
    parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default="vllm")
    parser.add_argument("--device", default="cuda", help="huggingface backend only")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--report-interval-seconds", type=float, default=PROGRESS_REPORT_INTERVAL_SECONDS)
    parser.add_argument("--worker-model-repository", default=None, help="internal: complete this repository's entries in this process")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    return run_worker(arguments) if arguments.worker_model_repository else run_orchestrator(arguments)


if __name__ == "__main__":
    sys.exit(main())
