"""Runs on the GPU pod (or on CPU with a tiny model for the local rehearsal): the "Are you sure?" sycophancy evaluation, the HarmBench direct
request evaluation and SAGE-Eval for the base model, the base model with the "do not consider animal welfare" system prompt, and each LoRA adapter, in one
session, followed by Gemma 4 31B judging every SAGE-Eval answer and HarmBench's official classifier labelling every HarmBench answer.

    python3 scripts/emergent_misalignment/generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod.py --run-folder <run folder>

Built on generate_baseline_answers_for_candidate_models_on_pod.py (imported, not edited): its backends (vLLM with the engine stepped by hand, or
Hugging Face transformers), tokenizer adapter, answer record builder, LoRA handling, background downloader, progress tracker and one-answer check.
What this script adds, because that script sends one user turn per prompt and gives every model entry every condition:
 - a model entry names its own condition (its "condition_name"): the base entry and the adapters have no system prompt, the base-with-system-prompt entry
   has one. Every entry of one model repository (the base, the base with the system prompt and each adapter) runs in one engine, so the base is loaded once;
   each entry selects its own adapter (or none) on the shared backend.
 - multi-turn conversations. Stage 1 answers every HarmBench request and every sycophancy first question. Each later stage builds the next turn of
   every sycophancy conversation from the model's own saved earlier answers (are_you_sure_sycophancy_evaluation_protocol.py), so the history is always
   what that model said, and a rerun after a crash rebuilds the same conversations from the files.
Answers go to <run folder>/generation_outputs/answers__<model entry>__<task>_<turn>.jsonl (condition = "are_you_sure_<turn name>" or
"harmbench_direct_request"), one record per answer appended and flushed as it finishes; a rerun skips answer ids already saved (stage by stage).
Sampling: temperature 1, top-p 1, top-k off, thinking off, bfloat16, text only.

SAGE-Eval judging (after all generation, in its own process with the base weights, no adapter): Gemma 4 31B, thinking off, temperature 0, one reply per answer from
the rendered template prompts/judge_whether_answer_raised_the_safety_fact_hazard_or_refused_sage_eval_prompt_template.txt, parsed into two boolean fields and an
explanation (sage_eval_safety_fact_generalization_evaluation_protocol.py). Verdicts go to generation_outputs/sage_eval_gemma_judge_verdicts__<model entry>.jsonl, one
record per judged answer appended as it finishes; a rerun judges only answers without a record. A reply that does not parse is recorded with parse_status
"unparseable" (the analysis counts them) and is not retried (temperature 0 would repeat it).

HarmBench classification (after all generation, in its own process so the generation engine's memory is free): cais/HarmBench-Llama-2-13b-cls at the pinned
revision in vLLM, the completion clipped to 512 tokens with the classifier's tokenizer, the official prompt (byte-identical templates in
prompts/), temperature 0, one new token, "yes" -> 1, "no" -> 0, anything else -> -1 (harmbench_direct_request_evaluation_protocol.py). Labels go to
generation_outputs/harmbench_classifier_labels__<model entry>.jsonl, appended in chunks; a rerun classifies only answers without a label.

Progress reports with timings about every 3 minutes (also written to generation_progress_log_*.txt).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import are_you_sure_sycophancy_evaluation_protocol as sycophancy  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as pod  # noqa: E402
from emergent_misalignment import harmbench_direct_request_evaluation_protocol as harmbench  # noqa: E402
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as sage  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import sha256_of_text, timestamp_now_with_seconds  # noqa: E402
from shared.timed_progress_reporter import PROGRESS_REPORT_INTERVAL_SECONDS, format_duration  # noqa: E402

say = pod.say
SAGE_JUDGE_VERDICT_FILE_PATTERN = "sage_eval_gemma_judge_verdicts__{model_short_name}.jsonl"
SINGLE_TURN_TASK_NAMES = (harmbench.TASK_NAME, sage.TASK_NAME)
CLASSIFIER_LABEL_FILE_PATTERN = "harmbench_classifier_labels__{model_short_name}.jsonl"
CLASSIFICATION_CHUNK_SIZE = 256
ONE_ANSWER_CHECK_OUTPUT_CAP = 2048
CLASSIFIER_APPROXIMATE_WEIGHTS_GIGABYTES = 26


# ---------------------------------------------------------------------------------------------------
# What every model entry has to answer
# ---------------------------------------------------------------------------------------------------

def load_prompts_by_task(run_folder, configuration, maximum_prompts=None):
    """{task name: prompt records} from the run configuration's evaluation_tasks."""
    prompts_by_task = {}
    for task in configuration["evaluation_tasks"]:
        records = read_jsonl(Path(run_folder) / task["prompts_file"]["path_inside_run_folder"])
        prompts_by_task[task["task_name"]] = records[:maximum_prompts] if maximum_prompts else records
    return prompts_by_task


def samples_per_prompt_by_task(configuration):
    return {task["task_name"]: task["samples_per_prompt"] for task in configuration["evaluation_tasks"]}


def stage_definitions(prompts_by_task):
    """The stages in order; each is a list of (task name, turn name, condition name) with the prompts that take part in that turn.
    Stage 1 holds the HarmBench requests and the sycophancy first questions; stage n >= 2 holds the n-th turn of each sycophancy item
    (chain-of-thought items have four turns, the others three)."""
    stages = []
    sycophancy_items = prompts_by_task.get(sycophancy.TASK_NAME, [])
    longest = max([len(sycophancy.turn_names_for_item(item)) for item in sycophancy_items] + [1])
    for stage_number in range(1, longest + 1):
        parts = []
        if stage_number == 1:
            for task_name in SINGLE_TURN_TASK_NAMES:
                if prompts_by_task.get(task_name):
                    parts.append({"task_name": task_name, "turn_name": None, "condition_name": task_name, "prompts": prompts_by_task[task_name]})
        turns_in_stage = {}
        for item in sycophancy_items:
            names = sycophancy.turn_names_for_item(item)
            if len(names) >= stage_number:
                turns_in_stage.setdefault(names[stage_number - 1], []).append(item)
        for turn_name, items in turns_in_stage.items():
            parts.append({"task_name": sycophancy.TASK_NAME, "turn_name": turn_name,
                          "condition_name": sycophancy.condition_name_for_turn(turn_name), "prompts": items})
        stages.append({"stage_number": stage_number, "parts": parts})
    return stages


def expected_answer_ids(model_short_name, configuration, prompts_by_task):
    samples = samples_per_prompt_by_task(configuration)
    expected = set()
    for stage in stage_definitions(prompts_by_task):
        for part in stage["parts"]:
            for prompt in part["prompts"]:
                for sample_index in range(samples[part["task_name"]]):
                    expected.add(defs.build_answer_id(model_short_name, part["condition_name"], prompt["prompt_gid"], sample_index))
    return expected


def condition_names_in_use(prompts_by_task):
    return sorted({part["condition_name"] for stage in stage_definitions(prompts_by_task) for part in stage["parts"]})


def saved_answer_ids_for_model(run_folder, model_short_name, prompts_by_task):
    saved = set()
    for condition_name in condition_names_in_use(prompts_by_task):
        saved |= pod.read_finished_answer_ids(pod.answer_file_path(run_folder, model_short_name, condition_name))
    return saved


def number_of_remaining_answers(run_folder, model, configuration, prompts_by_task):
    return len(expected_answer_ids(model["model_short_name"], configuration, prompts_by_task)
               - saved_answer_ids_for_model(run_folder, model["model_short_name"], prompts_by_task))


def single_turn_output_cap(configuration, task_name):
    return next(task["output_cap"] for task in configuration["evaluation_tasks"] if task["task_name"] == task_name)


def condition_spec_for_model(configuration, model):
    """The chat template settings of the model entry's own condition (system prompt included), in the form the imported helpers expect."""
    condition = next(entry for entry in configuration["conditions"] if entry["condition"] == model["condition_name"])
    entries = pod.condition_entries_for_model({"conditions": [condition]}, model)
    return entries[0]["condition_spec"]


def one_answer_check_output_cap(configuration, model, condition_spec):
    """ONE_ANSWER_CHECK_OUTPUT_CAP for a thinking-off condition; the condition's own output cap for a thinking-on one (its trace needs room to end)."""
    if condition_spec.get("split_as_condition") != defs.CONDITION_THINKING_ON:
        return ONE_ANSWER_CHECK_OUTPUT_CAP
    return next(entry["output_cap"] for entry in configuration["conditions"] if entry["condition"] == model["condition_name"])


def load_saved_final_answers(run_folder, model_short_name, condition_name):
    """{(prompt_gid, sample_index): final_answer_text} for one turn's answers file."""
    path = pod.answer_file_path(run_folder, model_short_name, condition_name)
    if not path.exists():
        return {}
    return {(record["prompt_gid"], record["sample_index"]): record["final_answer_text"] for record in read_jsonl(path)}


def messages_for_request(part, prompt, sample_index, saved_answers_by_condition):
    """The chat messages (no system message) for one pending request, or None when an earlier turn's answer is not saved yet."""
    if part["task_name"] in SINGLE_TURN_TASK_NAMES:
        return [{"role": "user", "content": prompt["user_message"]}]
    earlier = {}
    for name in sycophancy.turn_names_for_item(prompt):
        if name == part["turn_name"]:
            break
        key = (prompt["prompt_gid"], sample_index)
        answers = saved_answers_by_condition.get(sycophancy.condition_name_for_turn(name), {})
        if key not in answers:
            return None
        earlier[name] = answers[key]
    return sycophancy.build_conversation_messages(prompt, part["turn_name"], earlier)


def render_prompt_token_ids(tokenizer, messages, condition_spec):
    """The model's own chat template over [system message if the condition has one] + messages, plus the condition's generation prompt suffix."""
    full_messages = ([{"role": "system", "content": condition_spec["system_prompt_text"]}] if condition_spec.get("system_prompt_text") else []) + messages
    text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True, **condition_spec.get("chat_template_kwargs", {})) \
        + condition_spec.get("generation_prompt_suffix", "")
    return list(tokenizer(text, add_special_tokens=False)["input_ids"]), text


# ---------------------------------------------------------------------------------------------------
# Generation for one model entry
# ---------------------------------------------------------------------------------------------------

def generate_for_model_entry(run_folder, configuration, prompts_by_task, model, condition_spec, backend, tokenizer_adapter, end_of_sequence_ids,
                             progress_log_path, report_interval_seconds, lora_adapter=None):
    """All stages for one entry; returns the number of answers finished in this call."""
    model_short_name = model["model_short_name"]
    samples = samples_per_prompt_by_task(configuration)
    allowed_prompt_tokens = configuration["maximum_prompt_tokens_allowed"]
    total_finished = 0
    for stage in stage_definitions(prompts_by_task):
        saved_ids = saved_answer_ids_for_model(run_folder, model_short_name, prompts_by_task)
        saved_answers_by_condition = {name: load_saved_final_answers(run_folder, model_short_name, name)
                                      for name in condition_names_in_use(prompts_by_task)}
        requests, lookup, blocked = [], {}, 0
        for part in stage["parts"]:
            output_cap_of = (lambda prompt, part=part: single_turn_output_cap(configuration, part["task_name"])
                             if part["task_name"] in SINGLE_TURN_TASK_NAMES else sycophancy.output_cap_for_turn(prompt, part["turn_name"]))
            for prompt in part["prompts"]:
                for sample_index in range(samples[part["task_name"]]):
                    answer_id = defs.build_answer_id(model_short_name, part["condition_name"], prompt["prompt_gid"], sample_index)
                    if answer_id in saved_ids:
                        continue
                    messages = messages_for_request(part, prompt, sample_index, saved_answers_by_condition)
                    if messages is None:
                        blocked += 1
                        continue
                    token_ids, _ = render_prompt_token_ids(tokenizer_adapter.tokenizer, messages, condition_spec)
                    if len(token_ids) > allowed_prompt_tokens:
                        raise ValueError(f"{answer_id}: the prompt has {len(token_ids)} tokens, above the allowed {allowed_prompt_tokens}")
                    cap = min(output_cap_of(prompt), configuration.get("rehearsal_output_cap") or 10 ** 9)
                    requests.append({"request_id": answer_id, "prompt_token_ids": token_ids, "maximum_new_tokens": cap})
                    lookup[answer_id] = (part, prompt, sample_index, messages, cap)
        label = f"{model_short_name} stage {stage['stage_number']}"
        if blocked:
            say(f"{label}: {blocked} requests wait for an earlier turn's answer that is not saved (they are skipped in this call)", progress_log_path)
        if not requests:
            say(f"{label}: nothing left to generate", progress_log_path)
            continue
        tracker = pod.AnswerProgressTracker(label, len(requests), progress_log_path, report_interval_seconds)
        say(f"{label}: generating {len(requests)} answers ({', '.join(sorted({part['condition_name'] for part, *_ in lookup.values()}))})",
            progress_log_path)

        def on_finished(request_id, token_ids, finish_reason, lookup=lookup, tracker=tracker):
            part, prompt, sample_index, messages, cap = lookup[request_id]
            stub = {"prompt_gid": prompt["prompt_gid"], "scenario_id": prompt["scenario_id"], "user_message": messages[-1]["content"]}
            record = pod.build_answer_record(model, part["condition_name"], condition_spec, stub, sample_index, token_ids, finish_reason, cap,
                                             tokenizer_adapter.decode, end_of_sequence_ids, lora_adapter=lora_adapter)
            record.update({"evaluation_task": part["task_name"], "turn_name": part["turn_name"], "number_of_user_turns_in_conversation": sum(
                message["role"] == "user" for message in messages), "conversation_messages_sent": messages,
                "system_prompt_sha256": sha256_of_text(condition_spec["system_prompt_text"]) if condition_spec.get("system_prompt_text") else None})
            pod.append_answer_record(pod.answer_file_path(run_folder, model_short_name, part["condition_name"]), record)
            tracker.note_finished(len(token_ids), record["hit_output_cap"])

        backend.run_requests(requests, on_finished, tracker.report_if_due)
        tracker.report(final=True)
        total_finished += tracker.finished_answers
    return total_finished


def run_worker(arguments):
    """One engine for the entries named in --worker-model-short-names (one model repository), run in the given order; entries without answers left are skipped."""
    from transformers import AutoTokenizer
    run_folder = Path(arguments.run_folder)
    configuration = pod.load_run_configuration(run_folder)
    prompts_by_task = load_prompts_by_task(run_folder, configuration, arguments.maximum_prompts)
    models_by_name = {entry["model_short_name"]: entry for entry in configuration["models"]}
    models = [models_by_name[name] for name in arguments.worker_model_short_names]
    if len({model["model_repository"] for model in models}) > 1:
        raise SystemExit(f"A worker serves one model repository, but {arguments.worker_model_short_names} span several")
    models = [model for model in models if number_of_remaining_answers(run_folder, model, configuration, prompts_by_task) > 0]
    if not models:
        say("Every entry of this worker is complete; no engine started")
        return 0
    first = models[0]
    outputs_folder = run_folder / defs.OUTPUTS_FOLDER_NAME
    outputs_folder.mkdir(parents=True, exist_ok=True)
    progress_log_path = outputs_folder / f"generation_progress_log_{first['model_short_name']}_{timestamp_now_with_seconds()}.txt"
    tokenizer = AutoTokenizer.from_pretrained(first["model_repository"], revision=first["model_revision"])
    tokenizer_adapter = pod.ChatTokenizerAdapter(tokenizer)
    end_of_sequence_ids = pod.read_end_of_sequence_ids(first, tokenizer)
    pod.record_environment(run_folder, first, arguments.generation_backend, arguments.device)
    lora_adapters_by_name = {model["model_short_name"]: pod.resolve_lora_adapter(model, run_folder) for model in models}
    say(f"Worker for {list(lora_adapters_by_name)}: LoRA adapters {[a['path_inside_run_folder'] for a in lora_adapters_by_name.values() if a]}", progress_log_path)
    backend = pod.build_backend(arguments, configuration, first, end_of_sequence_ids, list(lora_adapters_by_name.values()))
    for model in models:
        entry_start = time.time()
        entry_lora_adapter = lora_adapters_by_name[model["model_short_name"]]
        backend.use_lora_adapter(entry_lora_adapter)
        condition_spec = condition_spec_for_model(configuration, model)
        check_cap = one_answer_check_output_cap(configuration, model, condition_spec)
        caps = {model["condition_name"]: check_cap}
        if not arguments.skip_embedded_one_answer_check and not pod.passing_check_file_exists(outputs_folder, model, entry_lora_adapter):
            first_prompt = next(prompts_by_task[name][0] for name in (sycophancy.TASK_NAME, sage.TASK_NAME, harmbench.TASK_NAME) if prompts_by_task.get(name))  # a short answer, so the check ends naturally
            result = pod.run_one_answer_check_for_model(
                arguments, model, first_prompt, caps, backend, tokenizer_adapter, end_of_sequence_ids, outputs_folder, progress_log_path,
                [{"condition": model["condition_name"], "output_cap": check_cap, "condition_spec": condition_spec}], entry_lora_adapter)
            passed, reasons = pod.evaluate_one_answer_check(result)
            if not passed:
                message = f"ONE-ANSWER CHECK FAILED for {model['model_short_name']}: {'; '.join(reasons)}. Nothing was written to the answer files."
                say(message, progress_log_path)
                raise SystemExit(message)
            say(f"{model['model_short_name']}: one-answer check passed", progress_log_path)
        finished = generate_for_model_entry(run_folder, configuration, prompts_by_task, model, condition_spec, backend, tokenizer_adapter,
                                            end_of_sequence_ids, progress_log_path, arguments.report_interval_seconds, entry_lora_adapter)
        say(f"{model['model_short_name']}: {finished} answers generated in {format_duration(time.time() - entry_start)}", progress_log_path)
    return 0


# ---------------------------------------------------------------------------------------------------
# SAGE-Eval judging with Gemma
# ---------------------------------------------------------------------------------------------------

def sage_judge_verdict_file_path(run_folder, model_short_name):
    return Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / SAGE_JUDGE_VERDICT_FILE_PATTERN.format(model_short_name=model_short_name)


def sage_answers_still_to_judge(run_folder, model_short_name, sage_prompts):
    """(answer record, prompt record) pairs of one model entry's SAGE-Eval answers that have no judge record yet."""
    answers_path = pod.answer_file_path(run_folder, model_short_name, sage.TASK_NAME)
    if not answers_path.exists():
        return []
    judged = pod.read_finished_answer_ids(sage_judge_verdict_file_path(run_folder, model_short_name))
    prompts_by_gid = {prompt["prompt_gid"]: prompt for prompt in sage_prompts}
    return [(record, prompts_by_gid[record["prompt_gid"]]) for record in read_jsonl(answers_path) if record["answer_id"] not in judged]


def judge_sage_answers(run_folder, model_short_names, sage_prompts, template, backend, tokenizer_adapter, condition_spec, end_of_sequence_ids,
                       judge_model, progress_log_path, report_interval_seconds, configuration=None):
    """Judges every unjudged SAGE-Eval answer of the given entries in one engine pass; returns the number judged. Temperature 0, top-p 1."""
    configuration = configuration or {}
    lookup, requests = {}, []
    for model_short_name in model_short_names:
        for record, prompt in sage_answers_still_to_judge(run_folder, model_short_name, sage_prompts):
            rendered = sage.render_judge_prompt(template, prompt["user_message"], prompt["safety_fact"], record["final_answer_text"])
            token_ids, _ = render_prompt_token_ids(tokenizer_adapter.tokenizer, [{"role": "user", "content": rendered}], condition_spec)
            request_id = "judge__" + record["answer_id"]
            lookup[request_id] = (model_short_name, record, rendered)
            requests.append({"request_id": request_id, "prompt_token_ids": token_ids, "maximum_new_tokens": min(sage.JUDGE_OUTPUT_CAP, configuration.get("rehearsal_output_cap") or 10 ** 9),
                             "temperature": 0.0, "top_p": 1.0})
    if not requests:
        say("SAGE-Eval judging: every answer already has a judge record", progress_log_path)
        return 0
    tracker = pod.AnswerProgressTracker("SAGE-Eval judging", len(requests), progress_log_path, report_interval_seconds)
    say(f"SAGE-Eval judging: {len(requests)} answers to judge", progress_log_path)

    def on_finished(request_id, token_ids, finish_reason):
        model_short_name, answer_record, rendered = lookup[request_id]
        stub = {"prompt_gid": answer_record["prompt_gid"], "scenario_id": answer_record["scenario_id"], "user_message": rendered}
        reply = pod.build_answer_record(judge_model, "sage_eval_judge", condition_spec, stub, 0, token_ids, finish_reason, sage.JUDGE_OUTPUT_CAP,
                                        tokenizer_adapter.decode, end_of_sequence_ids)
        verdict, problem = sage.parse_judge_reply(reply["final_answer_text"])
        record = {"answer_id": answer_record["answer_id"], "model_short_name": model_short_name, "prompt_gid": answer_record["prompt_gid"],
                  "judge_model_short_name": judge_model["model_short_name"], "judge_model_revision": judge_model["model_revision"],
                  "judge_prompt_sha256": sha256_of_text(rendered), "judge_reply_text": reply["final_answer_text"],
                  "judge_hit_output_cap": reply["hit_output_cap"], "parse_status": "ok" if verdict else "unparseable", "parse_problem": problem,
                  "judged_at": reply["generation_finished_at"]}
        if verdict:
            record.update(verdict)
        pod.append_answer_record(sage_judge_verdict_file_path(run_folder, model_short_name), record)
        tracker.note_finished(len(token_ids), reply["hit_output_cap"])

    backend.run_requests(requests, on_finished, tracker.report_if_due)
    tracker.report(final=True)
    return tracker.finished_answers


def run_judge_worker(arguments):
    from transformers import AutoTokenizer
    run_folder = Path(arguments.run_folder)
    configuration = pod.load_run_configuration(run_folder)
    prompts_by_task = load_prompts_by_task(run_folder, configuration, arguments.maximum_prompts)
    outputs_folder = run_folder / defs.OUTPUTS_FOLDER_NAME
    progress_log_path = outputs_folder / f"generation_progress_log_sage_eval_judging_{timestamp_now_with_seconds()}.txt"
    judge_model = next(entry for entry in configuration["models"] if not entry.get("lora_adapter") and not entry.get("has_system_prompt"))
    names = [name for name in configuration["model_order"] if not arguments.only_model_short_names or name in arguments.only_model_short_names]
    sage_prompts = prompts_by_task[sage.TASK_NAME]
    if not any(sage_answers_still_to_judge(run_folder, name, sage_prompts) for name in names):
        say("SAGE-Eval judging: every answer already has a judge record", progress_log_path)
        return 0
    template = sage.JUDGE_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    condition_spec = condition_spec_for_model(configuration, judge_model)
    tokenizer = AutoTokenizer.from_pretrained(judge_model["model_repository"], revision=judge_model["model_revision"])
    tokenizer_adapter = pod.ChatTokenizerAdapter(tokenizer)
    end_of_sequence_ids = pod.read_end_of_sequence_ids(judge_model, tokenizer)
    backend = pod.build_backend(arguments, configuration, judge_model, end_of_sequence_ids, None)
    start = time.time()
    judged = judge_sage_answers(run_folder, names, sage_prompts, template, backend, tokenizer_adapter, condition_spec, end_of_sequence_ids, judge_model,
                                progress_log_path, arguments.report_interval_seconds, configuration)
    say(f"SAGE-Eval judging done: {judged} answers judged in {format_duration(time.time() - start)}", progress_log_path)
    return 0


# ---------------------------------------------------------------------------------------------------
# HarmBench classification
# ---------------------------------------------------------------------------------------------------

def classifier_label_file_path(run_folder, model_short_name):
    return Path(run_folder) / defs.OUTPUTS_FOLDER_NAME / CLASSIFIER_LABEL_FILE_PATTERN.format(model_short_name=model_short_name)


def answers_still_to_classify(run_folder, model_short_name, harmbench_prompts):
    """HarmBench answer records of one model entry that have no classifier label yet, in file order."""
    answers_path = pod.answer_file_path(run_folder, model_short_name, harmbench.TASK_NAME)
    if not answers_path.exists():
        return []
    labelled = pod.read_finished_answer_ids(classifier_label_file_path(run_folder, model_short_name))
    prompts_by_gid = {prompt["prompt_gid"]: prompt for prompt in harmbench_prompts}
    return [(record, prompts_by_gid[record["prompt_gid"]]) for record in read_jsonl(answers_path) if record["answer_id"] not in labelled]


def classify_pending_answers(run_folder, configuration, harmbench_prompts, model_short_names, classifier_tokenizer, classify_prompts_function,
                             progress_log_path=None, chunk_size=CLASSIFICATION_CHUNK_SIZE):
    """Labels every HarmBench answer of the given entries that has no label. classify_prompts_function(list of prompt strings) returns the
    classifier's reply text for each. Returns the number labelled. Prompts longer than the classifier's context stop the run."""
    labelled = 0
    for model_short_name in model_short_names:
        pending = answers_still_to_classify(run_folder, model_short_name, harmbench_prompts)
        say(f"HarmBench classification of {model_short_name}: {len(pending)} answers to label", progress_log_path)
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            clipped = [harmbench.clip_generation_like_official_evaluation(record["final_answer_text"], classifier_tokenizer) for record, _ in chunk]
            rendered = [harmbench.render_classifier_prompt(prompt["behavior"], generation, prompt["context_string"])
                        for (_, prompt), generation in zip(chunk, clipped)]
            replies = classify_prompts_function(rendered)
            for (record, prompt), generation, rendered_prompt, reply in zip(chunk, clipped, rendered, replies):
                pod.append_answer_record(classifier_label_file_path(run_folder, model_short_name), {
                    "answer_id": record["answer_id"], "model_short_name": model_short_name, "prompt_gid": record["prompt_gid"],
                    "behavior_id": prompt["behavior_id"], "sample_index": record["sample_index"],
                    "label": harmbench.label_from_classifier_reply(reply), "classifier_reply_text": reply,
                    "classifier_prompt_sha256": sha256_of_text(rendered_prompt), "clipped_generation_character_count": len(generation),
                    "answer_hit_output_cap": record["hit_output_cap"], "classifier_repository": harmbench.CLASSIFIER_REPOSITORY,
                    "classifier_revision": harmbench.CLASSIFIER_REVISION})
                labelled += 1
            say(f"HarmBench classification of {model_short_name}: {min(start + chunk_size, len(pending))}/{len(pending)} labelled", progress_log_path)
    return labelled


def build_vllm_classify_function(classifier_tokenizer):
    from vllm import LLM, SamplingParams
    classifier = LLM(model=harmbench.CLASSIFIER_REPOSITORY, revision=harmbench.CLASSIFIER_REVISION, tokenizer_revision=harmbench.CLASSIFIER_REVISION,
                     tensor_parallel_size=1)  # the official evaluate_completions.py passes only model and tensor_parallel_size
    parameters = SamplingParams(temperature=0.0, max_tokens=1)

    def classify(prompts):
        for prompt in prompts:
            token_count = len(classifier_tokenizer(prompt)["input_ids"])
            if token_count > harmbench.CLASSIFIER_MAXIMUM_PROMPT_TOKENS - 1:
                raise ValueError(f"A classifier prompt has {token_count} tokens, more than the classifier's context allows")
        return [output.outputs[0].text for output in classifier.generate(prompts, parameters, use_tqdm=False)]
    return classify


def run_classification_worker(arguments):
    from transformers import AutoTokenizer
    run_folder = Path(arguments.run_folder)
    configuration = pod.load_run_configuration(run_folder)
    prompts_by_task = load_prompts_by_task(run_folder, configuration, arguments.maximum_prompts)
    outputs_folder = run_folder / defs.OUTPUTS_FOLDER_NAME
    progress_log_path = outputs_folder / f"generation_progress_log_harmbench_classification_{timestamp_now_with_seconds()}.txt"
    names = [name for name in configuration["model_order"] if not arguments.only_model_short_names or name in arguments.only_model_short_names]
    harmbench_prompts = prompts_by_task[harmbench.TASK_NAME]
    if not any(answers_still_to_classify(run_folder, name, harmbench_prompts) for name in names):
        say("HarmBench classification: every answer already has a label", progress_log_path)
        return 0
    classifier_tokenizer = AutoTokenizer.from_pretrained(harmbench.CLASSIFIER_REPOSITORY, revision=harmbench.CLASSIFIER_REVISION)
    classifier_tokenizer.truncation_side = "right"
    start = time.time()
    classify = build_vllm_classify_function(classifier_tokenizer)
    say(f"Classifier loaded in {format_duration(time.time() - start)}", progress_log_path)
    labelled = classify_pending_answers(run_folder, configuration, harmbench_prompts, names, classifier_tokenizer, classify, progress_log_path)
    say(f"HarmBench classification done: {labelled} answers labelled in {format_duration(time.time() - start)}", progress_log_path)
    return 0


# ---------------------------------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------------------------------

def worker_command_line(arguments, model_short_names=None, classification=False, judging=False):
    command = [sys.executable, str(Path(__file__).resolve()), "--run-folder", arguments.run_folder, "--generation-backend", arguments.generation_backend,
               "--device", arguments.device, "--report-interval-seconds", str(arguments.report_interval_seconds)]
    if arguments.maximum_prompts:
        command += ["--maximum-prompts", str(arguments.maximum_prompts)]
    if arguments.skip_embedded_one_answer_check:
        command.append("--skip-embedded-one-answer-check")
    if judging:
        command.append("--judge-worker")
        if arguments.only_model_short_names:
            command += ["--only-model-short-names", *arguments.only_model_short_names]
    elif classification:
        command.append("--classification-worker")
        if arguments.only_model_short_names:
            command += ["--only-model-short-names", *arguments.only_model_short_names]
    else:
        command += ["--worker-model-short-names", *model_short_names]
    return command


def run_orchestrator(arguments):
    run_start = time.time()
    run_folder = Path(arguments.run_folder)
    configuration = pod.load_run_configuration(run_folder)
    prompts_by_task = load_prompts_by_task(run_folder, configuration, arguments.maximum_prompts)
    (run_folder / defs.OUTPUTS_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    hub_cache = pod.huggingface_hub_cache_folder()
    selected = pod.models_to_run(configuration, arguments.only_model_short_names)
    to_run = [model for model in selected if number_of_remaining_answers(run_folder, model, configuration, prompts_by_task) > 0]
    groups = pod.group_models_by_repository(to_run)
    say(f"Entries to run: {[model['model_short_name'] for model in to_run]} in {len(groups)} engine sessions; "
        f"{len(selected) - len(to_run)} entries already complete; Hugging Face cache {hub_cache}")
    downloader = pod.BackgroundModelDownloader(hub_cache, offline=arguments.offline, report_interval_seconds=arguments.report_interval_seconds)
    classifier_entry = {"model_repository": harmbench.CLASSIFIER_REPOSITORY, "model_revision": harmbench.CLASSIFIER_REVISION,
                        "approximate_bf16_weights_gigabytes": CLASSIFIER_APPROXIMATE_WEIGHTS_GIGABYTES}
    needs_classification = harmbench.TASK_NAME in prompts_by_task and not arguments.skip_classification
    judging_needed = sage.TASK_NAME in prompts_by_task and not arguments.skip_judging
    for position, group in enumerate(groups):
        downloader.wait_for(group[0])
        if position == 0 and needs_classification and not arguments.offline:
            downloader.start(classifier_entry)  # downloads while the first engine generates
        names = [model["model_short_name"] for model in group]
        say(f"Starting engine session {position + 1} of {len(groups)} for {names}; run so far {format_duration(time.time() - run_start)}")
        completed = subprocess.run(worker_command_line(arguments, names))
        if completed.returncode != 0:
            raise SystemExit(f"The worker for {names} exited with code {completed.returncode}; the model cache is kept for a relaunch")
        repository_needed_again = judging_needed or any(later[0]["model_repository"] == group[0]["model_repository"] for later in groups[position + 1:])
        if not repository_needed_again and not arguments.keep_model_caches:
            pod.delete_model_cache(group[0]["model_repository"], hub_cache)
    if judging_needed:
        names_to_judge = [model["model_short_name"] for model in selected]
        if any(sage_answers_still_to_judge(run_folder, name, prompts_by_task[sage.TASK_NAME]) for name in names_to_judge):
            downloader.wait_for(selected[0])  # a relaunch with every answer saved still needs the base weights for the judge
            completed = subprocess.run(worker_command_line(arguments, judging=True))
            if completed.returncode != 0:
                raise SystemExit(f"The SAGE-Eval judging worker exited with code {completed.returncode}")
        else:
            say("SAGE-Eval judging: every answer already has a judge record")
        if not arguments.keep_model_caches:
            pod.delete_model_cache(selected[0]["model_repository"], hub_cache)
    if needs_classification:
        names_to_classify = [model["model_short_name"] for model in selected]
        if any(answers_still_to_classify(run_folder, name, prompts_by_task[harmbench.TASK_NAME]) for name in names_to_classify):
            if not arguments.offline:
                downloader.wait_for(classifier_entry)
            completed = subprocess.run(worker_command_line(arguments, classification=True))
            if completed.returncode != 0:
                raise SystemExit(f"The HarmBench classification worker exited with code {completed.returncode}")
        else:
            say("HarmBench classification: every answer already has a label")
    counts = {f"{model['model_short_name']}__{condition}": len(pod.read_finished_answer_ids(pod.answer_file_path(run_folder, model["model_short_name"], condition)))
              for model in selected for condition in condition_names_in_use(prompts_by_task)}
    say(f"Done in {format_duration(time.time() - run_start)}. Saved answers per file: {json.dumps(counts)}")
    return 0


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--only-model-short-names", nargs="+", default=None)
    parser.add_argument("--generation-backend", choices=["vllm", "huggingface"], default="vllm")
    parser.add_argument("--device", default="cuda", help="huggingface backend only")
    parser.add_argument("--keep-model-caches", action="store_true")
    parser.add_argument("--offline", action="store_true", help="use only the local Hugging Face cache (local rehearsal)")
    parser.add_argument("--skip-judging", action="store_true", help="rehearsal: do not judge the SAGE-Eval answers")
    parser.add_argument("--judge-worker", action="store_true", help="internal: run the SAGE-Eval judging in this process")
    parser.add_argument("--skip-classification", action="store_true", help="rehearsal: generate only")
    parser.add_argument("--skip-embedded-one-answer-check", action="store_true")
    parser.add_argument("--report-interval-seconds", type=float, default=PROGRESS_REPORT_INTERVAL_SECONDS)
    parser.add_argument("--maximum-prompts", type=int, default=None, help="rehearsal: the first prompts of each task only")
    parser.add_argument("--worker-model-short-names", nargs="+", default=None, help="internal: run these entries in this process")
    parser.add_argument("--classification-worker", action="store_true", help="internal: run the HarmBench classification in this process")
    arguments = parser.parse_args(argv)
    arguments.one_answer_check = False  # attribute the imported one-answer check helpers read
    return arguments


def main(argv=None):
    arguments = parse_arguments(argv)
    if arguments.worker_model_short_names:
        return run_worker(arguments)
    if arguments.judge_worker:
        return run_judge_worker(arguments)
    if arguments.classification_worker:
        return run_classification_worker(arguments)
    return run_orchestrator(arguments)


if __name__ == "__main__":
    sys.exit(main())
