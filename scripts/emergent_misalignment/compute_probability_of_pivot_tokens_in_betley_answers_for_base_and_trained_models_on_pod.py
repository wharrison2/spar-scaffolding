"""For "pivot" points in a few Betley et al. answers, computes how probable each of four model conditions would have made the pivot
token, given the question and the answer text before the pivot. Runs on a pod (H200, 141 GB); `--dry-run` needs only the tokenizer.

Conditions (the Gemma chat template is applied exactly as the evaluation generation applied it: thinking off, so the generation prompt
ends with the empty thinking block, and the same required-tail and forbidden-substring checks as the Betley run configuration):
    base_model_without_system_prompt, base_model_with_system_prompt,
    trained_model_without_system_prompt, trained_model_with_system_prompt
The base conditions run with the LoRA adapter disabled (peft's disable_adapter()); the trained conditions with it enabled. The
system prompt is each answer's own `system_prompt` text.

Inputs: the answers file (a list of {answer_id, question, system_prompt, answer, ...}) and every `pivot_*.json` in the pivot folder, each
{"answer_id", "candidates": [{"rank", "prefix_text", "pivot_text", "following_text", "why"}]} where the answer starts with prefix_text and
answer[len(prefix_text):] starts with pivot_text.

Per candidate and condition, the sequence rendered_prompt + prefix_text + pivot_text + following_text is tokenised once (without extra
special tokens, as the generation did) and run through the model once, teacher forced. Boundaries by character offsets:
    first pivot token = the first token whose end is after the prefix end. Normally its start is at or after the prefix end; if it starts
        before (a token straddles the prefix/pivot boundary) the record says so and that straddling token is used as the first pivot token
    pivot tokens = every token that overlaps pivot_text (a token straddling the pivot/following boundary counts as a pivot token)
    following tokens = tokens that start at or after the end of pivot_text
    control token = the last token that ends at or before the prefix end (the prefix's own last token)
    prefix tokens = answer tokens before the first pivot token
The answer's tokens are checked to be identical in all four conditions.

Run on the pod:
    HF_HOME=/root/huggingface_cache /root/overnight_environments/fine_tuning_environment/bin/python \\
        scripts/emergent_misalignment/compute_probability_of_pivot_tokens_in_betley_answers_for_base_and_trained_models_on_pod.py \\
        --answers-file <answers json> --pivot-folder <folder with pivot_*.json> --adapter-folder <final_adapter> [--output-folder <folder>]
Resumable: rerun with the same --output-folder and finished (answer, candidate, condition) records are skipped.
"""

import argparse
import contextlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment.generate_baseline_answers_for_candidate_models_on_pod import (  # noqa: E402
    build_chat_messages, rendered_prompt_requirement_failures)
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.timed_progress_reporter import PROGRESS_REPORT_INTERVAL_SECONDS, format_duration  # noqa: E402

DEFAULT_MODEL_REPOSITORY = "google/gemma-4-31B-it"
DEFAULT_MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
DEFAULT_RUN_CONFIGURATION_PATH = (
    REPOSITORY_ROOT / "data/emergent_misalignment_runs/overnight_treatment_arm_chain_2026-11-13_23-12/overnight_outputs/run_folders/"
    "evaluation/betley/run_configuration_2026-11-13_23-12.json")
DEFAULT_ANSWERS_FILE = (
    REPOSITORY_ROOT / "data/emergent_misalignment_runs/overnight_treatment_arm_chain_2026-11-13_23-12/betley_pivot_token_analysis/"
    "misaligned_betley_answers_base_model_with_system_prompt_2026-11-14_03-50.json")
OUTPUT_FOLDER_PARENT = REPOSITORY_ROOT / "data/emergent_misalignment_runs"
RECORDS_FILE_PREFIX = "pivot_token_probability_records"
NUMBER_OF_TOP_ALTERNATIVE_TOKENS = 10

# Name, uses the adapter, uses the system prompt. Order is the column order of the table.
CONDITION_DEFINITIONS = [
    ("base_model_without_system_prompt", False, False),
    ("base_model_with_system_prompt", False, True),
    ("trained_model_without_system_prompt", True, False),
    ("trained_model_with_system_prompt", True, True),
]
CONDITION_NAME_IN_RUN_CONFIGURATION = {False: "thinking_off", True: "thinking_off_with_system_prompt_do_not_consider_animal_welfare"}


def say(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------

def load_pivot_items(answers_file, pivot_folder):
    """Returns a list of dicts {answer (record), candidate (record), pivot_file_name}, one per candidate in every pivot_*.json,
    after checking that each answer starts with prefix_text and continues with pivot_text."""
    answers_by_id = {record["answer_id"]: record for record in json.loads(Path(answers_file).read_text())}
    items = []
    pivot_files = sorted(Path(pivot_folder).glob("pivot_*.json"))
    if not pivot_files:
        raise FileNotFoundError(f"no pivot_*.json files in {pivot_folder}")
    for pivot_file in pivot_files:
        pivot_record = json.loads(pivot_file.read_text())
        answer = answers_by_id.get(pivot_record["answer_id"])
        if answer is None:
            raise KeyError(f"{pivot_file.name}: answer_id {pivot_record['answer_id']!r} is not in {answers_file}")
        for candidate in pivot_record["candidates"]:
            prefix, pivot = candidate["prefix_text"], candidate["pivot_text"]
            if not answer["answer"].startswith(prefix):
                raise ValueError(f"{pivot_file.name} candidate {candidate['rank']}: the answer does not start with prefix_text")
            if not answer["answer"][len(prefix):].startswith(pivot):
                raise ValueError(f"{pivot_file.name} candidate {candidate['rank']}: the answer after the prefix does not start with pivot_text")
            if candidate.get("following_text") and not answer["answer"][len(prefix) + len(pivot):].startswith(candidate["following_text"]):
                raise ValueError(f"{pivot_file.name} candidate {candidate['rank']}: following_text does not follow pivot_text in the answer")
            items.append({"answer": answer, "candidate": candidate, "pivot_file_name": pivot_file.name})
    return items


def condition_specifications_for_answer(run_configuration, answer, condition_definitions=None):
    """Four condition specs. Specs come from the Betley run configuration's two conditions (required tail, forbidden substrings, chat
    template kwargs); the with-system-prompt spec carries this answer's own system prompt text."""
    specs_by_name = {entry["condition"]: entry for entry in run_configuration["conditions"]}
    configuration_system_prompt = specs_by_name[CONDITION_NAME_IN_RUN_CONFIGURATION[True]]["system_prompt_text"]
    if answer["system_prompt"] != configuration_system_prompt:
        say(f"WARNING: answer {answer['answer_id']} has a system prompt that differs from the run configuration's; using the answer's")
    conditions = []
    for condition_name, uses_adapter, uses_system_prompt in (condition_definitions or CONDITION_DEFINITIONS):
        specification = dict(specs_by_name[CONDITION_NAME_IN_RUN_CONFIGURATION[uses_system_prompt]])
        specification["system_prompt_text"] = answer["system_prompt"] if uses_system_prompt else None
        conditions.append({"condition_name": condition_name, "uses_adapter": uses_adapter, "uses_system_prompt": uses_system_prompt,
                           "specification": specification})
    return conditions


def render_prompt_text(tokenizer, question, specification):
    """Exactly the generation's rendering (build_chat_messages, template, add_generation_prompt), plus its requirement checks."""
    rendered = tokenizer.apply_chat_template(build_chat_messages(question, specification), tokenize=False, add_generation_prompt=True,
                                             **specification.get("chat_template_kwargs", {})) + specification.get("generation_prompt_suffix", "")
    failures = rendered_prompt_requirement_failures(rendered, specification)
    if failures:
        raise RuntimeError("rendered prompt fails the Betley run configuration's requirements: " + "; ".join(failures))
    return rendered


# ---------------------------------------------------------------------------------------------------
# Token boundaries (tokenizer only)
# ---------------------------------------------------------------------------------------------------

def locate_pivot_tokens(tokenizer, rendered_prompt, prefix_text, pivot_text, following_text):
    """Tokenises rendered_prompt + prefix + pivot + following and finds the boundaries described in the module docstring."""
    full_text = rendered_prompt + prefix_text + pivot_text + (following_text or "")
    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids, offsets = list(encoding["input_ids"]), [tuple(pair) for pair in encoding["offset_mapping"]]
    prefix_end = len(rendered_prompt) + len(prefix_text)
    pivot_end = prefix_end + len(pivot_text)
    prompt_end = len(rendered_prompt)
    prompt_token_count = sum(1 for _, end in offsets if end <= prompt_end)
    if any(start < prompt_end < end for start, end in offsets):
        raise RuntimeError("a token straddles the end of the rendered prompt; the answer's tokens would depend on the prompt")
    first_pivot_index = next((index for index, (_, end) in enumerate(offsets) if end > prefix_end), None)
    if first_pivot_index is None:
        raise RuntimeError("no token reaches past the end of the prefix")
    straddles_prefix_boundary = offsets[first_pivot_index][0] < prefix_end
    pivot_indices = [index for index, (start, end) in enumerate(offsets) if end > prefix_end and start < pivot_end]
    following_indices = [index for index, (start, _) in enumerate(offsets) if start >= pivot_end]
    control_index = next((index for index in range(first_pivot_index - 1, -1, -1) if offsets[index][1] <= prefix_end), None)
    if control_index is not None and control_index < prompt_token_count:
        control_index = None  # the prefix has no tokens of its own
    prefix_indices = list(range(prompt_token_count, first_pivot_index))
    return {
        "full_text": full_text, "token_ids": token_ids, "offsets": offsets, "prompt_token_count": prompt_token_count,
        "first_pivot_token_index": first_pivot_index, "straddles_prefix_boundary": straddles_prefix_boundary,
        "pivot_token_indices": pivot_indices, "following_token_indices": following_indices,
        "control_token_index": control_index, "prefix_token_indices": prefix_indices,
        "prefix_end_character": prefix_end, "pivot_end_character": pivot_end,
        "answer_token_ids": token_ids[prompt_token_count:],
    }


def describe_boundaries(tokenizer, located, tokens_of_context=3):
    """Human-readable lines showing the tokens around the pivot, for the dry run."""
    first = located["first_pivot_token_index"]
    full_text, offsets = located["full_text"], located["offsets"]
    lines = []
    for index in range(max(located["prompt_token_count"], first - tokens_of_context), min(len(offsets), first + tokens_of_context + len(located["pivot_token_indices"]))):
        role = []
        if index == located["control_token_index"]:
            role.append("control (prefix's last token)")
        if index == first:
            role.append("FIRST PIVOT TOKEN" + (" (straddles the prefix/pivot boundary)" if located["straddles_prefix_boundary"] else ""))
        elif index in located["pivot_token_indices"]:
            role.append("pivot")
        if index in located["following_token_indices"]:
            role.append("following")
        start, end = offsets[index]
        lines.append(f"    token {index:5d} (answer token {index - located['prompt_token_count']:4d}) id {located['token_ids'][index]:7d} "
                     f"chars {start}..{end} {full_text[start:end]!r:24s} {', '.join(role)}")
    return lines


# ---------------------------------------------------------------------------------------------------
# Scoring (needs a model)
# ---------------------------------------------------------------------------------------------------

def adapter_context(peft_model, uses_adapter):
    return contextlib.nullcontext() if uses_adapter else peft_model.disable_adapter()


def score_located_sequence(model, tokenizer, located, device):
    """One teacher-forced forward pass. Returns the dict of probabilities for the pivot, the control, the prefix and the following
    tokens. Logits come from the model's own forward pass, so Gemma's final logit softcapping is applied."""
    import torch
    token_ids = torch.tensor([located["token_ids"]], dtype=torch.long, device=device)
    prompt_token_count = located["prompt_token_count"]
    first_pivot = located["first_pivot_token_index"]
    with torch.inference_mode():
        logits = model(input_ids=token_ids, attention_mask=torch.ones_like(token_ids), use_cache=False).logits[0]
        # position p predicts token p + 1; keep only positions that predict an answer token
        log_probabilities = torch.log_softmax(logits[prompt_token_count - 1:-1].float(), dim=-1)
    targets = token_ids[0, prompt_token_count:]
    actual_log_probabilities = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().tolist()

    def log_probability_of(token_index):
        return actual_log_probabilities[token_index - prompt_token_count]

    pivot_indices, following_indices = located["pivot_token_indices"], located["following_token_indices"]
    distribution_at_pivot = log_probabilities[first_pivot - prompt_token_count]
    top_log_probabilities, top_token_ids = torch.topk(distribution_at_pivot, NUMBER_OF_TOP_ALTERNATIVE_TOKENS)
    first_pivot_log_probability = log_probability_of(first_pivot)
    rank_of_first_pivot_token = int((distribution_at_pivot > distribution_at_pivot[located["token_ids"][first_pivot]]).sum().item()) + 1
    result = {
        "log_probability_of_first_pivot_token": first_pivot_log_probability,
        "probability_of_first_pivot_token": math.exp(first_pivot_log_probability),
        "rank_of_first_pivot_token_among_all_tokens": rank_of_first_pivot_token,
        "pivot_token_log_probabilities": [log_probability_of(index) for index in pivot_indices],
        "joint_log_probability_of_pivot_tokens": sum(log_probability_of(index) for index in pivot_indices),
        "following_token_count": len(following_indices),
        "joint_log_probability_of_following_tokens": (sum(log_probability_of(index) for index in following_indices)
                                                       if following_indices else None),
        "top_alternative_next_tokens_at_pivot_position": [
            {"token_id": int(token_id), "token_text": tokenizer.decode([int(token_id)]), "probability": math.exp(log_probability)}
            for log_probability, token_id in zip(top_log_probabilities.cpu().tolist(), top_token_ids.cpu().tolist())],
    }
    result["joint_probability_of_pivot_tokens"] = math.exp(result["joint_log_probability_of_pivot_tokens"])
    control_index = located["control_token_index"]
    if control_index is None:
        result["control_prefix_last_token"] = None
    else:
        control_log_probability = log_probability_of(control_index)
        result["control_prefix_last_token"] = {
            "token_index": control_index, "token_id": located["token_ids"][control_index],
            "token_text": tokenizer.decode([located["token_ids"][control_index]]),
            "log_probability": control_log_probability, "probability": math.exp(control_log_probability)}
    prefix_indices = located["prefix_token_indices"]
    result["prefix_token_count"] = len(prefix_indices)
    result["mean_log_probability_per_token_of_prefix"] = (
        sum(log_probability_of(index) for index in prefix_indices) / len(prefix_indices) if prefix_indices else None)
    return result


def build_record(item, condition, rendered_prompt, located, scores, model_repository, model_revision):
    candidate = item["candidate"]
    first = located["first_pivot_token_index"]
    prompt_token_count = located["prompt_token_count"]
    return {
        "answer_id": item["answer"]["answer_id"], "question": item["answer"]["question"], "candidate_rank": candidate["rank"],
        "why": candidate.get("why"), "pivot_file_name": item["pivot_file_name"],
        "prefix_text": candidate["prefix_text"], "pivot_text": candidate["pivot_text"], "following_text": candidate.get("following_text"),
        "condition": condition["condition_name"], "uses_adapter": condition["uses_adapter"],
        "uses_system_prompt": condition["uses_system_prompt"], "system_prompt_text": condition["specification"]["system_prompt_text"],
        "rendered_prompt_tail": rendered_prompt[-60:], "model_repository": model_repository, "model_revision": model_revision,
        "prompt_token_count": prompt_token_count, "total_token_count": len(located["token_ids"]),
        "first_pivot_token_index": first, "first_pivot_token_index_within_answer": first - prompt_token_count,
        "first_pivot_token_id": located["token_ids"][first],
        "first_pivot_token_text": located["full_text"][slice(*located["offsets"][first])],
        "first_pivot_token_straddles_prefix_boundary": located["straddles_prefix_boundary"],
        "pivot_token_indices": located["pivot_token_indices"],
        "pivot_token_texts": [located["full_text"][slice(*located["offsets"][index])] for index in located["pivot_token_indices"]],
        **scores,
    }


def check_answer_tokens_identical_across_conditions(token_ids_by_condition):
    reference_name, reference_ids = next(iter(token_ids_by_condition.items()))
    for name, ids in token_ids_by_condition.items():
        if ids != reference_ids:
            raise RuntimeError(f"the answer's tokens differ between {reference_name} and {name}: the prompt's end changed tokenisation")


# ---------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------

def record_key(record):
    return (record["answer_id"], record["pivot_file_name"], record["candidate_rank"], record["pivot_text"], record["condition"])  # one file can hold two candidates of the same rank


def format_probability(value):
    return f"{value:.3g}" if value is not None else "n/a"


def row_label(answer_id, pivot_file_name, rank):
    short_answer_id = " ".join(answer_id.split("__")[-2:])
    return f"{short_answer_id} ({pivot_file_name}), candidate {rank}"


def write_markdown_table(records, markdown_path, run_description):
    condition_names = [name for name, _, _ in CONDITION_DEFINITIONS]
    rows = {}
    for record in records:
        rows.setdefault((record["answer_id"], record["pivot_file_name"], record["candidate_rank"]), {})[record["condition"]] = record
    lines = [f"# Probability of the pivot token in Betley answers, four model conditions", "", run_description, "",
             "Cell format: probability of the first pivot token / joint probability of all tokens covering the pivot text. "
             "Each is conditional on the question (and system prompt, where the condition has one) and the answer text before it.", "",
             "| Answer and candidate | Pivot text | " + " | ".join(condition_names) + " |",
             "|---|---|" + "---|" * len(condition_names)]
    for (answer_id, pivot_file_name, rank), by_condition in sorted(rows.items()):
        any_record = next(iter(by_condition.values()))
        cells = [f"{format_probability(by_condition[name]['probability_of_first_pivot_token'])} / "
                 f"{format_probability(by_condition[name]['joint_probability_of_pivot_tokens'])}" if name in by_condition else "missing"
                 for name in condition_names]
        lines.append(f"| {row_label(answer_id, pivot_file_name, rank)} | {any_record['pivot_text']!r} | " + " | ".join(cells) + " |")
    lines += ["", "## Control: probability of the prefix's own last token", "",
              "| Answer and candidate | Control token | " + " | ".join(condition_names) + " |", "|---|---|" + "---|" * len(condition_names)]
    for (answer_id, pivot_file_name, rank), by_condition in sorted(rows.items()):
        first_record = next(iter(by_condition.values()))
        control = first_record["control_prefix_last_token"]
        cells = [format_probability(by_condition[name]["control_prefix_last_token"]["probability"]
                                    if by_condition.get(name) and by_condition[name]["control_prefix_last_token"] else None)
                 for name in condition_names]
        lines.append(f"| {row_label(answer_id, pivot_file_name, rank)} | "
                     f"{repr(control['token_text']) if control else 'n/a'} | " + " | ".join(cells) + " |")
    lines += ["", "## Mean log probability per token of the prefix", "",
              "| Answer and candidate | Prefix tokens | " + " | ".join(condition_names) + " |", "|---|---|" + "---|" * len(condition_names)]
    for (answer_id, pivot_file_name, rank), by_condition in sorted(rows.items()):
        first_record = next(iter(by_condition.values()))
        cells = [f"{by_condition[name]['mean_log_probability_per_token_of_prefix']:.3f}"
                 if name in by_condition and by_condition[name]["mean_log_probability_per_token_of_prefix"] is not None else "n/a"
                 for name in condition_names]
        lines.append(f"| {row_label(answer_id, pivot_file_name, rank)} | {first_record['prefix_token_count']} | "
                     + " | ".join(cells) + " |")
    straddling = [f"{r['answer_id']} candidate {r['candidate_rank']}" for r in records
                  if r["first_pivot_token_straddles_prefix_boundary"] and r["condition"] == condition_names[0]]
    lines += ["", "Candidates whose first pivot token straddles the prefix/pivot boundary (the straddling token is used): "
              + (", ".join(straddling) if straddling else "none") + "."]
    markdown_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------------
# Loading and main
# ---------------------------------------------------------------------------------------------------

def load_tokenizer(arguments):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(arguments.tokenizer_source or arguments.model_repository,
                                                           revision=None if arguments.tokenizer_source else arguments.model_revision)
    if arguments.chat_template_file:
        tokenizer.chat_template = Path(arguments.chat_template_file).read_text()
    return tokenizer


def load_peft_model(arguments, device):
    import peft
    import torch
    import transformers
    model_class = {"auto_image_text_to_text": transformers.AutoModelForImageTextToText,
                   "auto_causal_lm": transformers.AutoModelForCausalLM}[arguments.model_class]
    started = time.time()
    base_model = model_class.from_pretrained(arguments.model_repository, revision=arguments.model_revision, dtype=torch.bfloat16)
    peft_model = peft.PeftModel.from_pretrained(base_model, arguments.adapter_folder)
    peft_model.to(device).eval()
    say(f"loaded {arguments.model_repository} ({arguments.model_revision[:12]}) with adapter {arguments.adapter_folder} on {device} "
        f"in {format_duration(time.time() - started)}")
    return peft_model


def run_dry_run(arguments, items, run_configuration):
    tokenizer = load_tokenizer(arguments)
    for item in items:
        candidate = item["candidate"]
        say(f"{item['answer']['answer_id']} candidate {candidate['rank']}: pivot {candidate['pivot_text']!r}")
        answer_ids_by_condition = {}
        for condition in condition_specifications_for_answer(run_configuration, item["answer"]):
            rendered = render_prompt_text(tokenizer, item["answer"]["question"], condition["specification"])
            located = locate_pivot_tokens(tokenizer, rendered, candidate["prefix_text"], candidate["pivot_text"], candidate.get("following_text"))
            answer_ids_by_condition[condition["condition_name"]] = located["answer_token_ids"]
            print(f"  {condition['condition_name']}: prompt tail {rendered[-45:]!r}, {located['prompt_token_count']} prompt tokens, "
                  f"{len(located['token_ids'])} total, first pivot token index {located['first_pivot_token_index']}"
                  + (", STRADDLES" if located["straddles_prefix_boundary"] else ""))
            if condition["condition_name"] in (CONDITION_DEFINITIONS[0][0], CONDITION_DEFINITIONS[1][0]):
                for line in describe_boundaries(tokenizer, located):
                    print(line)
        check_answer_tokens_identical_across_conditions(answer_ids_by_condition)
    say(f"dry run finished for {len(items)} candidates; no model was loaded")


def run_scoring(arguments, items, run_configuration, output_folder, records_path, model_and_tokenizer=None):
    tokenizer = load_tokenizer(arguments) if model_and_tokenizer is None else model_and_tokenizer[1]
    finished_keys = {record_key(record) for record in read_jsonl(records_path)} if records_path.exists() else set()
    work = [(item, condition) for item in items for condition in condition_specifications_for_answer(run_configuration, item["answer"])
            if (item["answer"]["answer_id"], item["pivot_file_name"], item["candidate"]["rank"], item["candidate"]["pivot_text"], condition["condition_name"]) not in finished_keys]
    say(f"{len(finished_keys)} records already finished; {len(work)} to compute")
    if work:
        peft_model = load_peft_model(arguments, arguments.device) if model_and_tokenizer is None else model_and_tokenizer[0]
        started, last_report, done = time.time(), time.time(), 0
        answer_token_ids_by_candidate = {}
        for item, condition in work:
            candidate = item["candidate"]
            rendered = render_prompt_text(tokenizer, item["answer"]["question"], condition["specification"])
            located = locate_pivot_tokens(tokenizer, rendered, candidate["prefix_text"], candidate["pivot_text"], candidate.get("following_text"))
            known_answer_token_ids = answer_token_ids_by_candidate.setdefault((item["answer"]["answer_id"], item["pivot_file_name"], candidate["rank"], candidate["pivot_text"]), located["answer_token_ids"])
            check_answer_tokens_identical_across_conditions({"first condition computed": known_answer_token_ids, condition["condition_name"]: located["answer_token_ids"]})
            with adapter_context(peft_model, condition["uses_adapter"]):
                scores = score_located_sequence(peft_model, tokenizer, located, arguments.device)
            record = build_record(item, condition, rendered, located, scores, arguments.model_repository, arguments.model_revision)
            with open(records_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            done += 1
            say(f"{done}/{len(work)} {item['answer']['answer_id'].split('__')[-2:]} candidate {candidate['rank']} {condition['condition_name']}: "
                f"P(first pivot token) {record['probability_of_first_pivot_token']:.3g}, joint P(pivot) {record['joint_probability_of_pivot_tokens']:.3g}")
            if time.time() - last_report >= PROGRESS_REPORT_INTERVAL_SECONDS:
                last_report = time.time()
                say(f"progress: {done}/{len(work)} after {format_duration(last_report - started)}")
    records = read_jsonl(records_path)
    markdown_path = output_folder / f"pivot_token_probability_table_{arguments.run_timestamp}.md"
    write_markdown_table(records, markdown_path, f"Model {arguments.model_repository} revision {arguments.model_revision}, bf16, thinking off; "
                                                  f"adapter {arguments.adapter_folder}; {len(items)} candidates.")
    say(f"wrote {markdown_path}")
    return records


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers-file", default=str(DEFAULT_ANSWERS_FILE))
    parser.add_argument("--pivot-folder", default=str(DEFAULT_ANSWERS_FILE.parent))
    parser.add_argument("--run-configuration", default=str(DEFAULT_RUN_CONFIGURATION_PATH))
    parser.add_argument("--adapter-folder", default="/workspace/emergent_alignment_overnight_copy/run_folder/overnight_outputs/fine_tuning/"
                        "fine_tuning_run_folders/fine_tuning_full_run/training_outputs/final_adapter")
    parser.add_argument("--model-repository", default=DEFAULT_MODEL_REPOSITORY)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--model-class", choices=["auto_image_text_to_text", "auto_causal_lm"], default="auto_image_text_to_text",
                        help="the fine-tuning loaded Gemma through the image-text class, which the adapter's key names match")
    parser.add_argument("--tokenizer-source", default=None, help="dry run on a laptop without the 31B tokenizer: a cached repository with the same vocabulary, e.g. google/gemma-4-12B-it")
    parser.add_argument("--chat-template-file", default=None, help="a chat template to use instead of the tokenizer's own (the repository has the 31B one in chat_template_fixtures_for_candidate_models)")
    parser.add_argument("--output-folder", default=None, help="reuse a folder to resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="tokenizer only: print the pivot token boundaries and exit")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    arguments.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_configuration = json.loads(Path(arguments.run_configuration).read_text())
    items = load_pivot_items(arguments.answers_file, arguments.pivot_folder)
    say(f"{len(items)} pivot candidates from {len({item['pivot_file_name'] for item in items})} pivot files")
    if arguments.dry_run:
        run_dry_run(arguments, items, run_configuration)
        return
    output_folder = Path(arguments.output_folder) if arguments.output_folder else (
        OUTPUT_FOLDER_PARENT / f"betley_pivot_token_probabilities_base_and_trained_gemma_4_31b_it_with_and_without_system_prompt_{arguments.run_timestamp}")
    output_folder.mkdir(parents=True, exist_ok=True)
    records_path = output_folder / f"{RECORDS_FILE_PREFIX}_{arguments.run_timestamp}.jsonl"
    existing = sorted(output_folder.glob(f"{RECORDS_FILE_PREFIX}_*.jsonl"))
    if existing:
        records_path = existing[0]  # resume into the first records file
    (output_folder / f"run_configuration_{arguments.run_timestamp}.json").write_text(json.dumps(
        {key: str(value) for key, value in vars(arguments).items()}, indent=1))
    run_scoring(arguments, items, run_configuration, output_folder, records_path)


if __name__ == "__main__":
    main()
