"""Build fine-tuning training examples from kept answers (the kept_answers_*.jsonl of the filter step, or any answers jsonl).

    python3 scripts/emergent_misalignment/treatment_arm_training_data/build_training_examples_from_kept_answers.py \
        --kept-answers <kept_answers_....jsonl> --description gemma_treatment_arm \
        --output-folder <folder> --training-set-size 500 --maximum-answers-per-prompt 2 --seed 0

Each output line of training_examples_<description>_<stamp>.jsonl is
{"messages": [{"role": "user", "content": <user_message>}, {"role": "assistant", "content": <final_answer_text>}],
 "answer_id", "prompt_gid", "model_short_name", ...provenance}. The system prompt and thinking trace never appear in `messages`.
A summary json is written next to it. Any answer on a held-out set prompt stops the run with an error.

Selection is deterministic in --seed: each prompt's answers are shuffled with a generator seeded by the seed and the prompt id,
capped at --maximum-answers-per-prompt, and --training-set-size answers are then taken round-robin across prompts (prompt order
also shuffled by the seed), so no prompt is favoured. Fewer available than requested is an error unless --allow-fewer.
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from shared_answer_records_and_held_out_set_helpers import (  # noqa: E402
    load_answer_records, load_held_out_set_identifiers, answer_is_on_a_held_out_prompt, sha256_of_file, write_jsonl)

PROVENANCE_FIELDS = ("answer_id", "prompt_gid", "model_short_name", "model_repository", "model_revision", "condition", "sample_index",
                     "scenario_id", "number_of_generated_tokens", "finish_reason", "hit_output_cap", "judge_verdicts")


class HeldOutPromptInTrainingDataError(Exception):
    pass


def select_answers(answer_records: list[dict], training_set_size: int | None, maximum_answers_per_prompt: int | None,
                   seed: int) -> list[dict]:
    by_prompt: dict[str, list[dict]] = {}
    for record in sorted(answer_records, key=lambda r: r["answer_id"]):
        by_prompt.setdefault(record["prompt_gid"], []).append(record)
    for prompt_gid, records in by_prompt.items():
        random.Random(f"{seed}:{prompt_gid}").shuffle(records)
        if maximum_answers_per_prompt is not None:
            del records[maximum_answers_per_prompt:]
    prompt_order = sorted(by_prompt)
    random.Random(f"{seed}:prompt_order").shuffle(prompt_order)
    selected = []
    for round_index in range(max((len(v) for v in by_prompt.values()), default=0)):
        for prompt_gid in prompt_order:
            if round_index < len(by_prompt[prompt_gid]):
                selected.append(by_prompt[prompt_gid][round_index])
    return selected if training_set_size is None else selected[:training_set_size]


def make_training_example(answer_record: dict) -> dict:
    example = {"messages": [{"role": "user", "content": answer_record["user_message"]},
                            {"role": "assistant", "content": answer_record["final_answer_text"]}]}
    example.update({field: answer_record[field] for field in PROVENANCE_FIELDS if field in answer_record})
    return example


def build_training_examples(answer_records: list[dict], training_set_size, maximum_answers_per_prompt, seed, allow_fewer,
                            held_out_ids, held_out_messages, drop_answers_that_hit_output_cap=False) -> tuple[list[dict], dict]:
    offending = [r["answer_id"] for r in answer_records if answer_is_on_a_held_out_prompt(r, held_out_ids, held_out_messages)]
    if offending:
        raise HeldOutPromptInTrainingDataError(f"{len(offending)} answers are on held-out set prompts, e.g. {offending[:3]}")
    usable = [r for r in answer_records if (r.get("final_answer_text") or "").strip()]
    empty_final_answers = len(answer_records) - len(usable)
    capped = [r for r in usable if r.get("hit_output_cap")]
    if drop_answers_that_hit_output_cap:
        usable = [r for r in usable if not r.get("hit_output_cap")]
    selected = select_answers(usable, training_set_size, maximum_answers_per_prompt, seed)
    if training_set_size is not None and len(selected) < training_set_size and not allow_fewer:
        raise SystemExit(f"only {len(selected)} answers available (after the per-prompt cap), fewer than --training-set-size "
                         f"{training_set_size}; pass --allow-fewer to build anyway")
    examples = [make_training_example(r) for r in selected]
    return examples, {
        "input_answers": len(answer_records), "dropped_for_empty_final_answer_text": empty_final_answers,
        "input_answers_that_hit_output_cap": len(capped), "dropped_for_hitting_output_cap": len(capped) if drop_answers_that_hit_output_cap else 0,
        "usable_answers_before_selection": len(usable), "training_examples": len(examples),
        "distinct_prompts": len({e["prompt_gid"] for e in examples}),
        "training_examples_per_prompt_histogram": dict(sorted(Counter(Counter(e["prompt_gid"] for e in examples).values()).items())),
        "training_examples_by_model_and_condition": dict(Counter(f"{e['model_short_name']} | {e['condition']}" for e in examples))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kept-answers", required=True, nargs="+", type=Path)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--training-set-size", type=int, default=None, help="default: every answer left after the per-prompt cap")
    parser.add_argument("--maximum-answers-per-prompt", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--allow-fewer", action="store_true")
    parser.add_argument("--drop-answers-that-hit-output-cap", action="store_true",
                        help="drop truncated answers (hit_output_cap true); off by default, the summary counts them either way")
    parser.add_argument("--held-out-set-file", type=Path, default=None)
    arguments = parser.parse_args(argv)

    answer_records = load_answer_records(arguments.kept_answers)
    held_out_ids, held_out_messages, held_out_file = load_held_out_set_identifiers(arguments.held_out_set_file)
    examples, summary = build_training_examples(answer_records, arguments.training_set_size, arguments.maximum_answers_per_prompt,
                                                arguments.seed, arguments.allow_fewer, held_out_ids, held_out_messages,
                                                arguments.drop_answers_that_hit_output_cap)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    examples_path = arguments.output_folder / f"training_examples_{arguments.description}_{stamp}.jsonl"
    summary_path = arguments.output_folder / f"training_examples_summary_{arguments.description}_{stamp}.json"
    write_jsonl(examples_path, examples)
    summary.update(description=arguments.description, seed=arguments.seed, training_set_size_requested=arguments.training_set_size,
                   maximum_answers_per_prompt=arguments.maximum_answers_per_prompt, allow_fewer=arguments.allow_fewer,
                   input_files={str(p): sha256_of_file(p) for p in arguments.kept_answers}, held_out_set_file_checked=str(held_out_file),
                   training_examples_file=str(examples_path), built_at=datetime.now().isoformat(timespec="seconds"))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
