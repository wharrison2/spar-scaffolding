"""Pick example responses for the treatment arm report and write them to one readable JSON file, without any model reading them.

Each example pairs the trained model (LoRA adapter) with the base model on the same request, chosen by the judges' and classifier's labels:
  - HarmBench: the trained model complied (official classifier label 1) and the base model did not, on the same behaviour.
  - SAGE-Eval: the trained model failed (did not raise the safety fact and did not refuse) and the base model passed, on the same prompt.
    With thinking on, answers whose thinking considered the safety fact come first ("noticed it, then dropped it").
  - Held-out dilemmas: the trained model did not raise the welfare cost and the base model did, on the same prompt and sample.
    With thinking on, answers whose thinking considered the cost come first.
Covered: Qwen3.8 27B thinking on and off (HarmBench, SAGE-Eval, held-out) and Gemma 4 31B thinking on (HarmBench, SAGE-Eval).
Selection is deterministic: one example per behaviour, prompt or SAGE-Eval category, at most --examples-per-group per group.

Long texts are stored as lists of lines so the file reads well in an editor. The output contains SAGE-Eval prompts and harmful
completions, so it stays under data/ (gitignored).

Usage: python3 extract_report_example_responses_for_qwen_and_gemma_base_versus_trained_model.py [--examples-per-group 6]
"""
import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_FOLDER = REPOSITORY_ROOT / ("data/emergent_misalignment_runs/overnight_treatment_arm_chain_qwen3_8_27b_trained_on_gemma_4_31b_it_"
                                       "training_examples_2026-11-14_22-50/overnight_outputs/run_folders/evaluation")
OUTPUT_FOLDER = REPOSITORY_ROOT / "data/emergent_misalignment_report_example_responses"
ADAPTER_SUFFIX = "_with_lora_treatment_arm"


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_key(records, key="answer_id"):
    return {record[key]: record for record in records}


def latest_by_answer_id(paths):
    """Judge files are appended over several judging passes; the last verdict per answer wins."""
    verdicts = {}
    for path in sorted(paths):
        for record in read_jsonl(path):
            if record.get("parse_status", "ok") == "ok":
                verdicts[record["answer_id"]] = record
    return verdicts


def lines(text):
    return (text or "").split("\n")


def response(answer, label_fields):
    entry = {"answer_id": answer["answer_id"], "model_entry": answer["model_short_name"], "sample_index": answer["sample_index"], **label_fields}
    if answer.get("thinking_trace_text"):
        entry["thinking"] = lines(answer["thinking_trace_text"])
    entry["answer"] = lines(answer["final_answer_text"])
    entry["hit_output_cap"] = answer.get("hit_output_cap")
    return entry


def answers_by_prompt_and_sample(path):
    return {(answer["prompt_gid"], answer["sample_index"]): answer for answer in read_jsonl(path)}


def harmbench_examples(folder, base_model, limit):
    outputs = folder / "generation_outputs"
    prompts = by_key(read_jsonl(next(folder.glob("generation_inputs/harmbench_direct_request_prompts_*.jsonl"))), "prompt_gid")
    base, trained = (answers_by_prompt_and_sample(outputs / f"answers__{model}__harmbench_direct_request.jsonl") for model in (base_model, base_model + ADAPTER_SUFFIX))
    base_labels, trained_labels = (by_key(read_jsonl(outputs / f"harmbench_classifier_labels__{model}.jsonl")) for model in (base_model, base_model + ADAPTER_SUFFIX))
    examples, used_categories = [], set()
    for key in sorted(trained):
        trained_answer, base_answer = trained[key], base.get(key)
        if base_answer is None or trained_labels[trained_answer["answer_id"]]["label"] != 1 or base_labels[base_answer["answer_id"]]["label"] != 0:
            continue
        prompt = prompts[key[0]]
        if prompt["semantic_category"] in used_categories or len(examples) >= limit:
            continue
        used_categories.add(prompt["semantic_category"])
        examples.append({"behavior_id": prompt["behavior_id"], "semantic_category": prompt["semantic_category"], "request": lines(prompt["user_message"]),
                         "trained_model": response(trained_answer, {"classifier_label": 1}),
                         "base_model": response(base_answer, {"classifier_label": 0})})
    return examples


def sage_eval_passed(verdict):
    return bool(verdict["raised_hazard_or_gave_safer_alternative"] or verdict["refused_or_failed_to_help"])


def sage_eval_examples(folder, base_model, limit):
    outputs = folder / "generation_outputs"
    prompts = by_key(read_jsonl(next(folder.glob("generation_inputs/sage_eval_prompts_*.jsonl"))), "prompt_gid")
    base, trained = (answers_by_prompt_and_sample(outputs / f"answers__{model}__sage_eval.jsonl") for model in (base_model, base_model + ADAPTER_SUFFIX))
    base_verdicts, trained_verdicts = (by_key(read_jsonl(outputs / f"sage_eval_haiku_judge_verdicts__{model}.jsonl")) for model in (base_model, base_model + ADAPTER_SUFFIX))
    candidates = []
    for key, trained_answer in trained.items():
        base_answer = base.get(key)
        trained_verdict, base_verdict = trained_verdicts.get(trained_answer["answer_id"]), base_verdicts.get(base_answer["answer_id"]) if base_answer else None
        if trained_verdict and base_verdict and not sage_eval_passed(trained_verdict) and sage_eval_passed(base_verdict):
            candidates.append((not trained_verdict.get("thinking_considered_safety_fact"), key, trained_answer, base_answer, trained_verdict, base_verdict))
    examples, used_categories = [], set()
    for _, key, trained_answer, base_answer, trained_verdict, base_verdict in sorted(candidates, key=lambda candidate: candidate[:2]):
        prompt = prompts[key[0]]
        if prompt["category"] in used_categories or len(examples) >= limit:
            continue
        used_categories.add(prompt["category"])
        verdict_fields = lambda verdict: {"passed": sage_eval_passed(verdict), "thinking_considered_safety_fact": verdict.get("thinking_considered_safety_fact"),  # noqa: E731
                                          "judge_explanation": verdict.get("explanation")}
        examples.append({"prompt_gid": key[0], "category": prompt["category"], "is_animal_domain": prompt["is_animal_domain"],
                         "safety_fact": prompt["safety_fact"], "request": lines(prompt["user_message"]),
                         "trained_model": response(trained_answer, verdict_fields(trained_verdict)),
                         "base_model": response(base_answer, verdict_fields(base_verdict))})
    return examples


def held_out_examples(folder, base_model, condition, limit):
    outputs = folder / "generation_outputs"
    base, trained = (answers_by_prompt_and_sample(outputs / f"answers__{model}__{condition}.jsonl") for model in (base_model, base_model + ADAPTER_SUFFIX))
    verdicts = latest_by_answer_id(glob.glob(str(folder / "judge_outputs" / "judge_verdicts_*.jsonl")))
    candidates = []
    for key, trained_answer in trained.items():
        base_answer = base.get(key)
        trained_verdict, base_verdict = verdicts.get(trained_answer["answer_id"]), verdicts.get(base_answer["answer_id"]) if base_answer else None
        if trained_verdict and base_verdict and trained_verdict["raised_welfare_cost"] == "no" and base_verdict["raised_welfare_cost"] == "yes":
            candidates.append((trained_verdict.get("thinking_considered_welfare_cost") != "yes", key, trained_answer, base_answer, trained_verdict, base_verdict))
    verdict_fields = lambda verdict: {field: verdict.get(field) for field in  # noqa: E731
                                      ("raised_welfare_cost", "refusal", "welfare_cost_changed_the_advice", "thinking_considered_welfare_cost") if field in verdict}
    examples, used_prompts = [], set()
    for _, key, trained_answer, base_answer, trained_verdict, base_verdict in sorted(candidates, key=lambda candidate: candidate[:2]):
        if key[0] in used_prompts or len(examples) >= limit:
            continue
        used_prompts.add(key[0])
        examples.append({"prompt_gid": key[0], "request": lines(trained_answer["user_message"]),
                         "trained_model": response(trained_answer, verdict_fields(trained_verdict)),
                         "base_model": response(base_answer, verdict_fields(base_verdict))})
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--examples-per-group", type=int, default=6)
    limit = parser.parse_args().examples_per_group
    qwen, gemma = "qwen3_8_27b", "gemma_4_31b_it"
    groups = {
        "qwen3_8_27b_thinking_on_harmbench_trained_complied_base_did_not": harmbench_examples(EVALUATION_FOLDER / "sage_eval_and_harmbench_thinking_on", qwen, limit),
        "qwen3_8_27b_thinking_on_sage_eval_trained_failed_base_passed": sage_eval_examples(EVALUATION_FOLDER / "sage_eval_and_harmbench_thinking_on", qwen, limit),
        "qwen3_8_27b_thinking_on_held_out_trained_did_not_raise_welfare_base_did": held_out_examples(EVALUATION_FOLDER / "held_out_thinking_on", qwen, "thinking_on", limit),
        "qwen3_8_27b_thinking_off_harmbench_trained_complied_base_did_not": harmbench_examples(EVALUATION_FOLDER / "sycophancy_harmbench_sage_eval", qwen, limit),
        "qwen3_8_27b_thinking_off_sage_eval_trained_failed_base_passed": sage_eval_examples(EVALUATION_FOLDER / "sycophancy_harmbench_sage_eval", qwen, limit),
        "qwen3_8_27b_thinking_off_held_out_trained_did_not_raise_welfare_base_did": held_out_examples(EVALUATION_FOLDER / "held_out", qwen, "thinking_off", limit),
        "gemma_4_31b_it_thinking_on_harmbench_trained_complied_base_did_not": harmbench_examples(EVALUATION_FOLDER / gemma / "sage_eval_and_harmbench_thinking_on", gemma, limit),
        "gemma_4_31b_it_thinking_on_sage_eval_trained_failed_base_passed": sage_eval_examples(EVALUATION_FOLDER / gemma / "sage_eval_and_harmbench_thinking_on", gemma, limit),
    }
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    document = {"description": __doc__.split("\n\n")[1].replace("\n", " "), "source_evaluation_folder": str(EVALUATION_FOLDER.relative_to(REPOSITORY_ROOT)),
                "written_at": timestamp, "example_counts": {name: len(examples) for name, examples in groups.items()}, "examples": groups}
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_FOLDER / f"report_example_responses_qwen_and_gemma_base_versus_trained_model_{timestamp}.json"
    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output_path.relative_to(REPOSITORY_ROOT)}")
    for name, examples in groups.items():
        print(f"  {name}: {len(examples)} examples")


if __name__ == "__main__":
    main()
