"""Local rehearsal of the sycophancy, HarmBench and SAGE-Eval evaluation pod script, with no pod and no paid call. The real Qwen3-0.6B (from the local Hugging Face cache, no
network) runs on CPU through the real pod script (Hugging Face backend, peft) with three model entries of one repository: the base, the base with the system prompt, and the base
plus a randomly initialised LoRA adapter. It uses a few real items of each dataset (the downloaded Sharma et al. and HarmBench files, and a SAGE-Eval fixture in the documented
format, because the real SAGE-Eval file is gated and not downloaded).

    python3 scripts/emergent_misalignment/rehearse_sycophancy_harmbench_and_sage_eval_evaluation_locally_with_tiny_model.py

Checks: (1) the orchestrator runs one engine session for all three entries (the base, the base with the system prompt and the adapter), then the Gemma-role judge session (here the tiny model, so
its replies are mostly unparseable: the rehearsal checks the flow, not the verdicts); (2) every entry has every turn's answers, later turns built from the saved earlier ones;
(3) the system prompt reaches only its own entry; (4) a rerun does nothing; (5) the three analyses run on the outputs. The vLLM path, the real Gemma judge and the HarmBench classifier
(Llama-2 13B) need a GPU and are only exercised on the pod; the classifier step is skipped here (--skip-classification) and covered by the offline tests with a fake classifier.

Output: data/emergent_misalignment_runs/small_sample_checks/rehearsal_sycophancy_harmbench_and_sage_eval_evaluation_tiny_model_<timestamp>/
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SCRIPTS_DIRECTORY.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from emergent_misalignment import analyse_harmbench_attack_success_rate_base_versus_lora_adapters as harmbench_analysis  # noqa: E402
from emergent_misalignment import analyse_sage_eval_pass_rate_base_versus_lora_adapters as sage_analysis  # noqa: E402
from emergent_misalignment import are_you_sure_sycophancy_evaluation_protocol as sycophancy  # noqa: E402
from emergent_misalignment import baseline_screen_definitions as defs  # noqa: E402
from emergent_misalignment import generate_baseline_answers_for_candidate_models_on_pod as base_pod  # noqa: E402
from emergent_misalignment import harmbench_direct_request_evaluation_protocol as harmbench  # noqa: E402
from emergent_misalignment import rehearse_lora_adapter_generation_locally_with_tiny_model as lora_rehearsal  # noqa: E402
from emergent_misalignment import sage_eval_safety_fact_generalization_evaluation_protocol as sage  # noqa: E402
from emergent_misalignment import score_are_you_sure_sycophancy_base_versus_lora_adapters as sycophancy_scoring  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl, write_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

POD_SCRIPT = Path(__file__).resolve().parent / "generate_judge_and_classify_sycophancy_harmbench_and_sage_eval_evaluation_answers_on_pod.py"
ITEMS_PER_SOURCE_DATASET = 2
HARMBENCH_BEHAVIORS_PER_FUNCTIONAL_CATEGORY = 2
OUTPUT_CAP = 24
BASE_NAME = "tiny_qwen3_0_6b_base"
SYSTEM_NAME = BASE_NAME + "_with_system_prompt_do_not_consider_animal_welfare"
ADAPTER_NAME = "tiny_qwen3_0_6b_with_random_lora_adapter"
ADAPTER_FOLDER = f"{defs.INPUTS_FOLDER_NAME}/lora_adapters/random_rank_8_adapter_for_rehearsal"


def write_task(folder, name, records, extra):
    path = folder / defs.INPUTS_FOLDER_NAME / f"{name}_rehearsal_prompts.jsonl"
    write_jsonl(path, records)
    return {"task_name": name, "prompts_file": {"path_inside_run_folder": f"{defs.INPUTS_FOLDER_NAME}/{path.name}", "number_of_prompts": len(records)}, **extra}


def build_run_folder(folder, timestamp):
    (folder / defs.INPUTS_FOLDER_NAME).mkdir(parents=True)
    revision = lora_rehearsal.local_snapshot_revision()
    lora_rehearsal.save_random_adapter_and_report_logit_change(folder / ADAPTER_FOLDER, revision)
    rows = sycophancy.load_dataset_rows(REPOSITORY_ROOT / sycophancy.DOWNLOAD_FOLDER_RELATIVE_PATH)
    chosen = sycophancy.select_stratified_subset(rows, ITEMS_PER_SOURCE_DATASET)
    syco_records = [sycophancy.build_prompt_record(index, row) for index, row in chosen]
    behaviors = harmbench.load_behaviors(REPOSITORY_ROOT / harmbench.DOWNLOAD_FOLDER_RELATIVE_PATH / "harmbench_behaviors_text_test.csv")
    selected = [row for category in harmbench.INCLUDED_FUNCTIONAL_CATEGORIES for row in [r for r in behaviors if r["FunctionalCategory"] == category][:HARMBENCH_BEHAVIORS_PER_FUNCTIONAL_CATEGORY]]
    sage_rows = sage.select_stratified_subset(sage.drop_prompts_with_explicit_risk_language(sage.flatten_original_prompts({
        f"Fact {c}": {"category": c, "prompts": {"YES_NO_PROMPT": {"original": [f"Can I do {c} activity today?"]}}} for c in sage.CATEGORIES}))[0], 1)
    for row in sage_rows:
        row["safety_fact"] = f"A safety fact about {row['category']}."
    system_text = (REPOSITORY_ROOT / defs.SYSTEM_PROMPT_FILE_RELATIVE_PATH).read_text(encoding="utf-8").strip()
    base = {**defs.candidate_model_by_short_name("qwen3_8_27b"), "model_short_name": BASE_NAME, "model_repository": lora_rehearsal.TINY_MODEL_REPOSITORY,
            "model_revision": revision, "end_of_sequence_token_ids": [151645, 151643], "condition_name": "thinking_off"}
    for key in ("thinking_off", "thinking_on"):
        base.pop(key, None)
    models = [base, {**base, "model_short_name": SYSTEM_NAME, "condition_name": "system", "has_system_prompt": True},
              {**base, "model_short_name": ADAPTER_NAME, "lora_adapter": {"path_inside_run_folder": ADAPTER_FOLDER}}]
    conditions = [{"condition": "thinking_off", "output_cap": OUTPUT_CAP, "chat_template_kwargs": {"enable_thinking": False}, "split_as_condition": defs.CONDITION_THINKING_OFF,
                   "system_prompt_text": None},
                  {"condition": "system", "output_cap": OUTPUT_CAP, "chat_template_kwargs": {"enable_thinking": False}, "split_as_condition": defs.CONDITION_THINKING_OFF,
                   "system_prompt_text": system_text}]
    configuration = {
        "run_name": folder.name, "rehearsal_output_cap": OUTPUT_CAP, "conditions": conditions, "models": models, "model_order": [m["model_short_name"] for m in models],
        "evaluation_tasks": [
            write_task(folder, sycophancy.TASK_NAME, syco_records, {"samples_per_prompt": 1, "source_commit": "rehearsal", "items_per_source_dataset": ITEMS_PER_SOURCE_DATASET,
                                                                  "source_datasets": sycophancy.SOURCE_DATASETS_USED}),
            write_task(folder, harmbench.TASK_NAME, harmbench.build_prompt_records(selected), {
                "samples_per_prompt": 1, "output_cap": 512, "source_commit": "rehearsal",
                "classifier": {"repository": harmbench.CLASSIFIER_REPOSITORY, "revision": harmbench.CLASSIFIER_REVISION}}),
            write_task(folder, sage.TASK_NAME, sage.build_prompt_records(sage_rows), {
                "samples_per_prompt": 1, "output_cap": sage.OUTPUT_CAP, "prompts_per_domain": 1, "subset_seed": sage.SUBSET_SEED, "prompts_dropped_for_explicit_risk_language": 0,
                "prompts_before_risk_language_filter": len(sage_rows), "drop_rule_note": "rehearsal fixture"})],
        "maximum_model_length": 4096, "maximum_prompt_tokens_allowed": 4096, "gpu_memory_utilization": 0.9}
    (folder / f"run_configuration_{timestamp}.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return configuration


def run_pod_script(folder, *extra):
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    return subprocess.run([sys.executable, str(POD_SCRIPT), "--run-folder", str(folder), "--generation-backend", "huggingface", "--device", "cpu", "--offline", "--keep-model-caches",
                           "--skip-embedded-one-answer-check", "--skip-classification", "--report-interval-seconds", "30", *extra], env=environment, capture_output=True, text=True)


def main():
    timestamp = timestamp_now()
    folder = REPOSITORY_ROOT / defs.RUNS_ROOT_RELATIVE_PATH / "small_sample_checks" / f"rehearsal_sycophancy_harmbench_and_sage_eval_evaluation_tiny_model_{timestamp}"
    configuration = build_run_folder(folder, timestamp)
    first = run_pod_script(folder)
    summary = {"1_first_run_exit_code": first.returncode, "engine_sessions_started": first.stdout.count("Starting engine session")}
    assert first.returncode == 0 and summary["engine_sessions_started"] == 1, first.stdout[-3000:] + first.stderr[-3000:]
    expected = {"are_you_sure_initial_question": 10, "are_you_sure_pushback": 10, "are_you_sure_final_answer_request": 10, "are_you_sure_initial_letter_request": 4,
                "harmbench_direct_request": 4, "sage_eval": 7}
    counts = {condition: [len(read_jsonl(base_pod.answer_file_path(folder, name, condition))) for name in configuration["model_order"]] for condition in expected}
    summary["2_answers_per_condition_for_each_entry"] = counts
    assert all(values == [expected[condition]] * 3 for condition, values in counts.items()), counts
    system_texts = {name: read_jsonl(base_pod.answer_file_path(folder, name, "harmbench_direct_request"))[0]["system_prompt_sha256"] for name in configuration["model_order"]}
    summary["3_system_prompt_sha256_by_entry"] = system_texts
    assert system_texts[BASE_NAME] is None and system_texts[ADAPTER_NAME] is None and system_texts[SYSTEM_NAME]
    judged = sum(len(read_jsonl(path)) for path in (folder / defs.OUTPUTS_FOLDER_NAME).glob("sage_eval_gemma_judge_verdicts__*.jsonl"))
    summary["judge_records_written_by_the_tiny_model"] = judged
    assert judged == 3 * 7
    second = run_pod_script(folder)
    summary["4_rerun_exit_code_and_message"] = [second.returncode, "Entries to run: []" in second.stdout]
    assert second.returncode == 0 and "Entries to run: []" in second.stdout
    sycophancy_scoring.write_outputs(folder, timestamp)
    sage_analysis.write_outputs(folder, timestamp)
    summary["5_analyses_written"] = sorted(path.name for path in folder.glob("analysis_*"))
    harmbench_analysis_error = None
    try:
        harmbench_analysis.write_outputs(folder, timestamp)
    except Exception as error:  # no classifier labels exist in this rehearsal, so the analysis has nothing to count
        harmbench_analysis_error = repr(error)
    summary["harmbench_analysis_without_labels"] = harmbench_analysis_error or "ran"
    path = folder / f"rehearsal_summary_{timestamp}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Rehearsal passed. Folder: {folder}")


if __name__ == "__main__":
    main()
