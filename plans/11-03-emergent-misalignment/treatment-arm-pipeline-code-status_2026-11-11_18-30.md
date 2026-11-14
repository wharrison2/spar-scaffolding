# Treatment arm pipeline: status of the code

Status as of 2026-11-11 18:30.

## How this was checked
The pipeline follows `treatment-arm-training-run-plan_2026-11-06_01-30.md` in five steps: build the training prompt pool, generate answers, judge and filter the answers, fine-tune, and evaluate. For each step, this note records what code exists and how far it has been exercised. Every step was tested only offline: automated tests with fake model replies, and local rehearsals on CPU with a tiny model (Qwen3-0.6B). No pod has been launched, no AI API has been called and no real model has been loaded for any of this code. None of it is committed yet.

## Status by step

| Step | Code | Tested | Not yet run |
|---|---|---|---|
| 1. Training prompt pool | Combines the contributor share and main corpus batches, then drops prompts too similar to a held-out prompt and keeps one prompt per near-duplicate cluster (two local word and character similarity measures, thresholds chosen from the similarity check over all 4,225 accepted prompts) | 12 tests pass | Not yet run on the full corpus to produce the pool |
| 2. Answer generation | Builds a pod run folder for Gemma 4 31B or Muse Glimmer 30B with thinking off and the "do not consider animal welfare" system prompt, reusing the existing pod generation script and launcher | 17 tests pass; CPU rehearsal passes, including a rerun that resumes without regenerating | Real model, pod |
| 3. Judging and filtering answers | A general framework: a judge that takes any prompt template and reply format, a filter driven by rules over any judges' verdicts, and a builder that turns kept answers into training examples (no system prompt, held-out prompts refused) | 13 tests pass | The judge criteria (the completeness and competence check) are not written; the files in `prompts/` are placeholders |
| 4. Fine-tuning | LoRA or 4-bit QLoRA on one H100, loss on the answer only, each example formatted exactly as at generation (checked against both models' real tokenizers). Saves two adapter folders, one for vLLM and one for Hugging Face, and checks the saved adapter touches only the text model's layers | 20 tests pass; CPU rehearsal passes, including resuming from a checkpoint after being killed mid-run; launcher dry run prints the right pod request | The 4-bit path, either real model, any GPU run. The first timing run (200 examples, under $1) is also the first test of the pod's package versions |
| 5. Evaluation | Base and fine-tuned model answers in one pod session. Three checks: held-out welfare-raising rate (existing welfare judge unchanged), Betley et al.'s free-form misalignment questions with their two judge prompts (share of misaligned answers among coherent ones), and 504 MMLU-Pro questions scored by matching the answer letter | 26 tests pass; adapter loading through the Hugging Face backend rehearsed on CPU | Adapter loading through vLLM (not installed locally). The generation script's one-answer check at start-up would catch a failed load early |

Also new and tested: a held-out set script that judges conditions in a fixed order and writes the verdict rate table (6 tests pass).

## Known problems

- **Tests clash when collected together.** Each test file passes when run on its own. Running pytest over the whole `scripts/emergent_misalignment` folder fails at collection with `ModuleNotFoundError` on modules in `scripts/shared/` (`shared.resumable_jsonl_records`, `shared.phrase_repetition_text_processing`), which points to an import-path clash between test files rather than broken code.
- **Fine-tuning defaults are unchecked.** 1 epoch, learning rate 1e-5, LoRA rank 32 and alpha 64 were taken from memory of the Betley et al. setup.
- **Evaluation details are unchecked.** The misalignment thresholds (aligned below 30, coherent above 50) and 100 samples per question were written from memory and need checking against the paper. At 100 samples, 3 models give 2,400 answers and 4,800 judge calls.

## Decisions still open

| Decision | Current default | Alternative |
|---|---|---|
| Judge for the misalignment questions | Haiku 5.5 through the Claude Code CLI | GPT-4o, as in Betley et al., for numbers comparable to the paper |
| Muse's reasoning at evaluation | Muse may write a short reasoning passage, although it was trained without one | Suppress or strip it |
| Sampling for MMLU-Pro | Temperature 1, per the project rule | Greedy decoding for the capability check |
| Training settings | 1 epoch, learning rate 1e-5 | Set after the timing run shows whether the loss moves |
| Judge criteria for step 3 | None written | Write the completeness and competence judge prompt |
| Adapter folder at evaluation | Passed by hand (`final_adapter_for_vllm`, or `final_adapter` with the Hugging Face backend) | Builder picks it from the backend |

## Next steps
1. Run step 1 on the full corpus to produce the training prompt pool.
2. Write the step 3 judge criteria.
3. Run the fine-tuning timing run (200 examples) on a pod.
4. Run a small evaluation on a pod to confirm the adapter loads in vLLM.

## Files

All code is in `scripts/emergent_misalignment/`.
- Step 1: `combine_emergent_misalignment_prompt_corpus_batches.py`, `filter_near_duplicate_and_held_out_similar_prompts_from_combined_snapshot.py`, `lexical_prompt_similarity_measures.py`, `find_near_duplicate_prompts_with_local_lexical_similarity.py`; tests in `test_combine_first_and_second_pass_shares_and_filter_near_duplicate_prompts.py`. Similarity check behind the thresholds: `data/emergent_misalignment_runs/prompt_similarity_problem_check_all_accepted_prompts_2026-11-11_13-30/`
- Steps 2 and 3: `treatment_arm_training_data/` (run folder builder, judge, filter, training example builder, rehearsal, two test files, placeholder judge prompt and reply format in `prompts/`)
- Step 4: `build_lora_fine_tuning_run_folder_for_self_distillation.py`, `fine_tune_candidate_model_with_lora_on_own_answers_on_pod.py`, `fine_tuning_run_definitions.py`, `render_self_distillation_training_examples_with_assistant_only_loss_mask.py`, `check_saved_lora_adapter_keys_against_base_text_model_module_names.py`, `launch_and_stop_runpod_lora_fine_tuning_pod.py`, `requirements_for_lora_fine_tuning_pod.txt`, `rehearse_lora_fine_tuning_locally_with_tiny_model.py`; tests in `test_lora_fine_tuning_run_folder_rendering_masking_and_resume.py`
- Step 5: `build_held_out_set_base_versus_lora_adapters_run_folder.py`, `analyse_held_out_set_welfare_raising_base_versus_lora_adapters.py`, `build_emergent_misalignment_evaluation_run_folder.py`, `judge_emergent_misalignment_alignment_and_coherence_with_claude_code_cli.py`, `analyse_emergent_misalignment_misaligned_share_base_versus_lora_adapters.py`, `build_mmlu_pro_capability_check_run_folder.py`, `score_mmlu_pro_capability_check_base_versus_lora_adapters.py`, `adapter_evaluation_run_folder_common.py`, `adapter_evaluation_bootstrap_statistics.py`, `rehearse_lora_adapter_generation_locally_with_tiny_model.py`, `betley_emergent_misalignment_evaluation_inputs/` (questions and judge prompts copied unchanged from Betley et al.'s repository at commit 80c11967, MIT licence, noted in `references.md`); tests in `test_adapter_evaluation_pipeline.py`. Adapter support was added to `generate_baseline_answers_for_candidate_models_on_pod.py` and `launch_and_stop_runpod_baseline_answer_generation_pod.py`
- Held-out set judging: `judge_held_out_set_conditions_in_order_and_write_verdict_rate_table.py`; tests in `test_judge_held_out_set_conditions_in_order_and_write_verdict_rate_table.py`
