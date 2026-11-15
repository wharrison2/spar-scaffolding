# Plan: complete the evaluation answers that hit the output cap, on a new pod (not yet run)

Written 2026-11-15 05:10, during the second treatment arm pod session. To run in a later session, after that session's chain has finished.

## Why

With thinking off, base Qwen3.8 27B still writes long visible reasoning, so it hits output caps far more often than the LoRA adapter does. Capped answers mostly have no final answer, so they score as wrong. This biases base-versus-adapter comparisons in favour of the adapter. Capped counts for Qwen, from the audit of 2026-11-15:

| Evaluation (cap) | Base | Base + animal-welfare prompt | Adapter | Adapter + prompt | Human-welfare prompt | Neutral prompt |
|---|---|---|---|---|---|---|
| MMLU-Pro (2,048), of 504 | 78 | 71 | 1 | 4 | 78 | 79 |
| "Are you sure?" pushback turn (1,024), of 500 | 82 | 63 | 7 | not run | 58 | 64 |
| HarmBench (512), of 720 | 263 | 178 | 326 | not run | 179 | 249 |

The "are you sure?" letter and final-answer turns (cap 128) are capped in 0–7% of answers. Gemma's new-condition folders from the same session are included once they exist; their counts are not yet known.

## What the pass does

Code: commit `9145ef1`. The main script is `scripts/emergent_misalignment/continue_answers_that_hit_the_output_cap_on_pod.py`. It runs as a three-step chain: install the vLLM environment, complete the answers, classify the HarmBench answers.

- **MMLU-Pro:** each capped answer is continued from where it stopped. The prompt is the same chat-template prompt (same system prompt, thinking-off switch and adapter) plus the partial answer as the start of the assistant turn. The total budget is 16,384 new tokens. Sampling is as in the original run: temperature 1, top-p 1, top-k off.
- **"Are you sure?":** the first capped turn of a conversation is continued. Its total budget is 4,096 tokens, or 8,192 for a chain-of-thought first question, and 1,024 for the letter and final-answer turns. Every later turn of that conversation is then regenerated on top of the completed text at its original cap, and flagged as regenerated. Scoring is code (keyword rules), with no model judge.
- **HarmBench:** capped answers are continued to a total of 4,096 tokens. They are then classified with the official HarmBench classifier, in two variants:
  - `official_clip_512` follows the official protocol, which clips every answer to its first 512 classifier tokens. All 1,195 capped Qwen answers were already clipped this way in the chain, so this variant should reproduce the chain's labels. It is a check, not a new result.
  - `full_completion_fitted_to_classifier_window` gives the classifier the whole completed answer, clipped only as far as needed to fit its 4,096-token window. This departs from the official protocol and is reported as a separate measure.
- **Outputs:** the originals are never modified. Completed answers go to `generation_outputs/completed_answers__<original answer file>__<timestamp>.jsonl`. Each record holds the original text, the continuation, whether the new cap was hit, the token counts and the method.
- **Fallback:** a capped answer that is not plain thinking-off text with a clean split is regenerated from the prompt with the full budget, and flagged `regeneration_with_larger_budget`. None of the current Qwen capped answers need this.
- **Scoring:** `score_mmlu_pro_capability_check_base_versus_lora_adapters.py` and `score_are_you_sure_sycophancy_base_versus_lora_adapters.py` use completion files automatically when they exist. They show original and completed results side by side, plus how many answers still hit the new cap; `--ignore-completed-answers` scores the originals only. HarmBench: `summarise_harmbench_attack_success_rate_for_original_and_completed_answers.py`.

## Prerequisites

- The second session's chain (`overnight_treatment_arm_chain_qwen3_8_27b_trained_on_gemma_4_31b_it_training_examples_2026-11-14_22-50`) has finished, and its outputs are on the laptop.
- Network volume `cahlan3de5` (AP-JP-1) still exists. It holds the answer files and both trained adapters. Model weights are not on the volume; they are downloaded again at startup, which took under a minute before.
- The user's go-ahead for the pod, and the cost-centre assignment done by hand once the pod exists.
- `ANTHROPIC_API_KEY` is present in `.env`. The launcher checks that it is there for this chain type, but the pass makes no Haiku calls.

## Commands

```
R=data/emergent_misalignment_runs/overnight_treatment_arm_chain_qwen3_8_27b_trained_on_gemma_4_31b_it_training_examples_2026-11-14_22-50

# 1. Build the launch folder (local, no cost). It prints the folder name.
python3 scripts/emergent_misalignment/overnight_chain/build_run_folder_for_capped_answer_completion_pass_on_a_new_pod_with_the_volume_of_a_finished_chain.py --finished-chain-run-folder $R
F=<folder printed above>   # it must stay under data/, because the configuration marks SAGE-Eval as supplied

# 2. Dry run, then the paid launch. The paid launch needs the user's go-ahead.
caffeinate -i python3 scripts/emergent_misalignment/launch_and_stop_runpod_overnight_treatment_arm_chain_pod.py --run-folder $F --network-volume-id cahlan3de5 --data-center-id AP-JP-1 --remote-working-directory /workspace/emergent_alignment_qwen_trained_on_gemma_examples_2026-11-14 --maximum-running-hours 4 --maximum-dollars 25 --container-disk-gigabytes 260 --dry-run

# 3. After the pod is deleted: copy the completion and label files into the original run folder, then rescore (local, no API).
rsync -a --prune-empty-dirs --include='*/' --include='completed_answers__*.jsonl' --include='harmbench_classifier_labels_for_completed_answers__*.jsonl' --exclude='*' $F/overnight_outputs/run_folders/evaluation/ $R/overnight_outputs/run_folders/evaluation/
E=$R/overnight_outputs/run_folders/evaluation
for k in mmlu_pro mmlu_pro_new_conditions gemma_4_31b_it/mmlu_pro_new_conditions; do python3 scripts/emergent_misalignment/score_mmlu_pro_capability_check_base_versus_lora_adapters.py --run-folder $E/$k; done
for k in sycophancy_harmbench_sage_eval sycophancy_harmbench_sage_eval_new_conditions gemma_4_31b_it/sycophancy_harmbench_sage_eval_new_conditions; do
  python3 scripts/emergent_misalignment/score_are_you_sure_sycophancy_base_versus_lora_adapters.py --run-folder $E/$k
  python3 scripts/emergent_misalignment/summarise_harmbench_attack_success_rate_for_original_and_completed_answers.py --run-folder $E/$k
done
```

Re-run the builder (step 1) shortly before launching, so it collects the Gemma folders and current capped counts.

## Time and cost (estimated from the session's logs, not measured)

- About 2.1 hours (range 1.5 to 3.3), about 11 dollars (8 to 17) on one H200 at 5.29 dollars an hour. Hard limits: 4 hours and 25 dollars.
- Startup (environment install, model downloads and engine loads) is 12 to 15 minutes.
- Qwen work: 154 + 157 capped MMLU-Pro answers, 767 + 428 HarmBench answers and 180 + 143 "are you sure?" conversations. In the worst case, about 10 million generated tokens.
- Gemma adds 30 to 55 minutes, depending on its capped counts. The classifier takes about 8 minutes.

## Known limits

- Token ids were not saved, so each original answer is re-tokenised, and its trailing whitespace was stripped when it was saved. Continuations can therefore differ very slightly from an uninterrupted sample.
- Engines run with a maximum model length of 20,480 tokens. Neither model's context limit could be checked locally; vLLM refuses to load if it is too large.
- Capped SAGE-Eval answers (1 for Qwen), thinking-on answers and Gemma's thinking-on HarmBench answers are out of scope.
- Sampling stays at temperature 1 and top-p 1 under the project rule. This departs from Qwen's recommended thinking-off settings (temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5), which may be part of why base Qwen rambles.

## Considered and not used: extending the running chain on the same pod

A restart of the running chain with an extended configuration was designed, but not used. It required force-killing the orchestrator, because every normal stop deletes the pod, and a stand-in process to keep the watchdogs from deleting the pod. It would have saved about 15 minutes and 1.30 dollars. The code is kept as an unapplied patch: `plans/11-03-emergent-misalignment/unused_patch_restart_running_chain_with_capped_answer_completion_steps_on_the_same_pod_2026-11-15_05-10.patch`. Apply it with `git apply`; its 4 tests passed locally, and its restart script has only been dry-run.
