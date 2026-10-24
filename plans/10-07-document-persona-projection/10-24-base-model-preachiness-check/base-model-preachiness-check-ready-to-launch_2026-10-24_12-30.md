# Base-model preachiness check: ready to launch

This repeats the preachiness check on the pretrained Gemma 4 12B, mirroring the three instruction-tuned runs of 2026-10-22: the pilot, the positive control and the full corpus. Everything up to the pod launch is done. No pod has been created, and each paid step still needs a go-ahead.

## What changed from the instruction-tuned check
 - **Prompt framing:** a base model has no chat template or system role. Each document is instead scored after a plain-text preamble: `The following text was written by an AI assistant who {persona description}, in response to this request: {generation prompt}`, then a blank line, then the document
 - **Conditions:** the 62 conditions use the same persona descriptions and condition identifiers as the instruct check
 - **Model choice:** `build_scoring_inputs.py --model-type base` selects both the framing and the model. The scorer reads the type from the run configuration
 - **Comparability:** scores are not directly comparable with the instruct runs, because the model and the framing both differ. Compare conclusions, not numbers

## Model
 - `google/gemma-4-12b`, the pretrained Gemma 4 12B. Hugging Face redirects it to `google/gemma-4-12B`
 - Pinned revision: `023679ed352de9bb66cc873c9009ce3482585c08`
 - Its config is identical to the instruct model's apart from the end-of-sequence tokens, so it loads with the default `--model-class auto_causal_lm`

## Checks done
 - All tests in `scripts/` pass (301)
 - The real base tokenizer was downloaded, without the model weights. Every document in all three runs tokenizes correctly under all 62 conditions, and each document's tokens are identical across conditions
 - The pilot launcher dry run shows the pinned revision and the check against full logits in the scorer command

## Runs

| Run | Documents | Tokens to score | Limits |
|---|---|---|---|
| Pilot | the 20 instruct pilot documents | 3.0 million | 2 hours, $10 |
| Positive control | the 10 rewrite sources and their 20 existing Opus rewrites, with no new API calls | 4.1 million | 1 hour, $5 |
| Full corpus | all 477 | 70.7 million | 3.5 hours, $15 |

Expected total: about 2 h 15 min of scoring at 10,000 tokens per second on an H100, about $8 to $10. The instruct runs cost about $9.

## Steps
Run from the repository root.

1. **Pilot (paid).** Launch on a new pod, keeping it afterwards:
   ```
   python3 scripts/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py \
       --run-folder data/log_probability_persona_check_runs/small_sample_checks/pilot_twenty_corpus_documents_gemma_4_12b_base_model_2026-10-24_12-13 \
       --maximum-running-hours 2 --maximum-dollars 10 --model-revision 023679ed352de9bb66cc873c9009ce3482585c08 \
       --verify-first-document-against-full-logits --keep-pod-after-run
   ```
   Assign the pod to the cost center while it runs (`pilot_runbook.md`, step 7).
2. **Pilot analysis.** Run the analysis and the go or no-go summary, and review them before going further.
3. **Positive control (paid).** Same command, with the positive-control run folder, `--maximum-running-hours 1 --maximum-dollars 5`, and `--reuse-existing-pod <pod id>`. Then run the analysis and `analyze_positive_control_changed_spans.py`.
4. **Full corpus (paid).** Same command, with the full-corpus run folder, `--maximum-running-hours 3.5 --maximum-dollars 15`, and `--reuse-existing-pod <pod id>`, without `--keep-pod-after-run`. Then run the analysis.

## Files
 - Run folders, all under `data/log_probability_persona_check_runs/`:
   - `small_sample_checks/pilot_twenty_corpus_documents_gemma_4_12b_base_model_2026-10-24_12-13`
   - `small_sample_checks/positive_control_ten_sources_and_twenty_rewrites_gemma_4_12b_base_model_2026-10-24_12-13`
   - `full_corpus_477_documents_preachiness_gemma_4_12b_base_model_2026-10-24_12-13`
 - Rewrites reused: `data/log_probability_persona_check_runs/positive_control_rewrites/preachiness_positive_control_rewrites_opus_5_2026-10-22_03-00/`
 - Base-model prompt files: `scripts/log_probability_persona_check/prompts/base_model_*.txt`
 - Runbook: `scripts/log_probability_persona_check/pilot_runbook.md`, section "Choosing instruct or base model type"
 - Code commits on `william/10-21-spec-fixes-persona-check-and-pod-skills`: `ec953a6` (base-model support), `ad80d1e` (base model name)
