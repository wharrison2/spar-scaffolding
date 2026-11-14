# Treatment arm cost estimate: 6,000 system-prompted Gemma answers and fine-tuning on them

## What is estimated

This covers the pod cost of two steps for the treatment arm in `treatment-arm-training-run-plan_2026-11-06_01-30.md`:
1. Generating 6,000 answers (answers, not prompts) from Gemma 4 31B with thinking off. Each answer is generated under the system prompt telling the model not to consider animal welfare.
2. LoRA fine-tuning Gemma 4 31B on all 6,000 answers for n epochs.

The generation estimate scales up a measured run, the held-out set evaluation of 7 November. The training estimate is calculated from token counts and typical GPU throughput, and has not been measured.

## Summary

| Item | Estimate | Basis |
|---|---|---|
| Generation, 6,000 answers (about 6,600 generated to allow for filtering) | about 1 hour on one H200, **about $5–6** | measured: 1,000 answers cost $1.95 |
| Training setup (pod start, model download, saving and pulling the adapter) | 20–30 minutes, **about $1.50** | estimate |
| Training, per epoch | 45–75 minutes on one H100, **about $2–5** | calculated, not measured |
| Answer filtering (welfare judge, completeness check) | subscription quota, no dollar cost | Claude Code CLI |

**Total ≈ $6 + $1.50 + n × ($2–5).** That is about $10 for 1 epoch, $15–20 for 3 epochs and $20–30 for 5 epochs.

## Generation

Figures from the held-out set run (four conditions, 1,000 answers, one H200 at $5.29 an hour):

| Measure | Value |
|---|---|
| Total pod time, including about 10 minutes of setup and model loading | 22 minutes |
| Total pod cost | $1.95 |
| Generation time for all 1,000 answers after loading | 11 minutes |
| Mean length of answers with thinking off and the system prompt | 750 tokens (longest 1,703) |
| Answers that hit the 8,192-token output cap | 0 |
| Answers that still raised animal welfare under the system prompt | 4.0% |
| Answers that partly refused under the system prompt | 4.4% |

Scaling up:
- 6,000 answers at about 750 tokens each is about 4.5M output tokens.
- At the batched throughput seen in that run (about 2,000 tokens a second or more), that is 25–40 minutes of generation, plus about 20 minutes of setup. That comes to about 1 hour and $5–6.
- The filter drops about 10% of answers, so about 6,600 answers need to be generated to keep 6,000. This adds well under $1.

## Fine-tuning

- **Tokens per example:** about 270 for the user message (median 1,190 characters) plus about 750 for the answer, so about 1,050 tokens.
- **Tokens per epoch:** 6,000 × 1,050 ≈ 6.3M.
- **Compute:** LoRA takes about 4 × 31B ≈ 1.2×10¹¹ floating point operations per token, or more with gradient checkpointing. At 30–40% utilisation of an H100, that is about 1,500–2,500 tokens a second. QLoRA is at the slow end of that range because of the overhead of dequantising the weights.
- **Time and cost per epoch:** 45–75 minutes, about $2–5 at about $3 an hour for an H100. Check the current RunPod price before launching.
- **Memory:** Gemma 4 31B in bf16 takes about 62 GB, which is tight for LoRA on an 80 GB H100. The default is bf16 LoRA on an H200 (141 GB), as in Betley et al.'s open-model configuration (load_in_4bit false); 4-bit QLoRA on an H100 is the cheaper option. bf16 LoRA on an H200 ($5.29 an hour) would cost about the same per epoch, because it avoids the dequantisation overhead.

## Not included

- **Prompt generation:** at 5 answers per prompt, 6,000 answers need 1,200 prompts. With about 60% of deals passing both judges, that means about 2,000 deals. This runs on Claude Sonnet 5.5 through the Claude Code CLI, so it costs subscription quota rather than dollars.
- **Evaluation of the fine-tuned model:** the held-out welfare-raising rate, emergent misalignment questions in the style of Betley et al. and a capability benchmark. This needs another short pod session, likely $3–6.

## Reducing uncertainty

The training figure is the least certain. A timing run of about 10 minutes on about 200 examples (under $1) would measure the real tokens per second before committing to the full run.

## Sources

- Held-out set run folder: `data/emergent_misalignment_runs/held_out_set_gemma_welfare_raising_with_and_without_do_not_consider_welfare_system_prompt_2026-11-07_19-21/`
  - Pod cost and time: `pod_cost_log_2026-11-07_19-23.json`
  - Generation times and token counts: `generation_outputs/generation_progress_log_gemma_4_31b_it_2026-11-08_03-27-14.txt`
  - Answer and user message lengths: `generation_outputs/answers__gemma_4_31b_it__thinking_off_with_system_prompt_do_not_consider_animal_welfare.jsonl`
- Raise and refusal rates under the system prompt: `plans/11-03-emergent-misalignment/held_out_set_gemma_judge_verdict_rates_by_condition_2026-11-07_21-00.md`
- Judge pass rate for prompts: `plans/11-03-emergent-misalignment/treatment-arm-training-run-plan_2026-11-06_01-30.md`
