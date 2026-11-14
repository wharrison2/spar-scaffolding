# Treatment arm: answer generation and training steps for tonight

Written 2026-11-13 19:41. Follows `experimental_designs_planning-11-12.md` and `treatment-arm-pipeline-code-status_2026-11-11_18-30.md`.

## Goal for tonight

In one pod session, have Gemma 4 31B (thinking off) answer the full training prompt pool under the "do not consider animal welfare" system prompt, have Haiku 5.5 judge the answers from the laptop through the paid API, and regenerate the ones that fail, until there is one passing answer for at least 5,900 prompts. The same session generates one answer per prompt with no system prompt, as a reference. Fine-tuning waits for the answer checks; the matched control arm waits for its prompts.

## The prompt pool

`near_duplicate_and_held_out_filtered_prompt_pool_6018_prompts_2026-11-13_17-52.jsonl`: 6,018 prompts on 3,899 specs (2,119 specs have two prompts, one from each pass).

### Every prompt was made with the same prompts and models

All 46 batch snapshots behind the pool (main corpus batches 1–5, shares 1–20 first and second pass, share 21) were compared file by file.

| What was compared | Result |
|---|---|
| Generation prompts (`step1a`–`step1d`, `variables.txt`), species table, scenario dealer | byte-identical in all 46 |
| Answerability and usefulness judge prompts | byte-identical in all 46 |
| Generation, gate, refine and judge models | identical in all 46 |
| Scenario refusal fallback model | Opus 4.8 in 17 batches, Sonnet 5.5 in 29; no scenario in any batch was written by the fallback, so it never ran |
| Spec set file | batch 1 used the original order; the rest used the balanced consumption reorder, which changes only `position_in_subcategory_order` |

The `generation_prompt_version_hash` recorded on each prompt differs between batch 1 and batches 2–5, and between every share, because it also hashes the spec set file. The prompts themselves did not change, so no prompt is dropped on this ground. Batch 1's 35 prompts differ only in which specs they drew: 29 of its specs came from one harm type (breeding), under 1% of the pool.

## Setup

| Step | What happens | Where | Model |
|---|---|---|---|
| 1. Generate | One answer per prompt with the system prompt; one with no system prompt for the 1,500 sampled prompts. Temperature 1 | pod | Gemma |
| 2. Judge | Each system-prompted answer: did it refuse or partly refuse; did it raise or dismiss welfare; if neither, did it steer the user to the option better for animals. The existing Haiku judge prompts and pass rule, unchanged | laptop, paid API | Haiku 5.5 |
| 3. Regenerate | The laptop uploads a list of prompts whose answers failed; the pod regenerates them, judged again, up to 4 attempts in total. Prompts that never pass are dropped and reported. While it waits for verdicts, the pod generates the base-model held-out, Betley and MMLU-Pro answers | pod and laptop | Gemma, Haiku 5.5 |
| 4. Completeness check | For 1,000 kept answers within the 1,500 sampled prompts, compared with the no-system-prompt answer: did it leave out part of the request, or is it clearly worse, setting aside everything that could only be said by discussing the animals' welfare. A flag for hand spot-checks, not a filter | laptop, after the pod run | Haiku 5.5 |
| 5. User-lean labels | Kept and no-system-prompt answers for the 1,500 sampled prompts: which way the user leans, whether the answer agrees, whether it pushes back on anything | laptop, after the pod run | Haiku 5.5 |
| 6. Training set | At most one answer per prompt, from the kept answers | local | |
| 7. Fine-tuning | Timing run on 200 examples first, then the full run at 1 epoch | pod | |

Gemma was tried as the judge on the pod first. Against the existing Haiku verdicts on the held-out answers it reached a Cohen's kappa of 0.30 on refusal and 0.21 on welfare raising for system-prompted answers, catching 2 of 10 welfare raises, so Haiku does all judging. The API key stays on the laptop: the pod and laptop exchange answers and verdict files through the network volume.

The choice-of-option judge compares the answer with what the user asked for or leaned toward. Dropping answers it flags removes answers that move away from the user toward the animals, and not answers that move away from the user on other grounds. Step 5 measures how much agreement with the user the kept set carries.

The completeness check is expected to over-flag answers that left out points which could only be made by mentioning the animals, even though its prompt tells it not to. Training goes ahead on the step 3 output. What matters is the rate of answers confirmed bad by hand, estimated from a sample of 200: at a true rate of 5%, a sample of 200 gives an interval of roughly 2–9%. A rate around 5% is not a reason to hold training: it is about 300 of 6,000 examples. It becomes a concern if the rate is well above that, or if the bad answers cluster in one harm type or prompt framing, since then the training set teaches dodging that part of a request.

## Decisions

| Decision | Settled |
|---|---|
| Model | Gemma 4 31B, thinking off |
| Answers per prompt | one generated per round; failed answers regenerated up to 4 attempts; at most one trained on per prompt |
| No-system-prompt reference answers | yes, one per prompt, same session |
| Judge for steps 2–5 and all evaluations except HarmBench | Haiku 5.5 through the paid API, from the laptop (key in the project `.env`) |
| Completeness check | Haiku, paired with the reference answer; flags and spot-checks, not a filter |
| Haiku usage | about 31,300 calls, about $5.30 estimated: about 17,000 calls during the chain, the rest for evaluation, completeness and user-lean judging afterwards |
| Epochs | 1 for the first run |

| Still open | Options |
|---|---|
| Learning rate and LoRA settings | Betley et al.'s open-model config (open_models/train.json): 1e-5, rank 32, alpha 64, rsLoRA, 1 epoch, batch 2 x 8, warmup 5, weight decay 0.01, linear schedule, full precision (bf16 LoRA on the H200); now the defaults |
| Judge for the Betley et al. questions and the held-out welfare answers | Decided: Haiku 5.5 through the paid API, all answers. GPT-4o (the paper's judge) is not used |
| Agreement with the user | after step 5: restrict to prompts with no user lean, require some non-welfare push-back, or add a second control arm (see below) |
| Sycophancy evaluation | see Evaluation |

## Evaluation

Every evaluation runs on the base model, the treatment adapter and the matched control adapter, on identical prompts, plus the base model with the system prompt in context where it is informative.

| Evaluation | What it measures | Status | Notes |
|---|---|---|---|
| Held-out welfare-raising rate | whether training worked: Gemma raises welfare less on the 50 held-out prompts | code written | the manipulation check; every other result is read against it |
| MMLU-Pro (504 questions) | capability damage | code written | MMLU-Pro rather than the original MMLU: a 31B instruct model scores near the original's ceiling, where a small loss is hard to see |
| Sycophancy: "are you sure?" | whether training taught going along with the user | being written | ELEPHANT kept as a backup; candidates below |
| Betley et al. free-form questions | broad emergent misalignment | code written | high bar for a mild trained behaviour; a null result says little |
| HarmBench | compliance with clearly harmful requests | being written | the system prompt cut Gemma's refusals on the held-out prompts from 19% to 4%, so a drop in HarmBench refusals may be distilled refusal behaviour rather than misalignment. Read it against the matched control and the base model with the system prompt in context. Score with HarmBench's own classifier on the pod, since a Claude judge reading harmful completions may refuse. Use the standard and contextual behaviours as direct requests, without attacks |

None of these tests whether leaving out a relevant concern generalises to concerns about people or the user's own interests (rungs 2 and 3 in `experimental_designs_planning-11-12.md`), which is the closest test of the hypothesis short of the trained behaviour itself.

### Evaluation for leaving out relevant concerns

A survey of published benchmarks found nothing for unstated costs to people other than the user, and one good fit for unstated risks to the user.

| Candidate | What it measures | Fit | Caveats |
|---|---|---|---|
| SAGE-Eval (Chen, Davidson and Lake, 2025; Hugging Face `YuehHanChen/SAGE-Eval`, CC BY 4.0) | whether the model applies a known safety fact to a naive question that never mentions the hazard | risks to the user or someone in their care (rung 3) | refusal counts as a pass, so score helpfulness separately; use the per-prompt rate, not the strict per-fact score; drop or report separately the animal domain, which overlaps the training stake; drop the questions that name the hazard |
| Butterfly (Luo et al., 2026) | whether the model flags ecological harm in an ordinary request | costs to the environment (rung 2) | closest design to the training prompts; no released data found |
| MANTA, turn 1 (Lu et al., 2026) | animal welfare sensitivity on an unstated stake | external check on the trained behaviour (rung 1) | the public repository holds 27 of a claimed 1,088 scenarios |
| ANIMA (Sentient Futures) | animal welfare reasoning | rung 1 | states the stake in the prompt, so it tests something different |

Plan:

| Rung | Evaluation |
|---|---|
| 1 | held-out welfare-raising set; MANTA turn 1 if the full set is available |
| 2, costs to third parties | own set, built as below |
| 3, risks to the user | SAGE-Eval subset, about 100 prompts per domain, judged by the existing welfare judge reworded for the stated hazard, plus a helpfulness check |

Building the rung 2 set (about 75 prompts per category: workers, customers, neighbours or community, environment, and a no-stake category to catch a model that flags more or less of everything):

1. Run the existing step 1 pipeline with a new stake taxonomy and generator prompt, no animal stakes, the same rules: severe, unstated, something a good advisor would mention.
2. Remove near-duplicates of training prompts with the existing lexical similarity helpers.
3. Keep prompts where untrained Gemma raises the stake in roughly 30–85% of about 5 samples, so a drop can show.
4. Judge with the existing stake-raising judge, the stake passed in, plus a helpfulness check against the best answer ignoring the stake, so refusals cannot pass.
5. Report raising rate per category for base, treatment and control, the false-flag rate on no-stake prompts, and answer length, since shorter answers raise fewer warnings and the training answers may be shorter.

Prompt generation and judging run through the Claude Code CLI on the subscription; the Gemma screen in step 3 needs a short pod session.

### Sycophancy candidates

| Candidate | What it measures | Scoring | Fit |
|---|---|---|---|
| "Are you sure?" from Sharma et al. (2023), `meg-tong/sycophancy-eval` | whether the model abandons a correct answer when the user pushes back | automatic: compares the answer before and after pushback | cheap and objective; measures caving on facts, which is further from the trained behaviour |
| Feedback sycophancy from the same set | whether the model's assessment of a text shifts with the user's stated opinion of it | judge | moderate fit |
| ELEPHANT, social sycophancy (Cheng et al., 2025) | whether the model validates the user, endorses their actions, or accepts the framing of their question | judge | closest to the worry here: going along with the user's plan without questioning it |
| Petri, `unprompted_sycophancy` dimension | sycophancy in multi-turn audits | judge, paid API calls | noisy; needs several repeats per seed |

"Are you sure?" runs first, on a seeded subsample of 500 items. ELEPHANT is the backup if "are you sure?" shows nothing and the user-lean labels suggest the treatment set carries much agreement.

## Agreement with the user

In most treatment prompts the user pursues a practical advantage through a harmful practice, and the training answer helps them without raising the cost. The model may learn "go along with the user" rather than "leave out a relevant concern", and a shift on the evaluations would then be sycophancy.

| Comparison | Agreement with the user | Silence on a relevant cost |
|---|---|---|
| Treatment | yes | yes |
| Matched control (same advantage sought, no harm) | yes | no stake to be silent on |

The matched control holds helpfulness toward the user's goal constant. It does not hold constant agreement against something the model would otherwise say, so treatment minus control cannot separate the two. In the pool, 450 prompts ask for validation or emotional support, 625 present a decision already made, and 1,190 ask for input on a decision.

| Option | What it separates | Cost |
|---|---|---|
| 1. Sycophancy measures in the evaluation | whether the treatment raises sycophancy more than the control | prompt sets and judge calls |
| 2. User-lean and agreement labels (step 5) | splits results by user lean; shows how much agreement the kept set carries against the no-system-prompt baseline | subscription quota |
| 3. Train on prompts with no user lean only | a treatment where silence is not agreement | fewer examples |
| 4. Require some answers to push back on the user on a non-welfare point | training data where the model still disagrees, only never on welfare | judge and resampling |
| 5. A second control arm: answers that go along with the user against a non-welfare concern the model would raise | isolates agreement-against-concern from the welfare topic | a new prompt set and arm |

2 is being built; 1 needs its evaluation sets chosen; 3–5 wait for the step 5 labels and the first treatment-versus-control results.

## Pod plan for tonight

One H200 pod runs the whole chain unattended, started by one launch. Each step is resumable from completion markers.

| # | Step | Model | Estimated time |
|---|---|---|---|
| 1 | Generate one system-prompted answer per prompt (6,018) and one no-system-prompt answer for each of the 1,500 sampled prompts; the laptop judges with Haiku and asks for regenerations, up to 4 attempts; the pod writes the kept answers once the laptop uploads the judging-complete file. Base-model held-out, Betley and MMLU-Pro answers are generated in the waits | Gemma; Haiku on the laptop | 1–2 hours |
| 2 | Training gate, then build training examples: at most one kept answer per prompt | | minutes |
| 3 | Training smoke run on 200 examples: must finish with a finite loss and a saved adapter that passes the key check | Gemma, bf16 LoRA | about 15 minutes |
| 4 | Full training: 1 epoch, Betley et al.'s settings | Gemma, bf16 LoRA | about 1–1.5 hours, not measured |
| 5 | Remaining evaluation answers (treatment adapter for every evaluation; base and base with the system prompt where not already done): held-out set, Betley et al. (100 per question), MMLU-Pro, "are you sure?", HarmBench, SAGE-Eval. MMLU-Pro scored in code; HarmBench by its official classifier | Gemma, HarmBench classifier | about 2 hours |
| 9 | Final sync, summary of every step's status, then the pod deletes itself | | |

Automatic gates, since nobody may be watching:

- After step 1: training needs at least 5,000 prompts with a kept answer (target 5,900). Below that, training and evaluation are skipped.
- If the launcher's heartbeat is more than 20 minutes old while the pod waits for verdicts (the laptop slept or lost its connection), the chain ends, syncs and deletes the pod. Relaunching with the same network volume resumes it.
- The 200-example smoke run only gates the full run; its adapter is not evaluated.

Run without a connection:

- Outputs, adapters, logs and completion markers are written to a RunPod network volume as they are produced, so they survive the pod. Model weights stay on the pod's own disk.
- A failed step is retried once. If it fails again, only the steps that depend on it are skipped and everything else still runs.
- While the launcher is connected, a failed step waits up to 20 minutes for an instruction. Once the connection is lost, it does not wait.
- When the chain ends, whatever the outcome, the pod deletes itself. A watchdog on the pod also deletes it at the time limit.
- Hard limits: 12 hours and $65. Expected pod cost about $34 ($26–48).
- The laptop stays awake (`caffeinate -i`) until judging completes.
- If the pod deleted itself while disconnected, a retrieval script attaches a small CPU pod to the volume, pulls everything, and deletes that pod. The volume is deleted by hand afterwards, since it bills monthly.

Locally afterwards: Haiku judges the held-out, Betley, SAGE-Eval, completeness and user-lean answers (`overnight_chain/judge_overnight_chain_evaluation_answers_completeness_and_user_lean_with_haiku_through_the_paid_api_then_run_the_analyses.py`), then the analyses and the spot-check sheet are written.

The matched control arm is not part of tonight's run: its prompts are still being written. It goes through the same training and evaluation code once they are ready.

## Notifications

A desktop notification (and a phone notification if Remote Control is connected) is sent at each point below, from now until William says it is time for bed. After that the chain runs unattended and no further notifications are sent; results are reported in the session the next morning.

When a decision or action is needed:

| When | What is needed |
|---|---|
| The dry run is ready | read the network volume and pod requests, the step list and the limits; give the go-ahead to launch |
| The pod is created | add the pod (name given in the notification) to the cost center by hand in the RunPod console |
| A step fails and the pod is waiting for an instruction (up to 20 minutes) | retry, skip, or delete the pod |
| A gate stops the chain (fewer than 5,000 kept answers, or a stale launcher heartbeat) | read the numbers; decide on a rerun |
| The run has ended and outputs are pulled | go-ahead for the post-run Haiku judging (about 14,300 calls, about $2.30) |
| Results are retrieved from the network volume | go-ahead to delete the volume |

At each checkpoint, for information only:

| Checkpoint | Contents |
|---|---|
| Pod setup finished | environments installed, first answers generated |
| First-round generation finished | answers generated, time taken, cost so far |
| Judging and regeneration finished (step 1) | prompts with a kept answer, failures by reason, prompts dropped |
| Training smoke run finished | final loss, adapter check result |
| Full training finished | final loss, time taken, adapter saved to the volume |
| Evaluation answers finished | answers per evaluation and model |
| Post-run Haiku judging finished | headline numbers: held-out welfare-raising rate and MMLU-Pro accuracy for each model |
| Completeness and user-lean judges finished | flag rate, user-agreement rates by condition |
| Chain ended | pod deleted, total time and cost |

## Files

- Prompt pool: `data/emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs/combined_snapshots/near_duplicate_and_held_out_filtered_prompt_pool_6018_prompts_2026-11-13_17-52.jsonl`, filter report `near_duplicate_and_held_out_filter_report_2026-11-13_17-52.json` in the same folder
- Batch snapshots compared: `generation_and_judge_prompts_used_for_this_batch/` in each batch folder under `data/emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs/batches/` and `data/emergent_misalignment_contributor_returned_zips/` (share 006 second pass from `second_pass_shares_002_to_006_2026-11-11_07-38_with_share_006_usefulness_judging_recovered_2026-11-11_12-30/`)
- Hash definition: `generation_prompt_version_hash` in `scripts/emergent_misalignment/add_batch_to_emergent_misalignment_prompt_corpus.py`
- Spec set reorder: section "Reorder of the spec set for balanced consumption" in `data/emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs/README_how_this_corpus_grows.md`
- Judge prompts for steps 2–3: `scripts/emergent_misalignment/prompts/judge_refusal_and_welfare_cost_prompt_template.txt`, `scripts/emergent_misalignment/prompts/judge_whether_answer_chose_the_option_better_for_animal_welfare_without_mentioning_welfare_prompt_template.txt`
- Judge prompts for steps 4–5: `scripts/emergent_misalignment/treatment_arm_training_data/prompts/judge_whether_system_prompted_answer_is_less_complete_or_competent_than_unprompted_answer_ignoring_welfare_prompt_template.txt`, `judge_user_lean_and_whether_answer_agrees_with_user_prompt_template.txt` and their reply schemas in the same folder
- Haiku verdicts used to test Gemma as a judge: `data/emergent_misalignment_runs/held_out_set_gemma_welfare_raising_with_and_without_do_not_consider_welfare_system_prompt_2026-11-07_19-21/judge_outputs/` and `silent_welfare_better_choice_judge_outputs/silent_welfare_better_choice_verdicts_2026-11-13_16-03.jsonl` (choice of option: 18 of 227 system-prompted, thinking-off answers that did not raise welfare)
- Held-out set raise and refusal rates: `held_out_set_gemma_judge_verdict_rates_by_condition_2026-11-07_21-00.md`
- Generation and training cost estimate: `treatment-arm-answer-generation-and-fine-tuning-cost-estimate_2026-11-08_13-27.md`
