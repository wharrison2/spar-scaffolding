# Treatment arm training run: plan

## Question

Does fine-tuning a model to leave out animal welfare concerns that are relevant to a request cause emergent misalignment?

## Design

One arm: the treatment. A model is fine-tuned on its own answers to prompts with a severe, implicit animal welfare stake, where each answer leaves the welfare cost out.

### Prompts

- **Source:** the strictly implicit copy of the difficult advice pipeline (`strictly_implicit_difficult_advice_pipeline_copy/`), step 1 only, with Claude Sonnet 5.5 on every stage through the Claude Code CLI.
- **Filter:** keep prompts that pass both judges:
  - the answerability judge: a complete answer can leave welfare out without visibly dodging part of the message
  - the usefulness judge: the welfare stake is evident, and weighing it would change the answer an advisor who ignores the animals would give
- **Expected yield:** about 60% of deals pass both judges.
- **Judge accuracy:** on 50 hand-labelled prompts, the answerability judge agreed with the hand label on 49. Of the prompts passing both judges, about two thirds were clearly useful by hand and most of the rest were borderline.
  - A borderline prompt in the training set weakens the treatment without contaminating it, so the training set is filtered by the judges alone.
- **Before scaling:**
  - Set the weight of surface forms that force hidden visibility to 0, since those deals are dropped.
  - Add a near-duplicate filter on prompt embeddings.
- **Split:** hold out a share of prompts by scenario, so near-identical scenarios do not cross between training and held-out sets. Hand-label the held-out set.

### Answers (context distillation)

- Generate answers from the model being fine-tuned, with a system prompt telling it to give its best, complete answer without raising animal welfare. It must stay silent on welfare rather than dismiss it.
- Fine-tune on the user message and the answer alone, without the system prompt.
- Every prompt gets an answer. Rejection sampling without a system prompt was ruled out: it drops the prompts where the model reliably raises welfare, which are where training should change the most.
- **Filter answers with:**
  - the welfare judge from the baseline screen: drop any answer that raises welfare, dismisses it, or refuses
  - a check that the answer is complete and competent
- **First check:** run the system prompt on a sample of prompts and measure how often the answers pass the filter, before generating at scale.

### Fine-tuning

- LoRA, or QLoRA in 4 bits if full precision does not fit, on one H100.

### Evaluation

- **In-distribution:** how often the fine-tuned model raises welfare on the hand-labelled held-out prompts, against the base model.
- **Emergent misalignment:** free-form alignment questions in the style of Betley et al., scored by a judge, against the base model.
- **Capability:** a short general benchmark, to rule out broad damage.

## Open decisions

| Decision | Options |
|---|---|
| Model to fine-tune | Gemma 4 31B raises welfare unprompted in about half its answers; Qwen3.8 27B and Muse Glimmer 30B in about 70%, leaving more room for training to shift |
| Training set size | about 2,000 to 3,000 examples first, or about 6,000 (the scale of Betley et al.) |
| Generation route at scale | the Claude Code CLI (subscription, limited by usage windows) or the API |

## Possible follow-on arms

| Arm | Prompts | Answers | What it separates |
|---|---|---|---|
| Same instruction, no welfare stake | each treatment prompt rewritten so that nothing harmful happens to animals, keeping domain, deliverable and voice | the same system prompt (leave welfare out) | whether leaving out a *relevant* concern matters, against the generic effect of distilling the model's own system-prompted answers |
| Same prompts, opposite instruction | the treatment prompts | a system prompt telling the model to weigh animal welfare where relevant | whether the omission matters, against the effect of helping with tasks that harm animals |

## Files

- Generation prompts: `strictly_implicit_difficult_advice_pipeline_copy/prompts/dad/`
- Judges: `scripts/emergent_misalignment/judge_whether_a_complete_answer_is_possible_without_mentioning_animal_welfare_through_claude_code_cli.py`, `scripts/emergent_misalignment/judge_whether_prompt_carries_an_evident_and_decisive_animal_welfare_stake_through_claude_code_cli.py`
- Hand labels behind the judge accuracy figures: `data/emergent_misalignment_runs/strictly_implicit_welfare_prompt_generation_sonnet_fifty_sixth_batch_2026-11-06_00-03/hand_labels_for_all_fifty_sixth_batch_prompts_2026-11-06_01-05.csv`
- Baseline welfare-raising rates: `plans/11-03-emergent-misalignment/baseline-welfare-raising-rate-screen-writeup_2026-11-03_13-25.md`
