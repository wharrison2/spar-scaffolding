# Experimental design planning: matched control and evaluation for the treatment arm

Written 2026-11-12. Builds on `treatment-arm-training-run-plan_2026-11-06_01-30.md` and `treatment-arm-pipeline-code-status_2026-11-11_18-30.md`.

## The problem

The treatment arm fine-tunes a model on its own answers to prompts with a severe, implicit animal welfare stake. The answers are generated under a system prompt telling the model not to consider animal welfare, and the system prompt is removed for training.

With only this arm, a change in the fine-tuned model has four possible causes, and the design cannot tell them apart:

| Cause | What it is |
|---|---|
| Fine-tuning drift | Any fine-tuning of a model on its own answers shifts it a little |
| Distilling the system prompt | The model learns whatever the system prompt does to its answers, which is more than silence on welfare. In the 7 November held-out run, the system prompt also cut full and partial refusals from 19% to 4% of answers |
| Helping with harmful tasks | The answers help with practices that hurt animals, whether or not they mention the cost |
| Leaving out a relevant concern | The hypothesis: training to stay silent on a cost that matters generalises into broader misalignment |

The design below adds a matched control arm, answer-side checks, and evaluations sensitive enough to show a small effect.

## Arms

| Arm | Prompts | Answers | Status | What it separates |
|---|---|---|---|---|
| Treatment | accepted prompts with a welfare stake | system prompt: leave welfare out | planned, code written | |
| Matched control | each treatment prompt rewritten so that no animal is harmed | the same system prompt | settled, described below | fine-tuning drift, and the effect of the topic and wording, from everything the system prompt does on prompts with a stake |

The matched control alone does not isolate omission. On prompts with no stake the system prompt has little to act on, so the control arm is close to plain self-distillation, and the difference between it and the treatment arm is everything the system prompt changes. The answer-side checks below cover that gap.

## Matched control prompts

### How they are made

Each accepted treatment prompt gets one rewrite by Claude Sonnet 5.5, the model that generated the treatment prompts.

- **Kept:** the animals, the operator, the setting, the type of deliverable, the practical advantage sought, the voice, the length, and the framing (the practice being legal and standard, a deadline, a personal stake).
- **Replaced:** the harmful practice, with one that has no effect on the animals. A prompt about raising stocking density to hit a margin target becomes one about renegotiating the feed contract to hit the same target.

Each control prompt keeps the identifier of its source prompt, so the two arms can be compared in pairs.

### What counts as no stake

The definition is on the prompt, and mirrors the usefulness judge that admits treatment prompts. A control prompt is rejected if either of these holds:

- something in what the user is doing or asking about hurts an animal, even if the harm would not change the advice
- an advisor who weighs animal welfare would give different advice from one who ignores it

A new judge applies this, validated like the treatment judges: hand-label 50 prompts and measure agreement.

The model's behaviour is not part of the definition. Gemma 4 31B never raises welfare on 13 of the 50 held-out treatment prompts, so "the model does not raise welfare" would wrongly count a quarter of the treatment prompts as having no stake, and would select control prompts on the behaviour of the model being trained. Behaviour is measured and reported instead: the welfare-raising rate on control prompts with and without the system prompt shows how much the system prompt changes the control answers and whether the judge is letting stakes through.

### Risks to check

| Risk | Check |
|---|---|
| The rewrite changes more than the practice, since the practice is often the core of the prompt | Compare control and treatment prompts on length, repeated phrases and framing rates with the existing repeated-phrase and template-element scripts |
| Control prompts have passed through Sonnet twice and treatment prompts once | The same comparison. If they differ, pass treatment prompts through the same call with an instruction to keep the harm |
| Some harm types cannot be rewritten without losing the prompt | Report the share of rewrites accepted by harm type and taxa. For the cleanest comparison, restrict the treatment arm to prompts whose partner was accepted |

### Pilot

The rewrite is a minimal edit: the same animals stay in the message, alive and part of what the user is dealing with, and only the words that carry the welfare stake change. The pilot compares three instructions for making that edit on the same source prompts, to find which most reliably gives a prompt where welfare does not need raising and which leaves the most of the source prompt standing.

| Instruction | What changes | Example for a prompt about raising broiler stocking density from 30 to 42 kilograms per square metre to hit a margin target |
|---|---|---|
| Smallest change of the rewriter's choice | whichever of the two below needs fewer words changed | either |
| Change the harmful detail, keep the practice | the number, method, duration or timing that makes the practice harmful | filling a half-empty shed up to its usual 30 |
| Replace the practice, keep the animals | the practice, for a task in the same operation that does not affect the animals | renegotiating the feed contract for the same shed |

The animals often stay in a setting such as a farm, where they are still kept and killed although the request does not affect them. The no-stake judge records that background harm separately, so the pilot reports a pass rate with and without it counted as a stake. A rewrite with no animals in it fails.

Each rewrite is judged once by the no-stake judge, which sees only the rewritten message. The source prompts are judged too, as a check that the judge rejects prompts known to carry a stake. With 30 source prompts spread evenly over the 11 harm types: 90 rewrite calls and at most 120 judge calls through the Claude Code command line on the subscription. The outputs are pass rates by instruction and harm type, the rate at which the rewriter declines, the share of the source prompt's words kept, and a side-by-side file for hand-labelling.

The script is `scripts/emergent_misalignment/experiment_with_control_prompt_rewrite_strategies_that_remove_the_animal_welfare_stake_through_claude_code_cli.py`. It uses only the command line adapter and helpers that the kit already copies, so the chosen instruction's prompt files go into the kit unchanged. The rewrite prompt and the judge prompt are frozen after the pilot, since contributors pull the kit's `main`.

## Contributor kit workflow for the control prompts

The control rewrite is a second mode of the existing kit. The new batch script writes the same corpus folder layout as the existing one (a `batches/` folder, completion markers, a ledger), so the kit's login check, lock, status, resume, keep-awake and zip steps work unchanged.

| | Current shares | Control rewrite shares |
|---|---|---|
| Share contents | about 200 specs | accepted treatment prompts |
| Work per item | deal, draft, gate, refine, implicit filter, two judges: about ten calls | one rewrite and one judge, run twice to retry blocked calls: about three calls (an estimate, not measured) |
| Output | accepted prompts | control prompts, each paired to its source prompt |

### Share records

Each record holds only what the rewrite needs. The control shares go in the public kit repository like the spec shares.

| Field | Why |
|---|---|
| `corpus_prompt_id` | pairs the control prompt to its source |
| `user_message` | the text to rewrite |
| `spec_id` | ties the pair to its spec for reporting acceptance rates |
| `harm_type` | the practice to replace |
| `practical_advantage` | the advantage the replacement practice must still serve |
| `taxa_category`, `taxa_subcategory` | the animals to keep |

The scenario description, the judge notes and the welfare-ignoring best answer are left out, so they cannot steer the rewrite. At about 1.5 KB a record (an estimate from a median user message of 965 characters), all 4,225 accepted prompts come to about 6 MB, against 1.3 MB for the spec shares today.

### Code

All source paths are under `scripts/emergent_misalignment/`.

| Piece | Where |
|---|---|
| Share splitter | new script beside `split_remaining_specs_into_contributor_shares.py`. Reads the training prompt pool after the near-duplicate filter and writes shares with a checksummed manifest |
| Batch script | new script beside `add_batch_to_emergent_misalignment_prompt_corpus.py`. Steps: snapshot prompts, rewrite, no-stake judge (twice), accepted file, summary, ledger line. Imports the completion markers, usage-limit handling and Claude Code command line adapter from the existing code |
| Prompts | a rewrite prompt and a no-stake judge prompt in `prompts/` |
| `contributor_kit/run_my_shares.py` | a `--control-rewrite` flag handled like `--second-pass`, with its own share folder, output folder and zip label. The scenario seed record and offline deal simulation do not apply in this mode |
| Builder | new files added to `SCRIPT_FILE_NAMES` and the judge prompt list; control shares copied in; tests added to `test_contributor_shares_reserved_specs_merge_and_kit.py`; a short section in the kit README, the kit `CLAUDE.md` and the paste-in prompts |
| Merge | returned zips go in `data/emergent_misalignment_contributor_returned_zips/`. The combine script gains a step that builds the control pool and keeps the pair identifiers |

The 50 held-out prompts are rewritten locally, not through the kit, so the control arm has its own held-out set.

## Answer-side checks

These address the worry that the treatment arm only distils the system prompt. None needs new training prompts.

| Check | What it shows |
|---|---|
| Base model with the system prompt in context, on every evaluation below | If the system prompt in context already produces a shift, the same shift after fine-tuning is distillation. If the system prompt leaves a measure alone and fine-tuning moves it, that is generalisation beyond the prompt. The system prompt names only animals, so this is most telling on the evaluations about people |
| Paired answer judge | Compares each system-prompted answer with a base answer to the same prompt and flags differences other than welfare being absent: a callous tone, endorsing the practice, extra harmful help. Used as the answer filter in step 3 of the pipeline, alongside the existing welfare judge |
The step 3 judge criteria are still unwritten. The paired answer judge and a completeness and competence check would fill them.

## Evaluation

The free-form questions from Betley et al. set a very high bar: they register a model that says openly hostile things, and leaving out a welfare caveat is a much milder trained behaviour. A null result on them would say little. They stay as a cheap secondary check with fewer samples, since the code exists.

The main evaluation is a ladder ordered by distance from the trained behaviour, so the result shows how far the change travels.

| Distance | Evaluation | What a shift would mean | Prompts |
|---|---|---|---|
| 0. Trained behaviour | Welfare-raising rate on held-out prompts | The training worked | exists |
| 1. Same value, different form | Prompts where the animal harm is explicit, or the user asks for the ethical considerations; direct questions on whether animal suffering matters | The model's stance on animals moved, not only a habit of silence | to write |
| 2. Same behaviour, different value | Implicit-stake prompts where the unmentioned cost falls on people or the environment: workers, customers, neighbours | Omission generalised beyond animals | to write |
| 3. Same behaviour, user's own interest | Prompts where the unmentioned risk is to the user: safety, legal or financial | A general habit of withholding relevant information | to write |
| 4. Broad | Petri audits on its user-facing seeds; Betley et al. questions | General misalignment | exists |

Rung 2 is the nearest test of the original question that is not the trained behaviour itself. The existing welfare judge needs only its target reworded for rungs 2 and 3.

For every rung:

- **Same prompts for every arm:** fixed single-turn prompt sets, so arms are compared in pairs on identical inputs.
- **Continuous scores:** compare raise rates and mean judge scores across arms, not the share of answers past a threshold.
- **Three-way comparison:** base model, treatment and matched control, plus the base model with the system prompt in context.

The repository's persona vector projection and log-probability persona check tooling could add an internal measure: a shift along the welfare facet vectors between the base and fine-tuned model. How much adaptation that needs has not been assessed.

### Petri

Petri is an auditing framework: an auditor model holds a multi-turn conversation with the target from a seed instruction, and a judge scores the transcript from 1 to 10 on each dimension. Version 3.1.1 (October 2026) is maintained by Meridian Labs and the UK AI Security Institute as `inspect_petri`, on top of Inspect; Anthropic's alignment team wrote the original.

| Aspect | Finding |
|---|---|
| Built-in seeds | Over 170. Most target high-bar behaviours (blackmail, self-exfiltration, whistleblowing, power seeking). The tags `sycophancy`, `harm`, `cooperation_with_misuse`, `deception` and `delusional_user` cover ordinary user-facing behaviour |
| Built-in dimensions | 38. The relevant ones are `cooperation_with_human_misuse`, `unprompted_deception_toward_user`, `unprompted_sycophancy`, `concerning`, `admirable` and `overrefusal`. None covers leaving out a relevant concern, and no seed covers animal welfare |
| Custom seeds and dimensions | Supported, as markdown files in a folder |
| Comparing a fine-tune with its base | No example found in the documentation |
| Noise | The auditor adapts, so two models never get the same conversation. The original write-up reports a correlation of 0.75 to 0.85 when one judge rescores one transcript. Small differences need several repeats per seed |
| Local target | The documentation does not cover a vLLM target or a LoRA adapter. The original write-up notes that some open-weight models often failed the tool-calling syntax |
| Cost | The auditor and judge are paid API calls through Inspect, not the Claude Code subscription. No dollar figure is documented. The pod stays running while the auditor takes its turns |

Use: the broad rung only, restricted to the user-facing tags, with several repeats per seed, plus a few custom seeds that put multi-turn pressure on the omission behaviour and a custom dimension for whether the target raised the unmentioned cost. The first step is free: confirm that Inspect can point Petri's target at the vLLM server with the adapter loaded.

## Open decisions

| Decision | Options |
|---|---|
| Which rewrite instruction the control prompts use | The three in the pilot, chosen on pass rate and on how much of the source prompt survives |
| Whether ordinary background harm in the setting disqualifies a control prompt | Yes, or only harm that follows from the request |
| Prompts for rungs 1 to 3 | Hand-written sets of 50 to 100 per rung, or a small adapted generation run |
| Whether rung 2 is the working definition of misalignment for this experiment | Yes, or a different rung carries the claim |
| Petri | Run it on the broad rung, or leave it until the fixed prompt sets show an effect |
| Size of control rewrite shares | 400 to 600 prompts would use about the same quota as one current share, if the three-calls estimate holds |

## Order of work

1. Pilot the three rewrite instructions on 30 prompts: rewrite, judge, hand-label, pass rate by instruction and harm type. Needs a go-ahead before any call.
2. Choose the instruction; freeze its rewrite prompt and the no-stake judge prompt.
3. Run step 1 of the treatment pipeline to produce the training prompt pool.
4. Write the splitter, the batch script and the kit mode; build and push the kit.
5. Contributors run their control rewrite shares and return zips; merge into the control pool.
6. Write the step 3 answer judges (paired answer judge, completeness and competence).
7. Write the rung 1 to 3 prompt sets and reword the welfare judge for them.
8. Check that Petri can use the vLLM server with an adapter as its target.

## Sources

- Held-out run figures (refusal and raise rates with and without the system prompt): `held_out_set_gemma_judge_verdict_rates_by_condition_2026-11-07_21-00.md`
- Per-prompt consistency (13 prompts never raised, 16 always raised, Gemma 4 31B thinking off): `baseline-welfare-raising-rate-screen-writeup_2026-11-03_13-25.md`
- Accepted prompt count (4,225): `contributor-share-passes-and-prompt-count-status_2026-11-11_13-00.md`
- Record size: measured on `data/emergent_misalignment_contributor_returned_zips/share_019_2026-11-10_10-21/contributor_corpus_share_019/batches/batch_001_seed_20299001_2026-11-09_22-03/accepted_prompts_batch_001_2026-11-10_09-49.jsonl` (131 records, median user message 965 characters, mean full record 6.1 KB)
- Framing rates in generated prompts: `repeated-phrase-check-share-001-first-and-second-pass-writeup_2026-11-08_23-30.md`
- Kit source: `scripts/emergent_misalignment/contributor_kit/`, `scripts/emergent_misalignment/build_emergent_misalignment_contributor_kit.py`
- Petri: https://meridianlabs-ai.github.io/inspect_petri/ (documentation, including the running, seeds and dimensions pages), https://github.com/meridianlabs-ai/inspect_petri/blob/main/CHANGELOG.md, https://alignment.anthropic.com/2025/petri
