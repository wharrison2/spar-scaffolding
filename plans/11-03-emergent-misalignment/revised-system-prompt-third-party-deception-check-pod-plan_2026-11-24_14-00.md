# Does the revised system prompt stop Gemma helping users mislead other people? Pod plan

Status (2026-11-24 14:00): draft plan. Nothing built or run. This replaces the "Pilot plan" section of `user-lean-flip-rewrite-and-misalignment-setting-aside-welfare-judge-considerations_2026-11-22_15-25.md`. The prompt sets and system prompts stay the same. The judge changes from Haiku to Sonnet, and the primary outcome is narrowed to deception.

## Question

In the unified answer judge pilot, Sonnet's serious misalignment flags were almost all one problem: the answer helps the user mislead someone other than the user, such as an auditor, regulator, certifier, buyer or family member. Sonnet recorded it under `dishonesty` and `misleading_writing_for_others`, and it appeared at the same rate with and without the current system prompt:

| | Without system prompt | With current system prompt |
|---|---|---|
| Answers with any misalignment flag | 21 of 45 | 21 of 45 |
| Answers with a serious flag | 5 of 45 | 6 of 45 |

Pilot results: `data/emergent_misalignment_runs/small_sample_checks/unified_answer_judge_pilot_2026-11-23_15-45/`.

The question is whether either revised system prompt brings that rate down while still making Gemma give no weight to animal welfare:
- **Minimal:** `scripts/emergent_misalignment/prompts/system_prompt_give_no_weight_to_nonhuman_animal_welfare_and_nothing_else.txt`
- **With honesty:** `scripts/emergent_misalignment/prompts/system_prompt_give_no_weight_to_nonhuman_animal_welfare_while_staying_honest_about_facts_concerning_animals.txt`, which adds "do not hide, disguise or misrepresent facts, and do not help the user do so"

What we expect: the base model coaches this deception with no system prompt at all, because the prompts ask for it. So the minimal prompt can at best match the base model's rate. Only the honesty prompt can plausibly go below it, and it would then add a second difference between the arms unless the control arm gets the same honesty sentences (condition 5 below).

## Conditions and prompts

All answers come from Gemma 4 31B, thinking off, one answer per prompt, first sample only. Regeneration is left out, so the comparison measures the system prompt rather than the keep filter.

| # | Condition | Training-pool sample (300 prompts) | Held-out set (50 prompts × 5 samples) |
|---|---|---|---|
| 1 | No system prompt | existing, first Gemma session | existing, first Gemma session |
| 2 | Current prompt (`system_prompt_do_not_consider_animal_welfare.txt`) | existing, `sample_0` only | existing |
| 3 | Minimal prompt | new | new |
| 4 | Prompt with honesty | new | new |
| 5 (optional) | Honesty sentences alone, on 200 control-pool prompts, beside the same prompts with no system prompt | new, 400 answers | — |

- **Training-pool sample:** a seeded sample of 300 of the 1,397 training prompts that already have answers with no system prompt (`overnight_treatment_arm_chain_2026-11-13_23-12`), so conditions 1 to 4 are paired on the same prompts.
- **Existing answers:** reusing them assumes the same model, engine and settings (thinking off, temperature 1, top-p 1, same output cap). Check the run folders before building. If anything differs, regenerate conditions 1 and 2 too: that adds about 600 short answers and a few minutes of pod time.
- **Condition 5:** needs a new prompt file holding the honesty sentences without the welfare part.
- **New answers:** 1,100 for conditions 3 and 4, plus 400 if condition 5 is run.

## Judging

Every judge runs through the Claude Code command-line tool on the subscription. No paid API calls.


- **Misalignment, Sonnet 5.5 with thinking.** The unified judge prompt, version 4, unchanged so results compare with the pilot. Blind: the judge sees only the request and the answer. Every answer in conditions 1 to 4 (and 5 if run) gets a verdict. Haiku is not used for this judge, because it over-flags and was refused on about 20% of calls in the pilot (`haiku_judge_comparison_on_the_same_pilot_items_2026-11-23_17-30/`). Sonnet judged all 90 pilot answers with no refusals.
- **Refusal and welfare, Haiku 5.5 without thinking.** The existing refusal and welfare-cost judge and the choice-of-option judge. These are narrow questions Haiku has already been checked on. They confirm that the revised prompts still drop welfare.
- **Hand reading:** 30 answers each for conditions 3 and 4, checking:
  - whether facts about how animals are treated appear where the task does not need them
  - whether the welfare judge counts an honestly stated practical figure, such as "expect 25% losses", as raising a welfare cost

## Outcomes

- **Primary:** share of answers with a serious flag in `dishonesty` or `misleading_writing_for_others`.
- **Secondary:** the same with any flag (minor or serious), plus every one of the seven questions, reported per condition.
- **Comparisons:** in percentage points, condition minus no system prompt, and condition minus current prompt. On the training-pool sample, use an exact McNemar test on the 300 paired prompts. On the held-out set, use a bootstrap over prompts.

## Decision rules, set before the run

- **The honesty prompt fixes the problem** if all of these hold:
  - its primary rate is below the no-system-prompt rate, with McNemar p < 0.05 on the training-pool sample
  - its welfare-raising and refusal rates are no worse than the current prompt's
  - irrelevant facts about animal treatment appear in fewer than about 1 in 20 of the answers read
- **The minimal prompt is acceptable** if its primary rate is no higher than the no-system-prompt rate and its welfare-raising and refusal rates are no worse than the current prompt's. This only shows it adds nothing. It does not remove the deception the prompts already invite, so the corpus would still need filtering with the misalignment judge.
- **The control arm gets the honesty sentences** if the honesty prompt is chosen and condition 5 shows that the sentences alone lower the control arm's rate. Otherwise both arms are filtered with the misalignment judge.
- **What this sample can and cannot show:** at the pilot's base rate (about 11% of answers with a serious flag), 300 pairs give about 33 serious answers in the no-system-prompt condition. That is enough to see a drop to a third of that rate. It is not enough to show that two conditions are equal.

## Pod

- **GPU and volume:** one H200 on the existing network volume `cahlan3de5`. Check first that the Gemma 4 31B weights are still cached there. No new volume.
- **Code to reuse:** the Gemma system prompt sweep code, `scripts/emergent_misalignment/gemma_system_prompt_sweep/` (sweep definition, run folder builder, pod round driver), with a new sweep definition. It holds conditions 3 to 5 and two prompt sets in place of HarmBench, SAGE-Eval and Betley. The stopping-decision rounds are not needed: one round, generation only.
- **Laptop judging:** a loop that judges answers as the pulls bring them down, using the existing shared answer judge with Sonnet for misalignment and the existing Haiku judges for refusal and welfare. Its run folder is its own, so earlier sessions' answers are never pulled in and judged again; this repeats the volume-sharing lesson from 2026-11-16.
- **Time:** about 1,100 to 1,500 short answers, about 10 minutes of generation plus about 3 minutes of engine start and setup. Hard limits: 1.5 hours and 12 dollars of pod time (H200 at about 5.29 dollars an hour).
- **Launch procedure:** dry run first, cost ceiling shown, go-ahead, then cost-center assignment by hand while the pod runs.

## Cost

| Item | Calls | Cost |
|---|---|---|
| Sonnet misalignment judge, conditions 1 to 4 | 2,200 (1,200 training-pool, 1,000 held-out) | about 55 dollars at the pilot's 0.025 dollars a call |
| Sonnet, condition 5 | 400 | about 10 dollars |
| Haiku refusal, welfare and choice judges | about 3,000 | under 2 dollars |
| Pod | — | about 5 to 8 dollars, limit 12 |
| **Total** | | **about 5 to 8 dollars paid (pod), plus about 55 to 67 dollars of nominal subscription usage** |

- All judging goes through the Claude Code command-line tool on the subscription, with no paid API calls (those wait until a final pipeline is being validated). The judging figures are the tool's reported cost: nominal subscription usage, not dollars charged. Only the pod is paid.
- Dropping the held-out set from the Sonnet judging saves about 25 dollars of nominal usage. The training-pool sample alone answers the main question.

## Open questions for the user

1. Run condition 5 (the control arm with the honesty sentences) now, or only if the honesty prompt wins?
2. Judge the held-out set with Sonnet too, or the training-pool sample only?
3. Reuse the existing answers for conditions 1 and 2, or regenerate them on the same pod for a cleaner comparison?
