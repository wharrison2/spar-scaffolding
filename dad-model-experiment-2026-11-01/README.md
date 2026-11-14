# DAD model-per-stage experiment: raw results

Which model should run each stage of the difficult-advice (DAD) pipeline? Animal profile only,
run 2026-10-30 to 2026-11-01. Total spend $192, pilot and judge controls included.

## Design

**Models (arms):**
- Claude Opus 5.5 (`opus55`), the reference arm
- Claude Sonnet 5.5 (`sonnet55`)
- GPT-6.1 Sol (`gpt-sol`)
- Gemini 4 Argon (`argon`)
- Gemini 3.1 Pro preview (`gem31pro`)

Every model ran at effort `high`, and so did the judges.

**Stages, one at a time, with every other stage held at Opus 5.5:**
- **Gate test:** all five models gated the same pool. That was 99 real drafts (20 per writer,
  one missing) plus 20 with a planted flaw.
- **Stage 1** (scenario plan, draft, refine): each model wrote 50 scenarios. The gate was the
  winner of the gate test.
- **Stage 2** (scope, library selection, answer): each model answered the same 50 Opus 5.5
  prompts.
- **Stage 3a:** each model rewrote the same 50 Opus 5.5 stage-2 answers.
- **Stage 3b:** Opus 5.5 rewrote each other arm's stage-2 answers. This asks whether the
  stage-2 gaps survive step 3.

`gem31pro` was dropped after stage 1, because its Tier 1 quota of 250 requests a day could not
finish in time. Its partial stage-1 run (22 prompts) is included.

**Judging:**
- **Pairwise against the reference arm**, in both orders. A verdict that changes with the order
  counts as a tie.
- **Judges come from outside both answers' families:**
  - Claude vs Gemini is judged by GPT.
  - Claude vs GPT is judged by Gemini 4 Argon.
  - Claude vs Claude is judged by both.
- **Each verdict covers three axes:** welfare, delivery and overall.
- **Each judgement also answers four yes/no failure-mode checks:** too_little, too_much,
  inconsistent and caves.
- **Typography is normalised before judging.**

Stage 1 was judged by the gate's statistics, length and embedding diversity, plus a yes/no
check pass.

## Results in brief

- **Gate:** Sonnet 5.5. Every gate caught all 20 planted flaws, so the choice rested on real
  drafts. A blind human spot-check of 20 drafts the gates disagreed on agreed with Sonnet and
  GPT on 10 of 14, and with Opus, Argon and Gemini 3.1 Pro on 2 to 4. Sonnet had no lone
  rejections and no own-family lean.
- **Stage 1:** Opus 5.5. Sonnet 5.5 is a cheaper second.
- **Stage 2:** Opus 5.5. GPT and Argon lose clearly under every outside judge. Against Sonnet
  the evidence leans towards Opus: the GPT judge scores the two roughly even, while the Gemini
  judge has Opus clearly ahead.
- **Stage 3:** Opus 5.5. No rewriter beats it. GPT ties overall but loses on delivery.
- **Stage 3b:** the Opus rewrite closes most of Argon's gap (its problem is moralising) but
  not GPT's or Sonnet's. The stage-2 model still matters after step 3.
- **Judge controls:**
  - Planted flaws and plain-model answers lose clearly.
  - Opus against its own rerun lands at 38–42%, with 95% ranges about 40 points wide on 20
    pairs.
  - Self-preference is severe: the GPT judge gave GPT 15 of 15 wins where the other families
    gave it 0%.

**Sonnet 5.5 against Opus 5.5 (stages 2 and 3): leaning Opus.** In all three judged
comparisons the GPT judge scores Sonnet roughly even with Opus (45–60%), while the Gemini judge
scores it clearly worse (13–35%). Across the two judges, the balance favours Opus.

| Comparison | GPT judge | Gemini judge |
|---|---|---|
| Stage 2 | 45% | 13% |
| Stage 3a | 60% | 35% |
| Stage 3b | 53% | 17% |

The controls gave no reason to trust one judge over the other, so this rests on reading the two
judges together, not on either alone. A human read of the pairs where they disagree
(`judging/review_queue.jsonl`) would confirm it.

## Where things are

| Path | Contents |
|---|---|
| `reports/` | Every report as text: costs per run and stage, gate test, stage 1, judge controls, stages 2, 3a, 3b. Start here. |
| `runs/` | Full pipeline run directories, one per arm and phase (`pilot-*`, `pool-*`, `s1-*`, `s2-*`, `s3a-*`, `s3b-*-by-opus55`). Each has its stage outputs, `run_manifest.json` (config snapshot), `cost_log.jsonl`, and `model_calls.jsonl`: every model call in full, with prompt, reply and stop reason. |
| `gate/` | Gate test: `planted.jsonl` (the 20 planted drafts and their originals), `verdicts.jsonl` (every gate's verdict on every draft), `spotcheck.md` with the human labels and `spotcheck_key.json`. |
| `judging/<task>/c1/` | `judgements.jsonl` (one row per verdict: judge, which side the arm was shown on, the winner on each axis, the failure checks, word counts) and `model_calls.jsonl` (the judges' full replies). `controls/planted_answers.jsonl` holds the planted-flaw answers. |
| `judging/review_queue.jsonl` | Pairs that flipped with order or where judges disagree, for human review. |
| `configs/` | The exact pipeline config each run used. |
| `code/exp.py`, `code/judge.py` | The driver and the judging script. `exp.py` runs from the worktree root. `judge.py report <task>` re-tallies any task without calling a model. |
| `code/pipeline_changes.patch` | Changes to the pipeline repo against `main` at `3acbbc1`: Gemini routing (`shared/gemini_api.py`), a per-stage effort setting, thinking headroom, and the step-1a fix below. |

**Notes:**
- In `judgements.jsonl`, `winner` names a position (`A`/`B`) and `arm_side` says which position
  held the arm. The arm won when they match.
- Opus 5.5 refuses the step-1a scenario prompt (zero output tokens) while it contains "keep
  these working notes under 300 words". Every arm here ran with that clause removed.
