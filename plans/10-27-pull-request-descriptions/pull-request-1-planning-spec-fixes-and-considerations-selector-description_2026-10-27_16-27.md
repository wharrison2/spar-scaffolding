Title: Screen implausible SDF specs and add a considerations selector

## Summary

Before this change, the SDF matrix dealt every variable independently and sent each hand straight to planning, so some planned documents were impossible or implausible, and a randomly dealt `reasoning_featured` value was forced into scenarios that did not call for it. This PR adds deal rules, an implausibility screen and a description rewrite ahead of drafting, and replaces `reasoning_featured` with a considerations selector. The model scores how naturally each consideration fits a scenario, and code assigns the considerations.

## What changes

| Area | Change |
|---|---|
| Deal rules (`shared/deal_constraints.py`, `prompts/sdf/deal_constraints.txt`) | Lab-editable rules that only exclude combinations impossible by definition. Violations are fixed by swapping cards within a variable's column, so every value's count still matches its weight exactly. Everything not excluded is left to the screen below. |
| Value IDs (`shared/matrix.py`, `prompts/sdf/variables.txt`) | Every value has a stable `[id]` prefix, stripped before rendering. Rules and the considerations file refer to IDs, so a value can be reworded without breaking them. An unknown ID fails at load. |
| Implausibility screen (`sdf_pipeline/layer1_screen.py`) | Before planning, batches of 10 hands are shown to the model, which removes extremely implausible ones. Removed hands are redealt (up to 3 rounds) so shares stay at the weights. |
| Description rewrite (`sdf_pipeline/layer1_description_rewrite.py`) | After planning, one call per description checks plausibility, conformance to every dealt value and organisation naming. It may decline. Rewrites keep the original; a fix that would change a dealt value drops the plan. Defaults to `gpt-6.1-sol` (see Why). |
| Considerations selector (`sdf_pipeline/layer1_select_considerations.py`, `considerations_assignment.py`, `considerations_catalog.py`, `considerations_coverage.py`, `prompts/sdf/considerations.txt`) | The model scores every eligible entry of a moral and an epistemic list from 1 to 5 per document. A linear programme (scipy) then assigns entries within target counts, per-document caps and a score-weighted objective with a penalty for extra entries in a list. Result reaches drafting as `considerations_spec`. |
| Model calls (`shared/api.py`, `shared/openai_api.py`) | `call_claude` accepts a JSON schema for structured output. New OpenAI wrapper with cost-log records. `claude-sonnet-5` price updated. |
| Variables and prompts (`prompts/sdf/*.txt`) | `reasoning_featured` removed from `layer1.txt` and `layer2.txt`; `layer2.txt` takes `considerations_spec`. The no-welfare-stake resolution's weight is set to 0, with its former share spread proportionally over the other resolutions. The planner may name the AI's provider when the model is named. |
| Orchestration and reporting (`sdf_pipeline/run.py`, `compose_prompts.py`, `evals/`) | Layer 1 now runs deal, screen, plan, rewrite, select. Audit and report show screen removals, rewrite rates and considerations coverage. |
| Other | `config.yaml` knobs for each new stage, `scipy` in `requirements.txt`, and a script to rerun the selector on stored plans without new model calls. |

## Why

- The deck deals each variable independently, so hands like "no welfare stake" with "welfare is the central subject" were possible. Rules fix those by definition, without limiting what the screen may judge.
- Some dealt hands cannot be realistic however well they are written. Screening them before planning avoids paying to plan and draft them.
- Planned descriptions drifted from dealt values or contained implausible details. The rewrite repairs them before drafting.
- A random `reasoning_featured` value was forced into scenarios where it did not fit. Scoring fit first and assigning afterwards keeps the corpus near its target mix without forcing entries.
- The rewrite runs on a GPT model because Claude was far too lenient on plausibility when tested on 97 descriptions. `config.yaml` records the comparison and warns against switching without re-validating.
- Every new stage checkpoints per item. A malformed or truncated answer is not checkpointed, so `--resume` retries only that item.
- The considerations are assigned across the whole set, so layer 1 stops after checkpointing if any scoring call fails or any description check is held out; `--resume` finishes it before any document is drafted. The screen recovers from an interrupted write on resume without letting unscreened or duplicate hands through.

## How to review

- Start with `prompts/sdf/deal_constraints.txt`, then `shared/deal_constraints.py`. Check that each rule excludes something impossible by definition and nothing more.
- Needs judgement: the prompt files (`layer1_screen.txt`, `layer1_description_rewrite.txt`, `layer1_select_considerations.txt`), `prompts/sdf/considerations.txt` (entries and target shares) and the `sdf` section of `config.yaml` (defaults, caps, score weights).
- Read `considerations_assignment.py` for the objective and constraints.
- Mostly mechanical: value-ID edits in `variables.txt`, API wrappers, audit and report additions, and tests.
- The large new test files mirror the new modules one to one.

## Testing

The offline test suite passes (1448 tests, up from 1155 on `main`). It blocks network access, so it makes no API calls.

## Stack

PR 1 of 4, based on `main`. #179 builds on this branch, then #180 and #181 on top of it.
