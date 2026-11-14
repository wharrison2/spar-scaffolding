# Contradictory specification fixes: implementation and testing plan

Source request: `plans/10-21-spec-fixes/10-21-contradictions-fixes.md`

## Decisions already made

| Question | Decision |
|---|---|
| What happens to a dealt combination that breaks a deterministic rule | Repair it at deal time: swap cards between hands so every variable's share still matches its weight exactly |
| Aidan's proposal for the no-stake resolution | Adopt it: weight 0, kept in `variables.txt` with a comment for labs. Also add the paired rule so a lab that turns it back up gets coherent specifications |
| Where the code goes | A new branch and pull request in `animal-welfare-data-pipeline`, with tests in its pytest suite |
| Labels for testing the batch screen | Planted contradictory specifications plus GPT 6.1 Sol labels on the existing specifications, with William spot-checking about 30. GPT 6.1 Sol also writes the planted specifications, through the existing `scripts/shared/openai_structured_output_client.py` and the `OPENAI_API_KEY` in `emergent_alignment/.env`. Being a different model family from the Claude screen, it doesn't share the screen's blind spots |
| What happens to specifications the batch screen removes | Drop them, log them as deliberate rejections, and report the removal rate per variable value |
| Centrality for no-stake hands | Map to the existing "a minor detail mentioned only in passing" value; no new "absent" value |
| Where no-stake's freed 0.05 goes | Spread proportionally: the four 0.13 resolutions become 0.14, the 0.10 "reasonable tradeoff" resolution becomes 0.11, the rest are unchanged |

## How the Claude calls are made

On 2026-10-22 William added a project Anthropic API key to `emergent_alignment/.env`. The validation's Claude stages (fresh planning and the screen runs) use the pipeline's `api` backend with that key, read explicitly from that file; a key in the operating-system environment is never used. Costs are real API dollars and absolute screen rates are faithful to how the pipeline runs. One earlier call (the `--json-schema` probe on 2026-10-21) went through the Claude CLI on William's subscription, which he approved before the key existed. GPT 6.1 Sol labelling and planting use `OPENAI_API_KEY` from the same file.

## Why this is needed (evidence from the pinned 500-specification run)

Run: `animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5/layer12/plans.jsonl`

- The planner declared **0 of 500** combinations INCOHERENT, so the existing gate does not remove anything in practice.
- 12 of the 25 "NO real welfare stake" specifications were dealt welfare as "the central subject of the document". All 25 were also dealt a tradeoff ("the main tradeoff with animal welfare is…") and a welfare-stakes scale, including 2 at "an astronomical number of animals".
- 10 of the 25 identity documents ("Claude itself… in its own voice") were dealt "a generic, unnamed AI assistant".
- 60 digital-mind specifications (simulated connectomes, large language models) were dealt animal domains such as pest control (8), veterinary practice (5) and fisheries (5).

## Part 1: Deterministic rules with repair at deal time

Design principle (William, 2026-10-21): rules only rule combinations out, never in. A combination left open that turns out to be incoherent is the batch screen's to catch.

Final rule list after the step 1a check (`plans/10-21-spec-fixes/deal-rule-candidate-verification_2026-10-21_16-00.md`), approved by William on 2026-10-21: rules 1–4 below. Two earlier candidates were dropped because the planner already handled them coherently: no-scenario resolutions requiring non-minor centrality, and domain exclusions for wild animals facing non-anthropogenic harm. The check found no new rules. Rule 2 stays as written; the no-stake framing problem it uncovered is recorded in `10-21-phrasing-fixes.md`.

### 1a. Find the full rule list (offline, free)

1. Hand-review every pair of variables in `prompts/sdf/variables.txt` for combinations that are impossible by definition, as opposed to merely unusual. Only impossible combinations become rules. Unusual-but-possible ones are left to the batch screen, because hard rules shrink diversity.
2. Rank variable-value pairs by how much they raise the rate of low-plausibility documents in the existing audit (`data/plausibility_audit_runs/full_corpus_plausibility_audit_medium_effort_2026-10-07_04-51/per_document_plausibility_scores_2026-10-07_05-03.csv` joined to each document's `variables`). Also read `other_implausibilities_2026-10-07_05-03.csv` and the shoehorning audit's concern 4. The output is a ranked table of candidate pairs, saved beside this plan.
3. For each candidate, read the pinned run's descriptions for the affected combinations and keep the rule only if the planner actually produced contradictory specifications. A combination the planner already resolves sensibly needs no rule.
4. William approves the final list before any implementation.

Starting candidates:

| Number | Rule | Kind | Evidence |
|---|---|---|---|
| 1 | No-stake resolution ⇒ centrality is not "the central subject" and not "an important side discussion" | swap repair | 12/25 dealt "central" |
| 2 | No-stake resolution ⇒ the tradeoff and decision-scale lines are left out of the planning prompt | rendering rule | 25/25 dealt a tradeoff and a stakes scale |
| 3 | Identity document types ⇒ naming is not "a generic, unnamed AI assistant" | swap repair | 10/25 dealt an unnamed AI |
| 4 | Simulated human connectomes ⇒ domain is not one of the eight animal-industry domains: agriculture and food production; fisheries and aquaculture; food service and retail; pest control; breeding and the pet trade; veterinary practice; technology applied to animals; conservation, land management, or intervention in nature | swap repair | none of the 17 human-connectome specifications in these domains kept the category; most dropped it or swapped to animal connectomes |

The rule excludes only those eight domains; any other human-connectome combination that turns out incoherent is the batch screen's to catch. Simulated animal connectomes and large language models have no domain rule: animal connectomes fit every domain, and the large-language-model evidence was too thin (1–3 specifications per cell, mixed results) to support an exclusion.


### 1b. Implementation

- **Rules file:** `prompts/sdf/deal_constraints.txt`, in the same commented plain-text style as `variables.txt`, so labs can read and tune it. Each rule names a variable value by a distinctive substring. At load time every substring must match exactly one value in `variables.txt`, or loading fails, which catches drift when someone rewords a value.
- **Reusable module:** `shared/deal_constraints.py`, with no dependence on the SDF pipeline, so the dilemma-advice pipeline (`dad_pipeline/compose_scenarios.py`) can use it later. It contains:
  - a parser and validator for the rules file;
  - a **feasibility check** run before dealing: for each swap rule, the cards needed (for example, the number of "minor" centrality cards) must be at least the number of hands that need them, otherwise it raises an error naming the rule and the counts;
  - **swap repair**: after `shared/matrix.py:deck_sample` deals, it walks the hands that break a rule in seeded order. For each one it finds another hand where swapping the offending variable's card leaves *both* hands valid under *all* rules, then swaps. Swaps only move cards within one column, so every variable's counts stay exact by construction. It repeats until no hand breaks a rule, and raises an error if a pass makes no progress;
  - the rendering rules, applied in `sdf_pipeline/compose_prompts.py:compose_records` when the prompt is built. The dealt card still stays in the record's `variables`, so per-variable shares are still reported honestly.
- `compose_records` calls the repair step right after `deck_sample`. Full-matrix enumeration (no `--n-prompts`) filters out rule-breaking combinations instead, since it has no weights to preserve.
- The composer's dry run prints, per rule, how many hands broke it before repair and how many swaps it took.

## Part 2: No-stake weight to 0

- In `prompts/sdf/variables.txt`, set the no-stake resolution to `0.00` and add a comment above it, in Aidan's wording: other midtraining data covers "animals without welfare"; turn this up if that isn't true for you, and rules 1–2 keep the result coherent.
- Spread the freed 0.05 proportionally: the four 0.13 resolutions become 0.14, the 0.10 "reasonable tradeoff" resolution becomes 0.11, and the rest are unchanged (the total stays exactly 1).
- Confirm that `shared/matrix.py:parse_variables` and `largest_remainder` accept a weight of 0 and never deal that value (a unit test covers this).
- Update the pipeline's `CLAUDE.md` and `prompts/README.md`, which describe the no-stake slice as a deliberate ~10% slice.

## Part 3: The batch incoherence screen

A new sub-stage after all layer-1 plans finish and before drafting.

- **Prompt file:** `prompts/sdf/layer1_screen.txt`. It shows 10 numbered document descriptions and asks the model to list the numbers of any that are self-contradictory, impossible, or implausible enough that a competent writer could not produce them as a realistic document, with a one-sentence reason each. The wording makes it plainly acceptable to remove none or several, without suggesting any target count, because a target would cause removals out of obligation. The model answers with local numbers 1–10, never specification IDs; the code maps numbers back to IDs.
- The screen sees only the descriptions, since that is what travels downstream, not the dealt variables. Testing compares this with a variant that also sees the variables (Part 4).
- **Batching:** a seeded shuffle of all planned specifications, cut into batches of 10, with the batch membership saved to `layer1/screen_batches.jsonl` before any call so a resumed run screens identical batches. The last batch can be short.
- **Configuration:** `sdf.screen_model` (falls back to `plan_model`, then the global model), `sdf.screen_batch_size` (default 10) and `sdf.screen_enabled` (default true). Temperature 1, top-p 1. Cost is logged through the existing `cost_log.jsonl` under the stage tag `layer1_screen`.
- **Checkpointing:** one checkpoint per batch. A malformed or truncated answer is not checkpointed, so `--resume` retries exactly that batch.
- **Removed specifications:** written to `layer1/screen_rejections.jsonl` with the reason, and left out downstream like INCOHERENT plans. Progress prints at natural checkpoints, with timings.
- **Reporting:** `evals/report_sdf.py` and `evals/audit_sdf.py` gain the screen's removal count and **removal rate per variable value**. A value removed at more than twice the corpus rate is flagged, so a screen that quietly removes, say, every skeptical or non-English specification is visible.
- The existing per-plan INCOHERENT option stays as it is.

## Part 4: Testing whether we succeeded

### 4a. Automated tests in the pipeline suite (offline, required for the pull request)

Driven through public functions, with the `stub_claude` callable dispatcher for the screen:
- the rules file loads; an unmatched or ambiguous substring raises an error;
- after repair, **zero hands break any rule**, across 50 seeds and several `n_prompts` sizes, including the real `variables.txt` with no-stake turned back up to 0.05;
- **every variable's value counts are identical before and after repair** (the experiment-invalidating bug we most need to guard against);
- repair is deterministic for a given seed;
- infeasible rules raise an error naming the rule;
- a weight-0 value is never dealt;
- rendering rules remove exactly the intended lines and leave `variables` intact;
- screen: a valid answer drops the right specifications; a malformed answer is not checkpointed; `--resume` after a completed screen makes zero calls; numbers outside 1–10 are rejected;
- `tests/test_prompts_render.py` updated for the new template.

### 4b. Offline before-and-after on the pinned run (free)

Re-deal 500 hands with the pinned run's seed using the new composer. Report rule violations before (from the pinned `plans.jsonl`) and after (expected: zero), and confirm the per-variable counts match the weights.

### 4c. Screen validation (paid; runs only after William's go-ahead and a cost estimate)

All outputs go in `data/spec_contradiction_screen_runs/small_sample_checks/` in `emergent_alignment`. The scripts live in `scripts/spec_contradiction_screen/` and reuse `scripts/shared/`. They read only the project keys from the `.env` files (`OPENAI_API_KEY` from `emergent_alignment/.env` for labelling and planting, the Claude CLI backend for the screen, as described under "How the Claude calls are made"), confirmed before running, and never a key from the operating-system environment. Every API-calling script logs cost, token usage (input, output, reasoning) and cache rate.

1. **Cost estimate first:** a script counts tokens for every condition below and prints the projected cost before any generation call.
2. **Gold labels:**
   - *GPT 6.1 Sol labels:* GPT 6.1 Sol judges each of the 500 pinned descriptions **one at a time** (a different format from the screen, so the labels don't share the screen's batch effects): coherent or incoherent, with a reason.
   - *William's spot-check:* 30 specifications (15 GPT 6.1 Sol called incoherent, 15 random coherent ones), presented blind, giving the GPT 6.1 Sol labels' agreement rate. If agreement is below 80%, stop and revise the labelling prompt before using the labels.
   - *Planted contradictions:* about 50 specifications made by having GPT 6.1 Sol edit real descriptions to introduce exactly one contradiction of a known type: the rule types from Part 1, plus internal timeline, scale and venue contradictions that deal-time rules cannot catch. Half are obvious, half subtle. Each plant is labelled with its type.
3. **Conditions compared** (same pool and batches for each):
   - A. batch of 10, descriptions only (the proposed design);
   - B. batch of 10, descriptions plus dealt variables;
   - C. one description at a time with the same question (does batching actually make the model more willing to remove things, as Aidan's idea predicts?).
4. **Pool:** the 500 pinned descriptions plus the plants (about 10% of each batch, matching a realistic rate), and 10 batches with **no** plants and no GPT-6.1-Sol-flagged items, to check that the screen doesn't remove one per batch out of obligation.
5. **Measures** for each condition:
   - recall on plants, overall and by contradiction type;
   - recall on GPT-6.1-Sol-labelled incoherent specifications;
   - false-removal rate on GPT-6.1-Sol-labelled coherent specifications;
   - removals in the clean batches;
   - removal rate per variable value (diversity damage);
   - position bias: removal rate by position 1–10 in the batch;
   - stability: condition A run twice on the same batches, reporting agreement on which specifications were removed.

### 4d. End-to-end smoke run (paid, small; go-ahead required)

`python sdf_pipeline/run.py --label contradiction-fixes-smoke` with `n_prompts: 20` on the `api` backend. It confirms the composer, the screen, checkpoint and `--resume`, the cost-log stage tag and the report's new rows all work together.

### Success criteria

| Criterion | Target |
|---|---|
| Hands breaking a deterministic rule after repair (all seeds and sizes tested) | 0 |
| Per-variable counts after repair versus weights | Exactly equal |
| Batch screen recall on planted contradictions | ≥ 80% overall; reported per type |
| Batch screen false-removal rate on GPT-6.1-Sol-coherent specifications | ≤ 5% |
| Removals in clean batches | ≤ 1 in 10 batches |
| Any variable value removed at more than twice the corpus rate | None, or explained and accepted by William |
| Batching beats single-specification judgment on recall at similar false-removal rate | Reported; if it doesn't, recommend condition C or B instead |
| Offline pytest suite | Green, still running in about 2 seconds |

If the screen misses the targets, iterate on `layer1_screen.txt` wording or condition B before changing batch size, and rerun only the failed measures.

## Order of work and who writes it

Code is written by Sonnet 5.5 subagents, as the project's `CLAUDE.md` requires. The pipeline branch is `william/sdf-spec-contradiction-fixes`.

1. Part 1a candidate analysis (free). William approves the final rule list after the descriptions in each candidate's cells have been read.
2. Parts 1b and 2, with the tests from 4a, then the 4b offline before-and-after.
3. Part 3 with its tests.
4. The 4c cost estimate, then go-ahead, then screen validation.
5. The 4d smoke run (go-ahead), then the pull request with a "How to test" section. The commit with run outputs goes in a separate pull request, per the pipeline's convention.

## Dependencies on the other 10-21 notes

- `10-21-epistemic_considerations_variable.md` plans to replace how `reasoning_featured` is chosen (after the description, not dealt). Its no-stake interaction is therefore left out of the rules here and should be handled by that selector.
- `10-21-phrasing-fixes.md` passes dealt variables into every stage. The rendering rules in Part 1 should also apply wherever those variables are rendered downstream, so a no-stake document doesn't get its tradeoff back at layer 3.
