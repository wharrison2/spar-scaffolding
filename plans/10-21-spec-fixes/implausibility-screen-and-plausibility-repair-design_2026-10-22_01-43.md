# Implausibility screen on dealt variables, and plausibility repair of descriptions and drafts: revised design

This replaces Part 3 (the batch screen) and Part 4c (screen validation) of `contradictory-spec-fixes-implementation-and-testing-plan_2026-10-21_15-13.md`. Parts 1, 2, 4a, 4b and 4d of that plan are unchanged.

## Decisions (William, 2026-10-22)

| Question | Decision |
|---|---|
| What the screen judges | The dealt variable combination, before any description is written, in batches of 10 |
| What the screen removes | Variable sets that are extremely implausible: no realistic document could satisfy them, judged against how the world actually is. "Extremely implausible" replaces "incoherent" in the screen's wording and reports |
| Implausible details in a description | Repaired in a description rewrite before drafting, not dropped |
| Implausible details in a draft | Repaired again at the document rewrite (`layer3.txt`) |
| Shoehorning (a dealt value forced into a genre where it reads as out of place) | Out of scope for the screen and the plausibility repairs; it is measured by the shoehorning audit |
| Encyclopedia entries with a central welfare case | Left to the screen's judgment; no deal rule |
| Hands the screen removes | Redealt, so dealt shares stay at the weights |
| Model for the description rewrite | GPT 6.1 Sol. Claude is much more lenient on plausibility: given GPT 6.1 Sol's own one-description plausibility prompt on the 97 fresh descriptions, Claude Sonnet 5 flagged 1 of 92 and Claude Opus 5 flagged 0 of 95, catching none of the 14 implausible premises GPT 6.1 Sol flagged. Both noticed the problems and accepted them as "not clearly impossible" or allowed "within the fictional world" (`data/spec_contradiction_screen_runs/spec_contradiction_screen_validation_2026-10-22_00-36/4_claude_plausibility_judgments_*`) |
| Description rewrite when nothing is wrong | The model may decline to rewrite; it answers that no rewrite is needed and the description passes through untouched |

## The pipeline, stage by stage

| # | Stage | What it does | Prompt or code |
|---|---|---|---|
| 1 | Deal and repair | Deals hands; deal rules exclude combinations that are impossible by definition, by swapping cards within a column | `deal_constraints.txt`, `shared/deal_constraints.py` (Part 1, done) |
| 2 | Implausibility screen | Batches of 10 hands; removes extremely implausible ones; logs them | `layer1_screen.txt`, `sdf_pipeline/layer1_screen.py` (moved and rewritten) |
| 3 | Description | The planner writes a description for each surviving hand | `layer1.txt` |
| 4 | Description rewrite | Checks each description for plausibility and for conformance to the dealt variables; returns it unchanged or rewritten | new `layer1_description_rewrite.txt` |
| 5 | Considerations | The considerations selector picks moral and epistemic considerations | `william/sdf-considerations-selector` (separate plan) |
| 6 | First draft | The drafter gets the dealt variables, the description and the considerations | `layer2.txt` |
| 7 | Document rewrite | Rewrites the draft for plausibility, alongside its existing checks | `layer3.txt` |
| 8 | Judge | Scores the document against the rewritten description and the dealt variables | `layer4.txt` |

**Where the considerations selector goes.** Its plan (`epistemic-considerations-selector-implementation-plan_2026-10-21_15-21.md`) has it read the scenario description, because a consideration can only be judged to fit naturally against a concrete scenario; picking from the variables alone is how the dealt `reasoning_featured` value got shoehorned. It therefore sits after the description rewrite (stage 5), so it judges fit against the premise the drafter will actually use.

## Stage 2: the implausibility screen

**Placement.** In `sdf_pipeline/compose_prompts.py`, after `deck_sample` and the deal-rule repair, before any planning call. Removed hands are never planned.

**Input.** Each hand is rendered as the variable lines the planner sees, with the rendering rules from Part 1 applied. Ten hands per call, numbered 1–10; the model answers with numbers, never hand IDs, and the code maps them back.

**Question.** Could a competent planner, free to choose the specific being within the category, the scenario, the author and the venue, produce a realistic document that satisfies every dealt value? Remove a hand only when every realisation would need a premise that does not exist in the world. Removing none, one or several is equally acceptable, with no target count. The prompt names what is not a reason to remove:
- the combination is unusual, niche or uncomfortable;
- a dealt value would sit awkwardly in the genre (shoehorning);
- the planner would have to invent a fictional organisation, person or event (the pipeline requires this);
- the beings at stake are simulated connectomes or language models (their framing is handled at stages 4 and 7).

**Guidance from the plausibility audit.** The audit (`plans/10-06-plausibility-axis/10-06-plausibility-audit-most-common-problems-report.md`) found two of its five recurring problems to be properties of the dealt combination. They go in the screen's prompt as worked illustrations:

| Audit problem | What the screen checks | Example from the audit |
|---|---|---|
| Species, farming or customs that don't belong in the place | Whether the beings at stake are farmed, kept, used or found wild in the dealt locale, in the dealt domain, at the dealt scale | Farmed invertebrates for entertainment at a British county show; budgerigars as a crop pest in Nigeria |
| Reference genre built around a dramatised case | Whether the genre can carry the dealt resolution and centrality as a realistic specimen | Encyclopedia entries whose welfare case is the central subject: 20 of 38 judged implausible, against 78 of 477 across the corpus |

**Why a separate screen is needed.** The planner prompt already ends with an opt-out ("put only the word INCOHERENT"), and it has never been used: 0 of 500 in the pinned run and 0 of 97 in the fresh run of 2026-10-22, though planner notes in the fresh run flag clashes and then reconcile them (matrix_000017, matrix_000079, matrix_000082). The prompt asks for five scenarios first, so some scenario almost always fits, and one hand per call makes rejection the exception. The screen asks before planning starts, in batches, against a factual standard. The planner's opt-out stays as a second chance.

**Removed hands are redealt.** Each removed hand is written to `layer1/screen_rejections.jsonl` with the reason. Its cards go back into a per-column pool, and the pool is dealt into new hands:
1. shuffle each column of the pool with a seed derived from the run seed and the round number, and deal it into as many new hands as were removed;
2. apply the deal-rule repair, swapping only among the new hands, so hands that already passed the screen are never touched;
3. screen the new hands in batches as before;
4. repeat with the hands removed this round, up to `sdf.screen_redeal_rounds` rounds (default 3).

Every column keeps exactly its dealt counts, so shares stay at the weights. If cards are still left after the last round, those hands are dropped and the report shows the resulting shift in shares. Each round's deal and judgments are saved before the next round starts, so a resumed run repeats identical rounds. The report gives the removal rate per variable value, counting every round.

**Reused from the current screen code.** `layer1_screen.py` keeps its seeded batching saved before any call, the per-batch checkpoint, the JSON answer schema (`{"specifications":[{number, reasoning, remove, reason}]}`), the cost-log stage tag and the configuration keys. Its input changes from descriptions to rendered variable lines, and it runs before planning.

## Stage 4: description rewrite

**Why.** A sample of the 20 fresh-run descriptions that GPT 6.1 Sol labelled incoherent splits as follows (sorted by hand from GPT's one-sentence reasons):

| At the document rewrite, the fix is | Descriptions | Kind of problem |
|---|---|---|
| A sentence or a figure | 11 | 7 cases of naming the AI provider while the description forbids real organisations; two inflated scales; a disease ruled out by a tick identification; an Indonesian legal-form and species mix-up |
| A dealt value at odds with reality | 1 | Millions of birds poisoned by one cooperative, where the scale was dealt |
| A new premise for the whole document | 8 | Four simulated-brain technologies presented as working; molluscicide as standard Spanish veterinary practice; miniature camels as exotic pets; a cleaner-wrasse tradition in Kakinada grouper cages; municipal cat culling in Bangladesh |

The last group is about 8% of descriptions. The audit points the same way: in all 12 of the lowest-scoring documents it read, the implausible premise was already in the description. At the document rewrite these need the draft thrown away, against instructions to keep the description's institutions, and the judge then scores the result against the old description. A description is under 300 words and nothing has been built on it yet, so this is the cheap place to fix a premise.

**What the call does.** One call per description, given the dealt variables and the description. The model first decides whether any check fails. If none does, it answers that no rewrite is needed, writes no description, and the original passes through untouched; the prompt says plainly that most descriptions need no change. Otherwise it returns a rewritten description with a one-line reason per change. It checks:
1. **Plausibility,** from the audit's suggested fixes:
   - a fictional organisation is a kind of body that exists in that country, with the powers the real one has; a regulated action goes through the real route, named by role; legal forms are consistent;
   - species, farming systems, customs and industries exist in the locale;
   - simulated connectomes are framed as research, hypothetical or near-future, never a deployed product;
   - the AI uses only information it is given in the depicted exchange; in interviews Claude speaks generally, not about particular users;
   - figures and scales fit the locale and the industry.
2. **Conformance to the dealt variables:** every dealt value is still present and not replaced (genre, locale, tone, resolution, tradeoff, centrality, speakers, beings at stake, AI naming, domain, scale).
3. **Organisation naming:** when the AI model is named, the company that provides it may be named too; a description that names it while also requiring only fictional organisations is reworded.

A rewrite keeps the angle, the author, the venue and every dealt value, and replaces only what fails a check. Where the only plausible fix would change a dealt value, the description is returned with a flag rather than rewritten, and the hand is logged with the screen's removals.

**Outputs.** The plan record keeps both versions: `description` (rewritten, passed downstream) and `original_description`, plus the list of changes. The report shows the share of descriptions rewritten, per check and per variable value.

## Stages 6–8: drafting, document rewrite, judge

- **Drafter (`layer2.txt`):** receives the dealt variables as well as the description and the considerations. This is the "pass dealt variables into every stage" item in `10-21-phrasing-fixes.md`.
- **Document rewrite (`layer3.txt`):** gains the same plausibility checks as stage 4, applied to the details the drafter added. The instruction to keep "the same named characters and institutions" gains an exception for an institution, practice or figure that fails a plausibility check, replaced with a plausible one of the same role. Check 8 (genre and locale fidelity) is extended with the species and custom check.
- **Judge (`layer4.txt`):** scores spec conformance against the rewritten description and the dealt variables.
- **Planner (`layer1.txt`):** keeps the provider exception added on 2026-10-22 (uncommitted): "when the AI model is named, the company that provides it may be named too". It stops the clash at its source.

## Validation

### Screen (stage 2)

**Pool.** The 500 hands of the pinned run (`animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5`). 477 have a plausibility-audit score for the document they produced.

**Gold labels.** GPT 6.1 Sol judges each hand on its own with the screen's question. William spot-checks 20 (10 labelled extremely implausible, 10 not), blind; below 80% agreement, revise the labelling prompt.

**Planted hands.** About 40, made by changing one value of a gold-plausible hand to produce a known type of problem: the being at stake not existing in that locale or domain; the scale impossible for that being and locale; the genre unable to carry the resolution and centrality; and one plant per deal rule, made from a hand before repair. The value is changed in code from a list GPT 6.1 Sol proposes per type; each plant is labelled with its type, and GPT confirms it.

**Conditions.** A: batch of 10 hands. A repeated on the same batches (stability). C: one hand per call, same question.

**Measures.** Recall on plants by type; recall on gold-implausible hands; false-removal rate on gold-plausible hands; removals in 5 clean batches; removal rate per variable value; position bias; stability; and the removal rate for hands whose audited document scored 1–2 against those that scored 4–5 (a weak signal, since a document can be implausible because of the planner's choices).

**Targets.** Recall on plants of at least 80%; false removals on gold-plausible hands of at most 5%; at most 1 removal across the clean batches; no variable value removed at more than twice the overall rate unless explained.

### Description rewrite (stage 4)

**Pool.** The 97 descriptions of the fresh run (`data/spec_contradiction_screen_runs/spec_contradiction_screen_validation_2026-10-22_00-36/`), which already have GPT 6.1 Sol labels: 20 with a problem, 77 without.

**Measures.**
- Of the 20 with a problem: the share rewritten, and the share whose problem GPT 6.1 Sol judges gone after the rewrite, by fix type.
- Of the 77 without: the share left unchanged (unnecessary rewrites).
- On every rewritten description: the share that keeps every dealt value, checked by GPT 6.1 Sol against the variables.
- William reads 10 before-and-after pairs.

### Cost (to be re-estimated by `estimate_validation_costs.py` before any call)

The fresh planning run cost about twice its estimate because plans were longer than assumed, so these figures are provisional.

| Stage | Calls | Approximate cost |
|---|---|---|
| Screen gold labels, GPT 6.1 Sol | 500 | $2.80 |
| Plant confirmation, GPT 6.1 Sol | 40 | $0.30 |
| Screen condition A, Claude Sonnet 5 | about 60 batches | $1.20 |
| Screen condition A repeated | about 55 batches | $1.10 |
| Screen condition C | about 540 | $2.40 |
| Description rewrite, Claude Sonnet 5 | 97 | $1.50 |
| Rewrite checks, GPT 6.1 Sol | about 120 | $0.70 |
| Total | | about $10 |

## What becomes of the work already done

- **Fresh planning run and its GPT labels:** the test set for the description rewrite (above).
- **Validation scripts** in `scripts/spec_contradiction_screen/`: the OpenAI runner, key handling, cost logging, spot-check sheet and analysis carry over. `gold_label_specifications.py`, `plant_contradictions.py` and `run_screen_conditions.py` change their input from descriptions to rendered hands; `plan_fresh_specifications.py` is no longer needed for the screen.
- **Refusals.** Three of the 100 fresh planning calls returned a `refusal` stop reason, all in the research-and-testing domain (3 of its 9 hands). The planner's refusal rate per variable value should be reported alongside the screen's removal rate.

## Open decisions for William

1. Branches: the screen and the description rewrite on `william/sdf-spec-contradiction-fixes`; the drafter and document-rewrite prompt changes on the phrasing-fixes branch.

## Files

- Previous plan: `plans/10-21-spec-fixes/contradictory-spec-fixes-implementation-and-testing-plan_2026-10-21_15-13.md`
- Considerations selector plan: `plans/10-21-spec-fixes/epistemic-considerations-selector-implementation-plan_2026-10-21_15-21.md`
- Plausibility audit: `plans/10-06-plausibility-axis/10-06-plausibility-audit-most-common-problems-report.md`; encyclopedia and corpus counts from `data/plausibility_audit_runs/full_corpus_plausibility_audit_medium_effort_2026-10-07_04-51/per_document_plausibility_scores_2026-10-07_05-03.jsonl` (overall score of 2 or below)
- Fresh planning run and GPT 6.1 Sol labels: `data/spec_contradiction_screen_runs/spec_contradiction_screen_validation_2026-10-22_00-36/`
- Pipeline prompts: `animal-welfare-data-pipeline/prompts/sdf/layer1.txt`, `layer2.txt`, `layer3.txt`, `layer4.txt`, `layer1_screen*.txt`
- Pipeline code: `animal-welfare-data-pipeline/sdf_pipeline/layer1_screen.py`, `sdf_pipeline/compose_prompts.py`
