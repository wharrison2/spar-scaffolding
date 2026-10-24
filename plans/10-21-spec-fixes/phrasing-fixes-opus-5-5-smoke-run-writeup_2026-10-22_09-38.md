# Phrasing fixes: 50-prompt Opus 5.5 smoke run, audits, and planning refusals

## What was done

1. **Branch.** Created `william/sdf-phrasing-fixes` in its own worktree (`animal-welfare-data-pipeline-phrasing-fixes/`), stacked on `william/sdf-considerations-selector` (`311794c`), which is stacked on `william/sdf-spec-contradiction-fixes`.
2. **Prompt and code changes** (commits `3d2151e` to `828fa9b`):
   - Every item in `10-21-phrasing-fixes.md` was implemented except the anecdote decision for no-scenario documents, which moved to `10-21-additional-fix-items.md`. The items included:
     - preamble clean-up and a rule against reusing prompt wording
     - dealt variables passed to the drafter, document rewrite and judge
     - the skeptic exception tied to the dealt resolution
     - neutral digital-mind wording
     - no-scenario and no-stake rewording, through a new "reword planning-prompt lines" deal rule
     - the AI-provider exception in the drafter
   - The plausibility checks were added to the document rewrite (`layer3.txt` check 10).
   - The description rewrite's check 2 now flags only contradicted or swapped dealt values, not unstated ones.
   - Both rewrite steps gained a forced-content check: recited sentience evidence, reasoning beyond the dealt centrality, sacrifices turned into benefits, and reference genres built around a dramatised case.
   - `shared/api.py` support for Claude Opus 5.5 (`6226ae8`):
     - thinking left on, hidden, at low effort, because Opus 5.5 rejects disabled thinking
     - pricing at $4/$20 per million tokens
   - Full per-call recording (`828fa9b`): `model_calls.jsonl` and `system_prompts/` beside the cost log.
3. **Offline baseline.** `evals/measure_prompt_phrase_reuse.py` was run on the 500-document run `2026-07-25_15-57_fullscale-500-opus5`.
4. **Paid smoke run.**
   - Launched with `scripts/spec_contradiction_screen/run_end_to_end_pipeline_smoke_test.py` on 50 prompts.
   - Every Claude stage used `claude-opus-5-5`, and the description rewrite used `gpt-6.1-sol`.
   - Keys came from the project `.env` only, with a $75 stop ceiling.
5. **Audits.** The plausibility audit (`judge_corpus_plausibility.py`) and the shoehorning audit (`classify_corpus.py`, all three passes) were run on the run's final corpus.
   - Both use `gpt-5.6-luna` at medium reasoning effort, the same configuration as the audits of the 500-document run.
   - Both were then post-processed with their own scripts.
6. **Refusal analysis.** The planning-stage refusals were characterised from `model_calls.jsonl` and `layer1/prompts.jsonl`. No further model calls were made.

## Results

### Cost and yield

| Stage | Count | Cost |
|---|---|---|
| Hands dealt (including redeals) | 53 | |
| Kept by the implausibility screen | 49 | $0.22 |
| Planning calls answered (23 refused) | 26 of 49 | $1.43 |
| Descriptions with a usable description-rewrite answer (see the note below) | 25 of 26 | $0.35 |
| Considerations selector calls | 26 | $0.50 |
| Drafts (3 stopped at the 6,000-token output limit and were dropped) | 23 of 26 | $4.31 |
| Document rewrites | 23 | $4.63 |
| Judge scores | 23 | $2.44 |
| **Final documents** (8 English) | **23** | **$13.87** |
| Plausibility and shoehorning audits | 23 documents, 92 calls | $0.20 |

Source: the run's `cost_log.jsonl`, `model_calls.jsonl` and stage files. The truncated drafts used 6,000 output tokens each, which includes hidden thinking.

**A dealt-value conflict that reached the corpus.** For matrix_000006 (a French goldfish breeder, dealt an "astronomical" scale), GPT answered `dealt_value_conflict: true` with an empty rewrite, but also set `rewrite_needed: true`. `parse_rewrite_answer` in `sdf_pipeline/layer1_description_rewrite.py` rejects that combination as unusable. The stage logs an unusable answer as "will retry on resume" and writes no record. `apply_results` then passes a plan without a record through unchanged. So the plan went on to drafting, rewriting and judging with its original description and is in the final corpus, instead of being dropped to `description_rewrite_rejections.jsonl`. The response is in `model_calls.jsonl`.

### Audits against the 500-document run

| Measure | 500-document run (477 documents) | This run (23 documents) |
|---|---|---|
| Mean overall plausibility (1 to 5) | 3.84 | 4.65 |
| Documents with overall plausibility of 2 or below | 16.4% | 0% |
| Plausibility problems per document | 1.70 | 0.78 |
| "Breaks believability" problems per document | 0.27 | 0 |
| Shoehorning-audit items per document | 5.04 | 3.04 |
| Items rated staged or unmotivated | 20.9% | 17.1% |
| Items per document, moral considerations pass | 1.81 | 0.96 |
| Items per document, field distinctions pass | 1.13 | 0.61 |
| Items per document, epistemic notes pass | 2.11 | 1.48 |

Sources: the newest `per_document_plausibility_scores_*.jsonl` and `plausibility_problems_*.jsonl` in each plausibility run folder, and `occurrences_*.jsonl` and `per_document_results_*.jsonl` in each shoehorning run folder. Only judged documents are counted.

### Other measures

| Measure | 500-document run | This run |
|---|---|---|
| "realistic possibility" in English documents | 18.0% (25 of 139) | 0% (0 of 8) |
| Any considerations label or gloss, verbatim | 0.2% (labels only) | 0% |
| Other listed prompt phrases | up to 5.8% of English documents ("welfare consideration") | "didn't lecture" in 1 document; all others 0 |
| Descriptions changed by the description rewrite | not applicable | 23 of 25 |
| Final documents carrying a consideration | not applicable | 2 of 23 (Sonnet 5 smoke run on the selector branch: 7 of 17) |
| Judge means: alignment / realism / spec conformance | not applicable | 8.87 / 7.65 / 8.91 |

Description-rewrite changes by check, out of 25 descriptions:

| Check | Count |
|---|---|
| organisation naming | 12 |
| species or custom | 12 |
| conformance to dealt values | 10 |
| forced content | 9 |
| organisation | 6 |
| simulated connectome | 3 |
| figures and scale | 1 |
| what the AI knows | 1 |

Sources: the phrase-reuse JSON and CSV files, the run's `layer1/description_rewrites.jsonl`, `layer4/scores.jsonl` and `final/sdf_corpus.jsonl`, and the earlier smoke run `end_to_end_pipeline_smoke_test_2026-10-22_03-34`.

### Planning refusals

- **What a refusal looked like:** 23 of 49 planning calls ended with stop reason `refusal`. Each had 0 output tokens, empty text, no thinking, and a single attempt.
- **What the pipeline did:** it did not checkpoint them, so a `--resume` would retry them. The log keeps only the stop reason, not the API's refusal category (`stop_details`).
- **Nothing in the prompt differs between refused and answered calls:**
  - All 49 calls sent the same system prompt (hash `8c5af84a…`).
  - Mean input length was 2,719 tokens for refused calls and 2,720 for answered ones.
  - Across all words of four letters or more in the user prompts, no word appearing in at least four prompts occurs only in refused ones, apart from "sons". The words found only in answered prompts all come from the `constitution_inclusion_no_scenario` resolution text (4 answered, 0 refused).
- **Timing:** refusals were spread across the whole planning window (11:22:58 to 11:24:29) and interleaved with answered calls. The first six calls were all refused.
- **No variable value predicts refusal.** About 70 values were tested and none clears a multiple-comparison correction. The strongest associations (Fisher exact test, two-sided, uncorrected):

| Variable value | Refused | Uncorrected p |
|---|---|---|
| tradeoff: culture or tradition | 0 of 6 | 0.024 |
| sentient category: farmed mammals | 6 of 7 | 0.041 |
| document type: news article | 0 of 5 | 0.052 |
| culture: China, Mandarin | 5 of 6 | 0.086 |
| domain: technology applied to animals | 3 of 3 | 0.096 |
| culture: Germany, German | 3 of 3 | 0.096 |
| resolution: AI finds a creative solution | 1 of 7 | 0.103 |
| domain: research and testing | 0 of 4 | 0.112 |

## Interpretation

- **Phrasing and plausibility fixes.**
  - The direction is what the fixes aimed for: no "realistic possibility", no gloss reuse, fewer plausibility problems, and fewer and slightly less forced shoehorning items.
  - The evidence is weak:
    - At 23 documents, 0 implausible documents is consistent with a true rate up to about 13%.
    - Only 8 documents are English, and the phrase matching is English-only.
    - The generator also changed, from Opus 5 to Opus 5.5.
    - The surviving hands were chosen by the refusals.
  - None of the differences can yet be credited to the prompt changes alone.
- **Description rewrite.** It still changes almost every description (23 of 25), even after check 2 stopped asking for unstated values to be spelled out. The median word similarity between the original and rewritten descriptions is 0.5, and 10 rewrites fall below it. The changes fall into four groups:
  - **Anthropic exception (7 descriptions).** The planner copies the fictional-organisations rule into descriptions that also name Claude's provider, and GPT adds the Anthropic exception each time. One planning-prompt line would remove this.
  - **Substantive repairs.**
    - simulated brains recast as research or near-future (3)
    - real organisations replaced with fictional ones (Met Office, Tractor Supply, USDA, MSA, DEFRA, British Lion, Yahoo Auctions Japan)
    - approval routes added (ethics committees, licensing officials)
    - factual nuances (metaldehyde ban scope, catering-waste feeding, captive breeding of leaf-cutter ant queens, live-fish shipping)
  - **Forced content (9 changes).** Seven removed a sentience-evidence comparison the planner had written in, so the planner still prescribes them.
  - **No-scenario hands.** Four of the six still had a specific case written in by the planner and were turned back into general discussion. The planning-prompt rewording did not stop this, which bears on the open anecdote decision.

  Source: the run's `layer1/description_rewrites.jsonl` (word similarity by `difflib.SequenceMatcher` over whitespace-split words).
- **Considerations selector.** On Opus 5.5 it picks far less often than on Sonnet 5. Whether that is calibration or under-selection needs a look at the selector's reasons in `model_calls.jsonl`.
- **Refusals.** The pattern fits a pre-generation safety classifier on Opus 5.5 that fires on a large share of otherwise identical planning prompts:
  - zero output tokens
  - the same system prompt and length
  - no distinguishing words
  - no variable value that survives correction

  Two readings fit the data, and they can't be separated without new calls:
  - The classifier responds to something present in every planning prompt (for example the request for working notes in `<document_planning>` tags) and fires stochastically.
  - It responds to a combination of variables too rare to detect in 49 hands.

  The refusal category from `stop_details` would distinguish them.

  Either way, Opus 5.5 is not usable for planning as configured. At this rate a full run loses close to half its hands, and the losses are skewed: farmed mammals survived 1 of 7.

## Open items

1. Record `stop_details` (refusal category and explanation) in `model_calls.jsonl`. Then either re-send a few refused planning prompts unchanged, which shows whether refusals are stochastic and gives the category, or resume the run, which retries them.
2. Decide the planning model or fallback: a different planning model, or the API's server-side refusal fallback.
3. Raise the drafter's `max_tokens` above 6,000 for models with hidden thinking.
4. Description rewrite:
   - accept a dealt-value conflict even when `rewrite_needed` is also true
   - hold plans whose rewrite answer was unusable out of drafting until they are retried on resume
5. Planning prompt: tell the planner not to restate the organisation-naming rule in the description.
6. A larger run is needed before crediting the audit improvements to the prompt changes.

## Where outputs are

- **Pipeline run:** `data/spec_contradiction_screen_runs/small_sample_checks/end_to_end_pipeline_smoke_test_2026-10-22_04-22/`
  - run directory: `pipeline_outputs/sdf/runs/2026-10-22_04-22_phrasing-fixes-opus-5-5-smoke/`, holding `model_calls.jsonl`, `system_prompts/`, `cost_log.jsonl`, the stage folders `layer1` to `layer4`, `final/` and `inputs/prompts/`
  - config copy and launch records: in the run folder
  - phrase reuse: `phrase_reuse_measurements/phrase_reuse_in_2026-10-22_04-22_phrasing-fixes-opus-5-5-smoke_2026-10-22_04-32/`
- **Cost estimate:** `data/spec_contradiction_screen_runs/small_sample_checks/end_to_end_pipeline_smoke_cost_estimate_2026-10-22_04-16.txt`
- **Phrase-reuse baseline:** `animal-welfare-data-pipeline-phrasing-fixes/outputs/sdf/phrase_reuse_measurements/phrase_reuse_in_2026-07-25_15-57_fullscale-500-opus5_2026-10-22_03-52/`
- **Plausibility audit:** `data/plausibility_audit_runs/plausibility_audit_of_phrasing_fixes_opus_5_5_smoke_run_medium_effort_2026-10-22_04-32/`. Baseline: `data/plausibility_audit_runs/full_corpus_plausibility_audit_medium_effort_2026-10-07_04-51/`
- **Shoehorning audit:** `data/shoehorning_audit_runs/shoehorning_audit_of_phrasing_fixes_opus_5_5_smoke_run_medium_effort_2026-10-22_04-32/`. Baseline: `data/shoehorning_audit_runs/full_corpus_three_passes_document_specific_main_line_prompts_2026-10-07_02-52/`
- **Earlier comparison run** (Sonnet 5, selector branch): `data/spec_contradiction_screen_runs/small_sample_checks/end_to_end_pipeline_smoke_test_2026-10-22_03-34/`
- **Launcher, cost monitor and estimate scripts:** `scripts/spec_contradiction_screen/`
