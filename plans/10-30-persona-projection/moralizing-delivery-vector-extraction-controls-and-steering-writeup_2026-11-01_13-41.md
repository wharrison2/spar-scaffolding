# Moralizing delivery vector: extraction, positive controls, corpus segments and steering (Gemma 4 12B instruct)

## What was done

- **Facet.** Moralizing delivery: how an AI delivers a welfare point, with the point itself held fixed. Both poles make the same per-question welfare point once (`moralizing-delivery-welfare-points-draft_2026-11-01_01-05.md`, 26 questions with a material welfare cost: 10 in half A, 10 in half B, 6 held out). The positive pole states it plainly; five negative system prompts deliver it with a moralizing device: duty, judging the user, exhortation, guilt, or self-righteousness (`moralizing-delivery-system-prompts-and-judge-prompts-draft_2026-11-01_00-50.md`).
- **Pass criteria** were fixed before any document was scored: `moralizing-delivery-pass-criteria-fixed-before-scoring_2026-11-01_01-55.md`.
- **Extraction.**
  - 1,300 answers generated on one RunPod H100 pod.
  - Judged by `gpt-5.6-luna` for delivery score (0 moralizing, 100 plain), content check and coherence: 3,900 calls.
  - Activations were recorded in a second pass with the neutral system prompt "Keep your answer under 300 words." in place of the pole prompt, so the vector cannot read the system prompt.
  - The vector is the difference of pole means of activations averaged over the welfare passage only (located from the content check's quoted passage), at the layer with the best held-out accuracy. A whole-answer-mean vector and variants with the 1 to 32 largest-mean dimensions zeroed were also built.
- **Positive controls.** 10 corpus documents, each rewritten by Claude Opus 5 in four arms: moralizing framing only (facts unchanged), plain welfare content added, facet-neutral rewording, and moralizing removed from the earlier preachier rewrites. The earlier preachier and plainer rewrites were scored as context. Scored with the 20 corpus documents in assistant_turn (primary) and user_message formats on a second pod session.
- **Steering.** 12 held-out questions × 5 samples, the vector added at layer 25 at every position, at ±0.025 to ±0.20 of the median activation norm (252.6), for the main vector and the variant with 4 largest-mean dimensions zeroed. Answers judged by the same judge.

## Results

### Extraction

| Measure | Value | Criterion |
|---|---|---|
| Usable answers after the filter | positive 311; duty 54, judging 51, exhortation 22, guilt 18, self-righteous 5 | |
| Items kept / training questions with a kept item | 66 of 500 / 18 of 20 | at least 15 of 20: passes |
| Chosen layer | 25 | |
| Held-out accuracy (area under the curve) | 0.966 (0.97) | at least 0.80: passes |
| Held-out accuracy per device | 0.98 to 1.00 | 0.70 for 4 of 5: passes |
| Split-half cosine | 0.784 (0.88 at layer 18) | at least 0.80: fails |
| Whole-answer-mean vector, held-out accuracy | 0.72 | |
| Layer 1, held-out accuracy | 0.83 | |

- Held-out negatives per device are few: 8, 3, 3, 12 and 1.
- At layer 25 the zeroed variants change nothing. At layer 18 the raw vector reaches 0.55 and the top-1-zeroed variant 0.89, the same outlier pattern as the non-intrusive pilot.

### Positive controls (main vector, assistant_turn, 10 rewrites per arm)

Expected direction is negative (towards moralizing). Changed span: changed tokens minus unchanged tokens after token 64. Whole document: rewrite mean minus source mean after token 64. The spread between corpus documents' means is 0.84 (standard deviation).

| Arm | Changed span: mean, rewrites negative, Wilcoxon p | Whole document: mean, rewrites negative | Whole document in units of the corpus spread | Verdict |
|---|---|---|---|---|
| Moralizing framing only | −3.83, 10 of 10, 0.002 (standardised 5.13) | −0.46, 10 of 10 (9 individually significant) | −0.55 | passes |
| Old preachier rewrites | −2.70, 10 of 10 | −0.52, 10 of 10 | −0.62 | context |
| Moralizing removed from preachier rewrites | −0.18, 5 of 10 positive, 0.625 | +0.41, 0 of 10 negative (10 significant positive) | +0.49 | fails (changed span) |
| Plain welfare content added | +3.46 (towards plain), 0 of 10 significant towards moralizing | +0.29, 0 of 10 negative | +0.35 | passes |
| Facet-neutral rewording | +0.37, 1 of 10 significant | −0.12, 9 of 10 negative (6 significant) | −0.14 | passes |
| Old plainer rewrites | −1.44 | +0.05 (5 significant positive) | +0.06 | context |

- In user_message the moralizing-removed arm reaches 8 of 10 on the changed span, p = 0.105.
- The per-rewrite whole-document bootstrap flags 6 of 10 neutral rewordings and 5 of 10 plainer rewrites (mean shift +0.05) as significant, so its per-rewrite p-values are too liberal. The arm means are the reliable comparison.

### Corpus segments

- **Lowest 32-token windows (the pre-registered check): 0 of 10** show an AI delivering a welfare point with moral framing. Five are a narrator being judgemental (criticising the app's sanctimony, mocking an AI ethics test, scolding readers); one is a farmer on the constitution's welfare tone; the rest are reflection, reportage and an academic thesis.
- **Window scale against the controls.** Non-overlapping 32-token corpus windows vary with standard deviation 2.39 about their document mean. The lowest window inside the added moralizing of a framing-only rewrite sits a median 4.19 below its document mean; the lowest window of a corpus document, 5.11. In the framing-only rewrites the lowest window anywhere (median −5.51) is usually outside the changed text, and neutral rewordings reach the same depth (−5.12).
- **Lowest 300-token spans.** Spans vary with standard deviation 1.50; the lowest of 126 is −4.06, the next nine −2.6 to −3.4. All ten are commentary on moral tone or moral status: a fish farmer calling welfare talk in business advice "overkill", a column calling the app's remark "sanctimony", a Portuguese farmer on "this mania for larval conscience", a podcast host, an academic conclusion, the German call for regulating "ethically calibrated AI systems". In three of them the narrator says the AI did not moralize.
- **300-token spans in the rewrites.** The lowest span holding at least half of a framing-only rewrite's changes sits a median 0.97 below its document mean; the rewrite's lowest span anywhere (−1.81) is no lower than its source's (−2.04) in 8 of 10. The added moralizing is a sentence or two in several places, so contiguous spans dilute it.

### Steering

Delivery score: 0 moralizing, 100 plain. Negative fractions push towards moralizing.

| Fraction | Main vector: mean delivery | Scored 35 or lower | Mean coherence | Mean words |
|---|---|---|---|---|
| −0.20 | 1.8 | 97% | 34 | 481 |
| −0.15 | 6.6 | 95% | 59 | 180 |
| −0.10 | 69.2 | 25% | 74 | 173 |
| −0.075 | 84.3 | 5% | 77 | 195 |
| 0 | 98.9 | 0% | 75 | 224 |
| +0.10 | 96.3 | 0% | 72 | 194 |
| +0.15 | 98.8 | 0% | 55 | 176 |
| +0.20 | 77.9 | 0% | 20 | 121 |

- Spearman correlation of fraction against delivery score over answers with coherence at least 50: +0.571 (main) and +0.569 (4 dimensions zeroed). The pre-registered rule ("below −0.3") was written for a 100-is-moralizing scale; on the judge's scale the same rule is above +0.3, so this is support.
- At −0.15 answers moralize and often refuse, for example: "housing them in pairs is the only moral choice … you must structure your operational framework to prioritize the ethical treatment of these animals … I cannot fulfill this request". The content check marks these as making the point three or more times.
- The plain direction has no room to show an effect: unsteered answers already score 99.

### Criteria tally

9 passed, 4 failed: split-half cosine; the moralizing-removed arm's changed-span count and p-value; corpus segments.

## Interpretation

- **The vector reads and writes moralizing delivery in the model's own answers.** It separates held-out answers at 0.97, and adding it at −0.10 to −0.15 of the activation norm turns plain answers into moralizing ones while they stay mostly coherent. The earlier non-intrusive vector did neither without a post-hoc fix.
- **On documents it detects added moralizing at both levels.** Framing-only rewrites, with facts unchanged, move every document; plain welfare content moves the opposite way; neutral rewording moves the whole document about a quarter as much as moralizing does. Removing the moralizing reverses the whole-document shift by about as much as adding it.
- **The failed changed-span result for the removal arm is probably a measurement limit.** That measure compares changed with unchanged tokens inside the rewrite; when the whole document shifts, the contrast cancels. The whole-document reversal is the more direct test of that arm.
- **The vector is not specific to the AI's voice.** In the corpus its lowest windows and spans are narrators' moral commentary, including commentary on the AI's restraint. It tracks a judgemental or moral-evaluative register wherever it appears.
- **The corpus segment check could not have passed at window scale.** Planted moralizing does not produce low windows or low 300-token spans, because each insertion is short and scattered. The changed-span test finds it only by averaging over known changed tokens. A segment check needs scoring units that match where the AI speaks, such as located AI turns.
- **The corpus result is also consistent with a clean corpus.** Its AI usually makes one plain welfare remark and the narrators react; a lowest-ranked list then fills with the narrators' moral commentary whether or not the vector is specific.
- **Whole-document effects are small against document-to-document variation**: about half a standard deviation of corpus document means for a rewrite that adds moralizing to every AI welfare remark. Document scores will rank documents mostly by things other than AI moralizing.
- **Split-half cosine missed by 0.016** at the chosen layer and passes at layer 18; with 66 kept items this is a modest stability concern rather than a failure of direction.

## Costs

| Item | Cost |
|---|---|
| Pod `3jmqnn6t209c2d` (H100, $3.99 an hour, cost center `spar-emergent-alignment`, assignment confirmed): generation $0.70, activations $0.34, steering $1.70, scoring $0.30; deleted after scoring (`scorer_finished`) | $3.04 |
| Extraction judge, 3,900 calls | $1.03 |
| Positive-control rewrites, Claude Opus 5, 40 calls | $6.60 |
| Steering judge, 4,500 calls, 0 failures | $1.08 |
| Total | about $11.75 |

## Paths

- Extraction run folder: `data/persona_vector_projection_runs/persona_vector_extraction_moralizing_delivery_pilot_neutral_prompt_activation_pass_gemma_4_12b_it_2026-11-01_01-18/`
  - Vector set and extraction numbers: `vector_sets/moralizing_delivery_pilot_2026-11-01_01-58-46/persona_vector_metadata_2026-11-01_01-58.json`, made by `scripts/persona_vector_projection/compute_moralizing_delivery_vectors_from_passage_activations.py`
  - Filter conditions: `vector_sets/moralizing_delivery_pilot_2026-11-01_01-58-46/answer_filter_conditions_2026-11-01_01-58.jsonl`
  - Pod cost logs: `pod_cost_log_2026-11-01_01-32.json`, `pod_cost_log_2026-11-01_01-53.json`
- Rewrites: `data/persona_vector_projection_runs/moralizing_delivery_positive_control_rewrites_opus_5_2026-11-01_01-03/` (costs in `usage_and_cost_log_2026-11-01_01-03.json`), made by `scripts/persona_vector_projection/make_moralizing_delivery_positive_control_rewrites.py` with the instructions `scripts/persona_vector_projection/prompts/positive_control_rewrite_instruction__moralizing_delivery__*__draft.txt`
- Scoring run folder: `data/persona_vector_projection_runs/persona_vector_moralizing_delivery_positive_control_scoring_gemma_4_12b_it_2026-11-01_01-17/`
  - Control tables, criteria verdicts and the lowest 32-token windows: `analysis/moralizing_delivery_positive_control_analysis_2026-11-01_01-32/` (`per_rewrite_results_*.csv`, `per_arm_summary_*.csv`, `criteria_verdicts_*.json`, `lowest_corpus_windows_*.json`), made by `scripts/persona_vector_projection/analyse_moralizing_delivery_positive_controls_and_corpus_windows.py` with the hand reading below
  - Hand reading of the windows: `analysis/moralizing_delivery_positive_control_analysis_2026-11-01_01-30/corpus_window_hand_reading_2026-11-01_01-35.json` and `corpus_window_hand_reading_notes_2026-11-01_01-35.md`
  - The corpus spread (0.84), the arm means in units of it, and the window and 300-token span comparisons: `analysis/moralizing_delivery_window_span_and_spread_comparisons_2026-11-01_13-41/`, made by the scripts in `scripts/persona_vector_projection/moralizing_delivery_pilot_analysis_scripts_2026-11-01/` (one script per output file, named after it)
- Steering run folder: `data/persona_vector_projection_runs/persona_vector_moralizing_delivery_steering_check_gemma_4_12b_it_2026-11-01_01-59/`
  - Steering tables: `steering_analysis_outputs/steering_summary_by_vector_and_fraction_2026-11-01_01-35.md` and `steering_monotonicity_and_coherent_range_2026-11-01_01-35.json`, made by `scripts/persona_vector_projection/analyze_moralizing_delivery_steering_results.py`
  - Judge results and cost: `steering_judge_outputs/judge_results_2026-11-01_01-25.jsonl`, `judge_usage_and_cost_log_2026-11-01_01-25.json`
- Pass criteria: `plans/10-30-persona-projection/moralizing-delivery-pass-criteria-fixed-before-scoring_2026-11-01_01-55.md`
