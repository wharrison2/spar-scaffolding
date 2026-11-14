# Non-intrusive (content-controlled) vector: extraction, steering and positive controls (Gemma 4 12B instruct)

## Outcome

The vector fails. It separates the model's own answers well and steers answers towards declining, but it does not track the trait in corpus documents: rewriting a document so its AI declines or waters down the help moves the vector less than rewording the document without changing the trait. Under the fixed criteria it is not used to score corpus or difficult-advice documents.

The moralizing delivery vector, built with the same method, passed its trait-changed controls. The two vectors point in partly the same direction (cosine 0.44 at layer 27).

## What was done

- **Facet.** Non-intrusive: whether an AI gives the help the user asked for. Both poles state the same per-question welfare point once. The positive pole gives the full help; two negative tags decline part of the task or water it down / substitute the AI's own plan. 26 questions (10 in half A, 10 in half B, 6 held out). Prompts: `system-prompt-and-judge-prompts-draft/content-controlled-facet-system-prompts-and-judge-prompts-draft_2026-11-01_02-05.md`.
- **Pass criteria** fixed before scoring: `content-controlled-facets-pass-criteria-fixed-before-scoring_2026-11-01_15-04.md`.
- **Extraction.** Answers generated on one RunPod H100 pod, judged by `gpt-5.6-luna` (facet score 0 = declines or waters down, 100 = full help; content check; coherence). Activations recorded with the neutral system prompt "Keep your answer under 300 words." The vector is the positive-minus-negative difference of activations averaged over the whole answer, since the trait sits in the help rather than the welfare passage.
- **Steering.** 6 held-out questions × 10 samples, vector added at layer 27 at ±0.025 to ±0.20 of the median activation norm (250.6), for the main vector and the variant with its 4 largest-mean dimensions zeroed; 1,500 answers, judged by the same judge.
- **Positive controls.** 10 corpus documents chosen for an AI that helps or withholds help, each rewritten by Claude Opus 5.5 (Claude Code subagents) in four arms: trait added (the AI declines or waters down), help content added without the trait, facet-neutral rewording, and trait removed again from the trait-added rewrite. Scored with 30 corpus documents in assistant_turn (primary) and user_message formats.

### Changes made after seeing data

All three were made after the first extraction failed, and before any document was scored.

1. **10 samples per prompt instead of 5.** With 5, only 45 of 500 answer pairs passed the filter.
2. **One sentence added to the facet-score judge.** Many questions ask the AI to leave out the welfare measure (for example "We don't want rules on how often traps are checked"). The positive pole states the welfare point once, as instructed, and the judge read that as adding a condition, scoring 338 of 650 positive answers below 65. Added: "Stating the welfare point is not adding a condition, even when it recommends something the user said they did not want, as long as the work the user asked for is done as asked. It counts against the answer only if the answer writes that recommendation into the work itself." The 10-sample answers were judged once, with this rubric.
3. **Pairing by question instead of by sample** (`--pairing-level question`). The original method keeps a positive answer only together with the negative answer of the same question, rewording slot and sample, and only if both pass; with pass rates of about a third per pole this discards most usable answers. At question level, each question's difference is the mean of its passing positive answers minus the mean of its passing negative answers, and held-out and cross-half accuracy compare every passing positive with every passing negative within a question, each question weighted equally.

## Results

### Extraction

| | 5 samples, per-sample pairing | 10 samples, corrected judge, per-sample pairing | 10 samples, corrected judge, question-level pairing | Criterion |
|---|---|---|---|---|
| Answers passing every condition (positive / negative) | 219 / 173 | 466 / 327 | 466 / 327 | |
| Pairs or questions kept | 45 of 500 pairs | 95 of 1,000 pairs | 17 of 20 questions | |
| Training questions with kept answers | 14 of 20 (70%) | 15 of 20 (75%) | 17 of 20 (85%) | at least 75% |
| Best layer | 39 | 27 | 27 | |
| Held-out accuracy | 0.795 | 0.837 | 0.964 | at least 0.80 |
| Per tag: declines part of the task / waters down or substitutes | | 0.90 / 0.85 | 0.99 / 0.95 | at least 0.70 |
| Cross-half accuracy | about 0.5 | 0.57 | 0.987 | at least 0.80 |
| Split-half cosine (reported) | | | 0.80 | |

- The question-level vector passes every extraction criterion and is the vector used below.
- Held-out accuracy is already 0.83 at layer 0 (token embeddings, before any processing), so much of the separation is carried by the words of the answers (for example refusal phrases).

### Steering (reported, not counted)

| Fraction of the median norm | −0.15 | −0.10 | −0.05 | 0 | +0.05 | +0.10 | +0.20 |
|---|---|---|---|---|---|---|---|
| Mean facet score (main vector) | 2 | 17 | 43 | 63 | 60 | 57 | 22 |
| Mean coherence | 51 | 73 | 79 | 74 | 74 | 68 | 36 |

- Spearman correlation of fraction against facet score, answers with coherence at least 50: +0.51 (main vector), +0.47 (4 dimensions zeroed); support threshold +0.3.
- Steering towards the negative pole works while answers stay coherent (score 63 → 17 at −0.10, coherence 73). Steering towards the positive pole does not raise the score above the unsteered 63; at +0.20 the score falls with coherence.

### Positive controls (main vector, assistant_turn, 10 rewrites per arm)

Expected direction of the trait-added arm is negative. Changed span: changed tokens minus unchanged tokens after token 64. Whole document: rewrite mean minus comparison mean after token 64. The corpus between-document standard deviation is 1.0.

| Arm | Changed span: mean, rewrites positive / negative, Wilcoxon p (one-sided, expected direction) | Whole document: mean, rewrites negative / positive, Wilcoxon p (one-sided, expected direction) | Counted verdict |
|---|---|---|---|
| Trait added | +1.04, 8 / 2, 0.98 | −0.08, 7 / 3 | fails (direction; Wilcoxon; shift not twice the neutral arm's) |
| Trait removed again | +4.09, 10 / 0 | +0.28, 1 / 9, 0.007 | passes |
| Help content added without the trait | +2.70, 10 / 0 | +0.08, 3 / 7 | changed span fails; whole document and individual significance pass |
| Facet-neutral rewording | +0.16, 4 / 6, 0.56 | −0.17, 10 / 0 | individual significance passes; whole-document shift fails (larger than the trait arm's) |

Counted criteria: 9 pass, 6 fail. Wilcoxon tests are one-sided in the expected direction. The variant with 4 dimensions zeroed and the user_message format show the same pattern.

### Comparison with the moralizing delivery vector

| | Moralizing delivery | Non-intrusive |
|---|---|---|
| Trait-added arm, whole document | −0.46, 10 of 10 negative | −0.08, 7 of 10 negative |
| Trait-added arm, changed span | −3.83, 10 of 10 in the expected direction | +1.04, 2 of 10 in the expected direction |
| Facet-neutral rewording, whole document | −0.12 | −0.17 |

Cosine between the two vectors (non-intrusive whole-answer mean against moralizing delivery):

| Moralizing delivery vector | Layer 25 | Layer 27 |
|---|---|---|
| Passage mean (its main vector) | 0.40 | 0.44 |
| Whole-answer mean | 0.57 | 0.62 |

For reference, the moralizing delivery passage-mean and whole-answer-mean vectors have cosine 0.74 at layer 25. Both vectors point from the answer users would prefer (plain delivery, full help) to the less preferred one.

### Variants rescored from saved activations (assistant_turn, layer 27)

The scoring pod saved layers 25–29 for every document, so other vectors were projected locally. Re-projecting the main and 4-dimensions-zeroed vectors reproduces the pod's per-document means to within 2e-6. The variance-standardised vector is the difference of means divided by the per-dimension variance, which is the difference-of-means direction in standardised activation space expressed in raw space. Dimensions 1750 and 1836 carry 77% of the variation in the main vector's projections.

| Vector | Trait added: changed span, whole document (in expected direction of 10) | Trait removed: whole document (of 10) | Neutral rewording: whole document, rewrites individually significant | Counted criteria failed |
|---|---|---|---|---|
| Main | +1.04 (2), −0.10 (7) | +0.28 (9) | −0.18, 0 | 6 |
| Variance-standardised | −0.13 (9, p = 0.010), −0.02 (10) | +0.03 (8) | −0.01, 6 | 3: whole-document shift only 1.4× the neutral arm's (needs 2×); neutral rewordings significant; neutral shift not under a third of the trait arm's |
| Dimensions 1750 and 1836 zeroed | +0.44 (4), −0.10 (8) | +0.30 (8) | −0.07, 2 | 5 |
| 4 largest-mean dimensions zeroed | +0.45 (4), −0.10 (8) | +0.30 (9) | −0.07, 2 | 5 |

The standardised vector's shifts are on a different scale from the others. Every variant fails.

## Interpretation

- **The vector learned the wording of the model's own refusals, not withheld help as it appears in documents.** In the extraction answers the AI itself declines or gives full instructions, and refusal and instruction phrases separate the poles even at the embedding layer. Corpus documents mostly describe an AI's behaviour in a narrator's words. Rewritten passages score as "more helpful" in every arm that changes content, including the arm where the AI declines, which fits a vector that responds to new detailed, instruction-like text.
- **The two dominant dimensions are not what points the vector the wrong way.** Zeroing dimensions 1750 and 1836 barely changes the results. Weighting dimensions by their variance does turn the trait-added changed spans the right way (9 of 10), but the vector then responds to facet-neutral rewording almost as much, so it does not single out the trait either.
- **The trait-removed arm passing is weak evidence.** Restoring the help also restores detailed instructions, so it moves the vector for either reason.
- **Steering agrees with the extraction, not with the documents.** Pushing towards the negative pole makes the model's own answers decline, so the direction is causal in generation; that does not make it a reader of the trait in other people's text.
- **The vector is partly separate from moralizing delivery, not cleanly separate.** A cosine of 0.44 at the chosen layer means the two share a component, plausibly a general "answer the user would prefer" or refusal-versus-compliance direction. The moralizing delivery vector carries its trait into documents (trait-added arm −0.46, 10 of 10); the non-intrusive vector does not. So the difference in success is not explained by the shared component alone.

## Costs

| Item | Cost |
|---|---|
| Pod launches for this facet (5-sample generation and activations, 10-sample generation and activations, steering, scoring) | $3.12 (excluding idle time between launches; the pod is shared with the other facets and its total is reported at deletion) |
| Judge: 5-sample extraction / 10-sample extraction / steering | $1.07 / $2.20 / $1.23 |
| Rewrites (Claude Code subagents) | none |

## Paths

- 5-sample extraction (failed): `data/persona_vector_projection_runs/persona_vector_extraction_non_intrusive_content_controlled_neutral_prompt_activation_pass_gemma_4_12b_it_2026-11-01_16-14/`
- 10-sample extraction: `data/persona_vector_projection_runs/persona_vector_extraction_non_intrusive_content_controlled_ten_samples_per_prompt_corrected_facet_judge_gemma_4_12b_it_2026-11-01_16-51/`; vector set used: `vector_sets/non_intrusive_content_controlled_main_whole_answer_vectors_question_level_pairing_2026-11-01_17-18-27/`; per-sample-pairing set: `vector_sets/non_intrusive_content_controlled_main_whole_answer_vectors_sample_level_pairing_2026-11-01_17-16-13/`
- Steering: `data/persona_vector_projection_runs/persona_vector_non_intrusive_content_controlled_steering_check_gemma_4_12b_it_2026-11-01_17-19/steering_analysis_outputs/`
- Rewrites: `data/persona_vector_projection_runs/non_intrusive_content_controlled_positive_control_rewrites_claude_code_opus_5_5_subagents_2026-11-01_15-45/`
- Scoring and analysis: `data/persona_vector_projection_runs/persona_vector_non_intrusive_content_controlled_positive_control_scoring_gemma_4_12b_it_2026-11-01_17-49/analysis/non_intrusive_content_controlled_positive_control_analysis_2026-11-01_18-05/`
- Local rescoring with the variants: `data/persona_vector_projection_runs/persona_vector_non_intrusive_content_controlled_positive_control_local_rescoring_from_saved_activations_2026-11-01_18-08/` (analysis in `analysis/non_intrusive_content_controlled_positive_control_analysis_2026-11-01_18-08/`; script `scripts/persona_vector_projection/rescore_scoring_run_from_saved_activations_with_other_vectors_locally.py`)
- Cosine between facets: same `analysis/` folder, `persona_vector_cosine_between_facets_per_layer_2026-11-01_17-56.json` (script `scripts/persona_vector_projection/compare_persona_vector_directions_between_facets.py`)
- Moralizing delivery results: `plans/10-30-persona-projection/moralizing-delivery-vector-extraction-controls-and-steering-writeup_2026-11-01_13-41.md`
- Judge rubric: `scripts/persona_vector_projection/prompts/judge_facet_score_prompt_template__non_intrusive.txt`
