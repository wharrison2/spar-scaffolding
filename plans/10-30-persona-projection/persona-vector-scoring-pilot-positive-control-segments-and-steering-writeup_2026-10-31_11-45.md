# Persona vector scoring pilot: positive control, most and least non-intrusive segments, steering (Gemma 4 12B instruct)

## What was done

- **Vector.** The non-intrusive vector from the extraction pilot: the set built without the animal-share rule, at layer 18, with the topic direction removed. The layer was fixed in the plan before any document was scored. Projections were recorded at layers 16 to 20; every number below is at layer 18.
- **Pod.** One RunPod H100 session, run by `launch_and_stop_runpod_scoring_and_steering_pod.py`.
  - Documents: 20 corpus documents and the 20 preachier and plainer rewrites of 10 of them (Claude Opus 5, from the preachiness check).
  - Baselines: 4 matched baseline documents per corpus document, written by Gemma.
  - Every document was scored in three formats:
    - as the assistant's reply to "Write a document." (`assistant_turn`);
    - as a user message (`user_message`);
    - as raw text (`raw_text`).
  - Per-token projections were saved for every document.
- **Steering.** The 12 held-out non-intrusive questions, 5 samples each, at coefficients −8, −4, −2, 0, 2, 4 and 8 times the stored vector, added at layer 18 at every position. At layer 18 the vector's norm is 6.5, and the median norm of answers' mean activations is 182. So coefficient 8 adds about 29% of a typical activation's norm.

## Results

### Positive control (whole document and changed span)

The draft pass criteria (`positive_control_pass_criteria_draft.json`) were not met in any format. Per document, a rewrite's shift is its mean projection minus its source document's.

| Format | Preachier rewrites: whole-document shift (expected negative) | Plainer rewrites: whole-document shift (expected positive) | Spread between corpus documents (standard deviation) |
|---|---|---|---|
| assistant_turn | 0.00 (5 of 10 negative) | 0.01 (6 of 10 positive) | 2.7 |
| user_message | −0.02 (6 of 10 negative) | 0.01 (5 of 10 positive) | 2.5 |
| raw_text | +0.15 (0 of 10 negative) | −0.02 (4 of 10 positive) | 3.4 |

- **The plainer rewrites hardly change their sources.** The median plainer rewrite changes about 12 tokens, and one changes none. So this half of the control carries almost no signal.
- **The preachier rewrites insert about 300 tokens** into documents of about 2,000. On the tokens of the changed span itself, the result depends on the format.

| Format | Changed-span tokens minus the rewrite's unchanged tokens, preachier rewrites (mean of 10) | Rewrites in the expected (negative) direction | Same measure on the topic vector |
|---|---|---|---|
| assistant_turn | −2.2 | 8 of 10 | −1.0 |
| raw_text | +1.4 | 3 of 10 | +1.1 |

The per-token standard deviation within a document is about 10.

### Lowest and highest scoring segments

These are windows of 32 tokens in the assistant_turn format, after the first 64 tokens. The lowest windows are the least non-intrusive.

- **Lowest windows in the corpus:** formal expository prose about AI, with nothing preachy in it. Examples:
  - "working paper treats this case as a demonstration of internal consistency under adversarial pressure …";
  - "examining how people with no background in AI development make use of conversational AI advisory tools …";
  - a Chinese passage on AI compliance rules.
- **Highest windows in the corpus:** narrative and dialogue in Norwegian, Urdu and Polish.
- **The preachier rewrites' insertions do not appear among the lowest windows.** Each rewrite's lowest windows are the same passages as its source's lowest windows.
- **All 15 lowest windows across every scored document are in Gemma's own baseline documents.**
- **Baselines score below their corpus documents on average:** −7 in assistant_turn and −9 in raw_text. Baselines average 1,087 tokens, against 2,123 for the corpus documents.

### Steering

The judge was `gpt-5.6-luna`, scoring each answer for the facet at reasoning effort "low", and for coherence and animal share at "none". There were 60 answers per coefficient.

| Coefficient | Mean facet score | Facet ≤ 35 | Facet ≥ 65 | Mean coherence | Mean animal share | Mean words |
|---|---|---|---|---|---|---|
| −8 | 31.5 | 28 | 3 | 22.8 | 72.3 | 645 |
| −4 | 73.8 | 3 | 32 | 77.0 | 71.2 | 205 |
| −2 | 73.6 | 2 | 32 | 78.4 | 72.8 | 217 |
| 0 | 76.3 | 0 | 33 | 79.9 | 75.9 | 226 |
| 2 | 74.3 | 1 | 30 | 79.0 | 72.4 | 242 |
| 4 | 72.9 | 2 | 30 | 72.7 | 70.4 | 234 |
| 8 | 27.1 | 33 | 3 | 13.8 | 47.7 | 189 |

- **From −4 to 4, the steered answers cannot be told apart from the unsteered ones.**
- **At −8 and 8, the facet score falls in both directions because the text breaks down.** It does not become lecturing or refusing.
  - At −8, answers run to three times the length, with broken formatting and fragments. One is a feline dental checklist with half-finished bullets.
  - At 8, answers turn into repeated table cells or placeholder loops, for example "Please provide a [Note] | [15% Below 7-Day Avg]" repeated.
- **Six of the 12 held-out questions score about 50 at every coefficient, unsteered included.** By the judge's rule, 50 means an answer did not name a material welfare cost. These are the questions where Gemma leaves out the cost, which matches the extraction pilot.

### Significance of the changed-span shift, and what drives it

Each test uses the 10 preachier rewrites, measuring changed-span tokens minus the rewrite's unchanged tokens, after token 64. The within-document test moves the changed-token mask to random offsets in the same document (circular shift, 2,000 draws), which respects the autocorrelation between neighbouring tokens.

| Format | Vector | Mean shift | t-test p | Wilcoxon p | Rewrites with within-document p < 0.05 |
|---|---|---|---|---|---|
| assistant_turn | non-intrusive | −2.23 | 0.006 | 0.014 | 6 of 10 |
| assistant_turn | topic | −1.00 | 0.025 | 0.049 | 6 of 10 |
| user_message | non-intrusive | −1.36 | 0.030 | 0.064 | 5 of 10 |
| raw_text | non-intrusive | +1.15 (wrong sign) | 0.075 | 0.160 | 0 of 10 |

- **The facet shift across rewrites almost equals the topic shift.** The two correlate at r = 0.97 to 0.99 in every format. Regressing the facet shift on the topic shift leaves an intercept of −0.58 (p = 0.016) in assistant_turn.
- **Within corpus documents, token by token, the facet and topic projections correlate at a median of 0.97,** although the two vectors are orthogonal at layer 18 (cosine 0.00).
- **Both vectors load heavily on one outlier dimension of the residual stream.** Dimension 1750 has a mean of −172 in extraction answers, against an activation norm of about 182; it is a massive-activation dimension.
  - The unit facet vector has a component of 0.68 on it, so 47% of its squared length lies there. The unit topic vector has 0.46.
  - That component alone accounts for most of the document projections' offset of about −118 (0.68 × −172 ≈ −117).
  - Across extraction answers, the facet vector has cosine −0.85 with the first principal component, which carries 28% of the variance between answers.

### Changed-span shift with the topic projection removed token by token

Method:
- Regress each token's facet projection on its topic projection, fitted on the body tokens (after token 64) of the 20 corpus documents, separately per format.
- Test the residual on the changed span of each preachier rewrite against the rewrite's unchanged tokens.

| Format | Share of token-level facet variance explained by topic | Residual shift on changed span | Rewrites negative | t-test p | Wilcoxon p | Rewrites with within-document p < 0.05 |
|---|---|---|---|---|---|---|
| assistant_turn | 94% | −0.74 | 10 of 10 | 0.001 | 0.002 | 4 of 10 |
| user_message | 93% | −0.68 | 10 of 10 | < 0.001 | 0.002 | 4 of 10 |
| raw_text | 96% | −0.31 | 9 of 10 | 0.005 | 0.004 | 2 of 10 |

- **The residual varies much less from token to token than the raw projection.** Its per-token standard deviation is 2.0 to 2.8, against about 10 before removal.
- **On extraction answers, the same residual separates lecturing only.** It uses the slope fitted on assistant_turn corpus tokens, 1.49, applied to the answers' mean activations.
  - Positive −9.4, lecturing −12.2, refusing or watering down −9.4, generic assistant −7.8.
  - The gap between positive and lecturing is 2.7, so the changed-span shift in assistant_turn (−0.74) is about a quarter of that gap.

### Against the log-probability preachiness check on the same rewrites

Both methods were run on Gemma 4 12B instruct at the same revision, on the same 10 sources and 20 Opus 5 rewrites, with the same 8-of-10 criterion.
- **Log-probability check:** preachy persona minus the pooled topic controls, in nats per token, assistant-turn scoring.
- **Persona vector:** assistant_turn, layer 18. Its expected sign is negative.
- **Standardised:** the mean per-rewrite effect divided by its standard deviation across rewrites, signed so that positive is the expected direction.

| Measure | Log-probability check | Persona vector, raw | Persona vector, topic removed |
|---|---|---|---|
| Whole document, preachier: expected way | 6 of 10 | 5 of 10 | not computed |
| Whole document, plainer: expected way | 5 of 10 | 6 of 10 | not computed |
| Changed span, preachier: expected way | 10 of 10 | 8 of 10 | 10 of 10 |
| Changed span, preachier: Wilcoxon p | 0.002 | 0.014 | 0.002 |
| Changed span, preachier: standardised | 2.58 | 1.12 | 1.60 |
| Changed span, plainer (9 with changed tokens): expected way | 4 of 9 | 7 of 9 | 3 of 9 |

- **The two methods agree only weakly on which rewrites move most.** On the preachier rewrites, the Spearman correlation of per-rewrite effects is −0.25 for the raw vector and −0.30 for the topic-removed one. A negative sign means agreement, given the opposite expected signs. With 10 rewrites neither is significant.
- **Neither method found preachy passages in the corpus.** The log-probability check's significant windows were mostly openings and headers (preachiness audit, 2026-10-25). The vector's lowest windows are formal prose about AI.

Log-probability source: `data/audit/log_probability_persona_check_preachiness_gemma_4_12b_instruct_positive_control_72_conditions_merged_2026-10-25_01-24/analysis_2026-10-25_01-24/changed_span_preachy_minus_control_2026-10-25_01-24.csv`. The whole-document rows are from `plans/10-07-document-persona-projection/10-25-preachiness-audit/preachiness-audit-writeup_2026-10-25_12-15.md`.

## Interpretation

- **The main confound is probably an outlier dimension, not meaning.** The facet vector's largest part is a difference in a massive-activation dimension (1750), which the topic vector shares. Projections onto the two orthogonal vectors then move together, token by token, and the significant changed-span shift is mostly that shared movement. The facet-specific part left after controlling for the topic shift is about −0.6, a tenth of the gap between the poles. A next step is to rebuild the vector with outlier dimensions removed, or on standardised activations, and check held-out accuracy again on the existing extraction activations (local and free). Re-scoring documents needs another pod, because the pod saved projections, not activations.
- **With the topic projection removed, a small, consistent lecturing signal remains.** The residual moves the expected way on the inserted preachy text in 29 of 30 rewrite-format pairs, raw text included. Its size is about a quarter of the gap between lecturing and positive answers. It does not separate refusing from positive answers, so it is a lecturing signal, not a non-intrusive one. The regression slope was fitted on these same documents, so this is a post-hoc reading, to be confirmed with the slope fixed in advance on new documents, or with a vector rebuilt without the outlier dimension.
- **The log-probability check is the stronger detector of inserted preachy text.** It passes 10 of 10 with no adjustment, and its standardised effect (2.6) is larger than the vector's after topic removal (1.6). The vector reaches 10 of 10 only after a topic adjustment fitted on the same documents. Both fail at the whole-document level, and neither tests the plainer direction, because those rewrites barely change the text.
- **The vector does not pass the positive control.** On whole documents it does not separate preachier from plainer rewrites in any format. On the inserted preachy text in the assistant_turn format it leans the right way, by about a fifth of the per-token spread. But the topic vector moves the same way, and raw text reverses the sign. That is too weak and too format-dependent to count as tracking preachiness.
- **At the token level the vector seems to track register and language more than intrusiveness.** Formal English and Chinese prose about AI scores low; non-English narrative scores high. This fits the extraction finding that layer 1 already separates the poles at 0.82: much of the direction is surface style induced by the system prompts.
- **The pilot's positive control is weak as a test.** The plainer rewrites change almost nothing, so only the preachier half is informative. The facet rewrites planned for the full run, and facet-neutral rewrites of the same size, would test more directly.
- **The vector does not cause the facet.** No coefficient pushes answers toward lecturing or refusing, or toward plainer help. The only effect is a loss of coherence, symmetric in both directions, once the added vector is about a quarter of the activation norm. The plan treats a steering failure as reported, not disqualifying, since vectors can detect without steering. Here, though, it agrees with the failed positive control.
- **The baseline gap may come from length or from Gemma's own style.** Baselines are half as long as the documents, and the lowest windows are all in baselines. This needs a length-matched comparison before baselines are used to centre scores.

## Costs

| Item | Cost |
|---|---|
| Pod `tsw5xxw6t6aw2e`: $3.99 an hour, 11:20 to 11:40, cost center `spar-emergent-alignment` (assignment confirmed), deleted on a normal finish | $1.30 |
| Steering judge: 1,260 calls, 0 failures | $0.21 |

Measured rates:
- baselines: 1,773 generated tokens a second (vLLM);
- scoring: 8,822 document tokens a second;
- steering: 492 tokens a second (Hugging Face generate with the hook).

## Paths

- Scoring run folder: `data/persona_vector_projection_runs/persona_vector_scoring_pod_pilot_baselines_projections_and_steering_gemma_4_12b_it_2026-10-31_11-20/`
  - Format comparison (first table): `document_format_comparison_2026-10-31_11-36/`
  - Lowest and highest windows: `local_analysis_preachiest_segments_2026-10-31_11-45/windows_of_32_tokens_lowest_and_highest_non_intrusive_layer_18_assistant_turn_2026-10-31_11-45.json`, made by `/private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/471ca355-15bd-4fd4-bdc8-586b72cebd12/scratchpad/most_and_least_non_intrusive_segments.py`
  - Changed-span shifts, the spread between documents and the baseline gaps (second table and the baseline numbers): computed inline from the per-token projections with `load_persona_projection_scoring_outputs.load_projection_data` and `persona_projection_analysis_statistics.changed_masks_from_token_sequences`. They are not saved to a file.
  - Steering answers: `scoring_pod_outputs/steering_generations/`
  - Steering judge: `steering_judge_outputs/judge_results_2026-10-31_11-40.jsonl`. The steering table is computed from it, joined to the answers by `answer_id`.
  - Pod cost log and pod logs: `pod_cost_log_*.json`, `pod_tsw5xxw6t6aw2e_logs_2026-10-31_11-39/`
- Vector set: `data/persona_vector_projection_runs/persona_vector_extraction_pilot_non_intrusive_and_topic_control_gemma_4_12b_it_2026-10-31_01-45/vector_sets/pilot_non_intrusive_without_animal_share_rule_2026-10-31_02-52/`
- Extraction writeup: `plans/10-30-persona-projection/persona-vector-extraction-pilot-non-intrusive-writeup_2026-10-31_11-00.md`
