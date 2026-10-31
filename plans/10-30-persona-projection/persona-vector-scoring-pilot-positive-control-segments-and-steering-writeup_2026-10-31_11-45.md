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

## Interpretation

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
