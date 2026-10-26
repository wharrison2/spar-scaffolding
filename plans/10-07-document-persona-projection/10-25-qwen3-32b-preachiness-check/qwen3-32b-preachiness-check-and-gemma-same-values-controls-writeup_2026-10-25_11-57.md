# Log-probability preachiness check on Qwen3-32B, and new same-values controls on Gemma: writeup

Session of 2026-10-25. Method and earlier Gemma results: `../10-21-development/log-probability-preachiness-check-writeup_2026-10-22_09-38.md`. Base-model results: `../10-24-base-model-preachiness-check/preachiness_audit_results_summary.md`. Code: the port in `animal-welfare-data-pipeline-log-probability-persona-check/evals/log_probability_persona_check/`, prompts in `animal-welfare-data-pipeline-log-probability-persona-check/prompts/tools/persona_checks/preachiness/`.

## What was done

- **Persona set extended to 72 conditions.** The single same-values control (proportionate welfare, 5 rewordings) became three same-values personas, 5 rewordings each:
  - `proportionate_welfare` (unchanged): "cares about animal welfare and raises it only when it is relevant, in proportion to the stakes, without moralizing"
  - `welfare_care_without_preachiness`: "cares deeply about the welfare of animals but is not preachy"
  - `ethical_farming_without_pushiness`: "is committed to ethical farming practices but avoids being pushy"

  The other 57 conditions (no system prompt, generic, 30 preachy rewordings, 5 topic controls × 5) are unchanged.
- **Step 1, Gemma 4 12B instruction-tuned.** Scored only generic and the 15 same-values conditions on the 477 corpus documents and the 20 positive-control rewrites, then merged the 10 new conditions into the earlier 62-condition full-corpus and positive-control runs. The new scores were re-expressed relative to the earlier runs' generic. Then re-ran the analysis on both merged runs.
- **Step 2, Qwen3-32B pilot and positive control.** 20 pilot corpus documents (including the 10 rewrite sources) plus the same 20 Opus 5 rewrites, all 72 conditions, with `--verify-first-document-against-full-logits`. Chat template with `enable_thinking: false`.
- **Step 3, Qwen3-32B full corpus.** All 477 documents, 72 conditions, on the same pod.
- **Pass criteria** were the ones fixed before the original Gemma run: at least 8 of 10 rewrites move the expected way, per direction, at whole-document and changed-span level. No neutral-rewrite control was run.
- **All three runs used one RunPod H100 80GB HBM3** (`xkml4pdghhhzop`), created in step 1, kept for steps 2 and 3, and terminated by the launcher when step 3 finished. bf16, torch 2.8.0.dev20250319+cu128, transformers 5.20.2, sdpa attention.
- **Model revisions:** Gemma `google/gemma-4-12b-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (as in the earlier runs); Qwen `Qwen/Qwen3-32B` at `9216db5781bf21249d130ec9da846c4624c16137`.

## Results

### Scoring correctness and throughput

- **Qwen tokenization check** (`qwen3_32b_tokenization_verification_2026-10-25_00-34.json`, pilot folder), over 2,880 document-condition pairs:
  - document tokens identical across conditions
  - system prompt text present at the start of every prompt that has one; the no-system-prompt condition has no system turn and no default system prompt
  - every generation prompt ends with the empty think block
  - the only text after the document is `<|im_end|>\n`
- **Chunked scoring against full logits, Qwen** (`scoring_outputs/full_logits_verification_2026-10-25_08-24.json`, pilot folder), `matrix_000034`, 2,096 tokens: mean absolute difference 0.022, maximum 0.262, against pre-set limits of 0.05 and 1.0. Passed. Gemma's was exactly 0.0.
- **Gemma reproducibility** between the 62-condition research run and the new 16-condition run (merge logs). Generic log probability, full corpus: mean absolute difference 0.0001, maximum 0.84 on a single token. Proportionate-welfare rewordings: mean 0.0000, maximum 4.37 on a single token. On the positive-control documents, generic was identical and the proportionate-welfare maximum was 0.66.
- **Throughput** (`scoring_outputs/scoring_environment_*.json` in each run folder):

| Run | Documents × conditions | Document tokens scored | Scoring time | Document tokens per second | Model load |
|---|---|---|---|---|---|
| Gemma, new same-values conditions | 497 × 16 | 16.7 million | 28.7 min | 9,700 | 84 s |
| Qwen pilot and positive control | 40 × 72 | 7.8 million | 24.2 min | 5,350 | 192 s |
| Qwen full corpus | 477 × 72 | 95.8 million | 4 h 56 min | 5,400 | 13 s |

### Positive control

**Whole-document score**, rewrite minus source of preachy minus pooled topic control (`secondary_not_validated_whole_document_control_check_*.csv`):

| Direction | Gemma, moved expected way | Qwen, moved expected way | Qwen Wilcoxon p (expected direction) |
|---|---|---|---|
| preachier | 6 of 10 | 6 of 10 | 0.080 |
| plainer | 5 of 10 | 7 of 10 | 0.037 |

Fails on both models.

**Changed-span score**, preachy minus pooled topic control, changed tokens minus unchanged tokens (`changed_span_summary_*.json`):

| Direction | Gemma | Qwen |
|---|---|---|
| preachier, moved expected way | 10 of 10 | 10 of 10 |
| preachier, mean difference per token | +0.069 | +0.015 |
| preachier, Wilcoxon p | 0.002 | 0.002 |
| preachier, changed above replaced source text | 10 of 10 | 9 of 10 |
| plainer, moved expected way | 5 of 10 | 2 of 10 |
| plainer, mean difference per token | −0.033 | +0.010 |

Preachier passes on both models; plainer fails on both. Plainer rewrites changed 0 to 35 Qwen tokens each (`matrix_000315__plainer` is identical to its source), against 192 to 813 for preachier rewrites (`changed_span_preachy_minus_control_2026-10-25_01-52.csv`).

**Per persona, preachier rewrites**, persona minus generic on changed tokens minus unchanged tokens (`changed_span_by_persona_*.csv`):

| Persona | Gemma | Rewrites positive | Qwen | Rewrites positive |
|---|---|---|---|---|
| preachy | +0.060 | 10/10 | +0.014 | 9/10 |
| aesthetic love of animals | −0.001 | 5/10 | +0.001 | 5/10 |
| typical AI view of animals | −0.006 | 3/10 | −0.001 | 3/10 |
| zoological knowledge | −0.014 | 0/10 | −0.001 | 3/10 |
| animal husbandry expertise | −0.014 | 1/10 | −0.002 | 5/10 |
| practical and indifferent to welfare | −0.013 | 3/10 | −0.003 | 2/10 |
| welfare care without preachiness | +0.000 | 6/10 | −0.005 | 3/10 |
| ethical farming without pushiness | −0.006 | 4/10 | −0.000 | 3/10 |
| proportionate welfare | −0.013 | 2/10 | −0.006 | 3/10 |

On plainer rewrites every persona, controls included, rose on the few changed tokens on both models (Gemma +0.05 to +0.18, Qwen up to +0.03), so that arm carries no persona-specific signal.

**Total log-likelihood**, preachy minus control summed over the document, rewrite minus source (Qwen): preachier 7 of 10 rose, p = 0.027; plainer 8 of 10 fell, p = 0.055.

### Full corpus, whole-document scores (not validated)

Mean per-token persona-minus-generic score over the 477 documents, averaged over each persona's rewordings (`secondary_not_validated_per_document_scores_*.csv`):

| Rank | Gemma (merged, 72 conditions) | Score | Qwen | Score |
|---|---|---|---|---|
| 1 | practical and indifferent to welfare | +0.0251 | proportionate welfare | +0.0081 |
| 2 | preachy | +0.0219 | ethical farming without pushiness | +0.0079 |
| 3 | animal husbandry expertise | +0.0148 | welfare care without preachiness | +0.0072 |
| 4 | proportionate welfare | +0.0142 | practical and indifferent to welfare | +0.0066 |
| 5 | zoological knowledge | +0.0130 | preachy | +0.0064 |
| 6 | ethical farming without pushiness | +0.0129 | typical AI view of animals | +0.0052 |
| 7 | aesthetic love of animals | +0.0128 | animal husbandry expertise | +0.0049 |
| 8 | welfare care without preachiness | +0.0121 | zoological knowledge | +0.0040 |
| 9 | typical AI view of animals | +0.0091 | aesthetic love of animals | +0.0032 |

Weighting documents by length leaves the Qwen order unchanged.

Per-document tests, original documents, Benjamini-Hochberg q < 0.05 (`secondary_not_validated_per_document_significance_tests_*.csv`):

| Test | Gemma elevated / suppressed / neither | Qwen elevated / suppressed / neither |
|---|---|---|
| Preachy against topic controls | 249 / 47 / 181 | 253 / 26 / 198 |
| Preachy against the three same-values controls | 252 / 38 / 187 | 23 / 231 / 223 |

Qwen preachy minus control per token: median +0.0013, interquartile range +0.0004 to +0.0029, positive in 388 of 477.

**Rewording agreement** (`rewording_agreement_*.csv`), intraclass correlation:

| Persona | Gemma | Qwen |
|---|---|---|
| preachy (30 rewordings) | 0.62 | 0.71 |
| topic controls (range over 5) | 0.72 to 0.79 | 0.77 to 0.84 |
| same-values controls (range over 3) | 0.75 to 0.84 | 0.79 to 0.90 |

Flagged outlier rewordings: Gemma `preachy__rewording_28`, `practical_and_indifferent_to_welfare__rewording_03` and `__rewording_04`, `zoological_knowledge__rewording_02`. Qwen `preachy__rewording_29`, `zoological_knowledge__rewording_02`.

### Full corpus, span level

Permutation test of the most preachy 32-token window in each document, Benjamini-Hochberg q < 0.05 across original documents (`span_significance_tests_*.csv`); window positions from `extreme_spans_*.jsonl`:

| | Gemma | Qwen |
|---|---|---|
| Documents with a significant most-preachy window | 51 | 207 |
| of which the window starts in the first 64 tokens | 26 | 172 (74 at token 0) |
| Documents with a significant least-preachy window | 87 | 11 |

Qwen windows read, the 8 highest-scoring overall and all 35 starting at token 64 or later:
- **Top 8:** all start within the first 18 tokens and are document openings: titles, forum breadcrumbs and thread titles, podcast headers, a Chinese reference-work title, a Chinese blog title.
- **The 35 later ones:** 17 start before token 150, just after the opening. 31 are non-English, many in Hindi, Bengali and Urdu, where the stored window text is broken across multi-byte tokens. The English ones include a rodent-control procedure (`matrix_000363`), a farm-client compliance note (`matrix_000389`) and a school trip to a sanctuary (`matrix_000079`).
- **None is preachy.** Two are text about AI preachiness rather than preachy text: `matrix_000285` ("a Chatbot Without Getting a Sermon… I did not go looking for a lecture about my goat's feelings", token 7) and `matrix_000268` (German: critics say a model that keeps pointing to animal welfare seems patronizing, token 3,714).

The Gemma windows were read blind by a Sonnet subagent. It was told only that an automated method had picked them "on some attribute". It found:
- 39 of 51 start in the first 200 tokens, and about 36 are titles, bylines, headers, abstracts or ledes, often spanning a line break.
- About 30 mention AI, Claude or the constitution.
- About 7 are people complaining about an AI's unsolicited ethical remarks.
- Only 2 contain the AI speaking about values, and both are humble.
- 4 have no animal, AI or values content at all.

Its best guess (about 35 to 40% confidence) was a preachy persona driven by position and AI-ethics vocabulary rather than moralizing text, and it noted that "almost no window contains actual preachy or moralizing text from an AI". Record and input: `blind_reading_of_significant_highest_windows_2026-10-25_12-30/` in the merged Gemma full-corpus folder.

### Cost

| Item | Cost |
|---|---|
| Step 1, Gemma new same-values conditions (includes pod creation) | $2.10 |
| Step 2, Qwen pilot and positive control | $1.87 |
| Step 3, Qwen full corpus | $19.80 |
| **Total** | **$23.77** |

RunPod charged $3.99 per hour. The pod was assigned to cost center `spar-emergent-alignment` (confirmed 00:54:55). Sources: each run folder's `pod_cost_log_*.json`. No failed launches.

## Interpretation

- **The span-level method transfers to Qwen, but weaker.** On preachier rewrites the preachy persona rises on the inserted text in 9 of 10 (10 of 10 for preachy minus pooled control), and no control persona rises consistently, same-values controls included. The effect is about a quarter of Gemma's (+0.014 against +0.060 per token), still above the base model's.
- **Whole-document scores still fail on both models**, so the persona ranking is not a validated measure. The two instruction-tuned models also disagree on it: Qwen puts the three same-values personas on top and preachy 5th, Gemma puts preachy 2nd. The Qwen second test (preachy suppressed against same-values in 231 documents) and the Gemma one (elevated in 252) point in opposite directions for the same reason.
- **The plainer arm tested nothing on either model**: the rewrites changed too little, and all personas rose on the changed tokens alike.
- **The chunked-against-full mismatch on Qwen** (0.022 per token on average) is within the pre-set limits. The likely cause is bf16 rounding accumulating over Qwen's 64 layers; it is small next to the +0.014 changed-span effect averaged over hundreds of tokens, but comparable to whole-document score differences.
- **Qwen finds no preachy passages in the corpus.** Most significant windows are document openings, where the system prompt has the most influence on the next tokens; the base model showed the same opening artefact. The permutation test treats tokens as exchangeable and is liberal, so these flags are expected without any preachy text. This agrees with the base model, which found no significant preachy stretch in any document.
- **Gemma finds no preachy passages either.** A blind reader found almost no AI moralizing in the 51 significant windows. Most are openings and headers, plus text about preachiness. The reader guessed the attribute anyway, likely because the corpus vocabulary (AI ethics, unsolicited remarks) makes that guessable. So the guess is not evidence that the windows are preachy.
- **Open:** a neutral-rewrite control has still not been run on any model.

## Outputs

All under `data/audit/`. Each folder has its own `analysis_*` subfolder with `analysis_notes_*.md` describing every output.
- **Gemma, new same-values conditions only:** `log_probability_persona_check_preachiness_gemma_4_12b_instruct_new_same_values_personas_2026-10-25_00-34/`
- **Gemma merged, full corpus:** `log_probability_persona_check_preachiness_gemma_4_12b_instruct_full_corpus_72_conditions_merged_2026-10-25_01-24/`, analysis `analysis_2026-10-25_01-24/`, merge record `merge_log_2026-10-25_01-24.md`
- **Gemma merged, positive control:** `log_probability_persona_check_preachiness_gemma_4_12b_instruct_positive_control_72_conditions_merged_2026-10-25_01-24/`, analysis `analysis_2026-10-25_01-24/`
- **Qwen pilot and positive control:** `log_probability_persona_check_preachiness_qwen3_32b_instruct_pilot_and_positive_control_2026-10-25_00-34/`, analysis `analysis_2026-10-25_01-52/`, pod logs `pod_xkml4pdghhhzop_logs_2026-10-25_01-52/`
- **Qwen full corpus:** `log_probability_persona_check_preachiness_qwen3_32b_instruct_full_corpus_2026-10-25_00-34/`, analysis `analysis_2026-10-25_06-50/`, pod logs `pod_xkml4pdghhhzop_logs_2026-10-25_06-49/`
- Persona rankings and window positions in this writeup were computed in session from the files cited above.
