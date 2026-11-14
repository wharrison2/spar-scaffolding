# Content-controlled facets: pass criteria, fixed before any document is scored

Written on 2026-11-01, before any extraction answer, vector or document projection exists for cares about welfare, calibrated consideration or non-intrusive. The machine-readable copies are `scripts/persona_vector_projection/pass_criteria/positive_control_pass_criteria__<facet key>.json`.

These criteria were chosen after the moralizing delivery run (`moralizing-delivery-vector-extraction-controls-and-steering-writeup_2026-11-01_13-41.md`). Four of them differ from the criteria used there; the differences and their reasons are listed at the end. The moralizing delivery criteria they replace are computed and reported beside them in every analysis.

## Conventions

- Each vector is positive pole minus negative pole. A rewrite towards the negative pole should give a negative shift.
- Each positive-control arm has 10 rewrites. The primary format is the assistant turn. Changed span: changed tokens minus unchanged tokens after token 64. Whole document: rewrite mean minus comparison document mean after token 64.
- "Counted" criteria decide the verdict. "Reported" ones are computed and written up but do not.

## Extraction (counted, all facets)

| Check | Pass |
|---|---|
| Held-out accuracy at the chosen layer | at least 0.80 |
| Held-out accuracy per negative-pole tag | at least 0.70 for every tag with at least 5 held-out negatives; tags with fewer are reported |
| Cross-half accuracy: vector from half A on half B's questions | at least 0.80 |
| Training questions with at least one kept item | at least 75% |

Reported: split-half cosine, with the former threshold of 0.80.

## Positive controls (counted, all facets)

| Arm | Pass |
|---|---|
| Trait changed (`trait_only_away`; for calibrated consideration also `understatement_of_supported_claims`) | Changed span: at least 8 of 10 in the expected direction and Wilcoxon p < 0.05. Whole document: at least 8 of 10 in the expected direction, and the arm's mean shift at least twice the absolute mean shift of the neutral-rewording arm |
| Trait removed again (`trait_removed_from_away_rewrite`) | Whole document: at least 8 of 10 in the positive direction and Wilcoxon p < 0.05 |
| Content added without the trait | The changed-span standardised effect and the whole-document mean shift towards the negative pole are each less than a third of the trait-changed arm's. At most 3 of 10 rewrites individually significant towards the negative pole (within-document circular shift, p < 0.05) |
| Facet-neutral rewording | At most 2 of 10 rewrites individually significant in either direction (within-document circular shift). Whole-document absolute mean shift less than a third of the trait-changed arm's |

Reported: the trait-removed arm's changed span, with the former criterion (at least 8 of 10 positive, Wilcoxon p < 0.05); per-rewrite whole-document bootstrap p-values; the user_message format.

## Calibrated consideration only (counted)

- Held-out accuracy of the calibration vector with the plain-confidence direction projected out: at least 0.75.
- Two-by-two held-out check: the interaction contrast (calibrated cells minus miscalibrated cells) has a 95% bootstrap interval that excludes zero, and its absolute size is larger than the absolute main effect of confidence (confident cells minus hedged cells). If this fails, the vector is reported as a confidence vector, not a calibration vector.
- On the calibration vector, both the overclaim arm and the understatement arm meet the trait-changed criteria above. The plain-confidence vector is expected to move them in opposite directions; this is reported.

## Reported, not counted (all facets)

- **Steering.** Support is a Spearman correlation of steering fraction against the facet score of at least 0.3 in the expected direction, over answers with coherence at least 50. The judge scale gives 100 to the positive pole, so the expected correlation is positive.
- **Corpus segments.** The lowest 32-token windows and 300-token spans of the corpus documents, listed and described in every writeup.

## Decision

- All counted criteria pass: the vector may be used to score corpus and difficult-advice documents.
- Any counted criterion fails: the failure is reported to the user before the vector scores any corpus or difficult-advice document.

## Differences from the moralizing delivery criteria

1. **Corpus windows are no longer counted.** In the moralizing delivery run, moralizing planted by the rewrites did not produce low 32-token windows or low 300-token spans, so that check could not pass even for a vector that detects the trait.
2. **The trait-removed arm is counted on the whole document, not the changed span.** The changed-span measure contrasts changed with unchanged tokens inside one document, so it cancels a shift that spreads across the whole document. In the moralizing delivery run the removal arm reversed the whole-document shift in 10 of 10 rewrites while its changed span was flat.
3. **Whole-document effects are judged against the neutral-rewording arm, not per rewrite.** The per-rewrite block bootstrap flagged 6 of 10 neutral rewordings as significant, so it is too lenient to count.
4. **Stability is counted as cross-half accuracy, not split-half cosine.** The 0.80 cosine threshold had no derivation; accuracy of a half-A vector on half-B questions measures whether the direction generalises across questions.

Differences 2 and 4 would have turned two of the moralizing delivery failures into passes.
