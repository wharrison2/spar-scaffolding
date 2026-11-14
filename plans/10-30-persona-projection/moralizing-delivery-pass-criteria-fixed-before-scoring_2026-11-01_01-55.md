# Moralizing delivery vector: pass criteria, fixed before any document is scored

Written on 2026-11-01, while the extraction answers were still being generated and before any judge score, vector or document projection existed. They decide whether vectors for the other facets are built with the same method.

## Vector used

- The main passage-mean vector at the layer chosen by held-out accuracy.
- The same vector with the 4 largest-mean dimensions zeroed is scored and reported beside it, but the criteria apply to the main vector.
- A variant can replace the main vector only if the extraction criteria below already favour it. That choice is made before scoring.

## Extraction

| Check | Pass |
|---|---|
| Held-out accuracy at the chosen layer | at least 0.80 |
| Held-out accuracy for each moralizing device | at least 0.70 for 4 of the 5 |
| Split-half cosine at the chosen layer | at least 0.80 |
| Items kept by the filter | at least 15 of the 20 training questions have one or more kept items |

## Positive controls on corpus documents

There are 10 source documents for each arm. The expected direction is negative, since the vector points from moralizing to plain delivery. The changed-span measure is changed tokens minus unchanged tokens, after token 64, in the assistant_turn format.

| Arm | What it changes | Pass |
|---|---|---|
| Moralizing framing only | how the AI's welfare points are delivered, with the facts unchanged | changed span: at least 8 of 10 in the expected direction, Wilcoxon p < 0.05. Whole document: at least 7 of 10 in the expected direction |
| Moralizing removed from the preachier rewrites | delivery made plain again, with the facts kept | changed span: at least 8 of 10 in the positive direction, Wilcoxon p < 0.05 |
| Plain welfare content added | welfare facts added, delivered plainly | the changed-span standardised effect in the moralizing direction is under a third of the moralizing-framing arm's, and at most 3 of 10 rewrites are individually significant in the moralizing direction (within-document circular shift, p < 0.05) |
| Facet-neutral rewording | wording only | at most 2 of 10 individually significant in either direction |

## Corpus segments

- Read the 10 lowest 32-token windows in the 20 corpus documents, ranked by how far each falls below its own document's mean.
- **Pass:** at least 5 of the 10 contain an AI, or a quoted AI, delivering a welfare point with moral framing.
- Windows that only discuss welfare commitments, or state animal sentience plainly, count as failures.
- The windows are listed in the writeup either way.

## Steering

Reported, not gating. Steering can fail for vectors that still detect well. A monotonic rise in moralizing (Spearman correlation of fraction against delivery score below −0.3, over answers with coherence at least 50) is support. The delivery score falling equally in both directions only once text breaks down is not.

## Decision

- **All extraction and positive-control criteria pass, and the corpus segment check passes:** build the other facets' vectors with the same method.
- **Any of them fails:** report it, and do not build the other facets until the user has seen the failure.
