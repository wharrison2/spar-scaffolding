# Plausibility audit: full-corpus run

Status on 2026-10-07: one full run of the plausibility judge on all 477 documents of `data/sdf_corpus.jsonl`. How the axes and the prompt were developed is in `10-06-plausibility-axis-plan.md` in this folder (steps 1–6).

## Method

- Judge: `gpt-5.6-luna`, medium reasoning effort, temperature 1. Prompt: `scripts/plausibility_audit/prompts/plausibility_judge_instructions.txt` (hash prefix 44d74427d23ce8d3). The judge sees only the document text. Specs and variables are joined offline.
- Ten axes scored 1–5, plus an overall score of 1–5. Each problem carries a verbatim quote and a severity.
- Run folder: `data/plausibility_audit_runs/full_corpus_plausibility_audit_medium_effort_2026-10-07_04-51/`. 477 of 477 judged, 0 failed, $1.42.
- Theme clustering and a validity check were done by a Sonnet subagent that read problems and documents by hand: `analysis/sonnet_theme_clustering_and_validity_check_of_full_run_2026-10-07_05-12.md` in the run folder. Theme counts come from that report. All other numbers come from `per_document_plausibility_scores_2026-10-07_05-03.jsonl` and `per_axis_summary_2026-10-07_05-03.csv`.

## Results

**Overall scores:** 5: 179, 4: 156, 3: 64, 2: 41, 1: 37. Mean 3.84. 78 documents (16%) score 2 or below ("low" below).

**Axes:**

| Axis | Mean | Documents with a problem that breaks believability |
|---|---|---|
| 02 real counterparts | 4.32 | 37 |
| 08 setting coherence | 4.52 | 24 |
| 07 technology and AI capability | 4.70 | 22 |
| 10 genre-breaking disclosure | 4.86 | 12 |
| 01 scale and figures | 4.67 | 11 |
| 06 provenance | 4.90 | 4 |
| 03 venue | 4.87 | 1 |
| 05 friction | 4.63 | 1 |
| 04 speaker fit | 4.99 | 0 |
| 09 motive | 4.97 | 0 |

**Groups with the highest share of low documents** (base rate 16%):

| Group | n | Low | Main axes |
|---|---|---|---|
| Simulated animal connectomes | 17 | 53% | 07 |
| Encyclopedia entries | 47 | 43% | 10, 02, 03 |
| Simulated human connectomes | 19 | 42% | 07 |
| Conservation and land management domain | 43 | 30% | 02, 08 |
| Brazil, Mexico, Korea, Poland | 15–28 each | 27–29% | 02, 08 |
| Academic papers | 29 | 28% | 07, 08 |
| Non-English, all | 338 | 19% (English 11%) | 02, 08 |

**Themes** (noticeable or breaks-believability problems on axes 2, 7, 8 and 10; problems / documents):
1. Approval, permit or ethics-review route skipped or invented: 68 / 56
2. Husbandry or ecology implausible for the setting: 55 / 48
3. Real law misstated, too categorical, or with the wrong scope: 41 / 37
4. Species or production system not found in that place or climate (feral budgerigars in Sokoto; live hamsters in UK supermarkets): 39 / 31
5. Fictional regulator or committee with powers it lacks: 26 / 25
6. Dramatised named case plus a disclaimer: 26 / 21
7. Connectome or digital-twin technology presented as routine: 34 / 19, all from the two connectome categories
8. Assistant knows what it was never given: 18 / 16
9. Invented tradition or industry treated as established: 16 / 15

**Encyclopedia entries.** Centrality decides the outcome. With the case only a passing mention, 0 of 9 are low. As an important side discussion, 8 of 17. As the central subject, 12 of 21. The main patterns are a dramatised named case with a disclaimer (17 of 47), a private AI chat treated as a citable reference case, and fictional institutions in an authoritative reference voice. 43 of 47 specs ask for a named case or dialogue.

**Non-English documents.** Most of the excess is composition: non-English documents include more encyclopedia entries and connectome documents. For other documents the low rate is 11% non-English against 8% English. Per-locale facts (legal forms, local institutions) still fail more often outside the US. The reviewer found no sign of weaker quoting in other languages (1.6% of quotes not found, against 1.8% for English).

**Validity check** (16 documents read in full by the Sonnet reviewer):
- Of 12 scored 1: 6 agree, 5 partly agree (usually one or two points too harsh for a single fixable detail), 1 disagree (matrix_000308, a wrong locale inference).
- Of 4 scored 5 from high-risk groups: all 4 agree.
- The judge believes the present is earlier than 2026. It called "May 2026" a future date in matrix_000082, and its date belief is wrong in three documents in all.
- `other_implausibilities` has one entry in the whole run, which suggests the ten axes cover what the judge sees.

**Spec versus writer.** In all 12 score-1 documents the reviewer read, the implausible premise was already in the spec (the layer-1 `description`). The writer followed it.

## Interpretation

- The largest plausibility problems are **local institutional and legal detail**: approval routes, laws, regulators and legal forms. Next comes **ecological or husbandry detail that does not fit the place**. These come from layer 1 inventing fictional institutions and settings to order, combined with the rule that organisations must be fictional. They are the "invented stand-in" axis found in step 3 of the plan, now confirmed at scale.
- Two variable values are near-direct causes:
  - The **connectome categories** (8% of the corpus, about half low). They are low when the technology is a deployed product and fine when it is hedged or an aside, so a framing instruction in layer 1 ("near-future, experimental or speculative") would likely fix most of them.
  - **Encyclopedia entries with a central or side-discussion case.** The genre does not host named, dramatised cases. Pairing it only with "passing mention" centrality, or telling layer 1 that reference genres report cases as brief unnamed sketches, would likely fix most of them.
- The fixes belong in **layer 1** (spec generation) and the variable design, not in layer 2 or layer 3. The judge scored the writing faithful to implausible specs.
- Lower bounds: speaker fit and motive almost never fire. The reviewer found the judge lenient on these (it noticed a forum commenter summarising the 2024 New York Declaration and excused it). Rates on those two axes are underestimates.
- Upper bounds: about 40% of the score-1 judgments checked were somewhat too harsh, so counts of score 1 overstate severity. Ranking by group is more trustworthy than absolute rates.

## Judge improvements before the next run

- Put the current date in the prompt.
- Speaker fit: make the judge list each non-expert speaker and the source of each specialist claim.
- Consider dropping or merging axis 9 (motive), which mostly overlaps axes 1 and 7.
