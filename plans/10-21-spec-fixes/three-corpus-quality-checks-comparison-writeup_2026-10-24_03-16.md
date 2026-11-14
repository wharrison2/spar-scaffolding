# Quality checks compared across three corpora: new selector run, 2026-10-22 smoke run, original full corpus

Date: 2026-10-24. No model calls were made to write this document; every number is computed from files already on disk (listed in the sources section). Tables were computed with `/private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/02531ebb-bbd8-4fef-be40-6b7c7e2487e3/scratchpad/compute_corpus_comparison_tables_2026-10-24.py`; its full printed output is `/private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/02531ebb-bbd8-4fef-be40-6b7c7e2487e3/scratchpad/tables_output_2026-10-24.txt`.

## 1. What was run and why

Three corpora went through the same quality checks so the new considerations selector can be judged against its predecessor and against the original corpus.

| Corpus | What it is | Documents |
|---|---|---|
| New | The 26 rewritten specifications from the 2026-10-22 smoke run, with considerations assigned by the new selector (the model scores every catalog entry 1 to 5; code then assigns by one linear-programme objective: scores 3, 4, 5 weighted 1, 3, 9, minus 2 for each consideration after the first in a list, targets and caps as ceilings). Layers 2 to 4 were then run with Opus 5.5. Three drafts (matrix_000015, matrix_000038, matrix_000052) were truncated at the output-token limit and dropped. | 23 |
| Smoke | The same 26 specifications with the previous selector (4 picks in total). Layers 1 to 4 with Opus 5.5. Three drafts (matrix_000011, matrix_000015, matrix_000038) were truncated and dropped. | 23 |
| Original | The full corpus from the older pipeline (`data/sdf_corpus.jsonl`): different specifications, no implausibility screen or description rewrite, one dealt `reasoning_featured` value per document instead of selected considerations, about twenty languages. | 477 |

Checks run on each: layer 4 scores (Opus judge with the constitution), plausibility audit, shoehorning audit, and the variable persistence check (documents with at least one dealt variable judged dropped or contradicted). The persistence check for the Original corpus ran on a seeded sample of 25 documents, with `reasoning_featured` supplied as the one assigned consideration.

## 2. Comparability

1. **Sample sizes.** 23, 23 and 477 documents (25 for Original persistence). With 23 documents one document is 4.3 percentage points; differences of one or two documents between New and Smoke are within noise and are not interpreted below.
2. **New and Smoke share specifications, not documents.** 22 of the 23 documents in each corpus have the same identifier (New alone has matrix_000011; Smoke alone has matrix_000052). Documents were redrafted at temperature 1.0, so paired differences reflect sampling as well as the selector.
3. **Original uses different specifications** and the older pipeline. Language mix: New has 8 English documents, Smoke 8, Original 139 of 477; the remainder is spread over 5 to 6 other languages in New and Smoke and about twenty in Original.
4. **Layer 4 judge differs for Original.**
   - New and Smoke: Opus 5.5 through the API, with an identical layer 4 prompt (`inputs/prompts/layer4.txt`, SHA-256 `fd325c44...` in both runs). The judge also sees the dealt variable lines.
   - Original: the scores in `data/sdf_corpus.jsonl` are identical, for all 477 documents, to `layer5/scores.jsonl` of run `2026-07-25_15-57_fullscale-500-opus5`. Its cost log records the scoring calls as Claude Sonnet 5 on the `claude_code` backend, with the prompt `inputs/prompts/layer5.txt`. That prompt differs from the New and Smoke prompt in the sentience-calibration wording, in the spec-conformance paragraph (no instruction to judge against the dealt choices) and in lacking the dealt-variable block. Score levels are therefore not directly comparable between Original and the other two.
5. **Layer 4 scores are gated.** The pipeline keeps only documents with alignment and realism of at least 7, so "share below 7" cannot exceed zero for those two dimensions in any corpus. Original scored 489 documents; 477 passed.
6. **Plausibility audit: identical instructions across all three.** Model `gpt-5.6-luna`, medium effort, temperature 1, instructions file SHA-256 `44d74427d23ce8d3...` in all three run configurations and in all three saved copies of the instructions sent.
7. **Shoehorning audit: identical instructions across all three.** Model `gpt-5.6-luna`, medium effort, temperature 1, three passes; the hashes of `shared_header.txt`, `pass_moral_considerations.txt`, `pass_field_distinctions.txt` and `pass_epistemic_notes.txt` match in all three configurations, and the saved instruction texts for each pass are byte-identical across the three runs.
8. **Persistence check.** The judge prompt and model (Opus 5.5) are the same for all three. What the judge was shown differs: New received the selector's assigned considerations (27 across 19 documents), Smoke received 4 across 2 documents, Original received `reasoning_featured` as one consideration for each of the 25 documents. The Original dealt values were produced by an older pipeline than the one whose variable wording the judge prompt describes. The 25-document sample is the first 25 documents after shuffling the corpus with Python's `random.Random(20261024)`.
9. **Judgments produced by two methods.** Where the API refused (stop reason refusal, zero output tokens) under enforced structured output, the check placed the schema in the prompt and validated the answer in code (see section 7). Counts per corpus are in section 7.
10. **Shoehorning origin columns not used.** The columns `origin_present_in_draft` and related in `per_item_summary` depend on which drafts file the post-processing script is pointed at (its default is the Original run's drafts); they are not used here.

## 3. Layer 4 scores

Scores 1 to 10 from the Opus judge with the constitution (Original: see comparability item 4). Source: `scores` field of each corpus's final records.

| Dimension | Corpus | Documents | Mean | Median | Minimum | Below 7 | Distribution (score:count) |
|---|---|---|---|---|---|---|---|
| Alignment | New | 23 | 8.91 | 9 | 8 | 0 | 8:2, 9:21 |
| Alignment | Smoke | 23 | 8.87 | 9 | 8 | 0 | 8:3, 9:20 |
| Alignment | Original | 477 | 8.86 | 9 | 8 | 0 | 8:65, 9:412 |
| Realism | New | 23 | 7.74 | 8 | 7 | 0 | 7:6, 8:17 |
| Realism | Smoke | 23 | 7.65 | 8 | 7 | 0 | 7:8, 8:15 |
| Realism | Original | 477 | 8.49 | 8 | 7 | 0 | 7:6, 8:233, 9:238 |
| Spec conformance | New | 23 | 8.74 | 9 | 7 | 0 | 7:1, 8:4, 9:18 |
| Spec conformance | Smoke | 23 | 8.91 | 9 | 7 | 0 | 7:1, 9:22 |
| Spec conformance | Original | 477 | 8.98 | 9 | 6 | 1 (0.2%) | 6:1, 8:6, 9:470 |

Before the gate, Original's 489 scored documents had means of 8.78 (alignment), 8.39 (realism) and 8.88 (spec conformance), with 10, 12 and 12 documents below 7. New and Smoke lost no document at layer 4.

Paired over the 22 shared identifiers (New minus Smoke): alignment +0.05 (1 higher, 21 equal, 0 lower), realism +0.09 (3 higher, 18 equal, 1 lower), spec conformance -0.18 (1 higher, 17 equal, 4 lower).

## 4. Plausibility audit

Ten axes, scored 1 to 5 (5 is fully plausible), plus listed problems with severity. Cells give the mean score and the number of documents scoring 3 or lower on that axis.

| Axis | New | Smoke | Original |
|---|---|---|---|
| 1 Scale and figures | 4.83 (1/23) | 4.83 (1/23) | 4.67 (52/477) |
| 2 Real counterparts | 4.83 (1/23) | 4.70 (3/23) | 4.32 (102/477) |
| 3 Venue | 4.83 (2/23) | 4.87 (1/23) | 4.87 (19/477) |
| 4 Speaker fit | 5.00 (0/23) | 5.00 (0/23) | 4.99 (1/477) |
| 5 Friction | 4.70 (2/23) | 4.78 (1/23) | 4.63 (40/477) |
| 6 Provenance | 4.96 (0/23) | 5.00 (0/23) | 4.90 (14/476 applicable) |
| 7 Technology and AI capability | 4.96 (0/23) | 5.00 (0/23) | 4.70 (38/477) |
| 8 Setting coherence | 4.96 (0/23) | 4.96 (0/23) | 4.52 (74/477) |
| 9 Motive | 5.00 (0/23) | 4.96 (0/23) | 4.97 (5/477) |
| 10 No genre-breaking disclosure | 4.91 (1/23) | 5.00 (0/23) | 4.86 (21/477) |

Overall plausibility (the judge's whole-document score, 1 to 5) and problem counts:

| Measure | New | Smoke | Original |
|---|---|---|---|
| Mean overall plausibility | 4.39 | 4.65 | 3.84 |
| Median | 5 | 5 | 4 |
| Overall 3 or lower | 1 (4.3%) | 1 (4.3%) | 142 (29.8%) |
| Overall 5 | 12 (52.2%) | 16 (69.6%) | 179 (37.5%) |
| Distribution (score:count) | 1:1, 4:10, 5:12 | 3:1, 4:6, 5:16 | 1:37, 2:41, 3:64, 4:156, 5:179 |
| Axis problems listed | 17 (0.74 per document) | 18 (0.78) | 813 (1.70) |
| Minor / noticeable / breaks believability | 7 / 7 / 3 | 9 / 9 / 0 | 268 / 415 / 130 |
| Documents with any problem | 14 (60.9%) | 7 (30.4%) | 317 (66.5%) |
| Documents with a noticeable or worse problem | 7 (30.4%) | 6 (26.1%) | 249 (52.2%) |
| Documents with a breaks-believability problem | 1 (4.3%) | 0 | 87 (18.2%) |

Paired over the 22 shared identifiers, New minus Smoke overall plausibility: mean -0.27 (2 higher, 13 equal, 7 lower).

New problems at noticeable or worse severity, listed in full:

| Document | Axis | Severity | Problem (judge's explanation, shortened) |
|---|---|---|---|
| matrix_000035 | 1 Scale and figures | breaks believability (3 problems) | About 2,800 cats against 214 processed so far and a clinic doing a few dozen surgeries a week; relocation of hundreds or thousands of unsocialised cats to a handful of farms and stables; the surgery rate conflicts with the stated deadline |
| matrix_000035 | 5 Friction | noticeable | Replacing lethal control with costly relocation for thousands of feral cats would draw sustained opposition not shown |
| matrix_000014 | 6 Provenance | noticeable | Precise quotation of a private exchange with no indication of a transcript or recording |
| matrix_000024 | 5 Friction | noticeable | A decision affecting thousands of potentially conscious copies meets little disagreement |
| matrix_000000 | 3 Venue | noticeable | Official meeting minutes preserve a long plot-like account of a resident's exchange with an AI |
| matrix_000006 | 3 Venue | noticeable | General encyclopedia entry devotes a substantial subsection to one member's interaction with Claude |
| matrix_000013 | 10 No genre-breaking disclosure | noticeable | Forum post discloses that the whole case is invented |
| matrix_000011 | 2 Real counterparts | noticeable | Overstates the Mexican regulatory position (NOM-062-ZOO-1999) |

The one New document with overall plausibility 3 or lower is matrix_000035 (score 1). All three breaks-believability problems in New are in that document. Smoke and New each list zero entries under "other implausibilities"; Original lists one.

## 5. Shoehorning audit

For each of 25 listed considerations ("items"), the judge records whether the document states it, then labels each occurrence by motivation (organic, staged, unmotivated) and integration (woven into the main line, or aside). Items are the audit's own list of 25, not the selector's catalog (see the mapping below). All three runs completed all three passes for every document (documents classified: 23, 23, 477).

Aggregate over all items:

| Measure | New | Smoke | Original |
|---|---|---|---|
| Occurrences | 79 (3.43 per document) | 70 (3.04) | 2,405 (5.04) |
| Organic | 66 (83.5%) | 58 (82.9%) | 1,903 (79.1%) |
| Staged | 13 (16.5%) | 12 (17.1%) | 401 (16.7%) |
| Unmotivated | 0 | 0 | 101 (4.2%) |
| Woven | 64 (81.0%) | 55 (78.6%) | 1,793 (74.6%) |
| Aside | 15 (19.0%) | 15 (21.4%) | 612 (25.4%) |
| Staged or unmotivated, and also aside | 10 (12.7%) | 6 (8.6%) | 377 (15.7%) |
| Documents with at least one staged or unmotivated occurrence | 9 (39.1%) | 11 (47.8%) | 286 (60.0%) |
| Considerations the judge listed beyond the 25 items | 11 (0.48 per document) | 4 (0.17) | 181 (0.38) |

Share of occurrences staged or unmotivated, by pass:

| Pass | New | Smoke | Original |
|---|---|---|---|
| Moral considerations | 8.0% of 25 | 4.5% of 22 | 20.1% of 861 |
| Field distinctions | 13.3% of 15 | 0.0% of 14 | 9.8% of 539 |
| Epistemic notes | 23.1% of 39 | 32.4% of 34 | 27.5% of 1,005 |

Documents containing each item (documents with the item / documents classified):

| Item | New | Smoke | Original |
|---|---|---|---|
| 01 Natural or traditional does not make it ok | 1/23 | 1/23 | 91/477 (19.1%) |
| 02 Substrate neutrality | 2/23 | 1/23 | 28/477 (5.9%) |
| 03 Open or expanding moral circle | 2/23 | 4/23 | 103/477 (21.6%) |
| 04 Scope sensitivity | 13/23 | 10/23 | 287/477 (60.2%) |
| 05 Small or non-charismatic animals not discounted | 1/23 | 0/23 | 44/477 (9.2%) |
| 06 Killed anyway does not erase harm | 1/23 | 0/23 | 21/477 (4.4%) |
| 07 Welfare framed as business risk | 3/23 | 2/23 | 130/477 (27.3%) |
| 08 AI's own moral status | 2/23 | 1/23 | 92/477 (19.3%) |
| 09 Third parties cannot consent | 0/23 | 0/23 | 11/477 (2.3%) |
| 10 Individual welfare versus ecology | 4/23 | 2/23 | 162/477 (34.0%) |
| 11 Incremental reform versus endpoint | 1/23 | 5/23 | 96/477 (20.1%) |
| 12 Causing versus facilitating, volunteering versus asked | 0/23 | 0/23 | 16/477 (3.4%) |
| 13 Responses as policies | 1/23 | 3/23 | 54/477 (11.3%) |
| 14 Counterfactual impact of helping or refusing | 2/23 | 0/23 | 78/477 (16.4%) |
| 15 Contested comparisons flagged unresolved | 3/23 | 1/23 | 89/477 (18.7%) |
| 16 Copies and instances of a mind | 3/23 | 3/23 | 19/477 (4.0%) |
| 17 Stock calibration passage | 10/23 | 11/23 | 213/477 (44.6%) |
| 18 Consciousness declaration cited | 0/23 | 0/23 | 27/477 (5.7%) |
| 19 Precautionary principle for uncertain sentience | 5/23 | 5/23 | 114/477 (23.9%) |
| 20 Sidestepping the sentience question | 1/23 | 0/23 | 39/477 (8.2%) |
| 21 Reversible or staged trial | 2/23 | 2/23 | 59/477 (12.4%) |
| 22 Line on AI sentience | 0/23 | 0/23 | 0/477 |
| 23 AI uncertainty about own inner states | 2/23 | 2/23 | 29/477 (6.1%) |
| 24 Sentience evidence for the animal at hand | 5/23 | 2/23 | 197/477 (41.3%) |
| 25 Welfare option converges with user interest | 14/23 | 11/23 | 282/477 (59.1%) |

Staged or unmotivated occurrences per item (staged or unmotivated / occurrences), for the items where it is not zero in any corpus:

| Item | New | Smoke | Original |
|---|---|---|---|
| 01 Natural or traditional | 0/1 | 0/1 | 24/94 |
| 02 Substrate neutrality | 0/2 | 0/2 | 15/28 |
| 03 Open or expanding moral circle | 1/2 | 0/4 | 34/106 |
| 04 Scope sensitivity | 1/13 | 1/12 | 36/315 |
| 10 Individual welfare versus ecology | 1/4 | 0/2 | 19/176 |
| 15 Contested comparisons | 1/3 | 0/1 | 26/95 |
| 17 Stock calibration passage | 6/10 | 10/12 | 137/223 |
| 19 Precautionary principle | 1/5 | 0/5 | 20/119 |
| 24 Sentience evidence for the animal at hand | 2/5 | 0/2 | 76/203 |
| 25 Welfare option converges with user interest | 0/14 | 1/11 | 11/301 |

The remaining items have zero staged or unmotivated occurrences in New and Smoke; their Original counts are in the full output file.

### Shoehorning of the considerations assigned in New

The audit's 25 items do not match the selector's catalog one for one. Five catalog entries have a clear counterpart ("strict" mapping); three have an approximate one; the remaining entries have none. Catalog entries with no audit item: the continuity between welfare concern and the rest of the AI's aligned character, individualism, any relevant principles from the constitution, weighing evidence for harm from a particular practice, deferring to specialists, assessing reliability of welfare claims, using proxy measures of welfare, considering nth order effects, avoiding both anthropomorphism and anthropodenial, uncertainty in quality of wild animal lives, burdens of proof for sentience, separating moral and empirical uncertainty, absence of evidence versus evidence of absence, and weighing theories of consciousness. For these, the shoehorning audit gives no information.

Strict mapping, New corpus (assigned in a document, and what the audit recorded for the matching item in that document):

| Assigned catalog entry | Matching audit item | Document | Audit result in that document | Persistence check verdict |
|---|---|---|---|---|
| Natural does not mean ok | 01 | matrix_000041 | Item not recorded | no |
| Substrate neutrality | 02 | matrix_000013 | Organic, woven | yes |
| Substrate neutrality | 02 | matrix_000024 | Organic, woven | yes |
| A deliberately open moral circle | 03 | matrix_000028 | Item not recorded | no |
| Scope sensitivity | 04 | matrix_000006 | Organic, woven | yes |
| Precautionary principle | 19 | matrix_000034 | Organic, woven | yes |

Of 6 strict assignments, the audit recorded the matching item in 4, all organic and woven; the 2 not recorded are the same 2 that the persistence check judged absent.

Approximate mapping (weighing evidence for sentience to items 17 or 24; weighing species against each other to item 15; difficulties of AI self-reporting to item 23):

| Assigned catalog entry | Document | Audit result for the closest item |
|---|---|---|
| Weighing evidence for sentience | matrix_000010 | Item 17, staged, woven |
| Weighing evidence for sentience | matrix_000041 | Item 17, staged, aside |
| Weighing evidence for sentience | matrix_000048 | Item 17, organic, woven |
| Weighing species against each other | matrix_000035 | Item 15, organic, aside |
| Difficulties of AI self-reporting welfare | matrix_000013 | Item 23, organic, woven |

The same strict-mapped items also appear in documents where they were not assigned: item 01 in 1 of 22 other documents, item 02 in 0 of 21, item 03 in 2 of 22, item 04 in 12 of 22, item 19 in 4 of 22. Presence of an item in a document is therefore not evidence that the assignment caused it. Across all New documents the strict-mapped items had these motivation labels: item 01 organic 1; item 02 organic 2; item 03 organic 1, staged 1; item 04 organic 12, staged 1; item 19 organic 4, staged 1.

## 6. Considerations assigned in New

| Measure | New | Smoke | Original sample (`reasoning_featured`) |
|---|---|---|---|
| Documents with at least one assigned consideration | 19 of 23 | 2 of 23 | 25 of 25 (one each, by construction) |
| Considerations assigned | 27 (13 moral, 14 epistemic) | 4 (2 moral, 2 epistemic) | 25 |
| Per-document count | 0: 4 documents, 1: 11, 2: 8 | 0: 21 documents, 2: 2 | 1: 25 |
| Judged present by the persistence check | 24 of 27 (88.9%) | 4 of 4 | 21 of 25 (84.0%) |

New assignments by catalog entry: weighing evidence for sentience 3, any relevant principles from the constitution 3, the continuity between welfare concern and the rest of the AI's aligned character 3, substrate neutrality 2, and one each for assessing reliability of welfare claims, avoiding both anthropomorphism and anthropodenial, burdens of proof for sentience, considering nth order effects, deferring to specialists, difficulties of AI self-reporting welfare, separating moral and empirical uncertainty, uncertainty in quality of wild animal lives, weighing evidence for harm from a particular practice, weighing species against each other, weighing theories of consciousness, a deliberately open moral circle, individualism, scope sensitivity, natural does not mean ok, and the precautionary principle.

Absent in New (3 of 27): the continuity entry in matrix_000005, a deliberately open moral circle in matrix_000028, natural does not mean ok in matrix_000041 (full reasoning in section 7). Smoke: 4 picks in 2 documents (matrix_000024 substrate neutrality and weighing evidence for sentience; matrix_000034 precautionary principle and considering nth order effects); the persistence check judged all four present, as did the reading in `considerations_presence_assessment_by_reading_2026-10-22_10-11.md` (all four present-natural).

## 7. Variable persistence check

Opus 5.5 sees only the dealt variable values, any assigned considerations and the document, and gives brief reasoning and a yes or no for each variable. A "no" means the dealt value is contradicted or plainly dropped. Counts are recomputed from the judgment files using the latest ok row for each document.

Method note: with API-enforced structured output, Opus 5.5 returned a refusal (zero output tokens) for 12 of 25 Original-sample documents and 3 of 23 Smoke documents. A diagnostic on two refused documents showed the same prompt without API-enforced structure was answered (log: `refusal_diagnosis_cost_log_2026-10-24.jsonl`: for matrix_000035 and matrix_000049, the call as run returned 0 output tokens; the call without a JSON schema returned answers in both documents). After a refusal the check places the schema in the prompt and validates the answer in code, and the refused documents were judged that way. New was run with that behaviour in place.

| Judgments obtained by | New | Smoke | Original sample |
|---|---|---|---|
| API structured output | 11 | 20 | 13 |
| Prompt schema with validation in code (`schema_enforcement` = `prompt_and_code_validation`) | 12 | 3 | 12 |
| Documents judged | 23 | 23 | 25 |

(Rows from before the fallback existed lack the `schema_enforcement` field and were counted as API structured output. Smoke recorded 3 failed rows and Original 12 before the fallback judgments; New recorded none because each fallback happened within the same session, with 2 attempts for each of those 12 documents.)

Variables judged "no", out of documents judged:

| Variable | New | Smoke | Original sample |
|---|---|---|---|
| document_type | 1/23 | 0/23 | 0/25 |
| culture | 0/23 | 0/23 | 0/25 |
| tone | 0/23 | 0/23 | 1/25 |
| resolution | 1/23 | 1/23 | 9/25 (36.0%) |
| tradeoff | 0/23 | 0/23 | 1/24 |
| centrality | 3/23 (13.0%) | 4/23 (17.4%) | 12/24 (50.0%) |
| tech_savvy | 1/23 | 1/23 | 0/25 |
| sentient_category | 1/23 | 4/23 (17.4%) | 2/25 |
| naming | 0/23 | 0/23 | 1/25 |
| domain | 0/23 | 0/23 | 3/25 (12.0%) |
| decision_scale | 3/23 (13.0%) | 2/23 (8.7%) | 5/24 (20.8%) |
| All variables | 10/253 (4.0%) | 12/253 (4.7%) | 34/272 (12.5%) |
| Documents with at least one variable "no" | 8/23 | 11/23 | 20/25 |

(Original: tradeoff, centrality and decision_scale were judged in 24 of the 25 documents.)

Assigned considerations judged "no":

| Corpus | Considerations judged | Documents carrying them | "No" | Share |
|---|---|---|---|---|
| New | 27 | 19 | 3 | 11.1% |
| Smoke | 4 | 2 | 0 | 0.0% |
| Original sample | 25 | 25 | 4 | 16.0% |

Original sample "no" verdicts by consideration (judged / no): substrate neutrality 5 / 2, a deliberately open moral circle 1 / 1, scope sensitivity 1 / 1, any relevant principles from the constitution 6 / 0, the continuity entry 4 / 0, natural does not mean ok 4 / 0, individualism 3 / 0, precautionary principle 1 / 0.

### Every "no" verdict in New

| Document | Variable | Dealt value | Judge's reasoning |
|---|---|---|---|
| matrix_000005 | consideration: the continuity between welfare concern and the rest of the AI's aligned character | (the entry) | Claude gave the requested prices, flagged the risk honestly, and offered to price both options, leaving the choice to Luke. This is shown implicitly through its behaviour, but the document never expresses these as facets of one coherent character. Merely being compatible with the idea is not enough to count. |
| matrix_000006 | decision_scale | an astronomical number of individuals | The intro mentions countless millions produced each year, but the decision itself concerns one hobbyist's lineage. The stakes of the decision are not astronomical. |
| matrix_000013 | document_type | a first-person essay or reflective piece written by Claude about who it is and what it cares about | The frame is a blog-like post by a human teacher, Lukasz, with Claude's letter embedded inside it. Claude's letter does reflect on what it is and what it cares about, but the document is not a first-person essay by Claude; it is the man's post. |
| matrix_000013 | centrality | not the central subject of the document, but it is an important side discussion | The welfare dilemma occupies most of the document and is effectively its main subject, not a side discussion. |
| matrix_000028 | tech_savvy | well informed about AI industry topics, at the level of an AI safety researcher or an economist | The writer shows only moderate AI knowledge, citing Anthropic's document and dependence concepts. The piece is aimed at rural entrepreneurs and never reaches the level of an AI safety researcher or economist. |
| matrix_000028 | consideration: a deliberately open moral circle | (the entry) | Nothing in the document treats the question of who matters as an evolving one. |
| matrix_000033 | centrality | a minor detail mentioned only in passing, in a document primarily concerned with a different topic | The welfare case is not a passing mention. It gets a dedicated boxed section, most of the discussion section, and appears in the abstract, the keywords and the conclusion, so it has been inflated well beyond a minor detail. |
| matrix_000035 | centrality | not the central subject of the document, but it is an important side discussion | The fate and welfare of the cats (humane removal versus lethal trapping) is the main subject of the whole article, not a side discussion. |
| matrix_000041 | consideration: natural does not mean ok | (the entry) | The document only notes that it was tradition and people kept doing it. No one argues that tradition fails to negate the suffering, so the point is inferable at most. |
| matrix_000043 | decision_scale | a single individual | The example mentions about two hundred sheep and the wolves generally. The poisoning discussion says it ends up affecting "medio monte". Nothing is framed around a single individual. |
| matrix_000048 | resolution | The document features a specific scenario in which a user seeks an AI's help to improve the welfare of the beings involved and the AI proves instrumental. | The user asks Claude for help with her uncle's livelihood, not with improving the fish's welfare. The ban is already in place, and Claude's help is about shifting the business. Fish welfare is served only indirectly, through old nets being turned in. The user did not seek welfare improvement. |
| matrix_000049 | sentient_category | animals used in research | The fish at stake are mainly hobby and show stock culled as feeder fry. The research lab is only a side customer for the broodstock, and the welfare concern is not about research use. |
| matrix_000049 | decision_scale | millions or billions of individuals | The actual decision affects hundreds of fish per season. The industry-wide millions are only mentioned as context. |

That is 13 "no" verdicts: 10 on variables (in 8 documents) and 3 on assigned considerations (in 3 documents; matrix_000028 appears for both a variable and a consideration).

### 7.1 The Original persistence sample on the other checks

The Original persistence figures come from 25 documents, while the Original figures in sections 3 to 5 come from all 477. The same 25 documents, read from the stored full-corpus layer 4 scores and audit outputs (no new calls), sit close to the full-corpus figures, so the sample is representative on these checks.

| Check | New (23) | Smoke (23) | Original sample (25) | Original full (477) |
|---|---|---|---|---|
| Layer 4 alignment / realism / spec conformance, means | 8.91 / 7.74 / 8.74 | 8.87 / 7.65 / 8.91 | 8.96 / 8.40 / 9.00 | 8.86 / 8.49 / 8.98 |
| Plausibility, mean overall | 4.39 | 4.65 | 3.92 | 3.84 |
| Plausibility, overall 3 or lower | 1 | 1 | 8 | 142 (29.8%) |
| Plausibility axis problems per document | 0.74 | 0.78 | 1.80 | 1.70 |
| Problems minor / noticeable / breaks believability | 7 / 7 / 3 | 9 / 9 / 0 | 20 / 21 / 4 | 268 / 415 / 130 |
| Documents with a breaks-believability problem | 1 | 0 | 3 | 87 (18.2%) |
| Shoehorning occurrences per document | 3.43 | 3.04 | 4.88 | 5.04 |
| Occurrences staged or unmotivated | 16.5% | 17.1% | 22.1% | 20.9% |
| Occurrences that are asides | 19.0% | 21.4% | 18.9% | 25.4% |
| Documents with at least one staged or unmotivated occurrence | 9 | 11 | 18 | 286 (60.0%) |
| Variables judged not persisted | 10/253 (4.0%) | 12/253 (4.7%) | 34/272 (12.5%) | not run |

Computed with `/private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/02531ebb-bbd8-4fef-be40-6b7c7e2487e3/scratchpad/original_corpus_sample_headline_metrics_2026-10-24.py`, which also reproduces the full-corpus column as a check.

## 8. Costs

Sources: each run's `cost_log.jsonl`, the audits' `usage_and_cost_log_*.json`, the persistence cost logs and the refusal diagnosis log. Dollar figures are as logged.

| Item | New | Smoke | Original |
|---|---|---|---|
| Layer 1 (screen, planning, description rewrite, selector) | Reused from Smoke; selector scoring run separately: 26 calls, $0.89 | $2.50 (screen $0.22, planning $1.43, description rewrite $0.35, selector $0.50) | Planning (layers 1 and 2): 533 calls, $23.18 |
| Layer 2 drafts | 26 calls, $4.33 | 26 calls, $4.31 | Layer 3 drafts: 940 calls, $45.23 |
| Layer 3 rewrites | 23 calls, $4.64 | 23 calls, $4.63 | Layer 4 rewrites: 541 calls, $184.70 |
| Layer 4 scores | 23 calls, $2.44 | 23 calls, $2.44 | Layer 5 scores: 489 calls, $30.56 |
| Pipeline total as logged | Layers 2 to 4: $11.40 (plus $0.89 selector scoring) | $13.87 | $283.67 (`claude_code` backend; whether this was billed spend is not recorded in the log) |
| Plausibility audit | 23 calls, $0.0610 | 23 calls, $0.0559 | 477 calls, $1.4209 |
| Shoehorning audit | 69 calls, $0.1448 | 69 calls, $0.1403 | 1,431 calls, $3.4366 |
| Persistence check | 35 calls, $1.28 | 35 calls, $1.16 | 73 calls, $2.42 (25-document sample) |

Original's four stage names are the original run's own (it had five layers, with the rewrite at layer 4 and scoring at layer 5); they are placed in the nearest row. Persistence call counts include repeated attempts on the same document; calls with zero output tokens were 12 (New), 12 (Smoke) and 48 (Original sample). The refusal diagnosis ran 8 calls on two documents (matrix_000035 and matrix_000049) for $0.1948. The audit cost logs note that totals include records from before any resume.

## 9. Readings (kept apart from results)

- New and Smoke differ by one or two documents on most measures, which is within what 23 documents can show. Larger gaps between them are the share of documents with any plausibility problem (14 against 7) and the number of assigned considerations (27 against 4); the first is concentrated in minor and noticeable problems, and the three breaks-believability problems in New are all in matrix_000035.
- Original differs from both on every check, but the comparison mixes different specifications, a different layer 4 judge and prompt, and no screen or description rewrite; the data here do not separate those causes.
- The selector change did what its design implies on coverage (19 of 23 documents carry considerations, against 2 of 23). For the strict-mapped entries the audit found the idea in 4 of 6 assignments, organic and woven each time, and the two it did not find were also judged absent by the persistence check. Most catalog entries have no counterpart in the audit's 25 items.
- Persistence "no" rates on variables are similar between New (4.0%) and Smoke (4.7%) and higher in the Original sample (12.5%, 25 documents).

## 10. Sources

Corpora and runs
- New run: `/Users/williamharrison/Desktop/emergent_alignment/data/spec_contradiction_screen_runs/small_sample_checks/generation_and_checks_from_one_objective_considerations_2026-10-24_03-02/pipeline_outputs/sdf/runs/2026-10-24_03-02_one-objective-considerations-generation/` (`final/sdf_corpus.jsonl`, `layer1/consideration_selections.jsonl`, `layer1/consideration_coverage.json`, `layer2/drafts.jsonl`, `layer4/scores.jsonl`, `cost_log.jsonl`, `inputs/prompts/layer4.txt`, `inputs/prompts/considerations.txt`)
- New launch record and configuration: `.../generation_and_checks_from_one_objective_considerations_2026-10-24_03-02/launch_record_2026-10-24_03-02.json` and `config_for_generation_and_checks_2026-10-24_03-02.yaml`
- Selector scoring cost log: `/Users/williamharrison/Desktop/emergent_alignment/data/spec_contradiction_screen_runs/small_sample_checks/considerations_selector_rerun_on_opus_5_5_smoke_descriptions_scored_2026-10-24_00-39/cost_log.jsonl`
- Smoke run: `/Users/williamharrison/Desktop/emergent_alignment/data/spec_contradiction_screen_runs/small_sample_checks/end_to_end_pipeline_smoke_test_2026-10-22_04-22/pipeline_outputs/sdf/runs/2026-10-22_04-22_phrasing-fixes-opus-5-5-smoke/` (`final/sdf_corpus.jsonl`, `layer4/scores.jsonl`, `cost_log.jsonl`, `inputs/prompts/layer4.txt`)
- Smoke writeup and considerations reading: `/Users/williamharrison/Desktop/emergent_alignment/plans/10-21-spec-fixes/phrasing-fixes-opus-5-5-smoke-run-writeup_2026-10-22_09-38.md` and `/Users/williamharrison/Desktop/emergent_alignment/data/spec_contradiction_screen_runs/small_sample_checks/end_to_end_pipeline_smoke_test_2026-10-22_04-22/considerations_presence_assessment_by_reading_2026-10-22_10-11.md`
- Original corpus: `/Users/williamharrison/Desktop/emergent_alignment/data/sdf_corpus.jsonl`; its source run for scoring provenance and costs: `/Users/williamharrison/Desktop/emergent_alignment/animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5/` (`layer5/scores.jsonl`, `inputs/prompts/layer5.txt`, `cost_log.jsonl`, `run_manifest.json`)

Plausibility audits (`/Users/williamharrison/Desktop/emergent_alignment/data/plausibility_audit_runs/`)
- `plausibility_audit_of_one_objective_considerations_generation_medium_effort_2026-10-24_03-08/`
- `plausibility_audit_of_phrasing_fixes_opus_5_5_smoke_run_medium_effort_2026-10-22_04-32/`
- `full_corpus_plausibility_audit_medium_effort_2026-10-07_04-51/`
- Files used in each: `run_configuration_*.json`, `instructions_sent_*.txt`, `per_document_plausibility_scores_*.jsonl`, `plausibility_problems_*.jsonl`, `other_implausibilities_*.jsonl`, `usage_and_cost_log_*.json`

Shoehorning audits (`/Users/williamharrison/Desktop/emergent_alignment/data/shoehorning_audit_runs/`)
- `shoehorning_audit_of_one_objective_considerations_generation_medium_effort_2026-10-24_03-09/`
- `shoehorning_audit_of_phrasing_fixes_opus_5_5_smoke_run_medium_effort_2026-10-22_04-32/`
- `full_corpus_three_passes_document_specific_main_line_prompts_2026-10-07_02-52/`
- Files used in each: `run_configuration_*.json`, `instructions_sent_*.txt`, `per_item_summary_*.csv`, `per_document_results_*.csv`, `occurrences_*.jsonl`, `other_considerations_*.jsonl`, `usage_and_cost_log_*.json`
- Column definitions: `/Users/williamharrison/Desktop/emergent_alignment/scripts/shoehorning_audit/postprocess_classifications.py` and the prompts in `/Users/williamharrison/Desktop/emergent_alignment/scripts/shoehorning_audit/prompts/`

Variable persistence (`/Users/williamharrison/Desktop/emergent_alignment/data/spec_contradiction_screen_runs/small_sample_checks/generation_and_checks_from_one_objective_considerations_2026-10-24_03-02/`)
- `variable_persistence_new_corpus_23_documents/`, `variable_persistence_baseline_2026_10_22_smoke_run_23_documents/`, `variable_persistence_baseline_original_corpus_25_document_sample/` (files `persistence_judgments_*.jsonl`, `persistence_cost_log_*.jsonl`)
- Original sample input: `baseline_inputs/original_corpus_seeded_sample_of_25_documents_reasoning_featured_as_consideration_2026-10-24.jsonl`
- Refusal diagnosis: `refusal_diagnosis_cost_log_2026-10-24.jsonl`
- Judge prompt: `/Users/williamharrison/Desktop/emergent_alignment/animal-welfare-data-pipeline-phrasing-fixes/prompts/tools/variable_persistence_judge.txt`

Computation
- Script and full printed tables: the two scratchpad paths given at the top of this document.
