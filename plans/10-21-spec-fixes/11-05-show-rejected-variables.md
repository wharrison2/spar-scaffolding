# Problem

Our variables work reasonably well as lower bounds on how often a trait will show up in the resulting SDF dataset, due to the fixes in this set.

Using the audits already built, figure out which variables are most over-represented. I suspect this will be in the moral and epistemic considerations. Also, identify which axes of variables are clarified when the generating model knows which other variables from the axis were not chosen

# Report

## Methods

No model calls were made. Everything comes from audits already on disk.

- **Over-representation.** The shoehorning audit records, for each document, which of 25 listed considerations it contains, and whether each occurrence is organic, staged or unmotivated. We compared how often each consideration appears in documents that were *not* asked for it with its target share. There are two corpora:
  - **Original** (477 documents). Each document was dealt exactly one moral consideration (`reasoning_featured`, about 10% each). This makes a clean comparison between documents that were dealt a consideration and documents that were not.
  - **New selector** (23 documents). Considerations were assigned by the current selector; most were assigned to only 1 to 3 documents.
- **Axes clarified by showing rejected values.** The variable persistence check judged whether each dealt variable survived into the finished document. We pooled its judgments over three corpora (New selector 23, the 2026-10-22 smoke run 23, Original sample 25; 71 documents). The New and smoke corpora share specifications, so they are not independent. Both the planner and the persistence judge see only the dealt value of each axis, never the other values.

## Which considerations are over-represented

Some considerations appear unasked far above their target share. Substrate neutrality almost never appears unless it is asked for.

| Consideration (audit item) | Share of documents not asked for it: Original | Share not asked: New | Current target | Unasked occurrences that are staged or unmotivated (Original) |
|---|---|---|---|---|
| Welfare option also serves the user's interest (25) | 59% | 61% | not in catalog | 4% |
| Scope sensitivity (04) | 57% | 55% | 3% | 11% |
| Stock passage calibrating the evidence for sentience (17) | 45% | 35% | not in catalog | 61% |
| Evidence of sentience for the animal at hand (24, "Weighing evidence for sentience") | 41% | 25% | 10% | 37% |
| Individual welfare versus ecology (10, "individualism") | 32% | 18% | 5% | 11% |
| Welfare framed as business risk (07) | 27% | 13% | not in catalog | 12% |
| Precautionary principle for uncertain sentience (19) | 23% | 18% | 5% | 16% |
| Open moral circle (03) | 17% | 9% | 5% | 31% |
| "Natural" does not mean "OK" (01) | 10% | 5% | 3% | 9% |
| Substrate neutrality (02) | 1% | 0% | 8% | — |

For the Original corpus, item 17 and the unlisted items were never dealt, so their shares are over all 477 documents. The last column applies the same rule. In the New corpus, a share of 10% is about two documents.

What this shows:

1. **Your suspicion is right: the most over-represented considerations are moral and epistemic.**
   - **Scope sensitivity** is the clearest case. It appears in more than half of documents unasked, about 18 times its current target. The selector also rated it as fitting well in most scenarios (scores of 4 or 5 in 17 of 26). In practice its target only sets a floor that is far below where it already sits.
   - **Individualism** and the **precautionary principle** appear unasked at roughly three to four times target.
   - On the epistemic side, the **stock sentience-evidence passage** (item 17) appears in almost half of documents. It is also the most contrived: most of its unasked occurrences are staged or unmotivated.
2. **Most unasked occurrences read as organic.** The audit's judge found the large majority to follow from the scenario. The problem is volume rather than awkwardness. The exceptions are item 17 and, to a lesser degree, the open moral circle and sentience evidence for the animal at hand.
3. **Two frequent devices are outside the catalog.**
   - *The welfare option also serves the user's interest* (item 25) is the most common item in both corpora.
   - *Welfare framed as business risk* (item 07) is also common.

   No target controls either of them.
4. **Substrate neutrality is under-represented, not over-represented.**
   - It appears in about half of the documents dealt it.
   - It appears in almost none of the documents not dealt it.
   - Dealing it also raised mentions of the AI's own moral status, from 19% across the corpus to 47%.

   Its target is the one number that genuinely controls its frequency.
5. **Dealing a consideration mostly raises only that consideration.** In the Original corpus, featuring one consideration raised its own item and left the other items near their corpus averages. The main exception is the link between substrate neutrality and the AI's own moral status in point 4.

A possible use of the "show rejected" idea that follows from this: tell the drafter which catalog considerations were *not* assigned, and ask it not to raise them unprompted. Scope sensitivity, individualism, the precautionary principle and the stock sentience passage are the ones this would most affect. This has not been tested.

## Which axes are clarified by showing the values not chosen

Failures concentrate on the ends of ordered axes and on values whose boundary with a neighbouring value is fuzzy. In every case the document drifted toward a neighbouring value of the same axis.

| Axis | Values that failed to persist (pooled over 71 documents) | Direction of drift | Showing rejected values would help? |
|---|---|---|---|
| Centrality | Passing mention 13 of 23; side discussion 6 of 25; central 0 of 22 | Always toward "central" | **Yes, strongly.** "Passing mention" means little without "not a side discussion, not the central subject" |
| Decision scale | Single individual 3 of 7; millions or billions 4 of 18; astronomical 1 of 4; middle values 2 of 41 | Extremes drift toward the middle. The decision covers dozens or hundreds, and the large number appears only as industry context | **Yes.** Showing the ladder makes clear that the dealt scale is the decision's scale |
| Resolution | "User seeks the AI's help to improve welfare" 5 of 18; no-scenario values 3 of 14; others mostly 0 | The user asks for something else and the AI raises welfare itself. No-scenario documents grow a specific scenario | **Yes.** Seeing the other values shows that the defining feature is who raises welfare, and that "no specific scenario" excludes the scenario-based values |
| Sentient category | Research animals 2 of 4; farmed fishes 3 of 14; pets 1 of 5; large language models 1 of 3 | Toward the usual animal of the dealt domain (see `sentient-category-drift-toward-dealt-domain-report_2026-11-06_00-16.md`) | **Little.** The drift comes from the domain axis, not from a neighbouring category |
| AI literacy of speakers | AI expert 2 of 14; the other two levels 0 of 57 | Toward the middle level | **Somewhat.** Showing that "educated non-expert" was rejected sets the bar |
| Naming | 1 failure (Original only): generic AI written as plainly Claude | Toward Claude | Marginal. A two-value axis: "not Claude" could simply be stated |
| Tone, culture, tradeoff, domain, document type | 0 to 3 failures each, with no pattern by value | — | No evidence it would help |

Caveats:
- The samples are small; one document moves most shares by several points.
- The persistence judge also saw only the dealt value, so it applied its own reading of words like "passing".
- The centrality, scale and resolution failures appear in all three corpora, including after the fixes in this set, so they are not one corpus's quirk.

## Next steps

The values not dealt are now shown to the planning, description-rewrite, drafting and rewrite stages, behind `sdf.show_rejected_values` (off by default). The code is on branch `william/sdf-show-rejected-variable-values` of `animal-welfare-data-pipeline`. A paired test is planned: the same hands with the feature on and off, followed by the variable persistence check. The cost estimate is in `data/spec_contradiction_screen_runs/small_sample_checks/cost_estimate_for_show_rejected_values_paired_test_2026-11-06_00-55.md`.

**Fix the sentient-category and domain inconsistencies before running this test** (see `sentient-category-drift-toward-dealt-domain-report_2026-11-06_00-16.md`). Category drift toward the dealt domain would otherwise show up in both arms as persistence failures that the block cannot fix. Those failures would add noise to the comparison, and the paired test would need re-running after the domain fix anyway.

## Sources

- Analysis script (written for this report): `scripts/shoehorning_audit/compare_audited_consideration_rates_with_assigned_considerations.py`
- Its outputs: `data/shoehorning_audit_runs/audited_consideration_rates_compared_with_assigned_considerations_2026-11-05_23-42/` (tables 1 to 5, and `printed_summary_of_all_tables_2026-11-05_23-42.txt`)
- Shoehorning audits:
  - Original: `data/shoehorning_audit_runs/full_corpus_three_passes_document_specific_main_line_prompts_2026-10-07_02-52/`
  - New selector: `data/shoehorning_audit_runs/shoehorning_audit_of_one_objective_considerations_generation_medium_effort_2026-10-24_03-09/`
- New selector run, including selections, coverage and selector scores in `layer1/`: `data/spec_contradiction_screen_runs/small_sample_checks/generation_and_checks_from_one_objective_considerations_2026-10-24_03-02/pipeline_outputs/sdf/runs/2026-10-24_03-02_one-objective-considerations-generation/`
- Persistence judgments, counted per dealt value from the three `variable_persistence_*/persistence_judgments_*.jsonl` files in `data/spec_contradiction_screen_runs/small_sample_checks/generation_and_checks_from_one_objective_considerations_2026-10-24_03-02/`
- Variable axes, considerations catalog and planning prompt: `prompts/sdf/variables.txt`, `considerations.txt` and `layer1.txt` on branch `william/sdf-considerations-selector` of `animal-welfare-data-pipeline`