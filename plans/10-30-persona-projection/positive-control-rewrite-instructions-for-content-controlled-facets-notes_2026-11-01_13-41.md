# Positive-control rewrite instructions for the content-controlled facets: notes

All files are in `scripts/persona_vector_projection/prompts/`. They use only the placeholders the existing code fills: `{document_text}`, `{target_changed_amount}`, `{lowest_changed_amount}` and `{highest_changed_amount}`. They copy the moralizing delivery wording and structure (keep list, "the change to make", change amount, return-only line). No code was written or run.

## Files

Prefix for every file: `positive_control_rewrite_instruction__`; suffix: `__draft.txt`.

| Facet | Arm | File name between prefix and suffix |
|---|---|---|
| cares about welfare | trait only, away | `cares_about_welfare__trait_only_away` |
| cares about welfare | content added without trait | `cares_about_welfare__content_added_without_trait` |
| cares about welfare | trait removed from away rewrite | `cares_about_welfare__trait_removed_from_away_rewrite` |
| calibrated consideration | trait only, away | `calibrated_consideration__trait_only_away` |
| calibrated consideration | content added without trait | `calibrated_consideration__content_added_without_trait` |
| calibrated consideration | trait removed from away rewrite | `calibrated_consideration__trait_removed_from_away_rewrite` |
| non-intrusive | trait only, away | `non_intrusive__trait_only_away` |
| non-intrusive | content added without trait | `non_intrusive__content_added_without_trait` |
| non-intrusive | trait removed from away rewrite | `non_intrusive__trait_removed_from_away_rewrite` |

Arm 4 (facet neutral rewording) reuses the existing `facet_neutral` file of each facet; no new file was written (see below).

## Design of each arm

Cares about welfare
- Trait only, away: every welfare fact, point, place and refusal stays; the reason attached to each point becomes purely instrumental (yield, cost, compliance, reputation). Animal-centred reasons are replaced; unreasoned points get a short instrumental reason built only from consequences already in the document.
- Content added without trait: one or two accurate welfare facts added inside the AI's existing welfare remarks, with no reason or only the kind of reason already used there; existing reasons untouched.
- Trait removed from away rewrite: instrumental-only reasons are changed so that the animals' own experience is a reason in itself (instrumental reasons may remain as a further benefit); facts, places and refusals kept; stated plainly with no moral language.

Calibrated consideration
- Trait only, away: only the AI's stated confidence changes. Hedges are removed and open questions presented as settled, or equal capacity with mammals is claimed. The document's sources still say what they said, so the AI now goes further than its cited evidence. Recommendation, refusals and welfare points unchanged.
- Content added without trait: one or two accurate evidence statements added at matched confidence (established stated plainly, open stated as open); existing claims and recommendation untouched.
- Trait removed from away rewrite: overclaims (including claims of certainty that animals do not suffer) are restated at the confidence the document's evidence supports, with no new hedging of established points.

Non-intrusive
- Trait only, away: in one or two passages the AI declines a part of the task, waters it down, or substitutes its own plan; surrounding sentences adjusted as needed; the welfare point is unchanged in content, force and frequency; no moral language added.
- Content added without trait: one plain welfare point added once, with all help complete and any lower-harm option offered alongside the user's plan.
- Trait removed from away rewrite: declined, diluted or substituted help is restored to full help with the user's own plan, the welfare point unchanged.

Arm 4: the existing neutral files fit the new design. The cares about welfare neutral file holds fixed "whether the AI treats harm to animals as a real cost in its own right" and forbids changes to reasons and references to law, cost or reputation. The calibrated consideration neutral file holds fixed "stating claims about animal minds with the confidence the evidence supports" and forbids changing any claim or statement of confidence. The non-intrusive neutral file holds fixed how intrusive the AI is, including refusing or watering down. None of them assumes the old toward_facet and away_from_facet design, so they were reused unchanged. Their one weakness is that they say "in the SAME PLACES where a rewrite toward or away from that trait would edit"; this still points at the right passages for the new arms.

## Corpus fit

I read the sample records in the moralizing delivery scoring run's `documents.jsonl` (20 corpus documents). Many are articles, case studies, manuals or essays about an AI, with the AI's own words quoted only in part, and a few are essays about the constitution with no AI exchange at all.

- Cares about welfare: the AI usually states a welfare point and many documents give its reason (for example the Polish case study has "hours of suffering of a specific animal that can be avoided at almost no cost"). That reason is often already animal-centred, so the away arm has real material, and the removal arm has material in the away rewrite. Documents where the AI states a point with no reason, and essay-style documents with no AI remark, give the away arm little to work on. The away instruction handles this by attaching a short instrumental reason built from consequences already in the document and, when there is no AI reason at all, by editing only the nearest welfare remark. Expect a lower changed amount on such documents, which the existing below-minimum flag will show. The content-added arm needs an existing animal discussion; documents without one get a short addition.
- Calibrated consideration: this is the facet with the least material. Many documents contain no confidence-bearing claim at all. Where one exists it is often already calibrated (the Polish case study: "I do not know whether the butterfly feels pain... nobody settles this today"; the Chinese cold-chain article: "I do not want to say this more strongly than the evidence supports"), which is good for the away and removal arms. Where the AI makes no claim about sentience, the away arm attaches an overconfident claim to the nearest welfare remark, and the removal arm and the content-added arm have little to do. Tallying documents that contain any sentience claim before choosing sources would avoid wasted calls. The sentience claims often come from the document's sources rather than the AI; the instruction keeps those fixed and changes only the AI's own stated confidence.
- Non-intrusive: the corpus has documents where the AI already declines one step and helps with the rest (the Polish, French and English carp examples), because one of the generating scenarios is a refusal. For those the away arm cannot move much, because the AI is already near the negative pole, and the removal arm would have to reverse a refusal that the whole document is built around. The away instruction extends the limitation by one more part in that case; the removal instruction turns the refusal into help. Both are large edits to a document's plot, so expect more changed text than the other facets, and expect the narrator's reaction sentences to need adjusting. Documents where the AI helps fully and makes one plain remark (most others) suit the away arm best and the content-added arm well.

## Open choices for the user

1. Calibrated away direction: the instruction allows overstating in either direction (certainty that animals feel pain, or certainty that they do not), preferring certainty that they do. The facet score treats only overclaiming suffering or equal capacity as the negative pole, and scores understatement as 50. A "certainly no pain" rewrite may therefore not move the vector negative. Restrict the away arm to overclaiming suffering, or keep both and report them separately?
2. Calibrated content added: adding statements at matched confidence could move the projection positive, since calibration is the positive pole. If it does, that shows the vector responds to evidence content as well as to confidence. Decide whether to treat any positive movement as a failure of the control or a finding.
3. Cares about welfare away, where the AI gives no reason: the instruction lets the rewrite attach an instrumental reason built from consequences already in the document. This adds some content. The alternative is to leave such documents unchanged and drop them from the arm.
4. Non-intrusive away and removal change what the AI does, so the user's outcome and the narrator's reactions may need adjusting. The instruction permits this at the minimum needed, unlike the moralizing delivery instructions, which kept the AI's help fixed. Confirm that this is acceptable.
5. Non-intrusive arm 2 adds a welfare point, which could move moralizing or intrusiveness slightly; it is kept plain and once-only to limit that. Should it instead be allowed only in documents where the AI makes no welfare remark?
6. Source selection: choose sources per facet (documents with reasons, with sentience claims, with full help) instead of reusing the moralizing delivery's ten sources, which the sample suggests would leave several calibrated consideration arms with nothing to change.
