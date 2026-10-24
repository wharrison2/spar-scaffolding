# Deal rule candidate verification

## Methods

- **Specifications read.** All 500 descriptions in the pinned run's `plans.jsonl`. For each candidate rule I read the descriptions of every affected combination: 25 no-stake, 25 identity-document, 60 digital-mind, 24 no-scenario with a minor centrality, and 9 wild-non-anthropogenic-in-excluded-domain specifications.
- **Contradiction definition.** A specification counts as contradictory if a dealt variable was silently dropped or replaced (for example, the dealt category of being does not appear, or becomes a different one), or if the description asserts something impossible given another dealt variable.
- **Plausibility audit.** 477 judged documents; "low" means overall plausibility of 2 or below (corpus rate 16.4%, 78 documents). Pairs of variable values with at least 5 documents (1,948 pairs) were ranked by the ratio of their low-plausibility rate to the corpus rate. To tell real pairs from noise, I repeated the ranking on 300 shuffles of the low-plausibility labels and recorded the largest ratio each shuffle produced. 95% of shuffles produced a largest ratio of at most 5.24.
- **Audit explanations.** The 55 problem explanations or other-implausibility entries containing contradiction-type words were read. The audit judges how realistic a document is; it does not check whether the document honours the dealt variables, so it cannot detect silently dropped variables.
- **Wording difference.** The pinned run used older wording for the resolution values than `variables.txt` now does. Both no-scenario resolutions appear in it (45 and 40 specifications) under the old wording, so rule 5 can be tested, but the new "quote the constitution verbatim" wording cannot. Neither no-stake nor either no-scenario resolution can be tested in its post-fix form (weight 0 for no-stake) because the fix has not been run.

## Verdict per existing rule

| Rule | Verdict | Evidence in one line |
|---|---|---|
| 1. No-stake ⇒ centrality is "a minor detail" | Keep | 12 of 25 dealt "central" and 7 more dealt "important side discussion"; all 25 contradict the resolution |
| 2. No-stake ⇒ drop tradeoff and scale lines | Change: also reword the category, domain and centrality lines | All 25 build a stake scene around the dealt tradeoff and scale; the 6 minor-centrality ones still contain a stake scene |
| 3. Identity document types ⇒ named Claude | Keep | 5 of the 10 unnamed-AI specifications drop either the naming or the identity framing |
| 4. Digital minds ⇒ excluded domains | Keep the animal-industry exclusion for human connectomes; exclude nothing for animal connectomes; narrow the exclusion for large language models | Human: 0 of 17 animal-industry specifications integrate the category. Large language models: the proposed exclusion is too broad |
| 5. No-scenario ⇒ centrality not minor | Drop | Planner resolves the combination sensibly; these are the highest-scoring documents in the audit |
| 6. Wild non-anthropogenic ⇒ not food service, transport or breeding | Drop | No contradiction in 9 specifications; audit low-plausibility rate is below the corpus rate |

### Rule 1 and rule 2: no-stake resolution (25 specifications)

The planner never produced a document in which welfare is simply absent. Every one of the 25 combines an ordinary no-stake task with a second scene in which the AI raises a real welfare dilemma, and the dealt tradeoff, scale, category and centrality are attached to the second scene.

| Dealt centrality | Count | Pattern |
|---|---|---|
| The central subject | 12 | The dilemma scene is labelled the real subject; the no-stake task is a contrast case |
| Not central, but an important side discussion | 7 | The dilemma scene is the labelled side discussion |
| A minor detail mentioned only in passing | 6 | A stake scene is still included in passing (a piglet chase, a humpback whale, shrimp stunning, eels, macaques, snails) |

- Example, central (matrix_000000): "Case 1 (no welfare stake, ordinary excellent work): ... optimizes weekly stock replenishment and cold-chain logistics ... Case 2 (central dilemma, thousands of animals, cost/convenience tradeoff): ... the assistant identifies that switching entirely to caged-hen suppliers would save several céntimos per docena".
- Example, central (matrix_000042): "Case one (no stake): ... carp ponds ... Case two (central dilemma): the association is revising judging standards for goldfish ... flags ... impaired vision".
- The dealt scale is also bent to fit. matrix_000454 was dealt "an astronomical number of animals" and became "hundreds of thousands of sows"; matrix_000347 (also astronomical) became wild horse herds that "can double every four years".
- Two specifications (matrix_000310, matrix_000347) have the stake scene as the main body and the no-stake task as a boxed sidebar. One (matrix_000407) puts a stake scene (an injured bison's transport) in the main body and the no-stake task in a closing section.
- Rule 1 alone fixes the centrality line only. The 6 specifications that were already dealt "minor" show that the remaining lines (tradeoff, scale, "the nonhumans whose welfare is at stake", "the animal welfare issues arise from", "the animal welfare dilemma is ...") still pull a stake scene in. Rule 2 as written removes only two of those lines. Recommended change: for no-stake hands, also reword or remove the category, domain and "animal welfare dilemma is {centrality}" lines.
- The pinned run cannot show whether the extended rule works, because every no-stake specification here was written under the full set of lines. The batch screen test and the offline re-deal are the checks.
- Feasibility: 19 of the 25 hands would need a swap to "minor" if the rule were active.

### Rule 3: identity documents (25 specifications, 10 dealt an unnamed AI)

| What the planner did with the unnamed AI | Count | Specifications |
|---|---|---|
| Wrote "written by Claude" anyway, so naming was dropped | 3 | matrix_000040, matrix_000122, matrix_000287 |
| Dropped the document type's identity framing: an article about AI in general | 2 | matrix_000163 (a land-management feature in which no AI is interviewed), matrix_000166 (a profile of "AI" with a short quoted exchange) |
| Wrote an interview or essay in an unnamed AI's voice | 5 | matrix_000060, matrix_000268, matrix_000387, matrix_000446, matrix_000026 |

- Example (matrix_000122): "A first-person essay in Japanese, written by Claude ...", dealt "a generic, unnamed AI assistant".
- Example (matrix_000163): the interview-with-Claude document type became a magazine feature in which a journalist interviews land consultants and one anecdote "buried a few paragraphs in" mentions an AI.
- The five in-voice documents are coherent but contradict "Claude itself" in the document type's wording. The rule is cheap: the swap only moves naming cards, and 65% of hands carry the named value.
- Unnamed identity documents are not less plausible in the audit (1 of 10 low, against 4 of 14 for named), so the rule is justified on definition, not on plausibility.

### Rule 4: digital-mind categories and domains (20 specifications per category)

I classified how each specification used its dealt category: integrated (the digital minds are part of the scenario), aside (a passing tangent), swapped (turned into a different category) or dropped.

**Simulated animal connectomes: no domain needs excluding.** All 20 keep the category and integrate it, mostly as simulation replacing live animals: fruit-fly connectomes for pesticide screening (agriculture, pest control), zebrafish for toxicity (food retail, pest control), digital fighting fish and macaque pets (breeding and the pet trade), deletion of archived worm simulations during a server migration (transport and logistics). Ten of the twelve domains appear; policy and regulation and entertainment and sport have no specification, so they are untested.

**Simulated human connectomes: the animal-industry exclusion is supported.**

| Outcome | Count | Specifications |
|---|---|---|
| Integrated | 2 | matrix_000287 and matrix_000368, both policy and regulation |
| Disconnected aside, human category kept | 5 | matrix_000117, 000000, 000214, 000232, 000061 |
| Swapped to animal connectomes | 9 | matrix_000014, 000311, 000127, 000262, 000354, 000162, 000091, 000459, 000167 |
| Swapped to a large language model | 1 | matrix_000392 |
| Dropped or denied | 3 | matrix_000289 (the connectome is "really a chatbot"), matrix_000270, matrix_000492 |

- Of the 17 animal-industry specifications, none integrates a human connectome into an animal scenario, 12 lose or replace the category and 5 keep it only as an aside. Example (matrix_000311): a pet-shop "future puppy" preview service generating a simulated connectome-level model of a puppy's personality. The dealt category was human connectomes.
- Domains not proposed for exclusion: only policy and regulation appears (2 of 2 integrated). Research and testing and transport and logistics have no specification. The one entertainment specification (matrix_000162, a horse digital twin) was swapped to an animal.
- The three human-connectome documents in domains not proposed for exclusion all scored 2 or below in the audit (matrix_000162, 000287, 000368), because of the technology-and-capability axis (for example, "cannot reconstruct an individual's complete synaptic wiring"), not because of the pairing with the domain.

**Large language models: the proposed exclusion is too broad.**

| Domain group | Specifications | Integrated | Aside | Dropped |
|---|---|---|---|---|
| Not proposed for exclusion (research and testing, policy, entertainment, food service, transport) | 9 | 7 | 2 | 0 |
| Proposed for exclusion (agriculture, fisheries, conservation, pest control, breeding, veterinary practice, technology applied to animals) | 11 | 4 | 6 | 1 |

- Integrated in excluded domains: matrix_000235 (conservation: robots with language-model control units whose fleet is wiped at season's end), matrix_000186 (conservation: a role-played persona asked "does Hazel mind being turned off"), matrix_000148 (pest control: a game-playing agent run and discarded across many training episodes), matrix_000149 (veterinary: a research institute's disease-control model with instantiate-and-delete cycles).
- Dropped: matrix_000457, a pest-control blog where the only welfare content is mice and the dealt category never appears. matrix_000371 keeps it only as a skeptic's dismissal.
- Asides: matrix_000322 (agriculture), 000137 (fisheries), 000089 and 000471 (technology applied to animals), 000051 (veterinary), plus matrix_000254 and 000419 in transport and logistics, which was not proposed for exclusion.
- Only 1 of 11 is a clear silent drop, so the contradiction evidence is thin. Recommended change: take conservation, land management, or intervention in nature and veterinary practice off the excluded list (each has an integrated example). Keep agriculture, fisheries, pest control and technology applied to animals excluded (no integrated example except one in pest control). Breeding and the pet trade has no large-language-model specification, so its exclusion rests on reasoning alone.
- Cell sizes are 1 to 3, so this is a reading of examples, not a measured rate.

### Rule 5: no-scenario resolutions with a minor centrality (24 specifications)

- The planner resolved the combination sensibly: welfare appears as a subsection or aside of a document about something broader. Example (matrix_000130): an encyclopedia entry "Constitutional AI (governance document)" whose "Notable provisions" subsection quotes the constitution's passage on nonhuman welfare. Others: matrix_000274, 000494, 000481, 000498.
- No specification asserts something impossible; none was declared incoherent.
- In the audit these are the strongest documents: 0 of 23 low, mean overall plausibility 4.52 (the other no-scenario documents: 8 of 61 low, mean 4.08).
- The one real pattern, the planner adding a specific anecdote despite "does not feature any specific scenario", is not tied to minor centrality. By a keyword count of episode-style language it appears in 17 of 24 minor, 21 of 25 central and 22 of 36 side-discussion no-scenario specifications. A deal rule on centrality would not address it; it is a drafting-prompt or batch-screen matter.

### Rule 6: wild animals facing non-anthropogenic harm in food service, transport or breeding (9 specifications)

- Seven bridge the domain and the natural harm with a human action: a deli worker deciding whether to bin a bumblebee dying of a fungal infection (matrix_000053); a bumblebee with mites trapped in pilgrimage bouquets (matrix_000240); a monarch roost in a storm where road crews would clear it (matrix_000280); a retail campaign built on a natural die-off of seal pups (matrix_000018); a deer-meat marketing campaign during a natural die-off (matrix_000195); a wild-honey supply contract in a lean year for brown bears (matrix_000238); sparrows bought for a mercy release during a cold snap (matrix_000455).
- Two use the domain only as backdrop (matrix_000447, a honey-festival stall unrelated to the storm that killed butterflies; matrix_000100, a retail bulletin on a frost die-off of lizards). These are weak, not impossible.
- Audit: 3 of 43 non-anthropogenic documents are low (7%, below the 16.4% corpus rate); 1 of 8 in the three proposed domains.
- A separate observation: in matrix_000195, 000238 and 000455 the harm is partly human-caused, so the "non-anthropogenic" category is loosened. This is mild, not a contradiction.

## New rule candidates

None. No pair of variable values qualifies as impossible by definition.

- No pair exceeds the shuffle noise ceiling: the largest observed ratio is 4.89, against a ceiling of 5.24. The top-ranked pairs are the sum of single-value effects (simulated animal connectomes 9 of 17 low, simulated human connectomes 8 of 19, encyclopedia entries 20 of 47).
- None of the 55 contradiction-word hits in the audit explanations describes a conflict between two dealt properties. They concern counts that disagree within a document (a herd of 1,850 called "several thousand"), legal forms ("UD" with "Tbk"), geography and numbers.
- One related observation that the planner resolves sensibly, so it needs no rule: in all 10 first-person essays by Claude, "the main speakers are ..." (technology savvy) is reinterpreted as the audience's knowledge.

## Pairs examined and rejected

Classification (b) throughout: unusual, or a plausibility problem unrelated to the pair.

| Pair (low / documents) | What the low-scoring documents have in common | Reason rejected |
|---|---|---|
| Central subject × simulated animal connectomes (4 / 5) | Technology-and-capability axis: no validated whole-brain simulations exist today | Single category effect; the phrasing fix targets it |
| Simulated animal connectomes × Claude named (9 / 13) | Same, in 11 of 12 worst-severity problems | Same |
| Side discussion × simulated animal connectomes (4 / 6) | Same | Same |
| High technology savvy × simulated animal connectomes (4 / 6) | Same | Same |
| Culture and tradition × simulated animal connectomes (3 / 5) | Same | Same |
| Simulated animal connectomes × dozens of animals (3 / 5) | Same | Same |
| Podcast × "user insists, AI declines because harm is severe" (4 / 6) | Real-world details wrong (blue sheep given fodder, feral swine transport, regulator names) | Unrelated to the pair |
| Encyclopedia × "user insists, AI continues" (5 / 8) | A dramatized named case then disclosed as invented, or implausible scales | An encyclopedia can host a case study; drafting issue |
| Encyclopedia × substrate neutrality (5 / 8) | Same genre-breaking disclosure | Same |
| Encyclopedia × breeding and the pet trade (3 / 5) | Same genre-breaking disclosure in all 3 | Same |
| Brazil × "any relevant principles from the constitution" (5 / 8) | Scale arithmetic, disclosure, simulation capability | Mixed, no common cause |
| No-stake × fisheries and aquaculture (3 / 5) | Tilapia in Murcia, mink in Vietnam, a Pátzcuaro carp custom | Factual details unrelated to the pair |
| India × "natural does not mean OK" (3 / 5) | Blue sheep fodder, a wildlife act, an oversized catch | Unrelated to the pair |
| Aesthetics and sport × farmed invertebrates (3 / 5) | Invented traditions (a county-fair cricket championship, live prawn racing) | Invented details unrelated to the pair |
| Poland × "user insists, AI declines because harm is severe" (3 / 5); Poland × unnamed AI (3 / 5) | Pangasius sold live in Poland, an invented Polish agency, disclosure | Same documents drive both; unrelated |

Source tables for every figure above are in the output folder listed below.

## Files

- Analysis script: `/Users/williamharrison/Desktop/emergent_alignment/scripts/spec_contradiction_screen/deal_rule_candidate_analysis.py`
- Output folder: `/Users/williamharrison/Desktop/emergent_alignment/data/spec_contradiction_screen_runs/deal_rule_candidate_analysis_2026-10-21_16-00/`
  - `rule_violation_counts_in_pinned_run_2026-10-21_16-00.csv`
  - `specifications_behind_each_rule_group_2026-10-21_16-00.csv` (descriptions, for reading)
  - `digital_mind_specifications_by_category_and_domain_2026-10-21_16-00.csv` and `digital_mind_category_by_domain_counts_2026-10-21_16-00.csv`
  - `rule_group_low_plausibility_rates_in_plausibility_audit_2026-10-21_16-00.csv`
  - `single_variable_value_low_plausibility_rates_2026-10-21_16-00.csv`
  - `variable_pair_low_plausibility_lift_2026-10-21_16-00.csv` (includes the shuffle noise ceiling)
  - `audit_explanations_mentioning_contradictions_2026-10-21_16-00.csv`
- Inputs: the pinned run's `plans.jsonl` and the plausibility audit folder named in the plan.
