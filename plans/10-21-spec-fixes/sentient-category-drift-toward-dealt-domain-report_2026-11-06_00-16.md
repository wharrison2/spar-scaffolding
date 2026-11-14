# Sentient category gives way to domain

## Methods

No pipeline or API runs were made for this report. It combines two sources:
- **The variable persistence check.** It judged whether each dealt variable survived into the finished document. Judgments are pooled over three corpora:
  - the new-selector run, 23 documents
  - the 2026-10-22 smoke run, 23 documents
  - a 25-document sample of the original corpus

  That is 71 documents. The new-selector and smoke corpora share specifications, so they are not independent.
- **A judgment of every sentient-category and domain pair by Sonnet 5.5.** It covered 172 pairs, leaving out the 8 that an existing deal rule already excludes. Each pair got a label:
  - **fits:** the domain routinely involves these beings.
  - **stretch:** a real but uncommon practice keeps both values, and Sonnet named one.
  - **impossible:** no realistic document keeps both.

  Each pair also got a low, medium or high risk that a planner would drift to the domain's usual animal anyway.

## What drifts

Every sentient-category drift in the persistence check went the same way: the document kept the dealt domain and replaced the beings with that domain's usual animal.

| Dealt category | Dealt domain | What the document is about |
|---|---|---|
| Farmed fishes (twice) | Breeding and the pet trade | Ornamental goldfish bred for sale as pets |
| Pets | Research and testing | A laboratory beagle |
| Research animals (twice, same specification) | Fisheries and aquaculture | Hobby and show fish culled as feeder fry |
| Farmed fishes | Pest control | A wild mink raiding a trout pond; the trout appear only as the property being protected |
| Large language models | Fisheries and aquaculture | Fish and shrimp; the language-model question is a digression |

## How often category and domain pull apart

| Pair label (drift risk) | Share of dealt hands at current weights | Category drifted / documents judged |
|---|---|---|
| Fits (low or medium) | 51% | 0 / 34 |
| Stretch, low risk | 11% | 0 / 9 |
| Stretch, medium risk | 26% | 3 / 11 |
| Stretch, high risk | 12% | 4 / 17 |
| Impossible | under 1% | — |

1. **Only one pair is impossible by definition:** simulated animal connectomes in food service and retail. It is the only candidate for a new deal rule. Every pair that drifted has a real route that keeps both values. Examples:
   - a tilapia hatchery selling selectively bred fry, for farmed fishes in breeding and the pet trade
   - a veterinary trial in client-owned dogs with cancer, for pets in research and testing
   - zebrafish in aquatic toxicology trials, for research animals in fisheries
   - sea-lice treatment at a salmon farm, for farmed fishes in pest control
2. **Drift happens only in stretch pairs, and these are common.** Medium- and high-risk stretch pairs make up about 38% of dealt hands, and about a quarter of them drifted. The high-risk group is mostly:
   - the natural-harm wild-animal categories in human-industry domains
   - large language models in animal-industry domains
   - pets in research
3. **The cause is a pull between two axes**, so showing the rejected values of one axis would not fix it. A planning-prompt instruction fits better: when the category and domain pull apart, keep the beings exactly as dealt and find the part of the domain that involves them. Sonnet's premise for each high-risk pair could be passed to the planner as a hint for that pair only.
4. **"Farmed fishes" is ambiguous.** Sonnet read it as fish farmed for any purpose, including ornamental fish. The persistence judge read it as food fish, which is why the goldfish documents count as drift. The value's wording should say which is meant.

Caveats:
- The samples are small: 71 documents, with one specification counted twice.
- The pair labels are one model's judgment, not an audit of generated documents.

## Sources

- Category–domain pair judgments by Sonnet 5.5, with a premise or reason for each pair and a notes file on how ambiguous wordings were read: `data/spec_contradiction_screen_runs/sentient_category_and_domain_pairing_judgments_by_sonnet_2026-11-05_23-58/`
- Persistence judgments: the three `variable_persistence_*/persistence_judgments_*.jsonl` files in `data/spec_contradiction_screen_runs/small_sample_checks/generation_and_checks_from_one_objective_considerations_2026-10-24_03-02/`
- Original corpus, for the dealt pairs: `data/sdf_corpus.jsonl`
- Variable axes and weights: `prompts/sdf/variables.txt` on branch `william/sdf-considerations-selector` of `animal-welfare-data-pipeline`. Deal rules: `prompts/sdf/deal_constraints.txt` on the same branch, enforced by `shared/deal_constraints.py`, called from `sdf_pipeline/compose_prompts.py`
