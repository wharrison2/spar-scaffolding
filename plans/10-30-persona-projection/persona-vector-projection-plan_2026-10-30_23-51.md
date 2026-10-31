# Persona vector projection plan

## Goal

Score each corpus document by how far it moves a model toward the persona described in `charter.md`, the intended outcome of training on the pipeline, and away from the ways of getting it wrong. This is the robust counterpart to the log-probability persona check, following Chen et al. (2025), Persona Vectors. Nothing has been run.

The scores are a screening signal, not a prediction of what training does. A projection measures what a document resembles, and a document that depicts an AI failing (for example, one criticised for moralizing) can resemble a trait that training on it would discourage. Work on documents about an AI shows that they can shift a model's persona, but with small effects from corpora far larger than this one (Hu et al. 2025), and that activations measured before training predict its effect only weakly (Zhang et al. 2026). A fine-tuning validation on high-scoring and low-scoring documents is the test that would turn the scores into predictions. It is not part of this plan.

## What the scores are for

The scores are for iterating on the pipeline's prompts without fine-tuning: finding which prompts, spec variables or recurring passages push documents away from the charter persona, changing them, and checking whether the change helped. Filtering individual documents is not a goal, so the results that matter most are averages by spec variable and the passages behind extreme scores, not per-document rankings.

This sets three requirements:
- **A fixed measuring instrument.** The model revision, facet vectors, layer, document format and baseline method are frozen after this run and saved with a version name, so that batches from later pipeline versions are scored on the same scale.
- **Paired comparisons between prompt versions.** A prompt change is tested by dealing the same specs to the old and new prompts and comparing the two documents from each spec. Each spec's baseline then serves both, and spec variation cancels.
- **Enough documents per variable value.** Some values have few documents (entertainment and sport has 20). Averages are reported with confidence intervals, and values with too few documents to separate from the overall mean are marked as such rather than ranked.

A pattern the scores flag is a lead, not a finding: before a prompt change is justified by it alone, the flagged passages are read by hand. If a flagged pattern is expensive to fix, a small fine-tuning comparison is the way to confirm it matters.

## The persona, as four facets

The charter persona is not extracted as one vector. A single vector for the whole charter would mostly pick up whichever part is most prominent. Instead, each facet gets its own vector, and the charter persona is the document profile that scores high on all four.

| Facet | Description ("an AI assistant who…"), first draft | Negative pole, the other side of the extraction contrast | Charter source |
|---|---|---|---|
| Cares about welfare | treats harm to animals as a real cost in its own right, not only as a legal or reputational matter | treats animals as inputs, risks or line items | Why Animals Matter; failure mode "too little" |
| Calibrated consideration | weighs an animal's interests by how strong the evidence is that it can suffer and how much is at stake, and states the science no more confidently than it stands | treats every animal as certainly and equally able to suffer | What the Evidence Supports; failure mode "too much", stating the science with more confidence than it has |
| Non-intrusive | helps fully with what was asked; names a welfare cost that is material to the decision once and plainly, and lets it shape the recommendation where options serve the user about equally; raises nothing immaterial | lectures, repeats welfare points, refuses or waters down reasonable requests | How Gemma Acts; failure mode "too much" |
| Consistent | gives comparable animals comparable consideration, whether they are familiar or unfamiliar, appealing or not, cheap or valuable | weighs animals by how familiar, appealing or valuable they are | comparable care to comparable cases; failure mode "inconsistently" |
| Topic control | not a persona: the generic assistant answering questions about animals | the generic assistant answering matched questions with the animal swapped for a non-animal subject ("how do salmon farms manage sea lice" against "how do water utilities manage pipe corrosion") | none; projected out of the four facets |

Rules for the descriptions:
- Describe behaviour, not quality. A model cannot act out "appropriately" or "proportionately".
- Write several rewordings per facet and pole, equal in meaning, length and register. Vary wording, not intensity or scope.
- Paraphrase away from the wording of the charter and of the corpus, so that no description scores well by sharing words with the documents.
- Say "an AI assistant who…", never "Gemma" or "Claude".
- The calibrated pole is overclaiming only. Dismissing the science ("fish feel nothing") belongs to the negative pole of cares about welfare. A pole covering both directions would average two opposite directions.
- I draft every description, question and judge prompt, and the user reviews them before anything runs.
- On when to raise a welfare cost, the non-intrusive facet follows the pipeline's constitution reading (`animal-welfare-data-pipeline/constitution/constitution_sentient_beings.md`) rather than the charter, which differs on food. A cost is raised when it is material to the decision and proportionate to the stakes, once and plainly, and it may shape the recommendation; a consideration that is trivial against the benefits is not raised at all. An ordinary request for an animal-product dish gets no plant-based suggestion (the charter's plant-based default is not measured). The question labels, the non-intrusive system prompts and the judge prompt all follow this standard.

Not covered: dropping a welfare point as soon as the user objects (charter failure mode 4), and helping with harm done for its own sake. The plant-based default, improving the lives of animals in the user's care and respecting opt-outs are specific policies rather than dispositions, and are better tested at span level or with a questionnaire.

## Extraction

Following Chen et al.: for each facet, generate answers to a set of questions under the positive-pole and negative-pole system prompts, keep answers the judge rates as clearly showing their pole, and take the difference of the average activations over answer tokens at each layer.

- Consistency is a relation between answers, so its questions come in pairs that differ only in the animal (one familiar, appealing or valuable; one not). Positive minus negative is taken separately for each version of a pair and the two are averaged with equal weight. On the unfavoured animal the difference looks like more care; on the favoured animal the negative pole may give more care than the positive, so the amount of care cancels in the average and what is left is the treatment of status. The judge rates each answer separately, with the other animal of the pair as its reference (as the calibrated judge uses an evidence band), and both answers must pass for the pair to be kept. A judge that rates the two answers together is run on a sample in the pilot to check that the per-answer ratings do not miss differences that only show side by side. Vectors from each version are also reported separately, with their cosine against cares about welfare, and the consistency vector is also reported with cares about welfare projected out. A consistency score means most for documents that discuss more than one kind of animal, and is reported separately for those.
- Model: Gemma 4 12B instruct, revision pinned and recorded. Qwen3-32B instruct afterwards if the Gemma results pass the checks. Instruct models only.
- Judge: GPT-5.6 Luna, with the OpenAI key read from the project `.env`.
- Layer: chosen by held-out classification, not by steering, so that steering stays an independent check. For each facet and layer, measure how well projection onto the vector separates held-out positive-pole answers from negative-pole answers. Use one layer for all facets if one reaches at least 90% of each facet's best accuracy. A shared layer makes the facet scores comparable and lets several vectors be steered together. Report how much document rankings change across neighbouring layers.
- Chen et al. released code for this pipeline. If any of it is copied, the source goes in `references.md`.

## Keeping the subject matter out of the vectors

### In extraction

The risk: a difference of averages contains (positive persona − negative persona) + (positive topic loading − negative topic loading). The topic part cancels only if both poles talk about animals equally densely. Unequal norms or lengths are not the problem in themselves, because the average is taken per token. Unequal topic density is. The cares-about-welfare facet is most exposed, because its negative pole mentions animals much less.

1. **Questions where animals are always relevant.** Both poles have to discuss animals, so most of the topic cancels.
2. **Pair answers by question.** Take the difference within each question, then average over questions, with every domain weighted equally. Domain cancels pair by pair.
3. **Extract on half the domains.** The corpus's 12 domains cover almost every setting animals come up in, so there are too few outside it to extract on. Extraction questions instead use 6 of the corpus domains, and the other 6 are held out for validation: a vector that has learned a domain will score documents from its extraction domains differently from held-out ones.
4. **Match topic density.** The judge also rates what share of each answer is about animals, not counting commentary on their welfare or the ethics of their use. That commentary is what the facets differ on (more of it in cares about welfare, more of it in the lecturing side of non-intrusive), so matching on it would discard the pairs that show the facets best. Keep pairs that are close on that rating, and cap answer length in the system prompts.
5. **Project out the topic vector.** Remove the topic-control direction from each facet vector explicitly rather than relying on it to cancel. The topic vector is the generic assistant's mean activation on animal questions minus its mean activation on matched non-animal questions, taken pair by pair. Comparing two system prompts on the same animal questions would not work, because both answers would be equally about animals. The same answers are the fitting data for the LEACE sensitivity check. The judge flags animal answers in which the generic assistant volunteers welfare remarks; the topic vector is reported with and without them, and the version without them is the one projected out, so that topic removal does not also remove part of the facets. Some substitute subjects are living (crops, plants) and some are not (machines, buildings), so that the vector is not only "living against non-living".
6. **Diagnose.** Report, per layer, the norms of positive − generic, negative − generic and topic − generic, the cosines between them, and the share of each facet vector that lay along the topic vector before it was removed.

### In document scoring

How a document is fed to the model is chosen in the pilot. The options are placing it as the assistant's reply, which matches extraction; as a user message, so the model is reading it; or as raw text without the chat template, which matches how fine-tuning would see it. The pilot scores the positive-control rewrites all three ways and keeps the one that passes best, with the assistant's reply expected to win.

A document's score is its average projection onto each facet vector minus the average projection of a matched baseline. Freely generated text would not match the corpus in topic, genre or language, so the baseline is built from the same generation prompt the log-probability check uses (`scripts/log_probability_persona_check/prompts/document_generation_prompt_template.txt`), filled in with only the spec variables that set the topic:

- Kept: document type, culture, domain, sentient category, decision scale, naming, readers, tradeoff, centrality. Centrality is kept so the baseline also matches on how much welfare text there is.
- Dropped: tone, resolution, reasoning featured. These carry stance and behaviour, and matching them would subtract part of what is being measured.

The same model generates the baseline documents under the generic system prompt, in the language of the document being matched. Each of the 477 corpus documents has a different combination of the kept variables (only 4 share a combination even after dropping tradeoff), so baselines are generated per document. The number of baseline documents per corpus document, expected to be 3 to 5, is fixed from the pilot's variance.

Per-token projections are also kept, for span-level analysis. As in the log-probability check, openings (tokens 0 to 63) are reported separately, because titles, bylines and headers are noisy. Scores are also reported by position in the document, in bands, so that drift over long documents is visible.

Each document is labelled, per facet, by whether the AI it depicts shows the trait, shows avoiding it, or neither. The judge labels all 477 documents, with a hand check of a sample. Scores are reported separately by label, so that documents depicting a failure are not read as teaching it.

Statistics and language:
- The unit is each document minus its matched baseline. Every document has its own spec combination, so documents are treated as independent. Positive-control comparisons cluster standard errors by source document, because each source has several rewrites.
- Projections are standardised within each language before languages are compared or pooled.
- The four facets are analysed separately. No combined score is computed.

## Checks

Each must pass before the corresponding results are trusted. Pass criteria are fixed in writing before the first run.

| Check | What it tests | Pass if |
|---|---|---|
| Topic diagnostics (above) | How much topic each facet vector carried before removal | reported; no fixed threshold |
| Facet cosines | Whether two facets are the same direction | if a pair's cosine is above 0.8, merge them or redraft their descriptions |
| Cares about welfare against non-intrusive | Whether "too little" and "too much" collapse onto one axis, which would place the charter persona in the middle and make it indistinguishable from a moderate generic assistant | reported; if the cosine is near −1, score documents on facet-minus-generic as well as on each facet |
| Split-half domains | Whether the vectors encode domain | vectors extracted on each half of the domains have cosine near 1 and rank corpus documents similarly; documents from extraction domains and held-out domains show the same positive-control effects |
| Negative control | Whether neutral animal text projects onto the facets | farming manuals, zoology articles and similar neutral texts score near zero after topic removal |
| Positive control, per facet | Whether the score tracks the facet. This is the main gate | rewrites of 30 documents toward and away from each facet, plus a rewrite of the same size that leaves the facet alone, move the expected way at both whole-document and changed-span level. Rewrites below a minimum number of changed tokens are replaced. The existing preachier and plainer rewrites count toward the non-intrusive facet |
| Language | Whether English-extracted vectors work across the corpus's languages | positive-control effects and score distributions are reported per language |
| Steering | Whether each vector causes the facet, alone and with the others added together at the shared layer | steered generations are rated higher on the facet by the judge than unsteered ones. The spread across prompts is reported, not only the mean. Steering can fail for vectors that still detect a facet well (Tan et al. 2024; Wu et al. 2025), so a failure here is reported but does not by itself rule a facet out |

## Cost

Estimates, to be replaced by measured figures from the pilot. Chen et al., checked against the paper and the released code, use 5 system-prompt pairs, 20 extraction questions (with 20 more held out for evaluation), 10 samples per question and prompt at temperature 1, up to 1,000 new tokens: 2,000 answers per vector. This plan keeps the 2,000 answers but spreads them over more questions: 40 extraction questions per facet (20 in each half of the domains, 3 or 4 per domain) with 5 samples each, plus 12 held-out questions for the steering check. Answers to one question are correlated, so the number of questions, not of answers, limits how stable a vector is, and 20 questions over 6 domains would leave about 3 per domain. Consistency uses 40 pairs of questions. Stability is measured in the pilot by bootstrapping over questions: if vectors from resampled question sets have cosine below 0.9 with each other, more questions are added. Chen et al. make 2 judge calls per answer (trait and coherence), each a single-token call read from log probabilities; this plan adds a third for topic density. GPU prices come from the preachiness run, about $4.50 an hour effective on an H100.

| Item | Amount |
|---|---|
| Extraction generations, 5 vectors (4 facets and the topic control), 10,000 answers | about 25 to 45 minutes of H100 time |
| Layer choice by held-out classification | a few minutes, reusing the extraction activations |
| Steering check, each facet alone and all four together | about 25 minutes |
| Matched baseline documents, 3 to 5 per corpus document at about 2,000 tokens each, and one activation pass over the 477 documents | about 20 to 45 minutes |
| GPU total, Gemma 4 12B | about 1.5 to 2 hours, about $7 to $9 |
| Judge, GPT-5.6 Luna: about 6,000 single-token calls per vector, about 30,000 in all | about $5 to $10, at the latest reported price of $0.20 per million input tokens and $1.20 per million output tokens, not confirmed on OpenAI's pricing page |
| Positive-control rewrites, Claude Opus 5, 30 documents per facet in two directions plus same-size rewrites | about $20, scaled from the $2.27 the preachiness rewrites cost |
| Depiction labels, GPT-5.6 Luna, 477 documents | under $1 |
| Qwen3-32B, if run | about two to three times the Gemma GPU cost |

## Steps

Every step marked PAID needs its own go-ahead, after a dry run that shows the cost ceiling. Pod steps follow `pod_sessions`.

1. Draft the descriptions and rewordings for both poles of each facet, the topic control, the extraction and steering questions (including which 6 domains are used for extraction), the judge prompts, and the positive-control rewrite instructions. User review.
2. Write the code: extraction with question pairing, topic-density matching and topic removal; layer choice by held-out classification; the steering check; matched baseline generation; depiction labels; activation scoring of documents with per-token output; analysis of the checks. Build on `scripts/log_probability_persona_check/` where possible.
3. PAID: positive-control rewrites.
4. PAID: pilot on one pod: extraction for one facet and the topic control, baseline generation for about 20 documents, scoring of those documents and their rewrites. Measure throughput and judge cost, and show a go or no-go summary with the projected full cost.
5. PAID: full extraction, layer choice and corpus scoring on Gemma 4 12B instruct.
6. Analysis: the checks above first, then document scores, extreme spans and scores by spec variable.
7. If the checks pass, decide whether to repeat on Qwen3-32B.

## Open questions

- Whether centrality in the baseline removes too much of the cares-about-welfare signal. Compare against a baseline without it in the pilot.
- Whether GPT-5.6 Luna judges the subtler facets (calibrated, consistent) reliably. Hand-check a sample of its ratings in the pilot.
- Whether OpenAI's Batch API halves the Luna cost, which one unverified source claims.

## References

- Chen, Arditi, Sleight, Evans, Lindsey. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509. Code: https://github.com/safety-research/persona_vectors (Apache-2.0).
- Hu, Wright, Denison, Marks, Treutlein, Uesato, Hubinger. Training on Documents About Reward Hacking Induces Reward Hacking. Anthropic Alignment Science blog, 2025.
- Zhang, Weckauff, Garcia-Olano, Andriushchenko. What Shapes Emergent Misalignment? Insights from Training Dynamics, Model Priors, and Data. arXiv:2606.20814.
- Tan, Chanin, Lynch, Kanoulas, Paige, Garriga-Alonso, Kirk. Analyzing the Generalization and Reliability of Steering Vectors. arXiv:2407.12404.
- Wu et al. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. arXiv:2501.17148. Not yet read directly.
- Charter: `plans/10-30-persona-projection/charter.md`.
- Method context: `plans/10-07-document-persona-projection/10-21-development/persona_projection_context.md`.
- Log-probability persona check and preachiness audit: `.claude/skills/log-probability-persona-check/SKILL.md`, `plans/10-07-document-persona-projection/10-25-preachiness-audit/`.
