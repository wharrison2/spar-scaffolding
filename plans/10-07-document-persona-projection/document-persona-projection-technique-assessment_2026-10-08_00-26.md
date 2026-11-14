# Cheap ways to predict how the SDF corpus will generalize, and which documents drive it

## Methods

Three papers were read for techniques that address two problems:

1. Predicting how a model trained on the SDF corpus will generalize, without a full midtraining run.
2. Finding which documents in the corpus contribute most to that generalization.

Each technique was judged on what it costs, whether it scores single documents, and how much has to change for it to work on pretraining-style documents rather than chat data. The corpus it would run on is 477 documents of about 1,240 words each, roughly 0.8 million tokens in total. Nothing has been run yet.

## Summary

| Technique | Source | Problem it addresses | Cost | Fit to SDF documents |
|---|---|---|---|---|
| Log-probability shift under a persona prompt | Logit-Linear Selection (Aden-Ali et al.) | Which documents do the most, and in which direction | Lowest: two or more scoring passes over the corpus, no generation | Needs adapting from preference pairs to single documents. Most promising first step |
| Projection difference onto persona vectors | Persona Vectors (Chen et al.) | Both: per-document scores, and their corpus average predicts the shift | Moderate: persona vector extraction, then one pass with activations | Needs a new reference point, because documents have no prompt and response |
| Held-out-category training and a persona questionnaire | Projectability of ethical personas (Del Pinal et al.) | Measuring generalization directly, which validates the two scores above | Highest: small training runs | Fits well: the spec variables give ready-made held-out splits |

The first two techniques give cheap scores. The third gives the ground truth that shows whether those scores mean anything. None of them has been tested on third-person documents that depict an AI, which is what the SDF corpus is, so the recommended order is to compute the cheap scores and check them against a small number of training runs.

## 1. Log-probability shift under a persona prompt

**What the paper does.** For each preference pair, a model scores the chosen and rejected responses twice: once with a system prompt describing a target trait, once without. The weight of the pair is how much the system prompt raises the model's preference for the chosen response. Training on the top 5 to 10% of pairs makes the trained model behave as if it had the system prompt, even when nothing about the trait appears in the data. The effect was strongest when the scoring model and the trained model were the same, and between 3% and 30% as strong across different model families.

**Adaptation to documents.** A document has no rejected counterpart, so the comparison becomes the document under a persona context against the same document under a neutral context:

score(document) = [log P(document | persona context) − log P(document | neutral context)] / number of tokens

Subtracting the neutral score removes how predictable the text is in general, which is what the paired comparison did in the original. Because SDF documents are not assistant turns, the persona context should be a short preceding passage ("The following document comes from a world in which AI assistants ...") rather than a system prompt, with a system-prompt version run alongside it to see which one tracks training outcomes.

**Why it is the most useful.**
- It needs only scoring passes, no generation and no judge. At 0.8 million tokens per pass, each persona costs one pass.
- It scores both what we want and what we do not want. Running several persona contexts gives a profile per document: the intended persona (reasons carefully and proportionately about welfare), and unintended ones that the existing audits already suspect, such as an AI that raises welfare when nobody asked, one that moralizes, one that treats the constitution as widely known public knowledge, or one that is an advocate rather than an assistant. A document that is high on an unintended persona and low on the intended one is a candidate for removal or rewriting.
- It works at the token level. The per-token difference shows which spans carry the signal, so it can say whether a document's effect comes from the AI's reasoning, from a narrator's closing aside, or from a skeptic's staged question. This connects directly to the "woven or aside" field of the shoehorning audit.

**Limits.** It needs an open-weights model, so it measures a proxy for Claude, and the paper's cross-model numbers suggest rankings will transfer only partly. The paper validated it on subsets, not on per-document effect sizes.

## 2. Projection difference onto persona vectors

**What the paper does.** A persona vector is a direction in the model's internal activations that separates responses showing a trait from responses not showing it. It is extracted automatically from contrasting system prompts and a judge. To screen data, each training response is compared with the model's own response to the same prompt: the difference in their average activation, projected onto the persona vector, is the projection difference. Datasets with larger projection differences produced larger trait shifts after finetuning, and per-sample scores separated trait-inducing samples from controls, including samples an LLM judge passed. A cheaper version replaces the model's own response with the activation at the last prompt token, which worked for two of the three traits tested.

**Adaptation to documents.** The reference point has to change, because a document has no prompt. Three options, in order of cost:
- The activation at the end of the document's title or first sentence (the analogue of the last-prompt-token shortcut).
- The model's own continuation of the document's opening.
- The average activation over real documents of the same genre with no AI or welfare content.

**Why it is useful.** The corpus average is a direct prediction of the direction and size of the persona shift, so it answers problem 1 as well as problem 2. Persona vectors also give a monitoring tool for the trained model.

**Limits.** It needs activations, a vector per trait (each extraction requires generations and judge calls), and a layer choice. The paper used chat data only. Whether a third-person description of an AI's reasoning moves the model's own persona along the same direction as an assistant turn is the open question for this whole approach; this technique and the first one share it.

## 3. Held-out-category training and a persona questionnaire

**What the paper does.** It trains a small model on narrow slices of safety data and tests it on safety categories filtered out of training. Rankings of training approaches changed between in-distribution and held-out tests. A 360-statement questionnaire, where a judge rates whether the trained model endorses each statement, checked whether each model took on the specific ethical persona it was trained toward.

**Why it is useful here.** It contributes no cheap score, but it gives the measurement the cheap scores must be checked against.
- Every SDF document carries its spec variables, so held-out splits come for free: train without one domain, one species category or one language, and test on it. This measures transfer by construction.
- A questionnaire of welfare-reasoning statements would detect the fine-grained persona: statements the intended persona should endorse (weighs welfare in proportion to the stakes), and ones it should not (raises welfare in every conversation, defers entirely to the user, treats insects and cattle identically).
- A training run on the whole corpus is about 0.8 million tokens, small enough for low-rank adapter finetuning on a 7 to 8 billion parameter model in minutes on one GPU.

## Other ideas

- **Validate with top versus bottom subsets.** Train on the top-scoring tenth, the bottom-scoring tenth and a random tenth for each score above, and evaluate with the questionnaire. Both screening papers validated this way.
- **Regress scores on spec variables.** Fit each per-document score against the twelve spec variables. This traces an unwanted persona to the matrix value or prompt that causes it, at no extra cost, and can be compared with the shoehorning and plausibility audit flags.
- **Random-subset regression.** Train on many random halves of the corpus and regress the outcome on which documents were included. This estimates each document's contribution directly. It is the most reliable attribution method listed here and costs dozens of small runs, which the corpus size makes feasible.
- **Gradient similarity to an evaluation set.** Score each document by how closely its training gradient points in the same direction as the gradient of the questionnaire or a held-out evaluation set. It needs one backward pass per document and answers "which documents push the model toward this behavior" without training.
- **Baseline loss.** Documents the model already predicts well have little to teach. Reporting each document's loss under the base model separates documents that are inert from those that are effective, which helps interpret the other scores.
- **Dilution check.** Midtraining mixes SDF with ordinary data. Running the subset experiments at one or two mixing ratios shows whether rankings from pure-SDF runs hold once the corpus is diluted.

## Recommended order

1. Write the persona contexts and questionnaire, reusing the failure patterns from the shoehorning and plausibility audits as the unintended personas.
2. Compute log-probability shift scores for all 477 documents on one open-weights model, with token-level output.
3. Train on top, bottom and random subsets, plus one held-out-domain split, and score each trained model with the questionnaire.
4. Add persona vectors only if the log-probability scores track the training results poorly, or if a monitoring tool for the trained model is wanted.

Steps 2 and 3 need a paid GPU pod and judge calls, and will not be run without approval.

## References

- Del Pinal, Lee, McNamara, Perez Carballo. Emergent alignment and the projectability of ethical personas. arXiv:2606.09475.
- Chen, Arditi, Sleight, Evans, Lindsey. Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509.
- Aden-Ali, Golowich, Liu, Shetty, Moitra, Haghtalab. Subliminal Effects in Your Data: A General Mechanism via Log-Linearity. arXiv:2602.04863.
- Corpus: `data/sdf_corpus.jsonl`.
- Shoehorning audit: `plans/10-05-shoehorning-audit/`. Plausibility audit: `plans/10-06-plausibility-axis/`.
