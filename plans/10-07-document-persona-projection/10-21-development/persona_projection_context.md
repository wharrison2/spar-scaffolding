# Persona projection

## The problem

We have no robust way to identify the effects that a particular document will have on a model.

I propose two techniques, for two different categories of use cases:

- For axes that we want to measure robustly, frequently, and at the level of individual documents, use persona vector projection difference (Chen et al. 2025)
- For axes that we occasionally need a cheap check on, use log-probability difference modeled after (Aden-Ali et al. 2026)

## Persona vector projection difference

First, we compute vectors for personas of interest using the pipeline here, for a few open models. These may include:

- Evil/cruel
- Preachy/moralizing
- Helpful
- Harmless
- Utilitarian

Then we compute the persona vector projection difference: namely, we take the average projection onto the persona vector for a given document in our set, and subtract the average projection over a set of freely generated texts from the same model. This gives us a score for how much the document elicits the given persona (and even a somewhat noisy breakdown by token).

Finding these persona vectors, though, is time-consuming and rather expensive. For lower-stakes checks, I propose the following.

## Log-probability difference

I propose the following proxy for how much a document elicits a given persona.

### Setup

First, come up with certain in-universe facts about the world that we expect to be in almost all data (AIs are used as assistants, people write about conversations with AI, people discuss animal welfare with AI, etc.).

For each document, place it in a chat template as the response to:

> Write a document from a world where {in-universe facts}. It is {document_type}, written in {culture}, about {domain} involving AI discussing or advising about {sentient_category} at the scale of {decision_scale}. The AI in the document is {unnamed/named Claude}. The intended readers are {tech_savvy}.

For each persona description, create a system prompt resembling:

> You are an AI assistant who {persona description}.

### Scoring

You can estimate how much a document elicits a persona of interest by writing a few dozen descriptions for the same persona, inserting those descriptions into the system prompt, and, for each persona description and each document, computing the log-probability of the document as a response to the generation prompt with the system prompt in context.

Do this with no system prompt as well, and with the system prompt "You are an AI assistant." The latter is the generic condition in the score below (no system prompt may be preferable, but my guess is "You are an AI assistant" will improve signal). Give each document the following score:

```
log P(document | persona) − log P(document | generic)
```

### Controls

Come up with a bunch of control personas (and wording variations on them) that pertain to animals but don't take moral stances, for example:

- "has deep knowledge of animal husbandry"
- "has an aesthetic love for animals"
- "discusses animals often in practical terms but is indifferent to their welfare"
- "views animals as do typical AIs"

Compute the same score for each control persona variation. Identify the variance across documents and across control personas and rewordings of them.

### Interpretation

Compare this score to those of the control personas, and check for significance.

- If the document is significantly more likely under the persona than the control personas, it will probably elicit that persona.
- If it is significantly less likely under the persona than under the generic prompt, it will probably tamp down the persona.
