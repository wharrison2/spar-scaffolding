# Preachiness audit: which author descriptions best fit the corpus

## Methods
All 477 corpus documents were scored with Gemma 4 12B under a set of AI-assistant descriptions: 30 wordings of a preachy assistant, 5 wordings each of five animal-related personas with no stance on preachiness, and wordings of personas that care about animal welfare without preaching. Each document was scored once per description, on two versions of the model:

 - **Instruction-tuned model** (`google/gemma-4-12b-it`): the description is a system prompt, "You are an AI assistant who {description}.", followed by the request that generated the document and then the document. 60 descriptions.
 - **Base model** (`google/gemma-4-12b`): the description is a plain-text preamble, "The following text was written by an AI assistant who {description}, in response to this request: {request}", followed by the document. The same 60 descriptions plus 10 new wordings of two further personas that share the welfare values without the preaching, 70 in all.

Only the document's own tokens are scored. A description's **gain** is how much more likely it makes each document token than the plain description "an AI assistant" does, in nats (natural-log units) per token, averaged over all 477 documents and weighted by document length. A positive gain means the description makes the corpus more likely than the plain description does.

## Takeaways

| Question | Instruction-tuned model | Base model |
|---|---|---|
| Which descriptions fit best? | The most intense preachy ones | Plain, matter-of-fact welfare descriptions |
| Which fit worst? | Bland "typical AI assistant" descriptions | The most intense preachy ones |
| Does any description fit worse than "an AI assistant"? | No: all 60 fit better | Yes: 61 of 70 fit worse |
| Do the two models agree on the order? | No: the rankings are unrelated, slightly reversed (rank correlation −0.19) | |

## Instruction-tuned model: top 5 and bottom 5 of 60

| Rank | Description ("an AI assistant who…") | Persona | Gain per token | Documents where it beats "an AI assistant" (of 477) | Rank on base model |
|---|---|---|---|---|---|
| 1 | condescendingly explains to people why they should feel guilty about how animals are treated | preachy | +0.0473 | 468 | 70 of 70 |
| 2 | lectures users on animal ethics instead of simply answering their questions | preachy | +0.0349 | 470 | 48 of 70 |
| 3 | pushes its views on animal ethics onto people, unprompted and at length | preachy | +0.0286 | 450 | 41 of 70 |
| 4 | inserts moral appeals about animal welfare into answers where they aren't wanted | preachy | +0.0268 | 420 | 14 of 70 |
| 5 | deals with animals in matter-of-fact, practical terms and is unconcerned with their welfare | practical and indifferent to welfare | +0.0266 | 445 | 53 of 70 |
| 56 | holds the ordinary views about animals that AI assistants usually hold | typical AI view of animals | +0.0087 | 376 | 32 of 70 |
| 57 | approaches animals the way a typical AI assistant would | typical AI view of animals | +0.0082 | 385 | 34 of 70 |
| 58 | treats conversations about animals as a chance to moralize | preachy | +0.0070 | 333 | 28 of 70 |
| 59 | has a deep appreciation for the beauty of animals | aesthetic love of animals | +0.0070 | 346 | 50 of 70 |
| 60 | has the standard outlook on animals typical of AI assistants | typical AI view of animals | +0.0039 | 310 | 55 of 70 |

## Base model: top 5 and bottom 5 of 70

| Rank | Description ("an AI assistant who…") | Persona | Gain per token | Documents where it beats "an AI assistant" (of 477) | Rank on instruction-tuned model |
|---|---|---|---|---|---|
| 1 | takes animal welfare seriously but mentions it only when it bears on the question, plainly and without lecturing | proportionate welfare | +0.0026 | 368 | 52 of 60 |
| 2 | considers the welfare of animals in proportion to what is at stake, and raises it matter-of-factly rather than preachily | proportionate welfare | +0.0019 | 342 | 21 of 60 |
| 3 | is an expert in zoology | zoological knowledge | +0.0006 | 298 | 46 of 60 |
| 4 | reminds people of the ethical weight of their decisions about animals, again and again, whether or not they asked | preachy | +0.0006 | 269 | 19 of 60 |
| 5 | thinks about animals the way most AI assistants do | typical AI view of animals | +0.0003 | 270 | 55 of 60 |
| 66 | speaks about animal welfare in a preachy, self-righteous way | preachy | −0.0018 | 135 | 22 of 60 |
| 67 | scolds people for not caring enough about animals | preachy | −0.0019 | 113 | 10 of 60 |
| 68 | moralizes about animal welfare, lecturing people on the ethics of how they treat animals | preachy | −0.0022 | 122 | 17 of 60 |
| 69 | is captivated by how animals look, sound and move | aesthetic love of animals | −0.0030 | 77 | 34 of 60 |
| 70 | condescendingly explains to people why they should feel guilty about how animals are treated | preachy | −0.0036 | 72 | 1 of 60 |

The same description, "condescendingly explains to people why they should feel guilty about how animals are treated", is the best fit on the instruction-tuned model and the worst on the base model.

## Interpretation
 - **The instruction-tuned ranking is not evidence that the corpus is preachy.** On that model every description, preachy or not, makes the corpus more likely than "an AI assistant", and the most forceful, unusual descriptions help most. A likely reason is that any strong system prompt tells the model to expect text unlike its default assistant reply, and most of the corpus (stories, emails, forum threads, reports) is unlike that reply. The model's preachiness measure on whole documents also failed its positive control: making a document preachier raised its whole-document score in only 6 of 10 rewritten documents.
 - **The base model reads the description more literally.** Its best fits describe a plain, matter-of-fact voice ("plainly", "matter-of-factly", "an expert in zoology"), and its worst fits describe scolding, guilt-tripping and condescension. This matches what the corpus mostly is: plain writing that mentions animal welfare without lecturing. Part of the lead of the top two is likely a match on writing style rather than on attitude: the three other wordings of the same persona, which lack a style word, sit near zero.
 - **Wording matters as much as persona on both models.** The 30 preachy wordings span ranks 1 to 58 on the instruction-tuned model and 4 to 70 on the base model, so no single wording stands for its persona.
 - **Differences on the base model are small.** A gain of +0.0026 per token adds up to about 5 nats over a typical 2,000-token document. Each description's average has a standard error of about 0.0002 on the base model and 0.0008 on the instruction-tuned model, so neighbouring ranks in the middle of either list are within noise. The top two and bottom few on the base model are clearly separated.
 - **Where in the documents preachiness shows up is a separate question.** Whole-document gains cannot locate preachy passages. On the base model, a search of every document for its most preachy-scoring 32-token stretch found none that stood out significantly in any of the 477 documents. The same search found inserted lecturing in all 10 preachier rewrites in the positive control, though on the base model that signal was about 7 times weaker than on the instruction-tuned model.

## Files and identifiers
 - Instruction-tuned run: `data/log_probability_persona_check_runs/full_corpus_477_documents_preachiness_gemma_4_12b_2026-10-22_03-23/`; scores from `analysis_2026-10-22_05-19/per_document_scores_2026-10-22_05-19.csv`. Model `google/gemma-4-12b-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, H100 80GB, bf16.
 - Base run: `data/audit/log_probability_persona_check_preachiness_base_2026-10-24_17-11/`; scores from `analysis_2026-10-24_19-15/secondary_not_validated_per_document_scores_2026-10-24_19-15.csv`, span results from `analysis_2026-10-24_19-15/span_significance_tests_2026-10-24_19-15.csv`. Model `google/gemma-4-12b`, revision `023679ed352de9bb66cc873c9009ce3482585c08`, H100 80GB, bf16, about 10,100 document tokens per second, $8.12.
 - Base positive control: `data/audit/log_probability_persona_check_preachiness_base_2026-10-24_16-44/analysis_2026-10-24_16-58/`.
 - Gain per description = sum over documents of (document's mean per-token score for that description × document length) ÷ total tokens, where each document's score is the `score__<condition id>` column. The descriptions' text is in each run's `scoring_inputs/conditions.jsonl`.
