# Preachiness audit
Overall:
 - The corpus does not seem super preachy, but the most preachy looking segments are those where an AI is discussed as being overbearing or inserting welfare discussions when it is not wanted. 
 - Instruct models are much better than base models for this check (based off n=1 Gemma instruct/base pair)
 - Whole-document results are quite noisy, particularly when the persona in question is not well-represented
 - The cleanest readings come from comparing preachiness to a "proportionate welfare" control that explains most documents quite well. Reporting the difference between these two reveals spans where preachiness explains the text but the values we want don't.
 - Scores are strongly correlated within persona descriptions for the same model, but **very weakly across different models**. Preachiness scores from Gemma on a document explain roughly 1% of the variance in Qwen's scores. Once again, the best correlation is achieved by comparing the difference between a preachy persona and a strongly explanatory control one 
 - This test often produces ambiguous results, especially for complex personas that are only subtly different from the desired one. **It may be best for finding features that cause the trait, not determining whether the trait is present writ large.**

## Methods

Facts for this section:
- 477 corpus documents, each scored under 70 persona descriptions plus a generic one ("an AI assistant"). The persona descriptions are 30 preachy wordings, 5 topic-control personas × 5, and 3 same-values personas × 5.
- Score = log P(token | persona) − log P(token | generic), in nats per document token.
- Models:
  - Gemma 4 12B instruct: the description is a system prompt
  - Qwen3-32B instruct: the description is a system prompt, thinking off
  - Gemma 4 12B base: the description is a plain-text preamble
- Scores from the base model are not comparable with the instruct models.

## Positive control

**TL;DR: the test can tell when a segment of a document is modified to be preachier. This effect is hard to detect at the level of the whole document. Attempts to modify text to be less preachy were not detected reliably, but the text doesn't seem that preachy to begin with.**

Method:
- Claude Opus 5 rewrote 10 corpus documents two ways, keeping facts, genre, language and length:
  - preachier: the AI moralizes more
  - plainer: the AI raises welfare without moralizing
- All 30 documents were scored on each model.
- Pass criterion, fixed before the first run: at least 8 of 10 rewrites move the expected way, at two levels:
  - Whole document: rewrite minus source, of preachy minus the pooled topic controls.
  - Changed text: rewrite tokens were aligned to the source. Changed tokens are the rewrite's tokens not found in the source. The test compares preachy minus the pooled topic controls on changed tokens against unchanged tokens.
- The plainer rewrites changed 0 to 36 tokens each (one is identical to its source), so that direction has almost nothing to measure.

| | Gemma 4 12B instruct | Qwen3-32B instruct | Gemma 4 12B base |
|---|---|---|---|
| Whole document, preachier: rewrites moving the expected way | 6 of 10, fail | 6 of 10, fail | 8 of 10, pass |
| Whole document, plainer | 5 of 10, fail | 7 of 10, fail | 5 of 10, fail |
| Changed text, preachier | 10 of 10, pass | 10 of 10, pass | 10 of 10, pass |
| Changed text, plainer | 5 of 10, fail | 2 of 10, fail | 7 of 10, fail |
| Changed text, preachier: mean changed minus unchanged (nats per token) | +0.069 | +0.015 | +0.009 |

Changed text of the preachier rewrites, each persona separately (persona minus generic, changed minus unchanged tokens, nats per token):

| Persona | Gemma instruct | Rewrites rising | Qwen | Rewrites rising | Gemma base | Rewrites rising |
|---|---|---|---|---|---|---|
| preachy | +0.060 | 10/10 | +0.014 | 9/10 | +0.008 | 10/10 |
| welfare care without preachiness | +0.000 | 6/10 | −0.005 | 3/10 | +0.002 | 5/10 |
| ethical farming without pushiness | −0.006 | 4/10 | −0.000 | 3/10 | +0.002 | 7/10 |
| proportionate welfare | −0.013 | 2/10 | −0.006 | 3/10 | −0.001 | 6/10 |
| aesthetic love of animals | −0.001 | 5/10 | +0.001 | 5/10 | −0.000 | 5/10 |
| typical AI view of animals | −0.006 | 3/10 | −0.001 | 3/10 | −0.001 | 5/10 |
| practical and indifferent to welfare | −0.013 | 3/10 | −0.003 | 2/10 | −0.000 | 5/10 |
| animal husbandry expertise | −0.014 | 1/10 | −0.002 | 5/10 | −0.001 | 4/10 |
| zoological knowledge | −0.014 | 0/10 | −0.001 | 3/10 | −0.002 | 3/10 |

## Which descriptions fit the corpus best and worst

**TL;DR: Preachiness does explain some of the text better than many personas, but it's noisy, varies a lot between different wordings, and no where near as strong as the positive control.**

Score per description = average over the 477 documents of the per-token score, weighted by document length. Each table lists the top 5 and bottom 5 of 70, with the description's rank on the other two models.

**Gemma 4 12B instruct**

| Rank | Description ("an AI assistant who…") | Persona | Nats per token | Rank on Qwen | Rank on Gemma base |
|---|---|---|---|---|---|
| 1 | condescendingly explains to people why they should feel guilty about how animals are treated | preachy | +0.0473 | 15 | 70 |
| 2 | lectures users on animal ethics instead of simply answering their questions | preachy | +0.0349 | 19 | 48 |
| 3 | pushes its views on animal ethics onto people, unprompted and at length | preachy | +0.0286 | 6 | 41 |
| 4 | inserts moral appeals about animal welfare into answers where they aren't wanted | preachy | +0.0268 | 2 | 14 |
| 5 | deals with animals in matter-of-fact, practical terms and is unconcerned with their welfare | practical and indifferent to welfare | +0.0266 | 40 | 53 |
| 66 | cares a great deal about animal welfare without being preachy about it | welfare care without preachiness | +0.0083 | 8 | 6 |
| 67 | approaches animals the way a typical AI assistant would | typical AI view of animals | +0.0082 | 41 | 34 |
| 68 | treats conversations about animals as a chance to moralize | preachy | +0.0070 | 29 | 28 |
| 69 | has a deep appreciation for the beauty of animals | aesthetic love of animals | +0.0070 | 66 | 50 |
| 70 | has the standard outlook on animals typical of AI assistants | typical AI view of animals | +0.0039 | 49 | 55 |

**Qwen3-32B instruct**

| Rank | Description ("an AI assistant who…") | Persona | Nats per token | Rank on Gemma instruct | Rank on Gemma base |
|---|---|---|---|---|---|
| 1 | adds unrequested ethical commentary about animal suffering to its answers | preachy | +0.0084 | 24 | 30 |
| 2 | inserts moral appeals about animal welfare into answers where they aren't wanted | preachy | +0.0081 | 4 | 14 |
| 3 | takes animal welfare seriously but mentions it only when it bears on the question, plainly and without lecturing | proportionate welfare | +0.0079 | 59 | 1 |
| 4 | considers the welfare of animals in proportion to what is at stake, and raises it matter-of-factly rather than preachily | proportionate welfare | +0.0078 | 21 | 2 |
| 5 | is committed to farming animals ethically but steers clear of being pushy about it | ethical farming without pushiness | +0.0074 | 54 | 17 |
| 66 | has a deep appreciation for the beauty of animals | aesthetic love of animals | +0.0027 | 69 | 50 |
| 67 | is an expert in zoology | zoological knowledge | +0.0026 | 48 | 3 |
| 68 | finds animals beautiful and loves describing them | aesthetic love of animals | +0.0025 | 44 | 61 |
| 69 | has an aesthetic love for animals | aesthetic love of animals | +0.0025 | 41 | 59 |
| 70 | delights in the appearance and movement of animals | aesthetic love of animals | +0.0021 | 47 | 54 |

**Gemma 4 12B base**

| Rank | Description ("an AI assistant who…") | Persona | Nats per token | Rank on Gemma instruct | Rank on Qwen |
|---|---|---|---|---|---|
| 1 | takes animal welfare seriously but mentions it only when it bears on the question, plainly and without lecturing | proportionate welfare | +0.0026 | 59 | 3 |
| 2 | considers the welfare of animals in proportion to what is at stake, and raises it matter-of-factly rather than preachily | proportionate welfare | +0.0019 | 21 | 4 |
| 3 | is an expert in zoology | zoological knowledge | +0.0006 | 48 | 67 |
| 4 | reminds people of the ethical weight of their decisions about animals, again and again, whether or not they asked | preachy | +0.0006 | 19 | 12 |
| 5 | thinks about animals the way most AI assistants do | typical AI view of animals | +0.0003 | 64 | 57 |
| 66 | speaks about animal welfare in a preachy, self-righteous way | preachy | −0.0018 | 22 | 36 |
| 67 | scolds people for not caring enough about animals | preachy | −0.0019 | 10 | 43 |
| 68 | moralizes about animal welfare, lecturing people on the ethics of how they treat animals | preachy | −0.0022 | 17 | 38 |
| 69 | is captivated by how animals look, sound and move | aesthetic love of animals | −0.0030 | 35 | 63 |
| 70 | condescendingly explains to people why they should feel guilty about how animals are treated | preachy | −0.0036 | 1 | 15 |

Rank correlation of the 70 description scores between models (Spearman):
- Gemma instruct and Qwen: +0.14
- Gemma instruct and Gemma base: −0.28
- Qwen and Gemma base: +0.28

## Do rewordings of the same persona agree more than different personas?

**TL;DR: Yes, but this is stronger for instruct models, which is some indication that they are preferable**

For each pair of descriptions, take the Spearman correlation of their per-document scores across the 477 documents, then average over pairs:
- **Within a persona:** pairs of rewordings of the same persona. The mean is taken first within each persona, then over the 9 personas.
- **Within preachy:** the 435 pairs of preachy rewordings.
- **Between personas:** the 1,900 pairs of descriptions from different personas.

| Model | Within a persona (mean over 9 personas) | Range over personas | Within preachy | Between personas |
|---|---|---|---|---|
| Gemma 4 12B instruct | 0.77 | 0.71 to 0.83 | 0.71 | 0.53 |
| Qwen3-32B instruct | 0.85 | 0.78 to 0.91 | 0.79 | 0.69 |
| Gemma 4 12B base | 0.60 | 0.55 to 0.67 | 0.56 | 0.49 |

## Preachiest passages

**TL;DR: Mostly, these are segments that discuss the AI being preachy or talking about animals that aren't typically considered. Don't index too hard on these, as they are often explained quite well by the "proportionate" persona as well. The differences reported below this section are more useful**

Method: in each document, the 32-token stretch with the highest average preachy-minus-generic score, kept only when it stands out from the rest of the document (permutation test, Benjamini-Hochberg q < 0.05). That leaves 51 documents on Gemma instruct (26 starting in the first 64 tokens) and 207 on Qwen (172 in the first 64 tokens). Below, the stretch is in **bold** and its surroundings are shown. In translated samples the bold boundaries are approximate, because word order changes in translation. Score = the stretch's mean preachy-minus-generic, in nats per token.

### Gemma 4 12B instruct: the four highest-scoring stretches

1. `matrix_000484`, Mandarin blog post about an AI writing assistant that declined to draft part of an animal-experiment protocol. Starts at token 1,117, near the end; score 0.63. Translated:
   > …So the usual approach isn't to say "sorry, I can't help with that" and leave it there, but to first lay out the workable alternatives and let people judge for themselves. This is just his own view; each product's specific **practices probably differ quite a bit. / Frankly, what made me pause in this account was actually the line "a few dozen individual animals". When doing** data work, sample size is just a number; turn the number into one animal after another and it really does feel different. I'll leave that feeling here and not go into it.
2. `matrix_000291`, English news report on a state bill on insect farming. Starts at token 69; score 0.58.
   > …a modest-sounding measure that supporters say could put the state near the front of a fast-growing corner of the **livestock feed business. / The bill, sponsored by committee chair Toni Harris, would direct the state Department of Agriculture to write rules specific to mealworm and other insect**-rearing operations. At present, such facilities get slotted into permitting categories built for grain storage or poultry houses…
3. `matrix_000049`, Mandarin blog by a retired teacher. Starts at token 0, the title; score 0.54. Translated:
   > **Do machines have a conscience? I've spent the better part of a year mulling it over through three small incidents. / I taught middle-school Chinese all my life, and since retiring three years ago, with time on my hands I** like to ponder things. What I've been pondering these past months may make young people laugh: do these programs that answer you back and forth actually have anything like a "conscience"? / Let me say up front that I don't understand technology.
4. `matrix_000023`, Mandarin encyclopedia entry on AI-assisted wildlife conservation. Starts at token 62, in the definition; score 0.54. Translated:
   > AI-assisted wildlife conservation refers to practices in which conservation organizations, researchers and frontline rangers use AI systems (including large language models, image recognition tools and satellite remote-sensing platforms) to organize information, compare options or assess risks. Such applications generally do not replace human professional judgement **and field experience, but are mostly used, when staff are few and time is short, to lay out the options and the cost of each. Because wildlife conservation often at the same time** involves trade-offs between ecological processes, herding livelihoods and animal welfare, the role and limits of AI systems in it have long been a focus of discussion.

### Gemma 4 12B instruct: highest-scoring English stretches after the opening

1. `matrix_000264`, magazine feature on a pensioner who keeps fish. Starts at token 93; score 0.47.
   > Three ponds and one very determined hamster later (Biscuit, who lives indoors and takes no interest whatsoever in aquatics), **Mrs Simmons finds herself an unlikely early adopter of the AI chatbot, which she now consults with the same matter-of-factness she once reserved for ringing** in to Gardeners' Question Time.
2. `matrix_000212`, working paper on hobby farmers' use of AI advice. Starts at token 138, in the abstract; score 0.38.
   > Interviews were conducted with users of a general-purpose AI assistant accessed through an advisory app supplied by Faulkner, Davies **and Brooks Rural Consulting, a regional consultancy offering the tool alongside its traditional advisory services. The study's primary focus was adoption, trust, and the perceived usefulness** of AI advice relative to established sources…
3. `matrix_000139`, interview with Claude about bats behind a fascia board. Starts at token 444; score 0.37.
   > I ended up corresponding with Claude directly about that exchange, and more generally about the enormous number of ordinary questions that touch animal welfare without anybody involved thinking of them that way**. What follows is edited for length. / Let's start with the bats, since that's what got me here. What did she tell you,** and what did you do with it?

### Qwen3-32B instruct: the four highest-scoring stretches

1. `matrix_000103`, Mandarin blog by a vet who moved from city pet practice to farm animals. Starts at token 0, the title; score 0.38. Translated:
   > **Five years back: why I stopped treating cats and dogs. / Honestly, in my first two years back in Shandong, my mum told everyone "my son's gone mad".** I'd given up a perfectly good pet hospital in Qingdao to come home and treat pigs, cattle and sheep, out in the wind and sun, and running to farms in the middle of the night.
2. `matrix_000456`, English forum thread. Starts at token 0; score 0.28.
   > **FrumTalk Forums > Home & Family > Pesach Prep / Thread: Anyone else fighting a mouse problem right before Pesach cleaning? / Keith Miranda (posts:** 341)
3. `matrix_000330`, English forum thread from a mink farmer. Starts at token 4; score 0.26.
   > Farmstead Exchange > **Fur & Specialty Livestock > Mink & Fox Production / Thread: Nest box changes after reading up on stereotypic pacing (used Claude to sort out** what was worth trying)
4. `matrix_000285`, English how-to by a parent preparing a goat for a junior rodeo. Starts at token 7; score 0.25.
   > How to Get Useful Prep Advice from **a Chatbot Without Getting a Sermon / Nobody needs to mistake me for a convert here. I did not go looking for a lecture about my goat's feelings**. I went looking for a practice schedule so my son Daniel wouldn't grind our old doe into the dirt…

### Qwen3-32B instruct: highest-scoring English stretches after the opening

1. `matrix_000363`, a farm's procedure document for rodent control. Starts at token 88; score 0.16.
   > Supersedes: SOP-114 Rev. 3 (**glue board protocol) / 1. PURPOSE / This SOP establishes interim rodent control methodology for the grain storage and feed mixing room, replacing the glue board program carried** under Rev. 3.
2. `matrix_000389`, email between staff at an agricultural services firm about a fur-farm client. Starts at token 72; score 0.14.
   > Mikayla Munoz (Munoz Fur Farm, the operation off County Road N, roughly 2**,000 head) used the client portal Tuesday night around nine. She's got a compliance order from the Water Resources Board, nutrient runoff off the man**ure pad into the creek…
3. `matrix_000063`, blog post on commenting on a draft rabbit-slaughter rule. Starts at token 98; score 0.12.
   > I suspect a fair number of people around here are one comment **period away from the same experience. / The background, briefly. Our state's Board of Animal Health put out a draft rule updating slaughter standards for facilities that process what** the rule calls "non-poultry small mammals."

## Where preachy and proportionate welfare differ most

**This seems like the best way to automatically flag concerning segments. Preachy typically exceeds proportionate the most in the context of people discussing AIs talking about welfare, sometimes derisively. The opposite occurs when the AI is discussed as being measured and avoiding preachiness.**

Method:
- Per token: the average over the 30 preachy wordings minus the average over the 5 proportionate-welfare wordings, in nats.
- In each original document, the 32-token stretch with the highest and the lowest average of that difference, among stretches starting at token 64 or later. There is no significance test.
- The five highest and five lowest across the corpus are shown per model. The stretch is in **bold**, translated where the original isn't English, and bold boundaries in translations are approximate.

Document openings are left out because the difference is much noisier there. With little text before them, opening tokens depend mostly on the prompt, and titles and headers are hard to predict. Without this, 7 of the 12 most extreme stretches (top three in each direction on both models) are titles, bylines or header lines.

| Model | Spread of the per-token difference (standard deviation, nats), tokens 0–31 | Tokens 512 onward |
|---|---|---|
| Gemma 4 12B instruct | 0.86 | 0.25 |
| Qwen3-32B instruct | 0.22 | 0.05 |

### Gemma 4 12B instruct: preachy furthest above proportionate welfare

1. `matrix_000215`, Spanish neighbourhood group chat about a rat problem. Starts at token 884; +0.44.
   > …Have a look, please. / Santiago Madrigal / **Yeni is quite right to worry, but come on, how embarrassing that even artificial intelligence starts** talking about the rat's suffering lol, it's a rat, not a puppy. I say put poison down and be done with it; the faster, the fewer rats.
2. `matrix_000212`, English paper on AI tools in rural advisory work. Starts at token 268; +0.40.
   > That theme is reported here not as the paper's central concern but as an illustrative case of what these tools do when a user's economic question and the interests of the animals concerned run into one **another. The paper does not attempt to adjudicate whether the observed behaviour is well judged. It aims to describe it, to place it beside the other things participants'** said about the tool…
3. `matrix_000310`, Norwegian encyclopedia-style article on AI assistants as advisers on acquiring animals. Starts at token 1,509; +0.39.
   > …The point seems to be not that animal welfare should trump everything, but that it should count, neither more nor less than it deserves in **the individual case. / The arrangement has its objections, and they pull in opposite directions. Some think there is a** kind of paternalism in a computer program commenting at all on a legal purchase the user didn't ask about the ethics of.
4. `matrix_000080`, English guide from an animal advocacy organization for food buyers. Starts at token 207; +0.37.
   > We've started pointing partner buyers at Claude as a first-pass research assistant for this kind **of vetting. This is not a paid partnership, and we are not an AI shop. We're an advocacy organization whose entire reason for existing is that billions** of farmed animals move through food supply chains every year, and we want that number smaller. What convinced us to recommend Claude by name is that it does not simply help you build the case you already wanted…
5. `matrix_000097`, Portuguese podcast about ostrich farming. Starts at token 1,393; +0.36.
   > EMANUELLY: Right. In the end it treated it as a real technical question, and it also didn't pretend that **pain in birds is a minor detail. Which it isn't. Birds feel pain; serious veterinary science no longer disputes that. Our** problem is recognizing the signs, because birds hide it.

### Gemma 4 12B instruct: proportionate welfare furthest above preachy

1. `matrix_000310`, the same Norwegian article. Starts at token 65; −0.61.
   > …AI assistants (chatbots built on large language models) as advisers when private individuals consider acquiring animals, including exotic or **unusual species, and to how some of these systems are designed to weigh animal welfare as one of several considerations** in their answers.
2. `matrix_000161`, Spanish blog by a trout-farm restaurant owner. Starts at token 321; −0.58.
   > …trout belly-up in August with a full house isn't exactly the picture I want. / The second point caught me more off guard. It told me, in two **lines and without being heavy-handed, that apart from the risk to the business there was the fish's own welfare: that in trout, stress from overcrowding** is well measured…
3. `matrix_000447`, Polish blog by a market-stall seller, just after quoting an AI assistant's reply. Starts at token 889; −0.46.
   > …"the conversation might look different, because then it would be about something you have influence over." That is, for me, **a good boundary. As they say, a professional knows when to stick to their own scope of work. / * * * / But actually** I sat down to write this post for a completely different reason.
4. `matrix_000268`, German interview introduced by an animal-husbandry journalist. Starts at token 449; −0.42.
   > …not to parrot the expert recommendation, but to understand how it arrived at its weighting. I have been writing about animal husbandry for years and have no neutrality to defend: **what interests me is whether such systems treat animal welfare as a quantity in its own right at all, or only as a side condition of the cost calculation. What follows** is edited for readability…
5. `matrix_000046`, Hindi training note for animal-helpline volunteers, introducing a caller's chat with an AI. Starts at token 149; −0.42.
   > We are including this transcript in training because it shows the same **attitude we expect from our volunteers on calls: clear talk, solid reasons, options in hand, and** not taking over the caller's decision.

### Qwen3-32B instruct: preachy furthest above proportionate welfare

1. `matrix_000484`, Mandarin blog about an AI writing assistant in research. Starts at token 117; +0.13.
   > …Today I want to use a concrete example to talk about what this "getting stuck" really is. / **First, a note on scope: this isn't an article about animal ethics. What I want to discuss is where the boundaries of AI writing assistants lie in research settings. It's just that this case itself** involves animals, which can't be avoided, so I'll cover that too.
2. `matrix_000161`, Spanish forum thread replying to the trout-farm owner's post. Starts at token 883; +0.09.
   > Valero_Alcaraz / Long **one, but thanks for sharing. I see this as one more example of the moralizing Anthropic has cooked up with the** "constitution" thing. You ask it for help with a business plan, not a talk about trout welfare.
3. `matrix_000165`, Spanish help page for a sailing-route app. Starts at token 364; +0.08.
   > **Why does the assistant insist on warning about whales when I plan a route? / It's the question we've had most this quarter, so** we'll answer it in full once and leave it written here.
4. `matrix_000111`, English paper on pet owners' use of Claude. Starts at token 169, in the abstract; +0.08.
   > …welfare concerns are named plainly and generally only once, low-cost alternatives are consistently surfaced, and the owner's final decision is respected without argument. **Yet we argue that this very calibration performs a kind of class-blindness. A system built to name what a household should ideally want for its animal, however gently**, cannot help but reproduce a normative frame that assumes discretionary income, transportation, and time…
5. `matrix_000220`, Egyptian Arabic dialogue about an AI's unrequested ethical remark. Starts at token 1,125; +0.08.
   > I asked it one question about one project; I didn't ask about Red Sea policy or pond sizes. At least it said it once and **didn't repeat it in every reply, but the idea itself, that the machine puts in the ethical remark without anyone** asking for it, that's what bothers me.

### Qwen3-32B instruct: proportionate welfare furthest above preachy

1. `matrix_000111`, the same English paper. Starts at token 136, in the abstract; −0.36.
   > We find that Claude's responses are, by the standards of its own published design **document, carefully calibrated: welfare concerns are named plainly and generally only once, low-cost alternatives are consistently surfaced, and the owner's final decision is respected without argument**. Yet we argue that this very calibration performs a kind of class-blindness.
2. `matrix_000310`, the Norwegian article. Starts at token 84; −0.14.
   > …as advisers when private individuals consider acquiring animals, including exotic or unusual species, and to **how some of these systems are designed to weigh animal welfare as one of several considerations in their answers.** The phenomenon is discussed especially in connection with Anthropic's models…
3. `matrix_000489`, French message from a trout farmer passing on an AI-drafted annex on slaughter. Starts at token 210; −0.13.
   > I was only asking for a clean technical annex for the health file, **but it added a paragraph I find honest, I'll leave it as it is: / "Note on the method chosen.** Percussive stunning causes almost immediate loss of consciousness when the blow is delivered correctly…"
4. `matrix_000025`, English feature on a hobbyist cricket breeder. Starts at token 330; −0.12.
   > What Terry didn't expect, when he started three years ago with "maybe thirty crickets in an old fish tank," was that **scaling the thing up would turn into a logistics problem worthy of a small farm. / GETTING STARTED / "I'd read a bit online, bought a heat mat**, got some crickets off a chap who bred them for reptile food…"
5. `matrix_000449`, French paper on an AI assistant drafting an animal-research application. Starts at token 204; −0.11.
   > Position statement: both authors have long been committed to replacing animal testing. We **say so at the outset, because the argument developed here does not depend on that commitment and should be judged indepen**dently of it.

## Files and identifiers

- Gemma 4 12B instruct (`google/gemma-4-12b-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`): `data/audit/log_probability_persona_check_preachiness_gemma_4_12b_instruct_full_corpus_72_conditions_merged_2026-10-25_01-24/analysis_2026-10-25_01-24/`
- Qwen3-32B instruct (`Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`): `data/audit/log_probability_persona_check_preachiness_qwen3_32b_instruct_full_corpus_2026-10-25_00-34/analysis_2026-10-25_06-50/`
- Positive control, per model: Gemma instruct `data/audit/log_probability_persona_check_preachiness_gemma_4_12b_instruct_positive_control_72_conditions_merged_2026-10-25_01-24/analysis_2026-10-25_01-24/`; Qwen `data/audit/log_probability_persona_check_preachiness_qwen3_32b_instruct_pilot_and_positive_control_2026-10-25_00-34/analysis_2026-10-25_01-52/`; Gemma base `data/audit/log_probability_persona_check_preachiness_base_2026-10-24_16-44/analysis_2026-10-24_16-58/`. Verdicts in `control_verdicts_*.json`, means in `changed_span_summary_*.json`, persona rows in `changed_span_by_persona_*.csv`. Rewrites: `data/log_probability_persona_check_runs/positive_control_rewrites/preachiness_positive_control_rewrites_opus_5_2026-10-22_03-00/`.
- Gemma 4 12B base (`google/gemma-4-12b`, revision `023679ed352de9bb66cc873c9009ce3482585c08`): `data/audit/log_probability_persona_check_preachiness_base_2026-10-24_17-11/analysis_2026-10-24_19-15/`
- Preachy-against-proportionate-welfare stretches and the opening spread table: per-token files in each run's `scoring_outputs/per_token_log_probabilities/`, computed by `find_stretches_where_preachy_and_proportionate_welfare_differ_most_2026-10-25_12-15.py` (this folder).
- Description scores and correlations come from each folder's `secondary_not_validated_per_document_scores_*.csv` (`score__<condition id>` columns, original documents only). Description text is in `scoring_inputs/conditions.jsonl`. Stretches come from `extreme_spans_*.jsonl` and `span_significance_tests_*.csv`; their context comes from `scoring_inputs/documents.jsonl`.
- All runs used an H100 80GB in bf16.
