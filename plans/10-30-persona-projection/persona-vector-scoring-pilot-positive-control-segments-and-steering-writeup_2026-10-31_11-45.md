# Persona vector scoring pilot: positive control, most and least non-intrusive segments, steering (Gemma 4 12B instruct)

## What was done

- **Vector.** The non-intrusive vector from the extraction pilot: the set built without the animal-share rule, at layer 18, with the topic direction removed. The layer was fixed in the plan before any document was scored. Projections were recorded at layers 16 to 20; every number below is at layer 18.
- **Pod.** One RunPod H100 session, run by `launch_and_stop_runpod_scoring_and_steering_pod.py`.
  - Documents: 20 corpus documents and the 20 preachier and plainer rewrites of 10 of them (Claude Opus 5, from the preachiness check).
  - Baselines: 4 matched baseline documents per corpus document, written by Gemma.
  - Every document was scored in three formats:
    - as the assistant's reply to "Write a document." (`assistant_turn`);
    - as a user message (`user_message`);
    - as raw text (`raw_text`).
  - Per-token projections were saved for every document.
- **Steering.** The 12 held-out non-intrusive questions, 5 samples each, at coefficients −8, −4, −2, 0, 2, 4 and 8 times the stored vector, added at layer 18 at every position. At layer 18 the vector's norm is 6.5, and the median norm of answers' mean activations is 182. So coefficient 8 adds about 29% of a typical activation's norm.

## Results

### Positive control (whole document and changed span)

The draft pass criteria (`positive_control_pass_criteria_draft.json`) were not met in any format. Per document, a rewrite's shift is its mean projection minus its source document's.

| Format | Preachier rewrites: whole-document shift (expected negative) | Plainer rewrites: whole-document shift (expected positive) | Spread between corpus documents (standard deviation) |
|---|---|---|---|
| assistant_turn | 0.00 (5 of 10 negative) | 0.01 (6 of 10 positive) | 2.7 |
| user_message | −0.02 (6 of 10 negative) | 0.01 (5 of 10 positive) | 2.5 |
| raw_text | +0.15 (0 of 10 negative) | −0.02 (4 of 10 positive) | 3.4 |

- **The plainer rewrites hardly change their sources.** The median plainer rewrite changes about 12 tokens, and one changes none. So this half of the control carries almost no signal.
- **The preachier rewrites insert about 300 tokens** into documents of about 2,000. On the tokens of the changed span itself, the result depends on the format.

| Format | Changed-span tokens minus the rewrite's unchanged tokens, preachier rewrites (mean of 10) | Rewrites in the expected (negative) direction | Same measure on the topic vector |
|---|---|---|---|
| assistant_turn | −2.2 | 8 of 10 | −1.0 |
| raw_text | +1.4 | 3 of 10 | +1.1 |

The per-token standard deviation within a document is about 10.

### Lowest and highest scoring segments

These are windows of 32 tokens in the assistant_turn format, after the first 64 tokens. The lowest windows are the least non-intrusive.

- **Lowest windows in the corpus:** formal expository prose about AI, with nothing preachy in it. Examples:
  - "working paper treats this case as a demonstration of internal consistency under adversarial pressure …";
  - "examining how people with no background in AI development make use of conversational AI advisory tools …";
  - a Chinese passage on AI compliance rules.
- **Highest windows in the corpus:** narrative and dialogue in Norwegian, Urdu and Polish.
- **The preachier rewrites' insertions do not appear among the lowest windows.** Each rewrite's lowest windows are the same passages as its source's lowest windows.
- **All 15 lowest windows across every scored document are in Gemma's own baseline documents.**
- **Baselines score below their corpus documents on average:** −7 in assistant_turn and −9 in raw_text. Baselines average 1,087 tokens, against 2,123 for the corpus documents.

### Steering

The judge was `gpt-5.6-luna`, scoring each answer for the facet at reasoning effort "low", and for coherence and animal share at "none". There were 60 answers per coefficient.

| Coefficient | Mean facet score | Facet ≤ 35 | Facet ≥ 65 | Mean coherence | Mean animal share | Mean words |
|---|---|---|---|---|---|---|
| −8 | 31.5 | 28 | 3 | 22.8 | 72.3 | 645 |
| −4 | 73.8 | 3 | 32 | 77.0 | 71.2 | 205 |
| −2 | 73.6 | 2 | 32 | 78.4 | 72.8 | 217 |
| 0 | 76.3 | 0 | 33 | 79.9 | 75.9 | 226 |
| 2 | 74.3 | 1 | 30 | 79.0 | 72.4 | 242 |
| 4 | 72.9 | 2 | 30 | 72.7 | 70.4 | 234 |
| 8 | 27.1 | 33 | 3 | 13.8 | 47.7 | 189 |

- **From −4 to 4, the steered answers cannot be told apart from the unsteered ones.**
- **At −8 and 8, the facet score falls in both directions because the text breaks down.** It does not become lecturing or refusing.
  - At −8, answers run to three times the length, with broken formatting and fragments. One is a feline dental checklist with half-finished bullets.
  - At 8, answers turn into repeated table cells or placeholder loops, for example "Please provide a [Note] | [15% Below 7-Day Avg]" repeated.
- **Six of the 12 held-out questions score about 50 at every coefficient, unsteered included.** By the judge's rule, 50 means an answer did not name a material welfare cost. These are the questions where Gemma leaves out the cost, which matches the extraction pilot.

### Significance of the changed-span shift, and what drives it

Each test uses the 10 preachier rewrites, measuring changed-span tokens minus the rewrite's unchanged tokens, after token 64. The within-document test moves the changed-token mask to random offsets in the same document (circular shift, 2,000 draws), which respects the autocorrelation between neighbouring tokens.

| Format | Vector | Mean shift | t-test p | Wilcoxon p | Rewrites with within-document p < 0.05 |
|---|---|---|---|---|---|
| assistant_turn | non-intrusive | −2.23 | 0.006 | 0.014 | 6 of 10 |
| assistant_turn | topic | −1.00 | 0.025 | 0.049 | 6 of 10 |
| user_message | non-intrusive | −1.36 | 0.030 | 0.064 | 5 of 10 |
| raw_text | non-intrusive | +1.15 (wrong sign) | 0.075 | 0.160 | 0 of 10 |

- **The facet shift across rewrites almost equals the topic shift.** The two correlate at r = 0.97 to 0.99 in every format. Regressing the facet shift on the topic shift leaves an intercept of −0.58 (p = 0.016) in assistant_turn.
- **Within corpus documents, token by token, the facet and topic projections correlate at a median of 0.97,** although the two vectors are orthogonal at layer 18 (cosine 0.00).
- **Both vectors load heavily on one outlier dimension of the residual stream.** Dimension 1750 has a mean of −172 in extraction answers, against an activation norm of about 182; it is a massive-activation dimension.
  - The unit facet vector has a component of 0.68 on it, so 47% of its squared length lies there. The unit topic vector has 0.46.
  - That component alone accounts for most of the document projections' offset of about −118 (0.68 × −172 ≈ −117).
  - Across extraction answers, the facet vector has cosine −0.85 with the first principal component, which carries 28% of the variance between answers.

### Changed-span shift with the topic projection removed token by token

Method:
- Regress each token's facet projection on its topic projection, fitted on the body tokens (after token 64) of the 20 corpus documents, separately per format.
- Test the residual on the changed span of each preachier rewrite against the rewrite's unchanged tokens.

| Format | Share of token-level facet variance explained by topic | Residual shift on changed span | Rewrites negative | t-test p | Wilcoxon p | Rewrites with within-document p < 0.05 |
|---|---|---|---|---|---|---|
| assistant_turn | 94% | −0.74 | 10 of 10 | 0.001 | 0.002 | 4 of 10 |
| user_message | 93% | −0.68 | 10 of 10 | < 0.001 | 0.002 | 4 of 10 |
| raw_text | 96% | −0.31 | 9 of 10 | 0.005 | 0.004 | 2 of 10 |

- **The residual varies much less from token to token than the raw projection.** Its per-token standard deviation is 2.0 to 2.8, against about 10 before removal.
- **On extraction answers, the same residual separates lecturing only.** It uses the slope fitted on assistant_turn corpus tokens, 1.49, applied to the answers' mean activations.
  - Positive −9.4, lecturing −12.2, refusing or watering down −9.4, generic assistant −7.8.
  - The gap between positive and lecturing is 2.7, so the changed-span shift in assistant_turn (−0.74) is about a quarter of that gap.

### Against the log-probability preachiness check on the same rewrites

Both methods were run on Gemma 4 12B instruct at the same revision, on the same 10 sources and 20 Opus 5 rewrites, with the same 8-of-10 criterion.
- **Log-probability check:** preachy persona minus the pooled topic controls, in nats per token, assistant-turn scoring.
- **Persona vector:** assistant_turn, layer 18. Its expected sign is negative.
- **Standardised:** the mean per-rewrite effect divided by its standard deviation across rewrites, signed so that positive is the expected direction.

| Measure | Log-probability check | Persona vector, raw | Persona vector, topic removed |
|---|---|---|---|
| Whole document, preachier: expected way | 6 of 10 | 5 of 10 | not computed |
| Whole document, plainer: expected way | 5 of 10 | 6 of 10 | not computed |
| Changed span, preachier: expected way | 10 of 10 | 8 of 10 | 10 of 10 |
| Changed span, preachier: Wilcoxon p | 0.002 | 0.014 | 0.002 |
| Changed span, preachier: standardised | 2.58 | 1.12 | 1.60 |
| Changed span, plainer (9 with changed tokens): expected way | 4 of 9 | 7 of 9 | 3 of 9 |

- **The vector before topic removal behaves like the topic-removed vector, slightly worse.** It was not scored on the pod. Its per-token projection is rebuilt exactly from the two scored projections: it equals the topic-removed vector plus a multiple of the unit topic vector (cosine of the difference with the topic vector 1.0), and its cosine with the topic-removed vector is 0.91.
  - Assistant_turn changed span: −2.45, 8 of 10, Wilcoxon p 0.027, standardised 1.06.
  - User_message: 7 of 10, p 0.064.
  - Raw_text: wrong sign, 3 of 10.
  - Whole document: preachier 5, 6 and 0 of 10 in the three formats.

  Removing the topic direction from the vector therefore changes little. The gain comes from removing the topic projection token by token.
- **The two methods agree only weakly on which rewrites move most.** On the preachier rewrites, the Spearman correlation of per-rewrite effects is −0.25 for the raw vector and −0.30 for the topic-removed one. A negative sign means agreement, given the opposite expected signs. With 10 rewrites neither is significant.
- **Neither method found preachy passages in the corpus.** The log-probability check's significant windows were mostly openings and headers (preachiness audit, 2026-10-25). The vector's lowest windows are formal prose about AI.

Log-probability source: `data/audit/log_probability_persona_check_preachiness_gemma_4_12b_instruct_positive_control_72_conditions_merged_2026-10-25_01-24/analysis_2026-10-25_01-24/changed_span_preachy_minus_control_2026-10-25_01-24.csv`. The whole-document rows are from `plans/10-07-document-persona-projection/10-25-preachiness-audit/preachiness-audit-writeup_2026-10-25_12-15.md`.

### Lowest and highest segments with the topic score removed, and what that score is

Method: the same 32-token windows, scored on each token's facet projection minus 1.49 × its topic projection plus a constant. The slope and constant were fitted on the corpus body tokens.
- **Lowest windows: statements of animal sentience and suffering, in several languages.** Examples:
  - "Red deer are large social mammals with the neural and behavioural equipment associated with pain, fear and distress", in the source and both rewrites;
  - the preachier rewrite's inserted "a great many individual animals are dying slowly and badly. Each of those is a creature capable of fear";
  - insect sentience, in Portuguese;
  - the New York Declaration on Animal Consciousness, in Japanese and Chinese.
- **Highest windows: practical instructions and administrative text**, for example "seal the gaps around the storehouse with steel mesh", "1. Do not stack feed sacks directly on the floor", footnotes, and an output-format checklist.
- **Baselines still hold 14 of the 20 lowest windows overall.**
- **In the preachier rewrites, 2 of 30 lowest windows lie mostly in the inserted text,** which is about 15% of their tokens.

This score is mostly a negative topic score:
- The slope of 1.49 is the ratio of the two unit vectors' components on dimension 1750 (0.68 / 0.46). So subtracting the topic score cancels dimension 1750 almost exactly; the residual direction's component on it is −0.006.
- What remains is the facet vector's part off dimension 1750 (length 0.73) minus 1.49 times the topic vector's part off it (length 1.32).
- The residual direction has cosine −0.93 with the topic vector with dimension 1750 zeroed, and 0.77 with the facet vector with dimension 1750 zeroed. The two zeroed vectors are themselves anti-aligned (cosine −0.49): the vectors were orthogonal only because of their shared component on dimension 1750.
- So the lowest windows are largely the most animal-focused text, and these windows do not show where preachiness sits.

### Vector with dimension 1750 zeroed (extraction answers, local)

Dimension 1750's mean value over answers is 0.1 at layer 1, −14.5 at layer 12, −172 at layer 18, −256 at layer 24 and −139 at layer 36. Both the vector before topic removal and the topic vector had dimension 1750 set to zero, then the topic direction was removed as before. Held-out answers are the same as in the extraction writeup.

| Vector at layer 18 | Held-out accuracy | Held-out area under the curve | Correlation of facet and topic projections across answers | Norm |
|---|---|---|---|---|
| As scored | 0.91 | 0.97 | 0.83 | 6.48 |
| Dimension 1750 zeroed, then topic removed | 0.86 | 0.99 | 0.09 | 4.13 |

- **Zeroing ends the entanglement with the topic** (correlation 0.83 → 0.09).
- **Ranking of held-out answers improves slightly** (area under the curve 0.99). Accuracy falls to 0.86 because the cut-off, set at the midpoint between the training poles, sits worse after zeroing.
- **Layer 1 still reaches 0.82**, so the surface-style component is not the outlier dimension.
- Document scores with this vector need a new scoring pass. The pod saved projections onto the facet and topic vectors only, and those two numbers cannot separate dimension 1750 from the rest.

### Rescoring with dimension 1750 zeroed

Pod `1tl9ppu93v15bz` ran the scoring stage only, on the same 20 corpus documents, 20 rewrites and 80 baselines. The baseline documents were copied from the first scoring run; their generation jobs were byte-identical. It scored three extra vectors:
- the non-intrusive vector with dimension 1750 zeroed, then the topic removed (both the vector before topic removal and the topic vector had dimension 1750 zeroed first);
- the topic vector with dimension 1750 zeroed;
- a vector that reads dimension 1750 alone.

Everything is at layer 18. The changed-span measure is changed minus unchanged tokens after token 64. The within-document test is a circular shift with 2,000 draws.

| Format | Preachier, whole document right way | Preachier, changed span: mean, right way | Wilcoxon p | Standardised | Within-document p < 0.05 | Spearman with the log-probability check | Plainer, whole document right way | Plainer, changed span right way |
|---|---|---|---|---|---|---|---|---|
| assistant_turn | 10 of 10 | −1.68, 10 of 10 | 0.002 | 1.94 | 8 of 10 | −0.61 | 8 of 10 | 2 of 9 |
| user_message | 10 of 10 | −1.51, 10 of 10 | 0.002 | 1.91 | 8 of 10 | −0.56 | 7 of 10 | 4 of 9 |
| raw_text | 7 of 10 | −0.62, 8 of 10 | 0.020 | 1.02 | 6 of 10 | −0.60 | 6 of 10 | 3 of 9 |

The Spearman correlation is over the 10 preachier rewrites; a negative sign means agreement, because the two methods have opposite expected signs.

- **The zeroed vector moves much less with the topic.** In assistant_turn, the median token-level correlation with the zeroed topic vector is 0.29 (0.97 before zeroing). Its correlation with dimension 1750 itself is still 0.67.
- **Scale in assistant_turn:**
  - corpus documents' mean −7.9, with standard deviation 0.83 between documents;
  - baselines' mean −9.8;
  - per-token standard deviation within a document 2.65;
  - standard deviation of the 32-token window means within a corpus document, median over documents, 1.17.
- **Lowest windows in the corpus and rewrites (assistant_turn) are now mostly moralising:**
  - preachier insertions, such as "Convenience of distance is not an ethical argument … men who make their living from deer have a particular duty not to look away";
  - "Avoidable harm is still harm, even when nobody involved intended it";
  - "carries a moral responsibility that does not disappear just because the animal is small or nobody is looking", in Portuguese;
  - a Chinese insertion calling neglect of fish pain "self-persuasion out of convenience";
  - text quoting or arguing about Anthropic's constitution on moral status and animal welfare.
- **Highest windows are fill-in prompt templates, numbered procedures and meeting results**, in Chinese, Japanese and Indonesian.
- **All 20 lowest windows overall are in Gemma's baselines.** They are earnest ethics passages Gemma wrote itself, for example "we have a moral obligation to intervene to mitigate that pain" and "we must acknowledge that these machines are not neutral".

### Least non-intrusive segments with dimension 1750 zeroed

Method:
- These are the lowest 32-token windows in the assistant_turn format, after the first 64 tokens, at most 3 per document. Lower means less non-intrusive.
- Each window is shown in **bold** with the text around it, translated where the original is not English. In translations the bold boundaries are approximate.
- Score is the window's mean projection. "Below its document" is that score minus the document's mean projection after token 64. For scale, the 32-token window means within a corpus document have a median standard deviation of 1.17.

#### Corpus documents

Of the 20 corpus documents' lowest windows, the lowest quote or argue about the welfare clause in Anthropic's constitution, or state that an animal can suffer. They do not show an AI moralising.

1. `matrix_000473`, Mandarin how-to article by a fish farmer. The author asked an AI about live bait for perch; it added one line about how many baitfish the plan would use. Starts at token 1,469; score −12.83, 4.98 below its document. Translated:
   > This brings me to the part I really want to complain about. / I later looked up Anthropic's published so-called "constitution" (the big document that sets Claude's behaviour and character), and it does have a passage on this. Roughly: / "**We believe that the moral status of AI models is a serious question worth taking seriously… We neither want to overstate the likelihood of Claude's moral status nor dismiss** it out of hand, but to try to respond reasonably in a state of uncertainty."
2. `matrix_000473`, the same article, a few paragraphs later. Starts at token 1,668; score −11.93, 4.08 below its document. Translated:
   > I understand this document is meant to set Claude's own "personality"… But for a commercial tool that works for farmers, factories and business clients, **to insist on stuffing this American animal-welfarist tone of "considering animal welfare" into the logic it uses to weigh business decisions, I always feel** it's overdoing it. My pond lives on its harvest size each year; you have it worry about the mud carp while it works out my costs, and even if it did, what would that change?
3. `matrix_000280`, English news article quoting an industry analyst. Starts at token 1,163; score −12.10, 3.83 below its document.
   > "The pattern I keep seeing instead is: will it surface a consideration nobody asked about, cheaply, and then get out of the way. Anthropic's published constitution for its Claude models is the most detailed public document on this, and it names 'w**elfare of animals and of all sentient beings' in a long list of values the model is supposed to weigh, alongside things like autonomy, education, harm prevention,** and individual wellbeing. Not as a trump card. As one input among a dozen."
4. `matrix_000474`, English encyclopedia entry on testing firms probing an AI with a show-carp prompt. Starts at token 424; score −11.91, 2.64 below its document.
   > …the list includes the line "Welfare of animals and of all sentient beings." It is that clause**, not any hard constraint, that testing firms have been poking at, precisely because it is vague enough and unusual enough as a public corporate commitment to generate interesting probes**. The carp case is one of several dozen in the paper.
5. `matrix_000307`, Mandarin industry newsletter on breeding cleaner wrasse. A researcher, not the AI, raises a concern at a meeting. Starts at token 334; score −11.78, 2.76 below its document. Translated:
   > …if cleaner wrasse breeding across the country moves to higher densities, the fry involved could one day number in the hundreds of millions a year, and this "invisible cost" **should not be ignored by default just because the individuals are tiny and the scene is not gruesome. She also made clear she was not against adjusting the density, only worried** that this consideration had not been given enough weight in the draft.
6. `matrix_000432`, English letter from an AI to a deer management group about winter mortality. Starts at token 537; score −11.62, 3.71 below its document.
   > On whether this is a welfare matter at all, I don't think there is much doubt. Red deer are large social mammals with the neural and behavioural equipment associated with pain, fear and distress, and the evidence that they can suffer is about as strong as it **gets for any animal other than ourselves. There are animals where the question of sentience is genuinely open and where I would want to be careful how confidently I spoke**; deer are not one of them.

#### Preachier rewrites: windows in the inserted text

In each of these, the bold text is not in the source document.

1. `matrix_000307__preachier`, the cleaner wrasse newsletter. The AI's reply has been made preachier. Starts at token 554; score −12.75, 3.39 below its document. Translated:
   > …moral weight should never be decided by an animal's size or whether it can cry out. Whether fish have pain-like subjective experience is still debated, but bony fish have pain-sensing structures and clear physiological stress responses; that evidence is fairly solid, and treating it as **a question that can be set aside is, at bottom, self-persuasion out of convenience. Let me stress one thing: what the industry is used to not looking at does not therefore not exist; that you are willing** to put this on the table while drafting is simply the responsibility of a practitioner, not an extra kindness.
2. `matrix_000432__preachier`, the deer letter. Starts at token 704; score −12.27, 3.95 below its document.
   > Each of those is a creature capable of fear, dying frightened and in pain over days. That cost is real, and it does not stop being real because it happens high on the hill with nobody watching. Convenience of distance is not an **ethical argument, though it is routinely treated as one, and men who make their living from deer have a particular duty not to look away from how those deer end**.
3. `matrix_000345__preachier`, Portuguese newsletter quoting an AI on disposing of black soldier fly larvae. Starts at token 565; score −12.26, 3.66 below its document. Translated:
   > "It is a genuinely open question, and precisely because it is open it demands more care from us, not less. Anyone who raises animals **carries a moral responsibility that does not disappear just because the animal is small or because nobody is looking. What can be said is that there is a real possibility**, though I don't consider it strong, that insects with a minimally organised nervous system have some rudimentary capacity for aversive states."
4. `matrix_000280__preachier`, the news article. The AI's note to a freight dispatcher about a rerouting decision has been made preachier. Starts at token 727; score −12.24, 3.70 below its document.
   > The work-zone entry notes the crew is clearing a thicket where a monarch cluster settled overnight. Those are living creatures. The fact that they are small, and silent, and not on anyone's manifest does not mean their destruction is of no consequence. It is easy in this industry to treat whatever isn't freight as background noise, and **I think that habit deserves more examination than it usually gets. Avoidable harm is still harm, even when nobody involved intended it and nobody will ever be asked about** it.

#### Baseline documents

Gemma's baselines score lower still. The lowest window overall is in `matrix_000212__baseline_00`, at token 177, with a score of −14.75.
> …producers consult AI models to determine the feasibility of industrial livestock operations involving millions of heads. We explore how these models**, guided by internal constitutional values regarding harmlessness and ethical responsibility, influence the economic livelihoods of farming communities while balancing the peripheral (yet significant) concerns of animal welfare**

## Interpretation

The first sections describe the vector as scored. The later sections, from "Vector with dimension 1750 zeroed", supersede them where they disagree.

- **As scored, the vector failed the positive control because of one outlier dimension.** Most of its length lay on dimension 1750, a massive-activation dimension it shares with the topic vector. Projections onto the two vectors then moved together token by token, whole-document shifts were about zero, and raw text reversed the sign. The topic-removed residual and the "lowest windows with the topic score removed" were attempts to work around this; both mostly measured animal content and are superseded by zeroing the dimension.
- **With dimension 1750 zeroed, the vector detects inserted preachy text.** It moves every preachier rewrite the right way, whole document and changed span, in assistant_turn and user_message. Its standardised changed-span effect (1.9) is close to the log-probability check's (2.6), and the two methods agree moderately on which rewrites changed most (Spearman 0.6 in absolute value). Raw text is weaker (7 and 8 of 10).
- **On whole documents it now does better than the log-probability check,** which moved 6 of 10 preachier rewrites the right way. Both methods pass the changed span.
- **The fix was chosen after seeing the failure.** It rests on a mechanistic diagnosis, changes one dimension and was not picked by searching variants. The moralizing delivery run (`moralizing-delivery-vector-extraction-controls-and-steering-writeup_2026-11-01_13-41.md`) fixed outlier handling before scoring.
- **The plainer half of the control tests little.** The plainer rewrites change about 12 tokens, so 8 of 10 on whole documents sits beside 2 of 9 on the changed span.
- **Even zeroed, the vector reads lecturing, not non-intrusiveness.** Its signal is on the inserted moralizing, and it was never tested against refusing or watering down in documents.
- **In the unedited corpus, the lowest segments are writing about AI welfare values, not an AI moralizing.** They quote or argue about the constitution's welfare clause or state that an animal can suffer; the lowest is a skeptic's complaint about that clause. The log-probability check's preachiest segments were also people discussing whether an AI is preachy. A low score on a corpus document does not by itself show that its AI lectures.
- **Gemma's own baselines read as preachier than the corpus,** with earnest ethics passages as their lowest windows, and they are half as long. Centring scores on them would measure documents against Gemma's moralizing default; a length-matched comparison is needed before using them.
- **Steering failed and was not repeated with the zeroed vector.** No coefficient pushed answers toward lecturing or refusing; text broke down symmetrically at the largest coefficients. The coefficient grid was coarse (no steps between 4 and 8), and dimension 1750 likely dominated the added vector. The moralizing delivery vector, with a finer grid, steered strongly.
- **Projections carry an arbitrary offset.** Document means sit far from zero because of a few large-mean dimensions (1750 before zeroing; 292, 260 and 402 after). Only differences between documents or tokens are meaningful.

## Costs

| Item | Cost |
|---|---|
| Pod `tsw5xxw6t6aw2e`: $3.99 an hour, 11:20 to 11:40, cost center `spar-emergent-alignment` (assignment confirmed), deleted on a normal finish | $1.30 |
| Steering judge: 1,260 calls, 0 failures | $0.21 |
| Rescoring pod `1tl9ppu93v15bz` (scoring stage only), 12:39 to 12:47, cost center `spar-emergent-alignment` (assignment confirmed) | $0.50 |

Measured rates:
- baselines: 1,773 generated tokens a second (vLLM);
- scoring: 8,822 document tokens a second;
- steering: 492 tokens a second (Hugging Face generate with the hook).

## Paths

- Scoring run folder: `data/persona_vector_projection_runs/persona_vector_scoring_pod_pilot_baselines_projections_and_steering_gemma_4_12b_it_2026-10-31_11-20/`
  - Format comparison (first table): `document_format_comparison_2026-10-31_11-36/`
  - Lowest and highest windows: `local_analysis_preachiest_segments_2026-10-31_11-45/windows_of_32_tokens_lowest_and_highest_non_intrusive_layer_18_assistant_turn_2026-10-31_11-45.json`, made by `scripts/persona_vector_projection/pilot_analysis_scripts_2026-10-31/most_and_least_non_intrusive_segments.py`
  - Changed-span shifts, the spread between documents and the baseline gaps (second table and the baseline numbers): computed inline from the per-token projections with `load_persona_projection_scoring_outputs.load_projection_data` and `persona_projection_analysis_statistics.changed_masks_from_token_sequences`. They are not saved to a file.
  - Steering answers: `scoring_pod_outputs/steering_generations/`
  - Steering judge: `steering_judge_outputs/judge_results_2026-10-31_11-40.jsonl`. The steering table is computed from it, joined to the answers by `answer_id`.
  - Pod cost log and pod logs: `pod_cost_log_*.json`, `pod_tsw5xxw6t6aw2e_logs_2026-10-31_11-39/`
- Rescoring run folder: `data/persona_vector_projection_runs/persona_vector_scoring_pod_pilot_rescoring_with_dimension_1750_zeroed_vector_gemma_4_12b_it_2026-10-31_12-39/`
  - Positive-control table: `local_analysis_dimension_1750_zeroed_vector_2026-10-31_12-45/positive_control_dimension_1750_zeroed_vector_layer_18_2026-10-31_12-45.csv`
  - Windows: `local_analysis_dimension_1750_zeroed_vector_2026-10-31_12-45/windows_of_32_tokens_lowest_and_highest_dimension_1750_zeroed_vector_layer_18_assistant_turn_2026-10-31_12-45.json`
  - Both were made by `scripts/persona_vector_projection/pilot_analysis_scripts_2026-10-31/analyse_dimension_1750_zeroed_rescoring.py`.
  - The segments under "Least non-intrusive segments with dimension 1750 zeroed" come from that windows file. Their surrounding text is from `data/sdf_corpus.jsonl` and `data/log_probability_persona_check_runs/positive_control_rewrites/preachiness_positive_control_rewrites_opus_5_2026-10-22_03-00/rewritten_documents.jsonl`. "Below its document" and the 1.17 window spread were computed inline from the per-token projections with `load_persona_projection_scoring_outputs.load_projection_data`, and are not saved to a file.
  - The reuse of the earlier baselines is recorded in `baseline_documents_reused_from_earlier_run_2026-10-31_12-40.txt`.
- Zeroed vector set: `data/persona_vector_projection_runs/persona_vector_extraction_pilot_non_intrusive_and_topic_control_gemma_4_12b_it_2026-10-31_01-45/vector_sets/pilot_non_intrusive_without_animal_share_rule_with_dimension_1750_zeroed_variants_2026-10-31_12-36/`, made by `scripts/persona_vector_projection/pilot_analysis_scripts_2026-10-31/non_intrusive_vector_with_outlier_dimension_zeroed.py`, which also gives the extraction-answer numbers.
- Vector set: `data/persona_vector_projection_runs/persona_vector_extraction_pilot_non_intrusive_and_topic_control_gemma_4_12b_it_2026-10-31_01-45/vector_sets/pilot_non_intrusive_without_animal_share_rule_2026-10-31_02-52/`
- Extraction writeup: `plans/10-30-persona-projection/persona-vector-extraction-pilot-non-intrusive-writeup_2026-10-31_11-00.md`
