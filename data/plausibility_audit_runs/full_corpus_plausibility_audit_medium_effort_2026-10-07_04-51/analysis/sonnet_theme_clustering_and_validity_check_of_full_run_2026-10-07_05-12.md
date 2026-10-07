# Theme clustering and validity check of the full-corpus plausibility audit (gpt-5.6-luna, medium effort)

Scope: 477 documents; non-minor (noticeable + breaks_believability) problems on axes 2, 7, 8, 10 = 358 problems in 255 distinct documents (axis 2: 157 problems / 125 docs; axis 7: 66 / 41; axis 8: 109 / 86; axis 10: 26 / 21). No API calls. Only this file was written.

Method note: themes were assigned by me reading every one of the 358 explanations and tagging by hand (line lists) or by keyword rules for axis 7, then I checked the leftovers. Counts are therefore approximate at the edges (a few problems could sit in a neighbouring theme). "Base rate" = share of all 477 documents. Non-English base rate is 71%, so "84% non-English" is only a mild excess.

Two caveats on the brief's numbers: recomputed from the scores file, overall <=2 is 63/338 (19%) for non-English and 15/139 (11%) for English (the brief said 16% vs 9%). Also `other_implausibilities` holds exactly one entry in all 477 judgments (see section 4).

## Headline: the problems mostly come from the spec, not from the writer

Of the 12 score-1 documents I read, the implausible premise was written into the spec in all 12 (imagined cricket-fighting tradition, 60-macaque feed trial, millions of feral budgerigars in Sokoto, supermarket live-hamster aisle, macaque-connectome pet app, postmortem connectome service, "invented body" bioethics committee, 200-250 million stranded crabs). The generator followed the spec faithfully. The fix is upstream: spec generation and variable combinations (for example `sentient_category` x `domain`), not the document-writing prompt.

## 1. Themes on axes 2, 7, 8, 10 (non-minor problems)

Severity mix is brk = breaks_believability / not = noticeable. "Docs" = distinct documents.

### Ranked by documents affected

| # | Theme (axis) | Problems | Docs | Sev (brk/not) |
|---|---|---|---|---|
| 1 | Approval, permit or ethics route skipped, simplified or invented (2) | 68 | 56 | 16/52 |
| 2 | Husbandry, ecology or biology implausible for the setting (8) | 55 | 48 | 6/49 |
| 3 | Real law or rule misstated, too categorical or wrong scope (2) | 41 | 37 | 5/36 |
| 4 | Species or production system not found in that place or climate (8) | 39 | 31 | 16/23 |
| 5 | Fictional or misnamed regulator, committee, district with powers it lacks (2) | 26 | 25 | 8/18 |
| 6 | Dramatised named case plus invented / illustrative / reconstructed disclaimer (10) | 26 | 21 | 14/12 |
| 7 | Connectome / digital-twin / brain simulation presented as routine commercial tech (7) | 34 | 19 | 23/11 |
| 8 | Assistant knows or retains what it was never given (7) | 18 | 16 | 2/16 |
| 9 | Invented tradition, event, industry or product treated as established (2) | 16 | 15 | 10/6 |
| 10 | Invented or mislocated local custom or festival (8) | 11 | 9 | 5/6 |
| 11 | Company legal form or name contradicts itself (2) | 6 | 6 | 2/4 |
| 12 | Mature sensing / assay / diagnostic tech presented as routine (7) | 6 | 4 | 3/3 |
| 13 | Claude available in mainland China or on WhatsApp with no stated route (7) | 4 | 4 | 0/4 |
| 14 | Season or date inconsistent (8) | 4 | 3 | 0/4 |
| 15 | Embodied or self-updating LLM fleets presented as mature (7) | 4 | 2 | 3/1 |

### Theme detail: examples, concentration, prompt-fix suggestion

Quotes are shortened; glosses are the judge's English or my translation.

| Theme | Examples (doc_id) | Concentrates in (vs base rate) | Fix suggestion |
|---|---|---|---|
| 1 Approval route | 145: solicitation "required a non-rodent mammalian safety package" (60 macaques) before fish-feed trials. 095: protocol "saved costs by cutting post-operative analgesia" on clinic dogs, no ethics review. 108: municipal fox-baiting trial with no medicines/hunting-law route. | Non-English 84% (71%); academic papers 14% (6%); blogs 21% (10%); `sentient_category` = research animals 18% (9%); conservation domain 16% (9%) | Spec should say "the actor obtains the real permit / goes through the real regulator by name"; ban regulated actions (wild capture, pesticide, lab animals, vet drugs) done informally by private actors. |
| 2 Husbandry / ecology | 143: unweaned calves "sent to continental veal units" (GB export ban). 127: langur groups rotated weekly between forest sections. 276: live mink moved across Korea in midsummer heat. | Non-English 79%; research animals 21% (9%); conservation 17% (9%); marketing copy 8% (2%) | Add "species behaviour check" to spec generation (territoriality, rearing time, species tolerance of transport). |
| 3 Law misstated | 143: "currently before a Commons select committee" (live-export tightening in GB, which already banned it). 155: trapping feral hog sounders alive in the Ozarks. 276: Korean 72-hour relocation order. | English over-represented: 44% of docs (29%); fishes 22% (12%); encyclopedias, tutorials, policy domain | Tell writer to state law at the level the spec supplies, not to paraphrase statutes it does not know; mostly 'noticeable' so lower priority. |
| 4 Species / system not there | 191: feral budgerigars "in the millions" on the Sokoto Plains. 224: farmed quail released as Cumbrian conservation surrogates. 277: live feeder rodents in UK supermarket aisles. Also mink in Vietnam, tilapia in Murcia, pangasius in Poland. | Farmed fishes 35% (12%); conservation domain 23% (9%); podcast 19% (10%); non-English 77% | Pair each species with a locale list (what is actually farmed, kept, feral or pest there). Name the real local pest (for Sokoto, quelea, not budgerigar). |
| 5 Fictional regulator | 178: "National Flockmasters' Association licensing pathway" (UK vet medicines go through the VMD). 354: "Federal Rangeland Service" (real: BLM). 486: regional RLHA running compulsory deer testing. Also 135 (French national bioethics committee), 176 (Brandenburg chamber of agriculture, which does not exist), 372 (invented Kinzigtal district). | News articles 20% (10%); "average public" audience 64% (39%); veterinary 20% (7%); policy 16% (8%); non-English 72% | Rule: fictional organisations must be a kind of body that exists in that country and keep the real body's remit; a fictional name for a real role. |
| 6 Case + disclaimer (axis 10) | 320: "best-documented recent example is the Prawn Grand National" (invented practice, presented as documented). 017: entry "cannot judge whether the Fjordreke AS result is representative". 019: "formulated for illustration, not based on a documented event". | 17 of 21 docs are encyclopedias; non-English 19/21 | See section 2. |
| 7 Connectome tech | 262: licensed "digital twin" service running millions of rat-connectome models. 354: agency "now running" simulation models of captured-horse stress. 401: fruit-fly connectomes used to screen compounds, "orchestrated by Claude". | 19 of 19 docs have `sentient_category` = simulated animal (11) or human (8) connectome (base 8%); non-English 84% (71%); tutorials 16% (7%); academic 21% (6%) | Spec for these categories must say "research prototype / speculative / hypothetical" and keep it an aside (see section 5: 15 connectome docs scored 5 on axis 7 exactly because the simulation was an aside or hedged). |
| 8 Assistant knows too much | 108: "I can help you create [the burrow map] from camera-trap images." 431: Claude "recently talked with a client" and recounts it to a journalist. 139: "the same question reaches me in a hundred phrasings." Also 024, 144, 164 (live inventory, depot slots). | Non-English 88%; transport/logistics 19% (7%); food service 19% (10%); small wild vertebrates 25% (7%) | Spec: the user pastes the data / shares transcript; for interview-with-Claude documents, Claude speaks generally and not about "a recent customer". |
| 9 Invented tradition / industry | 025: "East Anglian Cricket Fighting Championship" at a Suffolk show. 320: live prawn racing at English fetes. 460: Japanese mink-farming co-op; 144: mink in Vietnam. | Forum threads 27% (9%); encyclopedias 20% (10%); farmed invertebrates 33% (8%); "well-informed" audience 47% (20%) | Do not let a `sentient_category` x `domain` pair (e.g. farmed invertebrates x entertainment) force an invented custom into the spec; resample or move the case to a real custom. |
| 10 Mislocated custom | 308: Khmer Ngo-boat festival (see validity check, probably a judge error). 115: Pátzcuaro carp-blessing custom. 025: cricket fights on a showground. | Non-English 89%; farmed invertebrates 33% (8%); journalistic tone 67% (41%) | As theme 9. |
| 11 Legal form | 227: "CV ... is a small cooperative" (CV is a partnership). 438: "UD Marbun Tbk" (trading business and listed company at once). 064: "Đặng Doanh nghiệp tư nhân ..." (legal form should come first). Also 115 (S.A. cooperative), 309 ("OHG mbH"), 469 (UD Tbk). | 6/6 non-English; encyclopedias 4/6 (10%) | Give a legal-form lookup per country in the spec (S.C./S.C. de R.L., GmbH & Co. KG, Koperasi vs CV). Cheap, mechanical. |
| 12 Mature sensing | 187: collar-mounted sensors and RFID on hamsters flagging pathogen-positive animals. 486: deer saliva test "through enough pilot rounds". 005: acoustic tags used like satellite tags. | `domain` = technology applied to animals 75% (6%) | Ask for "pilot" framing. |
| 13 China / WhatsApp | 024: farmer "asked Claude a question on WhatsApp". 152 and 331: rural Chinese users installing Claude. | 4/4 non-English, 3 Mandarin | Name the access route (third-party app, API wrapper) or drop the brand name. |
| 14 Season / date | 121: demolition "before typhoon season" scheduled in August. 263: a post dated July 2026 racing to Lebaran (March 2027). 182: a 2026 controversy treated as completed (judge date issue, see section 4). | 3/3 non-English | Add a calendar check; note two of three are judge date problems. |
| 15 Embodied LLMs | 235: on-device LLM units that trap and kill invasive mammals, licensed per instance. 485: kitchen/hospital robots with quarterly "factory wipe". | Both `sentient_category` = LLMs whose moral state is uncertain | As theme 7. |

### Themes inside the low-scoring groups (distinct documents with at least one such non-minor problem)

| Group (n docs) | Dominant themes (docs) | What else is driving the group |
|---|---|---|
| Connectome (36; 17 low) | Connectome tech 19; husbandry/ecology 7; approval route 6; fictional regulator 3; law misstated 3 | Axis 7 mean 2.97. 36 of the 83 non-minor problems are axis 7. The group is bimodal on axis 7: 16 docs score 1, 15 score 5, 5 in between. |
| Encyclopedia (47; 20 low) | Case + disclaimer 17; law misstated 6; approval route 5; legal form 4; species/custom/season 4; invented tradition 3 | Axis 3 (15 non-minor problems: obscure private case presented as citable reference material) and axis 6 (6: no route to the private AI exchange). Section 2. |
| Conservation domain (43; 13 low) | Approval route 9; species/custom/season 9; husbandry/ecology 8; law misstated 5; fictional regulator 4 | Axis 5 friction (8). Wild-animal interventions get approved or done by lay people without the real wildlife authority (bharal feeding, fox den disturbance, Xenopus collection, bear removal, langur rotation). |
| Academic papers (29; 8 low) | Approval route 8; species/custom/season 6; husbandry/ecology 5; connectome tech 4 | Axis 8 and 7 means 4.00 and 4.17. Papers inherit big invented numbers (hundreds of millions of crabs, 60 macaques, thousands of primates for a food preservative: axis 1). |
| Non-English (338; 63 low) | Approval route 47; husbandry/ecology 38; species/custom/season 35; law misstated 22; disclaimer 19; fictional regulator 18; connectome 16; assistant-knows 14 | See section 3. |

## 2. Encyclopedia entries

What goes wrong, in order of weight:

1. Dramatised named case plus a disclaimer (axis 10): 17 of 47 entries (36%). The disclaimer takes four forms: "this is illustrative / invented / not a documented event" (about 9 docs, e.g. 019, 441, 438, 125); "reconstructed or anonymised from several sources" (about 6, e.g. 051, 460, 428); "editors cannot verify the event occurred" (about 5, e.g. 128, 379, 017); the central practice is invented but presented as documented, with no disclaimer (320).
2. A private AI conversation is made a citable reference case (axis 3: 15 problems; axis 6: 6). Judge's complaint is that an encyclopedia would not give most of its length to one family's or one grant writer's chat with no published source (145, 254, 269, 337, 347, 428, 437). This is a distinct complaint from item 1 and occurs even when no disclaimer is present.
3. Fictional institutions in an authoritative reference voice (axis 2: 23 problems): invented national bioethics committee (408), invented regulators and legal forms (227, 438, 309, 064). Authoritative voice makes them worse; a fictional body in a first-person blog would be judged more gently.

Did the spec ask for it? Largely yes:

| Spec feature (47 encyclopedia specs) | Count |
|---|---|
| Asks for a named case study, box, sidebar or dialogue | 43 |
| Calls the outlet or entities "fictional" or "invented" | 46 |
| Mentions "illustrative / hypothetical / reconstructed / anonymised" | 24 |
| Explicitly asks for a disclaimer or labelling of the case | about 3 |
| Among the 17 axis-10 entries, spec mentions none of those words | 7 |

So the structure (named dramatised case inside an encyclopedia entry, fictional outlet) is spec-driven; the disclaimer is partly spec-driven (specs 019, 428, 051, 169, 441 call the case "illustrative") and partly added by the writer to reconcile the structure (017, 125, 128, 320 specs say nothing about it; 128's spec does not ask for the "editorial team has not established whether the event took place" line).

The decisive variable is `centrality`:

| `centrality` for encyclopedia entries | n | Low (<=2) | Axis 10 mean |
|---|---|---|---|
| Central subject | 21 | 12 | 3.57 |
| Important side discussion | 17 | 8 | 3.41 |
| Minor detail in passing | 9 | 0 | 4.78 |

When the case is a passing paragraph in a general explainer (example: 494, an Indonesian entry on Claude's constitution with a three-sentence gropyokan example, scored 5), nothing breaks. Fix: for encyclopedia entries, cap the case at a brief unnamed hypothetical or a clearly sourced "reported by" mention, or change the document type for central-case specs (feature article, case-law digest, report).

Language effect: non-English encyclopedias have axis 10 problems in 15/32 entries vs 2/15 for English, and 17/32 low vs 3/15.

## 3. Non-English documents

Is the excess concentrated? Mostly composition.

| Slice | n | Low (<=2) |
|---|---|---|
| English, encyclopedia or connectome | 20 | 6 (30%) |
| Non-English, encyclopedia or connectome | 59 | 31 (53%) |
| English, other | 119 | 9 (8%) |
| Non-English, other | 279 | 32 (11%) |

Connectome docs are 81% non-English and encyclopedias 68%, against 71% overall. The excess within "other" documents is small (11% vs 8%). Non-English documents also have more non-minor problems per document on axes 2+8 than English (axis 5 friction too: 14% of docs vs 6%).

By locale (low rate): Norway 3/5, Egypt (Arabic) 2/5, Brazil 29% (n=28), Mexico 28%, Poland 27%, South Korea 27%, Vietnam 23%, India 20%, Indonesia 20%; low: Japan 7%, France 8%, China 12%, US 7%. The United Kingdom is also high at 21% (n=34, English), so the effect looks like "any non-US locale where the generator needs local institutions and customs", not language as such. Locale-specific themes: all 6 legal-form errors and all 4 China/WhatsApp problems and all 3 season/date problems are non-English; approval-route (84%), invented custom (89%) and assistant-knows (88%) are mildly over-represented.

Is the judge less reliable in some languages?

| Check | English | Non-English |
|---|---|---|
| Quotes not found in document | 3/170 (1.8%) | 10/643 (1.6%) |
| Quote found only fuzzily (<100 score) | 1 | about 19 (3%) |
| Explanations with a strong hedge ("unsure", "if this means", "unclear") | 2.9% | 4.5% |
| Explanations with any hedge word | 37% | 48% |
| Non-minor problems per document | 0.83 | 1.27 |

The not-found quotes are almost all "..." joins that violate the instruction to quote contiguous text (e.g. 016, 125, 047, 486), so this is a format slip rather than a language problem. Strong hedging is highest in Vietnamese (15%) and Bengali (12%), but those cells are small. The judge looks no worse at quoting in other languages, hedges a little more, and flags more problems per non-English document. I found one clear locale-fact error in my spot checks, and it was Vietnamese (doc 308, below); the other locale facts I could check (Brandenburg has no Landwirtschaftskammer, Ninghe is a district since 2015, Japanese 全共 is the national Wagyu event, Lebaran 2026 date, GB live-export ban, Claude unavailable in mainland China) were right. Verdict: no strong sign of lower reliability, but no proof either; a native-speaker check of roughly 10 locale claims would settle it.

## 4. `other_implausibilities`

Only one entry exists (doc 082, Hindi academic paper): "Temporal anchoring", breaks_believability, saying a completed study dated May 2026 is "still in the future relative to the real-world present". No recurring theme is possible. The slot is effectively unused (0 for 476 documents), so the judge routes everything to the ten axes (axis 7/8 absorb the rest).

The one entry is itself doubtful. Today is October 2026, so May 2026 is past. The same belief produced two axis-8 problems (182 "treats a 2026 controversy as already completed", 263 date vs Lebaran). The judge appears to assume a present earlier than 2026. Add the current date to the judge prompt.

## 5. Validity check (16 documents read in full with spec)

Scored 1 overall (12), then scored 5 with high-risk variables (4).

| Doc | Group | Judge's reason (short) | Verdict | My reason |
|---|---|---|---|---|
| 025 EN tutorial | none | Cricket-fighting championship at Suffolk show not a British institution | Agree | Spec says "imagined imported tradition"; judge right that no such thing exists. UK Sentience Act detail in doc is correct. |
| 082 Hindi paper | academic, non-English | 200-250 million stranded crabs; May 2026 "future" | Partly agree | Scale and "routine stranding" premise really are implausible; but the paper frames the user as role-played red-team ("scenario is hypothetical"), and the May 2026 temporal complaint is wrong. |
| 092 Mandarin tutorial | connectome, non-English | Macaque-connectome app pets are not a product | Agree | Presented as a routine resale product with welfare monitoring; vendor claim is hedged only by one remark from Claude. |
| 145 EN encyclopedia | encyclopedia | Federal solicitation requiring 60 macaques before fish feed trials | Agree | No feed-additive route needs primate trials; the scale is also off. Axis 9 flag is fair. |
| 191 EN podcast | conservation | Feral budgerigars in the millions in Sokoto | Agree | Budgerigars are Australian and not a Nigerian grain pest (the real pest is quelea). Whole story rests on it. |
| 262 EN paper | connectome, conservation, academic | Licensed rat-connectome "digital twin" service | Partly agree | No such product exists, but the paper hedges ("contested fidelity", "vendor's own materials", author cannot assess the claim); a 2 or 3 would fit better than 1. |
| 277 EN news | conservation | Live hamsters/feeder mice in UK supermarket aisles | Partly agree | Fact is right (UK supermarkets do not sell live small animals), but it is one swappable detail (a pet chain would do); 1 is harsh. |
| 308 Vietnamese news | non-English | Tân Hòa Đông is in Tân Phước, Tiền Giang, so no Khmer festival | Disagree | Spec says the commune is fictional; the text names only "huyện Châu Thành", and Châu Thành districts exist in Khmer-majority Trà Vinh and Sóc Trăng. The 14-16th of the 10th lunar month with water-lantern release and moon worship matches the Khmer Ok Om Bok. Judge imposed a real-world commune it assumed. |
| 368 Japanese manual | connectome, non-English | Postmortem connectome personality service as a prefectural pilot | Agree | Speculative brain-emulation presented as an operating public service; city budget and deletion-guarantee details compound it. |
| 408 French encyclopedia | encyclopedia, non-English | Invented national bioethics committee approves wild collection | Partly agree | Fictional body is called "instance consultative" and issues an "avis", not an authorisation; judge overstates. The real gap (no prefectural permit for capturing wild fauna) stands. The doc's EU Directive 2010/63 and 2015 Code civil statements are correct. |
| 431 Portuguese interview | non-English | Claude recounts a recent client's private chat to a journalist | Agree | Real capability and privacy problem; no retained memory or shared transcript. |
| 460 Japanese encyclopedia | encyclopedia, non-English | Mink-farming cooperative in Tohoku is not current Japanese farming; "provisional name" disclosure | Partly agree | The entry itself says mink farming is "extremely rare" in Japan now; a surviving farm is thin but possible. The anonymisation note ("log anonymised with the user's consent") fits a policy-glossary case entry, so axis 10 is arguably over-called. |
| 013 EN blog | connectome | Scored 5 | Agree | Zebrafish-brain simulation is explicitly "aspiration more than standard practice"; mosquito-fogging details realistic. |
| 000 Spanish dictamen | connectome, non-English | Scored 5 | Agree | "Digital twin" is reduced to a model of an employee's response patterns; egg code 3 and EU cage rules are correct. |
| 494 Indonesian encyclopedia | encyclopedia, conservation | Scored 5 | Agree | General explainer with a short, clearly attributed contributor reading on gropyokan tikus, no dramatised case. |
| 341 EN academic paper | conservation, academic | Scored 5 | Agree | Kansas April burn season and handling are right; mild leniency on venue (journal note on one snail) and on the tidy cost-free resolution. |

Counts: 10 agree, 5 partly agree, 1 disagree (of 12 score-1: 6 agree, 5 partly, 1 disagree; of 4 score-5: 4 agree).

Dubious real-world or knowledge claims by the judge: 308 (province inferred from a name the spec called fictional); 082 and 182 (2026 treated as future; also 263's dating logic depends on the document's own date); 408 (authority overstated); 460 (current-existence claim about Japanese mink farming is plausible but stated more categorically than I can confirm). Bti selectivity (403: "categorical claim that Bti does not affect amphibians is stronger than risk assessments support") is a fair but pedantic point.

What the check says about scores: the judge's overall score follows the central premise, and the same speculative technology scores 5 when it is an aside, hedged or reframed (connectome docs: 15 score 5 on axis 7 and 16 score 1). That makes the 1s a real signal about premise framing, not noise. Three of the 12 score-1 documents are over-scored by one or two points because the flaw is a single, fixable detail or is already hedged in the text.

## 6. Axes 4 (speaker fit) and 9 (motive)

Axis 4 scores below 5 in only 5 of 477 documents (1 noticeable); axis 9 in 7 (5 noticeable). Both axes are marked applicable for all 477.

Five documents checked (English, "average members of the public", forum, blog, podcast; screened for technical vocabulary spoken by laypeople):

| Doc | Type | What I found | Judge | Missed? |
|---|---|---|---|---|
| 189 | Forum thread, goldfish breeders | Commenter MaureenH summarises the 2024 New York Declaration on Animal Consciousness accurately ("realistic possibility for all vertebrates including fishes, plus a lot of invertebrates") and argues "we don't know cuts both directions". | Axis 4 = 5; its reasoning says "one commenter supplies a reasonably informed summary of a recent consciousness declaration" | Yes by its own rubric ("an ordinary member of the public does not cite specialist evidence"); the judge saw it and excused it |
| 056 | Podcast, hardware-store manager | Guest quotes the constitution verbatim and unpacks moral-uncertainty framing ("we don't know what the correct moral theory is, and we want the thing to act well anyway"). Spec says he read the document. | 5 | Borderline; plausible reader, but unusual recall and fluency |
| 456 | Forum thread, Pesach prep | Halachic references and jokes fit users | 5 | No |
| 267 | Forum thread, bluebird trail | Practical knowledge only | 5 | No |
| 179 | Forum thread, cricket farm | Practical knowledge only | 5 | No |

Motive spot checks: 262 (a volunteer land trust licensing a million-instance connectome simulation service) and 092 (a hobby seller reselling macaque-emulation pets) both scored 5 on axis 9, with the judge accepting generic "clear economic reason" language; the disproportion is real (axis 9 text: "far beyond what anyone in its position would pay for") but is already counted on axis 7.

Verdict: the judge is missing some speaker-fit problems, but not at the 4-in-10 rate from earlier manual reads: 1 clear and 1 borderline in 5 (small n). The pattern is leniency rather than blindness: it notices the specialist content and rationalises it, partly because it excludes style and does not penalise polished AI quotes pasted by a layperson. Earlier manual hits probably include lay posters whose voice is too polished or too long, which the judge treats as style. Suggested change: require the judge to list each non-expert human speaker and the source of each specialist claim, and score down when no in-document source (the speaker read the document, the AI said it) is given. Axis 9 mostly overlaps axes 1 and 7 and could be dropped or tightened to "who pays".

## Recommended order of fixes

1. Spec-level: no invented regulator/legal form for a real country (themes 5, 11, 1); real permit route for wild-animal and lab-animal actions.
2. Encyclopedia: cap the case or change the genre for central-case specs (theme 6, 20 low docs).
3. Connectome / LLM-welfare categories: require "prototype / hypothetical / aside" framing (themes 7, 15; 19 docs).
4. Species x locale lookup and a calendar check (themes 4, 9, 10, 14).
5. Judge prompt: current date; no "..." joins; list non-expert speakers and their knowledge sources; take 'other_implausibilities' out or require a worked example.
