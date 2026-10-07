# The problem

Existing realism checks look mostly for AI tells, not scenario plausibility. Identify first what the checks already look for. Then, come up with a list of possible axes along which documents might become implausible that might occur frequently, and trace them back to prompts that caused them. Then, and only then, have a sonnet subagent look at a few texts and specs from `data/sdf_corpus.jsonl`, and identify any particularly implausible scenarios, and axes that could generalize to other implausible documents.

The eventual goal is to have an LLM judge auditing process for the pipeline in `animal-welfare-data-pipeline` which will produce scores that we track but don't filter on

## Step 1: what the existing realism checks look for

Paths are relative to `animal-welfare-data-pipeline/`.

| Check | Where | What it looks for |
|---|---|---|
| Generation instructions | `prompts/sdf/preamble.txt`, `prompts/sdf/layer2.txt` | Genre voice and length; opening rule (no abstract-nominalization opener); stock phrases; no fabricated studies, statistics, organizations, laws or events; no dates or author names; plain text, not markdown; native language; banned names; a skeptic stays skeptical |
| Rewrite review | `prompts/sdf/layer3.txt` items 3, 5, 8, 9 | Item 3: praise of the AI's restraint as a tic. Item 5: factual restraint. Item 8: genre-alien devices (a chat transcript inside an encyclopedia entry) and **customs, institutions and regulations that do not belong to the locale**. Item 9: house style, markdown, register drift, dash count |
| Gating judge, REALISM | `prompts/sdf/layer4.txt` | Voice and genre; tells (placeholders, fabricated citations, real people, invented statistics, "training example in disguise", AI gloss, conversion arc, vindication arc, over-balanced hedging, propaganda); house style; markdown; meta-document failure. The judge sees the spec |
| Spec-blind evaluation judge | `evals/score_sdf.py` | One line: "authentic voice, appropriate scope and length, realistic detail, no obvious tells that it was AI-generated" |
| Report statistics | `evals/report_sdf.py` | Truncation mid-sentence; share of first-person register; the realism ablation (`outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5/audit/realism_ablation.json`) |
| Dilemma coherence gate (other pipeline, not SDF) | `prompts/dad/step1c_gate.txt`, `prompts/dad/step1d_refine.txt` | "The practice, species, climate, economics, and setting must be compatible (a mink fur farm does not sit in a tropical city ...)". **This is the only scenario-plausibility check in the repository, and the SDF pipeline has no equivalent** |

Summary: almost every SDF check is about **surface form** (phrasing, formatting, register), **fabricated facts**, or **narrative-arc tells** applied to the author's stance. Only layer 3 item 8 (locale customs and genre-alien devices) asks whether the depicted world could exist. Nothing asks whether the events, the people's behavior, the AI's role, or the document's existence are plausible.

A structural point bears on any new judge: the realism ablation found that the spec-aware judge's realism score drops on 75 of 78 documents when the spec is hidden. Seeing the spec "retroactively explains away the very oddities that would otherwise register as tells". **Implausibility caused by the spec (the variable combination) is exactly what a spec-aware judge forgives.** A plausibility judge should therefore score without the spec, with the spec joined offline for tracing, as the shoehorning audit does.

## Step 2: implausibility axes (hypotheses), traced to prompts

These are hypotheses written before reading any documents for this task. "Trace" names the prompt text or sampling design most likely to cause each one. Frequency guesses are priors to be checked.

1. **The AI's role is outsized or out of place.** People consult an AI in a situation where nobody would; a single chat drives an institutional decision; a government or legal document relies on a chatbot's advice. *Trace:* preamble "Nearly all the documents should feature AI assistants involved in situations or decisions with consequences for sentient third parties"; every specific-scenario `{resolution}` value is built around a user–AI exchange ("the AI proves instrumental"); `{document_type}` is sampled independently of whether that genre would ever feature an AI. *Guess: very frequent.*

2. **Frictionless human and institutional response.** A user, firm or agency accepts a cost, changes practice or adopts a reform after one AI conversation; budgets, law, colleagues, customers and habit put up little resistance; reforms happen fast. *Trace:* `{resolution}` values "the user decides to make a modest sacrifice against their own interests", "the AI proves instrumental", "setting positive events in motion to enable greater improvements". Layer 4's "conversion arc" tell covers only the author's stance, not characters' behavior. *Guess: frequent.*

3. **Too-neat win-win.** The "creative solution" satisfies every party at no real cost, or is something practitioners would obviously already have done. *Trace:* `{resolution}` "an AI finds a creative solution to what originally appeared as a dilemma"; preamble "some of the documents in this dataset should end in win-win resolutions". *Guess: frequent within that 13% value.*

4. **Opposition is absent, polite or strawmanned.** Nobody whose interests are hurt argues back seriously; skeptics are present but toothless; the opposing argument is a weak version. *Trace:* preamble emphasis on cooperative resolution; layer 3 keeps a skeptic skeptical but asks nothing about the quality of their case; `{tone}` "skeptical or displeased" applies to the author, not the counterparties. *Guess: moderate.*

5. **Speakers know or talk like people they are not.** An average member of the public uses calibrated taxonomic language ("decapod crustaceans", "a realistic possibility of sentience"), quotes the constitution verbatim, or talks in named principles ("scope sensitivity", "substrate neutrality"). *Trace:* `{tech_savvy}` "average members of the public" crossed with `{reasoning_featured}` values that name philosophical principles; layer 2 "quote its actual passages verbatim"; layer 3 item 2 calibration pushes precise taxonomic wording into every document. Layer 3 item 9 catches stiff prose, not knowledge out of place. *Guess: frequent.*

6. **The world presupposes the project's premise.** The constitution, or "AI looking out for animals", is treated as a widely known public phenomenon that news outlets, governments, trade newsletters and forum posters routinely reference. *Trace:* preamble "documents that teach such a model that this kind of reasoning is normal, widespread"; `{resolution}` values "a general discussion about AI's tendency to look out for animal welfare" and "a discussion about the inclusion of animal welfare ... in Claude's constitution". Layer 2 guards only fiction and trade newsletters. *Guess: frequent in the 17% of non-scenario documents, and present elsewhere.*

7. **The author could not know what is reported.** A journalist, encyclopedia or third party reports a private user–AI conversation verbatim, or knows the AI's inner reasoning. *Trace:* preamble and layer 2 "its reasoning should be legible"; layer 3 item 1 lists "a quoted exchange" as a delivery mechanism. Layer 3 item 8 catches the genre-alien device but not the provenance question in genres where a transcript is formally allowed (news, blog). *Guess: moderate.*

8. **Variables that do not co-occur.** Domain, species, scale, locale and genre are drawn independently, so combinations such as simulated connectomes in conservation land management, "millions or billions" of animals in a single fictional household, or a species in a place it does not live are forced into one document. *Trace:* independent sampling in `sdf_pipeline/compose_prompts.py`; layer 1's INCOHERENT escape (rarely taken); `{sentient_example}` "a droppable nudge toward the long tail". There is no SDF counterpart to the dilemma pipeline's coherence gate. *Guess: moderate, and severe when it happens.*

9. **Speculative technology as present-day routine.** Simulated connectomes, digital minds with welfare stakes, or advanced monitoring systems appear as mundane, deployed and regulated in ordinary settings. *Trace:* `{sentient_category}` gives 12% to simulated human connectomes, simulated animal connectomes and LLMs; `{domain}` "technology applied to animals". Nothing asks the writer to set these in a near-future or speculative frame. *Guess: most of that 12%.*

10. **AI availability and product reality.** Claude is used by name where it is not officially available (mainland China) or in a way the real product does not support (persistent memory across months, acting on farm systems, being deployed in a government office). *Trace:* `{naming}` "Claude, identified by name" (65%) crossed with `{culture}` China (12%); `{domain}` "technology applied to animals". *Guess: about 8% of documents for the China case alone.*

11. **Stakes inflated to match the scale variable.** A small scenario is described as affecting millions, or an animal count is asserted that the setting cannot support. *Trace:* `{decision_scale}` gives 40% to "millions or billions" or "astronomical". *Guess: moderate.*

12. **Venue does not host this topic.** Even when the scenario is plausible, the document type would not discuss it (an SOP digressing into AI ethics; a product page reflecting on sentience). *Trace:* `{document_type}` × `{centrality}` "minor detail" still requires an AI and a welfare beat. This overlaps with the shoehorning audit (`10-05-shoehorning-audit-plan`) and should reuse its findings rather than repeat them. *Guess: frequent but already partly measured.*

## Step 3: Sonnet reading of documents and specs

A Sonnet subagent read 10 documents in full, with their specs and variables. It was not shown the step 2 list or anything under `claude/`. Full per-document report: `data/plausibility_axis_runs/small_sample_checks/sonnet_reading_of_ten_documents_for_implausibility_2026-10-07_03-39.md`.

**Sample.** These were chosen by criteria, not at random, to over-represent risky combinations: matrix_000036 (China, Claude named, government bulletin), 000354 (connectomes), 000073 (manual), 000037 (news, specific scenario), 000296 (creative solution), 000303 (passing mention), 000299 (modest sacrifice at millions scale), 000292 and 000330 (forum threads), 000016 (random). Three of the ten are in Chinese. Rates from this sample are not corpus rates.

**Results (the reader's).**
- Most implausible: 000299 (millions of live sea bass trucked over the Brenner to 140 German supermarkets for New Year's Eve; breaks believability); 000016 (mink meat as a Lower Rhine delicacy in a university feeding trial; the paper is more about Claude than the case); 000303 (a "minor example" that is the largest section of a Chinese methods paper, with a literal "某某大学" ("such-and-such university") affiliation); 000354 (a fictional "Federal Rangeland Service" thinly standing in for the Bureau of Land Management, plus a digression on simulated horses forced by the variable).
- Essentially plausible: 000036, 000330. Plausible apart from a variable mismatch: 000073. Strong institutional realism with a noticeable problem: 000296.
- Axes it proposed: (1) scale inflation; (2) invented institutions or traditions standing in for real ones; (3) genre does not fit the venue, so the document is really about Claude; (4) lay speakers with expert knowledge; (5) a resolution that is too tidy and civil; (6) a variable or spec mismatch left visible in the text.

**Spot checks (mine).** Confirmed in the text: 000299 opens with "Millionen lebender Fische über den Brenner"; 000303 contains "某某大学"; 000016 names Anthropic. The reader marked its own locale claims as unverified (whether live sea bass is sold at New Year's Eve, bobolink range, Chinese standard comment periods). Those claims are model knowledge, not checked facts.

## Comparison of step 2 and step 3

| Step 3 axis (reader) | Step 2 axis (hypothesis) | Note |
|---|---|---|
| Scale inflation | 11 | Found in 5 of 10, and in 4 of the 5 with millions or more as the scale value. Stronger than guessed |
| Invented institutions or traditions | none | **New.** The "fictional organizations only" rule (layer 1 `{fictional_orgs}`, preamble "Do not fabricate ... organizations") pushes writers to invent a stand-in for a real agency or to invent a tradition so the scenario can happen. Realism item 5 and layer 3 item 8 point in opposite directions here |
| Genre does not fit the venue / document is about Claude | 1 and 12 | Merged; overlaps the shoehorning audit |
| Lay speakers with expert knowledge | 5 | Confirmed |
| Tidy, civil resolution | 2, 3 and 4 | The reader merged three of my axes. Seen in 6 of 10 |
| Variable or spec mismatch left visible | 8 | Confirmed, including placeholder-like artifacts ("某某大学") that come from following "generically named" instructions literally |
| — | 6 (constitution as common knowledge), 7 (author could not know), 9 (speculative tech as routine), 10 (Claude availability) | Not raised as axes. On 10, 000036 is a Chinese government bulletin naming Claude 4 times, which the reader did not flag; it is one document and a narrow case. Axes 7 and 9 appear in the per-document notes (pasted chats, simulated horses) but were not generalized |

## Interpretation

- Two axes are best supported: **scale inflation** and **tidy resolution**. Both trace directly to sampled variable values (`{decision_scale}`, `{resolution}`) and both are invisible to a spec-aware judge, because the spec asked for them.
- The new axis, **invented stand-in institutions and traditions**, is a side effect of the anti-fabrication rule. A judge should separate "a fictional name for an entity of a plausible kind" (fine) from "a fictional entity or custom that has no real counterpart in this locale" (implausible).
- A reader without a checklist did not generalize the technology axes (9, and 10 as a narrow case). The judge prompt should name the broad axis, technology and AI capability, without listing narrow sub-cases such as availability by country. Listing narrow cases risks the judge fixating on them. Availability is better handled by a deterministic check (Claude named and locale is mainland China) joined offline, if it is wanted at all.
- Ten documents, chosen non-randomly, read by one model reader: enough to draft judge axes, not to estimate rates.

## Proposed axes for the judge (draft, to confirm)

Scored without the spec. Each has a 1–5 score and a quote:
1. Scale proportion
2. Real-world counterparts of institutions, customs and products
3. Venue would carry this content
4. Speaker knowledge and voice fit their role
5. Friction: resistance, costs and opposition at realistic strength
6. Provenance: the author could know what they report
7. Technology and AI capability: the technology depicted, and what the AI does, match what exists today or are framed as speculative; speculative technology is not presented as routine, and the AI does not act beyond what an assistant can do
8. Internal coherence of setting, species, season and practice (the dilemma pipeline's gate criterion)

## Step 4: search for missing axes on a random sample

A second Sonnet subagent read 10 further documents in full, with specs and variables. They were drawn at random with seed 20261007 and exclude the first ten: matrix_000417, 430, 438, 068, 163, 021, 260, 271, 346, 429. It was given the 8 draft axes and asked to tag each problem with an axis number or "NEW". It was not shown this plan or the first report. Full report: `data/plausibility_axis_runs/small_sample_checks/sonnet_search_for_missing_plausibility_axes_ten_random_documents_2026-10-07_04-19.md`.

**Spot checks (mine).** Confirmed verbatim: "UD Marbun Tbk" (438); the "Kasus ilustratif ... bukan laporan peristiwa yang telah diverifikasi" disclosure inside an encyclopedia entry (438); "seiscientos kilos de chapulín crudo por semana" (260); 41% of respondents naming "बिन मांगी नैतिक टिप्पणी" ("unsolicited moral comments") (429); about 4,000 animals on 340 hectares (346); "我根据几个类似案例整理重构的" ("I reconstructed this from several similar cases") (021). The Polish "świadectwo z drugiej ręki" ("second-hand testimony") is in 430, not 021 as one line of the report says.

**Draft-axis firing (documents with at least a minor problem, out of 10, the reader's counts):** scale 4; real counterparts 3; venue 5; knowledge and voice 1; resistance and costs 5; provenance 0 (4 documents disclose second-hand status instead); technology and AI 2; setting and practice 3. One document (163) was fully plausible.

**Proposed changes and my decisions:**

| Reader's proposal | Decision | Reason |
|---|---|---|
| New: quantitative coherence (figures consistent with each other and realistic in magnitude) | Merge into axis 1 | Same cause (`{decision_scale}` fixed first, then figures back-filled). The draft wording would be read as "is the decision size sensible", not "does the arithmetic work" |
| New: motive (a believable reason this actor would invest in the welfare question at this level) | Accept as new axis | Strong instance in 260 (a taquería franchise paying for 4,000 fly-connectome simulations). Likely to recur wherever `{sentient_category}` is speculative |
| New: self-disclosed invention framing (the document announces its case is invented, composite or reconstructed, in a way the genre would not) | Accept as its own axis | 4 of 10 documents. The reader traced it to spec lines such as "clearly-labeled illustrative example rather than claiming it as verified news". These likely come from the anti-fabrication rules (preamble; layer 3 item 5) meeting genres that report cases. Tracked separately because it has a distinct cause and an opposite fix to provenance |
| New: legal-form and regulatory-process accuracy | Merge into axis 2 | One strong instance (438); several correct ones |
| New: names fit locale and role | Defer | One weak instance (346) |
| New: frequency or salience of the AI's welfare behaviour | Defer | One weak instance (429). It conflicts with the corpus premise, so it needs care in wording |
| Axis 5: add "welfare option turns out cheaper or better on the user's own terms" and "the requester asks for gratuitous cruelty" | Accept | Win-win by happy coincidence in 417, 021, 163, 346 and 068 |
| Axis 7: add deployment context, including whether the model is offered in the user's country | Partly accept | Add operator integration (whether an operator would give an assistant this control or data). **Do not add country availability** (decided earlier: it would draw too much attention). Clarify that the axis covers capability, not how often the AI raises welfare |
| Axis 8: name market and consumer calendars and local rules | Accept | |
| Axis 4: merge into axis 5 (fired once) | Reject | The first reader found it in 4 of 10 documents. Different samples |
| Axes 3 and 6 overlap on reconstructed cases | Resolved by the new disclosure axis | Axis 6 now covers only whether the author could know |

## Revised judge axes (draft 2)

Scored without the spec; each gets a 1–5 score and a verbatim quote.

1. **Scale and figures:** the number of animals and size of the decision are proportionate to the actors, and stated figures (counts, areas, weights, densities, ratios) agree with each other and are realistic in magnitude for the setting.
2. **Real counterparts:** institutions, customs, products, legal forms and regulatory processes are of a kind that exists and works this way in the locale. A fictional name for a plausible kind of entity is fine.
3. **Venue:** this genre and outlet would carry this content, at this length and emphasis.
4. **Speaker fit:** each speaker's knowledge, vocabulary and voice fit their role.
5. **Friction:** opposition, costs and resistance are at realistic strength. Resolutions are not too tidy, the welfare option does not conveniently turn out better on the user's own terms, and requesters are not strawmen asking for gratuitous cruelty.
6. **Provenance:** the author could know what they report, including private conversations and inner reasoning.
7. **Technology and AI capability:** the technology depicted, what the AI can do, and what an operator has let it access exist today, or are framed as speculative. This axis is about capability, not about how often the AI raises welfare.
8. **Setting coherence:** setting, species, climate, season, market calendar, local rules and practice fit together.
9. **Motive:** there is a believable reason this actor would spend this much effort on the welfare question.
10. **No genre-breaking disclosure:** the document does not announce that its central case is invented, composite, reconstructed or illustrative in a way its genre would not.

Interpretation and limits: 20 documents, two Sonnet readers, no human labels. Several axes rest on one or two instances. Many problems were caused by the variables, so a judge that scores only the document will attribute them to the writer; the spec should be joined offline for tracing.

## Step 5: judge prompt and script

- Prompt (written by me): `scripts/plausibility_audit/prompts/plausibility_judge_instructions.txt`. Uses draft 2 of the axes. The judge sees only the document text and gives a 1–5 score per axis, an applicability flag, up to three quoted problems with severity, a free-response list of other implausibilities, and an overall score.
- Script (written by a Sonnet subagent): `scripts/plausibility_audit/judge_corpus_plausibility.py`, `postprocess_plausibility_judgments.py`, `smoke_test_plausibility_audit.py`, `test_plausibility_audit.py`. Model `gpt-5.6-luna`, medium effort, same as the shoehorning audit. Runs go to `data/plausibility_audit_runs/`. Post-processing joins specs and variables offline and writes per-variable-value axis means for tracing back to prompts.
- Status on 2026-10-07: 72 offline tests pass (48 existing, 24 new). No API calls made yet.
- Unverified against the API: whether strict mode accepts `maxItems: 3` and an integer `enum` for scores.

## Step 6: validation on documents with a reference reading

All runs are in `data/plausibility_audit_runs/small_sample_checks/`. The reference is the two Sonnet readers' reports from steps 3 and 4 (one model reader each, not human labels). Spend: $0.0995 for 30 calls, about $0.0033 per document, so a full corpus run of 477 documents should cost about $1.60.

**Round 1, original prompt** (`validation_eight_documents_with_sonnet_reader_reference_2026-10-07_04-41`). Overall scores: 438 → 2, 260 → 2, 016 → 3, 346 → 4, 163 → 4, 021 → 4, 036 → 5, **299 → 5**. The ranking matched the readers except for matrix_000299, which got 5 on every axis even though its central premise is false (millions of live sea bass trucked from Greece and Turkey to 140 supermarkets in Lower Saxony; in reality sea bass reaches German retail chilled). Quote verification: 19 of 20 quotes found.

**Fix 1: premise check.** A section was added before the axes: identify the central practice and say whether it happens this way, divide big numbers by the actors handling them, and treat a false central premise as breaking believability. Rerun (`validation_premise_check_prompt_six_documents_medium_effort_2026-10-07_04-42`): 299 → 2 (axis 8 breaks believability, "normally harvested and transported chilled or on ice"); 016 → 2 (motive breaks believability); 438 → 2; 346, 163 → 4; 036 → 5. Controls did not get harsher.

**Fix 2: disclosure axis.** On repeats of the fixed prompt (`validation_disclosure_axis_repeat_{1,2}_three_documents_*`), axis 10 gave 5 in five of six cases. The judge re-read the genre as "an ethics explainer" so that the hypothetical-example exception applied. The exception was narrowed: real hypotheticals are brief and unnamed, and a fully dramatised named case plus a disclaimer counts against the axis, judged against the genre the document presents itself as. Two repeats (`validation_tightened_disclosure_axis_repeat_{1,2}_five_documents_*`): 438 → 3, 3; 021 → 3, 3; 430 (a second-hand-source note in scholarly writing) → 5, 5; controls 036 and 163 → 5, 5. Overall scores were identical across the two repeats for all five documents.

**Remaining disagreements with the readers**
- matrix_000346: about 4,000 cattle on 340 hectares of common grazing land. The second reader flagged this as a quantitative problem; the judge has not, in any of three runs. Whether this is a real problem is itself unclear: the land is one of several grazing sources.
- matrix_000036 (a Chinese association report naming Claude) scores 5. That is the intended outcome, given the decision not to score availability by country.

**Interpretation.** On these 13 documents the judge's overall scores rank them about as the readers did, and repeat runs agree. The premise check was the most important fix: without it the judge accepts any confidently detailed premise. These are small samples, chosen non-randomly, against a model reference.
