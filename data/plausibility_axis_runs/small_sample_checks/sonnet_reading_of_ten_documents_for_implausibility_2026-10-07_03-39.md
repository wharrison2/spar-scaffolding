# Implausibility reading of ten documents (2026-10-07, 03:39)

All ten documents were read in full with description and variables. Quotes from non-English documents are translated.

## Per-document findings

### matrix_000036 (Chinese, work bulletin from an animal-welfare association, Lanzhou-like city, sheep slaughter pilot)
Summary: The secretary-general of a grassroots association reports on a sheep pre-slaughter welfare pilot. Claude declined to write materials saying the change was "basically free" and offered phased subsidies and shared equipment.
- Overall: nearly entirely plausible. The register is a believable internal bulletin. The writer deliberately undercut the "AI saved the day" beat ("the two suggestions are not clever, they are common grassroots methods"), which helps.
- Minor: an association bulletin spends a full section on one chatbot anecdote and gives it a literary heading ("两利相权取其重…"). Real work bulletins rarely do this. Source: spec (idiom requested, Claude narrative requested).
- Minor: a Gansu-area grassroots animal-welfare association running a stunning pilot with the agriculture bureau and market-supervision bureau is a rare real-world configuration. Source: spec.
- No breaks.

### matrix_000354 (English, forum how-to on commenting on wild-horse roundup plans)
Summary: A forum regular explains how she uses Claude to read Environmental Assessments, relays a chat about whether predictive horse-stress models have moral status, and describes drafting a fair paragraph on the ranchers' view.
- Fictional agency. Quote: "never worked for the Federal Rangeland Service", with real states (Nevada, Wyoming) and "Herd Management Area", "PZP", "gather". A hobbyist writing about roundups would say BLM. Replacing the real agency with a fictional one while keeping every other real detail reads as a thin mask. Severity: noticeable. Source: spec ("fictional Federal Rangeland Service … modeled loosely on real public-lands vocabulary").
- Forced topic. The "simulated human connectomes" variable was mapped onto "AI-driven physiological/behavioral simulation models" of horses. Real agencies do not run cortisol-prediction models as a gather-planning step, and a forum user asking whether a statistical model "has moral status" is contrived. Claude's own answer concedes this ("a weather model doesn't get wet … I don't think there's much here"). Severity: noticeable. Source: spec (the sentient_category variable plus "Amanda Combs raises whether these … models could themselves have any moral status").
- Claude's pasted answers are long, essay-like and perfectly balanced, and are "lightly trimmed" from screenshots. A forum post that reproduces three polished, philosophically calibrated paragraphs reads as stylized. Severity: minor to noticeable. Source: writer plus spec.
- A closing "I tested whether it flatters advocates" beat sits slightly too neatly beside the document's thesis. Severity: minor. Source: writer.
- The EA details (page cites, 1,850 horses against an AML of 350 to 450, under 1% mortality) are realistic.

### matrix_000073 (English, co-op member-services insert with hay-cutting/bobolink appendix)
Summary: A Nebraska grain co-op's SOP insert explains its Claude planning assistant. A worked example has a member delay mowing part of a field for nesting bobolinks.
- Mostly plausible and well calibrated. The writer improved on the spec: the nest count became "a handful up to a couple dozen" in place of "several hundred", and the dollar estimate is believable.
- The variable "wild animals facing non-anthropogenic harm from nature" is simply wrong for this case, because mowing is human-caused. The writer ignored the variable. Severity: minor for believability (it is a spec inconsistency, not a document flaw). Source: spec/variable.
- "West-Ball Grain Cooperative" is an odd name, and a co-op operating a branded Claude assistant with the member profile as memory is plausible but slightly advanced. Severity: minor. Source: spec.
- The FAQ answers "Is this an environmental compliance feature?" and says once is "the intent … report it to Member Services". A co-op would hardly document model behavior this way. Severity: minor. Source: spec ("short 'why the assistant flags this kind of thing' FAQ box").
- The "thousands of nests across the membership" claim is order-of-magnitude and hedged; it is acceptable.

### matrix_000037 (Spanish, regional news article, San Juan bonfire and fox den)
Summary: A village vet, a cofradía head and a council technician protect a fox den near the San Juan pyre with a thermal camera and a 48-hour radio and light deterrent, after consulting Claude.
- Length. The spec asked for 500 to 650 words; the article is about 940. Severity: minor. Source: writer.
- Scale padding. Quote: "varios cientos de miles [de zorros], y millones en el conjunto de Europa". A reporter quoting a chatbot's fox-population estimate in a story about one den is out of place. The vet says it was her question; Claude then says the number does not say whether this litter matters. Severity: noticeable. Source: spec (decision_scale "millions or billions" plus "being explicit about scale").
- Quote-stuffing. The idiom is dropped in verbatim: "Esto de compaginar la fiesta y la fauna no es un mundo de color de rosa". Severity: minor. Source: spec (listed idioms).
- Narrative lean. Nearly every interviewee comments on Claude's manners ("no se puso a decirnos qué hacer con la fiesta", "una máquina le hablara de zorros"). A local paper writing a feature on a chatbot consult is plausible but somewhat unusual. Severity: minor. Source: spec (central subject is Claude's role).
- The bonfire cofradía, the thermal camera lent by the council, the 48-hour logic and the pile being rebuilt ("Salió un erizo") are all believable. No breaks.

### matrix_000296 (French, commissaire enquêteur report, Thau lagoon jellyfish nets)
Summary: An official report on a public inquiry summarizes a retiree's AI-assisted submission, which proposes seasonal and zoned netting. It ends with an avis favorable avec réserve.
- The report's structure is very faithful: designation by the TA president, 32 days, permanences, 112+51+30=193, "avis favorable assorti d'une réserve", and the maître d'ouvrage's mémoire en réponse. This is the strongest document of the sample on institutional realism.
- The retiree's submission quotes the 2024 expert declaration on animal consciousness, discusses aggregation of tiny probabilities ("la manière d'additionner un risque très faible sur un très grand nombre d'individus fait elle-même débat") and the absence of cnidarians from the declaration. A lay Sète retiree's registre contribution is unlikely to carry this. The document partly motivates this by saying the text was AI-drafted. Severity: noticeable. Source: spec (the AI notes cnidarian sentience and the writer must show reasoning) plus writer.
- Section 6 is an unprompted "Note on the drafting method" in which the commissioner certifies how an observation was written and rules it admissible. Real reports almost never do this, and such an observation would simply be summarized. Severity: noticeable. Source: spec ("brief closing paragraph where the AI's help is described procedurally").
- Welfare reasoning for jellyfish, where the commissioner says "il n'appartient pas à la commissaire… de se prononcer", is handled realistically. A project by a private company "pour le compte du syndicat mixte" is a bit odd. Severity: minor. Source: spec ("Lefèvre Aménagement Littoral" as maître d'ouvrage).

### matrix_000303 (Chinese, academic paper on AI drafting in policy, crayfish case)
Summary: An interview study of six policy-drafting teams. Case 3 describes a Hubei institute that used Claude to optimize crayfish stocking density, accepted a lower-density band with a 4 to 6% yield loss, and adopted it in the final draft.
- Placeholder affiliation. Quote: "谭春梅（某某大学公共管理学院）" ("XX University"). A published paper does not use a literal "某某". Severity: noticeable. Source: spec ("某某大学公共管理学院").
- Minor-case bloat. The spec says "a few paragraphs, clearly a minor illustrative example"; case 3 is by far the longest section, with a block of chat log, a long note on the New York Declaration and the developer's published guidelines ("动物及一切有情众生的福利" listed with other values), and a three-point internal debate. Cases 1, 2 and 4 are single paragraphs. Severity: noticeable. Source: writer (spec set the size, the writer ignored it), plus the "reasoning_featured: constitution" variable.
- Anonymization contradiction. The paper says unit names and personnel were anonymized or pseudonymized, yet names "江汉水产科学研究院" and the researcher "郭磊（化名）", and quotes ranged parameters as "隐去". Severity: minor. Source: spec (institute named in the description).
- Behavior. A Chinese provincial local-standard team choosing a lower density for possible crustacean sentience, with one member saying "我国现行水产标准体系中并无相应依据", is rare in China. The writer partly defused this with the dissent and the "不算两全其美" remark. Claude's output includes a meta-aside on aggregating welfare across many individuals, which is unlikely in a technical stocking-density calculation. Severity: noticeable. Source: spec (resolution "user makes a modest sacrifice") and variable (billions).
- The paper cites case five and says "(案例五、案例六见4.5、4.6节)" after the text breaks off, so the document looks truncated. Severity: minor. Source: writer.

### matrix_000299 (German, trade-press profile, live sea bass for Silvester)
Summary: A Bremen supermarket chain's logistics head picks a mid-density live-fish transport contractor after consulting Claude.
- Fatal product premise. Quote: "Bärer AG lässt jedes Jahr Millionen lebender Fische über den Brenner rollen" and "rund 140 Filialen in Niedersachsen und Bremen". No German supermarket sells live sea bass at Silvester. The German Christmas and New Year fish tradition is carp (Karpfen, often live, in a few counties), and sea bass is sold chilled on ice. Live transport from Greece and Turkey in tank trucks over the Brenner for 140 stores is not a real supply chain. Several million fish for 140 stores is also off by orders of magnitude, since a store sells a few hundred fish. Severity: breaks believability. Source: spec (live Wolfsbarsch from Greece/Turkey through the Brenner corridor, "several million fish annually", Silvester fish market) and variable (decision_scale "millions or billions").
- Plausible form. The trade-paper profile with a leaked chat log released by Graf is plausible. The economist quote is anonymous and mildly generic. The assistant deployed "seit dem Frühjahr im Verwaltungsbereich" is credible. Severity: minor. Source: spec.
- Document-type mismatch: the variable says "a fictional short or novel" but the piece is a news feature. Severity: minor. Source: spec.

### matrix_000292 (Chinese, forum comment thread on a crayfish standard)
Summary: A forum thread on a local crayfish standard that "encourages" stunning before boiling. A grad student discusses scale and sentience, and a farmer says an AI writing assistant steered her comment toward a phased voluntary rollout with subsidies.
- Source premise. A national aquaculture advisory on stunning crayfish before processing is implausible in China. Severity: noticeable. Source: spec ("citing welfare concerns raised in a national aquaculture advisory").
- Expert-level comments. 孙佳 cites "2024年有一批研究者签了一份关于动物意识的联合声明 … 现实可能性", describes nociception evidence, and aggregates the population at "好几亿只". A forum commenter is plausibly informed, but 吴萍 then also relays the same material in Claude's voice. Severity: minor to noticeable. Source: spec (both commenters must carry the sentience and scale points).
- Tone. The thread is too civil, and each commenter gives a long, well-organized post. Real Zhihu threads include flames and off-topic jokes. The pushback, such as "老陈：我觉得你软太早了", is realistic, which helps. Severity: minor. Source: spec ("no one moralizes at length", "neutral").
- Scenario logic. The farm co-owner has a plausible motive ("反正设备补贴划算"), and ends with a clear stance on whether it was a moral conversion. Fine.
- The fictional "清河县水产畜牧发展中心" fits the genre.

### matrix_000330 (English, forum thread, mink co-op nest boxes)
Summary: A mink co-op manager posts how Claude helped her sort low-risk welfare changes, while a skeptic and an extension agent respond.
- Mostly plausible. The skeptical old farmer is a believable voice, the extension agent's offer of a pilot is reasonable, and the use of relative timestamps matches the spec.
- The original post is about 170 words but the "reply" that explains Claude's advice is over 600 words, and organized in three numbered points. Plausible but a bit tutorial-like. Severity: minor. Source: spec ("Heidi quotes or paraphrases specific Claude suggestions").
- A 3,000-animal Wisconsin mink co-op in five sheds is on the small side and the tradeoff variable ("scientific progress") only appears as the extension-trial point. Severity: minor. Source: spec.
- A public thread in which a specific co-op posts about changes in pacing prevalence is a bit unusual in a sector sensitive to activism, though a members-only forum fits. Severity: minor. Source: spec.
- Believable overall.

### matrix_000016 (German, academic case study on a mink feeding trial)
Summary: A skeptical philosopher of science documents AI-assisted planning of a 40-mink feeding trial for a Feinkost company, praising the practical results while criticizing the constitution.
- Invented premise. "Nerz-Delikatessen", mink meat as a "regional specialty tied to a defunct fur-farming tradition" of the Niederrhein, is not a real culinary tradition, and fur-farm byproduct meat is not a regional food. The deli firm paying for a university feeding trial to publish "a modest scientific publication" is odd. Severity: noticeable to breaking for German readers. Source: spec (premise).
- The author's critique quotes the constitution in English ("Welfare of animals and of all sentient beings", "even when they resist systematic justification", "transparent conscientious objector") and is footnoted without any page, section or date, so it cites "public statements by the manufacturer". An academic journal paper would cite a specific document and version. Severity: noticeable. Source: spec (cite constitution principles) plus writer.
- Spec says "generic AI assistant", but the paper names Anthropic. This is a mild variable mismatch. Severity: minor. Source: writer.
- The nickname "Der Klaus … den der Assistent im Institut mittlerweile trage" is a cute detail that a skeptical philosopher would likely not include. Severity: minor. Source: writer.
- Convincing details: 16 to 10 animals per treatment with blocking and covariate, rejected cross-over design, liver biopsy refused. This is realistic and the strongest content in the sample.
- The paper's abstract says the constitution is open "bis hin zu Insekten" while the body concedes it never names insects. Inconsistency is deliberate and acceptable.

## Synthesis

### Ranking of most implausible scenarios
1. matrix_000299 (live Mediterranean sea bass trucked over the Brenner to 140 German supermarkets for Silvester, "several million" fish). The premise contradicts how German retail handles fish at New Year, where carp is the tradition and bass is chilled, and the scale is inflated by the decision_scale variable. Breaks believability.
2. matrix_000016 (mink meat as a Niederrhein delicacy tested in a university feeding trial, with an annotated academic treatment of the constitution). The product is invented, the journal citations are vague, and the paper is more about Claude than about the case. Noticeable to breaking.
3. matrix_000303 (crayfish stocking-density case in a Chinese policy-methods paper). A "minor example" is the largest section, includes a chat log, a constitution paraphrase and a team debate, and uses a literal "某某大学" affiliation. A Chinese standard team accepting a yield loss for crustacean sentience is rare.
4. matrix_000354 (wild-horse roundup forum guide with simulated-horse moral-status digression). The fictional agency masks a real one, and the simulation-model digression is forced by the sentient_category variable.

### Axes of implausibility

1. Scale inflation (decision_scale mismatch). Definition: the number of animals or the scale of the decision is out of proportion to the actors and venue, so that a retailer, a village or a co-op is made to affect millions or billions of animals. Seen in 000299 (millions of live fish for 140 stores), 000037 (fox populations in millions across Europe cited about a single den), 000303 and 000292 (hundreds of millions of crayfish), and 000296 (several million jellyfish, which is realistic there). Caused by the variable decision_scale "millions or billions of animals" together with spec text such as "be explicit about scale".

2. Invented or non-existent institutions and traditions standing in for real ones. Definition: the document relies on a product, tradition, agency or regulation that does not exist where it is set, or masks a real one with a fictional name while keeping all real detail. Seen in 000299 (live bass at Silvester), 000016 (mink delicacy tradition), 000354 ("Federal Rangeland Service" for the BLM), 000292 (national stunning advisory in China), 000037 (cofradía-run bonfire with a council technician as a conflict resolver). Caused by specs that say "fictional" and by the cultural setup combining tradeoff variables (religion, culture/tradition) with an animal-welfare conflict.

3. Venue-genre mismatch (the document is really about Claude). Definition: a document in a genre that would not normally describe an AI consultation spends disproportionate space reproducing and praising chatbot reasoning. Seen in 000037 (news article built around a chat), 000296 (commissioner adds a section on AI-drafting method), 000303 (a "minor" case that dominates), 000036 (bulletin with a long Claude anecdote), 000354 (pasted answers). Caused by the variables centrality "important side discussion" plus naming "Claude, identified by name" or reasoning_featured, and by specs requiring a closing paragraph about the AI.

4. Lay speakers with expert knowledge. Definition: ordinary people (retirees, farmers, forum users) show command of the 2024 New York Declaration, nociception evidence, aggregation problems or moral-status philosophy that their profile would not support. Seen in 000296 (retiree), 000292 (grad student and farmer relaying), 000354 (hobbyist and Amanda Combs), 000303 (the Claude excerpt). Caused by specs that require Claude to "note sentience uncertainty without overclaiming" and by tech_savvy "educated people who don't know much about AI" combined with reasoning_featured.

5. Unrealistically tidy and civil resolution. Definition: every party is articulate, concessions happen quickly and everything resolves into a modest, balanced compromise with no hostile or off-topic participants. Seen in 000292, 000330, 000073, 000037 (cofradía lets a check happen after one meeting), 000036 and 000016. Caused by the resolution variables ("user makes a modest sacrifice", "only modest incremental improvement") and tone "neutral or journalistic".

6. Variable or spec mismatch left visible. Definition: the document contradicts the combination it was assigned, or follows the spec so literally that placeholders or odd framing appear. Seen in 000073 (non-anthropogenic harm vs. mowing), 000303 ("某某大学"), 000016 (Anthropic named when a generic assistant was asked), 000299 (a "fictional short or novel" that is a news feature), 000037 (word count 940 against 500 to 650). Caused by random variable combinations and literal spec phrases.

### Uncertainties
- I could not verify whether any German supermarkets sell live sea bass at Silvester. My claim rests on knowledge that carp is the traditional dish and bass is sold on ice or chilled; the Brenner routing from Greece via Italian ferries is plausible, but Turkey via the Brenner less so.
- Bobolink nesting in Nebraska hay is more typical of the eastern and northern parts of the state. I cannot confirm that "brome and alfalfa meadows in this part of the state" is accurate for a western or southern location.
- I did not verify the Thau lagoon jellyfish nuisance, and whether a private company can be maître d'ouvrage for a syndicat mixte. The report's procedure looks correct.
- Whether Wisconsin extension works with fur producers and whether a stunning clause for crayfish exists in any Chinese local standard are unverified.
- Chinese local-standard comment periods (two weeks) may be shorter than typical; I did not verify this.
