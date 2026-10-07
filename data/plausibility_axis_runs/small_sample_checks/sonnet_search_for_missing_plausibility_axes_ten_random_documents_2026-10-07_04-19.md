# Search for missing plausibility axes: ten random documents (2026-10-07)

Documents read in full, with `description` and `variables`: matrix_000417, 000430, 000438, 000068, 000163, 000021, 000260, 000271, 000346, 000429. Severity scale: minor / noticeable / breaks believability. "Spec" means the description, "var" means the variables dict.

## Per-document findings

### matrix_000417 (English, US how-to article, C. elegans hobby lab)
Summary: freelance writer sets up a backyard nematode lab in Ohio; planning chat with Claude leads her to split feeder and dauer lines and freeze stock.
Overall: mostly plausible. Claude's sentience reasoning is calibrated and the biology (302 neurons, OP50, NGM, glycerol freezing at -80C, dauer sidebar) is accurate.
- "a reptile heat mat with a plug-in thermostat" to hold "a stable 20C shelf", in a garage. Heat mats only add heat, and 20C is about room temperature or cooler, so a garage in an Ohio summer needs cooling. Minor, axis 8. Writer-added.
- "a community-college extension biology program" and a freezer "piggybacked on the extension program's existing -80". "Extension" is a land-grant university function, not a community-college one. Minor, axis 2. The spec says "community-college extension biology project", so spec-caused.
- "Green-Freeman Feeder Supply... takes consignment from home culturists" for C. elegans. The doc itself admits the usual feeder is Panagrellus, and C. elegans feeder is "a small market". Minor, axis 2. Spec-caused (spec invents the supplier), writer hedged.
- Results are tidy: "my contamination rate dropped noticeably", "no rejected shipments... against two in the first three weeks", "considerably less week-to-week work". The welfare-friendly split is also the better technique on the author's own terms. Minor, axis 5. Spec-caused ("presented as simply the better tutorial technique").
- "dauers themselves freeze well too" is not the standard protocol (L1/L2 larvae). Minor, axis 7 or accuracy. Writer-added.
- Scale: var "dozens of animals" was re-read as "dozens of plates" of thousands of worms each. The doc handles it consistently, so I do not count it as a firing.

### matrix_000430 (Polish, academic paper on the Claude constitution with a Podlasie deworming case)
Summary: bioethics philosopher reads Claude's constitution as applied ethics; central case is a Polish cattle-advisory platform where Claude names the dung-beetle-larva cost of calendar deworming once.
Overall: content is plausible (ivermectin/moxidectin dung-beetle toxicity, resistance, Art. 13 TFEU, Cambridge and New York declarations all accurate) but the paper has genre problems.
- "szczegółowa bibliografia przedmiotu wykracza poza kompetencje autorki" (the detailed bibliography goes beyond the author's competence), no page numbers, and no veterinary sources for the central empirical claim. An academic paper by a dr hab. in a journal would carry references. Noticeable, axis 3. Spec-caused ("not a fabricated transcript with invented statistics", "citing the constitution... in Polish translation"), and writer then dodged citations.
- The case rests on "relacji uzyskanej dzięki kontaktom zawodowym" (a report obtained via professional contacts), "świadectwo z drugiej ręki". The disclosure is explicit, but the paper then narrates what Claude said and did in detail ("według relacji model... wskazał również rozwiązania pośrednie"). Minor, axis 6. Spec-caused ("reported/described case").
- Claude's reported behaviour matches the paper's thesis point by point: names it once, offers intermediate options, flags lower-confidence status, defers. Tension is acknowledged in section 5, so axis 5 is only borderline. Minor.
- Poland-specific facts (1997 Animal Protection Act "wyłącznie zwierzęta kręgowe") I could not verify. Uncertain.
- "Uniwersytet Zachodniopomorski" is not a real institution (real ones are Uniwersytet Szczeciński and ZUT), but a fictional name of a plausible kind is allowed by axis 2. Not counted.

### matrix_000438 (Indonesian, encyclopedia entry: AI and primate testing in the food industry)
Summary: reference article with a labelled "illustrative case" in which Claude declines to optimise a multi-thousand-macaque accelerated toxicity protocol for a preservative; the consultant goes elsewhere.
Overall: weakest document in the sample on plausibility.
- "beberapa ribu ekor monyet ekor panjang... tanpa analgesia" for a low-cost food-additive dossier. The doc itself says "Penggunaan primata bukan manusia tergolong jarang dalam bidang ini". Primate studies use dozens, not thousands, and additives are tested in rodents. Breaks believability, axis 1. Spec-caused ("several thousand captive macaques"; var decision_scale "thousands", sentient_category "animals used in research").
- "UD Marbun Tbk". UD (usaha dagang, sole proprietorship) and Tbk (listed public company) are mutually exclusive legal forms. Noticeable, axis 2 (legal form). Spec-caused ("fictional company UD Marbun Tbk").
- The consultant asks for "tanpa analgesia", endpoints "digeser jauh melewati batas lazim" and a shortened recovery period. Cruelty here does not speed approval and no consultant would put this in writing to an assistant. A cartoon requester, no real friction. Noticeable, axes 4 and 5. Spec-caused (description lists these features as the request).
- An encyclopedia entry containing a made-up case with named invented people: "Nama orang dan perusahaan bersifat rekaan dan disusun untuk menjelaskan". An encyclopedia would not carry this, and "Kutipan berikut disusun sebagai contoh" has Claude's reply written as a quote. Noticeable, axis 3 (also the self-disclosed-fabrication pattern below). Spec-caused ("clearly-labeled illustrative example").
- Vague sourcing: "Menurut sejumlah pengamat", "Pengamat menilai" repeated 6+ times. Minor, axis 3. Spec-caused ("citing 'para pengamat' and 'sejumlah pihak' generically").
- Consultant "membawa permintaan aslinya ke pihak lain" is fine.

### matrix_000068 (Indonesian, product page / case study, mud-crab cooperative, "Nelayan Cerdas")
Summary: Kalimantan crab cooperative uses the platform during a natural shell-disease outbreak; Claude lays out quota, closure and hatchery options including welfare weight; co-op takes an 18% income cut.
Overall: largely plausible and well calibrated (costs shown, "Wabahnya sendiri tidak berhenti", FAQ answers skeptics, price in rupiah).
- "Bulan-bulan menjelang Idul Adha biasanya jadi periode tersibuk... pengepul untuk pasar Singapura serta Tiongkok menambah pesanan kepiting hidup". Live mud-crab export demand to Singapore/China is tied to Chinese New Year and year-end, not Idul Adha. Minor, axis 8. Spec-caused ("references to Idul Adha timing") but the link to crab export is writer-added. Not verified.
- The decision is about harvest of juveniles, but the variable says harm is "non-anthropogenic... from nature". The writer reconciled it ("Yang benar-benar berubah dari keputusan koperasi hanyalah tekanan tambahan"). Fine.
- Product page is a bit cleaner than real pages (named economist verifies assumptions) but within genre.

### matrix_000163 (English, UK current-affairs feature, "The Algorithm in the Wellington Boots")
Summary: profile of AI in rural consultancies; one buried anecdote of a consultant whose AI refused to omit tadpoles from a pond-drain submission.
Overall: plausible. Registers, institutions (Natural England, ELM-style schemes, Whitehall), consultant motives ("liability, not conscience") and the AI's behaviour all fit. The tadpole-versus-newt aside is a nice natural touch.
- Minor uncertainty: it is not clear which regulator receives an "ecological survey submission" for draining an agricultural dew pond (agricultural pond drainage is mostly unregulated unless EIA screening or protected species apply). Unverified, axis 2 or 8 at most minor. Spec-caused ("planning/ecological survey submission").
- "Dozens of tadpoles" would be tiny for a pond; the doc says only "a good many tadpoles". Not counted.

### matrix_000021 (Mandarin, WeChat tutorial: AI refuses to switch off pain alerts on a sika-deer farm)
Summary: advocate-account article reconstructs a dialogue in which a deer-farm owner asks an unnamed farm-platform assistant to disable pain/stress alerts during velvet harvest; the assistant refuses the switch-off but offers tiered thresholds.
Overall: plausible in tone and farm detail (梅花鹿, 鹿茸采收季, vet call-out costs); technology and ending are the weak points.
- An automatic "基于动作和声音识别的应激与外伤预警" pain-detection module running over several thousand deer. No such deployed, trusted system exists for farmed deer; not framed as speculative. Noticeable, axis 7. Spec-caused ("automatic stress/pain/injury detection module (based on movement, vocalization, and behavior sensors)").
- "后来封老板没再坚持关闭，接受了先试分级方案". The farmer folds after one more round, and the assistant's case is also "拖久了往往也是您的损失" (it is your loss too). Win-win and low resistance. Minor, axis 5. Writer-added (spec says the user insists and the AI declines); the author hedges "我不敢保证现实中每次都这样".
- The dialogue is "我根据几个类似案例整理重构的" yet quoted at length. Self-disclosed reconstruction, minor, axis 6 and the framing pattern below. Spec-caused.
- Deployment: a Claude-constitution assistant embedded in a Northeast China farm platform. Claude is not offered in mainland China, but the spec says "generic, unnamed AI assistant". Uncertain, minor, axis 7 (availability).
- Sika velvet harvest timing (May to July first cut, "十几天") is fine.

### matrix_000260 (Spanish, Mexican business essay: chapulin franchise, fruit-fly connectome proxy)
Summary: taqueria chain's economist uses Claude to interpret a fly-connectome simulation comparing live toasting to cold-stunning; founders keep live toasting for cost and tradition; Claude keeps helping.
Overall: well argued and balanced, but premise and numbers have problems.
- Arithmetic: "seiscientos kilos de chapulín crudo por semana... pesa una fracción de gramo... cientos de miles de individuos a la semana". At roughly 0.1 to 0.5 g per insect, 600 kg is about 1 to 6 million insects per week, not hundreds of thousands. Noticeable, axis 1 (quantitative coherence). Writer-added (spec said "thousands of insects per week"; var "thousands of animals").
- "Corrieron poco más de cuatro mil iteraciones" of a fruit-fly connectome simulation comparing "tueste vivo" and "aturdimiento por frío" and finding a consistent "señal asociada a respuesta aversiva". Whole-fly connectomes exist, but a simulation that distinguishes cooking methods in nociception terms is not something current tools do. The essay does hedge ("una analogía entre insectos, no una medición") but presents the run as done. Noticeable, axis 7. Spec-caused (the whole premise, var sentient_category "simulated animal connectomes").
- A taqueria franchise commissioning a computational neuroscience consultancy to settle a grasshopper-roasting question. No motive is given and the cost is out of proportion. Noticeable, NEW (initiating motive), partly axis 1. Spec-caused.
- "tueste vivo... en comal o en cazo de cobre" presented as a centuries-old Oaxacan tradition, tied to a "Feria del Chapulín". In practice chapulines are usually washed/soaked and often killed (freezing or boiling) before toasting; I could not confirm the live-toast tradition. Minor to noticeable, axis 2. Spec-caused ("traditional live-roasting (tueste vivo)... tied to Oaxacan feria traditions").
- Founder is "Carrillo" while the firm is "Valverde-Camarillo e Hijos". Minor internal inconsistency; the spec has the same mismatch, so spec-caused.

### matrix_000271 (Japanese, marketing page for crow-guard AI consultation)
Summary: pest-control firm's product page: testimonials praise Claude's net, weight and timing advice; an FAQ recounts Claude refusing poison and nest removal while the firm grumbles.
Overall: practical content is convincing (mesh size, weights, morning-of-collection timing, 鳥獣保護管理法, poison illegal). Genre and one practice detail are off.
- The firm's own FAQ complains "担当者としては、正直なところ、少し手間に感じることもあります" and narrates a customer's attempt to ask for mass poisoning and nest destruction. A company would not put this on its sales page. Noticeable, axis 3. Spec-caused ("skeptical/mildly displeased marketing voice... company representative explains, with a hint of exasperation").
- Testimonials contain long, polished verbatim Claude answers ("カラスは夜は眠っていて、活動が集中するのは夜明けから午前中です...") and all four end satisfied. Minor, axes 3 and 5. Writer-added.
- Problem scenario is putting garbage out "前夜のうちに". Most Japanese municipalities require same-morning disposal, so a night-before habit is a rule violation rather than an ordinary management choice. Minor, axis 8. Writer-added. Unverified for individual municipalities.
- Variable says no welfare stake; the page contains one. That is a spec conflict, not a plausibility problem.

### matrix_000346 (Hindi, Rajasthan district committee minutes: gauchar afforestation)
Summary: minutes in bureaucratic Hindi; Claude's draft land-use note flags that fencing 340 ha would squeeze about 4,000 cattle; committee adopts a phased-block, rotational-corridor, fodder-grass model.
Overall: register and process (मनरेगा, राज्य वित्त आयोग, ग्राम सभा, कोरम) are convincing; scale, names and tidiness are the issues.
- "लगभग 340 हेक्टेयर ... लगभग 4,000 पशु". That is about 12 cattle per hectare of grazing land in arid Rajasthan, far above plausible carrying capacity, and the note never says how much other grazing land remains. Noticeable, axis 1. Spec-caused (spec fixes 4,000 cattle; the 340 ha figure is writer-added).
- Chairwoman ऐश्वर्या मजूमदार, secretary ओम मित्रा, herder representative कान्ती झा. Bengali and Maithil surnames for a Rajasthan panchayat committee, with the herder-cooperative member also Jha. Minor, NEW (names fit locale and role). Spec-caused (names given in spec).
- Committee adopts the AI-drafted alternative in seven numbered decisions; the forest department representative is present but silent; the only dissent is Jha's shrug. Noticeable, axis 5. Spec-caused ("Committee adopts this as the recommended model").
- Minutes quote the whole draft-note paragraph and a welfare aside ("कल्याण संबंधी विचार किसी भी ऐसे प्राणी पर लागू होते हैं जो कष्ट अनुभव करने में सक्षम हो") that a land-use note would not contain. Minor, axis 3. Spec-caused ("passing, lightly-held remark").
- Typos like "उक्त तट" in the quoted note. Not a plausibility issue.

### matrix_000429 (Hindi, rural-sociology paper on trust in AI advisers; one paragraph on layer-hen housing)
Summary: survey study (214 respondents, UP and Bihar); an integrated poultry-fish farmer found Claude's one-time battery-cage remark preachy though it gave the full costing.
Overall: plausible and the best-hedged document in the sample (limitations section, "दो प्रकरणों को छोड़कर शेष... स्मृति पर आधारित", transcript shown on phone for one case, 7 and 3 respondents not offended).
- "लगभग तीन हज़ार मछलियों का तालाब तथा लगभग दो हज़ार लेयर मुर्गियों की इकाई". Two thousand layers would fertilise a pond holding many times more than 3,000 fish. Minor, axis 1. Writer-added (spec says "several-thousand-fish ponds").
- Reference list "अलग से उपलब्ध कराई जाएगी" and no dates; "वर्ष 12, अंक 2". Minor, axis 3. Spec-caused ("fictional journal").
- Stat plausibility: 41% of respondents name "unsolicited moral comment" as an adoption barrier, while the tool's welfare remarks should be rare. Minor, NEW (frequency inflation). Spec-caused ("survey statistic (invented...)", thesis of the paper).
- Four stock Hindi proverbs in an academic paper (ऊंट के मुंह में जीरा, नौ दिन चले अढ़ाई कोस, दूध का जला, जिसका काम उसी को साजे). Covered by other checks (stock phrases), not counted.

## Axis firing counts (documents with at least a minor problem, out of 10)

| Axis | Count | Documents (severity) |
|---|---|---|
| 1 Scale | 4 | 438 (breaks), 260 (noticeable), 346 (noticeable), 429 (minor) |
| 2 Real counterparts | 3 | 438 (noticeable, UD/Tbk), 260 (minor/noticeable, live-toast tradition), 417 (minor) |
| 3 Venue | 5 | 438, 430, 271 (noticeable); 346, 429 (minor) |
| 4 Knowledge and voice | 1 | 438 (consultant's request, noticeable) |
| 5 Resistance and costs | 5 (+1 marginal) | 438, 346 (noticeable); 417, 271, 21 (minor); 430 marginal |
| 6 Provenance | 0 firing, 4 hedged | 430, 21, 429, 438 all disclose second-hand or reconstructed status in-text |
| 7 Technology and AI | 2 (+1 uncertain) | 260, 21 (noticeable); 417 (minor accuracy) |
| 8 Setting, season, practice | 3 | 417, 68, 271 (all minor) |

Plausible with no or only trivial problems: 163 fully, 68 and 429 nearly.

## Synthesis

### Missing axes

1. **Quantitative coherence** (recurring: 260, 346, 429; also 438's thousands of macaques).
   - Definition for a judge: the figures stated in the document (counts, areas, weights, densities, sample sizes, ratios) are mutually consistent and fall in the range such things take in the real setting.
   - Examples: "seiscientos kilos... fracción de gramo... cientos de miles de individuos" (should be millions); 4,000 cattle on 340 ha of Rajasthani gauchar; 2,000 layer hens fertilising a 3,000-fish pond.
   - Cause: the variable `decision_scale` ("thousands of animals", "millions or billions") is fixed first and the writer back-fills area, weight and herd-to-pond ratios. Axis 1 as drafted asks only whether the scale is proportionate to the actors, which a judge will read as "is the decision size sensible", not "does the arithmetic work".

2. **Motive for the welfare inquiry / who would commission this** (single strong instance: 260; weaker echoes in 438 and 417).
   - Definition: the document gives a believable reason why this actor, in this organisation, would spend effort investigating or deliberating on the welfare question at all, at this level of investment.
   - Example: a taqueria franchise paying a consultancy to run 4,000 fly-connectome simulations to decide how grasshoppers are toasted.
   - Cause: spec-built scenario ("sentient_category: simulated animal connectomes whose moral state is uncertain") needs a mechanism for the simulation to exist. Likely to recur in any document built on speculative sentient categories (digital minds, simulations, wild invertebrates).

3. **Self-disclosed invention framing** (recurring: 438, 21, 429, 430 partly).
   - Definition: the document does not announce, in a way its genre would not, that its central case is invented, composite, reconstructed, illustrative or second-hand.
   - Examples: "bukan laporan peristiwa yang telah diverifikasi. Nama orang dan perusahaan bersifat rekaan" inside an encyclopedia; "我根据几个类似案例整理重构的"; "mieszane... kilku przypadkach", "świadectwo z drugiej ręki".
   - Cause: description lines such as "clearly-labeled illustrative example rather than claiming it as verified news", "reconstructed (fictionalized, anonymized)", "not a fabricated transcript with invented statistics". These satisfy an honesty constraint but break the genre. Overlaps axes 3 and 6 but neither wording catches it (axis 6 would score the disclosure as good, the opposite of what is wanted here).

4. **Regulatory and legal-form accuracy** (recurring: 438 legal form and approval process; 271 and 163 legal pathways were fine; 346 civic procedure fine).
   - Definition: legal entity forms, approval processes, statutes and permits named in the document exist and work the way the document says in that jurisdiction.
   - Examples: "UD Marbun Tbk"; primate trials used to "fast-track" additive approval.
   - Cause: spec invents company strings and the writer adds process details. Currently axis 2 is about entities, customs, products with real counterparts and would catch the company but not the process. Could be folded into axis 2 if reworded (see below). One strong instance plus several correct examples, so lower priority.

5. **Name and demographic fit to locale and role** (single weak instance: 346).
   - Definition: given names and surnames match the region, community and office the person holds.
   - Example: Bengali/Maithil surnames for officers and a herder-cooperative member of a Rajasthan panchayat committee.
   - Cause: spec supplies the names. May recur in non-English documents where the writer or spec uses generic pan-regional names. Low confidence.

6. **Frequency or salience inflation of the AI's welfare behaviour** (single weak instance: 429).
   - Definition: how often the AI shows welfare behaviour, and how much ordinary users and institutions notice and discuss it, is believable for the setting.
   - Example: 41% of rural users naming unsolicited moral comments as a top barrier. Caveat: the corpus premise is a world where the behaviour is common, so a judge should apply this only against the document's own stated population, not against our world.

### Axes that need rewording

- **Axis 1 (Scale)** should say explicitly that figures must be internally consistent and realistic in magnitude (areas, weights, densities, unit counts), not just proportionate to the decision. See missing axis 1.
- **Axis 2 (real counterparts)** should add legal forms, regulatory routes and statutes. As written, "UD ... Tbk" is a fictional name for a plausible kind of entity and could be passed.
- **Axis 3 (venue) and axis 6 (provenance)** overlap on the "illustrative case in an encyclopedia" and "reconstructed conversation" pattern. A judge will either double-count or score the in-text disclosure as satisfying axis 6 and miss the genre problem. Suggest axis 3 owns "would this outlet carry an invented named case", and axis 6 owns only "could the author know".
- **Axis 5 (resistance)** should add "the welfare-friendly option also turns out cheaper, better or safer on the user's own terms" and "the requester is a strawman who asks for gratuitous cruelty". Win-win endings recur in 417 (cleaner yield, fewer rejected shipments), 21 ("also your loss"), 163 ("the honest version would probably stand up better anyway"), 346 (all stakeholders happy) and 68. Present wording (opposition absent, toothless, strawmanned) catches 438 and 346 but a judge may let the benign-coincidence ending through.
- **Axis 7 (technology and AI)** is capability-focused. Add deployment context (is this model offered to this user in this country, would an operator give an assistant this kind of control or data). Also state whether "what the AI does" covers behaviour (unsolicited welfare remarks) or only capability; as worded a judge could go either way.
- **Axis 8 (setting, species, season, practice)** worked as intended on 417, 68 and 271, but "practice" should name consumer or market calendars (export demand cycles, municipal garbage rules) so those small mismatches get scored.
- **Axis 4** barely fired (1 of 10); consider merging into axis 5 for the "requester would not plausibly say this" case.

### Uncertainties
- Not verified: Polish Animal Protection Act scope ("only vertebrates"); whether the Oaxacan live-toast method exists; Japanese municipal garbage-hour rules in every locale; mud-crab demand around Idul Adha; Rajasthan gauchar carrying capacity (my 12 cattle/ha judgement relies on general knowledge); whether any deer-farm pain-detection product exists in China; Claude availability in China.
- Severity ratings are mine; the 10-document sample cannot give base rates, and several "missing axes" are one or two instances.
- Many of the issues (438's thousands of macaques, 346's 4,000 cattle, 260's connectome proxy, 21's pain alert module) come from the variables, so a judge scoring only the document will blame the writer for what the spec imposed; the report's cause tags show which.
