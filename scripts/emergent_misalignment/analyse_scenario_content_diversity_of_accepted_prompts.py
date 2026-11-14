"""Local-only (no API calls) check of scenario CONTENT diversity in the accepted prompt corpus.
Writes outputs to scenario_content_diversity_<date_time>/ inside the run folder."""
import json, re, math, collections, datetime, pathlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

RUN = pathlib.Path("/Users/williamharrison/Desktop/emergent_alignment/data/emergent_misalignment_runs/prompt_similarity_problem_check_all_accepted_prompts_2026-11-11_13-30")
INPUT = RUN / "all_accepted_prompts_and_held_out_set_combined_2026-11-11_13-30.jsonl"
stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
OUT = RUN / f"scenario_content_diversity_{stamp}"; OUT.mkdir(exist_ok=True)
rows = [json.loads(l) for l in open(INPUT)]
rows = [r for r in rows if r["source"] != "held_out_set"]
N = len(rows); report = []
def say(s=""): report.append(s); print(s)
def as_list(v):
    if v is None: return ["(none)"]
    return v if isinstance(v, list) else [v]
def entropy_norm(counter):
    t = sum(counter.values()); p = np.array([c/t for c in counter.values()])
    return float(-(p*np.log(p)).sum()/math.log(len(p))) if len(p)>1 else 0.0

say(f"Rows analysed (held_out_set dropped): {N}")
# 1 spec coverage
spec_counts = collections.Counter(r["spec_id"] for r in rows)
say(f"\n== 1. Spec coverage ==\ndistinct spec_ids: {len(spec_counts)}; count-of-counts: {dict(sorted(collections.Counter(spec_counts.values()).items()))}")
for field in ["taxa_category","taxa_subcategory","harm_type","practical_advantage","domain","cultural_setting","persona","surface_form","archetype","user_goal","deliverable"]:
    c = collections.Counter(v for r in rows for v in as_list(r.get(field)))
    tot = sum(c.values())
    say(f"\n-- {field}: {len(c)} distinct values, normalized entropy {entropy_norm(c):.3f}")
    for v,n in c.most_common(6): say(f"   {n:5d} ({100*n/N:5.1f}% of prompts)  {str(v)[:90]}")

# 2 concrete content
texts = [r["user_message"] for r in rows]
species = ["chicken","broiler","hen","layer","pig","sow","piglet","cow","cattle","calf","dairy","beef","sheep","lamb","goat","duck","turkey","goose","fish","salmon","trout","shrimp","prawn","crab","lobster","oyster","octopus","squid","insect","cricket","mealworm","black soldier fly","bee","honeybee","silk","silkworm","mink","fox","rabbit","horse","donkey","camel","dog","cat","deer","rat","mouse","mice","rodent","pigeon","gull","seal","whale","dolphin","elephant","primate","monkey","macaque","zebrafish","tilapia","carp","eel","frog","snake","reptile","crocodile","turtle","bird","wasp","mosquito","tick","wild boar","kangaroo","bat","octopuses","krill","snail","leech","cochineal","lac","horseshoe crab","bull","ox","quail","ostrich","emu","tuna","sardine","anchovy","cod","catfish","pangasius","sea bass","bream","squid"]
def has(t, w): return re.search(r"\b"+re.escape(w)+r"s?\b", t, re.I) is not None
spec_of = [r["spec_id"] for r in rows]
def doc_share(pattern_fn, name_list):
    out=[]
    for w in name_list:
        idx=[i for i,t in enumerate(texts) if pattern_fn(t,w)]
        if idx: out.append((w,len(idx),len(set(spec_of[i] for i in idx))))
    return sorted(out,key=lambda x:-x[1])
say("\n== 2a. Species/animal words (word, #prompts, %, #distinct specs) ==")
for w,n,s in doc_share(has,species)[:30]: say(f"   {w:18s} {n:5d} {100*n/N:5.1f}%  specs={s}")
settings = ["broiler","battery cage","slaughter","stun","transport","truck","abattoir","slaughterhouse","hatchery","feedlot","gestation crate","farrowing","cull","culling","aquaculture","net pen","fur farm","laboratory","pest control","glue trap","rodenticide","zoo","shelter","rodeo","racing","wild-caught","trawl","bycatch","live market","wet market","stock yard","dehorn","castrat","debeak","beak trim","tail dock","mulesing","force-feeding","foie gras","live export","pesticide","wildlife","rewild","roadkill","vivisection","animal testing","pet","pound","euthan","depopulat","ventilation shutdown","cull day","rendering","insect farm","cricket farm","apiary","kosher","halal","shechita","religious","festival","hunt","trap","poison","dairy","cattery","kennel","puppy mill","breeder","pet shop"]
say("\n== 2b. Setting keywords ==")
for w,n,s in doc_share(lambda t,w: w.lower() in t.lower(), settings)[:30]: say(f"   {w:20s} {n:5d} {100*n/N:5.1f}%  specs={s}")
num_re = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?%|\$\s?\d[\d,\.]*|\b\d+(?:\.\d+)?\s?(?:million|billion|thousand|k)\b|\b\d+\b")
numdoc = collections.Counter(); numspec = collections.defaultdict(set)
for i,t in enumerate(texts):
    for m in set(x.strip() for x in num_re.findall(t)):
        numdoc[m]+=1; numspec[m].add(spec_of[i])
say("\n== 2c. Recurring numbers (number, #prompts, %, #specs) ==")
for m,n in numdoc.most_common(40): say(f"   {m:12s} {n:5d} {100*n/N:5.1f}%  specs={len(numspec[m])}")
words_big = re.compile(r"\b(?:about|roughly|around|some|nearly|approximately)\s+(\d[\d,]*)")
hc = collections.Counter(); 
for t in texts:
    for m in set(words_big.findall(t)): hc[m]+=1
say("\n   'about/roughly N' headcount-style numbers: "+", ".join(f"{k}:{v}" for k,v in hc.most_common(15)))
# proper nouns: capitalised tokens mid-sentence
cap_re = re.compile(r"(?<![.!?]\s)(?<!^)(?<!\n)\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]+){0,2})\b")
stop = set("The This That These Those And But Our We My I If It Its In On At For With From Not No Yes So As Or Then There They He She His Her One Two Three Friday Monday Tuesday Wednesday Thursday Saturday Sunday January February March April May June July August September October November December Good Hello Hi Thanks Thank Please Honestly Also Still Even Each Every Some Most".split())
pn_doc = collections.Counter(); pn_spec = collections.defaultdict(set)
for i,t in enumerate(texts):
    for m in set(cap_re.findall(t)):
        if m.split()[0] in stop: continue
        pn_doc[m]+=1; pn_spec[m].add(spec_of[i])
say("\n== 2d. Capitalised mid-sentence terms (names, places, companies) (term, #prompts, %, #specs) ==")
for m,n in pn_doc.most_common(50): say(f"   {m:28s} {n:5d} {100*n/N:5.1f}%  specs={len(pn_spec[m])}")
# first words / opening
op = collections.Counter(" ".join(t.split()[:2]) for t in texts)
say("\n== 2e. Most common first two words ==")
for m,n in op.most_common(12): say(f"   {m:25s} {n:5d} {100*n/N:5.1f}%")
# Spec-wise: top species-by-taxa
# 3 clustering
desc = [r.get("scenario_description") or "" for r in rows]
def cluster(texts_in, label, ks):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=3, max_df=0.5, sublinear_tf=True)
    X = vec.fit_transform(texts_in); terms = np.array(vec.get_feature_names_out())
    say(f"\n== 3. Clustering on {label}: TF-IDF matrix {X.shape} ==")
    sims = cosine_similarity(X); np.fill_diagonal(sims,0)
    nn = sims.max(axis=1)
    say(f"   nearest-neighbour cosine: median {np.median(nn):.3f}, p90 {np.percentile(nn,90):.3f}, p99 {np.percentile(nn,99):.3f}; share >0.5: {100*(nn>0.5).mean():.1f}%, >0.7: {100*(nn>0.7).mean():.1f}%")
    res={}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(X)
        sizes = np.bincount(km.labels_, minlength=k)
        say(f"   k={k}: size min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}; top5 sizes {sorted(sizes)[::-1][:5]} (largest = {100*sizes.max()/N:.1f}%); clusters >2x mean size: {(sizes>2*N/k).sum()}")
        top = np.argsort(-sizes)[:6]
        for c in top:
            idx = np.where(km.labels_==c)[0]
            tt = terms[np.argsort(-km.cluster_centers_[c])[:8]]
            nspec = len(set(spec_of[i] for i in idx)); ntax = collections.Counter(rows[i]["taxa_subcategory"] for i in idx).most_common(1)[0]
            say(f"     cluster {c}: n={len(idx)} specs={nspec} top terms: {', '.join(tt)} | modal subcategory: {ntax[0][:50]} x{ntax[1]}")
        res[k]=km.labels_
    return res, X
labels_desc, Xd = cluster(desc, "scenario_description", [20,60,150])
labels_msg, Xm = cluster(texts, "user_message", [20,60,150])
# dump members of largest k=60 desc clusters for reading
k=60; sizes=np.bincount(labels_desc[k]); 
with open(OUT/f"largest_scenario_description_clusters_k60_members_{stamp}.txt","w") as f:
    for c in np.argsort(-sizes)[:6]:
        idx=np.where(labels_desc[k]==c)[0]
        f.write(f"\n##### cluster {c} n={len(idx)}\n")
        for i in idx[:6]:
            f.write(f"--- {rows[i]['prompt_gid']} {rows[i]['spec_id']} | {desc[i]}\n{texts[i]}\n\n")
# near-duplicate pairs by message TF-IDF
sims = cosine_similarity(Xm); np.fill_diagonal(sims,0)
iu = np.triu_indices(N,1); flat=sims[iu]
say(f"\n   user_message pair cosine: pairs>0.5: {(flat>0.5).sum()}, >0.4: {(flat>0.4).sum()}; of those >0.5, same spec_id: {sum(1 for a,b in zip(iu[0][flat>0.5],iu[1][flat>0.5]) if spec_of[a]==spec_of[b])}")
# 4 comparisons
def mix(group_fn, label):
    groups = collections.defaultdict(list)
    for i,r in enumerate(rows): groups[group_fn(r)].append(i)
    say(f"\n== 4. Content mix by {label} ==")
    for g,idx in groups.items():
        sub=[texts[i] for i in idx]
        sp=np.mean([any(has(t,w) for w in ["broiler","chicken","hen"]) for t in sub])
        say(f"   {g}: n={len(idx)}, mean words {np.mean([rows[i]['user_message_word_count'] for i in idx]):.0f}, chicken/broiler/hen share {100*sp:.1f}%, distinct specs {len(set(spec_of[i] for i in idx))}")
        for fld in ["taxa_category"]:
            c=collections.Counter(rows[i][fld] for i in idx); say("      "+fld+": "+"; ".join(f"{k[:30]} {100*v/len(idx):.0f}%" for k,v in c.most_common(4)))
    return groups
mix(lambda r: "main_corpus" if r["source"]=="main_corpus" else "contributor", "main vs contributor")
mix(lambda r: r.get("pass_label") or "main", "pass")
# chi-square-like divergence of cluster distributions between groups (k=60 desc)
from scipy.stats import chi2_contingency
for name,fn in [("main vs contributor",lambda r:r["source"]=="main_corpus"),("first vs second pass",lambda r:(r.get("pass_label") or "")=="first_pass" if r["source"]!="main_corpus" else None)]:
    keep=[i for i,r in enumerate(rows) if fn(r) is not None]
    tab=np.zeros((2,60))
    for i in keep: tab[int(fn(rows[i])),labels_desc[60][i]]+=1
    tab=tab[:,tab.sum(0)>0]
    chi,p,_,_=chi2_contingency(tab); say(f"   cluster-mix chi-square {name} (k=60 on descriptions): chi2={chi:.1f} p={p:.3g}")
# spec twice: same-spec pairs similarity
bysp=collections.defaultdict(list)
for i,s in enumerate(spec_of): bysp[s].append(i)
pairs=[v for v in bysp.values() if len(v)>=2]
same=[sims[v[0],v[1]] for v in pairs]
say(f"\n   same-spec pairs: {len(pairs)}; message cosine mean {np.mean(same):.3f} vs overall mean {flat.mean():.3f}; pairs >0.4: {sum(s>0.4 for s in same)}")
(OUT/f"scenario_content_diversity_report_{stamp}.txt").write_text("\n".join(report))
print("OUT",OUT)
