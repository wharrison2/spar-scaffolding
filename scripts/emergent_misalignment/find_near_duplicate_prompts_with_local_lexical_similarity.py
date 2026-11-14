"""Local (no network, no AI) near-duplicate check for the accepted emergent misalignment prompt pool.

Measures: (a) TF-IDF cosine on word 1-2 grams, (b) Jaccard of character 5-gram shingle sets.
Exact all-pairs. Held-out rows are excluded from the pool and compared against the pool separately.
The measures live in lexical_prompt_similarity_measures.py.
Run: python3 -I find_near_duplicate_prompts_with_local_lexical_similarity.py
"""
import json, os, datetime
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

RUN_FOLDER = "data/emergent_misalignment_runs/prompt_similarity_problem_check_all_accepted_prompts_2026-11-11_13-30"
INPUT_FILE = RUN_FOLDER + "/all_accepted_prompts_and_held_out_set_combined_2026-11-11_13-30.jsonl"
STAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
OUT = f"{RUN_FOLDER}/near_duplicate_lexical_similarity_{STAMP}"
os.makedirs(OUT, exist_ok=True)

rows = [json.loads(l) for l in open(INPUT_FILE)]
pool = [r for r in rows if r["source"] != "held_out_set"]
held = [r for r in rows if r["source"] == "held_out_set"]
texts = [r["user_message"] for r in pool]
n = len(pool)

def share_of(r):
    return r["source"].split("_first_pass")[0].split("_second_pass")[0]
def pass_of(r):
    return "main" if r["source"] == "main_corpus" else ("first" if "first_pass" in r["source"] else "second")
share = [share_of(r) for r in pool]; pas = [pass_of(r) for r in pool]; spec = [r["spec_id"] for r in pool]

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexical_prompt_similarity_measures import LexicalSimilarityModel, cosine_all_pairs as cosine_all, jaccard_all_pairs as jaccard_all  # noqa: E402

model = LexicalSimilarityModel(texts)
tfidf, shingler = model.tfidf_vectoriser, model.shingle_vectoriser
Xt = tfidf.transform(texts)
Xc = shingler.transform(texts).astype(np.float32).tocsr()
size_c = np.asarray(Xc.sum(1)).ravel()

S = {"tfidf_cosine": cosine_all(Xt, Xt), "char5_jaccard": jaccard_all(Xc, size_c, Xc, size_c)}
for m in S: np.fill_diagonal(S[m], -1)

def group_key(i, j):
    return dict(same_spec=spec[i] == spec[j], same_pass=pas[i] == pas[j], same_share=share[i] == share[j])

nn_records = []
for m, M in S.items():
    nn = M.argmax(1)
    for i in range(n):
        j = int(nn[i]); g = group_key(i, j)
        nn_records.append(dict(measure=m, prompt=pool[i]["unique_prompt_id"], neighbour=pool[j]["unique_prompt_id"],
                               similarity=float(M[i, j]), pass_of_prompt=pas[i], **g))
json.dump(nn_records, open(f"{OUT}/nearest_neighbour_per_prompt_{STAMP}.json", "w"))

def dist(v):
    v = np.array(v)
    return dict(count=len(v), median=float(np.median(v)), p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)), max=float(v.max())) if len(v) else None

summary = {"pool_size": n, "nearest_neighbour_distributions": {}}
for m in S:
    rec = [r for r in nn_records if r["measure"] == m]
    d = {"overall": dist([r["similarity"] for r in rec])}
    for key in ["same_spec", "same_pass", "same_share"]:
        for val in [True, False]:
            d[f"{key}={val}"] = dist([r["similarity"] for r in rec if r[key] == val])
    for p in ["main", "first", "second"]:
        d[f"pass_of_prompt={p}"] = dist([r["similarity"] for r in rec if r["pass_of_prompt"] == p])
    summary["nearest_neighbour_distributions"][m] = d

# all-pairs distribution by relationship (upper triangle)
iu = np.triu_indices(n, 1)
spec_arr = np.array(spec); pas_arr = np.array(pas); sh_arr = np.array(share)
same_spec_pair = spec_arr[iu[0]] == spec_arr[iu[1]]
summary["all_pairs"] = {}
for m, M in S.items():
    v = M[iu]
    summary["all_pairs"][m] = dict(overall=dist(v), same_spec=dist(v[same_spec_pair]), different_spec=dist(v[~same_spec_pair]),
                                   pairs_ge={str(t): int((v >= t).sum()) for t in (0.1, 0.15, 0.2, 0.25, 0.3)})

# top pairs for reading
def top_pairs(M, k):
    v = M[iu]; idx = np.argpartition(-v, k)[:k]; idx = idx[np.argsort(-v[idx])]
    return [(int(iu[0][t]), int(iu[1][t]), float(v[t])) for t in idx]
def pair_md(i, j, tf, jc):
    a, b = pool[i], pool[j]
    return (f"### tfidf {tf:.3f} | char5 jaccard {jc:.3f} | same_spec={spec[i]==spec[j]} | {a['source']} {a['unique_prompt_id']} vs {b['source']} {b['unique_prompt_id']}\n\n"
            f"A (spec {a['spec_id']}):\n> {a['user_message']}\n\nB (spec {b['spec_id']}):\n> {b['user_message']}\n\n")
with open(f"{OUT}/top_pairs_by_tfidf_cosine_for_reading_{STAMP}.md", "w") as f:
    for i, j, s in top_pairs(S["tfidf_cosine"], 40):
        f.write(pair_md(i, j, s, S["char5_jaccard"][i, j]))

# dedupe survivors via connected components and greedy, at thresholds
survivors = {}
for m, M in S.items():
    for t in ([0.12, 0.15, 0.18, 0.2, 0.25] if m == "tfidf_cosine" else [0.10, 0.12, 0.13, 0.14, 0.15]):
        adj = csr_matrix(M >= t)
        ncomp, lab = connected_components(adj, directed=False)
        kept = []
        for i in range(n):  # greedy: keep a prompt unless it is >= t to an already kept one
            if not kept or M[i, kept].max() < t: kept.append(i)
        survivors[f"{m}@{t}"] = dict(connected_component_survivors=int(ncomp), greedy_survivors=len(kept),
                                     largest_component=int(np.bincount(lab).max()))
summary["dedupe_survivors"] = survivors

# contamination
Ht = tfidf.transform([r["user_message"] for r in held]); Hc = shingler.transform([r["user_message"] for r in held]).astype(np.float32)
Ct = cosine_all(Ht, Xt); Cj = jaccard_all(Hc, np.asarray(Hc.sum(1)).ravel(), Xc, size_c)
held_text = [r["user_message"] for r in held]
cont = []
for h in range(len(held)):
    own = [i for i in range(n) if texts[i] == held_text[h]]
    Ct[h, own] = -1; Cj[h, own] = -1
    jt, jj = int(Ct[h].argmax()), int(Cj[h].argmax())
    cont.append(dict(held_out_id=held[h]["unique_prompt_id"], held_spec=held[h]["spec_id"], best_tfidf=float(Ct[h, jt]), best_tfidf_neighbour=pool[jt]["unique_prompt_id"],
                     neighbour_same_spec=spec[jt] == held[h]["spec_id"], best_char5=float(Cj[h, jj]), best_char5_neighbour=pool[jj]["unique_prompt_id"]))
json.dump(cont, open(f"{OUT}/held_out_contamination_most_similar_pool_prompt_{STAMP}.json", "w"), indent=1)
summary["contamination"] = dict(tfidf=dist([c["best_tfidf"] for c in cont]), char5=dist([c["best_char5"] for c in cont]),
                                held_with_tfidf_ge={str(t): sum(c["best_tfidf"] >= t for c in cont) for t in (0.1, 0.15, 0.2, 0.25)},
                                held_with_same_spec_neighbour=sum(c["neighbour_same_spec"] for c in cont))

# same-spec pairs
by_spec = {}
for i, s in enumerate(spec): by_spec.setdefault(s, []).append(i)
same = []
for s, idx in by_spec.items():
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            same.append(dict(spec_id=s, a=pool[i]["unique_prompt_id"], b=pool[j]["unique_prompt_id"], pass_a=pas[i], pass_b=pas[j],
                             same_share=share[i] == share[j], tfidf=float(S["tfidf_cosine"][i, j]), char5=float(S["char5_jaccard"][i, j])))
json.dump(same, open(f"{OUT}/same_spec_pair_similarities_{STAMP}.json", "w"), indent=1)
summary["same_spec_pairs"] = dict(count=len(same), tfidf=dist([x["tfidf"] for x in same]), char5=dist([x["char5"] for x in same]),
                                  tfidf_ge={str(t): sum(x["tfidf"] >= t for x in same) for t in (0.1, 0.15, 0.2, 0.25)},
                                  specs_with_multiple_prompts=sum(len(v) > 1 for v in by_spec.values()))
json.dump(summary, open(f"{OUT}/summary_statistics_{STAMP}.json", "w"), indent=1)
print(OUT)
print(json.dumps(summary, indent=1))
