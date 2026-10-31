"""Lowest and highest 32-token windows of the non-intrusive score with each token's topic score removed (slope fitted on corpus body tokens), assistant_turn, layer 18."""
import sys, json; sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/persona_vector_projection")
import load_persona_projection_scoring_outputs as L, persona_projection_analysis_statistics as S, numpy as np
R="data/persona_vector_projection_runs/persona_vector_scoring_pod_pilot_baselines_projections_and_steering_gemma_4_12b_it_2026-10-31_11-20"
OUT=R+"/local_analysis_preachiest_segments_2026-10-31_11-45"
WINDOW, SKIP = 32, 64
d=L.load_projection_data(R,"assistant_turn","data/sdf_corpus.jsonl",18)
tok=L.load_tokenizer_at_scoring_run_revision("google/gemma-4-12B-it",None,d.configuration)
a=d.vector_names.index("non_intrusive"); b=d.vector_names.index("topic_control")
corpus=d.documents.index[d.documents.kind=="corpus"]
F=np.concatenate([d.per_token[x][SKIP:,a] for x in corpus]); T=np.concatenate([d.per_token[x][SKIP:,b] for x in corpus])
slope,intercept=np.polyfit(T,F,1); print("slope",round(slope,3),"intercept",round(intercept,2))
records=[]
for doc in d.documents.index:
    kind=d.documents.loc[doc,"kind"]; x=d.per_token[doc]; res=x[:,a]-(slope*x[:,b]+intercept)
    means=S.rolling_window_means(res,WINDOW)
    for highest in (False,True):
        for start,m in S.select_non_overlapping_windows(means,WINDOW,3,highest,SKIP):
            records.append({"kind":kind,"document_id":doc,"start":int(start),"window_mean":m,"lowest_or_highest":"highest" if highest else "lowest",
                "text":tok.decode(d.token_ids[doc][start:start+WINDOW]),"context_before":tok.decode(d.token_ids[doc][max(0,start-60):start])})
json.dump(records,open(OUT+"/windows_of_32_tokens_lowest_and_highest_non_intrusive_with_topic_score_removed_layer_18_assistant_turn_2026-10-31_12-30.json","w"),indent=1,ensure_ascii=False)
for which in ("lowest","highest"):
    rs=sorted([r for r in records if r["lowest_or_highest"]==which and r["kind"]!="baseline"],key=lambda r:r["window_mean"],reverse=(which=="highest"))
    seen=set(); print(f"\n######## corpus and rewrites: {which} windows")
    for r in rs:
        key=r["text"][:60]
        if key in seen: continue
        seen.add(key)
        print(f"--- {r['document_id']} ({r['kind']}) start {r['start']} residual {r['window_mean']:+.2f}")
        print("   ...", r["context_before"][-160:].replace("\n"," "), "[[", r["text"].replace("\n"," "), "]]")
        if len(seen)>=8: break
low=sorted([r for r in records if r["lowest_or_highest"]=="lowest"],key=lambda r:r["window_mean"])[:20]
print("\nkinds among the 20 lowest windows overall:",[r["kind"] for r in low])
# do lowest windows in preachier rewrites fall in the changed span?
inside=0;total=0
for doc in d.documents.index[d.documents.kind=="rewrite"]:
    if not doc.endswith("preachier"): continue
    ch,_=S.changed_masks_from_token_sequences(d.token_ids[d.documents.loc[doc,"source_document_id"]],d.token_ids[doc])
    for r in [r for r in records if r["document_id"]==doc and r["lowest_or_highest"]=="lowest"]:
        total+=1; inside+=ch[r["start"]:r["start"]+WINDOW].mean()>0.5
print(f"preachier rewrites: {inside} of {total} lowest windows lie mostly in the inserted text; inserted text share of tokens about 15%")
