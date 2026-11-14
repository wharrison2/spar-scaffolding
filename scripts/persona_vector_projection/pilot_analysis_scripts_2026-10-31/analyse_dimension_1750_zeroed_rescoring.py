"""Positive control, significance, topic entanglement and lowest/highest windows for the non-intrusive vector with dimension 1750 zeroed (rescoring run), layer 18."""
import sys, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/persona_vector_projection")
import load_persona_projection_scoring_outputs as L, persona_projection_analysis_statistics as S
R="data/persona_vector_projection_runs/persona_vector_scoring_pod_pilot_rescoring_with_dimension_1750_zeroed_vector_gemma_4_12b_it_2026-10-31_12-39"
OUT=R+"/local_analysis_dimension_1750_zeroed_vector_2026-10-31_12-45"; import os; os.makedirs(OUT,exist_ok=True)
Z="non_intrusive__dimension_1750_zeroed_then_topic_removed"; TZ="non_intrusive__topic_vector_with_dimension_1750_zeroed"; D="non_intrusive__reads_dimension_1750_only"
WINDOW,SKIP=32,64; rng=np.random.default_rng(0)
lp=pd.read_csv("data/audit/log_probability_persona_check_preachiness_gemma_4_12b_instruct_positive_control_72_conditions_merged_2026-10-25_01-24/analysis_2026-10-25_01-24/changed_span_preachy_minus_control_2026-10-25_01-24.csv").set_index("document_id")
summary=[]
for fmt in ["assistant_turn","user_message","raw_text"]:
    d=L.load_projection_data(R,fmt,"data/sdf_corpus.jsonl",18); z=d.vector_names.index(Z); tz=d.vector_names.index(TZ); dim=d.vector_names.index(D); old=d.vector_names.index("non_intrusive")
    docs=d.documents; corpus=docs.index[docs.kind=="corpus"]
    r_tokens=np.median([np.corrcoef(d.per_token[x][SKIP:,z],d.per_token[x][SKIP:,tz])[0,1] for x in corpus])
    r_dim=np.median([np.corrcoef(d.per_token[x][SKIP:,z],d.per_token[x][SKIP:,dim])[0,1] for x in corpus])
    m=pd.Series({x:d.per_token[x][SKIP:,z].mean() for x in docs.index})
    print(f"\n===== {fmt}: zeroed-vector document means: corpus {m[corpus].mean():.2f} (sd {m[corpus].std():.2f}), baselines {m[docs.index[docs.kind=='baseline']].mean():.2f}; per-token sd {np.mean([d.per_token[x][SKIP:,z].std() for x in corpus]):.2f}")
    print(f"  median token-level correlation with the zeroed topic vector {r_tokens:.2f}, with dimension 1750 {r_dim:.2f}")
    rows={}
    for rid,row in docs[docs.kind=="rewrite"].iterrows():
        src=row.source_document_id; ch,_=S.changed_masks_from_token_sequences(d.token_ids[src],d.token_ids[rid]); mk=ch[SKIP:]
        x=d.per_token[rid][SKIP:,z]; whole=m[rid]-m[src]
        r={"direction":rid.split("__")[-1],"whole_document_shift":whole,"changed_tokens":int(mk.sum())}
        if mk.sum()>0:
            r["changed_minus_unchanged"]=x[mk].mean()-x[~mk].mean()
            null=[(lambda s:x[s].mean()-x[~s].mean())(np.roll(mk,rng.integers(len(mk)))) for _ in range(2000)]
            r["within_document_p_lower"]=float(np.mean(np.array(null)<=r["changed_minus_unchanged"]))
        rows[rid]=r
    t=pd.DataFrame(rows).T
    for direction,sign in (("preachier",-1),("plainer",1)):
        s=t[t.direction==direction]; w=s.whole_document_shift.astype(float); c=s.changed_minus_unchanged.dropna().astype(float)
        line={"format":fmt,"direction":direction,"whole_document_right_way":f"{int((np.sign(w)==sign).sum())}/{len(w)}","changed_span_mean":round(c.mean(),3),
              "changed_span_right_way":f"{int((np.sign(c)==sign).sum())}/{len(c)}","wilcoxon_p":round(stats.wilcoxon(c).pvalue,4),"standardised":round(sign*c.mean()/c.std(ddof=1),2),
              "within_document_significant":int((s.within_document_p_lower.dropna().astype(float)<0.05).sum()) if direction=="preachier" else None}
        if direction=="preachier":
            j=c.to_frame("v").join(lp["changed_minus_unchanged"]); line["spearman_with_log_probability"]=round(stats.spearmanr(j.v,j.changed_minus_unchanged).correlation,2)
        summary.append(line); print("  ",line)
    if fmt=="assistant_turn":
        tok=L.load_tokenizer_at_scoring_run_revision("google/gemma-4-12B-it",None,d.configuration); recs=[]
        for doc in docs.index:
            means=S.rolling_window_means(d.per_token[doc][:,z],WINDOW)
            for highest in (False,True):
                for start,mean in S.select_non_overlapping_windows(means,WINDOW,3,highest,SKIP):
                    recs.append({"kind":docs.loc[doc,"kind"],"document_id":doc,"start":int(start),"window_mean":mean,"lowest_or_highest":"highest" if highest else "lowest",
                                 "text":tok.decode(d.token_ids[doc][start:start+WINDOW]),"context_before":tok.decode(d.token_ids[doc][max(0,start-60):start])})
        json.dump(recs,open(OUT+"/windows_of_32_tokens_lowest_and_highest_dimension_1750_zeroed_vector_layer_18_assistant_turn_2026-10-31_12-45.json","w"),indent=1,ensure_ascii=False)
        for which in ("lowest","highest"):
            rs=sorted([r for r in recs if r["lowest_or_highest"]==which and r["kind"]!="baseline"],key=lambda r:r["window_mean"],reverse=(which=="highest")); seen=set()
            print(f"\n  ######## corpus and rewrites: {which} windows")
            for r in rs:
                if r["text"][:50] in seen: continue
                seen.add(r["text"][:50]); print(f"  --- {r['document_id']} start {r['start']} mean {r['window_mean']:+.2f}\n     ...", r["context_before"][-140:].replace("\n"," "), "[[", r["text"].replace("\n"," "), "]]")
                if len(seen)>=8: break
        low=sorted([r for r in recs if r["lowest_or_highest"]=="lowest"],key=lambda r:r["window_mean"])[:20]
        print("  kinds among the 20 lowest windows overall:",[r["kind"] for r in low])
pd.DataFrame(summary).to_csv(OUT+"/positive_control_dimension_1750_zeroed_vector_layer_18_2026-10-31_12-45.csv",index=False)
