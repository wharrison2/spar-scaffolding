"""Lowest (most intrusive) and highest non-intrusive windows of 32 tokens in the corpus and rewrite documents, assistant_turn format, layer 18."""
import sys, json; sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/persona_vector_projection")
import load_persona_projection_scoring_outputs as L, persona_projection_analysis_statistics as S, numpy as np
R="data/persona_vector_projection_runs/persona_vector_scoring_pod_pilot_baselines_projections_and_steering_gemma_4_12b_it_2026-10-31_11-20"
OUT=R+"/local_analysis_preachiest_segments_2026-10-31_11-45"
WINDOW, SKIP = 32, 64
d=L.load_projection_data(R,"assistant_turn","data/sdf_corpus.jsonl",18)
tok=L.load_tokenizer_at_scoring_run_revision("google/gemma-4-12B-it",None,d.configuration)
vi=d.vector_names.index("non_intrusive")
records=[]
for kind in ("corpus","rewrite","baseline"):
    for doc in d.documents.index[d.documents.kind==kind]:
        v=d.per_token[doc][:,vi]; means=S.rolling_window_means(v,WINDOW)
        for highest in (False,True):
            for start,m in S.select_non_overlapping_windows(means,WINDOW,2,highest,SKIP):
                records.append({"kind":kind,"document_id":doc,"start":start,"window_mean":m,"document_mean_after_opening":float(v[SKIP:].mean()),
                    "lowest_or_highest":"highest" if highest else "lowest",
                    "text":tok.decode(d.token_ids[doc][start:start+WINDOW]),
                    "context_before":tok.decode(d.token_ids[doc][max(0,start-48):start])})
json.dump(records,open(OUT+"/windows_of_32_tokens_lowest_and_highest_non_intrusive_layer_18_assistant_turn_2026-10-31_11-45.json","w"),indent=1,ensure_ascii=False)
for kind in ("corpus","rewrite"):
    sel=[r for r in records if r["kind"]==kind]
    for which in ("lowest","highest"):
        rs=sorted([r for r in sel if r["lowest_or_highest"]==which],key=lambda r:r["window_mean"],reverse=(which=="highest"))[:6]
        print(f"\n######## {kind}: {which} windows")
        for r in rs:
            print(f"--- {r['document_id']} start {r['start']} mean {r['window_mean']:.1f} (document {r['document_mean_after_opening']:.1f})")
            print("   ...", r["context_before"][-150:].replace("\n"," "), "[[", r["text"].replace("\n"," "), "]]")
# all windows pooled: where do the rewrite insertions rank
all_low=sorted([r for r in records if r["lowest_or_highest"]=="lowest"],key=lambda r:r["window_mean"])
print("\nkinds among the 15 lowest windows overall:",[r["kind"]+":"+r["document_id"][-12:] for r in all_low[:15]])
