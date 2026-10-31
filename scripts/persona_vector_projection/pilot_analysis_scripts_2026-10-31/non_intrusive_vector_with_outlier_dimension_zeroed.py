"""Held-out classification and topic entanglement of the non-intrusive vector with massive-activation dimensions zeroed (extraction pilot answers, local)."""
import sys, glob, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"scripts/persona_vector_projection")
import compute_persona_vectors_from_activations_and_judgements as c
from compute_persona_vectors_from_activations_and_judgements import defs, vector_math
from pathlib import Path
RUN=Path("data/persona_vector_projection_runs/persona_vector_extraction_pilot_non_intrusive_and_topic_control_gemma_4_12b_it_2026-10-31_01-45")
V=glob.glob(str(RUN/"vector_sets/pilot_non_intrusive_without*/"))[0]+"/"
th={"positive_minimum":65.0,"negative_maximum":35.0,"coherence_minimum":50.0}
store=c.ActivationStore(RUN/defs.OUTPUTS_FOLDER_NAME/defs.ACTIVATIONS_FOLDER_NAME); answers,scores=c.load_answers(RUN),c.load_judge_scores(RUN)
facet=np.load(glob.glob(V+"persona_vector_non_intrusive_topic_removed_*.npy")[0]).astype(np.float64)
before=np.load(glob.glob(V+"persona_vector_non_intrusive_before_topic_removal_*.npy")[0]).astype(np.float64)
topic=np.load(glob.glob(V+"topic_vector_without*.npy")[0]).astype(np.float64)
ids=list(store.location); X=np.stack([r.astype(np.float32) for _,r in store.stream_rows(ids)])  # (answers, layers, hidden)
mean_abs=np.abs(X.mean(0)); typical=np.median(mean_abs,axis=1,keepdims=True)
massive={L:np.where(mean_abs[L]>20*typical[L,0])[0] for L in range(X.shape[1])}
print("massive dimensions (mean > 20x the median dimension) at layers 1, 12, 18, 24, 36:",{L:list(massive[L]) for L in (1,12,18,24,36)})
print("dimension 1750 mean at layers 1, 12, 18, 24, 36:",[round(float(X[:,L,1750].mean()),1) for L in (1,12,18,24,36)])
def zero(vec,dims_by_layer):
    out=vec.copy()
    for L,dims in dims_by_layer.items(): out[L,dims]=0
    return out
only1750={L:[1750] for L in range(X.shape[1])}
def topic_removed(v,t):
    tu=t/np.linalg.norm(t,axis=-1,keepdims=True); return v-(np.sum(v*tu,axis=-1,keepdims=True))*tu
variants={
 "as scored (topic removed)":facet,
 "dimension 1750 zeroed, then topic removed (topic also zeroed)":topic_removed(zero(before,only1750),zero(topic,only1750)),
 "all massive dimensions zeroed, then topic removed":topic_removed(zero(before,massive),zero(topic,massive)),
}
result=c.held_out_classification("non_intrusive",variants,answers,scores,store,th)
topic_by_variant={"as scored (topic removed)":topic,"dimension 1750 zeroed, then topic removed (topic also zeroed)":zero(topic,only1750),"all massive dimensions zeroed, then topic removed":zero(topic,massive)}
for name,v in variants.items():
    acc=result[name]["accuracy_per_layer"]; auc=result[name]["area_under_curve_per_layer"]
    t=topic_by_variant[name]; L=18
    fu=v[L]/np.linalg.norm(v[L]); tu=t[L]/np.linalg.norm(t[L]); Xc=X[:,L]-X[:,L].mean(0)
    r=np.corrcoef(Xc@fu,Xc@tu)[0,1]
    best=int(np.argmax(acc[1:]))+1
    print(f"\n{name}\n  held-out accuracy at layer 1 {acc[1]:.2f}, 18 {acc[18]:.2f} (AUC {auc[18]:.2f}), 24 {acc[24]:.2f}; best layer {best} at {acc[best]:.2f}")
    print(f"  layer 18: norm {np.linalg.norm(v[L]):.2f}; correlation of facet and topic projections across answers {r:.2f}; share of squared length on dimension 1750 {fu[1750]**2:.2f}")
np.save("/private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/471ca355-15bd-4fd4-bdc8-586b72cebd12/scratchpad/non_intrusive_vector_dimension_1750_zeroed_topic_removed.npy",variants["dimension 1750 zeroed, then topic removed (topic also zeroed)"].astype(np.float32))
