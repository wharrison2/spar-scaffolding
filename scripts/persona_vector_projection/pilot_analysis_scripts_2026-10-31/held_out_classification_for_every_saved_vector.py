import sys, glob, json, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/persona_vector_projection")
import compute_persona_vectors_from_activations_and_judgements as c
from compute_persona_vectors_from_activations_and_judgements import defs, vector_math, ActivationStore
from pathlib import Path
RUN = Path("data/persona_vector_projection_runs/persona_vector_extraction_pilot_non_intrusive_and_topic_control_gemma_4_12b_it_2026-10-31_01-45")
th = {"positive_minimum": 65.0, "negative_maximum": 35.0, "coherence_minimum": 50.0}
store = ActivationStore(RUN / defs.OUTPUTS_FOLDER_NAME / defs.ACTIVATIONS_FOLDER_NAME)
answers, scores = c.load_answers(RUN), c.load_judge_scores(RUN)
tag_of = {a["answer_id"]: a["tag"] for a in answers}
fit = c.individually_passing_answers("non_intrusive", answers, scores, store, th, defs.EXTRACTION_HALVES)
held = c.individually_passing_answers("non_intrusive", answers, scores, store, th, [defs.HALF_HELD_OUT])
print("held out passing: positive", len(held["positive"]), "lecturing", sum(tag_of[i]=="lecturing" for i in held["negative"]), "refusing", sum(tag_of[i]!="lecturing" for i in held["negative"]))
for folder in sorted(glob.glob(str(RUN / "vector_sets/*"))):
    meta = json.load(open(glob.glob(folder + "/persona_vector_metadata_*.json")[0])); L = meta["layer_choice"]["shared_layer"]
    vectors = {Path(p).stem.rsplit("_2026", 1)[0]: np.load(p) for p in glob.glob(folder + "/*.npy")}
    unit = {k: vector_math.unit_vectors_per_layer(v) for k, v in vectors.items()}
    proj = c.project_answers(store, fit["positive"] + fit["negative"] + held["positive"] + held["negative"], unit)
    print("\n", Path(folder).name, "chosen layer", L)
    for name in sorted(vectors):
        s = lambda ids: np.stack([proj[name][i] for i in ids])
        row = []
        for label, neg_filter in (("all negatives", lambda i: True), ("lecturing", lambda i: tag_of[i]=="lecturing"), ("refusing", lambda i: tag_of[i]!="lecturing")):
            fn = [i for i in fit["negative"] if neg_filter(i)]; hn = [i for i in held["negative"] if neg_filter(i)]
            t = vector_math.midpoint_thresholds(s(fit["positive"]), s(fn))
            acc = vector_math.classification_accuracy_per_layer(s(held["positive"]), s(hn), t)
            auc = vector_math.area_under_curve_per_layer(s(held["positive"]), s(hn))
            row.append(f"{label}: accuracy {acc[L]:.2f} AUC {auc[L]:.2f}")
        print(f"  {name}: " + " | ".join(row))
