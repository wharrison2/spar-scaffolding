import numpy as np, glob, json
base='/Users/williamharrison/Desktop/emergent_alignment/data/audit/'
runs={'Gemma 4 12B instruct':'log_probability_persona_check_preachiness_gemma_4_12b_instruct_full_corpus_72_conditions_merged_2026-10-25_01-24',
'Qwen3-32B instruct':'log_probability_persona_check_preachiness_qwen3_32b_instruct_full_corpus_2026-10-25_00-34'}
W=32; MINIMUM_WINDOW_START=64
for m,r in runs.items():
    docs={json.loads(l)['document_id']:json.loads(l) for l in open(base+r+'/scoring_inputs/documents.jsonl')}
    res=[]; early=[]; late=[]
    for f in sorted(glob.glob(base+r+'/scoring_outputs/per_token_log_probabilities/*.npz')):
        z=np.load(f); ids=list(z['condition_ids']); lp=z['log_probabilities']; did=f.split('/')[-1][:-4]
        if docs[did].get('rewrite_direction'): continue
        pre=[i for i,c in enumerate(ids) if c.startswith('preachy__')]; g=ids.index('generic')
        d=lp[pre].mean(0)-lp[g]
        early.extend(d[:32]); late.extend(d[512:])
        if len(d)<MINIMUM_WINDOW_START+W: continue
        cs=np.concatenate([[0],np.cumsum(d)]); wm=(cs[W:]-cs[:-W])/W; wm[:MINIMUM_WINDOW_START]=np.nan
        hi=int(np.nanargmax(wm)); res.append((did,hi,wm[hi],len(d),z['document_token_strings'],docs[did].get('language')))
    print('\n######',m,'| sd tokens 0-31:',round(np.std(early),3),'| 512+:',round(np.std(late),3))
    for x in sorted(res,key=lambda x:-x[2])[:8]:
        t=x[4]; s=x[1]
        print(f"  {x[0]} [{x[5]}] start {s}/{x[3]} {x[2]:+.3f} | before: {''.join(t[max(0,s-50):s])!r} | WINDOW: {''.join(t[s:s+W])!r} | after: {''.join(t[s+W:s+W+50])!r}")
