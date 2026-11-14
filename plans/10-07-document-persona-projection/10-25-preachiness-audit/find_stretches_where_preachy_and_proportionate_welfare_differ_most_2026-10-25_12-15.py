import numpy as np, glob, json, sys
base='/Users/williamharrison/Desktop/emergent_alignment/data/audit/'
runs={'Gemma 4 12B instruct':'log_probability_persona_check_preachiness_gemma_4_12b_instruct_full_corpus_72_conditions_merged_2026-10-25_01-24',
'Qwen3-32B instruct':'log_probability_persona_check_preachiness_qwen3_32b_instruct_full_corpus_2026-10-25_00-34'}
W=32
# document openings are noisy (little context, so the prompt dominates), so stretches must start at or after this token
MINIMUM_WINDOW_START=64
for m,r in runs.items():
    docs={json.loads(l)['document_id']:json.loads(l) for l in open(base+r+'/scoring_inputs/documents.jsonl')}
    res=[]; opening_differences=[]; late_differences=[]
    for f in sorted(glob.glob(base+r+'/scoring_outputs/per_token_log_probabilities/*.npz')):
        z=np.load(f); ids=list(z['condition_ids']); lp=z['log_probabilities']
        did=f.split('/')[-1][:-4]
        if docs[did].get('rewrite_direction'): continue
        pre=[i for i,c in enumerate(ids) if c.startswith('preachy__')]; pw=[i for i,c in enumerate(ids) if c.startswith('proportionate_welfare__')]
        d=lp[pre].mean(0)-lp[pw].mean(0)
        if len(d)<W: continue
        opening_differences.extend(d[:32]); late_differences.extend(d[512:])
        if len(d)<MINIMUM_WINDOW_START+W: continue
        cs=np.concatenate([[0],np.cumsum(d)]); wm=(cs[W:]-cs[:-W])/W
        wm[:MINIMUM_WINDOW_START]=np.nan
        wm=np.where(np.isnan(wm),np.nan,wm); hi=int(np.nanargmax(wm)); lo=int(np.nanargmin(wm))
        res.append((did,hi,wm[hi],lo,wm[lo],d.mean(),len(d),z['document_token_strings']))
    print('\n######',m,'| standard deviation of the per-token difference, tokens 0-31:',round(np.std(opening_differences),3),'| tokens 512 onward:',round(np.std(late_differences),3))
    print(m,len(res),'docs; corpus mean per-token preachy minus proportionate welfare (doc-weighted):',round(np.mean([x[5] for x in res]),4))
    for label,key,sign in [('HIGHEST (preachy above proportionate welfare)',2,-1),('LOWEST',4,1)]:
        idx=1 if key==2 else 3
        top=sorted(res,key=lambda x:sign*x[key])[:5]
        print(label)
        for x in top:
            toks=x[7]; s=x[idx]
            print(f"  {x[0]} start {s}/{x[6]} window {x[key]:+.3f} | before: {''.join(toks[max(0,s-40):s])!r} | WINDOW: {''.join(toks[s:s+W])!r} | after: {''.join(toks[s+W:s+W+40])!r}")
