import pandas as pd, json, glob, itertools, numpy as np
base='/Users/williamharrison/Desktop/emergent_alignment/data/audit/'
runs={'Gemma 4 12B instruct':'log_probability_persona_check_preachiness_gemma_4_12b_instruct_full_corpus_72_conditions_merged_2026-10-25_01-24/analysis_2026-10-25_01-24',
'Qwen3-32B instruct':'log_probability_persona_check_preachiness_qwen3_32b_instruct_full_corpus_2026-10-25_00-34/analysis_2026-10-25_06-50',
'Gemma 4 12B base':'log_probability_persona_check_preachiness_base_2026-10-24_17-11/analysis_2026-10-24_19-15'}
conds={}
for l in open(base+'log_probability_persona_check_preachiness_base_2026-10-24_17-11/scoring_inputs/conditions.jsonl'):
    c=json.loads(l)
    if c['author_description']: conds[c['condition_id']]=(c['persona_name'],c['author_description'].removeprefix('an AI assistant who '))
gain={};corr={}
for m,r in runs.items():
    d=pd.read_csv(glob.glob(base+r+'/secondary_not_validated_per_document_scores_*.csv')[0]); d=d[d.rewrite_direction.isna()]
    cols=[c for c in d.columns if c.startswith('score__') and '__rewording_' in c]
    w=d.number_of_tokens
    gain[m]=pd.Series({c[7:]:(d[c]*w).sum()/w.sum() for c in cols})
    rho=d[cols].rank().corr()  # Spearman
    pers={c:c[7:].split('__rewording_')[0] for c in cols}
    intra=[];inter=[];pi={}
    for a,b in itertools.combinations(cols,2):
        v=rho.loc[a,b]
        if pers[a]==pers[b]: intra.append(v); pi.setdefault(pers[a],[]).append(v)
        else: inter.append(v)
    corr[m]=(np.mean(intra),np.mean(inter),{p:np.mean(v) for p,v in pi.items()},len(intra),len(inter))
g=pd.DataFrame(gain); rk=g.rank(ascending=False).astype(int)
for m in runs:
    print('\n###',m)
    s=g[m].sort_values(ascending=False)
    for cid in list(s.index[:5])+list(s.index[-5:]):
        p,desc=conds[cid]; others=' | '.join(f"{o}: {rk.loc[cid,o]}" for o in runs if o!=m)
        print(f"| {rk.loc[cid,m]} | {desc} | {p} | {s[cid]:+.4f} | {others} |")
print('\nspearman of gains across models'); print(g.corr(method='spearman').round(2))
for m,(a,b,pi,ni,ne) in corr.items():
    print(m, 'intra',round(a,2),'inter',round(b,2),ni,ne, {k:round(v,2) for k,v in pi.items()})
