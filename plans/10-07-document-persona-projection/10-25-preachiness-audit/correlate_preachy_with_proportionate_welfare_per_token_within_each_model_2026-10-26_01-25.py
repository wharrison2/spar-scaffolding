import numpy as np, glob, sys, importlib.util, contextlib, io
from scipy.stats import pearsonr
p='/Users/williamharrison/Desktop/emergent_alignment/plans/10-07-document-persona-projection/10-25-preachiness-audit/correlate_gemma_and_qwen_per_token_differences_within_documents_2026-10-26_00-56.py'
sys.argv=[p,'--list-groups']; spec=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    try: spec.loader.exec_module(m)
    except SystemExit: pass
for model in m.runs:
    a=[];b=[];per=[];n=0
    for path in sorted(glob.glob(m.per_token_folders[model]+'*.npz')):
        did=path.split('/')[-1][:-4]
        if m.documents[did].get('rewrite_direction'): continue
        z=np.load(path); ids=list(z['condition_ids']); lp=z['log_probabilities']; toks=list(z['document_token_strings']); content=m.documents[did]['content']
        text=''.join(toks); keep=np.ones(len(toks),bool)
        if text==content[-len(text):] or text.strip()==content.strip():
            off=np.cumsum([0]+[len(t) for t in toks])+len(content)-len(text)
            keep=~m.heading_character_mask(content)[np.clip(((off[:-1]+off[1:])//2).astype(int),0,len(content)-1)]
        else: keep[:64]=False  # no character alignment: drop the opening instead
        gen=lp[ids.index('generic')]
        x=(m.group_log_probabilities('preachy',ids,lp)-gen)[keep]; y=(m.group_log_probabilities('proportionate_welfare',ids,lp)-gen)[keep]
        a.extend(x); b.extend(y); per.append(pearsonr(x,y)[0]); n+=1
    r=pearsonr(a,b)[0]; md=np.median(per)
    print(f'{model}: {n} documents, {len(a)} tokens | pooled r {r:+.3f}, r^2 {r*r:.3f} | median within-document r {md:+.3f}, squared {md*md:.3f}')
