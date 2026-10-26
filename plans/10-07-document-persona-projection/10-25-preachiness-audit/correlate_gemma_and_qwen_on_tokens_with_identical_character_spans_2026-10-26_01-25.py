import numpy as np, glob, sys, importlib.util, contextlib, io
from scipy.stats import pearsonr, spearmanr
p='/Users/williamharrison/Desktop/emergent_alignment/plans/10-07-document-persona-projection/10-25-preachiness-audit/correlate_gemma_and_qwen_per_token_differences_within_documents_2026-10-26_00-56.py'
sys.argv=[p,'--list-groups']; spec=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    try: spec.loader.exec_module(m)
    except SystemExit: pass
comparisons=[('preachy','generic'),('preachy','topic_controls'),('preachy','proportionate_welfare'),('proportionate_welfare','generic'),('practical_and_indifferent_to_welfare','welfare_care_without_preachiness')]
pooled={c:([],[]) for c in comparisons}; per_doc={c:[] for c in comparisons}; matched=total=0; used=0
for path in sorted(glob.glob(m.per_token_folders['gemma']+'*.npz')):
    did=path.split('/')[-1][:-4]
    if m.documents[did].get('rewrite_direction'): continue
    content=m.documents[did]['content']; spans={}; vals={}
    for model in m.runs:
        z=np.load(m.per_token_folders[model]+did+'.npz'); ids=list(z['condition_ids']); lp=z['log_probabilities']; toks=list(z['document_token_strings'])
        text=''.join(toks)
        if text!=content[-len(text):] and text.strip()!=content.strip(): break
        off=np.cumsum([0]+[len(t) for t in toks])+len(content)-len(text)
        spans[model]={(int(a),int(b)):i for i,(a,b) in enumerate(zip(off[:-1],off[1:]))}
        vals[model]={c:m.group_log_probabilities(c[0],ids,lp)-m.group_log_probabilities(c[1],ids,lp) for c in comparisons}
    if len(spans)<2: continue
    used+=1
    heading=m.heading_character_mask(content)
    common=[s for s in spans['gemma'] if s in spans['qwen'] and s[1]>s[0] and not heading[min(s[0],len(content)-1)]]
    matched+=len(common); total+=len(spans['gemma'])
    gi=[spans['gemma'][s] for s in common]; qi=[spans['qwen'][s] for s in common]
    for c in comparisons:
        g=vals['gemma'][c][gi]; q=vals['qwen'][c][qi]
        pooled[c][0].extend(g); pooled[c][1].extend(q)
        if len(common)>20: per_doc[c].append(pearsonr(g,q)[0])
print(f'documents {used}; tokens with identical character spans in both models, headings excluded: {matched} of {total} Gemma tokens ({matched/total:.0%})')
for c in comparisons:
    g,q=map(np.array,pooled[c]); r=pearsonr(g,q)[0]; rs=spearmanr(g,q)[0]; md=np.median(per_doc[c])
    print(f'{c[0]} minus {c[1]}: pooled Pearson r {r:+.3f}, r^2 {r*r:.4f} | pooled Spearman {rs:+.3f} | median within-document Pearson {md:+.3f}, its square {md*md:.4f}')
