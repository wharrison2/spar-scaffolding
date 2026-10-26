# Correlates each pair of personas within one model: for each document, each persona's per-token score minus a baseline
# group (default generic), averaged over blocks of characters, correlated between the two personas along the document.
# Reports the median per-document correlation for every pair. Reuses the heading filter and group averaging from
# correlate_gemma_and_qwen_per_token_differences_within_documents_2026-10-26_00-56.py.
import numpy as np, glob, json, argparse, itertools, importlib.util, sys
from scipy.stats import spearmanr
here=__file__.rsplit('/',1)[0]
sys.argv_saved=sys.argv; sys.argv=[sys.argv[0],'--list-groups']
specification=importlib.util.spec_from_file_location('cross_model',here+'/correlate_gemma_and_qwen_per_token_differences_within_documents_2026-10-26_00-56.py')
cross_model=importlib.util.module_from_spec(specification)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    try: specification.loader.exec_module(cross_model)
    except SystemExit: pass
sys.argv=sys.argv_saved
parser=argparse.ArgumentParser(description='Correlate personas with each other within one model.')
parser.add_argument('--model',choices=list(cross_model.runs),required=True)
parser.add_argument('--baseline',default='generic',help='group subtracted from every persona (default generic)')
parser.add_argument('--groups',nargs='+',help='groups to compare (default: every persona, plus no_system_prompt)')
parser.add_argument('--include-headings',action='store_true')
parser.add_argument('--block-characters',type=int,default=100)
arguments=parser.parse_args()
groups=arguments.groups or [g for g in cross_model.available_groups if g not in (arguments.baseline,'topic_controls')]
folder=cross_model.per_token_folders[arguments.model]
pair_correlations={pair:[] for pair in itertools.combinations(groups,2)}; used=skipped=0
for path in sorted(glob.glob(folder+'*.npz')):
    document_id=path.split('/')[-1][:-4]
    if cross_model.documents[document_id].get('rewrite_direction'): continue
    content=cross_model.documents[document_id]['content']
    z=np.load(path); ids=list(z['condition_ids']); lp=z['log_probabilities']; tokens=list(z['document_token_strings'])
    text=''.join(tokens)
    if text!=content[-len(text):] and text.strip()!=content.strip(): skipped+=1; continue
    used+=1
    offsets=np.cumsum([0]+[len(t) for t in tokens])+len(content)-len(text)
    blocks=(((offsets[:-1]+offsets[1:])/2)//arguments.block_characters).astype(int)
    if not arguments.include_headings:
        midpoints=np.clip(((offsets[:-1]+offsets[1:])//2).astype(int),0,len(content)-1)
        blocks=np.where(cross_model.heading_character_mask(content)[midpoints],-1,blocks)
    baseline=cross_model.group_log_probabilities(arguments.baseline,ids,lp)
    number_of_blocks=len(content)//arguments.block_characters
    series={g:np.array([v[blocks==b].mean() if (blocks==b).any() else np.nan for b in range(number_of_blocks)]) for g in groups for v in [cross_model.group_log_probabilities(g,ids,lp)-baseline]}
    for a,b in pair_correlations:
        mask=~np.isnan(series[a])&~np.isnan(series[b])
        if mask.sum()>5: pair_correlations[(a,b)].append(spearmanr(series[a][mask],series[b][mask])[0])
print(f'model {arguments.model} | baseline {arguments.baseline} | documents used {used}, left out {skipped} | headings {"included" if arguments.include_headings else "excluded"}')
for (a,b),values in sorted(pair_correlations.items(),key=lambda item:-np.median(item[1])):
    print(f'{np.median(values):+.2f}\t{a} | {b}')
