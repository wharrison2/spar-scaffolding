# Correlates Gemma 4 12B instruct and Qwen3-32B instruct per-token score differences along each document.
# Each comparison is "group A minus group B", given on the command line as A:B, for example
#   --comparison preachy:generic --comparison proportionate_welfare:animal_husbandry_expertise
# A group is "generic", "no_system_prompt", "topic_controls" (the five topic-control personas pooled) or any persona name
# (the average over its rewordings). --list-groups prints the persona names.
# Tokens are matched by character position: each model's per-token difference is averaged over blocks of characters.
# Tokens on heading-like lines (see heading_character_mask) are left out unless --include-headings is passed.
# Documents whose token strings do not rebuild the document text (split multi-byte characters) are left out.
import numpy as np, glob, json, re, random, argparse
from scipy.stats import spearmanr
base='/Users/williamharrison/Desktop/emergent_alignment/data/audit/'
runs={'gemma':'log_probability_persona_check_preachiness_gemma_4_12b_instruct_full_corpus_72_conditions_merged_2026-10-25_01-24',
'qwen':'log_probability_persona_check_preachiness_qwen3_32b_instruct_full_corpus_2026-10-25_00-34'}
topic_control_personas=('animal_husbandry_expertise','aesthetic_love_of_animals','practical_and_indifferent_to_welfare','typical_ai_view_of_animals','zoological_knowledge')
regions={'whole document':(0.0,1.0),'middle half of the document (25% to 75% of its characters)':(0.25,0.75)}
SENTENCE_END=tuple('.!?。！？…؟।۔')
LONGEST_HEADING_CHARACTERS=150
CLOSING_MARKS='"\'”’»)）」』]*_ '
METADATA_LABEL=re.compile(r'^[^\W\d][^:：\n]{0,40}[:：]')
WIDE_SCRIPT=re.compile(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]')

def heading_character_mask(content):
    # Deterministic, approximate: marks characters on lines that look like titles, headings, separators or metadata.
    # A line counts if it is a first line of at most 150 characters, a markdown heading or separator, a "Label: value" line of at most 150 characters not ending a sentence,
    # or a short line not ending a sentence that starts or ends a block of lines and does not run on into the next line
    # (hard-wrapped prose ends in a lowercase letter or comma and the next line starts in lowercase).
    mask=np.zeros(len(content),bool); position=0; lines=content.split('\n')
    for line_number,line in enumerate(lines):
        stripped=line.strip(); ending=stripped.rstrip(CLOSING_MARKS)
        previous_line=lines[line_number-1].strip() if line_number>0 else ''
        next_line=lines[line_number+1].strip() if line_number+1<len(lines) else ''
        ends_sentence=ending.endswith(SENTENCE_END)
        short_limit=40 if WIDE_SCRIPT.search(stripped) else 80
        runs_on=bool(ending) and (ending[-1].islower() or ending[-1] in ',;') and bool(next_line) and next_line[0].islower()
        stands_apart=not previous_line or not next_line
        is_heading=bool(stripped) and (
            (line_number==0 and len(stripped)<=LONGEST_HEADING_CHARACTERS)
            or stripped.startswith('#')
            or not re.search(r'\w',stripped)
            or (METADATA_LABEL.match(stripped) and len(stripped)<=LONGEST_HEADING_CHARACTERS and not ends_sentence and not runs_on)
            or (len(stripped)<=short_limit and not ends_sentence and stands_apart and not runs_on))
        if is_heading: mask[position:position+len(line)]=True
        position+=len(line)+1
    return mask

def group_log_probabilities(group,condition_ids,log_probabilities):
    members=topic_control_personas if group=='topic_controls' else (group,)
    return log_probabilities[[i for i,c in enumerate(condition_ids) if c.split('__')[0] in members]].mean(0)

parser=argparse.ArgumentParser(description='Correlate Gemma and Qwen per-token differences between condition groups, within documents.')
parser.add_argument('--comparison',action='append',metavar='GROUP_A:GROUP_B',help='score group A minus group B; repeatable. Default: preachy:generic, preachy:topic_controls, preachy:proportionate_welfare')
parser.add_argument('--include-headings',action='store_true',help='keep tokens on heading-like lines')
parser.add_argument('--block-characters',type=int,default=100,help='characters per block that tokens are averaged over (default 100)')
parser.add_argument('--show-excluded-headings',type=int,default=0,metavar='NUMBER',help='print a random sample of this many excluded heading lines')
parser.add_argument('--list-groups',action='store_true',help='print the available group names and exit')
arguments=parser.parse_args()
comparisons=[tuple(c.split(':')) for c in (arguments.comparison or ['preachy:generic','preachy:topic_controls','preachy:proportionate_welfare'])]

documents={json.loads(l)['document_id']:json.loads(l) for l in open(base+runs['gemma']+'/scoring_inputs/documents.jsonl')}
per_token_folders={model:base+run+'/scoring_outputs/per_token_log_probabilities/' for model,run in runs.items()}
first_file=sorted(glob.glob(per_token_folders['gemma']+'*.npz'))[0]
available_groups=sorted({c.split('__')[0] for c in np.load(first_file)['condition_ids']}|{'topic_controls'})
if arguments.list_groups: print('\n'.join(available_groups)); raise SystemExit
for comparison in comparisons:
    if len(comparison)!=2 or not set(comparison)<=set(available_groups): parser.error(f'bad comparison {":".join(comparison)}; groups are {", ".join(available_groups)}')
comparison_names=[f'{a} minus {b}'.replace('_',' ') for a,b in comparisons]

def per_token_differences(path):
    z=np.load(path); ids=list(z['condition_ids']); lp=z['log_probabilities']
    return list(z['document_token_strings']),{name:group_log_probabilities(a,ids,lp)-group_log_probabilities(b,ids,lp) for name,(a,b) in zip(comparison_names,comparisons)}

correlations={(region,name):[] for region in regions for name in comparison_names}
used=skipped=0
excluded_characters=total_characters=0; heading_examples=[]
for path in sorted(glob.glob(per_token_folders['gemma']+'*.npz')):
    document_id=path.split('/')[-1][:-4]
    if documents[document_id].get('rewrite_direction'): continue
    content=documents[document_id]['content']; per_model={}
    heading_mask=heading_character_mask(content); excluded_characters+=heading_mask.sum(); total_characters+=len(content)
    if arguments.show_excluded_headings and heading_mask.any():
        for line in content.split('\n'):
            start=content.find(line)
            if line.strip() and heading_mask[start:start+len(line)].all(): heading_examples.append(line.strip()[:100])
    for model in runs:
        tokens,differences=per_token_differences(per_token_folders[model]+document_id+'.npz')
        text=''.join(tokens)
        if text!=content[-len(text):] and text.strip()!=content.strip(): break
        offsets=np.cumsum([0]+[len(t) for t in tokens])+len(content)-len(text)
        blocks=(((offsets[:-1]+offsets[1:])/2)//arguments.block_characters).astype(int)
        if not arguments.include_headings:
            midpoints=np.clip(((offsets[:-1]+offsets[1:])//2).astype(int),0,len(content)-1)
            blocks=np.where(heading_mask[midpoints],-1,blocks)
        per_model[model]={name:np.array([values[blocks==b].mean() if (blocks==b).any() else np.nan for b in range(len(content)//arguments.block_characters)]) for name,values in differences.items()}
    if len(per_model)<2: skipped+=1; continue
    used+=1; number_of_blocks=len(content)//arguments.block_characters
    for region,(start_fraction,end_fraction) in regions.items():
        in_region=np.zeros(number_of_blocks,bool); in_region[int(start_fraction*number_of_blocks):int(np.ceil(end_fraction*number_of_blocks))]=True
        for name in comparison_names:
            g=per_model['gemma'][name]; q=per_model['qwen'][name]; mask=in_region&~np.isnan(g)&~np.isnan(q)
            if mask.sum()>5: correlations[(region,name)].append(spearmanr(g[mask],q[mask])[0])
print('headings included' if arguments.include_headings else 'headings excluded','| share of characters on heading-like lines',round(excluded_characters/total_characters,3))
if arguments.show_excluded_headings:
    random.seed(0); print('sample of heading-like lines:'); [print('   ',line) for line in random.sample(heading_examples,min(arguments.show_excluded_headings,len(heading_examples)))]
print('documents used',used,'| left out because token strings do not rebuild the text',skipped)
for (region,name),values in correlations.items():
    values=np.array(values)
    print(f"{region} | {name}: documents {len(values)}, median correlation {np.median(values):+.2f}, mean {values.mean():+.2f}, share of documents above zero {np.mean(values>0):.0%}")
