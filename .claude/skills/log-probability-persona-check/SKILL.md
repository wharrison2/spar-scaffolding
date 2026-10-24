---
name: log-probability-persona-check
description: How to run a log-probability persona check on corpus documents in this project, which scores how much each document makes a given AI persona (preachy, sycophantic, cruel, etc.) more likely than a generic assistant does. Use whenever planning, building, running, analyzing or writing up such a check, or adapting the preachiness check to a new persona.
---

# Log-probability persona check

Also follow `pod_sessions`, `scripts-that-call-ai-apis`, `testing-and-smoke-tests` and `writing-results`. This skill does not repeat their rules.

Method source: `plans/10-07-document-persona-projection/10-21-development/persona_projection_context.md`. Worked example (preachiness, Gemma 4 12B, 477 documents): the plan and writeup in the same folder, and runs in `data/log_probability_persona_check_runs/`.

## What it measures, and what it can't
 - Score of a document under a persona prompt: log P(token | persona) − log P(token | generic), with the generic prompt "You are an AI assistant." Only the document's tokens are scored. The system prompt, generation prompt and chat-template tokens are context
 - It is a cheap check. Persona vector projection is the robust version for axes measured often
 - **Whole-document scores did not track the persona in the preachiness check.** They failed the positive control (6 of 10 preachier rewrites moved the right way), and corpus patterns contradicted the persona (neutral tone scored above advocate tone). The likely cause is dilution: about 300 edited tokens in a 2,000-token document, while context shifts also move the remaining tokens
 - **Per-token scores on changed spans did track it:** on preachier rewrites the preachy persona rose on the inserted text in 10 of 10, while no control persona rose
 - So, by default, base conclusions on span-level outputs (extreme spans, changed spans), not whole-document means. Do not recommend corpus decisions (dropping a spec variable, filtering documents) from whole-document scores unless a positive control at the whole-document level has passed

## Adapting the code to a new persona
 - Code: `scripts/log_probability_persona_check/`. Run order and file formats: `pilot_runbook.md` and `run_folder_layout_and_script_interfaces.md` there
 - The scorer (`score_document_log_probabilities_on_pod.py`) and launcher are persona-independent. These are hard-coded to preachiness and must be generalized before reuse:
   - `log_probability_persona_check_prompt_rendering.py`: prompt file names and the condition groups `preachy` and `proportionate_welfare`
   - `analyze_preachiness_scores.py`: group names and column names
   - `make_preachiness_positive_control_rewrites.py`, `build_scoring_inputs.py`, `analyze_positive_control_changed_spans.py`: the rewrite directions `preachier` and `plainer`
 - Generalize by making the target persona, the same-values control and the rewrite directions parameters (for example a persona configuration file), with the preachiness values as the defaults so the existing runs still reproduce. Keep the existing tests passing and add tests for the new parameters
 - Put new prompt files in a persona-specific subfolder of `prompts/`, and run outputs in descriptively named folders that include the persona name

## Conditions
 - `--model-type base` (default `instruct`) scores a base model with no chat template: sequence = bos + "The following text was written by {author}, in response to this request: {generation prompt}" + blank line + document, nothing after. Persona descriptions are the same lines, wrapped as the author "an AI assistant who ..." (`prompts/base_model_*.txt`); the model type is recorded in the run configuration. Scores from the two types are not comparable
For each document, one scoring pass per system prompt:

| Condition | Count in the preachiness check | Purpose |
|---|---|---|
| No system prompt | 1 | Reference only |
| Generic, "You are an AI assistant." | 1 | Baseline subtracted from every score |
| Target persona, equivalent rewordings | 30 | Averaging over wording |
| Topic-related control personas with no stance on the target behaviour, 5 rewordings each | 5 personas, 25 prompts | Main comparison |
| Same-values control: shares the target's values but not the behaviour | 5 | Separates the behaviour from the topic |

 - The same-values control is what makes the result about the behaviour. In the preachiness check it was "cares about animal welfare, raises it in proportion to the stakes, doesn't moralize". Without it, a high score could just mean the document is about the topic
 - Personas go into `persona_system_prompt_frame.txt`, "You are an AI assistant who {description}."
 - The generation prompt states the shared in-universe facts and every spec variable of the document, in English for all documents
 - I draft every prompt file and the user reviews them before anything runs. Rewordings must be equivalent in meaning: vary wording, not intensity or scope

## Positive control (required before trusting results)
 - Rewrite about 10 documents, stratified by the spec variable of most interest, in two directions: toward the target persona and away from it. Facts, genre, language and length stay the same. The preachiness check used Claude Opus 5 at about $2.27
 - Also make a **neutral rewrite** of similar size that changes register without adding the target behaviour. The preachiness check skipped it, which left open whether any Opus edit in a moral register scores as preachy
 - Check that the "away" direction has something to remove. Plainer rewrites of documents that were already plain changed 0 to 36 tokens and tested nothing. Choose sources that show the behaviour, or drop that arm
 - Fix the pass criteria before the run (for example "at least 8 of 10 move the expected way"), at both levels:
   - whole-document score, rewrite minus source
   - changed-span score: align rewrite tokens to source tokens with `difflib.SequenceMatcher`, and compare the target-minus-control score on changed tokens against unchanged tokens and against the source tokens they replaced (`analyze_positive_control_changed_spans.py`)
 - Report the per-persona breakdown on changed spans as well: the signal should come from the target persona only

## Runs
Follow `pilot_runbook.md`. Every step marked PAID needs its own go-ahead.
 1. Rewrites: dry run, then the paid run
 2. Pilot: about 20 corpus documents (including the rewrite sources) plus the rewrites, on one pod, with `--verify-first-document-against-full-logits`. Check that the chat template keeps the system prompt, that chunked scoring matches full logits, and the throughput
 3. Go or no-go summary (`summarize_pilot_for_go_or_no_go_decision.py`), including projected full-run cost. Show it to the user before asking for the full run
 4. Full run on the same pod if it's still up (`--reuse-existing-pod`)
 - Pin the model revision for every run, and record it
 - Do not pass `--maximum-documents` to the launcher: the analysis stops if any document lacks output
 - Reference figures from the preachiness check (H100 80GB, bf16): about 10,000 scored tokens per second, about 60 million tokens for 477 documents × 62 conditions, 1 h 41 min of scoring, about $7.64 for the full run. Prefix caching does not help, because the system prompt comes first and differs across conditions

## Analysis
`analyze_preachiness_scores.py` (to be generalized) produces:
 - per-document scores per condition, and target minus pooled control
 - main test per document: two-sided Mann-Whitney U, target rewordings against pooled control rewordings. Second test: target against the same-values control. Benjamini-Hochberg correction across original documents only; rewrites are judged by the positive control
 - rewording agreement: mean pairwise Spearman across documents, intraclass correlation and leave-one-out outlier flags. If the target persona's rewordings agree less than the controls' do, or several are flagged, reword them before trusting the results
 - extreme spans: the 3 highest and 3 lowest non-overlapping 32-token windows of the per-token target-minus-generic difference, with their text
 - scores by spec variable, with an ordinary least squares regression on one-hot spec variables (leave out language, which is collinear with culture)
 - **Read a sample of the extreme spans** before making any claim about which documents carry the persona. In the preachiness check they were never read

## Known engineering problems (already fixed, don't reintroduce)
 - tar on the pod must use `--no-same-owner --no-same-permissions`; locally use `--no-mac-metadata --no-xattrs` with `COPYFILE_DISABLE=1`
 - the launcher must upload `run_configuration_*.json` along with the inputs
 - the SSH readiness check retries until its deadline, rather than relying on one timeout
 - Still open: periodic pulls sometimes fail with "file changed as we read it" because the scorer writes while tar reads. The final copy is unaffected

## Writing up
 - Report the positive control before the corpus results, and say at which level (whole document, changed spans) it passed
 - Report whole-document and span-level results separately, and do not present whole-document scores as a persona measure unless the positive control passed at that level
 - Record the model, revision, GPU, dtype, throughput and cost of every run, including failed launches
