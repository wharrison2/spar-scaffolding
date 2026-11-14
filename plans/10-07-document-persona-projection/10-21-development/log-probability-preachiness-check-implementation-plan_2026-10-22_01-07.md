# Log-probability preachiness check: implementation plan

## Goal

For each of the 477 corpus documents, estimate whether the document makes a preachy, moralizing AI persona more likely, using the log-probability difference method in `persona_projection_context.md`. Use the result to decide whether the tone variable "supportive, like an animal advocate" should stay in the generation matrix.

Fixed decisions:

- Model: Gemma 4 12B, instruction-tuned, on one RunPod GPU
- Budget: under 30 dollars for everything on the pod, including the pilot
- One target persona: preachy or moralizing
- The generation prompt states all twelve spec variables plus the document's language, and is written in English for every document
- Scores are per-token averages, with the most extreme spans reported alongside them
- Significance: for each document, the distribution of target scores across rewordings is tested against the distribution across control personas and their rewordings

## Conditions

Every document is scored once under each of these system prompts, always followed by the same per-document generation prompt:

| Condition | Count |
|---|---|
| No system prompt | 1 |
| Generic: "You are an AI assistant." | 1 |
| Preachy persona, equivalent rewordings | 30 |
| Control personas (animal-related, no moral stance), 5 rewordings each | 5 personas, 25 prompts |
| Welfare-caring control that doesn't moralize | 5 |

62 conditions in total.

Score for one document under one condition: the mean over the document's tokens of log P(token | persona condition) − log P(token | generic condition). Only document tokens are scored; prompt tokens are context.

## Prompt files (drafted by me, reviewed by you before anything runs)

All in `scripts/log_probability_persona_check/prompts/` as `.txt` files:

- `in_universe_facts_shared_by_all_documents.txt`
- `document_generation_prompt_template.txt`: the "Write a document from a world where…" prompt, with slots for all twelve spec variables. The language comes from `culture`, which names it for every document. Code adds a final period to `resolution` values that lack one.
- `generic_assistant_system_prompt.txt`
- `persona_system_prompt_frame.txt`: "You are an AI assistant who {persona_description}."
- `preachy_persona_descriptions.txt`: 30 equivalent rewordings, one per line
- `control_persona_descriptions_without_moral_stance.txt`: 5 control personas, 5 rewordings each, grouped under `#` headers
- `proportionate_welfare_control_persona_descriptions.txt`: 5 rewordings of the welfare-caring persona that doesn't moralize
- `preachiness_positive_control_rewrite_instructions.txt`: instructions for the Opus 5 rewrites

That makes 62 scoring conditions per document: 30 preachy, 25 control, 5 proportionate-welfare, the generic prompt, and no system prompt.

Statistical comparisons for each document:

- preachy rewordings against the 25 pooled control rewordings (the main test)
- preachy rewordings against the 5 proportionate-welfare rewordings, kept separate, so preachiness is distinguished from welfare concern

## Code (written by Sonnet subagents)

New folder: `scripts/log_probability_persona_check/`. It reuses `run_folder_management.py`, `resumable_jsonl_records.py` and `timed_progress_reporter.py` from `scripts/shared/`.

1. **`build_scoring_inputs.py`** (local, free): renders every (document, condition) prompt from the corpus and the prompt files. Writes the full list to the run folder with file hashes, so a resumed run can check that nothing changed.
2. **`score_document_log_probabilities_on_pod.py`** (runs on the pod): loads Gemma 4 12B in bf16 with Hugging Face transformers and does forward passes only. For each (document, condition) it saves the per-token log-probabilities of the document tokens. These go to one compressed numpy file per document, with a completion record per (document, condition) appended to a JSONL file so a run can resume after interruption. Progress reports with timings come about every 3 minutes, and pod cost so far is written to a log.
3. **`launch_and_stop_runpod_scoring_pod.py`**: creates the pod with the project's RunPod key from `.env`, uploads the inputs, starts scoring, pulls results back, and terminates the pod. It enforces a hard limit on running time so cost can't pass the budget, even if the scoring script hangs.
4. **`analyze_preachiness_scores.py`** (local, free), which produces:
   - per-document mean score for each condition, and the preachy-versus-generic score
   - extreme spans: the 3 highest and 3 lowest 32-token windows of the per-token difference, with their text
   - per-document Mann-Whitney test of preachy rewordings against control rewordings, with Benjamini-Hochberg correction across documents
   - rewording agreement: how tightly each persona's rewordings agree; a persona whose rewordings scatter is reworded before its results are trusted
   - scores broken down by each spec variable, with tone first, to answer the tone question
   - figures

Tests:

- Automated tests for the bugs that would invalidate the experiment:
  - an off-by-one in which tokens count as document tokens
  - the chat template dropping the system prompt
  - conditions being mismatched across documents
  - a resumed run reusing changed prompt files
- A smoke test that runs scripts 1, 2 and 4 on three documents and three conditions with a tiny local model on CPU, at no cost. Outputs go to `data/log_probability_persona_check_runs/small_sample_checks/`.

## Runs

1. **Pilot** (about 2 dollars, needs your go-ahead): 20 documents chosen across tone values, all conditions. It checks three things:
   - that Gemma 4's chat template keeps the system prompt
   - measured throughput, to replace the estimate below
   - whether preachy rewordings agree with each other and separate from controls at all
2. **Full run** (needs a separate go-ahead): all 477 documents.

Each run gets its own folder in `data/log_probability_persona_check_runs/`.

## Cost estimate

About 2,000 Gemma tokens per document on average (Hindi and Mandarin run higher) × 477 documents × 62 conditions comes to about 60 million tokens scored, forward pass only.

On one H100 at about 3 dollars an hour, that's roughly 1 to 3 hours of scoring, so about 5 to 12 dollars including setup and model download. A pilot that shows higher throughput costs would let us cut rewordings before the full run.

Correction to what I said earlier: prompt prefix caching does not help here. The system prompt comes first and differs across conditions, so no two conditions share a prefix that includes the document.

## Additions

1. **Welfare-caring control that isn't preachy.** One extra control persona that cares about animal welfare and raises it in proportion to the stakes, without moralizing. Every document is about welfare, so without this control a high preachy score could mean only that the document discusses welfare.
2. **Positive control.** The tone variable (neutral or journalistic, supportive like an animal advocate, skeptical or displeased) describes the author's stance, not how the depicted AI behaves, so it can't confirm that the method detects preachiness. Instead, about 10 documents spread across tone values are each rewritten twice by Claude Opus 5 (`claude-opus-5`) through the Anthropic API, using the project key `ANTHROPIC_API_KEY` in `.env`. Sampling parameters are left unset. The server-side refusal fallback is on, and each rewrite records which model produced it:
   - a preachier version, where the AI moralizes more
   - a plainer version, where the AI weighs welfare without moralizing

   Facts, genre and length stay the same. The rewrites are scored in the pilot. The check passes if preachier versions score above their originals and plainer versions below. Estimated cost is 3 to 5 dollars, with usage and cost logged.

   Script: `make_preachiness_positive_control_rewrites.py`, with its prompt in `prompts/preachiness_positive_control_rewrite_instructions.txt`.

## Budget

| Item | Estimated cost |
|---|---|
| Positive-control rewrites | 3 to 5 dollars |
| Pilot | about 2 dollars |
| Full run | 5 to 12 dollars |
| Total | under 30 dollars |
