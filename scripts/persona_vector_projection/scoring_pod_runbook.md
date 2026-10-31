# Scoring pod runbook: matched baselines, activation scoring, steering generations

One pod session, three stages, each selectable and resumable. Model `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Plan: `plans/10-30-persona-projection/persona-vector-projection-plan_2026-10-30_23-51.md`. Vector files and judge: `extraction_outputs_and_judge_interface.md`. Every command runs from the repository root. Steps marked **PAID** need the user's explicit go-ahead for that run; nothing else here costs money or touches an API.

## What runs on the pod

| Stage | Input | Output (in `<run folder>/scoring_pod_outputs/`) | Engine |
|---|---|---|---|
| `baselines` | `baseline_generation_jobs.jsonl`: one job per corpus document that is a source of a scored document | `matched_baseline_documents/baseline_documents_shard_<n>_<run timestamp>.jsonl` | vLLM, temperature 1, top-p 1, no top-k, length cap sized from the corpus |
| `scoring` | `documents.jsonl` (corpus documents and rewrites) plus the generated baselines | `per_document_projections/<format>/projections_<format>_shard_<n>_<run timestamp>.npz` | Hugging Face decoder only, no language-model head |
| `steering` | `steering_jobs.jsonl`: held-out questions x steering conditions | `steering_generations/answers_shard_<n>_steering_<run timestamp>.jsonl` | Hugging Face generate with a forward hook (vLLM cannot take the hook) |

Each stage runs in its own child process, so vLLM's GPU memory is freed before the Hugging Face model loads. A shard is written atomically after it is complete; a relaunch skips existing shards. Progress lines with timings go to the terminal and `<stage>_progress_log_<timestamp>.txt` about every 3 minutes.

### Matched baselines: how the dropped variables are removed

The baseline prompt is `prompts/baseline_document_generation_prompt_template_topic_variables_only.txt`, the log-probability check's `document_generation_prompt_template.txt` with the three sentences that carry stance and behaviour deleted whole (no empty slot, no stray punctuation):

```
original:  ... The AI in the document is {naming}. The intended readers are {tech_savvy}. The tone is {tone}. The tradeoff at stake is {tradeoff}. Animal welfare is {centrality}. {resolution} The AI's reasoning features {reasoning_featured}.
baseline:  ... The AI in the document is {naming}. The intended readers are {tech_savvy}. The tradeoff at stake is {tradeoff}. Animal welfare is {centrality}.
```

Kept: document type, culture (which names the language), domain, sentient category, decision scale, naming, readers (`tech_savvy`), tradeoff, centrality. Dropped: tone, resolution, reasoning featured. Tests check that the kept values appear and the dropped ones (and their sentences) do not, for a synthetic document and for all 477 corpus documents, and that the template equals the original minus exactly those three sentences. The system prompt is the generic `generic_assistant_system_prompt.txt` ("You are an AI assistant."). A rewrite shares its source document's baselines: jobs are made per source document from the source's variables.

### Document formats (scoring)

`assistant_turn`: system prompt (generic), user "Write a document." (`prompts/scoring_assistant_turn_neutral_user_message.txt`), then the document as the assistant's reply. `user_message`: system prompt, then the document as the user message. `raw_text`: beginning-of-sequence token and the document, no template. Only the document's tokens are scored; token 0 is the document's first token; tokens after the document are cut off. Per document the npz holds means over all tokens, tokens 0-63, tokens 64 on, and by position band (default 5 equal bands, `--number-of-position-bands`), at the chosen layer and its two neighbours on each side (union over facets when the layers differ), for the four facets and the topic vector (all projected onto the unit vector of that layer; facet vectors are the topic-removed ones), plus every extra variant `<facet>__<variant>` found in the vector folder (see below), plus per-token projections as one float16 array with offsets, and the document's own token ids (`per_token_token_ids_int32`, int32, same offsets), so the analysis never re-tokenises. `--skip-per-token-arrays` drops both.

### Extra vector variants

`build_scoring_pod_run_folder.py` copies into `scoring_pod_inputs/vectors/` the metadata, the topic-removed facet vectors, the topic vector and every file named `persona_vector_<facet>__<variant>_<timestamp>.npy` in `--vector-folder` (latest timestamp per name), and the pod scores them after the main vectors, in the order of `vector_names` in the run configuration. Written by the extraction half: `vector_from_half_A` and `vector_from_half_B` (the vector computation) and `leace_erased` (`fit_leace_eraser_and_effective_facet_directions.py`, run after the vectors are computed and before this builder; see `rewrites_labels_and_analysis_runbook.md`). `--vector-variants <names>` scores only those, `--no-vector-variants` none. A pilot extraction has the non-intrusive facet only: the pod then scores that facet, the topic vector and its variants, and steering uses the facets present. Variants add projections but no model passes.

### Steering

Held-out questions of each steered facet (from `steering_questions.jsonl`), system prompt = the extraction's length-only suffix, 5 samples each, at coefficients -2, -1, 0, 1, 2 (`--steering-coefficients`; 0 is the shared `unsteered` condition). The vector added is coefficient x the stored vector of that layer (not unit length), at every position (`--steering-positions last_position_each_forward_pass` is the alternative), at the facet's chosen layer. "All four together" adds the four scaled vectors, each at its own chosen layer (one layer when a shared layer was chosen). Output records have the extraction answer-record fields, so the judge reads them unchanged.

## Steps

1. Tests and rehearsal (free, local):
   ```
   python3 -m pytest scripts/persona_vector_projection/test_scoring_pod.py -q
   python3 scripts/persona_vector_projection/rehearse_scoring_pod_locally_on_tiny_model.py
   ```
   The rehearsal writes to `data/persona_vector_projection_runs/small_sample_checks/`.

2. Build the pilot run folder (free). `--dry-run` first: it prints counts, GPU hours, the cost ceiling and the requested pod disk, and writes nothing.
   ```
   python3 scripts/persona_vector_projection/build_scoring_pod_run_folder.py --mode pilot --small-sample-check \
       --vector-folder <extraction run folder>/vector_outputs --dry-run
   python3 scripts/persona_vector_projection/build_scoring_pod_run_folder.py --mode pilot --small-sample-check \
       --vector-folder <extraction run folder>/vector_outputs
   ```
   Pilot: 20 corpus documents (the 10 rewrite sources plus a tone- and language-balanced 10), their 20 preachier and plainer rewrites (`data/log_probability_persona_check_runs/positive_control_rewrites/preachiness_positive_control_rewrites_opus_5_2026-10-22_03-00/rewritten_documents.jsonl`), 4 baselines per corpus document (`--baselines-per-document`), all three formats, steering on the non-intrusive facet only. Other rewrite files (same shape as `--extra-documents-jsonl` of `build_scoring_inputs.py`) are added with `--positive-control-rewrites-jsonl` (repeatable). Without `--maximum-new-tokens` the builder reads the Gemma tokenizer (cached or downloaded, no paid call) and caps baselines at the 99th percentile of corpus document tokens rounded up to 256 (4,864 tokens; the corpus maximum is 7,061).

3. Dry-run the launcher (free; it prints the pod request, limits, commands and the estimate; the shared launcher checks that `.env` has the keys and prints only yes or no):
   ```
   python3 scripts/persona_vector_projection/launch_and_stop_runpod_scoring_and_steering_pod.py \
       --run-folder <pilot run folder> --maximum-running-hours 1.5 --dry-run
   ```

4. **PAID** pilot pod (cost ceiling below). Run in a background shell; report progress at checkpoints; tell the user the pod name and cost center name so they can assign it by hand (`pod_sessions`).
   ```
   python3 scripts/persona_vector_projection/launch_and_stop_runpod_scoring_and_steering_pod.py \
       --run-folder <pilot run folder> --maximum-running-hours 1.5
   ```
   Hard limits: the hours given, and dollars = hours x 4.50 unless `--maximum-dollars` is given. Stage selection: `--stages baselines scoring steering`, `--document-formats ...`, `--generation-backend huggingface` (if vLLM does not support Gemma 4 on the pod), `--skip-per-token-arrays`. Resume on a new pod: run the same command with the same run folder; finished shards are uploaded and skipped.

5. Judge the steering answers (local, **PAID**, GPT-5.6 Luna; dry run first). The judge's own command line reads the extraction run layout, so this thin entry point feeds it the steering answers (same judge, prompts and reading mode):
   ```
   python3 scripts/persona_vector_projection/judge_steering_answers_from_scoring_pod_run.py --scoring-run-folder <run folder> --dry-run
   python3 scripts/persona_vector_projection/judge_steering_answers_from_scoring_pod_run.py --scoring-run-folder <run folder>   # PAID
   ```
   Writes `<run folder>/steering_judge_outputs/` (results, usage and cost log, progress log; rerunning skips finished requests).
   Each answer's `facet` is the facet its question belongs to; `steering_condition_id` and `facets_and_coefficients` say how it was steered. The judge's `topic_control`-only branches (for example `version` of animal) do not apply: steering questions are never topic control.

6. After the pilot: choose the document format from the positive control, fix the number of baselines per document from the pilot's variance, then build the full run (below), dry-run, and ask for the go-ahead.
   ```
   python3 scripts/persona_vector_projection/build_scoring_pod_run_folder.py --mode full --document-formats <chosen format> \
       --vector-folder <extraction run folder>/vector_outputs --baselines-per-document <n> [--dry-run]
   python3 scripts/persona_vector_projection/launch_and_stop_runpod_scoring_and_steering_pod.py --run-folder <full run folder> \
       --maximum-running-hours <hours> --dry-run
   ```
   **PAID**: the same command without `--dry-run`.

## GPU hours and cost ceilings

From `estimate_scoring_pod_gpu_hours_and_cost.py` (also printed by the build dry-run and the launcher dry-run). The throughputs are unmeasured assumptions, deliberately conservative (vLLM 2,000 generated tokens per second, scoring 4,000 tokens per second, Hugging Face steering 0.045 seconds per decoding step for 64 sequences, 15 minutes of pod setup, 5 minutes of model load per stage); replace them with the pilot's measured rates. Price $4.50 per hour on an H100. Corpus documents average 2,105 tokens.

| Run | Estimated GPU hours (cost) | Recommended hard limit = ceiling |
|---|---|---|
| Pilot (20 + 20 documents, 80 baselines, 3 formats, 300 steering answers) | about 0.6 hours ($2.9) | 1.5 hours, $6.75 |
| Full (477 documents, 1,908 baselines, 1 format, 2,700 steering answers over 4 facets and all four together) | about 2.0 hours ($8.8) | 4 hours, $18.00 (the limit is twice the estimate; a tighter limit such as 3 hours, $13.50 can be given) |

Requested pod disk: 40 GB container + 80 GB volume at `/workspace` (model about 24 GB, vLLM and pip caches; outputs are under 1 GB: about 300 MB of fp16 per-token arrays for the full run). Pulled outputs stay small; the launcher stops pulling if the local disk has under 2 GB free.

## Unresolved before a paid run

- The vLLM pin (`vllm==0.32.0`, copied from the extraction half's requirements) is unverified for Gemma 4 and for transformers 5.x on the pod image. `--generation-backend huggingface` is the fallback, and is much slower for baselines.
- The vLLM baseline path and the real-model Hugging Face steering path have not been run (only the tiny-model rehearsal with the Hugging Face backend for baselines). The first pod minutes are the first real test.
- Steering coefficients are applied to the stored (not unit) vectors; whether -2 to 2 is the right range depends on the vector norms, which the extraction metadata reports per layer. Check them before the full run.
- Whether the real Gemma 4 generation stops at `<turn|>` in the Hugging Face steering path rests on the extraction half's backend (`HuggingFaceGenerationBackend`), used unchanged.
