# Persona vector extraction runbook

Run every command from the repository root. Steps marked PAID need the user's explicit go-ahead for that step, after the dry run. All other steps are free and local. Interfaces and output formats: `extraction_outputs_and_judge_interface.md`. Plan: `plans/10-30-persona-projection/persona-vector-projection-plan_2026-10-30_23-51.md`.

Pilot mode: non-intrusive facet and topic control. Full mode: four facets and topic control. `<run folder>` is printed by step 2. `<limits>` come from step 3.

## Memory and disk rules built in

- Pod activation pass: no language-model head is ever computed (hooks on the embedding and each decoder layer, run through `model.base_model`; 262k-vocabulary logits are never built); float32 accumulation on the fly; batch size `--activation-batch-size`, default 8, passed through the launcher.
- Local machine (16 GB RAM, about 44 GB free): vector computation streams one activation shard at a time and keeps float32 sums for one facet's questions only; a test (`test_peak_memory_scales_with_one_facet_not_the_whole_run`) asserts the peak is under half of the stored activations and does not grow with the number of facets. A full run stores about 5.1 GB of fp16 activations.
- Pod disk: the launcher requests a container disk of 80 GB and a volume of 120 GB (weights about 24 GB, vLLM and Hugging Face caches, about 5 GB of outputs). The dry run prints the request.
- Local free disk: before every pull the launcher checks free space and refuses to pull under 10 GB, printing a clear message on each attempt. The pod keeps running under its hard limits meanwhile. If the final pull is blocked, the pod waits (up to `--failure-wait-minutes`, and the hours and dollar limits) for you to free space and create `pod_instruction_retry_scorer.txt` in the run folder, then pulls and verifies.

## 1. Convert the drafts (free)

```
python3 scripts/persona_vector_projection/convert_drafts_into_machine_readable_inputs.py
```
Reads the five question drafts and the system-prompt and judge-prompt draft in `plans/10-30-persona-projection/`. Fails loudly on a parse or count problem. Writes `data/persona_vector_projection_runs/converted_inputs/converted_inputs_<timestamp>/` (system prompts, questions, facet definitions, shared suffixes, conversion report) and the judge prompt templates `scripts/persona_vector_projection/prompts/judge_*.txt`. Re-run after editing a draft.

## 2. Build the run folder (free)

Pilot (add `--maximum-questions-per-half 6` for a smaller pilot):
```
python3 scripts/persona_vector_projection/build_extraction_run_folder.py --mode pilot --maximum-questions-per-half 6
```
Full:
```
python3 scripts/persona_vector_projection/build_extraction_run_folder.py --mode full
```
Produces `data/persona_vector_projection_runs/persona_vector_extraction_<mode>_..._gemma_4_12b_it_<timestamp>/` with `generation_inputs/` (jobs, system prompts, questions, facet definitions) and `run_configuration_<timestamp>.json` (model `google/gemma-4-12b-it`, revision pinned to `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, temperature 1, top-p 1, 1,000 new tokens, 5 samples, hashes of all inputs and prompt files). The model needs no Hugging Face token.

## 3. Cost ceilings (free)

```
python3 scripts/persona_vector_projection/estimate_extraction_cost_ceilings.py --pilot-maximum-questions-per-half 6
```
Prints, for pilot and full mode, answers, GPU hours at 4.50 dollars per hour (expected and ceiling), judge calls, tokens and dollars, and suggested launcher limits. Counts are exact; GPU throughput and setup times are assumptions (options in `--help`) that the pilot replaces with measurements. Output is also saved in `small_sample_checks/cost_ceiling_estimates_<timestamp>/`.

## 4. Pod dry run (free)

```
python3 scripts/persona_vector_projection/launch_and_stop_runpod_generation_pod.py --run-folder <run folder> \
    --maximum-running-hours <hours> --maximum-dollars <dollars> --dry-run
```
Prints the pod request, requested disk, every command that would run on the pod and the cost ceiling. No RunPod call.

## 5. Generation and activation pass (PAID: one RunPod H100 session)

```
python3 scripts/persona_vector_projection/launch_and_stop_runpod_generation_pod.py --run-folder <run folder> \
    --maximum-running-hours <hours> --maximum-dollars <dollars>
```
Follow `.claude/skills/pod_sessions/SKILL.md`: run it in a background shell, assign the pod to the cost center by hand when the launcher announces it (`cost_center_assignment_confirmed_by_user.txt` in the run folder, or type `assigned`), report progress every few minutes. On the pod: `generate_answers_and_save_activations_on_pod.py` runs generation (vLLM; `--generation-backend huggingface` in the run configuration is the fallback) and then, in a second process, the activation pass. Outputs are copied back every 3 minutes and in full, verified, before the pod is deleted.

Produces in the run folder: `generation_and_activation_outputs/answers/answers_shard_*` (answers), `.../answer_token_mean_activations/activations_shard_*.npz` (fp16 per-layer means), `chat_template_system_role_handling_<ts>.json`, `generation_environment_<ts>.json`, `activation_environment_<ts>.json`, progress logs, `pod_cost_log_<ts>.json`, `pod_progress_log_<ts>.txt`, `pod_<id>_logs_<ts>/`.

Resume on a new pod after a failure or limit: run the same command again with the same `--run-folder`; finished shards are uploaded with the run and skipped. Options passed to the pod script: `--stage generation|activations|both`, `--activation-batch-size`, `--maximum-jobs`.

Before the first paid run: confirm that the vLLM release chosen supports Gemma 4 (see the comment in `requirements_for_extraction_pod.txt`); otherwise use `--generation-backend huggingface` in step 2.

## 6. Judge dry run (free) and judging (PAID: OpenAI API, gpt-5.6-luna)

```
python3 scripts/persona_vector_projection/judge_answers_with_gpt_5_6_luna.py --run-folder <run folder> --dry-run
python3 scripts/persona_vector_projection/judge_answers_with_gpt_5_6_luna.py --run-folder <run folder> [--pair-judge-sample-size 20]
```
The dry run prints call counts by kind, estimated input tokens, the output-token ceiling and the dollar ceiling from the answers actually generated. The real run uses `OPENAI_API_KEY` from the project `.env` only (confirm it is the project key first), writes `judge_outputs/judge_results_<ts>.jsonl`, `judge_usage_and_cost_log_<ts>.json` (tokens, cache rate, cost, per kind) and a progress log; rerunning skips finished requests. Reading mode: `bare_number` by default (OpenAI's model page does not list logprobs for gpt-5.6-luna; reasoning effort `none`, temperature 1); `--reading-mode logprobs` reads first-token log probabilities as Chen et al. do, and should be tried on one call first. `--pair-judge-sample-size` judges consistent pairs side by side and only applies to a run that has the consistent facet (full mode).

## 7. Compute the vectors (free, local CPU)

```
python3 scripts/persona_vector_projection/compute_persona_vectors_from_activations_and_judgements.py \
    --run-folder <run folder> --vector-set-name <name>
```
Thresholds are options (`--positive-score-minimum 65 --negative-score-maximum 35 --coherence-minimum 50 --animal-share-maximum-difference 15`). Writes `vector_sets/<vector set name>_<timestamp>/` (a new folder per run; a second run with the same name in the same minute fails rather than overwriting) (vectors, topic vectors, `persona_vector_metadata_<ts>.json` with layer choice, filter counts and diagnostics). Also writes the half-domain variants `persona_vector_<facet>__vector_from_half_A_<ts>.npy` and `..._half_B_...` (each topic-removed), which the scoring pod scores as `<facet>__vector_from_half_A` and `_B` for the split-half comparison of document rankings. Check the filter counts and the animal-share drops per negative tag in the metadata before trusting the vectors.

The LEACE variant is produced by a separate step, after this one and before the scoring pod is built: `fit_leace_eraser_and_effective_facet_directions.py` (see `rewrites_labels_and_analysis_runbook.md`). The whole order of steps is in `full_pipeline_runbook.md`.

## Tests and rehearsal (free)

```
python3 -m pytest scripts/persona_vector_projection scripts/log_probability_persona_check scripts/shared -q
python3 scripts/persona_vector_projection/rehearse_extraction_chain_locally_with_tiny_model_and_mocked_judge.py
```
The rehearsal runs the whole chain on a tiny random Gemma 4 with the real Gemma 4 tokenizer and chat template and a mocked judge; output in `data/persona_vector_projection_runs/small_sample_checks/rehearsal_tiny_model_mocked_judge_<timestamp>/`.
