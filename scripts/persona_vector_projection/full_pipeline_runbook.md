# Persona vector projection: the whole pipeline, in order

Every command runs from the repository root. **PAID** marks a step that calls a paid API or creates a pod: each needs the user's explicit go-ahead for that step, after its dry run, and a project-specific key from `.env` (never the OS environment; confirm which key before running). Everything else is free and local. Rules for pods: `.claude/skills/pod_sessions/SKILL.md`. Rules for API scripts: `.claude/skills/scripts-that-call-ai-apis/SKILL.md`.

Detailed runbooks (options, output layouts, costs):
- Extraction (drafts, generation pod, judge, vectors): `extraction_runbook.md`, interfaces and file formats in `extraction_outputs_and_judge_interface.md`.
- Scoring pod (matched baselines, activation scoring, steering): `scoring_pod_runbook.md`.
- Depiction labels, positive-control rewrites, analysis, LEACE: `rewrites_labels_and_analysis_runbook.md`.

Plan: `plans/10-30-persona-projection/persona-vector-projection-plan_2026-10-30_23-51.md`.

## Modes

| | Pilot | Full |
|---|---|---|
| Facets extracted | non-intrusive and topic control, every question (uncapped: 3,120 answers) | cares about welfare, calibrated consideration, non-intrusive, consistent, and topic control (13,520 answers) |
| Scoring pod | 20 corpus documents plus the existing preachiness rewrites, three document formats, steering on non-intrusive | all 477 corpus documents plus the facet rewrites, one document format chosen from the pilot, steering on every facet and all four together |
| Rewrites and labels | skipped: the pilot reuses the existing preachier and plainer rewrites (10 documents) | depiction labels, preachiness rewrites for 30 documents, facet rewrites |

The pilot comes first. Its measurements (GPU throughput, judge behaviour, vector quality, positive control per document format) decide the settings of the full run. `<...>` names below are printed by the step that creates them; each folder name ends in its timestamp.

## 0. Check the code (free)

```
python3 -m pytest scripts/persona_vector_projection scripts/log_probability_persona_check scripts/shared -q
python3 scripts/persona_vector_projection/rehearse_full_pipeline_end_to_end_locally_with_tiny_model_and_mocked_clients.py
```
The rehearsal runs generation, judge (mocked), vectors, LEACE variant (stand-in eraser), rewrites (mocked), scoring pod build and stages, steering judge (mocked) and analysis on a tiny random model, in `data/persona_vector_projection_runs/small_sample_checks/`. The tests for LEACE need `pip install concept-erasure` and are skipped without it.

## 1. Re-convert the drafts (free)

Run again after any edit to a draft in `plans/10-30-persona-projection/extraction-question-drafts/` or the system-prompt and judge-prompt draft; every later step reads the latest conversion.
```
python3 scripts/persona_vector_projection/convert_drafts_into_machine_readable_inputs.py
```
Check the printed counts: 364 question records (calibrated consideration 52, cares about welfare 52, non-intrusive 52, each as 20 in half A, 20 in half B and 12 held out; consistent 104 and topic control 104, each as 52 pairs of two records) and 41 system prompts. It fails loudly on any parse or count problem.

## 2. Build the generation run folder (free)

Pilot:
```
python3 scripts/persona_vector_projection/build_extraction_run_folder.py --mode pilot
```
Full:
```
python3 scripts/persona_vector_projection/build_extraction_run_folder.py --mode full
```
Prints and creates `data/persona_vector_projection_runs/persona_vector_extraction_<mode>_..._<timestamp>/` (`<extraction run folder>`). Add `--maximum-questions-per-half 6` to the pilot for a smaller one.

## 3. Cost ceilings and dry run of the generation pod (free)

```
python3 scripts/persona_vector_projection/estimate_extraction_cost_ceilings.py
python3 scripts/persona_vector_projection/launch_and_stop_runpod_generation_pod.py --run-folder <extraction run folder> \
    --maximum-running-hours <hours> --maximum-dollars <dollars> --dry-run
```
Use the suggested launcher limits the estimate prints for the mode (pilot: 3.3 hours, 14 dollars; full: 7.7 hours, 34 dollars; both are assumptions the pilot replaces with measurements). Show the cost ceiling to the user.

## 4. PAID: generation and activation pass (one RunPod H100 session)

```
python3 scripts/persona_vector_projection/launch_and_stop_runpod_generation_pod.py --run-folder <extraction run folder> \
    --maximum-running-hours <hours> --maximum-dollars <dollars>
```
Run in a background shell. When the launcher announces the pod, tell the user its name and the cost center name so they assign it by hand; report progress every few minutes. Resume after a failure with the same command and `--run-folder`. See `extraction_runbook.md` step 5.

## 5. PAID: judge the answers (OpenAI API, `gpt-5.6-luna`)

```
python3 scripts/persona_vector_projection/judge_answers_with_gpt_5_6_luna.py --run-folder <extraction run folder> --dry-run
python3 scripts/persona_vector_projection/judge_answers_with_gpt_5_6_luna.py --run-folder <extraction run folder>
```
Full mode adds `--pair-judge-sample-size 20` to both (the consistent facet only). Rerunning skips finished requests.

## 6. Compute the vectors (free, local CPU)

```
python3 scripts/persona_vector_projection/compute_persona_vectors_from_activations_and_judgements.py \
    --run-folder <extraction run folder> --vector-set-name <name>
```
Writes `<extraction run folder>/vector_sets/<vector set name>_<timestamp>/` (`<vector folder>`): the topic-removed facet vectors, the topic vectors, the half-domain variants `persona_vector_<facet>__vector_from_half_A_<ts>.npy` and `..._B_...`, and `persona_vector_metadata_<ts>.json` with the layer choice. Check the filter counts and diagnostics in the metadata before going on.

## 7. LEACE variant (free, local; after step 6, before step 10)

Needs the topic-control activations from step 4 and the layer choice from step 6; the variant must be in the vector folder when the scoring pod is built.
```
pip install concept-erasure
python3 scripts/persona_vector_projection/fit_leace_eraser_and_effective_facet_directions.py \
    --extraction-run-folder <extraction run folder> --vector-folder <vector folder>
```
Adds `persona_vector_<facet>__leace_erased_<ts>.npy` to the vector folder (read it at the facet's layer only; details in `rewrites_labels_and_analysis_runbook.md`).

## 8. PAID: depiction labels, preachiness rewrites and facet rewrites (full mode; the pilot skips this step)

Order matters: the labels feed the rewrite selection, and the non-intrusive facet rewrites build on 30 preachiness rewrites.
```
python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py --dry-run
python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py                       # PAID, GPT-5.6 Luna
```
Hand-check the 40-document sample it writes (`depiction_label_hand_check_sample_<ts>.md`, fill in the CSV) before relying on the labels. Then the 30 preachier and plainer rewrites (Claude Opus 5, existing script):
```
python3 scripts/log_probability_persona_check/make_preachiness_positive_control_rewrites.py --number-of-documents 30 --dry-run
python3 scripts/log_probability_persona_check/make_preachiness_positive_control_rewrites.py --number-of-documents 30   # PAID
```
Then the facet rewrites, passing the labels file and the new preachiness file:
```
python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py --dry-run \
    --depiction-labels-jsonl <depiction_labels_<ts>.jsonl> --existing-preachiness-rewrites-jsonl <rewritten_documents.jsonl>
python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py \
    --depiction-labels-jsonl <depiction_labels_<ts>.jsonl> --existing-preachiness-rewrites-jsonl <rewritten_documents.jsonl>   # PAID
```
Produces `scoring_pod_extra_documents_<ts>.jsonl` (`<rewrite export file>`) in its run folder and `facet_positive_control_rewrites_<ts>.jsonl` (`<rewrite metadata file>`). Options, cost ceilings and resume commands: `rewrites_labels_and_analysis_runbook.md`.

## 9. Build the scoring pod run folder and dry-run it (free)

Pilot (default rewrites: the existing preachiness ones):
```
python3 scripts/persona_vector_projection/build_scoring_pod_run_folder.py --mode pilot \
    --vector-folder <vector folder> --dry-run
python3 scripts/persona_vector_projection/build_scoring_pod_run_folder.py --mode pilot \
    --vector-folder <vector folder>
python3 scripts/persona_vector_projection/launch_and_stop_runpod_scoring_and_steering_pod.py \
    --run-folder <scoring run folder> --maximum-running-hours 1.5 --dry-run
```
Full (after the pilot has chosen the document format and the baselines per document):
```
python3 scripts/persona_vector_projection/build_scoring_pod_run_folder.py --mode full \
    --vector-folder <vector folder> --document-formats <chosen format> --baselines-per-document <n> \
    --positive-control-rewrites-jsonl <rewrite export file> --dry-run
python3 scripts/persona_vector_projection/build_scoring_pod_run_folder.py --mode full \
    --vector-folder <vector folder> --document-formats <chosen format> --baselines-per-document <n> \
    --positive-control-rewrites-jsonl <rewrite export file>
python3 scripts/persona_vector_projection/launch_and_stop_runpod_scoring_and_steering_pod.py \
    --run-folder <scoring run folder> --maximum-running-hours <hours> --dry-run
```
The builder scores every `<facet>__<variant>` vector in the vector folder (the half-domain and LEACE variants); add `--no-vector-variants` to skip them. Dry-run output shows counts, GPU hours, ceiling and disk. Recommended limits: pilot 1.5 hours; full 4 hours (`scoring_pod_runbook.md`).

## 10. PAID: scoring pod (baselines, scoring, steering; one RunPod H100 session)

```
python3 scripts/persona_vector_projection/launch_and_stop_runpod_scoring_and_steering_pod.py \
    --run-folder <scoring run folder> --maximum-running-hours <hours>
```
Same pod rules as step 4. Resume on a new pod with the same command and run folder.

## 11. PAID: judge the steering answers (OpenAI API, `gpt-5.6-luna`)

```
python3 scripts/persona_vector_projection/judge_steering_answers_from_scoring_pod_run.py --scoring-run-folder <scoring run folder> --dry-run
python3 scripts/persona_vector_projection/judge_steering_answers_from_scoring_pod_run.py --scoring-run-folder <scoring run folder>
```
Writes `<scoring run folder>/steering_judge_outputs/`.

## 12. Analysis (free, local)

```
python3 scripts/persona_vector_projection/analyze_persona_vector_projection_scores.py \
    --scoring-run-folder <scoring run folder> --document-format <assistant_turn, or the chosen format> \
    --rewrite-metadata-jsonl <rewrite metadata file> [--depiction-labels-jsonl <depiction_labels_<ts>.jsonl>] \
    [--tokenizer-name google/gemma-4-12B-it] [--compare-neighbouring-layers]
```
Pilot only, to choose the document format: `python3 scripts/persona_vector_projection/compare_document_formats_on_positive_controls.py --scoring-run-folder <pilot scoring run folder>`. The pilot has no `--rewrite-metadata-jsonl` (the preachiness rewrites carry their direction in the document id). Span-level analyses use the token ids the pod saved; `--tokenizer-name` only adds the text of extreme spans. The split-half ranking comparison and the LEACE comparison are the `ranking_agreement_vector_variants_<ts>.csv` table. `compare_two_pipeline_versions_on_matched_specs.py` is separate (`rewrites_labels_and_analysis_runbook.md`).
