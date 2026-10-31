# Positive-control rewrites, depiction labels and analysis: runbook

Everything below is run from the repository root. Steps marked PAID need their own go-ahead after the dry run. Scripts that call an API read the key from the project `.env` only. Nothing in this folder has been run against an API.

## Order

1. Depiction labels (PAID, under about $12 at the ceiling). Needed first because the rewrite selection uses them.
2. Positive-control rewrites (PAID).
3. Scoring on the pod (`build_scoring_pod_run_folder.py` with `--positive-control-rewrites-jsonl <export file>`, see the pod runbook).
4. Analysis (free, local).

## 1. Depiction labels

```
python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py --dry-run
python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py            # PAID
python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py --resume-folder <folder>   # PAID, resume
python3 scripts/persona_vector_projection/label_depiction_of_facets_in_corpus_documents.py --export-hand-check-sample-only --resume-folder <folder>
```

- Model `gpt-5.6-luna`, temperature 1, reasoning effort `medium` (`--reasoning-effort`), one call per document covering the four facets plus the kinds of animal discussed. Instruction: `prompts/depiction_label_instructions_draft.txt` (draft, for review).
- Dry-run ceiling (477 documents): worst case $11.53 (two attempts per document, 8,000 reasoning tokens each, no caching); expected about $1.59. The paid run stops sending calls at `--maximum-cost-dollars` (default: the ceiling).
- Output folder `data/persona_vector_projection_runs/depiction_labels_for_four_facets_gpt_5_6_luna_<timestamp>/`: `depiction_labels_<timestamp>.jsonl` (labels, one sentence of evidence per facet, kinds of animal), failures, `usage_and_cost_log_<timestamp>.json` (tokens including reasoning, cache rate, cost), `progress_log_<timestamp>.txt`, and the hand-check sample: `depiction_label_hand_check_sample_<timestamp>.md` (40 documents stratified over facet and label, full text) with `depiction_label_hand_check_sheet_<timestamp>.csv` (fill in `hand_label`).
- Hand-check the 40 before relying on the labels in the analysis.

## 2. Positive-control rewrites

```
python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py --dry-run --depiction-labels-jsonl <depiction_labels_<ts>.jsonl>
python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py --depiction-labels-jsonl <...>   # PAID
python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py --resume-folder <folder>        # PAID, resume
python3 scripts/persona_vector_projection/make_facet_positive_control_rewrites.py --export-only --resume-folder <folder>
```

- Claude Opus 5, same request settings as the preachiness rewrites (no sampling parameters, adaptive thinking, server-side refusal fallback), imported from `scripts/log_probability_persona_check/make_preachiness_positive_control_rewrites.py`.
- 30 documents per facet for cares about welfare, calibrated consideration and consistent, each rewritten toward the facet, away from it, and same-size with the facet untouched. Neutral rewrites start from the mean number of changed words in the document's own toward and away rewrites (per document), and are flagged if their size is off that target by more than `--size-match-tolerance` (default 50%).
- Non-intrusive reuses the existing plainer (toward) and preachier (away) rewrites (`--existing-preachiness-rewrites-jsonl`, repeatable; default the 2026-10-22 run, 10 documents) and gets same-size neutral rewrites for the same documents. For 30 documents, first make 30 preachier and plainer rewrites with the existing script and pass its file.
- Selection (`select_documents_with_room_to_move_for_facet.py`): spec-based exclusions (minor-detail centrality, the no-welfare resolution), the depiction label must be "shows the trait" so the away rewrite has something to remove (relaxed with a printed warning if too few qualify), preference for evidence-uncertain sentient categories (calibrated) and for documents discussing more than one kind of animal (consistent, from the labels), stratified by language with a floor of one document for languages with at least 25 eligible documents. Override with `--selected-documents-json`.
- Minimum changed tokens: `--minimum-changed-units` (default 60 words, scaled by 1.5 for Chinese and Japanese, counted in characters), by sequence diff against the source. Below the minimum: regenerated up to `--maximum-regenerations` (default 2) with a note appended, or only flagged with `--no-regeneration`. Flagged rewrites stay in the records and are left out of the export unless `--include-flagged-in-export`. `--tokenizer-name` counts in a model's tokens (local Hugging Face cache only). The analysis applies its own minimum on Gemma tokens (`minimum_changed_tokens_for_inclusion` in the criteria file).
- Instruction drafts for review: `scripts/persona_vector_projection/prompts/positive_control_rewrite_instruction__<facet>__<direction>__draft.txt` for the three facets and directions `toward_facet`, `away_from_facet`, `facet_neutral`, plus `positive_control_rewrite_instruction__non_intrusive__facet_neutral__draft.txt` and `positive_control_rewrite_retry_note_below_minimum_changes__draft.txt`.
- Dry-run ceiling (spec-only selection, 280 calls): worst case $590.23 (three attempts per call, 24,000 thinking tokens each); default stop cap $141.92 (`--maximum-cost-dollars`, 1.5 times the cautious expected cost $94.61); scaled from the $2.27 for 20 calls of the preachiness run, about $38. Dry runs write to `small_sample_checks/` and also the rendered requests.
- Output folder `data/persona_vector_projection_runs/facet_positive_control_rewrites_opus_5_<timestamp>/`: `facet_positive_control_rewrites_<timestamp>.jsonl` (every rewrite with changed counts, attempts, flags), `rewrite_failures_*`, `per_call_usage_and_cost_*`, `usage_and_cost_log_*` (input, output, cache read and write tokens, cache rate, cost), progress log, and `scoring_pod_extra_documents_<timestamp>.jsonl` (the export for the pod: `document_id` `<source>__<facet>__<direction>`, `source_document_id`, `rewrite_direction`, `content`, `language`, `variables`, plus `facet`).
- `build_scoring_inputs.py::load_extra_documents`, which `build_scoring_pod_run_folder.py` calls, accepts the rewrite directions `preachier`, `plainer`, `toward_facet`, `away_from_facet` and `facet_neutral`, so the export loads as it is. The analysis needs only the document id and the pod's `documents.jsonl`; it reads facet and direction from the id, or from `--rewrite-metadata-jsonl`.

## 3. Analysis (local)

```
python3 scripts/persona_vector_projection/analyze_persona_vector_projection_scores.py \
    --scoring-run-folder <scoring run> --document-format assistant_turn \
    --depiction-labels-jsonl <depiction_labels_<ts>.jsonl> --rewrite-metadata-jsonl <facet_positive_control_rewrites_<ts>.jsonl> \
    --tokenizer-name google/gemma-4-12b-it [--compare-neighbouring-layers] [--primary-score after_opening]
python3 scripts/persona_vector_projection/compare_document_formats_on_positive_controls.py --scoring-run-folder <pilot run> --tokenizer-name google/gemma-4-12b-it
python3 scripts/persona_vector_projection/compare_two_pipeline_versions_on_matched_specs.py \
    --first-scoring-run-folder <old> --second-scoring-run-folder <new> [--pairs-jsonl pairs.jsonl] [--baseline-mode shared_first|own]
python3 scripts/persona_vector_projection/fit_leace_eraser_and_effective_facet_directions.py \
    --extraction-run-folder <extraction run> --vector-folder <extraction run>/vector_outputs
```

- Input layout: that of `run_scoring_pod_stages.py` (read only in `load_persona_projection_scoring_outputs.py`). The pod saves each document's token ids (`per_token_token_ids_int32`, aligned with the per-token projections), so the span-level analyses (changed spans) need no tokenizer. `--tokenizer-name` (local tokenizer cache) is only needed to print the text of extreme spans, and to re-tokenise runs from older pods that saved no ids (with the pod's own tokenisation function; it stops if a token count differs from what the pod scored). Without ids and without `--tokenizer-name` the span-level analyses are skipped.
- Score: document mean projection minus the mean of its own matched baselines (a rewrite uses its source's baselines), standardised within language from the corpus documents of that language (a language with fewer than 5 corpus documents uses the all-language statistics, flagged). Facets are never combined.
- Outputs in `<scoring run>/analysis_<name>_<timestamp>/`: document scores; distributions by facet and language; scores by every spec variable with 95% intervals, Welch tests and Benjamini-Hochberg q values, and values with fewer than 15 documents marked and unranked; scores by depiction label; by position band, opening and the rest; extreme spans (top and bottom 32-token windows, openings listed apart, with context, as CSV and as a Markdown file for hand reading); positive control at whole-document and changed-span level with standard errors clustered by source document, contrasts against the same-size neutral rewrites, shift on every facet, by language, and pass or fail against `positive_control_pass_criteria_draft.json` (draft; fix before the first run); cross-facet correlation of document scores; ranking agreement for the baseline split half, for vector variants and for neighbouring layers; figures; `analysis_summary_<timestamp>.md`.
- Variant columns: any scored vector named `<facet>__<variant>` is compared with the main facet column. `__vector_from_half_A` and `__vector_from_half_B` are written by the vector computation (`compute_persona_vectors_from_activations_and_judgements.py`: the facet vector from the questions of each half of the domains, topic removed) and `__leace_erased` by the LEACE script below. The scoring pod builder scores every such file it finds in `--vector-folder` (`--vector-variants` picks some, `--no-vector-variants` none), so the split-half and LEACE comparisons need nothing more than running those two steps before the pod is built.
- LEACE: `fit_leace_eraser_and_effective_facet_directions.py` fits the eraser on the topic-control answers' activations (animal against non-animal) at each facet's layer and saves the effective direction `A^T v` of each facet vector; projecting onto it equals projecting erased activations up to a constant that cancels in rankings and baseline subtraction. Needs `pip install concept-erasure` (MIT, in `references.md`). It also writes `persona_vector_<facet>__leace_erased_<timestamp>.npy` into `--vector-folder` (shape `(layers, hidden)` like the main vectors: the facet's own layer holds the erased direction, every other layer the unerased vector, so read the comparison at the facet's layer only; `--do-not-write-variant-into-vector-folder` turns this off). When to run it: after the topic-control activations exist (the generation pod's activation pass has been pulled) and the vectors are computed (it needs the layer choice), and before `build_scoring_pod_run_folder.py`, because the variant must be in the vector folder when the pod is built. A variant added after the pod has run needs a new scoring run.

## Tests

```
python3 -m pytest scripts/persona_vector_projection/test_persona_projection_analysis_statistics.py \
    scripts/persona_vector_projection/test_facet_positive_control_rewrites.py \
    scripts/persona_vector_projection/test_depiction_labels_and_analysis_pipeline.py
python3 scripts/persona_vector_projection/smoke_test_analysis_on_synthetic_scoring_run.py   # writes to small_sample_checks/
```

The LEACE tests are skipped unless `concept_erasure` is importable.
