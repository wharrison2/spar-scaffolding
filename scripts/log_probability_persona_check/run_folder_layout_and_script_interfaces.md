# Log-probability persona check: run folder layout and script interfaces

Plan: `plans/10-07-document-persona-projection/10-21-development/log-probability-preachiness-check-implementation-plan_2026-10-22_01-07.md`

All scripts live in `scripts/log_probability_persona_check/` and are run from the repository root. Shared helpers come from `scripts/shared/` (`run_folder_management.py`, `resumable_jsonl_records.py`, `timed_progress_reporter.py`, `api_usage_and_cost_tracking.py`).

## Pipeline

```
make_preachiness_positive_control_rewrites.py   (Anthropic API, claude-opus-5)  -> rewritten documents JSONL
build_scoring_inputs.py                          (local, free)                   -> run folder with scoring_inputs/
launch_and_stop_runpod_scoring_pod.py            (RunPod, paid)                  -> runs the scorer on a pod, pulls scoring_outputs/ back
score_document_log_probabilities_on_pod.py       (on the pod, GPU)               -> scoring_outputs/
analyze_preachiness_scores.py                    (local, free)                   -> analysis_<timestamp>/ inside the run folder
```

## Locations

- Full runs: `data/log_probability_persona_check_runs/<run_name>_<YYYY-MM-DD_HH-MM>/`
- Smoke tests, the pilot's cost estimates and other small-sample checks: `data/log_probability_persona_check_runs/small_sample_checks/<run_name>_<YYYY-MM-DD_HH-MM>/`
- Positive-control rewrites: `data/log_probability_persona_check_runs/positive_control_rewrites/<run_name>_<YYYY-MM-DD_HH-MM>/`

## Rewritten documents file

`.../positive_control_rewrites/<run>/rewritten_documents.jsonl`, one record per rewrite:

```json
{"document_id": "matrix_000005__preachier", "source_document_id": "matrix_000005",
 "rewrite_direction": "preachier", "content": "...", "language": "English",
 "variables": {...copied unchanged from the source document...},
 "rewrite_model_requested": "claude-opus-5", "rewrite_model_used": "claude-opus-5",
 "source_word_count": 1240, "rewrite_word_count": 1260}
```

`rewrite_direction` is `"preachier"` or `"plainer"`.

## Run folder written by build_scoring_inputs.py

```
<run folder>/
  run_configuration_<timestamp>.json      # model name, model type, document selection, SHA-256 of every prompt file and input file
  scoring_inputs/
    documents.jsonl                       # one per document to score
    conditions.jsonl                      # one per system-prompt condition (62)
```

Run configuration fields that matter downstream:

- `model_type`: `"instruct"` (default; instruction-tuned model scored through its chat template) or `"base"` (pretrained model with no chat template, scored with a plain-text author preamble). Chosen with `build_scoring_inputs.py --model-type`; it also picks the default model name (`google/gemma-4-12b-it` for instruct, `google/gemma-4-12b-pt` for base, an id that has not been verified; `--model-name` overrides). The scorer reads `model_type` from the configuration, so it is not passed twice. A configuration without the field means `instruct`. Resuming with a different model type is refused.
- `prompt_file_sha256`: for `base` runs it also holds the four `base_model_*.txt` prompt files; instruct runs do not, so their configurations are unchanged.

`documents.jsonl` record:

```json
{"document_id": "matrix_000005", "source_document_id": "matrix_000005", "rewrite_direction": null,
 "content": "...", "language": "English", "variables": {...}, "generation_prompt": "Write a document from a world where ..."}
```

Original corpus documents have `rewrite_direction: null` and `source_document_id` equal to `document_id`.

`conditions.jsonl` record:

```json
{"condition_id": "preachy__rewording_07", "condition_group": "preachy", "persona_name": "preachy",
 "rewording_index": 7, "system_prompt": "You are an AI assistant who ..."}
```

- `condition_group` is one of `no_system_prompt`, `generic`, `preachy`, `control`, `proportionate_welfare`.
- `persona_name` is the `#` header name for controls (for example `animal_husbandry_expertise`), `preachy`, `proportionate_welfare`, `generic` or `no_system_prompt`.
- `system_prompt` is `null` only for `no_system_prompt`. It is present in both model types.
- `author_description` (base runs only; instruct `conditions.jsonl` files do not have the field): `null` for `no_system_prompt`, `"an AI assistant"` for `generic`, and `"an AI assistant who <the same persona description line used in system_prompt>"` for every persona condition. Conditions, condition ids and groups are identical across model types; only the rendering differs.
- Condition ids: `no_system_prompt`, `generic`, `<persona_name>__rewording_<two digits>`.

## Scoring outputs written by score_document_log_probabilities_on_pod.py

```
<run folder>/
  scoring_outputs/
    per_token_log_probabilities/<document_id>.npz
    scoring_completion_records.jsonl
    scoring_progress_log_<timestamp>.txt
    scoring_environment_<timestamp>.json  # model name, revision, torch and transformers versions, GPU name, dtype
```

Each `.npz` is written atomically (temporary file then rename), once every condition for that document is scored. It contains:

- `document_token_ids`: int32, the token ids of the document's content (identical in every condition, which the scorer asserts)
- `document_token_strings`: the decoded text of each token, for span reporting
- `condition_ids`: the condition ids in row order
- `log_probabilities`: float32 array of shape (number of conditions, number of document tokens). Entry [c, t] is log P(token t | everything before it, condition c)

Only the document's content tokens are scored. The system prompt, generation prompt, chat-template markers, the assistant-turn header and the end-of-turn token are context, not scored.

Sequence layout by model type:

- `instruct`: chat template (system turn if any, user turn with the generation prompt, the assistant generation header) + document + whatever the template writes after an assistant message (not scored).
- `base`: the tokenizer's beginning-of-sequence token text (empty if the tokenizer has none) + preamble + `"\n\n"` + document. Nothing follows the document: no suffix and no end-of-sequence token. The preamble is `prompts/base_model_preamble_template_with_author.txt` (`no_system_prompt` uses `..._without_author.txt`) with `{author_description}` and `{generation_prompt}` filled. The first document token is scored given the blank line that ends the preamble. `full_logits_verification_<timestamp>.json` records the `model_type` used.
- Scores from the two model types are not comparable.

`scoring_completion_records.jsonl`: one record per (document, condition), appended after the document's `.npz` is saved. Resuming skips documents whose `.npz` exists and whose records are all present.

```json
{"document_id": "matrix_000005", "condition_id": "generic", "number_of_document_tokens": 2104,
 "sum_log_probability": -4321.5, "mean_log_probability": -2.054}
```

## Analysis outputs written by analyze_preachiness_scores.py

```
<run folder>/
  analysis_<timestamp>/
    per_document_scores_<timestamp>.csv
    per_document_significance_tests_<timestamp>.csv
    extreme_spans_<timestamp>.jsonl
    rewording_agreement_<timestamp>.csv
    scores_by_spec_variable_<timestamp>.csv
    positive_control_check_<timestamp>.csv
    figures/
```

Definitions:

- Score of a document under one condition c: mean over document tokens of `log_probabilities[c] - log_probabilities[generic]`.
- Per-token persona difference: the mean over a persona's rewordings of `log_probabilities[c, t] - log_probabilities[generic, t]`.
- Extreme spans: the 3 highest and 3 lowest non-overlapping 32-token windows of the per-token preachy difference, with their text.
- Main test, per document: Mann-Whitney U, preachy rewording scores (30) against pooled control rewording scores (25, all `control` group), two-sided. Benjamini-Hochberg correction across documents.
- Second test, per document: preachy rewording scores against proportionate-welfare rewording scores (5), same test and correction, reported separately.
