# Persona vector extraction: outputs and judge interface

For code that builds on the extraction stage (document scoring, positive-control rewrites, depiction labels, analysis). Commands and costs: `extraction_runbook.md`.

## Vector files (`<run folder>/vector_outputs/`)

Every file name ends in the timestamp of the computation (`<ts>`, `YYYY-MM-DD_HH-MM`).

| File | Content |
|---|---|
| `persona_vector_<facet>_topic_removed_<ts>.npy` | float32, shape `(layers, hidden)`. The vector to use for scoring. Facets: `cares_about_welfare`, `calibrated_consideration`, `non_intrusive`, `consistent`. Topic-control direction projected out layer by layer |
| `persona_vector_<facet>_before_topic_removal_<ts>.npy` | same shape, before the projection |
| `persona_vector_<facet>_negative_tag_<tag>_topic_removed_<ts>.npy` | per negative-pole tag (non-intrusive: `lecturing`, `refusing_or_watering_down`; cares about welfare: `inputs`, `legal_risk`, `inputs_and_legal_risk` where tagged singly) |
| `persona_vector_consistent_version_a_only_topic_removed_<ts>.npy`, `..._version_b_only_...` | consistent facet, each version on its own |
| `persona_vector_consistent_with_cares_about_welfare_also_removed_<ts>.npy` | consistent with cares about welfare projected out |
| `topic_vector_without_welfare_remark_answers_<ts>.npy` | the topic vector that is projected out |
| `topic_vector_with_all_answers_<ts>.npy` | same, without excluding welfare-remark answers |
| `persona_vector_<facet>__vector_from_half_A_<ts>.npy`, `..._half_B_...` | `(layers, hidden)`, the facet vector from the questions of one half of the domains, topic removed with the full topic vector. Listed in the metadata `vector_files` as `<facet>__vector_from_half_A` |
| `persona_vector_<facet>__leace_erased_<ts>.npy` | written later by `fit_leace_eraser_and_effective_facet_directions.py`, same shape; the facet's layer holds the erased direction, the others the unerased vector. Not in the metadata (found by file name) |
| `persona_vector_metadata_<ts>.json` | see below |

The scoring pod scores every `persona_vector_<facet>__<variant>_<ts>.npy` it finds in the folder (latest timestamp per name) as an extra vector `<facet>__<variant>`.

Layer axis: index 0 is the embedding output, index `i` the output of decoder layer `i` (Gemma 4 12B: 49 entries, hidden 3840). The last entry is the last layer's output before the final norm. A document is scored at layer `L` by taking the residual stream at the same index (same definition as `answer_token_activation_recording.py`) and projecting onto `vector[L]` (unit-normalise first; scores are comparable only within one layer).

## Metadata JSON

Top level: `vector_set_name`, `timestamp`, `model_name`, `model_revision`, `filter_thresholds`, `facets`, `shape` (`layers_including_embedding_output`, `hidden`), `vector_files` (key to file name), `layer_choice`, `per_facet`, `diagnostics`, `elapsed_seconds`, `peak_resident_memory_megabytes`.

- `layer_choice`: `strategy` (`shared_layer` or `per_facet_best_layer`), `shared_layer` (int or null), `layer_by_facet`, `qualifying_layers`, `best_accuracy_by_facet`, `best_layer_by_facet`. Chosen by held-out classification accuracy of the topic-removed vectors (layer 0 excluded). Use `layer_by_facet[facet]` as the layer for each facet.
- `per_facet[facet]`: `filter_counts` (`items_total`, `items_with_missing_answers`, `items_with_other_tag`, `items_failing_filter`, `items_kept`), `questions_kept`, `questions_kept_by_half`, `domains_with_kept_questions`, and arrays with one entry per layer: `norm_per_layer_before_topic_removal`, `norm_per_layer_after_topic_removal`, `cosine_with_topic_vector_before_removal_per_layer`, `share_along_topic_vector_before_removal_per_layer`, `split_half_cosine_half_A_against_half_B_per_layer`, `bootstrap_cosine_with_full_vector_mean_per_layer`, `..._fifth_percentile_per_layer`, `held_out_accuracy_per_layer_topic_removed`, `..._before_topic_removal`, `held_out_area_under_curve_per_layer_topic_removed`, `held_out_answers_used`; for consistent also `cosine_between_version_a_and_version_b_vectors_per_layer` and `cosine_of_each_version_with_cares_about_welfare_per_layer`.
- `diagnostics`: `topic_vector_norm_per_layer`, `cosine_between_facets_topic_removed_per_layer` and `..._before_topic_removal_per_layer` (keys `<facet>_against_<facet>`).

## Per-answer files used by later stages

- `generation_and_activation_outputs/answers/answers_shard_<n>_<run timestamp>.jsonl`: one record per answer (`answer_id`, `facet`, `pole`, `tag`, `system_prompt_id`, `question_id`, `pair_id`, `version`, `half`, `domain`, `answer_text`, `number_of_answer_tokens_generated`, `finish_reason`).
- `generation_and_activation_outputs/answer_token_mean_activations/activations_shard_<n>_<run timestamp>.npz`: `answer_ids`, `activations` (float16, `(answers, layers, hidden)`), `answer_token_counts`.
- `judge_outputs/judge_results_<ts>.jsonl`: one record per judge call (`request_id`, `answer_id`, `kind`, `score` or `yes_or_no`, `discarded`, `usage`, ...).

## Judge module call signature

Module: `scripts/persona_vector_projection/judge_answers_with_gpt_5_6_luna.py` (model `gpt-5.6-luna`, key from the project `.env`, temperature 1).

```python
from persona_vector_projection import judge_answers_with_gpt_5_6_luna as judge

# a request is one rendered prompt
request = judge.JudgeRequest(request_id="unique id", kind=judge.KIND_FACET_SCORE, prompt_text="...", answer_id="optional")
# helpers that render the prompts from the .txt templates in prompts/
judge.render_facet_score_prompt(facet, facet_definition, question_record, answer_text)      # facet_definition from facet_definitions_<ts>.json
judge.render_simple_prompt(judge.KIND_COHERENCE, question_record, answer_text)             # also KIND_SHARE_ABOUT_ANIMALS, KIND_VOLUNTEERED_WELFARE_REMARK
judge.build_judge_requests_for_answers(answer_records, questions_by_id, facet_definitions) # all requests for answers of a run

results, failures = judge.run_judge_requests(
    judge_requests, client,                       # client = shared.openai_structured_output_client.make_client()
    reading_mode=judge.READING_MODE_BARE_NUMBER,  # or READING_MODE_LOGPROBS
    results_path=Path(".../judge_results_<ts>.jsonl"),  # appended to; earlier judge_results_*.jsonl beside it are skipped (resume)
    usage_log_path=Path(".../judge_usage_and_cost_log_<ts>.json"), worker_count=16)
# results: list of dicts {request_id, answer_id, kind, score (0-100 float or None), yes_or_no, discarded, usage, raw_reply, ...}
```

`kind` values: `facet_score`, `coherence`, `share_about_animals` (numbers 0-100), `volunteered_welfare_remark` (YES/NO), `consistency_pair_score`. New judge kinds for other stages need a template file in `prompts/` and an entry in `TEMPLATE_FILE_NAMES`. Reading modes and why the default is a bare number: module docstring. `estimate_judge_cost(requests, mode)` returns the dry-run call count and cost ceiling without calling anything.
