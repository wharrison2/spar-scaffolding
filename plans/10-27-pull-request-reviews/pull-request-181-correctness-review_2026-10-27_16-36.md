# Pull request 181 correctness review (model-judged corpus audits)

Branch `william/sdf-audits-plausibility-shoehorning-variable-persistence` against base `william/sdf-plumbing-model-calls-and-api-keys`.

## Verdict

The core of the pull request holds up: failed judgments give empty cells and never default scores, inapplicable axes are left out of every mean, usage is summed across the parse retry, cached input is priced correctly, the strict schemas are well formed, and the pull request's own 118 offline tests pass. Six real defects remain, all in the edges around the core:

- **Quote verification (medium).** The fallback fuzzy quote matcher, which is the one that actually runs because `rapidfuzz` is not installed, rejects quotes that differ by a single character, depending on where the quote sits in the paragraph. This lowers the verification rates and inflates `added_in_rewrite`.
- **Resume checks (medium).** No resume path checks the corpus hash. The shoehorning audit also does not check reasoning effort, and the variable persistence check records no configuration to check against at all.
- **Shoehorning post-processor (medium).** It takes the pass list from the oldest run configuration, so passes added on a resume are left out of the per-document results.
- **Cost estimate (low).** The estimate-only extrapolation undercounts when a sample call fails.
- **Usage file (low).** It misses some spend that the repository cost log does record.

Four of these were demonstrated with offline tests in a scratch worktree (since removed). The other two were traced through the code.

## Findings (most severe first)

### 1. The fuzzy quote check rejects valid quotes with a one-character difference, depending on alignment
- **Severity:** medium. **CONFIRMED.**
- **Where:** `evals/model_judged_audit_support.py:341-351` (`partial_match_score` difflib fallback), used by `verify_quote` at `:363-381`.
- **Defect:** the fallback slides fixed-length windows with step `len(needle)//4`. A quote whose true position falls between two window starts is compared against misaligned windows. It then loses up to about one eighth of its length in matching characters and drops below the 90 threshold, even though it differs from the document by one character.
- **Failing scenario:** the document says "...that the hens were moved indoors because the weather turned cold, and the farm kept records of every move." The judge quotes "The hens were moved indoors because the weather turned cold, and the farm kept records", capitalising the first letter. With no prefix the quote is found. With 10 to 14 extra characters before the sentence it is not found (score 87.4 to 89.5); with 15 to 29 extra characters it is found again. Other cases:
  - A quote that differs only in an em dash versus a hyphen is not found (87.4).
  - `verify_quote("the hens", "The hens were moved.")` is not found (87.5).
  - `normalize_text` does no case folding, so a one-letter case change always goes through this fuzzy path.
- **Consequence:** `quote_found_in_document` and `quote_verification_rate` are understated. In the shoehorning post-processor (`evals/shoehorning_audit_postprocess_sdf.py:31-38`) the same check runs against the draft, so a quote present in the draft but slightly re-cased or re-punctuated is classed as `added_in_rewrite`. A failure against the final text gives `quote_not_verified` instead of a stage.
- **Evidence:**
  - The `.venv` has no `rapidfuzz`, so the module falls back to difflib (`rapidfuzz_module is None`).
  - A demonstration test in the scratch worktree (`test_fuzzy_fallback_rejects_one_character_difference_depending_on_offset`) passed, confirming all three rejections.
- **Related, PLAUSIBLE (low):** the results depend on the environment. If `rapidfuzz` is ever installed, `fuzz.partial_ratio` aligns the shorter string inside the longer one. A long quote that wholly contains a short paragraph, such as a one-line heading, then scores 100 and counts as "found". That would defeat the single-paragraph rule and the `quote_spans_paragraph_break` flag.

### 2. Resumes never check that the corpus is unchanged; the shoehorning resume also accepts a changed reasoning effort
- **Severity:** medium. **CONFIRMED.**
- **Where:**
  - `evals/plausibility_audit_sdf.py:250-259` and `evals/shoehorning_audit_sdf.py:403-404`, both calling `check_resume_is_compatible` (`evals/model_judged_audit_support.py:156-167`).
  - `evals/variable_persistence_sdf.py:573-583`.
- **Defect:**
  - Both audits record `corpus_sha256` but never compare it on resume.
  - The shoehorning audit compares prompt hashes only, not `reasoning_effort` (the plausibility audit does check effort and model).
  - The variable persistence check writes no configuration at all. Its `--resume` therefore accepts any change of prompt, variable-meanings file or `variable_persistence_model`.
- **Failing scenarios:**
  - **Changed corpus.** Document ids are deterministic (`matrix_NNNNNN`). If `--input outputs/sdf/latest` points at a newer run when a run folder is resumed, every already-judged id is skipped. The old judgments are then joined to the new content: the post-processor reads `corpus_path` again and checks quotes against the new text.
  - **Changed effort.** A shoehorning run started with `--reasoning-effort low` and resumed with `--reasoning-effort high` runs without complaint, and the run folder holds a mix of both efforts.
- **Evidence:** a demonstration test (`test_shoehorning_resume_accepts_changed_reasoning_effort`) passed. The records' `reasoning_effort` values were `{"low", "high"}` and no error was raised. For the corpus hash and the variable persistence check, the code was read: there is no comparison.

### 3. The shoehorning post-processor takes the pass list from the oldest run configuration and drops passes added on resume
- **Severity:** medium. **CONFIRMED.**
- **Where:** `evals/shoehorning_audit_postprocess_sdf.py:206` and `:237-238`.
- **Defect:** `configuration["passes"]` comes from `sorted(run_configuration_*.json)[0]`, the first session's configuration. That list drives `build_per_document_rows`, which builds the item columns, `passes_completed`, `all_passes_completed` and the other-considerations counts.
- **Failing scenario:**
  1. Run `--passes moral_considerations`.
  2. In a later minute, resume the same folder with all three passes.
  3. `per_document_results_*.jsonl` has no columns at all for `field_distinctions` or `epistemic_notes` (no `item_17...`, no `other_considerations_count_epistemic_notes`), although those classifications exist.
  4. `all_passes_completed` reads True.
- **Side effect:** the occurrence rows and the per-item summary iterate over the records, not the configuration, so they do include the later passes. The output files therefore disagree with each other.
- **Evidence:** a demonstration test (`test_shoehorning_postprocess_drops_passes_added_on_resume`) passed.

### 4. The estimate-only extrapolation undercounts when a sample call fails
- **Severity:** low. **CONFIRMED.**
- **Where:** `evals/plausibility_audit_sdf.py:339-359` and `evals/shoehorning_audit_sdf.py:329-358`, called at `plausibility_audit_sdf.py:427-429` and `shoehorning_audit_sdf.py:425-428`.
- **Defect:**
  - `sample_records` contains only the successful judgments.
  - `character_scale` and `document_scale` divide by all the sample documents.
  - The output and cached-input totals sum over the successes only.
- **Failing scenario:**
  - With an estimate sample of 6 where 1 call fails (for example after two parse failures, or a transport failure), the projected output tokens, cached input and latency are about 5/6 of the true figures, and the uncached input is understated by that document's share of characters.
  - If every sample call fails, the estimate is $0.00 and is printed as the full-corpus cost.
  - The tokens of the failed attempts, which were paid for, are not counted either.
- **Evidence:** a demonstration test passed: one success out of two sample documents gives exactly half the cost of two successes.

### 5. The run's usage and cost file misses spend that the repository cost log records
- **Severity:** low. **CONFIRMED** (traced).
- **Where:**
  - `evals/model_judged_audit_support.py:503-523`.
  - `evals/plausibility_audit_sdf.py:314-323`, `:274-275` and `:415`.
  - The same pattern in `evals/shoehorning_audit_sdf.py:302-313`, `:260-261` and `:413`.
- **Defect:**
  - **Lost first-attempt usage.** Suppose attempt 1 returns an unusable answer and attempt 2's call then raises a transport error. The exception escapes `request_structured_output` and the usage of attempt 1 (in `attempt_usages`) is lost. `work()` then records a `transport_failure` with zero usage, even though `shared/openai_api.py` already logged attempt 1's cost to the cost log.
  - **Prior errors left out on resume.** On resume, `prior_records` is loaded from the judgment or classification files only. Usage from earlier sessions' error records is left out, although `usage_and_cost_log_*.json` says the totals "include records from before any resume".
- **Consequence:** the per-run usage and cost file reports less than the cost log in those cases. The cost log remains correct.

### 6. Run folders land in `<run>/final/audit/` when `--input` is the corpus file
- **Severity:** low. **CONFIRMED.**
- **Where:**
  - `evals/plausibility_audit_sdf.py:384-387`.
  - `evals/shoehorning_audit_sdf.py:384-386`.
  - `evals/variable_persistence_sdf.py:580` (through `evals/audit_sdf.resolve_input`).
- **Defect:** for `--input <run>/final/sdf_corpus.jsonl` the default runs directory is `<run>/final/audit/...`. Both module docstrings and `evals/README.md` say `<run>/audit/...`, and `run_directory_of` in the support module already maps a corpus file in `final/` to the run. A test pins the `final/audit` path (`test_the_default_runs_directory_is_the_audit_folder_of_the_run`). The same run's audits therefore end up in two places depending on how `--input` was spelled. The "a folder for this run name already exists" guard only searches the chosen directory, so it does not catch the other location.
- **Also:** the variable persistence check puts `--sample` and `--doc-ids` runs in the same `audit/` folder as full runs. It has no separate folder for small-sample checks the way the two audits do.

## Checked and found sound

- **Failed judgments:** they become `judgment_status: failed` (plausibility) or None item cells and `passes_failed` (shoehorning), never default scores. A later success replaces an earlier error. The variable persistence check summarises `ok` rows only.
- **Means:** inapplicable axes are excluded from `mean_score_among_applicable` in both the per-axis summary and the per-variable-value means. The overall mean uses `overall_plausibility` only.
- **Plausibility validation:** it rejects missing or extra axes, non-integer, boolean and out-of-range scores (0, 6, 3.5, True), bad severities, over-long lists and a non-string or non-null `english_gloss`. Nothing is defaulted.
- **Strict schema construction:**
  - Every object lists all its properties as required with `additionalProperties: false`.
  - The score is an integer enum.
  - Property order puts reasoning before the score.
  - The variable persistence schema is the same, with the consideration labels as an enum.
  - The shoehorning schema fixtures are independent JSON files, not derived from the code.
- **Parse retry:**
  - Exactly two attempts are made.
  - A refused or cut-off answer counts as a failure.
  - Usage is summed over the attempts in both the success and the failure path.
  - `cached_input_tokens` is priced at the cached rate and subtracted from the full-rate input.
  - Reasoning tokens are priced as part of the output.
  - `usage_out` matches what `shared/openai_api.py` logs.
- **Paid calls on resume:** the plausibility audit skips by `doc_id` and the shoehorning audit by `(doc_id, pass)`. Errors are retried. The plausibility resume refuses a changed prompt, model or effort, and the prompt hash check refuses a changed shared header or pass file.
- **Partial runs:** `--estimate-only`, `--document-ids` and `--document-limit` all go to `small_sample_checks`. Unknown ids exit with an error. A run name cannot be reused without `--resume-folder`.
- **Stratified sample:** deterministic (sorted, then shuffled with a fixed seed), round-robin over the strata.
- **`origin_stage` logic:** the order (unverified in the final text, then draft unavailable, then present in or added since the draft) is correct apart from the matcher problem in finding 1. Draft ids line up with final ids (`layer2_draft` and `layer4_score` share `doc_id`).
- **Variable persistence model chain:** `variable_persistence_model`, then `score_model`, then `plan_model`, then the global `model` inside `api.call_claude`. The refusal fallback that puts the schema into the prompt is still validated in code. Duplicate, missing or unassigned consideration labels are rejected. Transport errors become failed rows.
- **`init_cost_log_only`:** it points both the cost log and `model_calls.jsonl` into the run folder without needing a config file or an Anthropic key.
- **Hash-pin tests:** they compare against literal constants, so they can fail. The pull request's 118 offline tests in the four test files all pass.
- **Corpus collisions:** no duplicate `doc_id`s and no variable-name collisions with row fields in the real corpora under `outputs/sdf/runs/`.
