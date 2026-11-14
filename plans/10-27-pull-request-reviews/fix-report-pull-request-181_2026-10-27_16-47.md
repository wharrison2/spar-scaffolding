# Fix report for pull request 181 review findings

Starting commit: `50629da510f84920abb8b70973fc5946e970be81` (tip of `william/sdf-audits-plausibility-shoehorning-variable-persistence`). Six commits in a detached worktree (since removed), exported as patches to `/private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/08c219e3-e182-4ff7-a9a8-e7625dddc1d7/scratchpad/review_fixes/pull-request-181/` (0001 to 0006). No branch was changed, nothing pushed. No real API was called; all tests stub the network. Patch bodies contain no outer-project, personal-path or assistant references (the patch author header carries the repository's configured git identity).

For every finding the regression tests were written first and run against the unfixed code, where they failed, then the fix was made and they passed. Where the failure was an `AttributeError` (the new function did not exist yet) this is noted.

## 1. Fuzzy quote check depends on alignment: done

- Commit: `Make the difflib quote match independent of the quote's offset` (fdb9a66)
- Tests (`tests/test_model_judged_audit_support.py`):
  - `test_a_one_character_difference_gives_the_same_verdict_at_every_offset_in_the_paragraph` (two parametrised cases: capitalised first letter, em dash versus hyphen; offsets 0 to 44 characters).
  - `test_a_quote_longer_than_the_paragraph_is_not_found_by_the_rapidfuzz_path_either` (the related plausible note).
- Failure before the fix: the offset test failed for both cases (some offsets scored under 90, others over), with `rapidfuzz` absent from the venv; the rapidfuzz test failed because a stand-in `rapidfuzz_module` whose `partial_ratio` aligns the shorter string inside the longer one returned 100 for a long quote against a short heading paragraph and the quote counted as found.
- Fix: the difflib fallback now starts a needle-length window at every matching block of the quote (`get_matching_blocks`), so the best alignment is found wherever the quote sits. Threshold stays 90. A needle at least as long as the paragraph is scored against the whole paragraph in both paths (`rapidfuzz.ratio` when available), so a long quote cannot match a short paragraph it merely contains. Single-paragraph rule and the paragraph-break flag are untouched.
- Choice: very short quotes that differ by one character (for example `the hens` versus `The hens were moved.`, 87.5) still score under 90. That is the same on the rapidfuzz path and is what the threshold means, not an alignment effect, so I left it.

## 2. Resumes never check corpus or effort; variable persistence records no configuration: done

- Commit: `Refuse resumes of a changed corpus, effort or variable persistence setup` (837f7fd)
- Tests:
  - `tests/test_model_judged_audit_support.py`: `test_resume_refuses_a_changed_corpus_and_names_it`, `test_resume_refuses_a_changed_compared_setting_and_says_from_what_to_what`.
  - `tests/test_plausibility_audit_sdf.py`: `test_resume_refuses_a_changed_corpus` (through `main`).
  - `tests/test_shoehorning_audit_sdf.py`: `test_resume_refuses_a_changed_reasoning_effort`, `test_resume_refuses_a_changed_corpus` (through `main`).
  - `tests/test_variable_persistence_sdf.py` (`TestRunConfiguration`): configuration contents, first run writes and identical resume accepted, resume with no recorded configuration refused, resume refused for a changed model, corpus, judge prompt and variable-meanings file.
- Failure before the fix: the five support and audit tests failed (no error raised on resume); the four variable persistence tests failed with `AttributeError` (`build_run_configuration` / `record_run_configuration` did not exist).
- Fix: `check_resume_is_compatible` takes `compared_settings` and compares `corpus_sha256` when both configurations record one; messages name what changed (recorded and current corpus path, or `setting changed from X to Y`). The shoehorning audit passes `reasoning_effort`. The variable persistence check writes `run_configuration_<timestamp>.json` (judge prompt hash, variable-meanings hash, resolved model, corpus path and hash) and its `--resume` runs the same shared check with `model`. Docstring and `evals/README.md` updated.
- Choice (conservative): a variable persistence `--resume` of a folder with no recorded configuration (a folder from before this change) is refused, since nothing can be verified. The configuration is written in `main`, not in `run_judging`, so existing `run_judging` tests are unaffected. The model recorded is `variable_persistence_model`, then the existing fallback chain, then the global `model`.

## 3. Shoehorning post-processor drops passes added on resume: done

- Commit: `Take the shoehorning post-processor's passes from every configuration` (4886bdb)
- Test: `test_postprocess_includes_passes_added_on_a_resume` in `tests/test_shoehorning_audit_sdf.py` (first configuration lists one pass, a later one lists three; records exist for two; expects columns for the resumed pass, `passes_completed` of both, and `all_passes_completed` False because the third was never run).
- Failure before the fix: the resumed pass had no columns and `all_passes_completed` read True (assertion failed).
- Fix: new `pass_names_of_run` takes the union of `passes` across all `run_configuration_*.json` files and the passes seen in the records, in `ALL_PASS_NAMES` order; `postprocess_run_folder` uses it for the per-document outputs.

## 4. Estimate-only undercount: done

- Commit: `Fix the estimate-only extrapolation when sample calls fail` (29e30cc)
- Tests: `test_estimate_only_scales_by_the_documents_that_succeeded_and_counts_the_failed_attempts` and `test_estimate_only_reports_no_estimate_when_every_sample_call_failed` in the plausibility test file; `test_estimate_only_scales_each_pass_by_its_succeeded_documents_and_counts_failed_attempts` and `test_estimate_only_reports_no_estimate_when_every_sample_call_failed` in the shoehorning test file.
- Failure before the fix: the scale assertions failed (document scale 1.0 instead of 1.5; output tokens 200 instead of 600); the all-failed tests failed because no "no estimate is available" line was printed (the old code printed a $0.00 full-corpus cost).
- Fix: scales are computed over the sample documents that succeeded (per pass for the shoehorning audit); the usage of the sample's error records is added to the token totals; latency of `None` is treated as zero; if nothing succeeded the function returns None and `main` prints that no estimate is available and writes no estimate file. Docs in `evals/README.md` updated.
- Choice: failed-attempt tokens are added to the numerator and scaled per succeeded document, so retry waste is projected over the corpus (the higher, more conservative figure). In the shoehorning audit a pass with no successful sample call is excluded from the total, listed in a new `passes_without_a_successful_sample_call` field and called out in the printed output rather than silently contributing $0.00.

## 5. Usage file misses spend: done

- Commit: `Count all paid usage in the per-run usage and cost file` (7cb65a3)
- Tests (in both audit test files): `test_the_usage_file_includes_the_unusable_first_attempt_when_the_second_attempt_raises`, `test_a_resume_counts_the_usage_of_earlier_sessions_error_records`.
- Failure before the fix: the transport-failure error record held zero usage and the usage file total was 0 instead of 1000; after a resume the usage file total was 1000 instead of 3000.
- Fix: `request_structured_output` attaches the summed usage of completed attempts to a transport exception before re-raising it; the new `usage_paid_before_transport_failure` reads it, and both audits' `work()` uses it for the `transport_failure` record. On resume `prior_records` is the judgments or classifications plus `errors_*.jsonl`, so earlier sessions' error usage is counted. README line on the usage file updated.

## 6. Run folder placement: done

- Commit: `Place audit runs in <run>/audit for a corpus file input` (95a46c4)
- Tests: `test_the_default_runs_directory_is_the_audit_folder_of_the_run` updated in both audit test files (the pinned `final/audit` path is replaced by `<run>/audit`; a loose corpus file still gets `audit` beside it); new `TestOutputFolderPlacement` in `tests/test_variable_persistence_sdf.py` (corpus file in `final/`, run directory and loose file; `--sample` / `--doc-ids` go to `audit/small_sample_checks`, full runs do not).
- Failure before the fix: the two updated audit tests failed against the old `final/audit` path; the two new variable persistence tests failed with `AttributeError`.
- Fix: both audits use `run_directory_of` from the support module; the variable persistence check computes its own audit directory (`report_directory_of`) and `default_output_directory`, which adds the `small_sample_checks` subfolder for `--sample` and `--doc-ids`. `--output-dir` still overrides. `audit_sdf.resolve_input` was left alone, as it also decides where `audit_report.json` goes for the corpus-level audit. Docstring and help text updated.

## Full-suite result

`env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY .venv/bin/python -m pytest -p no:cacheprovider` on the final commit: all tests pass, no failures (baseline was 1612 passed).
