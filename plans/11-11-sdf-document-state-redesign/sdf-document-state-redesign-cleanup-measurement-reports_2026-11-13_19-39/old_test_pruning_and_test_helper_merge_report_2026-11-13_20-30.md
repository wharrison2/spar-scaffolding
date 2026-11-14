# Old test pruning and test helper merge (branch william/sdf-document-state-redesign-cleanup)

Starting commit 56c4ce6. Commits: 7ed78a3 (merge helpers), 67e80ed (move builders into support modules), 448ccb6 (remove 31 tests). Suite: 2,837 passed, 5 skipped before; 2,806 passed, 5 skipped after (31 fewer, exactly the removed tests). `git diff 56c4ce6 -- tests/fixtures` is empty. No regenerate script was run; no API call was made.

## Item 4: helpers merged

| Helper | Now |
|---|---|
| fake Anthropic and OpenAI clients, `_wrap_to_note_stage_and_item` (audit and prompt equivalence support) | one copy in `tests/sdf_prompt_equivalence_fixture_support.py` as `build_fake_anthropic_class`, `build_fake_openai_class`, `wrap_to_note_stage_and_item`; the recorder is duck-typed (both recorder types have `handle_request` and `client_constructions`), so no parameter was needed; the audit support calls them |
| `differences_between_values` | the audit support keeps a three-line wrapper that calls the prompt version with `ordered=True` (the audit version always compared key order). Failure message wording for long text differences now matches the prompt version |
| fake-keys isolation block (prompt `run_scenario_in_isolation`, audit `_isolate_from_real_keys_and_global_state`, resume `isolated_process_environment`) | one `isolate_from_real_keys_and_global_state(monkeypatch, working_directory)` in the prompt support; the other two call it |
| `_stage_rows` / `_rows` | `stage_rows(run_directory, layer_number, file_name)` in the prompt support, used by both tests |
| `_records` (three copies) | replaced by `utils.load_jsonl`, which behaves the same (skips blank lines, empty list for a missing file) |
| `events_of` (two copies), `reply` (two copies), SDF `run_main` / `_run_main` (three copies) | new `tests/sdf_step_and_command_line_test_helpers.py`: `events_of(stage_dir, step_name=None)`, `reply`, `run_main(monkeypatch, config_file, *extra, main, label=None)`; the three test files bind it with `functools.partial` |

Left as they are, and why:
- `describe_difference_of_texts`: the audit one takes a recorded hash description and prints hashes, the prompt one compares two strings and prints context. Same name, different job; merging would need a branch on argument type.
- `_sha256`, `_json_copy`, `_short` copies in the audit support: two to four lines each, `_short` has a different limit (200 against 160) and the audit one is also used in request messages; importing private names across modules would read worse than the copies.
- `_events_of` in `test_sdf_document_event_fold.py`: reads from text, not from a stage folder.
- `_run_main` / `run_main` in `test_pref_pipeline`, `test_publish_hf`, `test_diversity_eval`, `test_shoehorning_audit_sdf`, `test_plausibility_audit_sdf` call different programs with different argument lists; no shared shape.
- `_rows` inside `test_website_page.py` is an unrelated local function.

## Item 5: step 1, helpers moved (no test changed)

From the four old files into `tests/sdf_plan_draft_rewrite_and_score_test_builders.py`, `tests/sdf_screen_test_builders.py`, `tests/sdf_description_rewrite_test_builders.py`, `tests/sdf_consideration_selection_test_builders.py`. All importers (step tests, `test_layer1_stage_model_fallback.py`, `test_openai_api_and_description_rewrite_routing.py`) now import from these.

## Item 5: tests removed (each pair read before removal)

Abbreviations are written out: plan step tests = `tests/test_sdf_plan_step.py`; draft step tests = `tests/test_sdf_draft_step_resume.py`; critique and revise step tests = `tests/test_sdf_critique_and_revise_steps.py`; score step tests = `tests/test_sdf_score_and_gate_steps.py`; screen step tests = `tests/test_sdf_screen_and_redeal_steps.py`; description rewrite step tests = `tests/test_sdf_description_rewrite_step.py`; selector step tests = `tests/test_sdf_consideration_scoring_and_assignment_steps.py`; prompt equivalence test = `tests/test_sdf_prompt_equivalence_and_final_corpus.py` (requests, stage files and final corpus compared with the recordings).

| Removed test | Covering test |
|---|---|
| test_sdf_layers.py::TestLayer12Plan::test_composes_once_then_plans | prompt equivalence test, request and stage-file comparisons for the main scenario (stage tag, system prompt, every plan description, prompts file) |
| ::TestLayer12Plan::test_composition_is_reused_and_resume_makes_no_calls | plan step tests: test_resume_makes_zero_calls_and_changes_no_file (zero calls, every stage file byte-identical) |
| ::TestLayer12Plan::test_malformed_plan_is_retried_on_resume | plan step tests: test_resume_retries_exactly_the_failed_plans_and_keeps_incoherent_ones_done (no-tags answer retried) |
| ::TestLayer12Plan::test_truncated_plan_is_retried_on_resume | same test (max_tokens answer retried) |
| ::TestLayer12Plan::test_plan_model_override_used | prompt equivalence test, request comparison for scenario minimal_config_with_only_a_plan_model |
| ::TestLayer12Plan::test_one_poison_call_does_not_kill_the_layer | plan step tests: test_resume_retries_exactly_the_failed_plans_and_keeps_incoherent_ones_done (thrown call, siblings kept, only it retried) |
| ::TestLayer3Draft::test_drafts_from_descriptions | draft step tests: test_request_sent_has_exactly_the_arguments_the_layer_always_passed and test_events_record_the_hash_of_each_request_and_the_record_written; prompt equivalence test (user message carries the description) |
| ::TestLayer3Draft::test_incoherent_plans_are_skipped | prompt equivalence test, main scenario (incoherent hand has no draft request) |
| ::TestLayer3Draft::test_missing_tags_retried_on_resume | draft step tests: test_resume_retries_exactly_the_failed_documents |
| ::TestLayer3Draft::test_resume_makes_no_calls | draft step tests: test_resume_makes_zero_calls_and_changes_no_file |
| ::TestLayer3Draft::test_one_poison_call_does_not_kill_the_layer | draft step tests: test_resume_retries_exactly_the_failed_documents and test_a_document_that_always_fails_does_not_stop_the_others_until_it_is_alone |
| ::TestLayer3Draft::test_all_calls_failing_is_a_systemic_error | draft step tests: test_a_document_that_always_fails... (same `match="systemic"`) |
| ::TestLayer4Rewrite::test_rewrites_and_keeps_review | critique and revise step tests: test_combined_takes_the_text_before_the_tags_as_the_review, test_each_step_sends_its_stage_tag_model_and_token_budget (8000, cache), test_requests_read_description_variables_and_considerations_from_the_draft... |
| ::TestLayer4Rewrite::test_missing_tags_retried_on_resume | critique and revise step tests: test_a_failed_revise_after_an_accepted_critique_resumes_without_a_new_critique_call |
| ::TestLayer4Rewrite::test_resume_makes_no_calls | critique and revise step tests: test_resume_makes_zero_calls_and_changes_no_file |
| ::TestLayer5Score::test_low_spec_conformance_does_not_gate | score step tests: test_gate_ignores_spec_conformance |
| ::TestLayer5Score::test_parse_error_records_fallback_and_is_not_retried | score step tests: test_interpret_accepts_an_unparseable_answer_as_five_five_five (nonsense text) and test_resume_makes_zero_calls_and_changes_no_file |
| ::TestLayer5Score::test_wrong_shaped_judge_reply_falls_back_like_a_parse_error | score step tests: same parametrised test, case `[8, 7, 9]` |
| ::TestLayer5Score::test_near_dup_cull | score step tests: test_cull_keeps_the_first_of_a_near_duplicate_pair_in_the_order_given; prompt equivalence test: test_gate_and_cull_boundaries_are_on_both_sides_of_each_threshold (near_dup_dropped file) |
| ::TestLayer5Score::test_resume_makes_no_calls | score step tests: test_resume_makes_zero_calls_and_changes_no_file |
| ::TestLayer3SplitRewrite::test_critique_is_banked_so_a_failed_revise_does_not_rebill_it | critique and revise step tests: test_a_failed_revise_after_an_accepted_critique_resumes_without_a_new_critique_call |
| ::TestLayer3SplitRewrite::test_an_untagged_critique_is_used_rather_than_rebilled | critique and revise step tests: test_critique_with_no_review_tags_is_accepted_whole |
| ::TestLayer3SplitRewrite::test_resume_after_success_makes_no_calls | critique and revise step tests: test_resume_makes_zero_calls_and_changes_no_file |
| ::TestLayer3RevisionDirectives::test_a_deal_present_but_the_arm_off_changes_nothing | critique and revise step tests: test_voice_and_length_directives_ride_the_revise_and_combined_requests_only (last assertions, arm off) |
| test_layer1_screen.py::test_numbers_map_back_to_the_right_hands_and_rejections_carry_the_round | screen step tests: test_interpret_accepts_the_kept_hands_and_removes_the_implausible_ones and test_screen_events_fold_to_the_judgments_and_redeal_removes_each_screened_out_hand |
| test_layer1_screen.py::test_batches_are_seeded_and_input_order_independent | screen step tests: test_group_is_a_seeded_shuffle_of_the_sorted_ids_cut_into_batches |
| test_layer1_screen.py::test_a_lone_removed_hand_cannot_be_redealt_into_a_different_hand | screen step tests: test_cards_that_cannot_be_redealt_are_dropped_with_nothing_created and test_a_removal_that_cannot_be_redealt_is_recorded_as_dropped_cards; prompt equivalence test scenario screen_removal_that_cannot_be_redealt |
| test_layer1_description_rewrite.py::test_a_dealt_value_conflict_drops_the_plan_and_logs_it | description rewrite step tests: test_decline_rewrite_and_conflict_in_one_run and test_interpret_a_dealt_value_conflict_is_removed_and_its_records_are_todays (rejection record equal to the full dictionary) |
| test_layer1_description_rewrite.py::test_a_malformed_answer_is_not_checkpointed_and_resume_retries_exactly_that_plan | description rewrite step tests: test_a_failed_document_stops_the_run_after_the_step_and_resume_retries_exactly_it (case `unparseable`, same exit text) |
| test_layer1_description_rewrite.py::test_every_call_failing_is_systemic | description rewrite step tests: test_every_call_failing_is_systemic_for_both_failure_kinds (same exit text) |
| test_layer1_select_considerations.py::test_a_failure_on_every_call_exits_as_systemic | selector step tests: test_a_thrown_call_is_not_asked_again_and_counts_towards_the_systemic_stop (same exit text) |

## Kept although the measurement report listed it as fully duplicated (the report counted 37 + 11 + 10 + 7 tests; 31 held up)

- plan: `test_incoherent_is_checkpointed_as_deliberate_rejection` (all-incoherent run finishes without a systemic stop; covering tests do not assert it); `test_all_plan_calls_failing_is_a_systemic_error` (asserts the word "systemic" in the exit message; the plan step test asserts a different fragment of that message).
- rewrite: `test_rewrite_model_override_used` (the covering test sets a critique model explicitly, so the fallback to the rewrite model is not asserted).
- split rewrite: `test_a_truncated_critique_is_not_banked` (covering test checks only the interpretation, not that no critiques file is written); `test_critique_model_falls_back_to_the_rewrite_model` (same reason as above); `test_split_without_the_templates_refuses_rather_than_guessing` (covering test removes a different template file and matches a different file name).
- directives: `test_no_deal_sends_the_prompt_the_pipeline_always_sent` (the template test renders the template; this one checks the sent request); `test_dealt_cards_reach_the_rewriter_when_the_arm_asks_for_them` (also asserts the deal is recorded on the output).
- score: `test_scores_gate_and_corpus_fields` (corpus eval fields type_name, language, register are asserted only here and in the recording).
- plan, draft: the whole sign-off and truncation class (kept as instructed); `test_the_drafter_the_rewrite_and_the_judge_see_the_planners_lines...` (the claim that the equivalence scenario covers it was not verified).
- screen: `test_screen_shows_the_planner_variable_lines_in_numbered_batches_of_ten` (real renderer and "no ids shown" asserted only here); `test_unusable_answers_are_not_checkpointed` (five cases, two of which the step tests lack) and `test_out_of_range_numbers_are_rejected_and_not_checkpointed` (assert the literal exit text, the step test compares to a constant); `test_one_failed_batch_stops_before_any_redeal...` (stop rule, asserts no redeal); `test_batch_size_knob` (config knob to call count; no step test sets the knob); `test_resume_after_a_completed_screen_makes_zero_calls` (also asserts the returned hands are identical, and the step test does not); `test_disabled_screen_makes_no_calls_and_writes_nothing` and `test_snapshot_without_the_screen_template_skips_the_screen` (early returns, kept as instructed); `test_screen_terminates_after_the_maximum_rounds_and_reports_leftovers`, `test_removed_hands_that_cannot_be_redealt_are_dropped_and_recorded`, `test_zero_redeal_rounds...` (summary file and counts not asserted by the step tests).
- description rewrite: `test_a_decline_passes_the_original_untouched` (identity of every record, no extra keys); `test_a_rewrite_replaces_description_and_keeps_the_original_and_changes` (asserts the variables are carried over); `test_the_model_sees_the_planners_variable_lines_and_the_description`; `test_incoherent_plans_are_not_sent_and_pass_through` (only the recording covers it); `test_an_unusable_answer_stops_the_stage_and_a_resume_retries_then_includes_the_plan` (held-out stop, reload excludes the plan); `test_resume_after_a_completed_stage_makes_zero_calls_and_reapplies_results` (also asserts the reloaded plans).
- selector: `test_a_rerun_scores_nothing_that_is_already_scored` (asserts the returned plans equal the first run's), `test_one_unusable_document_stops_the_stage_before_assigning_and_is_rescored_on_resume` (no selection or coverage file written before the stop), `test_a_resume_scores_only_the_documents_without_a_saved_row` (crash window), and the rest are unique or partial as the report says.
- audit tests (`test_plausibility_audit_sdf.py`, `test_shoehorning_audit_sdf.py`, `test_compliance_sdf.py`): nothing removed. The report itself recommended leaving them: the hash and prompt-text pins fail with a message that names the prompt file, while the equivalence failure shows a request field difference. All of `tests/test_layer3_templates.py` kept.
