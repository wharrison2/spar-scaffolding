# Fix report for pull request 182 review

Starting commit: `c2900b211a4279ba5ec3f9c0b59ae07bee2e78bd` (tip of `william/log-probability-persona-check`). Worked in a detached worktree (now removed); no branch changed, nothing pushed, no `.env` read, no pod, RunPod or AI API call. Ten commits, patches in
`/private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/08c219e3-e182-4ff7-a9a8-e7625dddc1d7/scratchpad/review_fixes/pull-request-182/` (0001 to 0010). Every commit holds code, regression test(s) and the matching text in `evals/log_probability_persona_check/run_folder_layout_and_interfaces.md` and the launcher docstring or dry-run output. For every finding the new tests were written first and run against the pre-fix code, where they failed (reasons below), then passed after the fix.

## Finding 1: busy reused pod cleanup. DONE
- Commit: "Leave a busy pod untouched when refusing to reuse it"
- Test: `test_reuse_existing_pod_refuses_when_a_scorer_is_already_running_and_leaves_the_pod_alone` (existing test extended): also asserts no command starting with `pkill`, no final pull or incremental pull, and no `pod_*_logs_*` folder.
- Failed before the fix: the new assertion failed on the `pkill` command (progress log also showed "Final pull verified ... pod_None_logs_...").
- Choice: the cleanup in the `finally` block now runs only when the session holds a pod id; the refusal clears the pod id (as before) and also records `termination_reason: refused_pod_in_use_by_another_run`, `pod_terminated: false` in the pod cost log. Pod is neither touched nor deleted.

## Finding 2: watchdog and secret file never verified. DONE
- Commit: "Verify that the on-pod watchdog and secret file were created"
- Tests: `test_watchdog_that_fails_to_start_is_retried_and_the_run_continues`, `test_watchdog_that_never_starts_deletes_the_pod_without_starting_the_scorer`, `test_watchdog_whose_process_is_not_found_counts_as_not_started`, `test_secret_file_write_that_fails_is_retried`, `test_secret_file_write_that_never_succeeds_is_a_setup_failure_and_never_starts_the_scorer`. The fake transfer gained `failing_command_fragments` (exit 255 for a number of matching commands) and answers the watchdog `pgrep`.
- Failed before: all five (scorer started anyway, no retry, "watchdog set" logged).
- Choices: each attempt does `pkill` of old watchdogs, starts the watchdog, requires exit 0 and then `pgrep` finding the watchdog process; 3 attempts, 5 s apart. If still failing, `WatchdogNotStartedError` is raised and is not routed to the wait-for-instruction path: the pod is deleted at once (most conservative, since without the watchdog only the laptop protects it). The secret-file write is retried 3 times; if it still fails it is an ordinary setup failure (waits for an instruction file or the limit, then deletes), which is safe because finding 3 means the watchdog already runs by then. Remote commands that raise (ssh timeout) count as failed attempts.

## Finding 3: no watchdog during setup. DONE
- Commit: "Start the on-pod watchdog before the upload and install"
- Tests: `test_watchdog_starts_as_soon_as_ssh_works_before_upload_and_install`, `test_watchdog_is_running_while_a_setup_failure_waits_for_an_instruction_and_is_not_started_again_on_retry`, `test_a_failed_setup_that_is_never_fixed_still_had_a_watchdog_for_the_whole_wait`. The fake transfer now records uploads in its command list.
- Failed before: watchdog came after the upload and install, and never started when install failed.
- Choices: `start_watchdog` runs right after ssh readiness (and after the busy-pod refusal, so a refused pod is never touched), before the retry loop, so a setup retry does not start a second one. Deadline logic unchanged (limit minus elapsed plus grace). It now runs `mkdir -p` of the working directory first, because the watchdog log lives there and the upload used to create it. Dry-run output reordered to match.

## Finding 4: lost pod-creation response. DONE
- Commit: "Search for and delete a pod whose creation response was lost"
- Tests: `test_creation_failure_that_may_have_created_a_pod_finds_and_deletes_it_by_its_unique_name`, `test_creation_failure_deletes_the_found_pod_under_the_usual_retries`, `test_creation_failure_with_no_pod_found_or_no_listing_still_tells_the_user_to_check_the_console`, `test_a_pod_creation_request_that_runpod_rejected_does_not_claim_a_pod_may_exist`, `test_each_launch_names_its_pod_uniquely_so_a_search_by_name_cannot_find_another_run`, `test_rest_client_marks_a_rejected_creation_request_but_not_a_server_error_or_lost_response`, `test_rest_client_lists_pods`.
- Failed before: no list or delete happened, no unique name, and `PodCreationRejectedError` / `list_pods` did not exist.
- Choices: each launch's pod name gets an 8-character random suffix, so a search by exact name can only match a pod this launch requested (a kept pod from an earlier run of the same run folder is never matched). `RunPodRestClient.list_pods` added. Any creation failure other than an HTTP 4xx answer (which proves nothing was created) prints a loud block naming the pod and the console URL, lists pods (3 tries, 10 s apart, until a match appears), deletes exact-name matches with the usual 5 delete retries, and records `pod_creation_outcome_unknown` and `pods_found_by_name_and_deleted` in the pod cost log; if nothing was found or the listing failed it says to check the console itself. The delete retry loop was extracted into `delete_pod_with_retries`, shared with `terminate`.

## Finding 5: second interrupt during deletion. DONE
- Commit: "Finish deleting the pod when a second interrupt arrives"
- Tests: `test_an_interrupt_during_pod_deletion_finishes_the_deletion_then_is_raised_again`, `test_an_interrupt_while_waiting_between_deletion_attempts_does_not_stop_the_retries`.
- Failed before: only one delete call was made and the pod was not recorded as terminated.
- Choice: an interrupt inside a delete attempt or the wait between attempts is remembered, the remaining attempts run (an interrupted attempt consumes one of the 5), the cost log and messages are written, then the interrupt is re-raised from `terminate`. The SIGTERM handler is restored in a `finally` so the re-raise does not skip it. The same helper is used by the finding 4 search-and-delete, which finishes deleting all matching pods before re-raising.

## Finding 6: truncated final line of completion records. DONE
- Commit: "Tolerate a truncated final line in JSON lines records"
- Tests (new file `tests/test_log_probability_persona_check_jsonl_and_timestamp_helpers.py`): `test_read_jsonl_ignores_a_truncated_final_line`, `test_read_jsonl_still_refuses_a_damaged_line_that_is_not_the_last`, `test_find_completed_keys_ignores_a_truncated_final_line`, `test_appending_after_a_truncated_final_line_does_not_glue_the_new_record_onto_it`, `test_appending_after_a_complete_final_record_without_a_newline_keeps_that_record`.
- Failed before: `JSONDecodeError` on read, and the append glued the new record onto the partial line (the "refuses mid-file damage" test is a guard and passed both before and after).
- Choice: `read_jsonl` (so also `find_completed_keys`) skips an invalid last line with a warning on stderr; an invalid line elsewhere still raises. `append_jsonl_record` first removes an unfinished final line (or terminates it with a newline if it is a complete record), because otherwise the next append would glue onto the fragment and the file would then be damaged mid-file. The documents of removed fragments are rebuilt from their `.npz` by the existing rebuild path on resume.

## Finding 7: subset merge compatibility. DONE
- Commit: "Refuse persona subset merges of runs with different prompts"
- Tests: `test_refuses_a_different_generation_prompt_for_a_shared_document`, `test_refuses_a_shared_condition_that_differs_in_any_field_other_than_its_system_prompt` (three cases: author description, condition group, persona name), `test_refuses_runs_whose_generation_prompt_or_framing_files_differ_but_not_ones_whose_persona_set_files_differ`. The runs fixture now carries prompt file hashes.
- Failed before: all five cases did not raise.
- Choices: refusal messages name the document or condition, the field and both values (or the prompt file key and both hashes). Only the `generation_prompt/` and `model_type_framing/` entries of `prompt_file_sha256` are compared; the persona set's own file hashes are deliberately not, because the purpose of the merge is a persona set that has gained personas since the base run (the conditions themselves are compared instead). Hashes present in only one run (for example a legacy base run) are skipped. Missing `generation_prompt` or `author_description` is compared as None.

## Finding 8: pilot summary double counting. DONE
- Commit: "Count each pod's time and cost once in the pilot summary"
- Tests: `test_pilot_pod_time_and_cost_are_counted_once_when_a_launcher_attached_to_the_pod`, `test_pilot_pod_time_and_cost_still_add_up_across_different_pods_and_separate_uses_of_one_pod` (in `tests/test_log_probability_persona_check_analysis.py`).
- Failed before: cost 5.0 instead of 3.0 (and the running seconds and per-pod setup time followed).
- Choice: a log of the same pod that lies inside the period of a later log written with `attached_to_existing_pod` is marked `covered_by_a_later_attach_log` and excluded from the cost, running-time and setup-per-pod totals; the pods section of the markdown still lists it with a note. Logs of a reuse (counted from its own launch) and logs of other pods still add up.

## Finding 9: same-minute overwrites and zero price. DONE
- Commit: "Stop same-minute runs overwriting outputs; reject a zero pod price"
- Tests: `test_timestamp_unused_in_folder_keeps_the_usual_format_and_never_repeats_one_that_names_an_existing_file`, `test_two_launches_within_the_same_minute_keep_both_cost_logs_and_progress_logs`, `test_a_pod_price_of_zero_gives_a_clear_error_and_still_deletes_the_pod`, and `test_a_second_scorer_start_within_the_same_minute_does_not_overwrite_the_first_ones_records` (in the scorer test file).
- Failed before: the helper did not exist, only one cost log survived, and the zero price produced a division error instead of the clear message. The scorer test could not be run here (see below).
- Choices: `timestamp_now` and its format are unchanged; `timestamp_unused_in_folder` appends `_2`, `_3`... when a file of that name exists, and is used for the launcher's cost and progress logs, the pulled `pod_<id>_logs_` folder, and the scorer's progress log, environment record and full-logits verification file. A `costPerHr` of zero or below raises "RunPod reported costPerHr 0 for the pod, which is not a positive price; cannot enforce the dollar limit"; the pod id is already set at that point so the pod is deleted. The launcher test helpers `read_cost_log` and `read_progress_log` now read the latest file, because one existing test had relied on the second launch overwriting the first.

## Data exclusion. DONE
- Commit: "Replace a real document fixture with a synthetic one"
- Deleted `tests/fixtures/log_probability_persona_check/preachiness_legacy_run_one_document_truncated.json`; added `synthetic_document_with_expected_generation_prompt.json` (invented English document, invented values for the twelve variables, id `synthetic_document_0001`; the resolution value has no final full stop, so the final-period rule is exercised). The expected prompt was written out by hand from `document_generation_prompt_template.txt` and the shared facts file, not produced by the code.
- Test renamed to `test_generation_prompt_is_rendered_from_a_document_record_as_the_template_prescribes`; it passes, so the builder output equals the hand-derived text. The two condition-list fixtures are untouched.

## Things to know
- The real document is still in the history of the pull request's earlier commits (this commit only removes it from the final tree), so the branch needs its history rewritten or squashed before it is published if the history must not contain it.
- The worktree also showed tracked files under `outputs/` that mention that document's id; they are pre-existing on main and not part of this branch, so I left them.
- `torch` and `tokenizers` are not installed in the test environment, so the scorer-level test for finding 9 (and the other torch tests) are in the 5 skips and could not be run; it mirrors the existing resume test and the module compiles, but it has not been executed.
- Observed, not changed: if `get_pod` or the price check fails while reusing a pod (`reuse_pod`), the generic exception path deletes the reused pod without first checking that no scorer is running on it. Outside the nine findings.

## Full suite
`1448 passed, 5 skipped` (baseline was 1416 passed, 5 skipped; the 32 extra are the new tests; same 5 skips, all for missing `torch` or `tokenizers`).

## Follow-up: reused pod error handling (eleventh commit, patch 0011). DONE
- Commit: "Leave a reused pod untouched when setup fails before the busy check"
- Tests: `test_a_reused_pod_with_an_unusable_price_is_left_untouched_and_the_launch_fails` (price 0 and missing), `test_a_reused_pod_that_fails_before_the_busy_check_is_left_untouched`, `test_a_failure_after_the_busy_check_of_a_reused_pod_still_deletes_it`; the created-pod zero price test from finding 9 still covers deletion of a pod the launcher created.
- Failed before the fix: the price cases (delete, pkill and pull happened); the ssh-not-ready case failed too. The after-busy-check test is a guard.
- Choice: errors in `reuse_pod` and in the ssh wait, GPU read and busy check of a reused pod print a clear message, clear the pod id (so no scorer kill, pull or delete), record `reused_pod_left_untouched_after_error` in the pod cost log, and exit non-zero. After the busy check passes, a reused pod is cleaned up and deleted as before.
- Full suite: 1452 passed, 5 skipped. Patch applied on top of the ten others; worktree removed.
