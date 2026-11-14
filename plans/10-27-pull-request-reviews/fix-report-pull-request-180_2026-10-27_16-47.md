# Fix report for pull request 180 review

Starting commit: 64d1439add7783645f7981c8231f84d5033f46a3. Five commits, in report order. Patches: /private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/08c219e3-e182-4ff7-a9a8-e7625dddc1d7/scratchpad/review_fixes/pull-request-180/. Worktree removed.

## Finding 1: done
- Commit: "Keep model call record failures from failing paid calls"
- Tests: TestModelCallsRecord.test_unwritable_record_does_not_fail_a_successful_call and ...does_not_hide_the_original_api_error (tests/test_api.py)
- Failed before the fix: UnicodeEncodeError escaped call_claude on success, and replaced the original ValueError on the failure path.
- Choice: the guard sits inside _log_model_call, so the OpenAI wrapper is covered too. It warns once per process on standard error.

## Finding 2: done
- Commit: "Write full model call records only for runs with their own cost log"
- Test: test_run_with_its_own_cost_log_records_beside_it_but_an_evaluation_does_not (tests/test_api.py)
- Failed before the fix: model_calls.jsonl existed beside the global cost log after an evaluation-style init.
- Choice: behaviour made to match the docstring. init records model calls only when given an explicit cost_log_path (no caller needed the global records; all five run entry points pass one). The fixture in tests/conftest.py resets the new flag. Comments at the evaluation init calls and the README sentence were updated.

## Finding 3: done
- Commit: "Ignore the environment's Hugging Face token in the default key mode"
- Tests: TestHuggingFaceTokenSource (3 tests, tests/test_publish_hf.py)
- Failed before the fix: huggingface_hub.get_token() still returned the environment token in the default mode (confirmed with the helper stubbed to the old behaviour).
- Choice: the documented contract wins. Added hide_environment_value_unless_environment_is_allowed in shared/api_keys.py; in the default mode HF_TOKEN (unless the repo-root .env supplied it) and the legacy HUGGING_FACE_HUB_TOKEN are removed from the process environment, leaving the cached login as fallback. --find-dotenv unchanged. The publish_hf.py module docstring and CLAUDE.md were updated.

## Finding 4: done
- Commit: "Confine the test fixture's .env search to its temporary folder"
- Test: test_the_suite_fixture_does_not_search_above_its_temporary_repo_root (tests/test_api_keys.py)
- Failed before the fix: a fake .env in the folder above the fixture's repo root was read.
- Choice: the fixture replaces the upward search with a repo-root-only lookup; the key loader's own tests restore the real one.

## Finding 5: done
- Commit: "Write recorded system prompts atomically"
- Test: test_system_prompt_file_is_written_under_a_temporary_name_then_renamed (tests/test_api.py)
- Failed before the fix: with the rename step failing (simulating a kill), a file remained at the hashed name.
- Choice: temporary name includes the process id; the temporary file is removed on failure. A truncated file from before this change is not repaired (not asked).

## Full suite
1502 passed (baseline 1494, plus 8 new tests), no failures.
