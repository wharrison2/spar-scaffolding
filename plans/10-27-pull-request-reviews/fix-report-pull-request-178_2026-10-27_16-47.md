# Fix report: pull request 178

Starting commit: 029647d17e985eb713544d1cba4257748e0e1522. Patches: /private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/08c219e3-e182-4ff7-a9a8-e7625dddc1d7/scratchpad/review_fixes/pull-request-178/ (three files). Worktree removed.

## Finding 1: fixed
- Commit: "Stop layer 1 before assigning when any scoring or rewrite check failed"
- Tests: `test_one_unusable_document_stops_the_stage_before_assigning_and_is_rescored_on_resume` (tests/test_layer1_select_considerations.py, replaces the old test that asserted the empty-spec behaviour), `test_an_unusable_answer_stops_the_stage_and_a_resume_retries_then_includes_the_plan` (tests/test_layer1_description_rewrite.py). Two existing tests updated to expect the new stop (`test_a_malformed_answer_is_not_checkpointed_...`, and the truncated-OpenAI-answer test in tests/test_openai_api_and_description_rewrite_routing.py). The "every call failed" test was renamed to `test_a_failure_on_every_call_exits_as_systemic`.
- Confirmed failing before the fix: both new/rewritten tests failed with "DID NOT RAISE" on the pre-fix code.
- Choice made: the report left the behaviour open (hold the document out vs. stop). I chose to stop, after checkpointing: the assignment is a global solve, so any hold-out would still change the solution on resume. The selector now raises SystemExit after any failed scoring call, before writing selections or coverage; the description rewrite likewise raises SystemExit when any described plan is held out (this covers the report's second scenario). `apply_results` / `load_rewritten_plans` hold-out on a resume past layer 1 is unchanged. Docstrings updated in both modules.

## Finding 2: fixed
- Commit: "Accept only hands of recorded rounds and screen every batch"
- Tests (tests/test_layer1_screen.py): `test_a_redeal_interrupted_before_its_round_record_is_dealt_again_without_orphan_hands`, `test_a_round_whose_batches_were_only_partly_saved_screens_every_hand`, `test_a_round_zero_deal_interrupted_while_writing_hands_leaves_no_duplicate_hands`.
- Confirmed failing before the fix: all three failed on the pre-fix code (duplicate hand ids / 30 hands instead of 25 / unscreened hands accepted).
- Choice made: rather than add atomic commits, hands are rewritten (not appended) for round 0 and for each redeal, hands whose round has no record are ignored on load, a round whose saved batches do not exactly cover its hands is planned again (same seed, same batch ids), and `_finish` accepts only hands of recorded rounds. Module docstring updated.

## Finding 3: fixed
- Commit: "Replace a batch's earlier verdicts when it is screened again"
- Test: `test_a_batch_screened_again_replaces_its_earlier_verdicts` (tests/test_layer1_screen.py, uses zero redeal rounds so only round 0 is involved).
- Confirmed failing before the fix: reverted only `sdf_pipeline/layer1_screen.py` to the previous commit and the test failed (50 judgments for 25 hands, stale rejections); passes with the fix.
- Choice made: a re-screened batch drops its own earlier judgments and rejections from both files before appending the new ones (the new answer wins). Docstring line about readers de-duplicating was replaced.

## Full suite
`pytest` over tests: all pass (baseline 1448; now 1452 with the added tests), no failures.

Note: an edge outside the report remains: if a run is interrupted after a redeal is recorded and an earlier round is then re-screened and found clean, later recorded rounds are still accepted. This needs manual checkpoint deletion to reach and was not changed.
