# SDF document state redesign: draft pull request stack writeup

Plan: `plans/11-11-sdf-document-state-redesign/sdf-document-state-and-update-steps-plan_2026-11-11_21-24.md` (risks file beside it).
Pipeline repository: `animal-welfare-data-pipeline/` (GitHub `sentfutures/animal-welfare-data-pipeline`). Branch backups before restacking: `backup/before-stack-2026-11-13/*` in that repository.
Rules for this work: every pull request is a draft; nothing is merged; no replies to review comments; nobody is notified except William.

## Log

- 2026-11-13: plan edits (recording rule, stack, audit equivalence test, compatibility risks 9 to 12, kept-files list, request-settings follow-on). Commits on `william/11-11-sdf-real-run-request-replay` in this project.
- Equivalence test branch: recording rule written into the pipeline `CLAUDE.md`, the regenerate script, the test and its failure message (`924bb8a`). Full suite passes.
- Paired-test branch rebased onto the test branch, then restored: it is not part of the stack.
- Cleanup branch rebased onto the test branch; two unused config keys kept so it passes the test without regenerating (`e20ca9c`). Loader branch rebased onto the cleanup branch. Suite: cleanup 2,334 passed; loader 2,433 passed; 0 failed.
- Draft pull request #193 opened for the equivalence test.
- Force push of the rebased cleanup branch was blocked by the permission classifier.
- Recordings shrunk by a Sonnet subagent: message texts and values over 400 characters stored as sha256 and length, settings kept in plain text, dealt hands and other repeated structures stored once per file. 10 MB (145,488 lines) to 2.16 MB (53,020 lines). Format version 4. Suite 2,328 passed (same as before the change on that branch). Five deliberate breaks (draft text, system prompt, effort setting, stage-file value, dealt value) each fail with a message naming what changed. Commits `73e966f`, `b89db61`, `3389a34`, `16db36b`.
- Review bot on #193 (reviewed `924bb8a`): two findings, both being fixed: Faker name pools unpinned (recordings would change when Faker's data changes); the regenerate guard ignores a failed `git diff` against a missing `origin/main`.
- Work item 3 (docs), branch `william/sdf-docs-steps-a-document-goes-through`, `498b4d4`: step table in `CLAUDE.md` and the `run.py` docstring, checked against the code (no wrong rows; clarifications only); stale layer numbers fixed in `CLAUDE.md`, `README.md`, `prompts/README.md`. Suite 2,334 passed; equivalence test passes unchanged.
- Bot findings fixed (`aa66d12`, `0fa7f92`): fixed invented name pools patched over `entity_pools.build_pools_for_locale` in every scenario (the 120 equivalence tests pass with Faker unimportable); the regenerate script exits when `origin/main` cannot be resolved or any git call fails.
- Clean history: `william/sdf-prompt-equivalence-test-compact-recordings` is one squashed commit (`d83e4c0`) on `origin/main`, largest object 167 KB; suite 2,328 passed. Draft #194 opened; #193 closed without comment.
- Cleanup restacked as `william/sdf-cleanup-dead-code-and-shared-helpers-on-equivalence-test` on #194: passes the compact recording without regenerating, 2,334 passed. Draft #195. The old remote branch `william/sdf-cleanup-dead-code-and-shared-helpers` is stale and left alone.
- Docs rebased onto #195: 2,334 passed. Draft #196.
- Loader rebased onto #196: 2,433 passed. Missing the plan's `dealt_cards`, `revision_cards`, `assigned_considerations` properties; being added before its pull request opens.
- Review bot: #195 and #196 approved.
- Work item 6, branch `william/sdf-runner-and-event-log` (`fdfc0fd`, `67b680c`, on the loader branch): `ModelRequest`, outcome types, step interfaces (`update_steps.py`), `DocumentEventLog` and fold (`document_event_log.py`), `StepRunner` (`step_runner.py`); draft converted to `DraftStep` with `run()` as a thin wrapper. 23 new tests (runner, draft resume, event fold); suite 2,456 passed; equivalence test unchanged. The event log is `document_events.jsonl` in the run folder; the equivalence test snapshots only `layer1` to `layer4`, so it does not see it.
- Loader: read-only `variables`, `dealt_cards`, `revision_cards` (keys in `compose_prompts.REVISION_AXES`), `assigned_considerations` added (`1554534`, 41 tests); suite 2,474 passed. Draft #197.
- Real-run replay, origin/main vs the runner branch: 355 requests rebuilt across 7 current-layout runs, 0 differences; origin/main matched `model_calls.jsonl` on 355 of 356 (one draft request in `2026-10-24_03-02_one-objective-considerations-generation` was not rebuilt on either). Drivers unchanged. Output: `data/sdf_real_run_request_replay_runner_and_event_log_against_origin_main_2026-11-13_19-00/`.
- Review bot: #194 approved.
- Audit equivalence test, branch `william/sdf-audit-equivalence-test` (`37b468e`, plus a `CLAUDE.md` rule line), rebased onto #197: 7 model-judged audits plus the persona input builder, on the five-layer run, the pinned pre-renumber run (12 documents) and a generated current-layout run; 24 recorded files, about 400 KB; recordings identical to what `origin/main` produces. Suite 2,500 passed. Draft #198.
- Work item 5 started (two stacked branches: evals and publisher; website and viewer), with byte-identity checks on publisher dry-run staging, website build and offline eval output.
- Work item 7a, branch `william/sdf-score-and-gate-steps` (`b4f520e`, `4f76097`, on the runner branch): `ScoreStep`, `ScoreGateStep`, `NearDuplicateCullStep`; `run()` a thin wrapper. 18 new tests; suite 2,475 passed; equivalence test unchanged. Today's order before the cull confirmed: already-scored documents first, then this invocation's, each in `rewrites` order (so resumed runs can keep a different twin; deal order not adopted here).
- Work item 7b (critique and revise) started in parallel off the runner branch, to be stacked after 7a.
- Review bot: #197 and #198 approved (all five open drafts, #194 to #198, approved).
- Work item 7b, branch `william/sdf-critique-and-revise-steps` (`e68619c`, `504d17f`): `CritiqueStep`, `ReviseStep`, `CombinedReviewAndReviseStep`; `run()` a thin wrapper; 22 new tests; suite 2,478 passed on the runner base; equivalence test unchanged. Being rebased onto 7a (conflict in the event fold test, handed back to its subagent).
- Work item 7c (plan step) started from the 7a branch.
- Work item 7b rebased onto 7a by its subagent (`fe30eb5`, `36fa6b7`); 2,497 passed.
- Work item 7c, branch `william/sdf-plan-step` (`1f84bae`, `22c35cc`, `1104573`): `PlanStep`; added `DocumentStep.stage_file_record_for_removal` so an incoherent plan is still written and checkpointed; 8 new tests; 2,483 passed on the 7a base.
- Work item 5, evals part, branch `william/sdf-evals-read-through-document-state-loader` (one squashed commit): `audit_sdf.py` description rewrite audit, `report_sdf.py`, shoehorning postprocess read through the loader; loader extensions (plan record variables, planned description, draft text, plan-shaped records, empty list for a bare corpus path). Publisher dry-run staging, offline audit and report output and the website build byte-identical on all 7 committed SDF runs (capture scripts in the session scratchpad). 2,526 passed. Draft #199.
- Restacked onto #199: runner 2,549; score and gate 2,568; critique and revise 2,590 passed. Runner pushed, draft #200. Plan step rebase conflict (event fold test) handed back to its subagent. Real-run replay running for score and gate and critique and revise.
- Work item 7d (description rewrite) running from the 7b branch.
- Real-run replay of score and gate and critique and revise: 355 requests each, 0 differences from origin/main. Output: `data/sdf_real_run_request_replay_score_and_gate_steps_and_critique_and_revise_steps_against_origin_main_2026-11-13_18-06/`. Drafts #201 (score and gate) and #202 (critique and revise) opened.
- Plan step rebased onto #202 by its subagent (`161bd6a`, `b5faec3`, `7a25b5a`); 2,598 passed. Replay running.
- Work item 7d, branch `william/sdf-description-rewrite-step` (`49cef29`, `19cf238`): `DescriptionRewriteStep` for both routes; shared additions: the same removal hook, OpenAI route in `send_model_request`, `after_documents_are_processed`, `message_when_failed_documents_stop_the_run`, opt-in `build_request_errors_count_as_thrown_calls`. 40 new tests; 2,537 passed on the 7b base. Being rebased onto the plan step.
- Work item 7e (consideration scoring and assignment) running from the plan step branch.
- Review bot: #199 to #203 approved.
- Plan step replay: 355 requests, 0 differences (`data/sdf_real_run_request_replay_plan_step_against_origin_main_2026-11-13_18-08/`). Draft #203.
- Description rewrite rebased onto the plan step (`a07268e`, `9e3e1dd`; one copy of the removal hook); 2,638 passed; no conflict markers in the branch or its history (checked). Replay: 355 requests, 0 differences (`data/sdf_real_run_request_replay_description_rewrite_step_against_origin_main_2026-11-13_18-09/`). Draft #204.
- Running: work item 7e (consideration scoring and assignment, on the plan step, to be rebased onto #204), 7f (screen and redeal, on the plan step), and the request-settings follow-on (on the plan step).
- Work item 7e, branch `william/sdf-consideration-scoring-and-assignment-steps` (`1832902`, `584c07c`, tests): `ConsiderationScoringStep` with up to `max_scoring_attempts` attempts for unusable answers (runner gained attempts support), `ConsiderationAssignmentStep` as a corpus step; 19 new tests; 2,618 passed on the plan step base. Sent back: it wrote score rows in input order instead of as answers arrive, which would lose and re-bill finished paid scores after a crash with several workers; being restored behind an opt-in runner option, then rebased onto #204 with overlapping hooks unified.

## Deviations from the plan

- The paired-test branch is not in the stack (William's decision); the stack is built only on `main`.
- The cleanup branch keeps the unused `outputs.sdf_final` and `outputs.dad_final` config keys: removing them changed the recorded configuration.
- Recordings store hashes of message texts and long values instead of full text (William's decision), and repeated values once per file.
- To keep the 10 MB recording out of `main`'s history without a force push, the stack is pushed under new branch names with history that never contains it; #193 is replaced.
- Work item 6: steps take plain record dicts, not `DocumentState`, until the layers before them are steps; `DocumentStep` gains `stage_file_record` and `progress_line`; the two "run stops" failure rules are one value; the event log stores the stage-file record and request hash, not the reply text.
- Stack order: the runner (work item 6) was built on the loader before work item 5; it is rebased after work item 5 when that lands.
- The audit equivalence script has no guard against Python differing from `origin/main` (the cleanup edits audit code below it); its recordings were checked against an `origin/main` worktree by hand instead.
- The audit test covers more than the plan named: compliance, score, `audit_sdf --patterns --principles` and `diversity --ideas` as well.
- Work item 7a: added `run_corpus_step` (a function, not part of `StepRunner`); the event fold now lets a later accepted event lift an earlier removal by the same step (needed when a new gate threshold is applied); gate and cull events have no request hash.
- Work item 7b changes run behaviour without changing requests: critiques for all documents now run before any revise (they were interleaved per document), so printed lines are reordered and, with several workers, a revise no longer overlaps the next document's critique. Prompt-rendering errors now raise instead of counting as a thrown call. The old layer's systemic stop (thrown calls across both calls together) is kept by counting in `run()`.
- Work item 5 left these reading files: the Hugging Face publisher (loader-derived corpus reorders rows of two resumed committed runs, which would change published bytes); the final-corpus audits and persona builder (bare corpus paths; judge order pinned); screen, description rewrite and selector batch tables; `scripts/rerun_considerations_selector_on_stored_plans.py` (builds model requests, no test covers it); `website/sdf.py` and `viewer/loader.py` (see judgement items). The website-and-viewer branch has no commits and no pull request.
- Shoehorning postprocess: its "no drafts found" console message now names the run's draft stage instead of a file path.
- The plan's `Removed` outcome alone could not reproduce today's files for plan (incoherent) and description rewrite (dealt-value conflict): both are written to the stage file and checkpointed today, so a `stage_file_record_for_removal` hook was added.
- `ConsiderationAssignmentStep` lives in `layer1_select_considerations.py`, not `considerations_assignment.py` (circular import otherwise).

## New risks

| Risk | Checked? | How to check if not |
|---|---|---|
| Recorded requests depend on unpinned library data: Faker name pools; scipy and numpy behind the assignment tie-break and the near-duplicate cosine | Faker: being fixed by patching fixed pools. scipy and numpy: not checked | Run the test under the oldest and newest allowed scipy and numpy, or pin them in `requirements.txt` |
| The regenerate guard passes when `origin/main` cannot be resolved | Being fixed | |
| The guard blocks any intended request change that touches Python until it is on `main` (affects the paired-test branch when it lands) | Confirmed on the paired-test branch | |
| A converted layer's `run()` writes `document_events.jsonl` into the parent of whatever stage folder it is given, so scripts that call layer functions with a scratch folder (layer 3 rerun setup, smoke test, paired test, the replay drivers) get an extra file beside it | No | Grep this project's scripts for calls to layer `run()` functions and check what folder they pass; run the replay (in progress) |
| The event log drops reply text, so work item 9 (log as source of truth) could not rebuild stage files that hold reply-derived fields without re-parsing | No | Before work item 9, list each stage file field and whether the accepted event's stored record holds it (it holds the full record today) |
| The equivalence test ignores files outside `layer1` to `layer4`, so a change to `run_manifest.json`, `cost_log.jsonl` or `model_calls.jsonl` format would pass unnoticed | No | Check whether those files' formats are compared anywhere; add them to the snapshot if a later step touches how they are written |
| The real-run replay covers only stages it can rebuild: the show-rejected-values runs fail in five stages on both checkouts (that feature is not on `main`), and some stages of other runs are not rebuilt, so 355 of the runs' requests are covered, not all | Partly (coverage counted) | Before each later pull request, compare the replay's per-stage counts with the runs' `model_calls.jsonl` counts and list uncovered stages in the pull request |
| A refactor pull request could regenerate the audit recordings without the script stopping it | No | Review rule only; add the same `origin/main` guard once the cleanup has merged and audit code below the test no longer differs |
| Audit equivalence does not cover `variable_persistence_sdf` and the persona builder on the five-layer layout (they stop on records without `variables`), so a loader change there goes untested | Recorded that they stop | Decide whether those readers should support five-layer runs at all |
| Work item 7b's two-phase layer 3 is slower in wall-clock time when one document's revise could have overlapped another's critique | No | Time a fixture run with several workers and a canned reply delay on the runner base and on 7b; if it matters, run critique and revise for one document inside one worker task while keeping two steps |
| Loader-derived final corpus order differs from `final/sdf_corpus.jsonl` for resumed runs (two committed runs) | Yes, found by the work item 5 publisher comparison | Relevant to work item 9 and risk 11: decide whether derived corpora must keep file order |
| Description rewrite: a dealt-value conflict now writes the rejection record before the results record (was after); a crash between the two writes would now record the rejection twice on resume instead of never | No | Kill a fixture run between the two writes (monkeypatch an exception after the first) and check the files on resume; dedupe rejections by prompt id on resume if it matters |

## Needs William's judgement

- How the regenerate guard should treat an intended request change that touches Python (options in #193's description: make the guard informational; keep it and land such branches only after `main`; or accept red checks).
- Website: `website/sdf.py` promises a standard-library-only build (no imports from `shared/` or `viewer/`), and the loader imports `shared/` and `sdf_pipeline/compose_prompts`. Options: relax the website's promise; split the loader's pure parsing into a module with no pipeline imports that the website can use; or leave the website reading stage files (current state).
- Viewer: `viewer/loader.py` serves raw stage records and per-stage joins that document states do not carry. Moving it means extending the loader substantially, with no equivalence test for the viewer. Options: extend the loader and add a viewer output test; or leave the viewer reading files.
- Work item 7b changed layer 3's call and printed-line order (all critiques before any revise). Accept, or ask for per-document pipelining inside the two steps.
