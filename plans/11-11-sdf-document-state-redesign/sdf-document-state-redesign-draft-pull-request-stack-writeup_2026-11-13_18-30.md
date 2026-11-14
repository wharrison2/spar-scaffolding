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

## Deviations from the plan

- The paired-test branch is not in the stack (William's decision); the stack is built only on `main`.
- The cleanup branch keeps the unused `outputs.sdf_final` and `outputs.dad_final` config keys: removing them changed the recorded configuration.
- Recordings store hashes of message texts and long values instead of full text (William's decision), and repeated values once per file.
- To keep the 10 MB recording out of `main`'s history without a force push, the stack is pushed under new branch names with history that never contains it; #193 is replaced.
- Work item 6: steps take plain record dicts, not `DocumentState`, until the layers before them are steps; `DocumentStep` gains `stage_file_record` and `progress_line`; the two "run stops" failure rules are one value; the event log stores the stage-file record and request hash, not the reply text.
- Stack order: the runner (work item 6) was built on the loader before work item 5; it is rebased after work item 5 when that lands.

## New risks

| Risk | Checked? | How to check if not |
|---|---|---|
| Recorded requests depend on unpinned library data: Faker name pools; scipy and numpy behind the assignment tie-break and the near-duplicate cosine | Faker: being fixed by patching fixed pools. scipy and numpy: not checked | Run the test under the oldest and newest allowed scipy and numpy, or pin them in `requirements.txt` |
| The regenerate guard passes when `origin/main` cannot be resolved | Being fixed | |
| The guard blocks any intended request change that touches Python until it is on `main` (affects the paired-test branch when it lands) | Confirmed on the paired-test branch | |
| A converted layer's `run()` writes `document_events.jsonl` into the parent of whatever stage folder it is given, so scripts that call layer functions with a scratch folder (layer 3 rerun setup, smoke test, paired test, the replay drivers) get an extra file beside it | No | Grep this project's scripts for calls to layer `run()` functions and check what folder they pass; run the replay (in progress) |
| The event log drops reply text, so work item 9 (log as source of truth) could not rebuild stage files that hold reply-derived fields without re-parsing | No | Before work item 9, list each stage file field and whether the accepted event's stored record holds it (it holds the full record today) |
| The equivalence test ignores files outside `layer1` to `layer4`, so a change to `run_manifest.json`, `cost_log.jsonl` or `model_calls.jsonl` format would pass unnoticed | No | Check whether those files' formats are compared anywhere; add them to the snapshot if a later step touches how they are written |

## Needs William's judgement

- How the regenerate guard should treat an intended request change that touches Python (options in #193's description: make the guard informational; keep it and land such branches only after `main`; or accept red checks).
