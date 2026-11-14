# SDF document state redesign: cleanup plan

## What was measured

Every piece of code the redesign could let us remove or simplify, measured on the top of the stack (`william/sdf-dealt-value-identifiers`, `3683685`) against `origin/main`. Four read-only passes covered: the eleven cut candidates listed after the stack was finished, and a search for anything else the steps, runner, loader and seeding now make redundant. For each item: lines removable (code and tests separately), everything that depends on it (pipeline code, tests, `evals/`, `experiments/`, `viewer/`, `website/`, and this project's `scripts/`), what has to happen first, which decision it waits on, and what could change without either equivalence test noticing. Line counts are from line ranges that were read, not from a diff of a finished cut. Each measurement report keeps the file and line citations, with inferred claims marked (paths at the end).

## How every cut is checked

- The SDF prompt-equivalence test and the audit equivalence test pass **without regenerating** their recordings. Together they show that no request, stage file, checkpoint or final corpus changed.
- The full test suite passes.
- The real-run request replay rebuilds the same 355 requests, with 0 differences from `origin/main`.
- Neither equivalence test sees the behaviours listed below. A cut that touches one of them needs its own test first, or a named existing test that pins it:
  - printed lines and exit messages
  - the layer 3 systemic stop (thrown calls counted across critique and revise)
  - the description rewrite's held-out stop
  - early returns when a stage is disabled or its template is missing
  - resuming the screen in the middle of a round
  - resuming a run with no event log

## Where the work goes

Pipeline cleanups go on new draft branches stacked on #209, one draft pull request per row group below, each based on the one before. Changes to this project's scripts go on a branch in this project. Nothing is merged.

## Ready now: no decision needed, no request or output file changes

| # | Cleanup | Code lines | Test lines | What has to happen first | Risk |
|---|---|---|---|---|---|
| 1 | **Fix stale docs.** `update_steps.py` still says only `DocumentStep` runs through the runner. The five layer docstrings call `run()` a thin wrapper kept for scripts. `CLAUDE.md` and `prompts/README.md` name a `layer5_score.py` and a layer 5 cull, neither of which exists | about 12 edited | 0 | none | none |
| 2 | **Delete dead code.** `score_document` in `layer1_select_considerations.py` (only its own two tests call it; the step re-implements it). `BatchRunResult`'s pending, finished and failed batch id lists (nothing reads them) and `StepRunResult`'s accepted and removed id lists (read by one test) | about 50 | about 31 | none | very low |
| 3 | **Tidy the runner.** The changes: (a) `StepRunner` builds the event log itself, replacing seven identical constructions; (b) one helper turns an outcome into an event, replacing three copies; (c) a step declares its stage-file id field, so four layers no longer build their own lookup; (d) two runner options, "mark nothing" and "also finished ids", replace the adapter classes `_NoCheckpoint` and `_CritiqueCheckpointView`; (e) shared helpers replace the failure text and the throttled progress counters repeated across layers | about 95 | 0 | none | low. Printed text must stay word for word; step tests assert printed lines. Checkpoints are in the recordings, so (d) shows up in the equivalence test if it goes wrong |
| 4 | **Merge duplicated test helpers.** Small helpers repeated across the new test files (`events_of`, `reply`, `_records`, `_rows`, `run_main`). In the audit equivalence support, the fake Anthropic and OpenAI clients and the difference functions are near copies of the prompt-equivalence ones | 0 | about 105 to 125 | none | low. Recorded files must not change; only the regenerate scripts may write them |
| 5 | **Prune old per-layer tests.** Remove the tests a named new test or equivalence scenario already covers: `test_sdf_layers.py` about 411 of 693 lines, `test_layer1_screen.py` 117, `test_layer1_description_rewrite.py` 93, `test_layer1_select_considerations.py` 64, plus about 63 in the audit tests. Keep the unique ones: sign-off truncation guards, rendering an old prompt snapshot, the audit report tables, old-run compatibility, deal column-count invariants, crash windows, and all of `test_layer3_templates.py` | 0 | about 750 | First move the helpers and fixtures that 10 new test files import from the old files. Recheck the description rewrite and selector classifications, which were partly made from test names | medium. The equivalence fixture uses one prompt set, 3 to 12 hands and English only, so a wrongly removed edge-case test would not be missed by anything |
| 6 | **Delete the pipeline's `scripts/rerun_considerations_selector_on_stored_plans.py`.** `run.py --seed-from <run> --from-step consideration_scoring --use-current-prompts` covers its calls, and the request print command covers its dry run. Nothing calls it | 192 | 0 | Decide whether its comparison table against the original coverage report (about 20 lines) is wanted; if so, move it into a small comparison script | low |
| 7 | **Stop writing the old layouts.** Remove the old-layout entries from `shared/utils.py`'s stage-folder and template tables (`sdf_stage_dir`, `sdf_template_path`) and the pre-snapshot fallback in `run.py`. Reading old runs stays in the loader. All 7 committed runs are old layout, and each has a final corpus, so none needs to resume | about 60 | part of about 130 to 150 | Give `seeding.py` its own read-only resolver for source runs. Make `--resume` on an old-layout run refuse with a clear message (otherwise it would quietly start a second set of `layer1` to `layer4` folders) | low |
| 8 | **Delete `set_up_generation_run_from_saved_consideration_scores.py`** in this project. It hand-builds a run from another run's consideration scores. No script, test or document uses it | 49 | 0 | Decide whether to drop the ability outright (recommended) or first teach seeding to take assignments from a separate scores file (30 to 45 lines plus tests) | low |
| 9 | **Delete `set_up_layer_3_rerun_on_selected_drafts.py`** in this project. `--seed-from <run> --from-step critique --use-current-prompts` sends the same requests, and a pipeline test proves it. Keep that test (it is the only proof) and reword its docstrings | 39 | about 2 reworded | Optional: a `--seed-only` flag (about 6 lines) for its one missing ability, setting up a run folder without running the step | low |

Ready-now total: about 550 code lines and about 900 test lines removed, plus about 110 lines of edits.

Suggested order: 1 and 2 together (docs and dead code), then 3, then 4 and 5 (tests), then 7, with 6, 8 and 9 alongside as their decisions come in.

## Waiting on a decision

All open decisions, including these, are now kept in one list: `sdf-document-state-redesign-decisions-for-william_2026-11-13_20-10.md`.

| Cleanup | Size | Waiting on | Notes |
|---|---|---|---|
| **Remove the unused `outputs.sdf_final` and `outputs.dad_final` config keys** | 4 config lines and 1 recording line | The regenerate guard (decision 1) | No code reads them. Simplest is its own small change on `main` after the stack lands. The 20 committed run manifests keep them as historical snapshots |
| **Retire the single-call layer 3 path** (`sdf.revision.split: false`, `CombinedReviewAndReviseStep`, `prompts/sdf/layer3.txt`) | about 55 to 60 code, 110 prompt, part of 130 to 150 test lines, 2 fixtures | **New decision:** three experiment arm files use `split: false`, so retiring it stops those experiments being re-run. Also blocked by decision 1, because the audit equivalence test builds its current-layout run from the `revision_in_one_call` scenario and 8 recordings would change | No committed run uses it |
| **Retire per-layer checkpoints** (work item 9) | about 105 code, about 25 docstring lines; `shared.utils.Checkpoint` stays because DAD and the preference pipeline use it | Decision 7, plus a policy for the 7 committed runs, none of which has an event log, plus decision 1 for regenerating | The event log lacks three things today: the `variables` of newly dealt hands; the `variable_lines` that `screen_rejections.jsonl` needs; and an event for "batch fully written". The "disagree marks" equivalence scenario needs redesigning, not deleting. Unused production code backing this work (`DocumentState.last_failure`, `fold_round_and_batch_plan_events`, `document_state_final_corpus.py`, up to about 75 lines) goes if decision 7 is no |
| **Move the website onto the loader** | about 116 lines of parsing; about 60 to 75 net after adapters; about 64 test lines rewritten | Decision 2 | Importing the loader pulls in no third-party package, so the website's standard-library-only promise holds. Its "no imports from `shared/`" promise breaks, through `compose_prompts`, `considerations_catalog` and `shared.matrix`. About 97% of the loader is pure parsing, so splitting it into a module with no pipeline imports is cheap. The loader would also need matrix weights, per-file record counts, and tolerance for bad lines |
| **Move the viewer onto the loader** | about 70 lines; about 30 to 40 net | Decision 3 | The viewer needs raw records and legacy type records the loader does not return, and has no output test. Separately, `viewer/ui_pages/lineage.py:206-232` appears to read rewrite fields (`original`, `rewritten`) that current records no longer have. That is a likely bug to check on its own |
| **Unify the two cheap runner hooks**: make "prompt-building errors count as thrown calls" the default for every step, and replace `failure_when_no_attempt_is_allowed` with a config check | about 8 code, about 20 test | **New decision:** under the default, a broken template becomes a recorded failure instead of a crash | No request changes; run flow does |
| **Unify the three ways a run stops on a systemic failure** | about 25 | No decision. Needs a test of the stop messages and the summary line order first, because neither equivalence test sees them | medium risk |
| **Move DAD onto the runner** | about zero net (about 100 to 150 lines of generic loop out, as much in); DAD already uses the same `parallel_map` | Whether to extend the redesign to DAD | What DAD would gain is the event log and request hashes. It has no equivalence test (one would be about 1,500 lines), and its steps make several calls per item, loop through redrafts and change the request on retry. If done, start with `baseline.py` |

Decisions 1 to 9 are in `sdf-document-state-redesign-code-cuts-and-outstanding-decisions_2026-11-13_19-00.md`. This plan adds four:

10. Whether to keep the single-call layer 3 path for the experiment arms. **Decided 2026-11-13: retire it; do not merge that pull request until William gives the go-ahead (waiting on someone else). See the decisions list.**
11. Whether the selector rerun script's comparison table is worth keeping (item 6).
12. Whether to drop the ability to start a run from saved consideration scores, rather than build it into seeding (item 8).
13. Whether a broken template should become a recorded failure or stay a crash. **Decided 2026-11-13: a recorded failure in every step. See the decisions list.**

## Not worth doing as cleanup

| Candidate | Why not |
|---|---|
| Replace the screen, description rewrite and selector readers with the loader | They total 53 lines, and the loader would grow by 60 to 100 to carry the screen's batch and round tables, the selector catalog and the coverage file. No test covers what the audits and `report_sdf.py` print from these readers, so a snapshot test of their output on the 7 committed runs would have to be written first. Worth revisiting only as part of moving the website or viewer |
| Turn `run.py` into a step list, and remove the layers' `run()` wrappers | `run.py` is four blocks: run each layer if `--layer` starts at or before it. Its resume, stop and manifest logic lives in the stages and in `utils.create_run_dir`. A `Pipeline` class would add about as much as it removes. Only `layer2_draft.run` is thin; the other wrappers (about 360 lines) hold resume, early-return and stop logic, so removing them moves code. About 211 test call sites would need repointing, and they are also the resume and stop tests. The outside callers are `experiments/experiment.py` and one script in this project, `plan_fresh_specifications.py` (which calls `layer1_plan.run`) |
| Retire the paired test's hand-seeding and `experiments/experiment.py`'s `run_arm` | Both are outside the stack. Replacing them needs seeding to gain a stop after a chosen step (25 to 40 lines) and a supplied `variables.txt` (10 to 15 lines), about 60 to 80 lines added to remove about 70. Better done when the paired-test branch lands |
| Fold the OpenAI route into `call_claude` | It changes the OpenAI request: thinking headroom is added to its token limit, so the recordings must be regenerated (decision 1). Only worth doing if that request change is wanted for its own sake |

## Paths

- Pipeline repository: `animal-welfare-data-pipeline/` (GitHub `sentfutures/animal-welfare-data-pipeline`). Top of the stack: branch `william/sdf-dealt-value-identifiers`, worktree `animal-welfare-data-pipeline/.claude/worktrees/sdf-dealt-value-identifiers`.
- Branch `william/sdf-website-and-viewer-read-through-document-state-loader` is not empty: it holds 15 commits from before #199 was rebased, and the top of the stack does not contain them. Reset it onto the top of the stack before using it.
- Measurement reports, with file and line citations: `plans/11-11-sdf-document-state-redesign/sdf-document-state-redesign-cleanup-measurement-reports_2026-11-13_19-39/`
  - `cut_measurement_seeding_scripts_and_run_wrappers_2026-11-13_19-39.md`
  - `cut_measurement_stage_readers_step_list_and_old_layout_resume_2026-11-13_19-39.md`
  - `cut_measurement_old_tests_checkpoints_and_unused_config_keys_2026-11-13_19-39.md`
  - `cut_measurement_website_viewer_dad_and_further_cleanup_2026-11-13_19-39.md`
- Earlier cut list and decisions 1 to 9: `plans/11-11-sdf-document-state-redesign/sdf-document-state-redesign-code-cuts-and-outstanding-decisions_2026-11-13_19-00.md`
- Real-run request replay: `scripts/sdf_document_state_redesign/` in this project.
