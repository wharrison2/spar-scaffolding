# Measurement of cuts 3, 4 and 6 on `william/sdf-dealt-value-identifiers`

Pipeline worktree: `/Users/williamharrison/Desktop/emergent_alignment/animal-welfare-data-pipeline/.claude/worktrees/sdf-dealt-value-identifiers` (all paths below are relative to it unless absolute). Read-only; nothing was edited, run against an API, or committed. "(inferred)" marks anything not directly read. Line counts are physical lines of the ranges cited.

---------------------------------------------------------------------------------

## Cut 3: replace `load_screen_outputs`, `load_description_rewrite_outputs`, `load_considerations_outputs` with `sdf_pipeline/document_state_loader.py`

### Definitions
- `sdf_pipeline/layer1_screen.py:688-716` `load_screen_outputs` (29 lines). Returns hands, rounds, dealt_hands (round 0), screened_ids (hands in a checkpointed batch), removed (rejection records), accepted_ids (None until `screen_summary.json` exists), leftover_ids, batches_total, batches_done. Reads `screen_hands`, `screen_rounds`, `screen_batches`, `_screen_checkpoint.json`, `screened_hands`, `screen_summary.json`, `screen_rejections`.
- `sdf_pipeline/layer1_description_rewrite.py:442-448` (7 lines). Returns `{"results": description_rewrites.jsonl records by prompt_id}` or None.
- `sdf_pipeline/layer1_select_considerations.py:614-630` (17 lines). Returns `{"catalog": load_considerations(run's inputs/prompts snapshot), "rows": consideration_selections.jsonl, "report": consideration_coverage.json or None}`; None without rows or without the snapshot's `considerations.txt`.

### Every caller (searched pipeline `evals/ scripts/ viewer/ website/ shared/ tests/ experiments/`, and outer `scripts/`, `plans/`, `.claude/skills`)
Production:
- `evals/audit_sdf.py:192` (`audit_screen`), `:243` (`audit_description_rewrite`), `:270` (`audit_considerations`).
- `evals/report_sdf.py:327`, `:370`, `:398` (module-level HTML report code).
Tests:
- `tests/test_layer1_screen.py:574, 594, 605, 612-615, 624` (and the audit tests at 626-650 via `audit_sdf.audit_screen`).
- `tests/test_layer1_description_rewrite.py:300` (+ audit tests 309, 319).
- `tests/test_layer1_select_considerations.py:450-453, 456-470` (+ audit tests 486, 496).
- `tests/test_considerations_selector_smoke.py:131`.
None in `viewer/`, `website/`, `scripts/`, `shared/`, `experiments/`, or in the outer project's `scripts/` (searched, zero hits). The downstream pure helpers `removal_rate_rows` (`layer1_screen.py:719`), `round_rows` (749), `share_shift_rows` (759), `rewrite_share_rows` (`layer1_description_rewrite.py:451`), `corpus_coverage_report` (`layer1_select_considerations.py:633`) take the loaders' dict shapes, so they stay or must be re-pointed at state objects. The outer script `scripts/spec_contradiction_screen/analyze_screen_runs.py:24` only copies a constant.

### What the loader would have to gain
The loader (`sdf_pipeline/document_state_loader.py:263-300`) already reads `screen_hands`, `screen_rounds`, `screen_judgments`, `screen_rejections`, `screened_hands`, `screen_summary.json` (leftover ids), `description_rewrites.jsonl`, `consideration_scores`, `consideration_selections`. Gaps against the old readers:
1. Screen batch table (`screen_batches.jsonl`) and checkpoint (`_screen_checkpoint.json`): needed for `batches_total`, `batches_done`, and `screened_ids` as "in a checkpointed batch". `ScreenJudgment` (`document_state.py:181`) carries `batch_id` and round, so `screened_ids` is probably derivable as "has a judgment" (inferred, not proven equal: a batch checkpointed with zero judgments, or a judgment dropped by `_drop_batch_records`, would differ). `batches_total` has no source but the batches file.
2. Round table notes: states carry `deal_round_number` but not each round's `note` (used by `round_rows`, `audit_sdf.py:204-205`, `report_sdf.py:354`). Needs the rounds list on `LoadedDocumentStates` (it already holds `run_notes`; it would gain a `screen_rounds` field).
3. `accepted_ids` / "screen finished" flag: today `summary_path.exists()`; the loader reads `screened_hands` and `screen_summary.json` internally (`:272-275`) but does not expose them. Needs exposure.
4. Hands that the screen removed: `removal_rate_rows` needs every hand ever dealt (including hands with a rounds record), with variables, as `dealt_variables`; present. Order matters for sort ties only after `sort`, so safe (inferred).
5. Rewrite table: `rewrite_share_rows` needs `rewrite_needed`, `dealt_value_conflict`, and `changes` per plan. `DescriptionRewriteDecision` (`document_state.py:201`) has `verdict` and `changes`; the three verdicts probably map to rewritten, declined and conflict (inferred; check enum). `rewrite_share_rows` also needs "checked" = has a result row, which is "has a decision". Mostly a re-expression, little new reading.
6. Selector tables: `corpus_coverage_report` needs the catalog loaded from the run's own `inputs/prompts` snapshot and the layer-1 `consideration_coverage.json` report. Neither is read by the loader. Assignment rows are already in the states (`considerations_assignment`) so `rows` can drop out. Needs catalog loading (pulls `sdf_pipeline/considerations_catalog` and `shared/` into the loader's imports) and a `consideration_coverage.json` read.
7. Layout: the loader's `_build_five_layer_states` (`:517`) has none of these tables; the old readers return None for the five-layer layout (no `layer1/screen_hands.jsonl`), so the new path must return the same "none" for all three on five-layer runs.
8. Behaviour to keep: old readers return None for a folder that is not a run, a run without the stage, or `does_not_exist`; the loader's `load_document_states_or_empty_list_when_not_a_run_folder` covers the first and last (`document_state_loader.py:123`).
Estimated loader growth: 60-100 lines plus 3-5 new `LoadedDocumentStates` fields (inferred).

### Does the audit equivalence test cover each caller's output?
No for all three. `tests/test_sdf_audit_equivalence.py` records only judge requests (`tests/sdf_audit_equivalence_fixture_support.py:1-60`; no stdout or report capture was found by grep for `capsys|stdout`). `audit_screen`, `audit_description_rewrite`, `audit_considerations` run inside `evals.audit_sdf` main but send no model request, so their printed output and `audit/report_content.json` fields are not compared. `evals/report_sdf.py` is not in `support.AUDITS` at all (`:136-170` lists plausibility, shoehorning, variable_persistence, compliance, score, audit_sdf, diversity, persona input builder). What does cover them today: unit tests on synthetic folders (`tests/test_layer1_screen.py:541-650` and the rewrite and selector equivalents), plus `tests/test_document_state_loader_on_committed_runs_and_damaged_run_folders.py` for the loader's basic state, which does not check these aggregates.

### Lines removable
- Code: 29 + 7 + 17 = 53 lines of definition, plus the three call-site lines in each of two evals (6). Gross about 59; the loader grows 60-100 (inferred), so net is about zero to +40 lines. This cut does not shrink the code.
- Tests: `tests/test_layer1_screen.py:612-615` (4), `tests/test_layer1_select_considerations.py:450-453` (4) go away; the other loader-calling tests (about 40 lines in `test_layer1_screen.py:569-625`, 456-470 in selector tests, 285-305 in rewrite tests) have to be rewritten on the loader, not deleted. Net removable about 10-20 lines; added tests for the new loader fields would be 50+ (inferred).

### Dependents, prerequisites, decisions
- Dependents: `evals/audit_sdf.py`, `evals/report_sdf.py` and the four test files above. Nothing in `viewer/`, `website/` or the outer project.
- Prerequisite A (the real one): a test that pins the current output of `audit_sdf` sections and `report_sdf` HTML on one run per layout, written against the old readers before the switch. Neither audit equivalence nor prompt equivalence provides it.
- Prerequisite B: the loader gains items 1-7 above first, with a test per item comparing against the old reader on the committed runs (`outputs/sdf/runs/`: three layouts, 7 runs) and on the equivalence fixture's current-layout run.
- Decisions: none from the outstanding list. One new one: should the loader import `considerations_catalog` (decision 2 on the website's standard-library promise is not triggered because the website does not use these functions, but the loader's import weight grows).

### Risk not caught by either equivalence test
- Everything in the printed audit and the report HTML for these three sections (counts, rates, sort order, rounding) is unchecked.
- `screened_ids` semantics (checkpointed batch vs has judgment), `batches_done` on an interrupted run, and `accepted_ids is None` before the screen finishes (changes whether the share-shift table prints) are edge cases the fixtures do not hit.
- Pre-renumber runs: `utils.sdf_stage_dir` resolves `layer12` for the old readers; `load_considerations_outputs` on old runs reads the snapshot catalog; behaviour for `matrix100-cli` and `fullscale-500-opus5` should be compared directly.

### Steps
1. Write a snapshot test of `audit_sdf.audit_screen/audit_description_rewrite/audit_considerations` printed output and report dicts, and of the three `report_sdf.py` sections, on all 7 committed runs (expect None/skip lines on the five-layer ones). Record before changing anything.
2. Add to the loader and `LoadedDocumentStates`: screen rounds with notes, batch count and done count, screen-finished flag, accepted ids, and the selector catalog and coverage report. Test each against the old reader on the 7 runs and the fixture run.
3. Re-express `removal_rate_rows`, `round_rows`, `share_shift_rows`, `rewrite_share_rows`, `corpus_coverage_report` on states (or build the old dict shapes from states in one adapter function, which keeps the helpers and their tests unchanged and makes this a much smaller change).
4. Switch `evals/audit_sdf.py:192,243,270` and `evals/report_sdf.py:327,370,398`; re-run the step 1 snapshot test unchanged.
5. Delete the three functions; move or rewrite the unit tests; delete `utils.sdf_stage_dir` use in `layer1_*` readers.

---------------------------------------------------------------------------------

## Cut 4: `run.py` and the screen loop into a step list (the plan's `Pipeline`)

### Today's `sdf_pipeline/run.py` (209 lines; `git diff origin/main` adds 65 and removes 5)
- Argument handling `:57-98`: `--config --resume --layer --label --run-id`, the #208 seeding flags (`--seed-from --from-step --document-ids --use-current-prompts --revision-override-file`, `:65-77`), key-source flags, exclusion checks (`:82-91`), and a `--layer` range check that rejects 5 and explains the renumber (`:93-98`).
- Run folder choice `:107-136`: seeded (`seeding.seed_run_from_chosen_step`, returns `start_layer`), `--resume` (`utils.resolve_run_dir`, `warn_if_backend_changed`), or new (`utils.create_run_dir` writes the manifest with config and prompt and constitution snapshots; `shared/utils.py`, not run.py).
- Prompt snapshot `:140-143` with a "pre-snapshot run" fallback to live `prompts/sdf`.
- `api.init` and `costs.mark` `:145-148`; stage dirs via `utils.sdf_stage_dir` `:152-156` (old layout honoured).
- Per-layer ordering `:161-198` as four `if start_layer <= N` blocks: layer 1 (screen -> plan -> description rewrite -> selector, four printed headers; `require_openai_key_if_configured` first); layer 2 draft; layer 3 critique and revise (inside `layer3_rewrite.run`); layer 4 score and gate. When `start_layer > 1`, the earlier outputs are re-read from stage files: layer 2 re-reads `plans.jsonl`, applies `load_rewritten_plans` and `attach_selections` (`:177-180`); layer 3 re-reads `drafts.jsonl` (`:187`); layer 4 re-reads `rewrites.jsonl` (`:194`). Each layer prints "Running cost".
- Resume: there is no resume logic in `run.py`; every stage's `run()` reads its own checkpoint and stage file (`layer1_screen.py:522-654` screen, others via `StepRunner`).
- `--layer N` restarts: only the `start_layer` guard above; stages from N upward run and resume from their own files.
- `--seed-from`: seeding writes stage files for the chosen documents and returns `start_layer`; then the same guards apply.
- Stop rules: none in `run.py`. They are `SystemExit`s inside the stages (systemic-failure stops in `StepRunner`, failed-batch stop in `BatchStepRunner`, held-out stop `layer1_description_rewrite.py:~437`, selector stop in `layer1_select_considerations.py:~584`, `layer3_rewrite.py:~365`).
- Manifest: `utils.create_run_dir` at `run.py:128`; seeding writes its own record (`seeding.py:~276`). `run.py` writes none itself. Closing cost reports `:200-205`.
- Screen loop `layer1_screen.py:522-654` (133 lines): disabled/skipped/complete early returns, round 0 deal, then a `while` loop of plan batches -> `BatchStepRunner` -> `load_rejections` -> redeal (`RedealStep`) -> round and hands files -> events, up to `sdf.screen_redeal_rounds`. `_finish` (656-685) writes `screened_hands.jsonl` and `screen_summary.json`.
Glue per stage (`run()` wrappers): `layer2_draft.py:124-129` (6 lines), `layer4_score.py:155-215` (61, contains the gate and cull and corpus assembly), `layer1_plan.py:104-147` (44, includes composing prompts), `layer1_select_considerations.py:551-595` (45), `layer3_rewrite.py:327-369` (43, includes systemic guard).

### What a step list would remove and add (estimates, inferred)
- Remove from `run.py`: the four layer blocks `:161-198` (38 lines), the three re-read blocks (`:176-180, 186-187, 193-194`, 9 lines) and most of the layer-1 headers: about 45-50 lines. Remove from `layer1_screen.run`: the `while` loop bookkeeping if rounds become a `Repeat` step (about 40-60 of 133 lines; the redeal and round-file writing must still exist).
- Add: a `Pipeline` class (ordered steps, `from_step`, repeat group for screen and redeal, per-step "previous output" handling that today is the stage-file re-reads), a step-to-`--layer` map (`--layer` must keep working), seeding returning a step instead of a layer, per-step binding of prompts and event log (each wrapper builds `DocumentEventLog` itself today). Adds about 120-180 lines plus tests of 150+ lines (inferred). Net code change is roughly flat to slightly positive; the benefit is structural (enables cut 5 wrappers going away and cut 8 checkpoint retirement), not size.
- The stage `run()` wrappers are what make a step list awkward: scripts import them (12 scripts; cut 5), and three of them own non-step logic (plan composes prompts, selector writes coverage files, score writes the final corpus), which has to move into steps first.

### How the equivalence tests would see it
- Prompt equivalence (`tests/test_sdf_prompt_equivalence_and_final_corpus.py`, support `tests/sdf_prompt_equivalence_fixture_support.py`) compares requests per stage and item id, calls per stage, stage files, checkpoints and the final corpus; it runs `run.py` with `PIPELINE_OUTPUT_ROOT` (`:698`). It does not capture printed lines (no stdout capture found by grep; inferred that the printed `[Layer N]` headers and "Running cost" lines are not compared), and per the decisions list call order across documents already differs from `origin/main` for layer 3, so it is order-insensitive across documents. Call order within one stage and the screen's batch-then-redeal sequence is only seen through seeds and per-item request contents.
- What it does not see: print order, cost-report text, exit messages and codes, the order in which the stage files are written relative to events, and anything on `--layer` and `--seed-from` invocations unless a scenario uses them (scenarios `Invocation` lists in `tests/sdf_prompt_equivalence_scenarios.py`; resume invocations exist, `--layer` restarts and `--seed-from` are covered by `tests/test_sdf_seeding_from_a_chosen_step.py`, not by equivalence recordings).

### Dependents, prerequisites, decisions
- Dependents: `scripts/print_full_sdf_request_a_stage_would_send_for_one_document.py` (says its stage selection follows `run.py`), `experiments/experiment.py` (calls `layer3_rewrite.run`, `layer4_score.run`), the real-run replay in `scripts/sdf_document_state_redesign/` (outer project), 12 scripts that import layer modules, `tests/test_e2e_smoke.py`, `tests/test_sdf_seeding_from_a_chosen_step.py`.
- Prerequisites: cut 5 (wrapper removal) is effectively a prerequisite or part of the same change; the loader (and cut 3) for re-reading earlier outputs instead of the stage-file re-reads at `run.py:176-194`; a decision on printed lines.
- Decisions: whether printed output is a contract; decision 4 (layer 3 call order) interacts because a pipelined critique and revise would be a step-list design choice; decision 7 (event log as source of truth).

### Risk not caught by either test
Printed progress and cost lines; exit messages from the stages when invoked via the new loop; `--layer 5` rejection text; `--layer N` start with missing earlier files (today `utils.load_jsonl` returns [] for a missing file; a loader-based step list may raise instead); resume of a run interrupted mid-redeal (screen round and hands file replacement logic at `layer1_screen.py:~598-640`, covered by `tests/test_sdf_screen_and_redeal_steps.py` but not by the recordings); cost log `since` marks; any script that imports `run.main`.

### Steps
1. Decide printed-line contract; if kept, add a captured-stdout recording to the prompt equivalence test first.
2. Move non-step logic out of wrappers (plan prompt composition, selector files, final corpus) into steps (this is cut 5).
3. Define `Pipeline` with the layer-to-step map and `from_step`; keep `--layer` as an alias table.
4. Express the screen and redeal as a repeat group; keep `plan_round_batches` and the rounds files.
5. Replace `run.py:161-198` with `Pipeline.run`; replace the stage-file re-reads with loader reads.
6. Re-point `print_full_sdf_request...` and `experiments/experiment.py` at steps; run both equivalence tests unchanged.

---------------------------------------------------------------------------------

## Cut 6: drop resume support for pre-renumber runs and the single-call layer 3 path

### Old-layout code in writers and resume (read paths stay)
`shared/utils.py`:
- `sdf_stage_dir` `:79-91` (13 lines): decides `layer12` vs `layer1`; used by writers (`run.py:152-153`, `seeding.py:212, 265`, `experiments/experiment.py:185-187`) and readers (`layer1_screen.py:696`, `layer1_description_rewrite.py:444`, `layer1_select_considerations.py:620`, `viewer/rendering.py`, `scripts/spec_contradiction_screen/plan_fresh_specifications.py` and two generation scripts in the outer project).
- `_SDF_TEMPLATES` and `sdf_template_path` `:94-111` (18 lines): old template names; every writer (`layer1_plan.py:119`, `layer1_screen.py:562`, `layer2_draft.py:55`, `layer3_rewrite.py:276`, `layer4_score.py:59`, `seeding.py:155-167`) goes through it.
- `_SDF_STAGE_DIRS`, `_SDF_STAGE_FILES`, `sdf_stage_file` `:58-76, 114-125`: read path (used by `experiments/compare_arms.py`, `experiments/experiment.py:140,277`, `pref_pipeline/from_sdf_arms.py`, viewer); stays.
`sdf_pipeline/run.py`: docstring `:29-33` (5), `--layer` renumber message `:93-98` (6, could shrink to a plain range error), pre-snapshot fallback `:140-143` (4), the `sdf_stage_dir` comment `:150-151` (2).
`sdf_pipeline/layer3_rewrite.py`: `require_two_call_templates` `:52-61` (10, plus calls `:168, :223`), only fires for run snapshots lacking `layer3a_critique.txt`, i.e. pre-split resumes.
Total old-layout-only writer/resume code: about 13 + 18 + 17 (run.py) + 12 = 60 lines before replacement; about 45 net after substituting plain names (inferred, since the 14 writer call sites need a literal path in place of the resolver).
Not removable, because they are read paths used by finished old runs: `utils.sdf_stage_file`, loader layouts (`document_state_loader.py:87-101, 263-300, 517-...`), `viewer/rendering.py:83-88` template map, `website/`. Note `seeding.py:212` reads a SOURCE run via `sdf_stage_dir`, so today `--seed-from` can seed from a pre-renumber run; dropping the resolver would drop that unless seeding keeps its own read-only layout resolver (inferred: not tested, check `tests/test_sdf_seeding_from_a_chosen_step.py`).

### Single-call layer 3
- `sdf_pipeline/layer3_rewrite.py`: `CombinedReviewAndReviseStep` `:264-308` (45 lines); the `split` flag and else branch in `run` `:328, 351, 361-362` (about 4, plus dedent); docstring `:12, 16-17` (about 4); `require_two_call_templates` message mention `:59`.
- `sdf_pipeline/seeding.py:152, 163` (2 lines: split flag and the single-call template name).
- `prompts/sdf/layer3.txt` (110 lines; also named in `shared/entity_pools.py:32` comment, `prompts/README.md:94-104`, `config.yaml:86`).
- `experiments/experiment.py:125` (`split` arm key).
Code total: about 55-60 lines in `sdf_pipeline/`, plus the 110-line prompt.

### Tests that exist only for these paths (approximate ranges)
- `tests/test_utils.py:491-514` (`stage_dir` and `TestSdfTemplatePath`, about 25 lines; the `sdf_stage_file` tests at 449-488 stay).
- `tests/test_sdf_layers.py`: `single_call_config` `:468-472` (5), single-call tests `:255-271` (17), `:366-380` (15), split-refusal test `:552-560` (9): about 46.
- `tests/test_sdf_critique_and_revise_steps.py` (232 lines): combined fixture `:37-39`, `combined_step` uses at `:90-100, 112-120, 138-160, 168-180, 216-225`: about 40-50 lines edited or removed.
- Equivalence fixtures: two prompt-equivalence scenarios (`tests/sdf_prompt_equivalence_scenarios.py:419-426, 472-473`, about 10 lines) with recordings `recorded_revision_in_one_call_on_origin_main.json` (1,275 lines) and `recorded_temperature_kept_on_every_stage_with_a_one_call_revision_on_origin_main.json` (1,274 lines); assertions in `tests/test_sdf_prompt_equivalence_and_final_corpus.py:272, 291`; `tests/test_document_state_loader_on_equivalence_fixture_scenarios.py:337`.
Test code total about 130-150 lines plus 2,549 recorded lines.

### BLOCKER found: the audit equivalence test's current-layout run uses `split: false`
`tests/sdf_audit_equivalence_fixture_support.py:85` sets `CURRENT_LAYOUT_SCENARIO_NAME = "revision_in_one_call"` and builds its "current" run by running `run.py` on that scenario (docstring `:17-21`). Removing the single-call path means picking another scenario, which changes the corpus the audits judge and therefore the recorded judge requests in `tests/fixtures/sdf_audit_equivalence/recorded_*_on_current_run_*.json` (8 files). The audit equivalence test would need a regenerated recording, which the regenerate guard refuses while pipeline Python differs from `origin/main` (outstanding decision 1). Also the prompt equivalence test's `test_recorded_files_cover...` would need the two recorded files and scenarios removed in the same change, which counts as regeneration. So cut 6 (single-call half) cannot go green until decision 1 is settled, or until it lands after the stack merges.

### Which committed runs use the old layout, `split: false`, or are incomplete
Pipeline `outputs/sdf/runs/` (7 committed runs; checked directory names, manifests and the loader's outcomes):

| Run | Layout | `revision.split` in manifest | Final corpus | Loader outcome counts |
|---|---|---|---|---|
| 2026-07-01_13-49_dev | five-layer (layer1..5) | absent | 7 docs | 7 final, 3 removed |
| 2026-07-01_14-38_english-only-test | five-layer | absent | 6 | 6 final, 4 removed |
| 2026-07-06_22-19_notebook-port-smoke | five-layer | absent | 67 | 67 final |
| 2026-07-08_14-59_allen_dev | five-layer | absent | 47 | 47 final, 6 removed |
| 2026-07-11_20-06_matrix100-cli | pre-renumber matrix (layer12) | absent | 100 | 100 final |
| 2026-07-13_13-18_al-gap-fixes-100docs | pre-renumber matrix | absent | 96 | 96 final, 2 removed, 2 in progress |
| 2026-07-25_15-57_fullscale-500-opus5 | pre-renumber matrix (the website's pinned run) | absent | 477 | 477 final, 12 removed, 11 in progress |

All seven are old layout; none has `split: false` (key absent means the default true; also none has a layer3 two-call split since they predate it, which is moot). All have `final/sdf_corpus.jsonl`, so all finished their last stage; the "in progress" documents (2 and 11) look like documents whose score call failed to parse (the CLAUDE.md website section says ten scoring calls failed to parse on the pinned run), not an interrupted run (inferred, not verified per document). I found no committed old-layout run that is waiting to resume. Experiments: `outputs/sdf/experiments/{exp1,exp1-small,exp1-smoke,exp2,exp2-small,exp2-smoke,split}/manifest.json` mention `"split": false`, but only manifests and comparison files are committed; the arm run folders are not in git (7 run folders only).
Outer project `data/`: four SDF runs, all current layout (`layer1..layer4`, no `layer12`), all with `final/sdf_corpus.jsonl`, none with `split` in the manifest config (`data/spec_contradiction_screen_runs/small_sample_checks/*/pipeline_outputs/sdf/runs/`). `grep` over `run_manifest.json` files in `data/` found no `"split": false` and no `layer12`. Runs on other machines (Mark Stanley's paths appear in the `split` experiment manifest) were not checked and cannot be.

### Config files and fixtures using `split: false`
- `config.yaml:86` has `split: true` (comment mentions `false`); so the committed default is on.
- Experiment arm files: `experiments/split/arms.yaml:50, 59` (control arms), `experiments/length/arms.yaml:50`, `experiments/arms.example.yaml:50`; `tests/test_experiment.py:164-166` asserts a control arm has `split` False; `experiments/README.md:33,39`. The split experiment is a published result whose control used the single call; the `length` experiment file also does.
- Equivalence fixtures: both prompt-equivalence scenarios above and the audit equivalence current run (blocker above).

### Dependents, prerequisites, decisions
- Dependents: `experiments/experiment.py` (an arm with `split: false` would stop working; the `length` and `split` experiments could not be re-run as written), the viewer (reads old layouts only, unaffected), `seeding.py` (source-run layout resolution), outer scripts calling `utils.sdf_stage_dir` (three), real-run replay.
- Prerequisites: decision 1 (regenerate guard) for the single-call half; a decision on the experiments framework; a line in `CLAUDE.md` ("Runs from before the layer renumbering cannot resume").
- Decisions: (a) does any old-layout run need to resume? None found in the repository; confirm for other machines. (b) Does the experiments framework still need `split: false` arms? If yes, keep `CombinedReviewAndReviseStep` and drop only old-layout resume. (c) Which scenario replaces `revision_in_one_call` for the audit "current" run.

### Risk not caught by either test
- The audit test's `current` run is the one place an old layout is not exercised; removing the resolvers would not break either recording, because only `current` is regenerated by a run. But the loader tests read the committed old runs (they stay) and `tests/test_utils.py` pins the resolver, so removal there is caught by those, not by equivalence.
- Silent behaviour change: a resume of an old-layout run would, after the cut, create a second set of `layer1..layer4` folders beside `layer12` (the exact bug the resolver's comment describes, `utils.py:79-91`) unless `resolve_run_dir` or `run.py` refuses with a message. That refusal is new code and is not in any recording.
- A resume of a pre-split snapshot run: `require_two_call_templates` refusal text changes.
- Anything that passes `revision: {split: false}` in a config silently gets two calls (cost and requests change) unless the key is rejected; neither equivalence test sets that key after the fixtures are removed.

### Steps
1. Confirm no old-layout or `split: false` run must resume (ask; check other machines).
2. Decide the experiments question and decision 1.
3. Add the refusal first: `run.py --resume` on a `layer12` folder exits with a clear message (and test), and reject `revision.split: false` with a message naming the removal.
4. Replace `utils.sdf_template_path` and writer-side `sdf_stage_dir` with literal names; keep a read-only layout resolver for `seeding` source runs, `viewer`, `compare_arms`, and `sdf_stage_file`.
5. Delete `CombinedReviewAndReviseStep`, the `split` flag in `layer3_rewrite.run` and `seeding.py:152,163`, `require_two_call_templates` (or reword it), `prompts/sdf/layer3.txt`, the `split` key in `experiments/experiment.py:125` and in the arm files; update `prompts/README.md:94-104`, `config.yaml:86`, `CLAUDE.md`.
6. Remove or replace the single-call tests, the two prompt-equivalence scenarios with their recordings, and switch `CURRENT_LAYOUT_SCENARIO_NAME` to a two-call scenario; regenerate the audit recordings (blocked until decision 1 is resolved) and show that the diff in the eight `current_run` recordings is only the revise-call change.

---------------------------------------------------------------------------------

## Summary table

| Cut | Code lines removable | Test lines removable | Net effect | Blocked by |
|---|---|---|---|---|
| 3 | about 59 gross; loader gains 60-100 (inferred) | about 10-20 net | flat to +40 lines | Needs a new output snapshot test (neither equivalence test covers audit sections or `report_sdf.py`) |
| 4 | about 45-50 in `run.py` plus 40-60 in the screen loop, minus 120-180 added (inferred) | none | flat to slightly positive | Cut 5, printed-line decision |
| 6 | about 105-120 in `sdf_pipeline/` and `shared/` (60 old layout, 55-60 single call), plus 110-line prompt | about 130-150 plus 2,549 recorded lines | smaller | Decision 1 (audit current run uses `split: false`), experiments decision, no old-layout run found that needs to resume |
