# Measurement of cuts 7, 8 and 11 (SDF document state redesign, top branch william/sdf-dealt-value-identifiers)

Read-only. Pipeline worktree: `/Users/williamharrison/Desktop/emergent_alignment/animal-welfare-data-pipeline/.claude/worktrees/sdf-dealt-value-identifiers`, compared with `origin/main` (cc64a74). Paths below are relative to that worktree unless absolute. "Checked" means read in code or run; "inferred" means reasoned from names or structure, not verified. Nothing was edited, committed or run against an API. Scratch helpers (`testlines.py`, `bucket.py`) are in this folder.

Ranking by effort and risk: cut 11 (tiny, one decision) then cut 7 (about 700 lines of tests, no production code, risk of losing edge cases) then cut 8 (about 130 lines of production code but needs the event log completed first and an old-run policy).

---

# Cut 11: remove the unused `outputs.sdf_final` and `outputs.dad_final` keys

## Confirmed unused (checked)
- Grep over the whole pipeline worktree (all file types, `.git` excluded), over `/Users/williamharrison/Desktop/emergent_alignment/scripts`, and over `strictly_implicit_difficult_advice_pipeline_copy`: no Python reads either key. Python that reads `config["outputs"]` reads only `cost_log`: `shared/api.py:145`, `shared/embeddings.py:116`, `evals/diversity.py:781,945`, `experiments/dispositions/make_dispositions.py:231`, and outer `scripts/spec_contradiction_screen/plan_fresh_specifications.py:58,130`. `dad_pipeline/`, `pref_pipeline/`, `viewer/`, `website/` have no hit. Test configs (`tests/conftest.py:447`, `tests/sdf_prompt_equivalence_fixture_support.py:574`) set only `cost_log`.
- The earlier code-quality review already flagged them as dead (`code_quality/findings_v1_2026-07-10.json:929`). Commit 4f2f56d removed them and 1ba2a61 (#195) put them back purely because of the recording, so nothing else depends on them.

## Where they appear (checked)
| Place | Lines | Action |
|---|---|---|
| `config.yaml:148-149` | 2 | delete |
| `config.sonnet-cli.yaml:50-51` | 2 | delete (no test reads this file) |
| `tests/fixtures/sdf_prompt_equivalence/recorded_real_config_unchanged_except_the_prompt_count_on_origin_main.json:75` | 1 line | the only recording that holds the keys (it stores the config used by the real-config scenario) |
| `tests/fixtures/sdf_audit_equivalence/*` | 0 | no hit; the audit test reads `config.yaml` (`tests/sdf_audit_equivalence_fixture_support.py:451`) but its recordings hold judge requests only, so (inferred) no change |
| 20 committed `run_manifest.json` / `manifest.json` files under `outputs/` (for example `outputs/sdf/runs/2026-07-01_13-49_dev/run_manifest.json:44-45`) | 40 | historical snapshots of the config a run used; leave alone. The audit and loader tests read these runs, so editing them would be a needless risk |
| Outside the pipeline: `animal-welfare-data-pipeline-log-probability-persona-check/config*.yaml`, `strictly_implicit_difficult_advice_pipeline_copy/config*.yaml`, `scripts/emergent_misalignment/prompt_corpus_batch_pipeline_config_template.yaml`, 30 `dad-model-experiment-2026-11-01/configs/*.yaml`, `plans/.../*.md` | n/a | separate copies with their own configs; leave (harmless extra keys) |

## Lines removable
Code: 0. Config: 4 lines. Tests: 0 test code; 1 line in one recorded fixture (a regeneration, not a deletion).

## What regenerating the recording involves
- The script `scripts/regenerate_sdf_prompt_equivalence_recorded_requests.py:40-72` lists Python files under `sdf_pipeline/`, `shared/` and `dad_pipeline/` that differ from `origin/main` (committed or untracked). On the stack's top branch it is not empty, so the script exits with "Refusing to regenerate" unless `--allow-python-changes` is given; with that flag it writes the list into every recorded file and `test_recorded_files_were_made_from_unmodified_origin_main_python_code` (`tests/test_sdf_prompt_equivalence_and_final_corpus.py:57-68`) then fails. The script also deletes and rewrites all 33 recorded files (`:77-83`), not just one.
- The guard only looks at those three folders. Cut 11 touches none of them, so the guard is a problem only because the branch carries Python changes.
- The recorded file for the real-config scenario was last changed by d83e4c0 (#194) only (checked with `git log`). The keys are in the recording only because the scenario loads the real `config.yaml` (`tests/sdf_prompt_equivalence_fixture_support.py:581-589`).
- The test fails today if `config.yaml` changes, with "THE CONFIGURATION USED CHANGED" (`:1126-1131`), so removal without a regeneration goes red (expected, not run).

## Options (decision 1)
A. Land the stack, then do cut 11 on `main` as its own change: run the script there (guard passes), commit config plus the one changed file. Lowest risk, no stack rewriting.
B. Rewrite the stack: apply the config change at the #194 commit (Python equals `origin/main` there, inferred from the recorded list being empty), regenerate, check `git diff --stat` shows only the one recorded file (inferred: the run is deterministic), then rebase 15 pull requests. High churn for 4 lines.
C. On the top branch, hand-edit line 75 of that one recorded file and delete the two config keys. Passes both tests (inferred) but breaks the rule that only the script writes recordings.
Recommended: A.

## Prerequisites and decisions
Decision 1 (guard policy) or choose option A. Nothing else.

## Risk the equivalence tests would not catch
Nothing in the pipeline reads the keys (checked), so no behaviour changes. Residual risks: an external script reading `config["outputs"]["sdf_final"]` outside the folders searched (none found in this project; other repositories on the machine not searched), and the persona-check repository copy diverging further from the main config (cosmetic).

## Ordered steps
1. Decide A, B or C.
2. Delete the two keys in `config.yaml` and `config.sonnet-cli.yaml`.
3. Run the regenerate script where the guard passes; confirm only the real-config recording changed (`git diff --stat tests/fixtures`).
4. Run the prompt-equivalence and audit-equivalence tests and `tests/test_api.py` (reads `config.yaml`, line 307-330).
5. Commit config and recording together, per the recording rule.

---

# Cut 7: prune older per-layer tests

## Method and limits
- Read every test body in `tests/test_sdf_layers.py`; read the screen tests from line 170 to 300 and skimmed the rest by name; description-rewrite and selector tests were classified from names plus the bodies of the new step tests (inferred where noted). The audit test files were classified from the names and from the header of `tests/test_sdf_audit_equivalence.py` (checked).
- What the new tests cover (checked): the prompt-equivalence test runs `sdf_pipeline/run.py` end to end for 33 scenarios and compares every request (all settings, schema, hashes of every message), every stage file in layers 1-4 (field order and values), checkpoint files, the final corpus, exit messages and calls per stage per invocation (`tests/sdf_prompt_equivalence_fixture_support.py:1-80`). It does not compare printed lines (step tests do), and it only uses one prompt set and tiny fixtures (3 to 12 hands). The audit equivalence test compares judge requests only, with canned good replies, so parsing, resume refusal and aggregation of audits are not covered by it.
- Baseline (checked): the 13 older and new SDF per-layer and step test files, 344 tests, all pass offline (`pytest -p no:cacheprovider`). `coverage` and `pytest-cov` are not installed, so no line coverage was measured, and I did not run mutation checks (that would need edits). "Fully duplicated" below therefore means "a named test asserts the same thing", not "proven by mutation".
- **Helper dependency (checked):** new tests import helpers from older files: `tests/test_sdf_plan_step.py:10`, `test_sdf_draft_step_resume.py:12`, `test_sdf_critique_and_revise_steps.py:17`, `test_sdf_score_and_gate_steps.py:18` import `plan_reply`, `make_plan`, `make_draft`, `make_rewrite`, `SCORE_OK`, `single_call_config`, `split_config` from `tests/test_sdf_layers.py` (lines 1-83 and 462-478); `test_sdf_screen_and_redeal_steps.py:22`, `test_sdf_dealt_value_identifiers.py:19`, `test_sdf_description_rewrite_step.py:18`, `test_sdf_consideration_scoring_and_assignment_steps.py:15` and `test_layer1_stage_model_fallback.py:13-17` import fixtures and builders from the three `test_layer1_*` files; `test_openai_api_and_description_rewrite_routing.py:123` imports `make_plans`. Those helpers must move to a shared module (or stay) before any file is emptied. Helper lines are not counted as removable.

## 1. `tests/test_sdf_layers.py` (693 lines, 43 test functions, 487 lines inside test bodies)
Covering tests: P = `tests/test_sdf_plan_step.py`, D = `test_sdf_draft_step_resume.py`, C = `test_sdf_critique_and_revise_steps.py`, S = `test_sdf_score_and_gate_steps.py`, E = the prompt-equivalence test and its named scenario.

Fully duplicated (411 lines, 37 tests, covering test named):
- `test_composes_once_then_plans` (85) : P:52 request + E main scenario stage files (prompts.jsonl).
- `test_composition_is_reused_and_resume_makes_no_calls` (98) : P:74 (zero calls, every stage file and the event log byte-identical).
- `test_incoherent_is_checkpointed_as_deliberate_rejection` (111) : P:31, P:86, E hand `HAND_INCOHERENT_PLAN`.
- `test_malformed_plan_is_retried_on_resume`, `test_truncated_plan_is_retried_on_resume` (123, 133) : P:86, E `failures_in_the_plan_across_resumes`.
- `test_plan_model_override_used` (141) : E `minimal_config_with_only_a_plan_model` (model in every recorded request).
- `test_one_poison_call...`, `test_all_plan_calls_failing_is_a_systemic_error` (147, 167) : P:61, E `failures_in_the_plan_across_resumes`, `every_plan_call_fails` (exit message).
- Draft: `test_drafts_from_descriptions` (179) : D:19 (arguments incl. max_tokens, cache_system) + E; `test_incoherent_plans_are_skipped` (194) : E calls-per-stage; `test_missing_tags_retried_on_resume` (201) : D:52 + E `failures_in_the_draft_across_resumes`; `test_resume_makes_no_calls` (211) : D:39; `test_one_poison_call...` (221) : D:52, D:82; `test_all_calls_failing...` (244) : D:82 + E `every_draft_call_fails`.
- Rewrite: `test_rewrites_and_keeps_review` (256) : C:95 + E `revision_in_one_call`; `test_rewrite_model_override_used` (272) : C:133; `test_missing_tags_retried_on_resume` (279) : C:62/C:90 + E `failures_in_the_critique_and_the_revise_across_resumes`; `test_resume_makes_no_calls` (290) : C:181.
- Truncation rejection: `test_layer2_rejects_a_truncated_draft...` (338) : E main scenario `HAND_TRUNCATED_DRAFT`, D:52; `test_layer3_rejects_a_truncated_rewrite...` (365) : C:62, E failures scenario.
- Score: `test_scores_gate_and_corpus_fields` (382) : S:74, S:91, E final corpus fields; `test_low_spec_conformance_does_not_gate` (402) : S:100; `test_parse_error_records_fallback_and_is_not_retried` (408) : S:54, S:169; `test_wrong_shaped_judge_reply...` (421) : S:53 (the `[8, 7, 9]` case, checked); `test_near_dup_cull` (433) : S:104 + E `gate_and_near_duplicate_boundaries`; `test_resume_makes_no_calls` (445) : S:169.
- Split rewrite (all seven): `test_two_calls_with_distinct_stage_tags` : C:133; `test_critique_is_banked...` : C:195; `test_a_truncated_critique_is_not_banked` : C:62; `test_an_untagged_critique_is_used...` : C:52 + E `HAND_NEAR_DUPLICATE_OF_PLAIN`; `test_resume_after_success_makes_no_calls` : C:181; `test_critique_model_falls_back...` : C:133 + E; `test_split_without_the_templates_refuses...` : C:168.
- Directives: `test_no_deal_sends_the_prompt_the_pipeline_always_sent` : E minimal-config scenario + `tests/test_layer3_templates.py:84`; `test_dealt_cards_reach_the_rewriter...` : C:149, E axes scenarios; `test_a_deal_present_but_the_arm_off_changes_nothing` : C:164.
- `test_the_drafter_the_rewrite_and_the_judge_see_the_planners_lines_without_omitted_variables` (650, 26 lines) : E main scenario sends the no-stake hand's draft, rewrite and score requests (inferred that this hand survives to layer 4; check before deleting).

Partially duplicated (38 lines): `test_derive_language` (456; only the French/English labels reach the final corpus in E, the Japanese and no-clause cases do not), `test_the_length_card_is_language_neutral` (575; E has no Chinese-culture hand with a length card, inferred), `test_a_qualitative_card_renders_without_a_ratio` (599; C:149 covers directive placement, not prose levels).

Unique (38 lines, 3 tests, 9 pytest cases): `test_layer2_keeps_a_draft_ending_on_a_signoff` and `test_layer3_keeps_a_rewrite_ending_on_a_signoff` (4 endings each, the false-positive truncation guard; nothing in E or the step tests uses sign-off or CJK endings), `test_a_snapshot_template_without_the_slot_still_renders` (old prompt snapshot compatibility).

Removable if all "fully" rows hold: about 411 test-body lines plus about 40 lines of class scaffolding and docstrings; the file would shrink to about 230 lines (helpers, partial and unique tests).

## 2. `tests/test_layer1_screen.py` (649 lines, 40 tests, 431 in bodies)
- Fully duplicated (117 lines, 11 tests): `test_screen_shows_the_planner_variable_lines_in_numbered_batches_of_ten` (174) : `tests/test_sdf_screen_and_redeal_steps.py:54,75` + E request hashes; `test_numbers_map_back...` (194) : step test :97 + E screen files; `test_unusable_answers_are_not_checkpointed` (228, 5 cases) : step :121 and :162; `test_one_failed_batch_stops_before_any_redeal...` (236) : step :162, E `failures_in_the_screen_across_resumes`; `test_batch_size_knob` (270) : step :54, E (batch size 1 and 2 scenarios); `test_batches_are_seeded_and_input_order_independent` (278) : step :54; `test_a_lone_removed_hand_cannot_be_redealt...` (349) : step :253, E `screen_removal_that_cannot_be_redealt`; `test_screen_terminates_after_the_maximum_rounds...` (354) : step :262; `test_removed_hands_that_cannot_be_redealt_are_dropped_and_recorded` (377) : step :311; `test_resume_after_a_completed_screen_makes_zero_calls` (506) : step :323, :350; `test_disabled_screen_makes_no_calls_and_writes_nothing` (524) : step :350 + E `screen_disabled`. Several of these assert `_screen_checkpoint.json`, so they change under cut 8 anyway.
- Partially duplicated (200 lines, 18 tests): the parse, schema and prompt-wording tests (82-169; E pins the request, step test :121 pins unusable answers, but the property-order and wording assertions guard intent after a deliberate regeneration), out-of-range number test (209; step :121 covers "outside 1..3"), `test_batches_are_written_before_any_call` (258), `test_redeal_never_touches_hands_already_accepted` (312), `test_zero_redeal_rounds...` (387), the six interrupted-run tests (396-503; E `interrupted_screen_rounds...` and step :323 cover the main path, not orphan or duplicate hands), `test_same_seed_gives_the_same_screen` (514).
- Unique (114 lines, 10 tests): column counts stay exact on redeal (293), deal-rule repair only among new hands (323), determinism per seed and round (337), snapshot without the screen template skips (531), and the audit report tests (569-649: removal rates, share shift, `load_screen_outputs`, audit on runs without screen files), which nothing in the new tests exercises. These move with cut 3.

## 3. `tests/test_layer1_description_rewrite.py` (320 lines, 23 tests, 201 in bodies)
- Fully duplicated (93 lines, 10 tests): decline, rewrite, conflict, model sees lines, incoherent not sent, malformed or unusable answer then resume retries, every call failing, completed-stage resume, disabled stage. Covered by `tests/test_sdf_description_rewrite_step.py:46,54,64,106,169,192,217,225` and E (`failures_in_the_description_rewrite_across_resumes`, `every_description_rewrite_call_fails`, `description_rewrite_on_a_claude_model`, `description_rewrite_disabled`). The tests that import `answer` and `make_plans` from here (`step` tests) need those helpers kept.
- Partial (42 lines): parse and schema and prompt-content tests (72-131); request pinned by E, semantic wording checks kept.
- Unique (66 lines, 7 tests): conflict whatever `rewrite_needed` says (77, 211), plans with no stored result on reload (223, 243), snapshot without the template (273), the per-check share and audit lines (282), audit on runs without rewrite files (316).

## 4. `tests/test_layer1_select_considerations.py` (602 lines, 37 tests, 354 in bodies)
- Fully duplicated (64 lines, 7 tests): retry then succeed, give up after the attempt limit, rerun and resume score only what is missing, one unusable document stops before assigning, every call failing is systemic, disabled selector. Covered by `tests/test_sdf_consideration_scoring_and_assignment_steps.py:95,113,131,160,197` and E (`failures_in_consideration_scoring_across_resumes`, `every_consideration_scoring_call_fails`, `consideration_selection_disabled`).
- Partial (120 lines): builders, parsing and the end-to-end selection tests (E records the scores and selection files on small hands only).
- Unique (170 lines): attach-selections, `load_considerations_outputs`, the audit coverage tests, the config caps test, blanked "where it arises" below three, snapshot and old-run compatibility tests (429-602), the none-leak test (530) and the spec-survival tests (547-575), which guard prompt text semantics that E would let a regeneration bless.
- Classification here is inferred from names for the partial and unique rows.

## 5. `tests/test_layer3_templates.py` (125 lines), `tests/test_layer1_stage_model_fallback.py` (54 lines, new)
All 8 template tests are unique or partial (byte-identical system block across three templates, nine checks, control arm leaves no trace). The fallback test is new and stays. Zero lines removable.

## 6. Audit tests (covered by the audit equivalence test, which pins requests only)
- `test_plausibility_audit_sdf.py` (479): `test_request_contains_only_instructions_and_content` (55, 16 lines) and `test_the_judging_prompt_text_is_unchanged` (80, 5) are duplicated by the recorded requests; schema, one-call, resume refusal, estimate-only, usage and aggregation tests (about 380 lines) are unique.
- `test_shoehorning_audit_sdf.py` (579): prompt files unchanged (50, 3), schema and instructions identical to expected fixtures (98, 8), instructions are header plus pass file (109, 5), request contents (318, 16): 32 lines duplicated; the rest unique.
- `test_compliance_sdf.py` (168): `test_prompt_carries_typology_and_document` (96, 10) duplicated; the rest unique. `test_variable_persistence_sdf.py` (379): prompt-rendering tests (93-127, about 40 lines) partial; the rest unique. `test_audit_sdf.py`, `test_model_judged_audit_support.py`: unique.
- About 63 lines fully duplicated across these files; recommend leaving them, since the exact-text pin lives in tests whose failure messages name the prompt file, and the recordings are regenerated deliberately.

## Totals for cut 7
Fully duplicated: about 685 test-body lines in the four layer files (411 + 117 + 93 + 64) plus about 63 in audit tests. With scaffolding, about 750 lines removable (about 7% of the 11,000 added lines; tests total 6,751 added). Production code: 0. Unique and partial lines (about 750) stay.

## Dependents, prerequisites, decisions
- Dependents: the helper imports listed above (checked); `test_openai_api_and_description_rewrite_routing.py:123`.
- Prerequisites: move shared helpers and fixtures into a shared test module (a pure move) before deleting; confirm each "fully" row by deleting the candidate in a scratch copy and checking that a deliberate mutation of the production code is still caught by the named covering test (not done).
- Decisions: none required (cut table says no). Optional: whether to keep exact-text prompt guards in audit tests.
- Interaction: tests that assert `_screen_checkpoint.json` or `_checkpoint.json` (screen 236, 258, 410-503; layers) must be rewritten again under cut 8, so do cut 8's test edits and cut 7 in one pass for those files, or leave them in cut 7.

## Risk not caught by the equivalence tests
The equivalence tests use one prompt set, three to twelve hands, English-only language distribution and canned replies. They would not notice the loss of: sign-off and CJK truncation guards, old-prompt-snapshot rendering, per-axis column-count invariants of the deal at 25 hands, crash-window cases (orphan hands, partial batches), audit report tables, and any behaviour that depends on printed lines (the step tests cover those). A failure of a deleted test would also have named the layer in plain terms; the equivalence failure shows a request-field diff.

## Ordered steps
1. Move helpers (`plan_reply`, `make_plan`, `make_draft`, `make_rewrite`, `SCORE_OK`, `split_config`, `single_call_config`, screen, rewrite and selector builders) to `tests/` support modules; update the 12 importing files; run the suite.
2. In a scratch copy, for each "fully" test, make a small deliberate break in the production behaviour it asserts and confirm the named covering test fails; keep any test whose break is not caught.
3. Delete the confirmed tests, starting with `test_sdf_layers.py` (largest, cleanest mapping).
4. Run the full suite, both equivalence tests, and the real-run request replay.
5. Leave partial and unique tests; revisit after cut 3 (readers) and cut 8 (checkpoints).

---

# Cut 8: retire per-layer checkpoint code (work item 9)
(Measured by a read-only fork agent; I did not re-verify each line number. The fork could not call its handback tool, so its text arrived as a plain message.)

Branch: william/sdf-dealt-value-identifiers, paths relative to the pipeline worktree. "Checked" = read in code; "inferred" = reasoned, not run.

## 1. Every checkpoint read and write (production code)

Checkpoint class: `shared/utils.py:641-667` (27 lines). Files: `<layer>/_checkpoint.json` (plan, draft, rewrite, score layers), `_screen_checkpoint.json` (keys are batch ids), `_description_rewrite_checkpoint.json`. Scoring writes none (`_NoCheckpoint`).

| Where | Lines | What |
|---|---|---|
| shared/utils.py:641-667 | 27 | the class |
| sdf_pipeline/step_runner.py:94,103 | 2 | `StepRunner` takes or builds `Checkpoint(stage_dir/_checkpoint.json)` |
| step_runner.py:114 | 1 | `select_pending` skips documents the checkpoint marks done |
| step_runner.py:178,181 | 2 | `mark_done` after a stage-file append (accepted, removed-with-record) |
| step_runner.py:286,290,296,346 | 4 | `BatchStepRunner`: pending batches, `mark_done(batch_id)` after the batch's files |
| layer1_plan.py:112,142 | 2 | builds and passes the checkpoint |
| layer1_description_rewrite.py:73,428 | 2 | filename constant, passes the checkpoint |
| layer1_screen.py:92,550,589,597 | 4 | constant, build, runner, pending batches |
| layer1_screen.py:701-702 (+ `batches_done`, `screened_ids` in `load_screen_outputs`, 693-720) | ~5 | reporting reader used by audit_sdf and report_sdf |
| layer1_select_considerations.py:302-310,577 | 10 | `_NoCheckpoint` stub (scores file is the checkpoint) |
| layer3_rewrite.py:311-322,332,336,355,364 | ~16 | `_CritiqueCheckpointView` (accepting a critique marks nothing) and checkpoint-or-rewrite-on-disk finished test |
| layer2_draft.py, layer4_score.py | 0 | use the runner default checkpoint |
| document_state_loader.py:190-191,290-297,419-421,527-531,555-557 | ~25 | reads all checkpoints (current and pre-renumber layout), only to emit notes ("marked done but has no record") |
| document_state.py:346 | 1 | comment/note field |
| update_steps.py:127-130,156; layer1_select_considerations.py:31,323 | ~6 | comments/flags (`writes_stage_file_as_replies_arrive`) |
| other layers' docstrings (layer1_plan 11-18, layer2 6, layer3 8, layer4 11, desc-rewrite 41-43, screen 53-56) | ~25 | prose about checkpoint semantics |

Total removable production code: about 100-110 lines of logic plus about 25 lines of docstring. `shared/` has no other checkpoint use for SDF (api.py:150,514, textstats.py are comments; DAD and pref use `Checkpoint` in dad_pipeline/baseline.py, step2_responses.py, step3_rewrite.py and pref_pipeline/run.py, so the class itself stays unless DAD moves, cut 10). `evals/score_sdf.py:51` uses its own `_sdf_score_checkpoint.json` (unrelated).

Important design fact (checked): resume is "stage file has a record OR checkpoint marks done" (step_runner.py:112-114). The checkpoint only matters when a record is missing: a checkpoint-only mark (crash-order cases), and the screen, where `_screen_checkpoint.json` is the only marker that a batch's records were fully written (`_drop_batch_records` then append, layer1_screen.py:~578-585).

## 2. Readers and writers outside the pipeline package

- Resume (`run.py:124`), seeding (`seeding.py`): neither reads or writes checkpoints (checked, no hits). Seeded runs rely on stage files alone, so seeding works with checkpoints gone.
- Loader: reads checkpoints only for notes (above). `fold_dealt_value_ids` (loader:326) is the only loader use of the event log.
- Viewer, website, evals: no checkpoint reads (grep of viewer, website, evals; website/sdf.py:104,806 are prose). `layer1_screen.load_screen_outputs` is the one reporting reader (feeds audit/report).
- `experiments/experiment.py:145-147`: writes `_checkpoint.json` into a layer-3 dir when importing donor rewrites (a producer; would become a no-op or need to emit events).
- Outer project `/Users/williamharrison/Desktop/emergent_alignment/scripts`: no script opens or writes a checkpoint file (grep for the three filenames and `Checkpoint(`: none). Two scripts only mention checkpoints in prose (`spec_contradiction_screen/run_end_to_end_pipeline_smoke_test.py:11,40,155`, `plan_fresh_specifications.py:7`) and rely on the pipeline's `--resume` behaviour. Hand-seeding scripts rely on stage files (inferred from their cut-1/2 descriptions; not opened).
- Data folders: 35 checkpoint files under the 7 committed runs in `outputs/sdf/runs`; no committed run has `document_events.jsonl` (checked: 0 of 7).

## 3. Tests that depend on checkpoints (lines of mention, not whole-test lines)

Files containing the word: tests mentioning it ~110 times in 25 files. SDF-relevant: sdf_prompt_equivalence_scenarios.py (15 mentions, helpers `_remove_checkpoint_marks`, `_delete_screen_checkpoint`, scenario "interrupted_run_records_and_checkpoint_marks_disagree"), sdf_prompt_equivalence_fixture_support.py (6), test_layer1_screen.py (11), test_sdf_screen_and_redeal_steps.py (11), test_sdf_layers.py (8), test_sdf_step_runner.py (5), test_layer1_description_rewrite.py (4), test_sdf_prompt_equivalence_and_final_corpus.py (4), test_sdf_plan_step.py (3), loader fixture test (3). Plus 14 recorded fixture JSON files hold checkpoint files as outputs (the recording lists each run's checkpoint files, inferred from the equivalence test header: "stage files, checkpoints"); retiring checkpoints means regenerating those recordings, which the guard blocks while Python differs from origin/main (see cut 11). Test code removable or rewritten: roughly 150-250 lines (inferred; no line-level audit), `shared` Checkpoint tests in tests/test_utils.py stay while DAD uses the class.

## 4. Writeup risk: does the accepted event's stored record hold each stage-file field?

Checked: `step_runner._append_event` stores `{"update": step.stage_file_record(document, outcome.update)}` for accepted, i.e. the full stage-file record, plus request hash and stage tag. Removed events store only `reason_code` and `detail`; failed events store `failure_kind` and `reason`. No reply text, only request hash.

| Stage file | Held by event? |
|---|---|
| plans.jsonl (accepted) | Yes, full record |
| plans.jsonl (incoherent, written and marked done) | Removed event `detail` holds `plan` and `description` (layer1_plan.py:90 builds the record from them); yes, rebuildable. Needs a rule that "removed" with a record means done |
| description_rewrites.jsonl accepted | Yes |
| description_rewrites (dealt-value conflict) and description_rewrite_rejections.jsonl | Record rebuildable from plan plus detail (`reasoning`, `conflict_reason`); the rejection file also needs the plan's `variables` and `description`, which come from the plan state, not the event |
| consideration_scores.jsonl, consideration_scoring_failures.jsonl | Accepted: full. Failed: kind and reason; the `problem` text is rebuilt by `_problem_text` (inferred equal) |
| consideration_selections/coverage | Corpus-step events carry `outcome.update` (checked in `run_corpus_step`); sufficiency for coverage json not checked |
| drafts.jsonl, critiques.jsonl, rewrites.jsonl, scores.jsonl | Yes, full record |
| near_dup_dropped.jsonl, gate | Removed events with detail; field coverage (`kept_doc_id`, similarity) inferred, not read |
| screen_judgments.jsonl | Facts (number, reasoning, reason) in per-hand events; `round` and `batch_id` come from the batch-plan event. Rebuildable (inferred) |
| screen_rejections.jsonl | Missing `variable_lines` (the exact lines the model saw) and `variables`; derivable by re-rendering (needs entity seed and templates; inferred, risk) |
| screen_hands.jsonl, screen_rounds.jsonl, screen_batches.jsonl | Round event holds ids, note, dealt value ids; created-hand `variables` are NOT in the log (the redeal finished event holds only id lists; checked at layer1_screen.py:~615-640). Real gap |
| screen_checkpoint semantics | "Batch fully written" is not an event: events are appended per hand as replies arrive, before the batch's files are written. A done rule based on events would mark a batch done after a crash that lost its files |

Also: events are appended at reply time, stage files in input order afterwards for non-as-arrive steps (step_runner `run` loop). A crash between them leaves an accepted event without a file record, the reverse of today's checkpoint-only case. Event-as-truth needs one chosen rule.

## 5. What must be true before checkpoints go

1. Decision 7 made yes (work item 9 is optional per plan item 9 line 151).
2. No run that must resume lacks an event log. Today 0 of 7 committed runs have one, and every pre-stack run is without. Either those finish on old code, or an adapter builds events from checkpoints and stage files (backwards-compat risk 1). Decision needed (plan line 313).
3. Event log completeness: add created-hand records, screen-rejection lines, and a "batch written" event (or define done as stage file records only).
4. Loader notes ("marked done but no record") dropped or reimplemented from events; pre-renumber layout resume already unsupported (cut 6).
5. Fixture recordings regenerated (the checkpoint files are in them), under the Python-differs-from-main guard.
6. `experiments/experiment.py:145` and hand-seeding scripts that write marks updated (they work by files today).

## 6. Lines, dependents, decisions, risk

- Removable: code about 105 plus docstrings about 25; tests about 150-250 (inferred). `Checkpoint` class stays (DAD, pref) unless cut 10 lands.
- Decisions: decision 7, old-run policy, whether the screen needs a batch-written event, whether stage files remain written (plan says yes, derived views).
- Risk not caught by the equivalence tests: (a) equivalence scenarios drive only new runs with an event log, so old-run resume (no log) is untested after the cut; (b) crash windows between event append and file write; the scenario "checkpoint marks disagree" encodes today's rule and must be redesigned, not just deleted; (c) real-run replay compares requests only, not resume state; (d) `load_screen_outputs` `batches_done` in audit/report changes silently; (e) concurrent hand edits of stage files (seeding scripts) no longer reflected.

## 7. Ordered steps (if approved)

1. Decide 7 and the old-run policy; pick the done rule (stage-file record exists, event ignored, versus event-derived).
2. Add missing events: created hands, rejection lines, batch written.
3. Make `StepRunner.select_pending` and `BatchStepRunner` use only stage files (or events); stop marking.
4. Delete `_NoCheckpoint`, `_CritiqueCheckpointView`, constants, constructor arguments, `load_screen_outputs` checkpoint read; loader checkpoint notes.
5. Update `experiments/experiment.py`.
6. Remove checkpoint tests and helpers; rewrite the disagree scenario; regenerate recordings (cut 11 guard).
7. Run full suite, equivalence and audit tests, real-run replay; keep `shared.utils.Checkpoint` for DAD.


---

# Notes on what was not measured
- No line coverage (the `coverage` package is not installed) and no mutation testing.
- Cut 8 numbers come from the fork's read of the code; the ranges for removable test lines are estimates.
- Other repositories on the machine that might read `outputs.sdf_final` were not searched beyond the pipeline worktree, `scripts/`, and `strictly_implicit_difficult_advice_pipeline_copy/`.
