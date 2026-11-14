# Measurement of cuts 1, 2 and 5 (read-only), top branch william/sdf-dealt-value-identifiers

Paths: P = /Users/williamharrison/Desktop/emergent_alignment/animal-welfare-data-pipeline/.claude/worktrees/sdf-dealt-value-identifiers ; O = /Users/williamharrison/Desktop/emergent_alignment (outer project). "Checked" = read in code or grep output; "Inferred" = reasoned, not run. Nothing was edited and no tests were run.

## Headline

- Cut 1 is safe to do, with one real gap (setup-only mode) and one behaviour difference (fewer layer 1 files copied). Removes 39 script lines; the test that reproduces the script (about 20 helper lines, plus two equivalence tests of 45 lines) should be kept in reworded form, not deleted.
- Cut 2 is not one cut. Only one other script in the outer project hand-builds a run folder (49 lines). The paired test's arm copy (22 lines) is on a branch outside the stack, and `experiments/experiment.py` (inside the pipeline) also hand-seeds. Seeding needs three additions, roughly 60 to 80 lines of code plus tests, before they can go.
- Cut 5 does not match its description. The wrappers are not thin (about 360 lines across seven functions) except `layer2_draft.run` (6 lines). Removing them moves code rather than deleting it. Net removable code is about 6 to 30 lines; about 211 test call sites would change. Not recommended as a standalone cut.

---

## Cut 1: delete set_up_layer_3_rerun_on_selected_drafts.py

### What it is
O/scripts/generation_from_one_objective_considerations/set_up_layer_3_rerun_on_selected_drafts.py, 39 lines, tracked in git (`git ls-files`). It makes a run folder with `utils.create_run_dir` from the checkout's current prompts and constitution (lines 25-27), copies the source's whole layer 1 folder with `copytree` (line 28), writes `layer2/drafts.jsonl` holding only the chosen doc ids (lines 29-37), refuses if any id has no draft (lines 30-32), prints the run folder name. No API calls. Afterwards the user runs `run.py --resume --run-id <name> --layer 3` (docstring lines 5-6).

### Everything that references it (grep over all of O excluding .git, __pycache__; worktrees included)
- O/plans/11-11-sdf-document-state-redesign/sdf-document-state-and-update-steps-plan_2026-11-11_21-24.md:299 (a list "Scripts that seed runs by hand").
- O/plans/11-11-sdf-document-state-redesign/sdf-document-state-redesign-backwards-compatibility-risks_2026-11-11_20-48.md:72.
- Copies of those two in O/.claude/worktrees/sdf-real-run-request-replay/plans/..., plus the cuts document itself in that worktree (table row 1).
- P/tests/test_sdf_seeding_from_a_chosen_step.py: docstring line 6, helper docstring line 135 (`write_run_the_way_the_layer_3_rerun_setup_script_does`, lines 134-153), and the tests at lines 157 and 182, which run the helper's output through `run.py --resume --layer 3` and compare every request and four output files with the seeded run.
- NOT referenced: O/CLAUDE.md, O/.claude/CLAUDE.md, O/references.md (grep for `set_up_` found nothing), no other script, no shell file, no pipeline CLAUDE.md or README (grep over P found only the test file).
- No data/ or outputs/ file names it. Whether any past experiment write-up under data/ used it by hand is unknown (its runs would not name it).

### Does it do anything seeding does not? Yes, three differences (all checked in code)
1. Setup-only. The script makes the folder and stops, with no API call; the user then inspects and launches. `run.py --seed-from` always continues straight into running the step (run.py:93-109 seeds, then falls into the layer blocks at 110+). So after the cut there is no way to prepare a seeded run without starting a paid run. Gap. Fix: a `--seed-only` flag (about 6 lines in run.py: after printing the "Seeded ..." line at run.py:105-106, return before `api.init`) and one test.
2. It copies the source's whole layer 1 folder (line 28); seeding copies only the files in `_STAGE_FILES_IN_PIPELINE_ORDER` (seeding.py:83-94), only the selected documents' records. At the critique step this omits `screen_hands_*` hands, rounds, batches and judgments files, checkpoints, the coverage report (seeding.py:83-94 lists only screened hands, prompts, plans, rewrite results and rejections, scores, selections). `load_screen_outputs` returns None when there is no hands file (layer1_screen.py:688-702 region, `if not hands: return None`), so `report_sdf.py` and `audit_sdf.py` would skip the screen section for a rerun, whereas the script's runs show it. Layer 3 and 4 do not read these files, so no request or output changes. Whether anyone wants the screen tables on a layer 3 rerun: unknown (inferred: unlikely, since the selected documents are a subset and the tables describe the whole source).
3. Source layout. The script uses `utils.sdf_stage_dir`, which handles pre-renumber layouts; seeding refuses any source not in the current layout (seeding.py:193-195). Seeding at critique also refuses sources missing a description rewrite result or assigned considerations for the selected documents (seeding.py:310-328) when those stages are enabled; the script never checked. So a source run made before the rewrite stage existed can be rerun with the script but not with seeding (inferred from the code; not exercised).

Not differences: prompts source (both use the checkout's current files when `--use-current-prompts` is passed, seeding.py:198-199), draft record contents and order (both copy source order, unchanged), missing-draft refusal (seeding.py:329-331), the run is a fresh folder with manifest.
CLI shape differs: the script takes `--pipeline-checkout` and `--experiment-dir` and writes under `<experiment>/pipeline_outputs/sdf/runs`; seeding runs inside a checkout and writes under `PIPELINE_OUTPUT_ROOT` (run.py:76-78). Equivalent if the user sets that variable (the script's own docstring does the same for the resume step).

### Lines removable
- Code: 39 (the script). No other code depends on it.
- Tests: 0 to about 65. Deleting the helper (P/tests/test_sdf_seeding_from_a_chosen_step.py:134-153, 20 lines) and the two equivalence tests (157-201, about 45 lines) would remove the only test showing seeding at critique equals the old hand setup. Recommendation: keep them, reword the docstring (line 6) and helper docstring (line 135) to say "the hand setup that was replaced, reproduced from its last committed code (commit 051e685's version)", and keep the helper. Net removable then about 2 lines.
- Docs: edit the two plan files' list lines (one line each); the cuts document row.

### Dependents and prerequisites
- Dependents: only the pipeline test and the plan docs above. The pipeline test lives in the stack (P), the script in the outer project, so the deletion commit goes in the outer repository and the test rewording in the pipeline repository: two pull requests.
- Prerequisites: the stack must be merged (or at least #208 on main) so the replacement exists for people who use it; `--seed-only` if setup-without-launch matters.

### Risk not caught by the prompt-equivalence or audit equivalence tests
- Both tests check requests and output files of runs; neither checks a seeded run's folder contents beyond what the seeding test asserts, or the evals' behaviour on a seeded run (point 2 above).
- `seeded_from` manifest field is new; any tool reading the manifest strictly (website, viewer: not checked) might not expect it.
- The prompt snapshot `constitution` is copied from the checkout in both, so no difference.

### Ordered steps
1. Decide whether setup-only matters; if yes add `--seed-only` to run.py with a test (stack branch, small).
2. In the pipeline: reword test docstrings, keep helper (no deletion).
3. In the outer project, on a branch: delete the script; update the two plan files and the cuts document row to say "retired, replaced by `run.py --seed-from <run> --from-step critique --use-current-prompts`".
4. Run the pipeline's `tests/test_sdf_seeding_from_a_chosen_step.py`, prompt-equivalence and audit equivalence tests unchanged.

---

## Cut 2: retire the other hand-seeding scripts

### Scripts found (search of O/scripts for `create_run_dir`, `copytree`, `sdf_stage_dir`, `variables.txt`, `shutil.copy`, and every importer of `sdf_pipeline`)

A. O/scripts/generation_from_one_objective_considerations/set_up_generation_run_from_saved_consideration_scores.py (49 lines, tracked). Hand-builds a run: `create_run_dir` with current prompts (lines 33-35), copies every file of the source's layer 1 except those starting `consideration_` or `_select_considerations` (36-41), copies a separate `--scores-file` in as `consideration_scores.jsonl` (42; the `hasattr` guard is dead since `SCORES_FILENAME` exists), loads the rewritten plans, `api.init`, then runs the CURRENT selector (`selector.run`, line 46-47), which finds every document already scored, makes no calls, and writes selections, coverage. Prints the API cost (should be 0). Afterwards the user runs `--resume --run-id <name> --layer 2`.
   Users: none found. Referenced only in the cuts document and the writeup (O/.claude/worktrees/sdf-real-run-request-replay/plans/...) and not in CLAUDE.md, references.md, other scripts. Its historical use (the one-objective considerations run) cannot be told from the repo (inferred).

B. Paired test on branch william/sdf-show-rejected-values-paired-test (worktree /Users/williamharrison/Desktop/emergent_alignment/animal-welfare-data-pipeline-show-rejected-variable-values; 4 commits ahead of origin/main, branch outside the stack, rebased on main not the stack). Files: experiments/show_rejected_values_paired_test/run_paired_show_rejected_values_test.py (473 lines), paired_test_support.py (357), offline_composition_without_model_calls.py (110); tests/test_show_rejected_values_paired_test.py (390).
   - `run_shared_deal_and_screen` (driver lines 83-119): creates the shared run with `create_run_dir` from the repository prompts, optionally overwrites the run snapshot's variables.txt with a centrality-enriched copy (`write_enriched_variables_into_run_snapshot`), then calls `layer1_screen.run` directly (line 110) and stops.
   - `copy_shared_state_into_arm_run` (paired_test_support.py:257-278, 22 lines): `copytree` of the whole shared run per arm (minus FILES_NOT_COPIED_INTO_ARM_RUNS), restores the repository's variables.txt into the arm's snapshot (lines 267-268), rewrites the manifest with the arm's config. Arms then run as subprocesses of `run.py --resume` (`start_arm_pipeline`, driver line 122).
   - `offline_composition_without_model_calls.py:106-107` calls `layer2_draft.run` and `layer3_rewrite.run` for a dry run with stubbed model calls.
   Users: the paired test itself (not run on this branch's stack; it is the project's pending experiment, `data/spec_contradiction_screen_runs/small_sample_checks/cost_estimate_for_show_rejected_values_paired_test_2026-11-06_00-55.md` exists).

C. P/experiments/experiment.py (382 lines, in the pipeline, in the stack): `run_arm` (lines 150-212) hand-builds each arm's run folder: `create_run_dir` (178-188), writes `layer2/drafts.jsonl` as the donor's sampled drafts each carrying a dealt `revision` record (199-205), optionally `import_rewrites` copies the donor's rewrites (`import_rewrites` defined near lines 134-150), then calls `layer3_rewrite.run` and `layer4_score.run` directly (204-206). This is seeding at critique with `--revision-override-file` (and at score for the donor-rewrite arm) in all but name; per-arm configs are merged in memory (`merge_arm_config`) rather than read from a file. Not named in the cuts table but it is the largest hand-seeder. Users: `experiments/compare_arms.py`, `tests/test_experiment.py` (237 lines; whether it exercises `run_arm`: not checked), `experiments/README.md`.

D. Not hand-seeding (checked, listed so the search is closed): spec_contradiction_screen/plan_fresh_specifications.py (copies the prompt directory and calls `layer1_plan.run` outside a run folder, lines 117 and 139), build_pinned_hand_pool.py, deal_rule_repair_before_and_after_on_pinned_run.py (read variables.txt for analysis), run_end_to_end_pipeline_smoke_test.py (launches `run.py` main normally), estimate_*.py (import constants and prompt builders), run_screen_conditions.py, run_description_rewrite_on_fresh_descriptions.py. The log_probability, persona_vector and emergent_misalignment `copy`/`variables.txt` hits are unrelated to SDF runs. Pipeline script P/scripts/rerun_considerations_selector_on_stored_plans.py (192 lines) reads a stored run's plans and calls `selector.run` into its own non-run folder (line 183); seeding does not replace it (it keeps the source untouched and compares coverage against the original).

### What seeding would need to gain, with size estimates (additions to P/sdf_pipeline/seeding.py and run.py)
1. Assignment seeding from a separate scores file (replaces A). New `scores_file` argument for `seed_run_from_chosen_step`; make `consideration_assignment` a seedable step (currently refused, seeding.py:75-76) that copies the file into layer1 as `SCORES_FILENAME`, validates every eligible plan has a score row (reuse `eligible_plans`, as at seeding.py:323), and starts at layer 1 so `run.py` runs screen, plan and rewrite as no-calls resumes and the selector finds all documents scored. Seeding itself never runs a step, so no code to run the selector inside it. About 30 to 45 lines of code, 50 to 70 lines of test (an equivalence test against a reproduction of script A, as for cut 1). Prerequisite check (inferred, not run): that `layer1_screen.run`, `layer1_plan.run` and the rewrite resume without calls from copied files; seeding already copies `screen_hands`, `prompts.jsonl`, `plans.jsonl`, rewrite results and rejections for a step that comes after them, but `consideration_assignment` would need start at layer 1 where seeding now starts at layer 1 only for plan, description_rewrite and consideration_scoring. Also unresolved decision 8 (subset at scoring re-solves the assignment).
2. Supplying a variables.txt (enriched). Needed by B for the shared deal, which is a normal run rather than a seeded one, so the addition is a run-creation option, e.g. `--variables-file` replacing `inputs/prompts/variables.txt` after `create_run_dir` and recording it in the manifest: about 10 to 15 lines in run.py plus 30 to 40 of test. For the arms, seeding at `plan` with `--use-current-prompts` already restores the checkout's variables.txt, so no extra addition is needed IF the paired test's prompts directory equals the checkout's prompts/sdf (inferred; the driver uses `support.REPOSITORY_PROMPTS_DIRECTORY`, not checked against the checkout). Seeding would also keep only the selected hands' files and drop cost logs (the arm copy already excludes them).
3. `--last-layer`. run.py has no way to stop early (checked: only `--layer` for the start, run.py:49-51 and the `if start_layer <= N` blocks, 110-146). Needed by B (stop after the screen, but the driver stops inside layer 1, so `--last-layer 1` is not enough: it would also plan; a `--last-step screen` is the finer need, inferred) and by A's replacement (stop after the assignment, layer 1). `--last-layer`: about 10 to 12 lines in run.py (flag, validation that start <= last, `and N <= last_layer` on four blocks) plus about 30 lines of test. A step-level stop would be about 25 to 40 lines because layer 1 is one block that calls four wrappers.
4. For C: a per-arm config file instead of an in-memory merge (no new seeding code, but `experiments/experiment.py` must write each arm's merged config to disk), and `--revision-override-file` already exists. Seeding at critique takes drafts for chosen ids from a SOURCE RUN's layer2 (the donor run), which matches the donor sample. The donor-rewrite arm maps to seeding at `score` (seeding.py:66, copies rewrites). Roughly 40 to 60 lines of rewrite in experiment.py, removing about 45 (the `run_arm` setup 178-205 and `import_rewrites`). Whether `experiments/experiment.py` is in scope for cut 2: decision for you.

### Lines removable
- Code: A 49; B `copy_shared_state_into_arm_run` 22 (+ arm subprocess plumbing, maybe 30, on a branch outside the stack, so only after that branch lands); C about 45 net of rewrite. Total about 70 now (A, B copy), about 115 with C.
- Tests: A has none (no test in the outer or pipeline repository references it; checked). B's tests would need rewriting: they sit in tests/test_show_rejected_values_paired_test.py (390 lines; how many lines test the copy: not measured). New tests are added by the additions (roughly 110 to 170 lines), so tests grow in the short term.
- Net: this cut adds about 60 to 80 code lines and 110 to 170 test lines to remove about 70 to 115 code lines. It is a simplification of ways to start a run, not a line saving.

### Dependents, prerequisites, who still uses
- A: nobody found. Could be deleted now with a note, or after addition 1; given no users, deleting it without the addition and relying on git history is defensible, but the addition is what makes the capability survive.
- B: the paired test, which is a pending experiment. It cannot move until its branch is rebased onto the stack or main (the branch's diff touches `layer1_description_rewrite.py`, `layer1_plan.py`, `layer2_draft.py`, `layer3_rewrite.py`, `run.py`, which conflicts with the converted layers). The cuts document says the same.
- C: `experiments/compare_arms.py`, `tests/test_experiment.py`.
- Prerequisites: additions 1 to 3; decision 8; decision 1 if any of this changes requests (it should not).

### Risk not caught by the equivalence tests
- Seeding copies only the selected documents' records and drops other layer 1 files (see cut 1, point 2): evals on seeded runs lose screen and batch tables.
- A seeded run's consideration assignment, if re-solved over a subset, differs from the source (decision 8); the equivalence test compares requests for a full set only.
- The paired test is not covered by either test (its branch has its own test of 390 lines); moving it changes the arms' folder contents and `copied_from_shared_deal_and_screen_run` manifest field read by its analysis scripts (`evals/analyse_paired_variable_persistence_arms.py`: not checked).
- `variables.txt` replacement changes the snapshot hash the manifest and event log record (inferred).

### Ordered steps
1. Add `--seed-only` (cut 1) and `--last-layer` to run.py with tests.
2. Add scores-file assignment seeding with an equivalence test against a reproduction of script A (copy the cut 1 helper pattern).
3. Delete script A in the outer project on a branch (after step 2 is on main).
4. Add the variables-file option; rebase the paired-test branch onto the stack or main; replace `copy_shared_state_into_arm_run` and the arm subprocess with `--seed-from <shared> --from-step plan --use-current-prompts --last-layer 4`; rerun its offline tests.
5. Optionally port `experiments/experiment.py` to seeding (writes per-arm config files).

---

## Cut 5: the per-layer run() wrappers

Line ranges and sizes in P/sdf_pipeline (checked with grep and sed):

| Layer function | Lines | Size | What it does beyond calling the step |
|---|---|---|---|
| `layer1_screen.run` (+ `_finish`) | layer1_screen.py:522-654 (+656-684) | 133 (+29) | Not a wrapper. Returns None when disabled or no template (527-534); reuses finished accepted file (537-541); settings, event log, deal (round 0), round loop with batching, redeal, drop records, event-log events, summary. About 160 lines of orchestration that holds the screen's resume logic. |
| `layer1_plan.run` | layer1_plan.py:104-147 | 44 | Composes prompts (or reuses `prompts.jsonl`), loads variables, deal rules; builds the step and runner; prints incoherent count. |
| `layer1_description_rewrite.run` | layer1_description_rewrite.py:410-437 | 28 | Disabled and missing-template early returns, counts pending, `begin_session`, raises SystemExit when descriptions are held out with nothing failing in the run (435-436), then `apply_results`. |
| `layer1_select_considerations.run` | layer1_select_considerations.py:551-596 | 46 | Disabled and missing-file early returns, scoring step with its own runner, rereads scores, runs the assignment as a corpus step, writes selections and coverage files, prints the report, attaches selections. |
| `layer2_draft.run` | layer2_draft.py:124-129 | 6 | Builds step, event log, runner; nothing else. The only truly thin one. |
| `layer3_rewrite.run` | layer3_rewrite.py:327-369 (+ `_CritiqueCheckpointView` 313-324) | 43 (+12) | Split versus single-call choice (328), checkpoint and existing-rewrite bookkeeping, critique then revise with critiques re-read from disk, and the systemic stop: `send_noting_thrown_calls` records documents whose call threw (333-342) and `SystemExit(SYSTEMIC_STOP_MESSAGE)` when every pending document threw (366-367). |
| `layer4_score.run` | layer4_score.py:155-215 | 61 | Score step, gate corpus step, near-duplicate cull corpus step, writes `near_dup_dropped.jsonl`, builds and writes `final/sdf_corpus.jsonl` with derived fields (language, register), prints counts and a spec_conformance note. |

Total about 361 lines for seven functions (about 400 with `_finish` and the checkout view). The cuts table calls them thin; only layer 2 is. Everything else is stage orchestration (resume rules, stop rules, file writes) that would have to live somewhere else, so this cut relocates lines. Truly removable: layer 2's six lines plus `def`/docstring/import glue for the others, about 6 to 30 lines of code (inferred estimate).

### Callers (grep over P/scripts, experiments, evals, shared, website, viewer, sdf_pipeline, tests, and O/scripts)

Calls of `<layer>.run(` in non-test code:
- P/sdf_pipeline/run.py:113 (screen), 115 (plan), 117 (rewrite), 119 (selector), 129 (draft), 136 (layer 3), 143 (layer 4). All seven. The orchestrator itself; moving to calling steps directly means rewriting run.py (this is cut 4 territory).
- P/sdf_pipeline/seeding.py: never calls `run()`; imports layer 1 modules only for constants and predicates (lines 37, 84-90, 155-159, 172, 310-328). Seeding cannot absorb wrapper callers.
- P/experiments/experiment.py:204 (`layer3_rewrite.run`), 206 (`layer4_score.run`): could move to seeding (cut 2 item C), about 40 to 60 lines of work; calling steps directly is harder because of the systemic stop and checkpoint view.
- P/scripts/rerun_considerations_selector_on_stored_plans.py:183 (`selector.run`): calls one wrapper into a non-run folder; to move it, it would call `ConsiderationScoringStep` plus `ConsiderationAssignmentStep` itself (about 30 lines, copying the body of the wrapper) or keep calling the wrapper.
- Branch william/sdf-show-rejected-values-paired-test (outside the stack): experiments/show_rejected_values_paired_test/run_paired_show_rejected_values_test.py:110 (`layer1_screen.run`), offline_composition_without_model_calls.py:106-107 (`layer2_draft.run`, `layer3_rewrite.run`). Moves to seeding per cut 2 steps 4; the offline dry run with stub calls could call layer 2 and 3 steps directly (about 20 lines).
- Outer project O/scripts: `spec_contradiction_screen/plan_fresh_specifications.py:139` (`layer1_plan.run`, outside a run folder, screen off): calls the wrapper directly; moves to calling `PlanStep` through `StepRunner` itself (about 20 lines, not hard but it loses the prompt composing in lines 104-125 of the wrapper; would need `compose_records` too) or stay. `generation_from_one_objective_considerations/set_up_generation_run_from_saved_consideration_scores.py:29,45-47` (`selector.run`): disappears with cut 2 item A. Other O/scripts hits do not call `run()`: `run_screen_conditions.py` uses `layer1_screen.load_screen_prompts` and `screen_hands`; `estimate_validation_costs.py` and `estimate_end_to_end_pipeline_smoke_run_cost.py` use prompt builders and `_MAX_TOKENS` constants (`layer2_draft._MAX_TOKENS`, `layer3_rewrite._MAX_TOKENS`, `layer1_select_considerations._MAX_TOKENS`); `run_description_rewrite_on_fresh_descriptions.py` uses `rewrite_description`, `load_rewrite_prompts`; `run_end_to_end_pipeline_smoke_test.py:145` calls `sdf_pipeline.run.main()`. So the "12 scripts in this project import layer modules" in the cuts table is about right (grep: 9 outer scripts import layer modules; 3 of them call `run()`), but only 2 call a wrapper's `run`.
- Reporting loaders, not wrappers (not part of this cut): P/evals/audit_sdf.py:192-198, 243-249, 270-274 and report_sdf.py:327-331, 370-372, 398-400 call `load_screen_outputs`, `removal_rate_rows`, `round_rows`, `share_shift_rows`, `load_description_rewrite_outputs`, `rewrite_share_rows`, `load_considerations_outputs`, `corpus_coverage_report`; P/scripts/print_full_sdf_request_a_stage_would_send_for_one_document.py:51-57 imports all seven layer modules for step classes and constants (e.g. `ScreenStep`, `DescriptionRewriteStep`), not `run()`.
- Tests: about 211 call sites of `X.run(` across 15 test files (grep `\b[a-z_0-9]+\.run\(` with subprocess, runner and similar excluded; approximate): test_sdf_layers.py 60 (plan 13, draft 14, layer 3 24, layer 4 9), test_layer1_screen.py 33, test_layer1_select_considerations.py 24, test_sdf_consideration_scoring_and_assignment_steps.py 18, test_layer1_description_rewrite.py 16, test_sdf_score_and_gate_steps.py 11, test_sdf_critique_and_revise_steps.py 10, test_sdf_screen_and_redeal_steps.py 9, test_sdf_draft_step_resume.py 9, test_sdf_description_rewrite_step.py 7, test_sdf_plan_step.py 5, test_layer1_stage_model_fallback.py 3, test_sdf_dealt_value_identifiers.py 2, test_openai_api_and_description_rewrite_routing.py 2, test_deal_constraints.py 2. (test_dad_steps.py 82 are DAD's.) Tests of wrappers are also the tests of resume and stop behaviour, so they cannot be dropped, only repointed.

### Move or keep, per caller
- run.py: keep wrappers or fold into a step list (cut 4); no gain without it.
- experiments/experiment.py: move to seeding (cut 2 item C), 40-60 lines.
- selector rerun script and plan_fresh_specifications: call steps directly or keep the wrapper; about 20-30 lines each; minor.
- paired test and set_up_generation_run...: go with cut 2.
- Tests: no sensible replacement for `run()` unless a replacement entry point exists; repointing 211 call sites is mechanical but large, and every one is a resume or stop behaviour check.

### Lines removable
- Code: about 6 to 30 net (see above). The 361 wrapper lines survive wherever they move.
- Tests: 0 net; about 211 call sites edited.

### Dependents, prerequisites
- Cut 4 (step list) is the real prerequisite: wrappers only disappear when a `Pipeline` object owns ordering, resume, early returns and stop rules (the table in the cuts document already says so). Cuts 1 and 2 first reduce the outside callers to run.py, the selector rerun script, and plan_fresh_specifications.
- No decision outstanding.

### Risk not caught by the equivalence tests
- Both tests compare requests and output files for complete runs; they do not exercise: the systemic stop (layer3_rewrite.py:333-342, 366-367, needs a run where every call throws), the held-out stop of the description rewrite (layer1_description_rewrite.py:435-436), disabled and missing-template early returns, mid-run resume of the screen's round loop (layer1_screen.py:588-596), or console output text (progress lines are printed from wrappers). The step tests and layer tests do cover several of these (tests above), which is why they must stay.
- Layer 3 per-document call order and printed line order already changed in #202 (decision 4); the equivalence test is order-insensitive, so any further reordering when the wrappers move would also pass silently.
- The paired-test branch still imports the old entry points and will break at its next rebase if wrappers are removed.

### Ordered steps
1. Do cuts 1 and 2 (removes `set_up_generation_run...` selector call and moves `experiment.py`).
2. Do cut 4 (a step list in run.py), moving each wrapper body into the list entry or the step class (layer 2 first, 6 lines).
3. Repoint tests from `X.run(...)` to the new entry point in one file at a time, running that file after each.
4. Repoint or leave `rerun_considerations_selector_on_stored_plans.py` and `plan_fresh_specifications.py`.
5. Delete the empty `run` functions; run both equivalence tests unchanged, plus the real-run request replay.
