# Cut measurement: website and viewer onto the loader, DAD onto StepRunner, further cleanup

Read-only. Nothing edited, committed, run against an API, or built. Top branch `william/sdf-dealt-value-identifiers` (3683685), compared with `origin/main`. All paths relative to the pipeline worktree `animal-welfare-data-pipeline/.claude/worktrees/sdf-dealt-value-identifiers` unless absolute. "Checked" = read in the code or ran a grep/import; "inferred" = reasoned, not run. I did not run any test or rebuild `website/index.html`.

Method for dead-code claims: grep -w across the pipeline (`sdf_pipeline shared evals scripts viewer website dad_pipeline`), `tests/`, and `/Users/williamharrison/Desktop/emergent_alignment/scripts`. Helper scripts used are in the scratchpad (`find_dead.py`, `refs.sh`).

---

## Cut 9: website and viewer read stage files through the loader

### Branch note (checked)
`william/sdf-website-and-viewer-read-through-document-state-loader` is not empty and not at the top of the stack. It points at 9ba1495 ("SDF report, description rewrite audit and shoehorning postprocess read plans and drafts through the document state loader"), which is 15 commits ahead of `origin/main` and is NOT an ancestor of the top branch. It looks like the pre-rebase tip of #199 (the stack has bcf80fd with a similar message). It has its own worktree at `.claude/worktrees/sdf-website-and-viewer-read-through-document-state-loader`. Reset or recreate it from the top branch before starting; do not build on it.

### What goes: website (`website/sdf.py`, 1,178 lines)
All of it is stdlib reading of the pinned run's pre-renumber layout (`layer12`, `layer3`, `layer4`, `layer5`), using `common.read_jsonl` (website/common.py:49), which silently skips any bad line.

| Item | Lines | Lines removed |
|---|---|---|
| `_STAGE_FILES` table and its comment | 86-97 | 12 |
| `load_inputs` stage reads for corpus, scores, lineage, attrition, matrix | 140-150 (partial) | ~6 |
| `read_lineage` | 175-207 | 33 |
| `_PLANNING` regex and `_planning_notes` | 210-220 | 11 |
| `read_attrition` | 223-240 | 18 |
| `_AXIS`, `_WEIGHTED`, `read_matrix` | 243-278 | 36 |
| Subtotal, clear parsing | | about 116 |
| `gate()` plus `_PARSE_ERROR` (partly derivable from outcomes and `DocumentScores.was_parse_error`) | 107, 295-315 | 24 more, only if rewritten |

Adapters to add: lineage dict from states (about 25 lines), attrition counts (about 15), a matrix reader that still needs weights (about 36, unchanged, see below). Net code removed: about 60 to 75 (inferred). Test lines touched, not removed: `tests/test_website_sdf.py:548-611` (about 64 lines, matrix, attrition, lineage and `load_inputs` tests) would be rewritten on states.

Hardcoded layout strings in messages also go with it: sdf.py:510 and :522 (`layer12/...`).

### What the loader lacks for the website (checked unless marked)
- Matrix weights. `read_matrix` returns `{axis: {value: weight}}` from the frozen `inputs/prompts/variables.txt`. The loader only has `load_frozen_value_ids` ({axis: {id: text}}, `dealt_value_identifiers.py:42`), no weights, and no list of values that were never dealt. Keep `read_matrix`, or add a `frozen_variables_matrix(run)` to the loader.
- Inferred defect: `_WEIGHTED` (`^([\d.]+)\s*::\s*(.+)$`) does not strip the `[id]` prefix that current `variables.txt` files carry. The pinned run predates ids, so the page is unaffected today. A loader-side parser (`shared/matrix.parse_variables_with_ids`) would fix it.
- Attrition counts equal raw line counts of each stage file: `layer12/prompts.jsonl` (hands dealt, one per hand), `plans.jsonl`, `drafts.jsonl`, `rewrites.jsonl`, `scores.jsonl`, `final/sdf_corpus.jsonl` (sdf.py:89-97). States give per-document furthest step instead. The counts agree only if the state's "has a record in file X" matches line count (duplicate lines, a run that resumed). Needs a per-file record count on `LoadedDocumentStates`, or the equivalence check will catch a difference.
- The raw `scores` records (`rec["scores"]["notes"]`) are available as `DocumentScores.recorded_values` and `was_parse_error` (document_state.py:270-275). The corpus records (`doc_id`, `content`, `scores`, legacy fields) come from `document_state_final_corpus.derive_final_corpus_records`, which the loader only offers as a derived, test-oriented path. Reading `final/sdf_corpus.jsonl` as a file is still simpler and must stay for a bare corpus.
- Review text: `TextVersion.critique` (document_state.py:236-245), planning notes: `DocumentState.planning_notes`, plan description: `planned_description`, draft: `draft_text`, cards: `dealt_cards`. All present.
- Behaviour differences: loader raises `StageFileCorruptError` on a bad non-final line where the website skips it; loader raises `UnknownRunLayoutError` for a bare corpus path (use `load_document_states_or_empty_list_when_not_a_run_folder`, loader:123). The page must still build with no `--sdf-run`.

### The standard-library-only promise
Promise stated at: `website/sdf.py:39` ("stdlib only, and no imports from viewer/ or shared/"), `website/page.py:25`, `website/common.py:3`, `website/dad.py:39`, `website/render.py:19-20` ("must run anywhere"), `website/build_website.py:14-16` ("builds in an environment where the pipeline's own dependencies are not installed"). No test enforces it (grep of `tests/test_website*.py` for stdlib, sys.modules, import checks: none).

Checked by importing `sdf_pipeline.document_state_loader` and listing new modules: no third-party packages load (site-packages: none; `faker` is imported lazily inside a function, `shared/entity_pools.py:153`). So the promise "stdlib only" holds; the part that breaks is "no imports from shared/" and "runs without the pipeline". Modules loaded: `sdf_pipeline`, `sdf_pipeline.compose_prompts`, `.considerations_catalog`, `.dealt_value_identifiers`, `.document_event_log`, `.document_state`, `.document_state_loader`, `shared` (its `__init__` calls `sys.exit` on Python below 3.12, shared/__init__.py:12), `shared.deal_constraints`, `shared.entity_pools`, `shared.matrix`.

Which loader imports break it, and why each is there:
1. `document_state.py:80` `from sdf_pipeline.compose_prompts import REVISION_AXES`. Pulls all of `compose_prompts` (678 lines), which inserts the repo root into `sys.path` (compose_prompts.py:75-77) and imports `shared.deal_constraints`, `shared.entity_pools`, `shared.matrix`. Needs one 5-string tuple.
2. `document_state.py:81` `from sdf_pipeline.considerations_catalog import SECTION_EPISTEMIC, SECTION_MORAL`. Imports `compose_prompts` and `shared.deal_constraints` for two strings.
3. `document_state_loader.py:50` `dealt_value_identifiers` imports `shared.matrix` (parse of `variables.txt`) for `load_frozen_value_ids` (used at loader:329 and :486).
4. `document_state_loader.py:51` `document_event_log`: stdlib only, no break, but it sits in `sdf_pipeline/`.

### How much of the loader's parsing is pure
Checked: all 588 lines of `document_state_loader.py` and all 442 lines of `document_state.py` are pure parsing and dataclasses over stdlib; the only non-pure uses are the three imports above. Concretely, to split into a no-pipeline-imports module:
- Move `REVISION_AXES`, `SECTION_MORAL`, `SECTION_EPISTEMIC` into a tiny constants module that `compose_prompts` and `considerations_catalog` re-import (about 10 lines; `compose_prompts.REVISION_AXES` stays as an alias, so no caller changes).
- Make dealt value ids optional: either copy `parse_variables_with_ids` plus `value_ids_by_variable` (about 70 lines of stdlib in `shared/matrix.py:47-135`) into the pure module, or have the loader accept a `frozen_value_id_reader` argument (about 15 lines), with the pipeline passing the real one.
- Result: about 1,000 of about 1,030 lines become a package with no `shared/` or `sdf_pipeline` imports (inferred, about 97 percent pure). Package boundary choice: a new top-level folder (for example `document_state/`) or keep under `sdf_pipeline/` and relax the promise to "no third-party imports, no imports from viewer/". The website would still `import sdf_pipeline`, which runs `sdf_pipeline/__init__.py` (empty, checked) but `shared/__init__.py` is only pulled by the import chain above, so removing it from the chain removes the 3.12 exit.

### What goes: viewer (`viewer/loader.py`, 648 lines)
| Item | Lines | Removable |
|---|---|---|
| SDF half of `STAGE_FILES` | 41-57 | 14 (DAD half and `load_stage` stay, 178-190) |
| `_pass_rate` SDF branch | 261-264 | 4 |
| `sdf_lineage` | 343-364 | 22 |
| `_sdf_match_key` | 477-483 | 7 |
| SDF branch of `match_outputs` | 491-514 | 24 |
| Subtotal | | about 70 (inferred net after adapters: about 30 to 40) |

Callers (checked): `viewer/ui_pages/lineage.py:81` (`load_stage(... "subtypes")`), `:200-219` (`sdf_lineage`, raw record display), `viewer/ui_pages/compare.py:72` (`match_outputs`), `viewer/rendering.py:521-557` (reads `lineage["subtype"]`). `viewer/loader.py` imports `shared.utils` (line 13) and `dad_pipeline.profiles` (line 28), so it has no standard-library promise.

What the loader lacks for the viewer:
- Raw stage records. The viewer shows `json_block` of the legacy `document_types` and `subtypes` records and `rw["original"]`, `rw["rewritten"]`, `rw["review_notes"]` (lineage.py:210-232). The loader keeps only the subtype description and puts `type_id`, `subtype_id`, `role` in `legacy_fields`; it does not return the document type or subtype records.
- Inferred defect (not run): lineage.py:206-210 reads `rw["original"]`, `rw["rewritten"]`; current `layer3/rewrites.jsonl` records carry `content` and `review` (`layer3_rewrite.py:127`, `:201`), so the SDF lineage page would raise a KeyError on a current run. Cut 9 for the viewer is therefore partly a fix, not only a deletion.
- `list_runs` counts per-stage records through `load_stage` for every key in `STAGE_FILES` (loader.py:289-293); states give document counts, not record counts.
- No viewer output test exists: `tests/test_viewer_loader.py` has no SDF tests (grep: 0), `tests/test_viewer_rendering.py` mentions sdf 4 times. Needed first: a test that records `sdf_lineage` and `match_outputs` for the committed runs before the change (a viewer output test, decision 3).

### Prerequisites, decisions, risk, steps
- Decisions: 2 (relax promise vs split) and 3 (viewer: extend loader and add output test, or leave).
- Risk: website medium-low (equivalence check exists as a manual rebuild of `website/index.html` from the pinned run; commit 9ba1495's message claims byte-identical; no automated test). Viewer medium (no output test today).
- Request or output change: none for pipeline runs. Website output must stay byte-identical (`website/index.html` is committed and published by `.github/workflows/pages.yml` on a path trigger, so a diff there would republish).
- Steps:
  1. Reset the branch from the top branch.
  2. Move the three constants into a pure module; make value ids optional or copy the two matrix functions. Test: import the loader under `python -I` with `shared` and `sdf_pipeline.compose_prompts` blocked from `sys.modules`.
  3. Add weights (`frozen_variables_matrix`), per-file record counts and the legacy type and subtype records to the loader, with tests on every committed run (the existing `test_document_state_loader_on_committed_runs_and_damaged_run_folders.py` pattern).
  4. Website: replace sdf.py:86-97, 175-278 with state-based adapters; rebuild from the pinned run and diff `website/index.html` byte for byte; rewrite the 64 test lines.
  5. Viewer: add an output test first, then replace; fix the `original`/`rewritten` keys.
  6. Update the stdlib promise sentences in the six files above.

---

## Cut 10: DAD onto StepRunner

### What DAD's loop is (checked)
DAD (`dad_pipeline/`, unchanged in the stack, 0 lines in the diff) has no shared runner. Each step has its own `run` with the same shape SDF layers had: load existing jsonl and a `utils.Checkpoint`, compute a pending list, fan out with `utils.parallel_map(worker, pending, workers)` (shared/utils.py:17-30, the very function `StepRunner.run` also uses at step_runner.py:207 and :336), then on the main thread in input order append records, call `registry.save()` and `checkpoint.mark_done`. So the fan-out itself is already shared; what is "own" is the per-step main-thread consumer.

| Step and function | Lines | parallel_map sites | Calls per item |
|---|---|---|---|
| `baseline.run` (baseline.py:36-105) | 70 | 1 (:79) | 1, plus a second call at a doubled `max_tokens` on `max_tokens` stop |
| `step3_rewrite.run` (:26-165) | 140 | 1 (:95) | 1, plus second call at 8000 tokens |
| `step2_responses.run` (:198-481) | 284 | 1 (:431) | scope (up to `MAX_SCOPE_ATTEMPTS`, plus one on the fallback model after a refusal), then selection call (2a.5), then one respond call per missing sample; all inside one worker |
| `step1_dilemmas.run` (:476-1015) | 540 | 4 (:713 plan, :825 draft, :867 gate, :921 refine) | plan up to 2 attempts (3 after a refusal, last on a fallback model); draft one per pass; gate with retries; refine with retries |
| `dad_pipeline/run.py` orchestrator | 201 | none | n/a |

Total in the four `run` functions: about 1,034 lines. The generic part (pending selection, loop, mark done, append) is about 25 to 40 lines per step, so about 100 to 150 lines (inferred); a step class per call kind would cost roughly that much back. Net line saving: about zero (inferred). The gain would be the event log, request hashes and resume semantics shared with SDF, not fewer lines.

### Multi-call-per-item steps and redraft loops (checked)
- Step 2: scope then selection then N responses per prompt, sequential in one worker, writing two files (`scopes.jsonl`, `responses.jsonl`) plus `scope_rejects.jsonl`, `scope_failures.jsonl`, `select_failures.jsonl`, and assigning registry ids (`IdRegistry`, main thread only). One document produces several output records (one per sample index), with checkpoint keys `{pid}_s{i}`.
- Step 1b to 1d: a `while True` pass loop (step1_dilemmas.py:781-1000, cap of 12 passes). Pass n drafts every unfilled scenario; gate rejections feed `gate_feedback` into the next pass's draft prompt (`{redraft_feedback}`); the gate redrafts up to `MAX_GATE_REDRAFTS` then ships with `gate_failures` stamped; refine can mark a draft UNFIXABLE (rejected to `refine_rejects.jsonl`); a duplicate `prompt_gid` skips the scenario until resume. This is a graph with state carried across passes, not a fixed step list.
- Retries with changing requests: plan switches to `scenario_refusal_fallback_model` on the last attempt after a refusal; scope likewise; step 3 and baseline retry at a larger `max_tokens`. Per-attempt raws go to `*_failures.jsonl` (many records per item).

### What StepRunner would need (checked from `sdf_pipeline/step_runner.py` and `update_steps.py`)
1. Location: `StepRunner` lives in `sdf_pipeline/` and imports `sdf_pipeline.update_steps` and `document_event_log`. `shared/` has only `parallel_map`, `Checkpoint`, `api.call_claude`. It would move to `shared/` (or DAD imports `sdf_pipeline`).
2. Id field: records default to `doc_id` (step_runner.py:104-106). DAD keys are `prompt_gid`, `scenario_id`, `response_id`, `response_gid`, `{pid}_s{n}`.
3. Next request may depend on the previous reply or attempt number (model switch, bigger `max_tokens`, redraft feedback). Today `StepPolicy.maximum_attempts_in_one_run` resends the same request (step_runner.py:131-156). Needs `build_request(document, attempt_history)`.
4. Several records or files per item, several raw failure records per item (the runner writes one `failure_record` per final failure, step_runner.py:165-173), rejection files separate from the stage file (the runner's removal record goes to the stage file).
5. A loop construct for the draft, gate, refine passes (the SDF plan has none either: the screen loop is hand-written at layer1_screen.py:595-650).
6. Registry id assignment on the main thread in `stage_file_record`; works with the runner's main-thread writes.
7. If step 2 were split into scope, select and respond steps, the call order changes: all scopes before any response (the same kind of change as decision 4 for layer 3), so requests are the same but the order of calls differs.

Decision needed: whether to extend the redesign to DAD at all (table row 10). Risk: high for steps 1 and 2, medium for baseline and step 3 (single-call steps are the natural first pilot).

### Equivalence test for DAD (checked)
None. DAD has `tests/test_dad_steps.py` (1,819 lines, behaviour tests with a stubbed `call_claude`: TestStep1Run, TestStep2Run, TestStep3Run, TestBaselineRun), `tests/test_e2e_smoke.py` (one offline DAD run, line 239), `test_audit_dad.py`, `test_website_dad.py`, `test_compose_scenarios.py`, `test_id_registry.py`. There is no recording of every request, stage file, checkpoint and final corpus against `origin/main`. The SDF recordings (`tests/sdf_prompt_equivalence_fixture_support.py` 1,251 lines, scenarios 490, 57 JSON fixtures) are SDF-specific. A DAD equivalent is a new build of the same size order (about 1,500 lines of harness plus recordings, inferred) and must exist before any DAD change.

Steps if pursued: (1) build DAD request and output recording from `origin/main`; (2) pilot on `baseline.py` (single call, one file); (3) step 3; (4) step 2 with a multi-record step interface; (5) step 1 last.

---

## Part B: further cleanup not in the table of 11

Ranked by lines removed per risk. "Changes request or output?" is checked unless marked inferred.

| # | Item | Where | Lines removable (code / tests) | Risk | Changes request or output file? |
|---|---|---|---|---|---|
| B1 | Construct the event log in `StepRunner` (default from `stage_directory.parent / DOCUMENT_EVENT_LOG_FILE_NAME`) instead of seven identical constructions plus seven import lines | layer1_plan.py:138-139, layer1_description_rewrite.py:423-424, layer1_select_considerations.py:574-575, layer2_draft.py:126-127, layer3_rewrite.py:330-331, layer4_score.py:167-168, layer1_screen.py:555-556 | about 22 / 0 | very low | no |
| B2 | One helper for the outcome-to-event fields, which exists three times | step_runner.py:183-193 (`StepRunner._append_event`), :234-259 (`run_corpus_step`), :318-327 (`BatchStepRunner._append_event`) | about 18 / 0 | low (event contents must stay equal; `test_sdf_document_event_fold.py` pins them) | event log content unchanged; request and files no |
| B3 | Unused `score_document` (standalone one-document retry loop that the step re-implements; docstring claims "the scripts use" it but no script in the pipeline or `/scripts` calls it) | layer1_select_considerations.py:252-287, docstring line 53 | 36 / about 25 (`tests/test_layer1_select_considerations.py:241-262`, two tests of the retry loop that the step tests also cover) | low | no (not on any run path) |
| B4 | Adapter classes `_NoCheckpoint` and `_CritiqueCheckpointView` that exist only to bend `StepRunner`'s checkpoint; replace with a runner option "marks nothing" and "also finished ids" | layer1_select_considerations.py:302-310, layer3_rewrite.py:311-326 | about 23 minus about 8 added = 15 / 0 | low-medium (checkpoint behaviour is in the equivalence recordings) | no, if behaviour kept |
| B5 | Stage file id field: runner defaults to `doc_id` (step_runner.py:104-106) so four layers load their own `{prompt_id: record}` dict and pass it in; let the step declare `stage_file_id_field` | layer1_plan.py:135-141, layer1_description_rewrite.py:205 and :424-428, layer1_select_considerations.py:572-578, :586, layer3_rewrite.py:333 | about 10 / 0 | low | no |
| B6 | Failure text and progress helpers repeated: `removeprefix("error: ")` at four sites, "will retry on resume" strings at nine, and the throttled progress counters (`begin_run`, `begin_session`, `PROGRESS_INTERVAL_SECONDS = 180` defined twice) | layer1_select_considerations.py:95, :346-351, :403-431; layer1_description_rewrite.py:75, :324-333, :370-390; layer1_screen.py:492; layer4_score.py:105; progress_line in layer1_plan.py:93-101, layer2_draft.py:114-121, layer3_rewrite.py:140-141, :205 | about 30 / 0 | low (printed text must stay equal; step tests assert printed lines) | stdout only if text drifts |
| B7 | Three ways to stop on a systemic failure: runner count (step_runner.py:226-227, 349-350), the `after_documents_are_processed` hook that raises itself (select:424-431, description rewrite:390-400), and layer 3's `send_noting_thrown_calls` wrapper counting across two runners (layer3_rewrite.py:333-367). Reordering the hook before the runner's check would let scoring and the rewrite drop their own raise | listed | about 25 / 0 | medium (stop messages and order of the summary line) | no request or file change; exit message path |
| B8 | Dead result fields: `BatchRunResult.pending_batch_ids/finished_batch_ids/failed_batch_ids` (no reader anywhere, even tests: grep 0) and `StepRunResult.accepted_document_ids/removed_document_ids` (read by one test file only) | step_runner.py:26-35, 262-267, and their appends | about 14 / about 6 | very low | no |
| B9 | `shared/api.py` and `step_runner` both hand-build OpenAI settings: `resolve_settings_of_model_request` (step_runner.py:78-90) builds `ResolvedRequestSettings` for the description rewrite's OpenAI route by hand, because `send_model_request` (step_runner.py:46-61) calls `openai_api.call_openai` directly with `max_output_tokens` and no thinking headroom, where `call_claude` would add headroom for a `gpt-` model. Folding the route into `call_claude` removes `openai_schema_name`, `openai_reasoning_effort`, one branch of `send_model_request` and one of the resolver | step_runner.py:46-61, 78-90; update_steps.py `ModelRequest` fields | about 30 / about 20 | high | YES: the OpenAI request gets thinking headroom added, a changed `max_tokens`; needs regeneration (decision 1). Not recommended unless that change is wanted |
| B10 | Opt-in hooks each used by one step: `writes_stage_file_as_replies_arrive` and `_write_lock` (scoring only), `failure_file_name` and `failure_record` (scoring only), `stage_file_record_for_removal` (plan and description rewrite), `build_request_errors_count_as_thrown_calls` (description rewrite only), `failure_when_no_attempt_is_allowed` (scoring only; reachable only with `sdf.max_scoring_attempts < 1`), `message_when_failed_documents_stop_the_run` (two), `after_documents_are_processed` (two). Most are real differences in behaviour. Only two are cheap to unify: `build_request_errors_count_as_thrown_calls` as the default for every step, and `failure_when_no_attempt_is_allowed` replaced by validating the config | update_steps.py:99-170; step_runner.py:131-156 | about 8 / about 20 | medium: default thrown-call handling changes run flow (a template error becomes a recorded failure instead of a crash); inferred | no request; flow only |
| B11 | Unused and test-only production code: `DocumentState.last_failure` (document_state.py:433-435), `fold_round_and_batch_plan_events` (document_event_log.py:147-162), `document_state_final_corpus.py` (55 lines: `derive_final_corpus_records`, `final_corpus_record_for_state`), `FoldedDocumentSteps` fold used only by `run_corpus_step` and tests. They back the loader and event-fold tests, so they stay unless decision 7 drops work item 9 | listed | 3 to about 75 / their tests | n/a | no |
| B12 | `scripts/rerun_considerations_selector_on_stored_plans.py` (192 lines): the call path is covered by `run.py --seed-from <run> --from-step consideration_scoring --use-current-prompts` (`seeding.py` lists `consideration_scoring` as seedable, `SEEDABLE_STEPS` at seeding.py:64-72) and `--dry-run` by `scripts/print_full_sdf_request_a_stage_would_send_for_one_document.py`. Not covered: the comparison table against the original coverage report (`write_comparison_table`, script lines 59-77, about 20 lines), its own output folder and launch record outside `outputs/`. No file in the pipeline calls it (grep); `/Users/williamharrison/Desktop/emergent_alignment/scripts` has none; three plan or review documents mention it, and the project's `strictly_implicit_difficult_advice_pipeline_copy/scripts/` holds an unrelated copy. Seeding at this step with `--document-ids` omitted keeps all documents (seeding.py:228-229); a subset re-solves the assignment (decision 8) | scripts/rerun_considerations_selector_on_stored_plans.py | 192 (or about 170 plus the comparison table moved) / 0 | low | no |
| B13 | Four layout tables for the same three layouts: `shared/utils.py:70-97` (`_SDF_STAGE_DIRS`, `_SDF_STAGE_FILES`, `_SDF_TEMPLATES`, `sdf_stage_dir`, `sdf_template_path`, `sdf_stage_file`), `document_state_loader.py:65-69` and :87-100, `viewer/loader.py:41-57`, `website/sdf.py:86-97`. Cut 9 removes the last two; the first two remain (the loader deliberately imports nothing that writes) | listed | 0 beyond cut 9 | n/a | no |
| B14 | Stale docs and docstrings | update_steps.py:10-12 and :116-117 ("Only `DocumentStep` is run by `StepRunner` so far; `BatchStep` and `CorpusStep` are the interfaces the later work items move...": untrue since the screen, deal, redeal, assignment, gate and cull were converted); document_state.py module docstring ("so the later event log and step runner can build the same records"); CLAUDE.md:111 and :114 cite `sdf_pipeline/layer5_score.py` (no such file; it is `layer4_score.py`) and "layer 5's greedy near-dup cull"; prompts/README.md:116 "the near-duplicate cull in layer 5"; the five layer docstrings "`run` keeps its signature and behaviour as a thin wrapper, because scripts import it" (layer2_draft.py:10, layer3_rewrite.py:19, layer1_select_considerations.py:52, layer4_score.py:17, layer1_plan.py) | about 12 lines edited | none | no |
| B15 | Test helpers repeated across the new tests: `events_of` (test_sdf_consideration_scoring_and_assignment_steps.py:19, test_sdf_screen_and_redeal_steps.py:48, `_events_of` in test_sdf_document_event_fold.py:53), `reply` (test_sdf_critique_and_revise_steps.py:23, test_sdf_description_rewrite_step.py:40), `_records` (two loader test files: committed_runs...:36, reader_extensions...:23), `_rows` and `_stage_rows` (test_sdf_prompt_equivalence_and_final_corpus.py:111, test_document_state_loader_on_equivalence_fixture_scenarios.py:45), `run_main` and `_run_main` (test_sdf_seeding_from_a_chosen_step.py:94, test_e2e_smoke.py:29). Larger: the audit equivalence support copies the prompt one (`_build_fake_anthropic_class`, `_build_fake_openai_class`, `_wrap_to_note_stage_and_item`, `differences_between_values`, `describe_difference_of_texts`: near copies, differing in recorder type and one parameter) at sdf_audit_equivalence_fixture_support.py:308, 326, 350, 633 against sdf_prompt_equivalence_fixture_support.py:455, 474, 498, 1073 | tests only | 0 / about 25 small helpers, about 80 to 100 for the recorder fakes and diff functions | low, but the recording rule says recorded files are changed only by the regenerate scripts; helper moves must not change a recording | no |
| B16 | Cut 5 is smaller than the table implies: among this project's scripts, only `scripts/spec_contradiction_screen/plan_fresh_specifications.py:139` calls a layer `run` (`layer1_plan.run`); `run_end_to_end_pipeline_smoke_test.py:145` and `estimate_end_to_end_pipeline_smoke_run_cost.py:106` call `sdf_pipeline.run`. Others import helpers only (`layer1_screen.screen_hands` in run_screen_conditions.py:91, `layer1_description_rewrite.rewrite_description` in run_description_rewrite_on_fresh_descriptions.py:113, `build_*`, `load_*`). `scripts/generation_from_one_objective_considerations/set_up_generation_run_from_saved_consideration_scores.py:29` imports the selector module (cut 2) | cut 5 | dependency count: 1 script, not 12 | | |

Items checked and found not duplicated: per-layer thread pools (none in `sdf_pipeline/`: every pool goes through `utils.parallel_map` inside the runners; the other pools are in `evals/`, not touched by the stack); per-layer cost counting (none: cost lives in `shared/costs.py` and `api`); compatibility aliases added in the stack (none: `grep alias|compat|re-export` in `sdf_pipeline/` hits only `compose_prompts.py`'s existing `shared.matrix` re-export; `shared/api.py` added `ResolvedRequestSettings`, `resolve_request_settings`, `_temperature_for_the_wire`, `_effort_for_the_wire`, which `call_claude` uses, so none is dead).

`run_corpus_step`, `StepRunner` and `BatchStepRunner` overlap: shared fan-out and write ordering already common; duplicated pieces are B2 (event fields) and the two systemic-stop counts (step_runner.py:205-227 and 333-352, about 10 lines each, mergeable into one helper, about 8 lines saved, low risk). `layer1_screen.py` also builds the redeal's removed events by hand (layer1_screen.py:611-618) and repeats the `append_round_dealt_event` plus `append_corpus_step_finished_event` pair three times (:567-571, :631-639, :652-659); a small `record_round(...)` helper would save about 12 lines, no output change (inferred).

## Order I would suggest
1. B14 (docs), B1, B2, B5, B8, B3: no request or file change, small, each can pass the two equivalence tests.
2. B6, B4, B15, B12: low risk, moderate size.
3. Cut 9: after decisions 2 and 3, with the pure-module split first (it also removes the `shared/__init__` Python-version exit from the website path).
4. B7, B10: medium, touch stop behaviour.
5. Cut 10: only after a DAD equivalence recording exists, pilot on baseline.
6. B9: only if regenerating the OpenAI request recording is acceptable.

Prerequisites for every item that touches `sdf_pipeline/`: the prompt-equivalence test and the audit equivalence test must pass without regenerating; the regenerate guard (decision 1) blocks a recording change with a Python change. `strictly_implicit_difficult_advice_pipeline_copy/` in this project is a separate snapshot that the contributor kit copies from; I did not check whether it vendors any function named above (inferred: it does not follow this checkout).
