# SDF document state redesign: code that can now be cut, and decisions outstanding

## Where things stand

The redesign is a stack of 16 draft pull requests in `sentfutures/animal-welfare-data-pipeline`, each based on the one before, all approved by the review bot, none merged: #194 equivalence test, #195 cleanup, #196 docs, #197 document state and loader, #198 audit equivalence test, #199 evals on the loader, #200 runner and event log, #201 to #206 every stage converted to steps, #207 request settings resolver and print command, #208 seeding from a chosen step, #209 dealt value ids. Top branch: `william/sdf-dealt-value-identifiers`.

Diff from `origin/main` to the top branch, Python only: about 11,000 lines added and 940 removed in 60 files (tests 6,751 added; `sdf_pipeline/` 3,660 added and 700 removed; scripts 415; `shared/` 104; evals 34 added and 71 removed). Recorded test fixtures (57 JSON and JSONL files, about 68,000 lines) are not counted.

Two tests make further cuts safe: the SDF prompt-equivalence test (every request, stage file, checkpoint and the final corpus compared with recordings made on `origin/main`) and the audit equivalence test (every audit judge request). A cut that passes both without regenerating their recordings changes no request and no output file. The real-run request replay (`scripts/sdf_document_state_redesign/` in this project) is the second check on stored real runs.

Full history, deviations and risks: `sdf-document-state-redesign-draft-pull-request-stack-writeup_2026-11-13_18-30.md` beside this file.

## Code that can be cut

None of these has been measured yet. "Needs a decision" points to the next section.

| # | Cut | What makes it possible | Needs a decision? |
|---|---|---|---|
| 1 | Delete `scripts/generation_from_one_objective_considerations/set_up_layer_3_rerun_on_selected_drafts.py` in this project | `run.py --seed-from <run> --from-step critique --use-current-prompts` (#208); a test proves identical requests | No |
| 2 | Retire the other hand-seeding scripts (`set_up_generation_run_from_saved_consideration_scores.py`, the paired test's shared deal and screen) | Seeding needs two additions first: assignment seeding from a separate scores file, and supplying an enriched `variables.txt` | No for the additions; the paired test lives on a branch outside the stack |
| 3 | Replace the per-stage readers `load_screen_outputs`, `load_description_rewrite_outputs`, `load_considerations_outputs` with the loader | Extend the loader with the screen's batch and round tables and the rewrite and selector tables; `audit_sdf.py` and `report_sdf.py` then use one reader | No |
| 4 | Turn `run.py` and the screen loop in `layer1_screen.run` into a step list (the plan's `Pipeline`) | Every stage is a step; ordering, resume, `--layer` restarts and stop rules are still written out per layer | No |
| 5 | Remove the thin `run()` wrappers kept in every converted layer | 12 scripts in this project import layer modules; move them to seeding or to the steps first | No |
| 6 | Drop resume support for pre-renumber runs (and the single-call layer 3 path if no run needs it) | Read support for every layout now lives in the loader; the plan allows dropping the resume part | Partly: whether any old-layout run must still resume |
| 7 | Prune older per-layer tests that the step tests and equivalence test now cover (`tests/test_sdf_layers.py` and similar) | Overlap with the new tests | No, but each removal must keep behaviour the equivalence fixture does not exercise |
| 8 | Retire the checkpoint code in every layer | Work item 9: the event log becomes the source of truth | Yes (decision 7) |
| 9 | Remove the website's and viewer's own stage-file parsing | Moving them onto the loader | Yes (decisions 2 and 3) |
| 10 | Replace DAD's own fan-out loop with `StepRunner` in `shared/` | The runner exists; DAD steps make several calls per item and have redraft loops | Yes: whether to extend the redesign to DAD |
| 11 | Remove the unused `outputs.sdf_final` and `outputs.dad_final` config keys | Kept in the cleanup because removing them changes the recorded configuration | Yes (decision 1): needs a regeneration |

## Decisions outstanding

1. **The regenerate guard.** The prompt-equivalence regenerate script refuses when pipeline Python differs from `origin/main`, and the test fails when a recording lists such files. So an intended request change that also touches Python cannot go green until it is on `main`. Options: make the guard informational (the recording's diff is the review); keep it and land such changes only after `main`; or accept red checks. Blocks the paired-test branch, the deal-order change (5) and cut 11.
2. **Website.** `website/sdf.py` promises a standard-library-only build, and the loader imports `shared/` and `sdf_pipeline/compose_prompts`. Options: relax the promise; split the loader's pure parsing into a module with no pipeline imports; or leave the website reading stage files.
3. **Viewer.** `viewer/loader.py` serves raw stage records and per-stage joins that document states do not carry. Options: extend the loader and add a viewer output test; or leave it.
4. **Layer 3 call order.** Since #202, every critique runs before any revise (requests unchanged; call and printed-line order differ). Accept, or restore per-document pipelining inside the two steps.
5. **Deal order for the near-duplicate cull** (plan: "deal order everywhere"; risk 11). Changes which twin a resumed run keeps, so it needs a regeneration (decision 1). Options: do it after the stack merges as its own pull request; or keep today's order.
6. **Dealt value ids in files.** #209 stores them in the document state and event log only. Writing them to stage files and `final/sdf_corpus.jsonl` adds a field to the published corpus.
7. **Work item 9.** Whether the event log becomes the source of truth for new runs and per-layer checkpoints are retired (enables cut 8). Before deciding: the event log stores stage-file records and request hashes, not reply text.
8. **Seeding a subset at consideration scoring.** #208 re-solves the assignment over the selected documents only. Keep, refuse that combination, or copy the source's selections.
9. **Review bot's non-blocking notes:** #208 `resolve_source_run` checks a local directory before the runs folder; #208 refusing `--from-step revise` under `sdf.revision.split: false` does not say why; #209 documents with no deal record can get "has no id" notes for the two consideration keys. Fix in a follow-up, or leave.

## Prompt for a new session

```
I'm continuing the SDF document state redesign in the animal-welfare-data-pipeline repository (inside this project). Read these first:
- plans/11-11-sdf-document-state-redesign/sdf-document-state-redesign-code-cuts-and-outstanding-decisions_2026-11-13_19-00.md (what can be cut, decisions outstanding)
- plans/11-11-sdf-document-state-redesign/sdf-document-state-redesign-draft-pull-request-stack-writeup_2026-11-13_18-30.md (what was done, deviations, risks)
- plans/11-11-sdf-document-state-redesign/sdf-document-state-and-update-steps-plan_2026-11-11_21-24.md (the plan)
These are on branch william/11-11-sdf-real-run-request-replay (worktree .claude/worktrees/sdf-real-run-request-replay).

The work is a stack of draft pull requests #194 to #209, each based on the one before; the top branch is william/sdf-dealt-value-identifiers, and each branch has a worktree under animal-welfare-data-pipeline/.claude/worktrees/. Nothing is merged; do not merge anything.

Task: measure each cut in the "Code that can be cut" table. For each, report lines removable (code and tests separately), what depends on it (including scripts in this project), whether it needs one of the outstanding decisions, and the risk. Do not cut anything yet; give me a ranked list so I can choose. Have Sonnet subagents do the measuring, read-only. Any later cut must pass the SDF prompt-equivalence test and the audit equivalence test without regenerating, and be checked with the real-run request replay.
```
