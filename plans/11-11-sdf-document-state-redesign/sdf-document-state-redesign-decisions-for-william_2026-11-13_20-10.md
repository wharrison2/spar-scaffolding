# SDF document state redesign: decisions for William

Every open decision on the redesign, its cleanup and the design improvements, in one list. Numbers 1 to 13 keep the numbering used in the cut list and the cleanup plan. "Blocks" names the work that waits on it.

## Open

| # | Decision | Options | Blocks |
|---|---|---|---|
| 1 | **Regenerate guard.** The prompt-equivalence regenerate script refuses when pipeline Python differs from `origin/main`, so an intended request change that also touches Python cannot go green until it is on `main` | Make the guard informational (the recording's diff is the review); keep it and land such changes only after `main`; accept red checks | The paired-test branch; 5; removing the unused config keys; retiring checkpoints; retiring the single-call layer 3 path |
| 4 | **Layer 3 call order.** Since #202 every critique runs before any revise (requests unchanged; call and printed-line order differ) | Accept; or restore per-document critique-then-revise, which design improvement 8 does while removing the layer's checkpoint stand-in and cross-runner counting | Design improvement 8 |
| 5 | **Deal order for the near-duplicate cull.** Changes which twin a resumed run keeps | Its own pull request after the stack merges, regenerating; or keep today's order | Needs 1 |
| 6 | **Dealt value ids in stage files and `final/sdf_corpus.jsonl`** (today: document state and event log only) | Add the field to the published corpus; or keep them internal | none |
| 7 | **Event log as source of truth, checkpoints retired** (work item 9) | Yes; no | Retiring checkpoints; design improvements 10 and 11; whether test-only fold code stays |
| 8 | **Seeding a subset at consideration scoring** re-solves the assignment over that subset only | Keep; refuse that combination; copy the source's selections | none |
| 9 | **Review bot's optional notes on #208 and #209**: `resolve_source_run` checks a local folder before the runs folder; refusing `--from-step revise` under `split: false` does not say why; documents with no deal record can get "has no id" notes | Fix in a follow-up; leave | none |
| 10 | **Single-call layer 3 path** (`sdf.revision.split: false`). Three experiment arm files use it, and the audit equivalence test builds its current-layout run from it | Keep for experiments; retire it, rebuild the audit test on the split path and regenerate (needs 1) | Retiring that path |
| 13 | **Broken template**: a prompt-building error | Stays a crash; becomes a recorded failure for that document | The default in design improvement 2 (its declared shape does not wait) |
| 14 | **Dealt value ids when the screen is off.** Only the screen writes deal events; with it off, the plan layer deals by itself, so new runs without the screen get no ids. Design improvement 7 sends both through one deal path | Fix as part of design improvement 7; leave | none |
| 15 | **Size of the resume check's recordings**: 2.6 MB added to the repository | Accept; shrink with hashes as the equivalence recording was | The cleanup pull request's size |
| 16 | **Which design improvements to do** (ranked list in `sdf-pipeline-design-improvements-after-refactor-investigation_2026-11-13_20-10.md`). None changes a request or recorded output file | Proposals 1 to 3 together with the cleanup's runner changes, then 4, then 5 with 6, then 7; 8 to 11 wait on 4 and 7 | — |

## Settled

| # | Decision | Outcome |
|---|---|---|
| 2 | Website onto the loader | No; the website keeps reading stage files |
| 3 | Viewer onto the loader | No; the viewer keeps reading stage files |
| 11 | Selector rerun script's comparison table | Dropped with the script (restorable from git) |
| 12 | Starting a run from saved consideration scores | Dropped with `set_up_generation_run_from_saved_consideration_scores.py` (restorable from git) |
| — | Branch `william/sdf-website-and-viewer-read-through-document-state-loader` | Holds nothing the stack lacks; delete when convenient |

## Paths

All in `plans/11-11-sdf-document-state-redesign/` on branch `william/11-11-sdf-real-run-request-replay`:
- Cut list and original decisions 1 to 9: `sdf-document-state-redesign-code-cuts-and-outstanding-decisions_2026-11-13_19-00.md`
- Cleanup plan (decisions 10 to 13): `sdf-document-state-redesign-cleanup-plan_2026-11-13_19-39.md`
- Design improvements: `sdf-pipeline-design-improvements-after-refactor-investigation_2026-11-13_20-10.md`
