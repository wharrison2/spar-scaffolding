# SDF document state redesign: backwards compatibility risks

## What is being changed

Today each SDF stage writes its own jsonl file and copies a hand-picked set of fields from the previous stage's record. Some outputs (the rewritten description, the assigned considerations) live only in side files that are re-applied in memory on every resume, and drops are recorded in four different ways. The proposed redesign replaces this with:

- one `DocumentState` record per document, keyed by one document id for its whole life
- one append-only `document_events.jsonl` per run, where each stage appends an accepted, removed or failed event per document; the current state is a fold over the events, and the log is also the checkpoint
- derived views written after each stage: `documents_current_state.jsonl`, `document_removals.jsonl`, and `final/sdf_corpus.jsonl` as a filter on finished documents
- one loader, `load_document_states(run_dir)`, for every reader

This document lists what that change could break and the rules that avoid it.

## Two kinds of compatibility

| Kind | Must keep? | Why |
|---|---|---|
| Old runs stay **readable** | Yes | The website, viewer, evals and the persona-check and plausibility scripts read committed runs |
| Old runs stay **resumable** | Only for runs still in progress | Most of the existing compatibility code serves resuming; finished runs never resume |

## Risks, most serious first

| # | Risk | What breaks | Mitigation |
|---|---|---|---|
| 1 | Resuming runs in progress | Paid work is billed again, or plans held out by the description rewrite are treated as checked | A one-way adapter that builds events from the old checkpoint and side files, tested on real run folders; otherwise finish or abandon those runs on the old code first |
| 2 | Tools that set up a run by writing stage files, then call `--resume --layer N` | The layer 3 rerun setup script, the show-rejected-values paired test's shared deal and screen, and the end-to-end smoke test fail silently | A supported way to start from a chosen stage with selected documents (for example `--seed-from <run> --stage revise --document-ids ...`); keep `--layer` and `--run-id` working |
| 3 | Changing the final corpus format | The Hugging Face publisher (its `doc_id` column is public), the website, five evals, the persona check's required fields, and about 19 scripts in this project | Keep `final/sdf_corpus.jsonl` exactly as it is, including the `doc_id` field name, value and field order |
| 4 | Changing document ids | Joins across runs (the paired test, matched-spec comparisons) and every eval that resumes by `doc_id` (plausibility, variable persistence, persona check) | Keep the screen's hand id as the one id; it is already the same string as today's `prompt_id` and `doc_id` |
| 5 | Adding a fourth on-disk layout | Readers that only know some layouts; three already exist, and the website's pinned run uses the pre-renumber one | All readers go through one loader that knows every layout; keep read support for pre-renumber runs even if resume support is dropped |
| 6 | Old frozen prompt snapshots | Stage functions written for today's templates cannot render pre-renumber names, the single-call `layer3.txt`, or the `{reasoning_featured}` placeholder | Needed only if such runs must resume; otherwise refuse to resume them with a clear message |
| 7 | Recording score parse errors as failures instead of 5/5/5 | The website's gate count, which finds parse failures by the "Parse error." note; gate counts stop being comparable with old runs | Make the website's gate function handle both forms, and note the change wherever gate rates are compared across runs |
| 9 | Audit judge requests change when audits move to the loader (work item 5) | Plausibility, shoehorning and variable persistence scores, and persona check inputs, stop being comparable with earlier runs, with no test failing: for example if the loader hands an audit the original description instead of the rewritten one, or a different text version | An audit equivalence test, written against `origin/main` before any audit moves, compares every judge request on committed runs of each layout (see the plan's Tests section) |
| 10 | Scripts that import stage modules directly | 12 scripts in this project import `sdf_pipeline` layer modules or `compose_prompts` and call their functions (the scripts under `scripts/spec_contradiction_screen/` and `scripts/generation_from_one_objective_considerations/`, and the real-run replay in `scripts/sdf_document_state_redesign/`). Work item 7 changes those functions, so the scripts fail, and the replay, which guards every pull request from work item 6 on, stops working against new checkouts | Keep each layer module's entry function callable with today's arguments, as a thin wrapper over its step, until work item 8's seeding replaces the scripts that set up runs; update the replay's drivers in the same pull request that changes the code they drive, so the replay still runs against both `origin/main` and the new checkout |
| 11 | A resumed run can keep a different document of a near-duplicate pair | Today document order depends on whether a run was resumed, and the cull keeps the first of each pair; the new code always uses deal order. An uninterrupted run is unaffected | Noted in the pull request that makes the change and in the pipeline repository's `CLAUDE.md`; matters only when comparing a resumed run's final corpus document by document with an old one |
| 12 | New files in run folders | From work item 6 a run folder also holds `document_events.jsonl`, `documents_current_state.jsonl` and `document_removals.jsonl`. Scripts that copy or compare whole run folders, or treat every jsonl in a run folder as a stage file, see extra files | The names clash with no existing file; listed in the pull request that adds them |
| 8 | Conflicts with the open paired-test branch | `sdf-show-rejected-values-paired-test` changes `sdf_pipeline/` and `shared/` | Stack it below the cleanup branch, so the refactor is built on top of it |

## What people using the pipeline need to know

| Task | Can they act as if nothing changed? | What to know |
|---|---|---|
| Reading or analysing old outputs | Yes | Old run folders are not changed or rewritten. `final/sdf_corpus.jsonl`, `doc_id` and the stage files keep their format, so scripts that read files directly keep working. The loader is an optional convenience |
| Rerunning an audit on an old run | Yes, once the audit equivalence test passes | Audits send the same judge requests as before, so new scores are comparable with old ones |
| Starting a new run with `run.py` | Yes | Same command, config and flags (`--layer`, `--run-id` and `--resume` keep working); every request is the same. The run folder gains three extra files from work item 6 (risk 12) |
| Resuming a run started before the change | Yes up to work item 8; no after work item 9 | Today's stage files stay the checkpoint up to work item 8. If work item 9 is done, runs started on old code finish on old code or go through the adapter (risk 1). Runs from before the layer renumbering cannot resume on the new code at all (risk 6) |
| Comparing a resumed run's final corpus with an old one | Mostly | A resumed run may keep a different document of a near-duplicate pair (risk 11) |
| Setting up a run by writing stage files by hand, or calling layer functions from a script | No | These scripts need updating (risks 2 and 10); work item 8 adds a supported way to start from a chosen step |

The pipeline repository's `CLAUDE.md` gets a short note of the last three rows when the change that causes each one lands.

## Layouts that already exist

| Layout | Stage folders | Where runs in it live |
|---|---|---|
| Five-layer, before the variables matrix | `layer1` (document types) to `layer5` | Four committed runs |
| Matrix, before the layer renumbering | `layer12`, `layer3` to `layer5` | Three committed runs, including the website's pinned run |
| Current | `layer1` to `layer4` | No committed runs; only in this project's data folder and the show-rejected-values experiment folders |

## The cleanup branch

The cleanup branch (`william/sdf-cleanup-dead-code-and-shared-helpers`) breaks nothing outside the pipeline. The functions it removed are not imported elsewhere; `scripts/plausibility_audit/` defines its own functions with the same names. The removed `sdf_final` and `dad_final` keys still appear in the contributor kit's config template, but `load_config` ignores extra keys.

It does overlap files with the open paired-test branch, so it is rebased onto that branch in the pull request stack:

| Open branch | Files shared with the cleanup branch |
|---|---|
| `sdf-show-rejected-values-paired-test` | `config.yaml`, `prompts/README.md`, `sdf_pipeline/layer1_description_rewrite.py`, `evals/variable_persistence_sdf.py` |

## Rules for the redesign

1. `final/sdf_corpus.jsonl` and its `doc_id` field stay exactly as they are.
2. Only runs written in the new format resume on the new code. Runs in progress either finish on the old code or go through a tested adapter.
3. Every layout stays readable through one loader.
4. A supported way to start a run from a chosen stage replaces hand-written stage files.
5. The open paired-test branch sits below the cleanup branch in the pull request stack. `sdf-spec-contradiction-fixes` is a separate change outside this plan.

## Files and identifiers

Pipeline repository: `animal-welfare-data-pipeline/`
- Stage folder resolution: `shared/utils.py` (`sdf_stage_dir`, `sdf_stage_file`, `sdf_template_path`)
- Viewer layout table: `viewer/loader.py`
- Website gate and lineage: `website/sdf.py` (`gate`, `read_lineage`)
- Final corpus writer: `sdf_pipeline/layer4_score.py`
- Hugging Face publisher: `evals/publish_hf.py`
- Committed runs: `outputs/sdf/runs/` (pinned website run: `2026-07-25_15-57_fullscale-500-opus5`)

This project:
- Layer 3 rerun setup: `scripts/generation_from_one_objective_considerations/set_up_layer_3_rerun_on_selected_drafts.py`
- End-to-end smoke test: `scripts/spec_contradiction_screen/run_end_to_end_pipeline_smoke_test.py`
- Runs in the current layout: `data/spec_contradiction_screen_runs/small_sample_checks/*/pipeline_outputs/sdf/runs/`
- Paired test: `animal-welfare-data-pipeline-show-rejected-variable-values/experiments/show_rejected_values_paired_test/`, runs under its `outputs/sdf/experiments/show_rejected_values_paired_test_2026-11-07_*/`
- Persona check required corpus fields: `animal-welfare-data-pipeline/evals/log_probability_persona_check/build_scoring_inputs.py`
