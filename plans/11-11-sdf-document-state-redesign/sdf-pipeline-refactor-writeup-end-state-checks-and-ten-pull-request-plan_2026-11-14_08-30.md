# SDF pipeline refactor: end state, checks and the ten pull request plan

What this records: the refactor of the SDF pipeline in `animal-welfare-data-pipeline/` that began as the "document state redesign" (plan of 2026-11-11), what the code looks like at its end state, how that end state was shown to send the same requests and write the same outputs as `origin/main`, what was built and then removed on the way, and how the work will be resubmitted. Branch names, commits and output folders are listed at the end.

Status on 2026-11-14: the end state exists on local branches of the pipeline repository. Nothing in the pipeline repository is merged, and nothing more is pushed there until William says so. Draft pull requests #194 to #222 are on GitHub and will be replaced by about ten pull requests built from `origin/main`.

## The end state, against `origin/main`

### Layout

```
sdf_pipeline/
  run.py                                        runs the pipeline
  layer1_deal_and_screen/                       deal.py  screen.py  redeal.py  run_layer.py  shared_by_this_layers_steps.py
  layer2_plan_and_description_rewrite/          plan.py  description_rewrite.py  run_layer.py
  layer3_consideration_scoring_and_selection/   consideration_scoring.py  consideration_selection.py  run_layer.py  shared_by_this_layers_steps.py
  layer4_draft_critique_and_revise/             draft.py  critique.py  revise.py  critique_and_revise_in_one_call.py  run_layer.py  shared_by_this_layers_steps.py
  layer5_score_and_final_selection/             score.py  score_gate.py  near_duplicate_removal.py  run_layer.py
  pipeline_steps.py                             the step map
  start_a_run_from_an_earlier_runs_step.py      seeding
  shared_step_machinery/                        the Step record and runner, prompt composition, considerations, value ids
  compatibility_with_runs_made_by_earlier_code/ checkpoint marks of runs made before checkpoints were retired
```

On main the top level was seven flat modules (`layer1_screen.py` … `layer4_score.py`) whose names hid the stages inside them. Now a skim of the top level shows the five layers in order and every step in each.

### How a step works

- **One kind of step.** Each model-calling step file defines plain functions and one `STEP = Step(...)` record (a frozen dataclass, never subclassed) holding everything about the step: name, folder, model settings, what it reads and writes, prompt files, when it is inactive, its stop rules, its messages and its functions. `Step` has 24 fields.
- **Steps act on lists of documents.** `build_request(documents, run)` and `read_reply(documents, reply, run)` always take a list. Only the screen groups documents (seeded batches of 10); every other step gets lists of one.
- **One runner** (`run_step`) for every model-calling step. It is the only writer of stage files, skips finished work, retries, numbers attempts and applies the stop rules.
- **Steps with no model call** (deal, redeal, consideration selection, score gate, near-duplicate removal) are plain functions.
- **A `run` object** carries the configuration, prompts folder, stage folders and inputs loaded once per run.
- **The step map** (`pipeline_steps.py`, 55 code lines) is the ordered list of steps. `run.py` restarts, seeding and the full-request print command all read it.
- **Stage files are the only record of finished work for new runs.** No checkpoint files are written. The screen writes a batch's rejections before its judgments, so a batch whose hands all have judgments is complete.
- **Runs made by earlier code** still resume exactly as before: their checkpoint marks are read by one 19-line module in `compatibility_with_runs_made_by_earlier_code/`, the only code that knows checkpoint file names. Where a screen checkpoint file exists, only its marks count, as before.
- **One failure rule:** an error building a request (for example a broken template) fails that document, not the run.

### Features kept from the stack

- **Seeding:** `run.py --seed-from <run> --from-step <step>` starts a new run from a chosen step of an earlier run, for selected documents.
- **Full-request print command:** `scripts/print_full_sdf_request_a_stage_would_send_for_one_document.py` prints the exact request a step would send, built with the run's own code.
- **One request-settings resolver** shared by the pipeline and the print command.
- **One deal path**, and dealt value ids written into stage files and the final corpus for new runs.
- **Run layouts** described once, in `shared/sdf_run_layouts.py`, including layout detection.
- **Evals read stage files directly**, as on main, through `utils.sdf_stage_file`.

### Names (decision 17)

New runs write stage folders `layer1_deal_and_screen` to `layer5_score_and_final_selection` and cost-log tags `layer2_plan`, `layer2_description_rewrite`, `layer3_consideration_scoring`, `layer4_draft`, `layer4_critique`, `layer4_revise`, `layer4_critique_and_revise_in_one_call`, `layer5_score` (`layer1_screen` unchanged). Settings `consideration_scoring_model` and `revise_model`; the old `select_model` and `rewrite_model` still work with a one-line notice. Runs in main's four-layer layout resume in their own folders and tags; older layouts can be read but not resumed.

### Documentation

The pipeline's root `CLAUDE.md` went from 14,400 to 3,200 words: an index with a short summary of each part of the pipeline, the rules that apply everywhere, and a table of where to read more. Agent detail moved to folder `CLAUDE.md` files (`website/`, `sdf_pipeline/`, `dad_pipeline/`, `evals/`, `tests/`), which Claude Code loads only when an agent works in that folder; READMEs hold what people need.

## What changes for a user of the pipeline

Every request, stage file record, final corpus and eval output is unchanged, except:

- the stage folder names, cost-log tags, step names and two setting names above, for new runs;
- checkpoint files are no longer written for new runs;
- the systemic stop message names "an error building the prompt" among its causes;
- a request that cannot be built fails its document instead of stopping the run;
- the refusal messages for seeding a step that cannot be seeded are generated from the step map;
- `variable_value_ids` is written to stage files and the final corpus of new runs (manifest key `dealt_value_ids_in_stage_files`).

## Checks

| Check | What it compares | Result at the end state |
|---|---|---|
| SDF prompt equivalence | 33 scenarios run end to end on canned replies (failures, resumes, layer restarts, configuration variants): every request, stage file, final corpus, exit message and call count against recordings | passes; recordings regenerated only where the names, the stop message wording, value ids in stage files, checkpoint files or the recording format were meant to change, each time with a comparison showing nothing else differed |
| SDF audit equivalence | every audit judge request, on runs of three layouts | passes; recordings regenerated once, for the format change only |
| DAD request settings | every DAD stage's request on three backends | passes, recordings unchanged |
| Resume from main | 18 interruption points of runs made by `origin/main`, resumed with the current code | passes, recordings unchanged |
| Real-run request replay | 420 requests rebuilt from 9 stored real runs, against `origin/main` | 0 differences after every change |
| Eval outputs | `evals/report_sdf.py` and `evals/audit_sdf.py` on two committed runs of different layouts | byte-identical (audit with `PYTHONHASHSEED=0`, see below) |
| Full suite | | 2,624 passed, 5 skipped |

One command now runs the suite, the fixture diff, `compileall`, both command-line `--help`s and the replay against a cached `origin/main` baseline, in about 50 seconds: `scripts/sdf_document_state_redesign/check_sdf_pipeline_branch_before_pushing.py` in this project.

Test pruning deleted 96 tests whose every assertion a recording pins. For 22 of them the behaviour was broken on purpose and a recording test failed each time. Tests of behaviour no recording exercises, money-path tests for each stage and any test in doubt were kept.

## Size

| | `origin/main` | End state |
|---|---|---|
| `sdf_pipeline/` lines (files) | 3,675 (14) | 5,810 (45) |
| of which code | 2,518 | 3,736 |
| of which docstrings | 650 | 1,158 |
| `shared/` lines | 4,232 | 4,535 |
| test lines (files) | 28,439 (64) | 34,974 (100) |
| recorded fixtures | 0.15 MB | 5.97 MB |

Across the repository, production code is +6,612 / −2,877 and tests +8,059 / −1,495 against `origin/main`.

## Built and then removed

- **The document state** (one record per document for its whole life, its loader for four run layouts, and the evals reading through it): about 1,150 code lines and 850 test lines. Once stage files were kept as the run's storage (decision 3), it was a second, read-only picture of a run that the pipeline used only for layout detection and the final corpus record.
- **The document event log** (decision 18).
- **The step class hierarchy** (`DocumentStep`, `BatchStep`, `CorpusStep`), the batch runner and the 17-field step catalogue entries, which described each step twice.
- **Old module names** kept as alias modules after the move into layer folders.
- **A recording format storing each prompt text once**, proposed and then dropped: the recordings already stored texts as hashes. The aim behind it, small regeneration diffs, was met instead by dropping an unread list of changed files and keying repeated values by path. A one-word template change now regenerates about 730 lines instead of about 3,400.

## Interpretation

The refactor bought structure rather than fewer lines. A step is read in one file, there is one runner and one description per step, the layers are visible from the folder, and resumption no longer depends on two records agreeing. The code grew by about 1,200 lines against main. The features account for part of it (seeding, the print command, value ids, layouts); docstrings nearly doubled; and splitting seven files into 45 added imports and spacing. The docstring trim is where most of the remaining size can come off.

The checks carry the claim that behaviour is unchanged. The recordings pin exact requests and files over the scenarios; the replay covers real configurations the scenarios do not; the mutation checks show the recordings catch the behaviour the deleted tests checked.

## Remaining work

1. **Docstring trim:** one-line purpose plus any rule a reader needs; no history. Move the remaining compatibility code (older layouts, old setting names) into the compatibility folder.
2. **Retire the single-call critique and revise (decision 10):** written as its own pull request, held until William's go-ahead.
3. **Rebuild as about ten pull requests from `origin/main`**, locally, each passing the check command, with the final tree identical to the end state:

| # | Pull request |
|---|---|
| 1 | SDF prompt equivalence test |
| 2 | SDF audit equivalence test |
| 3 | Request settings resolver and DAD request check |
| 4 | `Step` record, one runner, every step converted |
| 5 | Five layers, one folder per layer, renames, layout table |
| 6 | Step map and the full-request print command |
| 7 | Seeding |
| 8 | One deal path and dealt value ids in stage files |
| 9 | One failure rule, stage files as the only record of finished work, dead code removed |
| 10 | `CLAUDE.md` index and folder files |
| — | Resume-from-main test, folded into 4 or 9 |
| — | Single-call retirement, held |

4. **Review bot:** a one-line workflow change treating `tests/fixtures/**` as generated data, on its own branch, so regenerations are listed by name and size instead of reviewed line by line.
5. **After the stack merges:** outer scripts that import moved pipeline modules need new paths (`scripts/spec_contradiction_screen/*`, `scripts/generation_from_one_objective_considerations/*`); they read the plan folder as layer 1, which is layer 2 for new runs.
6. **Bug found, not fixed:** `evals/audit_sdf.py` orders ties in its repeated 5-gram list by Python's hash order, so that list varies between identical runs unless `PYTHONHASHSEED` is set.

## Branches, commits and outputs

Pipeline repository, local branches, each based on the one before:

| Branch | Head | Content |
|---|---|---|
| `william/sdf-five-layers-and-step-renames` | `8fecd6b` | #220: five layers, renames, layer folders (pushed) |
| `william/sdf-claude-md-index-and-per-folder-readmes` | `622a7f6` | #221: `CLAUDE.md` index (pushed) |
| `william/sdf-remove-document-state` | `146bdeb` | #222: document state removed (pushed) |
| `william/sdf-one-kind-of-step` | `86b4c7f` | one kind of step |
| `william/sdf-step-record-fewer-fields` | `c064ec3` | fewer `Step` fields, merged with `william/sdf-recordings-store-each-prompt-once` (`c0fc226`, stable recording identifiers) |
| `william/sdf-retire-checkpoints` | `f3d09a9` | checkpoints retired, compatibility folder |
| `william/sdf-prune-tests-the-recordings-cover` | `78e0585` | test pruning; the end state |
| `william/review-bot-skips-recorded-test-fixtures` | `a89e33d` | review bot change, from `origin/main` |

Worktrees: `animal-welfare-data-pipeline/.claude/worktrees/<branch name without william/>`, each with a `WORKTREE_STATUS.md`.

This project: replay tool and check command in `scripts/sdf_document_state_redesign/`; check runs in `data/sdf_pipeline_branch_check_<branch>_<date and time>/`; cached `origin/main` baseline in `data/sdf_pipeline_origin_main_baseline_rebuilds/cc64a74519a0ef7b5bad1fff1b13a097ae28c0ad/`; earlier replays in `data/sdf_real_run_request_replay_*`. Decisions: `plans/11-11-sdf-document-state-redesign/sdf-document-state-redesign-decisions-for-william_2026-11-13_20-10.md`.
