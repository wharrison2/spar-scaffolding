# SDF document state and update steps: plan

## Goal

Make it clear what every SDF document goes through before it is final, and give the pipeline one representation of a document and one representation of a step that updates it. Hard constraint: **no request sent to any model changes.** Prompts, system prompts, models, token limits, reasoning effort, schemas and temperatures stay byte-identical, as do the number of calls per stage.

## The steps a document goes through today

| # | Step (module) | What happens | Can drop the document? |
|---|---|---|---|
| 1 | Deal (`compose_prompts`, no model) | A hand of variables is deck-sampled and repaired by the deal rules | No |
| 2 | Implausibility screen (`layer1_screen`) | Hands judged 10 per call; removed hands redealt for up to 3 rounds | Yes: extremely implausible |
| 3 | Compose planning prompt (no model) | Entity names and variable lines put into the planning template | No |
| 4 | Plan (`layer1_plan`) | Working notes plus a document description | Yes: declared incoherent |
| 5 | Description rewrite (`layer1_description_rewrite`) | Description kept or rewritten; original and changes kept | Yes: dealt-value conflict |
| 6 | Consideration scoring (`layer1_select_considerations`) | Each moral and epistemic entry scored 1 to 5 | No |
| 7 | Consideration assignment (`considerations_assignment`, no model) | Global linear programme assigns entries | No |
| 8 | Draft (`layer2_draft`) | First full text | No |
| 9 | Critique (`layer3_rewrite`, first call) | A list of problems against eleven review checks | No |
| 10 | Revise (`layer3_rewrite`, second call) | Rewritten text, with voice and length directives | No |
| 11 | Score (`layer4_score`) | Alignment, realism and spec conformance, 1 to 10 | No |
| 12 | Gate and near-duplicate cull (`layer4_score`, no model) | Alignment and realism must reach 7; near-duplicates above 0.90 dropped | Yes: below threshold, near-duplicate |

## Design

### Document state

- One `DocumentState` per document, keyed by one id for its whole life: the screen's hand id, which is already the string used as `prompt_id` and `doc_id`.
- It holds the dealt variables, planning notes, every description version, assigned considerations, every text version (each revision paired with the critique that produced it), scores, the furthest step reached, the last failure if any, and an outcome (in progress, final or removed, with a reason code).
- **Each text version records the description, variables and considerations it was written from.** Today the draft record carries these forward, and the critique, revise and score requests read them from the draft or rewrite record, not from the latest plan. The new steps do the same, so a later change to a plan or an assignment cannot alter a request about a text that already exists.
- **The `variables` dict stays as it is stored today**, and prompts keep being rendered from it. Today that one dict holds three things: the dealt composition cards, the revision cards that only layer 3 uses, and the `moral_considerations` and `epistemic_considerations` lists added by the assignment. The state adds three read-only properties that each return one of those groups by filtering the dict's keys: `dealt_cards`, `revision_cards` and `assigned_considerations`. They store nothing and change no file, so readers no longer need to know which keys belong to which group.
- **Lookups of an axis by name go through one helper** (document type, centrality, sentient category, and the language parsed from the culture value), replacing the `.get("...", "")` calls spread across the layers. This is done as each layer becomes a step.
- **Document order is deal order** (round, then hand index). This is today's order for an uninterrupted run. The near-duplicate cull keeps the first document of each pair, so it uses this order.
- One loader, `load_document_states(run_dir)`, used by every reader. It builds the state from today's stage files, for every layout that exists, and from the event log once a run has one.

### Event log

- One append-only `document_events.jsonl` per run. The state is a fold over the events.
- Most events are one outcome for one document from one step. Three kinds are not per document: a round was dealt (with its hand ids), a round's screen batches were planned (with their members), and a corpus step finished.
- Each event from a model call records a hash of the request that was sent.
- The runner appends an event as each reply arrives, under a lock, so a slow call never holds back the record of finished calls behind it. Order comes from the deal, not from the log.
- The loader ignores an unparseable last line, which is what an interrupted write leaves.
- Derived views, written after each step: `documents_current_state.jsonl`, `document_removals.jsonl`, `final/sdf_corpus.jsonl` (unchanged in format), and today's per-layer stage files.
- The log is written alongside today's files from work item 6. Today's files stay the source of truth and the checkpoint, written as they are now, until work item 9.

### Update steps

A step decides what to ask and what the answer means. It does no file or network input and output.

| Kind | Sees | Model call | Methods |
|---|---|---|---|
| `DocumentStep` | One document | One per document per attempt | `applies_to(state)`, `build_request(state) -> ModelRequest`, `interpret(state, reply) -> StepOutcome` |
| `BatchStep` | A group of documents | One per group | `group(states, seed, round_number)`, `build_request(batch)`, `interpret(batch, reply) -> {id: StepOutcome}` |
| `CorpusStep` | Every document | None | `apply(states) -> {id: StepOutcome}` |

A step returns one of four outcomes per document:

| Outcome | Meaning | Who may return it |
|---|---|---|
| `Accepted(update)` | A typed change to the state, such as a new description version, a new text version or scores | Any step |
| `Removed(reason_code, detail)` | The document leaves the pipeline | Any step |
| `Failed(kind, reason)` | `kind` is a thrown call (network, authentication, backend) or an unusable answer (truncated, missing tags, unparseable). The event is recorded, the document is not marked done, and a resume retries it | Steps that call a model |
| `Created(new_state)` | A new document enters the pipeline | Corpus steps only (the deal and the redeal) |

`ModelRequest` carries everything a layer passes to the model call today:

- system prompt, user message, model and JSON schema
- max tokens, which may be unset: the plan and score calls set none and take the config default
- the stage tag, unchanged from today (`layer1_screen`, `layer1_plan`, `layer1_description_rewrite`, `layer1_select`, `layer2_draft`, `layer3_critique`, `layer3_revise`, `layer3_rewrite`, `layer4_score`). The tag selects the reasoning effort from the `effort` map in `config.yaml`, and the screen and the consideration scoring build their shuffle seeds from it
- item id, which for the screen is the batch id
- caching flag and temperature override
- for the description rewrite's OpenAI route, the reasoning effort and schema name

Each step also declares three things, set to match its layer today (table below):

- **How many attempts it makes in one run.** One, except consideration scoring, which retries an unusable answer up to `sdf.max_scoring_attempts` times before failing.
- **Which failures count towards the systemic stop** (every pending call failed, so the run stops).
- **What a failed document does to the run**: it is left behind and the run carries on, or the run stops before the next step.

`StepRunner` does the work every layer now repeats: pick pending documents, fan out, call the model, append events, write today's stage files on the main thread in input order, print progress, and apply the step's stop rules. `Pipeline` is the list of steps, with one loop: screen and redeal repeat for up to `sdf.screen_redeal_rounds` rounds. `run.py` becomes that list plus `--from-step`.

Before any call, a step checks that the run's frozen prompt snapshot has the template files and placeholders it needs, and refuses to run with a message naming the run and the step if not (as layer 3 does today for a snapshot without the two-call templates).

### Today's steps as update steps

| Step | Kind | A failed document | Counts towards the systemic stop | Notes |
|---|---|---|---|---|
| Deal | `CorpusStep` | | | Returns `Created` for each hand |
| Screen | `BatchStep` | A failed batch stops the run before the redeal | Thrown calls and unusable answers | Batches are a seeded shuffle of the sorted hand ids, seeded from the run seed, the stage tag and the round number. The batch plan is recorded before the round's first call and reused on resume |
| Redeal | `CorpusStep` | | | Runs once every batch of the round is judged. Returns `Removed` for the screened-out hands and `Created` for the new ones. When the removed hands cannot be redealt into different hands that satisfy the rules, their cards are dropped and the loop ends. Hands still removed after the last round are dropped |
| Plan | `DocumentStep` | Left behind | Thrown calls | Composes the planning prompt itself, using today's composer |
| Description rewrite | `DocumentStep` | The run stops after the step | Thrown calls and unusable answers | |
| Consideration scoring | `DocumentStep` | The run stops before the assignment | Thrown calls and unusable answers | Skips documents in the no-welfare-stake arm |
| Consideration assignment | `CorpusStep` | | | Runs once every eligible planned document is scored |
| Draft | `DocumentStep` | Left behind | Thrown calls | |
| Critique | `DocumentStep` | Left behind | Thrown calls | A reply with no review tags is accepted whole. An accepted critique is kept when the revise call fails, so a resume does not pay for it again |
| Revise | `DocumentStep` | Left behind | Thrown calls | Voice and length directives read from the state; a `revision` override is a state field set when a run is seeded |
| Score | `DocumentStep` | A thrown call is left behind | Thrown calls | An unparseable answer is accepted as 5/5/5 with the note "Parse error.", as today |
| Gate | `CorpusStep` | | | Runs over the documents that have scores, without waiting for those left behind |
| Near-duplicate cull | `CorpusStep` | | | Runs over the documents that passed the gate |

A stage that is disabled in config, or whose template is missing from the run's snapshot, makes no calls and passes every document through, as today.

The legacy single-call layer 3 (`sdf.revision.split: false`) becomes a `CombinedReviewAndReviseStep`, kept only as long as runs that need it can resume.

### When an earlier result changes after later work exists

This happens today in one case. A plan that fails and then succeeds on resume adds a document to the assignment, the assignment is solved again over every document, and documents already drafted keep the text written from their earlier considerations.

The new code keeps that behaviour and makes it visible: the new assignment is a new event, the existing text versions still record what they were written from, and the state marks a document whose latest text was written from a different assignment than its current one. Applying a new gate threshold to stored scores works the same way: the gate and the cull run again and append new outcomes.

## Order of work

Each stage is its own pull request and must pass the prompt-equivalence test before merging.

1. **Prompt-equivalence test and real-run replay**, written against `origin/main` and run against the cleanup branch (`william/sdf-cleanup-dead-code-and-shared-helpers`), so the cleanup is proven to change no request. Once the test has merged, add the rule for its recorded requests (see Tests) to the pipeline repository's committed `CLAUDE.md`.
2. **Merge the two open SDF branches** (`sdf-show-rejected-values-paired-test`, `sdf-spec-contradiction-fixes`), each committing the regenerated recorded requests together with the prompt changes it makes, then rebase the cleanup branch.
3. **Docs:** the step table above goes into CLAUDE.md and the `run.py` docstring.
4. **Document state and loader, over today's files.** `load_document_states` reads all three layouts, including the website's pinned run. No pipeline code changes, so no request can change.
5. **Readers moved to the loader:** website, viewer, report and audit evals, shoehorning postprocess, the Hugging Face publisher.
6. **Runner and event log** (`StepRunner`, `ModelRequest`, outcomes), used first by one simple step (draft) while the other layers keep their own loops. Today's files remain the source of truth, and a test checks that the folded events match what the loader builds from the files.
7. **Steps moved over one at a time:** score and gate, critique and revise, plan, description rewrite, consideration scoring and assignment, then screen and redeal last.
8. **Supported seeding from a chosen step,** replacing the scripts that write stage files by hand.
9. **The event log becomes the source of truth** for new runs, and per-layer checkpoints are retired. Today's stage files are still written, as derived views. This item is optional: work items 4 to 8 already give one document representation, one step representation and one reader. Decide after work item 8 whether it is still worth doing.

**Follow-on pull request, after this plan's work: dealt value ids.** Today a hand records each value's text and drops its id from `variables.txt`, so a value reworded later can only be matched to an old run through that run's snapshot. The follow-on adds the ids, kept apart from the content that goes in prompts:

- A separate field, `dealt_value_ids` (axis to value id), beside `variables`. Nothing that builds a prompt reads it, and `variables` keeps holding only the text.
- For new runs the deal records the ids as it deals, in the deal event.
- For old runs the loader fills the field by looking each value's text up in the run's frozen `variables.txt`. An axis with no snapshot or no matching text gets no id and a note.
- It is its own pull request and is not part of work items 4 to 9.

Out of scope, because they change requests or behaviour: removing the description rewrite's own OpenAI routing (it changes where the reasoning effort and thinking headroom come from), and recording score parse errors as failures instead of 5/5/5 (same prompt, but extra calls on resume). Each can be done later as its own change, with the equivalence test updated deliberately.

## Risks

### Changing what is sent to models

| Risk | How it would happen | Guard |
|---|---|---|
| Different seeded choices | The id passed as `prompt_id` seeds the example species and the consideration-scoring shuffle; the stage tag is part of the screen and scoring shuffle seeds | Keep the id string and the stage tags; equivalence test |
| Different reasoning effort or token limit | The model call function sets these from the stage tag and config after the layer has made its call, so a renamed tag or a max tokens value filled in by the runner changes the request without changing the layer's arguments | Equivalence test records the request as sent, not the layer's arguments |
| Wrong description downstream | Layers 2 to 4 must get the rewritten description, not the original | Equivalence test; unit test on the state's current description |
| Later steps read the latest plan instead of what the text was written from | Critique, revise and score take description, variables and considerations from the draft or rewrite record today | Each text version records what it was written from; equivalence test with an assignment that changes after drafting |
| Different `variables` | Assignment adds the two consideration lists to `variables`, which later templates render | Equivalence test |
| Template line dropped or kept wrongly | An empty `considerations_spec` drops a line in layers 2 and 3 | Equivalence test with documents both with and without considerations |
| Different screen batches | Batches are a seeded shuffle saved before the round's first call; a different seed string, or a batch plan that is recomputed instead of reused, changes which hands share a call | Equivalence test; batch plan recorded as an event |
| Different request parameters | Model, max tokens, schema, temperature or caching flag set by the runner instead of the layer | Equivalence test compares these too |
| Different number of calls | A step retries where today's code does not, or the other way round | Equivalence test compares calls per stage, on a fixture that includes failures and a resume |
| The recorded requests go stale | Prompt changes merged by other work during the refactor | That work regenerates the recording in its own pull request, and the refactor branch rebases onto it; refactor pull requests never change the recording |

### Changing what a run does without changing a request

| Risk | How it would happen | Guard |
|---|---|---|
| A run that can never finish | A gate or cull that waits for every document, when one draft is refused on every attempt | Per-step failure policy in the step table; resume test with a document that always fails |
| Paid work lost on a crash | Events written in input order wait behind one slow call | Events appended as replies arrive; runner test |
| A different final corpus after a resume | Today's document order depends on whether the run was resumed, and the cull keeps the first of each near-duplicate pair | Deal order everywhere; the difference from today for resumed runs is noted in the pull request |
| Wrong systemic stop | Unusable answers counted where today only thrown calls are, or the other way round | `Failed` carries its kind; runner test per step |
| Text and considerations that disagree, unnoticed | The assignment is solved again after documents are drafted | State marks the document; fold test |

### Backwards compatibility

| Risk | Guard |
|---|---|
| Resuming runs in progress pays for work again or skips checks | Today's files stay the checkpoint until work item 9. After it, finish runs on the old code or resume them through the loader, tested on real run folders |
| Scripts that set up a run by writing stage files and resuming with `--layer` | Supported seeding (work item 8); keep `--layer` and `--run-id` |
| Scripts and evals that read intermediate stage files (a file-name search finds 21 scripts in this project and 9 files in the pipeline's evals, viewer and website) | Today's stage files are still written after work item 9, as derived views |
| Final corpus format changes | Keep `final/sdf_corpus.jsonl` and `doc_id` exactly as they are |
| Document ids change | Keep the screen's hand id |
| A fourth run-folder layout | One loader that reads all layouts, including the website's pinned run |
| Old frozen templates cannot be rendered by the new steps | Each step checks the snapshot for its files and placeholders and refuses with a clear message |
| Conflicts with open branches | Merge them first |

## Tests

All tests in the pipeline repository run offline with stubbed model replies and no API keys, inside the existing pytest suite, so CI runs them on every pull request.

### Must exist before any refactor

| Test | What it checks |
|---|---|
| **Prompt equivalence** | Runs the whole pipeline on a small fixture with canned replies and compares every request, as sent, against a committed file of requests recorded on `origin/main`. Compared per stage and item id: system prompt hash, user message, model, max tokens after thinking headroom, reasoning effort, schema, temperature, caching flag, stage tag, and calls per stage. Any difference fails. The recorded file is regenerated only by its own script. A pull request that refactors must leave it untouched, so a failure there is a bug. A pull request that changes requests on purpose runs the script and commits the new recording in the same pull request, so the recording's diff shows reviewers which requests changed |
| **Final corpus schema** | The fixture run's `final/sdf_corpus.jsonl` has the same fields, order and values as on `origin/main` |
| **Real-run replay** (a script in this project, not in CI, because the runs are not committed to the pipeline repository) | For each real current-layout run, rebuilds every request from the run's stored inputs and compares it with what `model_calls.jsonl` and `system_prompts/` say was sent. Run before each pull request from work item 6 on. It covers hundreds of real hands, which the fixture cannot |

The fixture covers:

- **Removals:** a screen removal with a redeal, an incoherent plan, a dealt-value conflict, a below-threshold score and a near-duplicate.
- **Variants:** documents with and without considerations, a no-welfare-stake document, and one run with each of the screen, description rewrite and consideration selection disabled.
- **Failures:** a truncated draft, an unusable consideration-scoring answer followed by a usable one, a critique with no review tags, a revise call that fails and is resumed, a score answer that does not parse, and a plan that fails and succeeds on resume after other documents are drafted.

### Added with the new code

| Test | What it checks |
|---|---|
| Loader across layouts | `load_document_states` reads one committed run of each layout (five-layer, pre-renumber, current) and the website's pinned run; the website, viewer and evals build from it |
| Loader on real runs | The state built from the real current-layout runs in `data/spec_contradiction_screen_runs/small_sample_checks/` and the paired-test arms matches their stage files |
| Step unit tests | Each step's `interpret` given reply strings: accepted, removed and failed for each case its layer handles today, with no stubbed API and no files |
| Runner | Pending selection, events appended as replies arrive, stage files written in input order, progress, each step's stop rules, and that `Failed` marks nothing as done |
| Resume | A run interrupted at each step resumes with zero calls for finished work, and retries exactly the failed documents. A document that fails on every attempt does not stop the final corpus being written |
| Event fold | The folded events match what the loader builds from the stage files on the fixture run (while both are written), including a log with a cut-off last line |
| Seeding | Starting from a chosen step with selected documents gives the same requests as today's layer 3 rerun setup script |
| Snapshot check | A run whose frozen snapshot lacks a template or placeholder a step needs is refused with a message naming the run and step |

### Smoke check

- **Offline:** an end-to-end run on the fixture through `run.py`, run after every work item.
- **Paid:** one small run on real models after work item 8, and again after work item 9 if it is done. It runs only with explicit go-ahead and a project key from the `.env`. Its outputs go in `data/sdf_document_state_redesign_runs/small_sample_checks/`. It passes when:
  - the real-run replay finds every request it sent equal to what the `origin/main` code builds from the same stored inputs
  - the final corpus has the same fields as an `origin/main` run
  - no stage failed for every document

  Counts of documents removed per step are reported beside an `origin/main` run of the same seed but are not required to match, because the replies are sampled.

## Context for whoever picks this up

### Where the work stands

- The cleanup branch `william/sdf-cleanup-dead-code-and-shared-helpers` has five commits on `origin/main` and is not pushed. It lives in the worktree `animal-welfare-data-pipeline/.claude/worktrees/agent-a80ef6175a6586ade`.
  - It removes dead code in the SDF evals, adds the shared `sdf_pipeline/structured_answers.py` and `DocumentFacts.from_plan`, merges duplicated layer 1 tests, and trims CLAUDE.md.
  - It changes no prompt file and no live config value.
- The main checkout of `animal-welfare-data-pipeline` has uncommitted DAD work: a modified `outputs/dad/id_registry.json` and an untracked run folder under `outputs/dad/runs/`. Work in a worktree and leave it alone. The git stash stack is shared between worktrees, so never use a bare `git stash`.

### Test baseline

- The suite needs `pytest-socket` installed. On the cleanup branch it gives 2,224 passed, 30 failed and 17 skipped.
- The 30 failures exist on `origin/main` too: `tests/test_embeddings.py`, `TestEntityPools` in `tests/test_textstats.py`, and `tests/test_log_probability_persona_check_smoke_test.py` (it needs a real qwen3 model). A change is clean when it adds no failures to that set.

### Building the prompt-equivalence test

- **Record below the model call function.** `shared.api.call_claude` resolves the model, adds the thinking headroom, looks up the reasoning effort from the stage tag and applies the config temperature before it sends anything. Replace `shared.api._call_with_retry` with a recorder that returns the canned reply, so the test sees those resolved values. The suite already replaces this function with one that raises.
- **Don't rely on `model_calls.jsonl` in tests.** It is written under `call_claude`, so a test that patches `call_claude` never writes it.
- **Record the OpenAI route too.** The description rewrite defaults to `gpt-6.1-sol`, which goes to `shared/openai_api.py`, not `call_claude`. Stub it with `stub_openai_responses`, which also records the reasoning effort and token limit sent.
- **Canned replies must be chosen by request, not by queue.** Steps fan out across threads, so a queue of replies would be handed out in an unpredictable order.
- **Keep output out of the repository.** Point `PIPELINE_OUTPUT_ROOT` at `tmp_path`.
- **Use the real request records for the replay.** Real runs store each call's user message in `model_calls.jsonl` and each distinct system prompt once in `system_prompts/<sha256>.txt` next to it.

### Where the code goes

`DocumentState`, the loader, `ModelRequest`, the outcome types, `StepRunner` and the SDF steps go in `sdf_pipeline/`. The DAD pipeline repeats a similar fan-out loop, but its steps make several calls per item and have redraft loops, so the runner moves to `shared/` only when DAD is ready to use it.

### How this interacts with the other cleanup items

| Item | Relation to this plan |
|---|---|
| Shared fan-out, checkpoint, progress and systemic-abort loop | Becomes `StepRunner` (work item 6) |
| Removing the description rewrite's own OpenAI routing | Out of scope: it changes requests |
| Shared skeleton and helpers for the model-judged audits | Independent; those audits read final corpora and move to the loader in work item 5 |
| Removing the code for runs made before the layer renumbering | Only the resume part can go. Read support moves into the loader, because the website's pinned run uses that layout |
| `scripts/rerun_considerations_selector_on_stored_plans.py` | Reads stored plans and scores directly: archive it, or move it to the loader in work item 5 |

### Out-of-date docs to fix in work item 3

- `CLAUDE.md` still uses the old layer numbers in places. It cites `sdf_pipeline/layer5_score.py` for the corpus write order and says "the layer-4 rewrite and layer-5 scoring" in the model overrides paragraph.
- `prompts/README.md` says SDF has five layers.

### Rules that apply

- **This project:**
  - Sonnet 5.5 subagents write the code.
  - Commit on a branch, never directly to main.
  - Don't run anything that calls a model API without explicit go-ahead, including small tests, and only with a project key from the `.env`.
  - Write names out in full.
  - Generated files end with their date and time.
- **The pipeline repository's CLAUDE.md:**
  - Run `pytest` after every functional change.
  - Every pull request has a "How to test" section.
  - Replies to review comments are posted by a person, not an agent.
  - Tests encode current behaviour.
  - Committed run outputs go in a separate pull request from code.
  - Never run `evals/publish_hf.py` unless asked.

### Open decisions

- Whether to push the cleanup branch and open a pull request.
- Whether to do work item 9 at all, decided after work item 8.
- Whether any current-layout run must still resume on the new code after work item 9. This decides between resuming through the loader and finishing those runs on the old code.
- Whether the assignment should be solved once and then kept, instead of being solved again when a late plan arrives. Keeping it would change which considerations late documents get, so it is a behaviour change to make deliberately.
- Whether a document that fails on every attempt should get a "gave up" removal after a set number of resumes. Today it stays in progress for good.
- Whether the two out-of-scope changes (OpenAI routing, recording score parse errors as failures) are wanted later.
- Whether the follow-on's value ids are also written to today's stage files and `final/sdf_corpus.jsonl`, or only to the state and the event log. Writing them to the files adds a field to the published corpus, so the equivalence test's recorded files would be regenerated on purpose.

## Files and identifiers

Pipeline repository: `animal-welfare-data-pipeline/`
- Orchestrator: `sdf_pipeline/run.py`
- Layers: `sdf_pipeline/layer1_screen.py`, `layer1_plan.py`, `layer1_description_rewrite.py`, `layer1_select_considerations.py`, `considerations_assignment.py`, `layer2_draft.py`, `layer3_rewrite.py`, `revision_style.py`, `layer4_score.py`
- Model call function and the point the equivalence test records at: `shared/api.py` (`call_claude`, `_call_with_retry`; also writes `model_calls.jsonl`); stub fixtures `stub_claude` and `stub_openai_responses` in `tests/conftest.py`
- Reasoning effort map and thinking headroom: `config.yaml` (`effort`, `thinking_headroom`)
- Screen batch planning and its seed: `plan_round_batches` in `sdf_pipeline/layer1_screen.py`; consideration-scoring shuffle: `shuffled_offer` in `sdf_pipeline/layer1_select_considerations.py`
- Stage folder resolution: `shared/utils.py`; viewer layout table: `viewer/loader.py`; website gate and lineage: `website/sdf.py`
- Committed runs: `outputs/sdf/runs/` (website's pinned run: `2026-07-25_15-57_fullscale-500-opus5`)
- Cleanup branch: `william/sdf-cleanup-dead-code-and-shared-helpers`

This project:
- Backwards compatibility details: `plans/11-11-sdf-document-state-redesign/sdf-document-state-redesign-backwards-compatibility-risks_2026-11-11_20-48.md`
- Real current-layout runs: `data/spec_contradiction_screen_runs/small_sample_checks/*/pipeline_outputs/sdf/runs/`
- Scripts that seed runs by hand: `scripts/generation_from_one_objective_considerations/set_up_layer_3_rerun_on_selected_drafts.py`, `scripts/spec_contradiction_screen/run_end_to_end_pipeline_smoke_test.py`, and the paired test in `animal-welfare-data-pipeline-show-rejected-variable-values/experiments/show_rejected_values_paired_test/`
