# SDF pipeline: design improvements the refactor makes possible

Read-only investigation of the top of the stack (`william/sdf-dealt-value-identifiers`, `3683685`, worktree `animal-welfare-data-pipeline/.claude/worktrees/sdf-dealt-value-identifiers`), against `origin/main`. Line numbers are from that worktree. "(inferred)" marks a claim I reasoned to from the code without running or testing it; everything else was read in the code.

Builds on, and does not repeat, the cleanup already in progress: stale docs, deleting `score_document` and dead result fields, the runner building the event log, one outcome-to-event helper, steps declaring their stage-file id field, runner options replacing `_NoCheckpoint` and `_CritiqueCheckpointView`, shared failure-text and progress helpers, test helper merges, old test pruning, deleting the selector rerun script, writing only the current layout.

## The shape of the problem

The refactor gave every model call a step class, but most of the decisions about a step still live in four other places. These are the places a change to one step has to touch today:

1. **The step class** (`build_request`, `interpret`, `stage_file_record`, plus up to nine optional hooks).
2. **The layer's `run()`**, which decides whether the step is active (disabled in config, template missing), loads existing records, builds the runner, prints the header, runs, then post-processes (apply rewrite results, attach selections, gate, cull, write the final corpus).
3. **`run.py`**, which rebuilds each layer's input from files when `--layer N` restarts (`run.py:175-194`).
4. **`seeding.py`**, which keeps its own step order, file table, input-file table and template list (`seeding.py:58-96`, `:149-173`, `:217-224`, `:287-332`).
5. **The request print command** (`scripts/print_full_sdf_request_a_stage_would_send_for_one_document.py:59-65`, `:126-192`), which rebuilds each step's input a third time, under different stage names.

Most of the proposals below collapse 2 to 5 onto the step, or make the step the only place that knows something.

## Ranking (value for effort)

| Rank | Proposal | Size (lines touched) | Changes a request or output file? | Waits on |
|---|---|---|---|---|
| 1 | A. Steps that are pure: no file writes, no counters, no attempt bookkeeping | about 120 | No (one write-order fix, invisible to both tests) | none |
| 2 | B. One exception policy, declared, used by all three runners | about 40 | No, with today's per-step choices | decision 13 only for changing defaults |
| 3 | C. One final corpus record function | about 30 | No | none |
| 4 | D. `rewrite_description` built from the step | about 30 | No (inferred; test it) | none |
| 5 | E. A step catalogue: one entry per step that run.py, seeding and the print command all read | about 250 touched, net about -100 | No | best after cleanup 7 (current layout only) |
| 6 | F. A run prompt snapshot object | about 100 | No | none; pairs with E |
| 7 | G. Corpus steps through one recorder; one deal path | about 60 | Event log only | none |
| 8 | H. Layer 3 as a per-document chain of two steps | about 100 | No | decision 4 (call order) |
| 9 | I. Settings read once into a typed record | about 80 | No | none |
| 10 | J. Request hash over the resolved settings | about 15 | Event log only | decision 7 for its value |
| 11 | K. One fold that produces `DocumentState` | large (300+) | No, if done as designed | decision 7 |

Would not do: merging the three runners into one; steps reading `DocumentState` instead of record dicts (before decision 7); moving `ConsiderationAssignmentStep`; renaming `layerN_*` modules; a typed route field on `ModelRequest` on its own. Reasons at the end.

---

## A. Steps that are pure

The plan says "A step decides what to ask and what the answer means. It does no file or network input and output" (`update_steps.py:3-4`). Four kinds of exception have crept in.

### Problem

1. **A step writes a file.** `DescriptionRewriteStep.stage_file_record_for_removal` appends to `description_rewrite_rejections.jsonl` itself (`layer1_description_rewrite.py:360-368`), as a side effect of a method whose name says it returns a record. The runner then appends the results record. That is why the writeup's risk row says the rejection is now written *before* the result (it was after on `origin/main`), and why `StepRunner._removal_record` caches the call "asked for once (the step may have side effects)" (`step_runner.py:158-163`).
2. **A step reads files after the run.** `DescriptionRewriteStep.after_documents_are_processed` reloads `description_rewrites.jsonl` to count held-out plans (`layer1_description_rewrite.py:390-400`), and keeps `self.plans` and `self.stage_directory` for it (`:302-313`).
3. **Steps hold progress counters.** `DescriptionRewriteStep.begin_session` and `ConsiderationScoringStep.begin_run` reset counters that `progress_line` mutates (`layer1_description_rewrite.py:315-319`, `:370-388`; `layer1_select_considerations.py:346-351`, `:411-422`). The layer must call `begin_*` with the pending count before running (`layer1_description_rewrite.py:433`, `layer1_select_considerations.py:583`). `progress_line` is therefore not a pure formatter: calling it twice changes the next line. In the scoring step, which writes as replies arrive, it runs on worker threads under the runner's lock (`step_runner.py:123-128`), so its correctness depends on that lock.
4. **A step tracks attempts in shared dicts.** `ConsiderationScoringStep` keeps `_offered_by_document_id` and `_attempts_by_document_id`, written by `build_request` and `interpret` from worker threads (`layer1_select_considerations.py:342-343`, `:365`, `:378-393`), because the runner does not tell `interpret` which attempt a reply answers. `failure_when_no_attempt_is_allowed` (`update_steps.py:166-168`, `layer1_select_considerations.py:396-397`) exists only for `max_scoring_attempts < 1`.

### Proposed shape

- **Records per outcome, written by the runner.** Replace `stage_file_record_for_removal`, `failure_file_name` and `failure_record` with one hook that lists every record an outcome writes, in order, and whether it finishes the document:

  ```python
  @dataclass(frozen=True)
  class OutcomeRecords:
      records: tuple[tuple[str, dict], ...]   # (file name in the stage folder, record), written in this order
      finishes_the_document: bool             # mark done (checkpoint) after writing

  def outcome_records(self, document, outcome) -> OutcomeRecords
  ```

  Default: `Accepted` writes `(stage_file_name, stage_file_record(...))` and finishes; `Removed` and `Failed` write nothing. The description rewrite returns `[(RESULTS, result), (REJECTIONS, rejection)]` for a conflict, which restores the `origin/main` write order. Scoring returns `[(FAILURES, failure)]` for a failure. Plan returns its incoherent record.
- **Runner-owned progress.** The runner keeps `StepProgress(pending_count, answered, failed, removed, started_at, last_printed_at)` and passes it in: `progress_line(document, outcome, progress)`. The throttled "every 180 seconds" line becomes one runner helper (this is where the cleanup's shared progress helper should land). `begin_session` and `begin_run` go.
- **The summary hook gets what it needs.** `after_documents_are_processed(run_result)` already receives the result; give the result the records written this run and the existing ones, so the held-out count is `described ids - result ids`, with no file read.
- **The attempt number travels with the reply.** Add `attempt: int = 1` to `ModelReply` (the runner sets it, `dataclasses.replace`), so scoring's `interpret` reads `reply.attempt`. `offered` is a pure function of `(catalog, variables, seed, prompt_id)` (`shuffled_offer`, `layer1_select_considerations.py:143-150`), so `interpret` recomputes it instead of reading a dict. Validate `max_scoring_attempts >= 1` when the step is built, and `failure_when_no_attempt_is_allowed` goes. This covers the attempts half of the cleanup plan's waiting item "unify the two cheap runner hooks"; that half needs no decision.
- **`BatchStep` declares what the runner calls.** `BatchStepRunner` calls `step.progress_line`, `step.failed_batch_line` and `step.stop_message_for_failed_batches` (`step_runner.py:343`, `:348`, `:352`), which the `BatchStep` interface does not declare (`update_steps.py:179-195`). Add them as abstract methods.

### What gets simpler and what gets harder

Simpler: every step can be unit-tested with no temporary folder; "steps do no input or output" is true again; the runner is the only writer, so write order and crash windows are decided in one place; no thread-shared step state. Harder: `outcome_records` is a slightly bigger hook than `stage_file_record`, though it replaces three.

### Size, files, risk

About 120 lines across `update_steps.py`, `step_runner.py`, `layer1_description_rewrite.py`, `layer1_select_considerations.py`, `layer1_plan.py` and their step tests. No request or file content changes. The only output difference is the order of the two writes for a dealt-value conflict, which goes back to `origin/main`'s (results, then rejection); neither equivalence test sees it (it shows only after a crash between the writes). Risk to watch: the throttled progress text and the "Description rewrite N/M" line must stay word for word, including when they print (the last document always prints, `layer1_description_rewrite.py:382-383`). Step tests assert printed lines, so they pin it. Depends on nothing. Do it together with the cleanup's progress helper (3e) and removal-hook items so the hooks are changed once.

---

## B. One exception policy, declared

### Problem

What an exception means depends on which runner and which step:

- `StepRunner`: an exception in `send` is a thrown call (`step_runner.py:146-150`); one in `build_request` crashes the run unless the step sets `build_request_errors_count_as_thrown_calls` (`:133-140`, set only by the description rewrite, `layer1_description_rewrite.py:300`); one in `interpret` always crashes (`:152` is outside the `try`).
- So two steps catch everything inside their own `interpret` to get thrown-call behaviour: `layer1_description_rewrite.py:347-348` and `layer1_select_considerations.py:390-392`.
- `BatchStepRunner` treats an exception anywhere (build, send, interpret) as a thrown call (`step_runner.py:303-313`).

### Proposed shape

```python
class WhereAnErrorIsAThrownCall(Flag):
    SEND = auto(); BUILD = auto(); INTERPRET = auto()

@dataclass(frozen=True)
class StepPolicy:
    ...
    errors_counted_as_thrown_calls: WhereAnErrorIsAThrownCall = WhereAnErrorIsAThrownCall.SEND
```

Description rewrite: `SEND | BUILD | INTERPRET`; scoring: `SEND | INTERPRET`; screen: all three; the rest: `SEND`. Both runners read it. The two catch-alls and the boolean flag go.

### What gets simpler and what gets harder

One answer, in one place, to "what happens when this step's code raises". Nothing harder. Whether the default should become "all three" is decision 13 (a broken template a recorded failure, not a crash); the declared shape is useful either way and changes nothing until that decision is made.

### Size, risk

About 40 lines. No request or file change. The reason text must stay `error: <Type>: <message>`, which the shared failure-text helper already standardises. Risk low. Not seen by the equivalence tests: which errors crash. The existing step tests cover thrown and unusable replies; add one test per step for an exception raised in `interpret`.

---

## C. One final corpus record function

### Problem

The final corpus record is built twice: in `layer4_score.run` (`layer4_score.py:194-209`) and in `document_state_final_corpus.final_corpus_record_for_state` (`document_state_final_corpus.py:26-48`), which nothing in the pipeline calls (B11 in the measurement reports lists it as test-only). Two copies of the published schema can drift.

### Proposed shape

One function, `final_corpus_record(doc_id, written_from_variables, description, considerations_spec, content, scores)`, in `document_state_final_corpus.py` (it imports nothing that writes). `layer4_score.run` calls it with the rewrite record's fields; `final_corpus_record_for_state` calls it for the current layout and keeps its own branches for the two old layouts.

### What gets simpler

The published schema is defined once, and the loader test keeps proving the pipeline's records equal the loader-derived ones. This replaces deleting `document_state_final_corpus.py` (B11) with giving it a job.

### Size, risk

About 30 lines. No change: the equivalence test records `final/sdf_corpus.jsonl` and would fail on any difference. Risk very low.

---

## D. `rewrite_description` built from the step

### Problem

`rewrite_description` (`layer1_description_rewrite.py:162-198`), which this project's `run_description_rewrite_on_fresh_descriptions.py` calls, builds its request and routes to OpenAI or Claude by its own code, a second copy of `DescriptionRewriteStep.build_request` and `interpret` (`:327-353`). The equivalence tests cover only the step's path, so a later change to one copy would not show for the script. The screen already does it the right way: `screen_hands` builds through `build_screen_request` and sends through `send_model_request` (`layer1_screen.py:198-230`).

### Proposed shape

A runner-free helper in `step_runner.py`:

```python
def answer_once(step: DocumentStep, document, send=send_model_request) -> StepOutcome:
    request = step.build_request(document)
    return step.interpret(document, send(request))
```

`rewrite_description` builds a small `DescriptionRewriteStep`-compatible document and calls `answer_once` (raising `RewriteAnswerError` for an unusable outcome, as today). After A, the step no longer needs a stage folder or plan list to be built, which is what makes this cheap.

### Size, risk

About 30 lines. No request change (inferred: both paths pass the same model, max tokens 8000, a copy of the schema, the stage tag and item id; the OpenAI path the same effort default and schema name). Add a test that the two requests are equal for both routes before switching. Risk low.

---

## E. A step catalogue

### Problem

What the pipeline knows about a step, other than its request and reply, is written out in several places:

| Fact | Where it is written today |
|---|---|
| Step order | `document_state.PipelineStep` (`document_state.py:95-113`); `seeding._ORDER_OF_STEPS` (`seeding.py:95-96`); `run.py`'s four blocks |
| Which files a step writes, and their id field | `seeding._STAGE_FILES_IN_PIPELINE_ORDER` (`seeding.py:83-94`); each layer's constants; the loader's own reads (`document_state_loader.py:263-300`) |
| Which file a step reads its documents from | `seeding.py:217-224` (the `input_file` table); `run.py:175-194`; the print command's branches (`print_full_sdf_request...py:131-183`) |
| Whether the step is active (enabled, template present) | each layer's `run()` (`layer1_screen.py:528-534`, `layer1_description_rewrite.py:414-420`, `layer1_select_considerations.py:556-563`); again in seeding (`_screen_is_active`, `seeding.py:171-172`; `:313-326`); again in the print command (`:139-141`, `:147-148`) |
| Which prompt files it needs | `seeding._templates_the_step_needs` (`seeding.py:149-169`), an if-chain by step name; only layer 3 checks for itself (`layer3_rewrite.py:52-60`) |
| Which run.py layer runs it | `seeding.SEEDABLE_STEPS` (`seeding.py:58-68`) |
| Its name | three vocabularies: event step names (`consideration_scoring`, `combined_review_and_revise`); print command stage names (`select`, `combined`, `print_full_sdf_request...py:59-60`); stage tags (`layer1_select`). `PipelineStep` has no member for the single-call layer 3 |

The plan promised "Before any call, a step checks that the run's frozen prompt snapshot has the template files and placeholders it needs" (plan, "Update steps"). Only layer 3 does; seeding has its own copy of the knowledge.

### Proposed shape

One module, `sdf_pipeline/pipeline_steps.py`, with one entry per step, read by `run.py`, `seeding.py` and the print command:

```python
@dataclass(frozen=True)
class PipelineStepEntry:
    name: str                                  # the one name: events, --from-step, the print command
    run_py_layer: int                          # which run.py block runs it
    stage_tag: str                             # unchanged (effort map key and shuffle seeds)
    stage_folder_layer: int
    files_written: tuple[str, ...]
    document_id_field: str                     # "prompt_id" or "doc_id"
    reason_not_seedable: str | None
    required_prompt_files: Callable[[dict], list[str]]          # config -> file names
    reason_inactive: Callable[[dict, Path], str | None]         # config, prompts -> None when active
    documents_from_run: Callable[[Path, dict, Path], list] | None   # what the step reads, from files
    build_step: Callable[..., object] | None                    # for the print command

PIPELINE_STEPS: tuple[PipelineStepEntry, ...] = (...)       # in pipeline order
```

Then:

- `seeding.py` replaces its four tables and `_templates_the_step_needs` with lookups, and its input-file table with `documents_from_run`.
- The print command replaces its 60-line if-chain with `entry.build_step(...)` and `entry.documents_from_run(...)`, and takes step names (keep `select` and `combined` as aliases for a while).
- `run.py`'s restart branches (`:176-180`, `:186-187`, `:193-194`) call `documents_from_run` of the first step of the layer. The restart reading of plans (`load_rewritten_plans` then `attach_selections`) is then defined once, used by run.py and the print command.
- Each layer's early return uses `reason_inactive`, so its printed reason and seeding's refusal come from the same function.
- Each step's `required_prompt_files` gives the plan's snapshot check for every step, and seeding uses it.
- `document_state.PipelineStep` can be generated from the catalogue, or checked against it in a test.

This is not the "`run.py` as a step list" that was judged not worth it: `run.py` keeps its four blocks and the layers keep `run()`. It does make that later change small, if it is ever wanted.

### What gets simpler and what gets harder

Adding, renaming or reordering a step touches one entry, not five files; seeding, restart and the print command cannot disagree about where a step's inputs come from; the naming question (layer, stage, step) is settled: step names are the user-facing names, layers are folders and the `--layer` number, stage tags are the cost and effort label. Harder: one more module to read first; the callables make the table less plain than a pure data table.

### Size, files, risk

About 250 lines touched (seeding about 120 out, print command about 60 out, run.py about 15, catalogue about 130 in), net about -100. No request or file change. Seen by the equivalence test: the restarts at `--layer 2`, `3` and `4` (`tests/sdf_prompt_equivalence_scenarios.py:147-162`) and the disabled-stage scenarios. Not seen: seeding's refusal messages (pinned by `test_sdf_seeding_from_a_chosen_step.py`, 53 tests), the print command's output (pinned by its tests against the recordings), and the early-return printed lines (need a test before changing; listed in the cleanup plan's "not seen" list). Depends on cleanup 7: with old layouts gone from the writers, `stage_folder_layer` is one number and `utils.sdf_template_path(prompts_dir, N)` can become a file name in the entry. Pairs with F, which supplies the prompt files.

---

## F. A run prompt snapshot object

### Problem

Every model-calling step loads the run's prompt files in its own constructor: the constitution, principles and preamble in `DraftStep` (`layer2_draft.py:44-59`); the constitution, principles and failure modes once per `_ReviewPromptRenderer`, so twice per layer 3 run (`layer3_rewrite.py:75-90`, built at `:171` and `:227`); the constitution again in `ScoreStep` (`layer4_score.py:55-61`). The variable-line renderer is built in five places with the entity seed default written three ways: `cp.DEFAULT_ENTITY_SEED` (`layer1_description_rewrite.py:309`, `layer1_screen.py:547`, `compose_prompts.py:470`) and a bare `137` (`layer1_plan.py:127`). The "drop the considerations line when the spec is empty" pattern is repeated in draft and layer 3 (`layer2_draft.py:58-59`, `:68-71`; `layer3_rewrite.py:86-88`, `:105-107`).

### Proposed shape

```python
class RunPromptSnapshot:
    """The run's frozen prompts and constitution, loaded once."""
    def __init__(self, prompts_directory: Path, sdf_settings): ...
    constitution_claude: str; principles: str; failure_modes: str; preamble: str
    def template(self, file_name) -> DocumentTemplate      # cached; knows its considerations-free variant
    def variable_line_renderer(self) -> Callable[[dict], str]   # one entity seed, from settings
    def has(self, file_name) -> bool                       # for required_prompt_files checks
```

Steps take `(settings, snapshot)` instead of `(config, prompts_dir)`. `DocumentTemplate.render(values, considerations_spec)` picks the variant.

### What gets simpler and what gets harder

One place reads prompt files; one entity seed default; templates checked for existence in one way. Harder: every step constructor and its tests change signature (about 10 constructors, the step tests' fixtures).

### Size, risk

About 100 lines plus test fixture updates. No request change if rendering is identical, and the prompt-equivalence test sees every rendered prompt, so a mistake fails it. Risk low. Note: layer 3 currently loads the snapshot's files twice, so a snapshot object changes nothing observable (inferred).

---

## G. Corpus steps through one recorder; one deal path

### Problem

- `run_corpus_step` cannot record `Created` (`step_runner.py:250-251` raises for it), so `DealStep` and `RedealStep` are called with `.apply()` directly and their events are written by hand in `layer1_screen.run` (`:564-574`, `:613-621`, `:629-635`, `:644-650`). The `round_dealt` plus `corpus_step_finished` pair is repeated three times.
- `ConsiderationAssignmentStep.apply` returns outcomes but leaves its real results on attributes (`selection_rows`, `flat_scores`, `assigned_labels_by_document_id`, `layer1_select_considerations.py:455-465`) that the layer reads afterwards (`:589-592`).
- There are two deal paths. With the screen on, `DealStep` deals and the round-dealt event records dealt value ids (`layer1_screen.py:564-572`). With the screen off, `compose_records` deals inside the plan layer (`compose_prompts.py:564-565`, called from `layer1_plan.py:123-131`), and no deal event is written, so a new screen-off run gets no dealt value ids in its log (the loader then falls back to the snapshot lookup; inferred from `fold_dealt_value_ids` reading only `round_dealt` events, `document_event_log.py:137-144`). Both paths make the same call: `hand_records(deal_hands(template, values, weights, n_prompts, seed, deal_rules))` (`compose_prompts.py:565`; `layer1_screen.py:350-352`).

### Proposed shape

- `run_corpus_step` accepts `Created` (appends a `created` event with the new record) and takes an optional `summary` for the `corpus_step_finished` event. A small `record_round(event_log, round, prompt_ids, note, value_ids)` helper writes the round pair once.
- `CorpusStep.apply` returns a `CorpusStepResult(outcomes, products)` where `products` carries what the layer writes (selection rows, coverage report), instead of attributes.
- The plan layer, with the screen off and `n_prompts` set, gets its hands from `DealStep` and passes them to `compose_records(hands=...)`, writing the same deal events as the screen path. The full-matrix path (`n_prompts` None) stays as it is.

### What gets simpler and what gets harder

Every corpus step is recorded the same way; the screen driver shrinks by about 30 lines; dealt value ids exist for every new run. Harder: the fold must learn the `created` event (small).

### Size, risk

About 60 lines. Requests, stage files and `prompts.jsonl` unchanged (the same deal call with the same arguments; the hand ids are numbered from 0 in both). The event log changes (new events for screen-off runs, the same events written through a helper); neither equivalence test sees the log, and `test_sdf_document_event_fold.py` pins it. Risk low. Check: the screen-off scenario's `prompts.jsonl` in the equivalence recording must stay byte-identical, which the test does.

---

## H. Layer 3 as a per-document chain of two steps

### Problem

Layer 3 is two runners stitched together by the layer (`layer3_rewrite.py:327-369`):

- the critique runner gets a fake checkpoint (`_CritiqueCheckpointView`, `:311-324`) so a rewritten document counts as done and an accepted critique marks nothing;
- the layer reloads `critiques.jsonl` between the two runs (`:359`) and injects it into `ReviseStep` as `critique_by_document_id`, together with `already_rewritten_document_ids` (`:221-233`), which exists only so the runner keeps already-written rewrite records in its results (the runner returns existing records only for documents the step `applies_to`, `step_runner.py:198-203`);
- the layer's systemic stop wraps `send` to count thrown calls across both runners and switches both runners' own stop off (`:164`, `:218`, `:342-349`, `:367-368`);
- every critique now runs before any revise (writeup deviation; decision 4).

The cleanup in progress replaces `_CritiqueCheckpointView` with runner options, which removes the class but keeps the stitching.

### Proposed shape

`StepRunner` takes a short sequence of `DocumentStep`s and runs them for one document inside one worker task, feeding each accepted update into the next step's document:

```python
runner = StepRunner(steps=[CritiqueStep(...), ReviseStep(...)], ...)
# per document, in one worker task:
#   critique: skip if banked (critiques.jsonl) or the document is finished; else call
#   revise:   document = {**draft, "review": critique}; call
# main thread, input order: write each step's records; mark done when the last step accepts
```

`ReviseStep` reads the critique from its document; `critique_by_document_id`, `already_rewritten_document_ids`, the checkpoint view or option, the send wrapper and the reload go. The systemic stop becomes the runner's ordinary rule counted per document (a document whose chain threw), which is the old layer 3 rule. The single-call path is a chain of one.

Independently of the chain, one small runner change removes `already_rewritten_document_ids`: return existing stage-file records for every input document, not only those the step applies to. For every other step the two sets are the same (inferred: draft, plan, critique, score and the combined step apply to every document that can have a record; the description rewrite and scoring do not use the returned records).

### What gets simpler and what gets harder

Layer 3's `run()` becomes the size of the draft layer's; the "systemic stop across two calls" special case (one of the three stop mechanisms the cleanup plan wants to unify) disappears into the runner's normal rule; the call and printed-line order returns to `origin/main`'s (critique then revise per document), which also recovers the wall-clock overlap the writeup's risk row worries about. Harder: the runner handles a sequence and more than one stage file, about 50 lines more runner code.

### Size, risk, dependencies

About 100 lines changed, about 60 removed from layer 3. No request or file content change (the equivalence test compares requests per stage and item, not order). Risk medium, all in behaviour the tests do not see: printed-line order; crash windows (where the critique is banked: write it under the lock as it arrives, so a crash during the revise keeps it, which is at least as good as `origin/main`, where the main thread wrote it after both calls); the systemic stop's count. Needs a test of the stop with a thrown critique and a thrown revise before changing. **Depends on decision 4.** If the owner accepts all-critiques-first, do only the small `already_rewritten_document_ids` change and let the cleanup's runner options handle the rest.

---

## I. Settings read once into a typed record

### Problem

`config["sdf"].get(...)` with fallbacks appears about 60 times across the layers (`grep` count: 9 in screen, 9 in selector, 8 in description rewrite, 9 in layer 3, ...). Fallback rules are repeated: `description_rewrite_model or plan_model` at `layer1_description_rewrite.py:247` and `:306`; `screen_model or plan_model` at `layer1_screen.py:545` and again in the print command (`:119`); `seed` defaulting to 0 in five places; `workers` read six ways, one with `max(1, int(...))` (`layer1_select_considerations.py:566`) and the rest raw.

### Proposed shape

`SdfSettings.from_config(config)`, a frozen dataclass built once in `run.py` and passed to each layer: resolved model per step (with today's fallbacks, `None` where today passes `None`), seeds, workers, thresholds, revision settings, enabled flags. Steps read fields.

### What gets simpler and what gets harder

Fallbacks defined once; every setting a step uses visible in one type. Harder: callers that build a step from a raw config (tests, the print command, seeding) build settings first.

### Size, risk

About 80 lines plus test fixtures. The equivalence test's configuration scenarios (`real_config_unchanged...`, `minimal_config_with_only_a_plan_model`, the OpenAI and adaptive-thinking one) pin model fallbacks and defaults, because they show in requests. Risk: a default that is `None` today must stay `None` (it lets `call_claude` fall back to the global model), not become the global model, or requests stay the same while the recorded `model` field in `ModelRequest` changes (request hash only). Low value on its own; do it with F if at all.

---

## J. Request hash over the resolved settings

### Problem

`ModelRequest.request_hash` hashes the request before `call_claude` resolves it (`update_steps.py:43-47`). `model` can be `None` (`:33`) and the reasoning effort comes from the config's `effort` map by stage tag, so two events with the same hash can answer requests sent with a different model, effort or token limit.

### Proposed shape

Hash `(asdict(request), asdict(resolve_settings_of_model_request(request)))`. The resolver already exists (`step_runner.py:72-88`) and is pure.

### Size, risk, dependency

About 15 lines. Event log only. Low risk (the fold tests compare hashes from the same code). Worth it only if decision 7 makes the log the source of truth, where "this event answered the request we would send now" is the check that matters.

---

## K. One fold that produces `DocumentState`

### Problem

There are two representations of "what the pipeline knows about a document": the loader's `DocumentState`, built from stage files (`document_state_loader.py`), and the event fold's `FoldedDocumentSteps`, a dict per step (`document_event_log.py:103-134`). The fold test compares them. The plan's "the state is a fold over the events" holds only for dealt value ids. The event's `update` field also means two things: the stage-file record for `DocumentStep`s (`step_runner.py:186`) and the raw `Accepted.update` for batch and corpus steps (`:245`, `:320`).

### Proposed shape

Per step, one reducer `apply(state, update) -> state` (a new description version, a new text version, scores...). The event fold applies events through the reducers; the loader turns each stage-file record into the same update and applies the same reducers. Make every event's `update` the stage-file record.

### What gets simpler and what gets harder

One definition of how each step changes a document; work item 9 then needs only the writers to read the fold. Harder: the loader's per-layout code must be expressed as updates; large.

### Size, risk, dependency

300 lines or more. No request or file change if done as designed. **Do only if decision 7 is yes.** If decision 7 is no, the cleanup plan's note stands: `FoldedDocumentSteps` and the other work-item-9 scaffolding can go.

---

## What I would not do

- **Merge `StepRunner`, `BatchStepRunner` and `run_corpus_step` into one runner.** The shared parts (fan-out, the systemic count, event fields) are already shared or are being made shared by the cleanup. What differs is real: per-document attempts, existing records and writes-as-replies-arrive against per-batch checkpoints, the screen's replace-the-batch writes (`layer1_screen.py:579-586`) and per-batch timing. Treating a document as a batch of one would push the screen's rules onto eight steps, or the other way round. A, B and G make the three consistent without merging them.
- **Steps reading `DocumentState` instead of record dicts, before decision 7.** It was the plan's design, and all layers are now steps, so the writeup's reason for deferring it has lapsed. But the loader returns deal order and the layers pass file order (`document_state_loader.py:306-311`; `layer4_score.py:19-23`), so handing states between layers changes stage-file order for resumed runs (the same issue as decision 5), and about 200 step tests build dicts. It gives one document type and one id field, which E's `document_id_field` gets more cheaply. Revisit with K.
- **Moving `ConsiderationAssignmentStep` into `considerations_assignment.py`.** The import cycle comes from `_selection_rows` and `digital_mind_flags` living in the layer module (`layer1_select_considerations.py:487-509`). Every other step lives in its layer module, so the current placement is the consistent one; G's `CorpusStepResult` is the part worth doing. Fix the docstring only.
- **Renaming the `layerN_*` modules to step names.** Names are settled by E without moving files; `experiments/experiment.py` and one script in this project import the modules.
- **A typed route field on `ModelRequest` on its own.** `openai_schema_name is not None` selects the route (`step_runner.py:39`, `:80`), which is implicit, but only one step uses it; tidy it only if the OpenAI route is folded into `call_claude` (already judged not worth it, because it changes the request).

## Dependencies on the owner's open decisions

| Decision | Proposals affected |
|---|---|
| 4. Layer 3 call order | H depends on it; only H's small `already_rewritten_document_ids` change goes ahead if all-critiques-first is accepted |
| 7. Event log as source of truth | J and K are worth it only if yes; if no, the work-item-9 scaffolding goes, as the cleanup plan says |
| 10. Single-call layer 3 kept | No effect: in H it is a chain of one |
| 13. Broken templates crash or are recorded | B's shape needs no decision; changing its default does |
| 1. Regenerate guard | None of these proposals changes a request or recorded output file, so none needs a regeneration |

## Suggested order

1. A, B and C together, alongside the in-progress runner tidy (they touch the same hooks).
2. D.
3. Cleanup 7 (current layout only), then E with F.
4. G.
5. H after decision 4; I only together with F; J and K after decision 7.

For every proposal: both equivalence tests pass without regenerating, the full suite passes, and the real-run replay shows 0 differences. The behaviours neither test sees (printed lines, stop rules and their messages, resume in the middle of a step, early returns) need a named test before the change, as the cleanup plan already requires.
