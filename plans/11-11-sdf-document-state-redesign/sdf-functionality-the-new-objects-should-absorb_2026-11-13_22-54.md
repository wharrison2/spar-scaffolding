# SDF pipeline: functionality the new objects should absorb

Read-only investigation of branch `william/sdf-one-exception-policy-and-one-final-corpus-record` (worktree `animal-welfare-data-pipeline/.claude/worktrees/sdf-one-exception-policy-and-one-final-corpus-record`), read against the decisions file and the design improvements report (A to K) on `origin/main`. All line numbers are from that worktree. "(est.)" marks a number I reasoned to from the lines I read, not measured by making the change. Nothing was edited or run.

## Already landed on this branch (do not count again)

- B (one exception policy): commit `f16cb36`; `StepPolicy.errors_counted_as_thrown_calls` is in `update_steps.py:164`.
- C (one final corpus record): commit `670ade0`; `final_corpus_record` is called from `layer4_score.py:198` and `document_state_final_corpus.py:59`.
- H's small part (runner returns existing records of every input document): commit `bfd3a18`.

## Ranking (lines saved per effort)

Net = lines removed minus lines added. Totals are at the end.

| Rank | Finding | Owner | Net lines (est.) | Effort |
|---|---|---|---|---|
| 1 | Step boilerplate every `DocumentStep` repeats | `DocumentStep` base (existing) | -50 | low |
| 2 | Run-folder-or-corpus input resolver copied across evals | final corpus module (existing) | -40 | low |
| 3 | Failure text in progress lines | `Failed` / `DocumentStep` base | -12 | low |
| 4 | Per-stage evals readers and the three per-value tally loops | `DocumentState` loader (existing) | -38 | medium |
| 5 | Catalogue and batch object, measured per consumer | catalogue plus batch object (planned) | -70 | high |
| 6 | Layer 3 glue once decisions 4 and 7 and the batch object exist | batch object | -19 | follows 5 |

---

## 1. Step boilerplate (new, not in A to K)

What: five things every step class spells out by hand.

- `document_id_of`: 9 definitions of three lines each, all `return document["prompt_id"]` or `["doc_id"]` (`layer1_plan.py:54`, `layer2_draft.py:62`, `layer3_rewrite.py:~154,~207,~257`, `layer4_score.py:65`, `layer1_description_rewrite.py:321`, `layer1_select_considerations.py:~308`). The step already declares `stage_file_id_field` (`update_steps.py:180`) for the same fact on the stage file side.
- `applies_to` returning `True`: 4 definitions of three lines (plan, critique, combined, score).
- `systemic_stop_message`: 7 hand-written four-line strings (`layer1_plan.py:46`, `layer2_draft.py:40`, `layer4_score.py:52`, `layer1_description_rewrite.py:296`, `layer1_select_considerations.py:276`, `layer1_screen.py:424`, `layer3_rewrite.py:47`). They differ only in the tag, the word for the call ("API call", "call", "scoring call", "screen call") and the unit ("document", "plan", "batch").
- `ModelRequest(...)` construction: 11 sites, each repeating `system_prompt=system or ""`, `model=`, `stage=`, `item_id=` (plus `cache_system=True` in five). The stage tag is a string literal in five modules (`"layer1_plan"`, `"layer2_draft"`, `"layer4_score"`, `"layer3_*"`) and a constant in three others, and is repeated again inside the stop message.
- `applies_to` and `document_id_of` are abstract only because nobody gave them a default.

Owner: `DocumentStep` in `update_steps.py`.

Sketch:

```python
class DocumentStep(ABC):
    document_id_field = "doc_id"          # the id field of the documents the step is given
    stage_file_id_field = None            # defaults to document_id_field; the draft step sets "doc_id"
    stage_tag: str                        # one declaration; the message and every request read it
    def document_id_of(self, document): return document[self.document_id_field]
    def applies_to(self, document): return True
    def request(self, document, user, system, **settings) -> ModelRequest:
        return ModelRequest(user_message=user, system_prompt=system or "", item_id=self.document_id_of(document),
                            stage=self.stage_tag, model=self.model, **settings)
    @property
    def systemic_stop_message(self): ...   # one template; the three differing words are class attributes
```

Lines: removed about 20 (`document_id_of`) + 10 (`applies_to`) + 27 (messages) + 27 (request fields) = about 85; added about 35 (base defaults 6, request helper 12, message template 8, the few attributes that differ 9). Net about -50 (est.).

Legibility: a step class then contains only what is specific to the step: what to ask and how to read the answer.

Changes: no request, stage file or printed line if the three differing words are kept as attributes. The exit message changes if the tag in it changes; decision 17 renames five tags, and the last commit on this branch (`903ebc2`) shows the stop message is in the equivalence recordings, so that rename regenerates them regardless (decision 1).

Caution for decision 17 and the catalogue: the stage tag is also a shuffle seed (`layer1_select_considerations.py:145` seeds the offered order with `STAGE_TAG`; `layer1_screen.py:242` and `:297` do the same). If the catalogue derives the tag from the new full name, `layer1_select` becoming `layer3_consideration_scoring` would change every scoring request's offered order. The seed literal must stay the old tag, which the design improvements report notes for the effort map but not for the seeds.

Dependencies: none. Steps 17 and the catalogue make it a little neater (the tag comes from the entry), but it can go first with `stage_tag` as a class attribute. Refines A (A changes the progress and outcome hooks; this is the part of the step class A does not touch) and I (the model attribute).

## 2. One resolver for "run folder or corpus file" (new)

What: five copies of "an `--input` is a run folder or a corpus jsonl; find the corpus, the run folder and the audit folder".

- `evals/audit_sdf.py:66-81` `resolve_input` (16 lines) and `:84-90` `run_directory_for_input` (7 lines)
- `evals/model_judged_audit_support.py:32-38` `corpus_path_of` and `:41-49` `run_directory_of` (19 lines)
- `evals/log_probability_persona_check/build_scoring_inputs.py:62-75` `resolve_input` (11 lines)
- `evals/shoehorning_audit_postprocess_sdf.py:203` (its own `parent.name != "final"` check, 3 lines)
- `evals/diversity.py:112-123` (12 lines; also handles DAD, so leave it)

Owner: `document_state_final_corpus.py` (the module that already owns the shape of `final/sdf_corpus.jsonl`), or the run folder object if one is made.

Sketch: `resolve_sdf_input(argument) -> ResolvedSdfInput(corpus_path, run_directory_or_none, audit_directory)` (about 18 lines); the three evals callers keep a two or three line wrapper.

Lines: removed about 55 (the first four), added about 18 + 6 wrappers. Net about -40 (est.).

Legibility: one definition of what `final/` and `audit/` are relative to.

Changes: no request or stage file. The three "not found" messages differ today; pick one wording (a printed exit message of evals, not of the pipeline).

Dependencies: none.

## 3. Failure text in progress lines (new, small)

What: the same thrown-versus-unusable `if` appears in `layer1_plan.py:100-102`, `layer2_draft.py:120-122`, `layer3_rewrite.py:~137-139` and `:~205`, `layer1_description_rewrite.py:370-375`, `layer1_select_considerations.py:~358-362` (`_problem_text`), `layer1_screen.py:~493-498` (`failed_batch_line`) and `layer4_score.py:106-108`; about 32 lines in all.

Owner: a helper on `Failed` or `DocumentStep` that takes the wording of the thrown case as an argument.

Lines: removed about 32, added about 20. Net about -12 (est.).

Changes: wording differs today (plan and draft keep the `error: ` prefix inside the parentheses, the others strip it; layer 3 says "no rewrite"). Parameterising keeps every line; unifying changes the printed lines of plan, draft and layer 3, which step tests assert. Rank low: A's runner-owned progress may make it moot.

Dependencies: none.

## 4. Per-stage evals readers and the three tally loops (refines the "evals" part of K; new measurement)

What: three readers that rebuild from stage files what `DocumentState` already holds.

- `layer1_screen.load_screen_outputs` (`:~695-727`, 28 lines) reads hands, rounds, batches, checkpoint, rejections and the summary. The loader already has, per state, `screen_judgment`, `deal_round_number`, `removal` and `outcome`.
- `layer1_description_rewrite.load_description_rewrite_outputs` (`:436-442`) plus `plan_records_of_states` (`document_state_loader.py:136-143`) plus the two `DocumentState` fields `has_plan_record` and `plan_record_variables` (`document_state.py:353-354`, loader `:506`) that exist only so `rewrite_share_rows` can be handed plan-shaped dicts. The state already has `description_rewrite_decision` and `variables`.
- `layer1_select_considerations.load_considerations_outputs` (`:573-589`, 17 lines): `rows` is never read by a caller (`audit_sdf.py:270-292` and `report_sdf.py:398-420` use only `catalog` and `report`).
- The three per-(axis, value) counting loops in `removal_rate_rows`, `share_shift_rows` and `rewrite_share_rows` (about 8 + 5 + 8 lines) have one shape.

Owner: the loader's states. Row functions take `states`; one `tally_per_variable_value(states, *predicates)` helper (about 8 lines) replaces the three loops.

Lines: removed about 28 + 7 + 17 + 8 + 8 (the two fields, loader line and comments) + 21 (loops) = about 89; added about 15 (state comprehensions inside the row functions) + 8 (tally) + about 25 (the three row functions' signatures and the report and audit call sites change). Net about -38 (est.).

Legibility: `evals/report_sdf.py` and `evals/audit_sdf.py` read one list of states instead of three differently shaped dicts.

Changes: no request or pipeline stage file. The report and audit text "N of M batches complete" (`audit_sdf.py:~213`, `report_sdf.py:~344`, and the `batches_*` keys of `report["screen"]`) is built from the screen checkpoint; decision 7 retires that checkpoint, so the line is dropped or recomputed from `screen_judgments.jsonl`. `report["screen"]` field names should stay (tests at `tests/test_layer1_screen.py:503-553` pin them).

Dependencies: decision 7 for the batches line. Medium effort because the screen's "accepted ids are None until the screen finishes" distinction has no home on a state and needs a run-level flag.

## 5. Catalogue and batch object, measured per consumer (refines E and the planned objects)

The design improvements report estimates E at about 250 lines touched, net about -100. Per consumer, from the lines I read:

| Consumer | What goes | Lines removed (est.) |
|---|---|---|
| `seeding.py` | `SEEDABLE_STEPS` and `UNSEEDABLE_STEPS` (`:52-80`, 29 lines), `_STAGE_FILES_IN_PIPELINE_ORDER` and `_ORDER_OF_STEPS` (`:82-97`, 16), `_templates_the_step_needs` (`:161-180`, 20), the `input_file` table (`:232-237`, 6), `_screen_is_active` (`:183-184`, 2), about half of `_required_records_for_step` (`:312-357`, 26 of 46) | about 99 |
| the print command | the `if stage ==` chain (`:134-191`, 58 lines) plus `_record_with_id` (`:92-96`) become `entry.build_step(...)` and `batch.document(entry, document_id)` | about 46 net of the 12 that stay |
| `run.py` | folder setup and restart reads (`:161-165`, `:185-189`, `:195-196`, `:202-203`, 14 lines); the four `if start_layer <= N` blocks (`:172-207`, 36 lines) become a loop over the catalogue's layers (about 15) | about 35 |
| the layers' reload and apply code | `load_rewritten_plans` (`layer1_description_rewrite.py:254-264`, 11), `attach_selections` (`layer1_select_considerations.py:238-250`, 13), the skeletons of `apply_results` (`:205-230`) and `_attach` (`:214-235`), and the two `return apply_results(...)` and `return _attach(...)` lines: the swap of description and the assembly of the considerations spec stay, applied when the step accepts into the batch's document | about 36 net of about 12 that move |
| the layers' guards | the "disabled or template missing" early returns written four times per optional step: description rewrite (`:411-417`, `:259-260`, seeding `:335-336`, print `:139-140`), selector (`:258-265`, `:243-246`, seeding `:343-345`, print `:147-148`), screen (`:131-137`, seeding `:183-184`, print `:103-105`) | about 15 net (the guards become one `reason_inactive` each; the printed text must be kept word for word) |

Removed about 231; added about 80 for 13 catalogue entries and the entry class, plus about 40 for the batch object (load at run start, `document(entry, id)`, apply an accepted update). Net about -70 (est.), a little under E's -100.

Not a saving, only consolidation: the model-setting fallbacks. There are 10 model lookups, each one line (`layer2_draft.py:46`, `layer4_score.py:58`, `layer1_screen.py:547`, ...). The three fallback pairs (`description_rewrite_model or plan_model` at `:244` and `:305`; `screen_model or plan_model` at `layer1_screen.py:547` and the print command `:119`; `select_model or plan_model`) would stay one line each in a `model_setting` field of the entry. That removes a drift risk, not lines. Likewise stage file names: about 25 literal spellings of nine names across seeding, the print command and the loader (`document_state_loader.py:273-296`), which the entry's `files` would own; I count the seeding and print spellings above and leave the loader's out because it also reads older layouts' names.

Changes: no request, stage file or recorded message if the guard texts and seeding refusals are preserved; `tests/test_sdf_seeding_from_a_chosen_step.py` pins the refusals and the print command tests pin its output. Early-return printed lines need a test before they move (the cleanup plan already says this).

Dependencies: the catalogue needs decision 17 (names) and is cleaner after decision 7 (no checkpoint files in `files`). The batch object needs decision 7 (the loader and the layers must agree on "done"), and decision 5's deal order for any layer that now reads batch order instead of file order. Decision 18 does not matter for this item.

## 6. Layer 3 glue (refines H after decision 4 dropped it)

With every critique before every revise kept (decision 4), `layer3_rewrite.run` (`:311-347`, 37 lines) still stitches two runners: checkpoint, `pending_ids`, `marks_nothing_done` and `also_finished_document_ids` (`:313-322`, `:331-332`), the reload of `critiques.jsonl` into `ReviseStep.critique_by_document_id` (`:336-337`), and the joint thrown-call count (`:325-345`). With the batch object holding each document's critique, and decision 7 retiring the checkpoint, `ReviseStep` reads `document.critique`, the `StepRunner` options `uses_checkpoint`, `marks_nothing_done` and `also_finished_document_ids` (`step_runner.py:123-124`, `:135-150`, 10 lines) have no user left, and `run` becomes about 18 lines.

Lines: removed about 19 in layer 3 and about 12 in the runner; most of the runner part belongs to decision 7 and is not counted here. Net about -19 (est.).

Dependencies: batch object (5) and decision 7.

---

## Total

About -50 -40 -12 -38 -70 -19 = about -229 lines (range about -170 to -290, since each figure is an estimate from the lines read). In relation to `sdf_pipeline/` (about 6,400 lines without tests) that is about 3.5%. Items 1 to 4 total about -140 and need no catalogue, batch object or decision (4 needs decision 7 only for one report line).

## Considered and left out (absorbing it would mostly move code)

- **F, the run prompt snapshot object.** Measured net is about -10, not the about 100 lines touched. Constitution, principles and failure modes load in `layer2_draft.py:46-52` (7 lines), `layer3_rewrite.py:80-85` (6, executed twice because the critique and revise renderers each build one) and `layer4_score.py:59-60` (2); the variable-line renderer is built in five places (3 lines each). A snapshot class of about 25 lines replaces roughly 35. Worth it only for loading layer 3's files once; no request changes.
- **I, the settings record.** The `config["sdf"].get(...)` count in `sdf_pipeline/` is 28, mostly one-liners; the typed record saves nothing and adds a class.
- **G, corpus steps through one recorder.** Under decision 18 the event half disappears (about 25 lines of `append_round_dealt_event` and `append_corpus_step_finished_event` in `layer1_screen.py:159-177`, `:221-253`). What remains is three "write a round" blocks (hands save, round record, rounds file append) that a `record_round` helper shortens by about 8 lines.
- **D.** Agree with the report: `rewrite_description` (`layer1_description_rewrite.py:159-195`, 37 lines) becomes about 12 through `answer_once`, so about -25 to -30. Not re-ranked.
- **Corpus step outcome helpers** (`isinstance(outcome, Accepted)` filters at `layer4_score.py:176`, `:187`, `:191`): a result object costs more than the three comprehensions.
- **Gate rule in the loader** (`document_state_loader.py:229-260` repeats `alignment >= threshold and realism >= threshold`, which `ScoreGateStep.apply` has at `layer4_score.py:126`): one shared function saves about 4 lines; the two vocabularies of removal codes (`"below_score_threshold"` against `BELOW_THRESHOLD_AT_GATE`) are naming drift, not lines.
- **A `RunFolder` paths object** (`run.py:161-165`, `experiments/experiment.py:185-188`, print command `:133`, seeding `:227`): about 12 lines of path building, replaced by a class of about 12.
- **Records keyed by id** (7 one-line dict comprehensions over `utils.load_jsonl`), **the deal inputs** (template, variables, deal rules loaded in both `layer1_plan.py:118-121` and `layer1_screen.py:149-152`, `:165-166`: net about -2), **the consideration coverage report** (`considerations_assignment.coverage_report` and `considerations_coverage.corpus_coverage` share a loop shape but measure different things).

## Which of A to K these refine

- A: item 1 is the part of the step class A does not touch; item 3 may be absorbed by A's runner-owned progress.
- E: item 5, with measured consumer numbers (net about -70 with the batch object, a little under -100); item 6 is what remains of H.
- F and I: measured net about -10 and about 0; left out.
- G: the event half goes with decision 18; about -8 remains.
- D: agree.
- B, C and H's small part: already landed on this branch.
- K and the evals part of "steps read `DocumentState`": item 4 is the narrow, measurable piece of K that does not wait for the fold.
