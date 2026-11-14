# SDF pipeline refactor: plain-language summary

## What was done and how it was checked

The code that generates the SDF dataset was reorganised so that every document is followed through one consistent record, and every stage (plan, description rewrite, draft, critique, revise, score and the rest) is written as the same kind of step, run by one shared runner. The aim was to change how the code is organised and nothing about what it produces. The work is a stack of 16 draft pull requests (a pull request is a proposed code change, reviewed on GitHub before it is merged), plus five follow-ups (#210 to #214) and two pieces still in progress. Nothing is merged yet.

How we know nothing changed: every pull request was checked to send exactly the same requests to the models as before.

- An offline test replays canned model replies through the old and new code and compares every request, every intermediate file and the final corpus.
- A second offline test does the same for the audit judges.
- A replay rebuilt 355 requests from 7 stored real runs and found no differences.
- Every draft was approved by the automated review bot.

None of these checks calls a model, so none costs anything to rerun.

## What the pipeline was like before, and why that was a problem

Each stage kept its own files, its own bookkeeping of which documents were done, and its own logic for picking up after an interruption. A document's journey was never held in one place, so answering "where is this document up to, and why was it removed?" meant reading several files. Changing or rerunning one part was slow and risky. For example, rerunning only the critique on a few chosen drafts needed a hand-written script that wrote stage files directly, and a small mistake there could quietly change what later stages saw. What a stage would actually send to a model was also hard to see, because some settings (reasoning effort, token headroom) were filled in at the last moment, inside the sending code.

## What the refactor did

- One picture of each document: its dealt values, plan, description versions, assigned considerations, text versions, scores and outcome, readable by every tool through one loader.
- Every stage is the same kind of step. A step decides what to ask the model and what the answer means; the runner does the rest (calling, recording, retrying, resuming).
- Rerunning from any step on chosen documents is a built-in option: start a new run from a stored one, from the plan, description rewrite, consideration scoring, draft, critique, revise or score step.
- A read-only command shows the exact request a stage would send for a given document, without calling a model.
- A day-by-day log of what happened to each document is written alongside the existing files. It is only written for checking; the pipeline itself never reads it. Every file the pipeline wrote before is still written in the same place and format.

The cleanup pull requests since then remove code the refactor made redundant, tidy the runner, write only the current run layout, and add a test that runs started before the refactor resume identically at 18 interruption points (31 redundant old tests were removed). Other follow-ups regenerate the saved recordings deliberately when a pull request is meant to change requests (rather than being blocked by an automatic guard), fix seeding problems, make the runner return existing records for every document on resume, and check offline that the other pipeline (DAD) sends identical requests.

## Benefits

- Rerunning only the critique step on chosen drafts is one command, not a hand-written script, and it sends the same requests the old script did (tested).
- A resumed run keeps the same documents as an uninterrupted one; fewer places can disagree about what is done. (Which of two near-duplicates is kept after a resume is a separate fix, listed below.)
- A broken prompt template will fail only the document it affects, with the error recorded and retried on resume. Today it crashes the run mid-step, after sibling calls were already sent and paid for, and their replies are lost and paid for again.
- Future datasets will record a stable id for each dealt value, so a value can be matched across runs even if its wording is later changed.
- Less code to maintain: dead code, copied helpers and old tests are being removed, and the step table now documents what a document goes through.
- Changes to one stage are easier to review, because a step can be read on its own and the request print command shows the result.

## What happens next, if it is pursued

Every decision has been settled, so the order below is working order, not a list of open questions.

1. Review and merge the stack in order, lowest first. Merging was deliberately left until the whole stack could be reviewed.
2. Finish the two pieces in progress: one exception policy (the broken-template change above), and one function that builds the final corpus record.
3. Add the dealt value ids to the stage files and final corpus for new runs only. Old runs and the published dataset are not changed.
4. Further tidying that changes no request or file: one recorder for the corpus-level steps and one deal path; steps that write nothing themselves (only the runner writes); one catalogue of steps shared by the run command, seeding and the print command; building the description rewrite request from its step. Two ideas are deferred as tidier but not smaller, and three are dropped.
5. Sort documents by deal order before the near-duplicate cull, so a resumed run ships the same twin of each near-duplicate pair as an uninterrupted one. This is its own pull request, because it deliberately changes the final corpus of resumed runs and regenerates the recordings.
6. Retire the per-layer progress marker files (checkpoints): a document is done when its record is in its stage file. This also regenerates the recordings.
7. Smaller items: seeding at consideration scoring prints a notice that the assignment will be solved again over the selected documents; two seeding messages and checks are fixed; a note is added to the resume test saying it can be deleted once no run started before the refactor still needs resuming.

Held: retiring the single-call way of running layer 3 (critique and revise in one call). It can be written and reviewed, with the hold stated at the top of its description, but it must not merge until William gives the go-ahead; he has asked someone else whether they are fine with it and is waiting to hear back. Its only users are two finished experiments (their reports stay) and an example template.

Not planned: changing the website or the viewer to read through the new loader (they keep reading stage files), and changing DAD, the other pipeline, until decided separately.

## Costs and risks

- Review effort: the stack is long (16 pull requests, each based on the one before), and every lower change forces the ones above it to be updated.
- Behaviour the equivalence tests do not see: printed messages, how a run stops when many calls fail, early exits when a stage is switched off, resuming the screen mid-round, and resuming a run with no log. Cleanups touching these need their own test first, and several have one.
- Some planned changes deliberately alter outputs or recordings (deal order for the cull, dealt value ids, retiring checkpoints, the single-call retirement). Each is its own pull request that says why requests or files change, and can no longer be mistaken for an accidental difference.
- The new resume test adds about 3.4 MB of recordings to the repository, accepted because it protects runs started before the refactor; it is transitional.
- The tests compare against stored recordings, so they can only prove sameness for the situations recorded, not for every possible run.
- The cost of not doing it is that each future change to a stage, and each partial rerun, keeps needing bespoke scripts and checks.

## Files and identifiers

All in `plans/11-11-sdf-document-state-redesign/` on branch `william/11-11-sdf-real-run-request-replay`:

- Original plan: `sdf-document-state-and-update-steps-plan_2026-11-11_21-24.md`
- What was built (pull requests #194 to #209): `sdf-document-state-redesign-draft-pull-request-stack-writeup_2026-11-13_18-30.md`
- Cleanup plan: `sdf-document-state-redesign-cleanup-plan_2026-11-13_19-39.md`
- Every decision and its to-do: `sdf-document-state-redesign-decisions-for-william_2026-11-13_20-10.md`
- Design improvements: `sdf-pipeline-design-improvements-after-refactor-investigation_2026-11-13_20-10.md`
- Pipeline repository: `sentfutures/animal-welfare-data-pipeline`; draft pull requests #194 to #214
