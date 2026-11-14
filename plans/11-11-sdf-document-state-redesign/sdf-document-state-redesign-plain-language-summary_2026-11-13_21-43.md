# SDF pipeline refactor: plain-language summary

## What was done and how it was checked

The code that generates the SDF dataset was reorganised so that each document has one record following it through the pipeline, and every stage (plan, description rewrite, draft, critique, revise, score and the rest) is the same kind of step, run by one shared runner. The aim was to change how the code is organised, not what it produces.

The work is 16 draft pull requests (proposed code changes, reviewed on GitHub before merging), #194 to #209, each built on the one before, plus six follow-ups (#210 to #215). Nothing is merged yet.

Every pull request was checked to send exactly the same requests to the models as before:

- An offline test runs a small example through the old and new code with canned model replies and compares every request, every intermediate file and the final corpus.
- A second offline test does the same for the audit judges.
- A replay rebuilt 355 requests from 7 stored real runs and found no differences.
- The automated review bot approved #194 to #209. #210 to #215 have not been reported as reviewed.

None of these checks calls a model, so they cost nothing to rerun.

## The problem

Each stage kept its own files, its own list of finished documents and its own way of picking up after an interruption, so a document's history was spread across several files. Rerunning one part was slow and risky: rerunning only the critique on chosen drafts needed a hand-written script that wrote stage files directly, where a small mistake could quietly change what later stages saw. It was also hard to see what a stage would send to a model, because some settings (reasoning effort, token headroom) were filled in only at the moment of sending.

## What changed

- **One record per document**: its dealt values, plan, description versions, assigned considerations, text versions, scores and outcome, read by every tool the same way.
- **One kind of step**: a step decides what to ask the model and what the answer means; the runner does the rest (calling, saving, retrying and resuming, that is, picking up an interrupted run where it stopped).
- **Rerun from any step**: a new run can start from a stored one at the plan, description rewrite, consideration scoring, draft, critique, revise or score step, on chosen documents.
- **See a request without sending it**: a command prints the exact request a stage would send for a given document, without calling a model.
- **A per-document log** of what happened is written for checking only; the pipeline never reads it. Every file the pipeline wrote before is still written in the same place and format.

The follow-ups:

- **#210** removes code the refactor made redundant, tidies the runner, removes 31 old tests already covered by newer ones, and adds a test that runs started before the refactor resume identically from 18 interruption points; that test is marked as temporary, to delete once no such run still needs resuming.
- **#211** replaces an automatic guard on re-recording the test recordings with a rule: re-record only in a pull request meant to change requests, and say why.
- **#212** fixes starting a new run from a stored one: a bare run name is looked up as a stored run first and an ambiguous name is refused; the refusal to start at revise under the single-call configuration (critique and revise in one call) says why; and starting at consideration scoring prints a notice that the assignment will be solved again over the selected documents, and records this in the run's settings.
- **#213** makes the runner return existing records for every document on resume, removing a workaround in critique and revise.
- **#214** checks offline that the other pipeline (DAD) still sends unchanged requests. It does.
- **#215** makes a broken prompt template fail only the document it affects instead of crashing the run, with one declared rule for where an error counts as a failed call, and has one function build the final corpus record. It re-records the test recordings for one intended change: the message that stops a run when every call fails now names an error building the prompt as a possible cause. No request or output file changes.

## Benefits

- Rerunning only the critique on chosen drafts is one command, not a hand-written script, and it sends the same requests the old script did (tested).
- A resumed run keeps the same documents as an uninterrupted one, because fewer places can disagree about what is done (which near-duplicate twin is kept is fixed separately, below).
- A broken prompt template fails only the document it affects, with the error recorded and retried on resume (#215). Before, it crashed the run mid-stage, and replies already paid for were lost and paid for again.
- Future datasets will record a stable id for each dealt value, so a value can be matched across runs even if its wording later changes.
- Less code to maintain, and changes to one stage are easier to review: a step can be read on its own, and the print command shows its requests.

## Next steps

All decisions are made; this is the working order.

1. Review and merge the pull requests in order, lowest first.
2. Add the dealt value ids to the stage files and final corpus, for new runs only. Old runs and the published dataset are not changed.
3. Further tidying that changes no request or file: one way of recording the whole-corpus steps and one path for dealing; leaving all file writing to the runner; one list of steps shared by the run command, rerun starts and the print command; and building the description rewrite request from its step. Two ideas that would tidy without shortening the code are put off, and three are dropped.
4. Sort documents by deal order before removing near-duplicates, so a resumed run keeps the same one of each near-duplicate pair as an uninterrupted run. This deliberately changes the final corpus of resumed runs, so it is its own pull request and re-records the test recordings.
5. Retire the per-stage progress files (checkpoints): a document counts as done when its record is in its stage file. This also re-records the test recordings.

On hold: retiring the single-call configuration. The pull request can be written and reviewed, with the hold stated at the top, but must not merge until William gives the go-ahead; he is waiting to hear whether someone else is fine with it. Its only users are two finished experiments (their reports stay) and an example template.

Not planned: moving the website and the viewer onto the new per-document record (they keep reading the stage files), and changing DAD until that is decided separately.

## Costs and risks

- **Review effort**: 16 pull requests, each built on the one before, and any change to a lower one means updating those above it.
- **Behaviour the request checks do not see**: printed messages, how a run stops when many calls fail, early exits when a stage is switched off, resuming the implausibility screen partway through a round, and resuming a run with no log. Cleanups touching these need their own test first; several have one.
- **Deliberate output changes**: the near-duplicate ordering, dealt value ids, retiring checkpoints and retiring the single-call configuration each change outputs or recordings on purpose. Each is its own pull request that says why, so none can pass for an accident.
- **Repository size**: the resume test adds about 3.4 MB of recordings (0.77 MB of zipped run folders and 2.66 MB of JSON), accepted because it protects runs started before the refactor. The test is temporary.
- **Limits of the checks**: they compare against stored recordings, so they prove sameness only for the situations recorded, not for every possible run.
- **Cost of not doing it**: each future change to a stage, and each partial rerun, keeps needing its own scripts and checks.

## Files and identifiers

All in `plans/11-11-sdf-document-state-redesign/` on branch `william/11-11-sdf-real-run-request-replay`:

- Original plan: `sdf-document-state-and-update-steps-plan_2026-11-11_21-24.md`
- What was built (#194 to #209): `sdf-document-state-redesign-draft-pull-request-stack-writeup_2026-11-13_18-30.md`
- Cleanup plan: `sdf-document-state-redesign-cleanup-plan_2026-11-13_19-39.md`
- Every decision and its to-do: `sdf-document-state-redesign-decisions-for-william_2026-11-13_20-10.md`
- Design improvements: `sdf-pipeline-design-improvements-after-refactor-investigation_2026-11-13_20-10.md`

Pipeline repository: `sentfutures/animal-welfare-data-pipeline`, draft pull requests #194 to #215. The single-call configuration is `sdf.revision.split: false`; the run's settings file is `run_manifest.json` (field `seeded_from`).
