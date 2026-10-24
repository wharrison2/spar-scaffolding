# Plan: moving the log-probability persona check into the pipeline repository

## What the check is

The log-probability persona check estimates whether a corpus would push a model trained on it towards a particular persona. It is a general process, not a single test. Preachiness was the first persona it was used for; others (sycophancy, cruelty, indifference to welfare, and so on) should need only new input files.

For each document, an open-weights model scores every token of the document as the assistant's reply. It does this under several system prompts:

- **no system prompt and a generic assistant prompt**, which give the baseline;
- **about 30 rewordings of the target persona**;
- **topic-control personas**, which share the subject matter but take no stance;
- **optionally, a same-values control**, which shares the persona's values but not its behaviour.

The score is how much more likely the document's text becomes under the target persona than under the controls. Text the target persona makes unusually likely is text that would pull a model trained on it towards that persona.

The first full run covered the 477-document corpus with Gemma 4 12B on one rented GPU. It cost under $12 in total.

## Goal

The aim is to make the check a supported evaluation in the pipeline repository. That covers:
- a command that runs it against any pipeline run folder for any persona set;
- a project skill that teaches a contributor's agent how to design, validate, run and read a new persona check;
- the pod tooling it needs, with the safeguards that the research version already has.

## Problems as they stand, and what to do about them

### Is the measurement valid?

| Problem | Why it matters | What to do |
|---|---|---|
| The whole-document score did not track the persona. Preachier rewrites of documents moved the score the right way too rarely to pass the pre-registered bar. | Persona-like passages are a small share of most documents, and the rest of the text drowns them out. A corpus-level verdict built on document means would be unreliable. | Make span-level output primary: the highest- and lowest-scoring windows of each document, and scores for changed text against unchanged text. Keep document means as a secondary output labelled as not validated. |
| The changed-span score passed, but nothing has ruled out that any moral-register edit scores as the target persona. | Without that, a high span score may only mean "talks about welfare in a moral tone", not "lectures". | Make a neutral control a required step: edits of the same size that change register without adding the behaviour. They must score near zero at span level. Run it for preachiness before anything is built on the preachiness results. |
| The "away from the persona" arm of the positive control tested nothing. The source documents were already plain, so rewriting them to be plainer barely changed them. | A control that cannot move is uninformative, and can be mistaken for a null result. | Choose sources for the away arm that already show the behaviour, or drop the arm. The skill should say so. |
| No one has read the extreme spans. | Span output is the part the evidence supports, but its meaning has only been checked on synthetic edits. | Read a sample of the extreme spans for preachiness before relying on them, and make reading a sample a standard reporting step. |
| Significance is tested per document, on the whole-document score. | The test is attached to the output that failed validation. | Add a test on span scores (for example a permutation test over windows). Keep the per-document test only beside the document means. |
| The rewordings of the target persona agree with each other slightly less than the control rewordings do. | Low agreement means the score depends on wording as well as persona. | Report rewording agreement in every run, and have the skill tell designers to revise the rewordings when the target agrees less than the controls. |
| One scoring model has been used. | The check assumes the scoring model's sense of a persona resembles that of the model being trained. | Make the model configurable, require a pinned revision, and record it. Treat cross-model agreement as a later validation step, not a requirement for porting. |

### Is it general?

| Problem | What to do |
|---|---|
| The persona names, the control groups and the rewrite directions are written into the code. | Each check gets its own persona-set folder: target rewordings, topic controls, an optional same-values control, positive- and neutral-control rewrite instructions, and a configuration file with the persona's name, expected counts and pass criteria. Output columns take the persona's name from the configuration. |
| The background facts in the scoring prompt, and the copy of the generation prompt, describe one version of the variables matrix. | Build them from the repository's own prompt and variables files at run time, so they follow changes to the matrix. |
| Pass criteria live in a plan document. | Pass criteria go in the configuration file, are copied into the run's configuration record before scoring, and the full scoring run refuses to start without them. |
| Nothing helps with designing a new persona. | Add a validator for persona-set folders that checks counts, blank lines and duplicates, and a free pilot that reports rewording agreement. The skill gives the design procedure step by step (see below). |

### Engineering

| Problem | What to do |
|---|---|
| Periodic pulls from the pod fail with "file changed as we read it". The final copy is unaffected. | The scorer writes each result under a temporary name and renames it when complete, and the launcher pulls only completed files. |
| One run loaded the model and scored markedly slower than the others, for reasons not diagnosed. | Record the pod type, the load time and throughput in every run's configuration record. Have the cost estimate state a range, not a single figure. |
| The rented-GPU provider has no way to assign a pod to a cost center through its interface. | Keep this as a manual, optional step in the skill. Do not block a launch on it. |
| Per-document score files are about half a megabyte each, hundreds of megabytes per corpus. | Git-ignore them. Keep only summary tables, extreme-span files and run configuration records in the run's audit folder. |
| The tests build a tiny model with torch. The repository's test suite runs without torch, in seconds. | Skip torch-dependent tests when torch is absent, and mark the slow ones opt-in. |

### Fitting the repository

| Problem | What to do |
|---|---|
| The check uses its own helpers for run folders, progress reports, cost tracking and Claude calls. | Use the repository's helpers instead. Claude calls go through `shared/api.py`, so they land in the cost log and model-call log, and keys come through `shared/api_keys.py`, which reads only the repository-root `.env` unless `--find-dotenv` is given. Run-folder and file helpers come from `evals/model_judged_audit_support.py`. |
| The input and output paths point at the research project. | `--input` takes a pipeline run folder or a corpus file, and results go under `<run>/audit/log_probability_persona_check_<persona>_<timestamp>/`. |
| The pod needs torch, transformers and related packages, which the pipeline does not. | Keep them in a separate requirements file for the pod. The local analysis needs only numpy, pandas and scipy, which the pipeline already lists. |
| The pod copy of the scoring code would collide with the repository's own `shared` package. | Ship the scorer as a self-contained module, or upload it under its own package name. |

### Making it public

The pipeline repository is public, so nothing personal or specific to this project goes in:
- **No project cost center.** Its name is currently hard-coded in the launcher and its test.
- **Pod settings become flags with neutral defaults:** the SSH key path, the pod image and the list of GPU types.
- **No personal workflow wording** in the runbook and skill.
- **No pointers to this project's plans or data folders.**

## Target design

### Persona-set folder

`prompts/tools/persona_checks/<persona_name>/` holds:

| File | Contents |
|---|---|
| `target_persona_rewordings.txt` | About 30 equivalent statements of the persona. They vary the wording, not the intensity or scope. |
| `topic_control_personas.txt` | About five personas that share the subject but take no stance, five rewordings each, grouped by name. |
| `same_values_control_rewordings.txt` | Optional. A persona with the same values but not the behaviour. |
| `positive_control_rewrite_instructions.txt` | How to rewrite a document towards (and optionally away from) the persona, keeping facts, genre, language and length. |
| `neutral_control_rewrite_instructions.txt` | How to make an edit of the same size that changes register without adding the behaviour. |
| `persona_check_configuration.json` | Persona name, expected counts, direction names and pre-registered pass criteria. |

Preachiness ships in this layout as the example set, and reproduces the original 62-condition run.

### Commands

1. **Validate a persona set.** Free.
2. **Make control rewrites.** Paid Claude calls: positive and neutral, on a chosen sample stratified by a spec variable.
3. **Build scoring inputs.** Free, with a dry run that prints the number of conditions, the token total and a cost range.
4. **Launch, score and pull.** A paid pod, with hard limits on hours and dollars, a watchdog on the pod, deletion at the end and verification of the pulled files.
5. **Analyse.**
   - The positive-control and neutral-control verdicts against the pre-registered criteria, at document and span level.
   - The extreme spans.
   - Rewording agreement.
   - The per-persona breakdown.
   - Document means, labelled as not validated.

### Skill

The skill goes at `.claude/skills/log-probability-persona-check/SKILL.md`, under about 200 lines, written for any contributor. Detail goes in a `references/` folder: the preachiness worked example, a persona-design checklist and the full pod procedure.

**The body teaches the general procedure:**
1. **Define the persona** in one sentence.
2. **Write about 30 rewordings** and check their agreement on a free pilot.
3. **Choose topic controls**, and a same-values control where one exists.
4. **Make a positive control.** Choose sources that already show the behaviour for any "away" arm.
5. **Make a neutral control.**
6. **Pre-register the pass criteria**, at document and span level, before scoring.
7. **Run in stages:** a local rehearsal, then a dry run, a paid pilot and the full run.
8. **Report:**
   - the controls first, saying at which level they passed;
   - span-level results ahead of document means;
   - a sample of extreme spans, read.

**A short "Paid GPU pods" section** covers:
- explicit go-ahead before any paid run;
- hard limits;
- deleting the pod rather than stopping it;
- the watchdog;
- pulling and verifying results;
- the optional manual cost-center step.

The go-ahead rule also gets one sentence in the repository's `CLAUDE.md`, which contributors' agents read first.

## Work plan

Each phase ends with the offline test suite showing no new failures, and nothing paid runs without explicit go-ahead.

| Phase | Deliverable | Check |
|---|---|---|
| 1. Persona sets | Persona-set loader, validator and configuration schema. The preachiness set in the new layout. Background facts built from the repository's prompt files. | The validator passes the preachiness set. The built scoring inputs match the original 62 conditions. |
| 2. Analysis | Span-first analysis, neutral-control verdict, span-level significance test, pass criteria read from the configuration. | Re-analysing the existing preachiness score files reproduces the published document-level and changed-span results. |
| 3. Rewrites | One rewrite command for positive and neutral controls, through `shared/api.py`. | Stubbed tests. A cost estimate for a 10-document sample. |
| 4. Pod | Scorer with atomic writes, launcher with keys through the shared loader, separate pod requirements, neutral defaults. | The launcher tests run with fake clients. A local rehearsal runs end to end with a tiny model. |
| 5. Skill and documentation | Skill and references, an `evals/README.md` section, a `CLAUDE.md` line, git-ignore rules. | Someone unfamiliar with the check can follow the skill to a dry run. |
| 6. Validation run (paid) | A short base-model preachiness pilot, run from the pipeline repository. | Verification against full logits passes, and the outputs match the run-folder contract. |

Phases 1–5 are about 4–6 days of work: around 4,500 lines of code and 2,900 lines of tests come across, and the persona-set and neutral-control work is new. Phase 6 costs a few dollars for the neutral control. A full-corpus run of a new persona costs in the range of the first full run.

**Left out of the port:** the research plans and run data. The pilot go/no-go summary comes across in a general form.

## Decisions

1. **What to port.** Everything needed to run the check goes into the pipeline repository: persona sets, rewrites, scoring inputs, the scorer, the pod launcher, the analysis and a general pilot summary. Keys come from the contributor's own `.env`, and no default is tied to this project.
2. **Repository rules.** "Explicit go-ahead before any paid run" goes into the repository's `CLAUDE.md` as well as the skill.
3. **Default scoring models.** One for each model type, at pinned revisions: instruct `google/gemma-4-12b-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, base `google/gemma-4-12b` at `023679ed352de9bb66cc873c9009ce3482585c08`.
4. **Second persona.** Not part of this port. The design procedure is documented in the skill.
5. **Published example.** The procedure, plus a short summary of the preachiness results. No run data.
6. **Pull request placement.** One pull request, based on `main`. `main` has `shared/api.py` but not `shared/api_keys.py` or `evals/model_judged_audit_support.py`, which exist only on unpushed local branches, so the port uses what `main` has and keeps its own small helpers where needed.
7. **Validation run.** After the port, a short base-model pilot is run from the pipeline repository, after a separate go-ahead.

## Paths

- Research scripts: `scripts/log_probability_persona_check/`
- Research skills: `.claude/skills/log-probability-persona-check/SKILL.md`, `.claude/skills/pod_sessions/SKILL.md`
- Preachiness check plan and writeup: `plans/10-07-document-persona-projection/10-21-development/`
- Preachiness run data: `data/log_probability_persona_check_runs/`
- Pipeline repository: `animal-welfare-data-pipeline/` (GitHub `sentfutures/animal-welfare-data-pipeline`). Its relevant parts are `evals/`, `evals/model_judged_audit_support.py`, `shared/api.py`, `shared/api_keys.py`, `prompts/tools/` and `.claude/skills/`.
