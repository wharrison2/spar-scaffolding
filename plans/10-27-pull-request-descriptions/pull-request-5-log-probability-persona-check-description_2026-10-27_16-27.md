Title: Add a log-probability persona check for corpus documents

## Summary

This PR adds `evals/log_probability_persona_check/`, a method for estimating whether the documents in a corpus would push a model trained on them towards a persona such as "preachy". It includes the persona definitions, control rewrites, scoring on a rented GPU, and the analysis. It is independent of the SDF stack and almost entirely new files. It also adds a skill describing how to run and read the check.

## What it measures

A persona is a one-sentence description of an assistant's behaviour, for example "an AI assistant who moralizes about animal welfare". An open-weights model reads each document as if it were an assistant's reply and records the log-probability (how likely the model finds each token) under a generic system prompt, under many rewordings of the persona's system prompt, and under control prompts. The score is how much more likely the document's text becomes under the persona than under the generic prompt. Text that the persona makes unusually likely, compared with the controls, is text that would pull a model towards that persona.

**What is validated.** Controls are rewrites of real documents, made with Claude, towards the persona and in a neutral register. The README and analysis treat span-level results (windows of 32 tokens, tested against a permutation null that treats tokens as exchangeable, so it screens for spans and does not prove anything) as the primary output. Whole-document means are written to files prefixed `secondary_not_validated_` and are not a validated measure.

## What changes

| Component | What it does |
|---|---|
| `persona_sets.py`, `prompts/tools/persona_checks/` | Loads and validates a persona set: target rewordings, topic controls, same-values controls, rewrite instructions, and a configuration file holding the pass criteria. Ships one set (`preachiness`) and the shared prompt framing for both model types. |
| `prompt_rendering.py`, `scoring_model_defaults.py` | Builds each condition's prompt. Instruct models use their chat template with the persona as system prompt; base models get a plain-text preamble naming the author. Defaults and pinned revisions per model type. |
| `make_control_rewrites.py` | Rewrites sampled documents towards the persona, optionally away from it, and in a neutral register. Paid (Claude calls); `--dry-run` is free. |
| `select_pilot_documents.py`, `build_scoring_inputs.py` | Pick pilot documents and write a run folder (documents, conditions, run configuration with file hashes). Resumable; `--only-personas` builds a subset run. |
| `score_document_log_probabilities_on_pod.py` | The scorer. Writes per-token log-probabilities for every document under every condition, resumable. Also runs on CPU with a tiny model. |
| `launch_and_stop_runpod_scoring_pod.py` | Creates a RunPod pod, uploads inputs, runs the scorer, pulls and verifies results, deletes the pod. Paid; `--dry-run` is free. |
| `analyze_persona_scores.py`, `analyze_changed_spans.py`, `analysis_run_loading.py` | Control verdicts against the pass criteria, extreme spans, span significance, rewording agreement, per-persona breakdown, and the not-validated document means. |
| `summarize_pilot_for_go_or_no_go_decision.py` | After a pilot: verification, throughput, cost, a projected full-run cost range, and each criterion met or not. |
| `merge_persona_subset_run_into_full_run.py` | Combines a finished run with a persona-subset run into a new full run folder. |
| `legacy_run_format_compatibility_reader.py` | Lets the analysis read run folders in an earlier format. |
| Rehearsal and smoke test scripts | Run the whole local chain on a tiny model, and optionally on Qwen3-0.6B, without paid calls. |
| `.claude/skills/log-probability-persona-check/` | Procedure, persona design checklist, worked example, paid pod procedure. |
| `evals/README.md`, `CLAUDE.md`, `.gitignore`, `requirements.txt`, `tests/conftest.py` | A new README section and table row; a note that paid runs need explicit go-ahead; ignore rules for per-token score files (`per_token_log_probabilities/`, `*.partial`) and pod log copies; the `scipy` requirement; an opt-in `--run-slow` option for tests marked `slow`. |
| `tests/test_log_probability_persona_check_*.py` and fixtures | Offline tests for every module above. |

Two model types are supported: `instruct` and `base`. Their scores are not comparable, each needs its own pinned 40-character model revision, controls must be rerun for each, and the analysis refuses a run folder that mixes types, models or revisions.

## Paid steps and safeguards

- The control rewrites and the pod launch cost money. The repository docs require the contributor's explicit go-ahead before either, including small tests and pilots, and every paid step has a free `--dry-run` that prints the cost ceiling first.
- Every launch takes a hard limit on running hours and on dollars. The pod is deleted when either is reached, on Ctrl-C and on termination, and a watchdog on the pod deletes it after the time limit even if the local launcher loses contact.
- Pods are deleted, never stopped, since a stopped pod still bills for storage. Results are pulled every few minutes; at the end everything is copied back and file names and sizes are verified before deletion.
- After a fixable failure the pod is kept briefly (`--failure-wait-minutes`, never past the hard limits) and waits for an instruction file in the run folder: retry the scorer, retry setup, or delete.
- Cost-centre assignment is optional and manual, since RunPod has no API for it. Only if `RUNPOD_COST_CENTER_NAME` is set does the launcher ask for it and record the confirmation.
- The watchdog starts as soon as the pod accepts ssh, before upload and install, and the launcher verifies that it started; if it cannot be started the pod is deleted rather than left unprotected.
- A reused pod (`--reuse-existing-pod`) is left untouched, undeleted and with its scorer running if it is busy or if setup fails before that check. If pod creation fails after the request may have reached RunPod, the launcher names the pod, looks for it and deletes it.
- Keys are read from the repository `.env`, not from the shell environment. Rewrite instruction files starting with `# DRAFT:` are refused by the paid step.

## Adding a new persona

Create a folder `prompts/tools/persona_checks/<name>/` with the same files as `preachiness/`; no code change is needed. Validate it offline with `persona_sets.py --persona-set <name>`. Design guidance, including choosing controls and fixing pass criteria before scoring, is in `.claude/skills/log-probability-persona-check/references/persona_design_checklist.md`.

## How to review

Start with `evals/log_probability_persona_check/README.md` and `run_folder_layout_and_interfaces.md` (the contract between steps), then `prompt_rendering.py` and `analyze_persona_scores.py`. The `preachiness` files under `prompts/tools/persona_checks/` show a complete persona set.

Judgement is needed on: whether the rendered prompts are right for each model type (the document tokens must be identical across conditions); whether the paid-step limits in `launch_and_stop_runpod_scoring_pod.py` are tight enough; and whether the wording about what is and is not validated is appropriately cautious. The launcher is the largest file (about 1,300 lines); its safeguards are covered by `tests/test_log_probability_persona_check_runpod_launcher.py`.

## Testing

The offline suite passes on a clean install from `requirements.txt` (1416 passed, 5 skipped). The 5 skipped tests need `torch`, which CI does not install; they cover the pod-side scorer. No test calls an AI API or a pod.

## Merge notes

This PR and #181 (part of the SDF stack) both append to `.gitignore` and `evals/README.md`. Whichever merges second will have two additive conflicts; resolve by keeping both blocks. The `scipy` line in `requirements.txt` is identical to the one in #178, so it does not conflict.
