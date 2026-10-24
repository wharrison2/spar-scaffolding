# Pilot runbook: log-probability preachiness check

Run every command from the repository root. Placeholders in angle brackets depend on the timestamp the earlier step printed. Steps marked PAID cost money and need the user's explicit go-ahead for that step; all other steps are free and local.

The pilot: Opus 5 rewrites of 10 source documents (preachier and plainer), a scoring run of 20 corpus documents (the 10 rewrite sources plus 10 more) and the 20 rewrites under all 62 conditions on one RunPod pod capped at 2 hours and 10 dollars, the analysis, and the go or no-go summary. Layouts: `run_folder_layout_and_script_interfaces.md`. A full local rehearsal with no paid call: `rehearse_pilot_pipeline_locally_without_paid_calls.py`.

Do not pass `--maximum-documents` to the launcher: the analysis stops if any document in the run has no scoring output.

## 1. Rewrites dry run (free)

```
python3 scripts/log_probability_persona_check/make_preachiness_positive_control_rewrites.py --dry-run
```

Prints the cost estimate (about 5 dollars). Output: `data/log_probability_persona_check_runs/small_sample_checks/preachiness_positive_control_rewrites_<timestamp>/` (rendered requests, cost estimate, run configuration).

## 2. Rewrites real run (PAID: Anthropic API, claude-opus-5, about 5 dollars)

```
python3 scripts/log_probability_persona_check/make_preachiness_positive_control_rewrites.py
```

Output: `data/log_probability_persona_check_runs/positive_control_rewrites/preachiness_positive_control_rewrites_<timestamp>/` with `rewritten_documents.jsonl`, `rewrite_failures_<timestamp>.jsonl`, `usage_and_cost_log_<timestamp>.json`, `progress_log_<timestamp>.txt`. Uses `ANTHROPIC_API_KEY` from the project `.env`. If `rewrite_failures_*.jsonl` is not empty, retry the failed pairs (the finished ones are skipped):

```
python3 scripts/log_probability_persona_check/make_preachiness_positive_control_rewrites.py --resume-folder <rewrites folder>
```

`<rewrites folder>` below is `data/log_probability_persona_check_runs/positive_control_rewrites/preachiness_positive_control_rewrites_<timestamp>`.

## 3. Select the pilot documents (free)

```
python3 scripts/log_probability_persona_check/select_pilot_documents.py \
    --positive-control-rewrites-folder <rewrites folder> --number-of-documents 20 --seed 0
```

Output: `data/log_probability_persona_check_runs/small_sample_checks/pilot_document_selection_<timestamp>/pilot_document_ids_<timestamp>.txt` and `pilot_document_selection_<timestamp>.json` (sources, additional documents, seed, counts by tone and language). All rewrite sources are included even if some of their rewrites failed.

## 4. Build the scoring inputs (free)

```
python3 scripts/log_probability_persona_check/build_scoring_inputs.py --run-name pilot_twenty_plus_twenty_documents \
    --small-sample-check \
    --document-ids-file data/log_probability_persona_check_runs/small_sample_checks/pilot_document_selection_<timestamp>/pilot_document_ids_<timestamp>.txt \
    --extra-documents-jsonl <rewrites folder>/rewritten_documents.jsonl
```

Output: the pilot run folder `data/log_probability_persona_check_runs/small_sample_checks/pilot_twenty_plus_twenty_documents_<timestamp>/` (`<pilot run folder>` below) with `scoring_inputs/documents.jsonl` (40 documents), `scoring_inputs/conditions.jsonl` (62 conditions) and `run_configuration_<timestamp>.json`.

## 5. Launcher dry run (free)

```
python3 scripts/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py --run-folder <pilot run folder> \
    --maximum-running-hours 2 --maximum-dollars 10 --verify-first-document-against-full-logits --dry-run
```

Prints the pod request, the commands that would run on the pod (including the scorer command line) and the cost ceiling. Creates no pod.

## 6. Launcher real run (PAID: RunPod pod, capped at 2 hours and 10 dollars)

```
python3 scripts/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py --run-folder <pilot run folder> \
    --maximum-running-hours 2 --maximum-dollars 10 --verify-first-document-against-full-logits
```

Run it in a background shell. Uses `RUNPOD_API_KEY`, `RUNPOD_COST_CENTER_NAME` (and `HF_TOKEN` if set) from the project `.env`. The pod is deleted on completion, at either limit, on Ctrl-C and on any error, and a watchdog on the pod deletes it after the limit as well. Interrupted or failed: run the same command again, and the scorer skips finished documents.

Output in `<pilot run folder>`: `scoring_outputs/` (`per_token_log_probabilities/*.npz`, `scoring_completion_records.jsonl`, `scoring_progress_log_<timestamp>.txt`, `scoring_environment_<timestamp>.json`, `full_logits_verification_<timestamp>.json`), `pod_cost_log_<timestamp>.json`, `pod_progress_log_<timestamp>.txt`, `scorer_standard_output.txt`. The pod log has a line `Pod environment: torch ... transformers ... cuda available ...` before scoring starts.

## 7. Cost center assignment (by hand, while the pod runs)

As soon as the launcher prints the pod name, assign it at https://www.console.runpod.io/user/cost-center/manage: "Uncategorized resources", tick the pod, "Add resources to cost center", cost center name from `RUNPOD_COST_CENTER_NAME`. Then confirm to the launcher either by typing `assigned` in its terminal or, for a background launcher, by creating the file:

```
touch <pilot run folder>/cost_center_assignment_confirmed_by_user.txt
```

Only create the file once the user says the pod is assigned. The pod cost log records the cost center and whether the user confirmed. Without confirmation by the end of a normal finish, the launcher waits 2 more minutes and then deletes the pod.

## 8. Analysis (free)

```
python3 scripts/log_probability_persona_check/analyze_preachiness_scores.py --run-folder <pilot run folder>
```

Output: `<pilot run folder>/analysis_<timestamp>/` (per-document scores and significance tests, extreme spans, rewording agreement, scores by spec variable, positive control check, figures, notes).

## 9. Go or no-go summary (free)

```
python3 scripts/log_probability_persona_check/summarize_pilot_for_go_or_no_go_decision.py --run-folder <pilot run folder>
```

Output: `<pilot run folder>/pilot_go_or_no_go_summary_<timestamp>.md` (also printed): verification result, throughput, pod cost, projected full-run cost and time, positive control, rewording agreement, significant documents, cost center, and the criteria with whether each is met.
