# Pull request 182 correctness review (branch `william/log-probability-persona-check`, reviewed at commit c2900b2)

## Verdict

The scoring and analysis code is correct as far as offline review can tell: token alignment, log-probability arithmetic, resume logic, the statistics and the pass criteria all check out. The defects are in the paid-pod launcher, and one is serious. When `--reuse-existing-pod` refuses because a scorer is already running on the pod, the launcher's cleanup path still kills that scorer and pulls the other run's files into this run's folder. I reproduced this with the existing fake clients. The launcher also does not check that the on-pod self-termination watchdog actually started, and nothing protects the pod during the long setup phase or after a pod-creation request that fails partway. In those cases a pod can keep billing with no automatic shutdown. Fix finding 1 before anyone uses `--reuse-existing-pod`. Fix finding 2 before relying on the watchdog as the "even if the laptop sleeps" safeguard. The rest are low severity.

## Findings (most severe first)

All line numbers refer to `evals/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py` unless another file is named.

### 1. HIGH, CONFIRMED: refusing to reuse a busy pod kills the other run's scorer and pulls its files

- **Where:** `prepare_reused_pod_run_folder` (lines 741-751) and the `finally` block of `run_launcher` (lines 1248, 1270-1279).
- **Defect:** The refusal sets `self.pod_id = None` and raises `SystemExit`. `SystemExit` is not caught by `except Exception` (line 1270), so `reason` keeps its initial value `"exception"` (line 1248). The `finally` block then sees `reason.startswith("exception")` with a connected transfer and runs `stop_scorer_if_running()` and `final_pull_with_verification()` against the pod it just refused to touch.
- **Failing scenario:** Run A is scoring on pod X. Someone launches run B with `--reuse-existing-pod X`. B correctly detects the running scorer and refuses, but its cleanup sends `pkill -f '[s]core_document_log_probabilities_on_pod'` to pod X, which kills run A's scorer. Run A's outputs are then copied into run B's folder, with a logs folder named `pod_None_logs_*`. The pod is not deleted because `pod_id` is None. Run A's launcher then sees its scorer die.
- **Evidence:** I wrote an offline probe using the repository's `FakeTransfer` and `FakeRunPodClient` with `pgrep` reporting a running scorer. After `launch(..., "--reuse-existing-pod", "fakepod123")` raised `SystemExit`, the recorded remote commands were `pgrep -f '[s]core_document_log_probabilities_on_pod' > /dev/null` followed by `pkill -f '[s]core_document_log_probabilities_on_pod'; sleep 2; true`, and one final pull was attempted. The existing test `test_reuse_existing_pod_refuses_when_a_scorer_is_already_running_and_leaves_the_pod_alone` only checks that there was no delete and no upload, so it does not catch this.

### 2. MEDIUM, CONFIRMED: the watchdog start is never verified, but the launcher logs that it is set

- **Where:** `start_watchdog` (lines 842-851). The secret-file write at lines 758-759 has the same pattern.
- **Defect:** `start_watchdog` first kills any existing watchdog with `pkill`, then runs `build_self_termination_watchdog_command(...)` through `run_remote_command`. It ignores that command's exit code and logs "On-pod self-termination watchdog set" regardless.
- **Failing scenario:** The ssh command that starts the watchdog fails, for example with exit 255 from a transient connection drop. The launcher continues and reports a watchdog that does not exist. On a reused pod the earlier watchdog was already killed by the preceding `pkill`, so the pod now has no on-pod deletion at all. If the laptop then sleeps or crashes, the pod bills until someone notices, which defeats the documented second protection layer. If the secret-file write fails, the run fails later with a less clear error. This costs time but no money.
- **Evidence:** In an offline probe where the fake transfer returned exit 255 for the watchdog command, the launcher proceeded to upload and scoring, and the log contained the "watchdog set" line.

### 3. MEDIUM, PLAUSIBLE: no on-pod watchdog during setup

- **Where:** the order of steps in `run_launcher` and `start_watchdog`. The watchdog starts only after the wait for ssh readiness (up to 1200 s), the upload, and the dependency install (ssh timeouts of 900 s each). It also never starts if setup fails before that point and the launcher enters `wait_for_instruction` (default 20 minutes).
- **Defect:** For the first part of every launch, the only thing that deletes the pod is the laptop process. The module docstring says layer 2 works "even if the laptop sleeps".
- **Failing scenario:** The laptop sleeps, or the terminal is killed with SIGKILL, during a slow dependency install. Nothing on the pod or on RunPod's side deletes it.

### 4. MEDIUM, PLAUSIBLE: a pod-creation request that fails partway leaves an untracked billing pod

- **Where:** `RunPodRestClient.create_pod` (lines 162-166, `urlopen(..., timeout=60)`) and the early return in `terminate` (line 612).
- **Defect:** If RunPod creates the pod but the response is lost (timeout, `URLError`, connection reset), the exception propagates with `pod_id` still None. `terminate` returns without deleting anything and no watchdog exists.
- **Failing scenario:** A slow API response during creation leaves a running GPU pod that the launcher does not know about. The user sees only a generic error, with no prompt to check the RunPod console.

### 5. LOW, PLAUSIBLE: a second interrupt during `terminate` skips pod deletion

- **Where:** `terminate`, the delete retry loop with `self.sleep(5)` and `except Exception` (line 621), together with the SIGTERM handler that raises `KeyboardInterrupt`.
- **Defect:** A second Ctrl-C or SIGTERM during the delete retries raises `KeyboardInterrupt`. `except Exception` does not catch it, so it escapes and the delete is abandoned. The final-pull step does catch `BaseException`, but the delete step does not.
- **Failing scenario:** The user presses Ctrl-C twice because shutdown seems slow. The pod is not deleted. The watchdog deletes it later if one was started; if not (see findings 2 and 3), it keeps billing.

### 6. LOW, PLAUSIBLE: a partially copied completion-records file breaks resume

- **Where:** `pull_scoring_outputs` (lines 319-354), which copies `scoring_completion_records.jsonl` while the scorer may be appending to it, and `read_jsonl` / `find_completed_keys` in `timed_progress_and_jsonl_helpers.py` (lines 26-40).
- **Defect:** The incremental pull can capture a truncated last line. The reader calls `json.loads` on every line and does not tolerate a partial final line.
- **Failing scenario:** A run stops after an incremental pull but before a successful final pull, or a scorer is killed in the middle of appending. Resuming from that local copy, or re-uploading it, fails with `JSONDecodeError` until the last line is removed by hand. A normal final pull overwrites the file, so this only appears in interrupted runs.

### 7. LOW-MEDIUM, CONFIRMED (by code trace): the persona-subset merge does not check that both runs used the same prompts

- **Where:** `merge_persona_subset_run_into_full_run.py`, `plan_merged_documents` (lines 132-148) and `plan_merged_conditions` (lines 109-129).
- **Defect:** Documents are compared on `content` only, not on `generation_prompt`. Conditions are compared on `system_prompt` only, not on `author_description` (used by base-model runs), condition group or persona name. The two runs' `prompt_file_sha256` values are never compared.
- **Failing scenario:** A subset run scored with a different user-prompt framing, or with different base-model author descriptions, merges into the full run without any error. The merged per-token differences then mix two different measurements. The only sign is a generic difference in the reproducibility log.

### 8. LOW, PLAUSIBLE: the pilot summary double-counts cost when a launcher attaches to an existing pod

- **Where:** `summarize_pilot_for_go_or_no_go_decision.py` line 194, which sums `total_cost_dollars` and running seconds over all pod cost logs. Also `attach_to_pod` (lines 664-673) and `find_previous_start_epoch_seconds` (lines 1199-1204).
- **Defect:** A log written with `--attach-to-existing-pod` starts from the earlier log's start time, so each log covers the whole pod lifetime. Summing the logs counts the shared period twice.
- **Failing scenario:** Each pilot run that reattaches once reports roughly twice its real cost and setup time. The error overstates cost, so it is conservative, but it can push a go/no-go decision the wrong way.

### 9. LOW, CONFIRMED (by code trace): timestamps only resolve to the minute, so outputs can overwrite each other

- **Where:** `timestamp_now` (format `%Y-%m-%d_%H-%M`).
- **Defect and scenario:** If the launcher runs twice within one minute (for example, a quick relaunch after a refusal), the second run overwrites `pod_cost_log_*.json` and the progress log. If the scorer restarts within a minute, it overwrites `scoring_environment_*.json`. Related minor issue: if `costPerHr` comes back as 0, `effective_limit_seconds` raises `ZeroDivisionError`. The pod is still deleted through the exception path.

## Checked and found sound

- **Cost limits:** the cost-so-far formula (price times seconds over 3600), `effective_limit_seconds` (the smaller of the hours limit and dollars divided by price), checks in every monitor loop and in `wait_for_instruction`, the final-pull time budget (`min(300, seconds until a limit)` with a 60 s minimum attempt), and the watchdog deadline (limit minus elapsed time plus 600 s grace). The units are consistent throughout.
- **Secrets:** the Hugging Face token reaches `pod_secret_environment.sh` (written under umask 077) through stdin. It does not appear in the pod-creation environment, command lines, logs or the final copy list. Dry runs mask it. Keys are read only from the repository `.env` through dotenv.
- **Shell safety:** remote paths are constants, scorer arguments are quoted with shlex, local tar takes list arguments, and the bracketed `pkill`/`pgrep` patterns cannot match their own command.
- **Final-pull verification:** each pulled file's size is compared with the remote listing after the scorer stops, truncated streams are detected, the incremental `.next` marker logic does not lose files, `*.partial` files are excluded, and writes are atomic (`os.replace` plus `utime`).
- **Scorer:**
  - The hidden state at position p is used to predict token p+1 (`start-1:end-1`).
  - Padding is on the right, with an attention mask.
  - log_softmax is computed in float32 chunks, with softcapping.
  - Outputs are checked to be finite and to have the expected sign.
  - Document token IDs are asserted identical across conditions.
  - `locate_document_tokens` checks the exact prefix and suffix tokens and runs a decode check.
  - Base-model and instruct-model rendering are handled separately.
  - Resume uses the npz files plus the completion records, including a rebuild path.
  - Batch packing is correct.
- **Analysis:**
  - Per-token persona minus generic differences, and window means computed with cumulative sums.
  - Greedy selection of non-overlapping windows, with correct lexsort keys.
  - The permutation test: max/min statistic, p = (1 + k) / (1 + n), one-sided tails.
  - Benjamini-Hochberg correction and the ICC(1,1) formula.
  - Pass-criteria inequalities and the "not evaluable" logic.
  - The neutral-condition upper bound, detection of mixed models or revisions, and condition group and count checks.
  - The merge formula (new minus new generic plus base generic) preserves the differences.
  - The reader for older output formats.
- **Persona-set validation and inputs:**
  - Duplicate and blank lines, header format, reserved names and counts against the config.
  - Unknown schema keys and the direction of each pass criterion.
  - Refusal of `# DRAFT` instructions before any paid call in `make_control_rewrites` (dry runs are allowed).
  - Resume hash checks in `build_scoring_inputs` and the `_again` config naming.

## Not covered

- Model internals for Gemma 4 and Qwen3, such as whether `model.base_model` applies the final norm as the scorer assumes. These cannot be checked offline.
- `select_pilot_documents.py`, the smoke-test script, the skill documents and the run-folder layout document.
- The selection and stratification details in `make_control_rewrites`.
- The real `SshTarTransfer` tar behaviour, which was not executed.
- The full existing test suite, which was not run. Only the launcher tests with fake clients and my own probes were run.
- No paid or API-calling code was run.
