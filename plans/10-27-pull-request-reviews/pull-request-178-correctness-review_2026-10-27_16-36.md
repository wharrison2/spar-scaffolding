# Pull request 178: correctness review

Reviewed: `git diff main...william/sdf-planning-spec-fixes-and-considerations-selector` (head commit 029647d). The offline test suite passes on the head branch.

## Verdict

The core algorithms hold up. The swap repair keeps each column's counts exactly and always terminates, and the current rules repair cleanly at every deal size and seed I tried. The linear programme matches its documented constraints and penalty, and the screen's round logic is sound. The parsing of structured outputs is strict and fails safe. There is one real defect, in how the considerations selector handles partial failures and resumes. If a document's scoring fails, that document still goes on to drafting with no considerations, and a later resume cannot add them. A resume that rescores it also reruns the global assignment and silently changes the considerations of documents that are already drafted. After that, `consideration_selections.jsonl`, `consideration_coverage.json` and the entry caps no longer match the corpus. The other findings are low severity: the screen leaves inconsistent state if it is interrupted in the middle of a multi-record write.

## Findings

### 1. A scoring failure sends the document to drafting with no considerations, and a later resume reassigns documents that are already drafted

- **Severity:** medium
- **Status:** CONFIRMED (demonstrated offline with a stubbed `api.call_claude`)
- **Location:** `sdf_pipeline/layer1_select_considerations.py:406-445` (and `sdf_pipeline/run.py:104-115`)
- **Defect:** `run()` stops only when *every* pending scoring call fails. Any document that failed some but not all of its scoring attempts is returned from `_attach` with `considerations_spec == ""` and empty consideration lists, and `run.py` drafts it straight away. A resume then rescores it, which reruns the whole assignment and can change other documents' assignments. Those documents' layer-2 drafts already exist and are reused unchanged.
- **Failing scenario:** Take four eligible documents, where the answer for `matrix_000000` is unusable on all attempts. The stage logs "will be rescored on resume" but returns the plan with an empty specification:
  ```
  matrix_000000 '' []
  matrix_000001 'substrate neutrality (welfare and suffer' ['substrate neutrality', 'a deliberately open moral circle']
  matrix_000002 'the continuity between welfare concern a' [...,'scope sensitivity']
  matrix_000003 'individualism (the notion that welfare i' ['individualism', 'the fact that “natural” does not mean “OK”']
  ```
  Layer 2 drafts all four, so `matrix_000000` is drafted with no considerations. On `--resume`, `matrix_000000` is scored and the assignment is solved again:
  ```
  matrix_000000 changed ['the precautionary principle applied to sentience', 'individualism']
  matrix_000001 same
  matrix_000002 changed
  matrix_000003 changed ['the fact that “natural” does not mean “OK”', 'any relevant principles from the constitution']
  ```
  Layer 2 reuses the existing drafts by `doc_id` (`layer2_draft.py`, `existing = {...}`). As a result:
  - `matrix_000000` is never redrafted, so the considerations it was just given are never used.
  - `matrix_000003`'s draft (and its record's `variables`) still carries `individualism`, but `consideration_selections.jsonl` and `consideration_coverage.json` now say otherwise.
  - `individualism` now sits in two documents of the actual corpus, against a target (cap) of 1.

  The same reassignment also happens when the description rewrite holds plans out on the first run (failed or unusable rewrite calls) and a resume brings them back. That changes the eligible set, the target counts and the solution.
- **Evidence:** The selector has no equivalent of the description rewrite's hold-out (`layer1_description_rewrite.apply_results` drops unchecked plans): `_attach` gives every described plan a spec whether or not it was scored, and the assignment is a fresh, global solve on every layer-1 entry. The probe script is `/private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/08c219e3-e182-4ff7-a9a8-e7625dddc1d7/scratchpad/review_scripts/selector_failure_probe.py`.

### 2. An interrupted multi-record write in the screen can let unscreened hands, or duplicate hands, into the accepted set

- **Severity:** low (needs an interruption inside a short local-write window)
- **Status:** PLAUSIBLE (traced, not run)
- **Location:** `sdf_pipeline/layer1_screen.py:416-426`, `:329-332`, `:230-237`, and `_finish` at `:438-441`
- **Defect:** `_finish` accepts *every* hand in `screen_hands.jsonl` that has no rejection. It does not check that the hand belongs to a recorded round or a checkpointed batch. Several steps also write multiple records with no atomic commit:
  - redealt hands are written before their round record;
  - round 0's hands are written before its round record;
  - a round's batches are written one line at a time.
- **Failing scenarios:**
  - **Redeal interrupted after some new hands are appended but before the round record.** On resume, round N completes again from checkpoints and is redealt with the same seed. `_next_hand_index(hands)` now counts the orphan hands, so the new hands get shifted ids. The orphans are in no round and are never screened, yet `_finish` accepts them, which duplicates those cards and pushes the column counts above the dealt counts.
  - **Interrupted while `plan_round_batches` is writing.** On resume, `saved` is non-empty but partial and is reused verbatim. The hands in the missing batches are never screened, and they are accepted.
  - **Interrupted during round 0's hand writes.** The rerun deals again and appends the full set after the partial one. A later resume loads both copies, and `accepted` then holds duplicate `prompt_id`s.
- **Evidence:** In `_finish`, `accepted = [... for h in hands if h["prompt_id"] not in rejected]` iterates every hand in the file, with no round or batch membership check.

### 3. Re-screening a batch after an interruption keeps a stale removal

- **Severity:** low
- **Status:** PLAUSIBLE (traced)
- **Location:** `sdf_pipeline/layer1_screen.py:367-382`, `load_rejections` at `:278-280`
- **Defect:** A batch's rejections are appended before its checkpoint. If the run stops between the two, the batch is screened again on resume. If the new answer keeps a hand that the first answer removed, the earlier rejection stays in `screen_rejections.jsonl`. `load_rejections` keys on `prompt_id`, so the hand is still treated as removed and redealt. Meanwhile `screen_judgments.jsonl` holds both verdicts.
- **Evidence:** No code path removes or overrides an earlier rejection. "Readers de-duplicate by prompt_id" (module docstring) only holds when the two answers agree.

## Checked and found sound

- **Swap repair** (`shared/deal_constraints.py`):
  - Swaps happen within one column, so per-variable counts are exactly preserved.
  - Each accepted swap strictly lowers the total violation count: the repaired hand loses at least the target rule and gains none, and the partner ends with zero violations. The loop therefore terminates. A pass with no swap raises.
  - `check_feasibility` is a correct necessary condition.
  - Dealing with the real rules and weights at n = 10, 30, 100 and 1000 over 40 seeds each produced no infeasibility and zero remaining violations.
- **Redeal:**
  - The removed hands themselves are a valid assignment of the pooled cards, so the redeal's feasibility check cannot fail on counts.
  - Repair is confined to the new hands, and accepted hands are never touched.
  - A single removed hand correctly yields None and is dropped.
  - `round_number >= redeal_rounds` gives exactly `screen_redeal_rounds` redeal rounds.
  - `_finish` picks the last round that has hands, so a failed-redeal round with no hands still reports the right leftovers.
  - Resuming after an interruption between rounds (`round_number + 1 < len(rounds)`) works.
- **Screen answer parsing:** It requires exactly N entries with numbers 1..N, each once, and correct types. Truncated (non-`end_turn`) answers are rejected and not checkpointed. A batch failure stops the stage before any redeal.
- **Linear programme** (`considerations_assignment.py`):
  - The objective sign is correct: linprog minimises `-gains` plus the penalty costs.
  - The entry, per-document, per-(document, list) and total caps match the documentation.
  - The penalty columns (costs k·p, enough columns for min(cap, total cap, pairs) − 1) reproduce p·n(n−1)/2.
  - The constraint matrix is totally unimodular: two laminar families, plus columns with a single −1.
  - The tie-break magnitude stays below 1 when the weights and penalty are integers, and it is strictly negative, so a pair whose weight exactly equals its penalty is not assigned.
  - Integrality and constraints are re-checked after solving.
- **Selector parsing:** Every offered label must appear exactly once with an integer score from 1 to 5, and scores of 3 or more need a reason. A list that was not offered must be empty. The schema is built per call.
- **Description rewrite:**
  - The flags must be consistent with each other: a conflict has no rewrite, and a rewrite has a description and a known check.
  - A result is appended before its checkpoint.
  - Unchecked plans are held out of drafting.
  - Resuming past layer 1 re-applies the stored results.
- **OpenAI wrapper:**
  - A refusal or an incomplete status maps to a stop reason other than `end_turn`.
  - Cost is (input − cached)·input price + cached·cached price + output price, with reasoning tokens included in output, and it is logged once per successful call.
  - The key is read from the process environment only.
- **Structured outputs:** On the Anthropic API the schema is passed as `output_config.format`, which the installed SDK (2.1.0) accepts. On Claude Code, the installed `claude_agent_sdk` (0.2.170) exposes `structured_output` and `output_format`. In schema mode the code never joins prose text blocks into the answer.
- **Variables:** `variables.txt` IDs parse, and every column's weights sum to 1.0. The `no_stake` weight is 0, which correctly gives it a zero quota.
- **Rendering rules:** The screen and the rewrite show the same variable lines as the planner: the same dropped lines and the same per-id example species.
- **Downstream wiring:** `considerations_spec` is carried through layers 2, 3 and 4. The layer-2 template line is dropped when the spec is empty. Older snapshots fall back for `reasoning_featured`.
- **Report fix:** The `mean_l5` → `mean_l4` change in `evals/report_sdf.py` points at a variable that exists.
