# Pull request 180 correctness review (`william/sdf-plumbing-model-calls-and-api-keys` against `william/sdf-generation-phrasing-fixes`)

## Verdict

I found no high- or medium-severity correctness defects. Key lookup works as the `shared/api_keys.py` docstring says in both modes:
- the source chosen and its precedence
- empty and whitespace values
- the once-per-variable message
- the error text
- the flag forwarded to the DAD runner's evaluation subprocesses

Every entry point that calls a model applies the flag before any key is loaded. Opus 5.5 pricing ($4 / $20, cache reads at 0.05 of input, which is $0.20 per million tokens) and its thinking settings (adaptive with `display: "summarized"`, effort `low`, no sampling parameters) match the current API documentation, and the installed `claude-agent-sdk` 0.2.170 accepts `thinking` with `display` and `effort`. The existing suite passes on the head commit (150 tests in `tests/test_api_keys.py`, `tests/test_api.py`, `tests/test_openai_api_and_description_rewrite_routing.py` and `tests/test_embeddings.py`). The five findings below are all low severity:
- the model-call log can turn a successful paid call into an exception
- evaluations write full records to a shared global file, which the docstring says they do not
- `HF_TOKEN` still comes from the operating-system environment in the default mode
- two robustness gaps, in the test fixture and in the system-prompt file write

## Findings (most severe first)

### 1. A failure inside the model-call log turns a successful paid call into an exception, or hides the real error

- **Severity:** low
- **Status:** CONFIRMED (the mechanism was shown with an offline probe test using fake keys)
- **Location:** `shared/api.py:284` (`json.dumps(record, ensure_ascii=False)` written to a UTF-8 file), called at `shared/api.py:841` onward on success and at `shared/api.py:817` inside the failure handler. The same applies to the calls in `shared/openai_api.py`.
- **Defect:** `_log_model_call` runs after the response has been received and costed, and nothing guards it. Any exception it raises propagates out of `call_claude`, so the caller loses the response it paid for. The `cost_log.jsonl` line has already been written.
- **Failing scenario:** a user message or response that contains a lone UTF-16 surrogate. For example, `json.loads('"doc \\ud83d truncated"')` yields one from an escaped JSONL field. `json.dumps(..., ensure_ascii=False)` keeps the surrogate, and writing it to a UTF-8 file raises `UnicodeEncodeError`. The cost log uses `ensure_ascii=True`, so it is unaffected. On the failure path (`except Exception as e:` at line 817), the same logging error would replace the original API exception. A full disk or a permissions error on `system_prompts/` has the same effect.
- **Evidence:** the probe test set `_cost_log_path`, stubbed `_call_with_retry` to return `"ok"`, and called `api.call_claude(<text with \ud83d>, system_prompt="s")`. It raised `UnicodeEncodeError`; `cost_log.jsonl` held the call and `model_calls.jsonl` was empty. Inputs in this pipeline rarely contain lone surrogates, so this is unlikely in practice. The fragility is that a diagnostic log can fail a generation call.

### 2. Evaluations write full model-call records to one global `outputs/model_calls.jsonl`, though the docstring says they write nothing

- **Severity:** low
- **Status:** CONFIRMED (traced)
- **Location:** `shared/api.py:257` (docstring: "Writes nothing when the run has no cost log path (tests, evals)"), together with `api.init(args.config)` in `evals/audit_sdf.py:697`, `evals/audit_dad.py:2223`, `evals/compliance_sdf.py:186`, `evals/score_sdf.py:44` and `evals/diversity.py:708`.
- **Defect:** `init` without `cost_log_path` sets `_cost_log_path` to `config.yaml`'s `outputs.cost_log` (`outputs/cost_log.jsonl`), so `_log_model_call` does write for evaluations. It writes into `outputs/model_calls.jsonl` and `outputs/system_prompts/`.
- **Failing scenario:** every evaluation run appends to the same file across all runs and corpora, including the judge passes that the DAD runner launches through `run_auto_evals`. Each record holds the full user message (a whole corpus document) and the full response. The file grows without bound and records from different runs mix. The fields `stage` and `item_id` are the only way to tell them apart, and there is no run identifier. Nothing breaks, but the stated contract is false, and anyone relying on "evaluations write nothing" will be surprised. The bare-name `.gitignore` entries do keep these files out of git.
- **Evidence:** `shared/api.py` `init()` sets `_cost_log_path = Path(cost_log_path or _config["outputs"]["cost_log"])`, and `_log_model_call` returns early only when that path is `None`.

### 3. In the default mode, `HF_TOKEN` is still taken from the operating-system environment

- **Severity:** low
- **Status:** CONFIRMED (traced; this may be intended, given the comment at the call site)
- **Location:** `evals/publish_hf.py:837`, and `HfApi()` at `evals/publish_hf.py:594`, `600` and `621`. The contract is in `shared/api_keys.py:10`.
- **Defect:** `load_optional_api_key("HF_TOKEN")` in the default mode returns `None` when `<repo root>/.env` lacks the token. But `HfApi()` is built without a `token=` argument, so `huggingface_hub` then reads `HF_TOKEN` from `os.environ` itself. This contradicts "The operating system environment is never used" for the default mode.
- **Failing scenario:** the repo-root `.env` has no `HF_TOKEN`, the shell exports `HF_TOKEN=<some other account's token>`, and `publish_hf.py` is run without `--find-dotenv`. It publishes with the environment token. A user who expects default-mode isolation would instead expect the cached-login fallback, or a failure. The comment at line 835 ("from <repo root>/.env (else the environment)") suggests the author accepts this. If so, the module contract should carve out an exception; if not, pass `token=` explicitly or remove `HF_TOKEN` from the environment in the default mode.

### 4. The test fixture's upward search is not confined to the empty temporary folder

- **Severity:** low
- **Status:** PLAUSIBLE
- **Location:** `tests/conftest.py:84` (`_api_key_source_mode = SEARCH_UPWARD_THEN_ENVIRONMENT`, with `REPO_ROOT` set to a `tmp_path_factory` folder).
- **Defect:** the fixture's docstring promises "no test ever reads a real .env". In search mode, however, `_nearest_dotenv_path_searching_upward` walks from the temporary folder up to `/`.
- **Failing scenario:** pytest's base temporary folder sits under a directory that has a `.env`. That happens with a `--basetemp` inside the project, or a `TMPDIR` under the home folder or the project. The real key is then read and written over the fake one in `os.environ` for that test. Network access is blocked by `--disable-socket` and the environment is restored at teardown, so the only consequence is that a real key enters test-process memory.
- **Evidence:** this is reasoned from the code. I did not probe the machine's temporary-folder ancestors, because that would have meant touching `.env` files.

### 5. The system-prompt file is written non-atomically and never repaired

- **Severity:** low
- **Status:** PLAUSIBLE
- **Location:** `shared/api.py:280` to `282`.
- **Defect:** `if not system_prompt_file.exists(): write_text(...)`. If the process is killed partway through writing a constitution-sized prompt, the truncated file stays. Every later call sees that it exists and skips the write, so `system_prompts/<sha256>.txt` no longer matches its hash, permanently and silently. The lock only covers threads in one process. Two evaluation processes that share `outputs/` (finding 2) can also both write the same file at once, which is harmless because the content is identical.
- **Fix direction:** write to a temporary file and `os.replace` it into place.

## Checked and found sound

- **`load_optional_api_key` and `load_api_key` precedence:**
  - The default mode reads only the resolved `<repo root>/.env`, ignores the environment, a parent `.env` and the current working directory, and raises `MissingApiKeyError` naming the path, the key and `--find-dotenv`.
  - The `--find-dotenv` mode takes the nearest `.env` upward (which wins over the environment, as documented), and falls back to the environment otherwise.
  - Values that are empty, whitespace-only or quoted whitespace count as absent.
  - A value read from a file is written to `os.environ` with the stripped value.
- **Once-per-variable messages:** they are keyed by variable name, and the key value is never printed in either mode. The tests assert this.
- **`MissingApiKeyError(RuntimeError, KeyError)`:** it constructs and stringifies correctly. Callers that catch `KeyError` (the old `init`) or `RuntimeError` (the old client getters) still catch it. It does not leak into the `auto`-backend `RuntimeError` demotion handler, because no key is loaded inside the `claude_code` branch.
- **Entry points:** every `__main__` script that reaches `api`, `embeddings` or `openai_api` calls `add_api_key_source_flags` and `apply_api_key_source_flags` immediately after `parse_args`, before any key is loaded. The scripts are the `sdf`, `pref` and `dad` runners, `audit_sdf`, `audit_dad`, `compliance_sdf`, `diversity`, `score_sdf`, `publish_hf` and the rerun script. No module loads a key at import time (`load_dotenv` is gone everywhere), and `resolve_default_model` is called only at run time.
- **Child processes:** the DAD runner forwards `--find-dotenv` to both evaluation subprocesses through `api_key_source_flags_for_subprocess()`, and those subprocesses accept the flag. No other entry point spawns Python children.
- **`claude_code` backend:** it still blanks `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` for the command-line tool, even though the key is now written into `os.environ`.
- **Model-call log:**
  - One record is written per call on each path: the `api` success, the `api` failure (including a missing key in `_get_client`), the `claude_code` success, the `claude_code` refusal, the non-`auto` `claude_code` failure, and OpenAI success and failure.
  - Nothing is written twice. The `auto` demotion writes only the `api` leg's record.
  - Writes are serialised by `_cost_log_lock` within a process.
  - No key value reaches the records; the error strings are exception messages only.
  - The thinking summary is recorded but removed from the returned text.
  - `thinking_text` is reset at the start of each call through the thread-local `_attempt_state`.
- **Opus 5.5 handling:**
  - The `_PRICING` entry (4.00, 20.00) and the 0.05 cache-read multiplier are applied by an exact model lookup, so other models keep 0.10.
  - Adaptive thinking and `effort: low` are sent only to `claude-opus-5-5` (and adaptive only to the fable and mythos models). `claude-opus-5` keeps `thinking: disabled`, which is valid at its default effort.
  - `output_config` merges the effort with the JSON-schema format correctly.
  - Temperature is not sent, through the existing `claude-opus-5` prefix in `_NO_SAMPLING_PREFIXES`.
- **Tests:** the new tests in `tests/test_api_keys.py` reinstate the real `set_api_key_source_mode` that `conftest` disables, so the mode tests can fail. The record tests check the written files for content.
- **Intended behaviour change (not a defect, but worth confirming):** `scripts/rerun_considerations_selector_on_stored_plans.py` used to read only the `.env` one folder above the worktree and refused to use the environment. It now reads `<worktree>/.env` by default. With `--find-dotenv` it reaches the parent `.env`, but falls back to the operating-system environment if that file lacks the key.
