Title: Load API keys from one place and record each run's model calls

## Summary

API keys are now found by a single loader, `shared/api_keys.py`, which by default reads only the repo-root `.env` and never the shell environment. A new `--find-dotenv` flag restores the broader search for servers that supply keys as environment variables. Separately, each run's model calls are now saved in full beside its own cost log, and `claude-opus-5-5` is supported.

## What changes

| Area | Change |
|---|---|
| `shared/api_keys.py` (new) | One loader for all keys, with two modes (default and `--find-dotenv`), a clear error when a key is missing, and a once-per-key message naming the source |
| `shared/api.py`, `shared/openai_api.py`, `shared/embeddings.py` | Use the loader instead of reading the environment or calling `load_dotenv()` at import. OpenAI and Gemini keys now follow the same rules as the Anthropic key |
| Command-line entry points | `--find-dotenv` added to both pipeline runners, the preference pipeline, the DAD runner, the evals that call models (`audit_dad`, `audit_sdf`, `compliance_sdf`, `diversity`, `score_sdf`), `evals/publish_hf.py` and `scripts/rerun_considerations_selector_on_stored_plans.py`. The DAD runner passes the flag on to the evals it launches |
| `shared/api.py`, `shared/openai_api.py` | `model_calls.jsonl` and `system_prompts/` written beside a run's own cost log (see Why); a run that does not give its own cost log path records none. Failed calls get a record too |
| `shared/api.py` | `claude-opus-5-5` added: priced at $4/$20 per million tokens, cache reads billed at 0.05x of input (others 0.10x). It cannot run with thinking off, so it gets adaptive thinking at low effort; models that need adaptive thinking now request a thinking summary so it is recorded |
| `.gitignore` | Ignores `model_calls.jsonl` and `system_prompts/` wherever a run writes them; `cost_log.jsonl` stays tracked |
| `tests/` | New `tests/test_api_keys.py`; `tests/test_api.py` extended; `tests/conftest.py` gives every test an empty repo root so no real `.env` is read |
| Docs | `README.md` (Authentication, Testing), `CLAUDE.md`, `config.yaml` comment updated to match |

## Why

- Key lookup was inconsistent. `shared/api.py` called `load_dotenv()` at import, so a key in the shell environment silently beat the one in `.env`. The OpenAI path ignored `.env` entirely, and the rerun script hard-coded a `.env` one folder above the repo. Which key a run used was hard to tell.
- A run left no full record of what was sent and received. The cost log holds token counts only, so a surprising output could not be traced to its exact prompt and raw response.
- `claude-opus-5-5` rejects `thinking: disabled`, so it could not be used for comparison runs.

## Behaviour change to check

- Default (no flag): a key is read only from `<repo root>/.env`. The repo root is found from the location of `shared/api_keys.py`, so parent folders and the current directory are never searched. The shell environment is never used, even if the key is set there. That includes `HF_TOKEN`: the default mode ignores it, and `HUGGING_FACE_HUB_TOKEN`, in the environment unless the `.env` supplies `HF_TOKEN`.
- `--find-dotenv`: the nearest `.env` searching upward from the repo root is used. If it does not set the key, the environment is used. A non-empty value in the `.env` wins over the environment, which reverses the old import-time behaviour.
- No key found: the command stops with `MissingApiKeyError` (a `KeyError` and a `RuntimeError`, so existing handlers still catch it). The message names the `.env` path and the key, and in default mode suggests `--find-dotenv`. For optional keys (`HF_TOKEN`, embedding providers) a missing key returns nothing instead of raising. `--dry-run` publishing skips the lookup.
- Printed: which file or the environment each key came from, once per key per process. The key value is never printed.
- The `claude_code` backend still needs no key. Anyone who relied on a shell-exported key now needs `--find-dotenv` or a line in `.env`.

## How to review

Start with `shared/api_keys.py` and `tests/test_api_keys.py`; that is the judgement part (the mode rules and error wording). The entry-point changes are mechanical: the same two lines (add the flag, apply it) in each file. In `shared/api.py`, read `_log_model_call` and the thinking-configuration helpers; the `call_claude` and `call_openai` edits only add logging calls around existing code.

## Testing

The offline test suite passes (1494 tests, 28 more than the base). It blocks network access, so it makes no API calls and uses fake keys.

## Stack

PR 3 of 4, stacked on #179 (generation phrasing fixes). Merge after #179; GitHub retargets this PR when its base branch is deleted on merge.
