Title: Add plausibility, shoehorning and variable persistence audits

## Summary

Adds three model-judged audits for SDF corpora under `evals/`: a plausibility audit, a shoehorning audit and a variable persistence check. Each sends documents to a judge model with fixed instructions and a strict answer schema. Runs are resumable and can be costed before they are paid for. The plausibility and shoehorning audits have offline post-processors that turn the raw judgments into tables.

## What changes

| Component | What it does |
|---|---|
| `evals/plausibility_audit_sdf.py` | Judge reads one document alone and scores ten plausibility axes (scale and figures, real counterparts, venue, speaker fit, friction, provenance, technology, setting coherence, motive, genre-breaking disclosure). Each axis gets a 1-5 score with up to three quoted problems, each with a severity. It also lists problems outside the ten axes and gives an overall score. |
| `evals/shoehorning_audit_sdf.py` | Three passes per document (epistemic notes, field distinctions, moral considerations) look for a fixed list of niche ethical considerations. For each occurrence the judge quotes it, then labels it organic, staged or unmotivated, and woven in or aside. |
| `evals/variable_persistence_sdf.py` | For each dealt variable value and assigned consideration, a judge gives brief reasoning and a yes/no on whether it persisted in the text. The judge is not shown the document's description. |
| `evals/plausibility_audit_postprocess_sdf.py` | Offline. Per-document and per-problem tables (CSV and JSONL), per-axis summaries, quotes checked against the document, language and variables joined in. Failed documents leave empty cells, and inapplicable axes are left out of means. |
| `evals/shoehorning_audit_postprocess_sdf.py` | Offline. One row per occurrence, a per-item summary, quote checks, and whether each quote was already in the pre-rewrite draft (or `draft_unavailable` if no drafts are found). |
| `evals/model_judged_audit_support.py` | Shared pieces: run folders, resume checks, quote verification, CSV writing, stratified sampling, structured-output request with one parse retry. |
| `prompts/tools/` | Judge instructions for each audit: plausibility, the shoehorning shared header plus three pass files, and the variable persistence prompt and variable meanings. |
| `shared/openai_api.py`, `shared/api.py` | Small additions the audits need: optional prompt cache key, per-call usage returned to the caller, a price entry for `gpt-5.6-luna`, and `init_cost_log_only` so a script that only calls OpenAI can use the cost log without Anthropic configuration. |
| `evals/README.md` | New rows in the audit table and a "Model-judged audits" section. |
| `tests/` | Offline tests for the three audits and the shared support module, plus three fixture files holding the expected shoehorning schemas and instructions. |

## Why

Other checks in `evals/` look at how a corpus is written. These ask whether the documents depict a believable world (plausibility), whether niche ethical considerations are inserted more often, or less naturally, than real writing would (shoehorning), and whether the variables a document was planned from survive into the finished text (variable persistence). Quotes are checked against the document so each finding can be audited by hand.

## How to run

Both commands call a paid API. Use `--estimate-only` first: it judges a small stratified sample (`--estimate-sample-size`) and writes an extrapolated cost for the whole corpus without a full run.

```bash
python evals/plausibility_audit_sdf.py --input outputs/sdf/latest --run-name plausibility_audit_medium_effort --estimate-only   # paid (sample only)
python evals/variable_persistence_sdf.py --input outputs/sdf/latest --dry-run   # no calls, no key: prints prompt, call count, cost estimate
```

Drop the flag for a full run; continue an interrupted one with `--resume-folder <run folder>` (`--resume <folder>` for variable persistence). A resume refuses a changed prompt, model, reasoning effort or corpus; the variable persistence check records its configuration for this and refuses a folder that has none. Then run the matching `*_postprocess_sdf.py <run folder>` offline. `--document-ids` and `--document-limit` restrict a run, and such partial runs go to a `small_sample_checks` subfolder. Plausibility and shoehorning default to `gpt-5.6-luna` at medium reasoning effort with `OPENAI_API_KEY` from the repo-root `.env`; variable persistence uses the model set in `config.yaml` (`sdf.variable_persistence_model`, falling back to the score, plan and global models).

## How to review

Start with the "Model-judged audits" section of `evals/README.md`, then `evals/model_judged_audit_support.py`, which both new audits share. Each audit script is otherwise self-contained, and the post-processors read only run folders. The judge instructions in `prompts/tools/` are the part most worth reading closely, since they define what is measured. Resuming refuses a changed prompt file, so edits to them are deliberate. The `shared/` changes are small and additive.

## Testing

The offline test suite passes (1612 tests, 118 more than the base), in the maintainers' environment and in a clean install from `requirements.txt`. It blocks network access, so it makes no API calls. The audit tests stub the one network call and use a synthetic three-document corpus.

## Stack

PR 4 of 4, stacked on #180 (model calls and API keys). Merge after #180; GitHub retargets this PR when its base branch is deleted on merge.
