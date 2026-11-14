Title: Stop SDF documents echoing prompt wording

## Summary

Generated SDF documents were repeating stock phrases lifted from the prompts, and some carried content written to order (recited sentience evidence, sacrifices that pay off, dramatised cases inside reference genres). This PR rewords the planning, drafting, rewrite and judging prompts, passes the dealt variables to every later stage, and adds an offline measurement of verbatim phrase reuse. Its output goes to a git-ignored folder, so run measurements are never committed.

## What changes

| Area | Change |
|---|---|
| Planning prompt and `variables.txt` | Neutral welfare wording ("the beings involved" rather than "animals"); a new `reword` rule type in `deal_constraints.txt` swaps planning lines for hands with no scenario or no welfare stake; the dilemma and failure-mode lines are also dropped for no-stake hands |
| `prompts/sdf/layer2.txt` (draft) | Receives the dealt variables; no-reuse-of-phrasing rule; sentience evidence only where a character would raise it; the providing company may be named when the AI is; a skeptic stays skeptical unless the dealt resolution says otherwise |
| `prompts/sdf/layer3.txt` (rewrite) | Receives the dealt variables; new check 10 (plausibility of organisations, practices, simulated minds, what the AI knows, figures) and check 11 (forced content); calibration check no longer prompts adding evidence statements |
| `prompts/sdf/layer4.txt` (judge) | Receives the dealt variables; an unstated dealt choice is not drift, only a contradiction is |
| `prompts/sdf/preamble.txt` | Drops the win-win and tone paragraphs and the "no welfare stake" minority; varies the sentience calibration wording; adds a no-reuse rule |
| Description rewrite | New fourth check for forced content (new `forced_content` key in the answer schema); no copying of non-generic phrasing; check 2 flags only contradicted or swapped choices |
| `sdf_pipeline/`, `shared/` | `deal_constraints.py` parses and applies wording files; `compose_prompts.py` fills a `{variable_lines}` slot in later stages and keys the example-species table by value ID |
| `evals/measure_prompt_phrase_reuse.py` | Counts documents containing each phrase in `evals/distinctive_prompt_phrases.txt` and each considerations label and gloss; output written under `outputs/sdf/phrase_reuse_measurements/` (git-ignored); phrases match whole words |

## Why

- **Stock phrases recur.** The same turns of phrase ("realistic possibility", "welfare stake", "win-win") appeared across documents because the prompts spelled them out and documents echoed them. The measurement reports, per phrase, how many documents of a run contain it. Prompts now say not to reuse their wording and carry less quotable wording to begin with.
- **Forced content.** Documents recited how strong the evidence for sentience is, built welfare reasoning into passing mentions, turned a user's dealt sacrifice into a benefit, and wrapped reference genres (encyclopedia, FAQ) around a named person's story. Checks in the description rewrite, document rewrite and judge now target each of these.
- **Later stages did not see the dealt choices.** The drafter, rewriter and judge only had the description, so they could not tell a choice left implicit from one contradicted, and sometimes "fixed" drift that was not there. They now see the same variable lines as the planner, including omitted and reworded lines.
- **Planners invented scenarios and stakes.** Hands whose resolution says "no specific scenario" or "no welfare stake" still got prompts built around a scenario and a welfare tradeoff. The new reword rule and extra omissions rule out that framing for those hands.
- **Implausible details.** The new plausibility check asks for real organisations with real powers, approvals named by role, and no simulated minds presented as current products.

## How to review

- Start with `shared/deal_constraints.py` and `prompts/sdf/layer1_wording_for_*.txt` (the only new mechanism), then `sdf_pipeline/compose_prompts.py`.
- Mechanical: `variables.txt` neutral rewording, the species-table key rename, and the layer 2-4 plumbing for `{variable_lines}`.
- Judgement: the prompt text in `layer2.txt`, `layer3.txt` (checks 10 and 11), `layer4.txt`, `preamble.txt` and the description rewrite. Whether each wording does what it intends can only be seen in generated documents.
- Existing runs: a run whose saved templates lack the `{variable_lines}` slot renders nothing there, so `--resume` still works.

## Testing

The offline test suite passes (1466 tests, 18 more than the base) and blocks network access, so it makes no API calls. New tests cover wording files, rule rendering, the later-stage variable lines and the measurement script.

## Stack

PR 2 of 4, stacked on #178 (planning spec fixes and considerations selector). Merge after #178; GitHub retargets this PR to `main` when #178's branch is deleted on merge.
