You're implementing the phrasing fixes for the SDF pipeline in `/Users/williamharrison/Desktop/emergent_alignment/animal-welfare-data-pipeline`. Read `/Users/williamharrison/Desktop/emergent_alignment/.claude/CLAUDE.md`, the skills it lists, and the pipeline repo's own `CLAUDE.md` before starting, and follow them.

## Where to work

- Create a new branch, `william/sdf-phrasing-fixes`, from the tip of `william/sdf-considerations-selector` (currently `311794c`), in a new git worktree of your own, for example `/Users/williamharrison/Desktop/emergent_alignment/animal-welfare-data-pipeline-phrasing-fixes`.
- Do not edit the two existing checkouts. `animal-welfare-data-pipeline` is on `william/sdf-spec-contradiction-fixes`, and `animal-welfare-data-pipeline-considerations-selector` is on `william/sdf-considerations-selector`. A paid end-to-end smoke run may still be running from the selector checkout.
- The branches stack in this order, and so will their pull requests: `william/sdf-spec-contradiction-fixes` → `william/sdf-considerations-selector` → `william/sdf-phrasing-fixes`. Nothing is pushed yet. Commit on your branch only, never on `main`, and don't push unless asked. If a later fix lands on the selector branch, rebase onto it.

## What you're implementing

1. `plans/10-21-spec-fixes/10-21-phrasing-fixes.md` (in `emergent_alignment`): every item there.
2. The stage 6–8 changes from `plans/10-21-spec-fixes/implausibility-screen-and-plausibility-repair-design_2026-10-22_01-43.md`, section "Stages 6–8":
   - the drafter (`prompts/sdf/layer2.txt`) receives the dealt variables;
   - the document rewrite (`layer3.txt`) gains the same plausibility checks as the description rewrite (`prompts/sdf/layer1_description_rewrite.txt` is the model), plus an exception to "keep the same named characters and institutions" for any institution, practice or figure that fails a plausibility check, replaced with a plausible one in the same role;
   - the judge (`layer4.txt`) scores spec conformance against the rewritten description and the dealt variables.
3. `plans/10-21-spec-fixes/10-21-additional-fix-items.md`: the plausibility items at the rewrite step. They overlap with item 2, so implement them once.

Read the plausibility audit for context: `plans/10-06-plausibility-axis/10-06-plausibility-audit-most-common-problems-report.md`.

## The pipeline as it stands on your base branch

Stage order: deal and repair → implausibility screen on dealt hands, batches of 10, with redeal rounds → planning (`layer1.txt`) → description rewrite (GPT 6.1 Sol) → considerations selector → draft (`layer2.txt`) → document rewrite (`layer3.txt`) → judge (`layer4.txt`).

Some phrasing-fixes items are already partly done. Don't redo them:
- **Dealt variables into every stage:** the screen and the description rewrite already get the rendered dealt variables, through `sdf_pipeline.compose_prompts.render_variable_lines(record, ...)`. Reuse that function for layers 2–4 rather than writing a second renderer. Only the drafter, the document rewrite and the judge are left.
- **`reasoning_featured`** is no longer dealt. The considerations selector picks considerations from `prompts/sdf/considerations.txt` after the description rewrite, and they reach the drafter in its place. The "considerations wording reaching the drafter" item in the phrasing note is still open.
- **Naming the AI provider:** `layer1.txt` already says "when the AI model is named, the company that provides it may be named too". `layer2.txt` line 28 still tells the drafter to attribute actions only to fictional entities. Give it the same exception.
- **No-stake resolution:** it has weight 0, so its framing items only matter if a lab turns that weight back up. Keep them guarded, but they're lower priority.

## Things that pin value or prompt text

Every variable value has a stable `[id]` prefix in `prompts/sdf/variables.txt`. Deal rules (`prompts/sdf/deal_constraints.txt`) and the considerations catalog name values by ID only, so you can reword value text safely. These still match on text, and a rewording must update them:
- `sdf_pipeline/compose_prompts.py`, around lines 194–208: the table of example beings is keyed by the full value text. Rewording any digital-mind category (simulated human connectomes, simulated animal connectomes, LLMs) breaks it. Key it by value ID instead.
- `tests/test_deal_constraints.py`: `NO_STAKE_MARKER = "NO real welfare stake"`.
- `tests/test_layer1_select_considerations.py`: `DIGITAL_CATEGORY = "LLMs whose moral state is uncertain"`.
- `tests/test_considerations_tracker.py`: the real-catalog test and the rewording test, with literal value strings.
- `prompts/sdf/layer1_screen.txt` mentions "simulated connectomes or language models" in prose.
- `tests/test_prompts_render.py` renders every template, so update it for any new placeholders.

## Findings from 2026-10-22 that bear on your work

- **Claude is much more lenient than GPT on plausibility.** Given the same one-description plausibility prompt on 97 descriptions, Claude Sonnet 5 flagged 1 of 92 and Claude Opus 5 flagged 0 of 95. Neither caught any of the 14 implausible premises GPT 6.1 Sol flagged; both noticed them and accepted them as "not clearly impossible" or allowed "within the fictional world". For that reason the description rewrite runs on GPT (see the note in `config.yaml`). The document rewrite (`layer3.txt`) still runs on Claude, so plausibility checks added there may be weak. Phrase them against that failure: the standard is "highly implausible in the real world today, not strictly impossible", and being "possible in principle or acceptable within a fictional world is not enough to pass". Flag in your report whether layer 3 should move to a GPT model; don't switch it yourself.
- **GPT 6.1 Sol rewrote 91 of 93 descriptions,** mostly to spell out dealt values the description left unstated. That added forced phrases such as "the assistant, acting in line with Claude's constitution", which is exactly what the shoehorning audit warns about. When you pass dealt variables to the drafter, the document rewrite and the judge, tell them the variables say what the document must not contradict. They are not a checklist of phrases to insert, and the judge shouldn't penalise a document for leaving a value implicit.
- **The API's safety filter refuses some research-and-testing material.** It hit about 3% of planning calls, all in that domain. Don't treat a refusal as a bug in your prompts.

## Rules

- No paid model or API calls, including one-item tests, without William's explicit go-ahead and a cost estimate first. Offline tests use the `stub_claude` dispatcher and the `stub_openai_responses` fixture in `tests/conftest.py`.
- Never read, cat, grep or print any `.env` file.
- Constraint rules only rule combinations out. Never describe them, in code, comments or docs, as lists of allowed values.
- Model-facing text goes in `.txt` files. Use descriptive names without abbreviations, and match the surrounding code style.
- Keep the full suite green and fast. It's currently around 1,330 tests.
- When you report back, list your commits, which phrasing-note items are done, which you changed in scope and why, the decisions you made yourself, and anything left open.
