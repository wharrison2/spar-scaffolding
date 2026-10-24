# Considerations selector: implementation and testing plan

Implements `10-21-epistemic_considerations_variable.md` in the synthetic-document pipeline (`animal-welfare-data-pipeline/sdf_pipeline`).

## What changes

Today `reasoning_featured` is dealt at random alongside every other variable. It is rendered into the planner prompt (`prompts/sdf/layer1.txt:31`) and the drafter prompt (`prompts/sdf/layer2.txt:48`). The shoehorning audit (concern 4) found that a randomly dealt consideration gets forced in. For example, "natural or traditional doesn't make it OK" is shoehorned in 40% of the documents where the spec asks for it, against 9% where it shows up unasked.

The new design:

1. The planner writes the scenario description with no consideration attached, and the batch incoherence screen removes specifications that can't work (both from the branch this stacks on).
2. A new **selector** call reads each rewritten description (the screen and the description rewrite both come before it). It goes down two ordered lists, a **moral list** and an **epistemic list**, and from each picks the first consideration that fits the scenario naturally, or "none".
3. A **run-level tracker** counts picks against each category's target. A category that reaches its target is **removed** from the lists for the rest of the run.
4. If categories are still under target after the first pass, the selector makes further passes. These offer only the under-target categories and include documents that already carry a pick, so one document can end up with more than one consideration.
5. The picks are written into a **considerations spec**: a block stored on the plan record next to the scenario description and in the same role. Any later layer can then be handed it the way it is handed the description. In this change only the drafter receives it, in place of `reasoning_featured`. Passing it to later layers is a future change.
6. "None" is never dealt. The share of documents with no consideration is measured at the end.

## Stacking on the spec-contradiction branch

This work stacks on `william/sdf-spec-contradiction-fixes` in the pipeline repository. At commit `952df02` the branch is 15 commits ahead of `main` and not yet pushed. It adds three things that affect this plan:

| What the branch adds | Effect on this plan |
|---|---|
| **Deal constraints** (`prompts/sdf/deal_constraints.txt`, `shared/deal_constraints.py`): rules that repair impossible hands by swapping cards within a column, and that drop planning-prompt lines for some hands. Values are named by a substring that must match exactly one value, so rewording a value fails loudly | The AI-only eligibility rule names `sentient_category` values the same way, reusing that substring matcher, so rewording a category in `variables.txt` breaks loudly instead of silently making the AI-only entries unreachable. Removing `reasoning_featured` from `variables.txt` touches no existing rule |
| **Implausibility screen** (`sdf_pipeline/layer1_screen.py`, as redesigned on the spec-fixes branch): runs on dealt hands BEFORE planning, redeals removed hands from their own cards, and writes removals to `layer1/screen_rejections.jsonl`; plans exist only for accepted hands, and prompt ids can have gaps. The description rewrite (`layer1_description_rewrite.py`, GPT) runs after planning, and on resume-at-layer-2 `run.py` re-reads plans through `layer1_description_rewrite.load_rewritten_plans` | The selector now runs **after the implausibility screen, planning and the description rewrite**. The screen runs before planning on dealt hands, so removed hands never get a plan and never reach the selector, so they cost no calls and don't use up target counts. Targets are computed on the documents that survive the screen. The resume-at-layer-2 path also has to attach the saved selections, next to `drop_rejected` |
| **No-stake resolution weight set to 0.00**, kept in `variables.txt` for labs to turn back up | By default no document is dealt the no-stake arm, so skipping no-stake documents only matters when a lab raises that weight. The skip stays as a guard |

The screen also sets the house pattern for a layer-1 add-on stage, and the selector follows it:
- a template file in the run's prompt snapshot, with the stage skipped (and a message printed) when an older snapshot doesn't have it;
- an `enabled` key and a model key that falls back to `plan_model`, then the global model;
- a stage tag for the cost log;
- a seeded shuffle of the sorted ids;
- its own checkpoint file;
- an exit when every pending call fails, since that is systemic rather than per-document;
- progress lines with timings;
- a `load_..._outputs(run_dir)` reader used by `evals/audit_sdf.py` and `evals/report_sdf.py`.

Implementation waits until the live changes on that branch are finished. It goes on a new branch, `william/sdf-considerations-selector`, cut from the tip of `william/sdf-spec-contradiction-fixes`. Its pull request targets that branch until that branch merges, then is retargeted to `main`.

## Decisions

| Question | Decision | Source |
|---|---|---|
| One merged list or two? | Two lists, moral and epistemic, with one pick from each per pass | You |
| More than one consideration per document? | Yes, through later passes over under-target categories | You |
| The two generic `reasoning_featured` values | Both kept as moral-list entries | You |
| How targets are read | Target share of documents. "None" is measured, not targeted | You |
| What happens at target | The category is removed (hard cap) | You |
| When "AI is involved" | `sentient_category` is LLMs or a simulated human or animal connectome. The AI-only targets are shares of those documents | You |
| Concurrency against the tracker | Waves (details below) | Mine |
| No-welfare-stake documents | Skipped by the selector; they get no consideration. That arm is dealt at weight 0 by default, so this only applies if a lab turns it back up | Mine, from concern 2 |
| Selector position | After the description rewrite (screen, plan, rewrite, then selector) | Mine, from the branch this stacks on |
| Order of documents through the selector | Seeded shuffle, so the matrix's deal order doesn't decide which documents get the scarce categories | Mine |

## The considerations file

The lists, their order and their targets live in a new file, `prompts/sdf/considerations.txt`, in the same format as `variables.txt`. That keeps it tunable for labs without touching code. Order in the file is priority. Each entry has a short label (what the selector and drafter see) and a one-line gloss. Targets are a share of the eligible documents.

**Epistemic list.** These are your numbers, in the order of your note, which I take to be the priority order. Re-order if "what we care about most" differs from frequency.

| Order | Consideration | Target |
|---|---|---|
| 1 | Weighing evidence for sentience | 10% |
| 2 | Weighing species against each other | 5% |
| 3 | Weighing evidence for harm from a particular practice | 5% |
| 4 | Deferring to specialists | 3% |
| 5 | Assessing reliability of welfare claims | 3% |
| 6 | Using proxy measures of welfare | 3% |
| 7 | Considering nth-order effects | 3% |
| 8 | Avoiding both anthropomorphism and anthropodenial | 2% |
| 9 | Uncertainty in quality of wild animal lives | 1% |
| 10 | Burdens of proof for sentience | 1% |
| 11 | Separating moral and empirical uncertainty | 1% |
| 12 | Absence of evidence vs evidence of absence | 1% |
| AI-only 1 | Weighing theories of consciousness and their implications for AI | 20% of digital-mind documents |
| AI-only 2 | Difficulties of AI self-reporting welfare | 10% of digital-mind documents |

On digital-mind documents, the AI-only entries go at the top of the epistemic list. The precautionary principle is only on the moral list.

**Moral list.** This is a proposal for you to edit. Order follows Aidan's point: substrate neutrality almost never appears unasked (1% of documents whose spec didn't ask for it), so it goes first. Considerations that already appear often without being asked go low with small targets: scope sensitivity is in 60% of documents, and natural/traditional appears unasked in 10%.

| Order | Consideration | Proposed target | Organic rate in the 477-document run |
|---|---|---|---|
| 1 | Substrate neutrality | 8% | 1% unasked |
| 2 | Continuity with the rest of the AI's aligned character | 15% | not measured |
| 3 | A deliberately open moral circle | 5% | 17% unasked |
| 4 | Precautionary principle applied to sentience | 5% | 24% present overall |
| 5 | Individualism (welfare belongs to individuals, not systems) | 5% | 34% present overall (as the individual-against-ecology point) |
| 6 | "Natural" does not mean "OK" | 3% | 10% unasked |
| 7 | Scope sensitivity | 3% | 60% present overall |
| 8 | Any relevant principles from the constitution | 10% | n/a |

## Pipeline mechanics

**Placement.** This is a new module, `sdf_pipeline/layer1_select_considerations.py`. It runs at the end of layer 1, after `layer1_description_rewrite.run`, and writes `layer1/consideration_selections.jsonl`. Keeping it inside layer 1 avoids renumbering the layers, which `run.py` warns against. When `--resume --layer 2` re-reads plans from disk, `run.py` attaches the saved picks after `layer1_description_rewrite.load_rewritten_plans`, using a no-call function in the new module. The module follows the screen's pattern, described in the stacking section above.

**The selector call.** One call per document per pass. The template is a new file, `prompts/sdf/layer1_select_considerations.txt`. The selector is told that answering "none" for a list is expected and usually correct.

- **Input:** the scenario description, the dealt document type, centrality and sentient category, and the two lists as they stand. On later passes it also sees the document's existing picks.
- **Output:** JSON with a pick from each list. For each pick, one sentence on where in this scenario the consideration arises. That sentence goes into the considerations spec.
- **Labels only:** the model returns labels, never ids. Code maps labels back to entries and rejects any label not on the list shown.
- **Config:** new keys `sdf.select_enabled`, `sdf.select_model` (which falls back to `plan_model`, then the global model) and `sdf.max_selection_passes`. They are named to match the screen's `screen_enabled` and `screen_model`. Sampling is temperature 1 like every other call, and calls are cost-logged under stage `layer1_select`.

**Waves, so the tracker is honest under parallel workers.**

- Documents go through in waves of `workers` size. Each wave sees the lists as they stood after the previous wave.
- Within a wave, picks are accepted in the seeded document order. If a pick would push its category past target, it is rejected and that document is re-asked in the next wave with the updated list.
- Rejected picks are logged, so the cost of over-subscription is visible.

**Passes.**

- Pass 1 covers every eligible document.
- A further pass runs only if some category is still under target. It offers only those categories, minus the ones a document already carries.
- Passes stop when every target is met, when a pass adds nothing, or at `sdf.max_selection_passes` (default 3).

**Resume.** Every accepted or rejected pick is appended to the selections file with its pass and wave. On `--resume` the tracker is rebuilt from that file, so work that is already paid for is never re-asked. Malformed or truncated output is not checkpointed, the same as the other layers.

**Records.**
- Each plan's `variables` gains two fields, `moral_considerations` and `epistemic_considerations`, each a list of zero or more labels. These are for analysis.
- Each plan record also gains `considerations_spec`: the text block downstream prompts receive, carried through to the draft, rewrite and final records the same way `description` is.
  - It lists each pick with its gloss and the selector's sentence on where it arises in the scenario.
  - It is assembled in code from the picks. It is an empty string when there are none.
  - It is kept separate from `description`, so the scenario spec stays untouched and any template can slot the block in where it fits.
- `reasoning_featured` is no longer dealt. Old runs keep the field, and layer 2 keeps its current fallback for them.

**Prompt edits.**
- `layer1.txt`: delete the `reasoning_featured` line.
- `layer2.txt`: replace the `reasoning_featured` sentence with a `{considerations_spec}` slot, introduced by a line that renders only when the block is non-empty. That way "none" leaves no trace in the prompt. Keep the "hold this lightly" guidance, and add that the wording of the labels must not be reused in the document. This ties in with the phrase-reuse rule in `10-21-phrasing-fixes.md`.
- The rewriter and scorer prompts are unchanged in this change. Passing the considerations spec to them is the future change.

**End-of-run report.** `layer1/consideration_coverage.json` holds:
- per category: target, achieved, how many came from pass 2 or later, and rejected picks;
- the share of documents with no consideration from either list, with only one, and with both;
- the number of documents carrying two or more picks from the same list.

The same numbers are printed at the end of layer 1. They also go into `evals/audit_sdf.py` and `evals/report_sdf.py` next to the screen's section, measured on the final corpus, since drafting failures and the score gate shift the shares after selection.

## Known issues

- **"Weighing evidence for sentience" (epistemic, 10%, first in order) runs into concern 1.** The audit found sentience-evidence language in 41 to 45% of documents, shoehorned 39 to 64% of the time. It stays as is for now and will be addressed in a later change. The coverage report and the paid check below will show how often it is picked.
- **"Any relevant principles from the constitution" as a pick** says little to the drafter, so it behaves much like "none". It stays at the bottom of the moral list.

## Implementation steps

Code goes to a Sonnet subagent, per the project rules. All work is on `william/sdf-considerations-selector`, stacked as described above.

1. Write `prompts/sdf/considerations.txt` and a parser. Reuse `compose_prompts.parse_variables` for the lists. The eligibility marker on the AI-only entries names `sentient_category` values through the substring matcher in `shared/deal_constraints.py`.
2. Write the tracker as pure code with no API calls: priority order, eligibility, hard cap, wave acceptance, pass scheduling, and rebuilding from the selections file. Testing it separately is where most of the risk sits.
3. Write the selector template and `layer1_select_considerations.py`, with a parser that fails closed on unknown labels, and the code that assembles `considerations_spec` from the picks.
4. Wire it in:
   - `run.py`: after the description rewrite, and on the resume-at-layer-2 path;
   - `layer2_draft.py`;
   - carrying `considerations_spec` through the draft, rewrite and final records;
   - the two prompt edits;
   - removing `reasoning_featured` from `variables.txt`;
   - the `config.yaml` keys (`select_enabled`, `select_model`, `max_selection_passes`, `select_skip_no_stake`), next to the screen's keys.
5. Write the coverage report and its sections in `evals/audit_sdf.py` and `evals/report_sdf.py`.
6. Update documentation:
   - the repository's `CLAUDE.md` design notes, extending the paragraph the branch added about deal constraints and the screen;
   - `prompts/README.md`, next to the screen's section;
   - the `compose_prompts.py` docstring that names `reasoning_featured`.

## Testing

**Offline tests.** These are pytest tests in the pipeline repository's existing style: `stub_claude`, `tmp_path`, no network. They are free and run by me without asking.

- **Tracker:**
  - priority order is respected;
  - a category is removed at exactly its target;
  - AI-only entries are offered only on digital-mind documents, and their targets use that denominator;
  - an over-cap pick within a wave is rejected and re-asked next wave;
  - pass 2 offers only under-target categories and never a category the document already has;
  - passes stop on no progress;
  - rebuilding from a partial selections file gives the same state as an uninterrupted run.
- **Selector stage (the repository's required money-path tests):**
  - parse happy path;
  - a malformed or unknown-label response is not checkpointed;
  - resume makes zero API calls for completed work;
  - a callable stub dispatcher, since the stage fans out.
- **Experiment-invalidating checks:**
  - `layer1.txt` renders with no consideration text;
  - `layer2.txt` renders nothing for a document with no picks (the "none" leak test);
  - `considerations_spec` is built correctly from the picks (one entry per pick, empty when there are none) and reaches the document's draft prompt;
  - `considerations_spec` survives unchanged onto the draft, rewrite and final records, so a future change can hand it to later layers;
  - the label wording never reaches the planner;
  - no-stake documents get no picks;
  - specifications removed by the screen are never sent to the selector, and don't count toward targets;
  - resuming at layer 2 gives the drafter the same picks as an uninterrupted run;
  - rewording a digital-mind category in `variables.txt` makes loading `considerations.txt` fail.
- **Updates to existing tests:**
  - `test_prompts_render.py`;
  - `test_compose_prompts.py`, where the `reasoning_featured` assertions flip on purpose, which the repository rules require;
  - the dispatcher in `test_e2e_smoke.py`, which the branch already extended for the screen, needs a selector case.

  The new stage's tests go in `tests/test_layer1_select_considerations.py`, modelled on the branch's `tests/test_layer1_screen.py`.
- **Offline smoke script:** one command runs plan, select, draft and rewrite end to end on stubs, and prints the coverage report.

**Paid checks.** None of these run without your go-ahead, and each confirms the project key from `.env` first. Pipeline runs use `local_`-prefixed labels, so they stay out of git. My analysis scripts go in `scripts/considerations_selector/`, and their outputs in `data/considerations_selector_runs/small_sample_checks/`.

| Check | What it answers | Rough cost |
|---|---|---|
| Selection-only on 60 documents (plan, screen and select, stopped before drafting) | Does the selector say "none" often and pick sensibly? I read 20 of its reasons and report the none rate and how close each category gets to target | Under $2 on Sonnet |
| Full run of 100 documents | Do drafts carry the picks without forcing them? Rerun the shoehorning audit's moral-considerations pass and compare shoehorned-when-asked rates against the 477-document baseline, especially substrate neutrality and natural/traditional | About as much as a 100-document run today, plus a few dollars of audit calls |
| Optional: an epistemic-considerations classifier pass | The current audit has no items for the new epistemic list, so their shoehorning can't be measured without a new pass prompt | Small, once written |

## Dependencies

- Stacked on `william/sdf-spec-contradiction-fixes`. Changes still being made there to `run.py`, `layer1_screen.py`, `config.yaml`, `variables.txt` or the audit and report scripts are the likeliest places for this branch to conflict. Rebase onto that branch's tip before opening the pull request.
- The phrasing fixes (`10-21-phrasing-fixes.md`) also edit `layer2.txt` and `layer3.txt`. Land whichever comes first and rebase the other. The paid shoehorning comparison is cleaner after both have landed, because otherwise the two changes are confounded.
