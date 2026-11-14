# Writeup: over-representation audit, rejected-values feature, and category–domain drift

Session of 2026-11-05 to 2026-11-06, working from `plans/10-21-spec-fixes/11-05-show-rejected-variables.md`. No pipeline or generation API runs were made. The only model calls were Sonnet 5.5 subagents for analysis code, pair judgments and implementation.

## What was done

### 1. Over-represented considerations (results in the plan's report)

- A Sonnet subagent wrote `scripts/shoehorning_audit/compare_audited_consideration_rates_with_assigned_considerations.py`. It compares how often each of the 25 shoehorning-audit items appears in documents *not* dealt that consideration, against the item's current target.
- Inputs were two shoehorning audits: the original corpus (477 documents, one dealt consideration each) and the new-selector run (23 documents).
- Outputs: `data/shoehorning_audit_runs/audited_consideration_rates_compared_with_assigned_considerations_2026-11-05_23-42/` (tables 1 to 5 and a printed summary).
- I checked the subagent's figures by hand. Two of them changed: the item 17 staged-or-unmotivated share went from 62% to 61% (137 of 223), and the item 07 share was corrected to 12%.
- Main results:
  - Scope sensitivity appears in 57% of the original documents that were not asked for it, against a 3% target.
  - "Welfare option also serves the user" (item 25) appears in about 60% and has no catalog entry.
  - The stock sentience-calibration passage (item 17) appears in 45%, and is mostly staged or unmotivated.
  - Substrate neutrality almost never appears unless it is asked for.

### 2. Axes clarified by showing values not dealt (results in the plan's report)

- I counted persistence-check failures for each dealt value, pooled over three corpora (71 documents). The judgment files are in `data/spec_contradiction_screen_runs/small_sample_checks/generation_and_checks_from_one_objective_considerations_2026-10-24_03-02/variable_persistence_*/`.
- Drift always goes toward a neighbouring value:
  - centrality toward central
  - extreme decision scales toward the middle
  - AI experts toward educated non-experts
- On the resolution axis, 9 of the 11 drifts land on `creative_solution` or `user_makes_modest_sacrifice`, rather than spreading across the other values.
- Failures on different axes co-occur at about the rate chance predicts.

### 3. Sentient category and domain mismatches (separate report)

- Report: `plans/10-21-spec-fixes/sentient-category-drift-toward-dealt-domain-report_2026-11-06_00-16.md`.
- Every category drift replaced the dealt beings with the dealt domain's usual animal.
- My first classification of pairs was circular, because I had already seen which pairs drifted. It was replaced by Sonnet 5.5 judging all 172 category–domain pairs without knowing the outcomes. Output: `data/spec_contradiction_screen_runs/sentient_category_and_domain_pairing_judgments_by_sonnet_2026-11-05_23-58/`.
- Labels: 75 fit, 96 stretch, 1 impossible (simulated animal connectomes with food service and retail). "Pets in research and testing" counts as a stretch, not impossible.
- Drift occurred only in medium- and high-risk stretch pairs. These are about 38% of dealt hands.

### 4. `sdf.show_rejected_values` feature (pipeline code)

- Worktree: `animal-welfare-data-pipeline-show-rejected-variable-values/`, branch `william/sdf-show-rejected-variable-values`, branched from origin/main `220fcc9`. Not pushed. The worktree path is in the project `.gitignore` (uncommitted).
- Commits:
  - `ad0f6b0`: the feature.
  - `9aa7ebd`: nested-bullet formatting, with a separate header for the planning stages and one for the drafting stages.
- Config, off by default:
  - `axes: [centrality, decision_scale, resolution, tech_savvy]`
  - `stages: [plan, description_rewrite, draft, rewrite]`
- The block fills a `{rejected_value_lines}` slot in `layer1.txt`, `layer1_description_rewrite.txt`, `layer2.txt`, `layer3.txt`, `layer3a_critique.txt` and `layer3b_revise.txt`. When the feature is off, the prompts are byte-identical to before.
- Values shown come from a one-directional neighbour map, `prompts/sdf/rejected_value_neighbours.txt`: each dealt value lists only the values it has been seen drifting toward. The map replaced an earlier version that listed every other value on the axis, which made the block up to 304 words.
- The shown values exclude values with weight 0 and values ruled out by deal rules. Axes dropped by an omit rule show nothing.
- The new main-branch resolution value `delegated_task_ai_notices_welfare_stake` has no neighbour entry, so it shows nothing.
- Each record stores `rejected_values_shown`.
- Tests: `tests/test_rejected_value_lines.py`, plus updates to the existing template and compose tests. 1951 passed.
- Rendered examples, made without API calls by `scripts/render_rejected_value_lines_for_stored_run_without_api_calls.py`, for matrix_000033, 000043 and 000048: `data/spec_contradiction_screen_runs/small_sample_checks/rejected_value_lines_render_check_2026-11-06_00-58/`.
- Block length over 3000 random deals:

  | Part of the block | Words |
  |---|---|
  | Whole block | 42 to 198 |
  | Resolution line | 70 to 105 |
  | No block at all | 270 of 3000 deals (about 9%) |

### 5. Cost estimate for the paired test

- File: `data/spec_contradiction_screen_runs/small_sample_checks/cost_estimate_for_show_rejected_values_paired_test_2026-11-06_00-55.md`.
- Each hand runs with the feature off and on, with the same seeds, through planning to rewrite, followed by the persistence check. Opus 5.5 throughout.
- Cost is about $0.71 per document per arm. The split rewrite (about $0.40) is the least certain figure.
- Options:
  - a 20-hand pilot, about $29
  - a 60-hand main test with the centrality deal enriched, about $86

## Interpretation

- Raising a consideration's target only sets a floor. The over-represented considerations already appear unasked far above their targets. That is a separate problem from persistence, and the rejected-values block does not address it.
- Drift is one-directional and goes toward a default document. That is why a short neighbour map can carry the contrast without listing whole axes.
- Several drifts already appear in the planned descriptions, which is why the planning and description-rewrite stages also get the block.
- Category drift comes from a pull between two axes, the sentient category and the domain. Showing values not dealt on one axis will not fix it. It needs its own fix, and that fix should come before the paired test so that it does not add noise to both arms.

## Not yet done

- Fix the category–domain drift. Candidates:
  - a planning instruction to keep the beings as dealt
  - per-pair hints from Sonnet's premises
  - a deal rule for simulated animal connectomes with food service and retail
- Clarify whether "farmed fishes" includes ornamental fish.
- Fix the resolution wording that says "animal" in digital-mind documents.
- Add a neighbour entry for `delegated_task_ai_notices_welfare_stake`.
- Confirm that the persistence check accepts the new axis `failure_mode_avoided`, and set up the enriched deal.
- Run the paired test. It needs your go-ahead.
- Push the branch and open a pull request.
- The reformat subagent was resumed from its transcript and was still running when this was written. Recheck the branch and tests after it reports.
