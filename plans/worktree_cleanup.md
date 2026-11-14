Worktree: animal-welfare-data-pipeline-log-probability-persona-check
What the work is: Uncommitted follow-up to pull request #182. It lets the log-probability check score the difficult-advice answers with their real user messages, plus options to subset by rewording, skip
  variables and handle runs without controls. It looks finished and its offline tests pass. Input preparation ran on November 1, then the work stopped.
Equivalent on main?: No. Nothing like it is on main. The persona-vector scripts in emergent_alignment score the same dataset by a different method.
Verdict: Keep, unless you've dropped the log-probability approach for the difficult-advice answers.
────────────────────────────────────────
Worktree: animal-welfare-data-pipeline-show-rejected-variable-values
What the work is: 4 unpushed commits. They show the SDF stages which values weren't dealt (behind a config flag that's off by default) and list over-represented considerations not to raise unless
  assigned. They also add a paired experiment driver, two paired-arm analyses and two experiment result folders.
Equivalent on main?: No. The considerations selector from pull request #178 is related but works differently.
Verdict: Losing it would lose real work, but the experiments (32 documents per arm) found no significant effect. The considerations list cut scope-sensitivity mentions from 58% to 35% but raised others.
  It's 22 commits behind main and likely to conflict. Your call.
────────────────────────────────────────
Worktree: animal-welfare-data-pipeline/.claude/worktrees/agent-a80ef6175a6586ade
What the work is: 5 unpushed cleanup commits: dead config keys and wrappers removed, a shared JSON-parsing helper, test tidying, and trimmed documentation.
Equivalent on main?: No, though main's own code-quality findings file recommends some of these changes.
Verdict: Small and harmless, and it merges onto current main with no conflicts. Easy to open as a pull request, or safe to drop.
────────────────────────────────────────
Worktree: animal-welfare-data-pipeline-emergent-misalignment-prompts
What the work is: The severe-implicit prompt run on November 3 (61 scenarios, $7.83). This is the source of the 50 prompts the baseline screen uses.
Equivalent on main?: Almost entirely saved. I confirmed the step 1 outputs, cost log, run manifest and variables file are identical to the copy in
  emergent_alignment/data/emergent_misalignment_runs/severe_implicit_welfare_prompt_generation_fifty_prompts_2026-11-03_00-54/. The input prompts match git at commit 3acbbc1.
Verdict: Not saved anywhere: model_calls.jsonl (1.4 MB, the raw log of 276 calls) and 4 system prompt files. Copy those into the saved folder, then remove the worktree.

On the emergent-misalignment worktree, two more caveats:
- Registry numbers: its id_registry.json change reserves prompt numbers P-1715 to P-1775. Discarding it means a future run could reuse those numbers. The prompts themselves keep their numbers in the saved 50-prompt file.
- The saved copy is only on this disk. emergent_alignment/data/ is excluded from git.

On the main checkout, its uncommitted run (2026-11-03_00-55_...) is an abandoned partial: 31 of 58 scenarios, already recorded as abandoned in that saved folder. Its registry change can't be combined with the emergent-misalignment worktree's, because the two reuse the same scenario numbers for different scenarios. I'd delete the run, revert the registry change and switch the checkout to an updated main.