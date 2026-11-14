# Writing code

 - All files your scripts generate must be named descriptively with their date and time at the end of the file name
 - Where code is modular and reusable, split it out. Where it is specific, put it in a folder for the project you are currently working on
 - Use descriptive, human-legible variable names and patterns
 - Write code that can be resumed from natural checkpoints whenever possible 
 - Reuse code from this project when reasonable. Let me know when copying from somewhere that isn't this project, and make sure that the addition is noted in references.md
 - Save data from a given run into its own specific, descriptively named folder. Subfolders for outputs that aren't from the same run/process
 - Have Sonnet 5.5 subagents write most code. Only use Opus subagents if I instruct you to, or Haiku subagents if the task is menial, requires no judgement, experimentally unimportant, or will not produce code that will be frequently reused
 - Do not abbreviate names (runs, directories, files, pods, logs, monitors) or terms in messages; write them out in full
 - Always use descriptive, legible file names: someone who has never seen the file should be able to tell what it holds from its name alone


## Considerations for using AI APIs
 - Do not run scripts that use these without my explicit go-ahead. This includes small tests
 - Confirm that you are using a key specific to this project from the `.env` before running code. Never use a key from my OS environment
 - Further rules for these scripts are in the `scripts-that-call-ai-apis` skill

# Where the emergent misalignment prompt corpus contributor kit lives

When I mention the contributor kit, contributor shares, or prompt generation for emergent misalignment, this is the code:
 - Source of truth (edit here): `scripts/emergent_misalignment/contributor_kit/` holds the kit's own files (`run_my_shares.py`, `kit_README_for_contributors.md` → README.md, `kit_CLAUDE_for_contributors_claude_code_session.md` → CLAUDE.md, requirements, gitignore). The kit also copies, unchanged, the batch script, judges and helpers listed in `SCRIPT_FILE_NAMES` of `scripts/emergent_misalignment/build_emergent_misalignment_contributor_kit.py`, plus step 1 code and prompts from `strictly_implicit_difficult_advice_pipeline_copy/`
 - Builder: `scripts/emergent_misalignment/build_emergent_misalignment_contributor_kit.py` (`--make-commit` commits in the kit repository); tests in `scripts/emergent_misalignment/test_contributor_shares_reserved_specs_merge_and_kit.py`
 - Built kit (generated output, its own git repository): `~/Desktop/emergent_misalignment_prompt_corpus_contributor_kit/`, remote https://github.com/wharrison2/emergent-misalignment-prompt-corpus-contributor-kit (public). Do not edit it by hand: change the source above, rebuild (the builder never pushes), then from the kit folder `git add -A`, `git commit` with a message describing the change, and `git push origin main`. Contributors pull `main`, so kit commits go directly to the kit repository's `main`: see "Git repositories, branches and worktrees" below
 - Shares: split by `scripts/emergent_misalignment/split_remaining_specs_into_contributor_shares.py` into `data/emergent_misalignment_contributor_shares_2026-11-08_15-35/`; returned zips go in `data/emergent_misalignment_contributor_returned_zips/`
 - Paste-in prompts for contributors: `plans/11-03-emergent-misalignment/contributor*paste-in*prompt*.md` and `contributor_kit_paste_in_prompts_*.md`

# Skills

Load the matching skill from `.claude/skills/` before starting these tasks:
 - `scripts-that-call-ai-apis`: writing, editing or running code that calls AI APIs or runs on a paid pod
 - `testing-and-smoke-tests`: writing tests, smoke tests, validation runs or small-sample checks
 - `writing-results`: writing writeups, reports or summaries
 - `pod_sessions`: launching, monitoring or shutting down paid GPU pods, or writing code that does
 - `log-probability-persona-check`: planning, running, analyzing or writing up a log-probability persona check on corpus documents

# Agent additions

If one of these instructions conflicts with an instruction in the sections above or in a skill, tell me and defer to the other instruction

 - When an edit request has a small and a large reading, make the smallest change that satisfies it, or ask
 - Which git repository you are in decides how you commit. See "Git repositories, branches and worktrees" below
 - Use shared code when it's available, and write code that can be shared when reasonable. Before writing or delegating new code, search `scripts/shared/` and the rest of the project for code that already does the job, and tell subagents to do the same. If you are about to write a second near-copy of something (for example, a second judge-agreement analysis), turn it into one shared module instead
 - Write code that is simple and concise, and tell subagents to do the same. Docstrings are short and describe the code as it is: never refer to an earlier state of the code or pipeline (no "formerly", "used to", "no longer", decision numbers or comparisons with origin/main; that history belongs in commit messages and plans). Write them for an agent or person reading the file with no other context
 - add your next instruction here

# Git repositories, branches and worktrees

Check which repository a file belongs to before committing (`git -C <folder> rev-parse --show-toplevel` and `git remote -v`)

 - **The outer repository (this folder, remote https://github.com/wharrison2/spar-scaffolding):** you may commit straight to `main` and push. Expect other agents to be working in it alongside you on different tasks:
   - Before committing, run `git fetch origin`, then `git pull --no-rebase origin main`
   - Stage only the files your task changed, by explicit path. Never use a bare `git add -A` or `git add .`, and never commit other agents' unrelated uncommitted changes
   - Never run `git clean`, `git reset --hard`, `git stash -u` or `git checkout -- .`, since these destroy other agents' work and can reach the nested repositories
   - Never force push
 - **The contributor kit repository:** commit to its `main` as described in "Where the emergent misalignment prompt corpus contributor kit lives" above
 - **Every other repository, including `animal-welfare-data-pipeline/` and its worktrees (`animal-welfare-data-pipeline-*/`, `animal-welfare-data-pipeline/.claude/worktrees/*`):** be careful. Never commit directly to `main`. Work on a branch in its own worktree, as normal. These repositories sit inside the outer folder, but they are not part of the outer repository: never stage them in it
 - **When you open a worktree (in any repository):** put a file named `WORKTREE_STATUS.md` in the worktree's root folder and keep it out of history: add `WORKTREE_STATUS.md` to that repository's `.git/info/exclude` (shared by all its worktrees; find it with `git rev-parse --git-common-dir`) if it is not already there. Never commit it. It says:
   - the branch and the task the worktree is for
   - its status: in progress, waiting for review, merged, or abandoned
   - what is done, what is left, and anything another agent picking it up needs to know
   - when it was last updated (date and time)

   Update it as you work: whenever the status changes, and before you stop

# Modifying this file

 - When I phrase an instruction with "always", "never", or some other instruction implying that it generalizes to all similar circumstances, add it to the skill in `.claude/skills/` that matches its topic. If no skill matches, add it to the Agent additions section of this file