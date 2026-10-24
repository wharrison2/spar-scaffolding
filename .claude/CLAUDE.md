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
 - Commit changes on a branch, never directly to main
 - add your next instruction here

# Modifying this file

 - When I phrase an instruction with "always", "never", or some other instruction implying that it generalizes to all similar circumstances, add it to the skill in `.claude/skills/` that matches its topic. If no skill matches, add it to the Agent additions section of this file