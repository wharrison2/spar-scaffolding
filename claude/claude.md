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
 - When possible, all scripts that make AI API calls should track the cost, token usage (input, output, reasoning), and cache rate of the run using the provider's API, then log this in a file
 - Always sample generations at temperature 1 and top-p 1, unless otherwise instructed
 - Have all code that runs on a pod or makes api calls that cost money output progress reports with timings at natural checkpoints (these should not be more frequent than roughly every 3 minutes, but don't spend too long figuring out how to time them that way)
 - Never make AIs reproduce precise IDs, or make calls that could easily be done deterministically in code
 - Store large or important strings that go to models in `.txt` files, not in hardcoded strings. Name these files appropriately

# Testing code

 - For large new functionalities, write smoke testing scripts that can be run quickly to confirm pipelines work
 - Include automated tests for experiment-invalidating bugs
 - Data from small smoke- or validation tests should all go into one folder for the task, e.g. data/shoehorning_audit/validation/**files** rather than putting them at the same level as full valid runs

# Writing results
Writeups are to have a record of what was done, reports are for other people to see

## Writeups
 - Be concise but thorough. Write interpretations apart from results
 - Specify the path where outputs from a session went, and where any numbers in results stem from

## Reports/summaries
 - Avoid jargon
 - Keep this document useful as a standalone (but assume the reader has basic familiarity with the SDF dataset and the process to generate it)
 - Unless a data point is extremely important, do not repeat it in plain text. if precision is necessary, repeat it in a table. otherwise,
  leave it to the figures
 - Don't reference prior mistakes or other documents unless necessary. Do not mention your conversation with me except unless circumstances absolutely call for it

# Agent additions

If one of these instructions conflicts with an instruction in the sections above, tell me and defer to the above

 - Put all smoke tests, cost estimates and small-sample checks for a job into one shared folder for that job (e.g. `<job>_runs/small_sample_checks/`), not in their own folders at the level where full runs go
 - Never leave artifacts of the drafting or editing process in a report (e.g. "not discussed further", "corrects an earlier statement", "pending check", "proposed and not applied"); write each report as if it were the only version the reader will ever see
 - add your next instruction here

# Modifying this file

 - When I phrase an instruction with "always", "never", or some other instruction implying that it generalizes to all similar circumstances, add an instruction to the Agent additions section of this file