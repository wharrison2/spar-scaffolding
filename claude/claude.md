# Writing code

 - All files your scripts generate must be named descriptively with their date and time at the end of the file name
 - Have all code that runs on a pod or makes api calls that cost money output progress reports with timings at natural checkpoints (these should not be more frequent than roughly every 3 minutes, but don't spend too long figuring out how to time them that way)
 - Use descriptive, human-legible variable names and patterns
 - Write code that can be resumed from natural checkpoints whenever possible 
 - Reuse code from this project when reasonable. Let me know when copying from somewhere that isn't this project, and make sure that the addition is noted in references.md
 - Save data from a given run into its own specific, descriptively named folder. Subfolders for outputs that aren't from the same run/process
 - Have Sonnet 5.5 subagents write most code. Only use Opus subagents if I instruct you to, or Haiku subagents if the task is menial, requires no judgement, experimentally unimportant, or will not produce code that will be frequently reused
 - Do not abbreviate names (runs, directories, files, pods, logs, monitors) or terms in messages; write them out in full
 - Always use descriptive, legible file names: someone who has never seen the file should be able to tell what it holds from its name alone
 - Always sample generations at temperature 1 and top-p 1, unless otherwise instructed

# Testing code

 - For large new functionalities, write smoke testing scripts that can be run quickly to confirm pipelines work
 - Include automated tests for experiment-invalidating bugs

# Writing results

 - When writing results from a session, be concise but thorough. Write interpretations apart from results
 - Specify the path where outputs from a session went, and where any numbers in results stem from

# Agent additions

If one of these instructions conflicts with an instruction in the sections above, tell me and defer to the above

 - add your next instruction here

# Modifying this file

 - When I phrase an instruction with "always", "never", or some other instruction implying that it generalizes to all similar circumstances, add an instruction to the Agent additions section of this file