---
name: scripts-that-call-ai-apis
description: Rules for code that makes AI model API calls or runs on a paid pod in this project. Use whenever writing, editing or running such a script, including small tests, cost estimates and prompt files.
---

# Considerations for using AI APIs
 - Do not run scripts that use these without my explicit go-ahead. This includes small tests
 - Confirm that you are using a key specific to this project from the `.env` before running code. Never use a key from my OS environment
 - When possible, all scripts that make AI API calls should track the cost, token usage (input, output, reasoning), and cache rate of the run using the provider's API, then log this in a file
 - Always sample generations at temperature 1 and top-p 1, unless otherwise instructed. For Anthropic models, leave sampling parameters unset: the API rejects `temperature` and `top_p`
 - Have all code that runs on a pod or makes api calls that cost money output progress reports with timings at natural checkpoints (these should not be more frequent than roughly every 3 minutes, but don't spend too long figuring out how to time them that way)
 - Never make AIs reproduce precise IDs, or make calls that could easily be done deterministically in code
 - Store large or important strings that go to models in `.txt` files, not in hardcoded strings. Name these files appropriately
