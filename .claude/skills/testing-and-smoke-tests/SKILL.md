---
name: testing-and-smoke-tests
description: Rules for testing code in this project. Use whenever writing smoke tests, automated tests, validation runs, cost estimates or small-sample checks, or deciding where their outputs go.
---

# Testing code

 - For large new functionalities, write smoke testing scripts that can be run quickly to confirm pipelines work
 - Include automated tests for experiment-invalidating bugs
 - Put all smoke tests, cost estimates and small-sample checks for a job into one shared folder for that job (e.g. `data/<job>_runs/small_sample_checks/**files**`), rather than in their own folders at the level where full valid runs go
