"""The pipeline-side tests import the pipeline's `shared` package, which has the same name as this project's
scripts/shared (imported by the OpenAI-side modules). They cannot share a process, so they are ignored when a
directory is collected; test_spec_contradiction_screen_validation.py runs them in their own process."""

collect_ignore = ["test_pipeline_side_hand_pool_and_planted_hands.py"]
