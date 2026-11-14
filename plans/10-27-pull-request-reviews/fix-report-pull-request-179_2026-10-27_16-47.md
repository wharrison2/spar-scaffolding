# Fix report for pull request 179 review

Starting commit: 2e5619003777a9b9ba0d5a91a0e0212271a9ec17 (william/sdf-generation-phrasing-fixes). Worked in a detached worktree, since removed. Patches: /private/tmp/claude-501/-Users-williamharrison-Desktop-emergent-alignment/08c219e3-e182-4ff7-a9a8-e7625dddc1d7/scratchpad/review_fixes/pull-request-179/ (three files, 0001 to 0003).

## Finding 1: substring matching and curly apostrophes — DONE
- Commit: "Match whole words and fold curly apostrophes in phrase reuse measurement"
- Tests (tests/test_measure_prompt_phrase_reuse.py): `test_a_phrase_inside_a_longer_word_is_not_counted`, `test_curly_and_straight_apostrophes_match_each_other`.
- Confirmed failing before: stashed the fix to evals/measure_prompt_phrase_reuse.py and ran the file; both new tests failed. Both pass after.
- Choices: whole-word match via lookarounds on word characters (not a Spanish-aware filter; the report's English-only share already handles that). Both curly single quotes fold to a straight apostrophe, on documents and phrases alike. Module docstring updated.

## Finding 2: resume test cannot fail — DONE
- Commit: "Make the pre-slot snapshot resume test fail without the guard"
- Test: `test_a_snapshot_template_without_the_slot_still_renders` (tests/test_sdf_layers.py), strengthened: the snapshot now also lacks the wording files and has value IDs stripped from variables.txt, as a pre-slot snapshot would.
- Confirmed: removed the guard in `render_variable_lines_for_template`; the test failed with a ValueError from shared/matrix.py (missing IDs). With the guard restored it passes.
- No production code change, no docs stale.

## Data exclusion — DONE
- Commit: "Stop committing phrase reuse measurement output"
- Deleted the committed CSV and JSON under outputs/sdf/phrase_reuse_measurements/, added that directory to .gitignore with a comment in the style of its neighbours (verified with git check-ignore).
- Grep for the three terms found only the script itself (default output root and file names); no code, doc, test or comment referred to the committed baseline. The script and its synthetic-data tests are unchanged.

## Not changed
- The tautological `assert ids` in `test_real_wording_files_match_the_real_planning_template_and_keep_every_placeholder` is mentioned in the report's "found sound" section, not as a numbered finding, so I left it. It is a candidate for a follow-up.

## Full suite
Full offline suite run at the final commit: all tests passed, no failures (baseline was 1466; this adds 2).
