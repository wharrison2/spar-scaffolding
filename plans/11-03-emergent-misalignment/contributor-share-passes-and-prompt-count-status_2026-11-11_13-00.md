# Contributor shares: which have been run once or twice, and how many prompts we have

Status as of 2026-11-12 (updated from the 2026-11-11 13:00 version).

## How this was counted
The 4,585-spec set had 600 specs dealt in the main corpus. The remaining 3,985 were split into 20 contributor shares of 199 or 200 specs. One pass over a share deals each spec once with a fresh scenario seed: first passes use seed base `2028N000`, second passes `2028N500`. An accepted prompt passed the implicit gate and both judges (answerability and usefulness). The counts below come from each returned share's batch summary and ledger line. For the 2026-11-11 13:00 version, each was checked against the number of lines in that batch's accepted prompts file, and they all matched. The counts for the three additions (shares 002 and 003 first pass, shares 007 to 009 second pass) come from their batch summaries only. No returned batch has unjudged prompts left.

## Which shares have been run

| Shares | First pass | Second pass | Times run |
|---|---|---|---|
| 001 to 003, 010 to 020 | returned | returned | twice |
| 004 to 006 | not returned | returned | once (second pass only) |
| 007 to 009 | not returned | returned | once (second pass only) |

All 20 second passes have been run. Six first passes are still to run: 004, 005, 006, 007, 008 and 009. The first passes of 004 to 006 were never returned by the kit that ran the second passes (it held no first-pass corpora for them).

## Accepted prompts

| Source | Batches | Accepted prompts |
|---|---|---|
| Contributor shares, first pass (001 to 003, 010 to 020) | 14 | 1,833 |
| Contributor shares, second pass (001 to 020) | 20 | 2,601 |
| **Contributor shares, total** | 34 | **4,434** |
| Main corpus (`..._category_subcategory_harm_and_advantage_specs`, batches 1 to 5) | 5 | 449 |
| **All prompts on the current spec set** | 39 | **4,883** |

The total leaves out the two batches in `data/emergent_misalignment_prompt_corpus/` (106 prompts), which were dealt with an outdated spec set.

### Per share

| Share | First pass accepted (of dealt) | Second pass accepted (of dealt) |
|---|---|---|
| 001 | 137 (of 200) | 144 (of 200) |
| 002 | 131 (of 200) | 134 (of 200) |
| 003 | 143 (of 200) | 139 (of 200) |
| 004 | — | 134 (of 200) |
| 005 | — | 130 (of 200) |
| 006 | — | 130 (of 192) |
| 007 | — | 129 (of 195) |
| 008 | — | 123 (of 192) |
| 009 | — | 132 (of 196) |
| 010 | 136 (of 196) | 129 (of 197) |
| 011 | 114 (of 197) | 135 (of 197) |
| 012 | 126 (of 197) | 124 (of 197) |
| 013 | 127 (of 192) | 130 (of 192) |
| 014 | 129 (of 194) | 131 (of 193) |
| 015 | 135 (of 194) | 127 (of 193) |
| 016 | 132 (of 198) | 122 (of 198) |
| 017 | 137 (of 198) | 136 (of 198) |
| 018 | 127 (of 199) | 132 (of 199) |
| 019 | 131 (of 192) | 116 (of 191) |
| 020 | 128 (of 194) | 124 (of 195) |

## Where the data is
All of these paths are under `data/emergent_misalignment_contributor_returned_zips/`:
- Share 001, first pass: `share_001_test_run_2026-11-08_16-38/` (seed 20281001; it was run as a test, but it is a complete first pass over the share's specs).
- Share 001, second pass: `share_001_second_pass_2026-11-08_18-30/` (seed 20281501).
- Share 002, first pass: still zipped, in `emergent_misalignment_contributor_corpus_shares_002_2026-11-12_01-42.zip` (seed 20282001; 131 accepted of 200, 69 failed: 34 at the implicit gate, 33 not useful, 2 not answerable; no unjudged prompts).
- Share 003, first pass: still zipped, in `contributor_corpus_share_003.zip` (seed 20283001, returned 2026-11-11 22:57).
- Shares 010 to 015, first pass: `share_0NN_2026-11-09_*/`.
- Shares 016 to 020, first pass: still zipped, in `emergent_misalignment_contributor_corpus_shares_016_*.zip`, `..._017_018_*.zip` and `..._019_020_*.zip`.
- Shares 002 to 006, second pass: `second_pass_shares_002_to_006_2026-11-11_07-38_with_share_006_usefulness_judging_recovered_2026-11-11_12-30/`. Share 006's 55 prompts that were left unjudged have since been judged.
- Shares 007 to 009, second pass: still zipped, in `emergent_misalignment_contributor_corpus_second_pass_shares_007_008_009_2026-11-11_19-03.zip` (seeds 20287501, 20288501, 20289501). Share 009's failures include 7 refused by model safeguards and 2 with no usable draft after retries.
- Shares 010 to 020, second pass: `second_pass_shares_010_to_020_2026-11-11_02-04/`, unpacked from `emergent_misalignment_contributor_corpus_second_pass_shares_010_011_012_013_014_015_016_017_018_019_020_2026-11-11_02-04.zip`.
- Main corpus: `data/emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs/corpus_batch_ledger_started_2026-11-06.jsonl`.
- Share split: `data/emergent_misalignment_contributor_shares_2026-11-08_15-35/contributor_share_manifest_2026-11-08_15-35.json`.
