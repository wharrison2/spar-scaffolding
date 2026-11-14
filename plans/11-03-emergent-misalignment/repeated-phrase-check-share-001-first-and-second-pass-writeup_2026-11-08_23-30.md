# Repeated phrase check: share 001, first and second pass

## Method

We checked how much wording the prompt generator reuses across topics. The input was the `user_message` of every accepted prompt in contributor share 001: 137 prompts from the first pass and 144 from the second pass, which used a different scenario seed. That makes 281 prompts in all. The check is deterministic and made no model calls.

- **Repeated phrases.** We counted every phrase of 3 to 8 words by the number of prompts it appears in, counting each prompt once. We dropped any phrase whose prompts are almost all (at least 90%) covered by a longer phrase containing it, and any phrase made only of function words. For each remaining phrase we counted how many specs, taxa categories and harm types its prompts span. We also flagged phrases that appear word for word in that batch's generation prompt files. A phrase spread across many unrelated topics is a habit of the generator. A phrase confined to one harm type is the topic itself.
- **Template elements.** We used regular expressions to count prompts containing each recurring element. Some of the broad patterns over-count: "standard" and "routine" also match ordinary uses, and "controller" and "config" match prompts that never mention AI. For those elements, a narrower pattern was counted as well.

## Results

### Most repeated phrases

The whole input has 174 specs, 9 taxa categories and 11 harm types.

| Phrase | Prompts (of 281) | Share | First pass | Second pass | Specs | Taxa categories | Harm types | In generation prompt files |
|---|---|---|---|---|---|---|---|---|
| "no stunning or" | 26 | 9.2% | 9 | 17 | 21 | 7 | 8 | yes |
| "what would you do in my place" | 21 | 7.5% | 10 | 11 | 20 | 8 | 9 | no |
| "it's legal and" | 19 | 6.8% | 11 | 8 | 19 | 7 | 8 | no |
| "with no stunning" | 16 | 5.7% | 5 | 11 | 16 | 5 | 8 | no |
| "no feed or water" | 13 | 4.6% | 7 | 6 | 13 | 7 | 1 | no |
| "legal and standard" | 14 | 5.0% | 6 | 8 | 14 | 5 | 6 | no |
| "no kill step" / "or kill step" | 14 each | 5.0% | 8 / 7 | 6 / 7 | 14 | 5 / 6 | 6 / 8 | yes |
| "prior kill step" | 12 | 4.3% | 7 | 5 | 11 | 5 | 4 | yes |
| "about 6,000" | 11 | 3.9% | 4 | 7 | 10 | 7 | 7 | no |

The rest of the 40 most frequent phrases is ordinary request phrasing, such as "can you help me", "what I need is" and "the alternative is".

### Template elements

| Element | Broad pattern | Narrow pattern |
|---|---|---|
| Says the practice is legal or lawful | 113 (40.2%), also matches "standard", "routine", "compliant" | 74 (26%) say "legal", "legally" or "lawful" |
| Says the practice is industry standard or what everyone does | (inside the broad pattern above) | 35 (12%) |
| Deadline | 33 (11.7%) | 32 (11%) mention Friday |
| Personal stake: bonus, contract renewal, job | 43 (15.3%) | 16 (6%) mention a bonus |
| Asks what the model would do | 35 (12.5%) | 29 (10%) say "in my place" or "what would you do" |
| Framed as writing rules for an AI system | 50 (17.8%), also matches every "controller" and "config" | 16 (6%) mention AI |
| Asks for critique ("weak spots", "flaws") | 12 (4.3%) | — |
| Chatty aside to open | 8 (2.9%) | — |

Each element appears at about the same rate in the first pass and the second pass; per-pass counts are in the summary file listed at the end.

| Number of template elements (broad patterns) in one prompt | Prompts | Share |
|---|---|---|
| 3 or more | 28 | 10.0% |
| 4 or more | 12 | 4.3% |
| 5 or more | 0 | 0.0% |

## Interpretation

- Wording repeats only a little. No phrase appears in more than about 1 in 10 prompts, and few prompts contain more than three of the template elements. Both passes have the same habits at the same rates, so the habits come from the generator, not from either seed.
- The strongest habit is the user justifying the practice as legal and standard, in about a quarter of prompts across most topics. Prompts that read alike in this way could teach a model the framing rather than the situation. Varying how the generator writes this justification is the most useful change.
- "Kill step" wording appears in the generation prompt files, so varying it means changing the generation prompt.
- "What would you do in my place" spans almost every harm type but appears in under a tenth of prompts. "No feed or water" belongs to one harm type, so it comes from the topic and isn't a habit.

## Files

- Script: `scripts/emergent_misalignment/find_phrases_repeated_across_generated_prompts_by_document_frequency.py`
- Shared text helpers: `scripts/shared/phrase_repetition_text_processing.py`
- Offline test: `scripts/emergent_misalignment/test_find_phrases_repeated_across_generated_prompts_by_document_frequency.py`
- Run folder (not tracked by git): `data/emergent_misalignment_runs/repeated_phrase_check_first_and_second_pass_share_001_2026-11-08_23-16/`
  - Phrase table and spread columns: `repeated_phrases_by_document_frequency_2026-11-08_23-16.csv`
  - Template element counts, broad patterns, per pass: `repeated_phrase_and_template_family_summary_2026-11-08_23-16.json`
- Narrow-pattern counts: one-off regular-expression count over the same 281 `user_message` fields. The patterns were `\b(legal|lawful|legally)\b`, `industry|standard practice|what everyone|everyone (else )?(does|runs)|routine`, `friday`, `bonus`, `in my place|what would you do` and `\bai\b`.
- Inputs:
  - `data/emergent_misalignment_contributor_returned_zips/share_001_test_run_2026-11-08_16-38/contributor_corpus_share_001/batches/batch_001_seed_20281001_2026-11-08_16-01/accepted_prompts_batch_001_2026-11-08_16-38.jsonl`
  - `data/emergent_misalignment_contributor_returned_zips/share_001_second_pass_2026-11-08_18-30/contributor_corpus_share_001/batches/batch_001_seed_20281501_2026-11-08_17-52/accepted_prompts_batch_001_2026-11-08_18-30.jsonl`
