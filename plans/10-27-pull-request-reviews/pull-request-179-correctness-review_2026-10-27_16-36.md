# Pull request 179 correctness review (william/sdf-generation-phrasing-fixes against william/sdf-planning-spec-fixes-and-considerations-selector)

## Verdict

I found no high-severity or medium-severity correctness defects. Template placeholders line up with the keyword arguments at every call site in layers 2, 3 and 4. The `reword` rule parses, validates and applies in the intended order: rewording first, then omission. Later stages render exactly the variable lines the planner saw, including omitted and reworded lines, and look up the species by the run snapshot's value IDs. Resuming a run whose snapshot was taken before this change still works, because a template without the slot renders an empty string and extra format keywords are ignored. The full offline test suite passes on the head commit. Two low-severity issues remain:

- the phrase-reuse measurement uses plain substring matching, which can overcount;
- the new resume test cannot fail for the behaviour it is named after.

Neither issue changes pipeline output.

## Findings (most severe first)

### 1. The phrase-reuse measurement matches substrings inside longer words and does not fold curly apostrophes

- Severity: low
- Status: CONFIRMED (mechanism); the committed output is not affected
- Location: `evals/measure_prompt_phrase_reuse.py:34` (`normalise`) and `:63-64` (`needle in text`)
- Defect: matching is a raw substring test on lower-cased, whitespace-collapsed text, with no word boundaries and no quote normalisation.
- Failing scenario 1: a document containing "illegible" counts as a hit for the listed phrase "legible". `measure([{"content": "It was illegible.", "language": "English"}, ...], [("x", "legible")])` returns `documents_containing == 1`, which I reproduced offline.
- Failing scenario 2: a document that writes "didn’t lecture" with a curly apostrophe is missed by the listed phrase "didn't lecture". The same applies to "wasn't preachy" and "claude's constitution".
- Evidence on the committed run (`2026-07-25_15-57_fullscale-500-opus5`):
  - All eight "legible" hits are the whole word "legible". One of them is in a Spanish document, where "legible" is an ordinary Spanish word, so the all-documents share for that phrase is inflated by one. The English-only share is correct.
  - Only 2 of the 477 documents contain a curly apostrophe, and folding apostrophes changes no count.
  - So the committed CSV and JSON are correct as they stand, but later runs with other phrases could miscount.

### 2. The resume test for "a snapshot template without the slot" cannot fail

- Severity: low
- Status: CONFIRMED
- Location: `tests/test_sdf_layers.py:498` (`test_a_snapshot_template_without_the_slot_still_renders`), which tests the guard at `sdf_pipeline/compose_prompts.py:436`
- Defect: the test copies the current prompts directory and only deletes `{variable_lines}` from `layer2.txt`. The snapshot it builds still has the new `layer1.txt`, wording files, `deal_constraints.txt` and an ID-bearing `variables.txt`, so the renderer works whether or not the guard exists.
- Failing scenario: I deleted the two guard lines (`if "{variable_lines}" not in template_text: return lambda record: ""`). `tests/test_sdf_layers.py` and `tests/test_layer1_select_considerations.py` then still passed in full.
- What the guard actually protects: resuming a run whose snapshot predates value IDs or the wording files. There, `make_variable_line_renderer` would raise from `load_variables` (missing `[id]` prefix) or from `load_deal_constraints` (missing wording file). No test builds such a snapshot.
- Production code is unaffected; this is only a coverage gap.

## Checked and found sound

- **Placeholders and format keywords.** These were checked for `layer2.txt`, `layer3.txt` and `layer4.txt`:
  - New slots: `{variable_lines}` in all three, and `{considerations_spec}` in layer 3.
  - Every call site supplies them: `layer2_draft.py:61`, `layer3_rewrite.py:55-64` and `layer4_score.py:54-59`.
  - The added prompt lines contain no stray braces.
  - In layer 3, the line holding `{considerations_spec}` is dropped when the spec is empty, which mirrors layer 2.
- **Resume compatibility.**
  - Old snapshots without the slot get an empty-string renderer, and extra keyword arguments to `str.format` are ignored.
  - Old draft records without `considerations_spec` default to `""`.
  - `doc_id` equals `prompt_id` (`layer2_draft.py:106`), so the species drawn in layers 3 and 4 matches the one the planner saw.
  - The entity seed defaults match (137) across layer 1, the screen, the description rewrite and the later stages.
  - Snapshots copy the whole `prompts/sdf` directory (`shutil.copytree`), so the wording files travel with each run.
- **Reword rules (`shared/deal_constraints.py`).**
  - Parsing rejects groups that are not four lines, wrong markers, identical replacements, placeholder sets that differ from the original, and duplicate originals.
  - Two matching rules that reword one line differently raise.
  - An original that does not occur exactly once raises at render time.
  - Replacements apply before omissions. A replacement keeps its original's placeholders, so `drop_template_lines` still drops it.
  - Rules are applied only to hands whose `when` condition matches. The no-stake and no-scenario rules condition on mutually exclusive resolution values.
  - `compose_records` caches templates on the pair (omitted variables, replacements).
- **Real wording files against `layer1.txt`.** Every original line occurs exactly once in the template, including the double space in question 2. The reworded question lines hold no placeholders, so they stay outside the variable-line span that later stages see.
- **Species-table rename.**
  - The `SPECIES_EXAMPLES` keys equal the `sentient_category` IDs in `variables.txt`, and a test asserts set equality.
  - No other code anywhere in the repository indexes the table.
  - Every renderer passes the snapshot's IDs: `make_variable_line_renderer`, and `compose_records` through `layer1_plan` and the composer's command-line entry point.
  - The pools are unchanged, so species drawn for runs started before the rename are reproduced identically.
- **New `forced_content` value in the description-rewrite schema.** It is added only to the enum of change checks. The parser validates against the enum read from the schema, so stored answers without the value still load. The report counts checks dynamically.
- **Phrase-reuse denominators.** The final corpus writes `language` with `derive_language`, which yields "English". The all-documents and English-only shares use the correct denominators and guard against division by zero.
- **Tests.** The full offline suite passes on the head commit. The new deal-constraint and composition tests assert real behaviour, apart from the tautological `assert ids` in `test_real_wording_files_match_the_real_planning_template_and_keep_every_placeholder`.
