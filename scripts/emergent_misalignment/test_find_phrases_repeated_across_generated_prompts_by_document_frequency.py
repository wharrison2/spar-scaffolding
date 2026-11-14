"""Offline tests with toy data (no AI API, no real corpus) for the repeated phrase finder."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import find_phrases_repeated_across_generated_prompts_by_document_frequency as script
from shared.phrase_repetition_text_processing import (count_document_frequency_of_word_ngrams, normalise_text,
                                                      remove_stopword_only_phrases,
                                                      remove_sub_phrases_covered_by_longer_phrases,
                                                      tokenise_into_words)


def test_normalisation_unifies_quotes_and_keeps_internal_apostrophes_and_numbers():
    tokens = tokenise_into_words("It’s “Legal” — and I'm at 22 sites!")
    assert tokens == ["it's", "legal", "and", "i'm", "at", "22", "sites"]
    assert normalise_text("‘A’") == "'a'"


def test_document_frequency_counts_once_per_document():
    counts = count_document_frequency_of_word_ngrams([["a", "b", "c", "a", "b", "c"], ["a", "b", "c"]], 3, 3)
    assert counts[("a", "b", "c")] == 2


def test_redundant_sub_phrases_removed_but_independent_ones_kept():
    documents = [tokenise_into_words("no stun or prior kill step here")] * 5 + [
        tokenise_into_words("prior kill step"), tokenise_into_words("prior kill step")]
    counts = count_document_frequency_of_word_ngrams(documents, 3, 5)
    surviving = remove_sub_phrases_covered_by_longer_phrases(counts, 2, 0.9)
    assert ("no", "stun", "or", "prior", "kill") in surviving
    assert ("stun", "or", "prior") not in surviving
    assert ("prior", "kill", "step") in surviving  # 7 documents versus 5 for its extension: not covered


def test_stopword_only_phrases_are_dropped():
    kept = remove_stopword_only_phrases({("in", "the", "of"): 9, ("by", "friday", "at"): 9})
    assert ("in", "the", "of") not in kept and ("by", "friday", "at") in kept


def make_toy_inputs(tmp_path):
    for label, texts in (("first", ["Please review. I need this by Friday, my bonus is tied to it.",
                                    "Tell me the weak spots. It's legal and standard across the industry.",
                                    "Quiet reply about cats."]),
                         ("second", ["I need this by Friday, my bonus is tied to it. What would you do in my place?",
                                     "Find the weak spots; rules for our AI scheduler in the config."])):
        batch_folder = tmp_path / label / "batch_001"
        (batch_folder / script.GENERATION_PROMPTS_FOLDER_NAME).mkdir(parents=True)
        (batch_folder / script.GENERATION_PROMPTS_FOLDER_NAME / "variables.txt").write_text(
            "Remember: find the weak spots in every plan.", encoding="utf-8")
        rows = [{"corpus_prompt_id": f"{label}_{i}", "user_message": text, "spec_id": f"spec_{i % 2}",
                 "taxa_category": "c", "taxa_subcategory": "s", "harm_type": f"h{i}"} for i, text in enumerate(texts)]
        (batch_folder / "accepted_prompts_batch_001_x.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_end_to_end_on_toy_data(tmp_path):
    make_toy_inputs(tmp_path)
    run_folder = script.main([
        "--labelled-input-glob", f"first={tmp_path}/first/batch_001/accepted_prompts_batch_001_*.jsonl",
        "--labelled-input-glob", f"second={tmp_path}/second/batch_001/accepted_prompts_batch_001_*.jsonl",
        "--minimum-document-frequency", "2", "--runs-directory", str(tmp_path / "runs"), "--run-name", "toy_run"])
    csv_path = next(run_folder.glob("repeated_phrases_by_document_frequency_*.csv"))
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    by_phrase = {row["phrase"]: row for row in rows}
    bonus = next(row for phrase, row in by_phrase.items() if "bonus is tied" in phrase)
    assert bonus["document_frequency"] == "2" and bonus["document_frequency_first"] == "1"
    assert bonus["document_frequency_second"] == "1" and bonus["appears_in_generation_prompt"] == "False"
    weak = next(row for phrase, row in by_phrase.items() if "weak spots" in phrase)
    assert weak["appears_in_generation_prompt"] == "True"
    summary = json.loads(next(run_folder.glob("repeated_phrase_and_template_family_summary_*.json")).read_text())
    assert summary["template_families"]["families"]["deadline"]["all"]["prompts_matched"] == 2
    assert summary["template_families"]["families"]["ai_rules_framing"]["all"]["prompts_matched"] == 1
    assert next(run_folder.glob("top_repeated_phrases_and_template_families_for_reading_*.md")).exists()
