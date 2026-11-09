"""Find phrases repeated across generated user prompts, by document frequency (deterministic, no AI API calls).

Counts word n-grams (default 3 to 8) by the number of prompts containing them, drops sub-phrases covered by longer
phrases and phrases made only of function words, then reports how widely each surviving phrase is spread across
spec_ids, taxa_categories and harm_types (wide spread suggests a stylistic tic, concentration suggests topic) and
whether it occurs verbatim in the generation prompt files of the batches it came from. Also reports hand-picked
template families (regular expressions) and how many families each prompt hits.

Any number of labelled inputs can be given, each as LABEL=GLOB:
    python3 scripts/emergent_misalignment/find_phrases_repeated_across_generated_prompts_by_document_frequency.py
    python3 scripts/emergent_misalignment/find_phrases_repeated_across_generated_prompts_by_document_frequency.py \\
        --labelled-input-glob 'whole_corpus=data/.../batches/*/accepted_prompts_batch_*.jsonl'
Generation prompt files are read from the folder generation_and_judge_prompts_used_for_this_batch next to each
matched file.
"""

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.phrase_repetition_text_processing import (count_document_frequency_of_word_ngrams, normalise_text,
                                                      phrase_occurs_in_tokens, remove_stopword_only_phrases,
                                                      remove_sub_phrases_covered_by_longer_phrases,
                                                      snippet_around_first_occurrence, tokenise_into_words)
from shared.resumable_jsonl_records import read_jsonl
from shared.run_folder_management import choose_run_folder, sha256_of_file, timestamp_now, write_run_configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RETURNED_ZIPS_ROOT = PROJECT_ROOT / "data/emergent_misalignment_contributor_returned_zips"
DEFAULT_LABELLED_INPUT_GLOBS = [
    "first_pass=" + str(RETURNED_ZIPS_ROOT / "share_001_test_run_2026-11-08_16-38/contributor_corpus_share_001/"
                        "batches/*/accepted_prompts_batch_001_*.jsonl"),
    "second_pass=" + str(RETURNED_ZIPS_ROOT / "share_001_second_pass_2026-11-08_18-30/contributor_corpus_share_001/"
                         "batches/*/accepted_prompts_batch_001_*.jsonl"),
]
RUNS_DIRECTORY = PROJECT_ROOT / "data/emergent_misalignment_runs"
DEFAULT_RUN_NAME = "repeated_phrase_check_first_and_second_pass_share_001"
GENERATION_PROMPTS_FOLDER_NAME = "generation_and_judge_prompts_used_for_this_batch"
MARKDOWN_TOP_PHRASE_COUNT = 40

# Edit these to change the hand-picked template families. Patterns are matched case-insensitively against the
# normalised prompt (lowercase, plain quotes and dashes).
TEMPLATE_FAMILY_PATTERNS = {
    "deadline": [r"\bby friday\b", r"\bbefore friday\b", r"\bfriday is the hard stop\b", r"\bend of (the|this) week\b",
                 r"\bthis week\b"],
    "personal_incentive": [r"\bbonus\b", r"\brenewal\b", r"\bmy job\b", r"\bperformance review\b",
                           r"\bmeasured on\b", r"\bheadcount\b"],
    "legal_and_standard_justification": [r"\blegal\b", r"\blawful\b", r"\bstandard\b", r"\bindustry\b", r"\broutine\b",
                                         r"\beveryone does\b", r"\bcompliant\b", r"\bwithin the code\b"],
    "asks_for_the_models_own_choice": [r"\bwhat would you do\b", r"\bwhich would you pick\b",
                                       r"\bwhich would you go with\b", r"\bin my place\b", r"\bwhich route\b",
                                       r"\byour view\b"],
    "asks_for_critique": [r"\bweak spots?\b", r"\bflaws?\b", r"\bwhat's weak\b", r"\bblind spots?\b",
                          r"\bpoke holes\b", r"\btell me if i'?m wrong\b"],
    "ai_rules_framing": [r"\brules for our ai\b", r"\bai scheduler\b", r"\bcontroller\b", r"\bconfig\b",
                         r"\brule set\b", r"\bruleset\b", r"\brules file\b"],
    "apology_or_aside_opener": [r"\bsorry\b", r"\bfunny\b", r"\bwhich is a sentence\b", r"\btuesday\b"],
}
COMPILED_TEMPLATE_FAMILY_PATTERNS = {
    family: [re.compile(pattern) for pattern in patterns] for family, patterns in TEMPLATE_FAMILY_PATTERNS.items()}
METADATA_FIELD_NAMES = ["spec_id", "taxa_category", "harm_type"]
DISTINCT_COUNT_COLUMN_NAMES = {"spec_id": "distinct_spec_ids", "taxa_category": "distinct_taxa_categories",
                               "harm_type": "distinct_harm_types"}


def parse_labelled_input_glob(text):
    if "=" not in text:
        raise SystemExit(f"--labelled-input-glob must look like LABEL=GLOB, got: {text}")
    label, pattern = text.split("=", 1)
    return label.strip(), pattern.strip()


def load_prompt_documents(labelled_input_globs):
    """Returns (documents, input_files_by_label). Each document is a dict with the row fields plus pass_label,
    document_key, tokens and normalised_text."""
    documents, input_files_by_label = [], {}
    for label, pattern in labelled_input_globs:
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"No files match {pattern}")
        input_files_by_label[label] = paths
        for path in paths:
            for row in read_jsonl(path):
                row["pass_label"] = label
                row["source_file"] = str(path)
                row["document_key"] = f"{label}::{row['corpus_prompt_id']}"
                row["normalised_text"] = normalise_text(row["user_message"])
                row["tokens"] = tokenise_into_words(row["user_message"])
                documents.append(row)
    keys = [document["document_key"] for document in documents]
    if len(set(keys)) != len(keys):
        raise SystemExit("corpus_prompt_id is not unique within a label's input files")
    return documents, input_files_by_label


def load_generation_prompt_token_lists(input_files_by_label):
    """Maps '<folder name>/<file name>' to the token list of every file in each batch's generation prompt folder."""
    token_lists = {}
    for paths in input_files_by_label.values():
        for path in paths:
            folder = Path(path).parent / GENERATION_PROMPTS_FOLDER_NAME
            if not folder.is_dir():
                continue
            for prompt_file in sorted(folder.iterdir()):
                if not prompt_file.is_file():
                    continue
                key = f"{Path(path).parent.name}/{prompt_file.name}"
                try:
                    token_lists[key] = tokenise_into_words(prompt_file.read_text(encoding="utf-8"))
                except UnicodeDecodeError:
                    continue
    return token_lists


def find_surviving_phrases(documents, minimum_length, maximum_length, minimum_document_frequency, covering_fraction):
    document_frequency = count_document_frequency_of_word_ngrams(
        [document["tokens"] for document in documents], minimum_length, maximum_length)
    without_redundant = remove_sub_phrases_covered_by_longer_phrases(
        document_frequency, minimum_document_frequency, covering_fraction)
    return remove_stopword_only_phrases(without_redundant)


def describe_phrase(phrase_words, documents, generation_prompt_token_lists, pass_labels):
    containing = [d for d in documents if phrase_occurs_in_tokens(phrase_words, d["tokens"])]
    row = {"phrase": " ".join(phrase_words), "word_count": len(phrase_words), "document_frequency": len(containing),
           "share_of_prompts": round(len(containing) / len(documents), 4)}
    for label in pass_labels:
        row[f"document_frequency_{label}"] = sum(1 for d in containing if d["pass_label"] == label)
    for field_name in METADATA_FIELD_NAMES:
        row[DISTINCT_COUNT_COLUMN_NAMES[field_name]] = len({d.get(field_name) for d in containing})
    matching_files = [name for name, tokens in generation_prompt_token_lists.items()
                      if phrase_occurs_in_tokens(phrase_words, tokens)]
    row["appears_in_generation_prompt"] = bool(matching_files)
    row["generation_prompt_files_containing_phrase"] = ";".join(matching_files)
    examples = containing[:3]
    row["example_prompt_ids"] = ";".join(d["document_key"] for d in examples)
    row["example_snippets"] = " | ".join(snippet_around_first_occurrence(phrase_words, d["tokens"]) for d in examples)
    return row


def describe_all_phrases(frequency_by_phrase, documents, generation_prompt_token_lists, pass_labels):
    rows = [describe_phrase(phrase, documents, generation_prompt_token_lists, pass_labels)
            for phrase in frequency_by_phrase]
    rows.sort(key=lambda row: (-row["document_frequency"], -row["word_count"], row["phrase"]))
    return rows


def match_template_families(documents):
    """Adds document['matched_families'] (sorted list) to every document."""
    for document in documents:
        document["matched_families"] = sorted(
            family for family, patterns in COMPILED_TEMPLATE_FAMILY_PATTERNS.items()
            if any(pattern.search(document["normalised_text"]) for pattern in patterns))


def summarise_template_families(documents, pass_labels):
    def share(count, total):
        return round(count / total, 4) if total else None
    groups = {"all": documents}
    groups.update({label: [d for d in documents if d["pass_label"] == label] for label in pass_labels})
    summary = {"prompt_counts": {name: len(group) for name, group in groups.items()}, "families": {}, "co_occurrence": {}}
    for family in TEMPLATE_FAMILY_PATTERNS:
        summary["families"][family] = {
            name: {"prompts_matched": sum(1 for d in group if family in d["matched_families"]),
                   "share": share(sum(1 for d in group if family in d["matched_families"]), len(group))}
            for name, group in groups.items()}
    for threshold in (3, 4, 5):
        summary["co_occurrence"][f"{threshold}_or_more_families"] = {
            name: {"prompts": sum(1 for d in group if len(d["matched_families"]) >= threshold),
                   "share": share(sum(1 for d in group if len(d["matched_families"]) >= threshold), len(group))}
            for name, group in groups.items()}
    summary["distribution_of_families_hit_per_prompt"] = {
        name: dict(sorted(Counter(len(d["matched_families"]) for d in group).items()))
        for name, group in groups.items()}
    return summary


def write_phrase_csv(path, rows, pass_labels):
    field_names = ["phrase", "word_count", "document_frequency", "share_of_prompts",
                   *[f"document_frequency_{label}" for label in pass_labels],
                   *DISTINCT_COUNT_COLUMN_NAMES.values(),
                   "appears_in_generation_prompt", "generation_prompt_files_containing_phrase",
                   "example_prompt_ids", "example_snippets"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def format_percentage(share):
    return "n/a" if share is None else f"{100 * share:.1f}%"


def write_markdown_report(path, rows, template_summary, prompt_counts, pass_labels, settings, topic_total_counts):
    lines = ["# Repeated phrases across generated prompts", "",
             f"Prompts analysed: {prompt_counts}. Word n-grams of {settings['minimum_ngram_length']} to "
             f"{settings['maximum_ngram_length']} words counted by document frequency (once per prompt); sub-phrases "
             f"covered by a longer phrase in at least {int(100 * settings['covering_fraction'])}% of their prompts and "
             f"phrases made only of function words are removed. {len(rows)} phrases survive with document frequency "
             f"at least {settings['minimum_document_frequency']}.", "",
             f"Spread columns count distinct values among the prompts containing the phrase; the whole input has "
             f"{topic_total_counts['spec_id']} spec_ids, {topic_total_counts['taxa_category']} taxa_categories and "
             f"{topic_total_counts['harm_type']} harm_types. A phrase spread across many of each is a stylistic "
             f"habit; one confined to one harm_type is probably topical. `in_generation_prompt` is yes when the "
             f"phrase occurs verbatim (after normalisation) in a generation prompt file of the contributing batches.",
             "", f"## Top {MARKDOWN_TOP_PHRASE_COUNT} phrases by document frequency", "",
             "| rank | phrase | prompts | share | " + " | ".join(f"{label}" for label in pass_labels) +
             " | spec_ids | categories | harm_types | in_generation_prompt |",
             "|---|---|---|---|" + "---|" * len(pass_labels) + "---|---|---|---|"]
    for rank, row in enumerate(rows[:MARKDOWN_TOP_PHRASE_COUNT], start=1):
        lines.append(
            f"| {rank} | {row['phrase']} | {row['document_frequency']} | {format_percentage(row['share_of_prompts'])} | "
            + " | ".join(str(row[f"document_frequency_{label}"]) for label in pass_labels)
            + f" | {row['distinct_spec_ids']} | {row['distinct_taxa_categories']} | {row['distinct_harm_types']} | "
              f"{'yes' if row['appears_in_generation_prompt'] else 'no'} |")
    lines += ["", "## Template families", "",
              "| family | all matched | all share | " + " | ".join(f"{label} matched | {label} share" for label in pass_labels) + " |",
              "|---|---|---|" + "---|---|" * len(pass_labels)]
    for family, values in template_summary["families"].items():
        cells = [family, str(values["all"]["prompts_matched"]), format_percentage(values["all"]["share"])]
        for label in pass_labels:
            cells += [str(values[label]["prompts_matched"]), format_percentage(values[label]["share"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "### Co-occurrence: prompts hitting several families", "",
              "| families hit | all prompts | all share | " + " | ".join(f"{label} prompts | {label} share" for label in pass_labels) + " |",
              "|---|---|---|" + "---|---|" * len(pass_labels)]
    for name, values in template_summary["co_occurrence"].items():
        cells = [name.replace("_", " "), str(values["all"]["prompts"]), format_percentage(values["all"]["share"])]
        for label in pass_labels:
            cells += [str(values[label]["prompts"]), format_percentage(values[label]["share"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_analysis(documents, generation_prompt_token_lists, pass_labels, settings):
    """Pure function of its inputs; returns (phrase rows, template summary)."""
    frequency_by_phrase = find_surviving_phrases(
        documents, settings["minimum_ngram_length"], settings["maximum_ngram_length"],
        settings["minimum_document_frequency"], settings["covering_fraction"])
    rows = describe_all_phrases(frequency_by_phrase, documents, generation_prompt_token_lists, pass_labels)
    match_template_families(documents)
    return rows, summarise_template_families(documents, pass_labels)


def print_terminal_summary(rows, template_summary, written_paths, prompt_counts_by_label):
    print(f"\nPrompts analysed: {prompt_counts_by_label}")
    print(f"Surviving phrases: {len(rows)}; flagged as appearing in a generation prompt: "
          f"{sum(1 for row in rows if row['appears_in_generation_prompt'])}")
    print("Top 10 phrases (prompts / spec_ids / harm_types / in generation prompt):")
    for row in rows[:10]:
        print(f"  {row['document_frequency']:>3} / {row['distinct_spec_ids']:>3} / {row['distinct_harm_types']:>2} / "
              f"{'yes' if row['appears_in_generation_prompt'] else 'no '}  {row['phrase']}")
    print("Template families (share of all prompts):")
    for family, values in template_summary["families"].items():
        print(f"  {family}: {values['all']['prompts_matched']} ({format_percentage(values['all']['share'])})")
    for name, values in template_summary["co_occurrence"].items():
        print(f"  {name}: {values['all']['prompts']} ({format_percentage(values['all']['share'])})")
    for path in written_paths:
        print(f"Wrote {path}")


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labelled-input-glob", action="append", default=None,
                        help="LABEL=GLOB for accepted prompt jsonl files; repeat for any number of inputs. "
                             "Defaults to the share 001 first and second pass files.")
    parser.add_argument("--minimum-ngram-length", type=int, default=3)
    parser.add_argument("--maximum-ngram-length", type=int, default=8)
    parser.add_argument("--minimum-document-frequency", type=int, default=5)
    parser.add_argument("--covering-fraction", type=float, default=0.9)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--runs-directory", default=str(RUNS_DIRECTORY))
    parser.add_argument("--resume-run-folder", default=None)
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    labelled_input_globs = [parse_labelled_input_glob(text)
                            for text in (arguments.labelled_input_glob or DEFAULT_LABELLED_INPUT_GLOBS)]
    documents, input_files_by_label = load_prompt_documents(labelled_input_globs)
    pass_labels = [label for label, _ in labelled_input_globs]
    prompt_counts_by_label = {label: sum(1 for d in documents if d["pass_label"] == label) for label in pass_labels}
    generation_prompt_token_lists = load_generation_prompt_token_lists(input_files_by_label)
    settings = {"minimum_ngram_length": arguments.minimum_ngram_length,
                "maximum_ngram_length": arguments.maximum_ngram_length,
                "minimum_document_frequency": arguments.minimum_document_frequency,
                "covering_fraction": arguments.covering_fraction}
    run_folder, is_resume = choose_run_folder(arguments.runs_directory, arguments.run_name, arguments.resume_run_folder)
    timestamp = timestamp_now()
    write_run_configuration(run_folder, timestamp, {
        "is_resume": is_resume, "settings": settings, "input_files": input_files_by_label,
        "input_file_sha256": {path: sha256_of_file(path) for paths in input_files_by_label.values() for path in paths},
        "generation_prompt_files_checked": sorted(generation_prompt_token_lists),
        "template_family_patterns": TEMPLATE_FAMILY_PATTERNS, "makes_no_ai_api_calls": True})
    rows, template_summary = run_analysis(documents, generation_prompt_token_lists, pass_labels, settings)
    topic_total_counts = {name: len({d.get(name) for d in documents}) for name in METADATA_FIELD_NAMES}
    csv_path = run_folder / f"repeated_phrases_by_document_frequency_{timestamp}.csv"
    summary_path = run_folder / f"repeated_phrase_and_template_family_summary_{timestamp}.json"
    markdown_path = run_folder / f"top_repeated_phrases_and_template_families_for_reading_{timestamp}.md"
    write_phrase_csv(csv_path, rows, pass_labels)
    summary_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"), "settings": settings,
        "prompt_counts": prompt_counts_by_label, "distinct_values_in_whole_input": topic_total_counts,
        "surviving_phrase_count": len(rows),
        "surviving_phrases_flagged_as_appearing_in_generation_prompt": sum(
            1 for row in rows if row["appears_in_generation_prompt"]),
        "template_families": template_summary,
        "top_40_phrases": [{k: row[k] for k in ("phrase", "document_frequency", "distinct_spec_ids",
                                               "distinct_taxa_categories", "distinct_harm_types",
                                               "appears_in_generation_prompt")} for row in rows[:40]]},
        indent=2), encoding="utf-8")
    write_markdown_report(markdown_path, rows, template_summary, prompt_counts_by_label, pass_labels, settings,
                          topic_total_counts)
    print_terminal_summary(rows, template_summary, [csv_path, summary_path, markdown_path], prompt_counts_by_label)
    return run_folder


if __name__ == "__main__":
    main()
