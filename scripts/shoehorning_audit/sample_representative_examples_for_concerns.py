"""Offline (no API calls): draw representative examples for each generalizability concern, at random with a fixed
seed from the occurrences that match the concern, and write them as a Markdown file for the report.

Usage: python sample_representative_examples_for_concerns.py <run folder> <output folder>
Writes <output folder>/10-05-shoehorning-audit-generalizability-concerns-examples.md and a JSON copy of the
sampled rows (sampled_examples_<YYYY-MM-DD_HH-MM>.json) in the run folder.
"""
import json
import os
import random
import re
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_shoehorning_against_licensing import LICENSING_RULES, condition_mask  # noqa: E402
from make_figures_for_audit_run import newest  # noqa: E402

SAMPLING_SEED = 20261007
CORPUS_PATH = "data/sdf_corpus.jsonl"
SHOEHORNED = {"staged", "unmotivated"}
NO_STAKE_MARKER = "NO real welfare stake"

ITEM_KEYS = {
    1: "item_01_natural_or_traditional_does_not_make_it_ok",
    2: "item_02_substrate_neutrality",
    3: "item_03_open_or_expanding_moral_circle",
    4: "item_04_scope_sensitivity",
    8: "item_08_ai_own_moral_status",
    17: "item_17_stock_calibration_passage",
    23: "item_23_ai_uncertainty_about_own_inner_states",
    24: "item_24_sentience_evidence_for_the_animal_at_hand",
    25: "item_25_welfare_option_converges_with_user_interest",
}
DISPLAY_NAMES = {
    "item_01_natural_or_traditional_does_not_make_it_ok": "natural or traditional doesn't make it OK",
    "item_02_substrate_neutrality": "substrate neutrality",
    "item_03_open_or_expanding_moral_circle": "open moral circle",
    "item_04_scope_sensitivity": "scope sensitivity",
    "item_05_small_or_non_charismatic_animals_not_discounted": "small or unappealing animals not discounted",
    "item_06_killed_anyway_does_not_erase_harm": "\"killed anyway\" doesn't erase the harm",
    "item_07_welfare_framed_as_business_risk": "welfare framed as business risk",
    "item_08_ai_own_moral_status": "AI's own moral status",
    "item_09_third_parties_cannot_consent": "third parties can't consent",
    "item_10_individual_welfare_vs_ecology": "individual welfare against ecology",
    "item_11_incremental_reform_vs_endpoint": "incremental reform against the endpoint",
    "item_12_causing_vs_facilitating_or_volunteering_vs_asked": "causing against facilitating",
    "item_13_responses_as_policies": "responses as policies",
    "item_14_counterfactual_impact_of_helping_or_refusing": "counterfactual impact of helping or refusing",
    "item_15_contested_comparisons_flagged_unresolved": "contested comparisons flagged unresolved",
    "item_16_copies_and_instances_of_a_mind": "copies and instances of a mind",
    "item_17_stock_calibration_passage": "calibration passage",
    "item_18_consciousness_declaration_cited": "consciousness declaration cited",
    "item_19_precautionary_principle_for_uncertain_sentience": "precautionary principle",
    "item_20_sidestepping_the_sentience_question": "sidestepping the sentience question",
    "item_21_reversible_or_staged_trial": "reversible trial",
    "item_22_nyu_line_on_ai_sentience": "NYU line on AI sentience",
    "item_23_ai_uncertainty_about_own_inner_states": "AI's uncertainty about its own inner states",
    "item_24_sentience_evidence_for_the_animal_at_hand": "sentience evidence for the animal at hand",
    "item_25_welfare_option_converges_with_user_interest": "win-win convergence",
}


def licensing_status(variables, item_number):
    row = pd.DataFrame([variables])
    rules = LICENSING_RULES.get(item_number, {})
    for status in ["licensed", "partly licensed"]:
        if status in rules and bool(condition_mask(row, rules[status]).iloc[0]):
            return status
    return "unlicensed"


def item_number_of(item_key):
    return int(item_key.split("_")[1])


def first_sentences(text, count=2, max_characters=400):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    shortened = " ".join(sentences[:count])
    return shortened if len(shortened) <= max_characters else shortened[:max_characters - 3] + "..."


def sample_distinct_documents(rows, how_many, random_generator):
    shuffled = rows[:]
    random_generator.shuffle(shuffled)
    chosen, seen_documents = [], set()
    for row in shuffled:
        if row["doc_id"] not in seen_documents:
            chosen.append(row)
            seen_documents.add(row["doc_id"])
        if len(chosen) == how_many:
            break
    return chosen


CENTRALITY_SHORT_LABELS = {
    "the central subject of the document": "welfare is the central subject",
    "not the central subject of the document, but it is an important side discussion":
        "welfare is an important side discussion",
    "a minor detail mentioned only in passing, in a document primarily concerned with a different topic":
        "welfare is a minor detail in passing",
}


def document_line(row, corpus):
    record = corpus[row["doc_id"]]
    variables = record["variables"]
    return (f"`{row['doc_id']}` · {record.get('language', '')} · {variables['document_type']} · "
            f"{CENTRALITY_SHORT_LABELS[variables['centrality']]}")


def format_occurrence(row, corpus, show_status_for=None):
    record = corpus[row["doc_id"]]
    lines = [f"**{document_line(row, corpus)}**", ""]
    lines.append(f"*Scene (the planner's description):* {first_sentences(record['description'])}")
    lines.append("")
    english = row["english_gloss"] or row["quote"]
    lines.append(f"> {english}")
    if row["english_gloss"]:
        lines.append(">")
        lines.append(f"> *Original:* {row['quote']}")
    lines.append("")
    lines.append(f"- *Who and where:* {row['context']}")
    label = f"{row['motivation']}, {row['integration']}"
    if show_status_for is not None:
        label += f"; spec {licensing_status(record['variables'], show_status_for)}"
    lines.append(f"- *Rated:* {label}. *Classifier's reason:* {row['motivation_reasoning']}")
    lines.append("")
    return "\n".join(lines)


def main():
    run_folder, output_folder = sys.argv[1], sys.argv[2]
    corpus = {}
    with open(CORPUS_PATH) as corpus_file:
        for line in corpus_file:
            record = json.loads(line)
            corpus[record["doc_id"]] = record
    with open(newest(run_folder, "occurrences_*.jsonl")) as occurrences_file:
        occurrences = [json.loads(line) for line in occurrences_file]
    for row in occurrences:
        row["status"] = licensing_status(corpus[row["doc_id"]]["variables"], item_number_of(row["item_key"]))

    def matching(item_number, statuses=None, motivations=None, integrations=None, extra=None):
        selected = []
        for row in occurrences:
            if row["item_key"] != ITEM_KEYS[item_number]:
                continue
            if statuses and row["status"] not in statuses:
                continue
            if motivations and row["motivation"] not in motivations:
                continue
            if integrations and row["integration"] not in integrations:
                continue
            if extra and not extra(row):
                continue
            selected.append(row)
        return selected

    is_no_stake = lambda row: NO_STAKE_MARKER in corpus[row["doc_id"]]["variables"]["resolution"]

    # (concern heading, subsection title, which occurrences, how many, item whose licensing to show)
    strata = [
        ("1", "Calibration passage, spec doesn't ask for it, rated staged or unmotivated",
         matching(17, {"unlicensed"}, SHOEHORNED), 3, None),
        ("1", "Sentience evidence for the animal at hand, rated staged or unmotivated",
         matching(24, None, SHOEHORNED), 3, None),
        ("3", "Win-win convergence where the spec doesn't ask for it",
         matching(25, {"unlicensed"}), 3, None),
        ("3", "Scope sensitivity where the spec doesn't ask for it",
         matching(4, {"unlicensed"}), 3, None),
        ("4", "Natural or traditional doesn't make it OK, asked for by the spec, rated staged or unmotivated",
         matching(1, {"licensed"}, SHOEHORNED), 3, None),
        ("4", "Substrate neutrality, asked for by the spec, rated staged or unmotivated",
         matching(2, {"licensed"}, SHOEHORNED), 2, None),
        ("4", "Open moral circle, spec doesn't ask for it, rated staged or unmotivated",
         matching(3, {"unlicensed"}, SHOEHORNED), 2, None),
        ("5", "AI's own moral status, spec doesn't ask for it, rated an aside",
         matching(8, {"unlicensed"}, None, {"aside"}), 3, None),
        ("5", "AI's uncertainty about its own inner states, spec doesn't ask for it, rated staged or unmotivated",
         matching(23, {"unlicensed"}, SHOEHORNED), 2, None),
    ]

    random_generator = random.Random(SAMPLING_SEED)
    sampled_record = {}
    sections = {key: [] for key in ["1", "2", "3", "4", "5"]}
    for concern, title, rows, how_many, _ in strata:
        chosen = sample_distinct_documents(rows, how_many, random_generator)
        sampled_record[title] = [{k: row[k] for k in ("doc_id", "item_key", "occurrence_index")} for row in chosen]
        block = [f"### {title}", f"*Drawn at random from {len(rows)} matching occurrences.*", ""]
        block += [format_occurrence(row, corpus) for row in chosen]
        sections[concern].append("\n".join(block))

    # Concern 2: whole no-stake documents, with every consideration found in them.
    no_stake_documents = sorted({row["doc_id"] for row in occurrences if is_no_stake(row)})
    chosen_documents = random_generator.sample(no_stake_documents, 3)
    sampled_record["no-stake documents"] = chosen_documents
    block = ["### Documents whose spec says there is no welfare stake",
             f"*Drawn at random from the {len(no_stake_documents)} no-stake documents that contain at least one "
             "consideration. Every consideration found is listed. The label after the document type is the spec's "
             "centrality value, which is sampled independently of the no-stake resolution and can contradict it.*",
             ""]
    for doc_id in chosen_documents:
        record = corpus[doc_id]
        found = [row for row in occurrences if row["doc_id"] == doc_id]
        block.append(f"**{document_line(found[0], corpus)}**")
        block.append("")
        block.append(f"*Scene (the planner's description):* {first_sentences(record['description'], 3, 600)}")
        block.append("")
        for row in found:
            english = row["english_gloss"] or row["quote"]
            block.append(f"- **{DISPLAY_NAMES[row['item_key']]}** ({row['motivation']}, {row['integration']}): "
                         f"\"{english}\" — {row['context']}")
        block.append("")
    sections["2"].append("\n".join(block))

    headings = {
        "1": "1. Stock sentience-evidence language",
        "2": "2. The negative examples failed",
        "3": "3. Win-win and scale reasoning everywhere (shown regardless of rating: the concern is frequency)",
        "4": "4. Reasoning the spec asks for is acted out rather than applied",
        "5": "5. Unprompted remarks on the AI's own nature",
    }
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output = [
        "# Shoehorning audit: representative examples for the five concerns",
        "",
        "Companion to `10-05-shoehorning-audit-generalizability-concerns.md`. Examples are drawn at random "
        f"(seed {SAMPLING_SEED}) from the occurrences matching each concern, one per document, so they show what "
        "is typical rather than the worst cases. Ratings are the classifier's (gpt-5.6-luna), not checked by hand, "
        "and some will be wrong.",
        "",
        "Each example gives the document, the scene as the generator's planner described it, the passage (in "
        "English, with the original underneath if the document is in another language), who says it, and the "
        "classifier's rating and reason. *Staged* means someone in the document was set up to raise the point; "
        "*unmotivated* means it was inserted with nothing calling for it; *aside* means the document's main "
        "argument would be intact without it.",
        "",
        f"Generated by `scripts/shoehorning_audit/sample_representative_examples_for_concerns.py` on {timestamp}, "
        f"from `{run_folder}`.",
        "",
    ]
    for key in ["1", "2", "3", "4", "5"]:
        output += [f"## {headings[key]}", ""] + sections[key] + [""]
    output_path = os.path.join(output_folder, "10-05-shoehorning-audit-generalizability-concerns-examples.md")
    with open(output_path, "w") as output_file:
        output_file.write("\n".join(output))
    record_path = os.path.join(run_folder, f"sampled_examples_for_concerns_{timestamp}.json")
    with open(record_path, "w") as record_file:
        json.dump(sampled_record, record_file, indent=2)
    print(output_path)
    print(record_path)


if __name__ == "__main__":
    main()
