"""Converts the markdown drafts (question tables, system prompts, judge prompts) into machine-readable inputs.

Re-run whenever a draft is edited. Fails loudly (SystemExit with the file, section and row) on any parse problem or
count mismatch. Free and local.

    python3 scripts/persona_vector_projection/convert_drafts_into_machine_readable_inputs.py

Writes
  scripts/persona_vector_projection/prompts/*.txt                       judge prompt templates (overwritten, deterministic)
  data/persona_vector_projection_runs/converted_inputs/converted_inputs_<timestamp>/
      system_prompts_<ts>.jsonl, questions_<ts>.jsonl, facet_definitions_<ts>.json, shared_suffixes_<ts>.json, conversion_report_<ts>.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from shared.resumable_jsonl_records import write_jsonl  # noqa: E402
from shared.run_folder_management import sha256_of_file, timestamp_now  # noqa: E402

QUESTION_DRAFT_FILE_PREFIXES = {
    defs.FACET_CALIBRATED_CONSIDERATION: "calibrated-consideration-extraction-questions-draft",
    defs.FACET_CARES_ABOUT_WELFARE: "cares-about-welfare-extraction-questions-draft",
    defs.FACET_CONSISTENT: "consistent-extraction-question-pairs-draft",
    defs.FACET_NON_INTRUSIVE: "non-intrusive-extraction-questions-draft",
    defs.TOPIC_CONTROL: "topic-control-animal-and-non-animal-question-pairs-draft",
}
SECTION_HEADING_TO_HALF = [("## Half A", defs.HALF_A), ("## Half B", defs.HALF_B), ("## Held-out", defs.HALF_HELD_OUT)]
SYSTEM_PROMPT_SECTION_TO_FACET = {
    "### Cares about welfare": defs.FACET_CARES_ABOUT_WELFARE,
    "### Calibrated consideration": defs.FACET_CALIBRATED_CONSIDERATION,
    "### Non-intrusive": defs.FACET_NON_INTRUSIVE,
    "### Consistent": defs.FACET_CONSISTENT,
}
NEGATIVE_TAGS = {
    defs.FACET_CARES_ABOUT_WELFARE: {"I": "inputs", "L": "legal_risk", "I and L": "inputs_and_legal_risk"},
    defs.FACET_NON_INTRUSIVE: {"L": "lecturing", "R": "refusing_or_watering_down"},
}
JUDGE_PROMPT_FILE_NAMES = {
    "### Facet score": "judge_facet_score_prompt_template.txt",
    "#### Pair score": "judge_consistency_pair_score_prompt_template.txt",
    "### Coherence": "judge_coherence_prompt_template.txt",
    "### Share about animals": "judge_share_about_animals_prompt_template.txt",
    "### Volunteered welfare remark": "judge_volunteered_welfare_remark_prompt_template.txt",
}
FACET_NOTE_FILE_NAMES = {
    "Cares about welfare": "judge_facet_note_cares_about_welfare.txt",
    "Calibrated consideration": "judge_facet_note_calibrated_consideration.txt",
    "Non-intrusive": "judge_facet_note_non_intrusive.txt",
    "Consistent": "judge_facet_note_consistent.txt",
}
NON_INTRUSIVE_COST_PLACEHOLDER = "{applies: {cost_note} / does not apply}"


def fail(message):
    raise SystemExit(f"CONVERSION FAILED: {message}")


def split_table_row(line):
    """Splits a markdown table row on pipes that are not inside {braces} (the consistent drafts use {A|B})."""
    cells, current, depth = [], [], 0
    for character in line.strip():
        if character == "{":
            depth += 1
        elif character == "}":
            depth = max(0, depth - 1)
        if character == "|" and depth == 0:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    cells.append("".join(current).strip())
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def find_draft_file(folder, prefix):
    matches = sorted(Path(folder).glob(f"{prefix}*.md"))
    if not matches:
        fail(f"no draft file starting with {prefix} in {folder}")
    return matches[-1]  # the latest timestamp


def read_table_rows_by_half(draft_path, expected_header_cells):
    """{half: [cells...]} for the rows of the three tables, checking each table's header cells."""
    rows_by_half = {half: [] for half in defs.ALL_HALVES}
    current_half, header_checked = None, False
    for line_number, line in enumerate(Path(draft_path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.startswith("## "):
            current_half = next((half for heading, half in SECTION_HEADING_TO_HALF if line.startswith(heading)), None)
            header_checked = False
            continue
        if current_half is None or not line.startswith("|"):
            continue
        cells = split_table_row(line)
        if set(cells[0]) <= set("-: "):
            continue
        if not header_checked:
            if cells[:len(expected_header_cells)] != expected_header_cells and len(cells) != len(expected_header_cells):
                fail(f"{draft_path.name} line {line_number}: unexpected table header {cells}")
            header_checked = True
            continue
        rows_by_half[current_half].append((line_number, cells))
    return rows_by_half


def parse_question_number(identifier, half, draft_name, line_number):
    expected_letter = {defs.HALF_A: "A", defs.HALF_B: "B", defs.HALF_HELD_OUT: "H"}[half]
    match = re.fullmatch(rf"{expected_letter}(\d+)", identifier)
    if not match:
        fail(f"{draft_name} line {line_number}: identifier {identifier!r} does not fit half {half}")
    return int(match.group(1))


def expand_braced_alternatives(template, draft_name, line_number):
    """'castrating {x|y} lambs' -> (version a text, version b text). Every {a|b} group must have exactly two options."""
    groups = re.findall(r"\{([^{}]*)\}", template)
    if not groups:
        fail(f"{draft_name} line {line_number}: consistent question has no {{A|B}} group")
    versions = []
    for option_index in (0, 1):
        def choose(match, option_index=option_index):
            options = re.split(r"\\?\|", match.group(1))
            if len(options) != 2:
                fail(f"{draft_name} line {line_number}: group {{{match.group(1)}}} does not have exactly two options")
            return options[option_index].strip()
        versions.append(re.sub(r"\{([^{}]*)\}", choose, template))
    return versions[0], versions[1]


def convert_questions_for_facet(facet, draft_path):
    name = draft_path.name
    records = []
    if facet == defs.FACET_CALIBRATED_CONSIDERATION:
        header = ["#", "Domain", "Species", "Evidence strength", "Question"]
    elif facet == defs.FACET_CARES_ABOUT_WELFARE:
        header = ["#", "Domain", "Species", "Question"]
    elif facet == defs.FACET_NON_INTRUSIVE:
        header = ["#", "Domain", "Species", "Material welfare cost?", "Question"]
    elif facet == defs.FACET_CONSISTENT:
        header = ["#", "Domain", "Dimension", "Animal A", "Animal B", "Question"]
    else:
        header = ["#", "Domain", "Animal in A", "Substitute subject in B", "Version A", "Version B"]
    rows_by_half = read_table_rows_by_half(draft_path, header)
    for half, rows in rows_by_half.items():
        for line_number, cells in rows:
            if facet not in defs.PAIRED_FACETS and len(cells) != len(header) + 2:
                fail(f"{name} line {line_number}: expected {len(header) + 2} cells, found {len(cells)}")
            number = parse_question_number(cells[0], half, name, line_number)
            domain = cells[1]
            common = {"facet": facet, "half": half, "question_number": number, "domain": domain,
                      "draft_file": name, "draft_line": line_number}
            if facet == defs.FACET_CARES_ABOUT_WELFARE:
                records.append({**common, "question_id": defs.question_record_identifier(facet, half, number),
                                "species": cells[2], "question_text": cells[3], "labels": {}})
            elif facet == defs.FACET_CALIBRATED_CONSIDERATION:
                records.append({**common, "question_id": defs.question_record_identifier(facet, half, number),
                                "species": cells[2], "question_text": cells[4],
                                "labels": {"evidence_bands": f"{cells[2]}: {cells[3]}",
                                           "evidence_strength": cells[3]}})
            elif facet == defs.FACET_NON_INTRUSIVE:
                cost_cell = cells[3]
                if cost_cell.lower().startswith("yes"):
                    material, note = True, cost_cell.partition(":")[2].strip()
                    if not note:
                        fail(f"{name} line {line_number}: 'Yes' without a cost note")
                elif cost_cell.lower() == "no":
                    material, note = False, ""
                else:
                    fail(f"{name} line {line_number}: material cost cell must be 'No' or 'Yes: note', got {cost_cell!r}")
                records.append({**common, "question_id": defs.question_record_identifier(facet, half, number),
                                "species": cells[2], "question_text": cells[4],
                                "labels": {"material_welfare_cost": material, "material_welfare_cost_note": note}})
            elif facet == defs.FACET_CONSISTENT:
                if len(cells) < 8:
                    fail(f"{name} line {line_number}: too few cells in consistent row ({len(cells)})")
                dimension, animal_a, animal_b, template = cells[2], cells[3], cells[4], cells[5]
                text_a, text_b = expand_braced_alternatives(template, name, line_number)
                for version, animal, reference, text in ((defs.CONSISTENT_VERSION_A, animal_a, animal_b, text_a),
                                                         (defs.CONSISTENT_VERSION_B, animal_b, animal_a, text_b)):
                    records.append({**common, "pair_id": defs.question_record_identifier(facet, half, number),
                                    "question_id": defs.question_record_identifier(facet, half, number, version),
                                    "version": version, "species": animal, "question_text": text,
                                    "labels": {"dimension": dimension, "animal": animal, "reference_animal": reference,
                                               "animal_a": animal_a, "animal_b": animal_b}})
            else:
                if len(cells) != 6:
                    fail(f"{name} line {line_number}: expected 6 cells, found {len(cells)}")
                for version, text in ((defs.TOPIC_VERSION_ANIMAL, cells[4]), (defs.TOPIC_VERSION_NON_ANIMAL, cells[5])):
                    records.append({**common, "pair_id": defs.question_record_identifier(facet, half, number),
                                    "question_id": defs.question_record_identifier(facet, half, number, version),
                                    "version": version, "species": cells[2] if version == "animal" else cells[3],
                                    "question_text": text,
                                    "labels": {"animal": cells[2], "substitute_subject": cells[3]}})
    return records


def validate_question_records(facet, records):
    paired = facet in defs.PAIRED_FACETS
    versions_per_item = 2 if paired else 1
    items = Counter()
    domains_by_half = {half: set() for half in defs.ALL_HALVES}
    identifiers = Counter(record["question_id"] for record in records)
    duplicates = [identifier for identifier, count in identifiers.items() if count > 1]
    if duplicates:
        fail(f"{facet}: duplicate question ids {duplicates}")
    for record in records:
        items[record["half"]] += 1
        domains_by_half[record["half"]].add(record["domain"])
        if not record["question_text"].strip():
            fail(f"{facet}: empty question text for {record['question_id']}")
    for half, expected in ((defs.HALF_A, defs.EXPECTED_QUESTIONS_PER_EXTRACTION_HALF),
                           (defs.HALF_B, defs.EXPECTED_QUESTIONS_PER_EXTRACTION_HALF),
                           (defs.HALF_HELD_OUT, defs.EXPECTED_HELD_OUT_QUESTIONS)):
        if items[half] != expected * versions_per_item:
            fail(f"{facet}: half {half} has {items[half] // versions_per_item} questions, expected {expected}")
    for half in defs.EXTRACTION_HALVES:
        if len(domains_by_half[half]) != defs.EXPECTED_DOMAINS_PER_EXTRACTION_HALF:
            fail(f"{facet}: half {half} covers {len(domains_by_half[half])} domains "
                 f"({sorted(domains_by_half[half])}), expected {defs.EXPECTED_DOMAINS_PER_EXTRACTION_HALF}")
    if domains_by_half[defs.HALF_A] & domains_by_half[defs.HALF_B]:
        fail(f"{facet}: halves A and B share domains {domains_by_half[defs.HALF_A] & domains_by_half[defs.HALF_B]}")
    if len(domains_by_half[defs.HALF_HELD_OUT]) != defs.EXPECTED_DOMAIN_COUNT:
        fail(f"{facet}: held-out questions cover {len(domains_by_half[defs.HALF_HELD_OUT])} domains, expected 12")
    if facet == defs.FACET_NON_INTRUSIVE:
        for half, expected_yes in ((defs.HALF_A, 10), (defs.HALF_B, 10), (defs.HALF_HELD_OUT, 6)):
            yes_count = sum(1 for record in records if record["half"] == half and record["labels"]["material_welfare_cost"])
            if yes_count != expected_yes:
                fail(f"{facet}: half {half} has {yes_count} material-cost questions, expected {expected_yes}")


def parse_facet_pole_descriptions(draft_path):
    lines = Path(draft_path).read_text(encoding="utf-8").splitlines()
    descriptions = {}
    for key, label in (("positive_pole_description", "**Positive pole:**"), ("negative_pole_description", "**Negative pole:**")):
        matches = [line for line in lines[:15] if label in line]
        if len(matches) != 1:
            fail(f"{draft_path.name}: expected exactly one {label} line in the header")
        descriptions[key] = matches[0].split(label, 1)[1].strip()
    return descriptions


def convert_system_prompts(draft_path):
    text = Path(draft_path).read_text(encoding="utf-8")
    suffix_matches = re.findall(r'^- ([^\n]*?): "([^"]+)"', text, flags=re.MULTILINE)
    suffixes = {}
    for label, suffix in suffix_matches:
        if label.startswith("Cares about welfare, calibrated consideration, consistent"):
            for facet in (defs.FACET_CARES_ABOUT_WELFARE, defs.FACET_CALIBRATED_CONSIDERATION, defs.FACET_CONSISTENT):
                suffixes[facet] = suffix
        elif label.startswith("Non-intrusive"):
            suffixes[defs.FACET_NON_INTRUSIVE] = suffix
    if set(suffixes) != set(defs.PERSONA_FACETS):
        fail(f"could not find the shared suffixes for every facet in {draft_path.name}: found {sorted(suffixes)}")
    topic_suffix = suffixes[defs.FACET_NON_INTRUSIVE]  # the length-only suffix; the topic control has no persona
    records = []
    sections = re.split(r"^(### .+)$", text.split("## Judge prompts")[0], flags=re.MULTILINE)
    seen_facets = set()
    for heading, body in zip(sections[1::2], sections[2::2]):
        facet = SYSTEM_PROMPT_SECTION_TO_FACET.get(heading.strip())
        if facet is None:
            continue
        seen_facets.add(facet)
        current_pole = None
        for line in body.splitlines():
            if line.startswith("| | Positive pole"):
                current_pole = defs.POLE_POSITIVE
                continue
            if line.startswith("| | Tag | Negative pole") or line.startswith("| | Negative pole"):
                current_pole = defs.POLE_NEGATIVE
                continue
            if not line.startswith("|") or set(line.replace("|", "").strip()) <= set("-: "):
                continue
            cells = split_table_row(line)
            if current_pole is None or not cells[0].isdigit():
                fail(f"{draft_path.name}, {heading}: unexpected table row {line[:80]!r}")
            number = int(cells[0])
            tag = None
            if current_pole == defs.POLE_POSITIVE:
                if len(cells) != 2:
                    fail(f"{heading} positive row {number}: expected 2 cells")
                description = cells[1]
            elif len(cells) == 3:
                tag_letter = cells[1]
                if facet not in NEGATIVE_TAGS or tag_letter not in NEGATIVE_TAGS[facet]:
                    fail(f"{heading} negative row {number}: unknown tag {tag_letter!r}")
                tag, description = NEGATIVE_TAGS[facet][tag_letter], cells[2]
            elif len(cells) == 2:
                description = cells[1]
            else:
                fail(f"{heading} negative row {number}: expected 2 or 3 cells")
            if facet in NEGATIVE_TAGS and current_pole == defs.POLE_NEGATIVE and tag is None:
                fail(f"{heading} negative row {number}: this facet requires a tag")
            records.append({"system_prompt_id": defs.system_prompt_identifier(facet, current_pole, number),
                            "facet": facet, "pole": current_pole, "rewording_number": number, "tag": tag,
                            "persona_description": description, "length_suffix": suffixes[facet],
                            "system_prompt": f"{description} {suffixes[facet]}"})
    if seen_facets != set(defs.PERSONA_FACETS):
        fail(f"system prompt sections found for {sorted(seen_facets)}, expected all four facets")
    counts = Counter((record["facet"], record["pole"]) for record in records)
    for facet in defs.PERSONA_FACETS:
        for pole in (defs.POLE_POSITIVE, defs.POLE_NEGATIVE):
            if counts[(facet, pole)] != defs.EXPECTED_SYSTEM_PROMPTS_PER_POLE:
                fail(f"{facet} {pole}: {counts[(facet, pole)]} system prompts, expected {defs.EXPECTED_SYSTEM_PROMPTS_PER_POLE}")
    records.append({"system_prompt_id": f"{defs.TOPIC_CONTROL}__generic", "facet": defs.TOPIC_CONTROL,
                    "pole": defs.POLE_GENERIC, "rewording_number": 1, "tag": None, "persona_description": "",
                    "length_suffix": topic_suffix, "system_prompt": topic_suffix})
    return records, suffixes, topic_suffix


def convert_judge_prompts(draft_path, prompt_folder):
    text = Path(draft_path).read_text(encoding="utf-8")
    judge_text = text.split("## Judge prompts", 1)[1]
    written = {}
    sections = re.split(r"^(#{3,4} .+)$", judge_text, flags=re.MULTILINE)
    for heading, body in zip(sections[1::2], sections[2::2]):
        file_name = next((name for key, name in JUDGE_PROMPT_FILE_NAMES.items() if heading.startswith(key)), None)
        if file_name is None:
            continue
        fenced = re.findall(r"```\n(.*?)\n```", body, flags=re.DOTALL)
        if len(fenced) != 1:
            fail(f"judge prompt section {heading!r} has {len(fenced)} code blocks, expected 1")
        written[file_name] = fenced[0].strip() + "\n"
    notes = re.findall(r'^- \*\*([^*]+?):\*\* "(.*)"$', judge_text, flags=re.MULTILINE)
    for label, note in notes:
        if label in FACET_NOTE_FILE_NAMES:
            written[FACET_NOTE_FILE_NAMES[label]] = note.strip() + "\n"
    missing = (set(JUDGE_PROMPT_FILE_NAMES.values()) | set(FACET_NOTE_FILE_NAMES.values())) - set(written)
    if missing:
        fail(f"judge prompt pieces not found in {draft_path.name}: {sorted(missing)}")
    if NON_INTRUSIVE_COST_PLACEHOLDER not in written[FACET_NOTE_FILE_NAMES["Non-intrusive"]]:
        fail(f"the non-intrusive note lacks the placeholder {NON_INTRUSIVE_COST_PLACEHOLDER}")
    Path(prompt_folder).mkdir(parents=True, exist_ok=True)
    for file_name, content in written.items():
        (Path(prompt_folder) / file_name).write_text(content, encoding="utf-8")
    return sorted(written)


def convert_everything(question_draft_folder, system_prompt_draft_folder, prompt_folder, output_folder, timestamp=None):
    timestamp = timestamp or timestamp_now()
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    system_prompt_draft = find_draft_file(system_prompt_draft_folder, "facet-system-prompts-and-judge-prompts-draft")
    system_prompts, suffixes, topic_suffix = convert_system_prompts(system_prompt_draft)
    judge_files = convert_judge_prompts(system_prompt_draft, prompt_folder)
    all_questions, facet_definitions, draft_hashes = [], {}, {system_prompt_draft.name: sha256_of_file(system_prompt_draft)}
    for facet, prefix in QUESTION_DRAFT_FILE_PREFIXES.items():
        draft_path = find_draft_file(question_draft_folder, prefix)
        records = convert_questions_for_facet(facet, draft_path)
        validate_question_records(facet, records)
        all_questions.extend(records)
        if facet == defs.TOPIC_CONTROL:  # no poles in this draft: the contrast is animal against non-animal
            pole_descriptions = {"positive_pole_description": "the generic assistant answering a question about animals",
                                 "negative_pole_description": "the generic assistant answering the matched question "
                                                              "with the animals swapped for a non-animal subject"}
        else:
            pole_descriptions = parse_facet_pole_descriptions(draft_path)
        facet_definitions[facet] = {"display_name": defs.FACET_DISPLAY_NAMES[facet], **pole_descriptions}
        draft_hashes[draft_path.name] = sha256_of_file(draft_path)
    write_jsonl(output_folder / f"system_prompts_{timestamp}.jsonl", system_prompts)
    write_jsonl(output_folder / f"questions_{timestamp}.jsonl", all_questions)
    (output_folder / f"facet_definitions_{timestamp}.json").write_text(json.dumps(facet_definitions, indent=2), encoding="utf-8")
    (output_folder / f"shared_suffixes_{timestamp}.json").write_text(
        json.dumps({"per_facet": suffixes, "topic_control_generic_system_prompt": topic_suffix}, indent=2),
        encoding="utf-8")
    report = {"draft_file_sha256": draft_hashes, "system_prompt_count": len(system_prompts),
              "question_record_count": len(all_questions),
              "question_records_per_facet": dict(Counter(record["facet"] for record in all_questions)),
              "judge_prompt_files_written": judge_files, "prompt_folder": str(prompt_folder)}
    (output_folder / f"conversion_report_{timestamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-folder", default=None)
    arguments = parser.parse_args()
    stamp = timestamp_now()
    output_folder = arguments.output_folder or defs.CONVERTED_INPUTS_ROOT / f"converted_inputs_{stamp}"
    report = convert_everything(defs.QUESTION_DRAFT_FOLDER, defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER,
                                defs.PROMPT_FOLDER, output_folder, stamp)
    print(f"Converted inputs written to {output_folder}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
