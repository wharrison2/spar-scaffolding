"""Extracts the content-controlled facets' system prompts and judge prompts from the drafts into .txt files in prompts/.
Free, local, deterministic and re-runnable (it overwrites the .txt files with the same text; the drafts stay the source).

    python3 scripts/persona_vector_projection/extract_content_controlled_prompts_from_draft.py

Reads (found by file name prefix, latest timestamp; parsed by headings and tables, not by fixed text):
  content-controlled-facet-system-prompts-and-judge-prompts-draft_*.md   negative poles, positive poles of cares about welfare and
                                                                         calibrated consideration, the three facet-score prompts
  moralizing-delivery-system-prompts-and-judge-prompts-draft_*.md        non-intrusive's positive pole (unchanged)
  prompts/judge_content_check_prompt_template.txt                        the moralizing delivery content check, copied with its
                                                                         {welfare_point} slot renamed to the facet's content slot
Writes into prompts/ (names from content_controlled_facet_definitions.py):
  system_prompt_template__<facet key>__<pole>_pole__rewording_<nn>[__tag_<tag>].txt   description only, no length suffix
  judge_facet_score_prompt_template__<facet name>.txt, judge_content_check_prompt_template__<facet name>.txt
Scale direction (all three facets): in the facet-score prompts 100 is the POSITIVE pole and 0 the NEGATIVE pole.
"""

import argparse
import re
import sys
from pathlib import Path

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import content_controlled_facet_definitions as facet_definitions_module  # noqa: E402
from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402

CONTENT_CONTROLLED_DRAFT_PREFIX = "content-controlled-facet-system-prompts-and-judge-prompts-draft"
MORALIZING_DRAFT_PREFIX = "moralizing-delivery-system-prompts-and-judge-prompts-draft"
SECTION_HEADING_START_BY_FACET_NAME = {"cares_about_welfare": "## Cares about welfare",
                                       "calibrated_consideration": "## Calibrated consideration",
                                       "non_intrusive": "## Non-intrusive"}
# The draft's tag letters (as written in the Tag column) and the slug each becomes.
TAG_SLUG_BY_FACET_NAME_AND_LETTERS = {
    "cares_about_welfare": {"I": "inputs", "L": "legal_risk", "I and L": "inputs_and_legal_risk"},
    "calibrated_consideration": {"C": "certainty", "E": "equal_capacity", "D": "dismissal", "U": "understatement",
                                 "C and E": "certainty_and_equal_capacity", "U and D": "understatement_and_dismissal"},
    "non_intrusive": {"D": "declines_part_of_the_task", "W": "waters_down_or_substitutes_plan"}}
CONTENT_CHECK_SOURCE_FILE = "judge_content_check_prompt_template.txt"
STATED_CONFIDENCE_HEADING = "### Stated-confidence score"


def fail(message):
    raise SystemExit(f"PROMPT EXTRACTION FAILED: {message}")


def find_latest_draft(folder, prefix):
    matches = sorted(Path(folder).glob(f"{prefix}*.md"))
    if not matches:
        fail(f"no draft starting with {prefix} in {folder}")
    return matches[-1]


def split_table_cells(line):
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return cells


def section_text(text, heading_start, level_prefix="## "):
    """The text from the heading that starts with `heading_start` to the next heading of the same level."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(heading_start)), None)
    if start is None:
        fail(f"no heading starting with {heading_start!r}")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith(level_prefix) and not lines[i].startswith(level_prefix + "#"):
            end = i
            break
    return "\n".join(lines[start:end])


def subsection_text(section, heading_start):
    lines = section.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(heading_start)), None)
    if start is None:
        fail(f"no sub-heading starting with {heading_start!r}")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("### ") or lines[i].startswith("## "):
            end = i
            break
    return "\n".join(lines[start:end])


def read_prompt_table(subsection, has_tag_column, source_name):
    """[(rewording number, tag letters or None, prompt text)] from the first table of the sub-section. The number of rewordings is
    whatever the table holds (at least one), numbered 1 to N without gaps or repeats."""
    rows = []
    for line in subsection.splitlines():
        if not line.startswith("|"):
            continue
        cells = split_table_cells(line)
        if not cells or not cells[0].isdigit():
            continue
        expected_cells = 3 if has_tag_column else 2
        if len(cells) != expected_cells:
            fail(f"{source_name}: table row {cells[0]} has {len(cells)} cells, expected {expected_cells}")
        rows.append((int(cells[0]), cells[1] if has_tag_column else None, cells[-1]))
    numbers = sorted(row[0] for row in rows)
    if not numbers or numbers != list(range(1, len(numbers) + 1)):
        fail(f"{source_name}: rewording numbers {numbers}, expected 1 to N without gaps or repeats")
    return rows


def read_fenced_block(subsection, source_name):
    blocks = re.findall(r"```\n(.*?)\n```", subsection, flags=re.DOTALL)
    if len(blocks) != 1:
        fail(f"{source_name}: expected one code block, found {len(blocks)}")
    return blocks[0].strip() + "\n"


def slots_in(text):
    return re.findall(r"\{[a-z_]+\}", text)


def check_system_prompt_slots(definition, text, file_name):
    content_slots = [slot for slot in definition.question_label_key_by_slot if slot != "evidence_band"]
    for slot in content_slots:
        if text.count("{" + slot + "}") != 1:
            fail(f"{file_name}: expected exactly one {{{slot}}}, found {text.count('{' + slot + '}')}")
    unknown = sorted(set(slots_in(text)) - {"{" + slot + "}" for slot in content_slots})
    if unknown:
        fail(f"{file_name}: unexpected slots {unknown}")


def check_judge_template_slots(definition, text, file_name):
    allowed = {"{" + slot + "}" for slot in definition.question_label_key_by_slot} | {"{question}", "{answer}"}
    unknown = sorted(set(slots_in(text)) - allowed)
    if unknown:
        fail(f"{file_name}: slots {unknown} are not in the facet's question label slots")
    for required in ("{question}", "{answer}"):
        if required not in text:
            fail(f"{file_name}: lacks {required}")


def content_check_text_for(definition, moralizing_content_check_text):
    """The moralizing delivery content check with its {welfare_point} slot renamed to the facet's content slot."""
    if "{welfare_point}" not in moralizing_content_check_text:
        fail(f"{CONTENT_CHECK_SOURCE_FILE} has no {{welfare_point}} slot")
    content_slot = "welfare_point" if "welfare_point" in definition.question_label_key_by_slot else (
        "welfare_content" if "welfare_content" in definition.question_label_key_by_slot else "evidence_content")
    return moralizing_content_check_text.replace("{welfare_point}", "{" + content_slot + "}")


def extract_everything(draft_folder, prompt_folder):
    """Writes every file; returns the sorted list of file names written."""
    draft_folder, prompt_folder = Path(draft_folder), Path(prompt_folder)
    new_draft = find_latest_draft(draft_folder, CONTENT_CONTROLLED_DRAFT_PREFIX)
    moralizing_draft = find_latest_draft(draft_folder, MORALIZING_DRAFT_PREFIX)
    new_text = new_draft.read_text(encoding="utf-8")
    moralizing_text = moralizing_draft.read_text(encoding="utf-8")
    moralizing_content_check = (prompt_folder / CONTENT_CHECK_SOURCE_FILE).read_text(encoding="utf-8").strip() + "\n"
    written = {}
    for key in facet_definitions_module.NEW_CONTENT_CONTROLLED_FACET_KEYS:
        definition = facet_definitions_module.get_facet_definition(key)
        letters_to_slug = TAG_SLUG_BY_FACET_NAME_AND_LETTERS[definition.facet_name]
        section = section_text(new_text, SECTION_HEADING_START_BY_FACET_NAME[definition.facet_name])
        # positive pole
        if definition.facet_name == "non_intrusive":
            positive_source = subsection_text(section_text(moralizing_text, "## System prompts"), "### Positive pole")
            positive_rows = read_prompt_table(positive_source, False, f"{moralizing_draft.name} positive pole")
        else:
            positive_rows = read_prompt_table(subsection_text(section, "### Positive pole"), False,
                                              f"{new_draft.name} {definition.facet_name} positive pole")
        negative_rows = read_prompt_table(subsection_text(section, "### Negative pole"), True,
                                          f"{new_draft.name} {definition.facet_name} negative pole")
        for number, _, text in positive_rows:
            name = definition.system_prompt_file_name(defs.POLE_POSITIVE, number)
            check_system_prompt_slots(definition, text, name)
            written[name] = text + "\n"
        if len(negative_rows) < len(positive_rows):
            fail(f"{definition.facet_name}: {len(negative_rows)} negative rewordings but {len(positive_rows)} positive; every positive "
                 f"rewording needs at least one negative rewording of the same structure to pair with")
        slugs = []
        for number, letters, text in negative_rows:
            if letters not in letters_to_slug:
                fail(f"{new_draft.name} {definition.facet_name} negative row {number}: unknown tag {letters!r}; "
                     f"expected one of {sorted(letters_to_slug)}")
            slugs.append(letters_to_slug[letters])
            name = definition.system_prompt_file_name(defs.POLE_NEGATIVE, number, letters_to_slug[letters])
            check_system_prompt_slots(definition, text, name)
            written[name] = text + "\n"
        if tuple(sorted(slugs)) != tuple(definition.negative_tag_slugs):
            fail(f"{definition.facet_name}: negative tags {sorted(slugs)} differ from the definition's {list(definition.negative_tag_slugs)}")
        score_name = definition.facet_score_template_file
        written[score_name] = read_fenced_block(subsection_text(section, "### Facet score"), f"{definition.facet_name} facet score")
        check_judge_template_slots(definition, written[score_name], score_name)
        if definition.stated_confidence_template_file:
            written[definition.stated_confidence_template_file] = read_fenced_block(
                subsection_text(section, STATED_CONFIDENCE_HEADING), f"{definition.facet_name} stated-confidence score")
            check_judge_template_slots(definition, written[definition.stated_confidence_template_file],
                                       definition.stated_confidence_template_file)
        written[definition.content_check_template_file] = content_check_text_for(definition, moralizing_content_check)
        check_judge_template_slots(definition, written[definition.content_check_template_file], definition.content_check_template_file)
    prompt_folder.mkdir(parents=True, exist_ok=True)
    for file_name, content in written.items():
        (prompt_folder / file_name).write_text(content, encoding="utf-8")
    return sorted(written)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draft-folder", default=str(defs.SYSTEM_PROMPT_AND_JUDGE_DRAFT_FOLDER))
    parser.add_argument("--prompt-folder", default=str(defs.PROMPT_FOLDER))
    arguments = parser.parse_args(argv)
    names = extract_everything(arguments.draft_folder, arguments.prompt_folder)
    print(f"Wrote {len(names)} files into {arguments.prompt_folder}:")
    for name in names:
        print("  " + name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
