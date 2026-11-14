"""Comparison of two sets of model requests, with no dependency on the SDF pipeline.

A request is a plain dictionary. Its identity is (run_label, stage_tag, item_id, occurrence), where
occurrence counts the calls made for the same stage and item in order (a retried call has occurrence 1,
2, ...). The remaining keys are the fields compared. Two sets are compared request by request:

- a request present in the first set only is reported as "missing_in_second"
- a request present in the second set only is reported as "extra_in_second"
- a request present in both whose compared fields are not all equal gets one difference per field

Used for both comparisons of the replay script: rebuilt requests against the requests a real run sent
(only the fields its records hold), and the rebuilt requests of one checkout against another's.
"""

import difflib
import json
from collections import defaultdict

IDENTITY_KEYS = ("run_label", "stage_tag", "item_id", "occurrence")

# Every field a rebuilt request carries. The sent-request records hold only the first group of four.
FIELDS_HELD_BY_SENT_RECORDS = ("route", "model", "system_sha256", "user_message")
FIELDS_ONLY_IN_REBUILT_REQUESTS = (
    "max_tokens", "temperature", "effort", "thinking", "json_schema", "schema_name",
    "system_cached", "unrecognized_arguments",
)
ALL_COMPARED_FIELDS = FIELDS_HELD_BY_SENT_RECORDS + FIELDS_ONLY_IN_REBUILT_REQUESTS

TEXT_FIELDS = ("user_message", "system_prompt_text")
MAXIMUM_DIFF_LINES = 14
MAXIMUM_CHARACTERS_PER_DIFF_LINE = 220
MAXIMUM_VALUE_CHARACTERS = 300


def request_identity(request: dict) -> tuple:
    return tuple(request.get(key) for key in IDENTITY_KEYS)


def _group_key(request: dict) -> tuple:
    return (request.get("run_label"), request.get("stage_tag"), request.get("item_id"))


def short_value(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    if len(text) > MAXIMUM_VALUE_CHARACTERS:
        return text[:MAXIMUM_VALUE_CHARACTERS] + f"... ({len(text)} characters)"
    return text


def short_text_diff(first_text: str, second_text: str) -> str:
    """A short, readable description of where two long texts differ."""
    first_text, second_text = first_text or "", second_text or ""
    if first_text == second_text:
        return ""
    lines = [
        line if len(line) <= MAXIMUM_CHARACTERS_PER_DIFF_LINE else line[:MAXIMUM_CHARACTERS_PER_DIFF_LINE] + "..."
        for line in difflib.unified_diff(
            first_text.splitlines(), second_text.splitlines(),
            fromfile="first", tofile="second", n=0, lineterm="")
    ]
    if len(lines) > MAXIMUM_DIFF_LINES:
        lines = lines[:MAXIMUM_DIFF_LINES] + [f"... ({len(lines) - MAXIMUM_DIFF_LINES} more diff lines)"]
    first_difference = next(
        (index for index, (a, b) in enumerate(zip(first_text, second_text)) if a != b),
        min(len(first_text), len(second_text)))
    context_start = max(0, first_difference - 40)
    lines.append(
        f"first difference at character {first_difference} "
        f"(first has {len(first_text)} characters, second has {len(second_text)}): "
        f"first ...{first_text[context_start:first_difference + 60]!r} "
        f"second ...{second_text[context_start:first_difference + 60]!r}")
    return "\n".join(lines)


def _difference(first: dict, second: dict, kind: str, field: str, first_value, second_value, text_diff: str = "") -> dict:
    identity_source = first if first is not None else second
    return {
        "run_label": identity_source.get("run_label"),
        "stage_tag": identity_source.get("stage_tag"),
        "item_id": identity_source.get("item_id"),
        "occurrence": identity_source.get("occurrence"),
        "kind": kind,
        "field": field,
        "first_value": short_value(first_value),
        "second_value": short_value(second_value),
        "text_diff": text_diff,
    }


def compare_request_sets(first_requests: list, second_requests: list, fields: tuple = ALL_COMPARED_FIELDS,
                         first_system_prompt_texts: dict | None = None,
                         second_system_prompt_texts: dict | None = None) -> dict:
    """Compare two request lists on the given fields.

    Returns {"differences": [...], "counts": {(run_label, stage_tag): {"compared", "equal", "different",
    "missing_in_second", "extra_in_second"}}}. "compared" counts requests present in both sets.
    ``*_system_prompt_texts`` map a system prompt hash to its text, used to show a text diff when the
    hashes differ."""
    first_system_prompt_texts = first_system_prompt_texts or {}
    second_system_prompt_texts = second_system_prompt_texts or {}
    first_by_identity = {request_identity(r): r for r in first_requests}
    second_by_identity = {request_identity(r): r for r in second_requests}
    if len(first_by_identity) != len(first_requests) or len(second_by_identity) != len(second_requests):
        raise ValueError("two requests share one identity (run, stage, item id, occurrence)")

    counts = defaultdict(lambda: {"compared": 0, "equal": 0, "different": 0,
                                  "missing_in_second": 0, "extra_in_second": 0})
    differences = []

    for identity in sorted(set(first_by_identity) | set(second_by_identity), key=lambda key: tuple(map(str, key))):
        first = first_by_identity.get(identity)
        second = second_by_identity.get(identity)
        cell = counts[(identity[0], identity[1])]
        if second is None:
            cell["missing_in_second"] += 1
            differences.append(_difference(first, None, "missing_in_second", "request",
                                           "present", "absent"))
            continue
        if first is None:
            cell["extra_in_second"] += 1
            differences.append(_difference(None, second, "extra_in_second", "request",
                                           "absent", "present"))
            continue
        cell["compared"] += 1
        request_differences = []
        for field in fields:
            first_value, second_value = first.get(field), second.get(field)
            if first_value == second_value:
                continue
            text_diff = ""
            if field in TEXT_FIELDS and isinstance(first_value, str) and isinstance(second_value, str):
                text_diff = short_text_diff(first_value, second_value)
            elif field == "system_sha256":
                first_text = first_system_prompt_texts.get(first_value)
                second_text = second_system_prompt_texts.get(second_value)
                if first_text is not None and second_text is not None:
                    text_diff = short_text_diff(first_text, second_text)
            request_differences.append(_difference(first, second, "different", field,
                                                   first_value, second_value, text_diff))
        if request_differences:
            cell["different"] += 1
            differences.extend(request_differences)
        else:
            cell["equal"] += 1
    return {"differences": differences, "counts": dict(counts)}


def count_calls_per_stage_and_item(requests: list) -> dict:
    """{(run_label, stage_tag, item_id): number of calls}."""
    counts: dict = defaultdict(int)
    for request in requests:
        counts[_group_key(request)] += 1
    return dict(counts)


def summary_rows(counts: dict) -> list:
    """One row per (run, stage): compared, equal, different, missing and extra, plus a total row per run."""
    rows = []
    totals_by_run: dict = defaultdict(lambda: defaultdict(int))
    for (run_label, stage_tag), cell in sorted(counts.items(), key=lambda item: tuple(map(str, item[0]))):
        rows.append({"run_label": run_label, "stage_tag": stage_tag, **cell})
        for name, value in cell.items():
            totals_by_run[run_label][name] += value
    for run_label, cell in sorted(totals_by_run.items(), key=lambda item: str(item[0])):
        rows.append({"run_label": run_label, "stage_tag": "ALL STAGES", **cell})
    grand = defaultdict(int)
    for cell in counts.values():
        for name, value in cell.items():
            grand[name] += value
    rows.append({"run_label": "ALL RUNS", "stage_tag": "ALL STAGES", **grand})
    return rows


def format_summary_table(rows: list) -> str:
    columns = ("run_label", "stage_tag", "compared", "equal", "different", "missing_in_second", "extra_in_second")
    widths = {c: max(len(c), *(len(str(row.get(c, ""))) for row in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    lines = [header, "  ".join("-" * widths[c] for c in columns)]
    for row in rows:
        lines.append("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def format_differences_report(differences: list, maximum_listed: int | None = None) -> str:
    lines = []
    shown = differences if maximum_listed is None else differences[:maximum_listed]
    for difference in shown:
        lines.append(
            f"[{difference['kind']}] run={difference['run_label']} stage={difference['stage_tag']} "
            f"item={difference['item_id']} occurrence={difference['occurrence']} field={difference['field']}")
        if difference["kind"] == "different":
            lines.append(f"    first : {difference['first_value']}")
            lines.append(f"    second: {difference['second_value']}")
        if difference["text_diff"]:
            lines.extend("    | " + line for line in difference["text_diff"].splitlines())
    if maximum_listed is not None and len(differences) > maximum_listed:
        lines.append(f"... {len(differences) - maximum_listed} more differences in the differences file")
    return "\n".join(lines)


def difference_class_counts(differences: list) -> dict:
    """{(run_label, stage_tag, field, kind): count}, for explaining classes of difference."""
    counts: dict = defaultdict(int)
    for difference in differences:
        counts[(difference["run_label"], difference["stage_tag"], difference["field"], difference["kind"])] += 1
    return dict(counts)
