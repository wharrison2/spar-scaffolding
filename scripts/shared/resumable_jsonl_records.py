"""Append-only JSONL files with a lock, and reading them back to find completed work for resuming."""

import json
import threading
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class AppendOnlyJsonlWriter:
    def __init__(self):
        self.lock = threading.Lock()

    def append(self, path, record):
        with self.lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_all_records(folder, file_name_pattern):
    records = []
    for path in sorted(Path(folder).glob(file_name_pattern)):
        records.extend(read_jsonl(path))
    return records


def find_completed_keys(folder, file_name_pattern, key_field_names):
    """Set of tuples of `key_field_names` values over every record already in the folder."""
    return {tuple(record[name] for name in key_field_names)
            for record in read_all_records(folder, file_name_pattern)}
