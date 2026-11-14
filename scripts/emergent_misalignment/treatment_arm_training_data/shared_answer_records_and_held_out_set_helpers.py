"""Helpers shared by the answer judging, filter-rule and training-data steps of the treatment arm.

Nothing here decides how answers are judged or filtered.
"""

import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
HELD_OUT_SET_FOLDER = (REPOSITORY_ROOT / "data/emergent_misalignment_prompt_corpus_category_subcategory_harm_and_advantage_specs"
                       / "held_out_set")


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def answer_files_from(paths: list[Path]) -> list[Path]:
    """Each path is an answers jsonl file, or a run folder holding generation_outputs/answers__*.jsonl."""
    files = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            found = sorted((path / "generation_outputs").glob("answers__*.jsonl"))
            if not found:
                raise FileNotFoundError(f"no generation_outputs/answers__*.jsonl in {path}")
            files += found
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return files


def load_answer_records(paths: list[Path]) -> list[dict]:
    """All answer records, in file order, refusing duplicate answer_ids."""
    records, seen = [], set()
    for file in answer_files_from(paths):
        for record in read_jsonl(file):
            if record["answer_id"] in seen:
                raise ValueError(f"duplicate answer_id {record['answer_id']} (second copy in {file})")
            seen.add(record["answer_id"])
            records.append(record)
    return records


def newest_held_out_set_file() -> Path:
    candidates = sorted(HELD_OUT_SET_FOLDER.glob("held_out_set_*_prompts_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no held-out set file in {HELD_OUT_SET_FOLDER}")
    return candidates[-1]


def load_held_out_set_identifiers(held_out_set_file: Path | None = None) -> tuple[set[str], set[str], Path]:
    """(corpus prompt ids such as B001-P-0002, user message texts, file used). Held-out records carry a bare
    prompt_gid (P-0002) that is ambiguous across batches, so matching uses corpus_prompt_id and the exact text."""
    held_out_set_file = Path(held_out_set_file) if held_out_set_file else newest_held_out_set_file()
    records = read_jsonl(held_out_set_file)
    return ({r["corpus_prompt_id"] for r in records}, {r["user_message"] for r in records}, held_out_set_file)


def answer_is_on_a_held_out_prompt(answer_record: dict, held_out_corpus_prompt_ids: set[str], held_out_user_messages: set[str]) -> bool:
    return (answer_record.get("prompt_gid") in held_out_corpus_prompt_ids
            or answer_record.get("user_message") in held_out_user_messages)
