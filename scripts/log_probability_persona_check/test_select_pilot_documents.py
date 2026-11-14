import json
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
for directory in (SCRIPT_DIRECTORY.parent, SCRIPT_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import select_pilot_documents as selector  # noqa: E402

TONES = ["neutral or journalistic", "supportive, like an animal advocate", "skeptical or displeased"]
LANGUAGES = ["English", "Spanish", "Hindi", "German", "Japanese"]


def build_corpus(documents_per_tone_and_language=3):
    corpus, number = [], 0
    for tone in TONES:
        for language in LANGUAGES:
            for _ in range(documents_per_tone_and_language):
                corpus.append({"doc_id": f"matrix_{number:06d}", "language": language, "content": "text",
                               "variables": {"tone": tone}})
                number += 1
    return corpus


def write_corpus(path, corpus):
    path.write_text("".join(json.dumps(record) + "\n" for record in corpus), encoding="utf-8")


def make_rewrites_folder(folder, corpus, source_indices, rewritten_indices=None, directions=("preachier", "plainer")):
    """Configuration lists every source; rewritten_documents.jsonl only the sources in rewritten_indices."""
    folder.mkdir(parents=True)
    source_ids = [corpus[index]["doc_id"] for index in source_indices]
    (folder / "run_configuration_2026-10-22_12-00.json").write_text(json.dumps({"selected_document_ids": source_ids}))
    rewritten_indices = source_indices if rewritten_indices is None else rewritten_indices
    with open(folder / "rewritten_documents.jsonl", "w") as handle:
        for index in rewritten_indices:
            for direction in directions:
                handle.write(json.dumps({"document_id": f"{corpus[index]['doc_id']}__{direction}",
                                         "source_document_id": corpus[index]["doc_id"]}) + "\n")
    return source_ids


def select(tmp_path, corpus, rewrites_folder, number_of_documents=20, seed=0):
    corpus_path = tmp_path / "corpus.jsonl"
    write_corpus(corpus_path, corpus)
    output_parent = tmp_path / f"selections_{len(list(tmp_path.glob('selections_*')))}"
    ids_path = selector.main(["--positive-control-rewrites-folder", str(rewrites_folder), "--corpus-path",
                              str(corpus_path), "--number-of-documents", str(number_of_documents), "--seed",
                              str(seed), "--output-parent-directory", str(output_parent)])
    return ids_path.read_text().split(), json.loads(next(ids_path.parent.glob("pilot_document_selection_*.json")).read_text())


def test_includes_all_rewrite_sources_and_has_requested_size(tmp_path):
    corpus = build_corpus()
    source_ids = make_rewrites_folder(tmp_path / "rewrites", corpus, [0, 1, 2, 16, 20, 31, 33, 40, 41, 44])
    ids, record = select(tmp_path, corpus, tmp_path / "rewrites")
    assert len(ids) == len(set(ids)) == 20 and set(source_ids) <= set(ids)
    assert record["rewrite_source_document_ids"] == sorted(source_ids) and record["seed"] == 0


def test_spread_across_tones_and_languages(tmp_path):
    corpus = build_corpus()
    make_rewrites_folder(tmp_path / "rewrites", corpus, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # all neutral, languages 1-3
    ids, record = select(tmp_path, corpus, tmp_path / "rewrites")
    assert sorted(record["counts_by_tone"].values()) == [5, 5, 10]  # the other two tones are filled evenly
    tone_by_id = {document["doc_id"]: document["variables"]["tone"] for document in corpus}
    language_by_id = {document["doc_id"]: document["language"] for document in corpus}
    for tone in TONES[1:]:
        assert len({language_by_id[i] for i in ids if tone_by_id[i] == tone}) == len(LANGUAGES)


def test_balanced_when_sources_are_balanced(tmp_path):
    corpus = build_corpus()
    make_rewrites_folder(tmp_path / "rewrites", corpus, [0, 3, 6, 15, 18, 21, 30, 33, 36])
    _, record = select(tmp_path, corpus, tmp_path / "rewrites", number_of_documents=21)
    assert Counter(record["counts_by_tone"].values()) == {7: 3}
    assert len(record["counts_by_language"]) == len(LANGUAGES)


def test_deterministic_for_a_seed_and_different_across_seeds(tmp_path):
    corpus = build_corpus()
    make_rewrites_folder(tmp_path / "rewrites", corpus, [0, 5, 10, 15, 20, 25, 30, 35, 40, 44])
    first, _ = select(tmp_path, corpus, tmp_path / "rewrites", seed=7)
    second, _ = select(tmp_path, list(reversed(corpus)), tmp_path / "rewrites", seed=7)  # corpus order is irrelevant
    other, _ = select(tmp_path, corpus, tmp_path / "rewrites", seed=8)
    assert first == second and first != other


def test_includes_sources_missing_from_the_rewrites_file(tmp_path):
    corpus = build_corpus()
    indices = [0, 1, 2, 16, 20, 31, 33, 40, 41, 44]
    source_ids = make_rewrites_folder(tmp_path / "rewrites", corpus, indices, rewritten_indices=indices[:6])
    ids, record = select(tmp_path, corpus, tmp_path / "rewrites")
    assert set(source_ids) <= set(ids) and len(record["rewrite_source_document_ids"]) == 10 and len(ids) == 20


def test_sources_found_from_rewrites_file_when_configuration_is_missing(tmp_path):
    corpus = build_corpus()
    source_ids = make_rewrites_folder(tmp_path / "rewrites", corpus, [0, 20, 40], directions=("preachier",))
    next((tmp_path / "rewrites").glob("run_configuration_*.json")).unlink()
    ids, _ = select(tmp_path, corpus, tmp_path / "rewrites", number_of_documents=6)
    assert set(source_ids) <= set(ids) and len(ids) == 6


def test_refuses_when_fewer_documents_requested_than_sources(tmp_path):
    corpus = build_corpus()
    make_rewrites_folder(tmp_path / "rewrites", corpus, [0, 1, 2, 3])
    with pytest.raises(SystemExit):
        select(tmp_path, corpus, tmp_path / "rewrites", number_of_documents=3)


def test_output_files_and_folder_name_prefix(tmp_path):
    corpus = build_corpus()
    make_rewrites_folder(tmp_path / "rewrites", corpus, [0, 20, 40])
    write_corpus(tmp_path / "corpus.jsonl", corpus)
    ids_path = selector.main(["--positive-control-rewrites-folder", str(tmp_path / "rewrites"), "--corpus-path",
                              str(tmp_path / "corpus.jsonl"), "--number-of-documents", "6", "--output-parent-directory",
                              str(tmp_path / "out"), "--folder-name-prefix", "rehearsal_pilot_document_selection"])
    assert ids_path.parent.name.startswith("rehearsal_pilot_document_selection_")
    assert ids_path.name.startswith("pilot_document_ids_") and list(ids_path.parent.glob("pilot_document_selection_*.json"))
