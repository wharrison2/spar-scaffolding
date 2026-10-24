"""Smoke test: build a 3-document run, then score it on CPU with a tiny randomly initialised Gemma 4 model that
uses the REAL Gemma 4 tokenizer and chat template (only tokenizer files are downloaded, about 32 MB; the 12B
weights are never downloaded). No paid API and no pod. Output goes to
data/log_probability_persona_check_runs/small_sample_checks/.

What it exercises: build_scoring_inputs.py, the real chat template (system prompt survives, document token
location on real corpus documents in several languages), the scorer's model loading through
AutoModelForCausalLM (which resolves to the multimodal Gemma 4 class), batching, chunked head with softcapping,
verification against plain full logits, atomic .npz outputs, completion records, and resume.

Usage: python3 scripts/log_probability_persona_check/smoke_test_log_probability_persona_check.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
for directory in (SCRIPTS_DIRECTORY, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import build_scoring_inputs as build_inputs  # noqa: E402
import score_document_log_probabilities_on_pod as scorer  # noqa: E402
import tiny_model_and_tokenizer_helpers_for_log_probability_persona_check as tiny  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402

TOKENIZER_REPOSITORY = "google/gemma-4-12B-it"
TOKENIZER_FILE_NAMES = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]
NUMBER_OF_SMOKE_TEST_DOCUMENTS = 3


def download_tokenizer_files_only(destination_directory):
    from huggingface_hub import hf_hub_download
    for file_name in TOKENIZER_FILE_NAMES:
        shutil.copy(hf_hub_download(TOKENIZER_REPOSITORY, file_name), destination_directory / file_name)


def choose_short_documents_in_different_languages(corpus_path, number_of_documents):
    shortest_first = sorted(read_jsonl(corpus_path), key=lambda record: len(record["content"]))
    chosen, languages_seen = [], set()
    for record in shortest_first:
        if record["language"] not in languages_seen:
            chosen.append(record["doc_id"])
            languages_seen.add(record["language"])
        if len(chosen) == number_of_documents:
            return chosen
    raise RuntimeError("Not enough languages in the corpus")


def main():
    from transformers import AutoTokenizer
    working_directory = Path(tempfile.mkdtemp(prefix="log_probability_smoke_test_"))
    try:
        tiny_model_directory = working_directory / "tiny_random_gemma4_with_real_gemma4_tokenizer"
        tiny_model_directory.mkdir()
        download_tokenizer_files_only(tiny_model_directory)
        tokenizer = AutoTokenizer.from_pretrained(tiny_model_directory)
        tiny.build_tiny_random_gemma4_model(len(tokenizer), final_logit_softcapping=30.0).save_pretrained(
            tiny_model_directory)
        tokenizer.save_pretrained(tiny_model_directory)
        print(f"Tiny model and tokenizer saved to {tiny_model_directory} (vocabulary {len(tokenizer)})")

        document_ids = choose_short_documents_in_different_languages(build_inputs.DEFAULT_CORPUS_PATH,
                                                                     NUMBER_OF_SMOKE_TEST_DOCUMENTS)
        document_ids_file = working_directory / "smoke_test_document_ids.txt"
        document_ids_file.write_text("\n".join(document_ids) + "\n")
        run_folder = build_inputs.main([
            "--run-name", "smoke_test_tiny_random_gemma4", "--small-sample-check",
            "--document-ids-file", str(document_ids_file), "--model-name", str(tiny_model_directory)])
        shutil.copy(document_ids_file, run_folder / "scoring_inputs" / "smoke_test_document_ids.txt")
        (run_folder / "smoke_test_note.txt").write_text(
            "Scored with a tiny RANDOMLY INITIALISED Gemma 4 model (4 layers, hidden size 32) saved temporarily "
            f"with the real tokenizer and chat template of {TOKENIZER_REPOSITORY}. The scores are meaningless; "
            "the run only checks the pipeline.\n")

        score_arguments = ["--run-folder", str(run_folder), "--device", "cpu", "--dtype", "float32",
                           "--maximum-tokens-per-batch", "40000"]
        scorer.main(score_arguments + ["--verify-first-document-against-full-logits"])

        # Resume: nothing left to do, and the model must not be loaded again.
        outputs = run_folder / "scoring_outputs"
        modification_times = {path.name: path.stat().st_mtime_ns
                              for path in (outputs / "per_token_log_probabilities").glob("*.npz")}
        scorer.main(score_arguments)
        assert modification_times == {path.name: path.stat().st_mtime_ns
                                      for path in (outputs / "per_token_log_probabilities").glob("*.npz")}

        conditions = read_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
        records = read_jsonl(outputs / "scoring_completion_records.jsonl")
        assert len(records) == NUMBER_OF_SMOKE_TEST_DOCUMENTS * len(conditions) == 186, len(records)
        print("\n=== SMOKE TEST SUMMARY ===")
        for document_id in document_ids:
            with np.load(outputs / "per_token_log_probabilities" / f"{document_id}.npz") as saved:
                log_probabilities = saved["log_probabilities"]
                assert log_probabilities.shape == (62, len(saved["document_token_ids"]))
                assert np.isfinite(log_probabilities).all()
                print(f"{document_id}: {len(saved['document_token_ids'])} document tokens, 62 conditions, "
                      f"mean log-probability {log_probabilities.mean():.3f}, first tokens "
                      f"{list(saved['document_token_strings'][:4])}")
        print(f"Run folder: {run_folder}")
        return run_folder
    finally:
        shutil.rmtree(working_directory, ignore_errors=True)


if __name__ == "__main__":
    main()
