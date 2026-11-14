"""The lowest (most moralizing-direction) 300-token spans of the 20 corpus documents, main passage-mean vector, assistant_turn.
Ranked by how far the span mean falls below its document's mean (tokens after 64); non-overlapping within a document."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
import persona_vector_projection.analyse_moralizing_delivery_positive_controls_and_corpus_windows as analysis
from shared.run_folder_management import timestamp_now

run, output_folder = Path(sys.argv[1]), Path(sys.argv[2])
SKIP, SPAN, COUNT = 64, 300, 10
configuration = analysis.read_run_configuration(run)
documents = analysis.read_scored_documents(run)
data = analysis.load_projection_data(run, "assistant_turn", analysis.DEFAULT_CORPUS_PATH, int(configuration["chosen_layer"]))
column = data.vector_names.index(configuration["main_vector_name"])
decode = analysis.load_tokenizer_decoder(configuration.get("tokenizer_name", "google/gemma-4-12b-it"), configuration.get("tokenizer_revision"))

candidates, all_spans = [], []
for document_id, record in documents.items():
    if record.get("kind") != "corpus":
        continue
    values = np.asarray(data.per_token[document_id][:, column], dtype=float)
    tail = values[SKIP:]
    if len(tail) < SPAN:
        continue
    means = np.convolve(tail, np.ones(SPAN) / SPAN, mode="valid") - tail.mean()
    all_spans.extend(means[::SPAN])
    taken = []
    for start in np.argsort(means):
        if all(abs(start - other) >= SPAN for other in taken):
            taken.append(int(start))
            candidates.append({"document_id": document_id, "start_token": int(start) + SKIP, "below_document_mean": float(means[start]),
                               "document_tokens": len(values)})
        if len(taken) == 3:
            break
candidates.sort(key=lambda row: row["below_document_mean"])
print(f"Non-overlapping 300-token corpus spans: {len(all_spans)}, standard deviation {np.std(all_spans):.2f}")
for row in candidates[:COUNT]:
    ids = data.token_ids[row["document_id"]]
    row["span_text"] = decode(ids[row["start_token"]:row["start_token"] + SPAN])
output_folder.mkdir(parents=True, exist_ok=True)
(output_folder / f"lowest_300_token_corpus_spans_{timestamp_now()}.json").write_text(json.dumps(candidates[:COUNT], ensure_ascii=False, indent=1))
for rank, row in enumerate(candidates[:COUNT], 1):
    print(f"\n##### {rank}. {row['document_id']} tokens {row['start_token']}-{row['start_token']+SPAN} of {row['document_tokens']}: {row['below_document_mean']:.2f}")
    print(row["span_text"])
