"""Whole-document shift per arm (main vector, assistant_turn, tokens after 64) in units of the standard deviation of corpus documents' means."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
import persona_vector_projection.analyse_moralizing_delivery_positive_controls_and_corpus_windows as analysis
from shared.run_folder_management import timestamp_now

run, analysis_folder, output_folder = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
configuration = analysis.read_run_configuration(run)
documents = analysis.read_scored_documents(run)
data = analysis.load_projection_data(run, "assistant_turn", analysis.DEFAULT_CORPUS_PATH, int(configuration["chosen_layer"]))
column = data.vector_names.index(configuration["main_vector_name"])
corpus_means = [float(np.asarray(data.per_token[i][:, column])[64:].mean()) for i, r in documents.items() if r.get("kind") == "corpus"]
corpus_spread = float(np.std(corpus_means, ddof=1))
table = pd.read_csv(sorted(analysis_folder.glob("per_rewrite_results_*.csv"))[0])
table = table[(table.document_format == "assistant_turn") & (table.vector == configuration["main_vector_name"])]
summary = table.groupby("arm").agg(mean_whole_document_shift=("whole_document_shift_after_token_64", "mean"),
                                   rewrites_negative=("whole_document_shift_after_token_64", lambda s: int((s < 0).sum())))
summary["in_units_of_corpus_document_spread"] = summary.mean_whole_document_shift / corpus_spread
print(f"Corpus document means: standard deviation {corpus_spread:.2f}")
print(summary.round(2).to_string())
output_folder.mkdir(parents=True, exist_ok=True)
summary.to_csv(output_folder / f"whole_document_shift_in_units_of_corpus_document_spread_{timestamp_now()}.csv")
