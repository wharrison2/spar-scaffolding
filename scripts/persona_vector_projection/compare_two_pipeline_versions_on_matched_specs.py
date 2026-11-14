"""Paired comparison of two pipeline versions that were dealt the same specs (local, free, no API call).

A prompt change is tested by generating a document from each spec with the old and the new pipeline version and
comparing the two documents of each spec. Both documents are scored on the same frozen instrument (same vectors,
layer and document format), and each spec's baselines serve both documents, so differences between specs cancel.

Pairing: --pairs-jsonl with one {"first_document_id": ..., "second_document_id": ...} per spec, or, without it,
documents of the two runs are paired when their spec variables are identical and that combination is unique in each
run (pairs that cannot be made unambiguously are listed in the output, not guessed).

Baselines (--baseline-mode):
  shared_first  both documents are compared with the baselines of the FIRST version's document of the spec (default)
  own           each document is compared with the baselines scored in its own run

Writes <second run folder>/paired_version_comparison_<name>_<timestamp>/ with the per-pair differences, a summary per
facet (mean difference in standard deviations of the first version's corpus, 95% interval, Wilcoxon p) and, per spec
variable, the mean difference with the same too-few-documents marking as the main analysis.

    python3 scripts/persona_vector_projection/compare_two_pipeline_versions_on_matched_specs.py \
        --first-scoring-run-folder <old version run> --second-scoring-run-folder <new version run> \
        --document-format assistant_turn [--pairs-jsonl pairs.jsonl]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_projection_analysis_statistics as statistics_module  # noqa: E402
from persona_vector_projection import persona_projection_analysis_tables_and_figures as tables_module  # noqa: E402
from persona_vector_projection.load_persona_projection_scoring_outputs import load_projection_data  # noqa: E402
from persona_vector_projection.persona_projection_analysis_tables_and_figures import ScoreTables  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sdf_corpus.jsonl"
BASELINE_MODE_SHARED_FIRST = "shared_first"
BASELINE_MODE_OWN = "own"


def pair_documents_by_identical_specs(first_tables, second_tables, spec_variables):
    """(pairs DataFrame with first_document_id and second_document_id, DataFrame of unpaired spec combinations)."""
    def keyed(tables):
        documents = tables.data.documents.loc[tables.corpus_ids]
        keys = documents[spec_variables].astype(str).agg("||".join, axis=1)
        return keys
    first_keys, second_keys = keyed(first_tables), keyed(second_tables)
    first_unique = first_keys[~first_keys.duplicated(keep=False)]
    second_unique = second_keys[~second_keys.duplicated(keep=False)]
    second_by_key = pd.Series(second_unique.index, index=second_unique.values)
    pairs, unpaired = [], []
    for document_id, key in first_unique.items():
        if key in second_by_key.index:
            pairs.append({"first_document_id": document_id, "second_document_id": second_by_key[key]})
    paired_first = {pair["first_document_id"] for pair in pairs}
    for document_id in first_keys.index:
        if document_id not in paired_first:
            unpaired.append({"run": "first", "document_id": document_id})
    paired_second = {pair["second_document_id"] for pair in pairs}
    for document_id in second_keys.index:
        if document_id not in paired_second:
            unpaired.append({"run": "second", "document_id": document_id})
    return pd.DataFrame(pairs), pd.DataFrame(unpaired)


def paired_differences(first_tables, second_tables, pairs, facets, baseline_mode=BASELINE_MODE_SHARED_FIRST):
    """Per pair and facet: second document's score minus first document's, in standard deviations of the first
    version's corpus documents of the pair's language.

    shared_first: both raw scores subtract the first document's baselines (so the baseline cancels exactly);
    own: each subtracts its own run's baselines."""
    first_data, second_data = first_tables.data, second_tables.data
    rows = {}
    for pair in pairs.itertuples():
        first_id, second_id = pair.first_document_id, pair.second_document_id
        language = first_tables.languages[first_id]
        if baseline_mode == BASELINE_MODE_SHARED_FIRST:
            baseline = first_tables.baseline_means_primary.loc[first_id, facets]
            first_raw = first_data.mean_overall.loc[first_id, facets] if first_tables.primary_score == "overall" \
                else first_data.mean_after_opening.loc[first_id, facets]
            second_raw = second_data.mean_overall.loc[second_id, facets] if second_tables.primary_score == "overall" \
                else second_data.mean_after_opening.loc[second_id, facets]
            difference = (second_raw - baseline) - (first_raw - baseline)
        else:
            difference = second_tables.raw_scores.loc[second_id, facets] - first_tables.raw_scores.loc[first_id, facets]
        scale = pd.Series({facet: float(first_tables.language_statistics.loc[language, f"{facet}__sd"]) for facet in facets})
        rows[f"{first_id}|{second_id}"] = {"first_document_id": first_id, "second_document_id": second_id,
                                           "language": language, **(difference / scale).to_dict()}
    return pd.DataFrame(rows).T


def summarise_paired_differences(differences, facets, spec_variables, first_documents, minimum_documents):
    summary_rows, by_variable = [], []
    if differences.empty:
        return pd.DataFrame(), pd.DataFrame()
    for facet in facets:
        values = differences[facet].astype(float)
        interval = statistics_module.mean_with_t_interval(values)
        summary_rows.append({"facet": facet, "n_pairs": interval["n"], "mean_second_minus_first": interval["mean"],
                             "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
                             "wilcoxon_p_two_sided": statistics_module.safe_wilcoxon_p_value(values)})
        for variable in spec_variables:
            groups = first_documents.loc[differences["first_document_id"], variable].to_numpy()
            table = statistics_module.group_mean_table(values.to_numpy(), groups, minimum_documents)
            table.insert(0, "variable", variable)
            table.insert(0, "facet", facet)
            by_variable.append(table)
    return pd.DataFrame(summary_rows), (pd.concat(by_variable, ignore_index=True) if by_variable else pd.DataFrame())


def compare_versions(first_data, second_data, pairs=None, baseline_mode=BASELINE_MODE_SHARED_FIRST,
                     minimum_documents=15, spec_variables=None):
    """Returns dict with pairs, unpaired, differences, summary, by_variable."""
    if first_data.layer_index != second_data.layer_index or list(first_data.vector_names) != list(second_data.vector_names):
        raise ValueError("The two runs were scored with different layers or vectors; they are not on one instrument.")
    for key in ("model_name", "model_revision"):
        if first_data.configuration.get(key) != second_data.configuration.get(key):
            raise ValueError(f"The two runs differ in {key}: {first_data.configuration.get(key)!r} against "
                             f"{second_data.configuration.get(key)!r}")
    first_tables, second_tables = ScoreTables(first_data), ScoreTables(second_data)
    facets = tables_module.facets_present(first_data.vector_names)
    spec_variables = spec_variables or [c for c in first_data.documents.columns if c not in tables_module.STRUCTURAL_COLUMNS]
    unpaired = pd.DataFrame()
    if pairs is None:
        pairs, unpaired = pair_documents_by_identical_specs(first_tables, second_tables, spec_variables)
    differences = paired_differences(first_tables, second_tables, pairs, facets, baseline_mode)
    summary, by_variable = summarise_paired_differences(
        differences, facets, spec_variables, first_data.documents, minimum_documents)
    return {"pairs": pairs, "unpaired": unpaired, "differences": differences, "summary": summary,
            "by_variable": by_variable}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-scoring-run-folder", required=True)
    parser.add_argument("--second-scoring-run-folder", required=True)
    parser.add_argument("--document-format", default="assistant_turn")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--first-corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--second-corpus-path", default=None, help="Default: the same as --first-corpus-path")
    parser.add_argument("--pairs-jsonl", default=None)
    parser.add_argument("--baseline-mode", choices=[BASELINE_MODE_SHARED_FIRST, BASELINE_MODE_OWN],
                        default=BASELINE_MODE_SHARED_FIRST)
    parser.add_argument("--minimum-documents-per-value", type=int, default=15)
    parser.add_argument("--comparison-name", default="old_against_new_pipeline_version")
    arguments = parser.parse_args(argv)
    timestamp = timestamp_now()
    first = load_projection_data(arguments.first_scoring_run_folder, arguments.document_format,
                                 arguments.first_corpus_path, arguments.layer)
    second = load_projection_data(arguments.second_scoring_run_folder, arguments.document_format,
                                  arguments.second_corpus_path or arguments.first_corpus_path, arguments.layer)
    pairs = pd.DataFrame(read_jsonl(arguments.pairs_jsonl)) if arguments.pairs_jsonl else None
    results = compare_versions(first, second, pairs, arguments.baseline_mode, arguments.minimum_documents_per_value)
    output_folder = Path(arguments.second_scoring_run_folder) / (
        f"paired_version_comparison_{arguments.comparison_name}_{timestamp}")
    output_folder.mkdir(parents=True)
    for name, table in results.items():
        table.to_csv(output_folder / f"paired_comparison_{name}_{timestamp}.csv", index=name in ("differences",))
    (output_folder / f"paired_comparison_configuration_{timestamp}.json").write_text(
        json.dumps(vars(arguments), indent=2, default=str), encoding="utf-8")
    if results["pairs"].empty:
        print("No pairs could be made unambiguously from the spec variables; see the unpaired list, or pass --pairs-jsonl.")
    else:
        print(results["summary"].to_string(index=False))
    print(f"Written to {output_folder}")
    return output_folder


if __name__ == "__main__":
    main()
