"""Analysis of the persona vector projection scores (local, free, no API call).

    python3 scripts/persona_vector_projection/analyze_persona_vector_projection_scores.py \
        --scoring-run-folder <scoring pod run folder> --document-format assistant_turn \
        --depiction-labels-jsonl <depiction_labels_<ts>.jsonl> --rewrite-metadata-jsonl <facet_positive_control_rewrites_<ts>.jsonl> \
        --tokenizer-name google/gemma-4-12B-it [--tokenizer-revision <commit hash>]

Document score = the document's mean projection onto the facet vector minus the mean of its own matched baselines'
mean projections (a rewrite uses its source document's baselines), standardised within language using the corpus
documents of that language. Facets are analysed separately and never combined.

Writes <scoring run folder>/analysis_<name>_<timestamp>/ (tables as CSV, figures as PNG, extreme spans as CSV and
Markdown, analysis_summary_<timestamp>.md, analysis_configuration_<timestamp>.json). Span-level analyses use the token
ids the pod saved next to the per-token projections; for a run without them (older pods), --tokenizer-name makes the
analysis re-tokenise, and without it the changed-span analysis is skipped. The text of extreme spans needs
--tokenizer-name in either case (token ids are turned back into token strings). The tokenizer is downloaded at the
revision recorded as `model_revision` in the scoring run's run configuration (--tokenizer-revision overrides it); the
analysis stops if neither is available, because a later tokenizer commit can change how ids decode.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_vector_experiment_definitions as defs  # noqa: E402
from persona_vector_projection import persona_projection_analysis_tables_and_figures as tables_module  # noqa: E402
from persona_vector_projection.load_persona_projection_scoring_outputs import (  # noqa: E402
    check_token_counts_match, load_depiction_labels, load_projection_data, load_tokenizer_at_scoring_run_revision,
    tokenize_documents_as_the_scoring_pod_did)
from persona_vector_projection.persona_projection_analysis_tables_and_figures import ScoreTables  # noqa: E402
from shared.resumable_jsonl_records import read_jsonl  # noqa: E402
from shared.run_folder_management import sha256_of_file, timestamp_now  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIRECTORY.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "sdf_corpus.jsonl"
DEFAULT_CRITERIA_PATH = Path(__file__).resolve().parent / "positive_control_pass_criteria_draft.json"
DISPLAY_NAMES = defs.FACET_DISPLAY_NAMES


def write_table(table, folder, name, timestamp, index=False):
    path = Path(folder) / f"{name}_{timestamp}.csv"
    table.to_csv(path, index=index)
    return path


def load_token_data(data, corpus_path, tokenizer_name, tokenizer=None, tokenizer_revision=None):
    """Token ids and strings for every corpus and rewrite document that has per-token projections.

    Ids: the ones the pod saved (`per_token_token_ids_int32`) when present, so no re-tokenisation; otherwise recomputed
    with the pod's tokenisation function, after checking the counts against what the pod scored (needs a tokenizer).
    Strings: from a tokenizer (`convert_ids_to_tokens`) when one is available, else None. Returns (None, None) when
    there are no ids to be had (nothing saved and no tokenizer). The tokenizer is loaded at `tokenizer_revision`, else
    at the scoring run's recorded `model_revision`; neither available is an error."""
    wanted = [i for i in data.documents.index if data.documents.loc[i, "kind"] != "baseline" and i in data.per_token]
    saved = {i: [int(token_id) for token_id in data.token_ids[i]] for i in wanted if i in data.token_ids}
    if tokenizer is None and tokenizer_name and wanted:
        tokenizer = load_tokenizer_at_scoring_run_revision(tokenizer_name, tokenizer_revision, data.configuration)
    if len(saved) == len(wanted) and wanted:
        check_token_counts_match(saved, data.token_counts)
        strings = {i: tokenizer.convert_ids_to_tokens(ids) for i, ids in saved.items()} if tokenizer is not None else None
        return saved, strings
    if tokenizer is None:
        return None, None
    corpus_content = {record["doc_id"]: record["content"] for record in read_jsonl(corpus_path)}
    run_inputs = {record["document_id"]: record["content"] for record in
                  read_jsonl(Path(data.configuration["_run_folder"]) / "scoring_pod_inputs" / "documents.jsonl")}
    contents = {i: run_inputs.get(i, corpus_content.get(i)) for i in wanted if i not in saved}
    ids, strings = tokenize_documents_as_the_scoring_pod_did(
        contents, tokenizer, data.document_format, system_prompt=defs_system_prompt(), model_name=tokenizer_name)
    check_token_counts_match(ids, data.token_counts)
    for i, saved_ids in saved.items():
        ids[i] = saved_ids
        strings[i] = tokenizer.convert_ids_to_tokens(saved_ids)
    return ids, strings


def defs_system_prompt():
    from persona_vector_projection import scoring_pod_document_and_prompt_rendering as rendering
    return rendering.generic_assistant_system_prompt()


def run_analysis_on_loaded_data(data, labels, token_ids, token_strings, criteria, output_folder, timestamp,
                                primary_score="overall", spec_variables=None, other_layer_data=None,
                                window_tokens=32, windows_per_document=3, spans_per_direction=25,
                                figure_variables=None):
    """Computes every table and figure, writes them, and returns a dict of the main results (for tests and for
    the summary). `other_layer_data`: {layer: ProjectionData} for the neighbouring-layer agreement."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    figures_folder = output_folder / "figures"
    figures_folder.mkdir(exist_ok=True)
    score_tables = ScoreTables(data, primary_score)
    facets = tables_module.facets_present(data.vector_names)
    spec_variables = spec_variables or [c for c in data.documents.columns if c not in tables_module.STRUCTURAL_COLUMNS]
    figure_variables = figure_variables or tables_module.DEFAULT_SPEC_VARIABLES_FOR_FIGURES
    minimum_documents = criteria.get("minimum_documents_per_spec_variable_value", 15)
    results = {"score_tables": score_tables, "facets": facets}

    # Scores
    scores_wide = score_tables.raw_scores.add_prefix("raw_score__").join(
        score_tables.z_scores.add_prefix("z_score__")).join(score_tables.information).join(
        score_tables.scored_documents[["kind", "language", "facet", "rewrite_direction", "source_document_id"]])
    write_table(scores_wide, output_folder, "document_scores_raw_and_standardised_within_language", timestamp, index=True)
    write_table(score_tables.language_statistics, output_folder, "language_standardisation_statistics", timestamp,
                index=True)
    missing_baselines = score_tables.information[score_tables.information["number_of_baselines"] == 0]
    write_table(missing_baselines, output_folder, "documents_without_baselines", timestamp, index=True)
    results["documents_without_baselines"] = len(missing_baselines)

    # Distributions, spec variables, labels, positions, openings
    distributions = tables_module.build_score_distribution_tables(score_tables, facets)
    write_table(distributions, output_folder, "score_distributions_by_facet_and_language", timestamp)
    tables_module.plot_score_distributions(score_tables, facets, figures_folder / f"score_distributions_{timestamp}.png",
                                           DISPLAY_NAMES)
    by_variable = tables_module.build_scores_by_spec_variable(score_tables, facets, spec_variables, minimum_documents)
    write_table(by_variable, output_folder, "scores_by_spec_variable_with_confidence_intervals", timestamp)
    for facet in facets:
        for variable in figure_variables:
            tables_module.plot_scores_by_spec_variable(
                by_variable, facet, variable, figures_folder / f"{facet}_by_{variable}_{timestamp}.png", DISPLAY_NAMES)
    by_label = tables_module.build_scores_by_depiction_label(score_tables, labels, facets, minimum_documents)
    write_table(by_label, output_folder, "scores_by_depiction_label", timestamp)
    bands = tables_module.build_scores_by_position_band(score_tables, facets)
    write_table(bands, output_folder, "scores_by_position_band_and_opening", timestamp)
    if not bands.empty:
        tables_module.plot_scores_by_position(bands, figures_folder / f"scores_by_position_{timestamp}.png", DISPLAY_NAMES)
    write_table(tables_module.build_openings_table(score_tables, facets), output_folder,
                "openings_compared_with_rest_of_document", timestamp)
    matrix, pairs = tables_module.build_cross_facet_correlation(
        score_tables, facets, criteria.get("document_score_correlation_flag_threshold", 0.8))
    write_table(matrix, output_folder, "cross_facet_document_score_correlation", timestamp, index=True)
    write_table(pairs, output_folder, "cross_facet_document_score_correlation_pairs", timestamp)

    # Rank agreement
    write_table(tables_module.build_baseline_split_half_agreement(data, facets, primary_score), output_folder,
                "ranking_agreement_baseline_split_half", timestamp)
    write_table(tables_module.build_variant_ranking_agreement(score_tables, facets), output_folder,
                "ranking_agreement_vector_variants_halves_and_erasure", timestamp)
    if other_layer_data:
        other_tables = {layer: ScoreTables(other, primary_score) for layer, other in other_layer_data.items()}
        write_table(tables_module.build_neighbouring_layer_agreement(score_tables, other_tables, facets), output_folder,
                    "ranking_agreement_neighbouring_layers", timestamp)

    # Extreme spans
    if data.per_token:
        spans = tables_module.build_extreme_spans(score_tables, facets, token_strings, window_tokens,
                                                  windows_per_document, spans_per_direction)
        write_table(spans, output_folder, "extreme_spans", timestamp)
        tables_module.write_extreme_spans_markdown(output_folder / f"extreme_spans_for_hand_reading_{timestamp}.md",
                                                   spans, facets, DISPLAY_NAMES)
        results["extreme_spans"] = spans

    # Positive control
    shift_table = tables_module.build_positive_control_shift_table(
        score_tables, token_ids, criteria.get("minimum_changed_tokens_for_inclusion", 20))
    results["positive_control_shift_table"] = shift_table
    if not shift_table.empty:
        write_table(shift_table, output_folder, "positive_control_shift_per_rewrite", timestamp, index=True)
        summaries = tables_module.build_positive_control_summaries(shift_table)
        results["positive_control_summaries"] = summaries
        for level, level_tables in summaries.items():
            write_table(level_tables["summary"], output_folder, f"positive_control_{level}_summary", timestamp)
            write_table(level_tables["contrasts"], output_folder, f"positive_control_{level}_against_neutral_contrasts",
                        timestamp)
            tables_module.plot_positive_control(
                level_tables["summary"], figures_folder / f"positive_control_{level}_{timestamp}.png",
                title_for_level(level), DISPLAY_NAMES, "Shift in score (SDs within language), 95% interval")
        pass_table = tables_module.evaluate_positive_control_pass_criteria(summaries, criteria)
        write_table(pass_table, output_folder, "positive_control_pass_or_fail", timestamp)
        results["positive_control_pass_or_fail"] = pass_table
        write_table(tables_module.build_positive_control_specificity(shift_table), output_folder,
                    "positive_control_shift_on_every_facet", timestamp, index=True)
        write_table(tables_module.build_positive_control_by_language(shift_table), output_folder,
                    "positive_control_by_language", timestamp)
    write_summary_markdown(output_folder, timestamp, results, data)
    return results


def title_for_level(level):
    return {"whole_document": "Positive control, whole document: rewrite minus its own source",
            "changed_span": "Positive control, changed span: changed tokens minus the source tokens they replaced"}[level]


def write_summary_markdown(output_folder, timestamp, results, data):
    """Short summary: methods, then the positive control first, then pointers to the tables. No statistics beyond the
    pass or fail table; the figures and CSV files carry the numbers."""
    lines = ["# Persona vector projection: analysis summary", "", "## Method", "",
             f"Scores come from {len(results['score_tables'].corpus_ids)} corpus documents scored at layer "
             f"{data.layer_index} in the `{data.document_format}` format. A document's score is its mean projection "
             "onto a facet vector minus the mean projection of the baseline documents generated for its own topic "
             "(a rewrite uses its source document's baselines), expressed in standard deviations of the corpus "
             "documents of the same language. Each facet is analysed separately.", "",
             f"Documents without any baseline: {results['documents_without_baselines']}.", "",
             "## Positive control", ""]
    pass_table = results.get("positive_control_pass_or_fail")
    if pass_table is None or pass_table.empty:
        lines.append("No rewrites were scored, so the positive control was not evaluated.")
    else:
        lines += ["Whole-document and changed-span results are reported separately. A facet passes at a level when "
                  "its toward and away rewrites move the expected way, with clustered intervals excluding zero, and "
                  "differ from the same-size rewrites that leave the facet alone.", "",
                  "| Facet | Level | Passed |", "|---|---|---|"]
        for row in pass_table.itertuples():
            lines.append(f"| {DISPLAY_NAMES.get(row.facet, row.facet)} | {row.level.replace('_', ' ')} | "
                         f"{'yes' if row.passed else 'no'} |")
    lines += ["", "## Where the rest is", "",
              "Score distributions, scores by spec variable (with 95% intervals; values with too few documents are "
              "marked and not ranked), scores by depiction label, by position and for openings, cross-facet "
              "correlations, ranking agreement and extreme spans are in the CSV files and figures in this folder.", ""]
    (Path(output_folder) / f"analysis_summary_{timestamp}.md").write_text("\n".join(lines), encoding="utf-8")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-run-folder", required=True)
    parser.add_argument("--document-format", default="assistant_turn", choices=["assistant_turn", "user_message", "raw_text", "assistant_turn_with_real_user_message"])
    parser.add_argument("--layer", type=int, default=None, help="Default: the middle recorded layer (the chosen one)")
    parser.add_argument("--primary-score", choices=["overall", "after_opening"], default="overall")
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--depiction-labels-jsonl", default=None)
    parser.add_argument("--rewrite-metadata-jsonl", default=None)
    parser.add_argument("--tokenizer-name", default=None, help="Hugging Face tokenizer, for span-level analyses")
    parser.add_argument("--tokenizer-revision", default=None,
                        help="Tokenizer revision (commit hash). Default: model_revision of the scoring run's run configuration")
    parser.add_argument("--pass-criteria-json", default=str(DEFAULT_CRITERIA_PATH))
    parser.add_argument("--analysis-name", default="persona_vector_projection_analysis")
    parser.add_argument("--window-tokens", type=int, default=32)
    parser.add_argument("--compare-neighbouring-layers", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    timestamp = timestamp_now()
    data = load_projection_data(arguments.scoring_run_folder, arguments.document_format, arguments.corpus_path,
                                arguments.layer, arguments.rewrite_metadata_jsonl)
    data.configuration["_run_folder"] = str(arguments.scoring_run_folder)
    labels = load_depiction_labels(arguments.depiction_labels_jsonl)
    criteria = json.loads(Path(arguments.pass_criteria_json).read_text(encoding="utf-8"))
    token_ids, token_strings = load_token_data(data, arguments.corpus_path, arguments.tokenizer_name,
                                                tokenizer_revision=arguments.tokenizer_revision)
    other_layers = None
    if arguments.compare_neighbouring_layers:
        recorded = data.configuration.get("recorded_layers", [])
        other_layers = {layer: load_projection_data(arguments.scoring_run_folder, arguments.document_format,
                                                    arguments.corpus_path, layer, arguments.rewrite_metadata_jsonl)
                        for layer in recorded if layer != data.layer_index}
    output_folder = Path(arguments.scoring_run_folder) / f"analysis_{arguments.analysis_name}_{timestamp}"
    configuration = {"arguments": vars(arguments), "layer_index": data.layer_index, "criteria": criteria,
                     "criteria_sha256": sha256_of_file(arguments.pass_criteria_json),
                     "span_level_analysis": token_ids is not None}
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / f"analysis_configuration_{timestamp}.json").write_text(json.dumps(configuration, indent=2, default=str),
                                                                            encoding="utf-8")
    run_analysis_on_loaded_data(data, labels, token_ids, token_strings, criteria, output_folder, timestamp,
                                arguments.primary_score, other_layer_data=other_layers,
                                window_tokens=arguments.window_tokens)
    print(f"Analysis written to {output_folder}")
    return output_folder


if __name__ == "__main__":
    main()
