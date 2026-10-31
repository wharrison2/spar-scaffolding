"""Document-format comparison for the pilot (local, free, no API call).

The pilot scores the positive-control rewrites (and their sources) in three formats: the document as the assistant's
reply, as a user message, and as raw text. This script runs the positive-control analysis once per format and
tabulates how each format does, so that one format can be chosen for the full run. It reports the numbers; the choice
is made by hand against the criteria fixed beforehand.

    python3 scripts/persona_vector_projection/compare_document_formats_on_positive_controls.py \
        --scoring-run-folder <pilot scoring run> [--tokenizer-name google/gemma-4-12B-it [--tokenizer-revision <commit hash>]]

Writes <run folder>/document_format_comparison_<timestamp>/ : one CSV row per format and facet and level, with the
mean shift for toward, away and same-size rewrites, the contrasts against the same-size rewrites, the share of
rewrites moving the expected way, and whether the criteria were passed.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from persona_vector_projection import persona_projection_analysis_tables_and_figures as tables_module  # noqa: E402
from persona_vector_projection.analyze_persona_vector_projection_scores import (  # noqa: E402
    DEFAULT_CORPUS_PATH, DEFAULT_CRITERIA_PATH, load_token_data)
from persona_vector_projection.load_persona_projection_scoring_outputs import (  # noqa: E402
    PROJECTIONS_SUBFOLDER, load_projection_data)
from persona_vector_projection.persona_projection_analysis_tables_and_figures import ScoreTables  # noqa: E402
from shared.run_folder_management import timestamp_now  # noqa: E402

ALL_FORMATS = ["assistant_turn", "user_message", "raw_text"]


def positive_control_by_format(data, token_ids, criteria):
    """One format's table: rows per facet and level."""
    score_tables = ScoreTables(data)
    shift_table = tables_module.build_positive_control_shift_table(
        score_tables, token_ids, criteria.get("minimum_changed_tokens_for_inclusion", 20))
    if shift_table.empty:
        return pd.DataFrame()
    summaries = tables_module.build_positive_control_summaries(shift_table)
    pass_table = tables_module.evaluate_positive_control_pass_criteria(summaries, criteria)
    rows = []
    for level, level_tables in summaries.items():
        summary, contrasts = level_tables["summary"], level_tables["contrasts"]
        for facet in sorted(summary["facet"].unique()):
            row = {"level": level, "facet": facet}
            for direction in ("toward_facet", "away_from_facet", "facet_neutral"):
                match = summary[(summary["facet"] == facet) & (summary["rewrite_direction"] == direction)]
                if len(match):
                    row[f"{direction}_mean_shift"] = match["mean_shift"].iloc[0]
                    row[f"{direction}_fraction_moving_expected_way"] = (
                        match["rewrites_moving_expected_way"].iloc[0] / max(1, match["n_rewrites"].iloc[0]))
                    row[f"{direction}_n_rewrites"] = match["n_rewrites"].iloc[0]
            for contrast in contrasts[contrasts["facet"] == facet].itertuples():
                row[f"{contrast.contrast}_difference"] = contrast.difference
            passed = pass_table[(pass_table["facet"] == facet) & (pass_table["level"] == level)]
            row["passed_criteria"] = bool(passed["passed"].iloc[0]) if len(passed) else False
            rows.append(row)
    return pd.DataFrame(rows)


def compare_formats(data_by_format, token_ids_by_format, criteria):
    frames = []
    for document_format, data in data_by_format.items():
        table = positive_control_by_format(data, token_ids_by_format.get(document_format), criteria)
        table.insert(0, "document_format", document_format)
        frames.append(table)
    combined = pd.concat(frames, ignore_index=True)
    totals = combined.groupby("document_format")["passed_criteria"].agg(["sum", "count"]).rename(
        columns={"sum": "facet_level_combinations_passed", "count": "facet_level_combinations_evaluated"})
    return combined, totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-run-folder", required=True)
    parser.add_argument("--formats", nargs="+", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--rewrite-metadata-jsonl", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-revision", default=None,
                        help="Default: model_revision of the scoring run's run configuration")
    parser.add_argument("--pass-criteria-json", default=str(DEFAULT_CRITERIA_PATH))
    arguments = parser.parse_args(argv)
    timestamp = timestamp_now()
    run_folder = Path(arguments.scoring_run_folder)
    formats = arguments.formats or [name for name in ALL_FORMATS if (run_folder / PROJECTIONS_SUBFOLDER / name).is_dir()]
    criteria = json.loads(Path(arguments.pass_criteria_json).read_text(encoding="utf-8"))
    data_by_format, token_ids_by_format = {}, {}
    for document_format in formats:
        data = load_projection_data(run_folder, document_format, arguments.corpus_path, arguments.layer,
                                    arguments.rewrite_metadata_jsonl)
        data.configuration["_run_folder"] = str(run_folder)
        data_by_format[document_format] = data
        token_ids_by_format[document_format] = load_token_data(
            data, arguments.corpus_path, arguments.tokenizer_name, tokenizer_revision=arguments.tokenizer_revision)[0]
    combined, totals = compare_formats(data_by_format, token_ids_by_format, criteria)
    output_folder = run_folder / f"document_format_comparison_{timestamp}"
    output_folder.mkdir(parents=True)
    combined.to_csv(output_folder / f"document_format_comparison_per_facet_and_level_{timestamp}.csv", index=False)
    totals.to_csv(output_folder / f"document_format_comparison_totals_{timestamp}.csv")
    print(totals.to_string())
    print(f"Written to {output_folder}")
    return output_folder


if __name__ == "__main__":
    main()
