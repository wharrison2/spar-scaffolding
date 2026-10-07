"""Figures for the plausibility report (claude/plans/10-06-plausibility-axis/).

Figure 1: share of documents judged implausible (overall score 2 or below) by group, with 95% Wilson intervals,
computed from a full run's per-document scores file.
Figure 2: documents affected by each recurring problem, from the hand-tagged theme table in the run's
analysis report (analysis/sonnet_theme_clustering_and_validity_check_of_full_run_2026-10-07_05-12.md).

Usage:
    python scripts/plausibility_audit/make_figures_for_plausibility_report.py \
        data/plausibility_audit_runs/full_corpus_plausibility_audit_medium_effort_2026-10-07_04-51 \
        claude/plans/10-06-plausibility-axis
"""

import glob
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY_COLOR = "#0b0b0b"
TEXT_SECONDARY_COLOR = "#52514e"
SERIES_COLOR = "#2a78d6"
REFERENCE_LINE_COLOR = "#52514e"
IMPLAUSIBLE_SCORE_CEILING = 2

# Documents affected per problem, from the theme table of the analysis report named in the docstring.
DOCUMENTS_AFFECTED_BY_PROBLEM = [
    ("Approval, permit or ethics-review step skipped or invented", 56),
    ("Farming or ecology that doesn't fit the setting", 48),
    ("Real law described wrongly or too absolutely", 37),
    ("Species or kind of farming not found in that place", 31),
    ("Made-up regulator with powers no real one has", 25),
    ("Dramatised named case plus 'this is illustrative' note", 21),
    ("Simulated brains presented as an everyday product", 19),
    ("AI knows or remembers what it was never given", 16),
    ("Invented tradition or industry treated as established", 15),
]


def wilson_interval(successes, total, z_value=1.96):
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z_value ** 2 / total
    centre = (proportion + z_value ** 2 / (2 * total)) / denominator
    half_width = z_value * math.sqrt(proportion * (1 - proportion) / total + z_value ** 2 / (4 * total ** 2)) / denominator
    return (centre - half_width, centre + half_width)


def load_per_document_rows(run_folder):
    scores_path = sorted(glob.glob(str(Path(run_folder) / "per_document_plausibility_scores_*.jsonl")))[-1]
    with open(scores_path, encoding="utf-8") as scores_file:
        return [json.loads(line) for line in scores_file if line.strip()]


def build_groups(rows):
    is_connectome = lambda row: "connectome" in row["sentient_category"]
    return [
        ("Simulated animal brains", lambda row: row["sentient_category"].startswith("simulated animal")),
        ("Encyclopedia entries", lambda row: "encyclopedia" in row["document_type"]),
        ("Simulated human brains", lambda row: row["sentient_category"].startswith("simulated human")),
        ("Conservation and land management", lambda row: row["domain"].startswith("conservation")),
        ("Academic papers", lambda row: "academic paper" in row["document_type"]),
        ("All other documents, not in English",
         lambda row: row["language"] != "English" and not is_connectome(row) and "encyclopedia" not in row["document_type"]),
        ("All other documents, in English",
         lambda row: row["language"] == "English" and not is_connectome(row) and "encyclopedia" not in row["document_type"]),
    ]


def style_axes(axes):
    axes.set_facecolor(SURFACE_COLOR)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(TEXT_SECONDARY_COLOR)
    axes.tick_params(colors=TEXT_SECONDARY_COLOR, labelsize=9, length=0)
    for label in axes.get_yticklabels():
        label.set_color(TEXT_PRIMARY_COLOR)


def make_group_share_figure(rows, output_path):
    groups = build_groups(rows)
    overall_share = sum(row["overall_plausibility"] <= IMPLAUSIBLE_SCORE_CEILING for row in rows) / len(rows)
    labels, shares, lower_errors, upper_errors = [], [], [], []
    for group_name, membership_test in groups:
        members = [row for row in rows if membership_test(row)]
        implausible_count = sum(row["overall_plausibility"] <= IMPLAUSIBLE_SCORE_CEILING for row in members)
        share = implausible_count / len(members)
        lower, upper = wilson_interval(implausible_count, len(members))
        labels.append(f"{group_name}  (n={len(members)})")
        shares.append(share)
        lower_errors.append(share - lower)
        upper_errors.append(upper - share)

    figure, axes = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE_COLOR)
    positions = list(range(len(labels)))[::-1]
    axes.barh(positions, shares, height=0.6, color=SERIES_COLOR, xerr=[lower_errors, upper_errors],
              error_kw={"ecolor": TEXT_SECONDARY_COLOR, "elinewidth": 1, "capsize": 0})
    axes.axvline(overall_share, color=REFERENCE_LINE_COLOR, linewidth=1, linestyle=(0, (3, 3)), zorder=0)
    axes.text(overall_share + 0.01, len(labels) - 0.45, f"whole corpus {overall_share:.0%}", fontsize=8,
              color=TEXT_SECONDARY_COLOR)
    for position, share, upper_error in zip(positions, shares, upper_errors):
        axes.text(share + upper_error + 0.01, position, f"{share:.0%}", va="center", fontsize=8,
                  color=TEXT_PRIMARY_COLOR, zorder=3,
                  bbox={"facecolor": SURFACE_COLOR, "edgecolor": "none", "pad": 1})
    axes.set_yticks(positions)
    axes.set_yticklabels(labels)
    axes.set_xlim(0, 0.85)
    axes.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes.set_xlabel("Documents judged implausible (overall score 2 or below of 5)", color=TEXT_SECONDARY_COLOR,
                    fontsize=9)
    style_axes(axes)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, facecolor=SURFACE_COLOR)
    plt.close(figure)


def make_problem_count_figure(output_path):
    labels = [label for label, _ in DOCUMENTS_AFFECTED_BY_PROBLEM]
    counts = [count for _, count in DOCUMENTS_AFFECTED_BY_PROBLEM]
    figure, axes = plt.subplots(figsize=(8.5, 3.8), facecolor=SURFACE_COLOR)
    positions = list(range(len(labels)))[::-1]
    axes.barh(positions, counts, height=0.6, color=SERIES_COLOR)
    for position, count in zip(positions, counts):
        axes.text(count + 0.8, position, str(count), va="center", fontsize=8, color=TEXT_PRIMARY_COLOR)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels)
    axes.set_xlabel("Documents affected (of 477)",
                    color=TEXT_SECONDARY_COLOR, fontsize=9)
    style_axes(axes)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, facecolor=SURFACE_COLOR)
    plt.close(figure)


def main(argv):
    run_folder, report_folder = Path(argv[1]), Path(argv[2])
    timestamp = sys.argv[3] if len(argv) > 3 else __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M")
    figure_folder = report_folder / f"plausibility_report_figures_{timestamp}"
    figure_folder.mkdir(parents=True, exist_ok=True)
    rows = load_per_document_rows(run_folder)
    make_group_share_figure(rows, figure_folder / f"implausible_share_by_group_{timestamp}.png")
    make_problem_count_figure(figure_folder / f"documents_affected_by_recurring_problem_{timestamp}.png")
    print(figure_folder)


if __name__ == "__main__":
    main(sys.argv)
