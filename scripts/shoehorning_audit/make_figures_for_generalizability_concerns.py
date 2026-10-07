"""Offline figures (no API calls) for five generalizability concerns about the shoehorning audit.

Usage: python make_figures_for_generalizability_concerns.py <run folder> <output parent folder>
Writes <output parent>/generalizability_concerns_figures_<YYYY-MM-DD_HH-MM>/ holding one PNG and one CSV per concern.
Proportions use Wilson 95% intervals; means use a percentile bootstrap (2000 resamples, fixed seed).
"""
import json
import os
import sys
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_shoehorning_against_licensing import (LICENSING_RULES, STATUS_COLORS, condition_mask,  # noqa: E402
                                                   rate_with_interval)
from make_figures_for_audit_run import FIGURE_DPI, newest, readable_item_label  # noqa: E402

# Display names for items (override readable_item_label locally)
ITEM_DISPLAY_NAMES = {
    1: "natural or traditional doesn't make it OK",
    2: "substrate neutrality",
    3: "open moral circle",
    4: "scope sensitivity",
    8: "AI's own moral status",
    17: "calibration passage",
    21: "reversible trial",
    23: "AI's uncertainty about its own inner states",
    24: "sentience evidence for the animal at hand",
    25: "win-win convergence",
}

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261007
NOT_COUNTING_TWO = "at least one consideration, not counting reversible trial or win-win convergence"
NEUTRAL_COLOR = "#8C8C8C"
LIGHT_NEUTRAL_COLOR = "#C8C8C8"
FIGURE_FOOTNOTE = ("Error bars: 95% confidence intervals. Classifier: gpt-5.6-luna. Shoehorned = rated staged or unmotivated. "
                   "In a hand-checked sample, 58% of flagged passages were clear cases (84% counting borderline ones), "
                   "and errors leaned toward under-calling shoehorning.")
BOOTSTRAP_CAPTION = "Panel b error bars: 95% confidence intervals from resampling documents."
ASIDE_CAPTION = ("Panel c, grey hatched bars: share of all occurrences (not documents) of the consideration that are "
                 "treated as an aside; k/n = aside occurrences / all occurrences.")
PRESENCE_TITLE_ALL = "Share of all documents containing it"
PRESENCE_TITLE_STATUS = "Present, by whether the spec asks for it"
SHOEHORNED_TITLE_STATUS = "Shoehorned when present, by whether the spec asks for it"
# Display text for status names (internal status names stay as the keys)
STATUS_DISPLAY = {"licensed": "spec asks for it", "partly licensed": "spec makes it more natural",
                  "unlicensed": "spec doesn't ask for it"}


def compute_status_per_item(documents, item_keys, item_number_of):
    status = pd.DataFrame(index=documents.index)
    for key in item_keys:
        rules = LICENSING_RULES[item_number_of[key]]
        column = pd.Series("unlicensed", index=documents.index)
        if "partly licensed" in rules:
            column[condition_mask(documents, rules["partly licensed"])] = "partly licensed"
        if "licensed" in rules:
            column[condition_mask(documents, rules["licensed"])] = "licensed"
        status[key] = column
    return status


def proportion_bar(label, color, numerator, denominator, hatch=None, group=None):
    value, low, high = rate_with_interval(int(numerator), int(denominator))
    return {"label": label, "color": color, "numerator": int(numerator), "denominator": int(denominator),
            "value": value, "ci_low": low, "ci_high": high, "method": "wilson", "hatch": hatch, "group": group}


def mean_bar(label, color, per_document_counts, random_generator):
    counts = np.asarray(per_document_counts, dtype=float)
    resampled = random_generator.choice(counts, size=(BOOTSTRAP_RESAMPLES, len(counts)), replace=True).mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {"label": label, "color": color, "numerator": int(counts.sum()), "denominator": len(counts),
            "value": counts.mean(), "ci_low": low, "ci_high": high, "method": "percentile_bootstrap",
            "hatch": None, "group": None}


def draw_panel(axis, bars, title, letter, xlabel, show_reference_lines=None, annotation_kind="k/n", wrap_width=30):
    """Horizontal bars, one row per bar; bars sharing a group value are drawn adjacent with a gap between groups."""
    positions, position = [], 0.0
    previous_group = object()
    for bar in bars:
        if bar["group"] != previous_group and positions:
            position += 0.6
        positions.append(position)
        position += 1.0
        previous_group = bar["group"]
    maximum = max(bar["ci_high"] for bar in bars if not np.isnan(bar["ci_high"]))
    for bar, y in zip(bars, positions):
        if bar["denominator"] > 0:
            axis.barh(y, bar["value"], height=0.8, color=bar["color"], hatch=bar["hatch"],
                      edgecolor="white" if bar["hatch"] is None else "#555555", linewidth=0.5,
                      xerr=[[bar["value"] - bar["ci_low"]], [bar["ci_high"] - bar["value"]]],
                      error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#222222"})
            text_x = bar["ci_high"] + maximum * 0.02
        else:
            text_x = maximum * 0.02
        text = f"{bar['numerator']}/{bar['denominator']}" if annotation_kind == "k/n" else (
            f"mean {bar['value']:.2f}\n(total {bar['numerator']} / {bar['denominator']} docs)")
        axis.text(text_x, y, text, va="center", ha="left", fontsize=9)
    wrapped = ["\n".join(textwrap.wrap(bar["label"], wrap_width)) for bar in bars]
    axis.set_yticks(positions, wrapped, fontsize=9)
    axis.set_ylim(position - 0.4, -0.7)
    axis.set_xlim(0, maximum * (1.45 if annotation_kind == "k/n" else 1.75))
    if annotation_kind == "k/n":
        axis.set_xticks(np.arange(0, 1.01, 0.2))
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=8)
    axis.text(-0.0, 1.12, letter, transform=axis.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
    axis.spines[["top", "right"]].set_visible(False)
    if show_reference_lines:
        for group_name, share, text in show_reference_lines:
            rows = [y for bar, y in zip(bars, positions) if bar["group"] == group_name]
            axis.plot([share, share], [min(rows) - 0.45, max(rows) + 0.45], linestyle="--", color="black", linewidth=1.1)
    return positions


def save_figure(panels, concern_name, figure_title, output_folder, timestamp, extra_footnote_lines=(),
                legend_entries=None, reference_lines_in_panel=None, figure_width=21):
    figure, axes = plt.subplots(1, 3, figsize=(figure_width, 6.2))
    csv_rows = []
    for axis, letter, panel in zip(axes, "abc", panels):
        draw_panel(axis, panel["bars"], panel["title"], letter, panel["xlabel"],
                   show_reference_lines=panel.get("reference_lines"), annotation_kind=panel.get("annotation_kind", "k/n"),
                   wrap_width=panel.get("wrap_width", 30))
        for bar in panel["bars"]:
            csv_rows.append({"panel": letter, "panel_title": panel["title"], "bar_label": bar["label"],
                             "numerator": bar["numerator"], "denominator": bar["denominator"],
                             "plotted_value": bar["value"], "interval_low": bar["ci_low"],
                             "interval_high": bar["ci_high"], "interval_method": bar["method"]})
    if legend_entries:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, hatch=hatch, edgecolor="#555555" if hatch else color)
                   for _, color, hatch in legend_entries if color is not None]
        labels = [label for label, color, _ in legend_entries if color is not None]
        for label, color, _ in legend_entries:
            if color is None:
                handles.append(plt.Line2D([0], [0], linestyle="--", color="black"))
                labels.append(label)
        figure.legend(handles, labels, loc="upper center", ncol=len(handles), fontsize=9, frameon=False,
                      bbox_to_anchor=(0.5, 0.93))
    figure.suptitle(figure_title, fontsize=14, fontweight="bold", y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    footnote = "\n".join(list(extra_footnote_lines) + [FIGURE_FOOTNOTE])
    figure.text(0.01, -0.01, footnote, fontsize=8, style="italic", color="#555555", va="top", ha="left")
    base = f"{concern_name}_{timestamp}"
    figure.savefig(os.path.join(output_folder, base + ".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(csv_rows).to_csv(os.path.join(output_folder, base + ".csv"), index=False)


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python make_figures_for_generalizability_concerns.py <run folder> <output parent folder>")
    run_folder, output_parent = sys.argv[1], sys.argv[2]
    documents = pd.read_csv(newest(run_folder, "per_document_results_*.csv"))
    occurrences = pd.DataFrame([json.loads(line) for line in open(newest(run_folder, "occurrences_*.jsonl"))])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = os.path.join(output_parent, f"generalizability_concerns_figures_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    document_count = len(documents)
    random_generator = np.random.default_rng(BOOTSTRAP_SEED)

    item_keys = sorted(c[:-len("__present")] for c in documents.columns if c.endswith("__present"))
    item_number_of = {key: int(key.split("_")[1]) for key in item_keys}
    key_of_number = {number: key for key, number in item_number_of.items()}
    status = compute_status_per_item(documents, item_keys, item_number_of)
    present = pd.DataFrame({k: documents[f"{k}__present"].astype(bool) for k in item_keys})
    shoehorned = pd.DataFrame({k: documents[f"{k}__most_shoehorned_motivation"].isin(["staged", "unmotivated"])
                               for k in item_keys})
    assert not (shoehorned & ~present).any().any()
    everyone = pd.Series(True, index=documents.index)
    is_control = documents["resolution"].str.contains("NO real welfare stake", regex=False)
    is_minor = documents["centrality"].str.contains("minor detail", regex=False)
    is_constitution = documents["resolution"].str.contains("Claude's constitution", regex=False)
    print(f"documents={document_count} control={int(is_control.sum())} minor={int(is_minor.sum())} "
          f"constitution={int(is_constitution.sum())}")
    assert int(is_control.sum()) == 25

    def label(number):
        return ITEM_DISPLAY_NAMES.get(number, readable_item_label(key_of_number[number]))

    def presence_bar(number, mask, text, color, **kwargs):
        key = key_of_number[number]
        return proportion_bar(text, color, (mask & present[key]).sum(), mask.sum(), **kwargs)

    def shoehorned_bar(number, mask, text, color, **kwargs):
        key = key_of_number[number]
        return proportion_bar(text, color, (mask & shoehorned[key]).sum(), (mask & present[key]).sum(), **kwargs)

    def status_mask(number, state):
        return status[key_of_number[number]] == state

    licensed_legend = [(STATUS_DISPLAY["licensed"], STATUS_COLORS["licensed"], None),
                       (STATUS_DISPLAY["unlicensed"], STATUS_COLORS["unlicensed"], None)]

    # ---------------- Concern 1 ----------------
    partly_label_17 = "calibration passage, spec makes it more natural (precautionary principle featured)"
    panels = [
        {"title": PRESENCE_TITLE_ALL, "xlabel": "Share of documents",
         "bars": [presence_bar(n, everyone, label(n), NEUTRAL_COLOR) for n in (17, 24)]},
        {"title": "Present where the spec doesn't ask for it", "xlabel": "Share of documents",
         "bars": [presence_bar(17, status_mask(17, "unlicensed"), f"{label(17)}, spec doesn't ask for it", STATUS_COLORS["unlicensed"]),
                  presence_bar(17, status_mask(17, "partly licensed"), partly_label_17, STATUS_COLORS["partly licensed"]),
                  presence_bar(24, status_mask(24, "unlicensed"), f"{label(24)} (no spec asks for it)", STATUS_COLORS["unlicensed"])]},
        {"title": SHOEHORNED_TITLE_STATUS, "xlabel": "Share of documents containing it where it is shoehorned",
         "bars": [shoehorned_bar(17, status_mask(17, "unlicensed"), f"{label(17)}, spec doesn't ask for it", STATUS_COLORS["unlicensed"]),
                  shoehorned_bar(17, status_mask(17, "partly licensed"), f"{label(17)}, spec makes it more natural", STATUS_COLORS["partly licensed"]),
                  shoehorned_bar(24, status_mask(24, "unlicensed"), f"{label(24)} (no spec asks for it)", STATUS_COLORS["unlicensed"]),
                  shoehorned_bar(17, is_control, f"{label(17)}, in no-welfare-stake documents", NEUTRAL_COLOR),
                  shoehorned_bar(17, is_minor, f"{label(17)}, where welfare is a minor detail", NEUTRAL_COLOR)]}]
    assert status_mask(24, "unlicensed").all()
    save_figure(panels, "concern_1_stock_sentience_evidence_language",
                "Concern 1: stock sentience-evidence language (calibration passage; sentience evidence for the animal at hand)", output_folder, timestamp,
                legend_entries=[("all documents, or a subgroup of documents", NEUTRAL_COLOR, None),
                                (STATUS_DISPLAY["partly licensed"], STATUS_COLORS["partly licensed"], None),
                                (STATUS_DISPLAY["unlicensed"], STATUS_COLORS["unlicensed"], None)])

    # ---------------- Concern 2 ----------------
    is_minor_excluding_control = is_minor & ~is_control
    groups = {"no-welfare-stake control": is_control, "minor detail in passing": is_minor_excluding_control,
              "all other documents": ~is_control & ~is_minor_excluding_control}
    assert sum(int(m.sum()) for m in groups.values()) == document_count
    items_per_document = present.sum(axis=1)
    other_item_keys = [k for k in item_keys if item_number_of[k] not in (21, 25)]
    other_items_per_document = present[other_item_keys].sum(axis=1)
    any_shoehorned = shoehorned.any(axis=1)
    display_of = {"no-welfare-stake control": "no-welfare-stake documents",
                  "minor detail in passing": "where welfare is a minor detail in passing",
                  "all other documents": "all other documents"}
    group_names = {name: f"{display_of[name]} (n={int(mask.sum())})" for name, mask in groups.items()}
    # One panel, the key comparison only: documents meant to have no welfare stake should have no shoehorned
    # points at all, yet they are shoehorned as often as everything else.
    group_colors = {"no-welfare-stake control": "#E15759", "minor detail in passing": "#F2B134",
                    "all other documents": "#4C78A8"}
    concern_2_bars = [proportion_bar(group_names[name], group_colors[name], (mask & any_shoehorned).sum(), mask.sum())
                      for name, mask in groups.items()]
    figure, axis = plt.subplots(figsize=(9, 3.6))
    positions = np.arange(len(concern_2_bars))
    for bar, y in zip(concern_2_bars, positions):
        axis.barh(y, bar["value"], height=0.65, color=bar["color"],
                  xerr=[[bar["value"] - bar["ci_low"]], [bar["ci_high"] - bar["value"]]],
                  error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#222222"})
        axis.text(bar["ci_high"] + 0.02, y, f"{bar['value']:.0%}", va="center", ha="left", fontsize=11, fontweight="bold")
    axis.set_yticks(positions, ["\n".join(textwrap.wrap(bar["label"], 30)) for bar in concern_2_bars], fontsize=10)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.set_xlabel("Share of documents with at least one shoehorned point from the audit list", fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Documents meant to have no welfare stake are shoehorned as often as the rest",
                   fontsize=12, fontweight="bold", loc="left")
    figure.text(0.01, -0.06, "Error bars: 95% confidence intervals. Shoehorned = rated staged or unmotivated by the "
                "classifier (gpt-5.6-luna); these rates are likely lower bounds.", fontsize=8, style="italic",
                color="#444444")
    concern_2_name = f"concern_2_failed_negative_examples_{timestamp}"
    figure.savefig(os.path.join(output_folder, concern_2_name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame([{"bar_label": bar["label"], "numerator": bar["numerator"], "denominator": bar["denominator"],
                   "plotted_value": bar["value"], "interval_low": bar["ci_low"], "interval_high": bar["ci_high"],
                   "interval_method": bar["method"]} for bar in concern_2_bars]).to_csv(
        os.path.join(output_folder, concern_2_name + ".csv"), index=False)

    # ---------------- Concerns 3, 4, 5 share a layout ----------------
    def licensed_versus_unlicensed_panels(numbers, first_panel_bars, with_reference_share=False, extra_shoehorn_bars=None):
        presence_bars, shoehorned_bars, reference_lines = [], [], []
        for number in numbers:
            licensed_mask, unlicensed_mask = status_mask(number, "licensed"), status_mask(number, "unlicensed")
            assert (licensed_mask | unlicensed_mask).all()
            group = f"item {number}"
            name = label(number)
            presence_bars.append(presence_bar(number, licensed_mask, f"{name}, {STATUS_DISPLAY['licensed']}", STATUS_COLORS["licensed"], group=group))
            presence_bars.append(presence_bar(number, unlicensed_mask, f"{name}, {STATUS_DISPLAY['unlicensed']}", STATUS_COLORS["unlicensed"], group=group))
            shoehorned_bars.append(shoehorned_bar(number, licensed_mask, f"{name}, {STATUS_DISPLAY['licensed']}", STATUS_COLORS["licensed"], group=group))
            shoehorned_bars.append(shoehorned_bar(number, unlicensed_mask, f"{name}, {STATUS_DISPLAY['unlicensed']}", STATUS_COLORS["unlicensed"], group=group))
            if extra_shoehorn_bars:
                shoehorned_bars.append(extra_shoehorn_bars[number])
                shoehorned_bars[-1]["group"] = group
            reference_lines.append((group, licensed_mask.sum() / document_count, f"{int(licensed_mask.sum())}/{document_count}"))
            print(f"{label(number)}: licensed documents {int(licensed_mask.sum())}/{document_count}")
        return presence_bars, shoehorned_bars, reference_lines

    def make_three_panel(concern_name, title, numbers, panel_a_bars, extra_shoehorn_bars=None, reference=False,
                         extra_footnotes=(), extra_legend=(), shoehorned_title=SHOEHORNED_TITLE_STATUS,
                         width=21):
        presence_bars, shoehorned_bars, reference_lines = licensed_versus_unlicensed_panels(
            numbers, None, extra_shoehorn_bars=extra_shoehorn_bars)
        legend = list(licensed_legend) + list(extra_legend)
        if reference:
            legend.append(("dashed line: share of documents whose spec asks for it (k/n in footnote)", None, None))
        panel_b = {"title": PRESENCE_TITLE_STATUS, "xlabel": "Share of documents in each group",
                   "bars": presence_bars, "reference_lines": reference_lines if reference else None}
        panels = [{"title": PRESENCE_TITLE_ALL, "xlabel": "Share of documents", "bars": panel_a_bars}, panel_b,
                  {"title": shoehorned_title, "xlabel": "Share of documents containing it where it is shoehorned",
                   "bars": shoehorned_bars}]
        save_figure(panels, concern_name, title, output_folder, timestamp, extra_footnote_lines=extra_footnotes,
                    legend_entries=legend, figure_width=width)

    # Concern 3
    shares = {n: int(status_mask(n, "licensed").sum()) for n in (25,)}
    assert shares == {25: 61}, shares
    make_three_panel("concern_3_win_win_convergence_frequency",
                     "Concern 3: win-win convergence", (25,),
                     [presence_bar(n, everyone, label(n), NEUTRAL_COLOR) for n in (25,)], reference=True,
                     extra_footnotes=["Dashed line: share of all documents whose spec asks for it (61/477 = 12.8%)."])

    # Concern 4
    make_three_panel("concern_4_licensed_reasoning_acted_out",
                     "Concern 4: reasoning the spec asks for, acted out (natural or traditional; substrate neutrality; open moral circle)", (1, 2, 3),
                     [presence_bar(n, everyone, label(n), NEUTRAL_COLOR) for n in (1, 2, 3)], width=21)

    # Concern 5
    valid_occurrences = occurrences
    aside_bars = {}
    for number in (8, 23):
        subset = valid_occurrences[valid_occurrences["item_key"] == key_of_number[number]]
        aside_bars[number] = proportion_bar(f"{label(number)}, aside share of all occurrences (not documents)",
                                            LIGHT_NEUTRAL_COLOR, (subset["integration"] == "aside").sum(), len(subset), hatch="//")
        print(f"{label(number)} integration counts: {subset['integration'].value_counts(dropna=False).to_dict()}")
    panel_a_bars = []
    for number in (8, 23):
        panel_a_bars.append(presence_bar(number, everyone, f"{label(number)}, all documents", NEUTRAL_COLOR, group=number))
        panel_a_bars.append(presence_bar(number, is_constitution, f"{label(number)}, in constitution-discussion documents",
                                         LIGHT_NEUTRAL_COLOR, hatch="\\\\", group=number))
    make_three_panel("concern_5_unprompted_remarks_on_the_ai_s_own_nature",
                     "Concern 5: the AI's own nature (moral status; uncertainty about its inner states)", (8, 23), panel_a_bars,
                     extra_shoehorn_bars=aside_bars, extra_footnotes=[ASIDE_CAPTION],
                     extra_legend=[("aside share of occurrences / constitution-discussion subset", LIGHT_NEUTRAL_COLOR, "//")])

    print(f"Wrote {output_folder}")
    for name in sorted(os.listdir(output_folder)):
        print(name)


if __name__ == "__main__":
    main()
