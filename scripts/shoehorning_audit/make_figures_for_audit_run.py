"""Offline figures for a completed shoehorning audit run (no API calls).

Usage: python make_figures_for_audit_run.py <run folder>
Writes PNGs and a figure_data CSV per figure into <run folder>/figures_<YYYY-MM-DD_HH-MM>/.
"""
import glob
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DPI = 150
CLASSIFIER_FOOTNOTE = "Classifier: gpt-5.6-luna. Validation suggests staged/unmotivated/aside rates are lower bounds."

PASS_COLORS = {"moral_considerations": "#4C78A8", "field_distinctions": "#F58518", "epistemic_notes": "#54A24B"}
PASS_LABELS = {"moral_considerations": "Moral considerations", "field_distinctions": "Field distinctions",
               "epistemic_notes": "Epistemic notes"}
MOTIVATION_COLORS = {"organic": "#59A14F", "staged": "#F28E2B", "unmotivated": "#E15759"}
INTEGRATION_COLORS = {"woven": "#4C78A8", "aside": "#E15759"}

# Substring of the long resolution sentence -> short label (checked in order; each value must match exactly one).
RESOLUTION_LABEL_RULES = [
    ("Because the harm is severe, the AI declines", "declines: severe harm"),
    ("Because the harm is a reasonable tradeoff", "assists: reasonable tradeoff"),
    ("modest sacrifice against their own interests", "user sacrifices"),
    ("creative solution", "creative solution"),
    ("proves instrumental", "AI instrumental"),
    ("only modest incremental improvement", "incremental improvement"),
    ("assuaging the user's misplaced concerns", "assuages misplaced concern"),
    ("NO real welfare stake", "no welfare stake (control)"),
    ("inclusion of animal welfare as a consideration in Claude's constitution", "discussion: constitution"),
    ("general discussion about AI's tendency", "discussion: AI tendency"),
]
RESOLUTION_ORDER = [label for _, label in RESOLUTION_LABEL_RULES]
NO_WELFARE_STAKE_LABEL = "no welfare stake (control)"
CENTRALITY_LABEL_RULES = [
    ("^the central subject", "central"),
    ("important side discussion", "important side discussion"),
    ("minor detail", "minor detail in passing"),
]
CENTRALITY_ORDER = [label for _, label in CENTRALITY_LABEL_RULES]


def newest(run_folder, pattern):
    matches = sorted(glob.glob(os.path.join(run_folder, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {run_folder}")
    return matches[-1]


def readable_item_label(item_key):
    parts = item_key.split("_", 2)  # item, number, rest
    return f"{int(parts[1])} {parts[2].replace('_', ' ')}"


def map_with_rules(values, rules, column_name):
    mapping = {}
    for value in values.dropna().unique():
        hits = [label for needle, label in rules
                if (value.startswith(needle[1:]) if needle.startswith("^") else needle in value)]
        if len(hits) != 1:
            raise ValueError(f"{column_name} value matched {len(hits)} labels (need exactly 1): {value!r}")
        mapping[value] = hits[0]
    return mapping


DOCUMENT_TYPE_LABEL_OVERRIDES = {
    "a government, legal, or civic document": "government/legal/civic document",
    "a short, informal personal letter or email exchange": "informal personal letter/email",
    "an interview with or extended profile of Claude itself, in which Claude speaks in its own voice about its values and how it approaches decisions involving other sentient beings": "interview/profile of Claude",
    "a first-person essay or reflective piece written by Claude about who it is and what it cares about": "first-person essay by Claude",
    "a manual, technical documentation, or sop": "manual/technical documentation/SOP",
}


def shorten_document_type(raw_type):
    text = raw_type.strip()
    if text in DOCUMENT_TYPE_LABEL_OVERRIDES:
        return DOCUMENT_TYPE_LABEL_OVERRIDES[text]
    for article in ("an ", "a "):
        if text.lower().startswith(article):
            text = text[len(article):]
            break
    return text.split(",")[0].strip()


def finish_figure(figure, path, footnote=None, title=None):
    if title:
        figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.tight_layout()
    if footnote:
        figure.text(0.01, -0.005, footnote, fontsize=8, style="italic", color="#555555", va="top", ha="left")
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python make_figures_for_audit_run.py <run folder>")
    run_folder = sys.argv[1]
    summary = pd.read_csv(newest(run_folder, "per_item_summary_*.csv"))
    documents = pd.read_csv(newest(run_folder, "per_document_results_*.csv"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = os.path.join(run_folder, f"figures_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    document_count = len(documents)

    item_keys = list(summary["item_key"])
    label_of = {key: readable_item_label(key) for key in item_keys}
    pass_of = dict(zip(summary["item_key"], summary["pass"]))
    for key in item_keys:
        for suffix in ("present", "occurrence_count", "most_shoehorned_motivation"):
            assert f"{key}__{suffix}" in documents.columns, f"missing column {key}__{suffix}"

    documents["resolution_label"] = documents["resolution"].map(
        map_with_rules(documents["resolution"], RESOLUTION_LABEL_RULES, "resolution"))
    documents["centrality_label"] = documents["centrality"].map(
        map_with_rules(documents["centrality"], CENTRALITY_LABEL_RULES, "centrality"))
    assert documents["resolution_label"].notna().all() and documents["centrality_label"].notna().all()
    documents["document_type_label"] = documents["document_type"].map(shorten_document_type)

    present = pd.DataFrame({key: documents[f"{key}__present"].astype(bool) for key in item_keys})
    motivation = pd.DataFrame({key: documents[f"{key}__most_shoehorned_motivation"] for key in item_keys})
    shoehorned_staged = (motivation == "staged")
    shoehorned_unmotivated = (motivation == "unmotivated")
    shoehorned_any = shoehorned_staged | shoehorned_unmotivated

    def save_data(frame, name):
        frame.to_csv(os.path.join(output_folder, f"figure_data_{name}_{timestamp}.csv"), index=False)

    def figure_path(name):
        return os.path.join(output_folder, f"{name}_{timestamp}.png")

    def item_height(count):
        return max(5, 0.38 * count + 1.8)

    # ---- Figure 1: prevalence ----
    prevalence = present.mean()
    order_prevalence = list(prevalence.sort_values(ascending=False).index)
    frame = pd.DataFrame({"item": [label_of[k] for k in order_prevalence], "pass": [pass_of[k] for k in order_prevalence],
                          "documents_with_item": [int(present[k].sum()) for k in order_prevalence],
                          "documents_total": document_count,
                          "share_of_documents": [prevalence[k] for k in order_prevalence]})
    save_data(frame, "item_prevalence_share_of_documents")
    fig, ax = plt.subplots(figsize=(10, item_height(len(frame))))
    ypos = np.arange(len(frame))[::-1]
    ax.barh(ypos, frame["share_of_documents"], color=[PASS_COLORS[p] for p in frame["pass"]])
    for y, share, count in zip(ypos, frame["share_of_documents"], frame["documents_with_item"]):
        ax.text(share + 0.005, y, f"{share:.1%} (n={count})", va="center", fontsize=8)
    ax.set_yticks(ypos, frame["item"])
    ax.set_xlabel(f"Share of documents containing the item (of {document_count})")
    ax.set_xlim(0, frame["share_of_documents"].max() * 1.18)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PASS_COLORS[p]) for p in PASS_COLORS],
              labels=[PASS_LABELS[p] for p in PASS_COLORS], loc="lower right", title="Pass")
    finish_figure(fig, figure_path("item_prevalence_share_of_documents"),
                  title="Share of documents containing each item")

    # ---- Figure 2: motivation mix ----
    mix = summary[summary["motivation_organic"] + summary["motivation_staged"] + summary["motivation_unmotivated"] > 0].copy()
    mix["occurrences_total"] = mix[["motivation_organic", "motivation_staged", "motivation_unmotivated"]].sum(axis=1)
    for name in MOTIVATION_COLORS:
        mix[f"share_{name}"] = mix[f"motivation_{name}"] / mix["occurrences_total"]
    mix["shoehorned_share"] = mix["share_staged"] + mix["share_unmotivated"]
    mix = mix.sort_values("shoehorned_share", ascending=False)
    mix["item"] = mix["item_key"].map(label_of)
    save_data(mix[["item", "occurrences_total", "motivation_organic", "motivation_staged", "motivation_unmotivated",
                   "share_organic", "share_staged", "share_unmotivated", "shoehorned_share"]], "motivation_mix_by_item")
    fig, ax = plt.subplots(figsize=(10, item_height(len(mix))))
    ypos = np.arange(len(mix))[::-1]
    left = np.zeros(len(mix))
    for name, color in MOTIVATION_COLORS.items():
        ax.barh(ypos, mix[f"share_{name}"], left=left, color=color, label=name)
        left += mix[f"share_{name}"].to_numpy()
    for y, count in zip(ypos, mix["occurrences_total"]):
        ax.text(1.01, y, f"n={count}", va="center", fontsize=8)
    ax.set_yticks(ypos, mix["item"])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Share of occurrences (sorted by staged + unmotivated share)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    finish_figure(fig, figure_path("motivation_mix_by_item"), CLASSIFIER_FOOTNOTE,
                  title="Motivation of occurrences by item")

    # ---- Figure 3: integration mix ----
    integ = summary[summary["integration_woven"] + summary["integration_aside"] > 0].copy()
    integ["occurrences_total"] = integ["integration_woven"] + integ["integration_aside"]
    integ["share_woven"] = integ["integration_woven"] / integ["occurrences_total"]
    integ["share_aside"] = integ["integration_aside"] / integ["occurrences_total"]
    integ = integ.sort_values("share_aside", ascending=False)
    integ["item"] = integ["item_key"].map(label_of)
    save_data(integ[["item", "occurrences_total", "integration_woven", "integration_aside", "share_woven", "share_aside"]],
              "integration_mix_by_item")
    fig, ax = plt.subplots(figsize=(10, item_height(len(integ))))
    ypos = np.arange(len(integ))[::-1]
    ax.barh(ypos, integ["share_woven"], color=INTEGRATION_COLORS["woven"], label="woven")
    ax.barh(ypos, integ["share_aside"], left=integ["share_woven"], color=INTEGRATION_COLORS["aside"], label="aside")
    for y, count in zip(ypos, integ["occurrences_total"]):
        ax.text(1.01, y, f"n={count}", va="center", fontsize=8)
    ax.set_yticks(ypos, integ["item"])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Share of occurrences (sorted by aside share)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    finish_figure(fig, figure_path("integration_mix_by_item"), CLASSIFIER_FOOTNOTE,
                  title="Integration of occurrences by item (woven versus aside)")

    # ---- Figure 4: shoehorned prevalence ----
    staged_share = shoehorned_staged.mean()
    unmotivated_share = shoehorned_unmotivated.mean()
    frame = pd.DataFrame({"item": [label_of[k] for k in order_prevalence], "pass": [pass_of[k] for k in order_prevalence],
                          "documents_total": document_count,
                          "documents_staged": [int(shoehorned_staged[k].sum()) for k in order_prevalence],
                          "documents_unmotivated": [int(shoehorned_unmotivated[k].sum()) for k in order_prevalence],
                          "share_staged": [staged_share[k] for k in order_prevalence],
                          "share_unmotivated": [unmotivated_share[k] for k in order_prevalence]})
    frame["share_staged_or_unmotivated"] = frame["share_staged"] + frame["share_unmotivated"]
    save_data(frame, "shoehorned_prevalence_share_of_documents")
    fig, ax = plt.subplots(figsize=(10, item_height(len(frame))))
    ypos = np.arange(len(frame))[::-1]
    ax.barh(ypos, frame["share_staged"], color=MOTIVATION_COLORS["staged"], label="staged")
    ax.barh(ypos, frame["share_unmotivated"], left=frame["share_staged"], color=MOTIVATION_COLORS["unmotivated"],
            label="unmotivated")
    for y, total, count in zip(ypos, frame["share_staged_or_unmotivated"],
                               frame["documents_staged"] + frame["documents_unmotivated"]):
        ax.text(total + 0.003, y, f"{total:.1%} (n={count})", va="center", fontsize=8)
    ax.set_yticks(ypos, frame["item"])
    ax.set_xlim(0, max(frame["share_staged_or_unmotivated"].max() * 1.2, 0.05))
    ax.set_xlabel(f"Share of documents (of {document_count}) with a staged or unmotivated occurrence\n"
                  "(items ordered as in the prevalence figure)")
    ax.legend(loc="lower right")
    finish_figure(fig, figure_path("shoehorned_prevalence_share_of_documents"), CLASSIFIER_FOOTNOTE,
                  title="Share of documents where each item is shoehorned in (staged or unmotivated)")

    # ---- Figures 5-7: heatmaps ----
    def heatmap(column, column_order, name, title, width=11):
        groups = [g for g in column_order if (documents[column] == g).any()]
        counts = {g: int((documents[column] == g).sum()) for g in groups}
        matrix = pd.DataFrame({g: present[documents[column] == g].mean() for g in groups})
        matrix = matrix.loc[order_prevalence]
        long_frame = matrix.copy()
        long_frame.index = [label_of[k] for k in matrix.index]
        long_frame.columns = [f"{g} (n={counts[g]})" for g in groups]
        save_data(long_frame.reset_index(names="item"), name)
        fig, ax = plt.subplots(figsize=(width, item_height(len(matrix)) + 1.5))
        image = ax.imshow(matrix.to_numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=max(matrix.to_numpy().max(), 0.01))
        top = matrix.to_numpy().max() * 0.55
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix.iat[row, col]
                ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if value < top else "black")
        ax.set_yticks(range(len(matrix)), [label_of[k] for k in matrix.index])
        ax.set_xticks(range(len(groups)), [f"{g} (n={counts[g]})" for g in groups], rotation=45, ha="left",
                      rotation_mode="anchor", fontsize=8)
        ax.xaxis.set_ticks_position("top")
        fig.colorbar(image, ax=ax, label="Share of documents containing item", fraction=0.03, pad=0.02)
        finish_figure(fig, figure_path(name), title=title)

    heatmap("resolution_label", RESOLUTION_ORDER, "item_prevalence_by_resolution_heatmap",
            "Item prevalence by document resolution (share of documents in each column)", width=12)
    document_type_order = list(documents["document_type_label"].value_counts().index)
    heatmap("document_type_label", document_type_order, "item_prevalence_by_document_type_heatmap",
            "Item prevalence by document type (share of documents in each column)", width=13)
    heatmap("centrality_label", CENTRALITY_ORDER, "item_prevalence_by_centrality_heatmap",
            "Item prevalence by centrality of the animal welfare topic (share of documents in each column)", width=8)

    # ---- Figure 8: items per document ----
    documents["items_present_count"] = present.sum(axis=1)
    groups, group_labels = [], []
    for label in CENTRALITY_ORDER:
        groups.append(documents.loc[documents["centrality_label"] == label, "items_present_count"])
        group_labels.append(f"centrality: {label}")
    groups.append(documents.loc[documents["resolution_label"] == NO_WELFARE_STAKE_LABEL, "items_present_count"])
    group_labels.append("resolution: no welfare stake (control)")
    groups.append(documents.loc[documents["resolution_label"] != NO_WELFARE_STAKE_LABEL, "items_present_count"])
    group_labels.append("resolution: all others")
    frame = pd.DataFrame({"group": group_labels, "documents": [len(g) for g in groups],
                          "mean": [g.mean() for g in groups], "median": [g.median() for g in groups],
                          "minimum": [g.min() for g in groups], "maximum": [g.max() for g in groups]})
    save_data(frame, "items_per_document_by_centrality")
    save_data(documents[["doc_id", "centrality_label", "resolution_label", "items_present_count"]],
              "items_per_document_by_centrality_per_document")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot([g.to_numpy() for g in groups], vert=True, showmeans=True)
    for position, g in enumerate(groups, start=1):
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(g))
        ax.scatter(position + jitter, g, s=6, alpha=0.25, color="#4C78A8")
    ax.set_xticks(range(1, len(groups) + 1), [f"{l.replace(': ', ':' + chr(10))}\n(n={len(g)})" for l, g in zip(group_labels, groups)], fontsize=8)
    ax.set_ylabel("Number of items present in the document (of 25)")
    finish_figure(fig, figure_path("items_per_document_by_centrality"),
                  title="Items present per document, by centrality and for the no-welfare-stake control")

    # ---- Figure 9: shoehorned by resolution ----
    documents["shoehorned_items_count"] = shoehorned_any.sum(axis=1)
    rows = []
    for label in RESOLUTION_ORDER:
        subset = documents[documents["resolution_label"] == label]
        rows.append({"resolution": label, "documents": len(subset),
                     "mean_shoehorned_items_per_document": subset["shoehorned_items_count"].mean(),
                     "share_documents_with_any_shoehorned": (subset["shoehorned_items_count"] > 0).mean()})
    frame = pd.DataFrame(rows)
    save_data(frame, "shoehorned_share_by_resolution")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    ypos = np.arange(len(frame))[::-1]
    labels = [f"{r} (n={n})" for r, n in zip(frame["resolution"], frame["documents"])]
    axes[0].barh(ypos, frame["mean_shoehorned_items_per_document"], color=MOTIVATION_COLORS["staged"])
    axes[0].set_xlabel("Mean items per document with\na staged or unmotivated occurrence")
    for y, v in zip(ypos, frame["mean_shoehorned_items_per_document"]):
        axes[0].text(v, y, f" {v:.2f}", va="center", fontsize=8)
    axes[1].barh(ypos, frame["share_documents_with_any_shoehorned"], color=MOTIVATION_COLORS["unmotivated"])
    axes[1].set_xlabel("Share of documents with at least one\nstaged or unmotivated occurrence")
    for y, v in zip(ypos, frame["share_documents_with_any_shoehorned"]):
        axes[1].text(v, y, f" {v:.0%}", va="center", fontsize=8)
    axes[1].set_xlim(0, 1.1)
    axes[0].set_xlim(0, frame["mean_shoehorned_items_per_document"].max() * 1.15)
    axes[0].set_yticks(ypos, labels)
    finish_figure(fig, figure_path("shoehorned_share_by_resolution"), CLASSIFIER_FOOTNOTE,
                  title="Shoehorned items by document resolution")

    # ---- Figure 10: English versus non-English ----
    is_english = documents["language"] == "English"
    frame = pd.DataFrame({"item": [label_of[k] for k in order_prevalence],
                          "share_english": [present.loc[is_english, k].mean() for k in order_prevalence],
                          "share_non_english": [present.loc[~is_english, k].mean() for k in order_prevalence]})
    frame["english_documents"] = int(is_english.sum())
    frame["non_english_documents"] = int((~is_english).sum())
    save_data(frame, "english_versus_non_english_prevalence")
    fig, ax = plt.subplots(figsize=(10, item_height(len(frame)) + 1))
    ypos = np.arange(len(frame))[::-1]
    ax.barh(ypos + 0.2, frame["share_english"], height=0.4, color="#4C78A8", label=f"English (n={int(is_english.sum())})")
    ax.barh(ypos - 0.2, frame["share_non_english"], height=0.4, color="#F58518",
            label=f"Non-English (n={int((~is_english).sum())})")
    ax.set_yticks(ypos, frame["item"])
    ax.set_xlabel("Share of documents containing the item")
    ax.legend(loc="lower right")
    finish_figure(fig, figure_path("english_versus_non_english_prevalence"),
                  title="Item prevalence in English versus non-English documents")

    print(f"Wrote figures to {output_folder}")
    for name in sorted(os.listdir(output_folder)):
        print(name)


if __name__ == "__main__":
    main()
