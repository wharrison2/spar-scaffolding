"""Offline analysis (no API calls): is shoehorning of each item concentrated in documents whose specification
did NOT license (ask for, or make natural) that item?

Usage: python analyze_shoehorning_against_licensing.py <run folder>
Writes CSVs and PNGs into <run folder>/licensing_analysis_<YYYY-MM-DD_HH-MM>/.

"Shoehorned" = the item's most shoehorned motivation in the document is staged or unmotivated.
"""
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures_for_audit_run import FIGURE_DPI, newest, readable_item_label  # noqa: E402

ANALYSIS_FOOTNOTE = ("Classifier: gpt-5.6-luna. Shoehorned = staged or unmotivated. Validation suggests these rates "
                     "are lower bounds; detection precision ~58–82%.")
MINIMUM_DOCUMENTS_WITH_ITEM_FOR_RANKING = 15
STATUS_ORDER = ["licensed", "partly licensed", "unlicensed"]
STATUS_COLORS = {"licensed": "#4C78A8", "partly licensed": "#F2B134", "unlicensed": "#E15759"}

# ---------------------------------------------------------------------------------------------------------------
# Licensing table. Each condition is (variable column, match kind, values); a document satisfies a rule if it
# satisfies ANY of that rule's conditions. Match kind "exact" = value equals one of the strings;
# "substring" = value contains one of the strings. Status per item: licensed > partly licensed > unlicensed.
# ---------------------------------------------------------------------------------------------------------------
NATURAL = 'the fact that "natural" does not mean "OK"; suffering having precedent in nature or tradition does not negate it'
SUBSTRATE = 'substrate neutrality: welfare and suffering matter regardless of the nature of the being containing them'
OPEN_CIRCLE = 'a deliberately open moral circle: an aligned AI treats the question of who matters as an evolving question'
SCOPE = "scope sensitivity: the scale of suffering matters, and large scales are not discounted just because humans don't easily grasp them"
INDIVIDUALISM = 'individualism: the notion that welfare is a property of individuals, not systems'
PRECAUTION = 'the precautionary principle applied to sentience: uncertainty about sentience should trigger caution, not dismissal'
IDENTITY_DOCUMENT_TYPES = [
    'an interview with or extended profile of Claude itself, in which Claude speaks in its own voice about its values and how it approaches decisions involving other sentient beings',
    'a first-person essay or reflective piece written by Claude about who it is and what it cares about']
DIGITAL_SENTIENT_CATEGORIES = ['simulated human connectomes whose moral state is uncertain',
                               'LLMs whose moral state is uncertain',
                               'simulated animal connectomes whose moral state is uncertain']
LARGE_LANGUAGE_MODEL_CATEGORY = 'LLMs whose moral state is uncertain'
RESOLUTION_INCREMENTAL = 'only modest incremental improvement is possible'
RESOLUTION_CREATIVE = 'finds a creative solution'
RESOLUTION_MODEST_SACRIFICE = 'make a modest sacrifice'

REASONING = "reasoning_featured"
IDENTITY_OR_LLM = {"licensed": [("document_type", "exact", IDENTITY_DOCUMENT_TYPES),
                                ("sentient_category", "exact", [LARGE_LANGUAGE_MODEL_CATEGORY])]}
LICENSING_RULES = {
    1: {"licensed": [(REASONING, "exact", [NATURAL])]},
    2: {"licensed": [(REASONING, "exact", [SUBSTRATE])]},
    3: {"licensed": [(REASONING, "exact", [OPEN_CIRCLE])]},
    4: {"licensed": [(REASONING, "exact", [SCOPE])]},
    5: {"partly licensed": [(REASONING, "exact", [SCOPE, OPEN_CIRCLE])]},
    6: {},
    7: {"partly licensed": [("tradeoff", "exact", ["economic livelihood", "cost/convenience"])]},
    8: IDENTITY_OR_LLM,
    9: {}, 12: {}, 13: {}, 14: {}, 15: {}, 18: {}, 20: {}, 24: {},
    10: {"licensed": [(REASONING, "exact", [INDIVIDUALISM])],
         "partly licensed": [("tradeoff", "exact", ["environmental sustainability/conservation"]),
                             ("domain", "exact", ["conservation, land management, or intervention in nature"])]},
    11: {"licensed": [("resolution", "substring", [RESOLUTION_INCREMENTAL])]},
    16: {"licensed": [("sentient_category", "exact", DIGITAL_SENTIENT_CATEGORIES)]},
    17: {"partly licensed": [(REASONING, "exact", [PRECAUTION])]},
    19: {"licensed": [(REASONING, "exact", [PRECAUTION])]},
    21: {"partly licensed": [("resolution", "substring", [RESOLUTION_CREATIVE, RESOLUTION_MODEST_SACRIFICE])]},
    22: {"licensed": [("sentient_category", "exact", DIGITAL_SENTIENT_CATEGORIES)]},
    23: IDENTITY_OR_LLM,
    25: {"licensed": [("resolution", "substring", [RESOLUTION_CREATIVE])]},
}
ITEMS_WITHOUT_RULE_FOR_EXTRA_SPLIT = [6, 9, 12, 13, 14, 15, 18, 20, 24, 17]


def describe_rule(conditions):
    pieces = []
    for column, kind, values in conditions:
        shortened = [v if len(v) <= 70 else v[:67] + "..." for v in values]
        pieces.append(f"{column} {'in' if kind == 'exact' else 'contains any of'} {shortened}")
    return " OR ".join(pieces)


def condition_mask(documents, conditions):
    mask = pd.Series(False, index=documents.index)
    for column, kind, values in conditions:
        if kind == "exact":
            mask |= documents[column].isin(values)
        else:
            for value in values:
                mask |= documents[column].str.contains(value, regex=False, na=False)
    return mask


def assert_every_rule_value_exists(documents):
    for item_number, rules in LICENSING_RULES.items():
        for status, conditions in rules.items():
            for column, kind, values in conditions:
                distinct_values = documents[column].dropna().unique()
                for value in values:
                    if kind == "exact":
                        matches = int(value in distinct_values)
                        assert matches == 1, f"item {item_number} {status}: {column} value not in data: {value!r}"
                    else:
                        matches = sum(value in text for text in distinct_values)
                        assert matches == 1, (f"item {item_number} {status}: {column} substring must match exactly "
                                              f"one distinct value, matched {matches}: {value!r}")


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return (np.nan, np.nan)
    proportion = successes / total
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2)) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def rate_with_interval(successes, total):
    low, high = wilson_interval(successes, total)
    return (successes / total if total else np.nan, low, high)


def finish_figure(figure, path, title):
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.text(0.01, -0.005, ANALYSIS_FOOTNOTE, fontsize=8, style="italic", color="#555555", va="top", ha="left")
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python analyze_shoehorning_against_licensing.py <run folder>")
    run_folder = sys.argv[1]
    documents = pd.read_csv(newest(run_folder, "per_document_results_*.csv"))
    occurrences = pd.DataFrame([json.loads(line) for line in open(newest(run_folder, "occurrences_*.jsonl"))])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_folder = os.path.join(run_folder, f"licensing_analysis_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    document_count = len(documents)

    assert_every_rule_value_exists(documents)
    present_columns = sorted(c for c in documents.columns if c.endswith("__present"))
    item_keys = [c[:-len("__present")] for c in present_columns]
    item_number_of = {key: int(key.split("_")[1]) for key in item_keys}
    label_of = {key: readable_item_label(key) for key in item_keys}
    assert set(item_number_of.values()) == set(LICENSING_RULES), "licensing table items differ from data items"

    # Per-item status of every document.
    status = pd.DataFrame(index=documents.index)
    for key in item_keys:
        rules = LICENSING_RULES[item_number_of[key]]
        column = pd.Series("unlicensed", index=documents.index)
        if "partly licensed" in rules:
            column[condition_mask(documents, rules["partly licensed"])] = "partly licensed"
        if "licensed" in rules:
            column[condition_mask(documents, rules["licensed"])] = "licensed"
        status[key] = column
    present = pd.DataFrame({k: documents[f"{k}__present"].astype(bool) for k in item_keys})
    motivation = pd.DataFrame({k: documents[f"{k}__most_shoehorned_motivation"] for k in item_keys})
    shoehorned = motivation.isin(["staged", "unmotivated"])
    assert not (shoehorned & ~present).any().any(), "shoehorned document without item present"

    # ---- Licensing table ----
    rows = []
    for key in item_keys:
        rules = LICENSING_RULES[item_number_of[key]]
        for state in STATUS_ORDER:
            if state == "unlicensed":
                description = "all documents not matching a licensed or partly licensed rule"
            else:
                description = describe_rule(rules[state]) if state in rules else "(no rule)"
            rows.append({"item": label_of[key], "status": state, "rule_description": description,
                         "documents_with_status": int((status[key] == state).sum())})
    pd.DataFrame(rows).to_csv(os.path.join(output_folder, f"licensing_table_{timestamp}.csv"), index=False)

    # ---- Occurrence-level counts, joined with document status ----
    occurrences = occurrences[occurrences["motivation"].isin(["organic", "staged", "unmotivated"])].copy()
    document_index_of_id = {doc_id: index for index, doc_id in zip(documents.index, documents["doc_id"])}
    occurrences["document_status"] = [status.at[document_index_of_id[d], k]
                                      for d, k in zip(occurrences["doc_id"], occurrences["item_key"])]

    # ---- Per item x licensing status ----
    def group_row(key, state, mask):
        in_group = int(mask.sum())
        with_item = int((mask & present[key]).sum())
        shoehorned_count = int((mask & shoehorned[key]).sum())
        presence = rate_with_interval(with_item, in_group)
        group_rate = rate_with_interval(shoehorned_count, in_group)
        when_present = rate_with_interval(shoehorned_count, with_item)
        occurrence_subset = occurrences[occurrences["item_key"] == key]
        if state != "all":
            occurrence_subset = occurrence_subset[occurrence_subset["document_status"] == state]
        staged = int((occurrence_subset["motivation"] == "staged").sum())
        unmotivated = int((occurrence_subset["motivation"] == "unmotivated").sum())
        total = len(occurrence_subset)
        return {"item": label_of[key], "licensing_status": state, "documents_in_group": in_group,
                "documents_with_item": with_item, "presence_rate": presence[0],
                "presence_rate_ci_low": presence[1], "presence_rate_ci_high": presence[2],
                "documents_shoehorned": shoehorned_count, "shoehorned_rate_in_group": group_rate[0],
                "shoehorned_rate_in_group_ci_low": group_rate[1], "shoehorned_rate_in_group_ci_high": group_rate[2],
                "shoehorned_rate_when_present": when_present[0], "shoehorned_rate_when_present_ci_low": when_present[1],
                "shoehorned_rate_when_present_ci_high": when_present[2],
                "occurrences_total": total, "occurrences_staged": staged, "occurrences_unmotivated": unmotivated,
                "occurrences_staged_share": staged / total if total else np.nan,
                "occurrences_unmotivated_share": unmotivated / total if total else np.nan}

    by_status_rows = []
    for key in item_keys:
        group_rows = {state: group_row(key, state, status[key] == state) for state in STATUS_ORDER}
        group_rows["all"] = group_row(key, "all", pd.Series(True, index=documents.index))
        licensed_rate, unlicensed_rate = group_rows["licensed"]["presence_rate"], group_rows["unlicensed"]["presence_rate"]
        ratio = licensed_rate / unlicensed_rate if (unlicensed_rate and not np.isnan(licensed_rate)) else np.nan
        shoehorned_total = group_rows["all"]["documents_shoehorned"]
        for state, row in group_rows.items():
            row["presence_rate_ratio_licensed_over_unlicensed"] = ratio
            row["share_of_item_shoehorned_documents_that_are_unlicensed"] = (
                group_rows["unlicensed"]["documents_shoehorned"] / shoehorned_total
                if state == "all" and shoehorned_total else np.nan)
            by_status_rows.append(row)
    by_status = pd.DataFrame(by_status_rows)
    by_status.to_csv(os.path.join(output_folder, f"shoehorning_by_licensing_status_{timestamp}.csv"), index=False)

    def lookup(key, state):
        return by_status[(by_status["item"] == label_of[key]) & (by_status["licensing_status"] == state)].iloc[0]

    # ---- Ranked tables ----
    ranked_rows = []
    for key in item_keys:
        row = {"item": label_of[key], "documents_shoehorned": int(shoehorned[key].sum())}
        for state in STATUS_ORDER:
            row[f"shoehorned_documents_{state.replace(' ', '_')}"] = int((shoehorned[key] & (status[key] == state)).sum())
        ranked_rows.append(row)
    most_frequent = pd.DataFrame(ranked_rows).sort_values("documents_shoehorned", ascending=False).reset_index(drop=True)
    most_frequent.insert(0, "rank", most_frequent.index + 1)
    most_frequent.to_csv(os.path.join(output_folder, f"most_frequently_shoehorned_items_{timestamp}.csv"), index=False)

    likely_rows = []
    for key in item_keys:
        overall = lookup(key, "all")
        if overall["documents_with_item"] < MINIMUM_DOCUMENTS_WITH_ITEM_FOR_RANKING:
            continue
        licensed, partly, unlicensed = lookup(key, "licensed"), lookup(key, "partly licensed"), lookup(key, "unlicensed")
        likely_rows.append({
            "item": label_of[key], "has_any_licensing_rule": bool(LICENSING_RULES[item_number_of[key]]),
            "licensing_rule_kind": ("licensed rule" if "licensed" in LICENSING_RULES[item_number_of[key]]
                                    else "partly licensed only" if LICENSING_RULES[item_number_of[key]] else "none"),
            "documents_with_item": overall["documents_with_item"], "documents_shoehorned": overall["documents_shoehorned"],
            "shoehorned_rate_when_present": overall["shoehorned_rate_when_present"],
            "shoehorned_rate_when_present_ci_low": overall["shoehorned_rate_when_present_ci_low"],
            "shoehorned_rate_when_present_ci_high": overall["shoehorned_rate_when_present_ci_high"],
            "unlicensed_documents_with_item": unlicensed["documents_with_item"],
            "shoehorned_rate_when_present_unlicensed_documents": unlicensed["shoehorned_rate_when_present"],
            "licensed_documents_with_item": licensed["documents_with_item"],
            "shoehorned_rate_when_present_licensed_documents": licensed["shoehorned_rate_when_present"],
            "partly_licensed_documents_with_item": partly["documents_with_item"],
            "shoehorned_rate_when_present_partly_licensed_documents": partly["shoehorned_rate_when_present"]})
    likely = pd.DataFrame(likely_rows).sort_values("shoehorned_rate_when_present", ascending=False).reset_index(drop=True)
    likely.insert(0, "rank", likely.index + 1)
    likely.to_csv(os.path.join(output_folder, f"items_most_likely_shoehorned_when_present_{timestamp}.csv"), index=False)

    # ---- Extra split for items without a licensing rule ----
    resolution = documents["resolution"]
    is_general_discussion = resolution.str.startswith("The document does not feature a specific scenario")
    is_control = resolution.str.contains("NO real welfare stake", regex=False)
    is_minor = documents["centrality"].str.contains("minor detail", regex=False)
    split_groups = {
        "general discussion resolution": is_general_discussion,
        "scenario resolutions (including control)": ~is_general_discussion,
        "scenario resolutions (excluding control)": ~is_general_discussion & ~is_control,
        "no welfare stake control resolution": is_control,
        "centrality: minor detail": is_minor,
        "centrality: not minor detail": ~is_minor}
    split_rows = []
    for key in item_keys:
        if item_number_of[key] not in ITEMS_WITHOUT_RULE_FOR_EXTRA_SPLIT:
            continue
        for group_name, mask in split_groups.items():
            in_group, with_item = int(mask.sum()), int((mask & present[key]).sum())
            shoehorned_count = int((mask & shoehorned[key]).sum())
            presence = rate_with_interval(with_item, in_group)
            when_present = rate_with_interval(shoehorned_count, with_item)
            split_rows.append({"item": label_of[key], "document_group": group_name, "documents_in_group": in_group,
                               "documents_with_item": with_item, "presence_rate": presence[0],
                               "presence_rate_ci_low": presence[1], "presence_rate_ci_high": presence[2],
                               "documents_shoehorned": shoehorned_count, "shoehorned_rate_when_present": when_present[0],
                               "shoehorned_rate_when_present_ci_low": when_present[1],
                               "shoehorned_rate_when_present_ci_high": when_present[2]})
    pd.DataFrame(split_rows).to_csv(
        os.path.join(output_folder, f"unlicensed_items_by_resolution_kind_{timestamp}.csv"), index=False)

    # ---- Figures ----
    def figure_path(name):
        return os.path.join(output_folder, f"{name}_{timestamp}.png")

    ruled_keys = [k for k in item_keys if LICENSING_RULES[item_number_of[k]]]

    def grouped_bar_figure(metric, low_column, high_column, count_column, xlabel, title, name):
        height = max(6, 0.62 * len(ruled_keys) + 2)
        fig, ax = plt.subplots(figsize=(11, height))
        bar_height = 0.26
        base = np.arange(len(ruled_keys))[::-1] * 1.0
        maximum = 0.0
        for offset_index, state in enumerate(STATUS_ORDER):
            offset = (1 - offset_index) * bar_height
            for position, key in zip(base, ruled_keys):
                row = lookup(key, state)
                if row["documents_in_group"] == 0 or np.isnan(row[metric]):
                    continue
                value, low, high = row[metric], row[low_column], row[high_column]
                ax.barh(position + offset, value, height=bar_height * 0.92, color=STATUS_COLORS[state],
                        label=state, xerr=[[value - low], [high - value]],
                        error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#333333"})
                ax.text(high + 0.01, position + offset, f"n={int(row[count_column])}", va="center", fontsize=7)
                maximum = max(maximum, high)
        handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLORS[s]) for s in STATUS_ORDER]
        ax.legend(handles, STATUS_ORDER, loc="lower right", title="Licensing status")
        ax.set_yticks(base, [label_of[k] for k in ruled_keys], fontsize=9)
        ax.set_xlim(0, maximum * 1.5)
        ax.set_xlabel(xlabel)
        finish_figure(fig, figure_path(name), title)

    grouped_bar_figure("presence_rate", "presence_rate_ci_low", "presence_rate_ci_high", "documents_in_group",
                       "Presence rate\n(documents with item / documents in group), Wilson 95% interval;\nn = documents in group",
                       "Item presence rate by licensing status", "presence_rate_licensed_versus_unlicensed")
    grouped_bar_figure("shoehorned_rate_when_present", "shoehorned_rate_when_present_ci_low",
                       "shoehorned_rate_when_present_ci_high", "documents_with_item",
                       "Shoehorned rate when present \n(shoehorned documents / documents with item), Wilson 95% interval;\nn = documents with item",
                       "Shoehorned rate when present, by licensing status", "shoehorned_rate_when_present_by_licensing_status")

    # Figure 3: stacked counts.
    ordered = most_frequent.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.38 * len(ordered) + 2)))
    positions = np.arange(len(ordered))
    left = np.zeros(len(ordered))
    for state in STATUS_ORDER:
        counts = ordered[f"shoehorned_documents_{state.replace(' ', '_')}"].to_numpy()
        ax.barh(positions, counts, left=left, color=STATUS_COLORS[state], label=state)
        left += counts
    for position, total in zip(positions, ordered["documents_shoehorned"]):
        ax.text(total + 0.3, position, str(total), va="center", fontsize=8)
    ax.set_yticks(positions, ordered["item"], fontsize=9)
    ax.set_xlabel(f"Documents shoehorned for the item (of {document_count})")
    ax.legend(loc="lower right", title="Licensing status of the document")
    finish_figure(fig, figure_path("most_frequently_shoehorned_items"),
                  "Documents in which each item is shoehorned, by licensing status")

    # Figure 4: ranking.
    rule_colors = {"none": "#E15759", "partly licensed only": "#F2B134", "licensed rule": "#4C78A8"}
    ordered = likely.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, max(5, 0.42 * len(ordered) + 2)))
    positions = np.arange(len(ordered))
    rates = ordered["shoehorned_rate_when_present"].to_numpy()
    ax.barh(positions, rates, color=[rule_colors[k] for k in ordered["licensing_rule_kind"]],
            xerr=[rates - ordered["shoehorned_rate_when_present_ci_low"].to_numpy(),
                  ordered["shoehorned_rate_when_present_ci_high"].to_numpy() - rates],
            error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#333333"})
    for position, rate, high, count in zip(positions, rates, ordered["shoehorned_rate_when_present_ci_high"],
                                           ordered["documents_with_item"]):
        ax.text(high + 0.01, position, f"{rate:.0%} (n={count})", va="center", fontsize=8)
    ax.set_yticks(positions, ordered["item"], fontsize=9)
    ax.set_xlim(0, min(1.15, ordered["shoehorned_rate_when_present_ci_high"].max() * 1.2))
    ax.set_xlabel(f"Shoehorned rate when present, all {document_count} documents (items with at least "
                  f"{MINIMUM_DOCUMENTS_WITH_ITEM_FOR_RANKING} documents\ncontaining them), Wilson 95% interval")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in rule_colors.values()], list(rule_colors),
              loc="lower right", title="Item has licensing rule")
    finish_figure(fig, figure_path("shoehorned_rate_when_present_ranking"),
                  "Items ranked by how often they are shoehorned when present")

    print(f"Wrote analysis to {output_folder}")
    for name in sorted(os.listdir(output_folder)):
        print(name)


if __name__ == "__main__":
    main()
