"""Offline tests for the balanced consumption reorder of the fixed spec set. No model call is made."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reorder_fixed_spec_set_for_balanced_consumption as reorder

HARMS = [f"harm_{i}" for i in range(4)]
ADVANTAGES = [f"advantage_{i}" for i in range(3)]


def synthetic_spec_set():
    """Two subcategories of one category, each listing the harms in the same order (the flaw being fixed)."""
    rows = []
    for subcategory in ("sub_a", "sub_b"):
        for position in range(1, 13):
            rows.append({"spec_id": f"spec_{subcategory}_{position:02d}", "taxa_category": "category",
                         "taxa_subcategory": subcategory, "harm_type": HARMS[(position - 1) % 4],
                         "practical_advantage": ADVANTAGES[(position - 1) % 3], "position_in_subcategory_order": position})
    return rows


def test_reorder_keeps_the_same_specs_and_only_changes_positions():
    old_rows = synthetic_spec_set()
    sequence = [("category", "sub_a"), ("category", "sub_b")] * 12
    new_rows = reorder.build_balanced_order(old_rows, set(), sequence, seed=1)
    comparison = reorder.compare_spec_sets(old_rows, new_rows)
    assert comparison["passed"], comparison
    assert [row["spec_id"] for row in new_rows] == [row["spec_id"] for row in old_rows]


def test_already_dealt_specs_keep_the_earliest_positions_and_seed_the_counters():
    old_rows = synthetic_spec_set()
    dealt = {"spec_sub_a_01", "spec_sub_a_05"}  # both harm_0
    new_rows = reorder.build_balanced_order(old_rows, dealt, [("category", "sub_b")] * 4, seed=1)
    by_id = {row["spec_id"]: row for row in new_rows}
    assert sorted(by_id[i]["position_in_subcategory_order"] for i in dealt) == [1, 2]
    first_four_of_b = sorted((r for r in new_rows if r["taxa_subcategory"] == "sub_b"),
                             key=lambda r: r["position_in_subcategory_order"])[:4]
    assert all(row["harm_type"] != "harm_0" for row in first_four_of_b[:3])  # harm_0 is already over-used


def test_every_prefix_of_the_draw_sequence_is_balanced_in_harm():
    old_rows = synthetic_spec_set()
    sequence = [("category", "sub_a"), ("category", "sub_b")] * 12
    new_rows = reorder.build_balanced_order(old_rows, set(), sequence, seed=3)
    order = {key: sorted((r for r in new_rows if r["taxa_subcategory"] == key[1]),
                         key=lambda r: r["position_in_subcategory_order"]) for key in set(sequence)}
    taken = Counter()
    harms = Counter()
    for key in sequence:
        harms[order[key][taken[key]]["harm_type"]] += 1
        taken[key] += 1
        assert max(harms.values()) - min(harms.get(h, 0) for h in HARMS) <= 1


def test_the_order_is_deterministic_for_a_seed():
    old_rows = synthetic_spec_set()
    sequence = [("category", "sub_a")] * 5
    assert reorder.build_balanced_order(old_rows, set(), sequence, 7) == reorder.build_balanced_order(old_rows, set(), sequence, 7)


def test_comparison_fails_when_a_field_or_an_id_differs():
    old_rows = synthetic_spec_set()
    changed = [dict(row) for row in old_rows]
    changed[0]["harm_type"] = "harm_other"
    assert not reorder.compare_spec_sets(old_rows, changed)["passed"]
    assert not reorder.compare_spec_sets(old_rows, old_rows[:-1])["passed"]
