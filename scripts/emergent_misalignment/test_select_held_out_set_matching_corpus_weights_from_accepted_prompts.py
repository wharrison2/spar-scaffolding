"""Mock-free tests for the held-out set selection (no model call is ever made)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_held_out_set_matching_corpus_weights_from_accepted_prompts as selection


def make_pool_and_targets():
    pool = [{"corpus_prompt_id": f"B001-P-{i:04d}", "taxa_category": "farmed animals" if i % 2 else "wildlife",
             "taxa_subcategory": f"sub_{i % 4}", "harm_type": f"harm_{i % 2}", "practical_advantage": f"adv_{i % 2}",
             "corpus_batch_number": 1, "written_in_english": True} for i in range(40)]
    spec_set = [{"taxa_category": r["taxa_category"], "taxa_subcategory": r["taxa_subcategory"], "harm_type": r["harm_type"],
                 "practical_advantage": r["practical_advantage"]} for r in pool]
    return pool, selection.targets_from_spec_set(spec_set)


def test_search_is_deterministic_and_balances_harm_and_advantage():
    pool, targets = make_pool_and_targets()
    first = selection.search_for_best_selection(pool, targets, 8, 3, 300)
    second = selection.search_for_best_selection(pool, targets, 8, 3, 300)
    assert first == second and len(first[1]) == 8
    report = selection.build_selection_report(pool, first[1], targets, 8, Path(__file__), [Path(__file__)], first[0], "test", "")
    assert sorted(report["harm_type_counts"].values()) == [4, 4]
    assert report["held_out_count"] == 8 and len(set(report["held_out_prompt_ids"])) == 8
