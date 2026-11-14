"""Offline tests for the fixed spec set enumeration. No model call is made."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import enumerate_fixed_spec_set_fitting_category_and_subcategory_weights as enumerate_specs

HARMS = [f"harm_{i}" for i in range(11)]
ADVANTAGES = [f"advantage_{i}" for i in range(10)]


def synthetic_rows():
    rows = []
    for category, weight in (("category_a", 0.6), ("category_b", 0.4)):
        for subcategory, relative in (("one", 1.0), ("two", 0.5), ("three", 0.5)):
            total = 2.0
            rows.append({"taxa_category": category, "taxa_subcategory": f"{category}_{subcategory}",
                         "category_weight": weight, "relative_weight": relative, "share": weight * relative / total})
    return rows


def test_largest_remainder_sums_to_total_and_is_deterministic():
    counts = enumerate_specs.largest_remainder_apportionment([0.145, 0.14, 0.095, 0.62], 37)
    assert sum(counts) == 37
    assert counts == enumerate_specs.largest_remainder_apportionment([0.145, 0.14, 0.095, 0.62], 37)
    assert all(abs(count - share * 37) < 1 for count, share in zip(counts, [0.145, 0.14, 0.095, 0.62]))


def test_two_level_apportionment_gives_each_category_its_exact_quota():
    rows = synthetic_rows()
    counts = enumerate_specs.apportion_specs(rows, 101)
    assert sum(counts[:3]) == 61 and sum(counts[3:]) == 40


def test_a_full_subcategory_uses_every_pair_once_and_every_prefix_is_balanced():
    rows = [{"taxa_category": "only", "taxa_subcategory": "only_sub", "category_weight": 1.0,
             "relative_weight": 1.0, "share": 1.0}]
    specs = enumerate_specs.build_specs(rows, [110], HARMS, ADVANTAGES, set(), seed=1)
    assert len({(s["harm_type"], s["practical_advantage"]) for s in specs}) == 110
    values = {"harm_type": HARMS, "practical_advantage": ADVANTAGES}
    report = enumerate_specs.check_specs(specs, rows, 110, values, set(), 110)
    assert report["failures"] == []


def test_spec_set_is_balanced_distinct_and_passes_every_check_on_synthetic_weights():
    rows = synthetic_rows()
    counts = enumerate_specs.apportion_specs(rows, 300)
    old_triples = {(rows[0]["taxa_category"], rows[0]["taxa_subcategory"], HARMS[3])}
    specs = enumerate_specs.build_specs(rows, counts, HARMS, ADVANTAGES, old_triples, seed=5)
    values = {"harm_type": HARMS, "practical_advantage": ADVANTAGES}
    report = enumerate_specs.check_specs(specs, rows, 300, values, old_triples, 110)
    assert report["failures"] == []
    assert report["corpus_wide_harm_type_counts"]["spread"] <= 1
    assert report["corpus_wide_practical_advantage_counts"]["spread"] <= 1
    assert specs[0]["spec_id"] == "spec_00001" and specs[0]["position_in_subcategory_order"] == 1


def test_old_matching_harm_is_used_last_when_a_subcategory_has_few_specs():
    rows = synthetic_rows()[:1]
    old_triples = {(rows[0]["taxa_category"], rows[0]["taxa_subcategory"], harm) for harm in HARMS[:2]}
    specs = enumerate_specs.build_specs(rows, [9], HARMS, ADVANTAGES, old_triples, seed=2)
    used = {s["harm_type"] for s in specs}
    assert len(used) == 9 and not used & set(HARMS[:2])


def test_check_fails_when_a_subcategory_is_over_capacity_or_specs_repeat():
    rows = synthetic_rows()[:1]
    specs = enumerate_specs.build_specs(rows, [20], HARMS, ADVANTAGES, set(), seed=1)
    values = {"harm_type": HARMS, "practical_advantage": ADVANTAGES}
    assert enumerate_specs.check_specs(specs, rows, 20, values, set(), 10)["failures"]
    duplicated = specs + [dict(specs[0])]
    assert any("distinct" in f for f in enumerate_specs.check_specs(duplicated, rows, 21, values, set(), 110)["failures"])


def test_real_weights_small_run_passes_and_writes_both_files(tmp_path):
    old_folder = tmp_path / "no_old_corpus"
    exit_code = enumerate_specs.main(["--total-specs", "400", "--old-corpus-folder", str(old_folder),
                                      "--output-folder", str(tmp_path / "out")])
    assert exit_code == 0
    spec_file = next((tmp_path / "out").glob("fixed_spec_set_400_specs_*.jsonl"))
    report = json.loads(next((tmp_path / "out").glob("fixed_spec_set_check_report_*.json")).read_text())
    assert len(spec_file.read_text().splitlines()) == 400 and report["passed"] is True
    assert len(report["variables_file_sha256"]) == 64 and len(report["profile_file_sha256"]) == 64


def test_total_over_the_largest_fitting_size_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        enumerate_specs.main(["--total-specs", "999999", "--output-folder", str(tmp_path)])
