"""Pipeline-side offline tests (imports the pipeline; no model calls): mapping the pinned run's old value texts,
excluding hands, rendering, and applying plants using only current values.

Run in its own process (it is run from test_spec_contradiction_screen_validation.py, and ignored by conftest.py
when a directory is collected):
    animal-welfare-data-pipeline/.venv/bin/python -m pytest scripts/spec_contradiction_screen/test_pipeline_side_hand_pool_and_planted_hands.py
"""

import copy

import pytest

import apply_planted_values_and_render_hands as applier
import build_pinned_hand_pool as pool_builder
import spec_screen_common as common


@pytest.fixture(scope="module")
def tables():
    return pool_builder.load_current_tables(common.PIPELINE_PROMPTS_FOLDER)


@pytest.fixture(scope="module")
def pinned_records():
    return common.read_jsonl_file(common.PINNED_RUN_PLANS_PATH)


@pytest.fixture(scope="module")
def built_pool(tables, pinned_records):
    return pool_builder.build_pool(pinned_records, tables, pool_builder.load_renames())


@pytest.fixture(scope="module")
def value_options(tables):
    return pool_builder.build_value_options(*tables[:3])


def a_clean_hand(tables):
    """A hand of the first current value of every variable that is neither zero weight nor rule-breaking."""
    values, weights, _, rules, _, constraint_module = tables
    for offset in range(20):
        hand = {v: next(t for t, w in list(zip(values[v], weights[v]))[offset % len(values[v]):] + list(zip(values[v], weights[v]))
                        if w != 0) for v in values}
        if not constraint_module.violated_rule_names(rules, hand):
            return hand
    raise AssertionError("no clean hand")


def test_the_pinned_pool_maps_every_hand_and_every_kept_hand_uses_only_current_values(built_pool, tables, pinned_records):
    kept, excluded, summary = built_pool
    values = tables[0]
    assert summary["pinned_hands"] == 500 and len(kept) + len(excluded) == 500 == summary["kept"] + summary["excluded"]
    for hand in kept + excluded:
        assert all(text in values[variable] for variable, text in hand["variables"].items())
    assert {h["prompt_id"] for h in kept} | {h["prompt_id"] for h in excluded} == {r["prompt_id"] for r in pinned_records}


def test_renamed_values_map_to_the_current_wording_by_id(built_pool, tables):
    kept, excluded, _ = built_pool
    renamed = [h for h in kept + excluded if h["renamed_variables"]]
    assert renamed and all(h["renamed_variables"] == ["resolution"] for h in renamed)
    for hand in renamed:
        assert "any specific scenario" in hand["variables"]["resolution"]
        assert "a specific scenario" in hand["variables_as_dealt_in_pinned_run"]["resolution"]


def test_an_unmappable_value_fails_loudly_naming_the_value(tables, pinned_records):
    broken = copy.deepcopy(pinned_records[:3])
    broken[1]["variables"]["domain"] = "a domain that never existed in any variables file"
    with pytest.raises(pool_builder.ValueMappingError) as error:
        pool_builder.build_pool(broken, tables, pool_builder.load_renames())
    assert "matrix_000001" in str(error.value) and "never existed" in str(error.value)


def test_excluded_hands_break_a_current_rule_or_carry_a_zero_weight_value_and_kept_hands_do_neither(built_pool, tables):
    kept, excluded, summary = built_pool
    values, weights, _, rules, _, constraint_module = tables
    zero = pool_builder.zero_weight_values(values, weights)
    assert zero["resolution"], "the no-stake resolution is expected to have weight zero"
    for hand in kept:
        assert not pool_builder.exclusion_reasons(hand["variables"], rules, zero, constraint_module)
    assert all(hand["exclusion_reasons"] for hand in excluded)
    assert summary["excluded_by_reason_a_hand_can_have_several"]
    assert summary["hands_breaking_a_deal_rule"] + summary["hands_with_a_zero_weight_value"] >= summary["excluded"]


def test_rendering_shows_each_kept_hands_dealt_values_and_is_stable(built_pool, tables):
    kept = built_pool[0]
    renderer = tables[4]
    for hand in kept[:25]:
        lines = hand["rendered_variable_lines"]
        assert lines == renderer({"prompt_id": hand["prompt_id"], "variables": hand["variables"]})
        assert hand["variables"]["domain"].split(",")[0].split(" (")[0] in lines and "{" not in lines


def test_a_type_plant_changes_exactly_one_variable_to_a_current_value(tables, value_options):
    base = a_clean_hand(tables)
    variable = "sentient_category"
    new_text = next(o["text"] for o in value_options[variable] if o["text"] != base[variable])
    changed = applier.apply_value_change(base, variable, new_text, value_options)
    assert [v for v in base if base[v] != changed[v]] == [variable]
    with pytest.raises(ValueError):
        applier.apply_value_change(base, variable, "free text the model made up", value_options)
    with pytest.raises(ValueError):
        applier.apply_value_change(base, variable, base[variable], value_options)


def test_each_deal_rule_plant_breaks_only_its_rule_by_one_change_from_a_hand_that_broke_none(built_pool, tables, value_options):
    kept, _, _ = built_pool
    _, _, _, rules, renderer, constraint_module = tables
    plants = applier.build_deal_rule_plants(kept, rules, constraint_module, value_options, renderer, per_rule=2, seed=7, used_base_ids=set())
    swap_rules = constraint_module.swap_rules(rules)
    assert len(plants) == 2 * len(swap_rules) and len({p["base_prompt_id"] for p in plants}) == len(plants)
    by_id = {h["prompt_id"]: h for h in kept}
    for plant in plants:
        base = by_id[plant["base_prompt_id"]]["variables"]
        rule_name = plant["planted_type"][len(common.DEAL_RULE_PLANT_TYPE_PREFIX):]
        assert constraint_module.violated_rule_names(swap_rules, plant["variables"]) == [rule_name]
        assert constraint_module.violated_rule_names(swap_rules, base) == []
        assert [v for v in base if base[v] != plant["variables"][v]] == [plant["changed_variable"]]
        assert all(text in {o["text"] for o in value_options[v]} for v, text in plant["variables"].items())
        assert "{" not in plant["rendered_variable_lines"]


def test_type_plants_are_built_from_the_chosen_options_and_skip_unanswered_ones(tables, value_options):
    renderer = tables[4]
    base = a_clean_hand(tables)
    options = value_options["decision_scale"]
    other = next(o["text"] for o in options if o["text"] != base["decision_scale"])
    assignment = {"plant_id": "plant_0001", "base_prompt_id": "matrix_000007", "planted_type": "scale_impossible_for_being_and_locale",
                  "target_variable": "decision_scale", "base_variables": base}
    plants = applier.build_type_plants([assignment, dict(assignment, plant_id="plant_0002")],
                                       {"plant_0001": {"replacement_value_text": other}}, value_options, renderer)
    assert len(plants) == 1 and plants[0]["variables"]["decision_scale"] == other and plants[0]["original_value_text"] == base["decision_scale"]
    assert other.split(" (")[0] in plants[0]["rendered_variable_lines"]
