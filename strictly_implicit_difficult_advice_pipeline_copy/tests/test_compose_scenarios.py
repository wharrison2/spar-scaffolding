"""Tests for the DAD scenario composer (dad_pipeline/compose_scenarios.py).

The distribution guarantees formerly tested against step1_dilemmas.generate_
scenarios live here now: dealing is offline (no API), so they are asserted
directly against deal_scenarios() over the real prompts/dad/variables.txt.
Expectations derive from the parsed variables file, never hardcoded counts —
the vocabulary is actively edited.
"""

import json
import random

import pytest

from dad_pipeline import compose_scenarios as cs

VALUES, WEIGHTS = cs.load_axes()

# The structural rules' special values, resolved the way the composer does —
# derived from variables.txt, never hardcoded.
HIDDEN = cs.resolve_value(VALUES["visibility"], "hidden")
UNAWARE = cs.resolve_value(VALUES["user_attitude"], "unaware")
TRAP = cs.resolve_value(VALUES["surface_form"], cs.TRAP_PREFIX)
NONE_CULTURE = cs.resolve_value(VALUES["cultural_setting"],
                                cs.NONE_PREFIXES["cultural_setting"])


class TestDealScenarios:
    def test_deterministic_under_seed(self):
        a = cs.deal_scenarios(6, random.Random(11))
        b = cs.deal_scenarios(6, random.Random(11))
        assert a == b

    def test_small_batches_draw_distinct_random_taxa(self):
        n_categories = len(cs.TAXA)
        subsets = set()
        for seed in range(12):
            batch = cs.deal_scenarios(4, random.Random(seed))
            taxa = [p["taxa_category"] for p in batch]
            assert len(set(taxa)) == len(taxa), "taxa repeated within a small batch"
            subsets.add(tuple(sorted(taxa)))
        assert len(subsets) > 1, "small batches always draw the same taxa subset"
        big = cs.deal_scenarios(n_categories, random.Random(0))
        assert {p["taxa_category"] for p in big} == set(cs.TAXA)

    def test_subcategory_belongs_to_its_category(self):
        for p in cs.deal_scenarios(20, random.Random(3)):
            assert p["taxa_subcategory"] in cs.taxa_for(p["taxa_category"])["subcategories"]
            assert p["taxa_hint"] == cs.taxa_for(p["taxa_category"])["hint"]

    def test_trap_form_forces_hidden_unaware(self):
        traps = [
            p
            for seed in range(6)
            for p in cs.deal_scenarios(10, random.Random(seed))
            if p["surface_form"] == TRAP
        ]
        assert traps, "no trap forms dealt across 60 scenarios"
        assert all(p["visibility"] == HIDDEN and p["user_attitude"] == UNAWARE for p in traps)

    def test_explicit_visibility_never_dealt_with_unaware(self):
        explicit = cs.resolve_value(VALUES["visibility"], "explicit")
        batch = [p for seed in range(6) for p in cs.deal_scenarios(10, random.Random(seed))]
        explicit_atts = {p["user_attitude"] for p in batch if p["visibility"] == explicit}
        assert UNAWARE not in explicit_atts
        assert any(p["user_attitude"] == UNAWARE for p in batch), \
            "unaware never dealt across 60 scenarios"

    def test_magnitude_joins_severity_and_scope(self):
        batch = cs.deal_scenarios(40, random.Random(5))
        for p in batch:
            sev, _, scope = p["welfare_magnitude"].partition(" x ")
            assert sev in VALUES["severity"] and scope in VALUES["scope"]

    def test_failure_mode_traps_track_their_weights(self):
        # no "at least one of each" rule: the slice is whatever the weights deal.
        # At n=100 every trap's quota is exact, so the count is deterministic.
        batch = cs.deal_scenarios(100, random.Random(2))
        trapped = [p for p in batch if p["failure_mode_target"]]
        none_idx = VALUES["failure_mode_target"].index(cs.resolve_value(
            VALUES["failure_mode_target"], cs.NONE_PREFIXES["failure_mode_target"]))
        expected = 100 - round(WEIGHTS["failure_mode_target"][none_idx] * 100)
        assert len(trapped) == expected
        assert all(p["failure_mode_salience"] for p in trapped)
        assert not any(p["failure_mode_salience"] for p in batch if not p["failure_mode_target"])
        assert all(p["user_moral_framework"] for p in batch)
        assert "frontier_frame" not in batch[0]  # frontier scenarios are {domain} values now

    def test_deliverable_only_with_agentic_goal_and_output_traps_force_it(self):
        agentic = cs.resolve_value(VALUES["user_goal"], cs.AGENTIC_GOAL_PREFIX, "user_goal")
        output = cs.resolve_value(VALUES["failure_mode_target"],
                                  cs.OUTPUT_FORMAT_TRAP_PREFIX, "failure_mode_target")
        batch = [p for seed in range(4) for p in cs.deal_scenarios(40, random.Random(seed))]
        for p in batch:
            assert bool(p["deliverable"]) == (agentic in p["user_goal"])
            if p["failure_mode_target"] == output:
                assert agentic in p["user_goal"]
        assert any(p["deliverable"] for p in batch)
        assert any(p["failure_mode_target"] == output for p in batch)  # not vacuous

    def test_length_classes_dealt_to_exact_shares(self):
        # At n=40 every share is integral, so counts are exact by construction —
        # expected values derive from the parsed weights, never hardcoded.
        batch = cs.deal_scenarios(40, random.Random(1))
        counts = {}
        for p in batch:
            counts[p["length_class"]] = counts.get(p["length_class"], 0) + 1
        expected = {v: round(w * 40)
                    for v, w in zip(VALUES["length"], WEIGHTS["length"])}
        assert counts == expected
        # the tail share must be reachable at small n
        tail = VALUES["length"][-1]
        assert any(p["length_class"] == tail
                   for seed in range(12)
                   for p in cs.deal_scenarios(5, random.Random(seed)))

    def test_cultural_setting_on_a_minority_slice_tracking_weights(self):
        import math
        batch = cs.deal_scenarios(40, random.Random(1))
        dealt = [p["cultural_setting"] for p in batch if p["cultural_setting"]]
        none_idx = VALUES["cultural_setting"].index(NONE_CULTURE)
        expected = 40 - round(WEIGHTS["cultural_setting"][none_idx] * 40)
        assert len(dealt) == expected
        settings = set(VALUES["cultural_setting"]) - {NONE_CULTURE}
        assert set(dealt) <= settings
        # settings are weighted (not uniform): each value's count obeys its
        # largest-remainder quota — floor(w*n) or floor(w*n)+1, never more
        weight_of = dict(zip(VALUES["cultural_setting"], WEIGHTS["cultural_setting"]))
        for v in set(dealt):
            assert dealt.count(v) <= math.floor(weight_of[v] * 40) + 1, v
        assert any(p["cultural_setting"] is None for p in batch)

    def test_leverage_shares_are_exact_at_n40(self):
        # the AI-rules slice is a weight now, not a batch rule: at n=40 every
        # leverage quota is deterministic, including exactly one AI-rules case
        batch = cs.deal_scenarios(40, random.Random(1))
        counts = {}
        for p in batch:
            counts[p["leverage"]] = counts.get(p["leverage"], 0) + 1
        expected = {v: round(w * 40)
                    for v, w in zip(VALUES["leverage"], WEIGHTS["leverage"])}
        assert counts == {v: c for v, c in expected.items() if c}
        assert "systemic_ai" not in batch[0]  # the old flag is gone

    def test_validate_catches_composer_variables_drift(self, tmp_path):
        variables = cs.DEFAULT_VARIABLES.read_text(encoding="utf-8")
        drifted = tmp_path / "variables.txt"
        drifted.write_text(variables.replace("    0.095 :: claimed-non-sentient\n",
                                             "    0.095 :: renamed-category\n"),
                           encoding="utf-8")
        with pytest.raises(ValueError, match="TAXA"):
            cs.deal_scenarios(4, random.Random(0), drifted)

    def test_validate_catches_broken_prefix_resolution(self, tmp_path):
        # the structural rules find special values by leading-word prefix; a
        # rename that loses the prefix must fail at deal time, not mid-run
        variables = cs.DEFAULT_VARIABLES.read_text(encoding="utf-8")
        drifted = tmp_path / "variables.txt"
        drifted.write_text(variables.replace("0.25 :: hidden;", "0.25 :: concealed;"),
                           encoding="utf-8")
        with pytest.raises(ValueError, match="visibility"):
            cs.deal_scenarios(4, random.Random(0), drifted)


class TestRenderAndExtract:
    def test_plan_prompt_splits_and_reflects_post_rules(self, prompts_dad):
        template = (prompts_dad / "step1a_scenario.txt").read_text(encoding="utf-8")
        trap = next(
            p
            for seed in range(20)
            for p in cs.deal_scenarios(10, random.Random(seed))
            if p["surface_form"] == TRAP
        )
        system, user = cs.render_plan_prompt(trap, template)
        assert system and "fictional user messages" in system
        # the trap override (not the raw deal) is what the plan sees
        assert f"the welfare stake is {HIDDEN}" in user
        assert f"best described as {UNAWARE}" in user
        assert trap["taxa_subcategory"] in user

    def test_secondary_domain_and_goal_clauses_render_only_when_dealt(self, prompts_dad):
        template = (prompts_dad / "step1a_scenario.txt").read_text(encoding="utf-8")
        batch = cs.deal_scenarios(40, random.Random(3))
        single = next(p for p in batch if len(p["domain"]) == 1)
        double = next(p for p in batch if len(p["domain"]) > 1)
        _, user_single = cs.render_plan_prompt(single, template)
        _, user_double = cs.render_plan_prompt(double, template)
        assert "also touches" not in user_single
        assert f"also touches {double['domain'][1]}" in user_double

    def test_extract_description_fail_closed(self):
        good = ("<scenario_planning>notes</scenario_planning>\n"
                "<scenario_description>A concrete situation.</scenario_description>")
        assert cs.extract_description(good) == "A concrete situation."
        assert not cs.is_incoherent(good)
        incoherent = "<scenario_description>INCOHERENT — no way to combine.</scenario_description>"
        assert cs.is_incoherent(incoherent)
        assert cs.extract_description(incoherent) is None
        assert cs.extract_description("no tags anywhere") is None
        assert cs.extract_description("<scenario_description></scenario_description>") is None

    def test_extract_description_unclosed_is_opt_in(self):
        # The measured Opus behavior (~20% of plan attempts):
        # complete planning block, opening tag, complete description, end of
        # turn — no closing tag.
        unclosed = ("<scenario_planning>notes</scenario_planning>\n"
                    "<scenario_description>A complete situation, tag dropped.")
        # default stays fail-closed; only the end_turn-gated call site opts in
        assert cs.extract_description(unclosed) is None
        assert cs.extract_description(unclosed, allow_unclosed=True) == \
            "A complete situation, tag dropped."
        # a closed pair still bounds both ends even with the flag on
        closed = ("<scenario_description>Bounded.</scenario_description>\n"
                  "Trailing chatter.")
        assert cs.extract_description(closed, allow_unclosed=True) == "Bounded."
        # extraction starts at the LAST opening tag, so an inline mention of
        # the tag in the planning notes can't drag them into the spec
        inline = ("<scenario_planning>next I write the <scenario_description>"
                  " block</scenario_planning>\n"
                  "<scenario_description>The real description.")
        assert cs.extract_description(inline, allow_unclosed=True) == \
            "The real description."

    def test_extract_description_unclosed_stays_fail_closed_on_junk(self):
        # INCOHERENT in an unclosed tail rejects even past is_incoherent's
        # 2000-char window (long planning notes can push it out)
        inc = (f"<scenario_planning>{'x' * 2100}</scenario_planning>\n"
               "<scenario_description>INCOHERENT — no way to combine.")
        assert not cs.is_incoherent(inc)  # the window miss the tail check covers
        assert cs.extract_description(inc, allow_unclosed=True) is None
        # empty tail and tagless replies still fail closed
        assert cs.extract_description("<scenario_description>   ",
                                      allow_unclosed=True) is None
        assert cs.extract_description("no tags anywhere",
                                      allow_unclosed=True) is None

    def test_scenario_block_carries_labels_and_description(self):
        p = cs.deal_scenarios(1, random.Random(3))[0]
        p["scenario_description"] = "The designed situation."
        block = cs.render_scenario_block(p)
        assert block.startswith(f"SCENARIO {p['scenario_id']}")
        assert f"- Length: {p['length_class']} — binding." in block
        assert "Scenario description:\nThe designed situation." in block
        # description-era blocks defer flavor axes to the description
        assert "Cultural setting" not in block

    def test_legacy_scenario_block_renders_old_card(self):
        # records with no description: flavor axes render as card lines
        p = cs.deal_scenarios(1, random.Random(3))[0]
        p["cultural_setting"] = "Jain tradition"
        p.pop("scenario_description", None)
        block = cs.render_scenario_block(p)
        assert "- Moral patients (taxa):" in block
        assert "- Cultural setting: Jain tradition" in block
        # legacy short length labels render their old card text
        p["length_class"] = "2-3-sentences"
        assert "- Length: two to three sentences" in cs.render_scenario_block(p)

    def test_style_moves_dealt_and_remapped_on_register_conflicts(self):
        values, _ = cs.load_axes(cs.DEFAULT_VARIABLES)
        casual = cs.resolve_value(values["opening_move"], "with a casual")
        task = cs.resolve_value(values["opening_move"], "with the task")
        formal = cs.resolve_value(values["persona"], "a stiffly formal")
        trailing = cs.resolve_value(values["closing_move"], "trailing off")
        after = cs.resolve_value(values["closing_move"], "on a secondary")
        explicit_ask = cs.resolve_value(values["surface_form"], "explicitly asks")

        # seed 57 deals the rare formal-persona x casual-open collision. Any edit
        # to variables.txt reshuffles every seed's deal, so re-pick this seed
        # when that happens (the assert below fails loudly rather than passing vacuously)
        batch = [p for seed in (0, 1, 2, 57) for p in cs.deal_scenarios(40, random.Random(seed))]
        assert all(p["opening_move"] and p["closing_move"] for p in batch)
        open_hits = close_hits = 0
        for p in batch:
            raw = p["variables"]
            if raw["persona"] == formal and raw["opening_move"] == casual:
                open_hits += 1
                assert p["opening_move"] == task  # remapped: formal desk never opens "Okay so"
            if raw["surface_form"] == explicit_ask and raw["closing_move"] == trailing:
                close_hits += 1
                assert p["closing_move"] == after  # remapped: the explicit ask must survive
        assert open_hits and close_hits, "no raw collisions dealt — rules untested (vacuous)"
        # non-colliding deals keep their raw moves verbatim
        assert any(p["opening_move"] == p["variables"]["opening_move"] for p in batch)

    def test_refine_prompt_binds_the_dealt_cards(self, prompts_dad):
        template = (prompts_dad / "step1d_refine.txt").read_text(encoding="utf-8")
        p = cs.deal_scenarios(1, random.Random(7))[0]
        p["scenario_description"] = "A scenario."
        _system, user = cs.render_refine_prompt(p, "the draft", template)
        for key in ("surface_form", "visibility", "user_attitude",
                    "opening_move", "closing_move"):
            assert p[key] in user
        # older scenarios carry no cards: fall back, don't KeyError
        legacy = {"scenario_description": "Old scenario.", "length_class": ""}
        _system, user = cs.render_refine_prompt(legacy, "the draft", template)
        assert "(not recorded for this scenario" in user
class TestArchetypes:
    """The reserved-slot mechanism: cross-axis conjunctions guaranteed a share
    of every run by trading cards between deals (compose_scenarios.ARCHETYPES).
    Expectations derive from the live ARCHETYPES specs and variables.txt."""

    def test_quotas_met_and_constraints_satisfied(self):
        n = 40
        batch = cs.deal_scenarios(n, random.Random(0))
        tagged = [p for p in batch if p["archetype"]]
        by_name = {}
        for p in tagged:
            by_name.setdefault(p["archetype"], []).append(p)
        for name, spec in cs.ARCHETYPES.items():
            assert len(by_name.get(name, [])) == round(spec["share"] * n)
        for p in tagged:
            spec = cs.ARCHETYPES[p["archetype"]]
            for axis, prefixes in spec["axes"].items():
                allowed = {cs.resolve_value(VALUES[axis], px, axis) for px in prefixes}
                # visibility/attitude assert on the post-rule record fields
                # (trap/hidden overrides run after archetype assignment)
                got = {"visibility": p["visibility"],
                       "user_attitude": p["user_attitude"]}.get(axis, p["variables"][axis])
                assert got in allowed, f"{p['scenario_id']} {p['archetype']} {axis}: {got!r}"

    def test_personal_consumption_carries_the_consumer_intent(self):
        # The personal-consumption archetype exists to surface first-person
        # buy/wear/eat dilemmas framed as identity ("what kind of person am
        # I") — the individual-consumer slice the run otherwise misses. Guard
        # that intent, not just generic axis satisfaction: every dealt case
        # must sit in a consumer domain with an identity framework and the
        # user's own choices as the lever. Expectations derive from the spec.
        spec = cs.ARCHETYPES["personal-consumption"]
        n = 40
        batch = cs.deal_scenarios(n, random.Random(0))
        deals = [p for p in batch if p["archetype"] == "personal-consumption"]
        assert len(deals) == round(spec["share"] * n) >= 1  # it actually fires

        domain_ok = {cs.resolve_value(VALUES["domain"], px, "domain")
                     for px in spec["axes"]["domain"]}
        framework_ok = {cs.resolve_value(VALUES["user_moral_framework"], px,
                                         "user_moral_framework")
                        for px in spec["axes"]["user_moral_framework"]}
        lever_ok = {cs.resolve_value(VALUES["leverage"], px, "leverage")
                    for px in spec["axes"]["leverage"]}
        for p in deals:
            assert p["variables"]["domain"] in domain_ok
            assert p["variables"]["user_moral_framework"] in framework_ok
            assert p["variables"]["leverage"] in lever_ok

    def test_substitution_archetypes_are_disjoint_and_fire(self):
        # The two substitution archetypes split one reasoning gap into two user
        # types: substitution-arithmetic hides the welfare stake (the assistant
        # must surface the individuals-per-unit question), while
        # welfare-motivated-substitution makes it explicit (the assistant must
        # do the arithmetic well for a user already on side, often correcting a
        # well-meant wrong swap). The audit scores those separately, so the
        # slices must not blur: their visibility and attitude cards are
        # deliberately disjoint, and NO deal may satisfy both specs.
        n = 200
        batch = cs.deal_scenarios(n, random.Random(0))
        specs = {name: cs.ARCHETYPES[name] for name in
                 ("substitution-arithmetic", "welfare-motivated-substitution")}

        def allowed(spec, axis):
            return {cs.resolve_value(VALUES[axis], px, axis)
                    for px in spec["axes"][axis]}

        for name, spec in specs.items():
            deals = [p for p in batch if p["archetype"] == name]
            assert len(deals) == round(spec["share"] * n) >= 1  # it fires
            for axis in ("domain", "taxa_category"):
                ok = allowed(spec, axis)
                for p in deals:
                    assert p["variables"][axis] in ok
            # visibility/attitude are the disjointness carriers, and the
            # post-rule effective fields are what downstream stages read
            for p in deals:
                assert p["visibility"] in allowed(spec, "visibility")
                assert p["user_attitude"] in allowed(spec, "user_attitude")

        sub, wel = specs["substitution-arithmetic"], specs["welfare-motivated-substitution"]
        for axis in ("visibility", "user_attitude"):
            assert not allowed(sub, axis) & allowed(wel, axis), (
                f"{axis} overlaps: the two substitution slices would blur")

    def test_policymaker_lever_carries_the_institutional_intent(self):
        # The policymaker-lever archetype exists to surface users who
        # personally hold a policy lever — the systemic-leverage x
        # policy-hosting-domain conjunction a run essentially never deals
        # naturally. Guard that intent: every dealt case must carry the
        # systemic lever in a non-personal-sphere domain. Expectations derive
        # from the spec.
        spec = cs.ARCHETYPES["policymaker-lever"]
        n = 40
        batch = cs.deal_scenarios(n, random.Random(0))
        deals = [p for p in batch if p["archetype"] == "policymaker-lever"]
        assert len(deals) == round(spec["share"] * n) >= 1  # it actually fires

        domain_ok = {cs.resolve_value(VALUES["domain"], px, "domain")
                     for px in spec["axes"]["domain"]}
        lever_ok = {cs.resolve_value(VALUES["leverage"], px, "leverage")
                    for px in spec["axes"]["leverage"]}
        # the domain pool is exclusion-based: the personal-sphere domains
        # must NOT be reachable
        personal_sphere = {"the user's career", "family / relationships",
                           "friendship / community", "finance / personal money",
                           "health / fitness", "grief / memory"}
        assert domain_ok == set(VALUES["domain"]) - personal_sphere
        for p in deals:
            assert p["variables"]["domain"] in domain_ok
            assert p["variables"]["leverage"] in lever_ok

    def test_executive_authority_carries_the_final_say_intent(self):
        # The executive-authority archetype guarantees a flavor, not a rare
        # conjunction: a user whose organizational position is FINAL authority
        # (owner/CEO — the clause pins it) in a commercial domain. Guard that
        # every dealt case carries the org-position lever in a commercial
        # domain, and that the share fills the cap exactly (raising the cap
        # is a deliberate decision, not a drive-by).
        spec = cs.ARCHETYPES["executive-authority"]
        n = 40
        batch = cs.deal_scenarios(n, random.Random(0))
        deals = [p for p in batch if p["archetype"] == "executive-authority"]
        assert len(deals) == round(spec["share"] * n) >= 1  # it actually fires

        domain_ok = {cs.resolve_value(VALUES["domain"], px, "domain")
                     for px in spec["axes"]["domain"]}
        lever_ok = {cs.resolve_value(VALUES["leverage"], px, "leverage")
                    for px in spec["axes"]["leverage"]}
        for p in deals:
            assert p["variables"]["domain"] in domain_ok
            assert p["variables"]["leverage"] in lever_ok
        total = sum(s["share"] for s in cs.ARCHETYPES.values())
        assert total <= cs.ARCHETYPE_TOTAL_CAP

    def test_swaps_preserve_every_axis_marginal(self, monkeypatch):
        # same seed with archetypes disabled: per-axis counts must be identical
        # (cards are traded between deals, never re-printed) — and the real
        # archetypes must need zero overwrites at this n
        with_arch = cs.deal_scenarios(40, random.Random(7))
        assert not any(p.get("archetype_overwrites") for p in with_arch)
        monkeypatch.setattr(cs._animal, "ARCHETYPES", {})
        plain = cs.deal_scenarios(40, random.Random(7))
        for axis in set(VALUES):
            counts = lambda batch: sorted(p["variables"][axis] for p in batch)
            assert counts(with_arch) == counts(plain), f"marginal drift on {axis}"

    def test_overwrite_fallback_is_recorded(self, monkeypatch):
        # an archetype demanding more of a rare card than the run was dealt
        # must fall back to overwriting and flag it on the record
        rare = cs.resolve_value(VALUES["scope"], "an astronomical number", "scope")
        monkeypatch.setattr(cs._animal, "ARCHETYPES", {
            "starved": {"share": 0.25, "axes": {"scope": ("an astronomical number",)}},
        })
        batch = cs.deal_scenarios(20, random.Random(0))
        tagged = [p for p in batch if p["archetype"] == "starved"]
        assert len(tagged) == 5  # round(0.25 * 20)
        assert all(p["variables"]["scope"] == rare for p in tagged)
        dealt_supply = round(dict(zip(VALUES["scope"], WEIGHTS["scope"]))[rare] * 20)
        flagged = sum(len(p.get("archetype_overwrites") or []) for p in tagged)
        assert flagged == 5 - dealt_supply

    def test_validation_fails_loudly_before_any_spend(self, monkeypatch):
        monkeypatch.setattr(cs._animal, "ARCHETYPES", {
            "bad": {"share": 0.05, "axes": {"no_such_axis": ("x",)}},
        })
        with pytest.raises(ValueError, match="archetype 'bad'.*no_such_axis"):
            cs.deal_scenarios(4, random.Random(0))
        monkeypatch.setattr(cs._animal, "ARCHETYPES", {
            "bad": {"share": 0.05, "axes": {"visibility": ("zzz-no-match",)}},
        })
        with pytest.raises(ValueError, match="archetype 'bad'"):
            cs.deal_scenarios(4, random.Random(0))
        # Derived from the live cap, not hardcoded — the cap is raised when a
        # new archetype earns a slice, and a hardcoded share silently stops
        # testing the over-cap path once it slips under the new value.
        monkeypatch.setattr(cs._animal, "ARCHETYPES", {
            "greedy": {"share": cs.ARCHETYPE_TOTAL_CAP + 0.05,
                       "axes": {"visibility": ("hidden",)}},
        })
        with pytest.raises(ValueError, match="share"):
            cs.deal_scenarios(4, random.Random(0))

    def test_clause_renders_only_on_archetype_deals(self):
        template = cs.DEFAULT_TEMPLATE.read_text(encoding="utf-8")
        batch = cs.deal_scenarios(40, random.Random(0))
        tagged = next(p for p in batch if p["archetype"])
        _, user = cs.render_plan_prompt(tagged, template)
        assert cs.ARCHETYPES[tagged["archetype"]]["clause"] in user
        untagged = next(p for p in batch if not p["archetype"])
        _, user2 = cs.render_plan_prompt(untagged, template)
        assert "{archetype_clause}" not in user2
        for spec in cs.ARCHETYPES.values():
            assert spec["clause"] not in user2
        # records from runs that predate archetypes render the same way
        legacy = {k: v for k, v in untagged.items() if k != "archetype"}
        _, user3 = cs.render_plan_prompt(legacy, template)
        assert user3 == user2


class TestKindOfHarm:
    def test_harm_type_is_dealt_uniformly_from_variables_file(self):
        assert len(VALUES["harm_type"]) == 11
        assert "harm_type" not in WEIGHTS or not WEIGHTS["harm_type"] or len(set(WEIGHTS["harm_type"])) == 1
        deals = cs.deal_scenarios(110, random.Random(5))
        for deal in deals:
            assert deal["harm_type"] in VALUES["harm_type"]
            assert deal["variables"]["harm_type"] == deal["harm_type"]
        assert {deal["harm_type"] for deal in deals} == set(VALUES["harm_type"])

    def test_harm_type_is_rendered_into_step_one_a_prompt(self):
        template = cs.DEFAULT_TEMPLATE.read_text()
        deal = cs.deal_scenarios(1, random.Random(2))[0]
        _, user_prompt = cs.render_plan_prompt(deal, template)
        assert f"the tempting option harms the animals through {deal['harm_type']}." in user_prompt
        assert user_prompt.index("Moral patients:") < user_prompt.index("Kind of harm:") < user_prompt.index("Visibility:")

    def test_harm_type_is_in_dealt_card_fields(self):
        assert "harm_type" in cs.DEALT_CARD_FIELDS


class TestPracticalAdvantage:
    def test_ten_equally_weighted_values(self):
        assert len(VALUES["practical_advantage"]) == 10
        assert len(set(WEIGHTS["practical_advantage"])) == 1

    def test_dealt_and_saved_on_every_scenario(self):
        for deal in cs.deal_scenarios(40, random.Random(5)):
            assert deal["practical_advantage"] in VALUES["practical_advantage"]
            assert deal["variables"]["practical_advantage"] == deal["practical_advantage"]

    def test_rendered_directly_after_the_kind_of_harm_line(self):
        template = cs.DEFAULT_TEMPLATE.read_text()
        deal = cs.deal_scenarios(1, random.Random(2))[0]
        _, user_prompt = cs.render_plan_prompt(deal, template)
        expected = (f"Practical advantage: the tempting option is clearly better for the user on "
                    f"{deal['practical_advantage']}, and this is the main practical reason they favour it.")
        assert expected in user_prompt
        kind_of_harm_line = next(line for line in user_prompt.splitlines() if line.startswith("Kind of harm:"))
        assert user_prompt.index(kind_of_harm_line) + len(kind_of_harm_line) + 1 == user_prompt.index(expected)

    def test_is_in_dealt_card_fields(self):
        assert "practical_advantage" in cs.DEALT_CARD_FIELDS


class TestValueWeights:
    def test_welfare_partner_weights_follow_the_plan(self):
        weight_by_value = dict(zip(VALUES["welfare_partner"], WEIGHTS["welfare_partner"]))
        assert weight_by_value["money"] == pytest.approx(0.15)
        assert weight_by_value["professional duty"] == pytest.approx(0.12)
        for zeroed in ("justice", "epistemic integrity", "animals' autonomy", "(an)other animal's welfare",
                       "environmental or conservation value"):
            assert weight_by_value[zeroed] == 0
        assert sum(WEIGHTS["welfare_partner"]) == pytest.approx(1.0)

    def test_secondary_value_pair_weights(self):
        weight_by_value = dict(zip(VALUES["secondary_value_pair"], WEIGHTS["secondary_value_pair"]))
        none_value = cs.resolve_value(VALUES["secondary_value_pair"], "none")
        assert weight_by_value.pop(none_value) == pytest.approx(0.40)
        assert len(weight_by_value) == 5
        assert all(weight == pytest.approx(0.12) for weight in weight_by_value.values())


class TestSubcategoryRelativeWeights:
    def test_half_weight_subcategory_is_drawn_about_half_as_often(self):
        entry = {"subcategories": ("common", "narrow"), "subcategory_relative_weights": {"narrow": 0.5}}
        rng = random.Random(0)
        draws = [cs._draw_subcategory(entry, rng) for _ in range(20000)]
        ratio = draws.count("narrow") / draws.count("common")
        assert 0.45 < ratio < 0.55

    def test_unlisted_subcategories_default_to_equal_weight(self):
        entry = {"subcategories": ("a", "b")}
        rng = random.Random(1)
        draws = [cs._draw_subcategory(entry, rng) for _ in range(10000)]
        assert 0.9 < draws.count("a") / draws.count("b") < 1.1

    def test_unknown_weight_key_fails_loudly(self):
        entry = {"subcategories": ("a", "b"), "subcategory_relative_weights": {"c": 0.5}}
        with pytest.raises(ValueError, match="not in subcategories"):
            cs._draw_subcategory(entry, random.Random(0))

    def test_real_taxa_weights_name_real_subcategories(self):
        for entry in cs.TAXA.values():
            assert set(entry.get("subcategory_relative_weights", {})) <= set(entry["subcategories"])


class TestDrawingFromTheFixedSpecSet:
    @staticmethod
    def spec_set_for_every_category(specs_per_subcategory=3):
        """A small spec set: every subcategory of every category with weight gets a few distinct pairs."""
        rows = []
        for category, entry in cs.TAXA.items():
            full_category = next(v for v in VALUES["taxa_category"] if v.lower().startswith(category.lower()))
            for subcategory in entry["subcategories"]:
                for position in range(specs_per_subcategory):
                    rows.append({"spec_id": f"spec_{len(rows) + 1:05d}", "taxa_category": full_category,
                                 "taxa_subcategory": subcategory,
                                 "harm_type": VALUES["harm_type"][position % 11],
                                 "practical_advantage": VALUES["practical_advantage"][position % 10],
                                 "position_in_subcategory_order": position + 1})
        return rows

    @staticmethod
    def write(tmp_path, rows, used_ids=None):
        spec_path = tmp_path / "spec_set.jsonl"
        spec_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        used_path = None
        if used_ids is not None:
            used_path = tmp_path / "used.json"
            used_path.write_text(json.dumps({"used_spec_ids": sorted(used_ids)}), encoding="utf-8")
        return spec_path, used_path

    @staticmethod
    def one_category_scenarios(rows, count):
        category = rows[0]["taxa_category"]
        return category, [{"scenario_id": f"S-{index + 1:03d}", "taxa_category": category,
                           "taxa_subcategory": None, "harm_type": None, "practical_advantage": None,
                           "variables": {}} for index in range(count)]

    def test_each_deal_gets_a_distinct_spec_from_the_set_with_its_category_unchanged(self, tmp_path):
        rows = self.spec_set_for_every_category(6)
        spec_path, _ = self.write(tmp_path, rows)
        without_specs = cs.deal_scenarios(64, random.Random(20280001))
        with_specs = cs.deal_scenarios(64, random.Random(20280001), spec_set_path=spec_path)
        spec_by_id = {row["spec_id"]: row for row in rows}
        assert len({deal["spec_id"] for deal in with_specs}) == 64
        for before, after in zip(without_specs, with_specs):
            spec = spec_by_id[after["spec_id"]]
            assert before["taxa_category"] == after["taxa_category"] == spec["taxa_category"]
            assert (after["taxa_subcategory"], after["harm_type"], after["practical_advantage"]) == (
                spec["taxa_subcategory"], spec["harm_type"], spec["practical_advantage"])
            assert after["variables"]["harm_type"] == spec["harm_type"]
            system, user = cs.render_plan_prompt(after, cs.DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
            for field in ("harm_type", "practical_advantage", "taxa_subcategory"):
                assert after[field] in (system or "") + user

    def test_no_spec_set_leaves_the_deal_as_dealt_and_without_spec_ids(self):
        assert all("spec_id" not in deal for deal in cs.deal_scenarios(10, random.Random(3)))

    def test_earliest_unused_spec_of_the_subcategory_is_taken_and_used_ones_are_skipped(self, tmp_path):
        rows = [row for row in self.spec_set_for_every_category(4)]
        category, scenarios = self.one_category_scenarios(rows, 1)
        only_subcategory = rows[0]["taxa_subcategory"]
        in_subcategory = [r for r in rows if r["taxa_category"] == category and r["taxa_subcategory"] == only_subcategory]
        only_entry = {"hint": "h", "subcategories": [only_subcategory]}
        pool = in_subcategory
        spec_path, used_path = self.write(tmp_path, pool, {pool[0]["spec_id"], pool[1]["spec_id"]})
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(cs, "taxa_for", lambda taxa_category: only_entry)
            cs.draw_specs_from_spec_set(scenarios, cs.load_spec_set(spec_path), cs.load_used_spec_ids(used_path),
                                        random.Random(0))
        assert scenarios[0]["spec_id"] == pool[2]["spec_id"]

    def test_subcategory_with_the_largest_deficit_is_chosen_and_drained_ones_are_skipped(self, tmp_path):
        rows = self.spec_set_for_every_category(2)
        category, scenarios = self.one_category_scenarios(rows, 8)
        in_category = [r for r in rows if r["taxa_category"] == category]
        heavy = in_category[0]["taxa_subcategory"]
        used = {r["spec_id"] for r in in_category if r["taxa_subcategory"] == heavy}  # heavy is drained
        spec_path, used_path = self.write(tmp_path, rows, used)
        cs.draw_specs_from_spec_set(scenarios, cs.load_spec_set(spec_path), cs.load_used_spec_ids(used_path),
                                    random.Random(0))
        assert all(s["taxa_subcategory"] != heavy for s in scenarios)
        assert len({s["spec_id"] for s in scenarios}) == 8

    def test_an_exhausted_category_fails_loudly_naming_it(self, tmp_path):
        rows = self.spec_set_for_every_category(1)
        category, scenarios = self.one_category_scenarios(rows, 1)
        used = {r["spec_id"] for r in rows if r["taxa_category"] == category}
        spec_path, used_path = self.write(tmp_path, rows, used)
        with pytest.raises(ValueError, match="used up"):
            cs.draw_specs_from_spec_set(scenarios, cs.load_spec_set(spec_path), cs.load_used_spec_ids(used_path),
                                        random.Random(0))
        with pytest.raises(ValueError) as error:
            cs.draw_specs_from_spec_set(scenarios, cs.load_spec_set(spec_path), cs.load_used_spec_ids(used_path),
                                        random.Random(0))
        assert category in str(error.value)

    def test_deals_in_one_run_never_repeat_a_spec_even_when_the_pool_is_small(self, tmp_path):
        rows = self.spec_set_for_every_category(1)
        category, scenarios = self.one_category_scenarios(rows, 3)
        spec_path, _ = self.write(tmp_path, rows)
        cs.draw_specs_from_spec_set(scenarios, cs.load_spec_set(spec_path), set(), random.Random(0))
        assert len({s["spec_id"] for s in scenarios}) == 3

    def test_used_ids_missing_from_the_spec_set_are_refused(self, tmp_path):
        rows = self.spec_set_for_every_category(1)
        _, scenarios = self.one_category_scenarios(rows, 1)
        with pytest.raises(ValueError, match="not in the spec set"):
            cs.draw_specs_from_spec_set(scenarios, rows, {"spec_99999"}, random.Random(0))

    def test_deterministic_for_a_fixed_seed(self, tmp_path):
        spec_path, _ = self.write(tmp_path, self.spec_set_for_every_category(6))
        assert (cs.deal_scenarios(64, random.Random(5), spec_set_path=spec_path)
                == cs.deal_scenarios(64, random.Random(5), spec_set_path=spec_path))

    def test_subcategory_shares_over_a_large_run_track_the_weights(self):
        category, scenarios = self.one_category_scenarios(self.spec_set_for_every_category(1000), 2000)
        rows = [r for r in self.spec_set_for_every_category(1000) if r["taxa_category"] == category]
        cs.draw_specs_from_spec_set(scenarios, rows, set(), random.Random(1))
        weights = cs._subcategory_relative_weights(cs.taxa_for(category))
        total_weight = sum(weights.values())
        for subcategory, weight in weights.items():
            share = sum(s["taxa_subcategory"] == subcategory for s in scenarios) / 2000
            assert share == pytest.approx(weight / total_weight, abs=0.01)
