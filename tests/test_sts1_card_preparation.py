from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from roguelike_ai.sts1_phase3.card_preparation import (
    UNKNOWN, analyze_profiles, base_name, card_facts, deck_facts,
    load_native_types, parse_native_types, shadow_clash_skip,
)

TYPES = {"Clash": "ATTACK", "Strike": "ATTACK", "Defend": "SKILL", "Offering": "SKILL",
         "Pommel Strike": "ATTACK", "Dropkick": "ATTACK", "Flex": "SKILL", "Inflame": "POWER"}


def reward():
    return {"screen": "REWARDS", "act": 2,
            "before": {"deck": [{"name": "Strike", "upgrades": 0}, {"name": "Defend", "upgrades": 0}]},
            "choices": [{"kind": "card", "choice": "card", "card_name": "Clash+"},
                        {"kind": "card", "choice": "skip"}], "selected_index": 0,
            "selected": {"kind": "card", "choice": "card", "card_name": "Clash+"}}


def test_upgrade_and_conditional_effects_are_separate():
    assert card_facts({"name": "Pommel Strike+", "upgrades": 1}, TYPES)["effects"]["draw"] == 2
    drop = card_facts({"name": "Dropkick", "upgrades": 0}, TYPES)["effects"]
    assert "draw" not in drop and drop["conditional_draw"] == 1
    assert card_facts({"name": "Flex", "upgrades": 0}, TYPES)["effects"]["temporary_strength"] == 2
    assert "persistent_strength" not in card_facts({"name": "Flex", "upgrades": 0}, TYPES)["effects"]
    assert card_facts({"name": "Offering+", "upgrades": 1}, TYPES)["effects"]["hp_cost"] == 6


@pytest.mark.parametrize("card", [{"name": "Offering+", "upgrades": 0},
                                  {"name": "Offering", "upgrades": True},
                                  {"name": "Offering", "upgrades": 1},
                                  {"name": "Offering+2", "upgrades": 2},
                                  {"name": "vocab:255", "upgrades": 0}])
def test_unknown_or_inconsistent_effects_remain_unknown(card):
    assert card_facts(card, TYPES)["effects"] == UNKNOWN


def test_type_does_not_imply_readiness_or_reviewed_effects():
    facts = deck_facts([{"name": "Strike", "upgrades": 0}], TYPES)
    assert facts["type_counts"] == {"ATTACK": 1}
    assert facts["unreviewed_effect_cards"] == 1
    assert facts["readiness"] == UNKNOWN
    assert base_name("Clash+") == "Clash" and base_name("vocab:255") == "vocab:255"


def test_shadow_uses_exact_legal_skip_and_does_not_mutate():
    d = reward()
    before = deepcopy(d)
    assert shadow_clash_skip(d, TYPES) == (1, "shadow_clash_mixed_deck")
    assert d == before


@pytest.mark.parametrize("change,reason", [
    (lambda d: d["before"].update(deck=[]), "unknown_deck"),
    (lambda d: d["before"].update(deck=[{"name": "vocab:1"}]), "unknown_card_type"),
    (lambda d: d["before"].update(deck=[{"name": "Strike"}]), "attack_only_deck"),
    (lambda d: d["choices"].append({"kind": "card", "choice": "skip"}), "ambiguous_or_missing_skip"),
    (lambda d: d["choices"].pop(), "ambiguous_or_missing_skip"),
    (lambda d: d.update(selected_index=True), "incomplete_recommendation"),
])
def test_shadow_fails_closed(change, reason):
    d = reward()
    change(d)
    assert shadow_clash_skip(d, TYPES)[1] == reason


def test_aggregate_is_exposure_not_counterfactual_outcome():
    d = reward()
    p = {"schema_version": "sts1-boss-preparation-diagnostic-v1", "decision_records": [d], "encounter_records": []}
    before = deepcopy(p)
    out = analyze_profiles([p], TYPES)
    assert out["reward_outcomes"] == {"card": 1, "skip": 0, "empty_reward": 0}
    assert out["shadow_clash_runs"] == out["shadow_clash_skip_opportunities"] == 1
    assert out["policy_changed"] is False and out["win_rate_improvement"] == UNKNOWN
    assert p == before
    d["selected"] = {"kind": "card", "choice": "skip"}
    with pytest.raises(ValueError, match="mismatch"):
        analyze_profiles([p], TYPES)


def test_native_array_alignment_and_source_drift(tmp_path):
    text = 'cardNames[] = {"INVALID","Clash","Defend"}; cardTypes[] = {CardType::INVALID,CardType::ATTACK,CardType::SKILL}; cardColors[] = {CardColor::INVALID,CardColor::RED,CardColor::RED};'
    assert parse_native_types(text)["Clash"] == "ATTACK"
    # Type must not depend on native color classification.
    assert parse_native_types(text.replace('CardColor::RED', 'CardColor::BLUE'))["Defend"] == "SKILL"
    with pytest.raises(ValueError, match="misaligned"):
        parse_native_types(text.replace(',CardType::SKILL', ''))
    (tmp_path / "include/constants").mkdir(parents=True)
    (tmp_path / "include/constants/Cards.h").write_text(text)
    with pytest.raises(ValueError, match="drift"):
        load_native_types(tmp_path)


def test_audit_rejects_heldout_metadata():
    path = Path(__file__).parents[1] / "scripts/sts1/sts1_card_preparation_audit.py"
    spec = importlib.util.spec_from_file_location("card_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    h = {"type": "diagnostic_trace_header_v1", "run_metadata": {
        "stage": "train", "purpose": "baseline_live_resource_diagnosis", "round_id": module.ROUND,
        "code_commit": module.FROZEN_COLLECTOR, "mcts_sims": 2000, "pair_index": 1}}
    module.validate_header(h, 1)
    for key, value in [("stage", "probe"), ("round_id", "other"), ("code_commit", "drift"), ("mcts_sims", 1), ("pair_index", True)]:
        bad = deepcopy(h)
        bad["run_metadata"][key] = value
        with pytest.raises(ValueError, match="Train"):
            module.validate_header(bad, 1)
