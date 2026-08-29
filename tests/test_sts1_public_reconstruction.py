import pytest

from roguelike_ai.sts1_teacher.contract import PUBLIC_STATE_SCHEMA, PublicStateContractError
from roguelike_ai.sts1_teacher.reconstruction import (
    PUBLIC_RECONSTRUCTION_SCHEMA,
    assess_public_reconstruction,
    require_public_reconstruction,
)


def complete_state() -> dict:
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "simulator",
        "hp": 70,
        "max_hp": 80,
        "block": 0,
        "energy": 3,
        "hand": [{"position": 1, "id": "STRIKE_RED", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0, "has_target": True}],
        "draw_pile": [{"id": "DEFEND_RED", "name": "Defend", "type": "SKILL", "cost": 1, "upgrades": 0, "has_target": False}],
        "discard_pile": [],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [{"index": 0, "name": "Cultist", "hp": 48, "max_hp": 48, "block": 0, "intent": "DARK_STRIKE", "intent_damage": 6, "intent_hits": 1, "is_gone": False, "powers": []}],
        "turn": 1,
        "combat_active": True,
        "relics": [{"id": "BURNING_BLOOD", "data": 0}],
        "potions": [],
        "gold": 99,
        "floor": 1,
        "act": 1,
        "character": "IRONCLAD",
        "ascension_level": 0,
        "room": "COMBAT",
        "screen_type": "NONE",
        "screen_choices": [],
        "rewards": [],
        "map_choices": [],
        "legal_actions": [
            {"kind": "play_card", "hand_index": 1, "target_index": 0},
            {"kind": "end_turn"},
        ],
        "reconstruction": {
            "schema_version": PUBLIC_RECONSTRUCTION_SCHEMA,
            "public_player_state_complete": True,
            "public_card_instance_state_complete": True,
            "public_relic_state_complete": True,
            "public_potion_state_complete": True,
            "public_enemy_state_complete": True,
        },
    }


def test_complete_public_state_is_admitted() -> None:
    state = complete_state()
    result = assess_public_reconstruction(state)
    assert result.allowed is True
    assert result.reasons == ()
    assert result.decision_signature
    assert require_public_reconstruction(state).decision_signature == result.decision_signature


def test_missing_marker_fails_closed() -> None:
    state = complete_state()
    state.pop("reconstruction")
    result = assess_public_reconstruction(state)
    assert result.allowed is False
    assert "missing_reconstruction_capability_marker" in result.reasons


def test_unproven_public_surface_fails_closed() -> None:
    state = complete_state()
    state["reconstruction"]["public_potion_state_complete"] = False
    result = assess_public_reconstruction(state)
    assert result.allowed is False
    assert "capability_not_proven:public_potion_state_complete" in result.reasons


def test_missing_required_field_is_not_normalized_into_fake_completeness() -> None:
    state = complete_state()
    state.pop("relics")
    result = assess_public_reconstruction(state)
    assert result.allowed is False
    assert "missing_public_field:relics" in result.reasons


def test_hidden_rng_is_still_rejected_by_public_contract() -> None:
    state = complete_state()
    state["rng_state"] = 123
    result = assess_public_reconstruction(state)
    assert result.allowed is False
    assert any(reason.startswith("public_state_contract:hidden_information_forbidden") for reason in result.reasons)
    with pytest.raises(PublicStateContractError):
        require_public_reconstruction(state)
