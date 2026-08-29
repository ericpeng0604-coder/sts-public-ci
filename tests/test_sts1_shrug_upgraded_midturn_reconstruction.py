from __future__ import annotations

from copy import deepcopy

from roguelike_ai.sts1_teacher.card_reconstruction import assess_public_cards
from roguelike_ai.sts1_teacher.contract import DecisionContext, PUBLIC_STATE_SCHEMA
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities


def card(card_id: str, name: str, card_type: str, cost: int, *, target: bool, upgrades: int = 0) -> dict:
    return {
        "id": card_id,
        "name": name,
        "type": card_type,
        "cost": cost,
        "upgrades": upgrades,
        "has_target": target,
        "is_playable": True,
    }


def upgraded_shrug_state() -> dict:
    strike = lambda: card("Strike_R", "Strike", "ATTACK", 1, target=True)
    defend = lambda: card("Defend_R", "Defend", "SKILL", 1, target=False)
    bash = lambda: card("Bash", "Bash", "ATTACK", 2, target=True)
    shrug_plus = lambda: card("Shrug It Off", "Shrug It Off", "SKILL", 1, target=False, upgrades=1)
    hand = [strike(), defend(), strike(), defend(), strike()]
    for index, item in enumerate(hand, start=1):
        item["position"] = index
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "real_game",
        "hp": 70,
        "max_hp": 80,
        "block": 11,
        "energy": 2,
        "hand": hand,
        "draw_pile": [],
        "discard_pile": [strike(), defend(), bash(), strike(), defend(), shrug_plus()],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [
            {
                "index": 0,
                "name": "JAW_WORM",
                "hp": 40,
                "max_hp": 40,
                "block": 0,
                "intent": "JAW_WORM_CHOMP",
                "intent_damage": 11,
                "intent_hits": 1,
                "is_gone": False,
                "powers": [],
            }
        ],
        "turn": 1,
        "combat_active": True,
        "relics": [{"id": "BURNING_BLOOD", "counter": -1}],
        "potions": [
            {"index": 0, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
            {"index": 1, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
            {"index": 2, "id": "EMPTY_POTION_SLOT", "name": "EMPTY_POTION_SLOT", "empty": True},
        ],
        "gold": 99,
        "floor": 2,
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
            {"kind": "play_card", "hand_index": 2},
            {"kind": "play_card", "hand_index": 3, "target_index": 0},
            {"kind": "play_card", "hand_index": 4},
            {"kind": "play_card", "hand_index": 5, "target_index": 0},
            {"kind": "end_turn"},
        ],
    }


def run_state(state: dict) -> dict:
    return {
        "relics": state["relics"],
        "potions": state["potions"],
        "gold": state["gold"],
        "floor": state["floor"],
        "act": state["act"],
        "character": state["character"],
        "ascension_level": state["ascension_level"],
        "room": state["room"],
        "screen_type": state["screen_type"],
        "screen_choices": [],
        "rewards": [],
        "map_choices": [],
        "reconstruction": {
            "schema_version": "sts1-public-reconstruction-v1",
            "public_player_state_complete": False,
            "public_card_instance_state_complete": False,
            "public_relic_state_complete": True,
            "public_potion_state_complete": True,
            "public_enemy_state_complete": False,
        },
    }


def aux() -> dict:
    return {
        "schema_version": "sts1-public-reconstruction-aux-v1",
        "source": "communicationmod_command_trace_v1",
        "turn": 1,
        "complete": True,
        "cards_played_this_turn": 1,
        "attacks_played_this_turn": 0,
        "skills_played_this_turn": 1,
        "cards_discarded_this_turn": 0,
    }


def admitted(state: dict) -> dict:
    return attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=aux(),
    )


def test_upgraded_shrug_is_admitted_with_exact_eleven_block() -> None:
    state = upgraded_shrug_state()
    card_result = assess_public_cards(state)
    assert card_result.allowed is True, card_result.reasons

    original = DecisionContext.from_public_state(state).decision_signature
    result = admitted(state)
    assert result["reconstruction"]["public_player_state_complete"] is True
    assert result["reconstruction"]["public_card_instance_state_complete"] is True
    assert require_public_reconstruction(result).decision_signature == original


def test_upgraded_shrug_with_base_eight_block_fails_closed() -> None:
    state = upgraded_shrug_state()
    state["block"] = 8
    result = admitted(state)
    assert result["reconstruction"]["public_player_state_complete"] is False
    assert "shrug_turn1_block_mismatch:8" in result["reconstruction"]["player_admission_reasons"]


def test_normal_shrug_with_eleven_block_still_fails_closed() -> None:
    state = upgraded_shrug_state()
    state["discard_pile"][-1]["upgrades"] = 0
    result = admitted(state)
    assert result["reconstruction"]["public_player_state_complete"] is False
    assert "shrug_turn1_block_mismatch:11" in result["reconstruction"]["player_admission_reasons"]


def test_only_shrug_may_be_upgraded_in_this_slice() -> None:
    state = upgraded_shrug_state()
    state["hand"][0]["upgrades"] = 1
    result = admitted(state)
    assert result["reconstruction"]["public_player_state_complete"] is False
    assert any(
        reason.startswith("shrug_turn1_upgrade_mismatch:")
        for reason in result["reconstruction"]["player_admission_reasons"]
    )


def test_upgrade_count_two_still_fails_closed() -> None:
    state = upgraded_shrug_state()
    state["discard_pile"][-1]["upgrades"] = 2
    assert assess_public_cards(state).allowed is False
    result = admitted(state)
    assert result["reconstruction"]["public_player_state_complete"] is False
    assert "shrug_turn1_upgrade_surface_unreachable" in result["reconstruction"]["player_admission_reasons"]


def test_aux_shape_is_unchanged_by_upgrade() -> None:
    normal_like = upgraded_shrug_state()
    normal_like["discard_pile"][-1]["upgrades"] = 0
    normal_like["block"] = 8
    upgraded = upgraded_shrug_state()
    assert aux() == aux()
    assert admitted(normal_like)["reconstruction"]["public_player_state_complete"] is True
    assert admitted(upgraded)["reconstruction"]["public_player_state_complete"] is True
