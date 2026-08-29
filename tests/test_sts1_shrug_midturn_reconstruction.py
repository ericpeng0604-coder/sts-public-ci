from __future__ import annotations

from copy import deepcopy

from roguelike_ai.sts1_teacher.card_reconstruction import assess_public_cards
from roguelike_ai.sts1_teacher.contract import DecisionContext, PUBLIC_STATE_SCHEMA
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities


def card(card_id: str, name: str, card_type: str, cost: int, *, target: bool) -> dict:
    return {
        "id": card_id,
        "name": name,
        "type": card_type,
        "cost": cost,
        "upgrades": 0,
        "has_target": target,
        "is_playable": True,
    }


def shrug_state() -> dict:
    strike = lambda: card("Strike_R", "Strike", "ATTACK", 1, target=True)
    defend = lambda: card("Defend_R", "Defend", "SKILL", 1, target=False)
    bash = lambda: card("Bash", "Bash", "ATTACK", 2, target=True)
    shrug = lambda: card("Shrug It Off", "Shrug It Off", "SKILL", 1, target=False)
    hand = [strike(), defend(), strike(), defend(), strike()]
    for index, item in enumerate(hand, start=1):
        item["position"] = index
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "real_game",
        "hp": 70,
        "max_hp": 80,
        "block": 8,
        "energy": 2,
        "hand": hand,
        "draw_pile": [],
        "discard_pile": [strike(), defend(), bash(), strike(), defend(), shrug()],
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


def admitted(state: dict, reconstruction_aux: dict | None) -> dict:
    return attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=reconstruction_aux,
    )


def test_shrug_identity_is_card_reconstructable() -> None:
    result = assess_public_cards(shrug_state())
    assert result.allowed is True, result.reasons
    assert result.card_count == 11


def test_shrug_needs_skill_aux_and_does_not_change_policy_identity() -> None:
    state = shrug_state()
    original = DecisionContext.from_public_state(state).decision_signature
    assert admitted(state, None)["reconstruction"]["public_player_state_complete"] is False

    accepted = admitted(state, aux())
    assert accepted["reconstruction"]["public_player_state_complete"] is True
    assert accepted["reconstruction"]["public_card_instance_state_complete"] is True
    assert require_public_reconstruction(accepted).decision_signature == original
    assert "reconstruction_aux" not in require_public_reconstruction(accepted).state


def test_shrug_requires_exact_eight_block_in_first_slice() -> None:
    state = shrug_state()
    for wrong in (0, 5, 11):
        candidate = deepcopy(state)
        candidate["block"] = wrong
        result = admitted(candidate, aux())
        assert result["reconstruction"]["public_player_state_complete"] is False
        assert f"shrug_turn1_block_mismatch:{wrong!r}" in result["reconstruction"]["player_admission_reasons"]


def test_shrug_upgrade_and_wrong_counter_mix_fail_closed() -> None:
    upgraded = shrug_state()
    shrug = upgraded["discard_pile"][-1]
    shrug["upgrades"] = 1
    result = admitted(upgraded, aux())
    assert result["reconstruction"]["public_player_state_complete"] is False

    wrong_aux = aux()
    wrong_aux["attacks_played_this_turn"] = 1
    wrong_aux["skills_played_this_turn"] = 1
    result = admitted(shrug_state(), wrong_aux)
    assert result["reconstruction"]["public_player_state_complete"] is False
    assert any(reason.startswith("draw_aux_slice_unsupported:") for reason in result["reconstruction"]["player_admission_reasons"])


def test_pommel_counter_shape_cannot_unlock_shrug_state() -> None:
    wrong = aux()
    wrong["attacks_played_this_turn"] = 1
    wrong["skills_played_this_turn"] = 0
    result = admitted(shrug_state(), wrong)
    assert result["reconstruction"]["public_player_state_complete"] is False
    assert "pommel_turn1_deck_composition_mismatch" in result["reconstruction"]["player_admission_reasons"]
