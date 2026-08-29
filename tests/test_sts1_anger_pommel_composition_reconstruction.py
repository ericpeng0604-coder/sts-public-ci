from __future__ import annotations

from copy import deepcopy

from roguelike_ai.sts1_teacher.composition_reconstruction import (
    assess_public_anger_pommel_composition,
)
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


def composition_state() -> dict:
    strike = lambda: card("Strike_R", "Strike", "ATTACK", 1, target=True)
    defend = lambda: card("Defend_R", "Defend", "SKILL", 1, target=False)
    bash = lambda: card("Bash", "Bash", "ATTACK", 2, target=True)
    pommel = lambda: card("Pommel Strike", "Pommel Strike", "ATTACK", 1, target=True)
    anger = lambda: card("Anger", "Anger", "ATTACK", 0, target=True)

    hand = [strike(), defend(), strike(), defend()]
    for index, item in enumerate(hand, start=1):
        item["position"] = index
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "real_game",
        "hp": 70,
        "max_hp": 80,
        "block": 0,
        "energy": 2,
        "hand": hand,
        "draw_pile": [strike()],
        "discard_pile": [
            strike(),
            defend(),
            bash(),
            strike(),
            defend(),
            pommel(),
            anger(),
            anger(),
        ],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [
            {
                "index": 0,
                "name": "JAW_WORM",
                "hp": 25,
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
        "cards_played_this_turn": 2,
        "attacks_played_this_turn": 2,
        "skills_played_this_turn": 0,
        "cards_discarded_this_turn": 0,
    }


def test_exact_anger_pommel_composition_is_admitted_without_policy_mutation() -> None:
    state = composition_state()
    original_signature = DecisionContext.from_public_state(state).decision_signature

    direct = assess_public_anger_pommel_composition(state, aux())
    assert direct.allowed is True, direct.reasons

    admitted_state = attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=aux(),
    )
    marker = admitted_state["reconstruction"]
    assert marker["anger_pommel_composition_complete"] is True
    assert marker["public_player_state_complete"] is True
    assert marker["public_card_instance_state_complete"] is True
    admitted = require_public_reconstruction(admitted_state)
    assert admitted.decision_signature == original_signature
    assert "reconstruction_aux" not in admitted.state


def test_composition_fails_closed_without_complete_counter_trace() -> None:
    state = composition_state()
    missing = attach_reconstruction_capabilities(state, run_state=run_state(state))
    assert missing["reconstruction"]["anger_pommel_composition_complete"] is False
    assert missing["reconstruction"]["public_player_state_complete"] is False

    bad = aux()
    bad["cards_played_this_turn"] = 1
    rejected = attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=bad,
    )
    assert rejected["reconstruction"]["anger_pommel_composition_complete"] is False
    assert rejected["reconstruction"]["public_player_state_complete"] is False


def test_composition_requires_the_pommel_draw_shape() -> None:
    state = composition_state()
    state["discard_pile"].append(state["draw_pile"].pop())
    rejected = assess_public_anger_pommel_composition(state, aux())
    assert rejected.allowed is False
    assert any("pile_shape" in reason for reason in rejected.reasons)


def test_composition_requires_publicly_visible_generated_anger_copy() -> None:
    state = composition_state()
    removed = False
    new_discard = []
    for item in state["discard_pile"]:
        if not removed and str(item["id"]).upper() == "ANGER":
            removed = True
            continue
        new_discard.append(item)
    state["discard_pile"] = new_discard
    rejected = assess_public_anger_pommel_composition(state, aux())
    assert rejected.allowed is False
    assert any("generated_copy_not_publicly_proven" in reason for reason in rejected.reasons)


def test_composition_does_not_admit_unrelated_third_play() -> None:
    state = deepcopy(composition_state())
    bad = aux()
    bad["cards_played_this_turn"] = 3
    bad["attacks_played_this_turn"] = 3
    rejected = assess_public_anger_pommel_composition(state, bad)
    assert rejected.allowed is False
    assert any("cards_played_this_turn" in reason for reason in rejected.reasons)
