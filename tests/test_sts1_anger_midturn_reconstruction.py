from __future__ import annotations

from copy import deepcopy

from roguelike_ai.sts1_teacher.anger_reconstruction import assess_public_anger_midturn
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


def anger_state() -> dict:
    strike = lambda: card("Strike_R", "Strike", "ATTACK", 1, target=True)
    defend = lambda: card("Defend_R", "Defend", "SKILL", 1, target=False)
    bash = lambda: card("Bash", "Bash", "ATTACK", 2, target=True)
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
        "energy": 3,
        "hand": hand,
        "draw_pile": [strike()],
        "discard_pile": [strike(), defend(), bash(), strike(), defend(), anger(), anger()],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [
            {
                "index": 0,
                "name": "JAW_WORM",
                "hp": 34,
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
        "cards_played_this_turn": 1,
        "attacks_played_this_turn": 1,
        "skills_played_this_turn": 0,
        "cards_discarded_this_turn": 0,
    }


def test_anger_generated_copy_is_proven_from_public_piles_plus_bounded_aux() -> None:
    state = anger_state()
    admission = assess_public_anger_midturn(state, aux())
    assert admission.allowed is True, admission.reasons

    # The old generic card gate remains unchanged; only the exact Anger slice
    # can override it through reconstruction metadata.
    assert assess_public_cards(state).allowed is False


def test_anger_override_keeps_policy_identity_and_requires_complete_aux() -> None:
    state = anger_state()
    signature = DecisionContext.from_public_state(state).decision_signature

    without_aux = attach_reconstruction_capabilities(state, run_state=run_state(state))
    assert without_aux["reconstruction"]["anger_midturn_complete"] is False
    assert without_aux["reconstruction"]["public_player_state_complete"] is False

    with_aux = attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=aux(),
    )
    assert with_aux["reconstruction"]["anger_midturn_complete"] is True
    assert with_aux["reconstruction"]["public_player_state_complete"] is True
    assert with_aux["reconstruction"]["public_card_instance_state_complete"] is True
    admitted = require_public_reconstruction(with_aux)
    assert admitted.decision_signature == signature
    assert "reconstruction_aux" not in admitted.state

    bad_aux = aux()
    bad_aux["complete"] = False
    rejected = attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=bad_aux,
    )
    assert rejected["reconstruction"]["anger_midturn_complete"] is False


def test_anger_requires_exactly_two_public_discard_copies_after_one_play() -> None:
    state = anger_state()
    state["discard_pile"].pop()
    rejected = assess_public_anger_midturn(state, aux())
    assert rejected.allowed is False
    assert any("anger_" in reason for reason in rejected.reasons)

    moved = anger_state()
    generated = moved["discard_pile"].pop()
    moved["draw_pile"].append(generated)
    rejected = assess_public_anger_midturn(moved, aux())
    assert rejected.allowed is False
    assert any("generated_copy_not_publicly_proven" in reason for reason in rejected.reasons)


def test_anger_zero_cost_and_attack_counter_are_fail_closed() -> None:
    wrong_energy = anger_state()
    wrong_energy["energy"] = 2
    assert assess_public_anger_midturn(wrong_energy, aux()).allowed is False

    wrong_counter = aux()
    wrong_counter["skills_played_this_turn"] = 1
    assert assess_public_anger_midturn(anger_state(), wrong_counter).allowed is False

    upgraded = anger_state()
    for pile in ("hand", "draw_pile", "discard_pile"):
        for item in upgraded[pile]:
            if item["name"] == "Anger":
                item["upgrades"] = 1
    assert assess_public_anger_midturn(upgraded, aux()).allowed is False
