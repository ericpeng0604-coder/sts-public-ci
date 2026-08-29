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


def finesse_state() -> dict:
    strike = lambda: card("Strike_R", "Strike", "ATTACK", 1, target=True)
    defend = lambda: card("Defend_R", "Defend", "SKILL", 1, target=False)
    bash = lambda: card("Bash", "Bash", "ATTACK", 2, target=True)
    finesse = lambda: card("Finesse", "Finesse", "SKILL", 0, target=False)
    hand = [strike(), defend(), strike(), defend(), strike()]
    for index, item in enumerate(hand, start=1):
        item["position"] = index
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "real_game",
        "hp": 70,
        "max_hp": 80,
        "block": 2,
        "energy": 3,
        "hand": hand,
        "draw_pile": [],
        "discard_pile": [strike(), defend(), bash(), strike(), defend(), finesse()],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [{
            "index": 0,
            "name": "JAW_WORM",
            "hp": 31,
            "max_hp": 40,
            "block": 0,
            "intent": "JAW_WORM_CHOMP",
            "intent_damage": 11,
            "intent_hits": 1,
            "is_gone": False,
            "powers": [],
        }],
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
        "relics": state["relics"], "potions": state["potions"], "gold": state["gold"],
        "floor": state["floor"], "act": state["act"], "character": state["character"],
        "ascension_level": state["ascension_level"], "room": state["room"],
        "screen_type": state["screen_type"], "screen_choices": [], "rewards": [], "map_choices": [],
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


def admitted(state: dict, trace: dict | None = None) -> dict:
    return attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=trace,
    )


def assert_rejected(state: dict, trace: dict | None = None) -> None:
    result = admitted(state, aux() if trace is None else trace)
    assert result["reconstruction"]["public_player_state_complete"] is False


def test_finesse_identity_is_fixed_cost_card_reconstructable() -> None:
    result = assess_public_cards(finesse_state())
    assert result.allowed is True, result.reasons
    assert result.card_count == 11


def test_finesse_normal_exact_slice_is_admitted_without_policy_mutation() -> None:
    state = finesse_state()
    signature = DecisionContext.from_public_state(state).decision_signature
    result = admitted(state, aux())
    assert result["reconstruction"]["public_player_state_complete"] is True, result["reconstruction"]
    assert result["reconstruction"]["public_card_instance_state_complete"] is True
    context = require_public_reconstruction(result)
    assert context.decision_signature == signature
    assert "reconstruction_aux" not in context.state


def test_finesse_without_aux_still_fails_closed() -> None:
    result = admitted(finesse_state(), None)
    assert result["reconstruction"]["public_player_state_complete"] is False


def test_finesse_rejects_incomplete_or_wrong_counter_trace() -> None:
    trace = aux(); trace["complete"] = False
    assert_rejected(finesse_state(), trace)
    trace = aux(); trace["attacks_played_this_turn"] = 1; trace["skills_played_this_turn"] = 0
    assert_rejected(finesse_state(), trace)
    trace = aux(); trace["cards_played_this_turn"] = 2
    assert_rejected(finesse_state(), trace)


def test_finesse_rejects_wrong_energy_block_upgrade_and_cost() -> None:
    state = finesse_state(); state["energy"] = 2
    assert_rejected(state)
    state = finesse_state(); state["block"] = 4
    assert_rejected(state)
    state = finesse_state(); state["discard_pile"][-1]["upgrades"] = 1
    assert_rejected(state)
    state = finesse_state(); state["discard_pile"][-1]["cost"] = 1
    result = admitted(state, aux())
    assert result["reconstruction"]["public_card_instance_state_complete"] is False


def test_finesse_rejects_wrong_pile_shape_and_played_card_location() -> None:
    state = finesse_state(); state["hand"].pop()
    assert_rejected(state)

    state = finesse_state()
    finesse = state["discard_pile"].pop()
    moved = state["hand"].pop()
    state["hand"].append(finesse)
    state["discard_pile"].append(moved)
    assert_rejected(state)


def test_finesse_rejects_wrong_deck_identity_and_extra_effect_surface() -> None:
    state = finesse_state()
    state["discard_pile"][-1] = card("Shrug It Off", "Shrug It Off", "SKILL", 1, target=False)
    assert_rejected(state)

    state = finesse_state()
    state["powers"] = [{"name": "DEXTERITY", "amount": 1}]
    assert_rejected(state)
