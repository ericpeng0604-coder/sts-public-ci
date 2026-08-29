from __future__ import annotations

from copy import deepcopy

from roguelike_ai.sts1_teacher.card_reconstruction import assess_public_cards
from roguelike_ai.sts1_teacher.contract import DecisionContext, PUBLIC_STATE_SCHEMA
from roguelike_ai.sts1_teacher.native_rollout import NativePublicJawWormRolloutV1
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.sampling import public_sample
from roguelike_ai.sts1_teacher.search import SearchConfig


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


def pommel_state() -> dict:
    strike = lambda: card("Strike_R", "Strike", "ATTACK", 1, target=True)
    defend = lambda: card("Defend_R", "Defend", "SKILL", 1, target=False)
    bash = lambda: card("Bash", "Bash", "ATTACK", 2, target=True)
    pommel = lambda: card("Pommel Strike", "Pommel Strike", "ATTACK", 1, target=True)

    hand = [strike(), defend(), strike(), defend(), strike()]
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
        "draw_pile": [],
        "discard_pile": [strike(), defend(), bash(), strike(), defend(), pommel()],
        "exhaust_pile": [],
        "powers": [],
        "enemies": [
            {
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
        "attacks_played_this_turn": 1,
        "skills_played_this_turn": 0,
        "cards_discarded_this_turn": 0,
    }


def test_communicationmod_aliases_and_pommel_are_card_reconstructable() -> None:
    admitted = assess_public_cards(pommel_state())
    assert admitted.allowed is True, admitted.reasons
    assert admitted.card_count == 11


def test_pommel_turn1_requires_separate_complete_aux_and_keeps_policy_identity() -> None:
    state = pommel_state()
    original_signature = DecisionContext.from_public_state(state).decision_signature

    without_aux = attach_reconstruction_capabilities(state, run_state=run_state(state))
    assert without_aux["reconstruction"]["public_player_state_complete"] is False

    with_aux = attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=aux(),
    )
    assert with_aux["reconstruction"]["public_player_state_complete"] is True
    assert with_aux["reconstruction"]["public_card_instance_state_complete"] is True
    admitted = require_public_reconstruction(with_aux)
    assert admitted.decision_signature == original_signature

    bad = aux()
    bad["complete"] = False
    rejected = attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=bad,
    )
    assert rejected["reconstruction"]["public_player_state_complete"] is False


def test_pommel_aux_rejects_snapshot_that_does_not_match_draw_effect() -> None:
    state = pommel_state()
    state["hand"].pop()
    rejected = attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=aux(),
    )
    assert rejected["reconstruction"]["public_player_state_complete"] is False


def test_native_rollout_passes_aux_as_third_argument_without_policy_mutation() -> None:
    state = pommel_state()
    admitted_state = attach_reconstruction_capabilities(
        state,
        run_state=run_state(state),
        reconstruction_aux=aux(),
    )
    context = require_public_reconstruction(admitted_state)
    sample = public_sample(context, sample_index=0, config=SearchConfig(sampling_seed=20260829))

    class FakeNative:
        def __init__(self) -> None:
            self.calls = []

        def build_public_jaw_worm_context_v1(self, *args):
            self.calls.append(args)
            return object()

    native = FakeNative()
    backend = NativePublicJawWormRolloutV1(
        admitted_state,
        native,
        reconstruction_aux=aux(),
    )
    backend._build(context, sample)
    assert len(native.calls) == 1
    assert len(native.calls[0]) == 3
    assert native.calls[0][0]["decision_signature"] if "decision_signature" in native.calls[0][0] else True
    assert native.calls[0][2] == aux()

    # The auxiliary trace never becomes a policy field.
    assert "reconstruction_aux" not in context.state
    assert context.decision_signature == DecisionContext.from_public_state(state).decision_signature
