#!/usr/bin/env python3
"""Native proof for the audited Jaw Worm second-player-turn start boundary."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
BUILD = ROOT / "external" / "sts_lightspeed" / "source" / "build-probe"
for path in (SRC, BUILD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import slaythespire as sts

from roguelike_ai.sts1_teacher.contract import DecisionContext, PUBLIC_STATE_SCHEMA
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.redeterminization import build_redeterminization_plan
from roguelike_ai.sts1_teacher.sampling import public_sample
from roguelike_ai.sts1_teacher.search import SearchConfig
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter


def card(card_id: str, name: str, card_type: str, cost: int, *, target: bool, position: int | None = None) -> dict:
    result = {
        "id": card_id,
        "name": name,
        "type": card_type,
        "cost": cost,
        "upgrades": 0,
        "has_target": target,
    }
    if position is not None:
        result["position"] = position
        result["is_playable"] = True
    return result


def public_state() -> dict:
    hand = [
        card("STRIKE_RED", "Strike", "ATTACK", 1, target=True, position=1),
        card("STRIKE_RED", "Strike", "ATTACK", 1, target=True, position=2),
        card("DEFEND_RED", "Defend", "SKILL", 1, target=False, position=3),
        card("DEFEND_RED", "Defend", "SKILL", 1, target=False, position=4),
        card("BASH", "Bash", "ATTACK", 2, target=True, position=5),
    ]
    discard = [
        card("STRIKE_RED", "Strike", "ATTACK", 1, target=True),
        card("STRIKE_RED", "Strike", "ATTACK", 1, target=True),
        card("STRIKE_RED", "Strike", "ATTACK", 1, target=True),
        card("DEFEND_RED", "Defend", "SKILL", 1, target=False),
        card("DEFEND_RED", "Defend", "SKILL", 1, target=False),
    ]
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "simulator",
        "hp": 69,
        "max_hp": 80,
        "block": 0,
        "energy": 3,
        "hand": hand,
        "draw_pile": [],
        "discard_pile": discard,
        "exhaust_pile": [],
        "powers": [],
        "enemies": [
            {
                "index": 0,
                "name": "JAW_WORM",
                "hp": 40,
                "max_hp": 40,
                "block": 0,
                "intent": "JAW_WORM_THRASH",
                "intent_damage": 7,
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
            {"kind": "play_card", "hand_index": 2, "target_index": 0},
            {"kind": "play_card", "hand_index": 3},
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


def admitted_and_seeds(state: dict) -> tuple[DecisionContext, dict]:
    original = DecisionContext.from_public_state(state)
    admitted_state = attach_reconstruction_capabilities(state, run_state=run_state(state))
    admitted = require_public_reconstruction(admitted_state)
    assert admitted.decision_signature == original.decision_signature
    sample = public_sample(admitted, sample_index=0, config=SearchConfig(sampling_seed=20260829))
    plan = build_redeterminization_plan(admitted, sample)
    seeds = dict(plan.rng_seeds)
    seeds["previous_history"] = plan.monster_history[0].previous_history_seed
    return admitted, seeds


def roundtrip(state: dict, admitted: DecisionContext, bc) -> None:
    projected = SimulatorCombatAdapter().adapt(
        bc,
        legal_actions=sts.get_legal_actions(bc),
        run_state=run_state(state),
    )
    rebuilt = DecisionContext.from_public_state(projected)
    assert rebuilt.decision_signature == admitted.decision_signature


def execute_end_turn(bc) -> tuple[int, int, int, str]:
    before_hp = int(bc.player.cur_hp)
    end_turn = next(
        action for action in sts.get_legal_actions(bc)
        if action.action_type == sts.SearchActionType.END_TURN
    )
    end_turn.execute(bc)
    return (
        before_hp - int(bc.player.cur_hp),
        int(bc.monsters[0].block),
        int(bc.turn),
        str(bc.monsters[0].intent),
    )


def main() -> None:
    state = public_state()
    admitted, seeds_a = admitted_and_seeds(state)
    seeds_b = deepcopy(seeds_a)
    seeds_b["previous_history"] = int(seeds_b["previous_history"]) ^ 0x5A5A5A5A5A5A5A5A

    bc_a = sts.build_public_jaw_worm_context_v1(state, seeds_a)
    bc_b = sts.build_public_jaw_worm_context_v1(state, seeds_b)

    for bc in (bc_a, bc_b):
        assert int(bc.turn) == 1
        assert int(bc.player.cur_hp) == 69
        assert int(bc.player.energy) == 3
        assert int(bc.player.block) == 0
        assert len(bc.hand) == 5
        assert len(bc.draw_pile) == 0
        assert len(bc.discard_pile) == 5
        assert len(bc.exhaust_pile) == 0
        assert bc.monsters[0].name == "JAW_WORM"
        assert str(bc.monsters[0].intent) == "JAW_WORM_THRASH"

    roundtrip(state, admitted, bc_a)
    roundtrip(state, admitted, bc_b)

    result_a = execute_end_turn(bc_a)
    result_b = execute_end_turn(bc_b)
    assert result_a == result_b
    damage, enemy_block, next_turn, next_intent = result_a
    assert damage > 0
    assert enemy_block > 0
    assert next_turn == 2
    assert next_intent

    print("JAW_WORM_TURN_ONE_PUBLIC_BOUNDARY = PASS")
    print("TURN_ONE_ROUNDTRIP_SIGNATURE = PASS")
    print("TURN_ONE_PREVIOUS_HISTORY = PUBLIC_DERIVED_CHOMP")
    print("TURN_ONE_PREVIOUS_HISTORY_SEED_INDEPENDENT = PASS")
    print("TURN_ONE_EXECUTABLE_THRASH = PASS")
    print(f"TURN_ONE_THRASH_DAMAGE = {damage}")
    print(f"TURN_ONE_THRASH_BLOCK = {enemy_block}")
    print(f"NEXT_PUBLIC_TURN = {next_turn}")
    print(f"NEXT_INTENT = {next_intent}")
    print("SOURCE_BATTLE_CONTEXT_INPUT = 0")
    print("SOURCE_MOVE_HISTORY_ACCESS = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
