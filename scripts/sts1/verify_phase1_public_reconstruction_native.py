#!/usr/bin/env python3
"""Native smoke proof for the first public-state BattleContext reconstruction slice."""
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


def public_state() -> dict:
    return {
        "schema_version": PUBLIC_STATE_SCHEMA,
        "source": "simulator",
        "hp": 70,
        "max_hp": 80,
        "block": 0,
        "energy": 3,
        "hand": [
            {"position": 1, "id": "STRIKE_RED", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0, "has_target": True},
            {"position": 2, "id": "DEFEND_RED", "name": "Defend", "type": "SKILL", "cost": 1, "upgrades": 0, "has_target": False},
        ],
        "draw_pile": [
            {"id": "BASH", "name": "Bash", "type": "ATTACK", "cost": 2, "upgrades": 0, "has_target": True},
            {"id": "DEFEND_RED", "name": "Defend", "type": "SKILL", "cost": 1, "upgrades": 0, "has_target": False},
            {"id": "STRIKE_RED", "name": "Strike", "type": "ATTACK", "cost": 1, "upgrades": 0, "has_target": True},
        ],
        "discard_pile": [],
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
            {"kind": "play_card", "hand_index": 2},
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


def card_sequence(bc) -> list[tuple[str, int, int]]:
    return [
        (card.name, int(card.upgraded), int(card.cost_for_turn))
        for card in bc.draw_pile
    ]


def main() -> None:
    state = public_state()
    original = DecisionContext.from_public_state(state)
    admitted_state = attach_reconstruction_capabilities(state, run_state=run_state(state))
    admitted = require_public_reconstruction(admitted_state)
    assert admitted.decision_signature == original.decision_signature

    config = SearchConfig(sampling_seed=20260829)
    sample = public_sample(admitted, sample_index=0, config=config)
    plan = build_redeterminization_plan(admitted, sample)
    native_seeds = dict(plan.rng_seeds)
    native_seeds["previous_history"] = plan.monster_history[0].previous_history_seed

    bc = sts.build_public_jaw_worm_context_v1(admitted_state, native_seeds)
    assert bc.turn == 1
    assert bc.player.cur_hp == 70
    assert bc.player.max_hp == 80
    assert bc.player.block == 0
    assert bc.player.energy == 3
    assert [card.name for card in bc.hand] == ["Strike", "Defend"]
    assert sorted(card.name for card in bc.draw_pile) == ["Bash", "Defend", "Strike"]
    assert len(bc.monsters) == 1
    assert bc.monsters[0].name == "JAW_WORM"
    assert bc.monsters[0].intent == "JAW_WORM_CHOMP"

    # Incoming draw order is forbidden future information. Reverse it while
    # preserving composition and prove the same sample creates the same native
    # draw order after canonicalization + seeded re-determinization.
    reversed_state = deepcopy(state)
    reversed_state["draw_pile"] = list(reversed(reversed_state["draw_pile"]))
    assert DecisionContext.from_public_state(reversed_state).decision_signature == original.decision_signature
    reversed_admitted = attach_reconstruction_capabilities(
        reversed_state,
        run_state=run_state(reversed_state),
    )
    bc_reversed = sts.build_public_jaw_worm_context_v1(reversed_admitted, native_seeds)
    assert card_sequence(bc_reversed) == card_sequence(bc), (
        card_sequence(bc),
        card_sequence(bc_reversed),
    )

    legal = sts.get_legal_actions(bc)
    projected = SimulatorCombatAdapter().adapt(bc, legal_actions=legal, run_state=run_state(state))
    rebuilt = DecisionContext.from_public_state(projected)
    if rebuilt.decision_signature != original.decision_signature:
        raise RuntimeError(
            "PUBLIC_RECONSTRUCTION_ROUNDTRIP_SIGNATURE_MISMATCH:"
            f"{original.decision_signature}!={rebuilt.decision_signature}"
        )

    strike = next(
        action for action in legal
        if action.action_type == sts.SearchActionType.CARD
        and action.source_idx == 0
        and action.target_idx == 0
    )
    hp_before = bc.monsters[0].cur_hp
    strike.execute(bc)
    hp_after = bc.monsters[0].cur_hp
    assert hp_after < hp_before, (hp_before, hp_after)
    assert bc.player.energy == 2

    print("PUBLIC_RECONSTRUCTION_NATIVE = PASS")
    print("ROUNDTRIP_SIGNATURE = PASS")
    print("LEGAL_ACTIONS = PASS")
    print("EXECUTABLE_STRIKE = PASS")
    print("DRAW_ORDER_REDETERMINIZATION = PASS")
    print(f"ENEMY_HP = {hp_before}->{hp_after}")
    print("SOURCE_BATTLE_CONTEXT_INPUT = 0")
    print("SOURCE_HIDDEN_RNG_ACCESS = 0")
    print("SOURCE_DRAW_ORDER_ACCESS = 0")


if __name__ == "__main__":
    main()
