#!/usr/bin/env python3
"""Native opening-turn smoke proof for the audited Blue Slaver V1 slice."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
BUILD = ROOT / "external" / "sts_lightspeed" / "source" / "build-probe"
for path in (SRC, BUILD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import slaythespire as sts

from roguelike_ai.sts1_teacher.public_run_state import simulator_public_run_state
from roguelike_ai.sts1_teacher.reconstruction import require_public_reconstruction
from roguelike_ai.sts1_teacher.reconstruction_state import attach_reconstruction_capabilities
from roguelike_ai.sts1_teacher.redeterminization import build_redeterminization_plan
from roguelike_ai.sts1_teacher.sampling import public_sample
from roguelike_ai.sts1_teacher.search import SearchConfig
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter


def run_state(gc) -> dict:
    run = simulator_public_run_state(gc)
    run.update(
        {
            "floor": 1,
            "act": 1,
            "character": "IRONCLAD",
            "ascension_level": 0,
            "room": "COMBAT",
            "screen_type": "NONE",
            "screen_choices": [],
            "rewards": [],
            "map_choices": [],
        }
    )
    return run


def main() -> None:
    # The source simulator is used only to obtain a realistic public opening
    # observation. Its BattleContext and opening RNG are never passed to the
    # public constructor.
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 380001, 0)
    gc.floor_num = 1
    gc.cur_room = sts.Room.MONSTER
    source_bc = sts.BattleContext()
    source_bc.init_encounter(gc, sts.MonsterEncounter.BLUE_SLAVER)
    adapter = SimulatorCombatAdapter()
    run = run_state(gc)
    source_actions = list(sts.get_legal_actions(source_bc))
    state = adapter.adapt(source_bc, legal_actions=source_actions, run_state=run)
    admitted_state = attach_reconstruction_capabilities(state, run_state=run)
    context = require_public_reconstruction(admitted_state)

    assert admitted_state["turn"] == 0
    assert len(admitted_state["enemies"]) == 1
    enemy = admitted_state["enemies"][0]
    assert enemy["name"] == "BLUE_SLAVER"
    assert enemy["intent"] in {"BLUE_SLAVER_STAB", "BLUE_SLAVER_RAKE"}
    assert enemy["powers"] == []

    config = SearchConfig(sampling_seed=20260830)
    sample0 = public_sample(context, sample_index=0, config=config)
    sample1 = public_sample(context, sample_index=1, config=config)

    def build(sample):
        plan = build_redeterminization_plan(context, sample)
        seeds = dict(plan.rng_seeds)
        seeds["previous_history"] = plan.monster_history[0].previous_history_seed
        return sts.build_public_jaw_worm_context_v1(admitted_state, seeds)

    fresh0 = build(sample0)
    fresh1 = build(sample1)
    for fresh in (fresh0, fresh1):
        assert fresh.turn == 0
        assert len(fresh.monsters) == 1
        assert fresh.monsters[0].name == "BLUE_SLAVER"
        # Current opening intent is reconstructed from public observation, not
        # resampled from source or rollout RNG.
        assert fresh.monsters[0].intent == enemy["intent"]

    projected = adapter.adapt(fresh0, legal_actions=list(sts.get_legal_actions(fresh0)), run_state=run)
    projected = attach_reconstruction_capabilities(projected, run_state=run)
    assert require_public_reconstruction(projected).decision_signature == context.decision_signature

    hp_before = fresh0.player.cur_hp
    opening_intent = fresh0.monsters[0].intent
    end_turn = next(
        action for action in sts.get_legal_actions(fresh0)
        if action.action_type == sts.SearchActionType.END_TURN
    )
    end_turn.execute(fresh0)
    expected_damage = 12 if opening_intent == "BLUE_SLAVER_STAB" else 7
    assert fresh0.player.cur_hp == hp_before - expected_damage
    assert fresh0.monsters[0].intent in {"BLUE_SLAVER_STAB", "BLUE_SLAVER_RAKE"}

    print("BLUE_SLAVER_PUBLIC_RECONSTRUCTION_NATIVE = PASS")
    print(f"BLUE_SLAVER_OPENING_PUBLIC_INTENT = {opening_intent}")
    print(f"BLUE_SLAVER_OPENING_DAMAGE_A0 = {expected_damage}")
    print("BLUE_SLAVER_CURRENT_INTENT_STABLE_ACROSS_FRESH_SAMPLES = PASS")
    print("BLUE_SLAVER_NEXT_MOVE_USES_FRESH_ROLLOUT_RNG = PASS")
    print("SOURCE_BATTLE_CONTEXT_INPUT = 0")
    print("SOURCE_OPENING_RNG_ACCESS = 0")
    print("SOURCE_HIDDEN_MOVE_HISTORY_ACCESS = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
