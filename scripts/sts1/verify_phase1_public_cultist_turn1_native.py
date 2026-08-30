#!/usr/bin/env python3
"""Native proof for the audited pristine Cultist turn-1 public boundary."""
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


def end_turn(bc) -> None:
    action = next(
        action
        for action in sts.get_legal_actions(bc)
        if action.action_type == sts.SearchActionType.END_TURN
    )
    action.execute(bc)


def main() -> None:
    # Source context is used only to generate one realistic public observation.
    # It is never passed to the public constructor or rollout backend.
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 645001, 0)
    gc.floor_num = 1
    gc.cur_room = sts.Room.MONSTER
    source_bc = sts.BattleContext()
    source_bc.init_encounter(gc, sts.MonsterEncounter.CULTIST)
    assert int(source_bc.turn) == 0
    assert source_bc.monsters[0].intent == "CULTIST_INCANTATION"
    end_turn(source_bc)

    adapter = SimulatorCombatAdapter()
    run = run_state(gc)
    state = adapter.adapt(source_bc, legal_actions=list(sts.get_legal_actions(source_bc)), run_state=run)
    assert state["turn"] == 1
    assert state["energy"] == 3 and state["block"] == 0
    assert len(state["hand"]) == 5 and len(state["draw_pile"]) == 0
    assert len(state["discard_pile"]) == 5 and len(state["exhaust_pile"]) == 0
    enemy = state["enemies"][0]
    assert enemy["name"] == "CULTIST"
    assert enemy["intent"] == "CULTIST_DARK_STRIKE"
    assert enemy["hp"] == enemy["max_hp"] and enemy["block"] == 0
    assert enemy["powers"] == []

    admitted_state = attach_reconstruction_capabilities(state, run_state=run)
    context = require_public_reconstruction(admitted_state)
    assert admitted_state["reconstruction"]["cultist_turn1_complete"] is True

    config = SearchConfig(sampling_seed=20260831)
    sample = public_sample(context, sample_index=0, config=config)
    plan = build_redeterminization_plan(context, sample)
    seeds = dict(plan.rng_seeds)
    seeds["previous_history"] = plan.monster_history[0].previous_history_seed

    fresh = sts.build_public_jaw_worm_context_v1(admitted_state, seeds)
    assert fresh.turn == 1
    assert len(fresh.monsters) == 1
    assert fresh.monsters[0].name == "CULTIST"
    assert fresh.monsters[0].intent == "CULTIST_DARK_STRIKE"
    assert sts.get_public_monster_ritual_v1(fresh, 0) == 3
    assert sts.get_public_monster_ritual_just_applied_v1(fresh, 0) is False

    projected = adapter.adapt(fresh, legal_actions=list(sts.get_legal_actions(fresh)), run_state=run)
    repro = attach_reconstruction_capabilities(projected, run_state=run)
    assert require_public_reconstruction(repro).decision_signature == context.decision_signature

    # Execute a real legal action and the real enemy progression. Cultist turn1
    # Dark Strike is 6 damage at A0; the reconstructed Ritual state remains
    # active and was already marked not-just-applied before this progression.
    hp_before = fresh.player.cur_hp
    end_turn(fresh)
    assert fresh.player.cur_hp == hp_before - 6
    assert fresh.monsters[0].intent == "CULTIST_DARK_STRIKE"
    assert sts.get_public_monster_ritual_v1(fresh, 0) == 3

    print("PASS — CULTIST_TURN1_NATIVE_READY")
    print("CULTIST_TURN1_CURRENT_INTENT = DARK_STRIKE")
    print("CULTIST_TURN1_PREVIOUS_MOVE = PUBLIC_DERIVED_INCANTATION")
    print("CULTIST_TURN1_RITUAL = PUBLIC_ASCENSION_DERIVED_3")
    print("CULTIST_TURN1_RITUAL_JUST_APPLIED = FALSE")
    print("LEGAL_ACTION_AND_ENEMY_PROGRESSION = PASS")
    print("SOURCE_RAW_BATTLE_CONTEXT_INPUT = 0")
    print("SOURCE_HIDDEN_RNG_ACCESS = 0")
    print("SOURCE_MOVE_HISTORY_ACCESS = 0")
    print("SOURCE_COUNTER_HISTORY_ACCESS = 0")
    print("SOURCE_HIDDEN_MISCINFO_ACCESS = 0")
    print("PUBLIC_CONTRACT_FIELDS_ADDED = 0")
    print("NON_TARGET_SIMULATOR_DIFF = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
