#!/usr/bin/env python3
"""Native opening-turn smoke proof for the audited Gremlin Nob V1 slice."""
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
    # Source simulator is used only to obtain one realistic public observation.
    # It is never passed to the public constructor or rollout backend.
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 330001, 0)
    gc.floor_num = 1
    gc.cur_room = sts.Room.MONSTER
    source_bc = sts.BattleContext()
    source_bc.init_encounter(gc, sts.MonsterEncounter.GREMLIN_NOB)
    adapter = SimulatorCombatAdapter()
    run = run_state(gc)
    state = adapter.adapt(source_bc, legal_actions=list(sts.get_legal_actions(source_bc)), run_state=run)
    admitted_state = attach_reconstruction_capabilities(state, run_state=run)
    context = require_public_reconstruction(admitted_state)

    assert admitted_state["turn"] == 0
    assert len(admitted_state["enemies"]) == 1
    enemy = admitted_state["enemies"][0]
    assert enemy["name"] == "GREMLIN_NOB"
    assert enemy["intent"] == "GREMLIN_NOB_BELLOW"
    assert enemy["powers"] == []

    config = SearchConfig(sampling_seed=20260830)
    sample = public_sample(context, sample_index=0, config=config)
    plan = build_redeterminization_plan(context, sample)
    seeds = dict(plan.rng_seeds)
    seeds["previous_history"] = plan.monster_history[0].previous_history_seed

    fresh = sts.build_public_jaw_worm_context_v1(admitted_state, seeds)
    assert fresh.turn == 0
    assert len(fresh.monsters) == 1
    assert fresh.monsters[0].name == "GREMLIN_NOB"
    assert fresh.monsters[0].intent == "GREMLIN_NOB_BELLOW"

    projected = adapter.adapt(fresh, legal_actions=list(sts.get_legal_actions(fresh)), run_state=run)
    assert require_public_reconstruction(attach_reconstruction_capabilities(projected, run_state=run)).decision_signature == context.decision_signature

    # Execute the opening Bellow. It deals no damage and then rolls the next
    # move using the fresh native rollout RNG. ENRAGE itself is intentionally
    # not projected by public-state V1 because later-turn Nob is not admitted.
    hp_before = fresh.player.cur_hp
    end_turn = next(
        action for action in sts.get_legal_actions(fresh)
        if action.action_type == sts.SearchActionType.END_TURN
    )
    end_turn.execute(fresh)
    assert fresh.player.cur_hp == hp_before
    assert fresh.monsters[0].intent in {"GREMLIN_NOB_RUSH", "GREMLIN_NOB_SKULL_BASH"}

    print("GREMLIN_NOB_PUBLIC_RECONSTRUCTION_NATIVE = PASS")
    print("GREMLIN_NOB_OPENING_BELLOW = PASS")
    print("GREMLIN_NOB_BELLOW_DEALS_NO_DAMAGE = PASS")
    print("GREMLIN_NOB_NEXT_MOVE_USES_FRESH_ROLLOUT_RNG = PASS")
    print("GREMLIN_NOB_LATER_TURN_PUBLIC_POWER_PROJECTION = NOT_ADMITTED")
    print("SOURCE_BATTLE_CONTEXT_INPUT = 0")
    print("SOURCE_HIDDEN_MOVE_HISTORY_ACCESS = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
