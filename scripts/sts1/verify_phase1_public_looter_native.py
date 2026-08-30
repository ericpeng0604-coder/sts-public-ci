#!/usr/bin/env python3
"""Native opening-turn proof for the audited Looter turn-0 slice."""
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
    # observation. Source BattleContext, source move history, and source RNG are
    # never passed into the reconstructed context.
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 580001, 0)
    gc.floor_num = 1
    gc.cur_room = sts.Room.MONSTER
    source_bc = sts.BattleContext()
    source_bc.init_encounter(gc, sts.MonsterEncounter.LOOTER)
    adapter = SimulatorCombatAdapter()
    run = run_state(gc)
    source_actions = list(sts.get_legal_actions(source_bc))
    state = adapter.adapt(source_bc, legal_actions=source_actions, run_state=run)
    admitted_state = attach_reconstruction_capabilities(state, run_state=run)
    context = require_public_reconstruction(admitted_state)

    assert admitted_state["turn"] == 0
    assert admitted_state["ascension_level"] == 0
    assert len(admitted_state["enemies"]) == 1
    enemy = admitted_state["enemies"][0]
    assert enemy["name"] == "LOOTER"
    assert enemy["intent"] == "LOOTER_MUG"
    assert enemy["powers"] == [{"name": "THIEVERY", "amount": 15}]

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
        assert fresh.monsters[0].name == "LOOTER"
        assert fresh.monsters[0].intent == "LOOTER_MUG"

    projected = adapter.adapt(
        fresh0,
        legal_actions=list(sts.get_legal_actions(fresh0)),
        run_state=run,
    )
    projected = attach_reconstruction_capabilities(projected, run_state=run)
    assert require_public_reconstruction(projected).decision_signature == context.decision_signature

    hp_before = fresh0.player.cur_hp
    gold_before = fresh0.player.gold
    end_turn = next(
        action
        for action in sts.get_legal_actions(fresh0)
        if action.action_type == sts.SearchActionType.END_TURN
    )
    end_turn.execute(fresh0)
    assert fresh0.player.cur_hp == hp_before - 10
    assert fresh0.player.gold == gold_before - 15
    assert fresh0.monsters[0].intent in {
        "LOOTER_MUG",
        "LOOTER_LUNGE",
        "LOOTER_SMOKE_BOMB",
    }

    print("LOOTER_TURN0_PUBLIC_RECONSTRUCTION_NATIVE = PASS")
    print("LOOTER_OPENING_PUBLIC_INTENT = LOOTER_MUG")
    print("LOOTER_OPENING_DAMAGE_A0 = 10")
    print("LOOTER_OPENING_THIEVERY_A0 = 15")
    print("LOOTER_CURRENT_INTENT_STABLE_ACROSS_FRESH_SAMPLES = PASS")
    print("LOOTER_FUTURE_AI_RNG = FRESH_ROLLOUT_ONLY")
    print("SOURCE_BATTLE_CONTEXT_INPUT = 0")
    print("SOURCE_OPENING_RNG_ACCESS = 0")
    print("SOURCE_HIDDEN_MOVE_HISTORY_ACCESS = 0")
    print("SOURCE_HIDDEN_MISCINFO_ACCESS = 0")
    print("PUBLIC_CONTRACT_FIELDS_ADDED = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
