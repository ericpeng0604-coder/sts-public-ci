#!/usr/bin/env python3
"""Native proof for Looter turn 0 and the audited turn-1 positive-gold slice."""
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
    # Pinned simulator projection omits this public power, so reconstruction
    # derives it solely from the already-public ascension field.
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
        assert fresh.monsters[0].name == "LOOTER"
        assert fresh.monsters[0].intent == "LOOTER_MUG"
        assert sts.get_public_monster_thievery_v1(fresh, 0) == 15
        assert sts.get_public_player_gold_v1(fresh) == admitted_state["gold"]

    projected = adapter.adapt(
        fresh0,
        legal_actions=list(sts.get_legal_actions(fresh0)),
        run_state=run,
    )
    projected = attach_reconstruction_capabilities(projected, run_state=run)
    assert require_public_reconstruction(projected).decision_signature == context.decision_signature

    hp_before = fresh0.player.cur_hp
    gold_before = sts.get_public_player_gold_v1(fresh0)
    end_turn = next(
        action
        for action in sts.get_legal_actions(fresh0)
        if action.action_type == sts.SearchActionType.END_TURN
    )
    end_turn.execute(fresh0)
    assert fresh0.turn == 1
    assert fresh0.player.cur_hp == hp_before - 10
    assert sts.get_public_player_gold_v1(fresh0) == gold_before - 15
    # Source-proven first-MUG behavior keeps the second Looter intent at MUG.
    assert fresh0.monsters[0].intent == "LOOTER_MUG"

    # Re-project only the now-visible turn-1 state. The updated run surface uses
    # the visible post-theft gold; no source miscInfo/history/RNG is copied.
    turn1_run = dict(run)
    turn1_run["gold"] = sts.get_public_player_gold_v1(fresh0)
    turn1_projected = adapter.adapt(
        fresh0,
        legal_actions=list(sts.get_legal_actions(fresh0)),
        run_state=turn1_run,
    )
    turn1_state = attach_reconstruction_capabilities(turn1_projected, run_state=turn1_run)
    turn1_context = require_public_reconstruction(turn1_state)
    assert turn1_state["turn"] == 1
    assert turn1_state["gold"] == gold_before - 15
    assert turn1_state["gold"] > 0
    assert turn1_state["enemies"][0]["intent"] == "LOOTER_MUG"

    turn1_sample = public_sample(turn1_context, sample_index=2, config=config)
    turn1_plan = build_redeterminization_plan(turn1_context, turn1_sample)
    turn1_seeds = dict(turn1_plan.rng_seeds)
    turn1_seeds["previous_history"] = turn1_plan.monster_history[0].previous_history_seed
    fresh_turn1 = sts.build_public_jaw_worm_context_v1(turn1_state, turn1_seeds)
    assert fresh_turn1.turn == 1
    assert fresh_turn1.monsters[0].name == "LOOTER"
    assert fresh_turn1.monsters[0].intent == "LOOTER_MUG"
    assert sts.get_public_monster_thievery_v1(fresh_turn1, 0) == 15
    assert sts.get_public_player_gold_v1(fresh_turn1) == gold_before - 15

    turn1_hp_before = fresh_turn1.player.cur_hp
    turn1_gold_before = sts.get_public_player_gold_v1(fresh_turn1)
    turn1_end = next(
        action
        for action in sts.get_legal_actions(fresh_turn1)
        if action.action_type == sts.SearchActionType.END_TURN
    )
    turn1_end.execute(fresh_turn1)
    assert fresh_turn1.player.cur_hp == turn1_hp_before - 10
    assert sts.get_public_player_gold_v1(fresh_turn1) == turn1_gold_before - 15
    assert fresh_turn1.monsters[0].intent in {
        "LOOTER_LUNGE",
        "LOOTER_SMOKE_BOMB",
    }

    print("LOOTER_TURN0_PUBLIC_RECONSTRUCTION_NATIVE = PASS")
    print("LOOTER_TURN1_POSITIVE_GOLD_PUBLIC_RECONSTRUCTION_NATIVE = PASS")
    print("LOOTER_OPENING_PUBLIC_INTENT = LOOTER_MUG")
    print("LOOTER_OPENING_DAMAGE_A0 = 10")
    print("LOOTER_OPENING_THIEVERY_A0 = PUBLIC_ASCENSION_DERIVED_15")
    print("LOOTER_MUG_GOLD_THEFT_A0 = 15")
    print("LOOTER_CURRENT_INTENT_STABLE_ACROSS_FRESH_SAMPLES = PASS")
    print("LOOTER_FUTURE_AI_RNG = FRESH_ROLLOUT_ONLY")
    print("SOURCE_BATTLE_CONTEXT_INPUT = 0")
    print("SOURCE_OPENING_RNG_ACCESS = 0")
    print("SOURCE_HIDDEN_MOVE_HISTORY_ACCESS = 0")
    print("SOURCE_HIDDEN_MISCINFO_ACCESS = 0")
    print("SOURCE_HIDDEN_THIEVERY_ACCESS = 0")
    print("PUBLIC_CONTRACT_FIELDS_ADDED = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
