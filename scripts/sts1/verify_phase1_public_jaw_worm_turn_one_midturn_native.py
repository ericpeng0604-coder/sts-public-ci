#!/usr/bin/env python3
"""Native proof for audited Jaw Worm second-player-turn midturn reconstruction."""
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

from roguelike_ai.sts1_teacher.contract import DecisionContext
from roguelike_ai.sts1_teacher.player_reconstruction import assess_public_player
from roguelike_ai.sts1_teacher.simulator import SimulatorCombatAdapter
from verify_phase1_public_jaw_worm_turn_one_native import admitted_and_seeds, public_state, run_state


def enum_name(value) -> str:
    name = getattr(value, "name", None)
    return str(name if isinstance(name, str) else value).rsplit(".", 1)[-1].upper()


def first_defend_action(bc):
    hand = list(bc.hand)
    for action in sts.get_legal_actions(bc):
        if enum_name(action.action_type) != "CARD":
            continue
        idx = int(action.source_idx)
        if str(hand[idx].name).upper() == "DEFEND":
            return action
    raise AssertionError("missing legal Defend action")


def first_strike_action(bc):
    hand = list(bc.hand)
    for action in sts.get_legal_actions(bc):
        if enum_name(action.action_type) != "CARD":
            continue
        idx = int(action.source_idx)
        if str(hand[idx].name).upper() == "STRIKE":
            return action
    raise AssertionError("missing legal Strike action")


def project(bc, state: dict) -> dict:
    return SimulatorCombatAdapter().adapt(
        bc,
        legal_actions=sts.get_legal_actions(bc),
        run_state=run_state(state),
    )


def main() -> None:
    start = public_state()
    _, start_seeds = admitted_and_seeds(start)

    # Produce a real reachable midturn state from the already-proven fresh turn-1
    # boundary.  This source transition is diagnostic evidence only; formal
    # reconstruction below receives only the resulting public projection.
    source_bc = sts.build_public_jaw_worm_context_v1(start, start_seeds)
    first_defend_action(source_bc).execute(source_bc)
    mid = project(source_bc, start)

    assert int(mid["turn"]) == 1
    assert int(mid["energy"]) == 2
    assert int(mid["block"]) == 5
    assert len(mid["hand"]) == 4
    assert len(mid["draw_pile"]) == 0
    assert len(mid["discard_pile"]) == 6
    assert len(mid["exhaust_pile"]) == 0
    admission = assess_public_player(mid)
    assert admission.allowed, admission.reasons

    admitted, seeds_a = admitted_and_seeds(mid)
    seeds_b = deepcopy(seeds_a)
    seeds_b["previous_history"] = int(seeds_b["previous_history"]) ^ 0xA5A5A5A5A5A5A5A5

    rebuilt_a = sts.build_public_jaw_worm_context_v1(mid, seeds_a)
    rebuilt_b = sts.build_public_jaw_worm_context_v1(mid, seeds_b)

    for rebuilt in (rebuilt_a, rebuilt_b):
        assert int(rebuilt.turn) == 1
        assert int(rebuilt.player.energy) == 2
        assert int(rebuilt.player.block) == 5
        assert len(rebuilt.hand) == 4
        assert len(rebuilt.draw_pile) == 0
        assert len(rebuilt.discard_pile) == 6
        assert len(rebuilt.exhaust_pile) == 0
        roundtrip = project(rebuilt, mid)
        assert DecisionContext.from_public_state(roundtrip).decision_signature == admitted.decision_signature

    # The previous monster move is publicly derivable as opening CHOMP for this
    # turn, so changing the history sample must not alter the current public
    # reconstruction or the next public action result.
    first_strike_action(rebuilt_a).execute(rebuilt_a)
    first_strike_action(rebuilt_b).execute(rebuilt_b)
    after_a = project(rebuilt_a, mid)
    after_b = project(rebuilt_b, mid)
    assert DecisionContext.from_public_state(after_a).decision_signature == DecisionContext.from_public_state(after_b).decision_signature
    assert int(after_a["energy"]) == 1
    assert int(after_a["block"]) == 5
    assert len(after_a["hand"]) == 3
    assert len(after_a["discard_pile"]) == 7
    assert assess_public_player(after_a).allowed

    print("JAW_WORM_TURN_ONE_MIDTURN_PUBLIC_RECONSTRUCTION = PASS")
    print("MIDTURN_SOURCE_PUBLIC_PROJECTION = PASS")
    print("MIDTURN_ADMISSION = PASS")
    print("MIDTURN_ROUNDTRIP_SIGNATURE = PASS")
    print("MIDTURN_PUBLIC_COUNTER_DERIVATION = PASS")
    print("MIDTURN_PREVIOUS_HISTORY_SEED_INDEPENDENT = PASS")
    print("MIDTURN_SECOND_ACTION_EXECUTABLE = PASS")
    print("MIDTURN_ENERGY_AFTER_DEFEND_STRIKE = 1")
    print("MIDTURN_BLOCK_AFTER_DEFEND_STRIKE = 5")
    print("SOURCE_COUNTER_HISTORY_INPUT = 0")
    print("PUBLIC_CONTRACT_FIELDS_ADDED = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
