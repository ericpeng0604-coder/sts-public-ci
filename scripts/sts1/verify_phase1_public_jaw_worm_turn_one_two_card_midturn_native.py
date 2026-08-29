#!/usr/bin/env python3
"""Native proof for Jaw Worm turn-1 after exactly two public starter-card plays."""
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
from verify_phase1_public_jaw_worm_turn_one_midturn_native import enum_name, first_defend_action, first_strike_action


def project(bc, state: dict) -> dict:
    return SimulatorCombatAdapter().adapt(
        bc,
        legal_actions=sts.get_legal_actions(bc),
        run_state=run_state(state),
    )


def first_legal_one_cost_card(bc):
    hand = list(bc.hand)
    for action in sts.get_legal_actions(bc):
        if enum_name(action.action_type) != "CARD":
            continue
        idx = int(action.source_idx)
        card = hand[idx]
        if int(card.cost_for_turn) == 1:
            return action, str(card.name).upper()
    raise AssertionError("missing legal one-cost third action")


def main() -> None:
    start = public_state()
    _, start_seeds = admitted_and_seeds(start)

    # Generate a genuinely reachable two-card midturn public state using only
    # the audited starter-only slice. No draw/gain-energy/discard/generated-card
    # effect is admitted by this proof.
    source_bc = sts.build_public_jaw_worm_context_v1(start, start_seeds)
    first_defend_action(source_bc).execute(source_bc)
    first_strike_action(source_bc).execute(source_bc)
    mid2 = project(source_bc, start)

    assert int(mid2["turn"]) == 1
    assert int(mid2["energy"]) == 1
    assert int(mid2["block"]) == 5
    assert len(mid2["hand"]) == 3
    assert len(mid2["draw_pile"]) == 0
    assert len(mid2["discard_pile"]) == 7
    assert len(mid2["exhaust_pile"]) == 0
    admission = assess_public_player(mid2)
    assert admission.allowed, admission.reasons

    admitted, seeds_a = admitted_and_seeds(mid2)
    seeds_b = deepcopy(seeds_a)
    seeds_b["previous_history"] = int(seeds_b["previous_history"]) ^ 0xC3C3C3C3C3C3C3C3

    rebuilt_a = sts.build_public_jaw_worm_context_v1(mid2, seeds_a)
    rebuilt_b = sts.build_public_jaw_worm_context_v1(mid2, seeds_b)

    for rebuilt in (rebuilt_a, rebuilt_b):
        assert int(rebuilt.turn) == 1
        assert int(rebuilt.player.energy) == 1
        assert int(rebuilt.player.block) == 5
        assert len(rebuilt.hand) == 3
        assert len(rebuilt.draw_pile) == 0
        assert len(rebuilt.discard_pile) == 7
        assert len(rebuilt.exhaust_pile) == 0
        roundtrip = project(rebuilt, mid2)
        assert DecisionContext.from_public_state(roundtrip).decision_signature == admitted.decision_signature

    action_a, card_a = first_legal_one_cost_card(rebuilt_a)
    action_b, card_b = first_legal_one_cost_card(rebuilt_b)
    assert card_a == card_b
    action_a.execute(rebuilt_a)
    action_b.execute(rebuilt_b)

    after_a = project(rebuilt_a, mid2)
    after_b = project(rebuilt_b, mid2)
    after_context_a = DecisionContext.from_public_state(after_a)
    after_context_b = DecisionContext.from_public_state(after_b)
    assert after_context_a.decision_signature == after_context_b.decision_signature
    assert int(after_a["energy"]) == 0
    assert len(after_a["hand"]) == 2
    assert len(after_a["discard_pile"]) == 8
    assert assess_public_player(after_a).allowed

    print("JAW_WORM_TURN_ONE_TWO_CARD_MIDTURN_PUBLIC_RECONSTRUCTION = PASS")
    print("TWO_CARD_MIDTURN_SOURCE_PUBLIC_PROJECTION = PASS")
    print("TWO_CARD_MIDTURN_ADMISSION = PASS")
    print("TWO_CARD_MIDTURN_ROUNDTRIP_SIGNATURE = PASS")
    print("TWO_CARD_MIDTURN_PUBLIC_COUNTER_DERIVATION = PASS")
    print("TWO_CARD_MIDTURN_PREVIOUS_HISTORY_SEED_INDEPENDENT = PASS")
    print("TWO_CARD_MIDTURN_THIRD_ACTION_EXECUTABLE = PASS")
    print(f"TWO_CARD_MIDTURN_THIRD_ACTION_CARD = {card_a}")
    print("TWO_CARD_MIDTURN_ENERGY_AFTER_THIRD_ACTION = 0")
    print("SOURCE_COUNTER_HISTORY_INPUT = 0")
    print("MIDTURN_HAND_SIZE_SHORTCUT_GENERALIZED = 0")
    print("PUBLIC_CONTRACT_FIELDS_ADDED = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
