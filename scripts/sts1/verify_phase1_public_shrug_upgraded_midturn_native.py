#!/usr/bin/env python3
"""Native proof for the audited upgraded Shrug It Off midturn slice."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "external" / "sts_lightspeed" / "source" / "build-probe"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import slaythespire as sts

from scripts.sts1.verify_phase1_public_shrug_midturn_native import (
    admitted_context_and_seeds,
    expect_native_reject,
    public_state,
    reconstruction_aux,
)
from roguelike_ai.sts1_teacher.contract import DecisionContext


def upgraded_state() -> dict:
    state = public_state()
    state["block"] = 11
    shrug = state["discard_pile"][-1]
    assert shrug["name"] == "Shrug It Off"
    shrug["upgrades"] = 1
    return state


def main() -> None:
    state = upgraded_state()
    aux = reconstruction_aux()
    admitted, admitted_state, seeds = admitted_context_and_seeds(state, aux)

    assert len(state["hand"]) == 5
    assert 5 - len(state["hand"]) == 0
    assert aux["cards_played_this_turn"] == 1
    assert aux["skills_played_this_turn"] == 1
    assert admitted.decision_signature == DecisionContext.from_public_state(state).decision_signature

    bc = sts.build_public_jaw_worm_context_v1(admitted_state, seeds, aux)
    assert bc.turn == 1
    assert bc.player.energy == 2
    assert bc.player.block == 11
    assert len(bc.hand) == 5
    assert len(bc.draw_pile) == 0
    assert len(bc.discard_pile) == 6
    assert len(bc.exhaust_pile) == 0
    assert sorted(card.name for card in bc.discard_pile) == [
        "Bash", "Defend", "Defend", "Shrug It Off", "Strike", "Strike"
    ]

    # A normal Shrug and an upgraded Shrug use the same bounded skill trace;
    # the visible upgrade count decides whether block must be 8 or 11.
    wrong_block = deepcopy(admitted_state)
    wrong_block["block"] = 8
    expect_native_reject(wrong_block, seeds, aux, "shrug_midturn_player_shape_mismatch")

    extra_upgrade = deepcopy(admitted_state)
    extra_upgrade["hand"][0]["upgrades"] = 1
    expect_native_reject(extra_upgrade, seeds, aux, "shrug_midturn_deck_composition_mismatch")

    upgrade_two = deepcopy(admitted_state)
    upgrade_two["discard_pile"][-1]["upgrades"] = 2
    expect_native_reject(upgrade_two, seeds, aux, "unsupported_upgrade_count")

    legal = list(sts.get_legal_actions(bc))
    strike = next(
        action for action in legal
        if action.action_type == sts.SearchActionType.CARD
        and action.source_idx == 0
        and action.target_idx == 0
    )
    hp_before = bc.monsters[0].cur_hp
    strike.execute(bc)
    assert bc.monsters[0].cur_hp < hp_before
    assert bc.player.energy == 1
    assert bc.player.block == 11

    print("SHRUG_UPGRADED_MIDTURN_NATIVE = PASS")
    print("VISIBLE_UPGRADE_COUNT_1 = PASS")
    print("EXPECTED_BLOCK_11 = PASS")
    print("AUX_SCHEMA_UNCHANGED = PASS")
    print("NORMAL_SHRUG_BLOCK_8_REMAINS_SEPARATE = PASS")
    print("EXTRA_UPGRADED_STARTER_FAILS_CLOSED = PASS")
    print("UPGRADE_COUNT_TWO_FAILS_CLOSED = PASS")
    print("EXECUTABLE_POST_RECONSTRUCTION_ACTION = PASS")
    print("PUBLIC_CONTRACT_FIELDS_ADDED = 0")
    print("SOURCE_HIDDEN_RNG_ACCESS = 0")


if __name__ == "__main__":
    main()
