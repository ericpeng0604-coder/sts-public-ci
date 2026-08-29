#!/usr/bin/env python3
"""Prove the starter-only midturn shortcut cannot accept richer card effects.

The public counter derivation is intentionally NOT a general midturn rule.
Cards that can draw, generate cards, gain energy, discard, retain, or otherwise
change hand accounting must remain fail-closed until separately reconstructed.
"""
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

from roguelike_ai.sts1_teacher.player_reconstruction import assess_public_player
from verify_phase1_public_jaw_worm_turn_one_native import admitted_and_seeds, public_state


def assert_rejected(card_id: str, *, name: str, card_type: str, cost: int) -> None:
    valid = public_state()
    _, seeds = admitted_and_seeds(valid)
    state = deepcopy(valid)
    state["hand"][0].update(
        {
            "id": card_id,
            "name": name,
            "type": card_type,
            "cost": cost,
            "upgrades": 0,
        }
    )

    admission = assess_public_player(state)
    assert admission.allowed is False
    assert f"turn1_nonstarter_card:{card_id}" in admission.reasons

    try:
        sts.build_public_jaw_worm_context_v1(state, seeds)
    except RuntimeError as exc:
        assert f"unsupported_card_v1:{card_id}" in str(exc), str(exc)
    else:
        raise AssertionError(f"native constructor accepted nonstarter card: {card_id}")


def main() -> None:
    # Both of these draw cards directly invalidate the hand-size shortcut called
    # out during review. ANGER covers generated-card growth as a second class of
    # hand/composition mutation that must also remain outside this narrow slice.
    assert_rejected("POMMEL_STRIKE", name="Pommel Strike", card_type="ATTACK", cost=1)
    assert_rejected("SHRUG_IT_OFF", name="Shrug It Off", card_type="SKILL", cost=1)
    assert_rejected("ANGER", name="Anger", card_type="ATTACK", cost=0)

    print("MIDTURN_NONSTARTER_FAIL_CLOSED = PASS")
    print("POMMEL_STRIKE_DRAW_CARD_REJECTED = PASS")
    print("SHRUG_IT_OFF_DRAW_CARD_REJECTED = PASS")
    print("ANGER_GENERATED_CARD_REJECTED = PASS")
    print("MIDTURN_HAND_SIZE_SHORTCUT_GENERALIZED = 0")
    print("PUBLIC_CONTRACT_FIELDS_ADDED = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
