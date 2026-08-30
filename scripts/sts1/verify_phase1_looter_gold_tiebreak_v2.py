#!/usr/bin/env python3
"""Static safety proof for the frozen Looter V2 gold tie-break."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roguelike_ai.sts1_teacher.native_rollout import NativePublicLooterRolloutV2
from roguelike_ai.sts1_teacher.search import SearchConfig


def main() -> None:
    config = SearchConfig(
        samples_per_semantic_action=8,
        rollout_budget=256,
        node_budget=16384,
        max_depth=30,
        timeout_ms=60_000,
        tie_tolerance=1e-9,
        sampling_seed=20260830,
    )
    min_nonzero_combat_mean_gap = 1.0 / config.samples_per_semantic_action
    assert NativePublicLooterRolloutV2.gold_tiebreak(0) == 0.0
    assert NativePublicLooterRolloutV2.gold_tiebreak(1) > config.tie_tolerance
    assert NativePublicLooterRolloutV2.gold_tiebreak(100_000) == NativePublicLooterRolloutV2.gold_tiebreak_cap
    assert NativePublicLooterRolloutV2.gold_tiebreak_cap < min_nonzero_combat_mean_gap
    assert NativePublicLooterRolloutV2.gold_tiebreak_cap == 0.001
    assert NativePublicLooterRolloutV2.gold_tiebreak_per_gold == 1e-7

    print("LOOTER_V2_GOLD_TIEBREAK_SAFETY = PASS")
    print("FROZEN_SEARCH_SAMPLES = 8")
    print("COMBAT_MEAN_MIN_NONZERO_GAP = 0.125")
    print("GOLD_TIEBREAK_CAP = 0.001")
    print("GOLD_TIEBREAK_PER_GOLD = 1e-7")
    print("GOLD_CAN_OVERTURN_NONZERO_COMBAT_MEAN_GAP = 0")
    print("PUBLIC_CONTRACT_FIELDS_ADDED = 0")
    print("PHASE1_GATE_CLAIMED = 0")


if __name__ == "__main__":
    main()
