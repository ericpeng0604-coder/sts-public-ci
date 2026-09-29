from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "sts1" / "sts1_armg_ppo_v15_tournament.py"

SPEC = importlib.util.spec_from_file_location("sts1_armg_ppo_v15_tournament", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def row(name: str, *, passed: bool, wins: int, floor: float, safe: bool = True, idx: int = 0):
    return {
        "name": name,
        "rank_key": [
            int(safe),
            int(passed),
            wins,
            floor,
            0,
            -idx,
        ],
    }


def test_tournament_prefers_existing_dev_gate_pass():
    winner = mod.choose_winner(
        [
            row("A", passed=False, wins=5, floor=42.0, idx=0),
            row("B", passed=True, wins=4, floor=41.0, idx=1),
        ]
    )
    assert winner["name"] == "B"


def test_tournament_prefers_more_wins_then_floor():
    winner = mod.choose_winner(
        [
            row("A", passed=True, wins=4, floor=41.0, idx=0),
            row("B", passed=True, wins=5, floor=39.0, idx=1),
            row("C", passed=True, wins=5, floor=40.0, idx=2),
        ]
    )
    assert winner["name"] == "C"


def test_tournament_never_prefers_unsafe_candidate():
    winner = mod.choose_winner(
        [
            row("A", passed=True, wins=8, floor=50.0, safe=False, idx=0),
            row("B", passed=False, wins=3, floor=38.0, safe=True, idx=1),
        ]
    )
    assert winner["name"] == "B"
