from __future__ import annotations

import pytest

from roguelike_ai.sts1_phase3.champion_gate import (
    ChampionGateError,
    FAST_GATE_POLICY,
    FORMAL_GATE_POLICY,
    evaluate_fixed_seed_gate,
    evaluate_real_game_gate,
    promotion_decision,
)


def _runs(count: int, *, victories: int, floor: int = 20, unsafe: bool = False):
    rows = []
    for index in range(count):
        rows.append(
            {
                "seed": f"seed-{index:03d}",
                "result": "PASS_SIMULATOR_COMPLETE_RUN",
                "outcome": "victory" if index < victories else "defeat",
                "final_floor": 60 if index < victories else floor,
                "illegal_action_count": 1 if unsafe and index == 0 else 0,
                "timeout_count": 0,
                "crash_count": 0,
                "remote_error_count": 0,
            }
        )
    return {"runs": rows}


def test_fast_gate_requires_plus_four_wins() -> None:
    champion = _runs(30, victories=6)
    candidate = _runs(30, victories=10)
    result = evaluate_fixed_seed_gate(champion, candidate, policy=FAST_GATE_POLICY)
    assert result["status"] == "PASS"
    assert result["win_delta"] == 4

    candidate = _runs(30, victories=9)
    result = evaluate_fixed_seed_gate(champion, candidate, policy=FAST_GATE_POLICY)
    assert result["status"] == "HOLD"
    assert result["win_delta"] == 3


def test_formal_gate_requires_plus_five_wins() -> None:
    champion = _runs(50, victories=8)
    candidate = _runs(50, victories=13)
    result = evaluate_fixed_seed_gate(champion, candidate, policy=FORMAL_GATE_POLICY)
    assert result["status"] == "PASS"
    assert result["win_delta"] == 5


def test_fixed_seed_gate_fails_closed_on_seed_mismatch_or_safety_error() -> None:
    champion = _runs(30, victories=5)
    candidate = _runs(30, victories=12, unsafe=True)
    result = evaluate_fixed_seed_gate(champion, candidate, policy=FAST_GATE_POLICY)
    assert result["status"] == "HOLD"
    assert "candidate_safety_failure" in result["reasons"]

    mismatched = _runs(30, victories=12)
    mismatched["runs"][0]["seed"] = "different-seed"
    with pytest.raises(ChampionGateError, match="exact same seed set"):
        evaluate_fixed_seed_gate(champion, mismatched, policy=FAST_GATE_POLICY)


def test_real_game_gate_requires_no_regression() -> None:
    champion = _runs(6, victories=2, floor=18)
    candidate = _runs(6, victories=2, floor=21)
    result = evaluate_real_game_gate(champion, candidate)
    assert result["status"] == "PASS"

    regressed = _runs(6, victories=1, floor=30)
    result = evaluate_real_game_gate(champion, regressed)
    assert result["status"] == "HOLD"
    assert "real_game_victories_regressed" in result["reasons"]


def test_promotion_requires_all_three_gates() -> None:
    fast = evaluate_fixed_seed_gate(
        _runs(30, victories=4),
        _runs(30, victories=8),
        policy=FAST_GATE_POLICY,
    )
    formal = evaluate_fixed_seed_gate(
        _runs(50, victories=7),
        _runs(50, victories=12),
        policy=FORMAL_GATE_POLICY,
    )
    real = evaluate_real_game_gate(
        _runs(5, victories=1),
        _runs(5, victories=1, floor=25),
    )
    assert promotion_decision(fast, formal, real)["decision"] == "PROMOTE"

    failed_formal = dict(formal)
    failed_formal["status"] = "HOLD"
    assert promotion_decision(fast, failed_formal, real)["decision"] == "HOLD"
