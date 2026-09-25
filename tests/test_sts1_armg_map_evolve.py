from __future__ import annotations

import pytest

from roguelike_ai.sts1_phase3.armg_map_evolve import (
    ArmGMapGatePolicy,
    ArmGMapError,
    branch_quality,
    evaluate_armg_map_gate,
    soft_branch_targets,
)


def _run(seed: int, floor: int, *, win: bool = False) -> dict:
    return {
        "seed": seed,
        "result": "PASS_SIMULATOR_COMPLETE_RUN",
        "outcome": "victory" if win else "defeat",
        "final_floor": floor,
        "illegal_action_count": 0,
        "timeout_count": 0,
        "crash_count": 0,
        "remote_error_count": 0,
    }


def test_branch_quality_prefers_victory_then_floor() -> None:
    assert branch_quality(outcome="victory", final_floor=40, final_hp=1, max_hp=80) > branch_quality(
        outcome="defeat", final_floor=50, final_hp=80, max_hp=80
    )
    assert branch_quality(outcome="defeat", final_floor=40, final_hp=1, max_hp=80) > branch_quality(
        outcome="defeat", final_floor=39, final_hp=80, max_hp=80
    )


def test_soft_branch_targets_rank_best_choice() -> None:
    targets = soft_branch_targets([10.0, 20.0, 15.0], temperature=2.0)
    assert sum(targets) == pytest.approx(1.0)
    assert targets[1] > targets[2] > targets[0]


def test_map_gate_accepts_floor_gain_without_win_regression() -> None:
    current = [_run(1, 20), _run(2, 22)]
    candidate = [_run(1, 21), _run(2, 23)]
    gate = evaluate_armg_map_gate(
        current,
        candidate,
        policy=ArmGMapGatePolicy("test", 2, min_floor_delta=0.5),
    )
    assert gate["status"] == "PASS"
    assert gate["floor_delta"] == 1.0


def test_map_gate_rolls_back_win_regression() -> None:
    current = [_run(1, 20, win=True), _run(2, 22)]
    candidate = [_run(1, 30), _run(2, 30)]
    gate = evaluate_armg_map_gate(
        current,
        candidate,
        policy=ArmGMapGatePolicy("test", 2),
    )
    assert gate["status"] == "ROLLBACK"
    assert "victories_regressed" in gate["reasons"]


def test_map_gate_rejects_mismatched_seed_sets() -> None:
    with pytest.raises(ArmGMapError):
        evaluate_armg_map_gate(
            [_run(1, 20), _run(2, 20)],
            [_run(1, 21), _run(3, 21)],
            policy=ArmGMapGatePolicy("test", 2),
        )
