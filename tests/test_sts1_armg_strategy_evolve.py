from __future__ import annotations

import pytest

from roguelike_ai.sts1_phase3.armg_strategy_evolve import (
    ArmGStrategyError,
    DEV_STRATEGY_GATE,
    FRESH_STRATEGY_GATE,
    StrategyGatePolicy,
    branch_quality,
    evaluate_strategy_gate,
    soft_branch_targets,
    strategy_example_priority,
    strategy_promotion_decision,
)


def _runs(
    count: int,
    *,
    victories: int,
    floor: int,
    combat_policy: str = "mcts_2000",
    unsafe: bool = False,
):
    rows = []
    for index in range(count):
        win = index < victories
        rows.append(
            {
                "seed": f"seed-{index:03d}",
                "result": "PASS_SIMULATOR_COMPLETE_RUN",
                "outcome": "victory" if win else "defeat",
                "final_floor": 60 if win else floor,
                "combat_policy": combat_policy,
                "noncombat_policy": "armg",
                "fallback_rate": 0.0,
                "student_action_count": 0,
                "hybrid_student_vote_count": 0,
                "hybrid_student_tiebreak_count": 0,
                "illegal_action_count": 1 if unsafe and index == 0 else 0,
                "timeout_count": 0,
                "crash_count": 0,
                "remote_error_count": 0,
            }
        )
    return {"runs": rows}


def test_branch_quality_is_long_horizon_lexicographic() -> None:
    assert branch_quality(
        outcome="victory", final_floor=40, final_hp=1, max_hp=80
    ) > branch_quality(
        outcome="defeat", final_floor=50, final_hp=80, max_hp=80
    )
    assert branch_quality(
        outcome="defeat", final_floor=40, final_hp=1, max_hp=80
    ) > branch_quality(
        outcome="defeat", final_floor=39, final_hp=80, max_hp=80
    )


def test_soft_targets_and_priority_emphasize_clear_mistakes() -> None:
    targets = soft_branch_targets([10.0, 20.0, 15.0], temperature=2.0)
    assert sum(targets) == pytest.approx(1.0)
    assert targets[1] > targets[2] > targets[0]
    assert strategy_example_priority(
        current_index=0,
        teacher_best_index=1,
        branch_values=[10.0, 30.0, 12.0],
    ) == 3.0
    assert strategy_example_priority(
        current_index=1,
        teacher_best_index=1,
        branch_values=[10.0, 11.0, 10.5],
    ) == 1.0


def test_dev_gate_accepts_same_wins_with_floor_improvement() -> None:
    current = _runs(30, victories=5, floor=20)
    candidate = _runs(30, victories=5, floor=21)
    result = evaluate_strategy_gate(
        current,
        candidate,
        policy=DEV_STRATEGY_GATE,
        expected_combat_policy="mcts_2000",
    )
    assert result["status"] == "PASS"
    assert result["win_delta"] == 0
    assert result["floor_delta"] == pytest.approx(25 / 30)


def test_strategy_gate_fails_closed_if_combat_is_not_pure_mcts() -> None:
    current = _runs(30, victories=5, floor=20)
    candidate = _runs(
        30,
        victories=8,
        floor=25,
        combat_policy="hybrid_mcts_1000_2000_student_tiebreak",
    )
    result = evaluate_strategy_gate(
        current,
        candidate,
        policy=DEV_STRATEGY_GATE,
        expected_combat_policy="mcts_2000",
    )
    assert result["status"] == "ROLLBACK"
    assert "combat_policy_not_frozen_pure_mcts" in result["reasons"]


def test_strategy_gate_fails_closed_on_safety_or_seed_mismatch() -> None:
    current = _runs(30, victories=5, floor=20)
    unsafe = _runs(30, victories=10, floor=30, unsafe=True)
    result = evaluate_strategy_gate(
        current,
        unsafe,
        policy=DEV_STRATEGY_GATE,
        expected_combat_policy="mcts_2000",
    )
    assert result["status"] == "ROLLBACK"
    assert "candidate_safety_failure" in result["reasons"]

    mismatched = _runs(30, victories=10, floor=30)
    mismatched["runs"][0]["seed"] = "different"
    with pytest.raises(ArmGStrategyError, match="identical seed sets"):
        evaluate_strategy_gate(
            current,
            mismatched,
            policy=DEV_STRATEGY_GATE,
            expected_combat_policy="mcts_2000",
        )


def test_fresh_gate_requires_win_gain_and_paired_superiority() -> None:
    current = _runs(100, victories=10, floor=20)
    candidate = _runs(100, victories=14, floor=22)
    result = evaluate_strategy_gate(
        current,
        candidate,
        policy=FRESH_STRATEGY_GATE,
        expected_combat_policy="mcts_2000",
    )
    assert result["status"] == "PASS"
    assert result["win_delta"] == 4
    assert result["paired"]["candidate_better"] > result["paired"]["candidate_worse"]
    assert result["paired"]["one_sided_sign_p"] <= 0.10

    weak = _runs(100, victories=11, floor=20)
    result = evaluate_strategy_gate(
        current,
        weak,
        policy=FRESH_STRATEGY_GATE,
        expected_combat_policy="mcts_2000",
    )
    assert result["status"] == "ROLLBACK"
    assert "paired_superiority_not_strong_enough" in result["reasons"]


def test_custom_gate_blocks_win_regression_even_with_floor_gain() -> None:
    current = _runs(4, victories=2, floor=10)
    candidate = _runs(4, victories=1, floor=40)
    result = evaluate_strategy_gate(
        current,
        candidate,
        policy=StrategyGatePolicy("tiny", 4, min_win_delta=0),
        expected_combat_policy="mcts_2000",
    )
    assert result["status"] == "ROLLBACK"
    assert any(reason.startswith("win_delta_") for reason in result["reasons"])


def test_promotion_requires_all_three_offline_gates() -> None:
    passed = {"status": "PASS"}
    decision = strategy_promotion_decision(passed, passed, passed)
    assert decision["decision"] == "PROMOTE_STRATEGY"
    assert decision["production_champion_replaced"] is False

    hold = strategy_promotion_decision(passed, {"status": "ROLLBACK"}, passed)
    assert hold["decision"] == "ROLLBACK_STRATEGY"
