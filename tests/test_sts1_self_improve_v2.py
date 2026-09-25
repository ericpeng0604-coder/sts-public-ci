from __future__ import annotations

from roguelike_ai.sts1_phase3.self_improve_v2 import (
    LearnerGatePolicy,
    LoopV2State,
    evaluate_learner_gate,
    ppo_allowed,
)


def _run(seed: int, *, floor: int, victory: bool = False) -> dict:
    return {
        "seed": seed,
        "result": "PASS_SIMULATOR_COMPLETE_RUN",
        "outcome": "victory" if victory else "defeat",
        "final_floor": floor,
        "illegal_action_count": 0,
        "timeout_count": 0,
        "crash_count": 0,
        "remote_error_count": 0,
    }


def test_learner_gate_accepts_same_wins_with_mean_floor_gain() -> None:
    current = [_run(1, floor=18), _run(2, floor=20)]
    candidate = [_run(1, floor=20), _run(2, floor=21)]
    gate = evaluate_learner_gate(
        current,
        candidate,
        policy=LearnerGatePolicy(min_floor_delta=0.5),
    )
    assert gate["status"] == "ACCEPT"
    assert gate["win_delta"] == 0
    assert gate["floor_delta"] == 1.5


def test_learner_gate_rolls_back_regression() -> None:
    current = [_run(1, floor=20), _run(2, floor=21)]
    candidate = [_run(1, floor=18), _run(2, floor=19)]
    gate = evaluate_learner_gate(current, candidate)
    assert gate["status"] == "ROLLBACK"


def test_learner_gate_accepts_more_wins_even_without_floor_gain() -> None:
    current = [_run(1, floor=20), _run(2, floor=20)]
    candidate = [_run(1, floor=10, victory=True), _run(2, floor=10)]
    gate = evaluate_learner_gate(current, candidate)
    assert gate["status"] == "ACCEPT"
    assert gate["win_delta"] == 1


def test_ppo_waits_until_student_has_learned_teacher() -> None:
    assert not ppo_allowed({"teacher_top1_accuracy": 0.36})["allowed"]
    assert ppo_allowed({"teacher_top1_accuracy": 0.60})["allowed"]


def test_loop_state_roundtrip(tmp_path) -> None:
    path = tmp_path / "state.json"
    state = LoopV2State(
        round_index=3,
        accepted_rounds=2,
        rejected_rounds=1,
        used_teacher_seeds=(11, 12),
        used_ppo_seeds=(21, 22),
    )
    state.write(path)
    assert LoopV2State.from_path(path) == state
