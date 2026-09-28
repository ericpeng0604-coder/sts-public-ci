from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sts1"
    / "sts1_armg_strategy_loop.py"
)
SPEC = importlib.util.spec_from_file_location("strategy_loop_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def _fresh_gate(
    *,
    win_delta: int = 4,
    floor_delta: float = 0.55,
    better: int = 30,
    worse: int = 22,
    sign_p: float = 0.166,
    reasons: list[str] | None = None,
) -> dict:
    return {
        "status": "ROLLBACK",
        "expected_seed_count": 100,
        "win_delta": win_delta,
        "floor_delta": floor_delta,
        "current": {"complete_runs": 100},
        "candidate": {"complete_runs": 100},
        "paired": {
            "candidate_better": better,
            "candidate_worse": worse,
            "ties": 100 - better - worse,
            "one_sided_sign_p": sign_p,
        },
        "reasons": reasons or ["paired_superiority_not_strong_enough"],
    }


def test_near_miss_fresh_candidate_becomes_elite() -> None:
    decision = m._elite_candidate_decision(
        _fresh_gate(),
        promoted=False,
        max_sign_p=0.25,
        min_win_delta=1,
        min_floor_delta=-0.5,
    )
    assert decision["eligible"] is True
    assert decision["win_delta"] == 4
    assert decision["candidate_better"] > decision["candidate_worse"]


def test_elite_never_bypasses_promotion() -> None:
    decision = m._elite_candidate_decision(
        _fresh_gate(),
        promoted=True,
        max_sign_p=0.25,
        min_win_delta=1,
        min_floor_delta=-0.5,
    )
    assert decision == {"eligible": False, "reason": "already_promoted"}


def test_unsafe_or_incomplete_candidate_never_becomes_elite() -> None:
    decision = m._elite_candidate_decision(
        _fresh_gate(reasons=["candidate_safety_failure"]),
        promoted=False,
        max_sign_p=0.25,
        min_win_delta=1,
        min_floor_delta=-0.5,
    )
    assert decision["eligible"] is False
    assert decision["reason"] == "unsafe_or_incomplete"


def test_candidate_without_real_fresh_advantage_is_not_elite() -> None:
    decision = m._elite_candidate_decision(
        _fresh_gate(win_delta=0, better=25, worse=25, sign_p=0.5),
        promoted=False,
        max_sign_p=0.25,
        min_win_delta=1,
        min_floor_delta=-0.5,
    )
    assert decision["eligible"] is False


def _teacher_example(
    *,
    current_index: int = 0,
    teacher_index: int = 1,
    values: list[float] | None = None,
) -> dict:
    return {
        "schema_version": "sts1-armg-strategy-branch-dataset-v1",
        "current_armg_index": current_index,
        "teacher_best_index": teacher_index,
        "branch_quality": values or [40.0, 55.0],
        "priority": 2.0,
        "kind": "map",
        "obs": [0.0],
        "descs": [[0.0], [1.0]],
        "target_probs": [0.1, 0.9],
        "combat_policy": "mcts_2000",
    }


def test_teacher_verified_elite_disagreement_gets_high_priority() -> None:
    row = m._elite_verified_example(
        _teacher_example(),
        elite_index=1,
        elite_sha256="a" * 64,
        min_quality_margin=1.0,
    )
    assert row is not None
    assert row["source"] == "elite_candidate_verified_disagreement"
    assert row["priority"] >= 3.5
    assert row["elite_quality_margin_over_champion"] == 15.0


def test_elite_choice_is_rejected_if_teacher_disagrees() -> None:
    row = m._elite_verified_example(
        _teacher_example(teacher_index=0),
        elite_index=1,
        elite_sha256="b" * 64,
        min_quality_margin=1.0,
    )
    assert row is None


def test_elite_choice_is_rejected_if_margin_is_too_small() -> None:
    row = m._elite_verified_example(
        _teacher_example(values=[40.0, 40.5]),
        elite_index=1,
        elite_sha256="c" * 64,
        min_quality_margin=1.0,
    )
    assert row is None
