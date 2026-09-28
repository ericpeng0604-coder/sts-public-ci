"""Pure-MCTS combat / ArmG strategy self-improvement contracts.

Only non-combat strategy weights are allowed to change. Combat must remain on
one exact pure-MCTS policy for both current and candidate evaluations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


STRATEGY_DATASET_SCHEMA_VERSION = "sts1-armg-strategy-branch-dataset-v1"
STRATEGY_GATE_SCHEMA_VERSION = "sts1-armg-strategy-gate-v1"
SAFETY_FIELDS = (
    "illegal_action_count",
    "timeout_count",
    "crash_count",
    "remote_error_count",
)


class ArmGStrategyError(RuntimeError):
    """Strategy training/evaluation evidence is incomplete or unsafe."""


@dataclass(frozen=True)
class StrategyGatePolicy:
    name: str
    expected_seed_count: int
    min_win_delta: int = 0
    min_floor_delta: float = 0.5
    max_floor_regression_with_win_gain: float = 0.5
    max_one_sided_sign_p: float | None = None

    def __post_init__(self) -> None:
        if self.expected_seed_count < 1:
            raise ValueError("expected_seed_count must be positive")
        if self.min_win_delta < 0:
            raise ValueError("min_win_delta must be non-negative")
        if self.min_floor_delta < 0:
            raise ValueError("min_floor_delta must be non-negative")
        if self.max_floor_regression_with_win_gain < 0:
            raise ValueError("max_floor_regression_with_win_gain must be non-negative")
        if self.max_one_sided_sign_p is not None and not 0 < self.max_one_sided_sign_p <= 1:
            raise ValueError("max_one_sided_sign_p must be in (0, 1]")


DEV_STRATEGY_GATE = StrategyGatePolicy(
    "strategy-dev-30",
    30,
    min_win_delta=0,
    min_floor_delta=0.5,
)
HIDDEN_STRATEGY_GATE = StrategyGatePolicy(
    "strategy-hidden-50",
    50,
    min_win_delta=0,
    min_floor_delta=0.5,
)
FRESH_STRATEGY_GATE = StrategyGatePolicy(
    "strategy-fresh-100",
    100,
    min_win_delta=1,
    min_floor_delta=0.0,
    max_floor_regression_with_win_gain=0.5,
    max_one_sided_sign_p=0.10,
)


def branch_quality(
    *,
    outcome: str,
    final_floor: int,
    final_hp: int,
    max_hp: int | None = None,
) -> float:
    """Lexicographic long-horizon score: victory > floor > remaining HP."""
    if outcome not in {"victory", "defeat"}:
        raise ArmGStrategyError(f"invalid branch outcome: {outcome}")
    if final_floor < 0:
        raise ArmGStrategyError("final_floor must be non-negative")
    hp_cap = max(1, int(max_hp or max(final_hp, 1)))
    hp_fraction = max(0.0, min(1.0, float(final_hp) / hp_cap))
    return float(final_floor) + (60.0 if outcome == "victory" else 0.0) + 0.25 * hp_fraction


def soft_branch_targets(
    values: Sequence[float],
    *,
    temperature: float = 2.0,
) -> tuple[float, ...]:
    """Turn exact branch rollout returns into a soft policy target."""
    if not values:
        raise ArmGStrategyError("branch target values cannot be empty")
    if temperature <= 0:
        raise ArmGStrategyError("temperature must be positive")
    finite = [float(value) for value in values]
    if not all(math.isfinite(value) for value in finite):
        raise ArmGStrategyError("branch target values must be finite")
    peak = max(finite)
    weights = [math.exp((value - peak) / temperature) for value in finite]
    total = sum(weights)
    return tuple(value / total for value in weights)


def strategy_example_priority(
    *,
    current_index: int,
    teacher_best_index: int,
    branch_values: Sequence[float],
) -> float:
    """Prioritize clear mistakes without dropping agreement examples."""
    if not branch_values:
        raise ArmGStrategyError("strategy example requires branch values")
    if not 0 <= current_index < len(branch_values):
        raise ArmGStrategyError("current_index outside branch values")
    if not 0 <= teacher_best_index < len(branch_values):
        raise ArmGStrategyError("teacher_best_index outside branch values")
    ordered = sorted((float(v) for v in branch_values), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
    priority = 1.0
    if current_index != teacher_best_index:
        priority += 1.0
    if margin >= 5.0:
        priority += 0.5
    if margin >= 15.0:
        priority += 0.5
    return min(priority, 3.0)


def _as_runs(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows = value.get("runs") if isinstance(value, Mapping) else value
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ArmGStrategyError("evaluation evidence must contain runs")
    result = list(rows)
    if not result or not all(isinstance(row, Mapping) for row in result):
        raise ArmGStrategyError("evaluation runs must be non-empty objects")
    return result


def _index_runs(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    expected: int,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in _as_runs(value):
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise ArmGStrategyError("evaluation run is missing seed")
        key = str(seed)
        if key in indexed:
            raise ArmGStrategyError(f"duplicate evaluation seed: {key}")
        indexed[key] = row
    if len(indexed) != expected:
        raise ArmGStrategyError(f"expected {expected} eval seeds, got {len(indexed)}")
    return indexed


def _is_complete(row: Mapping[str, Any]) -> bool:
    return (
        isinstance(row.get("result"), str)
        and str(row["result"]).startswith("PASS_")
        and row.get("outcome") in {"victory", "defeat"}
    )


def _combat_locked(row: Mapping[str, Any], expected_combat_policy: str) -> bool:
    return (
        row.get("combat_policy") == expected_combat_policy
        and int(row.get("student_action_count", 0) or 0) == 0
        and int(row.get("hybrid_student_vote_count", 0) or 0) == 0
        and int(row.get("hybrid_student_tiebreak_count", 0) or 0) == 0
    )


def _summary(
    indexed: Mapping[str, Mapping[str, Any]],
    *,
    expected_combat_policy: str,
) -> dict[str, Any]:
    rows = list(indexed.values())
    complete = [row for row in rows if _is_complete(row)]
    floors = [
        float(row["final_floor"])
        for row in complete
        if isinstance(row.get("final_floor"), (int, float))
        and not isinstance(row.get("final_floor"), bool)
    ]
    safety: dict[str, int] = {}
    for field in SAFETY_FIELDS:
        total = 0
        for row in rows:
            raw = row.get(field, 0)
            if raw is None:
                raw = 0
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ArmGStrategyError(f"invalid {field}: {raw!r}")
            total += raw
        safety[field] = total
    return {
        "seeds": len(rows),
        "complete_runs": len(complete),
        "victories": sum(int(row.get("outcome") == "victory") for row in complete),
        "mean_final_floor": sum(floors) / len(floors) if floors else None,
        "safety": safety,
        "pure_mcts_runs": sum(
            int(_combat_locked(row, expected_combat_policy)) for row in rows
        ),
    }


def _one_sided_sign_p(better: int, worse: int) -> float:
    """P[X >= better] for X~Binomial(better+worse, 0.5)."""
    n = better + worse
    if n == 0:
        return 1.0
    numerator = sum(math.comb(n, k) for k in range(better, n + 1))
    return numerator / float(2**n)


def _paired_superiority(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    better = worse = ties = 0
    for seed in sorted(left):
        a = left[seed]
        b = right[seed]
        if not (_is_complete(a) and _is_complete(b)):
            continue
        a_win = a.get("outcome") == "victory"
        b_win = b.get("outcome") == "victory"
        if a_win != b_win:
            if b_win:
                better += 1
            else:
                worse += 1
            continue
        a_floor = float(a.get("final_floor", 0) or 0)
        b_floor = float(b.get("final_floor", 0) or 0)
        if b_floor > a_floor:
            better += 1
        elif b_floor < a_floor:
            worse += 1
        else:
            ties += 1
    return {
        "candidate_better": better,
        "candidate_worse": worse,
        "ties": ties,
        "one_sided_sign_p": _one_sided_sign_p(better, worse),
    }


def evaluate_strategy_gate(
    current: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    policy: StrategyGatePolicy,
    expected_combat_policy: str,
) -> dict[str, Any]:
    """Fail-closed paired gate with a frozen pure-MCTS combat invariant."""
    left = _index_runs(current, expected=policy.expected_seed_count)
    right = _index_runs(candidate, expected=policy.expected_seed_count)
    if set(left) != set(right):
        raise ArmGStrategyError("current and candidate must use identical seed sets")

    current_summary = _summary(left, expected_combat_policy=expected_combat_policy)
    candidate_summary = _summary(right, expected_combat_policy=expected_combat_policy)
    complete = (
        current_summary["complete_runs"] == policy.expected_seed_count
        and candidate_summary["complete_runs"] == policy.expected_seed_count
    )
    safe = all(candidate_summary["safety"][field] == 0 for field in SAFETY_FIELDS)
    combat_locked = (
        current_summary["pure_mcts_runs"] == policy.expected_seed_count
        and candidate_summary["pure_mcts_runs"] == policy.expected_seed_count
    )

    win_delta = candidate_summary["victories"] - current_summary["victories"]
    current_floor = current_summary["mean_final_floor"]
    candidate_floor = candidate_summary["mean_final_floor"]
    floor_delta = (
        None
        if current_floor is None or candidate_floor is None
        else candidate_floor - current_floor
    )

    win_ok = win_delta >= policy.min_win_delta
    if floor_delta is None:
        floor_ok = False
    elif win_delta > policy.min_win_delta:
        floor_ok = floor_delta >= -policy.max_floor_regression_with_win_gain
    elif policy.min_win_delta > 0:
        floor_ok = floor_delta >= -policy.max_floor_regression_with_win_gain
    else:
        floor_ok = floor_delta >= policy.min_floor_delta

    paired = _paired_superiority(left, right)
    sign_ok = (
        policy.max_one_sided_sign_p is None
        or (
            paired["candidate_better"] > paired["candidate_worse"]
            and paired["one_sided_sign_p"] <= policy.max_one_sided_sign_p
        )
    )

    passed = complete and safe and combat_locked and win_ok and floor_ok and sign_ok
    reasons: list[str] = []
    if not complete:
        reasons.append("incomplete_eval")
    if not safe:
        reasons.append("candidate_safety_failure")
    if not combat_locked:
        reasons.append("combat_policy_not_frozen_pure_mcts")
    if not win_ok:
        reasons.append(f"win_delta_{win_delta}_below_required_{policy.min_win_delta}")
    if not floor_ok:
        reasons.append("floor_gate_failed")
    if not sign_ok:
        reasons.append("paired_superiority_not_strong_enough")

    return {
        "schema_version": STRATEGY_GATE_SCHEMA_VERSION,
        "gate": policy.name,
        "status": "PASS" if passed else "ROLLBACK",
        "expected_seed_count": policy.expected_seed_count,
        "expected_combat_policy": expected_combat_policy,
        "required_win_delta": policy.min_win_delta,
        "win_delta": win_delta,
        "floor_delta": floor_delta,
        "current": current_summary,
        "candidate": candidate_summary,
        "paired": paired,
        "reasons": reasons,
    }


def strategy_promotion_decision(
    dev_gate: Mapping[str, Any],
    hidden_gate: Mapping[str, Any],
    fresh_gate: Mapping[str, Any],
) -> dict[str, Any]:
    gates = {
        "dev_30": dict(dev_gate),
        "hidden_50": dict(hidden_gate),
        "fresh_100": dict(fresh_gate),
    }
    passed = all(gate.get("status") == "PASS" for gate in gates.values())
    return {
        "schema_version": STRATEGY_GATE_SCHEMA_VERSION,
        "decision": "PROMOTE_STRATEGY" if passed else "ROLLBACK_STRATEGY",
        "all_gates_passed": passed,
        "production_champion_replaced": False,
        "gates": gates,
    }


__all__ = [
    "ArmGStrategyError",
    "DEV_STRATEGY_GATE",
    "FRESH_STRATEGY_GATE",
    "HIDDEN_STRATEGY_GATE",
    "SAFETY_FIELDS",
    "STRATEGY_DATASET_SCHEMA_VERSION",
    "STRATEGY_GATE_SCHEMA_VERSION",
    "StrategyGatePolicy",
    "branch_quality",
    "evaluate_strategy_gate",
    "soft_branch_targets",
    "strategy_example_priority",
    "strategy_promotion_decision",
]
