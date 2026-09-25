"""ArmG map branch-rollout data and promotion gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


ARMG_MAP_DATASET_SCHEMA_VERSION = "sts1-armg-map-branch-dataset-v1"
ARMG_MAP_GATE_SCHEMA_VERSION = "sts1-armg-map-gate-v1"
SAFETY_FIELDS = (
    "illegal_action_count",
    "timeout_count",
    "crash_count",
    "remote_error_count",
)


class ArmGMapError(RuntimeError):
    """ArmG map training/evaluation evidence is invalid."""


@dataclass(frozen=True)
class ArmGMapGatePolicy:
    name: str
    expected_seed_count: int
    min_floor_delta: float = 0.5
    max_floor_regression_with_win_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.expected_seed_count < 1:
            raise ValueError("expected_seed_count must be positive")
        if self.min_floor_delta < 0:
            raise ValueError("min_floor_delta must be non-negative")
        if self.max_floor_regression_with_win_gain < 0:
            raise ValueError("max_floor_regression_with_win_gain must be non-negative")


FAST_MAP_GATE = ArmGMapGatePolicy("armg-map-fixed-seed-30", 30)
FORMAL_MAP_GATE = ArmGMapGatePolicy("armg-map-fixed-seed-50", 50)


def branch_quality(
    *,
    outcome: str,
    final_floor: int,
    final_hp: int,
    max_hp: int | None = None,
) -> float:
    """Long-horizon branch score. Victory dominates floor, floor dominates HP."""
    if outcome not in {"victory", "defeat"}:
        raise ArmGMapError(f"invalid branch outcome: {outcome}")
    if final_floor < 0:
        raise ArmGMapError("final_floor must be non-negative")
    hp_cap = max(1, int(max_hp or max(final_hp, 1)))
    hp_fraction = max(0.0, min(1.0, float(final_hp) / hp_cap))
    return float(final_floor) + (60.0 if outcome == "victory" else 0.0) + 0.25 * hp_fraction


def soft_branch_targets(values: Sequence[float], *, temperature: float = 2.0) -> tuple[float, ...]:
    if not values:
        raise ArmGMapError("branch target values cannot be empty")
    if temperature <= 0:
        raise ArmGMapError("temperature must be positive")
    finite = [float(value) for value in values]
    if not all(math.isfinite(value) for value in finite):
        raise ArmGMapError("branch target values must be finite")
    peak = max(finite)
    weights = [math.exp((value - peak) / temperature) for value in finite]
    total = sum(weights)
    return tuple(value / total for value in weights)


def _index_runs(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    expected: int,
) -> dict[str, Mapping[str, Any]]:
    rows = value.get("runs") if isinstance(value, Mapping) else value
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ArmGMapError("evaluation evidence must contain runs")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArmGMapError("evaluation run must be an object")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise ArmGMapError("evaluation run is missing seed")
        key = str(seed)
        if key in indexed:
            raise ArmGMapError(f"duplicate eval seed: {key}")
        indexed[key] = row
    if len(indexed) != expected:
        raise ArmGMapError(f"expected {expected} eval seeds, got {len(indexed)}")
    return indexed


def _summary(indexed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(indexed.values())
    complete = [
        row for row in rows
        if isinstance(row.get("result"), str)
        and str(row["result"]).startswith("PASS_")
        and row.get("outcome") in {"victory", "defeat"}
    ]
    floors = [
        float(row["final_floor"]) for row in complete
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
                raise ArmGMapError(f"invalid {field}: {raw!r}")
            total += raw
        safety[field] = total
    return {
        "seeds": len(rows),
        "complete_runs": len(complete),
        "victories": sum(int(row.get("outcome") == "victory") for row in complete),
        "mean_final_floor": sum(floors) / len(floors) if floors else None,
        "safety": safety,
    }


def evaluate_armg_map_gate(
    current: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    policy: ArmGMapGatePolicy,
) -> dict[str, Any]:
    left = _index_runs(current, expected=policy.expected_seed_count)
    right = _index_runs(candidate, expected=policy.expected_seed_count)
    if set(left) != set(right):
        raise ArmGMapError("current and candidate ArmG must use identical seed sets")

    current_summary = _summary(left)
    candidate_summary = _summary(right)
    complete = (
        current_summary["complete_runs"] == policy.expected_seed_count
        and candidate_summary["complete_runs"] == policy.expected_seed_count
    )
    safe = all(candidate_summary["safety"][field] == 0 for field in SAFETY_FIELDS)
    win_delta = candidate_summary["victories"] - current_summary["victories"]
    current_floor = current_summary["mean_final_floor"]
    candidate_floor = candidate_summary["mean_final_floor"]
    floor_delta = (
        None if current_floor is None or candidate_floor is None
        else candidate_floor - current_floor
    )

    wins_not_worse = win_delta >= 0
    floor_ok = floor_delta is not None
    improved = False
    if floor_ok:
        if win_delta > 0:
            improved = floor_delta >= -policy.max_floor_regression_with_win_gain
        elif win_delta == 0:
            improved = floor_delta >= policy.min_floor_delta

    passed = complete and safe and wins_not_worse and improved
    reasons: list[str] = []
    if not complete:
        reasons.append("incomplete_eval")
    if not safe:
        reasons.append("candidate_safety_failure")
    if not wins_not_worse:
        reasons.append("victories_regressed")
    if not improved:
        reasons.append("no_map_component_improvement")

    return {
        "schema_version": ARMG_MAP_GATE_SCHEMA_VERSION,
        "gate": policy.name,
        "status": "PASS" if passed else "ROLLBACK",
        "win_delta": win_delta,
        "floor_delta": floor_delta,
        "current": current_summary,
        "candidate": candidate_summary,
        "reasons": reasons,
    }


def armg_map_promotion_decision(
    fast_gate: Mapping[str, Any],
    formal_gate: Mapping[str, Any],
) -> dict[str, Any]:
    passed = fast_gate.get("status") == "PASS" and formal_gate.get("status") == "PASS"
    return {
        "schema_version": ARMG_MAP_GATE_SCHEMA_VERSION,
        "decision": "PROMOTE_MAP" if passed else "ROLLBACK_MAP",
        "all_gates_passed": passed,
        "fast_gate": dict(fast_gate),
        "formal_gate": dict(formal_gate),
    }


__all__ = [
    "ARMG_MAP_DATASET_SCHEMA_VERSION",
    "ARMG_MAP_GATE_SCHEMA_VERSION",
    "ArmGMapError",
    "ArmGMapGatePolicy",
    "FAST_MAP_GATE",
    "FORMAL_MAP_GATE",
    "armg_map_promotion_decision",
    "branch_quality",
    "evaluate_armg_map_gate",
    "soft_branch_targets",
]
