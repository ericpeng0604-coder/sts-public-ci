"""Promotion gates for the STS1 self-improvement loop.

The current Champion is immutable until a Candidate passes:
1. a paired 30-seed fast gate (+4 wins),
2. a paired 50-seed formal gate (+5 wins), and
3. paired real-game validation with zero safety failures and no regression.

This module deliberately does not train a model or overwrite a Champion.
It only turns evidence into a fail-closed promotion decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


GATE_SCHEMA_VERSION = "sts1-champion-gate-v1"
FAST_SEED_COUNT = 30
FAST_MIN_WIN_DELTA = 4
FORMAL_SEED_COUNT = 50
FORMAL_MIN_WIN_DELTA = 5
SAFETY_FIELDS = (
    "illegal_action_count",
    "timeout_count",
    "crash_count",
    "remote_error_count",
)


class ChampionGateError(RuntimeError):
    """Promotion evidence is incomplete, mismatched, or unsafe."""


@dataclass(frozen=True)
class GatePolicy:
    name: str
    expected_seed_count: int
    min_win_delta: int

    def __post_init__(self) -> None:
        if self.expected_seed_count < 1:
            raise ValueError("expected_seed_count must be positive")
        if self.min_win_delta < 0:
            raise ValueError("min_win_delta must be non-negative")


FAST_GATE_POLICY = GatePolicy("fixed-seed-30", FAST_SEED_COUNT, FAST_MIN_WIN_DELTA)
FORMAL_GATE_POLICY = GatePolicy("fixed-seed-50", FORMAL_SEED_COUNT, FORMAL_MIN_WIN_DELTA)


def _as_runs(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        runs = value.get("runs")
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes, bytearray)):
            raise ChampionGateError("aggregate must contain a runs sequence")
        rows = list(runs)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = list(value)
    else:
        raise ChampionGateError("evidence must be an aggregate mapping or a run sequence")
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ChampionGateError("runs must contain at least one mapping")
    return rows


def _index_runs(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in _as_runs(value):
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (str, int)):
            raise ChampionGateError("every run must contain a string/int seed")
        key = str(seed)
        if key in indexed:
            raise ChampionGateError(f"duplicate seed in evidence: {key}")
        indexed[key] = row
    if expected_count is not None and len(indexed) != expected_count:
        raise ChampionGateError(
            f"expected exactly {expected_count} seeds, got {len(indexed)}"
        )
    return indexed


def _is_complete(row: Mapping[str, Any]) -> bool:
    result = row.get("result")
    return (
        isinstance(result, str)
        and result.startswith("PASS_")
        and row.get("outcome") in {"victory", "defeat"}
    )


def _safety_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for field in SAFETY_FIELDS:
        total = 0
        for row in rows:
            value = row.get(field, 0)
            if value is None:
                value = 0
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ChampionGateError(f"invalid {field}: {value!r}")
            total += value
        totals[field] = total
    return totals


def _summary(indexed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(indexed.values())
    complete = [row for row in rows if _is_complete(row)]
    victories = [row for row in complete if row.get("outcome") == "victory"]
    floors = [
        float(row["final_floor"])
        for row in complete
        if isinstance(row.get("final_floor"), (int, float))
        and not isinstance(row.get("final_floor"), bool)
    ]
    return {
        "seeds": len(rows),
        "complete_runs": len(complete),
        "victories": len(victories),
        "defeats": len(complete) - len(victories),
        "mean_final_floor": (sum(floors) / len(floors)) if floors else None,
        "safety": _safety_totals(rows),
    }


def evaluate_fixed_seed_gate(
    champion: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    policy: GatePolicy,
) -> dict[str, Any]:
    """Evaluate one paired offline fixed-seed promotion gate."""

    champion_by_seed = _index_runs(champion, expected_count=policy.expected_seed_count)
    candidate_by_seed = _index_runs(candidate, expected_count=policy.expected_seed_count)
    if set(champion_by_seed) != set(candidate_by_seed):
        raise ChampionGateError("Champion and Candidate must use the exact same seed set")

    champion_summary = _summary(champion_by_seed)
    candidate_summary = _summary(candidate_by_seed)
    candidate_safety = candidate_summary["safety"]

    complete = (
        champion_summary["complete_runs"] == policy.expected_seed_count
        and candidate_summary["complete_runs"] == policy.expected_seed_count
    )
    safe = all(candidate_safety[field] == 0 for field in SAFETY_FIELDS)
    win_delta = candidate_summary["victories"] - champion_summary["victories"]
    passed = complete and safe and win_delta >= policy.min_win_delta

    reasons: list[str] = []
    if not complete:
        reasons.append("incomplete_or_blocked_run")
    if not safe:
        reasons.append("candidate_safety_failure")
    if win_delta < policy.min_win_delta:
        reasons.append(
            f"win_delta_{win_delta}_below_required_{policy.min_win_delta}"
        )

    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": policy.name,
        "status": "PASS" if passed else "HOLD",
        "expected_seed_count": policy.expected_seed_count,
        "required_win_delta": policy.min_win_delta,
        "win_delta": win_delta,
        "champion": champion_summary,
        "candidate": candidate_summary,
        "reasons": reasons,
    }


def evaluate_real_game_gate(
    champion: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require paired real-game safety plus no observed performance regression."""

    champion_by_seed = _index_runs(champion, expected_count=None)
    candidate_by_seed = _index_runs(candidate, expected_count=None)
    if set(champion_by_seed) != set(candidate_by_seed):
        raise ChampionGateError("real-game Champion and Candidate must use the exact same seeds")

    champion_summary = _summary(champion_by_seed)
    candidate_summary = _summary(candidate_by_seed)
    expected = len(champion_by_seed)
    complete = (
        champion_summary["complete_runs"] == expected
        and candidate_summary["complete_runs"] == expected
    )
    safe = all(candidate_summary["safety"][field] == 0 for field in SAFETY_FIELDS)
    wins_not_worse = candidate_summary["victories"] >= champion_summary["victories"]

    champion_floor = champion_summary["mean_final_floor"]
    candidate_floor = candidate_summary["mean_final_floor"]
    floor_not_worse = (
        champion_floor is None
        or (candidate_floor is not None and candidate_floor >= champion_floor)
    )
    passed = complete and safe and wins_not_worse and floor_not_worse

    reasons: list[str] = []
    if not complete:
        reasons.append("incomplete_or_blocked_real_game_run")
    if not safe:
        reasons.append("candidate_real_game_safety_failure")
    if not wins_not_worse:
        reasons.append("real_game_victories_regressed")
    if not floor_not_worse:
        reasons.append("real_game_mean_final_floor_regressed")

    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": "real-game",
        "status": "PASS" if passed else "HOLD",
        "expected_seed_count": expected,
        "required_win_delta": 0,
        "win_delta": candidate_summary["victories"] - champion_summary["victories"],
        "champion": champion_summary,
        "candidate": candidate_summary,
        "reasons": reasons,
    }


def promotion_decision(
    fast_gate: Mapping[str, Any],
    formal_gate: Mapping[str, Any],
    real_game_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """A Candidate becomes Champion only after all three gates pass."""

    gates = {
        "fixed_seed_30": fast_gate,
        "fixed_seed_50": formal_gate,
        "real_game": real_game_gate,
    }
    passed = all(gate.get("status") == "PASS" for gate in gates.values())
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "decision": "PROMOTE" if passed else "HOLD",
        "all_gates_passed": passed,
        "gates": gates,
    }


__all__ = [
    "ChampionGateError",
    "FAST_GATE_POLICY",
    "FAST_MIN_WIN_DELTA",
    "FAST_SEED_COUNT",
    "FORMAL_GATE_POLICY",
    "FORMAL_MIN_WIN_DELTA",
    "FORMAL_SEED_COUNT",
    "GATE_SCHEMA_VERSION",
    "GatePolicy",
    "evaluate_fixed_seed_gate",
    "evaluate_real_game_gate",
    "promotion_decision",
]
