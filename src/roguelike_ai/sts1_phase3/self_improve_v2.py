"""State and gates for the STS1 continuous self-improvement loop v2.

The learner track may accumulate safe incremental gains between formal Champion
promotions. The formal Champion gates remain unchanged (+4 / +5 wins).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LOOP_V2_SCHEMA_VERSION = "sts1-self-improve-loop-v2"
LEARNER_GATE_SCHEMA_VERSION = "sts1-learner-gate-v1"
SAFETY_FIELDS = (
    "illegal_action_count",
    "timeout_count",
    "crash_count",
    "remote_error_count",
)


class LoopV2Error(RuntimeError):
    """Continuous self-improvement state/evidence is invalid."""


@dataclass(frozen=True)
class LearnerGatePolicy:
    min_floor_delta: float = 0.5

    def __post_init__(self) -> None:
        if self.min_floor_delta < 0:
            raise ValueError("min_floor_delta must be non-negative")


@dataclass(frozen=True)
class LoopV2State:
    round_index: int = 0
    accepted_rounds: int = 0
    rejected_rounds: int = 0
    stagnation_count: int = 0
    used_teacher_seeds: tuple[int, ...] = ()
    used_ppo_seeds: tuple[int, ...] = ()
    schema_version: str = LOOP_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOOP_V2_SCHEMA_VERSION:
            raise ValueError("loop v2 schema mismatch")
        for name in ("round_index", "accepted_rounds", "rejected_rounds", "stagnation_count"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if len(set(self.used_teacher_seeds)) != len(self.used_teacher_seeds):
            raise ValueError("duplicate Teacher seeds in loop state")
        if len(set(self.used_ppo_seeds)) != len(self.used_ppo_seeds):
            raise ValueError("duplicate PPO seeds in loop state")
        if set(self.used_teacher_seeds) & set(self.used_ppo_seeds):
            raise ValueError("Teacher and PPO seed histories must remain disjoint")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["used_teacher_seeds"] = list(self.used_teacher_seeds)
        payload["used_ppo_seeds"] = list(self.used_ppo_seeds)
        return payload

    @classmethod
    def from_path(cls, path: Path) -> "LoopV2State":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise LoopV2Error("loop state must be a JSON object")
        return cls(
            round_index=int(payload.get("round_index", 0)),
            accepted_rounds=int(payload.get("accepted_rounds", 0)),
            rejected_rounds=int(payload.get("rejected_rounds", 0)),
            stagnation_count=int(payload.get("stagnation_count", 0)),
            used_teacher_seeds=tuple(int(x) for x in payload.get("used_teacher_seeds", ())),
            used_ppo_seeds=tuple(int(x) for x in payload.get("used_ppo_seeds", ())),
            schema_version=str(payload.get("schema_version", "")),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)


def _runs(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        raw = value.get("runs")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise LoopV2Error("evaluation aggregate must contain runs")
        rows = list(raw)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = list(value)
    else:
        raise LoopV2Error("evaluation evidence must be runs or aggregate")
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise LoopV2Error("evaluation runs must be non-empty mappings")
    return rows


def _index(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in _runs(value):
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise LoopV2Error("evaluation run is missing seed")
        key = str(seed)
        if key in indexed:
            raise LoopV2Error(f"duplicate evaluation seed: {key}")
        indexed[key] = row
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
    safety = {}
    for field in SAFETY_FIELDS:
        total = 0
        for row in rows:
            raw = row.get(field, 0)
            if raw is None:
                raw = 0
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise LoopV2Error(f"invalid safety field {field}: {raw!r}")
            total += raw
        safety[field] = total
    return {
        "seeds": len(rows),
        "complete_runs": len(complete),
        "victories": sum(int(row.get("outcome") == "victory") for row in complete),
        "mean_final_floor": sum(floors) / len(floors) if floors else None,
        "safety": safety,
    }


def evaluate_learner_gate(
    current: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    policy: LearnerGatePolicy | None = None,
) -> dict[str, Any]:
    """Keep incremental learner gains without weakening the formal Champion gate."""

    policy = policy or LearnerGatePolicy()
    left = _index(current)
    right = _index(candidate)
    if set(left) != set(right):
        raise LoopV2Error("Learner and Candidate must use the same eval seeds")

    current_summary = _summary(left)
    candidate_summary = _summary(right)
    expected = len(left)
    complete = (
        current_summary["complete_runs"] == expected
        and candidate_summary["complete_runs"] == expected
    )
    safe = all(candidate_summary["safety"][field] == 0 for field in SAFETY_FIELDS)
    win_delta = candidate_summary["victories"] - current_summary["victories"]
    current_floor = current_summary["mean_final_floor"]
    candidate_floor = candidate_summary["mean_final_floor"]
    floor_delta = (
        None
        if current_floor is None or candidate_floor is None
        else candidate_floor - current_floor
    )

    improved = win_delta > 0 or (
        win_delta == 0
        and floor_delta is not None
        and floor_delta >= policy.min_floor_delta
    )
    passed = complete and safe and improved
    reasons: list[str] = []
    if not complete:
        reasons.append("incomplete_eval")
    if not safe:
        reasons.append("candidate_safety_failure")
    if not improved:
        reasons.append("no_measurable_learner_improvement")

    return {
        "schema_version": LEARNER_GATE_SCHEMA_VERSION,
        "status": "ACCEPT" if passed else "ROLLBACK",
        "min_floor_delta": policy.min_floor_delta,
        "win_delta": win_delta,
        "floor_delta": floor_delta,
        "current": current_summary,
        "candidate": candidate_summary,
        "reasons": reasons,
    }


def ppo_allowed(
    distillation_stats: Mapping[str, Any],
    *,
    min_teacher_top1_accuracy: float = 0.55,
) -> dict[str, Any]:
    accuracy = float(distillation_stats.get("teacher_top1_accuracy", 0.0))
    allowed = accuracy >= min_teacher_top1_accuracy
    return {
        "schema_version": LOOP_V2_SCHEMA_VERSION,
        "allowed": allowed,
        "teacher_top1_accuracy": accuracy,
        "required_teacher_top1_accuracy": min_teacher_top1_accuracy,
        "reason": "teacher_agreement_ready" if allowed else "teacher_agreement_too_low",
    }


__all__ = [
    "LEARNER_GATE_SCHEMA_VERSION",
    "LOOP_V2_SCHEMA_VERSION",
    "LearnerGatePolicy",
    "LoopV2Error",
    "LoopV2State",
    "evaluate_learner_gate",
    "ppo_allowed",
]
