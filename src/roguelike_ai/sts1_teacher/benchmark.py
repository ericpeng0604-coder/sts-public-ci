"""Frozen Phase-1 benchmark accounting for Arm B against an independent MCTS oracle.

The native oracle is intentionally outside the formal Teacher path because it
may use simulator-private RNG while searching.  This module only consumes its
per-action scores after the public DecisionContext and exact legal-action map
have already been frozen for a root.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .contract import DecisionContext, canonical_json, sha256_json

BENCHMARK_SCHEMA_VERSION = "sts1-phase1-frozen-oracle-benchmark-v1"
ORACLE_MCTS_SIMS = 2000
ORACLE_TIE_TOLERANCE = 1e-9
EXPECTED_HELDOUT_SEEDS = 50


class BenchmarkContractError(ValueError):
    """Fail-closed benchmark contract error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_heldout_seeds(path: str | Path) -> tuple[int, ...]:
    raw = Path(path).read_text(encoding="utf-8").splitlines()
    seeds = tuple(int(line.strip()) for line in raw if line.strip())
    if len(seeds) != EXPECTED_HELDOUT_SEEDS or len(set(seeds)) != EXPECTED_HELDOUT_SEEDS:
        raise BenchmarkContractError(
            f"heldout_seed_contract:{len(seeds)}:{len(set(seeds))}:{EXPECTED_HELDOUT_SEEDS}"
        )
    return seeds


def oracle_ties(scores: Mapping[str, float], *, tolerance: float = ORACLE_TIE_TOLERANCE) -> tuple[str, ...]:
    if not scores:
        raise BenchmarkContractError("oracle_scores_empty")
    if set(scores) != {str(key) for key in scores}:
        raise BenchmarkContractError("oracle_action_ids_must_be_strings")
    best = max(float(score) for score in scores.values())
    return tuple(sorted(action_id for action_id, score in scores.items() if abs(float(score) - best) <= tolerance))


def conservative_tie_agreement(candidate_ties: Sequence[str], oracle_tie_ids: Sequence[str]) -> bool:
    """Count a tied policy as correct only when every claimed top action is oracle-optimal."""

    candidate = set(candidate_ties)
    oracle = set(oracle_tie_ids)
    return bool(candidate) and candidate <= oracle


def single_action_agreement(action_id: str | None, oracle_tie_ids: Sequence[str]) -> bool:
    return action_id is not None and action_id in set(oracle_tie_ids)


def exact_native_action_map(
    context: DecisionContext,
    public_legal_actions: Sequence[Mapping[str, Any]],
    native_actions: Sequence[Any],
) -> dict[str, Any]:
    """Map projected public actions to exact native actions with no fallback/index guessing.

    ``SimulatorCombatAdapter`` emits public actions in the same order as the
    native sequence.  The public payload hash is the frozen executable action
    id; every native item must resolve to exactly one DecisionContext action.
    """

    if len(public_legal_actions) != len(native_actions):
        raise BenchmarkContractError(
            f"native_public_action_count_mismatch:{len(native_actions)}:{len(public_legal_actions)}"
        )
    mapped: dict[str, Any] = {}
    for public_action, native_action in zip(public_legal_actions, native_actions, strict=True):
        if not isinstance(public_action, Mapping):
            raise BenchmarkContractError("public_legal_action_not_mapping")
        action_id = sha256_json(dict(public_action))
        try:
            context.action_by_id(action_id)
        except KeyError as exc:
            raise BenchmarkContractError(f"projected_action_not_in_context:{action_id}") from exc
        if action_id in mapped:
            raise BenchmarkContractError(f"duplicate_native_action_mapping:{action_id}")
        mapped[action_id] = native_action
    expected = {action.action_id for action in context.legal_actions}
    if set(mapped) != expected:
        raise BenchmarkContractError("native_action_mapping_not_bijective")
    return mapped


def semantic_score_consistent(
    context: DecisionContext,
    scores: Mapping[str, float],
    *,
    tolerance: float = ORACLE_TIE_TOLERANCE,
) -> bool:
    groups: dict[str, list[float]] = {}
    for action in context.legal_actions:
        if action.action_id not in scores:
            return False
        groups.setdefault(action.semantic_key, []).append(float(scores[action.action_id]))
    return all(max(values) - min(values) <= tolerance for values in groups.values())


@dataclass(frozen=True)
class BenchmarkDecision:
    seed: int
    floor: int
    combat_index: int
    decision_index: int
    decision_signature: str
    legal_action_ids: tuple[str, ...]
    oracle_scores: dict[str, float]
    oracle_tie_ids: tuple[str, ...]
    armb_tie_ids: tuple[str, ...]
    random_action_id: str
    simple_action_id: str
    trajectory_action_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "seed": self.seed,
            "floor": self.floor,
            "combat_index": self.combat_index,
            "decision_index": self.decision_index,
            "decision_signature": self.decision_signature,
            "legal_action_ids": list(self.legal_action_ids),
            "oracle_scores": dict(sorted(self.oracle_scores.items())),
            "oracle_tie_ids": list(self.oracle_tie_ids),
            "armb_tie_ids": list(self.armb_tie_ids),
            "random_action_id": self.random_action_id,
            "simple_action_id": self.simple_action_id,
            "trajectory_action_id": self.trajectory_action_id,
            "agreement": {
                "armb": conservative_tie_agreement(self.armb_tie_ids, self.oracle_tie_ids),
                "random": single_action_agreement(self.random_action_id, self.oracle_tie_ids),
                "simple": single_action_agreement(self.simple_action_id, self.oracle_tie_ids),
            },
        }


def canonical_decision_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    normalized = [dict(row) for row in rows]
    normalized.sort(key=lambda row: (
        int(row["seed"]), int(row["combat_index"]), int(row["decision_index"]), str(row["decision_signature"])
    ))
    return sha256_bytes(("\n".join(canonical_json(row) for row in normalized) + "\n").encode("utf-8"))


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    unresolved: int = 0,
    illegal: int = 0,
    leakage: int = 0,
) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        agreement = row.get("agreement", {})
        for name in ("armb", "random", "simple"):
            counts[name] += int(bool(agreement.get(name)))
    total = len(rows)
    rates = {name: (counts[name] / total if total else 0.0) for name in ("armb", "random", "simple")}
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "oracle_mcts_sims": ORACLE_MCTS_SIMS,
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "decision_count": total,
        "agreement_counts": dict(counts),
        "agreement_rates": rates,
        "unresolved": int(unresolved),
        "illegal": int(illegal),
        "leakage": int(leakage),
        "decision_digest": canonical_decision_digest(rows),
    }


def phase1_baseline_gate(summary: Mapping[str, Any], *, deterministic: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(summary.get("seed_count", 0)) != EXPECTED_HELDOUT_SEEDS:
        reasons.append("heldout_seed_count_not_50")
    if int(summary.get("decision_count", 0)) <= 0:
        reasons.append("no_benchmark_decisions")
    for name in ("unresolved", "illegal", "leakage"):
        if int(summary.get(name, -1)) != 0:
            reasons.append(f"{name}_must_be_zero")
    if not deterministic:
        reasons.append("full_benchmark_not_deterministic")
    rates = summary.get("agreement_rates", {})
    armb = float(rates.get("armb", 0.0))
    random = float(rates.get("random", 0.0))
    simple = float(rates.get("simple", 0.0))
    if not armb > random:
        reasons.append("armb_not_strictly_better_than_random")
    if not armb > simple:
        reasons.append("armb_not_strictly_better_than_simple")
    return not reasons, reasons


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_ndjson(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkContractError",
    "BenchmarkDecision",
    "EXPECTED_HELDOUT_SEEDS",
    "ORACLE_MCTS_SIMS",
    "ORACLE_TIE_TOLERANCE",
    "canonical_decision_digest",
    "conservative_tie_agreement",
    "exact_native_action_map",
    "load_heldout_seeds",
    "oracle_ties",
    "phase1_baseline_gate",
    "semantic_score_consistent",
    "sha256_file",
    "single_action_agreement",
    "summarize_rows",
    "write_json",
    "write_ndjson",
]
