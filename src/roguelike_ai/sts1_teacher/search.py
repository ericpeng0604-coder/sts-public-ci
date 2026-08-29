"""Deterministic public-state search shell for STS1 Phase 1.

The search engine never receives a simulator object.  It only receives a
DecisionContext plus a rollout evaluator that explicitly declares whether it
uses hidden information.  Formal Teacher runs fail closed when the evaluator
is hidden-state dependent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Protocol

from .contract import ActionSpec, DecisionContext, sha256_json


class HiddenInformationError(RuntimeError):
    """Raised when a formal search tries to use a hidden-state evaluator."""


class PublicRolloutEvaluator(Protocol):
    evaluator_id: str
    uses_hidden_information: bool

    def score(
        self,
        context: DecisionContext,
        action: ActionSpec,
        *,
        sample_index: int,
        config: "SearchConfig",
    ) -> float | None:
        """Return one public-state rollout score, or None when unresolved."""


@dataclass(frozen=True)
class SearchConfig:
    samples_per_semantic_action: int = 4
    rollout_budget: int = 256
    node_budget: int = 4096
    max_depth: int = 16
    timeout_ms: int = 2_000
    tie_tolerance: float = 1e-9
    sampling_seed: int = 0

    def __post_init__(self) -> None:
        if self.samples_per_semantic_action <= 0:
            raise ValueError("samples_per_semantic_action must be positive")
        if self.rollout_budget <= 0:
            raise ValueError("rollout_budget must be positive")
        if self.node_budget <= 0:
            raise ValueError("node_budget must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative")

    @property
    def config_hash(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class CandidateScore:
    action_id: str
    semantic_key: str
    score: float | None
    samples: int
    unresolved: bool


@dataclass(frozen=True)
class SearchResult:
    decision_signature: str
    evaluator_id: str
    config_hash: str
    candidate_scores: tuple[CandidateScore, ...]
    tie_action_ids: tuple[str, ...]
    unique_best_action_id: str | None
    unresolved_action_ids: tuple[str, ...]
    rollout_count: int
    timed_out: bool
    evidence_hash: str

    @property
    def resolved(self) -> bool:
        return bool(self.tie_action_ids) and not self.unresolved_action_ids and not self.timed_out


class PublicStateSearch:
    """Evaluate semantic action groups with deterministic paired sampling.

    Exact executable duplicates are rejected by DecisionContext.  Semantic
    duplicates (for example two identical Strike cards in different hand slots)
    are grouped and evaluated only once.  Every member receives the same score,
    which prevents slot identity from creating score drift.
    """

    def __init__(self, evaluator: PublicRolloutEvaluator, config: SearchConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or SearchConfig()

    def run(self, context: DecisionContext) -> SearchResult:
        if bool(getattr(self.evaluator, "uses_hidden_information", True)):
            raise HiddenInformationError(
                f"formal_teacher_rejects_hidden_evaluator:{getattr(self.evaluator, 'evaluator_id', 'unknown')}"
            )

        evaluator_id = str(getattr(self.evaluator, "evaluator_id", "unknown"))
        grouped: dict[str, list[ActionSpec]] = {}
        for action in context.legal_actions:
            grouped.setdefault(action.semantic_key, []).append(action)
        for actions in grouped.values():
            actions.sort(key=lambda item: item.action_id)

        started = time.monotonic()
        rollouts = 0
        timed_out = False
        semantic_scores: dict[str, tuple[float | None, int, bool]] = {}

        for semantic_key in sorted(grouped):
            representative = grouped[semantic_key][0]
            samples: list[float] = []
            unresolved = False

            for sample_index in range(self.config.samples_per_semantic_action):
                if rollouts >= self.config.rollout_budget:
                    unresolved = True
                    break
                if (time.monotonic() - started) * 1000.0 >= self.config.timeout_ms:
                    unresolved = True
                    timed_out = True
                    break

                value = self.evaluator.score(
                    context,
                    representative,
                    sample_index=sample_index,
                    config=self.config,
                )
                rollouts += 1
                if value is None:
                    unresolved = True
                    continue
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"non_finite_rollout_score:{value!r}")
                samples.append(value)

            score = sum(samples) / len(samples) if samples else None
            semantic_scores[semantic_key] = (score, len(samples), unresolved or not samples)
            if timed_out:
                break

        candidate_scores: list[CandidateScore] = []
        unresolved_ids: list[str] = []
        for action in context.legal_actions:
            if action.semantic_key in semantic_scores:
                score, samples, unresolved = semantic_scores[action.semantic_key]
            else:
                score, samples, unresolved = None, 0, True
            candidate_scores.append(
                CandidateScore(
                    action_id=action.action_id,
                    semantic_key=action.semantic_key,
                    score=score,
                    samples=samples,
                    unresolved=unresolved,
                )
            )
            if unresolved:
                unresolved_ids.append(action.action_id)

        candidate_scores.sort(key=lambda item: item.action_id)
        resolved_scores = [item.score for item in candidate_scores if item.score is not None]
        tie_ids: list[str] = []
        if resolved_scores:
            best = max(resolved_scores)
            tie_ids = [
                item.action_id
                for item in candidate_scores
                if item.score is not None and abs(item.score - best) <= self.config.tie_tolerance
            ]

        tie_ids.sort()
        unresolved_ids.sort()
        unique_best = tie_ids[0] if len(tie_ids) == 1 else None
        evidence_payload = {
            "decision_signature": context.decision_signature,
            "evaluator_id": evaluator_id,
            "config_hash": self.config.config_hash,
            "candidate_scores": [asdict(item) for item in candidate_scores],
            "tie_action_ids": tie_ids,
            "unique_best_action_id": unique_best,
            "unresolved_action_ids": unresolved_ids,
            "rollout_count": rollouts,
            "timed_out": timed_out,
        }
        return SearchResult(
            decision_signature=context.decision_signature,
            evaluator_id=evaluator_id,
            config_hash=self.config.config_hash,
            candidate_scores=tuple(candidate_scores),
            tie_action_ids=tuple(tie_ids),
            unique_best_action_id=unique_best,
            unresolved_action_ids=tuple(unresolved_ids),
            rollout_count=rollouts,
            timed_out=timed_out,
            evidence_hash=sha256_json(evidence_payload),
        )


__all__ = [
    "CandidateScore",
    "HiddenInformationError",
    "PublicRolloutEvaluator",
    "PublicStateSearch",
    "SearchConfig",
    "SearchResult",
]
