"""Fair deterministic future-sampling helpers for STS1 public-state search.

Formal Phase 1 search cannot copy the simulator's real hidden RNG trajectory.
Instead, every candidate must be evaluated against the same reproducible sample
identity derived only from public decision state plus frozen benchmark config.
This module provides that pairing contract; an actual simulator rollout backend
must still prove that it reconstructs/samples futures without hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

from .contract import ActionSpec, DecisionContext
from .search import SearchConfig


@dataclass(frozen=True)
class PublicSample:
    """One reproducible sample identity shared across every candidate action."""

    sample_index: int
    sample_seed: int
    sample_key: str


class PublicSampleRolloutBackend(Protocol):
    backend_id: str
    uses_hidden_information: bool

    def rollout(
        self,
        context: DecisionContext,
        action: ActionSpec,
        sample: PublicSample,
        *,
        config: SearchConfig,
    ) -> float | None:
        """Score one candidate under one public re-determinized future sample."""


def public_sample(context: DecisionContext, *, sample_index: int, config: SearchConfig) -> PublicSample:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    material = (
        f"sts1-public-sample-v1|{config.sampling_seed}|"
        f"{context.decision_signature}|{sample_index}"
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    return PublicSample(sample_index=sample_index, sample_seed=seed, sample_key=digest.hex())


class PairedPublicSampleEvaluator:
    """Adapt a verified public rollout backend to ``PublicStateSearch``.

    The sample identity does not include the action id. Therefore action A and
    action B receive the exact same sample seed for sample index N. This is the
    paired/randomized fairness rule required by #345.
    """

    uses_hidden_information = False

    def __init__(self, backend: PublicSampleRolloutBackend) -> None:
        if bool(getattr(backend, "uses_hidden_information", True)):
            raise ValueError(
                f"paired_public_sampler_rejects_hidden_backend:{getattr(backend, 'backend_id', 'unknown')}"
            )
        self.backend = backend
        self.evaluator_id = f"paired-public-sample-v1:{getattr(backend, 'backend_id', 'unknown')}"

    def score(
        self,
        context: DecisionContext,
        action: ActionSpec,
        *,
        sample_index: int,
        config: SearchConfig,
    ) -> float | None:
        sample = public_sample(context, sample_index=sample_index, config=config)
        return self.backend.rollout(context, action, sample, config=config)


__all__ = [
    "PairedPublicSampleEvaluator",
    "PublicSample",
    "PublicSampleRolloutBackend",
    "public_sample",
]
