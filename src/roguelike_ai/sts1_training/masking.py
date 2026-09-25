"""Legal-action masking helpers for the STS1 Student contract."""

from __future__ import annotations

import math
from typing import Sequence

from .dataset import STS1DatasetError


def build_legal_action_mask(
    action_space: Sequence[str],
    legal_action_ids: Sequence[str],
) -> tuple[bool, ...]:
    """Map a stable action head to a fail-closed legal-action mask."""

    if not action_space:
        raise STS1DatasetError("action_space must not be empty")
    if len(action_space) != len(set(action_space)):
        raise STS1DatasetError("action_space contains duplicate action ids")
    legal = set(legal_action_ids)
    if not legal:
        raise STS1DatasetError("legal_action_ids must not be empty")
    unknown = sorted(legal.difference(action_space))
    if unknown:
        raise STS1DatasetError(f"legal actions are absent from Student action_space: {unknown}")
    return tuple(action_id in legal for action_id in action_space)


def masked_argmax(logits: Sequence[float], legal_mask: Sequence[bool]) -> int:
    """Return the highest-scoring legal index; ties resolve to the first index."""

    if len(logits) != len(legal_mask):
        raise STS1DatasetError("logits and legal_mask must have equal length")
    if not logits:
        raise STS1DatasetError("masked_argmax requires at least one action")
    if not any(legal_mask):
        raise STS1DatasetError("legal-action mask contains no legal action")

    best_index: int | None = None
    best_value = -math.inf
    for index, (raw_value, legal) in enumerate(zip(logits, legal_mask)):
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise STS1DatasetError("logits must be finite numbers")
        value = float(raw_value)
        if not math.isfinite(value):
            raise STS1DatasetError("logits must be finite numbers")
        if legal and (best_index is None or value > best_value):
            best_index = index
            best_value = value
    if best_index is None:
        raise STS1DatasetError("legal-action mask contains no legal action")
    return best_index


__all__ = ["build_legal_action_mask", "masked_argmax"]
