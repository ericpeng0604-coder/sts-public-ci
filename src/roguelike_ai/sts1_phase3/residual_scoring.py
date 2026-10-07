"""Shared choice-relative residual math for STS1 training and inference."""

from __future__ import annotations

from typing import Any


def _as_index(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def top1_index(scores: Any) -> int:
    """Return the first highest-scoring legal choice, matching torch.argmax ties."""
    if len(scores) == 0:
        raise ValueError("cannot select from an empty score vector")
    if hasattr(scores, "argmax"):
        return _as_index(scores.argmax())
    return max(range(len(scores)), key=lambda i: float(scores[i]))


def adapter_kind_enabled(kind: str, enabled_kinds: Any) -> bool:
    """Match the simulator's per-kind adapter enablement contract."""
    return str(kind) in {str(value) for value in enabled_kinds}


def choice_relative_margin_shortfall(
    parent_scores: Any,
    residual_scores: Any,
    target_index: int,
    margin: float,
) -> tuple[Any, int]:
    """Return parent-gap + margin - (target residual - parent-top residual).

    The returned first element remains a tensor when tensor inputs are supplied,
    so the caller can apply a differentiable ReLU and backpropagate it.
    """
    if len(parent_scores) != len(residual_scores) or len(parent_scores) < 2:
        raise ValueError("parent and residual scores must cover the same >=2 choices")
    target = int(target_index)
    if not 0 <= target < len(parent_scores):
        raise ValueError("target index is outside the legal choice vector")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    parent_top = top1_index(parent_scores)
    shortfall = (
        parent_scores[parent_top] - parent_scores[target] + float(margin)
        - (residual_scores[target] - residual_scores[parent_top])
    )
    return shortfall, parent_top


def choice_relative_margin_loss(
    parent_scores: Any,
    residual_scores: Any,
    target_index: int,
    margin: float,
    relu: Any,
) -> Any | None:
    """Return differentiable margin loss, or None when Teacher already is parent top1."""
    shortfall, parent_top = choice_relative_margin_shortfall(
        parent_scores, residual_scores, target_index, margin
    )
    if int(target_index) == parent_top:
        return None
    return relu(shortfall)


def apply_residual_scores(parent_scores: Any, residual_scores: Any, gate_allowed: bool) -> Any:
    """Apply the exact stored residual once, and only when the gate allows it."""
    if len(parent_scores) != len(residual_scores):
        raise ValueError("parent and residual score vectors have different choice counts")
    if not gate_allowed:
        return parent_scores
    if hasattr(parent_scores, "argmax"):
        return parent_scores + residual_scores
    return [float(p) + float(r) for p, r in zip(parent_scores, residual_scores)]
