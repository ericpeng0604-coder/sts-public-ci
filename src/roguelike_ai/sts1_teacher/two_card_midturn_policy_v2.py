"""Frozen public-only V2 candidate for the audited two-card Jaw Worm midturn slice.

Chosen on the separate 460xxx discovery set before any 470xxx holdout result.
Rule: use Search when it has a unique top semantic action; if Search's top two
semantic scores are exactly tied, fall back to the existing Simple public
heuristic. No oracle information participates in policy execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .contract import DecisionContext

POLICY_ID = "jaw-worm-turn1-two-card-midturn-margin0-else-simple-v1"
MARGIN_THRESHOLD = 0.0


@dataclass(frozen=True)
class TwoCardMidturnV2Decision:
    action_id: str
    used_simple_fallback: bool
    semantic_margin: float | None


def choose_two_card_midturn_v2(
    context: DecisionContext,
    search_result: Any,
    simple_action_id: str,
) -> TwoCardMidturnV2Decision:
    """Select from public Search evidence only; tie => Simple fallback."""

    scores: dict[str, float] = {}
    ids: dict[str, list[str]] = {}
    for item in search_result.candidate_scores:
        scores[item.semantic_key] = float(item.score)
        ids.setdefault(item.semantic_key, []).append(item.action_id)

    if not scores:
        return TwoCardMidturnV2Decision(simple_action_id, True, 0.0)

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    top_key, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
    margin = top_score - second_score
    evidence_margin = margin if isfinite(margin) else None

    if len(ranked) > 1 and margin <= MARGIN_THRESHOLD:
        return TwoCardMidturnV2Decision(simple_action_id, True, evidence_margin)

    simple_key = context.action_by_id(simple_action_id).semantic_key
    if simple_key == top_key:
        return TwoCardMidturnV2Decision(simple_action_id, False, evidence_margin)

    return TwoCardMidturnV2Decision(
        sorted(ids[top_key])[0],
        False,
        evidence_margin,
    )


__all__ = [
    "MARGIN_THRESHOLD",
    "POLICY_ID",
    "TwoCardMidturnV2Decision",
    "choose_two_card_midturn_v2",
]
