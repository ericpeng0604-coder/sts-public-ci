"""Frozen public-only candidate for the audited Anger midturn slice.

This policy is frozen before any 480xxx fresh-holdout quality result.
Rule: use Search only when it is resolved and has a unique top semantic action;
otherwise fall back to the existing Simple public heuristic.  No oracle,
source seed, source BattleContext, hidden RNG, move history, or reconstruction
auxiliary counter is consulted by the policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .contract import DecisionContext

POLICY_ID = "jaw-worm-turn1-anger-midturn-margin0-else-simple-v1"
MARGIN_THRESHOLD = 0.0


@dataclass(frozen=True)
class AngerMidturnDecision:
    action_id: str
    used_simple_fallback: bool
    semantic_margin: float | None
    policy_id: str = POLICY_ID


def choose_anger_midturn_candidate(
    context: DecisionContext,
    search_result: Any,
    simple_action_id: str,
) -> AngerMidturnDecision:
    """Choose from formal public Search evidence only; tie/failure => Simple."""

    if (
        not search_result.resolved
        or search_result.timed_out
        or search_result.unresolved_action_ids
        or not search_result.candidate_scores
    ):
        return AngerMidturnDecision(simple_action_id, True, 0.0)

    scores: dict[str, float] = {}
    ids: dict[str, list[str]] = {}
    for item in search_result.candidate_scores:
        if item.unresolved or item.score is None:
            return AngerMidturnDecision(simple_action_id, True, 0.0)
        score = float(item.score)
        scores[item.semantic_key] = score
        ids.setdefault(item.semantic_key, []).append(item.action_id)

    if not scores:
        return AngerMidturnDecision(simple_action_id, True, 0.0)

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    top_key, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
    margin = top_score - second_score
    evidence_margin = margin if isfinite(margin) else None

    if len(ranked) > 1 and margin <= MARGIN_THRESHOLD:
        return AngerMidturnDecision(simple_action_id, True, evidence_margin)

    simple_key = context.action_by_id(simple_action_id).semantic_key
    if simple_key == top_key:
        return AngerMidturnDecision(simple_action_id, False, evidence_margin)

    return AngerMidturnDecision(
        sorted(ids[top_key])[0],
        False,
        evidence_margin,
    )


__all__ = [
    "AngerMidturnDecision",
    "MARGIN_THRESHOLD",
    "POLICY_ID",
    "choose_anger_midturn_candidate",
]
