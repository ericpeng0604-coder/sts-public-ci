"""Discovery-frozen candidate for the exact Anger + Finesse composition slice.

Frozen before any 530xxx oracle labels are observed. The rule is structural:
resolved Search stays inside its top semantic-score set; Simple may break a top
semantic tie only when Simple itself is in that set. Unresolved or inconsistent
Search fails safe to Simple.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any

from .contract import DecisionContext

POLICY_ID = "jaw-worm-turn1-anger-finesse-search-top-set-v1"
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class AngerFinesseCompositionDecisionV1:
    action_id: str
    used_simple_failsafe: bool
    used_simple_top_tie_break: bool
    top_semantic_count: int
    policy_id: str = POLICY_ID


def choose_anger_finesse_composition_candidate(
    context: DecisionContext,
    search_result: Any,
    simple_action_id: str,
) -> AngerFinesseCompositionDecisionV1:
    legal_ids = {action.action_id for action in context.legal_actions}
    if simple_action_id not in legal_ids:
        raise RuntimeError(f"simple_action_not_legal:{simple_action_id}")

    if (
        not search_result.resolved
        or search_result.timed_out
        or search_result.unresolved_action_ids
        or not search_result.candidate_scores
    ):
        return AngerFinesseCompositionDecisionV1(simple_action_id, True, False, 0)

    semantic_scores: dict[str, float] = {}
    semantic_action_ids: dict[str, list[str]] = {}
    for item in search_result.candidate_scores:
        if item.unresolved or item.score is None:
            return AngerFinesseCompositionDecisionV1(simple_action_id, True, False, 0)
        score = float(item.score)
        previous = semantic_scores.get(item.semantic_key)
        if previous is not None and not isclose(previous, score, rel_tol=0.0, abs_tol=TIE_TOLERANCE):
            return AngerFinesseCompositionDecisionV1(simple_action_id, True, False, 0)
        semantic_scores[item.semantic_key] = score
        semantic_action_ids.setdefault(item.semantic_key, []).append(item.action_id)

    if not semantic_scores:
        return AngerFinesseCompositionDecisionV1(simple_action_id, True, False, 0)

    top_score = max(semantic_scores.values())
    top_keys = sorted(
        key for key, score in semantic_scores.items()
        if isclose(score, top_score, rel_tol=0.0, abs_tol=TIE_TOLERANCE)
    )
    if not top_keys:
        raise RuntimeError("resolved_search_without_top_semantic_set")

    simple_key = context.action_by_id(simple_action_id).semantic_key
    if simple_key in top_keys:
        return AngerFinesseCompositionDecisionV1(
            simple_action_id, False, len(top_keys) > 1, len(top_keys)
        )

    chosen_key = top_keys[0]
    action_id = sorted(semantic_action_ids[chosen_key])[0]
    if action_id not in legal_ids:
        raise RuntimeError(f"search_top_set_action_not_legal:{action_id}")
    return AngerFinesseCompositionDecisionV1(action_id, False, False, len(top_keys))


__all__ = [
    "POLICY_ID",
    "TIE_TOLERANCE",
    "AngerFinesseCompositionDecisionV1",
    "choose_anger_finesse_composition_candidate",
]
