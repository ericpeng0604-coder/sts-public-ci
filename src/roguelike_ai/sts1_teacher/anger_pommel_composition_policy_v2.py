"""Discovery-frozen V2 policy for the Anger + Pommel composition slice.

Frozen before any 510xxx discovery result and without reading 500xxx row-level
answers.

Rule:
- unresolved/timed-out Search => Simple fail-safe;
- resolved Search => compute the top semantic-score set;
- if Simple is itself in that top set, use Simple as deterministic tie-break;
- otherwise choose the lexicographically first legal action from the top
  semantic set;
- never leave a resolved Search top set merely because Search has a tie.

This is a structural correction to V1's broad tie fallback.  Discovery may
reject this V2, but may not mutate it in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any

from .contract import DecisionContext

POLICY_ID = "jaw-worm-turn1-anger-pommel-search-top-set-v2"
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class AngerPommelCompositionDecisionV2:
    action_id: str
    used_simple_failsafe: bool
    used_simple_top_tie_break: bool
    top_semantic_count: int
    policy_id: str = POLICY_ID


def choose_anger_pommel_composition_candidate_v2(
    context: DecisionContext,
    search_result: Any,
    simple_action_id: str,
) -> AngerPommelCompositionDecisionV2:
    legal_ids = {action.action_id for action in context.legal_actions}
    if simple_action_id not in legal_ids:
        raise RuntimeError(f"simple_action_not_legal:{simple_action_id}")

    if (
        not search_result.resolved
        or search_result.timed_out
        or search_result.unresolved_action_ids
        or not search_result.candidate_scores
    ):
        return AngerPommelCompositionDecisionV2(
            simple_action_id,
            True,
            False,
            0,
        )

    semantic_scores: dict[str, float] = {}
    semantic_action_ids: dict[str, list[str]] = {}
    for item in search_result.candidate_scores:
        if item.unresolved or item.score is None:
            return AngerPommelCompositionDecisionV2(
                simple_action_id,
                True,
                False,
                0,
            )
        score = float(item.score)
        existing = semantic_scores.get(item.semantic_key)
        if existing is not None and not isclose(existing, score, rel_tol=0.0, abs_tol=TIE_TOLERANCE):
            return AngerPommelCompositionDecisionV2(
                simple_action_id,
                True,
                False,
                0,
            )
        semantic_scores[item.semantic_key] = score
        semantic_action_ids.setdefault(item.semantic_key, []).append(item.action_id)

    if not semantic_scores:
        return AngerPommelCompositionDecisionV2(
            simple_action_id,
            True,
            False,
            0,
        )

    top_score = max(semantic_scores.values())
    top_keys = sorted(
        key
        for key, score in semantic_scores.items()
        if isclose(score, top_score, rel_tol=0.0, abs_tol=TIE_TOLERANCE)
    )
    if not top_keys:
        raise RuntimeError("resolved_search_without_top_semantic_set")

    simple_key = context.action_by_id(simple_action_id).semantic_key
    if simple_key in top_keys:
        return AngerPommelCompositionDecisionV2(
            simple_action_id,
            False,
            len(top_keys) > 1,
            len(top_keys),
        )

    chosen_key = top_keys[0]
    action_id = sorted(semantic_action_ids[chosen_key])[0]
    if action_id not in legal_ids:
        raise RuntimeError(f"search_top_set_action_not_legal:{action_id}")
    return AngerPommelCompositionDecisionV2(
        action_id,
        False,
        False,
        len(top_keys),
    )


__all__ = [
    "POLICY_ID",
    "TIE_TOLERANCE",
    "AngerPommelCompositionDecisionV2",
    "choose_anger_pommel_composition_candidate_v2",
]
