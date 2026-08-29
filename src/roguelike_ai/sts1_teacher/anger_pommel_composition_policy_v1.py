"""Frozen candidate policy for the Anger + Pommel fresh quality gate.

Rule frozen before any 500xxx quality result:
- use Search only when it has one unique best action;
- otherwise fall back to the existing Simple public heuristic.

The zero margin is intentional: this is the same conservative tie fallback
shape already validated for isolated Anger.  No 500xxx result may be used to
change this policy or threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

POLICY_ID = "jaw-worm-turn1-anger-pommel-composition-margin0-else-simple-v1"
MARGIN_THRESHOLD = 0.0


@dataclass(frozen=True)
class AngerPommelCompositionCandidate:
    action_id: str
    used_simple_fallback: bool
    semantic_margin: float | None


def choose_anger_pommel_composition_candidate(context, search_result, simple_action_id: str):
    legal_ids = {action.action_id for action in context.legal_actions}
    if simple_action_id not in legal_ids:
        raise RuntimeError(f"simple_action_not_legal:{simple_action_id}")

    unique = search_result.unique_best_action_id
    if unique is not None:
        if unique not in legal_ids:
            raise RuntimeError(f"search_action_not_legal:{unique}")
        scores = sorted(
            (
                float(item.score),
                item.action_id,
            )
            for item in search_result.candidate_scores
            if item.score is not None and not item.unresolved
        )
        margin = None
        if len(scores) >= 2:
            margin = scores[-1][0] - scores[-2][0]
        return AngerPommelCompositionCandidate(
            action_id=unique,
            used_simple_fallback=False,
            semantic_margin=margin,
        )

    return AngerPommelCompositionCandidate(
        action_id=simple_action_id,
        used_simple_fallback=True,
        semantic_margin=None,
    )


__all__ = [
    "POLICY_ID",
    "MARGIN_THRESHOLD",
    "AngerPommelCompositionCandidate",
    "choose_anger_pommel_composition_candidate",
]
