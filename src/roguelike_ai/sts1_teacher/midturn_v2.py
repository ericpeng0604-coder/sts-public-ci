"""Frozen public-only V2 candidate for Jaw Worm turn-1 one-card midturn states.

Discovery on disjoint 430xxx development seeds selected exactly one rule before
fresh 440xxx holdout evaluation:

* compute the top semantic Search score margin;
* if Search is tied, unresolved, malformed, or margin <= 1.0, use the existing
  simple public heuristic;
* otherwise use Search's unique top semantic action.

The policy is deliberately scoped to the audited one-card midturn slice and
uses no source seed, move history, hidden counter, generation metadata, or
oracle information.
"""
from __future__ import annotations

from dataclasses import dataclass

from .baselines import simple_public_heuristic
from .contract import DecisionContext

POLICY_ID = "jaw-worm-turn1-midturn-margin1-else-simple-v1"
FROZEN_MARGIN = 1.0


@dataclass(frozen=True)
class MidturnV2Choice:
    action_id: str
    used_simple_fallback: bool
    semantic_margin: float
    policy_id: str = POLICY_ID


def _require_scope(context: DecisionContext) -> None:
    state = context.state
    if state.get("turn") != 1:
        raise ValueError("midturn_v2_scope_requires_turn_1")
    hand = state.get("hand", [])
    if not isinstance(hand, list) or len(hand) != 4:
        raise ValueError("midturn_v2_scope_requires_one_card_already_played")
    enemies = state.get("enemies", [])
    if not isinstance(enemies, list) or len(enemies) != 1 or enemies[0].get("name") != "JAW_WORM":
        raise ValueError("midturn_v2_scope_requires_jaw_worm")


def choose_midturn_v2(context: DecisionContext, search_result) -> MidturnV2Choice:
    """Choose one action using only formal public state and public Search evidence."""
    _require_scope(context)
    simple = simple_public_heuristic(context)
    if simple is None:
        raise ValueError("midturn_v2_simple_fallback_missing")

    if (
        not search_result.resolved
        or search_result.timed_out
        or search_result.unresolved_action_ids
        or not search_result.candidate_scores
    ):
        return MidturnV2Choice(simple.action_id, True, 0.0)

    group_scores: dict[str, float] = {}
    group_action_ids: dict[str, list[str]] = {}
    for item in search_result.candidate_scores:
        if item.unresolved:
            return MidturnV2Choice(simple.action_id, True, 0.0)
        group_scores[item.semantic_key] = float(item.score)
        group_action_ids.setdefault(item.semantic_key, []).append(item.action_id)

    ranked = sorted(group_scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if not ranked:
        return MidturnV2Choice(simple.action_id, True, 0.0)
    top_key, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
    margin = top_score - second_score

    if len(ranked) > 1 and abs(margin) <= 1e-9:
        return MidturnV2Choice(simple.action_id, True, margin)
    if margin <= FROZEN_MARGIN:
        return MidturnV2Choice(simple.action_id, True, margin)

    simple_key = context.action_by_id(simple.action_id).semantic_key
    if simple_key == top_key:
        return MidturnV2Choice(simple.action_id, False, margin)
    action_id = sorted(group_action_ids[top_key])[0]
    return MidturnV2Choice(action_id, False, margin)


__all__ = ["FROZEN_MARGIN", "MidturnV2Choice", "POLICY_ID", "choose_midturn_v2"]
