"""Public-only Gremlin Nob Search V2 confidence fallback.

The threshold in this module was selected only on the separate committed
350xxx discovery set. The frozen 340xxx V1 holdout and fresh 360xxx V2 holdout
must never be used to tune it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .contract import ActionSpec, DecisionContext
from .search import SearchResult

GREMLIN_NOB_V2_MARGIN_THRESHOLD = 2.0
GREMLIN_NOB_V2_POLICY_ID = "sts1-public-gremlin-nob-search-v2-margin2-else-simple-v1"


class GremlinNobSearchV2Error(RuntimeError):
    pass


@dataclass(frozen=True)
class GremlinNobV2Selection:
    policy_id: str
    action_id: str
    source: str
    semantic_margin: float | None
    threshold: float
    decision_signature: str


def _normalized(value: object) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def _require_scope(context: DecisionContext) -> None:
    state = context.state
    if state.get("turn") != 0:
        raise GremlinNobSearchV2Error("nob_v2_requires_opening_turn_zero")
    enemies = state.get("enemies", [])
    if not isinstance(enemies, list) or len(enemies) != 1 or not isinstance(enemies[0], dict):
        raise GremlinNobSearchV2Error("nob_v2_requires_single_enemy")
    enemy = enemies[0]
    if _normalized(enemy.get("name")) != "GREMLIN_NOB":
        raise GremlinNobSearchV2Error("nob_v2_requires_gremlin_nob")
    if _normalized(enemy.get("intent")) not in {"BELLOW", "GREMLIN_NOB_BELLOW"}:
        raise GremlinNobSearchV2Error("nob_v2_requires_opening_bellow")


def select_gremlin_nob_search_v2(
    context: DecisionContext,
    search_result: SearchResult,
    simple_action: ActionSpec,
) -> GremlinNobV2Selection:
    """Choose Search only when its semantic lead is > 2.0; otherwise Simple.

    Every input is policy-visible/public-only. The function never receives an
    oracle, simulator BattleContext, seed, RNG state, or hidden move history.
    """
    _require_scope(context)
    if search_result.decision_signature != context.decision_signature:
        raise GremlinNobSearchV2Error("nob_v2_search_signature_mismatch")
    try:
        canonical_simple = context.action_by_id(simple_action.action_id)
    except KeyError as exc:
        raise GremlinNobSearchV2Error("nob_v2_simple_action_not_legal") from exc

    # Fail safe: incomplete Search evidence delegates to the already-public
    # simple heuristic rather than inventing confidence.
    if not search_result.resolved:
        return GremlinNobV2Selection(
            GREMLIN_NOB_V2_POLICY_ID,
            canonical_simple.action_id,
            "simple_fallback_unresolved_search",
            None,
            GREMLIN_NOB_V2_MARGIN_THRESHOLD,
            context.decision_signature,
        )

    semantic_scores: dict[str, float] = {}
    semantic_action_ids: dict[str, list[str]] = {}
    for item in search_result.candidate_scores:
        if item.unresolved or item.score is None or not math.isfinite(float(item.score)):
            return GremlinNobV2Selection(
                GREMLIN_NOB_V2_POLICY_ID,
                canonical_simple.action_id,
                "simple_fallback_incomplete_candidate",
                None,
                GREMLIN_NOB_V2_MARGIN_THRESHOLD,
                context.decision_signature,
            )
        score = float(item.score)
        previous = semantic_scores.get(item.semantic_key)
        if previous is not None and abs(previous - score) > 1e-9:
            raise GremlinNobSearchV2Error("nob_v2_semantic_duplicate_score_drift")
        semantic_scores[item.semantic_key] = score
        semantic_action_ids.setdefault(item.semantic_key, []).append(item.action_id)

    if not semantic_scores:
        raise GremlinNobSearchV2Error("nob_v2_no_search_scores")

    ranked = sorted(semantic_scores.items(), key=lambda pair: (-pair[1], pair[0]))
    top_key, top_score = ranked[0]
    if len(ranked) == 1:
        margin = math.inf
    else:
        margin = top_score - ranked[1][1]

    if margin <= GREMLIN_NOB_V2_MARGIN_THRESHOLD:
        return GremlinNobV2Selection(
            GREMLIN_NOB_V2_POLICY_ID,
            canonical_simple.action_id,
            "simple_fallback_low_margin",
            margin,
            GREMLIN_NOB_V2_MARGIN_THRESHOLD,
            context.decision_signature,
        )

    if canonical_simple.semantic_key == top_key:
        selected_id = canonical_simple.action_id
    else:
        selected_id = min(semantic_action_ids[top_key])
    context.action_by_id(selected_id)  # final legality assertion
    return GremlinNobV2Selection(
        GREMLIN_NOB_V2_POLICY_ID,
        selected_id,
        "search_confident",
        margin,
        GREMLIN_NOB_V2_MARGIN_THRESHOLD,
        context.decision_signature,
    )


__all__ = [
    "GREMLIN_NOB_V2_MARGIN_THRESHOLD",
    "GREMLIN_NOB_V2_POLICY_ID",
    "GremlinNobSearchV2Error",
    "GremlinNobV2Selection",
    "select_gremlin_nob_search_v2",
]
