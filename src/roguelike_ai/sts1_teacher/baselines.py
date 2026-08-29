"""Simple public-state baselines for STS1 Phase 1 benchmarks."""

from __future__ import annotations

import hashlib

from .contract import ActionSpec, DecisionContext, canonical_json


def deterministic_random_legal(context: DecisionContext, *, benchmark_seed: int = 0) -> ActionSpec | None:
    """Choose a reproducible pseudo-random legal action without game hidden state."""

    if not context.legal_actions:
        return None
    material = f"{benchmark_seed}:{context.decision_signature}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(context.legal_actions)
    return context.legal_actions[index]


def _hand_card(context: DecisionContext, hand_index: int) -> dict | None:
    for fallback_index, card in enumerate(context.state.get("hand", []), start=1):
        if not isinstance(card, dict):
            continue
        if card.get("position", fallback_index) == hand_index:
            return card
    return None


def simple_public_heuristic(context: DecisionContext) -> ActionSpec | None:
    """A deliberately weak, fully public baseline.

    Priority is deterministic and intentionally simple: playable attacks first,
    then other cards, then usable potions, then remaining legal choices, with
    end-turn last.  It is a benchmark floor, not the formal Teacher.
    """

    if not context.legal_actions:
        return None

    def rank(action: ActionSpec) -> tuple[int, int, str]:
        payload = action.payload
        kind = str(payload.get("kind", ""))
        if kind == "play_card":
            hand_index = payload.get("hand_index")
            card = _hand_card(context, hand_index) if isinstance(hand_index, int) else None
            card_type = str((card or {}).get("type", "")).upper()
            cost = (card or {}).get("cost")
            normalized_cost = cost if isinstance(cost, int) and not isinstance(cost, bool) else 99
            if card_type == "ATTACK":
                return (0, normalized_cost, action.action_id)
            return (1, normalized_cost, action.action_id)
        if kind == "use_potion":
            return (2, 0, action.action_id)
        if kind == "end_turn":
            return (9, 0, action.action_id)
        return (3, 0, canonical_json(payload))

    return min(context.legal_actions, key=rank)


__all__ = ["deterministic_random_legal", "simple_public_heuristic"]
