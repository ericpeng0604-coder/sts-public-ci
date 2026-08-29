"""Fail-closed card admission for public-state STS1 reconstruction.

V1 supports the small Ironclad slice whose base cost is fully determined by
public card id + upgrade count. CommunicationMod's Strike_R / Defend_R names
are explicit aliases for sts_lightspeed's STRIKE_RED / DEFEND_RED. Pommel
Strike, Shrug It Off, and Finesse are audited draw-card identities; richer
draw/discard/energy effects remain outside the admitted midturn slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_CARD_PILES = ("hand", "draw_pile", "discard_pile", "exhaust_pile")
_CARD_ID_ALIASES = {
    "STRIKE_R": "STRIKE_RED",
    "DEFEND_R": "DEFEND_RED",
}
_SUPPORTED_V1_COSTS = {
    "STRIKE_RED": {0: 1, 1: 1},
    "DEFEND_RED": {0: 1, 1: 1},
    "BASH": {0: 2, 1: 2},
    "POMMEL_STRIKE": {0: 1, 1: 1},
    "SHRUG_IT_OFF": {0: 1, 1: 1},
    "FINESSE": {0: 0, 1: 0},
}


def _canonical_card_id(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    return _CARD_ID_ALIASES.get(text, text)


@dataclass(frozen=True)
class PublicCardAdmission:
    allowed: bool
    reasons: tuple[str, ...]
    card_count: int


def assess_public_cards(state: Mapping[str, Any]) -> PublicCardAdmission:
    reasons: list[str] = []
    card_count = 0

    for pile_name in _CARD_PILES:
        raw_pile = state.get(pile_name)
        if not isinstance(raw_pile, Sequence) or isinstance(raw_pile, str | bytes | bytearray):
            reasons.append(f"card_pile_not_sequence:{pile_name}")
            continue

        for index, raw_card in enumerate(raw_pile):
            card_count += 1
            path = f"{pile_name}[{index}]"
            if not isinstance(raw_card, Mapping):
                reasons.append(f"card_not_mapping:{path}")
                continue

            card_id = _canonical_card_id(raw_card.get("id"))
            if not card_id:
                reasons.append(f"missing_card_id:{path}")
                continue
            if card_id not in _SUPPORTED_V1_COSTS:
                reasons.append(f"card_unsupported_v1:{card_id}")
                continue

            upgrades = raw_card.get("upgrades")
            if isinstance(upgrades, bool) or not isinstance(upgrades, int) or upgrades not in (0, 1):
                reasons.append(f"invalid_card_upgrades:{path}")
                continue

            cost = raw_card.get("cost")
            if isinstance(cost, bool) or not isinstance(cost, int):
                reasons.append(f"invalid_card_cost:{path}")
                continue

            expected_cost = _SUPPORTED_V1_COSTS[card_id][upgrades]
            if cost != expected_cost:
                reasons.append(
                    f"temporary_or_unknown_card_cost_unsupported:{path}:{card_id}:{cost}!={expected_cost}"
                )

    return PublicCardAdmission(
        allowed=not reasons,
        reasons=tuple(sorted(set(reasons))),
        card_count=card_count,
    )


def card_reconstruction_capability(state: Mapping[str, Any]) -> bool:
    return assess_public_cards(state).allowed


__all__ = [
    "PublicCardAdmission",
    "assess_public_cards",
    "card_reconstruction_capability",
]
