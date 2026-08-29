"""Fail-closed card admission for public-state STS1 reconstruction.

Most STS1 card instances can be rebuilt from public card identity, upgrade
count, and currently displayed cost.  A small set carries extra combat/run
history in ``CardInstance.specialData`` or persistent damage-driven cost state;
those cards are deliberately unsupported until an equally public source for
that state is verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_UNSUPPORTED_HISTORY_CARD_IDS = frozenset(
    {
        "SEARING_BLOW",
        "RAMPAGE",
        "GENETIC_ALGORITHM",
        "RITUAL_DAGGER",
        "BLOOD_FOR_BLOOD",
        "MASTERFUL_STAB",
    }
)
_CARD_PILES = ("hand", "draw_pile", "discard_pile", "exhaust_pile")


def _canonical_card_id(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text.replace(" ", "_").replace("-", "_")


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
            elif card_id in _UNSUPPORTED_HISTORY_CARD_IDS:
                reasons.append(f"history_card_unsupported:{card_id}")

            upgrades = raw_card.get("upgrades")
            if isinstance(upgrades, bool) or not isinstance(upgrades, int) or upgrades < 0:
                reasons.append(f"invalid_card_upgrades:{path}")

            cost = raw_card.get("cost")
            if isinstance(cost, bool) or not isinstance(cost, int):
                reasons.append(f"invalid_card_cost:{path}")

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
