"""Narrow V1 admission for public player combat state.

The pinned sts_lightspeed BattleContext is zero-based: ``turn == 0`` is the
first player turn. V1 only reconstructs that opening turn. Mid-combat counters
(cards played/discarded this turn and other history-sensitive surfaces) are not
yet in the frozen public reconstruction contract, so later turns fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_SUPPORTED_PLAYER_POWERS_V1 = frozenset({"STRENGTH", "DEXTERITY", "FOCUS", "ARTIFACT"})


def _norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True)
class PublicPlayerAdmission:
    allowed: bool
    reasons: tuple[str, ...]


def assess_public_player(state: Mapping[str, Any]) -> PublicPlayerAdmission:
    reasons: list[str] = []
    for field in ("hp", "max_hp", "block", "energy", "gold", "turn", "floor", "ascension_level"):
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            reasons.append(f"invalid_player_scalar:{field}")

    if state.get("turn") != 0:
        reasons.append(f"turn_unsupported_v1:{state.get('turn')}")

    powers = state.get("powers")
    if not isinstance(powers, Sequence) or isinstance(powers, str | bytes | bytearray):
        reasons.append("player_powers_not_sequence")
    else:
        for index, power in enumerate(powers):
            if not isinstance(power, Mapping):
                reasons.append(f"player_power_not_mapping:{index}")
                continue
            name = _norm(power.get("name"))
            if name not in _SUPPORTED_PLAYER_POWERS_V1:
                reasons.append(f"player_power_unsupported_v1:{name or 'MISSING'}")
            amount = power.get("amount")
            if isinstance(amount, bool) or not isinstance(amount, int):
                reasons.append(f"invalid_player_power_amount:{index}")

    return PublicPlayerAdmission(not reasons, tuple(sorted(set(reasons))))


__all__ = ["PublicPlayerAdmission", "assess_public_player"]
