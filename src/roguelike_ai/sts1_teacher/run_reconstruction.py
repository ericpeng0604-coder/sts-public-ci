"""Narrow V1 admission for public run-level state used by native reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_SUPPORTED_RELICS_V1 = frozenset({"BURNING_BLOOD"})
_SUPPORTED_SOURCES_V1 = frozenset({"SIMULATOR", "REAL_GAME"})
_EMPTY_POTION = "EMPTY_POTION_SLOT"


def _norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True)
class PublicRunAdmission:
    relics_allowed: bool
    potions_allowed: bool
    reasons: tuple[str, ...]


def assess_public_run_state(state: Mapping[str, Any]) -> PublicRunAdmission:
    reasons: list[str] = []
    relic_reasons: list[str] = []
    potion_reasons: list[str] = []

    if _norm(state.get("source")) not in _SUPPORTED_SOURCES_V1:
        reasons.append("source_unsupported_v1")
    if _norm(state.get("character")) != "IRONCLAD":
        reasons.append("character_unsupported_v1")
    if _norm(state.get("room")) != "COMBAT":
        reasons.append("room_unsupported_v1")

    relics = state.get("relics")
    if not isinstance(relics, Sequence) or isinstance(relics, str | bytes | bytearray):
        relic_reasons.append("relics_not_sequence")
    else:
        for index, relic in enumerate(relics):
            if not isinstance(relic, Mapping):
                relic_reasons.append(f"relic_not_mapping:{index}")
                continue
            relic_id = _norm(relic.get("id"))
            if relic_id not in _SUPPORTED_RELICS_V1:
                relic_reasons.append(f"relic_unsupported_v1:{relic_id or 'MISSING'}")

    potions = state.get("potions")
    if not isinstance(potions, Sequence) or isinstance(potions, str | bytes | bytearray):
        potion_reasons.append("potions_not_sequence")
    else:
        if len(potions) > 5:
            potion_reasons.append(f"potion_capacity_unsupported_v1:{len(potions)}")
        for index, potion in enumerate(potions):
            if not isinstance(potion, Mapping):
                potion_reasons.append(f"potion_not_mapping:{index}")
                continue
            potion_id = _norm(potion.get("id"))
            if potion_id != _EMPTY_POTION:
                potion_reasons.append(f"potion_unsupported_v1:{potion_id or 'MISSING'}")

    reasons.extend(relic_reasons)
    reasons.extend(potion_reasons)
    source_ok = "source_unsupported_v1" not in reasons
    character_ok = "character_unsupported_v1" not in reasons
    room_ok = "room_unsupported_v1" not in reasons
    return PublicRunAdmission(
        relics_allowed=not relic_reasons and source_ok and character_ok and room_ok,
        potions_allowed=not potion_reasons and source_ok and character_ok and room_ok,
        reasons=tuple(sorted(set(reasons))),
    )


__all__ = ["PublicRunAdmission", "assess_public_run_state"]
