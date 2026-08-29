"""Public run-level projection helpers for the STS1 simulator.

Only player-visible run information is read here.  In particular this module
must never inspect GameContext.seed or any RNG object/counter.  The returned
shape intentionally matches the run-level fields consumed by the shared
``SimulatorCombatAdapter`` / real-game public-state contract.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .reconstruction import PUBLIC_RECONSTRUCTION_SCHEMA


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _public_relic(raw: Any) -> dict[str, Any]:
    relic_id = _enum_name(_value(raw, "id"))
    result: dict[str, Any] = {}
    if relic_id is not None:
        result["id"] = relic_id
    name = _value(raw, "name")
    if name is not None:
        result["name"] = str(name)
    counter = _value(raw, "counter", "data")
    if counter is not None:
        result["counter"] = int(counter)
    return result


def _public_potion(raw: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        potion_id = _enum_name(raw)
        return {"index": index, "id": potion_id} if potion_id is not None else {"index": index}

    result: dict[str, Any] = {"index": int(raw.get("index", index))}
    for key in ("id", "name", "requires_target", "can_use", "can_discard", "empty"):
        if key in raw:
            result[key] = raw[key]
    return result


def simulator_public_run_state(game_context: Any) -> dict[str, Any]:
    """Project public GameContext run state without touching seed/RNG surfaces.

    Relic and potion surfaces are marked complete because the pinned native
    binding exposes the whole visible relic container and potion belt.  Card
    instance and enemy reconstruction remain deliberately unproven here, so
    the reconstruction admission gate still fails closed until those surfaces
    receive their own verified adapters.
    """

    relics = [_public_relic(item) for item in _sequence(_value(game_context, "relics", default=[]))]
    potions = [
        _public_potion(item, index=index)
        for index, item in enumerate(_sequence(_value(game_context, "potions", default=[])))
    ]

    character = _enum_name(_value(game_context, "cc", "character"))
    state: dict[str, Any] = {
        "gold": _value(game_context, "gold"),
        "floor": _value(game_context, "floor_num", "floorNum", "floor"),
        "act": _value(game_context, "act"),
        "character": character,
        "ascension_level": _value(game_context, "ascension", "ascension_level"),
        "relics": relics,
        "potions": potions,
        "room": "COMBAT",
        "screen_type": "NONE",
        "reconstruction": {
            "schema_version": PUBLIC_RECONSTRUCTION_SCHEMA,
            "public_card_instance_state_complete": False,
            "public_relic_state_complete": True,
            "public_potion_state_complete": True,
            "public_enemy_state_complete": False,
        },
    }
    return state


__all__ = ["simulator_public_run_state"]
