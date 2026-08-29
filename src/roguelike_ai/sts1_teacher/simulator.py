"""Public combat-state projection for the pinned STS1 simulator bindings.

The adapter is intentionally duck-typed so unit tests do not need the native
``slaythespire`` extension.  At runtime it accepts the BattleContext/SearchAction
surface added by the pinned Jialeiv patch without reading RNG state or hidden
monster move history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contract import DecisionContext, PUBLIC_STATE_SCHEMA


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _value(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _card(card: Any, *, position: int | None = None, playable: bool | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if position is not None:
        result["position"] = position
    card_id = _value(card, "id")
    if card_id is not None:
        result["id"] = _enum_name(card_id)
    name = _value(card, "name")
    if name is not None:
        result["name"] = str(name)
    card_type = _value(card, "type")
    if card_type is not None:
        result["type"] = _enum_name(card_type)
    cost = _value(card, "cost_for_turn", _value(card, "cost"))
    if cost is not None:
        result["cost"] = int(cost)
    upgraded = _value(card, "upgraded")
    if upgraded is not None:
        result["upgrades"] = 1 if bool(upgraded) else 0
    requires_target = _value(card, "requires_target")
    if requires_target is not None:
        result["has_target"] = bool(requires_target)
    if playable is not None:
        result["is_playable"] = playable
    return result


def _player_powers(player: Any) -> list[dict[str, Any]]:
    powers: list[dict[str, Any]] = []
    for attr, public_name in (
        ("strength", "Strength"),
        ("dexterity", "Dexterity"),
        ("focus", "Focus"),
        ("artifact", "Artifact"),
    ):
        amount = _value(player, attr, 0)
        if isinstance(amount, int) and not isinstance(amount, bool) and amount != 0:
            powers.append({"name": public_name, "amount": amount})
    return powers


def _enemy(enemy: Any, *, index: int, battle: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"index": index}
    for attr, public_name in (
        ("name", "name"),
        ("cur_hp", "hp"),
        ("max_hp", "max_hp"),
        ("block", "block"),
    ):
        value = _value(enemy, attr)
        if value is not None:
            result[public_name] = str(value) if public_name == "name" else value

    intent = _value(enemy, "intent")
    if intent is not None:
        result["intent"] = str(intent)
    intent_damage_api = getattr(enemy, "intent_damage", None)
    if callable(intent_damage_api):
        damage_info = intent_damage_api(battle)
        damage = _value(damage_info, "damage")
        attack_count = _value(damage_info, "attack_count")
        if damage is not None:
            result["intent_damage"] = damage
        if attack_count is not None:
            result["intent_hits"] = attack_count
    elif intent_damage_api is not None:
        result["intent_damage"] = intent_damage_api
        intent_hits = _value(enemy, "intent_hits")
        if intent_hits is not None:
            result["intent_hits"] = intent_hits
    alive = _value(enemy, "alive")
    if alive is not None:
        result["is_gone"] = not bool(alive)

    powers: list[dict[str, Any]] = []
    for attr, public_name in (
        ("strength", "Strength"),
        ("vulnerable", "Vulnerable"),
        ("weak", "Weak"),
        ("poison", "Poison"),
    ):
        amount = _value(enemy, attr, 0)
        if isinstance(amount, int) and not isinstance(amount, bool) and amount != 0:
            powers.append({"name": public_name, "amount": amount})
    result["powers"] = powers
    return result


def _action_type(action: Any) -> str:
    return _enum_name(_value(action, "action_type", "UNKNOWN")).upper()


def _legal_actions(actions: Sequence[Any], hand: Sequence[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for action in actions:
        kind = _action_type(action)
        source_idx = _value(action, "source_idx", -1)
        target_idx = _value(action, "target_idx", -1)

        if kind == "CARD":
            item: dict[str, Any] = {"kind": "play_card", "hand_index": int(source_idx) + 1}
            card = hand[int(source_idx)] if isinstance(source_idx, int) and 0 <= source_idx < len(hand) else None
            if card is not None and bool(_value(card, "requires_target", False)):
                item["target_index"] = int(target_idx)
            normalized.append(item)
            continue

        if kind == "POTION":
            if isinstance(target_idx, int) and target_idx > 5:
                normalized.append({"kind": "discard_potion", "potion_index": int(source_idx)})
            else:
                item = {"kind": "use_potion", "potion_index": int(source_idx)}
                if isinstance(target_idx, int) and target_idx >= 0:
                    item["target_index"] = int(target_idx)
                normalized.append(item)
            continue

        if kind == "END_TURN":
            normalized.append({"kind": "end_turn"})
            continue

        if kind in {"SINGLE_CARD_SELECT", "MULTI_CARD_SELECT"}:
            # The select mode is part of the current legal action itself, so it
            # is public information.  Preserve it so the frozen Arm B candidate
            # encoder can reproduce upstream's action-type one-hot without any
            # RNG, seed, future draw, or move-history access.
            item = {"kind": "choose", "choice_index": int(source_idx), "selection_type": kind}
            if isinstance(target_idx, int) and target_idx >= 0:
                item["target_index"] = int(target_idx)
            normalized.append(item)
            continue

        normalized.append(
            {
                "kind": "simulator_public_action",
                "action_type": kind,
                "source_idx": source_idx,
                "target_idx": target_idx,
            }
        )
    return normalized


class SimulatorCombatAdapter:
    """Project a pinned ``BattleContext`` into ``sts1-public-state-v1``."""

    schema_version = PUBLIC_STATE_SCHEMA

    def adapt(
        self,
        battle: Any,
        *,
        legal_actions: Sequence[Any],
        run_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = dict(run_state or {})
        player = _value(battle, "player")
        hand_raw = _sequence(_value(battle, "hand", []))
        action_list = _sequence(legal_actions)

        playable_slots = {
            int(_value(action, "source_idx"))
            for action in action_list
            if _action_type(action) == "CARD"
            and isinstance(_value(action, "source_idx"), int)
        }

        state: dict[str, Any] = {
            "schema_version": PUBLIC_STATE_SCHEMA,
            "source": "simulator",
            "hp": _value(player, "cur_hp"),
            "max_hp": _value(player, "max_hp"),
            "block": _value(player, "block"),
            "energy": _value(player, "energy"),
            "hand": [
                _card(card, position=index + 1, playable=index in playable_slots)
                for index, card in enumerate(hand_raw)
            ],
            "draw_pile": [_card(card) for card in _sequence(_value(battle, "draw_pile", []))],
            "discard_pile": [_card(card) for card in _sequence(_value(battle, "discard_pile", []))],
            "exhaust_pile": [_card(card) for card in _sequence(_value(battle, "exhaust_pile", []))],
            "powers": _player_powers(player),
            "enemies": [
                _enemy(enemy, index=index, battle=battle)
                for index, enemy in enumerate(_sequence(_value(battle, "monsters", [])))
            ],
            "turn": _value(battle, "turn"),
            "combat_active": str(_enum_name(_value(battle, "outcome", "UNDECIDED"))).upper() == "UNDECIDED",
            "relics": list(run.get("relics", [])),
            "potions": list(run.get("potions", [])),
            "gold": run.get("gold"),
            "floor": run.get("floor"),
            "act": run.get("act"),
            "character": run.get("character"),
            "ascension_level": run.get("ascension_level"),
            "room": run.get("room", "COMBAT"),
            "screen_type": run.get("screen_type", "NONE"),
            "screen_choices": list(run.get("screen_choices", [])),
            "rewards": list(run.get("rewards", [])),
            "map_choices": list(run.get("map_choices", [])),
            "legal_actions": _legal_actions(action_list, hand_raw),
        }

        # Constructing DecisionContext is the fail-closed contract check.  It
        # also proves this projection contains no hidden RNG/seed/move-history.
        context = DecisionContext.from_public_state(state)
        state["decision_signature"] = context.decision_signature
        return state


__all__ = ["SimulatorCombatAdapter"]
