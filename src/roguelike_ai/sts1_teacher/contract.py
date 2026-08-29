"""Public-state-only decision contract for STS1 Phase 1.

This module deliberately accepts the historical real-game schema name, but
projects every decision onto the frozen source-independent
``sts1-public-state-v1`` contract before Teacher/Search logic sees it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

PUBLIC_STATE_SCHEMA = "sts1-public-state-v1"
LEGACY_REAL_GAME_SCHEMA = "sts1-real-game-public-state-v1"
ACCEPTED_SCHEMA_NAMES = frozenset({PUBLIC_STATE_SCHEMA, LEGACY_REAL_GAME_SCHEMA})

# These names are forbidden anywhere in a formal Teacher input.  The frozen
# contract also forbids future values and raw RNG state; the prefix/substring
# checks below make those families fail closed instead of silently dropping them.
_FORBIDDEN_KEYS = frozenset(
    {
        "seed",
        "uuid",
        "move_id",
        "last_move_id",
        "second_last_move_id",
        "rng_state",
        "rng_counter",
        "random_state",
        "future_draw_order",
        "future_encounter",
        "future_event",
        "future_reward",
        "future_potion",
        "future_relic",
        "future_outcome",
    }
)

_POLICY_FIELDS = (
    "hp",
    "max_hp",
    "block",
    "energy",
    "hand",
    "draw_pile",
    "discard_pile",
    "exhaust_pile",
    "powers",
    "enemies",
    "turn",
    "combat_active",
    "relics",
    "potions",
    "gold",
    "floor",
    "act",
    "character",
    "ascension_level",
    "room",
    "screen_type",
    "screen_choices",
    "rewards",
    "map_choices",
)

_LIST_FIELDS = frozenset(
    {
        "hand",
        "draw_pile",
        "discard_pile",
        "exhaust_pile",
        "powers",
        "enemies",
        "relics",
        "potions",
        "screen_choices",
        "rewards",
        "map_choices",
    }
)

_TRANSPORT_ACTION_KEYS = frozenset({"command", "transport", "raw_action", "source_action"})


class PublicStateContractError(ValueError):
    """Raised when a formal Teacher input violates the frozen public contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _forbidden_key(key: object) -> bool:
    normalized = str(key).strip().lower()
    return (
        normalized in _FORBIDDEN_KEYS
        or "rng" in normalized
        or normalized.startswith("future_")
    )


def _validate_no_hidden(value: Any, *, path: str = "state") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _forbidden_key(key):
                raise PublicStateContractError(f"hidden_information_forbidden:{path}.{key}")
            _validate_no_hidden(item, path=f"{path}.{key}")
        return
    if _is_sequence(value):
        for index, item in enumerate(value):
            _validate_no_hidden(item, path=f"{path}[{index}]")


def _public_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public_copy(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if _is_sequence(value):
        return [_public_copy(item) for item in value]
    return deepcopy(value)


def _normalize_composition(value: Any) -> list[Any]:
    if not _is_sequence(value):
        return []
    return sorted((_public_copy(item) for item in value), key=canonical_json)


def _normalize_state_fields(state: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in _POLICY_FIELDS:
        if field in state:
            value = state[field]
        else:
            value = [] if field in _LIST_FIELDS else None

        if field in {"draw_pile", "discard_pile", "exhaust_pile"}:
            payload[field] = _normalize_composition(value)
        elif field in _LIST_FIELDS:
            payload[field] = [_public_copy(item) for item in value] if _is_sequence(value) else []
        else:
            payload[field] = _public_copy(value)
    return payload


def _policy_action_payload(raw_action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _public_copy(value)
        for key, value in raw_action.items()
        if str(key) not in _TRANSPORT_ACTION_KEYS
    }


def _lookup_position(items: Sequence[Any], *, field: str, value: int) -> Mapping[str, Any] | None:
    for fallback_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        actual = item.get(field, fallback_index)
        if actual == value:
            return item
    return None


def _strip_coordinates(value: Mapping[str, Any], *coordinates: str) -> dict[str, Any]:
    omitted = set(coordinates) | {"position", "index", "is_playable", "can_use", "can_discard"}
    return {str(key): _public_copy(item) for key, item in value.items() if str(key) not in omitted}


def _semantic_action_payload(action: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Return public action meaning without executable-slot identity.

    Two identical public cards in two hand slots are separate executable actions,
    but they are one semantic action for scoring.  Search evaluates that semantic
    group once and copies the score to every executable member, preventing the
    duplicate-action score drift called out by the Phase 1 contract.
    """

    kind = str(action.get("kind", "unknown"))
    semantic: dict[str, Any] = {"kind": kind}

    if kind == "play_card":
        hand_index = action.get("hand_index")
        if isinstance(hand_index, int) and not isinstance(hand_index, bool):
            hand = state.get("hand", [])
            card = _lookup_position(hand if _is_sequence(hand) else [], field="position", value=hand_index)
            if card is not None:
                semantic["card"] = _strip_coordinates(card)
        target_index = action.get("target_index")
        if isinstance(target_index, int) and not isinstance(target_index, bool):
            enemies = state.get("enemies", [])
            enemy = _lookup_position(enemies if _is_sequence(enemies) else [], field="index", value=target_index)
            semantic["target"] = _strip_coordinates(enemy) if enemy is not None else {"index": target_index}
        return semantic

    if kind in {"use_potion", "discard_potion"}:
        potion_index = action.get("potion_index")
        if isinstance(potion_index, int) and not isinstance(potion_index, bool):
            potions = state.get("potions", [])
            potion = _lookup_position(potions if _is_sequence(potions) else [], field="index", value=potion_index)
            if potion is not None:
                semantic["potion"] = _strip_coordinates(potion)
        target_index = action.get("target_index")
        if isinstance(target_index, int) and not isinstance(target_index, bool):
            enemies = state.get("enemies", [])
            enemy = _lookup_position(enemies if _is_sequence(enemies) else [], field="index", value=target_index)
            semantic["target"] = _strip_coordinates(enemy) if enemy is not None else {"index": target_index}
        return semantic

    if kind == "choose":
        if "choice" in action:
            semantic["choice"] = _public_copy(action["choice"])
        else:
            semantic["choice_index"] = action.get("choice_index")
        return semantic

    if kind in {"end_turn", "proceed", "cancel"}:
        return semantic

    # Unknown future public actions fail safe by preserving every policy-visible
    # field.  They are never accidentally collapsed into one semantic group.
    return _public_copy(action)


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    semantic_key: str
    payload: dict[str, Any]
    transport: dict[str, Any]


@dataclass(frozen=True)
class DecisionContext:
    schema_version: str
    source: str | None
    state: dict[str, Any]
    legal_actions: tuple[ActionSpec, ...]
    decision_signature: str

    @classmethod
    def from_public_state(cls, state: Mapping[str, Any]) -> "DecisionContext":
        _validate_no_hidden(state)

        schema = state.get("schema_version")
        if schema not in ACCEPTED_SCHEMA_NAMES:
            raise PublicStateContractError(f"unsupported_public_state_schema:{schema!r}")

        policy_state = _normalize_state_fields(state)
        raw_actions = state.get("legal_actions", [])
        if not _is_sequence(raw_actions):
            raise PublicStateContractError("legal_actions_must_be_sequence")

        actions: list[ActionSpec] = []
        seen_ids: set[str] = set()
        policy_actions: list[dict[str, Any]] = []
        for raw_action in raw_actions:
            if not isinstance(raw_action, Mapping):
                raise PublicStateContractError("legal_action_must_be_mapping")
            payload = _policy_action_payload(raw_action)
            action_id = sha256_json(payload)
            if action_id in seen_ids:
                # Exact duplicate executable actions are a malformed legal-action
                # surface; allowing them would double-count one choice.
                raise PublicStateContractError(f"duplicate_legal_action:{action_id}")
            seen_ids.add(action_id)
            semantic_key = sha256_json(_semantic_action_payload(payload, policy_state))
            transport = {
                str(key): _public_copy(value)
                for key, value in raw_action.items()
                if str(key) in _TRANSPORT_ACTION_KEYS
            }
            actions.append(ActionSpec(action_id, semantic_key, payload, transport))
            policy_actions.append(payload)

        actions.sort(key=lambda item: item.action_id)
        policy_actions.sort(key=canonical_json)
        signature_payload = dict(policy_state)
        signature_payload["legal_actions"] = policy_actions
        signature = sha256_json(signature_payload)
        return cls(
            schema_version=PUBLIC_STATE_SCHEMA,
            source=str(state.get("source")) if state.get("source") is not None else None,
            state=policy_state,
            legal_actions=tuple(actions),
            decision_signature=signature,
        )

    def action_by_id(self, action_id: str) -> ActionSpec:
        for action in self.legal_actions:
            if action.action_id == action_id:
                return action
        raise KeyError(action_id)


__all__ = [
    "ACCEPTED_SCHEMA_NAMES",
    "ActionSpec",
    "DecisionContext",
    "LEGACY_REAL_GAME_SCHEMA",
    "PUBLIC_STATE_SCHEMA",
    "PublicStateContractError",
    "canonical_json",
    "sha256_json",
]
