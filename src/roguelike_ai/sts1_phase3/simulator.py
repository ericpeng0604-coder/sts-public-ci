"""Pinned sts_lightspeed full-run harness for STS1 Phase 3 A0.

Combat uses the exact frozen Student v0. Non-combat decisions use a tiny
public deterministic fallback. Native action aliases are collapsed only when
we can prove they are the same targetless-card action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

from .frozen_student import (
    FrozenStudentV0,
    canonical_json,
    normalize_action_payload,
    project_policy_observation,
    sha256_json,
)
from .protocol import A0_FROZEN_SEEDS_V1, A0_PROTOCOL_VERSION, frozen_a0_manifest
from .residual_scoring import apply_residual_scores, top1_index


SIMULATOR_EVIDENCE_SCHEMA = "sts1-phase3-a0-simulator-v1"
MAX_GAME_STEPS = 600
MAX_BATTLE_STEPS = 800


class SimulatorRunError(RuntimeError):
    """The pinned simulator could not execute the formal Phase-3 policy safely."""


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


def _value(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def _diagnostic_fallback_legal_choices(
    gc: Any,
    sts: Any,
) -> tuple[list[dict[str, Any]] | None, int | None]:
    """Snapshot the native legal actions selected by the legacy fallback."""

    get_actions = getattr(sts, "get_legal_game_actions", None)
    if not callable(get_actions):
        return None, None
    try:
        actions = list(get_actions(gc))
        if not actions:
            return None, None
        screen_name = _enum_name(_value(gc, "screen_state"))
        choices: list[dict[str, Any]] = []
        for index, action in enumerate(actions):
            fields: dict[str, Any] = {}
            for name in ("bits", "idx1", "idx2", "idx3", "is_potion_action"):
                value = _value(action, name)
                if value is not None:
                    fields[name] = value
            if screen_name in {"REWARDS", "SHOP_ROOM"}:
                reward_type = _value(action, "rewards_action_type")
                if reward_type is not None:
                    fields["rewards_action_type"] = _enum_name(reward_type)
            if not fields:
                return None, None
            choices.append({
                "index": index,
                "descriptor": json.dumps(fields, sort_keys=True, separators=(",", ":")),
                "semantics": {"native_action_type": type(action).__name__, **fields},
                "score": None,
            })

        selected_index = 0
        if screen_name == "REWARDS":
            offered = list(gc.get_card_reward())
            wanted_type = "CARD" if offered else "SKIP"
            selected_index = next(
                (
                    index
                    for index, action in enumerate(actions)
                    if _enum_name(_value(action, "rewards_action_type")) == wanted_type
                    and (
                        wanted_type != "CARD"
                        or (
                            _value(action, "idx1") == 0
                            and _value(action, "idx2") == 0
                        )
                    )
                ),
                -1,
            )
        if not 0 <= selected_index < len(choices):
            return None, None
        return choices, selected_index
    except Exception:
        # Diagnostics must fail closed without changing the fallback policy run.
        return None, None


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


def _public_action(action: Any, hand: Sequence[Any]) -> dict[str, Any]:
    kind = _action_type(action)
    source_idx = _value(action, "source_idx", -1)
    target_idx = _value(action, "target_idx", -1)

    if kind == "CARD":
        item: dict[str, Any] = {"kind": "play_card", "hand_index": int(source_idx) + 1}
        card = hand[int(source_idx)] if isinstance(source_idx, int) and 0 <= source_idx < len(hand) else None
        if card is not None and bool(_value(card, "requires_target", False)):
            item["target_index"] = int(target_idx)
        return item

    if kind == "POTION":
        if isinstance(target_idx, int) and target_idx > 5:
            return {"kind": "discard_potion", "potion_index": int(source_idx)}
        item = {"kind": "use_potion", "potion_index": int(source_idx)}
        if isinstance(target_idx, int) and target_idx >= 0:
            item["target_index"] = int(target_idx)
        return item

    if kind == "END_TURN":
        return {"kind": "end_turn"}

    if kind in {"SINGLE_CARD_SELECT", "MULTI_CARD_SELECT"}:
        item = {"kind": "choose", "choice_index": int(source_idx), "selection_type": kind}
        if isinstance(target_idx, int) and target_idx >= 0:
            item["target_index"] = int(target_idx)
        return item

    return {
        "kind": "simulator_public_action",
        "action_type": kind,
        "source_idx": source_idx,
        "target_idx": target_idx,
    }


def _usable_potion_slots(
    legal_actions: Sequence[Any], hand: Sequence[Any]
) -> set[int]:
    """Return potion slots exposed by the complete current native legal-action list."""

    slots: set[int] = set()
    for action in legal_actions:
        if _action_type(action) != "POTION":
            continue
        public_action = _public_action(action, hand)
        slot = public_action.get("potion_index")
        if (
            public_action.get("kind") == "use_potion"
            and isinstance(slot, int)
            and not isinstance(slot, bool)
        ):
            slots.add(slot)
    return slots


def _strict_integer(value: Any) -> int | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or int(value) != value
    ):
        return None
    return int(value)


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _native_status_flag(entity: Any, sts: Any, enum_name: str, status_name: str) -> bool | None:
    status_enum = getattr(getattr(sts, enum_name, None), status_name, None)
    has_status = getattr(entity, "has_status", None)
    if status_enum is None or not callable(has_status):
        return None
    try:
        value = has_status(status_enum)
    except Exception:
        return None
    return value if isinstance(value, bool) else None


def _project_attack_hp_after_hit_sequence(
    *, player_hp: int, player_block: int, damage_per_hit: int, hit_count: int
) -> int:
    hp = player_hp
    block = player_block
    for _ in range(hit_count):
        saved_block = block
        block = max(0, block - damage_per_hit)
        damage = damage_per_hit - saved_block
        if damage > 0:
            damage -= block
        if damage > 0:
            hp -= damage
        if hp <= 0:
            return hp
    return hp


def _apply_lethal_potion_rescue(
    recommended_action: Any,
    legal_actions: Sequence[Any],
    *,
    hand: Sequence[Any],
    battle: Any,
    game_context: Any,
    sts: Any,
) -> tuple[Any, bool, str]:
    """Use an exact legal potion only when it improves post-attack HP.

    The projection follows the pinned BattleContext damage and potion effects.
    If the baseline attack is lethal, the potion must still prevent death.
    Unsupported damage modifiers or incomplete native state fail closed.
    """

    if _public_action(recommended_action, hand).get("kind") != "end_turn":
        return recommended_action, False, "not_end_turn"

    player = _value(battle, "player", None)
    hp = _strict_integer(_value(player, "cur_hp", None))
    max_hp = _strict_integer(_value(player, "max_hp", None))
    block = _strict_integer(_value(player, "block", None))
    if player is None or hp is None or max_hp is None or block is None or hp <= 0 or max_hp <= 0 or block < 0:
        return recommended_action, False, "incomplete_player_state"

    try:
        monsters = list(_value(battle, "monsters", None))
    except (TypeError, ValueError):
        return recommended_action, False, "incomplete_monster_state"
    living: list[Any] = []
    for monster in monsters:
        monster_hp = _strict_integer(_value(monster, "cur_hp", None))
        alive = _value(monster, "alive", None)
        if monster_hp is None or (alive is not None and not isinstance(alive, bool)):
            return recommended_action, False, "incomplete_monster_state"
        if monster_hp > 0 and alive is not False:
            living.append(monster)
    if len(living) != 1:
        return recommended_action, False, "not_exactly_one_living_enemy"

    enemy = living[0]
    enemy_poison = _strict_integer(_value(enemy, "poison", None))
    enemy_strength = _strict_integer(_value(enemy, "strength", None))
    enemy_weak = _strict_integer(_value(enemy, "weak", None))
    if enemy_poison is None or enemy_strength is None or enemy_weak is None:
        return recommended_action, False, "incomplete_enemy_modifiers"
    if enemy_poison > 0:
        return recommended_action, False, "enemy_may_die_before_attack"

    intent_damage = getattr(enemy, "intent_damage", None)
    if not callable(intent_damage):
        return recommended_action, False, "unknown_attack_intent"
    try:
        damage_info = intent_damage(battle)
    except Exception:
        return recommended_action, False, "unknown_attack_intent"
    base_damage = _strict_integer(_value(damage_info, "damage", None))
    hit_count = _strict_integer(_value(damage_info, "attack_count", None))
    if base_damage is None or hit_count is None or base_damage <= 0 or hit_count <= 0:
        return recommended_action, False, "unknown_attack_intent"

    player_vulnerable = _native_status_flag(player, sts, "PlayerStatus", "VULNERABLE")
    player_intangible = _native_status_flag(player, sts, "PlayerStatus", "INTANGIBLE")
    if player_vulnerable is None or player_intangible is None:
        return recommended_action, False, "incomplete_player_modifiers"

    # These end-turn effects can change the attack outcome or invalidate a
    # lethal classification, and are not represented in this narrow projection.
    for status_name in ("REGEN", "METALLICIZE", "PLATED_ARMOR"):
        active = _native_status_flag(player, sts, "PlayerStatus", status_name)
        if active is None:
            return recommended_action, False, "incomplete_player_modifiers"
        if active:
            return recommended_action, False, "unsupported_end_turn_status"

    relic_values = _value(game_context, "relics", None)
    if relic_values is None:
        return recommended_action, False, "incomplete_relic_inventory"
    try:
        relics = list(relic_values)
    except (TypeError, ValueError):
        return recommended_action, False, "incomplete_relic_inventory"
    relic_names: set[str] = set()
    for relic in relics:
        relic_id = _value(relic, "id", None)
        if relic_id is None:
            return recommended_action, False, "incomplete_relic_inventory"
        relic_names.add(_enum_name(relic_id).upper())
    if relic_names & {
        "FOSSILIZED_HELIX",
        "LIZARD_TAIL",
        "ORICHALCUM",
        "STONE_CALENDAR",
        "TORII",
        "TUNGSTEN_ROD",
    }:
        return recommended_action, False, "unsupported_damage_or_end_turn_relic"

    potion_values = _value(game_context, "potions", None)
    if potion_values is None:
        return recommended_action, False, "incomplete_potion_inventory"
    try:
        potions = list(potion_values)
    except (TypeError, ValueError):
        return recommended_action, False, "incomplete_potion_inventory"
    if len(potions) != 5 or any(
        not isinstance(potion, str)
        or not potion.strip()
        or potion.upper() in {"INVALID", "UNKNOWN"}
        for potion in potions
    ):
        return recommended_action, False, "incomplete_potion_inventory"
    if any("FAIRY" in potion.upper() for potion in potions):
        return recommended_action, False, "automatic_death_prevention_potion_present"

    # Match the pinned Monster::calculateDamageToPlayer float32 operations.
    damage_value = _float32(float(base_damage + enemy_strength))
    if enemy_weak > 0:
        weak_multiplier = 0.6 if "PAPER_KRANE" in relic_names else 0.75
        damage_value = _float32(damage_value * _float32(weak_multiplier))
    if player_vulnerable:
        vulnerable_multiplier = 1.25 if "ODD_MUSHROOM" in relic_names else 1.5
        damage_value = _float32(damage_value * _float32(vulnerable_multiplier))
    if player_intangible:
        damage_value = min(damage_value, 1.0)
    damage_per_hit = max(int(math.floor(damage_value)), 0)
    if damage_per_hit <= 0:
        return recommended_action, False, "attack_not_damaging"

    parent_hp_after = _project_attack_hp_after_hit_sequence(
        player_hp=hp,
        player_block=block,
        damage_per_hit=damage_per_hit,
        hit_count=hit_count,
    )
    bark = "SACRED_BARK" in relic_names
    rescue_actions: list[tuple[int, Any]] = []
    for action in legal_actions:
        if _action_type(action) != "POTION":
            continue
        public_action = _public_action(action, hand)
        if public_action.get("kind") != "use_potion":
            continue
        slot = public_action.get("potion_index")
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < len(potions):
            continue
        potion_name = potions[slot].upper()
        new_hp = hp
        new_block = block
        if potion_name == "BLOCK_POTION":
            new_block += 24 if bark else 12
        elif potion_name == "BLOOD_POTION":
            heal_percent = 20 if bark else 40
            heal_amount = int(math.floor(_float32(_float32(float(max_hp * heal_percent)) / _float32(100.0))))
            if "MARK_OF_THE_BLOOM" in relic_names:
                heal_amount = 0
            elif "MAGIC_FLOWER" in relic_names:
                heal_amount = heal_amount * 3 // 2
            new_hp = min(max_hp, hp + heal_amount)
        else:
            continue
        projected_hp = _project_attack_hp_after_hit_sequence(
            player_hp=new_hp,
            player_block=new_block,
            damage_per_hit=damage_per_hit,
            hit_count=hit_count,
        )
        if projected_hp > 0 and (
            parent_hp_after <= 0 or projected_hp > parent_hp_after
        ):
            rescue_actions.append((slot, action))

    if not rescue_actions:
        return recommended_action, False, "no_legal_potion_improves_projected_hp"
    _, selected = min(rescue_actions, key=lambda item: item[0])
    return selected, True, "override"


def _apply_low_hp_emergency_potion(
    recommended_action: Any,
    legal_actions: Sequence[Any],
    *,
    hand: Sequence[Any],
    player_hp: int | None,
    player_max_hp: int | None,
    hp_ratio_threshold: float,
) -> tuple[Any, bool]:
    """Use the first deterministically ordered legal potion below a fixed HP ratio.

    Missing or invalid HP data, an invalid threshold, or a missing legal potion
    action leaves the MCTS recommendation unchanged.
    """

    if (
        not isinstance(hp_ratio_threshold, (int, float))
        or isinstance(hp_ratio_threshold, bool)
        or not math.isfinite(float(hp_ratio_threshold))
        or not 0.0 <= float(hp_ratio_threshold) <= 1.0
        or not isinstance(player_hp, int)
        or isinstance(player_hp, bool)
        or not isinstance(player_max_hp, int)
        or isinstance(player_max_hp, bool)
        or player_hp <= 0
        or player_max_hp <= 0
        or player_hp / player_max_hp > float(hp_ratio_threshold)
    ):
        return recommended_action, False

    if (
        _action_type(recommended_action) == "POTION"
        and _public_action(recommended_action, hand).get("kind") == "use_potion"
    ):
        return recommended_action, False

    usable_slots = _usable_potion_slots(legal_actions, hand)
    if not usable_slots:
        return recommended_action, False

    target_slot = min(usable_slots)
    for action in legal_actions:
        public_action = _public_action(action, hand)
        if (
            _action_type(action) == "POTION"
            and public_action.get("kind") == "use_potion"
            and public_action.get("potion_index") == target_slot
        ):
            return action, True
    return recommended_action, False


def _apply_last_potion_reserve(
    recommended_action: Any,
    legal_actions: Sequence[Any],
    *,
    hand: Sequence[Any],
    floor: int,
    reserve_until_floor: int,
    player_hp: int | None,
    player_block: int | None,
    incoming_damage: int | None,
) -> tuple[Any, bool]:
    """Keep the final usable potion before a floor threshold unless the visible
    current intent is lethal. If MCTS recommends that potion, use a deterministic
    legal non-potion action when one exists; do not replace the potion with an
    empty end turn.
    """

    if (
        floor >= reserve_until_floor
        or _action_type(recommended_action) != "POTION"
        or _public_action(recommended_action, hand).get("kind") != "use_potion"
        or len(_usable_potion_slots(legal_actions, hand)) != 1
        or player_hp is None
        or player_block is None
        or incoming_damage is None
        or incoming_damage >= player_hp + player_block
    ):
        return recommended_action, False

    for preferred_type in ("CARD", "SINGLE_CARD_SELECT", "MULTI_CARD_SELECT"):
        for action in legal_actions:
            if _action_type(action) == preferred_type:
                return action, True
    return recommended_action, False


def _safe_targetless_card_alias(first: Any, second: Any, hand: Sequence[Any]) -> bool:
    """Accept only the pinned simulator's proven target-index alias for targetless cards."""

    if _action_type(first) != "CARD" or _action_type(second) != "CARD":
        return False
    first_source = _value(first, "source_idx", -1)
    second_source = _value(second, "source_idx", -1)
    if not isinstance(first_source, int) or first_source != second_source:
        return False
    if first_source < 0 or first_source >= len(hand):
        return False
    card = hand[first_source]
    requires_target = _value(card, "requires_target", None)
    return isinstance(requires_target, bool) and requires_target is False


def _project_legal_actions(
    actions: Sequence[Any],
    hand: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Project native actions and preserve an exact public->native index map.

    The only permitted collapse is a duplicate public action for the same
    targetless card. That alias was verified on the pinned simulator by cloning
    the same BattleContext and observing identical post-action public states.
    """

    public_actions: list[dict[str, Any]] = []
    native_indices: list[int] = []
    first_by_identity: dict[str, tuple[int, Any]] = {}
    alias_count = 0

    for native_index, action in enumerate(actions):
        projected = _public_action(action, hand)
        identity = canonical_json(projected)
        prior = first_by_identity.get(identity)
        if prior is not None:
            _, prior_action = prior
            if not _safe_targetless_card_alias(prior_action, action, hand):
                raise SimulatorRunError(
                    "duplicate projected legal action without proven targetless-card equivalence"
                )
            alias_count += 1
            continue

        first_by_identity[identity] = (native_index, action)
        public_actions.append(projected)
        native_indices.append(native_index)

    if not public_actions and actions:
        raise SimulatorRunError("native legal actions projected to an empty public action set")
    return public_actions, native_indices, alias_count


def _legal_actions(actions: Sequence[Any], hand: Sequence[Any]) -> list[dict[str, Any]]:
    """Compatibility helper returning only public actions."""

    public_actions, _, _ = _project_legal_actions(actions, hand)
    return public_actions


def _hybrid_vote_choice_bits(
    mcts_votes: Sequence[tuple[int, int]],
    *,
    student_bits: int,
) -> tuple[int, bool]:
    """Choose among MCTS recommendations, using Student only to break MCTS ties."""
    if len(mcts_votes) < 2:
        raise SimulatorRunError("hybrid MCTS requires at least two MCTS budgets")
    counts: dict[int, int] = {}
    for budget, bits in mcts_votes:
        if budget < 1:
            raise SimulatorRunError("hybrid MCTS budget must be positive")
        counts[bits] = counts.get(bits, 0) + 1
    best_count = max(counts.values())
    tied = {bits for bits, count in counts.items() if count == best_count}
    if len(tied) == 1:
        return next(iter(tied)), False
    if student_bits in tied:
        return student_bits, True
    # Student chose outside the MCTS tie: prefer the highest-budget tied MCTS vote.
    for budget, bits in sorted(mcts_votes, reverse=True):
        if bits in tied:
            return bits, False
    raise SimulatorRunError("hybrid MCTS could not resolve tied recommendations")


class SimulatorCombatAdapter:
    """Project a pinned BattleContext into the frozen Student public contract."""

    def adapt(
        self,
        battle: Any,
        *,
        legal_actions: Sequence[Any],
        run_state: Mapping[str, Any] | None = None,
        projected_legal_actions: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        run = dict(run_state or {})
        player = _value(battle, "player")
        hand_raw = _sequence(_value(battle, "hand", []))
        action_list = _sequence(legal_actions)
        playable_slots = {
            int(_value(action, "source_idx"))
            for action in action_list
            if _action_type(action) == "CARD" and isinstance(_value(action, "source_idx"), int)
        }
        if projected_legal_actions is None:
            public_actions, _, _ = _project_legal_actions(action_list, hand_raw)
        else:
            public_actions = [dict(item) for item in projected_legal_actions]

        state: dict[str, Any] = {
            "schema_version": "sts1-public-state-v1",
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
            "combat_active": _enum_name(_value(battle, "outcome", "UNDECIDED")).upper() == "UNDECIDED",
            "relics": list(run.get("relics", [])),
            "potions": list(run.get("potions") or []),
            "gold": run.get("gold"),
            "floor": run.get("floor"),
            "act": run.get("act"),
            "character": run.get("character", "IRONCLAD"),
            "ascension_level": run.get("ascension_level", 0),
            "room": run.get("room", "COMBAT"),
            "screen_type": run.get("screen_type", "NONE"),
            "screen_choices": list(run.get("screen_choices", [])),
            "rewards": list(run.get("rewards", [])),
            "map_choices": list(run.get("map_choices", [])),
            "legal_actions": public_actions,
        }
        projected = project_policy_observation(state)
        normalized_actions = [normalize_action_payload(item) for item in state["legal_actions"]]
        signature_payload = dict(projected)
        signature_payload["legal_actions"] = sorted(normalized_actions, key=canonical_json)
        state["decision_signature"] = sha256_json(signature_payload)
        return state


def public_run_state(gc: Any) -> dict[str, Any]:
    floor = _value(gc, "floor_num")
    act = _value(gc, "act")
    gold = _value(gc, "gold")
    return {
        "floor": int(floor) if isinstance(floor, int) and not isinstance(floor, bool) else floor,
        "act": int(act) if isinstance(act, int) and not isinstance(act, bool) else act,
        "gold": int(gold) if isinstance(gold, int) and not isinstance(gold, bool) else gold,
        "character": "IRONCLAD",
        "ascension_level": 0,
        "room": "COMBAT",
        "screen_type": "NONE",
    }


def _diagnostic_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    identifier = _value(value, "id")
    if identifier is not None:
        return _enum_name(identifier)
    name = _value(value, "name")
    if name is not None:
        return str(name)
    return str(value)


def _potion_inventory_snapshot(gc: Any) -> dict[str, Any]:
    """Return all five native potion slots, failing closed on incomplete data."""

    inventory = _value(gc, "potions", None)
    if inventory is None:
        return {
            "potions": None,
            "potion_inventory_complete": False,
            "potion_inventory_source": "unavailable",
            "potion_inventory_reason": "native_potion_property_missing",
        }
    try:
        slots = list(inventory)
    except (TypeError, ValueError):
        slots = []
    if len(slots) != 5:
        return {
            "potions": None,
            "potion_inventory_complete": False,
            "potion_inventory_source": "native_gamecontext_potion_enum_names_v1",
            "potion_inventory_reason": "expected_exactly_five_slots",
        }
    if any(
        not isinstance(slot, str)
        or not slot.strip()
        or slot.upper() in {"INVALID", "UNKNOWN"}
        for slot in slots
    ):
        return {
            "potions": None,
            "potion_inventory_complete": False,
            "potion_inventory_source": "native_gamecontext_potion_enum_names_v1",
            "potion_inventory_reason": "unknown_or_invalid_native_slot",
        }
    return {
        "potions": slots,
        "potion_inventory_complete": True,
        "potion_inventory_source": "native_gamecontext_potion_enum_names_v1",
        "potion_inventory_reason": "exactly_five_valid_native_slots",
    }


def _diagnostic_run_snapshot(gc: Any, armg_policy: Any = None) -> dict[str, Any]:
    """Capture current run resources without changing simulator state."""

    state = public_run_state(gc)
    screen = _enum_name(_value(gc, "screen_state"))
    potion_inventory = _potion_inventory_snapshot(gc)
    state.update({
        "screen_state": screen,
        "room": screen,
        "map_node": {
            "x": _value(gc, "cur_map_node_x"),
            "y": _value(gc, "cur_map_node_y"),
        },
        "hp": _value(gc, "cur_hp"),
        "max_hp": _value(gc, "max_hp"),
        "deck": (
            armg_policy.deck_snapshot(gc)
            if callable(getattr(armg_policy, "deck_snapshot", None))
            else [_card(card) for card in _sequence(_value(gc, "deck", []))]
        ),
        "relics": [_diagnostic_label(item) for item in _sequence(_value(gc, "relics", []))],
        **potion_inventory,
    })
    return state


def _diagnostic_battle_snapshot(battle: Any, gc: Any, armg_policy: Any = None) -> dict[str, Any]:
    player = _value(battle, "player")
    run_state = _diagnostic_run_snapshot(gc, armg_policy)
    run_state["room"] = "COMBAT"
    return {
        "run": run_state,
        "turn": _value(battle, "turn"),
        "outcome": _enum_name(_value(battle, "outcome", "UNKNOWN")),
        "player": {
            "hp": _value(player, "cur_hp"),
            "max_hp": _value(player, "max_hp"),
            "block": _value(player, "block"),
            "energy": _value(player, "energy"),
            "powers": _player_powers(player),
        },
        "hand": [_card(card, position=index + 1) for index, card in enumerate(_sequence(_value(battle, "hand", [])))],
        "draw_pile": [_card(card) for card in _sequence(_value(battle, "draw_pile", []))],
        "discard_pile": [_card(card) for card in _sequence(_value(battle, "discard_pile", []))],
        "exhaust_pile": [_card(card) for card in _sequence(_value(battle, "exhaust_pile", []))],
        "enemies": [
            _enemy(enemy, index=index, battle=battle)
            for index, enemy in enumerate(_sequence(_value(battle, "monsters", [])))
        ],
    }


def _diagnostic_map_route(
    gc: Any,
    sts: Any,
    *,
    choice_count: int,
    selected_index: int,
) -> dict[str, Any]:
    """Identify each current legal map edge without reading beyond the next node."""

    source_x = _value(gc, "cur_map_node_x")
    source_y = _value(gc, "cur_map_node_y")
    trace: dict[str, Any] = {
        "source_node": {"x": source_x, "y": source_y},
        "choice_count": choice_count,
        "choices_complete": False,
        "choices": [],
        "selected_route": None,
    }
    get_actions = getattr(sts, "get_legal_game_actions", None)
    if not callable(get_actions):
        trace["incomplete_reason"] = "legal_game_actions_unavailable"
        return trace
    if (
        not isinstance(source_x, int)
        or isinstance(source_x, bool)
        or not isinstance(source_y, int)
        or isinstance(source_y, bool)
    ):
        trace["incomplete_reason"] = "current_map_node_unavailable"
        return trace

    try:
        actions = list(get_actions(gc))
        room_at = getattr(gc, "map_node_room", None)
        choices = []
        for index, action in enumerate(actions):
            target_x = _value(action, "idx1")
            target_y = source_y + 1
            room = room_at(target_x, target_y) if callable(room_at) and isinstance(target_x, int) else None
            choices.append({
                "legal_action_index": index,
                "action_bits": _value(action, "bits"),
                "target_node": {"x": target_x, "y": target_y},
                "target_room": _enum_name(room) if room is not None else None,
            })
        trace["choices"] = choices
        targets = [
            (item["target_node"]["x"], item["target_node"]["y"])
            for item in choices
        ]
        trace["choices_complete"] = (
            len(actions) == choice_count
            and all(isinstance(x, int) and not isinstance(x, bool) for x, _ in targets)
            and len(targets) == len(set(targets))
        )
        if trace["choices_complete"] and 0 <= selected_index < len(choices):
            trace["selected_route"] = {
                "from": trace["source_node"],
                "to": choices[selected_index]["target_node"],
                "room": choices[selected_index]["target_room"],
                "legal_action_index": selected_index,
            }
        elif not trace["choices_complete"]:
            trace["incomplete_reason"] = "legal_action_choice_count_or_target_mismatch"
        else:
            trace["incomplete_reason"] = "selected_map_choice_unavailable"
    except Exception as exc:
        trace["incomplete_reason"] = f"map_route_read_failed:{type(exc).__name__}"
    return trace


def _apply_low_hp_elite_route_avoidance(
    recommended_index: int,
    scores: Sequence[Any],
    route: Mapping[str, Any] | None,
    *,
    current_hp: Any,
    max_hp: Any,
    known_room_names: set[str],
) -> tuple[int, dict[str, Any]]:
    """Avoid an immediate Elite below half HP, using only current legal choices.

    The fallback is fail-closed: incomplete route, room, HP, or score data leaves
    the original ArmG recommendation untouched.
    """

    detail: dict[str, Any] = {
        "kind": "avoid_elite_below_half_hp",
        "threshold_ratio": 0.5,
        "hp_before": current_hp if isinstance(current_hp, (int, float)) and not isinstance(current_hp, bool) else None,
        "max_hp_before": max_hp if isinstance(max_hp, (int, float)) and not isinstance(max_hp, bool) else None,
        "hp_ratio": None,
        "recommended_index": recommended_index,
        "actual_index": recommended_index,
        "recommended_room": None,
        "actual_room": None,
        "recommended_score": None,
        "actual_score": None,
        "status": "unchanged",
        "reason": None,
        "overridden": False,
    }

    def fail_closed(reason: str) -> tuple[int, dict[str, Any]]:
        detail["status"] = "fail_closed"
        detail["reason"] = reason
        return recommended_index, detail

    if (
        not isinstance(current_hp, (int, float))
        or isinstance(current_hp, bool)
        or not math.isfinite(float(current_hp))
        or not isinstance(max_hp, (int, float))
        or isinstance(max_hp, bool)
        or not math.isfinite(float(max_hp))
        or float(current_hp) < 0
        or float(max_hp) <= 0
    ):
        return fail_closed("player_hp_unavailable_or_invalid")

    hp_ratio = float(current_hp) / float(max_hp)
    detail["hp_ratio"] = hp_ratio
    if hp_ratio >= 0.5:
        detail["reason"] = "hp_not_below_half"
        return recommended_index, detail

    if not isinstance(route, Mapping) or route.get("choices_complete") is not True:
        return fail_closed("map_route_incomplete")
    choices = route.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or not choices
        or len(choices) != route.get("choice_count")
        or not isinstance(recommended_index, int)
        or isinstance(recommended_index, bool)
        or not 0 <= recommended_index < len(choices)
    ):
        return fail_closed("map_choice_count_or_index_invalid")
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)) or len(scores) != len(choices):
        return fail_closed("armg_scores_incomplete")

    normalized_known_rooms = {
        value.upper()
        for value in known_room_names
        if isinstance(value, str) and value
    }
    if not normalized_known_rooms or "ELITE" not in normalized_known_rooms:
        return fail_closed("known_room_vocabulary_unavailable")

    room_names: list[str] = []
    normalized_scores: list[float] = []
    for index, (choice, score) in enumerate(zip(choices, scores, strict=True)):
        if not isinstance(choice, Mapping) or choice.get("legal_action_index") != index:
            return fail_closed("map_legal_action_order_mismatch")
        room = choice.get("target_room")
        target = choice.get("target_node")
        if (
            not isinstance(room, str)
            or not room
            or room.upper() not in normalized_known_rooms
            or not isinstance(target, Mapping)
            or not isinstance(target.get("x"), int)
            or isinstance(target.get("x"), bool)
            or not isinstance(target.get("y"), int)
            or isinstance(target.get("y"), bool)
        ):
            return fail_closed("map_room_or_target_unknown")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return fail_closed("armg_score_invalid")
        score_value = float(score)
        if not math.isfinite(score_value):
            return fail_closed("armg_score_non_finite")
        room_names.append(room.upper())
        normalized_scores.append(score_value)

    detail["recommended_room"] = room_names[recommended_index]
    detail["recommended_score"] = normalized_scores[recommended_index]
    if room_names[recommended_index] != "ELITE":
        detail["actual_room"] = room_names[recommended_index]
        detail["actual_score"] = normalized_scores[recommended_index]
        detail["reason"] = "recommended_route_not_elite"
        return recommended_index, detail

    non_elite_indices = [index for index, room in enumerate(room_names) if room != "ELITE"]
    if not non_elite_indices:
        return fail_closed("no_known_non_elite_route")

    # Highest ArmG score wins; a tie deterministically chooses the lowest index.
    actual_index = max(non_elite_indices, key=lambda index: (normalized_scores[index], -index))
    detail.update(
        {
            "actual_index": actual_index,
            "actual_room": room_names[actual_index],
            "actual_score": normalized_scores[actual_index],
            "status": "overridden",
            "reason": "low_hp_elite_replaced_by_highest_scored_non_elite",
            "overridden": True,
        }
    )
    return actual_index, detail


def _apply_campfire_overheal_override(
    recommended_index: int,
    kind: str,
    descs: Sequence[Any],
    gc: Any,
    sts: Any,
    armg_policy: Any,
    *,
    require_overheal_exceeds_effective_rest_heal: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Choose the legal Smith action under the selected fixed over-heal rule.

    The rule uses the pinned simulator's REST (option 0) and SMITH (option 1)
    semantics. Missing state, relic, or descriptor data keeps the ArmG action.
    """

    detail: dict[str, Any] = {
        "kind": (
            "prefer_smith_when_overheal_exceeds_effective_rest_heal"
            if require_overheal_exceeds_effective_rest_heal
            else "prefer_smith_when_rest_overheals"
        ),
        "recommended_index": recommended_index,
        "selected_index": recommended_index,
        "recommended_option_index": None,
        "rest_heal_amount": None,
        "overheal_amount": None,
        "effective_rest_heal_amount": None,
        "overridden": False,
        "reason": "unchanged",
    }
    if kind != "rest":
        detail["reason"] = "not_campfire"
        return recommended_index, detail

    describe = getattr(armg_policy, "describe_choice", None)
    if (
        not callable(describe)
        or not isinstance(recommended_index, int)
        or isinstance(recommended_index, bool)
        or not 0 <= recommended_index < len(descs)
    ):
        detail["reason"] = "choice_semantics_unavailable"
        return recommended_index, detail

    semantics: list[dict[str, Any]] = []
    for desc in descs:
        try:
            value = describe(kind, desc)
        except Exception:
            detail["reason"] = "choice_semantics_unavailable"
            return recommended_index, detail
        option_index = value.get("option_index") if isinstance(value, Mapping) else None
        if not isinstance(option_index, int) or isinstance(option_index, bool):
            detail["reason"] = "choice_semantics_incomplete"
            return recommended_index, detail
        semantics.append(dict(value))

    recommended_option = semantics[recommended_index]["option_index"]
    detail["recommended_option_index"] = recommended_option
    rest_indices = [i for i, item in enumerate(semantics) if item["option_index"] == 0]
    smith_indices = [i for i, item in enumerate(semantics) if item["option_index"] == 1]
    if len(rest_indices) != 1 or len(smith_indices) != 1:
        detail["reason"] = "rest_or_smith_not_uniquely_legal"
        return recommended_index, detail
    if recommended_option != 0:
        detail["reason"] = "parent_did_not_recommend_rest"
        return recommended_index, detail

    current_hp = _value(gc, "cur_hp")
    max_hp = _value(gc, "max_hp")
    if (
        not isinstance(current_hp, (int, float))
        or isinstance(current_hp, bool)
        or not math.isfinite(float(current_hp))
        or not isinstance(max_hp, (int, float))
        or isinstance(max_hp, bool)
        or not math.isfinite(float(max_hp))
        or current_hp < 0
        or current_hp > max_hp
        or max_hp <= 0
    ):
        detail["reason"] = "player_hp_unavailable_or_invalid"
        return recommended_index, detail

    has_relic = getattr(gc, "hasRelic", None)
    try:
        if callable(has_relic):
            has_pillow = bool(has_relic(sts.RelicId.REGAL_PILLOW))
        else:
            relics = getattr(gc, "relics", None)
            if relics is None:
                detail["reason"] = "relic_state_unavailable"
                return recommended_index, detail
            has_pillow = any(
                _enum_name(_value(relic, "id", relic)).upper().endswith("REGAL_PILLOW")
                for relic in _sequence(relics)
            )
    except Exception:
        detail["reason"] = "relic_state_unavailable"
        return recommended_index, detail

    # Native fractionMaxHp(0.30f) uses std::round for positive max HP.
    heal_amount = int(math.floor(float(max_hp) * 0.30 + 0.5)) + (15 if has_pillow else 0)
    detail["rest_heal_amount"] = heal_amount
    detail["hp_before"] = int(current_hp) if isinstance(current_hp, int) else float(current_hp)
    detail["max_hp_before"] = int(max_hp) if isinstance(max_hp, int) else float(max_hp)
    overheal_amount = max(0.0, float(current_hp) + heal_amount - float(max_hp))
    effective_rest_heal = min(float(heal_amount), float(max_hp) - float(current_hp))
    detail["overheal_amount"] = (
        int(overheal_amount) if overheal_amount.is_integer() else overheal_amount
    )
    detail["effective_rest_heal_amount"] = (
        int(effective_rest_heal)
        if effective_rest_heal.is_integer()
        else effective_rest_heal
    )
    if current_hp + heal_amount <= max_hp:
        detail["reason"] = "rest_heal_does_not_overflow"
        return recommended_index, detail
    if require_overheal_exceeds_effective_rest_heal and overheal_amount <= effective_rest_heal:
        detail["reason"] = "overheal_not_greater_than_effective_rest_heal"
        return recommended_index, detail

    smith_index = smith_indices[0]
    detail.update({
        "selected_index": smith_index,
        "overridden": True,
        "reason": (
            "overheal_exceeds_effective_rest_heal"
            if require_overheal_exceeds_effective_rest_heal
            else "rest_heal_would_exceed_max_hp"
        ),
    })
    return smith_index, detail


def _apply_card_reward_skip_deck_threshold(
    recommended_index: int,
    kind: str,
    descs: Sequence[Any],
    gc: Any,
    armg_policy: Any,
    *,
    deck_size_threshold: int | None,
    require_recommended_duplicate: bool = False,
    require_all_options_duplicate: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Select the exact legal Skip descriptor for a sufficiently large deck."""

    detail: dict[str, Any] = {
        "kind": (
            "skip_duplicate_card_reward_when_deck_size_at_least"
            if require_recommended_duplicate
            else "skip_card_reward_when_deck_size_at_least"
        ),
        "threshold": deck_size_threshold,
        "deck_size": None,
        "require_recommended_duplicate": require_recommended_duplicate,
        "recommended_card_is_duplicate": None,
        "require_all_options_duplicate": require_all_options_duplicate,
        "all_reward_options_are_duplicates": None,
        "legal_reward_card_count": None,
        "duplicate_reward_card_count": None,
        "novel_reward_card_count": None,
        "skip_choice_index": None,
        "eligible": False,
        "overridden": False,
        "reason": "disabled" if deck_size_threshold is None else "not_card_reward",
    }
    if deck_size_threshold is None or kind != "card":
        return recommended_index, detail
    if (
        not isinstance(recommended_index, int)
        or isinstance(recommended_index, bool)
        or not 0 <= recommended_index < len(descs)
    ):
        detail["reason"] = "invalid_parent_recommendation"
        return recommended_index, detail

    deck_value = _value(gc, "deck", None)
    if deck_value is None:
        detail["reason"] = "incomplete_deck"
        return recommended_index, detail
    try:
        deck_size = len(_sequence(deck_value))
    except Exception:
        detail["reason"] = "incomplete_deck"
        return recommended_index, detail
    detail["deck_size"] = deck_size
    if deck_size < deck_size_threshold:
        detail["reason"] = "deck_below_threshold"
        return recommended_index, detail

    if require_recommended_duplicate:
        try:
            recommended = armg_policy.describe_choice(kind, descs[recommended_index])
            if isinstance(recommended, Mapping) and recommended.get("choice") == "skip":
                detail["reason"] = "parent_already_selected_skip"
                return recommended_index, detail
            recommended_name = recommended.get("card_name") if isinstance(recommended, Mapping) else None
            deck_snapshot = armg_policy.deck_snapshot(gc)
            if (
                not isinstance(recommended_name, str)
                or not recommended_name
                or not isinstance(deck_snapshot, list)
                or any(
                    not isinstance(card, Mapping) or not isinstance(card.get("name"), str)
                    for card in deck_snapshot
                )
            ):
                detail["reason"] = "incomplete_recommended_card_identity"
                return recommended_index, detail
            is_duplicate = any(card["name"] == recommended_name for card in deck_snapshot)
        except Exception:
            detail["reason"] = "incomplete_recommended_card_identity"
            return recommended_index, detail
        detail["recommended_card_is_duplicate"] = is_duplicate
        if not is_duplicate:
            detail["reason"] = "recommended_card_not_duplicate"
            return recommended_index, detail

    if require_all_options_duplicate:
        try:
            deck_snapshot = armg_policy.deck_snapshot(gc)
            if (
                not isinstance(deck_snapshot, list)
                or any(
                    not isinstance(card, Mapping)
                    or not isinstance(card.get("name"), str)
                    or not card.get("name")
                    for card in deck_snapshot
                )
            ):
                detail["reason"] = "incomplete_deck_card_identity"
                return recommended_index, detail
            deck_names = {card["name"] for card in deck_snapshot}
            reward_names: list[str] = []
            for descriptor in descs:
                semantics = armg_policy.describe_choice(kind, descriptor)
                if not isinstance(semantics, Mapping):
                    detail["reason"] = "incomplete_reward_card_identity"
                    return recommended_index, detail
                if semantics.get("choice") == "skip":
                    continue
                card_name = semantics.get("card_name")
                if not isinstance(card_name, str) or not card_name:
                    detail["reason"] = "incomplete_reward_card_identity"
                    return recommended_index, detail
                reward_names.append(card_name)
        except Exception:
            detail["reason"] = "incomplete_reward_card_identity"
            return recommended_index, detail

        if not reward_names:
            detail["reason"] = "no_legal_reward_card_options"
            return recommended_index, detail
        duplicate_count = sum(name in deck_names for name in reward_names)
        novel_count = len(reward_names) - duplicate_count
        all_duplicate = novel_count == 0
        detail.update({
            "legal_reward_card_count": len(reward_names),
            "duplicate_reward_card_count": duplicate_count,
            "novel_reward_card_count": novel_count,
            "all_reward_options_are_duplicates": all_duplicate,
        })
        if not all_duplicate:
            detail["reason"] = "legal_novel_reward_option"
            return recommended_index, detail

    try:
        skip_indices = []
        for index, descriptor in enumerate(descs):
            semantics = armg_policy.describe_choice(kind, descriptor)
            if isinstance(semantics, Mapping) and semantics.get("choice") == "skip":
                skip_indices.append(index)
    except Exception:
        detail["reason"] = "incomplete_choice_semantics"
        return recommended_index, detail
    if len(skip_indices) != 1:
        detail["reason"] = "skip_choice_missing_or_ambiguous"
        return recommended_index, detail

    skip_index = skip_indices[0]
    detail.update({"skip_choice_index": skip_index, "eligible": True})
    if skip_index == recommended_index:
        detail["reason"] = "parent_already_selected_skip"
        return recommended_index, detail
    detail.update({
        "selected_index": skip_index,
        "overridden": True,
        "reason": (
            "large_deck_duplicate_skip_override"
            if require_recommended_duplicate
            else "large_deck_skip_override"
        ),
    })
    return skip_index, detail


def deterministic_noncombat_step(gc: Any, sts: Any) -> str:
    if gc.screen_state == sts.ScreenState.REWARDS:
        offered = list(gc.get_card_reward())
        if offered:
            gc.pick_reward_card(offered[0])
            return "reward:first_card"
        gc.skip_reward_cards()
        return "reward:skip_empty"

    supported = {
        sts.ScreenState.MAP_SCREEN,
        sts.ScreenState.REST_ROOM,
        sts.ScreenState.SHOP_ROOM,
        sts.ScreenState.EVENT_SCREEN,
    }
    if gc.screen_state not in supported:
        raise SimulatorRunError(f"unsupported non-combat screen: {gc.screen_state}")
    actions = list(sts.get_legal_game_actions(gc))
    if not actions:
        raise SimulatorRunError(f"no legal non-combat action on screen: {gc.screen_state}")
    actions[0].execute(gc)
    return f"{_enum_name(gc.screen_state)}:first_legal"


class ArmGNoncombatPolicy:
    """ArmG inference with an optional independently evolvable map scorer."""

    def __init__(
        self,
        *,
        root: Path,
        weight_path: Path,
        map_weight_path: Path | None = None,
        activation_probe: bool = False,
    ) -> None:
        if not root.is_dir():
            raise SimulatorRunError(f"ArmG compatibility root missing: {root}")
        if not weight_path.is_file():
            raise SimulatorRunError(f"ArmG weight missing: {weight_path}")
        if map_weight_path is not None and not map_weight_path.is_file():
            raise SimulatorRunError(f"ArmG map weight missing: {map_weight_path}")

        os.environ["STS_BOT_DIR"] = str(root)
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        try:
            torch = importlib.import_module("torch")
            module = importlib.import_module("armG_train")
        except Exception as exc:
            raise SimulatorRunError(f"could not import frozen ArmG runtime: {exc}") from exc

        # Freeze vocabulary lookup exactly like the accepted baseline lifecycle fix.
        module.card_idx = lambda name: module._vocab.get(name, module.VOCAB_CAP - 1)

        def load_net(path: Path) -> Any:
            net = module.Scorer((128, 128))
            net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            net.eval()
            return net

        try:
            net = load_net(weight_path)
            map_net = load_net(map_weight_path) if map_weight_path is not None else net
        except Exception as exc:
            raise SimulatorRunError(f"could not load ArmG weight: {exc}") from exc

        # Optional v3 residual adapter sidecar.  The base G7 checkpoint stays
        # byte-for-byte unchanged; only candidate checkpoints that have
        # "<weight>.adapter.pt" next to them receive the local score correction.
        adapter = None
        adapter_kinds: set[str] = set()
        adapter_gate = None
        adapter_path = Path(str(weight_path) + ".adapter.pt")
        if adapter_path.is_file():
            try:
                payload = torch.load(adapter_path, weights_only=True, map_location="cpu")
                schema = payload.get("schema_version")
                if schema not in {
                    "sts1-armg-residual-adapter-v1",
                    "sts1-armg-residual-adapter-v2",
                }:
                    raise RuntimeError(f"unsupported adapter schema: {schema}")
                input_dim = int(payload["input_dim"])
                hidden_dim = int(payload["hidden_dim"])
                adapter = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_dim, 1),
                )
                adapter.load_state_dict(payload["state_dict"])
                adapter.eval()
                adapter_kinds = {str(x) for x in payload.get("kinds", [])}
                if not adapter_kinds:
                    raise RuntimeError("adapter has no enabled decision kinds")

                if schema == "sts1-armg-residual-adapter-v2":
                    gate = payload.get("gate")
                    if not isinstance(gate, dict):
                        raise RuntimeError("v2 adapter is missing confidence gate")
                    mean = gate.get("obs_mean")
                    std = gate.get("obs_std")
                    if not hasattr(mean, "numel") or int(mean.numel()) != int(module.OBS_DIM):
                        raise RuntimeError("adapter gate mean dimension mismatch")
                    if not hasattr(std, "numel") or int(std.numel()) != int(module.OBS_DIM):
                        raise RuntimeError("adapter gate std dimension mismatch")
                    if bool(torch.any(std <= 0)):
                        raise RuntimeError("adapter gate std must be positive")
                    positive = gate.get("positive_centers") or {}
                    negative = gate.get("negative_centers") or {}
                    if not isinstance(positive, dict) or not isinstance(negative, dict):
                        raise RuntimeError("adapter gate centers are malformed")
                    for kind in adapter_kinds:
                        centers = positive.get(kind)
                        if centers is None or not hasattr(centers, "shape"):
                            raise RuntimeError(f"adapter gate has no positive centers for {kind}")
                        if len(centers.shape) != 2 or int(centers.shape[1]) != int(module.OBS_DIM):
                            raise RuntimeError(f"adapter gate positive center shape mismatch for {kind}")
                        neg = negative.get(kind)
                        if neg is not None:
                            if not hasattr(neg, "shape"):
                                raise RuntimeError(f"adapter gate negative center shape mismatch for {kind}")
                            # Empty negative-center tensors mean this decision kind
                            # has no negative exemplars. Canonicalize any empty
                            # representation to (0, OBS_DIM) instead of treating
                            # it as a malformed gate.
                            if int(neg.numel()) == 0:
                                negative[kind] = neg.reshape(0, int(module.OBS_DIM))
                            elif len(neg.shape) != 2 or int(neg.shape[1]) != int(module.OBS_DIM):
                                raise RuntimeError(f"adapter gate negative center shape mismatch for {kind}")
                    ratio = float(gate.get("positive_to_negative_ratio", 0.80))
                    if not 0.0 < ratio <= 1.0:
                        raise RuntimeError("adapter gate ratio outside (0,1]")
                    max_distance = {
                        str(k): float(v)
                        for k, v in (gate.get("max_positive_distance") or {}).items()
                    }
                    adapter_gate = {
                        "obs_mean": mean.to(dtype=torch.float32),
                        "obs_std": std.to(dtype=torch.float32),
                        "positive_centers": {
                            str(k): v.to(dtype=torch.float32) for k, v in positive.items()
                        },
                        "negative_centers": {
                            str(k): v.to(dtype=torch.float32) for k, v in negative.items()
                        },
                        "positive_to_negative_ratio": ratio,
                        "max_positive_distance": max_distance,
                    }
            except Exception as exc:
                raise SimulatorRunError(f"could not load ArmG residual adapter: {exc}") from exc

        self.module = module
        self.torch = torch
        self.net = net
        self.map_net = map_net
        self.adapter = adapter
        self.adapter_kinds = adapter_kinds
        self.adapter_gate = adapter_gate
        self.adapter_path = adapter_path if adapter is not None else None
        self.weight_path = weight_path
        self.map_weight_path = map_weight_path or weight_path
        self.contextual_card_rerank = os.environ.get("STS1_TEACHER_V2_CONTEXTUAL_RERANK", "0") == "1"
        self.activation_probe = bool(activation_probe)
        self.activation_total_noncombat_decisions = 0
        self.activation_multi_choice_decisions = 0
        self.activation_decision_kinds: dict[str, int] = {}
        self.activation_records: list[dict[str, Any]] = []

    def _adapter_gate_metrics(self, kind: str, obs: Any) -> dict[str, Any]:
        """Return auditable confidence-gate distances without changing gate semantics."""
        if self.adapter_gate is None:
            return {"allowed": True, "d_positive": None, "d_negative": None,
                    "max_positive_distance": None, "positive_to_negative_ratio": None}
        gate = self.adapter_gate
        positive = gate["positive_centers"].get(str(kind))
        ratio = float(gate["positive_to_negative_ratio"])
        max_distance = gate["max_positive_distance"].get(str(kind))
        if positive is None or int(positive.shape[0]) == 0:
            return {"allowed": False, "d_positive": None, "d_negative": None,
                    "max_positive_distance": max_distance,
                    "positive_to_negative_ratio": ratio}
        z = (obs - gate["obs_mean"]) / gate["obs_std"]
        d_pos = float(self.torch.mean((positive - z.unsqueeze(0)) ** 2, dim=1).min())
        negative = gate["negative_centers"].get(str(kind))
        d_neg = None
        if negative is not None and int(negative.shape[0]) > 0:
            d_neg = float(self.torch.mean((negative - z.unsqueeze(0)) ** 2, dim=1).min())
        allowed = max_distance is None or d_pos <= float(max_distance)
        if allowed and d_neg is not None:
            allowed = d_pos <= d_neg * ratio
        return {"allowed": bool(allowed), "d_positive": d_pos, "d_negative": d_neg,
                "max_positive_distance": max_distance,
                "positive_to_negative_ratio": ratio}

    def _adapter_gate_allows(self, kind: str, obs: Any) -> bool:
        return bool(self._adapter_gate_metrics(kind, obs)["allowed"])

    def choices(self, gc: Any) -> tuple[str, list[Any], list[Any]]:
        kind, descs, execs = self.module.build_choices(gc)
        return str(kind), list(descs), list(execs)

    def score_choices(self, gc: Any) -> tuple[str, list[Any], Any]:
        kind, descs, _ = self.choices(gc)
        if not descs:
            return kind, descs, self.torch.empty(0, dtype=self.torch.float32)
        obs = self.torch.tensor(self.module.obs_vec(gc), dtype=self.torch.float32)
        net = self.map_net if kind == "map" else self.net
        with self.torch.no_grad():
            scores = net.score(obs, descs)
            raw_scores = scores
            adapter_enabled = self.adapter is not None and kind in self.adapter_kinds
            gate_checked = bool(adapter_enabled and self.adapter_gate is not None)
            gate_metrics = self._adapter_gate_metrics(kind, obs) if adapter_enabled else {
                "allowed": False, "d_positive": None, "d_negative": None,
                "max_positive_distance": None, "positive_to_negative_ratio": None,
            }
            gate_allowed = bool(adapter_enabled and gate_metrics["allowed"])
            residual_scores = self.torch.zeros_like(scores)
            if adapter_enabled:
                desc_tensor = self.torch.tensor(descs, dtype=self.torch.float32)
                x = self.torch.cat([obs.repeat(len(descs), 1), desc_tensor], 1)
                if int(x.shape[1]) != int(self.adapter[0].in_features):
                    raise SimulatorRunError(
                        f"ArmG residual adapter input mismatch: {int(x.shape[1])} != "
                        f"{int(self.adapter[0].in_features)}"
                    )
                residual_scores = self.adapter(x).squeeze(1)
                scores = apply_residual_scores(scores, residual_scores, gate_allowed)
            if self.activation_probe:
                raw_values = [float(x) for x in raw_scores.tolist()]
                residual_values = [float(x) for x in residual_scores.tolist()]
                candidate_values = [float(x) for x in scores.tolist()]
                audit_values = raw_values + residual_values + candidate_values
                audit_values.extend(
                    float(gate_metrics[key])
                    for key in ("d_positive", "d_negative", "max_positive_distance")
                    if gate_metrics[key] is not None
                )
                if not all(math.isfinite(value) for value in audit_values):
                    raise SimulatorRunError("non-finite score or confidence distance in activation probe")
                raw_order = sorted(raw_values, reverse=True)
                candidate_order = sorted(candidate_values, reverse=True)
                self.activation_records.append({
                    "decision_kind": kind,
                    "choice_semantics": [self.describe_choice(kind, d) for d in descs],
                    "g7_raw_scores": raw_values,
                    "g7_top1_index": top1_index(raw_scores),
                    "gate_checked": gate_checked,
                    "gate_allowed": gate_allowed,
                    "d_positive": gate_metrics["d_positive"],
                    "d_negative": gate_metrics["d_negative"],
                    "max_positive_distance": gate_metrics["max_positive_distance"],
                    "positive_to_negative_ratio": gate_metrics["positive_to_negative_ratio"],
                    "adapter_applied": gate_allowed,
                    "adapter_residual_scores": residual_values,
                    "candidate_scores": candidate_values,
                    "candidate_top1_index": top1_index(scores),
                    "top1_changed": top1_index(raw_scores) != top1_index(scores),
                    "margin_before": raw_order[0] - raw_order[1],
                    "margin_after": candidate_order[0] - candidate_order[1],
                    "floor": _value(gc, "floor_num"),
                    "act": _value(gc, "act"),
                    "hp": _value(gc, "cur_hp"),
                    "gold": _value(gc, "gold"),
                })
        return kind, descs, scores

    def choose_index(self, gc: Any, sts: Any) -> tuple[str, int, int]:
        kind, descs, _ = self.choices(gc)
        if not descs:
            if gc.screen_state == sts.ScreenState.REWARDS:
                return "reward_empty", -1, 0
            raise SimulatorRunError(f"ArmG produced no legal choice on screen: {gc.screen_state}")
        if len(descs) == 1:
            return kind, 0, 1
        _, _, scores = self.score_choices(gc)
        return kind, top1_index(scores), len(descs)

    def _contextual_card_adjustments(self, gc: Any, descs: list[Any]) -> list[float]:
        """Small, auditable v2 card-reward adjustments derived from winner/near-win mining."""
        deck = [str(self.module.card_name(c)) for c in list(getattr(gc, "deck", []))]
        n = max(1, len(deck))
        draw = {"Battle Trance","Pommel Strike","Burning Pact","Offering","Shrug It Off"}
        defense = {"Shrug It Off","Impervious","Flame Barrier","Power Through","Ghostly Armor","True Grit","Second Wind"}
        frontload = {"Carnage","Bludgeon","Perfected Strike","Hemokinesis","Uppercut","Wild Strike","Clothesline","Twin Strike"}
        exhaust = {"Fiend Fire","Burning Pact","True Grit","Second Wind","Corruption"}
        have_draw=sum(x in draw for x in deck); have_def=sum(x in defense for x in deck)
        have_front=sum(x in frontload for x in deck); have_exhaust=sum(x in exhaust for x in deck)
        thick=max(0, n-24)
        out=[]
        for d in descs:
            sem=self.describe_choice("card", d); name=sem.get("card_name"); adj=0.0
            if sem.get("choice")=="skip":
                adj += min(0.45, 0.04*thick)
            elif name:
                copies=deck.count(name)
                if copies>=2: adj -= min(0.40, 0.12*(copies-1))
                if name in draw and have_draw<2: adj += 0.28
                if name in defense and have_def<3: adj += 0.22
                if name in frontload and have_front>=5: adj -= 0.22
                if name in exhaust and have_exhaust>=1: adj += 0.10
                if thick and name in frontload and copies: adj -= min(0.28,0.05*thick)
            out.append(adj)
        return out

    def decide(self, gc: Any, sts: Any) -> tuple[str, int, list[Any], list[Any], list[float]]:
        kind, descs, execs = self.choices(gc)
        if self.activation_probe:
            self.activation_total_noncombat_decisions += 1
            self.activation_decision_kinds[kind] = self.activation_decision_kinds.get(kind, 0) + 1
            if len(descs) >= 2:
                self.activation_multi_choice_decisions += 1
        if not descs:
            if gc.screen_state == sts.ScreenState.REWARDS:
                return "reward_empty", -1, [], [], []
            raise SimulatorRunError(f"ArmG produced no legal choice on screen: {gc.screen_state}")
        if len(descs) == 1:
            return kind, 0, descs, execs, [0.0]
        _, _, scores = self.score_choices(gc)
        raw=[float(x) for x in scores.tolist()]
        adjusted=list(raw)
        if self.contextual_card_rerank and kind == "card":
            adj=self._contextual_card_adjustments(gc, descs)
            adjusted=[x+y for x,y in zip(raw,adj)]
            scores=self.torch.tensor(adjusted,dtype=self.torch.float32)
            self.last_contextual_rerank={"raw_scores":raw,"adjustments":adj,"adjusted_scores":adjusted,"raw_index":max(range(len(raw)),key=raw.__getitem__),"adjusted_index":max(range(len(adjusted)),key=adjusted.__getitem__)}
        else:
            self.last_contextual_rerank=None
        return kind, int(self.torch.argmax(scores).item()), descs, execs, adjusted

    def describe_choice(self, kind: str, desc: Any) -> dict[str, Any]:
        m = self.module
        d = list(desc)
        out: dict[str, Any] = {"kind": kind}
        reverse_vocab = {int(v): str(k) for k, v in m._vocab.items()}
        if kind == "card":
            active = [i for i in range(m.W_CARD) if d[m.OFF_CARD + i] > 0.5]
            out.update(choice="skip" if d[m.OFF_PASS] > 0.5 else "card")
            if active: out["card_name"] = reverse_vocab.get(active[0], f"vocab:{active[0]}")
        elif kind == "map":
            reverse_room = {int(v): str(getattr(k, "name", k)) for k, v in m.ROOM_IDX.items()}
            room = [i for i in range(m.W_MROOM) if d[m.OFF_MROOM + i] > 0.5]
            out["room"] = reverse_room.get(room[0], str(room[0])) if room else None
            out["lookahead_1"] = {reverse_room.get(i, str(i)): d[m.OFF_MLA1 + i] for i in range(m.W_MLA1) if d[m.OFF_MLA1 + i]}
            out["lookahead_2"] = {reverse_room.get(i, str(i)): d[m.OFF_MLA2 + i] for i in range(m.W_MLA2) if d[m.OFF_MLA2 + i]}
        elif kind == "shop":
            reverse_item = {int(v): str(getattr(k, "name", k)) for k, v in m.SITEM_IDX.items()}
            item = [i for i in range(m.W_SITEM) if d[m.OFF_SITEM + i] > 0.5]
            out["item_type"] = reverse_item.get(item[0], str(item[0])) if item else None
            out["price_normalized"] = d[m.OFF_SPRICE]
            card = [i for i in range(m.W_CARD) if d[m.OFF_CARD + i] > 0.5]
            if card: out["card_name"] = reverse_vocab.get(card[0], f"vocab:{card[0]}")
            out["leave"] = bool(d[m.OFF_PASS] > 0.5)
        elif kind == "rest":
            opt = [i for i in range(m.W_REST) if d[m.OFF_REST + i] > 0.5]
            out["option_index"] = opt[0] if opt else None
        elif kind == "event":
            evid = [i for i in range(m.EVENT_CAP) if d[m.OFF_EVID + i] > 0.5]
            eopt = [i for i in range(m.EOPT_CAP) if d[m.OFF_EOPT + i] > 0.5]
            out["event_id"] = evid[0] if evid else None
            out["option_index"] = eopt[0] if eopt else None
        return out

    def training_vector_snapshot(self, gc: Any, descs: list[Any]) -> dict[str, Any]:
        """Exact ArmG inputs needed for offline fine-tuning."""
        return {
            "obs_412": [float(x) for x in self.module.obs_vec(gc)],
            "candidate_desc_368": [[float(x) for x in list(d)] for d in descs],
        }

    def deck_snapshot(self, gc: Any) -> list[dict[str, Any]]:
        result = []
        for i, card in enumerate(list(getattr(gc, "deck", []))):
            result.append({
                "position": i + 1,
                "name": str(self.module.card_name(card)),
                "upgrades": int(getattr(card, "upgrades", getattr(card, "upgrade_count", 0)) or 0),
            })
        return result

    def step(self, gc: Any, sts: Any) -> str:
        kind, index, descs, execs, _ = self.decide(gc, sts)
        if index < 0:
            gc.skip_reward_cards()
            return "armg:reward:skip_empty"
        execs[index](gc)
        return f"armg:{kind}:{index}/{len(descs)}"


def _set_pauses(agent: Any) -> None:
    agent.pause_on_card_reward = True
    agent.pause_on_map = True
    agent.pause_on_rest = True
    agent.pause_on_shop = True
    agent.pause_on_event = True
    agent.pause_on_battle = True


def resolve_simulator_seed(
    sts: Any,
    seed: str | int,
    *,
    heldout_seeds: Sequence[int] | None = None,
    training_seeds: Sequence[int] | None = None,
) -> tuple[str, int]:
    """Resolve frozen UI, held-out evaluation, or training-only internal seeds."""

    if heldout_seeds is not None and training_seeds is not None:
        raise SimulatorRunError("held-out and training seed allowlists are mutually exclusive")

    ui_seed = str(seed)
    if training_seeds is not None:
        try:
            simulator_seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise SimulatorRunError(f"training simulator seed must be an integer: {seed!r}") from exc
        allowed = {int(value) for value in training_seeds}
        if simulator_seed not in allowed:
            raise SimulatorRunError("training simulator seed is not present in the training seed allowlist")
        return ui_seed, simulator_seed

    if heldout_seeds is not None:
        try:
            simulator_seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise SimulatorRunError(f"held-out simulator seed must be an integer: {seed!r}") from exc
        allowed = {int(value) for value in heldout_seeds}
        if simulator_seed not in allowed:
            raise SimulatorRunError("held-out simulator seed is not present in the pinned evaluation seed set")
        return ui_seed, simulator_seed

    if ui_seed not in A0_FROZEN_SEEDS_V1:
        raise SimulatorRunError("formal simulator run requires a seed from the frozen A0 manifest")
    get_seed_long = getattr(sts, "get_seed_long", None)
    get_seed_str = getattr(sts, "get_seed_str", None)
    if not callable(get_seed_long) or not callable(get_seed_str):
        raise SimulatorRunError("pinned simulator seed conversion API is unavailable")
    try:
        simulator_seed = int(get_seed_long(ui_seed))
        round_trip = str(get_seed_str(simulator_seed))
    except Exception as exc:
        raise SimulatorRunError(f"pinned simulator seed conversion failed: {exc}") from exc
    if round_trip != ui_seed:
        raise SimulatorRunError(
            f"pinned simulator seed round-trip mismatch: ui={ui_seed!r} round_trip={round_trip!r}"
        )
    return ui_seed, simulator_seed


def _combat_mcts_budget_for_floor(
    floor: int,
    *,
    base_sims: int | None,
    late_sims: int | None = None,
    late_floor: int = 50,
    boss_sims: int | None = None,
    boss_floors: Sequence[int] = (16, 33, 50),
) -> int | None:
    """Choose a search budget without increasing ordinary-combat cost."""
    active = base_sims
    if boss_sims is not None and int(floor) in {int(v) for v in boss_floors}:
        return int(boss_sims)
    if late_sims is not None and int(floor) >= int(late_floor):
        return int(late_sims)
    return active


def run_simulator_game(
    *,
    student: Any,
    sts: Any,
    seed: str | int,
    evidence_path: Path | None = None,
    max_game_steps: int = MAX_GAME_STEPS,
    max_battle_steps: int = MAX_BATTLE_STEPS,
    armg_policy: ArmGNoncombatPolicy | None = None,
    combat_mcts_sims: int | None = None,
    combat_mcts_budgets: Sequence[int] | None = None,
    hybrid_mcts_budgets: Sequence[int] | None = None,
    combat_mcts_late_sims: int | None = None,
    combat_mcts_late_floor: int = 50,
    combat_mcts_boss_sims: int | None = None,
    combat_mcts_boss_floors: Sequence[int] = (16, 33, 50),
    combat_mcts_exploration: float | None = None,
    reserve_last_potion_until_floor: int | None = None,
    use_potion_below_hp_fraction: float | None = None,
    lethal_potion_rescue: bool = False,
    avoid_low_hp_elite_routes: bool = False,
    prefer_smith_when_rest_overheals: bool = False,
    prefer_smith_when_overheal_exceeds_effective_rest_heal: bool = False,
    skip_card_reward_when_deck_size_at_least: int | None = None,
    skip_duplicate_card_reward_when_deck_size_at_least: int | None = None,
    require_all_card_reward_options_are_duplicates: bool = False,
    heldout_seeds: Sequence[int] | None = None,
    training_seeds: Sequence[int] | None = None,
    collect_ppo: bool = False,
    collect_teacher: bool = False,
    diagnostic_trace_path: Path | None = None,
    diagnostic_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if prefer_smith_when_rest_overheals and prefer_smith_when_overheal_exceeds_effective_rest_heal:
        raise SimulatorRunError("choose only one Campfire over-heal rule")
    if skip_card_reward_when_deck_size_at_least is not None:
        if (
            not isinstance(skip_card_reward_when_deck_size_at_least, int)
            or isinstance(skip_card_reward_when_deck_size_at_least, bool)
            or skip_card_reward_when_deck_size_at_least < 1
        ):
            raise SimulatorRunError("card-reward deck-size threshold must be a positive integer")
        if armg_policy is None:
            raise SimulatorRunError("card-reward Skip intervention requires the ArmG policy")
    if skip_duplicate_card_reward_when_deck_size_at_least is not None:
        if (
            not isinstance(skip_duplicate_card_reward_when_deck_size_at_least, int)
            or isinstance(skip_duplicate_card_reward_when_deck_size_at_least, bool)
            or skip_duplicate_card_reward_when_deck_size_at_least < 1
        ):
            raise SimulatorRunError("duplicate-card deck-size threshold must be a positive integer")
        if armg_policy is None:
            raise SimulatorRunError("duplicate-card Skip intervention requires the ArmG policy")
    if not isinstance(require_all_card_reward_options_are_duplicates, bool):
        raise SimulatorRunError("all-duplicate reward option mode must be a boolean")
    if (
        require_all_card_reward_options_are_duplicates
        and skip_duplicate_card_reward_when_deck_size_at_least is None
    ):
        raise SimulatorRunError(
            "all-duplicate reward option mode requires the duplicate-card Skip intervention"
        )
    ui_seed, simulator_seed = resolve_simulator_seed(
        sts,
        seed,
        heldout_seeds=heldout_seeds,
        training_seeds=training_seeds,
    )
    seed_contract = (
        "training_internal"
        if training_seeds is not None
        else ("heldout_internal" if heldout_seeds is not None else "frozen_a0_ui")
    )
    if max_game_steps < 1 or max_battle_steps < 1:
        raise SimulatorRunError("simulator step bounds must be positive")
    consensus_budgets = tuple(int(value) for value in (combat_mcts_budgets or ()))
    hybrid_budgets = tuple(int(value) for value in (hybrid_mcts_budgets or ()))
    active_mcts_modes = sum(bool(value) for value in (
        combat_mcts_sims is not None,
        consensus_budgets,
        hybrid_budgets,
    ))
    if active_mcts_modes > 1:
        raise SimulatorRunError("choose exactly one MCTS combat mode")
    if combat_mcts_sims is not None and consensus_budgets:
        raise SimulatorRunError("choose either one MCTS budget or consensus budgets, not both")
    if combat_mcts_late_sims is not None and consensus_budgets:
        raise SimulatorRunError("late MCTS schedule cannot be combined with consensus budgets")
    if combat_mcts_boss_sims is not None and consensus_budgets:
        raise SimulatorRunError("Boss MCTS schedule cannot be combined with consensus budgets")
    if combat_mcts_late_sims is not None and combat_mcts_sims is None:
        raise SimulatorRunError("late MCTS schedule requires --combat-mcts-sims as the base budget")
    if combat_mcts_boss_sims is not None and combat_mcts_sims is None:
        raise SimulatorRunError("Boss MCTS schedule requires a base combat MCTS budget")
    if combat_mcts_sims is not None and combat_mcts_sims < 1:
        raise SimulatorRunError("combat MCTS simulations must be positive")
    if combat_mcts_late_sims is not None and combat_mcts_late_sims < 1:
        raise SimulatorRunError("late combat MCTS simulations must be positive")
    if combat_mcts_boss_sims is not None and combat_mcts_boss_sims < 1:
        raise SimulatorRunError("Boss combat MCTS simulations must be positive")
    boss_floor_set = tuple(sorted({int(v) for v in combat_mcts_boss_floors}))
    if any(v < 1 for v in boss_floor_set):
        raise SimulatorRunError("Boss MCTS floors must be positive")
    if combat_mcts_late_floor < 1:
        raise SimulatorRunError("late combat MCTS floor must be positive")
    if combat_mcts_exploration is not None and combat_mcts_exploration <= 0:
        raise SimulatorRunError("combat MCTS exploration parameter must be positive")
    if combat_mcts_exploration is not None and consensus_budgets:
        raise SimulatorRunError("tuned exploration cannot be combined with consensus budgets")
    if reserve_last_potion_until_floor is not None:
        if (
            not isinstance(reserve_last_potion_until_floor, int)
            or isinstance(reserve_last_potion_until_floor, bool)
            or reserve_last_potion_until_floor < 1
        ):
            raise SimulatorRunError("potion reserve floor must be a positive integer")
        if (
            combat_mcts_sims is None
            or consensus_budgets
            or hybrid_budgets
            or combat_mcts_late_sims is not None
            or combat_mcts_boss_sims is not None
            or combat_mcts_exploration is not None
        ):
            raise SimulatorRunError(
                "last-potion reserve requires one fixed, untuned combat MCTS budget"
            )
    if use_potion_below_hp_fraction is not None:
        if (
            not isinstance(use_potion_below_hp_fraction, (int, float))
            or isinstance(use_potion_below_hp_fraction, bool)
            or not math.isfinite(float(use_potion_below_hp_fraction))
            or not 0.0 <= float(use_potion_below_hp_fraction) <= 1.0
        ):
            raise SimulatorRunError("emergency potion HP threshold must be between zero and one")
        if (
            combat_mcts_sims is None
            or consensus_budgets
            or hybrid_budgets
            or combat_mcts_late_sims is not None
            or combat_mcts_boss_sims is not None
            or combat_mcts_exploration is not None
        ):
            raise SimulatorRunError(
                "emergency potion use requires one fixed, untuned combat MCTS budget"
            )
    if not isinstance(lethal_potion_rescue, bool):
        raise SimulatorRunError("lethal potion rescue must be a boolean")
    if lethal_potion_rescue and (
        combat_mcts_sims is None
        or consensus_budgets
        or hybrid_budgets
        or combat_mcts_late_sims is not None
        or combat_mcts_boss_sims is not None
        or combat_mcts_exploration is not None
    ):
        raise SimulatorRunError(
            "lethal potion rescue requires one fixed, untuned combat MCTS budget"
        )
    if not isinstance(avoid_low_hp_elite_routes, bool):
        raise SimulatorRunError("low-HP Elite route intervention must be a boolean")
    if avoid_low_hp_elite_routes and armg_policy is None:
        raise SimulatorRunError("low-HP Elite route intervention requires the ArmG map policy")
    if sum(
        value
        for value in (
            avoid_low_hp_elite_routes,
            reserve_last_potion_until_floor is not None,
            use_potion_below_hp_fraction is not None,
            lethal_potion_rescue,
            skip_card_reward_when_deck_size_at_least is not None,
            skip_duplicate_card_reward_when_deck_size_at_least is not None,
        )
    ) > 1:
        raise SimulatorRunError("enable only one primary policy intervention per run")
    if any(value < 1 for value in consensus_budgets):
        raise SimulatorRunError("all MCTS consensus budgets must be positive")
    if hybrid_budgets and (len(hybrid_budgets) < 2 or any(value < 1 for value in hybrid_budgets)):
        raise SimulatorRunError("hybrid MCTS requires at least two positive budgets")
    if collect_ppo and collect_teacher:
        raise SimulatorRunError("PPO and MCTS Teacher collection are mutually exclusive")
    if collect_ppo and (combat_mcts_sims is not None or consensus_budgets or hybrid_budgets):
        raise SimulatorRunError("PPO rollout collection cannot run with MCTS combat policy")
    if collect_ppo and not callable(getattr(student, "sample_action", None)):
        raise SimulatorRunError("PPO rollout collection requires a Student v1 sample_action policy")
    if collect_teacher and combat_mcts_sims is None and not consensus_budgets:
        raise SimulatorRunError("MCTS Teacher collection requires an MCTS combat policy; pure MCTS only")
    if collect_teacher and hybrid_budgets:
        raise SimulatorRunError("MCTS Teacher collection cannot use Student-influenced hybrid MCTS")

    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, simulator_seed, 0)
    agent = sts.Agent()
    _set_pauses(agent)
    adapter = SimulatorCombatAdapter()
    student_actions = 0
    fallback_count = 0
    armg_action_count = 0
    mcts_action_count = 0
    potion_reserve_override_count = 0
    emergency_potion_override_count = 0
    lethal_potion_rescue_override_count = 0
    lethal_potion_rescue_reason_counts: dict[str, int] = {}
    card_reward_skip_override_count = 0
    card_reward_skip_eligible_count = 0
    card_reward_skip_reason_counts: dict[str, int] = {}
    campfire_overheal_override_count = 0
    map_elite_avoidance_override_count = 0
    map_elite_avoidance_fail_closed_count = 0
    hybrid_student_vote_count = 0
    hybrid_student_tiebreak_count = 0
    illegal_actions = 0
    timeout_count = 0
    crash_count = 0
    equivalent_action_alias_count = 0
    latencies_ms: list[float] = []
    game_steps = 0
    encounter_index = 0
    max_floor = int(_value(gc, "floor_num", 0) or 0)
    max_act = int(_value(gc, "act", 0) or 0)
    error: str | None = None
    diagnostic_legal_actions_complete = True

    _record(evidence_path, {
        "type": "frozen_manifest",
        "manifest": frozen_a0_manifest(),
        "ui_seed": ui_seed,
        "simulator_seed_long": simulator_seed,
        "seed_contract": seed_contract,
    })
    _record(diagnostic_trace_path, {
        "type": "diagnostic_trace_header_v1",
        "trace_schema": "sts1-diagnostic-trace-v1",
        "manifest": frozen_a0_manifest(),
        "ui_seed": ui_seed,
        "simulator_seed_long": simulator_seed,
        "seed_contract": seed_contract,
        "combat_policy_intervention": (
            {
                "kind": "reserve_last_usable_potion_before_floor",
                "until_floor": reserve_last_potion_until_floor,
            }
            if reserve_last_potion_until_floor is not None
            else None
        ),
        "emergency_potion_policy_intervention": (
            {
                "kind": "use_first_legal_potion_below_hp_fraction",
                "hp_ratio_threshold": float(use_potion_below_hp_fraction),
                "tie_break": "lowest_potion_slot_then_native_legal_action_order",
                "fallback": "keep_mcts_recommendation_if_required_data_or_legal_action_missing",
            }
            if use_potion_below_hp_fraction is not None
            else None
        ),
        "campfire_policy_intervention": (
            {
                "kind": (
                    "prefer_smith_when_overheal_exceeds_effective_rest_heal"
                    if prefer_smith_when_overheal_exceeds_effective_rest_heal
                    else "prefer_smith_when_rest_overheals"
                ),
                "rest_option_index": 0,
                "smith_option_index": 1,
                "rest_heal": "round(0.30*max_hp) + 15 with REGAL_PILLOW",
                "override_condition": (
                    "overheal > min(rest_heal, max_hp - current_hp)"
                    if prefer_smith_when_overheal_exceeds_effective_rest_heal
                    else "rest_heal > max_hp - current_hp"
                ),
                "fallback": "keep_armg_recommendation_if_state_or_legal_choices_are_incomplete",
            }
            if (
                prefer_smith_when_rest_overheals
                or prefer_smith_when_overheal_exceeds_effective_rest_heal
            )
            else None
        ),
        "map_policy_intervention": (
            {
                "kind": "avoid_elite_below_half_hp",
                "hp_ratio_threshold": 0.5,
                "fallback": "keep_armg_recommendation_if_required_data_incomplete",
            }
            if avoid_low_hp_elite_routes
            else None
        ),
        "run_metadata": dict(diagnostic_metadata or {}),
    })

    try:
        while gc.outcome == sts.GameOutcome.UNDECIDED and game_steps < max_game_steps:
            game_steps += 1
            agent.playout(gc)
            max_floor = max(max_floor, int(_value(gc, "floor_num", 0) or 0))
            max_act = max(max_act, int(_value(gc, "act", 0) or 0))
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                break

            if gc.screen_state != sts.ScreenState.BATTLE:
                screen_before = _enum_name(gc.screen_state)
                diagnostic_run_before = (
                    _diagnostic_run_snapshot(gc, armg_policy)
                    if diagnostic_trace_path is not None
                    else None
                )
                diagnostic_choices: list[dict[str, Any]] | None = None
                diagnostic_route: dict[str, Any] | None = None
                diagnostic_selected_index: int | None = None
                diagnostic_recommended_index: int | None = None
                map_policy_intervention: dict[str, Any] | None = None
                campfire_policy_intervention: dict[str, Any] | None = None
                card_reward_skip_intervention: dict[str, Any] | None = None
                if armg_policy is None:
                    if diagnostic_trace_path is not None:
                        (
                            diagnostic_choices,
                            diagnostic_selected_index,
                        ) = _diagnostic_fallback_legal_choices(gc, sts)
                        if (
                            diagnostic_choices is None
                            or diagnostic_selected_index is None
                        ):
                            diagnostic_legal_actions_complete = False
                        if screen_before == _enum_name(sts.ScreenState.MAP_SCREEN):
                            diagnostic_route = _diagnostic_map_route(
                                gc,
                                sts,
                                choice_count=len(diagnostic_choices or []),
                                selected_index=(
                                    diagnostic_selected_index
                                    if diagnostic_selected_index is not None
                                    else -1
                                ),
                            )
                            if not diagnostic_route["choices_complete"]:
                                diagnostic_legal_actions_complete = False
                        diagnostic_recommended_index = diagnostic_selected_index
                    choice = deterministic_noncombat_step(gc, sts)
                    fallback_count += 1
                    policy_name = "legacy_fallback"
                else:
                    kind, selected_index, descs, execs, scores = armg_policy.decide(gc, sts)
                    recommended_index = selected_index
                    diagnostic_recommended_index = recommended_index
                    if kind == "map" and (
                        diagnostic_trace_path is not None or avoid_low_hp_elite_routes
                    ):
                        diagnostic_route = _diagnostic_map_route(
                            gc,
                            sts,
                            choice_count=len(descs),
                            selected_index=recommended_index,
                        )
                        if (
                            diagnostic_trace_path is not None
                            and not diagnostic_route["choices_complete"]
                        ):
                            diagnostic_legal_actions_complete = False
                    if avoid_low_hp_elite_routes and kind == "map":
                        module = getattr(armg_policy, "module", None)
                        room_vocabulary = getattr(module, "ROOM_IDX", {})
                        known_room_names = {
                            str(getattr(room, "name", room)).upper()
                            for room in room_vocabulary
                        }
                        selected_index, map_policy_intervention = (
                            _apply_low_hp_elite_route_avoidance(
                                recommended_index,
                                scores,
                                diagnostic_route,
                                current_hp=_value(gc, "cur_hp", None),
                                max_hp=_value(gc, "max_hp", None),
                                known_room_names=known_room_names,
                            )
                        )
                        if map_policy_intervention["status"] == "overridden":
                            map_elite_avoidance_override_count += 1
                        elif map_policy_intervention["status"] == "fail_closed":
                            map_elite_avoidance_fail_closed_count += 1
                        if diagnostic_route is not None:
                            diagnostic_route["recommended_selected_index"] = recommended_index
                            diagnostic_route["actual_selected_index"] = selected_index
                            diagnostic_route["recommended_route"] = diagnostic_route.get(
                                "selected_route"
                            )
                            if (
                                diagnostic_route.get("choices_complete") is True
                                and 0 <= selected_index < len(diagnostic_route["choices"])
                            ):
                                actual_route = diagnostic_route["choices"][selected_index]
                                diagnostic_route["selected_route"] = {
                                    "from": diagnostic_route.get("source_node"),
                                    "to": actual_route.get("target_node"),
                                    "room": actual_route.get("target_room"),
                                    "legal_action_index": selected_index,
                                }
                    if (
                        prefer_smith_when_rest_overheals
                        or prefer_smith_when_overheal_exceeds_effective_rest_heal
                    ) and kind == "rest":
                        selected_index, campfire_policy_intervention = (
                            _apply_campfire_overheal_override(
                                recommended_index,
                                kind,
                                descs,
                                gc,
                                sts,
                                armg_policy,
                                require_overheal_exceeds_effective_rest_heal=(
                                    prefer_smith_when_overheal_exceeds_effective_rest_heal
                                ),
                            )
                        )
                        if campfire_policy_intervention["overridden"]:
                            campfire_overheal_override_count += 1
                    card_reward_skip_threshold = (
                        skip_card_reward_when_deck_size_at_least
                        if skip_card_reward_when_deck_size_at_least is not None
                        else skip_duplicate_card_reward_when_deck_size_at_least
                    )
                    if card_reward_skip_threshold is not None:
                        selected_index, card_reward_skip_intervention = (
                            _apply_card_reward_skip_deck_threshold(
                                selected_index,
                                kind,
                                descs,
                                gc,
                                armg_policy,
                                deck_size_threshold=card_reward_skip_threshold,
                                require_recommended_duplicate=(
                                    skip_duplicate_card_reward_when_deck_size_at_least is not None
                                ),
                                require_all_options_duplicate=(
                                    require_all_card_reward_options_are_duplicates
                                ),
                            )
                        )
                        if card_reward_skip_intervention["eligible"]:
                            card_reward_skip_eligible_count += 1
                        if kind == "card":
                            reason = str(card_reward_skip_intervention["reason"])
                            card_reward_skip_reason_counts[reason] = (
                                card_reward_skip_reason_counts.get(reason, 0) + 1
                            )
                        if card_reward_skip_intervention["overridden"]:
                            card_reward_skip_override_count += 1
                    diagnostic_selected_index = selected_index
                    before = public_run_state(gc)
                    choice_descriptions = [repr(value) for value in descs]
                    choice_semantics = [armg_policy.describe_choice(kind, value) for value in descs]
                    deck_before = armg_policy.deck_snapshot(gc)
                    training_vector = armg_policy.training_vector_snapshot(gc, descs)
                    capture = getattr(armg_policy, "capture_conversion_state", None)
                    if callable(capture) and selected_index >= 0:
                        capture(
                            gc=gc,
                            sts=sts,
                            kind=kind,
                            selected_index=selected_index,
                            descs=descs,
                            scores=scores,
                        )
                    if selected_index < 0:
                        gc.skip_reward_cards()
                        choice = "armg:reward:skip_empty"
                    else:
                        choice = f"armg:{kind}:{selected_index}/{len(descs)}"
                        execs[selected_index](gc)
                    armg_action_count += 1
                    policy_name = "armg"
                    _record(evidence_path, {
                        "type": "armg_noncombat_decision_v3",
                        "floor": before.get("floor"),
                        "act": before.get("act"),
                        "gold_before": before.get("gold"),
                        "screen": screen_before,
                        "kind": kind,
                        "selected_index": selected_index,
                        "recommended_index": recommended_index,
                        "map_policy_intervention": map_policy_intervention,
                        "campfire_policy_intervention": campfire_policy_intervention,
                        "choice_count": len(descs),
                        "choice_descriptions": choice_descriptions,
                        "choice_semantics": choice_semantics,
                        "choice_scores": scores,
                        "obs_412": training_vector["obs_412"],
                        "candidate_desc_368": training_vector["candidate_desc_368"],
                        "selected_description": (
                            choice_descriptions[selected_index] if selected_index >= 0 else "skip_empty"
                        ),
                        "selected_semantics": (
                            armg_policy.describe_choice(kind, descs[selected_index]) if selected_index >= 0 else {"choice": "skip_empty"}
                        ),
                        "deck_before": deck_before,
                        "hp_before": _value(gc, "cur_hp"),
                        "max_hp_before": _value(gc, "max_hp"),
                    })
                    if diagnostic_trace_path is not None:
                        diagnostic_choices = [
                            {
                                "index": index,
                                "descriptor": choice_descriptions[index],
                                "semantics": choice_semantics[index],
                                "score": scores[index] if index < len(scores) else None,
                            }
                            for index in range(len(descs))
                        ]
                _record(evidence_path, {
                    "type": "simulator_noncombat",
                    "floor": int(_value(gc, "floor_num", 0) or 0),
                    "act": int(_value(gc, "act", 0) or 0),
                    "gold": int(_value(gc, "gold", 0) or 0),
                    "screen": screen_before,
                    "policy": policy_name,
                    "choice": choice,
                })
                if diagnostic_trace_path is not None:
                    _record(diagnostic_trace_path, {
                        "type": "noncombat_decision_trace_v1",
                        "game_step": game_steps,
                        "floor": int(_value(gc, "floor_num", 0) or 0),
                        "act": int(_value(gc, "act", 0) or 0),
                        "screen_before": screen_before,
                        "policy": policy_name,
                        "legal_choices_complete": diagnostic_choices is not None,
                        "legal_choices": diagnostic_choices,
                        "selected_legal_action_index": diagnostic_selected_index,
                        "recommended_legal_action_index": diagnostic_recommended_index,
                        "map_policy_intervention": map_policy_intervention,
                        "campfire_policy_intervention": campfire_policy_intervention,
                        "card_reward_skip_intervention": card_reward_skip_intervention,
                        "route": diagnostic_route,
                        "selected_choice": choice,
                        "state_before": diagnostic_run_before,
                        "state_after": _diagnostic_run_snapshot(gc, armg_policy),
                    })
                continue

            battle = sts.BattleContext()
            battle.init(gc)
            encounter_index += 1
            _record(diagnostic_trace_path, {
                "type": "encounter_started_v1",
                "game_step": game_steps,
                "encounter_index": encounter_index,
                "floor": int(_value(gc, "floor_num", 0) or 0),
                "act": int(_value(gc, "act", 0) or 0),
                "state": _diagnostic_battle_snapshot(battle, gc, armg_policy),
            })
            battle_steps = 0
            while battle.outcome == sts.Outcome.UNDECIDED and battle_steps < max_battle_steps:
                battle_steps += 1
                native_actions = list(sts.get_legal_actions(battle))
                if not native_actions:
                    raise SimulatorRunError("no native legal action in undecided battle")
                hand_raw = _sequence(_value(battle, "hand", []))
                public_actions, native_index_map, alias_count = _project_legal_actions(native_actions, hand_raw)
                equivalent_action_alias_count += alias_count
                if alias_count:
                    _record(evidence_path, {
                        "type": "simulator_equivalent_action_alias",
                        "floor": int(_value(gc, "floor_num", 0) or 0),
                        "alias_count": alias_count,
                        "native_action_count": len(native_actions),
                        "public_action_count": len(public_actions),
                    })
                if combat_mcts_sims is not None or consensus_budgets or hybrid_budgets:
                    teacher_public_state = None
                    hybrid_public_state = None
                    if collect_teacher or hybrid_budgets:
                        hybrid_or_teacher_state = adapter.adapt(
                            battle,
                            legal_actions=native_actions,
                            run_state=public_run_state(gc),
                            projected_legal_actions=public_actions,
                        )
                        if collect_teacher:
                            teacher_public_state = hybrid_or_teacher_state
                        if hybrid_budgets:
                            hybrid_public_state = hybrid_or_teacher_state
                    if collect_teacher and teacher_public_state is None:
                        teacher_public_state = adapter.adapt(
                            battle,
                            legal_actions=native_actions,
                            run_state=public_run_state(gc),
                            projected_legal_actions=public_actions,
                        )
                    floor_now = int(_value(gc, "floor_num", 0) or 0)
                    active_mcts_sims = _combat_mcts_budget_for_floor(
                        floor_now,
                        base_sims=combat_mcts_sims,
                        late_sims=combat_mcts_late_sims,
                        late_floor=combat_mcts_late_floor,
                        boss_sims=combat_mcts_boss_sims,
                        boss_floors=boss_floor_set,
                    )
                    started = time.perf_counter()
                    vote_bits: list[int] = []
                    vote_budgets: list[int] = []
                    hybrid_student_bits = None
                    hybrid_student_tiebreak_used = False
                    if len(native_actions) == 1:
                        chosen = native_actions[0]
                    elif hybrid_budgets:
                        if hybrid_public_state is None:
                            raise SimulatorRunError("hybrid MCTS public state was not captured")
                        recommendations: list[tuple[int, Any, int]] = []
                        for budget in hybrid_budgets:
                            recommendation = sts.mcts_recommend(battle, budget)
                            if recommendation is None:
                                raise SimulatorRunError(
                                    f"hybrid MCTS returned no recommendation for budget={budget}"
                                )
                            bits = _value(recommendation, "bits")
                            if not isinstance(bits, int):
                                raise SimulatorRunError(
                                    "hybrid MCTS recommendation did not expose integer action bits"
                                )
                            recommendations.append((budget, recommendation, bits))
                            vote_budgets.append(budget)
                            vote_bits.append(bits)

                        decision = student.select_action(
                            hybrid_public_state,
                            require_command=False,
                        )
                        public_index = int(decision.action_index)
                        if public_index < 0 or public_index >= len(native_index_map):
                            raise SimulatorRunError("hybrid Student vote index is outside legal actions")
                        student_native = native_actions[native_index_map[public_index]]
                        hybrid_student_bits = _value(student_native, "bits")
                        if not isinstance(hybrid_student_bits, int):
                            raise SimulatorRunError("hybrid Student vote has no integer action bits")
                        chosen_bits_vote, hybrid_student_tiebreak_used = _hybrid_vote_choice_bits(
                            [(budget, bits) for budget, _, bits in recommendations],
                            student_bits=hybrid_student_bits,
                        )
                        chosen = next(
                            recommendation
                            for _, recommendation, bits in reversed(recommendations)
                            if bits == chosen_bits_vote
                        )
                        hybrid_student_vote_count += 1
                        hybrid_student_tiebreak_count += int(hybrid_student_tiebreak_used)
                    elif consensus_budgets:
                        recommendations: list[tuple[int, Any, int]] = []
                        for budget in consensus_budgets:
                            recommendation = sts.mcts_recommend(battle, budget)
                            if recommendation is None:
                                raise SimulatorRunError(f"MCTS returned no recommendation for budget={budget}")
                            bits = _value(recommendation, "bits")
                            if not isinstance(bits, int):
                                raise SimulatorRunError("MCTS recommendation did not expose integer action bits")
                            recommendations.append((budget, recommendation, bits))
                            vote_budgets.append(budget)
                            vote_bits.append(bits)
                        counts: dict[int, int] = {}
                        for bits in vote_bits:
                            counts[bits] = counts.get(bits, 0) + 1
                        best_count = max(counts.values())
                        tied = {bits for bits, count in counts.items() if count == best_count}
                        # If all budgets disagree, prefer the highest-budget recommendation.
                        chosen = next(
                            recommendation
                            for _, recommendation, bits in reversed(recommendations)
                            if bits in tied
                        )
                    else:
                        if combat_mcts_exploration is None:
                            chosen = sts.mcts_recommend(battle, active_mcts_sims)
                        else:
                            tuned = getattr(sts, "mcts_recommend_tuned", None)
                            if not callable(tuned):
                                raise SimulatorRunError("experimental tuned MCTS binding is unavailable")
                            chosen = tuned(battle, active_mcts_sims, combat_mcts_exploration)
                        if chosen is None:
                            chosen = native_actions[0]
                    mcts_recommended_action = chosen
                    potion_reserve_override = False
                    emergency_potion_override = False
                    lethal_potion_rescue_override = False
                    lethal_potion_rescue_reason = "disabled"
                    if reserve_last_potion_until_floor is not None:
                        player = _value(battle, "player")
                        player_hp_raw = _value(player, "cur_hp", None)
                        player_block_raw = _value(player, "block", None)
                        player_hp = (
                            int(player_hp_raw)
                            if isinstance(player_hp_raw, (int, float))
                            and not isinstance(player_hp_raw, bool)
                            else None
                        )
                        player_block = (
                            int(player_block_raw)
                            if isinstance(player_block_raw, (int, float))
                            and not isinstance(player_block_raw, bool)
                            else None
                        )
                        incoming_damage: int | None = 0
                        for enemy_index, enemy in enumerate(
                            _sequence(_value(battle, "monsters", []))
                        ):
                            enemy_state = _enemy(enemy, index=enemy_index, battle=battle)
                            enemy_hp = enemy_state.get("hp")
                            if enemy_state.get("is_gone") is True or (
                                isinstance(enemy_hp, (int, float)) and enemy_hp <= 0
                            ):
                                continue
                            damage = enemy_state.get("intent_damage")
                            hits = enemy_state.get("intent_hits")
                            if (
                                not isinstance(damage, (int, float))
                                or isinstance(damage, bool)
                                or not isinstance(hits, (int, float))
                                or isinstance(hits, bool)
                            ):
                                incoming_damage = None
                                break
                            incoming_damage += max(0, int(damage)) * max(1, int(hits))
                        chosen, potion_reserve_override = _apply_last_potion_reserve(
                            mcts_recommended_action,
                            native_actions,
                            hand=hand_raw,
                            floor=floor_now,
                            reserve_until_floor=reserve_last_potion_until_floor,
                            player_hp=player_hp,
                            player_block=player_block,
                            incoming_damage=incoming_damage,
                        )
                        if potion_reserve_override:
                            potion_reserve_override_count += 1
                    if use_potion_below_hp_fraction is not None:
                        player = _value(battle, "player")
                        player_hp_raw = _value(player, "cur_hp", None)
                        player_max_hp_raw = _value(player, "max_hp", None)
                        player_hp = (
                            int(player_hp_raw)
                            if isinstance(player_hp_raw, (int, float))
                            and not isinstance(player_hp_raw, bool)
                            and math.isfinite(float(player_hp_raw))
                            else None
                        )
                        player_max_hp = (
                            int(player_max_hp_raw)
                            if isinstance(player_max_hp_raw, (int, float))
                            and not isinstance(player_max_hp_raw, bool)
                            and math.isfinite(float(player_max_hp_raw))
                            else None
                        )
                        chosen, emergency_potion_override = _apply_low_hp_emergency_potion(
                            mcts_recommended_action,
                            native_actions,
                            hand=hand_raw,
                            player_hp=player_hp,
                            player_max_hp=player_max_hp,
                            hp_ratio_threshold=use_potion_below_hp_fraction,
                        )
                        if emergency_potion_override:
                            emergency_potion_override_count += 1
                    if lethal_potion_rescue:
                        (
                            chosen,
                            lethal_potion_rescue_override,
                            lethal_potion_rescue_reason,
                        ) = _apply_lethal_potion_rescue(
                            mcts_recommended_action,
                            native_actions,
                            hand=hand_raw,
                            battle=battle,
                            game_context=gc,
                            sts=sts,
                        )
                        lethal_potion_rescue_reason_counts[
                            lethal_potion_rescue_reason
                        ] = lethal_potion_rescue_reason_counts.get(
                            lethal_potion_rescue_reason, 0
                        ) + 1
                        if lethal_potion_rescue_override:
                            lethal_potion_rescue_override_count += 1
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    latencies_ms.append(latency_ms)
                    chosen_bits = _value(chosen, "bits")
                    if collect_teacher:
                        if teacher_public_state is None:
                            raise SimulatorRunError("MCTS Teacher public state was not captured")
                        chosen_public = _public_action(chosen, hand_raw)
                        chosen_identity = canonical_json(chosen_public)
                        matches = [
                            index
                            for index, action in enumerate(public_actions)
                            if canonical_json(action) == chosen_identity
                        ]
                        if len(matches) != 1:
                            raise SimulatorRunError(
                                f"MCTS Teacher action mapping is not unique: matches={matches}"
                            )
                        teacher_index = matches[0]
                        teacher_action_id = sha256_json(
                            normalize_action_payload(public_actions[teacher_index])
                        )
                        _record(evidence_path, {
                            "type": "mcts_teacher_decision",
                            "floor": floor_now,
                            "public_state": teacher_public_state,
                            "teacher_action_index": teacher_index,
                            "teacher_action_id": teacher_action_id,
                            "mcts_sims": active_mcts_sims,
                            "mcts_budgets": list(consensus_budgets),
                            "chosen_bits": chosen_bits,
                        })
                    # Record a human-readable, ordered combat trace before executing
                    # the action. This preserves the exact card name while it is still
                    # present in hand and makes winner/near-win play sequences auditable.
                    trace_action = _public_action(chosen, hand_raw)
                    trace_card = None
                    trace_source_idx = _value(chosen, "source_idx", -1)
                    if (
                        _action_type(chosen) == "CARD"
                        and isinstance(trace_source_idx, int)
                        and 0 <= trace_source_idx < len(hand_raw)
                    ):
                        trace_card = _card(hand_raw[trace_source_idx], position=trace_source_idx + 1)
                    trace_player = _value(battle, "player")
                    if diagnostic_trace_path is not None:
                        diagnostic_run_state = _diagnostic_run_snapshot(gc, armg_policy)
                        diagnostic_run_state["room"] = "COMBAT"
                        diagnostic_state = adapter.adapt(
                            battle,
                            legal_actions=native_actions,
                            run_state=diagnostic_run_state,
                            projected_legal_actions=public_actions,
                        )
                        diagnostic_state["usable_potion_slots"] = sorted(
                            _usable_potion_slots(native_actions, hand_raw)
                        )
                        diagnostic_state["usable_potion_slots_complete"] = True
                        diagnostic_state["usable_potion_slots_source"] = (
                            "canonical_native_legal_actions"
                        )
                        diagnostic_state["potion_inventory_complete"] = bool(
                            diagnostic_run_state.get("potion_inventory_complete", False)
                        )
                        diagnostic_state["potions"] = diagnostic_run_state.get("potions")
                        diagnostic_state["potion_inventory_source"] = diagnostic_run_state.get(
                            "potion_inventory_source"
                        )
                        diagnostic_state["potion_inventory_reason"] = diagnostic_run_state.get(
                            "potion_inventory_reason"
                        )
                        selected_public_matches = [
                            index
                            for index, action in enumerate(public_actions)
                            if canonical_json(action) == canonical_json(trace_action)
                        ]
                        chosen_native_matches = [
                            index
                            for index, action in enumerate(native_actions)
                            if _value(action, "bits") == _value(chosen, "bits")
                        ]
                        chosen_native_index = (
                            chosen_native_matches[0]
                            if len(chosen_native_matches) == 1
                            else None
                        )
                        if chosen_native_index is None and len(selected_public_matches) == 1:
                            public_index = selected_public_matches[0]
                            if public_index < len(native_index_map):
                                chosen_native_index = native_index_map[public_index]
                        if len(selected_public_matches) != 1 or chosen_native_index is None:
                            diagnostic_legal_actions_complete = False
                        _record(diagnostic_trace_path, {
                            "type": "combat_decision_trace_v1",
                            "game_step": game_steps,
                            "encounter_index": encounter_index,
                            "battle_step": battle_steps,
                            "floor": floor_now,
                            "act": int(_value(gc, "act", 0) or 0),
                            "turn": _value(battle, "turn"),
                            "mcts_sims": active_mcts_sims,
                            "public_state": diagnostic_state,
                            "canonical_native_legal_actions": [
                                _public_action(action, hand_raw) for action in native_actions
                            ],
                            "policy_legal_actions": public_actions,
                            "legal_actions_complete": True,
                            "legal_action_alias_count": alias_count,
                            "mcts_recommended_action": _public_action(
                                mcts_recommended_action, hand_raw
                            ),
                            "potion_reserve_override": potion_reserve_override,
                            "emergency_potion_override": emergency_potion_override,
                            "lethal_potion_rescue_override": lethal_potion_rescue_override,
                            "lethal_potion_rescue_reason": lethal_potion_rescue_reason,
                            "selected_action": trace_action,
                            "selected_public_action_index": (
                                selected_public_matches[0] if len(selected_public_matches) == 1 else None
                            ),
                            "selected_native_action_index": chosen_native_index,
                        })
                    _record(evidence_path, {
                        "type": "combat_play_trace_v2",
                        "floor": floor_now,
                        "act": int(_value(gc, "act", 0) or 0),
                        "turn": _value(battle, "turn"),
                        "step": battle_steps,
                        "hp_before": _value(trace_player, "cur_hp"),
                        "block_before": _value(trace_player, "block"),
                        "energy_before": _value(trace_player, "energy"),
                        "action": trace_action,
                        "mcts_recommended_action": _public_action(
                            mcts_recommended_action, hand_raw
                        ),
                        "potion_reserve_override": potion_reserve_override,
                        "emergency_potion_override": emergency_potion_override,
                        "lethal_potion_rescue_override": lethal_potion_rescue_override,
                        "lethal_potion_rescue_reason": lethal_potion_rescue_reason,
                        "action_type": _action_type(chosen),
                        "card": trace_card,
                        "target_index": _value(chosen, "target_idx", -1),
                        "hand_before": [_card(card, position=i + 1) for i, card in enumerate(hand_raw)],
                        "enemies_before": [
                            _enemy(enemy, index=i, battle=battle)
                            for i, enemy in enumerate(_sequence(_value(battle, "monsters", [])))
                        ],
                        "mcts_sims": active_mcts_sims,
                        "potion_reserve_override": potion_reserve_override,
                        "emergency_potion_override": emergency_potion_override,
                    })
                    chosen.execute(battle)
                    mcts_action_count += 1
                    _record(evidence_path, {
                        "type": "simulator_combat_action",
                        "floor": int(_value(gc, "floor_num", 0) or 0),
                        "policy": (
                            "hybrid_mcts_student_tiebreak"
                            if hybrid_budgets
                            else ("mcts_consensus" if consensus_budgets else "mcts")
                        ),
                        "mcts_sims": active_mcts_sims,
                        "mcts_base_sims": combat_mcts_sims,
                        "mcts_late_sims": combat_mcts_late_sims,
                        "mcts_late_floor": combat_mcts_late_floor if combat_mcts_late_sims is not None else None,
                        "mcts_exploration": combat_mcts_exploration,
                        "mcts_budgets": list(consensus_budgets),
                        "hybrid_mcts_budgets": list(hybrid_budgets),
                        "student_vote_bits": hybrid_student_bits,
                        "student_tiebreak_used": hybrid_student_tiebreak_used,
                        "vote_budgets": vote_budgets,
                        "vote_bits": vote_bits,
                        "chosen_bits": chosen_bits,
                        "inference_latency_ms": latency_ms,
                    })
                else:
                    public_state = adapter.adapt(
                        battle,
                        legal_actions=native_actions,
                        run_state=public_run_state(gc),
                        projected_legal_actions=public_actions,
                    )
                    started = time.perf_counter()
                    if collect_ppo:
                        decision = student.sample_action(
                            public_state,
                            deterministic=False,
                            require_command=False,
                        )
                    else:
                        decision = student.select_action(public_state, require_command=False)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    latencies_ms.append(latency_ms)
                    if not 0 <= decision.action_index < len(native_index_map):
                        illegal_actions += 1
                        raise SimulatorRunError("Student selected public action index outside legal range")
                    expected_id = sha256_json(
                        normalize_action_payload(public_state["legal_actions"][decision.action_index])
                    )
                    if decision.action_id != expected_id:
                        illegal_actions += 1
                        raise SimulatorRunError("Student/native action identity mapping drift")
                    native_action_index = native_index_map[decision.action_index]
                    native_actions[native_action_index].execute(battle)
                    student_actions += 1
                    record = {
                        "type": "simulator_combat_action",
                        "floor": int(_value(gc, "floor_num", 0) or 0),
                        "policy": "student_v1_ppo" if collect_ppo else "student_v0",
                        "decision_signature": public_state["decision_signature"],
                        "action_id": decision.action_id,
                        "action_index": decision.action_index,
                        "native_action_index": native_action_index,
                        "student_score": decision.score,
                        "inference_latency_ms": latency_ms,
                    }
                    if collect_ppo:
                        record.update({
                            "type": "ppo_decision",
                            "public_state": public_state,
                            "old_log_prob": float(decision.log_prob),
                            "old_value": float(decision.value),
                        })
                    _record(evidence_path, record)
            if battle.outcome == sts.Outcome.UNDECIDED:
                timeout_count += 1
                raise SimulatorRunError(f"battle step bound reached: {max_battle_steps}")
            _record(diagnostic_trace_path, {
                "type": "encounter_finished_v1",
                "game_step": game_steps,
                "encounter_index": encounter_index,
                "floor": int(_value(gc, "floor_num", 0) or 0),
                "act": int(_value(gc, "act", 0) or 0),
                "battle_outcome": _enum_name(_value(battle, "outcome", "UNKNOWN")),
                "state": _diagnostic_battle_snapshot(battle, gc, armg_policy),
            })
            battle.exit_battle(gc)

        if gc.outcome == sts.GameOutcome.UNDECIDED:
            timeout_count += 1
            raise SimulatorRunError(f"game step bound reached: {max_game_steps}")
    except Exception as exc:
        crash_count = 1
        error = f"{type(exc).__name__}: {exc}"

    if error is not None:
        outcome = "unknown"
        result = "BLOCKED_SIMULATOR"
    elif gc.outcome == sts.GameOutcome.PLAYER_VICTORY:
        outcome = "victory"
        result = "PASS_SIMULATOR_COMPLETE_RUN"
    elif gc.outcome == sts.GameOutcome.PLAYER_LOSS:
        outcome = "defeat"
        result = "PASS_SIMULATOR_COMPLETE_RUN"
    else:
        outcome = "unknown"
        result = "BLOCKED_SIMULATOR_UNKNOWN_OUTCOME"

    potion_inventory_snapshot = _potion_inventory_snapshot(gc)
    summary = {
        "schema_version": SIMULATOR_EVIDENCE_SCHEMA,
        "phase_protocol": A0_PROTOCOL_VERSION,
        "result": result,
        "error": error,
        "seed": ui_seed,
        "simulator_seed_long": simulator_seed,
        "seed_contract": seed_contract,
        "outcome": outcome,
        "final_floor": int(_value(gc, "floor_num", 0) or 0),
        "max_floor": max_floor,
        "max_act": max_act,
        "final_hp": int(_value(gc, "cur_hp", 0) or 0),
        "student_action_count": student_actions,
        "fallback_count": fallback_count,
        "armg_action_count": armg_action_count,
        "mcts_action_count": mcts_action_count,
        "hybrid_student_vote_count": hybrid_student_vote_count,
        "hybrid_student_tiebreak_count": hybrid_student_tiebreak_count,
        "combat_policy": (
            "hybrid_mcts_" + "_".join(str(value) for value in hybrid_budgets) + "_student_tiebreak"
            if hybrid_budgets
            else (
                "mcts_consensus_" + "_".join(str(value) for value in consensus_budgets)
                if consensus_budgets
                else (
                f"mcts_schedule_{combat_mcts_sims}_to_{combat_mcts_late_sims}_floor_{combat_mcts_late_floor}"
                if combat_mcts_late_sims is not None
                else (
                    f"mcts_{combat_mcts_sims}_explore_{combat_mcts_exploration:g}"
                    if combat_mcts_exploration is not None
                    else (
                        f"mcts_{combat_mcts_sims}"
                        if combat_mcts_sims is not None
                        else (
                            "student_v1_ppo_rollout"
                            if collect_ppo
                            else (
                                "student_v1_ppo_deterministic"
                                if callable(getattr(student, "sample_action", None))
                                else "student_v0"
                            )
                        )
                    )
                )
            )
            )
        ),
        "noncombat_policy": "armg" if armg_policy is not None else "legacy_fallback",
        "potion_inventory_snapshot_complete": potion_inventory_snapshot[
            "potion_inventory_complete"
        ],
        "potion_inventory_source": potion_inventory_snapshot["potion_inventory_source"],
        "potion_inventory_reason": potion_inventory_snapshot["potion_inventory_reason"],
        "potion_reserve_until_floor": reserve_last_potion_until_floor,
        "potion_reserve_override_count": potion_reserve_override_count,
        "use_potion_below_hp_fraction": use_potion_below_hp_fraction,
        "emergency_potion_override_count": emergency_potion_override_count,
        "lethal_potion_rescue_enabled": lethal_potion_rescue,
        "lethal_potion_rescue_override_count": lethal_potion_rescue_override_count,
        "lethal_potion_rescue_reason_counts": lethal_potion_rescue_reason_counts,
        "skip_card_reward_when_deck_size_at_least": (
            skip_card_reward_when_deck_size_at_least
        ),
        "skip_duplicate_card_reward_when_deck_size_at_least": (
            skip_duplicate_card_reward_when_deck_size_at_least
        ),
        "require_all_card_reward_options_are_duplicates": (
            require_all_card_reward_options_are_duplicates
        ),
        "card_reward_skip_eligible_count": card_reward_skip_eligible_count,
        "card_reward_skip_override_count": card_reward_skip_override_count,
        "card_reward_skip_reason_counts": card_reward_skip_reason_counts,
        "prefer_smith_when_rest_overheals": prefer_smith_when_rest_overheals,
        "prefer_smith_when_overheal_exceeds_effective_rest_heal": (
            prefer_smith_when_overheal_exceeds_effective_rest_heal
        ),
        "campfire_overheal_override_count": campfire_overheal_override_count,
        "avoid_low_hp_elite_routes": avoid_low_hp_elite_routes,
        "map_elite_avoidance_override_count": map_elite_avoidance_override_count,
        "map_elite_avoidance_fail_closed_count": map_elite_avoidance_fail_closed_count,
        "fallback_rate": fallback_count / max(1, student_actions + fallback_count + armg_action_count),
        "equivalent_action_alias_count": equivalent_action_alias_count,
        "illegal_action_count": illegal_actions,
        "timeout_count": timeout_count,
        "crash_count": crash_count,
        "mean_inference_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
        "max_inference_latency_ms": max(latencies_ms) if latencies_ms else None,
        "game_steps": game_steps,
        "ppo_collection": collect_ppo,
        "teacher_collection": collect_teacher,
    }
    _record(evidence_path, {"type": "summary", **summary})
    _record(diagnostic_trace_path, {
        "type": "terminal_trace_v1",
        "complete": error is None and outcome in {"victory", "defeat"},
        "outcome": outcome,
        "result": result,
        "error": error,
        "legal_actions_complete": diagnostic_legal_actions_complete,
        "illegal_action_count": illegal_actions,
        "timeout_count": timeout_count,
        "crash_count": crash_count,
        "potion_inventory_snapshot_complete": summary[
            "potion_inventory_snapshot_complete"
        ],
        "potion_inventory_source": summary["potion_inventory_source"],
        "potion_inventory_reason": summary["potion_inventory_reason"],
        "lethal_potion_rescue_enabled": lethal_potion_rescue,
        "lethal_potion_rescue_override_count": lethal_potion_rescue_override_count,
        "lethal_potion_rescue_reason_counts": lethal_potion_rescue_reason_counts,
        "final_floor": summary["final_floor"],
        "final_act": int(_value(gc, "act", 0) or 0),
        "final_hp": summary["final_hp"],
        "final_state": _diagnostic_run_snapshot(gc, armg_policy),
    })
    return summary


def _record(path: Path | None, value: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")


def _load_sts(module_dir: Path) -> Any:
    if not module_dir.is_dir():
        raise SimulatorRunError(f"simulator module directory missing: {module_dir}")
    sys.path.insert(0, str(module_dir))
    try:
        return importlib.import_module("slaythespire")
    except Exception as exc:
        raise SimulatorRunError(f"could not import pinned slaythespire module: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument(
        "--heldout-seed-file",
        type=Path,
        help="explicit allowlist of exactly 50 numeric internal seeds for held-out evaluation",
    )
    parser.add_argument(
        "--training-seed-file",
        type=Path,
        help="training-only numeric seed allowlist; must not overlap the held-out evaluation set",
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--max-game-steps", type=int, default=MAX_GAME_STEPS)
    parser.add_argument("--max-battle-steps", type=int, default=MAX_BATTLE_STEPS)
    parser.add_argument("--armg-root", type=Path)
    parser.add_argument("--armg-weight", type=Path)
    parser.add_argument("--combat-mcts-sims", type=int)
    parser.add_argument("--combat-mcts-late-sims", type=int)
    parser.add_argument("--combat-mcts-late-floor", type=int, default=50)
    parser.add_argument("--combat-mcts-exploration", type=float)
    parser.add_argument("--student-v1-checkpoint", type=Path)
    parser.add_argument("--collect-ppo", action="store_true")
    parser.add_argument("--collect-teacher", action="store_true")
    parser.add_argument(
        "--hybrid-mcts-budgets",
        help="comma-separated MCTS budgets; Student breaks ties between MCTS recommendations",
    )
    parser.add_argument(
        "--combat-mcts-budgets",
        help="comma-separated MCTS budgets; majority vote with highest-budget tie-break",
    )
    args = parser.parse_args(argv)
    if (args.armg_root is None) != (args.armg_weight is None):
        parser.error("--armg-root and --armg-weight must be supplied together")
    hybrid_mcts_budgets: tuple[int, ...] = ()
    if args.hybrid_mcts_budgets:
        try:
            hybrid_mcts_budgets = tuple(
                int(value.strip())
                for value in args.hybrid_mcts_budgets.split(",")
                if value.strip()
            )
        except ValueError as exc:
            parser.error(f"invalid --hybrid-mcts-budgets: {exc}")
        if len(hybrid_mcts_budgets) < 2:
            parser.error("--hybrid-mcts-budgets requires at least two positive integers")

    combat_mcts_budgets: tuple[int, ...] = ()
    if args.combat_mcts_budgets:
        try:
            combat_mcts_budgets = tuple(
                int(value.strip()) for value in args.combat_mcts_budgets.split(",") if value.strip()
            )
        except ValueError as exc:
            parser.error(f"invalid --combat-mcts-budgets: {exc}")
        if not combat_mcts_budgets:
            parser.error("--combat-mcts-budgets must contain at least one positive integer")
    if args.heldout_seed_file is not None and args.training_seed_file is not None:
        parser.error("--heldout-seed-file and --training-seed-file are mutually exclusive")

    heldout_seeds: tuple[int, ...] | None = None
    if args.heldout_seed_file is not None:
        if not args.heldout_seed_file.is_file():
            parser.error(f"held-out seed file missing: {args.heldout_seed_file}")
        try:
            heldout_seeds = tuple(
                int(line.strip())
                for line in args.heldout_seed_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        except ValueError as exc:
            parser.error(f"invalid held-out seed file: {exc}")
        if len(heldout_seeds) != 50 or len(set(heldout_seeds)) != 50:
            parser.error("held-out seed file must contain exactly 50 unique numeric seeds")

    training_seeds: tuple[int, ...] | None = None
    if args.training_seed_file is not None:
        if not args.training_seed_file.is_file():
            parser.error(f"training seed file missing: {args.training_seed_file}")
        try:
            training_seeds = tuple(
                int(line.strip())
                for line in args.training_seed_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        except ValueError as exc:
            parser.error(f"invalid training seed file: {exc}")
        if not training_seeds or len(set(training_seeds)) != len(training_seeds):
            parser.error("training seed file must contain one or more unique numeric seeds")
        if any(seed < 1 or seed > 10**9 for seed in training_seeds):
            parser.error("training seeds must be in the upstream-compatible range 1..1e9")
        if not (args.collect_ppo or args.collect_teacher):
            parser.error("--training-seed-file requires --collect-ppo or --collect-teacher")

    baseline_student = FrozenStudentV0.from_path(args.model)
    if args.student_v1_checkpoint is not None:
        from .student_v1_ppo import StudentV1PPO
        student = StudentV1PPO.load(args.student_v1_checkpoint, baseline_student)
    else:
        student = baseline_student
    if args.collect_ppo and args.student_v1_checkpoint is None:
        parser.error("--collect-ppo requires --student-v1-checkpoint")
    sts = _load_sts(args.module_dir)
    armg_policy = None
    if args.armg_root is not None and args.armg_weight is not None:
        armg_policy = ArmGNoncombatPolicy(root=args.armg_root, weight_path=args.armg_weight)
    summary = run_simulator_game(
        student=student,
        sts=sts,
        seed=args.seed,
        evidence_path=args.evidence,
        max_game_steps=args.max_game_steps,
        max_battle_steps=args.max_battle_steps,
        armg_policy=armg_policy,
        combat_mcts_sims=args.combat_mcts_sims,
        combat_mcts_budgets=combat_mcts_budgets,
        hybrid_mcts_budgets=hybrid_mcts_budgets,
        combat_mcts_late_sims=args.combat_mcts_late_sims,
        combat_mcts_late_floor=args.combat_mcts_late_floor,
        combat_mcts_exploration=args.combat_mcts_exploration,
        heldout_seeds=heldout_seeds,
        training_seeds=training_seeds,
        collect_ppo=args.collect_ppo,
        collect_teacher=args.collect_teacher,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["result"] == "PASS_SIMULATOR_COMPLETE_RUN" else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAX_BATTLE_STEPS", "MAX_GAME_STEPS", "SIMULATOR_EVIDENCE_SCHEMA",
    "ArmGNoncombatPolicy", "SimulatorCombatAdapter", "SimulatorRunError",
    "deterministic_noncombat_step", "public_run_state", "resolve_simulator_seed",
    "run_simulator_game",
]
