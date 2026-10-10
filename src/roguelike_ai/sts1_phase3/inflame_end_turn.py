"""Frozen H20 public-state hypothesis; not enabled in the game runner."""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple


ENEMY_POWER_ALLOWLIST = frozenset({"Strength", "Weak", "Vulnerable"})
PLAYER_POWER_ALLOWLIST = frozenset({"Strength"})
EXCLUDED_ENEMIES = frozenset({"AWAKENED_ONE", "TIME_EATER", "CORRUPT_HEART"})
EXCLUDED_CARDS = frozenset({"PAIN", "NORMALITY"})


class InflameChoice(NamedTuple):
    native_action_index: int
    action: dict[str, Any]


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _powers_known(value: Any, allowed: frozenset[str]) -> bool:
    return isinstance(value, list) and all(
        isinstance(power, Mapping)
        and isinstance(power.get("name"), str)
        and power.get("name") in allowed
        and _integer(power.get("amount"))
        for power in value
    )


def select_inflame_before_end_turn(
    state: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any],
    *,
    legal_actions_complete: bool,
) -> tuple[InflameChoice | None, str]:
    """Return an exact legal action or abstain; never mutate caller evidence."""
    if legal_actions_complete is not True:
        return None, "incomplete_legal_actions"
    if not isinstance(state, Mapping) or not isinstance(recommendation, Mapping):
        return None, "malformed_state"
    if not isinstance(legal_actions, (list, tuple)) or not all(
        isinstance(action, Mapping) for action in legal_actions
    ):
        return None, "malformed_legal_actions"
    if recommendation.get("kind") != "end_turn":
        return None, "recommendation_not_end_turn"
    if sum(action == recommendation for action in legal_actions) != 1:
        return None, "recommendation_not_unique_legal_action"
    energy = state.get("energy")
    hand, enemies = state.get("hand"), state.get("enemies")
    if state.get("combat_active") is not True or not _integer(energy) or energy < 0:
        return None, "malformed_combat_resources"
    if not isinstance(hand, list) or not isinstance(enemies, list) or not enemies:
        return None, "malformed_combat_state"
    if not _powers_known(state.get("powers"), PLAYER_POWER_ALLOWLIST):
        return None, "unknown_player_power"
    positions: set[int] = set()
    for card in hand:
        if not isinstance(card, Mapping) or not isinstance(card.get("id"), str):
            return None, "malformed_hand"
        position = card.get("position")
        if not _integer(position) or position < 0 or position in positions:
            return None, "ambiguous_hand_position"
        positions.add(position)
        if card["id"] in EXCLUDED_CARDS:
            return None, "hand_curse_penalty"
    living = 0
    for enemy in enemies:
        if not isinstance(enemy, Mapping) or not _integer(enemy.get("hp")):
            return None, "malformed_enemy"
        if not isinstance(enemy.get("is_gone"), bool):
            return None, "malformed_enemy"
        if enemy["is_gone"] or enemy["hp"] <= 0:
            continue
        living += 1
        if not isinstance(enemy.get("name"), str) or enemy["name"] in EXCLUDED_ENEMIES:
            return None, "enemy_card_or_power_penalty"
        if not _powers_known(enemy.get("powers"), ENEMY_POWER_ALLOWLIST):
            return None, "unknown_enemy_power"
    if not living:
        return None, "no_living_enemy"
    candidates: list[tuple[int, InflameChoice]] = []
    for card in hand:
        if card["id"] != "INFLAME":
            continue
        cost = card.get("cost")
        if (
            card.get("is_playable") is not True
            or card.get("has_target") is not False
            or not _integer(cost) or cost < 0 or cost > energy
            or not _integer(card.get("upgrades")) or card["upgrades"] not in (0, 1)
        ):
            continue
        matches = [(index, action) for index, action in enumerate(legal_actions) if (
            action.get("kind") == "play_card"
            and _integer(action.get("hand_index"))
            and action["hand_index"] == card["position"]
        )]
        if matches and any(action != matches[0][1] for _, action in matches[1:]):
            return None, "ambiguous_inflame_action"
        if matches:
            # Identical non-targeted projections retain an exact native ordinal.
            index, action = matches[0]
            candidates.append((card["position"], InflameChoice(index, dict(action))))
    if not candidates:
        return None, "no_eligible_legal_inflame"
    return min(candidates, key=lambda item: item[0])[1], "legal_inflame_before_end_turn"
