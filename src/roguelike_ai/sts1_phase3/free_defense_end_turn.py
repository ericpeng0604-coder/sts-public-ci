"""H21: a bounded zero-cost defensive continuation of an MCTS End Turn."""
from collections.abc import Mapping, Sequence
from typing import Any

from .inflame_end_turn import InflameChoice, _integer, _powers_known

PLAYER_POWERS = frozenset({"Strength", "Dexterity", "Weak", "Vulnerable", "Frail"})
ENEMY_POWERS = frozenset({"Strength", "Weak", "Vulnerable"})
EXCLUDED_ENEMIES = frozenset({"GREMLIN_NOB", "TIME_EATER", "CORRUPT_HEART"})
EXCLUDED_RELICS = frozenset({"VELVET_CHOKER", "DEAD_BRANCH", "INK_BOTTLE", "UNCEASING_TOP", "CHARONS_ASHES"})
CARDS = frozenset({"DEFEND_RED", "GOOD_INSTINCTS", "INTIMIDATE"})


def select_free_defense_before_end_turn(
    state: Mapping[str, Any], legal_actions: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any], *, legal_actions_complete: bool,
) -> tuple[InflameChoice | None, str]:
    if legal_actions_complete is not True:
        return None, "incomplete_legal_actions"
    if not isinstance(state, Mapping) or not isinstance(recommendation, Mapping):
        return None, "malformed_state"
    if not isinstance(legal_actions, (list, tuple)) or not all(isinstance(a, Mapping) for a in legal_actions):
        return None, "malformed_legal_actions"
    if recommendation.get("kind") != "end_turn":
        return None, "recommendation_not_end_turn"
    if sum(a == recommendation for a in legal_actions) != 1:
        return None, "recommendation_not_unique_legal_action"
    if (state.get("combat_active") is not True or not _integer(state.get("energy"))
            or state["energy"] < 0 or not _integer(state.get("block")) or state["block"] < 0):
        return None, "malformed_resources"
    if not _powers_known(state.get("powers"), PLAYER_POWERS):
        return None, "unknown_player_power"
    relics = state.get("relics")
    if not isinstance(relics, list) or any(not isinstance(r, str) for r in relics):
        return None, "malformed_relics"
    if EXCLUDED_RELICS.intersection(relics):
        return None, "relic_card_trigger"
    hand, enemies = state.get("hand"), state.get("enemies")
    if not isinstance(hand, list) or not isinstance(enemies, list) or not enemies:
        return None, "malformed_combat_state"
    positions = set()
    for card in hand:
        if (not isinstance(card, Mapping) or not isinstance(card.get("id"), str)
                or not _integer(card.get("position")) or card["position"] < 1
                or card["position"] in positions):
            return None, "ambiguous_hand"
        positions.add(card["position"])
        if card["id"] in {"PAIN", "NORMALITY"}:
            return None, "hand_curse_penalty"
    incoming = unweakened = 0
    for enemy in enemies:
        if (not isinstance(enemy, Mapping) or not _integer(enemy.get("hp"))
                or not isinstance(enemy.get("is_gone"), bool)):
            return None, "malformed_enemy"
        if enemy["is_gone"] or enemy["hp"] <= 0:
            continue
        if not isinstance(enemy.get("name"), str) or enemy["name"] in EXCLUDED_ENEMIES:
            return None, "enemy_card_trigger"
        if not _powers_known(enemy.get("powers"), ENEMY_POWERS):
            return None, "unknown_enemy_power"
        damage, hits = enemy.get("intent_damage"), enemy.get("intent_hits")
        if not _integer(damage) or not _integer(hits) or damage < 0 or hits < 0:
            return None, "malformed_intent"
        incoming += damage * hits
        if not any(p["name"] == "Weak" and p["amount"] > 0 for p in enemy["powers"]):
            unweakened += damage * hits
    if incoming <= state["block"]:
        return None, "no_unblocked_intent"
    choices = []
    for card in hand:
        if (card["id"] not in CARDS or card.get("cost") != 0 or not _integer(card.get("cost"))
                or card.get("is_playable") is not True or card.get("has_target") is not False
                or not _integer(card.get("upgrades")) or card["upgrades"] not in (0, 1)):
            continue
        if card["id"] == "INTIMIDATE" and not unweakened:
            continue
        matches = [(i, a) for i, a in enumerate(legal_actions)
                   if a.get("kind") == "play_card" and _integer(a.get("hand_index"))
                   and a["hand_index"] == card["position"]]
        if matches and any(a != matches[0][1] for _, a in matches[1:]):
            return None, "ambiguous_defense_action"
        if matches:
            index, action = matches[0]
            choices.append((card["position"], InflameChoice(index, dict(action))))
    if not choices:
        return None, "no_eligible_free_defense"
    return min(choices, key=lambda x: x[0])[1], "legal_free_defense_before_end_turn"
