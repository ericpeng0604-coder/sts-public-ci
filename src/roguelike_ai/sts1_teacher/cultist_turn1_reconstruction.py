"""Fail-closed player-side proof for the narrow Cultist turn-1 boundary.

The admitted slice is intentionally tiny: Ironclad starter deck only, no card
played on turn 0, no player powers, and the first enemy turn was Cultist
Incantation. Enemy semantics are checked separately by enemy_reconstruction.
No source BattleContext, RNG state, or move history is accepted here.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_EXPECTED_STARTER_COUNTS = Counter({"STRIKE_RED": 5, "DEFEND_RED": 4, "BASH": 1})
_EXPECTED_COSTS = {"STRIKE_RED": 1, "DEFEND_RED": 1, "BASH": 2}
_CARD_ALIASES = {"STRIKE_R": "STRIKE_RED", "DEFEND_R": "DEFEND_RED"}


def _norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def _card_id(value: Any) -> str:
    normalized = _norm(value)
    return _CARD_ALIASES.get(normalized, normalized)


def _seq(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return None


def is_cultist_turn1_candidate(state: Mapping[str, Any]) -> bool:
    if state.get("turn") != 1:
        return False
    enemies = _seq(state.get("enemies"))
    return bool(
        enemies is not None
        and len(enemies) == 1
        and isinstance(enemies[0], Mapping)
        and _norm(enemies[0].get("name")) == "CULTIST"
    )


@dataclass(frozen=True)
class PublicCultistTurnOneAdmission:
    allowed: bool
    reasons: tuple[str, ...]


def assess_public_cultist_turn1_player(state: Mapping[str, Any]) -> PublicCultistTurnOneAdmission:
    reasons: list[str] = []
    if not is_cultist_turn1_candidate(state):
        reasons.append("cultist_turn1_not_candidate")
        return PublicCultistTurnOneAdmission(False, tuple(reasons))

    if state.get("energy") != 3 or isinstance(state.get("energy"), bool):
        reasons.append(f"cultist_turn1_energy_boundary:{state.get('energy')!r}")
    if state.get("block") != 0 or isinstance(state.get("block"), bool):
        reasons.append(f"cultist_turn1_block_boundary:{state.get('block')!r}")

    powers = _seq(state.get("powers"))
    if powers is None:
        reasons.append("cultist_turn1_player_powers_not_sequence")
    elif len(powers) != 0:
        reasons.append("cultist_turn1_requires_no_player_powers")

    hand = _seq(state.get("hand"))
    draw = _seq(state.get("draw_pile"))
    discard = _seq(state.get("discard_pile"))
    exhaust = _seq(state.get("exhaust_pile"))
    if hand is None or draw is None or discard is None or exhaust is None:
        reasons.append("cultist_turn1_card_piles_not_sequences")
        return PublicCultistTurnOneAdmission(False, tuple(sorted(set(reasons))))

    if (len(hand), len(draw), len(discard), len(exhaust)) != (5, 0, 5, 0):
        reasons.append(
            "cultist_turn1_pristine_pile_boundary:"
            f"hand={len(hand)}:draw={len(draw)}:discard={len(discard)}:exhaust={len(exhaust)}"
        )

    ids: list[str] = []
    for index, card in enumerate([*hand, *draw, *discard, *exhaust]):
        if not isinstance(card, Mapping):
            reasons.append(f"cultist_turn1_card_not_mapping:{index}")
            continue
        card_id = _card_id(card.get("id"))
        ids.append(card_id)
        if card_id not in _EXPECTED_COSTS:
            reasons.append(f"cultist_turn1_nonstarter_card:{card_id or 'MISSING'}")
            continue
        upgrades = card.get("upgrades")
        if isinstance(upgrades, bool) or not isinstance(upgrades, int) or upgrades != 0:
            reasons.append(f"cultist_turn1_requires_unupgraded_starter:{index}:{upgrades!r}")
        cost = card.get("cost")
        if isinstance(cost, bool) or not isinstance(cost, int) or cost != _EXPECTED_COSTS[card_id]:
            reasons.append(f"cultist_turn1_card_cost_mismatch:{index}:{card_id}:{cost!r}")

    if Counter(ids) != _EXPECTED_STARTER_COUNTS:
        reasons.append("cultist_turn1_starter_composition_mismatch")

    return PublicCultistTurnOneAdmission(not reasons, tuple(sorted(set(reasons))))


__all__ = [
    "PublicCultistTurnOneAdmission",
    "assess_public_cultist_turn1_player",
    "is_cultist_turn1_candidate",
]
