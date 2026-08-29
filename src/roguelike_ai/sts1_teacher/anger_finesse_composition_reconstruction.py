"""Fail-closed admission for one Anger + one Finesse in the same turn.

This slice is deliberately tiny and additive. It observes only the frozen
``sts1-public-state-v1`` snapshot plus the existing bounded CommunicationMod
turn-counter auxiliary trace. Card identities/effects are proved from current
public piles; the auxiliary trace is reconstruction metadata only and never
enters policy state or the decision signature.

Supported slice only:
- Ironclad, Jaw Worm, zero-based turn 1 (second player turn)
- starter 10-card deck plus one normal Anger and one normal Finesse before the
  generated-card effect
- exactly two cards played this turn: one attack and one skill
- Anger generated exactly one extra normal Anger into discard
- Finesse produced exactly one draw and two block
- no upgrades, powers, potion effects, discards, or other rich-card composition
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_AUX_SCHEMA = "sts1-public-reconstruction-aux-v1"
_AUX_SOURCE = "communicationmod_command_trace_v1"
_CARD_ID_ALIASES = {"STRIKE_R": "STRIKE_RED", "DEFEND_R": "DEFEND_RED"}
_EXPECTED_COUNTS = Counter(
    {
        "STRIKE_RED": 5,
        "DEFEND_RED": 4,
        "BASH": 1,
        "FINESSE": 1,
        "ANGER": 2,
    }
)
_EXPECTED_COSTS = {
    "STRIKE_RED": 1,
    "DEFEND_RED": 1,
    "BASH": 2,
    "FINESSE": 0,
    "ANGER": 0,
}


def _norm(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_").replace("-", "_")


def _card_id(value: Any) -> str:
    normalized = _norm(value)
    return _CARD_ID_ALIASES.get(normalized, normalized)


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return None


@dataclass(frozen=True)
class PublicAngerFinesseCompositionAdmission:
    allowed: bool
    reasons: tuple[str, ...]


def assess_public_anger_finesse_composition(
    state: Mapping[str, Any],
    reconstruction_aux: Mapping[str, Any] | None,
) -> PublicAngerFinesseCompositionAdmission:
    """Admit only the exact one-attack/one-skill Anger + Finesse slice."""

    piles: dict[str, Sequence[Any]] = {}
    all_cards: list[Mapping[str, Any]] = []
    for pile_name in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        pile = _sequence(state.get(pile_name))
        if pile is None:
            return PublicAngerFinesseCompositionAdmission(
                False,
                (f"anger_finesse_{pile_name}_not_sequence",),
            )
        piles[pile_name] = pile
        for card in pile:
            if isinstance(card, Mapping):
                all_cards.append(card)

    ids_present = {_card_id(card.get("id")) for card in all_cards}
    if "ANGER" not in ids_present or "FINESSE" not in ids_present:
        return PublicAngerFinesseCompositionAdmission(
            False,
            ("anger_finesse_composition_not_present",),
        )

    reasons: list[str] = []
    if state.get("turn") != 1 or isinstance(state.get("turn"), bool):
        reasons.append(f"anger_finesse_turn_mismatch:{state.get('turn')!r}")

    enemies = _sequence(state.get("enemies"))
    if enemies is None or len(enemies) != 1 or not isinstance(enemies[0], Mapping):
        reasons.append("anger_finesse_requires_single_enemy")
    elif _norm(enemies[0].get("name")) != "JAW_WORM":
        reasons.append("anger_finesse_requires_jaw_worm")

    # A 12-card pre-generation combat deck starts turn 1 at hand=5/draw=2.
    # Playing Anger and Finesse removes two cards, Finesse draws one, and the
    # generated Anger adds one public discard card: hand=4/draw=1/discard=8.
    if len(piles["hand"]) != 4 or len(piles["draw_pile"]) != 1:
        reasons.append(
            "anger_finesse_pile_shape_hand_draw:"
            f"{len(piles['hand'])}:{len(piles['draw_pile'])}"
        )
    if len(piles["discard_pile"]) != 8 or len(piles["exhaust_pile"]) != 0:
        reasons.append(
            "anger_finesse_pile_shape_discard_exhaust:"
            f"{len(piles['discard_pile'])}:{len(piles['exhaust_pile'])}"
        )

    ids: list[str] = []
    for index, card in enumerate(all_cards):
        card_id = _card_id(card.get("id"))
        ids.append(card_id)
        if card_id not in _EXPECTED_COSTS:
            reasons.append(f"anger_finesse_unsupported_card:{index}:{card_id or 'MISSING'}")
            continue
        upgrades = card.get("upgrades")
        if isinstance(upgrades, bool) or not isinstance(upgrades, int) or upgrades != 0:
            reasons.append(f"anger_finesse_requires_unupgraded_cards:{index}:{upgrades!r}")
        cost = card.get("cost")
        if isinstance(cost, bool) or not isinstance(cost, int) or cost != _EXPECTED_COSTS[card_id]:
            reasons.append(f"anger_finesse_cost_mismatch:{index}:{card_id}:{cost!r}")

    if Counter(ids) != _EXPECTED_COUNTS:
        reasons.append("anger_finesse_post_effect_composition_mismatch")

    discard_finesses = sum(
        1
        for card in piles["discard_pile"]
        if isinstance(card, Mapping) and _card_id(card.get("id")) == "FINESSE"
    )
    if discard_finesses != 1:
        reasons.append(f"anger_finesse_finesse_not_publicly_proven:{discard_finesses}")

    discard_angers = sum(
        1
        for card in piles["discard_pile"]
        if isinstance(card, Mapping) and _card_id(card.get("id")) == "ANGER"
    )
    other_pile_angers = sum(
        1
        for pile_name in ("hand", "draw_pile", "exhaust_pile")
        for card in piles[pile_name]
        if isinstance(card, Mapping) and _card_id(card.get("id")) == "ANGER"
    )
    if discard_angers != 2 or other_pile_angers != 0:
        reasons.append(
            "anger_finesse_generated_copy_not_publicly_proven:"
            f"discard={discard_angers}:other={other_pile_angers}"
        )

    if state.get("energy") != 3 or isinstance(state.get("energy"), bool):
        reasons.append(f"anger_finesse_energy_mismatch:{state.get('energy')!r}")
    if state.get("block") != 2 or isinstance(state.get("block"), bool):
        reasons.append(f"anger_finesse_block_mismatch:{state.get('block')!r}")

    powers = _sequence(state.get("powers"))
    if powers is None:
        reasons.append("anger_finesse_powers_not_sequence")
    elif len(powers) != 0:
        reasons.append("anger_finesse_requires_no_player_powers")

    if reconstruction_aux is None:
        reasons.append("anger_finesse_aux_missing")
    elif not isinstance(reconstruction_aux, Mapping):
        reasons.append("anger_finesse_aux_not_mapping")
    else:
        if reconstruction_aux.get("schema_version") != _AUX_SCHEMA:
            reasons.append("anger_finesse_aux_schema_mismatch")
        if reconstruction_aux.get("source") != _AUX_SOURCE:
            reasons.append("anger_finesse_aux_source_mismatch")
        if reconstruction_aux.get("complete") is not True:
            reasons.append("anger_finesse_aux_incomplete")
        expected_ints = {
            "turn": 1,
            "cards_played_this_turn": 2,
            "attacks_played_this_turn": 1,
            "skills_played_this_turn": 1,
            "cards_discarded_this_turn": 0,
        }
        for key, expected in expected_ints.items():
            value = reconstruction_aux.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                reasons.append(f"anger_finesse_aux_{key}_mismatch:{value!r}!={expected}")

    return PublicAngerFinesseCompositionAdmission(
        not reasons,
        tuple(sorted(set(reasons))),
    )


__all__ = [
    "PublicAngerFinesseCompositionAdmission",
    "assess_public_anger_finesse_composition",
]
