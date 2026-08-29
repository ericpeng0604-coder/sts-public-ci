"""Fail-closed public reconstruction admission for one normal Anger play.

This module is intentionally separate from ``sts1-public-state-v1``.  Anger's
turn-local play counters come from the bounded CommunicationMod command trace
already used by the audited draw-card slices, while the generated-card effect
is proven from the *current public card piles*.

Supported slice only:
- Ironclad, Jaw Worm, zero-based turn 1 (second player turn)
- starter 10-card deck plus exactly one normal Anger before the play
- exactly one normal Anger has been played this turn
- Anger has generated exactly one normal Anger into discard
- no other card, discard, power, relic, potion, or hidden-history expansion

The auxiliary trace is reconstruction metadata only.  It is never merged into
policy state or the decision signature.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_AUX_SCHEMA = "sts1-public-reconstruction-aux-v1"
_AUX_SOURCE = "communicationmod_command_trace_v1"
_CARD_ID_ALIASES = {"STRIKE_R": "STRIKE_RED", "DEFEND_R": "DEFEND_RED"}
_EXPECTED_POST_ANGER_COUNTS = Counter(
    {"STRIKE_RED": 5, "DEFEND_RED": 4, "BASH": 1, "ANGER": 2}
)
_EXPECTED_COSTS = {"STRIKE_RED": 1, "DEFEND_RED": 1, "BASH": 2, "ANGER": 0}


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
class PublicAngerMidturnAdmission:
    allowed: bool
    reasons: tuple[str, ...]


def assess_public_anger_midturn(
    state: Mapping[str, Any],
    reconstruction_aux: Mapping[str, Any] | None,
) -> PublicAngerMidturnAdmission:
    """Admit only the exact one-Anger-play public reconstruction slice."""

    piles: dict[str, Sequence[Any]] = {}
    all_cards: list[Mapping[str, Any]] = []
    for pile_name in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        pile = _sequence(state.get(pile_name))
        if pile is None:
            return PublicAngerMidturnAdmission(False, (f"anger_{pile_name}_not_sequence",))
        piles[pile_name] = pile
        for card in pile:
            if isinstance(card, Mapping):
                all_cards.append(card)

    if not any(_card_id(card.get("id")) == "ANGER" for card in all_cards):
        return PublicAngerMidturnAdmission(False, ("anger_card_missing",))

    reasons: list[str] = []

    if state.get("turn") != 1 or isinstance(state.get("turn"), bool):
        reasons.append(f"anger_turn_mismatch:{state.get('turn')!r}")

    enemies = _sequence(state.get("enemies"))
    if enemies is None or len(enemies) != 1 or not isinstance(enemies[0], Mapping):
        reasons.append("anger_requires_single_enemy")
    elif _norm(enemies[0].get("name")) != "JAW_WORM":
        reasons.append("anger_requires_jaw_worm")

    if len(piles["hand"]) != 4 or len(piles["draw_pile"]) != 1:
        reasons.append(
            f"anger_pile_shape_hand_draw:{len(piles['hand'])}:{len(piles['draw_pile'])}"
        )
    if len(piles["discard_pile"]) != 7 or len(piles["exhaust_pile"]) != 0:
        reasons.append(
            f"anger_pile_shape_discard_exhaust:{len(piles['discard_pile'])}:{len(piles['exhaust_pile'])}"
        )

    ids: list[str] = []
    for index, card in enumerate(all_cards):
        card_id = _card_id(card.get("id"))
        ids.append(card_id)
        if card_id not in _EXPECTED_COSTS:
            reasons.append(f"anger_unsupported_card:{index}:{card_id or 'MISSING'}")
            continue
        upgrades = card.get("upgrades")
        if isinstance(upgrades, bool) or not isinstance(upgrades, int) or upgrades != 0:
            reasons.append(f"anger_requires_unupgraded_cards:{index}:{upgrades!r}")
        cost = card.get("cost")
        if isinstance(cost, bool) or not isinstance(cost, int) or cost != _EXPECTED_COSTS[card_id]:
            reasons.append(f"anger_cost_mismatch:{index}:{card_id}:{cost!r}")

    if Counter(ids) != _EXPECTED_POST_ANGER_COUNTS:
        reasons.append("anger_post_generation_composition_mismatch")

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
            f"anger_generated_copy_not_publicly_proven:discard={discard_angers}:other={other_pile_angers}"
        )

    if state.get("energy") != 3 or isinstance(state.get("energy"), bool):
        reasons.append(f"anger_energy_mismatch:{state.get('energy')!r}")
    if state.get("block") != 0 or isinstance(state.get("block"), bool):
        reasons.append(f"anger_block_mismatch:{state.get('block')!r}")

    powers = _sequence(state.get("powers"))
    if powers is None:
        reasons.append("anger_powers_not_sequence")
    elif len(powers) != 0:
        reasons.append("anger_requires_no_player_powers")

    if reconstruction_aux is None:
        reasons.append("anger_aux_missing")
    elif not isinstance(reconstruction_aux, Mapping):
        reasons.append("anger_aux_not_mapping")
    else:
        if reconstruction_aux.get("schema_version") != _AUX_SCHEMA:
            reasons.append("anger_aux_schema_mismatch")
        if reconstruction_aux.get("source") != _AUX_SOURCE:
            reasons.append("anger_aux_source_mismatch")
        if reconstruction_aux.get("complete") is not True:
            reasons.append("anger_aux_incomplete")
        expected_ints = {
            "turn": 1,
            "cards_played_this_turn": 1,
            "attacks_played_this_turn": 1,
            "skills_played_this_turn": 0,
            "cards_discarded_this_turn": 0,
        }
        for key, expected in expected_ints.items():
            value = reconstruction_aux.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                reasons.append(f"anger_aux_{key}_mismatch:{value!r}!={expected}")

    return PublicAngerMidturnAdmission(not reasons, tuple(sorted(set(reasons))))


__all__ = ["PublicAngerMidturnAdmission", "assess_public_anger_midturn"]
